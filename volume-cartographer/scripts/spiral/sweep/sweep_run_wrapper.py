"""Per-run entry point for wandb sweeps over fit_spiral.py.

Executed by ``wandb agent`` (see sweep_fit_spiral.py) once per sweep run with
the sweep-assigned parameters passed as a single JSON argv element
(``--params-json '${args_json}'``). The wrapper:

1. merges the sweep params onto an optional base config, coerces them to the
   exact types Config validation demands (wandb hands back ``4.0`` for
   integers and ``0/1`` for booleans), and fail-fast validates with Config;
2. writes the fully-resolved config to ``<out>/config.json`` (and the raw
   sweep params to ``<out>/sweep_params.json``) before any GPU work;
3. reuses matching completed seed results from any ``--reuse-root`` and runs
   only the missing seeds, each fit via torchrun across every GPU in
   CUDA_VISIBLE_DEVICES with ``optimizer_random_seed`` overridden and its own
   ``<out>/seed_<s>/`` dir; seed fits always run with wandb disabled;
4. runs the eval chain per seed — render_ink.py (lasagna flatten + ink
   render) then get_ink_metrics.py (nnU-Net ink scoring);
5. writes an auditable aggregate manifest, then logs only across-seed mean/std
   metrics (``ink/*``, ``final/*``) to the agent-created sweep run.  Thus each
   parameter configuration is exactly one online wandb run.

Standalone smoke test (no agent, no sweep):

    WANDB_MODE=disabled WANDB_RUN_ID=smoke1 CUDA_VISIBLE_DEVICES=0 \
        python sweep_run_wrapper.py --dataset <dataset> --skip-eval --seeds 1 \
        --params-json '{"optimizer_num_training_steps": 200}'
"""

import argparse
import glob
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

SPIRAL_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SPIRAL_DIR))

from config import Config

DEFAULT_INK_PYTHON = Path.home() / 'villa/vesuvius/.venv/bin/python'

INK_SUMMARY_KEYS = (
    'total_fg_pixels',
    'total_pixels',
    'overall_fg_fraction',
    'overall_line_score',
    'overall_column_score',
)

# Each torchrun rank owns one GPU.  Keep its host-side Torch pools bounded so
# concurrent sweep fits do not each consume a full machine's CPU threads.
SWEEP_THREADS_PER_GPU = 4
SEED_RESULT_SCHEMA_VERSION = 1

# A seed process must not inherit any identity that could attach it to the
# agent-created aggregate run or create a derived online run.
SEED_WANDB_IDENTITY_VARS = (
    'WANDB_RUN_ID',
    'WANDB_NAME',
    'WANDB_RUN_GROUP',
    'WANDB_RESUME',
    'WANDB_SWEEP_ID',
    'WANDB_SWEEP_PARAM_PATH',
)

RUN_NAME_LABELS = {
    'patch_uuid_filter_regex': 'patch',
    'patch_2d_sampling_max_area': 'cap',
    'patch_strip_sampling': 'strip',
    'loss_weight_dense_spacing': 'dense',
    'patch_sampling_area_exponent': 'areaexp',
    'pcl_stratified_pcl_sampling': 'strat',
    'input_disable_tracks': 'tracks',
    'input_disable_fibers': 'fibers',
    'pcl_use_fiber_links': 'links',
}
MAX_RUN_NAME_LENGTH = 128


def _short_run_name_value(key, value):
    if key in ('input_disable_tracks', 'input_disable_fibers'):
        return 'off' if value else 'on'
    if isinstance(value, bool):
        return 'on' if value else 'off'
    if key == 'patch_uuid_filter_regex':
        if value is None:
            return 'all'
        if value == '^(?!.*band-seed).*$':
            return 'no-band-seed'
    if key == 'patch_2d_sampling_max_area' and value is None:
        return 'unlimited'
    if value is None:
        return 'none'
    if isinstance(value, float):
        return f'{value:g}'
    if isinstance(value, (dict, list)):
        value = json.dumps(value, sort_keys=True, separators=(',', ':'))
    text = re.sub(r'[^A-Za-z0-9_.+-]+', '-', str(value)).strip('-') or 'empty'
    if len(text) > 24:
        digest = hashlib.sha256(str(value).encode()).hexdigest()[:6]
        text = f'{text[:17]}~{digest}'
    return text


def format_run_name(keys, resolved_config):
    """Readable deterministic display name from the dimensions that vary."""
    parts = []
    for key in keys:
        if key not in resolved_config:
            continue
        label = RUN_NAME_LABELS.get(key, key)
        parts.append(f'{label}={_short_run_name_value(key, resolved_config[key])}')
    if not parts:
        return None
    full_name = ','.join(parts)
    if len(full_name) <= MAX_RUN_NAME_LENGTH:
        return full_name
    digest = hashlib.sha256(full_name.encode()).hexdigest()[:8]
    return f'{full_name[:MAX_RUN_NAME_LENGTH - len(digest) - 1]}~{digest}'


def write_json_atomic(path, value):
    """Write JSON without leaving a valid-looking partial result on failure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError, TypeError):
        return None


def numeric_metrics(values):
    """Keep JSON numeric scalars that can safely participate in aggregation."""
    if not isinstance(values, dict):
        return {}
    return {key: value for key, value in values.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value)}


def seed_fit_environment(base_env, seed_overrides, seed_out):
    """Environment for a local-only seed fit with no online wandb identity."""
    env = dict(base_env)
    env['FIT_SPIRAL_CONFIG_OVERRIDES'] = json.dumps(seed_overrides)
    env['FIT_SPIRAL_OUT_DIR'] = str(seed_out)
    env['FIT_SPIRAL_NUM_THREADS'] = str(SWEEP_THREADS_PER_GPU)
    for var in SEED_WANDB_IDENTITY_VARS:
        env.pop(var, None)
    env['WANDB_MODE'] = 'disabled'
    return env


def _candidate_from_result(path, seed, seed_config, require_ink):
    data = read_json(path)
    if (not isinstance(data, dict) or data.get('seed') != seed
            or data.get('config') != seed_config):
        return None
    final = numeric_metrics(data.get('final'))
    ink = numeric_metrics(data.get('ink'))
    if not final or (require_ink and not ink):
        return None
    return {
        'seed': seed,
        'config': seed_config,
        'final': final,
        'ink': ink,
        'completed_ns': Path(path).stat().st_mtime_ns,
        'source': str(Path(path).resolve()),
        'source_kind': 'result_manifest',
    }


def _legacy_candidates(run_dir, seed, seed_config, require_ink):
    """Read results produced before seed result manifests were introduced."""
    seed_dir = Path(run_dir) / f'seed_{seed}'
    if read_json(seed_dir / 'config.json') != seed_config:
        return []
    candidates = []
    if require_ink:
        paths = seed_dir.glob('*/meshes/fitted*/ink_metric/metrics.json')
        for metrics_path in paths:
            fit_dir = metrics_path.parents[3]
            satisfaction_path = fit_dir / 'satisfaction_summary.json'
            metrics = read_json(metrics_path)
            final = numeric_metrics(read_json(satisfaction_path))
            ink = numeric_metrics(metrics.get('summary') if isinstance(metrics, dict)
                                  else None)
            ink = {key: ink[key] for key in INK_SUMMARY_KEYS if key in ink}
            if not final or not ink:
                continue
            candidates.append({
                'seed': seed,
                'config': seed_config,
                'final': final,
                'ink': ink,
                'completed_ns': metrics_path.stat().st_mtime_ns,
                'source': str(metrics_path.resolve()),
                'source_kind': 'legacy_metrics',
            })
    else:
        for satisfaction_path in seed_dir.glob('*/satisfaction_summary.json'):
            final = numeric_metrics(read_json(satisfaction_path))
            if not final:
                continue
            candidates.append({
                'seed': seed,
                'config': seed_config,
                'final': final,
                'ink': {},
                'completed_ns': satisfaction_path.stat().st_mtime_ns,
                'source': str(satisfaction_path.resolve()),
                'source_kind': 'legacy_satisfaction',
            })
    return candidates


def reusable_seed_candidates(reuse_roots, resolved_config, seed, seed_config,
                             require_ink):
    """Return every valid completed result for one exact config and seed."""
    candidates = []
    for root in map(Path, reuse_roots):
        for run_dir in sorted(root.iterdir()):
            if not run_dir.is_dir() or read_json(run_dir / 'config.json') != resolved_config:
                continue
            result_path = run_dir / f'seed_{seed}' / 'result.json'
            result = _candidate_from_result(result_path, seed, seed_config, require_ink)
            if result is not None:
                candidates.append(result)
            else:
                candidates.extend(_legacy_candidates(
                    run_dir, seed, seed_config, require_ink))
    return candidates


def select_reusable_seed(reuse_roots, resolved_config, seed, seed_config,
                         require_ink):
    """Choose the newest valid result, with path as a deterministic tie-break."""
    candidates = reusable_seed_candidates(
        reuse_roots, resolved_config, seed, seed_config, require_ink)
    if not candidates:
        return None
    candidates.sort(key=lambda value: (value['completed_ns'], value['source']))
    selected = dict(candidates[-1])
    selected['candidate_sources'] = [candidate['source'] for candidate in candidates]
    return selected


def make_seed_result(seed, seed_config, final, ink, provenance):
    return {
        'schema_version': SEED_RESULT_SCHEMA_VERSION,
        'seed': seed,
        'config': seed_config,
        'final': numeric_metrics(final),
        'ink': numeric_metrics(ink),
        'provenance': provenance,
        'completed_unix_ns': time.time_ns(),
    }


def coerce_params(params):
    """Coerce sweep-assigned values to the exact types Config validation demands.

    A dict value under a non-Config key is a *group* parameter (one composite
    sweep value that sets several Config keys at once — the conditional-
    parameter workaround, see sweep_fit_spiral.py); its entries are flattened
    into the result.
    """
    fields = Config.catalog()['schema']['fields']
    flat = {}
    for key, value in params.items():
        entries = (value.items() if key not in fields and isinstance(value, dict)
                   else [(key, value)])
        for inner_key, inner_value in entries:
            if inner_key in flat:
                raise SystemExit(f'config key {inner_key!r} assigned more than '
                                 'once across sweep params/groups')
            flat[inner_key] = inner_value
    unknown = sorted(set(flat) - set(fields))
    if unknown:
        raise SystemExit(f'unknown Config keys in sweep params: {unknown}')
    coerced = {}
    for key, value in flat.items():
        spec = fields[key]
        if value is None and spec['nullable']:
            coerced[key] = None
            continue
        kind = spec['type']
        if kind == 'boolean' and type(value) is not bool:
            if value in (0, 1, 0.0, 1.0):
                value = bool(value)
            elif isinstance(value, str) and value.lower() in ('true', 'false'):
                value = value.lower() == 'true'
        elif kind == 'integer' and type(value) is float and value.is_integer():
            value = int(value)
        coerced[key] = value
    return coerced


def run_stage(name, cmd, env=None):
    """Run one pipeline stage, forwarding SIGINT/SIGTERM (agent stop) to it."""
    print(f'[sweep_run_wrapper] {name}: {" ".join(map(str, cmd))}', flush=True)
    child = subprocess.Popen(cmd, env=env)

    def forward(signum, _frame):
        child.send_signal(signum)

    previous = {s: signal.signal(s, forward) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        returncode = child.wait()
    finally:
        for s, handler in previous.items():
            signal.signal(s, handler)
    if returncode != 0:
        raise SystemExit(f'{name} failed with exit code {returncode}')


def resolve_run_dir(seed_out):
    """The fit names its run dir <seed_out>/<date>_<scroll>_slice-...; seed_out
    is normally unique per fit.  Interrupted/retried fits can leave multiple
    directories, so use the most recently updated one."""
    run_dirs = [p for p in Path(seed_out).iterdir() if (p / 'meshes').is_dir()]
    if not run_dirs:
        raise SystemExit(f'expected a fit run dir with meshes/ under {seed_out}')
    return max(run_dirs, key=lambda path: (path.stat().st_mtime_ns, str(path)))


def resolve_meshes_dir(run_dir):
    mesh_dirs = sorted((run_dir / 'meshes').glob('fitted*'))
    if len(mesh_dirs) != 1:
        raise SystemExit(f'expected exactly one meshes/fitted* dir in '
                         f'{run_dir}, found {[str(p) for p in mesh_dirs]}')
    return mesh_dirs[0]


def save_ink_preview(ink_dir, out_path, factor):
    """One downsampled image of the full lasagna-flattened ink render.

    The render is stored as horizontally-chopped ``*_flat[.NNN].jpg`` tiles;
    each tile is downsampled by ``factor`` and they are re-concatenated in
    name order (the .NNN suffixes are zero-padded).
    """
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None  # the tiles are large but trusted
    tiles = sorted(Path(ink_dir).glob('*_flat*.jpg'))
    scaled = []
    for tile_path in tiles:
        with Image.open(tile_path) as tile:
            scaled.append(tile.resize((max(1, tile.width // factor),
                                       max(1, tile.height // factor)),
                                      Image.LANCZOS))
    preview = Image.new('L', (sum(t.width for t in scaled),
                              max(t.height for t in scaled)))
    x = 0
    for tile in scaled:
        preview.paste(tile.convert('L'), (x, 0))
        x += tile.width
    preview.save(out_path, quality=90)
    return out_path


def wandb_available():
    return (os.environ.get('WANDB_RUN_ID')
            and os.environ.get('WANDB_MODE', 'online') != 'disabled')


def log_to_run(payload, files=(), run_name=None):
    """Attach once to the agent-created run and log aggregate results.

    Project/entity/run id all come from the agent's environment; the last
    logged values populate run.summary, which the sweep optimizer reads.
    """
    if not wandb_available():
        print(f'[sweep_run_wrapper] wandb disabled; payload: {payload}', flush=True)
        return
    import wandb
    os.environ.setdefault('WANDB_RESUME', 'allow')
    run = wandb.init(name=run_name)
    run.log(payload)
    for path in files:
        run.save(str(path), base_path=str(Path(path).parent), policy='now')
    run.finish()


def combined_objective(objective, seed, per_seed_final, per_seed_ink):
    """The weighted sum of 'final/<k>' / 'ink/<k>' metrics for one seed."""
    sources = {'final': per_seed_final, 'ink': per_seed_ink}
    total = 0.0
    for name, weight in objective.items():
        prefix, _, key = name.partition('/')
        value = sources.get(prefix, {}).get(seed, {}).get(key)
        if value is None:
            raise SystemExit(f'objective term {name!r} is missing for seed {seed}; '
                             f'available: final/{sorted(per_seed_final.get(seed, {}))}, '
                             f'ink/{sorted(per_seed_ink.get(seed, {}))}')
        total += weight * value
    return total


def aggregate_across_seeds(per_seed, prefix):
    """Mean (and, with >1 seed, sample std) for common numeric keys.

    The mean carries the plain ``<prefix>/<key>`` name so it is what the
    sweep's objective metric resolves to. Keys missing for any seed are not
    aggregated and per-seed values stay solely in aggregate_results.json.
    """
    payload = {}
    if not per_seed:
        return payload
    common = set.intersection(*(set(v) for v in per_seed.values()))
    for key in sorted(common):
        values = [per_seed[seed][key] for seed in per_seed]
        mean = sum(values) / len(values)
        payload[f'{prefix}/{key}'] = mean
        if len(values) > 1:
            payload[f'{prefix}/{key}_std'] = (
                sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5
    return payload


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset', required=True, help='dataset root passed to fit_spiral.py')
    ap.add_argument('--scroll-spec', default=None, help='passed through to fit_spiral.py')
    ap.add_argument('--cache', default=None, help='passed through to fit_spiral.py')
    ap.add_argument('--params-json', required=True,
                    help="sweep params as one JSON dict (the agent's ${args_json})")
    ap.add_argument('--base-config', default=None,
                    help='JSON of Config overrides applied under the sweep params')
    ap.add_argument('--out-root', default=os.environ.get('FIT_SPIRAL_OUT_DIR', 'sweep_out'),
                    help='per-run output goes to <out-root>/<sweep_id>/<run_id>')
    ap.add_argument('--reuse-root', action='append', default=[],
                    help='old sweep output directory searched for exact matching '
                         'completed seeds; repeat for multiple sources')
    ap.add_argument('--run-name-keys-json', default='[]',
                    help='internal JSON list of varying Config keys included in '
                         'the aggregate wandb run display name')
    ap.add_argument('--seeds', default='1,2,3',
                    help='comma list of optimizer_random_seed values; the full '
                         'fit+eval chain runs once per seed and the sweep '
                         'objective is the across-seed mean')
    ap.add_argument('--objective-json', default=None,
                    help='JSON dict {metric: weight} defining the combined '
                         'sweep objective, e.g. \'{"ink/overall_column_score": 1.0, '
                         '"final/satisfied_patch_ratio": 1.0}\'; the weighted sum '
                         'is computed per seed and logged as "objective" '
                         '(mean across seeds) plus "objective_std"')
    ap.add_argument('--ink-volume', default=None, help='ink zarr for render_ink.py')
    ap.add_argument('--vc-render-bin', default=None,
                    help='vc_render_tifxyz binary (sibling vc_tifxyz_trim is auto-found)')
    ap.add_argument('--ink-python', default=str(DEFAULT_INK_PYTHON),
                    help='interpreter with nnunetv2/huggingface_hub for get_ink_metrics.py')
    ap.add_argument('--render-procs', type=int, default=6, help='render_ink.py -j')
    ap.add_argument('--flatboi-threads', type=int, default=8)
    ap.add_argument('--tta', action='store_true',
                    help='enable mirroring TTA in get_ink_metrics.py (off by default: ~faster)')
    ap.add_argument('--ink-preview-downsample', type=int, default=8,
                    help='downsample factor for the flattened-ink-render preview '
                         'image saved locally per computed seed (0 disables)')
    ap.add_argument('--skip-eval', action='store_true',
                    help='stop after the fits (no flatten/ink metrics)')
    args = ap.parse_args()

    if not args.skip_eval and not args.ink_volume:
        ap.error('--ink-volume is required unless --skip-eval is given')
    reuse_roots = [Path(path).resolve() for path in args.reuse_root]
    invalid_reuse_roots = [str(path) for path in reuse_roots if not path.is_dir()]
    if invalid_reuse_roots:
        ap.error(f'--reuse-root must name existing directories: '
                 f'{invalid_reuse_roots}')
    seeds = [int(t) for t in args.seeds.split(',') if t.strip()]
    if not seeds or len(set(seeds)) != len(seeds):
        ap.error(f'--seeds must be distinct integers, got {args.seeds!r}')

    params = coerce_params(json.loads(args.params_json))
    base = json.loads(Path(args.base_config).read_text()) if args.base_config else {}
    overrides = base | params
    if 'optimizer_random_seed' in overrides and len(seeds) > 1:
        raise SystemExit('optimizer_random_seed cannot be a sweep param or base-config '
                         'override: the wrapper assigns it per seed (--seeds)')
    resolved_config = Config(overrides).as_dict()  # fail fast before any GPU work
    run_name_keys = json.loads(args.run_name_keys_json)
    if (not isinstance(run_name_keys, list)
            or any(not isinstance(key, str) for key in run_name_keys)):
        ap.error('--run-name-keys-json must be a JSON list of strings')
    run_name = format_run_name(run_name_keys, resolved_config)

    run_id = os.environ.get('WANDB_RUN_ID') or f'local-{uuid.uuid4().hex[:8]}'
    sweep_id = os.environ.get('WANDB_SWEEP_ID', 'nosweep')
    out_base = Path(args.out_root).resolve() / sweep_id / run_id
    out_base.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out_base / 'config.json', resolved_config)
    write_json_atomic(out_base / 'sweep_params.json', {
        'params': params,
        'seeds': seeds,
        'reuse_roots': [str(path) for path in reuse_roots],
    })

    per_seed_final = {}
    per_seed_ink = {}
    seed_provenance = {}
    for seed in seeds:
        seed_overrides = overrides | {'optimizer_random_seed': seed}
        seed_config = Config(seed_overrides).as_dict()
        seed_out = out_base / f'seed_{seed}'
        seed_out.mkdir(exist_ok=True)
        write_json_atomic(seed_out / 'config.json', seed_config)

        reused = select_reusable_seed(
            reuse_roots, resolved_config, seed, seed_config,
            require_ink=not args.skip_eval)
        if reused is not None:
            per_seed_final[seed] = reused['final']
            per_seed_ink[seed] = reused['ink']
            seed_provenance[seed] = {
                'kind': 'reused',
                'selected_source': reused['source'],
                'source_kind': reused['source_kind'],
                'candidate_sources': reused['candidate_sources'],
            }
            write_json_atomic(seed_out / 'result.json', make_seed_result(
                seed, seed_config, reused['final'], reused['ink'],
                seed_provenance[seed]))
            print(f'[sweep_run_wrapper] reuse hit seed={seed}: '
                  f'{reused["source"]} ({len(reused["candidate_sources"])} '
                  f'candidate(s))', flush=True)
            continue

        print(f'[sweep_run_wrapper] reuse miss seed={seed}; running locally',
              flush=True)
        fit_env = seed_fit_environment(
            os.environ, seed_overrides, seed_out)

        nproc = len([t for t in fit_env.get('CUDA_VISIBLE_DEVICES', '0').split(',')
                     if t.strip()])
        fit_cmd = [sys.executable]
        if nproc > 1:
            fit_cmd += ['-m', 'torch.distributed.run', '--standalone',
                        f'--nproc-per-node={nproc}']
        fit_cmd += [str(SPIRAL_DIR / 'fit_spiral.py'), '--dataset', args.dataset]
        if args.scroll_spec:
            fit_cmd += ['--scroll-spec', args.scroll_spec]
        if args.cache:
            fit_cmd += ['--cache', args.cache]
        run_stage(f'fit[seed={seed}]', fit_cmd, env=fit_env)

        run_dir = resolve_run_dir(seed_out)
        satisfaction_path = run_dir / 'satisfaction_summary.json'
        satisfaction = numeric_metrics(read_json(satisfaction_path))
        if not satisfaction:
            raise SystemExit(f'fit produced no valid satisfaction summary at '
                             f'{satisfaction_path}')
        per_seed_final[seed] = satisfaction

        if args.skip_eval:
            per_seed_ink[seed] = {}
            seed_provenance[seed] = {
                'kind': 'computed',
                'fit_run_dir': str(run_dir.resolve()),
            }
            write_json_atomic(seed_out / 'result.json', make_seed_result(
                seed, seed_config, satisfaction, {}, seed_provenance[seed]))
            continue

        meshes_dir = resolve_meshes_dir(run_dir)
        render_cmd = [sys.executable, str(SPIRAL_DIR / 'render_ink.py'), str(meshes_dir),
                      '--volume', args.ink_volume,
                      '-j', str(args.render_procs),
                      '--flatboi-threads', str(args.flatboi_threads)]
        if args.vc_render_bin:
            render_cmd += ['--vc-render-bin', args.vc_render_bin]
        run_stage(f'render_ink[seed={seed}]', render_cmd)

        # render_ink exits 0 even when the lasagna flatten fails (it warns and
        # skips the render), so the absence of flat strips is the failure signal.
        ink_dir = meshes_dir / 'ink'
        if not glob.glob(str(ink_dir / '*_flat*.jpg')):
            log_to_run({'eval/flatten_failed': 1, 'eval/flatten_failed_seed': seed},
                       run_name=run_name)
            raise SystemExit(f'lasagna flatten produced no ink strips in {ink_dir}')

        if args.ink_preview_downsample > 0:
            save_ink_preview(
                ink_dir,
                out_base / f'seed_{seed}_ink_flat_{args.ink_preview_downsample}x.jpg',
                args.ink_preview_downsample)

        ink_cmd = [args.ink_python, str(SPIRAL_DIR / 'get_ink_metrics.py'), str(ink_dir)]
        if not args.tta:
            ink_cmd.append('--no-tta')
        run_stage(f'get_ink_metrics[seed={seed}]', ink_cmd)

        metrics_path = meshes_dir / 'ink_metric' / 'metrics.json'
        ink_summary = json.loads(metrics_path.read_text())['summary']
        per_seed_ink[seed] = {k: ink_summary[k] for k in INK_SUMMARY_KEYS
                              if k in ink_summary}
        seed_provenance[seed] = {
            'kind': 'computed',
            'fit_run_dir': str(run_dir.resolve()),
            'metrics_path': str(metrics_path.resolve()),
        }
        write_json_atomic(seed_out / 'result.json', make_seed_result(
            seed, seed_config, satisfaction, per_seed_ink[seed],
            seed_provenance[seed]))

    payload = aggregate_across_seeds(per_seed_final, 'final')
    payload.update(aggregate_across_seeds(per_seed_ink, 'ink'))
    if args.objective_json:
        objective = json.loads(args.objective_json)
        per_seed_obj = {s: combined_objective(objective, s, per_seed_final,
                                              per_seed_ink) for s in seeds}
        mean = sum(per_seed_obj.values()) / len(per_seed_obj)
        payload['objective'] = mean
        if len(per_seed_obj) > 1:
            payload['objective_std'] = (
                sum((v - mean) ** 2 for v in per_seed_obj.values())
                / (len(per_seed_obj) - 1)) ** 0.5
    if not args.skip_eval:
        payload['eval/flatten_failed'] = 0
    aggregate_path = out_base / 'aggregate_results.json'
    write_json_atomic(aggregate_path, {
        'schema_version': SEED_RESULT_SCHEMA_VERSION,
        'sweep_id': sweep_id,
        'run_id': run_id,
        'run_name': run_name,
        'config': resolved_config,
        'params': params,
        'seeds': seeds,
        'inputs': {
            'dataset': str(Path(args.dataset).resolve()),
            'scroll_spec': (str(Path(args.scroll_spec).resolve())
                            if args.scroll_spec else None),
            'ink_volume': (str(Path(args.ink_volume).resolve())
                           if args.ink_volume else None),
            'tta': args.tta,
            'skip_eval': args.skip_eval,
        },
        'per_seed': {
            str(seed): {
                'final': per_seed_final[seed],
                'ink': per_seed_ink[seed],
                'provenance': seed_provenance[seed],
            }
            for seed in seeds
        },
        'wandb_metrics': payload,
    })
    log_to_run(payload, files=[aggregate_path], run_name=run_name)
    print(f'[sweep_run_wrapper] done ({len(seeds)} seeds): {json.dumps(payload)}',
          flush=True)


if __name__ == '__main__':
    main()
