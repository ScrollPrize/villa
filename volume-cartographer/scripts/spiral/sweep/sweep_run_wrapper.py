"""Per-run entry point for wandb sweeps over fit_spiral.py.

Executed by ``wandb agent`` (see sweep_fit_spiral.py) once per sweep run with
the sweep-assigned parameters passed as a single JSON argv element
(``--params-json '${args_json}'``). The wrapper:

1. merges the sweep params onto an optional base config, coerces them to the
   exact types Config validation demands (wandb hands back ``4.0`` for
   integers and ``0/1`` for booleans), and fail-fast validates with Config;
2. writes the fully-resolved config to ``<out>/config.json`` (and the raw
   sweep params to ``<out>/sweep_params.json``) before any GPU work;
3. runs the fit once per seed in ``--seeds`` (default 3 seeds), each fit via
   torchrun across every GPU in CUDA_VISIBLE_DEVICES with
   ``optimizer_random_seed`` overridden and its own ``<out>/seed_<s>/`` dir;
   each seed fit logs to its OWN wandb run (id ``<sweep_run_id>-seed<s>``,
   grouped as ``<sweep_id>-<sweep_run_id>``) so per-iteration loss curves are
   not concatenated across seeds;
4. runs the eval chain per seed — render_ink.py (lasagna flatten + ink
   render) then get_ink_metrics.py (nnU-Net ink scoring);
5. logs per-seed metrics (``seed<s>/ink/*``, ``seed<s>/final/*``), their
   across-seed mean/std (``ink/*``, ``final/*``), and the seed run ids to the
   agent-created sweep run, whose summary — the mean — is what the sweep
   optimizer reads.

Standalone smoke test (no agent, no sweep):

    WANDB_MODE=disabled WANDB_RUN_ID=smoke1 CUDA_VISIBLE_DEVICES=0 \
        python sweep_run_wrapper.py --dataset <dataset> --skip-eval --seeds 1 \
        --params-json '{"optimizer_num_training_steps": 200}'
"""

import argparse
import glob
import json
import os
import signal
import subprocess
import sys
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
    is unique per fit, so there is exactly one."""
    run_dirs = [p for p in Path(seed_out).iterdir() if (p / 'meshes').is_dir()]
    if len(run_dirs) != 1:
        raise SystemExit(f'expected exactly one fit run dir with meshes/ under '
                         f'{seed_out}, found {[str(p) for p in run_dirs]}')
    return run_dirs[0]


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


def log_to_run(payload, files=(), images=None):
    """Re-attach to the run the fits already logged to and append the eval
    results and across-seed aggregates.

    Project/entity/run id all come from the agent's environment; the last
    logged values populate run.summary, which the sweep optimizer reads.
    ``images`` maps metric names to image paths, logged as wandb.Image.
    """
    if not wandb_available():
        print(f'[sweep_run_wrapper] wandb disabled; payload: {payload}, '
              f'images: {images}', flush=True)
        return
    import wandb
    os.environ.setdefault('WANDB_RESUME', 'allow')
    run = wandb.init()
    run.log(payload | {name: wandb.Image(str(path))
                       for name, path in (images or {}).items()})
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
    """Per-seed metrics plus mean (and, with >1 seed, sample std) per key.

    The mean carries the plain ``<prefix>/<key>`` name so it is what the
    sweep's objective metric resolves to. Keys missing for some seed are
    reported per-seed but not aggregated.
    """
    payload = {}
    for seed, values in per_seed.items():
        payload.update({f'seed{seed}/{prefix}/{k}': v for k, v in values.items()})
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
    ap.add_argument('--seeds', default='1,2,3',
                    help='comma list of optimizer_random_seed values; the full '
                         'fit+eval chain runs once per seed and the sweep '
                         'objective is the across-seed mean')
    ap.add_argument('--objective-json', default=None,
                    help='JSON dict {metric: weight} defining the combined '
                         'sweep objective, e.g. \'{"ink/overall_column_score": 1.0, '
                         '"final/satisfied_patch_ratio": 1.0}\'; the weighted sum '
                         'is computed per seed and logged as "objective" '
                         '(mean across seeds) and "seed<s>/objective"')
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
                         'image logged per seed (0 disables)')
    ap.add_argument('--skip-eval', action='store_true',
                    help='stop after the fits (no flatten/ink metrics)')
    args = ap.parse_args()

    if not args.skip_eval and not args.ink_volume:
        ap.error('--ink-volume is required unless --skip-eval is given')
    seeds = [int(t) for t in args.seeds.split(',') if t.strip()]
    if not seeds or len(set(seeds)) != len(seeds):
        ap.error(f'--seeds must be distinct integers, got {args.seeds!r}')

    params = coerce_params(json.loads(args.params_json))
    base = json.loads(Path(args.base_config).read_text()) if args.base_config else {}
    overrides = base | params
    if 'optimizer_random_seed' in overrides and len(seeds) > 1:
        raise SystemExit('optimizer_random_seed cannot be a sweep param or base-config '
                         'override: the wrapper assigns it per seed (--seeds)')
    Config(overrides)  # fail fast before any GPU work

    run_id = os.environ.get('WANDB_RUN_ID') or f'local-{uuid.uuid4().hex[:8]}'
    sweep_id = os.environ.get('WANDB_SWEEP_ID', 'nosweep')
    out_base = Path(args.out_root).resolve() / sweep_id / run_id
    out_base.mkdir(parents=True, exist_ok=True)
    (out_base / 'config.json').write_text(
        json.dumps(Config(overrides).as_dict(), indent=2) + '\n')
    (out_base / 'sweep_params.json').write_text(
        json.dumps({'params': params, 'seeds': seeds}, indent=2) + '\n')

    per_seed_final = {}
    per_seed_ink = {}
    metric_files = [out_base / 'config.json']
    preview_images = {}
    for seed in seeds:
        seed_overrides = overrides | {'optimizer_random_seed': seed}
        seed_out = out_base / f'seed_{seed}'
        seed_out.mkdir(exist_ok=True)
        (seed_out / 'config.json').write_text(
            json.dumps(Config(seed_overrides).as_dict(), indent=2) + '\n')

        fit_env = os.environ.copy()
        fit_env['FIT_SPIRAL_CONFIG_OVERRIDES'] = json.dumps(seed_overrides)
        fit_env['FIT_SPIRAL_OUT_DIR'] = str(seed_out)
        fit_env['FIT_SPIRAL_NUM_THREADS'] = str(SWEEP_THREADS_PER_GPU)
        fit_env.setdefault('WANDB_MODE', 'online')
        # Each seed fit gets its OWN wandb run (so per-iteration loss curves
        # are not concatenated across seeds): detach the fit from the
        # agent-created sweep run and give it a deterministic derived run id.
        # The seed runs share a group so they sit together in the UI; only the
        # sweep run (WANDB_RUN_ID, logged to below) carries the aggregates the
        # sweep optimizer reads.
        for var in ('WANDB_SWEEP_ID', 'WANDB_SWEEP_PARAM_PATH'):
            fit_env.pop(var, None)
        fit_env['WANDB_RUN_ID'] = f'{run_id}-seed{seed}'
        fit_env['WANDB_NAME'] = f'{run_id}-seed{seed}'
        fit_env['WANDB_RUN_GROUP'] = f'{sweep_id}-{run_id}'
        fit_env['WANDB_RESUME'] = 'allow'

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
        if satisfaction_path.exists():
            per_seed_final[seed] = json.loads(satisfaction_path.read_text())

        if args.skip_eval:
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
            log_to_run({'eval/flatten_failed': 1, 'eval/flatten_failed_seed': seed})
            raise SystemExit(f'lasagna flatten produced no ink strips in {ink_dir}')

        if args.ink_preview_downsample > 0:
            preview_images[f'seed{seed}/ink_render'] = save_ink_preview(
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
        metric_files.append(metrics_path)

    payload = aggregate_across_seeds(per_seed_final, 'final')
    payload.update(aggregate_across_seeds(per_seed_ink, 'ink'))
    payload.update({f'seed{s}/wandb_run_id': f'{run_id}-seed{s}' for s in seeds})
    if args.objective_json:
        objective = json.loads(args.objective_json)
        per_seed_obj = {s: combined_objective(objective, s, per_seed_final,
                                              per_seed_ink) for s in seeds}
        payload.update({f'seed{s}/objective': v for s, v in per_seed_obj.items()})
        mean = sum(per_seed_obj.values()) / len(per_seed_obj)
        payload['objective'] = mean
        if len(per_seed_obj) > 1:
            payload['objective_std'] = (
                sum((v - mean) ** 2 for v in per_seed_obj.values())
                / (len(per_seed_obj) - 1)) ** 0.5
    if not args.skip_eval:
        payload['eval/flatten_failed'] = 0
    log_to_run(payload, files=metric_files, images=preview_images)
    print(f'[sweep_run_wrapper] done ({len(seeds)} seeds): {json.dumps(payload)}',
          flush=True)


if __name__ == '__main__':
    main()
