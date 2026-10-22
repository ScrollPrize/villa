"""Managed three-arm replay on immutable prepared inputs and frozen GT.

No machine-specific paths, optimization implementation, or new GT protocol.
See docs/fiber_layout_benchmark.md for input contracts and CLI usage.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from scipy.special import expit
from . import fiber_uv_layout as uv
from .fiber_gt_constraint_score import (
    arrays, piece_at, event_residuals, enabled_l1, gt_solve_width_metrics)
from .fiber_uv_arc_phase import event_arc_points
from .fiber_winding_order import order_margin
from lasagna.inference_provenance import atomic_write, sha256_file

ARMS = ('free', 'joint', 'selectable')
PROTOCOL = 'frozen-gt-map-half-up-v1'
SOLVER_MODULE = 'vesuvius.neural_tracing.fiber_trace_3d.fiber_implicit_winding'


def read(path):
    return json.loads(path.read_text())


def score(spec, solve, reference_root):
    solve, reference_root = Path(solve), Path(reference_root)
    cfg = read(solve / 'config.json')
    d = arrays(cfg['tensor_cache'])
    sol = arrays(solve / 'solution.npz')
    gc = read(reference_root / 'config.json')
    gd, gs = arrays(gc['tensor_cache']), arrays(reference_root / 'solution.npz')
    for key in ('source_id', 'start', 'end', 'family'):
        np.testing.assert_array_equal(sol[key], d[key])
    events = read(Path(spec['events']))
    if isinstance(events, dict):
        events = events['events']
    events = [dict(e, test_piece=piece_at(d, e['test_source'], e['test_arc'])) for e in events]
    if any(e['test_piece'] is None for e in events):
        raise ValueError('GT event outside test pieces')
    _, _, keep, pick = uv.hard_assignment(gs['state'], len(gs['widths']), gc, best_nonbroken=True)
    if not keep.all():
        raise ValueError('Frozen reference contains disabled fibers')
    # Match the saved historical protocol, including its label tie convention.
    mapped = sol['state'].copy()
    mapped[:, 0] = np.floor(mapped[:, 0] + .5)
    required = event_residuals(events, d, mapped, None, len(sol['widths']), gd, gs, pick,
                              dict(cfg, soft_winding_model=True), gc, sol['widths'], saved_geometry=sol)
    enabled = sol['state'][[e['test_piece'] for e in events], 3] >= 0
    parallel = np.array([e['parallel'] for e in events], dtype=bool)
    metrics = enabled_l1(required, enabled, parallel)
    residual = required - metrics['gt_global_offset']
    family = np.array([e['family_delta'] for e in events])
    # Undo the serialized residual convention; this does not calibrate signs.
    displacement = residual + np.array([e['signed_distance'] for e in events]) - .5 * family
    normal = np.array([e['normal_cosine'] for e in events])
    order = order_margin(displacement, np.sign(normal), family, parallel, cfg.get('sign_order_leeway', .5))
    signs = {}
    for name, kind in [('parallel', parallel), ('perpendicular', ~parallel)]:
        use = enabled & kind & (normal != 0)
        signs[name] = dict(violations=int((use & (order < 0)).sum()), events=int(use.sum()),
                           percent=float(100 * (use & (order < 0)).sum() / use.sum()) if use.any() else None)
    q = event_arc_points(gd, gs['state'], pick, gs['widths'],
                         np.array([e['gt_piece'] for e in events]), np.array([e['gt_arc'] for e in events]))
    metrics['gt_solve_widths'] = gt_solve_width_metrics(sol['widths'], gs['widths'],
                                                        np.floor(q).astype(int), metrics['gt_global_offset'])
    retention = None
    if spec.get('gt_source_ids'):
        ids = np.asarray(spec['gt_source_ids'])
        if not np.isin(ids, d['source_id']).all():
            raise ValueError('Requested GT identities are missing from solve')
        mask = np.isin(d['source_id'], ids)
        lengths = d['lengths'][mask]
        probs = expit(sol['state'][mask, 3])
        if spec.get('force_gt_enabled') and not np.all(sol['state'][mask, 3] == 100):
            raise ValueError('Forced GT enablement changed')
        if lengths.sum() <= 0:
            raise ValueError('GT length must be positive')
        retention = dict(kept_vx=float(lengths[probs >= .5].sum()), total_vx=float(lengths.sum()),
                         percent=float(100 * lengths[probs >= .5].sum() / lengths.sum()))
    summary = read(solve / 'summary.json')
    if not summary.get('completed'):
        raise ValueError('Cannot score an incomplete solve')
    return dict(score=metrics, signs=signs, retention=retention, final=summary['final'],
                seconds=summary['seconds'], solution_sha256=sha256_file(solve / 'solution.npz'))


def ranking(results):
    sign_sum = sum(results[a]['signs']['perpendicular']['violations'] for a in ARMS)
    fraction = results['selectable']['retention']['percent'] / 100
    return dict(sign_sum=sign_sum, retained_gt_fraction=fraction,
                score=sign_sum / fraction if fraction else None,
                perp_l1_sum=sum(results[a]['score']['gt_perpendicular']['l1'] for a in ARMS))


def validate_expected(result, expected):
    for kind in ('parallel', 'perpendicular'):
        for key in ('violations', 'events'):
            if result['signs'][kind][key] != expected['signs'][kind][key]:
                raise ValueError(f'Historical {kind} sign {key} changed')
        for metric in ('l1', 'correct_percent'):
            np.testing.assert_allclose(result['score']['gt_' + kind][metric],
                                       expected['score']['gt_' + kind][metric], rtol=0, atol=1e-10)
    if result['solution_sha256'] != expected['solution_sha256']:
        raise ValueError('Historical checkpoint changed')
    np.testing.assert_allclose(result['final']['loss'], expected['final']['loss'], rtol=0, atol=1e-10)
    if result['retention'] is not None:
        gt = expected['gt_enablement']
        total = sum(x['length_vx'] for x in gt)
        kept = sum(x['length_vx'] for x in gt if x['enabled_probability'] >= .5)
        np.testing.assert_allclose([result['retention']['kept_vx'], result['retention']['total_vx']],
                                   [kept, total], rtol=0, atol=1e-8)


def load_settings(config_path, runtime_path):
    config_path, runtime_path = Path(config_path).resolve(), Path(runtime_path).resolve()
    settings, runtime = read(config_path), read(runtime_path)
    if settings.get('version') != 1 or settings.get('protocol') != PROTOCOL:
        raise ValueError('Unsupported benchmark version or scoring protocol')
    if set(settings['arms']) != set(ARMS) or set(runtime['devices']) != set(ARMS):
        raise ValueError('Exactly free, joint and selectable arms are required')
    if runtime.get('gpu_manager', {}).get('exclusive') is not False:
        raise ValueError('This replay requires explicit nonexclusive manager grants')
    devices = [str(runtime['devices'][a]) for a in ARMS]
    if len(set(devices)) != 3 or any(',' in x or not x for x in devices):
        raise ValueError('Assign three distinct single-GPU lanes')
    def resolve(value, base):
        return str((base / value).resolve())
    settings['reference_root'] = resolve(settings['reference_root'], config_path.parent)
    runtime['output_root'] = resolve(runtime['output_root'], runtime_path.parent)
    for spec in settings['arms'].values():
        for key in ('solver_config', 'events', 'historical_solve', 'expected_report'):
            spec[key] = resolve(spec[key], config_path.parent)
    if settings['arms']['free'].get('gt_source_ids'):
        raise ValueError('Free arm must not include GT training fibers')
    for arm in ('joint', 'selectable'):
        if not settings['arms'][arm].get('gt_source_ids'):
            raise ValueError('GT arms require explicit source IDs')
    if (settings['arms']['joint'].get('force_gt_enabled') is not True or
            settings['arms']['selectable'].get('force_gt_enabled') is not False):
        raise ValueError('Incorrect forced/selectable GT policies')
    return settings, runtime


def prepare(settings, runtime):
    """Resolve execution options, hash immutable solve/scoring inputs; no CUDA."""
    root = Path(runtime['output_root'])
    inputs, configs = {}, {}
    def track(path):
        path = Path(path).resolve()
        inputs[str(path)] = sha256_file(path)
    reference = Path(settings['reference_root'])
    for path in (reference / 'config.json', reference / 'solution.npz'):
        track(path)
    track(read(reference / 'config.json')['tensor_cache'])
    for arm, spec in settings['arms'].items():
        for key in ('solver_config', 'events', 'expected_report'):
            track(spec[key])
        original = read(Path(spec['solver_config']))
        cfg = dict(original.get('fiber_implicit_winding', original))
        # Input paths in solver configs follow the existing solver API: absolute.
        for key in ('tensor_cache', 'initial_solution'):
            if not Path(cfg[key]).is_absolute():
                raise ValueError(f'{key} must be absolute in prepared solver configs')
            track(cfg[key])
        init_config = Path(cfg['initial_solution']).parent / 'config.json'
        track(init_config)
        initial = read(init_config)
        track(initial['soft_samples'])
        if inputs[str(Path(cfg['tensor_cache']).resolve())] != initial['tensor_sha256']:
            raise ValueError('Initializer tensor fingerprint mismatch')
        forced = set(cfg.get('force_keep_source_ids', []))
        wanted = set(spec.get('gt_source_ids', [])) if spec.get('force_gt_enabled') else set()
        if forced != wanted:
            raise ValueError(f'{arm}: forced GT policy differs from solver config')
        output = root / arm / 'solve'
        if output.resolve() == Path(spec['historical_solve']).resolve():
            raise ValueError('Output would overwrite historical solve')
        cfg.update(output_root=str(output), devices=str(runtime['devices'][arm]))
        cfg['gpu_manager'] = dict(runtime['gpu_manager'], label=f'fiber-layout-benchmark-{arm}')
        if not cfg.get('estimate_fixed_length_vram') and not cfg['gpu_manager'].get('vram_bytes_per_gpu'):
            raise ValueError('Explicit VRAM budget or solver estimation required')
        configs[arm] = cfg
    code = {str(p): sha256_file(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    manifest = dict(version=1, protocol=PROTOCOL, settings=settings, runtime=runtime,
                    inputs=inputs, code=code, configs=configs)
    stamp = root / 'manifest.json'
    if stamp.exists() and read(stamp) != manifest:
        raise ValueError('Inputs, settings or code changed; use a new output root')
    root.mkdir(parents=True, exist_ok=True)
    atomic_write(stamp, manifest, durable=False)
    for arm, cfg in configs.items():
        atomic_write(root / (arm + '.json'), cfg, durable=False)
    return manifest


def validate_reference(settings, root):
    results = {}
    for arm in ARMS:
        spec = settings['arms'][arm]
        results[arm] = score(spec, spec['historical_solve'], settings['reference_root'])
        validate_expected(results[arm], read(Path(spec['expected_report'])))
    result = dict(arms=results, ranking=ranking(results))
    atomic_write(root / 'reference_validation.json', result, durable=False)
    return result


def launch_arm(arm, root):
    """The solver's managed entrypoint owns acquisition, lifetime and release."""
    environment = dict(os.environ)
    environment['PYTHONPATH'] = os.pathsep.join(
        [str(Path(__file__).resolve().parents[3]), environment.get('PYTHONPATH', '')])
    command = [sys.executable, '-m', SOLVER_MODULE, '--config', str(root / (arm + '.json'))]
    with (root / (arm + '.log')).open('w') as log:
        subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)


def write_report(root, results, previous):
    result = dict(protocol=PROTOCOL, arms=results, ranking=ranking(results), previous=previous)
    atomic_write(root / 'comparison.json', result, durable=False)
    lines = ['# Fiber layout benchmark replay', '',
             'Frozen GT scoring; old/new use identical inputs and metric protocol.', '',
             '| Arm | Run | Loss | Perp L1 | Perp correct % | Perp signs | Par L1 | Par correct % | Par signs | GT kept % | Width factor | Solve s |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    def number(value, digits):
        return '—' if value is None else f'{value:.{digits}f}'
    for arm in ARMS:
        for name, row in [('old', previous['arms'][arm]), ('new', results[arm])]:
            s, signs = row['score'], row['signs']
            retention = row['retention']
            values = [arm, name, f"{row['final']['loss']:.6f}", f"{s['gt_perpendicular']['l1']:.6f}",
                      f"{s['gt_perpendicular']['correct_percent']:.2f}",
                      f"{signs['perpendicular']['violations']}/{signs['perpendicular']['events']}",
                      f"{s['gt_parallel']['l1']:.6f}", f"{s['gt_parallel']['correct_percent']:.2f}",
                      f"{signs['parallel']['violations']}/{signs['parallel']['events']}",
                      f"{retention['percent']:.2f}" if retention else '—',
                      number(s['gt_solve_widths'].get('mean_normalized_factor'), 4), f"{row['seconds']:.1f}"]
            lines.append('| ' + ' | '.join(values) + ' |')
    lines += ['', 'Times use nonexclusive GPU grants, not isolated throughput measurements.', '']
    (root / 'report.md').write_text('\n'.join(lines))
    return result


def run(settings, runtime, action='run'):
    torch.set_num_threads(int(runtime.get('scoring_threads', 4)))
    root = Path(runtime['output_root'])
    prepare(settings, runtime)
    if action == 'prepare':
        return dict(prepared=True, output_root=str(root))
    previous = validate_reference(settings, root)
    if action == 'validate-reference':
        return dict(validated=True, ranking=previous['ranking'])
    started, results = time.monotonic(), {}
    def publish(state, **extra):
        atomic_write(root / 'status.json', dict(status=state, completed=list(results),
                     seconds=time.monotonic()-started, **extra), durable=False)
    try:
        if action == 'run':
            # Fail before starting any arm rather than mix fresh and stale runs.
            if any((root / a / 'solve').exists() for a in ARMS):
                raise FileExistsError('Existing solve output; use score or a new output root')
            publish('running')
            with ThreadPoolExecutor(max_workers=3) as pool:
                jobs = {pool.submit(launch_arm, a, root): a for a in ARMS}
                for future in as_completed(jobs):
                    arm = jobs[future]
                    future.result()
                    results[arm] = score(settings['arms'][arm], root / arm / 'solve', settings['reference_root'])
                    atomic_write(root / arm / 'benchmark.json', results[arm], durable=False)
                    publish('running')
        elif action == 'score':
            results = {a: score(settings['arms'][a], root / a / 'solve', settings['reference_root']) for a in ARMS}
        else:
            raise ValueError('Unsupported action')
        # Re-hash inputs/code after completion too; fail closed on mutation.
        prepare(settings, runtime)
        result = write_report(root, results, previous)
        publish('finished', ranking=result['ranking'])
        return dict(output_root=str(root), ranking=result['ranking'])
    except Exception as error:
        publish('failed', error=repr(error))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--runtime', required=True, type=Path)
    parser.add_argument('--action', choices=['prepare', 'validate-reference', 'run', 'score'], default='prepare')
    args = parser.parse_args(argv)
    settings, runtime = load_settings(args.config, args.runtime)
    print(json.dumps(run(settings, runtime, args.action), indent=2))


if __name__ == '__main__':
    main()
