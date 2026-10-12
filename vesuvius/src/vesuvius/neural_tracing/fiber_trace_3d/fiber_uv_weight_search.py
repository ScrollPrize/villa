"""Resumable, final-metric coordinate search using the unchanged joint solver."""
import hashlib
import json
import math
from pathlib import Path


def configuration_key(values):
    return hashlib.sha256(json.dumps(values, sort_keys=True, allow_nan=False).encode()).hexdigest()[:20]


def coordinate_search(baseline, evaluate, grids, tolerance=1e-4, accepted=None):
    """Restart every coordinate after improvement; finish at a grid-local minimum.

    evaluate must cache complete configurations. Resuming by replay is deliberate:
    cached decisions reproduce the same path, including interrupted acceptances.
    """
    current = dict(baseline)
    best = evaluate(current)
    for grid in grids:
        while True:
            improved = False
            for name, values in grid.items():
                for value in values:
                    if value == current[name]:
                        continue
                    candidate = dict(current, **{name: value})
                    result = evaluate(candidate)
                    if result['ref_l1'] < best['ref_l1'] - tolerance:
                        old = best
                        current, best = candidate, result
                        if accepted:
                            accepted(name, current, best, old)
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
    return current, best


def metrics(summary):
    final = summary['final']
    reference = final['reference']
    l1 = reference['mean_absolute_winding_error']
    if l1 is None or not math.isfinite(l1) or reference['matched_fibers'] == 0 or not math.isfinite(final['loss']):
        raise ValueError('No finite final reference L1; trial cannot be ranked')
    return dict(ref_l1=l1, loss=final['loss'],
        within_two_percent=reference['within_two_windings_percent'],
        retained_length_percent=final['retained_arc_percent'],
        matched_fibers=reference['matched_fibers'],
        width_factor=reference['widths']['mean_normalized_factor'],
        sign_percent=final['sign_violation_percent'],
        fit_range=reference['fitted_winding_q10_q90'],
        reference_range=reference['reference_winding_q10_q90'],
        solve_seconds=summary['solve_wall_seconds'])


def run(config):
    from lasagna.inference_provenance import atomic_write
    from .fiber_uv_layout import solve
    import gc
    import torch
    import numpy as np
    from .fiber_normal_sign import sha256_file

    if not config['gpu_manager'].get('exclusive'):
        raise ValueError('Weight search requires an exclusive managed lease')
    spec = config['loss_weight_search']
    root = Path(config['output_root'])
    root.mkdir(parents=True, exist_ok=True)
    base = {k: v for k, v in config.items() if k != 'loss_weight_search'}
    if base.get('warm_start') or base.get('reference_initialization'):
        raise ValueError('This search requires fresh initialization')
    names = spec['weights']
    initial = {name: float(base[name]) for name in names}
    if not all(math.isfinite(v) and v > 0 for v in initial.values()):
        raise ValueError('Search baseline weights must be finite and positive')
    grids = [{name: [initial[name] * factor for factor in factors] for name in names}
             for factors in ([.25, 1., 4.], [.25, .5, 1., 2., 4.])]
    paths = [Path(base['tensor_cache']), Path(base['tensor_cache']).with_suffix('.json'),
             Path(base['reference_projection']), Path(base['reference_widths']['meta'])]
    paths += sorted(Path(__file__).parent.glob('fiber_uv*.py'))
    paths += [Path(__file__).with_name('fiber_reference_comparison.py'),
              Path(base['reference_subset'])]
    signature = Path(base['reference_projection']).with_name('projection_signature.json')
    if signature.exists():
        paths.append(signature)
    manifest = dict(base=base, grids=grids, tolerance=1e-4,
                    hashes={str(p): sha256_file(p) for p in paths})
    manifest_path = root / 'search_manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError('Search configuration changed; use a new output root')
    atomic_write(manifest_path, manifest, durable=False)
    results = []
    acceptances = []
    announced_path = root / 'announced.json'
    announced = set(json.loads(announced_path.read_text())) if announced_path.exists() else set()

    def evaluate(values):
        key = configuration_key(values)
        trial_root = root / 'trials' / key
        child = dict(base, **values, output_root=str(trial_root))
        trial_root.mkdir(parents=True, exist_ok=True)
        saved_config = trial_root / 'config.json'
        if saved_config.exists() and json.loads(saved_config.read_text()) != child:
            raise ValueError('Trial config mismatch: ' + str(trial_root))
        atomic_write(saved_config, child, durable=False)
        summary_path = trial_root / 'summary.json'
        if not summary_path.exists():
            atomic_write(root / 'active_trial.json', dict(key=key, weights=values), durable=False)
            print('WEIGHT SEARCH TRIAL ' + json.dumps(dict(key=key, weights=values)), flush=True)
            solve(child)
            gc.collect()
            torch.cuda.empty_cache()
        summary = json.loads(summary_path.read_text())
        if summary['final']['optimizer_updates'] != child['steps']:
            raise ValueError('Incomplete trial cannot be ranked')
        with np.load(trial_root / 'solution.npz') as solution:
            if not np.isfinite(solution['state']).all() or not np.isfinite(solution['widths']).all() or np.any(solution['widths'] <= 100):
                raise ValueError('Nonfinite final solution')
            with np.load(base['tensor_cache']) as data:
                for field in ('source_id', 'start', 'end', 'family'):
                    if not np.array_equal(solution[field], data[field]):
                        raise ValueError('Changed solution identity: ' + field)
        result = dict(metrics(summary), key=key, weights=values, output_root=str(trial_root))
        if not any(row['key'] == key for row in results):
            results.append(result)
            atomic_write(root / 'results.json', results, durable=False)
        return result

    def accept(name, values, best, old):
        row = dict(best, changed_weight=name, previous_ref_l1=old['ref_l1'])
        acceptances.append(row)
        atomic_write(root / 'acceptances.json', acceptances, durable=False)
        atomic_write(root / 'best.json', best, durable=False)
        if best['key'] not in announced:
            print('WEIGHT SEARCH IMPROVED ' + json.dumps(row), flush=True)
            announced.add(best['key'])
            atomic_write(announced_path, sorted(announced), durable=False)

    values, best = coordinate_search(initial, evaluate, grids, accepted=accept)
    final = dict(best=best, weights=values, trials=len(results), acceptances=acceptances,
        termination='Every alternative grid value of every coordinate tested at final incumbent; no L1 improvement >1e-4',
        caveat='Reference is tuning data; retained cohort can change. Finite-grid coordinate optimum only.')
    atomic_write(root / 'best.json', best, durable=False)
    atomic_write(root / 'summary.json', final, durable=False)
    print('WEIGHT SEARCH COMPLETE ' + json.dumps(final), flush=True)
    return final
