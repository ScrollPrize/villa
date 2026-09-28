"""Read-only endpoint cross-check on real candidates before discarding v1 trials."""
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import json
from pathlib import Path
import sys
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO/'vesuvius/src'), str(REPO)]
from vesuvius.neural_tracing.fiber_trace_3d.short_failure_benchmark import failure_trial_descriptor


def main():
    root = Path('/ephemeral/hendrik/fiber3d_train2/evaluations/train_failure_union_20260927_v1/reproduction_search')
    candidates = json.loads((root/'candidates.json').read_text())
    models = json.loads((root/'protocol.json').read_text())['models']
    @lru_cache(maxsize=None)
    def original(path):
        points = np.asarray(json.loads(Path(path).read_text())['line_points'], dtype=np.float64)
        return points, np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    @lru_cache(maxsize=None)
    def reference(model, case, reverse):
        doc = json.loads((Path(models[model]['root'])/'cases'/str(case)/'trace.json').read_text())
        points = np.asarray(doc['reference_points_base_xyz'], dtype=np.float64)
        if reverse: points = points[::-1]
        return points, np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    def check(candidate):
        points, arc = reference(candidate['model'], candidate['case'], candidate['reverse'])
        full, full_arc = original(candidate['piece_key'][0])
        a, b = candidate['run']['original_arcs_base']
        errors = []
        for direction, extent in (('forward', candidate['target']-a), ('reverse', b-candidate['target'])):
            for distance in sorted({100, int(extent//100)*100}):
                if not 0 < distance <= extent: continue
                trial = failure_trial_descriptor(candidate, direction, distance, points, arc)
                lo, hi = trial['original_arcs_base']
                for s, name in ((lo if direction == 'forward' else hi, 'expected_start_base_xyz'),
                                (hi if direction == 'forward' else lo, 'expected_end_base_xyz')):
                    actual = [np.interp(s, full_arc, full[:, k]) for k in range(3)]
                    error = float(np.linalg.norm(np.asarray(actual)-trial[name]))
                    if not np.isfinite(error) or error > .25:
                        raise ValueError(f'candidate {candidate["id"]} endpoint error {error}')
                    errors.append(error)
        return errors
    with ThreadPoolExecutor(max_workers=128) as pool:
        errors = [v for row in pool.map(check, candidates) for v in row]
    print(json.dumps(dict(candidates=len(candidates), endpoints=len(errors),
                         max_error_base_voxels=max(errors), tolerance=.25)), flush=True)


if __name__ == '__main__': main()
