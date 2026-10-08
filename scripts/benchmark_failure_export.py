"""Read-only real-artifact CPU benchmark, invoked through mining runner."""
from concurrent.futures import ProcessPoolExecutor
from itertools import islice
import json
import multiprocessing
from pathlib import Path
import statistics
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO/'vesuvius/src'), str(REPO)]
from vesuvius.neural_tracing.fiber_trace_3d.failure_trial_audit import audit_trace_batch, bounded_map


def main():
    batches = Path('/ephemeral/hendrik/fiber3d_train2/evaluations/train_failure_union_20260927_v1/'
                   'reproduction_search/batches')
    root = max(batches.glob('*/attempt*/full/summary.json'), key=lambda p: p.stat().st_mtime_ns).parent
    paths = [str(p) for p in islice((root/'cases').glob('*/trace.json'), 512)]
    assert len(paths) == 512
    evaluation = json.loads((root.parent.parent/'evaluation.json').read_text())
    tasks = [(paths[i:i+16], evaluation.get('excursion_grace_base_voxels', 0), True, {})
             for i in range(0, len(paths), 16)]
    reference = None
    measurements = []
    for workers in (1, 16, 32):
        times = []
        pool = None if workers == 1 else ProcessPoolExecutor(max_workers=workers,
                mp_context=multiprocessing.get_context('spawn'))
        try:
            for iteration in range(3):
                start = time.perf_counter()
                rows = list(map(audit_trace_batch, tasks) if pool is None else
                            bounded_map(pool, audit_trace_batch, tasks, pending=workers*2))
                elapsed = time.perf_counter()-start
                if reference is None:
                    reference = rows
                assert rows == reference
                times.append(elapsed)
                print(json.dumps(dict(workers=workers, iteration=iteration, seconds=elapsed,
                                      artifacts_per_second=len(paths)/elapsed)), flush=True)
        finally:
            if pool is not None:
                pool.shutdown()
        measurements.append(dict(workers=workers, iterations=3, seconds=times,
            mean=statistics.mean(times), median=statistics.median(times),
            minimum=min(times), maximum=max(times)))
    print(json.dumps(dict(dataset=str(root), artifacts=len(paths), bytes=sum(Path(p).stat().st_size for p in paths),
                         equal=True, measurements=measurements)), flush=True)


if __name__ == '__main__':
    main()
