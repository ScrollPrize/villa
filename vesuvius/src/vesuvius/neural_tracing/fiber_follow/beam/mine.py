"""Find the controlled spans on which the hand beam fails, to oversample them.

The hand tracer succeeds on most spans, so pools with an off-fiber hand choice
are rare in uniformly sampled traces. A one-off ``whole_fiber_metric`` pass
over the training fibers records, per fiber, the span indices that restarted
(``fiber.spans`` order equals the metric's segment order). The result is
cached as JSON bound to the fiber geometry identities.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import dataclasses
import json
import multiprocessing
from pathlib import Path
import time

from vesuvius.neural_tracing.fiber_follow.beam.native import NativeBeam, fiber_input_from_traced
from vesuvius.neural_tracing.fiber_follow.data import fiber_manifest

CACHE_VERSION = 1


def _failed_spans(beam: NativeBeam, fiber, grid_scale: float, error_threshold_base: float) -> list:
    metric = beam.whole_fiber_metric(fiber_input_from_traced(fiber, grid_scale), error_threshold_base=error_threshold_base)
    return [i for i, seg in enumerate(metric['segments']) if not seg['success']]


_WORKER = {}


def _init_worker(spec, grid_scale):
    # Each process opens its own readers (they must never cross a fork).
    _WORKER['beam'] = NativeBeam(spec, grid_scale)


def _worker_failed_spans(fiber, grid_scale, error_threshold_base):
    return _failed_spans(_WORKER['beam'], fiber, grid_scale, error_threshold_base)


def mine_hard_spans(beam: NativeBeam, fibers, grid_scale: float, error_threshold_base: float = 20.0,
                    progress=None, workers: int = 0) -> dict:
    """``workers`` > 1 traces fibers concurrently in single-threaded forkserver
    processes (one fiber's metric does not depend on any other fiber)."""
    hard = {}
    total = 0
    start = time.time()

    def record(fiber, failed, done):
        nonlocal total
        total += len(failed)
        if failed:
            hard[fiber.name] = failed
        if progress and done % 25 == 0:
            progress(dict(fibers=done, hard_spans=total, seconds=round(time.time() - start, 1)))

    if workers <= 1:
        for k, fiber in enumerate(fibers):
            record(fiber, _failed_spans(beam, fiber, grid_scale, error_threshold_base), k + 1)
        return hard
    spec = dataclasses.replace(beam.spec, parallel_threads=1)
    with ProcessPoolExecutor(max(1, min(workers, len(fibers))), mp_context=multiprocessing.get_context('forkserver'),
                             initializer=_init_worker, initargs=(spec, grid_scale)) as pool:
        # Longest fibers first, so a few long ones do not finish last. Progress
        # counts completions; the result is keyed by fiber and order-independent.
        futures = {pool.submit(_worker_failed_spans, fibers[i], grid_scale, error_threshold_base): i
                   for i in sorted(range(len(fibers)), key=lambda i: -len(fibers[i].points))}
        try:
            for done, future in enumerate(as_completed(futures), 1):
                record(fibers[futures[future]], future.result(), done)
        except BaseException:
            pool.shutdown(cancel_futures=True)
            raise
    # Same key order as in-process mining (the JSON cache is then identical).
    return {f.name: hard[f.name] for f in fibers if f.name in hard}


def load_or_mine_hard_spans(path, beam: NativeBeam, fibers, grid_scale: float, *, error_threshold_base=20.0,
                            progress=None, workers: int = 0) -> dict:
    path = Path(path)
    manifest = fiber_manifest(fibers)
    if path.exists():
        cached = json.loads(path.read_text())
        if cached.get('version') == CACHE_VERSION and cached.get('fiber_manifest') == manifest \
                and cached.get('error_threshold_base') == error_threshold_base:
            return cached['hard_spans']
        raise ValueError(f'{path} was mined for different fibers or settings; delete it to re-mine')
    hard = mine_hard_spans(beam, fibers, grid_scale, error_threshold_base, progress, workers)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.partial.json')
    tmp.write_text(json.dumps(dict(version=CACHE_VERSION, fiber_manifest=manifest,
                                   error_threshold_base=error_threshold_base, beam_spec=beam.spec.to_dict(),
                                   hard_spans=hard)))
    tmp.replace(path)
    return hard
