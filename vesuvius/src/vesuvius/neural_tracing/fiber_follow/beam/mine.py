"""Find the controlled spans on which the hand beam fails, to oversample them.

The hand tracer succeeds on most spans, so pools with an off-fiber hand choice
are rare in uniformly sampled traces. A one-off ``whole_fiber_metric`` pass
over the training fibers records, per fiber, the span indices that restarted
(``fiber.spans`` order equals the metric's segment order). The result is
cached as JSON bound to the fiber geometry identities.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

from vesuvius.neural_tracing.fiber_follow.beam.native import NativeBeam, fiber_input_from_traced
from vesuvius.neural_tracing.fiber_follow.data import fiber_manifest

CACHE_VERSION = 1


def mine_hard_spans(beam: NativeBeam, fibers, grid_scale: float, error_threshold_base: float = 20.0,
                    progress=None) -> dict:
    hard = {}
    total = 0
    start = time.time()
    for k, fiber in enumerate(fibers):
        metric = beam.whole_fiber_metric(fiber_input_from_traced(fiber, grid_scale), error_threshold_base=error_threshold_base)
        failed = [i for i, seg in enumerate(metric['segments']) if not seg['success']]
        total += len(failed)
        if failed:
            hard[fiber.name] = failed
        if progress and (k + 1) % 25 == 0:
            progress(dict(fibers=k + 1, hard_spans=total, seconds=round(time.time() - start, 1)))
    return hard


def load_or_mine_hard_spans(path, beam: NativeBeam, fibers, grid_scale: float, *, error_threshold_base=20.0,
                            progress=None) -> dict:
    path = Path(path)
    manifest = fiber_manifest(fibers)
    if path.exists():
        cached = json.loads(path.read_text())
        if cached.get('version') == CACHE_VERSION and cached.get('fiber_manifest') == manifest \
                and cached.get('error_threshold_base') == error_threshold_base:
            return cached['hard_spans']
        raise ValueError(f'{path} was mined for different fibers or settings; delete it to re-mine')
    hard = mine_hard_spans(beam, fibers, grid_scale, error_threshold_base, progress)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.partial.json')
    tmp.write_text(json.dumps(dict(version=CACHE_VERSION, fiber_manifest=manifest,
                                   error_threshold_base=error_threshold_base, beam_spec=beam.spec.to_dict(),
                                   hard_spans=hard)))
    tmp.replace(path)
    return hard
