"""Open-ended tracing with the beam (plus optional learned hook), in the
``ModelTracer``/``FieldTracer`` interface used by ``evaluate`` and ``diag``."""
from __future__ import annotations

import numpy as np

from vesuvius.neural_tracing.fiber_follow.beam.native import NativeBeam


def _reason(raw: str) -> str:
    if raw.startswith('hook_stop'):
        return 'confidence'
    if raw.startswith('trace_distance'):
        return 'max_len'
    return raw.split(':')[0]


class BeamTracer:
    """Traces each seed for up to ``max_len`` grid voxels with ``trace_open``.

    ``hook`` is applied to every seed; with a ``ModelBeamHook`` whose
    ``stop_threshold`` is set, the trace ends where the model stops trusting
    every candidate, mirroring the follower's confidence stop.
    """

    def __init__(self, beam: NativeBeam, hook=None, max_len: float = 6000.):
        self.beam, self.hook, self.max_len = beam, hook, float(max_len)
        self.p = self  # ``rollout_diag`` temporarily sets ``tracer.p.max_len``

    def trace(self, seeds_xyz, headings, histories=None, abort=None, on_decision=None):
        paths, reasons = [], []
        for seed, heading in zip(np.asarray(seeds_xyz, float), np.asarray(headings, float)):
            try:
                points, reason, _ = self.beam.trace_open(seed, heading, self.max_len, hook=self.hook)
            except ValueError as exc:  # no valid prediction at the seed, etc.
                points, reason = seed[None].copy(), f'invalid:{exc}'
            paths.append(np.asarray(points, np.float64))
            reasons.append(_reason(reason))
            if abort is not None and abort():
                break
        return paths, reasons

    def close(self):
        pass
