"""Annotated fibers: the fiber type (densified, kink-repaired points with arclength, spans and provenance) and its geometry manifest."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from vc3d_fiber_format import FiberTraceSegmentMetadata



@dataclass(frozen=True)
class FiberSpan:
    start: float
    end: float
    provenance: FiberTraceSegmentMetadata


@dataclass
class TracedFiber:
    name: str
    points: np.ndarray  # (N, 3) trace-grid xyz; all annotated vertices, gaps <= 1 voxel
    s: np.ndarray
    tag: str
    spans: tuple[FiberSpan, ...] = ()
    endpoint_stop: tuple[bool, bool] = (False, False)
    source_hash: str = ""
    excluded_tail_length: float = 0.0
    kink_repairs: int = 0  # annotation kinks replaced by smooth bridges at load (annotation_repair)
    foldbacks: tuple[float, ...] = ()  # arcs where the annotation doubles back; kept out of training

    @property
    def length(self) -> float:
        return float(self.s[-1])


def fiber_manifest(fibers):
    """Ordered identities for caches containing fiber indices and arc positions."""
    if hasattr(fibers, 'manifest_entries'):
        return fibers.manifest_entries()
    return [dict(name=f.name, source_hash=f.source_hash,
                 geometry_hash=hashlib.sha256(np.asarray(f.points, dtype="<f8").tobytes()).hexdigest(),
                 endpoint_stop=list(f.endpoint_stop)) for f in fibers]
