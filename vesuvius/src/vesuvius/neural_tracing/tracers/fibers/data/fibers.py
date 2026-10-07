"""Fiber-tracer data over annotated fibers: the data version, the z-band validation split and which training states it allows."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vesuvius.neural_tracing.tracers.shared.annotations.traced import TracedFiber
from vesuvius.neural_tracing.tracers.shared.geometry import CropSpec, block_start


DATA_VERSION = 2


@dataclass(frozen=True)
class ZBand:
    """Held-out band in trace-grid z (inclusive-exclusive)."""

    lo: float
    hi: float

    def contains(self, z):
        return (z >= self.lo) & (z < self.hi)


def split_fibers(fibers: list[TracedFiber], val_band: ZBand):
    """Split by the controlled geometry centroid; filter training states separately.

    Fibers whose annotation doubles back are kept out of training (validation is unchanged).
    """
    train, val = [], []
    for f in fibers:
        z = np.average((f.points[:-1, 2]+f.points[1:, 2])/2, weights=np.diff(f.s))
        if val_band.contains(z):
            val.append(f)
        elif not f.foldbacks:
            train.append(f)
    return train, val


def training_state_allowed(item, crop: CropSpec, band: ZBand | None):
    """Exclude the perturbed state, entire sampled block, and used history/labels.

    Keep the original 48-voxel position guard and also check the actual rotated
    crop's read footprint (including interpolation support). Applied before I/O
    to fresh and cached states. The circumsphere covers any CT-selected roll.
    """
    if band is None:
        return True
    pos, frame = item["pos"], item["frame"]
    if band.lo - 48 <= pos[2] < band.hi + 48:
        return False
    start = block_start(pos, frame, crop)[0]
    if start < band.hi and start + crop.block_size > band.lo:
        return False
    zs = [np.array([pos[2]])]
    for key, mask in (("hist_local", "hmask"), ("gt_history", "gt_history_mask"),
                      ("fut_local", "fmask")):
        if key in item:
            points = item[key][item[mask] > 0]
            if len(points):
                zs.append((points @ frame.T + pos)[:, 2])
    if "plane_ab" in item:
        # The forward-plane labels can look farther along GT than arclength targets.
        loc = np.c_[item["plane_ab"], item["planes"]]
        loc = loc[item["plane_mask"] > 0]
        if len(loc):
            zs.append((loc @ frame.T + pos)[:, 2])
    if "dense_ab" in item:
        loc = np.c_[item["dense_ab"], item["dense_planes"]][item["dense_mask"] > 0]
        if len(loc):
            zs.append((loc @ frame.T + pos)[:, 2])
    z = np.concatenate(zs)
    return not (z.min() < band.hi and z.max() >= band.lo)


