"""CT context around a proposed join; a join needs CT evidence to be accepted.

The coarse context and native flank checks are physical evidence, not a test of
fibre identity. In particular, averaging can blur neighbouring sheets together.
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy.ndimage import map_coordinates

CT_FEATURE_DIM = 32
MAX_ROI = 128
MAX_CHUNKS = 8


def _prefix(points, length=16.):
    d = np.linalg.norm(np.diff(points, axis=0), axis=1)
    s = np.r_[0., np.cumsum(d)]
    end = min(float(s[-1]), length)
    t = np.linspace(0., end, 9)
    return np.stack([np.interp(t, s, points[:, i]) for i in range(3)], axis=1)


def _coarsen(volume):
    """True box means, with every native voxel included exactly once."""
    coarse = volume.astype(np.float32) / 255.
    for axis in range(3):
        edges = np.linspace(0, coarse.shape[axis], 17, dtype=int)
        coarse = np.add.reduceat(coarse, edges[:-1], axis=axis)
        divisor = np.diff(edges).reshape([16 if k == axis else 1 for k in range(3)])
        coarse /= divisor
    return coarse


class CTSupport:
    """Reads at most one <=128³ region per join.

    ``read(low_xyz, high_xyz)`` returns the CT of that box as a ZYX array.
    ``features(a, b)`` returns ``valid``, a ``status`` and the 32-value
    ``vector``; a join is only accepted when ``valid`` is true.
    """
    def __init__(self, read, shape_xyz):
        self.read = read
        self.shape_xyz = np.asarray(shape_xyz, dtype=int)

    def _plan(self, a, b):
        a, b = (np.asarray(p, dtype=np.float64) for p in (a, b))
        for p in (a, b):
            if p.ndim != 2 or p.shape[1] != 3 or len(p) < 2 or not np.isfinite(p).all():
                raise ValueError("CT inputs must be finite tip-first XYZ snippets with at least two points")
        ca, cb = _prefix(a), _prefix(b)
        points = np.vstack((ca, cb))
        if np.any(points < 0) or np.any(points > self.shape_xyz-1):
            raise ValueError("out_of_bounds")
        low = np.floor(points.min(axis=0)-4).astype(int)
        high = np.ceil(points.max(axis=0)+5).astype(int)
        # Keep a true coarse physical neighbourhood even for a very short gap:
        # 64 native voxels become 16 cells (>=4 native voxels/cell).
        extra = np.maximum(0, 64-(high-low))
        low -= extra//2
        high += extra-extra//2
        if np.any(high-low > MAX_ROI):
            raise ValueError("roi_too_large")
        if np.any(low < 0) or np.any(high > self.shape_xyz):
            raise ValueError("out_of_bounds")
        keys = list(itertools.product(*(range(int(lo//128), int((hi-1)//128)+1) for lo, hi in zip(low[::-1], high[::-1]))))
        if len(keys) > MAX_CHUNKS:
            raise ValueError("too_many_chunks")
        return a, b, ca, cb, low, high

    def features(self, a, b):
        metadata = {"valid": False, "status": "unavailable", "vector": np.full(CT_FEATURE_DIM, np.nan, np.float32)}
        try:
            a, b, ca, cb, low, high = self._plan(a, b)
            volume = np.asarray(self.read(low, high))
            metadata["nonzeroFraction"] = float(np.count_nonzero(volume)/volume.size)
            if not volume.any():
                metadata["status"] = "no_ct_support"
                return metadata
            coarse = _coarsen(volume)

            def sample(points):
                return map_coordinates(volume.astype(np.float32), (np.asarray(points)-low)[:, ::-1].T,
                                       order=1, mode="nearest", prefilter=False)/255.

            bridge_points = a[0] + np.linspace(0, 1, 33)[:, None]*(b[0]-a[0])
            bridge = sample(bridge_points)
            continuation_a, continuation_b = sample(ca), sample(cb)
            delta = b[0]-a[0]
            norm = np.linalg.norm(delta)
            direction = delta/norm if norm > 1e-9 else np.array([1., 0., 0.])
            side = np.cross(direction, np.eye(3)[np.argmin(np.abs(direction))])
            side /= np.linalg.norm(side)
            up = np.cross(direction, side)
            fine_points = bridge_points[::2]
            center = sample(fine_points)
            flanks = np.stack([sample(fine_points+offset) for offset in (2*side, -2*side, 2*up, -2*up)])
            contrast = center-flanks.mean(axis=0)
            # The coarse field is box-averaged; coordinates identify cell centres.
            coarse_points = ((bridge_points-low+.5)/(high-low)*16-.5)[:, ::-1].T
            coarse_bridge = map_coordinates(coarse, coarse_points, order=1, mode="nearest", prefilter=False)
            tip_a, tip_b = float(continuation_a[:3].mean()), float(continuation_b[:3].mean())
            tips = (tip_a+tip_b)/2
            q = lambda p, f: float(np.quantile(p, f))
            values = [coarse.mean(), coarse.std(), q(coarse,.1), q(coarse,.5), q(coarse,.9), (coarse>0).mean(),
                      bridge.mean(), bridge.std(), bridge.min(), q(bridge,.1), q(bridge,.9), tip_a, tip_b,
                      abs(tip_a-tip_b), bridge.mean()-tips, (bridge<tips*.5).mean(), center.mean(), center.std(),
                      flanks.mean(), flanks.std(), contrast.mean(), q(contrast,.1), q(contrast,.9),
                      (contrast>0).mean(), np.abs(np.diff(bridge)).mean(), continuation_a.mean(),
                      continuation_b.mean(), abs(continuation_a.mean()-continuation_b.mean()),
                      coarse_bridge.mean(), coarse_bridge.std(), np.abs(bridge-coarse_bridge).mean(),
                      (bridge>max(.01, tips*.5)).mean()]
            vector = np.asarray(values, dtype=np.float32)
            if vector.shape != (CT_FEATURE_DIM,) or not np.isfinite(vector).all():
                raise ValueError("Invalid CT feature vector")
            metadata.update({"vector": vector, "valid": True, "status": "ok"})
        except Exception as exc:
            reason = str(exc)
            metadata.update(status=reason if reason in {"out_of_bounds", "roi_too_large", "too_many_chunks"} else "unavailable",
                            error=reason[:240] or type(exc).__name__)
        return metadata
