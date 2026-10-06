"""Local tracing frames, oriented crops and polyline helpers.

World coordinates are trace-grid ``x, y, z`` (float). A frame is a 3x3
matrix whose columns are ``(u, v, f)`` in world xyz: ``f`` is the heading,
``u, v`` span the cross-section. Oriented crops are laid out
``C, D, H, W`` = channels, forward (f), v, u.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class CropSpec:
    depth: int = 64  # samples along f
    width: int = 64  # samples along u and v
    behind: int = 16  # samples behind the current point
    spacing: float = 1.0
    @property
    def forward_coords(self) -> np.ndarray:
        return (np.arange(self.depth) - self.behind) * self.spacing

    @property
    def lateral_coords(self) -> np.ndarray:
        return (np.arange(self.width) - (self.width - 1) / 2.0) * self.spacing

    @property
    def center_offset(self) -> float:
        fc = self.forward_coords
        return 0.5 * float(fc[0] + fc[-1])

    @property
    def block_size(self) -> int:
        fc = self.forward_coords
        h = 0.5 * float(self.lateral_coords[-1] - self.lateral_coords[0])
        r = np.sqrt((fc - self.center_offset) ** 2 + 2 * h**2).max()
        return int(2 * np.ceil(r + 2.0))


def normalize(v: np.ndarray, axis: int = -1) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=axis, keepdims=True), 1e-9)


def frame_from_heading(f: np.ndarray, u_hint: np.ndarray | None = None) -> np.ndarray:
    """Frame with columns (u, v, f); ``u`` parallel-transported from ``u_hint``."""
    f = normalize(np.asarray(f, dtype=np.float64))
    if u_hint is not None:
        u = u_hint - np.dot(u_hint, f) * f
        if np.linalg.norm(u) < 1e-3:
            u_hint = None
    if u_hint is None:
        ref = np.array([0.0, 0.0, 1.0]) if abs(f[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        u = np.cross(ref, f)
    u = normalize(u)
    v = np.cross(f, u)
    return np.stack([u, v, f], axis=1)


def crop_local_grid(spec: CropSpec) -> np.ndarray:
    """Local (a=u, b=v, c=f) coordinates, shape (D, H, W, 3)."""
    lc = spec.lateral_coords
    fc = spec.forward_coords
    c, b, a = np.meshgrid(fc, lc, lc, indexing="ij")
    return np.stack([a, b, c], axis=-1)


def block_start(pos_xyz: np.ndarray, frame: np.ndarray, spec: CropSpec) -> np.ndarray:
    """zyx start of the axis-aligned block that contains the oriented crop."""
    center = pos_xyz + spec.center_offset * frame[:, 2]
    half = spec.block_size // 2
    return np.floor(center[::-1]).astype(np.int64) - half + 1


# ---------------------------------------------------------------- polylines


def arclength(p: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])


def resample_polyline(p: np.ndarray, step: float) -> np.ndarray:
    s = arclength(p)
    if s[-1] < step:
        return p[[0, -1]].copy()
    t = np.arange(0.0, s[-1] + 1e-9, step)
    return np.stack([np.interp(t, s, p[:, i]) for i in range(3)], 1)


def interp_at(p: np.ndarray, s: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.stack([np.interp(t, s, p[:, i]) for i in range(3)], -1)


def tangent_at(p: np.ndarray, s: np.ndarray, t: float, half: float = 3.0) -> np.ndarray:
    a = interp_at(p, s, np.array([max(t - half, 0.0)]))[0]
    b = interp_at(p, s, np.array([min(t + half, s[-1])]))[0]
    return normalize(b - a)


def dense_line(points, step):
    """Densify every original segment, preserving corners and both endpoints."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2 or not np.isfinite(points).all():
        raise ValueError('Expected a finite polyline with at least two xyz points')
    if not np.isfinite(step) or step <= 0:
        raise ValueError('Sampling step must be positive')
    # Vectorized, bit-identical to the per-segment loop ``a + arange(n)/n*(b-a)`` with
    # n = ceil(norm(b-a)/step). Only n and the zero-length test depend on the length;
    # where the vectorized length could round across either threshold, the segment
    # takes the loop's exact np.linalg.norm.
    delta = points[1:]-points[:-1]
    length = np.sqrt(np.einsum('ij,ij->i', delta, delta))
    ratio = length/step
    close = (np.abs(ratio-np.rint(ratio)) <= 1e-9*np.maximum(ratio, 1.)) | (np.abs(length-1e-9) <= 1e-12)
    for i in np.flatnonzero(close):
        length[i] = np.linalg.norm(delta[i])
    keep = length > 1e-9
    if not keep.any():
        raise ValueError('Polyline has zero length')
    counts = np.maximum(1, np.ceil(length[keep]/step).astype(np.int64))
    segment = np.repeat(np.arange(len(counts)), counts)
    k = np.arange(len(segment))-np.repeat(np.cumsum(counts)-counts, counts)
    t = (k/counts[segment])[:, None]
    return np.concatenate([points[:-1][keep][segment]+t*delta[keep][segment], points[-1:]])


class PolylineIndex:
    """Reusable exact-distance index of a polyline's segments."""
    def __init__(self, target):
        self.target = np.asarray(target, float)
        self.a = self.target[:-1]
        self.delta = np.diff(self.target, axis=0)
        self.length2 = np.einsum('ij,ij->i', self.delta, self.delta)
        self.tree = cKDTree(self.a+self.delta/2)
        self.half = np.sqrt(self.length2.max())/2


def exact_nearest(points, target, index=None):
    """Exact nearest points on a polyline, including interiors of long segments."""
    p = np.asarray(points, float)
    index = index or PolylineIndex(target)
    # Bound temporary candidate arrays for large queries.
    if len(p) > 256:
        chunks = [exact_nearest(p[i:i+256], target, index) for i in range(0, len(p), 256)]
        return tuple(np.concatenate(values) for values in zip(*chunks))
    a, delta, length2 = index.a, index.delta, index.length2
    # Midpoint broad phase: the nearest segment must be within current best
    # distance + half the longest segment of the query point.
    tree = index.tree
    _, first = tree.query(p)
    def project(points, ids):
        u = np.clip(((points-a[ids])*delta[ids]).sum(-1)/np.maximum(length2[ids], 1e-20), 0, 1)
        q = a[ids]+u[:, None]*delta[ids]
        return np.linalg.norm(points-q, axis=-1), q, u
    best = project(p, first)[0]
    # Preserve the traversal order of the former single-point queries, including
    # first-candidate tie breaking. SciPy otherwise sorts batched query results.
    groups = tree.query_ball_point(p, best+index.half+1e-8, return_sorted=False)
    counts = np.fromiter(map(len, groups), dtype=np.int64, count=len(p))
    offsets = np.r_[0, np.cumsum(counts)[:-1]]
    ids = np.concatenate(groups).astype(np.int64, copy=False)
    distance, nearest, u = project(np.repeat(p, counts, axis=0), ids)
    minima = np.minimum.reduceat(distance, offsets)
    positions = np.arange(len(ids))
    chosen = np.minimum.reduceat(np.where(distance == np.repeat(minima, counts),
                                         positions, len(ids)), offsets)
    return distance[chosen], nearest[chosen], ids[chosen], u[chosen]
