"""Clean the stitched fibers before they are written.

Inferred joins that turn too sharply are cut, fibers near the black outside
the papyrus and short fibers are removed, and the points of every fiber are
thinned. The cuts and removals are decided on the full geometry.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

# Directions on each side of a join are measured over this many voxels.
TANGENT_SUPPORT = 8.0
# Thinned fibers stay this close to every original point, and lose at most
# this fraction of the length of any stretch they replace.
TOLERANCE = 0.05
LENGTH_LOSS = 0.001
# Exterior black is located on cubes of this many voxels.
CELL = 4


def arclength(points: np.ndarray) -> np.ndarray:
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]


def join_angle(points: np.ndarray, arc: np.ndarray, low: int, high: int) -> float | None:
    """Turn in degrees between the directions entering ``points[low]`` and
    leaving ``points[high]``: 0 straight on, 180 back. None at a fiber's end."""

    def at(s: float) -> np.ndarray:
        return np.array([np.interp(s, arc, points[:, j]) for j in range(3)])

    incoming = points[low] - at(max(0.0, arc[low] - TANGENT_SUPPORT))
    outgoing = at(min(arc[-1], arc[high] + TANGENT_SUPPORT)) - points[high]
    norm = float(np.linalg.norm(incoming) * np.linalg.norm(outgoing))
    if norm < 1e-12:
        return None
    return float(np.degrees(np.arccos(np.clip(incoming @ outgoing / norm, -1.0, 1.0))))


def cut_sharp_joins(traces: Sequence[dict[str, Any]], max_angle: float) -> list[dict[str, Any]]:
    """Fibers split at the inferred joins turning more than ``max_angle`` degrees, without those joins."""
    result = []
    for trace in traces:
        gaps = sorted(trace.get("gaps", []))
        if not gaps or max_angle >= 180:
            result.append(trace)
            continue
        points = np.asarray(trace["points"], dtype=np.float64)
        arc = arclength(points)
        cuts = []
        for low, high in gaps:
            angle = join_angle(points, arc, low, high)
            if angle is not None and angle > max_angle:
                cuts.append((low, high))
        if not cuts:
            result.append(trace)
            continue
        for start, end in zip([0] + [high for _, high in cuts], [low for low, _ in cuts] + [len(points) - 1]):
            if end > start:
                kept = [[low - start, high - start] for low, high in gaps if start <= low and high <= end]
                result.append(dict(trace, points=points[start:end + 1], gaps=kept))
    return result


def simplify(points: np.ndarray, pins: Sequence[int] = (), tolerance: float = TOLERANCE,
             length_loss: float = LENGTH_LOSS) -> np.ndarray:
    """Indices of the points kept: the ends, ``pins`` and as few others as keep
    every point within ``tolerance``, in order along each kept segment, and
    each segment at most ``length_loss`` shorter than the stretch it replaces."""
    p = np.asarray(points, dtype=np.float64)
    n = len(p)
    keep = np.zeros(n, dtype=bool)
    keep[[0, n - 1]] = True
    keep[list(pins)] = True
    if n <= 2:
        return np.flatnonzero(keep)
    arc = arclength(p)
    anchors = np.flatnonzero(keep)
    stack = list(zip(anchors[:-1].tolist(), anchors[1:].tolist()))
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        d = p[b] - p[a]
        squared = float(d @ d)
        q = p[a + 1:b] - p[a]
        t = q @ d / squared if squared > 0 else np.zeros(len(q))
        backwards = np.flatnonzero(t < np.r_[0.0, t[:-1]] - 1e-10)
        error = q - np.clip(t, 0.0, 1.0)[:, None] * d
        distances = np.einsum("ij,ij->i", error, error)
        worst = int(np.argmax(distances))
        original = arc[b] - arc[a]
        too_short = original - np.sqrt(squared) > length_loss * original + 1e-10
        if distances[worst] <= tolerance**2 and not len(backwards) and not too_short:
            continue
        if distances[worst] > tolerance**2:
            split = a + 1 + worst
        elif len(backwards):
            split = a + backwards[0] if backwards[0] > 0 else a + 1
        else:
            split = a + 1 + int(np.searchsorted(arc[a + 1:b], (arc[a] + arc[b]) / 2))
        split = min(max(split, a + 1), b - 1)
        keep[split] = True
        stack += [(a, split), (split, b)]
    return np.flatnonzero(keep)


def thin(trace: dict[str, Any]) -> dict[str, Any]:
    """``trace`` with fewer points; the ends of its joins are kept."""
    points = np.asarray(trace["points"], dtype=np.float64)
    gaps = trace.get("gaps", [])
    kept = simplify(points, [i for gap in gaps for i in gap])
    return dict(trace, points=points[kept],
                gaps=[[int(np.searchsorted(kept, low)), int(np.searchsorted(kept, high))] for low, high in gaps])


class ExteriorBlack:
    """Where the CT is exactly zero outside the papyrus, on cubes of ``CELL`` voxels.

    The cubes without any non-zero voxel that connect to the border of the
    region are the outside; the cubes holding a zero voxel next to them are
    black too. Enclosed holes and isolated zero voxels are ignored.
    """

    def __init__(self, low_xyz: Sequence[int], high_xyz: Sequence[int]):
        self.low = np.asarray(low_xyz, dtype=int)
        shape = tuple(int(n) for n in (-(-(np.asarray(high_xyz, dtype=int) - self.low) // CELL))[::-1])
        self.any_zero = np.zeros(shape, dtype=bool)
        self.all_zero = np.ones(shape, dtype=bool)

    def add(self, ct: np.ndarray, low_xyz: Sequence[int]) -> None:
        """Record a ZYX region of the CT starting at ``low_xyz``; regions may overlap."""
        zero = np.asarray(ct) == 0
        start = (np.asarray(low_xyz, dtype=int) - self.low)[::-1]
        cuts = [np.unique(np.r_[0, np.arange(-s % CELL, n, CELL)]) for s, n in zip(start, zero.shape)]
        cells = tuple(slice(s // CELL, s // CELL + len(c)) for s, c in zip(start, cuts))
        for grid, ufunc in ((self.any_zero, np.logical_or), (self.all_zero, np.logical_and)):
            reduced = zero
            for axis, c in enumerate(cuts):
                reduced = ufunc.reduceat(reduced, c, axis=axis)
            grid[cells] = ufunc(grid[cells], reduced)

    def near(self, traces: Sequence[dict[str, Any]], distance: float) -> np.ndarray:
        """Whether each fiber has a point within ``distance`` voxels of a black cube's centre, or inside one."""
        from scipy.ndimage import binary_dilation, binary_erosion, binary_fill_holes
        from scipy.spatial import cKDTree

        result = np.zeros(len(traces), dtype=bool)
        cube = np.ones((3, 3, 3), dtype=bool)
        black = binary_dilation(~binary_fill_holes(~self.all_zero), cube) & self.any_zero
        if not len(traces) or not black.any():
            return result
        # The nearest black cube to a point outside the black is on its surface.
        surface = black & ~binary_erosion(black, cube)
        tree = cKDTree((np.argwhere(surface)[:, ::-1] + 0.5) * CELL + self.low)
        points = np.concatenate([np.asarray(t["points"], dtype=np.float64) for t in traces])
        owner = np.repeat(np.arange(len(traces)), [len(t["points"]) for t in traces])
        hit = np.isfinite(tree.query(points, distance_upper_bound=distance + 1e-9)[0])
        cell = np.floor((points - self.low) / CELL).astype(int)[:, ::-1]
        inside = np.all((cell >= 0) & (cell < black.shape), axis=1)
        hit[inside] |= black[tuple(cell[inside].T)]
        result[owner[hit]] = True
        return result


def clean(traces: Sequence[dict[str, Any]], max_angle: float, min_length: float,
          black: ExteriorBlack | None = None, black_distance: float = 0.0) -> list[dict[str, Any]]:
    traces = cut_sharp_joins(traces, max_angle)
    if black is not None and black_distance > 0:
        traces = [t for t, near in zip(traces, black.near(traces, black_distance)) if not near]
    traces = [t for t in traces if arclength(np.asarray(t["points"], dtype=np.float64))[-1] >= min_length]
    return [thin(t) for t in traces]
