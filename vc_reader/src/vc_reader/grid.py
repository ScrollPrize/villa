"""Volume views through the native grid sampler: planes and tifxyz surfaces sampled
from pyramid levels with layers, composites, cache-only reads and pixel masks, and
progressive frames (drawn from the caches now, completed by background reads).

Levels come from :meth:`ZarrReader.grid_level` (a pyramid level, optionally with its
:meth:`ZarrReader.cuda` reader); the first level of a request is preferred and later
ones fill its holes. On the GPU, give the levels' readers one :class:`CudaPool` so
they share one device cache: ``r.grid_level(cuda=r.cuda(pool=pool))``. Points and geometry are XYZ in the working voxels (level 0 by
default); arrays are ZYX. Progressive frames read only the caches: open the volume
with a ``memory_cache`` (or a disk cache for remote volumes), else they stay empty.

A viewer's frame::

    levels = [group.level(i).grid_level() for i in range(len(group.levels))]
    start = grid.level_for_view(levels, zoom)
    use = levels[start:start + 1 + grid.fallback_levels(levels, start, w, h, zoom)]
    pending = grid.Pending()
    frame = grid.sample_grid_cached(use, plane=plane, pending=pending)  # draw frame.values
    grid.load(pending)  # e.g. on a worker thread; then sample again
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple, Sequence

import numpy as np

from . import _reader
from ._reader import CompositeOptions, GridLevel, fallback_levels, level_for_view
from .reader.cuda import CudaPool

__all__ = [
    "LEVEL_MISSING", "LEVEL_NONE", "CompositeOptions", "CudaPool", "GridLevel", "GridResult", "Pending",
    "fallback_levels", "level_for_view", "load", "pixels_per_voxel", "sample_grid", "sample_grid_cached",
    "voxel_extent",
]

LEVEL_NONE = -2     # no layer of the pixel had a point
LEVEL_MISSING = -1  # a layer's chunks were in no level's caches (cache-only reads)


class GridResult(NamedTuple):
    values: np.ndarray  # (layers, height, width) float32, NaN without a value; one layer with a composite
    level: np.ndarray   # (height, width) int8: index of the request level that supplied the pixel, or LEVEL_*
    cuda_failure: str   # why a GPU pass fell back to the CPU, else ""


class Pending:
    """The reads progressive frames lacked; it keeps their levels and surfaces alive."""

    def __init__(self) -> None:
        self._engine = _reader.GridPending()
        self._keep: list[Any] = []

    @property
    def empty(self) -> bool:
        return self._engine.empty

    @property
    def reads(self) -> tuple[int, int]:
        """(coarse, target) read counts."""
        return self._engine.coarse_reads, self._engine.target_reads

    def clear(self) -> None:
        self._engine.clear()
        self._keep.clear()


def voxel_extent(level: GridLevel) -> tuple[float, float, float]:
    """Working voxels one index of the level spans, per ZYX axis."""
    return tuple(_reader.grid_voxel_extent(level))


def _raster(raster: Sequence[float] | None) -> tuple[float, ...] | None:
    if raster is None:
        return None
    raster = tuple(float(v) for v in raster)
    if len(raster) not in (6, 8):
        raise ValueError("raster is (x0, y0, sx, sy, width, height[, kx, ky])")
    return raster + (0.0, 0.0) if len(raster) == 6 else raster


def _plane(plane: Any) -> tuple | None:
    if plane is None:
        return None
    origin, dx, dy, normal, width, height = plane
    return tuple(tuple(float(v) for v in a) for a in (origin, dx, dy, normal)) + (int(width), int(height))


def pixels_per_voxel(surface: Any, raster: Sequence[float], *, interpolation: str = "bicubic") -> float:
    """View pixels per working voxel of a raster over a tifxyz surface; 0 without a point under it."""
    return _reader.pixels_per_voxel(surface, _raster(raster), interpolation)


def _composite(composite: str | CompositeOptions | None) -> CompositeOptions | None:
    return CompositeOptions(composite) if isinstance(composite, str) else composite


def _sample(levels, plane, surface, raster, interpolation, xyz, normals, order, offsets, cached_only, pixels,
            composite, pending):
    as_f32 = lambda a: None if a is None else np.ascontiguousarray(a, dtype=np.float32)
    values, level, failure = _reader.sample_grid(
        list(levels), _plane(plane), surface, _raster(raster), interpolation, as_f32(xyz), as_f32(normals),
        int(order), [float(o) for o in offsets], bool(cached_only),
        None if pixels is None else np.ascontiguousarray(pixels, dtype=np.uint8), _composite(composite),
        None if pending is None else pending._engine)
    return GridResult(values, level, failure)


def sample_grid(levels: Sequence[GridLevel], *, plane: Any = None, surface: Any = None,
                raster: Sequence[float] | None = None, interpolation: str = "bicubic", xyz: Any = None,
                normals: Any = None, order: int = 1, offsets: Sequence[float] = (0.0,), cached_only: bool = False,
                pixels: Any = None, composite: str | CompositeOptions | None = None) -> GridResult:
    """Sample levels over a plane ``(origin, dx, dy, normal, width, height)``, a tifxyz
    surface with a raster ``(x0, y0, sx, sy, width, height[, kx, ky])`` in grid columns and
    rows, or prepared ``xyz`` (H, W, 3) with optional ``normals``. Reads missing chunks
    unless ``cached_only``."""
    return _sample(levels, plane, surface, raster, interpolation, xyz, normals, order, offsets, cached_only, pixels,
                   composite, None)


def sample_grid_cached(levels: Sequence[GridLevel], *, plane: Any = None, surface: Any = None,
                       raster: Sequence[float] | None = None, interpolation: str = "bicubic", order: int = 1,
                       offsets: Sequence[float] = (0.0,), pixels: Any = None,
                       composite: str | CompositeOptions | None = None, pending: Pending | None = None) -> GridResult:
    """A progressive frame: levels[0] is the view's level, later ones fill holes; sampled
    from the caches alone. ``pending`` collects the reads the frame lacked."""
    if pending is not None:
        pending._keep.extend((list(levels), surface))
    return _sample(levels, plane, surface, raster, interpolation, None, None, order, offsets, True, pixels,
                   composite, pending)


def load(pending: Pending, after_part: Callable[[], bool] | None = None) -> str:
    """Run the pending reads: the coarse preview, then the view's level. ``after_part()``
    runs after each part that had reads; returning False stops. Returns the first GPU
    failure a read fell back from, else ""."""
    return _reader.load_grid(pending._engine, after_part)
