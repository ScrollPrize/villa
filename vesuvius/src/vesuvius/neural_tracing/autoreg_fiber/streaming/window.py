"""Sliding 128^3 window reader on top of a :class:`ChunkLRUCache`.

The streaming tracer advances point-by-point through a large remote volume,
re-anchoring the 128^3 window when the predicted leading edge gets too close
to a face. This module handles:

  * anchoring (clamping to volume bounds, padding for out-of-volume regions),
  * chunk-aware reads that share native zarr chunks with the LRU cache,
  * the trigger condition for re-anchoring,
  * fire-and-forget prefetch of the chunks for a *next* anchor (so the next
    fetch_crop overlaps S3 latency with model inference on the current crop),
  * world<->local coordinate conversion.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from vesuvius.neural_tracing.autoreg_fiber.streaming.chunk_cache import ChunkLRUCache


def _iter_chunk_indices(
    min_corner: np.ndarray,
    crop_size: tuple[int, int, int],
    chunk_shape: tuple[int, int, int],
    volume_shape: tuple[int, int, int],
) -> Iterable[tuple[int, int, int]]:
    """Yield every chunk index that overlaps the window
    ``[min_corner, min_corner+crop_size)``, clipped to volume bounds."""

    end = min_corner + np.array(crop_size, dtype=np.int64)
    zlo, ylo, xlo = (int(v) for v in min_corner)
    zhi, yhi, xhi = (int(v) for v in end)
    cz, cy, cx = chunk_shape
    vz, vy, vx = volume_shape
    z_start, y_start, x_start = max(0, zlo) // cz, max(0, ylo) // cy, max(0, xlo) // cx
    z_end = max(0, min(zhi - 1, vz - 1)) // cz
    y_end = max(0, min(yhi - 1, vy - 1)) // cy
    x_end = max(0, min(xhi - 1, vx - 1)) // cx
    for zi in range(z_start, z_end + 1):
        for yi in range(y_start, y_end + 1):
            for xi in range(x_start, x_end + 1):
                yield (zi, yi, xi)


class WindowedVolumeReader:
    """Slides a fixed-size 3D window over a chunk-cached volume.

    Parameters
    ----------
    cache:
        A :class:`ChunkLRUCache` wrapping the source zarr array.
    crop_size:
        The window size, ``(D, H, W)``. Defaults to ``(128, 128, 128)``.
    reanchor_margin:
        How close (in voxels) the leading point may get to any face before
        :meth:`needs_reanchor` returns True. Defaults to 24 (= three patches
        at the autoreg_fiber default ``patch_size=8``).
    """

    def __init__(
        self,
        cache: ChunkLRUCache,
        *,
        crop_size: tuple[int, int, int] = (128, 128, 128),
        reanchor_margin: int = 24,
    ) -> None:
        self.cache = cache
        self.crop_size = tuple(int(v) for v in crop_size)
        self.reanchor_margin = int(reanchor_margin)
        self.chunk_shape = cache.chunk_shape
        self.volume_shape = cache.volume_shape
        if len(self.crop_size) != 3 or len(self.volume_shape) != 3:
            raise ValueError("WindowedVolumeReader currently supports 3D volumes only")
        self.min_corner = np.zeros(3, dtype=np.int64)

    # --- anchor management --------------------------------------------- #

    def anchor_on(self, world_zyx: np.ndarray | tuple[float, float, float]) -> np.ndarray:
        """Center the window on ``world_zyx`` (clamped to volume bounds).

        Returns the new ``min_corner``. The anchor is rounded to the nearest
        integer voxel.
        """

        center = np.asarray(world_zyx, dtype=np.float64).reshape(3)
        half = np.array(self.crop_size, dtype=np.float64) / 2.0
        corner = np.floor(center - half + 0.5).astype(np.int64)
        max_corner = np.array(self.volume_shape, dtype=np.int64) - np.array(self.crop_size, dtype=np.int64)
        max_corner = np.maximum(max_corner, 0)
        corner = np.clip(corner, 0, max_corner)
        self.min_corner = corner
        return self.min_corner.copy()

    def needs_reanchor(self, local_xyz: np.ndarray | tuple[float, float, float]) -> bool:
        """True if ``local_xyz`` is within ``reanchor_margin`` of a face."""

        local = np.asarray(local_xyz, dtype=np.float64).reshape(3)
        crop = np.array(self.crop_size, dtype=np.float64)
        margin = float(self.reanchor_margin)
        return bool(((local < margin) | (local > crop - margin)).any())

    def world_to_local(self, world_zyx: np.ndarray) -> np.ndarray:
        return np.asarray(world_zyx, dtype=np.float32) - self.min_corner.astype(np.float32)

    def local_to_world(self, local_zyx: np.ndarray) -> np.ndarray:
        return np.asarray(local_zyx, dtype=np.float32) + self.min_corner.astype(np.float32)

    def in_volume_bounds(self, world_zyx: np.ndarray | tuple[float, float, float]) -> bool:
        """True if the integer-rounded world coordinate is inside the volume."""

        rounded = np.floor(np.asarray(world_zyx, dtype=np.float64) + 0.5).astype(np.int64)
        return bool(
            (rounded >= 0).all() and (rounded < np.array(self.volume_shape, dtype=np.int64)).all()
        )

    # --- chunk reads --------------------------------------------------- #

    def fetch_crop(self) -> np.ndarray:
        """Return the current 128^3 window as a contiguous ``float32`` array.

        Out-of-volume regions (only possible if the window cannot be fully
        clamped — i.e. the volume is smaller than the window in some axis)
        are zero-padded.
        """

        return self._read_window(self.min_corner)

    def prefetch_anchor(self, world_zyx: np.ndarray | tuple[float, float, float]) -> None:
        """Asynchronously fetch every chunk that would be needed if the next
        anchor were ``world_zyx``. The actual fetch will overlap with whatever
        the caller does next."""

        center = np.asarray(world_zyx, dtype=np.float64).reshape(3)
        half = np.array(self.crop_size, dtype=np.float64) / 2.0
        corner = np.floor(center - half + 0.5).astype(np.int64)
        max_corner = np.array(self.volume_shape, dtype=np.int64) - np.array(self.crop_size, dtype=np.int64)
        max_corner = np.maximum(max_corner, 0)
        corner = np.clip(corner, 0, max_corner)
        for cidx in _iter_chunk_indices(corner, self.crop_size, self.chunk_shape, self.volume_shape):
            self.cache.prefetch(cidx)

    def _read_window(self, min_corner: np.ndarray) -> np.ndarray:
        out = np.zeros(self.crop_size, dtype=np.float32)
        crop = np.array(self.crop_size, dtype=np.int64)
        window_end = min_corner + crop
        vshape = np.array(self.volume_shape, dtype=np.int64)
        cshape = np.array(self.chunk_shape, dtype=np.int64)
        for cidx in _iter_chunk_indices(min_corner, self.crop_size, self.chunk_shape, self.volume_shape):
            chunk = self.cache.get(cidx)
            chunk_origin = np.array(cidx, dtype=np.int64) * cshape
            chunk_end = np.minimum(chunk_origin + cshape, vshape)
            inter_start = np.maximum(chunk_origin, min_corner)
            inter_end = np.minimum(chunk_end, window_end)
            if not (inter_end > inter_start).all():
                continue
            chunk_off = inter_start - chunk_origin
            win_off = inter_start - min_corner
            size = inter_end - inter_start
            out[
                win_off[0] : win_off[0] + size[0],
                win_off[1] : win_off[1] + size[1],
                win_off[2] : win_off[2] + size[2],
            ] = chunk[
                chunk_off[0] : chunk_off[0] + size[0],
                chunk_off[1] : chunk_off[1] + size[1],
                chunk_off[2] : chunk_off[2] + size[2],
            ].astype(np.float32, copy=False)
        return out


__all__ = ["WindowedVolumeReader"]
