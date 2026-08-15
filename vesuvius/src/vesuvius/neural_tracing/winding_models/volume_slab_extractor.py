from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import vc


@dataclass(frozen=True)
class SlabFrame:
    """Right-handed ray-aligned frame for a sampled slab.

    Slab voxel (i, j, k) sits at ``origin + spacing * (i * axis_a + j *
    axis_b + k * direction)`` in segment/display space; the ray runs along
    the k axis through the transverse center, so k equals the ray's t in
    sample units.
    """

    origin: np.ndarray
    axis_a: np.ndarray
    axis_b: np.ndarray
    direction: np.ndarray
    spacing: float

    def to_slab(self, points_xyz: np.ndarray) -> np.ndarray:
        """Map segment-space XYZ points to fractional (i, j, k) slab indices."""
        offsets = np.asarray(points_xyz, dtype=np.float64) - self.origin
        axes = np.stack([self.axis_a, self.axis_b, self.direction], axis=-1)
        return offsets @ axes / self.spacing

    def to_world(self, points_ijk: np.ndarray) -> np.ndarray:
        """Map fractional (i, j, k) slab indices to segment-space XYZ."""
        axes = np.stack([self.axis_a, self.axis_b, self.direction])
        return self.origin + self.spacing * np.asarray(points_ijk, np.float64) @ axes


class VolumeSlabExtractor:
    """Fused extraction of a ray-aligned 3-D slab.

    The slab is transverse_size x transverse_size across the ray and
    ray_length along it, sampled at ``spacing`` voxel steps on every axis
    so the ray axis stays at full volume resolution.
    """

    def __init__(
        self,
        volume_paths: list[Path],
        *,
        transverse_size: int = 96,
        ray_length: int = 384,
        spacing: float = 1.0,
        sampling: str = "trilinear",
        tile_size: int = 32,
        cache_bytes: int = 0,
        io_threads: int = 0,
        segment_to_volume_xyz: list[np.ndarray] | None = None,
    ) -> None:
        if transverse_size < 2 or ray_length < 2:
            raise ValueError("transverse_size and ray_length must be at least 2")
        if not math.isfinite(spacing) or spacing <= 0:
            raise ValueError("spacing must be finite and positive")
        if sampling not in {"nearest", "trilinear"}:
            raise ValueError("sampling must be 'nearest' or 'trilinear'")
        if tile_size <= 0:
            raise ValueError("tile_size must be positive")

        self.volume_paths = [Path(path) for path in volume_paths]
        self.transverse_size = int(transverse_size)
        self.ray_length = int(ray_length)
        self.spacing = float(spacing)
        self.sampling = sampling
        self.tile_size = int(tile_size)
        self.cache_bytes = int(cache_bytes)
        self.io_threads = int(io_threads)
        if segment_to_volume_xyz is None:
            segment_to_volume_xyz = [np.eye(4) for _ in volume_paths]
        if len(segment_to_volume_xyz) != len(volume_paths):
            raise ValueError("one segment-to-volume transform is required per volume")
        self.segment_to_volume_xyz = [
            np.asarray(transform, dtype=np.float64)
            for transform in segment_to_volume_xyz
        ]
        if any(transform.shape != (4, 4) for transform in self.segment_to_volume_xyz):
            raise ValueError("segment-to-volume transforms must have shape [4, 4]")
        self._volume_handles: dict[tuple[int, int], vc.Volume] = {}

    @staticmethod
    def scaled_volume_path(path: Path, scale: int) -> Path:
        """Select a physical zarr array when ``path`` is a multiscale group."""
        scaled = path / str(scale)
        return scaled if (scaled / ".zarray").is_file() else path

    @staticmethod
    def load_segment_to_volume_transform(
        path: Path,
        scale: int,
        *,
        segment_downscale: int = 1,
        use_registration: bool = True,
    ) -> np.ndarray:
        """Load the fixed-segment to selected-volume-array XYZ transform."""
        target_downscale = 2**scale
        if segment_downscale <= 0:
            raise ValueError("segment_downscale must be positive")
        if not use_registration:
            transform = np.eye(4, dtype=np.float64)
            transform[:3, :3] *= segment_downscale / target_downscale
            return transform

        transform_path = path / "transform.json"
        if not transform_path.is_file():
            raise ValueError(f"registration transform is missing: {transform_path}")

        with transform_path.open() as transform_file:
            values = np.asarray(
                json.load(transform_file)["transformation_matrix"], dtype=np.float64
            )
        if values.shape != (3, 4):
            raise ValueError(
                f"expected a 3x4 transformation_matrix in {transform_path}"
            )

        # transform.json maps target-volume base voxels to source-volume base
        # voxels. Segment coordinates are expressed in source base voxels
        # divided by segment_downscale; the selected target array is base voxels
        # divided by target_downscale.
        inverse_linear = np.linalg.inv(values[:, :3])
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = inverse_linear * segment_downscale / target_downscale
        transform[:3, 3] = -inverse_linear @ values[:, 3] / target_downscale
        return transform

    def _volume(self, volume_idx: int) -> vc.Volume:
        # DataLoader workers must not inherit another process's cache and I/O
        # thread pool, so handles are deliberately opened lazily per process.
        key = (os.getpid(), volume_idx)
        volume = self._volume_handles.get(key)
        if volume is None:
            volume = vc.Volume.open(str(self.volume_paths[volume_idx]))
            if self.cache_bytes > 0:
                volume.set_cache_budget(self.cache_bytes)
            if self.io_threads > 0:
                volume.set_io_threads(self.io_threads)
            self._volume_handles[key] = volume
        return volume

    def slab_frame(
        self, direction: np.ndarray, ray_origin: np.ndarray
    ) -> SlabFrame:
        """Deterministic right-handed slab frame around the ray."""
        direction = np.asarray(direction, dtype=np.float64)
        norm = np.linalg.norm(direction)
        if not math.isfinite(norm) or norm <= 0:
            raise ValueError("direction must be finite and nonzero")
        direction = direction / norm

        reference = np.zeros(3, dtype=np.float64)
        reference[int(np.argmin(np.abs(direction)))] = 1.0
        axis_a = np.cross(reference, direction)
        axis_a /= np.linalg.norm(axis_a)
        axis_b = np.cross(direction, axis_a)

        origin = (
            np.asarray(ray_origin, dtype=np.float64)
            - 0.5 * (self.transverse_size - 1) * self.spacing * (axis_a + axis_b)
        )
        return SlabFrame(origin, axis_a, axis_b, direction, self.spacing)

    def extract(
        self,
        volume_idx: int,
        direction: np.ndarray,
        ray_origin: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, SlabFrame]:
        """Sample the slab image and validity, both [H, W, L], plus the frame.

        The slab is realized as H parallel planes of shape [W, L] offset
        along axis_a, so one fused sample_planes call owns all I/O.
        """
        frame = self.slab_frame(direction, ray_origin)
        size = self.transverse_size
        origins = frame.origin[None] + (
            self.spacing * np.arange(size, dtype=np.float64)[:, None]
            * frame.axis_a[None]
        )
        x_steps = np.broadcast_to(self.spacing * frame.direction, (size, 3))
        y_steps = np.broadcast_to(self.spacing * frame.axis_b, (size, 3))

        transform = self.segment_to_volume_xyz[volume_idx]
        linear = transform[:3, :3]
        images, valid, _ = self._volume(volume_idx).sample_planes(
            np.ascontiguousarray(
                origins @ linear.T + transform[:3, 3], dtype=np.float32
            ),
            np.ascontiguousarray(x_steps @ linear.T, dtype=np.float32),
            np.ascontiguousarray(y_steps @ linear.T, dtype=np.float32),
            (size, self.ray_length),
            sampling=self.sampling,
            tile_size=self.tile_size,
        )
        # Boolean validity: downstream boolean masking must never degrade
        # into integer fancy indexing.
        return np.asarray(images), np.asarray(valid, dtype=bool), frame

    def sample_points(self, volume_idx: int, points_xyz: np.ndarray) -> np.ndarray:
        """Sample volume intensities at display/segment-space XYZ points."""
        transform = self.segment_to_volume_xyz[volume_idx]
        coords = np.asarray(points_xyz, dtype=np.float64) @ transform[:3, :3].T
        coords += transform[:3, 3]
        coords = np.ascontiguousarray(coords[np.newaxis], dtype=np.float32)
        values, _, _ = self._volume(volume_idx).sample_coords(
            coords,
            np.ones(coords.shape[:2], dtype=bool),
            sampling=self.sampling,
            tile_size=self.tile_size,
        )
        return np.asarray(values).reshape(-1)
