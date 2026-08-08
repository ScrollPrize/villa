from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import vc

PlaneGeometry = tuple[np.ndarray, np.ndarray, np.ndarray]


class VolumePlaneExtractor:
    """Fast, fused extraction of intersecting ray-aligned planes."""

    def __init__(
        self,
        volume_paths: list[Path],
        *,
        shape: tuple[int, int] = (256, 256),
        spacing: float = 1.0,
        num_planes: int = 2,
        sampling: str = "trilinear",
        tile_size: int = 32,
        cache_bytes: int = 0,
        io_threads: int = 0,
        segment_to_volume_xyz: list[np.ndarray] | None = None,
    ) -> None:
        if len(shape) != 2 or any(int(size) <= 0 for size in shape):
            raise ValueError("shape must contain positive [height, width]")
        if not math.isfinite(spacing) or spacing <= 0:
            raise ValueError("spacing must be finite and positive")
        if num_planes not in {2, 4}:
            raise ValueError("num_planes must be 2 or 4")
        if sampling not in {"nearest", "trilinear"}:
            raise ValueError("sampling must be 'nearest' or 'trilinear'")
        if tile_size <= 0:
            raise ValueError("tile_size must be positive")

        self.volume_paths = [Path(path) for path in volume_paths]
        self.shape = tuple(int(size) for size in shape)
        self.spacing = float(spacing)
        self.num_planes = int(num_planes)
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

    def intersecting_geometry(
        self,
        direction: np.ndarray,
        ray_origin: np.ndarray,
    ) -> PlaneGeometry:
        """Construct configured transverse planes intersecting along the ray."""
        direction = np.asarray(direction, dtype=np.float64)
        norm = np.linalg.norm(direction)
        if not math.isfinite(norm) or norm <= 0:
            raise ValueError("direction must be finite and nonzero")
        direction /= norm

        reference = np.zeros(3, dtype=np.float64)
        reference[int(np.argmin(np.abs(direction)))] = 1.0
        transverse_a = np.cross(reference, direction)
        transverse_a /= np.linalg.norm(transverse_a)
        transverse_b = np.cross(direction, transverse_a)

        transverse_axes = [transverse_a, transverse_b]
        if self.num_planes == 4:
            inverse_sqrt_two = 1.0 / math.sqrt(2.0)
            transverse_axes.extend(
                (
                    (transverse_a + transverse_b) * inverse_sqrt_two,
                    (transverse_a - transverse_b) * inverse_sqrt_two,
                )
            )

        # Both image axes retain the same physical voxel spacing. Increasing
        # ray length therefore increases image width rather than resampling.
        x_steps = self.spacing * np.broadcast_to(
            direction, (self.num_planes, direction.size)
        )
        y_steps = self.spacing * np.stack(transverse_axes)
        height, _ = self.shape
        ray_origin = np.asarray(ray_origin, dtype=np.float64)
        origins = np.broadcast_to(ray_origin, (self.num_planes, 3)).copy()
        origins -= 0.5 * (height - 1) * y_steps
        return tuple(
            np.ascontiguousarray(values, dtype=np.float32)
            for values in (origins, x_steps, y_steps)
        )

    def extract(
        self,
        volume_idx: int,
        direction: np.ndarray,
        ray_origin: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, PlaneGeometry]:
        geometry = self.intersecting_geometry(direction, ray_origin)
        sampling_geometry = self.sampling_geometry(volume_idx, geometry)
        images, valid, _ = self._volume(volume_idx).sample_planes(
            *sampling_geometry,
            self.shape,
            sampling=self.sampling,
            tile_size=self.tile_size,
        )
        return images, valid, geometry

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

    def sampling_geometry(
        self, volume_idx: int, geometry: PlaneGeometry
    ) -> PlaneGeometry:
        """Map display/segment plane geometry into volume-array XYZ."""
        transform = self.segment_to_volume_xyz[volume_idx]
        linear = transform[:3, :3]
        origins, x_steps, y_steps = geometry
        return (
            np.ascontiguousarray(
                origins @ linear.T + transform[:3, 3], dtype=np.float32
            ),
            np.ascontiguousarray(x_steps @ linear.T, dtype=np.float32),
            np.ascontiguousarray(y_steps @ linear.T, dtype=np.float32),
        )
