"""Chunk-based slicing utilities extracted from the legacy dataset pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from vesuvius.utils.utils import pad_or_crop_2d, pad_or_crop_3d

from ..find_valid_patches import (
    find_valid_patches,
    bounding_box_volume,
    compute_bounding_box_3d,
)
from ..save_valid_patches import load_cached_patches, save_valid_patches

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChunkSliceConfig:
    """Configuration controlling volumetric chunk slicing."""

    patch_size: Tuple[int, ...]
    stride: Optional[Tuple[int, ...]]
    min_labeled_ratio: float
    min_bbox_percent: float
    allow_unlabeled: bool
    downsample_level: int
    num_workers: int
    cache_enabled: bool
    cache_dir: Optional[Path]


@dataclass
class ChunkVolume:
    """Container describing a volume and associated labels for chunk slicing."""

    index: int
    name: str
    image: object
    labels: Dict[str, Optional[object]]
    label_source: Optional[object]
    cache_key_path: Optional[Path]


@dataclass
class ChunkPatch:
    """Metadata describing a chunk patch to extract."""

    volume_index: int
    volume_name: str
    position: Tuple[int, ...]
    patch_size: Tuple[int, ...]
    weight: Optional[float] = None


@dataclass
class ChunkResult:
    """Image/label payload extracted for a chunk patch."""

    image: np.ndarray
    labels: Dict[str, np.ndarray]
    is_unlabeled: bool
    patch_info: Dict[str, object]


class ChunkSlicer:
    """Utility that enumerates and extracts volumetric chunk patches."""

    def __init__(self, config: ChunkSliceConfig, target_names: Sequence[str]) -> None:
        if not config.patch_size:
            raise ValueError("ChunkSlicer requires a non-empty patch_size")
        if len(config.patch_size) not in (2, 3):
            raise ValueError(
                f"ChunkSlicer patch_size must have length 2 or 3; got {config.patch_size}"
            )

        if not target_names:
            raise ValueError("ChunkSlicer requires at least one target name")

        self.config = config
        self.target_names = list(target_names)
        self._volumes: List[ChunkVolume] = []
        self._labeled_indices: List[int] = []
        self._patches: List[ChunkPatch] = []
        self._weights: Optional[List[float]] = None
        self.normalizer = None

        self._is_2d = len(config.patch_size) == 2

    # Registration ---------------------------------------------------------------------------------

    def register_volume(self, volume: ChunkVolume) -> None:
        """Register a volume for subsequent chunk slicing."""

        expected_index = len(self._volumes)
        if volume.index != expected_index:
            raise ValueError(
                f"ChunkVolume index {volume.index} does not match expected {expected_index}"
            )

        missing_targets = [t for t in self.target_names if t not in volume.labels]
        if missing_targets:
            raise ValueError(f"ChunkVolume missing labels for targets: {missing_targets}")

        self._volumes.append(volume)
        has_labels = any(label is not None for label in volume.labels.values())
        if volume.label_source is not None or has_labels:
            self._labeled_indices.append(volume.index)

    def register_volumes(self, volumes: Iterable[ChunkVolume]) -> None:
        for volume in volumes:
            self.register_volume(volume)

    def set_normalizer(self, normalizer) -> None:
        self.normalizer = normalizer

    # Index construction ----------------------------------------------------------------------------

    def build_index(self, *, validate: bool) -> Tuple[List[ChunkPatch], Optional[List[float]]]:
        if not self._volumes:
            raise RuntimeError("ChunkSlicer.build_index called before any volumes were registered")

        if validate and not self._labeled_indices:
            if not self.config.allow_unlabeled:
                raise RuntimeError(
                    "Chunk slice validation requested but no labeled volumes are registered"
                )
            logger.info(
                "ChunkSlicer: validation disabled because volumes lack labels; enumerating all positions"
            )
            validate = False

        patches: List[ChunkPatch] = []

        raw_valid_entries: Optional[List[Dict[str, object]]] = None

        if validate:
            labeled_volumes = [self._volumes[idx] for idx in self._labeled_indices]
            label_arrays = [vol.label_source for vol in labeled_volumes]
            label_names = [vol.name for vol in labeled_volumes]

            cache_paths = [vol.cache_key_path for vol in labeled_volumes]
            raw_valid_entries: Optional[List[Dict[str, object]]] = None

            labeled_positions: List[Tuple[int, Tuple[int, ...]]] = []
            raw_valid_entries = []

            if self.config.num_workers <= 0:
                labeled_positions, raw_valid_entries = self._compute_valid_positions_sequential(
                    labeled_volumes
                )
            else:
                cache_supported = self.config.cache_enabled and all(cache_paths)

                cached: Optional[List[Dict[str, object]]] = None
                if cache_supported:
                    cache_list = [Path(p) for p in cache_paths]  # type: ignore[arg-type]
                    cached = load_cached_patches(
                        train_data_paths=cache_list,
                        label_paths=cache_list,
                        patch_size=tuple(self.config.patch_size),
                        min_labeled_ratio=self.config.min_labeled_ratio,
                        bbox_threshold=self.config.min_bbox_percent,
                        downsample_level=self.config.downsample_level,
                        cache_path=str(self.config.cache_dir) if self.config.cache_dir else None,
                    )

                if cached is not None:
                    logger.info("ChunkSlicer: loaded %s patches from cache", len(cached))
                    for entry in cached:
                        labeled_idx = int(entry['volume_index'])
                        if labeled_idx >= len(labeled_volumes):
                            raise RuntimeError(
                                f"Cached patch references volume index {labeled_idx} which is unavailable"
                            )
                        position = tuple(int(v) for v in entry['position'])
                        patches.append(
                            ChunkPatch(
                                volume_index=labeled_volumes[labeled_idx].index,
                                volume_name=labeled_volumes[labeled_idx].name,
                                position=position,
                                patch_size=tuple(self.config.patch_size),
                            )
                        )
                else:
                    labeled_positions, raw_valid_entries = self._compute_valid_positions(
                        label_arrays, label_names
                    )

                    if cache_supported and raw_valid_entries:
                        cache_list = [Path(p) for p in cache_paths]  # type: ignore[arg-type]
                        save_valid_patches(
                            valid_patches=[
                                {
                                    "volume_idx": int(entry['volume_idx']),
                                    "volume_name": str(entry['volume_name']),
                                    "start_pos": list(entry['start_pos']),
                                }
                                for entry in raw_valid_entries
                            ],
                            train_data_paths=cache_list,
                            label_paths=cache_list,
                            patch_size=tuple(self.config.patch_size),
                            min_labeled_ratio=self.config.min_labeled_ratio,
                            bbox_threshold=self.config.min_bbox_percent,
                            downsample_level=self.config.downsample_level,
                            cache_path=str(self.config.cache_dir) if self.config.cache_dir else None,
                        )

            for labeled_idx, position in labeled_positions:
                volume = labeled_volumes[labeled_idx]
                patches.append(
                    ChunkPatch(
                        volume_index=volume.index,
                        volume_name=volume.name,
                        position=position,
                        patch_size=tuple(self.config.patch_size),
                    )
                )

        if not validate or any(v.label_source is None for v in self._volumes):
            for volume in self._volumes:
                if not validate or volume.label_source is None:
                    patches.extend(self.enumerate(volume, stride=self.config.stride))

        if not patches:
            raise RuntimeError("Chunk slicing produced zero patches across all volumes")

        self._patches = patches
        self._weights = None
        return patches, self._weights

    def _compute_valid_positions(
        self,
        label_arrays: Sequence[object],
        label_names: Sequence[str],
    ) -> Tuple[List[Tuple[int, Tuple[int, ...]]], List[Dict[str, object]]]:
        """Run find_valid_patches and map results to labeled volume indices."""

        valid = find_valid_patches(
            label_arrays=label_arrays,
            label_names=label_names,
            patch_size=tuple(self.config.patch_size),
            bbox_threshold=self.config.min_bbox_percent,
            label_threshold=self.config.min_labeled_ratio,
            num_workers=self.config.num_workers,
            downsample_level=self.config.downsample_level,
        )

        positions: List[Tuple[int, Tuple[int, ...]]] = []
        for entry in valid:
            labeled_idx = int(entry['volume_idx'])
            start_pos = tuple(int(v) for v in entry['start_pos'])
            positions.append((labeled_idx, start_pos))
        return positions, valid

    def _compute_valid_positions_sequential(
        self, labeled_volumes: Sequence[ChunkVolume]
    ) -> Tuple[List[Tuple[int, Tuple[int, ...]]], List[Dict[str, object]]]:
        positions: List[Tuple[int, Tuple[int, ...]]] = []
        entries: List[Dict[str, object]] = []

        for labeled_idx, volume in enumerate(labeled_volumes):
            label_array = volume.label_source
            if label_array is None:
                continue

            candidate_patches = self.enumerate(volume, stride=self.config.stride)
            for candidate in candidate_patches:
                mask_patch = self._extract_label_patch(label_array, candidate.position)
                if mask_patch is None:
                    continue

                mask = np.asarray(mask_patch)
                if mask.ndim > len(candidate.patch_size):
                    mask = mask[0]
                mask = mask.astype(bool, copy=False)
                if not mask.any():
                    continue

                bbox = compute_bounding_box_3d(mask)
                if bbox is None:
                    continue

                bb_vol = bounding_box_volume(bbox)
                patch_vol = mask.size
                if patch_vol == 0:
                    continue

                if (bb_vol / patch_vol) < self.config.min_bbox_percent:
                    continue

                labeled_ratio = np.count_nonzero(mask) / patch_vol
                if labeled_ratio < self.config.min_labeled_ratio:
                    continue

                position = tuple(int(v) for v in candidate.position)
                positions.append((labeled_idx, position))
                entries.append(
                    {
                        'volume_idx': labeled_idx,
                        'volume_name': volume.name,
                        'start_pos': list(position),
                    }
                )

        return positions, entries

    @property
    def patches(self) -> List[ChunkPatch]:
        return list(self._patches)

    @property
    def weights(self) -> Optional[List[float]]:
        if self._weights is None:
            return None
        return list(self._weights)

    # Enumeration -----------------------------------------------------------------------------------

    def enumerate(self, volume: ChunkVolume, stride: Optional[Tuple[int, ...]] = None) -> List[ChunkPatch]:
        stride_values = self._resolve_stride(stride)
        spatial_shape = self._extract_spatial_shape(volume.image)

        if self._is_2d:
            if len(spatial_shape) != 2:
                raise ValueError(
                    f"Chunk volume '{volume.name}' expected 2D spatial shape but found {spatial_shape}"
                )
            positions = self._iter_2d_positions(spatial_shape, stride_values)
        else:
            if len(spatial_shape) != 3:
                raise ValueError(
                    f"Chunk volume '{volume.name}' expected 3D spatial shape but found {spatial_shape}"
                )
            positions = self._iter_3d_positions(spatial_shape, stride_values)

        patches: List[ChunkPatch] = []
        for coords in positions:
            patches.append(
                ChunkPatch(
                    volume_index=volume.index,
                    volume_name=volume.name,
                    position=coords,
                    patch_size=tuple(self.config.patch_size),
                )
            )
        return patches

    # Extraction ------------------------------------------------------------------------------------

    def extract(self, patch: ChunkPatch, normalizer=None) -> ChunkResult:
        volume = self._get_volume(patch.volume_index)
        normalizer = normalizer if normalizer is not None else self.normalizer

        image_patch = self._extract_image_patch(volume.image, patch.position)
        if normalizer is not None:
            image_patch = normalizer.run(image_patch)
        else:
            image_patch = image_patch.astype(np.float32, copy=False)
        image_tensor = np.ascontiguousarray(image_patch[np.newaxis, ...], dtype=np.float32)

        labels: Dict[str, np.ndarray] = {}
        is_unlabeled = True
        for target_name in self.target_names:
            label_arr = volume.labels.get(target_name)
            label_patch = self._extract_label_patch(label_arr, patch.position)
            if label_patch is None:
                label_patch = np.zeros_like(image_patch, dtype=np.float32)
            else:
                label_patch = label_patch.astype(np.float32, copy=False)
                if np.any(np.abs(label_patch) > 0):
                    is_unlabeled = False

            if label_patch.ndim == image_patch.ndim:
                label_tensor = np.ascontiguousarray(label_patch[np.newaxis, ...], dtype=np.float32)
            elif label_patch.ndim == image_patch.ndim + 1:
                label_tensor = np.ascontiguousarray(label_patch, dtype=np.float32)
            else:
                raise ValueError(
                    f"Label array for target '{target_name}' has unexpected ndim {label_patch.ndim}"
                )

            labels[target_name] = label_tensor

        patch_info = {
            'plane': 'volume',
            'slice_index': -1,
            'position': list(int(v) for v in patch.position),
            'patch_size': list(int(v) for v in patch.patch_size),
            'angles': {
                'yaw_rad': 0.0,
                'tilt_x_rad': 0.0,
                'tilt_y_rad': 0.0,
                'tilt_z_rad': 0.0,
            },
            'volume_name': volume.name,
        }

        return ChunkResult(
            image=image_tensor.astype(np.float32, copy=False),
            labels=labels,
            is_unlabeled=is_unlabeled,
            patch_info=patch_info,
        )

    # Internal helpers ------------------------------------------------------------------------------

    def _resolve_stride(self, stride: Optional[Tuple[int, ...]]) -> Tuple[int, ...]:
        if stride is not None:
            return tuple(int(v) for v in stride)
        if self.config.stride is not None:
            return tuple(int(v) for v in self.config.stride)
        return tuple(int(v) for v in self.config.patch_size)

    def _extract_spatial_shape(self, array: np.ndarray) -> Tuple[int, ...]:
        if hasattr(array, 'spatial_shape'):
            return tuple(int(v) for v in array.spatial_shape)  # type: ignore[attr-defined]

        arr = np.asarray(array)
        if self._is_2d:
            if arr.ndim == 2:
                return tuple(int(v) for v in arr.shape)
            if arr.ndim == 3:
                return tuple(int(v) for v in arr.shape[-2:])
            raise ValueError(f"Unsupported 2D array ndim {arr.ndim} for chunk slicing")
        else:
            if arr.ndim == 3:
                return tuple(int(v) for v in arr.shape)
            if arr.ndim == 4:
                return tuple(int(v) for v in arr.shape[-3:])
            raise ValueError(f"Unsupported 3D array ndim {arr.ndim} for chunk slicing")

    def _iter_2d_positions(self, spatial_shape: Tuple[int, int], stride: Tuple[int, ...]) -> List[Tuple[int, int]]:
        height, width = spatial_shape
        if len(stride) != 2:
            raise ValueError(f"Stride for 2D chunks must have length 2; got {stride}")
        ph, pw = self.config.patch_size[-2:]
        sh, sw = stride

        if ph <= 0 or pw <= 0:
            raise ValueError("Patch dimensions must be positive")
        if height <= 0 or width <= 0:
            return []

        y_positions = list(range(0, max(1, height - ph + 1), max(1, sh)))
        x_positions = list(range(0, max(1, width - pw + 1), max(1, sw)))

        if y_positions and (y_positions[-1] + ph < height):
            y_positions.append(height - ph)
        if x_positions and (x_positions[-1] + pw < width):
            x_positions.append(width - pw)

        return [(y, x) for y in y_positions for x in x_positions]

    def _iter_3d_positions(
        self,
        spatial_shape: Tuple[int, int, int],
        stride: Tuple[int, ...],
    ) -> List[Tuple[int, int, int]]:
        depth, height, width = spatial_shape
        if len(stride) != 3:
            raise ValueError(f"Stride for 3D chunks must have length 3; got {stride}")
        pd, ph, pw = self.config.patch_size
        sd, sh, sw = stride

        if pd <= 0 or ph <= 0 or pw <= 0:
            raise ValueError("Patch dimensions must be positive")
        if depth <= 0 or height <= 0 or width <= 0:
            return []

        z_positions = list(range(0, max(1, depth - pd + 1), max(1, sd)))
        y_positions = list(range(0, max(1, height - ph + 1), max(1, sh)))
        x_positions = list(range(0, max(1, width - pw + 1), max(1, sw)))

        if z_positions and (z_positions[-1] + pd < depth):
            z_positions.append(depth - pd)
        if y_positions and (y_positions[-1] + ph < height):
            y_positions.append(height - ph)
        if x_positions and (x_positions[-1] + pw < width):
            x_positions.append(width - pw)

        return [(z, y, x) for z in z_positions for y in y_positions for x in x_positions]

    def _get_volume(self, index: int) -> ChunkVolume:
        try:
            return self._volumes[index]
        except IndexError as exc:
            raise IndexError(f"Chunk volume index {index} out of range") from exc

    def _extract_image_patch(self, image: np.ndarray, position: Tuple[int, ...]) -> np.ndarray:
        pos = tuple(int(v) for v in position)
        patch_size = tuple(int(v) for v in self.config.patch_size)

        if hasattr(image, 'read_window'):
            patch = image.read_window(pos, patch_size)
            arr = np.asarray(patch)
            return self._finalize_image_patch(arr, pos)

        arr = np.asarray(image)
        if self._is_2d:
            if len(pos) != 2:
                raise ValueError(f"2D chunk position must have two coordinates; got {position}")
            y, x = pos
            ph, pw = patch_size[-2:]
            if arr.ndim == 2:
                patch = arr[y : y + ph, x : x + pw]
                return pad_or_crop_2d(patch, (ph, pw)).astype(np.float32, copy=False)
            if arr.ndim == 3 and arr.shape[0] == 1:
                patch = arr[0, y : y + ph, x : x + pw]
                return pad_or_crop_2d(patch, (ph, pw)).astype(np.float32, copy=False)
            raise ValueError("2D chunk extraction expects image data with shape (H, W) or (1, H, W)")

        if len(pos) != 3:
            raise ValueError(f"3D chunk position must have three coordinates; got {position}")
        z, y, x = pos
        pd, ph, pw = patch_size

        if arr.ndim == 3:
            patch = arr[z : z + pd, y : y + ph, x : x + pw]
            return pad_or_crop_3d(patch, (pd, ph, pw)).astype(np.float32, copy=False)
        if arr.ndim == 4 and arr.shape[0] == 1:
            patch = arr[0, z : z + pd, y : y + ph, x : x + pw]
            return pad_or_crop_3d(patch, (pd, ph, pw)).astype(np.float32, copy=False)
        raise ValueError("3D chunk extraction expects image data with shape (D, H, W) or (1, D, H, W)")

    def _extract_label_patch(
        self,
        label_array: Optional[np.ndarray],
        position: Tuple[int, ...],
    ) -> Optional[np.ndarray]:
        if label_array is None:
            return None

        pos = tuple(int(v) for v in position)
        patch_size = tuple(int(v) for v in self.config.patch_size)

        if hasattr(label_array, 'read_window'):
            patch = label_array.read_window(pos, patch_size)
            arr = np.asarray(patch)
            return self._finalize_label_patch(arr, pos)

        arr = np.asarray(label_array)

        if self._is_2d:
            ph, pw = self.config.patch_size[-2:]
            if len(pos) != 2:
                raise ValueError(f"2D label position must have two coordinates; got {position}")
            y, x = pos
            if arr.ndim == 2:
                patch = arr[y : y + ph, x : x + pw]
                return pad_or_crop_2d(patch, (ph, pw)).astype(np.float32, copy=False)
            if arr.ndim == 3:
                channels = arr.shape[0]
                padded = [
                    pad_or_crop_2d(arr[c, y : y + ph, x : x + pw], (ph, pw)).astype(np.float32, copy=False)
                    for c in range(channels)
                ]
                return np.stack(padded, axis=0)
            raise ValueError(
                "2D chunk extraction expects label data with shape (H, W) or (C, H, W)"
            )

        # 3D case
        if len(pos) != 3:
            raise ValueError(f"3D label position must have three coordinates; got {position}")
        z, y, x = pos
        pd, ph, pw = self.config.patch_size

        if arr.ndim == 3:
            patch = arr[z : z + pd, y : y + ph, x : x + pw]
            return pad_or_crop_3d(patch, (pd, ph, pw)).astype(np.float32, copy=False)
        if arr.ndim == 4:
            channels = arr.shape[0]
            padded = [
                pad_or_crop_3d(arr[c, z : z + pd, y : y + ph, x : x + pw], (pd, ph, pw)).astype(np.float32, copy=False)
                for c in range(channels)
            ]
            return np.stack(padded, axis=0)
        raise ValueError("3D chunk extraction expects label data with shape (D, H, W) or (C, D, H, W)")

    def _finalize_image_patch(self, patch: np.ndarray, position: Tuple[int, ...]) -> np.ndarray:
        if self._is_2d:
            ph, pw = self.config.patch_size[-2:]
            if patch.ndim == 2:
                return pad_or_crop_2d(patch, (ph, pw)).astype(np.float32, copy=False)
            if patch.ndim == 3 and patch.shape[0] == 1:
                return pad_or_crop_2d(patch[0], (ph, pw)).astype(np.float32, copy=False)
            if patch.ndim == 3:
                channels = patch.shape[0]
                padded = [
                    pad_or_crop_2d(patch[c], (ph, pw)).astype(np.float32, copy=False)
                    for c in range(channels)
                ]
                return np.stack(padded, axis=0)
            raise ValueError("2D chunk extraction expects image data with shape (H, W) or (C, H, W)")

        pd, ph, pw = self.config.patch_size
        if patch.ndim == 3:
            return pad_or_crop_3d(patch, (pd, ph, pw)).astype(np.float32, copy=False)
        if patch.ndim == 4 and patch.shape[0] == 1:
            return pad_or_crop_3d(patch[0], (pd, ph, pw)).astype(np.float32, copy=False)
        if patch.ndim == 4:
            channels = patch.shape[0]
            padded = [
                pad_or_crop_3d(patch[c], (pd, ph, pw)).astype(np.float32, copy=False)
                for c in range(channels)
            ]
            return np.stack(padded, axis=0)
        raise ValueError("3D chunk extraction expects image data with shape (D, H, W) or (C, D, H, W)")

    def _finalize_label_patch(self, patch: np.ndarray, position: Tuple[int, ...]) -> np.ndarray:
        if self._is_2d:
            ph, pw = self.config.patch_size[-2:]
            if patch.ndim == 2:
                return pad_or_crop_2d(patch, (ph, pw)).astype(np.float32, copy=False)
            if patch.ndim == 3:
                channels = patch.shape[0]
                padded = [
                    pad_or_crop_2d(patch[c], (ph, pw)).astype(np.float32, copy=False)
                    for c in range(channels)
                ]
                return np.stack(padded, axis=0)
            raise ValueError("2D chunk extraction expects label data with shape (H, W) or (C, H, W)")

        pd, ph, pw = self.config.patch_size
        if patch.ndim == 3:
            return pad_or_crop_3d(patch, (pd, ph, pw)).astype(np.float32, copy=False)
        if patch.ndim == 4:
            channels = patch.shape[0]
            padded = [
                pad_or_crop_3d(patch[c], (pd, ph, pw)).astype(np.float32, copy=False)
                for c in range(channels)
            ]
            return np.stack(padded, axis=0)
        raise ValueError("3D chunk extraction expects label data with shape (D, H, W) or (C, D, H, W)")
