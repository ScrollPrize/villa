"""Chunk-based slicing utilities extracted from the legacy dataset pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

import numpy as np
from tqdm import tqdm

from vesuvius.utils.utils import pad_or_crop_2d, pad_or_crop_3d

from ..find_valid_patches import (
    find_valid_patches,
    bounding_box_volume,
    compute_bounding_box_3d,
    collapse_patch_to_spatial,
    zero_ignore_labels,
)
from ..save_valid_patches import load_cached_patches, save_valid_patches
from ..mesh.handles import MeshHandle

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChunkSliceConfig:
    """Configuration controlling volumetric chunk slicing."""

    patch_size: Tuple[int, ...]
    stride: Optional[Tuple[int, ...]]
    min_labeled_ratio: float
    min_bbox_percent: float
    allow_unlabeled: bool
    valid_patch_find_resolution: int
    num_workers: int
    cache_enabled: bool
    cache_dir: Optional[Path]
    label_channel_selector: Optional[Union[int, Tuple[int, ...]]] = None
    valid_patch_value: Optional[Union[int, float]] = None
    bg_sampling_enabled: bool = False
    bg_to_fg_ratio: float = 0.5
    # Unlabeled foreground detection for semi-supervised learning
    unlabeled_fg_enabled: bool = False
    unlabeled_fg_threshold: float = 0.05  # Min fraction of non-zero image voxels
    unlabeled_fg_bbox_threshold: float = 0.15  # Min bbox coverage for image data
    unlabeled_fg_volume_ids: Optional[Set[str]] = None  # Volume IDs to scan for unlabeled FG


@dataclass
class ChunkVolume:
    """Container describing a volume and associated labels for chunk slicing."""

    index: int
    name: str
    image: object
    labels: Dict[str, Optional[object]]
    label_source: Optional[object]
    cache_key_path: Optional[Path]
    label_ignore_value: Optional[Union[int, float]] = None
    meshes: Mapping[str, MeshHandle] = field(default_factory=dict)
    # Image source for unlabeled foreground validation (zarr array/group)
    image_source: Optional[object] = None


@dataclass
class ChunkPatch:
    """Metadata describing a chunk patch to extract."""

    volume_index: int
    volume_name: str
    position: Tuple[int, ...]
    patch_size: Tuple[int, ...]
    weight: Optional[float] = None
    is_bg_only: bool = False
    is_unlabeled_fg: bool = False  # Unlabeled but has image data (for semi-supervised)


@dataclass
class ChunkResult:
    """Image/label payload extracted for a chunk patch."""

    image: np.ndarray
    labels: Dict[str, np.ndarray]
    is_unlabeled: bool
    patch_info: Dict[str, object]
    meshes: Dict[str, Dict[str, object]] = field(default_factory=dict)


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
        self._label_channel_selector = config.label_channel_selector

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
            # Only disable validation if unlabeled_fg filtering is also disabled
            # When unlabeled_fg_enabled, we still want to filter patches by image content
            if not self.config.unlabeled_fg_enabled:
                logger.info(
                    "ChunkSlicer: validation disabled because volumes lack labels; enumerating all positions"
                )
                validate = False
            else:
                logger.info(
                    "ChunkSlicer: no labeled volumes, but unlabeled_fg_enabled=True; filtering by image content"
                )

        fg_patches: List[ChunkPatch] = []
        bg_patches: List[ChunkPatch] = []
        unlabeled_fg_patches: List[ChunkPatch] = []

        if validate:
            # When no labeled volumes but unlabeled_fg is enabled, use all volumes
            if not self._labeled_indices and self.config.unlabeled_fg_enabled:
                labeled_volumes = list(self._volumes)
                logger.info("Using all %d volumes for unlabeled FG filtering", len(labeled_volumes))
            else:
                labeled_volumes = [self._volumes[idx] for idx in self._labeled_indices]
            label_arrays = [vol.label_source for vol in labeled_volumes]
            label_names = [vol.name for vol in labeled_volumes]

            # Fall back to image path when no label path (for unlabeled datasets)
            cache_paths = []
            for vol in labeled_volumes:
                if vol.cache_key_path is not None:
                    cache_paths.append(vol.cache_key_path)
                elif hasattr(vol.image, 'path'):
                    cache_paths.append(Path(vol.image.path))
                elif vol.image_source is not None and hasattr(vol.image_source, 'path'):
                    cache_paths.append(Path(vol.image_source.path))
                else:
                    cache_paths.append(None)

            fg_positions: List[Tuple[int, Tuple[int, ...]]] = []
            bg_positions: List[Tuple[int, Tuple[int, ...]]] = []
            unlabeled_fg_positions: List[Tuple[int, Tuple[int, ...]]] = []
            raw_fg_entries: List[Dict[str, object]] = []
            raw_bg_entries: List[Dict[str, object]] = []

            if self.config.num_workers <= 0:
                (
                    fg_positions,
                    bg_positions,
                    raw_fg_entries,
                    raw_bg_entries,
                    unlabeled_fg_positions,
                    raw_unlabeled_fg_entries,
                ) = self._compute_valid_positions_sequential(
                    labeled_volumes
                )
            else:
                cache_supported = self.config.cache_enabled and all(cache_paths)

                cached = None
                if cache_supported:
                    cache_list = [Path(p) for p in cache_paths]  # type: ignore[arg-type]
                    cached = load_cached_patches(
                        train_data_paths=cache_list,
                        label_paths=cache_list,
                        patch_size=tuple(self.config.patch_size),
                        min_labeled_ratio=self.config.min_labeled_ratio,
                        bbox_threshold=self.config.min_bbox_percent,
                        valid_patch_find_resolution=self.config.valid_patch_find_resolution,
                        cache_path=str(self.config.cache_dir) if self.config.cache_dir else None,
                        valid_patch_value=self.config.valid_patch_value,
                        bg_sampling_enabled=self.config.bg_sampling_enabled,
                        bg_to_fg_ratio=self.config.bg_to_fg_ratio,
                        unlabeled_fg_enabled=self.config.unlabeled_fg_enabled,
                        unlabeled_fg_threshold=self.config.unlabeled_fg_threshold,
                        unlabeled_fg_bbox_threshold=self.config.unlabeled_fg_bbox_threshold,
                    )

                if cached is not None:
                    # Load FG patches from cache
                    logger.info("ChunkSlicer: loaded %s FG patches from cache", len(cached.get('fg_patches', [])))
                    for entry in cached.get('fg_patches', []):
                        labeled_idx = int(entry['volume_index'])
                        if labeled_idx >= len(labeled_volumes):
                            raise RuntimeError(
                                f"Cached patch references volume index {labeled_idx} which is unavailable"
                            )
                        position = tuple(int(v) for v in entry['position'])
                        fg_patches.append(
                            ChunkPatch(
                                volume_index=labeled_volumes[labeled_idx].index,
                                volume_name=labeled_volumes[labeled_idx].name,
                                position=position,
                                patch_size=tuple(self.config.patch_size),
                                is_bg_only=False,
                            )
                        )

                    # Load BG patches from cache
                    if cached.get('bg_patches'):
                        logger.info("ChunkSlicer: loaded %s BG patches from cache", len(cached['bg_patches']))
                        for entry in cached['bg_patches']:
                            labeled_idx = int(entry['volume_index'])
                            if labeled_idx >= len(labeled_volumes):
                                continue  # Skip invalid BG patch references
                            position = tuple(int(v) for v in entry['position'])
                            bg_patches.append(
                                ChunkPatch(
                                    volume_index=labeled_volumes[labeled_idx].index,
                                    volume_name=labeled_volumes[labeled_idx].name,
                                    position=position,
                                    patch_size=tuple(self.config.patch_size),
                                    is_bg_only=True,
                                )
                            )

                    # Load unlabeled FG patches from cache (for semi-supervised)
                    if cached.get('unlabeled_fg_patches'):
                        logger.info(
                            "ChunkSlicer: loaded %s unlabeled FG patches from cache",
                            len(cached['unlabeled_fg_patches']),
                        )
                        for entry in cached['unlabeled_fg_patches']:
                            labeled_idx = int(entry['volume_index'])
                            if labeled_idx >= len(labeled_volumes):
                                continue  # Skip invalid patch references
                            position = tuple(int(v) for v in entry['position'])
                            unlabeled_fg_patches.append(
                                ChunkPatch(
                                    volume_index=labeled_volumes[labeled_idx].index,
                                    volume_name=labeled_volumes[labeled_idx].name,
                                    position=position,
                                    patch_size=tuple(self.config.patch_size),
                                    is_bg_only=False,
                                    is_unlabeled_fg=True,
                                )
                            )
                else:
                    ignore_values = [vol.label_ignore_value for vol in labeled_volumes]
                    (
                        fg_positions,
                        bg_positions,
                        raw_fg_entries,
                        raw_bg_entries,
                        unlabeled_fg_positions,
                        raw_unlabeled_fg_entries,
                    ) = self._compute_valid_positions(
                        labeled_volumes, label_arrays, label_names, ignore_values
                    )

                    if cache_supported and (raw_fg_entries or raw_bg_entries or raw_unlabeled_fg_entries):
                        cache_list = [Path(p) for p in cache_paths]  # type: ignore[arg-type]
                        save_valid_patches(
                            valid_patches=[
                                {
                                    "volume_idx": int(entry['volume_idx']),
                                    "volume_name": str(entry['volume_name']),
                                    "start_pos": list(entry['start_pos']),
                                }
                                for entry in raw_fg_entries
                            ],
                            train_data_paths=cache_list,
                            label_paths=cache_list,
                            patch_size=tuple(self.config.patch_size),
                            min_labeled_ratio=self.config.min_labeled_ratio,
                            bbox_threshold=self.config.min_bbox_percent,
                            valid_patch_find_resolution=self.config.valid_patch_find_resolution,
                            cache_path=str(self.config.cache_dir) if self.config.cache_dir else None,
                            valid_patch_value=self.config.valid_patch_value,
                            bg_only_patches=[
                                {
                                    "volume_idx": int(entry['volume_idx']),
                                    "volume_name": str(entry['volume_name']),
                                    "start_pos": list(entry['start_pos']),
                                }
                                for entry in raw_bg_entries
                            ] if raw_bg_entries else None,
                            bg_sampling_enabled=self.config.bg_sampling_enabled,
                            bg_to_fg_ratio=self.config.bg_to_fg_ratio,
                            unlabeled_fg_patches=[
                                {
                                    "volume_idx": int(entry['volume_idx']),
                                    "volume_name": str(entry['volume_name']),
                                    "start_pos": list(entry['start_pos']),
                                }
                                for entry in raw_unlabeled_fg_entries
                            ] if raw_unlabeled_fg_entries else None,
                            unlabeled_fg_enabled=self.config.unlabeled_fg_enabled,
                            unlabeled_fg_threshold=self.config.unlabeled_fg_threshold,
                            unlabeled_fg_bbox_threshold=self.config.unlabeled_fg_bbox_threshold,
                        )

            # Create ChunkPatch objects from positions (when not loaded from cache)
            for labeled_idx, position in fg_positions:
                volume = labeled_volumes[labeled_idx]
                fg_patches.append(
                    ChunkPatch(
                        volume_index=volume.index,
                        volume_name=volume.name,
                        position=position,
                        patch_size=tuple(self.config.patch_size),
                        is_bg_only=False,
                    )
                )

            for labeled_idx, position in bg_positions:
                volume = labeled_volumes[labeled_idx]
                bg_patches.append(
                    ChunkPatch(
                        volume_index=volume.index,
                        volume_name=volume.name,
                        position=position,
                        patch_size=tuple(self.config.patch_size),
                        is_bg_only=True,
                    )
                )

            # Create ChunkPatch objects for unlabeled foreground patches
            for labeled_idx, position in unlabeled_fg_positions:
                volume = labeled_volumes[labeled_idx]
                unlabeled_fg_patches.append(
                    ChunkPatch(
                        volume_index=volume.index,
                        volume_name=volume.name,
                        position=position,
                        patch_size=tuple(self.config.patch_size),
                        is_bg_only=False,
                        is_unlabeled_fg=True,
                    )
                )

        # Add unvalidated patches for volumes without labels (unless unlabeled_fg_enabled,
        # in which case the filtered patches are already in unlabeled_fg_patches)
        if not validate or (any(v.label_source is None for v in self._volumes) and not self.config.unlabeled_fg_enabled):
            for volume in self._volumes:
                if not validate or volume.label_source is None:
                    fg_patches.extend(self.enumerate(volume, stride=self.config.stride))

        # Combine FG, BG, and unlabeled FG patches
        # Order: fg_patches, unlabeled_fg_patches, bg_patches
        all_patches = fg_patches + unlabeled_fg_patches + bg_patches
        self._n_fg = len(fg_patches)  # Store labeled FG count for train/val split
        self._n_unlabeled_fg = len(unlabeled_fg_patches)  # Store unlabeled FG count

        if not all_patches:
            raise RuntimeError("Chunk slicing produced zero patches across all volumes")

        # Compute sampling weights if BG sampling is enabled
        weights: Optional[List[float]] = None
        if self.config.bg_sampling_enabled and bg_patches:
            weights = self._compute_sampling_weights(len(fg_patches), len(bg_patches))
            logger.info(
                "ChunkSlicer: computed sampling weights for %d FG and %d BG patches (ratio=%.2f)",
                len(fg_patches), len(bg_patches), self.config.bg_to_fg_ratio
            )

        if unlabeled_fg_patches:
            logger.info(
                "ChunkSlicer: found %d unlabeled foreground patches for semi-supervised training",
                len(unlabeled_fg_patches),
            )

        self._patches = all_patches
        self._weights = weights
        return all_patches, weights

    def _compute_sampling_weights(self, n_fg: int, n_bg: int) -> List[float]:
        """Compute sampling weights so BG patches are sampled at bg_to_fg_ratio.

        If bg_to_fg_ratio=0.5 and we have 1000 FG patches, we want ~500 BG samples per epoch.
        Weight for each BG patch = target_bg_samples / n_bg = (n_fg * ratio) / n_bg
        """
        if n_bg == 0:
            return [1.0] * n_fg

        target_bg_samples = n_fg * self.config.bg_to_fg_ratio
        bg_weight = target_bg_samples / n_bg if n_bg > 0 else 0.0

        # FG patches have weight 1.0, BG patches have computed weight
        return [1.0] * n_fg + [bg_weight] * n_bg

    def _compute_valid_positions(
        self,
        labeled_volumes: Sequence[ChunkVolume],
        label_arrays: Sequence[object],
        label_names: Sequence[str],
        ignore_values: Sequence[Optional[Union[int, float]]],
    ) -> Tuple[
        List[Tuple[int, Tuple[int, ...]]],
        List[Tuple[int, Tuple[int, ...]]],
        List[Dict[str, object]],
        List[Dict[str, object]],
        List[Tuple[int, Tuple[int, ...]]],
        List[Dict[str, object]],
    ]:
        """Run find_valid_patches and map results to labeled volume indices.

        Returns:
            fg_positions, bg_positions, fg_entries, bg_entries, unlabeled_fg_positions, unlabeled_fg_entries
        """

        channel_selectors = None
        if self._label_channel_selector is not None:
            channel_selectors = [self._label_channel_selector] * len(label_arrays)

        # Check if all label sources can be used with find_valid_patches.
        # Valid sources are: zarr Arrays (have shape), zarr Groups (have keys method
        # for accessing resolution levels), or any array-like with shape attribute.
        # For unlabeled FG detection, we can use image sources instead of labels.
        all_valid_for_vectorized = True

        # When doing unlabeled FG detection with no labels, check image sources instead
        if self.config.unlabeled_fg_enabled and all(arr is None for arr in label_arrays):
            # Check image sources for vectorized processing
            for vol in labeled_volumes:
                img_src = vol.image_source
                if img_src is None:
                    all_valid_for_vectorized = False
                    break
                # Check if we can get a raw array from the handle
                raw = getattr(img_src, 'raw', lambda: img_src)()
                has_shape = getattr(raw, "shape", None) is not None
                has_keys = callable(getattr(raw, "keys", None))
                if not (has_shape or has_keys):
                    all_valid_for_vectorized = False
                    break
        else:
            for arr in label_arrays:
                has_shape = getattr(arr, "shape", None) is not None
                has_keys = callable(getattr(arr, "keys", None))
                if not (has_shape or has_keys):
                    all_valid_for_vectorized = False
                    break

        if not all_valid_for_vectorized:
            logger.info(
                "ChunkSlicer: falling back to sequential patch validation because some label sources "
                "lack array-style shape attributes or zarr group keys"
            )
            return self._compute_valid_positions_sequential(labeled_volumes)

        valid_patch_values = None
        if self.config.valid_patch_value is not None:
            valid_patch_values = [self.config.valid_patch_value] * len(label_arrays)

        # Collect image arrays for unlabeled foreground detection if enabled
        # Use raw() to get the underlying zarr group for resolution level access
        image_arrays = None
        if self.config.unlabeled_fg_enabled:
            image_arrays = []
            for vol in labeled_volumes:
                img_src = vol.image_source
                if img_src is not None:
                    raw = getattr(img_src, 'raw', lambda: img_src)()
                    image_arrays.append(raw)
                else:
                    image_arrays.append(None)

        result = find_valid_patches(
            label_arrays=label_arrays,
            label_names=label_names,
            patch_size=tuple(self.config.patch_size),
            bbox_threshold=self.config.min_bbox_percent,
            label_threshold=self.config.min_labeled_ratio,
            num_workers=self.config.num_workers,
            valid_patch_find_resolution=self.config.valid_patch_find_resolution,
            channel_selectors=channel_selectors,
            ignore_labels=ignore_values,
            valid_patch_values=valid_patch_values,
            collect_bg_only=self.config.bg_sampling_enabled,
            # Unlabeled foreground detection params
            image_arrays=image_arrays,
            collect_unlabeled_fg=self.config.unlabeled_fg_enabled,
            unlabeled_fg_threshold=self.config.unlabeled_fg_threshold,
            unlabeled_fg_bbox_threshold=self.config.unlabeled_fg_bbox_threshold,
        )

        fg_positions: List[Tuple[int, Tuple[int, ...]]] = []
        bg_positions: List[Tuple[int, Tuple[int, ...]]] = []
        unlabeled_fg_positions: List[Tuple[int, Tuple[int, ...]]] = []

        for entry in result['fg_patches']:
            labeled_idx = int(entry['volume_idx'])
            start_pos = tuple(int(v) for v in entry['start_pos'])
            fg_positions.append((labeled_idx, start_pos))

        for entry in result['bg_patches']:
            labeled_idx = int(entry['volume_idx'])
            start_pos = tuple(int(v) for v in entry['start_pos'])
            bg_positions.append((labeled_idx, start_pos))

        for entry in result.get('unlabeled_fg_patches', []):
            labeled_idx = int(entry['volume_idx'])
            start_pos = tuple(int(v) for v in entry['start_pos'])
            unlabeled_fg_positions.append((labeled_idx, start_pos))

        return (
            fg_positions,
            bg_positions,
            result['fg_patches'],
            result['bg_patches'],
            unlabeled_fg_positions,
            result.get('unlabeled_fg_patches', []),
        )

    def _compute_valid_positions_sequential(
        self, labeled_volumes: Sequence[ChunkVolume]
    ) -> Tuple[
        List[Tuple[int, Tuple[int, ...]]],
        List[Tuple[int, Tuple[int, ...]]],
        List[Dict[str, object]],
        List[Dict[str, object]],
        List[Tuple[int, Tuple[int, ...]]],
        List[Dict[str, object]],
    ]:
        """Sequential fallback for patch validation.

        Returns:
            fg_positions, bg_positions, fg_entries, bg_entries, unlabeled_fg_positions, unlabeled_fg_entries
        """
        fg_positions: List[Tuple[int, Tuple[int, ...]]] = []
        bg_positions: List[Tuple[int, Tuple[int, ...]]] = []
        fg_entries: List[Dict[str, object]] = []
        bg_entries: List[Dict[str, object]] = []
        unlabeled_fg_positions: List[Tuple[int, Tuple[int, ...]]] = []
        unlabeled_fg_entries: List[Dict[str, object]] = []

        for labeled_idx, volume in enumerate(labeled_volumes):
            label_array = volume.label_source

            # Unlabeled FG collection requires image source
            should_collect_unlabeled_fg = (
                self.config.unlabeled_fg_enabled and volume.image_source is not None
            )

            # Handle volumes with NO labels (e.g., self-supervised training)
            if label_array is None:
                if should_collect_unlabeled_fg:
                    # Enumerate all positions and check image content only
                    candidate_patches = list(self.enumerate(volume, stride=self.config.stride))
                    logger.info("Filtering %d candidate patches for volume %s", len(candidate_patches), volume.name)
                    for candidate in tqdm(
                        candidate_patches,
                        desc=f"Filtering unlabeled patches ({volume.name})",
                        leave=False,
                    ):
                        position = tuple(int(v) for v in candidate.position)
                        patch_vol = int(np.prod(candidate.patch_size))
                        if self._check_image_has_data(volume.image_source, position, patch_vol):
                            unlabeled_fg_positions.append((labeled_idx, position))
                            unlabeled_fg_entries.append({
                                'volume_idx': labeled_idx,
                                'volume_name': volume.name,
                                'start_pos': list(position),
                            })
                continue

            ignore_value = volume.label_ignore_value
            # BG collection only valid when ignore_value is set
            should_collect_bg = self.config.bg_sampling_enabled and ignore_value is not None

            candidate_patches = self.enumerate(volume, stride=self.config.stride)
            for candidate in candidate_patches:
                mask_patch = self._extract_label_patch(label_array, candidate.position)
                if mask_patch is None:
                    continue
                if ignore_value is not None:
                    mask_patch = zero_ignore_labels(mask_patch, ignore_value)

                mask = collapse_patch_to_spatial(
                    mask_patch,
                    spatial_ndim=len(candidate.patch_size),
                    channel_selector=self._label_channel_selector,
                )
                if self.config.valid_patch_value is not None:
                    mask = (mask == self.config.valid_patch_value)
                else:
                    mask = np.abs(mask) > 0

                position = tuple(int(v) for v in candidate.position)
                patch_vol = mask.size

                # Check if this patch is effectively unlabeled
                labeled_ratio = np.count_nonzero(mask) / patch_vol if patch_vol > 0 else 0
                is_effectively_unlabeled = labeled_ratio < self.config.min_labeled_ratio

                if not mask.any():
                    # No foreground - this is a BG-only patch or unlabeled FG candidate
                    if should_collect_bg:
                        bg_positions.append((labeled_idx, position))
                        bg_entries.append({
                            'volume_idx': labeled_idx,
                            'volume_name': volume.name,
                            'start_pos': list(position),
                        })
                    # Also check for unlabeled FG
                    if should_collect_unlabeled_fg:
                        if self._check_image_has_data(volume.image_source, position, patch_vol):
                            unlabeled_fg_positions.append((labeled_idx, position))
                            unlabeled_fg_entries.append({
                                'volume_idx': labeled_idx,
                                'volume_name': volume.name,
                                'start_pos': list(position),
                            })
                    continue

                bbox = compute_bounding_box_3d(mask)
                if bbox is None:
                    if is_effectively_unlabeled and should_collect_unlabeled_fg:
                        if self._check_image_has_data(volume.image_source, position, patch_vol):
                            unlabeled_fg_positions.append((labeled_idx, position))
                            unlabeled_fg_entries.append({
                                'volume_idx': labeled_idx,
                                'volume_name': volume.name,
                                'start_pos': list(position),
                            })
                    continue

                bb_vol = bounding_box_volume(bbox)
                if patch_vol == 0:
                    continue

                bbox_ratio = bb_vol / patch_vol
                passes_bbox = bbox_ratio >= self.config.min_bbox_percent
                passes_label = labeled_ratio >= self.config.min_labeled_ratio

                if passes_bbox and passes_label:
                    # Valid labeled patch
                    fg_positions.append((labeled_idx, position))
                    fg_entries.append({
                        'volume_idx': labeled_idx,
                        'volume_name': volume.name,
                        'start_pos': list(position),
                    })
                elif should_collect_unlabeled_fg:
                    # Failed label validation - check if it's an unlabeled FG patch
                    if self._check_image_has_data(volume.image_source, position, patch_vol):
                        unlabeled_fg_positions.append((labeled_idx, position))
                        unlabeled_fg_entries.append({
                            'volume_idx': labeled_idx,
                            'volume_name': volume.name,
                            'start_pos': list(position),
                        })

        return fg_positions, bg_positions, fg_entries, bg_entries, unlabeled_fg_positions, unlabeled_fg_entries

    def _check_image_has_data(
        self, image_source: object, position: Tuple[int, ...], patch_vol: int
    ) -> bool:
        """Check if an image patch has sufficient non-zero data for unlabeled FG."""
        try:
            image_patch = self._extract_image_patch_for_validation(image_source, position)
            if image_patch is None:
                return False
            image_nonzero = np.count_nonzero(image_patch)
            img_fraction = image_nonzero / patch_vol if patch_vol > 0 else 0
            if img_fraction < self.config.unlabeled_fg_threshold:
                logger.debug(
                    "_check_image_has_data: img_fraction %.4f < threshold %.4f at %s",
                    img_fraction, self.config.unlabeled_fg_threshold, position
                )
                return False
            # Compute bbox coverage
            img_mask = image_patch != 0
            bbox = compute_bounding_box_3d(img_mask)
            if bbox is None:
                logger.debug("_check_image_has_data: bbox is None at position %s", position)
                return False
            bb_vol = bounding_box_volume(bbox)
            img_bbox_fraction = bb_vol / patch_vol if patch_vol > 0 else 0
            if img_bbox_fraction < self.config.unlabeled_fg_bbox_threshold:
                logger.debug(
                    "_check_image_has_data: bbox_fraction %.4f < threshold %.4f at %s",
                    img_bbox_fraction, self.config.unlabeled_fg_bbox_threshold, position
                )
                return False
            return True
        except Exception as e:
            logger.warning("_check_image_has_data: exception at position %s: %s", position, e)
            return False

    def _extract_image_patch_for_validation(
        self, image_source: object, position: Tuple[int, ...]
    ) -> Optional[np.ndarray]:
        """Extract image patch for unlabeled FG validation."""
        if image_source is None:
            return None
        pos = tuple(int(v) for v in position)
        patch_size = tuple(int(v) for v in self.config.patch_size)
        try:
            # Handle ArrayHandle types (e.g., ZarrArrayHandle) with read_window method
            if hasattr(image_source, 'read_window'):
                arr = image_source.read_window(pos, patch_size)
            elif self._is_2d:
                y, x = pos
                ph, pw = patch_size[-2:]
                arr = np.asarray(image_source[y:y + ph, x:x + pw])
            else:
                z, y, x = pos
                pd, ph, pw = patch_size
                arr = np.asarray(image_source[z:z + pd, y:y + ph, x:x + pw])
            # Reduce to spatial only (squeeze channel dims if present)
            while arr.ndim > len(patch_size):
                arr = arr[0] if arr.shape[0] == 1 else arr.reshape(arr.shape[-len(patch_size):])
            return arr
        except Exception:
            return None

    @property
    def patches(self) -> List[ChunkPatch]:
        return list(self._patches)

    @property
    def weights(self) -> Optional[List[float]]:
        if self._weights is None:
            return None
        return list(self._weights)

    @property
    def n_fg(self) -> int:
        """Number of foreground patches (patches with labels)."""
        return getattr(self, '_n_fg', len(self._patches))

    @property
    def n_unlabeled_fg(self) -> int:
        """Number of unlabeled foreground patches (has image data but no labels)."""
        return getattr(self, '_n_unlabeled_fg', 0)

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

        # Extract raw (un-padded) region first so normalization does not include
        # any zero padding. Then pad/crop to the configured patch size.
        image_patch = self._extract_image_patch(volume.image, patch.position)
        if normalizer is not None:
            image_patch = normalizer.run(image_patch)
        else:
            image_patch = image_patch.astype(np.float32, copy=False)
        image_patch = self._finalize_image_patch(image_patch, patch.position)
        image_tensor = np.ascontiguousarray(image_patch[np.newaxis, ...], dtype=np.float32)

        labels: Dict[str, np.ndarray] = {}
        is_unlabeled = True
        for target_name in self.target_names:
            label_arr = volume.labels.get(target_name)
            label_patch = self._extract_label_patch(label_arr, patch.position)
            if label_patch is None:
                label_patch = np.zeros_like(image_patch, dtype=np.float32)
            else:
                if is_unlabeled and np.count_nonzero(label_patch):
                    is_unlabeled = False

            if label_patch.ndim == image_patch.ndim:
                label_tensor = np.ascontiguousarray(label_patch[np.newaxis, ...])
            elif label_patch.ndim == image_patch.ndim + 1:
                label_tensor = np.ascontiguousarray(label_patch)
            else:
                raise ValueError(
                    f"Label array for target '{target_name}' has unexpected ndim {label_patch.ndim}"
                )

            labels[target_name] = label_tensor

        mesh_payloads: Dict[str, Dict[str, object]] = {}
        for mesh_id, handle in volume.meshes.items():
            payload = handle.read()
            mesh_payloads[mesh_id] = {
                "payload": payload,
                "metadata": handle.metadata,
            }

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
            'global_position': list(int(v) for v in patch.position),
            'global_end': [
                int(start + size) for start, size in zip(patch.position, patch.patch_size)
            ],
            'source_path': str(getattr(volume.image, 'path', '')) if hasattr(volume.image, 'path') else '',
            'label_source_paths': {
                name: str(getattr(handle, 'path', '')) if hasattr(handle, 'path') else ''
                for name, handle in volume.labels.items()
            },
        }
        if mesh_payloads:
            patch_info['meshes'] = {
                mesh_id: {
                    'path': str(handle.path),
                    'source_volume': handle.metadata.source_volume_id,
                }
                for mesh_id, handle in volume.meshes.items()
            }

        return ChunkResult(
            image=image_tensor.astype(np.float32, copy=False),
            labels=labels,
            is_unlabeled=is_unlabeled,
            patch_info=patch_info,
            meshes=mesh_payloads,
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
            # Defer padding/cropping so normalization can run on the raw region first
            patch = image.read_window(pos, patch_size)
            arr = np.asarray(patch)
            return arr

        arr = np.asarray(image)
        if self._is_2d:
            if len(pos) != 2:
                raise ValueError(f"2D chunk position must have two coordinates; got {position}")
            y, x = pos
            ph, pw = patch_size[-2:]
            if arr.ndim == 2:
                patch = arr[y : y + ph, x : x + pw]
                return patch.astype(np.float32, copy=False)
            if arr.ndim == 3 and arr.shape[0] == 1:
                patch = arr[0, y : y + ph, x : x + pw]
                return patch.astype(np.float32, copy=False)
            raise ValueError("2D chunk extraction expects image data with shape (H, W) or (1, H, W)")

        if len(pos) != 3:
            raise ValueError(f"3D chunk position must have three coordinates; got {position}")
        z, y, x = pos
        pd, ph, pw = patch_size

        if arr.ndim == 3:
            patch = arr[z : z + pd, y : y + ph, x : x + pw]
            return patch.astype(np.float32, copy=False)
        if arr.ndim == 4 and arr.shape[0] == 1:
            patch = arr[0, z : z + pd, y : y + ph, x : x + pw]
            return patch.astype(np.float32, copy=False)
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
                return pad_or_crop_2d(patch, (ph, pw))
            if arr.ndim == 3:
                layout = self._infer_label_layout_2d(arr.shape, (y, x))
                if layout == "channel_first":
                    channels = arr.shape[0]
                    padded = [
                        pad_or_crop_2d(
                            arr[c, y : y + ph, x : x + pw], (ph, pw)
                        )
                        for c in range(channels)
                    ]
                    return np.stack(padded, axis=0)
                if layout == "channel_last":
                    patch = arr[y : y + ph, x : x + pw, :]
                    flat = patch.reshape(patch.shape[0], patch.shape[1], -1)
                    flat = np.moveaxis(flat, -1, 0)
                    padded = [
                        pad_or_crop_2d(flat[c], (ph, pw))
                        for c in range(flat.shape[0])
                    ]
                    return np.stack(padded, axis=0)
                raise ValueError(
                    "2D chunk extraction encountered an ambiguous channel layout; "
                    f"position {position} is compatible with both leading and trailing channel axes for array shape {arr.shape}."
                )
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
            return pad_or_crop_3d(patch, (pd, ph, pw))
        if arr.ndim == 4:
            layout = self._infer_label_layout_3d(arr.shape, (z, y, x))
            if layout == "channel_first":
                channels = arr.shape[0]
                padded = [
                    pad_or_crop_3d(
                        arr[c, z : z + pd, y : y + ph, x : x + pw], (pd, ph, pw)
                    )
                    for c in range(channels)
                ]
                return np.stack(padded, axis=0)
            if layout == "channel_last":
                patch = arr[z : z + pd, y : y + ph, x : x + pw, :]
                flat = patch.reshape(patch.shape[0], patch.shape[1], patch.shape[2], -1)
                flat = np.moveaxis(flat, -1, 0)
                padded = [
                    pad_or_crop_3d(flat[c], (pd, ph, pw))
                    for c in range(flat.shape[0])
                ]
                return np.stack(padded, axis=0)
            raise ValueError(
                "3D chunk extraction encountered an ambiguous 4D label layout; "
                f"position {position} matches both leading and trailing channel interpretations for array shape {arr.shape}."
            )
        if arr.ndim == 5:
            if not self._position_fits_volume((z, y, x), arr.shape[:3]):
                raise ValueError(
                    f"3D chunk extraction position {position} exceeds label array spatial shape {arr.shape[:3]}"
                )
            patch = arr[z : z + pd, y : y + ph, x : x + pw, :, :]
            flat = patch.reshape(patch.shape[0], patch.shape[1], patch.shape[2], -1)
            flat = np.moveaxis(flat, -1, 0)
            padded = [
                pad_or_crop_3d(flat[c], (pd, ph, pw))
                for c in range(flat.shape[0])
            ]
            return np.stack(padded, axis=0)
        raise ValueError(
            "3D chunk extraction expects label data with shape (D, H, W), (C, D, H, W), "
            "(D, H, W, C), or higher-dimensional channel groupings trailing the spatial axes."
        )

    def _finalize_image_patch(self, patch: np.ndarray, position: Tuple[int, ...]) -> np.ndarray:
        if self._is_2d:
            ph, pw = self.config.patch_size[-2:]
            if patch.ndim == 2:
                return pad_or_crop_2d(patch, (ph, pw))
            if patch.ndim == 3 and patch.shape[0] == 1:
                return pad_or_crop_2d(patch[0], (ph, pw))
            if patch.ndim == 3:
                channels = patch.shape[0]
                padded = [
                    pad_or_crop_2d(patch[c], (ph, pw))
                    for c in range(channels)
                ]
                return np.stack(padded, axis=0)
            raise ValueError("2D chunk extraction expects image data with shape (H, W) or (C, H, W)")

        pd, ph, pw = self.config.patch_size
        if patch.ndim == 3:
            return pad_or_crop_3d(patch, (pd, ph, pw))
        if patch.ndim == 4 and patch.shape[0] == 1:
            return pad_or_crop_3d(patch[0], (pd, ph, pw))
        if patch.ndim == 4:
            channels = patch.shape[0]
            padded = [
                pad_or_crop_3d(patch[c], (pd, ph, pw))
                for c in range(channels)
            ]
            return np.stack(padded, axis=0)
        raise ValueError("3D chunk extraction expects image data with shape (D, H, W) or (C, D, H, W)")

    def _finalize_label_patch(self, patch: np.ndarray, position: Tuple[int, ...]) -> np.ndarray:
        if self._is_2d:
            ph, pw = self.config.patch_size[-2:]
            if patch.ndim == 2:
                return pad_or_crop_2d(patch, (ph, pw))
            if patch.ndim == 3:
                channels = patch.shape[0]
                padded = [
                    pad_or_crop_2d(patch[c], (ph, pw))
                    for c in range(channels)
                ]
                return np.stack(padded, axis=0)
            raise ValueError("2D chunk extraction expects label data with shape (H, W) or (C, H, W)")

        pd, ph, pw = self.config.patch_size
        if patch.ndim == 3:
            return pad_or_crop_3d(patch, (pd, ph, pw))
        if patch.ndim == 4:
            channels = patch.shape[0]
            padded = [
                pad_or_crop_3d(patch[c], (pd, ph, pw))
                for c in range(channels)
            ]
            return np.stack(padded, axis=0)
        raise ValueError("3D chunk extraction expects label data with shape (D, H, W) or (C, D, H, W)")

    @staticmethod
    def _position_fits_volume(position: Tuple[int, ...], spatial_shape: Sequence[int]) -> bool:
        if len(position) != len(spatial_shape):
            raise ValueError(
                f"Position dimensionality {len(position)} does not match spatial shape {spatial_shape}"
            )
        return all(0 <= coord < size for coord, size in zip(position, spatial_shape))

    @classmethod
    def _infer_label_layout_2d(cls, shape: Tuple[int, ...], position: Tuple[int, int]) -> str:
        if len(shape) != 3:
            raise ValueError(f"2D label layout inference expects a 3D array; got shape {shape}")

        channel_first_spatial = (int(shape[1]), int(shape[2]))
        channel_last_spatial = (int(shape[0]), int(shape[1]))

        first_valid = cls._position_fits_volume(position, channel_first_spatial)
        last_valid = cls._position_fits_volume(position, channel_last_spatial)

        if first_valid and not last_valid:
            return "channel_first"
        if last_valid and not first_valid:
            return "channel_last"
        if first_valid and last_valid:
            return "ambiguous"
        raise ValueError(
            f"2D label position {position} falls outside both leading and trailing spatial bounds for array shape {shape}"
        )

    @classmethod
    def _infer_label_layout_3d(cls, shape: Tuple[int, ...], position: Tuple[int, int, int]) -> str:
        if len(shape) != 4:
            raise ValueError(f"3D label layout inference expects a 4D array; got shape {shape}")

        channel_first_spatial = (int(shape[1]), int(shape[2]), int(shape[3]))
        channel_last_spatial = (int(shape[0]), int(shape[1]), int(shape[2]))

        first_valid = cls._position_fits_volume(position, channel_first_spatial)
        last_valid = cls._position_fits_volume(position, channel_last_spatial)

        if first_valid and not last_valid:
            return "channel_first"
        if last_valid and not first_valid:
            return "channel_last"
        if first_valid and last_valid:
            return "ambiguous"
        raise ValueError(
            f"3D label position {position} falls outside both leading and trailing spatial bounds for array shape {shape}"
        )
