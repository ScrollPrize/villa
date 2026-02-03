"""
Simplified zarr-based dataset for volumetric training.

This module provides a streamlined Dataset implementation that:
- Only reads OME-Zarr stores (no adapter abstraction)
- Inlines patch extraction logic (no slicer abstraction)
- Preserves semi-supervised learning support
- Handles normalization and augmentation
- Supports samples_mapping.json for packed sparse zarr datasets
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset

from vesuvius.utils.utils import pad_or_crop_3d, pad_or_crop_2d
from .intensity_properties import initialize_intensity_properties
from ..training.normalization import get_normalization
from ..augmentation.pipelines import create_training_transforms
from ..augmentation.transforms.utils.perf import collect_augmentation_names


logger = logging.getLogger(__name__)


@dataclass
class VolumeInfo:
    """Metadata for a loaded OME-Zarr volume."""
    volume_id: str
    image_path: Path
    image_array: zarr.Array
    label_paths: Dict[str, Optional[Path]]
    label_arrays: Dict[str, Optional[zarr.Array]]
    spatial_shape: Tuple[int, ...]
    has_labels: bool = True


@dataclass
class PatchInfo:
    """Describes a single extractable patch."""
    volume_index: int
    volume_name: str
    position: Tuple[int, ...]
    patch_size: Tuple[int, ...]
    is_unlabeled_fg: bool = False
    # For packed sparse zarr: boundary beyond which data belongs to neighbor
    safe_boundary: Optional[Tuple[int, ...]] = None


class ZarrDataset(Dataset):
    """
    PyTorch Dataset for 3D volumetric data stored in OME-Zarr format.

    Handles:
    - Direct OME-Zarr reading (no adapter abstraction)
    - Inline 3D patch extraction (no slicer abstraction)
    - Semi-supervised support (labeled + unlabeled foreground patches)
    - Normalization and augmentation
    - Patch validation and caching

    Expected directory structure:
        data_path/
            images/
                volume1.zarr/       # OME-Zarr with resolution levels
                volume2.zarr/
            labels/
                volume1_ink.zarr/   # Labels named as {volume_id}_{target}.zarr
                volume2_ink.zarr/

    Parameters
    ----------
    mgr : ConfigManager
        Configuration manager with dataset settings.
    is_training : bool
        Whether this is a training dataset (enables augmentation).
    """

    def __init__(
        self,
        mgr,
        is_training: bool = True,
    ) -> None:
        super().__init__()
        self.mgr = mgr
        self.is_training = is_training
        self._profile_augmentations = bool(getattr(mgr, 'profile_augmentations', False))
        self._augmentation_names: List[str] = []

        # Configuration from manager
        self.data_path = Path(mgr.data_path)
        self.patch_size = tuple(mgr.train_patch_size)
        self.targets = getattr(mgr, 'targets', {})
        self.target_names = [
            name for name, info in self.targets.items()
            if not info.get("auxiliary_task", False)
        ]
        if not self.target_names:
            self.target_names = ['ink']  # Default target

        # Determine 2D vs 3D
        self.is_2d = len(self.patch_size) == 2

        # Validation parameters
        self.min_labeled_ratio = getattr(mgr, 'min_labeled_ratio', 0.10)
        self.min_bbox_percent = getattr(mgr, 'min_bbox_percent', 0.95)
        self.skip_patch_validation = getattr(mgr, 'skip_patch_validation', False)
        self.allow_unlabeled_data = getattr(mgr, 'allow_unlabeled_data', False)

        # Mapping file parameters (for packed sparse zarr)
        self.allow_gap_extension = getattr(mgr, 'allow_gap_extension', True)

        # OME-Zarr parameters
        self.ome_zarr_resolution = getattr(mgr, 'ome_zarr_resolution', 0)
        self.valid_patch_find_resolution = getattr(mgr, 'valid_patch_find_resolution', 1)

        # Semi-supervised parameters
        self.unlabeled_fg_enabled = getattr(mgr, 'unlabeled_foreground_enabled', False)
        self.unlabeled_fg_threshold = getattr(mgr, 'unlabeled_foreground_threshold', 0.05)
        self.unlabeled_fg_bbox_threshold = getattr(mgr, 'unlabeled_foreground_bbox_threshold', 0.15)
        self.unlabeled_fg_volume_ids = set(getattr(mgr, 'unlabeled_foreground_volumes', []) or [])

        # Caching
        self.cache_enabled = getattr(mgr, 'cache_valid_patches', True)
        self.cache_dir = self.data_path / '.patches_cache'

        # Initialize storage
        self._volumes: List[VolumeInfo] = []
        self._patches: List[PatchInfo] = []
        self._n_labeled_fg: int = 0
        self._n_unlabeled_fg: int = 0

        # Normalization (initialized after loading volumes)
        self.normalization_scheme = getattr(mgr, 'normalization_scheme', 'zscore')
        self.intensity_properties = getattr(mgr, 'intensity_properties', None) or {}
        self.normalizer = None

        # Transforms (initialized after normalization)
        self.transforms = None

        # Initialize
        self._discover_and_load_volumes()
        self._initialize_normalization()
        self._build_patch_index()
        self._initialize_transforms()

        logger.info(
            "ZarrDataset initialized: %d volumes, %d patches (%d labeled, %d unlabeled)",
            len(self._volumes),
            len(self._patches),
            self._n_labeled_fg,
            self._n_unlabeled_fg,
        )

    # -------------------------------------------------------------------------
    # Volume Discovery and Loading
    # -------------------------------------------------------------------------

    def _discover_and_load_volumes(self) -> None:
        """Discover and load all OME-Zarr volumes."""
        images_dir = self.data_path / "images"
        labels_dir = self.data_path / "labels"

        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")

        # Find all image zarrs
        image_zarrs = {
            p.stem: p for p in images_dir.iterdir()
            if p.is_dir() and p.suffix == ".zarr"
        }

        if not image_zarrs:
            raise ValueError(f"No .zarr directories found in {images_dir}")

        logger.info("Found %d image volumes in %s", len(image_zarrs), images_dir)

        # Build volume list
        for volume_id, image_path in sorted(image_zarrs.items()):
            label_paths: Dict[str, Optional[Path]] = {}
            label_arrays: Dict[str, Optional[zarr.Array]] = {}
            has_any_label = False

            for target in self.target_names:
                label_path = labels_dir / f"{volume_id}_{target}.zarr"
                if label_path.exists():
                    label_paths[target] = label_path
                    label_arrays[target] = self._open_zarr(label_path)
                    has_any_label = True
                else:
                    if not self.allow_unlabeled_data:
                        raise FileNotFoundError(
                            f"Label not found: {label_path} (set allow_unlabeled_data=True to allow)"
                        )
                    label_paths[target] = None
                    label_arrays[target] = None
                    logger.warning("No label found for volume '%s' target '%s'", volume_id, target)

            image_array = self._open_zarr(image_path)
            spatial_shape = self._get_spatial_shape(image_array)

            self._volumes.append(VolumeInfo(
                volume_id=volume_id,
                image_path=image_path,
                image_array=image_array,
                label_paths=label_paths,
                label_arrays=label_arrays,
                spatial_shape=spatial_shape,
                has_labels=has_any_label,
            ))

            logger.info(
                "Loaded volume '%s': shape=%s, has_labels=%s",
                volume_id, spatial_shape, has_any_label
            )

    def _open_zarr(self, path: Path) -> zarr.Array:
        """Open a zarr array at the configured resolution level."""
        store = zarr.open(path, mode="r")

        # Handle zarr Group (OME-Zarr format)
        if isinstance(store, zarr.hierarchy.Group):
            # Try to get the requested resolution level
            level_key = str(self.ome_zarr_resolution)
            if level_key in store:
                return store[level_key]
            # Fallback keys
            for key in ["0", "data", "arr_0"]:
                if key in store and hasattr(store[key], "shape"):
                    return store[key]
            raise ValueError(f"Cannot find array in zarr Group at {path}")

        return store

    def _get_spatial_shape(self, arr: zarr.Array) -> Tuple[int, ...]:
        """Extract spatial shape (Z, Y, X) or (Y, X) from array."""
        shape = arr.shape
        if self.is_2d:
            if len(shape) == 2:
                return tuple(shape)
            if len(shape) >= 2:
                return tuple(shape[-2:])
        else:
            if len(shape) == 3:
                return tuple(shape)
            if len(shape) >= 3:
                return tuple(shape[-3:])
        raise ValueError(f"Unsupported array shape for {'2D' if self.is_2d else '3D'}: {shape}")

    # -------------------------------------------------------------------------
    # Normalization
    # -------------------------------------------------------------------------

    def _initialize_normalization(self) -> None:
        """Initialize intensity properties and normalizer."""
        skip_sampling = getattr(self.mgr, 'skip_intensity_sampling', True)

        if not skip_sampling and not self.intensity_properties and self.normalization_scheme in ['zscore', 'ct']:
            # Build target_volumes dict for intensity computation (legacy format)
            target_volumes = self._build_target_volumes_for_intensity()
            self.intensity_properties = initialize_intensity_properties(
                target_volumes=target_volumes,
                normalization_scheme=self.normalization_scheme,
                existing_properties=None,
                cache_enabled=self.cache_enabled,
                cache_dir=self.cache_dir,
                mgr=self.mgr,
            )

        self.normalizer = get_normalization(self.normalization_scheme, self.intensity_properties)

    def _build_target_volumes_for_intensity(self) -> Dict:
        """Build legacy target_volumes format for intensity_properties module."""
        first_target = self.target_names[0]
        volumes_list = []
        for vol in self._volumes:
            label_arr = vol.label_arrays.get(first_target)
            volumes_list.append({
                'data': {
                    'data': vol.image_array,
                    'label': label_arr,
                }
            })
        return {first_target: volumes_list}

    # -------------------------------------------------------------------------
    # Patch Index Building
    # -------------------------------------------------------------------------

    def _build_patch_index(self) -> None:
        """Build the index of valid patches."""
        # Check for samples_mapping.json (packed sparse zarr format)
        mapping_file = self.data_path / 'samples_mapping.json'
        if mapping_file.exists():
            self._load_from_mapping(mapping_file)
            return

        # Try to load from pre-computed cache
        cache_data = self._try_load_cache()
        if cache_data is not None:
            self._load_from_cache(cache_data)
            return

        # No cache found - enumerate all patches without validation
        logger.warning(
            "No patch cache found. Enumerating all patches without validation. "
            "Run `vesuvius.build_patch_cache --config <config.yaml>` to generate cache."
        )
        self._enumerate_all_patches()

        logger.info(
            "Patch index built: %d labeled FG, %d unlabeled FG, %d total",
            self._n_labeled_fg, self._n_unlabeled_fg, len(self._patches)
        )

    def _load_from_mapping(self, mapping_file: Path) -> None:
        """
        Load valid patches from samples_mapping.json (packed sparse zarr format).

        This method generates patches directly from known sample positions,
        avoiding the need to scan the entire sparse volume for valid regions.

        Parameters
        ----------
        mapping_file : Path
            Path to the samples_mapping.json file
        """
        with open(mapping_file) as f:
            mapping = json.load(f)

        # Check for target name mismatch
        mapping_target = mapping.get('target_name')
        if mapping_target and mapping_target not in self.target_names:
            logger.warning(
                "Mapping file target '%s' not in config targets %s. "
                "Labels may not be found. Consider adding '%s' to your config targets.",
                mapping_target, self.target_names, mapping_target
            )

        layout = mapping.get('layout', {})
        sample_shape = tuple(layout.get('sample_shape', []))
        gap_size = layout.get('gap_size', 0)
        samples = mapping.get('samples', [])

        # Check if patch size would intrude into neighboring samples
        if sample_shape:
            axis_names = ['Z', 'Y', 'X'] if not self.is_2d else ['Y', 'X']
            for i, (ps, ss) in enumerate(zip(self.patch_size, sample_shape)):
                max_safe_size = ss + gap_size
                if ps > max_safe_size:
                    logger.warning(
                        "patch_size[%s]=%d > sample_shape[%s]=%d + gap_size=%d. "
                        "Patches may extend into neighbor region - will zero out intrusion.",
                        axis_names[i], ps, axis_names[i], ss, gap_size
                    )

        if not sample_shape or not samples:
            logger.warning("Invalid mapping file, falling back to standard validation")
            if self.skip_patch_validation:
                self._enumerate_all_patches()
            else:
                self._find_valid_patches()
            return

        logger.info(f"Loading patches from mapping file: {len(samples)} samples")

        # Track unlabeled samples
        unlabeled_sample_count = 0

        # Generate patches within each sample region
        for sample_info in samples:
            zarr_pos = tuple(sample_info['zarr_position'])
            # Use sample-specific shape if available, otherwise layout shape
            s_shape = tuple(sample_info.get('sample_shape', sample_shape))

            # Read pre-computed label validation from preprocessing (if available)
            has_labels = sample_info.get('has_labels', True)
            if not has_labels:
                unlabeled_sample_count += 1

            # Compute effective region size for tiling
            # If allow_gap_extension is True, extend region into gap for more patches
            if self.allow_gap_extension and gap_size > 0:
                effective_shape = tuple(s + gap_size for s in s_shape)
            else:
                effective_shape = s_shape

            # Generate patch positions within this sample region
            if self.is_2d:
                positions = self._iter_2d_positions_in_region(zarr_pos, effective_shape)
            else:
                positions = self._iter_3d_positions_in_region(zarr_pos, effective_shape)

            # Compute safe boundary for this sample (sample data + gap, before next sample)
            safe_boundary = tuple(p + s + gap_size for p, s in zip(zarr_pos, s_shape))

            for pos in positions:
                # Skip patches where >50% would be padding (outside sample data region)
                data_ratio = self._compute_data_overlap_ratio(pos, zarr_pos, s_shape)
                if data_ratio < 0.5:
                    continue

                self._patches.append(PatchInfo(
                    volume_index=0,  # Single packed volume
                    volume_name=self._volumes[0].volume_id if self._volumes else 'volume',
                    position=pos,
                    patch_size=self.patch_size,
                    is_unlabeled_fg=not has_labels,
                    safe_boundary=safe_boundary,
                ))

        # Update counts
        self._n_labeled_fg = sum(1 for p in self._patches if not p.is_unlabeled_fg)
        self._n_unlabeled_fg = sum(1 for p in self._patches if p.is_unlabeled_fg)

        if unlabeled_sample_count > 0:
            logger.warning(
                "%d of %d samples have no labels (marked as unlabeled)",
                unlabeled_sample_count, len(samples)
            )
        logger.info(
            f"Generated {len(self._patches)} patches from {len(samples)} samples "
            f"({self._n_labeled_fg} labeled, {self._n_unlabeled_fg} unlabeled)"
        )

    def _iter_2d_positions_in_region(
        self,
        region_start: Tuple[int, int],
        region_size: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        """Generate 2D patch positions within a specific region."""
        ry, rx = region_start
        rh, rw = region_size
        ph, pw = self.patch_size

        positions = []
        for y in range(ry, ry + max(1, rh - ph + 1), ph):
            for x in range(rx, rx + max(1, rw - pw + 1), pw):
                positions.append((y, x))
        return positions

    def _iter_3d_positions_in_region(
        self,
        region_start: Tuple[int, int, int],
        region_size: Tuple[int, int, int],
    ) -> List[Tuple[int, int, int]]:
        """Generate 3D patch positions within a specific region."""
        rz, ry, rx = region_start
        rd, rh, rw = region_size
        pd, ph, pw = self.patch_size

        positions = []
        for z in range(rz, rz + max(1, rd - pd + 1), pd):
            for y in range(ry, ry + max(1, rh - ph + 1), ph):
                for x in range(rx, rx + max(1, rw - pw + 1), pw):
                    positions.append((z, y, x))
        return positions

    def _compute_data_overlap_ratio(
        self,
        patch_pos: Tuple[int, ...],
        sample_pos: Tuple[int, ...],
        sample_shape: Tuple[int, ...],
    ) -> float:
        """
        Compute the fraction of a patch that overlaps with actual sample data.

        Parameters
        ----------
        patch_pos : Tuple[int, ...]
            Start position of the patch in zarr coordinates
        sample_pos : Tuple[int, ...]
            Start position of the sample in zarr coordinates
        sample_shape : Tuple[int, ...]
            Shape of the sample data (excluding gap)

        Returns
        -------
        float
            Ratio of patch volume that overlaps with sample data (0.0 to 1.0)
        """
        overlap_volume = 1
        patch_volume = 1

        for i, ps in enumerate(self.patch_size):
            patch_start = patch_pos[i]
            patch_end = patch_start + ps
            sample_start = sample_pos[i]
            sample_end = sample_start + sample_shape[i]

            # Compute overlap on this axis
            overlap_start = max(patch_start, sample_start)
            overlap_end = min(patch_end, sample_end)
            overlap = max(0, overlap_end - overlap_start)

            overlap_volume *= overlap
            patch_volume *= ps

        return overlap_volume / patch_volume if patch_volume > 0 else 0.0

    def _enumerate_all_patches(self) -> None:
        """Enumerate all patches without validation (for dense labels)."""
        stride = tuple(self.patch_size)

        for vol_idx, vol in enumerate(self._volumes):
            if self.is_2d:
                positions = self._iter_2d_positions(vol.spatial_shape, stride)
            else:
                positions = self._iter_3d_positions(vol.spatial_shape, stride)

            for pos in positions:
                self._patches.append(PatchInfo(
                    volume_index=vol_idx,
                    volume_name=vol.volume_id,
                    position=pos,
                    patch_size=self.patch_size,
                ))

        self._n_labeled_fg = len(self._patches)

    def _try_load_cache(self):
        """Attempt to load patch cache using current validation params."""
        from vesuvius.models.preprocessing.patches import try_load_patch_cache

        # Get valid_patch_value from target config
        valid_patch_value = None
        for target_name in self.target_names:
            info = self.targets.get(target_name, {})
            if 'valid_patch_value' in info:
                valid_patch_value = info['valid_patch_value']
                break

        volume_ids = [vol.volume_id for vol in self._volumes]

        return try_load_patch_cache(
            cache_dir=self.cache_dir,
            data_path=self.data_path,
            volume_ids=volume_ids,
            patch_size=self.patch_size,
            min_labeled_ratio=self.min_labeled_ratio,
            bbox_threshold=self.min_bbox_percent,
            valid_patch_find_resolution=self.valid_patch_find_resolution,
            valid_patch_value=valid_patch_value,
            unlabeled_fg_enabled=self.unlabeled_fg_enabled,
            unlabeled_fg_threshold=self.unlabeled_fg_threshold,
            unlabeled_fg_bbox_threshold=self.unlabeled_fg_bbox_threshold,
        )

    def _load_from_cache(self, cache_data) -> None:
        """Load patches from cache data."""
        # Build volume name -> index mapping
        volume_name_to_idx = {vol.volume_id: idx for idx, vol in enumerate(self._volumes)}

        # Add FG patches
        for entry in cache_data.fg_patches:
            vol_idx = volume_name_to_idx.get(entry.volume_name, entry.volume_idx)
            self._patches.append(PatchInfo(
                volume_index=vol_idx,
                volume_name=entry.volume_name,
                position=entry.position,
                patch_size=self.patch_size,
                is_unlabeled_fg=False,
            ))
        self._n_labeled_fg = len(cache_data.fg_patches)

        # Add unlabeled FG patches
        for entry in cache_data.unlabeled_fg_patches:
            vol_idx = volume_name_to_idx.get(entry.volume_name, entry.volume_idx)
            self._patches.append(PatchInfo(
                volume_index=vol_idx,
                volume_name=entry.volume_name,
                position=entry.position,
                patch_size=self.patch_size,
                is_unlabeled_fg=True,
            ))
        self._n_unlabeled_fg = len(cache_data.unlabeled_fg_patches)

        logger.info(
            "Loaded %d patches from cache (%d labeled, %d unlabeled)",
            len(self._patches), self._n_labeled_fg, self._n_unlabeled_fg
        )

    def _iter_2d_positions(
        self,
        spatial_shape: Tuple[int, int],
        stride: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """Generate all 2D patch positions with given stride."""
        height, width = spatial_shape
        ph, pw = self.patch_size
        sh, sw = stride

        y_positions = list(range(0, max(1, height - ph + 1), sh))
        x_positions = list(range(0, max(1, width - pw + 1), sw))

        return [(y, x) for y in y_positions for x in x_positions]

    def _iter_3d_positions(
        self,
        spatial_shape: Tuple[int, int, int],
        stride: Tuple[int, int, int]
    ) -> List[Tuple[int, int, int]]:
        """Generate all 3D patch positions with given stride."""
        depth, height, width = spatial_shape
        pd, ph, pw = self.patch_size
        sd, sh, sw = stride

        z_positions = list(range(0, max(1, depth - pd + 1), sd))
        y_positions = list(range(0, max(1, height - ph + 1), sh))
        x_positions = list(range(0, max(1, width - pw + 1), sw))

        return [(z, y, x) for z in z_positions for y in y_positions for x in x_positions]

    # -------------------------------------------------------------------------
    # Transforms
    # -------------------------------------------------------------------------

    def _get_skeleton_targets(self) -> tuple[List[str], Dict[str, int]]:
        """Detect targets that need skeleton computation based on loss config.

        Returns
        -------
        tuple[List[str], Dict[str, int]]
            A tuple of (skeleton_targets, skeleton_ignore_values) where:
            - skeleton_targets: List of target names needing skeleton computation
            - skeleton_ignore_values: Dict mapping target_name -> ignore_label value
        """
        SKELETON_LOSSES = {'MedialSurfaceRecall', 'SoftSkeletonRecallLoss', 'DC_SkelREC_and_CE_loss'}
        skeleton_targets = []
        skeleton_ignore_values = {}

        for target_name, target_info in self.targets.items():
            if 'losses' in target_info:
                for loss_cfg in target_info['losses']:
                    if loss_cfg.get('name') in SKELETON_LOSSES:
                        skeleton_targets.append(target_name)
                        # Extract ignore_label if configured
                        ignore_label = target_info.get('ignore_label')
                        if ignore_label is None:
                            ignore_label = target_info.get('ignore_index')
                        if ignore_label is None:
                            ignore_label = target_info.get('ignore_value')
                        if ignore_label is not None:
                            skeleton_ignore_values[target_name] = ignore_label
                        break

        return skeleton_targets, skeleton_ignore_values

    def _initialize_transforms(self) -> None:
        """Initialize augmentation transforms."""
        skeleton_targets, skeleton_ignore_values = self._get_skeleton_targets()

        if self.is_training:
            no_spatial = getattr(self.mgr, 'no_spatial_augmentation', False)
            no_scaling = getattr(self.mgr, 'no_scaling_augmentation', False)

            self.transforms = create_training_transforms(
                patch_size=self.patch_size,
                no_spatial=no_spatial,
                no_scaling=no_scaling,
                skeleton_targets=skeleton_targets if skeleton_targets else None,
                skeleton_ignore_values=skeleton_ignore_values if skeleton_ignore_values else None,
            )
            if self._profile_augmentations:
                self._augmentation_names = collect_augmentation_names(self.transforms)
        elif skeleton_targets:
            # Validation: only apply skeleton generation (no augmentation)
            from vesuvius.models.augmentation.pipelines.training_transforms import create_validation_transforms
            self.transforms = create_validation_transforms(
                skeleton_targets=skeleton_targets,
                skeleton_ignore_values=skeleton_ignore_values if skeleton_ignore_values else None,
            )
            if self._profile_augmentations:
                self._augmentation_names = collect_augmentation_names(self.transforms)

    # -------------------------------------------------------------------------
    # Dataset Interface
    # -------------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._patches)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        patch = self._patches[index]
        vol = self._volumes[patch.volume_index]
        size = patch.patch_size
        slices = tuple(slice(p, p + s) for p, s in zip(patch.position, size))
        pad_or_crop = pad_or_crop_2d if self.is_2d else pad_or_crop_3d

        def mask_neighbor_intrusion(data: np.ndarray) -> np.ndarray:
            safe_boundary = patch.safe_boundary
            if safe_boundary is None:
                return data

            ndim = len(patch.position)
            for axis in range(ndim):
                patch_start = patch.position[axis]
                patch_end = patch_start + data.shape[axis]
                boundary = safe_boundary[axis]

                if patch_end > boundary:
                    intrusion_start = max(0, boundary - patch_start)
                    axis_slices = [slice(None)] * ndim
                    axis_slices[axis] = slice(intrusion_start, None)
                    data[tuple(axis_slices)] = 0

            return data

        def load_array(
            arr: Optional[zarr.Array], *, normalize: bool = False, return_original_shape: bool = False
        ) -> np.ndarray | Tuple[np.ndarray, Tuple[int, ...]]:
            if arr is None:
                data = np.zeros(size, dtype=np.float32)
                original_shape = tuple(0 for _ in size)  # All padded
            else:
                data = np.asarray(arr[slices])
                original_shape = data.shape
                if normalize and self.normalizer is not None:
                    data = self.normalizer.run(data)
                data = pad_or_crop(data.astype(np.float32), size)
            data = mask_neighbor_intrusion(data)
            if return_original_shape:
                return data, original_shape
            return data

        image, original_shape = load_array(vol.image_array, normalize=True, return_original_shape=True)

        # Create padding mask: 1.0 = valid data, 0.0 = padded region
        padding_mask = np.zeros(size, dtype=np.float32)
        valid_slices = tuple(slice(0, min(o, s)) for o, s in zip(original_shape, size))
        padding_mask[valid_slices] = 1.0
        padding_mask = mask_neighbor_intrusion(padding_mask)

        result: Dict[str, torch.Tensor] = {
            'image': torch.from_numpy(image[np.newaxis, ...]),
            'padding_mask': torch.from_numpy(padding_mask[np.newaxis, ...]),
            'patch_info': {
                'volume_name': vol.volume_id,
                'position': patch.position,
            },
        }

        is_unlabeled = True
        for target_name in self.target_names:
            label_arr = vol.label_arrays.get(target_name)
            label_data = load_array(label_arr)
            if label_arr is not None and np.count_nonzero(label_data) > 0:
                is_unlabeled = False
            result[target_name] = torch.from_numpy(label_data[np.newaxis, ...])

        result['is_unlabeled'] = is_unlabeled or patch.is_unlabeled_fg

        if self.transforms is not None:
            if self._profile_augmentations and self._augmentation_names:
                result['_aug_perf'] = {name: 0.0 for name in self._augmentation_names}
            result = self.transforms(**result)

        return result

    # -------------------------------------------------------------------------
    # Semi-supervised Support
    # -------------------------------------------------------------------------

    def get_labeled_unlabeled_patch_indices(self) -> Tuple[List[int], List[int]]:
        """
        Get indices split into labeled and unlabeled patches.

        Returns
        -------
        labeled_indices : List[int]
            Patches with valid labels (for supervised loss)
        unlabeled_indices : List[int]
            Unlabeled foreground patches (for consistency loss)
        """
        labeled = []
        unlabeled = []

        for idx, patch in enumerate(self._patches):
            if patch.is_unlabeled_fg:
                unlabeled.append(idx)
            else:
                # Check if volume has labels
                vol = self._volumes[patch.volume_index]
                if vol.has_labels:
                    labeled.append(idx)
                else:
                    unlabeled.append(idx)

        return labeled, unlabeled

    @property
    def n_fg(self) -> int:
        """Number of foreground (labeled) patches."""
        return self._n_labeled_fg

    @property
    def n_unlabeled_fg(self) -> int:
        """Number of unlabeled foreground patches."""
        return self._n_unlabeled_fg

    @property
    def valid_patches(self) -> List[PatchInfo]:
        """Access to patch list for compatibility."""
        return self._patches

    @property
    def target_volumes(self) -> Dict[str, List[Dict]]:
        """
        Build target_volumes structure for backward compatibility.

        Provided for compatibility with code that expects this format.
        """
        first_target = self.target_names[0] if self.target_names else 'ink'
        volumes_list = []
        for vol in self._volumes:
            volumes_list.append({
                'volume_id': vol.volume_id,
                'data': {
                    'data': vol.image_array,
                    'label': vol.label_arrays.get(first_target),
                }
            })
        return {first_target: volumes_list}
