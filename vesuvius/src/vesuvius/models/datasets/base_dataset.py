import logging
from pathlib import Path
import os
import json
import math
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import fsspec
import zarr
from torch.utils.data import Dataset
from multiprocessing import Pool, cpu_count
from functools import partial
# Augmentation pipeline
from vesuvius.models.augmentation.pipelines import create_training_transforms
from vesuvius.models.augmentation.transforms.utils.compose import ComposeTransforms

from ..training.normalization import get_normalization
from .intensity_properties import initialize_intensity_properties
from .slicers import (
    PlaneSliceConfig,
    PlaneSlicePatch,
    PlaneSliceVolume,
    PlaneSlicer,
    ChunkSliceConfig,
    ChunkPatch,
    ChunkVolume,
    ChunkSlicer,
)

class BaseDataset(Dataset):
    """
    A PyTorch Dataset base class for handling both 2D and 3D data from various sources.
    
    Subclasses must implement the _initialize_volumes() method to specify how
    data is loaded from their specific data source.
    """
    def __init__(self,
                 mgr,
                 is_training=True):
        """
        Initialize the dataset with configuration from the manager.
        
        Parameters
        ----------
        mgr : ConfigManager
            Manager containing configuration parameters
        is_training : bool
            Whether this dataset is for training (applies augmentations) or validation

        """
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.mgr = mgr
        self.is_training = is_training
        self._preserve_label_dtype = getattr(self, "_preserve_label_dtype", False)

        self.model_name = mgr.model_name
        self.targets = mgr.targets               # e.g. {"ink": {...}, "normals": {...}}
        self._vector_name_tokens = (
            "normal",
            "normals",
            "t_u",
            "t_v",
            "uv",
            "frame",
            "vector",
        )
        self._vector_target_names: Set[str] = self._discover_vector_targets()
        self._vector_target_lower: Set[str] = {name.lower() for name in self._vector_target_names}
        self.patch_size = mgr.train_patch_size   # Expected to be [z, y, x]
        self.min_labeled_ratio = mgr.min_labeled_ratio
        self.min_bbox_percent = mgr.min_bbox_percent
        
        # if you are certain your data contains dense labels (everything is labeled), you can choose
        # to skip the valid patch finding
        self.skip_patch_validation = getattr(mgr, 'skip_patch_validation', False)

        # for semi-supervised workflows, unlabeled data is obviously needed,
        # we want a flag for this so in fully supervised workflows we can assert that all images have
        # corresponding labels (so we catch it early)
        self.allow_unlabeled_data = getattr(mgr, 'allow_unlabeled_data', False)

        # Initialize normalization (will be set after computing intensity properties)
        self.normalization_scheme = getattr(mgr, 'normalization_scheme', 'zscore')
        self.intensity_properties = getattr(mgr, 'intensity_properties', {})
        self.normalizer = None  # Will be initialized after volumes are loaded

        self.target_volumes = {}
        self.valid_patches = []
        self.patch_weights = None
        self._plane_patches: List[PlaneSlicePatch] = []
        self._chunk_patches: List[ChunkPatch] = []
        self.is_2d_dataset = None
        self.data_path = Path(mgr.data_path) if hasattr(mgr, 'data_path') else None
        self.zarr_arrays = []
        self.zarr_names = []
        self.data_paths = []

        # Slice sampling configuration (2D slices from 3D inputs)
        self.slice_sampling_enabled = bool(getattr(mgr, 'slice_sampling_enabled', False))
        self.slice_sample_planes = list(getattr(mgr, 'slice_sample_planes', []))
        self.slice_plane_weights = dict(getattr(mgr, 'slice_plane_weights', {}))
        self.slice_plane_patch_sizes = dict(getattr(mgr, 'slice_plane_patch_sizes', {}))
        self.slice_primary_plane = getattr(mgr, 'slice_primary_plane', None)
        self.slice_random_rotation_planes = dict(getattr(mgr, 'slice_random_rotation_planes', {}))
        self.slice_random_tilt_planes = dict(getattr(mgr, 'slice_random_tilt_planes', {}))
        self.slice_label_interpolation = getattr(mgr, 'slice_label_interpolation', {})
        self.slice_save_plane_masks = bool(getattr(mgr, 'slice_save_plane_masks', False))
        self.slice_plane_mask_mode = getattr(mgr, 'slice_plane_mask_mode', 'plane')
        if self.slice_plane_mask_mode not in ('volume', 'plane'):
            self.slice_plane_mask_mode = 'plane'

        self.plane_slicer: Optional[PlaneSlicer] = None
        self.chunk_slicer: Optional[ChunkSlicer] = None

        if self.slice_sampling_enabled and not self.slice_sample_planes:
            self.slice_sample_planes = ['z']
            self.slice_plane_weights = {'z': 1.0}

        # Disable cached patch lookup in slice mode (cache incompatible with plane metadata)
        if self.slice_sampling_enabled and getattr(mgr, 'cache_valid_patches', True):
            self.logger.info("Slice sampling mode: disabling cache_valid_patches (not supported)")
            self.cache_enabled = False
        else:
            self.cache_enabled = getattr(mgr, 'cache_valid_patches', True)

        self.cache_dir = None
        if self.data_path is not None:
            self.cache_dir = self.data_path / '.patches_cache'
            self.logger.info("Cache directory: %s", self.cache_dir)
            self.logger.info("Cache enabled: %s", self.cache_enabled)

        self._initialize_volumes()

        if self.slice_sampling_enabled:
            self._setup_plane_slicer()
            self._setup_chunk_slicer()  # Also set up chunk slicer for fast 3D validation
        else:
            self._setup_chunk_slicer()

        ref_target = list(self.target_volumes.keys())[0]
        ref_entry = self.target_volumes[ref_target][0]
        ref_source = self._get_entry_label(ref_entry)
        if ref_source is None:
            ref_source = self._get_entry_image(ref_entry)
        if ref_source is None:
            raise ValueError("Unable to determine reference shape; volume lacks image and label sources")
        ref_shape = self._resolve_spatial_shape(ref_source)

        # Allow explicit override from config to avoid misclassification.
        force_2d = getattr(self.mgr, 'force_2d', False)
        if self.slice_sampling_enabled:
            force_2d = True
        if not force_2d and hasattr(self.mgr, 'dataset_config'):
            force_2d = bool(self.mgr.dataset_config.get('force_2d', False))
        force_3d = getattr(self.mgr, 'force_3d', False)
        if self.slice_sampling_enabled:
            force_3d = False
        if not force_3d and hasattr(self.mgr, 'dataset_config'):
            force_3d = bool(self.mgr.dataset_config.get('force_3d', False))

        if force_2d and force_3d:
            raise ValueError("Both force_2d and force_3d are set; choose only one.")

        if force_2d:
            self.is_2d_dataset = True
        elif force_3d:
            self.is_2d_dataset = False
        else:
            # Only treat as 2D when the data truly has 2 dimensions
            self.is_2d_dataset = (len(ref_shape) == 2)
        
        if self.is_2d_dataset:
            print("Detected 2D dataset")
            if len(self.patch_size) == 3:
                self.patch_size = list(self.patch_size[-2:])
                print(f"Adjusted patch size for 2D data: {self.patch_size}")
        else:
            print("Detected 3D dataset")

        # Check if we should skip intensity sampling
        skip_intensity_sampling = getattr(mgr, 'skip_intensity_sampling', False)

        if skip_intensity_sampling:
            print("Skipping intensity sampling as requested")
            # Use default values if intensity properties not provided
            if not self.intensity_properties:
                self.intensity_properties = {
                    'mean': 0.0,
                    'std': 1.0,
                    'min': 0.0,
                    'max': 1.0,
                    'percentile_00_5': 0.0,
                    'percentile_99_5': 1.0
                }
        else:
            self.intensity_properties = initialize_intensity_properties(
                target_volumes=self.target_volumes,
                normalization_scheme=self.normalization_scheme,
                existing_properties=self.intensity_properties,
                cache_enabled=self.cache_enabled,
                cache_dir=self.cache_dir,
                mgr=self.mgr,
                sample_ratio=0.001,
                max_samples=1000000
            )

        self.normalizer = get_normalization(self.normalization_scheme, self.intensity_properties)
        if self.plane_slicer is not None:
            self.plane_slicer.set_normalizer(self.normalizer)
        if self.chunk_slicer is not None:
            self.chunk_slicer.set_normalizer(self.normalizer)

        self.transforms = None
        if self.is_training:
            self.transforms = self._create_training_transforms()
            print("Training transforms initialized")
        else:
            # For validation, we might still need skeleton transform
            self.transforms = self._create_validation_transforms()
            if self.transforms is not None:
                print("Validation transforms initialized")

        # Check if patches will be inherited from another dataset (e.g., train -> val)
        self.inherit_patches_externally = getattr(mgr, 'inherit_patches_externally', False)

        if self.inherit_patches_externally:
            # Skip all patch indexing - patches will be copied from another dataset
            print("Patches will be inherited from training dataset")
            self._setup_chunk_slicer()  # Still need slicer for extraction
        elif not self.skip_patch_validation:
            self._get_valid_patches()
        else:
            print("Skipping patch validation as requested")
            if self.slice_sampling_enabled:
                self._generate_all_slice_patches()
            else:
                self._build_chunk_index(validate=False)

    def _initialize_volumes(self):
        """
        Initialize volumes from the data source.
        
        This method must be implemented by subclasses to specify how
        data is loaded from their specific data source (napari, TIFs, Zarr, etc.).
        
        The implementation should:
        1. Populate self.target_volumes in the format:
           {
               'target_name': [
                   {
                       'volume_id': str,
                       'image': ArrayHandle | np.ndarray,
                       'label': ArrayHandle | np.ndarray | None,
                       'label_path': str | None,
                       'label_source': object | None,
                       'has_label': bool,
                   },
                   ...
               ],
               ...
           }

        2. Optionally populate zarr metadata used for cache compatibility:
           - self.zarr_arrays: Sequence of underlying zarr arrays (if available)
           - self.zarr_names: Names for each stored label volume
           - self.data_paths: Source paths for each stored label volume
        """
        raise NotImplementedError("Subclasses must implement _initialize_volumes() method")

    def _setup_plane_slicer(self) -> None:
        if not self.slice_sampling_enabled:
            return

        if not self.slice_sample_planes:
            raise ValueError("slice_sample_planes must define at least one plane when slice sampling is enabled")

        target_names = list(self.target_volumes.keys())
        if not target_names:
            raise ValueError("Plane slicing requires target volumes to be populated")

        plane_patch_sizes: Dict[str, Sequence[int]] = {}
        for axis in self.slice_sample_planes:
            override = self.slice_plane_patch_sizes.get(axis)
            if override is None:
                raise KeyError(f"slice_plane_patch_sizes missing entry for plane '{axis}'")
            if len(override) != 2:
                raise ValueError(f"Patch size for plane '{axis}' must have length 2; got {override}")
            plane_patch_sizes[axis] = tuple(int(v) for v in override)

        plane_weights = {axis: float(self.slice_plane_weights.get(axis, 1.0)) for axis in self.slice_sample_planes}

        config = PlaneSliceConfig(
            sample_planes=tuple(self.slice_sample_planes),
            plane_weights=plane_weights,
            plane_patch_sizes=plane_patch_sizes,
            primary_plane=self.slice_primary_plane,
            min_labeled_ratio=float(self.min_labeled_ratio),
            min_bbox_percent=float(self.min_bbox_percent),
            allow_unlabeled=bool(self.allow_unlabeled_data),
            random_rotation_planes=dict(self.slice_random_rotation_planes),
            random_tilt_planes=dict(self.slice_random_tilt_planes),
            label_interpolation=dict(self.slice_label_interpolation),
            save_plane_masks=bool(self.slice_save_plane_masks),
            plane_mask_mode=self.slice_plane_mask_mode,
        )

        slicer = PlaneSlicer(config=config, target_names=target_names)

        first_target = target_names[0]
        reference_volumes = self.target_volumes[first_target]

        for idx, reference_info in enumerate(reference_volumes):
            image_array = self._get_entry_image(reference_info)
            volume_name = reference_info.get('volume_id', f"volume_{idx}")

            labels: Dict[str, Optional[np.ndarray]] = {}
            for target_name in target_names:
                try:
                    entry = self.target_volumes[target_name][idx]
                except IndexError as exc:
                    raise IndexError(
                        f"Target '{target_name}' missing volume index {idx} required for plane slicing"
                    ) from exc
                labels[target_name] = self._get_entry_label(entry)

            slicer.register_volume(
                PlaneSliceVolume(
                    index=idx,
                    name=volume_name,
                    image=image_array,
                    labels=labels,
                    meshes=reference_info.get('meshes', {}),
                )
            )

        self.plane_slicer = slicer

    def _setup_chunk_slicer(self) -> None:
        target_names = list(self.target_volumes.keys())
        if not target_names:
            raise ValueError("Chunk slicing requires target volumes to be populated")

        patch_size = tuple(int(v) for v in self.patch_size)
        # When in 2D slice mode, derive 3D patch size for validation
        if len(patch_size) == 2 and self.slice_sampling_enabled:
            cube_dim = min(patch_size)
            patch_size = (cube_dim, cube_dim, cube_dim)
            self.logger.info("Derived 3D validation patch size %s from 2D patch size", patch_size)
        stride_override = getattr(self.mgr, 'chunk_stride', None)
        if stride_override is not None:
            stride_override = tuple(int(v) for v in stride_override)

        channel_selector = self._resolve_label_channel_selector(target_names)
        ignore_map = self._resolve_target_ignore_labels(target_names)
        valid_patch_value = self._resolve_valid_patch_value(target_names)

        # Get unlabeled foreground config
        unlabeled_fg_enabled = bool(getattr(self.mgr, 'unlabeled_foreground_enabled', False))
        unlabeled_fg_threshold = float(getattr(self.mgr, 'unlabeled_foreground_threshold', 0.05))
        unlabeled_fg_bbox_threshold = float(getattr(self.mgr, 'unlabeled_foreground_bbox_threshold', 0.15))
        unlabeled_fg_volumes = getattr(self.mgr, 'unlabeled_foreground_volumes', None)
        unlabeled_fg_volume_ids: Optional[Set[str]] = None
        if unlabeled_fg_volumes is not None:
            unlabeled_fg_volume_ids = set(unlabeled_fg_volumes)

        config = ChunkSliceConfig(
            patch_size=patch_size,
            stride=stride_override,
            min_labeled_ratio=float(self.min_labeled_ratio),
            min_bbox_percent=float(self.min_bbox_percent),
            allow_unlabeled=bool(self.allow_unlabeled_data),
            valid_patch_find_resolution=int(getattr(self.mgr, 'valid_patch_find_resolution', 1)),
            num_workers=int(getattr(self.mgr, 'num_workers', 8)),
            cache_enabled=bool(self.cache_enabled),
            cache_dir=self.cache_dir,
            label_channel_selector=channel_selector,
            valid_patch_value=valid_patch_value,
            bg_sampling_enabled=bool(getattr(self.mgr, 'bg_sampling_enabled', False)),
            bg_to_fg_ratio=float(getattr(self.mgr, 'bg_to_fg_ratio', 0.5)),
            # Unlabeled foreground detection for semi-supervised learning
            unlabeled_fg_enabled=unlabeled_fg_enabled,
            unlabeled_fg_threshold=unlabeled_fg_threshold,
            unlabeled_fg_bbox_threshold=unlabeled_fg_bbox_threshold,
            unlabeled_fg_volume_ids=unlabeled_fg_volume_ids,
        )

        slicer = ChunkSlicer(config=config, target_names=target_names)

        first_target = target_names[0]
        reference_volumes = self.target_volumes[first_target]

        for idx, reference_info in enumerate(reference_volumes):
            image_array = self._get_entry_image(reference_info)
            volume_name = reference_info.get('volume_id', f"volume_{idx}")

            labels: Dict[str, Optional[np.ndarray]] = {}
            label_source = None
            cache_key_path: Optional[Path] = None
            label_ignore_value: Optional[Union[int, float]] = None

            for target_name in target_names:
                try:
                    entry = self.target_volumes[target_name][idx]
                except IndexError as exc:
                    raise IndexError(
                        f"Target '{target_name}' missing volume index {idx} required for chunk slicing"
                    ) from exc
                label_array = self._get_entry_label(entry)
                labels[target_name] = label_array
                ignore_candidate = ignore_map.get(target_name)
                source_candidate = self._get_entry_label_source(entry)
                if source_candidate is None:
                    source_candidate = label_array
                if label_source is None and source_candidate is not None:
                    label_source = source_candidate
                    if label_ignore_value is None and ignore_candidate is not None:
                        label_ignore_value = ignore_candidate
                elif label_ignore_value is None and ignore_candidate is not None:
                    label_ignore_value = ignore_candidate
                if cache_key_path is None:
                    path_candidate = self._get_entry_label_path(entry)
                    if path_candidate:
                        cache_key_path = Path(path_candidate)

            # Fall back to image path for cache key if no labels (for unlabeled datasets)
            if cache_key_path is None:
                if hasattr(image_array, 'path'):
                    cache_key_path = Path(image_array.path)
                elif hasattr(image_array, 'store') and hasattr(image_array.store, 'path'):
                    cache_key_path = Path(image_array.store.path)

            # Assign per-volume image source for unlabeled FG detection
            image_source = None
            if unlabeled_fg_enabled:
                # If specific volumes are listed, only use those; otherwise use all volumes
                should_use_volume = (
                    unlabeled_fg_volume_ids is None or volume_name in unlabeled_fg_volume_ids
                )
                if should_use_volume:
                    # Use the already-loaded image array directly
                    # This works for self-supervised training where we don't have labels
                    image_source = image_array
                    self.logger.info("Using image array for unlabeled FG detection (volume=%s)", volume_name)

            slicer.register_volume(
                ChunkVolume(
                    index=idx,
                    name=volume_name,
                    image=image_array,
                    labels=labels,
                    label_source=label_source,
                    cache_key_path=cache_key_path,
                    label_ignore_value=label_ignore_value,
                    meshes=reference_info.get('meshes', {}),
                    image_source=image_source,
                )
            )

        self.chunk_slicer = slicer

    def _resolve_label_channel_selector(
        self, target_names: Sequence[str]
    ) -> Optional[Union[int, Tuple[int, ...]]]:
        for target_name in target_names:
            info = self.targets.get(target_name) or {}
            selector = info.get('valid_patch_channel')
            if selector is not None:
                return self._normalize_channel_selector(selector)
        return None

    def _resolve_target_ignore_labels(
        self, target_names: Sequence[str]
    ) -> Dict[str, Union[int, float]]:
        ignore_map: Dict[str, Union[int, float]] = {}
        for target_name in target_names:
            info = self.targets.get(target_name) or {}
            for alias in ("ignore_index", "ignore_label", "ignore_value"):
                if alias in info:
                    value = info.get(alias)
                    if value is not None:
                        ignore_map[target_name] = value  # store first non-null alias per target
                        break
        return ignore_map

    def _resolve_valid_patch_value(
        self, target_names: Sequence[str]
    ) -> Optional[Union[int, float]]:
        """Resolve valid_patch_value from target config.

        When set, only voxels matching this value count as labeled for patch validation
        (instead of all non-zero values).
        """
        for target_name in target_names:
            info = self.targets.get(target_name) or {}
            value = info.get('valid_patch_value')
            if value is not None:
                return value
        return None

    def _derive_image_path_from_label(self, label_path: Path, target_name: str) -> Path:
        """Derive image zarr path from label path.

        Transforms: labels/sample_ink.zarr -> images/sample.zarr
        Uses configured label_dirname and image_dirname for directory swap.
        """
        label_dirname = getattr(self.mgr, "label_dirname", "labels")
        image_dirname = getattr(self.mgr, "image_dirname", "images")

        # Swap label dir -> image dir
        parts = list(label_path.parts)
        try:
            idx = parts.index(label_dirname)
            parts[idx] = image_dirname
        except ValueError:
            pass  # No label dir found, keep as-is

        # Remove target suffix from stem
        stem = label_path.stem
        suffix = f"_{target_name}"
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]

        # Reconstruct path
        new_path = Path(*parts[:-1]) / f"{stem}{label_path.suffix}"
        return new_path

    @staticmethod
    def _normalize_channel_selector(
        selector: object,
    ) -> Optional[Union[int, Tuple[int, ...]]]:
        if selector is None:
            return None
        if isinstance(selector, int):
            return int(selector)
        if isinstance(selector, (list, tuple)):
            if not selector:
                raise ValueError("valid_patch_channel cannot be an empty sequence")
            return tuple(int(v) for v in selector)
        if isinstance(selector, dict):
            if 'flatten_index' in selector:
                return int(selector['flatten_index'])
            if 'index' in selector:
                value = selector['index']
            elif 'indices' in selector:
                value = selector['indices']
            else:
                raise ValueError(
                    "valid_patch_channel dict must contain 'flatten_index', 'index', or 'indices'"
                )
            if isinstance(value, int):
                return int(value)
            if isinstance(value, (list, tuple)):
                if not value:
                    raise ValueError("valid_patch_channel indices cannot be empty")
                return tuple(int(v) for v in value)
            raise TypeError(
                "valid_patch_channel 'index'/'indices' must be int or sequence of ints"
            )
        raise TypeError("valid_patch_channel must be int, sequence of ints, or dict")

    def _build_chunk_index(self, *, validate: bool) -> None:
        if self.chunk_slicer is None:
            self._setup_chunk_slicer()
        if self.chunk_slicer is None:
            raise RuntimeError("Chunk slicer failed to initialise")

        patches, weights = self.chunk_slicer.build_index(validate=validate)
        self._chunk_patches = patches
        self.patch_weights = weights
        self.valid_patches = [
            {
                "volume_index": patch.volume_index,
                "volume_name": patch.volume_name,
                "position": list(patch.position),
                "patch_size": list(patch.patch_size),
            }
            for patch in patches
        ]

        volume_count = len({patch.volume_index for patch in patches})
        mode = "validated" if validate else "enumerated"
        self.logger.info(
            "Prepared %s %s chunk patches across %s volume(s)",
            len(patches),
            mode,
            volume_count,
        )

    def _needs_skeleton_transform(self):
        """
        Check if any configured loss requires skeleton data.
        
        Returns
        -------
        bool
            True if skeleton transform should be added to the pipeline
        """
        return bool(self._skeleton_loss_targets())

    def _skeleton_loss_targets(self):
        """
        Identify targets that need skeletonised ground truth.
        """
        skeleton_losses = {'MedialSurfaceRecall', 'DC_SkelREC_and_CE_loss', 'SoftSkeletonRecallLoss'}
        skeleton_targets = []

        for target_name, target_info in self.targets.items():
            if "losses" not in target_info:
                continue
            for loss_cfg in target_info["losses"]:
                loss_name = loss_cfg.get("name")
                if loss_name in skeleton_losses:
                    skeleton_targets.append(target_name)
                    break

        return tuple(skeleton_targets)

    def _get_valid_patches(self):
        """Find valid patches based on labeled ratio requirements."""
        if self.slice_sampling_enabled:
            if self.plane_slicer is None:
                raise RuntimeError("Plane slicer not initialized despite slice_sampling_enabled=True")
            if self.chunk_slicer is None:
                raise RuntimeError("Chunk slicer not initialized despite slice_sampling_enabled=True")

            # Use fast 3D validation via chunk_slicer - slices sampled on-the-fly during __getitem__
            self._build_chunk_index(validate=not self.skip_patch_validation)

            # Use 3D chunks directly - we'll sample 2D slices from them at training time
            chunk_patches = self.chunk_slicer.patches
            self.valid_patches = [
                {
                    "volume_index": p.volume_index,
                    "volume_name": p.volume_name,
                    "position": list(p.position),
                    "patch_size": list(p.patch_size),
                }
                for p in chunk_patches
            ]
            self.patch_weights = self.chunk_slicer.weights
            volume_count = len({p.volume_index for p in chunk_patches}) or 1
            self.logger.info(
                "Using %s validated 3D chunks for 2D slice sampling across %s volume(s)",
                len(self.valid_patches),
                volume_count,
            )
            return

        self._build_chunk_index(validate=not self.skip_patch_validation)

    def _sample_plane_from_chunk(self, chunk_info: Dict) -> PlaneSlicePatch:
        """Sample a random 2D plane patch from a validated 3D chunk.

        Randomly selects a plane (z, y, x) and slice index within the chunk,
        returning a PlaneSlicePatch that can be extracted by the plane_slicer.
        """
        import random

        plane_config = self.plane_slicer.config
        sample_planes = list(plane_config.sample_planes)

        # Randomly pick a plane based on weights
        plane_weights = [plane_config.plane_weights.get(p, 1.0) for p in sample_planes]
        total_weight = sum(plane_weights)
        if total_weight > 0:
            plane_weights = [w / total_weight for w in plane_weights]
            axis = random.choices(sample_planes, weights=plane_weights, k=1)[0]
        else:
            axis = random.choice(sample_planes)

        z, y, x = chunk_info['position']
        dz, dy, dx = chunk_info['patch_size']
        patch_size = tuple(plane_config.plane_patch_sizes[axis])

        # Pick a random slice index within the chunk
        if axis == "z":
            slice_idx = random.randint(z, z + dz - 1)
            position = (y, x)
        elif axis == "y":
            slice_idx = random.randint(y, y + dy - 1)
            position = (z, x)
        else:  # x
            slice_idx = random.randint(x, x + dx - 1)
            position = (z, y)

        return PlaneSlicePatch(
            volume_index=chunk_info['volume_index'],
            volume_name=chunk_info['volume_name'],
            plane=axis,
            slice_index=slice_idx,
            position=position,
            patch_size=patch_size,
        )

    def _generate_plane_patches_from_3d(self) -> Tuple[List[PlaneSlicePatch], List[float]]:
        """Generate 2D plane patches from validated 3D chunk positions.

        For each validated 3D chunk, generates 2D patches along the configured
        sample planes (z, y, x) that slice through the valid 3D region.
        """
        if self.chunk_slicer is None or self.plane_slicer is None:
            raise RuntimeError("Both chunk_slicer and plane_slicer must be initialized")

        plane_config = self.plane_slicer.config
        chunk_patches = self.chunk_slicer.patches

        plane_patches: Dict[str, List[PlaneSlicePatch]] = {axis: [] for axis in plane_config.sample_planes}

        for chunk in chunk_patches:
            z, y, x = chunk.position
            dz, dy, dx = chunk.patch_size

            for axis in plane_config.sample_planes:
                patch_size = tuple(plane_config.plane_patch_sizes[axis])

                if axis == "z":
                    # Sample z-slices through this 3D region
                    for slice_idx in range(z, z + dz):
                        plane_patches["z"].append(PlaneSlicePatch(
                            volume_index=chunk.volume_index,
                            volume_name=chunk.volume_name,
                            plane="z",
                            slice_index=slice_idx,
                            position=(y, x),
                            patch_size=patch_size,
                        ))
                elif axis == "y":
                    # Sample y-slices through this 3D region
                    for slice_idx in range(y, y + dy):
                        plane_patches["y"].append(PlaneSlicePatch(
                            volume_index=chunk.volume_index,
                            volume_name=chunk.volume_name,
                            plane="y",
                            slice_index=slice_idx,
                            position=(z, x),
                            patch_size=patch_size,
                        ))
                elif axis == "x":
                    # Sample x-slices through this 3D region
                    for slice_idx in range(x, x + dx):
                        plane_patches["x"].append(PlaneSlicePatch(
                            volume_index=chunk.volume_index,
                            volume_name=chunk.volume_name,
                            plane="x",
                            slice_index=slice_idx,
                            position=(z, y),
                            patch_size=patch_size,
                        ))

        # Flatten and compute weights
        all_patches: List[PlaneSlicePatch] = []
        axis_counts: Dict[str, int] = {}
        for axis, patches in plane_patches.items():
            if patches:
                all_patches.extend(patches)
                axis_counts[axis] = len(patches)

        if not all_patches:
            raise RuntimeError("No plane patches generated from validated 3D regions")

        # Compute weights based on plane_weights config
        weights: List[float] = []
        for patch in all_patches:
            axis_weight = float(plane_config.plane_weights.get(patch.plane, 1.0))
            count = axis_counts.get(patch.plane, 0)
            weight_per_patch = axis_weight / count if count > 0 and axis_weight > 0 else 0.0
            weights.append(weight_per_patch)

        return all_patches, weights

    def __len__(self):
        return len(self.valid_patches)

    @property
    def n_fg(self) -> int:
        """Number of foreground patches (patches with labels)."""
        if hasattr(self, 'chunk_slicer') and hasattr(self.chunk_slicer, 'n_fg'):
            return self.chunk_slicer.n_fg
        return len(self)

    @property
    def n_unlabeled_fg(self) -> int:
        """Number of unlabeled foreground patches (has image data but no labels)."""
        if hasattr(self, 'chunk_slicer') and hasattr(self.chunk_slicer, 'n_unlabeled_fg'):
            return self.chunk_slicer.n_unlabeled_fg
        return 0

    def _resolve_spatial_shape(self, source) -> Tuple[int, ...]:
        if hasattr(source, 'spatial_shape'):
            return tuple(int(v) for v in source.spatial_shape)
        shape = getattr(source, 'shape', None)
        if shape is not None:
            if callable(shape):
                shape = shape()
            return tuple(int(v) for v in shape)
        array = np.asarray(source)
        return tuple(int(v) for v in array.shape)

    @staticmethod
    def _get_entry_image(entry):
        if 'image' in entry:
            return entry['image']
        data = entry.get('data')
        if isinstance(data, dict):
            return data.get('data')
        return None

    @staticmethod
    def _get_entry_label(entry):
        if 'label' in entry:
            return entry['label']
        data = entry.get('data')
        if isinstance(data, dict):
            return data.get('label')
        return None

    @staticmethod
    def _get_entry_label_source(entry):
        if 'label_source' in entry:
            return entry['label_source']
        data = entry.get('data')
        if isinstance(data, dict):
            return data.get('label_source')
        return None

    @staticmethod
    def _get_entry_label_path(entry):
        if 'label_path' in entry:
            return entry['label_path']
        data = entry.get('data')
        if isinstance(data, dict):
            return data.get('label_path')
        return None

    def _normalize_vector_target_names(self, values) -> Set[str]:
        result: Set[str] = set()
        if values is None:
            return result
        if isinstance(values, (list, tuple, set)):
            for item in values:
                if item is None:
                    continue
                result.add(str(item))
        elif isinstance(values, str):
            result.add(values)
        return result

    def _discover_vector_targets(self) -> Set[str]:
        targets: Set[str] = set()

        dataset_cfg = getattr(self.mgr, 'dataset_config', {}) or {}
        targets.update(self._normalize_vector_target_names(dataset_cfg.get('vector_targets')))

        mgr_vector = getattr(self.mgr, 'vector_targets', None)
        if mgr_vector is not None:
            targets.update(self._normalize_vector_target_names(mgr_vector))

        for name, info in (self.targets or {}).items():
            if self._target_info_is_vector(info):
                targets.add(name)
            elif any(token in name.lower() for token in self._vector_name_tokens):
                targets.add(name)

        return targets

    @staticmethod
    def _target_info_is_vector(info) -> bool:
        if not isinstance(info, dict):
            return False
        for key in ('vector', 'is_vector'):
            value = info.get(key)
            if isinstance(value, bool) and value:
                return True
        for key in ('data_kind', 'kind', 'type', 'representation', 'role'):
            value = info.get(key)
            if isinstance(value, str) and value.lower() == 'vector':
                return True
        return False

    def _is_vector_target(self, target_name: str) -> bool:
        name_lower = target_name.lower()
        if name_lower in self._vector_target_lower:
            return True
        if any(token in name_lower for token in self._vector_name_tokens):
            return True
        info = self.targets.get(target_name)
        if info is None and name_lower in self.targets:
            info = self.targets.get(name_lower)
        return self._target_info_is_vector(info)

    def _should_skip_spatial_transforms(
        self,
        label_patches: Mapping[str, np.ndarray],
        mesh_payloads: Mapping[str, Dict[str, object]],
    ) -> bool:
        if mesh_payloads:
            return True
        for target_name in label_patches:
            if self._is_vector_target(target_name):
                return True
        return False

    def _extract_chunk_patch(self, chunk_patch: ChunkPatch) -> Dict[str, torch.Tensor]:
        if self.chunk_slicer is None:
            raise RuntimeError("Chunk slicer not initialized")
        chunk_result = self.chunk_slicer.extract(chunk_patch, normalizer=self.normalizer)

        label_patches: Dict[str, np.ndarray] = {}
        for name, array in chunk_result.labels.items():
            arr = np.ascontiguousarray(array)
            if not self._preserve_label_dtype:
                arr = arr.astype(np.float32, copy=False)
            label_patches[name] = arr

        data_dict: Dict[str, torch.Tensor] = {
            'image': torch.from_numpy(chunk_result.image),
            'is_unlabeled': chunk_result.is_unlabeled,
        }

        for target_name, array in label_patches.items():
            data_dict[target_name] = torch.from_numpy(array)

        mesh_payloads = chunk_result.meshes or {}
        if mesh_payloads:
            data_dict['meshes'] = mesh_payloads

        should_skip = self._should_skip_spatial_transforms(label_patches, mesh_payloads)
        if should_skip:
            data_dict['_skip_spatial_transforms'] = True
            vector_targets = [name for name in label_patches if self._is_vector_target(name)]
            if vector_targets:
                data_dict['vector_targets'] = vector_targets

        patch_info = dict(chunk_result.patch_info)
        if chunk_patch.weight is not None:
            patch_info['weight'] = float(chunk_patch.weight)
        data_dict['patch_info'] = patch_info

        return data_dict

    def _create_training_transforms(self):
        """
        Create training augmentation transforms using the standard pipeline.
        """
        no_spatial = getattr(self.mgr, 'no_spatial', False)
        only_spatial_and_intensity = getattr(self.mgr, 'only_spatial_and_intensity', False)
        allowed_rotation_axes = getattr(self.mgr, 'allowed_rotation_axes', None)

        # Get skeleton targets and their ignore values
        skeleton_targets = self._skeleton_loss_targets()
        skeleton_ignore_values = None
        if skeleton_targets:
            skeleton_ignore_values = {}
            for target_name in skeleton_targets:
                cfg = self.targets.get(target_name, {}) if isinstance(self.targets, dict) else {}
                for alias in ("ignore_index", "ignore_label", "ignore_value"):
                    value = cfg.get(alias)
                    if value is not None:
                        skeleton_ignore_values[target_name] = value
                        break

        return create_training_transforms(
            patch_size=self.mgr.train_patch_size,
            no_spatial=no_spatial,
            only_spatial_and_intensity=only_spatial_and_intensity,
            allowed_rotation_axes=allowed_rotation_axes,
            skeleton_targets=skeleton_targets,
            skeleton_ignore_values=skeleton_ignore_values,
        )

    def _create_validation_transforms(self):
        """
        Create validation transforms.
        For validation, we only apply skeleton transform if needed (no augmentations).
        """
        skeleton_targets = self._skeleton_loss_targets()
        if not skeleton_targets:
            return None

        from vesuvius.models.augmentation.transforms.utils.skeleton_transform import MedialSurfaceTransform
        
        transforms = []
        ignore_values = {}
        for target_name in skeleton_targets:
            cfg = self.targets.get(target_name, {}) if isinstance(self.targets, dict) else {}
            for alias in ("ignore_index", "ignore_label", "ignore_value"):
                value = cfg.get(alias)
                if value is not None:
                    ignore_values[target_name] = value
                    break
        transforms.append(
            MedialSurfaceTransform(
                do_tube=False,
                target_keys=skeleton_targets,
                ignore_values=ignore_values or None,
            )
        )
        print(f"Added MedialSurfaceTransform to validation pipeline for targets: {', '.join(skeleton_targets)}")
        
        return ComposeTransforms(transforms)
    
    def __getitem__(self, index):
        """
        Returns a dictionary with the following format:
        {
            'image': torch.Tensor,           # Shape: [C, H, W] (2D) or [C, D, H, W] (3D)
            'is_unlabeled': bool,            # True if patch has no valid labels
            'target_name_1': torch.Tensor,   # Shape: [C, H, W] (2D) or [C, D, H, W] (3D)
            'target_name_2': torch.Tensor,   # Additional targets as configured
            ...                              # (e.g., 'ink', 'normals', 'distance_transform', etc.)
        }
        
        Where:
        - C = number of channels (usually 1 for image, can vary for targets)
        - H, W = height and width of patch
        - D = depth of patch (3D only)
        - All tensors are float32
        - Labels are 0 for background, >0 for foreground
        - Unlabeled patches will have all-zero label tensors
        """

        if not self.valid_patches:
            raise IndexError("Dataset contains no valid patches. Check data paths and patch generation settings.")
        if index >= len(self.valid_patches) or index < -len(self.valid_patches):
            # Allow optional wrap-around if explicitly enabled in config
            wrap = bool(getattr(self.mgr, 'wrap_indices', False)) or \
                   bool(getattr(self.mgr, 'wrap_dataset_indices', False)) or \
                   (hasattr(self.mgr, 'dataset_config') and bool(self.mgr.dataset_config.get('wrap_indices', False)))
            if wrap:
                index = index % len(self.valid_patches)
            else:
                raise IndexError(f"Index {index} out of range for dataset of length {len(self.valid_patches)}")
        if self.slice_sampling_enabled:
            # Sample a random 2D slice from the validated 3D chunk
            chunk_info = self.valid_patches[index]
            plane_patch = self._sample_plane_from_chunk(chunk_info)
            plane_result = self.plane_slicer.extract(plane_patch)
            data_dict = {
                'image': torch.from_numpy(plane_result.image),
                'is_unlabeled': plane_result.is_unlabeled,
            }
            for target_name, label_array in plane_result.labels.items():
                data_dict[target_name] = torch.from_numpy(label_array)

            if plane_result.plane_mask is not None:
                mask_array = plane_result.plane_mask.astype(np.uint8, copy=False)
                data_dict['plane_mask'] = torch.from_numpy(mask_array)

            data_dict['patch_info'] = plane_result.patch_info

            mesh_payloads = plane_result.meshes or {}
            if mesh_payloads:
                data_dict['meshes'] = mesh_payloads

            vector_targets = [name for name in plane_result.labels if self._is_vector_target(name)]
            if mesh_payloads or vector_targets:
                data_dict['_skip_spatial_transforms'] = True
                if vector_targets:
                    data_dict['vector_targets'] = vector_targets
        else:
            if not self._chunk_patches:
                raise RuntimeError("Chunk slicer index not prepared")
            chunk_patch = self._chunk_patches[index]
            data_dict = self._extract_chunk_patch(chunk_patch)


        metadata = data_dict.pop('patch_info', None)
        plane_mask = data_dict.pop('plane_mask', None)

        # if we don't want to perform augmentation on gpu , we might as well do here with cpu in the dataloader workers
        if self.transforms is not None and not (self.is_training and getattr(self.mgr, 'augment_on_device', False)):
            data_dict = self.transforms(**data_dict)

        if metadata is not None:
            data_dict['patch_info'] = metadata
        if plane_mask is not None:
            data_dict['plane_mask'] = plane_mask

        data_dict.pop('_skip_spatial_transforms', None)

        return data_dict
