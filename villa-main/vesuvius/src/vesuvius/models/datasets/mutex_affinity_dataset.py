"""
MutexAffinityDataset - Dataset for mutex watershed affinity graphs.

This dataset pairs raw volumes with Mutex Watershed affinity graphs,
supporting both attractive and repulsive affinity targets.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import zarr
from concurrent.futures import ProcessPoolExecutor, as_completed
from torch.utils.data import Dataset

from vesuvius.utils.utils import pad_or_crop_3d
from .find_valid_patches import find_valid_patches
from .intensity_properties import initialize_intensity_properties
from ..training.normalization import get_normalization
from ..augmentation.pipelines import create_training_transforms
from ..augmentation.transforms.utils.perf import collect_augmentation_names


logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Simple ArrayHandle classes (inlined from deleted adapters)
# -----------------------------------------------------------------------------

class ZarrArrayHandle:
    """Handle for zarr arrays with windowed read support."""

    def __init__(self, array, *, path: Optional[Path] = None, spatial_shape: Optional[Tuple[int, ...]] = None):
        self._array = array
        self._path = path
        self._spatial_shape = spatial_shape or self._infer_spatial_shape(array)

    @staticmethod
    def _infer_spatial_shape(array) -> Tuple[int, ...]:
        shape = tuple(int(v) for v in getattr(array, "shape", ()))
        if len(shape) >= 3:
            return tuple(shape[-3:])
        if len(shape) == 2:
            return shape
        return shape

    @property
    def spatial_shape(self) -> Tuple[int, ...]:
        return self._spatial_shape

    def read(self) -> np.ndarray:
        return np.asarray(self._array[:])

    def read_window(self, start: Tuple[int, ...], size: Tuple[int, ...]) -> np.ndarray:
        slices = tuple(slice(s, s + sz) for s, sz in zip(start, size))
        return np.asarray(self._array[slices])

    def raw(self):
        return self._array


class TiffArrayHandle:
    """Handle for TIFF files with windowed read support."""

    def __init__(self, path: Path, *, spatial_shape: Tuple[int, ...], dtype: np.dtype):
        self._path = path
        self._spatial_shape = spatial_shape
        self._dtype = dtype
        self._zarr_store = None

    @property
    def spatial_shape(self) -> Tuple[int, ...]:
        return self._spatial_shape

    def _ensure_zarr(self):
        if self._zarr_store is None:
            import tifffile
            self._zarr_store = tifffile.imread(str(self._path), aszarr=True)
        return self._zarr_store

    def read(self) -> np.ndarray:
        import tifffile
        return tifffile.imread(str(self._path))

    def read_window(self, start: Tuple[int, ...], size: Tuple[int, ...]) -> np.ndarray:
        store = self._ensure_zarr()
        arr = zarr.open(store, mode='r')
        slices = tuple(slice(s, s + sz) for s, sz in zip(start, size))
        return np.asarray(arr[slices])


# -----------------------------------------------------------------------------
# Patch and Volume dataclasses
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class TargetSpec:
    """Configuration describing how to load a single affinity target."""
    affinity_key: str
    mask_key: Optional[str]
    invert_for_loss: bool = False


@dataclass
class PatchInfo:
    """Describes a single extractable patch."""
    volume_index: int
    volume_name: str
    position: Tuple[int, ...]
    patch_size: Tuple[int, ...]


# -----------------------------------------------------------------------------
# MutexAffinityDataset
# -----------------------------------------------------------------------------

class MutexAffinityDataset(Dataset):
    """
    Dataset that pairs raw volumes with Mutex Watershed affinity graphs.

    The dataset expects directories under ``dataset_config.data_path``:

    - ``image_dirname`` (default ``images``) holding volumetric intensities
      either as ``.zarr`` stores or 3D ``.tif/.tiff`` stacks.
    - ``affinity_dirname`` (default ``affinity_graph``) containing the Zarr
      stores produced by ``scripts/generate_mutex_graph.py``. Each store must
      expose ``affinities/<name>`` and ``mask/<name>`` arrays.

    Targets are declared in ``dataset_config.affinity_targets`` with entries of the form::

        affinity_targets:
          mutex_attractive:
            affinity_key: "affinities/attractive"
            mask_key: "mask/attractive"
            invert: true
          mutex_repulsive:
            affinity_key: "affinities/repulsive"
            mask_key: "mask/repulsive"
            invert: false
    """

    DEFAULT_TARGETS: Mapping[str, TargetSpec] = {
        "mutex_attractive": TargetSpec(
            affinity_key="affinities/attractive",
            mask_key="mask/attractive",
            invert_for_loss=True,
        ),
        "mutex_repulsive": TargetSpec(
            affinity_key="affinities/repulsive",
            mask_key="mask/repulsive",
            invert_for_loss=False,
        ),
    }

    def __init__(self, mgr, is_training: bool = True) -> None:
        super().__init__()
        self.mgr = mgr
        self.is_training = is_training
        self.targets = getattr(mgr, 'targets', {})
        self._profile_augmentations = bool(getattr(mgr, 'profile_augmentations', False))
        self._augmentation_names: List[str] = []

        # Configuration
        self.patch_size = tuple(mgr.train_patch_size)
        self.is_2d = len(self.patch_size) == 2

        # Validation parameters
        self.min_labeled_ratio = getattr(mgr, 'min_labeled_ratio', 0.10)
        self.min_bbox_percent = getattr(mgr, 'min_bbox_percent', 0.95)
        self.skip_patch_validation = getattr(mgr, 'skip_patch_validation', False)

        # Storage
        self._affinity_specs: Dict[str, TargetSpec] = {}
        self._mask_handles: Dict[str, List[Optional[ZarrArrayHandle]]] = {}
        self._volume_ids: List[str] = []
        self._image_handles: List = []
        self._affinity_handles: Dict[str, List] = {}
        self._patches: List[PatchInfo] = []

        # Normalization
        self.normalization_scheme = getattr(mgr, 'normalization_scheme', 'zscore')
        self.intensity_properties = getattr(mgr, 'intensity_properties', None) or {}
        self.normalizer = None

        # Transforms
        self.transforms = None

        # target_volumes for compatibility
        self.target_volumes: Dict[str, List[Dict]] = {}

        # Initialize
        self._initialize_volumes()
        self._initialize_normalization()
        self._build_patch_index()
        self._initialize_transforms()

        logger.info(
            "MutexAffinityDataset initialized: %d volumes, %d patches",
            len(self._volume_ids), len(self._patches)
        )

    def _initialize_volumes(self) -> None:
        """Discover and load image volumes and affinity graphs."""
        dataset_cfg = getattr(self.mgr, "dataset_config", {}) or {}
        data_path = Path(dataset_cfg.get("data_path", getattr(self.mgr, "data_path", "."))).resolve()

        image_dirname = dataset_cfg.get("image_dirname", "images")
        affinity_dirname = dataset_cfg.get("affinity_dirname", "affinity_graph")

        image_root = data_path / image_dirname
        affinity_root = data_path / affinity_dirname

        if not image_root.exists():
            raise FileNotFoundError(f"Images directory not found: {image_root}")
        if not affinity_root.exists():
            raise FileNotFoundError(f"Affinity directory not found: {affinity_root}")

        # Discover raw image volumes
        image_map: Dict[str, Path] = {}
        for path in image_root.iterdir():
            if path.suffix == ".zarr" and path.is_dir():
                image_map[path.stem] = path
            elif path.is_file() and path.suffix.lower() in {".tif", ".tiff"}:
                image_map[path.stem] = path

        if not image_map:
            raise ValueError(f"No image volumes discovered in {image_root}")

        # Resolve affinity targets configuration
        configured_targets = dataset_cfg.get("affinity_targets") or {}
        affinity_specs = self._normalize_target_specs(configured_targets)
        active_targets = {
            name: spec
            for name, spec in affinity_specs.items()
            if name in (self.targets or {})
        }
        if not active_targets:
            active_targets = {
                name: spec
                for name, spec in self.DEFAULT_TARGETS.items()
                if name in (self.targets or {})
            }

        if not active_targets:
            raise ValueError(
                "MutexAffinityDataset requires at least one target present in "
                "both dataset_config.affinity_targets and mgr.targets"
            )

        self._affinity_specs = active_targets
        self._mask_handles = {target: [] for target in active_targets}
        self._affinity_handles = {target: [] for target in active_targets}
        self.target_volumes = {target: [] for target in active_targets}

        # Handle uint8 conversions
        affinity_paths = sorted(
            path
            for path in affinity_root.iterdir()
            if path.is_dir() and path.suffix == ".zarr"
        )
        if not affinity_paths:
            raise ValueError(f"No affinity .zarr stores found in {affinity_root}")

        conversion_plan = self._plan_uint8_conversions(affinity_paths, active_targets)
        self._execute_uint8_conversions(conversion_plan)

        suffixes: Sequence[str] = tuple(
            str(suffix)
            for suffix in dataset_cfg.get("affinity_volume_suffixes", ["_surface"])
            if suffix is not None
        )

        for affinity_path in affinity_paths:
            volume_id = affinity_path.stem

            # Find matching image
            candidate_names = [volume_id]
            for suffix in suffixes:
                if suffix and volume_id.endswith(suffix):
                    candidate_names.append(volume_id[: -len(suffix)])

            image_path = None
            for candidate in candidate_names:
                image_path = image_map.get(candidate)
                if image_path is not None:
                    break

            if image_path is None:
                raise FileNotFoundError(
                    f"No matching image volume found for affinity store '{volume_id}'"
                )

            image_handle = self._open_image_handle(image_path)
            graph_root = zarr.open_group(str(affinity_path), mode="r")

            self._volume_ids.append(volume_id)
            self._image_handles.append(image_handle)

            for target_name, spec in active_targets.items():
                affinity_array = self._extract_required_array(
                    graph_root, spec.affinity_key, store_path=affinity_path
                )
                affinity_handle = ZarrArrayHandle(
                    affinity_array,
                    path=affinity_path,
                    spatial_shape=self._infer_spatial_shape(affinity_array),
                )

                mask_handle = None
                if spec.mask_key is not None:
                    mask_array = self._extract_required_array(
                        graph_root, spec.mask_key, store_path=affinity_path
                    )
                    mask_handle = ZarrArrayHandle(
                        mask_array,
                        path=affinity_path,
                        spatial_shape=self._infer_spatial_shape(mask_array),
                    )

                self._affinity_handles[target_name].append(affinity_handle)
                self._mask_handles[target_name].append(mask_handle)

                # Build target_volumes for compatibility
                entry = {
                    "volume_id": volume_id,
                    "data": {
                        "data": image_handle._array if hasattr(image_handle, '_array') else None,
                        "label": affinity_array,
                    }
                }
                self.target_volumes[target_name].append(entry)

    def _initialize_normalization(self) -> None:
        """Initialize intensity properties and normalizer."""
        skip_sampling = getattr(self.mgr, 'skip_intensity_sampling', True)

        if not skip_sampling and not self.intensity_properties and self.normalization_scheme in ['zscore', 'ct']:
            if self.target_volumes:
                self.intensity_properties = initialize_intensity_properties(
                    target_volumes=self.target_volumes,
                    normalization_scheme=self.normalization_scheme,
                    existing_properties=None,
                    cache_enabled=True,
                    cache_dir=Path(self.mgr.data_path) / '.patches_cache',
                    mgr=self.mgr,
                )

        self.normalizer = get_normalization(self.normalization_scheme, self.intensity_properties)

    def _build_patch_index(self) -> None:
        """Build the index of valid patches."""
        if not self._image_handles:
            return

        stride = tuple(self.patch_size)
        first_target = list(self._affinity_specs.keys())[0]

        for vol_idx, (volume_id, image_handle) in enumerate(zip(self._volume_ids, self._image_handles)):
            spatial_shape = image_handle.spatial_shape

            if self.skip_patch_validation:
                # Enumerate all positions
                if self.is_2d:
                    ph, pw = self.patch_size
                    sh, sw = stride
                    y_positions = list(range(0, max(1, spatial_shape[0] - ph + 1), sh))
                    x_positions = list(range(0, max(1, spatial_shape[1] - pw + 1), sw))
                    for y in y_positions:
                        for x in x_positions:
                            self._patches.append(PatchInfo(
                                volume_index=vol_idx,
                                volume_name=volume_id,
                                position=(y, x),
                                patch_size=self.patch_size,
                            ))
                else:
                    pd, ph, pw = self.patch_size
                    sd, sh, sw = stride
                    z_positions = list(range(0, max(1, spatial_shape[0] - pd + 1), sd))
                    y_positions = list(range(0, max(1, spatial_shape[1] - ph + 1), sh))
                    x_positions = list(range(0, max(1, spatial_shape[2] - pw + 1), sw))
                    for z in z_positions:
                        for y in y_positions:
                            for x in x_positions:
                                self._patches.append(PatchInfo(
                                    volume_index=vol_idx,
                                    volume_name=volume_id,
                                    position=(z, y, x),
                                    patch_size=self.patch_size,
                                ))
            else:
                # Use find_valid_patches
                label_handle = self._affinity_handles[first_target][vol_idx]
                label_array = label_handle.raw() if label_handle else None

                if label_array is not None:
                    result = find_valid_patches(
                        label_arrays=[label_array],
                        label_names=[volume_id],
                        patch_size=self.patch_size,
                        bbox_threshold=self.min_bbox_percent,
                        label_threshold=self.min_labeled_ratio,
                        valid_patch_find_resolution=getattr(self.mgr, 'valid_patch_find_resolution', 1),
                    )

                    for entry in result['fg_patches']:
                        pos = tuple(entry['start_pos'])
                        self._patches.append(PatchInfo(
                            volume_index=vol_idx,
                            volume_name=volume_id,
                            position=pos,
                            patch_size=self.patch_size,
                        ))

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

    def _extract_patch(self, patch: PatchInfo) -> Dict[str, Any]:
        """Extract image and affinity patches."""
        vol_idx = patch.volume_index
        start = patch.position
        size = patch.patch_size

        image_handle = self._image_handles[vol_idx]

        # Extract image
        image_data = image_handle.read_window(start, size)
        if self.normalizer is not None:
            image_data = self.normalizer.run(image_data)
        if self.is_2d:
            from vesuvius.utils.utils import pad_or_crop_2d
            image_data = pad_or_crop_2d(image_data.astype(np.float32), size)
        else:
            image_data = pad_or_crop_3d(image_data.astype(np.float32), size)
        image_tensor = torch.from_numpy(image_data[np.newaxis, ...])

        result = {'image': image_tensor, 'is_unlabeled': False}

        # Extract affinity targets and masks
        for target_name, handles in self._affinity_handles.items():
            handle = handles[vol_idx]
            affinity_data = handle.read_window(start, size)
            if self.is_2d:
                from vesuvius.utils.utils import pad_or_crop_2d
                affinity_data = pad_or_crop_2d(affinity_data.astype(np.float32), size)
            else:
                affinity_data = pad_or_crop_3d(affinity_data.astype(np.float32), size)
            result[target_name] = torch.from_numpy(affinity_data[np.newaxis, ...])

            # Extract mask if available
            mask_handle = self._mask_handles[target_name][vol_idx]
            if mask_handle is not None:
                mask_data = mask_handle.read_window(start, size)
                mask_bool = np.asarray(mask_data != 0)
                result[f"{target_name}_mask"] = torch.from_numpy(np.ascontiguousarray(mask_bool))

        return result

    def __len__(self) -> int:
        return len(self._patches)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        patch = self._patches[index]
        data_dict = self._extract_patch(patch)

        if self.transforms is not None:
            if self._profile_augmentations and self._augmentation_names:
                data_dict['_aug_perf'] = {name: 0.0 for name in self._augmentation_names}
            data_dict = self.transforms(**data_dict)

        return data_dict

    # -------------------------------------------------------------------------
    # Helper utilities
    # -------------------------------------------------------------------------

    @staticmethod
    def _normalize_target_specs(raw: Mapping[str, Mapping[str, object]]) -> Dict[str, TargetSpec]:
        specs: Dict[str, TargetSpec] = {}
        for name, cfg in raw.items():
            if not isinstance(cfg, Mapping):
                continue
            affinity_key = str(cfg.get("affinity_key", "")).strip()
            if not affinity_key:
                continue
            mask_key = cfg.get("mask_key")
            mask_key_str = str(mask_key).strip() if mask_key is not None else None
            specs[name] = TargetSpec(
                affinity_key=affinity_key,
                mask_key=mask_key_str,
                invert_for_loss=bool(cfg.get("invert", False)),
            )
        return specs

    @staticmethod
    def _extract_required_array(group, key: str, *, store_path: Path):
        if key not in group:
            raise KeyError(f"Dataset '{key}' not found in {store_path}")
        array = group[key]
        if not hasattr(array, "shape"):
            raise TypeError(f"Expected zarr array for key '{key}' in {store_path}")
        return array

    @staticmethod
    def _infer_spatial_shape(array) -> Tuple[int, ...]:
        shape = tuple(int(v) for v in getattr(array, "shape", ()))
        if not shape:
            raise ValueError("Unable to infer spatial shape from empty array")
        if len(shape) >= 3:
            return tuple(shape[-3:])
        if len(shape) == 2:
            return shape
        raise ValueError(f"Unsupported array shape {shape} for affinity volume")

    def _open_image_handle(self, path: Path):
        if path.suffix == ".zarr" and path.is_dir():
            array = zarr.open(str(path), mode="r")
            spatial_shape = self._infer_spatial_shape(array)
            return ZarrArrayHandle(array, path=path, spatial_shape=spatial_shape)

        suffix = path.suffix.lower()
        if suffix not in {".tif", ".tiff"}:
            raise ValueError(f"Unsupported image format for {path}")

        import tifffile
        with tifffile.TiffFile(str(path)) as tif:
            series = tif.series[0]
            spatial_shape = tuple(int(v) for v in series.shape)
            dtype = np.dtype(series.dtype)

        return TiffArrayHandle(path, spatial_shape=spatial_shape, dtype=dtype)

    def _plan_uint8_conversions(
        self,
        affinity_paths: Sequence[Path],
        active_targets: Mapping[str, TargetSpec],
    ) -> Dict[Path, List[str]]:
        plan: Dict[Path, List[str]] = {}
        for path in affinity_paths:
            try:
                group = zarr.open_group(str(path), mode="r")
            except Exception as exc:
                print(f"[MutexDataset] Warning: failed to inspect {path.name}: {exc}")
                continue

            keys: List[str] = []
            for spec in active_targets.values():
                for key in filter(None, [spec.affinity_key, spec.mask_key]):
                    if key in keys:
                        continue
                    if key not in group:
                        continue
                    array = group[key]
                    dtype = getattr(array, "dtype", None)
                    if dtype is None:
                        continue
                    if np.issubdtype(dtype, np.integer) and dtype.itemsize == 1:
                        continue
                    keys.append(key)

            if keys:
                plan[path] = keys

        return plan

    def _execute_uint8_conversions(self, plan: Dict[Path, List[str]]) -> None:
        if not plan:
            return

        items = list(plan.items())
        print(f"[MutexDataset] Converting {len(items)} affinity store(s) to uint8...", flush=True)

        if len(items) == 1:
            path, keys = items[0]
            print(f"[MutexDataset]   -> {path.name}: {len(keys)} dataset(s)", flush=True)
            try:
                _, converted, skipped = _convert_store_to_uint8((str(path), keys))
                if converted:
                    print(f"[MutexDataset]   -> {path.name}: converted {len(converted)} dataset(s)", flush=True)
                if skipped:
                    print(f"[MutexDataset]   -> {path.name}: skipped {len(skipped)} dataset(s)", flush=True)
            except Exception as exc:
                print(f"[MutexDataset]   -> {path.name}: conversion failed ({exc})", flush=True)
            return

        max_workers = min(len(items), max(1, os.cpu_count() or 1))
        futures = {}
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            for path, keys in items:
                print(f"[MutexDataset]   -> {path.name}: {len(keys)} dataset(s)", flush=True)
                future = pool.submit(_convert_store_to_uint8, (str(path), keys))
                futures[future] = path

            for future in as_completed(futures):
                path = futures[future]
                try:
                    _, converted, skipped = future.result()
                    if converted:
                        print(f"[MutexDataset]   -> {path.name}: converted {len(converted)} dataset(s)", flush=True)
                    if skipped:
                        print(f"[MutexDataset]   -> {path.name}: skipped {len(skipped)} dataset(s)", flush=True)
                except Exception as exc:
                    print(f"[MutexDataset]   -> {path.name}: conversion failed ({exc})", flush=True)

    # -------------------------------------------------------------------------
    # Public accessors
    # -------------------------------------------------------------------------

    @property
    def affinity_specs(self) -> Mapping[str, TargetSpec]:
        return self._affinity_specs

    @property
    def volume_ids(self) -> Sequence[str]:
        return tuple(self._volume_ids)

    @property
    def valid_patches(self) -> List[PatchInfo]:
        return self._patches

    def get_labeled_unlabeled_patch_indices(self) -> Tuple[List[int], List[int]]:
        """All patches are labeled in MutexAffinityDataset."""
        return list(range(len(self._patches))), []


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def _iter_chunk_slices(shape: Sequence[int], chunks: Sequence[int]) -> Iterable[Tuple[slice, ...]]:
    actual = []
    for dim, chunk in zip(shape, chunks):
        size = int(chunk) if chunk and chunk > 0 else int(dim)
        actual.append(range(0, int(dim), size))

    for starts in product(*actual):
        slices = []
        for start, dim, chunk in zip(starts, shape, chunks):
            size = int(chunk) if chunk and chunk > 0 else int(dim)
            stop = min(int(dim), int(start) + size)
            slices.append(slice(int(start), stop))
        yield tuple(slices)


def _convert_store_to_uint8(payload: Tuple[str, Sequence[str]]):
    store_path_str, keys = payload
    store_path = Path(store_path_str)
    converted: List[str] = []
    skipped: List[str] = []

    group = zarr.open_group(str(store_path), mode="r+")

    for key in keys:
        if key not in group:
            skipped.append(key)
            continue

        array = group[key]
        dtype = getattr(array, "dtype", None)
        if dtype is None or (np.issubdtype(dtype, np.integer) and dtype.itemsize == 1):
            skipped.append(key)
            continue

        parent, leaf = key.rsplit("/", 1) if "/" in key else ("", key)
        dest_group = group[parent] if parent else group

        tmp_name = f"{leaf}__tmp_uint8"
        if tmp_name in dest_group:
            del dest_group[tmp_name]

        chunks = getattr(array, "chunks", None)
        if chunks is None:
            chunks = tuple(int(dim) for dim in array.shape)
        else:
            chunks = tuple(int(c) if c is not None else int(dim) for c, dim in zip(chunks, array.shape))

        compressor = getattr(array, "compressor", None)
        tmp_ds = dest_group.create_dataset(
            tmp_name,
            shape=array.shape,
            dtype=np.uint8,
            chunks=chunks,
            compressor=compressor,
            overwrite=True,
        )

        # First, check if all data is binary before writing
        is_binary = True
        for chunk_slices in _iter_chunk_slices(array.shape, chunks):
            data = array[chunk_slices]
            if not np.all((data == 0) | (data == 1)):
                is_binary = False
                break

        if not is_binary:
            skipped.append(key)
            continue

        # Now, write the data to the temporary dataset
        for chunk_slices in _iter_chunk_slices(array.shape, chunks):
            data = array[chunk_slices]
            tmp_ds[chunk_slices] = data.astype(np.uint8, copy=False)
        if leaf in dest_group:
            del dest_group[leaf]
        dest_group.move(tmp_name, leaf)
        converted.append(key)

    return store_path_str, converted, skipped
