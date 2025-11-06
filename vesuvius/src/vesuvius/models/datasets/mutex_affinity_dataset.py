from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import os
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import zarr
from concurrent.futures import ProcessPoolExecutor, as_completed

from .base_dataset import BaseDataset
from .adapters.base_io import TiffArrayHandle, ZarrArrayHandle


@dataclass(frozen=True)
class TargetSpec:
    """Configuration describing how to load a single affinity target."""

    affinity_key: str
    mask_key: Optional[str]
    invert_for_loss: bool = False


class MutexAffinityDataset(BaseDataset):
    """
    Dataset that pairs raw volumes with Mutex Watershed affinity graphs.

    The dataset expects two sibling directories under ``dataset_config.data_path``:

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

    Only targets that appear both in ``mgr.targets`` and in
    ``affinity_targets`` are loaded.

    For each sample, the dataset returns tensors named after the targets plus
    an additional ``f\"{target}_mask\"`` entry containing the corresponding
    validity mask (values in {0, 1}).
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
        self._affinity_specs: Dict[str, TargetSpec] = {}
        self._mask_handles: Dict[str, List[Optional[ZarrArrayHandle]]] = {}
        self._volume_ids: List[str] = []
        self._preserve_label_dtype = True
        super().__init__(mgr, is_training=is_training)

    # ------------------------------------------------------------------
    # BaseDataset hooks
    # ------------------------------------------------------------------
    def _initialize_volumes(self) -> None:
        dataset_cfg = getattr(self.mgr, "dataset_config", {}) or {}
        data_path = Path(dataset_cfg.get("data_path", getattr(self.mgr, "data_path", "."))).resolve()
        image_dirname = dataset_cfg.get("image_dirname", "images")
        label_dirname = dataset_cfg.get("label_dirname", "labels")
        affinity_dirname = dataset_cfg.get("affinity_dirname", "affinity_graph")
        image_extensions = tuple(
            ext if ext.startswith(".") else f".{ext}"
            for ext in dataset_cfg.get(
                "image_extensions",
                [".zarr", ".tif", ".tiff"],
            )
        )
        label_extensions = tuple(
            ext if ext.startswith(".") else f".{ext}"
            for ext in dataset_cfg.get(
                "label_extensions",
                dataset_cfg.get("image_extensions", [".zarr", ".tif", ".tiff"]),
            )
        )
        allow_unlabeled = bool(getattr(self.mgr, "allow_unlabeled_data", False))

        image_root = data_path / image_dirname
        label_root = data_path / label_dirname
        affinity_root = data_path / affinity_dirname

        if not image_root.exists():
            raise FileNotFoundError(f"Images directory not found: {image_root}")
        if not affinity_root.exists():
            raise FileNotFoundError(f"Affinity directory not found: {affinity_root}")
        if not label_root.exists() and not allow_unlabeled:
            raise FileNotFoundError(f"Labels directory not found: {label_root}")

        # Discover raw image volumes
        image_map: Dict[str, Path] = {}
        for path in image_root.iterdir():
            if path.suffix == ".zarr" and path.is_dir():
                image_map[path.stem] = path
            elif path.is_file() and path.suffix.lower() in image_extensions:
                image_map[path.stem] = path

        if not image_map:
            raise ValueError(f"No image volumes discovered in {image_root}")

        # Build lookup for standard label targets (e.g., surfaces)
        label_lookup: Dict[Tuple[str, str], Path] = {}
        if label_root.exists():
            for path in label_root.iterdir():
                if path.is_dir():
                    continue
                suffix = path.suffix.lower()
                if suffix not in label_extensions:
                    continue
                stem = path.stem
                parts = stem.rsplit("_", 1)
                if len(parts) != 2:
                    continue
                volume_id, target_name = parts
                key = (volume_id, target_name)
                if key in label_lookup:
                    raise ValueError(f"Duplicate label detected for volume '{volume_id}' target '{target_name}'")
                label_lookup[key] = path

        primary_targets = list(self.targets.keys())

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

        # Persist normalized settings so the trainer can reuse them
        self._affinity_specs = active_targets
        dataset_cfg.setdefault("affinity_targets", {})
        stored_targets = dataset_cfg["affinity_targets"]
        for name, spec in active_targets.items():
            existing_cfg = configured_targets.get(name, {})
            stored_targets[name] = {
                "affinity_key": spec.affinity_key,
                "mask_key": spec.mask_key,
                "invert": bool(spec.invert_for_loss),
                "visualization": existing_cfg.get("visualization", "affinity"),
            }
            # also ensure mgr.targets has visualization defaults
            target_cfg = self.mgr.targets.get(name, {})
            if "activation" not in target_cfg:
                target_cfg["activation"] = "sigmoid"
            target_cfg.setdefault("visualization", "affinity")
            self.mgr.targets[name] = target_cfg

        non_affinity_targets = [t for t in primary_targets if t not in active_targets]

        # Prepare containers expected by BaseDataset
        self.target_volumes = {target: [] for target in primary_targets}
        self.zarr_arrays = []
        self.zarr_names = []
        self.data_paths = []
        self._mask_handles = {target: [] for target in active_targets}
        self._volume_ids = []

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
            image_stem = image_path.stem

            self._volume_ids.append(volume_id)

            for target_name, spec in active_targets.items():
                affinity_array = self._extract_required_array(
                    graph_root,
                    spec.affinity_key,
                    store_path=affinity_path,
                )
                affinity_handle = ZarrArrayHandle(
                    affinity_array,
                    path=affinity_path,
                    spatial_shape=self._infer_spatial_shape(affinity_array),
                )
                mask_handle = None
                if spec.mask_key is not None:
                    mask_array = self._extract_required_array(
                        graph_root,
                        spec.mask_key,
                        store_path=affinity_path,
                    )
                    mask_handle = ZarrArrayHandle(
                        mask_array,
                        path=affinity_path,
                        spatial_shape=self._infer_spatial_shape(mask_array),
                    )

                entry = {
                    "volume_id": volume_id,
                    "image": image_handle,
                    "label": affinity_handle,
                    "label_path": str(affinity_path),
                    "label_source": affinity_array,
                    "has_label": True,
                }

                self.target_volumes[target_name].append(entry)
                self._mask_handles[target_name].append(mask_handle)
                self.zarr_arrays.append(affinity_array)
                self.zarr_names.append(f"{volume_id}_{target_name}")
                self.data_paths.append(str(affinity_path))

            for target_name in non_affinity_targets:
                label_path = label_lookup.get((image_stem, target_name))
                if label_path is None:
                    # Fallback: allow matching volume_id variants (e.g., *_surface suffix)
                    for candidate in candidate_names:
                        label_path = label_lookup.get((candidate, target_name))
                        if label_path is not None:
                            break

                if label_path is None:
                    if allow_unlabeled:
                        entry = {
                            "volume_id": volume_id,
                            "image": image_handle,
                            "label": None,
                            "label_path": None,
                            "label_source": None,
                            "has_label": False,
                        }
                        self.target_volumes[target_name].append(entry)
                        continue
                    raise FileNotFoundError(
                        f"Label for target '{target_name}' not found for volume '{image_stem}'"
                    )

                label_handle = self._open_image_handle(label_path)
                label_source = None
                if hasattr(label_handle, "raw"):
                    try:
                        label_source = label_handle.raw()
                    except Exception:
                        label_source = None

                entry = {
                    "volume_id": volume_id,
                    "image": image_handle,
                    "label": label_handle,
                    "label_path": str(label_path),
                    "label_source": label_source,
                    "has_label": True,
                }
                self.target_volumes[target_name].append(entry)
                if label_source is not None:
                    self.zarr_arrays.append(label_source)
                    self.zarr_names.append(f"{volume_id}_{target_name}")
                    self.data_paths.append(str(label_path))

        self.mgr.dataset_config = dataset_cfg

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------
    def _extract_chunk_patch(self, chunk_patch):
        data_dict = super()._extract_chunk_patch(chunk_patch)

        if not self._mask_handles:
            return data_dict

        start = tuple(int(v) for v in chunk_patch.position)
        size = tuple(int(v) for v in chunk_patch.patch_size)

        for target_name, handles in self._mask_handles.items():
            handle = handles[chunk_patch.volume_index]
            if handle is None:
                continue
            mask_patch = handle.read_window(start, size)
            mask_array = np.asarray(mask_patch)
            if mask_array.size == 0:
                continue
            mask_bool = np.asarray(mask_array != 0)
            mask_tensor = torch.from_numpy(np.ascontiguousarray(mask_bool))
            data_dict[f"{target_name}_mask"] = mask_tensor

        return data_dict

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
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
        print(
            f"[MutexDataset] Converting {len(items)} affinity store(s) to uint8...",
            flush=True,
        )

        if len(items) == 1:
            path, keys = items[0]
            print(
                f"[MutexDataset]   -> {path.name}: {len(keys)} dataset(s)",
                flush=True,
            )
            try:
                _, converted, skipped = _convert_store_to_uint8((str(path), keys))
                if converted:
                    print(
                        f"[MutexDataset]   -> {path.name}: converted {len(converted)} dataset(s)",
                        flush=True,
                    )
                if skipped:
                    print(
                        f"[MutexDataset]   -> {path.name}: skipped {len(skipped)} dataset(s)",
                        flush=True,
                    )
            except Exception as exc:
                print(f"[MutexDataset]   -> {path.name}: conversion failed ({exc})", flush=True)
            return

        max_workers = min(len(items), max(1, os.cpu_count() or 1))
        futures = {}
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            for path, keys in items:
                print(
                    f"[MutexDataset]   -> {path.name}: {len(keys)} dataset(s)",
                    flush=True,
                )
                future = pool.submit(_convert_store_to_uint8, (str(path), keys))
                futures[future] = path

            for future in as_completed(futures):
                path = futures[future]
                try:
                    _, converted, skipped = future.result()
                    if converted:
                        print(
                            f"[MutexDataset]   -> {path.name}: converted {len(converted)} dataset(s)",
                            flush=True,
                        )
                    if skipped:
                        print(
                            f"[MutexDataset]   -> {path.name}: skipped {len(skipped)} dataset(s)",
                            flush=True,
                        )
                except Exception as exc:
                    print(f"[MutexDataset]   -> {path.name}: conversion failed ({exc})", flush=True)

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

    @staticmethod
    def _open_image_handle(path: Path):
        if path.suffix == ".zarr" and path.is_dir():
            array = zarr.open(str(path), mode="r")
            spatial_shape = MutexAffinityDataset._infer_spatial_shape(array)
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

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------
    @property
    def affinity_specs(self) -> Mapping[str, TargetSpec]:
        return self._affinity_specs

    @property
    def volume_ids(self) -> Sequence[str]:
        return tuple(self._volume_ids)


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
