"""Zarr-backed dataset adapter."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import zarr

from vesuvius.utils.io.zarr_utils import _get_zarr_path, _is_ome_zarr

from .base_io import (
    AdapterConfig,
    DataSourceAdapter,
    DiscoveredItem,
    LoadedVolume,
    VolumeMetadata,
    ZarrArrayHandle,
)


class ZarrAdapter(DataSourceAdapter):
    """Streams volumes directly from on-disk zarr hierarchies."""

    def __init__(self, config: AdapterConfig, *, logger: Optional[logging.Logger] = None) -> None:
        super().__init__(config, logger=logger)
        self._discovered: List[DiscoveredItem] = []
        self._metadata: Dict[str, VolumeMetadata] = {}
        self._image_arrays: Dict[str, object] = {}
        self._label_arrays: Dict[Tuple[str, str], object] = {}
        self._axes_cache: Dict[Path, Optional[Sequence[Mapping[str, object]]]] = {}

    # Discovery ------------------------------------------------------------------------------------

    def discover(self) -> Sequence[DiscoveredItem]:
        images_dir = self.config.data_path / self.config.image_dirname
        labels_dir = self.config.data_path / self.config.label_dirname

        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")

        has_labels = labels_dir.exists()
        if not has_labels and not self.config.allow_unlabeled:
            raise FileNotFoundError(
                f"Labels directory not found: {labels_dir} (allow_unlabeled={self.config.allow_unlabeled})"
            )

        image_zarrs = {
            path.stem: path
            for path in images_dir.iterdir()
            if path.is_dir() and path.suffix == ".zarr"
        }
        if not image_zarrs:
            raise ValueError(f"No .zarr directories found in {images_dir}")

        volume_label_map: Dict[str, Dict[str, Optional[Path]]] = {}
        image_map: Dict[str, Path] = {}

        if has_labels:
            for label_dir in labels_dir.iterdir():
                if not (label_dir.is_dir() and label_dir.suffix == ".zarr"):
                    continue
                stem = label_dir.stem
                for target in self.config.targets:
                    suffix = f"_{target}"
                    if stem.endswith(suffix):
                        image_id = stem[: -len(suffix)]
                        volume_id = image_id
                        image_path = self._resolve_image_path(
                            image_zarrs=image_zarrs,
                            image_id=image_id,
                            target=target,
                        )
                        if image_path is None:
                            raise ValueError(f"No image zarr found for label '{stem}'")
                        if volume_id not in volume_label_map:
                            volume_label_map[volume_id] = {t: None for t in self.config.targets}
                        volume_label_map[volume_id][target] = label_dir
                        image_map.setdefault(volume_id, image_path)
                        break

        discovered: List[DiscoveredItem] = []

        for stem, image_path in image_zarrs.items():
            volume_id = self._normalize_volume_id(stem)
            image_map.setdefault(volume_id, image_path)
            volume_label_map.setdefault(
                volume_id, {t: None for t in self.config.targets}
            )

        for volume_id, label_paths in sorted(volume_label_map.items()):
            image_path = image_map.get(volume_id)
            if image_path is None:
                raise ValueError(f"No image zarr found for volume '{volume_id}'")

            missing = [t for t, path in label_paths.items() if path is None]
            if missing and not self.config.allow_unlabeled:
                raise ValueError(
                    f"Volume '{volume_id}' is missing labels for targets {missing} (allow_unlabeled=False)"
                )

            discovered.append(
                DiscoveredItem(
                    volume_id=volume_id,
                    image_path=image_path,
                    label_paths=label_paths,
                )
            )

        if not discovered:
            raise ValueError("No zarr volumes discovered")

        self._discovered = discovered
        self.logger.info("Registered %d zarr volumes", len(discovered))
        return tuple(discovered)

    def _resolve_image_path(
        self, *, image_zarrs: Mapping[str, Path], image_id: str, target: str
    ) -> Optional[Path]:
        if image_id in image_zarrs:
            return image_zarrs[image_id]
        candidate = f"{image_id}_{target}"
        return image_zarrs.get(candidate)

    def _normalize_volume_id(self, stem: str) -> str:
        for target in self.config.targets:
            suffix = f"_{target}"
            if stem.endswith(suffix):
                return stem[: -len(suffix)]
        return stem

    # Preparation ---------------------------------------------------------------------------------

    def prepare(self, discovered: Sequence[DiscoveredItem]) -> None:
        metadata: Dict[str, VolumeMetadata] = {}

        for item in discovered:
            image_array = self._open_zarr_array(item.image_path)
            spatial_shape = self._extract_spatial_shape(
                image_array.shape, item.image_path
            )
            axes = self._derive_axes(image_array.shape, item.image_path)

            label_dtypes: Dict[str, Optional[np.dtype]] = {}
            targets_with_labels: List[str] = []

            for target, label_path in item.label_paths.items():
                if label_path is None:
                    label_dtypes[target] = None
                    continue

                label_array = self._open_zarr_array(label_path)
                label_spatial = self._extract_spatial_shape(
                    label_array.shape, label_path
                )
                if label_spatial != spatial_shape:
                    raise ValueError(
                        "Label spatial shape mismatch for volume '%s' target '%s': image=%s label=%s"
                        % (item.volume_id, target, spatial_shape, label_spatial)
                    )
                label_dtypes[target] = label_array.dtype
                targets_with_labels.append(target)
                self._label_arrays[(item.volume_id, target)] = label_array

            metadata[item.volume_id] = VolumeMetadata(
                volume_id=item.volume_id,
                image_path=item.image_path,
                spatial_shape=spatial_shape,
                dtype=image_array.dtype,
                axes=axes,
                targets_with_labels=tuple(targets_with_labels),
                label_paths=dict(item.label_paths),
                label_dtypes=label_dtypes,
            )
            self._image_arrays[item.volume_id] = image_array

        self._metadata = metadata
        self.logger.info("Prepared metadata for %d zarr volumes", len(metadata))

    def _open_zarr_array(self, path: Path):
        resolution = self.config.zarr_resolution
        if resolution is None:
            resolution = 0 if _is_ome_zarr(path) else None
        zarr_path = _get_zarr_path(path, resolution_level=resolution)
        result = zarr.open(zarr_path, mode="r")

        # If we got a Group instead of an Array, try to find the array inside
        if isinstance(result, zarr.hierarchy.Group):
            # Check for common array names or numbered levels
            for key in ["0", "data", "arr_0", ""]:
                if key in result:
                    candidate = result[key]
                    if hasattr(candidate, "shape"):
                        self.logger.debug(
                            "Opened zarr Group at %s, using nested array '%s'", path, key
                        )
                        return candidate
            # If no named array found, check if the group itself has array data
            # (some zarr stores have the array at the root)
            if hasattr(result, "shape"):
                return result
            raise ValueError(
                f"Zarr at {path} is a Group without a recognizable array. "
                f"Available keys: {list(result.keys())}"
            )
        return result

    def _extract_spatial_shape(
        self, shape: Sequence[int], path: Optional[Path] = None
    ) -> Tuple[int, ...]:
        axes_metadata = self._load_axes_metadata(path) if path is not None else None

        if axes_metadata:
            spatial_indices: List[int] = [
                idx
                for idx, axis in enumerate(axes_metadata)
                if str(axis.get("name", "")).lower() in {"z", "y", "x"}
            ]

            if not spatial_indices:
                spatial_indices = [
                    idx
                    for idx, axis in enumerate(axes_metadata)
                    if str(axis.get("type", "")).lower() == "space"
                ]

            if spatial_indices:
                return tuple(int(shape[idx]) for idx in spatial_indices)

        if len(shape) == 2:
            return tuple(int(v) for v in shape)
        if len(shape) == 3:
            return tuple(int(v) for v in shape)
        if len(shape) >= 4:
            return tuple(int(v) for v in shape[-3:])
        raise ValueError(f"Unsupported array shape {shape} for spatial inference")

    def _derive_axes(self, shape: Sequence[int], path: Optional[Path] = None) -> str:
        axes_metadata = self._load_axes_metadata(path) if path is not None else None

        if axes_metadata:
            return "".join(str(axis.get("name", "?")).upper() for axis in axes_metadata)

        if len(shape) == 2:
            return "YX"
        if len(shape) == 3:
            return "ZYX"
        if len(shape) == 4:
            return "CZYX"
        if len(shape) == 5:
            return "TCZYX"
        return "?"

    def _load_axes_metadata(
        self, path: Optional[Path]
    ) -> Optional[Sequence[Mapping[str, object]]]:
        if path is None:
            return None

        root = self._resolve_store_root(Path(path))
        if root is None:
            return None

        cache_key = root.resolve()
        if cache_key in self._axes_cache:
            return self._axes_cache[cache_key]

        attrs_path = root / ".zattrs"
        axes: Optional[Sequence[Mapping[str, object]]] = None
        if attrs_path.exists():
            try:
                with attrs_path.open("r", encoding="utf-8") as handle:
                    attrs = json.load(handle)
                multiscales = attrs.get("multiscales")
                if isinstance(multiscales, list) and multiscales:
                    candidate = multiscales[0].get("axes")
                    if isinstance(candidate, list) and candidate:
                        axes = tuple(candidate)
            except Exception as exc:  # pragma: no cover - best effort logging
                self.logger.debug(
                    "Failed to load axes metadata for %s: %s", attrs_path, exc
                )

        self._axes_cache[cache_key] = axes
        return axes

    def _resolve_store_root(self, path: Path) -> Optional[Path]:
        current = path
        if current.is_file():
            current = current.parent

        for candidate in (current,) + tuple(current.parents):
            if candidate.is_dir() and (candidate / ".zattrs").exists():
                return candidate
            if candidate == candidate.parent:
                break
        return None

    # Materialisation -----------------------------------------------------------------------------

    def iter_volumes(self) -> Iterator[LoadedVolume]:
        if not self._discovered:
            raise RuntimeError("ZarrAdapter.iter_volumes called before discover()")
        if not self._metadata:
            raise RuntimeError("ZarrAdapter.iter_volumes called before prepare()")

        for item in self._discovered:
            metadata = self._metadata[item.volume_id]
            image_array = self._image_arrays[metadata.volume_id]

            image_handle = ZarrArrayHandle(
                image_array,
                path=metadata.image_path,
                spatial_shape=metadata.spatial_shape,
            )

            label_handles: Dict[str, Optional[ZarrArrayHandle]] = {}
            for target, label_path in metadata.label_paths.items():
                if label_path is None:
                    label_handles[target] = None
                    continue
                label_array = self._label_arrays.get((metadata.volume_id, target))
                if label_array is None:
                    label_array = self._open_zarr_array(label_path)
                    self._label_arrays[(metadata.volume_id, target)] = label_array
                label_handles[target] = ZarrArrayHandle(
                    label_array,
                    path=label_path,
                    spatial_shape=metadata.spatial_shape,
                )

            yield LoadedVolume(
                metadata=metadata,
                image=image_handle,
                labels=label_handles,
            )
