"""Dataset orchestrator that bridges adapters and slicers."""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from .adapters import (
    AdapterConfig,
    DataSourceAdapter,
    ImageAdapter,
    LoadedVolume,
    NapariAdapter,
    ZarrAdapter,
)
from .mesh.base import LoadedMesh, MeshDataSourceAdapter
from .mesh.filesystem import MeshAdapter
from .mesh.handles import MeshHandle
from .base_dataset import BaseDataset


AdapterName = str


class DatasetOrchestrator(BaseDataset):
    """A BaseDataset subclass that sources data through configurable adapters."""

    _ADAPTERS: Dict[AdapterName, type[DataSourceAdapter]] = {
        "image": ImageAdapter,
        "zarr": ZarrAdapter,
        "napari": NapariAdapter,
    }

    _MESH_ADAPTERS: Dict[AdapterName, type[MeshDataSourceAdapter]] = {
        "mesh": MeshAdapter,
    }

    def __init__(
        self,
        mgr,
        *,
        adapter: AdapterName,
        adapter_kwargs: Optional[Dict[str, object]] = None,
        is_training: bool = True,
        logger: Optional[logging.Logger] = None,
        mesh_config: Optional[Dict[str, object]] = None,
    ) -> None:
        self.adapter_name = adapter
        self.adapter_kwargs = adapter_kwargs or {}
        self._runtime_logger = logger or logging.getLogger(__name__)
        self._adapter: Optional[DataSourceAdapter] = None
        self._adapter_config: Optional[AdapterConfig] = None
        self._primary_targets: Sequence[str] = ()
        self._loaded_volumes: List[LoadedVolume] = []
        self._mesh_adapter: Optional[MeshDataSourceAdapter] = None
        self._mesh_config_override = mesh_config or {}
        self._loaded_meshes: List[LoadedMesh] = []
        self._volume_mesh_handles: Dict[str, Dict[str, MeshHandle]] = {}
        super().__init__(mgr, is_training=is_training)
        self.logger = self._runtime_logger

    # BaseDataset hooks ----------------------------------------------------------------------------

    def _initialize_volumes(self) -> None:
        adapter_cls = self._resolve_adapter_class(self.adapter_name)
        targets = self._resolve_primary_targets()
        self._primary_targets = tuple(targets)

        config = self._build_adapter_config(targets)
        self._adapter_config = config

        adapter = adapter_cls(config, logger=self._runtime_logger, **self.adapter_kwargs)
        discovered = adapter.discover()
        adapter.prepare(discovered)
        volumes = list(adapter.iter_volumes())
        if not volumes:
            raise ValueError("Adapter produced no volumes")

        self._attach_meshes(volumes)

        self._populate_target_volumes(targets, volumes)

        # Persist adapter so downstream code can reuse metadata if needed
        self._adapter = adapter
        self._loaded_volumes = volumes

    # Internal helpers -----------------------------------------------------------------------------

    def _resolve_adapter_class(self, name: AdapterName) -> type[DataSourceAdapter]:
        try:
            return self._ADAPTERS[name]
        except KeyError as exc:  # pragma: no cover - defensive branch
            raise ValueError(
                f"Unknown adapter '{name}'. Available adapters: {sorted(self._ADAPTERS)}"
            ) from exc

    def _resolve_mesh_adapter_class(self, name: AdapterName) -> type[MeshDataSourceAdapter]:
        try:
            return self._MESH_ADAPTERS[name]
        except KeyError as exc:  # pragma: no cover - defensive branch
            raise ValueError(
                f"Unknown mesh adapter '{name}'. Available mesh adapters: {sorted(self._MESH_ADAPTERS)}"
            ) from exc

    def _resolve_primary_targets(self) -> Sequence[str]:
        if not hasattr(self.mgr, "targets"):
            raise ValueError("ConfigManager must expose 'targets' for dataset construction")
        targets = [
            name
            for name, info in self.mgr.targets.items()
            if not info.get("auxiliary_task", False)
        ]
        if not targets:
            raise ValueError("No primary targets defined in configuration")
        return targets

    def _build_adapter_config(self, targets: Sequence[str]) -> AdapterConfig:
        data_path = Path(getattr(self.mgr, "data_path", "."))
        allow_unlabeled = bool(getattr(self.mgr, "allow_unlabeled_data", False))
        image_dirname = getattr(self.mgr, "image_dirname", "images")
        label_dirname = getattr(self.mgr, "label_dirname", "labels")
        extensions = getattr(
            self.mgr,
            "image_extensions",
            (".tif", ".tiff", ".png", ".jpg", ".jpeg"),
        )
        zarr_resolution = getattr(self.mgr, "ome_zarr_resolution", None)
        tiff_chunk_shape = getattr(self.mgr, "tiff_chunk_shape", None)
        if tiff_chunk_shape is not None:
            tiff_chunk_shape = tuple(int(v) for v in tiff_chunk_shape)

        return AdapterConfig(
            data_path=data_path,
            targets=tuple(targets),
            allow_unlabeled=allow_unlabeled,
            image_dirname=image_dirname,
            label_dirname=label_dirname,
            image_extensions=tuple(extensions),
            zarr_resolution=zarr_resolution,
            tiff_chunk_shape=tiff_chunk_shape,
            mesh_dirname=getattr(self.mgr, "mesh_dirname", "meshes"),
        )

    def _populate_target_volumes(
        self, targets: Sequence[str], volumes: Iterable[LoadedVolume]
    ) -> None:
        self.target_volumes = {target: [] for target in targets}
        self.zarr_arrays = []
        self.zarr_names = []
        self.data_paths = []

        for volume in volumes:
            volume_id = volume.metadata.volume_id
            volume_meshes = self._volume_mesh_handles.get(volume_id, {})

            for target in targets:
                label_handle = volume.labels.get(target)
                label_path = volume.metadata.label_paths.get(target)
                # Use .raw() for label_source when available - this returns a zarr
                # Group (for OME-ZARR) or Array that supports direct slicing and can
                # be used with find_valid_patches for fast vectorized patch validation.
                # The handle is still used for 'label' to provide read_window for extraction.
                label_source = None
                if label_handle is not None:
                    raw_obj = getattr(label_handle, "raw", None)
                    if callable(raw_obj):
                        label_source = raw_obj()

                entry = {
                    "volume_id": volume_id,
                    "image": volume.image,
                    "label": label_handle,
                    "label_path": label_path,
                    "label_source": label_source,
                    "has_label": label_handle is not None,
                    "meshes": volume_meshes,
                }
                self.target_volumes[target].append(entry)

                if label_source is not None and label_path is not None:
                    self.zarr_arrays.append(label_source)
                    self.zarr_names.append(f"{volume_id}_{target}")
                    self.data_paths.append(str(label_path))

    def _attach_meshes(self, volumes: List[LoadedVolume]) -> None:
        mesh_cfg = self._resolve_mesh_config()
        if not mesh_cfg or not bool(mesh_cfg.get("enabled", False)):
            return

        adapter_name = str(mesh_cfg.get("adapter", "mesh"))
        adapter_cls = self._resolve_mesh_adapter_class(adapter_name)

        mesh_config = self._build_mesh_adapter_config(mesh_cfg)
        adapter_kwargs = dict(mesh_cfg.get("adapter_kwargs", {}))

        manifest_path = mesh_cfg.get("manifest") or mesh_cfg.get("metadata_path")
        if (
            manifest_path is not None
            and "metadata_path" not in adapter_kwargs
            and isinstance(manifest_path, (str, Path))
        ):
            manifest_path = Path(manifest_path)
            if not manifest_path.is_absolute():
                manifest_path = mesh_config.data_path / manifest_path
            adapter_kwargs["metadata_path"] = manifest_path

        metadata_inline = mesh_cfg.get("metadata")
        if metadata_inline is not None and "metadata" not in adapter_kwargs:
            adapter_kwargs["metadata"] = metadata_inline

        if "default_source_volume" not in adapter_kwargs and mesh_cfg.get("default_source_volume") is not None:
            adapter_kwargs["default_source_volume"] = mesh_cfg.get("default_source_volume")

        if "source_map" not in adapter_kwargs and mesh_cfg.get("source_map") is not None:
            adapter_kwargs["source_map"] = mesh_cfg.get("source_map")

        if "transform_map" not in adapter_kwargs and mesh_cfg.get("transform_map") is not None:
            adapter_kwargs["transform_map"] = mesh_cfg.get("transform_map")

        mesh_adapter = adapter_cls(mesh_config, logger=self._runtime_logger, **adapter_kwargs)
        discovered = mesh_adapter.discover()
        mesh_adapter.prepare(discovered)
        meshes = list(mesh_adapter.iter_meshes())
        if not meshes:
            self._runtime_logger.warning("Mesh adapter produced no meshes")
            return

        self._mesh_adapter = mesh_adapter
        self._loaded_meshes = meshes

        orphan_meshes: List[LoadedMesh] = []

        volume_aliases: Dict[str, str] = {}

        def _register_alias(alias: object, primary: str) -> None:
            if alias is None:
                return
            alias_str = str(alias)
            if not alias_str:
                return
            alias_str = alias_str.rstrip("/")
            if not alias_str:
                return
            if alias_str not in volume_aliases:
                volume_aliases[alias_str] = primary
            path_obj = Path(alias_str)
            name = path_obj.name
            if name and name not in volume_aliases:
                volume_aliases[name] = primary
            stem = path_obj.stem
            if stem and stem not in volume_aliases:
                volume_aliases[stem] = primary
            suffix = path_obj.suffix
            if suffix:
                trimmed = alias_str[: -len(suffix)]
                if trimmed and trimmed not in volume_aliases:
                    volume_aliases[trimmed] = primary

        for idx, volume in enumerate(volumes):
            primary_id = volume.metadata.volume_id
            mapping = self._volume_mesh_handles.setdefault(primary_id, {})
            volumes[idx] = replace(volume, meshes=mapping)

            _register_alias(primary_id, primary_id)
            image_path = getattr(volume.metadata, "image_path", None)
            _register_alias(image_path, primary_id)
            label_paths = getattr(volume.metadata, "label_paths", {}) or {}
            for label_path in label_paths.values():
                _register_alias(label_path, primary_id)

        def _resolve_volume_id(raw_id: str) -> Optional[str]:
            candidates = [raw_id]
            try:
                as_path = Path(raw_id)
                candidates.append(as_path.name)
                candidates.append(as_path.stem)
                if as_path.suffix:
                    trimmed = raw_id[:-len(as_path.suffix)]
                    candidates.append(trimmed)
            except Exception:  # pragma: no cover - defensive for odd volume IDs
                pass

            for candidate in candidates:
                if candidate in volume_aliases:
                    return volume_aliases[candidate]
            return None

        for mesh in meshes:
            volume_id = mesh.metadata.source_volume_id
            if volume_id is None:
                orphan_meshes.append(mesh)
                continue
            resolved_id = _resolve_volume_id(volume_id)
            if resolved_id is None:
                raise ValueError(
                    f"Mesh '{mesh.metadata.mesh_id}' references unknown volume '{volume_id}'"
                )
            self._volume_mesh_handles[resolved_id][mesh.metadata.mesh_id] = mesh.handle

        if orphan_meshes:
            self._runtime_logger.warning(
                "%d meshes lack associated source volumes and will not be linked to dataset patches",
                len(orphan_meshes),
            )

    def _resolve_mesh_config(self) -> Dict[str, object]:
        dataset_cfg = getattr(self.mgr, "dataset_config", {}) or {}
        config_from_dataset = dict(dataset_cfg.get("meshes", {}))
        override = dict(self._mesh_config_override)
        config_from_dataset.update(override)
        return config_from_dataset

    def _build_mesh_adapter_config(self, mesh_cfg: Mapping[str, object]) -> AdapterConfig:
        data_path = Path(getattr(self.mgr, "data_path", "."))
        adapter_config = self._adapter_config
        mesh_dirname = mesh_cfg.get("dirname") or mesh_cfg.get("mesh_dirname")
        if mesh_dirname is None and adapter_config is not None:
            mesh_dirname = adapter_config.mesh_dirname
        if mesh_dirname is None:
            mesh_dirname = getattr(self.mgr, "mesh_dirname", "meshes")

        mesh_extensions = mesh_cfg.get("extensions") or mesh_cfg.get("mesh_extensions")
        if mesh_extensions is None and adapter_config is not None:
            mesh_extensions = adapter_config.mesh_extensions
        if mesh_extensions is None:
            mesh_extensions = (".ply", ".obj")
        if isinstance(mesh_extensions, str):
            mesh_extensions = [mesh_extensions]

        metadata_filename = mesh_cfg.get("manifest_filename") or mesh_cfg.get("mesh_metadata_filename")
        if metadata_filename is None and adapter_config is not None:
            metadata_filename = adapter_config.mesh_metadata_filename

        allow_unlabeled = bool(getattr(self.mgr, "allow_unlabeled_data", False))
        image_extensions = getattr(
            self.mgr,
            "image_extensions",
            (".tif", ".tiff", ".png", ".jpg", ".jpeg"),
        )
        if isinstance(image_extensions, Sequence) and not isinstance(image_extensions, (str, bytes)):
            image_extensions = tuple(image_extensions)
        else:
            image_extensions = (str(image_extensions),)

        tiff_chunk_shape = getattr(self.mgr, "tiff_chunk_shape", None)
        if tiff_chunk_shape is not None:
            tiff_chunk_shape = tuple(int(v) for v in tiff_chunk_shape)

        return AdapterConfig(
            data_path=data_path,
            targets=tuple(self._primary_targets or self._resolve_primary_targets()),
            allow_unlabeled=allow_unlabeled,
            image_dirname=getattr(self.mgr, "image_dirname", "images"),
            label_dirname=getattr(self.mgr, "label_dirname", "labels"),
            image_extensions=image_extensions,
            zarr_resolution=getattr(self.mgr, "ome_zarr_resolution", None),
            tiff_chunk_shape=tiff_chunk_shape,
            mesh_dirname=str(mesh_dirname),
            mesh_extensions=tuple(mesh_extensions),
            mesh_metadata_filename=metadata_filename,
        )

    # Additional dataset helpers ------------------------------------------------------------------

    def get_labeled_unlabeled_patch_indices(self):
        """Classify patches for semi-supervised workflows.

        Returns:
            labeled_indices: Indices of patches with valid labels (for supervised loss)
            unlabeled_indices: Indices of patches that are unlabeled foreground or
                from unlabeled volumes (for consistency loss in semi-supervised training)
        """

        labeled_indices: List[int] = []
        unlabeled_indices: List[int] = []

        if not self.valid_patches:
            return labeled_indices, unlabeled_indices

        # Prefer patch-level classification via ChunkSlicer if available
        if hasattr(self, 'chunk_slicer') and self.chunk_slicer is not None:
            patches = self.chunk_slicer.patches
            for idx, patch in enumerate(patches):
                if hasattr(patch, 'is_unlabeled_fg') and patch.is_unlabeled_fg:
                    # Unlabeled foreground: has image data but no labels
                    unlabeled_indices.append(idx)
                elif hasattr(patch, 'is_bg_only') and patch.is_bg_only:
                    # BG-only patches are excluded from both lists
                    pass
                else:
                    # Labeled foreground patches
                    labeled_indices.append(idx)
            return labeled_indices, unlabeled_indices

        # Fallback: volume-level classification
        first_target = next(iter(self.target_volumes))

        for idx, patch_info in enumerate(self.valid_patches):
            vol_idx = patch_info['volume_index']

            if vol_idx < len(self.target_volumes[first_target]):
                volume_info = self.target_volumes[first_target][vol_idx]
                has_label = volume_info.get('has_label', False)
                if has_label:
                    labeled_indices.append(idx)
                else:
                    unlabeled_indices.append(idx)
            else:  # pragma: no cover - defensive
                self._runtime_logger.warning(
                    "Patch %s references missing volume index %s", idx, vol_idx
                )
                unlabeled_indices.append(idx)

        return labeled_indices, unlabeled_indices
