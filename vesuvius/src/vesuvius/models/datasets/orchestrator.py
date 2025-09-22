"""Dataset orchestrator that bridges adapters and slicers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from .adapters import (
    AdapterConfig,
    DataSourceAdapter,
    ImageAdapter,
    LoadedVolume,
    NapariAdapter,
    ZarrAdapter,
)
from .base_dataset import BaseDataset


AdapterName = str


class DatasetOrchestrator(BaseDataset):
    """A BaseDataset subclass that sources data through configurable adapters."""

    _ADAPTERS: Dict[AdapterName, type[DataSourceAdapter]] = {
        "image": ImageAdapter,
        "zarr": ZarrAdapter,
        "napari": NapariAdapter,
    }

    def __init__(
        self,
        mgr,
        *,
        adapter: AdapterName,
        adapter_kwargs: Optional[Dict[str, object]] = None,
        is_training: bool = True,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.adapter_name = adapter
        self.adapter_kwargs = adapter_kwargs or {}
        self._runtime_logger = logger or logging.getLogger(__name__)
        self._adapter: Optional[DataSourceAdapter] = None
        self._loaded_volumes: List[LoadedVolume] = []
        super().__init__(mgr, is_training=is_training)
        self.logger = self._runtime_logger

    # BaseDataset hooks ----------------------------------------------------------------------------

    def _initialize_volumes(self) -> None:
        adapter_cls = self._resolve_adapter_class(self.adapter_name)
        targets = self._resolve_primary_targets()

        config = self._build_adapter_config(targets)

        adapter = adapter_cls(config, logger=self._runtime_logger, **self.adapter_kwargs)
        discovered = adapter.discover()
        adapter.prepare(discovered)
        volumes = list(adapter.iter_volumes())
        if not volumes:
            raise ValueError("Adapter produced no volumes")

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

            for target in targets:
                label_handle = volume.labels.get(target)
                label_path = volume.metadata.label_paths.get(target)
                label_source = None
                if label_handle is not None and hasattr(label_handle, "raw"):
                    label_source = label_handle.raw()

                entry = {
                    "volume_id": volume_id,
                    "image": volume.image,
                    "label": label_handle,
                    "label_path": label_path,
                    "label_source": label_source,
                    "has_label": label_handle is not None,
                }
                self.target_volumes[target].append(entry)

                if label_handle is not None and getattr(label_handle, "raw", None) is not None and label_path is not None:
                    self.zarr_arrays.append(label_handle.raw())
                    self.zarr_names.append(f"{volume_id}_{target}")
                    self.data_paths.append(str(label_path))

    # Additional dataset helpers ------------------------------------------------------------------

    def get_labeled_unlabeled_patch_indices(self):
        """Mirror ImageDataset helper for semi-supervised workflows."""

        labeled_indices: List[int] = []
        unlabeled_indices: List[int] = []

        if not self.valid_patches:
            return labeled_indices, unlabeled_indices

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
