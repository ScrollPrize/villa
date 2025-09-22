"""Napari viewer adapter for in-memory datasets."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - optional dependency
    import napari  # type: ignore
    from napari.layers import Image as NapariImageLayer  # type: ignore
    from napari.layers import Labels as NapariLabelsLayer  # type: ignore
except Exception:  # pragma: no cover - import guard for test environments
    napari = None
    NapariImageLayer = None
    NapariLabelsLayer = None

from .base_io import (
    AdapterConfig,
    DataSourceAdapter,
    DiscoveredItem,
    LoadedVolume,
    NumpyArrayHandle,
    VolumeMetadata,
)


class NapariAdapter(DataSourceAdapter):
    """Ingests image/label layers from a napari viewer instance."""

    def __init__(
        self,
        config: AdapterConfig,
        *,
        viewer=None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        super().__init__(config, logger=logger)
        self._viewer = viewer
        self._discovered: List[DiscoveredItem] = []
        self._metadata: Dict[str, VolumeMetadata] = {}
        self._image_arrays: Dict[str, np.ndarray] = {}
        self._label_arrays: Dict[Tuple[str, str], np.ndarray] = {}

    # Discovery ------------------------------------------------------------------------------------

    def discover(self) -> Sequence[DiscoveredItem]:
        viewer = self._resolve_viewer()
        layers = list(getattr(viewer, "layers", []))
        if not layers:
            raise ValueError("Napari viewer contains no layers")

        image_layers = [layer for layer in layers if self._is_image_layer(layer)]
        if not image_layers:
            raise ValueError("No image layers found in napari viewer")

        label_lookup: Dict[str, Dict[str, object]] = {}
        for layer in layers:
            if not self._is_label_layer(layer):
                continue
            for image_layer in image_layers:
                prefix = f"{image_layer.name}_"
                if layer.name.startswith(prefix):
                    target = layer.name[len(prefix) :]
                    label_lookup.setdefault(image_layer.name, {})[target] = layer

        discovered: List[DiscoveredItem] = []
        for image_layer in image_layers:
            volume_id = image_layer.name
            target_map: Dict[str, Optional[Path]] = {}
            available_labels = label_lookup.get(image_layer.name, {})

            for target in self.config.targets:
                label_layer = available_labels.get(target)
                if label_layer is None:
                    target_map[target] = None
                else:
                    fake_path = Path(f"/napari/{volume_id}_{target}")
                    target_map[target] = fake_path

            missing = [t for t, p in target_map.items() if p is None]
            if missing and not self.config.allow_unlabeled:
                raise ValueError(
                    f"Volume '{volume_id}' is missing labels for targets {missing} (allow_unlabeled=False)"
                )

            discovered.append(
                DiscoveredItem(
                    volume_id=volume_id,
                    image_path=Path(f"/napari/{volume_id}"),
                    label_paths=target_map,
                )
            )

        if not discovered:
            raise ValueError("No napari volumes discovered")

        self._discovered = discovered
        self.logger.info("Registered %d napari volumes", len(discovered))
        return tuple(discovered)

    def _resolve_viewer(self):
        if self._viewer is not None:
            return self._viewer
        if napari is None:
            raise RuntimeError("napari is not installed and no viewer was provided")
        viewer = napari.current_viewer()  # type: ignore[attr-defined]
        if viewer is None:
            raise RuntimeError("No active napari viewer found")
        return viewer

    def _is_image_layer(self, layer) -> bool:
        if NapariImageLayer is not None and isinstance(layer, NapariImageLayer):
            return True
        return getattr(layer, "layer_type", None) == "image"

    def _is_label_layer(self, layer) -> bool:
        if NapariLabelsLayer is not None and isinstance(layer, NapariLabelsLayer):
            return True
        return getattr(layer, "layer_type", None) == "labels"

    # Preparation ---------------------------------------------------------------------------------

    def prepare(self, discovered: Sequence[DiscoveredItem]) -> None:
        viewer = self._resolve_viewer()
        layer_map = {layer.name: layer for layer in getattr(viewer, "layers", [])}

        metadata: Dict[str, VolumeMetadata] = {}
        for item in discovered:
            image_layer = layer_map.get(item.volume_id)
            if image_layer is None:
                raise ValueError(f"Image layer '{item.volume_id}' no longer present in viewer")

            image_array = np.asarray(image_layer.data)
            spatial_shape = self._extract_spatial_shape(image_array.shape)
            axes = self._derive_axes(image_array.shape)

            label_dtypes: Dict[str, Optional[np.dtype]] = {}
            targets_with_labels: List[str] = []

            for target, label_path in item.label_paths.items():
                if label_path is None:
                    label_dtypes[target] = None
                    continue

                label_layer_name = f"{item.volume_id}_{target}"
                label_layer = layer_map.get(label_layer_name)
                if label_layer is None:
                    raise ValueError(
                        f"Expected label layer '{label_layer_name}' missing from viewer"
                    )
                label_array = np.asarray(label_layer.data)
                label_spatial = self._extract_spatial_shape(label_array.shape)
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
        self.logger.info("Prepared metadata for %d napari volumes", len(metadata))

    def _extract_spatial_shape(self, shape: Sequence[int]) -> Tuple[int, ...]:
        if len(shape) == 2:
            return tuple(int(v) for v in shape)
        if len(shape) == 3:
            return tuple(int(v) for v in shape)
        if len(shape) >= 4:
            return tuple(int(v) for v in shape[-3:])
        raise ValueError(f"Unsupported array shape {shape} for spatial inference")

    def _derive_axes(self, shape: Sequence[int]) -> str:
        if len(shape) == 2:
            return "YX"
        if len(shape) == 3:
            return "ZYX"
        if len(shape) == 4:
            return "CZYX"
        if len(shape) == 5:
            return "TCZYX"
        return "?"

    # Materialisation -----------------------------------------------------------------------------

    def iter_volumes(self) -> Iterator[LoadedVolume]:
        if not self._discovered:
            raise RuntimeError("NapariAdapter.iter_volumes called before discover()")
        if not self._metadata:
            raise RuntimeError("NapariAdapter.iter_volumes called before prepare()")

        for item in self._discovered:
            metadata = self._metadata[item.volume_id]
            image_array = self._image_arrays[metadata.volume_id]

            image_handle = NumpyArrayHandle(
                image_array,
                path=metadata.image_path,
                spatial_shape=metadata.spatial_shape,
            )

            label_handles: Dict[str, Optional[NumpyArrayHandle]] = {}
            for target, label_path in metadata.label_paths.items():
                if label_path is None:
                    label_handles[target] = None
                    continue
                label_array = self._label_arrays[(metadata.volume_id, target)]
                label_handles[target] = NumpyArrayHandle(
                    label_array,
                    path=label_path,
                    spatial_shape=metadata.spatial_shape,
                )

            yield LoadedVolume(
                metadata=metadata,
                image=image_handle,
                labels=label_handles,
            )
