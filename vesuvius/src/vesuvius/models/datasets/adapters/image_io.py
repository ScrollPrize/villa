"""Streaming adapter for filesystem-backed image datasets."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import tifffile

from .base_io import (
    AdapterConfig,
    DataSourceAdapter,
    DiscoveredItem,
    LoadedVolume,
    NumpyArrayHandle,
    TiffArrayHandle,
    VolumeMetadata,
)


class ImageAdapter(DataSourceAdapter):
    """Filesystem-backed adapter that streams TIFF and raster image volumes on demand."""

    def __init__(self, config: AdapterConfig, *, logger: Optional[logging.Logger] = None) -> None:
        super().__init__(config, logger=logger)
        self._discovered: List[DiscoveredItem] = []
        self._metadata: Dict[str, VolumeMetadata] = {}
        self._tiff_exts = {".tif", ".tiff"}

    # Discovery ------------------------------------------------------------------------------------

    def discover(self) -> Sequence[DiscoveredItem]:
        images_dir = self.config.data_path / self.config.image_dirname
        labels_dir = self.config.data_path / self.config.label_dirname

        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")

        if not labels_dir.exists() and not self.config.allow_unlabeled:
            raise FileNotFoundError(
                f"Labels directory not found: {labels_dir} (allow_unlabeled={self.config.allow_unlabeled})"
            )

        ext_set = {ext.lower() for ext in self.config.image_extensions}

        image_files: Dict[str, Path] = {}
        for path in images_dir.iterdir():
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix not in ext_set:
                continue
            stem = path.stem
            if stem in image_files:
                raise ValueError(f"Duplicate image stem detected: {stem}")
            image_files[stem] = path

        if not image_files:
            raise ValueError(f"No image files found in {images_dir}")

        self.logger.info("Discovered %d image files in %s", len(image_files), images_dir)

        label_lookup: Dict[Tuple[str, str], Path] = {}
        if labels_dir.exists():
            for label_path in labels_dir.iterdir():
                if not label_path.is_file():
                    continue
                if label_path.suffix.lower() not in ext_set:
                    continue
                parts = label_path.stem.rsplit("_", 1)
                if len(parts) != 2:
                    continue
                image_id, target = parts
                label_lookup[(image_id, target)] = label_path

        discovered: List[DiscoveredItem] = []
        for volume_id, image_path in sorted(image_files.items()):
            label_paths: Dict[str, Optional[Path]] = {}
            for target in self.config.targets:
                label_paths[target] = self._resolve_label_path(
                    label_lookup=label_lookup,
                    volume_id=volume_id,
                    target=target,
                    image_path=image_path,
                )

            missing = [target for target, path in label_paths.items() if path is None]
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

        self.logger.info("Registered %d volumes for image streaming", len(discovered))
        self._discovered = discovered
        return tuple(discovered)

    def _resolve_label_path(
        self,
        *,
        label_lookup: Mapping[Tuple[str, str], Path],
        volume_id: str,
        target: str,
        image_path: Path,
    ) -> Optional[Path]:
        direct = label_lookup.get((volume_id, target))
        if direct is not None:
            return direct

        # Allow image files suffixed with _{target} to reuse same stem.
        image_stem = image_path.stem
        if image_stem != volume_id:
            embedded = label_lookup.get((image_stem, target))
            if embedded is not None:
                return embedded

        return None

    # Preparation ---------------------------------------------------------------------------------

    def prepare(self, discovered: Sequence[DiscoveredItem]) -> None:
        metadata: Dict[str, VolumeMetadata] = {}

        for item in discovered:
            image_axes, spatial_shape, image_dtype = self._probe_image(item.image_path)

            label_dtypes: Dict[str, Optional[np.dtype]] = {}
            targets_with_labels: List[str] = []

            for target, label_path in item.label_paths.items():
                if label_path is None:
                    label_dtypes[target] = None
                    continue

                label_axes, label_shape, label_dtype = self._probe_image(label_path)
                label_spatial = self._extract_spatial_shape(label_shape, label_axes)

                if label_spatial != spatial_shape:
                    raise ValueError(
                        "Label spatial shape mismatch for volume '%s' target '%s': image=%s label=%s"
                        % (item.volume_id, target, spatial_shape, label_spatial)
                    )

                label_dtypes[target] = label_dtype
                targets_with_labels.append(target)

            metadata[item.volume_id] = VolumeMetadata(
                volume_id=item.volume_id,
                image_path=item.image_path,
                spatial_shape=spatial_shape,
                dtype=image_dtype,
                axes=image_axes,
                targets_with_labels=tuple(targets_with_labels),
                label_paths=dict(item.label_paths),
                label_dtypes=label_dtypes,
            )

        self._metadata = metadata
        self.logger.info("Prepared metadata for %d volumes", len(metadata))

    def _probe_image(self, path: Path) -> Tuple[str, Tuple[int, ...], np.dtype]:
        if self._is_tiff(path):
            with tifffile.TiffFile(str(path)) as tif:
                series = tif.series[0]
                axes = series.axes
                shape = tuple(int(dim) for dim in series.shape)
                dtype = np.dtype(series.dtype)
            spatial_shape = self._extract_spatial_shape(shape, axes)
            return axes, spatial_shape, dtype

        array = self._read_raster(path)
        axes = "YX"
        spatial_shape = tuple(int(dim) for dim in array.shape)
        dtype = array.dtype
        return axes, spatial_shape, dtype

    def _is_tiff(self, path: Path) -> bool:
        return path.suffix.lower() in self._tiff_exts

    _SPATIAL_AXIS_ALIASES = {
        "X": "X",
        "Y": "Y",
        "Z": "Z",
        "Q": "Z",  # Some TIFF stacks encode depth as 'Q'
    }

    def _extract_spatial_shape(self, shape: Sequence[int], axes: str) -> Tuple[int, ...]:
        spatial: List[int] = []
        for dim, axis in zip(shape, axes):
            alias = self._SPATIAL_AXIS_ALIASES.get(axis.upper())
            if alias in {"Z", "Y", "X"}:
                spatial.append(int(dim))
        if not spatial:
            raise ValueError(f"Unable to determine spatial axes from axes='{axes}' shape={shape}")
        return tuple(spatial)

    def _read_raster(self, path: Path) -> np.ndarray:
        array = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if array is None:
            raise ValueError(f"Failed to read image file: {path}")
        if array.ndim == 3:
            array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
        return np.asarray(array)

    # Materialisation -----------------------------------------------------------------------------

    def iter_volumes(self) -> Iterator[LoadedVolume]:
        if not self._discovered:
            raise RuntimeError("ImageAdapter.iter_volumes called before discover()")
        if not self._metadata:
            raise RuntimeError("ImageAdapter.iter_volumes called before prepare()")

        for item in self._discovered:
            metadata = self._metadata[item.volume_id]

            image_handle = self._build_array_handle(
                metadata.image_path,
                spatial_shape=metadata.spatial_shape,
                dtype=metadata.dtype,
            )

            label_handles: Dict[str, Optional[TiffArrayHandle]] = {}
            for target, label_path in metadata.label_paths.items():
                dtype = metadata.label_dtypes.get(target)
                if label_path is None or dtype is None:
                    label_handles[target] = None
                    continue
                label_handles[target] = self._build_array_handle(
                    label_path,
                    spatial_shape=metadata.spatial_shape,
                    dtype=dtype,
                )

            yield LoadedVolume(
                metadata=metadata,
                image=image_handle,
                labels=label_handles,
            )

    def _build_array_handle(
        self,
        path: Path,
        *,
        spatial_shape: Tuple[int, ...],
        dtype: np.dtype,
    ) -> TiffArrayHandle | NumpyArrayHandle:
        if self._is_tiff(path):
            return TiffArrayHandle(
                path,
                spatial_shape=spatial_shape,
                dtype=dtype,
                chunk_shape=self.config.tiff_chunk_shape,
            )

        array = self._read_raster(path)
        if array.shape != spatial_shape:
            raise ValueError(
                f"Raster shape mismatch for {path}: expected {spatial_shape}, observed {array.shape}"
            )
        return NumpyArrayHandle(array, path=path, spatial_shape=spatial_shape)
