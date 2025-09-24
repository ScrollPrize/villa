"""Shared interfaces and dataclasses for dataset I/O adapters."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Iterator, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class AdapterConfig:
    """Common configuration surfaced to data-source adapters."""

    data_path: Path
    targets: Sequence[str]
    allow_unlabeled: bool = False
    image_dirname: str = "images"
    label_dirname: str = "labels"
    image_extensions: Sequence[str] = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
    zarr_resolution: Optional[int] = None
    tiff_chunk_shape: Optional[Tuple[int, ...]] = None


@dataclass(frozen=True)
class VolumeMetadata:
    """Describes the physical characteristics of a discovered volume."""

    volume_id: str
    image_path: Path
    spatial_shape: Tuple[int, ...]
    dtype: np.dtype
    axes: str
    targets_with_labels: Sequence[str]
    label_paths: Mapping[str, Optional[Path]]
    label_dtypes: Mapping[str, Optional[np.dtype]]


@dataclass(frozen=True)
class DiscoveredItem:
    """Represents a raw filesystem discovery prior to adapter preparation."""

    volume_id: str
    image_path: Path
    label_paths: Mapping[str, Optional[Path]]


class ArrayHandle(ABC):
    """Abstract interface for lazily reading image or label payloads."""

    def __init__(self, path: Path, *, spatial_shape: Tuple[int, ...], dtype: np.dtype) -> None:
        self._path = Path(path)
        self._spatial_shape = tuple(int(v) for v in spatial_shape)
        self._dtype = np.dtype(dtype)

    @property
    def path(self) -> Path:
        return self._path

    @property
    def spatial_shape(self) -> Tuple[int, ...]:
        return self._spatial_shape

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @abstractmethod
    def read(self) -> np.ndarray:
        """Eagerly materialize the full payload into memory."""

    @abstractmethod
    def read_window(self, start: Sequence[int], size: Sequence[int]) -> np.ndarray:
        """Materialize a spatial window without loading the full array."""

    def close(self) -> None:
        """Allow adapters to release any resources; optional by default."""

    def raw(self):  # pragma: no cover - default implementation
        """Expose underlying array-like object when available."""
        return None


class TiffArrayHandle(ArrayHandle):
    """Windowed reader backed by a TIFF file via ``tifffile.aszarr``."""

    def __init__(
        self,
        path: Path,
        *,
        spatial_shape: Tuple[int, ...],
        dtype: np.dtype,
        chunk_shape: Optional[Tuple[int, ...]] = None,
    ) -> None:
        super().__init__(path, spatial_shape=spatial_shape, dtype=dtype)
        self._lock = Lock()
        self._chunk_shape = chunk_shape

    def read(self) -> np.ndarray:
        return self._read_region(tuple(slice(0, dim) for dim in self._spatial_shape))

    def read_window(self, start: Sequence[int], size: Sequence[int]) -> np.ndarray:
        if len(start) != len(size):
            raise ValueError("start and size must have identical dimensionality")
        if len(start) != len(self._spatial_shape):
            raise ValueError(
                f"Window dimensionality {len(start)} does not match array ndim {len(self._spatial_shape)}"
            )
        base_region = tuple(slice(int(s), int(s) + int(l)) for s, l in zip(start, size))
        return self._read_region(base_region)

    def _read_region(self, base_region: Tuple[slice, ...]) -> np.ndarray:
        import tifffile

        with self._lock:
            with tifffile.TiffFile(str(self._path)) as tif:
                try:
                    zarr_store = tif.aszarr(chunkshape=self._chunk_shape) if self._chunk_shape else tif.aszarr()
                    region = self._expand_region(base_region, getattr(zarr_store, 'ndim', None))
                    data = zarr_store[region]
                except Exception:
                    full = tif.asarray()
                    region = self._expand_region(base_region, full.ndim)
                    data = full[region]
                else:
                    data = np.asarray(data)
                data = np.asarray(data, dtype=self._dtype)
        return np.ascontiguousarray(data)

    @staticmethod
    def _expand_region(base_region: Tuple[slice, ...], ndim: Optional[int]) -> Tuple[slice, ...]:
        if ndim is None or len(base_region) == ndim:
            return base_region
        if len(base_region) > ndim:
            raise ValueError("Region dimensionality exceeds array dimensionality")
        prefix = (slice(None),) * (ndim - len(base_region))
        return prefix + base_region


class ZarrArrayHandle(ArrayHandle):
    """Windowed reader backed by a zarr array object."""

    def __init__(self, array, *, path: Path, spatial_shape: Tuple[int, ...]) -> None:
        self._array = array
        self._shape = tuple(int(v) for v in getattr(array, "shape", ()))
        super().__init__(path, spatial_shape=spatial_shape, dtype=getattr(array, "dtype", np.float32))
        self._spatial_ndim = len(self._spatial_shape)
        self._orientation, self._extra_shape = self._infer_orientation(self._shape, self._spatial_shape)

    def read(self) -> np.ndarray:
        data = np.asarray(self._array[...], dtype=self._dtype)
        data = self._to_channel_first(data)
        return np.ascontiguousarray(data)

    def read_window(self, start: Sequence[int], size: Sequence[int]) -> np.ndarray:
        if len(start) != len(size):
            raise ValueError("start and size must have identical dimensionality")
        base_region = tuple(slice(int(s), int(s) + int(l)) for s, l in zip(start, size))
        region = self._expand_region(base_region)
        data = np.asarray(self._array[region], dtype=self._dtype)
        data = self._to_channel_first(data)
        return np.ascontiguousarray(data)

    def raw(self):  # pragma: no cover - simple accessor
        return self._array

    def _expand_region(self, base_region: Tuple[slice, ...]) -> Tuple[slice, ...]:
        ndim = getattr(self._array, 'ndim', len(base_region))
        if len(base_region) > ndim:
            raise ValueError("Region dimensionality exceeds array dimensionality")

        spatial_ndim = self._spatial_ndim
        if ndim == len(base_region):
            return base_region

        shape = self._shape
        if spatial_ndim and len(shape) == ndim:
            front = tuple(int(v) for v in shape[:spatial_ndim])
            back = tuple(int(v) for v in shape[-spatial_ndim:])

            if front == self._spatial_shape:
                # spatial dimensions lead the array; append remaining axes
                suffix = (slice(None),) * (ndim - spatial_ndim)
                return base_region + suffix

            if back == self._spatial_shape:
                # spatial dimensions trail the array; prepend leading axes
                prefix = (slice(None),) * (ndim - spatial_ndim)
                return prefix + base_region

        prefix = (slice(None),) * (ndim - len(base_region))
        return prefix + base_region

    @staticmethod
    def _infer_orientation(
        shape: Tuple[int, ...], spatial: Tuple[int, ...]
    ) -> Tuple[str, Tuple[int, ...]]:
        spatial_ndim = len(spatial)

        if not shape or not spatial:
            return ("unknown", ())

        if len(shape) == spatial_ndim:
            return ("none", ())

        front = tuple(int(v) for v in shape[:spatial_ndim])
        back = tuple(int(v) for v in shape[-spatial_ndim:])

        if front == spatial:
            return ("spatial_leading", tuple(int(v) for v in shape[spatial_ndim:]))
        if back == spatial:
            return ("spatial_trailing", tuple(int(v) for v in shape[:-spatial_ndim]))

        return ("unknown", tuple(int(v) for v in shape))

    def _to_channel_first(self, data: np.ndarray) -> np.ndarray:
        if not self._extra_shape:
            return data

        orientation = self._orientation
        spatial_ndim = self._spatial_ndim

        extra_size = int(np.prod(self._extra_shape))

        if orientation == "spatial_leading":
            # spatial dims lead the array, extras trail.
            spatial_shape = data.shape[:spatial_ndim]
            reshaped = data.reshape(spatial_shape + (extra_size,))
            return np.moveaxis(reshaped, -1, 0)

        if orientation == "spatial_trailing":
            # extras lead the array, spatial dims trail.
            spatial_shape = data.shape[-spatial_ndim:]
            reshaped = data.reshape((extra_size,) + spatial_shape)
            return reshaped

        # Unknown orientation; fall back to original layout
        return data


class NumpyArrayHandle(ArrayHandle):
    """Simple array handle wrapping an in-memory numpy array."""

    def __init__(self, array: np.ndarray, *, path: Path, spatial_shape: Tuple[int, ...]) -> None:
        self._array = np.asarray(array)
        super().__init__(path, spatial_shape=spatial_shape, dtype=self._array.dtype)

    def read(self) -> np.ndarray:
        return np.ascontiguousarray(self._array.astype(self._dtype, copy=False))

    def read_window(self, start: Sequence[int], size: Sequence[int]) -> np.ndarray:
        if len(start) != len(size):
            raise ValueError("start and size must have identical dimensionality")
        slices = tuple(slice(int(s), int(s) + int(l)) for s, l in zip(start, size))
        data = self._array[tuple(slices)]
        return np.ascontiguousarray(np.asarray(data, dtype=self._dtype))

@dataclass(frozen=True)
class LoadedVolume:
    """Normalized structure emitted by adapters for downstream slicers."""

    metadata: VolumeMetadata
    image: ArrayHandle
    labels: Mapping[str, Optional[ArrayHandle]]

    def close(self) -> None:
        self.image.close()
        for handle in self.labels.values():
            if handle is not None:
                handle.close()


class DataSourceAdapter(ABC):
    """Base class for all dataset I/O providers."""

    def __init__(self, config: AdapterConfig, *, logger: Optional[logging.Logger] = None) -> None:
        self.config = config
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self._discovered: Sequence[DiscoveredItem] = ()
        self._prepared: MutableMapping[str, VolumeMetadata] = {}

    @abstractmethod
    def discover(self) -> Sequence[DiscoveredItem]:
        """Scan backing storage and return discovered entries."""

    @abstractmethod
    def prepare(self, discovered: Sequence[DiscoveredItem]) -> None:
        """Perform validation/metadata gathering ahead of materialization."""

    @abstractmethod
    def iter_volumes(self) -> Iterator[LoadedVolume]:
        """Yield fully-initialized ``LoadedVolume`` records."""

    def run(self) -> Iterator[LoadedVolume]:
        """Convenience helper to run the full discovery pipeline."""

        self.logger.debug("Starting adapter discovery")
        discovered = self.discover()
        self.logger.debug("Discovered %d entries", len(discovered))
        self.prepare(discovered)
        return self.iter_volumes()
