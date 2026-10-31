"""Writing local Zarr arrays and groups: NumPy indexing and conversion over ``vcr::zarr::Writer``."""

from __future__ import annotations

import os
from collections.abc import MutableMapping
from typing import Any, Iterator, Sequence

import numpy as np

from .. import _reader
from ._indexing import parse_index
from .zarr_reader import ZarrReader


def _plain(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


class _Attributes(MutableMapping):
    """A node's attributes: every change is written, and only this mapping's keys change."""

    def __init__(self, path: str):
        self._path = path

    def _read(self) -> dict[str, Any]:
        return _reader.read_attributes(self._path)

    def __getitem__(self, key: str) -> Any:
        return self._read()[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.update({key: value})

    def __delitem__(self, key: str) -> None:
        attributes = self._read()
        del attributes[key]
        _reader.write_attributes(self._path, attributes)

    def __iter__(self) -> Iterator[str]:
        return iter(self._read())

    def __len__(self) -> int:
        return len(self._read())

    def update(self, *args: Any, **kwargs: Any) -> None:
        attributes = self._read()
        attributes.update(*args, **kwargs)
        _reader.write_attributes(self._path, attributes)

    def put(self, attributes: dict[str, Any]) -> None:
        """Replace all attributes."""
        _reader.write_attributes(self._path, dict(attributes))


class ZarrWriter:
    """A local Zarr v2/v3 array opened for writing. Use :meth:`create` or :meth:`open`.

    Chunk files are replaced atomically, so readers never see a partial chunk. Distinct
    chunks may be written from any number of threads or processes; partial writes to one
    chunk, and processes writing one shard, need the caller's coordination. Picklable.
    """

    def __init__(self, native, skip_empty: bool = False):
        self._native = native
        self.path = native.path
        self.skip_empty = skip_empty
        self.attrs = _Attributes(self.path)
        self._reader: ZarrReader | None = None

    @classmethod
    def create(
        cls,
        path: str | os.PathLike[str],
        *,
        shape: Sequence[int],
        chunks: Sequence[int],
        dtype: Any,
        compressor: str | None = "blosc",
        cname: str = "lz4",
        shuffle: int = 1,
        level: int = -1,
        separator: str = ".",
        fill_value: Any = 0,
        version: int = 2,
        shards: Sequence[int] | None = None,
        overwrite: bool = False,
        skip_empty: bool = False,
    ) -> ZarrWriter:
        """Create an array. ``compressor``: None, blosc, zstd, zlib, gzip, lz4, vc-delta3d or volcomp;
        ``cname`` and ``shuffle`` (0 none, 1 byte, 2 bit) configure blosc. ``level`` <= 0 is the codec's
        default. ``overwrite`` replaces whatever is at the path. ``skip_empty`` does not store chunks that
        come out entirely ``fill_value``. ``shards`` is a v3 array's outer shape in elements.
        """
        native = _reader.Writer.create(
            os.fspath(path), [int(n) for n in shape], [int(n) for n in chunks], np.dtype(dtype).str,
            compressor or "none", int(level), cname, int(shuffle), bool(overwrite), separator,
            _plain(fill_value), int(version), [int(n) for n in shards or ()])
        return cls(native, skip_empty)

    @classmethod
    def open(cls, path: str | os.PathLike[str], *, skip_empty: bool = False) -> ZarrWriter:
        """An existing local array."""
        return cls(_reader.Writer.open(os.fspath(path)), skip_empty)

    def __getstate__(self):
        return self.path, self.skip_empty

    def __setstate__(self, state):
        self.__init__(_reader.Writer.open(state[0]), state[1])

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self._native.meta.shape)

    @property
    def chunks(self) -> tuple[int, ...]:
        return tuple(self._native.meta.chunks)

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self._native.meta.dtype)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def fill_value(self) -> Any:
        return np.frombuffer(self._native.meta.fill, self.dtype, 1)[0]

    def __setitem__(self, key: Any, value: Any) -> None:
        start, size, squeeze = parse_index(key, self.shape)
        kept = tuple(n for n, s in zip(size, squeeze) if not s)
        data = np.ascontiguousarray(np.broadcast_to(np.asarray(value, self.dtype), kept)).reshape(size)
        self._native.write_region(start, size, data, self.skip_empty)

    def __getitem__(self, key: Any) -> np.ndarray:
        """What is stored now, by the indexing of :class:`ZarrReader`."""
        if self._reader is None:
            self._reader = ZarrReader.open(self.path, read_mode="slab")
        return self._reader[key]

    def set_fill_value(self, value: Any) -> None:
        """Change the fill of absent chunks, keeping stored ones. No other writer may be in use."""
        self._native.set_fill_value(_plain(value))
        self._reader = None

    def chunk_exists(self, chunk: Sequence[int]) -> bool:
        return self._native.chunk_exists([int(n) for n in chunk])

    def write_chunk(self, chunk: Sequence[int], data: Any) -> None:
        """One whole decoded chunk, without reading anything."""
        array = np.ascontiguousarray(np.asarray(data, self.dtype))
        self._native.write_chunk([int(n) for n in chunk], array, self.skip_empty)

    def remove_chunk(self, chunk: Sequence[int]) -> bool:
        return self._native.remove_chunk([int(n) for n in chunk])


class ZarrWriteGroup:
    """A local Zarr group opened for writing. Use :func:`create_group`."""

    def __init__(self, path: str, version: int):
        self.path = path
        self.version = version
        self.attrs = _Attributes(path)

    def create_array(self, name: str, **kwargs: Any) -> ZarrWriter:
        """A child array: the arguments of :meth:`ZarrWriter.create`."""
        kwargs.setdefault("version", self.version)
        return ZarrWriter.create(os.path.join(self.path, name), **kwargs)

    def create_group(self, name: str) -> ZarrWriteGroup:
        """A child group, kept when it exists."""
        return create_group(os.path.join(self.path, name), version=self.version)

    def _child(self, name: str) -> str:
        return os.path.join(self.path, name)

    def __contains__(self, name: str) -> bool:
        return _reader.is_zarr_node(self._child(name))

    def __getitem__(self, name: str) -> ZarrWriter | ZarrWriteGroup:
        child = self._child(name)
        if not _reader.is_zarr_node(child):
            raise KeyError(name)
        if _reader.is_zarr_array(child):
            return ZarrWriter.open(child)
        return ZarrWriteGroup(child, self.version)

    def __delitem__(self, name: str) -> None:
        if not _reader.remove_node(self._child(name)):
            raise KeyError(name)

    def keys(self) -> list[str]:
        return _reader.node_children(self.path)

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())


def create_group(path: str | os.PathLike[str], *, attrs: dict[str, Any] | None = None,
                 version: int = 2, overwrite: bool = False) -> ZarrWriteGroup:
    """A local group, kept when it exists unless ``overwrite``; ``attrs`` replace its attributes."""
    path = os.fspath(path)
    if overwrite:
        _reader.remove_node(path)
    _reader.create_group(path, attrs, version)
    return ZarrWriteGroup(path, version)


__all__ = ["ZarrWriter", "ZarrWriteGroup", "create_group"]
