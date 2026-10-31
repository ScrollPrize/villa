"""Writing Zarr arrays and groups, local or on S3: NumPy indexing and conversion over ``vcr::zarr::Writer``.

Remote locations are ``s3://`` URLs. ``auth`` is an :class:`S3Credentials`, or ``"aws"`` (the
default) for AWS credentials from the environment; see :mod:`vc_reader.reader.auth`.
"""

from __future__ import annotations

import os
from collections.abc import MutableMapping
from typing import Any, Iterator, Sequence

import numpy as np

from .. import _reader
from ._indexing import parse_index
from .auth import resolve_access
from .zarr_reader import ZarrReader


def _plain(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _write_access(location: str, auth: Any):
    return resolve_access(location, "aws" if auth == "auto" else auth)


def compress_bytes(data: Any, compressor: dict[str, Any] | None, *, itemsize: int = 1, shape: Sequence[int] = ()) -> bytes:
    """``data``, any contiguous buffer, encoded by a numcodecs compressor config, such as ``{"id": "blosc", "cname": "zstd", "clevel": 1,
    "shuffle": 1}``, as elements of ``itemsize`` bytes. ``shape`` is only read by vc-delta3d and volcomp.
    """
    return _reader.compress_bytes(np.frombuffer(data, np.uint8), compressor, int(itemsize), [int(n) for n in shape])


def decompress_bytes(data: Any, compressor: dict[str, Any] | None, size: int) -> bytes:
    """The reverse of :func:`compress_bytes`, decoding to exactly ``size`` bytes."""
    return _reader.decompress_bytes(np.frombuffer(data, np.uint8), compressor, int(size))


class ObjectStore:
    """The objects below a local directory or an ``s3://`` prefix, by relative name.

    Replacing an object is atomic. Reads of a public prefix work with ``auth=None``. Picklable.
    """

    def __init__(self, location: str | os.PathLike[str], *, auth: Any = "aws", connections: int = 16):
        location = os.fspath(location)
        self._native = _reader.Store(location, _write_access(location, auth), connections)
        self.location = self._native.location

    def __getstate__(self):
        return self.location, self._native.access

    def __setstate__(self, state):
        self.__init__(state[0], auth=state[1])

    def read(self, name: str) -> bytes | None:
        """The object's bytes; None when it does not exist."""
        return self._native.read(name)

    def write(self, name: str, data: bytes | str) -> None:
        self._native.write(name, data.encode() if isinstance(data, str) else bytes(data))

    def exists(self, name: str) -> bool:
        return self._native.exists(name)

    def remove(self, name: str) -> bool:
        """Whether the object was there."""
        return self._native.remove(name)

    def remove_all(self) -> bool:
        """Removes everything below the location; whether anything was there."""
        return self._native.remove_all()

    def keys(self) -> list[str]:
        """Every object below the location, relative to it, in key order."""
        return [key for key, _ in self._native.list()]

    def sizes(self) -> dict[str, int]:
        return dict(self._native.list())

    def children(self) -> list[str]:
        """The immediate subdirectories (prefixes), sorted."""
        return self._native.children()


class _Attributes(MutableMapping):
    """A node's attributes: every change is written, and only this mapping's keys change."""

    def __init__(self, path: str, access):
        self._path = path
        self._access = access

    def _read(self) -> dict[str, Any]:
        return _reader.read_attributes(self._path, self._access)

    def _write(self, attributes: dict[str, Any]) -> None:
        _reader.write_attributes(self._path, attributes, self._access)

    def __getitem__(self, key: str) -> Any:
        return self._read()[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.update({key: value})

    def __delitem__(self, key: str) -> None:
        attributes = self._read()
        del attributes[key]
        self._write(attributes)

    def __iter__(self) -> Iterator[str]:
        return iter(self._read())

    def __len__(self) -> int:
        return len(self._read())

    def update(self, *args: Any, **kwargs: Any) -> None:
        attributes = self._read()
        attributes.update(*args, **kwargs)
        self._write(attributes)

    def put(self, attributes: dict[str, Any]) -> None:
        """Replace all attributes."""
        self._write(dict(attributes))


class ZarrWriter:
    """A Zarr v2/v3 array, local or on S3, opened for writing. Use :meth:`create` or :meth:`open`.

    Chunks are replaced atomically, so readers never see a partial chunk. Distinct chunks
    may be written from any number of threads or processes; partial writes to one chunk,
    and processes writing one shard, need the caller's coordination. Picklable.
    """

    def __init__(self, native, skip_empty: bool = False, access=None):
        self._native = native
        self.path = native.path
        self.skip_empty = skip_empty
        self._access = access if access is not None else _reader.anonymous_access()
        self.attrs = _Attributes(self.path, self._access)
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
        packbits: bool = False,
        overwrite: bool = False,
        skip_empty: bool = False,
        auth: Any = "aws",
    ) -> ZarrWriter:
        """Create an array. ``compressor``: None, blosc, zstd, zlib, gzip, lz4, vc-delta3d or volcomp;
        ``cname`` and ``shuffle`` (0 none, 1 byte, 2 bit) configure blosc. ``level`` <= 0 is the codec's
        default. ``overwrite`` replaces whatever is at the path. ``skip_empty`` does not store chunks that
        come out entirely ``fill_value``. ``shards`` is a v3 array's outer shape in elements. ``packbits``
        stores a bool array's elements as bits, in front of the compressor.
        """
        path = os.fspath(path)
        access = _write_access(path, auth)
        native = _reader.Writer.create(
            path, [int(n) for n in shape], [int(n) for n in chunks], np.dtype(dtype).str,
            compressor or "none", int(level), cname, int(shuffle), bool(overwrite), separator,
            _plain(fill_value), int(version), [int(n) for n in shards or ()], bool(packbits), access)
        return cls(native, skip_empty, access)

    @classmethod
    def create_like(
        cls,
        path: str | os.PathLike[str],
        source: ZarrReader | ZarrWriter,
        *,
        shape: Sequence[int] | None = None,
        chunks: Sequence[int] | None = None,
        dtype: Any = None,
        fill_value: Any = None,
        separator: str | None = None,
        overwrite: bool = False,
        skip_empty: bool = False,
        auth: Any = "aws",
    ) -> ZarrWriter:
        """An array like ``source``: its shape, chunks, dtype, codec, separator, fill, format, shards
        and packbits, except for what is given here. Attributes are not kept.
        """
        path = os.fspath(path)
        access = _write_access(path, auth)
        like = source._native.meta if isinstance(source, ZarrWriter) else source.meta._native
        native = _reader.Writer.create_like(
            path, like, None if shape is None else [int(n) for n in shape],
            None if chunks is None else [int(n) for n in chunks], None if dtype is None else np.dtype(dtype).str,
            _plain(fill_value), separator, bool(overwrite), access)
        return cls(native, skip_empty, access)

    @classmethod
    def open(cls, path: str | os.PathLike[str], *, skip_empty: bool = False, auth: Any = "aws") -> ZarrWriter:
        """An existing array."""
        path = os.fspath(path)
        access = _write_access(path, auth)
        return cls(_reader.Writer.open(path, access), skip_empty, access)

    def __getstate__(self):
        return self.path, self.skip_empty, self._access

    def __setstate__(self, state):
        self.__init__(_reader.Writer.open(state[0], state[2]), state[1], state[2])

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
            self._reader = ZarrReader.open(self.path, read_mode="slab", auth=self._access)
        return self._reader[key]

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        """What is stored now, as with ``writer[...]``."""
        if copy is False:
            raise ValueError("reading a Zarr array requires a copy")
        values = self[...]
        return values if dtype is None else values.astype(dtype, copy=False)

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
    """A Zarr group, local or on S3, opened for writing. Use :func:`create_group`."""

    def __init__(self, path: str, version: int, access):
        self.path = path
        self.version = version
        self._access = access
        self.attrs = _Attributes(path, access)

    def create_array(self, name: str, **kwargs: Any) -> ZarrWriter:
        """A child array: the arguments of :meth:`ZarrWriter.create`."""
        kwargs.setdefault("version", self.version)
        kwargs.setdefault("auth", self._access)
        return ZarrWriter.create(os.path.join(self.path, name), **kwargs)

    def create_group(self, name: str) -> ZarrWriteGroup:
        """A child group, kept when it exists."""
        return create_group(os.path.join(self.path, name), version=self.version, auth=self._access)

    def _child(self, name: str) -> str:
        return os.path.join(self.path, name)

    def _node(self, name: str):
        return _reader.Store(self._child(name), self._access, 1)

    def __contains__(self, name: str) -> bool:
        return self._node(name).is_node()

    def __getitem__(self, name: str) -> ZarrWriter | ZarrWriteGroup:
        node = self._node(name)
        if not node.is_node():
            raise KeyError(name)
        if node.is_array():
            return ZarrWriter.open(self._child(name), auth=self._access)
        return ZarrWriteGroup(self._child(name), self.version, self._access)

    def __delitem__(self, name: str) -> None:
        if not _reader.remove_node(self._child(name), self._access):
            raise KeyError(name)

    def keys(self) -> list[str]:
        return _reader.node_children(self.path, self._access)

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())


def create_group(path: str | os.PathLike[str], *, attrs: dict[str, Any] | None = None,
                 version: int = 2, overwrite: bool = False, auth: Any = "aws") -> ZarrWriteGroup:
    """A group, kept when it exists unless ``overwrite``; ``attrs`` replace its attributes."""
    path = os.fspath(path)
    access = _write_access(path, auth)
    if overwrite:
        _reader.remove_node(path, access)
    _reader.create_group(path, attrs, version, access)
    return ZarrWriteGroup(path, version, access)


def create_pyramid_levels(root: str | os.PathLike[str], levels: int, *, factors: Sequence[int] = (2, 2, 2),
                          overwrite: bool = False) -> None:
    """Create local arrays ``root/1 .. root/levels`` (metadata only) to go with ``root/0``: its dtype and
    codec, its chunks capped to each level's shape, each shape the level below divided by ``factors``, rounded up.
    """
    _reader.create_pyramid_levels(os.fspath(root), int(levels), [int(f) for f in factors], bool(overwrite))


def write_pyramid_level(root: str | os.PathLike[str], level: int, *, factors: Sequence[int] = (2, 2, 2),
                        reduction: str = "mean", threads: int = 0,
                        chunk_range: tuple[Sequence[int], Sequence[int]] | None = None) -> None:
    """Write local ``root/level`` from the level below. ``reduction``: ``mean`` (uint8, uint16, float32;
    integers round to nearest, ties to even), ``nearest`` (a block's first voxel, any dtype) or ``mode``
    (most common nonzero value, smallest on a tie; integers). Chunks entirely fill are not stored. ``chunk_range`` is
    ``(begin, end)`` in output chunks, so separate processes can write disjoint ranges.
    """
    begin, end = (None, None) if chunk_range is None else ([int(n) for n in chunk_range[0]], [int(n) for n in chunk_range[1]])
    _reader.write_pyramid_level(os.fspath(root), int(level), [int(f) for f in factors], reduction, int(threads), begin, end)


def write_pyramid(root: str | os.PathLike[str], levels: int, *, factors: Sequence[int] = (2, 2, 2),
                  reduction: str = "mean", threads: int = 0, overwrite: bool = False) -> None:
    """Create and write ``root/1 .. root/levels`` from the local level 0 at ``root/0``.
    See :func:`create_pyramid_levels` and :func:`write_pyramid_level`.
    """
    _reader.write_pyramid(os.fspath(root), int(levels), [int(f) for f in factors], reduction, int(threads), bool(overwrite))


def ome_multiscales(*, name: str = "image", levels: int = 5, factors: Sequence[int] = (2, 2, 2),
                    scale: Sequence[float] = (1.0, 1.0, 1.0), unit: str = "", reduction: str = "mean",
                    axes: Sequence[str] = ("z", "y", "x")) -> dict[str, Any]:
    """One OME-NGFF 0.4 ``multiscales`` entry for a pyramid with ``levels`` levels above level 0."""
    return _reader.ome_multiscales(name, int(levels), [int(f) for f in factors], [float(s) for s in scale], unit, reduction,
                                   [str(a) for a in axes])


__all__ = ["ObjectStore", "ZarrWriter", "ZarrWriteGroup", "create_group", "create_pyramid_levels", "ome_multiscales",
           "write_pyramid", "write_pyramid_level"]
