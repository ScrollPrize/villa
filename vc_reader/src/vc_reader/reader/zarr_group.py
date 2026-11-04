"""Zarr v2/v3 groups: OME-Zarr pyramids and other groups of arrays, local or remote.

A group opens its arrays as :class:`~vc_reader.reader.ZarrReader` objects with
the group's settings (credentials, disk caches, offline mode, threads) and one
shared memory cache::

    g = ZarrGroup.open("s3://bucket/volume.zarr", cache_dir="/data/cache", memory_cache=4 << 30)
    g.levels                    # ("0", "1", ...): OME-Zarr multiscales datasets, else arrays "0", "1", ...
    r = g.level(2)              # ZarrReader of the third level
    g["labels"], "mask" in g    # children by name: a ZarrReader or a ZarrGroup
    g.keys()                    # child names (see keys() for remote groups)

Offline (``offline=True``), the group's metadata comes from the disk caches. A
group opened online with a writable ``cache_dir`` stores its ``.zgroup`` and
``.zattrs`` next to its arrays' decoded caches so that it can be opened offline later.
"""

from __future__ import annotations

import os
from typing import Any, Iterator, Sequence

from .. import _reader
from .auth import is_remote
from .zarr_reader import MemoryCache, ZarrReader, _make_settings, _open_options
from .store_metadata import StoreMetadata


class ZarrGroup:
    """A Zarr v2/v3 group opened for reading (``vcr::zarr::Group``). Use :meth:`open`."""

    def __init__(self, native, settings: dict[str, Any], cache: MemoryCache | None):
        self._native = native
        self._pid = os.getpid()
        self._settings = settings
        self.memory_cache = cache
        self.location: str = settings["location"]
        self.access = settings["access"]
        self.attrs: dict[str, Any] = native.attributes
        self.version: int = native.version
        self._metadata: StoreMetadata | None = None
        self._readers: dict[str, ZarrReader] = {}
        self._groups: dict[str, ZarrGroup] = {}

    @classmethod
    def open(
        cls,
        location: str | os.PathLike[str],
        *,
        auth: Any = "auto",
        cache_dir: str | os.PathLike[str] | Sequence[str | os.PathLike[str]] | None = None,
        cache_writes: bool = True,
        cache_max_bytes: int | None = None,
        compress_cache: bool = False,
        allow_lossy_cache: bool = False,
        memory_cache: int | MemoryCache | None = None,
        threads: int | None = None,
        connections: int = 16,
        read_mode: str = "mmap",
        offline: bool = False,
        refresh: bool = False,
    ) -> ZarrGroup:
        """Open the Zarr v2/v3 group at ``location`` (a directory, ``http(s)://`` or ``s3://`` URL).

        The arguments are those of :meth:`ZarrReader.open`, applied to every
        array opened through the group. A ``memory_cache`` byte budget becomes
        one :class:`MemoryCache` that the group's arrays share. ``refresh`` replaces
        the metadata kept with the volume (see :meth:`ZarrReader.open`).
        """
        settings, cache = _make_settings(location, auth=auth, cache_dir=cache_dir, cache_writes=cache_writes,
                                         cache_max_bytes=cache_max_bytes, compress_cache=compress_cache,
                                         allow_lossy_cache=allow_lossy_cache, memory_cache=memory_cache,
                                         threads=threads, connections=connections, read_mode=read_mode,
                                         offline=offline)
        group = cls(_reader.Group.open(settings["location"], _open_options(settings, cache)), settings, cache)
        group._metadata = StoreMetadata.from_json(group._group.metadata_json(refresh))
        return group

    @property
    def _group(self):
        """The native group, rebuilt from its state in a forked process (nothing is requested again)."""
        if self._pid != os.getpid():
            state = self._native.state(True)
            self._native = _reader.Group.from_state(state, _open_options(self._settings, self.memory_cache))
            self._pid = os.getpid()
        return self._native

    @property
    def metadata(self) -> StoreMetadata:
        """What is known about the volume: its store's ``meta.json`` / ``metadata.json``
        and the Open Data catalog, read when it was opened."""
        if self._metadata is None:
            self._metadata = StoreMetadata.from_json(self._group.metadata_json())
        return self._metadata

    @property
    def voxel_size(self) -> float | None:
        """Micrometers per voxel of level 0; ``None`` when the store does not say."""
        return self.metadata.voxel_size_um

    def __getstate__(self):
        return {"state": self._group.state(), "settings": self._settings, "memory_cache": self.memory_cache,
                "metadata": self._metadata}

    def __setstate__(self, state):
        native = _reader.Group.from_state(state["state"], _open_options(state["settings"], state["memory_cache"]))
        self.__init__(native, state["settings"], state["memory_cache"])
        self._metadata = state["metadata"]

    def __repr__(self) -> str:
        return f"ZarrGroup({self.location!r})"

    @property
    def offline(self) -> bool:
        return bool(self._settings["offline"]) and is_remote(self.location)

    # -- OME-Zarr levels -------------------------------------------------------------

    @property
    def multiscales(self) -> list[dict[str, Any]]:
        """The OME-NGFF ``multiscales`` metadata (empty if the group has none)."""
        return self._native.multiscales

    @property
    def levels(self) -> tuple[str, ...]:
        """Paths of the pyramid levels, finest first.

        The first ``multiscales`` entry's datasets; without OME-Zarr metadata, the arrays
        named with numbers: all of them, gaps included, in a local or consolidated group,
        otherwise up to the first one that is absent.
        """
        return tuple(self._group.levels())

    def placement(self, path: str) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Where the level at ``path`` sits: ``level0 = scale * index + offset``, ``(scale, offset)`` in ZYX."""
        scale, offset = self._group.placement(path)
        return tuple(scale), tuple(offset)

    def level(self, index: int) -> ZarrReader:
        """The array of pyramid level ``index`` (a position in :attr:`levels`; negative counts from the coarsest)."""
        levels = self.levels
        try:
            path = levels[index]
        except IndexError:
            raise IndexError(f"{self.location} has {len(levels)} levels, not {index}") from None
        return self.array(path)

    # -- children ------------------------------------------------------------------------

    def _node(self, key: str) -> str | None:
        """"array" or "group" for a child path, None if there is none (memoized)."""
        try:
            return self._group.node(str(key))
        except ValueError as error:
            raise KeyError(str(error)) from None

    def __contains__(self, key: object) -> bool:
        try:
            return self._node(str(key)) is not None
        except KeyError:
            return False

    def __getitem__(self, key: str) -> ZarrReader | ZarrGroup:
        node = self._node(key)
        if node is None:
            raise KeyError(f"{key!r} is not in {self.location}")
        return self.array(key) if node == "array" else self.group(key)

    def get(self, key: str, default: Any = None) -> ZarrReader | ZarrGroup | Any:
        return self[key] if key in self else default

    def array(self, key: str) -> ZarrReader:
        """The child array at path ``key`` (opened once, then reused)."""
        key = str(key).strip("/")
        reader = self._readers.get(key)
        if reader is not None and reader._array is not None:
            return reader
        if self._node(key) != "array":
            raise KeyError(f"no array {key!r} in {self.location}")
        opened = self._group.array(key)
        settings = dict(self._settings, location=f"{self.location}/{key}", group=self.location, level_key=key,
                        store_metadata=self.metadata)
        reader = ZarrReader(settings, self.memory_cache, opened)
        self._readers[key] = reader
        return reader

    def group(self, key: str) -> ZarrGroup:
        """The child group at path ``key``, with this group's settings and memory cache."""
        key = str(key).strip("/")
        if key in self._groups:
            return self._groups[key]
        if self._node(key) != "group":
            raise KeyError(f"no group {key!r} in {self.location}")
        child = ZarrGroup(self._group.group(key), dict(self._settings, location=f"{self.location}/{key}"),
                          self.memory_cache)
        self._groups[key] = child
        return child

    def keys(self) -> list[str]:
        """Names of the arrays and groups directly in this group, sorted.

        Local groups are listed from the directory, remote ones from consolidated
        ``.zmetadata`` if present, else by listing the S3 prefix (``s3://`` or a
        known public bucket). Plain HTTP servers cannot be listed: open children
        by name. Offline, the children the disk caches hold are listed.
        """
        return self._group.keys()

    def array_keys(self) -> list[str]:
        """Those of :meth:`keys` that are arrays."""
        return self._group.array_keys()

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    # -- lifetime ------------------------------------------------------------------------

    def close(self) -> None:
        """Close every array opened through this group (and its subgroups)."""
        for reader in self._readers.values():
            reader.close()
        for g in self._groups.values():
            g.close()
        self._readers.clear()
        self._groups.clear()

    def __enter__(self) -> ZarrGroup:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def open_node(location: str | os.PathLike[str], **kwargs: Any) -> ZarrReader | ZarrGroup:
    """The array or group at ``location``, whichever it is; the arguments of :meth:`ZarrGroup.open`."""
    settings, _ = _make_settings(location, auth=kwargs.get("auth", "auto"), cache_dir=kwargs.get("cache_dir"),
                                 offline=kwargs.get("offline", False))
    if _reader.is_array_node(settings["location"], _open_options(settings, None)):
        return ZarrReader.open(location, **kwargs)
    return ZarrGroup.open(location, **kwargs)


def open_group(location: str | os.PathLike[str], **kwargs: Any) -> ZarrGroup:
    """Shorthand for :meth:`ZarrGroup.open`."""
    return ZarrGroup.open(location, **kwargs)


__all__ = ["ZarrGroup", "open_group"]
