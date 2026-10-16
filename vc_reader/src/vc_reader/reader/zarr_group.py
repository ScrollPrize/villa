"""Zarr v2 groups: OME-Zarr pyramids and other groups of arrays, local or remote.

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

import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterator, Sequence

from ..errors import CacheMissError
from .auth import Access, is_remote, split_s3
from .disk_cache import VC3D, atomic_write, cached_node_dirs, cached_source_zarray, decoded_cache_path
from .zarr_reader import MemoryCache, ZarrReader, _make_spec, cache_url, read_node_file

#: Settings of ZarrGroup.open that its arrays are opened with.
_READER_OPTIONS = ("cache_dir", "cache_writes", "compress_cache", "allow_lossy_cache", "threads", "connections",
                   "read_mode", "offline")


class ZarrGroup:
    """A zarr v2 group opened for reading. Use :meth:`open`."""

    def __init__(self, location: str, options: dict[str, Any], cache: MemoryCache | None, access: Access,
                 attrs: dict[str, Any]):
        self.location = location
        self.access = access
        self.memory_cache = cache
        self.attrs: dict[str, Any] = attrs
        self._options = options
        self._nodes: dict[str, tuple[str, dict | None] | None] = {}  # child -> ("array", .zarray) / ("group", None)
        self._readers: dict[str, ZarrReader] = {}
        self._groups: dict[str, ZarrGroup] = {}
        self._consolidated: dict[str, Any] | None | bool = False  # False: not looked up yet

    @classmethod
    def open(
        cls,
        location: str | os.PathLike[str],
        *,
        auth: Any = "auto",
        cache_dir: str | os.PathLike[str] | Sequence[str | os.PathLike[str]] | None = None,
        cache_writes: bool = True,
        compress_cache: bool = False,
        allow_lossy_cache: bool = False,
        memory_cache: int | MemoryCache | None = None,
        threads: int | None = None,
        connections: int = 16,
        read_mode: str = "mmap",
        offline: bool = False,
    ) -> ZarrGroup:
        """Open the zarr v2 group at ``location`` (a directory, ``http(s)://`` or ``s3://`` URL).

        The arguments are those of :meth:`ZarrReader.open`, applied to every
        array opened through the group. A ``memory_cache`` byte budget becomes
        one :class:`MemoryCache` that the group's arrays share.
        """
        loc = os.fspath(location).rstrip("/")
        options = dict(cache_dir=cache_dir, cache_writes=cache_writes, compress_cache=compress_cache,
                       allow_lossy_cache=allow_lossy_cache, threads=threads, connections=connections,
                       read_mode=read_mode, offline=offline)
        # Validates the options and resolves the credentials once for the whole group.
        spec, cache = _make_spec(loc, auth=auth, memory_cache=memory_cache, **options)
        access: Access = spec["access"]
        options["cache_dir"] = spec["cache_dirs"]
        remote = is_remote(loc)
        if remote and offline:
            dirs = cached_node_dirs(spec["cache_dirs"], cache_url(loc, access))
            if not dirs:
                raise CacheMissError(f"{loc} is not in the disk caches {spec['cache_dirs']} (offline)")
            attrs = next((a for a in (_read_json(d / ".zattrs") for d in dirs) if a is not None), {})
        else:
            zgroup = read_node_file(loc, ".zgroup", access)
            zattrs = read_node_file(loc, ".zattrs", access)
            if zgroup is None and zattrs is None:
                if read_node_file(loc, ".zarray", access) is not None:
                    raise ValueError(f"{loc} is an array, not a group: open it with ZarrReader.open")
                raise FileNotFoundError(f"no zarr v2 group at {loc} (.zgroup not found)")
            attrs = json.loads(zattrs) if zattrs else {}
            if remote and cache_writes and spec["cache_dirs"]:
                _cache_group_metadata(spec["cache_dirs"], cache_url(loc, access), zgroup, zattrs)
        return cls(loc, options, cache, access, attrs)

    def __getstate__(self):
        budget = self.memory_cache.max_bytes if self.memory_cache is not None else 0
        return {"location": self.location, "options": self._options, "memory_cache_bytes": budget,
                "access": self.access, "attrs": self.attrs, "nodes": self._nodes}

    def __setstate__(self, state):
        budget = state["memory_cache_bytes"]
        self.__init__(state["location"], state["options"], MemoryCache(budget) if budget else None,
                      state["access"], state["attrs"])
        self._nodes = state["nodes"]

    def __repr__(self) -> str:
        return f"ZarrGroup({self.location!r})"

    @property
    def offline(self) -> bool:
        return bool(self._options["offline"]) and is_remote(self.location)

    # -- OME-Zarr levels -------------------------------------------------------------

    @property
    def multiscales(self) -> list[dict[str, Any]]:
        """The OME-NGFF ``multiscales`` metadata (empty if the group has none)."""
        ms = self.attrs.get("multiscales") or (self.attrs.get("ome") or {}).get("multiscales") or []
        return list(ms) if isinstance(ms, list) else []

    @property
    def levels(self) -> tuple[str, ...]:
        """Paths of the pyramid levels, finest first.

        The first ``multiscales`` entry's datasets; without OME-Zarr metadata,
        the arrays named ``"0"``, ``"1"``, ... up to the first one that is absent.
        """
        ms = self.multiscales
        if ms:
            return tuple(str(d["path"]) for d in ms[0].get("datasets", []))
        out: list[str] = []
        while (node := self._node(str(len(out)))) is not None and node[0] == "array":
            out.append(str(len(out)))
        return tuple(out)

    def level(self, index: int) -> ZarrReader:
        """The array of pyramid level ``index`` (a position in :attr:`levels`; negative counts from the coarsest)."""
        levels = self.levels
        try:
            path = levels[index]
        except IndexError:
            raise IndexError(f"{self.location} has {len(levels)} levels, not {index}") from None
        return self.array(path)

    # -- children ------------------------------------------------------------------------

    def _child(self, key: str) -> str:
        key = str(key).strip("/")
        if not key or any(part in ("", ".", "..") for part in key.split("/")):
            raise KeyError(f"invalid child path {key!r}")
        return key

    def _node(self, key: str) -> tuple[str, dict | None] | None:
        """("array", .zarray) or ("group", None) for a child path, None if there is none (memoized)."""
        key = self._child(key)
        if key in self._nodes:
            return self._nodes[key]
        loc = f"{self.location}/{key}"
        consolidated = self._consolidated_metadata()
        node: tuple[str, dict | None] | None = None
        if consolidated is not None:
            if f"{key}/.zarray" in consolidated:
                node = ("array", dict(consolidated[f"{key}/.zarray"]))
            elif f"{key}/.zgroup" in consolidated or f"{key}/.zattrs" in consolidated:
                node = ("group", None)
        elif self.offline:
            dirs = self._options["cache_dir"]
            zarray = cached_source_zarray(dirs, cache_url(loc, self.access),
                                          group_url=cache_url(self.location, self.access), level_key=key)
            if zarray is not None:
                node = ("array", zarray)
            elif any((d / ".zgroup").exists() or (d / ".zattrs").exists()
                     for d in cached_node_dirs(dirs, cache_url(loc, self.access))):
                node = ("group", None)
        else:
            zarray = read_node_file(loc, ".zarray", self.access)
            if zarray is not None:
                node = ("array", json.loads(zarray))
            elif read_node_file(loc, ".zgroup", self.access) is not None:
                node = ("group", None)
        self._nodes[key] = node
        return node

    def _consolidated_metadata(self) -> dict[str, Any] | None:
        """The group's consolidated ``.zmetadata`` (online only), or None."""
        if self._consolidated is False:
            data = None if self.offline else read_node_file(self.location, ".zmetadata", self.access)
            self._consolidated = json.loads(data).get("metadata") if data else None
        return self._consolidated  # type: ignore[return-value]

    def __contains__(self, key: object) -> bool:
        try:
            return self._node(str(key)) is not None
        except KeyError:
            return False

    def __getitem__(self, key: str) -> ZarrReader | ZarrGroup:
        node = self._node(key)
        if node is None:
            raise KeyError(f"{key!r} is not in {self.location}")
        return self.array(key) if node[0] == "array" else self.group(key)

    def get(self, key: str, default: Any = None) -> ZarrReader | ZarrGroup | Any:
        return self[key] if key in self else default

    def array(self, key: str) -> ZarrReader:
        """The child array at path ``key`` (opened once, then reused)."""
        key = self._child(key)
        reader = self._readers.get(key)
        if reader is not None and reader._array is not None:
            return reader
        node = self._node(key)
        if node is None or node[0] != "array":
            raise KeyError(f"no array {key!r} in {self.location}")
        spec, _ = _make_spec(f"{self.location}/{key}", group=self.location, auth=self.access, meta=node[1],
                             **self._options)
        reader = ZarrReader(spec, self.memory_cache)
        self._readers[key] = reader
        return reader

    def group(self, key: str) -> ZarrGroup:
        """The child group at path ``key``, with this group's settings and memory cache."""
        key = self._child(key)
        if key in self._groups:
            return self._groups[key]
        node = self._node(key)
        if node is None or node[0] != "group":
            raise KeyError(f"no group {key!r} in {self.location}")
        loc = f"{self.location}/{key}"
        if self.offline:
            dirs = cached_node_dirs(self._options["cache_dir"], cache_url(loc, self.access))
            attrs = next((a for a in (_read_json(d / ".zattrs") for d in dirs) if a is not None), {})
        else:
            zattrs = read_node_file(loc, ".zattrs", self.access)
            attrs = json.loads(zattrs) if zattrs else {}
        g = ZarrGroup(loc, self._options, self.memory_cache, self.access, attrs)
        self._groups[key] = g
        return g

    def keys(self) -> list[str]:
        """Names of the arrays and groups directly in this group, sorted.

        Local groups are listed from the directory, remote ones from consolidated
        ``.zmetadata`` if present, else by listing the S3 prefix (``s3://`` or a
        known public bucket). Plain HTTP servers cannot be listed: open children
        by name. Offline, the children the disk caches hold are listed.
        """
        consolidated = self._consolidated_metadata()
        if consolidated is not None:
            names = {k.split("/", 1)[0] for k in consolidated
                     if k.count("/") == 1 and k.endswith((".zarray", ".zgroup", ".zattrs"))}
        elif not is_remote(self.location):
            names = {e.name for e in os.scandir(self.location) if e.is_dir() and _is_node_dir(Path(e.path))}
        elif self.offline:
            names = set()
            for d in cached_node_dirs(self._options["cache_dir"], cache_url(self.location, self.access)):
                names |= {e.name for e in os.scandir(d)
                          if e.is_dir() and not e.name.startswith(".") and _is_node_dir(Path(e.path))}
        else:
            names = {name for name in self._list_s3() if self._node(name) is not None}
        return sorted(names)

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def _list_s3(self) -> list[str]:
        """Immediate sub-prefixes of this group's S3 prefix (ListObjectsV2)."""
        from ..artifacts import to_s3
        from .. import _native

        s3 = self.location if self.location.startswith("s3://") else to_s3(self.location)
        if s3 is None:
            raise ValueError(f"cannot list {self.location}: plain HTTP has no listing and the group has no "
                             "consolidated .zmetadata; open its children by name")
        _, key = split_s3(s3)
        prefix = key.strip("/") + "/" if key.strip("/") else ""
        names: list[str] = []
        token = None
        while True:
            body = _native.http_get(self.access.list_url(s3, token), auth=self.access.native())
            if body is None:
                raise FileNotFoundError(f"cannot list {s3}: bucket not found")
            root = ET.fromstring(body)
            for el in root.iter():
                if el.tag.rsplit("}", 1)[-1] == "CommonPrefixes":
                    p = next((c.text for c in el if c.tag.rsplit("}", 1)[-1] == "Prefix"), None) or ""
                    name = p[len(prefix):].strip("/")
                    if name:
                        names.append(name)
            truncated = _xml_text(root, "IsTruncated") == "true"
            token = _xml_text(root, "NextContinuationToken")
            if not truncated or not token:
                return names

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


def _xml_text(root: ET.Element, name: str) -> str | None:
    for el in root:
        if el.tag.rsplit("}", 1)[-1] == name:
            return el.text
    return None


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, NotADirectoryError, ValueError):
        return None


def _is_node_dir(path: Path) -> bool:
    return any((path / name).is_file() for name in (".zarray", ".zgroup", ".zattrs"))


def _cache_group_metadata(cache_dirs: Sequence[str], url: str, zgroup: bytes | None, zattrs: bytes | None) -> None:
    """Store a remote group's metadata in the first plain cache directory, for opening it offline."""
    root = next((d for d in cache_dirs if os.fspath(d) != VC3D), None)
    if root is None:
        return
    target = Path(decoded_cache_path(Path(root).expanduser(), url))
    try:
        for name, data in ((".zgroup", zgroup), (".zattrs", zattrs)):
            if data is not None and _read_bytes(target / name) != data:
                atomic_write(target / name, data)
    except OSError:
        pass  # a read-only cache directory: the group still opens online


def _read_bytes(path: Path) -> bytes | None:
    try:
        return path.read_bytes()
    except (FileNotFoundError, NotADirectoryError):
        return None


def open_group(location: str | os.PathLike[str], **kwargs: Any) -> ZarrGroup:
    """Shorthand for :meth:`ZarrGroup.open`."""
    return ZarrGroup.open(location, **kwargs)


__all__ = ["ZarrGroup", "open_group"]
