"""Fast reads of zarr v2 arrays, local or remote, with optional disk and memory chunk caches.

All reading happens in the native extension (``vc_reader._native``, C++): local
chunk files through persistent memory maps (or ``pread``), HTTP fetches with
libcurl, raw / blosc / vc-delta3d decoding, disk-cache reads and appends in
each cache's own format, an in-memory LRU, and assembly into the outputs, on
native threads without the GIL. Python finds the caches, checks them against
the remote ``.zarray`` and holds VC3D's lease.

Values are exactly the stored values; missing chunks and out-of-bounds regions
read as ``fill_value``.
"""

from __future__ import annotations

import itertools
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .. import _native
from .disk_cache import CacheLayer, open_layers
from .zarr_meta import ArrayMeta

MemoryCache = _native.MemoryCache
READ_MODES = ("mmap", "slab")


def http_url(url: str) -> str:
    """https URL for an ``s3://`` URL (public buckets only), otherwise unchanged."""
    if not url.startswith("s3://"):
        return url
    from ..artifacts import to_https

    https = to_https(url)
    if https is None:
        bucket, _, key = url[len("s3://"):].partition("/")
        https = f"https://{bucket}.s3.amazonaws.com/{key}"
    return https


def _is_remote(location: str) -> bool:
    return location.startswith(("http://", "https://", "s3://"))


def _read_zarray(location: str) -> ArrayMeta:
    if _is_remote(location):
        data = _native.http_get(http_url(location) + "/.zarray")
        if data is None:
            raise FileNotFoundError(f"no zarr v2 array at {location} (.zarray not found)")
        return ArrayMeta.from_bytes(data)
    return ArrayMeta.from_bytes(Path(location, ".zarray").read_bytes())


class ZarrReader:
    """A zarr v2 array opened for reading. Use :meth:`open`.

    Indexing is ``[z, y, x]`` like the stored array.
    """

    def __init__(self, spec: dict[str, Any], cache: MemoryCache | None):
        self.memory_cache = cache
        self.location: str = spec["location"]
        self.meta = ArrayMeta.from_zarray(spec["meta"]) if spec["meta"] else _read_zarray(self.location)
        self._spec = dict(spec, meta=dict(self.meta.raw))
        codec = self.meta.codec_id  # raises for arrays the native reader cannot decode
        self.cache_layers: list[CacheLayer] = []
        self.write_layer: CacheLayer | None = None
        layers: list[tuple] = []
        write_index, remote_url = -1, ""
        if _is_remote(self.location):
            remote_url = http_url(self.location)
            if spec["cache_dirs"]:
                group = http_url(spec["group"]) if spec["group"] else None
                self.cache_layers, self.write_layer = open_layers(
                    spec["cache_dirs"], remote_url, self.meta, group_url=group, level_key=spec["level_key"],
                    writable=spec["cache_writes"], allow_lossy=spec["allow_lossy_cache"])
                layers = [layer.native_spec() for layer in self.cache_layers]
                if self.write_layer is not None:
                    write_index = self.cache_layers.index(self.write_layer)
            identity = remote_url
        else:
            root = os.path.abspath(self.location)
            layers = [("array", root, codec, "", False)]
            identity = "file://" + root
        threads = spec["threads"] or (32 if remote_url else min(16, os.cpu_count() or 4))
        try:
            self._array = _native.Array(
                list(self.meta.shape), list(self.meta.chunks), self.meta.dtype.itemsize,
                np.asarray(self.meta.fill_value, dtype=self.meta.dtype).tobytes(), self.meta.separator, identity,
                layers, write_layer=write_index, remote_url=remote_url, remote_codec=codec if remote_url else "",
                read_mode=spec["read_mode"], threads=threads, connections=spec["connections"], cache=cache)
        except BaseException:
            for layer in self.cache_layers:
                layer.close()
            raise

    # -- construction -----------------------------------------------------------

    @classmethod
    def open(
        cls,
        location: str | os.PathLike[str],
        *,
        level: int | str | None = None,
        cache_dir: str | os.PathLike[str] | Sequence[str | os.PathLike[str]] | None = None,
        cache_writes: bool = True,
        allow_lossy_cache: bool = False,
        memory_cache: int | MemoryCache | None = None,
        threads: int | None = None,
        connections: int = 16,
        read_mode: str = "mmap",
    ) -> ZarrReader:
        """Open the zarr v2 array at ``location`` (a directory, ``https://`` or public ``s3://`` URL).

        ``level`` appends a sub-path (an OME-Zarr pyramid level).

        ``cache_dir`` is one directory or a list of them, searched in order for
        an existing cache of a remote array in any supported format: vc_reader
        and vesuvius decoded caches, VC3D mirror and Delta3D caches (see
        :mod:`vc_reader.reader.disk_cache`). The entry ``"vc3d"`` stands for
        VC3D's configured remote cache root. Misses are fetched and appended to
        the first cache that holds the array, in its own format, or to a new
        decoded cache under the first directory. ``cache_writes=False`` only
        reads. A cache whose chunks are quantized (lossy relative to the source)
        is skipped unless ``allow_lossy_cache``.

        ``memory_cache`` is a byte budget or a :class:`MemoryCache` shared
        between readers. ``threads`` (default 16 local, 32 remote) read and
        decode; at most ``connections`` HTTP requests are in flight.
        ``read_mode`` is how uncompressed chunk files are read: ``"mmap"``
        (persistent memory maps, the default) or ``"slab"`` (``pread``).
        """
        if read_mode not in READ_MODES:
            raise ValueError(f"read_mode must be one of {READ_MODES}")
        loc = os.fspath(location).rstrip("/")
        if level is not None:
            group, level_key = loc, str(level)
            loc = f"{loc}/{level}"
        else:
            # An OME-Zarr level is a digits-only path under its group.
            head, _, tail = loc.rpartition("/")
            group, level_key = (head, tail) if tail.isdigit() and head else (None, None)
        if cache_dir is None:
            dirs = None
        elif isinstance(cache_dir, (str, os.PathLike)):
            dirs = [os.fspath(cache_dir)]
        else:
            dirs = [os.fspath(d) for d in cache_dir]
        if memory_cache is None or memory_cache == 0:
            cache = None
        elif isinstance(memory_cache, MemoryCache):
            cache = memory_cache
        elif isinstance(memory_cache, int):
            cache = MemoryCache(memory_cache)
        else:
            raise TypeError("memory_cache must be a byte count or a vc_reader.reader.MemoryCache")
        spec = dict(location=loc, group=group, level_key=level_key, cache_dirs=dirs, cache_writes=cache_writes,
                    allow_lossy_cache=allow_lossy_cache, threads=threads, connections=connections,
                    read_mode=read_mode, meta=None)
        return cls(spec, cache)

    def __getstate__(self):
        # The new process reopens from the recorded metadata (no network round trip),
        # with an empty memory cache of the same size.
        budget = self.memory_cache.max_bytes if self.memory_cache is not None else 0
        return {"spec": self._spec, "memory_cache_bytes": budget}

    def __setstate__(self, state):
        budget = state["memory_cache_bytes"]
        self.__init__(state["spec"], MemoryCache(budget) if budget else None)

    def __repr__(self) -> str:
        return f"ZarrReader({self.location!r}, shape={self.shape}, chunks={self.chunks}, dtype={self.dtype})"

    def close(self) -> None:
        """Stop the reader's threads and release its cache leases."""
        self._array = None
        for layer in self.cache_layers:
            layer.close()

    def __enter__(self) -> ZarrReader:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- array properties --------------------------------------------------------

    @property
    def shape(self) -> tuple[int, ...]:
        return self.meta.shape

    @property
    def chunks(self) -> tuple[int, ...]:
        return self.meta.chunks

    @property
    def dtype(self) -> np.dtype:
        return self.meta.dtype

    @property
    def fill_value(self) -> Any:
        return self.meta.fill_value

    @property
    def ndim(self) -> int:
        return self.meta.ndim

    def stats(self) -> dict[str, Any]:
        """Counters: memory/disk hits, remote fetches and bytes, cache write errors."""
        return self._array.stats()

    def chunk_indices(self, start: Sequence[int], size: Sequence[int]) -> list[tuple[int, ...]]:
        """Indices of the chunks that ``[start, start+size)`` overlaps (clipped to the array)."""
        lo = [max(int(s), 0) for s in start]
        hi = [min(int(s) + int(n), dim) for s, n, dim in zip(start, size, self.shape)]
        if any(h <= l for l, h in zip(lo, hi)):
            return []
        return list(itertools.product(*(range(l // c, (h - 1) // c + 1) for l, h, c in zip(lo, hi, self.chunks))))

    # -- reads ---------------------------------------------------------------------

    def read(self, start: Sequence[int], size: Sequence[int], out: np.ndarray | None = None) -> np.ndarray:
        """Block ``[start, start+size)``; parts outside the array read as ``fill_value``."""
        return self.read_many([(start, size)], None if out is None else [out])[0]

    def read_many(
        self,
        regions: Iterable[tuple[Sequence[int], Sequence[int]]],
        out: Sequence[np.ndarray | None] | None = None,
    ) -> list[np.ndarray]:
        """Several ``(start, size)`` blocks in one call; their chunk reads run in parallel.

        A chunk several regions need is read once. ``out`` optionally gives one
        destination array (or None) per region; reusing destinations between
        calls avoids first-touch page faults on new memory.
        """
        if self._array is None:
            raise ValueError("reader is closed")
        nd = self.ndim
        regions = [(tuple(int(v) for v in s), tuple(int(v) for v in n)) for s, n in regions]
        if out is not None and len(out) != len(regions):
            raise ValueError(f"out has {len(out)} arrays for {len(regions)} regions")
        results, targets = [], []
        for i, (start, size) in enumerate(regions):
            if len(start) != nd or len(size) != nd or any(n < 0 for n in size):
                raise ValueError(f"start/size must have {nd} non-negative entries")
            given = None if out is None else out[i]
            if given is None:
                given = np.empty(size, dtype=self.dtype)
            elif given.shape != size or given.dtype != self.dtype:
                raise ValueError(f"out must have shape {size} and dtype {self.dtype}")
            results.append(given)
            targets.append(given if given.flags.c_contiguous else np.empty(size, dtype=self.dtype))
        self._array.read_many([list(s) for s, _ in regions], [list(n) for _, n in regions],
                              [t.reshape(-1).view(np.uint8) for t in targets], self.dtype.itemsize)
        for res, tgt in zip(results, targets):
            if res is not tgt:
                res[...] = tgt
        return results

    def __getitem__(self, key: Any) -> np.ndarray:
        start, size, squeeze = _parse_index(key, self.shape)
        out = self.read(start, size)
        return out[tuple(0 if s else slice(None) for s in squeeze)] if any(squeeze) else out


def _parse_index(key: Any, shape: tuple[int, ...]) -> tuple[list[int], list[int], list[bool]]:
    """numpy basic indexing (ints, unit-step slices, Ellipsis) -> start, size, axes to drop."""
    if not isinstance(key, tuple):
        key = (key,)
    if any(k is Ellipsis for k in key):
        i = next(i for i, k in enumerate(key) if k is Ellipsis)
        key = key[:i] + (slice(None),) * (len(shape) - len(key) + 1) + key[i + 1:]
    if len(key) > len(shape):
        raise IndexError(f"too many indices for array of dimension {len(shape)}")
    key = key + (slice(None),) * (len(shape) - len(key))
    start, size, squeeze = [], [], []
    for k, dim in zip(key, shape):
        if isinstance(k, slice):
            a, b, step = k.indices(dim)
            if step != 1:
                raise IndexError("only unit-step slices are supported")
            start.append(a)
            size.append(max(b - a, 0))
            squeeze.append(False)
        else:
            i = int(k)
            if i < 0:
                i += dim
            if not 0 <= i < dim:
                raise IndexError(f"index {k} out of bounds for axis of size {dim}")
            start.append(i)
            size.append(1)
            squeeze.append(True)
    return start, size, squeeze


def open_zarr(location: str | os.PathLike[str], **kwargs: Any) -> ZarrReader:
    """Shorthand for :meth:`ZarrReader.open`."""
    return ZarrReader.open(location, **kwargs)


__all__: Iterable[str] = ["MemoryCache", "ZarrReader", "open_zarr", "http_url"]
