"""Fast reads of zarr v2 arrays, local or remote, with optional disk and memory chunk caches.

All reading happens in the native extension (``vc_reader._native``, C++): local
chunk files through persistent memory maps (or ``pread``), HTTP fetches with
libcurl, raw / blosc / vc-delta3d decoding, disk-cache reads and appends in
each cache's own format, an in-memory LRU, and assembly into the outputs, on
native threads without the GIL. Python finds the caches, checks them against
the remote ``.zarray`` and holds VC3D's lease.

Values are exactly the stored values; missing chunks and out-of-bounds regions
read as ``fill_value``.

Remote arrays are read anonymously, with HTTP basic auth, or with signed S3
requests (see :mod:`vc_reader.reader.auth`). ``offline=True`` reads only the
disk caches: metadata comes from the caches and a chunk in none of them raises
:class:`~vc_reader.errors.CacheMissError` instead of being fetched.

Processes and fork: a reader's work runs on background threads (the read pool
and the disk-cache writers), and a forked process (e.g. a PyTorch DataLoader
worker on Linux) inherits the reader but not its threads. A reader therefore
notices when it is used in a different process than it was opened in, and
rebuilds its native engine there from its saved settings (as unpickling does);
the inherited engine is left untouched, and the parent still finishes its own
queued cache writes. A shared MemoryCache starts empty in each worker and
retains one budget shared by all the readers that reference it.
Workers started with ``spawn``/``forkserver`` receive a pickled reader, which is
reopened the same way.
"""

from __future__ import annotations

import itertools
import json
import operator
import os
import threading
import weakref
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .. import _native
from ..errors import CacheMissError
from .auth import Access, http_url, is_remote, resolve_access
from .disk_cache import CacheLayer, cached_source_zarray, open_layers, cached_node_dirs, atomic_write, disk_budget
from .memory_cache import MemoryCache
from .zarr_meta import ArrayMeta

READ_MODES = ("mmap", "slab")

# Native engines inherited through fork, kept alive and never touched: their
# threads do not exist in the child, and their locks may have been held by them.
_INHERITED: list[Any] = []


def read_node_file(location: str, name: str, access: Access) -> bytes | None:
    """``<location>/<name>`` (e.g. ``.zarray``) of a local or remote node; None if it does not exist."""
    if not is_remote(location):
        try:
            return Path(location, name).read_bytes()
        except (FileNotFoundError, NotADirectoryError):
            return None
    try:
        return _native.http_get(access.url(f"{location}/{name}"), auth=access.native())
    except PermissionError as e:
        if access.kind == "anonymous":
            raise PermissionError(f"{e} (requested anonymously: pass auth= with credentials)") from None
        raise


def cache_url(location: str, access: Access) -> str:
    """The URL that names a remote location in disk caches and the memory cache.

    Signed S3 locations keep their ``s3://`` form; others the https form they
    always had, so existing caches are found.
    """
    return location if access.kind == "s3" else http_url(location)


def _read_zarray(location: str, access: Access) -> ArrayMeta:
    data = read_node_file(location, ".zarray", access)
    if data is None:
        raise FileNotFoundError(f"no zarr v2 array at {location} (.zarray not found)")
    return ArrayMeta.from_bytes(data)


class ZarrReader:
    """A zarr v2 array opened for reading. Use :meth:`open`.

    Indexing is ``[z, y, x]`` like the stored array.

    Safe across fork: used in a forked process (e.g. a DataLoader worker), the
    reader rebuilds its native engine there (see the module docstring).
    """

    def __init__(self, spec: dict[str, Any], cache: MemoryCache | None):
        self.memory_cache = cache
        self._prefetch_scheduler = self._prefetch_finalizer = None
        self._prefetch_lock = threading.Lock()
        self._prefetch_client = spec.get("prefetcher")
        self._prefetch_target = None
        self._pid = os.getpid()
        self.location: str = spec["location"]
        remote = is_remote(self.location)
        if self._prefetch_client is not None:
            from .prefetch import PrefetchClient
            if not isinstance(self._prefetch_client, PrefetchClient):
                raise TypeError("prefetcher must be a PrefetchService.client")
            if remote and (spec["offline"] or not spec["cache_dirs"] or not spec["cache_writes"]):
                raise ValueError("a prefetch service needs an online array with a writable cache_dir")
            self._prefetch_target, meta = self._prefetch_client.register(spec)
            spec = dict(spec, meta=meta)
        self.access: Access = spec["access"]
        self.offline: bool = bool(spec["offline"]) and remote
        key_url = cache_url(self.location, self.access) if remote else self.location
        group_url = cache_url(spec["group"], self.access) if remote and spec["group"] else None
        if spec["meta"]:
            self.meta = ArrayMeta.from_zarray(spec["meta"])
        elif self.offline:
            cached = cached_source_zarray(spec["cache_dirs"], key_url, group_url=group_url,
                                          level_key=spec["level_key"])
            if cached is None:
                raise CacheMissError(f"{self.location} is not in the disk caches {spec['cache_dirs']} (offline)")
            self.meta = ArrayMeta.from_zarray(cached)
        else:
            self.meta = _read_zarray(self.location, self.access)
        self._spec = dict(spec, meta=dict(self.meta.raw))
        self._attrs = spec.get("attrs")
        # Raises for arrays the native reader cannot decode; offline, chunks only come from the caches.
        codec = "" if self.offline else self.meta.codec_id
        self.cache_layers: list[CacheLayer] = []
        self.write_layer: CacheLayer | None = None
        layers: list[tuple] = []
        write_index, remote_url = -1, ""
        self._auth = _native_auth = self.access.native() if remote and not self.offline else ("", "", "", "", "")
        if remote:
            remote_url = self.access.url(self.location)
            if spec["cache_dirs"]:
                self.cache_layers, self.write_layer = open_layers(
                    spec["cache_dirs"], key_url, self.meta, group_url=group_url, level_key=spec["level_key"],
                    writable=spec["cache_writes"] and not self.offline and self._prefetch_client is None,
                    allow_lossy=spec["allow_lossy_cache"],
                    compress=spec["compress_cache"], cache_max_bytes=spec["cache_max_bytes"])
                layers = [layer.native_spec() for layer in self.cache_layers]
                if self.write_layer is not None:
                    write_index = self.cache_layers.index(self.write_layer)
            if self.offline and not self.cache_layers:
                raise CacheMissError(f"no disk cache of {self.location} under {spec['cache_dirs']} (offline)")
            identity = key_url
        else:
            root = os.path.abspath(self.location)
            layers = [("array", root, codec, "", False)]
            identity = "file://" + root
        threads = spec["threads"] or (32 if remote_url else min(16, os.cpu_count() or 4))
        try:
            budget = (disk_budget(self.write_layer.budget_root, spec["cache_max_bytes"])
                      if self.write_layer is not None and self.write_layer.budget_root else None)
            self._array = _native.Array(
                list(self.meta.shape), list(self.meta.chunks), self.meta.dtype.itemsize,
                np.asarray(self.meta.fill_value, dtype=self.meta.dtype).tobytes(), self.meta.separator, identity,
                layers, write_layer=write_index, remote_url=remote_url, remote_codec=codec if remote_url else "",
                auth=_native_auth, offline=self.offline or self._prefetch_client is not None,
                read_mode=spec["read_mode"], threads=threads,
                connections=spec["connections"], cache=cache._native if cache is not None else None,
                disk_cache=budget)
        except BaseException:
            for layer in self.cache_layers:
                layer.close()
            raise
        # Queued cache writes finish even if the reader is dropped without close().
        self._finalizer = weakref.finalize(self, _shutdown, self._array, self.cache_layers, self._pid)
        if self._prefetch_client is not None and remote:
            from functools import partial
            self._array.set_fetcher(partial(self._prefetch_client.demand, self._prefetch_target))

    # -- construction -----------------------------------------------------------

    @classmethod
    def open(
        cls,
        location: str | os.PathLike[str],
        *,
        level: int | str | None = None,
        cache_dir: str | os.PathLike[str] | Sequence[str | os.PathLike[str]] | None = None,
        cache_writes: bool = True,
        cache_max_bytes: int | None = None,
        compress_cache: bool = False,
        allow_lossy_cache: bool = False,
        memory_cache: int | MemoryCache | None = None,
        threads: int | None = None,
        connections: int = 16,
        read_mode: str = "mmap",
        auth: Any = "auto",
        offline: bool = False,
        prefetcher=None,
    ) -> ZarrReader:
        """Open the zarr v2 array at ``location`` (a directory, ``http(s)://`` or ``s3://`` URL).

        ``level`` appends a sub-path (an OME-Zarr pyramid level).

        ``auth`` says how remote requests authenticate: ``"auto"`` (anonymous,
        except AWS credentials from the environment for ``s3://`` buckets
        without a public https form), ``None``, ``"aws"``, an
        :class:`~vc_reader.reader.S3Credentials` or a ``(username, password)``
        tuple for HTTP basic auth. See :mod:`vc_reader.reader.auth`.

        ``offline=True`` never contacts the remote: the array's metadata and
        chunks come from the disk caches (``cache_dir`` is required), and a
        chunk none of them holds raises :class:`~vc_reader.errors.CacheMissError`.
        Chunks the caches mark as missing read as ``fill_value``. Local arrays
        are unaffected.

        ``cache_dir`` is one directory or a list of them, searched in order for
        an existing cache of a remote array in any supported format: vc_reader
        and vesuvius decoded caches, VC3D mirror and Delta3D caches (see
        :mod:`vc_reader.reader.disk_cache`). The entry ``"vc3d"`` stands for
        VC3D's configured remote cache root. Misses are fetched and appended to
        the first cache that holds the array, in its own format, or to a new
        decoded cache under the first directory. Cache files are written in the
        background after the read returns (see :meth:`flush`).
        ``compress_cache=True`` stores that new cache's chunks with lossless
        Delta3D (3-D uint8/uint16 arrays). ``cache_writes=False`` only reads. A
        cache whose chunks are quantized (lossy relative to the source) is
        skipped unless ``allow_lossy_cache``.

        ``cache_max_bytes`` optionally caps stored chunk payload bytes across
        decoded arrays in the first explicit cache directory. Oldest writes
        are evicted first; metadata and VC3D-managed caches are retained.
        Capped readers only append to that decoded cache. The budget is shared
        across processes via SQLite; offline/read-only opens never evict.

        ``memory_cache`` is a byte budget or a :class:`MemoryCache` shared
        between readers. ``threads`` (default 16 local, 32 remote) read and
        decode; at most ``connections`` HTTP requests are in flight.
        ``read_mode`` is how uncompressed chunk files are read: ``"mmap"``
        (persistent memory maps, the default) or ``"slab"`` (``pread``).

        The reader works on background threads. In a process forked after this
        call (e.g. a Linux DataLoader worker) it rebuilds them on first use.

        ``prefetcher=service.client`` routes array opening and chunk misses
        through a shared :class:`PrefetchService`. It requires a writable disk
        cache and cannot be combined with remote ``offline=True``. Without a
        service, :meth:`prefetch` lazily starts an in-process scheduler.
        """
        spec, cache = _make_spec(location, level=level, cache_dir=cache_dir, cache_writes=cache_writes,
                                 cache_max_bytes=cache_max_bytes,
                                 compress_cache=compress_cache, allow_lossy_cache=allow_lossy_cache,
                                 memory_cache=memory_cache, threads=threads, connections=connections,
                                 read_mode=read_mode, auth=auth, offline=offline, prefetcher=prefetcher)
        return cls(spec, cache)

    def __getstate__(self):
        return {"spec": dict(self._spec, attrs=self._attrs), "memory_cache": self.memory_cache}

    def __setstate__(self, state):
        self.__init__(state["spec"], state["memory_cache"])

    def __repr__(self) -> str:
        return f"ZarrReader({self.location!r}, shape={self.shape}, chunks={self.chunks}, dtype={self.dtype})"

    def _check_process(self) -> None:
        """After a fork, rebuild the native engine in this process (see the module docstring)."""
        if self._pid == os.getpid() or self._array is None:
            return
        self._abandon_inherited()
        self.__init__(dict(self._spec, attrs=self._attrs), self.memory_cache)

    def _abandon_inherited(self) -> None:
        self._finalizer.detach()  # the parent flushes and closes its own engine
        if self._prefetch_finalizer is not None:
            self._prefetch_finalizer.detach()
        _INHERITED.append((self._array, self.cache_layers, self._prefetch_scheduler))
        self._prefetch_scheduler = self._prefetch_finalizer = None
        self._array, self.cache_layers, self.write_layer = None, [], None

    def flush(self) -> None:
        """Wait until every chunk fetched so far is written to the disk cache."""
        self._check_process()
        if self._array is not None:
            self._array.flush()

    def close(self) -> None:
        """Finish queued cache writes, stop the reader's threads and release its cache leases."""
        if self._pid != os.getpid():
            if self._array is not None:
                self._abandon_inherited()
            return
        if self._prefetch_finalizer is not None:
            self._prefetch_finalizer()
        self._array = None
        self._finalizer()

    def __enter__(self) -> ZarrReader:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- array properties --------------------------------------------------------

    @property
    def attrs(self) -> dict[str, Any]:
        """Array attributes, loaded lazily. Editing this dict does not change the store.

        Remote attributes read online are saved alongside writable caches for
        offline use. A cache predating attribute support raises offline until
        its attributes have been fetched once.
        """
        if self._attrs is None:
            self._ready()
            if self.offline:
                for root in cached_node_dirs(self._spec["cache_dirs"], cache_url(self.location, self.access)):
                    # Decoded-cache .zattrs belongs to the cache, not the source.
                    path = root / ".vc_reader_source_attrs.json"
                    if path.is_file():
                        self._attrs = json.loads(path.read_bytes())
                        break
                else:
                    raise CacheMissError(f"array attributes for {self.location} are not cached (offline)")
            else:
                data = read_node_file(self.location, ".zattrs", self.access)
                self._attrs = json.loads(data) if data is not None else {}
                self._cache_attrs()
        return self._attrs

    def _cache_attrs(self) -> None:
        if self._attrs is not None and self.write_layer is not None:
            try:
                atomic_write(Path(self.write_layer.root) / ".vc_reader_source_attrs.json",
                             json.dumps(self._attrs).encode())
            except OSError:
                pass  # Attribute access still works with a read-only cache.

    @property
    def compressor(self):
        return self.meta.compressor

    @property
    def filters(self):
        return list(self.meta.filters) or None

    @property
    def order(self) -> str:
        return self.meta.order

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
        self._check_process()
        if self._array is None:
            raise ValueError("reader is closed")
        return self._array.stats()

    def chunk_indices(self, start: Sequence[int], size: Sequence[int]) -> list[tuple[int, ...]]:
        """Indices of the chunks that ``[start, start+size)`` overlaps (clipped to the array)."""
        lo = [max(int(s), 0) for s in start]
        hi = [min(int(s) + int(n), dim) for s, n, dim in zip(start, size, self.shape)]
        if any(h <= l for l, h in zip(lo, hi)):
            return []
        return list(itertools.product(*(range(l // c, (h - 1) // c + 1) for l, h, c in zip(lo, hi, self.chunks))))

    # -- reads ---------------------------------------------------------------------

    def _prefetch_backend(self):
        self._ready()
        if self._prefetch_client is not None:
            return (self._prefetch_client._backend, self._prefetch_target,
                    self._prefetch_client.max_plan_chunks)
        with self._prefetch_lock:
            if self._prefetch_scheduler is None:
                from .prefetch import _local
                scheduler, target = _local(self)
                self._prefetch_scheduler = scheduler
                self._prefetch_finalizer = weakref.finalize(self, scheduler.close)
                if is_remote(self.location):
                    ref = weakref.ref(scheduler)

                    def demand(index):
                        backend = ref()
                        if backend is None:
                            raise RuntimeError("prefetch scheduler is closed")
                        return backend.demand(target, index)

                    self._array.set_fetcher(demand)
        return self._prefetch_scheduler, "array", self._prefetch_scheduler.max_plan_chunks

    def prefetch(self, regions):
        """Schedule ``(start, size)`` regions without allocating output crops.

        Returns a :class:`PrefetchRequest` with ``wait(timeout)``, ``done()``,
        ``exception()`` and ``cancel()``. Normal reads automatically join/promote
        outstanding remote requests. Completion does not pin cached chunks.
        """
        import uuid
        from .prefetch import PrefetchRequest, _keys
        backend, target, limit = self._prefetch_backend()
        owner = uuid.uuid4().hex
        backend.update(owner, target, _keys(self, [regions], limit))
        return PrefetchRequest(backend, owner)

    def prefetch_window(self):
        """An owned lookahead window; ``update([regions, ...])`` replaces its plan.

        Batches are ordered nearest first. Closing the window withdraws queued
        work that no other caller needs. Create windows inside loader workers.
        """
        from .prefetch import PrefetchWindow
        backend, target, limit = self._prefetch_backend()
        return PrefetchWindow(self, backend, target, limit)

    def prefetch_stats(self):
        """Scheduler counters, or an empty dict if prefetch has not been enabled."""
        self._ready()
        if self._prefetch_client is not None:
            return self._prefetch_client._backend.stats()
        return self._prefetch_scheduler.stats() if self._prefetch_scheduler else {}

    def read(
        self, start: Sequence[int], size: Sequence[int], out: np.ndarray | None = None, *,
        dtype: Any = None,
    ) -> np.ndarray:
        """Read a block, optionally cast it; see :meth:`read_many`."""
        return self.read_many([(start, size)], None if out is None else [out],
                              dtype=dtype)[0]

    def read_many(
        self,
        regions: Iterable[tuple[Sequence[int], Sequence[int]]],
        out: Sequence[np.ndarray | None] | None = None,
        *, dtype: Any = None,
    ) -> list[np.ndarray]:
        """Several ``(start, size)`` blocks in one call; their chunk reads run in parallel.

        A chunk several regions need is read once. ``out`` optionally gives one
        destination array (or None) per region. Arrays the reader allocates come
        from a pool of recycled buffers (see :func:`set_output_pool_bytes`), so
        a loop that drops its previous outputs gets already-touched memory, as
        if it passed them back as ``out``.

        Defaults return stored values/dtype, with ``fill_value`` outside the
        array. Request ``dtype=np.float32`` or ``np.float64`` to cast directly
        into floating output. No cast is inferred from ``out``. The source,
        cached chunks and reader dtype are unchanged. Use
        ``vc_reader.augment.Normalize`` in an augmentation pipeline for
        min/max or zero-preserving z-score normalization.
        """
        self._ready()
        nd = self.ndim
        out_dtype, floating = _crop_output_dtype(self.dtype, self.dtype, dtype)
        regions = [(tuple(int(v) for v in s), tuple(int(v) for v in n)) for s, n in regions]
        if out is not None and len(out) != len(regions):
            raise ValueError(f"out has {len(out)} arrays for {len(regions)} regions")
        results, targets = [], []
        for i, (start, size) in enumerate(regions):
            if len(start) != nd or len(size) != nd or any(n < 0 for n in size):
                raise ValueError(f"start/size must have {nd} non-negative entries")
            given = None if out is None else out[i]
            if given is None:
                given = _empty(size, out_dtype)
            elif not isinstance(given, np.ndarray) or given.shape != size or given.dtype != out_dtype:
                raise ValueError(f"out must have shape {size} and dtype {out_dtype}")
            if not given.flags.writeable:
                raise ValueError("out must be writable")
            if floating and any(np.shares_memory(given, previous) for previous in results):
                raise ValueError("output buffers must not overlap")
            results.append(given)
            targets.append(given if given.flags.c_contiguous and (not floating or given.flags.aligned)
                           else np.empty(size, dtype=out_dtype))
        starts, sizes = [list(s) for s, _ in regions], [list(n) for _, n in regions]
        buffers = [t.reshape(-1).view(np.uint8) for t in targets]
        if floating:
            dt = self.dtype
            self._array.read_many_float(starts, sizes, buffers, dt.kind, dt.itemsize, not dt.isnative,
                                        out_dtype == np.float64)
        else:
            self._array.read_many(starts, sizes, buffers, self.dtype.itemsize)
        for res, tgt in zip(results, targets):
            if res is not tgt:
                res[...] = tgt
        return results

    def read_points(self, points: Any) -> np.ndarray:
        """The values at integer points: ``points`` of shape ``(..., ndim)`` -> values of shape ``(...)``.

        Each point's chunk is read once, however many points fall in it. Points
        outside the array read as ``fill_value`` (indices are not wrapped).
        """
        self._ready()
        pts = np.asarray(points)
        if pts.ndim == 0 or pts.shape[-1] != self.ndim:
            raise ValueError(f"points must have shape (..., {self.ndim})")
        if pts.dtype.kind not in "iu" and pts.size:
            raise TypeError(f"points must be integers, got {pts.dtype}")
        flat = np.ascontiguousarray(pts.reshape(-1, self.ndim), dtype=np.int64)
        out = _empty((flat.shape[0],), self.dtype)
        self._array.read_points(flat, out.view(np.uint8), self.dtype.itemsize)
        return out.reshape(pts.shape[:-1])

    def read_affine(
        self, shape: Sequence[int], origin: Any, axes: Any, *, order: int = 1,
        dtype: Any = None, out: np.ndarray | None = None,
    ) -> np.ndarray:
        """Sample an oriented 3-D crop without constructing a coordinate array.

        ``source_zyx = origin + axes @ [z, y, x]`` maps output voxel indices
        into this array's voxel coordinates (not level-0 or physical units).
        Columns of the 3x3 ``axes`` matrix are source-space steps along output
        z, y, x. ``origin`` is the first sample, not the crop centre. Rotations,
        anisotropic spacing, reflections, shears and singleton-axis planes
        are supported. No antialiasing is applied when downsampling.

        Interpolation and boundaries follow :meth:`sample`: order 1 uses
        float64 arithmetic and returns float32 by default (or float64);
        order 0 rounds ties to even and returns the stored dtype. Missing
        chunks and outside neighbours use the source's ``fill_value``.
        Explicit ``dtype`` also permits float32/float64 conversion for order 0.
        ``out`` must be writable, aligned, C-contiguous and have the requested
        shape and dtype. It is returned unchanged as an object. Coordinates
        are generated natively, with bounded point-planning scratch space.
        """
        return self.read_affine_many([(shape, origin, axes)], order=order, dtype=dtype,
                                     out=None if out is None else [out])[0]

    def read_affine_many(
        self, grids: Iterable[tuple[Sequence[int], Any, Any]], *, order: int = 1,
        dtype: Any = None, out: Sequence[np.ndarray | None] | None = None,
    ) -> list[np.ndarray]:
        """Sample ``(shape, origin, axes)`` grids using :meth:`read_affine` semantics.

        Grids may have different shapes. The native call processes them in
        order, parallelizing chunk work within each grid and sharing the
        reader's caches. Chunk loads are deduplicated within each point batch
        (up to 2**20 points); reuse across grids depends on the caches.
        ``out`` optionally supplies one destination (or None) per grid.
        Output buffers must not overlap. No full source box or coordinate
        grid is assembled. Use an augmentation pipeline to normalize the output.
        """
        self._ready()
        if self.ndim != 3:
            raise ValueError("affine reads require a 3-D source array")
        if order not in (0, 1):
            raise ValueError("order must be 0 (nearest) or 1 (linear)")
        default = self.dtype if order == 0 else np.dtype(np.float32)
        out_dtype, floating = _crop_output_dtype(self.dtype, default, dtype)
        if order == 1 and out_dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise ValueError("dtype must be float32 or float64 for order=1")
        if order == 1:
            _check_float_source(self.dtype)
        grids = list(grids)
        if out is not None and len(out) != len(grids):
            raise ValueError("out must have one array per grid")
        shapes, origins, matrices, results = [], [], [], []
        for i, (shape, origin, axes) in enumerate(grids):
            try:
                shape = tuple(operator.index(n) for n in shape)
            except TypeError as exc:
                raise ValueError("shape must have three non-negative integers") from exc
            if len(shape) != 3 or any(n < 0 for n in shape):
                raise ValueError("shape must have three non-negative integers")
            # Python integers avoid overflow before the native allocation.
            count = 1
            for n in shape:
                if n > np.iinfo(np.intp).max:
                    raise ValueError("affine shape is too large")
                count *= n
            if count > np.iinfo(np.intp).max // out_dtype.itemsize:
                raise ValueError("affine shape is too large")
            origin = np.asarray(origin, dtype=np.float64)
            axes = np.asarray(axes, dtype=np.float64)
            if origin.shape != (3,) or not np.isfinite(origin).all():
                raise ValueError("origin must have shape (3,) and be finite")
            if axes.shape != (3, 3) or not np.isfinite(axes).all():
                raise ValueError("axes must have shape (3, 3) and be finite")
            given = None if out is None else out[i]
            if given is None:
                given = _empty(shape, out_dtype)
            elif not isinstance(given, np.ndarray) or given.shape != shape or given.dtype != out_dtype:
                raise ValueError(f"out must have shape {shape} and dtype {out_dtype}")
            if not (given.flags.writeable and given.flags.c_contiguous and given.flags.aligned):
                raise ValueError("out must be writable, aligned and C-contiguous")
            if any(np.shares_memory(given, previous) for previous in results):
                raise ValueError("output buffers must not overlap")
            shapes.append(shape)
            origins.append(origin)
            matrices.append(axes)
            results.append(given)
        dt = self.dtype
        self._array.read_affine_many(
            shapes, np.ascontiguousarray(origins, dtype=np.float64).reshape(-1, 3),
            np.ascontiguousarray(matrices, dtype=np.float64).reshape(-1, 3, 3),
            [a.reshape(-1).view(np.uint8) for a in results], order, dt.kind, dt.itemsize,
            not dt.isnative, out_dtype == np.float64, floating,
        )
        return results

    def sample(self, points: Any, *, order: int = 1, dtype: Any = np.float32) -> np.ndarray:
        """Values at real-valued points: ``points`` of shape ``(..., ndim)`` -> values of shape ``(...)``.

        Element ``i`` sits at coordinate ``i`` along each axis.

        ``order=1``: n-linear (trilinear for 3-D) interpolation of the
        neighbouring elements, computed in float64 and returned as ``dtype``
        (float32 or float64). Neighbours outside the array count as
        ``fill_value``, as in ``scipy.ndimage.map_coordinates(mode="grid-constant",
        cval=fill_value)``. Points with a non-finite coordinate give NaN.

        ``order=0``: the nearest element (halves round to even, like ``np.rint``),
        exactly as stored, in the array's dtype; outside points read as ``fill_value``.

        The neighbours are read natively, each chunk once per batch of points.
        Use :meth:`read_points` for integer indices.
        """
        self._ready()
        pts = np.asarray(points)
        if pts.ndim == 0 or pts.shape[-1] != self.ndim:
            raise ValueError(f"points must have shape (..., {self.ndim})")
        flat = np.ascontiguousarray(pts.reshape(-1, self.ndim), dtype=np.float64)
        if order == 0:
            out = _empty((flat.shape[0],), self.dtype)
            self._array.sample_nearest(flat, out.view(np.uint8), self.dtype.itemsize)
        elif order == 1:
            out_dtype = np.dtype(dtype)
            if out_dtype not in (np.float32, np.float64):
                raise ValueError("dtype must be float32 or float64")
            out = _empty((flat.shape[0],), out_dtype)
            dt = self.dtype
            self._array.sample_linear(flat, out.view(np.uint8), dt.kind, dt.itemsize, not dt.isnative,
                                      out_dtype == np.float64)
        else:
            raise ValueError("order must be 0 (nearest) or 1 (linear)")
        return out.reshape(pts.shape[:-1])

    def __getitem__(self, key: Any) -> np.ndarray:
        keys = key if isinstance(key, tuple) else (key,)
        if any(isinstance(k, (np.ndarray, list)) for k in keys):
            if len(keys) < self.ndim or any(isinstance(k, slice) or k is Ellipsis for k in keys):
                return self._read_mixed_index(keys)
            return self.read_points(_index_points(keys, self.shape))
        start, size, squeeze = _parse_index(key, self.shape)
        out = self.read(start, size)
        return out[tuple(0 if s else slice(None) for s in squeeze)] if any(squeeze) else out

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        """Materialize the array, as with ``reader[:]``."""
        if copy is False:
            raise ValueError("reading a Zarr array requires a copy")
        values = self[...]
        return values if dtype is None else values.astype(dtype, copy=False)

    def _read_mixed_index(self, keys: tuple) -> np.ndarray:
        """NumPy integer-array/slice indexing, assembled as a batch of thin blocks."""
        original = keys
        keys = _expand_index(keys, self.ndim)
        if not any(isinstance(key, slice) for key in keys):
            return self.read_points(_index_points(keys, self.shape))
        arrays, axes, starts, sizes, slice_shape = [], [], [], [], []
        for axis, (key, dim) in enumerate(zip(keys, self.shape)):
            if isinstance(key, slice):
                lo, hi, step = key.indices(dim)
                if step != 1:
                    raise IndexError("only unit-step slices are supported")
                starts.append(lo)
                sizes.append(max(hi - lo, 0))
                slice_shape.append(max(hi - lo, 0))
            else:
                arrays.append(_integer_indices(key, dim))
                axes.append(axis)
                starts.append(0)
                sizes.append(1)
        arrays = np.broadcast_arrays(*arrays)
        advanced_shape = arrays[0].shape
        # Scalars participate in NumPy's advanced-axis placement too. A
        # separating ellipsis still matters when it expands to zero axes.
        separated = axes[-1] - axes[0] + 1 != len(axes)
        if any(k is Ellipsis for k in original):
            e = next(i for i, k in enumerate(original) if k is Ellipsis)
            separated |= (any(not isinstance(k, slice) for k in original[:e]) and
                          any(not isinstance(k, slice) for k in original[e + 1:]))
        insert = 0 if separated else axes[0]
        output_shape = tuple(slice_shape[:insert]) + advanced_shape + tuple(slice_shape[insert:])
        result = _empty(output_shape, self.dtype)
        regions, outputs = [], []
        for idx in np.ndindex(advanced_shape):
            start = starts.copy()
            for axis, indices in zip(axes, arrays):
                start[axis] = int(indices[idx])
            selection = (slice(None),) * insert + idx + (slice(None),) * (len(slice_shape) - insert)
            # Singleton dimensions can be inserted without copying the output.
            view = result[selection]
            for axis in axes:
                view = np.expand_dims(view, axis)
            regions.append((start, sizes))
            outputs.append(view)
        self.read_many(regions, out=outputs)
        return result

    def _ready(self) -> None:
        """Rebuild after a fork, check the reader is open, and pass on refreshed credentials."""
        self._check_process()
        if self._array is None:
            raise ValueError("reader is closed")
        if self.access.source is not None and not self.offline:
            auth = self.access.native()
            if auth is not self._auth:
                self._array.set_auth(auth)
                self._auth = auth


def _memory_cache(memory_cache: int | MemoryCache | None) -> MemoryCache | None:
    if memory_cache is None or memory_cache == 0:
        return None
    if isinstance(memory_cache, MemoryCache):
        return memory_cache
    if isinstance(memory_cache, int):
        return MemoryCache(memory_cache)
    raise TypeError("memory_cache must be a byte count or a vc_reader.reader.MemoryCache")


def _make_spec(
    location: str | os.PathLike[str],
    *,
    level: int | str | None = None,
    group: str | None = None,
    cache_dir: str | os.PathLike[str] | Sequence[str | os.PathLike[str]] | None = None,
    cache_writes: bool = True,
    cache_max_bytes: int | None = None,
    compress_cache: bool = False,
    allow_lossy_cache: bool = False,
    memory_cache: int | MemoryCache | None = None,
    threads: int | None = None,
    connections: int = 16,
    read_mode: str = "mmap",
    auth: Any = "auto",
    offline: bool = False,
    prefetcher=None,
    meta: dict | None = None,
) -> tuple[dict[str, Any], MemoryCache | None]:
    """The settings a :class:`ZarrReader` is built (and rebuilt in other processes) from, and its memory cache.

    ``group`` names the array's parent group explicitly (instead of inferring
    an OME-Zarr level from a digits-only last path part); ``meta`` is its
    already-read ``.zarray``.
    """
    if read_mode not in READ_MODES:
        raise ValueError(f"read_mode must be one of {READ_MODES}")
    loc = os.fspath(location).rstrip("/")
    if level is not None:
        group, level_key = loc, str(level)
        loc = f"{loc}/{level}"
    elif group is not None:
        group = group.rstrip("/")
        level_key = loc[len(group) + 1:]
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
    if offline and is_remote(loc) and not dirs:
        raise ValueError("offline reads of a remote array need cache_dir")
    if cache_max_bytes is not None:
        cache_max_bytes = operator.index(cache_max_bytes)
        if cache_max_bytes < 0:
            raise ValueError("cache_max_bytes must be nonnegative or None")
        if is_remote(loc) and not dirs:
            raise ValueError("cache_max_bytes requires cache_dir")
    access = auth if isinstance(auth, Access) else resolve_access(loc, auth)
    spec = dict(location=loc, group=group, level_key=level_key, cache_dirs=dirs, cache_writes=cache_writes,
                cache_max_bytes=cache_max_bytes,
                compress_cache=compress_cache, allow_lossy_cache=allow_lossy_cache, threads=threads,
                connections=connections, read_mode=read_mode, meta=meta, access=access, offline=offline,
                prefetcher=prefetcher)
    return spec, _memory_cache(memory_cache)


def _check_float_source(dtype: np.dtype) -> None:
    if not (dtype.kind in "biu" and dtype.itemsize in (1, 2, 4, 8)
            or dtype.kind == "f" and dtype.itemsize in (4, 8)):
        raise TypeError("floating output requires a bool, integer, float32 or float64 source")


def _crop_output_dtype(source: np.dtype, default: np.dtype, dtype: Any) -> tuple[np.dtype, bool]:
    result = np.dtype(dtype) if dtype is not None else default
    is_float = result in (np.dtype(np.float32), np.dtype(np.float64))
    if dtype is not None and not is_float and result != source:
        raise ValueError("dtype must be float32, float64 or the stored dtype")
    floating = dtype is not None and is_float
    if floating:
        _check_float_source(source)
    return result, floating


def _empty(shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    """An uninitialized array from the recycled output pool (every byte is overwritten by a read)."""
    return _native.empty(int(np.prod(shape, dtype=np.int64)) * dtype.itemsize).view(dtype).reshape(shape)


def set_output_pool_bytes(max_bytes: int) -> None:
    """Keep up to ``max_bytes`` (default 1 GiB) of freed output buffers for reuse; 0 disables the pool.

    A freshly allocated output costs a page fault per 4 KiB page when first
    written, which for data in the page cache is about as long as the read itself.
    Buffers of arrays the program has dropped are reused by later reads of the
    same size instead.
    """
    _native.set_output_pool_bytes(int(max_bytes))


def output_pool_stats() -> dict[str, int]:
    """The output pool's budget, retained bytes, and allocation hits and misses."""
    return dict(_native.output_pool_stats())


def _shutdown(array, layers: list[CacheLayer], pid: int) -> None:
    if os.getpid() != pid:  # a forked child exiting: the engine belongs to the parent
        return
    array.flush()
    for layer in layers:
        layer.close()


def _parse_index(key: Any, shape: tuple[int, ...]) -> tuple[list[int], list[int], list[bool]]:
    """numpy basic indexing (ints, unit-step slices, Ellipsis) -> start, size, axes to drop."""
    if not isinstance(key, tuple):
        key = (key,)
    key = _expand_index(key, len(shape))
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
            i = operator.index(k)
            if i < 0:
                i += dim
            if not 0 <= i < dim:
                raise IndexError(f"index {k} out of bounds for axis of size {dim}")
            start.append(i)
            size.append(1)
            squeeze.append(True)
    return start, size, squeeze


def _index_points(keys: tuple, shape: tuple[int, ...]) -> np.ndarray:
    """numpy integer-array indexing (one integer or integer array per axis, broadcast) -> points."""
    if len(keys) != len(shape) or any(isinstance(k, slice) or k is Ellipsis or k is None for k in keys):
        raise IndexError("index arrays must be combined with integers only, one per axis "
                         "(use read() for blocks, read_points() for points)")
    arrays = [_integer_indices(k, dim) for k, dim in zip(keys, shape)]
    return np.stack(np.broadcast_arrays(*arrays), axis=-1)


def _integer_indices(key: Any, dim: int) -> np.ndarray:
    a = np.asarray(key)
    if a.dtype.kind == "b":
        raise IndexError("boolean indexing is not supported")
    if a.dtype.kind not in "iu" and a.size:
        raise IndexError(f"index arrays must be integers, got {a.dtype}")
    if a.dtype.kind == "u" and a.size and a.max() >= dim:
        raise IndexError(f"index out of bounds for axis of size {dim}")
    a = a.astype(np.int64)
    a = np.where(a < 0, a + dim, a)
    if a.size and (a.min() < 0 or a.max() >= dim):
        raise IndexError(f"index out of bounds for axis of size {dim}")
    return a


def _expand_index(keys: tuple, ndim: int) -> tuple:
    if sum(k is Ellipsis for k in keys) > 1:
        raise IndexError("an index can only have a single ellipsis")
    if any(k is Ellipsis for k in keys):
        i = next(i for i, k in enumerate(keys) if k is Ellipsis)
        keys = keys[:i] + (slice(None),) * max(0, ndim - len(keys) + 1) + keys[i + 1:]
    if len(keys) > ndim:
        raise IndexError(f"too many indices for array of dimension {ndim}")
    if any(k is None for k in keys):
        raise IndexError("newaxis indexing is not supported")
    return keys + (slice(None),) * (ndim - len(keys))


def open_zarr(location: str | os.PathLike[str], **kwargs: Any) -> ZarrReader:
    """Shorthand for :meth:`ZarrReader.open`."""
    return ZarrReader.open(location, **kwargs)


__all__: Iterable[str] = ["MemoryCache", "ZarrReader", "open_zarr", "http_url", "set_output_pool_bytes",
                          "output_pool_stats"]
