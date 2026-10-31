"""Fast reads of Zarr v2/v3 arrays, local or remote, with optional disk and memory chunk caches.

All reading happens in the compiled extension (``vc_reader._reader``, C++): local
chunk files through persistent memory maps (or ``pread``), HTTP fetches with
libcurl, raw / blosc / vc-delta3d decoding, disk-cache reads and appends in
each cache's own format, an in-memory LRU, and assembly into the outputs, on
C++ threads without the GIL. Opening is native too (``vcr::zarr::openArray``):
this module adds NumPy indexing and conversion, and handling of fork and pickle.

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
rebuilds its C++ engine there from its saved settings (as unpickling does);
the inherited engine is left untouched, and the parent still finishes its own
queued cache writes. A shared MemoryCache starts empty in each worker and
retains one budget shared by all the readers that reference it.
Workers started with ``spawn``/``forkserver`` receive a pickled reader, which is
reopened the same way.
"""

from __future__ import annotations

import operator
import os
import warnings
import weakref
from typing import Any, Iterable, Sequence

import numpy as np

from .. import _reader
from .auth import http_url, is_remote, resolve_access
from .memory_cache import MemoryCache
from ._indexing import parse_index, expand_index
from .store_metadata import StoreMetadata
from .zarr_meta import ArrayMeta

# C++ engines inherited through fork, kept alive and never touched: their
# threads do not exist in the child, and their locks may have been held by them.
_INHERITED: list[Any] = []


def _open_options(settings: dict[str, Any], cache: MemoryCache | None):
    """The native options for these settings, in this process."""
    return _reader.OpenOptions(
        settings["access"], settings["cache_dirs"] or [], settings["cache_writes"], settings["cache_max_bytes"],
        settings["compress_cache"], settings["allow_lossy_cache"], cache._engine if cache is not None else None,
        settings["threads"] or (1 if _loader_worker() else 0), settings["connections"], settings["read_mode"],
        settings["offline"])


class ZarrReader:
    """A Zarr v2/v3 array opened for reading. Use :meth:`open`.

    Indexing is ``[z, y, x]`` like the stored array.

    Safe across fork: used in a forked process (e.g. a DataLoader worker), the
    reader rebuilds its C++ engine there (see the module docstring).
    """

    def __init__(self, settings: dict[str, Any], cache: MemoryCache | None, opened=None):
        self.memory_cache = cache
        self._pid = os.getpid()
        self.location: str = settings["location"]
        self.access = settings["access"]
        self.offline: bool = bool(settings["offline"]) and is_remote(self.location)
        self._options = _open_options(settings, cache)
        computed = settings.get("computed")
        if opened is None and computed is not None:
            opened = _reader.open_computed(**computed, memory_cache=cache._engine if cache is not None else None,
                                           threads=settings["threads"] or (1 if _loader_worker() else 0))
        elif opened is None:
            opened = _reader.open_array(self.location, self._options, settings["group"], settings["level_key"],
                                        settings["meta"], settings["attrs"])
        self._opened = opened
        self._array = opened.array
        self.threads: int = self._array.threads
        self.meta = ArrayMeta.from_native(opened.meta)
        self._settings = dict(settings, meta=self.meta.raw)
        self._attrs = opened.known_attributes
        self._metadata: StoreMetadata | None = settings["store_metadata"]
        self.cache_layers = opened.layers
        self.write_layer = self.cache_layers[opened.writer] if opened.writer >= 0 else None
        for busy in opened.busy:
            warnings.warn(f"VC3D cache {busy} is being replaced by another process (its lease is held "
                          "exclusively); not using it", stacklevel=3)
        # Queued cache writes finish even if the reader is dropped without close().
        self._finalizer = weakref.finalize(self, _shutdown, opened, self._pid)

    # -- construction -----------------------------------------------------------

    def cuda(self, *, cache_bytes=512 << 20, device=0):
        """Optional CUDA adapter for 3-D uint8 CT; outputs remain on device.

        Owns a bounded device chunk cache. Create one per worker/device and
        reuse it across reads; close it or use it as a context manager.
        """
        from .cuda import CudaReader
        return CudaReader(self, cache_bytes=cache_bytes, device=device)

    @classmethod
    def computed(
        cls,
        compute: Any,
        *,
        shape: Sequence[int],
        chunks: Sequence[int],
        dtype: str = "|u1",
        fill_value: Any = 0,
        identity: str = "",
        cache_dir: str | os.PathLike[str] | None = None,
        cache_max_bytes: int | None = None,
        compress_cache: bool = False,
        memory_cache: int | MemoryCache | None = None,
        threads: int | None = None,
    ) -> ZarrReader:
        """An array whose chunks ``compute(chunk_index)`` returns, read and sampled like a stored one.

        ``compute`` returns the whole chunk, or ``None`` when it reads as ``fill_value``; it runs on the
        reader's threads holding the GIL. ``identity`` names what the values depend on and is required with
        ``cache_dir``, where computed chunks persist for later opens. Pickling needs a picklable ``compute``.
        """
        element = np.dtype(dtype)

        def chunk_bytes(index):
            result = compute(index)
            return None if result is None else np.asarray(result, element).tobytes()

        settings, cache = _make_settings(f"computed:{identity}", level=None, cache_dir=None, cache_writes=False,
                                         cache_max_bytes=None, compress_cache=False, allow_lossy_cache=False,
                                         memory_cache=memory_cache, threads=threads, connections=1,
                                         read_mode="mmap", auth=None, offline=False)
        settings["computed"] = {
            "shape": [int(n) for n in shape], "chunks": [int(n) for n in chunks], "dtype": element.str,
            "fill_value": fill_value, "identity": identity, "compute": chunk_bytes,
            "cache_dir": None if cache_dir is None else os.fspath(cache_dir),
            "cache_max_bytes": cache_max_bytes, "compress_cache": compress_cache,
        }
        return cls(settings, cache)

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
        refresh: bool = False,
    ) -> ZarrReader:
        """Open the Zarr v2/v3 array at ``location`` (a directory, ``http(s)://`` or ``s3://`` URL).

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
        between readers. ``threads`` (default 16 local, 32 remote; one in a
        PyTorch DataLoader worker, whose processes already read in parallel)
        read and decode; at most ``connections`` HTTP requests are in flight.
        ``read_mode`` is how uncompressed chunk files are read: ``"mmap"``
        (persistent memory maps, the default) or ``"slab"`` (``pread``).

        ``refresh`` reads the store's documents and the catalog as published now and
        replaces the metadata kept with the volume (see :attr:`metadata`), which is
        otherwise gathered only the first time a volume is opened.

        The reader works on background threads. In a process forked after this
        call (e.g. a Linux DataLoader worker) it rebuilds them on first use.
        """
        settings, cache = _make_settings(location, level=level, cache_dir=cache_dir, cache_writes=cache_writes,
                                         cache_max_bytes=cache_max_bytes,
                                         compress_cache=compress_cache, allow_lossy_cache=allow_lossy_cache,
                                         memory_cache=memory_cache, threads=threads, connections=connections,
                                         read_mode=read_mode, auth=auth, offline=offline)
        reader = cls(settings, cache)
        reader._metadata = StoreMetadata.from_json(
            _reader.volume_metadata_json(reader._opened.store, reader._options, refresh))
        return reader

    def __getstate__(self):
        return {"settings": dict(self._settings, attrs=self._attrs, store_metadata=self._metadata),
                "memory_cache": self.memory_cache}

    def __setstate__(self, state):
        self.__init__(state["settings"], state["memory_cache"])

    @property
    def metadata(self) -> StoreMetadata:
        """What is known about the volume: its store's ``meta.json`` / ``metadata.json``
        and the Open Data catalog, read when it was opened."""
        if self._metadata is None:
            self._metadata = StoreMetadata.from_json(_reader.volume_metadata_json(self._opened.store, self._options))
        return self._metadata

    @property
    def voxel_size_zyx(self) -> tuple[float, float, float] | None:
        """Micrometers per voxel of this array along Z, Y, X; ``None`` when the store does not say."""
        metadata = self.metadata
        size = _reader.level_voxel_size_zyx(metadata.voxel_size_um, metadata.documents, self._opened.level_key)
        return None if size is None else tuple(size)

    @property
    def voxel_size(self) -> float | None:
        """Micrometers per voxel of this array; ``None`` when the store does not say or its
        voxels are not cubic (see :attr:`voxel_size_zyx`)."""
        size = self.voxel_size_zyx
        return size[0] if size is not None and size[0] == size[1] == size[2] else None

    def __repr__(self) -> str:
        return f"ZarrReader({self.location!r}, shape={self.shape}, chunks={self.chunks}, dtype={self.dtype})"

    def _check_process(self) -> None:
        """After a fork, rebuild the C++ engine in this process (see the module docstring)."""
        if self._pid == os.getpid() or self._array is None:
            return
        self._abandon_inherited()
        self.__init__(dict(self._settings, attrs=self._attrs, store_metadata=self._metadata), self.memory_cache)

    def _abandon_inherited(self) -> None:
        self._finalizer.detach()  # the parent flushes and closes its own engine
        _INHERITED.append(self._opened)
        self._opened, self._array, self.cache_layers, self.write_layer = None, None, [], None

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
            self._attrs = self._opened.attributes(self._options)
        return self._attrs

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

    # -- reads ---------------------------------------------------------------------

    def read(self, selection, *, out=None, dtype=None):
        """Read a NumPy-style selection, optionally converting directly into out.

        Accept integers, unit-step slices and ellipsis, with the same clipping
        and axis removal as indexing. ``out`` uses the selected output shape.
        """
        return self.read_many([selection], out=None if out is None else [out], dtype=dtype)[0]

    def read_many(self, selections, *, out=None, dtype=None):
        """Read slice selections together, sharing chunk reads across the batch.

        ``out`` supplies one optional destination per selection. Conversion
        writes directly into the destinations; no intermediate crop is needed.
        """
        plans = [parse_index(selection, self.shape) for selection in selections]
        if out is not None and len(out) != len(plans):
            raise ValueError(f'out has {len(out)} arrays for {len(plans)} selections')
        buffers = None
        if out is not None:
            buffers = []
            for given, (_, size, squeeze) in zip(out, plans):
                shape = tuple(n for n, drop in zip(size, squeeze) if not drop)
                if given is not None:
                    if not isinstance(given, np.ndarray) or given.shape != shape:
                        raise ValueError(f'out must have shape {shape}')
                    axes = tuple(i for i, drop in enumerate(squeeze) if drop)
                    buffers.append(np.expand_dims(given, axes) if axes else given)
                else:
                    buffers.append(None)
        values = self.read_regions([(start, size) for start, size, _ in plans], out=buffers, dtype=dtype)
        results = []
        for i, (value, (_, _, squeeze)) in enumerate(zip(values, plans)):
            if out is not None and out[i] is not None:
                results.append(out[i])
            else:
                index = tuple(0 if drop else slice(None) for drop in squeeze)
                results.append(value[index] if any(squeeze) else value)
        return results

    def read_region(
        self, start: Sequence[int], size: Sequence[int], out: np.ndarray | None = None, *,
        dtype: Any = None,
    ) -> np.ndarray:
        """Read an explicitly sized region, filling coordinates outside the array."""
        return self.read_regions([(start, size)], None if out is None else [out],
                              dtype=dtype)[0]

    def read_regions(
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
        ``vc_reader.transforms.Normalize`` in an transform pipeline for
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
        dtype: Any = None, out: np.ndarray | None = None, coordinate_origin=None,
    ) -> np.ndarray:
        """Sample an oriented 3-D crop without constructing a coordinate array.

        ``source_zyx = origin + axes @ [z, y, x]`` maps output voxel indices
        into this array's voxel coordinates (not level-0 or physical units).
        Columns of the 3x3 ``axes`` matrix are source-space steps along output
        z, y, x. ``origin`` is the first sample, not the crop centre. Rotations,
        anisotropic spacing, reflections, shears and singleton-axis planes
        are supported. No antialiasing is applied when downsampling.

        Interpolation and boundaries follow :meth:`sample`: uint8 CT with
        uint8/float32 output uses local float32 geometry and arithmetic.
        Explicit float64 output retains the legacy float64 reference path;
        order 0 rounds half-voxel ties up and returns the stored dtype. Missing
        chunks and outside neighbours use the source's ``fill_value``.
        Explicit ``dtype`` also permits float32/float64 conversion for order 0.
        ``out`` must be writable, aligned, C-contiguous and have the requested
        shape and dtype. It is returned unchanged as an object. Coordinates
        are generated in C++, with bounded point-planning scratch space.
        """
        return self.read_affine_many([(shape, origin, axes)], order=order, dtype=dtype,
                                     out=None if out is None else [out], coordinate_origin=coordinate_origin)[0]

    def read_affine_many(
        self, grids: Iterable[tuple[Sequence[int], Any, Any]], *, order: int = 1,
        dtype: Any = None, out: Sequence[np.ndarray | None] | None = None,
        coordinate_origin=None,
    ) -> list[np.ndarray]:
        """Sample ``(shape, origin, axes)`` grids using :meth:`read_affine` semantics.

        Grids may have different shapes. The C++ call processes them in
        order, parallelizing chunk work within each grid and sharing the
        reader's caches. Chunk loads are deduplicated within each point batch
        (up to 2**20 points); reuse across grids depends on the caches.
        ``out`` optionally supplies one destination (or None) per grid.
        Output buffers must not overlap. Dense uint8/float32 grids may use a
        contiguous source box, capped at 64 MiB and eight source voxels per
        output voxel. Sparse grids use direct chunks. No dense coordinate grid
        is assembled. Use an transform pipeline to normalize the output.
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
        if self.dtype == np.uint8 and out_dtype != np.float64:
            from ._sampling import affine32
            prepared = [affine32(*g, coordinate_origin) for g in grids]
            results = []
            for i, (shape, _, _, _) in enumerate(prepared):
                given = None if out is None else out[i]
                if given is None:
                    given = _empty(shape, out_dtype)
                if not isinstance(given, np.ndarray) or given.shape != shape or given.dtype != out_dtype:
                    raise ValueError(f"out must have shape {shape} and dtype {out_dtype}")
                if not (given.flags.writeable and given.flags.c_contiguous and given.flags.aligned):
                    raise ValueError("out must be writable, aligned, C-contiguous")
                if any(np.shares_memory(given, previous) for previous in results):
                    raise ValueError("output buffers must not overlap")
                results.append(given)
            for (shape, base, origin, axes), result in zip(prepared, results):
                target = result if order or out_dtype == np.uint8 else np.empty(shape, np.uint8)
                self._array._affine_u8_float32(shape, origin.tolist(), axes.ravel().tolist(), base,
                                             target.reshape(-1).view(np.uint8), order)
                if target is not result:
                    result[...] = target
            return results
        if coordinate_origin is not None:
            raise ValueError("coordinate_origin requires uint8 CT and uint8/float32 output")
        shapes, origins, matrices, results = [], [], [], []
        for i, (shape, origin, axes) in enumerate(grids):
            try:
                shape = tuple(operator.index(n) for n in shape)
            except TypeError as exc:
                raise ValueError("shape must have three non-negative integers") from exc
            if len(shape) != 3 or any(n < 0 for n in shape):
                raise ValueError("shape must have three non-negative integers")
            # Python integers avoid overflow before the C++ allocation.
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

    def sample(self, points: Any, *, order: int = 1, dtype: Any = np.float32, coordinate_origin=None) -> np.ndarray:
        """Values at real-valued points: ``points`` of shape ``(..., ndim)`` -> values of shape ``(...)``.

        Element ``i`` sits at coordinate ``i`` along each axis.

        ``order=1``: n-linear (trilinear for 3-D) interpolation of the
        neighbouring elements, computed in float64 and returned as ``dtype``
        (float32 or float64). For 3-D uint8 CT, float32 output instead uses
        float32 weights and accumulation. Compact point batches use local
        float32 coordinates after splitting off an integer global origin;
        dispersed batches retain float64 coordinates to preserve voxel positions.
        Pass coordinate_origin=(z,y,x) with
        already-local float32 points to preserve precision at large locations.
        Neighbours outside the array count as
        ``fill_value``, as in ``scipy.ndimage.map_coordinates(mode="grid-constant",
        cval=fill_value)``. Points with a non-finite coordinate give NaN.

        ``order=0``: the nearest element (half-voxel ties round up, as SciPy's ``order=0``),
        exactly as stored, in the array's dtype; outside points read as ``fill_value``.

        The neighbours are read in C++, each chunk once per batch of points.
        Use :meth:`read_points` for integer indices.
        """
        self._ready()
        if self.ndim == 3 and self.dtype == np.uint8 and (order == 0 or np.dtype(dtype) == np.float32):
            from ._sampling import points32
            if order not in (0, 1):
                raise ValueError("order must be 0 or 1")
            local, base, shape = points32(points, coordinate_origin)
            result = _empty((len(local),), np.dtype(np.uint8 if order == 0 else np.float32))
            self._array._sample_u8_float32(local, base, result.view(np.uint8), order)
            return result.reshape(shape)
        if coordinate_origin is not None:
            raise ValueError("coordinate_origin requires 3-D uint8 CT and uint8/float32 output")
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
        start, size, squeeze = parse_index(key, self.shape)
        out = self.read_region(start, size)
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
        keys = expand_index(keys, self.ndim)
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
        self.read_regions(regions, out=outputs)
        return result

    def _ready(self) -> None:
        """Rebuild after a fork and check the reader is open."""
        self._check_process()
        if self._array is None:
            raise ValueError("reader is closed")


def _loader_worker() -> bool:
    """True in a PyTorch DataLoader worker. A reader used in a forked worker rebuilds
    its engine there, so the check runs in the worker itself."""
    import sys
    torch = sys.modules.get('torch')
    return torch is not None and torch.utils.data.get_worker_info() is not None


def _memory_cache(memory_cache: int | MemoryCache | None) -> MemoryCache | None:
    if memory_cache is None or memory_cache == 0:
        return None
    if isinstance(memory_cache, MemoryCache):
        return memory_cache
    if isinstance(memory_cache, int):
        return MemoryCache(memory_cache)
    raise TypeError("memory_cache must be a byte count or a vc_reader.reader.MemoryCache")


def _make_settings(
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
) -> tuple[dict[str, Any], MemoryCache | None]:
    """What a :class:`ZarrReader` is built (and rebuilt in other processes) from, and its memory cache."""
    loc = os.fspath(location).rstrip("/")
    level_key = None
    if level is not None:
        group, level_key = loc, str(level)
        loc = f"{loc}/{level}"
    if cache_dir is None:
        dirs = None
    elif isinstance(cache_dir, (str, os.PathLike)):
        dirs = [os.fspath(cache_dir)]
    else:
        dirs = [os.fspath(d) for d in cache_dir]
    if cache_max_bytes is not None:
        cache_max_bytes = operator.index(cache_max_bytes)
    settings = dict(location=loc, group=group, level_key=level_key, cache_dirs=dirs, cache_writes=cache_writes,
                    cache_max_bytes=cache_max_bytes, compress_cache=compress_cache,
                    allow_lossy_cache=allow_lossy_cache, threads=threads, connections=connections,
                    read_mode=read_mode, meta=None, attrs=None, access=resolve_access(loc, auth, offline=offline),
                    offline=offline, store_metadata=None)
    return settings, _memory_cache(memory_cache)


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
    return _reader.empty(int(np.prod(shape, dtype=np.int64)) * dtype.itemsize).view(dtype).reshape(shape)


def set_output_pool_bytes(max_bytes: int) -> None:
    """Keep up to ``max_bytes`` (default 1 GiB) of freed output buffers for reuse; 0 disables the pool.

    A freshly allocated output costs a page fault per 4 KiB page when first
    written, which for data in the page cache is about as long as the read itself.
    Buffers of arrays the program has dropped are reused by later reads of the
    same size instead.
    """
    _reader.set_output_pool_bytes(int(max_bytes))


def output_pool_stats() -> dict[str, int]:
    """The output pool's budget, retained bytes, and allocation hits and misses."""
    return dict(_reader.output_pool_stats())


def _shutdown(opened, pid: int) -> None:
    if os.getpid() != pid:  # a forked child exiting: the engine belongs to the parent
        return
    opened.close()


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


def open_zarr(location: str | os.PathLike[str], **kwargs: Any) -> ZarrReader:
    """Shorthand for :meth:`ZarrReader.open`."""
    return ZarrReader.open(location, **kwargs)


__all__: Iterable[str] = ["MemoryCache", "ZarrReader", "open_zarr", "http_url", "set_output_pool_bytes",
                          "output_pool_stats"]
