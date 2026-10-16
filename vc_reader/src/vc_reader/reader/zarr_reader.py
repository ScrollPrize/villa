"""Fast reads of zarr v2 arrays, local or remote, with optional disk and memory chunk caches.

This bypasses zarr-python: it parses ``.zarray`` itself and fetches, decodes
and copies chunks on a thread pool (network I/O, file reads, numcodecs codecs
and large numpy copies all release the GIL). Values are exactly the stored
values; missing chunks and out-of-bounds regions read as ``fill_value``.
"""

from __future__ import annotations

import itertools
import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Iterable, Sequence

import numpy as np

from .cache import ABSENT, MemoryChunkCache
from .disk_cache import DiskCachedSource
from .sources import ChunkSource, HttpSource, LocalSource, http_url, willneed
from .zarr_meta import ArrayMeta

#: Below this many bytes of copying, warm local reads run inline in the caller.
_INLINE_COPY_BYTES = 1 << 20
#: Pool reads whose pieces average less than this are dispatched in hinted groups.
_GROUP_PIECE_BYTES = 1 << 20


class ZarrReader:
    """A zarr v2 array opened for reading.

    Use :meth:`open`. Indexing is ``[z, y, x]`` like the stored array.
    """

    def __init__(
        self,
        source: ChunkSource,
        *,
        memory_cache: MemoryChunkCache | None = None,
        workers: int | None = None,
        inline_bytes: int = _INLINE_COPY_BYTES,
        spec: dict[str, Any] | None = None,
    ):
        self.source = source
        self.inline_bytes = inline_bytes
        self.meta: ArrayMeta = source.meta
        self.memory_cache = memory_cache if memory_cache is not None and memory_cache.max_bytes > 0 else None
        # Local reads: 16 threads beat 32 on an NVMe RAID (warm 512^3 blocks 0.060 -> 0.048 s);
        # remote: requests are capped separately by HttpSource(connections=).
        self.workers = workers or (32 if not isinstance(source, LocalSource) else min(16, os.cpu_count() or 4))
        self._spec = spec
        self._init_runtime()

    def _init_runtime(self) -> None:
        self._pool: ThreadPoolExecutor | None = None
        self._pool_lock = threading.Lock()
        self._inflight: dict[tuple[int, ...], Future] = {}
        self._inflight_lock = threading.Lock()
        # Raw local chunks are cheap to read; everything else is worth
        # deduplicating when several reads want the same chunk at once.
        self._local_raw = isinstance(self.source, LocalSource) and self.meta.is_raw
        self._mapped = self.source.maps_files and self.memory_cache is None

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
        memory_cache: int | MemoryChunkCache | None = None,
        workers: int | None = None,
        connections: int = 16,
        read_mode: str = "slab",
        inline_bytes: int = _INLINE_COPY_BYTES,
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

        ``memory_cache`` is a byte budget or a :class:`MemoryChunkCache` shared
        between readers. ``read_mode`` picks how uncompressed local chunks are
        read (see :class:`~vc_reader.reader.sources.LocalSource`). Local reads
        smaller than ``inline_bytes`` run in the calling thread.
        """
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
        cache = MemoryChunkCache(memory_cache) if isinstance(memory_cache, int) else memory_cache
        spec = dict(location=loc, group=group, level_key=level_key, cache_dirs=dirs, cache_writes=cache_writes,
                    allow_lossy_cache=allow_lossy_cache, workers=workers, connections=connections,
                    read_mode=read_mode, inline_bytes=inline_bytes, meta=None)
        return cls._from_spec(spec, cache)

    @classmethod
    def _from_spec(cls, spec: dict[str, Any], cache: MemoryChunkCache | None) -> ZarrReader:
        loc = spec["location"]
        meta = ArrayMeta.from_zarray(spec["meta"]) if spec["meta"] else None
        if loc.startswith(("http://", "https://", "s3://")):
            url = http_url(loc)
            source: ChunkSource = HttpSource(url, meta, connections=spec["connections"])
            if spec["cache_dirs"]:
                group = http_url(spec["group"]) if spec["group"] else None
                source = DiskCachedSource.open(
                    source, spec["cache_dirs"], url, group_url=group, level_key=spec["level_key"],
                    read_mode=spec["read_mode"], writable=spec["cache_writes"],
                    allow_lossy=spec["allow_lossy_cache"])
        else:
            source = LocalSource(loc, meta, read_mode=spec["read_mode"])
        spec = dict(spec, meta=dict(source.meta.raw))
        return cls(source, memory_cache=cache, workers=spec["workers"], inline_bytes=spec["inline_bytes"], spec=spec)

    def __getstate__(self):
        if self._spec is None:
            raise TypeError("only readers made with ZarrReader.open can be pickled")
        return {"spec": self._spec, "memory_cache": self.memory_cache}

    def __setstate__(self, state):
        # Reopen in the new process from recorded metadata (no network round trip).
        other = self._from_spec(state["spec"], state["memory_cache"])
        self.__dict__.update(other.__dict__)

    def __repr__(self) -> str:
        return f"ZarrReader({self.source.identity!r}, shape={self.shape}, chunks={self.chunks}, dtype={self.dtype})"

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None
        self.source.close()

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

    # -- chunks ----------------------------------------------------------------------

    def _executor(self) -> ThreadPoolExecutor:
        if self._pool is None:
            with self._pool_lock:
                if self._pool is None:
                    self._pool = ThreadPoolExecutor(self.workers, thread_name_prefix="vc_reader")
        return self._pool

    def chunk(self, idx: Sequence[int]) -> np.ndarray | None:
        """Full chunk ``idx`` (read-only when it comes from a cache or a mapping); None if not stored."""
        return self._chunk(tuple(int(i) for i in idx), mapped=False)

    def _chunk(self, idx: tuple[int, ...], mapped: bool) -> np.ndarray | None:
        mc = self.memory_cache
        if mc is not None:
            hit = mc.get((self.source.identity, idx))
            if hit is not ABSENT:
                return hit
        if self._local_raw:
            arr = self.source.get(idx, mapped=mapped)
            if mc is not None:
                mc.put((self.source.identity, idx), arr)
            return arr
        with self._inflight_lock:
            fut = self._inflight.get(idx)
            owner = fut is None
            if owner:
                fut = self._inflight[idx] = Future()
        if not owner:
            return fut.result()
        try:
            arr = self.source.get(idx, mapped=mapped)
            if mc is not None:
                mc.put((self.source.identity, idx), arr)
            fut.set_result(arr)
            return arr
        except BaseException as exc:
            fut.set_exception(exc)
            raise
        finally:
            with self._inflight_lock:
                del self._inflight[idx]

    def chunk_indices(self, start: Sequence[int], size: Sequence[int]) -> list[tuple[int, ...]]:
        """Indices of the stored chunks that ``[start, start+size)`` overlaps (clipped to the array)."""
        lo = [max(int(s), 0) for s in start]
        hi = [min(int(s) + int(n), dim) for s, n, dim in zip(start, size, self.shape)]
        if any(h <= l for l, h in zip(lo, hi)):
            return []
        c = self.chunks
        return list(itertools.product(*(range(l // ci, (h - 1) // ci + 1) for l, h, ci in zip(lo, hi, c))))

    def prefetch(self, start: Sequence[int], size: Sequence[int]) -> list[Future]:
        """Start loading the chunks of a region into the caches; returns the futures."""
        pool = self._executor()
        return [pool.submit(self._chunk, idx, False) for idx in self.chunk_indices(start, size)]

    # -- region reads ----------------------------------------------------------------

    def _plan(self, start: Sequence[int], size: Sequence[int], out: np.ndarray | None):
        """Output array plus the (chunk, chunk slices, output slices) pieces of one region."""
        nd = self.ndim
        start = tuple(int(s) for s in start)
        size = tuple(int(n) for n in size)
        if len(start) != nd or len(size) != nd or any(n < 0 for n in size):
            raise ValueError(f"start/size must have {nd} non-negative entries")
        if out is None:
            out = np.empty(size, dtype=self.dtype)
        elif out.shape != size or out.dtype != self.dtype:
            raise ValueError(f"out must have shape {size} and dtype {self.dtype}")
        lo = tuple(max(s, 0) for s in start)
        hi = tuple(min(s + n, dim) for s, n, dim in zip(start, size, self.shape))
        if any(h <= l for l, h in zip(lo, hi)):
            out.fill(self.fill_value)
            return out, []
        if lo != start or hi != tuple(s + n for s, n in zip(start, size)):
            out.fill(self.fill_value)  # out-of-bounds margins; the inside is overwritten below
        c = self.chunks
        pieces = []
        for idx in itertools.product(*(range(l // ci, (h - 1) // ci + 1) for l, h, ci in zip(lo, hi, c))):
            src = tuple(slice(max(l, i * ci) - i * ci, min(h, (i + 1) * ci) - i * ci) for l, h, i, ci in zip(lo, hi, idx, c))
            dst = tuple(slice(max(l, i * ci) - s, min(h, (i + 1) * ci) - s) for l, h, i, ci, s in zip(lo, hi, idx, c, start))
            pieces.append((idx, src, out[dst]))
        return out, pieces

    def _copy_piece(self, idx: tuple[int, ...], src: tuple[slice, ...], dst: np.ndarray) -> None:
        """Fill ``dst`` with ``chunk[idx][src]`` (or fill_value), by the cheapest route available."""
        if self.memory_cache is None:
            if self._local_raw and not self._mapped:
                if not self.source.read_into(idx, src, dst):
                    dst.fill(self.fill_value)
                return
            if isinstance(self.source, DiskCachedSource) and not self._mapped:
                hit = self.source.read_cached_into(idx, src, dst)
                if hit is not None:
                    if not hit:
                        dst.fill(self.fill_value)
                    return
        arr = self._chunk(idx, self._mapped)
        if arr is None:
            dst.fill(self.fill_value)
            return
        if self._mapped and arr.ndim == 3:
            willneed(arr, tuple(s.start for s in src), tuple(s.stop for s in src))
        dst[...] = arr[src]

    def _run(self, pieces: list, copy_bytes: int | None) -> None:
        """Fill every piece; ``copy_bytes=None`` forces the worker pool."""
        mc = self.memory_cache
        if mc is not None:
            # Memory hits are just copies: do them here, hand the rest to the pool.
            rest = []
            ident = self.source.identity
            for idx, src, dst in pieces:
                hit = mc.get((ident, idx))
                if hit is ABSENT:
                    rest.append((idx, src, dst))
                elif hit is None:
                    dst.fill(self.fill_value)
                else:
                    dst[...] = hit[src]
            pieces = rest
        if not pieces:
            return
        # Small warm local reads are cheaper in this thread than via the pool.
        # Memory-cache misses always go to the pool: they read whole chunks.
        small_local = self._local_raw and self.memory_cache is None and copy_bytes is not None \
            and copy_bytes < self.inline_bytes
        if small_local or (copy_bytes is not None and len(pieces) == 1):
            if self._mapped:
                # Map every piece and queue all their pages before the first copy blocks on one.
                arrays = [self._chunk(idx, True) for idx, _, _ in pieces]
                for arr, (_, src, _) in zip(arrays, pieces):
                    if arr is not None and arr.ndim == 3:
                        willneed(arr, tuple(s.start for s in src), tuple(s.stop for s in src))
                for arr, (_, src, dst) in zip(arrays, pieces):
                    if arr is None:
                        dst.fill(self.fill_value)
                    else:
                        dst[...] = arr[src]
            elif small_local:
                for found, (_, _, dst) in zip(self.source.read_many_into(pieces), pieces):
                    if not found:
                        dst.fill(self.fill_value)
            else:
                for piece in pieces:
                    self._copy_piece(*piece)
            return
        # Workers read their pieces directly. The calling thread does not queue
        # readahead first: with parallel readers the device already has queue
        # depth, and a serial hint pass over every piece halved cold throughput
        # (512^3 blocks: 2.6 vs 5.1 GB/s on an NVMe RAID).
        pool = self._executor()
        if self._local_raw and not self._mapped and self.memory_cache is None and \
                (copy_bytes or sum(d.nbytes for _, _, d in pieces)) < _GROUP_PIECE_BYTES * len(pieces):
            # Many small pieces (crop batches): a few contiguous groups per worker, each
            # hinted by its worker before it reads, so readahead is queued in parallel and
            # per-piece task overhead disappears (cold 256 x 128^3 batch: 0.200 -> 0.172 s).
            # Large pieces stay one task each: there grouping cost 20% (warm 512^3 blocks).
            n = min(len(pieces), self.workers * 2)
            groups = [pieces[len(pieces) * i // n:len(pieces) * (i + 1) // n] for i in range(n)]

            def work(group):
                for ok, (_, _, dst) in zip(self.source.read_many_into(group), group):
                    if not ok:
                        dst.fill(self.fill_value)

            for f in [pool.submit(work, g) for g in groups]:
                f.result()
            return
        for f in [pool.submit(self._copy_piece, *p) for p in pieces]:
            f.result()

    def read(self, start: Sequence[int], size: Sequence[int], out: np.ndarray | None = None) -> np.ndarray:
        """Block ``[start, start+size)``; parts outside the array read as ``fill_value``."""
        out, pieces = self._plan(start, size, out)
        self._run(pieces, out.nbytes)
        return out

    def read_many(
        self,
        regions: Iterable[tuple[Sequence[int], Sequence[int]]],
        out: Sequence[np.ndarray | None] | None = None,
    ) -> list[np.ndarray]:
        """Several ``(start, size)`` blocks at once: all their chunk reads share the worker pool.

        Faster than calling :meth:`read` in a loop when each region is small,
        because reads of different regions overlap. ``out`` optionally gives one
        destination array (or None) per region; reusing destinations between
        calls avoids first-touch page faults on new memory (warm 256 x 128^3
        batches: about 25% faster).
        """
        regions = list(regions)
        if out is not None and len(out) != len(regions):
            raise ValueError(f"out has {len(out)} arrays for {len(regions)} regions")
        outs, pieces = [], []
        for i, (start, size) in enumerate(regions):
            o, p = self._plan(start, size, None if out is None else out[i])
            outs.append(o)
            pieces.extend(p)
        # Several regions always go to the pool, so their reads overlap.
        self._run(pieces, outs[0].nbytes if len(outs) == 1 else None)
        return outs

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


__all__: Iterable[str] = ["ZarrReader", "open_zarr"]
