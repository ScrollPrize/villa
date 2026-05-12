"""LRU cache + async prefetch in front of a zarr array.

The fiber tracer fetches 128^3 windows from a large remote (S3-backed) zarr.
As the trace advances, successive windows overlap heavily with the previous
one, so reading at the *native chunk* level (typically 128^3 for the
``vesuvius-challenge-open-data`` zarrs) and caching the most recent chunks
keeps the network cost roughly proportional to the trace length rather than
to the number of window evaluations.

The cache is thread-safe so a small ``ThreadPoolExecutor`` can prefetch the
chunks needed for the *next* window in parallel with model inference on the
*current* window. The synchronous ``get`` path awaits any in-flight prefetch
for the same chunk so reads are not duplicated.

This module does not depend on PyTorch — it operates purely on numpy arrays.
"""

from __future__ import annotations

import concurrent.futures
import threading
from dataclasses import dataclass, field
from typing import Any, Protocol

import cachetools
import numpy as np

from vesuvius.data.utils import open_zarr


class _ZarrLike(Protocol):
    """Minimal duck-typed interface we need from a zarr array."""

    chunks: tuple[int, ...]
    shape: tuple[int, ...]
    dtype: Any

    def __getitem__(self, item: Any) -> np.ndarray: ...


@dataclass
class CacheStats:
    """Counters for cache hit / miss / prefetch behaviour."""

    hits: int = 0
    prefetch_hits: int = 0
    misses: int = 0
    prefetch_scheduled: int = 0
    bytes_read: int = 0

    @property
    def total_requests(self) -> int:
        return self.hits + self.prefetch_hits + self.misses

    def as_dict(self) -> dict[str, int]:
        return {
            "hits": self.hits,
            "prefetch_hits": self.prefetch_hits,
            "misses": self.misses,
            "prefetch_scheduled": self.prefetch_scheduled,
            "bytes_read": self.bytes_read,
        }


class ChunkLRUCache:
    """LRU cache over native chunks of a single zarr array.

    Parameters
    ----------
    source:
        The zarr array (or any duck-typed equivalent) to read from. Must expose
        ``shape``, ``chunks`` and slice-indexed reads via ``__getitem__``.
    maxsize:
        Number of native chunks to keep cached. Defaults to 32 (~256 MB for
        128^3 float32 chunks).
    num_prefetch_workers:
        Worker thread count for ``prefetch``. Set to 0 to disable async
        prefetch entirely (everything goes through the synchronous ``get``).
    """

    def __init__(
        self,
        source: _ZarrLike,
        *,
        maxsize: int = 32,
        num_prefetch_workers: int = 2,
    ) -> None:
        if int(maxsize) <= 0:
            raise ValueError(f"maxsize must be positive; got {maxsize!r}")
        self.source = source
        self.chunk_shape = tuple(int(v) for v in source.chunks)
        self.volume_shape = tuple(int(v) for v in source.shape)
        if len(self.chunk_shape) != len(self.volume_shape):
            raise ValueError("source.shape and source.chunks must have the same rank")
        self._cache: cachetools.LRUCache[tuple[int, ...], np.ndarray] = cachetools.LRUCache(maxsize=int(maxsize))
        self._pending: dict[tuple[int, ...], concurrent.futures.Future] = {}
        self._lock = threading.RLock()
        self._executor: concurrent.futures.ThreadPoolExecutor | None
        if int(num_prefetch_workers) > 0:
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=int(num_prefetch_workers),
                thread_name_prefix="fiber-prefetch",
            )
        else:
            self._executor = None
        self.stats = CacheStats()

    # --- public API ----------------------------------------------------- #

    def get(self, chunk_idx: tuple[int, ...]) -> np.ndarray:
        """Return the chunk at ``chunk_idx`` (synchronously).

        Waits on an in-flight prefetch for the same chunk if one exists.
        The returned array is *not* defensively copied; callers must not
        mutate it. Cached chunks are kept at the source's native dtype.
        """

        key = tuple(int(v) for v in chunk_idx)
        with self._lock:
            entry = self._cache.get(key)
            if entry is not None:
                self.stats.hits += 1
                return entry
            pending = self._pending.get(key)
        if pending is not None:
            result = pending.result()
            with self._lock:
                self.stats.prefetch_hits += 1
            return result
        chunk = self._fetch_and_store(key)
        with self._lock:
            self.stats.misses += 1
        return chunk

    def prefetch(self, chunk_idx: tuple[int, ...]) -> None:
        """Schedule an asynchronous fetch of ``chunk_idx`` if it is not already
        cached or in flight. A no-op when the prefetch pool is disabled.
        """

        if self._executor is None:
            return
        key = tuple(int(v) for v in chunk_idx)
        with self._lock:
            if key in self._cache or key in self._pending:
                return
            future = self._executor.submit(self._fetch_and_store, key)
            self._pending[key] = future
            self.stats.prefetch_scheduled += 1
        future.add_done_callback(lambda _f, c=key: self._mark_done(c))

    def contains(self, chunk_idx: tuple[int, ...]) -> bool:
        with self._lock:
            return tuple(int(v) for v in chunk_idx) in self._cache

    def shutdown(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)

    # --- internals ------------------------------------------------------ #

    def _mark_done(self, chunk_idx: tuple[int, ...]) -> None:
        with self._lock:
            self._pending.pop(chunk_idx, None)

    def _fetch_and_store(self, chunk_idx: tuple[int, ...]) -> np.ndarray:
        slices = tuple(
            slice(idx * size, min((idx + 1) * size, axis_len))
            for idx, size, axis_len in zip(chunk_idx, self.chunk_shape, self.volume_shape, strict=True)
        )
        raw = self.source[slices]
        chunk = np.asarray(raw)
        if chunk.shape != self.chunk_shape:
            # Edge chunk: pad to native size so window-slicing math stays uniform.
            padded = np.zeros(self.chunk_shape, dtype=chunk.dtype)
            padded[tuple(slice(0, dim) for dim in chunk.shape)] = chunk
            chunk = padded
        with self._lock:
            self._cache[chunk_idx] = chunk
            self.stats.bytes_read += int(chunk.nbytes)
        return chunk


def open_streaming_volume(
    url: str,
    *,
    storage_options: dict[str, Any] | None = None,
    maxsize: int = 32,
    num_prefetch_workers: int = 2,
) -> ChunkLRUCache:
    """Open a (typically S3-backed) zarr at ``url`` and wrap it in a
    :class:`ChunkLRUCache`.

    Honours the same ``storage_options`` as
    :func:`vesuvius.data.utils.open_zarr`. When the zarr stores its data
    under a ``"0"`` group (the standard OME-NGFF layout), that level is
    selected automatically — matching what the training dataset does.
    """

    arr = open_zarr(str(url), mode="r", storage_options=storage_options or {})
    if hasattr(arr, "keys") and "0" in arr:
        arr = arr["0"]
    return ChunkLRUCache(arr, maxsize=maxsize, num_prefetch_workers=num_prefetch_workers)


__all__ = ["CacheStats", "ChunkLRUCache", "open_streaming_volume"]
