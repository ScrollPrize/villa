"""Tests for the chunk-level LRU + async prefetch used by streaming inference."""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np

from vesuvius.neural_tracing.autoreg_fiber.streaming.chunk_cache import ChunkLRUCache


class _CountingZarrLike:
    """A duck-typed zarr-like that materialises chunks lazily and counts reads.

    Each chunk (z,y,x) contains the constant value ``z*1e6 + y*1e3 + x`` so
    that callers can assert which chunk they got.
    """

    def __init__(
        self,
        volume_shape: tuple[int, int, int] = (32, 32, 32),
        chunk_shape: tuple[int, int, int] = (16, 16, 16),
        *,
        read_latency_s: float = 0.0,
    ) -> None:
        self.shape = tuple(int(v) for v in volume_shape)
        self.chunks = tuple(int(v) for v in chunk_shape)
        self.dtype = np.float32
        self.reads: list[Any] = []
        self.read_latency_s = float(read_latency_s)
        self._read_lock = threading.Lock()

    def __getitem__(self, item):
        if self.read_latency_s > 0:
            time.sleep(self.read_latency_s)
        with self._read_lock:
            self.reads.append(item)
        slices = item if isinstance(item, tuple) else (item,)
        out = np.zeros(self.shape, dtype=self.dtype)
        nz = -(-self.shape[0] // self.chunks[0])
        ny = -(-self.shape[1] // self.chunks[1])
        nx = -(-self.shape[2] // self.chunks[2])
        for zi in range(nz):
            for yi in range(ny):
                for xi in range(nx):
                    z0, y0, x0 = zi * self.chunks[0], yi * self.chunks[1], xi * self.chunks[2]
                    z1 = min(z0 + self.chunks[0], self.shape[0])
                    y1 = min(y0 + self.chunks[1], self.shape[1])
                    x1 = min(x0 + self.chunks[2], self.shape[2])
                    out[z0:z1, y0:y1, x0:x1] = float(zi * 1_000_000 + yi * 1_000 + xi)
        return out[slices]


def test_get_caches_and_avoids_duplicate_reads() -> None:
    src = _CountingZarrLike()
    cache = ChunkLRUCache(src, maxsize=8, num_prefetch_workers=0)

    first = cache.get((0, 0, 0))
    second = cache.get((0, 0, 0))
    assert first is second  # same array returned (no defensive copy)
    assert len(src.reads) == 1
    stats = cache.stats.as_dict()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["prefetch_hits"] == 0
    cache.shutdown()


def test_lru_evicts_oldest_chunk_when_full() -> None:
    src = _CountingZarrLike()
    cache = ChunkLRUCache(src, maxsize=2, num_prefetch_workers=0)
    cache.get((0, 0, 0))
    cache.get((0, 0, 1))
    cache.get((0, 1, 0))  # evicts (0,0,0)
    assert not cache.contains((0, 0, 0))
    assert cache.contains((0, 0, 1))
    assert cache.contains((0, 1, 0))
    # touching (0,0,0) again must re-read from source
    pre_reads = len(src.reads)
    cache.get((0, 0, 0))
    assert len(src.reads) == pre_reads + 1
    cache.shutdown()


def test_chunk_values_match_source_contents() -> None:
    src = _CountingZarrLike()
    cache = ChunkLRUCache(src, maxsize=8, num_prefetch_workers=0)
    chunk = cache.get((1, 0, 1))
    # The mock fills each chunk uniformly with z*1e6 + y*1e3 + x.
    assert np.allclose(chunk, 1_000_001.0)
    cache.shutdown()


def test_prefetch_marks_chunk_as_pending_and_resolves_to_cached() -> None:
    src = _CountingZarrLike(read_latency_s=0.05)
    cache = ChunkLRUCache(src, maxsize=8, num_prefetch_workers=2)
    cache.prefetch((1, 1, 1))
    # The prefetched chunk should be cached after the future resolves.
    chunk = cache.get((1, 1, 1))
    assert np.allclose(chunk, 1_001_001.0)
    stats = cache.stats.as_dict()
    assert stats["prefetch_scheduled"] == 1
    # The synchronous get either hit the cache (if the future finished first)
    # or awaited the in-flight future — either is correct.
    assert stats["hits"] + stats["prefetch_hits"] == 1
    assert stats["misses"] == 0
    cache.shutdown()


def test_prefetch_is_idempotent_for_in_flight_chunks() -> None:
    src = _CountingZarrLike(read_latency_s=0.05)
    cache = ChunkLRUCache(src, maxsize=8, num_prefetch_workers=2)
    cache.prefetch((0, 1, 0))
    cache.prefetch((0, 1, 0))  # second call must not schedule another read
    cache.prefetch((0, 1, 0))
    cache.get((0, 1, 0))  # await
    assert cache.stats.prefetch_scheduled == 1
    cache.shutdown()


def test_edge_chunks_are_padded_to_native_shape() -> None:
    # Volume 20x20x20, chunks 16x16x16 — the (1,0,0)/(0,1,0)/(0,0,1) chunks are
    # truncated to 4 voxels along their respective axis. Cache must pad them so
    # downstream slicing math stays uniform.
    src = _CountingZarrLike(volume_shape=(20, 16, 16), chunk_shape=(16, 16, 16))
    cache = ChunkLRUCache(src, maxsize=4, num_prefetch_workers=0)
    chunk = cache.get((1, 0, 0))
    assert chunk.shape == (16, 16, 16)
    # First 4 voxels along z contain the real data; the rest should be zero-padded.
    assert chunk[:4].mean() != 0.0
    assert chunk[4:].sum() == 0.0
    cache.shutdown()
