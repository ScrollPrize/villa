"""Tests for the sliding 128^3 window reader."""

from __future__ import annotations

from typing import Any

import numpy as np

from vesuvius.neural_tracing.autoreg_fiber.streaming.chunk_cache import ChunkLRUCache
from vesuvius.neural_tracing.autoreg_fiber.streaming.window import WindowedVolumeReader


class _ArrayBackedZarrLike:
    """Adapts a numpy ndarray to the minimum ``ChunkLRUCache`` interface."""

    def __init__(self, data: np.ndarray, chunk_shape: tuple[int, int, int]):
        self._data = data
        self.shape = tuple(data.shape)
        self.chunks = tuple(int(v) for v in chunk_shape)
        self.dtype = data.dtype
        self.reads: list[Any] = []

    def __getitem__(self, item):
        self.reads.append(item)
        return self._data[item]


def _make_reader(
    *,
    volume_shape: tuple[int, int, int] = (64, 64, 64),
    chunk_shape: tuple[int, int, int] = (16, 16, 16),
    crop_size: tuple[int, int, int] = (32, 32, 32),
    cache_size: int = 32,
    num_prefetch_workers: int = 0,
    reanchor_margin: int = 4,
) -> tuple[WindowedVolumeReader, np.ndarray, _ArrayBackedZarrLike]:
    rng = np.random.default_rng(0)
    data = rng.uniform(0.0, 1.0, size=volume_shape).astype(np.float32)
    source = _ArrayBackedZarrLike(data, chunk_shape=chunk_shape)
    cache = ChunkLRUCache(source, maxsize=cache_size, num_prefetch_workers=num_prefetch_workers)
    reader = WindowedVolumeReader(cache, crop_size=crop_size, reanchor_margin=reanchor_margin)
    return reader, data, source


def test_fetch_crop_returns_centered_window() -> None:
    reader, data, _ = _make_reader()
    reader.anchor_on((32, 32, 32))
    crop = reader.fetch_crop()
    assert crop.shape == (32, 32, 32)
    # min_corner = floor(32 - 16) = 16; the crop should equal data[16:48, 16:48, 16:48].
    assert np.allclose(reader.min_corner, np.array([16, 16, 16]))
    np.testing.assert_array_equal(crop, data[16:48, 16:48, 16:48])


def test_anchor_clamps_to_volume_bounds() -> None:
    reader, data, _ = _make_reader()
    # Anchor near the high corner — should clamp so the window stays inside.
    reader.anchor_on((100, 100, 100))
    assert (reader.min_corner == np.array([32, 32, 32])).all()
    crop = reader.fetch_crop()
    np.testing.assert_array_equal(crop, data[32:64, 32:64, 32:64])

    # Anchor near the low corner.
    reader.anchor_on((-10, -10, -10))
    assert (reader.min_corner == np.array([0, 0, 0])).all()
    crop = reader.fetch_crop()
    np.testing.assert_array_equal(crop, data[0:32, 0:32, 0:32])


def test_needs_reanchor_respects_margin() -> None:
    reader, _data, _ = _make_reader(crop_size=(32, 32, 32), reanchor_margin=4)
    assert reader.needs_reanchor((2, 16, 16))   # near low z face
    assert reader.needs_reanchor((30, 16, 16))  # near high z face
    assert reader.needs_reanchor((16, 2, 16))
    assert reader.needs_reanchor((16, 16, 30))
    assert not reader.needs_reanchor((16, 16, 16))  # comfortably interior


def test_world_local_roundtrip() -> None:
    reader, _data, _ = _make_reader()
    reader.anchor_on((40, 40, 40))
    world = np.array([45.5, 30.1, 28.0], dtype=np.float32)
    local = reader.world_to_local(world)
    back = reader.local_to_world(local)
    np.testing.assert_allclose(back, world, rtol=0, atol=1e-5)


def test_sliding_window_reuses_chunks_via_cache() -> None:
    # Crop is 32^3, chunks are 16^3. Shifting the anchor by 8 in z keeps the
    # window inside the same set of z-chunks (no new chunk rows along z), so
    # the LRU should answer most chunks from cache.
    reader, _data, source = _make_reader(
        volume_shape=(64, 64, 64),
        chunk_shape=(16, 16, 16),
        crop_size=(32, 32, 32),
        cache_size=64,
        num_prefetch_workers=0,
        reanchor_margin=4,
    )
    reader.anchor_on((16 + 16, 16 + 16, 16 + 16))  # min_corner = (16,16,16); chunks (1..2,1..2,1..2) -> 8 reads
    reader.fetch_crop()
    initial_reads = len(source.reads)
    # Shift anchor by +8 along z: window covers chunks (1..3, 1..2, 1..2) -> 12 chunks total,
    # 4 of which are new along z; the other 8 are already in cache.
    reader.anchor_on((16 + 16 + 8, 16 + 16, 16 + 16))
    reader.fetch_crop()
    delta_reads = len(source.reads) - initial_reads
    assert delta_reads == 4, f"expected 4 new chunk reads after small slide, saw {delta_reads}"


def test_prefetch_warms_cache_for_next_anchor() -> None:
    reader, _data, source = _make_reader(num_prefetch_workers=2, cache_size=32)
    reader.anchor_on((16, 16, 16))
    reader.fetch_crop()
    initial_reads = len(source.reads)
    # Prefetch chunks for a future anchor far enough away that nothing is reused.
    reader.prefetch_anchor((48, 48, 48))
    pre_scheduled = reader.cache.stats.prefetch_scheduled
    # Block until everything resolves by issuing the actual fetch.
    reader.anchor_on((48, 48, 48))
    reader.fetch_crop()
    # Prefetch must have scheduled something for the future anchor's chunks.
    assert pre_scheduled > 0
    # Every chunk needed by the second window was served by the cache (either
    # already there because the future resolved before fetch_crop, or awaited
    # in-flight). The only "miss" reads would be the initial window's chunks.
    assert reader.cache.stats.misses == initial_reads
    assert reader.cache.stats.hits + reader.cache.stats.prefetch_hits >= pre_scheduled


def test_in_volume_bounds_handles_edges() -> None:
    reader, _data, _ = _make_reader(volume_shape=(64, 64, 64))
    assert reader.in_volume_bounds((0, 0, 0))
    assert reader.in_volume_bounds((63, 63, 63))
    assert not reader.in_volume_bounds((-1, 0, 0))
    assert not reader.in_volume_bounds((0, 0, 64))
