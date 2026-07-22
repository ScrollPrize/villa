import numpy as np
import pytest
import zarr

from vesuvius.data.utils import LRUCacheStore, open_zarr


@pytest.fixture
def local_array(tmp_path):
    path = str(tmp_path / "vol.zarr")
    a = zarr.create_array(store=path, shape=(64, 64, 64), chunks=(16, 16, 16), dtype="uint8")
    a[:] = np.random.default_rng(0).integers(0, 255, (64, 64, 64), dtype="uint8")
    return path


@pytest.mark.unit
def test_cached_reads_are_byte_identical(local_array):
    plain = zarr.open(local_array, mode="r")
    cached = open_zarr(local_array, mode="r", cache=True, cache_size_mb=64)
    assert np.array_equal(plain[:], cached[:])
    # second full read is served from the cache and still identical
    assert np.array_equal(plain[:], cached[:])


@pytest.mark.unit
def test_cache_populates_and_respects_size_bound(local_array):
    cached = open_zarr(local_array, mode="r", cache=True, cache_size_mb=64)
    _ = cached[0:32, 0:32, 0:32]
    store = cached.store
    assert isinstance(store, LRUCacheStore)
    assert len(store._lru) > 0
    assert store._current_size <= 64 * 2**20

    # a zero-size cache retains nothing (evicts everything on entry)
    tiny = open_zarr(local_array, mode="r", cache=True, cache_size_mb=0)
    _ = tiny[:]
    assert len(tiny.store._lru) == 0
    assert tiny.store._current_size == 0


@pytest.mark.unit
def test_cache_default_off_leaves_store_unwrapped(local_array):
    plain = open_zarr(local_array, mode="r", cache=False)
    assert not isinstance(plain.store, LRUCacheStore)


@pytest.mark.unit
def test_concurrent_identical_key_misses_keep_accounting_exact(local_array):
    import asyncio

    from zarr.core.buffer import default_buffer_prototype

    cached = open_zarr(local_array, mode="r", cache=True, cache_size_mb=64)
    store = cached.store

    async def hammer():
        await asyncio.gather(
            *[store.get("c/0/0/0", default_buffer_prototype()) for _ in range(20)]
        )

    asyncio.run(hammer())
    # zarr fetches chunks concurrently; a double-counted miss would leave
    # _current_size above the bytes actually held.
    assert store._current_size == sum(len(v) for v in store._lru.values())


@pytest.mark.unit
def test_lru_evicts_oldest_first(local_array):
    cached = open_zarr(local_array, mode="r", cache=True, cache_size_mb=64)
    store = cached.store
    _ = cached[0:16, 0:16, 0:16]
    first_keys = set(store._lru)
    _ = cached[48:64, 48:64, 48:64]
    # shrink the bound so the next insert must evict the oldest entries
    store._max_size = store._current_size
    _ = cached[16:32, 16:32, 16:32]
    assert len(store._lru) > 0
    assert store._current_size <= store._max_size
    assert not first_keys <= set(store._lru)
