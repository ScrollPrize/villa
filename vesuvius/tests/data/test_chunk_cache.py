import numpy as np
import pytest
import zarr

from vesuvius.data.utils import open_zarr

_ZARR_V3 = int(zarr.__version__.split('.', 1)[0]) >= 3
requires_zarr_v3 = pytest.mark.skipif(
    not _ZARR_V3, reason="open_zarr(cache=True) requires zarr>=3 (zarr.experimental.cache_store.CacheStore)"
)


@pytest.fixture
def local_array(tmp_path):
    path = str(tmp_path / "vol.zarr")
    if _ZARR_V3:
        a = zarr.create_array(store=path, shape=(64, 64, 64), chunks=(16, 16, 16), dtype="uint8")
    else:
        a = zarr.open(path, mode="w", shape=(64, 64, 64), chunks=(16, 16, 16), dtype="uint8")
    a[:] = np.random.default_rng(0).integers(0, 255, (64, 64, 64), dtype="uint8")
    return path


@requires_zarr_v3
@pytest.mark.unit
def test_cached_reads_are_byte_identical(local_array):
    plain = zarr.open(local_array, mode="r")
    cached = open_zarr(local_array, mode="r", cache=True, cache_size_mb=64)
    assert np.array_equal(plain[:], cached[:])
    # second full read is served from the cache and still identical
    assert np.array_equal(plain[:], cached[:])


@requires_zarr_v3
@pytest.mark.unit
def test_cache_populates_and_respects_size_bound(local_array):
    from zarr.experimental.cache_store import CacheStore

    cached = open_zarr(local_array, mode="r", cache=True, cache_size_mb=64)
    _ = cached[0:32, 0:32, 0:32]
    store = cached.store
    assert isinstance(store, CacheStore)
    info = store.cache_info()
    assert info["cached_keys"] > 0
    assert info["current_size"] <= 64 * 2**20

    # a zero-size cache retains nothing (evicts everything on entry)
    tiny = open_zarr(local_array, mode="r", cache=True, cache_size_mb=0)
    _ = tiny[:]
    tiny_info = tiny.store.cache_info()
    assert tiny_info["cached_keys"] == 0
    assert tiny_info["current_size"] == 0


@pytest.mark.unit
def test_cache_default_off_leaves_store_unwrapped(local_array):
    plain = open_zarr(local_array, mode="r", cache=False)
    if _ZARR_V3:
        from zarr.experimental.cache_store import CacheStore
        assert not isinstance(plain.store, CacheStore)


@pytest.mark.skipif(_ZARR_V3, reason="only exercises the zarr 2.x guard rail")
@pytest.mark.unit
def test_cache_true_raises_under_zarr2(local_array):
    with pytest.raises(NotImplementedError):
        open_zarr(local_array, mode="r", cache=True)


@requires_zarr_v3
@pytest.mark.unit
def test_repeat_reads_hit_cache(local_array):
    cached = open_zarr(local_array, mode="r", cache=True, cache_size_mb=64)
    _ = cached[0:16, 0:16, 0:16]
    stats_after_first = cached.store.cache_stats()
    _ = cached[0:16, 0:16, 0:16]
    stats_after_second = cached.store.cache_stats()
    assert stats_after_second["hits"] > stats_after_first["hits"]


@requires_zarr_v3
@pytest.mark.unit
def test_lru_evicts_oldest_first(local_array):
    cached = open_zarr(local_array, mode="r", cache=True, cache_size_mb=64)
    store = cached.store
    _ = cached[0:16, 0:16, 0:16]
    first_keys = set(store._state.cache_order)
    _ = cached[48:64, 48:64, 48:64]
    # shrink the bound so the next insert must evict the oldest entries
    store.max_size = store._state.current_size
    _ = cached[16:32, 16:32, 16:32]
    info = store.cache_info()
    assert info["cached_keys"] > 0
    assert info["current_size"] <= store.max_size
    assert not first_keys <= set(store._state.cache_order)
    assert store.cache_stats()["evictions"] > 0
