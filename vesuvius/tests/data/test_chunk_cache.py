import numpy as np
import pytest
import zarr
import zarr.storage

from vesuvius.data.utils import open_zarr

_ZARR_V3 = int(zarr.__version__.split('.', 1)[0]) >= 3
requires_zarr_v3 = pytest.mark.skipif(
    not _ZARR_V3, reason="open_zarr(cache=True) requires zarr>=3 (zarr.experimental.cache_store.CacheStore)"
)

# Never resolves, so a test that reaches the network fails loudly instead of
# quietly depending on it. The store factory is always patched to a local store.
REMOTE_URL = "https://example.invalid/vol.zarr"

_METADATA_NAMES = ("zarr.json", ".zarray", ".zgroup", ".zattrs", ".zmetadata")


def _is_metadata(key):
    return key.rsplit("/", 1)[-1] in _METADATA_NAMES


@pytest.fixture
def local_array(tmp_path):
    path = str(tmp_path / "vol.zarr")
    if _ZARR_V3:
        a = zarr.create_array(store=path, shape=(64, 64, 64), chunks=(16, 16, 16), dtype="uint8")
    else:
        a = zarr.open(path, mode="w", shape=(64, 64, 64), chunks=(16, 16, 16), dtype="uint8")
    a[:] = np.random.default_rng(0).integers(0, 255, (64, 64, 64), dtype="uint8")
    return path


def _patch_remote_store(monkeypatch, source_path, chunks_forbidden=False):
    """Make open_zarr's remote store factory hand back a local store.

    Lets the disk-cache branch be exercised against REMOTE_URL without any
    network access. With ``chunks_forbidden`` the stand-in serves store
    metadata but raises on any chunk fetch, so a read that survives proves the
    chunk came off the disk cache rather than the "remote".
    """
    if _ZARR_V3:
        from zarr.storage import FsspecStore, LocalStore

        class _NoChunkStore(LocalStore):
            async def get(self, key, prototype, byte_range=None):
                if chunks_forbidden and not _is_metadata(key):
                    raise AssertionError(f"disk cache missed: refetched {key!r}")
                return await super().get(key, prototype, byte_range)

        def fake_from_url(url, storage_options=None, read_only=True):
            return _NoChunkStore(source_path, read_only=read_only)

        monkeypatch.setattr(FsspecStore, "from_url", staticmethod(fake_from_url))
    else:
        # Subclass rather than replace: zarr 2's normalize_store calls
        # FSStore._fsspec_installed() on the module global, so a bare stub
        # function breaks zarr.open before the cache is ever reached.
        class _NoChunkStore(zarr.storage.FSStore):
            def __init__(self, url, mode="r", **storage_options):
                super().__init__(source_path, mode=mode, **storage_options)

            def __getitem__(self, key):
                if chunks_forbidden and not _is_metadata(key):
                    raise AssertionError(f"disk cache missed: refetched {key!r}")
                return super().__getitem__(key)

        monkeypatch.setattr(zarr.storage, "FSStore", _NoChunkStore)


def _cached_files(cache_dir):
    return [p for p in cache_dir.rglob("*") if p.is_file()] if cache_dir.exists() else []


@pytest.mark.unit
def test_cache_dir_wraps_remote_reads(monkeypatch, tmp_path, local_array):
    expected = zarr.open(local_array, mode="r")[:]
    cache_dir = tmp_path / "chunkcache"

    _patch_remote_store(monkeypatch, local_array)
    arr = open_zarr(REMOTE_URL, mode="r", cache_dir=cache_dir)
    assert np.array_equal(arr[:], expected)

    # Chunks landed on disk, namespaced by the remote URL.
    assert (cache_dir / "https" / "example.invalid" / "vol.zarr").is_dir()
    assert _cached_files(cache_dir)

    # Re-open against a store that refuses to serve chunks: the reads still
    # succeed, so they came from the cache directory.
    _patch_remote_store(monkeypatch, local_array, chunks_forbidden=True)
    warm = open_zarr(REMOTE_URL, mode="r", cache_dir=cache_dir)
    assert np.array_equal(warm[:], expected)


@pytest.mark.unit
def test_cache_dir_wins_over_memory_cache(monkeypatch, tmp_path, local_array):
    # Documented precedence: cache_dir is checked first, so passing both does
    # not fall through to the in-memory branch (which would raise on zarr 2).
    cache_dir = tmp_path / "chunkcache"

    _patch_remote_store(monkeypatch, local_array)
    arr = open_zarr(REMOTE_URL, mode="r", cache_dir=cache_dir, cache=True)

    assert _cached_files(cache_dir)
    if _ZARR_V3:
        from zarr.experimental.cache_store import CacheStore
        assert not isinstance(arr.store, CacheStore)


@pytest.mark.unit
def test_cache_dir_ignored_for_local_paths(tmp_path, local_array):
    expected = zarr.open(local_array, mode="r")[:]
    cache_dir = tmp_path / "chunkcache"

    arr = open_zarr(local_array, mode="r", cache_dir=cache_dir)

    assert np.array_equal(arr[:], expected)
    assert _cached_files(cache_dir) == []


@pytest.mark.skipif(_ZARR_V3, reason="only exercises the zarr 2.x disk-cache leg")
@pytest.mark.unit
def test_cache_dir_works_under_zarr2(monkeypatch, tmp_path, local_array):
    # Inverse of test_cache_true_raises_under_zarr2: the disk cache is not
    # gated on zarr 3, so cache_dir must work here rather than raise.
    expected = zarr.open(local_array, mode="r")[:]
    cache_dir = tmp_path / "chunkcache"

    _patch_remote_store(monkeypatch, local_array)
    arr = open_zarr(REMOTE_URL, mode="r", cache_dir=cache_dir)

    assert np.array_equal(arr[:], expected)
    assert _cached_files(cache_dir)


@pytest.mark.unit
def test_cache_dir_max_gb_forwarded(monkeypatch, tmp_path, local_array):
    from vesuvius.data import chunk_cache

    captured = {}
    real_store = chunk_cache.DiskCacheStore

    def recording_store(remote, **kwargs):
        captured.update(kwargs)
        return real_store(remote, **kwargs)

    monkeypatch.setattr(chunk_cache, "DiskCacheStore", recording_store)
    _patch_remote_store(monkeypatch, local_array)

    open_zarr(REMOTE_URL, mode="r", cache_dir=tmp_path / "chunkcache", cache_max_gb=0.5)

    assert captured["max_bytes"] == int(0.5 * 2**30)
    assert captured["cache_dir"] == str(tmp_path / "chunkcache")
    assert captured["url"] == REMOTE_URL

    # Unset means unbounded.
    captured.clear()
    open_zarr(REMOTE_URL, mode="r", cache_dir=tmp_path / "chunkcache")
    assert captured["max_bytes"] is None


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
