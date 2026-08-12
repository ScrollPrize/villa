import asyncio
import os
import time
from collections.abc import MutableMapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import zarr
import zarr.storage

from vesuvius.data import chunk_cache
from vesuvius.data.chunk_cache import (
    _CACHE_STAMP_NAME,
    _NEGATIVE_MARKER_SUFFIX,
    DiskCacheStore,
    DiskCacheStoreV2,
    OfflineCacheMiss,
)


ZARR_V3 = int(zarr.__version__.split(".", 1)[0]) >= 3

if ZARR_V3:
    from zarr.abc.store import OffsetByteRequest, RangeByteRequest, SuffixByteRequest
else:
    # Only the zarr 3 store takes byte-range requests.
    OffsetByteRequest = RangeByteRequest = SuffixByteRequest = ()

requires_zarr_v3 = pytest.mark.skipif(not ZARR_V3, reason="zarr 3 store wrapper only")
requires_zarr_v2 = pytest.mark.skipif(ZARR_V3, reason="zarr 2 store paths only")

_CHUNK_KEY = "c/0/0/0" if ZARR_V3 else "0.0.0"
_METADATA_KEY = "zarr.json" if ZARR_V3 else ".zarray"


class _ExplodingMapping(MutableMapping):
    """Zarr 2 remote that fails any access, to prove reads came from disk."""

    def __getitem__(self, key):
        raise AssertionError(f"remote was read for {key!r}")

    def __contains__(self, key):
        raise AssertionError(f"remote was probed for {key!r}")

    def __setitem__(self, key, value):
        raise AssertionError("remote was written")

    def __delitem__(self, key):
        raise AssertionError("remote was deleted from")

    def __iter__(self):
        return iter(())

    def __len__(self):
        return 0


class _CountingMapping(dict):
    """Zarr 2 remote that records every key read."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reads = []

    def __getitem__(self, key):
        self.reads.append(key)
        return super().__getitem__(key)


class _NoCrawlMapping(MutableMapping):
    """Zarr 2 remote that fails any access which would list the whole store."""

    def __init__(self, inner):
        self._inner = inner

    def __getitem__(self, key):
        return self._inner[key]

    def __contains__(self, key):
        return key in self._inner

    def __setitem__(self, key, value):
        raise AssertionError("remote was written")

    def __delitem__(self, key):
        raise AssertionError("remote was deleted from")

    def __iter__(self):
        raise AssertionError("remote keys were iterated")

    def __len__(self):
        raise AssertionError("remote keys were counted")

    def keys(self):
        raise AssertionError("remote keys were listed")

    def find(self, *args, **kwargs):
        raise AssertionError("remote was recursively listed")

    def listdir(self, path=""):
        return self._inner.listdir(path)

    def getsize(self, path=None):
        return self._inner.getsize(path)


def _marker_path(store, key: str) -> Path:
    return Path(store._cache_dir) / (key + _NEGATIVE_MARKER_SUFFIX)


def _cached_path(store, key: str) -> Path:
    return Path(store._cache_dir) / key


def _chunk_key(index: int) -> str:
    return f"c/0/0/{index}" if ZARR_V3 else f"0.0.{index}"


def _remote_with(payloads: dict):
    """In-memory remote holding `payloads`, for whichever zarr is installed."""
    if not ZARR_V3:
        return dict(payloads)

    from zarr.core.buffer.core import default_buffer_prototype
    from zarr.storage import MemoryStore

    async def build():
        prototype = default_buffer_prototype()
        remote = MemoryStore()
        for key, value in payloads.items():
            await remote.set(key, prototype.buffer.from_bytes(value))
        return remote.with_read_only(True)

    return asyncio.run(build())


def _read(store, key: str, byte_range=None):
    """Read one key through the store, returning its bytes or None."""
    if not ZARR_V3:
        return store[key]

    from zarr.core.buffer.core import default_buffer_prototype

    async def exercise():
        result = await store.get(key, default_buffer_prototype(), byte_range)
        return None if result is None else result.to_bytes()

    return asyncio.run(exercise())


def _write_marker(store, key: str) -> None:
    """Leave a negative marker the way a pre-exemption cache would have."""
    marker = _marker_path(store, key)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_bytes(b"")


def _probe(store, key: str) -> bool:
    """Existence-check one key through the store, without reading it."""
    if not ZARR_V3:
        return key in store

    async def exercise():
        return await store.exists(key)

    return asyncio.run(exercise())


def _cached_bytes(root) -> int:
    """Total size of the non-temp files under a cache root."""
    total = 0
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if ".tmp." in name:
                continue
            total += os.stat(os.path.join(dirpath, name)).st_size
    return total


def test_persists_across_instances(tmp_path):
    if not ZARR_V3:
        remote = {_CHUNK_KEY: b"payload"}
        first = DiskCacheStore(remote, str(tmp_path), url="memory://dataset")
        assert first[_CHUNK_KEY] == b"payload"

        second = DiskCacheStore(
            _ExplodingMapping(), str(tmp_path), url="memory://dataset"
        )
        assert second[_CHUNK_KEY] == b"payload"
        return

    from zarr.core.buffer.core import default_buffer_prototype
    from zarr.storage import MemoryStore

    class _ExplodingStore(MemoryStore):
        async def get(self, key, prototype, byte_range=None):
            raise AssertionError(f"remote was read for {key!r}")

    async def exercise():
        prototype = default_buffer_prototype()
        remote = MemoryStore()
        await remote.set(_CHUNK_KEY, prototype.buffer.from_bytes(b"payload"))

        first = DiskCacheStore(
            remote.with_read_only(True), str(tmp_path), url="memory://dataset"
        )
        written = await first.get(_CHUNK_KEY, prototype)
        assert written is not None
        assert written.to_bytes() == b"payload"

        second = DiskCacheStore(
            _ExplodingStore(read_only=True), str(tmp_path), url="memory://dataset"
        )
        cached = await second.get(_CHUNK_KEY, prototype)
        assert cached is not None
        assert cached.to_bytes() == b"payload"

    asyncio.run(exercise())


def test_metadata_keys_not_negative_cached(tmp_path):
    if not ZARR_V3:
        remote = _CountingMapping()
        cache = DiskCacheStore(remote, str(tmp_path), url="memory://dataset")

        with pytest.raises(KeyError):
            cache[_METADATA_KEY]
        assert not _marker_path(cache, _METADATA_KEY).exists()
        with pytest.raises(KeyError):
            cache[_METADATA_KEY]
        assert remote.reads.count(_METADATA_KEY) == 2

        with pytest.raises(KeyError):
            cache[_CHUNK_KEY]
        assert _marker_path(cache, _CHUNK_KEY).exists()
        with pytest.raises(KeyError):
            cache[_CHUNK_KEY]
        assert remote.reads.count(_CHUNK_KEY) == 1
        return

    from zarr.core.buffer.core import default_buffer_prototype
    from zarr.storage import MemoryStore

    class _CountingStore(MemoryStore):
        reads: list = []

        async def get(self, key, prototype, byte_range=None):
            self.reads.append(key)
            return await super().get(key, prototype, byte_range)

    async def exercise():
        prototype = default_buffer_prototype()
        remote = _CountingStore(read_only=True)
        remote.reads = []
        cache = DiskCacheStore(remote, str(tmp_path), url="memory://dataset")

        assert await cache.get(_METADATA_KEY, prototype) is None
        assert not _marker_path(cache, _METADATA_KEY).exists()
        assert await cache.get(_METADATA_KEY, prototype) is None
        assert remote.reads.count(_METADATA_KEY) == 2

        assert await cache.get(_CHUNK_KEY, prototype) is None
        assert _marker_path(cache, _CHUNK_KEY).exists()
        assert await cache.get(_CHUNK_KEY, prototype) is None
        assert remote.reads.count(_CHUNK_KEY) == 1

    asyncio.run(exercise())


def test_stale_metadata_marker_is_ignored(tmp_path):
    # Markers left by an older cache must not hide metadata that exists now.
    if not ZARR_V3:
        remote = {_METADATA_KEY: b"{}"}
        cache = DiskCacheStore(remote, str(tmp_path), url="memory://dataset")
        marker = _marker_path(cache, _METADATA_KEY)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_bytes(b"")

        assert cache[_METADATA_KEY] == b"{}"
        assert _METADATA_KEY in cache
        return

    from zarr.core.buffer.core import default_buffer_prototype
    from zarr.storage import MemoryStore

    async def exercise():
        prototype = default_buffer_prototype()
        remote = MemoryStore()
        await remote.set(_METADATA_KEY, prototype.buffer.from_bytes(b"{}"))
        cache = DiskCacheStore(
            remote.with_read_only(True), str(tmp_path), url="memory://dataset"
        )
        marker = _marker_path(cache, _METADATA_KEY)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_bytes(b"")

        found = await cache.get(_METADATA_KEY, prototype)
        assert found is not None
        assert found.to_bytes() == b"{}"
        assert await cache.exists(_METADATA_KEY)

    asyncio.run(exercise())


def test_eviction_bounds_cache_size(tmp_path):
    payload = b"x" * 100
    keys = [_chunk_key(i) for i in range(6)]
    store = DiskCacheStore(
        _remote_with({key: payload for key in keys}),
        str(tmp_path),
        url="memory://dataset",
        max_bytes=300,
    )

    stamp = time.time() - 1000.0
    for index, key in enumerate(keys):
        assert _read(store, key) == payload
        if index < len(keys) - 1:
            # Pin the eviction order; the final write keeps its real (newest) mtime.
            os.utime(_cached_path(store, key), (stamp + index, stamp + index))

    assert _cached_bytes(tmp_path) <= 300
    assert _cached_path(store, keys[-1]).exists()
    assert not _cached_path(store, keys[0]).exists()


def test_hit_refreshes_recency(tmp_path):
    payload = b"y" * 100
    keys = [_chunk_key(i) for i in range(3)]
    store = DiskCacheStore(
        _remote_with({key: payload for key in keys}),
        str(tmp_path),
        url="memory://dataset",
        max_bytes=250,
    )

    stamp = time.time() - 1000.0
    assert _read(store, keys[0]) == payload
    os.utime(_cached_path(store, keys[0]), (stamp, stamp))
    assert _read(store, keys[1]) == payload
    os.utime(_cached_path(store, keys[1]), (stamp + 1, stamp + 1))

    # A disk hit on the older chunk makes it the most recently used one.
    assert _read(store, keys[0]) == payload

    # This write pushes the cache over the cap and evicts exactly one chunk.
    assert _read(store, keys[2]) == payload

    assert not _cached_path(store, keys[1]).exists()
    assert _cached_path(store, keys[0]).exists()
    assert _cached_path(store, keys[2]).exists()


def test_existence_probe_refreshes_recency(tmp_path):
    # An existence check served from disk is a cache hit too: it skips the
    # remote round-trip, so it must count as use.
    payload = b"e" * 100
    keys = [_chunk_key(i) for i in range(3)]
    store = DiskCacheStore(
        _remote_with({key: payload for key in keys}),
        str(tmp_path),
        url="memory://dataset",
        max_bytes=250,
    )

    stamp = time.time() - 1000.0
    assert _read(store, keys[0]) == payload
    os.utime(_cached_path(store, keys[0]), (stamp, stamp))
    assert _read(store, keys[1]) == payload
    os.utime(_cached_path(store, keys[1]), (stamp + 1, stamp + 1))

    assert _probe(store, keys[0]) is True

    # This write pushes the cache over the cap and evicts exactly one chunk.
    assert _read(store, keys[2]) == payload

    assert not _cached_path(store, keys[1]).exists()
    assert _cached_path(store, keys[0]).exists()


def test_v2_contains_refreshes_recency(tmp_path):
    payload = b"c" * 100
    keys = [f"0.0.{i}" for i in range(3)]
    store = DiskCacheStoreV2(
        {key: payload for key in keys},
        str(tmp_path),
        url="memory://dataset",
        max_bytes=250,
    )

    stamp = time.time() - 1000.0
    assert store[keys[0]] == payload
    os.utime(_cached_path(store, keys[0]), (stamp, stamp))
    assert store[keys[1]] == payload
    os.utime(_cached_path(store, keys[1]), (stamp + 1, stamp + 1))

    assert keys[0] in store

    assert store[keys[2]] == payload  # write, crossing the cap

    assert not _cached_path(store, keys[1]).exists()
    assert _cached_path(store, keys[0]).exists()


def test_eviction_skips_temp_files(tmp_path):
    payload = b"w" * 100
    keys = [_chunk_key(i) for i in range(3)]
    store = DiskCacheStore(
        _remote_with({key: payload for key in keys}),
        str(tmp_path),
        url="memory://dataset",
        max_bytes=300,
    )

    # A half-written file from another process must be neither counted nor removed.
    partial = tmp_path / "chunk.tmp.999.1"
    partial.write_bytes(b"p" * 1000)

    for key in keys:
        assert _read(store, key) == payload

    assert partial.exists()
    for key in keys:
        assert _cached_path(store, key).exists()


def test_v2_store_evicts_with_recency(tmp_path):
    # The zarr 2 mapping is a plain MutableMapping over a dict remote, so its
    # paths run under either zarr; assert them here rather than only on the
    # zarr 2 CI leg.
    payload = b"v" * 100
    keys = [f"0.0.{i}" for i in range(3)]
    store = DiskCacheStoreV2(
        {key: payload for key in keys},
        str(tmp_path),
        url="memory://dataset",
        max_bytes=250,
    )

    stamp = time.time() - 1000.0
    assert store[keys[0]] == payload
    os.utime(_cached_path(store, keys[0]), (stamp, stamp))
    assert store[keys[1]] == payload
    os.utime(_cached_path(store, keys[1]), (stamp + 1, stamp + 1))

    assert store[keys[0]] == payload  # hit, refreshing recency
    assert store[keys[2]] == payload  # write, crossing the cap

    assert _cached_bytes(tmp_path) <= 250
    assert not _cached_path(store, keys[1]).exists()
    assert _cached_path(store, keys[0]).exists()
    assert _cached_path(store, keys[2]).exists()


def test_unbounded_by_default(tmp_path):
    payload = b"z" * 100
    keys = [_chunk_key(i) for i in range(8)]
    store = DiskCacheStore(
        _remote_with({key: payload for key in keys}),
        str(tmp_path),
        url="memory://dataset",
    )

    for key in keys:
        assert _read(store, key) == payload

    assert _cached_bytes(tmp_path) == 800
    for key in keys:
        assert _cached_path(store, key).exists()


@pytest.mark.parametrize("bad", [0, -1, -2**30])
def test_rejects_non_positive_max_bytes(tmp_path, bad):
    # Every size check in the stores is `if self._max_bytes`, so a 0 cap would
    # be indistinguishable from None and silently give an unbounded cache.
    # Asserted on both classes, as with the other dual-class tests here.
    with pytest.raises(ValueError, match="positive byte count"):
        DiskCacheStore(
            _remote_with({}), str(tmp_path), url="memory://dataset", max_bytes=bad
        )
    with pytest.raises(ValueError, match="positive byte count"):
        DiskCacheStoreV2({}, str(tmp_path), url="memory://dataset", max_bytes=bad)


def test_with_read_only_preserves_max_bytes(tmp_path):
    # The cap is forwarded to the copy, and forwarding an already-validated
    # value must not trip the new check.
    for store in (
        DiskCacheStore(_remote_with({}), str(tmp_path), url="memory://dataset", max_bytes=250),
        DiskCacheStoreV2({}, str(tmp_path), url="memory://dataset", max_bytes=250),
    ):
        assert store.with_read_only(True)._max_bytes == 250


@requires_zarr_v3
def test_offline_metadata_marker_reads_as_missing(tmp_path):
    # Cache trees written before metadata was exempted from negative caching
    # still hold metadata markers. Offline there is nothing else to answer with,
    # so the marker must read as missing rather than raise OfflineCacheMiss.
    from zarr.core.buffer.core import default_buffer_prototype
    from zarr.storage import MemoryStore

    async def exercise():
        prototype = default_buffer_prototype()
        store = DiskCacheStore(
            MemoryStore(read_only=True),
            str(tmp_path),
            url="memory://dataset",
            offline=True,
        )
        _write_marker(store, ".zattrs")
        _write_marker(store, _CHUNK_KEY)

        assert await store.get(".zattrs", prototype) is None
        assert await store.exists(".zattrs") is False
        assert await store.get(_CHUNK_KEY, prototype) is None
        assert await store.exists(_CHUNK_KEY) is False

        with pytest.raises(OfflineCacheMiss):
            await store.get(".zgroup", prototype)

    asyncio.run(exercise())


def test_v2_offline_metadata_marker_reads_as_missing(tmp_path):
    store = DiskCacheStoreV2(
        _ExplodingMapping(), str(tmp_path), url="memory://dataset", offline=True
    )
    _write_marker(store, ".zattrs")
    _write_marker(store, "0.0.0")

    with pytest.raises(KeyError):
        store[".zattrs"]
    assert ".zattrs" not in store
    with pytest.raises(KeyError):
        store["0.0.0"]
    assert "0.0.0" not in store

    # An unmarked key offline is still a miss, not a silent None.
    with pytest.raises(OfflineCacheMiss):
        store[".zgroup"]


@requires_zarr_v3
def test_read_survives_cache_write_failure(tmp_path, monkeypatch):
    payload = b"n" * 64
    key = _chunk_key(0)
    store = DiskCacheStore(
        _remote_with({key: payload}), str(tmp_path), url="memory://dataset", max_bytes=1000
    )

    def full_disk(target, data):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(chunk_cache, "_atomic_write_bytes", full_disk)

    assert _read(store, key) == payload
    assert not _cached_path(store, key).exists()
    assert _cached_bytes(tmp_path) == 0


def test_v2_read_survives_cache_write_failure(tmp_path, monkeypatch):
    payload = b"n" * 64
    store = DiskCacheStoreV2(
        {"0.0.0": payload}, str(tmp_path), url="memory://dataset", max_bytes=1000
    )

    def full_disk(target, data):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(chunk_cache, "_atomic_write_bytes", full_disk)

    assert store["0.0.0"] == payload
    assert not _cached_path(store, "0.0.0").exists()
    assert _cached_bytes(tmp_path) == 0


@requires_zarr_v2
def test_v2_group_open_does_not_crawl_remote(tmp_path):
    # Without listdir on the store, zarr 2 falls back to iterating every key,
    # which on an FSStore is a full recursive listing of the remote.
    remote_dir = tmp_path / "remote"
    root = zarr.open_group(str(remote_dir), mode="w")
    child = root.create_dataset("child", shape=(4,), chunks=(4,), dtype="u1")
    child[:] = 7

    remote = _NoCrawlMapping(zarr.storage.FSStore(str(remote_dir), mode="r"))
    store = DiskCacheStoreV2(remote, str(tmp_path / "cache"), url="memory://dataset")

    group = zarr.open_group(store=store, mode="r")
    assert list(group.array_keys()) == ["child"]
    assert int(group["child"][0]) == 7


@requires_zarr_v3
def test_warm_byte_ranges_slice_cached_bytes(tmp_path):
    payload = bytes(range(32))
    key = _chunk_key(0)
    store = DiskCacheStore(
        _remote_with({key: payload}), str(tmp_path), url="memory://dataset"
    )
    assert _read(store, key) == payload

    assert _read(store, key, RangeByteRequest(4, 9)) == payload[4:9]
    assert _read(store, key, OffsetByteRequest(28)) == payload[28:]
    assert _read(store, key, SuffixByteRequest(5)) == payload[-5:]
    assert _read(store, key, SuffixByteRequest(len(payload) + 8)) == payload
    # A zero-length suffix is the empty tail, not the whole object.
    assert _read(store, key, SuffixByteRequest(0)) == b""

    with pytest.raises(ValueError, match="unexpected byte range"):
        _read(store, key, object())


@requires_zarr_v3
def test_byte_range_miss_delegates_without_caching(tmp_path):
    payload = bytes(range(32))
    key = _chunk_key(0)
    store = DiskCacheStore(
        _remote_with({key: payload}), str(tmp_path), url="memory://dataset"
    )

    assert _read(store, key, RangeByteRequest(2, 6)) == payload[2:6]
    # A partial response cannot populate a whole-object cache entry.
    assert not _cached_path(store, key).exists()
    assert _cached_bytes(tmp_path) == 0


def test_stamp_survives_eviction(tmp_path):
    payload = b"s" * 100
    keys = [_chunk_key(i) for i in range(6)]
    store = DiskCacheStore(
        _remote_with({key: payload for key in keys}),
        str(tmp_path),
        url="memory://dataset",
        max_bytes=300,
    )

    stamp = tmp_path / _CACHE_STAMP_NAME
    assert stamp.exists()

    for key in keys:
        assert _read(store, key) == payload

    assert _cached_bytes(tmp_path) <= 300
    assert stamp.exists()


def test_unstamped_cache_root_is_never_swept(tmp_path):
    # A cache_dir the user already keeps files in is not ours to delete from.
    foreign = tmp_path / "someones-notes.txt"
    foreign.write_bytes(b"f" * 1000)

    payload = b"x" * 100
    keys = [_chunk_key(i) for i in range(6)]
    store = DiskCacheStore(
        _remote_with({key: payload for key in keys}),
        str(tmp_path),
        url="memory://dataset",
        max_bytes=300,
    )
    assert not (tmp_path / _CACHE_STAMP_NAME).exists()

    with pytest.warns(UserWarning, match="did not write"):
        for key in keys:
            assert _read(store, key) == payload

    assert foreign.exists()
    for key in keys:
        assert _cached_path(store, key).exists()


def test_orphaned_temp_files_age_out(tmp_path):
    DiskCacheStore(_remote_with({}), str(tmp_path), url="memory://dataset", max_bytes=100)

    orphan = tmp_path / "chunk.tmp.999.1"
    orphan.write_bytes(b"o" * 500)
    fresh = tmp_path / "chunk.tmp.999.2"
    fresh.write_bytes(b"o" * 500)
    stale = time.time() - chunk_cache._ORPHAN_TEMP_AGE_SECONDS - 60.0
    os.utime(orphan, (stale, stale))

    chunk_cache._maybe_evict(str(tmp_path), 100, 10_000)

    assert not orphan.exists()
    assert fresh.exists()


def test_cache_path_stays_under_cache_dir(tmp_path):
    root = str(tmp_path)
    for build in (
        lambda url: DiskCacheStore(_remote_with({}), root, url=url),
        lambda url: DiskCacheStoreV2({}, root, url=url),
    ):
        # The layout for ordinary remote URLs is what existing caches are on disk.
        remote = build("https://dl.ash2txt.org/full-scrolls/s1.zarr")
        assert remote._cache_dir == os.path.join(
            root, "https", "dl.ash2txt.org", "full-scrolls", "s1.zarr"
        )

        # os.path.join alone would hand back '/abs/x.zarr', outside the root.
        local = build("file:///abs/x.zarr")
        assert local._cache_dir == os.path.join(root, "file", "abs", "x.zarr")

        ported = build("https://host:8080/x.zarr?token=1")
        assert ported._cache_dir.startswith(root + os.sep)

        with pytest.raises(ValueError, match="escapes cache_dir"):
            build("https://host/../../../../etc/x.zarr")


def test_concurrent_reads_share_one_cache_file(tmp_path):
    payload = b"t" * 256
    key = _chunk_key(0)
    store = DiskCacheStore(
        _remote_with({key: payload}), str(tmp_path), url="memory://dataset"
    )

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: _read(store, key), range(8)))

    assert results == [payload] * 8
    written = [
        os.path.join(dirpath, name)
        for dirpath, _dirnames, filenames in os.walk(store._cache_dir)
        for name in filenames
    ]
    assert written == [str(_cached_path(store, key))]


def test_read_refetches_when_cached_file_vanishes(tmp_path, monkeypatch):
    # The file can be evicted between the isfile check and the open; the read
    # falls through to the remote instead of raising FileNotFoundError.
    payload = b"r" * 64
    key = _chunk_key(0)
    store = DiskCacheStore(
        _remote_with({key: payload}), str(tmp_path), url="memory://dataset"
    )
    assert _read(store, key) == payload

    cached = str(_cached_path(store, key))
    real_isfile = os.path.isfile

    def isfile_then_unlink(path):
        found = real_isfile(path)
        if found and path == cached:
            os.unlink(path)
        return found

    monkeypatch.setattr(os.path, "isfile", isfile_then_unlink)
    assert _read(store, key) == payload
    monkeypatch.undo()

    assert _cached_path(store, key).exists()
