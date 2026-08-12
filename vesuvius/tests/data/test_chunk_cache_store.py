import asyncio
import os
import time
from collections.abc import MutableMapping
from pathlib import Path

import pytest
import zarr

from vesuvius.data.chunk_cache import (
    _NEGATIVE_MARKER_SUFFIX,
    DiskCacheStore,
    DiskCacheStoreV2,
)


ZARR_V3 = int(zarr.__version__.split(".", 1)[0]) >= 3

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


def _read(store, key: str):
    """Read one key through the store, returning its bytes or None."""
    if not ZARR_V3:
        return store[key]

    from zarr.core.buffer.core import default_buffer_prototype

    async def exercise():
        result = await store.get(key, default_buffer_prototype())
        return None if result is None else result.to_bytes()

    return asyncio.run(exercise())


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
