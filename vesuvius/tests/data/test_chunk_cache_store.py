import asyncio
from collections.abc import MutableMapping
from pathlib import Path

import pytest
import zarr

from vesuvius.data.chunk_cache import (
    _NEGATIVE_MARKER_SUFFIX,
    DiskCacheStore,
    default_chunk_cache_dir,
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


def test_default_chunk_cache_dir_env(tmp_path, monkeypatch):
    monkeypatch.setenv("VESUVIUS_CACHE_DIR", str(tmp_path))
    assert default_chunk_cache_dir() == tmp_path / "chunks"

    monkeypatch.delenv("VESUVIUS_CACHE_DIR", raising=False)
    assert default_chunk_cache_dir() == Path.home() / ".cache" / "vesuvius" / "chunks"
