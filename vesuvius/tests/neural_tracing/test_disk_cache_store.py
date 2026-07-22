import asyncio

import pytest
import zarr

from vesuvius.neural_tracing.datasets.common import _DiskCacheStore, OfflineCacheMiss


ZARR_V3 = int(zarr.__version__.split(".", 1)[0]) >= 3


def test_disk_cache_store_caches_hits_and_misses(tmp_path):
    if not ZARR_V3:
        remote = {"chunk": b"payload"}
        cache = _DiskCacheStore(remote, str(tmp_path), url="memory://dataset")

        assert cache["chunk"] == b"payload"
        del remote["chunk"]
        assert cache["chunk"] == b"payload"
        assert "chunk" in cache

        read_only_copy = cache.with_read_only(True)
        assert read_only_copy["chunk"] == b"payload"

        with pytest.raises(KeyError):
            cache["missing"]
        remote["missing"] = b"too late"
        with pytest.raises(KeyError):
            cache["missing"]
        assert "missing" not in cache
        return

    from zarr.abc.store import RangeByteRequest
    from zarr.core.buffer.core import default_buffer_prototype
    from zarr.storage import MemoryStore

    async def exercise():
        prototype = default_buffer_prototype()
        remote = MemoryStore()
        await remote.set("chunk", prototype.buffer.from_bytes(b"payload"))
        cache = _DiskCacheStore(
            remote.with_read_only(True),
            str(tmp_path),
            url="memory://dataset",
        )

        first = await cache.get("chunk", prototype)
        assert first is not None
        assert first.to_bytes() == b"payload"

        await remote.delete("chunk")
        cached = await cache.get("chunk", prototype)
        assert cached is not None
        assert cached.to_bytes() == b"payload"
        cached_range = await cache.get("chunk", prototype, RangeByteRequest(1, 4))
        assert cached_range is not None
        assert cached_range.to_bytes() == b"ayl"
        assert await cache.exists("chunk")

        read_only_copy = cache.with_read_only(True)
        copied = await read_only_copy.get("chunk", prototype)
        assert copied is not None
        assert copied.to_bytes() == b"payload"

        assert await cache.get("missing", prototype) is None
        await remote.set("missing", prototype.buffer.from_bytes(b"too late"))
        assert await cache.get("missing", prototype) is None
        assert not await cache.exists("missing")

    asyncio.run(exercise())


def test_disk_cache_store_offline_miss_and_range_read(tmp_path):
    if not ZARR_V3:
        remote = {"chunk": b"payload"}
        online = _DiskCacheStore(remote, str(tmp_path), url="memory://online")
        assert online["chunk"] == b"payload"

        offline = _DiskCacheStore(
            remote,
            str(tmp_path),
            url="memory://offline",
            offline=True,
        )
        with pytest.raises(OfflineCacheMiss):
            offline["chunk"]
        return

    from zarr.abc.store import RangeByteRequest
    from zarr.core.buffer.core import default_buffer_prototype
    from zarr.storage import MemoryStore

    async def exercise():
        prototype = default_buffer_prototype()
        remote = MemoryStore()
        await remote.set("chunk", prototype.buffer.from_bytes(b"payload"))

        online = _DiskCacheStore(
            remote.with_read_only(True),
            str(tmp_path),
            url="memory://online",
        )
        partial = await online.get("chunk", prototype, RangeByteRequest(1, 4))
        assert partial is not None
        assert partial.to_bytes() == b"ayl"

        offline = _DiskCacheStore(
            remote.with_read_only(True),
            str(tmp_path),
            url="memory://offline",
            offline=True,
        )
        with pytest.raises(OfflineCacheMiss):
            await offline.get("chunk", prototype)

    asyncio.run(exercise())
