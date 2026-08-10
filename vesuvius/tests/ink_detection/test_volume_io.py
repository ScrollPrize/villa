from __future__ import annotations

import asyncio
import os
from pathlib import Path
import pickle

import pytest
import zarr

if int(zarr.__version__.split(".", 1)[0]) < 3:
    pytest.skip(
        "Zarr 3 store contract tests require Zarr 3",
        allow_module_level=True,
    )

from zarr.abc.store import (
    OffsetByteRequest,
    RangeByteRequest,
    SuffixByteRequest,
)
from zarr.core.buffer.core import default_buffer_prototype

from vesuvius.ink_detection import volume_io
from vesuvius.ink_detection.volume_io import (
    AsyncDiskCachedStore,
    PidAwareFsspecStore,
    normalize_volume_url,
    open_volume,
)


def _data_files(cache_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in cache_dir.iterdir()
        if not path.name.endswith(".tmp")
    )


class _GuardedMemoryStore(zarr.storage.MemoryStore):
    def __init__(self) -> None:
        super().__init__(read_only=False)
        self.get_calls = []

    async def get(self, key, prototype, byte_range=None):
        self.get_calls.append((key, byte_range))
        return await super().get(key, prototype, byte_range)

    async def get_partial_values(self, prototype, key_ranges):
        raise AssertionError("partial reads must use cached complete values")

    async def _get_many(self, requests):
        raise AssertionError("batched reads must use cached complete values")
        yield


class _SerializableProcessStore:
    def __init__(self, read_only: bool) -> None:
        self.read_only = read_only

    async def get(self, key, prototype, byte_range=None):
        return prototype.buffer.from_bytes(key.encode())

    def close(self):
        return None


def _serializable_store_factory(url, options, read_only):
    return _SerializableProcessStore(read_only)


def _stable_process_id():
    return 500


def test_async_cache_serves_ranges_and_batches_from_complete_values(tmp_path):
    async def scenario():
        prototype = default_buffer_prototype()
        source = _GuardedMemoryStore()
        await source.set("value", prototype.buffer.from_bytes(b"0123456789"))
        cache = AsyncDiskCachedStore(source, tmp_path / "cache", max_bytes=100)

        ranged = await cache.get("value", prototype, RangeByteRequest(2, 6))
        assert ranged.to_bytes() == b"2345"
        partial = await cache.get_partial_values(
            prototype,
            (
                ("value", OffsetByteRequest(7)),
                ("value", SuffixByteRequest(3)),
                ("value", SuffixByteRequest(0)),
            ),
        )
        assert [value.to_bytes() for value in partial] == [b"789", b"789", b""]
        many = [
            item
            async for item in cache._get_many(
                (
                    ("value", prototype, None),
                    ("value", prototype, RangeByteRequest(0, 2)),
                )
            )
        ]
        assert [(key, value.to_bytes()) for key, value in many] == [
            ("value", b"0123456789"),
            ("value", b"01"),
        ]
        assert source.get_calls == [("value", None)]

    asyncio.run(scenario())


def test_async_cache_scans_only_after_local_size_crosses_budget(tmp_path, monkeypatch):
    prototype = default_buffer_prototype()
    at_limit_dir = tmp_path / "at-limit"
    at_limit_dir.mkdir()
    (at_limit_dir / "first").write_bytes(b"a" * 4)
    (at_limit_dir / "second").write_bytes(b"b" * 6)
    source = _GuardedMemoryStore()
    AsyncDiskCachedStore(source, at_limit_dir, max_bytes=10)
    assert [path.name for path in _data_files(at_limit_dir)] == [
        "first",
        "second",
    ]

    async def scenario():
        insert_dir = tmp_path / "insert-bounded"
        inserted_values = (
            ("a/chunk", b"a" * 2),
            ("b/chunk", b"b" * 2),
            ("c/chunk", b"c" * 3),
            ("d/chunk", b"d" * 5),
        )
        for key, value in inserted_values:
            await source.set(key, prototype.buffer.from_bytes(value))
        cache = AsyncDiskCachedStore(source, insert_dir, max_bytes=10)
        snapshot_calls = 0
        original_snapshot = volume_io._cache_snapshot

        def counted_snapshot(cache_dir):
            nonlocal snapshot_calls
            snapshot_calls += 1
            return original_snapshot(cache_dir)

        monkeypatch.setattr(volume_io, "_cache_snapshot", counted_snapshot)
        for timestamp, (key, _) in enumerate(inserted_values[:3], start=1):
            await cache.get(key, prototype)
            os.utime(cache._path(key), ns=(timestamp, timestamp))
        assert sum(path.stat().st_size for path in _data_files(insert_dir)) == 7
        assert snapshot_calls == 0
        await cache.get("d/chunk", prototype)
        assert snapshot_calls == 1
        data_files = _data_files(insert_dir)
        assert [path.name for path in data_files] == ["c__chunk", "d__chunk"]
        assert sum(path.stat().st_size for path in data_files) == 8

        oversized = AsyncDiskCachedStore(
            source, tmp_path / "oversized", max_bytes=5
        )
        await source.set("oversized", prototype.buffer.from_bytes(b"x" * 6))
        value = await oversized.get("oversized", prototype)
        assert value.to_bytes() == b"x" * 6
        assert _data_files(tmp_path / "oversized") == []
        assert not (insert_dir / ".cache.lock").exists()

    asyncio.run(scenario())


def test_async_cache_initializes_counter_without_evicting_or_cleaning(tmp_path):
    cache_dir = tmp_path / "existing"
    cache_dir.mkdir()
    (cache_dir / "first").write_bytes(b"a" * 5)
    (cache_dir / "second").write_bytes(b"b" * 7)
    abandoned = cache_dir / ".abandoned.tmp"
    abandoned.write_bytes(b"temporary")

    cache = AsyncDiskCachedStore(
        _GuardedMemoryStore(), cache_dir, max_bytes=10
    )

    assert cache._size == 12
    assert [path.name for path in _data_files(cache_dir)] == ["first", "second"]
    assert abandoned.exists()
    assert not (cache_dir / ".cache.lock").exists()


def test_async_cache_zero_budget_retains_no_nonempty_values(tmp_path):
    prototype = default_buffer_prototype()
    cache_dir = tmp_path / "zero"
    cache_dir.mkdir()
    (cache_dir / "existing").write_bytes(b"x")
    source = _GuardedMemoryStore()
    cache = AsyncDiskCachedStore(source, cache_dir, max_bytes=0)
    assert [path.name for path in _data_files(cache_dir)] == ["existing"]
    assert cache._size == 1

    async def scenario():
        await source.set("value", prototype.buffer.from_bytes(b"value"))
        value = await cache.get("value", prototype)
        assert value.to_bytes() == b"value"
        assert _data_files(cache_dir) == []

    asyncio.run(scenario())


def test_async_cache_wrappers_may_briefly_overshoot_shared_budget(tmp_path):
    async def scenario():
        prototype = default_buffer_prototype()
        source = _GuardedMemoryStore()
        for key in ("a", "b", "c", "d"):
            await source.set(key, prototype.buffer.from_bytes(key.encode() * 4))
        cache_dir = tmp_path / "shared"
        first = AsyncDiskCachedStore(source, cache_dir, max_bytes=10)
        second = AsyncDiskCachedStore(source, cache_dir, max_bytes=10)
        await first.get("a", prototype)
        os.utime(first._path("a"), ns=(10, 10))
        await second.get("b", prototype)
        os.utime(second._path("b"), ns=(20, 20))
        assert (await first.get("a", prototype)).to_bytes() == b"a" * 4
        os.utime(first._path("a"), ns=(30, 30))
        await second.get("c", prototype)
        assert [path.name for path in _data_files(cache_dir)] == ["a", "b", "c"]
        assert sum(path.stat().st_size for path in _data_files(cache_dir)) == 12
        await second.get("d", prototype)
        assert [path.name for path in _data_files(cache_dir)] == ["c", "d"]
        assert sum(path.stat().st_size for path in _data_files(cache_dir)) == 8

    asyncio.run(scenario())


def test_remote_store_rebuilds_per_pid_and_forwards_isolated_options():
    prototype = default_buffer_prototype()
    process_id = [100]
    constructions = []

    class ProcessStore:
        def __init__(self, marker: int) -> None:
            self.marker = marker

        async def get(self, key, prototype, byte_range=None):
            assert key == "value"
            return prototype.buffer.from_bytes(str(self.marker).encode())

        def close(self):
            return None

    def factory(url, options, read_only):
        constructions.append((url, options, read_only))
        return ProcessStore(len(constructions))

    authored_options = {"anon": True, "skip_instance_cache": False}
    store = PidAwareFsspecStore(
        "s3://vesuvius-challenge-open-data/value.zarr",
        authored_options,
        store_factory=factory,
        pid_provider=lambda: process_id[0],
    )

    async def scenario():
        first = await store.get("value", prototype)
        same_process = await store.get("value", prototype)
        process_id[0] = 101
        second_process = await store.get("value", prototype)
        assert first.to_bytes() == same_process.to_bytes() == b"1"
        assert second_process.to_bytes() == b"2"

    asyncio.run(scenario())
    assert authored_options == {"anon": True, "skip_instance_cache": False}
    assert constructions == [
        (
            "s3://vesuvius-challenge-open-data/value.zarr",
            {"anon": True, "skip_instance_cache": True},
            True,
        ),
        (
            "s3://vesuvius-challenge-open-data/value.zarr",
            {"anon": True, "skip_instance_cache": True},
            True,
        ),
    ]


def test_pid_store_with_read_only_reconstructs_durable_state():
    factory_calls = []

    def factory(url, options, read_only):
        factory_calls.append((url, options, read_only))
        return _SerializableProcessStore(read_only)

    pid_provider = lambda: 42
    store = PidAwareFsspecStore(
        "s3://vesuvius-challenge-open-data/value.zarr",
        {"anon": True},
        store_factory=factory,
        pid_provider=pid_provider,
    )
    writable = store.with_read_only(False)
    assert writable is not store
    assert writable.read_only is False
    assert writable.url == store.url
    assert writable.storage_options == store.storage_options
    assert writable.storage_options is not store.storage_options
    assert writable._store_factory is factory
    assert writable._pid_provider is pid_provider
    assert writable._process_store is None

    async def scenario():
        value = await writable.get("value", default_buffer_prototype())
        assert value.to_bytes() == b"value"
        with pytest.raises(NotImplementedError, match="do not support writes"):
            await writable.set("value", value)
        with pytest.raises(NotImplementedError, match="do not support deletes"):
            await writable.delete("value")

    asyncio.run(scenario())
    assert factory_calls == [
        (
            "s3://vesuvius-challenge-open-data/value.zarr",
            {"anon": True, "skip_instance_cache": True},
            False,
        )
    ]


def test_async_cache_with_read_only_preserves_cache_settings(tmp_path):
    source = zarr.storage.MemoryStore(read_only=False)
    cache = AsyncDiskCachedStore(source, tmp_path / "cache", max_bytes=17)
    read_only = cache.with_read_only(True)
    assert read_only is not cache
    assert read_only.read_only is True
    assert read_only.cache_dir == cache.cache_dir
    assert read_only.max_bytes == 17
    assert read_only._store is not source
    writable = read_only.with_read_only(False)
    assert writable.read_only is False
    assert writable.cache_dir == cache.cache_dir
    assert writable.max_bytes == 17


def test_pid_store_pickle_discards_populated_transport():
    store = PidAwareFsspecStore(
        "s3://vesuvius-challenge-open-data/value.zarr",
        {"anon": True},
        store_factory=_serializable_store_factory,
        pid_provider=_stable_process_id,
    )

    async def populate(candidate):
        value = await candidate.get("live", default_buffer_prototype())
        assert value.to_bytes() == b"live"

    asyncio.run(populate(store))
    assert store._process_store is not None
    restored = pickle.loads(pickle.dumps(store))
    assert restored._process_store is None
    assert restored._process_id is None
    assert restored.url == store.url
    assert restored.storage_options == store.storage_options
    assert restored._store_factory is _serializable_store_factory
    assert restored._pid_provider is _stable_process_id
    asyncio.run(populate(restored))


def test_async_cache_propagates_transport_exception_unchanged(tmp_path):
    class TransportFailure(Exception):
        pass

    failure = TransportFailure("transport failed")

    class FailingStore(_GuardedMemoryStore):
        async def get(self, key, prototype, byte_range=None):
            raise failure

    cache = AsyncDiskCachedStore(
        FailingStore(), tmp_path / "cache", max_bytes=10
    )

    async def scenario():
        with pytest.raises(TransportFailure) as raised:
            await cache.get("missing", default_buffer_prototype())
        assert raised.value is failure

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("authored", "expected", "is_public"),
    [
        (
            "s3://vesuvius-challenge-open-data/a/b.zarr",
            "s3://vesuvius-challenge-open-data/a/b.zarr",
            True,
        ),
        (
            "https://vesuvius-challenge-open-data.s3.amazonaws.com/a/b.zarr",
            "s3://vesuvius-challenge-open-data/a/b.zarr",
            True,
        ),
        (
            "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/a/b.zarr",
            "s3://vesuvius-challenge-open-data/a/b.zarr",
            True,
        ),
        (
            "https://vesuvius-challenge-open-data.s3-us-east-1.amazonaws.com/a/b.zarr",
            "s3://vesuvius-challenge-open-data/a/b.zarr",
            True,
        ),
        (
            "https://s3.amazonaws.com/vesuvius-challenge-open-data/a/b.zarr",
            "s3://vesuvius-challenge-open-data/a/b.zarr",
            True,
        ),
        (
            "https://s3.us-east-1.amazonaws.com/vesuvius-challenge-open-data/a/b.zarr",
            "s3://vesuvius-challenge-open-data/a/b.zarr",
            True,
        ),
        (
            "https://s3-us-east-1.amazonaws.com/vesuvius-challenge-open-data/a/b.zarr",
            "s3://vesuvius-challenge-open-data/a/b.zarr",
            True,
        ),
        (
            "https://vesuvius-challenge-open-data.invalid/a.zarr",
            "https://vesuvius-challenge-open-data.invalid/a.zarr",
            False,
        ),
    ],
)
def test_public_s3_url_normalization(authored, expected, is_public):
    assert normalize_volume_url(authored) == (expected, is_public)


@pytest.mark.parametrize("with_cache", [False, True])
def test_open_volume_uses_canonical_public_s3_store_without_network(
    tmp_path, monkeypatch, with_cache
):
    captured = {}

    class ArrayRoot:
        shape = (1, 1, 1)

    def fake_open(*, store, mode):
        captured["store"] = store
        captured["mode"] = mode
        return ArrayRoot()

    monkeypatch.setattr(volume_io.zarr, "open", fake_open)
    cache_options = (
        {"cache_dir": tmp_path, "cache_max_gb": 1} if with_cache else {}
    )
    result = open_volume(
        "https://vesuvius-challenge-open-data.s3.amazonaws.com/a.zarr",
        0,
        **cache_options,
    )
    assert isinstance(result, ArrayRoot)
    remote = (
        captured["store"]._store if with_cache else captured["store"]
    )
    assert remote.url == "s3://vesuvius-challenge-open-data/a.zarr"
    assert remote.storage_options == {"anon": True}
    assert captured["mode"] == "r"


def test_missing_resolution_uses_node_error_when_listing_fails(monkeypatch):
    class FailingGroup:
        def __getitem__(self, key):
            raise KeyError(key)

        def group_keys(self):
            raise RuntimeError("listing unavailable")

        def array_keys(self):
            raise AssertionError("listing should stop after the first failure")

    monkeypatch.setattr(volume_io, "open_vesuvius_zarr", lambda *args, **kwargs: FailingGroup())
    with pytest.raises(zarr.errors.NodeNotFoundError, match="resolution '3'") as raised:
        open_volume("missing.zarr", 3)
    assert isinstance(raised.value.__cause__, KeyError)

    class ArrayRoot:
        shape = (2, 2, 2)

    monkeypatch.setattr(volume_io, "open_vesuvius_zarr", lambda *args, **kwargs: ArrayRoot())
    with pytest.raises(zarr.errors.NodeNotFoundError, match="zarr array"):
        open_volume("array.zarr", 2)
    assert open_volume(
        "array.zarr", 2, root_array_is_requested_level=True
    ).shape == (2, 2, 2)


def test_missing_node_error_has_zarr2_fallback(monkeypatch):
    class CompatiblePathError(Exception):
        pass

    monkeypatch.delattr(volume_io.zarr.errors, "NodeNotFoundError")
    monkeypatch.setattr(
        volume_io.zarr.errors,
        "PathNotFoundError",
        CompatiblePathError,
        raising=False,
    )
    assert isinstance(
        volume_io._missing_node_error("missing"), CompatiblePathError
    )
