"""Reader contracts exercised by vesuvius loaders and multi-process training."""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import pickle
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("vc_reader._native")

from test_group import _make_group
from test_zarr_reader import _serve, _write_zarr, local_array, data  # noqa: F401
from vc_reader.reader import CacheMissError, MemoryCache, ZarrGroup, ZarrReader


@pytest.mark.parametrize("key", [
    ([5, 1, 5, -1], slice(2, 19), slice(3, 24)),
    (slice(3, 20), slice(1, 19), [4, 0, -2]),
    (slice(1, 20), [4, 0], slice(0, 20)),
    (0, slice(2, 19), [4, 0]),
    (slice(2, 19), 0, [4, 0]),
    ([4, 0], slice(2, 19), 0),
    ([4, 0], Ellipsis),
    (Ellipsis, [4, 0]),
    ([[4], [0]], slice(2, 19), [3, 2, 1]),
    (slice(2, 19), [[4], [0]], [3, 2, 1]),
    (np.array(1), Ellipsis),
    ([], slice(None), slice(None)),
    (slice(4, 4), [1, 2], slice(None)),
    (0, Ellipsis, [1, 2], slice(None)),
])
def test_mixed_index_matches_numpy(local_array, key):
    root, expected = local_array
    with ZarrReader.open(root, threads=2) as reader:
        np.testing.assert_array_equal(reader[key], expected[key])


def test_array_conversion(local_array):
    root, expected = local_array
    with ZarrReader.open(root, threads=2) as reader:
        np.testing.assert_array_equal(np.asarray(reader), expected)
        np.testing.assert_array_equal(np.asarray(reader, dtype=np.float32), expected.astype(np.float32))
        with pytest.raises(ValueError, match="copy"):
            reader.__array__(copy=False)


def test_scalar_array_index_with_empty_ellipsis(tmp_path):
    path = tmp_path / "array"
    _write_zarr(str(path), np.arange(8, dtype=np.uint16), (4,))
    with ZarrReader.open(path, threads=1) as reader:
        assert reader[np.array(3), ...] == 3


def test_attributes_and_discovery(tmp_path, data):
    path, _ = _make_group(tmp_path, data)
    attrs = {"_ARRAY_DIMENSIONS": ["z", "y", "x"], "meaning": "CT"}
    (path / "0" / ".zattrs").write_text(json.dumps(attrs))
    with ZarrGroup.open(path, threads=1) as group:
        assert group.array_keys() == ["0", "1", "2"]
        assert group.group_keys() == ["labels"]
        assert group["0"].attrs == attrs
        assert group["1"].attrs == {}
        assert group["1"].compressor["id"] == "blosc"
        assert group["0"].filters is None and group["0"].order == "C"
        other = pickle.loads(pickle.dumps(group["0"]))
        try:
            assert other.attrs == attrs
        finally:
            other.close()
    closed = ZarrReader.open(path / "0", threads=1)
    closed.close()
    with pytest.raises(ValueError, match="closed"):
        closed.attrs


@pytest.mark.parametrize("consolidated", [False, True])
def test_remote_attributes_cached_for_offline(tmp_path, data, consolidated):
    path, _ = _make_group(tmp_path / "www", data)
    attrs = {"_ARRAY_DIMENSIONS": ["z", "y", "x"]}
    (path / "0" / ".zattrs").write_text(json.dumps(attrs))
    if consolidated:
        metadata = {str(p.relative_to(path)): json.loads(p.read_text())
                    for p in path.rglob(".*") if p.name in (".zarray", ".zgroup", ".zattrs")}
        (path / ".zmetadata").write_text(json.dumps({"zarr_consolidated_format": 1, "metadata": metadata}))
    server, base = _serve(path.parent)
    cache = tmp_path / "cache"
    try:
        with ZarrGroup.open(f"{base}/vol.zarr", cache_dir=cache, threads=1) as group:
            assert group["0"].attrs == attrs
            assert group["1"].attrs == {}
            if consolidated:
                assert group.array_keys() == ["0", "1", "2"]
                assert group.group_keys() == ["labels"]
        server.requests.clear()
        with ZarrGroup.open(f"{base}/vol.zarr", cache_dir=cache, offline=True, threads=1) as group:
            assert group.array_keys() == ["0", "1"]
            assert group["0"].attrs == attrs
            assert group["1"].attrs == {}
        assert server.requests == []
    finally:
        server.shutdown()


def _check_shared(readers):
    a, b = readers
    a.read((0, 0, 0), a.chunks)
    before = a.memory_cache.misses
    b.read((0, 0, 0), b.chunks)
    assert a.memory_cache is b.memory_cache
    assert a.memory_cache.misses == before
    assert a.memory_cache.nbytes <= a.memory_cache.max_bytes
    for reader in readers:
        reader.close()


def _shared_worker(readers, queue):
    try:
        _check_shared(readers)
        queue.put(None)
    except BaseException as exc:
        queue.put(repr(exc))


def test_shared_budget_pickle(local_array):
    root, _ = local_array
    cache = MemoryCache(8192)
    readers = [ZarrReader.open(root, memory_cache=cache, threads=1) for _ in range(2)]
    try:
        _check_shared(pickle.loads(pickle.dumps(readers)))
    finally:
        for reader in readers:
            reader.close()


@pytest.mark.parametrize("method", ["fork", "spawn", "forkserver"])
@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
def test_shared_budget_worker(tmp_path, method):
    if method not in mp.get_all_start_methods():
        pytest.skip(f"{method} unavailable")
    path = tmp_path / "array"
    _write_zarr(str(path), np.ones((8, 8, 8), np.uint8), (4, 4, 4))
    cache = MemoryCache(128)
    readers = [ZarrReader.open(path, memory_cache=cache, threads=1) for _ in range(2)]
    readers[0].read((0, 0, 0), (4, 4, 4))
    ctx = mp.get_context(method)
    queue = ctx.Queue()
    child = ctx.Process(target=_shared_worker, args=(readers, queue))
    try:
        child.start()
        child.join(30)
        assert not child.is_alive(), "worker hung"
        assert child.exitcode == 0
        assert queue.get(timeout=5) is None
        assert readers[0].memory_cache is readers[1].memory_cache is cache
        assert cache.nbytes == 64
    finally:
        if child.is_alive():
            child.kill()
            child.join()
        queue.close()
        for reader in readers:
            reader.close()


def test_group_shares_budget_after_pickle(tmp_path, data):
    path, _ = _make_group(tmp_path, data)
    with ZarrGroup.open(path, memory_cache=8192, threads=1) as group:
        copied_group, copied_array = pickle.loads(pickle.dumps((group, group["0"])))
        try:
            assert copied_group.memory_cache is copied_array.memory_cache
            assert copied_group["1"].memory_cache is copied_array.memory_cache
        finally:
            copied_group.close()
            copied_array.close()


@pytest.mark.parametrize("payload", [b"broken", b"", None])
def test_zstd_corrupt_or_wrong_size(tmp_path, payload):
    from numcodecs import Zstd
    path = tmp_path / "array"
    _write_zarr(str(path), np.ones((4, 4, 4), np.uint8), (4, 4, 4), compressor={"id": "zstd"})
    (path / "0/0/0").write_bytes(payload if payload is not None else Zstd().encode(bytes(32)))
    with ZarrReader.open(path, threads=1) as reader:
        with pytest.raises(RuntimeError, match="zstd"):
            reader[:]
