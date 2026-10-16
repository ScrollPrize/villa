"""Bounded decoded caches: accounting, eviction, offline reads and workers."""
from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
import sqlite3

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("vc_reader._native")

from test_group import _make_group
from test_zarr_reader import _serve, _write_zarr, _vc3d_mirror, group_server, data  # noqa: F401
from vc_reader.reader import CacheMissError, ZarrGroup, ZarrReader
from vc_reader.reader.disk_cache import decoded_cache_path


@pytest.fixture
def remote(tmp_path):
    data = np.arange(8**3, dtype=np.uint16).reshape((8, 8, 8))
    www = tmp_path / "www"
    for name in ("a", "b"):
        _write_zarr(str(www / name), data, (4, 4, 4))
    server, base = _serve(www)
    try:
        yield base, data, server
    finally:
        server.shutdown()


def _chunk_bytes(root):
    return sum(p.stat().st_size for p in Path(root).rglob("*")
               if p.is_file() and p.name.isdigit())


def _indexed_bytes(root):
    with sqlite3.connect(Path(root) / ".vc_reader_cache.sqlite3") as db:
        return db.execute("SELECT total FROM budget").fetchone()[0]


def test_limit_eviction_and_refetch(remote, tmp_path):
    base, data, server = remote
    cache = tmp_path / "cache"
    url = f"{base}/a"
    target = Path(decoded_cache_path(cache, url))
    with ZarrReader.open(url, cache_dir=cache, cache_max_bytes=256, threads=1) as reader:
        for start in ((0, 0, 0), (0, 0, 4), (0, 4, 0)):
            np.testing.assert_array_equal(reader.read(start, (4, 4, 4)),
                                          data[tuple(slice(i, i+4) for i in start)])
            reader.flush()
        assert not (target / "0/0/0").exists()
        assert (target / "0/0/1").exists() and (target / "0/1/0").exists()
        assert _chunk_bytes(cache) == _indexed_bytes(cache) == 256
        reader[0:4, 0:4, 0:4]
        reader.flush()
        assert (target / "0/0/0").exists()
        assert _chunk_bytes(cache) == 256
        assert reader.stats()["remote_fetches"] == 4
        assert reader.stats()["write_errors"] == 0
    server.requests.clear()
    with ZarrReader.open(url, cache_dir=cache, cache_max_bytes=0, offline=True, threads=1) as reader:
        np.testing.assert_array_equal(reader[:4, :4, :4], data[:4, :4, :4])
        with pytest.raises(CacheMissError):
            reader[:4, :4, 4:8]
    assert server.requests == []
    assert _chunk_bytes(cache) == 256  # offline opening never evicts


@pytest.mark.parametrize("limit", [0, 127])
def test_oversized_chunks_are_not_cached(remote, tmp_path, limit):
    base, data, _ = remote
    cache = tmp_path / "cache"
    with ZarrReader.open(f"{base}/a", cache_dir=cache, cache_max_bytes=limit, threads=1) as reader:
        np.testing.assert_array_equal(reader[:4, :4, :4], data[:4, :4, :4])
        reader.flush()
        assert _chunk_bytes(cache) == 0
        assert reader.stats()["write_errors"] == 0


def test_budget_shared_across_arrays(remote, tmp_path):
    base, _, _ = remote
    cache = tmp_path / "cache"
    with ZarrReader.open(f"{base}/a", cache_dir=cache, cache_max_bytes=128, threads=1) as a, \
         ZarrReader.open(f"{base}/b", cache_dir=cache, cache_max_bytes=128, threads=1) as b:
        a[:4, :4, :4]
        a.flush()
        b[:4, :4, :4]
        b.flush()
        assert _chunk_bytes(cache) == _indexed_bytes(cache) == 128
        assert not (Path(decoded_cache_path(cache, f"{base}/a")) / "0/0/0").exists()


def test_reopening_updates_shared_limit(remote, tmp_path):
    base, _, _ = remote
    cache = tmp_path / "cache"
    with ZarrReader.open(f"{base}/a", cache_dir=cache, cache_max_bytes=512, threads=1) as a:
        a[:]
        a.flush()
        with ZarrReader.open(f"{base}/b", cache_dir=cache, cache_max_bytes=128, threads=1):
            assert _indexed_bytes(cache) == 128
            a[:4, :4, :4]
            a.flush()
            assert _chunk_bytes(cache) == _indexed_bytes(cache) == 128


def test_existing_cache_pruned_metadata_retained(remote, tmp_path):
    base, _, _ = remote
    cache = tmp_path / "cache"
    url = f"{base}/a"
    with ZarrReader.open(url, cache_dir=cache, threads=1) as reader:
        reader[:]
    target = Path(decoded_cache_path(cache, url))
    attrs = (target / ".zattrs").read_bytes()
    (target / "notes.txt").write_text("keep")
    (cache / "foreign-cache").mkdir()
    (cache / "foreign-cache/0").write_bytes(bytes(1024))
    with ZarrReader.open(url, cache_dir=cache, cache_max_bytes=128, threads=1):
        assert _chunk_bytes(target) == _indexed_bytes(cache) == 128
        assert (target / ".zattrs").read_bytes() == attrs
        assert (target / ".zarray").is_file()
        assert (target / "notes.txt").read_text() == "keep"
        assert (cache / "foreign-cache/0").stat().st_size == 1024


def test_readonly_does_not_enforce_limit(remote, tmp_path):
    base, _, _ = remote
    cache = tmp_path / "cache"
    with ZarrReader.open(f"{base}/a", cache_dir=cache, threads=1) as reader:
        reader[:]
    before = _chunk_bytes(cache)
    with ZarrReader.open(f"{base}/a", cache_dir=cache, cache_writes=False, cache_max_bytes=0, threads=1) as reader:
        reader[:]
    assert _chunk_bytes(cache) == before
    assert not (cache / ".vc_reader_cache.sqlite3").exists()


def test_compressed_cache_accounts_stored_bytes(remote, tmp_path):
    base, data, _ = remote
    cache = tmp_path / "cache"
    with ZarrReader.open(f"{base}/a", cache_dir=cache, cache_max_bytes=200, compress_cache=True, threads=1) as reader:
        np.testing.assert_array_equal(reader[:], data)
        reader.flush()
        assert _chunk_bytes(cache) == _indexed_bytes(cache) <= 200
        assert reader.stats()["write_errors"] == 0


def _bounded_worker(url, cache, ready, go, queue):
    try:
        with ZarrReader.open(url, cache_dir=cache, cache_max_bytes=256, threads=1) as reader:
            ready.set()
            if not go.wait(20):
                raise TimeoutError("parent did not start workers")
            reader[:]
            reader.flush()
            assert reader.stats()["write_errors"] == 0, reader.stats()
        queue.put(None)
    except BaseException as exc:
        queue.put(repr(exc))


def test_worker_writes_share_disk_limit(remote, tmp_path):
    base, _, _ = remote
    cache = str(tmp_path / "cache")
    ctx = mp.get_context("spawn")
    go, queue = ctx.Event(), ctx.Queue()
    ready = [ctx.Event(), ctx.Event()]
    children = [ctx.Process(target=_bounded_worker, args=(f"{base}/{name}", cache, event, go, queue))
                for name, event in zip(("a", "b"), ready)]
    try:
        for child in children:
            child.start()
        assert all(event.wait(20) for event in ready)
        go.set()
        for child in children:
            child.join(30)
            assert not child.is_alive() and child.exitcode == 0
            assert queue.get(timeout=5) is None
        assert _chunk_bytes(cache) == _indexed_bytes(cache) <= 256
    finally:
        for child in children:
            if child.is_alive():
                child.kill()
                child.join()
        queue.close()


def test_group_passes_limit_to_levels(tmp_path):
    path, _ = _make_group(tmp_path / "www", np.ones((32, 32, 32), np.uint8))
    server, base = _serve(path.parent)
    cache = tmp_path / "cache"
    try:
        with ZarrGroup.open(f"{base}/vol.zarr", cache_dir=cache, cache_max_bytes=4096, threads=1) as group:
            group["0"][:16, :16, :16]
            group["0"].flush()
            group["1"][:]
            group["1"].flush()
            assert _chunk_bytes(cache) == _indexed_bytes(cache) == 4096
    finally:
        server.shutdown()


def test_limit_validation():
    with pytest.raises(ValueError, match="nonnegative"):
        ZarrReader.open("http://unused/a", cache_max_bytes=-1)
    with pytest.raises(ValueError, match="requires cache_dir"):
        ZarrReader.open("http://unused/a", cache_max_bytes=10)


def test_bounded_reader_keeps_vc3d_cache_readonly(group_server, tmp_path):
    server, base, source = group_server
    url = f"{base}/vol.zarr"
    vc3d = _vc3d_mirror(tmp_path, url, source, ["0/0/0"])
    before = {str(p.relative_to(vc3d)): p.read_bytes() for p in vc3d.rglob("*") if p.is_file()}
    with ZarrReader.open(url, level=0, cache_dir=tmp_path, cache_max_bytes=4096, threads=1) as reader:
        assert reader.write_layer.kind == "decoded"
        server.requests.clear()
        reader[:16, :16, :16]
        assert server.requests == []
        reader[:16, 16:32, :16]
        reader.flush()
        assert reader.stats()["write_errors"] == 0
    after = {str(p.relative_to(vc3d)): p.read_bytes() for p in vc3d.rglob("*") if p.is_file()}
    assert before == after
