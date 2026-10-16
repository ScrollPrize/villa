"""Small local fixtures for scheduling, publication and process handoff."""
from concurrent.futures import CancelledError, ThreadPoolExecutor
import multiprocessing as mp
import threading

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("vc_reader._native")

from test_zarr_reader import _serve, _write_zarr
from vc_reader.reader import MemoryCache, PrefetchService, ZarrReader
from vc_reader.reader.prefetch import _Scheduler


@pytest.fixture
def remote(tmp_path):
    data = np.arange(8**3, dtype=np.uint16).reshape((8, 8, 8))
    www = tmp_path / "www"
    _write_zarr(www / "a", data, (4, 4, 4), missing={(1, 1, 1)})
    server, base = _serve(www)
    try:
        yield base + "/a", data, server
    finally:
        server.shutdown()
        server.server_close()


def test_local_prefetch_and_window(tmp_path):
    data = np.arange(8**3, dtype=np.uint16).reshape((8, 8, 8))
    root = tmp_path / "a"
    _write_zarr(root, data, (4, 4, 4), compressor={"id": "zstd", "level": 1})
    cache = MemoryCache(256)
    with ZarrReader.open(root, memory_cache=cache, connections=2, threads=1) as reader:
        request = reader.prefetch([((0, 0, 0), (4, 4, 4))] * 2)
        request.wait(5)
        assert request.done() and request.exception() is None
        np.testing.assert_array_equal(reader[:4, :4, :4], data[:4, :4, :4])
        assert reader.stats()["memory_hits"] >= 1
        with reader.prefetch_window() as window:
            batches = [[((0, 0, 0), data.shape)], [((-5, -5, -5), (2, 2, 2))]]
            window.update(batches)
            window.wait(5)
            completed = reader.prefetch_stats()["completed"]
            window.update(batches)
            window.wait(5)
            assert reader.prefetch_stats()["completed"] == completed
        assert cache.nbytes <= cache.max_bytes


def test_demand_reserved_capacity_and_join():
    gate, started, demanded = threading.Event(), threading.Event(), threading.Event()
    calls = []

    def load(index):
        calls.append(index)
        if index == (0,):
            started.set()
            assert gate.wait(5)
        if index == (2,):
            demanded.set()
        return index, True

    scheduler = _Scheduler(connections=2, max_pending_chunks=2, max_inflight_bytes=128)
    scheduler.register("a", load, 64)
    try:
        scheduler.update("window", "a", [(0,), (1,)], True)
        assert started.wait(5)
        with ThreadPoolExecutor(2) as pool:
            urgent = pool.submit(scheduler.demand, "a", (2,))
            assert demanded.wait(5)  # Does not wait behind the blocked speculation.
            assert urgent.result(5) == (2,)
            joined = pool.submit(scheduler.demand, "a", (0,))
            with scheduler.cv:
                assert scheduler.cv.wait_for(lambda: scheduler.jobs[("a", (0,))].demand is not None, 5)
            gate.set()
            assert joined.result(5) == (0,)
        scheduler.wait("window", 5)
        assert calls.count((0,)) == 1
        assert calls.index((2,)) < calls.index((1,))
        assert scheduler.stats()["active_bytes"] == 0
    finally:
        gate.set()
        scheduler.close()


def test_replace_cancel_and_small_queue():
    gate, started = threading.Event(), threading.Event()
    calls = []

    def load(index):
        calls.append(index)
        if index == (0,):
            started.set()
            assert gate.wait(5)
        return None, True

    scheduler = _Scheduler(connections=1, max_pending_chunks=1)
    scheduler.register("a", load, 64)
    try:
        scheduler.update("old", "a", [(0,), (1,), (2,)], True)
        assert started.wait(5)
        scheduler.update("other", "a", [(1,)])
        scheduler.update("old", "a", [(0,), (3,), (4,)], True)
        scheduler.update("cancel", "a", [(5,)])
        assert scheduler.cancel("cancel")
        with pytest.raises(CancelledError):
            scheduler.wait("cancel", 1)
        gate.set()
        scheduler.wait("old", 5)
        scheduler.wait("other", 5)
        assert set(calls) == {(0,), (1,), (3,), (4,)}
        assert len(calls) == 4
    finally:
        gate.set()
        scheduler.close()


def test_remote_publication_and_missing(remote, tmp_path):
    url, data, server = remote
    cache = tmp_path / "cache"
    with ZarrReader.open(url, cache_dir=cache, connections=2, threads=1) as reader:
        request = reader.prefetch([((0, 0, 0), data.shape)])
        request.wait(10)
        server.requests.clear()
        with ZarrReader.open(url, cache_dir=cache, offline=True, threads=1) as offline:
            expected = data.copy()
            expected[4:, 4:, 4:] = 7
            np.testing.assert_array_equal(offline[:], expected)
        assert server.requests == []


def _worker_read(reader):
    with reader:
        with reader.prefetch_window() as window:
            window.update([[((0, 0, 0), (4, 4, 4))]])
            window.wait(10)
            return reader[:4, :4, :4], reader.stats()["remote_fetches"]


def _inherited_read(reader, sender):
    try:
        sender.send(_worker_read(reader))
    finally:
        sender.close()


@pytest.mark.skipif("fork" not in mp.get_all_start_methods(), reason="fork is unavailable")
def test_inherited_local_scheduler(tmp_path):
    data = np.arange(64, dtype=np.uint8).reshape((4, 4, 4))
    root = tmp_path / "a"
    _write_zarr(root, data, (4, 4, 4))
    with ZarrReader.open(root, memory_cache=128, connections=1, threads=1) as reader:
        reader.prefetch([((0, 0, 0), (4, 4, 4))]).wait(5)
        context = mp.get_context("fork")
        receiver, sender = context.Pipe(duplex=False)
        process = context.Process(target=_inherited_read, args=(reader, sender))
        process.start()
        sender.close()
        try:
            assert receiver.poll(10)
            result, _ = receiver.recv()
            np.testing.assert_array_equal(result, data)
            process.join(5)
            assert process.exitcode == 0
            np.testing.assert_array_equal(reader[:], data)
        finally:
            if process.is_alive():
                process.terminate()
                process.join(5)
            receiver.close()


@pytest.mark.parametrize("method", ["spawn", "fork"] if "fork" in mp.get_all_start_methods() else ["spawn"])
def test_service_shared_between_workers(remote, tmp_path, method):
    url, data, server = remote
    with PrefetchService(connections=2, max_pending_chunks=2) as service:
        with ZarrReader.open(url, cache_dir=tmp_path / "cache", prefetcher=service.client,
                             threads=1) as reader:
            with mp.get_context(method).Pool(2) as workers:
                outputs = workers.map(_worker_read, [reader, reader])
            for result, fetched in outputs:
                np.testing.assert_array_equal(result, data[:4, :4, :4])
                assert fetched == 0  # The coordinator owns remote requests.
            assert server.requests.count("/a/0/0/0") == 1


@pytest.mark.parametrize("limit", [1, 128])
def test_service_demand_eviction_and_oversized_chunk(remote, tmp_path, limit):
    url, data, server = remote
    with PrefetchService(connections=2, max_pending_chunks=2, max_inflight_bytes=256) as service:
        # One chunk or less: demand must work across eviction without a cache lease.
        with ZarrReader.open(url, cache_dir=tmp_path / "cache", cache_max_bytes=limit,
                             prefetcher=service.client, threads=2) as reader:
            request = reader.prefetch([((0, 0, 0), (4, 4, 4))])
            if limit < 128:
                with pytest.raises(RuntimeError, match="published"):
                    request.wait(10)
            else:
                request.wait(10)
            expected = data.copy()
            expected[4:, 4:, 4:] = 7
            np.testing.assert_array_equal(reader[:], expected)
            np.testing.assert_array_equal(reader.read_points([[0, 1, 2], [7, 7, 7]]),
                                          [data[0, 1, 2], 7])
            assert reader.stats()["remote_fetches"] == 0
            assert service.stats()["active_bytes"] <= 256


def test_memory_only_and_offline_prefetch(remote, tmp_path):
    from vc_reader.reader import CacheMissError
    url, data, server = remote
    region = [((0, 0, 0), (4, 4, 4))]
    with ZarrReader.open(url, memory_cache=128, connections=1, threads=1) as reader:
        reader.prefetch(region).wait(5)
        server.requests.clear()
        np.testing.assert_array_equal(reader[:4, :4, :4], data[:4, :4, :4])
        assert server.requests == []
    cache = tmp_path / "cache"
    ZarrReader.open(url, cache_dir=cache, threads=1).close()  # metadata only
    with ZarrReader.open(url, cache_dir=cache, offline=True, connections=1, threads=1) as reader:
        server.requests.clear()
        with pytest.raises(CacheMissError):
            reader.prefetch(region).wait(5)
        assert server.requests == []


def test_validation_failure_and_shutdown(tmp_path):
    data = np.arange(64, dtype=np.uint8).reshape((4, 4, 4))
    root = tmp_path / "bad"
    _write_zarr(root, data, (4, 4, 4))
    (root / "0/0/0").write_bytes(b"short")
    with ZarrReader.open(root, threads=1, connections=1) as reader:
        with pytest.raises(ValueError, match="nonnegative"):
            reader.prefetch([((0, 0, 0), (-1, 4, 4))])
        request = reader.prefetch([((0, 0, 0), (4, 4, 4))])
        with pytest.raises(RuntimeError):
            request.wait(5)
        assert request.done() and request.exception() is not None
        reader.prefetch([]).wait(1)
    with pytest.raises(ValueError, match="closed"):
        reader.prefetch([])


def test_service_offline_and_stop(remote, tmp_path):
    url, _, _ = remote
    service = PrefetchService(connections=1)
    with pytest.raises(ValueError, match="online"):
        ZarrReader.open(url, cache_dir=tmp_path / "cache", offline=True, prefetcher=service.client)
    with ZarrReader.open(url, cache_dir=tmp_path / "cache", prefetcher=service.client, threads=1) as reader:
        service.close()
        with pytest.raises(RuntimeError, match="service stopped"):
            reader[:4, :4, :4]
