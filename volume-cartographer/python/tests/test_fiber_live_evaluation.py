"""Demand publication ownership, native consumer and real CPU inference."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
import os
from pathlib import Path
import socket
import threading
import time
from types import SimpleNamespace as NS

import numpy as np
import pytest

live = pytest.importorskip("lasagna.live_fiber_evaluation", reason="live integration needs sibling Lasagna package")
DemandPublicationSession, evaluate_live = live.DemandPublicationSession, live.evaluate_live
from test_fiber_chunk_trace import ft, inputs
from test_fiber_evaluation import config
from vc.fiber_evaluation import evaluate


class Store:
    def __init__(self, arrays, ready=False):
        self.products = [NS(level="", channels=[NS(relative_path=p) for p in arrays])]
        self.geometry = NS(chunk=256, output_shape=(256, 256, 256))
        self.published = ready
        self.requests = []

    def request(self, index, **kwargs):
        self.requests.append(tuple(index))
        return tuple(index)

    def ready(self, origin):
        return self.published

    def pending_tile_key(self):
        return 0 if self.requests and not self.published else None


def connect(session):
    return ft.LiveBridge(session.endpoint["port"], session.endpoint["token"], 5, True)


def test_callable_cached_evaluation(inputs):
    cfg = config(inputs)
    cfg.pop("chunks")
    cfg.pop("chunk_cache")
    cfg["backend"] = "beam"
    expected = evaluate(cfg, base_directory=inputs)
    store = Store([inputs / c for c in ("presence", "nx", "ny", "width")], ready=True)
    actual = evaluate_live(cfg, {"s": store}, None, source="s", base_directory=inputs)
    assert actual["cases"] == expected["cases"]
    assert store.requests
    assert "token" not in actual["config"]["live_endpoint"]


def test_callable_cancel_joins_consumer_before_producer(inputs):
    store = Store([inputs / c for c in ("presence", "nx", "ny", "width")])
    control = ft.EvaluationControl()
    closed = []
    @contextmanager
    def producer():
        try:
            yield lambda: None  # Simulate a manager waiting for resources.
        finally:
            closed.append(control.status()["finished"])
    def progress(status):
        if store.requests:
            control.cancel("caller cancelled during resource wait")
    with pytest.raises(RuntimeError, match="caller cancelled"):
        evaluate_live(config(inputs), {"s": store}, producer, source="s",
                      base_directory=inputs, control=control, progress=progress)
    assert closed == [True]


def test_publication_waits_and_closes_producer(tmp_path):
    store = Store([tmp_path])
    events = []
    publish = threading.Event()
    @contextmanager
    def producer():
        events.append("start")
        def pump():
            store.published = publish.is_set()
        try:
            yield pump
        finally:
            events.append("close")
    with DemandPublicationSession({"s": store}, producer) as session:
        client = connect(session)
        with ThreadPoolExecutor(2) as pool:
            futures = [pool.submit(client.wait, "s", 0, 0, [str(tmp_path)], [[0, 0, 0]]) for _ in range(2)]
            # Only publication, never scheduling or heartbeat, satisfies demand.
            deadline = time.monotonic() + 5
            while not events:
                assert not session.error
                assert time.monotonic() < deadline, "producer did not start"
                threading.Event().wait(.01)
            assert not any(f.done() for f in futures)
            publish.set()
            for f in futures:
                f.result(timeout=5)
        client.cancel("test complete")
    assert events == ["start", "close"]


def test_producer_failure_and_cancellation(tmp_path):
    store = Store([tmp_path])
    closed = []
    @contextmanager
    def producer():
        try:
            def pump():
                raise RuntimeError("producer failed")
            yield pump
        finally:
            closed.append(True)
    with pytest.raises(RuntimeError, match="producer failed"):
        with DemandPublicationSession({"s": store}, producer) as session:
            client = connect(session)
            try:
                client.wait("s", 0, 0, [str(tmp_path)], [[0, 0, 0]])
            finally:
                client.cancel("test complete")
    assert closed == [True]
    with pytest.raises(RuntimeError, match="cache miss"):
        with DemandPublicationSession({"s": store}, None) as session:
            client = connect(session)
            try:
                client.wait("s", 0, 0, [str(tmp_path)], [[0, 0, 0]])
            finally:
                client.cancel("test complete")


@pytest.mark.parametrize("change", [{"source": "wrong"}, {"chunks": [{"array": "/wrong", "index_zyx": [0, 0, 0]}]},
                                     {"remaining_base_voxels": -1}])
def test_invalid_requests_do_not_schedule(tmp_path, change):
    store = Store([tmp_path])
    with pytest.raises(ValueError):
        with DemandPublicationSession({"s": store}, None) as session:
            with socket.create_connection(("127.0.0.1", session.endpoint["port"]), timeout=5) as sock:
                messages = [dict(version=1, type="hello", token=session.endpoint["token"]),
                            dict(version=1, type="demand", request_id=0, source="s", remaining_base_voxels=0,
                                 chunks=[dict(array=str(tmp_path), index_zyx=[0, 0, 0])])]
                messages[-1].update(change)
                sock.sendall(b"".join((json.dumps(m)+"\n").encode() for m in messages))
                assert json.loads(sock.makefile("rb").readline())["type"] == "error"
    assert not store.requests


def test_real_ct_demand_worker_publication_and_warm_reuse(tmp_path):
    path = os.environ.get("LASAGNA_TEST_VOLUME")
    if not path:
        pytest.skip("set LASAGNA_TEST_VOLUME and LASAGNA_TEST_ORIGIN_ZYX")
    zarr = pytest.importorskip("zarr")
    pytest.importorskip("torch")
    from lasagna import tiled_predict3d as tiled
    from lasagna.demand_predict3d import DemandGeometry, DemandStore, DemandWorkerPool
    from lasagna.tests.test_demand_predict3d import outputs

    origin = tuple(map(int, os.environ["LASAGNA_TEST_ORIGIN_ZYX"].split(",")))
    data = np.asarray(zarr.open_array(path, mode="r")[tuple(slice(v, v+16) for v in origin)])
    assert data.shape == (16,)*3 and np.ptp(data) > 0
    geometry = DemandGeometry(data.shape, 16, 0, 0, 1, 16)
    adapter, output = outputs(tmp_path, data.shape, 16)
    launches = []
    with DemandStore(tmp_path, {"test": "real-CT"}, geometry, adapter, output, accumulator_dtype="float32") as store:
        def reader(tile):
            spec = tiled._tile_read_spec(data.shape, (0, 0, 0), *tile, 16, 0)
            return tiled._finalize_tile_read(data[spec.slices_zyx], spec)
        @contextmanager
        def producer():
            launches.append(True)
            with DemandWorkerPool({"s": store}, {"s": reader}, ["cpu"], input_workers=1,
                                  accumulator_workers=1, flush_workers=1) as pool:
                yield pool.pump
        for factory in (producer, None):
            with DemandPublicationSession({"s": store}, factory) as session:
                client = connect(session)
                try:
                    client.wait("s", 0, 0, [str(tmp_path / "value.zarr/0")], [[0, 0, 0]])
                    result = zarr.open_array(str(tmp_path / "value.zarr/0"), mode="r")[:]
                    np.testing.assert_array_equal(result, data)
                finally:
                    client.cancel("test complete")
    assert launches == [True]
