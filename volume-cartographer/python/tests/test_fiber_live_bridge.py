"""Loopback protocol tests; cached Paris4 validation never launches inference."""
import gc
import json
import os
import socket
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from test_fiber_chunk_trace import ft, inputs, handles, chunk_config, trace_config

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="live transport requires POSIX sockets")


class Coordinator:
    def __init__(self, handler):
        self.listener = socket.socket()
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen()
        self.listener.settimeout(5)
        self.port = self.listener.getsockname()[1]
        self.handler = handler
        self.requests = []
        self.errors = []
        self.stopping = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def send(self, connection, **message):
        connection.sendall((json.dumps(dict(version=1, **message)) + "\n").encode())

    def ready(self, connection, request):
        self.send(connection, type="ready", request_id=request["request_id"])

    def run(self):
        try:
            with self.listener.accept()[0] as connection:
                connection.settimeout(5)
                self.connection = connection
                with connection.makefile("rb") as reader:
                    hello = json.loads(reader.readline())
                    assert hello == dict(version=1, type="hello", token="test-token")
                    for line in reader:
                        request = json.loads(line)
                        assert request["version"] == 1 and request["type"] == "demand"
                        self.requests.append(request)
                        self.handler(self, connection, request)
        except (OSError, ValueError) as error:
            if not self.stopping.is_set():
                self.errors.append(error)
        except BaseException as error:
            self.errors.append(error)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stopping.set()
        if hasattr(self, "connection"):
            try:
                self.connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        self.listener.close()
        self.thread.join(6)
        assert not self.thread.is_alive()
        assert not self.errors


def bridge(server, **kwargs):
    return ft.LiveBridge(server.port, "test-token", **kwargs)


def wait(client, index=0):
    client.wait("prediction", index, 256, ["/prediction/presence/0"], [[0, 0, index]])


def test_concurrent_acknowledgements_and_proof_cache():
    def respond(server, connection, request):
        if len(server.requests) == 4:
            for req in reversed(server.requests):
                server.ready(connection, req)
    with Coordinator(respond) as server:
        client = bridge(server)
        with ThreadPoolExecutor(4) as pool:
            list(pool.map(lambda i: wait(client, i), range(4)))
        for i in range(4):
            wait(client, i)
        assert len(server.requests) == 4
        client.cancel("test cancellation")
        with pytest.raises(RuntimeError, match="test cancellation"):
            wait(client)
        with pytest.raises(RuntimeError, match="test cancellation"):
            client.wait("prediction", 0, 0, [], [])


def test_heartbeat_keeps_wait_alive_but_does_not_publish():
    received = threading.Event()
    def respond(server, connection, request):
        received.set()
        for _ in range(7):
            server.send(connection, type="heartbeat")
            time.sleep(.2)
        server.ready(connection, request)
    with Coordinator(respond) as server:
        client = bridge(server, timeout_seconds=1, heartbeat_wait=True)
        with ThreadPoolExecutor(1) as pool:
            future = pool.submit(wait, client)
            assert received.wait(2)
            time.sleep(1.1)
            assert not future.done()
            future.result(timeout=3)
        client.cancel()


@pytest.mark.parametrize("response,match", [
    ({"type":"error","message":"publication failed"}, "publication failed"),
    ({"type":"ready","request_id":999}, "unknown live request"),
    ({"type":"surprise"}, "invalid live response"),
])
def test_errors_wake_waiters(response, match):
    with Coordinator(lambda s,c,r: s.send(c, **response)) as server:
        client = bridge(server)
        with pytest.raises(RuntimeError, match=match):
            wait(client)
        with pytest.raises(RuntimeError):
            client.check_active()


def test_timeout_and_cancellation():
    with Coordinator(lambda s,c,r: None) as server:
        client = bridge(server, timeout_seconds=1)
        with pytest.raises(RuntimeError, match="timed out"):
            wait(client)
    with Coordinator(lambda s,c,r: None) as server:
        client = bridge(server)
        with ThreadPoolExecutor(1) as pool:
            future = pool.submit(wait, client)
            client.cancel("stop now")
            with pytest.raises(RuntimeError, match="stop now"):
                future.result(timeout=2)


def test_disconnect_and_invalid_requests():
    with Coordinator(lambda s,c,r: c.shutdown(socket.SHUT_RDWR)) as server:
        client = bridge(server)
        with pytest.raises(ValueError, match="sizes differ"):
            client.wait("source", 0, 1, ["a"], [])
        with pytest.raises(ValueError, match="source"):
            client.wait("", 0, 1, [], [])
        with pytest.raises(ValueError, match="finite"):
            client.wait("source", 0, float("nan"), [], [])
        with pytest.raises(RuntimeError, match="disconnected"):
            wait(client)
    with pytest.raises(ValueError, match="invalid live inference endpoint"):
        ft.LiveBridge(0, "token")


def test_empty_cancellation_reason_still_fails():
    with Coordinator(lambda s,c,r: s.ready(c,r)) as server:
        client = bridge(server)
        client.cancel("")
        with pytest.raises(RuntimeError, match="cancelled"):
            client.check_active()


def test_field_publication_and_bridge_ownership(inputs):
    offline, normals = handles(inputs)
    cfg, chunks = trace_config(), chunk_config()
    expected = ft.ChunkCache(offline, normals, cfg, chunks, str(inputs/"offline"), "fixture").get([0,0,0])
    with Coordinator(lambda s,c,r: s.ready(c,r)) as server:
        field, normals = handles(inputs)
        client = bridge(server)
        field.set_live_bridge(client, "fixture")
        del client
        gc.collect()
        cache = ft.ChunkCache(field, normals, cfg, chunks, str(inputs/"live"), "fixture")
        actual = cache.get([0,0,0])
        assert server.requests
        arrays = {c["array"] for r in server.requests for c in r["chunks"]}
        assert any("width" in path for path in arrays)
        for a,b in zip(expected.lines_base, actual.lines_base, strict=True):
            np.testing.assert_array_equal(a,b)
        del cache, field
        gc.collect()


def test_one_worker_yields_publication_and_duplicate_chunk_waits(inputs):
    if not ft.case_queue_supported():
        pytest.skip("native case contexts unavailable")
    def respond(server, connection, request):
        # The first build waits; its duplicate waits on the same cache future.
        # A third case must still run on this ONE OS worker to publish both.
        if len(server.requests) == 2:
            for req in reversed(server.requests):
                server.ready(connection, req)
        elif len(server.requests) > 2:
            server.ready(connection, request)
    with Coordinator(respond) as server:
        client = bridge(server, timeout_seconds=3)
        field, normals = handles(inputs)
        field.set_live_bridge(client, "fixture")
        cfg = chunk_config()
        cache = ft.ChunkCache(field, normals, trace_config(), cfg, str(inputs/"queued"), "queue")
        # Different construction keys but the same actual prediction storage.
        # This fixture has one storage chunk, so simultaneous demands prove
        # readiness isn't cached before the acknowledgement.
        result = cache.get_many([[0,0,0],[0,0,0],[-1,0,0]], workers=1, contexts_per_worker=3)
        assert len(server.requests) >= 2
        assert len(result)==3
        assert result[0].key==result[1].key
        assert result[2].key==[-1,0,0]
        for a,b in zip(result[0].lines_base,result[1].lines_base,strict=True):
            np.testing.assert_array_equal(a,b)
        client.cancel()


def test_queued_publication_failure_cancels_siblings(inputs):
    if not ft.case_queue_supported():
        pytest.skip("native case contexts unavailable")
    with Coordinator(lambda s,c,r: s.send(c,type="error",message="inference failed")) as server:
        client=bridge(server)
        field,normals=handles(inputs)
        field.set_live_bridge(client,"fixture")
        cache=ft.ChunkCache(field,normals,trace_config(),chunk_config(),str(inputs/"queue-fail"),"queue")
        with pytest.raises(RuntimeError,match="inference failed"):
            cache.get_many([[0,0,0]]*10,workers=2,contexts_per_worker=2)
        client.cancel()


def test_cached_paris4_live_replay(tmp_path):
    from test_fiber_chunk_trace import test_real_paris4_chunk_replay
    if not os.environ.get("VC_TEST_WIDTH_MANIFEST"):
        pytest.skip("set cached Paris4 fixture paths")
    original = ft.open_prediction_field
    with Coordinator(lambda s,c,r: s.ready(c,r)) as server:
        client = bridge(server)
        def opened(*args, **kwargs):
            field = original(*args, **kwargs)
            field.set_live_bridge(client, "paris4-cached")
            return field
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(ft, "open_prediction_field", opened)
            test_real_paris4_chunk_replay(tmp_path)
        assert server.requests
        client.cancel()
