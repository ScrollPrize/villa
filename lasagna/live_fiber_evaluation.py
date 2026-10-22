"""Publication coordinator connecting callable evaluation to demand inference.

Stores are borrowed, exclusively for this call. A lazy context-manager factory
owns the producer (and any GPU lease) and yields its nonblocking pump callback.
No model construction, GPU selection or resource policy is hidden here.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
import copy
import json
import math
from pathlib import Path
import secrets
import selectors
import socket
import threading
import time


class DemandPublicationSession:
    """One authenticated native bridge connection; coordinator owns store access.

    ``sources`` maps source IDs to DemandStore objects. Array allowlists come
    from those stores' products, never from untrusted request paths. On a cache
    miss ``producer_factory()`` enters once and yields ``pump()``; it must
    close/join workers before releasing its resource grant. Stores stay open
    until this session closes. Never pump or mutate them concurrently.
    """

    def __init__(self, sources, producer_factory, *, timeout_seconds=600):
        if not sources or any(not isinstance(k, str) or not k for k in sources):
            raise ValueError("nonempty source IDs required")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive and finite")
        self.sources = dict(sources)
        self.arrays = {key: {str((Path(c.relative_path) / str(p.level)).resolve())
                            for p in store.products for c in p.channels}
                       for key, store in self.sources.items()}
        self.factory = producer_factory
        self.timeout = timeout_seconds
        self.stop = threading.Event()
        self.error = None
        self.thread = None
        self.listener = None
        self.connection = None

    def __enter__(self):
        if self.thread is not None:
            raise RuntimeError("publication sessions are single-use")
        self.listener = socket.socket()
        try:
            self.listener.bind(("127.0.0.1", 0))
            self.listener.listen(1)
            self.listener.setblocking(False)
            self.endpoint = dict(port=self.listener.getsockname()[1], token=secrets.token_hex(32),
                                 timeout_seconds=self.timeout, heartbeat_wait=True)
            self.thread = threading.Thread(target=self._serve, name="fiber-publication")
            self.thread.start()
        except BaseException:
            self.listener.close()
            raise
        return self

    @staticmethod
    def _send(connection, message):
        connection.sendall((json.dumps(dict(version=1, **message), allow_nan=False) + "\n").encode())

    def _targets(self, message):
        request_id = message.get("request_id")
        if message.get("type") != "demand" or type(request_id) is not int or request_id < 0:
            raise ValueError("invalid demand request")
        source = message.get("source")
        if not isinstance(source, str) or source not in self.sources:
            raise ValueError("unknown prediction source")
        remaining = message.get("remaining_base_voxels")
        if not isinstance(remaining, (int, float)) or not math.isfinite(remaining) or remaining < 0:
            raise ValueError("invalid remaining distance")
        chunks = message.get("chunks")
        if not isinstance(chunks, list) or not chunks:
            raise ValueError("nonempty chunk dependencies required")
        store = self.sources[source]
        indices = set()
        # Validate the whole request before scheduling any work.
        for chunk in chunks:
            if not isinstance(chunk, dict) or not isinstance(chunk.get("array"), str):
                raise ValueError("invalid chunk dependency")
            if str(Path(chunk["array"]).resolve()) not in self.arrays[source]:
                raise ValueError("array does not belong to prediction source")
            index = chunk.get("index_zyx")
            if not isinstance(index, list) or len(index) != 3 or any(type(v) is not int or v < 0 for v in index):
                raise ValueError("invalid chunk index")
            if any(v * store.geometry.chunk >= size for v, size in zip(index, store.geometry.output_shape)):
                raise ValueError("chunk index outside prediction volume")
            indices.add(tuple(index))
        return request_id, [(store, store.request(i, remaining=remaining)) for i in sorted(indices)]

    @contextmanager
    def _resources(self):
        with ExitStack() as cleanup:
            try:
                yield cleanup
            except BaseException as error:
                self.error = error
                if self.connection is not None:
                    try:
                        self._send(self.connection, dict(type="error", message=str(error)))
                    except OSError:
                        pass
            finally:
                # A native cancellation closes the socket before every case has
                # unwound. Only the caller's context exit certifies their join.
                # Signal errors above BEFORE waiting, so consumers can unwind.
                self.stop.wait()

    def _serve(self):
        connection = None
        try:
            with self._resources() as cleanup:
                selector = cleanup.enter_context(selectors.DefaultSelector())
                selector.register(self.listener, selectors.EVENT_READ)
                pending, waiters, seen = b"", {}, set()
                authenticated = False
                pump = None
                last_heartbeat = time.monotonic()
                while not self.stop.is_set():
                    for key, _ in selector.select(.005):
                        if key.fileobj is self.listener:
                            connection, _ = self.listener.accept()
                            self.connection = connection
                            connection.settimeout(1)
                            selector.unregister(self.listener)
                            selector.register(connection, selectors.EVENT_READ)
                            continue
                        received = connection.recv(65536)
                        if not received:
                            if waiters or not authenticated:
                                raise RuntimeError("evaluator disconnected with unfinished requests")
                            return
                        pending += received
                        while b"\n" in pending:
                            line, pending = pending.split(b"\n", 1)
                            if len(line) > 1 << 20:
                                raise ValueError("oversized demand message")
                            message = json.loads(line)
                            if not isinstance(message, dict) or message.get("version") != 1:
                                raise ValueError("unsupported publication protocol")
                            if not authenticated:
                                token = message.get("token")
                                if message.get("type") != "hello" or not isinstance(token, str) or not secrets.compare_digest(token, self.endpoint["token"]):
                                    raise ValueError("invalid publication authentication")
                                authenticated = True
                                continue
                            # Concurrent callers can send IDs out of order.
                            identity = message.get("request_id")
                            if type(identity) is not int or identity in seen:
                                raise ValueError("duplicate/invalid request ID")
                            identity, targets = self._targets(message)
                            if len(waiters) >= 4096:
                                raise ValueError("too many outstanding publication requests")
                            waiters[identity] = targets
                            seen.add(identity)
                        if len(pending) > 1 << 20:
                            raise ValueError("oversized demand message")
                    if authenticated:
                        if pump is None and any(s.pending_tile_key() is not None for s in self.sources.values()):
                            if self.factory is None:
                                raise RuntimeError("prediction cache miss without an inference producer")
                            pump = cleanup.enter_context(self.factory())
                            if not callable(pump):
                                raise TypeError("producer context must yield a pump callback")
                        if pump is not None:
                            pump()
                        for identity, targets in list(waiters.items()):
                            if all([s.ready(origin) for s, origin in targets]):
                                self._send(connection, dict(type="ready", request_id=identity))
                                del waiters[identity]
                        if time.monotonic() - last_heartbeat >= 1:
                            self._send(connection, dict(type="heartbeat"))
                            last_heartbeat = time.monotonic()
        except BaseException as error:
            self.error = error
            if connection is not None:
                try:
                    self._send(connection, dict(type="error", message=str(error)))
                except OSError:
                    pass
        finally:
            if connection is not None:
                connection.close()

    def __exit__(self, kind, error, traceback):
        self.stop.set()
        self.thread.join()
        self.listener.close()
        if error is None and self.error is not None:
            raise self.error


def evaluate_live(config, sources, producer_factory, *, source, **options):
    """Run callable evaluation with a scoped publication/producer session.

    CPU consumers finish before producer teardown; no temporary config files,
    subprocess evaluator, live config reload or model replacement. The factory
    must return promptly (use nonblocking manager polling in its pump callback).
    Already-ready stores never start the producer or request GPU resources.
    """
    from vc.fiber_evaluation import evaluate

    if source not in sources:
        raise ValueError("evaluation source not registered")
    if "live_endpoint" in config:
        raise ValueError("evaluate_live owns its publication endpoint")
    cfg = copy.deepcopy(config)
    with DemandPublicationSession(sources, producer_factory) as session:
        cfg["live_endpoint"] = dict(session.endpoint, source=source)
        return evaluate(cfg, **options)
