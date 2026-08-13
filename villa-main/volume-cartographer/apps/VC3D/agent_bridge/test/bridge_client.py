"""Minimal JSON-RPC 2.0 client for the VC3D Agent Bridge.

Speaks the newline-delimited JSON-RPC-over-QLocalSocket protocol described in
apps/VC3D/agent_bridge/SPEC.md. On POSIX, QLocalServer is backed by a Unix
domain socket, so a plain AF_UNIX SOCK_STREAM client works directly -- no Qt
dependency needed on the test side.
"""
from __future__ import annotations

import itertools
import json
import queue
import socket
import threading
import time
from typing import Any, Callable, Optional


class BridgeError(Exception):
    def __init__(self, code: int, message: str, data: Optional[dict] = None):
        super().__init__(f"[{code}] {message} {data if data else ''}".strip())
        self.code = code
        self.message = message
        self.data = data or {}


class BridgeClient:
    def __init__(self, sock_path: str, connect_timeout: float = 10.0):
        self.sock_path = sock_path
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(connect_timeout)
        self.sock.connect(sock_path)
        self.sock.settimeout(None)

        self._id_counter = itertools.count(1)
        self._pending: dict[int, "queue.Queue[dict]"] = {}
        self._pending_lock = threading.Lock()
        self.notifications: "queue.Queue[dict]" = queue.Queue()
        self._send_lock = threading.Lock()
        self._closed = False
        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()

    def _reader_loop(self) -> None:
        f = self.sock.makefile("rb")
        try:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                if isinstance(msg, dict) and "id" in msg and ("result" in msg or "error" in msg):
                    mid = msg["id"]
                    with self._pending_lock:
                        q = self._pending.get(mid)
                    if q is not None:
                        q.put(msg)
                elif isinstance(msg, dict) and "method" in msg and "id" not in msg:
                    self.notifications.put(msg)
        except OSError:
            pass
        finally:
            self._closed = True

    def call(self, method: str, params: Optional[dict] = None, timeout: float = 15.0):
        """Sends a JSON-RPC request and blocks for the matching response.

        Returns (result_dict, elapsed_seconds). Raises BridgeError on a
        JSON-RPC error response, TimeoutError if no response arrives in time.
        """
        mid = next(self._id_counter)
        req: dict[str, Any] = {"jsonrpc": "2.0", "id": mid, "method": method}
        if params is not None:
            req["params"] = params

        q: "queue.Queue[dict]" = queue.Queue()
        with self._pending_lock:
            self._pending[mid] = q

        payload = (json.dumps(req) + "\n").encode("utf-8")
        t0 = time.monotonic()
        with self._send_lock:
            self.sock.sendall(payload)
        try:
            msg = q.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError(f"RPC '{method}' (id={mid}) timed out after {timeout}s")
        finally:
            with self._pending_lock:
                self._pending.pop(mid, None)

        elapsed = time.monotonic() - t0
        if "error" in msg:
            e = msg["error"]
            raise BridgeError(e.get("code", -1), e.get("message", ""), e.get("data"))
        return msg.get("result", {}), elapsed

    def wait_for_notification(self, method: Optional[str] = None,
                               predicate: Optional[Callable[[dict], bool]] = None,
                               timeout: float = 30.0) -> dict:
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"waiting for notification method={method} timed out")
            try:
                msg = self.notifications.get(timeout=remaining)
            except queue.Empty:
                raise TimeoutError(f"waiting for notification method={method} timed out")
            if method is not None and msg.get("method") != method:
                continue
            params = msg.get("params", {})
            if predicate is not None and not predicate(params):
                continue
            return params

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass


def parse_handshake_line(line: str) -> Optional[str]:
    """Parses 'VC3D-AGENT-BRIDGE: listening name=<n> path=<p>' -> path, or None."""
    marker = "VC3D-AGENT-BRIDGE: listening "
    idx = line.find(marker)
    if idx == -1:
        return None
    rest = line[idx + len(marker):].strip()
    path_key = "path="
    pidx = rest.find(path_key)
    if pidx == -1:
        return None
    return rest[pidx + len(path_key):].strip()
