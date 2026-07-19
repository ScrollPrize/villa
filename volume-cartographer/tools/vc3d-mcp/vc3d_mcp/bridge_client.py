"""Thin asyncio client for the VC3D Agent Bridge JSON-RPC-over-local-socket channel.

Speaks exactly the wire protocol described in
``apps/VC3D/agent_bridge/SPEC.md`` section 1.2:

- one JSON-RPC 2.0 message per line, UTF-8, LF-terminated;
- requests get a numeric ``id`` and are answered 1:1, in the order the
  bridge chooses to answer them (the bridge processes them serially, but we
  do not assume that here -- responses are correlated by ``id``, not by
  send order);
- messages without an ``id`` are server -> client notifications (only
  ``job.progress`` exists today) and are broadcast to every connected
  client. This module tees them into a :class:`~vc3d_mcp.jobs.JobTracker`.

The bridge's socket is a Qt ``QLocalServer``. On Unix (Linux/macOS -- the
only platforms VC3D currently ships for) that is a plain ``AF_UNIX`` stream
socket bound to a filesystem path, so a stdlib ``asyncio.open_unix_connection``
is sufficient; no Qt/QLocalSocket dependency is needed on the Python side.

Socket resolution (see README.md "Socket discovery" for the full writeup):

1. If the configured value is an absolute/relative path that exists on
   disk, connect to it directly as an AF_UNIX socket.
2. Otherwise treat it as a bare ``QLocalServer`` name (what
   ``--agent-bridge-name`` takes) and probe the handful of locations Qt is
   known to place local-server sockets under on this platform
   (``$TMPDIR/<name>``, ``/tmp/<name>``). First one that exists wins.
3. If VC3D was spawned by us (not exercised yet -- see SPEC.md section
   5, "spawns VC3D itself"), ``BridgeClient.socket_path_from_handshake``
   extracts the authoritative path out of the
   ``VC3D-AGENT-BRIDGE: listening name=... path=...`` stdout line instead
   of guessing.

SPEC.md does not pin down exact Qt local-socket filesystem conventions
(they are platform- and Qt-version-dependent), so step 2 is a best-effort
fallback documented here as an explicit assumption; step 1 (an explicit
full path, e.g. forwarded via ``VC3D_AGENT_BRIDGE_SOCKET``) is the
recommended, unambiguous way to point this client at a running bridge.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

from .jobs import JobTracker

HANDSHAKE_RE = re.compile(
    r"^VC3D-AGENT-BRIDGE:\s*listening\s+name=(?P<name>\S+)\s+path=(?P<path>.+)$"
)


class BridgeError(Exception):
    """A JSON-RPC error object returned by the bridge (SPEC.md section 2.5)."""

    def __init__(self, code: int, message: str, data: Any = None):
        self.code = code
        self.message = message
        self.data = data
        super().__init__(message)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.data is not None:
            d["data"] = self.data
        return d

    def __str__(self) -> str:  # noqa: D401 - used verbatim as MCP error text
        return json.dumps(self.to_dict())


class BridgeConnectionError(Exception):
    """Raised when the socket can't be reached / drops / times out."""


@dataclass
class BridgeClientConfig:
    socket: str
    connect_timeout: float = 5.0
    request_timeout: float = 30.0
    # asyncio.StreamReader's default line-length limit (64 KiB) is comfortably
    # above a single bridge message (screenshots go through filePath/base64
    # but base64 payloads can still be large); raise it generously.
    read_buffer_limit: int = 8 * 1024 * 1024


@dataclass
class _Pending:
    future: "asyncio.Future[Any]"


class BridgeClient:
    """Connects to one VC3D Agent Bridge socket and speaks its JSON-RPC 2.0."""

    def __init__(self, config: BridgeClientConfig):
        self._config = config
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._read_task: asyncio.Task | None = None
        self._pending: dict[int, _Pending] = {}
        self._next_id = 1
        self._write_lock = asyncio.Lock()
        self.jobs = JobTracker()
        self._connected_event = asyncio.Event()
        self._closed = False

    # -- connection lifecycle -------------------------------------------------

    @staticmethod
    def resolve_socket_path(configured: str) -> str:
        """Turn a configured `--socket`/env-var value into a concrete path.

        See the module docstring's "Socket resolution" writeup.
        """
        if os.path.exists(configured):
            return configured

        candidates = []
        tmpdir = os.environ.get("TMPDIR")
        if tmpdir:
            candidates.append(os.path.join(tmpdir.rstrip("/"), configured))
        candidates.append(os.path.join("/tmp", configured))
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate

        # Nothing found on disk. Return the configured value unmodified so the
        # eventual connect() failure carries the name the caller actually
        # asked for, rather than a synthesized candidate that's confusing to
        # read in an error message.
        return configured

    @staticmethod
    def socket_path_from_handshake(line: str) -> tuple[str, str] | None:
        """Parse the `VC3D-AGENT-BRIDGE: listening name=... path=...` line.

        Returns ``(name, path)`` or ``None`` if the line doesn't match.
        """
        m = HANDSHAKE_RE.match(line.strip())
        if not m:
            return None
        return m.group("name"), m.group("path")

    async def connect(self) -> None:
        path = self.resolve_socket_path(self._config.socket)
        try:
            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_unix_connection(path, limit=self._config.read_buffer_limit),
                timeout=self._config.connect_timeout,
            )
        except (OSError, asyncio.TimeoutError) as exc:
            raise BridgeConnectionError(
                f"could not connect to VC3D agent bridge at {path!r} "
                f"(configured as {self._config.socket!r}): {exc}"
            ) from exc
        self._closed = False
        self._read_task = asyncio.create_task(self._read_loop(), name="vc3d-bridge-reader")
        self._connected_event.set()

    async def ensure_connected(self) -> None:
        if self._writer is None or self._writer.is_closing():
            await self.connect()

    async def close(self) -> None:
        self._closed = True
        if self._read_task is not None:
            self._read_task.cancel()
        if self._writer is not None:
            self._writer.close()
        for pending in self._pending.values():
            if not pending.future.done():
                pending.future.set_exception(BridgeConnectionError("connection closed"))
        self._pending.clear()

    @property
    def connected(self) -> bool:
        return self._writer is not None and not self._writer.is_closing()

    # -- request/response -----------------------------------------------------

    async def call(
        self, method: str, params: dict[str, Any] | None = None, timeout: float | None = None
    ) -> Any:
        """Send a JSON-RPC request and await its response.

        Raises :class:`BridgeError` for a JSON-RPC error reply, or
        :class:`BridgeConnectionError` on transport failure/timeout.
        """
        await self.ensure_connected()
        assert self._writer is not None

        req_id = self._next_id
        self._next_id += 1
        message: dict[str, Any] = {"jsonrpc": "2.0", "id": req_id, "method": method}
        if params is not None:
            message["params"] = params

        loop = asyncio.get_running_loop()
        fut: asyncio.Future[Any] = loop.create_future()
        self._pending[req_id] = _Pending(future=fut)

        line = json.dumps(message, separators=(",", ":")) + "\n"
        async with self._write_lock:
            try:
                self._writer.write(line.encode("utf-8"))
                await self._writer.drain()
            except OSError as exc:
                self._pending.pop(req_id, None)
                raise BridgeConnectionError(f"write failed: {exc}") from exc

        try:
            return await asyncio.wait_for(fut, timeout=timeout or self._config.request_timeout)
        except asyncio.TimeoutError as exc:
            self._pending.pop(req_id, None)
            raise BridgeConnectionError(
                f"timed out waiting for response to {method!r} (id={req_id})"
            ) from exc

    # -- background reader -----------------------------------------------------

    async def _read_loop(self) -> None:
        assert self._reader is not None
        try:
            while True:
                line = await self._reader.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                self._dispatch_line(line)
        except asyncio.CancelledError:
            pass
        except Exception as exc:  # noqa: BLE001 - surface to all waiters below
            self._fail_all_pending(BridgeConnectionError(f"reader loop crashed: {exc}"))
            return
        self._fail_all_pending(BridgeConnectionError("connection closed by peer"))

    def _fail_all_pending(self, exc: Exception) -> None:
        for pending in self._pending.values():
            if not pending.future.done():
                pending.future.set_exception(exc)
        self._pending.clear()

    def _dispatch_line(self, raw: bytes) -> None:
        try:
            msg = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return  # malformed line from the peer; nothing sane to do with it

        if not isinstance(msg, dict):
            return

        if "id" in msg and msg["id"] is not None:
            req_id = msg["id"]
            pending = self._pending.pop(req_id, None)
            if pending is None:
                return
            if "error" in msg:
                err = msg["error"] or {}
                pending.future.set_exception(
                    BridgeError(
                        code=err.get("code", -32000),
                        message=err.get("message", "unknown error"),
                        data=err.get("data"),
                    )
                )
            else:
                pending.future.set_result(msg.get("result"))
            return

        # Notification (no id).
        method = msg.get("method")
        params = msg.get("params") or {}
        if method == "job.progress":
            self.jobs.on_progress(params)
