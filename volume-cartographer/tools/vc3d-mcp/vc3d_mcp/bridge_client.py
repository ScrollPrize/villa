"""Thin asyncio JSON-RPC 2.0 client for the VC3D Agent Bridge."""

from __future__ import annotations

import asyncio
import json
import os
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from .transport import (
    BridgeReader,
    BridgeWriter,
    open_local_connection,
    resolve_local_endpoint,
)

HANDSHAKE_RE = re.compile(
    r"^VC3D-AGENT-BRIDGE:\s*listening\s+name=(?P<name>\S+)\s+path=(?P<path>.+)$"
)

REGISTRY_DIR = os.path.join(os.path.expanduser("~"), ".vc3d", "agent_bridge")
PROGRESS_QUEUE_SIZE = 64


@dataclass(frozen=True)
class RegistryEntry:
    endpoint: str
    file_path: str
    started_at: float


def discover_registry_entries(registry_dir: str = REGISTRY_DIR) -> list[RegistryEntry]:
    """Return valid registry records, newest first.

    Endpoint connection is the liveness check. It is authoritative, avoids PID
    reuse races, and does not rely on Unix signal behavior.
    """
    if not os.path.isdir(registry_dir):
        return []

    entries: list[RegistryEntry] = []
    for name in os.listdir(registry_dir):
        if not name.endswith(".json"):
            continue
        file_path = os.path.join(registry_dir, name)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except (OSError, json.JSONDecodeError):
            remove_registry_entry(file_path)
            continue

        if not isinstance(obj, dict):
            remove_registry_entry(file_path)
            continue

        path = obj.get("path")
        if not isinstance(path, str) or not path:
            remove_registry_entry(file_path)
            continue

        started_at = obj.get("startedAt")
        started_at = float(started_at) if isinstance(started_at, (int, float)) else 0.0
        if started_at < 0:
            started_at = 0.0
        if started_at <= 0:
            try:
                started_at = os.path.getmtime(file_path)
            except OSError:
                started_at = 0.0

        entries.append(RegistryEntry(path, file_path, started_at))

    return sorted(entries, key=lambda entry: entry.started_at, reverse=True)


def remove_registry_entry(file_path: str) -> None:
    try:
        os.unlink(file_path)
    except OSError:
        pass


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
    # A 2048x2048 RGBA PNG can approach 16 MiB before base64 encoding. Leave
    # enough room for that inline response and its JSON envelope.
    read_buffer_limit: int = 32 * 1024 * 1024


@dataclass
class _Conn:
    reader: BridgeReader
    writer: BridgeWriter
    pending: dict[int, "asyncio.Future[Any]"] = field(default_factory=dict)
    read_task: asyncio.Task | None = None


class BridgeClient:
    """Connects to one VC3D Agent Bridge socket and speaks its JSON-RPC 2.0.

    Reconnects lazily: if the connection drops, in-flight calls fail and the
    next call opens a fresh one. All per-connection state lives in a _Conn,
    so a dying connection can only ever tear down itself.
    """

    def __init__(self, config: BridgeClientConfig):
        self._config = config
        self._conn: _Conn | None = None
        self._connect_lock = asyncio.Lock()
        self._write_lock = asyncio.Lock()
        self._next_id = 1
        self._closed = False
        self._job_progress: dict[str, set[asyncio.Queue[dict[str, Any]]]] = {}

    @staticmethod
    def resolve_socket_path(configured: str) -> str:
        """Turn a configured QLocalServer name or path into a native endpoint."""
        return resolve_local_endpoint(configured)

    @staticmethod
    def socket_path_from_handshake(line: str) -> tuple[str, str] | None:
        """Parse the `VC3D-AGENT-BRIDGE: listening name=... path=...` line.

        Returns ``(name, path)`` or ``None`` if the line doesn't match.
        """
        m = HANDSHAKE_RE.match(line.strip())
        if not m:
            return None
        return m.group("name"), m.group("path")

    @property
    def connected(self) -> bool:
        return self._conn is not None and not self._conn.writer.is_closing()

    async def _ensure_conn(self) -> _Conn:
        if self._closed:
            raise BridgeConnectionError("bridge client has been closed")
        conn = self._conn
        if conn is not None and not conn.writer.is_closing():
            return conn
        async with self._connect_lock:
            if self._closed:
                raise BridgeConnectionError("bridge client has been closed")
            conn = self._conn
            if conn is not None and not conn.writer.is_closing():
                return conn
            path = self.resolve_socket_path(self._config.socket)
            try:
                reader, writer = await open_local_connection(
                    path,
                    limit=self._config.read_buffer_limit,
                    connect_timeout=self._config.connect_timeout,
                )
            except (OSError, asyncio.TimeoutError) as exc:
                raise BridgeConnectionError(
                    f"could not connect to VC3D agent bridge at {path!r} "
                    f"(configured as {self._config.socket!r}): {exc}"
                ) from exc
            conn = _Conn(reader=reader, writer=writer)
            conn.read_task = asyncio.create_task(
                self._read_loop(conn), name="vc3d-bridge-reader"
            )
            self._conn = conn
            return conn

    async def connect(self) -> None:
        await self._ensure_conn()

    @asynccontextmanager
    async def subscribe_job_progress(
        self, job_id: str
    ) -> AsyncIterator[asyncio.Queue[dict[str, Any]]]:
        """Yield a bounded queue of best-effort progress for one job."""
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(
            maxsize=PROGRESS_QUEUE_SIZE
        )
        subscribers = self._job_progress.setdefault(job_id, set())
        subscribers.add(queue)
        try:
            yield queue
        finally:
            subscribers.discard(queue)
            if not subscribers:
                self._job_progress.pop(job_id, None)

    async def close(self) -> None:
        self._closed = True
        async with self._connect_lock:
            conn, self._conn = self._conn, None
            if conn is None:
                return
            if conn.read_task is not None:
                conn.read_task.cancel()
            self._drop(conn, BridgeConnectionError("bridge client has been closed"))
            try:
                await conn.writer.wait_closed()
            except Exception:  # noqa: BLE001 - closing a dead writer can raise
                pass

    def _drop(self, conn: _Conn, exc: Exception) -> None:
        """Fail `conn`'s pending calls, close its writer, forget it if current."""
        if self._conn is conn:
            self._conn = None
        for fut in conn.pending.values():
            if not fut.done():
                fut.set_exception(exc)
        conn.pending.clear()
        if not conn.writer.is_closing():
            conn.writer.close()

    async def call(
        self, method: str, params: dict[str, Any] | None = None, timeout: float | None = None
    ) -> Any:
        """Send a JSON-RPC request and await its response.

        Raises BridgeError for a JSON-RPC error reply, BridgeConnectionError
        on transport failure or timeout.
        """
        conn = await self._ensure_conn()
        req_id = self._next_id
        self._next_id += 1
        message: dict[str, Any] = {"jsonrpc": "2.0", "id": req_id, "method": method}
        if params is not None:
            message["params"] = params
        fut: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        conn.pending[req_id] = fut
        line = json.dumps(message, separators=(",", ":")) + "\n"
        try:
            async with self._write_lock:
                if conn.writer.is_closing():
                    raise BridgeConnectionError(
                        "bridge connection closed before the request could be sent"
                    )
                try:
                    conn.writer.write(line.encode("utf-8"))
                    await conn.writer.drain()
                except OSError as exc:
                    self._drop(conn, BridgeConnectionError(f"write failed: {exc}"))
                    raise BridgeConnectionError(f"write failed: {exc}") from exc
            try:
                return await asyncio.wait_for(
                    fut, timeout=timeout if timeout is not None else self._config.request_timeout
                )
            except asyncio.TimeoutError as exc:
                raise BridgeConnectionError(
                    f"timed out waiting for response to {method!r} (id={req_id})"
                ) from exc
        finally:
            # Any exit -- success, timeout, cancellation while queued for the
            # write lock or while awaiting the reply -- removes the entry.
            conn.pending.pop(req_id, None)
            # A transport failure may have resolved fut via _drop while we raised
            # our own error instead of awaiting it; retrieve the exception so
            # asyncio does not warn that it went unobserved.
            if fut.done() and not fut.cancelled():
                fut.exception()

    async def _read_loop(self, conn: _Conn) -> None:
        try:
            while True:
                line = await conn.reader.readline()
                if not line:
                    break
                line = line.strip()
                if line:
                    self._dispatch_line(conn, line)
        except asyncio.CancelledError:
            return  # close() owns teardown
        except Exception as exc:  # noqa: BLE001 - fail this conn's waiters
            self._drop(conn, BridgeConnectionError(f"reader loop crashed: {exc}"))
            return
        self._drop(conn, BridgeConnectionError("connection closed by peer"))

    def _dispatch_line(self, conn: _Conn, raw: bytes) -> None:
        try:
            msg = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return  # malformed line from the peer; nothing sane to do with it
        if not isinstance(msg, dict):
            return
        if msg.get("id") is None:
            if msg.get("method") == "job.progress":
                params = msg.get("params")
                job_id = params.get("jobId") if isinstance(params, dict) else None
                if isinstance(job_id, str):
                    for queue in tuple(self._job_progress.get(job_id, ())):
                        if queue.full():
                            queue.get_nowait()
                        queue.put_nowait(params)
            return
        fut = conn.pending.pop(msg["id"], None)
        if fut is None or fut.done():
            return
        if "error" in msg:
            err = msg["error"] or {}
            fut.set_exception(BridgeError(
                code=err.get("code", -32000),
                message=err.get("message", "unknown error"),
                data=err.get("data"),
            ))
        else:
            fut.set_result(msg.get("result"))
