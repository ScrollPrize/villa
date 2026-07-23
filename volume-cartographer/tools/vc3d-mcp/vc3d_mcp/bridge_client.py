"""Thin asyncio JSON-RPC 2.0 client for the VC3D Agent Bridge AF_UNIX socket.

Socket resolution (see README.md "Socket discovery" for the full writeup):

1. If the configured value is an absolute/relative path that exists on
   disk, connect to it directly as an AF_UNIX socket.
2. Otherwise treat it as a bare ``QLocalServer`` name (what
   ``--agent-bridge-name`` takes) and probe the handful of locations Qt is
   known to place local-server sockets under on this platform
   (``$TMPDIR/<name>``, ``/tmp/<name>``). First one that exists wins.
3. If VC3D was spawned by us (the auto-launch path in ``server.py``),
   ``BridgeClient.socket_path_from_handshake`` extracts the authoritative
   path out of the ``VC3D-AGENT-BRIDGE: listening name=... path=...`` stdout
   line instead of guessing.

Step 1 (an explicit full path, e.g. forwarded via ``VC3D_AGENT_BRIDGE_SOCKET``)
is the unambiguous option; step 2 is a best-effort fallback.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

HANDSHAKE_RE = re.compile(
    r"^VC3D-AGENT-BRIDGE:\s*listening\s+name=(?P<name>\S+)\s+path=(?P<path>.+)$"
)

# Where a running VC3D publishes its bridge for auto-discovery. Mirrors the
# ~/.fit_services convention used by LasagnaServiceManager::discoverServices():
# one JSON file per process ("<pid>.json") holding {pid, name, path, startedAt},
# with dead-PID entries reaped on scan. See AgentBridgeServer::writeRegistryFile.
REGISTRY_DIR = os.path.join(os.path.expanduser("~"), ".vc3d", "agent_bridge")


def _pid_alive(pid: int) -> bool:
    """True if a process with `pid` currently exists (Unix ``kill(pid, 0)``).

    Mirrors the ``kill(pid, 0) != 0 -> stale`` liveness check in
    ``LasagnaServiceManager::discoverServices()``. A ``PermissionError`` means
    the process exists but is owned by someone else -- still alive.
    """
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def discover_registry_socket(registry_dir: str = REGISTRY_DIR) -> str | None:
    """Scan the bridge registry for the newest live bridge's socket path.

    Reads every ``*.json`` file in ``registry_dir``, dropping (and deleting)
    entries whose ``pid`` is no longer alive -- the same stale-cleanup behavior
    as ``LasagnaServiceManager::discoverServices()``. Returns the ``path`` of
    the live entry with the greatest ``startedAt`` (newest launch), or ``None``
    when the directory is absent or holds no live entry.
    """
    if not os.path.isdir(registry_dir):
        return None

    best_path: str | None = None
    best_started_at = -1.0
    for name in os.listdir(registry_dir):
        if not name.endswith(".json"):
            continue
        file_path = os.path.join(registry_dir, name)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except (OSError, json.JSONDecodeError):
            # Unreadable / malformed entry: reap it, matching the C++ scanner
            # which removes non-object files.
            _remove_registry_file(file_path)
            continue

        if not isinstance(obj, dict):
            _remove_registry_file(file_path)
            continue

        pid = obj.get("pid")
        pid = int(pid) if isinstance(pid, (int, float)) else -1
        if pid <= 0:
            _remove_registry_file(file_path)
            continue

        if not _pid_alive(pid):
            _remove_registry_file(file_path)
            continue

        path = obj.get("path")
        if not isinstance(path, str) or not path:
            continue

        started_at = obj.get("startedAt")
        started_at = float(started_at) if isinstance(started_at, (int, float)) else 0.0
        # Tie-break by file mtime so entries lacking a usable startedAt still
        # order sensibly; startedAt is the primary key.
        if started_at < 0:
            started_at = 0.0
        key = started_at
        if key <= 0:
            try:
                key = os.path.getmtime(file_path)
            except OSError:
                key = 0.0

        if key > best_started_at:
            best_started_at = key
            best_path = path

    return best_path


def _remove_registry_file(file_path: str) -> None:
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
    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter
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

        # Nothing found on disk; return unmodified so a failed connect() names
        # what the caller actually asked for, not a synthesized candidate.
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
                reader, writer = await asyncio.wait_for(
                    asyncio.open_unix_connection(path, limit=self._config.read_buffer_limit),
                    timeout=self._config.connect_timeout,
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
        if not isinstance(msg, dict) or msg.get("id") is None:
            return  # notification (e.g. job.progress): unused, ignore
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
