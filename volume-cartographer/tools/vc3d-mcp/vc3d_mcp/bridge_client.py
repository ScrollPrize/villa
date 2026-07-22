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
3. If VC3D was spawned by us (the auto-launch path in ``server.py``),
   ``BridgeClient.socket_path_from_handshake``
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
from dataclasses import dataclass
from typing import Any

from .jobs import JobTracker

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
        self._connect_lock = asyncio.Lock()
        self.jobs = JobTracker()
        self._closed = False
        # Bumped on every successful connect(); passed into _read_loop so a
        # stale reader only tears down state it still owns (prevents a dead
        # reader from clearing a newer connection).
        self._generation = 0

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
        self.jobs.reset_error()
        self._generation += 1
        self._read_task = asyncio.create_task(
            self._read_loop(self._generation), name="vc3d-bridge-reader"
        )

    def _is_live(self) -> bool:
        return (
            self._writer is not None
            and not self._writer.is_closing()
            and self._read_task is not None
            and not self._read_task.done()
        )

    async def ensure_connected(self) -> None:
        if self._closed:
            raise BridgeConnectionError("bridge client has been closed")
        if self._is_live():
            return
        async with self._connect_lock:
            # Double-check under the lock: a racing caller may have connected.
            if self._closed:
                raise BridgeConnectionError("bridge client has been closed")
            if self._is_live():
                return
            await self.connect()

    async def close(self) -> None:
        # Idempotent: safe to call multiple times / after a drop.
        self._closed = True
        task, self._read_task = self._read_task, None
        if task is not None:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        writer, self._writer = self._writer, None
        self._reader = None
        if writer is not None and not writer.is_closing():
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:  # noqa: BLE001 - closing a dead writer can raise
                pass
        self._on_disconnect(BridgeConnectionError("connection closed"))

    def _on_disconnect(self, exc: Exception) -> None:
        """Fail all pending RPC futures and wake/fail all job waiters."""
        self._fail_all_pending(exc)
        self.jobs.fail_all(exc)

    def _reset_transport(self) -> None:
        """Drop the dead reader/writer so the next call reconnects."""
        self._read_task = None
        self._reader = None
        writer = self._writer
        self._writer = None
        if writer is not None and not writer.is_closing():
            writer.close()

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
                # Write failed: the connection is dead. Invalidate it so the
                # next call reconnects, and wake anyone else waiting on it.
                err = BridgeConnectionError(f"write failed: {exc}")
                self._pending.pop(req_id, None)
                self._reset_transport()
                self._on_disconnect(err)
                raise err from exc

        effective_timeout = timeout if timeout is not None else self._config.request_timeout
        try:
            return await asyncio.wait_for(fut, timeout=effective_timeout)
        except asyncio.TimeoutError as exc:
            raise BridgeConnectionError(
                f"timed out waiting for response to {method!r} (id={req_id})"
            ) from exc
        except asyncio.CancelledError:
            # Preserve caller cancellation; the finally below cleans up _pending.
            raise
        finally:
            # Timeout OR caller cancellation: the reader never popped us.
            # Idempotent on success (dispatch already removed the entry).
            self._pending.pop(req_id, None)

    # -- background reader -----------------------------------------------------

    async def _read_loop(self, generation: int) -> None:
        # Read from our own captured reader so a stale loop keeps draining its
        # own (dead) socket rather than a newer connection's.
        reader = self._reader
        assert reader is not None
        try:
            while True:
                line = await reader.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                self._dispatch_line(line)
        except asyncio.CancelledError:
            return  # close() cancelled us; it owns teardown + pending.
        except Exception as exc:  # noqa: BLE001 - surface to all waiters below
            if generation == self._generation:
                self._reset_transport()
                self._on_disconnect(BridgeConnectionError(f"reader loop crashed: {exc}"))
            return
        # EOF: drop the now-dead socket so the next call reconnects -- but only
        # if we still own the active connection (a stale reader must not clear
        # a newer one).
        if generation == self._generation:
            self._reset_transport()
            self._on_disconnect(BridgeConnectionError("connection closed by peer"))

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
