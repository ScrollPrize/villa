"""GPU pause/resume server and client via Unix domain socket.

Allows other applications to temporarily claim the GPU while a training
run is in progress.  The training loop calls ``server.check(offload, reload)``
once per batch; if a ``pause`` command has been received the check blocks
until ``resume`` is sent.

Stack-based ownership
---------------------
Multiple processes can chain: A (training) holds GPU → B (inference)
pauses A → C pauses B → C finishes, B resumes → B finishes, A resumes.

Each GPU-using process runs its own ``GpuPauseServer`` on a PID-based
socket (``/tmp/gpu_pause.<pid>.sock``).  A stack file
(``~/.gpu_owner_stack``) tracks the LIFO order.  ``gpu_pause_context``
reads the stack to find who to pause, starts its own server (so it can
be paused by a third process), and on exit resumes the previous owner.

Backward compatibility: if the stack is empty, callers fall back to the
legacy socket ``~/.gpu_pause.sock`` so an old training process (that
doesn't register on the stack) is still discoverable.

Protocol
--------
Socket protocol is unchanged — text-based, newline-delimited:

1. Server sends version banner: ``gpu_pause v1\\n``
2. Client sends one command: ``pause\\n``, ``resume\\n``, or ``status\\n``
3. Server replies with one line and closes the connection.

CLI
---
::

    python lasagna/gpu_pause.py pause
    python lasagna/gpu_pause.py resume
    python lasagna/gpu_pause.py status
"""
from __future__ import annotations

import atexit
import fcntl
import os
import socket
import sys
import threading
from pathlib import Path

PROTOCOL_VERSION = "gpu_pause v1"
LEGACY_SOCKET_PATH = "~/.gpu_pause.sock"
STACK_FILE = Path("~/.gpu_owner_stack").expanduser()
SOCKET_DIR = "/tmp"

_socket_counter = 0
_socket_counter_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Stack file: tracks GPU ownership as a LIFO stack of (pid, socket_path)
# ---------------------------------------------------------------------------


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _stack_lock():
    """Open and flock the stack file. Returns the file object (caller closes)."""
    STACK_FILE.parent.mkdir(parents=True, exist_ok=True)
    f = open(STACK_FILE, "a+")
    fcntl.flock(f, fcntl.LOCK_EX)
    f.seek(0)
    return f


def _stack_read(f) -> list[tuple[int, str]]:
    """Parse stack file lines into [(pid, socket_path), ...], oldest first."""
    entries = []
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            pid_s, sock = line.split(":", 1)
            entries.append((int(pid_s), sock))
        except (ValueError, IndexError):
            continue
    return entries


def _stack_write(f, entries: list[tuple[int, str]]):
    f.seek(0)
    f.truncate()
    for pid, sock in entries:
        f.write(f"{pid}:{sock}\n")
    f.flush()


def _stack_push(pid: int, socket_path: str):
    f = _stack_lock()
    try:
        entries = _stack_read(f)
        # Remove any stale entry for this PID
        entries = [(p, s) for p, s in entries if p != pid]
        entries.append((pid, socket_path))
        _stack_write(f, entries)
    finally:
        fcntl.flock(f, fcntl.LOCK_UN)
        f.close()


def _stack_pop(pid: int):
    try:
        f = _stack_lock()
    except Exception:
        return
    try:
        entries = _stack_read(f)
        entries = [(p, s) for p, s in entries if p != pid]
        _stack_write(f, entries)
    finally:
        fcntl.flock(f, fcntl.LOCK_UN)
        f.close()


def _stack_top() -> tuple[int, str] | None:
    """Return (pid, socket_path) of the current GPU owner, or None.

    Prunes dead PIDs from the top of the stack.
    """
    try:
        f = _stack_lock()
    except Exception:
        return None
    try:
        entries = _stack_read(f)
        changed = False
        while entries:
            pid, sock = entries[-1]
            if _pid_alive(pid):
                break
            entries.pop()
            changed = True
        if changed:
            _stack_write(f, entries)
        return entries[-1] if entries else None
    finally:
        fcntl.flock(f, fcntl.LOCK_UN)
        f.close()


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


class GpuPauseServer:
    """Listener that accepts pause/resume commands over a Unix socket.

    Starts a daemon thread on construction.  The training loop calls
    :meth:`check` once per batch — it returns immediately unless a
    ``pause`` command is pending.

    Registers itself on the GPU owner stack so other processes can
    discover and pause it.
    """

    def __init__(self, socket_path: str | None = None, register: bool = True):
        global _socket_counter
        pid = os.getpid()
        if socket_path is None:
            with _socket_counter_lock:
                seq = _socket_counter
                _socket_counter += 1
            suffix = f".{seq}" if seq > 0 else ""
            socket_path = f"{SOCKET_DIR}/gpu_pause.{pid}{suffix}.sock"
        self._path = Path(socket_path).expanduser()
        self._pid = pid
        self._paused = False
        self._register = register

        self._pending_pause_conn: socket.socket | None = None
        self._pending_resume_conn: socket.socket | None = None
        self._lock = threading.Lock()
        self._resume_event = threading.Event()

        # Remove stale socket.
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.unlink(missing_ok=True)

        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server.bind(str(self._path))
        self._server.listen(2)
        self._server.settimeout(1.0)

        self._closed = False
        self._thread = threading.Thread(target=self._listen, daemon=True)
        self._thread.start()

        if register:
            _stack_push(pid, str(self._path))
            atexit.register(_stack_pop, pid)

    def _listen(self):
        while not self._closed:
            try:
                conn, _ = self._server.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                self._handle(conn)
            except Exception:
                try:
                    conn.close()
                except Exception:
                    pass

    def _handle(self, conn: socket.socket):
        conn.settimeout(10.0)
        conn.sendall(f"{PROTOCOL_VERSION}\n".encode())
        data = b""
        while b"\n" not in data and len(data) < 256:
            chunk = conn.recv(256)
            if not chunk:
                conn.close()
                return
            data += chunk
        cmd = data.decode().strip().lower()

        if cmd == "status":
            status = "paused" if self._paused else "running"
            conn.sendall(f"{status}\n".encode())
            conn.close()
        elif cmd == "pause":
            if self._paused:
                conn.sendall(b"already_paused\n")
                conn.close()
                return
            with self._lock:
                self._pending_pause_conn = conn
        elif cmd == "resume":
            if not self._paused:
                conn.sendall(b"not_paused\n")
                conn.close()
                return
            with self._lock:
                self._pending_resume_conn = conn
            self._resume_event.set()
        else:
            conn.sendall(b"unknown_command\n")
            conn.close()

    def check(self, offload_fn, reload_fn):
        """Call once per batch.  Non-blocking unless pause is pending."""
        with self._lock:
            conn = self._pending_pause_conn
            if conn is None:
                return
            self._pending_pause_conn = None

        # Pause requested — offload GPU state.
        print("[gpu_pause] pausing — offloading GPU...", flush=True)
        offload_fn()
        self._paused = True
        print("[gpu_pause] paused — GPU free", flush=True)
        try:
            conn.sendall(b"ok\n")
            conn.close()
        except Exception:
            pass

        # Block until resume, with watchdog for dead pausers.
        self._resume_event.clear()
        while not self._resume_event.wait(timeout=5.0):
            # Check if the process that paused us is still alive.
            top = _stack_top()
            if top is None or top[0] == self._pid:
                # Pauser died — we're back on top. Auto-resume.
                print("[gpu_pause] pauser died, auto-resuming", flush=True)
                break

        # Resume — reload GPU state.
        print("[gpu_pause] resuming — reloading GPU...", flush=True)
        reload_fn()
        self._paused = False
        print("[gpu_pause] running", flush=True)

        with self._lock:
            conn = self._pending_resume_conn
            self._pending_resume_conn = None
        if conn is not None:
            try:
                conn.sendall(b"ok\n")
                conn.close()
            except Exception:
                pass

    def close(self):
        self._closed = True
        try:
            self._server.close()
        except Exception:
            pass
        self._path.unlink(missing_ok=True)
        if self._register:
            _stack_pop(self._pid)
        self._thread.join(timeout=3.0)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


def gpu_pause_client(
    command: str,
    socket_path: str = LEGACY_SOCKET_PATH,
    timeout: float = 300.0,
) -> str:
    """Send a command to a running GpuPauseServer and return the response."""
    path = Path(socket_path).expanduser()
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    sock.connect(str(path))

    # Read version banner.
    banner = b""
    while b"\n" not in banner:
        banner += sock.recv(256)
    version = banner.decode().strip()
    if version != PROTOCOL_VERSION:
        sock.close()
        raise RuntimeError(
            f"Protocol mismatch: server={version!r}, "
            f"expected={PROTOCOL_VERSION!r}"
        )

    # Send command, read reply.
    sock.sendall(f"{command}\n".encode())
    reply = b""
    while b"\n" not in reply:
        chunk = sock.recv(256)
        if not chunk:
            break
        reply += chunk
    sock.close()
    return reply.decode().strip()


def _find_owner_socket() -> str | None:
    """Find the socket of the current GPU owner.

    Checks the stack file first, falls back to legacy socket.
    """
    top = _stack_top()
    if top is not None:
        return top[1]
    # Legacy fallback: old training process without stack registration.
    legacy = Path(LEGACY_SOCKET_PATH).expanduser()
    if legacy.exists():
        return str(legacy)
    return None


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


class gpu_pause_context:
    """Context manager that claims the GPU, pausing the current owner.

    Starts its own ``GpuPauseServer`` so this process can in turn be
    paused by a third process.  On exit, resumes the previous owner.

    If ``offload_fn`` / ``reload_fn`` are provided, this process is
    pausable — call :meth:`check` periodically in your work loop.

    Usage::

        with gpu_pause_context() as ctx:
            model = load_model()
            for chunk in chunks:
                ctx.check()  # allow newer process to pause us
                process(chunk)
    """

    def __init__(
        self,
        offload_fn=None,
        reload_fn=None,
        timeout: float = 300.0,
    ):
        self._offload_fn = offload_fn or (lambda: None)
        self._reload_fn = reload_fn or (lambda: None)
        self._timeout = timeout
        self._did_pause = False
        self._prev_socket: str | None = None
        self._server: GpuPauseServer | None = None

    def __enter__(self):
        # Find and pause the current GPU owner.
        owner_sock = _find_owner_socket()
        if owner_sock is not None:
            try:
                result = gpu_pause_client("pause", owner_sock, self._timeout)
                if result == "ok":
                    self._did_pause = True
                    self._prev_socket = owner_sock
                    print("[gpu_pause] paused previous owner", flush=True)
                elif result == "already_paused":
                    print("[gpu_pause] owner already paused", flush=True)
                else:
                    print(f"[gpu_pause] pause returned: {result}", flush=True)
            except (FileNotFoundError, ConnectionRefusedError):
                pass  # owner died or socket stale
            except Exception as e:
                print(f"[gpu_pause] pause failed: {e}", flush=True)

        # Start own server so we can be paused by a future process.
        self._server = GpuPauseServer(register=True)
        return self

    def check(self):
        """Call periodically to allow a newer process to pause us."""
        if self._server is not None:
            self._server.check(self._offload_fn, self._reload_fn)

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Stop own server (pops from stack).
        if self._server is not None:
            self._server.close()
            self._server = None

        # Resume previous owner.
        if self._did_pause and self._prev_socket:
            try:
                result = gpu_pause_client(
                    "resume", self._prev_socket, self._timeout,
                )
                print(f"[gpu_pause] resumed previous owner ({result})",
                      flush=True)
            except Exception as e:
                print(f"[gpu_pause] resume failed: {e}", flush=True)
        return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("pause", "resume", "status"):
        print(f"Usage: {sys.argv[0]} pause|resume|status", file=sys.stderr)
        sys.exit(1)
    cmd = sys.argv[1]

    sock = _find_owner_socket()
    if sock is None:
        print("No GPU owner found (stack empty, no legacy socket).",
              file=sys.stderr)
        sys.exit(1)

    try:
        result = gpu_pause_client(cmd, sock)
    except FileNotFoundError:
        print("No training process listening (socket not found).",
              file=sys.stderr)
        sys.exit(1)
    except ConnectionRefusedError:
        print("Connection refused (stale socket?).", file=sys.stderr)
        sys.exit(1)
    print(result)
