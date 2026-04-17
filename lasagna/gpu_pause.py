"""GPU pause/resume server and client via Unix domain socket.

Allows other applications to temporarily claim the GPU while a training
run is in progress.  The training loop calls ``server.check(offload, reload)``
once per batch; if a ``pause`` command has been received the check blocks
until ``resume`` is sent.

Protocol
--------
Socket path: ``~/.gpu_pause.sock``

Text-based, newline-delimited.  Each client connection:

1. Server sends version banner: ``gpu_pause v1\\n``
2. Client sends one command: ``pause\\n``, ``resume\\n``, or ``status\\n``
3. Server replies with one line and closes the connection.

Commands:

- ``pause``  — training finishes the current batch, calls *offload_fn*
  (move model/optimizer to CPU, ``torch.cuda.empty_cache()``), then
  replies ``ok\\n`` and idles.
- ``resume`` — training calls *reload_fn* (move back to GPU), replies
  ``ok\\n``, and continues.
- ``status`` — replies ``running\\n`` or ``paused\\n``.

Integration (few lines)
-----------------------
::

    from gpu_pause import GpuPauseServer

    pause_server = GpuPauseServer()

    def offload():
        model.cpu()
        for s in optimizer.state.values():
            for k, v in s.items():
                if isinstance(v, torch.Tensor):
                    s[k] = v.cpu()
        torch.cuda.empty_cache()

    def reload():
        model.to(device)
        for s in optimizer.state.values():
            for k, v in s.items():
                if isinstance(v, torch.Tensor):
                    s[k] = v.to(device)

    # In the batch loop, after optimizer.step:
    pause_server.check(offload, reload)

    # At exit:
    pause_server.close()

CLI
---
::

    python lasagna/gpu_pause.py pause
    python lasagna/gpu_pause.py resume
    python lasagna/gpu_pause.py status
"""
from __future__ import annotations

import os
import socket
import sys
import threading
from pathlib import Path

PROTOCOL_VERSION = "gpu_pause v1"
DEFAULT_SOCKET_PATH = "~/.gpu_pause.sock"


class GpuPauseServer:
    """Listener that accepts pause/resume commands over a Unix socket.

    Starts a daemon thread on construction.  The training loop calls
    :meth:`check` once per batch — it returns immediately unless a
    ``pause`` command is pending.
    """

    def __init__(self, socket_path: str = DEFAULT_SOCKET_PATH):
        self._path = Path(socket_path).expanduser()
        self._paused = False

        # Pending client connection waiting for a reply (set by the
        # listener thread, consumed by check()).
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
        self._server.settimeout(1.0)  # so the thread can check _closed

        self._closed = False
        self._thread = threading.Thread(target=self._listen, daemon=True)
        self._thread.start()

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
            # Store the connection — check() will reply and close it
            # after offloading.
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

        # Block until resume.
        self._resume_event.clear()
        self._resume_event.wait()

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
        self._thread.join(timeout=3.0)


def gpu_pause_client(
    command: str,
    socket_path: str = DEFAULT_SOCKET_PATH,
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


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("pause", "resume", "status"):
        print(f"Usage: {sys.argv[0]} pause|resume|status", file=sys.stderr)
        sys.exit(1)
    cmd = sys.argv[1]
    try:
        result = gpu_pause_client(cmd)
    except FileNotFoundError:
        print("No training process listening (socket not found).", file=sys.stderr)
        sys.exit(1)
    except ConnectionRefusedError:
        print("Connection refused (stale socket?).", file=sys.stderr)
        sys.exit(1)
    print(result)
