"""The follow service's socket server and parent watch on every platform (Windows: Winsock AF_UNIX through ws2_32
and a process handle); no checkpoint, network or GPU. Written by slchicoineau-creator and Baz248 for #2163."""
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from vesuvius.neural_tracing.tracers.fibers.tracing import service


def connect(path):
    if os.name != 'nt':
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.connect(str(path))
        return client
    import ctypes
    client = socket.socket(1, socket.SOCK_STREAM)
    address = service._sockaddr_un(str(path))
    assert ctypes.windll.ws2_32.connect(ctypes.c_size_t(client.fileno()), ctypes.byref(address),
                                        ctypes.sizeof(address)) == 0
    return client


def test_one_json_line_per_connection(monkeypatch):
    monkeypatch.setattr(service, 'SERVICE', SimpleNamespace(handle=lambda request: dict(echo=request['op'])),
                        raising=False)
    path = Path(tempfile.gettempdir())/f'vc3d_follow_test_{uuid.uuid4().hex[:8]}.sock'  # short: AF_UNIX limit 108
    server = service.UnixServer(str(path), service.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        for op in ('hello', 'normals', 'extrapolate'):
            with connect(path) as client:
                client.sendall(json.dumps(dict(op=op)).encode()+b'\n')
                reply = json.loads(client.makefile('rb').readline())
            assert reply['ok'] and reply['echo'] == op
    finally:
        server.shutdown()
        server.server_close()
        path.unlink(missing_ok=True)


def test_socket_path_too_long_is_a_clear_error():
    assert service._sockaddr_un('x'*107).sun_path == b'x'*107
    with pytest.raises(OSError, match='too long'):
        service._sockaddr_un('x'*108)  # would fill sun_path with no terminating NUL


def test_parent_check_follows_a_live_then_ended_process():
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
    try:
        alive = service.parent_alive(child.pid)
        assert alive() and alive()
        assert child.poll() is None  # checking does not signal or end the parent
    finally:
        child.kill()
        child.wait()
    assert not alive()


def test_watch_parent_stops_the_server_when_the_parent_ends():
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(6)'])
    threading.Thread(target=child.wait, daemon=True).start()  # reap it, as VC3D's own parent would
    stopped = threading.Event()
    watcher = threading.Thread(target=service.watch_parent, args=(child.pid, SimpleNamespace(shutdown=stopped.set)),
                               daemon=True)
    watcher.start()
    assert not stopped.wait(3)  # the parent still runs: the service stays up
    assert stopped.wait(15)
