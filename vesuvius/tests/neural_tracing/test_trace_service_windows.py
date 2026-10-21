"""trace_service without the CUDA extras and on every platform: it imports without cupy/cuCIM (only heatmap blob
detection needs them, and says so), serves one JSON line per connection over AF_UNIX (Windows: Winsock through ws2_32),
and its --parent_pid watch follows the process VC3D passed. No checkpoint, no GPU."""
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

import pytest
import torch


def _import_service(monkeypatch):
    for name in ("cupy", "cucim", "cucim.skimage", "cucim.skimage.measure"):
        monkeypatch.setitem(sys.modules, name, None)  # not installed, as on Windows
    for name in ("vesuvius.neural_tracing.trace_service", "vesuvius.neural_tracing.infer"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    import vesuvius.neural_tracing.trace_service as service

    return service


def _connect(path):
    if os.name != "nt":
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.connect(str(path))
        return client
    import ctypes

    from vesuvius.neural_tracing.trace_service import _sockaddr_un

    client = socket.socket(1, socket.SOCK_STREAM)
    address = _sockaddr_un(str(path))
    assert ctypes.windll.ws2_32.connect(ctypes.c_size_t(client.fileno()), ctypes.byref(address),
                                        ctypes.sizeof(address)) == 0
    return client


def test_service_imports_without_cupy(monkeypatch):
    service = _import_service(monkeypatch)
    assert service.COPY_REQUEST_TYPE == "displacement_copy_grow"


def test_heatmap_blob_detection_names_cupy_when_missing(monkeypatch):
    _import_service(monkeypatch)
    from vesuvius.neural_tracing.infer import Inference

    with pytest.raises(ImportError, match="cupy"):
        Inference.get_blob_coordinates(None, torch.zeros(4, 4, 4), torch.zeros(3))


def test_one_json_line_per_connection(monkeypatch):
    service = _import_service(monkeypatch)
    path = Path(tempfile.gettempdir()) / f"neural_tracer_test_{uuid.uuid4().hex[:8]}.sock"  # short: AF_UNIX limit
    sock = service._unix_listen(str(path))

    def serve():
        for _ in range(2):
            conn = service._unix_accept(sock)
            service.handle_connection(conn, lambda request: service.process_request(request, {}))

    server = threading.Thread(target=serve, daemon=True)
    server.start()
    try:
        for _ in range(2):
            with _connect(path) as client:
                client.settimeout(30)
                client.sendall(b'{"request_type": "no_such_request"}\n')
                client.shutdown(socket.SHUT_WR)
                reply = json.loads(client.makefile("rb").readline())
            assert "Unknown request_type 'no_such_request'" in reply["error"]
        server.join(10)
        assert not server.is_alive()
    finally:
        sock.close()
        path.unlink(missing_ok=True)


def test_socket_path_too_long_is_a_clear_error(monkeypatch):
    service = _import_service(monkeypatch)
    assert service._sockaddr_un("x" * 107).sun_path == b"x" * 107
    with pytest.raises(OSError, match="too long"):
        service._sockaddr_un("x" * 108)  # would fill sun_path with no terminating NUL


def test_parent_watch_follows_the_given_pid(monkeypatch):
    service = _import_service(monkeypatch)
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        if os.name == "nt":
            alive = service._parent_alive(child.pid)
            assert alive() and alive()
            assert child.poll() is None  # checking does not signal or end the process
        else:
            assert service._parent_alive(os.getppid())()
            assert not service._parent_alive(child.pid)()  # POSIX keeps the "still our parent" rule
    finally:
        child.kill()
        child.wait()
    if os.name == "nt":
        assert not alive()


_STAND_IN = """
import os, subprocess, sys
service = subprocess.Popen(sys.argv[1:] + ["--parent_pid", str(os.getpid())],
                           stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
print(service.pid, flush=True)
print(service.stdout.readline().decode().strip(), flush=True)  # "neural-tracing service listening on ..."
"""


def _running(service, pid):
    if os.name == "nt":
        return service._parent_alive(pid)()
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:  # a zombie only waits for its new parent to reap it
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return True


def test_service_exits_when_the_parent_that_reads_its_output_is_gone(monkeypatch, tmp_path):
    """VC3D reads the service's stdout and stderr through pipes, so once VC3D is gone the watch's last print fails
    (Windows: EINVAL, POSIX: EPIPE). The service must still exit instead of running on with the model in memory."""
    service = _import_service(monkeypatch)
    script = Path(service.__file__)
    path = Path(tempfile.gettempdir()) / f"neural_tracer_test_{uuid.uuid4().hex[:8]}.sock"
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [str(script.parents[2]), os.environ.get("PYTHONPATH")])))
    stand_in = subprocess.run([sys.executable, "-c", _STAND_IN, sys.executable, str(script), "--volume_zarr",
                               str(tmp_path), "--volume_scale", "0", "--socket_path", str(path)],
                              capture_output=True, text=True, env=env, timeout=300)
    pid = int(stand_in.stdout.split()[0])
    try:
        assert "listening" in stand_in.stdout, stand_in.stdout + stand_in.stderr
        deadline = time.monotonic() + 60
        while _running(service, pid) and time.monotonic() < deadline:
            time.sleep(0.5)
        assert not _running(service, pid), "the service outlived its parent"
    finally:
        if _running(service, pid):
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True) if os.name == "nt"                 else os.kill(pid, 9)
        path.unlink(missing_ok=True)
