"""The follow service's socket server and parent watch on every platform (Windows: Winsock AF_UNIX through ws2_32
and a process handle), the CT reader fallback when VC3D ships no Python bindings and the cache folder it uses, the
eager gather without Triton, and the once-only chunk-cache metadata; no checkpoint, network or GPU."""
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

import numpy as np
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


def test_missing_vc3d_bindings_fall_back_to_the_remote_reader(monkeypatch, capsys):
    from vesuvius.neural_tracing.tracers.shared.crop_sampling import volume
    monkeypatch.setitem(sys.modules, 'vc', None)
    monkeypatch.delitem(sys.modules, 'vc.volume', raising=False)
    monkeypatch.setenv('VESUVIUS_CT_READER', 'vc3d')
    made = []
    monkeypatch.setattr(volume, 'RemoteChunkedArray', lambda *args, **kwargs: made.append(args) or 'remote')
    spec = volume.CTVolumeSpec('s3://vesuvius-challenge-open-data/test.zarr', ct_level=3)
    assert volume.open_ct_array(spec, 1 << 20) == 'remote'
    assert made and made[0][:2] == ('s3://vesuvius-challenge-open-data/test.zarr', 3)
    assert 'VC3D bindings not found' in capsys.readouterr().out


def _follow_cache_dir(monkeypatch, tmp_path, recorded, given=None):
    """The CT cache folder the follow service hands the remote reader under VC3D's environment without `vc`: the
    service's own option handling, then the real configure_volume and open_ct_array on a volume spec that records
    ``recorded`` (as a checkpoint does), with the reader stubbed (no checkpoint, no network)."""
    from vesuvius.neural_tracing.tracers.fibers.tracing import runtime
    from vesuvius.neural_tracing.tracers.shared.crop_sampling import volume
    monkeypatch.setitem(sys.modules, 'vc', None)
    monkeypatch.delitem(sys.modules, 'vc.volume', raising=False)
    monkeypatch.setenv('VESUVIUS_CT_READER', 'vc3d')
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path/'xdg'))
    used = []

    class Opened(Exception):
        pass

    def remote(url, level, cache_dir, *args, **kwargs):
        used.append(cache_dir)
        raise Opened

    def build_tracer(options):
        spec = volume.CTVolumeSpec('s3://vesuvius-challenge-open-data/test.zarr', ct_level=2, cache_dir=str(recorded))
        runtime.configure_volume(options, spec, SimpleNamespace(), 'checkpoint.pt')
        volume.open_ct_array(spec, 1 << 20)

    monkeypatch.setattr(volume, 'RemoteChunkedArray', remote)
    monkeypatch.setattr(service, 'build_tracer', build_tracer)
    args = SimpleNamespace(checkpoint='checkpoint.pt', device='cpu', precision='fp32', confidence=.5,
                           state_dir=str(tmp_path/'state'), gpu_chunks=8, base_um=None, ct=None, ct_level=None,
                           ct_grid_scale=None, grid_scale=None, cache_dir=given, frame_checkpoint=None)
    with pytest.raises(Opened):
        service.FiberFollow(args)
    return used


@pytest.mark.parametrize('present', [False, True], ids=['recorded_folder_absent', 'recorded_folder_present'])
def test_vc3d_fallback_uses_the_default_cache_not_the_recorded_folder(monkeypatch, tmp_path, capsys, present):
    recorded = tmp_path/'recorded_volume_cache'  # stands in for the training machine's folder a checkpoint records
    if present:
        recorded.mkdir()
    default = str(tmp_path/'xdg'/'vesuvius'/'volume_cache')
    assert _follow_cache_dir(monkeypatch, tmp_path, recorded) == [default]
    assert f'CT chunk cache: {default}' in capsys.readouterr().out
    assert recorded.exists() == present  # never created


def test_vc3d_fallback_keeps_a_given_cache_dir(monkeypatch, tmp_path):
    given = str(tmp_path/'mine')
    assert _follow_cache_dir(monkeypatch, tmp_path, tmp_path/'recorded_volume_cache', given) == [given]


def test_cuda_gather_is_not_compiled_without_triton(monkeypatch):
    import torch
    import torch.utils._triton as triton_helpers
    from vesuvius.neural_tracing.tracers.shared.crop_sampling import gpu
    cuda = torch.device('cuda')  # a device object only: nothing is allocated
    monkeypatch.setattr(triton_helpers, 'has_triton', lambda: False)
    assert not gpu._compiles_on(cuda)
    assert gpu._compiles_on(torch.device('cpu'))
    monkeypatch.setattr(gpu, '_GATHERS', {})
    monkeypatch.setattr(gpu, '_gather_kernel', lambda *shifts: lambda *tensors: 'eager')
    monkeypatch.setattr(torch, 'compile', lambda *args, **kwargs: pytest.fail('compiled without Triton'))
    assert gpu._gather_trilinear(SimpleNamespace(device=cuda), *[None]*6, (5, 5, 5)) == 'eager'
    monkeypatch.setattr(triton_helpers, 'has_triton', lambda: True)
    assert gpu._compiles_on(cuda)


def test_a_missing_triton_helper_is_no_crash(monkeypatch):
    import importlib.util
    import torch
    from vesuvius.neural_tracing.tracers.shared.crop_sampling import gpu
    monkeypatch.setitem(sys.modules, 'torch.utils._triton', None)
    assert gpu._compiles_on(torch.device('cuda')) == (importlib.util.find_spec('triton') is not None)


def test_chunk_cache_metadata_is_published_once(tmp_path):
    from vesuvius.neural_tracing.tracers.shared.crop_sampling.volume import RemoteChunkedArray
    array = SimpleNamespace(shape=(4, 4, 4), chunks=(2, 2, 2), dtype=np.uint8, fill_value=0)
    local = tmp_path/'s3'/'bucket'/'test.zarr'/'2'
    RemoteChunkedArray.publish_metadata(local, array, 's3://bucket/test.zarr', 2)
    assert json.loads((local/'.zarray').read_text())['shape'] == [4, 4, 4]
    (local/'.zarray').write_text('{"marker": 1}')  # on Windows another reader may hold this file open
    RemoteChunkedArray.publish_metadata(local, array, 's3://bucket/test.zarr', 2)
    assert (local/'.zarray').read_text() == '{"marker": 1}'
