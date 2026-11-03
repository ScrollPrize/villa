from concurrent.futures import ThreadPoolExecutor
import errno
from pathlib import Path
import subprocess
import sys

import pytest

from villa_gpu_manager import bootstrap
from villa_gpu_manager.protocol import ProtocolError


def test_auto_inventory_uses_logical_cpus(monkeypatch):
    import villa_gpu_manager.cli as cli
    monkeypatch.setattr(sys, "argv", ["bootstrap", "--serve", "/tmp/example.sock"])
    monkeypatch.setattr(bootstrap, "visible_gpu_uuids", lambda: ["GPU-fake"])
    monkeypatch.setattr(bootstrap, "_shared_uids", lambda p: None)
    def count(*, logical):
        assert logical is True
        return 224
    monkeypatch.setattr(bootstrap.psutil, "cpu_count", count)
    calls = []
    monkeypatch.setattr(cli, "main", lambda args: calls.append(args))
    bootstrap.main()
    assert calls[0][calls[0].index("--cpu-cores") + 1] == "224"


def test_inventory_validator_accepts_logical_worker_capacity(monkeypatch):
    from villa_gpu_manager.processes import nvidia_inventory
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: "GPU-fake, 81920\n")
    monkeypatch.setattr(bootstrap.psutil, "cpu_count", lambda logical=True: 224 if logical else 112)
    for count in (128, 224):
        assert nvidia_inventory(count, ["GPU-fake"], 1024**3)["cpu_cores"] == count
    with pytest.raises(ProtocolError, match="logical CPU inventory"):
        nvidia_inventory(225, ["GPU-fake"], 1024**3)


def test_visibility_resolves_indices_and_uuid_prefix(monkeypatch):
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: "0, GPU-aaa\n1, GPU-bbb\n")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert bootstrap.visible_gpu_uuids() == ["GPU-aaa", "GPU-bbb"]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,GPU-a")
    assert bootstrap.visible_gpu_uuids() == ["GPU-bbb", "GPU-aaa"]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert bootstrap.visible_gpu_uuids() == []
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "9")
    with pytest.raises(ValueError):
        bootstrap.visible_gpu_uuids()
    for mask in ("0,-1,1", ",-1,0", "0,", "0,0"):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
        with pytest.raises(ValueError):
            bootstrap.visible_gpu_uuids()


@pytest.mark.parametrize("error", [PermissionError(errno.EACCES, "denied"),
                                   ProtocolError("version", "incompatible")])
def test_bootstrap_never_replaces_unauthorized_or_incompatible_service(monkeypatch, error):
    def fail(_):
        raise error
    monkeypatch.setattr(bootstrap, "_inventory", fail)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("must not start"))
    with pytest.raises(type(error)):
        bootstrap.ensure_manager("/irrelevant.sock")


def test_concurrent_starters_reuse_one_real_service(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    path = str(tmp_path / "service" / "manager.sock")
    children = []
    popen = subprocess.Popen
    def launch(command, **kwargs):
        child = popen([sys.executable, str(Path(__file__).with_name("bootstrap_server.py")), command[-1]], **kwargs)
        children.append(child)
        return child
    monkeypatch.setattr(subprocess, "Popen", launch)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: bootstrap.ensure_manager(path, timeout=5), range(2)))
        assert results[0] == results[1] == (path, {"cpu_cores": 4, "gpus": {"GPU-fake": 100}})
        count = len(children)
        assert bootstrap.ensure_manager(path) == results[0]
        assert len(children) == count
        with bootstrap.Client(path) as client:
            assert client.status()["tasks"] == []
        assert Path(path + ".json").is_file()
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            child.wait(timeout=5)
