"""GPU_PAUSE_DISABLE=1 lets a second predict3d run without pausing the first."""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gpu_pause  # noqa: E402


@pytest.fixture
def owner(monkeypatch, tmp_path):
    monkeypatch.setattr(gpu_pause, "STACK_FILE", tmp_path / "stack")
    monkeypatch.setattr(gpu_pause, "LEGACY_SOCKET_PATH", str(tmp_path / "legacy.sock"))
    monkeypatch.setattr(gpu_pause, "SOCKET_DIR", str(tmp_path))
    server = gpu_pause.GpuPauseServer(register=True)
    yield server
    server.close()


def test_disable_does_not_look_for_an_owner(monkeypatch):
    monkeypatch.setenv("GPU_PAUSE_DISABLE", "1")

    def fail():
        raise AssertionError("looked up the GPU owner")

    monkeypatch.setattr(gpu_pause, "_find_owner_socket", fail)
    with gpu_pause.gpu_pause_context(timeout=5):
        pass


@pytest.mark.parametrize("value", ["1", "true", "yes"])
def test_disable_leaves_the_owner_running(owner, monkeypatch, value):
    monkeypatch.setenv("GPU_PAUSE_DISABLE", value)
    with gpu_pause.gpu_pause_context(timeout=5) as ctx:
        ctx.check()  # a no-op, so callers need no changes
        assert gpu_pause.gpu_pause_client("status", owner._path.as_posix(), 5) == "running"
    # it never registered, so the original owner is still on top of the stack
    assert gpu_pause._stack_top()[0] == os.getpid()
