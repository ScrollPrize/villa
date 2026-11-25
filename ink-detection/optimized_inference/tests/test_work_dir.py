"""Downloaded layers and model weights go under WORK_DIR, which defaults to /workspace.

Outside the container /workspace usually cannot be created (macOS has a read-only root), so a
local run must be able to point the work directory somewhere else."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import entrypoint  # noqa: E402


@pytest.fixture
def inference_env(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL", "some/model")
    monkeypatch.setenv("START_LAYER", "0")
    monkeypatch.setenv("END_LAYER", "4")
    monkeypatch.setenv("STEP", "inference")
    monkeypatch.setenv("SURFACE_VOLUME_ZARR", str(tmp_path))
    monkeypatch.delenv("WORK_DIR", raising=False)


def test_work_dir_defaults_to_workspace(inference_env):
    assert entrypoint.parse_env().work_dir == "/workspace"


def test_work_dir_comes_from_the_environment(inference_env, monkeypatch, tmp_path):
    monkeypatch.setenv("WORK_DIR", str(tmp_path / "work"))
    assert entrypoint.parse_env().work_dir == str(tmp_path / "work")


def test_inference_step_downloads_weights_under_work_dir(inference_env, monkeypatch, tmp_path):
    class StopAfterDownload(Exception):
        pass

    seen = {}

    def fake_download(model_name, dest_dir, profiler=None):
        seen["dest_dir"] = dest_dir
        raise StopAfterDownload

    monkeypatch.setenv("WORK_DIR", str(tmp_path / "work"))
    monkeypatch.setattr(entrypoint, "download_model_weights", fake_download)
    with pytest.raises(StopAfterDownload):
        entrypoint.run_inference_step(entrypoint.parse_env())
    assert seen["dest_dir"] == str(tmp_path / "work" / "models")
    assert (tmp_path / "work" / "models").is_dir()
