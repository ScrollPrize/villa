import json
from copy import deepcopy

import pytest
from vesuvius.neural_tracing.fiber_trace_3d.cooperative_setup import task_priority


@pytest.mark.parametrize("settings,override,expected", [
    ({}, None, 0), ({"priority": 7}, None, 7),
    ({"request": {"priority": 3}}, None, 3),
    ({"priority": 7, "request": {"priority": 3}}, None, 7),
    ({"priority": 7}, 0, 0),
])
def test_priority_precedence(settings, override, expected):
    assert task_priority(settings, override) == expected


@pytest.mark.parametrize("value", [True, 1.5, "7"])
def test_priority_rejects_non_integer(value):
    with pytest.raises(ValueError, match="integer"):
        task_priority({"priority": value})
import torch

from test_memory_calibration import configuration
from vesuvius.neural_tracing.fiber_trace_3d.cooperative_setup import (
    load_calibration, prepare_settings, derive_request,
)
from vesuvius.neural_tracing.fiber_trace_3d.memory_calibration import calibration_report_path
from vesuvius.neural_tracing.fiber_trace_3d.cooperative_training import launch_settings
from vesuvius.neural_tracing.fiber_trace_3d.training_state import config_fingerprint


def fixture_config(tmp_path):
    raw = configuration()
    raw["training"]["run_path"] = str(tmp_path / "runs")
    observation = raw["training"]["cooperative"]["memory_calibration"]
    identity = config_fingerprint(raw)
    observation["config_fingerprint"] = identity
    for row in observation["observations"]:
        row.update(config_fingerprint=identity, torch_version=str(torch.__version__), cuda_version=torch.version.cuda or "cpu-test")
    return raw


def test_zero_manager_config_loads_report_and_derives_request(monkeypatch, tmp_path):
    from villa_gpu_manager import bootstrap
    monkeypatch.setattr(torch.version, "cuda", "cpu-test")
    raw = fixture_config(tmp_path)
    observation = raw["training"].pop("cooperative")["memory_calibration"]
    config = tmp_path / "config.json"
    target = calibration_report_path(config, raw)
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps(observation))
    original = deepcopy(raw)
    inventory = {"cpu_cores": 16, "gpus": {"GPU-a": 200, "GPU-b": 200, "GPU-c": 200}}
    monkeypatch.setattr(bootstrap, "ensure_manager", lambda path: (path, inventory))
    monkeypatch.setattr(bootstrap, "compatible_gpu_uuids", lambda names: ["GPU-a", "GPU-b"])
    prepared = prepare_settings(config, raw, socket="/tmp/test.sock")
    settings = launch_settings(prepared)
    assert settings.request["max_gpus"] == 2
    assert settings.request["allowed_gpu_uuids"] == ["GPU-a", "GPU-b"]
    assert settings.workers == 16
    assert settings.batch_size == 128
    assert raw == original
    assert config_fingerprint(prepared) == config_fingerprint(raw)


def test_missing_calibration_requests_exclusive_real_startup(monkeypatch, tmp_path):
    from villa_gpu_manager import bootstrap
    inventory = {"cpu_cores": 224, "gpus": {"GPU-a": 80 * 1024**3, "GPU-b": 80 * 1024**3}}
    monkeypatch.setattr(bootstrap, "ensure_manager", lambda path: (path, inventory))
    monkeypatch.setattr(bootstrap, "visible_gpu_uuids", lambda: ["GPU-a", "GPU-b"])
    raw = fixture_config(tmp_path)
    raw["training"].pop("cooperative")
    raw["training"]["sample_pipeline_workers"] = 128
    raw["test_datasets"] = ["test-fixture"]
    raw["training"]["test_interval"] = 10
    prepared = prepare_settings(tmp_path / "config.json", raw)
    settings = launch_settings(prepared)
    assert settings.calibration is None
    assert settings.request["exclusive"]
    assert settings.request["priority"] == 100
    assert settings.request["max_gpus"] == 2
    assert settings.workers == 128
    assert "memory_calibration" not in prepared["training"]["cooperative"]


def test_missing_calibration_accepts_explicit_visible_managed_subset(monkeypatch, tmp_path):
    from villa_gpu_manager import bootstrap
    inventory = {
        "cpu_cores": 224,
        "gpus": {"GPU-a": 80 * 1024**3, "GPU-b": 80 * 1024**3},
    }
    monkeypatch.setattr(bootstrap, "ensure_manager", lambda path: (path, inventory))
    monkeypatch.setattr(bootstrap, "visible_gpu_uuids", lambda: ["GPU-a"])
    raw = fixture_config(tmp_path)
    raw["training"].pop("cooperative")
    raw["test_datasets"] = ["test-fixture"]
    raw["training"]["test_interval"] = 10
    prepared = prepare_settings(tmp_path / "config.json", raw)
    request = prepared["training"]["cooperative"]["request"]
    assert request["exclusive"]
    assert request["max_gpus"] == 1
    assert request["allowed_gpu_uuids"] == ["GPU-a"]


def test_request_honors_calibrated_rank_coverage(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.version, "cuda", "cpu-test")
    raw = fixture_config(tmp_path)
    for row in raw["training"]["cooperative"]["memory_calibration"]["observations"]:
        row["world_size"] = 2
    obs, cal = load_calibration(tmp_path / "config.json", raw)
    inventory = {"cpu_cores": 128, "gpus": {f"GPU-{i}": 200 for i in range(8)}}
    request = derive_request(raw["training"], inventory, list(inventory["gpus"]), obs, cal)
    assert request["max_gpus"] == 2
    with pytest.raises(ValueError, match="no visible"):
        derive_request(raw["training"], inventory, [], obs, cal)


def test_explicit_request_cannot_bypass_visibility(monkeypatch, tmp_path):
    from villa_gpu_manager import bootstrap
    monkeypatch.setattr(torch.version, "cuda", "cpu-test")
    raw = fixture_config(tmp_path)
    monkeypatch.setattr(bootstrap, "ensure_manager", lambda path: (path,
        {"cpu_cores": 128, "gpus": {"GPU-a": 200, "GPU-b": 200}}))
    monkeypatch.setattr(bootstrap, "compatible_gpu_uuids", lambda _: ["GPU-b"])
    prepared = prepare_settings(tmp_path / "config.json", raw)
    assert prepared["training"]["cooperative"]["request"]["allowed_gpu_uuids"] == ["GPU-b"]
    assert prepared["training"]["cooperative"]["request"]["max_gpus"] == 1
    monkeypatch.setattr(bootstrap, "compatible_gpu_uuids", lambda _: [])
    with pytest.raises(ValueError, match="no visible"):
        prepare_settings(tmp_path / "config.json", raw)
