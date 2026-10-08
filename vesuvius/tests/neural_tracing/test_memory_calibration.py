from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from vesuvius.neural_tracing.fiber_trace_3d.memory_calibration import calibration_from_observations
from vesuvius.neural_tracing.fiber_trace_3d.cooperative_training import launch_settings
from vesuvius.neural_tracing.fiber_trace_3d.training_state import config_fingerprint


def configuration():
    raw = {"training": {"device": "cuda", "optimizer_global_batch_size": 128,
        "max_steps": 100, "sample_pipeline_workers": 32, "sample_pipeline_slots": 256,
        "sync_batchnorm": False, "ddp_broadcast_buffers": False}}
    fingerprint = config_fingerprint(raw)
    rows = [{"microbatch_size": size, "initialization_reserved_bytes": 20,
             "peak_reserved_bytes": 20 + 10 * size, "peak_allocated_bytes": 10 + 10 * size,
             "driver_peak_bytes": 30 + 10 * size, "gpu_name": "fake",
             "torch_version": "test", "cuda_version": "test", "world_size": 8,
             "config_fingerprint": fingerprint} for size in (1, 4)]
    raw["training"]["cooperative"] = {
        "socket": "/tmp/example.sock",
        "request": {"max_cpu_cores": 32, "max_gpus": 8, "min_vram_bytes_per_gpu": 50,
                    "max_vram_bytes_per_gpu": 200, "allowed_gpu_uuids": [], "priority": 0, "exclusive": False},
        "memory_calibration": calibration_from_observations(fingerprint, rows, reserve_bytes=10),
    }
    return raw


@pytest.mark.parametrize("override,expected", [(None, 107), ("2", 2)])
def test_probe_is_gated_by_exclusive_lease(monkeypatch, tmp_path, override, expected):
    from villa_gpu_manager import bootstrap, cohort, exclusive, processes
    from vesuvius.neural_tracing.fiber_trace_3d import memory_calibration as module, train
    raw = configuration()
    raw["training"]["cooperative"]["priority"] = 7
    monkeypatch.setattr(train, "_load_raw_config", lambda _: raw)
    grant = {"cpu_cores": 2, "gpus": {"GPU-a": 100}}
    monkeypatch.setattr(bootstrap, "ensure_manager", lambda _: ("fake", grant))
    events = []
    class Lease:
        def __init__(self, path, allocation, **kwargs):
            assert kwargs["priority"] == expected
            assert allocation == grant
        def __enter__(self):
            events.append("grant")
            return self
        def authorize(self):
            events.append("authorize")
        def check(self):
            events.append("check")
        def __exit__(self, *args):
            assert self.cohort.closed
            events.append("release")
    class FakeCohort:
        closed = False
        ready = True
        idle = True
        done = True
        processes = [{"pid": 123}]
        def __init__(self, factory, *args, **kwargs):
            assert events == ["grant"]
            assert factory.rendezvous.startswith("file://")
            events.append("spawn")
            self.state = SimpleNamespace(load=lambda: [{"gpu_uuid": "GPU-a"}], close=lambda: None)
        def poll(self):
            assert "authorize" in events
        def drain(self):
            self.closed = True
        def abort(self):
            self.closed = True
    class Monitor:
        def __init__(self, *args, **kwargs):
            self.last_snapshot = {"GPU-a": {"processes": {}, "total": 100, "used": 0}}
            self.last_snapshot_at = module.time.monotonic()
        def poll(self, pids):
            self.last_snapshot["GPU-a"]["processes"] = {p: 50 for p in pids}
        def close(self):
            pass
    monkeypatch.setattr(exclusive, "ExclusiveTask", Lease)
    monkeypatch.setattr(cohort, "ProcessCohort", FakeCohort)
    monkeypatch.setattr(processes, "NvidiaMonitor", Monitor)
    monkeypatch.setattr(module, "calibration_from_observations", lambda _, rows, **kw: rows)
    output = tmp_path / "report.json"
    args = [str(tmp_path / "config.json"), "--gpu-uuids", "GPU-a", "--microbatch-sizes", "1",
            "--cpu-workers", "2", "--vram-limit-bytes", "100", "--reserve-bytes", "10",
            "--output", str(output)]
    if override is not None:
        args += ["--priority", override]
    assert module.main(args) == 0
    assert events == ["grant", "spawn", "authorize", "check", "release"]
    assert json.loads(output.read_text())[0]["driver_peak_bytes"] == 50


def test_calibration_bounds_observations_and_does_not_extrapolate():
    settings = launch_settings(configuration())
    for size, peak in settings.calibration.measured_peaks:
        assert settings.calibration.estimated_bytes(size) >= peak + 10
    assert settings.calibration.microbatch_for(10**9) == 4
    assert settings.calibration.microbatch_for(49) == 0
    assert settings.batch_size == 128


def test_socket_default_and_override_precedence(monkeypatch, tmp_path):
    from villa_gpu_manager import paths
    from villa_gpu_manager.paths import default_socket_path
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(paths, "_listening", lambda _: False)
    monkeypatch.setattr(paths, "_writable_location", lambda _: False)
    raw = configuration()
    assert launch_settings(raw, socket="/tmp/override.sock").socket == "/tmp/override.sock"
    assert launch_settings(raw).socket == "/tmp/example.sock"
    del raw["training"]["cooperative"]["socket"]
    assert launch_settings(raw).socket == default_socket_path()
    assert default_socket_path() == str(tmp_path / ".local/run/villa-gpu-manager/manager.sock")
    with pytest.raises(ValueError, match="socket path"):
        launch_settings(raw, socket="")


def test_operational_controls_do_not_change_calibration_identity():
    raw = configuration()
    original = config_fingerprint(raw)
    raw["training"]["cooperative"]["request"]["priority"] = 10
    assert config_fingerprint(raw) == original
    assert launch_settings(raw).request["priority"] == 10
    raw["training"]["optimizer_global_batch_size"] = 64
    with pytest.raises(ValueError, match="different training config"):
        launch_settings(raw)


def test_cannot_supply_unrelated_peak_records():
    raw = configuration()
    raw["training"]["cooperative"]["memory_calibration"]["observations"][0]["driver_peak_bytes"] += 1
    with pytest.raises(ValueError, match="recorded observations"):
        launch_settings(raw)


def test_queue_slots_must_hold_current_effective_batch():
    raw = configuration()
    raw["training"]["sample_pipeline_slots"] = 64
    # Re-identify synthetic observations to test the slot constraint itself.
    identity = config_fingerprint(raw)
    calibration = raw["training"]["cooperative"]["memory_calibration"]
    calibration["config_fingerprint"] = identity
    for row in calibration["observations"]:
        row["config_fingerprint"] = identity
    with pytest.raises(ValueError, match="one effective batch"):
        launch_settings(raw)


def test_every_usable_microbatch_needs_distributed_calibration():
    raw = configuration()
    raw["training"]["cooperative"]["memory_calibration"]["observations"][1]["world_size"] = 1
    with pytest.raises(ValueError, match="every usable microbatch"):
        launch_settings(raw)


def test_production_configuration_rejects_cpu():
    raw = configuration()
    raw["training"]["device"] = "cpu"
    identity = config_fingerprint(raw)
    calibration = raw["training"]["cooperative"]["memory_calibration"]
    calibration["config_fingerprint"] = identity
    for row in calibration["observations"]:
        row["config_fingerprint"] = identity
    with pytest.raises(ValueError, match="requires CUDA"):
        launch_settings(raw)
