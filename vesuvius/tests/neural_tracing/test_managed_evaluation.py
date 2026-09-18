"""No GPU probing/CUDA; verify UUID mapping and managed CLI routing."""
import json
import sys

import pytest

from vesuvius.neural_tracing.fiber_trace_3d import managed_evaluation as managed

INVENTORY = "0, GPU-aa\n1, GPU-bb\n2, GPU-cc\n"


def test_gpu_identity_mapping(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert managed.resolve_gpu_uuids("2,0", inventory_text=INVENTORY) == ["GPU-cc", "GPU-aa"]
    assert managed.resolve_gpu_uuids("1,0", inventory_text=INVENTORY, visibility="GPU-cc,GPU-aa") == ["GPU-aa", "GPU-cc"]
    for devices in ("0,0", "MIG-abc", "GPU-missing"):
        with pytest.raises(ValueError):
            managed.resolve_gpu_uuids(devices, inventory_text=INVENTORY)
    with pytest.raises(ValueError, match="hidden"):
        managed.resolve_gpu_uuids("GPU-cc", inventory_text=INVENTORY, visibility="0,1")


def test_launch_uses_local_ordinals_and_existing_wrapper(monkeypatch):
    from villa_gpu_manager import exclusive
    from villa_gpu_manager import bootstrap
    monkeypatch.setattr(bootstrap, "ensure_manager", lambda path: (path, {}))
    monkeypatch.setattr(managed, "resolve_gpu_uuids", lambda devices: ["GPU-cc", "GPU-aa"])
    recorded = {}
    def run(command, **kwargs):
        recorded.update(command=command, **kwargs)
        return 7
    monkeypatch.setattr(exclusive, "run_exclusive", run)
    assert managed.launch_managed([sys.executable, "entry.py"], devices="2,0",
                                  settings={"socket": "/tmp/fake", "priority": 101}) == 7
    assert recorded["command"][-3:] == ["--unmanaged", "--devices", "0,1"]
    assert recorded["gpu_uuids"] == ["GPU-cc", "GPU-aa"]
    assert recorded["priority"] == 101
    assert str(managed.Path(managed.__file__).resolve().parents[3]) in recorded["environment"]["PYTHONPATH"]


def test_external_gpu_processes_block_benchmark(monkeypatch):
    monkeypatch.setattr(managed.subprocess, "check_output", lambda *a, **k: "GPU-aa, 123\n")
    with pytest.raises(RuntimeError, match="external GPU processes"):
        managed.assert_gpus_idle(["GPU-aa"])
    managed.assert_gpus_idle(["GPU-bb"])
    monkeypatch.setattr(managed.subprocess, "check_output", lambda *a, **k: "")
    managed.assert_gpus_idle(["GPU-aa"])


def test_live_run_uses_demand_scoped_manager_settings_without_mutation():
    from vesuvius.neural_tracing.fiber_trace_3d import live_evaluate
    config = {"gpu_manager": {"enabled": True, "priority": 50,
                              "vram_bytes_per_gpu": 4 << 30}}
    settings = live_evaluate.demand_gpu_manager_settings(
        config, "0,1", manager_socket="custom.sock", manager_priority=123)
    assert settings == {"enabled": True, "priority": 123,
                        "vram_bytes_per_gpu": 4 << 30,
                        "devices": "0,1", "socket": "custom.sock"}
    assert config["gpu_manager"] == {"enabled": True, "priority": 50,
                                      "vram_bytes_per_gpu": 4 << 30}
    assert live_evaluate.demand_gpu_manager_settings(
        config, "0,1", unmanaged=True) is None
    assert live_evaluate.demand_gpu_manager_settings({}, "0,1") is None
    assert live_evaluate.runtime_inference_settings(
        {"tile_size": 64, "demand_gpu_manager": {"stale": True}}, None
    ) == {"tile_size": 64}
    assert live_evaluate.runtime_inference_settings(
        {"tile_size": 64, "demand_gpu_manager": {"stale": True}}, settings
    ) == {"tile_size": 64, "demand_gpu_manager": settings}
    with pytest.raises(ValueError, match="vram_bytes_per_gpu"):
        live_evaluate.demand_gpu_manager_settings(
            {"gpu_manager": {"enabled": True}}, "0")


def test_live_precision_identity_resolution_does_not_touch_cuda(monkeypatch, tmp_path):
    import torch
    from vesuvius.neural_tracing.fiber_trace_3d.infer import _resolve_inference_precision
    checkpoint = tmp_path / "snapshot.pt"
    torch.save({"config": {"training": {"mixed_precision": "bf16"}}}, checkpoint)
    monkeypatch.setattr(torch.cuda, "device",
                        lambda *a, **k: pytest.fail("CUDA touched before demand grant"))
    assert _resolve_inference_precision(
        "auto", checkpoint=checkpoint, devices=(torch.device("cuda:0"),),
        validate_devices=False) == ("bf16", "checkpoint")


def test_visualization_resume_selection_round_robins_missing_cases(tmp_path):
    from vesuvius.neural_tracing.fiber_trace_3d.live_evaluate import resume_visualization_selection

    cases = [
        (0, "a", [[0, 0, 0], [1, 0, 0]]),
        (1, "a", [[1, 0, 0], [2, 0, 0]]),
        (2, "b", [[0, 1, 0], [1, 1, 0]]),
        (3, "a", [[2, 0, 0], [3, 0, 0]]),
        (4, "b", [[1, 1, 0], [2, 1, 0]]),
    ]
    for index, source, points in cases:
        case = tmp_path / "cases" / str(index)
        case.mkdir(parents=True)
        (case / "trace.json").write_text(json.dumps({
            "provenance": {"case_index": index, "source_id": source},
            "reference_points_base_xyz": points,
        }))
    rendered = tmp_path / "cases/0/rendered"
    rendered.mkdir()
    (rendered / "render.json").write_text(json.dumps({"format": "vc_fiber_inspection", "pages": 1}))
    (rendered / "strips_0.jpg").write_bytes(b"complete")

    assert [case for case, _ in resume_visualization_selection(tmp_path, ["a", "b"], 3)] == [1, 2, 3]
    assert [case for case, _ in resume_visualization_selection(tmp_path, ["a", "b"], 10)] == [1, 2, 3, 4]
