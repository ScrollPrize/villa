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


def test_live_run_routes_before_checkpoint_cuda_or_native(monkeypatch, tmp_path):
    from vesuvius.neural_tracing.fiber_trace_3d import live_evaluate
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"gpu_manager": {"enabled": True}}))
    seen = []
    monkeypatch.setattr(managed, "launch_managed", lambda command, **kw: seen.append((command, kw)) or 0)
    monkeypatch.setattr(live_evaluate, "freeze_checkpoint", lambda *a: pytest.fail("checkpoint touched before grant"))
    assert live_evaluate.run(path, "missing.pt", "0,1", "missing-binary", render=True) == 0
    assert "--render" in seen[0][0] and seen[0][1]["devices"] == "0,1"
