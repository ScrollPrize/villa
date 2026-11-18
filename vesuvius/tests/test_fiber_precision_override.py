import pytest
import torch
from vesuvius.neural_tracing.fiber_trace_3d.inference_adapter import FiberTrace3DPredictAdapter
from vesuvius.neural_tracing.fiber_trace_3d.infer import _resolve_inference_precision


@pytest.mark.parametrize("precision,expected", [("fp32", "off"), ("bf16", "bf16"), ("fp16", "fp16")])
def test_explicit_precision_survives_checkpoint_metadata(tmp_path, precision, expected):
    checkpoint = tmp_path / "model.pt"
    torch.save({"config": {"training": {"mixed_precision": "bf16", "learning_rate": 0.01}}}, checkpoint)
    mode, source = _resolve_inference_precision(precision, checkpoint=checkpoint,
        devices=(torch.device("cpu"),), validate_devices=False)
    assert source == "cli"
    adapter = FiberTrace3DPredictAdapter({"training": {"mixed_precision": mode}}, checkpoint=checkpoint)
    assert adapter.config["training"]["mixed_precision"] == expected
    assert adapter.config["training"]["learning_rate"] == 0.01


def test_checkpoint_precision_used_when_runtime_does_not_supply_one(tmp_path):
    checkpoint = tmp_path / "model.pt"
    torch.save({"config": {"training": {"mixed_precision": "bf16"}}}, checkpoint)
    adapter = FiberTrace3DPredictAdapter({}, checkpoint=checkpoint)
    assert adapter.config["training"]["mixed_precision"] == "bf16"


def test_cpu_auto_fp32_reaches_adapter_without_amx(tmp_path, monkeypatch):
    from vesuvius.utils.models import helpers

    monkeypatch.setattr(helpers, "_cpu_has_amx_bf16", lambda: False)
    checkpoint = tmp_path / "model.pt"
    torch.save({"config": {"training": {"mixed_precision": "bf16"}}}, checkpoint)
    mode, source = _resolve_inference_precision("auto", checkpoint=checkpoint,
        devices=(torch.device("cpu"),), validate_devices=False)
    assert (mode, source) == ("off", "checkpoint-cpu-fp32")
    adapter = FiberTrace3DPredictAdapter({"training": {"mixed_precision": mode}}, checkpoint=checkpoint)
    assert adapter.config["training"]["mixed_precision"] == "off"
