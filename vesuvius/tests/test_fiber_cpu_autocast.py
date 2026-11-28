import pytest
import torch
from vesuvius.neural_tracing.fiber_trace_3d import infer
from vesuvius.utils.models import helpers


@pytest.mark.parametrize("validate", [False, True])
def test_cpu_auto_fp32_but_explicit_bf16_retained(monkeypatch, validate):
    monkeypatch.setattr(helpers, "_cpu_has_amx_bf16", lambda: False)
    monkeypatch.setattr(infer, "_checkpoint_mixed_precision_policy", lambda checkpoint: "bf16")
    devices = (torch.device("cpu"),)
    assert infer._resolve_inference_precision("auto", checkpoint=None, devices=devices, validate_devices=validate) == ("off", "checkpoint-cpu-fp32")
    for requested, expected in [("fp32", "off"), ("bf16", "bf16")]:
        assert infer._resolve_inference_precision(requested, checkpoint=None, devices=devices, validate_devices=validate) == (expected, "cli")


def test_accelerator_auto_policy_retained(monkeypatch):
    monkeypatch.setattr(infer, "_checkpoint_mixed_precision_policy", lambda checkpoint: "bf16")
    assert infer._resolve_inference_precision("auto", checkpoint=None, devices=(torch.device("cuda"),), validate_devices=False) == ("bf16", "checkpoint")


def test_amx_cpu_auto_keeps_checkpoint_bf16(monkeypatch):
    monkeypatch.setattr(helpers, "_cpu_has_amx_bf16", lambda: True)
    monkeypatch.setattr(infer, "_checkpoint_mixed_precision_policy", lambda checkpoint: "bf16")
    assert infer._resolve_inference_precision("auto", checkpoint=None, devices=(torch.device("cpu"),), validate_devices=False) == ("bf16", "checkpoint")
