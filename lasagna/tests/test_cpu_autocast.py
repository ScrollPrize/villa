import ast
from pathlib import Path

import pytest
import torch

from vesuvius.utils.models import helpers
from vesuvius.utils.models.helpers import inference_autocast_enabled


@pytest.fixture
def no_amx(monkeypatch):
    monkeypatch.setattr(helpers, "_cpu_has_amx_bf16", lambda: False)


@pytest.mark.parametrize("kind,expected", [("cpu", False), ("cuda", True), ("mps", True), ("xpu", True)])
def test_policy(no_amx, kind, expected):
    assert inference_autocast_enabled(torch.device(kind)) is expected


def test_policy_keeps_cpu_bf16_on_amx(monkeypatch):
    monkeypatch.setattr(helpers, "_cpu_has_amx_bf16", lambda: True)
    assert inference_autocast_enabled(torch.device("cpu")) is True


@pytest.mark.parametrize("caps,expected", [
    ({"avx512_bf16": True, "amx_bf16": False}, False),
    ({"avx512_bf16": True, "amx_bf16": True}, True),
    ({}, False),
])
def test_amx_probe_reads_capabilities(monkeypatch, caps, expected):
    monkeypatch.setattr(torch.cpu, "get_capabilities", lambda: caps, raising=False)
    helpers._cpu_has_amx_bf16.cache_clear()
    try:
        assert helpers._cpu_has_amx_bf16() is expected
    finally:
        helpers._cpu_has_amx_bf16.cache_clear()


def test_amx_probe_failure_means_fp32(monkeypatch):
    def broken():
        raise RuntimeError("probe failed")
    monkeypatch.setattr(torch.cpu, "get_capabilities", broken, raising=False)
    helpers._cpu_has_amx_bf16.cache_clear()
    try:
        assert helpers._cpu_has_amx_bf16() is False
    finally:
        helpers._cpu_has_amx_bf16.cache_clear()


def test_predict3d_adapter_uses_fp32_on_cpu(no_amx):
    from lasagna.preprocess_cos_omezarr import LasagnaCosPredict3DAdapter

    adapter = LasagnaCosPredict3DAdapter.__new__(LasagnaCosPredict3DAdapter)
    model = torch.nn.Conv3d(1, 8, 1)
    with torch.inference_mode():
        y = adapter.run_tile_inference(model, torch.ones(1, 1, 2, 2, 2), device=torch.device("cpu"))
    assert y.dtype == torch.float32
    assert y.isfinite().all()


def test_predict3d_autocast_uses_shared_policy():
    source = Path(__file__).resolve().parents[1] / "preprocess_cos_omezarr.py"
    tree = ast.parse(source.read_text())
    adapter = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "LasagnaCosPredict3DAdapter"
    )
    method = next(
        node for node in adapter.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_tile_inference"
    )
    contexts = [
        node for node in ast.walk(method)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "autocast"
    ]
    assert len(contexts) == 1
    enabled = next(k.value for k in contexts[0].keywords if k.arg == "enabled")
    assert ast.unparse(enabled) == "inference_autocast_enabled(device)"
