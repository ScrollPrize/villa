import torch
from torch import nn

from vesuvius.ink_detection.inference import inference_runtime



def test_skips_compile_for_cuda_model_without_triton(monkeypatch):
    model = nn.Linear(2, 2)
    calls = []
    monkeypatch.setattr(torch, "compile", lambda *a, **k: calls.append(1) or model)
    monkeypatch.setattr(inference_runtime, "_has_triton", lambda: False)
    monkeypatch.setattr(inference_runtime, "_uses_cuda", lambda m: True)
    out, compiled = inference_runtime.maybe_compile_model(model, enabled=True, mode="reduce-overhead")
    assert out is model and compiled is False and calls == []


def test_still_compiles_cpu_model_without_triton(monkeypatch):
    model = nn.Linear(2, 2)
    calls = []
    monkeypatch.setattr(torch, "compile", lambda m, **k: calls.append(1) or m)
    monkeypatch.setattr(inference_runtime, "_has_triton", lambda: False)
    out, compiled = inference_runtime.maybe_compile_model(model, enabled=True, mode="reduce-overhead")
    assert compiled is True and calls == [1]


def test_compiles_cuda_model_with_triton(monkeypatch):
    model = nn.Linear(2, 2)
    calls = []
    monkeypatch.setattr(torch, "compile", lambda m, **k: calls.append(1) or m)
    monkeypatch.setattr(inference_runtime, "_has_triton", lambda: True)
    monkeypatch.setattr(inference_runtime, "_uses_cuda", lambda m: True)
    out, compiled = inference_runtime.maybe_compile_model(model, enabled=True, mode="reduce-overhead")
    assert compiled is True and calls == [1]
