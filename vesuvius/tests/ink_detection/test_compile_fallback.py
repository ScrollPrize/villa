"""Compilation failures must fall back even after a successful batch."""
import pytest
import torch
from torch import nn
from torch._dynamo.exc import BackendCompilerFailed
from torch._inductor.exc import TritonMissing

from vesuvius.ink_detection.inference.inference_runtime import maybe_compile_model


@pytest.mark.parametrize("failure_on", [1, 2])
def test_real_backend_failure_falls_back_and_stays_eager(monkeypatch, caplog, failure_on):
    torch._dynamo.reset()
    calls = []

    def backend(graph, inputs, **kwargs):
        calls.append(1)
        if len(calls) == failure_on:
            raise RuntimeError("deliberate backend compilation failure")
        return graph.forward

    real_compile = torch.compile
    monkeypatch.setattr(torch, "compile", lambda model, **kw: real_compile(model, backend=backend, **kw))
    eager = nn.Linear(3, 2).eval()
    model, enabled = maybe_compile_model(eager, enabled=True, mode="default")
    try:
        with torch.no_grad(), caplog.at_level("WARNING"):
            for batch_size in (4, 2, 1, 4):
                batch = torch.arange(batch_size * 3, dtype=torch.float32).reshape(batch_size, 3)
                torch.testing.assert_close(model(batch), eager(batch), rtol=0, atol=0)
        assert enabled
        assert len(calls) == failure_on
        assert model._compiled_model is None
        assert sum("continuing eagerly" in r.message for r in caplog.records) == 1
    finally:
        torch._dynamo.reset()


@pytest.mark.parametrize("after_success", [False, True])
@pytest.mark.parametrize("error_type", [RuntimeError, ValueError, TypeError, torch.OutOfMemoryError])
def test_model_errors_propagate_without_eager_retry(monkeypatch, caplog, after_success, error_type):
    eager_calls = []
    error = error_type("ordinary model failure")

    class Eager(nn.Module):
        def forward(self, batch):
            eager_calls.append(1)
            return batch + 1

    class Compiled(nn.Module):
        def forward(self, batch):
            if batch.shape[0] == 1:
                raise error
            return batch + 1

    compiled = Compiled()
    monkeypatch.setattr(torch, "compile", lambda model, **kwargs: compiled)
    model, _ = maybe_compile_model(Eager(), enabled=True, mode="default")
    if after_success:
        model(torch.zeros(2, 3))
    with pytest.raises(error_type) as caught:
        model(torch.zeros(1, 3))
    assert caught.value is error
    assert eager_calls == []
    assert model._compiled_model is compiled
    assert "continuing eagerly" not in caplog.text


def test_direct_triton_missing_falls_back(monkeypatch, caplog):
    class Broken(nn.Module):
        def forward(self, batch):
            raise TritonMissing(first_useful_frame=None)

    eager = nn.Linear(3, 2).eval()
    monkeypatch.setattr(torch, "compile", lambda model, **kwargs: Broken())
    model, _ = maybe_compile_model(eager, enabled=True, mode="default")
    batch = torch.ones(2, 3)
    with caplog.at_level("WARNING"):
        torch.testing.assert_close(model(batch), eager(batch), rtol=0, atol=0)
    assert model._compiled_model is None
    assert "continuing eagerly" in caplog.text


def test_eager_error_after_compiler_failure_is_not_hidden(monkeypatch):
    error = ValueError("eager model also fails")

    class Eager(nn.Module):
        def forward(self, batch):
            raise error

    class Broken(nn.Module):
        def forward(self, batch):
            raise BackendCompilerFailed(
                lambda graph, inputs: None, RuntimeError("backend failed"), first_useful_frame=None
            )

    monkeypatch.setattr(torch, "compile", lambda model, **kwargs: Broken())
    model, _ = maybe_compile_model(Eager(), enabled=True, mode="default")
    with pytest.raises(ValueError) as caught:
        model(torch.ones(2, 3))
    assert caught.value is error
    assert model._compiled_model is None


def test_real_compiled_model_error_is_not_retried(monkeypatch, caplog):
    torch._dynamo.reset()
    calls = []

    @torch.compiler.disable
    def fail():
        calls.append(1)
        raise ValueError("model-owned runtime failure")

    class Eager(nn.Module):
        def forward(self, batch):
            result = batch + 1
            if batch.shape[0] == 1:
                fail()
            return result

    real_compile = torch.compile
    monkeypatch.setattr(torch, "compile", lambda model, **kw: real_compile(model, backend="eager", **kw))
    model, _ = maybe_compile_model(Eager(), enabled=True, mode="default")
    try:
        model(torch.zeros(2, 3))
        with pytest.raises(ValueError, match="model-owned runtime failure"):
            model(torch.zeros(1, 3))
        assert len(calls) == 1
        assert model._compiled_model is not None
        assert "continuing eagerly" not in caplog.text
    finally:
        torch._dynamo.reset()
