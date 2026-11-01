import pytest
from vesuvius.neural_tracing.fiber_trace_3d import fiber_uv_layout as uv


def test_default_backend_is_shared_eager():
    assert uv.pair_evaluator({}) is uv.pair_objective
    with pytest.raises(ValueError,match='pair_backend'):uv.pair_evaluator({'pair_backend':'oops'})


def test_compilation_uses_shared_function_fullgraph(monkeypatch):
    import torch
    calls=[]
    def compile(fn,**kwargs):calls.append((fn,kwargs));return 'compiled'
    monkeypatch.setattr(torch,'compile',compile)
    assert uv.pair_evaluator({'pair_backend':'compiled'})=='compiled'
    assert calls==[(uv.pair_objective,{'fullgraph':True,'dynamic':False})]
