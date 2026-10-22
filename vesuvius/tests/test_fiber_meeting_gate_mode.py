import pytest
import torch
from vesuvius.neural_tracing.fiber_trace_3d.fiber_geometry_confidence import meeting_confidence_gate


def test_modes_and_historical_defaults():
    cfg=dict(geometry_confidence_gated=True,geometry_confidence_low=.5,geometry_confidence_high=.6)
    p=torch.tensor(.7)
    assert meeting_confidence_gate(p,p,dict(cfg,meeting_gate_mode='endpoints'))==1
    assert meeting_confidence_gate(p,p,dict(cfg,meeting_gate_mode='joint'))==0
    assert meeting_confidence_gate(p,p,cfg)==1
    assert meeting_confidence_gate(p,p,dict(cfg,bucket_uv_dynamic_same_sheet=True))==0
    with pytest.raises(ValueError):meeting_confidence_gate(p,p,dict(cfg,meeting_gate_mode='bad'))


def test_gate_is_detached_and_not_double_applied():
    p=torch.tensor(.9,requires_grad=True);q=torch.tensor(.8,requires_grad=True)
    cfg=dict(geometry_confidence_gated=True,geometry_confidence_low=.5,geometry_confidence_high=.6)
    for mode in ('joint','endpoints'):
        gate=meeting_confidence_gate(p,q,dict(cfg,meeting_gate_mode=mode))
        assert gate==1 and not gate.requires_grad
