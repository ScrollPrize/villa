import torch
from vesuvius.neural_tracing.fiber_trace_3d.fiber_bucket_uv import meeting_geometry_mass,meeting_cost
from vesuvius.neural_tracing.fiber_trace_3d.fiber_geometry_confidence import confidence_gate


def test_detach_preserves_force_but_removes_probability_escape_gradient():
    p=torch.tensor(.58,requires_grad=True)
    delta=torch.tensor([914.,197.],requires_grad=True)
    cfg=dict(geometry_confidence_low=.5,geometry_confidence_high=.6)
    gate=confidence_gate(p,cfg)
    old=meeting_geometry_mass(p,cfg)*gate*meeting_cost(delta)
    new=meeting_geometry_mass(p,dict(cfg,bucket_uv_meeting_detach_mass=True))*gate*meeting_cost(delta)
    torch.testing.assert_close(old,new)
    gp,gd=torch.autograd.grad(old,(p,delta),retain_graph=True)
    np,nd=torch.autograd.grad(new,(p,delta),allow_unused=True)
    assert gp>0 and np is None
    torch.testing.assert_close(gd,nd)
    # This is not a frozen initial weight: subsequent probability changes matter.
    assert meeting_geometry_mass(torch.tensor(.7),dict(bucket_uv_meeting_detach_mass=True))!=p.detach()
