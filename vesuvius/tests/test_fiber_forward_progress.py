import pytest
import torch
from vesuvius.neural_tracing.fiber_trace_3d.fiber_affine_geometry import direction_cone


@pytest.mark.parametrize('family', [0,1])
@pytest.mark.parametrize('sign', [-1.,1.])
@pytest.mark.parametrize('slope', [-1.,0.,.05,1.])
def test_progress_has_restoring_gradient_at_reversal_and_collapse(family,sign,slope):
    p=torch.zeros(2,2,dtype=torch.float64)
    p[1,family]=sign*slope*100
    p.requires_grad_()
    args=(torch.tensor([0,0]),torch.tensor([0.,100.]),torch.tensor([family]),torch.tensor([sign]))
    old,_=direction_cone(p,*args,{})
    new,_=direction_cone(p,*args,dict(geometry_forward_progress_weight=10,geometry_minimum_forward_slope=.1))
    extra=new-old
    assert extra.item()==pytest.approx(10*max(0,.1-slope)**2)
    grad=torch.autograd.grad(extra,p)[0]
    if slope<.1:assert grad[1,family]*sign<0
    else:assert torch.count_nonzero(grad)==0


def test_batched_groups_equal_separate_and_ignore_inactive_labels():
    p=torch.tensor([[[0.,0.],[0.,0.]], [[-100.,0.],[100.,0.]]],requires_grad=True)
    args=(torch.tensor([0,0]),torch.tensor([0.,100.]),torch.tensor([0]),torch.tensor([1.]))
    cfg=dict(geometry_forward_progress_weight=10)
    active=torch.tensor([[True,False]])
    a,_=direction_cone(p,*args,cfg,active)
    both,_=direction_cone(torch.cat((p,p*2),1),*args,cfg,active,groups=2)
    b,_=direction_cone(p*2,*args,cfg,active)
    torch.testing.assert_close(both,torch.stack((a,b)))
    grad=torch.autograd.grad(both.sum(),p)[0]
    assert torch.count_nonzero(grad[:,1])==0
