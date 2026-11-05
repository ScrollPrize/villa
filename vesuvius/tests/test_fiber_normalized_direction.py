import math
import pytest
import torch
from vesuvius.neural_tracing.fiber_trace_3d.fiber_affine_geometry import direction_cone


def energy(p,family=0,sign=1.,active=None,groups=1,minimum=.1):
    return direction_cone(p,torch.tensor([0,0]),torch.tensor([0.,100.]),
        torch.tensor([family]),torch.tensor([sign]),
        dict(geometry_direction_loss='normalized_vector',
             geometry_minimum_forward_slope=minimum),active,groups=groups)[0]


@pytest.mark.parametrize('family',[0,1])
@pytest.mark.parametrize('sign',[-1.,1.])
@pytest.mark.parametrize('angle',[0,5,30,60])
def test_scale_invariance_and_no_radial_angular_gradient(family,sign,angle):
    for length in (30.,100.,400.):
        p=torch.zeros(2,2,dtype=torch.float64)
        p[1,family]=sign*length*math.cos(math.radians(angle))
        p[1,1-family]=length*math.sin(math.radians(angle));p.requires_grad_()
        loss=energy(p,family,sign)
        assert loss.item()==pytest.approx(2*(1-math.cos(math.radians(angle))),abs=1e-12)
        g=torch.autograd.grad(loss,p)[0]
        assert (g[1]*p[1]).sum().item()==pytest.approx(0,abs=1e-12)


@pytest.mark.parametrize('family',[0,1])
@pytest.mark.parametrize('sign',[-1.,1.])
@pytest.mark.parametrize('slope',[-1.,0.,1e-10,.05])
def test_safeguard_restores_reverse_collapsed_and_short_segments(family,sign,slope):
    p=torch.zeros(2,2,dtype=torch.float64)
    p[1,family]=sign*100*slope;p.requires_grad_()
    loss=energy(p,family,sign);g=torch.autograd.grad(loss,p)[0]
    assert torch.isfinite(loss) and torch.isfinite(g).all()
    assert g[1,family]*sign<0
    if slope==0:assert loss.item()==pytest.approx(2.01)


def test_batched_active_masks_match_independent_values_and_gradients():
    p=torch.tensor([[[0.,0.],[0.,0.]],[[90.,25.],[-50.,10.]]],dtype=torch.float64,requires_grad=True)
    active=torch.tensor([[True,False]])
    a=energy(p,active=active);b=energy(p*2,active=active)
    packed=energy(torch.cat((p,p*2),1),active=active,groups=2)
    torch.testing.assert_close(packed,torch.stack((a,b)))
    g=torch.autograd.grad(packed.sum(),p,retain_graph=True)[0]
    torch.testing.assert_close(g,torch.autograd.grad(a+b,p)[0])
    assert torch.count_nonzero(g[:,1])==0


def test_reject_double_progress_penalty():
    with pytest.raises(ValueError,match='own progress hinge'):
        direction_cone(torch.zeros(2,2),torch.tensor([0,0]),torch.tensor([0.,100.]),
            torch.tensor([0]),torch.tensor([1.]),dict(geometry_direction_loss='normalized_vector',
            geometry_forward_progress_weight=1))
