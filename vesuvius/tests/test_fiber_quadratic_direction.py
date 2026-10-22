import math
import pytest
import torch
from vesuvius.neural_tracing.fiber_trace_3d.fiber_affine_geometry import direction_cone


@pytest.mark.parametrize('family',[0,1])
@pytest.mark.parametrize('sign',[-1.,1.])
@pytest.mark.parametrize('angle',[0,2,5,30,90,180])
def test_vector_direction_formula(family,sign,angle):
    p=torch.zeros(2,2,dtype=torch.float64)
    p[1,family]=sign*100*math.cos(math.radians(angle))
    p[1,1-family]=100*math.sin(math.radians(angle));p.requires_grad_()
    loss,_=direction_cone(p,torch.tensor([0,0]),torch.tensor([0.,100.]),torch.tensor([family]),torch.tensor([sign]),dict(geometry_direction_loss='quadratic_vector'))
    c=math.cos(math.radians(angle))
    assert loss.item()==pytest.approx(2*(1-c)+max(0,.1-c)**2,abs=1e-12)
    g=torch.autograd.grad(loss,p)[0]
    if angle==180:assert g[1,family]*sign<0


def test_vector_direction_batched_and_collapsed():
    p=torch.zeros(2,2,2,requires_grad=True)
    args=(torch.tensor([0,0]),torch.tensor([0.,100.]),torch.tensor([0]),torch.tensor([1.]))
    cfg=dict(geometry_direction_loss='quadratic_vector')
    a,_=direction_cone(p,*args,cfg,torch.tensor([[True,False]]))
    g=torch.autograd.grad(a,p)[0]
    assert g[1,0,0]<0 and torch.count_nonzero(g[:,1])==0
    b,_=direction_cone(torch.cat((p,p),1),*args,cfg,torch.tensor([[True,False]]),groups=2)
    torch.testing.assert_close(b,torch.stack((a,a)))


def test_do_not_silently_mix_old_progress_weight():
    with pytest.raises(ValueError,match='own progress hinge'):
        direction_cone(torch.zeros(2,2),torch.tensor([0,0]),torch.tensor([0.,100.]),
                       torch.tensor([0]),torch.tensor([1.]),
                       dict(geometry_direction_loss='quadratic_vector',geometry_forward_progress_weight=10000))
