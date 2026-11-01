import torch
import pytest
from vesuvius.neural_tracing.fiber_trace_3d.fiber_affine_geometry import arc_length_energy


def test_shared_matches_existing_arc_weighted_loss():
    delta=torch.tensor([[3.,4.],[0.,12.]],dtype=torch.float64,requires_grad=True)
    gap=torch.tensor([10.,6.],dtype=torch.float64)
    old=((torch.linalg.vector_norm(delta/gap[:,None],dim=1)-1).square()*gap).sum()/gap.sum()
    new=arc_length_energy(delta[:,None],gap)
    torch.testing.assert_close(new,old)
    a=torch.autograd.grad(new,delta,retain_graph=True)[0]
    b=torch.autograd.grad(old,delta)[0]
    torch.testing.assert_close(a,b)


def test_local_batches_masks_and_restoring_gradients():
    d=torch.tensor([[[5.,0.],[100.,0.]],[[0.,20.],[0.,100.]]],requires_grad=True)
    gap=torch.tensor([10.,10.]);active=torch.tensor([[True,False],[True,False]])
    one=arc_length_energy(d,gap,active)
    both=arc_length_energy(torch.cat((d,d*2),1),gap,active,groups=2)
    torch.testing.assert_close(both,torch.stack((one,arc_length_energy(d*2,gap,active))))
    g=torch.autograd.grad(one,d)[0]
    assert g[0,0,0]<0 and g[1,0,1]>0
    assert torch.count_nonzero(g[:,1])==0


def test_empty_and_unit_length():
    empty=torch.empty(0,2,2,requires_grad=True)
    assert arc_length_energy(empty,torch.empty(0)).item()==0
    assert arc_length_energy(torch.tensor([[[3.,4.]]]),torch.tensor([5.])).item()==0


def test_uniform_relative_errors_and_gradients_do_not_depend_on_arc_length():
    gap=torch.tensor([1.,1000.],dtype=torch.float64)
    ratios=torch.tensor([.8,1.2],dtype=torch.float64,requires_grad=True)
    d=torch.stack((ratios*gap,torch.zeros_like(gap)),-1)[:,None]
    loss=arc_length_energy(d,gap,weighting='uniform')
    torch.testing.assert_close(loss,torch.tensor(.04,dtype=torch.float64))
    torch.testing.assert_close(torch.autograd.grad(loss,ratios)[0],torch.tensor([-.2,.2],dtype=torch.float64))


def test_uniform_batched_masks_and_different_errors():
    d=torch.tensor([[[.5,0.],[99.,0.]],[[1500.,0.],[99.,0.]]],requires_grad=True)
    gap=torch.tensor([1.,1000.]);mask=torch.tensor([[True,False],[True,False]])
    a=arc_length_energy(d,gap,mask,weighting='uniform')
    b=arc_length_energy(d*2,gap,mask,weighting='uniform')
    packed=arc_length_energy(torch.cat((d,d*2),1),gap,mask,groups=2,weighting='uniform')
    torch.testing.assert_close(packed,torch.stack((a,b)))
    assert a.item()==pytest.approx(.25)
    gp=torch.autograd.grad(packed.sum(),d,retain_graph=True)[0]
    torch.testing.assert_close(gp,torch.autograd.grad(a+b,d)[0])
    assert torch.count_nonzero(gp[:,1])==0


def test_invalid_segment_weighting_rejected():
    with pytest.raises(ValueError,match='weighting'):
        arc_length_energy(torch.empty(0,1,2),torch.empty(0),weighting='typo')
