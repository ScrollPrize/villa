import pytest
import torch
from vesuvius.neural_tracing.fiber_trace_3d import fiber_uv_layout as uv


def loss(mode, position, retain=1., sign=1., sigma=.5):
    config=dict(layout_mode='shared_anchor',distance_objective=mode,
                sign_weight=0.,metric_weight=0.,distance_sigma=sigma)
    pa=torch.tensor([[1.]],dtype=torch.float64)*retain
    logits=torch.tensor([[0.,4.]],dtype=torch.float64,requires_grad=True)
    pb=logits.softmax(-1)*retain
    qa=torch.tensor([[10.]],dtype=torch.float64)
    qb=torch.tensor([[11.,position]],dtype=torch.float64,requires_grad=True)
    z=torch.zeros(1,dtype=torch.float64)
    table=torch.as_tensor(uv.make_distance_table(1.,.5),dtype=torch.float64)
    value=uv.pair_objective(pa,pb,qa,qb,qa*0,qb*0,z,z,z+sign,z+1,table,config)
    return value,logits,qb


@pytest.mark.parametrize('mode',['squared','huber','huber_likelihood'])
def test_farther_wrong_mass_costs_more_and_gradient_moves_it(mode):
    near,_,_=loss(mode,20.)
    far,logits,qb=loss(mode,30.)
    assert far>near+1
    far.backward()
    assert logits.grad[0,1]>0 and logits.grad[0,0]<0
    assert qb.grad[0,1]>0


@pytest.mark.parametrize('mode',['likelihood','squared','huber','huber_likelihood'])
def test_retention_mass_gates_without_renormalizing(mode):
    full,_,_=loss(mode,20.)
    partial,_,_=loss(mode,20.,retain=.4)
    zero,_,_=loss(mode,20.,retain=0.)
    torch.testing.assert_close(partial,full*.16)
    assert zero==0


@pytest.mark.parametrize('sign',[-1.,0.,1.])
@pytest.mark.parametrize('r',[-1.,1.])
def test_huber_boundary_value_and_derivative(sign,r):
    q=torch.tensor([[2.+r*.5]],dtype=torch.float64,requires_grad=True)
    p=torch.ones_like(q);z=torch.zeros(1,dtype=torch.float64)
    # Negative-sign observations use negative coordinate offsets.
    qb=q if sign>=0 else -q
    value=uv.pair_objective(p,p,q*0,qb,q*0,q*0,z,z,z+sign,z+2,
        torch.zeros((2,2),dtype=torch.float64),dict(layout_mode='shared_anchor',
        distance_objective='huber',sign_weight=0,metric_weight=0,distance_sigma=.5))
    value.backward()
    torch.testing.assert_close(value,torch.tensor(.5,dtype=torch.float64))
    torch.testing.assert_close(q.grad,torch.tensor([[r*2]],dtype=torch.float64))


def test_invalid_sigma_and_objective():
    for sigma in (0,float('nan'),float('inf')):
        with pytest.raises(ValueError,match='sigma'):loss('huber',20,sigma=sigma)
    with pytest.raises(ValueError,match='distance_objective'):loss('invalid',20)


@pytest.mark.parametrize('mode',['squared','huber','huber_likelihood'])
def test_wrong_sign_distance_is_not_clamped(mode):
    near,_,_=loss(mode,0.)
    far,_,qb=loss(mode,-10.)
    assert far>near+1
    far.backward()
    assert qb.grad[0,1]<0


def test_pair_metric_only_vertical_preserves_distance_phase_correction():
    qa=torch.tensor([[.1,1.3]],dtype=torch.float64,requires_grad=True)
    qb=torch.tensor([[.4,1.7]],dtype=torch.float64,requires_grad=True)
    va=torch.tensor([[0.,64.]],dtype=torch.float64,requires_grad=True)
    vb=torch.tensor([[128.,3000.]],dtype=torch.float64,requires_grad=True)
    z=torch.zeros(1,dtype=torch.float64)
    first=uv.pair_geometry(qa,qb,va,vb,z,z,z+1,{})
    second=uv.pair_geometry(qa+.2,qb-.3,va,vb,z+.15,z,z+1,{})
    expected=.01*((vb[:,None,:]-va[:,:,None])/128).square().clamp(max=100)
    torch.testing.assert_close(first[3],expected)
    torch.testing.assert_close(second[3],expected)
    torch.testing.assert_close(second[0],first[0]-.65)
    gradients=torch.autograd.grad(first[3].sum(),(qa,qb,va,vb),allow_unused=True)
    assert gradients[0] is None and gradients[1] is None
    torch.testing.assert_close(gradients[2],torch.tensor([[-.02/128,-.01/128]],dtype=torch.float64))
    torch.testing.assert_close(gradients[3],torch.tensor([[.03/128,0.]],dtype=torch.float64))


@pytest.mark.parametrize('mode',['likelihood','squared','huber','huber_likelihood'])
def test_no_pair_phase_penalty_in_any_objective(mode):
    p=torch.ones((1,1),dtype=torch.float64);z=torch.zeros(1,dtype=torch.float64)
    table=torch.zeros((2,2),dtype=torch.float64)
    config=dict(layout_mode='shared_anchor',distance_objective=mode,
                sign_weight=0.,distance_weight=0.)
    q=torch.tensor([[.2]],dtype=torch.float64,requires_grad=True)
    value=uv.pair_objective(p,p,q*0,q,q*0,q*0,z,z,z+1,z,table,config)
    torch.testing.assert_close(value,torch.tensor(0.,dtype=torch.float64))
    value.backward()
    torch.testing.assert_close(q.grad,torch.zeros_like(q))
