import numpy as np
import pytest
import torch
from vesuvius.neural_tracing.fiber_trace_3d import fiber_uv_layout as uv


CFG=dict(constraint_class_balance=True,perpendicular_relative_weight=10)


def test_recommended_state_lr_default():
    assert uv.DEFAULT_STATE_LR == pytest.approx(0.007348469228349533 * 10)


def test_full_graph_class_totals_and_default_identity():
    parallel=np.r_[np.ones(19222,bool),np.zeros(5549,bool)]
    w=uv.constraint_class_weights(parallel,CFG)
    np.testing.assert_allclose(w.mean(),1,rtol=1e-6)
    np.testing.assert_allclose(w[~parallel].sum()/w[parallel].sum(),10,rtol=1e-6)
    np.testing.assert_array_equal(uv.constraint_class_weights(parallel,{}),1)


@pytest.mark.parametrize('kind',[True,False])
def test_absent_class_renormalizes(kind):
    np.testing.assert_array_equal(uv.constraint_class_weights(np.full(9,kind),CFG),1)
    assert uv.constraint_class_weights([],CFG).size==0


@pytest.mark.parametrize('value',[0,-1,float('nan'),float('inf')])
def test_invalid_ratio(value):
    with pytest.raises(ValueError,match='positive'):
        uv.constraint_class_weights([True,False],dict(CFG,perpendicular_relative_weight=value))


def test_unsupported_solver_fails():
    with pytest.raises(ValueError,match='cost_target'):
        uv.constraint_class_weights([True,False],dict(CFG,assignment_update='cost_target'))


@pytest.mark.parametrize('term',['distance','sign','vertical','fixed'])
def test_all_pair_terms_and_gradients_have_same_class_weight(term):
    parallel=np.array([True,True,True,False])
    weights=torch.tensor(uv.constraint_class_weights(parallel,CFG),dtype=torch.double)
    cfg=dict(layout_mode='shared_anchor',distance_objective='squared',
             distance_weight=float(term=='distance'),sign_weight=float(term=='sign'),
             metric_weight=float(term=='vertical'))
    q=torch.tensor([[-.1,-1.]]*4,dtype=torch.double,requires_grad=True)
    logits=torch.tensor([[1.,0.,-2.]]*4,dtype=torch.double,requires_grad=True)
    p=logits.softmax(1)[:,:2];z=torch.zeros(4,dtype=torch.double)
    charge=torch.ones(4,dtype=torch.double)*30 if term=='fixed' else None
    args=(p,p,q*0,q,q*0,q+30,z,z,z+1,z+1,torch.zeros(2,2),cfg,charge)
    plain=uv.pair_objective(*args)
    plain_grad=torch.autograd.grad(plain,(q,logits),retain_graph=True)
    weighted=uv.pair_objective(*args,event_weights=weights)
    weighted_grad=torch.autograd.grad(weighted,(q,logits))
    torch.testing.assert_close(weighted,plain)
    for a,b in zip(weighted_grad,plain_grad):
        torch.testing.assert_close(a,b*weights[:,None])


def test_fixed_charge_outweighs_rejection_after_parallel_downweight():
    events,pieces=24771,2199
    w=float(uv.constraint_class_weights([True]*19222+[False]*5549,CFG)[0])
    q=torch.tensor(.5,requires_grad=True)
    charge=uv.calibrated_sign_charge(torch.ones(1),torch.ones(1),1.,events,pieces,10.,32.,'fixed')
    penalty=uv.fixed_sign_terms((1-q).reshape(1,1),torch.tensor([[.5]]),
        torch.tensor([[[-1e-12]]]),charge)[0]*w/events+10*q/pieces
    penalty.backward()
    assert q.grad<0
