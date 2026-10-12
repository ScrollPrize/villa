import numpy as np
import pytest
import torch
from vesuvius.neural_tracing.fiber_trace_3d import fiber_uv_layout as uv


CFG=dict(layout_mode='shared_anchor',distance_objective='squared',
         bucket_position_split=True,metric_weight=0.,position_distance_weight=.1)


def test_categorical_does_not_silently_ignore_split():
    with pytest.raises(ValueError,match='cost_target'):
        uv.solve(dict(CFG,assignment_update='cost_target'))


def objective(qa,qb,**kwargs):
    cfg=dict(CFG,**kwargs.pop('config',{}))
    a=torch.tensor([[1.,0.]])
    b=torch.tensor([[0.,1.]])
    z=torch.zeros_like(qa)
    one=torch.ones(1)
    return uv.pair_objective(a,b,qa,qb,z,z,torch.zeros(1),torch.zeros(1),
                             one,one,None,cfg,**kwargs)


def test_position_weight_and_independent_topology():
    qa=torch.tensor([[0.,1.]],requires_grad=True)
    qb=torch.tensor([[0.,3.]],requires_grad=True)
    # Chosen buckets differ by1 (perfect); positions differ by3 -> 8*.1.
    value=objective(qa,qb)
    torch.testing.assert_close(value,torch.tensor(.8))
    value.backward()
    torch.testing.assert_close(qb.grad,torch.tensor([[0.,.8]]))
    for q in (qb,qb+1000):
        assert objective(qa,q,config={'position_distance_weight':0.}).item()==0


def test_sign_charge_ignores_position_selected_order():
    qa=torch.tensor([[100.,101.]])
    qb=torch.tensor([[-100.,-99.]])
    # Bucket ordering is correct even though position order is wrong.
    value=objective(qa,qb,config={'position_distance_weight':0.},
                    charge=torch.tensor([1000.]),selected_ordered=torch.tensor([-200.]))
    assert value.item()==0
    # Fixed correction can make selected bucket order wrong; expected and
    # selected charge agree, and hinge is40 per wrong turn.
    value=objective(qa,qb,config={'position_distance_weight':0.},
                    charge=torch.tensor([1000.]),bucket_shift=torch.tensor([-2.]))
    assert value.item()==1048.


def test_fixed_seam_and_endpoint_shift():
    d=dict(i=np.array([0,0]),j=np.array([1,1]),
           phase=np.array([.95,.05]),
           node_phase=np.array([[.95],[.05]]),
           phase_a=np.array([.95,1.1]),phase_b=np.array([.05,.1]),
           phase_delta=np.array([.1,0.]))
    np.testing.assert_array_equal(uv.bucket_event_shift(d),[-1,-1])


def test_float32_anchor_rounded_up_at_seam():
    d=dict(i=np.array([0]),j=np.array([1]),phase=np.array([.999999999,.05]),
           node_phase=np.array([[1.],[.05]],dtype=np.float32),
           phase_a=np.array([1.]),phase_b=np.array([.05]),phase_delta=np.array([.05]))
    np.testing.assert_array_equal(uv.bucket_event_shift(d),[-1])


def test_shared_phase_changes_topology_and_has_gradient():
    phase=torch.tensor([0.,-2.],requires_grad=True)
    value=objective(torch.zeros(1,2),torch.zeros(1,2),
                    config={'position_distance_weight':0.},
                    bucket_shift=(phase[1]-phase[0]).reshape(1))
    assert value.item()==48.
    value.backward()
    torch.testing.assert_close(phase.grad,torch.tensor([48.,-48.]))


def test_shared_phase_original_fraction_not_double_counted():
    d=dict(i=np.array([0]),j=np.array([1]),phase=np.array([.95,.05]),
           node_phase=np.array([[.95],[.05]]),phase_a=np.array([1.1]),
           phase_b=np.array([.1]),phase_delta=np.array([0.]))
    # Original endpoint difference -1, plus learned delta +.3 => -.7.
    correction=uv.bucket_event_shift(d)+.3
    np.testing.assert_allclose(correction,[-.7])
    with pytest.raises(ValueError,match='shared-anchor'):
        uv.solve(dict(bucket_shared_phase=True,bucket_position_split=True))


def test_shared_phase_readout_and_global_shift_invariance():
    d=dict(i=np.array([0]),j=np.array([1]),family=np.array([0,1]),
           ds_a=np.array([0.]),ds_b=np.array([0.]),phase_delta=np.array([.1]),
           bucket_shift=np.array([-1.]))
    geometry=(None,None,np.ones(2,dtype=bool),np.array([10,11]),
              np.array([100.,-100.]),np.zeros(2),np.array([0.,200.]))
    state=np.zeros((2,4));state[:,2]=[.2,1.2]
    cfg=dict(CFG,bucket_shared_phase=True)
    assert uv.edge_readout(d,geometry,np.array([200.]),np.array([0]),cfg,state)[2][0]==1.5
    state[:,2]+=10
    np.testing.assert_allclose(uv.edge_readout(d,geometry,np.array([200.]),np.array([0]),cfg,state)[2],[1.5])
    with pytest.raises(ValueError,match='requires state'):
        uv.edge_readout(d,geometry,np.array([200.]),np.array([0]),cfg)


def test_readout_uses_bucket_not_width():
    d=dict(i=np.array([0]),j=np.array([1]),family=np.array([0,1]),
           ds_a=np.array([0.]),ds_b=np.array([0.]),phase_delta=np.array([.1]),
           bucket_shift=np.array([-1.]))
    geometry=(None,None,np.array([True,True]),np.array([10,11]),
              np.array([100.,-100.]),np.zeros(2),np.array([0.,200.]))
    *_,delta,active=uv.edge_readout(d,geometry,np.array([200.]),np.array([0]),CFG)
    assert delta[0]==.5
    assert active[0]


@pytest.mark.parametrize('update',[{'distance_objective':'likelihood'},
    {'local_window_radius':1},{'position_distance_weight':-1}])
def test_invalid_modes(update):
    with pytest.raises(ValueError):
        objective(torch.zeros(1,2),torch.zeros(1,2),config=update)
