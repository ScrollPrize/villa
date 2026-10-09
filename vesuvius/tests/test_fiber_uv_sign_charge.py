import numpy as np
import pytest
import torch

from vesuvius.neural_tracing.fiber_trace_3d.fiber_uv_layout import (
    calibrated_sign_charge, fixed_sign_terms, proposal_weights,
    hard_geometry, edge_readout, selected_ordering, violation_indices)


def test_warm_width_roundtrip_including_near_lower_bound():
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_uv_layout import inverse_widths,width_boundaries
    w=torch.tensor([100.001,100.1,101.,1000.,80000.],dtype=torch.double)
    restored,_=width_boundaries(inverse_widths(w))
    torch.testing.assert_close(restored,w,rtol=0,atol=1e-12)


def test_warm_start_checks_hash_and_piece_identity(tmp_path):
    import json
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_uv_layout import load_warm_start,sha256_file
    d=dict(source_id=np.array([1]),start=np.array([0]),end=np.array([10]),family=np.array([0]))
    path=tmp_path/'solution.npz';cache=tmp_path/'cache.npz'
    signature={'model':'fixed'}
    (tmp_path/'summary.json').write_text(json.dumps({'input':{'signature':signature}}))
    cache.with_suffix('.json').write_text(json.dumps({'signature':signature}))
    np.savez(path,state=np.zeros((1,303)),widths=np.ones(100)*101,**d)
    cfg=dict(warm_start=str(path),warm_start_sha256=sha256_file(path),tensor_cache=str(cache))
    assert load_warm_start(cfg,d,(1,303))[0].shape==(1,303)
    with pytest.raises(ValueError,match='piece'):load_warm_start(cfg,dict(d,source_id=np.array([2])),(1,303))
    with pytest.raises(ValueError,match='Changed'):load_warm_start(dict(cfg,warm_start_sha256='wrong'),d,(1,303))


def test_tiny_negative_gets_full_charge_not_zero_or_unknown():
    p=torch.tensor([[.8],[.8],[.8]])
    ordered=torch.tensor([[[-1e-12]],[[0.]],[[1e-12]]])
    result=fixed_sign_terms(p,p,ordered,torch.ones(3)*10)
    torch.testing.assert_close(result,torch.tensor([6.4,0.,0.]))


def test_diffuse_selected_pair_is_not_diluted():
    p=torch.full((1,100),.008)
    ordered=torch.ones(1,100,100);ordered[0,0,0]=-1e-8
    torch.testing.assert_close(fixed_sign_terms(p,p,ordered,torch.tensor([10.])),torch.tensor([6.4]))


@pytest.mark.parametrize('q',[.01,.25,.5])
def test_charge_beats_cheaper_endpoint_rejection_cost(q):
    n,e,r=100,1000,10.
    degree_a=torch.tensor([2.]);degree_b=torch.tensor([20.]);mean=torch.tensor(10.)
    charge=calibrated_sign_charge(degree_a,degree_b,mean,e,n,r,4.)
    qa=torch.tensor(q,requires_grad=True);qb=torch.tensor(.5,requires_grad=True)
    cost=fixed_sign_terms((1-qa).reshape(1,1),(1-qb).reshape(1,1),
                         torch.tensor([[[-1e-9]]]),charge).sum()/e
    cost=cost+r/n*(.6*qa+1.5*qb)
    cost.backward()
    assert qa.grad<0  # Gradient descent raises broken probability, including at .5.


def test_proposal_weights_are_unbiased_for_every_event():
    active=torch.tensor([True,True,False,False,False])
    prob=.5/5+.5*active/2
    weight=proposal_weights(active,5,2,.5)
    torch.testing.assert_close(prob*weight,torch.full((5,),.2))
    torch.testing.assert_close(proposal_weights(active,5,0,.5),torch.ones(5))
    with pytest.raises(ValueError):proposal_weights(active,5,2,1.)


def fixture():
    s=np.zeros((2,12),np.float32);s[:,1]=4;s[:,3]=-10
    d=dict(phase=np.array([.1,.20000000001]),z=np.zeros(2),family=np.array([0,1]),
           i=np.array([0]),j=np.array([1]),ds_a=np.array([1.00000000001]),ds_b=np.array([2.]),
           phase_delta=np.array([.123456789]),sign=np.array([1.]))
    return d,s,np.array([10.,20.,30.],np.float32)


def test_selected_geometry_matches_canonical_double_audit():
    d,s,w=fixture();config={'layout_mode':'shared_anchor','broken_threshold':.5,'sign_tolerance':0}
    geom=hard_geometry(d,s,w,config)
    expected=edge_readout(d,geom,w,np.array([0]))[2]
    t=lambda x:torch.tensor(x)
    actual=selected_ordering(t(s[:1]),t(s[1:]),t(d['phase'][:1]),t(d['phase'][1:]),
        t(d['z'][:1]),t(d['z'][1:]),t(d['family'][:1]),t(d['family'][1:]),
        t(d['ds_a']),t(d['ds_b']),t(w),t(d['phase_delta']),t(d['sign']))
    np.testing.assert_allclose(actual.numpy(),expected,rtol=0,atol=1e-14)
    d['sign'][:]=-1
    np.testing.assert_array_equal(violation_indices(d,s,w,config),[0])


def test_broken_threshold_does_not_use_joint_argmax():
    d,s,w=fixture();s[:,:4]=0;s[:,3]=np.log(2.)  # Broken .4 is largest but below .5.
    assert not hard_geometry(d,s,w,{'layout_mode':'shared_anchor'})[2].any()
    assert hard_geometry(d,s,w,{'layout_mode':'shared_anchor','broken_threshold':.5})[2].all()
    s[:,3]=np.log(4.)
    assert not hard_geometry(d,s,w,{'layout_mode':'shared_anchor','broken_threshold':.5})[2].any()
