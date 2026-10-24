import numpy as np
import torch
from vesuvius.neural_tracing.fiber_trace_3d.fiber_uv_layout import (
    coordinates,q_to_u,u_to_q,anchor_offsets,offset_regularization,
    compatibility_loss,continuation_likelihood,pair_objective,make_distance_table,audit,
    aligned_geometry,direction_inputs)


def test_aligned_geometry_uses_whole_fiber_sign_even_through_fold():
    points=np.array([[10.,0,0],[10,2,2],[10,1,1]])
    axis=np.array([[0.,0,-10],[0,0,10]])
    a,t,phase,_=aligned_geometry(points,0,axis,1,'ccw')
    ar,tr,pr,_=aligned_geometry(points,0,axis,-1,'cw')
    np.testing.assert_allclose(t,a);np.testing.assert_allclose(tr,-a)
    np.testing.assert_allclose(pr,-phase)
    _,v,_,_=aligned_geometry(points,1,axis,1,'cw')
    np.testing.assert_allclose(v,a)  # No local +/-Z flips at the fold.


def test_frozen_direction_provenance_and_masks(tmp_path):
    import json
    import pytest
    ids=np.arange(4);labels=np.array([0,0,1,2])
    signature={key:key for key in ('database','afv','features','stage1','hv_solution','umbilicus')}
    signature['h_direction_solution']='saved_h_hash'
    config={}
    hsign=np.array([1,0,0,0]);vsign=np.array([0,0,-1,0])
    for family in ('h','v'):
        root=tmp_path/family;root.mkdir();config[family+'_direction_solution']=str(root/'solution.npz')
        meta=dict(signature)
        if family=='v':meta['inherited_h_solution']='saved_h_hash'
        (root/'summary.json').write_text(json.dumps({'input':{'signature':meta}}))
        extra=dict(direction_sign_ccw=hsign,direction_sign_cw=-hsign) if family=='h' else dict(
            direction_sign_up=vsign,inherited_h_direction_sign_ccw=hsign,inherited_h_direction_sign_cw=-hsign)
        np.savez(root/'solution.npz',ids=ids,labels=np.array([0,2,1,2]),**extra)
    frozen,signs=direction_inputs(config,signature,ids,labels)
    np.testing.assert_array_equal(frozen,[0,2,1,2]);assert signs[0]==1 and signs[2]==-1
    config['h_convention']='cw'
    _,reverse=direction_inputs(config,signature,ids,labels)
    assert reverse[0]==-1 and reverse[2]==-1
    with pytest.raises(ValueError,match='IDs'):direction_inputs(config,signature,ids+1,labels)
    with pytest.raises(ValueError,match='hash'):direction_inputs(config,dict(signature,h_direction_solution='wrong'),ids,labels)


def test_unwrapped_anchor_seams_and_inverse():
    w=torch.tensor([10.,20.,30.],dtype=torch.double)
    b=torch.cat((w.new_zeros(1),w.cumsum(0)))
    q=torch.tensor([-.5,.999,1.,1.001,2.9,3.5],dtype=torch.double,requires_grad=True)
    torch.testing.assert_close(u_to_q(q_to_u(q,w,b),w,b),q)
    assert torch.autograd.gradcheck(lambda q:q_to_u(q,w,b),(q[[0,1,3,4,5]],))
    s=torch.zeros(2,12,dtype=torch.double);s[:,4]=torch.tensor([-.6,1.2])
    u,v,actual=coordinates(s,torch.zeros(2),torch.zeros(2),torch.zeros(2),torch.zeros(2),w,b)
    torch.testing.assert_close(actual,torch.arange(3)[None,:]+s[:,4,None])


def test_centered_offsets_and_strong_quarter_hinge():
    s=torch.zeros(1,12,requires_grad=True)
    with torch.no_grad():s[0,6:9]=torch.tensor([.3,-.3,0.])
    du,dv=anchor_offsets(s,3)
    assert abs(du.mean())<1e-7
    loss=offset_regularization(s,3,{'offset_soft_weight':0.,'offset_limit_weight':100.})
    torch.testing.assert_close(loss,torch.tensor(100*2*.05**2/3))
    loss.backward();assert s.grad[0,6]>0 and s.grad[0,7]<0
    shifted=s.detach().clone();shifted[:,6:9]+=1
    torch.testing.assert_close(anchor_offsets(shifted,3)[0],du)


def test_compatible_low_probability_is_reinforced():
    logits=torch.tensor([[0.,-2.]],requires_grad=True)
    a=torch.tensor([[1.,0.]])
    cost=torch.tensor([[[8.,0.],[0.,8.]]])
    loss=compatibility_loss(a,logits.softmax(1),cost)
    loss.backward()
    assert logits.grad[0,1]<0 and logits.grad[0,0]>0
    assert compatibility_loss(a,torch.tensor([[0.,1.]]),cost)<loss
    assert compatibility_loss(a*0,logits.softmax(1),cost)==0
    torch.testing.assert_close(compatibility_loss(a*.5,logits.softmax(1)*.25,cost),loss.detach()*.125)


def test_likelihood_matches_direct_and_continuation_translation():
    pa=torch.tensor([[.2,.6]],dtype=torch.double);pb=torch.tensor([[.1,.8]],dtype=torch.double)
    cost=torch.tensor([[[1.,2.],[3.,.4]]],dtype=torch.double,requires_grad=True)
    expected=-.8*.9*torch.log(((pa/.8)[:,:,None]*(pb/.9)[:,None,:]*torch.exp(-cost)).sum())
    torch.testing.assert_close(compatibility_loss(pa,pb,cost),expected)
    assert torch.autograd.gradcheck(lambda c:compatibility_loss(pa,pb,c),(cost,))
    a=torch.tensor([[10.,20.]]);b=torch.tensor([[11.,21.]]);p=torch.tensor([[.3,.7]])
    torch.testing.assert_close(continuation_likelihood(p,p,a,b,a,b,10),
                               continuation_likelihood(p,p,a+100000,b+100000,a,b,10))


def test_separate_sign_penalty_cannot_hide_in_compatible_pair():
    p=torch.tensor([[.5,.5]]);qa=torch.tensor([[0.,1.]]);qb=torch.tensor([[0.,1.]])
    z=torch.zeros_like(qa);zero=torch.zeros(1);table=torch.tensor(make_distance_table(1,.5))
    args=(p,p,qa,qb,z,z,zero,zero,zero+1,zero,table)
    weak=pair_objective(*args,{'layout_mode':'shared_anchor','sign_weight':0})
    strong=pair_objective(*args,{'layout_mode':'shared_anchor','sign_weight':10})
    torch.testing.assert_close(strong-weak,torch.tensor(10.))


def test_anchor_audit_matches_derived_coordinates():
    k=3;s=np.zeros((2,12),np.float32);s[:,1]=4;s[:,4]=[-.6,1.2]
    w=np.array([10.,20.,30.],np.float32)
    d=dict(phase=np.array([.1,.2]),z=np.zeros(2),family=np.array([0,1]),
           i=np.array([0]),j=np.array([1]),ds_a=np.array([1.]),ds_b=np.array([2.]),
           phase_delta=np.array([0.]),sign=np.array([1.]),distance=np.array([2.]),
           parallel=np.array([False]),links=np.empty((0,4)),lengths=np.ones(2))
    out=audit(d,s,w,{'layout_mode':'shared_anchor'})
    _,_,q=coordinates(torch.tensor(s),torch.tensor(d['phase']),torch.tensor(d['z']),
        torch.tensor(d['family']),torch.tensor([1.,2.]),torch.tensor(w),torch.tensor([0.,10.,30.,60.]))
    np.testing.assert_allclose(out['endpoint_q_range'],[q[0,1],q[1,1]])
    assert out['min_bucket_separation_turns']==1
