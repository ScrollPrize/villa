import numpy as np
import pytest
import torch
from vesuvius.neural_tracing.fiber_trace_3d.fiber_cavity_sigma import CavitySigma, combine, grouped_pairs


def graph(edges, means, **cfg):
    n=len(means)
    state=torch.tensor([[m,0.,0.,4.] for m in means],requires_grad=True)
    i,j=torch.tensor(edges,dtype=torch.long).T
    data=dict(i=i,j=j,family=torch.ones(n,dtype=torch.long),ds_a=torch.zeros(len(i)),ds_b=torch.zeros(len(i)),
        sign=torch.ones(len(i)),distance=torch.zeros(len(i)),parallel=torch.ones(len(i),dtype=torch.bool))
    config=dict(cavity_update_every=1,cavity_damping=1.,sigma_max=1.,winding_sigma=.5,**cfg)
    ext=CavitySigma(dict(source_id=np.arange(n),**{k:v.numpy() for k,v in data.items()}),config,'cpu')
    return ext,state,data


def step(ext,state,data,it=0):
    ext.update(state,torch.ones(20)*100,data,it)


def test_one_neighbor_and_reverse_exclusion():
    ext,s,d=graph([(0,1)],[0.,5.])
    step(ext,s,d)
    torch.testing.assert_close(ext.node_var,torch.full((2,),1.25,dtype=torch.float64))
    assert ext.mean[0]==0 and ext.mean[1]==5
    assert not ext.values.requires_grad
    # The recipient's previous message cannot change an outgoing degree-one message.
    ext.mean[1]=100
    step(ext,s,d,1)
    assert ext.mean[0]==0
    assert ext.node_var.max()>ext.values.square().max() # clipping is explicit


def test_stars_agreement_conflict_and_duplicate():
    agree,s,d=graph([(0,1),(0,2)],[0.,0.,0.]);step(agree,s,d)
    conflict,s,d=graph([(0,1),(0,2)],[0.,-2.,2.]);step(conflict,s,d)
    assert agree.node_var[0]==pytest.approx(.625)
    assert conflict.node_var[0]==pytest.approx(4.625)
    dup,s,d=graph([(0,1),(0,1),(0,2)],[0.,0.,0.]);step(dup,s,d)
    torch.testing.assert_close(dup.node_var,agree.node_var)
    assert dup.degree[0]==2


def test_group_reversal_and_disagreement():
    i=torch.tensor([0,1,0]);j=1-i
    a,b,m,v,count=grouped_pairs(i,j,torch.tensor([1.,-1.,3.]),torch.ones(3),2)
    assert (int(a[0]),int(b[0]),int(count[0]))==(0,1,3)
    assert m[0]==pytest.approx(5/3)
    assert v[0]==pytest.approx(1+8/9)


def test_chain_cycles_live_geometry_and_restore(tmp_path):
    ext,s,d=graph([(0,1),(1,2)],[0.,0.,0.])
    for it in range(4):step(ext,s,d,it)
    assert ext.node_var[0]==pytest.approx(1.5) # prior at node2 + two noisy edges
    path=tmp_path/'saved.npz';np.savez(path,**ext.arrays())
    other,_,_=graph([(0,1),(1,2)],[0.,0.,0.])
    with np.load(path) as saved:other.restore(saved)
    with torch.no_grad():s[:,0]+=10
    step(other,s,d,4);step(ext,s,d,4)
    torch.testing.assert_close(other.mean,ext.mean)
    torch.testing.assert_close(other.values,ext.values)
    loop,s,d=graph([(0,1),(1,2),(2,0)],[0.,1.,2.])
    for it in range(30):step(loop,s,d,it)
    assert torch.isfinite(loop.node_var).all() and (loop.node_var>0).all()


def test_live_centers_follow_current_state_not_initial_state():
    ext,s,d=graph([(0,1),(0,2)],[0.,-2.,2.]);step(ext,s,d)
    assert ext.node_var[0]==pytest.approx(4.625)
    with torch.no_grad():s[:,0]=0
    step(ext,s,d,1)
    torch.testing.assert_close(ext.prior_mean,torch.zeros(3,dtype=torch.float64))
    assert ext.node_var[0]==pytest.approx(.625)
    assert not ext.values.requires_grad


def test_live_endpoint_provider_updates_translation_and_widths():
    ext,s,d=graph([(0,1)],[0.,0.])
    calls=[]
    def provider(state,widths,data,edges):
        calls.append(float(widths[0]))
        a=state[data['i'][edges]].clone();b=state[data['j'][edges]].clone()
        # Represents a current shared endpoint, not the anchor's U.
        b[:,1]=widths[0]/100
        z=state.new_zeros(len(edges))
        return a,b,z,z
    ext.update(s,torch.ones(20)*100,d,0,pair_states=provider)
    assert ext.shift[0]==-1
    ext.update(s,torch.ones(20)*200,d,1,pair_states=provider)
    assert ext.shift[0]==-2
    assert calls==[100,200]


def test_cap_and_isolated():
    total=torch.tensor([[20.,20.,0.,80.],[0.,0.,0.,0.]])
    mean,var,dis=combine(total,torch.tensor([0.,3.]),torch.ones(2),cap=8)
    assert var[0]==pytest.approx(4.125) and dis[0]==4
    assert mean[1]==3 and var[1]==1
    ext,s,d=graph([(0,1)],[0.,0.,0.]);step(ext,s,d)
    assert ext.degree[2]==0 and ext.values[2]==1


def test_reject_split_sources():
    with pytest.raises(ValueError,match='whole distinct'):
        CavitySigma(dict(source_id=[1,1],family=[0,0]),{},'cpu')


def test_source_grouped_reentry_does_not_multiply_evidence():
    original,s,d=graph([(0,1)],[0.,0.]);step(original,s,d)
    _,s,d=graph([(0,1),(2,1)],[0.,0.,2.])
    # Reanchoring piece2 by+2 leaves its physical anchor unchanged via phase−2.
    with torch.no_grad():s[2,1]=-2
    raw=dict(source_id=np.array([10,20,10]),**{k:v.numpy() for k,v in d.items()})
    ext=CavitySigma(raw,dict(original.config,cavity_group_source_pieces=True),'cpu')
    step(ext,s,d)
    torch.testing.assert_close(ext.degree,original.degree)
    torch.testing.assert_close(ext.node_var,original.node_var)
    torch.testing.assert_close(ext.values,original.values[torch.tensor([0,1,0])])


def test_fixed_reference_grouped_evidence():
    def make(count):
        ext,s,d=graph([(0,1)],[0.,0.])
        d.update(reference_owner=torch.zeros(count,dtype=torch.long),
            reference_source=torch.full((count,),12,dtype=torch.long),
            reference_state=torch.tensor([[0.,0.,0.,100.]]*count),
            reference_family=torch.ones(count,dtype=torch.long),reference_ds=torch.zeros(count),
            reference_sign=torch.ones(count),reference_distance=torch.zeros(count),
            reference_parallel=torch.ones(count,dtype=torch.bool))
        step(ext,s,d)
        return ext,s,d
    one,s,d=make(1);many,_,_=make(10)
    assert one.reference_pairs==many.reference_pairs==1
    torch.testing.assert_close(one.values,many.values)
    assert one.node_var[0]<.5 and one.degree[0]==2
    torch.testing.assert_close(one.reference_kwargs(d)['sigma_a'],one.values[:1])


def test_signed_translation_and_restore_rejection(tmp_path):
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_soft_winding as soft
    ext,s,d=graph([(0,1)],[2.,4.])
    d['family']=torch.tensor([0,1]);d['parallel'][:]=False
    d['sign'][:]=-1;d['distance'][:]=1.5;d['ds_a'][:]=150
    with torch.no_grad():s[:,1]=torch.tensor([.8,-.2])
    raw=dict(source_id=np.arange(2),**{k:v.numpy() for k,v in d.items()})
    ext=CavitySigma(raw,ext.config,'cpu');step(ext,s,d)
    expected,_p,_v=soft.pair_errors(s[:1],s[1:],d['family'][:1],d['family'][1:],d['ds_a'],d['ds_b'],
        torch.ones(20)*100,d['sign'],d['distance'],d['parallel'])
    assert ext.shift[0]==pytest.approx(float((s[1,0]-s[0,0]-expected/d['sign']).detach()))
    assert ext.shift[1]==-ext.shift[0]
    path=tmp_path/'saved.npz';np.savez(path,**ext.arrays())
    changed=CavitySigma(raw,dict(ext.config,cavity_constraint_sigma=.25),'cpu')
    with np.load(path) as saved, pytest.raises(ValueError,match='inputs/configuration'):
        changed.restore(saved)
