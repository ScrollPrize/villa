import numpy as np
import pytest
import torch
from vesuvius.neural_tracing.fiber_trace_3d import fiber_uv_event_offsets as eo
from vesuvius.neural_tracing.fiber_trace_3d import fiber_uv_layout as uv


def test_short_arc_shared_parameters_exact_and_gradients():
    meta=eo.adjacency([1,1,1,1,2],[0,2,5,9,0],[0,0,0,0,1],4.)
    assert meta['parameter_index'].tolist()==[0,0,0,1,2]
    assert meta['parameter_count']==3
    top=eo.tensor_adjacency(meta,'cpu')
    p=torch.tensor([[.1,.2],[.11,.21],[.3,.4]],dtype=torch.double,requires_grad=True)
    expanded=p[top['parameter_index']]
    assert torch.equal(expanded[0],expanded[2])
    expanded.sum().backward()
    torch.testing.assert_close(p.grad,torch.tensor([[3.,3.],[1.,1.],[1.,1.]],dtype=torch.double))
    assert torch.autograd.gradcheck(lambda x:eo.regularization(x,top,{})[0],(p,))
    stats=eo.diagnostics(1000*expanded.detach().numpy(),meta)
    assert stats['tied_edges']==2 and stats['tied_delta_max']==0


def test_adjacency_keeps_coincident_nodes_and_uses_sources():
    m=eo.adjacency([2,1,1,1,2],[4,10,0,10,0],[1,0,0,0,1])
    assert m['left'].tolist()==[2,1,4]
    assert m['right'].tolist()==[1,3,0]
    assert m['gap'].tolist()==[10,0,4]


def test_regularizer_units_spacing_and_hv_symmetry():
    cfg=dict(event_offset_longitudinal_weight=3,event_offset_transverse_weight=5,event_offset_anchor_weight=.1)
    p=torch.tensor([[0.,0.],[.01,.02]],dtype=torch.double,requires_grad=True)
    top=eo.tensor_adjacency(eo.adjacency([1,1],[0,100],[0,0]),'cpu')
    loss,along,across,anchor=eo.regularization(p,top,cfg)
    assert along.item()==pytest.approx(.01)  # (10/100)^2
    assert across.item()==pytest.approx(.04)
    other=eo.tensor_adjacency(eo.adjacency([1,1],[0,100],[1,1]),'cpu')
    torch.testing.assert_close(loss,eo.regularization(p.flip(1),other,cfg)[0])
    assert torch.autograd.gradcheck(lambda x:eo.regularization(x,top,cfg)[0],(p,))
    # Same linear field, more nodes: integrated slope energy unchanged.
    mid=torch.tensor([[0.,0.],[.005,.01],[.01,.02]],dtype=torch.double)
    dense=eo.tensor_adjacency(eo.adjacency([1,1,1],[0,50,100],[0,0,0]),'cpu')
    _,a,b,_=eo.regularization(mid,dense,cfg)
    torch.testing.assert_close(a,along);torch.testing.assert_close(b,across)


def test_coincident_regularization_finite_and_anchor_constant_drift():
    top=eo.tensor_adjacency(eo.adjacency([1,1],[0,0],[0,0]),'cpu')
    p=torch.tensor([[.1,.1],[.1,.1]],requires_grad=True)
    loss,a,b,c=eo.regularization(p,top,{})
    assert a==0 and b==0 and c>0
    loss.backward();assert torch.isfinite(p.grad).all() and (p.grad>0).all()
    p2=torch.tensor([[0.,0.],[.1,.2]],requires_grad=True)
    loss=eo.regularization(p2,top,{})[0];loss.backward()
    assert torch.isfinite(p2.grad).all() and loss>0


def test_roundoff_coincident_arcs_not_singular_but_subvoxel_gaps_preserved():
    p=torch.tensor([[0.,0.],[.001,.002]])
    scores=[]
    for gap in [0.,1e-12,1e-4]:
        top=eo.tensor_adjacency(eo.adjacency([1,1],[0,gap],[0,0]),'cpu')
        scores.append(eo.regularization(p,top,{})[0])
    torch.testing.assert_close(scores[0],scores[1])
    assert scores[2]>1000*scores[1]


def test_offset_geometry_shared_across_windings_and_gradients():
    k=4;s=torch.zeros(2,3*k+3,dtype=torch.double)
    phase=torch.tensor([.2,.8],dtype=torch.double);z=torch.zeros(2,dtype=torch.double)
    fam=torch.tensor([0,1]);ds=torch.tensor([13.,-17.],dtype=torch.double)
    w=torch.tensor([200.,300.,400.,500.],dtype=torch.double,requires_grad=True)
    offsets=torch.tensor([[20.,-30.],[-15.,40.]],dtype=torch.double,requires_grad=True)
    bounds=torch.cat([w.new_zeros(1),w.cumsum(0)])
    base=uv.coordinates(s,phase,z,fam,ds,w,bounds)
    shifted=uv.coordinates(s,phase,z,fam,ds,w,bounds,uv_offset=offsets)
    torch.testing.assert_close(shifted[0],base[0]+offsets[:,0,None])
    torch.testing.assert_close(shifted[1],base[1]+offsets[:,1,None])
    torch.testing.assert_close(shifted[2],uv.u_to_q(shifted[0],w,bounds))
    def evaluate(o,width):
        b=torch.cat([width.new_zeros(1),width.cumsum(0)])
        return uv.coordinates(s,phase,z,fam,ds,width,b,'combined',uv_offset=o)
    assert torch.autograd.gradcheck(evaluate,(offsets,w))
    combined=evaluate(offsets,w)
    for a,b in zip(combined,shifted):torch.testing.assert_close(a,b)


def test_offsets_cannot_change_coarse_loss():
    k=3;p=torch.tensor([[.1,.2,.3]])
    q=torch.tensor([[.2,1.2,2.2]],requires_grad=True);v=torch.zeros_like(q)
    cfg=dict(layout_mode='shared_anchor',bucket_position_split=True,distance_objective='squared',
             position_distance_weight=0,metric_weight=0,sign_weight=20)
    zero=torch.zeros(1);one=torch.ones(1)
    def loss(x):return uv.pair_objective(p,p,x,q,v,v,zero,one,one,one,torch.zeros(2),cfg,bucket_shift=zero)
    torch.testing.assert_close(loss(q),loss(q+1000))
    assert torch.autograd.grad(loss(q),q)[0].abs().max()==0


def test_prepare_reads_true_arcs_and_serializes(tmp_path):
    import sqlite3
    path=tmp_path/'events.sqlite'
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE constraints(id INTEGER PRIMARY KEY,fiber_a,fiber_b,arc_a,arc_b)')
        db.executemany('INSERT INTO constraints VALUES(?,?,?,?,?)',[(7,10,20,40.,90.),(8,10,20,60.,70.)])
    d=dict(event_ids=np.array([7,8]),i=np.array([0,1]),j=np.array([2,2]),
           source_id=np.array([10,10,20]),family=np.array([0,0,1]),
           start=np.array([0,50,0]),end=np.array([50,100,100]))
    m=eo.prepare(d,dict(database=str(path)))
    assert m['arc'].tolist()==[40,60,90,70]
    assert m['gap'].tolist()==[20,20]  # first gap crosses a piece cut
    d['links']=np.array([[0,1,50,0]])
    m=eo.prepare(d,dict(database=str(path)))
    assert m['arc'].tolist()==[40,60,90,70,50,50]
    assert m['kind'].tolist()==[0,0,0,0,1,1]
    assert m['side'].tolist()==[0,0,1,1,0,1]
    assert m['gap'].tolist()==[10,0,10,20]
    np.savez(tmp_path/'saved.npz',**m)
    with np.load(tmp_path/'saved.npz') as saved:
        for key in m:np.testing.assert_array_equal(m[key],saved[key])
