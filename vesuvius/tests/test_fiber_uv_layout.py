import numpy as np
import pytest
import torch
from vesuvius.neural_tracing.fiber_trace_3d.fiber_uv_layout import (
    split_bounds,fiber_geometry,signed_order,u_to_q,coordinates,pair_objective,
    make_distance_table,table_cost,continuation_cost,audit)


def test_split_no_event_loss_or_illegal_remainders():
    arcs=np.array([100,200,300,4500,4600,4700,8800,8900,9000])
    b=split_bounds(13000,arcs,4166)
    assert np.all(np.diff(b)>=4166)
    owners=np.searchsorted(b[1:-1],arcs,side='right')
    assert np.all(np.bincount(owners,minlength=len(b)-1)>=3)
    np.testing.assert_array_equal(split_bounds(2000,[100,200,300],4166),[0,2000])
    np.testing.assert_array_equal(split_bounds(10000,[100,200],4166),[0,10000])


def test_arc_map_reversal_and_angle_seam():
    theta=np.linspace(3,3.3,20);xyz=np.c_[100*np.cos(theta),100*np.sin(theta),theta*10]
    axis=np.array([[0,0,0],[0,0,100]])
    for family in [0,1]:
        a,t,p,r=fiber_geometry(xyz,family,axis)
        ar,tr,pr,rr=fiber_geometry(xyz[::-1],family,axis)
        np.testing.assert_allclose(t-t[8],tr[::-1]-tr[-9],atol=1e-10)
        np.testing.assert_allclose(p-p[0],pr[::-1]-pr[-1],atol=1e-10)
        assert np.max(abs(np.diff(p)))<.01


def test_normals_use_same_connector_and_unknown_preserved():
    c=np.array([[1,0,0]]*3);a=np.array([[-1,0,0]]*3);b=np.array([[-1,0,0],[1,0,0],[0,1,0]])
    np.testing.assert_array_equal(signed_order(c,a,b),[1,0,0])
    np.testing.assert_array_equal(signed_order(-c,b,a),[-1,0,0])


def test_inverse_widths_seams_and_gradient():
    w=torch.tensor([10.,20.,30.],dtype=torch.double,requires_grad=True)
    u=torch.tensor([-.1,9.99,10.,10.01,29.99,30.01,60.1],dtype=torch.double,requires_grad=True)
    def inverse(u,w):return u_to_q(u,w,torch.cat((w.new_zeros(1),w.cumsum(0))))
    q=inverse(u,w)
    assert q[2]==1
    assert torch.all(q[1:]-q[:-1]>0)
    # Exact seam derivative is one-sided; finite differences only away from seams.
    assert torch.autograd.gradcheck(inverse,(u[[0,1,3,4,5,6]],w))


def test_distribution_truncation_and_table_accuracy():
    from scipy.special import logsumexp
    sigma=.5;table=make_distance_table(3.,sigma)
    rng=np.random.default_rng(1)
    mu=rng.uniform(0,3,30);x=rng.uniform(0,6,(30,2,2));support=np.arange(0,104.5,.5)
    expected=[]
    for m,xx in zip(mu,x):
        lw=-.5*((support-m)/sigma)**2;lw-=logsumexp(lw)
        assert np.isclose(np.exp(lw).sum(),1)
        expected.append(-logsumexp(lw[:,None,None]-.5*((support[:,None,None]-xx)/.25)**2,axis=0))
    actual=table_cost(torch.tensor(x),torch.tensor(mu),torch.tensor(table)).numpy()
    np.testing.assert_allclose(actual,expected,atol=.006,rtol=0)


def test_pair_expectation_matches_enumeration_and_gradients():
    torch.manual_seed(0);dtype=torch.double
    table=torch.tensor(make_distance_table(3,.5),dtype=dtype)
    pa=torch.tensor([[.3,.5]],dtype=dtype);pb=torch.tensor([[.4,.5]],dtype=dtype)
    qa=torch.tensor([[.1,1.1]],dtype=dtype,requires_grad=True);qb=torch.tensor([[.12,1.12]],dtype=dtype,requires_grad=True)
    z=torch.zeros((1,2),dtype=dtype);zero=torch.zeros(1,dtype=dtype)
    def fun(a,b):return pair_objective(pa,pb,a,b,z,z,zero,torch.ones_like(zero),torch.ones_like(zero),
                                     zero+.5,table,{})
    total=0
    for i in range(2):
        for j in range(2):
            total+=pair_objective(pa[:,i:i+1],pb[:,j:j+1],qa[:,i:i+1],qb[:,j:j+1],z[:,:1],z[:,:1],
                zero,zero+1,zero+1,zero+.5,table,{})
    torch.testing.assert_close(fun(qa,qb),total)
    assert torch.autograd.gradcheck(fun,(qa,qb),atol=1e-5)


def test_continuation_moments_and_seam_identity():
    pa=torch.tensor([[.2,.7]]);pb=torch.tensor([[.3,.6]])
    a=torch.tensor([[10.,30.]]);b=torch.tensor([[11.,32.]])
    z=torch.zeros_like(a)
    expected=(pa[:,:,None]*pb[:,None,:]*((a[:,:,None]-b[:,None,:])/10).square()).sum()
    torch.testing.assert_close(continuation_cost(pa,pb,a,b,z,z,10),expected)
    # Piece A anchor at U=9, endpoint+1; B anchor at U=11, endpoint-1.
    widths=torch.tensor([10.,20.]);bounds=torch.tensor([0.,10.,30.]);state=torch.zeros((2,7))
    u,v,q=coordinates(state,torch.tensor([.9,.05]),torch.zeros(2),torch.zeros(2,dtype=torch.long),
                      torch.tensor([1.,-1.]),widths,bounds)
    assert u[0,0]==u[1,1]==10


def test_coherent_and_inconsistent_sign_cost():
    table=torch.tensor(make_distance_table(1,.5));p=torch.ones(1,1);z=torch.zeros(1,1);zero=torch.zeros(1)
    def loss(q):return pair_objective(p,p,z,q,z,z,zero,zero+1,zero+1,zero+.5,table,{})
    assert loss(z)<loss(z-1)
    # Rejected endpoint has no attached pair cost.
    assert pair_objective(p*0,p,z,z-1,z,z,zero,zero+1,zero+1,zero+.5,table,{})==0


def test_continuation_large_translation():
    p=torch.tensor([[0.,1.]])
    a=torch.tensor([[0.,1000000.]])
    b=torch.tensor([[0.,1000001.]])
    z=torch.zeros_like(a)
    torch.testing.assert_close(continuation_cost(p,p,a,b,z,z,1.),torch.tensor(1.))


def test_all_initial_hypotheses_reachable():
    logits=np.maximum(-14.,-.5*((np.arange(100)-3)/3)**2).astype(np.float32)
    p=torch.tensor(logits).softmax(0)
    assert (p>0).all()


def test_signed_boundary_no_reflection_swap_and_negative_gradient():
    table=torch.tensor(make_distance_table(3,.5));p=torch.ones(1,1);zero=torch.zeros(1);z=zero[:,None]
    def loss(a,b,sgn):return pair_objective(p,p,a,b,z,z,zero,zero,zero+sgn,zero+2,table,{})
    a=z.clone();b=torch.tensor([[-2.]],requires_grad=True)
    value=loss(a,b,1);value.backward()
    assert b.grad.item()<0
    assert value>loss(a,-b,1)
    torch.testing.assert_close(value,loss(b,a,-1))
    assert abs(loss(a,z-1e-6,1)-loss(a,z+1e-6,1))<.001


def test_prepare_reads_store_fibers_by_uuid(tmp_path):
    import json
    from lasagna.inference_provenance import sha256_file
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_uv_layout import prepare, report
    from vesuvius.neural_tracing.tracers.shared.annotations.fiber_store import fibers_digest, fiber_source_version
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_graph_store import store_digest
    from .fiber_graph_fixture import uid, measured_store
    ids=list(map(uid,[3,4,5]))
    theta=np.linspace(0,1,11)
    curves=[np.c_[50+20*np.cos(theta),50+20*np.sin(theta),np.full(11,40.)],
            np.c_[50+30*np.cos(theta),50+30*np.sin(theta),np.full(11,40.)],
            np.c_[np.full(11,75.),np.full(11,50.),np.linspace(10,90,11)]]
    store=tmp_path/'fibers.vcobj'
    lengths=[float(np.linalg.norm(np.diff(x,axis=0),axis=1).sum()) for x in curves]
    axis=tmp_path/'axis.json'
    axis.write_text(json.dumps(dict(metadata=dict(volume='123-scan.zarr',volume_slices=100,volume_height=100,volume_width=100,
        voxelsize_um=2.4),control_points=[dict(x=50,y=50,z=0),dict(x=50,y=50,z=99)])))
    # Persistent relation UUIDs can derive from large source ordinals; tensor
    # event indices remain dense positions, not lossy floating source IDs.
    event_ids=[2**53+7,2**53+9,2**53+11]
    # Endpoint arcs need not match points here; prepare only checks they lie on the fiber.
    database=measured_store(tmp_path,[(event_ids[0],3,4,False,1.),(event_ids[1],4,5,False,1.),(event_ids[2],3,5,True,.5)],
        curves=dict(zip([3,4,5],curves)),volume='123',
        field_overrides={eid:dict(arc_a=a,arc_b=b) for eid,a,b in zip(event_ids,[1.,3.,5.],[2.,4.,6.])},
        point_overrides={eid:(curves[a][1].tolist(),curves[b][1].tolist())
                         for eid,a,b in zip(event_ids,[0,1,0],[1,2,2])})
    stage1=tmp_path/'stage1.npz'
    np.savez(stage1,ids=ids,normal_sign=np.ones(3,np.int8),sign_logit=np.ones(3),sign_probability=np.ones(3),
             broken=np.zeros(3,bool),normal_convention=np.asarray('inward'))
    hv=tmp_path/'hv'/'solution.npz';hv.parent.mkdir()
    np.savez(hv,ids=np.asarray(ids),labels=np.array([0,0,1],np.uint8))
    (hv.parent/'summary.json').write_text(json.dumps(dict(input=dict(signature=dict(database=store_digest(database),stage1=sha256_file(stage1))))))
    features=tmp_path/'features.json'
    features.write_text(json.dumps(dict(format='fiber-axial-priors',version=1,source_ids=ids,
        agreement=[0,0,0],fibers_digest=fibers_digest(store),fiber_source=fiber_source_version(store),
        umbilicus_sha256=sha256_file(axis))))
    cache=tmp_path/'tensors.npz'
    config=dict(tensor_cache=str(cache),database=str(database),features=str(features),stage1=str(stage1),
                hv_solution=str(hv),fibers=str(store),umbilicus=str(axis),winding_count=4)
    prepare(config)
    with np.load(cache) as d:
        assert d['ids'].dtype.kind=='U' and d['ids'].tolist()==ids
        assert d['source_id'].dtype.kind=='U' and d['source_id'].tolist()==ids
        np.testing.assert_array_equal(d['source_index'],[0,1,2])
        np.testing.assert_allclose(d['end'],lengths)
        np.testing.assert_array_equal(d['event_ids'],np.arange(3))
        assert [(d['source_id'][a],d['source_id'][b]) for a,b in zip(d['i'],d['j'])]==[(ids[0],ids[1]),(ids[1],ids[2]),(ids[0],ids[2])]
        np.testing.assert_array_equal(d['parallel'],[False,False,True])
        data={k:d[k] for k in d.files}
    stats=json.loads(cache.with_suffix('.json').read_text())
    assert stats['signature']['fibers']==fibers_digest(store) and stats['active_fibers']==3
    # Audit reads the same native event IDs without converting them to float64.
    state=np.zeros((3,13));state[:,0]=10.;widths=np.full(4,1000.)
    metrics=audit(data,state,widths,config)
    metrics.update(seconds=0.,equivalent_event_epochs=0.)
    np.savez(tmp_path/'solution.npz',state=state,widths=widths,
             **{k:data[k] for k in ('source_id','start','end','family')})
    (tmp_path/'summary.json').write_text(json.dumps(dict(config=config,input=stats,
        baseline=metrics,final=metrics,peak_allocated_bytes=0)))
    assert report(dict(runs=[str(tmp_path)],report=str(tmp_path/'audit.md')))['runs']==1
    assert json.loads((tmp_path/'audit.json').read_text())[0]['partition_audit']['events']==3
