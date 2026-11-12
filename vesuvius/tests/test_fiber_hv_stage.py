from pathlib import Path
import numpy as np
import pytest
import torch
from vesuvius.neural_tracing.fiber_trace_3d.fiber_hv_stage import losses, probabilities, hard_metrics
from .fiber_graph_fixture import uid, measured_store


def test_preparation_freezes_stage1_and_active_degree(tmp_path):
    import json
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_hv_stage import prepare
    from vesuvius.neural_tracing.tracers.shared.annotations.fiber_store import fibers_digest
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_graph_store import store_digest
    ids=list(map(uid,[10,20,30,40]))
    dbpath=measured_store(tmp_path,[(0,10,20,True,1.),(1,10,30,False,.5)],isolated_ids=[40])
    solution=tmp_path/'solution.npz';np.savez(solution,ids=ids,broken=[False,True,False,False])
    (tmp_path/'summary.json').write_text(json.dumps(dict(input=dict(signature=dict(database_sha256=store_digest(dbpath))))))
    feature=tmp_path/'features.json'
    payload=dict(format='fiber-axial-priors',version=1,source_ids=ids,agreement=[1,1,-1,0],
                 fibers_digest=fibers_digest(dbpath),umbilicus_sha256='b')
    feature.write_text(json.dumps(payload))
    cache=tmp_path/'tensors.npz';config=dict(database=str(dbpath),features=str(feature),stage1=str(solution),tensor_cache=str(cache))
    prepare(config)
    with np.load(cache) as data:
        np.testing.assert_array_equal(data['active'],[0,2,3]);np.testing.assert_array_equal(data['degree'],[1,1,0])
        np.testing.assert_array_equal(data['i'],[0]);np.testing.assert_array_equal(data['j'],[1])
        np.testing.assert_array_equal(data['inherited'],[False,True,False,False])
        assert data['ids'].dtype.kind=='U' and list(data['ids'])==ids
    before=cache.stat().st_mtime_ns;prepare(config);assert cache.stat().st_mtime_ns==before
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_graph_store import FiberGraph
    graph=FiberGraph(dbpath)
    reused=dict(config,tensor_cache=str(tmp_path/'reused.npz'))
    prepare(reused,graph=graph)
    with np.load(cache) as original, np.load(reused['tensor_cache']) as shared:
        for name in original.files:np.testing.assert_array_equal(original[name],shared[name])
    graph.source=dict(graph.source,commit_sequence=-1)
    with pytest.raises(ValueError,match='another snapshot'):
        prepare(dict(config,tensor_cache=str(tmp_path/'wrong.npz')),graph=graph)
    feature.write_text(json.dumps(dict(payload,fibers_digest='wrong snapshot')))
    with pytest.raises(ValueError,match='content mismatch'):prepare(config)
    feature.write_text(json.dumps(dict(payload,source_ids=ids[::-1])))
    with pytest.raises(ValueError,match='IDs differ'):prepare(config)


def test_feature_export_records_store_identity(tmp_path,monkeypatch):
    import json
    from vesuvius.neural_tracing.fiber_trace_3d import constraint_extraction as native
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_hv_stage import run_features
    from vesuvius.neural_tracing.tracers.shared.annotations.fiber_store import fibers_digest, write_fibers
    store=tmp_path/'source.vcobj';write_fibers(store,[(dict(),np.array([[0.,0,0],[1.,0,0]]))],volume='123')
    axis=tmp_path/'axis.json';axis.write_text('{}')
    (tmp_path/'native.json').write_text(json.dumps(dict(fibers=str(store),umbilicus=str(axis),axial_prior_output=str(tmp_path/'features.json'))))
    calls=[]
    monkeypatch.setattr(native,'run_native_constraints',lambda path,settings:calls.append(json.loads(Path(path).read_text())))
    run_features(dict(native_config=str(tmp_path/'native.json'),output_root=str(tmp_path/'out')))
    assert calls[0]['fibers_digest']==fibers_digest(store) and len(calls[0]['umbilicus_sha256'])==64


def test_preparation_does_not_require_prior_sign_stage(tmp_path):
    import json
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_hv_stage import prepare
    from vesuvius.neural_tracing.tracers.shared.annotations.fiber_store import fibers_digest
    database=measured_store(tmp_path,[(0,10,20,False,.5)],isolated_ids=[30])
    ids=list(map(uid,[10,20,30]));features=tmp_path/'features.json'
    features.write_text(json.dumps(dict(format='fiber-axial-priors',version=1,source_ids=ids,
        agreement=[-1,1,0],fibers_digest=fibers_digest(database),umbilicus_sha256='test')))
    cache=tmp_path/'hvb.npz'
    prepare(dict(database=str(database),features=str(features),tensor_cache=str(cache)))
    with np.load(cache) as data:
        np.testing.assert_array_equal(data['active'],[0,1,2])
        assert not data['inherited'].any()


def test_scalar_objective_and_gradients():
    h=torch.tensor([.3,-.7,1.2],dtype=torch.double,requires_grad=True)
    b=torch.tensor([-2.,-1.,.2],dtype=torch.double,requires_grad=True)
    a=torch.tensor([1.,-1.,.1],dtype=torch.double)
    i=torch.tensor([0,1]);j=torch.tensor([1,2]);r=torch.tensor([-1.,1.],dtype=torch.double)
    cost=torch.tensor([.1,.2,.15],dtype=torch.double)
    actual=losses(h,b,a,i,j,r,cost,.001,.01)[0]
    p=probabilities(h,b);expected=0
    for k in range(2):
        x,y=int(i[k]),int(j[k])
        expected+=(p[x,0]*p[y,0]+p[x,1]*p[y,1] if r[k]<0 else p[x,0]*p[y,1]+p[x,1]*p[y,0])/2
    for k in range(3):
        expected+=(.001*(1-b[k].sigmoid())*(1-a[k]*(2*h[k].sigmoid()-1))/2+cost[k]*p[k,2]-.01*(p[k]*p[k].log()).sum())/3
    torch.testing.assert_close(actual,expected)
    assert torch.autograd.gradcheck(lambda x,y:losses(x,y,a,i,j,r,cost,.001,.01)[0],(h,b))
    torch.testing.assert_close(actual,losses(-h,b,-a,i,j,r,cost,.001,.01)[0])


def test_hard_metrics_empty_and_isolated():
    r=hard_metrics(np.array([[0,0,1],[0,0,1]]),np.zeros(2),np.array([0]),np.array([1]),np.array([-1]),np.ones(2))
    assert r['all']['violation_percent'] is None and r['newly_broken']==2
    # Argmax tie is deterministic H; inherited fibers never appear in these active arrays.
    r=hard_metrics(np.array([[.5,.5,0]]),np.zeros(1),np.array([],int),np.array([],int),np.array([]),np.ones(1))
    assert r['h_fibers']==1 and r['all']['retained_percent'] is None


def test_entropy_hardening_and_conflicting_fiber():
    # Four consistent fibers and one connected incompatibly to all four.
    h=torch.nn.Parameter(torch.tensor([-3.,-3.,3.,3.,.1]))
    b=torch.nn.Parameter(torch.full((5,),-4.))
    a=torch.tensor([-1.,-1.,1.,1.,0.])
    i=torch.tensor([0,1,0,1,4,4,4,4]);j=torch.tensor([1,2,2,3,0,1,2,3])
    r=torch.tensor([1.,-1.,-1.,-1.,1.,-1.,1.,-1.])
    opt=torch.optim.Adam([h,b],lr=.05)
    for step in range(400):
        loss=losses(h,b,a,i,j,r,torch.full((5,),.1),.001,.01 if step>200 else 0)[0]
        opt.zero_grad();loss.backward();opt.step()
    p=probabilities(h,b).detach().numpy();assert p[4].argmax()==2
    assert (p[:4].argmax(1)!=2).all()
    assert hard_metrics(p,a.numpy(),i.numpy(),j.numpy(),r.numpy(),np.ones(5))['all']['violations']==0
