import numpy as np
import torch
from vesuvius.neural_tracing.fiber_trace_3d.fiber_hv_stage import losses, probabilities, hard_metrics


def test_preparation_freezes_stage1_and_active_degree(tmp_path):
    import json,sqlite3
    from lasagna.inference_provenance import sha256_file
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_hv_stage import prepare
    afvpath=tmp_path/'source.afv';afv=sqlite3.connect(afvpath)
    afv.execute('CREATE TABLE fibers(id,length)');afv.executemany('INSERT INTO fibers VALUES(?,?)',[(i,100) for i in (10,20,30,40)])
    afv.commit();afv.close()
    dbpath=tmp_path/'input.sqlite';db=sqlite3.connect(dbpath)
    db.executescript('CREATE TABLE metadata(key,value);CREATE TABLE fibers(id,source_index,length_base);'
        'CREATE TABLE constraints(id,fiber_a,fiber_b,parallel_score,perpendicular_score);')
    db.executemany('INSERT INTO metadata VALUES(?,?)',[(k,json.dumps(v)) for k,v in
        dict(complete=True,provenance=dict(afv_uuid='test',config=dict(afv=str(afvpath)))).items()])
    db.executemany('INSERT INTO fibers VALUES(?,?,?)',[(10,0,100),(20,1,100),(30,2,100),(40,3,100)])
    db.executemany('INSERT INTO constraints VALUES(?,?,?,?,?)',[(0,10,20,1,0),(1,10,30,0,1)])
    db.commit();db.close()
    solution=tmp_path/'solution.npz';np.savez(solution,ids=[10,20,30,40],broken=[False,True,False,False])
    (tmp_path/'summary.json').write_text(json.dumps(dict(input=dict(signature=dict(database_sha256=sha256_file(dbpath))))))
    feature=tmp_path/'features.json';feature.write_text(json.dumps(dict(format='fiber-axial-priors',version=1,
        afv_uuid='test',source_ids=[10,20,30,40],agreement=[1,1,-1,0],afv_sha256=sha256_file(afvpath),umbilicus_sha256='b')))
    cache=tmp_path/'tensors.npz';config=dict(database=str(dbpath),features=str(feature),stage1=str(solution),tensor_cache=str(cache))
    prepare(config)
    with np.load(cache) as data:
        np.testing.assert_array_equal(data['active'],[0,2,3]);np.testing.assert_array_equal(data['degree'],[1,1,0])
        np.testing.assert_array_equal(data['i'],[0]);np.testing.assert_array_equal(data['j'],[1])
        np.testing.assert_array_equal(data['inherited'],[False,True,False,False])
    before=cache.stat().st_mtime_ns;prepare(config);assert cache.stat().st_mtime_ns==before


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
