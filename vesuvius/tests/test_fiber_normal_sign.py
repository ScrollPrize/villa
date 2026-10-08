import torch
import pytest
import json
import sqlite3
import numpy as np
from lasagna.inference_provenance import sha256_file

from vesuvius.neural_tracing.fiber_trace_3d.fiber_normal_sign import losses, metrics, prepare


def test_pair_agreement_and_break_suppression():
    i=torch.tensor([0]);j=torch.tensor([1]);a=torch.zeros(2)
    u=torch.tensor([20.,20.]);v=torch.tensor([-20.,-20.])
    good=losses(u,v,a,i,j,torch.ones(1),0.,.1)[0]
    bad=losses(u,v,a,i,j,-torch.ones(1),0.,.1)[0]
    assert good < 1e-6 and bad > .99
    v[0]=20.
    disabled=losses(u,v,a,i,j,-torch.ones(1),0.,.1)[0]
    assert disabled==pytest.approx(.05)
    v[1]=20.
    assert losses(u,v,a,i,j,-torch.ones(1),0.,.1)[0]==pytest.approx(.1)


def test_scalar_enumeration_and_gauge_invariance():
    u=torch.tensor([.3,-.7,1.],dtype=torch.double,requires_grad=True)
    v=torch.tensor([-2.,.1,-1.],dtype=torch.double,requires_grad=True)
    a=torch.tensor([.2,-.3,.4],dtype=torch.double)
    i=torch.tensor([0,1,0]);j=torch.tensor([1,2,2])
    dot=torch.tensor([.8,-.4,.2],dtype=torch.double)
    value=losses(u,v,a,i,j,dot,.01,.2)[0]
    s=2*u.sigmoid()-1;b=v.sigmoid()
    expected=sum((1-b[x])*(1-b[y])*(abs(c)-c*s[x]*s[y])/2 for x,y,c in zip(i,j,dot))/dot.abs().sum()
    expected+=.01*((1-a*s)/2).mean()+.2*b.mean()
    torch.testing.assert_close(value,expected)
    gauge=torch.tensor([-1.,1.,-1.],dtype=torch.double)
    transformed=losses(u*gauge,v,a*gauge,i,j,dot*gauge[i]*gauge[j],.01,.2)[0]
    torch.testing.assert_close(value,transformed)
    assert torch.autograd.gradcheck(lambda x,y:losses(x,y,a,i,j,dot,.01,.2)[0],(u,v))


def test_empty_edges_and_hard_ties():
    u=torch.tensor([0.,1.],requires_grad=True);v=torch.tensor([0.,-1.],requires_grad=True)
    idx=torch.empty(0,dtype=torch.long)
    value=losses(u,v,torch.zeros(2),idx,idx,torch.empty(0),.01,.2)[0]
    assert torch.isfinite(value)
    value.backward()
    report=metrics(u,v,torch.zeros(2),idx,idx,torch.empty(0))
    assert report['broken_fibers_percent']==50.
    assert report['retained_events_percent'] is None
    assert report['weighted_disagreement_retained_percent'] is None


def test_synthetic_adam_recovers_opposite_sign():
    # CPU only synthetic unit test; production solver requires CUDA.
    u=torch.nn.Parameter(torch.tensor([1.,.1]));v=torch.nn.Parameter(torch.full((2,),-4.6))
    a=torch.tensor([1.,-.2]);i=torch.tensor([0]);j=torch.tensor([1]);dot=torch.tensor([-1.])
    opt=torch.optim.Adam([u,v],lr=.1)
    for _ in range(250):
        opt.zero_grad();losses(u,v,a,i,j,dot,.001,1.)[0].backward();opt.step()
    assert u[0]>0 and u[1]<0
    assert (v<0).all()
    assert metrics(u,v,a,i,j,dot)['weighted_disagreement_retained_percent']==0.


def test_cached_tensor_identity_and_invalid_edges(tmp_path):
    path=tmp_path/'source.sqlite'
    db=sqlite3.connect(path)
    db.executescript('CREATE TABLE metadata(key,value); CREATE TABLE fibers(id,source_index,normal_sign);'
                     'CREATE TABLE constraints(id,fiber_a,fiber_b,perpendicular_score,parallel_score,normal_a,normal_b);')
    for key,value in dict(schema_version=2,complete=True,normal_orientation=dict(sample_spacing_base=16)).items():
        db.execute('INSERT INTO metadata VALUES(?,?)',(key,json.dumps(value)))
    db.executemany('INSERT INTO fibers VALUES(?,?,?)',[(11,0,1),(12,1,1),(23,2,1)])
    pos=json.dumps(dict(valid=True,xyz=[0,0,1],reason=''))
    neg=json.dumps(dict(valid=True,xyz=[0,0,-1],reason=''))
    invalid=json.dumps(dict(valid=False,xyz=None,reason='missing'))
    db.executemany('INSERT INTO constraints VALUES(?,?,?,?,?,?,?)',[
        (0,11,12,1.,0.,pos,neg),(1,11,23,1.,0.,pos,invalid),(2,12,23,0.,1.,pos,pos)])
    db.commit();db.close()
    prior=tmp_path/'prior.json'
    prior.write_text(json.dumps(dict(format='fiber-outward-priors',source_database=str(path),
        source_database_sha256=sha256_file(path),source_ids=[11,12,23],agreement=[1.,-1.,0.],
        statistics=dict(sample_spacing_base=16))))
    config=dict(database=str(path),priors=str(prior),tensor_cache=str(tmp_path/'tensor.npz'))
    cache=prepare(config)
    with np.load(cache) as data:
        assert data['dot'].tolist()==[-1.]
        assert data['i'].tolist()==[0] and data['j'].tolist()==[1]
    stats=json.loads(cache.with_suffix('.json').read_text())
    assert stats['invalid_events_skipped']==1 and stats['isolated_fibers']==1
    assert prepare(config)==cache
    db=sqlite3.connect(path);db.execute('UPDATE constraints SET normal_b=? WHERE id=0',(pos,));db.commit();db.close()
    with pytest.raises(ValueError,match='different database content'):
        prepare(config)
