from pathlib import Path
import torch
import pytest
import json
import numpy as np

from vesuvius.neural_tracing.fiber_trace_3d.fiber_normal_sign import losses, metrics, prepare
from .fiber_graph_fixture import uid, measured_store


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
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_graph_store import FiberGraph
    path=measured_store(tmp_path,[(0,11,12,False,.5),(1,11,23,False,.5),(2,12,23,True,1.)],
        field_overrides={0:dict(normal_b_x=-1.),1:dict(normal_b_valid=0)})
    ids=list(map(uid,[11,12,23]))
    graph=FiberGraph(path)
    prior=tmp_path/'prior.json'
    payload=dict(format='fiber-outward-priors',source_store=graph.source,source_ids=ids,agreement=[1.,-1.,0.],
        statistics=dict(sample_spacing_base=16))
    prior.write_text(json.dumps(payload))
    config=dict(database=str(path),priors=str(prior),tensor_cache=str(tmp_path/'tensor.npz'))
    cache=prepare(config)
    with np.load(cache) as data:
        assert data['dot'].tolist()==[-1.]
        assert data['i'].tolist()==[0] and data['j'].tolist()==[1]
        assert data['ids'].tolist()==ids and data['event_ids'].tolist()==[0]
    stats=json.loads(cache.with_suffix('.json').read_text())
    assert stats['invalid_events_skipped']==1 and stats['isolated_fibers']==1
    assert prepare(config)==cache
    reused=dict(config,tensor_cache=str(tmp_path/'reused.npz'))
    prepare(reused,graph=graph)
    with np.load(cache) as original, np.load(reused['tensor_cache']) as shared:
        for name in original.files:np.testing.assert_array_equal(original[name],shared[name])
    graph.source=dict(graph.source,commit_sequence=-1)
    with pytest.raises(ValueError,match='another snapshot'):
        prepare(dict(config,tensor_cache=str(tmp_path/'wrong.npz')),graph=graph)
    graph.source=payload['source_store']
    prior.write_text(json.dumps(dict(payload,source_store=dict(graph.source,commit_sequence=-1))))
    with pytest.raises(ValueError,match='another store snapshot'):
        prepare(config)


def test_crossing_filter_passes(tmp_path):
    from vc_object_store import CROSSING_FINDINGS
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_graph_store import FiberGraph
    path=measured_store(tmp_path,[(0,11,12,False,.5),(1,11,23,False,.5),(2,12,23,False,.5)])
    ids=list(map(uid,[11,12,23]));graph=FiberGraph(path)
    prior=tmp_path/'prior.json'
    prior.write_text(json.dumps(dict(format='fiber-outward-priors',source_store=graph.source,source_ids=ids,agreement=[1.,-1.,0.])))
    config=dict(database=str(path),priors=str(prior),crossing_filter=True)
    # This store's extraction recorded no findings: filtering is an error, never a silent unfiltered solve.
    assert not graph.has_crossing_findings and not graph.crossing_findings.any()
    with pytest.raises(ValueError,match='needs an extraction with crossing_findings'):
        prepare(config,graph=graph,return_arrays=True)
    # Event 0: nothing left. Event 1: mixed, with an alternative of |dot| 0.5. Event 2: no finding.
    graph.has_crossing_findings=True
    graph.crossing_findings=np.asarray([CROSSING_FINDINGS['nothing_left'],CROSSING_FINDINGS['mixed'],0],np.uint8)
    graph.crossing_alternative=np.asarray([0.,.5,0.])
    plain,_=prepare(dict(config,crossing_filter=False),graph=graph,return_arrays=True)
    assert plain['event_ids'].tolist()==[0,1,2] and plain['dot'].tolist()==[1.,1.,1.]
    # First pass: sign-free, drops nothing-left and mixed.
    data,stats=prepare(config,graph=graph,return_arrays=True)
    assert data['event_ids'].tolist()==[2] and stats['crossing_filter_dropped']==2 and stats['invalid_events_skipped']==0
    assert prepare(dict(config,crossing_filter_first_pass=['nothing_left']),graph=graph,return_arrays=True)[0]['event_ids'].tolist()==[1,2]
    # Second pass: only nothing-left is dropped. Under signs that make event 1 disagree it takes its
    # agreeing alternative; under signs it agrees with it stays as it is.
    data,stats=prepare(config,graph=graph,return_arrays=True,signs=[1,1,-1])
    assert data['event_ids'].tolist()==[1,2] and data['dot'].tolist()==[-.5,1.] and stats['crossing_filter_replaced']==1
    data,stats=prepare(config,graph=graph,return_arrays=True,signs=[1,1,1])
    assert data['dot'].tolist()==[1.,1.] and stats['crossing_filter_replaced']==0
    with pytest.raises(ValueError,match='needs crossing_filter'):
        prepare(dict(config,crossing_filter=False),graph=graph,return_arrays=True,signs=[1,1,1])


def test_winding_distance_factor_and_strict_metric():
    torch=pytest.importorskip('torch')
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_normal_sign import winding_distance_factor, metrics
    distance=torch.tensor([0.5,1.,2.,3.,float('nan')])
    factor=winding_distance_factor(distance,'inverse_square')
    assert torch.allclose(factor,torch.tensor([1.,1.,.25,1/9,1.]))
    assert torch.equal(winding_distance_factor(distance,None),torch.ones(5))
    with pytest.raises(ValueError):winding_distance_factor(distance,'linear')
    # Three fibres, four crossings; fibre 2 is signed against the others.
    u=torch.tensor([3.,3.,-3.]);v=torch.full((3,),-3.);a=torch.zeros(3)
    i=torch.tensor([0,0,1,1]);j=torch.tensor([1,2,2,2])
    dot=torch.tensor([.95,.95,.95,.5])          # the last crossing's axes differ by 60 degrees
    distance=torch.tensor([0.5,0.5,2.,0.5])     # the third crossing is two windings away
    m=metrics(u,v,a,i,j,dot,distance=distance,factor=winding_distance_factor(distance,'inverse_square'))
    # strict set: nearly parallel (<30 deg) and within one winding -> crossings 0 and 1; crossing 1 disagrees.
    assert m['strict_events']==2 and m['strict_disagreement_percent']==pytest.approx(50.)
    # weighted: wrong crossings 1,2,3 with weights .95, .95/4, .5 over the total .95+.95+.2375+.5
    assert m['weighted_disagreement_all_percent']==pytest.approx(100*(.95+.2375+.5)/(.95+.95+.2375+.5))
    plain=metrics(u,v,a,i,j,dot)
    assert 'strict_events' not in plain and plain['weighted_disagreement_all_percent']==pytest.approx(100*(.95+.95+.5)/(.95*3+.5))
