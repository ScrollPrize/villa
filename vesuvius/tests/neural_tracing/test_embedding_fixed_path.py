import numpy as np
import pytest
import torch

from vesuvius.neural_tracing.fiber_trace_3d.embedding_fixed_path import (
    ema_distance_sequence, mapped_distance, fiber_split, make_windows,
)
from vesuvius.neural_tracing.fiber_trace_3d.embedding_integration import integrate, window_distances, integration_flags


@pytest.mark.parametrize('family,params',[
    ('ema',[.25,8]),('gate',[.25,8,2,2,2]),('angular',[.25,8,2]),('dual',[.25,8,64,0,2])])
def test_integrators_gradients_cadence_and_parent_score(family,params):
    e=torch.tensor([[[1.,0.],[0.,1.],[0.,1.],[1.,0.]]])
    d=torch.tensor([[0.,8.,8.,8.]])
    p=torch.ones_like(d)
    parameters=torch.tensor([params],requires_grad=True)
    cost,state=integrate(e,d,p,parameters,family,torch.tensor([16.],dtype=torch.float64))
    assert float(cost[0,0,0].detach())==1
    assert float(cost[0,0,1].detach())==1  # no update until second observation
    cost.sum().backward()
    assert torch.isfinite(parameters.grad).all()
    assert float(state[2][0,0])==8
    if family=='angular':
        angle=torch.acos(state[0][0,0,0].clamp(-1,1))
        assert float(angle.detach())<=np.deg2rad(4)+1e-5


def test_finite_window_candidate_expiry_and_default_ema():
    e=torch.tensor([[[1.,0.],[0.,1.],[0.,1.],[1.,0.]]]);d=torch.tensor([[0.,8.,8.,8.]])
    costs=window_distances(e,d,torch.ones_like(d,dtype=torch.bool),torch.tensor([8.]),torch.tensor([8.]),3)
    torch.testing.assert_close(costs,torch.tensor([[[1.,0.,1.]]]))
    reference,_=ema_distance_sequence(e,d,torch.tensor(8.))
    tested,_=integrate(e,d,torch.ones_like(d),torch.tensor([[.25,8.]]),'ema',torch.tensor([0.]))
    torch.testing.assert_close(tested[0],reference)


def test_native_integrator_parity_synthetic(tmp_path):
    import os,json,subprocess
    binary=os.environ.get('VC_REPLAY_TEST_BINARY')
    if not binary:pytest.skip('native binary not configured')
    requests=[];expected=[]
    e=torch.tensor([[[1.,0.],[.6,.8],[0.,1.],[0.,1.],[1.,0.]]])
    d=torch.tensor([[0.,7.99,8.01,8.,8.]]);presence=torch.tensor([[1.,.8,.6,.2,1.]])
    cases=[('ema',[.25,8.]),('gate',[.25,8.,3,2,2]),('angular',[.25,8.,2.]),('dual',[.25,8.,64.,0.,2.]),('window',[.25])]
    for family,p in cases:
        for spacing in [0.,8.,16.]:
            if family=='window':cost=window_distances(e,d,torch.ones_like(d,dtype=torch.bool),torch.tensor([spacing]),torch.tensor([16.]),4)
            else:cost,_=integrate(e,d,presence,torch.tensor([p]),family,torch.tensor([spacing]))
            expected.append((cost[0,0]*p[0]).tolist())
            samples=[dict(embedding=x,distance=y,presence=z) for x,y,z in zip(e[0].tolist(),d[0].tolist(),presence[0].tolist())]
            requests.append(dict(trace_args=integration_flags(family,spacing,p,16),samples=samples))
    path=tmp_path/'queries.json';path.write_text(json.dumps(requests))
    actual=json.loads(subprocess.check_output([binary,'integrate-embeddings',str(path)],text=True))
    for a,b in zip(actual,expected):np.testing.assert_allclose(a[1:],b,atol=2e-6)


@pytest.mark.parametrize('family,params',[
    ('ema',[.25,32.]),('gate',[.25,32.,2.,2.,2.]),
    ('angular',[.25,32.,90.]),('dual',[.25,32.,128.,0.,2.])])
def test_split_padded_warmup_preserves_admission_clocks(family,params):
    e=torch.tensor([[[1.,0.],[.6,.8],[0.,1.],[.8,.6],[1.,0.]],
                    [[1.,0.],[1.,0.],[.6,.8],[.8,.6],[1.,0.]]])
    d=torch.tensor([[0.,7.99,8.01,7.,9.],[0.,0.,7.,7.,9.]])
    mask=torch.tensor([[True]*5,[True,False,True,True,True]])
    spacing=torch.tensor([16.],dtype=torch.float64);p=torch.tensor([params])
    whole,_=integrate(e,d,torch.ones_like(d),p,family,spacing,mask)
    _,state=integrate(e[:,:3],d[:,:3],torch.ones_like(d[:,:3]),p,family,spacing,mask[:,:3],collect=False)
    assert state[2].tolist()==[[0.,7.]]
    tail,_=integrate(e[:,2:],d[:,2:],torch.ones_like(d[:,2:]),p,family,spacing,initial=state)
    torch.testing.assert_close(tail,whole[:,:,2:])


def test_native_angular_antipodal_parity(tmp_path):
    import os,json,subprocess
    binary=os.environ.get('VC_REPLAY_TEST_BINARY')
    if not binary:pytest.skip('native binary not configured')
    for observation in ([-1.,0.],[-.9999995,.001]):
        e=torch.tensor([[[1.,0.],observation,[1.,0.]]]);d=torch.tensor([[0.,64.,8.]])
        cost,_=integrate(e,d,torch.ones_like(d),torch.tensor([[.25,32.,90.]]),'angular',torch.tensor([64.]))
        samples=[dict(embedding=x,distance=y,presence=1) for x,y in zip(e[0].tolist(),d[0].tolist())]
        path=tmp_path/'antipodal.json'
        path.write_text(json.dumps([dict(trace_args=integration_flags('angular',64,[.25,32.,90.]),samples=samples)]))
        actual=json.loads(subprocess.check_output([binary,'integrate-embeddings',str(path)],text=True))[0]
        np.testing.assert_allclose(actual[1:],(cost[0,0]*.25).tolist(),atol=2e-6)


def test_ema_parent_order_gradient_and_cancellation():
    e = torch.tensor([[[1.,0.],[0.,1.],[1.,0.]]],dtype=torch.float64)
    d = torch.tensor([[0.,8.,8.]],dtype=torch.float64)
    h = torch.tensor(8.,dtype=torch.float64,requires_grad=True)
    cost,state=ema_distance_sequence(e,d,h)
    assert float(cost[0,0].detach()) == 1
    assert float(cost[0,1].detach()) == pytest.approx(1-2**-.5)
    cost.sum().backward()
    assert torch.isfinite(h.grad) and abs(float(h.grad))>0
    eps=1e-4
    lo=ema_distance_sequence(e,d,h.detach()-eps)[0].sum()
    hi=ema_distance_sequence(e,d,h.detach()+eps)[0].sum()
    assert float(h.grad)==pytest.approx(float((hi-lo)/(2*eps)),rel=1e-5)
    antipodal=torch.tensor([[[1.,0.],[-1.,0.]]])
    _,s=ema_distance_sequence(antipodal,torch.tensor([[0.,8.]]),torch.tensor(8.))
    assert torch.equal(s,antipodal[:,0])
    half=torch.tensor(8.,requires_grad=True)
    _,s=ema_distance_sequence(antipodal,torch.tensor([[0.,8.]]),half)
    s.sum().backward()
    assert torch.isfinite(half.grad)


def test_mapping_zero_monotone_and_finite():
    x=torch.linspace(0,2,100)
    y=mapped_distance(x,torch.tensor(.3),torch.tensor(.01))
    assert float(y[0])==0
    assert torch.isfinite(y).all() and (torch.diff(y)>=0).all()


def test_split_groups_duplicates_and_directions(tmp_path):
    cases=[]
    for i in range(10):
        p=tmp_path/f'{i}.json';p.write_text(str(i))
        cases.extend([{'provenance':dict(fiber_path=str(p),coordinate_space='A/scan')}]*2)
    duplicate=tmp_path/'duplicate.json';duplicate.write_text('0')
    cases.append({'provenance':dict(fiber_path=str(duplicate),coordinate_space='A/scan')})
    a=fiber_split(cases);b=fiber_split(list(reversed(cases)))
    assert a==b and a[str(duplicate)]==a[str(tmp_path/'0.json')]
    assert {v['split'] for v in a.values()}=={'fit','selection','test'}


def test_windows_share_history_and_do_not_cross_segments():
    points=np.stack([np.arange(300)*8,np.zeros(300),np.zeros(300)],axis=1)
    case=dict(reference_points_base_xyz=points.tolist(),provenance=dict(trace_to_base=2,effective_trace=dict(step_voxels=4)),
              replay=dict(failures=[],segments=[dict(trace_points_base_xyz=points.tolist(),matches=[
                  dict(trace_point_index=i,matched_reference_arc_base=i*8,euclidean_error_base_voxels=0) for i in range(300)])]))
    windows,rejected=make_windows(case,{})
    assert len(windows)==1 and not rejected
    w=windows[0];k=w['score_start']
    assert w['kind']=='control' and w['good'][:k]==w['bad'][:k]
    assert np.asarray(w['good']).shape==np.asarray(w['bad']).shape
