import numpy as np
import torch
from vesuvius.neural_tracing.fiber_trace_3d.fiber_piece_split import split_graph_geometry,rejection_costs
from vesuvius.neural_tracing.fiber_trace_3d.fiber_v_endpoint_offsets import topology


def graph():
    arcs=np.arange(500,8000,500,dtype=float);n=len(arcs)
    d=dict(source_id=np.array([10,20]),source_index=np.arange(2),family=np.array([0,1]),
        start=np.zeros(2),end=np.full(2,8000.),lengths=np.full(2,8000.),
        phase=np.zeros(2),z=np.zeros(2),radius=np.ones(2)*100,
        initial_k=np.zeros(2),node_ds=np.array([[0,4000,8000],[0,-4000,-8000]]),
        i=np.zeros(n,int),j=np.ones(n,int),ds_a=arcs,ds_b=-arcs,
        parallel=np.zeros(n,bool),links=np.empty((0,4)))
    curves={10:np.array([[100,0,0],[100,0,8000.]]),20:np.array([[100,0,0],[100,0,8000.]])}
    return d,curves,np.array([[0,0,0],[0,0,8000.]])


def test_split_preserves_arcs_lengths_support_and_protected_gt():
    d,c,a=graph();s=split_graph_geometry(d,2000,4,c,a,protected_sources=[20])
    assert len(s['family'])==4 and len(s['links'])==2
    assert np.sum(s['lengths'][s['source_id']==10])==8000
    np.testing.assert_allclose(s['start'][s['i']]+s['ds_a'],d['ds_a'])
    np.testing.assert_allclose(s['start'][s['j']]-s['ds_b'],-d['ds_b'])
    assert np.all(np.bincount(s['i'],minlength=4)[:3]>=4)
    for left,right,ds,zero in s['links']:
        assert s['start'][int(left)]+ds==s['start'][int(right)]+zero
    t=topology(s,0,piece_local=True)
    assert set(t['v_offset_source'])==set(range(4))
    assert all(t['v_offset_source'][l]==t['v_offset_source'][r]
               for l,r in zip(t['v_offset_left'],t['v_offset_right']))


def test_negative_direction_split_and_rejection_incident_counts():
    d,c,a=graph();s=split_graph_geometry(d,2000,4,c,a)
    np.testing.assert_allclose(s['start'][s['j']]-s['ds_b'],-d['ds_b'])
    costs=rejection_costs(s,dict(rejection_cost=2,rejection_perpendicular_cost=3,
        rejection_parallel_cost=7,rejection_continuation_cost=5))
    degree=np.bincount(np.r_[s['i'],s['j']],minlength=len(costs))
    links=np.bincount(s['links'][:,:2].astype(int).ravel(),minlength=len(costs))
    np.testing.assert_allclose(costs,2+3*degree+5*links)
    np.testing.assert_allclose(rejection_costs(s,dict(rejection_cost=2)),2)


def test_split_curve_continuation_queries_have_live_anchor_gradients():
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_shared_curve import SharedCurve
    d,c,a=graph();s=split_graph_geometry(d,2000,4,c,a)
    top=topology(s,0,piece_local=True)
    cfg=dict(straight_fixed_length_geometry=True,shared_curve_integer_geometry=True,
             winding_sigma=.5)
    curve=SharedCurve(s,top['v_offset_source'],top['v_offset_arc'],cfg,'cpu')
    curve.prepare_fields(s,{})
    state=torch.zeros(len(s['family']),4,dtype=torch.float64);state[:,0]=5
    widths=torch.ones(20,dtype=torch.float64)*10000
    curve.initialize(state,widths)
    qa,_,va=curve.evaluate(state,widths,'link_a');qb,_,vb=curve.evaluate(state,widths,'link_b')
    ((qa-qb).square().sum()+(va-vb).square().sum()).backward()
    assert curve.u.grad is not None and curve.u.grad.abs().sum()>0
    assert curve.v.grad is not None and curve.v.grad.abs().sum()>0


def test_initial_pieces_are_continuous_across_phase_seams():
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_piece_split import initial_state
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_soft_winding import points
    d,c,a=graph();d['phase'][:]=.9;s=split_graph_geometry(d,2000,4,c,a)
    widths=torch.ones(20,dtype=torch.float64)*1000
    state=torch.as_tensor(initial_state(s,np.full(len(s['family']),5.),widths.numpy()))
    links=s['links'];family=torch.as_tensor(s['family'])
    qa,_,va=points(state[links[:,0].astype(int)],family[links[:,0].astype(int)],torch.as_tensor(links[:,2]),widths)
    qb,_,vb=points(state[links[:,1].astype(int)],family[links[:,1].astype(int)],torch.as_tensor(links[:,3]),widths)
    torch.testing.assert_close(qa,qb);torch.testing.assert_close(va,vb)


def test_continuation_marginalizes_errors_not_positions():
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_implicit_winding import pair_cost,hypotheses,hypothesis_points
    cfg=dict(winding_sigma=.5,distance_sigma=.125,v_pair_scale=1000.,
             soft_pair_weight=1.,probability_pair_weight=.3)
    a=torch.tensor([[5.,.2,0.,4.]],dtype=torch.float64,requires_grad=True)
    b=a.detach().clone().requires_grad_()
    family=torch.tensor([0]);zero=torch.zeros(1,dtype=torch.float64)
    widths=torch.full((20,),10000.,dtype=torch.float64)
    args=(a,b,family,family,zero,zero,widths,zero,zero,torch.tensor([False]),cfg)
    winding,_,vertical=pair_cost(*args,exact_endpoint_target=zero)
    h,lp=hypotheses(a,.5);q,_,_=hypothesis_points(h,family,zero,widths)
    mass=lp.exp()[:,:,None]*lp.exp()[:,None,:]
    expected=-.3*torch.logsumexp((mass.log()-((q[:,None,:]-q[:,:,None])/.125).square()).flatten(1),1)
    torch.testing.assert_close(winding,expected)
    assert winding.item()>0.1  # Identical means must not hide incompatible mass.
    assert vertical.item()==0
    shifted=b.detach().clone();shifted[:,0]+=1;shifted[:,2]+=250
    w,_,v=pair_cost(a,shifted,family,family,zero,zero,widths,zero,zero,
        torch.tensor([False]),cfg,exact_endpoint_target=zero)
    assert w.item()>winding.item()
    torch.testing.assert_close(v,torch.tensor([1.3*.25**2],dtype=torch.float64))
    # Calibration/sign tuning must never move the exact zero continuation target.
    other=pair_cost(*args[:-1],dict(cfg,lasagna_winding_scale=8,perpendicular_offset=2,
        unified_order_loss=True,gt_pair_mode='sign_only'),exact_endpoint_target=zero)
    torch.testing.assert_close(other[0],winding)


def test_continuation_uses_live_per_hypothesis_split_endpoint_geometry():
    from types import SimpleNamespace
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_shared_curve import SharedCurve
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_intermediate_uv import SharedPairGeometry
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_implicit_winding import pair_cost
    d,c,axis=graph();s=split_graph_geometry(d,2000,4,c,axis)
    top=topology(s,0,piece_local=True)
    cfg=dict(straight_fixed_length_geometry=True,shared_curve_integer_geometry=True,
             winding_sigma=.5,distance_sigma=.125,v_pair_scale=1000.,soft_pair_weight=1.,probability_pair_weight=.3)
    curve=SharedCurve(s,top['v_offset_source'],top['v_offset_arc'],cfg,'cpu')
    curve.prepare_fields(s,{})
    state=torch.zeros(len(s['family']),4,dtype=torch.float64,requires_grad=True)
    with torch.no_grad():state[:,0]=5
    widths=torch.linspace(5000,20000,20,dtype=torch.float64,requires_grad=True)
    curve.initialize(state,widths)
    links=torch.as_tensor(s['links']);a,b=links[:,:2].long().T
    geometry=SharedPairGeometry(SimpleNamespace(curve=curve),state,widths,
        torch.arange(len(links)),('link_a','link_b'))
    family=torch.as_tensor(s['family']);zero=state.new_zeros(len(links))
    costs=pair_cost(state[a],state[b],family[a],family[b],links[:,2],links[:,3],widths,
        zero,zero,torch.zeros_like(a,dtype=torch.bool),cfg,geometry=geometry,exact_endpoint_target=zero)
    sum(x.sum() for x in costs).backward()
    for grad in (state.grad,curve.u.grad,curve.v.grad,widths.grad):
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum()>0
