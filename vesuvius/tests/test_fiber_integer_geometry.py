"""N controls integer-label probabilities, never a selected curve's shape."""
import numpy as np
import torch
from .test_fiber_affine_geometry import setup
from vesuvius.neural_tracing.fiber_trace_3d.fiber_shared_curve import saved_points
from vesuvius.neural_tracing.fiber_trace_3d import fiber_implicit_winding as density
from vesuvius.neural_tracing.fiber_trace_3d.fiber_geometry_cache import width_boundaries


def fixture():
    d,cfg,td,s,w,e,_=setup()
    w=torch.linspace(700,1800,len(w),requires_grad=True)
    return d,cfg,td,s,w,e


def test_selected_geometry_independent_of_fractional_N_at_nodes_and_between():
    d,cfg,td,s,w,e=fixture();curve=e.curve
    ids=np.array([0,0,0,0,1]);ds=np.array([-40.,10.,55.,310.,40.])
    curve.add_plan('test',ids,ds);labels=torch.tensor([[2.,3.]]*4+[[3.,4.]])
    a=curve.hypothesis_queries(s,w,curve.plans['test'],labels)
    changed=s.clone();changed[:,0]=changed[:,0]+.1
    b=curve.hypothesis_queries(changed,w,curve.plans['test'],labels)
    for x,y in zip(a,b):torch.testing.assert_close(x,y,rtol=0,atol=0)
    grad=torch.autograd.grad(sum(x.sum() for x in a),s,allow_unused=True)[0]
    assert grad is None or torch.equal(grad[:,0],torch.zeros_like(grad[:,0]))
    h,_=e.local_hypotheses(s)
    q,v=curve.node_hypotheses(s,w,h[e.owner,:,0])
    torch.testing.assert_close(q,h[e.owner,:,0]+curve.coordinates()[:,None,0].to(s))


def test_integer_relabel_equivalence_and_saved_readout():
    d,cfg,td,s,w,e=fixture();curve=e.curve
    ids=d['i'];ds=d['ds_a'];h,lp=density.hypotheses(s,.5)
    before=curve.hypothesis_queries(s,w,curve.plans['a'],h[ids,:,0])
    xy=curve.coordinates().detach().clone();xy[:,0]-=1
    curve.set_coordinates(xy)
    shifted=s.clone();shifted[:,0]+=1;shifted[:,1]-=1
    hh,ll=density.hypotheses(shifted,.5)
    after=curve.hypothesis_queries(shifted,w,curve.plans['a'],hh[ids,:,0])
    torch.testing.assert_close(before[0],after[0]);torch.testing.assert_close(before[1],after[1]+1)
    torch.testing.assert_close(before[2],after[2]);torch.testing.assert_close(lp,ll)
    saved=dict(curve.arrays(),state=shifted.detach().numpy())
    mapped=density.map_state(shifted)
    expected=curve.hypothesis_queries(mapped.double(),w.double(),curve.plans['a'],mapped[ids,0,None].double())
    actual=saved_points(d,saved,shifted.detach().numpy(),w.detach().numpy(),ids,ds,shared_only=True)
    for x,y in zip(actual,expected):np.testing.assert_allclose(x,y[:,0].detach().numpy(),rtol=1e-6,atol=1e-6)


def test_length_is_expected_integer_energy_not_energy_at_mean_N():
    d,cfg,td,s,w,e=fixture();curve=e.curve
    curve.sigma_values=e.cavity.values
    h,lp=density.hypotheses(s,e.cavity.values,cfg['sigma_max'])
    xy=curve.coordinates();q=h[e.owner,:,0].double()+xy[:,None,0]
    x=density.soft.uv.q_to_u(q,w.double(),width_boundaries(w.double()))
    pos=torch.stack((x,xy[:,None,1].expand_as(x)),-1)
    r=torch.where(~curve.first)[0];l=r-1;gap=curve.arc[r]-curve.arc[l]
    error=(torch.linalg.vector_norm(pos[r]-pos[l],dim=-1)/gap[:,None]-1).square()
    weights=lp.exp()[e.owner[r]]*gap[:,None]
    expected=cfg.get('shared_curve_length_weight',10.)*(error*weights).sum()/weights.sum()
    actual=curve.terms(s,w)['shared_curve_length']
    torch.testing.assert_close(actual,expected)
    gradients=torch.autograd.grad(actual,(s,w),retain_graph=True)
    assert all(torch.isfinite(g).all() for g in gradients)
    # Constant widths remove all geometric label dependence; N cannot then
    # obtain a length-loss gradient through any hidden continuous mapping.
    equal=torch.full_like(w,1000.)
    cost=curve.terms(s,equal)['shared_curve_length']
    gn=torch.autograd.grad(cost,s)[0][:,0]
    torch.testing.assert_close(gn,torch.zeros_like(gn),atol=2e-4,rtol=0)


def test_no_perpendicular_nodes_are_rejected_not_given_fallback_geometry():
    import pytest
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_shared_curve import SharedCurve
    d,cfg,td,s,w,e=fixture()
    with pytest.raises(ValueError,match='no perpendicular nodes'):
        SharedCurve(d,np.array([],dtype=int),np.array([]),cfg,'cpu')
