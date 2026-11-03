import numpy as np
import torch
import pytest
from vesuvius.neural_tracing.fiber_trace_3d import fiber_shared_curve as c
from vesuvius.neural_tracing.fiber_trace_3d import fiber_bucket_uv as b
from .test_fiber_bucket_uv import fixture


def setup(integer_geometry=True):
    d,cfg,td,s,w=fixture();d['links']=np.empty((0,4));cfg.update(shared_curve_geometry=True,shared_curve_hypothesis_mapping=True,lr=.073)
    cfg['shared_curve_integer_geometry']=integer_geometry
    ext=b.BucketUV(d,cfg,'cpu');ext.prepare_fields(d,{})
    ext.update(s,w,td,0)
    return d,cfg,td,s,w,ext


def test_interpolation_extrapolation():
    l,r,a=c.interpolation_plan(np.array([0,0]),np.array([10.,20.]),np.zeros(5,int),np.array([0.,10.,15.,20.,30.]))
    np.testing.assert_array_equal(l,0);np.testing.assert_array_equal(r,1)
    np.testing.assert_allclose(a,[-1,0,.5,1,2])


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('mapping',[False,True])
def test_fixed_queries_all_node_cases(dtype,mapping):
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_geometry_cache import geometry_evaluation
    d=dict(family=np.array([0,1,0]),start=np.zeros(3),end=np.full(3,40.),
           node_ds=np.tile([0.,40.],(3,1)))
    curve=c.SharedCurve(d,np.array([0,0,1,2]),np.array([10.,20.,15.,15.]),
                        dict(fiber_affine_geometry=True,shared_curve_hypothesis_mapping=mapping),'cpu')
    state=torch.tensor([[1.,.2,100.,2.],[2.,.4,200.,2.],[3.,.6,300.,2.]],dtype=dtype,requires_grad=True)
    widths=torch.tensor([100.,120.,160.,220.,280.],dtype=dtype,requires_grad=True)
    curve.initialize(state,widths)
    curve.add_plan('test',np.repeat(np.arange(3),5),np.tile([0.,10.,15.,20.,40.],3))
    with torch.no_grad():curve.u.add_(.07);curve.v.add_(2.)
    parameters=(state,widths,*curve.parameters())
    def evaluate(fixed):
        curve.config['fixed_shared_queries']=fixed
        with geometry_evaluation():
            value=torch.stack(curve.evaluate(state,widths,'test',torch.tensor([14,0,6,2,2,8,9])),1)
            return value,torch.autograd.grad(value.square().sum(),parameters,allow_unused=True)
    a,ga=evaluate(False);b,gb=evaluate(True)
    torch.testing.assert_close(a,b)
    for x,y in zip(ga,gb):
        if x is None:assert y is None
        else:torch.testing.assert_close(x,y,rtol=2e-5,atol=2e-5)


def test_shared_curve_visualization_numpy_labels(tmp_path):
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_bucket_uv_visualization import geometry
    d,cfg,td,s,w,e=setup()
    cfg['bucket_uv_active_window']=True
    e=b.BucketUV(d,cfg,'cpu');e.prepare_fields(d,{})
    e.update(s,w,td,0)
    with torch.no_grad():
        s[:,0].copy_(torch.tensor([2.,3.]))
        e.cavity.values.fill_(.15)
    saved=dict(e.arrays(),state=s.detach().numpy(),widths=w.detach().numpy(),
               winding_sigma=e.cavity.values.numpy())
    np.savez(tmp_path/'solution.npz',**saved)
    with np.load(tmp_path/'solution.npz') as checkpoint:
        before,after,maximum,predicted,actual,ext,labels=geometry(d,checkpoint,cfg)
    assert len(before[0])==2
    assert np.isfinite(np.concatenate(before[1])).all()
    assert np.isfinite(np.concatenate(after[1])).all()
    assert isinstance(labels,np.ndarray)


def test_initial_coordinates_and_pair_gradients():
    d,cfg,td,s,w,e=setup()
    for key,ids,ds in [('a',d['i'],d['ds_a']),('b',d['j'],d['ds_b'])]:
        actual=e.curve.evaluate(s,w,key)
        expected=b.density.expected_points(s[ids],td['family'][ids],torch.as_tensor(ds).float(),w,.5)
        for x,y in zip(actual,expected):torch.testing.assert_close(x,y,atol=1e-5,rtol=1e-5)
    sa,sb,da,db=e.pair_states(s,w,td,torch.arange(3))
    loss=sum(x.sum() for x in b.density.pair_cost(sa,sb,td['family'][td['i']],td['family'][td['j']],
        da,db,w,td['sign'],td['distance'],td['parallel'],cfg,sigma_a=e.cavity.values[td['i']],sigma_b=e.cavity.values[td['j']]))
    loss.backward()
    assert e.curve.u.grad.abs().max()>0 and e.curve.v.grad.abs().max()>0
    assert s.grad[:,1:3].abs().max()==0 # no obsolete straight-coordinate dependency


def test_forward_and_length():
    d,cfg,td,s,w,e=setup()
    with torch.no_grad():
        terms=e.curve.terms(s,w)
        assert terms['shared_curve_length']<1e-8
        e.curve.u.add_(torch.randn_like(e.curve.u)*5)
        e.curve.v.add_(torch.randn_like(e.curve.v)*5)
        xy=e.curve.coordinates();owner=e.owner
        for f in (0,1):assert torch.all(torch.diff(xy[owner==f,f])>=0)
    # A 3-4-5 UV section has correct length, though a nonzero alignment cost.
    dd=dict(family=np.array([0]),start=np.array([0.]),node_ds=np.array([[0.,2.5,5.]]))
    curve=c.SharedCurve(dd,np.array([0,0]),np.array([0.,5.]),{},'cpu')
    curve.set_coordinates(torch.tensor([[.1,0.],[.103,4.]],dtype=torch.float64))
    terms=curve.terms(torch.tensor([[1.,0.,0.,4.]]),torch.ones(5)*1000)
    assert terms['shared_curve_length']<1e-7
    assert terms['shared_curve_alignment']>0


def test_saved_geometry_and_parallel_query():
    d,cfg,td,s,w,e=setup()
    ids=np.array([0,0,0,1,1]);ds=np.array([-50.,50.,400.,50.,400.])
    e.curve.add_plan('probe',ids,ds)
    with torch.no_grad():e.curve.v[:3].add_(torch.tensor([0.,5.,10.]))
    mapped=b.density.map_state(s)
    actual=tuple(x[:,0] for x in e.curve.hypothesis_queries(mapped,w,e.curve.plans['probe'],mapped[ids,0,None]))
    saved=e.curve.arrays()
    expected=c.saved_points(d,saved,s.detach().numpy(),w.detach().numpy(),ids,ds)
    for a,z in zip(actual,expected):np.testing.assert_allclose(a.detach(),z,rtol=1e-5,atol=1e-4)
    e2=b.BucketUV(d,cfg,'cpu');e2.curve.set_coordinates(torch.tensor(saved['shared_curve_phase_v']))
    torch.testing.assert_close(e.curve.coordinates(),e2.curve.coordinates())


def test_missing_nodes_fail_and_single_node_has_no_invented_slope():
    d,cfg,td,s,w,e=setup()
    with pytest.raises(ValueError,match='no perpendicular nodes'):
        c.SharedCurve(d,np.array([0]),np.array([10.]),cfg,'cpu')
    curve=c.SharedCurve(d,np.array([0,1]),np.array([10.,10.]),cfg,'cpu');curve.initialize(s,w)
    ids=np.array([0,0,1,1]);ds=np.array([-100.,100.,-100.,100.]);curve.add_plan('q',ids,ds)
    values=curve.evaluate(s,w,'q')
    for value in values:
        torch.testing.assert_close(value[::2],value[1::2])
    with torch.no_grad():s[:,1:3].add_(123.)
    for before,after in zip(values,curve.evaluate(s,w,'q')):
        torch.testing.assert_close(before,after)


def test_shared_objective_cannot_silently_use_straight_geometry():
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_soft_winding as soft
    with pytest.raises(ValueError,match='requires its curve'):
        soft.objective(None,None,None,None,None,dict(shared_curve_geometry=True))


def test_saved_scoring_requires_curve():
    with pytest.raises(ValueError,match='Missing saved'):
        c.saved_points({}, {},None,None,[],[])


def test_subvoxel_knots_at_large_coordinates():
    d=dict(family=np.array([0]),start=np.array([0.]),node_ds=np.array([[0.,1.,2.]]))
    arc=np.array([0.,.002,1.]);curve=c.SharedCurve(d,np.zeros(3,int),arc,{},'cpu')
    state=torch.tensor([[10.,.25,40000.,4.]])
    widths=torch.ones(30)*8000
    curve.initialize(state,widths)
    terms=curve.terms(state,widths)
    assert terms['shared_curve_length']<1e-15
    assert terms['shared_curve_bending']<1e-10 # double roundoff amplified by .002-vx span


def test_reversed_source_order_still_forward_in_uv():
    d=dict(family=np.array([1]),start=np.array([0.]),node_ds=np.array([[0.,-5.,-10.]]))
    curve=c.SharedCurve(d,np.zeros(3,int),np.array([0.,5.,10.]),{},'cpu')
    state=torch.tensor([[2.,.2,40000.,4.]])
    widths=torch.ones(10)*1000
    curve.initialize(state,widths)
    with torch.no_grad():curve.v[1:].add_(10)
    assert torch.all(torch.diff(curve.coordinates()[:,1])<0)


def test_checkpoint_tiny_increment_roundtrip():
    d,cfg,td,s,w,e=setup()
    with torch.no_grad():e.curve.u[1]=-1000
    snapshot=e.curve.arrays()
    restored=b.BucketUV(d,cfg,'cpu').curve;restored.restore(snapshot)
    torch.testing.assert_close(restored.u,e.curve.u,rtol=0,atol=0)
    # Legacy coordinate-only checkpoints can contain equal (not reversed) knots.
    del snapshot['shared_curve_raw_u'];del snapshot['shared_curve_raw_v']
    restored.restore(snapshot)
    torch.testing.assert_close(restored.coordinates(),e.curve.coordinates(),rtol=0,atol=1e-12)


def test_sign_report_uses_learned_curve():
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_winding_order import ordering
    d,cfg,td,s,w,e=setup();mapped=s.detach().clone();mapped[:,0]=mapped[:,0].round()
    before=ordering(mapped,w.detach(),d,cfg,saved=e.curve.arrays())
    with torch.no_grad():e.curve.u[e.owner==1]+=3
    after=ordering(mapped,w.detach(),d,cfg,saved=e.curve.arrays())
    torch.testing.assert_close(after-before,torch.full_like(before,3))


def test_parallel_family_filter_plan_indices_match_training():
    d,cfg,_,_,_=fixture();d['links']=np.empty((0,4));d['parallel'][0]=True
    cfg.update(shared_curve_geometry=True,filter_constraint_families=True,lr=.073)
    ext=b.BucketUV(d,cfg,'cpu');ext.prepare_fields(d,{})
    np.testing.assert_array_equal(ext.curve.plans['a'][1].numpy(),[100.,250.])


@pytest.mark.parametrize('family',[0,1])
def test_alignment_penalizes_transverse_motion_and_not_translation(family):
    d=dict(family=np.array([family]),start=np.array([0.]),node_ds=np.array([[0.,5.,10.]]))
    curve=c.SharedCurve(d,np.zeros(3,int),np.array([0.,5.,10.]),{},'cpu')
    state=torch.tensor([[2.,.2,40000.,4.]])
    widths=torch.ones(10)*1000
    curve.initialize(state,widths)
    initial=curve.coordinates().detach().clone()
    translated=initial+torch.tensor([.1,20.])
    curve.set_coordinates(translated)
    assert curve.terms(state,widths)['shared_curve_alignment']<1e-12
    bent=translated.clone()
    bent[:,1-family]+=torch.tensor([0.,1.,2.])/(1000 if family==1 else 1)
    curve.set_coordinates(bent)
    terms=curve.terms(state,widths)
    torch.testing.assert_close(terms['shared_curve_alignment'],torch.tensor(.4,dtype=torch.float64),rtol=1e-5,atol=1e-7)
    terms['shared_curve_alignment'].backward()
    transverse=curve.v if family==0 else curve.u
    assert transverse.grad.abs().max()>0


def test_initial_pair_losses_preserved_for_constant_widths():
    d,cfg,td,s,w,e=setup()
    # With constant widths the old per-hypothesis integration and new shared
    # phase are identical. All three pair terms (including sign) must survive.
    with torch.no_grad():w.fill_(1000.)
    e.curve.ready=False;e.curve.initialize(s,w)
    edges=torch.arange(len(d['i']));i=td['i'];j=td['j']
    cfg.update(sign_violation_weight=30,soft_pair_weight=1,probability_pair_weight=.3)
    kwargs=dict(sigma_a=e.cavity.values[i],sigma_b=e.cavity.values[j])
    expected=b.density.pair_cost(s[i],s[j],td['family'][i],td['family'][j],
        td['ds_a'].to(s),td['ds_b'].to(s),w,td['sign'],td['distance'],td['parallel'],
        dict(cfg,shared_curve_hypothesis_mapping=False),**kwargs)
    sa,sb,da,db=e.pair_states(s,w,td,edges)
    actual=b.density.pair_cost(sa,sb,td['family'][i],td['family'][j],
        da,db,w,td['sign'],td['distance'],td['parallel'],cfg,**kwargs)
    for a,z in zip(actual,expected):torch.testing.assert_close(a,z,rtol=1e-5,atol=1e-6)


def test_legacy_physical_transplant_and_saved_scoring_agree():
    d,cfg,td,s,w,e=setup(integer_geometry=False)
    with torch.no_grad():
        w.copy_(torch.arange(1,11)*1000.)
        e.curve.ready=False;e.curve.initialize(s,w)
    nodes=torch.where(e.owner==0)[0]
    labels=torch.tensor([[1.,5.]]).expand(len(nodes),-1)
    q,v=e.curve.node_hypotheses(s,w,labels,nodes)
    anchor=e.curve.evaluate(s.double(),w.double(),'_anchor')[1][0]
    # Original H fiber is straight at init: each winding's phase advance uses
    # that winding's own width, not a copy of the continuous-N phase advance.
    for k in range(2):
        h=s[0].expand(len(nodes),-1).clone();h[:,0]=labels[:,k];h[:,1]=anchor
        expected=c.soft.points(h,torch.zeros(len(nodes),dtype=torch.long),e.curve.arc[nodes].to(s),w)[0]
        torch.testing.assert_close(q[:,k],expected,atol=2e-6,rtol=1e-6)
    advance=q-labels-anchor.to(s)
    assert not torch.allclose(advance[:,0],advance[:,1])
    snapshot=dict(e.curve.arrays(),state=s.detach().numpy())
    mapped=s.detach().clone();mapped[0,0]=5
    saved=c.saved_points(d,snapshot,mapped.numpy(),w.detach().numpy(),np.array([0,0,0]),d['ds_a'])
    np.testing.assert_allclose(saved[0],q[:,1].detach(),atol=2e-6)
    # Width gradients are present through the hypothesis-specific map.
    q.sum().backward();assert w.grad.abs().max()>0


def test_corrected_pair_grouping_matches_ungrouped():
    d,cfg,td,s,w,e=setup()
    with torch.no_grad():w.copy_(torch.arange(1,11)*1000.);e.curve.ready=False;e.curve.initialize(s,w)
    i,j=td['i'],td['j'];sa,sb,da,db=e.pair_states(s,w,td,torch.arange(3))
    args=(sa,sb,td['family'][i],td['family'][j],da,db,w,td['sign'].to(s),td['distance'].to(s),td['parallel'])
    kwargs=dict(sigma_a=e.cavity.values[i],sigma_b=e.cavity.values[j])
    plain=b.density.pair_cost(*args,cfg,**kwargs)
    grouped=b.density.pair_cost(*args,dict(cfg,pair_sigma_groups=True),**kwargs)
    for a,z in zip(plain,grouped):torch.testing.assert_close(a,z,rtol=1e-5,atol=1e-6)


@pytest.mark.parametrize('family',[0,1])
def test_legacy_physical_curve_displacement_is_shared_not_winding_increment(family):
    d,cfg,td,s,w,e=setup(integer_geometry=False)
    with torch.no_grad():
        w.copy_(torch.arange(1,11)*1000.);e.curve.ready=False;e.curve.initialize(s,w)
        if family==1:
            nodes=torch.where(e.owner==1)[0]
            e.curve.u[nodes]+=torch.tensor([0.,.01,.02],dtype=e.curve.u.dtype)
    nodes=torch.where(e.owner==family)[0]
    labels=torch.tensor([[1.,5.]]).expand(len(nodes),-1)
    q,_=e.curve.node_hypotheses(s,w,labels,nodes)
    anchor=e.curve.evaluate(s,w,'_anchor')[1][family]
    bounds=torch.cat((w.new_zeros(1),w.cumsum(0)))
    travel=c.soft.uv.q_to_u(q,w,bounds)-c.soft.uv.q_to_u(labels+anchor,w,bounds)
    torch.testing.assert_close(travel[:,0],travel[:,1],atol=.005,rtol=1e-5)
    assert travel.abs().max()>0
