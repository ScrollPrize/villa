import json

import numpy as np
import pytest
import torch

from vesuvius.neural_tracing.fiber_trace_3d import fiber_soft_winding as soft


def fixture():
    d = dict(family=np.array([0, 1]), phase=np.array([.2, .3]), z=np.array([0., 10.]),
        i=np.array([0]), j=np.array([1]), ds_a=np.array([0.]), ds_b=np.array([0.]),
        sign=np.array([1.]), distance=np.array([.5]), parallel=np.array([False]),
        links=np.array([[0., 1., 0., 0.]]), source_id=np.array([1, 2]),
        start=np.zeros(2), end=np.ones(2), radius=np.array([100., 200.]),
        radial_initial_centers=np.array([1., 1.]), radial_initial_widths=np.array([1000., 1200., 1400.]))
    nodes = dict(owner=np.array([0, 1]), ds=np.zeros(2), phase=d['phase'],
        z=d['z'], weight=np.ones(2), v_owner=np.array([0, 1]), v_ds=np.zeros(2),
        v_z=d['z'], v_weight=np.ones(2))
    return d, nodes


def test_initialization_width_penalty_removed():
    with pytest.raises(ValueError, match='width_prior was removed'):
        soft.validate({'width_prior': .001})
    soft.validate({'width_prior': 0.})  # Historical disabled configs remain readable.
    d, nodes = fixture()
    cfg = dict(width_smooth=0.)
    data, samples = soft.tensor_inputs(d, nodes, cfg, 'cpu')
    state = torch.tensor([[.25,.2,0.,2.],[.25,.3,10.,2.]])
    raw = torch.tensor([1000.,1200.,1400.])
    loss, terms = soft.objective(state, raw, raw, data, samples, cfg)
    other, _ = soft.objective(state, raw, raw * 2, data, samples, cfg)
    assert 'width_prior' not in terms
    torch.testing.assert_close(loss, other)


@pytest.mark.parametrize('batch',[46,32768])
def test_graph_fitting_batch_has_every_edge_once_without_rng(batch):
    torch.manual_seed(123)
    before=torch.get_rng_state().clone()
    edges=soft.constraint_batch_indices(46,batch,'cpu')
    assert edges.tolist()==list(range(46))
    assert torch.equal(before,torch.get_rng_state())


def test_smaller_batch_retains_seeded_random_sampling():
    torch.manual_seed(123)
    expected=torch.randint(46,(8,))
    torch.manual_seed(123)
    actual=soft.constraint_batch_indices(46,8,'cpu')
    assert torch.equal(actual,expected)


@pytest.mark.parametrize('shared',[False,True])
def test_angle_prior_uses_phase_not_distribution_center(shared):
    d,nodes=fixture();cfg=dict(prior_weight=1.)
    # Isolate the angular prior: shared continuation requires its real pair geometry.
    d['links'] = np.empty((0, 4))
    data,samples=soft.tensor_inputs(d,nodes,cfg,'cpu')
    state=torch.tensor([[.25,.2,0.,2.],[.25,.3,10.,2.]],requires_grad=True)
    raw=torch.tensor([1000.,1200.,1400.])
    class SharedEndpoint:
        curve=True
        def field_points(self,state,widths,key):
            return state[:,0]+state[:,1],state[:,1],state[:,2]
        def terms(self,*args):return {}
    ext=SharedEndpoint() if shared else None
    _,terms=soft.objective(state,raw,raw,data,samples,cfg,extension=ext)
    expected=state[:,3].sigmoid().mean()*0
    torch.testing.assert_close(terms['anchor'],expected)
    grad=torch.autograd.grad(terms['anchor'],state)[0]
    assert (grad[:,0]==0).all()
    # Changing the distribution center alone does not change anchor phase.
    changed=state+torch.tensor([[.125,0.,0.,0.],[.125,0.,0.,0.]])
    _,other=soft.objective(changed,raw,raw,data,samples,cfg,extension=ext)
    torch.testing.assert_close(terms['anchor'],other['anchor'])
    changed=state+torch.tensor([[0.,.125,0.,0.],[0.,.125,0.,0.]])
    _,other=soft.objective(changed,raw,raw,data,samples,cfg,extension=ext)
    assert other['anchor']>terms['anchor']


def test_gauge_invariance_entire_objective():
    d, nodes = fixture(); cfg = dict(perpendicular_relative_weight=2.5)
    data, samples = soft.tensor_inputs(d, nodes, cfg, 'cpu')
    state = torch.tensor([[.8, .2, 0., 2.], [1.1, .3, 10., -1.]], requires_grad=True)
    raw = torch.tensor([1000., 1200., 1400.], requires_grad=True)
    before, terms = soft.objective(state, raw, raw.detach(), data, samples, cfg)
    shifted = state + torch.tensor([[2., -2., 0., 0.], [-1., 1., 0., 0.]])
    after, other = soft.objective(shifted, raw, raw.detach(), data, samples, cfg)
    for key in terms: torch.testing.assert_close(terms[key], other[key], atol=1e-6, rtol=1e-5)
    before.backward()
    assert torch.isfinite(state.grad).all() and torch.isfinite(raw.grad).all()


def test_radial_width_initialization_is_not_a_default_target():
    d,nodes=fixture();cfg={}
    data,samples=soft.tensor_inputs(d,nodes,cfg,'cpu')
    state=torch.tensor([[.8,.2,0.,2.],[1.1,.3,10.,-1.]])
    widths=torch.tensor([800.,1600.,2200.],requires_grad=True)
    initial=torch.tensor([1000.,1200.,1400.])
    _,terms=soft.objective(state,widths,initial,data,samples,cfg)
    assert 'width_prior' not in terms


def test_bounded_diagnostics_preserve_full_loss():
    d,nodes=fixture()
    for key in ('i','j','ds_a','ds_b','sign','distance','parallel'):
        d[key]=np.repeat(d[key],7)
    d['distance']=np.linspace(.1,1.5,7)
    cfg=dict(perpendicular_relative_weight=2.5,diagnostic_batch_size=3)
    data,samples=soft.tensor_inputs(d,nodes,cfg,'cpu')
    state=torch.tensor([[.8,.2,0.,2.],[1.1,.3,10.,-1.]])
    raw=torch.tensor([1000.,1200.,1400.])
    _,expected=soft.objective(state,raw,raw,data,samples,cfg)
    _,actual=soft.diagnostic_objective(state,raw,raw,data,samples,cfg,None,None)
    assert actual.keys()==expected.keys()
    for key in expected:torch.testing.assert_close(actual[key],expected[key])


def test_reuse_constraint_work_with_configured_pair_loss():
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_implicit_winding as density
    d,nodes=fixture();cfg=dict(perpendicular_relative_weight=2.5,winding_sigma=.5)
    data,samples=soft.tensor_inputs(d,nodes,cfg,'cpu')
    state=torch.tensor([[.8,.2,0.,2.],[1.1,.3,10.,-1.]],requires_grad=True)
    raw=torch.tensor([1000.,1200.,1400.],requires_grad=True)
    a,ta=soft.objective(state,raw,raw.detach(),data,samples,cfg,pair_cost=density.pair_cost)
    ga=torch.autograd.grad(a,(state,raw))
    b,tb=soft.objective(state,raw,raw.detach(),data,samples,dict(cfg,reuse_constraint_work=True),pair_cost=density.pair_cost)
    gb=torch.autograd.grad(b,(state,raw))
    for key in ta:torch.testing.assert_close(ta[key],tb[key],atol=0,rtol=0)
    for x,y in zip(ga,gb):torch.testing.assert_close(x,y,atol=0,rtol=0)


def test_constraint_family_filter():
    d,nodes=fixture()
    d['i']=np.zeros(4,dtype=np.int64)
    d['j']=np.array([0,1,0,1])
    d['parallel']=np.array([True,True,False,False])
    for k in ('ds_a','ds_b','sign','distance'):d[k]=np.arange(4,dtype=float)
    filtered,stats=soft.constraint_family_filter(d,True)
    assert filtered['distance'].tolist()==[0,3]
    assert stats['all']['removed']==2
    assert stats['parallel']['removed']==stats['perpendicular']['removed']==1
    assert stats['all']['removed_percent']==50
    assert len(d['i'])==4 and filtered['family'] is d['family']
    data,_=soft.tensor_inputs(d,nodes,{'filter_constraint_families':True},'cpu')
    assert len(data['class_weight'])==len(data['i'])==2
    expected=soft.uv.constraint_class_weights(filtered['parallel'],{'constraint_class_balance':True})
    np.testing.assert_allclose(data['class_weight'].numpy(),expected)
    unchanged,_=soft.constraint_family_filter(d,False)
    np.testing.assert_array_equal(unchanged['i'],d['i'])


def test_exact_arc_across_width_boundary_and_extrapolation():
    widths = torch.tensor([100., 200.], dtype=torch.float64)
    state = torch.tensor([[0., .9, 7., 0.], [0., .9, 7., 0.]], dtype=torch.float64)
    q, u, v = soft.points(state, torch.tensor([0, 1]), torch.tensor([30., -30.]), widths)
    torch.testing.assert_close(q, torch.tensor([1.1, .9], dtype=torch.float64))
    torch.testing.assert_close(v, torch.tensor([7., -23.], dtype=torch.float64))
    q, _, _ = soft.points(state[:1], torch.tensor([0]), torch.tensor([-190.]), widths)
    assert q.item() == pytest.approx(-1.)


def errors(na, ua, nb, ub, *, fd=0, sign=1., distance=1.):
    a = torch.tensor([[na, ua, 0., 0.]], dtype=torch.float64, requires_grad=True)
    b = torch.tensor([[nb, ub, 0., 0.]], dtype=torch.float64, requires_grad=True)
    fa, fb = (0, 1) if fd == 1 else (1, 0) if fd == -1 else (0, 0)
    result = soft.pair_errors(a, b, torch.tensor([fa]), torch.tensor([fb]),
        torch.zeros(1), torch.zeros(1), torch.tensor([1000.] * 5),
        torch.tensor([sign]), torch.tensor([distance]), torch.tensor([fd == 0]))
    return a, b, result


def test_seam_alignment_and_phase_gradient():
    a, b, (wind, phase, _) = errors(0., .99, 1., .01, distance=0.)
    assert wind.item() == pytest.approx(.02)
    assert phase.item() == pytest.approx(.02)
    phase.square().sum().backward()
    assert b.grad[0, 1] > 0 and a.grad[0, 1] < 0


def test_winding_gradient_is_not_nearest_integer_self_target():
    a, b, (wind, _, _) = errors(0., .2, 3.4, .2, distance=1.)
    wind.square().sum().backward()
    assert wind.item() == pytest.approx(2.4)
    assert b.grad[0, 0] > 0 and a.grad[0, 0] < 0


def test_half_turn_gauge_tie():
    _, _, first = errors(0., 0., 0., .5)
    _, _, shifted = errors(0., 0., -1., 1.5)
    for a, b in zip(first, shifted): torch.testing.assert_close(a, b)


@pytest.mark.parametrize('fd,nb,sign', [(1, 0., 1.), (-1, 1., 1.), (-1, 0., -1.), (1, 0., 0.)])
def test_perpendicular_calibration(fd, nb, sign):
    _, _, (wind, _, _) = errors(0., .2, nb, .2, fd=fd, sign=sign, distance=.5)
    assert wind.item() == pytest.approx(0.)


def test_rejection_gradient_and_entropy():
    d, nodes = fixture(); data, samples = soft.tensor_inputs(d, nodes, {}, 'cpu')
    state = torch.tensor([[0., .2, 0., 0.], [40., .2, 0., 0.]], requires_grad=True)
    raw = torch.tensor([1000., 1200., 1400.])
    loss, terms = soft.objective(state, raw, raw, data, samples, dict(continuation_weight=0.))
    loss.backward()
    assert (state.grad[:, 3] > 0).all()  # Descent disables incompatible fibers.
    assert terms['entropy'].item() == pytest.approx(.025 * np.log(2))


def test_angular_sampling_reversals():
    positions = soft.sample_arcs(np.array([0., 10., 20.]), np.array([0., .5, 0.]), 0., 20., .125)
    np.testing.assert_allclose(positions, np.arange(0., 20.1, 2.5))


@pytest.mark.parametrize('key,value', [('steps', 0), ('log_every', 0), ('batch_size', -1),
    ('distance_sigma', 0), ('v_pair_scale', 0), ('angular_sample_degrees', float('nan')),
    ('phase_weight', -1), ('steps', 1.5), ('devices', '0,1')])
def test_invalid_settings(key, value):
    with pytest.raises(ValueError): soft.validate({key: value})


def test_continuation_cannot_hide_whole_turn_jump():
    d, nodes = fixture(); data, samples = soft.tensor_inputs(d, nodes, {}, 'cpu')
    state = torch.tensor([[0., .2, 0., 4.], [0., .2, 0., 4.]])
    raw = torch.tensor([1000., 1200., 1400.])
    _, first = soft.objective(state, raw, raw, data, samples, {})
    state[1, 0] += 1
    _, second = soft.objective(state, raw, raw, data, samples, {})
    assert first['continuation'] == 0
    assert second['continuation'] > 9


def test_sample_preparation_reuses_signed_geometry(monkeypatch):
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_normal_sign_visualization as vis
    from vesuvius.neural_tracing.tracers.shared.annotations import afv
    class Database:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def execute(self, *args): return [('root', '{}')]
    monkeypatch.setattr(soft.uv, '_ro', lambda path: Database())
    monkeypatch.setattr(vis, 'load_umbilicus_frame', lambda *args: np.array([[0., 0., 0.], [0., 0., 20.]]))
    xyz = np.array([[10., 0., 0.], [10., 0., 20.]])
    monkeypatch.setattr(afv, 'read_afv_polylines', lambda *args: [({'id': 5}, xyz)])
    d = dict(source_id=np.array([5]), family=np.array([1]), start=np.array([2.]), end=np.array([18.]),
             node_ds=np.array([[0., -8., -16.]]), i=np.array([0]), j=np.array([0]),
             ds_a=np.array([-3.]), ds_b=np.array([-10.]))
    samples = soft.prepare_samples(dict(afv='unused', umbilicus='unused'), d)
    np.testing.assert_allclose(samples['ds'], [0., -16.])
    np.testing.assert_allclose(samples['z'], [2., 18.])
    np.testing.assert_allclose(samples['v_z'], [5., 12.])
    assert samples['weight'].sum() == pytest.approx(1.)
    assert samples['v_weight'].sum() == pytest.approx(1.)

    cfg_dense=dict(afv='unused',umbilicus='unused',arc_prior_spacing_vx=5.)
    dense=soft.prepare_samples(cfg_dense,d,existing=samples)
    np.testing.assert_allclose(dense['ds'],samples['ds'])  # V: no extra phase prior
    np.testing.assert_allclose(dense['v_ds'],[-16.,-15.,-10.,-5.,-3.,0.])
    np.testing.assert_allclose(dense['v_z'],2.-dense['v_ds'])
    assert dense['v_weight'].sum()==pytest.approx(1.)
    d['family'][:]=0
    dense_h=soft.prepare_samples(cfg_dense,d,existing=samples)
    np.testing.assert_allclose(dense_h['ds'],[0.,-5.,-10.,-15.,-16.])
    np.testing.assert_allclose(dense_h['v_ds'],samples['v_ds'])
    assert dense_h['weight'].sum()==pytest.approx(1.)
    for invalid in (0.,-1.,float('nan')):
        with pytest.raises(ValueError,match='arc_prior_spacing_vx'):
            soft.prepare_samples(dict(cfg_dense,arc_prior_spacing_vx=invalid),d)

    # Node-only priors use exact perpendicular topology, deduplicating equal
    # base arcs; parallel bases and endpoint fallbacks must not reappear.
    monkeypatch.setattr(afv, 'read_afv_polylines', lambda *args: [({'id': k}, xyz) for k in (5,6,7)])
    d=dict(source_id=np.array([5,6,7]),family=np.array([1,0,0]),
        start=np.full(3,2.),end=np.full(3,18.),
        node_ds=np.array([[0.,-8.,-16.],[0.,8.,16.],[0.,8.,16.]]),
        i=np.array([0,0,0,1]),j=np.array([1,1,1,2]),
        ds_a=np.array([-3.,-10.,-3.,1.]),ds_b=np.array([4.,9.,4.,15.]),
        parallel=np.array([False,False,False,True]))
    cfg=dict(afv='unused',umbilicus='unused',shared_curve_geometry=True,shared_node_priors_only=True)
    nodes=soft.prepare_samples(cfg,d)
    np.testing.assert_array_equal(nodes['owner'],[0,0,1,1])
    np.testing.assert_array_equal(nodes['v_owner'],nodes['owner'])
    np.testing.assert_allclose(nodes['ds'],[-3.,-10.,4.,9.])
    np.testing.assert_allclose(nodes['v_ds'],nodes['ds'])
    np.testing.assert_allclose(nodes['z'],[5.,12.,6.,11.])
    np.testing.assert_allclose(nodes['v_z'],nodes['z'])
    np.testing.assert_allclose(nodes['weight'],.5)
    np.testing.assert_allclose(nodes['v_weight'],.5)
    d['parallel'][:]=True
    empty=soft.prepare_samples(cfg,d)
    assert all(v.shape==(0,) for v in empty.values())


def test_full_combination_keeps_shared_priors_once():
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_implicit_winding as imp
    d,nodes=fixture();cfg=dict(winding_sigma=.5,soft_pair_weight=1,probability_pair_weight=.3)
    data,samples=soft.tensor_inputs(d,nodes,cfg,'cpu')
    state=torch.tensor([[0.,.2,0.,4.],[1.,.3,2.,4.]])
    raw=torch.tensor([1000.,1200.,1400.])
    _,base=soft.objective(state,raw,raw,data,samples,cfg)
    _,mixed=soft.objective(state,raw,raw,data,samples,cfg,pair_cost=imp.pair_cost)
    _,pure=soft.objective(state,raw,raw,data,samples,dict(cfg,probability_pair_weight=0),pair_cost=imp.pair_cost)
    for key in base:
        torch.testing.assert_close(base[key],pure[key])
        if key not in ('winding','phase','vertical'):assert torch.equal(base[key],mixed[key])


def test_plateau_switch_and_stop_saved_on_nonlog_iteration(tmp_path,monkeypatch):
    d,nodes=fixture()
    np.savez(tmp_path/'data.npz',**d);np.savez(tmp_path/'samples.npz',**nodes)
    def constant(state,*args,**kwargs):
        loss=state.sum()*0+1
        return loss,dict(test=loss)
    monkeypatch.setattr(soft,'diagnostic_objective',constant)
    cfg=dict(tensor_cache=str(tmp_path/'data.npz'),soft_samples=str(tmp_path/'samples.npz'),
        output_root=str(tmp_path/'solve'),winding_count=3,steps=20,log_every=10,batch_size=1,
        lr=.01,width_lr=.02,plateau_lr_patience=2)
    result=soft.solve(cfg,device='cpu')
    assert result['stopping_reason']=='plateau'
    assert result['final']['iteration']==4
    assert result['final']['learning_rates']==pytest.approx([.001,.002])
    monitor=json.loads((tmp_path/'solve/plateau_history.json').read_text())
    assert [(e['iteration'],e['action']) for e in monitor['events']]==[(2,'reduce'),(4,'stop')]
    assert len(monitor['rows'])==5
    assert (tmp_path/'solve/solution.npz').exists()


def test_trusted_fibers_cannot_be_disabled(tmp_path):
    d,nodes=fixture()
    np.savez(tmp_path/'data.npz',**d);np.savez(tmp_path/'samples.npz',**nodes)
    cfg=dict(tensor_cache=str(tmp_path/'data.npz'),soft_samples=str(tmp_path/'samples.npz'),
        output_root=str(tmp_path/'solve'),winding_count=3,steps=5,log_every=1,batch_size=1,
        lr=.01,width_lr=.01,force_keep_fibers=True,rejection_cost=0,entropy_weight=1)
    soft.solve(cfg,device='cpu')
    with np.load(tmp_path/'solve/solution.npz') as saved:
        np.testing.assert_array_equal(saved['state'][:,3],100.)
        assert np.isfinite(saved['state']).all()


def test_solver_cpu_smoke(tmp_path):
    d, nodes = fixture()
    np.savez(tmp_path / 'data.npz', **d); np.savez(tmp_path / 'samples.npz', **nodes)
    cfg = dict(tensor_cache=str(tmp_path / 'data.npz'), soft_samples=str(tmp_path / 'samples.npz'),
        output_root=str(tmp_path / 'solve'), winding_count=3, steps=5, log_every=1, batch_size=1,
        lr=.01, width_lr=.01,save_diagnostic_checkpoints=True)
    result = soft.solve(cfg, device='cpu')
    history = json.loads((tmp_path / 'solve/history.json').read_text())
    assert result['final']['loss'] < history[0]['loss']
    assert len(history) == 6
    with np.load(tmp_path / 'solve/solution.npz') as solution:
        assert np.isfinite(solution['state']).all()
        assert (solution['widths'] > 0).all()
        with np.load(tmp_path/'solve/checkpoints/000005.npz') as last:
            for key in ('state','widths'):np.testing.assert_array_equal(last[key],solution[key])
        cfg.update(output_root=str(tmp_path/'without_checkpoints'),save_diagnostic_checkpoints=False)
        soft.solve(cfg,device='cpu')
        with np.load(tmp_path/'without_checkpoints/solution.npz') as plain:
            for key in ('state','widths'):np.testing.assert_array_equal(plain[key],solution[key])
    assert len(list((tmp_path/'solve/checkpoints').glob('*.npz')))==6
    cfg.update(output_root=str(tmp_path/'uncapped'),cap_batch_to_graph=False,batch_size=23)
    # Even legacy "uncapped" recipes must not duplicate a graph that fits.
    result=soft.solve(cfg,device='cpu')
    assert result['effective_batch_size']==1
    assert result['constraint_sampling']=='full_graph'
