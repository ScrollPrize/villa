import numpy as np
import pytest
import torch
from vesuvius.neural_tracing.fiber_trace_3d import fiber_v_endpoint_offsets as off
from vesuvius.neural_tracing.fiber_trace_3d import fiber_soft_winding as soft
from vesuvius.neural_tracing.fiber_trace_3d import fiber_implicit_winding as imp


def fixture():
    d=dict(source_id=np.array([7,7,8]),start=np.array([0.,100.,0.]),end=np.array([100.,200.,200.]),
        family=np.array([0,0,1]),node_ds=np.array([[0,50,100],[0,50,100],[0,-100,-200.]]),
        i=np.array([0,1,1]),j=np.array([2,2,2]),ds_a=np.array([99.,1.,60.]),ds_b=np.array([-10.,-30.,-70.]),
        parallel=np.zeros(3,dtype=bool),sign=np.ones(3),distance=np.ones(3)*.5,links=np.array([[0,1,100,0]]))
    cfg=dict(v_offset_match_weight=2.,v_offset_smooth_h=100.,v_offset_smooth_v=1000.,
        v_offset_anchor_weight=.001,v_offset_lr=2.,winding_sigma=.5,filter_constraint_families=True)
    return d,cfg


def test_exact_deformation_no_merging_and_equalities():
    d,_=fixture();top=off.topology(d,0)
    assert top['v_offset_a'][0]!=top['v_offset_a'][1]
    delta=np.array([30.,32.,-20.])
    o,check=off.exact_deformation(top,delta,10,10)
    np.testing.assert_allclose(delta+o[top['v_offset_b']]-o[top['v_offset_a']],0,atol=1e-10)
    assert check['relative_stationarity']<1e-10
    # Scaling the entire deformation objective changes neither solution nor fit.
    oo,_=off.exact_deformation(top,delta,100,100)
    np.testing.assert_allclose(o,oo,atol=1e-9)


def test_exact_deformation_rigid_translation_free():
    d,_=fixture();top=off.topology(d,0)
    o,check=off.exact_deformation(top,np.full(3,100.),10,10)
    assert check['deformation_energy']<1e-15
    assert np.max(np.abs(o))>1 # free rigid shifts are not penalized


def test_exact_hard_reference_targets():
    d,_=fixture();top=off.topology(d,0)
    delta=np.array([30.,32.,-20.]);nodes=np.array([top['v_offset_a'][0],top['v_offset_a'][2]])
    values=np.array([500.,600.])
    o,check=off.exact_deformation(top,delta,10,10,'u',.001,nodes,values)
    np.testing.assert_allclose(o[nodes],values,atol=1e-10)
    np.testing.assert_allclose(delta+o[top['v_offset_b']]-o[top['v_offset_a']],0,atol=1e-10)
    with pytest.raises(ValueError,match='Inconsistent fixed'):
        off.exact_deformation(top,delta,10,10,fixed_nodes=[nodes[0],nodes[0]],fixed_values=[1,2])


def test_exact_transverse_anchor_not_longitudinal():
    d,_=fixture();top=off.topology(d,0)
    for axis,family in [('u',1),('v',0)]:
        o,c=off.exact_deformation(top,np.full(3,100.),10,10,axis,.001)
        np.testing.assert_allclose(o[top['v_offset_family']==family],0,atol=1e-5)
        assert np.max(np.abs(o[top['v_offset_family']!=family]))>99
        assert c['max_equality_error']<1e-10


def test_exact_deformation_rejects_conflicting_cycle():
    d,_=fixture();d['i'][:]=0;d['ds_a'][:]=1;d['ds_b'][:]=-10
    top=off.topology(d,0)
    with pytest.raises(ValueError,match='cycle'):
        off.exact_deformation(top,np.array([1.,2.,1.]),10,10)


def test_fixed_refinement_frozen_and_decreasing(tmp_path):
    d,cfg=fixture()
    state=torch.tensor([[2.,.1,0.,8.],[2.,.1,0.,8.],[2.,.1,110.,8.]])
    widths=torch.ones(6)*1000
    cfg.update(u_endpoint_offsets=True,u_offset_match_weight=20,u_offset_smooth_h=10,
        u_offset_smooth_v=10,u_offset_anchor_weight=.001,steps=40,log_every=40,
        output_root=str(tmp_path/'out'),initial_solution=str(tmp_path/'initial.npz'))
    np.savez(cfg['initial_solution'],state=state.numpy(),widths=widths.numpy(),
        **{k:d[k] for k in ('source_id','start','end','family')})
    retained,_,mask,_,_=off.fixed_geometry(d,state,widths,cfg)
    assert retained.all() and mask.all()
    rejected=state.clone();rejected[2,3]=-8
    assert not off.fixed_geometry(d,rejected,widths,cfg)[2].any()
    result=off.refine_fixed(d,state,widths,cfg)
    assert result['final']['u']['mean']<result['before']['u']['mean']
    assert result['final']['v']['mean']<result['before']['v']['mean']
    with np.load(tmp_path/'out'/'solution.npz') as saved:
        np.testing.assert_array_equal(saved['state'],state.numpy())
        np.testing.assert_array_equal(saved['widths'],widths.numpy())


@pytest.mark.parametrize('cavity', [False, True])
def test_exact_saved_solution(tmp_path,cavity,monkeypatch):
    d,cfg=fixture()
    state=np.array([[2.,.1,0.,8.],[2.,.1,0.,8.],[2.,.1,110.,8.]],dtype=np.float32)
    widths=np.ones(6,dtype=np.float32)*1000
    np.savez(tmp_path/'data.npz',**d)
    extra=dict(winding_sigma=np.array([2.,.2,.2],dtype=np.float32)) if cavity else {}
    np.savez(tmp_path/'initial.npz',state=state,widths=widths,**extra,
        **{k:d[k] for k in ('source_id','start','end','family')})
    cfg.update(tensor_cache=str(tmp_path/'data.npz'),initial_solution=str(tmp_path/'initial.npz'),
        output_root=str(tmp_path/'out'),u_offset_smooth_h=10,u_offset_smooth_v=10,exact_transverse_anchor=.001)
    if cavity:cfg.update(cavity_winding_sigma=True,sigma_max=8.)
    if cavity:
        monkeypatch.setattr(imp,'run',lambda *a,**k:pytest.fail('Postpass must not launch optimizer'))
        result=off.run_exact(cfg)
    else:
        result=off.run_exact(cfg)
    assert result['after']['u']['max']<1e-10
    assert result['after']['v']['max']<1e-10
    from lasagna.inference_provenance import sha256_file
    solution_hash=sha256_file(tmp_path/'out/solution.npz')
    validated=off.run_exact(dict(cfg,validate_existing=True))
    assert validated['config']['initial_solution_sha256']==sha256_file(tmp_path/'initial.npz')
    assert sha256_file(tmp_path/'out/solution.npz')==solution_hash
    with pytest.raises(ValueError,match='configuration'):
        off.run_exact(dict(cfg,validate_existing=True,exact_transverse_anchor=.01))
    with np.load(tmp_path/'out'/'solution.npz') as s:
        assert s['state'].tobytes()==state.tobytes()
        assert s['widths'].tobytes()==widths.tobytes()
        assert s['u_offsets'].dtype==np.float64
        assert s['v_offset_a'][0]!=s['v_offset_a'][1]
        if cavity:
            np.testing.assert_array_equal(s['retained'],[False,True,True])
            np.testing.assert_array_equal(s['fixed_event_indices'],[1,2])
            np.testing.assert_array_equal(s['winding_sigma'],extra['winding_sigma'])


def test_exact_saved_reference_stays_fixed(tmp_path):
    d,cfg=fixture()
    state=np.array([[2.,.1,0.,8.],[2.,.1,0.,8.],[2.,.1,110.,8.]],dtype=np.float32)
    widths=np.ones(6,dtype=np.float32)*1000
    np.savez(tmp_path/'data.npz',**d)
    np.savez(tmp_path/'initial.npz',state=state,widths=widths,
        **{k:d[k] for k in ('source_id','start','end','family')})
    np.savez(tmp_path/'reference.npz',owner=np.array([0]),source=np.array([900]),
        state=np.array([[2.,.2,500.,100.]]),u=np.array([2400.]),family=np.array([1]),
        ds=np.array([50.]),sign=np.ones(1),distance=np.full(1,.5),parallel=np.zeros(1,dtype=bool))
    cfg.update(tensor_cache=str(tmp_path/'data.npz'),initial_solution=str(tmp_path/'initial.npz'),
        output_root=str(tmp_path/'out'),u_offset_smooth_h=10,u_offset_smooth_v=10,exact_transverse_anchor=.001,
        fixed_reference_events=str(tmp_path/'reference.npz'))
    result=off.run_exact(cfg)
    assert result['fixed_reference_meetings']==1
    assert result['reference_after']['u']['max']<1e-9
    assert result['reference_after']['v']['max']<1e-9
    with np.load(tmp_path/'out/solution.npz') as saved:
        node=saved['fixed_reference_nodes'][0]
        # Frozen reference physical U=2400, not current-width q=2200.
        assert saved['u_offsets'][node]==pytest.approx(250,abs=1e-4)
        assert saved['v_offsets'][node]==pytest.approx(500)
        np.testing.assert_array_equal(saved['state'],state)


def test_whole_fiber_topology_and_direction():
    d,cfg=fixture(); t=off.topology(d)
    assert t['v_offset_a'][0]==t['v_offset_a'][1] # across split boundary
    np.testing.assert_allclose(t['v_offset_arc'],[100,160,10,30,70])
    np.testing.assert_array_equal(t['v_offset_left'],[0,2,3])
    d['source_id'][1]=9
    t=off.topology(d)
    assert t['v_offset_a'][0]!=t['v_offset_a'][1]


def test_smoothness_spacing_and_gradients():
    d,cfg=fixture(); e=off.EndpointOffsets(d,cfg,'cpu')
    with torch.no_grad():e.values.copy_(torch.tensor([0.,6.,0.,2.,6.]))
    terms=e.regularization()
    assert terms['v_offset_smooth_h'].item()==pytest.approx(1.)
    assert terms['v_offset_smooth_v'].item()==pytest.approx(10.)
    sum(terms.values()).backward();assert torch.isfinite(e.values.grad).all()


def test_bounded_merge_and_filtered_edge_index():
    d,cfg=fixture()
    d['i']=np.zeros(3,dtype=np.int64);d['ds_a']=np.array([0.,3.,6.])
    t=off.topology(d)
    assert t['v_offset_a'][0]==t['v_offset_a'][1]
    assert t['v_offset_a'][2]!=t['v_offset_a'][1]
    d['parallel'][0]=True # H/V parallel event removed
    e=off.EndpointOffsets(d,cfg,'cpu')
    np.testing.assert_array_equal(e.a.numpy(),t['v_offset_a'][1:])
    np.testing.assert_array_equal(e.b.numpy(),t['v_offset_b'][1:])


def test_probability_gating_and_offset_match():
    d,cfg=fixture(); e=off.EndpointOffsets(d,cfg,'cpu')
    data,_=soft.tensor_inputs(d,{},cfg,'cpu')
    state=torch.tensor([[2.2,.1,0.,4.],[2.,.1,0.,4.],[2.,.1,110.,4.]],requires_grad=True)
    widths=torch.ones(6)*1000
    edges=torch.tensor([0])
    initial=e.terms(state,widths,data,edges)['v_offset_match']
    assert initial>0
    initial.backward();assert state.grad[:,0].abs().sum()>0
    with torch.no_grad():e.values[e.b[0]]=-100.
    assert e.terms(state,widths,data,edges)['v_offset_match']==0
    data['parallel'][:]=True
    assert e.terms(state,widths,data,edges)['v_offset_match']==0


@pytest.mark.parametrize('with_u',[False,True])
@pytest.mark.parametrize('application',['added_only','all_constraints'])
def test_extension_solver_save(tmp_path,with_u,application):
    from .test_fiber_soft_winding import fixture as base_fixture
    d,samples=base_fixture();_,cfg=fixture()
    cfg['offset_application']=application
    if with_u:cfg.update(u_endpoint_offsets=True,u_offset_match_weight=2,u_offset_smooth_h=1000,
        u_offset_smooth_v=100,u_offset_anchor_weight=.001)
    d['node_ds']=np.array([[0,.5,1.],[0,.5,1.]])
    np.savez(tmp_path/'data.npz',**d);np.savez(tmp_path/'samples.npz',**samples)
    cfg.update(tensor_cache=str(tmp_path/'data.npz'),soft_samples=str(tmp_path/'samples.npz'),
        output_root=str(tmp_path/'solve'),steps=2,log_every=1,winding_count=3)
    result=soft.solve(cfg,device='cpu',pair_cost=imp.pair_cost,extension_factory=off.EndpointOffsets)
    assert result['final']['v_offset_nodes']==2
    with np.load(tmp_path/'solve/solution.npz') as saved:
        assert 'v_offsets' in saved and 'v_offset_a' in saved
        assert np.isfinite(saved['v_offsets']).all()
        assert ('u_offsets' in saved)==with_u
    cfg.update(initial_solution=str(tmp_path/'solve/solution.npz'),output_root=str(tmp_path/'reload'))
    soft.solve(cfg,device='cpu',pair_cost=imp.pair_cost,extension_factory=off.EndpointOffsets)
    import json
    history=json.loads((tmp_path/'reload/history.json').read_text())
    assert history[0]['loss']==pytest.approx(result['final']['loss'],abs=1e-6)


def test_u_offset_physical_match_and_gating():
    d,cfg=fixture();cfg.update(u_endpoint_offsets=True,u_offset_match_weight=2,
        u_offset_smooth_h=1000,u_offset_smooth_v=100,u_offset_anchor_weight=.001,
        winding_sigma=.01)
    e=off.EndpointOffsets(d,cfg,'cpu');data,_=soft.tensor_inputs(d,{},cfg,'cpu')
    state=torch.tensor([[2.,.1,0.,4.],[2.,.1,0.,4.],[2.,.3,110.,4.]],requires_grad=True)
    widths=torch.ones(6)*1000;edges=torch.tensor([0])
    base=e.terms(state,widths,data,edges)
    assert base['u_offset_match']>0
    base['u_offset_match'].backward();assert e.u_values.grad.abs().sum()>0
    with torch.no_grad():e.u_values[e.b[0]]=-101. # raw xa=2199,xb=2300
    result=e.terms(state,widths,data,edges)
    assert result['u_offset_match']<1e-10
    torch.testing.assert_close(result['v_offset_match'],base['v_offset_match'])
    # Swapped longitudinal/transverse regularization weights.
    with torch.no_grad():e.u_values.copy_(torch.tensor([0.,6.,0.,2.,6.]))
    terms=e.regularization()
    assert terms['u_offset_smooth_h'].item()==pytest.approx(10.)
    assert terms['u_offset_smooth_v'].item()==pytest.approx(1.)


def test_all_pair_classes_use_corrected_physical_coordinates():
    d,cfg=fixture();cfg.update(winding_sigma=.1)
    a=torch.tensor([[2.,.1,0.,4.]]*4,requires_grad=True)
    b=torch.tensor([[2.,.2,100.,4.]]*4)
    fa=torch.tensor([0,1,0,0]);fb=torch.tensor([0,1,1,1])
    par=torch.tensor([True,True,False,False]);b[3,0]=3.
    z=torch.zeros(4);w=torch.ones(6)*1000
    args=(a,b,fa,fb,z,z,w,torch.ones(4),torch.ones(4),par,cfg)
    base=imp.pair_cost(*args)
    du=torch.full((4,),-100.,requires_grad=True);dv=torch.full((4,),-100.,requires_grad=True)
    actual=imp.pair_cost(*args,offsets_a=(z,z),offsets_b=(du,dv))
    # Current source semantics include endpoint phase in the measured-distance term.
    # Equal widths give an independent explicit-coordinate oracle for these offsets.
    shifted = b.clone()
    shifted[:, 1] -= .1
    shifted[:, 2] -= 100.
    reference = imp.pair_cost(a, shifted, fa, fb, z, z, w,
                              torch.ones(4), torch.ones(4), par, cfg)
    for result, expected in zip(actual, reference):
        torch.testing.assert_close(result, expected)
    assert not torch.equal(base[0], actual[0])
    assert actual[1].max()<1e-10 and actual[2].max()==0
    zero=imp.pair_cost(*args,offsets_a=(z,z),offsets_b=(z,z))
    for x,y in zip(base,zero):torch.testing.assert_close(x,y,atol=1e-6,rtol=1e-5)


def test_whole_fiber_interpolation():
    d,cfg=fixture();e=off.EndpointOffsets(d,cfg,'cpu')
    samples=dict(owner=np.array([0,1,2]),ds=np.array([99.,30.,-50.]),
                 v_owner=np.array([0,1,2]),v_ds=np.array([99.,30.,-50.]))
    e.prepare_fields(d,samples)
    with torch.no_grad():e.values.copy_(torch.tensor([10.,70.,100.,120.,160.]))
    _,v=e.field_offsets('unary')
    torch.testing.assert_close(v,torch.tensor([10.,40.,140.]))
    torch.testing.assert_close(e.field_offsets('link_a')[1],e.field_offsets('link_b')[1])
    v.sum().backward();assert e.values.grad.abs().sum()>0
