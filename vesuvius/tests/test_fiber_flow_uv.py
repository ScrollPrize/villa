import numpy as np
import torch
import pytest
from vesuvius.neural_tracing.fiber_trace_3d.fiber_flow_uv import supported
from vesuvius.neural_tracing.fiber_trace_3d.fiber_flow_uv import saved_assignment_inputs
from vesuvius.neural_tracing.fiber_trace_3d.fiber_scaled_lbfgs import scaled_joint
from .fiber_graph_fixture import uid


def test_supported_preserves_indices_and_perp_support():
    d=dict(source_id=np.arange(4),family=np.array([0,1,0,0]),start=np.zeros(4),end=np.ones(4),
           node_ds=np.ones((4,2)),i=np.array([0,1,2]),j=np.array([1,2,3]),parallel=np.array([False,False,True]))
    for key in ('ds_a','ds_b','distance','sign'):d[key]=np.ones(3)
    result,keep,edges=supported(d)
    assert keep.tolist()==[True,True,True,False]
    assert edges.tolist()==[True,True,False]
    assert result['source_id'].tolist()==[0,1,2]


def test_freeze_winding_keeps_labels_but_fits_geometry():
    state=torch.nn.Parameter(torch.tensor([[3.,0,0,100.]],dtype=torch.float64))
    raw=torch.nn.Parameter(torch.ones(2,dtype=torch.float64))
    ext=torch.nn.Module();ext.register_parameter('u',torch.nn.Parameter(torch.tensor([2.],dtype=torch.float64)))
    def obj():return state[:,0].square().sum()+(ext.u-5).square().sum()+raw.sum()*0,{}
    scaled_joint(obj,state,raw,ext,100,freeze_winding=True,freeze_widths=True)
    assert state[0,0]==3
    assert abs(float(ext.u.detach())-5)<1e-5


def test_saved_assignment_uses_main_component_and_unsigned_arcs():
    d=dict(source_id=np.array(list(map(uid,[7,8,9]))),family=np.array([0,1,0]),start=np.zeros(3),
        end=np.ones(3)*10,node_ds=np.array([[0.,-10],[0,10],[0,10]]),links=np.empty((0,4)),
        i=np.array([0]),j=np.array([1]),ds_a=np.array([-3.]),ds_b=np.array([4.]),
        distance=np.array([.5]),sign=np.ones(1),parallel=np.zeros(1,bool))
    s=dict(source_id=d['source_id'],component=np.array([0,0,1]),winding=np.array([-4,-3,99]))
    out,labels=saved_assignment_inputs(d,s)
    np.testing.assert_array_equal(out['source_id'],list(map(uid,[7,8])))
    np.testing.assert_array_equal(labels,[-4,-3])
    np.testing.assert_array_equal(out['ds_a'],[3])
    np.testing.assert_array_equal(out['node_ds'],[[0,10],[0,10]])
    np.testing.assert_array_equal(d['ds_a'],[-3])


def test_saved_assignment_stages_native_store_without_resolving_windings(tmp_path,monkeypatch):
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_flow_uv as uv
    from vesuvius.neural_tracing.tracers.shared.annotations.fiber_store import write_initial_fibers, read_polylines
    ids=np.array([uid(7),uid(8)])
    d=dict(source_id=ids,family=np.array([0,1]),start=np.zeros(2),end=np.ones(2)*10,
        node_ds=np.array([[0.,-10.],[0.,10.]]),links=np.empty((0,4)),i=np.array([0]),j=np.array([1]),
        ds_a=np.array([-3.]),ds_b=np.array([4.]),distance=np.array([.5]),sign=np.ones(1),parallel=np.zeros(1,bool))
    source=tmp_path/'fibers.vcobj'
    write_initial_fibers(source,[(dict(id=s),np.array([[0.,0,0],[10.,0,0]])) for s in ids],
                         volume=uid(999),workers=2,packed=True)
    tensor=tmp_path/'tensors.npz';np.savez(tensor,**d)
    solution=tmp_path/'solution.npz';np.savez(solution,source_id=ids,component=[0,0],winding=[-4,-3])
    def fit(config):
        stage=tmp_path/'fit/fixed_input'
        assert config['flow_root']==str(stage)
        assert (stage/'selected_with_gt.vcobj').resolve()==source
        assert [r['id'] for r,_ in read_polylines(stage/'selected_with_gt.vcobj')]==ids.tolist()
        np.testing.assert_array_equal(np.load(stage/'selected_solution.npz')['winding'],[-4,-3])
        np.testing.assert_array_equal(np.load(stage/'selected_tensors.npz')['ds_a'],[3.])
        return {'checked':True}
    monkeypatch.setattr(uv,'run',fit)
    result=uv.run_saved(dict(fibers=str(source),tensor_cache=str(tensor),solution=str(solution),
        output_root=str(tmp_path/'fit'),parameters={},umbilicus='axis.json',gpu_manager={}))
    assert result=={'checked':True}


def test_solve_summary_identifies_persisted_result_run(tmp_path,monkeypatch):
    import json
    import uuid
    from vc_object_store import CollectionReader
    from .fiber_graph_fixture import measured_store
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_flow_uv as uv
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_shared_curve as shared,fiber_scaled_lbfgs as scaled
    source=measured_store(tmp_path,[(0,10,20,False,.5)])
    root=tmp_path/'fit';root.mkdir();flow=tmp_path/'flow';flow.mkdir()
    (flow/'selected_with_gt.vcobj').symlink_to(source)
    np.savez(root/'tensors.npz',source_id=np.array([uid(10),uid(20)]),
        family=[0,1],start=[0.,0.],end=[10.,10.],node_ds=[[0.,10.],[0.,10.]],direction=[1.,1.],
        i=[0],j=[1],ds_a=[5.],ds_b=[5.],parallel=[False],fixed_target=[0.],
        fixed_winding=[2.,2.],phase=[.1,.1],z=[10.,10.],width_initial=np.arange(1000.,1600.,100.),
        sample_owner=[0,0,1,1],sample_ds=[0.,10.,0.,10.],sample_phase=[.1,.11,.1,.1],sample_z=[10.,10.,10.,20.])
    (root/'selection.json').write_text(json.dumps(dict(global_integer_shift=3)))
    # Exercise the real geometry/readout/store writer on CPU; optimizer and CUDA
    # placement are irrelevant to this result-identity regression.
    original_tensor=torch.as_tensor;original_curve=shared.SharedCurve
    def cpu_tensor(value,*args,**kwargs):
        if kwargs.get('device')=='cuda':kwargs['device']='cpu'
        return original_tensor(value,*args,**kwargs)
    monkeypatch.setattr(torch,'as_tensor',cpu_tensor)
    monkeypatch.setattr(shared,'SharedCurve',lambda d,owner,arc,config,device:original_curve(d,owner,arc,config,'cpu'))
    monkeypatch.setattr(scaled,'uncertainty_joint',lambda *args,**kwargs:dict(iterations=0))
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda:0)
    output=tmp_path/'uv.vcobj'
    summary=uv.solve(dict(output_root=str(root),flow_root=str(flow),result_store=str(output)))
    assert uuid.UUID(summary['run_uuid'])
    assert json.loads((root/'summary.json').read_text())['run_uuid']==summary['run_uuid']
    with CollectionReader(str(output)) as store:
        runs=store.extension_rows('ext_fiber_graph_solve_runs')
        positions=store.extension_rows('ext_fiber_graph_solve_positions')
    assert len(runs)==1 and runs[0][0]==uuid.UUID(summary['run_uuid']).bytes
    assert all(row[2]==3 and row[3]==2 for row in positions)
    assert abs(positions[0][-3]-.1)<1e-12  # U excludes the selected k=2.


def test_fixed_flow_objective_batches_preserve_full_loss_and_gradients():
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_shared_curve import SharedCurve
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_flow_uv_objective import FixedFlowObjective
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_soft_winding as soft
    d=dict(family=np.array([0,1,0]),start=np.zeros(3),end=np.full(3,20.),
        node_ds=np.tile([0.,20.],(3,1)),i=np.array([0,1,0,2,1]),j=np.array([1,2,2,0,2]),
        ds_a=np.array([1.,2.,4.,5.,3.]),ds_b=np.array([2.,4.,5.,7.,6.]),
        parallel=np.array([False,False,True,True,False]),fixed_target=np.array([0.,1.,0.,-1.,0.]),
        sample_owner=np.array([0,0,1,1,1,2,2]),sample_ds=np.array([0.,20.,0.,10.,20.,0.,20.]),
        sample_phase=np.array([.1,.15,.2,.2,.2,.3,.35]),sample_z=np.arange(7.)*2.)
    config=dict(straight_fixed_length_geometry=True,shared_curve_integer_geometry=True,
                width_parameterization='monotone_relative',minimum_winding_width=100.,batch_size=99)
    state=torch.nn.Parameter(torch.tensor([[2.,.1,10.,100.],[2.,.2,20.,100.],[3.,.3,30.,100.]],dtype=torch.float64))
    widths=torch.tensor([100.,110.,130.,160.,200.,250.],dtype=torch.float64)
    raw=torch.nn.Parameter(soft.encode_widths(widths,config))
    curve=SharedCurve(d,np.arange(3),np.zeros(3),config,'cpu');curve.initialize(state,widths)
    targets=(state,raw,*curve.parameters())
    full=FixedFlowObjective(curve,state,raw,d,widths,config)
    value,terms=full();gradient=torch.autograd.grad(value,targets,allow_unused=True)
    # Independently retain the original all-at-once reduction as the oracle.
    decoded=soft.decode_widths(raw,config)
    qa,qb,va,vb,xa,xb=full.edge_values(slice(None),decoded)
    t=lambda x:torch.as_tensor(x,dtype=torch.float64)
    parallel=torch.as_tensor(d['parallel']);family=torch.as_tensor(d['family'][d['i']])
    same=torch.as_tensor(full.same);weight=t(full.weights)
    _,phase,z=full.query(d['sample_owner'],d['sample_ds'],decoded)
    sample_weight=1/t(np.bincount(d['sample_owner'])[d['sample_owner']])
    reference=dict(
        winding=(weight*((~parallel)|(family==1))*(qb-qa-t(d['fixed_target'])).square()).mean(),
        vertical=(weight*((~parallel)|(family==0))*((vb-va)/1000.).square()).mean(),
        meeting=((xa[same]-xb[same]).square()+(va[same]-vb[same]).square()).mean()/128.**2,
        phase_prior=(soft.circular(phase-t(d['sample_phase'])).square()*sample_weight).sum()/3*.000625,
        z_prior=(((z-t(d['sample_z']))/1000.).square()*sample_weight).sum()/3*.01)
    reference.update(soft.width_regularization(raw,decoded,widths,config))
    for key,expected in reference.items():torch.testing.assert_close(terms[key],expected,atol=1e-12,rtol=1e-12)
    reference_gradient=torch.autograd.grad(sum(reference.values()),targets,allow_unused=True)
    for expected,actual in zip(reference_gradient,gradient):
        if expected is not None:torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)
    for batch in (1,2,3):
        objective=FixedFlowObjective(curve,state,raw,d,widths,dict(config,batch_size=batch))
        total=0.;accumulated=[torch.zeros_like(p) for p in targets]
        for part,_ in objective.parts():
            total+=float(part.detach())
            for target,grad in zip(accumulated,torch.autograd.grad(part,targets,allow_unused=True)):
                if grad is not None:target.add_(grad)
        np.testing.assert_allclose(total,float(value.detach()),rtol=1e-12)
        for expected,actual in zip(gradient,accumulated):
            if expected is not None:torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-6)
        for expected,actual in zip(full.errors(),objective.errors()):np.testing.assert_array_equal(actual,expected)
        for expected,actual in zip(full.samples(),objective.samples()):np.testing.assert_array_equal(actual,expected)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA requires managed host validation')
def test_fixed_flow_objective_cuda_microbenchmark():
    """Bounded synthetic GPU closure smoke; report, not a flaky speed assertion."""
    import json
    import time
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_shared_curve import SharedCurve
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_flow_uv_objective import FixedFlowObjective
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_soft_winding as soft
    n,m=8192,131072;rng=np.random.default_rng(42)
    d=dict(family=np.arange(n)%2,start=np.zeros(n),end=np.full(n,20.),
        node_ds=np.tile([0.,20.],(n,1)),i=rng.integers(n,size=m),j=rng.integers(n,size=m),
        ds_a=rng.uniform(0,20,m),ds_b=rng.uniform(0,20,m),parallel=np.arange(m)%3==0,
        fixed_target=np.zeros(m),sample_owner=np.repeat(np.arange(n),2),sample_ds=np.tile([0.,20.],n),
        sample_phase=np.full(n*2,.2),sample_z=np.zeros(n*2))
    config=dict(straight_fixed_length_geometry=True,shared_curve_integer_geometry=True,
                width_parameterization='monotone_relative',batch_size=32768)
    state=torch.nn.Parameter(torch.tensor(np.column_stack((np.full(n,2.),np.full(n,.2),np.zeros(n),np.full(n,100.))),
                                          device='cuda',dtype=torch.float64))
    widths=torch.tensor([1000.,1100.,1300.,1600.,2000.,2500.],device='cuda',dtype=torch.float64)
    raw=torch.nn.Parameter(soft.encode_widths(widths,config))
    curve=SharedCurve(d,np.arange(n),np.zeros(n),config,'cuda');curve.initialize(state,widths)
    objective=FixedFlowObjective(curve,state,raw,d,widths,config);targets=(state,raw,*curve.parameters())
    torch.cuda.reset_peak_memory_stats();times=[]
    for repeat in range(3):
        gradients=[torch.zeros_like(p) for p in targets]
        torch.cuda.synchronize();started=time.perf_counter()
        for part,_ in objective.parts():
            assert torch.isfinite(part)
            for total,gradient in zip(gradients,torch.autograd.grad(part,targets,allow_unused=True)):
                if gradient is not None:total.add_(gradient)
        torch.cuda.synchronize();times.append(time.perf_counter()-started)
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
    print(json.dumps(dict(benchmark='fixed_flow_uv_closure',fibers=n,edges=m,batch_size=32768,
        warmup_seconds=times[0],closure_seconds=times[1:],
        peak_vram_bytes=torch.cuda.max_memory_allocated(),
        full_135m_edge_equivalent_seconds=135000000/m*np.mean(times[1:]))),flush=True)
