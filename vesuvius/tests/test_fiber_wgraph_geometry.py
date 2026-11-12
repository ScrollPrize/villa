"""Production winding optimizer and fixed-assignment adapter regressions."""
import os
import numpy as np
import pytest

from vesuvius.neural_tracing.fiber_trace_3d.fiber_linear_irls import solve_l1
from vesuvius.neural_tracing.fiber_trace_3d.fiber_flow_uv import saved_assignment_inputs


def test_linear_irls_recovers_integer_chain_without_wrapping():
    # Inconsistent redundant measurement avoids exact-CG zero residuals while
    # exercising robust, nonwrapped winding distances spanning several turns.
    a=np.array([0,1,2,0]);b=np.array([1,2,3,3])
    target=np.array([1.,2.,-1.,2.1]);weight=np.array([1.,1.,1.,.1])
    x,residual=solve_l1(4,a,b,target,weight,np.zeros(4),1e-5,
                      dev='cpu',irls=5,cg_iters=3,log=lambda _:None)
    assert np.isfinite(x).all() and np.isfinite(residual).all()
    # Historic epsilon=.05 smoothing distributes .005 to each strong edge.
    np.testing.assert_allclose(np.diff(x),target[:3]+.005,atol=2e-6)
    np.testing.assert_allclose(residual,np.abs(x[b]-x[a]-target),atol=1e-6)


def test_uv_can_explicitly_retain_all_components():
    d=dict(source_id=np.array(['a','b','c']),family=np.array([0,1,0]),start=np.zeros(3),
        end=np.ones(3)*10,node_ds=np.array([[0.,-10],[0,10],[0,10]]),links=np.empty((0,4)),
        i=np.array([0]),j=np.array([1]),ds_a=np.array([-3.]),ds_b=np.array([4.]),
        distance=np.array([.5]),sign=np.ones(1),parallel=np.zeros(1,bool))
    s=dict(source_id=d['source_id'],component=np.array([0,0,1]),winding=np.array([-4,-3,99]))
    out,labels=saved_assignment_inputs(d,s,'all')
    np.testing.assert_array_equal(out['source_id'],d['source_id'])
    np.testing.assert_array_equal(labels,s['winding'])


def test_packed_priors_match_original_polyline_sampling(tmp_path):
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_min_flow import Seam
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_min_flow_full import PackedPaths,KEY_DTYPE,PACKED_GEOMETRY_FORMAT
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_packed_priors import geometry_priors,fixed_targets
    axis=np.array([[0.,0,-20],[0,0,20]])
    seam=Seam(axis)
    curves=[np.array([[3.,-4,1],[4,-2,2],[3,2,3],[2,4,4]]),
            np.array([[7.,0,1],[7,0,3],[7,1,8]])]
    ids=np.array(['a','b']);offset=np.r_[0,np.cumsum(list(map(len,curves)))]
    keys=np.empty(offset[-1],KEY_DTYPE);turns=np.empty(offset[-1],np.int64)
    for i,xyz in enumerate(curves):
        sl=slice(offset[i],offset[i+1]);keys['source'][sl]=i
        keys['arc'][sl]=np.r_[0,np.cumsum(np.linalg.norm(np.diff(xyz,axis=0),axis=1))]
        turns[sl]=seam.polyline(xyz)
    np.savez(tmp_path/'catalog.npz',format_version=PACKED_GEOMETRY_FORMAT,volume='volume',
             source_id=ids,point_offset=offset,axis=axis)
    np.save(tmp_path/'keys.npy',keys);np.save(tmp_path/'xyz.npy',np.concatenate(curves));np.save(tmp_path/'turns.npy',turns)
    paths=PackedPaths(tmp_path)
    result=geometry_priors(paths,ids,np.array([0,1]),spacing=2.,batch_fibers=1,batch_samples=2)
    for i,xyz in enumerate(curves):
        arc=keys['arc'][offset[i]:offset[i+1]]
        samples=np.linspace(0,arc[-1],max(2,int(np.ceil(arc[-1]/2.))+1))
        selected=result['sample_owner']==i
        np.testing.assert_allclose(result['sample_arc'][selected],samples,atol=2e-15)
        positions,_=paths.at_many(np.full(len(samples),ids[i]),samples)
        np.testing.assert_allclose(result['sample_z'][selected],positions[:,2],atol=2e-15)
        assert result['radii'][i]==np.median(np.linalg.norm(seam.relative(xyz),axis=1))
        assert result['minimum_turn'][i]==turns[offset[i]:offset[i+1]].min()
        assert result['maximum_turn'][i]==turns[offset[i]:offset[i+1]].max()
    data=dict(source_id=ids,i=np.array([0,1,0]),j=np.array([1,0,1]),
              ds_a=np.array([1.,2.,3.]),ds_b=np.array([2.,1.,4.]))
    paths.data=data;labels=np.array([2.,3.])
    a,ta=paths.ends('a');b,tb=paths.ends('b')
    expected=labels[data['j']]-labels[data['i']]+tb-ta-paths.seam.segments(a,b)
    np.testing.assert_array_equal(fixed_targets(paths,data,labels,batch_size=1),expected)


@pytest.mark.parametrize('devices',['0','1'])
@pytest.mark.parametrize('exclusive',[False,True])
def test_winding_stage_uses_typed_inputs_and_writes_referenced_results(tmp_path,monkeypatch,devices,exclusive):
    import uuid
    from pathlib import Path
    from vc_object_store import CollectionReader
    from .fiber_graph_fixture import measured_store,uid
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_graph_store import write_results,PREFIX
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_min_flow_full as flow,fiber_normal_sign as normal
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_wgraph_geometry as adapter
    graph=measured_store(tmp_path,[(0,10,20,False,.5)])
    ids=np.array([uid(10),uid(20)])
    classified=tmp_path/'classified.vcobj';signed=tmp_path/'signed.vcobj'
    classification_run=write_results(classified,[graph],'classification','test',{'fibers':[
        dict(fiber_uuid=fid,enabled=1,family=i) for i,fid in enumerate(ids)]})
    sign_run=write_results(signed,[classified],'normal_sign','test',{'fibers':[
        dict(fiber_uuid=fid,normal_sign=-1,sign_probability=.1) for fid in ids]},input_runs=[classification_run])
    def prepare(settings):
        root=Path(settings['output_root'])
        assert settings['database']==str(Path(graph).resolve())
        assert settings['fibers']==str(signed)
        assert settings['graph_cache']==str(tmp_path/'graph-cache')
        with np.load(settings['comparison']) as saved:
            np.testing.assert_array_equal(saved['ids'],ids)
            np.testing.assert_array_equal(saved['labels'],[0,1])
        with np.load(settings['normal_solution']) as saved:
            np.testing.assert_array_equal(saved['normal_sign'],[-1,-1])
            assert saved['normal_convention'].item()=='inward'
        parts=root/'graph_parts';parts.mkdir()
        np.savez(parts/'000.npz',a=[0],b=[1],target=[0],weight=[1.])
        return dict(source_id=ids,family=np.array([0,1]))
    def managed(settings,module,source):
        assert settings['devices']==devices
        assert settings['gpu_manager']['exclusive'] is exclusive
        np.savez(Path(settings['output_root'])/'solution.npz',winding=[4,4],continuous=[4.01,4.02],component=[0,0])
        return dict(weighted_l1=.01,mean_residual=.01,max_residual=.01)
    monkeypatch.setattr(flow,'prepare',prepare)
    monkeypatch.setattr(normal,'launch_tensor_solve',managed)
    output=tmp_path/'winding.vcobj'
    result=adapter.run_stage('winding',dict(database=str(signed),result_store=str(output),
        output_root=str(tmp_path/'work'),graph_cache=str(tmp_path/'graph-cache'),devices=devices,
        gpu_manager=dict(exclusive=exclusive)))
    with CollectionReader(str(output)) as store:
        assert len(store.objects('fiber'))==2
        definitions={t['name']:t for t in store.extension_tables()}
        columns=[c['name'] for c in definitions[PREFIX+'fibers']['columns']]
        rows=[dict(zip(columns,row)) for row in store.extension_rows(PREFIX+'fibers')
              if row[1]==uuid.UUID(result['run_uuid']).bytes]
        assert len(rows)==2
        assert [r['selected_winding'] for r in rows]==[4,4]
        assert [r['winding_mean'] for r in rows]==[4.01,4.02]
        columns=[c['name'] for c in definitions[PREFIX+'inputs']['columns']]
        inputs=[dict(zip(columns,row)) for row in store.extension_rows(PREFIX+'inputs')]
        assert any(r['input_run_uuid']==uuid.UUID(sign_run).bytes for r in inputs)


def test_packed_worker_reuses_graph_cache_without_readback(tmp_path,monkeypatch):
    from .fiber_graph_fixture import measured_store
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_min_flow_full as full
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_graph_store import FiberGraph
    source=measured_store(tmp_path,[(0,10,20,False,.5)])
    cache=tmp_path/'shared-columns'
    FiberGraph(source).save_cache(cache)
    def forbidden(*args,**kwargs):raise AssertionError('Unexpected full graph readback')
    monkeypatch.setattr(FiberGraph,'__init__',forbidden)
    monkeypatch.setattr(full,'PackedPaths',lambda *args,**kwargs:object())
    full._init(dict(database=str(source),output_root=str(tmp_path/'work'),graph_cache=str(cache)),'constraints')
    assert len(full._worker[2].event_ids)==1
    assert not (tmp_path/'work'/'constraint_columns').exists()


@pytest.mark.parametrize('exclusive',[False,True])
def test_wrapped_stage_preserves_explicit_exclusive_setting(tmp_path,exclusive):
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_wgraph_stages import _solver_config
    config=_solver_config(dict(gpu_manager=dict(exclusive=exclusive)),tmp_path)
    assert config['gpu_manager']['exclusive'] is exclusive
    assert _solver_config({},tmp_path)['gpu_manager']['exclusive'] is False


@pytest.mark.cuda
@pytest.mark.skipif(not os.environ.get('VILLA_GPU_MANAGER_GRANT'),reason='CUDA smoke requires a managed GPU grant')
def test_linear_irls_cuda_matches_cpu():
    import torch
    if not torch.cuda.is_available():pytest.skip('CUDA is unavailable in this managed environment')
    args=(4,np.array([0,1,2,0]),np.array([1,2,3,3]),np.array([1.,2.,-1.,2.1]),
          np.array([1.,1.,1.,.1]),np.zeros(4),1e-5)
    cpu,cr=solve_l1(*args,dev='cpu',irls=5,cg_iters=3,log=lambda _:None)
    gpu,gr=solve_l1(*args,dev='cuda',irls=5,cg_iters=3,log=lambda _:None)
    assert np.isfinite(gpu).all() and np.isfinite(gr).all()
    # The global additive gauge is immaterial; compare all relative positions.
    np.testing.assert_allclose(gpu-gpu.mean(),cpu-cpu.mean(),atol=5e-5,rtol=5e-5)
    np.testing.assert_allclose(gr,cr,atol=5e-5,rtol=5e-5)


def test_explicit_run_selectors_cannot_mix_classification_sign_and_winding(tmp_path,monkeypatch):
    from .fiber_graph_fixture import measured_store
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_graph_store import write_results,FiberGraph
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_wgraph_geometry as adapter
    graph=measured_store(tmp_path,[(0,10,20,False,.5)])
    previous=graph
    runs={}
    for name,stage,parent in [('c1','classification',None),('s1','normal_sign','c1'),
                              ('w1','winding','s1'),('c2','classification',None),
                              ('s2','normal_sign','c2')]:
        output=tmp_path/(name+'.vcobj')
        runs[name]=write_results(output,[previous],stage,'test',{'fibers':[]},
                                input_runs=[runs[parent] if parent else None])
        previous=output
    def forbidden(*args,**kwargs):raise AssertionError('Lineage check must precede graph preparation')
    monkeypatch.setattr(FiberGraph,'__init__',forbidden)
    config=dict(database=str(previous),output_root=str(tmp_path/'work'))
    # Multiple classification/sign runs exist; winding resolves its exact pins.
    assert adapter._input_lineage(config,runs['w1'])==dict(classification=runs['c1'],normal_sign=runs['s1'])
    with pytest.raises(ValueError,match='classification.*ancestry'):
        adapter._prepare(dict(config,classification_run_uuid=runs['c1'],normal_sign_run_uuid=runs['s2']))
    with pytest.raises(ValueError,match='normal-sign.*ancestry'):
        adapter._run_uv(dict(config,input_run_uuid=runs['w1'],normal_sign_run_uuid=runs['s2']))
    with pytest.raises(ValueError,match='classification.*ancestry'):
        adapter._run_uv(dict(config,input_run_uuid=runs['w1'],classification_run_uuid=runs['c2']))
    assert not (tmp_path/'work').exists()
