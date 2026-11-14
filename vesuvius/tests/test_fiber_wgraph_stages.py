"""Stage orchestration uses existing solvers and immutable typed result stores."""
import json
import os
from pathlib import Path
import uuid

import numpy as np
import pytest
from vc_object_store import CollectionReader

from .fiber_graph_fixture import measured_store,uid
from vesuvius.neural_tracing.fiber_trace_3d import fiber_wgraph_stages as stages
from vesuvius.neural_tracing.fiber_trace_3d import fiber_normal_sign as normal
from vesuvius.neural_tracing.tracers.shared.annotations.fiber_store import fibers_digest
from vesuvius.neural_tracing.fiber_trace_3d.fiber_graph_store import original_fiber_source


def test_hvb_then_sign_preserves_classification_and_shares_graph(tmp_path,monkeypatch):
    graph=measured_store(tmp_path,[(0,10,20,False,.5)],isolated_ids=[30])
    ids=np.asarray(list(map(uid,[10,20,30])))
    with CollectionReader(str(graph)) as store:source=store.source
    feature=tmp_path/'features.json'
    _,fiber_source=original_fiber_source(graph)
    feature.write_text(json.dumps(dict(format='fiber-axial-priors',version=1,source_ids=ids.tolist(),
        agreement=[-1,1,0],fibers_digest=fibers_digest(graph),fiber_source=fiber_source,umbilicus_sha256='test')))
    priors=tmp_path/'priors.json'
    priors.write_text(json.dumps(dict(format='fiber-outward-priors',source_store=source,
                                      source_ids=ids.tolist(),agreement=[1,1,1])))
    calls=[]
    def managed(config,module,source_file):
        calls.append(dict(config))
        return stages._run_direct_stage(config['workflow_stage'],config)
    def existing_solver(config,stage,*,data,stats,return_solution):
        assert return_solution and 'tensor_cache' not in config
        if stage=='normal_sign':
            saved=dict(ids=ids,broken=data['keep']<=.5,normal_sign=np.array([-1,1,1]),sign_probability=np.array([.1,.9,.8]))
        else:
            saved=dict(ids=ids,labels=np.array([0,1,2]),probabilities=np.array([[.9,.05,.05],[.1,.8,.1],[.1,.1,.8]]))
        return dict(runs=[dict(converged=True)]),saved
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_wrapped_stages as wrapped
    monkeypatch.setattr(normal,'launch_tensor_solve',managed)
    monkeypatch.setattr(wrapped,'solve_stage',existing_solver)
    def forbidden(*args,**kwargs):raise AssertionError('Direct stages must not hand off solver tensors through files')
    monkeypatch.setattr(np,'load',forbidden);monkeypatch.setattr(np,'savez',forbidden)
    hvb=tmp_path/'hvb.vcobj';sign=tmp_path/'sign.vcobj';cache=tmp_path/'shared-cache'
    first=stages.run_stage('hvb',dict(database=str(graph),result_store=str(hvb),
        features=str(feature),graph_cache=str(cache)))
    second=stages.run_stage('sign',dict(database=str(hvb),result_store=str(sign),
        priors=str(priors),graph_cache=str(cache)))
    assert second['input_run_uuid']==first['run_uuid']
    classifications=stages._fiber_rows(sign,first['run_uuid'])
    signs=stages._fiber_rows(sign,second['run_uuid'])
    assert [classifications[fid]['enabled'] for fid in ids]==[1,1,0]
    assert all(signs[fid]['enabled'] is None and signs[fid]['family'] is None for fid in ids)
    with CollectionReader(str(sign)) as store:
        assert len(store.objects('fiber'))==3
        assert len(store.fiber_constraints()['edge_uuid'])==1
        definition=next(t for t in store.extension_tables() if t['name']==stages.PREFIX+'inputs')
        columns=[c['name'] for c in definition['columns']]
        inputs=[dict(zip(columns,row)) for row in store.extension_rows(definition['name'])]
        assert any(row['input_run_uuid']==uuid.UUID(first['run_uuid']).bytes for row in inputs)
    assert stages._base_database(dict(database=str(sign)),source)==Path(graph).resolve()
    assert stages._base_database(dict(database=str(sign)))==Path(graph).resolve()
    assert calls[1]['workflow_stage']=='sign' and not calls[1]['gpu_manager']['exclusive']
    assert not cache.exists()
    with pytest.raises(FileExistsError):
        stages.run_stage('hvb',dict(database=str(graph),result_store=str(hvb)))


def test_ambiguous_run_requires_selection(tmp_path):
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_graph_store import write_results
    graph=measured_store(tmp_path,[(0,1,2,False,.5)])
    first=tmp_path/'first.vcobj';second=tmp_path/'second.vcobj'
    run1=write_results(first,[graph],'classification','test',{'fibers':[]})
    run2=write_results(second,[first],'classification','test',{'fibers':[]})
    with pytest.raises(ValueError,match='specify input_run_uuid'):
        stages._selected_run(second,'classification')
    assert stages._selected_run(second,'classification',run1)==run1
    assert stages._selected_run(second,'classification',run2)==run2


def test_axial_only_request_cannot_reorient_or_overwrite_normal_priors(tmp_path,monkeypatch):
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_hv_stage as hv
    source=measured_store(tmp_path,[(0,10,20,False,.5)])
    ids=np.asarray([uid(10),uid(20)]);cache=tmp_path/'graph-cache'
    monkeypatch.setattr(stages.FiberGraph,'cached',lambda *args:pytest.fail('Filesystem graph tensor caches are forbidden'))
    prior=tmp_path/'normal-priors.json';prior.write_text('existing normal priors')
    native=dict(orient_normals=True,database='original-graph.vcobj',normal_prior_output=str(prior),
                orient_database='original-orientation.vcobj',normal_curves_output='original-curves.json')
    def features(config,*,graph):
        generated=config['native_config']
        assert config['database']==str(source)
        assert np.array_equal(graph.ids,ids)
        assert generated==native
        assert 'axial_prior_output' not in generated
        return hv.AxialEvidence(graph.source,ids,np.array([-1,1]),'test')
    def solver(config,stage,**kwargs):
        return {},dict(ids=ids,labels=np.array([0,1]),probabilities=np.array([[1.,0.,0.],[0.,1.,0.]]))
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_wrapped_stages as wrapped
    monkeypatch.setattr(hv,'axial_evidence',features)
    monkeypatch.setattr(hv,'fibers_digest',lambda *a,**k:pytest.fail('Native evidence must not audit every fiber'))
    monkeypatch.setattr(wrapped,'solve_stage',solver)
    monkeypatch.setattr(normal,'launch_tensor_solve',lambda config,*args:stages._run_direct_stage(config['workflow_stage'],config))
    stages.run_stage('hvb',dict(database=str(source),result_store=str(tmp_path/'hvb.vcobj'),
        graph_cache=str(cache),native_config=native))
    assert native['orient_normals'] is True and native['normal_prior_output']==str(prior)
    assert prior.read_text()=='existing normal priors'


def test_generated_axial_columns_own_their_data_without_a_feature_file(tmp_path,monkeypatch):
    """Use the actual native axial kernel, then close/delete its source handles."""
    import gc
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_hv_stage as hv
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_graph_store import FiberGraph
    source=measured_store(tmp_path,[(0,10,20,False,.5)],curves={
        10:[[10.,10.,5.],[30.,10.,5.]],20:[[10.,10.,0.],[10.,10.,60.]]})
    graph=FiberGraph(source)
    axis=tmp_path/'axis.json'
    axis.write_text(json.dumps(dict(metadata=dict(volume=uid(999)+'-scan',volume_width=100,
        volume_height=100,volume_slices=80,voxelsize_um=2.),control_points=[
            dict(x=0.,y=0.,z=0.),dict(x=0.,y=0.,z=79.)])))
    native=dict(umbilicus=str(axis),axial_sample_spacing=16.,orient_normals=True,
                normal_prior_output=str(tmp_path/'must-not-create.json'))
    def forbidden(*args,**kwargs):raise AssertionError('Axial preparation must not stage numeric arrays')
    monkeypatch.setattr(np,'load',forbidden);monkeypatch.setattr(np,'save',forbidden)
    monkeypatch.setattr(np,'savez',forbidden);monkeypatch.setattr(np.lib.format,'open_memmap',forbidden)
    monkeypatch.setattr(hv,'fibers_digest',forbidden)
    result=hv.axial_evidence(dict(database=str(source),native_config=native,workers=2,output_root=str(tmp_path/'no-output')),
                             graph=graph)
    assert result.source==graph.source
    score=result.agreement;expected=score.copy()
    del result,graph
    gc.collect()
    np.testing.assert_array_equal(score,expected)
    np.testing.assert_array_equal(score,[-1.,1.])
    assert score.dtype==np.float32 and not score.flags.writeable
    assert not (tmp_path/'must-not-create.json').exists() and not (tmp_path/'no-output').exists()
    assert native['orient_normals'] is True


@pytest.mark.cuda
@pytest.mark.skipif(not os.environ.get('VILLA_GPU_MANAGER_GRANT'),reason='CUDA stage test requires a managed GPU grant')
def test_actual_cuda_hvb_and_sign_write_native_result_stores_without_tensor_files(tmp_path,monkeypatch):
    """Real preparation + wrapped CUDA solves + native exporters for both stages."""
    import torch
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable in the managed environment')
    source=measured_store(tmp_path,[(0,10,20,False,.5)])
    ids=np.asarray([uid(10),uid(20)])
    with CollectionReader(str(source)) as store:position=store.source
    _,fiber_pin=original_fiber_source(source)
    feature=tmp_path/'existing-features.json';prior=tmp_path/'existing-priors.json'
    feature.write_text(json.dumps(dict(format='fiber-axial-priors',version=1,source_ids=ids.tolist(),
        agreement=[-1.,1.],fibers_digest=fibers_digest(source),fiber_source=fiber_pin,umbilicus_sha256='test')))
    prior.write_text(json.dumps(dict(format='fiber-outward-priors',source_store=position,
        source_ids=ids.tolist(),agreement=[1.,1.])))
    # The test process already owns its manager grant. Replace only the process
    # launcher, not preparation, GPU optimization, or the native result writer.
    def managed(config,*args):return stages._run_direct_stage(config['workflow_stage'],config)
    def forbidden(*args,**kwargs):raise AssertionError('Production GPU stages must not use tensor-file handoffs')
    monkeypatch.setattr(normal,'launch_tensor_solve',managed)
    monkeypatch.setattr(np,'load',forbidden);monkeypatch.setattr(np,'save',forbidden)
    monkeypatch.setattr(np,'savez',forbidden);monkeypatch.setattr(np.lib.format,'open_memmap',forbidden)
    options=dict(workers=2,cost_modes=['uniform'],broken_penalties=[10.],entropy_weights=[0.],
        prior_weight=.1,irls_iterations=6,irls_cg_iterations=6,irls_log_every=2)
    hvb=tmp_path/'hvb.vcobj';signed=tmp_path/'sign.vcobj'
    first=stages.run_stage('hvb',dict(options,database=str(source),result_store=str(hvb),
        features=str(feature),output_root=str(tmp_path/'hvb-work')))
    second=stages.run_stage('sign',dict(options,database=str(hvb),result_store=str(signed),
        priors=str(prior),output_root=str(tmp_path/'sign-work')))
    classification=stages._fiber_columns(signed,first['run_uuid'],ids,required=('family','enabled'),exact=True)
    signs=stages._fiber_columns(signed,second['run_uuid'],ids,required=('normal_sign',),exact=True)
    np.testing.assert_array_equal(classification['family'],[0,1])
    np.testing.assert_array_equal(classification['enabled'],[1,1])
    assert np.isin(signs['normal_sign'],[-1,1]).all()
    assert not signs['enabled_valid'].any() and not signs['family_valid'].any()
    assert second['input_run_uuid']==first['run_uuid']
    for result in (first,second):
        run=result['solver_summary']['runs'][0]
        assert np.isfinite(run['final']['loss'])
        assert run['iterations']<=6
    for directory in (tmp_path/'hvb-work',tmp_path/'sign-work'):
        assert not list(directory.glob('*.npz')) and not list(directory.glob('*.npy'))
    with CollectionReader(str(signed)) as store:
        assert len(store.objects('fiber'))==2
        assert len(store.fiber_constraints()['edge_uuid'])==1
