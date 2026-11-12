"""Stage orchestration uses existing solvers and immutable typed result stores."""
import json
from pathlib import Path
import uuid

import numpy as np
import pytest
from vc_object_store import CollectionReader

from .fiber_graph_fixture import measured_store,uid
from vesuvius.neural_tracing.fiber_trace_3d import fiber_wgraph_stages as stages
from vesuvius.neural_tracing.fiber_trace_3d import fiber_normal_sign as normal
from vesuvius.neural_tracing.tracers.shared.annotations.fiber_store import fibers_digest


def test_hvb_then_sign_preserves_classification_and_shares_graph(tmp_path,monkeypatch):
    graph=measured_store(tmp_path,[(0,10,20,False,.5)],isolated_ids=[30])
    ids=np.asarray(list(map(uid,[10,20,30])))
    with CollectionReader(str(graph)) as store:source=store.source
    feature=tmp_path/'features.json'
    feature.write_text(json.dumps(dict(format='fiber-axial-priors',version=1,source_ids=ids.tolist(),
        agreement=[-1,1,0],fibers_digest=fibers_digest(graph),umbilicus_sha256='test')))
    priors=tmp_path/'priors.json'
    priors.write_text(json.dumps(dict(format='fiber-outward-priors',source_store=source,
                                      source_ids=ids.tolist(),agreement=[1,1,1])))
    calls=[]
    def existing_solver(config,module,source_file):
        calls.append(dict(config))
        output=Path(config['output_root'])
        if config.get('fixed_keep'):
            with np.load(config['tensor_cache']) as cache:keep=cache['keep']
            np.savez(output/'solution_0.1.npz',ids=ids,broken=keep<=.5,
                     normal_sign=np.array([-1,1,1]),sign_probability=np.array([.1,.9,.8]))
        else:
            np.savez(output/'solution_uniform_b0.1_e0.npz',ids=ids,labels=np.array([0,1,2]),
                     probabilities=np.array([[.9,.05,.05],[.1,.8,.1],[.1,.1,.8]]))
        return dict(runs=[dict(converged=True)])
    monkeypatch.setattr(normal,'launch_tensor_solve',existing_solver)
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
    assert calls[1]['fixed_keep'] and not calls[1]['gpu_manager']['exclusive']
    assert (cache/'manifest.json').exists()
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
    stages.FiberGraph.cached(source,cache)
    monkeypatch.setattr(stages.FiberGraph,'__init__',lambda *args:pytest.fail('Existing graph cache must be reused'))
    prior=tmp_path/'normal-priors.json';prior.write_text('existing normal priors')
    native=dict(orient_normals=True,database='original-graph.vcobj',normal_prior_output=str(prior),
                orient_database='original-orientation.vcobj',normal_curves_output='original-curves.json')
    def features(config):
        generated=config['native_config']
        assert generated['orient_normals'] is False and generated['axial_prior_only'] is True
        assert not any(key in generated for key in ('database','normal_prior_output','orient_database','normal_curves_output'))
        Path(generated['axial_prior_output']).write_text(json.dumps(dict(format='fiber-axial-priors',version=1,
            source_ids=ids.tolist(),agreement=[-1,1],fibers_digest=fibers_digest(source),umbilicus_sha256='test')))
    def solver(config,*args):
        np.savez(Path(config['output_root'])/'solution_uniform_b0.1_e0.npz',ids=ids,
                 labels=[0,1],probabilities=[[1.,0.,0.],[0.,1.,0.]])
        return {}
    monkeypatch.setattr(hv,'run_features',features)
    monkeypatch.setattr(normal,'launch_tensor_solve',solver)
    stages.run_stage('hvb',dict(database=str(source),result_store=str(tmp_path/'hvb.vcobj'),
        graph_cache=str(cache),native_config=native))
    assert native['orient_normals'] is True and native['normal_prior_output']==str(prior)
    assert prior.read_text()=='existing normal priors'
