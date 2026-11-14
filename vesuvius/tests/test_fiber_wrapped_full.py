"""Full-source benchmark routing: normal axes never replace winding density."""
from vesuvius.neural_tracing.fiber_trace_3d import fiber_wrapped_full as full
import pytest


def _supervisor_identity(markers):
    import os
    import multiprocessing
    import time
    from pathlib import Path
    markers = Path(markers)
    (markers / str(os.getpid())).touch()
    deadline = time.monotonic() + 20
    while len(list(markers.iterdir())) < 2:
        if time.monotonic() > deadline:
            raise RuntimeError('Second independent supervisor did not start')
        time.sleep(.01)
    return os.getpid(), multiprocessing.get_start_method()


def test_managed_solvers_have_independent_spawned_process_trees(tmp_path):
    import os
    with full.managed_solver_pool() as pool:
        jobs = [pool.submit(_supervisor_identity, str(tmp_path)) for _ in range(2)]
        identities = [job.result(timeout=30) for job in jobs]
    assert len({pid for pid, _ in identities}) == 2
    assert all(pid != os.getpid() and method == 'spawn' for pid, method in identities)


def config(tmp_path):
    axis=tmp_path/'umbilicus.json';axis.write_text('[]')
    return dict(output_root=str(tmp_path),source='/inputs/fibers.afv',coordinate_space='sample/volume',
        umbilicus=str(axis),
        lasagna=dict(root='s3://bucket/published',manifest='normal.lasagna.json'),
        compound=dict(root='/inputs/prediction.zarr',provenance='/inputs/inference.json'))


def test_normal_sources_share_explicit_published_winding_source(tmp_path):
    c=config(tmp_path)
    baseline=full.source_settings(c,'lasagna');compound=full.source_settings(c,'compound')
    assert baseline['normal_manifest']==compound['winding_manifest']
    assert baseline['normal_provenance']==compound['winding_provenance']
    assert compound['normal_manifest']=='/inputs/prediction.zarr'
    assert compound['normal_source']=='fiber' and baseline['normal_source']=='lasagna'
    with pytest.raises(ValueError,match='Unknown normal source'):
        full.source_settings(c,'fallback')


def test_extraction_is_full_volume_same_geometry_separate_caches(tmp_path):
    c=config(tmp_path)
    a=full.extraction_settings(c,'lasagna');b=full.extraction_settings(c,'compound')
    assert a['fibers']==b['fibers']==str(tmp_path/'fibers.vcobj')
    assert a['full_volume'] is b['full_volume'] is True
    assert a['remote_cache']!=b['remote_cache']
    assert a['database']!=b['database']
    assert a['repetitions']==b['repetitions']==1
    assert 'minimum_xyz' not in a and 'fiber_ids' not in b
    for settings in (a,b):
        assert settings['orient_normals'] is True
        assert settings['normal_cache_mode']=='auto' and settings['normal_cache_fraction']==0.5
        assert settings['database'].endswith('/oriented.vcobj')
        assert settings['normal_prior_output'].endswith('/outward_priors.json')
        assert settings['normal_sample_spacing']==16.
        assert settings['chunk_size_base_voxels']==2048


def test_extraction_preserves_explicit_chunk_size(tmp_path):
    c=config(tmp_path);c['extraction']={'chunk_size_base_voxels':1024}
    assert full.extraction_settings(c,'compound')['chunk_size_base_voxels']==1024


def test_separate_orientation_and_prior_actions_do_not_resample(tmp_path):
    c=config(tmp_path)
    for action in ('orient','priors'):
        with pytest.raises(ValueError,match='included in extract'):
            full.run(dict(c,actions=[action]))


def test_complete_does_not_repeat_normal_sampling(tmp_path,monkeypatch):
    c=config(tmp_path);c.update(variants=['compound'],actions=['complete'])
    monkeypatch.setattr(full,'await_extraction',lambda *args:None)
    calls=[]
    run=full.run
    def capture(config):calls.append(config['actions'])
    monkeypatch.setattr(full,'run',capture)
    run(c)
    assert calls==[['solve','report']]


def test_import_is_whole_source_and_preserves_coordinates(tmp_path):
    c=config(tmp_path)
    result=full.import_settings(c,tmp_path/'fibers',tmp_path/'report')
    assert result['volume_id']=='volume' and result['scale_factor']==1
    assert result['fiber_limit']==0 and result['verify_readback'] is False
    assert full.import_settings(c,tmp_path/'pilot',tmp_path/'report',20)['fiber_limit']==20


def test_two_solver_pipelines_use_their_own_sign_exclusions(tmp_path):
    c=config(tmp_path)
    c.update(solver_parameters=dict(steps=10000, broken_penalties=[.1]),
             solver_devices=dict(adam='2', wrapped_irls='3'),
             gpu_manager=dict(exclusive=False, vram_bytes_per_gpu=24*1024**3))
    for variant in ('lasagna', 'compound'):
        adam=full.solver_settings(c,variant,'adam','classification')
        irls=full.solver_settings(c,variant,'wrapped_irls','classification')
        assert adam['stage1']!=irls['stage1']
        assert adam['database']==irls['database']
        assert adam['steps']==irls['steps']==10000
        assert adam['entropy_weights']==[.01] and irls['entropy_weights']==[0.]
        assert adam['devices']=='2' and irls['devices']=='3'
        assert not adam['gpu_manager']['exclusive']
        a=full.solver_settings(c,variant,'adam','normal_sign')
        b=full.solver_settings(c,variant,'wrapped_irls','normal_sign')
        assert a['tensor_cache']==b['tensor_cache']


def test_native_failure_preserves_error_tail_and_independent_log(tmp_path,monkeypatch):
    import json
    import subprocess
    import sys
    c=dict(output_root=str(tmp_path/'run'),native_log_mirror=str(tmp_path/'mirror'))
    monkeypatch.setattr(full,'resolve_native_binary',lambda *args:sys.executable)
    popen=subprocess.Popen
    monkeypatch.setattr(full.subprocess,'Popen',lambda args,**kw:popen(
        [sys.executable,'-c','import sys; print("native failure",file=sys.stderr); sys.exit(1)'],**kw))
    with pytest.raises(RuntimeError,match='Native step failed'):
        full.native_step(c,'failed','unused',{})
    result=json.loads((tmp_path/'run/failed.run.json').read_text())
    assert result['returncode']==1 and 'native failure' in result['output_tail']
    assert (tmp_path/'mirror/failed.log').read_text()=='native failure\n'


def test_await_extraction_rejects_failed_dependency(tmp_path):
    import json
    c=dict(output_root=str(tmp_path),extraction_attempt_suffix='_test')
    (tmp_path/'lasagna_extract_test.run.json').write_text(json.dumps(dict(returncode=1)))
    with pytest.raises(RuntimeError,match='Extraction failed'):
        full.await_extraction(c,'lasagna')
