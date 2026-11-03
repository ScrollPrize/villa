from vesuvius.neural_tracing.fiber_trace_3d.fiber_uv_weight_search import coordinate_search, configuration_key


def test_restarts_coordinates_and_replays_cached_results():
    # B only helps after A; A's second improvement only helps after B.
    costs = {(1, 1): 5, (2, 1): 4, (1, 2): 6, (2, 2): 3}
    cache = {}
    calls = []
    def evaluate(v):
        key = configuration_key(v)
        if key not in cache:
            calls.append(key)
            cache[key] = dict(ref_l1=costs[v['a'], v['b']])
        return cache[key]
    grid = [dict(b=[1, 2], a=[1, 2])]
    values, best = coordinate_search(dict(a=1, b=1), evaluate, grid)
    assert values == dict(a=2, b=2) and best['ref_l1'] == 3
    before = len(calls)
    assert coordinate_search(dict(a=1, b=1), evaluate, grid) == (values, best)
    assert len(calls) == before == 4


def test_tolerance_and_fine_grid():
    accepted = []
    def evaluate(v):
        return dict(ref_l1={1: 3., 2: 2.99999, 3: 2.}[v['a']])
    values, _ = coordinate_search(dict(a=1), evaluate,
        [dict(a=[1, 2]), dict(a=[1, 2, 3])], accepted=lambda *x: accepted.append(x))
    assert values == dict(a=3) and len(accepted) == 1


def test_key_independent_of_order():
    assert configuration_key(dict(a=1., b=2.)) == configuration_key(dict(b=2., a=1.))


def test_selected_retry_uses_original_lane_and_checks_exact_input(tmp_path):
    import json
    import hashlib
    import pytest
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_uv_weight_search import selected_retry_device, validate_selected_retry
    assert selected_retry_device(tmp_path,['0','1']) is None
    (tmp_path/'eval').mkdir()
    child=dict(devices='1',lr=.1)
    settings=dict(devices='1',base_config=str(tmp_path/'base.json'))
    payload=json.dumps({'fiber_implicit_winding':child},indent=2,sort_keys=True)+'\n'
    (tmp_path/'eval/inputs.json').write_text(json.dumps(dict(settings=settings,
        inputs={str(tmp_path/'base.json'):hashlib.sha256(payload.encode()).hexdigest()})))
    # An earlier failed retry overwrote the configs with a different lane.
    (tmp_path/'base.json').write_text(json.dumps({'fiber_implicit_winding':dict(child,devices='0')}))
    assert selected_retry_device(tmp_path,['0','1'])=='1'
    validate_selected_retry(tmp_path,child,settings)
    with pytest.raises(ValueError,match='solver configuration'):
        validate_selected_retry(tmp_path,dict(child,lr=.2),settings)
    with pytest.raises(ValueError,match='evaluation settings'):
        validate_selected_retry(tmp_path,child,dict(settings,devices='0'))
    with pytest.raises(ValueError,match='not in search lanes'):
        selected_retry_device(tmp_path,['0'])


def test_parallel_prefill_preserves_serial_acceptance_and_final_neighbors():
    grid=[dict(a=[.25,1,4],b=[.25,1,4]),dict(a=[.25,.5,1,2,4],b=[.25,.5,1,2,4])]
    def cost(v): return dict(ref_l1=(v['a']-v['b'])**2+(v['b']-2)**2)
    serial=[];parallel=[];cache={};batches=[]
    expected=coordinate_search(dict(a=1,b=1),cost,grid,accepted=lambda *x:serial.append(x[:2]))
    def evaluate(v):
        key=configuration_key(v)
        if key not in cache:cache[key]=cost(v)
        return cache[key]
    def prefill(values):
        batches.append(values)
        for v in reversed(values):evaluate(v)
    actual=coordinate_search(dict(a=1,b=1),evaluate,grid,evaluate_many=prefill,
                             accepted=lambda *x:parallel.append(x[:2]))
    assert expected==actual and serial==parallel and batches
    current,best=actual
    for name,values in grid[-1].items():
        for value in values:
            candidate=dict(current,**{name:value})
            assert configuration_key(candidate) in cache
            assert cost(candidate)['ref_l1']>=best['ref_l1']-1e-4


def test_secondary_metric_requires_joint_improvement():
    def evaluate(v):
        return dict(ref_l1={1:3,2:2,3:2.5,4:2.6}[v['a']],fine={1:2,2:3,3:1,4:.5}[v['a']])
    value,best=coordinate_search(dict(a=1),evaluate,[dict(a=[1,2,3,4])],secondary_metric='fine')
    assert value==dict(a=3) and best['fine']==1


def test_density_controller_resume_and_certificate(tmp_path,monkeypatch):
    import json
    import subprocess
    import numpy as np
    import pytest
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_uv_weight_search as search
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_gt_constraint_score as gt
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_implicit_match_report as report
    for name in ('solution.npz','tensor.npz','samples.npz'):
        np.savez(tmp_path/name,state=np.zeros((1,4)),widths=np.ones(2))
    (tmp_path/'config.json').write_text(json.dumps(dict(soft_samples=str(tmp_path/'samples.npz'))))
    base=dict(initial_solution=str(tmp_path/'solution.npz'),tensor_cache=str(tmp_path/'tensor.npz'),
              distance_weight=1.,steps=2,gpu_manager=dict(exclusive=False),winding_sigma=.5)
    path=tmp_path/'base.json';path.write_text(json.dumps(dict(fiber_implicit_winding=base)))
    cfg=dict(base_config=str(path),output_root=str(tmp_path/'search'),weights=['distance_weight'],
             devices=[2,3],gt_score={})
    class Scorer:
        hashes={}
        def __init__(self,*args):pass
        def score(self,solution,config):
            return dict(gt_enabled_l1=abs(config['distance_weight']-2),gt_enabled_correct_percent=100.)
    monkeypatch.setattr(gt,'FrozenGTScorer',Scorer)
    monkeypatch.setattr(report,'run',lambda settings:[])
    calls=[]
    def fake_run(command,**kwargs):
        child=json.loads(search.Path(command[-1]).read_text())['fiber_implicit_winding']
        root=search.Path(child['output_root']);calls.append(child['distance_weight'])
        np.savez(root/'solution.npz',state=np.zeros((1,4)),widths=np.ones(2))
        (root/'summary.json').write_text(json.dumps(dict(config=child,
            final=dict(iteration=2,loss=1.),solve_wall_seconds=1.)))
    monkeypatch.setattr(subprocess,'run',fake_run)
    first=search.run_density(cfg);count=len(calls)
    assert first['weights']==dict(distance_weight=2.) and len(first['certificate'])==5
    assert search.run_density(cfg)==first and len(calls)==count
    trial=next((tmp_path/'search/trials').glob('*/result.json'))
    row=json.loads(trial.read_text());row['weights']['distance_weight']=99
    trial.write_text(json.dumps(row))
    with pytest.raises(ValueError,match='key mismatch'):search.run_density(cfg)


def test_gt_objective_not_spiral_and_invalid_trial():
    def evaluate(v):
        return dict(ref_l1=-v['a'],gt_enabled_l1={1:2.,2:None,3:1.,4:3.}[v['a']])
    values,best=coordinate_search(dict(a=1),evaluate,[dict(a=[1,2,3,4])],metric='gt_enabled_l1')
    assert values==dict(a=3) and best['gt_enabled_l1']==1.


def test_disk_resume_and_manifest_guard(tmp_path, monkeypatch):
    import json
    import numpy as np
    import pytest
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_uv_layout as uv
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_uv_weight_search as search
    data = dict(source_id=np.array(['00000000-0000-4000-8000-000000000001']), start=np.array([0.]),
                end=np.array([100.]), family=np.array([0]))
    tensor = tmp_path / 'tensor.npz'
    np.savez(tensor, **data)
    tensor.with_suffix('.json').write_text('{}')
    meta = tmp_path / 'meta.json'
    meta.write_text('{}')
    projection = tmp_path / 'projection.npz'
    np.savez(projection, test=np.array([1]))
    cfg = dict(output_root=str(tmp_path / 'search'), gpu_manager=dict(exclusive=True),
        loss_weight_search=dict(weights=['distance_weight']), distance_weight=1.,
        tensor_cache=str(tensor), reference_projection=str(projection),
        reference_subset=str(meta), reference_widths=dict(meta=str(meta)), steps=10)
    calls = []
    def fake_solve(child):
        root = search.Path(child['output_root'])
        calls.append(child['distance_weight'])
        np.savez(root/'solution.npz', **data, state=np.zeros((1, 3)), widths=np.array([200.]))
        final = dict(optimizer_updates=10, loss=1., reference=dict(
            mean_absolute_winding_error=abs(child['distance_weight']-2.),
            matched_fibers=1, within_two_windings_percent=100.,
            widths=dict(mean_normalized_factor=1.), fitted_winding_q10_q90=[1, 2],
            reference_winding_q10_q90=[1, 2]), retained_arc_percent=100., sign_violation_percent=0.)
        (root/'summary.json').write_text(json.dumps(dict(final=final, solve_wall_seconds=1.)))
    monkeypatch.setattr(uv, 'solve', fake_solve)
    result = search.run(cfg)
    assert result['weights'] == dict(distance_weight=2.)
    count = len(calls)
    assert search.run(cfg) == result and len(calls) == count
    with pytest.raises(ValueError, match='configuration changed'):
        search.run(dict(cfg, steps=11))
    # A completed file alone is insufficient to certify a trial.
    saved = next((tmp_path/'search'/'trials').glob('*/summary.json'))
    summary = json.loads(saved.read_text())
    summary['final']['optimizer_updates'] = 9
    saved.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match='Incomplete trial'):
        search.run(cfg)
