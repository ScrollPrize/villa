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


def test_disk_resume_and_manifest_guard(tmp_path, monkeypatch):
    import json
    import numpy as np
    import pytest
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_uv_layout as uv
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_uv_weight_search as search
    data = dict(source_id=np.array([1]), start=np.array([0.]),
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
