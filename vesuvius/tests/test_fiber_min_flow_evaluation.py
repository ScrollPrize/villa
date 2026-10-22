from types import SimpleNamespace
import json

import numpy as np
import pytest

from vesuvius.neural_tracing.fiber_trace_3d.fiber_min_flow_evaluation import (
    large_reference, boundary_diagnostics, compare_solution, sign_counts,
)
from vesuvius.neural_tracing.fiber_trace_3d.fiber_min_flow_experiment import load_config, run


def reference_inputs():
    data = dict(source_id=np.array([10]))
    gt = dict(node_phase=np.zeros((1, 2)))
    solution = dict(winding=np.array([2]), component=np.array([0]))
    # First comparison crosses the seam; both fibers also cross on their way
    # to the endpoints. Second comparison has no crossings. Both should be +1.
    paths = SimpleNamespace(at=lambda source, arc: (arc, int(arc == 1)),
                            seam=SimpleNamespace(segment=lambda a, b: int(a == 1)))
    gt_paths = SimpleNamespace(at=lambda source, arc: (arc, 2*int(arc == 1)))
    reference = dict(fiber_offsets={'20': 3}, excluded_source_ids=[])
    events = [dict(event_id=i, gt_source=20, test_source=10, test_arc=i, gt_arc=i,
                   gt_piece=0, signed_distance=1., family_delta=0, parallel=True) for i in (0, 1)]
    return data, solution, paths, reference, events, gt, gt_paths


def test_reference_compensates_both_fibers_and_connector():
    args = reference_inputs()
    details = []
    result = large_reference(*args, details=details)
    assert result['all']['correct_percent'] == 100
    assert result['component_gauges']['0']['offset'] == 0
    assert [e['predicted_gt_minus_test'] for e in details] == [1, 1]
    assert details[1]['connector_crossings'] == 1
    assert details[1]['fiber_crossings'] == 1
    assert details[1]['gt_fiber_crossings'] == 2
    pieces = SimpleNamespace(locate=lambda source, arc: 0)
    assert large_reference(*args, pieces=pieces) == result


def test_empty_reference_coverage_and_unknown_signs():
    args = list(reference_inputs())
    args[3] = dict(fiber_offsets={}, excluded_source_ids=[20])
    result = large_reference(*args)
    assert result['all']['events'] == 0
    assert result['all']['l1'] is None
    assert result['unscored_events'] == 2
    json.dumps(result, allow_nan=False)
    signs = sign_counts(np.array([True, True, False]), np.array([0, -1, 1]),
                        np.array([8, 1, 0]), np.array([0, 0, 1]))
    assert signs == {'parallel': {'events': 1, 'violations': 1},
                     'perpendicular': {'events': 1, 'violations': 1}}


def test_component_alignment_reports_global_disagreement():
    args = list(reference_inputs())
    args[0] = dict(source_id=np.array([10, 11]))
    args[1] = dict(winding=np.array([2, 7]), component=np.array([0, 1]))
    args[4][1]['test_source'] = 11
    result = large_reference(*args)
    assert result['all']['l1'] == 0
    assert result['single_global_gauge']['metrics']['all']['l1'] > 0
    assert result['component_gauges']['1']['offset'] == -5


def test_boundaries_do_not_count_arbitrary_component_gauges_as_breaks():
    p = SimpleNamespace(parent=np.array([0, 0, 0, 1]))
    s = dict(winding=np.array([0, 1, 100, -10]), component=np.array([0, 0, 1, 1]))
    assert boundary_diagnostics(p, s) == dict(boundaries=2, unidentifiable_boundaries=1,
        identifiable_nonzero_boundary_jumps=1, identifiable_max_jump=1)


def test_baseline_comparison_checks_topology_and_aligns_gauge():
    a = {k: np.array([0, 1]) for k in ('source_id','a','b','target','weight','one_sided')}
    a.update(winding=np.array([0, 1]), component=np.array([0, 0]))
    b = dict(a, winding=np.array([9, 10]), component=np.array([4, 4]))
    assert compare_solution(a, b)['aligned_label_changes'] == 0
    assert compare_solution(a, dict(b, a=np.array([1, 0])))['comparable'] is False
    assert compare_solution(a, dict(b, component=np.array([0, 1])))['comparable'] is False


def test_config_resolves_declaring_file_paths_and_rejects_name_collisions(tmp_path):
    nested = tmp_path/'nested'
    nested.mkdir()
    (nested/'dataset.json').write_text(json.dumps(dict(afv='fibers.afv', umbilicus='axis.json', tensor_cache='cache.npz')))
    config = dict(output_root='results', cases=[dict(name='a', dataset='nested/dataset.json')])
    path = tmp_path/'run.json'
    path.write_text(json.dumps(config))
    resolved, dependencies = load_config(path)
    assert resolved['cases'][0]['dataset']['afv'] == str(nested/'fibers.afv')
    assert str(nested/'dataset.json') in dependencies
    config['cases'] *= 2
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='unique'):
        load_config(path)
    config['cases'] = [dict(name='../bad', dataset={})]
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='safe'):
        load_config(path)


def test_runner_without_gt_or_baseline_and_split_sign_direction(tmp_path, monkeypatch):
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_min_flow_experiment as module
    d = dict(source_id=np.array([10,20]), family=np.array([0,0]), i=np.array([0]), j=np.array([1]),
             sign=np.array([-1]), parallel=np.array([True]), links=np.empty((0,4)),
             node_ds=np.array([[0,10],[0,10]]), start=np.zeros(2), end=np.full(2,10.),
             ds_a=np.array([5]), ds_b=np.array([5]))
    np.savez(tmp_path/'cache.npz', **d)
    for name in ('fibers.afv','axis.json'):
        (tmp_path/name).write_text('unused synthetic geometry')
    monkeypatch.setattr(module, 'Paths', lambda *args: None)
    def graph(*args):
        return (dict(a=np.array([1]), b=np.array([0]), target=np.array([1]),
                     weight=np.array([1.]), one_sided=np.array([True])),
                dict(valid=np.array([True]), fiber_a=np.array([0]), fiber_b=np.array([0]),
                     connector=np.array([0]), flow_event_index=np.array([0])))
    monkeypatch.setattr(module, 'graph', graph)
    case = dict(name='test', dataset=dict(afv='fibers.afv', umbilicus='axis.json', tensor_cache='cache.npz'),
                sign_only=True, split=dict(minimum_length_vx=0))
    path=tmp_path/'config.json'
    path.write_text(json.dumps(dict(output_root='output', cases=[case])))
    result = run(path)['test']
    assert 'reference' not in result and 'baseline_comparison' not in result
    assert result['certificate']['optimal']
    with np.load(tmp_path/'output/test/solution.npz') as solution:
        assert solution['a'].tolist() == [1]
        assert solution['b'].tolist() == [0]
        assert solution['winding'][0]-solution['winding'][1] >= 1
    with pytest.raises(ValueError, match='empty'):
        run(path)


def test_reference_rejects_contradictory_or_fractional_labels():
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_min_flow_evaluation import reference_offsets
    with pytest.raises(ValueError, match='Excluded'):
        reference_offsets(dict(fiber_offsets={'1': 2}, excluded_source_ids=[1]))
    with pytest.raises(ValueError, match='integers'):
        reference_offsets(dict(fiber_offsets={'1': .5}))
