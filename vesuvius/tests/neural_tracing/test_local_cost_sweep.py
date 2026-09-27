import importlib.util
import pytest
from pathlib import Path

spec = importlib.util.spec_from_file_location('local_costs', Path(__file__).resolve().parents[2] / 'src/vesuvius/neural_tracing/fiber_trace_3d/local_cost_sweep.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_matrix_and_independent_switches():
    template = dict(reference_dir='refs', input_files=[], commands=[dict(binary='vc_fiber_trace_chunk', args=[
        'direction-ablation', 'trace', '--output', 'old', '--threads', '64', '--winding-fixed-orientation'])])
    points = module.matrix()
    assert len(points) == 24
    for point in points:
        cfg = module.trial_config(template, '/tmp/trials', point, 48)
        args = cfg['commands'][0]['args']
        assert '--winding-fixed-orientation' not in args
        assert ('--winding-continuity-cut-cost' in args) == (point['cut'] != 'hard')
        assert ('--local-no-parallel' in args) != point['parallel']
        assert args[args.index('--winding-defect-cost') + 1] == str(point['defect'])
    assert '--winding-fixed-orientation' in template['commands'][0]['args']
    template['commands'][0]['args'].remove('--winding-fixed-orientation')
    cfg = module.trial_config(template, '/tmp/trials', dict(parallel=True, defect=100, cut=1, fixed_orientation=True), 48)
    assert '--winding-fixed-orientation' in cfg['commands'][0]['args']


def test_distinct_event_audit(tmp_path):
    prefix = tmp_path / 'trial'
    Path(str(prefix) + '_winding_factors.csv').write_text(
        'local_event,perpendicular,parallel,piece_a,piece_b,arc_a,arc_b\n'
        '1,1,0,0,1,10,20\n1,1,0,0,2,10,40\n1,1,0,0,3,12,80\n0,1,0,4,5,0,0\n')
    assert module.audit_events(prefix) == dict(pieces_with_orthogonal_events=4, multiple_event_pieces=1, maximum_event_groups=2,
                                              multiple_orthogonal_factor_pieces=1, maximum_orthogonal_factors=3)


def test_distance_cost_uses_separate_flag_and_artifact_name():
    template = dict(reference_dir='refs', input_files=[], commands=[dict(args=[
        'direction-ablation', 'trace', '--output', 'old', '--threads', '64'])])
    cfg = module.trial_config(template, '/tmp/trials', dict(parallel=False, defect=10, cut=10, distance_cost=True), 48)
    assert cfg['commands'][0]['name'] == 'orthogonal-d10-c10-distance'
    args = cfg['commands'][0]['args']
    assert '--winding-continuity-step-cost' in args
    assert '--winding-continuity-cut-cost' not in args


def test_shift_metric_ignores_global_offset_and_counts_suffix_changes():
    ladder = [dict(reference=i / 2, winding=i / 2 + 3 + (i >= 13)) for i in range(26)]
    result = module.ladder_shift_metrics(ladder)
    assert result['ladder_shifts'] == 1
    assert abs(result['ladder_shift_percent'] - 100 * 25 / 26) < 1e-10
    assert module.ladder_shift_metrics([])['ladder_shift_percent'] is None


def test_certified_counts_only_allow_explicit_parallel_omission():
    module.validate_certified_counts(dict(certified=12, retained_certified=12, support_rejected=0))
    module.validate_certified_counts(dict(certified=12, retained_certified=10, parallel_omitted=2))
    with pytest.raises(ValueError):
        module.validate_certified_counts(dict(certified=12, retained_certified=11))
    with pytest.raises(ValueError):
        module.validate_certified_counts(dict(certified=12, retained_certified=12, support_rejected=1))


def test_legacy_control_does_not_enable_local_extraction():
    template = dict(reference_dir='refs', input_files=[], commands=[dict(args=[
        'direction-ablation', 'trace', '--output', 'old', '--threads', '64', '--local-constraint-events'])])
    cfg = module.trial_config(template, '/tmp/trials', dict(parallel=True, defect=100, cut='hard', fixed_orientation=True, local_events=False), 48)
    assert cfg['commands'][0]['name'].endswith('-legacy')
    assert '--local-constraint-events' not in cfg['commands'][0]['args']
