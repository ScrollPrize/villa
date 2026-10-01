from pathlib import Path
import pytest
from reviewed_fiber_evaluation import capture_cohort, validate_result, validate_workload


def test_raw_unfiltered_selection():
    from reviewed_fiber_evaluation import validate_raw_unfiltered
    saved = dict(metric_name='test/loss_total')
    evaluation = dict(trace_args=['--beam-width','8'])
    validate_raw_unfiltered(saved,evaluation)
    with pytest.raises(ValueError):
        validate_raw_unfiltered(dict(saved,snapshot_kind='ema'),evaluation)
    with pytest.raises(ValueError):
        validate_raw_unfiltered(dict(metric_name='test_ema_0p999/loss_total'),evaluation)
    with pytest.raises(ValueError):
        validate_raw_unfiltered(saved,dict(trace_args=['--replay-reverse-interval-base','256']))


def test_ema99_selection():
    from reviewed_fiber_evaluation import validate_ema99
    validate_ema99(dict(snapshot_kind='ema',metric_name='test_ema_0p99/loss_total'))
    for saved in (dict(metric_name='test/loss_total'),
                  dict(snapshot_kind='ema',metric_name='test_ema_0p99/loss_total',ema_metadata={'decay':.999}),
                  dict(snapshot_kind='ema',metric_name='test_ema_0p999/loss_total')):
        with pytest.raises(ValueError):
            validate_ema99(saved)


def test_halfsigma_namespace_length_and_uniqueness():
    from reviewed_fiber_evaluation import halfsigma_namespace
    names = [halfsigma_namespace(m) for m in ('halfsigma_raw','halfsigma_ema99','halfsigma_current_raw_v1','halfsigma_current_raw_v2','halfsigma_current_raw_v3')]
    assert len(set(names)) == 5
    assert all(0<len(n)<=64 and '/' not in n for n in names)


def test_completed_result_must_cover_all_cases():
    result = dict(completed=True, cases=2, evaluated_cases=2, route_state_limit_cases=0)
    validate_result(result)
    with pytest.raises(RuntimeError):
        validate_result(dict(result, evaluated_cases=1))
    with pytest.raises(RuntimeError):
        validate_result(dict(result, route_state_limit_cases=1))
    with pytest.raises(RuntimeError):
        validate_result(dict(result, infrastructure_failures=1))


def test_workload_is_bound_to_cohort(tmp_path):
    cohort, output = tmp_path / 'cohort', tmp_path / 'run'
    manifest = {'files': [{'source': 'PHerc0139'}]}
    config = dict(split_root=str(cohort), output_root=str(output), split='test',
                  sources=[{'id': 'PHerc0139', 'expected_test_count': 1}])
    validate_workload(config, cohort, output, manifest)
    with pytest.raises(ValueError):
        validate_workload(dict(config, split_root='other'), cohort, output, manifest)


@pytest.mark.parametrize('version', [3, 4])
def test_editor_hints_do_not_change_geometry(version):
    from vc3d_fiber_format import parse_vc3d_fiber_format
    document = dict(type='vc3d_fiber', version=version, optimization_mode='native_fiber_trace3d',
        line_points=[[1, 2, 3], [4, 5, 6]], control_points=[dict(position=[1, 2, 3],
        direction=[1, 0, 0], display_normal=[0, 0, 1], display_normal_source='manual')])
    parsed = parse_vc3d_fiber_format(document)
    assert parsed.line_points_xyz == ((1, 2, 3), (4, 5, 6))
    assert parsed.control_points_xyz == ((1, 2, 3),)
    document['control_points'][0]['direction'] = [1, 2]
    with pytest.raises(ValueError):
        parse_vc3d_fiber_format(document)


def test_capture_preserves_bytes_and_rejects_changes(tmp_path):
    source = tmp_path / 'input'
    dest = tmp_path / 'frozen'
    (source / 'PHerc0139').mkdir(parents=True)
    path = source / 'PHerc0139/a.json'
    path.write_text('{"control_points": [1, 2]}\n')
    first = capture_cohort(source, dest)
    assert capture_cohort(source, dest) == first
    assert (dest / 'test/PHerc0139/a.json').read_bytes() == path.read_bytes()
    path.write_text('{}')
    with pytest.raises(ValueError, match='cohort changed'):
        capture_cohort(source, dest)


def test_empty_cohort_rejected(tmp_path):
    with pytest.raises(ValueError, match='empty'):
        capture_cohort(tmp_path, tmp_path / 'output')


def test_truncated_input_not_frozen(tmp_path):
    source = tmp_path / 'input'
    dest = tmp_path / 'frozen'
    (source / 'PHerc0139').mkdir(parents=True)
    (source / 'PHerc0139/a.json').write_text('{"unfinished":')
    with pytest.raises(ValueError, match='invalid reviewed annotation'):
        capture_cohort(source, dest)
    assert not dest.exists()
