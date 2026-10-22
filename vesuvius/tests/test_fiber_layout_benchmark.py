"""Portable workflow tests; never acquire a GPU or start optimization."""
import copy
import json
from pathlib import Path

import pytest

from vesuvius.neural_tracing.fiber_trace_3d import fiber_layout_benchmark as bench
from vesuvius.neural_tracing.fiber_trace_3d import managed_evaluation


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))
    return str(path)


@pytest.fixture
def configured(tmp_path):
    tensor = Path(write(tmp_path / 'tensors.json', {}))
    initial = Path(write(tmp_path / 'init/solution.npz', {}))
    sample = write(tmp_path / 'init/samples.npz', {})
    write(initial.parent / 'config.json', dict(tensor_sha256=bench.sha256_file(tensor), soft_samples=sample))
    gt = tmp_path / 'reference'
    write(gt / 'config.json', dict(tensor_cache=str(tensor)))
    write(gt / 'solution.npz', {})
    settings = dict(version=1, protocol=bench.PROTOCOL, reference_root='reference', arms={})
    for arm in bench.ARMS:
        config = dict(tensor_cache=str(tensor), initial_solution=str(initial),
                      estimate_fixed_length_vram=True, steps=3000, seed=17,
                      distance_weight=.25, devices='old', output_root='old',
                      gpu_manager=dict(exclusive=True))
        if arm == 'joint':
            config['force_keep_source_ids'] = [101]
        path = tmp_path / arm
        write(path / 'config.json', dict(fiber_implicit_winding=config))
        write(path / 'events.json', [])
        write(path / 'expected.json', {})
        historical_tensor = write(path / 'historical/tensors.npz', {})
        write(path / 'historical/config.json', dict(tensor_cache=historical_tensor))
        write(path / 'historical/solution.npz', {})
        write(path / 'historical/summary.json', {})
        settings['arms'][arm] = dict(solver_config=f'{arm}/config.json', events=f'{arm}/events.json',
                                    historical_solve=f'{arm}/historical', expected_report=f'{arm}/expected.json')
        if arm != 'free':
            settings['arms'][arm].update(gt_source_ids=[101], force_gt_enabled=arm == 'joint')
    runtime = dict(output_root='output', devices=dict(zip(bench.ARMS, ['0', '1', '2'])),
                   gpu_manager=dict(exclusive=False, cpu_cores=4, priority=100))
    config_path = Path(write(tmp_path / 'benchmark.json', settings))
    runtime_path = Path(write(tmp_path / 'runtime.json', runtime))
    return config_path, runtime_path


def test_prepare_preserves_science_and_separates_runtime(configured):
    settings, runtime = bench.load_settings(*configured)
    before = copy.deepcopy(settings)
    manifest = bench.prepare(settings, runtime)
    assert settings == before
    for arm in bench.ARMS:
        old = bench.read(Path(settings['arms'][arm]['solver_config']))['fiber_implicit_winding']
        new = manifest['configs'][arm]
        assert {k: v for k, v in old.items() if k not in ('devices', 'gpu_manager', 'output_root')} == {
            k: v for k, v in new.items() if k not in ('devices', 'gpu_manager', 'output_root')}
        assert new['gpu_manager']['exclusive'] is False
        assert new['devices'] == runtime['devices'][arm]
    assert bench.prepare(settings, runtime) == manifest
    Path(settings['arms']['free']['events']).write_text('[{}]')
    with pytest.raises(ValueError, match='changed'):
        bench.prepare(settings, runtime)


def test_prepare_checks_initializer_and_forcing(configured):
    settings, runtime = bench.load_settings(*configured)
    path = Path(settings['arms']['joint']['solver_config'])
    config = bench.read(path)
    config['fiber_implicit_winding']['force_keep_source_ids'] = []
    write(path, config)
    with pytest.raises(ValueError, match='forced GT policy'):
        bench.prepare(settings, runtime)
    config['fiber_implicit_winding']['force_keep_source_ids'] = [101]
    write(path, config)
    init = Path(config['fiber_implicit_winding']['initial_solution']).parent / 'config.json'
    meta = bench.read(init)
    meta['tensor_sha256'] = 'wrong'
    write(init, meta)
    with pytest.raises(ValueError, match='fingerprint'):
        bench.prepare(settings, runtime)


@pytest.mark.parametrize('name', ['config.json', 'solution.npz', 'summary.json', 'tensors.npz'])
def test_historical_inputs_are_immutable(configured, name):
    settings, runtime = bench.load_settings(*configured)
    historical = Path(settings['arms']['free']['historical_solve'])
    path = historical / name
    manifest = bench.prepare(settings, runtime)
    assert manifest['inputs'][str(path)] == bench.sha256_file(path)
    data = bench.read(path)
    data['changed'] = True
    write(path, data)
    with pytest.raises(ValueError, match='changed'):
        bench.prepare(settings, runtime)


@pytest.mark.parametrize('field,value', [('exclusive', True), ('exclusive', None)])
def test_no_manager_bypass(configured, field, value):
    path = configured[1]
    runtime = bench.read(path)
    runtime['gpu_manager'][field] = value
    write(path, runtime)
    with pytest.raises(ValueError, match='nonexclusive'):
        bench.load_settings(*configured)


def test_distinct_devices_and_protocol(configured):
    runtime = bench.read(configured[1])
    runtime['devices']['joint'] = '0'
    write(configured[1], runtime)
    with pytest.raises(ValueError, match='distinct'):
        bench.load_settings(*configured)
    config = bench.read(configured[0])
    config['protocol'] = 'different-reference'
    write(configured[0], config)
    with pytest.raises(ValueError, match='protocol'):
        bench.load_settings(*configured)


def test_launch_only_managed_module(monkeypatch, tmp_path):
    recorded = {}
    def execute(command, **kwargs):
        recorded.update(command=command, **kwargs)
    monkeypatch.setattr(bench.subprocess, 'run', execute)
    bench.launch_arm('free', tmp_path, 'GPU-aaa')
    assert recorded['command'][1:3] == ['-m', bench.SOLVER_MODULE]
    assert '--unmanaged' not in recorded['command']
    assert recorded['command'][-2:] == ['--devices', 'GPU-aaa']
    assert recorded['check'] is True
    assert str(Path(bench.__file__).resolve().parents[3]) in recorded['env']['PYTHONPATH']


def result(count=2, percent=50):
    return dict(signs={kind: dict(violations=count, events=10) for kind in ('parallel', 'perpendicular')},
                retention=dict(percent=percent, kept_vx=percent, total_vx=100),
                score={f'gt_{kind}': dict(l1=.3, correct_percent=90) for kind in ('parallel', 'perpendicular')},
                solution_sha256='hash', final=dict(loss=.4))


def test_ranking_and_expected():
    rows = {a: result() for a in bench.ARMS}
    assert bench.ranking(rows)['score'] == 12
    rows['selectable']['retention']['percent'] = 100
    assert bench.ranking(rows)['score'] == 6
    rows['selectable']['retention']['percent'] = 0
    assert bench.ranking(rows)['score'] is None
    expected = dict(result(), gt_enablement=[dict(length_vx=50, enabled_probability=.5),
                                            dict(length_vx=50, enabled_probability=.49)])
    bench.validate_expected(result(), expected)
    bad = result(count=3)
    with pytest.raises(ValueError, match='sign'):
        bench.validate_expected(bad, expected)


@pytest.mark.parametrize('change', ['cancel', 'missing', 'extra', 'diagnostic', 'mode'])
def test_validate_all_scientific_final_fields(change):
    actual = result()
    actual['final'].update(terms=dict(winding=.3, rejection=.1), sigma_mean=.5, sigma_mode='fixed')
    expected = copy.deepcopy(actual)
    expected['gt_enablement'] = [dict(length_vx=50, enabled_probability=.5),
                               dict(length_vx=50, enabled_probability=.49)]
    if change == 'cancel':
        actual['final']['terms'].update(winding=.35, rejection=.05)
    elif change == 'missing':
        del actual['final']['terms']['rejection']
    elif change == 'extra':
        actual['final']['terms']['new_loss'] = 0
    elif change == 'diagnostic':
        actual['final']['sigma_mean'] = .6
    else:
        actual['final']['sigma_mode'] = 'learned'
    with pytest.raises((AssertionError, ValueError), match='final'):
        bench.validate_expected(actual, expected)


def test_expected_allows_elapsed_time_changes():
    actual = result()
    expected = dict(result(), gt_enablement=[dict(length_vx=50, enabled_probability=.5),
                                            dict(length_vx=50, enabled_probability=.49)])
    actual['final']['seconds'] = 10
    expected['final']['seconds'] = 20
    bench.validate_expected(actual, expected)


def test_prepare_never_launches(monkeypatch, configured):
    settings, runtime = bench.load_settings(*configured)
    monkeypatch.setattr(bench, 'launch_arm', lambda *a: pytest.fail('GPU launch'))
    monkeypatch.setattr(bench, 'validate_reference', lambda *a: pytest.fail('scoring'))
    monkeypatch.setattr(bench, 'resolve_gpu_uuids', lambda *a: pytest.fail('GPU probe'))
    assert bench.run(settings, runtime, 'prepare')['prepared']


def test_workflow_all_arms_and_no_rerun(monkeypatch, configured):
    settings, runtime = bench.load_settings(*configured)
    launches = []
    monkeypatch.setattr(bench, 'validate_reference', lambda *a: {'validated': True})
    monkeypatch.setattr(bench, 'resolve_gpu_uuids', lambda *a: ['GPU-a', 'GPU-b', 'GPU-c'])
    def launch(arm, root, device):
        launches.append(arm)
        assert device == dict(zip(bench.ARMS, ['GPU-a', 'GPU-b', 'GPU-c']))[arm]
        (root / arm / 'solve').mkdir(parents=True)
    monkeypatch.setattr(bench, 'launch_arm', launch)
    monkeypatch.setattr(bench, 'score', lambda *a: result())
    monkeypatch.setattr(bench, 'write_report', lambda root, rows, previous: dict(ranking=bench.ranking(rows)))
    output = bench.run(settings, runtime)
    assert set(launches) == set(bench.ARMS)
    assert output['ranking']['score'] == 12
    assert bench.read(Path(runtime['output_root']) / 'status.json')['status'] == 'finished'
    assert bench.read(Path(runtime['output_root']) / 'resolved_devices.json') == dict(
        zip(bench.ARMS, ['GPU-a', 'GPU-b', 'GPU-c']))
    with pytest.raises(FileExistsError):
        bench.run(settings, runtime)
    assert len(launches) == 3


def test_workflow_reports_failure(monkeypatch, configured):
    settings, runtime = bench.load_settings(*configured)
    monkeypatch.setattr(bench, 'validate_reference', lambda *a: {})
    monkeypatch.setattr(bench, 'resolve_gpu_uuids', lambda *a: ['GPU-a', 'GPU-b', 'GPU-c'])
    def launch(*args):
        raise RuntimeError('managed child failed')
    monkeypatch.setattr(bench, 'launch_arm', launch)
    with pytest.raises(RuntimeError, match='managed child'):
        bench.run(settings, runtime)
    status = bench.read(Path(runtime['output_root']) / 'status.json')
    assert status['status'] == 'failed'
    assert 'managed child' in status['error']


@pytest.mark.parametrize('devices,visibility', [
    (['0', 'GPU-aa', '2'], None),
    (['GPU-a', 'GPU-aa', '2'], None),
    (['0', 'GPU-bb', '2'], 'GPU-bb,GPU-aa,GPU-cc'),
])
def test_alias_collision_rejected_before_launch(monkeypatch, configured, devices, visibility):
    settings, runtime = bench.load_settings(*configured)
    runtime['devices'] = dict(zip(bench.ARMS, devices))
    if visibility is None:
        monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    else:
        monkeypatch.setenv('CUDA_VISIBLE_DEVICES', visibility)
    monkeypatch.setattr(bench, 'resolve_gpu_uuids', lambda ids: managed_evaluation.resolve_gpu_uuids(
        ids, inventory_text='0, GPU-aa\n1, GPU-bb\n2, GPU-cc\n'))
    monkeypatch.setattr(bench, 'validate_reference', lambda *a: {})
    monkeypatch.setattr(bench, 'launch_arm', lambda *a: pytest.fail('GPU launch'))
    with pytest.raises(ValueError, match='distinct'):
        bench.run(settings, runtime)
    assert bench.read(Path(runtime['output_root']) / 'status.json')['status'] == 'failed'


def test_mixed_device_ids_follow_visibility(monkeypatch, configured):
    settings, runtime = bench.load_settings(*configured)
    runtime['devices'] = dict(zip(bench.ARMS, ['0', 'GPU-a', '2']))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-bb,GPU-aa,GPU-cc')
    monkeypatch.setattr(bench, 'resolve_gpu_uuids', lambda ids: managed_evaluation.resolve_gpu_uuids(
        ids, inventory_text='0, GPU-aa\n1, GPU-bb\n2, GPU-cc\n'))
    monkeypatch.setattr(bench, 'validate_reference', lambda *a: {})
    def launch(arm, root, device):
        assert device == dict(zip(bench.ARMS, ['GPU-bb', 'GPU-aa', 'GPU-cc']))[arm]
        (root / arm / 'solve').mkdir(parents=True)
    monkeypatch.setattr(bench, 'launch_arm', launch)
    monkeypatch.setattr(bench, 'score', lambda *a: result())
    monkeypatch.setattr(bench, 'write_report', lambda *a: dict(ranking={}))
    bench.run(settings, runtime)


@pytest.mark.parametrize('action', ['validate-reference', 'score'])
def test_cpu_actions_never_probe(monkeypatch, configured, action):
    settings, runtime = bench.load_settings(*configured)
    monkeypatch.setattr(bench, 'resolve_gpu_uuids', lambda *a: pytest.fail('GPU probe'))
    monkeypatch.setattr(bench, 'launch_arm', lambda *a: pytest.fail('GPU launch'))
    monkeypatch.setattr(bench, 'validate_reference', lambda *a: dict(ranking={}))
    monkeypatch.setattr(bench, 'score', lambda *a: result())
    monkeypatch.setattr(bench, 'write_report', lambda *a: dict(ranking={}))
    bench.run(settings, runtime, action)
