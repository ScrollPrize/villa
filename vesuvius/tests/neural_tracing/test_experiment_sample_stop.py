"""CPU-only controller checks: do not initialize GPUs or signal real tasks."""
import importlib.util
import json
from pathlib import Path
from unittest.mock import Mock

import pytest


@pytest.fixture
def runner():
    path = Path(__file__).resolve().parents[2] / 'scripts/fiber3d_experiment.py'
    spec = importlib.util.spec_from_file_location('experiment_stop_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_only_completed_test_lines(runner):
    assert runner.completed_test_samples('samples=601600 update=4700 loss_total=1\n') == 0
    assert runner.completed_test_samples('test_timing samples=601600 update=4700 total_seconds=5\n') == 601600


@pytest.mark.parametrize('state', ['dead', 'unknown'])
def test_never_signal_dead_or_unknown(runner, monkeypatch, state):
    import villa_gpu_manager.processes as processes
    monkeypatch.setattr(processes, 'process_state', lambda *a: state)
    kill = Mock()
    monkeypatch.setattr(runner.os, 'kill', kill)
    if state == 'unknown':
        with pytest.raises(RuntimeError, match='verify'):
            runner.stop_production_identity({'pid':123})
    else:
        runner.stop_production_identity({'pid':123})
    kill.assert_not_called()


def test_monitor_waits_for_ema_best_and_binds_launch(runner, monkeypatch, tmp_path):
    import torch
    import villa_gpu_manager.processes as processes
    launch = tmp_path / 'launch'
    launch.mkdir()
    run = tmp_path / 'run'
    identity = {'pid':123, 'start_us':456, 'boot_us':789}
    (launch / 'controller.json').write_text(json.dumps(identity))
    (launch / 'train.log').write_text(f'run_dir={run}\ntest_timing samples=601600 update=4700 total_seconds=5\n')
    control = tmp_path / 'control.json'
    control.write_text(json.dumps({'production_stop':{'launch':str(launch), 'samples':600000}}))
    monkeypatch.setattr(runner, 'CONTROL', control)
    monkeypatch.setattr(processes, 'process_state', lambda *a: 'alive')
    sleeps = []
    monkeypatch.setattr(runner.time, 'sleep', lambda seconds: sleeps.append(seconds))
    stopped = Mock()
    monkeypatch.setattr(runner, 'stop_production_identity', stopped)
    def load(path, **kwargs):
        assert kwargs['map_location'] == 'cpu' and kwargs['mmap']
        if path.name == 'current.pt':
            return dict(step=4700, samples_seen=601600, metric=2,
                        ema={'tracks':{'ema_0p99':{'best_metric':1}}})
        if path.name == 'best.pt':
            return {'metric':1}
        if path.name == 'current_ema_0p99.pt':
            return {'step':4700}
        return {'metric':1 if sleeps else 3}
    monkeypatch.setattr(torch, 'load', load)
    runner.production_watch()
    assert sleeps == [2]
    stopped.assert_called_once_with(identity)
    result = json.loads((launch / 'sample_stop_result.json').read_text())
    assert result['snapshot_samples'] == 601600


def test_campaign_failure_stops_owned_detached_training(runner, monkeypatch, tmp_path):
    initial = tmp_path / 'raw'
    initial.mkdir()
    (initial / 'selection.json').write_text(json.dumps({'run':'s1a'}))
    control = tmp_path / 'control.json'
    control.write_text(json.dumps({'comparison':{
        'output':str(tmp_path / 'campaign'), 's1a_raw_evaluation':str(initial),
        'stages':[{'name':'s1a', 'existing_run':'s1a', 'config':'s1.json', 'variants':[]},
                  {'name':'json', 'config':'json.json', 'variants':[]}]
    }}))
    monkeypatch.setattr(runner, 'CONTROL', control)
    monkeypatch.setattr(runner.subprocess, 'check_output', lambda *a, **kw: json.dumps({'global_result':{}}))
    folder = tmp_path / 'launch'
    folder.mkdir()
    identity = {'pid':123, 'start_us':456, 'boot_us':789}
    (folder / 'controller.json').write_text(json.dumps(identity))
    process = Mock()
    process.poll.side_effect = RuntimeError('injected failure')
    monkeypatch.setattr(runner, 'production_start', lambda *a: (folder, process))
    stopped = Mock()
    monkeypatch.setattr(runner, 'stop_production_identity', stopped)
    with pytest.raises(RuntimeError, match='injected'):
        runner.comparison()
    stopped.assert_called_once_with(identity)
    assert 'injected' in json.loads((tmp_path / 'campaign/comparison.json').read_text())['error']


@pytest.mark.parametrize('variant', ['raw', 'ema'])
def test_followup_selection_includes_parent_and_enforces_cutoff(runner, monkeypatch, tmp_path, variant):
    import torch
    from vesuvius.neural_tracing.fiber_trace_3d import live_evaluate
    name = 'best.pt' if variant == 'raw' else 'best_ema_0p99.pt'
    for source in ('parent', 'continued'):
        snapshots = tmp_path / source / 'snapshots'
        snapshots.mkdir(parents=True)
        (snapshots / name).touch()
    template = tmp_path / 'template.json'
    template.write_text(json.dumps({'evaluation':{}}))
    settings = dict(run=str(tmp_path / 'continued'), additional_runs=[str(tmp_path / 'parent')],
        variant=variant, max_samples=601600, template=str(template), output=str(tmp_path / 'eval'))
    def load(path, **kwargs):
        parent = path.parent.parent.name == 'parent'
        return dict(metric=1 if parent else 2, step=3900 if parent else 4700,
            samples_seen=499200 if parent else 601600,
            metric_name='test/loss_total' if variant == 'raw' else 'test_ema_0p99/loss_total')
    monkeypatch.setattr(torch, 'load', load)
    monkeypatch.setattr(live_evaluate, 'freeze_checkpoint', lambda path, dest: (path, 'sha'))
    monkeypatch.setattr(runner.subprocess, 'run', Mock())
    runner.trace_evaluate(settings)
    selection = json.loads((tmp_path / 'eval/selection.json').read_text())
    assert len(selection['candidates']) == 2
    assert selection['selected']['samples'] == 499200
    settings['max_samples'] = 600000
    with pytest.raises(ValueError, match='cutoff'):
        runner.trace_evaluate(settings)


def test_followup_resumes_raw_current_only_after_both_audits(runner, monkeypatch, tmp_path):
    import villa_gpu_manager.processes as processes
    launch = tmp_path / 'launch'
    launch.mkdir()
    (launch / 'sample_stop_result.json').write_text(json.dumps(
        dict(snapshot_samples=601600, run=str(tmp_path / 'continued'), identity={'pid':123})))
    settings = dict(launch=str(launch), max_samples=601600, parent_run='parent',
        template='template', output=str(tmp_path / 'eval'), continuation_config='continue.json')
    control = tmp_path / 'control.json'
    control.write_text(json.dumps({'comparison_followup':settings}))
    monkeypatch.setattr(runner, 'CONTROL', control)
    monkeypatch.setattr(processes, 'process_state', lambda *a: 'dead')
    calls = []
    def evaluate(options):
        calls.append(options['variant'])
        output = Path(options['output'])
        output.mkdir()
        (output / 'selection.json').write_text('{}')
        return output
    monkeypatch.setattr(runner, 'trace_evaluate', evaluate)
    monkeypatch.setattr(runner.subprocess, 'check_output', lambda *a, **kw: '{}')
    def start(config, resume):
        assert calls == ['raw', 'ema']
        assert resume == tmp_path / 'continued/snapshots/current.pt'
        return tmp_path / 'next', Mock(pid=456)
    monkeypatch.setattr(runner, 'production_start', start)
    runner.comparison_followup()
