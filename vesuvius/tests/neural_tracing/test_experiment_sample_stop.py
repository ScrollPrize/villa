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
