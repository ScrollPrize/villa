import torch
import pytest
from vesuvius.neural_tracing.fiber_trace_3d.train import (
    _FullResolutionEvaluation, _make_test_loader_raw_config, _model_output_stride,
)


def test_evaluation_overrides_do_not_change_training():
    raw = {'batch_size':16, 'sampling_strategy':'scroll_balanced_arclength',
           'datasets':[{'dense_supervision':True}],
           'test_datasets':[{'dense_supervision':False}], 'augment_enabled':True}
    test = _make_test_loader_raw_config(raw, {'test_batch_size':8,
                                            'test_sampling_strategy':'control_points'})
    assert test['batch_size'] == 8
    assert test['sampling_strategy'] == 'control_points'
    assert test['datasets'][0]['dense_supervision'] is False
    assert test['augment_enabled'] is False
    assert raw['batch_size'] == 16 and raw['augment_enabled'] is True
    assert raw['datasets'][0]['dense_supervision'] is True


@pytest.mark.parametrize('mode', ['nearest','trilinear'])
def test_historical_output_grid(mode):
    model = torch.nn.AvgPool3d(4)
    wrapped = _FullResolutionEvaluation(model, mode)
    x = torch.ones(2,7,16,16,16)
    assert _model_output_stride(wrapped) == 1
    torch.testing.assert_close(wrapped(x), x)


def test_checkpoint_evaluation_parser():
    from vesuvius.neural_tracing.fiber_trace_3d.evaluation_benchmark import parser
    args = parser().parse_args(['--reference','config.json','--output','/tmp/eval',
                               '--checkpoint','weights.pt','--batch-size','8'])
    assert args.checkpoint.name == 'weights.pt'
    assert args.batch_size == 8


def test_experiment_stop_requires_guard_exit(tmp_path, monkeypatch):
    import importlib.util
    import json
    from pathlib import Path
    import villa_gpu_manager.processes as processes
    path = Path(__file__).resolve().parents[2] / 'scripts/fiber3d_experiment.py'
    spec = importlib.util.spec_from_file_location('repro_runner_test', path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    monkeypatch.setattr(runner, 'OUTPUT', tmp_path)
    folder = tmp_path / 'run'
    folder.mkdir()
    (tmp_path / 'latest').write_text(str(folder))
    (folder / 'controller.json').write_text(json.dumps({'pid':1}))
    (folder / 'guard.json').write_text(json.dumps({'pid':2}))
    monkeypatch.setattr(processes, 'process_state', lambda row, uid: 'alive' if row['pid']==2 else 'dead')
    ticks = iter([0,46])
    monkeypatch.setattr(runner.time, 'monotonic', lambda: next(ticks))
    with pytest.raises(RuntimeError, match='ownership must be retained'):
        runner.status(stop=True)
