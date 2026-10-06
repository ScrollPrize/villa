"""Resuming and initializing report differences instead of refusing: checkpoints, replay, model configuration."""
import numpy as np
import pytest
import torch

from model_fixtures import config, run_document
from vesuvius.neural_tracing.fiber_follow.data import data as D
from vesuvius.neural_tracing.fiber_follow.models.model import build_model
from vesuvius.neural_tracing.fiber_follow.train import run_config
from vesuvius.neural_tracing.fiber_follow.train.runloop import read_checkpoint
from vesuvius.neural_tracing.fiber_follow.train.train import configured_model, initialize_model_weights
from test_data_and_scoring import fiber, make_states


def test_checkpoints_load_without_a_data_policy_and_legacy_types_point_to_the_converter(tmp_path):
    path = tmp_path/'ck.pt'
    torch.save(dict(model_type='regression', model={}), path)
    assert read_checkpoint(path, 'cpu')['model_type'] == 'regression'
    for legacy in ('unified', 'unified_flow', 'coordinate_regression'):
        torch.save(dict(model_type=legacy, data_policy='controlled_spans_v3'), path)
        with pytest.raises(ValueError, match='convert_checkpoint.py'):
            read_checkpoint(path, 'cpu')


def test_replay_caches_that_do_not_load_or_match_are_skipped(tmp_path, capsys):
    f = fiber()
    good, moved = make_states(f), make_states(f)
    good.save(tmp_path/'good.npz')
    moved.save(tmp_path/'moved.npz')
    np.savez(tmp_path/'old.npz', pos=np.zeros((1, 3)))
    loaded = D.load_replay([tmp_path/'good.npz', tmp_path/'missing.npz', tmp_path/'moved.npz'])
    assert len(loaded) == 2 and 'missing.npz' in capsys.readouterr().out
    assert D.load_replay([tmp_path/'old.npz']) == [] and 'old.npz' in capsys.readouterr().out  # no metadata
    other = fiber(length=120)
    assert D.usable_replay(loaded, [f], 4, 8.) == loaded
    assert D.usable_replay(loaded, [other], 4, 8.) == [] and 'incompatible fibers' in capsys.readouterr().out
    assert D.usable_replay(loaded, [f], 8, 8.) == [] and 'history length' in capsys.readouterr().out
    assert D.usable_replay(loaded, [f], 4, 4.) == [] and 'coordinate scale' in capsys.readouterr().out


def test_resume_takes_the_checkpoint_model_and_reports_ignored_settings():
    run = run_config.resolve(run_document(fine=dict(depth=24, width=24, behind=8, spacing=.5), n_future=4,
                                          gate_plane=None, hidden=16, heads=2, layers=1, ffn=32, cnn_channels=[4, 8],
                                          cnn_blocks=[1, 1], recurrent_refinement_steps=0, n_history=32))
    messages = []
    assert configured_model(run, None, messages.append) == run_config.model_config(run) and not messages
    recorded = config(n_future=4, recurrent_refinement_steps=2)
    resumed = configured_model(run, dict(model_type='regression', model_cfg=recorded.to_dict()), messages.append)
    assert resumed == recorded
    assert len(messages) == 1 and 'recurrent_refinement_steps' in messages[0] and 'fine' in messages[0]
    messages.clear()
    configured_model(run, dict(model_type='regression', model_cfg=run_config.model_config(run).to_dict()), messages.append)
    assert not messages


def test_initialization_loads_matching_tensors_and_reports_the_rest():
    torch.manual_seed(0)
    source = build_model(config(recurrent_refinement_steps=2))
    target = config(hidden=32, recurrent_refinement_steps=0)
    initial = dict(model=source.state_dict(), ema=source.state_dict())
    model, ema = initialize_model_weights(target, 'cpu', initial)
    report = model.initialization_report
    assert any(k.startswith('refinement_') for k in report['unexpected'])
    assert 'cell_token.weight' in report['mismatched'] and 'cell_token.weight' in report['fresh']
    # Same-shape tensors (the CNN) come from the checkpoint, in the model and the EMA.
    for name in ('cnn.stages.0.blocks.0.conv1.conv.weight',):
        torch.testing.assert_close(model.state_dict()[name], source.state_dict()[name])
        torch.testing.assert_close(ema.state_dict()[name], source.state_dict()[name])
    excluded, _ = initialize_model_weights(target, 'cpu', initial, exclude=('cnn.',))
    assert 'cnn.stages.0.blocks.0.conv1.conv.weight' in excluded.initialization_report['fresh']
    fresh, _ = initialize_model_weights(target, 'cpu')
    assert fresh.initialization_report is None
