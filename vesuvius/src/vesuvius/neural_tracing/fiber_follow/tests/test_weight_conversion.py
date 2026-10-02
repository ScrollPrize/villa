"""The one-off conversion preserves tensors without admitting legacy runtime models."""
from dataclasses import asdict
import copy

import pytest
import torch

from model_fixtures import coordinate_config, coordinate_batch
from vesuvius.neural_tracing.fiber_follow.models.model import build_model
from vesuvius.neural_tracing.fiber_follow.scripts.convert_v17_weights import convert
from vesuvius.neural_tracing.fiber_follow.train.train import initialize_model_weights, checkpoint_config, load_checkpoint
from vesuvius.neural_tracing.fiber_follow.train.runloop import read_checkpoint, resume_training
from vesuvius.neural_tracing.fiber_follow.data.data import SampleConfig, DATA_POLICY
from vesuvius.neural_tracing.fiber_follow.data.volume import FiberVolumeSpec


def test_conversion_preserves_predictions_and_rejects_incompatible_or_overwritten_files(tmp_path):
    cfg = coordinate_config()
    model = build_model(cfg).eval()
    old_cfg = cfg.to_dict(); old_cfg.pop('model_type')
    old_cfg['fine'].update(gate_direction=False, history_render='points', history_sigma=1.)
    old_cfg.update(direction_inputs=False, input_mode='ct', channels=32, encoder='patch4', token_only=True,
                   history_encoder='fine', history_path_tokens=True, path_geometry_tokens=True)
    old = dict(architecture='axial_patch4_residual_stem_tokens_fiber_slabs_v17', model_cfg=old_cfg,
               model=model.state_dict(), ema=copy.deepcopy(model.state_dict()), step=81000,
               optimizer={'old': True}, rng={'old': True}, data_policy=DATA_POLICY,
               ct_normalization=dict(method='crop_zscore_v1', volumes={}),
               vol_spec=FiberVolumeSpec('unused').to_dict(), fiber_manifest=[{'name': 'frozen'}],
               sample_cfg=asdict(SampleConfig(crop=cfg.fine)), n_commit=cfg.n_future)
    source, dest = tmp_path/'old.pt', tmp_path/'new.pt'
    torch.save(old, source)
    with pytest.raises(ValueError, match='Unsupported checkpoint model type'):
        read_checkpoint(source, 'coordinate_regression', 'cpu')
    result = convert(source, dest)
    assert checkpoint_config(result) == cfg
    assert result['fiber_manifest'] == old['fiber_manifest']
    assert not {'optimizer', 'rng', 'architecture'} & result.keys()
    fresh, ema = initialize_model_weights(cfg, 'cpu', result)
    loaded, *_ = load_checkpoint(dest, 'cpu')
    for key, value in old['model'].items():
        torch.testing.assert_close(fresh.state_dict()[key], value, atol=0, rtol=0)
    b = coordinate_batch(cfg, 1)
    with torch.no_grad():
        expected = model(b['x'], b['hist'], b['hmask'])
        actual = loaded(b['x'], b['hist'], b['hmask'])
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
    with pytest.raises(ValueError, match='optimizer state'):
        resume_training(result, fresh, ema, torch.optim.AdamW(fresh.parameters()))
    with pytest.raises(FileExistsError):
        convert(source, dest)
    old['model_cfg']['direction_inputs'] = True
    torch.save(old, source)
    with pytest.raises(ValueError, match='direction_inputs'):
        convert(source, tmp_path/'bad.pt')
