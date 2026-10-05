"""memory='none': no memory modules, inputs, metrics or log lines; identity defaults train no identity loss."""
from types import SimpleNamespace

import pytest
import torch

from model_fixtures import coordinate_config, coordinate_batch
from test_decision_memory import decision_inputs
from test_flow_model import config as flow_config
from vesuvius.neural_tracing.fiber_follow.models.model import build_model
from vesuvius.neural_tracing.fiber_follow.train.train import (
    initialize_model_weights, initialize_training_optimizer, optimizer_update, prepare_training)
from vesuvius.neural_tracing.fiber_follow.train.training_log import TrainingInterval, format_training_log


def without_memory(batch):
    batch['x'] = {k: v for k, v in batch['x'].items() if not k.startswith('history_')}
    return batch


@pytest.mark.parametrize('make', [coordinate_config, flow_config])
def test_no_memory_model_has_no_memory_parameters_and_runs(make):
    torch.manual_seed(3)
    model = build_model(make(memory='none')).eval()
    assert not hasattr(model, 'history_encoder') and not hasattr(model, 'history_attention')
    assert model.confidence_scorer.history_attention is None
    assert not any('history' in name for name, _ in model.named_parameters())
    batch = without_memory(coordinate_batch(model.cfg))
    with torch.no_grad():
        out = model(batch['x'], batch['hist'], batch['hmask'])
    assert 'memory_entry' not in out and torch.isfinite(out['confidence']).all()


@pytest.mark.parametrize('make', [coordinate_config, flow_config])
def test_no_memory_equals_the_memory_model_reading_empty_memory(make):
    # Empty memory rows contribute exactly nothing, so the shared weights give identical outputs.
    torch.manual_seed(4)
    memory = build_model(make(memory='decisions', stem='stride2', stem_blocks=1)).eval()
    none = build_model(make(memory='none', stem='stride2', stem_blocks=1)).eval()
    missing, unexpected = none.load_state_dict(memory.state_dict(), strict=False)
    assert not missing and all('history' in key for key in unexpected)
    batch = without_memory(coordinate_batch(memory.cfg))
    with torch.no_grad():
        torch.manual_seed(0)
        a = memory(dict(batch['x'], **decision_inputs(memory.cfg, 2)), batch['hist'], batch['hmask'])
        torch.manual_seed(0)
        b = none(batch['x'], batch['hist'], batch['hmask'])
    for key in ('points', 'confidence'):
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)


def test_memory_none_rejects_identity_objectives():
    for options in (dict(identity_objective='verify'), dict(identity_dim=8), dict(identity_objective='readout')):
        with pytest.raises(ValueError):
            coordinate_config(memory='none', **options)


def test_identity_defaults_build_no_infonce_or_verifier():
    model = build_model(coordinate_config(memory='decisions', stem='stride2', stem_blocks=1))
    assert model.cfg.identity_mode is None and model.identity_projection is None
    assert not hasattr(model, 'identity_verifier') and not hasattr(model, 'identity_embedding')


def test_training_without_memory_logs_no_memory_metrics():
    torch.manual_seed(5)
    model, ema = initialize_model_weights(coordinate_config(memory='none'), 'cpu')
    opt, _, _ = initialize_training_optimizer(model, ema, SimpleNamespace(lr=.001, reset_optimizer=False))
    prepare_training(model, backend='eager')
    metrics = optimizer_update(model, ema, opt, [without_memory(coordinate_batch(model.cfg))], 1, .001)
    assert math_finite(metrics['loss']) and metrics['rest_grad_norm'] > 0
    assert not [key for key in metrics if key.startswith(('history_', 'memory_'))]
    interval = TrainingInterval()
    interval.add(metrics)
    summary = interval.summary()
    assert 'history_grad_norm_max' not in summary and 'history_clipped_updates' not in summary
    text = format_training_log(dict(step=50, geometry=1., loss=1., lr=.001, interval=summary, n_future=4, tolerance=1.5,
                                    interval_update_seconds=1., interval_data_seconds=.1, interval_samples_per_second=4.))
    assert 'historical slabs' not in text and 'history max' not in text and 'rest max' in text


def math_finite(value):
    return value == value and abs(value) != float('inf')


def test_observations_without_memory_carry_no_memory_inputs(monkeypatch):
    import numpy as np
    from test_history_slabs import fake_ct, observation
    from vesuvius.neural_tracing.fiber_follow.data.observations import ObservationBuilder
    fake_ct(monkeypatch)
    monkeypatch.setattr('vesuvius.neural_tracing.fiber_follow.data.observations.image_crop',
        lambda items, vol, crop, pool=None, **kw: torch.zeros(len(items), 1, crop.depth, crop.width, crop.width))
    item = observation(np.c_[np.sin(np.arange(301)/12), np.zeros(301), np.arange(301)])
    builder = ObservationBuilder(coordinate_config(memory='none'))
    x = builder.images([item], None)
    assert not [key for key in x if key.startswith('history_')]
    assert 'path_geometry' in x
    assert list(builder.prefetch_bounds(dict(item), SimpleNamespace(input_scale=1.)))  # the current crop only
