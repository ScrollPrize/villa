"""The observation builder emits only the crop transformer's inputs; training logs carry no memory metrics."""
from types import SimpleNamespace

import numpy as np
import torch

from model_fixtures import coordinate_batch, coordinate_config, fake_ct, observation
from vesuvius.neural_tracing.fiber_follow.train.train import (
    initialize_model_weights, initialize_training_optimizer, optimizer_update, prepare_training)
from vesuvius.neural_tracing.fiber_follow.train.training_log import TrainingInterval, format_training_log


def test_training_logs_one_gradient_group_and_no_memory_metrics():
    torch.manual_seed(5)
    model, ema = initialize_model_weights(coordinate_config(), 'cpu')
    opt, _, _ = initialize_training_optimizer(model, ema, SimpleNamespace(lr=.001, reset_optimizer=False))
    prepare_training(model, backend='eager')
    metrics = optimizer_update(model, ema, opt, [coordinate_batch(model.cfg)], 1, .001, grad_clip=1e-3)
    assert np.isfinite(metrics['loss']) and metrics['grad_norm'] > 0 and metrics['grad_clip_scale'] < 1
    assert not [key for key in metrics if key.startswith(('history_', 'memory_', 'verify_', 'tube_'))]
    interval = TrainingInterval()
    interval.add(metrics)
    interval.add(metrics)
    summary = interval.summary()
    assert summary['clipped_updates'] == 2 and summary['grad_norm_max'] == metrics['grad_norm']
    text = format_training_log(dict(step=50, geometry=1., loss=1., lr=.001, interval=summary, n_future=4, tolerance=1.5,
                                    interval_update_seconds=1., interval_data_seconds=.1, interval_samples_per_second=4.))
    assert 'historical slabs' not in text and 'clipped 2/2 updates' in text


def test_observations_carry_only_crop_transformer_inputs(monkeypatch):
    from vesuvius.neural_tracing.fiber_follow.data.observations import ObservationBuilder
    fake_ct(monkeypatch)
    monkeypatch.setattr('vesuvius.neural_tracing.fiber_follow.data.observations.image_crop',
        lambda items, vol, crop, pool=None, **kw: torch.zeros(len(items), 1, crop.depth, crop.width, crop.width))
    item = observation(np.c_[np.sin(np.arange(301)/12), np.zeros(301), np.arange(301)])
    builder = ObservationBuilder(coordinate_config())
    x = builder.images([item], None)
    assert set(x) == {'fine', 'seed', 'seed_mask', 'seed_age', 'seed_tangent', 'path_geometry', 'path_geometry_valid',
                      'ct_frame_source', 'ct_frame_energy', 'ct_frame_gap'}
    assert list(builder.prefetch_bounds(dict(item), SimpleNamespace(input_scale=1.)))  # the current crop only
