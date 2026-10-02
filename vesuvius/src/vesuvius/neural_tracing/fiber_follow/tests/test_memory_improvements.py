import copy
import io
import json

import numpy as np
import pytest
import torch

from slab_fixtures import cfg, slab_batch
from test_history_slabs import fake_ct, observation
from vesuvius.neural_tracing.fiber_follow.regression.history_slabs import load_slabs, HistoryEncoder
from vesuvius.neural_tracing.fiber_follow.regression.model import build_model
from vesuvius.neural_tracing.fiber_follow.regression.train import prepare_training, optimizer_update
from vesuvius.neural_tracing.fiber_follow.shared.policy import OperatingPolicy
from vesuvius.neural_tracing.fiber_follow.regression.live_continuation import LiveContinuationSource
from vesuvius.neural_tracing.fiber_follow.shared.training_log import DirectTrainingInterval


def path_batch(c, count=2):
    batch = slab_batch(c, count)
    points = torch.zeros(count, 8, 3, 3)
    points[..., 2] = torch.tensor([-1., 0., 1.])
    tangents = torch.zeros_like(points); tangents[..., 2] = 1
    batch['x'].update(history_path_points=points, history_path_tangents=tangents,
                      history_path_valid=batch['x']['history_valid'][..., None].expand(-1, -1, 3).clone())
    return batch


def test_path_samples_use_observed_geometry_and_mask_seed_boundary(monkeypatch):
    fake_ct(monkeypatch)
    path = np.c_[np.arange(100.)*.1, np.zeros(100), np.arange(100.)]
    item = observation(path)
    a = load_slabs([item], None, cfg(history_path_tokens=True))
    b = load_slabs([dict(item, gt_history=np.full((32, 3), np.nan), offtrack=True)], None, cfg(history_path_tokens=True))
    for key in ('history_path_points', 'history_path_tangents', 'history_path_valid'):
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
    assert a['history_path_valid'][0, 0].tolist() == [False, True, True]
    assert not a['history_path_valid'][~a['history_valid']].any()


@pytest.mark.parametrize('empty', [False, True])
def test_path_tokens_keep_spatial_bank_and_mask_padding(empty):
    c = cfg(history_path_tokens=True)
    encoder = HistoryEncoder(c)
    batch = path_batch(c)['x']
    if empty: batch['history_valid'][:] = False
    batch['history_path_valid'][0, 0, 0] = False
    batch['history_path_points'][0, 0, 0] = float('nan')
    tokens, padding = encoder(batch['history_slabs'], batch['history_valid'], batch['history_pose'],
        path_points=batch['history_path_points'], path_tangents=batch['history_path_tangents'], path_valid=batch['history_path_valid'])
    assert tokens.shape == (2, 8*581, c.hidden)
    assert padding[0, 578] and torch.isfinite(tokens).all()
    assert tokens[padding].eq(0).all()
    if not empty:
        tokens[:, 579].square().sum().backward()
        assert encoder.path_projection[-1].weight.grad.abs().sum() > 0
        assert encoder.convolution[0].weight.grad.abs().sum() > 0


def test_path_model_compiled_update_trains_memory():
    c = cfg(history_path_tokens=True)
    model = prepare_training(build_model(c), 2, backend='eager')
    batch = path_batch(c)
    ema = copy.deepcopy(build_model(c))
    before = model.history_encoder.path_projection[-1].weight.detach().clone()
    metrics = optimizer_update(model, ema, torch.optim.AdamW(model.parameters()), [batch], 1, .0001,
                               device='cpu', compute_metrics=False)
    assert np.isfinite(metrics['loss'])
    assert not torch.equal(before, model.history_encoder.path_projection[-1].weight)


def test_stratified_limits_balance_across_worker_copies():
    source = LiveContinuationSource(policy=OperatingPolicy(), steps=(12, 32))
    worker = copy.copy(source)
    rng = np.random.default_rng(45)
    try:
        limits = [s.draw_limit(rng) for _ in range(300) for s in (source, worker)]
        bands = np.array([(v-12)//7 for v in limits])
        assert np.bincount(bands).tolist() == [200, 200, 200]
        assert min(limits) == 12 and max(limits) == 32
    finally: source.close()
