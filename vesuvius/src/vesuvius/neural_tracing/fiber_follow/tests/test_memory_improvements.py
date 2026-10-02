import copy
import io
import json

import numpy as np
import pytest
import torch

from slab_fixtures import cfg, slab_batch
from test_history_slabs import fake_ct, observation
from test_refinement_resume import checkpoint
from vesuvius.neural_tracing.fiber_follow.regression.history_slabs import load_slabs, HistoryEncoder
from vesuvius.neural_tracing.fiber_follow.regression.model import build_model
from vesuvius.neural_tracing.fiber_follow.regression.memory_resume import expand_memory_checkpoint, migrate_fixture_normalization, fork_run
from vesuvius.neural_tracing.fiber_follow.regression.supervision import paired_identity_loss
from vesuvius.neural_tracing.fiber_follow.regression.train import checkpoint_config, prepare_training, optimizer_update
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


def paired_batch():
    return dict(identity_pair_id=torch.tensor([41, 7, 41, -1]),
                candidate_points=torch.zeros(4, 2, 4, 3),
                candidate_labels=torch.tensor([[[1.]*4, [0.]*4], [[1.]*4, [0.]*4],
                                               [[0.]*4, [1.]*4], [[0.]*4, [1.]*4]]),
                candidate_mask=torch.ones(4, 2, 4), identity_observable=torch.ones(4, dtype=torch.bool))


def test_pair_ranking_matches_ids_and_moves_both_histories():
    batch = paired_batch()
    logits = torch.zeros(4, 2, 4, requires_grad=True)
    terms = paired_identity_loss(logits, batch)
    assert terms['pair_rank_comparisons'] == 8
    assert terms['pair_rank_per_state'][[1, 3]].eq(0).all()
    terms['pair_rank_per_state'].sum().backward()
    assert logits.grad[0, 0].lt(0).all() and logits.grad[2, 0].gt(0).all()
    assert logits.grad[[1, 3]].eq(0).all()
    improved = paired_identity_loss(logits.detach()-logits.grad, batch)
    assert improved['pair_rank_correct'] == 8
    assert improved['pair_rank_per_state'].sum() < terms['pair_rank_per_state'].sum()


@pytest.mark.parametrize('reason', ['unknown', 'unobservable', 'mismatched', 'duplicate', 'same_label'])
def test_pair_ranking_excludes_invalid_comparisons(reason):
    batch = paired_batch()
    if reason == 'unknown': batch['candidate_mask'][2] = 0
    if reason == 'unobservable': batch['identity_observable'][2] = 0
    if reason == 'mismatched': batch['candidate_points'][2] += 1
    if reason == 'duplicate': batch['identity_pair_id'][1] = 41
    if reason == 'same_label': batch['candidate_labels'][2] = batch['candidate_labels'][0]
    logits = torch.randn(4, 2, 4, requires_grad=True)
    terms = paired_identity_loss(logits, batch)
    assert terms['pair_rank_comparisons'] == 0
    terms['pair_rank_per_state'].sum().backward()
    assert logits.grad.eq(0).all()


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


def test_path_model_compiled_update_trains_memory_and_pair_objective():
    c = cfg(history_path_tokens=True)
    model = prepare_training(build_model(c), 2, backend='eager')
    batch = path_batch(c)
    pairs = paired_batch()
    for key, value in pairs.items(): batch[key] = value[[0, 2]]
    batch['candidate_points'][..., 2] = torch.arange(1, 5)
    ema = copy.deepcopy(build_model(c))
    before = model.history_encoder.path_projection[-1].weight.detach().clone()
    metrics = optimizer_update(model, ema, torch.optim.AdamW(model.parameters()), [batch], 1, .0001,
                               device='cpu', pair_rank_weight=.25, compute_metrics=False)
    assert np.isfinite(metrics['loss']) and metrics['identity']['pair_rank_comparisons'] == 8
    assert not torch.equal(before, model.history_encoder.path_projection[-1].weight)


def test_stratified_limits_balance_across_worker_copies():
    source = LiveContinuationSource(steps=(12, 32), stratified=True)
    worker = copy.copy(source)
    rng = np.random.default_rng(45)
    try:
        limits = [s.draw_limit(rng) for _ in range(300) for s in (source, worker)]
        bands = np.array([(v-12)//7 for v in limits])
        assert np.bincount(bands).tolist() == [200, 200, 200]
        assert min(limits) == 12 and max(limits) == 32
    finally: source.close()


def test_migration_preserves_every_existing_weight_and_named_moment(tmp_path):
    original = checkpoint(tmp_path)
    original['training_options']['live_continuation'] = True
    rng = torch.get_rng_state().clone()
    changed = expand_memory_checkpoint(original)
    assert torch.equal(rng, torch.get_rng_state())
    old = build_model(checkpoint_config(original))
    new = build_model(checkpoint_config(changed))
    for section in ('model', 'ema'):
        for name, value in original[section].items():
            torch.testing.assert_close(changed[section][name], value, rtol=0, atol=0)
        new.load_state_dict(changed[section], strict=True)
    ids = {name: i for i, (name, _) in enumerate(new.named_parameters())}
    for i, (name, _) in enumerate(old.named_parameters()):
        for key, value in original['optimizer']['state'][i].items():
            torch.testing.assert_close(changed['optimizer']['state'][ids[name]][key], value, rtol=0, atol=0)
    opt = torch.optim.AdamW(new.parameters()); opt.load_state_dict(changed['optimizer'])
    for p in new.parameters(): p.grad = torch.ones_like(p)
    opt.step()
    assert {k for k in original['training_options'] if original['training_options'][k] != changed['training_options'][k]} == {
        'history_path_tokens', 'pair_rank_weight', 'live_continuation_steps', 'live_continuation_stratified'}
    assert changed['step'] == original['step'] and changed['lr_restart_step'] == original['lr_restart_step']


def test_fixture_migration_changes_only_normalization_metadata():
    metadata = dict(provenance=dict(volume=dict(ct_normalization={'old': True}, grid_scale=8)), fibers=['a'])
    stream = io.BytesIO(); np.savez(stream, __metadata__=json.dumps(metadata), points=np.arange(12).reshape(4, 3))
    payload = migrate_fixture_normalization(stream.getvalue(), dict(vol_spec=dict(ct_normalization={'new': True})))
    with np.load(io.BytesIO(payload)) as updated:
        np.testing.assert_array_equal(updated['points'], np.arange(12).reshape(4, 3))
        expected = copy.deepcopy(metadata); expected['provenance']['volume']['ct_normalization'] = {'new': True}
        assert json.loads(str(updated['__metadata__'])) == expected
