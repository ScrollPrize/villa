"""Task budgets and dense path counterfactuals after removal of InfoNCE."""
import copy
from collections import defaultdict

import numpy as np
import pytest
import torch

from test_identity import config, batch
from test_neighbor_bank import make_bank, add_shard, publish
from test_neighbor_following import clean_sample
from vesuvius.neural_tracing.fiber_follow.regression.data import IdentityObservationBuilder, IdentitySampling
from vesuvius.neural_tracing.fiber_follow.regression.feature_sequences import sequence_batches, stream_loss_weights
from vesuvius.neural_tracing.fiber_follow.regression.identity_decisions import decision_pair, path_candidates
from vesuvius.neural_tracing.fiber_follow.regression.model import DirectFollower
from vesuvius.neural_tracing.fiber_follow.regression.supervision import candidate_targets, loss_terms, geometry_mask
from vesuvius.neural_tracing.fiber_follow.regression.survival_confidence import survival_predictions, survival_loss
from vesuvius.neural_tracing.fiber_follow.regression.train import optimizer_update
from vesuvius.neural_tracing.fiber_follow.shared.geometry import CropSpec
from vesuvius.neural_tracing.fiber_follow.shared.training_log import DirectTrainingInterval


def test_stream_length_does_not_change_endpoint_or_total_budget(monkeypatch):
    import vesuvius.neural_tracing.fiber_follow.regression.feature_sequences as module
    monkeypatch.setattr(module, 'stream_rows', lambda item, builder, band: item)
    class Builder:
        cfg = config(feature_history_loss_fraction=.25)
        def __call__(self, items, vol):
            return dict(hist=torch.zeros(len(items), 1, 3))
    streams = [[dict(source=source, identity_seed=j) for _ in range(length)]
               for j, (source, length) in enumerate([(5, 6), (5, 9), (0, 18), (3, 130)])]
    totals, endpoints, observed = defaultdict(float), {}, defaultdict(int)
    for chunk in sequence_batches(Builder(), streams, None):
        for b in chunk['feature_sequence']:
            for sid, weight, end in zip(b['stream_id'].tolist(), b['loss_weight'].tolist(), b['stream_end'].tolist()):
                totals[sid] += weight
                observed[sid] += 1
                if end: endpoints[sid] = weight
    assert list(observed.values()) == [6, 9, 18, 130]
    np.testing.assert_allclose(list(totals.values()), 1.)
    np.testing.assert_allclose(list(endpoints.values()), .75)
    assert sum(endpoints.values())/sum(totals.values()) == pytest.approx(.75)
    assert (endpoints[0]+endpoints[1])/sum(totals.values()) == pytest.approx(.375)
    assert stream_loss_weights(1, .25).tolist() == [1.]


@pytest.mark.parametrize('value', [-.1, 1., float('nan')])
def test_invalid_history_budget(value):
    with pytest.raises(ValueError, match='History loss'):
        config(feature_history_loss_fraction=value)


def matched_batch(tmp_path, monkeypatch, choice):
    bank, parent = make_bank(tmp_path)
    publish(tmp_path, [add_shard(tmp_path, 0, x=4., z_range=(20., 180.))])
    cfg = config(n_future=16, fine=CropSpec(depth=48, width=25, behind=24, spacing=1.))
    rng = np.random.default_rng(17)
    pair = decision_pair(bank, clean_sample(cfg), cfg, rng, choice=choice)
    assert pair is not None
    builder = IdentityObservationBuilder(cfg, [parent], negative_bank=bank,
        sampling=IdentitySampling(decision_fraction=1.))
    for row in pair:
        builder.prepare(row, row.get('supervision_fiber', parent), rng)
    # Test geometry/labels without volume reads or the image encoder.
    monkeypatch.setattr(builder, 'images', lambda items, vol: dict(seed_mask=torch.zeros(len(items), 1)))
    return cfg, pair, builder(pair, None)


def test_same_local_observation_has_opposite_geometry_choices_and_late_failures(tmp_path, monkeypatch):
    cfg, pair, b = matched_batch(tmp_path, monkeypatch, True)
    for key in ('pos', 'frame', 'hist_local', 'hmask', 'candidate_points'):
        np.testing.assert_array_equal(pair[0][key], pair[1][key])
    assert not np.array_equal(pair[0]['seed_pos'], pair[1]['seed_pos'])
    assert geometry_mask(b, cfg).all()
    assert not {'identity_points', 'positive_mask', 'negative_mask'} & b.keys()
    assert b['candidate_points'].shape == (2, 4, 16, 3)
    for i in range(2):
        plain = b['candidate_kind'][i] == 0
        assert b['candidate_labels'][i, plain].all(-1).sum() == 1
        mixed = b['candidate_kind'][i] == 1
        labels = b['candidate_labels'][i, mixed]
        assert ((labels[:, 0] == 1) & (labels[:, -1] == 0)).sum() == 1
    # Geometry gradients must tell the generator to choose different fibers,
    # even when it proposes the identical midpoint curve for both observations.
    points = torch.zeros(2, cfg.n_future, 3)
    points[..., 2] = torch.arange(1, cfg.n_future+1)
    points.requires_grad_()
    hazards = torch.zeros(2, cfg.n_future, requires_grad=True)
    logits, confidence = survival_predictions(hazards)
    out = dict(points=points, hazard_logits=hazards, confidence_logits=logits, confidence=confidence)
    terms = loss_terms(out, b, cfg)
    terms['geometry_per_state'].sum().backward()
    assert (points.grad[0, :, :2]*points.grad[1, :, :2]).sum() < 0
    assert hazards.grad is None


def test_departed_pair_only_teaches_stopping_for_original_identity(tmp_path, monkeypatch):
    cfg, pair, b = matched_batch(tmp_path, monkeypatch, False)
    departed = b['offtrack'].bool()
    assert departed.sum() == 1
    assert not geometry_mask(b, cfg)[departed].any()
    assert geometry_mask(b, cfg)[~departed].all()
    labels = b['candidate_labels'][departed]
    masks = b['candidate_mask'][departed]
    hazards = torch.zeros_like(labels, requires_grad=True)
    loss, valid = survival_loss(hazards, labels, masks)
    assert valid[..., 0].all() and not valid[..., 1:].any()
    loss.sum().backward()
    assert hazards.grad[..., 0].lt(0).all() and hazards.grad[..., 1:].count_nonzero() == 0


def test_splice_onsets_cover_forecast_and_dense_labels_respect_censoring_and_tolerance():
    cfg = config(n_future=16, fine=CropSpec(depth=48, width=25, behind=24, spacing=1.))
    curves = np.zeros((2, 16, 3), np.float32)
    curves[..., 2] = np.arange(1, 17)
    curves[1, :, 0] = 4.
    b = batch(cfg, 1)
    b['identity_observable'] = torch.ones(1, dtype=torch.bool)
    first_failures = set()
    for seed in range(32):
        paths, support, kinds = path_candidates(curves, np.ones((2, 16), bool), np.random.default_rng(seed))
        b.update(candidate_points=torch.from_numpy(paths[None]), candidate_mask=torch.from_numpy(support[None]))
        labels, known = candidate_targets(b, cfg, 1.5)
        for k in np.flatnonzero(kinds == 1):
            if labels[0, k, 0] and not labels[0, k, -1]:
                first_failures.add(int(torch.nonzero(labels[0, k] == 0)[0]))
    assert min(first_failures) < 5 and max(first_failures) >= 12
    b['dense_mask'][:, 16:] = 0  # censor beyond forward plane five
    labels, known = candidate_targets(b, cfg, 1.5)
    pure = int(np.flatnonzero((paths[:, :, 0] == 0).all(-1))[0])
    assert labels[0, pure].all() and not known[0, pure, 5:].any()
    b['dense_mask'].fill_(1)
    labels, known = candidate_targets(b, cfg, 5.)
    assert labels.all() and known.all()  # caller's tolerance, not a hidden 1.5 default
    b['identity_observable'].zero_()
    assert not candidate_targets(b, cfg, 1.5)[1].any()


def test_weighted_optimizer_update_is_microbatch_invariant_and_logs_budget():
    torch.manual_seed(21)
    model = DirectFollower(config())
    b = batch(model.cfg)
    b.update(loss_weight=torch.tensor([.25/129, .75]), stream_end=torch.tensor([False, True]),
             decision_kind=torch.tensor([0., 1.]))
    def take(value, j):
        return {k: take(v, j) for k, v in value.items()} if isinstance(value, dict) else value[j:j+1]
    models = [copy.deepcopy(model), copy.deepcopy(model)]
    metrics = []
    for m, batches in zip(models, [[b], [take(b, 0), take(b, 1)]]):
        metrics.append(optimizer_update(m, copy.deepcopy(m), torch.optim.SGD(m.parameters(), lr=.01),
                                        batches, 1, .01, compute_metrics=False))
    assert metrics[0]['loss'] == pytest.approx(metrics[1]['loss'], rel=1e-5)
    for a, z in zip(models[0].parameters(), models[1].parameters()):
        torch.testing.assert_close(a, z, atol=1e-7, rtol=1e-5)
    interval = DirectTrainingInterval()
    interval.add(metrics[0])
    summary = interval.summary()
    assert summary['endpoint_states'] == summary['choice_endpoint_states'] == 1
    assert summary['matched_endpoint_weight'] == .75
    assert summary['supervision_weight'] == pytest.approx(.75+.25/129)


def test_pair_is_discarded_together_if_one_history_crosses_holdout(monkeypatch):
    import vesuvius.neural_tracing.fiber_follow.regression.feature_sequences as module
    monkeypatch.setattr(module, 'stream_rows', lambda item, builder, band: item['rows'])
    class Builder:
        cfg = config()
        def __call__(self, items, vol):
            return dict(hist=torch.zeros(len(items), 1, 3), source=torch.tensor([r['source'] for r in items]))
    items = [dict(pair_observation_seed=123, rows=[]),
             dict(pair_observation_seed=123, rows=[dict(source=5, identity_seed=1)]),
             dict(rows=[dict(source=0, identity_seed=2)])]
    batches = list(sequence_batches(Builder(), items, None))
    assert len(batches) == 1
    assert batches[0]['feature_sequence'][0]['source'].tolist() == [0]


def test_unlabeled_candidate_padding_does_not_run_scorer(monkeypatch):
    from test_identity_decisions import candidate_batch
    model = DirectFollower(config())
    b = candidate_batch(model.cfg)
    b['candidate_mask'].zero_()
    calls = []
    forward = model.forward
    def observed(*args, **kwargs):
        calls.append(kwargs.get('candidates'))
        return forward(*args, **kwargs)
    monkeypatch.setattr(model, 'forward', observed)
    optimizer_update(model, copy.deepcopy(model), torch.optim.SGD(model.parameters(), lr=.01),
                     [b], 1, .01, compute_metrics=False)
    assert calls == [None]
