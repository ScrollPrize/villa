"""Sparse decisions retain all visual evidence and train causal remote memory."""
import copy
import itertools

import numpy as np
import pytest
import torch

from test_trajectory_memory import cfg, memory_batch
from vesuvius.neural_tracing.fiber_follow.regression.feature_sequences import decision_plan
from vesuvius.neural_tracing.fiber_follow.regression.model import build_model
from vesuvius.neural_tracing.fiber_follow.regression.stratified_replay import ObservationHistory
from vesuvius.neural_tracing.fiber_follow.regression.train import (
    optimizer_update, prepare_training, training_prediction, training_memory_transition,
    training_observation_features, begin_training_update, finish_training_update,
)
from vesuvius.neural_tracing.fiber_follow.regression.supervision import loss_terms


def stream(c, length=7, batch=1, seed=7):
    weights, selected, retained = decision_plan(length, c, np.random.default_rng(seed))
    rows = []
    for t in range(length):
        row = memory_batch(c, batch, t)
        row.update(stream_id=torch.arange(batch), stream_reset=torch.full((batch,), t == 0),
                   stream_end=torch.full((batch,), t == length-1), stream_index=torch.full((batch,), t),
                   decision_mask=torch.full((batch,), weights[t] > 0),
                   loss_weight=torch.full((batch,), weights[t], dtype=torch.float32),
                   encoder_indices=torch.tensor(selected[t]).expand(batch, -1).clone(),
                   retain_until=torch.full((batch,), int(retained[t])))
        rows.append(row)
    return [dict(feature_sequence=rows[t:t+2]) for t in range(0, length, 2)]


def test_sparse_plan_causal_reproducible_bounded_and_unbiased():
    c = cfg()
    expected = np.r_[np.full(8, .25/8), .75]
    for seed in range(32):
        weights, selected, retained = decision_plan(9, c, np.random.default_rng(seed))
        assert np.count_nonzero(weights) == 3
        assert weights.sum() == pytest.approx(1.)
        assert weights[-1] == .75
        for t, indices in enumerate(selected):
            past = indices[indices >= 0]
            assert len(past) <= 3 and (past < t).all()
            assert all(retained[i] >= t for i in past)
        assert np.count_nonzero(retained >= 0) <= 9
    # Enumerate all equally likely query subsets instead of using a flaky
    # Monte Carlo assertion about a particular random seed sequence.
    estimates = []
    for pair in itertools.combinations(range(8), 2):
        class Enumerated:
            def choice(self, *args, **kwargs):
                return np.array(pair)
            def integers(self, lo, hi):
                return lo
        estimates.append(decision_plan(9, c, Enumerated())[0])
    np.testing.assert_allclose(np.mean(estimates, axis=0), expected, atol=1e-12)
    for a, b in zip(decision_plan(9, c, np.random.default_rng(7)), decision_plan(9, c, np.random.default_rng(7))):
        np.testing.assert_array_equal(a, b)
    assert decision_plan(1, c, np.random.default_rng(0))[0].tolist() == [1.]
    assert decision_plan(9, cfg(feature_history_decisions=0), np.random.default_rng(0))[0].tolist() == [0.]*8+[1.]


def test_observation_only_never_decodes_and_endpoint_is_predicted_once():
    torch.manual_seed(731)
    model = prepare_training(build_model(cfg(feature_history_decisions=0)), backend='eager')
    chunks = stream(model.cfg, length=7)
    calls = dict(observed=0, predicted=0)
    observe, predict = model.collect_observation_features, model.training_forward
    def observing(*args):
        assert not torch.is_grad_enabled()
        calls['observed'] += len(args[1])
        return observe(*args)
    def predicting(*args):
        calls['predicted'] += len(args[1])
        return predict(*args)
    model.collect_observation_features, model.training_forward = observing, predicting
    history = ObservationHistory()
    ema = copy.deepcopy(model)
    opt = torch.optim.AdamW(model.parameters())
    before = {k: v.clone() for k, v in model.state_dict().items()}
    result = optimizer_update(model, ema, opt, chunks[:-1], 1, .001, stream_states=history, compute_metrics=False)
    assert result['supervised_states'] == 0 and not result['optimizer_applied']
    assert not opt.state
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    assert len(history.streams[0]) == 6
    assert sum(row['batch'] is not None for row in history.streams[0]) <= 3
    assert all(v.grad_fn is None and not v.requires_grad for row in history.streams[0] for v in row['features'])
    result = optimizer_update(model, ema, opt, chunks[-1:], 1, .001, stream_states=history, compute_metrics=False)
    assert calls == dict(observed=6, predicted=1)
    assert result['supervised_states'] == result['endpoint_states'] == 1
    assert result['memory_replay_observations'] == 6
    assert result['history_encoder_crops'] == 2
    assert result['supervision_weight'] == 1.
    assert model.recurrent_memory.gate.weight.grad.abs().sum() > 0
    assert not history.streams


def test_sparse_training_matches_explicit_replay_loss_and_gradients():
    torch.manual_seed(732)
    model = prepare_training(build_model(cfg(recurrent_refinement_steps=1)), backend='eager')
    reference = build_model(model.cfg)
    reference.load_state_dict(model.state_dict())
    prepare_training(reference, backend='eager')
    chunks = stream(model.cfg, length=9)
    rows = [row for chunk in chunks for row in chunk['feature_sequence']]
    count = sum(bool(row['decision_mask'][0]) for row in rows)
    history = ObservationHistory()
    result = optimizer_update(model, copy.deepcopy(reference), torch.optim.SGD(model.parameters(), lr=0.),
        chunks, 1, 0., compute_metrics=False, stream_states=history, memory_grad_clip=0., rest_grad_clip=0.)
    # Independent explicit implementation: no collector or persisted writer graph.
    cached = []
    expected_loss = 0.
    begin_training_update(reference)
    for t, row in enumerate(rows):
        if bool(row['decision_mask'][0]):
            state = None
            selected = set(row['encoder_indices'][0].tolist())-{-1}
            for i, previous in enumerate(rows[:t]):
                features = training_observation_features(reference, previous['x'], previous['hist'], previous['hmask']) if i in selected else cached[i]
                state, _ = training_memory_transition(reference, *features, previous['x'], state)
            output = training_prediction(reference, row['x'], row['hist'], row['hmask'], memory=state)
            terms = loss_terms(output, row, reference.cfg)
            loss = ((terms['geometry_per_state']+.5*terms['confidence_per_state'])*row['loss_weight']).sum()/count
            loss.backward()
            expected_loss += loss.detach().item()
            features = tuple(output['observation_'+k] for k in ('tokens', 'xyz', 'valid'))
        else:
            with torch.no_grad():
                features = reference.observation_features(row['x'], row['hist'], row['hmask'])
        cached.append(tuple(v.detach().clone() for v in features))
    finish_training_update(reference)
    assert result['loss'] == pytest.approx(expected_loss, rel=1e-5, abs=1e-6)
    assert result['supervised_states'] == 3 and result['observed_states'] == 9
    assert result['endpoint_states'] == 1 and not history.streams
    for (name, p), (_, q) in zip(model.named_parameters(), reference.named_parameters()):
        if p.grad is None:
            assert q.grad is None, name
        else:
            torch.testing.assert_close(p.grad, q.grad, atol=3e-6, rtol=3e-4, msg=name)


def test_remote_crop_gradients_survive_eviction_and_retention_is_released():
    torch.manual_seed(733)
    model = prepare_training(build_model(cfg(memory_steps=2, feature_history_decisions=0)), backend='eager')
    chunks = stream(model.cfg, length=7)
    rows = [row for chunk in chunks for row in chunk['feature_sequence']]
    for row in rows:
        row['x']['feature_seed_here'].zero_()  # no anchor shortcut
        row['retain_until'].fill_(-1)
    rows[0]['retain_until'].fill_(6)
    rows[-1]['encoder_indices'][:] = torch.tensor([0, -1, -1])
    history = ObservationHistory()
    opt = torch.optim.SGD(model.parameters(), lr=0.)
    optimizer_update(model, copy.deepcopy(model), opt, chunks[:-1], 1, 0., stream_states=history, compute_metrics=False)
    selected_image = history.streams[0][0]['batch']['x']['fine']
    selected_image.requires_grad_(True)
    result = optimizer_update(model, copy.deepcopy(model), opt, chunks[-1:], 1, 0., stream_states=history, compute_metrics=False)
    assert result['history_encoder_crops'] == 1
    assert selected_image.grad is not None and selected_image.grad.abs().sum() > 0
    assert model.recurrent_memory.gate.weight.grad.abs().sum() > 0
    assert not history.streams


def test_history_rejects_missing_reset_and_future_encoder_selection():
    model = prepare_training(build_model(cfg()), backend='eager')
    rows = stream(model.cfg, length=3)[0]['feature_sequence']
    history = ObservationHistory()
    with pytest.raises(ValueError, match='Missing or out-of-order'):
        history.start(rows[1])
    history.start(rows[0])
    rows[0]['encoder_indices'][0, 0] = 0
    with pytest.raises(ValueError, match='precede'):
        history.reconstruct(model, rows[0], 'cpu')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
@pytest.mark.parametrize('encoder', ['conv', 'patch4'])
def test_cuda_sparse_collection_replay_gradients_and_graph_reuse(encoder):
    from torch._dynamo.utils import counters
    from vesuvius.neural_tracing.fiber_follow.regression.train import conv_memory_format
    torch.manual_seed(734)
    reference = build_model(cfg(encoder=encoder, memory_steps=2, recurrent_refinement_steps=1)).to('cuda', memory_format=conv_memory_format('cuda'))
    compiled = copy.deepcopy(reference)
    chunks = stream(reference.cfg, length=7, batch=2)
    results = []
    for model in (prepare_training(reference, backend='eager'), prepare_training(compiled)):
        ema = copy.deepcopy(reference)
        opt = torch.optim.SGD(model.parameters(), lr=0.)
        for repeat in range(2):
            before = counters['stats']['unique_graphs']
            result = optimizer_update(model, ema, opt, chunks, 1, 0., device='cuda',
                compute_metrics=False, memory_grad_clip=0., rest_grad_clip=0.)
            if repeat:
                assert counters['stats']['unique_graphs'] == before
        results.append(result)
    assert results[0]['loss'] == pytest.approx(results[1]['loss'], rel=.02, abs=.002)
    assert results[1]['supervised_states'] == 6
    assert results[1]['observation_only_states'] == 8
    for prefix in ('encoder.', 'decoder.', 'recurrent_memory.', 'confidence_scorer.'):
        gradients = [torch.cat([p.grad.flatten() for name, p in m.named_parameters()
                               if name.startswith(prefix) and p.grad is not None]).float()
                     for m in (reference, compiled)]
        assert all(torch.isfinite(g).all() for g in gradients)
        assert torch.nn.functional.cosine_similarity(*gradients, dim=0) > .99
        assert (gradients[1].norm()/gradients[0].norm()).item() == pytest.approx(1., rel=.05)
