import copy

import numpy as np
import pytest
import torch

from test_trajectory_memory import cfg, memory_batch, state_from, training_chunk
from vesuvius.neural_tracing.fiber_follow.regression.model import build_model
from vesuvius.neural_tracing.fiber_follow.regression.stratified_replay import stratified_indices
from vesuvius.neural_tracing.fiber_follow.regression.train import optimizer_update, prepare_training
from vesuvius.neural_tracing.fiber_follow.regression.stratified_replay import ObservationHistory


def config(**kwargs):
    return cfg(feature_detail_tokens=4, recurrent_refinement_steps=1, **kwargs)


def test_history_conditions_fine_features_and_path_without_rewriting_observations():
    torch.manual_seed(51)
    model = build_model(config()).eval()
    b = memory_batch(model.cfg, 1)
    a = model(b['x'], b['hist'], b['hmask'])
    state = {k: v.detach() for k, v in state_from(model, a).items()}
    altered = {k: v.clone() for k, v in state.items()}
    altered['cache'] += torch.randn_like(altered['cache'])*3
    b['x']['feature_seed_here'].zero_()
    fine = []
    hook = model.encoder.dense_decoder.register_forward_hook(lambda m, args, result: fine.append(result))
    a = model(b['x'], b['hist'], b['hmask'], memory=state)
    z = model(b['x'], b['hist'], b['hmask'], memory=altered)
    hook.remove()
    assert len(fine) == 2  # One dense decode per observed crop.
    assert (fine[0]-fine[1]).abs().max() > 1e-7
    assert not any('probe' in k or 'admission' in k for k in model.state_dict())
    assert not any('probe' in k or 'admission' in k for k in a)
    # Slot compression uses incoming slots and the observation, not retrieved beliefs.
    torch.testing.assert_close(a['memory_slots'], z['memory_slots'], rtol=0, atol=0)
    torch.testing.assert_close(a['observation_tokens'], z['observation_tokens'], rtol=0, atol=0)
    torch.testing.assert_close(a['memory_cache'][:, -1], z['memory_cache'][:, -1], rtol=0, atol=0)
    assert (a['points']-z['points']).abs().max() > 1e-7


def test_detail_padding_and_remote_pose():
    model = build_model(config()).eval()
    b = memory_batch(model.cfg, 1)
    b['hmask'].zero_()
    a = model(b['x'], b['hist'], b['hmask'])
    state = state_from(model, a)
    n = model.recurrent_memory.coarse_count
    assert state['cache_mask'][0, -1, n:].tolist() == [True, False, False, False]
    tokens, padding = model.recurrent_memory.read_tokens(state)
    assert padding[0, -3:].all()
    old_anchor = state['anchor'].clone()
    b['x']['feature_seed_here'].zero_()
    b['x']['query_position'] += 100
    a = model(b['x'], b['hist'], b['hmask'], memory=state)
    torch.testing.assert_close(a['memory_anchor'], old_anchor)
    assert a['memory_cache'].shape[1] == model.cfg.memory_steps


def test_candidate_queries_are_candidate_specific_and_read_memory():
    model = build_model(config()).eval()
    b = memory_batch(model.cfg, 1)
    curves = torch.zeros(1, 2, model.cfg.n_future, 3)
    curves[..., 2] = model.planes
    curves[:, 1, :, 0] = 2
    queries = []
    hook = model.confidence_scorer.query.register_forward_hook(lambda m, args, result: queries.append(result))
    out = model(b['x'], b['hist'], b['hmask'], candidates=curves)
    hook.remove()
    assert len(queries) == model.cfg.recurrent_refinement_steps+3
    assert not torch.equal(queries[-2], queries[-1])
    out['candidate_confidence_logits'].sum().backward()
    assert model.confidence_scorer.layers[0].multihead_attn.in_proj_weight.grad.abs().sum() > 0


def test_stratification_reproducible_and_covers_evicted_evidence():
    a = stratified_indices(130, np.random.default_rng(1))
    assert a == stratified_indices(130, np.random.default_rng(1))
    ages = sorted(129-i for i in a)
    assert len(ages) == 3 and 1 <= ages[0] <= 4 and 5 <= ages[1] <= 16 and 17 <= ages[2] <= 129
    assert any(129-min(stratified_indices(130, np.random.default_rng(seed))) > 64 for seed in range(20))


def test_evicted_crop_receives_gradient_through_all_writer_updates():
    torch.manual_seed(53)
    model = build_model(config(memory_steps=2))
    first = memory_batch(model.cfg, 1)
    first['x']['fine'].requires_grad_()
    state = None
    for t in range(7):
        b = first if t == 0 else memory_batch(model.cfg, 1, step=t)
        if t == 0:
            # No immutable anchor shortcut: old evidence must survive the writer.
            b['x']['feature_seed_here'].zero_()
        features = model.observation_features(b['x'], b['hist'], b['hmask'])
        if t > 0:
            features = tuple(v.detach() for v in features)
        state, _ = model.recurrent_memory.observe_tokens(*features, b['x'], state)
    state['slots'].square().sum().backward()
    assert first['x']['fine'].grad.abs().sum() > 0


def test_replay_across_updates_keeps_selected_crops_only_and_evicts_finished_streams():
    torch.manual_seed(54)
    model = build_model(config())
    ema, opt, states = copy.deepcopy(model), torch.optim.AdamW(model.parameters()), ObservationHistory()
    prepare_training(model, backend='eager')
    for start in (0, 2, 4):
        chunk = training_chunk(model.cfg, start=start, end=start == 4)
        for i, b in enumerate(chunk['feature_sequence']):
            b['decision_mask'] = b['stream_end'].clone()
            b['loss_weight'] = b['decision_mask'].float()
            b['retain_until'] = torch.full((2,), 5 if start+i in (0, 3) else -1)
            b['encoder_indices'] = torch.tensor([[0, 3, -1]]).expand(2, -1).clone() if start+i == 5 else torch.full((2, 3), -1)
        metrics = optimizer_update(model, ema, opt, [chunk], start+1, .001, compute_metrics=False, stream_states=states)
        assert np.isfinite(metrics['loss'])
        if start < 4:
            assert not metrics['supervised_states'] and not metrics['optimizer_applied']
            assert not opt.state
            assert all(not v.requires_grad for rows in states.streams.values() for row in rows for v in row['features'])
    assert metrics['supervised_states'] == 2 and metrics['history_encoder_crops'] == 4
    assert not states.streams


def test_memory_with_adaptive_refinement_compiled_backward():
    model = build_model(config())
    b = memory_batch(model.cfg, 1)
    from vesuvius.neural_tracing.fiber_follow.regression.train import prepare_training, training_prediction
    prepare_training(model, backend='eager')
    out = training_prediction(model, b['x'], b['hist'], b['hmask'])
    out['points'].square().mean().backward()
    assert model.encoder_memory_projection.weight.grad.abs().sum() > 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize('scored', ['confidence_logits', 'candidate_confidence_logits'])
def test_confidence_has_no_gradient_to_path_generator_but_trains_encoder_and_memory(scored):
    torch.manual_seed(56)
    model = build_model(config()).eval()
    first = memory_batch(model.cfg, 1)
    first['x']['fine'].requires_grad_()
    state = state_from(model, model(first['x'], first['hist'], first['hmask']))
    b = memory_batch(model.cfg, 1, step=1)
    curves = b['hist'].new_zeros(1, 2, model.cfg.n_future, 3)
    curves[..., 2] = model.planes
    curves[:, 1, :, 0] = 2
    out = model(b['x'], b['hist'], b['hmask'], candidates=curves, memory=state)
    out[scored].sum().backward()
    for name, p in model.named_parameters():
        if name.startswith(('coordinates.', 'decoder.', 'query.', 'refinement_')):
            assert p.grad is None, name
    for p in (model.encoder.stem[0].weight, model.encoder.dense_decoder[-1].weight,
              model.encoder_memory_projection.weight, model.recurrent_memory.gate.weight,
              model.confidence_scorer.layers[0].multihead_attn.in_proj_weight):
        assert p.grad is not None and p.grad.abs().sum() > 0
    assert first['x']['fine'].grad.abs().sum() > 0


def test_fixed_candidate_scores_do_not_change_when_generator_changes():
    torch.manual_seed(58)
    model = build_model(config()).eval()
    b = memory_batch(model.cfg, 1)
    curves = b['hist'].new_zeros(1, 2, model.cfg.n_future, 3)
    curves[..., 2] = model.planes
    curves[:, 1, :, 0] = 2
    with torch.no_grad():
        a = model(b['x'], b['hist'], b['hmask'], candidates=curves)
        model.coordinates.bias.add_(.5)
        model.refinement_fusion[-1].weight.normal_()
        model.decoder.layers[0].linear1.weight.normal_()
        z = model(b['x'], b['hist'], b['hmask'], candidates=curves)
    assert (a['points']-z['points']).abs().max() > .1
    torch.testing.assert_close(a['candidate_confidence_logits'], z['candidate_confidence_logits'], rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_revision2_compiled_bf16_gradients_match_eager():
    from vesuvius.neural_tracing.fiber_follow.regression.train import prepare_training, move_batch, training_prediction
    torch.manual_seed(57)
    eager = build_model(config()).cuda()
    compiled = copy.deepcopy(eager)
    b = move_batch(memory_batch(eager.cfg, 2), 'cuda')
    candidates = b['hist'].new_zeros(2, 2, eager.cfg.n_future, 3)
    candidates[..., 2] = eager.planes
    candidates[:, 1, :, 0] = 2
    b['candidate_mask'] = torch.ones_like(candidates[..., 0], dtype=torch.bool)
    b['candidate_labels'] = torch.zeros_like(candidates[..., 0])
    b['candidate_labels'][:, 0] = 1.
    losses = []
    predictions = []
    for model in (eager, prepare_training(compiled)):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            forward = (lambda *a, **kw: training_prediction(model, *a, **kw)) if model is compiled else model
            out = forward(b['x'], b['hist'], b['hmask'], candidates=candidates)
            state = state_from(eager, out)
            out = forward(b['x'], b['hist'], b['hmask'], candidates=candidates, memory=state)
            # Train every actual proposal, as the trainer does. Differentiating
            # only an argmax-selected proposal makes this numerical-gradient
            # test discontinuous at nearly tied BF16 confidence scores.
            from vesuvius.neural_tracing.fiber_follow.regression.supervision import loss_terms
            loss = (loss_terms(out, b, eager.cfg)['geometry_per_state'].mean()
                    +out['candidate_confidence_logits'].square().mean())
        losses.append(loss.detach())
        predictions.append({key: out[key].detach() for key in
                            ('refinement_points', 'refinement_confidence', 'refinement_mask')})
        loss.backward()
    torch.testing.assert_close(*losses, atol=.01, rtol=.02)
    for key in predictions[0]:
        torch.testing.assert_close(predictions[0][key], predictions[1][key], atol=.01, rtol=.02)
    for prefix in ('encoder.', 'decoder.', 'recurrent_memory.', 'encoder_memory_', 'confidence_scorer.'):
        grads = [torch.cat([p.grad.flatten() for name, p in model.named_parameters()
                           if name.startswith(prefix) and p.grad is not None]) for model in (eager, compiled)]
        assert all(torch.isfinite(g).all() for g in grads)
        assert torch.nn.functional.cosine_similarity(*grads, dim=0) > .99
        assert (grads[1].norm()/grads[0].norm()).item() == pytest.approx(1., rel=.05)


def test_revision2_cold_seed_and_live_trace(monkeypatch):
    import test_trajectory_memory as existing
    # Exercise the established production input builder and tracer with the new
    # state schema, including a remote seed encoded once and active trace carry.
    monkeypatch.setattr(existing, 'cfg', config)
    existing.test_cold_remote_seed_is_encoded_once_and_warm_trace_reads_only_current_crop(monkeypatch)


def test_revision2_streams_mark_stratified_observations(monkeypatch, tmp_path):
    import test_trajectory_memory as existing
    monkeypatch.setattr(existing, 'cfg', config)
    existing.test_builder_streams_causal_main_crops_and_keeps_paired_endpoints_identical(tmp_path, monkeypatch)
