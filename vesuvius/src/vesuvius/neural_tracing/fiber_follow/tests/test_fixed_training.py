"""Fixed compiler contracts without changing adaptive losses or stream gradients."""
import copy

import pytest
import torch

from test_trajectory_memory import cfg, memory_batch, state_from, training_chunk
from vesuvius.neural_tracing.fiber_follow.regression.model import build_model
from vesuvius.neural_tracing.fiber_follow.regression.train import compile_training_model, optimizer_update
from vesuvius.neural_tracing.fiber_follow.regression.supervision import loss_terms
from vesuvius.neural_tracing.fiber_follow.regression.feature_sequences import FeatureStreamStates
from vesuvius.neural_tracing.fiber_follow.regression.compiled_training import pack_feature_chunks


def compare_gradients(left, right):
    for (name, p), (_, q) in zip(left.named_parameters(), right.named_parameters()):
        if p.grad is None:
            assert q.grad is None, name
        else:
            torch.testing.assert_close(p.grad, q.grad, rtol=3e-4, atol=2e-6, msg=name)


def test_one_crop_graph_across_rows_memory_candidates_and_acceptance():
    torch.manual_seed(111)
    eager = build_model(cfg(recurrent_refinement_steps=2))
    raw = copy.deepcopy(eager)
    graphs = []
    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward
    compiled = compile_training_model(raw, backend=backend)
    for count, detached, scored, threshold in (
            (2, False, False, 1.), (1, True, True, 0.),
            (1, False, False, .5), (2, True, True, .5),
            (2, False, True, 0.), (1, True, False, 1.)):
        batch = memory_batch(eager.cfg, count)
        candidates = batch['hist'].new_zeros(count, 4, eager.cfg.n_future, 3)
        candidates[..., 2] = eager.planes
        batch['candidate_mask'] = torch.ones(count, 4, eager.cfg.n_future)
        batch['candidate_labels'] = torch.ones_like(batch['candidate_mask'])
        outputs = []
        for model in (eager, compiled):
            model.zero_grad(set_to_none=True)
            if model is compiled:
                model.begin_update()
            old = model(batch['x'], batch['hist'], batch['hmask'], confidence_threshold=threshold)
            state = state_from(eager, old)
            if detached:
                state = {key: value.detach() for key, value in state.items()}
            out = model(batch['x'], batch['hist'], batch['hmask'], memory=state,
                        candidates=candidates if scored else None, confidence_threshold=threshold)
            terms = loss_terms(out, batch, eager.cfg)
            loss = terms['geometry_per_state'].sum()+.5*terms['confidence_per_state'].sum()
            if scored:
                loss = loss+out['candidate_hazard_logits'].square().mean()
            loss.backward()
            if model is compiled:
                model.finish_update()
            outputs.append(out)
        for name in ('points', 'confidence', 'hazard_logits'):
            torch.testing.assert_close(outputs[0][name], outputs[1][name], rtol=1e-5, atol=1e-6)
        used = outputs[0]['refinement_mask'].shape[1]
        assert torch.equal(outputs[0]['refinement_mask'], outputs[1]['refinement_mask'][:, :used])
        assert not outputs[1]['refinement_mask'][:, used:].any()
        compare_gradients(eager, raw)
    assert len(graphs) == 2  # One crop graph; one optional candidate-scoring graph.


def test_replay_has_one_encoder_and_one_transition_graph():
    torch.manual_seed(112)
    raw = build_model(cfg(recurrent_refinement_steps=2))
    graphs = []
    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward
    model = compile_training_model(raw, backend=backend)
    batch = memory_batch(raw.cfg, 1)
    pose = {key: batch['x'][key] for key in ('query_position', 'query_frame', 'feature_seed_here')}
    state = None
    for i in range(4):
        features = model.replay_observation_features(batch['x'], batch['hist'], batch['hmask'])
        if i % 2:
            features = tuple(value.detach() for value in features)
        state, _ = model.replay_transition(*features, pose, state)
    state['slots'].square().sum().backward()
    assert len(graphs) == 2
    assert raw.encoder.stem[0].weight.grad.abs().sum() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cuda_replay_checkpoint_gradients_and_graph_reuse():
    from torch._dynamo.utils import counters
    from vesuvius.neural_tracing.fiber_follow.regression.train import move_batch
    torch.manual_seed(116)
    eager = build_model(cfg(recurrent_refinement_steps=2)).cuda()
    raw = copy.deepcopy(eager)
    compiled = compile_training_model(raw)
    batch = move_batch(memory_batch(raw.cfg, 1), 'cuda')
    pose = {key: batch['x'][key] for key in ('query_position', 'query_frame', 'feature_seed_here')}
    before = counters['stats']['unique_graphs']
    losses = []
    for model in (eager, compiled):
        transition = (model.replay_transition if model is compiled else
                      model.recurrent_memory.observe_tokens)
        # Backward recomputes the encoder, then a second replay must reuse it.
        for repeat in range(2):
            model.zero_grad(set_to_none=True)
            state = None
            with torch.autocast('cuda', dtype=torch.bfloat16):
                for i in range(3):
                    features = model.replay_observation_features(batch['x'], batch['hist'], batch['hmask'])
                    if i == 1:
                        features = tuple(value.detach() for value in features)
                    state, retrieved = transition(*features, pose, state)
                loss = state['slots'].square().mean()+retrieved.square().mean()
            loss.backward()
        losses.append(loss.detach())
    assert counters['stats']['unique_graphs']-before == 2
    torch.testing.assert_close(*losses, atol=.01, rtol=.02)
    for prefix in ('encoder.', 'recurrent_memory.'):
        grads = [torch.cat([p.grad.flatten() for name, p in model.named_parameters()
                            if name.startswith(prefix) and p.grad is not None]) for model in (eager, raw)]
        assert torch.nn.functional.cosine_similarity(*grads, dim=0) > .99
        torch.testing.assert_close(grads[0].norm(), grads[1].norm(), rtol=.05, atol=1e-6)


def test_partial_acceptance_keeps_policy_and_geometry_gradients():
    torch.manual_seed(115)
    eager = build_model(cfg(recurrent_refinement_steps=2))
    raw = copy.deepcopy(eager)
    compiled = compile_training_model(raw, backend='eager')
    batch = memory_batch(eager.cfg, 2)
    with torch.no_grad():
        confidence = eager(batch['x'], batch['hist'], batch['hmask'])['refinement_confidence'][:, 0, -1]
    assert confidence[0] != confidence[1]
    threshold = float(confidence.mean())
    predictions = []
    for model in (eager, compiled):
        out = model(batch['x'], batch['hist'], batch['hmask'], confidence_threshold=threshold)
        terms = loss_terms(out, batch, eager.cfg)
        (terms['geometry_per_state'].sum()+terms['confidence_per_state'].sum()).backward()
        predictions.append(out)
    assert predictions[0]['refinement_mask'][:, 1].sum() == 1
    used = predictions[0]['refinement_mask'].shape[1]
    assert torch.equal(predictions[0]['refinement_mask'], predictions[1]['refinement_mask'][:, :used])
    assert torch.equal(predictions[0]['selected_refinement'], predictions[1]['selected_refinement'])
    torch.testing.assert_close(predictions[0]['points'], predictions[1]['points'], rtol=1e-5, atol=1e-6)
    compare_gradients(eager, raw)


def test_packing_keeps_real_rows_weights_and_chunk_boundaries():
    c = cfg()
    chunks = []
    for key, length in ((10, 2), (20, 1), (30, 2)):
        rows = [memory_batch(c, 1, t) for t in range(length)]
        for t, row in enumerate(rows):
            row.update(stream_id=torch.tensor([key]), stream_reset=torch.tensor([t == 0]),
                       stream_end=torch.tensor([t == length-1]), loss_weight=torch.tensor([.25+t*.5]))
        chunks.append(dict(feature_sequence=rows))
    packed = pack_feature_chunks(chunks, 2)
    def entries(items):
        return sorted((int(key), bool(reset), bool(end), float(weight))
            for chunk in items for batch in chunk['feature_sequence'] for key, reset, end, weight in
            zip(batch['stream_id'], batch['stream_reset'], batch['stream_end'], batch['loss_weight']))
    assert entries(packed) == entries(chunks)
    assert len(packed) == 2
    assert packed[0]['feature_sequence'][0]['stream_id'].tolist() == [10, 30]
    duplicate = chunks+[chunks[0]]
    assert pack_feature_chunks(duplicate, 2) is duplicate


def test_masked_retries_preserve_adamw_skipped_parameter_updates():
    torch.manual_seed(113)
    eager = build_model(cfg(recurrent_refinement_steps=2, feature_replay_weight=0.))
    raw = copy.deepcopy(eager)
    compiled = compile_training_model(raw, backend='eager')
    models = (eager, compiled)
    optimizers = [torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.1) for model in models]
    emas = [copy.deepcopy(eager), copy.deepcopy(raw)]
    # Populate identical momentum, then accept every initial proposal. A zero
    # grad would incorrectly decay parameters and advance optimizer moments.
    for model, opt in zip(models, optimizers):
        for module in (model.refinement_fusion, model.refinement_stage):
            for p in module.parameters():
                p.grad = torch.ones_like(p)
        opt.step()
    protected = {name: p.detach().clone() for name, p in raw.named_parameters() if name.startswith('refinement_')}
    data = [training_chunk(eager.cfg, end=True)]
    for model, opt, ema in zip(models, optimizers, emas):
        with torch.no_grad():
            model.confidence_scorer.failure.weight.zero_()
            model.confidence_scorer.failure.bias.fill_(-20.)
        optimizer_update(model, ema, opt, data, 2, .001, compute_metrics=False)
    compare_gradients(eager, raw)
    for name, p in raw.named_parameters():
        if name in protected:
            torch.testing.assert_close(p, protected[name], rtol=0, atol=0)
            assert optimizers[1].state[p]['step'] == 1
    assert all(p.grad is None for p in raw.refinement_fusion.parameters())


def test_packed_stream_gradients_and_detach_match_unpacked_training():
    torch.manual_seed(114)
    eager = build_model(cfg(recurrent_refinement_steps=2, feature_replay_weight=0.))
    raw = copy.deepcopy(eager)
    compiled = compile_training_model(raw, backend='eager')
    chunks = [training_chunk(eager.cfg)]
    # Separate independent rows into the partial chunks emitted by workers.
    partial = []
    def row(batch, j):
        return {key: row(value, j) if isinstance(value, dict) else value[j:j+1]
                for key, value in batch.items()}
    for j in range(2):
        partial.append(dict(feature_sequence=[row(batch, j) for batch in chunks[0]['feature_sequence']]))
    states = [FeatureStreamStates(), FeatureStreamStates()]
    metrics = []
    for model, carried in zip((eager, compiled), states):
        metrics.append(optimizer_update(model, copy.deepcopy(eager), torch.optim.AdamW(model.parameters()),
            partial, 1, 0., compute_metrics=False, stream_states=carried))
    assert metrics[0]['observed_states'] == metrics[1]['observed_states'] == 4
    assert metrics[0]['loss'] == pytest.approx(metrics[1]['loss'], rel=1e-5, abs=1e-6)
    compare_gradients(eager, raw)
    for key in states[0].states:
        for name, value in states[0].states[key].items():
            torch.testing.assert_close(value, states[1].states[key][name], rtol=1e-5, atol=1e-6)
            assert not states[1].states[key][name].requires_grad
