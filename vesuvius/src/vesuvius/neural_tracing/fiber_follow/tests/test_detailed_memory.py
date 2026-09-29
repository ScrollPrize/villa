import copy
from dataclasses import replace

import numpy as np
import pytest
import torch

from test_trajectory_memory import cfg, memory_batch, state_from, training_chunk
from vesuvius.neural_tracing.fiber_follow.regression.model import build_model
from vesuvius.neural_tracing.fiber_follow.regression.stratified_replay import stratified_indices
from vesuvius.neural_tracing.fiber_follow.regression.train import optimizer_update
from vesuvius.neural_tracing.fiber_follow.regression.feature_sequences import FeatureStreamStates


def config(**kwargs):
    return cfg(feature_memory_revision=2, feature_detail_tokens=4, recurrent_refinement_steps=1, **kwargs)


def test_cache_influences_admission_and_writer_but_not_cached_observation():
    torch.manual_seed(51)
    model = build_model(config()).eval()
    b = memory_batch(model.cfg, 1)
    a = model(b['x'], b['hist'], b['hmask'])
    state = {k: v.detach() for k, v in state_from(model, a).items()}
    altered = {k: v.clone() for k, v in state.items()}
    altered['cache'] += torch.randn_like(altered['cache'])*3
    b['x']['feature_seed_here'].zero_()
    a = model(b['x'], b['hist'], b['hmask'], memory=state)
    z = model(b['x'], b['hist'], b['hmask'], memory=altered)
    assert (a['memory_probe']-z['memory_probe']).abs().max() > 1e-7
    assert (a['memory_slots']-z['memory_slots']).abs().max() > 1e-7
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
    assert a['memory_cache_admission'].shape == (1, model.cfg.memory_steps)


def test_candidate_queries_are_candidate_specific_and_read_memory():
    model = build_model(config()).eval()
    torch.nn.init.normal_(model.confidence_memory_head[-1].weight, std=.1)
    b = memory_batch(model.cfg, 1)
    curves = torch.zeros(1, 2, model.cfg.n_future, 3)
    curves[..., 2] = model.planes
    curves[:, 1, :, 0] = 2
    queries = []
    hook = model.confidence_memory_attention.register_forward_pre_hook(lambda m, args: queries.append(args[0]))
    out = model(b['x'], b['hist'], b['hmask'], candidates=curves)
    hook.remove()
    assert len(queries) == 3
    assert not torch.equal(queries[1], queries[2])
    out['candidate_confidence_logits'].sum().backward()
    assert model.confidence_memory_attention.in_proj_weight.grad.abs().sum() > 0


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
        state.pop('probe')
    state['slots'].square().sum().backward()
    assert first['x']['fine'].grad.abs().sum() > 0


def test_replay_across_updates_keeps_selected_crops_only_and_evicts_finished_streams():
    torch.manual_seed(54)
    model = build_model(config())
    ema, opt, states = copy.deepcopy(model), torch.optim.AdamW(model.parameters()), FeatureStreamStates()
    for start in (0, 2, 4):
        chunk = training_chunk(model.cfg, start=start, end=start == 4)
        for i, b in enumerate(chunk['feature_sequence']):
            b['replay_select'] = torch.full((2,), start+i in (0, 3))
        metrics = optimizer_update(model, ema, opt, [chunk], start+1, .001, compute_metrics=False, stream_states=states)
        assert np.isfinite(metrics['loss'])
        if start < 4:
            assert not metrics['replay_endpoints']
            assert all(not v.requires_grad for rows in states.replay.streams.values() for row in rows for v in row['features'])
    assert metrics['replay_endpoints'] == 2 and metrics['replay_encoder_crops'] == 6
    assert not states.states and not states.replay.streams and not states.replay.pending


def test_revision2_fullgraph_backward():
    model = build_model(config())
    b = memory_batch(model.cfg, 1)
    out = torch.compile(model, backend='eager', fullgraph=True)(b['x'], b['hist'], b['hmask'])
    out['points'].square().mean().backward()
    assert model.encoder_memory_projection.weight.grad.abs().sum() > 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_upgrade_preserves_existing_tensors_moments_and_rng():
    from vesuvius.neural_tracing.fiber_follow.regression.upgrade_feature_memory import upgrade_checkpoint
    from vesuvius.neural_tracing.fiber_follow.shared.runloop import training_rng_state
    model = build_model(cfg(recurrent_refinement_steps=1))
    opt = torch.optim.AdamW(model.parameters())
    optimizer_update(model, copy.deepcopy(model), opt, [training_chunk(model.cfg)], 1, .001, compute_metrics=False)
    ck = dict(architecture=model.architecture, model_cfg=model.cfg.to_dict(), model=model.state_dict(),
              ema=model.state_dict(), optimizer=opt.state_dict(), rng=training_rng_state(), step=1)
    upgraded = upgrade_checkpoint(ck)
    new = build_model(replace(model.cfg, **{k: upgraded['model_cfg'][k] for k in
                      ('feature_memory_revision', 'feature_detail_tokens', 'feature_stream_steps', 'feature_replay_weight')}))
    new.load_state_dict(upgraded['model'], strict=True)
    new_opt = torch.optim.AdamW(new.parameters())
    new_opt.load_state_dict(upgraded['optimizer'])
    old_params = dict(model.named_parameters())
    for name, value in ck['model'].items():
        torch.testing.assert_close(upgraded['model'][name], value, rtol=0, atol=0)
        torch.testing.assert_close(upgraded['ema'][name], value, rtol=0, atol=0)
    for name, p in new.named_parameters():
        if name in old_params:
            for key, value in opt.state[old_params[name]].items():
                torch.testing.assert_close(new_opt.state[p][key], value, rtol=0, atol=0)
        else:
            assert not new_opt.state.get(p)
    assert upgraded['rng'] is ck['rng'] and upgraded['step'] == ck['step']


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_revision2_compiled_bf16_gradients_match_eager():
    from vesuvius.neural_tracing.fiber_follow.regression.train import compile_training_model, move_batch
    torch.manual_seed(57)
    eager = build_model(config()).cuda()
    torch.nn.init.normal_(eager.confidence_memory_head[-1].weight, std=.03)
    compiled = copy.deepcopy(eager)
    b = move_batch(memory_batch(eager.cfg, 2), 'cuda')
    candidates = b['hist'].new_zeros(2, 2, eager.cfg.n_future, 3)
    candidates[..., 2] = eager.planes
    candidates[:, 1, :, 0] = 2
    losses = []
    for model in (eager, compile_training_model(compiled)):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = model(b['x'], b['hist'], b['hmask'], candidates=candidates)
            state = state_from(eager, out)
            out = model(b['x'], b['hist'], b['hmask'], candidates=candidates, memory=state)
            loss = out['points'].square().mean()+out['candidate_confidence_logits'].square().mean()+out['memory_probe'].square().mean()
        losses.append(loss.detach())
        loss.backward()
    torch.testing.assert_close(*losses, atol=.01, rtol=.02)
    for prefix in ('encoder.', 'decoder.', 'recurrent_memory.', 'encoder_memory_', 'confidence_memory_'):
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
