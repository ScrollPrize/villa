"""Shared-decoder refinement, differentiable caches and checkpoint migration."""
import copy
import hashlib
import json
from dataclasses import replace

import pytest
import torch

from test_trajectory_memory import cfg, memory_batch, state_from, training_chunk
from vesuvius.neural_tracing.fiber_follow.regression.model import build_model, PathDecoderLayer
from vesuvius.neural_tracing.fiber_follow.regression.supervision import loss_terms
from vesuvius.neural_tracing.fiber_follow.regression.train import (
    checkpoint_config, optimizer_update, build_parser, compile_training_model,
)
from vesuvius.neural_tracing.fiber_follow.shared.runloop import resume_training, training_rng_state


def test_cached_decoder_matches_recomputed_forward_and_gradients():
    torch.manual_seed(41)
    layer = PathDecoderLayer(16, 2, 32, dropout=0., batch_first=True, norm_first=True)
    other = copy.deepcopy(layer)
    memory = torch.randn(2, 23, 16, requires_grad=True)
    copied = memory.detach().clone().requires_grad_()
    x = torch.randn(2, 4, 16)
    padding = torch.zeros(2, 23, dtype=torch.bool)
    padding[0, 17:] = True
    kv = other.project_memory(copied)
    a = layer(x, memory, memory_key_padding_mask=padding)
    a = layer(a, memory, memory_key_padding_mask=padding)
    b = other.forward_cached(x, kv, padding)
    b = other.forward_cached(b, kv, padding)
    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
    a.square().mean().backward()
    b.square().mean().backward()
    torch.testing.assert_close(memory.grad, copied.grad, atol=1e-7, rtol=1e-4)
    for p, q in zip(layer.parameters(), other.parameters()):
        torch.testing.assert_close(p.grad, q.grad, atol=1e-6, rtol=1e-4)


def test_refinement_starts_neutral_encodes_writes_and_projects_once(monkeypatch):
    torch.manual_seed(42)
    old = build_model(cfg()).eval()
    model = build_model(replace(old.cfg, recurrent_refinement_steps=2)).eval()
    model.load_state_dict(old.state_dict(), strict=False)
    b = memory_batch(model.cfg)
    calls = dict(encoder=0, writer=0, projection=0, decoder=0)
    for obj, method, key in ((model.encoder, 'encode', 'encoder'),
                             (model.recurrent_memory, 'observe_tokens', 'writer'),
                             (model.decoder.layers[0], 'project_memory', 'projection'),
                             (model.decoder.layers[0], 'forward_cached', 'decoder')):
        original = getattr(obj, method)
        def count(*args, _original=original, _key=key, **kwargs):
            calls[_key] += 1
            return _original(*args, **kwargs)
        monkeypatch.setattr(obj, method, count)
    expected = old(b['x'], b['hist'], b['hmask'])
    out = model(b['x'], b['hist'], b['hmask'])
    assert calls == dict(encoder=1, writer=1, projection=1, decoder=3)
    torch.testing.assert_close(out['points'], expected['points'], atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(out['points'], out['initial_points'], atol=1e-6, rtol=1e-5)
    assert out['refinement_points'].shape[1] == 3
    for key in state_from(model, out):
        torch.testing.assert_close(out['memory_'+key], expected['memory_'+key], rtol=0, atol=0)


def test_refinement_geometry_reaches_proposal_stage_fusion_and_old_observations():
    torch.manual_seed(43)
    model = build_model(cfg(recurrent_refinement_steps=1))
    torch.nn.init.normal_(model.refinement_delta.weight, std=.03)
    earlier = memory_batch(model.cfg, step=0)
    b = memory_batch(model.cfg, step=1)
    earlier['x']['fine'].requires_grad_()
    old = model(earlier['x'], earlier['hist'], earlier['hmask'])
    out = model(b['x'], b['hist'], b['hmask'], memory=state_from(model, old))
    gradient, = torch.autograd.grad(out['points'].square().mean(), out['initial_points'], retain_graph=True)
    assert gradient.abs().sum() > 0
    loss_terms(out, b, model.cfg)['geometry_per_state'].mean().backward()
    assert earlier['x']['fine'].grad.abs().sum() > 0
    for name, p in model.named_parameters():
        if name.startswith(('refinement_', 'coordinates.')):
            assert p.grad is not None and p.grad.abs().sum() > 0, name
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_bounds_and_masked_geometry():
    c = cfg(recurrent_refinement_steps=2, max_recovery_distance=2.)
    model = build_model(c)
    with torch.no_grad():
        model.refinement_delta.bias.fill_(100.)
    b = memory_batch(c)
    b['offtrack'][0] = 1
    b['dense_mask'][1].zero_()
    out = model(b['x'], b['hist'], b['hmask'])
    curves = out['refinement_points']
    assert curves[..., :2].diff(dim=1).norm(dim=-1).max() <= c.recurrent_refinement_limit+1e-5
    assert curves[:, :, 0].norm(dim=-1).max() <= c.max_recovery_distance+1e-5
    assert curves[..., :2].abs().max() <= c.lateral_limit
    torch.testing.assert_close(curves[..., 2], model.planes.expand_as(curves[..., 2]))
    loss = loss_terms(out, b, c)['geometry_per_state']
    assert loss.eq(0).all()
    loss.sum().backward()
    assert model.refinement_delta.weight.grad.eq(0).all()




def test_refinement_fullgraph_capture():
    model = build_model(cfg(recurrent_refinement_steps=1))
    b = memory_batch(model.cfg)
    wrapped = torch.compile(model, backend='eager', fullgraph=True)
    out = wrapped(b['x'], b['hist'], b['hmask'])
    loss_terms(out, b, model.cfg)['geometry_per_state'].mean().backward()
    assert torch.isfinite(model.refinement_delta.weight.grad).all()




@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_compiled_refinement_bf16_two_decision_gradients():
    from vesuvius.neural_tracing.fiber_follow.regression.train import move_batch
    torch.manual_seed(45)
    c = cfg(recurrent_refinement_steps=1)
    eager = build_model(c).cuda()
    torch.nn.init.normal_(eager.refinement_delta.weight, std=.03)
    compiled = copy.deepcopy(eager)
    chunk = [move_batch(b, 'cuda') for b in training_chunk(c)['feature_sequence']]
    losses = []
    for model in (eager, compile_training_model(compiled)):
        state = None
        loss = 0
        with torch.autocast('cuda', dtype=torch.bfloat16):
            for b in chunk:
                candidates = b['hist'].new_zeros(2, 2, c.n_future, 3)
                candidates[..., 2] = eager.planes
                candidates[:, 1, :, 0] = 2.
                b['candidate_mask'] = torch.ones_like(candidates[..., 0], dtype=torch.bool)
                b['candidate_labels'] = torch.zeros_like(candidates[..., 0])
                b['candidate_labels'][:, 0] = 1.
                out = model(b['x'], b['hist'], b['hmask'], memory=state,
                            candidates=candidates)
                state = state_from(eager, out)
                terms = loss_terms(out, b, c)
                loss = (loss+terms['geometry_per_state'].mean()+terms['confidence_per_state'].mean()
                        +out['candidate_confidence_logits'].square().mean())
        losses.append(loss.detach())
        loss.backward()
    torch.testing.assert_close(*losses, atol=.005, rtol=.02)
    for prefix in ('encoder.', 'decoder.', 'refinement_', 'recurrent_memory.'):
        grads = [torch.cat([p.grad.flatten() for name, p in model.named_parameters()
                           if name.startswith(prefix) and p.grad is not None]) for model in (eager, compiled)]
        assert all(torch.isfinite(g).all() for g in grads)
        assert torch.nn.functional.cosine_similarity(*grads, dim=0).item() > .99
        assert (grads[1].norm()/grads[0].norm()).item() == pytest.approx(1., rel=.05)
