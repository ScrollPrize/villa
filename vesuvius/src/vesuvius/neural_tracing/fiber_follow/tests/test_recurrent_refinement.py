"""Shared absolute-coordinate prediction, differentiable caches and feedback."""
import copy
import hashlib
import json
from dataclasses import replace

import pytest
import torch

from slab_fixtures import cfg, slab_batch as memory_batch
from vesuvius.neural_tracing.fiber_follow.regression.model import build_model, PathDecoderLayer
from vesuvius.neural_tracing.fiber_follow.regression.supervision import loss_terms
from vesuvius.neural_tracing.fiber_follow.regression.train import (
    checkpoint_config, optimizer_update, build_parser, prepare_training, training_prediction,
)
from vesuvius.neural_tracing.fiber_follow.shared.runloop import resume_training, training_rng_state
from label_fixtures import set_terminal, state_labels


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


def test_passes_share_coordinate_head_and_encode_write_project_once(monkeypatch):
    torch.manual_seed(42)
    model = build_model(cfg(recurrent_refinement_steps=2)).eval()
    b = memory_batch(model.cfg)
    calls = dict(encoder=0, writer=0, projection=0, scorer_projection=0, decoder=0, coordinates=0, scoring=0)
    for obj, method, key in ((model.encoder, 'encode', 'encoder'),
                             (model.history_encoder, 'forward', 'writer'),
                             (model.decoder.layers[0], 'project_memory', 'projection'),
                             (model.confidence_scorer.layers[0], 'project_memory', 'scorer_projection'),
                             (model.decoder.layers[0], 'forward_cached', 'decoder'),
                             (model.coordinates, 'forward', 'coordinates'),
                             (model, 'hazard_logits', 'scoring')):
        original = getattr(obj, method)
        def count(*args, _original=original, _key=key, **kwargs):
            calls[_key] += 1
            return _original(*args, **kwargs)
        monkeypatch.setattr(obj, method, count)
    out = model(b['x'], b['hist'], b['hmask'])
    assert calls == dict(encoder=1, writer=1, projection=1, scorer_projection=1, decoder=3, coordinates=3, scoring=3)
    assert not hasattr(model, 'refinement_delta')
    assert out['refinement_points'].shape[1] == 3
    assert out['refinement_hazard_logits'].shape == (len(b['hist']), 3, model.cfg.n_future)




def test_bounds_and_masked_geometry():
    c = cfg(recurrent_refinement_steps=2, max_recovery_distance=2.)
    model = build_model(c)
    with torch.no_grad():
        model.coordinates.bias.fill_(100.)
    b = memory_batch(c)
    set_terminal(b, 0)
    b['dense_mask'][1].zero_()
    out = model(b['x'], b['hist'], b['hmask'])
    curves = out['refinement_points']
    assert curves[:, :, 0].norm(dim=-1).max() <= c.max_recovery_distance+1e-5
    assert curves[..., :2].abs().max() <= c.lateral_limit
    torch.testing.assert_close(curves[..., 2], model.planes.expand_as(curves[..., 2]))
    loss = loss_terms(out, b, c)['geometry_per_state']
    assert loss.eq(0).all()
    loss.sum().backward()
    assert model.coordinates.weight.grad.eq(0).all()




def test_adaptive_refinement_compiles_and_backpropagates():
    model = build_model(cfg(recurrent_refinement_steps=1))
    b = memory_batch(model.cfg)
    prepare_training(model, backend='eager')
    out = training_prediction(model, b['x'], b['hist'], b['hmask'])
    loss_terms(out, b, model.cfg)['geometry_per_state'].mean().backward()
    assert torch.isfinite(model.coordinates.weight.grad).all()




