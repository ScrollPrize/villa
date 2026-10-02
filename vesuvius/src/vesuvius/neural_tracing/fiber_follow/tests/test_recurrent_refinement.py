"""Cached decoder attention and the work shared by every refinement pass."""
import copy

import torch

from model_fixtures import aligned_config, aligned_batch
from vesuvius.neural_tracing.fiber_follow.regression.model import build_model, PathDecoderLayer


def test_cached_decoder_matches_recomputed_forward_masks_keys_and_reuses_projections():
    torch.manual_seed(41)
    layer = PathDecoderLayer(16, 2, 32, dropout=0., batch_first=True, norm_first=True)
    other = copy.deepcopy(layer)
    memory = torch.randn(2, 23, 16, requires_grad=True)
    copied = memory.detach().clone().requires_grad_()
    x = torch.randn(2, 4, 16)
    padding = torch.zeros(2, 23, dtype=torch.bool)
    padding[0, 17:] = True
    padding[1, 3:9] = True
    kv = other.project_memory(copied)
    pointers = [v.data_ptr() for v in kv]
    a = b = x
    for _ in range(3):
        a = layer(a, memory, memory_key_padding_mask=padding)
        b = other.forward_cached(b, kv, padding)
    assert pointers == [v.data_ptr() for v in kv]
    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
    a.square().mean().backward()
    b.square().mean().backward()
    torch.testing.assert_close(memory.grad, copied.grad, atol=1e-7, rtol=1e-4)
    assert copied.grad[padding].eq(0).all()
    for p, q in zip(layer.parameters(), other.parameters()):
        torch.testing.assert_close(p.grad, q.grad, atol=1e-6, rtol=1e-4)
    selected = PathDecoderLayer.select_memory([kv], torch.tensor([1]))
    for value, original in zip(selected[0], kv):
        torch.testing.assert_close(value, original[1:2], rtol=0, atol=0)
    with torch.no_grad():
        changed = copied.detach().clone()
        changed[padding] = 100.
        torch.testing.assert_close(other.forward_cached(x, other.project_memory(changed), padding),
                                   other.forward_cached(x, kv, padding), rtol=0, atol=0)


def test_passes_share_coordinate_head_and_encode_write_project_once(monkeypatch):
    torch.manual_seed(42)
    model = build_model(aligned_config()).eval()
    b = aligned_batch(model.cfg)
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
    out = model(b['x'], b['hist'], b['hmask'], confidence_threshold=1.)
    passes = model.cfg.recurrent_refinement_steps+1
    assert calls == dict(encoder=1, writer=1, projection=1, scorer_projection=1,
                         decoder=passes, coordinates=passes, scoring=passes)
    assert out['refinement_points'].shape[1] == passes
    assert out['refinement_hazard_logits'].shape == (len(b['hist']), passes, model.cfg.n_future)
