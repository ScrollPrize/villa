"""Coarse-only features, checkpoint compatibility and reusable attention memory."""
import copy
from dataclasses import replace

import pytest
import torch

from test_trajectory_memory import cfg, memory_batch
from vesuvius.neural_tracing.fiber_follow.regression.model import (
    DirectConfig, PathDecoderLayer, TOKEN_ARCHITECTURE, PATCH_ARCHITECTURE, build_model,
)
from vesuvius.neural_tracing.fiber_follow.regression.supervision import loss_terms
from vesuvius.neural_tracing.fiber_follow.regression.train import (
    checkpoint_config, prepare_training, training_prediction, training_observation_features,
    resolve_token_only,
)


def token_config(**kwargs):
    return cfg(encoder='patch4', token_only=True, recurrent_refinement_steps=1, **kwargs)


def test_only_coarse_features_enter_paths_scorer_and_memory(monkeypatch):
    model = build_model(token_config())
    assert model.encoder.reconstruction is None
    assert model.output_plane_features is None
    assert model.confidence_scorer.plane_projection is None
    batch = memory_batch(model.cfg, 1)
    reads = []
    original = model.sample_local
    def sample(features, points):
        assert features.shape[1:] == (model.cfg.hidden, *model.cfg.token_shape)
        reads.append(points.shape[1])
        return original(features, points)
    monkeypatch.setattr(model, 'sample_local', sample)
    original_extract = model.recurrent_memory.extract
    def extract(deep, fine, *args):
        assert fine is deep
        return original_extract(deep, fine, *args)
    monkeypatch.setattr(model.recurrent_memory, 'extract', extract)
    out = model(batch['x'], batch['hist'], batch['hmask'])
    assert reads
    assert out['points'].shape == (1, model.cfg.n_future, 3)
    assert model.query[0].in_features == model.cfg.hidden+1
    assert model.recurrent_memory.detail_projection[0].normalized_shape == (model.cfg.hidden+1,)
    assert model.confidence_scorer.query[0].normalized_shape == (4*(model.cfg.hidden+1)+10,)
    assert model.cfg.architecture == TOKEN_ARCHITECTURE
    production = DirectConfig(encoder='patch4', token_only=True)
    assert production.token_shape == (30,26,26)
    assert 30*26*26+(production.n_history+1)+16+48+64*48 == 23545


def test_token_sampling_uses_patch_centers_and_crop_support():
    model = build_model(token_config())
    xyz = model.encoder.token_xyz.reshape(*model.cfg.token_shape,3)
    fields = xyz.permute(3,0,1,2)[None]
    points = xyz[2,1,2].reshape(1,1,3)
    sampled, support = model.sample_local(fields, points)
    assert support.all()
    torch.testing.assert_close(sampled, points)
    sampled, support = model.sample_local(fields, points+10000)
    assert not support.any() and not sampled.any()


@pytest.mark.parametrize('checkpointing', [True, False])
def test_token_training_matches_inference_and_reaches_old_images(checkpointing):
    torch.manual_seed(271)
    eager = build_model(token_config(history_encoder_checkpointing=checkpointing))
    compiled = prepare_training(copy.deepcopy(eager), backend='eager')
    for model in (eager, compiled):
        old, current = (memory_batch(model.cfg, 1, step=i) for i in range(2))
        # Give both implementations exactly the same inputs.
        torch.manual_seed(272)
        old['x']['fine'] = torch.randn_like(old['x']['fine']).requires_grad_()
        current['x']['fine'] = torch.randn_like(current['x']['fine'])
        features = training_observation_features(model, old['x'], old['hist'], old['hmask'])
        state, _ = model.recurrent_memory.observe_tokens(*features, old['x'])
        forward = training_prediction if model is compiled else lambda m,*a,**kw: m(*a,**kw)
        out = forward(model, current['x'], current['hist'], current['hmask'], memory=state)
        terms = loss_terms(out, current, model.cfg)
        loss = terms['geometry_per_state'].mean()+.5*terms['confidence_per_state'].mean()
        loss.backward()
        assert old['x']['fine'].grad.abs().sum() > 0
        assert model.encoder.patch_projection.weight.grad.abs().sum() > 0
        if model is eager:
            expected = out
        else:
            for key in ('points','confidence','hazard_logits'):
                torch.testing.assert_close(out[key], expected[key], rtol=1e-5, atol=1e-6)
    for (name,p), (_,q) in zip(eager.named_parameters(), compiled.named_parameters()):
        if p.grad is None:
            assert q.grad is None, name
        else:
            torch.testing.assert_close(p.grad, q.grad, rtol=3e-4, atol=2e-6, msg=name)


def test_history_checkpoint_toggle_preserves_values_and_gradients():
    torch.manual_seed(274)
    checked = build_model(token_config())
    plain = build_model(replace(checked.cfg, history_encoder_checkpointing=False))
    plain.load_state_dict(checked.state_dict())
    batch = memory_batch(checked.cfg, 1)
    results = []
    for model in (checked, plain):
        features = model.replay_observation_features(batch['x'], batch['hist'], batch['hmask'])
        features[0].square().sum().backward()
        results.append(features[0])
    torch.testing.assert_close(*results, rtol=0, atol=0)
    for (name,p), (_,q) in zip(checked.named_parameters(),plain.named_parameters()):
        if p.grad is not None:
            torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0,msg=name)


def test_token_architecture_is_distinct_and_legacy_patch_defaults_survive():
    legacy = cfg(encoder='patch4').to_dict()
    del legacy['token_only'], legacy['history_encoder_checkpointing']
    loaded = checkpoint_config(dict(architecture=PATCH_ARCHITECTURE, model_cfg=legacy))
    assert not loaded.token_only and loaded.history_encoder_checkpointing
    token = token_config().to_dict()
    checkpoint = dict(architecture=TOKEN_ARCHITECTURE, model_cfg=token)
    assert checkpoint_config(checkpoint).token_only
    assert resolve_token_only(None, checkpoint)
    assert resolve_token_only(True, checkpoint)
    assert not resolve_token_only(None)
    with pytest.raises(ValueError, match='must match'):
        resolve_token_only(False, checkpoint)
    with pytest.raises(ValueError, match='architecture'):
        checkpoint_config(dict(architecture=PATCH_ARCHITECTURE, model_cfg=token))
    with pytest.raises(ValueError, match='patch4'):
        DirectConfig(token_only=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_compacted_attention_matches_masked_values_gradients_and_reuses_storage():
    torch.manual_seed(273)
    layer = PathDecoderLayer(32,4,64,dropout=0.,batch_first=True,norm_first=True).cuda()
    reference = copy.deepcopy(layer)
    memory = torch.randn(2,43,32,device='cuda',requires_grad=True)
    other = memory.detach().clone().requires_grad_()
    query = torch.randn(2,4,32,device='cuda')
    padding = torch.zeros(2,43,dtype=torch.bool,device='cuda')
    padding[0,19:] = True
    padding[1,7:13] = True
    with torch.autocast('cuda',dtype=torch.bfloat16):
        dense = reference.project_memory(other)
        compact = layer.compact_memory(layer.project_memory(memory),padding)
        assert [row[0].shape[2] for row in compact] == [19,37]
        pointers = [[v.data_ptr() for v in row] for row in compact]
        a = b = query
        for _ in range(3):
            a = reference.forward_cached(a,dense,padding)
            b = layer.forward_cached(b,compact,padding)
        assert pointers == [[v.data_ptr() for v in row] for row in compact]
        selected = PathDecoderLayer.select_memory([compact],torch.tensor([1],device='cuda'))
        assert selected[0][0][0] is compact[1][0]
    torch.testing.assert_close(a,b,rtol=.03,atol=.02)
    a.square().mean().backward(); b.square().mean().backward()
    torch.testing.assert_close(memory.grad,other.grad,rtol=.06,atol=3e-4)
    for p,q in zip(layer.parameters(),reference.parameters()):
        torch.testing.assert_close(p.grad,q.grad,rtol=.06,atol=6e-4)
