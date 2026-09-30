"""Coarse-only features, checkpoint compatibility and reusable attention memory."""
import copy
from dataclasses import replace

import pytest
import torch

from slab_fixtures import cfg, slab_batch as memory_batch
from vesuvius.neural_tracing.fiber_follow.regression.model import (
    DirectConfig, PathDecoderLayer, TOKEN_ARCHITECTURE, PATCH_ARCHITECTURE, build_model,
)
from vesuvius.neural_tracing.fiber_follow.regression.supervision import loss_terms
from vesuvius.neural_tracing.fiber_follow.regression.train import (
    checkpoint_config, prepare_training, training_prediction,
    resolve_token_only,
)


def token_config(**kwargs):
    return cfg(encoder='patch4', token_only=True, recurrent_refinement_steps=1, **kwargs)




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
