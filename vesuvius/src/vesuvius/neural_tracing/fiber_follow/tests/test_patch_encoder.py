"""Patch layout, physical coordinates, history gradients and encoder selection."""
import pytest
import torch
from torch import nn

from test_identity import config
from test_trajectory_memory import memory_batch, state_from
from test_training_defaults import REQUIRED
from vesuvius.neural_tracing.fiber_follow.regression.model import (
    ARCHITECTURE, PATCH_ARCHITECTURE, DirectConfig, build_model, sample_features,
)
from vesuvius.neural_tracing.fiber_follow.regression.patch_encoder import patchify, unpatchify
from vesuvius.neural_tracing.fiber_follow.regression.supervision import loss_terms
from vesuvius.neural_tracing.fiber_follow.regression.train import build_parser, checkpoint_config, resolve_encoder


def test_patch_shuffle_preserves_order_padding_and_gradients():
    image = torch.arange(2*3*5*7*9,dtype=torch.float32).reshape(2,3,5,7,9).requires_grad_()
    patches = patchify(image)
    torch.testing.assert_close(patches[0,0,0,0],image[0,:,:4,:4,:4].permute(1,2,3,0).flatten())
    rebuilt = unpatchify(patches,image.shape[-3:])
    torch.testing.assert_close(rebuilt,image,atol=0,rtol=0)
    rebuilt.sum().backward()
    torch.testing.assert_close(image.grad,torch.ones_like(image),atol=0,rtol=0)


def test_patch_centers_align_sampling_and_history_without_changing_conv_model():
    cfg = config(encoder='patch4')
    model = build_model(cfg)
    conv = build_model(config())
    assert DirectConfig(encoder='patch4').token_shape == (30,26,26)
    assert DirectConfig().token_shape == (15,51,51)
    assert conv.cfg.token_stride == (8,2,2)
    xyz = model.encoder.token_xyz.reshape(*cfg.token_shape,3)
    expected = torch.tensor([1.5-(cfg.fine.width-1)/2,1.5-(cfg.fine.width-1)/2,1.5-cfg.fine.behind])*cfg.fine.spacing
    torch.testing.assert_close(xyz[0,0,0],expected)
    point = xyz[2,1,3].reshape(1,1,3)
    sampled,supported = sample_features(xyz.permute(3,0,1,2)[None],point,cfg.fine,cfg.token_stride,cfg.token_offset)
    assert supported.all()
    torch.testing.assert_close(sampled,point)
    references = torch.zeros(1,cfg.n_history+1,3)
    references[:,-1] = point[:,0]
    mask = torch.zeros(1,cfg.n_history+1,dtype=torch.bool)
    mask[:,-1] = True
    rendered = model.encoder.conditioning(references,mask)
    assert rendered[0,2,1,3,2] == 1 and rendered[...,2].sum() == 1
    fine,deep = model.encoder.encode(torch.rand(1,2,cfg.fine.depth,cfg.fine.width,cfg.fine.width),references,mask)
    assert fine.shape == (1,cfg.channels,cfg.fine.depth,cfg.fine.width,cfg.fine.width)
    assert deep.shape == (1,cfg.hidden,*cfg.token_shape)
    assert not any(isinstance(module,nn.Conv3d) for module in model.modules())


def test_patch_history_and_reconstruction_receive_geometry_gradients():
    torch.manual_seed(19)
    model = build_model(config(encoder='patch4'))
    old, current = (memory_batch(model.cfg,step=i) for i in range(2))
    for row in (old,current):
        row['hmask'].zero_()
        row['x']['seed_mask'].zero_()
        row['x']['fine'].requires_grad_()
    first = model(old['x'],old['hist'],old['hmask'])
    result = model(current['x'],current['hist'],current['hmask'],memory=state_from(model,first))
    loss_terms(result,current,model.cfg)['geometry_per_state'].mean().backward()
    for gradient in (old['x']['fine'].grad,current['x']['fine'].grad,
                     model.encoder.patch_projection.weight.grad,model.encoder.reconstruction.weight.grad):
        assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_encoder_selection_and_legacy_checkpoint_defaults():
    assert build_parser().parse_args(REQUIRED).encoder is None
    assert build_parser().parse_args(REQUIRED+['--encoder','patch4']).encoder == 'patch4'
    assert resolve_encoder(None) == 'conv'
    legacy = config().to_dict()
    del legacy['encoder']
    old = dict(architecture=ARCHITECTURE,model_cfg=legacy)
    patch = dict(architecture=PATCH_ARCHITECTURE,model_cfg=config(encoder='patch4').to_dict())
    assert checkpoint_config(old).encoder == resolve_encoder(None,old) == 'conv'
    assert resolve_encoder(None,patch) == resolve_encoder('patch4',patch) == 'patch4'
    with pytest.raises(ValueError,match='Encoder must match'):
        resolve_encoder('conv',patch)
    with pytest.raises(ValueError,match='Encoder must match'):
        resolve_encoder('patch4',old)
    with pytest.raises(ValueError,match='architecture'):
        checkpoint_config(dict(patch,model_cfg=legacy))
