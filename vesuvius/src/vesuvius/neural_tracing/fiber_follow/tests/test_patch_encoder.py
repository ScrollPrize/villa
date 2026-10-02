"""Patch layout, physical coordinates, history gradients and encoder selection."""
import pytest
import torch
from torch import nn

from model_fixtures import config
from model_fixtures import slab_batch as memory_batch
from model_fixtures import REQUIRED
from vesuvius.neural_tracing.fiber_follow.regression.model import (
    ARCHITECTURE, PATCH_ARCHITECTURE, DirectConfig, build_model, sample_features,
)
from vesuvius.neural_tracing.fiber_follow.regression.patch_encoder import patchify, unpatchify
from vesuvius.neural_tracing.fiber_follow.regression.supervision import loss_terms
from vesuvius.neural_tracing.fiber_follow.regression.train import build_parser, checkpoint_config, resolve_encoder


def test_patch_shuffle_preserves_order_and_gradients():
    image = torch.arange(2*3*8*12*16,dtype=torch.float32).reshape(2,3,8,12,16).requires_grad_()
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
    assert DirectConfig().token_shape == (15,52,52)
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
    convolutions = [module for module in model.encoder.modules() if isinstance(module,nn.Conv3d)]
    assert convolutions == [model.encoder.patch_projection]
    assert convolutions[0].kernel_size == (6,6,6) and convolutions[0].stride == (4,4,4)


def test_patch_history_and_reconstruction_receive_geometry_gradients():
    torch.manual_seed(19)
    model = build_model(config(encoder='patch4'))
    old, current = (memory_batch(model.cfg,step=i) for i in range(2))
    for row in (old,current):
        row['hmask'].zero_()
        row['x']['seed_mask'].zero_()
        row['x']['fine'].requires_grad_()
    first = model(old['x'],old['hist'],old['hmask'])
    current['x']['history_slabs'].requires_grad_()
    result = model(current['x'],current['hist'],current['hmask'])
    loss_terms(result,current,model.cfg)['geometry_per_state'].mean().backward()
    for gradient in (current['x']['history_slabs'].grad,current['x']['fine'].grad,
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


def test_context_reconstructs_dense_patch_features_only_once():
    model = build_model(config(encoder='patch4'))
    b = memory_batch(model.cfg)
    calls = []
    hook = model.encoder.reconstruction.register_forward_hook(lambda *args: calls.append(1))
    try:
        model.context(b['x'], b['hist'], b['hmask'])
    finally:
        hook.remove()
    assert len(calls) == 1


@pytest.mark.parametrize('shape', [(8,8,8), (12,16,20), (12,20,20)])
def test_overlap_projection_retains_patch_centers_and_border_samples(shape):
    torch.manual_seed(187)
    projection = build_model(config(encoder='patch4')).encoder.patch_projection.double()
    channels, width = projection.in_channels, projection.out_channels
    weight = torch.randn(width,64*channels,dtype=torch.float64)
    bias = torch.randn(width,dtype=torch.float64)
    # The old 4-cube occupies the center of the new 6-cube. A zero halo
    # must reproduce every original patch, including boundary cells.
    with torch.no_grad():
        projection.weight.zero_()
        projection.weight[:,:,1:5,1:5,1:5] = weight.reshape(width,4,4,4,channels).permute(0,4,1,2,3)
        projection.bias.copy_(bias)
    image = torch.randn(2,channels,*shape,dtype=torch.float64,requires_grad=True)
    original = torch.nn.functional.linear(patchify(image),weight,bias)
    overlap = projection(image).permute(0,2,3,4,1)
    torch.testing.assert_close(overlap,original,rtol=1e-12,atol=1e-12)
    a, = torch.autograd.grad(original.square().sum(),image)
    b, = torch.autograd.grad(overlap.square().sum(),image)
    torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-10)


def test_overlap_sees_and_backpropagates_across_all_three_patch_boundaries():
    projection = build_model(config(encoder='patch4')).encoder.patch_projection
    with torch.no_grad():
        projection.weight.zero_()
        projection.weight[0,0].fill_(1.)
        projection.bias.zero_()
    image = torch.zeros(1,2,12,12,12)
    image[0,0,4,4,4] = 1.
    image.requires_grad_()
    tokens = projection(image)
    expected = torch.zeros_like(tokens)
    expected[0,0,:2,:2,:2] = 1.
    torch.testing.assert_close(tokens,expected,rtol=0,atol=0)
    tokens[0,0,0,0,0].backward()
    assert image.grad[0,0,4,4,4] == 1  # Outside the old first 4-cube.
    assert projection.weight.grad[0,0,5,5,5] == 1


@pytest.mark.parametrize('architecture', ['axial_patch4_fiber_slabs_v10', 'axial_patch4_tokens_fiber_slabs_v10'])
def test_old_nonoverlapping_patch_checkpoints_require_fresh_training(architecture):
    with pytest.raises(ValueError,match='Unsupported checkpoint architecture'):
        checkpoint_config(dict(architecture=architecture,model_cfg=config(encoder='patch4').to_dict()))


@pytest.mark.parametrize('shape', [(24, 17), (25, 20)])
def test_incomplete_patch_crops_are_rejected(shape):
    from vesuvius.neural_tracing.fiber_follow.shared.geometry import CropSpec
    with pytest.raises(ValueError, match='multiples of four'):
        DirectConfig(encoder='patch4', fine=CropSpec(depth=shape[0], width=shape[1], behind=8))
    with pytest.raises(ValueError, match='multiples of four'):
        patchify(torch.zeros(1, 1, shape[0], shape[1], shape[1]))


def test_default_crop_and_tokens_are_laterally_centered():
    from vesuvius.neural_tracing.fiber_follow.shared.geometry import crop_local_grid
    cfg = DirectConfig(encoder='patch4', token_only=True)
    image_grid = torch.from_numpy(crop_local_grid(cfg.fine))
    assert image_grid.shape == (120, 104, 104, 3)
    torch.testing.assert_close(image_grid[0, 0, 0, :2], -image_grid[0, -1, -1, :2])
    xyz = build_model(cfg).encoder.token_xyz.reshape(*cfg.token_shape, 3)
    torch.testing.assert_close(xyz[0, 0, 0, :2], -xyz[0, -1, -1, :2])
