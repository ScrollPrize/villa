"""The sheet-strip model's Metal kernels on Apple GPUs against float64 references on the CPU"""
import copy

import pytest
import torch
import torch.nn.functional as F

from vesuvius.neural_tracing.tracers.fibers.sheet_strip.model import SheetStripConfig, SheetStripFollower
from vesuvius.neural_tracing.tracers.fibers.tests.test_sheet_strip import SMALL, STRIP

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason='needs PyTorch with MPS')


def relative(got, want):
    want = want.cpu().double()
    return ((got.cpu().double()-want).abs().max()/want.pow(2).mean().sqrt()).item()  # MPS has no float64


def test_metal_crop_cnn_matches_float64():
    from vesuvius.neural_tracing.tracers.models.blocks import CropCNN
    from vesuvius.neural_tracing.tracers.models.mps import optimize_for_mps
    from vesuvius.neural_tracing.tracers.models.mps_cnn import MetalCropCNN

    torch.manual_seed(0)
    model = torch.nn.Module()
    model.cnn = CropCNN((12, 24, 64), (2, 2, 1), (1, 2, 2), activation='leaky_relu', in_channels=2, context=64)
    with torch.no_grad():
        # every parameter random, the zero-initialised context fuse and the norms' affines included
        for module in model.modules():
            if isinstance(module, torch.nn.Conv3d):
                module.weight.normal_(0., module.weight[0].numel()**-.5)
                if module.bias is not None:
                    module.bias.normal_(0., .1)
            elif isinstance(module, torch.nn.InstanceNorm3d):
                module.weight.normal_(1., .2)
                module.bias.normal_(0., .2)
    ref = copy.deepcopy(model.cnn).double()
    image = torch.randn(2, 2, 32, 32, 32)
    model.to('mps')
    optimize_for_mps(model)
    assert type(model.cnn) is MetalCropCNN
    with torch.no_grad():
        got = model.cnn.features(image.to('mps'))
        want = ref.features(image.double())
    assert got.shape == want.shape
    assert torch.isfinite(got).all()
    assert relative(got, want) < 1e-4


def test_metal_patch_tokens_match_grid_sample():
    from vesuvius.neural_tracing.tracers.models.mps import optimize_for_mps
    from vesuvius.neural_tracing.tracers.models.mps_tokens import MetalPatchTokens
    from vesuvius.neural_tracing.tracers.models.patch_tokens import PatchTokens

    cfg = SheetStripConfig(**dict(SMALL, multiscale_fine_width=None), **STRIP)
    tokens = SheetStripFollower(cfg).cell_token
    ref = copy.deepcopy(tokens)  # PatchTokens samples in float32 whatever its weights, so the reference is float32
    lattice = torch.randn(3, 16, 8, 8, 8)
    holder = torch.nn.Module()
    holder.tokens = tokens.to('mps')
    optimize_for_mps(holder)
    assert type(holder.tokens) is MetalPatchTokens
    with torch.no_grad():
        # channels-last storage, as MetalCropCNN hands it over
        mps_lattice = lattice.permute(0, 2, 3, 4, 1).contiguous().to('mps').permute(0, 4, 1, 2, 3)
        got = holder.tokens(mps_lattice)
        want = PatchTokens.forward(ref, lattice)
    assert relative(got, want) < 1e-5


def test_metal_zscore_matches_float64():
    from vesuvius.neural_tracing.tracers.shared.crop_sampling.mps_zscore import suits, zscore_crops_metal
    from vesuvius.neural_tracing.tracers.shared.crop_sampling.normalization import zscore_crops_torch

    g = torch.Generator().manual_seed(0)
    crops = torch.randint(0, 256, (4, 1, 24, 20, 16), generator=g).float()/255
    crops[1] = .4  # constant: all zeros
    x = crops.to('mps')
    assert suits(x)
    got = zscore_crops_metal(x)
    want = zscore_crops_torch(crops.double())
    assert (got[1] == 0).all()
    assert (got.cpu().double()-want).abs().max() < 1e-5


def test_masked_attention_matches_float64():
    from vesuvius.neural_tracing.tracers.models.mps_attention import fused_attention_suits, masked_attention

    g = torch.Generator().manual_seed(0)
    b, h, lq, lk = 2, 4, 70, 101  # not multiples of the kernel's tiles
    qkv = torch.randn(b, lk, 3, h, 32, generator=g)
    q, k, v = (qkv[:, :lq, 0].transpose(1, 2), qkv[:, :, 1].transpose(1, 2), qkv[:, :, 2].transpose(1, 2))
    mask = torch.rand(b, 1, lq, lk, generator=g) > .1
    mask[0, 0, 3] = False  # a query that reads no key outputs zeros
    mask[1, 0, :, 32:64] = False  # a whole key block masked
    mps = [t.to('mps') for t in (q, k, v)]
    assert fused_attention_suits(*mps, mask.to('mps'))
    got = masked_attention(*mps, mask.to('mps'))
    want = F.scaled_dot_product_attention(q.double(), k.double(), v.double(), attn_mask=mask)
    want[0, :, 3] = 0.
    assert torch.isfinite(got).all()
    assert (got[0, :, 3] == 0).all()
    assert relative(got, want) < 1e-5  # fp32 paths measure 2-4e-6 here, CPU SDPA included
