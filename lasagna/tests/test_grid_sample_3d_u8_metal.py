"""The Metal uint8 3D grid_sample (MPS) against the CPU reference.

The reference is grid_sample_3d_u8_diff's fallback (F.grid_sample on a float
copy, zeros padding, align_corners=True) run on the CPU, where F.grid_sample
has a 3D backward. Points include positions outside the volume (zero corners)
and the grid is also passed as a non-contiguous view.
"""
import importlib.util
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")

_HERE = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


metal = _load("grid_sample_3d_u8_metal")
diff_module = _load("grid_sample_3d_u8_diff")


def _case(seed=0, C=2, Z=17, Y=19, X=23, D=3, H=11, W=13):
    gen = torch.Generator().manual_seed(seed)
    volume = torch.randint(0, 256, (C, Z, Y, X), generator=gen, dtype=torch.uint8)
    offset = torch.tensor([5.0, -3.0, 7.0])
    inv_scale = torch.tensor([0.5, 0.25, 2.0])
    size = torch.tensor([X, Y, Z], dtype=torch.float32)
    # fullres (x, y, z) positions reaching about one voxel past every face
    local = torch.rand(D, H, W, 3, generator=gen) * (size + 2.0) - 1.0
    grid = local / inv_scale + offset
    return volume, grid, offset, inv_scale


def _reference(volume, grid, offset, inv_scale):
    return diff_module._grid_sample_3d_u8_diff_cpu(volume, grid, offset, inv_scale)


def _mps(*tensors):
    return [t.to("mps") for t in tensors]


def test_uint8_forward_matches_rounded_reference():
    volume, grid, offset, inv_scale = _case()
    expected = _reference(volume, grid, offset, inv_scale).round().clamp(0, 255)
    got = metal.grid_sample_3d_u8(*_mps(volume, grid, offset, inv_scale)).cpu()
    assert got.dtype == torch.uint8 and got.shape == expected.shape
    diff = (got.float() - expected).abs()
    # Rounding can differ only for values within float error of .5
    assert float(diff.max()) <= 1.0
    assert float((diff > 0).float().mean()) < 1e-3


def test_half_values_round_half_away_from_zero():
    # roundf (CUDA) and round (Metal) send 10.5 to 11; rounding to even would give 10.
    volume = torch.zeros(1, 2, 2, 2, dtype=torch.uint8)
    volume[0, :, :, 0] = 10
    volume[0, :, :, 1] = 11
    grid = torch.tensor([[[[0.5, 0.0, 0.0], [0.5, 1.0, 1.0]]]])  # (1, 1, 2, 3), x halfway
    offset = torch.zeros(3)
    inv_scale = torch.ones(3)
    got = metal.grid_sample_3d_u8(*_mps(volume, grid, offset, inv_scale)).cpu()
    assert got.flatten().tolist() == [11, 11]


def test_non_finite_positions_read_zero_with_zero_gradient():
    # As in grid_sample_3d_u8_ops: a NaN or infinite coordinate reads 0 and has a zero gradient.
    volume = torch.zeros(1, 2, 2, 2, dtype=torch.uint8)
    volume[0, :, :, 0] = 10
    volume[0, :, :, 1] = 11
    nan, inf = float("nan"), float("inf")
    grid = torch.tensor([[[[0.5, 0.0, 0.0], [nan, 0.0, 0.0], [0.5, inf, 0.0], [0.5, 0.0, -inf], [nan, nan, nan]]]])
    v, g, o, s = _mps(volume, grid, torch.zeros(3), torch.ones(3))
    assert metal.grid_sample_3d_u8(v, g, o, s).cpu().flatten().tolist() == [11, 0, 0, 0, 0]
    g.requires_grad_(True)
    out = metal.grid_sample_3d_u8_diff(v, g, o, s)
    out.sum().backward()
    assert out.detach().cpu().flatten().tolist() == [10.5, 0.0, 0.0, 0.0, 0.0]
    grad = g.grad.cpu()
    assert grad[0, 0, 0].tolist() == [1.0, 0.0, 0.0]  # d/dx of 10 + x
    assert torch.equal(grad[0, 0, 1:], torch.zeros(4, 3))


def test_diff_forward_and_grid_gradient_match_reference():
    volume, grid, offset, inv_scale = _case(seed=1)
    weights = torch.randn(volume.shape[0], *grid.shape[:3], generator=torch.Generator().manual_seed(2))
    g_ref = grid.clone().requires_grad_(True)
    ref = _reference(volume, g_ref, offset, inv_scale)
    (ref * weights).sum().backward()
    v, g, o, s = _mps(volume, grid, offset, inv_scale)
    g.requires_grad_(True)
    got = metal.grid_sample_3d_u8_diff(v, g, o, s)
    (got * weights.to("mps")).sum().backward()
    torch.testing.assert_close(got.detach().cpu(), ref.detach(), rtol=0, atol=2e-3)
    torch.testing.assert_close(g.grad.cpu(), g_ref.grad, rtol=1e-4, atol=1e-3)


def test_non_contiguous_grid_matches_contiguous():
    volume, grid, offset, inv_scale = _case(seed=3, W=26)
    v, g, o, s = _mps(volume, grid, offset, inv_scale)
    strided = g[:, :, ::2, :]
    assert not strided.is_contiguous()
    torch.testing.assert_close(
        metal.grid_sample_3d_u8_diff(v, strided, o, s),
        metal.grid_sample_3d_u8_diff(v, strided.contiguous(), o, s), rtol=0, atol=0)
    torch.testing.assert_close(
        metal.grid_sample_3d_u8(v, strided, o, s),
        metal.grid_sample_3d_u8(v, strided.contiguous(), o, s), rtol=0, atol=0)


def test_empty_grid():
    volume, grid, offset, inv_scale = _case()
    v, g, o, s = _mps(volume, grid[:0], offset, inv_scale)
    g.requires_grad_(True)
    out = metal.grid_sample_3d_u8_diff(v, g, o, s)
    out.sum().backward()
    assert out.shape == (volume.shape[0], 0, grid.shape[1], grid.shape[2])
    assert metal.grid_sample_3d_u8(v, g.detach(), o, s).shape == out.shape


def test_diff_wrapper_uses_metal_on_mps():
    # grid_sample_3d_u8_diff's public entry point must not take the F.grid_sample
    # fallback on MPS (its 3D backward is not implemented there).
    volume, grid, offset, inv_scale = _case(seed=4)
    v, g, o, s = _mps(volume, grid, offset, inv_scale)
    g.requires_grad_(True)
    out = diff_module.grid_sample_3d_u8_diff(v, g, o, s)
    out.sum().backward()
    assert g.grad is not None and torch.isfinite(g.grad).all()
