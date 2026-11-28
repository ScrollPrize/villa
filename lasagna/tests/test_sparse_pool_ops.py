"""Torch-ops pooled sparse sampler (CPU and MPS) against dense references.

Two references:
- sparse_pool_cases.dense_reference: float64 trilinear interpolation of the
  zero-extended dense volume, with the kernels' masking (non-finite, outside
  the chunk grid, chunk not loaded -> 0). Tolerance 2e-3 on values in [0, 255]:
  the kernels take the in-cell fraction from p - 32 * ci + 1 in float32, whose
  spacing near 33 is 3.8e-6 voxel, times a neighbour difference of at most 255.
- F.grid_sample (align_corners=True, zeros padding) on the dense float volume,
  CPU, with the same masking. Volume sizes are 32k + 1 so its coordinate
  normalisation is a power-of-two scaling; tolerance 2e-3 as above.
MPS runs the same torch ops and must agree with the CPU bitwise.
"""
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import sparse_pool_ops as ops  # noqa: E402
from sparse_pool_cases import (  # noqa: E402
    dense_reference, make_case, missing_chunks_reference, sample_mask,
)

DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
VALUE_ATOL = 2e-3


def _round_half_away(x: np.ndarray) -> np.ndarray:
    return np.floor(x + 0.5)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("C", [1, 3])
def test_sample_f32_matches_dense_reference(device, C):
    case = make_case(seed=10 + C, C=C)
    pool, table, grid, offset, inv_scale = case.to(device)
    got = ops.sample_f32(pool, table, grid, offset, inv_scale).cpu()
    ref, _ = dense_reference(case)
    assert got.dtype == torch.float32 and got.shape == (C, *case.grid.shape[:3])
    np.testing.assert_allclose(got.reshape(C, -1).numpy(), ref, rtol=0, atol=VALUE_ATOL)
    mask = sample_mask(case)
    assert mask.sum() > 0.4 * mask.size and (~mask).sum() > 0  # both branches exercised
    assert (got.reshape(C, -1).numpy()[:, ~mask] == 0).all()


@pytest.mark.parametrize("device", DEVICES)
def test_sample_u8_is_the_rounded_f32_sample(device):
    case = make_case(seed=3, C=2)
    args = case.to(device)
    u8 = ops.sample_u8(*args).cpu()
    f32 = ops.sample_f32(*args).cpu().numpy()
    assert u8.dtype == torch.uint8
    np.testing.assert_array_equal(u8.numpy(), np.clip(_round_half_away(f32), 0, 255).astype(np.uint8))
    ref, _ = dense_reference(case)
    diff = np.abs(u8.reshape(2, -1).numpy().astype(np.float64) - np.clip(_round_half_away(ref), 0, 255))
    near_half = np.abs(ref - np.floor(ref) - 0.5) < VALUE_ATOL
    assert (diff[~near_half] == 0).all() and diff.max() <= 1


@pytest.mark.parametrize("device", DEVICES)
def test_half_values_round_half_away_from_zero(device):
    # roundf sends 10.5 to 11 (torch.round would give 10, half to even).
    volume = torch.zeros(1, 33, 33, 33, dtype=torch.uint8)
    volume[0, :, :, 4] = 10
    volume[0, :, :, 5] = 11
    pool, table = _pack_all(volume)
    grid = torch.tensor([[[[4.5, 3.0, 3.0], [4.5, 7.25, 9.0]]]])
    args = [t.to(device) for t in (pool, table, grid, torch.zeros(3), torch.ones(3))]
    assert ops.sample_u8(*args).cpu().flatten().tolist() == [11, 11]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("C", [1, 3])
def test_grad_matches_dense_reference(device, C):
    case = make_case(seed=20 + C, C=C, inv_scale=(0.5, 2.0, 1.0 / 3.0))
    pool, table, grid, offset, inv_scale = case.to(device)
    go = torch.randn(C, *case.grid.shape[:3], generator=torch.Generator().manual_seed(5))
    got = ops.sample_grad(pool, table, grid, offset, inv_scale, go.to(device)).cpu()
    _, ref = dense_reference(case, go.reshape(C, -1).double().numpy())
    assert got.shape == case.grid.shape
    scale = float(case.inv_scale.max()) * float(go.abs().max()) * C
    np.testing.assert_allclose(got.reshape(-1, 3).numpy(), ref, rtol=1e-4, atol=VALUE_ATOL * scale)
    assert (got.reshape(-1, 3).numpy()[~sample_mask(case)] == 0).all()


def _dense_grid_sample(volume, grid, offset, inv_scale):
    """F.grid_sample of the float volume at full-res positions (CPU, differentiable)."""
    C, Z, Y, X = volume.shape
    p = (grid - offset) * inv_scale
    size = torch.tensor([X - 1, Y - 1, Z - 1], dtype=torch.float32)
    g = p / size * 2 - 1
    return F.grid_sample(volume.float().unsqueeze(0), g.unsqueeze(0), mode="bilinear",
                         padding_mode="zeros", align_corners=True).squeeze(0)


@pytest.mark.parametrize("device", DEVICES)
def test_values_and_grad_match_f_grid_sample(device):
    case = make_case(seed=7, C=2, inv_scale=(0.5, 0.25, 1.0), H=31, W=37)
    C = 2
    mask = torch.from_numpy(sample_mask(case))
    g_ref = case.grid.clone().requires_grad_(True)
    ref = _dense_grid_sample(case.volume, g_ref, case.offset, case.inv_scale)
    ref = torch.where(mask.view(1, *case.grid.shape[:3]), ref, torch.zeros_like(ref))
    go = torch.randn(ref.shape, generator=torch.Generator().manual_seed(9))
    (ref * go).sum().backward()
    pool, table, grid, offset, inv_scale = case.to(device)
    got = ops.sample_f32(pool, table, grid, offset, inv_scale).cpu()
    torch.testing.assert_close(got, ref.detach(), rtol=0, atol=VALUE_ATOL)
    grad = ops.sample_grad(pool, table, grid, offset, inv_scale, go.to(device)).cpu()
    # Random interior points only: on an exact voxel boundary F.grid_sample's
    # normalisation can round p below the integer and pick the other cell.
    p = ((case.grid - case.offset) * case.inv_scale).reshape(-1, 3)
    off_boundary = ((p - p.round()).abs() > 1e-3).all(dim=1).view(*case.grid.shape[:3], 1)
    scale = float(case.inv_scale.max()) * float(go.abs().max()) * C
    torch.testing.assert_close(torch.where(off_boundary, grad, 0.0), torch.where(off_boundary, g_ref.grad, 0.0),
                               rtol=1e-4, atol=VALUE_ATOL * scale)
    assert int(off_boundary.sum()) > 0.5 * off_boundary.numel()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_mps_torch_ops_equal_cpu_bitwise():
    case = make_case(seed=11, C=2, inv_scale=(0.5, 2.0, 1.0 / 3.0))
    go = torch.randn(2, *case.grid.shape[:3], generator=torch.Generator().manual_seed(1))
    outs = {}
    for device in ("cpu", "mps"):
        args = case.to(device)
        outs[device] = (ops.sample_u8(*args).cpu(), ops.sample_f32(*args).cpu(),
                        ops.sample_grad(*args, go.to(device)).cpu())
    for a, b in zip(outs["cpu"], outs["mps"]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("device", DEVICES)
def test_non_contiguous_grid_matches_contiguous(device):
    case = make_case(seed=4, W=38)
    pool, table, grid, offset, inv_scale = case.to(device)
    strided = grid[:, :, ::2, :]
    assert not strided.is_contiguous()
    for fn in (ops.sample_u8, ops.sample_f32):
        torch.testing.assert_close(fn(pool, table, strided, offset, inv_scale),
                                   fn(pool, table, strided.contiguous(), offset, inv_scale), rtol=0, atol=0)


@pytest.mark.parametrize("device", DEVICES)
def test_empty_grid_and_empty_pool(device):
    case = make_case(seed=5, C=2)
    pool, table, grid, offset, inv_scale = case.to(device)
    empty = grid[:0]
    assert ops.sample_u8(pool, table, empty, offset, inv_scale).shape == (2, 0, *grid.shape[1:3])
    assert ops.sample_f32(pool, table, empty, offset, inv_scale).shape == (2, 0, *grid.shape[1:3])
    go = torch.zeros(2, 0, *grid.shape[1:3], device=device)
    assert ops.sample_grad(pool, table, empty, offset, inv_scale, go).shape == empty.shape
    no_pool = pool[:0]
    no_table = torch.full_like(table, -1)
    assert int(ops.sample_u8(no_pool, no_table, grid, offset, inv_scale).count_nonzero()) == 0
    assert int(ops.sample_f32(no_pool, no_table, grid, offset, inv_scale).count_nonzero()) == 0
    go = torch.ones(2, *grid.shape[:3], device=device)
    assert int(ops.sample_grad(no_pool, no_table, grid, offset, inv_scale, go).count_nonzero()) == 0
    # A table with a zero-size axis (an empty volume axis) samples nothing, even with a non-empty pool.
    flat_table = table[:, :0]
    assert int(ops.sample_u8(pool, flat_table, grid, offset, inv_scale).count_nonzero()) == 0
    assert int(ops.sample_f32(pool, flat_table, grid, offset, inv_scale).count_nonzero()) == 0
    assert int(ops.sample_grad(pool, flat_table, grid, offset, inv_scale, go).count_nonzero()) == 0
    assert ops.missing_chunks(grid, flat_table, offset, inv_scale).shape == (0, 3)


def test_batches_join_seamlessly(monkeypatch):
    case = make_case(seed=6, C=2)
    args = case.to("cpu")
    go = torch.randn(2, *case.grid.shape[:3], generator=torch.Generator().manual_seed(2))
    whole = (ops.sample_u8(*args), ops.sample_f32(*args), ops.sample_grad(*args, go))
    monkeypatch.setattr(ops, "_BATCH", 97)
    batched = (ops.sample_u8(*args), ops.sample_f32(*args), ops.sample_grad(*args, go))
    for a, b in zip(whole, batched):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_argument_checks():
    case = make_case(seed=8)
    pool, table, grid, offset, inv_scale = case.to("cpu")
    with pytest.raises(ValueError, match="grid"):
        ops.sample_f32(pool, table, grid.reshape(-1, 3), offset, inv_scale)
    with pytest.raises(ValueError, match="chunk_table"):
        ops.sample_f32(pool, table.long(), grid, offset, inv_scale)
    with pytest.raises(ValueError, match="pool"):
        ops.sample_f32(pool.float(), table, grid, offset, inv_scale)
    with pytest.raises(ValueError, match="grid"):
        ops.sample_u8(pool, table, grid.double(), offset, inv_scale)


@pytest.mark.parametrize("device", DEVICES)
def test_missing_chunks_matches_bruteforce_cuda_semantics(device):
    rng = np.random.default_rng(12)
    cZ, cY, cX = 5, 6, 7
    table = np.full((cZ, cY, cX), -1, np.int32)
    table[rng.random(table.shape) < 0.3] = 0
    origin = np.array([3.0, -5.0, 11.0], np.float32)
    spacing = np.array([2.0, 4.0, 0.75], np.float32)
    n = 300
    local = rng.random((n, 3)) * np.array([cX, cY, cZ]) * 32 * 1.4 - 0.2 * np.array([cX, cY, cZ]) * 32
    xyz = (local * spacing + origin).astype(np.float32)
    xyz[:8] = [[np.nan, 50, 50], [50, np.inf, 50], [50, 50, -np.inf], [1e30, 50, 50],
               [-1e30, -1e30, -1e30], [origin[0] - 1.0, 50, 50], [origin[0] - 70.0, 50, 50],
               [origin[0] + 64.0, origin[1] + 128.0, origin[2] + 24.0]]
    want = missing_chunks_reference(xyz, table, origin, spacing)
    got = ops.missing_chunks(torch.from_numpy(xyz).reshape(10, 30, 3).to(device),
                             torch.from_numpy(table).to(device), torch.from_numpy(origin).to(device),
                             torch.from_numpy(spacing).to(device))
    assert got.dtype == torch.long and got.device.type == device
    np.testing.assert_array_equal(got.cpu().numpy(), want)
    assert len(want) > 0


@pytest.mark.parametrize("device", DEVICES)
def test_missing_chunks_single_point_marks_its_clipped_neighbourhood(device):
    table = torch.full((4, 4, 4), -1, dtype=torch.int32, device=device)
    table[0, 0, 1] = 3  # loaded: never returned
    xyz = torch.tensor([[1.0, 1.0, 1.0]], device=device)  # chunk (0, 0, 0)
    got = ops.missing_chunks(xyz, table, torch.zeros(3, device=device), torch.ones(3, device=device))
    want = [[z, y, x] for z in (0, 1) for y in (0, 1) for x in (0, 1) if (z, y, x) != (0, 0, 1)]
    assert got.cpu().tolist() == want
    none = ops.missing_chunks(xyz[:0], table, torch.zeros(3, device=device), torch.ones(3, device=device))
    assert none.shape == (0, 3)


def _pack_all(volume):
    from sparse_pool_cases import chunk_grid, pack
    loaded = np.ones(chunk_grid(volume.shape[1:]), bool)
    return pack(volume, loaded)


@pytest.mark.parametrize("device", DEVICES)
def test_missing_chunks_special_points_one_at_a_time(device):
    # Each point alone, so its marks are not hidden by other points: a non-finite
    # coordinate takes chunk 0 on its axis, huge ones clamp, negatives truncate.
    table = np.full((5, 6, 7), -1, np.int32)
    origin = np.array([3.0, -5.0, 11.0], np.float32)
    spacing = np.array([2.0, 4.0, 0.75], np.float32)
    mid = origin + spacing * 32 * np.array([3.5, 2.5, 2.5], np.float32)
    specials = []
    for axis in range(3):
        for value in (np.nan, np.inf, -np.inf, 1e30, -1e30, origin[axis] - 1.0, origin[axis] - 70.0 * spacing[axis]):
            p = mid.copy()
            p[axis] = value
            specials.append(p)
    args = [torch.from_numpy(a).to(device) for a in (table, origin, spacing)]
    for p in specials:
        want = missing_chunks_reference(p[None], table, origin, spacing)
        got = ops.missing_chunks(torch.from_numpy(p[None].astype(np.float32)).to(device), *args)
        np.testing.assert_array_equal(got.cpu().numpy(), want, err_msg=f"point {p.tolist()}")


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape_zyx", [(20, 70, 129), (65, 20, 129), (65, 70, 20), (5, 7, 9), (1, 1, 1), (1, 33, 64)])
def test_one_chunk_along_an_axis(device, shape_zyx):
    # One-cell chunk grids along z, y or x, and volumes smaller than one chunk (down to one voxel):
    # values, gradients and missing chunks against the dense references.
    case = make_case(seed=82, C=2, shape_zyx=shape_zyx, loaded_frac=0.8 if min(shape_zyx) > 9 else 1.0)
    args = case.to(device)
    go = torch.randn(2, *case.grid.shape[:3], generator=torch.Generator().manual_seed(1))
    ref_v, ref_g = dense_reference(case, go.reshape(2, -1).double().numpy())
    got = ops.sample_f32(*args).cpu().reshape(2, -1).double().numpy()
    np.testing.assert_allclose(got, ref_v, atol=VALUE_ATOL, rtol=0)
    g = ops.sample_grad(*args, go.to(device)).cpu().reshape(-1, 3).double().numpy()
    scale = float(go.abs().max()) * 2 * float(case.inv_scale.max())
    np.testing.assert_allclose(g, ref_g, rtol=1e-4, atol=VALUE_ATOL * scale)
    origin, spacing = case.offset.numpy(), (1.0 / case.inv_scale).numpy()
    rows = ops.missing_chunks(case.grid.to(device), case.chunk_table.to(device),
                              torch.from_numpy(origin).to(device), torch.from_numpy(spacing).to(device))
    np.testing.assert_array_equal(rows.cpu().numpy(), missing_chunks_reference(
        case.grid.reshape(-1, 3).numpy(), case.chunk_table.numpy(), origin, spacing))
