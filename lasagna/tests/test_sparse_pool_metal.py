"""The Metal pooled sparse sampler (sparse_pool_metal) against the torch-ops reference on MPS.

Tolerances: the kernels may differ from sparse_pool_ops only where a Metal
compiler contracts a multiply and an add into one FMA (one rounding instead of two), as nvcc does
for the CUDA kernels.
- Zero pattern (non-finite, outside the chunk grid, chunk not loaded): identical.
- sample_f32: abs diff <= 1e-4 (eight products and sums of values <= 255, ulp 1.5e-5).
- sample_u8: identical where the reference float value is more than 1e-4 from a .5 boundary,
  +-1 within it.
- sample_grad: rtol 1e-5, atol 1e-4 * C * max|grad_output| * max(inv_scale).
- missing_chunks: identical rows in identical (row-major) order.
The kernels switch contraction off (sparse_pool_metal.CONTRACTION == "off"), so on such a build
they also equal the torch ops bitwise; test_samplers_equal_the_torch_ops_bitwise checks that. Its
cases have no subnormal coordinates, which Metal and MPS torch ops flush to zero and the CPU
does not.
"""
import os
import sys
import warnings

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import sparse_pool_ops as ops  # noqa: E402
from sparse_pool_cases import (  # noqa: E402
    chunk_grid, dense_reference, make_case, missing_chunks_reference, pack, sample_mask,
)

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")

if torch.backends.mps.is_available():
    import sparse_pool_metal as metal  # noqa: E402
else:  # pragma: no cover
    metal = None

MPS = "mps"
VALUE_ATOL = 1e-4
INV_SCALES = [(0.5, 0.25, 1.0), (0.5, 2.0, 1.0 / 3.0)]
FUNCTIONS = ["sample_u8", "sample_f32", "sample_grad"]


@pytest.fixture(autouse=True)
def _compiled_and_auto(monkeypatch):
    metal.compile()
    monkeypatch.delenv("LASAGNA_METAL", raising=False)
    monkeypatch.setattr(metal, "_WARNED", set())


def _go(case, seed=5):
    C = case.pool.shape[1]
    return torch.randn(C, *case.grid.shape[:3], generator=torch.Generator().manual_seed(seed))


def _run(module, name, args, go=None):
    fn = getattr(module, name)
    out = fn(*args, go) if name == "sample_grad" else fn(*args)
    return out.cpu()


def _assert_close(name, got, args, go=None, mask=None):
    """got (from Metal) against sparse_pool_ops on the same MPS tensors, with the tolerances above."""
    ref = _run(ops, name, args, go)
    assert got.dtype == ref.dtype and got.shape == ref.shape
    if mask is not None:  # the zero pattern, from an independent float64 formulation
        zero = ~torch.from_numpy(mask)
        if name == "sample_grad":
            assert (got.reshape(-1, 3)[zero] == 0).all()
        else:
            assert (got.reshape(got.shape[0], -1)[:, zero] == 0).all()
    if name == "sample_f32":
        torch.testing.assert_close(got, ref, rtol=0, atol=VALUE_ATOL)
    elif name == "sample_u8":
        ref_f = _run(ops, "sample_f32", args)
        diff = (got.int() - ref.int()).abs()
        near_half = (ref_f - ref_f.floor() - 0.5).abs() <= VALUE_ATOL
        assert int(diff.max()) <= 1
        assert not bool((diff[~near_half] != 0).any())
    else:
        C = args[0].shape[1]
        scale = C * float(go.abs().max()) * float(args[4].abs().max())
        torch.testing.assert_close(got, ref, rtol=1e-5, atol=VALUE_ATOL * scale)
        assert torch.equal(got == 0, ref == 0)


# ---------------------------------------------------------------- the samplers against the torch ops

@pytest.mark.parametrize("name", FUNCTIONS)
@pytest.mark.parametrize("inv_scale", INV_SCALES)
@pytest.mark.parametrize("C", [1, 3])
def test_samplers_match_the_torch_ops(name, inv_scale, C):
    case = make_case(seed=40 + C, C=C, inv_scale=inv_scale)
    args = case.to(MPS)
    go = _go(case).to(MPS) if name == "sample_grad" else None
    mask = sample_mask(case)
    assert mask.sum() > 0.4 * mask.size and (~mask).sum() > 0  # both branches exercised
    _assert_close(name, _run(metal, name, args, go), args, go, mask)


@pytest.mark.parametrize("inv_scale", INV_SCALES)
@pytest.mark.parametrize("C", [1, 3])
def test_samplers_equal_the_torch_ops_bitwise(inv_scale, C):
    if metal.CONTRACTION != "off":
        pytest.skip(f"contraction is {metal.CONTRACTION!r}: only the tolerance tests apply")
    case = make_case(seed=50 + C, C=C, inv_scale=inv_scale)
    args = case.to(MPS)
    go = _go(case).to(MPS)
    for name in FUNCTIONS:
        torch.testing.assert_close(_run(metal, name, args, go), _run(ops, name, args, go), rtol=0, atol=0)


@pytest.mark.parametrize("C", [1, 3])
def test_values_and_gradient_match_the_float64_dense_reference(C):
    # Independent of sparse_pool_ops: trilinear interpolation of the dense volume in float64.
    # 2e-3 = the float32 in-cell fraction's spacing near 33 (3.8e-6) times a difference <= 255.
    case = make_case(seed=60 + C, C=C, inv_scale=(0.5, 2.0, 1.0 / 3.0))
    args = case.to(MPS)
    go = _go(case)
    values, grad = dense_reference(case, go.reshape(C, -1).double().numpy())
    got = metal.sample_f32(*args).cpu().reshape(C, -1).numpy()
    np.testing.assert_allclose(got, values, rtol=0, atol=2e-3)
    got_grad = metal.sample_grad(*args, go.to(MPS)).cpu().reshape(-1, 3).numpy()
    scale = C * float(go.abs().max()) * float(case.inv_scale.max())
    np.testing.assert_allclose(got_grad, grad, rtol=1e-4, atol=2e-3 * scale)


def _exact_case(C=2, seed=0):
    """Volume (C, 65, 70, 129), chunk grid (3, 3, 5), with chunks 1, 4, 7, ... (row-major) unloaded;
    offset 0 and inv_scale 1, so the grid holds the local voxel coordinates exactly."""
    gen = torch.Generator().manual_seed(seed)
    volume = torch.randint(0, 256, (C, 65, 70, 129), generator=gen, dtype=torch.uint8)
    loaded = np.ones(chunk_grid(volume.shape[1:]), bool)
    loaded.reshape(-1)[1::3] = False
    pool, table = pack(volume, loaded, seed=seed)
    return volume, loaded, pool, table


def _boundary_positions():
    """Local (x, y, z) positions on the kernels' float edges along x, y and z."""
    below_32 = float(np.float32(32.0) - np.float32(2.0 ** -19))  # l = p + 1 rounds up to exactly 33.0
    # 160 = 32 * cX: the first x outside the chunk grid (y and z stay inside it).
    values = [0.0, 0.5, 1.0, 31.0, 31.5, below_32, 31.999998, 32.0, 32.5, 33.0, 63.0, 63.999996,
              64.0, 64.5, 95.5, 96.0, 127.5, 128.0, 128.5, 129.0, 159.5, 160.0, 160.5, 191.9,
              -1e-3, -0.5, -1.0, -31.9, -32.0, -32.5]
    rows = []
    for v in values:
        for axis in range(3):
            p = [10.25, 20.75, 30.5]
            p[axis] = v
            rows.append(p)
        rows.append([v, v, v])
    rows += [[np.nan, 5.0, 5.0], [5.0, np.inf, 5.0], [5.0, 5.0, -np.inf], [1e30, 5.0, 5.0], [-1e30, 5.0, 5.0],
             [3.4e38, 3.4e38, 3.4e38], [128.999, 69.999, 64.999], [159.5, 95.5, 95.5], [160.0, 96.0, 96.0]]
    return np.asarray(rows, np.float32)


@pytest.mark.parametrize("C", [1, 3])
def test_exact_positions_on_voxel_and_chunk_edges(C):
    volume, loaded, pool, table = _exact_case(C)
    pts = _boundary_positions()
    # The in-chunk coordinate l = p - 32 * floor(p / 32) + 1 reaches exactly 33.0 in float32 for
    # some of these: both corners clamp to 33 and the upper one gets weight 0.
    finite = np.isfinite(pts).all(1) & (np.abs(pts) < 1e6).all(1)
    with np.errstate(invalid="ignore"):
        ci = np.floor(pts / np.float32(32.0)).astype(np.float32)
        lcoord = (pts - ci * np.float32(32.0)) + np.float32(1.0)
    cz, cy, cx = (np.where(finite, ci[:, k], -1).astype(np.int64) for k in (2, 1, 0))
    in_grid = finite & (cx >= 0) & (cx < 5) & (cy >= 0) & (cy < 3) & (cz >= 0) & (cz < 3)
    sampled = in_grid.copy()
    sampled[in_grid] = loaded[cz[in_grid], cy[in_grid], cx[in_grid]]
    for axis in range(3):  # l == 33.0 along every axis, in loaded chunks
        assert (sampled & (lcoord[:, axis] == np.float32(33.0))).any(), axis
    # just past the chunk grid along x, inside it along y and z, with the next table row loaded
    assert ((ci[:, 0] == 5) & (cy >= 0) & (cy < 2) & (cz >= 0) & (cz < 3)).any()
    grid = torch.from_numpy(pts).reshape(1, 1, -1, 3)
    args = [t.to(MPS) for t in (pool, table, grid, torch.zeros(3), torch.ones(3))]
    go = torch.randn(C, 1, 1, pts.shape[0], generator=torch.Generator().manual_seed(3))
    for name in FUNCTIONS:
        got = _run(metal, name, args, go.to(MPS))
        _assert_close(name, got, args, go.to(MPS))
        if metal.CONTRACTION == "off":
            assert torch.equal(got, _run(ops, name, args, go.to(MPS))), name


@pytest.mark.parametrize("shape_zyx", [(20, 70, 129), (65, 20, 129), (65, 70, 20), (5, 7, 9)])
def test_one_chunk_along_an_axis(shape_zyx):
    # Chunk grids with a single cell along z, y or x, and a volume smaller than one chunk (1, 1, 1).
    case = make_case(seed=82, C=1, shape_zyx=shape_zyx, loaded_frac=0.8 if min(shape_zyx) > 9 else 1.0)
    assert 1 in case.chunk_table.shape
    args = case.to(MPS)
    go = _go(case).to(MPS)
    mask = sample_mask(case)
    assert mask.any() and (~mask).any()
    for name in FUNCTIONS:
        got = _run(metal, name, args, go)
        _assert_close(name, got, args, go, mask)
        if metal.CONTRACTION == "off":
            assert torch.equal(got, _run(ops, name, args, go)), name
    origin, spacing = case.offset.numpy(), (1.0 / case.inv_scale).numpy()
    xyz = case.grid.reshape(-1, 3).numpy()
    rows = metal.missing_chunks(args[2], args[1], *(torch.from_numpy(a).to(MPS) for a in (origin, spacing)))
    np.testing.assert_array_equal(rows.numpy(), missing_chunks_reference(xyz, case.chunk_table.numpy(), origin, spacing))


def test_half_values_round_half_away_from_zero():
    # roundf sends 10.5 to 11; a round-half-to-even kernel would give 10. Weights 0.5 and 0.5 on
    # integers make the value exactly 10.5 whatever the compiler contracts.
    volume = torch.zeros(1, 33, 33, 33, dtype=torch.uint8)
    volume[0, :, :, 4] = 10
    volume[0, :, :, 5] = 11
    volume[0, :, :, 7] = 254
    volume[0, :, :, 8] = 255
    pool, table = pack(volume, np.ones(chunk_grid(volume.shape[1:]), bool))
    grid = torch.tensor([[[[4.5, 3.0, 3.0], [4.5, 7.25, 9.0], [7.5, 1.0, 2.0], [4.25, 1.0, 1.0]]]])
    args = [t.to(MPS) for t in (pool, table, grid, torch.zeros(3), torch.ones(3))]
    assert metal.sample_u8(*args).cpu().flatten().tolist() == [11, 11, 255, 10]
    assert metal.sample_f32(*args).cpu().flatten().tolist() == [10.5, 10.5, 254.5, 10.25]


# ---------------------------------------------------------------------------- layouts and shapes

def _grid_views(grid):
    """Non-contiguous views holding the same points as grid (D, H, W, 3), with their contiguous copies."""
    D, H, W, _ = grid.shape
    views = {"every_other_w": grid[:, :, ::2], "cropped": grid[:, 1:, 1:], "permuted": grid.permute(1, 0, 2, 3),
             "expanded": grid[:1].expand(3, -1, -1, -1)}
    wide = torch.zeros(D, H, W, 4, device=grid.device)
    wide[..., 1:] = grid
    views["offset_stride4"] = wide[..., 1:]
    spread = torch.zeros(D, H, W, 6, device=grid.device)
    spread[..., ::2] = grid
    views["coord_stride2"] = spread[..., ::2]
    return views


@pytest.mark.parametrize("name", FUNCTIONS)
def test_non_contiguous_grids_match_their_contiguous_copies(name):
    case = make_case(seed=70, C=2, W=38)
    pool, table, grid, offset, inv_scale = case.to(MPS)
    for label, view in _grid_views(grid).items():
        assert not view.is_contiguous(), label
        C = pool.shape[1]
        go = torch.randn(C, *view.shape[:3], generator=torch.Generator().manual_seed(1)).to(MPS)
        args = (pool, table, view, offset, inv_scale)
        got = _run(metal, name, args, go)
        want = _run(metal, name, (pool, table, view.contiguous(), offset, inv_scale), go)
        assert torch.equal(got, want), label
        _assert_close(name, got, args, go)


def test_non_contiguous_table_and_vectors():
    case = make_case(seed=71, C=1)
    pool, table, grid, offset, inv_scale = case.to(MPS)
    wide = torch.full((*table.shape[:2], 2 * table.shape[2]), -7, dtype=torch.int32, device=MPS)
    wide[:, :, ::2] = table
    table_view = wide[:, :, ::2]
    pairs = torch.stack([offset, torch.zeros(3, device=MPS)], dim=1)  # offset as a stride-2 view
    off_view = pairs[:, 0]
    assert not table_view.is_contiguous() and not off_view.is_contiguous()
    go = _go(case).to(MPS)
    for name in FUNCTIONS:
        got = _run(metal, name, (pool, table_view, grid, off_view, inv_scale), go)
        assert torch.equal(got, _run(metal, name, (pool, table, grid, offset, inv_scale), go)), name
    xyz = grid.reshape(-1, 3)
    origin, spacing = torch.tensor([1.0, 2.0, 3.0], device=MPS), torch.tensor([2.0, 4.0, 0.75], device=MPS)
    assert torch.equal(metal.missing_chunks(xyz, table_view, origin, spacing),
                       metal.missing_chunks(xyz, table, origin, spacing))


def test_empty_grid_pool_and_table():
    case = make_case(seed=72, C=2)
    pool, table, grid, offset, inv_scale = case.to(MPS)
    for empty in (grid[:0], grid[:, :0], grid[:, :, :0]):
        assert metal.sample_u8(pool, table, empty, offset, inv_scale).shape == (2, *empty.shape[:3])
        assert metal.sample_f32(pool, table, empty, offset, inv_scale).shape == (2, *empty.shape[:3])
        go = torch.zeros(2, *empty.shape[:3], device=MPS)
        assert metal.sample_grad(pool, table, empty, offset, inv_scale, go).shape == empty.shape
    go = torch.ones(2, *grid.shape[:3], device=MPS)
    no_pool, no_table = pool[:0], torch.full_like(table, -1)
    empty_table = table[:, :0]
    for p, t in ((no_pool, no_table), (pool, empty_table)):
        u8 = metal.sample_u8(p, t, grid, offset, inv_scale)
        f32 = metal.sample_f32(p, t, grid, offset, inv_scale)
        g = metal.sample_grad(p, t, grid, offset, inv_scale, go)
        assert u8.dtype == torch.uint8 and u8.shape == (2, *grid.shape[:3]) and int(u8.count_nonzero()) == 0
        assert f32.dtype == torch.float32 and f32.shape == u8.shape and int(f32.count_nonzero()) == 0
        assert g.shape == grid.shape and int(g.count_nonzero()) == 0
        assert torch.equal(f32, ops.sample_f32(p, t, grid, offset, inv_scale))
    no_channels = pool[:, :0]
    assert metal.sample_f32(no_channels, table, grid, offset, inv_scale).shape == (0, *grid.shape[:3])
    g = metal.sample_grad(no_channels, table, grid, offset, inv_scale, go[:0])
    assert g.shape == grid.shape and int(g.count_nonzero()) == 0


def test_many_samples_in_one_dispatch():
    # 2,069,385 samples: many thread groups and a partial last one.
    case = make_case(seed=73, C=1, D=5, H=411, W=1007)
    args = case.to(MPS)
    go = _go(case).to(MPS)
    for name in FUNCTIONS:
        _assert_close(name, _run(metal, name, args, go), args, go)


# ---------------------------------------------------------------- lifetime: pool and table per call

def test_pool_and_table_are_read_afresh_on_every_call():
    # sync() may reallocate the pool (1.5x, copying the loaded slots) and fills table entries in
    # place; the backward samples the pool and table as they are at backward time.
    volume, loaded, pool, table = _exact_case(C=1, seed=4)
    # Chunks (0, 1, 0), (0, 0, 2) and (0, 1, 3): all loaded (_exact_case unloads 1, 4, 7, ...).
    grid = torch.from_numpy(np.array([[[[10.5, 40.25, 30.75], [70.5, 5.5, 6.5], [100.25, 33.5, 2.5]]]], np.float32))
    zeros, ones = torch.zeros(3, device=MPS), torch.ones(3, device=MPS)
    p, t, g = pool.to(MPS), table.to(MPS), grid.to(MPS)
    first = metal.sample_f32(p, t, g, zeros, ones).cpu()
    # Unload one sampled chunk, then load it again into a new slot of a grown pool.
    cz, cy, cx = 0, 0, 2  # chunk of (70.5, 5.5, 6.5)
    old_slot = int(t[cz, cy, cx])
    assert old_slot >= 0
    t[cz, cy, cx] = -1
    hidden = metal.sample_f32(p, t, g, zeros, ones).cpu()
    assert hidden[0, 0, 0, 1] == 0 and first[0, 0, 0, 1] != 0
    assert hidden[0, 0, 0, 0] == first[0, 0, 0, 0] != 0
    go = torch.ones(1, 1, 1, 3, device=MPS)
    assert int(metal.sample_grad(p, t, g, zeros, ones, go)[0, 0, 1].count_nonzero()) == 0
    grown = torch.empty((p.shape[0] * 3 // 2 + 1, *p.shape[1:]), dtype=torch.uint8, device=MPS)
    grown[:p.shape[0]] = p
    grown[-1] = p[old_slot]
    t[cz, cy, cx] = grown.shape[0] - 1
    assert torch.equal(metal.sample_f32(grown, t, g, zeros, ones).cpu(), first)
    grad = metal.sample_grad(grown, t, g, zeros, ones, go).cpu()
    assert int(grad[0, 0, 1].count_nonzero()) > 0
    assert torch.equal(grad, ops.sample_grad(grown, t, g, zeros, ones, go).cpu())


def test_kernels_never_write_their_inputs():
    case = make_case(seed=74, C=2)
    args = case.to(MPS)
    go = _go(case).to(MPS)
    origin, spacing = torch.tensor([1.0, 2.0, 3.0], device=MPS), torch.tensor([0.5, 0.75, 3.0], device=MPS)
    inputs = (*args, go, origin, spacing)
    before = [t.clone() for t in inputs]
    for name in FUNCTIONS:
        _run(metal, name, args, go)
    metal.missing_chunks(args[2], args[1], origin, spacing)
    torch.mps.synchronize()
    for a, b in zip(inputs, before):  # bit patterns: the grid holds NaN edge points
        bits = torch.int32 if a.dtype == torch.float32 else a.dtype
        assert torch.equal(a.view(bits), b.view(bits))


# ------------------------------------------------------------------ fallbacks, warnings, LASAGNA_METAL

def _spy(monkeypatch, name):
    calls = []
    real = getattr(ops, name)

    def spy(*a, **k):
        calls.append(name)
        return real(*a, **k)

    monkeypatch.setattr(ops, name, spy)
    return calls


def _reason_setup(reason, monkeypatch):
    """Arguments for a call the kernels do not take, for each kind of reason."""
    case = make_case(seed=75, C=1)
    device = "cpu" if reason == "device" else MPS
    if reason == "samples":
        monkeypatch.setattr(metal, "_MAX_SAMPLES", 100)
    elif reason == "table_dim":
        monkeypatch.setattr(metal, "_MAX_TABLE_DIM", 4)
    elif reason == "table_cells":
        monkeypatch.setattr(metal, "_MAX_TABLE_CELLS", 10)
    return case, case.to(device), device


REASONS = ["device", "samples", "table_dim", "table_cells"]
MATCH = {"device": "not on MPS", "samples": "one dispatch", "table_dim": "chunk table",
         "table_cells": "chunk table"}


@pytest.mark.parametrize("reason", REASONS)
@pytest.mark.parametrize("name", FUNCTIONS + ["missing_chunks"])
def test_calls_the_kernels_do_not_take_use_the_torch_ops_with_one_warning(name, reason, monkeypatch):
    case, args, device = _reason_setup(reason, monkeypatch)
    go = _go(case).to(device)
    origin, spacing = torch.tensor([1.0, 2.0, 3.0], device=device), torch.tensor([2.0, 4.0, 0.75], device=device)
    want = (ops.missing_chunks(args[2], args[1], origin, spacing) if name == "missing_chunks"
            else _run(ops, name, args, go)).cpu()
    calls = _spy(monkeypatch, name)

    def call():
        if name == "missing_chunks":
            rows = metal.missing_chunks(args[2], args[1], origin, spacing)
            assert rows.device.type == "cpu"  # on every path: the cache needs the rows on the host
            return rows
        return _run(metal, name, args, go)

    with pytest.warns(RuntimeWarning, match=MATCH[reason]):
        got = call()
    assert torch.equal(got, want) and calls == [name]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert torch.equal(call(), want)
    # One warning per process and reason: the second call is quiet.
    assert not [w for w in caught if "sparse_pool_metal" in str(w.message)]
    assert calls == [name, name]


@pytest.mark.parametrize("reason", REASONS)
@pytest.mark.parametrize("name", FUNCTIONS + ["missing_chunks"])
def test_lasagna_metal_1_raises_instead_of_falling_back(name, reason, monkeypatch):
    case, args, device = _reason_setup(reason, monkeypatch)
    monkeypatch.setenv("LASAGNA_METAL", "1")
    go = _go(case).to(device)
    origin, spacing = torch.tensor([1.0, 2.0, 3.0], device=device), torch.tensor([2.0, 4.0, 0.75], device=device)
    calls = _spy(monkeypatch, name)
    with pytest.raises(RuntimeError, match="LASAGNA_METAL=1") as info:
        if name == "missing_chunks":
            metal.missing_chunks(args[2], args[1], origin, spacing)
        else:
            _run(metal, name, args, go)
    assert MATCH[reason] in str(info.value) and calls == []


def test_supported_calls_never_touch_the_torch_ops(monkeypatch):
    monkeypatch.setenv("LASAGNA_METAL", "1")
    case = make_case(seed=76, C=2)
    args = case.to(MPS)
    go = _go(case).to(MPS)
    calls = [_spy(monkeypatch, name) for name in FUNCTIONS + ["missing_chunks"]]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for name in FUNCTIONS:
            _run(metal, name, args, go)
        metal.missing_chunks(args[2], args[1], torch.zeros(3, device=MPS), torch.ones(3, device=MPS))
    assert not [w for w in caught if "sparse_pool_metal" in str(w.message)]
    assert all(c == [] for c in calls)


def test_argument_checks_match_the_torch_ops():
    case = make_case(seed=77)
    pool, table, grid, offset, inv_scale = case.to(MPS)
    bad = [
        ("grid", (pool, table, grid.reshape(-1, 3), offset, inv_scale)),
        ("chunk_table", (pool, table.long(), grid, offset, inv_scale)),
        ("pool", (pool.float(), table, grid, offset, inv_scale)),
        ("pool", (pool[:, :, :, :, :33], table, grid, offset, inv_scale)),
        ("grid", (pool, table, grid.cpu().double(), offset, inv_scale)),  # MPS has no float64
        ("offset", (pool, table, grid, offset[:2], inv_scale)),
        ("inv_scale", (pool, table, grid, offset, inv_scale.cpu().double())),
        ("grid", (pool, table, grid.cpu(), offset, inv_scale)),
    ]
    for match, args in bad:
        for name in ("sample_u8", "sample_f32"):
            with pytest.raises(ValueError, match=match):
                getattr(metal, name)(*args)
            with pytest.raises(ValueError, match=match):
                getattr(ops, name)(*args)
    go = torch.zeros(1, *grid.shape[:3], device=MPS)
    for bad_go in (go.cpu().double(), go[:, :1], go.cpu()):
        with pytest.raises(ValueError, match="grad_output"):
            metal.sample_grad(pool, table, grid, offset, inv_scale, bad_go)
    o, s = torch.zeros(3, device=MPS), torch.ones(3, device=MPS)
    for args in ((grid.cpu().double(), table, o, s), (grid[..., :2], table, o, s), (grid, table.long(), o, s),
                 (grid, table, o[:2], s), (grid, table, o, s.cpu()), (grid.cpu(), table, o, s)):
        with pytest.raises(ValueError):
            metal.missing_chunks(*args)
        with pytest.raises(ValueError):
            ops.missing_chunks(*args)


# ----------------------------------------------------------------------------------- compile()

def test_compile_builds_once_and_records_contraction():
    lib = metal._LIB
    metal.compile()
    assert metal._LIB is lib and metal.CONTRACTION in ("off", "compiler default")


def test_compile_self_check_rejects_a_wrong_library(monkeypatch):
    # A library that compiles but samples without the +1 halo must not be used.
    halo = "const float lx = px - cxf * 32.0f + 1.0f;"
    assert metal._SOURCE.count(halo) == 1
    monkeypatch.setattr(metal, "_SOURCE", metal._SOURCE.replace(halo, "const float lx = px - cxf * 32.0f;"))
    monkeypatch.setattr(metal, "_LIB", None)
    with pytest.raises(RuntimeError, match="self-check") as info:
        metal.compile()
    message = str(info.value)
    assert "sample_f32 max |diff|" in message and "first bad element" in message and "grid [" in message
    assert metal._LIB is None


# ------------------------------------------------------------------------------- missing_chunks

def _missing_args(xyz, table, origin, spacing):
    return [torch.from_numpy(np.ascontiguousarray(a)).to(MPS) for a in (xyz, table, origin, spacing)]


def _assert_rows_in_grid(rows, table_shape):
    """sync() refuses chunks outside the grid: every row must be a cell of the table."""
    rows = rows.cpu()
    assert rows.dtype == torch.long and rows.dim() == 2 and rows.shape[1] == 3
    assert bool((rows >= 0).all()) and bool((rows < torch.tensor(table_shape)).all())


@pytest.mark.parametrize("spacing", [(2.0, 4.0, 0.75), (0.3, 1.7, 3.0), (5.5, 0.9, 2.2)])
def test_missing_chunks_matches_the_bruteforce_cuda_semantics(spacing):
    rng = np.random.default_rng(12)
    cZ, cY, cX = 5, 6, 7
    table = np.full((cZ, cY, cX), -1, np.int32)
    table[rng.random(table.shape) < 0.3] = 0
    origin = np.array([3.0, -5.0, 11.0], np.float32)
    spacing = np.array(spacing, np.float32)
    n = 3000
    span = np.array([cX, cY, cZ]) * 32
    local = rng.random((n, 3)) * span * 1.4 - 0.2 * span
    xyz = (local * spacing + origin).astype(np.float32)
    xyz[:8] = [[np.nan, 50, 50], [50, np.inf, 50], [50, 50, -np.inf], [1e30, 50, 50],
               [-1e30, -1e30, -1e30], [origin[0] - 1.0, 50, 50], [origin[0] - 70.0, 50, 50],
               [origin[0] + 64.0, origin[1] + 128.0, origin[2] + 24.0]]
    want = missing_chunks_reference(xyz, table, origin, spacing)
    args = _missing_args(xyz.reshape(10, 300, 3), table, origin, spacing)
    got = metal.missing_chunks(*args)
    assert got.dtype == torch.long and got.device.type == "cpu"  # on the host, where the cache needs them
    _assert_rows_in_grid(got, table.shape)
    np.testing.assert_array_equal(got.cpu().numpy(), want)
    assert torch.equal(got.cpu(), ops.missing_chunks(*args).cpu())
    assert len(want) > 0


@pytest.mark.parametrize("spacing", [(0.75, 0.3, 1.7), (3.0, 2.2, 0.9)])
def test_missing_chunks_at_chunk_boundaries(spacing):
    # Points within a few float32 steps of every chunk boundary, for spacings whose division is
    # inexact: the truncation must see (v - origin) / spacing / 32 rounded exactly as the CUDA
    # kernel computes it, in that order.
    origin = np.array([1.5, -2.25, 7.0], np.float32)
    spacing = np.array(spacing, np.float32)
    cZ, cY, cX = 9, 8, 7
    table = np.full((cZ, cY, cX), -1, np.int32)
    rows = []
    for axis, n in enumerate((cX, cY, cZ)):
        for k in range(n + 1):
            edge = np.float32(origin[axis] + np.float32(k * 32) * spacing[axis])
            for steps in range(-3, 4):
                v = edge
                for _ in range(abs(steps)):
                    v = np.nextafter(v, np.float32(np.inf if steps > 0 else -np.inf), dtype=np.float32)
                p = (origin + spacing * np.float32(40.0)).astype(np.float32)
                p[axis] = v
                rows.append(p)
    xyz = np.asarray(rows, np.float32)
    args = [torch.from_numpy(a).to(MPS) for a in (table, origin, spacing)]
    for p in xyz:  # one at a time, so that no point's marks hide another's
        want = missing_chunks_reference(p[None], table, origin, spacing)
        got = metal.missing_chunks(torch.from_numpy(p[None].copy()).to(MPS), *args).cpu().numpy()
        np.testing.assert_array_equal(got, want, err_msg=f"point {p.tolist()}")


def test_missing_chunks_special_points_one_at_a_time():
    # A non-finite coordinate takes chunk 0 on its axis, huge ones clamp, negatives truncate.
    table = np.full((5, 6, 7), -1, np.int32)
    origin = np.array([3.0, -5.0, 11.0], np.float32)
    spacing = np.array([2.0, 4.0, 0.75], np.float32)
    mid = origin + spacing * 32 * np.array([3.5, 2.5, 2.5], np.float32)
    specials = []
    for axis in range(3):
        for value in (np.nan, np.inf, -np.inf, 1e30, -1e30, 3.4e38, -3.4e38, origin[axis] - 1.0,
                      origin[axis] - 70.0 * spacing[axis], origin[axis] + 1e-30):
            p = mid.copy()
            p[axis] = value
            specials.append(p)
    args = [torch.from_numpy(a).to(MPS) for a in (table, origin, spacing)]
    for p in specials:
        want = missing_chunks_reference(p[None], table, origin, spacing)
        got = metal.missing_chunks(torch.from_numpy(p[None].astype(np.float32)).to(MPS), *args)
        np.testing.assert_array_equal(got.cpu().numpy(), want, err_msg=f"point {p.tolist()}")


def test_missing_chunks_single_point_marks_its_clipped_neighbourhood():
    table = torch.full((4, 4, 4), -1, dtype=torch.int32, device=MPS)
    table[0, 0, 1] = 3  # loaded: never returned
    xyz = torch.tensor([[1.0, 1.0, 1.0]], device=MPS)  # chunk (0, 0, 0)
    o, s = torch.zeros(3, device=MPS), torch.ones(3, device=MPS)
    got = metal.missing_chunks(xyz, table, o, s)
    want = [[z, y, x] for z in (0, 1) for y in (0, 1) for x in (0, 1) if (z, y, x) != (0, 0, 1)]
    assert got.cpu().tolist() == want
    for empty in (metal.missing_chunks(xyz[:0], table, o, s), metal.missing_chunks(xyz, table[:0], o, s)):
        assert empty.shape == (0, 3) and empty.dtype == torch.long and empty.device.type == "cpu"
    table.fill_(0)  # everything loaded: nothing missing
    none = metal.missing_chunks(xyz, table, o, s)
    assert none.shape == (0, 3) and none.dtype == torch.long and none.device.type == "cpu"


@pytest.mark.parametrize("loaded_frac", [0.0, 0.5])
def test_missing_chunks_with_every_cell_marked_by_a_different_thread(loaded_frac):
    # Points at chunks (1 + 3i, 1 + 3j, 1 + 3k) tile a 30^3 chunk grid with disjoint 3x3x3 blocks,
    # so neighbouring cells (and the bits of one mask word) are marked by different threads at once.
    n = 30
    rng = np.random.default_rng(int(loaded_frac * 10))
    table = np.where(rng.random((n, n, n)) < loaded_frac, 0, -1).astype(np.int32)
    centres = np.arange(1, n, 3)
    zz, yy, xx = np.meshgrid(centres, centres, centres, indexing="ij")
    xyz = (np.stack([xx, yy, zz], -1).reshape(-1, 3) * 32 + 16).astype(np.float32)
    args = _missing_args(xyz, table, np.zeros(3, np.float32), np.ones(3, np.float32))
    got = metal.missing_chunks(*args)
    _assert_rows_in_grid(got, table.shape)
    got = got.cpu().numpy()
    np.testing.assert_array_equal(got, np.argwhere(table < 0))
    assert torch.equal(torch.from_numpy(got), ops.missing_chunks(*args).cpu())


@pytest.mark.parametrize("shape", [(32, 32, 32), (64, 32, 32), (1, 1, 65500), (3, 5, 7)])
def test_missing_chunks_scan_block_edges(shape):
    # A point at every cell centre: every cell is needed, so the result is every unloaded cell. The
    # shapes put the scan's 1024 blocks at their edges: exactly one word per block (32^3), two
    # (64 x 32 x 32), a partial last block ending in a partial word (65,500 cells), most blocks empty.
    rng = np.random.default_rng(sum(shape))
    table = np.where(rng.random(shape) < 0.3, 0, -1).astype(np.int32)
    zz, yy, xx = np.meshgrid(*(np.arange(n) for n in shape), indexing="ij")
    xyz = (np.stack([xx, yy, zz], -1).reshape(-1, 3) * 32 + 16).astype(np.float32)
    args = _missing_args(xyz, table, np.zeros(3, np.float32), np.ones(3, np.float32))
    got = metal.missing_chunks(*args)
    _assert_rows_in_grid(got, table.shape)
    np.testing.assert_array_equal(got.numpy(), np.argwhere(table < 0))
    assert bool(table.reshape(-1)[-32:].min() < 0)  # the last word has rows


def test_missing_chunks_on_the_cos_table_shape():
    # The PHerc0826 cos level's chunk grid (265, 128, 128): 4.34 M cells, 136k mask words.
    rng = np.random.default_rng(5)
    table = np.where(rng.random((265, 128, 128)) < 0.6, 1, -1).astype(np.int32)
    origin = np.zeros(3, np.float32)
    spacing = np.full(3, 2.0, np.float32)
    centre = np.array([2000.0, 2000.0, 7000.0])
    xyz = (centre + rng.normal(0, 600, (100_000, 3)) * np.array([1.0, 1.0, 3.0])).astype(np.float32)
    xyz[::1000] = (rng.random((100, 3)) * np.array([8200, 8200, 17000]) - 50).astype(np.float32)
    args = _missing_args(xyz, table, origin, spacing)
    want = ops.missing_chunks(*[t.cpu() for t in args])
    assert want.shape[0] > 10_000
    # 30 calls: torch.nonzero on MPS sometimes returns extra rows here (torch 2.11); the kernels'
    # own compaction must be right every time.
    for _ in range(30):
        got = metal.missing_chunks(*args)
        _assert_rows_in_grid(got, table.shape)
        assert torch.equal(got.cpu(), want)


def _sheet(n, shape, rng):
    """n full-res points (spacing 2) on a wavy sheet through the middle of a chunk grid `shape`
    (cZ, cY, cX): the marked range then holds a slice of the grid's cells, not all of them."""
    size = np.array(shape[::-1], np.float64) * 64.0  # full-res extent (x, y, z)
    u, v = rng.random(n), rng.random(n)
    x = size[0] * (0.3 + 0.2 * u)
    y = size[1] * (0.3 + 0.2 * v)
    z = size[2] * (0.5 + 0.03 * np.sin(9.0 * u) * np.cos(7.0 * v))
    return np.stack([x, y, z], 1).astype(np.float32)


@pytest.mark.parametrize("n", [2_500, 100_000, 1_000_000])
@pytest.mark.parametrize("shape", [(265, 128, 128), (3, 4, 5)])
def test_missing_chunks_rows_equal_the_torch_ops(n, shape):
    # The fit's call sizes, on the PHerc0826 cos level's chunk grid and on a tiny one: exactly the
    # torch ops' rows (computed on the CPU), in the same order, all inside the grid.
    rng = np.random.default_rng(n + shape[0])
    table = np.where(rng.random(shape) < 0.5, 0, -1).astype(np.int32)
    origin, spacing = np.zeros(3, np.float32), np.full(3, 2.0, np.float32)
    xyz = _sheet(n, shape, rng)
    got = metal.missing_chunks(*_missing_args(xyz, table, origin, spacing))
    _assert_rows_in_grid(got, shape)
    want = ops.missing_chunks(*(torch.from_numpy(a) for a in (xyz, table, origin, spacing)))
    assert want.shape[0] > 0 and torch.equal(got, want)


def test_missing_chunks_two_far_apart_clusters():
    # Two clusters at opposite corners of the cos grid: the marked range spans almost all of it.
    rng = np.random.default_rng(3)
    shape = (265, 128, 128)
    table = np.where(rng.random(shape) < 0.5, 0, -1).astype(np.int32)
    origin, spacing = np.zeros(3, np.float32), np.full(3, 2.0, np.float32)
    corner = np.array(shape[::-1], np.float64) * 64.0  # full-res (x, y, z) of the far corner
    near = rng.random((3000, 3)) * 200.0 + 70.0
    far = corner - 70.0 - rng.random((3000, 3)) * 200.0
    xyz = np.concatenate([near, far]).astype(np.float32)
    got = metal.missing_chunks(*_missing_args(xyz, table, origin, spacing))
    _assert_rows_in_grid(got, shape)
    want = ops.missing_chunks(*(torch.from_numpy(a) for a in (xyz, table, origin, spacing)))
    assert torch.equal(got, want)
    assert int(want[0, 0]) <= 1 and int(want[-1, 0]) >= shape[0] - 2  # the range spans the grid


@pytest.mark.parametrize("staged_rows", [1, 7, 1 << 16])
def test_missing_chunks_with_more_rows_than_the_staging_buffer(staged_rows, monkeypatch):
    # Rows past the staging buffer take a second write and copy; the result is the same, on the CPU.
    monkeypatch.setattr(metal, "_STAGED_ROWS", staged_rows)
    rng = np.random.default_rng(8)
    table = np.where(rng.random((9, 10, 11)) < 0.5, 0, -1).astype(np.int32)
    xyz = (rng.random((500, 3)) * np.array([352, 320, 288])).astype(np.float32)
    args = _missing_args(xyz, table, np.zeros(3, np.float32), np.ones(3, np.float32))
    got = metal.missing_chunks(*args)
    assert got.device.type == "cpu" and got.shape[0] > 7
    np.testing.assert_array_equal(got.numpy(), missing_chunks_reference(xyz, table, np.zeros(3), np.ones(3)))


def test_missing_chunks_non_contiguous_points():
    rng = np.random.default_rng(6)
    table = torch.from_numpy(np.where(rng.random((6, 7, 8)) < 0.4, 0, -1).astype(np.int32)).to(MPS)
    xyz = torch.from_numpy((rng.random((40, 50, 3)) * np.array([256, 224, 192])).astype(np.float32)).to(MPS)
    o, s = torch.tensor([1.0, 2.0, 3.0], device=MPS), torch.tensor([1.0, 1.0, 1.0], device=MPS)
    wide = torch.zeros(40, 50, 4, device=MPS)
    wide[..., 1:] = xyz
    for view in (xyz[:, ::2], xyz[1:, 3:], wide[..., 1:], xyz.permute(1, 0, 2), xyz[5, 7]):
        assert torch.equal(metal.missing_chunks(view, table, o, s), metal.missing_chunks(view.contiguous(), table, o, s))
        assert torch.equal(metal.missing_chunks(view, table, o, s), ops.missing_chunks(view, table, o, s).cpu())


# ------------------------------------------------------------------- 64-bit pool offsets (opt-in)

@pytest.mark.skipif(os.environ.get("LASAGNA_METAL_BIG_POOL") != "1",
                    reason="needs about 4.3 GB of GPU memory; set LASAGNA_METAL_BIG_POOL=1")
@pytest.mark.parametrize("C", [1, 3])
def test_pool_offsets_beyond_4_gib(C):
    # The last slot of a pool over 4 GiB starts past 2^32 bytes: 32-bit offsets would wrap.
    slots = (1 << 32) // (C * ops.BRICK) + 2
    assert (slots - 1) * C * ops.BRICK > 1 << 32
    volume, loaded, pool, table = _exact_case(C=C, seed=9)
    cz, cy, cx = (int(v) for v in np.argwhere(loaded)[3])
    slot = int(table[cz, cy, cx])
    one = torch.from_numpy(np.argwhere(loaded)[3])
    pts = (one.flip(0).double() * 32 + torch.rand(4000, 3, generator=torch.Generator().manual_seed(2)) * 32).float()
    grid = pts.reshape(1, 1, -1, 3).to(MPS)
    zeros, ones = torch.zeros(3, device=MPS), torch.ones(3, device=MPS)
    small_table = torch.full(table.shape, -1, dtype=torch.int32)
    small_table[cz, cy, cx] = 0
    go = torch.randn(C, 1, 1, 4000, generator=torch.Generator().manual_seed(3)).to(MPS)
    small = (pool[slot:slot + 1].to(MPS), small_table.to(MPS), grid, zeros, ones)
    want = {name: _run(metal, name, small, go) for name in FUNCTIONS}
    big = torch.zeros((slots, C, 34, 34, 34), dtype=torch.uint8, device=MPS)
    try:
        big[-1] = small[0][0]
        big_table = small_table.clone()
        big_table[cz, cy, cx] = slots - 1
        args = (big, big_table.to(MPS), grid, zeros, ones)
        for name in FUNCTIONS:
            assert torch.equal(_run(metal, name, args, go), want[name]), name
    finally:  # one 4.3 GB pool at a time: give it back before the next case
        del big
        args = None
        torch.mps.synchronize()
        torch.mps.empty_cache()
