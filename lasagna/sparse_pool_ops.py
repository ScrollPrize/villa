"""Sparse chunk-pool 3D sampling in torch ops, for devices without the CUDA kernels.

The CUDA sparse cache (sparse_cache.SparseChunkGroupCache) keeps a table of raw
device pointers that its .cu kernels dereference. On CPU and MPS the chunks
live in one pooled tensor instead, and the table holds slot indices:

    pool         uint8 (S, C, 34, 34, 34). A slot holds one 32^3 chunk of all C
                 channels with a one-voxel margin on every side: voxel
                 (iz, iy, ix) of chunk (cz, cy, cx)'s slot is volume voxel
                 (32*cz - 1 + iz, 32*cy - 1 + iy, 32*cx - 1 + ix), or the
                 array's fill value where that lies outside the array.
    chunk_table  int32 (cZ, cY, cX): the slot of each chunk, -1 if not loaded.

sample_u8, sample_f32, sample_grad and missing_chunks follow
sparse_grid_sample_3d_u8_kernel.cu, the forward and backward kernels of
sparse_grid_sample_3d_u8_diff_kernel.cu, and sparse_prefetch_chunks_kernel.cu
term by term: the same float32 operations in the same order. Results agree
with the CUDA kernels up to float rounding, since nvcc contracts mul+add into
FMAs. These functions are the CPU path, the MPS path when the Metal kernels are
off, and the reference the Metal kernels are tested against.

Samples are processed in batches of _BATCH points to bound the temporaries.
"""
from __future__ import annotations

from itertools import product

import torch

CHUNK = 32
PADDED = CHUNK + 2
BRICK = PADDED ** 3  # bytes per channel per slot
EMPTY = -1
_STRIDE_Z = PADDED * PADDED
_STRIDE_Y = PADDED
_MAX_LOCAL = PADDED - 1  # 33: last voxel index inside a padded chunk
_BATCH = 1 << 20


def _check(pool, chunk_table, grid, offset, inv_scale) -> None:
    if pool.dtype != torch.uint8 or pool.dim() != 5 or tuple(pool.shape[2:]) != (PADDED,) * 3:
        raise ValueError(
            f"pool must be uint8 (S, C, {PADDED}, {PADDED}, {PADDED}), got {pool.dtype} {tuple(pool.shape)}")
    if not pool.is_contiguous():
        raise ValueError("pool must be contiguous")
    if chunk_table.dtype != torch.int32 or chunk_table.dim() != 3:
        raise ValueError(f"chunk_table must be int32 (cZ, cY, cX), got {chunk_table.dtype} {tuple(chunk_table.shape)}")
    if grid.dtype != torch.float32 or grid.dim() != 4 or grid.size(3) != 3:
        raise ValueError(f"grid must be float32 (D, H, W, 3), got {grid.dtype} {tuple(grid.shape)}")
    for name, t in (("offset", offset), ("inv_scale", inv_scale)):
        if t.dtype != torch.float32 or t.numel() != 3:
            raise ValueError(f"{name} must be float32 (3,)")
    for name, t in (("chunk_table", chunk_table), ("grid", grid), ("offset", offset), ("inv_scale", inv_scale)):
        if t.device != pool.device:
            raise ValueError(f"{name} is on {t.device}, the pool on {pool.device}")


def _locate(points, chunk_table, n_channels, offset, inv_scale, *, clamp_first):
    """The kernels' addressing for (n, 3) float32 points.

    Returns valid (n,) bool; idx (n, 8) int64, the flattened-pool offsets of the
    eight corners of channel 0 in the kernels' order (000, 100, 010, 110, 001,
    101, 011, 111 as xyz bits); and the fractions fx, fy, fz (n,) float32.
    Invalid samples (non-finite, outside the chunk grid, chunk not loaded) get
    in-range dummy offsets and must be zeroed by the caller.

    clamp_first: the diff kernels clamp ix0 to [0, 33] before ix1 = ix0 + 1; the
    uint8 kernel takes ix1 = ix0 + 1 first and clamps both. Both orders give the
    same corners whenever the local coordinate is >= 0, which floor(p / 32)
    guarantees; each kernel's order is kept anyway.
    """
    cZ, cY, cX = (int(v) for v in chunk_table.shape)
    p = (points - offset.view(1, 3)) * inv_scale.view(1, 3)
    finite = torch.isfinite(p).all(dim=1)
    p = torch.where(finite.unsqueeze(1), p, torch.zeros_like(p))
    px, py, pz = p.unbind(1)
    # (int)floorf(p / 32.0f), bounds-checked while still float so that huge
    # coordinates never reach an integer conversion.
    cxf = torch.floor(px / float(CHUNK))
    cyf = torch.floor(py / float(CHUNK))
    czf = torch.floor(pz / float(CHUNK))
    in_grid = (
        finite
        & (cxf >= 0) & (cxf < cX)
        & (cyf >= 0) & (cyf < cY)
        & (czf >= 0) & (czf < cZ)
    )
    zero = torch.zeros_like(cxf)
    cxf = torch.where(in_grid, cxf, zero)
    cyf = torch.where(in_grid, cyf, zero)
    czf = torch.where(in_grid, czf, zero)
    px = torch.where(in_grid, px, zero)
    py = torch.where(in_grid, py, zero)
    pz = torch.where(in_grid, pz, zero)
    lin = (czf.long() * cY + cyf.long()) * cX + cxf.long()
    slot = chunk_table.reshape(-1)[lin].long()
    valid = in_grid & (slot >= 0)
    slot = torch.where(valid, slot, torch.zeros_like(slot))

    # Local coordinates in the padded chunk: px - ci_x * 32.0f + 1.0f.
    lx = px - cxf * float(CHUNK) + 1.0
    ly = py - cyf * float(CHUNK) + 1.0
    lz = pz - czf * float(CHUNK) + 1.0
    lx0f = torch.floor(lx)
    ly0f = torch.floor(ly)
    lz0f = torch.floor(lz)
    fx = lx - lx0f
    fy = ly - ly0f
    fz = lz - lz0f
    corners = []
    for l0f in (lx0f, ly0f, lz0f):
        i0 = l0f.long()
        if clamp_first:
            i0 = i0.clamp(0, _MAX_LOCAL)
            i1 = (i0 + 1).clamp(0, _MAX_LOCAL)
        else:
            i1 = (i0 + 1).clamp(0, _MAX_LOCAL)
            i0 = i0.clamp(0, _MAX_LOCAL)
        corners.append((i0, i1))
    (ix0, ix1), (iy0, iy1), (iz0, iz1) = corners
    base = slot * (n_channels * BRICK) + iz0 * _STRIDE_Z + iy0 * _STRIDE_Y + ix0
    sx = ix1 - ix0
    sy = (iy1 - iy0) * _STRIDE_Y
    sz = (iz1 - iz0) * _STRIDE_Z
    zeros = torch.zeros_like(sx)
    steps = torch.stack([zeros, sx, sy, sx + sy, sz, sz + sx, sz + sy, sz + sy + sx], dim=1)
    idx = base.unsqueeze(1) + steps
    return valid, idx, fx, fy, fz


def _weights(fx, fy, fz):
    """The trilinear weights w000 .. w111, each evaluated as in the kernels."""
    ofx = 1.0 - fx
    ofy = 1.0 - fy
    ofz = 1.0 - fz
    return (
        ofx * ofy * ofz,
        fx * ofy * ofz,
        ofx * fy * ofz,
        fx * fy * ofz,
        ofx * ofy * fz,
        fx * ofy * fz,
        ofx * fy * fz,
        fx * fy * fz,
    )


def _interpolate(flat, idx, channel, weights):
    """sum_k w_k * v_k over the eight corners, accumulated in the kernels' order."""
    v = flat[idx + channel * BRICK].to(torch.float32)
    val = weights[0] * v[:, 0]
    for k in range(1, 8):
        val = val + weights[k] * v[:, k]
    return val


def _round_half_away(val):
    """roundf for val >= 0 (all trilinear values here), exactly: floor, then up if frac >= 0.5."""
    t = torch.floor(val)
    return t + (val - t >= 0.5).to(val.dtype)


def _sample(pool, chunk_table, grid, offset, inv_scale, *, as_uint8):
    _check(pool, chunk_table, grid, offset, inv_scale)
    S, C = int(pool.shape[0]), int(pool.shape[1])
    D, H, W = (int(v) for v in grid.shape[:3])
    N = D * H * W
    dtype = torch.uint8 if as_uint8 else torch.float32
    out = torch.zeros((C, N), dtype=dtype, device=pool.device)
    if N == 0 or S == 0 or chunk_table.numel() == 0:
        return out.reshape(C, D, H, W)
    flat = pool.reshape(-1)
    points = grid.reshape(N, 3)
    for start in range(0, N, _BATCH):
        stop = min(N, start + _BATCH)
        valid, idx, fx, fy, fz = _locate(
            points[start:stop], chunk_table, C, offset, inv_scale, clamp_first=not as_uint8)
        weights = _weights(fx, fy, fz)
        for c in range(C):
            val = _interpolate(flat, idx, c, weights)
            if as_uint8:
                val = _round_half_away(val).clamp(0.0, 255.0)
            val = torch.where(valid, val, torch.zeros_like(val))
            out[c, start:stop] = val.to(dtype)
    return out.reshape(C, D, H, W)


def sample_u8(pool, chunk_table, grid, offset, inv_scale):
    """sparse_grid_sample_3d_u8: (C, D, H, W) uint8.

    grid: (D, H, W, 3) float32 full-resolution (x, y, z); sampled at local voxel
    coordinates p = (grid - offset) * inv_scale. Trilinear in the chunk
    floor(p / 32), rounded half away from zero and clamped to [0, 255]. Zero
    where p is not finite, outside the chunk grid, or its chunk is not loaded.
    """
    return _sample(pool, chunk_table, grid, offset, inv_scale, as_uint8=True)


def sample_f32(pool, chunk_table, grid, offset, inv_scale):
    """sparse_grid_sample_3d_u8_diff's forward: (C, D, H, W) float32, unrounded."""
    return _sample(pool, chunk_table, grid, offset, inv_scale, as_uint8=False)


def sample_grad(pool, chunk_table, grid, offset, inv_scale, grad_output):
    """sparse_grid_sample_3d_u8_diff's backward: d(sum grad_output * out)/d grid, (D, H, W, 3) float32.

    Per sample, sum over channels of grad_output[c] * (d interp / d fx, fy, fz)
    * inv_scale, the derivative of the trilinear interpolant inside its cell.
    Zero for the samples sample_f32 zeroes. No gradient for the pool, the
    table, offset or inv_scale.
    """
    _check(pool, chunk_table, grid, offset, inv_scale)
    S, C = int(pool.shape[0]), int(pool.shape[1])
    D, H, W = (int(v) for v in grid.shape[:3])
    N = D * H * W
    if grad_output.dtype != torch.float32 or tuple(grad_output.shape) != (C, D, H, W):
        raise ValueError(f"grad_output must be float32 {(C, D, H, W)}, got {grad_output.dtype} {tuple(grad_output.shape)}")
    if grad_output.device != pool.device:
        raise ValueError(f"grad_output is on {grad_output.device}, the pool on {pool.device}")
    grad = torch.zeros((N, 3), dtype=torch.float32, device=pool.device)
    if N == 0 or S == 0 or chunk_table.numel() == 0:
        return grad.reshape(D, H, W, 3)
    flat = pool.reshape(-1)
    points = grid.reshape(N, 3)
    go_all = grad_output.reshape(C, N)
    isx, isy, isz = inv_scale.reshape(3).unbind(0)
    for start in range(0, N, _BATCH):
        stop = min(N, start + _BATCH)
        valid, idx, fx, fy, fz = _locate(
            points[start:stop], chunk_table, C, offset, inv_scale, clamp_first=True)
        ofx = 1.0 - fx
        ofy = 1.0 - fy
        ofz = 1.0 - fz
        dgx = torch.zeros_like(fx)
        dgy = torch.zeros_like(fx)
        dgz = torch.zeros_like(fx)
        for c in range(C):
            go = go_all[c, start:stop]
            v = flat[idx + c * BRICK].to(torch.float32)
            v000, v100, v010, v110, v001, v101, v011, v111 = v.unbind(1)
            dfx = (ofy * ofz * (v100 - v000)
                   + fy * ofz * (v110 - v010)
                   + ofy * fz * (v101 - v001)
                   + fy * fz * (v111 - v011))
            dfy = (ofx * ofz * (v010 - v000)
                   + fx * ofz * (v110 - v100)
                   + ofx * fz * (v011 - v001)
                   + fx * fz * (v111 - v101))
            dfz = (ofx * ofy * (v001 - v000)
                   + fx * ofy * (v101 - v100)
                   + ofx * fy * (v011 - v010)
                   + fx * fy * (v111 - v110))
            dgx = dgx + go * dfx * isx
            dgy = dgy + go * dfy * isy
            dgz = dgz + go * dfz * isz
        g = torch.stack([dgx, dgy, dgz], dim=1)
        grad[start:stop] = torch.where(valid.unsqueeze(1), g, torch.zeros_like(g))
    return grad.reshape(D, H, W, 3)


def missing_chunks(xyz, chunk_table, origin, spacing):
    """sparse_prefetch_chunks' missing_chunks: (M, 3) int64 (cz, cy, cx), row-major order.

    For every point, chunk coordinate trunc((v - origin) / spacing / 32) per
    axis (truncation toward zero, 0 for a non-finite coordinate), clamped to the
    chunk grid on each axis separately; that chunk and its 26 neighbours inside
    the grid are needed. Returns the needed chunks whose table entry is empty.
    """
    if xyz.dtype != torch.float32 or xyz.size(-1) != 3:
        raise ValueError(f"xyz must be float32 (..., 3), got {xyz.dtype} {tuple(xyz.shape)}")
    if chunk_table.dtype != torch.int32 or chunk_table.dim() != 3:
        raise ValueError("chunk_table must be int32 (cZ, cY, cX)")
    for name, t in (("origin", origin), ("spacing", spacing)):
        if t.dtype != torch.float32 or t.numel() != 3:
            raise ValueError(f"{name} must be float32 (3,)")
    dev = chunk_table.device
    for name, t in (("xyz", xyz), ("origin", origin), ("spacing", spacing)):
        if t.device != dev:
            raise ValueError(f"{name} is on {t.device}, chunk_table on {dev}")
    cZ, cY, cX = (int(v) for v in chunk_table.shape)
    pts = xyz.reshape(-1, 3)
    if pts.shape[0] == 0 or cZ * cY * cX == 0:
        return torch.empty((0, 3), dtype=torch.long, device=dev)
    origin = origin.reshape(3)
    spacing = spacing.reshape(3)
    coords = []
    for axis, n in ((0, cX), (1, cY), (2, cZ)):
        v = pts[:, axis]
        q = (v - origin[axis]) / spacing[axis] / float(CHUNK)
        q = torch.where(torch.isfinite(v), q, torch.zeros_like(q))
        # (int) of a NaN (0 / 0) is 0 on CUDA; infinities saturate, then clamp.
        q = torch.nan_to_num(q, nan=0.0)
        coords.append(torch.trunc(q).clamp(0, n - 1).long())
    cx, cy, cz = coords
    # Mark the 26-neighbourhood inside the bounding box of the chunks hit (plus
    # one chunk, clipped to the grid), the same set the CUDA kernel marks on
    # the whole grid.
    lo_hi = torch.stack([cz.min(), cy.min(), cx.min(), cz.max(), cy.max(), cx.max()]).tolist()
    lo = [max(0, int(lo_hi[i]) - 1) for i in range(3)]
    hi = [min(n - 1, int(lo_hi[3 + i]) + 1) for i, n in enumerate((cZ, cY, cX))]
    box = [hi[i] - lo[i] + 1 for i in range(3)]
    centers = torch.zeros(box, dtype=torch.bool, device=dev)
    centers[cz - lo[0], cy - lo[1], cx - lo[2]] = True
    needed = torch.zeros_like(centers)
    for d in product((-1, 0, 1), repeat=3):
        dst = []
        src = []
        for axis, step in enumerate(d):
            n = box[axis]
            dst.append(slice(max(0, step), n + min(0, step)))
            src.append(slice(max(0, -step), n - max(0, step)))
        needed[tuple(dst)] |= centers[tuple(src)]
    table = chunk_table[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1]
    # Compact on the CPU: torch.nonzero on MPS (torch 2.11) intermittently returns extra
    # garbage rows for large masks. The mask is at most the chunk grid (a few MB).
    missing = torch.nonzero((needed & (table < 0)).cpu())
    return (missing + torch.tensor(lo, dtype=torch.long)).to(dev)
