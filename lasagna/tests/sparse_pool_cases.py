"""Shared cases for the pooled sparse samplers (torch ops and Metal).

make_case() builds a random uint8 volume, packs a chosen subset of its 32^3
chunks into a pool + slot table (slots deliberately not in row-major order),
and sample positions that cover every branch of the kernels: in-chunk points,
exact voxel and chunk boundaries, half-voxel positions, the partial last chunk,
points just outside the volume and outside the chunk grid, unloaded chunks,
and non-finite or huge coordinates.

dense_reference() is an independent float64 formulation: trilinear
interpolation of the zero-extended dense volume at p = (grid - offset) *
inv_scale (computed in float32, as the kernels do), zero where p is not
finite, outside the chunk grid, or in a chunk that is not loaded.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

CHUNK = 32
PADDED = 34


@dataclass
class Case:
    volume: torch.Tensor       # (C, Z, Y, X) uint8, CPU
    pool: torch.Tensor         # (S, C, 34, 34, 34) uint8, CPU
    chunk_table: torch.Tensor  # (cZ, cY, cX) int32, CPU
    grid: torch.Tensor         # (D, H, W, 3) float32 full-res (x, y, z), CPU
    offset: torch.Tensor       # (3,) float32
    inv_scale: torch.Tensor    # (3,) float32
    loaded: np.ndarray         # (cZ, cY, cX) bool

    def to(self, device):
        d = torch.device(device)
        return (self.pool.to(d), self.chunk_table.to(d), self.grid.to(d),
                self.offset.to(d), self.inv_scale.to(d))


def padded_brick(volume: np.ndarray, coord) -> np.ndarray:
    """(C, 34, 34, 34) brick of chunk `coord`: 1-voxel margin, zero (fill) outside the array."""
    C = volume.shape[0]
    out = np.zeros((C, PADDED, PADDED, PADDED), np.uint8)
    src = tuple(slice(max(0, c * CHUNK - 1), min(n, c * CHUNK + CHUNK + 1))
                for c, n in zip(coord, volume.shape[1:]))
    dst = tuple(slice(s.start - (c * CHUNK - 1), s.stop - (c * CHUNK - 1)) for s, c in zip(src, coord))
    out[(slice(None), *dst)] = volume[(slice(None), *src)]
    return out


def pack(volume: torch.Tensor, loaded: np.ndarray, seed: int = 0):
    """Pool + slot table holding the chunks where `loaded` is True, in shuffled slot order."""
    vol = volume.numpy()
    C = vol.shape[0]
    coords = [tuple(int(v) for v in c) for c in np.argwhere(loaded)]
    order = np.random.default_rng(seed).permutation(len(coords))
    pool = np.zeros((len(coords), C, PADDED, PADDED, PADDED), np.uint8)
    table = np.full(loaded.shape, -1, np.int32)
    for slot, i in enumerate(order):
        pool[slot] = padded_brick(vol, coords[i])
        table[coords[i]] = slot
    return torch.from_numpy(pool), torch.from_numpy(table)


def chunk_grid(shape_zyx):
    return tuple((n + CHUNK - 1) // CHUNK for n in shape_zyx)


def edge_points(shape_zyx, rng) -> np.ndarray:
    """Local (x, y, z) voxel positions on the kernels' edges."""
    Z, Y, X = shape_zyx
    size = np.array([X, Y, Z], np.float64)
    pts = []
    for v in (0.0, 0.5, 1.0, 31.0, 31.5, 31.999, 32.0, 32.0001, 33.0, 63.5, 64.0):
        pts.append([v, v, v])
        pts.append([v, size[1] / 2 + 0.25, size[2] / 2 + 0.25])
    pts += [
        [-1e-3, 5.0, 5.0], [-0.5, 5.0, 5.0], [-1.0, 5.0, 5.0], [5.0, -0.25, 5.0], [5.0, 5.0, -31.9],
        list(size - 1.0), list(size - 0.5), list(size - 1e-3), list(size), list(size + 0.5),
        list(np.ceil(size / CHUNK) * CHUNK - 0.5),  # inside the chunk grid, past the array
        list(np.ceil(size / CHUNK) * CHUNK),        # first position outside the chunk grid
        [np.nan, 5.0, 5.0], [5.0, np.inf, 5.0], [5.0, 5.0, -np.inf], [1e30, 5.0, 5.0], [-1e30, -1e30, -1e30],
    ]
    pts += [list(rng.integers(0, size).astype(np.float64)) for _ in range(16)]  # exact voxel centres
    return np.asarray(pts, np.float64)


def make_case(*, seed=0, C=1, shape_zyx=(65, 70, 129), offset=(5.0, -3.0, 7.0),
              inv_scale=(0.5, 0.25, 1.0), loaded_frac=0.7, D=3, H=17, W=19) -> Case:
    """Random case. Sizes of 32k + 1 keep F.grid_sample's normalisation exact."""
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    Z, Y, X = shape_zyx
    volume = torch.randint(0, 256, (C, Z, Y, X), generator=gen, dtype=torch.uint8)
    grid_shape = chunk_grid(shape_zyx)
    loaded = rng.random(grid_shape) < loaded_frac
    pool, table = pack(volume, loaded, seed=seed)
    n = D * H * W
    size = np.array([X, Y, Z], np.float64)
    local = rng.random((n, 3)) * (size + 4.0) - 2.0  # about two voxels past every face
    edges = edge_points(shape_zyx, rng)
    local[: len(edges)] = edges[: n]
    off = np.asarray(offset, np.float32)
    inv = np.asarray(inv_scale, np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        full = (local / inv.astype(np.float64) + off.astype(np.float64)).astype(np.float32)
    grid = torch.from_numpy(full.reshape(D, H, W, 3))
    return Case(volume, pool, table, grid, torch.from_numpy(off), torch.from_numpy(inv), loaded)


def local_coords(case: Case) -> torch.Tensor:
    """p = (grid - offset) * inv_scale in float32, as the kernels compute it: (N, 3)."""
    return (case.grid.reshape(-1, 3) - case.offset.view(1, 3)) * case.inv_scale.view(1, 3)


def sample_mask(case: Case) -> np.ndarray:
    """(N,) bool: the samples the kernels interpolate (finite, in the chunk grid, chunk loaded)."""
    p = local_coords(case).double().numpy()
    finite = np.isfinite(p).all(axis=1)
    cZ, cY, cX = case.loaded.shape
    with np.errstate(invalid="ignore"):
        ci = np.floor(np.where(finite[:, None], p, 0.0) / CHUNK).astype(np.int64)
    inside = finite & (ci[:, 0] >= 0) & (ci[:, 0] < cX) & (ci[:, 1] >= 0) & (ci[:, 1] < cY) \
        & (ci[:, 2] >= 0) & (ci[:, 2] < cZ)
    ok = inside.copy()
    ok[inside] = case.loaded[ci[inside, 2], ci[inside, 1], ci[inside, 0]]
    return ok


def dense_reference(case: Case, grad_output: np.ndarray | None = None):
    """float64 values (C, N) and, given grad_output (C, N), the grid gradient (N, 3)."""
    vol = case.volume.numpy().astype(np.float64)
    C, Z, Y, X = vol.shape
    p = local_coords(case).double().numpy()
    mask = sample_mask(case)
    N = p.shape[0]
    values = np.zeros((C, N))
    grad = np.zeros((N, 3))
    q = p[mask]
    i0 = np.floor(q).astype(np.int64)
    f = q - i0

    def corner(c, dx, dy, dz):
        ix, iy, iz = i0[:, 0] + dx, i0[:, 1] + dy, i0[:, 2] + dz
        ok = (ix >= 0) & (ix < X) & (iy >= 0) & (iy < Y) & (iz >= 0) & (iz < Z)
        out = np.zeros(len(q))
        out[ok] = vol[c, iz[ok], iy[ok], ix[ok]]
        return out

    fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]
    inv = case.inv_scale.double().numpy()
    for c in range(C):
        v = {(dx, dy, dz): corner(c, dx, dy, dz) for dx in (0, 1) for dy in (0, 1) for dz in (0, 1)}
        val = np.zeros(len(q))
        dfx = np.zeros(len(q))
        dfy = np.zeros(len(q))
        dfz = np.zeros(len(q))
        for (dx, dy, dz), vv in v.items():
            wx = fx if dx else 1.0 - fx
            wy = fy if dy else 1.0 - fy
            wz = fz if dz else 1.0 - fz
            val += wx * wy * wz * vv
            dfx += (1.0 if dx else -1.0) * wy * wz * vv
            dfy += wx * (1.0 if dy else -1.0) * wz * vv
            dfz += wx * wy * (1.0 if dz else -1.0) * vv
        values[c, mask] = val
        if grad_output is not None:
            go = grad_output[c, mask]
            grad[mask, 0] += go * dfx * inv[0]
            grad[mask, 1] += go * dfy * inv[1]
            grad[mask, 2] += go * dfz * inv[2]
    return values, grad


def missing_chunks_reference(xyz: np.ndarray, chunk_table: np.ndarray, origin, spacing) -> np.ndarray:
    """Brute-force sparse_prefetch_chunks_kernel.cu: (M, 3) int64 (cz, cy, cx), row-major order."""
    cZ, cY, cX = chunk_table.shape
    pts = np.asarray(xyz, np.float32).reshape(-1, 3)
    o = np.asarray(origin, np.float32)
    s = np.asarray(spacing, np.float32)
    needed = np.zeros((cZ, cY, cX), bool)
    for x, y, z in pts:
        c = []
        for v, oo, ss, n in ((x, o[0], s[0], cX), (y, o[1], s[1], cY), (z, o[2], s[2], cZ)):
            if not np.isfinite(v):
                k = 0
            else:
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    q = np.float32(np.float32(np.float32(v - oo) / ss) / np.float32(CHUNK))
                k = 0 if np.isnan(q) else (n - 1 if q >= n else (0 if q <= 0 else int(np.trunc(q))))
            c.append(min(max(k, 0), n - 1))
        cx, cy, cz = c
        needed[max(cz - 1, 0):cz + 2, max(cy - 1, 0):cy + 2, max(cx - 1, 0):cx + 2] = True
    return np.argwhere(needed & (chunk_table < 0)).astype(np.int64)
