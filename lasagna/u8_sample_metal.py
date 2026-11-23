"""Native Metal uint8 3D trilinear sampling for MPS tensors.

MSL ports of the CUDA kernels (``grid_sample_3d_u8*_kernel.cu``, ``sparse_grid_sample_3d_u8*_kernel.cu`` and
``sparse_prefetch_chunks_kernel.cu``), with the same float32 arithmetic, corner choice, clamping and zeroing
rules. The source is compiled once per process with ``torch.mps.compile_shader``, so no Metal toolchain or C++
build is needed, and every dispatch is encoded on PyTorch's current MPS stream, ordered with the surrounding MPS
ops. The sparse kernels read the slot-indexed layout described in ``u8_sample_torch``. Gradients flow only to the
sample positions, as in the CUDA kernels.
"""
from __future__ import annotations

import numpy as np
import torch

CHUNK = 32
PADDED = CHUNK + 2

_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

// geometry for one call; strides are in elements of the (D, H, W, 3) grid tensor
struct Dims {
    long a0, a1, a2, a3;   // sparse: cZ, cY, cX, C; dense: C, Z, Y, X
    long n;                // D * H * W
    long w_grid, h_grid;
    long sd, sh, sw, sc;
};

inline float3 load_pos(device const float* grid, constant Dims& d, uint n,
                       device const float* offset, device const float* inv_scale) {
    long w = long(n) % d.w_grid;
    long t = long(n) / d.w_grid;
    long h = t % d.h_grid;
    long dd = t / d.h_grid;
    long b = dd * d.sd + h * d.sh + w * d.sw;
    float px = (grid[b + 0 * d.sc] - offset[0]) * inv_scale[0];
    float py = (grid[b + 1 * d.sc] - offset[1]) * inv_scale[1];
    float pz = (grid[b + 2 * d.sc] - offset[2]) * inv_scale[2];
    return float3(px, py, pz);
}

// sparse: locate the padded brick and its in-brick corners; false means the point samples to zero
struct Brick {
    ulong base;                // byte offset of channel 0 of the brick in the pool
    int ix0, iy0, iz0, ix1, iy1, iz1;
    float fx, fy, fz;
};

inline bool sparse_locate(float3 p, constant Dims& d, device const int* table, thread Brick& b) {
    if (!isfinite(p.x) || !isfinite(p.y) || !isfinite(p.z)) return false;
    int cix = int(floor(p.x / 32.0f));
    int ciy = int(floor(p.y / 32.0f));
    int ciz = int(floor(p.z / 32.0f));
    if (cix < 0 || cix >= d.a2 || ciy < 0 || ciy >= d.a1 || ciz < 0 || ciz >= d.a0) return false;
    int slot = table[(long(ciz) * d.a1 + ciy) * d.a2 + cix];
    if (slot == 0) return false;
    b.base = ulong(slot - 1) * ulong(d.a3) * 39304ul;
    float lx = p.x - float(cix) * 32.0f + 1.0f;
    float ly = p.y - float(ciy) * 32.0f + 1.0f;
    float lz = p.z - float(ciz) * 32.0f + 1.0f;
    float lx0 = floor(lx), ly0 = floor(ly), lz0 = floor(lz);
    b.fx = lx - lx0; b.fy = ly - ly0; b.fz = lz - lz0;
    b.ix0 = clamp(int(lx0), 0, 33); b.iy0 = clamp(int(ly0), 0, 33); b.iz0 = clamp(int(lz0), 0, 33);
    b.ix1 = clamp(b.ix0 + 1, 0, 33); b.iy1 = clamp(b.iy0 + 1, 0, 33); b.iz1 = clamp(b.iz0 + 1, 0, 33);
    return true;
}

// the eight corners of one channel in the kernels' order 000 100 010 110 001 101 011 111 (x, y, z)
inline void sparse_corners(device const uchar* ch, thread const Brick& b, thread float* v) {
    const int SZ = 34 * 34, SY = 34;
    v[0] = float(ch[b.iz0 * SZ + b.iy0 * SY + b.ix0]);
    v[1] = float(ch[b.iz0 * SZ + b.iy0 * SY + b.ix1]);
    v[2] = float(ch[b.iz0 * SZ + b.iy1 * SY + b.ix0]);
    v[3] = float(ch[b.iz0 * SZ + b.iy1 * SY + b.ix1]);
    v[4] = float(ch[b.iz1 * SZ + b.iy0 * SY + b.ix0]);
    v[5] = float(ch[b.iz1 * SZ + b.iy0 * SY + b.ix1]);
    v[6] = float(ch[b.iz1 * SZ + b.iy1 * SY + b.ix0]);
    v[7] = float(ch[b.iz1 * SZ + b.iy1 * SY + b.ix1]);
}

inline float interp(thread const float* v, float fx, float fy, float fz) {
    float w000 = (1.0f - fx) * (1.0f - fy) * (1.0f - fz);
    float w100 = fx          * (1.0f - fy) * (1.0f - fz);
    float w010 = (1.0f - fx) * fy          * (1.0f - fz);
    float w110 = fx          * fy          * (1.0f - fz);
    float w001 = (1.0f - fx) * (1.0f - fy) * fz;
    float w101 = fx          * (1.0f - fy) * fz;
    float w011 = (1.0f - fx) * fy          * fz;
    float w111 = fx          * fy          * fz;
    float val = 0.0f;
    val += w000 * v[0]; val += w100 * v[1]; val += w010 * v[2]; val += w110 * v[3];
    val += w001 * v[4]; val += w101 * v[5]; val += w011 * v[6]; val += w111 * v[7];
    return val;
}

inline float3 interp_grad(thread const float* v, float fx, float fy, float fz) {
    float dfx = (1.0f - fy) * (1.0f - fz) * (v[1] - v[0])
              + fy          * (1.0f - fz) * (v[3] - v[2])
              + (1.0f - fy) * fz          * (v[5] - v[4])
              + fy          * fz          * (v[7] - v[6]);
    float dfy = (1.0f - fx) * (1.0f - fz) * (v[2] - v[0])
              + fx          * (1.0f - fz) * (v[3] - v[1])
              + (1.0f - fx) * fz          * (v[6] - v[4])
              + fx          * fz          * (v[7] - v[5]);
    float dfz = (1.0f - fx) * (1.0f - fy) * (v[4] - v[0])
              + fx          * (1.0f - fy) * (v[5] - v[1])
              + (1.0f - fx) * fy          * (v[6] - v[2])
              + fx          * fy          * (v[7] - v[3]);
    return float3(dfx, dfy, dfz);
}

// CUDA's (uint8)clamp((int)roundf(v), 0, 255); MSL round() also rounds halves away from zero
inline uchar to_u8(float v) {
    return uchar(clamp(int(round(v)), 0, 255));
}

kernel void sparse_fwd_f32(device const int* table [[buffer(0)]],
                           device const uchar* pool [[buffer(1)]],
                           device const float* grid [[buffer(2)]],
                           device const float* offset [[buffer(3)]],
                           device const float* inv_scale [[buffer(4)]],
                           device float* out [[buffer(5)]],
                           constant Dims& d [[buffer(6)]],
                           uint n [[thread_position_in_grid]]) {
    if (long(n) >= d.n) return;
    Brick b;
    bool ok = sparse_locate(load_pos(grid, d, n, offset, inv_scale), d, table, b);
    for (long c = 0; c < d.a3; c++) {
        float val = 0.0f;
        if (ok) {
            float v[8];
            sparse_corners(pool + b.base + ulong(c) * 39304ul, b, v);
            val = interp(v, b.fx, b.fy, b.fz);
        }
        out[c * d.n + n] = val;
    }
}

kernel void sparse_fwd_u8(device const int* table [[buffer(0)]],
                          device const uchar* pool [[buffer(1)]],
                          device const float* grid [[buffer(2)]],
                          device const float* offset [[buffer(3)]],
                          device const float* inv_scale [[buffer(4)]],
                          device uchar* out [[buffer(5)]],
                          constant Dims& d [[buffer(6)]],
                          uint n [[thread_position_in_grid]]) {
    if (long(n) >= d.n) return;
    Brick b;
    bool ok = sparse_locate(load_pos(grid, d, n, offset, inv_scale), d, table, b);
    for (long c = 0; c < d.a3; c++) {
        uchar val = 0;
        if (ok) {
            float v[8];
            sparse_corners(pool + b.base + ulong(c) * 39304ul, b, v);
            val = to_u8(interp(v, b.fx, b.fy, b.fz));
        }
        out[c * d.n + n] = val;
    }
}

kernel void sparse_bwd(device const int* table [[buffer(0)]],
                       device const uchar* pool [[buffer(1)]],
                       device const float* grid [[buffer(2)]],
                       device const float* offset [[buffer(3)]],
                       device const float* inv_scale [[buffer(4)]],
                       device const float* grad_out [[buffer(5)]],
                       device float* grad_grid [[buffer(6)]],
                       constant Dims& d [[buffer(7)]],
                       uint n [[thread_position_in_grid]]) {
    if (long(n) >= d.n) return;
    Brick b;
    float3 g = float3(0.0f);
    if (sparse_locate(load_pos(grid, d, n, offset, inv_scale), d, table, b)) {
        float isx = inv_scale[0], isy = inv_scale[1], isz = inv_scale[2];
        for (long c = 0; c < d.a3; c++) {
            float v[8];
            sparse_corners(pool + b.base + ulong(c) * 39304ul, b, v);
            float go = grad_out[c * d.n + n];
            float3 df = interp_grad(v, b.fx, b.fy, b.fz);
            g.x += go * df.x * isx;
            g.y += go * df.y * isy;
            g.z += go * df.z * isz;
        }
    }
    grad_grid[ulong(n) * 3 + 0] = g.x;
    grad_grid[ulong(n) * 3 + 1] = g.y;
    grad_grid[ulong(n) * 3 + 2] = g.z;
}

// dense (C, Z, Y, X) volume: each out-of-range corner reads as zero
inline void dense_corners(device const uchar* vol, constant Dims& d, float3 p, thread float* f,
                          thread long* idx, thread bool* ok) {
    // a non-finite or huge coordinate samples as zero (fin masks every corner); its fractions are zeroed so no
    // NaN weight meets a zero corner. CUDA's dense kernels differ for NaN: (int)NaN is 0 there, so its corners
    // pass and the sample is NaN; the sparse kernels give 0 as here
    bool fin = isfinite(p.x) && isfinite(p.y) && isfinite(p.z)
            && fabs(p.x) < 1.0e9f && fabs(p.y) < 1.0e9f && fabs(p.z) < 1.0e9f;
    if (!fin) p = float3(0.0f);
    float fx0 = floor(p.x), fy0 = floor(p.y), fz0 = floor(p.z);
    f[0] = p.x - fx0; f[1] = p.y - fy0; f[2] = p.z - fz0;
    long ix0 = fin ? long(fx0) : -2, iy0 = fin ? long(fy0) : -2, iz0 = fin ? long(fz0) : -2;
    long xs[2] = {ix0, ix0 + 1}, ys[2] = {iy0, iy0 + 1}, zs[2] = {iz0, iz0 + 1};
    for (int k = 0; k < 8; k++) {
        long x = xs[k & 1], y = ys[(k >> 1) & 1], z = zs[(k >> 2) & 1];
        ok[k] = fin && x >= 0 && x < d.a3 && y >= 0 && y < d.a2 && z >= 0 && z < d.a1;
        idx[k] = ok[k] ? (z * d.a2 + y) * d.a3 + x : 0;
    }
}

kernel void dense_fwd_f32(device const uchar* vol [[buffer(0)]],
                          device const float* grid [[buffer(1)]],
                          device const float* offset [[buffer(2)]],
                          device const float* inv_scale [[buffer(3)]],
                          device float* out [[buffer(4)]],
                          constant Dims& d [[buffer(5)]],
                          uint n [[thread_position_in_grid]]) {
    if (long(n) >= d.n) return;
    float f[3]; long idx[8]; bool ok[8];
    dense_corners(vol, d, load_pos(grid, d, n, offset, inv_scale), f, idx, ok);
    long vstride = d.a1 * d.a2 * d.a3;
    for (long c = 0; c < d.a0; c++) {
        device const uchar* vc = vol + c * vstride;
        float v[8];
        for (int k = 0; k < 8; k++) v[k] = ok[k] ? float(vc[idx[k]]) : 0.0f;
        out[c * d.n + n] = interp(v, f[0], f[1], f[2]);
    }
}

kernel void dense_fwd_u8(device const uchar* vol [[buffer(0)]],
                         device const float* grid [[buffer(1)]],
                         device const float* offset [[buffer(2)]],
                         device const float* inv_scale [[buffer(3)]],
                         device uchar* out [[buffer(4)]],
                         constant Dims& d [[buffer(5)]],
                         uint n [[thread_position_in_grid]]) {
    if (long(n) >= d.n) return;
    float f[3]; long idx[8]; bool ok[8];
    dense_corners(vol, d, load_pos(grid, d, n, offset, inv_scale), f, idx, ok);
    long vstride = d.a1 * d.a2 * d.a3;
    for (long c = 0; c < d.a0; c++) {
        device const uchar* vc = vol + c * vstride;
        float v[8];
        for (int k = 0; k < 8; k++) v[k] = ok[k] ? float(vc[idx[k]]) : 0.0f;
        out[c * d.n + n] = to_u8(interp(v, f[0], f[1], f[2]));
    }
}

kernel void dense_bwd(device const uchar* vol [[buffer(0)]],
                      device const float* grid [[buffer(1)]],
                      device const float* offset [[buffer(2)]],
                      device const float* inv_scale [[buffer(3)]],
                      device const float* grad_out [[buffer(4)]],
                      device float* grad_grid [[buffer(5)]],
                      constant Dims& d [[buffer(6)]],
                      uint n [[thread_position_in_grid]]) {
    if (long(n) >= d.n) return;
    float f[3]; long idx[8]; bool ok[8];
    dense_corners(vol, d, load_pos(grid, d, n, offset, inv_scale), f, idx, ok);
    float isx = inv_scale[0], isy = inv_scale[1], isz = inv_scale[2];
    long vstride = d.a1 * d.a2 * d.a3;
    float3 g = float3(0.0f);
    for (long c = 0; c < d.a0; c++) {
        device const uchar* vc = vol + c * vstride;
        float v[8];
        for (int k = 0; k < 8; k++) v[k] = ok[k] ? float(vc[idx[k]]) : 0.0f;
        float go = grad_out[c * d.n + n];
        float3 df = interp_grad(v, f[0], f[1], f[2]);
        g.x += go * df.x * isx;
        g.y += go * df.y * isy;
        g.z += go * df.z * isz;
    }
    grad_grid[ulong(n) * 3 + 0] = g.x;
    grad_grid[ulong(n) * 3 + 1] = g.y;
    grad_grid[ulong(n) * 3 + 2] = g.z;
}

// prefetch with origin and spacing passed as values: writes the missing mask (needed and not loaded)
kernel void collect_missing(device const float* xyz [[buffer(0)]],
                            constant float* geo [[buffer(1)]],    // origin xyz, spacing xyz
                            device const int* table [[buffer(2)]],
                            device atomic_int* stamp [[buffer(3)]],  // per chunk: last round that listed it
                            device atomic_int* out [[buffer(4)]],    // [count, chunk ids...]
                            constant long* dims [[buffer(5)]],       // n_points, cZ, cY, cX, capacity, round
                            uint i [[thread_position_in_grid]]) {
    if (long(i) >= dims[0]) return;
    int cZ = int(dims[1]), cY = int(dims[2]), cX = int(dims[3]);
    int cap = int(dims[4]), round_id = int(dims[5]);
    float x = xyz[ulong(i) * 3 + 0], y = xyz[ulong(i) * 3 + 1], z = xyz[ulong(i) * 3 + 2];
    float fx = isfinite(x) ? (x - geo[0]) / geo[3] / 32.0f : 0.0f;
    float fy = isfinite(y) ? (y - geo[1]) / geo[4] / 32.0f : 0.0f;
    float fz = isfinite(z) ? (z - geo[2]) / geo[5] / 32.0f : 0.0f;
    int cx = clamp(int(clamp(fx, -2.0f, float(cX) + 1.0f)), 0, cX - 1);
    int cy = clamp(int(clamp(fy, -2.0f, float(cY) + 1.0f)), 0, cY - 1);
    int cz = clamp(int(clamp(fz, -2.0f, float(cZ) + 1.0f)), 0, cZ - 1);
    for (int dz = -1; dz <= 1; ++dz) {
        int zz = cz + dz;
        if (zz < 0 || zz >= cZ) continue;
        for (int dy = -1; dy <= 1; ++dy) {
            int yy = cy + dy;
            if (yy < 0 || yy >= cY) continue;
            for (int dx = -1; dx <= 1; ++dx) {
                int xx = cx + dx;
                if (xx < 0 || xx >= cX) continue;
                int k = (zz * cY + yy) * cX + xx;
                if (table[k] != 0) continue;
                // the first thread to stamp a chunk this round lists it; the count keeps rising past the
                // capacity so the host learns the true total and can retry with a larger list
                if (atomic_exchange_explicit(&stamp[k], round_id, memory_order_relaxed) == round_id) continue;
                int j = atomic_fetch_add_explicit(&out[0], 1, memory_order_relaxed);
                if (j < cap) atomic_store_explicit(&out[1 + j], k, memory_order_relaxed);
            }
        }
    }
}

// prefetch: mark the 26-neighbourhood of each sample's chunk (index by truncation, as the CUDA kernel)
kernel void mark_needed(device const float* xyz [[buffer(0)]],
                        device const float* origin [[buffer(1)]],
                        device const float* spacing [[buffer(2)]],
                        device uchar* needed [[buffer(3)]],
                        constant long* dims [[buffer(4)]],   // n_points, cZ, cY, cX
                        uint i [[thread_position_in_grid]]) {
    if (long(i) >= dims[0]) return;
    int cZ = int(dims[1]), cY = int(dims[2]), cX = int(dims[3]);
    float x = xyz[ulong(i) * 3 + 0], y = xyz[ulong(i) * 3 + 1], z = xyz[ulong(i) * 3 + 2];
    float fx = isfinite(x) ? (x - origin[0]) / spacing[0] / 32.0f : 0.0f;
    float fy = isfinite(y) ? (y - origin[1]) / spacing[1] / 32.0f : 0.0f;
    float fz = isfinite(z) ? (z - origin[2]) / spacing[2] / 32.0f : 0.0f;
    // clamp before the int conversion so out-of-range floats stay defined
    int cx = clamp(int(clamp(fx, -2.0f, float(cX) + 1.0f)), 0, cX - 1);
    int cy = clamp(int(clamp(fy, -2.0f, float(cY) + 1.0f)), 0, cY - 1);
    int cz = clamp(int(clamp(fz, -2.0f, float(cZ) + 1.0f)), 0, cZ - 1);
    for (int dz = -1; dz <= 1; ++dz) {
        int zz = cz + dz;
        if (zz < 0 || zz >= cZ) continue;
        for (int dy = -1; dy <= 1; ++dy) {
            int yy = cy + dy;
            if (yy < 0 || yy >= cY) continue;
            for (int dx = -1; dx <= 1; ++dx) {
                int xx = cx + dx;
                if (xx < 0 || xx >= cX) continue;
                needed[(long(zz) * cY + yy) * cX + xx] = 1;
            }
        }
    }
}
"""

_lib = None
_GROUP = 256


def _library():
	global _lib
	if _lib is None:
		_lib = torch.mps.compile_shader(_SOURCE)
	return _lib


def available() -> bool:
	"""True when MPS is usable and the kernels compile."""
	if not torch.backends.mps.is_available():
		return False
	try:
		_library()
	except Exception:
		return False
	return True


def _check_grid(grid: torch.Tensor) -> None:
	if grid.device.type != "mps" or grid.dtype != torch.float32 or grid.dim() != 4 or grid.shape[3] != 3:
		raise ValueError(f"grid must be an MPS float32 (D, H, W, 3) tensor, got {grid.dtype} {tuple(grid.shape)} on {grid.device}")


def _dims(a: tuple[int, int, int, int], grid: torch.Tensor) -> tuple[list[int], int]:
	D, H, W, _ = (int(v) for v in grid.shape)
	n = D * H * W
	# Dims is bound as a constant struct of int64s; the grid keeps its own strides
	return [*a, n, W, H, *(int(s) for s in grid.stride())], n


def _f3(t: torch.Tensor, device: torch.device) -> torch.Tensor:
	return t.to(device=device, dtype=torch.float32).reshape(3).contiguous()


def _launch(name: str, n: int, *args) -> None:
	if n == 0:
		return
	getattr(_library(), name)(*args, threads=n, group_size=min(_GROUP, n))


def _sparse_fwd(table, pool, grid, offset, inv_scale, *, u8: bool) -> torch.Tensor:
	_check_grid(grid)
	if table.dtype != torch.int32 or not table.is_contiguous():
		raise ValueError("table must be a contiguous int32 slot table")
	C = int(pool.shape[1])
	dims, n = _dims((*(int(v) for v in table.shape), C), grid)
	D, H, W, _ = grid.shape
	out = torch.empty(C, D, H, W, dtype=torch.uint8 if u8 else torch.float32, device=grid.device)
	_launch("sparse_fwd_u8" if u8 else "sparse_fwd_f32", n,
			table, pool.contiguous(), grid, _f3(offset, grid.device), _f3(inv_scale, grid.device), out, dims)
	return out


class _SparseSampleDiff(torch.autograd.Function):
	@staticmethod
	def forward(ctx, table, pool, grid, offset, inv_scale):
		offset = _f3(offset, grid.device)
		inv_scale = _f3(inv_scale, grid.device)
		out = _sparse_fwd(table, pool, grid, offset, inv_scale, u8=False)
		# the cache writes its table and pool in place when a sync loads chunks, which may happen between this
		# forward and its backward (anticipatory pulls); saved tensors would fail autograd's version check there.
		# Kept as attributes, as the CUDA wrapper keeps its chunk table: slots are only appended, so every slot
		# this forward read still holds the same chunk, and a grown pool leaves this reference intact
		ctx.table, ctx.pool = table, pool
		ctx.save_for_backward(grid, offset, inv_scale)
		return out

	@staticmethod
	def backward(ctx, grad_output):
		grid, offset, inv_scale = ctx.saved_tensors
		table, pool = ctx.table, ctx.pool
		C = int(pool.shape[1])
		dims, n = _dims((*(int(v) for v in table.shape), C), grid)
		grad_grid = torch.empty(*grid.shape, dtype=torch.float32, device=grid.device)
		_launch("sparse_bwd", n, table, pool, grid, offset, inv_scale,
				grad_output.contiguous(), grad_grid, dims)
		return None, None, grad_grid, None, None


def sparse_sample(table, pool, grid, offset, inv_scale, *, diff: bool) -> torch.Tensor:
	"""Metal version of ``u8_sample_torch.sparse_sample`` (same arguments and result)."""
	if diff:
		return _SparseSampleDiff.apply(table, pool.contiguous(), grid, offset, inv_scale)
	return _sparse_fwd(table, pool, grid.detach(), offset, inv_scale, u8=True)


def _dense_fwd(volume, grid, offset, inv_scale, *, u8: bool) -> torch.Tensor:
	_check_grid(grid)
	if volume.dtype != torch.uint8 or volume.dim() != 4:
		raise ValueError("volume must be a (C, Z, Y, X) uint8 tensor")
	C = int(volume.shape[0])
	dims, n = _dims(tuple(int(v) for v in volume.shape), grid)
	D, H, W, _ = grid.shape
	out = torch.empty(C, D, H, W, dtype=torch.uint8 if u8 else torch.float32, device=grid.device)
	_launch("dense_fwd_u8" if u8 else "dense_fwd_f32", n,
			volume.contiguous(), grid, _f3(offset, grid.device), _f3(inv_scale, grid.device), out, dims)
	return out


class _DenseSampleDiff(torch.autograd.Function):
	@staticmethod
	def forward(ctx, volume, grid, offset, inv_scale):
		volume = volume.contiguous()
		offset = _f3(offset, grid.device)
		inv_scale = _f3(inv_scale, grid.device)
		out = _dense_fwd(volume, grid, offset, inv_scale, u8=False)
		ctx.save_for_backward(volume, grid, offset, inv_scale)
		return out

	@staticmethod
	def backward(ctx, grad_output):
		volume, grid, offset, inv_scale = ctx.saved_tensors
		dims, n = _dims(tuple(int(v) for v in volume.shape), grid)
		grad_grid = torch.empty(*grid.shape, dtype=torch.float32, device=grid.device)
		_launch("dense_bwd", n, volume, grid, offset, inv_scale, grad_output.contiguous(), grad_grid, dims)
		return None, grad_grid, None, None


def dense_sample(volume, grid, offset, inv_scale, *, diff: bool) -> torch.Tensor:
	"""Metal version of ``u8_sample_torch.dense_sample`` (same arguments and result)."""
	if diff:
		return _DenseSampleDiff.apply(volume, grid, offset, inv_scale)
	return _dense_fwd(volume, grid.detach(), offset, inv_scale, u8=True)


def missing_chunks(xyz_fullres, table, origin, spacing) -> torch.Tensor:
	"""Metal version of ``u8_sample_torch.missing_chunks``: (M, 3) int64 (cz, cy, cx) on the device."""
	cZ, cY, cX = (int(v) for v in table.shape)
	xyz = xyz_fullres.detach().reshape(-1, 3).to(torch.float32).contiguous()
	n = int(xyz.shape[0])
	needed = torch.zeros(cZ, cY, cX, dtype=torch.uint8, device=table.device)
	_launch("mark_needed", n, xyz, _f3(origin, xyz.device), _f3(spacing, xyz.device), needed, [n, cZ, cY, cX])
	return torch.nonzero(needed.bool() & (table == 0))


class MissingChunkCollector:
	"""Lists the unloaded chunks that several prefetch calls need, with one small readback per round.

	``add`` launches a kernel that appends each unloaded chunk in the 26-neighbourhood of every sample's chunk
	(the rule of ``missing_chunks``) to a device list, once per round thanks to a per-chunk round stamp, so no
	full-grid mask is zeroed, transferred or scanned. ``take`` reads the list back and starts a new round. On
	MPS each readback waits for the queue, so a round of several ``add`` calls costs one wait instead of one each.
	"""

	def __init__(self, table_shape: tuple[int, int, int], device: torch.device, capacity: int = 4096) -> None:
		self.shape = tuple(int(v) for v in table_shape)
		cZ, cY, cX = self.shape
		if cZ * cY * cX >= 2 ** 31:
			raise ValueError(f"chunk grid {self.shape} is too large for int32 chunk ids")
		self._stamp = torch.zeros(cZ * cY * cX, dtype=torch.int32, device=device)
		self._capacity = int(capacity)
		self._out = torch.zeros(1 + self._capacity, dtype=torch.int32, device=device)
		self._round = 0
		# inputs of the open round, kept so an overflowing round can be replayed into a larger list
		self._adds: list[tuple[torch.Tensor, torch.Tensor, list[float]]] = []

	def add(self, xyz_fullres: torch.Tensor, table: torch.Tensor, origin, spacing) -> None:
		"""Queue the chunks these (..., 3) fullres sample positions need; origin and spacing are xyz floats."""
		if not self._adds:
			self._open_round()
		xyz = xyz_fullres.detach().reshape(-1, 3).to(torch.float32).contiguous()
		geo = [float(v) for v in origin] + [float(v) for v in spacing]
		self._adds.append((xyz, table, geo))
		self._launch(xyz, table, geo)

	def take(self) -> np.ndarray:
		"""(M, 3) int64 (cz, cy, cx) of the round's missing chunks in row-major order, then close the round."""
		if not self._adds:
			return np.zeros((0, 3), np.int64)
		host = self._out.cpu().numpy()
		count = int(host[0])
		if count > self._capacity:
			# rare (mostly the first round of a fit): replay the round into a list that fits
			self._capacity = 1 << (count - 1).bit_length()
			self._out = torch.zeros(1 + self._capacity, dtype=torch.int32, device=self._out.device)
			self._open_round()
			for xyz, table, geo in self._adds:
				self._launch(xyz, table, geo)
			host = self._out.cpu().numpy()
			count = int(host[0])
		self._adds = []
		lin = np.sort(host[1:1 + count].astype(np.int64))
		return np.stack(np.unravel_index(lin, self.shape), axis=1)

	def _open_round(self) -> None:
		# a fresh round id makes every old stamp stale, so the stamps never need clearing
		self._round += 1
		self._out[:1].zero_()

	def _launch(self, xyz: torch.Tensor, table: torch.Tensor, geo: list[float]) -> None:
		n = int(xyz.shape[0])
		if n == 0:
			return
		cZ, cY, cX = self.shape
		_launch("collect_missing", n, xyz, geo, table, self._stamp, self._out,
				[n, cZ, cY, cX, self._capacity, self._round])
