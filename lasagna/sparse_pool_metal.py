"""Metal kernels for the pooled sparse chunk sampler on Apple GPUs (MPS).

The Metal counterpart of sparse_pool_ops, for PooledChunkGroupCache on MPS:
the same four functions with the same signatures, layout (see sparse_pool_ops)
and semantics, which follow sparse_grid_sample_3d_u8_kernel.cu, the forward
and backward kernels of sparse_grid_sample_3d_u8_diff_kernel.cu and
sparse_prefetch_chunks_kernel.cu. sparse_pool_ops runs each call as dozens of
small torch ops; here every sampler call is one dispatch with one thread per
sample, and missing_chunks is a few small dispatches and one host sync; it
returns its rows on the CPU, where its caller (the cache's prefetch) needs them.

Numerics. The kernels evaluate the .cu kernels' float32 operations in the
same order, as sparse_pool_ops does. Floating-point contraction is switched
off (#pragma METAL fp contract(off), and every product that is added sits in a
statement of its own), so each multiply and add rounds on its own as in the
torch ops, and the results equal sparse_pool_ops' bitwise on MPS. If a Metal
compiler rejects the pragma, the library is built without it (CONTRACTION
records which); results then differ from the torch ops only by FMA-level
rounding, as the CUDA kernels' do (nvcc contracts mul+add), within the
tolerances stated in tests/test_sparse_pool_metal.py. Metal
flushes float32 subnormals to zero, as MPS torch ops do.

Lifetime. Nothing is cached across calls but the compiled library: pool and
table are bound afresh on every call (the pool may be reallocated between
calls, and the table gains entries in place). The kernels read and never
write pool, table and grid. Output elements are each written by one thread.

Calls the kernels do not take run on sparse_pool_ops instead, with one
RuntimeWarning per process naming the reason: tensors on a device other than
MPS, more than _MAX_SAMPLES samples or points in one call (one dispatch, 32-bit
thread index), and chunk tables with a dimension of 2**24 or more (bounds are
compared in float32, as in sparse_pool_ops) or 2**31 cells or more. With
LASAGNA_METAL=1, an explicit request for the Metal kernels, such a call raises
RuntimeError instead.

compile() builds the library once per process with torch.mps.compile_shader
(PyTorch's safe math mode, so isfinite survives) and runs a small known-answer
check against sparse_pool_ops on MPS; it raises if either fails, and
sparse_pool_cache then samples with the torch ops.
"""
from __future__ import annotations

import os
import warnings

import torch

import sparse_pool_ops as _ops

CHUNK = _ops.CHUNK
PADDED = _ops.PADDED
BRICK = _ops.BRICK

_GROUP = 256
_MAX_SAMPLES = (1 << 31) - 1  # threads per dispatch: uint thread index, one dispatch per call
_MAX_TABLE_DIM = 1 << 24      # chunk-grid sizes are compared with float32 chunk indices
_MAX_TABLE_CELLS = 1 << 31    # missing_chunks numbers cells with 32-bit indices
_SCAN_THREADS = 1024          # missing_chunks: blocks of the bitmask, one thread group scans them
_STAGED_ROWS = 1 << 16        # missing_chunks: rows that come to the host with the count, in one copy

_LIB = None
CONTRACTION: str | None = None  # after compile(): "off" (the pragma compiled) or "compiler default"
_WARNED: set[str] = set()

_PRAGMA_MARK = "// FP_CONTRACT_PRAGMA"
_PRAGMA = "#pragma METAL fp contract(off)"

_SOURCE = r'''
#include <metal_stdlib>
using namespace metal;
// FP_CONTRACT_PRAGMA

#define BRICK 39304L     // 34^3 bytes per channel per slot
#define STRIDE_Z 1156L   // 34 * 34
#define STRIDE_Y 34L
#define MAX_LOCAL 33

// Sampler parameters (constant long* ip):
// [0] C, [1] cZ, [2] cY, [3] cX, [4] N, [5] grid stride per sample, [6] grid stride per coordinate,
// [7] table stride z, [8] table stride y, [9] table stride x, [10] S (pool slots)

struct Cell {
    long base;         // pool byte offset of corner 000 of channel 0
    long sx, sy, sz;   // byte steps to the corner one up along x, y, z (0 where clamped)
    float fx, fy, fz;  // fractions in the cell
};

// The .cu kernels' addressing for sample n. Returns false when the sample is zero: p not finite,
// outside the chunk grid, or in a chunk that is not loaded; nothing is read from the pool then.
// CLAMP_FIRST: the diff kernels clamp i0 to [0, 33] and then i1 = clamp(i0 + 1); the uint8 kernel
// takes i1 = i0 + 1 and clamps both. They agree because l >= 1; each kernel keeps its own order.
template <bool CLAMP_FIRST>
inline bool locate(device const int* table, device const float* grid,
                   device const float* offset, device const float* inv_scale,
                   constant long* ip, uint n, thread Cell& c) {
    const long g = long(n) * ip[5];
    const long gc = ip[6];
    const float px = (grid[g] - offset[0]) * inv_scale[0];
    const float py = (grid[g + gc] - offset[1]) * inv_scale[1];
    const float pz = (grid[g + 2 * gc] - offset[2]) * inv_scale[2];
    if (!isfinite(px) || !isfinite(py) || !isfinite(pz)) {
        return false;
    }
    // (int)floorf(p / 32.0f), bounds-checked while still float so that huge values are never
    // converted to integers.
    const float cxf = floor(px / 32.0f);
    const float cyf = floor(py / 32.0f);
    const float czf = floor(pz / 32.0f);
    if (!(cxf >= 0.0f && cxf < float(ip[3]) && cyf >= 0.0f && cyf < float(ip[2]) &&
          czf >= 0.0f && czf < float(ip[1]))) {
        return false;
    }
    const long slot = long(table[long(czf) * ip[7] + long(cyf) * ip[8] + long(cxf) * ip[9]]);
    // -1: not loaded. A slot past the pool never comes from the cache; it reads as not loaded
    // rather than out of bounds.
    if (slot < 0 || slot >= ip[10]) {
        return false;
    }
    // Local coordinates in the padded chunk: p - ci * 32.0f + 1.0f (ci * 32 is exact).
    const float lx = px - cxf * 32.0f + 1.0f;
    const float ly = py - cyf * 32.0f + 1.0f;
    const float lz = pz - czf * 32.0f + 1.0f;
    const float lx0 = floor(lx);
    const float ly0 = floor(ly);
    const float lz0 = floor(lz);
    c.fx = lx - lx0;
    c.fy = ly - ly0;
    c.fz = lz - lz0;
    int ix0, iy0, iz0, ix1, iy1, iz1;
    if (CLAMP_FIRST) {
        ix0 = clamp(int(lx0), 0, MAX_LOCAL);
        iy0 = clamp(int(ly0), 0, MAX_LOCAL);
        iz0 = clamp(int(lz0), 0, MAX_LOCAL);
        ix1 = clamp(ix0 + 1, 0, MAX_LOCAL);
        iy1 = clamp(iy0 + 1, 0, MAX_LOCAL);
        iz1 = clamp(iz0 + 1, 0, MAX_LOCAL);
    } else {
        ix0 = int(lx0);
        iy0 = int(ly0);
        iz0 = int(lz0);
        ix1 = clamp(ix0 + 1, 0, MAX_LOCAL);
        iy1 = clamp(iy0 + 1, 0, MAX_LOCAL);
        iz1 = clamp(iz0 + 1, 0, MAX_LOCAL);
        ix0 = clamp(ix0, 0, MAX_LOCAL);
        iy0 = clamp(iy0, 0, MAX_LOCAL);
        iz0 = clamp(iz0, 0, MAX_LOCAL);
    }
    // 64-bit offsets: a pool over 4 GiB is about 109k slots at C = 1.
    c.base = slot * ip[0] * BRICK + long(iz0) * STRIDE_Z + long(iy0) * STRIDE_Y + long(ix0);
    c.sx = long(ix1 - ix0);
    c.sy = long(iy1 - iy0) * STRIDE_Y;
    c.sz = long(iz1 - iz0) * STRIDE_Z;
    return true;
}

struct Corners {
    float v000, v100, v010, v110, v001, v101, v011, v111;
};

inline Corners corners(device const uchar* pool, thread const Cell& c, long channel) {
    device const uchar* p = pool + c.base + channel * BRICK;
    Corners v;
    v.v000 = float(p[0]);
    v.v100 = float(p[c.sx]);
    v.v010 = float(p[c.sy]);
    v.v110 = float(p[c.sy + c.sx]);
    v.v001 = float(p[c.sz]);
    v.v101 = float(p[c.sz + c.sx]);
    v.v011 = float(p[c.sz + c.sy]);
    v.v111 = float(p[c.sz + c.sy + c.sx]);
    return v;
}

// val = w000 * v000; val += w100 * v100; ... in the .cu kernels' order, every product and every
// sum rounded on its own (one statement each).
inline float interpolate(thread const Cell& c, thread const Corners& v) {
    const float fx = c.fx, fy = c.fy, fz = c.fz;
    const float ofx = 1.0f - fx, ofy = 1.0f - fy, ofz = 1.0f - fz;
    const float w000 = ofx * ofy * ofz;
    const float w100 = fx * ofy * ofz;
    const float w010 = ofx * fy * ofz;
    const float w110 = fx * fy * ofz;
    const float w001 = ofx * ofy * fz;
    const float w101 = fx * ofy * fz;
    const float w011 = ofx * fy * fz;
    const float w111 = fx * fy * fz;
    float val = w000 * v.v000;
    float t;
    t = w100 * v.v100; val = val + t;
    t = w010 * v.v010; val = val + t;
    t = w110 * v.v110; val = val + t;
    t = w001 * v.v001; val = val + t;
    t = w101 * v.v101; val = val + t;
    t = w011 * v.v011; val = val + t;
    t = w111 * v.v111; val = val + t;
    return val;
}

kernel void sparse_sample_u8(device const uchar* pool [[buffer(0)]],
                             device const int* table [[buffer(1)]],
                             device const float* grid [[buffer(2)]],
                             device const float* offset [[buffer(3)]],
                             device const float* inv_scale [[buffer(4)]],
                             device uchar* out [[buffer(5)]],
                             constant long* ip [[buffer(6)]],
                             uint n [[thread_position_in_grid]]) {
    const long N = ip[4];
    if (long(n) >= N) {
        return;
    }
    const long C = ip[0];
    Cell c;
    if (!locate<false>(table, grid, offset, inv_scale, ip, n, c)) {
        for (long ch = 0; ch < C; ++ch) {
            out[ch * N + n] = 0;
        }
        return;
    }
    for (long ch = 0; ch < C; ++ch) {
        const float val = interpolate(c, corners(pool, c, ch));
        // roundf: half away from zero.
        out[ch * N + n] = uchar(clamp(int(round(val)), 0, 255));
    }
}

kernel void sparse_sample_f32(device const uchar* pool [[buffer(0)]],
                              device const int* table [[buffer(1)]],
                              device const float* grid [[buffer(2)]],
                              device const float* offset [[buffer(3)]],
                              device const float* inv_scale [[buffer(4)]],
                              device float* out [[buffer(5)]],
                              constant long* ip [[buffer(6)]],
                              uint n [[thread_position_in_grid]]) {
    const long N = ip[4];
    if (long(n) >= N) {
        return;
    }
    const long C = ip[0];
    Cell c;
    if (!locate<true>(table, grid, offset, inv_scale, ip, n, c)) {
        for (long ch = 0; ch < C; ++ch) {
            out[ch * N + n] = 0.0f;
        }
        return;
    }
    for (long ch = 0; ch < C; ++ch) {
        out[ch * N + n] = interpolate(c, corners(pool, c, ch));
    }
}

// d(sum_c grad_output[c] * out[c]) / d grid: the diff kernel's backward. No atomics: every sample
// writes its own gradient.
kernel void sparse_sample_grad(device const uchar* pool [[buffer(0)]],
                               device const int* table [[buffer(1)]],
                               device const float* grid [[buffer(2)]],
                               device const float* offset [[buffer(3)]],
                               device const float* inv_scale [[buffer(4)]],
                               device const float* grad_output [[buffer(5)]],
                               device float* grad_grid [[buffer(6)]],
                               constant long* ip [[buffer(7)]],
                               uint n [[thread_position_in_grid]]) {
    const long N = ip[4];
    if (long(n) >= N) {
        return;
    }
    device float* g = grad_grid + long(n) * 3;
    Cell c;
    if (!locate<true>(table, grid, offset, inv_scale, ip, n, c)) {
        g[0] = 0.0f;
        g[1] = 0.0f;
        g[2] = 0.0f;
        return;
    }
    const long C = ip[0];
    const float isx = inv_scale[0], isy = inv_scale[1], isz = inv_scale[2];
    const float fx = c.fx, fy = c.fy, fz = c.fz;
    const float ofx = 1.0f - fx, ofy = 1.0f - fy, ofz = 1.0f - fz;
    float dgx = 0.0f, dgy = 0.0f, dgz = 0.0f;
    for (long ch = 0; ch < C; ++ch) {
        const float go = grad_output[ch * N + n];
        const Corners v = corners(pool, c, ch);
        float t;
        // d(interp)/d(fx)
        float dfx = ofy * ofz * (v.v100 - v.v000);
        t = fy * ofz * (v.v110 - v.v010); dfx = dfx + t;
        t = ofy * fz * (v.v101 - v.v001); dfx = dfx + t;
        t = fy * fz * (v.v111 - v.v011); dfx = dfx + t;
        // d(interp)/d(fy)
        float dfy = ofx * ofz * (v.v010 - v.v000);
        t = fx * ofz * (v.v110 - v.v100); dfy = dfy + t;
        t = ofx * fz * (v.v011 - v.v001); dfy = dfy + t;
        t = fx * fz * (v.v111 - v.v101); dfy = dfy + t;
        // d(interp)/d(fz)
        float dfz = ofx * ofy * (v.v001 - v.v000);
        t = fx * ofy * (v.v101 - v.v100); dfz = dfz + t;
        t = ofx * fy * (v.v011 - v.v010); dfz = dfz + t;
        t = fx * fy * (v.v111 - v.v110); dfz = dfz + t;
        // Chain rule: d(local)/d(fullres) = inv_scale.
        t = go * dfx * isx; dgx = dgx + t;
        t = go * dfy * isy; dgy = dgy + t;
        t = go * dfz * isz; dgz = dgz + t;
    }
    g[0] = dgx;
    g[1] = dgy;
    g[2] = dgz;
}

// missing_chunks. The chunk coordinate of sparse_prefetch_chunks_kernel.cu:
// (int)((v - origin) / spacing / 32.0f) for finite v, else 0; then clamped to [0, n - 1]. The .cu
// conversion truncates toward zero, saturates out-of-range values and sends NaN (0 / 0) to 0; the
// comparisons below give the same result without converting values outside [1, n).
inline int chunk_coord(float v, float origin, float spacing, int n) {
    if (!isfinite(v)) {
        return 0;
    }
    const float q = precise::divide(precise::divide(v - origin, spacing), 32.0f);
    if (!(q >= 1.0f)) {
        return 0;  // NaN, negative, or below 1 (truncates to 0)
    }
    if (q >= float(n)) {
        return n - 1;
    }
    return min(int(q), n - 1);
}

// Marks, in a bitmask over the chunk grid (bit = row-major cell index), every chunk in the
// 26-neighbourhood of each point's chunk that lies inside the grid and is not loaded. After the mask
// it records the range of marked cells, so that the scan and the write cover only the words that hold
// them: bits[words] = max over threads of ~(smallest marked cell), bits[words + 1] = max of (largest
// marked cell + 1); both stay 0 when nothing is marked.
// ip: [0] points, [1] cZ, [2] cY, [3] cX, [4] point stride, [5] coordinate stride,
//     [6] table stride z, [7] table stride y, [8] table stride x, [9] words
kernel void sparse_mark_missing(device const float* xyz [[buffer(0)]],
                                device const float* origin [[buffer(1)]],
                                device const float* spacing [[buffer(2)]],
                                device const int* table [[buffer(3)]],
                                device atomic_uint* bits [[buffer(4)]],
                                constant long* ip [[buffer(5)]],
                                uint i [[thread_position_in_grid]]) {
    if (long(i) >= ip[0]) {
        return;
    }
    const int cZ = int(ip[1]), cY = int(ip[2]), cX = int(ip[3]);
    const long p = long(i) * ip[4];
    const long pc = ip[5];
    const int cx = chunk_coord(xyz[p], origin[0], spacing[0], cX);
    const int cy = chunk_coord(xyz[p + pc], origin[1], spacing[1], cY);
    const int cz = chunk_coord(xyz[p + 2 * pc], origin[2], spacing[2], cZ);
    uint first = 0xFFFFFFFFu;  // smallest cell this thread marks
    uint end = 0u;             // largest cell this thread marks, plus one
    for (int z = max(cz - 1, 0); z <= min(cz + 1, cZ - 1); ++z) {
        for (int y = max(cy - 1, 0); y <= min(cy + 1, cY - 1); ++y) {
            for (int x = max(cx - 1, 0); x <= min(cx + 1, cX - 1); ++x) {
                if (table[long(z) * ip[6] + long(y) * ip[7] + long(x) * ip[8]] >= 0) {
                    continue;  // loaded
                }
                const uint cell = uint((z * cY + y) * cX + x);
                first = min(first, cell);
                end = max(end, cell + 1u);
                const uint bit = 1u << (cell & 31u);
                device atomic_uint* word = bits + (cell >> 5);
                if ((atomic_load_explicit(word, memory_order_relaxed) & bit) == 0u) {
                    atomic_fetch_or_explicit(word, bit, memory_order_relaxed);
                }
            }
        }
    }
    // One atomic pair per SIMD group for the range.
    const uint group_first = simd_min(first);
    const uint group_end = simd_max(end);
    if (simd_is_first() && group_end != 0u) {
        atomic_fetch_max_explicit(bits + ip[9], ~group_first, memory_order_relaxed);
        atomic_fetch_max_explicit(bits + ip[9] + 1, group_end, memory_order_relaxed);
    }
}

// Block t of T of the words that hold marked cells, [w0, w1) from the range sparse_mark_missing
// records: [w0 + t * K, w0 + (t + 1) * K) clipped to w1, with K = ceil((w1 - w0) / T). Empty when
// nothing is marked. The scan and the write split the words identically.
inline void block_words(device const uint* bits, long words, long t, long T,
                        thread long& lo, thread long& hi) {
    const uint end = bits[words + 1];
    long w0 = 0, w1 = 0;
    if (end != 0u) {
        w0 = long((~bits[words]) >> 5);
        w1 = long((end - 1u) >> 5) + 1;
    }
    const long K = (w1 - w0 + T - 1) / T;
    lo = min(w0 + t * K, w1);
    hi = min(lo + K, w1);
}

// One thread group of T threads: thread t counts the set bits of its block of words, and the group
// scans the counts. starts[t] = the first output row of block t; starts[T] = M, the number of rows.
// Every thread reaches every barrier (no early return).
// ip: [0] words
kernel void sparse_scan_missing(device const uint* bits [[buffer(0)]],
                                device long* starts [[buffer(1)]],
                                constant long* ip [[buffer(2)]],
                                uint t [[thread_position_in_threadgroup]],
                                uint T [[threads_per_threadgroup]]) {
    threadgroup long buf[1024];
    long lo, hi;
    block_words(bits, ip[0], long(t), long(T), lo, hi);
    long count = 0;
    for (long w = lo; w < hi; ++w) {
        count += long(popcount(bits[w]));
    }
    buf[t] = count;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint off = 1; off < T; off <<= 1) {  // inclusive scan (Hillis-Steele)
        const long v = (t >= off) ? buf[t - off] : 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf[t] += v;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    starts[t] = buf[t] - count;
    if (t == T - 1) {
        starts[T] = buf[t];
    }
}

// Writes staged = [M, cz0, cy0, cx0, cz1, ...] (int32): M, then (cz, cy, cx) of the first `cap`
// set bits in row-major order, block t from row starts[t]. Rows past `cap` are not written, so every
// store stays inside the buffer.
// ip: [0] words, [1] T (blocks), [2] cY, [3] cX, [4] cap (rows)
kernel void sparse_write_missing(device const uint* bits [[buffer(0)]],
                                 device const long* starts [[buffer(1)]],
                                 device int* staged [[buffer(2)]],
                                 constant long* ip [[buffer(3)]],
                                 uint t [[thread_position_in_grid]]) {
    if (long(t) >= ip[1]) {
        return;
    }
    if (t == 0) {
        staged[0] = int(starts[ip[1]]);
    }
    const long cap = ip[4];
    long pos = starts[t];
    long lo, hi;
    block_words(bits, ip[0], long(t), ip[1], lo, hi);
    const uint cX = uint(ip[3]);
    const uint cYX = uint(ip[2]) * cX;
    for (long w = lo; w < hi && pos < cap; ++w) {
        uint b = bits[w];
        while (b != 0u && pos < cap) {
            const uint cell = uint(w) * 32u + ctz(b);
            b &= b - 1u;
            const uint cz = cell / cYX;
            const uint rem = cell - cz * cYX;
            const uint cy = rem / cX;
            device int* row = staged + 1 + pos * 3;
            row[0] = int(cz);
            row[1] = int(cy);
            row[2] = int(rem - cy * cX);
            ++pos;
        }
    }
}
'''


def compile() -> None:
    """Build the shader library (once per process) and check it against sparse_pool_ops.

    Raises if the library does not compile or its results differ from the torch ops on a small
    fixed case; sparse_pool_cache then samples with the torch ops.
    """
    global _LIB, CONTRACTION
    if _LIB is not None:
        return
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available")
    try:
        lib = torch.mps.compile_shader(_SOURCE.replace(_PRAGMA_MARK, _PRAGMA))
        contraction = "off"
    except Exception:  # a Metal compiler without the fp contract pragma
        lib = torch.mps.compile_shader(_SOURCE)
        contraction = "compiler default"
    _LIB = lib
    try:
        _self_check()
    except Exception:
        _LIB = None
        raise
    CONTRACTION = contraction


def _lib():
    if _LIB is None:
        compile()
    return _LIB


def _metal_required() -> bool:
    """LASAGNA_METAL=1 asks for the Metal kernels explicitly: no fallback to the torch ops."""
    return os.environ.get("LASAGNA_METAL") == "1"


def _unsupported(function: str, key: str, reason: str) -> None:
    """A call the kernels do not take. Raises under LASAGNA_METAL=1; otherwise warns once per
    process per kind of reason, and the caller runs the call on sparse_pool_ops."""
    if _metal_required():
        raise RuntimeError(
            f"LASAGNA_METAL=1 asks for the Metal kernels, but sparse_pool_metal.{function} "
            f"does not take this call: {reason}")
    if key not in _WARNED:
        _WARNED.add(key)
        warnings.warn(f"sparse_pool_metal.{function}: {reason}; such calls use the torch ops "
                      "(sparse_pool_ops) instead", RuntimeWarning, stacklevel=3)


def _table_problem(chunk_table) -> str | None:
    if max(chunk_table.shape) >= _MAX_TABLE_DIM or chunk_table.numel() >= _MAX_TABLE_CELLS:
        return (f"chunk table {tuple(chunk_table.shape)} has a dimension of {_MAX_TABLE_DIM} or more, "
                f"or {_MAX_TABLE_CELLS} cells or more")
    return None


def _takes(function: str, device: torch.device, n: int, chunk_table) -> bool:
    """Whether the kernels take this call (warning or raising, as _unsupported, when not)."""
    if device.type != "mps":
        _unsupported(function, "device", f"tensors on {device}, not on MPS")
        return False
    if n > _MAX_SAMPLES:
        _unsupported(function, "samples", f"{n} samples or points, above the {_MAX_SAMPLES} one dispatch takes")
        return False
    problem = _table_problem(chunk_table)
    if problem is not None:
        _unsupported(function, "table", problem)
        return False
    return True


def _vec3(t):
    return t.reshape(3).contiguous()


def _sampler_params(pool, chunk_table, points):
    S, C = (int(v) for v in pool.shape[:2])
    cZ, cY, cX = (int(v) for v in chunk_table.shape)
    return [C, cZ, cY, cX, int(points.shape[0]), *(int(s) for s in points.stride()),
            *(int(s) for s in chunk_table.stride()), S]


def _sample(pool, chunk_table, grid, offset, inv_scale, *, as_uint8):
    _ops._check(pool, chunk_table, grid, offset, inv_scale)
    C = int(pool.shape[1])
    D, H, W = (int(v) for v in grid.shape[:3])
    N = D * H * W
    if not _takes("sample_u8" if as_uint8 else "sample_f32", pool.device, N, chunk_table):
        return _ops.sample_u8(pool, chunk_table, grid, offset, inv_scale) if as_uint8 \
            else _ops.sample_f32(pool, chunk_table, grid, offset, inv_scale)
    dtype = torch.uint8 if as_uint8 else torch.float32
    if N == 0 or pool.numel() == 0 or chunk_table.numel() == 0:
        return torch.zeros((C, D, H, W), dtype=dtype, device=pool.device)
    # A view whenever the grid's (D, H, W) strides merge (e.g. xyz[:, :, ::2]); a copy otherwise.
    points = grid.reshape(N, 3)
    out = torch.empty((C, N), dtype=dtype, device=pool.device)
    kernel = _lib().sparse_sample_u8 if as_uint8 else _lib().sparse_sample_f32
    kernel(pool, chunk_table, points, _vec3(offset), _vec3(inv_scale), out,
           _sampler_params(pool, chunk_table, points), threads=N, group_size=_GROUP)
    return out.reshape(C, D, H, W)


def sample_u8(pool, chunk_table, grid, offset, inv_scale):
    """sparse_pool_ops.sample_u8 on MPS: (C, D, H, W) uint8."""
    return _sample(pool, chunk_table, grid, offset, inv_scale, as_uint8=True)


def sample_f32(pool, chunk_table, grid, offset, inv_scale):
    """sparse_pool_ops.sample_f32 on MPS: (C, D, H, W) float32, unrounded."""
    return _sample(pool, chunk_table, grid, offset, inv_scale, as_uint8=False)


def sample_grad(pool, chunk_table, grid, offset, inv_scale, grad_output):
    """sparse_pool_ops.sample_grad on MPS: (D, H, W, 3) float32 gradient w.r.t. the grid."""
    _ops._check(pool, chunk_table, grid, offset, inv_scale)
    C = int(pool.shape[1])
    D, H, W = (int(v) for v in grid.shape[:3])
    N = D * H * W
    if grad_output.dtype != torch.float32 or tuple(grad_output.shape) != (C, D, H, W):
        raise ValueError(f"grad_output must be float32 {(C, D, H, W)}, got {grad_output.dtype} {tuple(grad_output.shape)}")
    if grad_output.device != pool.device:
        raise ValueError(f"grad_output is on {grad_output.device}, the pool on {pool.device}")
    if not _takes("sample_grad", pool.device, N, chunk_table):
        return _ops.sample_grad(pool, chunk_table, grid, offset, inv_scale, grad_output)
    if N == 0 or pool.numel() == 0 or chunk_table.numel() == 0:
        return torch.zeros((D, H, W, 3), dtype=torch.float32, device=pool.device)
    points = grid.reshape(N, 3)
    go = grad_output.contiguous()
    grad = torch.empty((N, 3), dtype=torch.float32, device=pool.device)
    _lib().sparse_sample_grad(pool, chunk_table, points, _vec3(offset), _vec3(inv_scale), go, grad,
                              _sampler_params(pool, chunk_table, points), threads=N, group_size=_GROUP)
    return grad.reshape(D, H, W, 3)


def missing_chunks(xyz, chunk_table, origin, spacing):
    """sparse_pool_ops.missing_chunks for MPS inputs: (M, 3) int64 (cz, cy, cx), row-major order,
    returned ON THE CPU: its caller, the cache's prefetch, needs the rows on the host,
    and handing them over there takes one GPU round trip instead of two.

    One dispatch marks the not-loaded chunks of every point's clipped 26-neighbourhood in a
    bitmask over the chunk grid, and the range of marked cells; one thread group counts the set bits
    per block of the words in that range and scans the counts; a last dispatch writes M and the rows
    into a fixed staging buffer, which comes to the host in one copy. More than _STAGED_ROWS rows
    take a second copy. torch.nonzero is not used: on MPS it sometimes returns rows that are not
    there (torch 2.11).
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
    n = xyz.numel() // 3
    if not _takes("missing_chunks", dev, n, chunk_table):
        return _ops.missing_chunks(xyz, chunk_table, origin, spacing).cpu()
    cells = cZ * cY * cX
    if n == 0 or cells == 0:
        return torch.empty((0, 3), dtype=torch.long)
    pts = xyz.reshape(n, 3)
    lib = _lib()
    words = (cells + 31) // 32
    blocks = _scan_threads(lib)
    bits = torch.zeros(words + 2, dtype=torch.int32, device=dev)  # the mask, then the marked range
    lib.sparse_mark_missing(pts, _vec3(origin), _vec3(spacing), chunk_table, bits,
                            [n, cZ, cY, cX, *(int(s) for s in pts.stride()),
                             *(int(s) for s in chunk_table.stride()), words],
                            threads=n, group_size=_GROUP)
    starts = torch.empty(blocks + 1, dtype=torch.long, device=dev)
    lib.sparse_scan_missing(bits, starts, [words], threads=blocks, group_size=blocks)
    cap = min(cells, _STAGED_ROWS)
    while True:
        staged = torch.empty(1 + 3 * cap, dtype=torch.int32, device=dev)
        lib.sparse_write_missing(bits, starts, staged, [words, blocks, cY, cX, cap],
                                 threads=blocks, group_size=min(blocks, _GROUP))
        host = staged.cpu()  # the one host sync when M <= cap
        m = int(host[0])
        if m <= cap:
            return host[1:1 + 3 * m].view(m, 3).to(torch.long)
        cap = m  # more rows than the staging buffer: write them all, once more


def _scan_threads(lib) -> int:
    """Threads of sparse_scan_missing's single thread group (its threadgroup buffer holds 1024)."""
    limit = getattr(lib.sparse_scan_missing, "max_threads_per_threadgroup", _GROUP)
    return max(1, min(_SCAN_THREADS, int(limit)))


def _self_check() -> None:
    """Run the kernels once on a small fixed case against sparse_pool_ops.

    Catches a library that compiles but computes something else, e.g. a fast-math build that drops
    the non-finite guards, or a change in how compile_shader binds arguments. The reference runs on
    the CPU (MPS torch ops equal it bitwise here, and building their MPS graphs would cost about half
    a second); the tolerances are those stated in tests/test_sparse_pool_metal.py. Uses its own
    generator; the global RNG is untouched.
    """
    dev = torch.device("mps")
    gen = torch.Generator().manual_seed(20261122)
    C = 2
    table = torch.tensor([[[0, -1, 1], [2, -1, 3]], [[-1, 4, -1], [5, -1, -1]]], dtype=torch.int32)
    pool = torch.randint(0, 256, (6, C, PADDED, PADDED, PADDED), generator=gen, dtype=torch.uint8)
    span = torch.tensor([3.0 * CHUNK, 2.0 * CHUNK, 2.0 * CHUNK])
    local = torch.rand(5, 7, 8, 3, generator=gen) * (span + 8.0) - 4.0
    nan, inf = float("nan"), float("inf")
    special = torch.tensor([
        [nan, 1.0, 1.0], [1.0, inf, 1.0], [1.0, 1.0, -inf], [1e30, 1.0, 1.0], [-1e30, -1e30, -1e30],
        [0.0, 0.0, 0.0], [32.0 - 2.0 ** -19, 31.5, 0.5], [32.0, 32.0, 32.0], [95.5, 63.5, 63.5],
        [96.0, 1.0, 1.0], [-1e-3, 5.0, 5.0], [5.0, -0.5, 5.0], [33.0, 64.0 - 2.0 ** -18, 1.0],
    ])
    local.view(-1, 3)[:len(special)] = special
    offset = torch.tensor([3.0, -2.0, 5.0])
    inv_scale = torch.tensor([0.5, 2.0, 1.0])
    origin = torch.tensor([1.0, -3.0, 2.0])
    spacing = torch.tensor([0.75, 3.0, 1.5])
    with torch.no_grad():
        grid = local / inv_scale + offset
        # A strided view with a storage offset, as slices of the fit's tensors give.
        store = torch.zeros(5, 7, 8, 4)
        store[..., 1:] = grid
        go = torch.randn(C, 5, 7, 8, generator=gen)
        host = (pool, table, store[..., 1:], offset, inv_scale)
        gpu = (pool.to(dev), table.to(dev), store.to(dev)[..., 1:], offset.to(dev), inv_scale.to(dev))
        points = grid.reshape(-1, 3)
        problems = []

        def check(name, ok, ref, got, per_sample):
            # ok: (C, N) or (N, 3) bool; per_sample: elements per sample along the flat index.
            ok = ok.reshape(-1)
            if bool(ok.all()):
                return
            i = int((~ok).nonzero()[0, 0])
            n = i % points.shape[0] if per_sample == 0 else i // per_sample
            diff = (ref.double() - got.double()).abs().max()
            problems.append(f"{name} max |diff| {float(diff):.4g}, first bad element {i} (sample {n}, "
                            f"grid {points[n].tolist()}): torch ops {ref.reshape(-1)[i].item()}, "
                            f"metal {got.reshape(-1)[i].item()}")

        ref_f, got_f = _ops.sample_f32(*host), sample_f32(*gpu).cpu()
        check("sample_f32", ((ref_f == 0) == (got_f == 0)) & ((ref_f - got_f).abs() <= 1e-4), ref_f, got_f, 0)
        ref_u, got_u = _ops.sample_u8(*host), sample_u8(*gpu).cpu()
        du = (ref_u.int() - got_u.int()).abs()
        near_half = (ref_f - ref_f.floor() - 0.5).abs() <= 1e-4
        check("sample_u8", (du == 0) | (near_half & (du == 1)), ref_u, got_u, 0)
        ref_g, got_g = _ops.sample_grad(*host, go), sample_grad(*gpu, go.to(dev)).cpu()
        atol = 1e-4 * C * float(go.abs().max()) * float(inv_scale.max())
        check("sample_grad", ((ref_g == 0) == (got_g == 0)) & ((ref_g - got_g).abs() <= atol + 1e-5 * ref_g.abs()),
              ref_g, got_g, 3)
        ref_m = _ops.missing_chunks(host[2], table, origin, spacing)
        got_m = missing_chunks(gpu[2], gpu[1], origin.to(dev), spacing.to(dev))
        if not torch.equal(ref_m, got_m):
            problems.append(f"missing_chunks: torch ops {ref_m.shape[0]} chunks {ref_m[:4].tolist()}..., "
                            f"metal {got_m.shape[0]} chunks {got_m[:4].tolist()}...")
    if problems:
        raise RuntimeError("sparse_pool_metal self-check against sparse_pool_ops failed: " + "; ".join(problems))
