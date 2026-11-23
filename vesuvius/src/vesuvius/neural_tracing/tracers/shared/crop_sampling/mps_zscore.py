"""``zscore_crops_torch`` on Apple GPUs in three Metal passes instead of about seven PyTorch ones

Per crop: tile partials (count, mean, M2 about the crop's first value, min, max) merged as Chan et al., then the
population mean and standard deviation, the divisor ``max(std, ZSCORE_EPSILON)`` and the constant-crop rule (all
zeros when min == max, found exactly), then one in-place pass ``(x - mean) / divisor``. Float32 throughout, as
``zscore_crops_torch`` on MPS; outputs agree to about 1e-6
"""
from __future__ import annotations

import torch

from vesuvius.neural_tracing.tracers.shared.crop_sampling.normalization import ZSCORE_EPSILON

_TILES = 64  # partial tiles per crop: 64 x 256 threads keeps every GPU core busy at batch 16
_SOURCE = r'''
#include <metal_stdlib>
using namespace metal;

struct Part { float n, mean, m2, lo, hi; };

static Part merge(Part a, Part b) {
    if (a.n == 0.0f) return b;
    if (b.n == 0.0f) return a;
    const float n = a.n + b.n, delta = b.mean - a.mean;
    return Part{n, a.mean + delta * (b.n / n), a.m2 + b.m2 + delta * delta * (a.n * b.n / n), min(a.lo, b.lo),
                max(a.hi, b.hi)};
}

// grid (TILES * 256, B); x [B, N] with N % 4 == 0; parts [B, TILES, 5]. Sums are shifted by the crop's first value,
// so the per-thread variance does not cancel. ints: N, TILES
kernel void zscore_partials(device const float4* x [[buffer(0)]], device float* parts [[buffer(1)]],
                            device const int* ints [[buffer(2)]], uint2 tg [[threadgroup_position_in_grid]],
                            uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                            uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint N4 = uint(ints[0]) / 4, T = uint(ints[1]), b = tg.y;
    device const float4* row = x + ulong(b) * N4;
    const float shift = ((device const float*)row)[0];
    const uint per = (N4 + T - 1) / T, lo = tg.x * per, hi = min(lo + per, N4);
    float s1 = 0.0f, s2 = 0.0f, mn = INFINITY, mx = -INFINITY, n = 0.0f;
    for (uint i = lo + tid; i < hi; i += 256) {
        const float4 v = row[i];
        const float4 d = v - shift;
        s1 += d.x + d.y + d.z + d.w;
        s2 += dot(d, d);
        mn = min(mn, min(min(v.x, v.y), min(v.z, v.w)));
        mx = max(mx, max(max(v.x, v.y), max(v.z, v.w)));
        n += 4.0f;
    }
    Part p = n > 0.0f ? Part{n, shift + s1 / n, max(s2 - s1 * s1 / n, 0.0f), mn, mx}
                      : Part{0.0f, 0.0f, 0.0f, INFINITY, -INFINITY};
    for (uint offset = 16; offset > 0; offset /= 2) {
        Part o{simd_shuffle_down(p.n, offset), simd_shuffle_down(p.mean, offset), simd_shuffle_down(p.m2, offset),
               simd_shuffle_down(p.lo, offset), simd_shuffle_down(p.hi, offset)};
        if (lane + offset < 32) p = merge(p, o);
    }
    threadgroup Part tile[8];
    if (lane == 0) tile[sg] = p;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        Part a = tile[0];
        for (int s = 1; s < 8; ++s) a = merge(a, tile[s]);
        device float* o = parts + (ulong(b) * T + tg.x) * 5;
        o[0] = a.n; o[1] = a.mean; o[2] = a.m2; o[3] = a.lo; o[4] = a.hi;
    }
}

// one thread per crop: stats [B, 2] = (mean, 1 / max(std, eps)), or (0, 0) for a constant crop. ints: N, TILES, B
kernel void zscore_finalize(device const float* parts [[buffer(0)]], device float2* stats [[buffer(1)]],
                            device const int* ints [[buffer(2)]], constant float& eps [[buffer(3)]],
                            uint b [[thread_position_in_grid]]) {
    const uint T = uint(ints[1]);
    if (b >= uint(ints[2])) return;
    Part a{0.0f, 0.0f, 0.0f, INFINITY, -INFINITY};
    for (uint t = 0; t < T; ++t) {
        device const float* q = parts + (ulong(b) * T + t) * 5;
        a = merge(a, Part{q[0], q[1], q[2], q[3], q[4]});
    }
    const float std = sqrt(a.m2 / a.n);
    stats[b] = a.lo == a.hi ? float2(0.0f) : float2(a.mean, 1.0f / max(std, eps));
}

// in place, grid (N / 4, B): x = (x - mean) * inverse. ints: N
kernel void zscore_apply(device float4* x [[buffer(0)]], device const float2* stats [[buffer(1)]],
                         device const int* ints [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {
    const uint N4 = uint(ints[0]) / 4;
    if (gid.x >= N4) return;
    const float2 s = stats[gid.y];
    device float4* v = x + ulong(gid.y) * N4 + gid.x;
    *v = (*v - s.x) * s.y;
}
'''
_LIBRARY = {}


def _library():
    if 'lib' not in _LIBRARY:
        _LIBRARY['lib'] = torch.mps.compile_shader(_SOURCE)
    return _LIBRARY['lib']


def suits(crops):
    """Whether :func:`zscore_crops_metal` takes ``crops``: contiguous float32 MPS crops, voxels per crop % 4 == 0"""
    return (crops.device.type == 'mps' and crops.dtype == torch.float32 and crops.is_contiguous()
            and crops.shape[-3:].numel() % 4 == 0 and len(crops) > 0)


def zscore_crops_metal(crops):
    """``zscore_crops_torch(crops)`` in place on MPS (see :func:`suits`); returns ``crops``"""
    n = crops.shape[-3:].numel()  # one z-score per (sample, channel) volume, as zscore_crops_torch
    b = crops.numel()//n
    lib = _library()
    ints = torch.tensor((n, _TILES, b), dtype=torch.int32, device=crops.device)
    parts = torch.empty((b, _TILES, 5), dtype=torch.float32, device=crops.device)
    stats = torch.empty((b, 2), dtype=torch.float32, device=crops.device)
    lib.zscore_partials(crops, parts, ints, threads=(_TILES*256, b), group_size=(256, 1))
    lib.zscore_finalize(parts, stats, ints, float(ZSCORE_EPSILON), threads=b, group_size=min(b, 64))
    lib.zscore_apply(crops, stats, ints, threads=(n//4, b), group_size=(256, 1))
    return crops
