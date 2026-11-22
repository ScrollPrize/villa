"""Fused Metal RK4 flow integration (direct low-resolution sampling) on MPS.

The Metal counterpart of flow_triton's TritonRK4DirectIntegrate, the path the
shipped default configuration (Cartesian field, model_flow_field_direct_lr)
takes on CUDA. Without it MPS falls back to the upsampled field: every step
upsamples the low-resolution lattice to full resolution, and each slab's
integration runs eagerly (two grid_samples plus stage arithmetic per RK4
stage sample forward, ~25 ops per stage in the adjoint). It also fits a
slightly different interpolant than CUDA (see flow_triton.direct_lr_enabled).

Here, as in the Triton kernels, one thread integrates one point through every
(slab, step, stage) sample in registers, sampling the low-resolution lattice
directly, and one thread runs that point's whole adjoint sweep, scattering the
two field gradients with float atomics into the caller-owned accumulators.
The kernels follow _rk4d_fwd_kernel / _rk4d_bwd_kernel term by term, so the
numerical contract is flow_triton's: same arithmetic and evaluation order, with
mul+add contraction into FMAs and atomic accumulation order free.
tests/test_flow_direct_fused.py checks both fused backends against an eager
PyTorch implementation of the same integrator.

Compiled once per process with torch.mps.compile_shader (PyTorch's safe math
mode). Set FIT_SPIRAL_METAL=0 to turn the kernels off. Lattices with a single
cell along any axis are not taken (lattice_shape_problem): the field then
integrates as it does without a kernel (CartesianFlowField.get_integrator).
"""
import os

import torch

import flow_triton

_GROUP = 256
_LIB = None

_SOURCE = r'''
#include <metal_stdlib>
using namespace metal;

// fp: lo az,bz,ay,by,ax,bx, lo_scale, hi az,bz,ay,by,ax,bx, hi_scale,
//     h, h/2, h/6
// ip: n, loZ, loY, loX, hiZ, hiY, hiX, n_steps, num_slabs
struct Lattice {
    float az, bz, ay, by, ax, bx, scale;
    int Z, Y, X;
    long ch;      // one channel: Z * Y * X
    long stride;  // one slab: 3 channels
};

inline Lattice lattice(constant float* fp, constant long* ip, int which) {
    Lattice l;
    const int f = which * 7;
    const int i = 1 + which * 3;
    l.az = fp[f + 0]; l.bz = fp[f + 1];
    l.ay = fp[f + 2]; l.by = fp[f + 3];
    l.ax = fp[f + 4]; l.bx = fp[f + 5];
    l.scale = fp[f + 6];
    l.Z = int(ip[i + 0]); l.Y = int(ip[i + 1]); l.X = int(ip[i + 2]);
    l.ch = long(l.Z) * long(l.Y) * long(l.X);
    l.stride = l.ch * 3;
    return l;
}

// One trilinear sample (border clamp, ATen grid_sampler_3d corner weights).
inline float3 sample_one(device const float* ptr, float pz, float py, float px,
                         thread const Lattice& l) {
    const float zm1f = float(l.Z - 1);
    const float ym1f = float(l.Y - 1);
    const float xm1f = float(l.X - 1);
    const float cz = min(max(pz * l.az + l.bz, 0.0f), zm1f);
    const float cy = min(max(py * l.ay + l.by, 0.0f), ym1f);
    const float cx = min(max(px * l.ax + l.bx, 0.0f), xm1f);
    const float z0f = floor(cz);
    const float y0f = floor(cy);
    const float x0f = floor(cx);
    const int z0 = int(z0f);
    const int y0 = int(y0f);
    const int x0 = int(x0f);
    const float wz1 = cz - z0f;
    const float wy1 = cy - y0f;
    const float wx1 = cx - x0f;
    const float wz0 = (z0f + 1.0f) - cz;
    const float wy0 = (y0f + 1.0f) - cy;
    const float wx0 = (x0f + 1.0f) - cx;
    float v0 = 0.0f;
    float v1 = 0.0f;
    float v2 = 0.0f;
    for (int dz = 0; dz < 2; ++dz) {
        for (int dy = 0; dy < 2; ++dy) {
            for (int dx = 0; dx < 2; ++dx) {
                const int z = z0 + dz;
                const int y = y0 + dy;
                const int x = x0 + dx;
                const float wz = dz == 1 ? wz1 : wz0;
                const float wy = dy == 1 ? wy1 : wy0;
                const float wx = dx == 1 ? wx1 : wx0;
                const float w = (wx * wy) * wz;
                // z, y, x >= 0 after the clamp (a NaN coordinate is flushed
                // to 0 by the fmax/fmin clamp, as in the Triton kernel).
                if (z < l.Z && y < l.Y && x < l.X) {
                    const long idx = (long(z) * l.Y + y) * l.X + x;
                    v0 += ptr[idx] * w;
                    v1 += ptr[l.ch + idx] * w;
                    v2 += ptr[2 * l.ch + idx] * w;
                }
            }
        }
    }
    return float3(v0, v1, v2);
}

inline float3 sample_pair(float3 p, device const float* lo,
                          thread const Lattice& L, device const float* hi,
                          thread const Lattice& H) {
    const float3 a = sample_one(lo, p.x, p.y, p.z, L);
    const float3 b = sample_one(hi, p.x, p.y, p.z, H);
    return float3(a.x * L.scale + b.x * H.scale,
                  a.y * L.scale + b.y * H.scale,
                  a.z * L.scale + b.z * H.scale);
}

// Points are (z, y, x) in float3 (.x = z, .y = y, .z = x).
inline float3 load3(device const float* p, long row) {
    return float3(p[row * 3 + 0], p[row * 3 + 1], p[row * 3 + 2]);
}

inline void store3(device float* p, long row, float3 v) {
    p[row * 3 + 0] = v.x;
    p[row * 3 + 1] = v.y;
    p[row * 3 + 2] = v.z;
}

template <bool REVERSE, bool STORE_STAGES>
kernel void rk4d_fwd(device const float* y_buf [[buffer(0)]],
                     device float* out [[buffer(1)]],
                     device float* stages [[buffer(2)]],
                     device const float* lo_base [[buffer(3)]],
                     device const float* hi_base [[buffer(4)]],
                     constant float* fp [[buffer(5)]],
                     constant long* ip [[buffer(6)]],
                     uint gid [[thread_position_in_grid]]) {
    const long n = ip[0];
    const long i = long(gid);
    if (i >= n) {
        return;
    }
    const Lattice L = lattice(fp, ip, 0);
    const Lattice H = lattice(fp, ip, 1);
    const float h = fp[14];
    const float h_half = fp[15];
    const float h_sixth = fp[16];
    const int n_steps = int(ip[7]);
    const int num_slabs = int(ip[8]);
    float3 y = load3(y_buf, i);
    for (int step = 0; step < n_steps * num_slabs; ++step) {
        int slab = step / n_steps;
        if (REVERSE) {
            slab = num_slabs - 1 - slab;
        }
        device const float* lo = lo_base + long(slab) * L.stride;
        device const float* hi = hi_base + long(slab) * H.stride;
        if (STORE_STAGES) {
            store3(stages, long(step * 4 + 0) * n + i, y);
        }
        const float3 k1 = sample_pair(y, lo, L, hi, H);
        const float3 x2 = float3(y.x + h_half * k1.x, y.y + h_half * k1.y,
                                 y.z + h_half * k1.z);
        if (STORE_STAGES) {
            store3(stages, long(step * 4 + 1) * n + i, x2);
        }
        const float3 k2 = sample_pair(x2, lo, L, hi, H);
        const float3 x3 = float3(y.x + h_half * k2.x, y.y + h_half * k2.y,
                                 y.z + h_half * k2.z);
        if (STORE_STAGES) {
            store3(stages, long(step * 4 + 2) * n + i, x3);
        }
        const float3 k3 = sample_pair(x3, lo, L, hi, H);
        const float3 x4 = float3(y.x + h * k3.x, y.y + h * k3.y,
                                 y.z + h * k3.z);
        if (STORE_STAGES) {
            store3(stages, long(step * 4 + 3) * n + i, x4);
        }
        const float3 k4 = sample_pair(x4, lo, L, hi, H);
        y = float3(
            y.x + h_sixth * (((k1.x + 2.0f * k2.x) + 2.0f * k3.x) + k4.x),
            y.y + h_sixth * (((k1.y + 2.0f * k2.y) + 2.0f * k3.y) + k4.y),
            y.z + h_sixth * (((k1.z + 2.0f * k2.z) + 2.0f * k3.z) + k4.z));
    }
    store3(out, i, y);
}

// Sampler backward for one field: scatter dL/d(field values) into acc
// (unscaled; the caller applies the field's value scale to the whole
// accumulator afterwards) and return dL/d(point).
template <bool HAS_ACC>
inline float3 bwd_one(float3 g, float3 p, device const float* ptr,
                      device atomic_float* acc, thread const Lattice& l) {
    const float zm1f = float(l.Z - 1);
    const float ym1f = float(l.Y - 1);
    const float xm1f = float(l.X - 1);
    const float zm2f = float(l.Z - 2);
    const float ym2f = float(l.Y - 2);
    const float xm2f = float(l.X - 2);
    const float crz = p.x * l.az + l.bz;
    const float cry = p.y * l.ay + l.by;
    const float crx = p.z * l.ax + l.bx;
    float cz = min(max(crz, 0.0f), zm1f);
    float cy = min(max(cry, 0.0f), ym1f);
    float cx = min(max(crx, 0.0f), xm1f);
    cz = isnan(crz) ? crz : cz;
    cy = isnan(cry) ? cry : cy;
    cx = isnan(crx) ? crx : cx;
    const float lz = min(max(floor(isnan(cz) ? 0.0f : cz), 0.0f), zm2f);
    const float ly = min(max(floor(isnan(cy) ? 0.0f : cy), 0.0f), ym2f);
    const float lx = min(max(floor(isnan(cx) ? 0.0f : cx), 0.0f), xm2f);
    const float fz1 = cz - lz;
    const float fy1 = cy - ly;
    const float fx1 = cx - lx;
    const float fz0 = 1.0f - fz1;
    const float fy0 = 1.0f - fy1;
    const float fx0 = 1.0f - fx1;
    const int z0 = int(lz);
    const int y0 = int(ly);
    const int x0 = int(lx);
    float gcz = 0.0f;
    float gcy = 0.0f;
    float gcx = 0.0f;
    for (int dz = 0; dz < 2; ++dz) {
        for (int dy = 0; dy < 2; ++dy) {
            for (int dx = 0; dx < 2; ++dx) {
                const float fz = dz == 1 ? fz1 : fz0;
                const float fy = dy == 1 ? fy1 : fy0;
                const float fx = dx == 1 ? fx1 : fx0;
                const float w = (fz * fy) * fx;
                const long idx = (long(z0 + dz) * l.Y + (y0 + dy)) * l.X + (x0 + dx);
                if (HAS_ACC) {
                    atomic_fetch_add_explicit(acc + idx, g.x * w,
                                              memory_order_relaxed);
                    atomic_fetch_add_explicit(acc + l.ch + idx, g.y * w,
                                              memory_order_relaxed);
                    atomic_fetch_add_explicit(acc + 2 * l.ch + idx, g.z * w,
                                              memory_order_relaxed);
                }
                const float v0 = ptr[idx];
                const float v1 = ptr[l.ch + idx];
                const float v2 = ptr[2 * l.ch + idx];
                const float vdg = ((v0 * g.x + v1 * g.y) + v2 * g.z) * l.scale;
                const float sz = dz == 1 ? 1.0f : -1.0f;
                const float sy = dy == 1 ? 1.0f : -1.0f;
                const float sx = dx == 1 ? 1.0f : -1.0f;
                gcz += ((vdg * sz) * fy) * fx;
                gcy += ((vdg * sy) * fz) * fx;
                gcx += ((vdg * sx) * fz) * fy;
            }
        }
    }
    const float mz = (crz >= 0.0f && crz <= zm1f) ? 1.0f : 0.0f;
    const float my = (cry >= 0.0f && cry <= ym1f) ? 1.0f : 0.0f;
    const float mx = (crx >= 0.0f && crx <= xm1f) ? 1.0f : 0.0f;
    return float3((gcz * mz) * l.az, (gcy * my) * l.ay, (gcx * mx) * l.ax);
}

template <bool HAS_ACC>
inline float3 bwd_stage_pair(float3 g, float3 p,
                             device const float* lo, device atomic_float* acc_lo,
                             thread const Lattice& L,
                             device const float* hi, device atomic_float* acc_hi,
                             thread const Lattice& H) {
    const float3 a = bwd_one<HAS_ACC>(g, p, lo, acc_lo, L);
    const float3 b = bwd_one<HAS_ACC>(g, p, hi, acc_hi, H);
    return float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

template <bool REVERSE, bool HAS_ACC>
kernel void rk4d_bwd(device const float* grad_y [[buffer(0)]],
                     device float* grad_pts [[buffer(1)]],
                     device const float* stages [[buffer(2)]],
                     device const float* lo_base [[buffer(3)]],
                     device const float* hi_base [[buffer(4)]],
                     device float* acc_lo_base [[buffer(5)]],
                     device float* acc_hi_base [[buffer(6)]],
                     constant float* fp [[buffer(7)]],
                     constant long* ip [[buffer(8)]],
                     uint gid [[thread_position_in_grid]]) {
    const long n = ip[0];
    const long i = long(gid);
    if (i >= n) {
        return;
    }
    const Lattice L = lattice(fp, ip, 0);
    const Lattice H = lattice(fp, ip, 1);
    const float h = fp[14];
    const float h_half = fp[15];
    const float h_sixth = fp[16];
    const int n_steps = int(ip[7]);
    const int num_slabs = int(ip[8]);
    float3 g = load3(grad_y, i);
    for (int step = n_steps * num_slabs - 1; step >= 0; --step) {
        int slab = step / n_steps;
        if (REVERSE) {
            slab = num_slabs - 1 - slab;
        }
        const long lo_off = long(slab) * L.stride;
        const long hi_off = long(slab) * H.stride;
        device const float* lo = lo_base + lo_off;
        device const float* hi = hi_base + hi_off;
        device atomic_float* acc_lo = (device atomic_float*)(acc_lo_base + lo_off);
        device atomic_float* acc_hi = (device atomic_float*)(acc_hi_base + hi_off);
        const float3 g6 = float3(g.x * h_sixth, g.y * h_sixth, g.z * h_sixth);
        float3 p = load3(stages, long(step * 4 + 3) * n + i);
        const float3 b4 = bwd_stage_pair<HAS_ACC>(g6, p, lo, acc_lo, L, hi, acc_hi, H);
        p = load3(stages, long(step * 4 + 2) * n + i);
        const float3 b3 = bwd_stage_pair<HAS_ACC>(
            float3(g6.x * 2.0f + b4.x * h, g6.y * 2.0f + b4.y * h,
                   g6.z * 2.0f + b4.z * h),
            p, lo, acc_lo, L, hi, acc_hi, H);
        p = load3(stages, long(step * 4 + 1) * n + i);
        const float3 b2 = bwd_stage_pair<HAS_ACC>(
            float3(g6.x * 2.0f + b3.x * h_half, g6.y * 2.0f + b3.y * h_half,
                   g6.z * 2.0f + b3.z * h_half),
            p, lo, acc_lo, L, hi, acc_hi, H);
        p = load3(stages, long(step * 4 + 0) * n + i);
        const float3 b1 = bwd_stage_pair<HAS_ACC>(
            float3(g6.x + b2.x * h_half, g6.y + b2.y * h_half,
                   g6.z + b2.z * h_half),
            p, lo, acc_lo, L, hi, acc_hi, H);
        g = float3((((g.x + b4.x) + b3.x) + b2.x) + b1.x,
                   (((g.y + b4.y) + b3.y) + b2.y) + b1.y,
                   (((g.z + b4.z) + b3.z) + b2.z) + b1.z);
    }
    store3(grad_pts, i, g);
}

#define FWD(NAME, REV, STORE) \
template [[host_name(NAME)]] kernel void rk4d_fwd<REV, STORE>( \
    device const float*, device float*, device float*, device const float*, \
    device const float*, constant float*, constant long*, uint);
FWD("rk4d_fwd", false, false)
FWD("rk4d_fwd_rev", true, false)
FWD("rk4d_fwd_stages", false, true)
FWD("rk4d_fwd_stages_rev", true, true)

#define BWD(NAME, REV, ACC) \
template [[host_name(NAME)]] kernel void rk4d_bwd<REV, ACC>( \
    device const float*, device float*, device const float*, \
    device const float*, device const float*, device float*, device float*, \
    constant float*, constant long*, uint);
BWD("rk4d_bwd", false, false)
BWD("rk4d_bwd_rev", true, false)
BWD("rk4d_bwd_acc", false, true)
BWD("rk4d_bwd_acc_rev", true, true)
'''


def _lib():
    global _LIB
    if _LIB is None:
        _LIB = torch.mps.compile_shader(_SOURCE)
    return _LIB


def rk4_metal_available(*tensors):
    # The device and dtype test. Whether the kernels take the lattices'
    # shape is lattice_shape_problem's question.
    if os.environ.get('FIT_SPIRAL_METAL', '1') == '0':
        return False
    if not torch.backends.mps.is_available():
        return False
    return all(
        t.device.type == 'mps' and t.dtype == torch.float32
        for t in tensors if t is not None
    )


def lattice_shape_problem(low, high):
    """Why the fused direct-LR kernels (these and flow_triton's) cannot
    integrate these lattices, or None.

    Their backward's trilinear corner arithmetic needs two cells along every
    axis of both lattices (its lower corner is clamped to size - 2). A narrow
    fit window gives fewer: 100 slices with no z margin are six fine z cells
    at the default 16-voxel spacing, so one coarse cell at scale 6.
    """
    for name, field in (('coarse', low), ('fine', high)):
        for axis, size in zip('zyx', field.shape[2:]):
            if int(size) < 2:
                return (f'the {name} lattice has {int(size)} cell'
                        f'{"" if int(size) == 1 else "s"} along {axis} '
                        f'(shape {tuple(field.shape[2:])}; the fused kernels '
                        'need at least 2 per axis)')
    return None


def _params(n, low, high, lo_scale, hi_scale, h, n_steps):
    problem = lattice_shape_problem(low, high)
    if problem is not None:
        raise ValueError(problem)
    g = flow_triton._field_geoms(low, high)
    fp = [*g[:6], float(lo_scale), *g[6:], float(hi_scale),
          float(h), float(h / 2), float(h / 6)]
    ip = [int(n), *(int(s) for s in low.shape[2:]),
          *(int(s) for s in high.shape[2:]), int(n_steps), int(low.shape[0])]
    return fp, ip


def _run_direct_fwd(y0, low, high, lo_scale, hi_scale, h, n_steps, reverse,
                    stages):
    n = y0.shape[0]
    out = torch.empty_like(y0)
    if n > 0:
        fp, ip = _params(n, low, high, lo_scale, hi_scale, h, n_steps)
        name = 'rk4d_fwd' + ('_stages' if stages is not None else '') \
            + ('_rev' if reverse else '')
        getattr(_lib(), name)(
            y0, out, stages if stages is not None else out,
            low, high, fp, ip, threads=n, group_size=_GROUP)
    return out


def rk4_direct_integrate(y0, low, high, lo_scale, hi_scale, acc_lo, acc_hi,
                         h, n_steps, reverse=False):
    # Same contract as flow_triton.rk4_direct_integrate: low :: slabs, 3, Zl,
    # Yl, Xl and high :: slabs, 3, Z, Y, X, contiguous; the slabs are applied
    # in order (reverse order with `reverse`).
    if torch.is_grad_enabled() and y0.requires_grad:
        return MetalRK4DirectIntegrate.apply(
            y0, low, high, lo_scale, hi_scale, acc_lo, acc_hi, h, n_steps,
            reverse)
    return _run_direct_fwd(
        y0.contiguous(), low, high, lo_scale, hi_scale, h, n_steps, reverse,
        None)


class MetalRK4DirectIntegrate(torch.autograd.Function):
    # RK4 integration sampling the LR lattice directly (no per-step upsample).
    # Field gradients are scattered unscaled into the two caller-owned
    # accumulators; the caller applies each field's value scale afterwards
    # (see CartesianFlowField.apply_accumulated_field_grad).

    @staticmethod
    def forward(ctx, y0, low, high, lo_scale, hi_scale, acc_lo, acc_hi, h,
                n_steps, reverse):
        ctx.set_materialize_grads(False)
        y0 = y0.contiguous()
        stages = flow_triton._stage_buffer(y0, low.shape[0], n_steps)
        out = _run_direct_fwd(
            y0, low, high, lo_scale, hi_scale, h, n_steps, reverse, stages)
        ctx.save_for_backward(low, high, stages)
        ctx.scales = (float(lo_scale), float(hi_scale))
        ctx.accs = (acc_lo, acc_hi)
        ctx.h = float(h)
        ctx.n_steps = int(n_steps)
        ctx.reverse = bool(reverse)
        return out

    @staticmethod
    def backward(ctx, grad_y):
        if grad_y is None:
            return (None,) * 10
        low, high, stages = ctx.saved_tensors
        grad_y = grad_y.contiguous()
        n = grad_y.shape[0]
        grad_pts = torch.empty_like(grad_y)
        acc_lo, acc_hi = ctx.accs
        lo_scale, hi_scale = ctx.scales
        if n > 0:
            fp, ip = _params(n, low, high, lo_scale, hi_scale, ctx.h,
                             ctx.n_steps)
            has_acc = acc_lo is not None
            name = 'rk4d_bwd' + ('_acc' if has_acc else '') \
                + ('_rev' if ctx.reverse else '')
            getattr(_lib(), name)(
                grad_y, grad_pts, stages, low, high,
                acc_lo if has_acc else low, acc_hi if has_acc else high,
                fp, ip, threads=n, group_size=_GROUP)
        return grad_pts, None, None, None, None, None, None, None, None, None
