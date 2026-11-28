"""Fused Metal kernels for the gap-expander winding radii on Apple GPUs (MPS).

The Metal counterpart of gap_triton.py, which runs only on CUDA. Without a
fused kernel, MPS takes the eager path: GapExpandingTransform materialises the
whole [N, num_windings] pipeline (winding coordinates, a grid_sample of the
logit lattice, the lower-bounded softplus, a cumsum and the bracketing
gathers, plus searchsorted for the inverse) and its backward again. Here one
thread walks one point's windings in registers and keeps only the two
bracketing radii; the backward re-walks them and scatters the logit-lattice
gradient with float atomics.

The kernels follow gap_triton's arithmetic and evaluation order term by term,
so the numerical contract is the same: per-winding arithmetic as the eager
path, a sequential running sum instead of torch.cumsum's scan order, possible
mul+add contraction into FMAs, and atomic accumulation order in the backward.
tests/test_gap_fused.py checks both fused backends against the eager path.

One difference from the Triton kernels changes a gradient, at one boundary:
the point gradient is zero exactly on the logit lattice's last row or column,
as on MPS's eager path (gathers that repeat that row or column:
geom_utils.gather_grid_sample_border), where the Triton kernels take the slope
towards a missing, zero-weighted neighbour. (CUDA's eager path, grid_sample,
gives no gradient on the first row or column either; this kernel, like MPS's
eager path, keeps the one-sided slope there.)

Two further differences from the Triton kernels change no result:
- Each thread stops at its own last useful winding instead of its block's.
  The Triton kernels walk to the block's largest bracketing index, and every
  winding past a point's own contributes nothing to that point.
- Lattice indices are clamped into the lattice before loads, so a NaN
  coordinate cannot read out of bounds. Finite coordinates are already in
  range after the border clamp.

The kernels are compiled once per process with torch.mps.compile_shader
(PyTorch's safe math mode). Set FIT_SPIRAL_METAL=0 to use the eager path.
"""
import math
import os

import torch

_TWO_PI = 2 * math.pi
_GROUP = 256
_LIB = None

_SOURCE = r'''
#include <metal_stdlib>
using namespace metal;

// max(x, 0) + log(1 + exp(-abs(x))) is stable in both tails.
inline float gap_softplus(float v) {
    return max(v, 0.0f) + log(1.0f + exp(-abs(v)));
}

// Avoid exp(-value) overflow in the negative tail.
inline float gap_sigmoid(float v) {
    float e = exp(-abs(v));
    return v >= 0.0f ? 1.0f / (1.0f + e) : e / (1.0f + e);
}

struct RowWeights {
    float uy_raw;
    int y0;
    float wy0;
    float wy1;
    bool y1_ok;
};

// z -> grid_sample row coordinate and bilinear row weights, shared by every
// winding of a point (ATen grid_sampler_2d, align_corners=True, border).
inline RowWeights row_weights(float z, float min_z, float zrange, int nz,
                              float nzm1f) {
    RowWeights r;
    float zg = (z - min_z) / zrange * 2.0f - 1.0f;
    r.uy_raw = ((zg + 1.0f) / 2.0f) * nzm1f;
    float uy = min(max(r.uy_raw, 0.0f), nzm1f);
    float y0f = floor(uy);
    r.y0 = clamp(int(y0f), 0, nz - 1);
    r.wy1 = uy - y0f;
    r.wy0 = (y0f + 1.0f) - uy;
    r.y1_ok = r.y0 + 1 < nz;
    return r;
}

struct LogitSample {
    float logit;
    int x0;
    float wx0;
    float wx1;
    bool x1_ok;
    float v_nw;
    float v_ne;
    float v_sw;
    float v_se;
};

// Bilinear sample of the logit lattice at (uy, ux) with border padding.
inline LogitSample sample_logit(device const float* logits, float ux_raw,
                                RowWeights r, int t, float tm1f) {
    LogitSample s;
    float ux = min(max(ux_raw, 0.0f), tm1f);
    float x0f = floor(ux);
    s.x0 = clamp(int(x0f), 0, t - 1);
    s.wx1 = ux - x0f;
    s.wx0 = (x0f + 1.0f) - ux;
    s.x1_ok = s.x0 + 1 < t;
    long base0 = long(r.y0) * long(t);
    long base1 = base0 + long(t);
    s.v_nw = logits[base0 + s.x0];
    s.v_ne = s.x1_ok ? logits[base0 + s.x0 + 1] : 0.0f;
    s.v_sw = r.y1_ok ? logits[base1 + s.x0] : 0.0f;
    s.v_se = (r.y1_ok && s.x1_ok) ? logits[base1 + s.x0 + 1] : 0.0f;
    s.logit = (s.v_nw * (s.wx0 * r.wy0) + s.v_ne * (s.wx1 * r.wy0))
        + (s.v_sw * (s.wx0 * r.wy1) + s.v_se * (s.wx1 * r.wy1));
    return s;
}

// Winding w's grid_sample column coordinate at normalised angle tn.
inline float winding_ux(device const float* idx, int w, float tn,
                        float idx_total, float tm1f) {
    float idx_w = idx[w];
    float idx_w1 = idx[w + 1];
    float coord = idx_w + tn * (idx_w1 - idx_w);
    float xg = coord / idx_total * 2.0f - 1.0f;
    return ((xg + 1.0f) / 2.0f) * tm1f;
}

template <bool SEARCH, bool HAS_TF>
kernel void gap_fwd(device const float* theta [[buffer(0)]],
                    device const float* zs [[buffer(1)]],
                    device long* iw_buf [[buffer(2)]],
                    device const float* target [[buffer(3)]],
                    device float* r_in_buf [[buffer(4)]],
                    device float* r_out_buf [[buffer(5)]],
                    device const float* logits [[buffer(6)]],
                    device const float* idx [[buffer(7)]],
                    device const float* dr_buf [[buffer(8)]],
                    constant long& n_points [[buffer(9)]],
                    constant long& n_windings [[buffer(10)]],
                    constant long& n_t [[buffer(11)]],
                    constant long& n_z [[buffer(12)]],
                    constant float& tm1f [[buffer(13)]],
                    constant float& nzm1f [[buffer(14)]],
                    constant float& two_pi [[buffer(15)]],
                    constant float& idx_total [[buffer(16)]],
                    constant float& min_z [[buffer(17)]],
                    constant float& zrange [[buffer(18)]],
                    constant float& tf [[buffer(19)]],
                    constant float& min_gap [[buffer(20)]],
                    constant float& softplus_bias [[buffer(21)]],
                    constant float& softplus_denominator [[buffer(22)]],
                    constant float& softplus_scale [[buffer(23)]],
                    uint gid [[thread_position_in_grid]]) {
    if (long(gid) >= n_points) {
        return;
    }
    const int W = int(n_windings);
    const int T = int(n_t);
    const float dr = dr_buf[0];
    const float tn = theta[gid] / two_pi;
    const RowWeights r = row_weights(zs[gid], min_z, zrange, int(n_z), nzm1f);
    const float wz = dr * tn;

    int iw = 0;
    bool found = false;
    float tgt = 0.0f;
    int w_stop = W;
    if (SEARCH) {
        tgt = target[gid];
    } else {
        iw = int(iw_buf[gid]);
        // Nothing past the bracketing pair is consumed.
        w_stop = min(iw + 2, W);
    }

    float running = 0.0f;
    float r_in = 0.0f;
    float r_out = 0.0f;
    float r_prev = 0.0f;
    for (int w = 0; w < w_stop; ++w) {
        const float ux_raw = winding_ux(idx, w, tn, idx_total, tm1f);
        const LogitSample s = sample_logit(logits, ux_raw, r, T, tm1f);
        const float argument = softplus_bias + softplus_scale * s.logit;
        const float ratio = gap_softplus(argument) / softplus_denominator;
        float gap = min_gap + (dr - min_gap) * ratio;
        if (HAS_TF) {
            gap = dr + tf * (gap - dr);
        }
        const float radii_w = wz + running;
        if (SEARCH) {
            const bool newly = !found && radii_w >= tgt;
            const int iw_new = min(max(w - 1, 0), W - 2);
            // found at w >= 1: the bracketing radii are the previous and current
            if (newly && iw_new == w - 1) {
                r_in = r_prev;
                r_out = radii_w;
            }
            // found at w == 0 (clip): the inner radius is the current one
            if (newly && iw_new == w) {
                r_in = radii_w;
            }
            if (newly) {
                iw = iw_new;
            }
            found = found || newly;
            // the outer radius for the w == 0 case arrives one iteration later
            if (found && w == iw + 1) {
                r_out = radii_w;
            }
            // never-found default: iw = W - 2, bracketed by the last two radii
            if (!found && w == W - 2) {
                r_in = radii_w;
            }
            if (!found && w == W - 1) {
                r_out = radii_w;
            }
            if (!found) {
                iw = W - 2;
            }
            r_prev = radii_w;
            if (found && w >= iw + 1) {
                break;  // later windings change nothing
            }
        } else {
            if (w == iw) {
                r_in = radii_w;
            }
            if (w == iw + 1) {
                r_out = radii_w;
            }
        }
        running = running + gap;
    }
    r_in_buf[gid] = r_in;
    r_out_buf[gid] = r_out;
    if (SEARCH) {
        iw_buf[gid] = long(iw);
    }
}

template <bool HAS_TF>
kernel void gap_bwd(device const float* theta [[buffer(0)]],
                    device const float* zs [[buffer(1)]],
                    device const long* iw_buf [[buffer(2)]],
                    device const float* g_in_buf [[buffer(3)]],
                    device const float* g_out_buf [[buffer(4)]],
                    device float* g_theta_buf [[buffer(5)]],
                    device float* g_z_buf [[buffer(6)]],
                    device float* g_dr_buf [[buffer(7)]],
                    device float* g_logits [[buffer(8)]],
                    device const float* logits [[buffer(9)]],
                    device const float* idx [[buffer(10)]],
                    device const float* dr_buf [[buffer(11)]],
                    constant long& n_points [[buffer(12)]],
                    constant long& n_windings [[buffer(13)]],
                    constant long& n_t [[buffer(14)]],
                    constant long& n_z [[buffer(15)]],
                    constant float& tm1f [[buffer(16)]],
                    constant float& nzm1f [[buffer(17)]],
                    constant float& two_pi [[buffer(18)]],
                    constant float& idx_total [[buffer(19)]],
                    constant float& min_z [[buffer(20)]],
                    constant float& zrange [[buffer(21)]],
                    constant float& tf [[buffer(22)]],
                    constant float& min_gap [[buffer(23)]],
                    constant float& softplus_bias [[buffer(24)]],
                    constant float& softplus_denominator [[buffer(25)]],
                    constant float& softplus_scale [[buffer(26)]],
                    uint gid [[thread_position_in_grid]]) {
    if (long(gid) >= n_points) {
        return;
    }
    const int W = int(n_windings);
    const int T = int(n_t);
    const float dr = dr_buf[0];
    const float tn = theta[gid] / two_pi;
    const RowWeights r = row_weights(zs[gid], min_z, zrange, int(n_z), nzm1f);
    const int iw = int(iw_buf[gid]);
    const float g_in = g_in_buf[gid];
    const float g_out = g_out_buf[gid];
    device atomic_float* acc = (device atomic_float*)g_logits;

    const float g_io = g_in + g_out;
    // winding_zero_radii = dr * theta_norm contributes to both outputs
    float g_tn = g_io * dr;
    float g_dr = g_io * tn;
    float g_uy = 0.0f;
    // gap_w feeds radii_v for every v > w (exclusive cumsum): both bracketing
    // radii for w < iw, only the outer one for w == iw, neither past it.
    const int w_stop = min(iw + 1, W);
    for (int w = 0; w < w_stop; ++w) {
        const float g_d = w < iw ? g_io : g_out;
        if (g_d == 0.0f) {
            continue;
        }
        const float idx_w = idx[w];
        const float idx_w1 = idx[w + 1];
        const float ux_raw = winding_ux(idx, w, tn, idx_total, tm1f);
        const LogitSample s = sample_logit(logits, ux_raw, r, T, tm1f);
        const float argument = softplus_bias + softplus_scale * s.logit;
        const float ratio = gap_softplus(argument) / softplus_denominator;
        const float sigmoid = gap_sigmoid(argument);
        float dr_derivative;
        float gap_logit_derivative;
        if (HAS_TF) {
            dr_derivative = 1.0f + tf * (ratio - 1.0f);
            gap_logit_derivative = tf * (dr - min_gap) * sigmoid
                * softplus_scale / softplus_denominator;
        } else {
            dr_derivative = ratio;
            gap_logit_derivative = (dr - min_gap) * sigmoid
                * softplus_scale / softplus_denominator;
        }
        g_dr += g_d * dr_derivative;
        const float g_logit = g_d * gap_logit_derivative;
        // scatter dL/dlogits into the 4 bilinear corners
        const long base0 = long(r.y0) * long(T);
        const long base1 = base0 + long(T);
        atomic_fetch_add_explicit(acc + base0 + s.x0,
                                  g_logit * (s.wx0 * r.wy0),
                                  memory_order_relaxed);
        if (s.x1_ok) {
            atomic_fetch_add_explicit(acc + base0 + s.x0 + 1,
                                      g_logit * (s.wx1 * r.wy0),
                                      memory_order_relaxed);
        }
        if (r.y1_ok) {
            atomic_fetch_add_explicit(acc + base1 + s.x0,
                                      g_logit * (s.wx0 * r.wy1),
                                      memory_order_relaxed);
        }
        if (r.y1_ok && s.x1_ok) {
            atomic_fetch_add_explicit(acc + base1 + s.x0 + 1,
                                      g_logit * (s.wx1 * r.wy1),
                                      memory_order_relaxed);
        }
        // coordinate gradients: none outside the lattice (the border clamp)
        // nor exactly on its last column, where MPS's eager border sample
        // repeats that column (see gy_ok below)
        const float gx_ok = (ux_raw >= 0.0f && ux_raw < tm1f) ? 1.0f : 0.0f;
        const float g_ux = g_logit
            * ((s.v_ne - s.v_nw) * r.wy0 + (s.v_se - s.v_sw) * r.wy1) * gx_ok;
        const float g_coord = g_ux * (tm1f / 2.0f) * 2.0f / idx_total;
        g_tn += g_coord * (idx_w1 - idx_w);
        g_uy += g_logit * ((s.v_sw - s.v_nw) * s.wx0 + (s.v_se - s.v_ne) * s.wx1);
    }
    // z has no gradient outside the lattice (the border clamp) nor exactly on
    // its last row: there MPS's eager path (gathers that repeat the last row:
    // geom_utils.gather_grid_sample_border) is flat in z, while the missing
    // row's zero weights here would give a slope. On the first row both take
    // the one-sided slope into the lattice.
    const float gy_ok = (r.uy_raw >= 0.0f && r.uy_raw < nzm1f) ? 1.0f : 0.0f;
    g_theta_buf[gid] = g_tn / two_pi;
    g_z_buf[gid] = g_uy * gy_ok * (nzm1f / 2.0f) * 2.0f / zrange;
    g_dr_buf[gid] = g_dr;
}

template [[host_name("gap_fwd_bracket")]] kernel void gap_fwd<false, false>(
    device const float*, device const float*, device long*, device const float*,
    device float*, device float*, device const float*, device const float*,
    device const float*, constant long&, constant long&, constant long&,
    constant long&, constant float&, constant float&, constant float&,
    constant float&, constant float&, constant float&, constant float&,
    constant float&, constant float&, constant float&, constant float&, uint);
template [[host_name("gap_fwd_bracket_tf")]] kernel void gap_fwd<false, true>(
    device const float*, device const float*, device long*, device const float*,
    device float*, device float*, device const float*, device const float*,
    device const float*, constant long&, constant long&, constant long&,
    constant long&, constant float&, constant float&, constant float&,
    constant float&, constant float&, constant float&, constant float&,
    constant float&, constant float&, constant float&, constant float&, uint);
template [[host_name("gap_fwd_search")]] kernel void gap_fwd<true, false>(
    device const float*, device const float*, device long*, device const float*,
    device float*, device float*, device const float*, device const float*,
    device const float*, constant long&, constant long&, constant long&,
    constant long&, constant float&, constant float&, constant float&,
    constant float&, constant float&, constant float&, constant float&,
    constant float&, constant float&, constant float&, constant float&, uint);
template [[host_name("gap_fwd_search_tf")]] kernel void gap_fwd<true, true>(
    device const float*, device const float*, device long*, device const float*,
    device float*, device float*, device const float*, device const float*,
    device const float*, constant long&, constant long&, constant long&,
    constant long&, constant float&, constant float&, constant float&,
    constant float&, constant float&, constant float&, constant float&,
    constant float&, constant float&, constant float&, constant float&, uint);
template [[host_name("gap_bwd")]] kernel void gap_bwd<false>(
    device const float*, device const float*, device const long*,
    device const float*, device const float*, device float*, device float*,
    device float*, device float*, device const float*, device const float*,
    device const float*, constant long&, constant long&, constant long&,
    constant long&, constant float&, constant float&, constant float&,
    constant float&, constant float&, constant float&, constant float&,
    constant float&, constant float&, constant float&, constant float&, uint);
template [[host_name("gap_bwd_tf")]] kernel void gap_bwd<true>(
    device const float*, device const float*, device const long*,
    device const float*, device const float*, device float*, device float*,
    device float*, device float*, device const float*, device const float*,
    device const float*, constant long&, constant long&, constant long&,
    constant long&, constant float&, constant float&, constant float&,
    constant float&, constant float&, constant float&, constant float&,
    constant float&, constant float&, constant float&, constant float&, uint);
'''


def _lib():
    global _LIB
    if _LIB is None:
        _LIB = torch.mps.compile_shader(_SOURCE)
    return _LIB


def gap_metal_available(*tensors):
    if os.environ.get('FIT_SPIRAL_METAL', '1') == '0':
        return False
    if not torch.backends.mps.is_available():
        return False
    return all(
        t.device.type == 'mps' and t.dtype == torch.float32
        for t in tensors if t is not None
    )


def _consts(n, W, T, NZ, idx_total, min_z, max_z, truncate_frac, min_gap,
            softplus_bias, softplus_denominator, softplus_scale):
    # Python ints bind as Metal `long`, floats as `float`.
    return (int(n), int(W), int(T), int(NZ), float(T - 1), float(NZ - 1),
            _TWO_PI, float(idx_total), float(min_z),
            float(max_z) - float(min_z),
            0.0 if truncate_frac is None else float(truncate_frac),
            float(min_gap), float(softplus_bias),
            float(softplus_denominator), float(softplus_scale))


class _GapRadiiMetal(torch.autograd.Function):
    # Fused replacement for get_transformed_winding_radii + the bracketing
    # gathers (and searchsorted, for the inverse). Differentiable in theta, z,
    # the pinned+scaled logits, and dr_per_winding.

    @staticmethod
    def forward(ctx, theta, z, logits2d, idx, idx_total, dr, iw, target,
                truncate_frac, min_z, max_z, min_gap, softplus_bias,
                softplus_denominator, softplus_scale):
        n = theta.shape[0]
        W = idx.shape[0] - 1
        NZ, T = logits2d.shape
        search = target is not None
        has_tf = truncate_frac is not None
        r_in = torch.empty_like(theta)
        r_out = torch.empty_like(theta)
        if search:
            iw = torch.empty(n, device=theta.device, dtype=torch.int64)
        consts = _consts(n, W, T, NZ, idx_total, min_z, max_z, truncate_frac,
                         min_gap, softplus_bias, softplus_denominator,
                         softplus_scale)
        if n > 0:
            name = ('gap_fwd_search' if search else 'gap_fwd_bracket') \
                + ('_tf' if has_tf else '')
            getattr(_lib(), name)(
                theta, z, iw, target if search else theta, r_in, r_out,
                logits2d, idx, dr, *consts, threads=n, group_size=_GROUP)
        ctx.save_for_backward(theta, z, logits2d, idx, dr, iw)
        ctx.consts = consts
        ctx.has_tf = has_tf
        ctx.mark_non_differentiable(iw)
        return r_in, r_out, iw

    @staticmethod
    def backward(ctx, g_in, g_out, _g_iw):
        theta, z, logits2d, idx, dr, iw = ctx.saved_tensors
        n = theta.shape[0]
        if g_in is None:
            g_in = torch.zeros_like(theta)
        if g_out is None:
            g_out = torch.zeros_like(theta)
        g_theta = torch.empty_like(theta)
        g_z = torch.empty_like(theta)
        g_dr_partial = torch.empty_like(theta)
        g_logits = torch.zeros_like(logits2d)
        if n > 0:
            getattr(_lib(), 'gap_bwd_tf' if ctx.has_tf else 'gap_bwd')(
                theta, z, iw, g_in.contiguous(), g_out.contiguous(),
                g_theta, g_z, g_dr_partial, g_logits, logits2d, idx, dr,
                *ctx.consts, threads=n, group_size=_GROUP)
        g_dr = g_dr_partial.sum().reshape(dr.shape)
        return (g_theta, g_z, g_logits, None, None, g_dr, None, None,
                None, None, None, None, None, None, None)


def gap_bracketing_radii(theta, z, pinned_scaled_logits, idx, idx_total, dr,
                         inner_winding_clipped, truncate_frac, min_z, max_z,
                         min_gap, softplus_bias, softplus_denominator,
                         softplus_scale):
    # _call path: the bracketing winding index is already known.
    shape = theta.shape
    r_in, r_out, _ = _GapRadiiMetal.apply(
        theta.reshape(-1).contiguous(), z.reshape(-1).contiguous(),
        pinned_scaled_logits.reshape(pinned_scaled_logits.shape[-2:]).contiguous(),
        idx, idx_total, dr.contiguous(),
        inner_winding_clipped.reshape(-1).contiguous(), None,
        truncate_frac, min_z, max_z, min_gap, softplus_bias,
        softplus_denominator, softplus_scale)
    return r_in.view(shape), r_out.view(shape)


def gap_search_radii(theta, z, pinned_scaled_logits, idx, idx_total, dr,
                     transformed_radius, truncate_frac, min_z, max_z,
                     min_gap, softplus_bias, softplus_denominator,
                     softplus_scale):
    # _inverse path: searchsorted over the (increasing) transformed radii is
    # folded into the winding walk.
    shape = theta.shape
    r_in, r_out, iw = _GapRadiiMetal.apply(
        theta.reshape(-1).contiguous(), z.reshape(-1).contiguous(),
        pinned_scaled_logits.reshape(pinned_scaled_logits.shape[-2:]).contiguous(),
        idx, idx_total, dr.contiguous(), None,
        transformed_radius.reshape(-1).contiguous(),
        truncate_frac, min_z, max_z, min_gap, softplus_bias,
        softplus_denominator, softplus_scale)
    return r_in.view(shape), r_out.view(shape), iw.view(shape)
