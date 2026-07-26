"""Fused Triton kernels for the gap-expander winding-radius pipeline.

GapExpandingTransform._call/_inverse need, per query point, only the two
winding radii bracketing the point -- but the eager path materialises the full
[N, num_windings] pipeline in global memory (winding coords, grid_sample'd
logits, exp, cumsum, gathers, plus searchsorted for the inverse), and its
backward again. Here one kernel walks the windings in registers per point:
each winding's logit is bilinearly sampled from the (pinned, scaled) logit
lattice, exponentiated, and accumulated into the running radius; only the two
bracketing radii are kept. The backward re-walks the windings, scattering the
logit-lattice gradient with atomics and accumulating theta/z/dr gradients in
registers.

Numerical contract: same per-winding arithmetic as the eager path, with two
tolerated deviations -- the sequential running sum differs from torch.cumsum's
parallel-scan order, and mul+add chains may contract into FMAs (see
flow_triton). Validated by tolerance tests plus 1-step-checkpoint comparison
against the run-to-run noise floor.

Set FIT_SPIRAL_TRITON=0 to fall back to the eager implementation.
"""
import math
import os

import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:  # pragma: no cover - triton is present on CUDA installs
    _HAS_TRITON = False

_TWO_PI = 2 * math.pi


def gap_triton_available(*tensors):
    if not _HAS_TRITON or os.environ.get('FIT_SPIRAL_TRITON', '1') == '0':
        return False
    return all(
        t.is_cuda and t.dtype == torch.float32 for t in tensors if t is not None
    )


if _HAS_TRITON:

    @triton.jit
    def _sample_logit(logits_ptr, ux_raw, y0, wy0, wy1, y0_ok, y1_ok,
                      T, tm1f, lane_mask):
        # Bilinear sample of the logit lattice at (uy, ux), replicating ATen
        # grid_sampler_2d(align_corners=True, padding_mode='border'). The y
        # row weights are shared across windings and passed in.
        ux = tl.minimum(tl.maximum(ux_raw, 0.0), tm1f)
        x0f = tl.math.floor(ux)
        x0 = x0f.to(tl.int32)
        wx1 = ux - x0f
        wx0 = (x0f + 1.0) - ux
        x0_ok = lane_mask
        x1_ok = lane_mask & (x0 + 1 < T)
        base0 = y0.to(tl.int64) * T
        base1 = base0 + T
        v_nw = tl.load(logits_ptr + base0 + x0, mask=y0_ok & x0_ok, other=0.0)
        v_ne = tl.load(logits_ptr + base0 + x0 + 1, mask=y0_ok & x1_ok, other=0.0)
        v_sw = tl.load(logits_ptr + base1 + x0, mask=y1_ok & x0_ok, other=0.0)
        v_se = tl.load(logits_ptr + base1 + x0 + 1, mask=y1_ok & x1_ok, other=0.0)
        logit = (v_nw * (wx0 * wy0) + v_ne * (wx1 * wy0)) \
            + (v_sw * (wx0 * wy1) + v_se * (wx1 * wy1))
        return logit, x0, wx0, wx1, x1_ok, v_nw, v_ne, v_sw, v_se

    @triton.jit
    def _row_weights(z, min_z, zrange, NZ, nzm1f, lane_mask):
        # z -> grid_sample row coordinate and bilinear row weights (shared by
        # every winding of a point).
        zg = (z - min_z) / zrange * 2.0 - 1.0
        uy_raw = ((zg + 1.0) / 2.0) * nzm1f
        uy = tl.minimum(tl.maximum(uy_raw, 0.0), nzm1f)
        y0f = tl.math.floor(uy)
        y0 = y0f.to(tl.int32)
        wy1 = uy - y0f
        wy0 = (y0f + 1.0) - uy
        y0_ok = lane_mask
        y1_ok = lane_mask & (y0 + 1 < NZ)
        return uy_raw, uy, y0, wy0, wy1, y0_ok, y1_ok

    @triton.jit
    def _gap_fwd_kernel(theta_ptr, z_ptr, iw_ptr, target_ptr,
                        r_in_ptr, r_out_ptr,
                        logits_ptr, idx_ptr, dr_ptr,
                        N, W, T, NZ, tm1f, nzm1f,
                        two_pi, idx_total, min_z, zrange, tf,
                        SEARCH: tl.constexpr, HAS_TF: tl.constexpr,
                        BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        i = pid * BLOCK + tl.arange(0, BLOCK)
        m = i < N
        theta = tl.load(theta_ptr + i, mask=m, other=0.0)
        z = tl.load(z_ptr + i, mask=m, other=0.0)
        dr = tl.load(dr_ptr)
        tn = theta / two_pi
        _, _, y0, wy0, wy1, y0_ok, y1_ok = _row_weights(
            z, min_z, zrange, NZ, nzm1f, m)
        wz = dr * tn

        if SEARCH:
            target = tl.load(target_ptr + i, mask=m, other=0.0)
            iw = tl.zeros(theta.shape, dtype=tl.int32)
            found = tl.zeros(theta.shape, dtype=tl.int1)
            w_stop = W
        else:
            iw = tl.load(iw_ptr + i, mask=m, other=0).to(tl.int32)
            # Nothing past the block's largest bracketing pair is consumed.
            w_stop = tl.minimum(tl.max(tl.where(m, iw, 0), 0) + 2, W)

        running = tl.zeros(theta.shape, dtype=tl.float32)
        r_in = tl.zeros(theta.shape, dtype=tl.float32)
        r_out = tl.zeros(theta.shape, dtype=tl.float32)
        r_prev = tl.zeros(theta.shape, dtype=tl.float32)
        for w in range(w_stop):
            idx_w = tl.load(idx_ptr + w)
            idx_w1 = tl.load(idx_ptr + w + 1)
            coord = idx_w + tn * (idx_w1 - idx_w)
            xg = coord / idx_total * 2.0 - 1.0
            ux_raw = ((xg + 1.0) / 2.0) * tm1f
            logit, _, _, _, _, _, _, _, _ = _sample_logit(
                logits_ptr, ux_raw, y0, wy0, wy1, y0_ok, y1_ok, T, tm1f, m)
            s = tl.exp(logit * 2.0e2)
            if HAS_TF:
                s = 1.0 + tf * (s - 1.0)
            radii_w = wz + running
            if SEARCH:
                newly = (~found) & (radii_w >= target)
                iw_new = tl.minimum(tl.maximum(w - 1, 0), W - 2)
                # found at w>=1: bracketing radii are the previous and current
                r_in = tl.where(newly & (iw_new == w - 1), r_prev, r_in)
                r_out = tl.where(newly & (iw_new == w - 1), radii_w, r_out)
                # found at w==0 (clip): inner radius is the current one
                r_in = tl.where(newly & (iw_new == w), radii_w, r_in)
                iw = tl.where(newly, iw_new, iw)
                found = found | newly
                # outer radius for the w==0 case arrives one iteration later
                r_out = tl.where(found & (w == iw + 1), radii_w, r_out)
                # never-found default: iw = W-2, bracketed by the last two
                r_in = tl.where((~found) & (w == W - 2), radii_w, r_in)
                r_out = tl.where((~found) & (w == W - 1), radii_w, r_out)
                iw = tl.where(found, iw, W - 2)
                r_prev = radii_w
            else:
                r_in = tl.where(w == iw, radii_w, r_in)
                r_out = tl.where(w == iw + 1, radii_w, r_out)
            running = running + dr * s
        tl.store(r_in_ptr + i, r_in, mask=m)
        tl.store(r_out_ptr + i, r_out, mask=m)
        if SEARCH:
            tl.store(iw_ptr + i, iw.to(tl.int64), mask=m)

    @triton.jit
    def _gap_bwd_kernel(theta_ptr, z_ptr, iw_ptr, g_in_ptr, g_out_ptr,
                        g_theta_ptr, g_z_ptr, g_dr_ptr, g_logits_ptr,
                        logits_ptr, idx_ptr, dr_ptr,
                        N, W, T, NZ, tm1f, nzm1f,
                        two_pi, idx_total, min_z, zrange, tf,
                        HAS_TF: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        i = pid * BLOCK + tl.arange(0, BLOCK)
        m = i < N
        theta = tl.load(theta_ptr + i, mask=m, other=0.0)
        z = tl.load(z_ptr + i, mask=m, other=0.0)
        iw = tl.load(iw_ptr + i, mask=m, other=0).to(tl.int32)
        g_in = tl.load(g_in_ptr + i, mask=m, other=0.0)
        g_out = tl.load(g_out_ptr + i, mask=m, other=0.0)
        dr = tl.load(dr_ptr)
        tn = theta / two_pi
        uy_raw, _, y0, wy0, wy1, y0_ok, y1_ok = _row_weights(
            z, min_z, zrange, NZ, nzm1f, m)

        g_io = g_in + g_out
        # winding_zero_radii = dr * theta_norm contributes to both outputs
        g_tn = g_io * dr
        g_dr = g_io * tn
        g_uy = tl.zeros(theta.shape, dtype=tl.float32)
        # g_d is identically zero past each point's iw, so the block only
        # needs to walk to its largest iw + 1.
        w_stop = tl.minimum(tl.max(tl.where(m, iw, 0), 0) + 1, W)
        for w in range(w_stop):
            # d_w = dr*scale_w feeds radii_v for every v > w (exclusive cumsum)
            g_d = tl.where(w < iw, g_io, tl.where(w < iw + 1, g_out, 0.0))
            live = m & (g_d != 0.0)
            idx_w = tl.load(idx_ptr + w)
            idx_w1 = tl.load(idx_ptr + w + 1)
            coord = idx_w + tn * (idx_w1 - idx_w)
            xg = coord / idx_total * 2.0 - 1.0
            ux_raw = ((xg + 1.0) / 2.0) * tm1f
            logit, x0, wx0, wx1, x1_ok, v_nw, v_ne, v_sw, v_se = _sample_logit(
                logits_ptr, ux_raw, y0, wy0, wy1, y0_ok, y1_ok, T, tm1f, live)
            s_raw = tl.exp(logit * 2.0e2)
            if HAS_TF:
                s_used = 1.0 + tf * (s_raw - 1.0)
                g_s_raw = g_d * dr * tf
            else:
                s_used = s_raw
                g_s_raw = g_d * dr
            g_dr += g_d * s_used
            g_logit = g_s_raw * s_raw * 2.0e2
            # scatter dL/dlogits into the 4 bilinear corners
            base0 = y0.to(tl.int64) * T
            base1 = base0 + T
            tl.atomic_add(g_logits_ptr + base0 + x0, g_logit * (wx0 * wy0), mask=live & y0_ok)
            tl.atomic_add(g_logits_ptr + base0 + x0 + 1, g_logit * (wx1 * wy0), mask=live & y0_ok & x1_ok)
            tl.atomic_add(g_logits_ptr + base1 + x0, g_logit * (wx0 * wy1), mask=live & y1_ok)
            tl.atomic_add(g_logits_ptr + base1 + x0 + 1, g_logit * (wx1 * wy1), mask=live & y1_ok & x1_ok)
            # coordinate gradients (border clamp zeroes out-of-range coords)
            gx_ok = ((ux_raw >= 0.0) & (ux_raw <= tm1f)).to(tl.float32)
            g_ux = g_logit * ((v_ne - v_nw) * wy0 + (v_se - v_sw) * wy1) * gx_ok
            g_coord = g_ux * (tm1f / 2.0) * 2.0 / idx_total
            g_tn += tl.where(live, g_coord * (idx_w1 - idx_w), 0.0)
            g_uy += tl.where(live, g_logit * ((v_sw - v_nw) * wx0 + (v_se - v_ne) * wx1), 0.0)
        gy_ok = ((uy_raw >= 0.0) & (uy_raw <= nzm1f)).to(tl.float32)
        g_z = g_uy * gy_ok * (nzm1f / 2.0) * 2.0 / zrange
        tl.store(g_theta_ptr + i, g_tn / two_pi, mask=m)
        tl.store(g_z_ptr + i, g_z, mask=m)
        tl.store(g_dr_ptr + i, g_dr, mask=m)


_BLOCK = 128


class _GapRadii(torch.autograd.Function):
    # Fused replacement for get_transformed_winding_radii + the bracketing
    # gathers (and searchsorted, for the inverse). Differentiable in theta, z,
    # the pinned+scaled logits, and dr_per_winding.

    @staticmethod
    def forward(ctx, theta, z, logits2d, idx, idx_total, dr, iw, target,
                truncate_frac, min_z, max_z):
        n = theta.shape[0]
        W = idx.shape[0] - 1
        NZ, T = logits2d.shape
        search = target is not None
        r_in = torch.empty_like(theta)
        r_out = torch.empty_like(theta)
        if search:
            iw = torch.empty(n, device=theta.device, dtype=torch.int64)
        if n > 0:
            _gap_fwd_kernel[(triton.cdiv(n, _BLOCK),)](
                theta, z, iw, target if search else theta,
                r_in, r_out, logits2d, idx, dr,
                n, W, T, NZ, float(T - 1), float(NZ - 1),
                _TWO_PI, float(idx_total),
                float(min_z), float(max_z) - float(min_z),
                0.0 if truncate_frac is None else float(truncate_frac),
                SEARCH=search, HAS_TF=truncate_frac is not None,
                BLOCK=_BLOCK,
            )
        ctx.save_for_backward(theta, z, logits2d, idx, dr, iw)
        ctx.truncate_frac = truncate_frac
        ctx.idx_total = float(idx_total)
        ctx.min_z = float(min_z)
        ctx.max_z = float(max_z)
        ctx.mark_non_differentiable(iw)
        return r_in, r_out, iw

    @staticmethod
    def backward(ctx, g_in, g_out, _g_iw):
        theta, z, logits2d, idx, dr, iw = ctx.saved_tensors
        n = theta.shape[0]
        W = idx.shape[0] - 1
        NZ, T = logits2d.shape
        if g_in is None:
            g_in = torch.zeros_like(theta)
        if g_out is None:
            g_out = torch.zeros_like(theta)
        g_theta = torch.empty_like(theta)
        g_z = torch.empty_like(theta)
        g_dr_partial = torch.empty_like(theta)
        g_logits = torch.zeros_like(logits2d)
        if n > 0:
            _gap_bwd_kernel[(triton.cdiv(n, _BLOCK),)](
                theta, z, iw, g_in.contiguous(), g_out.contiguous(),
                g_theta, g_z, g_dr_partial, g_logits,
                logits2d, idx, dr,
                n, W, T, NZ, float(T - 1), float(NZ - 1),
                _TWO_PI, ctx.idx_total,
                ctx.min_z, ctx.max_z - ctx.min_z,
                0.0 if ctx.truncate_frac is None else float(ctx.truncate_frac),
                HAS_TF=ctx.truncate_frac is not None, BLOCK=_BLOCK,
            )
        g_dr = g_dr_partial.sum().reshape(dr.shape)
        return g_theta, g_z, g_logits, None, None, g_dr, None, None, None, None, None


def gap_bracketing_radii(theta, z, pinned_scaled_logits, idx, idx_total, dr,
                         inner_winding_clipped, truncate_frac, min_z, max_z):
    # _call path: the bracketing winding index is already known.
    shape = theta.shape
    r_in, r_out, _ = _GapRadii.apply(
        theta.reshape(-1).contiguous(), z.reshape(-1).contiguous(),
        pinned_scaled_logits.reshape(pinned_scaled_logits.shape[-2:]),
        idx, idx_total, dr, inner_winding_clipped.reshape(-1).contiguous(), None,
        truncate_frac, min_z, max_z)
    return r_in.view(shape), r_out.view(shape)


def gap_search_radii(theta, z, pinned_scaled_logits, idx, idx_total, dr,
                     transformed_radius, truncate_frac, min_z, max_z):
    # _inverse path: searchsorted over the (increasing) transformed radii is
    # folded into the winding walk.
    shape = theta.shape
    r_in, r_out, iw = _GapRadii.apply(
        theta.reshape(-1).contiguous(), z.reshape(-1).contiguous(),
        pinned_scaled_logits.reshape(pinned_scaled_logits.shape[-2:]),
        idx, idx_total, dr, None, transformed_radius.reshape(-1).contiguous(),
        truncate_frac, min_z, max_z)
    return r_in.view(shape), r_out.view(shape), iw.view(shape)
