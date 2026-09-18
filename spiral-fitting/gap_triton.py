"""Fused Triton kernels for the gap-expander winding-radius pipeline.

GapExpandingTransform._call/_inverse need, per query point, only the two
winding radii bracketing the point -- but the eager path materialises the full
[N, num_windings] pipeline in global memory (winding coords, grid_sample'd
logits, lower-bounded softplus, cumsum, gathers, plus searchsorted for the inverse), and its
backward again. Here one kernel walks the windings in registers per point:
each winding's logit is bilinearly sampled from the (pinned, scaled) logit
lattice, mapped through the stable lower-bounded softplus, and accumulated;
only the two
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
    def _softplus(value):
        # max(x, 0) + log(1 + exp(-abs(x))) is stable in both tails.
        return tl.maximum(value, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(value)))

    @triton.jit
    def _sigmoid(value):
        # Avoid exp(-value) overflow in the negative tail.
        exp_negative_abs = tl.exp(-tl.abs(value))
        return tl.where(
            value >= 0.0,
            1.0 / (1.0 + exp_negative_abs),
            exp_negative_abs / (1.0 + exp_negative_abs),
        )

    @triton.jit
    def _gap_fwd_kernel(theta_ptr, z_ptr, iw_ptr, target_ptr,
                        r_in_ptr, r_out_ptr,
                        logits_ptr, idx_ptr, dr_ptr,
                        N, W, T, NZ, tm1f, nzm1f,
                        two_pi, idx_total, min_z, zrange, tf,
                        min_gap, softplus_bias, softplus_denominator,
                        softplus_scale,
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
            argument = softplus_bias + softplus_scale * logit
            ratio = _softplus(argument) / softplus_denominator
            gap = min_gap + (dr - min_gap) * ratio
            if HAS_TF:
                gap = dr + tf * (gap - dr)
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
            running = running + gap
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
                        min_gap, softplus_bias, softplus_denominator,
                        softplus_scale,
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
            # gap_w feeds radii_v for every v > w (exclusive cumsum)
            g_d = tl.where(w < iw, g_io, tl.where(w < iw + 1, g_out, 0.0))
            live = m & (g_d != 0.0)
            idx_w = tl.load(idx_ptr + w)
            idx_w1 = tl.load(idx_ptr + w + 1)
            coord = idx_w + tn * (idx_w1 - idx_w)
            xg = coord / idx_total * 2.0 - 1.0
            ux_raw = ((xg + 1.0) / 2.0) * tm1f
            logit, x0, wx0, wx1, x1_ok, v_nw, v_ne, v_sw, v_se = _sample_logit(
                logits_ptr, ux_raw, y0, wy0, wy1, y0_ok, y1_ok, T, tm1f, live)
            argument = softplus_bias + softplus_scale * logit
            ratio = _softplus(argument) / softplus_denominator
            sigmoid = _sigmoid(argument)
            if HAS_TF:
                dr_derivative = 1.0 + tf * (ratio - 1.0)
                gap_logit_derivative = (
                    tf * (dr - min_gap) * sigmoid
                    * softplus_scale / softplus_denominator)
            else:
                dr_derivative = ratio
                gap_logit_derivative = (
                    (dr - min_gap) * sigmoid
                    * softplus_scale / softplus_denominator)
            g_dr += g_d * dr_derivative
            g_logit = g_d * gap_logit_derivative
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
                truncate_frac, min_z, max_z, min_gap, softplus_bias,
                softplus_denominator, softplus_scale):
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
                float(min_gap), float(softplus_bias),
                float(softplus_denominator), float(softplus_scale),
                SEARCH=search, HAS_TF=truncate_frac is not None,
                BLOCK=_BLOCK,
            )
        ctx.save_for_backward(theta, z, logits2d, idx, dr, iw)
        ctx.truncate_frac = truncate_frac
        ctx.idx_total = float(idx_total)
        ctx.min_z = float(min_z)
        ctx.max_z = float(max_z)
        ctx.min_gap = float(min_gap)
        ctx.softplus_bias = float(softplus_bias)
        ctx.softplus_denominator = float(softplus_denominator)
        ctx.softplus_scale = float(softplus_scale)
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
                ctx.min_gap, ctx.softplus_bias, ctx.softplus_denominator,
                ctx.softplus_scale,
                HAS_TF=ctx.truncate_frac is not None, BLOCK=_BLOCK,
            )
        g_dr = g_dr_partial.sum().reshape(dr.shape)
        return (g_theta, g_z, g_logits, None, None, g_dr, None, None,
                None, None, None, None, None, None, None)


def gap_bracketing_radii(theta, z, pinned_scaled_logits, idx, idx_total, dr,
                         inner_winding_clipped, truncate_frac, min_z, max_z,
                         min_gap, softplus_bias, softplus_denominator,
                         softplus_scale):
    # _call path: the bracketing winding index is already known.
    shape = theta.shape
    r_in, r_out, _ = _GapRadii.apply(
        theta.reshape(-1).contiguous(), z.reshape(-1).contiguous(),
        pinned_scaled_logits.reshape(pinned_scaled_logits.shape[-2:]),
        idx, idx_total, dr, inner_winding_clipped.reshape(-1).contiguous(), None,
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
    r_in, r_out, iw = _GapRadii.apply(
        theta.reshape(-1).contiguous(), z.reshape(-1).contiguous(),
        pinned_scaled_logits.reshape(pinned_scaled_logits.shape[-2:]),
        idx, idx_total, dr, None, transformed_radius.reshape(-1).contiguous(),
        truncate_frac, min_z, max_z, min_gap, softplus_bias,
        softplus_denominator, softplus_scale)
    return r_in.view(shape), r_out.view(shape), iw.view(shape)


def pinned_affine_map_eager(query, search_knots, base_x, base_y, interval_scale,
                            counts, ray_ids=None, search_query=None):
    """Uncapped per-ray affine intervals, including unit-slope extrapolation.

    Tables include the base anchor in column zero. ``interval_scale[:, j]``
    belongs to the interval starting at anchor j. Search uses left insertion,
    matching the correctness implementation at exact anchor coordinates.
    """
    shape = query.shape
    if ray_ids is None:
        ray_ids = torch.arange(query.shape[0], device=query.device)
        ray_ids = ray_ids.reshape((-1,) + (1,) * (query.ndim - 1)).expand(shape)
    rows = ray_ids.reshape(-1)
    search = query if search_query is None else search_query
    # Fixed logarithmic iterations avoid materialising [samples, anchors]
    # for bundles sharing a ray, and avoid a device sync inside the loop.
    lo = torch.ones_like(rows)
    hi = counts[rows]
    search = search.detach().reshape(-1)
    for _ in range(search_knots.shape[1].bit_length()):
        mid = (lo + hi) // 2
        value = search_knots[rows, mid.clamp(max=search_knots.shape[1] - 1)].detach()
        below = (mid < counts[rows]) & (value < search)
        active = lo < hi
        lo = torch.where(active & below, mid + 1, lo)
        hi = torch.where(active & ~below, mid, hi)
    idx = lo - 1
    scale = interval_scale[rows, idx]
    scale = torch.where(idx + 1 < counts[rows], scale, torch.ones_like(scale))
    return (base_y[rows, idx] + scale * (query.reshape(-1) - base_x[rows, idx])).reshape(shape)


if _HAS_TRITON:
    @triton.jit
    def _pinned_affine_fwd(Q, SQ, K, X, Y, S, C, ROW, OUT, IDX,
                           N: tl.constexpr, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = i < N
        row = tl.load(ROW + i, mask, other=0).to(tl.int64)
        count = tl.load(C + row, mask, other=1).to(tl.int32)
        q = tl.load(Q + i, mask, other=0.)
        search = tl.load(SQ + i, mask, other=0.)
        lo = tl.full((BLOCK,), 1, tl.int32)
        hi = count
        while tl.sum((lo < hi).to(tl.int32), 0) > 0:
            mid = (lo + hi) // 2
            val = tl.load(K + row * WIDTH + mid, mask & (mid < count), other=float('inf'))
            below = val < search
            active = lo < hi
            lo = tl.where(active & below, mid + 1, lo)
            hi = tl.where(active & ~below, mid, hi)
        idx = lo - 1
        offset = row * WIDTH + idx
        x = tl.load(X + offset, mask, other=0.)
        y = tl.load(Y + offset, mask, other=0.)
        scale = tl.load(S + offset, mask & (idx + 1 < count), other=1.)
        tl.store(OUT + i, y + scale * (q - x), mask)
        tl.store(IDX + i, idx, mask)

    @triton.jit
    def _pinned_affine_bwd(Q, X, S, C, ROW, IDX, GO, GQ, GX, GY, GS,
                           N: tl.constexpr, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = i < N
        row = tl.load(ROW + i, mask, other=0).to(tl.int64)
        idx = tl.load(IDX + i, mask, other=0).to(tl.int32)
        count = tl.load(C + row, mask, other=1).to(tl.int32)
        offset = row * WIDTH + idx
        interior = idx + 1 < count
        q = tl.load(Q + i, mask, other=0.)
        x = tl.load(X + offset, mask, other=0.)
        scale = tl.load(S + offset, mask & interior, other=1.)
        grad = tl.load(GO + i, mask, other=0.)
        tl.store(GQ + i, grad * scale, mask)
        tl.atomic_add(GX + offset, -grad * scale, mask)
        tl.atomic_add(GY + offset, grad, mask)
        tl.atomic_add(GS + offset, grad * (q - x), mask & interior)


class _PinnedAffine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, search_query, knots, x, y, scale, counts, rows):
        out = torch.empty_like(query)
        idx = torch.empty_like(rows)
        if query.numel():
            _pinned_affine_fwd[(triton.cdiv(query.numel(), 128),)](
                query, search_query, knots, x, y, scale, counts, rows, out, idx,
                query.numel(), knots.shape[1], 128, enable_fp_fusion=False)
        ctx.save_for_backward(query, x, scale, counts, rows, idx)
        return out

    @staticmethod
    def backward(ctx, grad):
        query, x, scale, counts, rows, idx = ctx.saved_tensors
        gq = torch.empty_like(query)
        gx, gy, gs = torch.zeros_like(x), torch.zeros_like(x), torch.zeros_like(scale)
        if query.numel():
            _pinned_affine_bwd[(triton.cdiv(query.numel(), 128),)](
                query, x, scale, counts, rows, idx, grad.contiguous(), gq, gx, gy, gs,
                query.numel(), x.shape[1], 128, enable_fp_fusion=False)
        return gq, None, None, gx, gy, gs, None, None


def pinned_affine_map(query, search_knots, base_x, base_y, interval_scale,
                      counts, ray_ids=None, search_query=None):
    """Fused affine anchor lookup; eager fallback on CPU/float64 or opt-out.

    No fixed capacity or pin dropping. Backward scatters gradients to the
    shared per-ray tables; CUDA atomic accumulation has association noise.
    """
    search_query = query if search_query is None else search_query
    if not gap_triton_available(query, search_query, search_knots, base_x, base_y, interval_scale):
        return pinned_affine_map_eager(query, search_knots, base_x, base_y,
                                       interval_scale, counts, ray_ids, search_query)
    shape = query.shape
    if ray_ids is None:
        ray_ids = torch.arange(query.shape[0], device=query.device)
        ray_ids = ray_ids.reshape((-1,) + (1,) * (query.ndim - 1)).expand(shape)
    return _PinnedAffine.apply(
        query.reshape(-1).contiguous(), search_query.reshape(-1).contiguous(),
        search_knots.contiguous(), base_x.contiguous(), base_y.contiguous(),
        interval_scale.contiguous(), counts.contiguous(), ray_ids.reshape(-1).contiguous()).reshape(shape)


# ---------------------------------------------------------------------------
# Pin anchor accumulation: per query ray, walk the neighbourhood cells of one
# CSR pin table and accumulate the singular compact kernel into per-slot
# sums, without materialising the (query, pin) pair list. The backward
# re-walks the same pairs (nothing but the inputs is saved).
# ---------------------------------------------------------------------------

_KERNEL_TINY = 1.0e-12
_D_REG = 1.0e-30
# Lanes per program and warps for the anchor walk. Programs run long
# divergent loops, so small blocks keep enough programs in flight.
_ANCHOR_BLOCK = 32
_ANCHOR_WARPS = 1


if _HAS_TRITON:
    from triton.language.extra import libdevice as _libdevice

    @triton.jit
    def _anchor_sums_fwd(TQ, ZQ, OFF, PT, PZ, PET, PEZ, PSLOT, PR, PTGT,
                         MASS, RSUM, SSUM,
                         DR, n_theta, n_z, theta_width, z_width, min_z,
                         radius_theta, radius_z, num_slots, N,
                         TWO_PI: tl.constexpr, KERNEL_TINY: tl.constexpr,
                         D_REG: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = i < N
        dr = tl.load(DR)
        tq = tl.load(TQ + i, mask, other=0.)
        zq = tl.load(ZQ + i, mask, other=0.)
        tb = tl.floor(tq / theta_width).to(tl.int32) % n_theta
        zb = tl.floor((zq - min_z) / z_width).to(tl.int32)
        zb = tl.minimum(tl.maximum(zb, 0), n_z - 1)
        row = i.to(tl.int64) * num_slots
        for dt in range(-radius_theta, radius_theta + 1):
            ctb = (tb + dt + n_theta) % n_theta
            for dz in range(-radius_z, radius_z + 1):
                czb = zb + dz
                ok = mask & (czb >= 0) & (czb < n_z)
                czc = tl.minimum(tl.maximum(czb, 0), n_z - 1)
                cell = (ctb * n_z + czc).to(tl.int64)
                begin = tl.load(OFF + cell, ok, other=0)
                end = tl.load(OFF + cell + 1, ok, other=0)
                count = tl.where(ok, end - begin, 0).to(tl.int32)
                max_count = tl.max(count, 0)
                for j in range(0, max_count):
                    m = ok & (j < count)
                    p = begin + j
                    pt = tl.load(PT + p, m, other=0.)
                    pz = tl.load(PZ + p, m, other=0.)
                    pet = tl.load(PET + p, m, other=1.)
                    pez = tl.load(PEZ + p, m, other=1.)
                    delta = tq - pt
                    wrap = _libdevice.rint(delta / TWO_PI)
                    u = (delta - wrap * TWO_PI) / pet
                    v = (zq - pz) / pez
                    d = tl.sqrt(u * u + v * v + D_REG)
                    inside = d < 1.
                    safe_d = tl.where(inside, d, 1.)
                    k = tl.where(inside, (1. - safe_d) / (safe_d + KERNEL_TINY), 0.)
                    slot = tl.load(PSLOT + p, m, other=0).to(tl.int32) - wrap.to(tl.int32)
                    keep = m & (k > 0.) & (slot >= 0) & (slot < num_slots)
                    pr = tl.load(PR + p, keep, other=0.)
                    ptgt = tl.load(PTGT + p, keep, other=0.)
                    target = ptgt - dr * wrap
                    addr = row + slot
                    tl.atomic_add(MASS + addr, k, keep)
                    tl.atomic_add(RSUM + addr, k * pr, keep)
                    tl.atomic_add(SSUM + addr, k * target, keep)

    @triton.jit
    def _anchor_sums_bwd(TQ, ZQ, OFF, PT, PZ, PET, PEZ, PSLOT, PR, PTGT,
                         GMASS, GRSUM, GSSUM,
                         GTQ, GZQ, GPT, GPZ, GPR, GPTGT, GDR,
                         DR, n_theta, n_z, theta_width, z_width, min_z,
                         radius_theta, radius_z, num_slots, N,
                         TWO_PI: tl.constexpr, KERNEL_TINY: tl.constexpr,
                         D_REG: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = i < N
        dr = tl.load(DR)
        tq = tl.load(TQ + i, mask, other=0.)
        zq = tl.load(ZQ + i, mask, other=0.)
        tb = tl.floor(tq / theta_width).to(tl.int32) % n_theta
        zb = tl.floor((zq - min_z) / z_width).to(tl.int32)
        zb = tl.minimum(tl.maximum(zb, 0), n_z - 1)
        row = i.to(tl.int64) * num_slots
        gtq = tl.zeros((BLOCK,), tl.float32)
        gzq = tl.zeros((BLOCK,), tl.float32)
        gdr = tl.zeros((BLOCK,), tl.float32)
        for dt in range(-radius_theta, radius_theta + 1):
            ctb = (tb + dt + n_theta) % n_theta
            for dz in range(-radius_z, radius_z + 1):
                czb = zb + dz
                ok = mask & (czb >= 0) & (czb < n_z)
                czc = tl.minimum(tl.maximum(czb, 0), n_z - 1)
                cell = (ctb * n_z + czc).to(tl.int64)
                begin = tl.load(OFF + cell, ok, other=0)
                end = tl.load(OFF + cell + 1, ok, other=0)
                count = tl.where(ok, end - begin, 0).to(tl.int32)
                max_count = tl.max(count, 0)
                for j in range(0, max_count):
                    m = ok & (j < count)
                    p = begin + j
                    pt = tl.load(PT + p, m, other=0.)
                    pz = tl.load(PZ + p, m, other=0.)
                    pet = tl.load(PET + p, m, other=1.)
                    pez = tl.load(PEZ + p, m, other=1.)
                    delta = tq - pt
                    wrap = _libdevice.rint(delta / TWO_PI)
                    u = (delta - wrap * TWO_PI) / pet
                    v = (zq - pz) / pez
                    d = tl.sqrt(u * u + v * v + D_REG)
                    inside = d < 1.
                    safe_d = tl.where(inside, d, 1.)
                    k = tl.where(inside, (1. - safe_d) / (safe_d + KERNEL_TINY), 0.)
                    slot = tl.load(PSLOT + p, m, other=0).to(tl.int32) - wrap.to(tl.int32)
                    keep = m & (k > 0.) & (slot >= 0) & (slot < num_slots)
                    pr = tl.load(PR + p, keep, other=0.)
                    ptgt = tl.load(PTGT + p, keep, other=0.)
                    target = ptgt - dr * wrap
                    addr = row + slot
                    gm = tl.load(GMASS + addr, keep, other=0.)
                    gr = tl.load(GRSUM + addr, keep, other=0.)
                    gs = tl.load(GSSUM + addr, keep, other=0.)
                    gk = gm + gr * pr + gs * target
                    tl.atomic_add(GPR + p, gr * k, keep)
                    tl.atomic_add(GPTGT + p, gs * k, keep)
                    gdr += tl.where(keep, -gs * k * wrap, 0.)
                    # d(1-d)/(d+tiny) / dd = -(1+tiny)/(d+tiny)^2 inside the footprint.
                    dkdd = tl.where(inside, -(1. + KERNEL_TINY) / ((safe_d + KERNEL_TINY) * (safe_d + KERNEL_TINY)), 0.)
                    gd = gk * dkdd
                    gu = gd * (u / d)
                    gv = gd * (v / d)
                    gt = tl.where(keep, gu / pet, 0.)
                    gz = tl.where(keep, gv / pez, 0.)
                    gtq += gt
                    gzq += gz
                    tl.atomic_add(GPT + p, -gt, keep)
                    tl.atomic_add(GPZ + p, -gz, keep)
        tl.store(GTQ + i, gtq, mask)
        tl.store(GZQ + i, gzq, mask)
        tl.atomic_add(GDR, tl.sum(gdr, 0))


class _AnchorSums(torch.autograd.Function):
    @staticmethod
    def forward(ctx, theta_q, z_q, pin_theta, pin_z, pin_r, pin_target, dr,
                offsets, pin_eps_theta, pin_eps_z, pin_slot, geometry, num_slots):
        num_queries = theta_q.shape[0]
        mass = torch.zeros([num_queries, num_slots], dtype=theta_q.dtype, device=theta_q.device)
        R_sum = torch.zeros_like(mass)
        S_sum = torch.zeros_like(mass)
        if num_queries and pin_theta.numel():
            _anchor_sums_fwd[(triton.cdiv(num_queries, _ANCHOR_BLOCK),)](
                theta_q, z_q, offsets, pin_theta, pin_z, pin_eps_theta, pin_eps_z,
                pin_slot, pin_r, pin_target, mass, R_sum, S_sum,
                dr.detach().reshape(1).contiguous(), *geometry, num_slots, num_queries,
                _TWO_PI, _KERNEL_TINY, _D_REG, _ANCHOR_BLOCK, num_warps=_ANCHOR_WARPS,
                enable_fp_fusion=False)
        ctx.save_for_backward(theta_q, z_q, pin_theta, pin_z, pin_r, pin_target, dr,
                              offsets, pin_eps_theta, pin_eps_z, pin_slot)
        ctx.geometry = geometry
        ctx.num_slots = num_slots
        return mass, R_sum, S_sum

    @staticmethod
    def backward(ctx, g_mass, g_R, g_S):
        (theta_q, z_q, pin_theta, pin_z, pin_r, pin_target, dr,
         offsets, pin_eps_theta, pin_eps_z, pin_slot) = ctx.saved_tensors
        num_queries = theta_q.shape[0]
        g_tq = torch.zeros_like(theta_q)
        g_zq = torch.zeros_like(z_q)
        g_pt, g_pz, g_pr, g_ptgt = (torch.zeros_like(pin_theta) for _ in range(4))
        g_dr = torch.zeros([1], dtype=theta_q.dtype, device=theta_q.device)
        if num_queries and pin_theta.numel():
            _anchor_sums_bwd[(triton.cdiv(num_queries, _ANCHOR_BLOCK),)](
                theta_q, z_q, offsets, pin_theta, pin_z, pin_eps_theta, pin_eps_z,
                pin_slot, pin_r, pin_target,
                g_mass.contiguous(), g_R.contiguous(), g_S.contiguous(),
                g_tq, g_zq, g_pt, g_pz, g_pr, g_ptgt, g_dr,
                dr.detach().reshape(1).contiguous(), *ctx.geometry, ctx.num_slots, num_queries,
                _TWO_PI, _KERNEL_TINY, _D_REG, _ANCHOR_BLOCK, num_warps=_ANCHOR_WARPS,
                enable_fp_fusion=False)
        return (g_tq, g_zq, g_pt, g_pz, g_pr, g_ptgt, g_dr.reshape(dr.shape),
                None, None, None, None, None, None)


def anchor_sums(theta_q, z_q, pin_theta, pin_z, pin_r, pin_target, dr, *, offsets,
                pin_eps_theta, pin_eps_z, pin_slot, n_theta, n_z, theta_width, z_width,
                min_z, radius_theta, radius_z, num_slots):
    """Fused per-(query, slot) kernel sums over a CSR pin table.

    Returns ``(mass, R_sum, S_sum)`` ``[Q, num_slots]``; ``None`` when the
    fused path is unavailable (caller falls back to the pair-list eager
    implementation). Footprints and slots are constants; gradients flow to
    the query rays, the pin values and ``dr``.
    """
    if not gap_triton_available(theta_q, z_q, pin_theta, pin_z, pin_r, pin_target):
        return None
    if not (dr.is_cuda and dr.dtype == torch.float32):
        return None
    geometry = (int(n_theta), int(n_z), float(theta_width), float(z_width), float(min_z),
                int(radius_theta), int(radius_z))
    return _AnchorSums.apply(
        theta_q.contiguous(), z_q.contiguous(), pin_theta.contiguous(), pin_z.contiguous(),
        pin_r.contiguous(), pin_target.contiguous(), dr,
        offsets.contiguous(), pin_eps_theta.contiguous(), pin_eps_z.contiguous(),
        pin_slot.contiguous(), geometry, int(num_slots))


# ---------------------------------------------------------------------------
# Batched tridiagonal solve (Thomas algorithm), one lane per system, for the
# pin radius blend. Backward is the adjoint (transposed) solve.
# ---------------------------------------------------------------------------


if _HAS_TRITON:
    @triton.jit
    def _thomas_kernel(SUB, DIAG, SUP, RHS, OUT, W, N, BLOCK: tl.constexpr):
        # Row-major [N, W] systems; sub[0] and sup[W-1] are ignored.
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = i < N
        base = i.to(tl.int64) * W
        m_prev = tl.load(DIAG + base, mask, other=1.)
        y_prev = tl.load(RHS + base, mask, other=0.)
        sup_prev = tl.load(SUP + base, mask, other=0.)
        tl.store(DIAG + base, m_prev, mask)
        tl.store(OUT + base, y_prev, mask)
        for j in range(1, W):
            sub = tl.load(SUB + base + j, mask, other=0.)
            diag = tl.load(DIAG + base + j, mask, other=1.)
            rhs = tl.load(RHS + base + j, mask, other=0.)
            factor = sub / m_prev
            m_prev = diag - factor * sup_prev
            y_prev = rhs - factor * y_prev
            # Scratch: the eliminated diagonal and right-hand side, in place
            # in the output buffers (DIAG is a private copy).
            tl.store(DIAG + base + j, m_prev, mask)
            tl.store(OUT + base + j, y_prev, mask)
            sup_prev = tl.load(SUP + base + j, mask, other=0.)
        x_next = y_prev / m_prev
        tl.store(OUT + base + W - 1, x_next, mask)
        for jj in range(1, W):
            j = W - 1 - jj
            y = tl.load(OUT + base + j, mask, other=0.)
            m = tl.load(DIAG + base + j, mask, other=1.)
            sup = tl.load(SUP + base + j, mask, other=0.)
            x_next = (y - sup * x_next) / m
            tl.store(OUT + base + j, x_next, mask)


def _thomas_eager(sub, diag, sup, rhs):
    width = rhs.shape[-1]
    m = [diag[:, 0]]
    y = [rhs[:, 0]]
    for j in range(1, width):
        factor = sub[:, j] / m[-1]
        m.append(diag[:, j] - factor * sup[:, j - 1])
        y.append(rhs[:, j] - factor * y[-1])
    x = [y[-1] / m[-1]]
    for j in range(width - 2, -1, -1):
        x.append((y[j] - sup[:, j] * x[-1]) / m[j])
    return torch.stack(x[::-1], dim=-1)


def _thomas_launch(sub, diag, sup, rhs):
    out = torch.empty_like(rhs)
    scratch = diag.detach().clone()
    n, width = rhs.shape
    if n and width:
        _thomas_kernel[(triton.cdiv(n, 128),)](
            sub.detach().contiguous(), scratch, sup.detach().contiguous(), rhs.detach().contiguous(),
            out, width, n, 128, enable_fp_fusion=False)
    return out


class _Thomas(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sub, diag, sup, rhs):
        x = _thomas_launch(sub, diag, sup, rhs)
        ctx.save_for_backward(sub, diag, sup, x)
        return x

    @staticmethod
    def backward(ctx, grad_x):
        sub, diag, sup, x = ctx.saved_tensors
        # A^T has sub'_j = sup_{j-1}, sup'_j = sub_{j+1}.
        sub_t = torch.cat([torch.zeros_like(sup[:, :1]), sup[:, :-1]], dim=-1)
        sup_t = torch.cat([sub[:, 1:], torch.zeros_like(sub[:, :1])], dim=-1)
        g_rhs = _thomas_launch(sub_t, diag, sup_t, grad_x.contiguous())
        x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=-1)
        x_next = torch.cat([x[:, 1:], torch.zeros_like(x[:, :1])], dim=-1)
        g_diag = -g_rhs * x
        g_sub = -g_rhs * x_prev
        g_sup = -g_rhs * x_next
        return g_sub, g_diag, g_sup, g_rhs


def tridiagonal_solve(sub, diag, sup, rhs):
    """Solve ``A x = rhs`` for row-major ``[N, W]`` tridiagonal systems.

    ``sub[:, j]`` multiplies ``x[:, j-1]`` in row ``j`` (``sub[:, 0]`` unused) and
    ``sup[:, j]`` multiplies ``x[:, j+1]`` (``sup[:, W-1]`` unused). Same
    elimination order as the eager Thomas loop; fused on CUDA float32.
    """
    if not gap_triton_available(sub, diag, sup, rhs):
        return _thomas_eager(sub, diag, sup, rhs)
    return _Thomas.apply(sub.contiguous(), diag.contiguous(), sup.contiguous(), rhs.contiguous())
