"""Fused Triton kernels for the piecewise-stationary RK4 flow integration.

The eager path (flow_fields._RK4SparseFlowIntegrate) launches ~10 kernels per
stage sample in forward (two grid_samples plus stage arithmetic) and ~25 in
backward (_sparse_backward_impl: corner gathers, weight products, reductions,
index_add_). Per point the whole integration is independent, so the entire
forward (num_slabs x n_steps x 4 stage samples) runs as ONE kernel here, and
the entire adjoint sweep as ONE kernel, with all intermediates kept in
registers.

Every lattice carries a leading slab axis: slab t is a stationary velocity
field integrated for unit time (n_steps RK4 steps of size h), and the slabs
compose in order. The kernels walk the flattened (slab, step) sequence, so
composing N stationary flows costs one launch per direction, not N. With
REVERSE the slab order is reversed (and the caller negates h), which is the
inverse of the composition.

Numerical contract: same arithmetic and evaluation order as the eager path,
but the compiler is free to contract mul+add chains into FMAs (like nvcc does
inside ATen's own kernels), so results differ from eager at the last-ulp
level. That is below the run-to-run noise the eager path already has from its
atomic index_add_ field-gradient scatter (which the backward here reproduces
with atomics as well). Equivalence is enforced by tolerance-based unit tests
(tests/test_speedup_equivalence.py) plus 1-step-checkpoint comparison against
the run-to-run noise floor on the real fit.

Set FIT_SPIRAL_TRITON=0 to fall back to the eager implementation.
"""
import os

import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:  # pragma: no cover - triton is present on CUDA installs
    _HAS_TRITON = False


_warned_missing_triton = False


def _warn_missing_triton():
    # PyTorch publishes no triton wheel for Windows, so this fallback is the
    # default there rather than the exception the import guard reads like. Say
    # so once, the way tracks.py does when the native crossing consolidator is
    # missing; silently running a much slower path with slightly different
    # results is worse than a line of output.
    global _warned_missing_triton
    if not _warned_missing_triton:
        _warned_missing_triton = True
        print(
            'WARNING: triton is unavailable; the fused RK4 flow kernels fall back '
            'to the much slower eager path, whose results differ slightly')


def rk4_triton_available(*tensors):
    if not _HAS_TRITON:
        _warn_missing_triton()
        return False
    if os.environ.get('FIT_SPIRAL_TRITON', '1') == '0':
        return False
    return all(
        t.is_cuda and t.dtype == torch.float32 for t in tensors if t is not None
    )


def direct_lr_enabled(default=False):
    # Sample the LR flow lattice directly at query points instead of
    # upsampling it to HR resolution every step. Same function except within
    # HR cells that straddle an LR grid plane (the upsample linearises there);
    # kills the F.interpolate forward+backward and its full-resolution
    # transient. Off by default because it slightly changes the effective
    # interpolant. Normally set via the model_flow_field_direct_lr config key
    # (threaded through as `default`); FIT_SPIRAL_DIRECT_LR, when set,
    # overrides the config.
    value = os.environ.get('FIT_SPIRAL_DIRECT_LR')
    if value is None:
        return bool(default)
    return value == '1'


_PERM_STRIDE = 4096


def permute_enabled():
    # Decorrelate adjacent kernel lanes' field-gradient atomics via a
    # deterministic interleave permutation (per-point results bitwise
    # identical; only atomic accumulation order changes). MEASURED SLOWER on
    # H100 (2026-07-18 bench_rk4: bwd natural 14.7 ms vs interleave 22.2 at
    # 2.4M pts): the hardware already aggregates same-address atomics
    # intra-warp, and natural order wins on load locality. Kept off; gate
    # retained for the benchmark.
    return os.environ.get('FIT_SPIRAL_RK4_PERMUTE', '0') == '1'


def _interleave_perm(n, device):
    # Transpose of a (rows, stride) index grid: consecutive output slots are
    # `stride` points apart, so a warp's 32 lanes touch 32 different flow
    # cells. Deterministic (no RNG state touched).
    rows = (n + _PERM_STRIDE - 1) // _PERM_STRIDE
    idx = torch.arange(
        rows * _PERM_STRIDE, device=device).view(rows, _PERM_STRIDE)
    idx = idx.t().reshape(-1)
    if rows * _PERM_STRIDE != n:
        idx = idx[idx < n]
    return idx


if _HAS_TRITON:
    from triton.language.extra import libdevice

    @triton.jit
    def _fwd_sample(pz, py, px, low_ptr, high_ptr, scale,
                    Z, Y, X, zm1f, ym1f, xm1f, ch, lane_mask):
        # One trilinear sample of low + high*scale, replicating
        # sample_field's grid normalisation followed by ATen
        # grid_sampler_3d(align_corners=True, padding_mode='border').
        cz = ((pz * 2.0 - 1.0) + 1.0) / 2.0 * zm1f
        cy = ((py * 2.0 - 1.0) + 1.0) / 2.0 * ym1f
        cx = ((px * 2.0 - 1.0) + 1.0) / 2.0 * xm1f
        # clip_coordinates: min(size-1, max(coord, 0)); fmax/fmin flush NaN.
        cz = tl.minimum(tl.maximum(cz, 0.0), zm1f)
        cy = tl.minimum(tl.maximum(cy, 0.0), ym1f)
        cx = tl.minimum(tl.maximum(cx, 0.0), xm1f)
        z0f = tl.math.floor(cz)
        y0f = tl.math.floor(cy)
        x0f = tl.math.floor(cx)
        z0 = z0f.to(tl.int32)
        y0 = y0f.to(tl.int32)
        x0 = x0f.to(tl.int32)
        wz1 = cz - z0f
        wy1 = cy - y0f
        wx1 = cx - x0f
        wz0 = (z0f + 1.0) - cz
        wy0 = (y0f + 1.0) - cy
        wx0 = (x0f + 1.0) - cx

        lo0 = tl.zeros(pz.shape, dtype=tl.float32)
        lo1 = tl.zeros(pz.shape, dtype=tl.float32)
        lo2 = tl.zeros(pz.shape, dtype=tl.float32)
        hi0 = tl.zeros(pz.shape, dtype=tl.float32)
        hi1 = tl.zeros(pz.shape, dtype=tl.float32)
        hi2 = tl.zeros(pz.shape, dtype=tl.float32)
        for dz in tl.static_range(2):
            for dy in tl.static_range(2):
                for dx in tl.static_range(2):
                    z = z0 + dz
                    y = y0 + dy
                    x = x0 + dx
                    wz = wz1 if dz == 1 else wz0
                    wy = wy1 if dy == 1 else wy0
                    wx = wx1 if dx == 1 else wx0
                    w = (wx * wy) * wz
                    inb = lane_mask & (z < Z) & (y < Y) & (x < X)
                    idx = (z.to(tl.int64) * Y + y) * X + x
                    lo0 += tl.load(low_ptr + idx, mask=inb, other=0.0) * w
                    lo1 += tl.load(low_ptr + ch + idx, mask=inb, other=0.0) * w
                    lo2 += tl.load(low_ptr + 2 * ch + idx, mask=inb, other=0.0) * w
                    hi0 += tl.load(high_ptr + idx, mask=inb, other=0.0) * w
                    hi1 += tl.load(high_ptr + ch + idx, mask=inb, other=0.0) * w
                    hi2 += tl.load(high_ptr + 2 * ch + idx, mask=inb, other=0.0) * w
        return lo0 + hi0 * scale, lo1 + hi1 * scale, lo2 + hi2 * scale

    @triton.jit
    def _rk4_fwd_kernel(y_ptr, out_ptr, stages_ptr,
                        low_base, high_base, scale,
                        N, Z, Y, X, zm1f, ym1f, xm1f,
                        h, h_half, h_sixth, n_steps, num_slabs,
                        REVERSE: tl.constexpr,
                        STORE_STAGES: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        i = pid * BLOCK + tl.arange(0, BLOCK)
        m = i < N
        ch = tl.full((), Z, tl.int64) * Y * X
        slab_stride = ch * 3
        yz = tl.load(y_ptr + i * 3 + 0, mask=m, other=0.0)
        yy = tl.load(y_ptr + i * 3 + 1, mask=m, other=0.0)
        yx = tl.load(y_ptr + i * 3 + 2, mask=m, other=0.0)
        # Flattened (slab, step) walk: step // n_steps is the slab being
        # applied, so the stage-point index `step * 4 + k` stays contiguous.
        for step in range(n_steps * num_slabs):
            slab = step // n_steps
            if REVERSE:
                slab = num_slabs - 1 - slab
            slab_off = slab.to(tl.int64) * slab_stride
            low_ptr = low_base + slab_off
            high_ptr = high_base + slab_off
            if STORE_STAGES:
                s = (step * 4 + 0) * N.to(tl.int64)
                tl.store(stages_ptr + (s + i) * 3 + 0, yz, mask=m)
                tl.store(stages_ptr + (s + i) * 3 + 1, yy, mask=m)
                tl.store(stages_ptr + (s + i) * 3 + 2, yx, mask=m)
            k1z, k1y, k1x = _fwd_sample(yz, yy, yx, low_ptr, high_ptr, scale,
                                        Z, Y, X, zm1f, ym1f, xm1f, ch, m)
            x2z = yz + h_half * k1z
            x2y = yy + h_half * k1y
            x2x = yx + h_half * k1x
            if STORE_STAGES:
                s = (step * 4 + 1) * N.to(tl.int64)
                tl.store(stages_ptr + (s + i) * 3 + 0, x2z, mask=m)
                tl.store(stages_ptr + (s + i) * 3 + 1, x2y, mask=m)
                tl.store(stages_ptr + (s + i) * 3 + 2, x2x, mask=m)
            k2z, k2y, k2x = _fwd_sample(x2z, x2y, x2x, low_ptr, high_ptr, scale,
                                        Z, Y, X, zm1f, ym1f, xm1f, ch, m)
            x3z = yz + h_half * k2z
            x3y = yy + h_half * k2y
            x3x = yx + h_half * k2x
            if STORE_STAGES:
                s = (step * 4 + 2) * N.to(tl.int64)
                tl.store(stages_ptr + (s + i) * 3 + 0, x3z, mask=m)
                tl.store(stages_ptr + (s + i) * 3 + 1, x3y, mask=m)
                tl.store(stages_ptr + (s + i) * 3 + 2, x3x, mask=m)
            k3z, k3y, k3x = _fwd_sample(x3z, x3y, x3x, low_ptr, high_ptr, scale,
                                        Z, Y, X, zm1f, ym1f, xm1f, ch, m)
            x4z = yz + h * k3z
            x4y = yy + h * k3y
            x4x = yx + h * k3x
            if STORE_STAGES:
                s = (step * 4 + 3) * N.to(tl.int64)
                tl.store(stages_ptr + (s + i) * 3 + 0, x4z, mask=m)
                tl.store(stages_ptr + (s + i) * 3 + 1, x4y, mask=m)
                tl.store(stages_ptr + (s + i) * 3 + 2, x4x, mask=m)
            k4z, k4y, k4x = _fwd_sample(x4z, x4y, x4x, low_ptr, high_ptr, scale,
                                        Z, Y, X, zm1f, ym1f, xm1f, ch, m)
            yz = yz + h_sixth * (((k1z + 2.0 * k2z) + 2.0 * k3z) + k4z)
            yy = yy + h_sixth * (((k1y + 2.0 * k2y) + 2.0 * k3y) + k4y)
            yx = yx + h_sixth * (((k1x + 2.0 * k2x) + 2.0 * k3x) + k4x)
        tl.store(out_ptr + i * 3 + 0, yz, mask=m)
        tl.store(out_ptr + i * 3 + 1, yy, mask=m)
        tl.store(out_ptr + i * 3 + 2, yx, mask=m)

    @triton.jit
    def _bwd_stage(gz, gy, gx, pz, py, px,
                   low_ptr, high_ptr, acc_ptr, scale,
                   Z, Y, X, zm1f, ym1f, xm1f, zm2f, ym2f, xm2f, ch,
                   HAS_ACC: tl.constexpr, lane_mask):
        # Sampler backward for one stage point, mirroring
        # flow_fields._sparse_backward_impl.
        crz = pz * zm1f
        cry = py * ym1f
        crx = px * xm1f
        # torch.clamp propagates NaN; fmin/fmax flush it, so reinstate.
        cz = tl.minimum(tl.maximum(crz, 0.0), zm1f)
        cy = tl.minimum(tl.maximum(cry, 0.0), ym1f)
        cx = tl.minimum(tl.maximum(crx, 0.0), xm1f)
        cz = tl.where(crz != crz, crz, cz)
        cy = tl.where(cry != cry, cry, cy)
        cx = tl.where(crx != crx, crx, cx)
        # lo = nan_to_num(coord).floor().clamp(0, size-2)
        lz = tl.minimum(tl.maximum(tl.math.floor(tl.where(cz != cz, 0.0, cz)), 0.0), zm2f)
        ly = tl.minimum(tl.maximum(tl.math.floor(tl.where(cy != cy, 0.0, cy)), 0.0), ym2f)
        lx = tl.minimum(tl.maximum(tl.math.floor(tl.where(cx != cx, 0.0, cx)), 0.0), xm2f)
        fz1 = cz - lz
        fy1 = cy - ly
        fx1 = cx - lx
        fz0 = 1.0 - fz1
        fy0 = 1.0 - fy1
        fx0 = 1.0 - fx1
        z0 = lz.to(tl.int32)
        y0 = ly.to(tl.int32)
        x0 = lx.to(tl.int32)

        gcz = tl.zeros(pz.shape, dtype=tl.float32)
        gcy = tl.zeros(pz.shape, dtype=tl.float32)
        gcx = tl.zeros(pz.shape, dtype=tl.float32)
        for dz in tl.static_range(2):
            for dy in tl.static_range(2):
                for dx in tl.static_range(2):
                    fz = fz1 if dz == 1 else fz0
                    fy = fy1 if dy == 1 else fy0
                    fx = fx1 if dx == 1 else fx0
                    w = (fz * fy) * fx
                    idx = ((z0 + dz).to(tl.int64) * Y + (y0 + dy)) * X + (x0 + dx)
                    if HAS_ACC:
                        tl.atomic_add(acc_ptr + idx, gz * w, mask=lane_mask)
                        tl.atomic_add(acc_ptr + ch + idx, gy * w, mask=lane_mask)
                        tl.atomic_add(acc_ptr + 2 * ch + idx, gx * w, mask=lane_mask)
                    v0 = tl.load(low_ptr + idx, mask=lane_mask, other=0.0) \
                        + tl.load(high_ptr + idx, mask=lane_mask, other=0.0) * scale
                    v1 = tl.load(low_ptr + ch + idx, mask=lane_mask, other=0.0) \
                        + tl.load(high_ptr + ch + idx, mask=lane_mask, other=0.0) * scale
                    v2 = tl.load(low_ptr + 2 * ch + idx, mask=lane_mask, other=0.0) \
                        + tl.load(high_ptr + 2 * ch + idx, mask=lane_mask, other=0.0) * scale
                    vdg = (v0 * gz + v1 * gy) + v2 * gx
                    sz = 1.0 if dz == 1 else -1.0
                    sy = 1.0 if dy == 1 else -1.0
                    sx = 1.0 if dx == 1 else -1.0
                    gcz += ((vdg * sz) * fy) * fx
                    gcy += ((vdg * sy) * fz) * fx
                    gcx += ((vdg * sx) * fz) * fy
        mz = ((crz >= 0.0) & (crz <= zm1f)).to(tl.float32)
        my = ((cry >= 0.0) & (cry <= ym1f)).to(tl.float32)
        mx = ((crx >= 0.0) & (crx <= xm1f)).to(tl.float32)
        return (gcz * mz) * zm1f, (gcy * my) * ym1f, (gcx * mx) * xm1f

    @triton.jit
    def _bwd_stage_defer(gz, gy, gx, pz, py, px,
                         low_ptr, high_ptr, scale,
                         Z, Y, X, zm1f, ym1f, xm1f, zm2f, ym2f, xm2f, ch,
                         lane_mask):
        # _bwd_stage without the atomic scatter: returns the point gradient
        # plus the cell coords and the 8 bilinear corner weights so the
        # caller can accumulate the field-gradient contribution in registers.
        crz = pz * zm1f
        cry = py * ym1f
        crx = px * xm1f
        cz = tl.minimum(tl.maximum(crz, 0.0), zm1f)
        cy = tl.minimum(tl.maximum(cry, 0.0), ym1f)
        cx = tl.minimum(tl.maximum(crx, 0.0), xm1f)
        cz = tl.where(crz != crz, crz, cz)
        cy = tl.where(cry != cry, cry, cy)
        cx = tl.where(crx != crx, crx, cx)
        lz = tl.minimum(tl.maximum(tl.math.floor(tl.where(cz != cz, 0.0, cz)), 0.0), zm2f)
        ly = tl.minimum(tl.maximum(tl.math.floor(tl.where(cy != cy, 0.0, cy)), 0.0), ym2f)
        lx = tl.minimum(tl.maximum(tl.math.floor(tl.where(cx != cx, 0.0, cx)), 0.0), xm2f)
        fz1 = cz - lz
        fy1 = cy - ly
        fx1 = cx - lx
        fz0 = 1.0 - fz1
        fy0 = 1.0 - fy1
        fx0 = 1.0 - fx1
        z0 = lz.to(tl.int32)
        y0 = ly.to(tl.int32)
        x0 = lx.to(tl.int32)

        gcz = tl.zeros(pz.shape, dtype=tl.float32)
        gcy = tl.zeros(pz.shape, dtype=tl.float32)
        gcx = tl.zeros(pz.shape, dtype=tl.float32)
        for dz in tl.static_range(2):
            for dy in tl.static_range(2):
                for dx in tl.static_range(2):
                    fz = fz1 if dz == 1 else fz0
                    fy = fy1 if dy == 1 else fy0
                    fx = fx1 if dx == 1 else fx0
                    idx = ((z0 + dz).to(tl.int64) * Y + (y0 + dy)) * X + (x0 + dx)
                    v0 = tl.load(low_ptr + idx, mask=lane_mask, other=0.0) \
                        + tl.load(high_ptr + idx, mask=lane_mask, other=0.0) * scale
                    v1 = tl.load(low_ptr + ch + idx, mask=lane_mask, other=0.0) \
                        + tl.load(high_ptr + ch + idx, mask=lane_mask, other=0.0) * scale
                    v2 = tl.load(low_ptr + 2 * ch + idx, mask=lane_mask, other=0.0) \
                        + tl.load(high_ptr + 2 * ch + idx, mask=lane_mask, other=0.0) * scale
                    vdg = (v0 * gz + v1 * gy) + v2 * gx
                    sz = 1.0 if dz == 1 else -1.0
                    sy = 1.0 if dy == 1 else -1.0
                    sx = 1.0 if dx == 1 else -1.0
                    gcz += ((vdg * sz) * fy) * fx
                    gcy += ((vdg * sy) * fz) * fx
                    gcx += ((vdg * sx) * fz) * fy
        mz = ((crz >= 0.0) & (crz <= zm1f)).to(tl.float32)
        my = ((cry >= 0.0) & (cry <= ym1f)).to(tl.float32)
        mx = ((crx >= 0.0) & (crx <= xm1f)).to(tl.float32)
        return ((gcz * mz) * zm1f, (gcy * my) * ym1f, (gcx * mx) * xm1f,
                z0, y0, x0, fz0, fz1, fy0, fy1, fx0, fx1)

    @triton.jit
    def _rk4_bwd_kernel(grad_y_ptr, grad_pts_ptr, stages_ptr,
                        low_base, high_base, acc_base, scale,
                        N, Z, Y, X, zm1f, ym1f, xm1f, zm2f, ym2f, xm2f,
                        h, h_half, h_sixth, n_steps, num_slabs,
                        REVERSE: tl.constexpr,
                        HAS_ACC: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        i = pid * BLOCK + tl.arange(0, BLOCK)
        m = i < N
        ch = tl.full((), Z, tl.int64) * Y * X
        slab_stride = ch * 3
        gz = tl.load(grad_y_ptr + i * 3 + 0, mask=m, other=0.0)
        gy = tl.load(grad_y_ptr + i * 3 + 1, mask=m, other=0.0)
        gx = tl.load(grad_y_ptr + i * 3 + 2, mask=m, other=0.0)
        for step in range(n_steps * num_slabs - 1, -1, -1):
            slab = step // n_steps
            if REVERSE:
                slab = num_slabs - 1 - slab
            slab_off = slab.to(tl.int64) * slab_stride
            low_ptr = low_base + slab_off
            high_ptr = high_base + slab_off
            acc_ptr = acc_base + slab_off
            s1 = (step * 4 + 0) * N.to(tl.int64)
            s2 = (step * 4 + 1) * N.to(tl.int64)
            s3 = (step * 4 + 2) * N.to(tl.int64)
            s4 = (step * 4 + 3) * N.to(tl.int64)
            g6z = gz * h_sixth
            g6y = gy * h_sixth
            g6x = gx * h_sixth
            pz = tl.load(stages_ptr + (s4 + i) * 3 + 0, mask=m, other=0.0)
            py = tl.load(stages_ptr + (s4 + i) * 3 + 1, mask=m, other=0.0)
            px = tl.load(stages_ptr + (s4 + i) * 3 + 2, mask=m, other=0.0)
            b4z, b4y, b4x = _bwd_stage(
                g6z, g6y, g6x, pz, py, px, low_ptr, high_ptr, acc_ptr, scale,
                Z, Y, X, zm1f, ym1f, xm1f, zm2f, ym2f, xm2f, ch, HAS_ACC, m)
            pz = tl.load(stages_ptr + (s3 + i) * 3 + 0, mask=m, other=0.0)
            py = tl.load(stages_ptr + (s3 + i) * 3 + 1, mask=m, other=0.0)
            px = tl.load(stages_ptr + (s3 + i) * 3 + 2, mask=m, other=0.0)
            b3z, b3y, b3x = _bwd_stage(
                g6z * 2.0 + b4z * h, g6y * 2.0 + b4y * h, g6x * 2.0 + b4x * h,
                pz, py, px, low_ptr, high_ptr, acc_ptr, scale,
                Z, Y, X, zm1f, ym1f, xm1f, zm2f, ym2f, xm2f, ch, HAS_ACC, m)
            pz = tl.load(stages_ptr + (s2 + i) * 3 + 0, mask=m, other=0.0)
            py = tl.load(stages_ptr + (s2 + i) * 3 + 1, mask=m, other=0.0)
            px = tl.load(stages_ptr + (s2 + i) * 3 + 2, mask=m, other=0.0)
            b2z, b2y, b2x = _bwd_stage(
                g6z * 2.0 + b3z * h_half, g6y * 2.0 + b3y * h_half, g6x * 2.0 + b3x * h_half,
                pz, py, px, low_ptr, high_ptr, acc_ptr, scale,
                Z, Y, X, zm1f, ym1f, xm1f, zm2f, ym2f, xm2f, ch, HAS_ACC, m)
            pz = tl.load(stages_ptr + (s1 + i) * 3 + 0, mask=m, other=0.0)
            py = tl.load(stages_ptr + (s1 + i) * 3 + 1, mask=m, other=0.0)
            px = tl.load(stages_ptr + (s1 + i) * 3 + 2, mask=m, other=0.0)
            b1z, b1y, b1x = _bwd_stage(
                g6z + b2z * h_half, g6y + b2y * h_half, g6x + b2x * h_half,
                pz, py, px, low_ptr, high_ptr, acc_ptr, scale,
                Z, Y, X, zm1f, ym1f, xm1f, zm2f, ym2f, xm2f, ch, HAS_ACC, m)
            gz = ((gz + b4z) + b3z + b2z) + b1z
            gy = ((gy + b4y) + b3y + b2y) + b1y
            gx = ((gx + b4x) + b3x + b2x) + b1x
        tl.store(grad_pts_ptr + i * 3 + 0, gz, mask=m)
        tl.store(grad_pts_ptr + i * 3 + 1, gy, mask=m)
        tl.store(grad_pts_ptr + i * 3 + 2, gx, mask=m)


if _HAS_TRITON:

    # ---- direct-LR mode: per-field lattice geometry, two accumulators ----
    # Each field f is sampled at c_f = p_hr_coord * a_f + b_f per dim, where
    # for the HR field (a, b) = (S_hr - 1, 0) and for the LR field the affine
    # composes grid_sample's align_corners=True unnormalisation with
    # F.interpolate(align_corners=False)'s source-index map, so the direct
    # sample agrees with the upsample-then-sample path on HR lattice planes.

    @triton.jit
    def _sample_one(ptr, pz, py, px, az, bz, ay, by, ax, bx,
                    Z, Y, X, lane_mask):
        zm1f = (Z - 1).to(tl.float32)
        ym1f = (Y - 1).to(tl.float32)
        xm1f = (X - 1).to(tl.float32)
        cz = tl.minimum(tl.maximum(pz * az + bz, 0.0), zm1f)
        cy = tl.minimum(tl.maximum(py * ay + by, 0.0), ym1f)
        cx = tl.minimum(tl.maximum(px * ax + bx, 0.0), xm1f)
        z0f = tl.math.floor(cz)
        y0f = tl.math.floor(cy)
        x0f = tl.math.floor(cx)
        z0 = z0f.to(tl.int32)
        y0 = y0f.to(tl.int32)
        x0 = x0f.to(tl.int32)
        wz1 = cz - z0f
        wy1 = cy - y0f
        wx1 = cx - x0f
        wz0 = (z0f + 1.0) - cz
        wy0 = (y0f + 1.0) - cy
        wx0 = (x0f + 1.0) - cx
        ch = Z.to(tl.int64) * Y * X
        v0 = tl.zeros(pz.shape, dtype=tl.float32)
        v1 = tl.zeros(pz.shape, dtype=tl.float32)
        v2 = tl.zeros(pz.shape, dtype=tl.float32)
        for dz in tl.static_range(2):
            for dy in tl.static_range(2):
                for dx in tl.static_range(2):
                    z = z0 + dz
                    y = y0 + dy
                    x = x0 + dx
                    wz = wz1 if dz == 1 else wz0
                    wy = wy1 if dy == 1 else wy0
                    wx = wx1 if dx == 1 else wx0
                    w = (wx * wy) * wz
                    inb = lane_mask & (z < Z) & (y < Y) & (x < X)
                    idx = (z.to(tl.int64) * Y + y) * X + x
                    v0 += tl.load(ptr + idx, mask=inb, other=0.0) * w
                    v1 += tl.load(ptr + ch + idx, mask=inb, other=0.0) * w
                    v2 += tl.load(ptr + 2 * ch + idx, mask=inb, other=0.0) * w
        return v0, v1, v2

    @triton.jit
    def _sample_pair(pz, py, px,
                     lo_ptr, lo_az, lo_bz, lo_ay, lo_by, lo_ax, lo_bx,
                     loZ, loY, loX, lo_scale,
                     hi_ptr, hi_az, hi_bz, hi_ay, hi_by, hi_ax, hi_bx,
                     hiZ, hiY, hiX, hi_scale, lane_mask):
        l0, l1, l2 = _sample_one(lo_ptr, pz, py, px, lo_az, lo_bz, lo_ay,
                                 lo_by, lo_ax, lo_bx, loZ, loY, loX, lane_mask)
        h0, h1, h2 = _sample_one(hi_ptr, pz, py, px, hi_az, hi_bz, hi_ay,
                                 hi_by, hi_ax, hi_bx, hiZ, hiY, hiX, lane_mask)
        return (l0 * lo_scale + h0 * hi_scale,
                l1 * lo_scale + h1 * hi_scale,
                l2 * lo_scale + h2 * hi_scale)

    @triton.jit
    def _rk4d_fwd_kernel(y_ptr, out_ptr, stages_ptr,
                         lo_base, lo_az, lo_bz, lo_ay, lo_by, lo_ax, lo_bx,
                         loZ, loY, loX, lo_scale,
                         hi_base, hi_az, hi_bz, hi_ay, hi_by, hi_ax, hi_bx,
                         hiZ, hiY, hiX, hi_scale,
                         N, h, h_half, h_sixth, n_steps, num_slabs,
                         REVERSE: tl.constexpr,
                         STORE_STAGES: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        i = pid * BLOCK + tl.arange(0, BLOCK)
        m = i < N
        lo_stride = loZ.to(tl.int64) * loY * loX * 3
        hi_stride = hiZ.to(tl.int64) * hiY * hiX * 3
        yz = tl.load(y_ptr + i * 3 + 0, mask=m, other=0.0)
        yy = tl.load(y_ptr + i * 3 + 1, mask=m, other=0.0)
        yx = tl.load(y_ptr + i * 3 + 2, mask=m, other=0.0)
        for step in range(n_steps * num_slabs):
            slab = step // n_steps
            if REVERSE:
                slab = num_slabs - 1 - slab
            lo_ptr = lo_base + slab.to(tl.int64) * lo_stride
            hi_ptr = hi_base + slab.to(tl.int64) * hi_stride
            if STORE_STAGES:
                s = (step * 4 + 0) * N.to(tl.int64)
                tl.store(stages_ptr + (s + i) * 3 + 0, yz, mask=m)
                tl.store(stages_ptr + (s + i) * 3 + 1, yy, mask=m)
                tl.store(stages_ptr + (s + i) * 3 + 2, yx, mask=m)
            k1z, k1y, k1x = _sample_pair(
                yz, yy, yx,
                lo_ptr, lo_az, lo_bz, lo_ay, lo_by, lo_ax, lo_bx, loZ, loY, loX, lo_scale,
                hi_ptr, hi_az, hi_bz, hi_ay, hi_by, hi_ax, hi_bx, hiZ, hiY, hiX, hi_scale, m)
            x2z = yz + h_half * k1z
            x2y = yy + h_half * k1y
            x2x = yx + h_half * k1x
            if STORE_STAGES:
                s = (step * 4 + 1) * N.to(tl.int64)
                tl.store(stages_ptr + (s + i) * 3 + 0, x2z, mask=m)
                tl.store(stages_ptr + (s + i) * 3 + 1, x2y, mask=m)
                tl.store(stages_ptr + (s + i) * 3 + 2, x2x, mask=m)
            k2z, k2y, k2x = _sample_pair(
                x2z, x2y, x2x,
                lo_ptr, lo_az, lo_bz, lo_ay, lo_by, lo_ax, lo_bx, loZ, loY, loX, lo_scale,
                hi_ptr, hi_az, hi_bz, hi_ay, hi_by, hi_ax, hi_bx, hiZ, hiY, hiX, hi_scale, m)
            x3z = yz + h_half * k2z
            x3y = yy + h_half * k2y
            x3x = yx + h_half * k2x
            if STORE_STAGES:
                s = (step * 4 + 2) * N.to(tl.int64)
                tl.store(stages_ptr + (s + i) * 3 + 0, x3z, mask=m)
                tl.store(stages_ptr + (s + i) * 3 + 1, x3y, mask=m)
                tl.store(stages_ptr + (s + i) * 3 + 2, x3x, mask=m)
            k3z, k3y, k3x = _sample_pair(
                x3z, x3y, x3x,
                lo_ptr, lo_az, lo_bz, lo_ay, lo_by, lo_ax, lo_bx, loZ, loY, loX, lo_scale,
                hi_ptr, hi_az, hi_bz, hi_ay, hi_by, hi_ax, hi_bx, hiZ, hiY, hiX, hi_scale, m)
            x4z = yz + h * k3z
            x4y = yy + h * k3y
            x4x = yx + h * k3x
            if STORE_STAGES:
                s = (step * 4 + 3) * N.to(tl.int64)
                tl.store(stages_ptr + (s + i) * 3 + 0, x4z, mask=m)
                tl.store(stages_ptr + (s + i) * 3 + 1, x4y, mask=m)
                tl.store(stages_ptr + (s + i) * 3 + 2, x4x, mask=m)
            k4z, k4y, k4x = _sample_pair(
                x4z, x4y, x4x,
                lo_ptr, lo_az, lo_bz, lo_ay, lo_by, lo_ax, lo_bx, loZ, loY, loX, lo_scale,
                hi_ptr, hi_az, hi_bz, hi_ay, hi_by, hi_ax, hi_bx, hiZ, hiY, hiX, hi_scale, m)
            yz = yz + h_sixth * (((k1z + 2.0 * k2z) + 2.0 * k3z) + k4z)
            yy = yy + h_sixth * (((k1y + 2.0 * k2y) + 2.0 * k3y) + k4y)
            yx = yx + h_sixth * (((k1x + 2.0 * k2x) + 2.0 * k3x) + k4x)
        tl.store(out_ptr + i * 3 + 0, yz, mask=m)
        tl.store(out_ptr + i * 3 + 1, yy, mask=m)
        tl.store(out_ptr + i * 3 + 2, yx, mask=m)

    @triton.jit
    def _bwd_one(gz, gy, gx, pz, py, px, ptr, acc_ptr, vscale,
                 az, bz, ay, by, ax, bx, Z, Y, X,
                 HAS_ACC: tl.constexpr, lane_mask):
        # Sampler backward for one field: scatter dL/d(field values) into
        # acc (unscaled -- the caller applies the field's value scale to the
        # whole accumulator afterwards) and return dL/d(point).
        zm1f = (Z - 1).to(tl.float32)
        ym1f = (Y - 1).to(tl.float32)
        xm1f = (X - 1).to(tl.float32)
        zm2f = (Z - 2).to(tl.float32)
        ym2f = (Y - 2).to(tl.float32)
        xm2f = (X - 2).to(tl.float32)
        crz = pz * az + bz
        cry = py * ay + by
        crx = px * ax + bx
        cz = tl.minimum(tl.maximum(crz, 0.0), zm1f)
        cy = tl.minimum(tl.maximum(cry, 0.0), ym1f)
        cx = tl.minimum(tl.maximum(crx, 0.0), xm1f)
        cz = tl.where(crz != crz, crz, cz)
        cy = tl.where(cry != cry, cry, cy)
        cx = tl.where(crx != crx, crx, cx)
        lz = tl.minimum(tl.maximum(tl.math.floor(tl.where(cz != cz, 0.0, cz)), 0.0), zm2f)
        ly = tl.minimum(tl.maximum(tl.math.floor(tl.where(cy != cy, 0.0, cy)), 0.0), ym2f)
        lx = tl.minimum(tl.maximum(tl.math.floor(tl.where(cx != cx, 0.0, cx)), 0.0), xm2f)
        fz1 = cz - lz
        fy1 = cy - ly
        fx1 = cx - lx
        fz0 = 1.0 - fz1
        fy0 = 1.0 - fy1
        fx0 = 1.0 - fx1
        z0 = lz.to(tl.int32)
        y0 = ly.to(tl.int32)
        x0 = lx.to(tl.int32)
        ch = Z.to(tl.int64) * Y * X

        gcz = tl.zeros(pz.shape, dtype=tl.float32)
        gcy = tl.zeros(pz.shape, dtype=tl.float32)
        gcx = tl.zeros(pz.shape, dtype=tl.float32)
        for dz in tl.static_range(2):
            for dy in tl.static_range(2):
                for dx in tl.static_range(2):
                    fz = fz1 if dz == 1 else fz0
                    fy = fy1 if dy == 1 else fy0
                    fx = fx1 if dx == 1 else fx0
                    w = (fz * fy) * fx
                    idx = ((z0 + dz).to(tl.int64) * Y + (y0 + dy)) * X + (x0 + dx)
                    if HAS_ACC:
                        tl.atomic_add(acc_ptr + idx, gz * w, mask=lane_mask)
                        tl.atomic_add(acc_ptr + ch + idx, gy * w, mask=lane_mask)
                        tl.atomic_add(acc_ptr + 2 * ch + idx, gx * w, mask=lane_mask)
                    v0 = tl.load(ptr + idx, mask=lane_mask, other=0.0)
                    v1 = tl.load(ptr + ch + idx, mask=lane_mask, other=0.0)
                    v2 = tl.load(ptr + 2 * ch + idx, mask=lane_mask, other=0.0)
                    vdg = ((v0 * gz + v1 * gy) + v2 * gx) * vscale
                    sz = 1.0 if dz == 1 else -1.0
                    sy = 1.0 if dy == 1 else -1.0
                    sx = 1.0 if dx == 1 else -1.0
                    gcz += ((vdg * sz) * fy) * fx
                    gcy += ((vdg * sy) * fz) * fx
                    gcx += ((vdg * sx) * fz) * fy
        mz = ((crz >= 0.0) & (crz <= zm1f)).to(tl.float32)
        my = ((cry >= 0.0) & (cry <= ym1f)).to(tl.float32)
        mx = ((crx >= 0.0) & (crx <= xm1f)).to(tl.float32)
        return (gcz * mz) * az, (gcy * my) * ay, (gcx * mx) * ax

    @triton.jit
    def _bwd_stage_pair(gz, gy, gx, pz, py, px,
                        lo_ptr, acc_lo_ptr, lo_az, lo_bz, lo_ay, lo_by, lo_ax, lo_bx,
                        loZ, loY, loX, lo_scale,
                        hi_ptr, acc_hi_ptr, hi_az, hi_bz, hi_ay, hi_by, hi_ax, hi_bx,
                        hiZ, hiY, hiX, hi_scale,
                        HAS_ACC: tl.constexpr, lane_mask):
        lz, ly, lx = _bwd_one(gz, gy, gx, pz, py, px, lo_ptr, acc_lo_ptr, lo_scale,
                              lo_az, lo_bz, lo_ay, lo_by, lo_ax, lo_bx,
                              loZ, loY, loX, HAS_ACC, lane_mask)
        hz, hy, hx = _bwd_one(gz, gy, gx, pz, py, px, hi_ptr, acc_hi_ptr, hi_scale,
                              hi_az, hi_bz, hi_ay, hi_by, hi_ax, hi_bx,
                              hiZ, hiY, hiX, HAS_ACC, lane_mask)
        return lz + hz, ly + hy, lx + hx

    @triton.jit
    def _rk4d_bwd_kernel(grad_y_ptr, grad_pts_ptr, stages_ptr,
                         lo_base, acc_lo_base, lo_az, lo_bz, lo_ay, lo_by, lo_ax, lo_bx,
                         loZ, loY, loX, lo_scale,
                         hi_base, acc_hi_base, hi_az, hi_bz, hi_ay, hi_by, hi_ax, hi_bx,
                         hiZ, hiY, hiX, hi_scale,
                         N, h, h_half, h_sixth, n_steps, num_slabs,
                         REVERSE: tl.constexpr,
                         HAS_ACC: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        i = pid * BLOCK + tl.arange(0, BLOCK)
        m = i < N
        lo_stride = loZ.to(tl.int64) * loY * loX * 3
        hi_stride = hiZ.to(tl.int64) * hiY * hiX * 3
        gz = tl.load(grad_y_ptr + i * 3 + 0, mask=m, other=0.0)
        gy = tl.load(grad_y_ptr + i * 3 + 1, mask=m, other=0.0)
        gx = tl.load(grad_y_ptr + i * 3 + 2, mask=m, other=0.0)
        for step in range(n_steps * num_slabs - 1, -1, -1):
            slab = step // n_steps
            if REVERSE:
                slab = num_slabs - 1 - slab
            lo_off = slab.to(tl.int64) * lo_stride
            hi_off = slab.to(tl.int64) * hi_stride
            lo_ptr = lo_base + lo_off
            hi_ptr = hi_base + hi_off
            acc_lo_ptr = acc_lo_base + lo_off
            acc_hi_ptr = acc_hi_base + hi_off
            s1 = (step * 4 + 0) * N.to(tl.int64)
            s2 = (step * 4 + 1) * N.to(tl.int64)
            s3 = (step * 4 + 2) * N.to(tl.int64)
            s4 = (step * 4 + 3) * N.to(tl.int64)
            g6z = gz * h_sixth
            g6y = gy * h_sixth
            g6x = gx * h_sixth
            pz = tl.load(stages_ptr + (s4 + i) * 3 + 0, mask=m, other=0.0)
            py = tl.load(stages_ptr + (s4 + i) * 3 + 1, mask=m, other=0.0)
            px = tl.load(stages_ptr + (s4 + i) * 3 + 2, mask=m, other=0.0)
            b4z, b4y, b4x = _bwd_stage_pair(
                g6z, g6y, g6x, pz, py, px,
                lo_ptr, acc_lo_ptr, lo_az, lo_bz, lo_ay, lo_by, lo_ax, lo_bx, loZ, loY, loX, lo_scale,
                hi_ptr, acc_hi_ptr, hi_az, hi_bz, hi_ay, hi_by, hi_ax, hi_bx, hiZ, hiY, hiX, hi_scale,
                HAS_ACC, m)
            pz = tl.load(stages_ptr + (s3 + i) * 3 + 0, mask=m, other=0.0)
            py = tl.load(stages_ptr + (s3 + i) * 3 + 1, mask=m, other=0.0)
            px = tl.load(stages_ptr + (s3 + i) * 3 + 2, mask=m, other=0.0)
            b3z, b3y, b3x = _bwd_stage_pair(
                g6z * 2.0 + b4z * h, g6y * 2.0 + b4y * h, g6x * 2.0 + b4x * h,
                pz, py, px,
                lo_ptr, acc_lo_ptr, lo_az, lo_bz, lo_ay, lo_by, lo_ax, lo_bx, loZ, loY, loX, lo_scale,
                hi_ptr, acc_hi_ptr, hi_az, hi_bz, hi_ay, hi_by, hi_ax, hi_bx, hiZ, hiY, hiX, hi_scale,
                HAS_ACC, m)
            pz = tl.load(stages_ptr + (s2 + i) * 3 + 0, mask=m, other=0.0)
            py = tl.load(stages_ptr + (s2 + i) * 3 + 1, mask=m, other=0.0)
            px = tl.load(stages_ptr + (s2 + i) * 3 + 2, mask=m, other=0.0)
            b2z, b2y, b2x = _bwd_stage_pair(
                g6z * 2.0 + b3z * h_half, g6y * 2.0 + b3y * h_half, g6x * 2.0 + b3x * h_half,
                pz, py, px,
                lo_ptr, acc_lo_ptr, lo_az, lo_bz, lo_ay, lo_by, lo_ax, lo_bx, loZ, loY, loX, lo_scale,
                hi_ptr, acc_hi_ptr, hi_az, hi_bz, hi_ay, hi_by, hi_ax, hi_bx, hiZ, hiY, hiX, hi_scale,
                HAS_ACC, m)
            pz = tl.load(stages_ptr + (s1 + i) * 3 + 0, mask=m, other=0.0)
            py = tl.load(stages_ptr + (s1 + i) * 3 + 1, mask=m, other=0.0)
            px = tl.load(stages_ptr + (s1 + i) * 3 + 2, mask=m, other=0.0)
            b1z, b1y, b1x = _bwd_stage_pair(
                g6z + b2z * h_half, g6y + b2y * h_half, g6x + b2x * h_half,
                pz, py, px,
                lo_ptr, acc_lo_ptr, lo_az, lo_bz, lo_ay, lo_by, lo_ax, lo_bx, loZ, loY, loX, lo_scale,
                hi_ptr, acc_hi_ptr, hi_az, hi_bz, hi_ay, hi_by, hi_ax, hi_bx, hiZ, hiY, hiX, hi_scale,
                HAS_ACC, m)
            gz = ((gz + b4z) + b3z + b2z) + b1z
            gy = ((gy + b4y) + b3y + b2y) + b1y
            gx = ((gx + b4x) + b3x + b2x) + b1x
        tl.store(grad_pts_ptr + i * 3 + 0, gz, mask=m)
        tl.store(grad_pts_ptr + i * 3 + 1, gy, mask=m)
        tl.store(grad_pts_ptr + i * 3 + 2, gx, mask=m)


if _HAS_TRITON:

    @triton.jit
    def _flush_cell(acc_ptr, ch, Y, X, cz, cy, cx, mask,
                    rz0, rz1, rz2, rz3, rz4, rz5, rz6, rz7,
                    ry0, ry1, ry2, ry3, ry4, ry5, ry6, ry7,
                    rx0, rx1, rx2, rx3, rx4, rx5, rx6, rx7):
        # Fire the deferred per-corner sums for the held cell. Corner order
        # matches _bwd_stage's (dz, dy, dx) loop; channels are z / y / x at
        # offsets 0 / ch / 2ch, exactly like the per-stage atomics.
        base = (cz.to(tl.int64) * Y + cy) * X + cx
        i1 = base + 1
        i2 = base + X
        i3 = base + X + 1
        i4 = base + Y * X
        i5 = base + Y * X + 1
        i6 = base + (Y + 1) * X
        i7 = base + (Y + 1) * X + 1
        tl.atomic_add(acc_ptr + base, rz0, mask=mask)
        tl.atomic_add(acc_ptr + i1, rz1, mask=mask)
        tl.atomic_add(acc_ptr + i2, rz2, mask=mask)
        tl.atomic_add(acc_ptr + i3, rz3, mask=mask)
        tl.atomic_add(acc_ptr + i4, rz4, mask=mask)
        tl.atomic_add(acc_ptr + i5, rz5, mask=mask)
        tl.atomic_add(acc_ptr + i6, rz6, mask=mask)
        tl.atomic_add(acc_ptr + i7, rz7, mask=mask)
        tl.atomic_add(acc_ptr + ch + base, ry0, mask=mask)
        tl.atomic_add(acc_ptr + ch + i1, ry1, mask=mask)
        tl.atomic_add(acc_ptr + ch + i2, ry2, mask=mask)
        tl.atomic_add(acc_ptr + ch + i3, ry3, mask=mask)
        tl.atomic_add(acc_ptr + ch + i4, ry4, mask=mask)
        tl.atomic_add(acc_ptr + ch + i5, ry5, mask=mask)
        tl.atomic_add(acc_ptr + ch + i6, ry6, mask=mask)
        tl.atomic_add(acc_ptr + ch + i7, ry7, mask=mask)
        tl.atomic_add(acc_ptr + 2 * ch + base, rx0, mask=mask)
        tl.atomic_add(acc_ptr + 2 * ch + i1, rx1, mask=mask)
        tl.atomic_add(acc_ptr + 2 * ch + i2, rx2, mask=mask)
        tl.atomic_add(acc_ptr + 2 * ch + i3, rx3, mask=mask)
        tl.atomic_add(acc_ptr + 2 * ch + i4, rx4, mask=mask)
        tl.atomic_add(acc_ptr + 2 * ch + i5, rx5, mask=mask)
        tl.atomic_add(acc_ptr + 2 * ch + i6, rx6, mask=mask)
        tl.atomic_add(acc_ptr + 2 * ch + i7, rx7, mask=mask)

    @triton.jit
    def _defer_accumulate(acc_ptr, ch, Y, X, m,
                          gsz, gsy, gsx, z0, y0, x0,
                          fz0, fz1, fy0, fy1, fx0, fx1,
                          have, acz, acy, acx,
                          rz0, rz1, rz2, rz3, rz4, rz5, rz6, rz7,
                          ry0, ry1, ry2, ry3, ry4, ry5, ry6, ry7,
                          rx0, rx1, rx2, rx3, rx4, rx5, rx6, rx7):
        # Add one stage's field-gradient contribution to the register
        # accumulators, flushing lanes whose stage point left the held cell.
        moved = have & ((z0 != acz) | (y0 != acy) | (x0 != acx))
        _flush_cell(acc_ptr, ch, Y, X, acz, acy, acx, m & moved,
                    rz0, rz1, rz2, rz3, rz4, rz5, rz6, rz7,
                    ry0, ry1, ry2, ry3, ry4, ry5, ry6, ry7,
                    rx0, rx1, rx2, rx3, rx4, rx5, rx6, rx7)
        fresh = moved | (~have)
        zero = tl.zeros(gsz.shape, dtype=tl.float32)
        rz0 = tl.where(fresh, zero, rz0)
        rz1 = tl.where(fresh, zero, rz1)
        rz2 = tl.where(fresh, zero, rz2)
        rz3 = tl.where(fresh, zero, rz3)
        rz4 = tl.where(fresh, zero, rz4)
        rz5 = tl.where(fresh, zero, rz5)
        rz6 = tl.where(fresh, zero, rz6)
        rz7 = tl.where(fresh, zero, rz7)
        ry0 = tl.where(fresh, zero, ry0)
        ry1 = tl.where(fresh, zero, ry1)
        ry2 = tl.where(fresh, zero, ry2)
        ry3 = tl.where(fresh, zero, ry3)
        ry4 = tl.where(fresh, zero, ry4)
        ry5 = tl.where(fresh, zero, ry5)
        ry6 = tl.where(fresh, zero, ry6)
        ry7 = tl.where(fresh, zero, ry7)
        rx0 = tl.where(fresh, zero, rx0)
        rx1 = tl.where(fresh, zero, rx1)
        rx2 = tl.where(fresh, zero, rx2)
        rx3 = tl.where(fresh, zero, rx3)
        rx4 = tl.where(fresh, zero, rx4)
        rx5 = tl.where(fresh, zero, rx5)
        rx6 = tl.where(fresh, zero, rx6)
        rx7 = tl.where(fresh, zero, rx7)
        acz = tl.where(fresh, z0, acz)
        acy = tl.where(fresh, y0, acy)
        acx = tl.where(fresh, x0, acx)
        have = have | m
        w0 = (fz0 * fy0) * fx0
        w1 = (fz0 * fy0) * fx1
        w2 = (fz0 * fy1) * fx0
        w3 = (fz0 * fy1) * fx1
        w4 = (fz1 * fy0) * fx0
        w5 = (fz1 * fy0) * fx1
        w6 = (fz1 * fy1) * fx0
        w7 = (fz1 * fy1) * fx1
        rz0 += gsz * w0
        rz1 += gsz * w1
        rz2 += gsz * w2
        rz3 += gsz * w3
        rz4 += gsz * w4
        rz5 += gsz * w5
        rz6 += gsz * w6
        rz7 += gsz * w7
        ry0 += gsy * w0
        ry1 += gsy * w1
        ry2 += gsy * w2
        ry3 += gsy * w3
        ry4 += gsy * w4
        ry5 += gsy * w5
        ry6 += gsy * w6
        ry7 += gsy * w7
        rx0 += gsx * w0
        rx1 += gsx * w1
        rx2 += gsx * w2
        rx3 += gsx * w3
        rx4 += gsx * w4
        rx5 += gsx * w5
        rx6 += gsx * w6
        rx7 += gsx * w7
        return (have, acz, acy, acx,
                rz0, rz1, rz2, rz3, rz4, rz5, rz6, rz7,
                ry0, ry1, ry2, ry3, ry4, ry5, ry6, ry7,
                rx0, rx1, rx2, rx3, rx4, rx5, rx6, rx7)

    @triton.jit
    def _rk4_bwd_coalesced_kernel(grad_y_ptr, grad_pts_ptr, stages_ptr,
                                  low_base, high_base, acc_base, scale,
                                  N, Z, Y, X, zm1f, ym1f, xm1f,
                                  zm2f, ym2f, xm2f,
                                  h, h_half, h_sixth, n_steps, num_slabs,
                                  REVERSE: tl.constexpr,
                                  BLOCK: tl.constexpr):
        # _rk4_bwd_kernel with the field-gradient atomics deferred through
        # register accumulators: RK4 stage points move slowly relative to the
        # lattice cell, so most consecutive stages hit the same 8 corners and
        # one flush replaces up to 4*n_steps per-stage atomic bursts. The
        # accumulated VALUE is the same set of addends in a different
        # association order — inside the atomics-order tolerance the
        # accumulator already has. The point gradients are untouched
        # (bitwise identical to _rk4_bwd_kernel).
        pid = tl.program_id(0)
        i = pid * BLOCK + tl.arange(0, BLOCK)
        m = i < N
        ch = tl.full((), Z, tl.int64) * Y * X
        slab_stride = ch * 3
        gz = tl.load(grad_y_ptr + i * 3 + 0, mask=m, other=0.0)
        gy = tl.load(grad_y_ptr + i * 3 + 1, mask=m, other=0.0)
        gx = tl.load(grad_y_ptr + i * 3 + 2, mask=m, other=0.0)
        have = tl.zeros(i.shape, dtype=tl.int1)
        acz = tl.zeros(i.shape, dtype=tl.int32)
        acy = tl.zeros(i.shape, dtype=tl.int32)
        acx = tl.zeros(i.shape, dtype=tl.int32)
        rz0 = tl.zeros(i.shape, dtype=tl.float32)
        rz1 = tl.zeros(i.shape, dtype=tl.float32)
        rz2 = tl.zeros(i.shape, dtype=tl.float32)
        rz3 = tl.zeros(i.shape, dtype=tl.float32)
        rz4 = tl.zeros(i.shape, dtype=tl.float32)
        rz5 = tl.zeros(i.shape, dtype=tl.float32)
        rz6 = tl.zeros(i.shape, dtype=tl.float32)
        rz7 = tl.zeros(i.shape, dtype=tl.float32)
        ry0 = tl.zeros(i.shape, dtype=tl.float32)
        ry1 = tl.zeros(i.shape, dtype=tl.float32)
        ry2 = tl.zeros(i.shape, dtype=tl.float32)
        ry3 = tl.zeros(i.shape, dtype=tl.float32)
        ry4 = tl.zeros(i.shape, dtype=tl.float32)
        ry5 = tl.zeros(i.shape, dtype=tl.float32)
        ry6 = tl.zeros(i.shape, dtype=tl.float32)
        ry7 = tl.zeros(i.shape, dtype=tl.float32)
        rx0 = tl.zeros(i.shape, dtype=tl.float32)
        rx1 = tl.zeros(i.shape, dtype=tl.float32)
        rx2 = tl.zeros(i.shape, dtype=tl.float32)
        rx3 = tl.zeros(i.shape, dtype=tl.float32)
        rx4 = tl.zeros(i.shape, dtype=tl.float32)
        rx5 = tl.zeros(i.shape, dtype=tl.float32)
        rx6 = tl.zeros(i.shape, dtype=tl.float32)
        rx7 = tl.zeros(i.shape, dtype=tl.float32)
        for step in range(n_steps * num_slabs - 1, -1, -1):
            slab = step // n_steps
            if REVERSE:
                slab = num_slabs - 1 - slab
            slab_off = slab.to(tl.int64) * slab_stride
            low_ptr = low_base + slab_off
            high_ptr = high_base + slab_off
            acc_ptr = acc_base + slab_off
            s1 = (step * 4 + 0) * N.to(tl.int64)
            s2 = (step * 4 + 1) * N.to(tl.int64)
            s3 = (step * 4 + 2) * N.to(tl.int64)
            s4 = (step * 4 + 3) * N.to(tl.int64)
            g6z = gz * h_sixth
            g6y = gy * h_sixth
            g6x = gx * h_sixth
            pz = tl.load(stages_ptr + (s4 + i) * 3 + 0, mask=m, other=0.0)
            py = tl.load(stages_ptr + (s4 + i) * 3 + 1, mask=m, other=0.0)
            px = tl.load(stages_ptr + (s4 + i) * 3 + 2, mask=m, other=0.0)
            (b4z, b4y, b4x, z0, y0, x0,
             fz0, fz1, fy0, fy1, fx0, fx1) = _bwd_stage_defer(
                g6z, g6y, g6x, pz, py, px, low_ptr, high_ptr, scale,
                Z, Y, X, zm1f, ym1f, xm1f, zm2f, ym2f, xm2f, ch, m)
            (have, acz, acy, acx,
             rz0, rz1, rz2, rz3, rz4, rz5, rz6, rz7,
             ry0, ry1, ry2, ry3, ry4, ry5, ry6, ry7,
             rx0, rx1, rx2, rx3, rx4, rx5, rx6, rx7) = _defer_accumulate(
                acc_ptr, ch, Y, X, m, g6z, g6y, g6x, z0, y0, x0,
                fz0, fz1, fy0, fy1, fx0, fx1, have, acz, acy, acx,
                rz0, rz1, rz2, rz3, rz4, rz5, rz6, rz7,
                ry0, ry1, ry2, ry3, ry4, ry5, ry6, ry7,
                rx0, rx1, rx2, rx3, rx4, rx5, rx6, rx7)
            g3z = g6z * 2.0 + b4z * h
            g3y = g6y * 2.0 + b4y * h
            g3x = g6x * 2.0 + b4x * h
            pz = tl.load(stages_ptr + (s3 + i) * 3 + 0, mask=m, other=0.0)
            py = tl.load(stages_ptr + (s3 + i) * 3 + 1, mask=m, other=0.0)
            px = tl.load(stages_ptr + (s3 + i) * 3 + 2, mask=m, other=0.0)
            (b3z, b3y, b3x, z0, y0, x0,
             fz0, fz1, fy0, fy1, fx0, fx1) = _bwd_stage_defer(
                g3z, g3y, g3x, pz, py, px, low_ptr, high_ptr, scale,
                Z, Y, X, zm1f, ym1f, xm1f, zm2f, ym2f, xm2f, ch, m)
            (have, acz, acy, acx,
             rz0, rz1, rz2, rz3, rz4, rz5, rz6, rz7,
             ry0, ry1, ry2, ry3, ry4, ry5, ry6, ry7,
             rx0, rx1, rx2, rx3, rx4, rx5, rx6, rx7) = _defer_accumulate(
                acc_ptr, ch, Y, X, m, g3z, g3y, g3x, z0, y0, x0,
                fz0, fz1, fy0, fy1, fx0, fx1, have, acz, acy, acx,
                rz0, rz1, rz2, rz3, rz4, rz5, rz6, rz7,
                ry0, ry1, ry2, ry3, ry4, ry5, ry6, ry7,
                rx0, rx1, rx2, rx3, rx4, rx5, rx6, rx7)
            g2z = g6z * 2.0 + b3z * h_half
            g2y = g6y * 2.0 + b3y * h_half
            g2x = g6x * 2.0 + b3x * h_half
            pz = tl.load(stages_ptr + (s2 + i) * 3 + 0, mask=m, other=0.0)
            py = tl.load(stages_ptr + (s2 + i) * 3 + 1, mask=m, other=0.0)
            px = tl.load(stages_ptr + (s2 + i) * 3 + 2, mask=m, other=0.0)
            (b2z, b2y, b2x, z0, y0, x0,
             fz0, fz1, fy0, fy1, fx0, fx1) = _bwd_stage_defer(
                g2z, g2y, g2x, pz, py, px, low_ptr, high_ptr, scale,
                Z, Y, X, zm1f, ym1f, xm1f, zm2f, ym2f, xm2f, ch, m)
            (have, acz, acy, acx,
             rz0, rz1, rz2, rz3, rz4, rz5, rz6, rz7,
             ry0, ry1, ry2, ry3, ry4, ry5, ry6, ry7,
             rx0, rx1, rx2, rx3, rx4, rx5, rx6, rx7) = _defer_accumulate(
                acc_ptr, ch, Y, X, m, g2z, g2y, g2x, z0, y0, x0,
                fz0, fz1, fy0, fy1, fx0, fx1, have, acz, acy, acx,
                rz0, rz1, rz2, rz3, rz4, rz5, rz6, rz7,
                ry0, ry1, ry2, ry3, ry4, ry5, ry6, ry7,
                rx0, rx1, rx2, rx3, rx4, rx5, rx6, rx7)
            g1z = g6z + b2z * h_half
            g1y = g6y + b2y * h_half
            g1x = g6x + b2x * h_half
            pz = tl.load(stages_ptr + (s1 + i) * 3 + 0, mask=m, other=0.0)
            py = tl.load(stages_ptr + (s1 + i) * 3 + 1, mask=m, other=0.0)
            px = tl.load(stages_ptr + (s1 + i) * 3 + 2, mask=m, other=0.0)
            (b1z, b1y, b1x, z0, y0, x0,
             fz0, fz1, fy0, fy1, fx0, fx1) = _bwd_stage_defer(
                g1z, g1y, g1x, pz, py, px, low_ptr, high_ptr, scale,
                Z, Y, X, zm1f, ym1f, xm1f, zm2f, ym2f, xm2f, ch, m)
            (have, acz, acy, acx,
             rz0, rz1, rz2, rz3, rz4, rz5, rz6, rz7,
             ry0, ry1, ry2, ry3, ry4, ry5, ry6, ry7,
             rx0, rx1, rx2, rx3, rx4, rx5, rx6, rx7) = _defer_accumulate(
                acc_ptr, ch, Y, X, m, g1z, g1y, g1x, z0, y0, x0,
                fz0, fz1, fy0, fy1, fx0, fx1, have, acz, acy, acx,
                rz0, rz1, rz2, rz3, rz4, rz5, rz6, rz7,
                ry0, ry1, ry2, ry3, ry4, ry5, ry6, ry7,
                rx0, rx1, rx2, rx3, rx4, rx5, rx6, rx7)
            gz = ((gz + b4z) + b3z + b2z) + b1z
            gy = ((gy + b4y) + b3y + b2y) + b1y
            gx = ((gx + b4x) + b3x + b2x) + b1x
            # The held sums belong to this slab's lattice. The reverse walk
            # finishes a slab at its first step, so flush there (this also
            # covers the very last step of the walk); the next slab's stage
            # then starts a fresh cell.
            slab_done = (step % n_steps) == 0
            _flush_cell(acc_ptr, ch, Y, X, acz, acy, acx, m & have & slab_done,
                        rz0, rz1, rz2, rz3, rz4, rz5, rz6, rz7,
                        ry0, ry1, ry2, ry3, ry4, ry5, ry6, ry7,
                        rx0, rx1, rx2, rx3, rx4, rx5, rx6, rx7)
            have = have & ((step % n_steps) != 0)
        tl.store(grad_pts_ptr + i * 3 + 0, gz, mask=m)
        tl.store(grad_pts_ptr + i * 3 + 1, gy, mask=m)
        tl.store(grad_pts_ptr + i * 3 + 2, gx, mask=m)


if _HAS_TRITON:

    # ---- stationary cylindrical mode: two packed ragged lattices ----

    _TWO_PI_F32 = tl.constexpr(6.2831854820251465)
    _AXIS_EPS_F32 = tl.constexpr(1.1920928955078125e-07)

    @triton.jit
    def _cyl_coords(py, px):
        qy = py * 2.0 - 1.0
        qx = px * 2.0 - 1.0
        on_axis = (tl.abs(qy) <= _AXIS_EPS_F32) & (tl.abs(qx) <= _AXIS_EPS_F32)
        safe_y = tl.where(on_axis, 0.0, qy)
        safe_x = tl.where(on_axis, 1.0, qx)
        raw_radius = tl.sqrt(safe_y * safe_y + safe_x * safe_x)
        radius = tl.minimum(raw_radius, 1.0)
        radius = tl.where(on_axis, 0.0, radius)
        phi = libdevice.atan2(safe_y, safe_x)
        # This deliberately keeps negative signed zero unchanged, matching
        # torch.remainder(atan2(-0, +x), 2*pi).
        phi = tl.where(phi < 0.0, phi + _TWO_PI_F32, phi)
        sin_phi = safe_y / raw_radius
        cos_phi = safe_x / raw_radius
        return qy, qx, raw_radius, radius, phi, sin_phi, cos_phi, on_axis

    @triton.jit
    def _cyl_sample_lattice(pz, radius, phi, field_ptr, num_phi_ptr,
                            offsets_ptr, NZ, NR, TOTAL, lane_mask):
        zn = pz * 2.0 - 1.0
        zc = ((zn + 1.0) * 0.5) * (NZ - 1).to(tl.float32)
        zc = tl.minimum(tl.maximum(zc, 0.0), (NZ - 1).to(tl.float32))
        z0f = tl.minimum(tl.math.floor(zc), (NZ - 2).to(tl.float32))
        z0 = z0f.to(tl.int32)
        fz1 = zc - z0f
        fz0 = 1.0 - fz1

        rc = radius * (NR - 1).to(tl.float32)
        r0f = tl.minimum(tl.math.floor(rc), (NR - 2).to(tl.float32))
        r0 = r0f.to(tl.int32)
        fr1 = rc - r0f
        fr0 = 1.0 - fr1
        ch = NZ.to(tl.int64) * TOTAL
        v0 = tl.zeros(pz.shape, dtype=tl.float32)
        v1 = tl.zeros(pz.shape, dtype=tl.float32)
        v2 = tl.zeros(pz.shape, dtype=tl.float32)
        for dr in tl.static_range(2):
            ring = r0 + dr
            nphi = tl.load(num_phi_ptr + ring, mask=lane_mask, other=1).to(tl.int32)
            offset = tl.load(offsets_ptr + ring, mask=lane_mask, other=0).to(tl.int64)
            pc = phi * (nphi.to(tl.float32) / _TWO_PI_F32)
            p0f = tl.math.floor(pc)
            p0 = p0f.to(tl.int32)
            p0 = tl.where(p0 >= nphi, p0 - nphi, p0)
            p1 = tl.where(p0 + 1 >= nphi, 0, p0 + 1)
            fp1 = pc - p0f
            fp0 = 1.0 - fp1
            wr = fr1 if dr == 1 else fr0
            load_mask = lane_mask & (ring != 0)
            for dz in tl.static_range(2):
                z = z0 + dz
                wz = fz1 if dz == 1 else fz0
                for dp in tl.static_range(2):
                    pp = p1 if dp == 1 else p0
                    wp = fp1 if dp == 1 else fp0
                    w = (wr * wz) * wp
                    idx = z.to(tl.int64) * TOTAL + offset + pp
                    v0 += tl.load(field_ptr + idx, mask=load_mask, other=0.0) * w
                    v1 += tl.load(field_ptr + ch + idx, mask=load_mask, other=0.0) * w
                    v2 += tl.load(field_ptr + 2 * ch + idx, mask=load_mask, other=0.0) * w
        return v0, v1, v2

    @triton.jit
    def _cyl_sample_pair(pz, py, px,
                         lo_ptr, lo_num_phi_ptr, lo_offsets_ptr,
                         loNZ, loNR, loTOTAL,
                         hi_ptr, hi_num_phi_ptr, hi_offsets_ptr,
                         hiNZ, hiNR, hiTOTAL, CUBIC: tl.constexpr, lane_mask):
        # CUBIC selects the tricubic B-spline interpolant (_cylb_sample_lattice,
        # BSplineCylindricalFlowField) over the trilinear one; it is a
        # compile-time constant, so each variant is its own kernel.
        qy, qx, raw, radius, phi, sin_phi, cos_phi, on_axis = _cyl_coords(py, px)
        if CUBIC:
            lz, lr, lp = _cylb_sample_lattice(
                pz, radius, phi, lo_ptr, lo_num_phi_ptr, lo_offsets_ptr,
                loNZ, loNR, loTOTAL, lane_mask)
            hz, hr, hp = _cylb_sample_lattice(
                pz, radius, phi, hi_ptr, hi_num_phi_ptr, hi_offsets_ptr,
                hiNZ, hiNR, hiTOTAL, lane_mask)
        else:
            lz, lr, lp = _cyl_sample_lattice(
                pz, radius, phi, lo_ptr, lo_num_phi_ptr, lo_offsets_ptr,
                loNZ, loNR, loTOTAL, lane_mask)
            hz, hr, hp = _cyl_sample_lattice(
                pz, radius, phi, hi_ptr, hi_num_phi_ptr, hi_offsets_ptr,
                hiNZ, hiNR, hiTOTAL, lane_mask)
        vz = lz + hz
        vr = lr + hr
        vp = lp + hp
        return (vz,
                vr * sin_phi + vp * cos_phi,
                vr * cos_phi - vp * sin_phi)

    @triton.jit
    def _rk4c_fwd_kernel(y_ptr, out_ptr, stages_ptr,
                         lo_base, lo_num_phi_ptr, lo_offsets_ptr,
                         loNZ, loNR, loTOTAL,
                         hi_base, hi_num_phi_ptr, hi_offsets_ptr,
                         hiNZ, hiNR, hiTOTAL,
                         N, h, h_half, h_sixth, n_steps, num_slabs,
                         REVERSE: tl.constexpr, CUBIC: tl.constexpr,
                         STORE_STAGES: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        i = pid * BLOCK + tl.arange(0, BLOCK)
        m = i < N
        lo_stride = loNZ.to(tl.int64) * loTOTAL * 3
        hi_stride = hiNZ.to(tl.int64) * hiTOTAL * 3
        yz = tl.load(y_ptr + i * 3, mask=m, other=0.0)
        yy = tl.load(y_ptr + i * 3 + 1, mask=m, other=0.0)
        yx = tl.load(y_ptr + i * 3 + 2, mask=m, other=0.0)
        for step in range(n_steps * num_slabs):
            slab = step // n_steps
            if REVERSE:
                slab = num_slabs - 1 - slab
            lo_ptr = lo_base + slab.to(tl.int64) * lo_stride
            hi_ptr = hi_base + slab.to(tl.int64) * hi_stride
            if STORE_STAGES:
                s = (step * 4) * N.to(tl.int64)
                tl.store(stages_ptr + (s + i) * 3, yz, mask=m)
                tl.store(stages_ptr + (s + i) * 3 + 1, yy, mask=m)
                tl.store(stages_ptr + (s + i) * 3 + 2, yx, mask=m)
            k1z, k1y, k1x = _cyl_sample_pair(
                yz, yy, yx,
                lo_ptr, lo_num_phi_ptr, lo_offsets_ptr, loNZ, loNR, loTOTAL,
                hi_ptr, hi_num_phi_ptr, hi_offsets_ptr, hiNZ, hiNR, hiTOTAL, CUBIC, m)
            x2z, x2y, x2x = yz + h_half * k1z, yy + h_half * k1y, yx + h_half * k1x
            if STORE_STAGES:
                s = (step * 4 + 1) * N.to(tl.int64)
                tl.store(stages_ptr + (s + i) * 3, x2z, mask=m)
                tl.store(stages_ptr + (s + i) * 3 + 1, x2y, mask=m)
                tl.store(stages_ptr + (s + i) * 3 + 2, x2x, mask=m)
            k2z, k2y, k2x = _cyl_sample_pair(
                x2z, x2y, x2x,
                lo_ptr, lo_num_phi_ptr, lo_offsets_ptr, loNZ, loNR, loTOTAL,
                hi_ptr, hi_num_phi_ptr, hi_offsets_ptr, hiNZ, hiNR, hiTOTAL, CUBIC, m)
            x3z, x3y, x3x = yz + h_half * k2z, yy + h_half * k2y, yx + h_half * k2x
            if STORE_STAGES:
                s = (step * 4 + 2) * N.to(tl.int64)
                tl.store(stages_ptr + (s + i) * 3, x3z, mask=m)
                tl.store(stages_ptr + (s + i) * 3 + 1, x3y, mask=m)
                tl.store(stages_ptr + (s + i) * 3 + 2, x3x, mask=m)
            k3z, k3y, k3x = _cyl_sample_pair(
                x3z, x3y, x3x,
                lo_ptr, lo_num_phi_ptr, lo_offsets_ptr, loNZ, loNR, loTOTAL,
                hi_ptr, hi_num_phi_ptr, hi_offsets_ptr, hiNZ, hiNR, hiTOTAL, CUBIC, m)
            x4z, x4y, x4x = yz + h * k3z, yy + h * k3y, yx + h * k3x
            if STORE_STAGES:
                s = (step * 4 + 3) * N.to(tl.int64)
                tl.store(stages_ptr + (s + i) * 3, x4z, mask=m)
                tl.store(stages_ptr + (s + i) * 3 + 1, x4y, mask=m)
                tl.store(stages_ptr + (s + i) * 3 + 2, x4x, mask=m)
            k4z, k4y, k4x = _cyl_sample_pair(
                x4z, x4y, x4x,
                lo_ptr, lo_num_phi_ptr, lo_offsets_ptr, loNZ, loNR, loTOTAL,
                hi_ptr, hi_num_phi_ptr, hi_offsets_ptr, hiNZ, hiNR, hiTOTAL, CUBIC, m)
            yz += h_sixth * (((k1z + 2.0 * k2z) + 2.0 * k3z) + k4z)
            yy += h_sixth * (((k1y + 2.0 * k2y) + 2.0 * k3y) + k4y)
            yx += h_sixth * (((k1x + 2.0 * k2x) + 2.0 * k3x) + k4x)
        tl.store(out_ptr + i * 3, yz, mask=m)
        tl.store(out_ptr + i * 3 + 1, yy, mask=m)
        tl.store(out_ptr + i * 3 + 2, yx, mask=m)

    @triton.jit
    def _cyl_bwd_lattice(glz, glr, glp, pz, radius, phi,
                          field_ptr, num_phi_ptr, offsets_ptr, acc_ptr,
                          NZ, NR, TOTAL, HAS_ACC: tl.constexpr, lane_mask):
        zn = pz * 2.0 - 1.0
        zraw = ((zn + 1.0) * 0.5) * (NZ - 1).to(tl.float32)
        zc = tl.minimum(tl.maximum(zraw, 0.0), (NZ - 1).to(tl.float32))
        z0f = tl.minimum(tl.math.floor(zc), (NZ - 2).to(tl.float32))
        z0 = z0f.to(tl.int32)
        fz1, fz0 = zc - z0f, 1.0 - (zc - z0f)
        rc = radius * (NR - 1).to(tl.float32)
        r0f = tl.minimum(tl.math.floor(rc), (NR - 2).to(tl.float32))
        r0 = r0f.to(tl.int32)
        fr1, fr0 = rc - r0f, 1.0 - (rc - r0f)
        ch = NZ.to(tl.int64) * TOTAL
        vz = tl.zeros(pz.shape, tl.float32)
        vr = tl.zeros(pz.shape, tl.float32)
        vp = tl.zeros(pz.shape, tl.float32)
        gzcoord = tl.zeros(pz.shape, tl.float32)
        grcoord = tl.zeros(pz.shape, tl.float32)
        gphi = tl.zeros(pz.shape, tl.float32)
        for dr in tl.static_range(2):
            ring = r0 + dr
            nphi = tl.load(num_phi_ptr + ring, mask=lane_mask, other=1).to(tl.int32)
            offset = tl.load(offsets_ptr + ring, mask=lane_mask, other=0).to(tl.int64)
            pscale = nphi.to(tl.float32) / _TWO_PI_F32
            pc = phi * pscale
            p0f = tl.math.floor(pc)
            p0 = p0f.to(tl.int32)
            p0 = tl.where(p0 >= nphi, p0 - nphi, p0)
            p1 = tl.where(p0 + 1 >= nphi, 0, p0 + 1)
            fp1, fp0 = pc - p0f, 1.0 - (pc - p0f)
            wr = fr1 if dr == 1 else fr0
            sr = 1.0 if dr == 1 else -1.0
            value_mask = lane_mask & (ring != 0)
            for dz in tl.static_range(2):
                z = z0 + dz
                wz = fz1 if dz == 1 else fz0
                sz = 1.0 if dz == 1 else -1.0
                for dp in tl.static_range(2):
                    pp = p1 if dp == 1 else p0
                    wp = fp1 if dp == 1 else fp0
                    sp = 1.0 if dp == 1 else -1.0
                    w = (wr * wz) * wp
                    idx = z.to(tl.int64) * TOTAL + offset + pp
                    a = tl.load(field_ptr + idx, mask=value_mask, other=0.0)
                    b = tl.load(field_ptr + ch + idx, mask=value_mask, other=0.0)
                    c = tl.load(field_ptr + 2 * ch + idx, mask=value_mask, other=0.0)
                    vz += a * w
                    vr += b * w
                    vp += c * w
                    dot = (a * glz + b * glr) + c * glp
                    gzcoord += ((dot * sz) * wr) * wp
                    grcoord += ((dot * sr) * wz) * wp
                    gphi += (((dot * sp) * wr) * wz) * pscale
                    if HAS_ACC:
                        tl.atomic_add(acc_ptr + idx, glz * w, mask=value_mask)
                        tl.atomic_add(acc_ptr + ch + idx, glr * w, mask=value_mask)
                        tl.atomic_add(acc_ptr + 2 * ch + idx, glp * w, mask=value_mask)
        zmask = (zraw >= 0.0) & (zraw <= (NZ - 1).to(tl.float32))
        return (vz, vr, vp,
                gzcoord * zmask.to(tl.float32) * (NZ - 1).to(tl.float32),
                grcoord * (NR - 1).to(tl.float32), gphi)

    @triton.jit
    def _cyl_bwd_stage(gz, gy, gx, pz, py, px,
                       lo_ptr, lo_num_phi_ptr, lo_offsets_ptr, lo_acc_ptr,
                       loNZ, loNR, loTOTAL,
                       hi_ptr, hi_num_phi_ptr, hi_offsets_ptr, hi_acc_ptr,
                       hiNZ, hiNR, hiTOTAL,
                       HAS_ACC: tl.constexpr, CUBIC: tl.constexpr, lane_mask):
        qy, qx, raw, radius, phi, sin_phi, cos_phi, on_axis = _cyl_coords(py, px)
        glz = gz
        glr = gy * sin_phi + gx * cos_phi
        glp = gy * cos_phi - gx * sin_phi
        if CUBIC:
            lvz, lvr, lvp, lgz, lgr, lgp = _cylb_bwd_lattice(
                glz, glr, glp, pz, radius, phi,
                lo_ptr, lo_num_phi_ptr, lo_offsets_ptr, lo_acc_ptr,
                loNZ, loNR, loTOTAL, HAS_ACC, lane_mask)
            hvz, hvr, hvp, hgz, hgr, hgp = _cylb_bwd_lattice(
                glz, glr, glp, pz, radius, phi,
                hi_ptr, hi_num_phi_ptr, hi_offsets_ptr, hi_acc_ptr,
                hiNZ, hiNR, hiTOTAL, HAS_ACC, lane_mask)
        else:
            lvz, lvr, lvp, lgz, lgr, lgp = _cyl_bwd_lattice(
                glz, glr, glp, pz, radius, phi,
                lo_ptr, lo_num_phi_ptr, lo_offsets_ptr, lo_acc_ptr,
                loNZ, loNR, loTOTAL, HAS_ACC, lane_mask)
            hvz, hvr, hvp, hgz, hgr, hgp = _cyl_bwd_lattice(
                glz, glr, glp, pz, radius, phi,
                hi_ptr, hi_num_phi_ptr, hi_offsets_ptr, hi_acc_ptr,
                hiNZ, hiNR, hiTOTAL, HAS_ACC, lane_mask)
        vr, vp = lvr + hvr, lvp + hvp
        out_y = vr * sin_phi + vp * cos_phi
        out_x = vr * cos_phi - vp * sin_phi
        basis_gphi = gy * out_x - gx * out_y
        g_radius = lgr + hgr
        # clamp(max=1) has the inclusive subgradient used by eager torch.clamp.
        g_radius *= (raw <= 1.0).to(tl.float32)
        g_phi = lgp + hgp + basis_gphi
        inv_raw = 1.0 / raw
        gpy = 2.0 * (g_radius * sin_phi + g_phi * cos_phi * inv_raw)
        gpx = 2.0 * (g_radius * cos_phi - g_phi * sin_phi * inv_raw)
        gpy = tl.where(on_axis, 0.0, gpy)
        gpx = tl.where(on_axis, 0.0, gpx)
        return lgz + hgz, gpy, gpx

    @triton.jit
    def _rk4c_bwd_kernel(grad_y_ptr, grad_pts_ptr, stages_ptr,
                         lo_base, lo_num_phi_ptr, lo_offsets_ptr, lo_acc_base,
                         loNZ, loNR, loTOTAL,
                         hi_base, hi_num_phi_ptr, hi_offsets_ptr, hi_acc_base,
                         hiNZ, hiNR, hiTOTAL,
                         N, h, h_half, h_sixth, n_steps, num_slabs,
                         REVERSE: tl.constexpr, CUBIC: tl.constexpr,
                         HAS_ACC: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        i = pid * BLOCK + tl.arange(0, BLOCK)
        m = i < N
        lo_stride = loNZ.to(tl.int64) * loTOTAL * 3
        hi_stride = hiNZ.to(tl.int64) * hiTOTAL * 3
        # The cubic accumulator is padded to 4 channels (see the cubic
        # lattice section); the trilinear one has the field's own layout.
        if CUBIC:
            lo_acc_stride = loNZ.to(tl.int64) * loTOTAL * _CYLB_ACC_CH
            hi_acc_stride = hiNZ.to(tl.int64) * hiTOTAL * _CYLB_ACC_CH
        else:
            lo_acc_stride = lo_stride
            hi_acc_stride = hi_stride
        gz = tl.load(grad_y_ptr + i * 3, mask=m, other=0.0)
        gy = tl.load(grad_y_ptr + i * 3 + 1, mask=m, other=0.0)
        gx = tl.load(grad_y_ptr + i * 3 + 2, mask=m, other=0.0)
        for step in range(n_steps * num_slabs - 1, -1, -1):
            slab = step // n_steps
            if REVERSE:
                slab = num_slabs - 1 - slab
            lo_ptr = lo_base + slab.to(tl.int64) * lo_stride
            hi_ptr = hi_base + slab.to(tl.int64) * hi_stride
            lo_acc_ptr = lo_acc_base + slab.to(tl.int64) * lo_acc_stride
            hi_acc_ptr = hi_acc_base + slab.to(tl.int64) * hi_acc_stride
            s1 = (step * 4) * N.to(tl.int64)
            s2 = (step * 4 + 1) * N.to(tl.int64)
            s3 = (step * 4 + 2) * N.to(tl.int64)
            s4 = (step * 4 + 3) * N.to(tl.int64)
            g6z, g6y, g6x = gz * h_sixth, gy * h_sixth, gx * h_sixth
            pz = tl.load(stages_ptr + (s4 + i) * 3, mask=m, other=0.0)
            py = tl.load(stages_ptr + (s4 + i) * 3 + 1, mask=m, other=0.0)
            px = tl.load(stages_ptr + (s4 + i) * 3 + 2, mask=m, other=0.0)
            b4z, b4y, b4x = _cyl_bwd_stage(
                g6z, g6y, g6x, pz, py, px,
                lo_ptr, lo_num_phi_ptr, lo_offsets_ptr, lo_acc_ptr, loNZ, loNR, loTOTAL,
                hi_ptr, hi_num_phi_ptr, hi_offsets_ptr, hi_acc_ptr, hiNZ, hiNR, hiTOTAL,
                HAS_ACC, CUBIC, m)
            pz = tl.load(stages_ptr + (s3 + i) * 3, mask=m, other=0.0)
            py = tl.load(stages_ptr + (s3 + i) * 3 + 1, mask=m, other=0.0)
            px = tl.load(stages_ptr + (s3 + i) * 3 + 2, mask=m, other=0.0)
            b3z, b3y, b3x = _cyl_bwd_stage(
                g6z * 2.0 + b4z * h, g6y * 2.0 + b4y * h, g6x * 2.0 + b4x * h,
                pz, py, px,
                lo_ptr, lo_num_phi_ptr, lo_offsets_ptr, lo_acc_ptr, loNZ, loNR, loTOTAL,
                hi_ptr, hi_num_phi_ptr, hi_offsets_ptr, hi_acc_ptr, hiNZ, hiNR, hiTOTAL,
                HAS_ACC, CUBIC, m)
            pz = tl.load(stages_ptr + (s2 + i) * 3, mask=m, other=0.0)
            py = tl.load(stages_ptr + (s2 + i) * 3 + 1, mask=m, other=0.0)
            px = tl.load(stages_ptr + (s2 + i) * 3 + 2, mask=m, other=0.0)
            b2z, b2y, b2x = _cyl_bwd_stage(
                g6z * 2.0 + b3z * h_half, g6y * 2.0 + b3y * h_half, g6x * 2.0 + b3x * h_half,
                pz, py, px,
                lo_ptr, lo_num_phi_ptr, lo_offsets_ptr, lo_acc_ptr, loNZ, loNR, loTOTAL,
                hi_ptr, hi_num_phi_ptr, hi_offsets_ptr, hi_acc_ptr, hiNZ, hiNR, hiTOTAL,
                HAS_ACC, CUBIC, m)
            pz = tl.load(stages_ptr + (s1 + i) * 3, mask=m, other=0.0)
            py = tl.load(stages_ptr + (s1 + i) * 3 + 1, mask=m, other=0.0)
            px = tl.load(stages_ptr + (s1 + i) * 3 + 2, mask=m, other=0.0)
            b1z, b1y, b1x = _cyl_bwd_stage(
                g6z + b2z * h_half, g6y + b2y * h_half, g6x + b2x * h_half,
                pz, py, px,
                lo_ptr, lo_num_phi_ptr, lo_offsets_ptr, lo_acc_ptr, loNZ, loNR, loTOTAL,
                hi_ptr, hi_num_phi_ptr, hi_offsets_ptr, hi_acc_ptr, hiNZ, hiNR, hiTOTAL,
                HAS_ACC, CUBIC, m)
            gz = ((gz + b4z) + b3z + b2z) + b1z
            gy = ((gy + b4y) + b3y + b2y) + b1y
            gx = ((gx + b4x) + b3x + b2x) + b1x
        tl.store(grad_pts_ptr + i * 3, gz, mask=m)
        tl.store(grad_pts_ptr + i * 3 + 1, gy, mask=m)
        tl.store(grad_pts_ptr + i * 3 + 2, gx, mask=m)


_BLOCK = 128
# Kept separate so cylindrical tuning cannot perturb the established
# Cartesian launch geometry. FIT_SPIRAL_CYL_BLOCK is a benchmark-only tuning
# override; production defaults to the measured block below.
_CYL_BLOCK = int(os.environ.get('FIT_SPIRAL_CYL_BLOCK', '128'))


def coalesce_enabled():
    # Deferred (register-coalesced) field-gradient atomics in the RK4
    # adjoint. The backward is ~97% atomic-bound (bench_rk4 2026-07-18:
    # 14.7 ms with acc vs 0.47 ms without at 2.4M pts); consecutive stage
    # points usually share a lattice cell, so deferring turns up to
    # 4*n_steps atomic bursts into one. Point gradients bitwise identical;
    # acc changes only in addend association (atomics-order class).
    return os.environ.get('FIT_SPIRAL_RK4_COALESCE', '1') != '0'


def _launch_args(field):
    # field :: slabs, 3, Z, Y, X
    Z, Y, X = field.shape[2], field.shape[3], field.shape[4]
    return (
        Z, Y, X,
        float(Z - 1), float(Y - 1), float(X - 1),
    )


def _stage_buffer(y0, num_slabs, n_steps):
    # One (slab, step, RK4 stage) position record per point, in the order the
    # forward kernel visits them, for the adjoint sweep.
    return torch.empty(
        int(num_slabs) * int(n_steps) * 4, y0.shape[0], 3,
        device=y0.device, dtype=y0.dtype)


def _run_fwd_kernel(y0, low_field, high_field, high_scale, h, n_steps, reverse,
                    stages):
    n = y0.shape[0]
    out = torch.empty_like(y0)
    Z, Y, X, zm1f, ym1f, xm1f = _launch_args(low_field)
    if n > 0:
        _rk4_fwd_kernel[(triton.cdiv(n, _BLOCK),)](
            y0, out, stages if stages is not None else out,
            low_field, high_field, float(high_scale),
            n, Z, Y, X, zm1f, ym1f, xm1f,
            float(h), float(h / 2), float(h / 6), int(n_steps),
            int(low_field.shape[0]),
            REVERSE=bool(reverse),
            STORE_STAGES=stages is not None, BLOCK=_BLOCK,
        )
    return out


def rk4_integrate(y0, low_field, high_field, high_scale, acc, h, n_steps,
                  reverse=False):
    # Entry point mirroring flow_fields._RK4SparseFlowIntegrate.apply, over
    # every slab of [slabs, 3, Z, Y, X] fields (slabs applied in reverse order
    # with `reverse`). The grad-vs-inference split lives here because
    # Function.forward always runs with grad disabled, so it cannot decide
    # itself whether stage points for the adjoint sweep must be kept.
    if torch.is_grad_enabled() and y0.requires_grad:
        return TritonRK4Integrate.apply(
            y0, low_field, high_field, high_scale, acc, h, n_steps, reverse)
    return _run_fwd_kernel(
        y0.contiguous(), low_field, high_field, high_scale, h, n_steps,
        reverse, None)


class TritonRK4Integrate(torch.autograd.Function):
    # Fused counterpart of flow_fields._RK4SparseFlowIntegrate over all slabs;
    # same saved-state footprint (the 4*n_steps stage points per slab), same
    # accumulator contract for the field gradient.

    @staticmethod
    def forward(ctx, y0, low_field, high_field, high_scale, acc, h, n_steps,
                reverse):
        ctx.set_materialize_grads(False)
        y0 = y0.contiguous()
        perm = None
        if acc is not None and permute_enabled() and y0.shape[0] > _PERM_STRIDE:
            perm = _interleave_perm(y0.shape[0], y0.device)
            y0 = y0[perm].contiguous()
        stages = _stage_buffer(y0, low_field.shape[0], n_steps)
        out = _run_fwd_kernel(
            y0, low_field, high_field, high_scale, h, n_steps, reverse, stages)
        if perm is not None:
            unperm = torch.empty_like(out)
            unperm[perm] = out
            out = unperm
        ctx.save_for_backward(low_field, high_field, stages)
        ctx.perm = perm
        ctx.high_scale = float(high_scale)
        ctx.acc = acc
        ctx.h = float(h)
        ctx.n_steps = int(n_steps)
        ctx.reverse = bool(reverse)
        return out

    @staticmethod
    def backward(ctx, grad_y):
        if grad_y is None:
            return (None,) * 8
        low_field, high_field, stages = ctx.saved_tensors
        perm = ctx.perm
        if perm is not None:
            grad_y = grad_y[perm]
        grad_y = grad_y.contiguous()
        n = grad_y.shape[0]
        grad_pts = torch.empty_like(grad_y)
        acc = ctx.acc
        h = ctx.h
        Z, Y, X, zm1f, ym1f, xm1f = _launch_args(low_field)
        num_slabs = int(low_field.shape[0])
        if n > 0 and acc is not None and coalesce_enabled():
            _rk4_bwd_coalesced_kernel[(triton.cdiv(n, _BLOCK),)](
                grad_y, grad_pts, stages,
                low_field, high_field, acc, ctx.high_scale,
                n, Z, Y, X, zm1f, ym1f, xm1f,
                float(Z - 2), float(Y - 2), float(X - 2),
                h, float(h / 2), float(h / 6), ctx.n_steps, num_slabs,
                REVERSE=ctx.reverse, BLOCK=_BLOCK,
            )
        elif n > 0:
            _rk4_bwd_kernel[(triton.cdiv(n, _BLOCK),)](
                grad_y, grad_pts, stages,
                low_field, high_field,
                acc if acc is not None else low_field, ctx.high_scale,
                n, Z, Y, X, zm1f, ym1f, xm1f,
                float(Z - 2), float(Y - 2), float(X - 2),
                h, float(h / 2), float(h / 6), ctx.n_steps, num_slabs,
                REVERSE=ctx.reverse,
                HAS_ACC=acc is not None, BLOCK=_BLOCK,
            )
        if perm is not None:
            unperm = torch.empty_like(grad_pts)
            unperm[perm] = grad_pts
            grad_pts = unperm
        return grad_pts, None, None, None, None, None, None, None


def _field_geoms(low_field, high_field):
    # (a, b) per dim per field s.t. lattice coord = normalised_point * a + b.
    # Fields are [slabs, 3, Z, Y, X]; every slab shares the geometry.
    geom = []
    for l, hh in zip(low_field.shape[2:], high_field.shape[2:]):
        if l == hh:
            geom += [float(hh - 1), 0.0]
        else:
            # grid_sample(align_corners=True) onto the HR lattice composed
            # with F.interpolate(align_corners=False)'s source-index map.
            s = l / hh
            geom += [float((hh - 1) * s), float(0.5 * s - 0.5)]
    for hh in high_field.shape[2:]:
        geom += [float(hh - 1), 0.0]
    return geom  # lo az,bz,ay,by,ax,bx then hi az,bz,ay,by,ax,bx


def _run_direct_fwd(y0, low, high, lo_scale, hi_scale, h, n_steps, reverse,
                    stages):
    n = y0.shape[0]
    out = torch.empty_like(y0)
    if n > 0:
        g = _field_geoms(low, high)
        _rk4d_fwd_kernel[(triton.cdiv(n, _BLOCK),)](
            y0, out, stages if stages is not None else out,
            low, *g[:6], low.shape[2], low.shape[3], low.shape[4], float(lo_scale),
            high, *g[6:], high.shape[2], high.shape[3], high.shape[4], float(hi_scale),
            n, float(h), float(h / 2), float(h / 6), int(n_steps),
            int(low.shape[0]),
            REVERSE=bool(reverse),
            STORE_STAGES=stages is not None, BLOCK=_BLOCK,
        )
    return out


def rk4_direct_integrate(y0, low, high, lo_scale, hi_scale, acc_lo, acc_hi,
                         h, n_steps, reverse=False):
    # low :: slabs, 3, Zl, Yl, Xl and high :: slabs, 3, Z, Y, X; the slabs
    # are applied in order (reverse order with `reverse`).
    if torch.is_grad_enabled() and y0.requires_grad:
        return TritonRK4DirectIntegrate.apply(
            y0, low, high, lo_scale, hi_scale, acc_lo, acc_hi, h, n_steps,
            reverse)
    return _run_direct_fwd(
        y0.contiguous(), low, high, lo_scale, hi_scale, h, n_steps, reverse,
        None)


class TritonRK4DirectIntegrate(torch.autograd.Function):
    # RK4 integration sampling the LR lattice directly (no per-step upsample).
    # Field gradients are scattered unscaled into the two caller-owned
    # accumulators; the caller applies each field's value scale afterwards
    # (see CartesianFlowField.apply_accumulated_field_grad).

    @staticmethod
    def forward(ctx, y0, low, high, lo_scale, hi_scale, acc_lo, acc_hi, h,
                n_steps, reverse):
        ctx.set_materialize_grads(False)
        y0 = y0.contiguous()
        stages = _stage_buffer(y0, low.shape[0], n_steps)
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
        h = ctx.h
        if n > 0:
            g = _field_geoms(low, high)
            _rk4d_bwd_kernel[(triton.cdiv(n, _BLOCK),)](
                grad_y, grad_pts, stages,
                low, acc_lo if acc_lo is not None else low, *g[:6],
                low.shape[2], low.shape[3], low.shape[4], lo_scale,
                high, acc_hi if acc_hi is not None else high, *g[6:],
                high.shape[2], high.shape[3], high.shape[4], hi_scale,
                n, h, float(h / 2), float(h / 6), ctx.n_steps,
                int(low.shape[0]),
                REVERSE=ctx.reverse,
                HAS_ACC=acc_lo is not None, BLOCK=_BLOCK,
            )
        return grad_pts, None, None, None, None, None, None, None, None, None


def cyl_defer_enabled():
    # Joint per-RK4-step field-gradient scatter in the cubic cylindrical
    # adjoint (_rk4cb_bwd_deferred_kernel) instead of one atomic burst per
    # stage. Point gradients are unchanged; the accumulator differs only in
    # addend association (atomics-order class).
    return os.environ.get('FIT_SPIRAL_CYL_DEFER', '1') != '0'


def _cyl_geometry(field, cubic):
    # (nz, total_phi) of a packed cylindrical lattice tensor: the parameter
    # layout [slabs, 3, nz, total_phi] for the trilinear kernels, the
    # channels-last [slabs, nz, total_phi, 3] copy for the cubic ones.
    if cubic:
        return field.shape[1], field.shape[2]
    return field.shape[2], field.shape[3]


def _run_cylindrical_fwd(y0, low, low_num_phi, low_offsets,
                         high, high_num_phi, high_offsets,
                         h, n_steps, reverse, cubic, stages):
    n = y0.shape[0]
    out = torch.empty_like(y0)
    if n > 0:
        lo_nz, lo_total = _cyl_geometry(low, cubic)
        hi_nz, hi_total = _cyl_geometry(high, cubic)
        _rk4c_fwd_kernel[(triton.cdiv(n, _CYL_BLOCK),)](
            y0, out, stages if stages is not None else out,
            low, low_num_phi, low_offsets,
            lo_nz, low_num_phi.numel(), lo_total,
            high, high_num_phi, high_offsets,
            hi_nz, high_num_phi.numel(), hi_total,
            n, float(h), float(h / 2), float(h / 6), int(n_steps),
            int(low.shape[0]),
            REVERSE=bool(reverse), CUBIC=bool(cubic),
            STORE_STAGES=stages is not None, BLOCK=_CYL_BLOCK,
        )
    return out


def rk4_cylindrical_integrate(y0, low, low_num_phi, low_offsets,
                              high, high_num_phi, high_offsets,
                              acc_low, acc_high, h, n_steps, reverse=False,
                              cubic=False):
    """Integrate a pair of packed cylindrical flow lattices, slab by slab.

    The slabs are applied in order (reverse order with ``reverse``).
    ``cubic`` samples the lattices with the tricubic B-spline interpolant of
    BSplineCylindricalFlowField instead of the trilinear one, and changes
    the tensor layouts: trilinear takes ``low`` / ``high`` in the parameter
    layout [slabs, 3, nz, total_phi] and accumulators of the same shape;
    cubic takes channels-last copies [slabs, nz, total_phi, 3] and padded
    channels-last accumulators [slabs, nz, total_phi, 4] (see the cubic
    lattice section above).
    """
    if torch.is_grad_enabled() and (
            y0.requires_grad or low.requires_grad or high.requires_grad):
        return TritonRK4CylindricalIntegrate.apply(
            y0, low, low_num_phi, low_offsets,
            high, high_num_phi, high_offsets,
            acc_low, acc_high, h, n_steps, reverse, cubic)
    return _run_cylindrical_fwd(
        y0.contiguous(), low, low_num_phi, low_offsets,
        high, high_num_phi, high_offsets, h, n_steps, reverse, cubic, None)


class TritonRK4CylindricalIntegrate(torch.autograd.Function):
    """One forward kernel and one analytic-adjoint kernel for cylindrical RK4."""

    @staticmethod
    def forward(ctx, y0, low, low_num_phi, low_offsets,
                high, high_num_phi, high_offsets,
                acc_low, acc_high, h, n_steps, reverse, cubic):
        ctx.set_materialize_grads(False)
        y0 = y0.contiguous()
        stages = _stage_buffer(y0, low.shape[0], n_steps)
        out = _run_cylindrical_fwd(
            y0, low, low_num_phi, low_offsets,
            high, high_num_phi, high_offsets, h, n_steps, reverse, cubic,
            stages)
        ctx.save_for_backward(
            low, low_num_phi, low_offsets,
            high, high_num_phi, high_offsets, stages)
        ctx.accs = (acc_low, acc_high)
        ctx.h = float(h)
        ctx.n_steps = int(n_steps)
        ctx.reverse = bool(reverse)
        ctx.cubic = bool(cubic)
        return out

    @staticmethod
    def backward(ctx, grad_y):
        if grad_y is None:
            return (None,) * 13
        (low, low_num_phi, low_offsets,
         high, high_num_phi, high_offsets, stages) = ctx.saved_tensors
        grad_y = grad_y.contiguous()
        grad_pts = torch.empty_like(grad_y)
        n = grad_y.shape[0]
        acc_low, acc_high = ctx.accs
        if n > 0:
            has_acc = acc_low is not None
            lo_nz, lo_total = _cyl_geometry(low, ctx.cubic)
            hi_nz, hi_total = _cyl_geometry(high, ctx.cubic)
            args = (
                grad_y, grad_pts, stages,
                low, low_num_phi, low_offsets,
                acc_low if has_acc else low,
                lo_nz, low_num_phi.numel(), lo_total,
                high, high_num_phi, high_offsets,
                acc_high if has_acc else high,
                hi_nz, high_num_phi.numel(), hi_total,
                n, ctx.h, float(ctx.h / 2), float(ctx.h / 6), ctx.n_steps,
                int(low.shape[0]),
            )
            grid = (triton.cdiv(n, _CYL_BLOCK),)
            if ctx.cubic and has_acc and cyl_defer_enabled():
                _rk4cb_bwd_deferred_kernel[grid](
                    *args, REVERSE=ctx.reverse, BLOCK=_CYL_BLOCK)
            else:
                _rk4c_bwd_kernel[grid](
                    *args, REVERSE=ctx.reverse, CUBIC=ctx.cubic,
                    HAS_ACC=has_acc, BLOCK=_CYL_BLOCK)
        return (grad_pts,) + (None,) * 12


if _HAS_TRITON:

    # ---- b-spline mode: tricubic 4x4x4 stencil per lattice ----
    # Each lattice is sampled at coord = p * (size - 1) per dim
    # (align_corners=True), with the query clamped to the lattice box and
    # out-of-range stencil taps clamped to the edge (replicate). This is the
    # same interpolant as flow_fields.sample_field_bspline: the eager path
    # evaluates it as 8 border-clamped trilinear fetches (the Sigg & Hadwiger
    # decomposition), this kernel evaluates the 64-tap weighted sum directly,
    # so results agree up to FP association (the tolerance-based contract at
    # the top of this file).
    #
    # Field VALUES are read from channels-last [slabs, Z, Y, X, 3] copies so
    # each tap's three components come from consecutive addresses (one cache
    # line) instead of three planes a full channel apart -- with 64 taps per
    # lattice per stage the read path dominates. The gradient ACCUMULATORS
    # stay in the parameter layout [slabs, 3, Z, Y, X] so they serve directly
    # as gradients (atomics don't coalesce across addresses anyway). Both
    # layouts share the slab stride 3 * Z * Y * X.

    @triton.jit
    def _bspline_weights(f):
        # Uniform cubic B-spline basis over taps lo-1 .. lo+2, matching
        # sample_field_bspline's w0..w3.
        omf = 1.0 - f
        f2 = f * f
        f3 = f2 * f
        w0 = (omf * omf) * omf * (1.0 / 6.0)
        w1 = (3.0 * f3 - 6.0 * f2 + 4.0) * (1.0 / 6.0)
        w2 = (-3.0 * f3 + 3.0 * f2 + 3.0 * f + 1.0) * (1.0 / 6.0)
        w3 = f3 * (1.0 / 6.0)
        return w0, w1, w2, w3

    @triton.jit
    def _bspline_dweights(f):
        # d/df of _bspline_weights (sums to zero).
        omf = 1.0 - f
        f2 = f * f
        d0 = -0.5 * (omf * omf)
        d1 = 1.5 * f2 - 2.0 * f
        d2 = -1.5 * f2 + f + 0.5
        d3 = 0.5 * f2
        return d0, d1, d2, d3

    @triton.jit
    def _bspline_sample_one(ptr, pz, py, px, Z, Y, X, lane_mask):
        zm1f = (Z - 1).to(tl.float32)
        ym1f = (Y - 1).to(tl.float32)
        xm1f = (X - 1).to(tl.float32)
        cz = tl.minimum(tl.maximum(pz * zm1f, 0.0), zm1f)
        cy = tl.minimum(tl.maximum(py * ym1f, 0.0), ym1f)
        cx = tl.minimum(tl.maximum(px * xm1f, 0.0), xm1f)
        lzf = tl.math.floor(cz)
        lyf = tl.math.floor(cy)
        lxf = tl.math.floor(cx)
        wz0, wz1, wz2, wz3 = _bspline_weights(cz - lzf)
        wy0, wy1, wy2, wy3 = _bspline_weights(cy - lyf)
        wx0, wx1, wx2, wx3 = _bspline_weights(cx - lxf)
        lz = lzf.to(tl.int32)
        ly = lyf.to(tl.int32)
        lx = lxf.to(tl.int32)
        v0 = tl.zeros(pz.shape, dtype=tl.float32)
        v1 = tl.zeros(pz.shape, dtype=tl.float32)
        v2 = tl.zeros(pz.shape, dtype=tl.float32)
        # Outer axes iterate as device-side loops with branchless weight
        # selection; a fully static 4x4x4 unroll (x 4 RK4 stages x 2
        # lattices) made ptxas compile times run to tens of minutes.
        for dz in range(4):
            z = tl.minimum(tl.maximum(lz + (dz - 1), 0), Z - 1)
            wz = tl.where(dz == 0, wz0, tl.where(dz == 1, wz1, tl.where(dz == 2, wz2, wz3)))
            for dy in range(4):
                y = tl.minimum(tl.maximum(ly + (dy - 1), 0), Y - 1)
                wy = tl.where(dy == 0, wy0, tl.where(dy == 1, wy1, tl.where(dy == 2, wy2, wy3)))
                wzy = wz * wy
                row = (z.to(tl.int64) * Y + y) * X
                for dx in tl.static_range(4):
                    x = tl.minimum(tl.maximum(lx + (dx - 1), 0), X - 1)
                    wx = wx0 if dx == 0 else (wx1 if dx == 1 else (wx2 if dx == 2 else wx3))
                    w = wzy * wx
                    idx3 = (row + x) * 3
                    v0 += tl.load(ptr + idx3, mask=lane_mask, other=0.0) * w
                    v1 += tl.load(ptr + idx3 + 1, mask=lane_mask, other=0.0) * w
                    v2 += tl.load(ptr + idx3 + 2, mask=lane_mask, other=0.0) * w
        return v0, v1, v2

    @triton.jit
    def _bspline_sample_pair(pz, py, px,
                             lo_ptr, loZ, loY, loX,
                             hi_ptr, hiZ, hiY, hiX, lane_mask):
        l0, l1, l2 = _bspline_sample_one(lo_ptr, pz, py, px, loZ, loY, loX, lane_mask)
        h0, h1, h2 = _bspline_sample_one(hi_ptr, pz, py, px, hiZ, hiY, hiX, lane_mask)
        return l0 + h0, l1 + h1, l2 + h2

    @triton.jit
    def _rk4b_fwd_kernel(y_ptr, out_ptr, stages_ptr,
                         lo_base, loZ, loY, loX,
                         hi_base, hiZ, hiY, hiX,
                         N, h, h_half, h_sixth, n_steps, num_slabs,
                         REVERSE: tl.constexpr,
                         STORE_STAGES: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        i = pid * BLOCK + tl.arange(0, BLOCK)
        m = i < N
        lo_stride = loZ.to(tl.int64) * loY * loX * 3
        hi_stride = hiZ.to(tl.int64) * hiY * hiX * 3
        yz = tl.load(y_ptr + i * 3 + 0, mask=m, other=0.0)
        yy = tl.load(y_ptr + i * 3 + 1, mask=m, other=0.0)
        yx = tl.load(y_ptr + i * 3 + 2, mask=m, other=0.0)
        for step in range(n_steps * num_slabs):
            slab = step // n_steps
            if REVERSE:
                slab = num_slabs - 1 - slab
            lo_ptr = lo_base + slab.to(tl.int64) * lo_stride
            hi_ptr = hi_base + slab.to(tl.int64) * hi_stride
            if STORE_STAGES:
                s = (step * 4 + 0) * N.to(tl.int64)
                tl.store(stages_ptr + (s + i) * 3 + 0, yz, mask=m)
                tl.store(stages_ptr + (s + i) * 3 + 1, yy, mask=m)
                tl.store(stages_ptr + (s + i) * 3 + 2, yx, mask=m)
            k1z, k1y, k1x = _bspline_sample_pair(
                yz, yy, yx, lo_ptr, loZ, loY, loX, hi_ptr, hiZ, hiY, hiX, m)
            x2z = yz + h_half * k1z
            x2y = yy + h_half * k1y
            x2x = yx + h_half * k1x
            if STORE_STAGES:
                s = (step * 4 + 1) * N.to(tl.int64)
                tl.store(stages_ptr + (s + i) * 3 + 0, x2z, mask=m)
                tl.store(stages_ptr + (s + i) * 3 + 1, x2y, mask=m)
                tl.store(stages_ptr + (s + i) * 3 + 2, x2x, mask=m)
            k2z, k2y, k2x = _bspline_sample_pair(
                x2z, x2y, x2x, lo_ptr, loZ, loY, loX, hi_ptr, hiZ, hiY, hiX, m)
            x3z = yz + h_half * k2z
            x3y = yy + h_half * k2y
            x3x = yx + h_half * k2x
            if STORE_STAGES:
                s = (step * 4 + 2) * N.to(tl.int64)
                tl.store(stages_ptr + (s + i) * 3 + 0, x3z, mask=m)
                tl.store(stages_ptr + (s + i) * 3 + 1, x3y, mask=m)
                tl.store(stages_ptr + (s + i) * 3 + 2, x3x, mask=m)
            k3z, k3y, k3x = _bspline_sample_pair(
                x3z, x3y, x3x, lo_ptr, loZ, loY, loX, hi_ptr, hiZ, hiY, hiX, m)
            x4z = yz + h * k3z
            x4y = yy + h * k3y
            x4x = yx + h * k3x
            if STORE_STAGES:
                s = (step * 4 + 3) * N.to(tl.int64)
                tl.store(stages_ptr + (s + i) * 3 + 0, x4z, mask=m)
                tl.store(stages_ptr + (s + i) * 3 + 1, x4y, mask=m)
                tl.store(stages_ptr + (s + i) * 3 + 2, x4x, mask=m)
            k4z, k4y, k4x = _bspline_sample_pair(
                x4z, x4y, x4x, lo_ptr, loZ, loY, loX, hi_ptr, hiZ, hiY, hiX, m)
            yz = yz + h_sixth * (((k1z + 2.0 * k2z) + 2.0 * k3z) + k4z)
            yy = yy + h_sixth * (((k1y + 2.0 * k2y) + 2.0 * k3y) + k4y)
            yx = yx + h_sixth * (((k1x + 2.0 * k2x) + 2.0 * k3x) + k4x)
        tl.store(out_ptr + i * 3 + 0, yz, mask=m)
        tl.store(out_ptr + i * 3 + 1, yy, mask=m)
        tl.store(out_ptr + i * 3 + 2, yx, mask=m)

    @triton.jit
    def _bspline_bwd_one(gz, gy, gx, pz, py, px, ptr, acc_ptr, Z, Y, X,
                         HAS_ACC: tl.constexpr, lane_mask):
        # Sampler backward for one lattice: scatter dL/d(control points) into
        # acc and return dL/d(point). Same clamp/floor cell choice as the
        # forward sampler, so value and gradient agree with autograd through
        # sample_field_bspline (tap-index clamping is piecewise-constant and
        # contributes no gradient; the query-box clamp contributes the
        # inclusive in-range mask, like torch.clamp).
        zm1f = (Z - 1).to(tl.float32)
        ym1f = (Y - 1).to(tl.float32)
        xm1f = (X - 1).to(tl.float32)
        crz = pz * zm1f
        cry = py * ym1f
        crx = px * xm1f
        cz = tl.minimum(tl.maximum(crz, 0.0), zm1f)
        cy = tl.minimum(tl.maximum(cry, 0.0), ym1f)
        cx = tl.minimum(tl.maximum(crx, 0.0), xm1f)
        lzf = tl.math.floor(cz)
        lyf = tl.math.floor(cy)
        lxf = tl.math.floor(cx)
        fz = cz - lzf
        fy = cy - lyf
        fx = cx - lxf
        wz0, wz1, wz2, wz3 = _bspline_weights(fz)
        wy0, wy1, wy2, wy3 = _bspline_weights(fy)
        wx0, wx1, wx2, wx3 = _bspline_weights(fx)
        dz0, dz1, dz2, dz3 = _bspline_dweights(fz)
        dy0, dy1, dy2, dy3 = _bspline_dweights(fy)
        dx0, dx1, dx2, dx3 = _bspline_dweights(fx)
        lz = lzf.to(tl.int32)
        ly = lyf.to(tl.int32)
        lx = lxf.to(tl.int32)
        ch = Z.to(tl.int64) * Y * X
        gcz = tl.zeros(pz.shape, dtype=tl.float32)
        gcy = tl.zeros(pz.shape, dtype=tl.float32)
        gcx = tl.zeros(pz.shape, dtype=tl.float32)
        for dz in range(4):
            z = tl.minimum(tl.maximum(lz + (dz - 1), 0), Z - 1)
            wz = tl.where(dz == 0, wz0, tl.where(dz == 1, wz1, tl.where(dz == 2, wz2, wz3)))
            dwz = tl.where(dz == 0, dz0, tl.where(dz == 1, dz1, tl.where(dz == 2, dz2, dz3)))
            for dy in range(4):
                y = tl.minimum(tl.maximum(ly + (dy - 1), 0), Y - 1)
                wy = tl.where(dy == 0, wy0, tl.where(dy == 1, wy1, tl.where(dy == 2, wy2, wy3)))
                dwy = tl.where(dy == 0, dy0, tl.where(dy == 1, dy1, tl.where(dy == 2, dy2, dy3)))
                row = (z.to(tl.int64) * Y + y) * X
                for dx in tl.static_range(4):
                    x = tl.minimum(tl.maximum(lx + (dx - 1), 0), X - 1)
                    wx = wx0 if dx == 0 else (wx1 if dx == 1 else (wx2 if dx == 2 else wx3))
                    dwx = dx0 if dx == 0 else (dx1 if dx == 1 else (dx2 if dx == 2 else dx3))
                    idx = row + x
                    idx3 = idx * 3
                    w = (wz * wy) * wx
                    if HAS_ACC:
                        tl.atomic_add(acc_ptr + idx, gz * w, mask=lane_mask)
                        tl.atomic_add(acc_ptr + ch + idx, gy * w, mask=lane_mask)
                        tl.atomic_add(acc_ptr + 2 * ch + idx, gx * w, mask=lane_mask)
                    v0 = tl.load(ptr + idx3, mask=lane_mask, other=0.0)
                    v1 = tl.load(ptr + idx3 + 1, mask=lane_mask, other=0.0)
                    v2 = tl.load(ptr + idx3 + 2, mask=lane_mask, other=0.0)
                    vdg = (v0 * gz + v1 * gy) + v2 * gx
                    gcz += vdg * ((dwz * wy) * wx)
                    gcy += vdg * ((wz * dwy) * wx)
                    gcx += vdg * ((wz * wy) * dwx)
        mz = ((crz >= 0.0) & (crz <= zm1f)).to(tl.float32)
        my = ((cry >= 0.0) & (cry <= ym1f)).to(tl.float32)
        mx = ((crx >= 0.0) & (crx <= xm1f)).to(tl.float32)
        return (gcz * mz) * zm1f, (gcy * my) * ym1f, (gcx * mx) * xm1f

    @triton.jit
    def _bspline_bwd_pair(gz, gy, gx, pz, py, px,
                          lo_ptr, acc_lo_ptr, loZ, loY, loX,
                          hi_ptr, acc_hi_ptr, hiZ, hiY, hiX,
                          HAS_ACC: tl.constexpr, lane_mask):
        lz, ly, lx = _bspline_bwd_one(gz, gy, gx, pz, py, px, lo_ptr, acc_lo_ptr,
                                      loZ, loY, loX, HAS_ACC, lane_mask)
        hz, hy, hx = _bspline_bwd_one(gz, gy, gx, pz, py, px, hi_ptr, acc_hi_ptr,
                                      hiZ, hiY, hiX, HAS_ACC, lane_mask)
        return lz + hz, ly + hy, lx + hx

    @triton.jit
    def _rk4b_bwd_kernel(grad_y_ptr, grad_pts_ptr, stages_ptr,
                         lo_base, acc_lo_base, loZ, loY, loX,
                         hi_base, acc_hi_base, hiZ, hiY, hiX,
                         N, h, h_half, h_sixth, n_steps, num_slabs,
                         REVERSE: tl.constexpr,
                         HAS_ACC: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        i = pid * BLOCK + tl.arange(0, BLOCK)
        m = i < N
        lo_stride = loZ.to(tl.int64) * loY * loX * 3
        hi_stride = hiZ.to(tl.int64) * hiY * hiX * 3
        gz = tl.load(grad_y_ptr + i * 3 + 0, mask=m, other=0.0)
        gy = tl.load(grad_y_ptr + i * 3 + 1, mask=m, other=0.0)
        gx = tl.load(grad_y_ptr + i * 3 + 2, mask=m, other=0.0)
        for step in range(n_steps * num_slabs - 1, -1, -1):
            slab = step // n_steps
            if REVERSE:
                slab = num_slabs - 1 - slab
            lo_off = slab.to(tl.int64) * lo_stride
            hi_off = slab.to(tl.int64) * hi_stride
            lo_ptr = lo_base + lo_off
            hi_ptr = hi_base + hi_off
            acc_lo_ptr = acc_lo_base + lo_off
            acc_hi_ptr = acc_hi_base + hi_off
            s1 = (step * 4 + 0) * N.to(tl.int64)
            s2 = (step * 4 + 1) * N.to(tl.int64)
            s3 = (step * 4 + 2) * N.to(tl.int64)
            s4 = (step * 4 + 3) * N.to(tl.int64)
            g6z = gz * h_sixth
            g6y = gy * h_sixth
            g6x = gx * h_sixth
            pz = tl.load(stages_ptr + (s4 + i) * 3 + 0, mask=m, other=0.0)
            py = tl.load(stages_ptr + (s4 + i) * 3 + 1, mask=m, other=0.0)
            px = tl.load(stages_ptr + (s4 + i) * 3 + 2, mask=m, other=0.0)
            b4z, b4y, b4x = _bspline_bwd_pair(
                g6z, g6y, g6x, pz, py, px,
                lo_ptr, acc_lo_ptr, loZ, loY, loX,
                hi_ptr, acc_hi_ptr, hiZ, hiY, hiX, HAS_ACC, m)
            pz = tl.load(stages_ptr + (s3 + i) * 3 + 0, mask=m, other=0.0)
            py = tl.load(stages_ptr + (s3 + i) * 3 + 1, mask=m, other=0.0)
            px = tl.load(stages_ptr + (s3 + i) * 3 + 2, mask=m, other=0.0)
            b3z, b3y, b3x = _bspline_bwd_pair(
                g6z * 2.0 + b4z * h, g6y * 2.0 + b4y * h, g6x * 2.0 + b4x * h,
                pz, py, px,
                lo_ptr, acc_lo_ptr, loZ, loY, loX,
                hi_ptr, acc_hi_ptr, hiZ, hiY, hiX, HAS_ACC, m)
            pz = tl.load(stages_ptr + (s2 + i) * 3 + 0, mask=m, other=0.0)
            py = tl.load(stages_ptr + (s2 + i) * 3 + 1, mask=m, other=0.0)
            px = tl.load(stages_ptr + (s2 + i) * 3 + 2, mask=m, other=0.0)
            b2z, b2y, b2x = _bspline_bwd_pair(
                g6z * 2.0 + b3z * h_half, g6y * 2.0 + b3y * h_half, g6x * 2.0 + b3x * h_half,
                pz, py, px,
                lo_ptr, acc_lo_ptr, loZ, loY, loX,
                hi_ptr, acc_hi_ptr, hiZ, hiY, hiX, HAS_ACC, m)
            pz = tl.load(stages_ptr + (s1 + i) * 3 + 0, mask=m, other=0.0)
            py = tl.load(stages_ptr + (s1 + i) * 3 + 1, mask=m, other=0.0)
            px = tl.load(stages_ptr + (s1 + i) * 3 + 2, mask=m, other=0.0)
            b1z, b1y, b1x = _bspline_bwd_pair(
                g6z + b2z * h_half, g6y + b2y * h_half, g6x + b2x * h_half,
                pz, py, px,
                lo_ptr, acc_lo_ptr, loZ, loY, loX,
                hi_ptr, acc_hi_ptr, hiZ, hiY, hiX, HAS_ACC, m)
            gz = ((gz + b4z) + b3z + b2z) + b1z
            gy = ((gy + b4y) + b3y + b2y) + b1y
            gx = ((gx + b4x) + b3x + b2x) + b1x
        tl.store(grad_pts_ptr + i * 3 + 0, gz, mask=m)
        tl.store(grad_pts_ptr + i * 3 + 1, gy, mask=m)
        tl.store(grad_pts_ptr + i * 3 + 2, gx, mask=m)


def _run_bspline_fwd(y0, low, high, h, n_steps, reverse, stages):
    # low / high :: slabs, Z, Y, X, 3 (channels-last).
    n = y0.shape[0]
    out = torch.empty_like(y0)
    if n > 0:
        _rk4b_fwd_kernel[(triton.cdiv(n, _BLOCK),)](
            y0, out, stages if stages is not None else out,
            low, low.shape[1], low.shape[2], low.shape[3],
            high, high.shape[1], high.shape[2], high.shape[3],
            n, float(h), float(h / 2), float(h / 6), int(n_steps),
            int(low.shape[0]),
            REVERSE=bool(reverse),
            STORE_STAGES=stages is not None, BLOCK=_BLOCK,
        )
    return out


def rk4_bspline_integrate(y0, low, high, acc_lo, acc_hi, h, n_steps,
                          reverse=False):
    # RK4 integration of the summed LR + HR tricubic B-spline lattices (see
    # BSplineFlowField), slab by slab (reverse order with `reverse`). `low`
    # and `high` are contiguous CHANNELS-LAST [slabs, Z, Y, X, 3] copies of
    # the lattices; the caller-owned accumulators stay in the parameter
    # layout [slabs, 3, Z, Y, X] and ARE the parameter gradients (both
    # lattices enter the sum unscaled).
    assert low.shape[-1] == 3 and high.shape[-1] == 3
    if torch.is_grad_enabled() and y0.requires_grad:
        return TritonRK4BSplineIntegrate.apply(
            y0, low, high, acc_lo, acc_hi, h, n_steps, reverse)
    return _run_bspline_fwd(y0.contiguous(), low, high, h, n_steps, reverse, None)


class TritonRK4BSplineIntegrate(torch.autograd.Function):
    # Same saved-state footprint and accumulator contract as
    # TritonRK4DirectIntegrate, with tricubic B-spline sampling.

    @staticmethod
    def forward(ctx, y0, low, high, acc_lo, acc_hi, h, n_steps, reverse):
        ctx.set_materialize_grads(False)
        y0 = y0.contiguous()
        stages = _stage_buffer(y0, low.shape[0], n_steps)
        out = _run_bspline_fwd(y0, low, high, h, n_steps, reverse, stages)
        ctx.save_for_backward(low, high, stages)
        ctx.accs = (acc_lo, acc_hi)
        ctx.h = float(h)
        ctx.n_steps = int(n_steps)
        ctx.reverse = bool(reverse)
        return out

    @staticmethod
    def backward(ctx, grad_y):
        if grad_y is None:
            return (None,) * 8
        low, high, stages = ctx.saved_tensors
        grad_y = grad_y.contiguous()
        n = grad_y.shape[0]
        grad_pts = torch.empty_like(grad_y)
        acc_lo, acc_hi = ctx.accs
        h = ctx.h
        if n > 0:
            _rk4b_bwd_kernel[(triton.cdiv(n, _BLOCK),)](
                grad_y, grad_pts, stages,
                low, acc_lo if acc_lo is not None else low,
                low.shape[1], low.shape[2], low.shape[3],
                high, acc_hi if acc_hi is not None else high,
                high.shape[1], high.shape[2], high.shape[3],
                n, h, float(h / 2), float(h / 6), ctx.n_steps,
                int(low.shape[0]),
                REVERSE=ctx.reverse,
                HAS_ACC=acc_lo is not None, BLOCK=_BLOCK,
            )
        return grad_pts, None, None, None, None, None, None, None


if _HAS_TRITON:

    # ---- cubic cylindrical lattice: 4x4x4 stencil on the packed ragged
    # lattice (BSplineCylindricalFlowField._sample_lattice) ----
    # Cubic along z and r with edge-clamped tap indices, periodic cubic along
    # phi within each ring, each of the four stencil rings using its own phi
    # parameterisation. Ring 0 (the pinned axis ring) is masked out of both
    # the value reads and the gradient scatter, which is how the fused path
    # reproduces the eager sampler's zeroed r=0 slice.
    #
    # Memory layout. With 64 taps per lattice per stage this interpolant is
    # bound by memory transactions, so the kernels read field VALUES from
    # channels-last copies [slabs, nz, total_phi, 3]: a tap's three
    # components share one 32-byte sector and a ring row's four phi taps are
    # 48 contiguous bytes, instead of twelve sectors three planes apart. The
    # gradient ACCUMULATOR is channels-last as well and padded to four
    # channels, [slabs, nz, total_phi, 4], so each tap's scatter is one
    # 16-byte vector atomic (atom.v4.f32, sm_90+); CylindricalFlowField
    # permutes it back to the parameter layout once per iteration. Measured
    # on an RTX 5090 at the 330x31103 lattice of a 32-voxel production fit,
    # the value gathers run 2.2x and the scatter 2.8x faster than the planar
    # layout (scratch micro-benchmark, 2026-09-10).
    #
    # Deferred scatter (FIT_SPIRAL_CYL_DEFER, default on). The four stage
    # points of one RK4 step usually share a stencil cell (the per-step
    # displacement is a fraction of a cell), so instead of four 64-tap
    # atomic bursts per lattice per step, _rk4cb_bwd_deferred_kernel first
    # runs the four stages' point-gradient sweeps (loads only), then
    # scatters all four at once over the union of their stencil boxes with
    # per-tap sums of the stages' weighted gradients. The union degrades
    # gracefully: stages that drifted into neighbouring cells widen the
    # loops by the drift and contribute zero weight outside their own box.
    # bench_cylindrical_rk4 (RTX 5090 shared with a fit, 330x200x200, 9
    # steps, 2.4M points, p50): cubic fwd 251 -> 77 ms and bwd 1063 -> 499 ms
    # from the layout, bwd 499 -> 124 ms from the deferral (its best case:
    # the benchmark's near-zero field never drifts stages apart), against
    # 34 / 211 ms for the trilinear kernels. The deferred kernel compiles to
    # 248 registers with no spills.

    _CYLB_ACC_CH = tl.constexpr(4)

    @triton.jit
    def _cylb_ring_phi(phi, nphi):
        # Continuous phi index on a ring of nphi cells: floor cell (as
        # int32, wrapped into [0, nphi) -- phi can round to exactly 2pi),
        # its fraction and the ring's index-per-radian scale.
        pscale = nphi.to(tl.float32) / _TWO_PI_F32
        pc = phi * pscale
        p0f = tl.math.floor(pc)
        p0 = p0f.to(tl.int32)
        p0 = tl.where(p0 >= nphi, p0 - nphi, p0)
        return p0, pc - p0f, pscale

    @triton.jit
    def _cylb_wrap(p, nphi):
        # Cyclic wrap of a tap index in [-nphi, 2*nphi) into [0, nphi) by
        # compare-and-add. Same result as the eager path's Python modulo for
        # every unpinned ring (all have >= 6 cells) without the integer
        # division; ring 0's taps are masked, so its index only has to be
        # finite.
        p = tl.where(p < 0, p + nphi, p)
        return tl.where(p >= nphi, p - nphi, p)

    @triton.jit
    def _cylb_wrap_diff(d, nphi):
        # Signed cyclic difference of two cell indices on a ring, in
        # [-nphi/2, nphi/2].
        half = nphi // 2
        d = tl.where(d > half, d - nphi, d)
        return tl.where(d < -half, d + nphi, d)

    @triton.jit
    def _cylb_select4(w0, w1, w2, w3, idx):
        # w[idx] for idx in 0..3, else 0: a stage's weight at a union tap
        # outside its own stencil box.
        return tl.where(idx == 0, w0,
                        tl.where(idx == 1, w1,
                                 tl.where(idx == 2, w2,
                                          tl.where(idx == 3, w3, 0.0))))

    @triton.jit
    def _cylb_load_tap(field_ptr, idx, mask):
        # The three components of flat cell idx from a channels-last slab.
        base = field_ptr + idx * 3
        a = tl.load(base, mask=mask, other=0.0)
        b = tl.load(base + 1, mask=mask, other=0.0)
        c = tl.load(base + 2, mask=mask, other=0.0)
        return a, b, c

    @triton.jit
    def _cylb_scatter_tap(acc_ptr, idx, az, ar, ap, mask):
        # One 16-byte vector atomic per tap into the padded channels-last
        # accumulator: the [BLOCK, 4] block is contiguous and 16-byte
        # aligned along its channel axis, so Triton emits atom.v4.f32; the
        # padding lane adds 0.
        lane = tl.arange(0, 4)[None, :]
        vals = tl.where(lane == 0, az[:, None],
                        tl.where(lane == 1, ar[:, None],
                                 tl.where(lane == 2, ap[:, None], 0.0)))
        ptrs = acc_ptr + (idx * _CYLB_ACC_CH)[:, None] + lane
        tl.atomic_add(ptrs, vals, mask=mask[:, None])

    @triton.jit
    def _cylb_sample_lattice(pz, radius, phi, field_ptr, num_phi_ptr,
                             offsets_ptr, NZ, NR, TOTAL, lane_mask):
        # field_ptr :: channels-last [nz, total_phi, 3] slab.
        zn = pz * 2.0 - 1.0
        zc = ((zn + 1.0) * 0.5) * (NZ - 1).to(tl.float32)
        zc = tl.minimum(tl.maximum(zc, 0.0), (NZ - 1).to(tl.float32))
        z0f = tl.math.floor(zc)
        z0 = z0f.to(tl.int32)
        wz0, wz1, wz2, wz3 = _bspline_weights(zc - z0f)

        rc = radius * (NR - 1).to(tl.float32)
        r0f = tl.math.floor(rc)
        r0 = r0f.to(tl.int32)
        wr0, wr1, wr2, wr3 = _bspline_weights(rc - r0f)
        v0 = tl.zeros(pz.shape, dtype=tl.float32)
        v1 = tl.zeros(pz.shape, dtype=tl.float32)
        v2 = tl.zeros(pz.shape, dtype=tl.float32)
        for dr in range(4):
            ring = tl.minimum(tl.maximum(r0 + (dr - 1), 0), NR - 1)
            wr = tl.where(dr == 0, wr0, tl.where(dr == 1, wr1, tl.where(dr == 2, wr2, wr3)))
            nphi = tl.load(num_phi_ptr + ring, mask=lane_mask, other=1).to(tl.int32)
            offset = tl.load(offsets_ptr + ring, mask=lane_mask, other=0).to(tl.int64)
            p0, fp, pscale = _cylb_ring_phi(phi, nphi)
            wp0, wp1, wp2, wp3 = _bspline_weights(fp)
            load_mask = lane_mask & (ring != 0)
            for dz in range(4):
                z = tl.minimum(tl.maximum(z0 + (dz - 1), 0), NZ - 1)
                wz = tl.where(dz == 0, wz0, tl.where(dz == 1, wz1, tl.where(dz == 2, wz2, wz3)))
                wrz = wr * wz
                row = z.to(tl.int64) * TOTAL + offset
                for dp in tl.static_range(4):
                    pp = _cylb_wrap(p0 + (dp - 1), nphi)
                    wp = wp0 if dp == 0 else (wp1 if dp == 1 else (wp2 if dp == 2 else wp3))
                    w = wrz * wp
                    a, b, c = _cylb_load_tap(field_ptr, row + pp, load_mask)
                    v0 += a * w
                    v1 += b * w
                    v2 += c * w
        return v0, v1, v2

    @triton.jit
    def _cylb_bwd_lattice(glz, glr, glp, pz, radius, phi,
                          field_ptr, num_phi_ptr, offsets_ptr, acc_ptr,
                          NZ, NR, TOTAL, HAS_ACC: tl.constexpr, lane_mask):
        # Same contract as _cyl_bwd_lattice for the cubic interpolant: the
        # sampled local components, the gradient w.r.t. the continuous z and
        # r lattice coordinates (already scaled to pz / radius) and w.r.t.
        # phi; dL/d(control points) scattered into the padded channels-last
        # acc. Tap-index clamping and wrapping are piecewise-constant (no
        # gradient); the z clamp contributes torch.clamp's inclusive
        # in-range mask.
        zn = pz * 2.0 - 1.0
        zraw = ((zn + 1.0) * 0.5) * (NZ - 1).to(tl.float32)
        zc = tl.minimum(tl.maximum(zraw, 0.0), (NZ - 1).to(tl.float32))
        z0f = tl.math.floor(zc)
        z0 = z0f.to(tl.int32)
        fz = zc - z0f
        wz0, wz1, wz2, wz3 = _bspline_weights(fz)
        dz0, dz1, dz2, dz3 = _bspline_dweights(fz)

        rc = radius * (NR - 1).to(tl.float32)
        r0f = tl.math.floor(rc)
        r0 = r0f.to(tl.int32)
        fr = rc - r0f
        wr0, wr1, wr2, wr3 = _bspline_weights(fr)
        dr0, dr1, dr2, dr3 = _bspline_dweights(fr)
        vz = tl.zeros(pz.shape, tl.float32)
        vr = tl.zeros(pz.shape, tl.float32)
        vp = tl.zeros(pz.shape, tl.float32)
        gzcoord = tl.zeros(pz.shape, tl.float32)
        grcoord = tl.zeros(pz.shape, tl.float32)
        gphi = tl.zeros(pz.shape, tl.float32)
        for dr in range(4):
            ring = tl.minimum(tl.maximum(r0 + (dr - 1), 0), NR - 1)
            wr = tl.where(dr == 0, wr0, tl.where(dr == 1, wr1, tl.where(dr == 2, wr2, wr3)))
            dwr = tl.where(dr == 0, dr0, tl.where(dr == 1, dr1, tl.where(dr == 2, dr2, dr3)))
            nphi = tl.load(num_phi_ptr + ring, mask=lane_mask, other=1).to(tl.int32)
            offset = tl.load(offsets_ptr + ring, mask=lane_mask, other=0).to(tl.int64)
            p0, fp, pscale = _cylb_ring_phi(phi, nphi)
            wp0, wp1, wp2, wp3 = _bspline_weights(fp)
            dp0, dp1, dp2, dp3 = _bspline_dweights(fp)
            value_mask = lane_mask & (ring != 0)
            for dz in range(4):
                z = tl.minimum(tl.maximum(z0 + (dz - 1), 0), NZ - 1)
                wz = tl.where(dz == 0, wz0, tl.where(dz == 1, wz1, tl.where(dz == 2, wz2, wz3)))
                dwz = tl.where(dz == 0, dz0, tl.where(dz == 1, dz1, tl.where(dz == 2, dz2, dz3)))
                row = z.to(tl.int64) * TOTAL + offset
                for dp in tl.static_range(4):
                    pp = _cylb_wrap(p0 + (dp - 1), nphi)
                    wp = wp0 if dp == 0 else (wp1 if dp == 1 else (wp2 if dp == 2 else wp3))
                    dwp = dp0 if dp == 0 else (dp1 if dp == 1 else (dp2 if dp == 2 else dp3))
                    w = (wr * wz) * wp
                    idx = row + pp
                    a, b, c = _cylb_load_tap(field_ptr, idx, value_mask)
                    vz += a * w
                    vr += b * w
                    vp += c * w
                    dot = (a * glz + b * glr) + c * glp
                    gzcoord += dot * ((dwz * wr) * wp)
                    grcoord += dot * ((wz * dwr) * wp)
                    gphi += (dot * ((wz * wr) * dwp)) * pscale
                    if HAS_ACC:
                        _cylb_scatter_tap(acc_ptr, idx, glz * w, glr * w, glp * w, value_mask)
        zmask = (zraw >= 0.0) & (zraw <= (NZ - 1).to(tl.float32))
        return (vz, vr, vp,
                gzcoord * zmask.to(tl.float32) * (NZ - 1).to(tl.float32),
                grcoord * (NR - 1).to(tl.float32), gphi)

    @triton.jit
    def _cylb_stage_local(gz, gy, gx, pz, py, px, NZ, NR):
        # Per-stage inputs of the joint scatter: the stage's upstream
        # gradient in the local (z, r, phi) basis, its z / r stencil cells
        # and fractions and its angle -- computed exactly as _cyl_bwd_stage
        # and _cylb_bwd_lattice do for the per-stage scatter.
        qy, qx, raw, radius, phi, sin_phi, cos_phi, on_axis = _cyl_coords(py, px)
        glr = gy * sin_phi + gx * cos_phi
        glp = gy * cos_phi - gx * sin_phi
        zn = pz * 2.0 - 1.0
        zc = ((zn + 1.0) * 0.5) * (NZ - 1).to(tl.float32)
        zc = tl.minimum(tl.maximum(zc, 0.0), (NZ - 1).to(tl.float32))
        z0f = tl.math.floor(zc)
        rc = radius * (NR - 1).to(tl.float32)
        r0f = tl.math.floor(rc)
        return gz, glr, glp, z0f.to(tl.int32), zc - z0f, r0f.to(tl.int32), rc - r0f, phi

    @triton.jit
    def _cylb_scatter_step(g1z, g1y, g1x, p1z, p1y, p1x,
                           g2z, g2y, g2x, p2z, p2y, p2x,
                           g3z, g3y, g3x, p3z, p3y, p3x,
                           g4z, g4y, g4x, p4z, p4y, p4x,
                           num_phi_ptr, offsets_ptr, acc_ptr, NZ, NR, TOTAL,
                           lane_mask):
        # Joint dL/d(control points) scatter of one RK4 step's four stages
        # (gradient g_s at stage point p_s) into one lattice's accumulator.
        # Each tap of the union of the stages' 4x4x4 boxes receives the sum
        # over stages of g_s * w_s(tap), w_s being zero outside stage s's own
        # box; so every stage's contribution is exactly the per-stage
        # scatter's, merged in registers before the atomic. Loop extents
        # are block-wide maxima of the per-lane union sizes; lanes with a
        # smaller union mask their extra taps off.
        az, ar, ap, az0, afz, ar0, afr, aphi = _cylb_stage_local(g1z, g1y, g1x, p1z, p1y, p1x, NZ, NR)
        bz, br, bp, bz0, bfz, br0, bfr, bphi = _cylb_stage_local(g2z, g2y, g2x, p2z, p2y, p2x, NZ, NR)
        cz, cr, cp, cz0, cfz, cr0, cfr, cphi = _cylb_stage_local(g3z, g3y, g3x, p3z, p3y, p3x, NZ, NR)
        dz, dr, dp, dz0, dfz, dr0, dfr, dphi = _cylb_stage_local(g4z, g4y, g4x, p4z, p4y, p4x, NZ, NR)
        awz0, awz1, awz2, awz3 = _bspline_weights(afz)
        bwz0, bwz1, bwz2, bwz3 = _bspline_weights(bfz)
        cwz0, cwz1, cwz2, cwz3 = _bspline_weights(cfz)
        dwz0, dwz1, dwz2, dwz3 = _bspline_weights(dfz)
        awr0, awr1, awr2, awr3 = _bspline_weights(afr)
        bwr0, bwr1, bwr2, bwr3 = _bspline_weights(bfr)
        cwr0, cwr1, cwr2, cwr3 = _bspline_weights(cfr)
        dwr0, dwr1, dwr2, dwr3 = _bspline_weights(dfr)

        zmin = tl.minimum(tl.minimum(az0, bz0), tl.minimum(cz0, dz0))
        zmax = tl.maximum(tl.maximum(az0, bz0), tl.maximum(cz0, dz0))
        rmin = tl.minimum(tl.minimum(ar0, br0), tl.minimum(cr0, dr0))
        rmax = tl.maximum(tl.maximum(ar0, br0), tl.maximum(cr0, dr0))
        # Stage box offsets within the union.
        azo, bzo, czo, dzo = az0 - zmin, bz0 - zmin, cz0 - zmin, dz0 - zmin
        aro, bro, cro, dro = ar0 - rmin, br0 - rmin, cr0 - rmin, dr0 - rmin
        ZE = tl.max(zmax - zmin, axis=0) + 4
        RE = tl.max(rmax - rmin, axis=0) + 4
        for j in range(RE):
            ring = tl.minimum(tl.maximum(rmin - 1 + j, 0), NR - 1)
            nphi = tl.load(num_phi_ptr + ring, mask=lane_mask, other=1).to(tl.int32)
            offset = tl.load(offsets_ptr + ring, mask=lane_mask, other=0).to(tl.int64)
            ari, bri, cri, dri = j - aro, j - bro, j - cro, j - dro
            awr = _cylb_select4(awr0, awr1, awr2, awr3, ari)
            bwr = _cylb_select4(bwr0, bwr1, bwr2, bwr3, bri)
            cwr = _cylb_select4(cwr0, cwr1, cwr2, cwr3, cri)
            dwr = _cylb_select4(dwr0, dwr1, dwr2, dwr3, dri)
            acov = (ari >= 0) & (ari < 4)
            bcov = (bri >= 0) & (bri < 4)
            ccov = (cri >= 0) & (cri < 4)
            dcov = (dri >= 0) & (dri < 4)
            # Each stage's phi cell on this ring's parameterisation; the
            # union along phi is built around stage 1's cell, with stages
            # not covering this ring pulled to it so they don't widen it.
            ap0, afp, apscale = _cylb_ring_phi(aphi, nphi)
            bp0, bfp, bpscale = _cylb_ring_phi(bphi, nphi)
            cp0, cfp, cpscale = _cylb_ring_phi(cphi, nphi)
            dp0, dfp, dpscale = _cylb_ring_phi(dphi, nphi)
            awp0, awp1, awp2, awp3 = _bspline_weights(afp)
            bwp0, bwp1, bwp2, bwp3 = _bspline_weights(bfp)
            cwp0, cwp1, cwp2, cwp3 = _bspline_weights(cfp)
            dwp0, dwp1, dwp2, dwp3 = _bspline_weights(dfp)
            bd = tl.where(bcov, _cylb_wrap_diff(bp0 - ap0, nphi), 0)
            cd = tl.where(ccov, _cylb_wrap_diff(cp0 - ap0, nphi), 0)
            dd = tl.where(dcov, _cylb_wrap_diff(dp0 - ap0, nphi), 0)
            pmin = tl.minimum(tl.minimum(bd, cd), tl.minimum(dd, 0))
            pmax = tl.maximum(tl.maximum(bd, cd), tl.maximum(dd, 0))
            apo, bpo, cpo, dpo = -pmin, bd - pmin, cd - pmin, dd - pmin
            pbase = ap0 + pmin - 1  # first union cell, like zmin - 1 / rmin - 1
            PE = tl.max(pmax - pmin, axis=0) + 4
            ring_mask = lane_mask & (ring != 0)
            for i in range(ZE):
                z = tl.minimum(tl.maximum(zmin - 1 + i, 0), NZ - 1)
                azi, bzi, czi, dzi = i - azo, i - bzo, i - czo, i - dzo
                awzr = _cylb_select4(awz0, awz1, awz2, awz3, azi) * awr
                bwzr = _cylb_select4(bwz0, bwz1, bwz2, bwz3, bzi) * bwr
                cwzr = _cylb_select4(cwz0, cwz1, cwz2, cwz3, czi) * cwr
                dwzr = _cylb_select4(dwz0, dwz1, dwz2, dwz3, dzi) * dwr
                acovz = acov & (azi >= 0) & (azi < 4)
                bcovz = bcov & (bzi >= 0) & (bzi < 4)
                ccovz = ccov & (czi >= 0) & (czi < 4)
                dcovz = dcov & (dzi >= 0) & (dzi < 4)
                row = z.to(tl.int64) * TOTAL + offset
                for k in range(PE):
                    pp = _cylb_wrap(pbase + k, nphi)
                    api, bpi, cpi, dpi = k - apo, k - bpo, k - cpo, k - dpo
                    aw = awzr * _cylb_select4(awp0, awp1, awp2, awp3, api)
                    bw = bwzr * _cylb_select4(bwp0, bwp1, bwp2, bwp3, bpi)
                    cw = cwzr * _cylb_select4(cwp0, cwp1, cwp2, cwp3, cpi)
                    dw = dwzr * _cylb_select4(dwp0, dwp1, dwp2, dwp3, dpi)
                    active = ring_mask & (
                        (acovz & (api >= 0) & (api < 4))
                        | (bcovz & (bpi >= 0) & (bpi < 4))
                        | (ccovz & (cpi >= 0) & (cpi < 4))
                        | (dcovz & (dpi >= 0) & (dpi < 4)))
                    sz = ((az * aw + bz * bw) + cz * cw) + dz * dw
                    sr = ((ar * aw + br * bw) + cr * cw) + dr * dw
                    sp = ((ap * aw + bp * bw) + cp * cw) + dp * dw
                    _cylb_scatter_tap(acc_ptr, row + pp, sz, sr, sp, active)

    @triton.jit
    def _rk4cb_bwd_deferred_kernel(grad_y_ptr, grad_pts_ptr, stages_ptr,
                                   lo_base, lo_num_phi_ptr, lo_offsets_ptr, lo_acc_base,
                                   loNZ, loNR, loTOTAL,
                                   hi_base, hi_num_phi_ptr, hi_offsets_ptr, hi_acc_base,
                                   hiNZ, hiNR, hiTOTAL,
                                   N, h, h_half, h_sixth, n_steps, num_slabs,
                                   REVERSE: tl.constexpr, BLOCK: tl.constexpr):
        # _rk4c_bwd_kernel (CUBIC, HAS_ACC) with the field-gradient scatter
        # deferred to one joint _cylb_scatter_step per lattice per RK4 step.
        # The point-gradient recursion is unchanged: each stage's
        # _cyl_bwd_stage runs without HAS_ACC, and the stage gradients /
        # points it consumed are handed to the joint scatter afterwards.
        pid = tl.program_id(0)
        i = pid * BLOCK + tl.arange(0, BLOCK)
        m = i < N
        lo_stride = loNZ.to(tl.int64) * loTOTAL * 3
        hi_stride = hiNZ.to(tl.int64) * hiTOTAL * 3
        lo_acc_stride = loNZ.to(tl.int64) * loTOTAL * _CYLB_ACC_CH
        hi_acc_stride = hiNZ.to(tl.int64) * hiTOTAL * _CYLB_ACC_CH
        gz = tl.load(grad_y_ptr + i * 3, mask=m, other=0.0)
        gy = tl.load(grad_y_ptr + i * 3 + 1, mask=m, other=0.0)
        gx = tl.load(grad_y_ptr + i * 3 + 2, mask=m, other=0.0)
        for step in range(n_steps * num_slabs - 1, -1, -1):
            slab = step // n_steps
            if REVERSE:
                slab = num_slabs - 1 - slab
            lo_ptr = lo_base + slab.to(tl.int64) * lo_stride
            hi_ptr = hi_base + slab.to(tl.int64) * hi_stride
            lo_acc_ptr = lo_acc_base + slab.to(tl.int64) * lo_acc_stride
            hi_acc_ptr = hi_acc_base + slab.to(tl.int64) * hi_acc_stride
            s1 = (step * 4) * N.to(tl.int64)
            s2 = (step * 4 + 1) * N.to(tl.int64)
            s3 = (step * 4 + 2) * N.to(tl.int64)
            s4 = (step * 4 + 3) * N.to(tl.int64)
            g6z, g6y, g6x = gz * h_sixth, gy * h_sixth, gx * h_sixth
            p4z = tl.load(stages_ptr + (s4 + i) * 3, mask=m, other=0.0)
            p4y = tl.load(stages_ptr + (s4 + i) * 3 + 1, mask=m, other=0.0)
            p4x = tl.load(stages_ptr + (s4 + i) * 3 + 2, mask=m, other=0.0)
            g4z, g4y, g4x = g6z, g6y, g6x
            b4z, b4y, b4x = _cyl_bwd_stage(
                g4z, g4y, g4x, p4z, p4y, p4x,
                lo_ptr, lo_num_phi_ptr, lo_offsets_ptr, lo_ptr, loNZ, loNR, loTOTAL,
                hi_ptr, hi_num_phi_ptr, hi_offsets_ptr, hi_ptr, hiNZ, hiNR, hiTOTAL,
                False, True, m)
            p3z = tl.load(stages_ptr + (s3 + i) * 3, mask=m, other=0.0)
            p3y = tl.load(stages_ptr + (s3 + i) * 3 + 1, mask=m, other=0.0)
            p3x = tl.load(stages_ptr + (s3 + i) * 3 + 2, mask=m, other=0.0)
            g3z, g3y, g3x = g6z * 2.0 + b4z * h, g6y * 2.0 + b4y * h, g6x * 2.0 + b4x * h
            b3z, b3y, b3x = _cyl_bwd_stage(
                g3z, g3y, g3x, p3z, p3y, p3x,
                lo_ptr, lo_num_phi_ptr, lo_offsets_ptr, lo_ptr, loNZ, loNR, loTOTAL,
                hi_ptr, hi_num_phi_ptr, hi_offsets_ptr, hi_ptr, hiNZ, hiNR, hiTOTAL,
                False, True, m)
            p2z = tl.load(stages_ptr + (s2 + i) * 3, mask=m, other=0.0)
            p2y = tl.load(stages_ptr + (s2 + i) * 3 + 1, mask=m, other=0.0)
            p2x = tl.load(stages_ptr + (s2 + i) * 3 + 2, mask=m, other=0.0)
            g2z, g2y, g2x = (g6z * 2.0 + b3z * h_half, g6y * 2.0 + b3y * h_half,
                             g6x * 2.0 + b3x * h_half)
            b2z, b2y, b2x = _cyl_bwd_stage(
                g2z, g2y, g2x, p2z, p2y, p2x,
                lo_ptr, lo_num_phi_ptr, lo_offsets_ptr, lo_ptr, loNZ, loNR, loTOTAL,
                hi_ptr, hi_num_phi_ptr, hi_offsets_ptr, hi_ptr, hiNZ, hiNR, hiTOTAL,
                False, True, m)
            p1z = tl.load(stages_ptr + (s1 + i) * 3, mask=m, other=0.0)
            p1y = tl.load(stages_ptr + (s1 + i) * 3 + 1, mask=m, other=0.0)
            p1x = tl.load(stages_ptr + (s1 + i) * 3 + 2, mask=m, other=0.0)
            g1z, g1y, g1x = g6z + b2z * h_half, g6y + b2y * h_half, g6x + b2x * h_half
            b1z, b1y, b1x = _cyl_bwd_stage(
                g1z, g1y, g1x, p1z, p1y, p1x,
                lo_ptr, lo_num_phi_ptr, lo_offsets_ptr, lo_ptr, loNZ, loNR, loTOTAL,
                hi_ptr, hi_num_phi_ptr, hi_offsets_ptr, hi_ptr, hiNZ, hiNR, hiTOTAL,
                False, True, m)
            _cylb_scatter_step(
                g1z, g1y, g1x, p1z, p1y, p1x, g2z, g2y, g2x, p2z, p2y, p2x,
                g3z, g3y, g3x, p3z, p3y, p3x, g4z, g4y, g4x, p4z, p4y, p4x,
                lo_num_phi_ptr, lo_offsets_ptr, lo_acc_ptr, loNZ, loNR, loTOTAL, m)
            _cylb_scatter_step(
                g1z, g1y, g1x, p1z, p1y, p1x, g2z, g2y, g2x, p2z, p2y, p2x,
                g3z, g3y, g3x, p3z, p3y, p3x, g4z, g4y, g4x, p4z, p4y, p4x,
                hi_num_phi_ptr, hi_offsets_ptr, hi_acc_ptr, hiNZ, hiNR, hiTOTAL, m)
            gz = ((gz + b4z) + b3z + b2z) + b1z
            gy = ((gy + b4y) + b3y + b2y) + b1y
            gx = ((gx + b4x) + b3x + b2x) + b1x
        tl.store(grad_pts_ptr + i * 3, gz, mask=m)
        tl.store(grad_pts_ptr + i * 3 + 1, gy, mask=m)
        tl.store(grad_pts_ptr + i * 3 + 2, gx, mask=m)
