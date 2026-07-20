"""Fused Triton kernels for the time-invariant RK4 flow integration.

The eager path (flow_fields._RK4SparseFlowIntegrate) launches ~10 kernels per
stage sample in forward (two grid_samples plus stage arithmetic) and ~25 in
backward (_sparse_backward_impl: corner gathers, weight products, reductions,
index_add_). Per point the whole integration is independent, so the entire
forward (n_steps x 4 stage samples) runs as ONE kernel here, and the entire
adjoint sweep as ONE kernel, with all intermediates kept in registers.

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


def rk4_triton_available(*tensors):
    if not _HAS_TRITON or os.environ.get('FIT_SPIRAL_TRITON', '1') == '0':
        return False
    return all(
        t.is_cuda and t.dtype == torch.float32 for t in tensors if t is not None
    )


def direct_lr_enabled():
    # Sample the LR flow lattice directly at query points instead of
    # upsampling it to HR resolution every step. Same function except within
    # HR cells that straddle an LR grid plane (the upsample linearises there);
    # kills the F.interpolate forward+backward and its full-resolution
    # transient. Off by default because it slightly changes the effective
    # interpolant.
    return os.environ.get('FIT_SPIRAL_DIRECT_LR', '0') == '1'


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
                        low_ptr, high_ptr, scale,
                        N, Z, Y, X, zm1f, ym1f, xm1f,
                        h, h_half, h_sixth, n_steps,
                        STORE_STAGES: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        i = pid * BLOCK + tl.arange(0, BLOCK)
        m = i < N
        ch = tl.full((), Z, tl.int64) * Y * X
        yz = tl.load(y_ptr + i * 3 + 0, mask=m, other=0.0)
        yy = tl.load(y_ptr + i * 3 + 1, mask=m, other=0.0)
        yx = tl.load(y_ptr + i * 3 + 2, mask=m, other=0.0)
        for step in range(n_steps):
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
                        low_ptr, high_ptr, acc_ptr, scale,
                        N, Z, Y, X, zm1f, ym1f, xm1f, zm2f, ym2f, xm2f,
                        h, h_half, h_sixth, n_steps,
                        HAS_ACC: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        i = pid * BLOCK + tl.arange(0, BLOCK)
        m = i < N
        ch = tl.full((), Z, tl.int64) * Y * X
        gz = tl.load(grad_y_ptr + i * 3 + 0, mask=m, other=0.0)
        gy = tl.load(grad_y_ptr + i * 3 + 1, mask=m, other=0.0)
        gx = tl.load(grad_y_ptr + i * 3 + 2, mask=m, other=0.0)
        for step in range(n_steps - 1, -1, -1):
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
                         lo_ptr, lo_az, lo_bz, lo_ay, lo_by, lo_ax, lo_bx,
                         loZ, loY, loX, lo_scale,
                         hi_ptr, hi_az, hi_bz, hi_ay, hi_by, hi_ax, hi_bx,
                         hiZ, hiY, hiX, hi_scale,
                         N, h, h_half, h_sixth, n_steps,
                         STORE_STAGES: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        i = pid * BLOCK + tl.arange(0, BLOCK)
        m = i < N
        yz = tl.load(y_ptr + i * 3 + 0, mask=m, other=0.0)
        yy = tl.load(y_ptr + i * 3 + 1, mask=m, other=0.0)
        yx = tl.load(y_ptr + i * 3 + 2, mask=m, other=0.0)
        for step in range(n_steps):
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
                         lo_ptr, acc_lo_ptr, lo_az, lo_bz, lo_ay, lo_by, lo_ax, lo_bx,
                         loZ, loY, loX, lo_scale,
                         hi_ptr, acc_hi_ptr, hi_az, hi_bz, hi_ay, hi_by, hi_ax, hi_bx,
                         hiZ, hiY, hiX, hi_scale,
                         N, h, h_half, h_sixth, n_steps,
                         HAS_ACC: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        i = pid * BLOCK + tl.arange(0, BLOCK)
        m = i < N
        gz = tl.load(grad_y_ptr + i * 3 + 0, mask=m, other=0.0)
        gy = tl.load(grad_y_ptr + i * 3 + 1, mask=m, other=0.0)
        gx = tl.load(grad_y_ptr + i * 3 + 2, mask=m, other=0.0)
        for step in range(n_steps - 1, -1, -1):
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
                                  low_ptr, high_ptr, acc_ptr, scale,
                                  N, Z, Y, X, zm1f, ym1f, xm1f,
                                  zm2f, ym2f, xm2f,
                                  h, h_half, h_sixth, n_steps,
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
        for step in range(n_steps - 1, -1, -1):
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
        _flush_cell(acc_ptr, ch, Y, X, acz, acy, acx, m & have,
                    rz0, rz1, rz2, rz3, rz4, rz5, rz6, rz7,
                    ry0, ry1, ry2, ry3, ry4, ry5, ry6, ry7,
                    rx0, rx1, rx2, rx3, rx4, rx5, rx6, rx7)
        tl.store(grad_pts_ptr + i * 3 + 0, gz, mask=m)
        tl.store(grad_pts_ptr + i * 3 + 1, gy, mask=m)
        tl.store(grad_pts_ptr + i * 3 + 2, gx, mask=m)


_BLOCK = 128


def coalesce_enabled():
    # Deferred (register-coalesced) field-gradient atomics in the RK4
    # adjoint. The backward is ~97% atomic-bound (bench_rk4 2026-07-18:
    # 14.7 ms with acc vs 0.47 ms without at 2.4M pts); consecutive stage
    # points usually share a lattice cell, so deferring turns up to
    # 4*n_steps atomic bursts into one. Point gradients bitwise identical;
    # acc changes only in addend association (atomics-order class).
    return os.environ.get('FIT_SPIRAL_RK4_COALESCE', '1') != '0'


def _launch_args(field):
    Z, Y, X = field.shape[1], field.shape[2], field.shape[3]
    return (
        Z, Y, X,
        float(Z - 1), float(Y - 1), float(X - 1),
    )


def _run_fwd_kernel(y0, low_field, high_field, high_scale, h, n_steps, stages):
    n = y0.shape[0]
    out = torch.empty_like(y0)
    Z, Y, X, zm1f, ym1f, xm1f = _launch_args(low_field)
    if n > 0:
        _rk4_fwd_kernel[(triton.cdiv(n, _BLOCK),)](
            y0, out, stages if stages is not None else out,
            low_field, high_field, float(high_scale),
            n, Z, Y, X, zm1f, ym1f, xm1f,
            float(h), float(h / 2), float(h / 6), int(n_steps),
            STORE_STAGES=stages is not None, BLOCK=_BLOCK,
        )
    return out


def rk4_integrate(y0, low_field, high_field, high_scale, acc, h, n_steps):
    # Entry point mirroring flow_fields._RK4SparseFlowIntegrate.apply. The
    # grad-vs-inference split lives here because Function.forward always runs
    # with grad disabled, so it cannot decide itself whether stage points for
    # the adjoint sweep must be kept.
    if torch.is_grad_enabled() and y0.requires_grad:
        return TritonRK4Integrate.apply(
            y0, low_field, high_field, high_scale, acc, h, n_steps)
    return _run_fwd_kernel(
        y0.contiguous(), low_field, high_field, high_scale, h, n_steps, None)


class TritonRK4Integrate(torch.autograd.Function):
    # Drop-in replacement for flow_fields._RK4SparseFlowIntegrate; same
    # arguments, same saved-state footprint (the 4*n_steps stage points), same
    # accumulator contract for the field gradient.

    @staticmethod
    def forward(ctx, y0, low_field, high_field, high_scale, acc, h, n_steps):
        ctx.set_materialize_grads(False)
        y0 = y0.contiguous()
        perm = None
        if acc is not None and permute_enabled() and y0.shape[0] > _PERM_STRIDE:
            perm = _interleave_perm(y0.shape[0], y0.device)
            y0 = y0[perm].contiguous()
        stages = torch.empty(
            int(n_steps) * 4, y0.shape[0], 3, device=y0.device, dtype=y0.dtype)
        out = _run_fwd_kernel(
            y0, low_field, high_field, high_scale, h, n_steps, stages)
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
        return out

    @staticmethod
    def backward(ctx, grad_y):
        if grad_y is None:
            return None, None, None, None, None, None, None
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
        if n > 0 and acc is not None and coalesce_enabled():
            _rk4_bwd_coalesced_kernel[(triton.cdiv(n, _BLOCK),)](
                grad_y, grad_pts, stages,
                low_field, high_field, acc, ctx.high_scale,
                n, Z, Y, X, zm1f, ym1f, xm1f,
                float(Z - 2), float(Y - 2), float(X - 2),
                h, float(h / 2), float(h / 6), ctx.n_steps,
                BLOCK=_BLOCK,
            )
        elif n > 0:
            _rk4_bwd_kernel[(triton.cdiv(n, _BLOCK),)](
                grad_y, grad_pts, stages,
                low_field, high_field,
                acc if acc is not None else low_field, ctx.high_scale,
                n, Z, Y, X, zm1f, ym1f, xm1f,
                float(Z - 2), float(Y - 2), float(X - 2),
                h, float(h / 2), float(h / 6), ctx.n_steps,
                HAS_ACC=acc is not None, BLOCK=_BLOCK,
            )
        if perm is not None:
            unperm = torch.empty_like(grad_pts)
            unperm[perm] = grad_pts
            grad_pts = unperm
        return grad_pts, None, None, None, None, None, None


def _field_geoms(low_field, high_field):
    # (a, b) per dim per field s.t. lattice coord = normalised_point * a + b.
    geom = []
    for l, hh in zip(low_field.shape[1:], high_field.shape[1:]):
        if l == hh:
            geom += [float(hh - 1), 0.0]
        else:
            # grid_sample(align_corners=True) onto the HR lattice composed
            # with F.interpolate(align_corners=False)'s source-index map.
            s = l / hh
            geom += [float((hh - 1) * s), float(0.5 * s - 0.5)]
    for hh in high_field.shape[1:]:
        geom += [float(hh - 1), 0.0]
    return geom  # lo az,bz,ay,by,ax,bx then hi az,bz,ay,by,ax,bx


def _run_direct_fwd(y0, low, high, lo_scale, hi_scale, h, n_steps, stages):
    n = y0.shape[0]
    out = torch.empty_like(y0)
    if n > 0:
        g = _field_geoms(low, high)
        _rk4d_fwd_kernel[(triton.cdiv(n, _BLOCK),)](
            y0, out, stages if stages is not None else out,
            low, *g[:6], low.shape[1], low.shape[2], low.shape[3], float(lo_scale),
            high, *g[6:], high.shape[1], high.shape[2], high.shape[3], float(hi_scale),
            n, float(h), float(h / 2), float(h / 6), int(n_steps),
            STORE_STAGES=stages is not None, BLOCK=_BLOCK,
        )
    return out


def rk4_direct_integrate(y0, low, high, lo_scale, hi_scale, acc_lo, acc_hi,
                         h, n_steps):
    if torch.is_grad_enabled() and y0.requires_grad:
        return TritonRK4DirectIntegrate.apply(
            y0, low, high, lo_scale, hi_scale, acc_lo, acc_hi, h, n_steps)
    return _run_direct_fwd(
        y0.contiguous(), low, high, lo_scale, hi_scale, h, n_steps, None)


class TritonRK4DirectIntegrate(torch.autograd.Function):
    # RK4 integration sampling the LR lattice directly (no per-step upsample).
    # Field gradients are scattered unscaled into the two caller-owned
    # accumulators; the caller applies each field's value scale afterwards
    # (see CartesianFlowField.apply_accumulated_field_grad).

    @staticmethod
    def forward(ctx, y0, low, high, lo_scale, hi_scale, acc_lo, acc_hi, h, n_steps):
        ctx.set_materialize_grads(False)
        y0 = y0.contiguous()
        stages = torch.empty(
            int(n_steps) * 4, y0.shape[0], 3, device=y0.device, dtype=y0.dtype)
        out = _run_direct_fwd(y0, low, high, lo_scale, hi_scale, h, n_steps, stages)
        ctx.save_for_backward(low, high, stages)
        ctx.scales = (float(lo_scale), float(hi_scale))
        ctx.accs = (acc_lo, acc_hi)
        ctx.h = float(h)
        ctx.n_steps = int(n_steps)
        return out

    @staticmethod
    def backward(ctx, grad_y):
        if grad_y is None:
            return None, None, None, None, None, None, None, None, None
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
                low.shape[1], low.shape[2], low.shape[3], lo_scale,
                high, acc_hi if acc_hi is not None else high, *g[6:],
                high.shape[1], high.shape[2], high.shape[3], hi_scale,
                n, h, float(h / 2), float(h / 6), ctx.n_steps,
                HAS_ACC=acc_lo is not None, BLOCK=_BLOCK,
            )
        return grad_pts, None, None, None, None, None, None, None, None
