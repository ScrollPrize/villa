"""Tiled iBOT projection/reductions, with the existing half-precision GEMM contract.

No FP32 [samples, prototypes] logits or targets are stored. Backward temporarily
stores one row chunk of half-precision logit gradients for cuBLAS dX/dW GEMMs.
Prototype reductions use deterministic partial buffers, not atomic additions.
"""
import torch
import torch.distributed as dist

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None


def supported(features, weight):
    return (triton is not None and features.is_cuda and features.dtype in (torch.float16, torch.bfloat16)
            and weight.dtype == features.dtype and features.is_contiguous() and weight.is_contiguous()
            and torch.cuda.get_device_capability(features.device)[0] >= 8)


if triton is not None:
    @triton.jit
    def _projection(X, W, m, n, M, K: tl.constexpr, D: tl.constexpr,
                    BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr):
        d = tl.arange(0, BD)
        acc = tl.full((BM, BN), 0., tl.float32)
        for start in range(tl.cdiv(D, BD)):
            dd = start * BD + d
            x = tl.load(X + m[:, None] * D + dd[None, :], (m[:, None] < M) & (dd[None, :] < D), other=0)
            w = tl.load(W + n[None, :] * D + dd[:, None], (n[None, :] < K) & (dd[:, None] < D), other=0)
            acc = tl.dot(x, w, acc)
        # torch.mm under AMP rounds its output BEFORE the loss promotes to float32.
        return acc.to(X.dtype.element_ty).to(tl.float32)

    @triton.jit
    def _sinkhorn(X, W, Scales, Partial, M, K: tl.constexpr, D: tl.constexpr,
                  TEMP, MODE: tl.constexpr,
                  BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr):
        im, ik = tl.program_id(0), tl.program_id(1)
        m, k = im * BM + tl.arange(0, BM), ik * BN + tl.arange(0, BN)
        logits = _projection(X, W, m, k, M, K, D, BM, BN, BD) / TEMP
        e = tl.where((m[:, None] < M) & (k[None, :] < K), tl.exp(logits), 0.)
        if MODE == 1:
            b = tl.load(Scales + k, k < K, other=0.)
            sums = tl.sum(e * b[None, :], 1)
            tl.store(Partial + m * tl.cdiv(K, BN) + ik, sums, m < M)
        else:
            if MODE == 2:
                a = tl.load(Scales + m, m < M, other=0.)
                e = e * a[:, None]
            sums = tl.sum(e, 0)
            tl.store(Partial + im * K + k, sums, k < K)

    @triton.jit
    def _loss_stats(S, SW, T, TW, LogB, Stats, M, K: tl.constexpr, D: tl.constexpr,
                    ST, TT,
                    BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr):
        im, ik = tl.program_id(0), tl.program_id(1)
        m, k = im * BM + tl.arange(0, BM), ik * BN + tl.arange(0, BN)
        s = _projection(S, SW, m, k, M, K, D, BM, BN, BD) / ST
        t = _projection(T, TW, m, k, M, K, D, BM, BN, BD) / TT
        t = t + tl.load(LogB + k, k < K, other=0.)[None, :]
        s = tl.where(k[None, :] < K, s, -float("inf"))
        t = tl.where(k[None, :] < K, t, -float("inf"))
        sm, tm = tl.max(s, 1), tl.max(t, 1)
        se, te = tl.exp(s - sm[:, None]), tl.exp(t - tm[:, None])
        ss, ts = tl.sum(se, 1), tl.sum(te, 1)
        weighted = tl.sum(te * tl.where(k[None, :] < K, s, 0.), 1)
        offset = m * tl.cdiv(K, BN) + ik
        stride = M * tl.cdiv(K, BN)
        tl.store(Stats + offset, sm, m < M)
        tl.store(Stats + stride + offset, ss, m < M)
        tl.store(Stats + 2 * stride + offset, tm, m < M)
        tl.store(Stats + 3 * stride + offset, ts, m < M)
        tl.store(Stats + 4 * stride + offset, weighted, m < M)

    @triton.jit
    def _finish_stats(Stats, Rows, SLse, TLse, M, NT: tl.constexpr, BLOCK: tl.constexpr):
        m = tl.program_id(0)
        k = tl.arange(0, BLOCK)
        stride = M * NT
        offset = m * NT + k
        sm = tl.load(Stats + offset, k < NT, other=-float("inf"))
        ss = tl.load(Stats + stride + offset, k < NT, other=0.)
        tm = tl.load(Stats + 2 * stride + offset, k < NT, other=-float("inf"))
        ts = tl.load(Stats + 3 * stride + offset, k < NT, other=0.)
        tw = tl.load(Stats + 4 * stride + offset, k < NT, other=0.)
        smax, tmax = tl.max(sm, 0), tl.max(tm, 0)
        tsums = tl.sum(tl.exp(tm - tmax) * ts, 0)
        slse = smax + tl.log(tl.sum(tl.exp(sm - smax) * ss, 0))
        tlse = tmax + tl.log(tsums)
        likelihood = tl.sum(tl.exp(tm - tmax) * tw, 0) / tsums - slse
        tl.store(Rows + m, likelihood)
        tl.store(SLse + m, slse)
        tl.store(TLse + m, tlse)

    @triton.jit
    def _logit_grad(S, SW, T, TW, LogB, SLse, TLse, GradRows, Grad,
                    M, K: tl.constexpr, D: tl.constexpr,
                    ST, TT,
                    BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr):
        m = tl.program_id(0) * BM + tl.arange(0, BM)
        k = tl.program_id(1) * BN + tl.arange(0, BN)
        s = _projection(S, SW, m, k, M, K, D, BM, BN, BD) / ST
        t = _projection(T, TW, m, k, M, K, D, BM, BN, BD) / TT
        t += tl.load(LogB + k, k < K, other=0.)[None, :]
        ps = tl.exp(s - tl.load(SLse + m, m < M, other=0.)[:, None])
        pt = tl.exp(t - tl.load(TLse + m, m < M, other=0.)[:, None])
        grad = (pt - ps) * (tl.load(GradRows + m, m < M, other=0.) / ST)[:, None]
        tl.store(Grad + m[:, None] * K + k[None, :], grad, (m[:, None] < M) & (k[None, :] < K))


_BM, _BN, _BD = 32, 128, 64


def _reduce_projection(features, weight, temp, scales, mode):
    m, d = features.shape
    k = weight.shape[0]
    shape = (m, triton.cdiv(k, _BN)) if mode == 1 else (triton.cdiv(m, _BM), k)
    partial = torch.empty(shape, device=features.device, dtype=torch.float32)
    if m:
        _sinkhorn[(triton.cdiv(m, _BM), triton.cdiv(k, _BN))](
            features, weight, scales, partial, m, k, d, temp, mode, _BM, _BN, _BD,
            num_warps=4,
        )
    return partial.sum(dim=1 if mode == 1 else 0)


@torch.no_grad()
def prototype_scales(features, weight, temp, sample_count, n_iterations):
    count = sample_count.to(device=features.device, dtype=torch.float32).clone()
    if dist.is_initialized():
        dist.all_reduce(count)
    k = weight.shape[0]

    def columns(scales, mode):
        sums = _reduce_projection(features, weight, temp, scales, mode)
        if dist.is_initialized():
            dist.all_reduce(sums)
        return sums

    b = 1. / (columns(features, 0) * k)
    for _ in range(n_iterations - 1):
        a = 1. / (_reduce_projection(features, weight, temp, b, 1) * count)
        sums = columns(a, 2)
        b = b / (sums * b * k)
    return b.log()


class ProjectionLogLikelihood(torch.autograd.Function):
    @staticmethod
    def forward(ctx, s, sw, t, tw, log_b, st, tt, chunk_size):
        n, d = s.shape
        k = sw.shape[0]
        rows = s.new_empty(n, dtype=torch.float32)
        slse, tlse = torch.empty_like(rows), torch.empty_like(rows)
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            m = end - start
            nt = triton.cdiv(k, _BN)
            stats = s.new_empty((5, m, nt), dtype=torch.float32)
            _loss_stats[(triton.cdiv(m, _BM), nt)](
                s[start:end], sw, t[start:end], tw, log_b, stats, m, k, d, st, tt, _BM, _BN, _BD,
                num_warps=4,
            )
            _finish_stats[(m,)](stats, rows[start:end], slse[start:end], tlse[start:end],
                                m, nt, triton.next_power_of_2(nt))
            del stats
        ctx.save_for_backward(s, sw, t, tw, log_b, slse, tlse)
        ctx.st, ctx.tt, ctx.chunk_size = st, tt, chunk_size
        return rows

    @staticmethod
    def backward(ctx, grad_rows):
        s, sw, t, tw, log_b, slse, tlse = ctx.saved_tensors
        n, d = s.shape
        k = sw.shape[0]
        dx = torch.empty_like(s)
        dw = sw.new_zeros(sw.shape, dtype=torch.float32)
        grad_rows = grad_rows.contiguous()
        # cuBLAS receives the same half-precision dlogits as the existing backend.
        with torch.autocast("cuda", enabled=False):
            for start in range(0, n, ctx.chunk_size):
                end = min(start + ctx.chunk_size, n)
                m = end - start
                dg = s.new_empty((m, k))
                _logit_grad[(triton.cdiv(m, _BM), triton.cdiv(k, _BN))](
                    s[start:end], sw, t[start:end], tw, log_b, slse[start:end], tlse[start:end],
                    grad_rows[start:end], dg, m, k, d, ctx.st, ctx.tt, _BM, _BN, _BD, num_warps=4,
                )
                dx[start:end] = torch.mm(dg, sw)
                dw += torch.mm(dg.t(), s[start:end]).float()
                del dg
        return dx, dw.to(sw.dtype), None, None, None, None, None, None
