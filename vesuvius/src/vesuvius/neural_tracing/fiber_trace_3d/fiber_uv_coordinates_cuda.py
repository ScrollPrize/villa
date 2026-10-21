"""Optional CUDA implementation; imported only for FP32 CUDA backward."""
import torch
import triton
import triton.language as tl


@triton.jit
def _partials(A, D, W, S, T, Q, GU, GQ, GA, GD, P,
              N: tl.constexpr, K: tl.constexpr, BX: tl.constexpr, BW: tl.constexpr,
              PRECISE: tl.constexpr):
    x = tl.program_id(0) * BX + tl.arange(0, BX)
    j = tl.program_id(1) * BW + tl.arange(0, BW)
    valid = x < N
    a = tl.load(A+x, valid, 0)
    d = tl.load(D+x, valid, 0)
    s = tl.load(S+x, valid, 0)
    t = tl.load(T+x, valid, 0)
    q = tl.load(Q+x, valid, 0)
    gu = tl.load(GU+x, valid, 0)
    gq = tl.load(GQ+x, valid, 0)
    ws, wt = tl.load(W+s), tl.load(W+t)
    if PRECISE:
        h = tl.div_rn(gq, wt)
        same_q = tl.div_rn(gq*d, ws*ws)
    else:
        h = gq / wt
        same_q = gq*d / (ws*ws)
    g = gu + h
    if tl.program_id(1) == 0:
        tl.store(GA+x, g*ws, valid)
        tl.store(GD+x, g, valid)
    before_s = j[:, None] < s[None, :]
    before_t = j[:, None] < t[None, :]
    at_s = j[:, None] == s[None, :]
    at_t = j[:, None] == t[None, :]
    prefix = gu[None, :] * before_s
    cross = h[None, :] * (before_s.to(tl.float32)-before_t.to(tl.float32))
    cross += (g*(a-s))[None, :] * at_s
    cross -= (h*(q-t))[None, :] * at_t
    same = (gu*(a-s)-same_q)[None, :] * at_s
    v = prefix + tl.where((s == t)[None, :], same, cross)
    reduced = tl.sum(tl.where(valid[None, :], v, 0), axis=1)
    tl.store(P + tl.program_id(0)*K+j, reduced, j < K)


@triton.jit
def _finish(P, WGRAD, K: tl.constexpr, PARTS: tl.constexpr, BLOCK: tl.constexpr):
    j = tl.program_id(0)
    x = tl.arange(0, BLOCK)
    v = tl.load(P+x*K+j, x < PARTS, 0)
    tl.store(WGRAD+j, tl.sum(v, axis=0))


def backward(a, d, w, s, t, q, gu, gq, division):
    n, k = a.numel(), w.numel()
    if n == 0:
        return torch.empty_like(a), torch.empty_like(d), torch.zeros_like(w)
    parts = triton.cdiv(n, 256)
    partial = torch.empty((parts, k), device=a.device, dtype=a.dtype)
    ga, gd, gw = torch.empty_like(a), torch.empty_like(d), torch.empty_like(w)
    _partials[(parts, triton.cdiv(k, 16))](
        a, d, w, s, t, q, gu, gq, ga, gd, partial, n, k, 256, 16, division == 'precise',
        num_warps=8, enable_fp_fusion=False)
    _finish[(k,)](partial, gw, k, parts, triton.next_power_of_2(parts),
                   enable_fp_fusion=False)
    return ga, gd, gw
