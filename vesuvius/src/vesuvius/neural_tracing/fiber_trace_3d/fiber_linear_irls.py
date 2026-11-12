"""Shared linear winding IRLS, extracted from winding_fields/sheetgraph.py.

Source: origin/winding-field-exps, spiral-fitting/experiments/winding_fields/
sheetgraph.py::solve_l1. Arithmetic, float32 precision, weight schedule and
fixed-count CG are retained so production does not import an experiment file.
The source-file SHA256 is f625385ef5c8a60eea1cde3b150b89d83df69eb0603d2160ee7f18eda7d05b9f
(the same source binding as the recorded full-volume IRLS benchmark).
"""
import numpy as np
import torch


def round_components(x, component):
    """Historic component circular-phase removal, then nearest integer."""
    phase = np.arctan2(np.bincount(component, np.sin(2*np.pi*x)),
                       np.bincount(component, np.cos(2*np.pi*x))) / (2*np.pi)
    return np.rint(x-phase[component]).astype(np.int64)


def solve_l1(n, a, b, d, wt, prior, prior_w, dev='cuda', irls=15, cg_iters=300, eps=0.05, log=print, ineq=None, ineq_up=0.0):
    """min sum wt*|x_b - x_a - d| + prior_w*|x - prior|^2 via IRLS; CG on the GPU.
    Edges flagged in `ineq` are one-sided: cost wt*max(0, d - (x_b - x_a)) (step at least d)."""
    a = torch.from_numpy(a).long().to(dev); b = torch.from_numpy(b).long().to(dev)
    d = torch.from_numpy(d).float().to(dev); wt = torch.from_numpy(wt).float().to(dev)
    prior = torch.from_numpy(np.asarray(prior)).float().to(dev)
    pw = torch.from_numpy(np.broadcast_to(np.asarray(prior_w, dtype=np.float32), (n,)).copy()).to(dev)
    x = prior.clone()
    iq = torch.zeros_like(d, dtype=torch.bool) if ineq is None else torch.from_numpy(ineq).to(dev)
    for it in range(irls + 1):
        r = x[b] - x[a] - d
        c = wt if it == 0 else wt / torch.clamp(r.abs(), min=eps)
        if it > 0:
            c = torch.where(iq & (r >= 0), c * ineq_up, c)
        def A(v):
            f = c * (v[b] - v[a])
            out = pw * v
            out.index_add_(0, b, f); out.index_add_(0, a, -f)
            return out
        rhs = pw * prior
        f = c * d
        rhs.index_add_(0, b, f); rhs.index_add_(0, a, -f)
        diag = pw.clone(); diag.index_add_(0, a, c); diag.index_add_(0, b, c)
        rr = rhs - A(x); zz = rr / diag; p = zz.clone(); rz = (rr * zz).sum()
        for _ in range(cg_iters):
            Ap = A(p); al = rz / (p * Ap).sum().clamp(min=1e-20)
            x = x + al * p; rr = rr - al * Ap; zz = rr / diag
            rzn = (rr * zz).sum(); p = zz + (rzn / rz) * p; rz = rzn
        res = (x[b] - x[a] - d)
        res = torch.where(iq, torch.relu(-res), res.abs())
        if it % 5 == 0 or it == irls:
            log(f'  IRLS {it}: weighted L1 {float((wt * res).sum()):.0f}, edges off by >0.5: {float((res > 0.5).float().mean()):.4f}')
    rr = x[b] - x[a] - d
    return x.cpu().numpy(), torch.where(iq, torch.relu(-rr), rr.abs()).cpu().numpy()
