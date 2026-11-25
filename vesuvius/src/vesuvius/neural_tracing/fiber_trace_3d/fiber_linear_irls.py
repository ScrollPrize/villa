"""Linear winding IRLS in double precision with fixed-count preconditioned CG.

Single precision is not enough: with weakly attached groups next to strongly weighted ones the CG loses the
true residual and leaves those groups at arbitrary windings. Every round therefore measures how far its
solution still is from solving its linear system, per node, and the solve fails when a node is left unsettled."""
import numpy as np
import torch


def round_components(x, component):
    """Historic component circular-phase removal, then nearest integer."""
    phase = np.arctan2(np.bincount(component, np.sin(2*np.pi*x)),
                       np.bincount(component, np.cos(2*np.pi*x))) / (2*np.pi)
    return np.rint(x-phase[component]).astype(np.int64)


def solve_l1(n, a, b, d, wt, prior, prior_w, dev='cuda', irls=15, cg_iters=300, eps=0.05, log=print, ineq=None, ineq_up=0.0,
             max_node_update=0.05, stats=None):
    """min sum wt*|x_b - x_a - d| + prior_w*|x - prior|^2 via IRLS; CG on the GPU.
    Edges flagged in `ineq` are one-sided: cost wt*max(0, d - (x_b - x_a)) (step at least d).

    Every round's linear solve is checked with the true residual rhs - A x divided by the diagonal: the
    distance, in windings, one more Jacobi step would still move each node. A weakly attached group that
    the CG left behind shows there while it vanishes in the residual norm. A round whose largest node
    update exceeds `max_node_update` raises (None switches the check off); `stats`, a dict, receives the
    per-round values (`node_update`, `solution_range`) and their worst (`largest_node_update`,
    `largest_solution_range`)."""
    a = torch.from_numpy(a).long().to(dev); b = torch.from_numpy(b).long().to(dev)
    d = torch.from_numpy(d).double().to(dev); wt = torch.from_numpy(wt).double().to(dev)
    prior = torch.from_numpy(np.asarray(prior)).double().to(dev)
    pw = torch.from_numpy(np.broadcast_to(np.asarray(prior_w, dtype=np.float64), (n,)).copy()).to(dev)
    x = prior.clone()
    iq = torch.zeros_like(d, dtype=torch.bool) if ineq is None else torch.from_numpy(ineq).to(dev)
    residuals = []; ranges = []
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
        reached = float(((rhs - A(x)) / diag).abs().max())
        residuals.append(reached); ranges.append(float(x.max() - x.min()))
        if stats is not None:
            stats.update(node_update=list(residuals), solution_range=list(ranges),
                         largest_node_update=max(residuals), largest_solution_range=max(ranges))
        if max_node_update is not None and not reached <= max_node_update:
            raise RuntimeError(f'IRLS round {it}: the linear solve did not converge (a node is still {reached:.3g} '
                               f'windings from its solution, limit {max_node_update:g}; solution range {ranges[-1]:.0f})')
        res = (x[b] - x[a] - d)
        res = torch.where(iq, torch.relu(-res), res.abs())
        if it % 5 == 0 or it == irls:
            log(f'  IRLS {it}: weighted L1 {float((wt * res).sum()):.0f}, edges off by >0.5: {float((res > 0.5).double().mean()):.4f}, '
                f'largest node update {reached:.1e}, solution range {ranges[-1]:.0f}')
    rr = x[b] - x[a] - d
    return x.cpu().numpy(), torch.where(iq, torch.relu(-rr), rr.abs()).cpu().numpy()
