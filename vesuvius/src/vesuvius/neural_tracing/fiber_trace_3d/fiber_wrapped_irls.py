"""Periodic L1 graph fitting with per-node rejection, using matrix-free IRLS.

Phase has period two: integer-even/odd positions are the two classes. Wrapping
belongs to the residual, never to stored labels or CG's linear operator. The
epsilon treatment is Huber L1, with IRLS weights 1/max(abs(residual), epsilon).
The complete objective is nonconvex. Each local quadratic touches/majorizes the
wrapped Huber objective. Keep updates also use a touching quadratic majorizer.
"""
import time

import torch


def residual(delta, target=0.):
    return torch.remainder(delta - target + 1., 2.) - 1.


def l1_cost(r, epsilon=0.):
    a = r.abs()
    if epsilon == 0:
        return a
    if epsilon < 0:
        raise ValueError('epsilon must be nonnegative')
    return torch.where(a < epsilon, .5 * r.square() / epsilon, a - .5 * epsilon)


def class_score(phase):
    """Circular interpolation between labels; NOT a calibrated probability."""
    return residual(phase).abs()


def objective(phase, keep, i, j, target, weight, prior_target, prior_weight,
              prior_constant, rejection, *, mask_prior, epsilon):
    pair = (weight * keep[i] * keep[j] * l1_cost(residual(phase[j]-phase[i], target), epsilon)).sum()
    unary = prior_weight * l1_cost(residual(phase, prior_target), epsilon) + prior_constant
    prior = (unary * keep).sum() if mask_prior else unary.sum()
    broken = (rejection * (1-keep)).sum()
    return pair + prior + broken, (pair, prior, broken)


def _scatter_pair(values, i, j, n, *, difference=False):
    out = values.new_zeros(n)
    out.index_add_(0, i, -values if difference else values)
    out.index_add_(0, j, values)
    return out


def quadratic_step(phase, i, j, pair_target, pair_weight, unary_target,
                   unary_weight, *, max_cg=100, cg_tolerance=1e-5):
    """Jacobi-preconditioned CG, initialized at the current continuous phase."""
    n = phase.numel()
    # Self edges have no phase derivative (but do affect per-node rejection).
    c = torch.where(i == j, 0., pair_weight)
    diag = unary_weight + _scatter_pair(c, i, j, n)

    def matvec(v):
        return unary_weight*v + _scatter_pair(c*(v[j]-v[i]), i, j, n, difference=True)

    rhs = unary_weight*unary_target + _scatter_pair(c*pair_target, i, j, n, difference=True)
    from .fiber_matrix_free_cg import solve
    return solve(matvec,rhs,phase,diag,iterations=max_cg,tolerance=cg_tolerance)


def keep_step(phase, keep, i, j, target, weight, prior_target, prior_weight,
              prior_constant, rejection, *, mask_prior, epsilon):
    """Parallel bounded update; diagonal majorizer covers cross-node coupling."""
    edge = weight*l1_cost(residual(phase[j]-phase[i], target), epsilon)
    grad = -rejection.clone()
    grad.index_add_(0, i, edge*keep[j])
    grad.index_add_(0, j, edge*keep[i])
    if mask_prior:
        grad += prior_weight*l1_cost(residual(phase, prior_target), epsilon)+prior_constant
    diag = _scatter_pair(edge, i, j, keep.numel())
    candidate = (keep-grad/diag.clamp_min(torch.finfo(keep.dtype).tiny)).clamp(0, 1)
    return torch.where(diag > 0, candidate, torch.where(grad < 0, 1., torch.where(grad > 0, 0., keep)))


@torch.no_grad()
def solve(phase, keep, i, j, target, weight, prior_target, prior_weight,
          prior_constant, rejection, *, mask_prior=False, epsilon=.05,
          iterations=100, max_cg=100, cg_tolerance=1e-5, tolerance=1e-7,
          callback=None, log=None, fixed_keep=False):
    """All graph arrays are fixed device tensors; no per-fiber Python loops.

    Backtracking rejects finite-CG/roundoff increases of the actual periodic
    objective. It does not change the objective or retain old wrapping choices.
    """
    if not 0 < epsilon < 1 or iterations < 1 or max_cg < 1 or cg_tolerance <= 0 or tolerance < 0:
        raise ValueError('invalid IRLS settings')
    if phase.ndim != 1 or keep.shape != phase.shape or not phase.numel():
        raise ValueError('expected nonempty node vectors')
    for x in (phase, keep, target, weight, prior_target, prior_weight, prior_constant, rejection):
        if not bool(torch.isfinite(x).all()):
            raise ValueError('nonfinite graph input')
    if bool(((keep < 0)|(keep > 1)).any()) or any(bool((x < 0).any()) for x in (weight, prior_weight, prior_constant, rejection)):
        raise ValueError('invalid keep or negative cost')
    phase, keep = phase.detach().clone(), keep.detach().clone()
    args = (i,j,target,weight,prior_target,prior_weight,prior_constant,rejection)
    kwargs = dict(mask_prior=mask_prior,epsilon=epsilon)
    start = time.monotonic()
    history = []
    total_cg = 0
    stalled = 0

    def record(step, cg, backtracks):
        value, terms = objective(phase,keep,*args,**kwargs)
        exact, _ = objective(phase,keep,*args,mask_prior=mask_prior,epsilon=0.)
        values = [float(x) for x in (value,*terms,exact)]
        row = dict(step=step, loss=values[0], pair_loss=values[1], prior_loss=values[2],
                   broken_loss=values[3], exact_wrapped_l1_objective=values[4],
                   seconds=time.monotonic()-start,cg_iterations=cg,total_cg_iterations=total_cg,
                   backtracks=backtracks)
        if callback is not None:
            row.update(callback(step, phase, keep) or {})
        history.append(row)
        if log is not None:
            log(row)
        return values[0]

    previous = record(0,0,0)
    for step in range(1,iterations+1):
        delta = phase[j]-phase[i]
        r = residual(delta,target)
        ur = residual(phase,prior_target)
        c = weight*keep[i]*keep[j]/r.abs().clamp_min(epsilon)
        uc = prior_weight*(keep if mask_prior else 1)/ur.abs().clamp_min(epsilon)
        proposed,cg = quadratic_step(phase,i,j,delta-r,c,phase-ur,uc,
                                     max_cg=max_cg,cg_tolerance=cg_tolerance)
        if not bool(torch.isfinite(proposed).all()):
            raise RuntimeError('Nonfinite CG phase update')
        total_cg += cg
        old_phase, old_keep = phase, keep
        accepted = False
        for backtracks in range(21):
            phase = old_phase+(proposed-old_phase)*(2.**-backtracks)
            keep = old_keep if fixed_keep else keep_step(phase,old_keep,*args,**kwargs)
            value = float(objective(phase,keep,*args,**kwargs)[0])
            if value <= previous:
                accepted = True
                break
        if not accepted:
            phase,keep = old_phase,old_keep
        current = record(step,cg,backtracks if accepted else 21)
        stalled = stalled+1 if previous-current <= tolerance*max(1.,abs(previous)) else 0
        previous = current
        if stalled >= 5:
            break
    return dict(phase=phase,keep=keep,history=history,iterations=step,
                total_cg_iterations=total_cg,converged=stalled>=5)
