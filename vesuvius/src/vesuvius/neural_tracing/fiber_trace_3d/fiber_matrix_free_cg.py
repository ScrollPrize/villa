"""Shared Jacobi-preconditioned CG for the graph solvers' local quadratics."""
import torch


def solve(matvec, rhs, initial, diagonal, *, iterations=100, tolerance=1e-5, preconditioner=None):
    if iterations<1 or tolerance<=0:
        raise ValueError('CG iterations and tolerance must be positive')
    inv=torch.where(diagonal>0,diagonal.reciprocal(),0.)
    apply=preconditioner if preconditioner is not None else lambda r:r*inv
    x=initial.clone();r=rhs-matvec(x);z=apply(r);p=z.clone()
    rz=(r*z).sum();first=rz.clone();tiny=torch.finfo(x.dtype).tiny
    for step in range(iterations):
        ap=matvec(p);pap=(p*ap).sum()
        alpha=torch.where(pap>tiny,rz/pap.clamp_min(tiny),0.)
        x=x+alpha*p;r=r-alpha*ap;z=apply(r)
        next_rz=(r*z).sum()
        beta=torch.where(rz>tiny,next_rz/rz.clamp_min(tiny),0.)
        p=z+beta*p;rz=next_rz
        if (step+1)%10==0 or step+1==iterations:
            if bool(rz<=tolerance**2*first.clamp_min(tiny)):
                break
    return x,step+1
