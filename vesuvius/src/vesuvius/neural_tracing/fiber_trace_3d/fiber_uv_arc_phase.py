"""Learned phase and physical arc travel; no local angular/seam correction."""
import math
import numpy as np


def coarse_coordinates(state, phase, family, ds, widths, boundaries,
                       backend='reference', division='standard'):
    import torch
    from . import fiber_uv_layout as uv
    k=len(widths)
    q=torch.arange(k,device=state.device)[None,:]+(phase+state[:,k+1])[:,None]
    distance=(family==0)[:,None]*ds[:,None]
    if backend=='combined':
        from .fiber_uv_coordinates import combined_coordinates
        return combined_coordinates(q,distance,widths,boundaries,division)[1]
    return uv.u_to_q(uv.q_to_u(q,widths,boundaries)+distance,widths,boundaries)


def coarse_points(d,state,pick,widths,ids,ds):
    """CPU counterpart for selected hypotheses, including extrapolated phase."""
    k=len(widths);bounds=np.r_[0.,np.cumsum(widths)]
    q=pick[ids]+d['phase'][ids]+state[ids,k+1]
    idx=np.clip(np.floor(q).astype(int),0,k-1)
    u=bounds[idx]+(q-idx)*widths[idx]+(d['family'][ids]==0)*ds
    idx=np.clip(np.searchsorted(bounds,u,side='right')-1,0,k-1)
    return idx+(u-bounds[idx])/widths[idx]


def event_signed_arcs(d,ids,arcs):
    """Map source arc to the signed travel of a prepared piece."""
    span=d['end'][ids]-d['start'][ids]
    direction=np.divide(d['node_ds'][ids,2]-d['node_ds'][ids,0],span,
                        out=np.zeros_like(span,dtype=float),where=span!=0)
    return d['node_ds'][ids,0]+direction*(arcs-d['start'][ids])


def event_arc_points(d,state,pick,widths,ids,arcs):
    ds=event_signed_arcs(d,ids,arcs)
    return coarse_points(d,state,pick,widths,ids,ds)


def local_width(qa,qb,widths):
    """Gather E*K endpoint widths, then broadcast their mean in the fused loss.

    No pairwise index lookup or separately fitted pair width. Detached only for
    loss scaling; the coordinate mapping still differentiates fitted widths.
    """
    a=qa.detach().floor().long().clamp(0,len(widths)-1)
    b=qb.detach().floor().long().clamp(0,len(widths)-1)
    return .5*(widths.detach()[a][:,:,None]+widths.detach()[b][:,None,:])


def continuation(pa,pb,qa,qb,va,vb,widths,scale):
    from . import fiber_uv_layout as uv
    du=(qb[:,None,:]-qa[:,:,None])*local_width(qa,qb,widths)
    dv=vb[:,None,:]-va[:,:,None]
    return uv.compatibility_loss(pa,pb,(du/scale).square()+(dv/scale).square(),1.)
