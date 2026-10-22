"""Integer sheet hypotheses: layer spacing calibrates measurements, not UV."""
import numpy as np


def margin(sign,family_delta,parallel):
    return ((sign*family_delta<0)&~parallel).astype(float)


def target(distance,sign,family_delta,parallel,offset=.5):
    center=sign*distance-np.where(parallel,0.,offset*family_delta)
    # Signed MAP tie chooses the smaller outward distance; unsigned is ambiguous.
    ordered=sign*center
    rounded=np.ceil(ordered-.5)
    return sign*np.maximum(rounded,margin(sign,family_delta,parallel))


def contact_metrics(d,geometry,widths,delta,indices):
    from . import fiber_uv_layout as uv
    p,_,keep,pick,u,v,b=geometry;i=d['i'][indices];j=d['j'][indices]
    confident=keep&(p[np.arange(len(p)),pick]>=.5)
    same=confident[i]&confident[j]&~d['parallel'][indices]&(d['family'][i]!=d['family'][j])&(delta==0)
    qa,va=uv.geometry_points(d,u,v,b,widths,i,d['ds_a'][indices])
    qb,vb=uv.geometry_points(d,u,v,b,widths,j,d['ds_b'][indices])
    ua=u[i]+(d['family'][i]==0)*d['ds_a'][indices]
    ub=u[j]+(d['family'][j]==0)*d['ds_b'][indices]
    def stats(a):
        a=abs(a[same])
        return dict(mean=float(a.mean()),median=float(np.median(a)),rms=float(np.sqrt(np.mean(a*a))),
                    p95=float(np.quantile(a,.95)),maximum=float(a.max())) if len(a) else None
    return dict(events=int(same.sum()),u_base=stats(ub-ua),v_base=stats(vb-va),windings=stats(qb-qa))
