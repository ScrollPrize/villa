"""Linear physical-length initialization from measured phases and enclosures.

No reference solution is read. Integer labels are supplied by the caller;
angles are soft observations, not exact arc-length coordinates.
"""
import numpy as np
from scipy.optimize import lsq_linear


def enclosure_brackets(d):
    """Adjacent crossings of the same H/V pair with opposite normal signs."""
    groups={}
    for e,(i,j) in enumerate(zip(d['i'],d['j'])):
        if d['parallel'][e] or d['family'][i]==d['family'][j]:continue
        h,v=(i,j) if d['family'][i]==0 else (j,i)
        ds=d['ds_a'][e] if i==h else d['ds_b'][e]
        sign=d['sign'][e] if i==h else -d['sign'][e]
        groups.setdefault((int(h),int(v)),[]).append((float(ds),e,float(sign)))
    result=[]
    for (h,v),rows in sorted(groups.items()):
        rows.sort()
        for a,b in zip(rows,rows[1:]):
            if a[2]*b[2]>=0:continue
            inner,outer=(a,b) if a[2]>0 else (b,a)
            result.append((h,v,inner[1],outer[1]))
    return result


def cumulative_coeff(q,count):
    k=int(np.clip(np.floor(q),0,count-1))
    a=np.zeros(count);a[:k]=1;a[k]=q-k
    return a


def fit_enclosure_widths(d,samples,offsets,widths,gauge,active):
    count=len(widths);active=np.asarray(active);hs=np.where(d['family']==0)[0]
    qv={}
    for v in np.where(d['family']==1)[0]:
        phases=np.r_[d['phase_a'][d['i']==v],d['phase_b'][d['j']==v]]
        qv[v]=offsets[v]+float(np.median(phases))
    center=cumulative_coeff(gauge,count)
    rows=[];rhs=[];bracket_rows=[]
    def add(q,h,ds,scale):
        a=np.zeros(len(active)+len(hs))
        coeff=cumulative_coeff(q,count)-center
        a[:len(active)]=coeff[active]
        a[len(active)+list(hs).index(h)]=-1
        outside=coeff.copy();outside[active]=0
        rows.append(a/scale);rhs.append((ds-outside@widths)/scale)
    for h,v,inner,outer in enclosure_brackets(d):
        di=float(d['ds_a'][inner] if d['i'][inner]==h else d['ds_b'][inner])
        do=float(d['ds_a'][outer] if d['i'][outer]==h else d['ds_b'][outer])
        add(qv[v],h,di,10);add(qv[v]+1,h,do,10)
        bracket_rows.append(dict(h=int(d['source_id'][h]),v=int(d['source_id'][v]),
            inner=inner,outer=outer,turn_arc_vx=do-di,q_inner=qv[v]))
    if not bracket_rows:raise ValueError('Enclosure width initialization needs an H-V-H bracket')
    for f,ds,phase in zip(samples['owner'],samples['ds'],samples['phase']):
        if d['family'][f]==0:add(offsets[f]+phase,int(f),float(ds),1000)
    for j in range(len(active)-1):
        a=np.zeros(len(active)+len(hs));a[j]=-1/1000;a[j+1]=1/1000
        rows.append(a);rhs.append(0)
    fit=lsq_linear(np.array(rows),np.array(rhs),
        bounds=(np.r_[np.full(len(active),100.),np.full(len(hs),-np.inf)],
                np.r_[np.full(len(active),50000.),np.full(len(hs),np.inf)]),tol=1e-12,max_iter=1000)
    if not fit.success:raise RuntimeError(fit.message)
    result=np.asarray(widths,dtype=float).copy();result[active]=fit.x[:len(active)]
    return result,dict(zip(hs,fit.x[len(active):])),qv,bracket_rows
