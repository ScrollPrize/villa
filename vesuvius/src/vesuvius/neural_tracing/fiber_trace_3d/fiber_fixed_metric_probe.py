"""Fixed-assignment diagnostic: exact meetings plus lengths or H/V angles."""
import json
import time
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize
from scipy.linalg import qr
from lasagna.inference_provenance import atomic_write
from .fiber_gt_constraint_score import arrays
from .fiber_bucket_uv import BucketUV
from .fiber_bucket_uv_visualization import current_eligible_edges
from .fiber_affine_geometry import direction_plan
from . import fiber_soft_winding as soft


class Equalities:
    """Scaled coordinates; exact meetings and squared relative segment lengths."""
    def __init__(self,n,left,right,gap,a,b):
        self.n=n;self.left=np.asarray(left);self.right=np.asarray(right)
        self.gap=np.asarray(gap);self.a=np.asarray(a);self.b=np.asarray(b)

    def fun(self,x):
        p=x.reshape(self.n,2);d=p[self.right]-p[self.left]
        return np.r_[(p[self.b]-p[self.a]).ravel(),(d*d).sum(1)/self.gap**2-1]

    def jac(self,x):
        p=x.reshape(self.n,2);m=2*len(self.a)
        jac=np.zeros((m+len(self.left),2*self.n))
        for axis in (0,1):
            row=2*np.arange(len(self.a))+axis
            jac[row,2*self.b+axis]+=1;jac[row,2*self.a+axis]-=1
            v=2*(p[self.right,axis]-p[self.left,axis])/self.gap**2
            row=m+np.arange(len(self.left))
            jac[row,2*self.right+axis]+=v;jac[row,2*self.left+axis]-=v
        return jac


class StraightEqualities:
    """Exact meetings and axis-aligned segments, with redundant rows removed."""
    def __init__(self,n,left,right,family,direction,a,b):
        rows=[]
        for i,j in zip(a,b):
            for axis in (0,1):
                row=np.zeros(2*n);row[2*j+axis]=1;row[2*i+axis]-=1
                rows.append(row)
        forward=[]
        for i,j,axis,sign in zip(left,right,family,direction):
            axis=int(axis)
            row=np.zeros(2*n);row[2*j+1-axis]=1;row[2*i+1-axis]-=1
            rows.append(row)
            row=np.zeros(2*n);row[2*j+axis]=sign;row[2*i+axis]-=sign
            forward.append(row)
        self.all_rows=np.asarray(rows)
        _,r,piv=qr(self.all_rows.T,pivoting=True,mode='economic')
        rank=np.linalg.matrix_rank(r)
        self.matrix=self.all_rows[piv[:rank]]
        self.forward=np.asarray(forward)

    def fun(self,x):return self.matrix@x
    def jac(self,x):return self.matrix


def run(settings):
    torch.set_num_threads(4)
    start=time.monotonic();src=Path(settings['source']);root=Path(settings['output_root'])
    if (root/'solve/solution.npz').exists():raise FileExistsError(root)
    root.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((src/'config.json').read_text());d=arrays(cfg['tensor_cache']);s=arrays(src/'solution.npz')
    ext=BucketUV(d,cfg,'cpu');ext.restore(s)
    state=torch.as_tensor(s['state']);widths=torch.as_tensor(s['widths'])
    ext.cavity.values=torch.as_tensor(s['winding_sigma'])
    with torch.no_grad():
        _,lp=ext.local_hypotheses(state);col=lp.argmax(1)
        prob=(lp.exp().amax(1)*state[:,3].sigmoid()).numpy()
        edges=current_eligible_edges(d,ext,state,widths,col,prob)
        pos=ext.positions().detach().double();nodes=torch.arange(len(ext.owner));cols=col[ext.owner]
        initial=pos[nodes,cols].numpy()
    r,l,family,direction,gap,_=direction_plan(ext.owner,ext.v_offset_arc,ext.curve.family,ext.curve.direction)
    a=ext.v_offset_a[edges].numpy();b=ext.v_offset_b[edges].numpy()
    scale=1000.;origin=initial.mean(0);x0=((initial-origin)/scale).ravel()
    eq=Equalities(len(initial),l.numpy(),r.numpy(),gap.numpy()/scale,a,b)
    mode=settings.get('fixed_metric','length')
    if mode not in ('length','angle'):raise ValueError(mode)
    if mode=='angle':
        eq=StraightEqualities(len(initial),l.numpy(),r.numpy(),family.numpy(),direction.numpy(),a,b)
    # Only replace final MAP positions. All other labels, layers, widths,
    # assignment probabilities and the dynamically selected event set stay fixed.
    ext.affine=None
    data,_=soft.tensor_inputs(d,{},cfg,'cpu');all_edges=torch.arange(len(d['i']))
    def evaluate(x,gradient=False):
        p=pos.clone();p[nodes,cols]=torch.as_tensor(x.reshape(-1,2)*scale+origin)
        ext.values=torch.nn.Parameter(p)
        terms=ext.terms(state,widths,data,all_edges)
        total=sum(terms.values())
        if gradient:
            g=torch.autograd.grad(total,ext.values)[0][nodes,cols].numpy().ravel()*scale
            return float(total.detach()),g
        return {k:float(v.detach()) for k,v in terms.items()}
    before=evaluate(x0);history=[]
    def record(x):
        if len(history)%25==0:
            print(json.dumps(dict(iteration=len(history),equality_max=float(np.abs(eq.fun(x)).max()),seconds=time.monotonic()-start)),flush=True)
        history.append(float(np.abs(eq.fun(x)).max()))
    constraints=dict(type='eq',fun=eq.fun,jac=eq.jac)
    if mode=='angle':
        # Zero angle also requires positive longitudinal progress: do not allow
        # a reversed or collapsed segment to satisfy the transverse equalities.
        constraints=[constraints,dict(type='ineq',fun=lambda x:eq.forward@x-1e-9,
                                     jac=lambda x:eq.forward)]
    feasible=minimize(lambda x:(.5*np.sum((x-x0)**2),x-x0),x0,jac=True,
        method='SLSQP',constraints=constraints,callback=record,
        options=dict(maxiter=settings.get('max_iterations',2000),ftol=1e-12))
    if np.abs(eq.fun(feasible.x)).max()>1e-7:
        atomic_write(root/'infeasible.json',dict(message=str(feasible.message),max_error=float(np.abs(eq.fun(feasible.x)).max()),history=history),durable=False)
        raise RuntimeError('Exact feasibility not reached; see infeasible.json')
    # Minimize the original implemented objective on the feasible manifold.
    norm=max(1.,sum(before.values()))
    def objective(x):
        value,grad=evaluate(x,True);return value/norm,grad/norm
    fitted=minimize(objective,feasible.x,jac=True,method='SLSQP',constraints=constraints,
        callback=record,options=dict(maxiter=settings.get('max_iterations',2000),ftol=1e-11))
    x=fitted.x if np.abs(eq.fun(fitted.x)).max()<1e-7 else feasible.x
    after=evaluate(x);final=x.reshape(-1,2)*scale+origin
    # Is the required movement representable by the existing fast affine DOFs,
    # or does it need slow per-node updates as well?
    movement=final-initial;affine_residual=movement.copy()
    owners=ext.owner.numpy();arc=ext.v_offset_arc.numpy()
    for owner in np.unique(owners):
        ix=np.flatnonzero(owners==owner);span=np.ptp(arc[ix])
        t=(arc[ix]-arc[ix].mean())/max(float(span),1.)
        design=np.column_stack((np.ones(len(ix)),t))
        affine_residual[ix]-=design@np.linalg.lstsq(design,movement[ix],rcond=None)[0]
    path=[]
    for alpha in (0.,.001,.01,.1,.25,.5,.75,1.):
        terms=evaluate(x0+alpha*(x-x0))
        path.append(dict(fraction=alpha,geometry_objective=sum(terms.values())))
    result=dict(source=str(src),fixed_metric=mode,meetings=len(edges),fixed_events=edges.tolist(),
        segments=len(l),before=before,after=after,
        loss_changes={k:after[k]-before[k] for k in before},
        nearest_feasible_terms=evaluate(feasible.x),
        feasibility_status=str(feasible.message),optimization_status=str(fitted.message),
        optimization_success=bool(fitted.success),seconds=time.monotonic()-start,
        meeting_max_vx=np.abs(final[b]-final[a]).max(0).tolist(),
        length_max_relative=float(np.max(np.abs(np.linalg.norm(final[r]-final[l],axis=1)/gap.numpy()-1))),
        movement_max_vx=float(np.max(np.linalg.norm(final-initial,axis=1))),
        nonaffine_movement_max_vx=float(np.linalg.norm(affine_residual,axis=1).max()),
        straight_path_objective=path,
        history=history)
    # Save through the original affine representation for standard rendering.
    saved_ext=BucketUV(d,cfg,'cpu');saved_ext.restore(s)
    p=pos.clone();p[nodes,cols]=torch.as_tensor(final)
    if saved_ext.affine is not None:
        saved_ext.affine.initialize(p)
        saved_ext.values=torch.nn.Parameter(torch.zeros_like(p))
    else:saved_ext.values=torch.nn.Parameter(p)
    out=dict(s);out.update(saved_ext.arrays())
    solve=root/'solve';solve.mkdir()
    np.savez(solve/'solution.npz',**out)
    reread=arrays(solve/'solution.npz')
    for key in ('state','widths','bucket_uv_labels','bucket_uv_active',
                'shared_curve_phase_v','intermediate_uv_scaled_qv'):
        if key in s and not np.array_equal(s[key],reread[key]):
            raise RuntimeError('Frozen state changed: '+key)
    saved_xy=reread['bucket_uv_positions'][nodes.numpy(),cols.numpy()]
    if np.abs(eq.fun(((saved_xy-origin)/scale).ravel())).max()>1e-7:
        raise RuntimeError('Saved geometry violates exact constraints')
    if mode=='angle':
        if np.max(np.abs(eq.all_rows@saved_xy.ravel()))>1e-7:
            raise RuntimeError('Saved geometry violates redundant straightness constraints')
        if np.min(eq.forward@saved_xy.ravel())<=0:
            raise RuntimeError('Saved geometry reverses or collapses a segment')
    atomic_write(solve/'config.json',dict(cfg,output_root=str(solve)),durable=False)
    atomic_write(root/'report.json',result,durable=False)
    print(json.dumps({k:v for k,v in result.items() if k!='history'}),flush=True)
    if settings.get('visualization') is not None:
        from .fiber_uv_visualization import run as render
        render(dict(settings['visualization'],bucket_comparison=str(solve),output_root=str(root/'visualization')))
    return result
