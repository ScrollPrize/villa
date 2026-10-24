"""Read-only conditioning audit of the exact shared-only full objective."""
import json
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from lasagna.inference_provenance import atomic_write
from .fiber_gt_constraint_score import arrays
from .fiber_bucket_uv import BucketUV
from . import fiber_soft_winding as soft, fiber_implicit_winding as density


@threadpool_limits.wrap(limits=1,user_api='blas')
def run(settings):
    torch.set_num_threads(4)
    cfg=json.loads(Path(settings['config']).read_text());cfg.update(settings.get('parameters',{}))
    d=arrays(cfg['tensor_cache'])
    samples=arrays(cfg['soft_samples']);saved=arrays(settings['solution'])
    if settings.get('geometry_only'):
        # Explicit cross-formulation diagnostic, never a checkpoint resume:
        # evaluate old coordinates using the requested objective's uncertainty.
        saved=dict(saved);saved.pop('winding_sigma',None)
    state=torch.tensor(saved['state'],dtype=torch.float64,requires_grad=True)
    w=torch.tensor(saved['widths'],dtype=torch.float64)-1e-4
    raw=(w+torch.log(-torch.expm1(-w))).detach().requires_grad_(True)
    ext=BucketUV(d,cfg,'cpu');ext.prepare_fields(d,samples);ext.restore(saved)
    if not ext.curve.ready:ext.curve.initialize(state,w+1e-4)
    initial=torch.tensor(arrays(cfg['initial_solution'])['widths'],dtype=torch.float64)
    data,nodes=soft.tensor_inputs(d,samples,cfg,'cpu')
    captured={};components={}
    def pair_cost(*args,**kwargs):
        value=density.pair_cost(*args,**kwargs)
        if settings.get('loss_components_only'):
            local=list(args)
            local[-1]=dict(args[-1],gt_pair_mode='sign_only')
            order=density.pair_cost(*local,**kwargs)[0]
            p=state[:,3].sigmoid()
            gate=data['class_weight']*p[data['i']]*p[data['j']]
            for family,mask in [('perpendicular',~data['parallel']),('parallel',data['parallel'])]:
                components[family]={name:float((gate[mask]*cost[mask]).sum().detach()*cfg['distance_weight']/len(gate))
                    for name,cost in [('order',order),('measured',value[0]-order)]}
        if settings.get('pair_branches'):
            sa,sb=args[:2]
            geometry=kwargs.get('geometry')
            if geometry is not None:
                pa,pb=geometry.points_both(sa[:,None,:],sb[:,None,:])
                phase=pb[1][:,0]-pa[1][:,0]
            else:phase=sb[:,1]-sa[:,1]
            captured.update(cost=value[0].detach().numpy().copy(),
                shift=torch.floor(phase+.5).detach().numpy().copy(),
                ka=torch.floor(sa[:,0]+.5).detach().numpy().copy(),
                kb=torch.floor(sb[:,0]+.5).detach().numpy().copy())
        return value
    def objective():return soft.objective(state,raw,initial,data,nodes,cfg,pair_cost=pair_cost,extension=ext)
    value,terms=objective()
    if settings.get('loss_components_only'):
        result=dict(loss=float(value.detach()),terms={k:float(v.detach()) for k,v in terms.items()},components=components,
                    geometry_only=bool(settings.get('geometry_only')),source=settings['solution'])
        atomic_write(Path(settings['output']),result,durable=False);print(json.dumps(result),flush=True)
        return result
    names=['N_and_enabled','widths']+[n for n,p in ext.named_parameters()]
    params=[state,raw]+list(ext.parameters())
    scales=[torch.ones_like(state),torch.full_like(raw,1000.)]+[
        torch.full_like(p,100. if n.endswith('.v') else .01) for n,p in ext.named_parameters()]
    grads=torch.autograd.grad(value,params,allow_unused=True)
    rows=[]
    for name,p,s,g in zip(names,params,scales,grads):
        if g is None:continue
        if settings.get('blocks') and name not in settings['blocks']:continue
        z=g*s;norm=float(z.norm());direction=z/(norm or 1)*s
        old=p.detach().clone();analytic=float((g*direction).sum())
        checks=[];pair_derivatives=[]
        for step in (1e-4,1e-3):
            vals=[];parts=[];captures=[]
            for sign in (-1,1):
                with torch.no_grad():
                    p.copy_(old+sign*step*direction);loss,part=objective()
                    vals.append(float(loss));parts.append({k:float(v) for k,v in part.items()})
                    captures.append(dict(captured))
            with torch.no_grad():p.copy_(old)
            checks.append(dict(step=step,derivative=(vals[1]-vals[0])/(2*step),
                curvature=(vals[1]+vals[0]-2*float(value.detach()))/step**2,
                terms_minus=parts[0],terms_plus=parts[1]))
            if settings.get('pair_branches'):
                cm,cp=captures
                pair_derivatives.append((cp['cost']-cm['cost'])/(2*step))
                changed=np.flatnonzero(np.logical_or.reduce([cm[k]!=cp[k] for k in ('shift','ka','kb')]))
                checks[-1]['pair_branch_changes']=[dict(edge=int(i),source_a=int(d['source_id'][d['i'][i]]),
                    source_b=int(d['source_id'][d['j'][i]]),parallel=bool(d['parallel'][i]),
                    minus={k:float(cm[k][i]) for k in cm},plus={k:float(cp[k][i]) for k in cp}) for i in changed]
        rows.append(dict(name=name,count=p.numel(),scaled_gradient_norm=norm,
            scaled_gradient_max=float(z.abs().max()),analytic_derivative=analytic,finite_difference=checks))
        if pair_derivatives:
            discrepancy=pair_derivatives[1]-pair_derivatives[0]
            rows[-1]['largest_pair_derivative_changes']=[dict(edge=int(i),
                source_a=int(d['source_id'][d['i'][i]]),source_b=int(d['source_id'][d['j'][i]]),
                parallel=bool(d['parallel'][i]),derivatives=[float(a[i]) for a in pair_derivatives])
                for i in np.argsort(abs(discrepancy))[-8:][::-1]]
    gaps=(ext.curve.arc[1:]-ext.curve.arc[:-1])[ext.curve.owner[1:]==ext.curve.owner[:-1]].numpy()
    result=dict(loss=float(value.detach()),terms={k:float(v.detach()) for k,v in terms.items()},blocks=rows,
        segment_gaps=dict(count=len(gaps),quantiles=np.quantile(gaps,[0,.01,.05,.5,.95,1]).tolist(),
                          below_1=int((gaps<1).sum()),below_0p01=int((gaps<.01).sum())),
        source=settings['solution'])
    atomic_write(Path(settings['output']),result,durable=False);print(json.dumps(result),flush=True)
    return result


def restart(settings):
    """Restart optimizer history only: same objective and saved full state."""
    from .fiber_implicit_winding import run as solve
    source=Path(settings['source'])
    cfg=json.loads((source/'config.json').read_text())
    cfg.update(initial_solution=str(source/'solution.npz'),
               output_root=settings['output_root'],devices=settings['devices'])
    return solve(cfg)
