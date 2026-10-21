"""Shared scaled joint L-BFGS used by the GT probe and managed larger solves."""
import time
import json
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits
from lasagna.inference_provenance import atomic_write


def estimated_fixed_length_vram(edge_count, batch_size, sigma=.5):
    """Measured FP64 fixed-length sigma=.5 model: allocator +10% +1GiB.

    Round up to GiB. Not a general estimate for other geometry/support models.
    """
    if edge_count < 1 or batch_size < 1:
        raise ValueError('Memory estimate requires positive edge and batch counts')
    gib=1024**3
    if not np.isfinite(sigma) or sigma<=0:raise ValueError('Sigma must be positive')
    support_factor=max(1.,((2*np.ceil(6*sigma)+1)/7)**2)
    estimate=min(edge_count,batch_size)*.75*1024**2*1.1*support_factor+gib
    return int(np.ceil(estimate/gib))*gib


def scaled_joint(objective, state, raw, extension, steps, callback=None,
                 optimize_width_only=False, optimize_rejection=False, local_gap_scaling=False,
                 direct_rejection=False, fixed_enabled_mask=None, objective_parts=None, direct_widths=False,
                 monotone_widths=False, minimum_winding_width=1000.,
                 width_coordinate_scale=1000., phase_coordinate_scale=.01,
                 relative_widths=False, relative_width_coordinate_scale=.01,
                 optimizer_kind='lbfgs', adam_lr=.1,
                 plateau_window=0, plateau_rtol=1e-6, plateau_atol=1e-9,
                 square_match_initial_scale=False, closure_cache=True, soft_minimum_width=False,
                 lbfgs_lr=1., winding_coordinate_scale=1., rejection_coordinate_scale=1.,
                 vertical_coordinate_scale=100., frozen_fiber_mask=None, freeze_widths=False):
    """Exact GT-probe scales, line search, tolerances and full objective.

    SciPy owns the optimizer/history; tensors may reside on CPU or CUDA.
    U/V state columns are replaced by shared curves on every fiber.
    """
    if not all(np.isfinite(x) and x>0 for x in (width_coordinate_scale,phase_coordinate_scale)):
        raise ValueError('Optimizer coordinate scales must be finite and positive')
    if soft_minimum_width and (optimizer_kind!='torch_lbfgs' or not relative_widths):
        raise ValueError('Soft minimum experiment requires Torch L-BFGS relative widths')
    named=list(extension.named_parameters())
    if not all(np.isfinite(x) and x>0 for x in (lbfgs_lr,winding_coordinate_scale,rejection_coordinate_scale,vertical_coordinate_scale)):
        raise ValueError('Optimizer step and scales must be finite and positive')
    targets=[state,raw]+[p for _,p in named]
    scales=[torch.ones_like(state),torch.full_like(raw,width_coordinate_scale)]+[
        torch.full_like(p,vertical_coordinate_scale if name.endswith('.v') else phase_coordinate_scale) for name,p in named]
    scales[0][:,0]=winding_coordinate_scale
    scales[0][:,3]=rejection_coordinate_scale
    if relative_widths:
        if not monotone_widths:raise ValueError('Relative widths require monotone bounds')
        if not np.isfinite(relative_width_coordinate_scale) or relative_width_coordinate_scale<=0:
            raise ValueError('Relative width coordinate scale must be positive')
        scales[1][1:]=relative_width_coordinate_scale
    if local_gap_scaling and not getattr(extension.curve,'straight',False):
        # Relative length penalties have curvature proportional to 1/gap^2.
        # Keep the GT-tested scales for >=100vx spans, reduce ONLY independent
        # knot residual scales near short spans; affine shifts/tilts stay free.
        curve=extension.curve
        gap=torch.full_like(curve.arc,float('inf'))
        right=torch.nonzero(curve.owner[1:]==curve.owner[:-1]).flatten()+1
        distance=curve.arc[right]-curve.arc[right-1]
        gap[right]=torch.minimum(gap[right],distance)
        gap[right-1]=torch.minimum(gap[right-1],distance)
        factor=(gap/100.).clamp(max=1.)
        if not bool((factor>0).all()):raise ValueError('Nonpositive curve knot spacing')
        for index,(name,p) in enumerate(named,2):
            if name in ('curve.u','curve.v'):scales[index]*=factor.to(p)
    masks=[torch.zeros_like(state,dtype=torch.bool),torch.ones_like(raw,dtype=torch.bool)]+[
        torch.ones_like(p,dtype=torch.bool) for _,p in named]
    masks[0][:,0]=True
    curve=getattr(extension,'curve',None)
    if curve is not None and hasattr(curve,'owner'):
        if len(torch.unique(curve.owner))!=len(state):
            raise ValueError('Remove fibers without perpendicular nodes before solving')
    if optimize_rejection:masks[0][:,3]=True
    if fixed_enabled_mask is not None:
        fixed=torch.as_tensor(fixed_enabled_mask,device=state.device,dtype=torch.bool)
        if fixed.shape!=(len(state),):raise ValueError('Invalid fixed-enabled mask')
        if not bool((state[fixed,3]==100).all()):raise ValueError('Fixed-enabled logits must be 100')
        masks[0][fixed,3]=False
    if optimize_width_only:
        masks[0].zero_()
        for mask in masks[2:]:mask.zero_()
    if frozen_fiber_mask is not None:
        frozen=torch.as_tensor(frozen_fiber_mask,device=state.device,dtype=torch.bool)
        if frozen.shape!=(len(state),) or not getattr(curve,'straight',False):
            raise ValueError('Frozen fiber geometry requires the straight whole-fiber model')
        masks[0][frozen]=False
        for index,(name,param) in enumerate(named,2):
            if name not in ('curve.u','curve.v') or param.shape!=(len(state),):
                raise ValueError('Unsupported parameter in conditional fiber fit: '+name)
            masks[index][frozen]=False
    if freeze_widths:masks[1].zero_()
    sizes=[int(m.sum()) for m in masks];cuts=np.cumsum([0]+sizes)
    probability_mask=torch.zeros_like(state,dtype=torch.bool)
    if direct_rejection and optimize_rejection and not optimize_width_only:
        probability_mask[:,3]=masks[0][:,3]
    packed_probability=probability_mask[masks[0]].cpu().numpy()
    use_probability=bool(packed_probability.any())
    def assign(x):
        with torch.no_grad():
            for p,s,m,a,b in zip(targets,scales,masks,cuts[:-1],cuts[1:]):
                p[m]=torch.from_numpy(x[a:b]).to(p)*s[m]
            if use_probability:
                state[probability_mask]=torch.logit(state[probability_mask])
    def evaluate():
        value=state.new_zeros(())
        gradients=[torch.zeros_like(p) for p in targets]
        # Each weighted part includes its share of global regularizers; sum of
        # shares is one. Release its graph before constructing the next part.
        parts=objective_parts() if objective_parts is not None else (objective(),)
        for part,_ in parts:
            if not torch.isfinite(part):raise FloatingPointError('Nonfinite L-BFGS objective')
            gs=torch.autograd.grad(part,targets,allow_unused=True)
            value+=part.detach()
            for total,g in zip(gradients,gs):
                if g is not None:total.add_(g)
        if use_probability and gradients[0] is not None:
            g=gradients[0].clone();prob=state[probability_mask].sigmoid()
            g[probability_mask]/=prob*(1-prob)
            gradients=[g]+gradients[1:]
        grad=torch.cat([((g if g is not None else torch.zeros_like(p))*s)[m].detach()
            for p,s,m,g in zip(targets,scales,masks,gradients)])
        return value.detach(),grad
    def fun(x):
        assign(x)
        value,grad=evaluate()
        return float(value),grad.cpu().numpy()
    if optimizer_kind in ('torch_lbfgs','scipy_squared'):
        if use_probability or direct_widths or not monotone_widths:
            raise ValueError('Torch squared L-BFGS requires monotone widths and logit rejection')
        if plateau_window<0 or plateau_rtol<0 or plateau_atol<0:
            raise ValueError('Invalid plateau stopping settings')
        x=torch.cat([(p/s)[m].detach() for p,s,m in zip(targets,scales,masks)])
        bounded=torch.zeros_like(x,dtype=torch.bool);bounded[cuts[1]:cuts[2]]=True
        lower=torch.zeros_like(x)
        if sizes[1]:lower[cuts[1]]=minimum_winding_width/scales[1][0]
        if soft_minimum_width and sizes[1]:
            bounded[cuts[1]]=False;lower[cuts[1]]=0.
        if bool((x[bounded]<lower[bounded]).any()):raise ValueError('Invalid initial widths')
        coordinate=torch.where(bounded,(x-lower).clamp_min(0).sqrt(),x).requires_grad_()
        factor=torch.ones_like(x)
        if square_match_initial_scale:
            factor[bounded]=2*coordinate.detach()[bounded].abs().clamp_min(1e-6)
            with torch.no_grad():coordinate.mul_(factor)
        def assign_torch():
            with torch.no_grad():
                physical=torch.where(bounded,lower+(coordinate/factor).square(),coordinate)
                for p,s,m,a,b in zip(targets,scales,masks,cuts[:-1],cuts[1:]):p[m]=physical[a:b]*s[m]
        opt=torch.optim.LBFGS([coordinate],lr=lbfgs_lr,max_iter=1,max_eval=40,
            history_size=50,tolerance_grad=1e-8,tolerance_change=1e-13,line_search_fn='strong_wolfe')
        evaluations=0;cache_hits=0;eval_seconds=0.;cache=[]
        def closure():
            nonlocal evaluations,cache_hits,eval_seconds
            assign_torch()
            if closure_cache:
                for point,value,gradient in cache:
                    if torch.equal(point,coordinate.detach()):
                        coordinate.grad=gradient.clone();cache_hits+=1;return value
            started_eval=time.perf_counter()
            value,grad=evaluate();evaluations+=1
            coordinate.grad=grad*torch.where(bounded,2*coordinate.detach()/factor.square(),torch.ones_like(coordinate))
            eval_seconds+=time.perf_counter()-started_eval
            if closure_cache:
                cache.append((coordinate.detach().clone(),value,coordinate.grad.clone()))
                if len(cache)>4:cache.pop(0)
            return value
        if optimizer_kind=='scipy_squared':
            def scipy_fun(q):
                with torch.no_grad():coordinate.copy_(torch.from_numpy(q).to(coordinate))
                value=closure()
                return float(value),coordinate.grad.cpu().numpy().copy()
            count=0
            def scipy_callback(q):
                nonlocal count
                count+=1
                with torch.no_grad():coordinate.copy_(torch.from_numpy(q).to(coordinate))
                assign_torch()
                if callback is not None:callback(count)
            with threadpool_limits(limits=1,user_api='blas'):
                fit=minimize(scipy_fun,coordinate.detach().cpu().numpy().copy(),jac=True,
                    method='L-BFGS-B',callback=scipy_callback,
                    options=dict(maxiter=steps,maxls=40,maxcor=50,ftol=1e-13,gtol=1e-8))
            with torch.no_grad():coordinate.copy_(torch.from_numpy(fit.x).to(coordinate))
            assign_torch()
            return dict(kind='scipy_squared',success=bool(fit.success),message=str(fit.message),
                iterations=int(fit.nit),evaluations=evaluations,evaluation_seconds=eval_seconds,
                cache_hits=cache_hits,scaled_gradient_max=float(np.abs(fit.jac).max()))
        # Evaluate accepted positions explicitly: step() returns the PRE-step
        # loss, which must not be mistaken for the accepted line-search value.
        best=float(closure());history=[best];message='Iteration budget exhausted';success=False
        iteration=0
        for iteration in range(1,steps+1):
            opt.step(closure)
            value=float(closure());best=min(best,value);history.append(best)
            if callback is not None:callback(iteration)
            if float(coordinate.grad.abs().max())<=1e-8:
                success=True;message='Gradient tolerance reached';break
            if plateau_window and iteration>=plateau_window:
                previous=history[-plateau_window-1]
                if previous-best<=plateau_atol+plateau_rtol*max(abs(previous),abs(best)):
                    success=True;message=f'Objective plateau over {plateau_window} accepted-step calls';break
        assign_torch()
        return dict(kind='torch_lbfgs_softmin' if soft_minimum_width else 'torch_lbfgs_squared',success=success,message=message,
            iterations=iteration,evaluations=evaluations,scaled_gradient_max=float(coordinate.grad.abs().max()),
            cache_hits=cache_hits,evaluation_seconds=eval_seconds,
            plateau_window=plateau_window,plateau_rtol=plateau_rtol,plateau_atol=plateau_atol)
    x=np.concatenate([(p/s)[m].detach().cpu().numpy() for p,s,m in zip(targets,scales,masks)])
    bounds=None
    if use_probability:
        indices=np.flatnonzero(packed_probability)
        x[indices]=state[probability_mask].detach().sigmoid().cpu().numpy()
        # Finite storage logits, with dtype-safe bounds. In the FP64 solver the
        # interval is [1e-8,1-1e-8]; objective itself is unchanged.
        epsilon=max(1e-8,4*torch.finfo(state.dtype).eps)
        x[indices]=np.clip(x[indices],epsilon,1-epsilon)
        bounds=[(None,None)]*len(x)
        for i in indices:bounds[i]=(epsilon,1-epsilon)
    if direct_widths:
        if bool((raw<1e-4).any()):raise ValueError('Direct widths must be >=1e-4')
        if bounds is None:bounds=[(None,None)]*len(x)
        for index,scale in zip(range(cuts[1],cuts[2]),scales[1][masks[1]].detach().cpu().numpy()):
            bounds[index]=(1e-4/scale,None)
    if monotone_widths:
        if direct_widths:raise ValueError('Choose one width parameterization')
        if not np.isfinite(minimum_winding_width) or minimum_winding_width<=0:
            raise ValueError('minimum_winding_width must be positive')
        lower=torch.zeros_like(raw);lower[0]=minimum_winding_width
        if bool((raw<lower).any()):raise ValueError('Invalid monotone width parameters')
        if bounds is None:bounds=[(None,None)]*len(x)
        for index,limit in zip(range(cuts[1],cuts[2]),(lower/scales[1])[masks[1]].detach().cpu().numpy()):
            bounds[index]=(float(limit),None)
    iteration=0
    def on_step(x):
        nonlocal iteration
        iteration+=1
        if callback is not None:
            assign(x);callback(iteration)
    with threadpool_limits(limits=1,user_api='blas'):
        if optimizer_kind=='adam':
            if not np.isfinite(adam_lr) or adam_lr<=0:raise ValueError('Adam lr must be positive')
            # Reuse exactly the same packed coordinates, objective gradient,
            # parameter masks and bounds as L-BFGS. No alternate model path.
            coordinate=torch.tensor(x,dtype=torch.float64,requires_grad=True)
            opt=torch.optim.Adam([coordinate],lr=adam_lr)
            lower=torch.tensor([a if a is not None else -np.inf for a,b in bounds],dtype=coordinate.dtype) if bounds else None
            upper=torch.tensor([b if b is not None else np.inf for a,b in bounds],dtype=coordinate.dtype) if bounds else None
            for _ in range(steps):
                _,gradient=fun(coordinate.detach().numpy())
                opt.zero_grad(set_to_none=True)
                coordinate.grad=torch.from_numpy(gradient.copy())
                opt.step()
                if bounds:
                    with torch.no_grad():coordinate.clamp_(lower,upper)
                on_step(coordinate.detach().numpy())
            assign(coordinate.detach().numpy())
            return dict(kind='scaled_adam',iterations=steps,evaluations=steps,
                        success=False,message='Iteration budget exhausted; not a convergence claim')
        if optimizer_kind!='lbfgs':raise ValueError('Unknown packed optimizer')
        fit=minimize(fun,x,jac=True,method='L-BFGS-B',bounds=bounds,callback=on_step,
            options=dict(maxiter=steps,maxls=40,maxcor=50,ftol=1e-13,gtol=1e-8))
    assign(fit.x)
    return dict(success=bool(fit.success),message=str(fit.message),iterations=int(fit.nit),
                evaluations=int(fit.nfev),scaled_gradient_max=float(np.abs(fit.jac).max()))


def uncertainty_joint(objective,state,raw,extension,steps,data,config,callback=None,**kwargs):
    """Refresh detached uncertainty only BETWEEN independent line-search solves.

    Never mutate the objective inside a SciPy closure or retain curvature across
    a changed uncertainty objective. Fixed uncertainty keeps the original solve.
    """
    from .fiber_soft_winding import decode_widths
    kwargs['direct_widths']=config.get('width_parameterization')=='direct'
    kwargs['monotone_widths']=config.get('width_parameterization') in ('monotone_additive','monotone_relative')
    kwargs['relative_widths']=config.get('width_parameterization')=='monotone_relative'
    kwargs['relative_width_coordinate_scale']=float(config.get('relative_width_coordinate_scale',.01))
    kwargs['optimizer_kind']=config.get('packed_optimizer','lbfgs')
    kwargs['adam_lr']=float(config.get('packed_adam_lr',.1))
    kwargs['plateau_window']=int(config.get('lbfgs_plateau_window',0))
    kwargs['plateau_rtol']=float(config.get('lbfgs_plateau_rtol',1e-6))
    kwargs['plateau_atol']=float(config.get('lbfgs_plateau_atol',1e-9))
    kwargs['square_match_initial_scale']=bool(config.get('lbfgs_square_match_initial_scale',False))
    kwargs['closure_cache']=bool(config.get('lbfgs_closure_cache',True))
    kwargs['soft_minimum_width']=bool(config.get('soft_minimum_width',False))
    kwargs['minimum_winding_width']=float(config.get('minimum_winding_width',1000.))
    kwargs['width_coordinate_scale']=float(config.get('lbfgs_width_coordinate_scale',1000.))
    kwargs['phase_coordinate_scale']=float(config.get('lbfgs_phase_coordinate_scale',.01))
    kwargs['lbfgs_lr']=float(config.get('lbfgs_lr',1.))
    kwargs['winding_coordinate_scale']=float(config.get('lbfgs_winding_coordinate_scale',1.))
    kwargs['rejection_coordinate_scale']=float(config.get('lbfgs_rejection_coordinate_scale',1.))
    kwargs['vertical_coordinate_scale']=float(config.get('lbfgs_vertical_coordinate_scale',100.))
    if not config.get('cavity_winding_sigma'):
        return scaled_joint(objective,state,raw,extension,steps,callback,**kwargs)
    interval=int(config.get('cavity_update_every',10))
    if interval<1:raise ValueError('cavity_update_every must be positive')
    completed=0;blocks=[]
    while completed<steps:
        if blocks or not getattr(getattr(extension,'cavity',None),'ready',False):
            extension.update(state,decode_widths(raw,config),data,len(blocks)*interval)
        offset=completed
        result=scaled_joint(objective,state,raw,extension,min(interval,steps-completed),
            (lambda iteration:callback(offset+iteration)) if callback else None,**kwargs)
        blocks.append(result)
        # A converged inner problem still needs subsequent uncertainty updates.
        # Count outer slots separately from actual accepted optimizer steps.
        completed+=min(interval,steps-completed)
    return dict(kind='alternating_cavity_lbfgs',iterations=completed,
        accepted_iterations=sum(b['iterations'] for b in blocks),
        evaluations=sum(b['evaluations'] for b in blocks),blocks=blocks,
        message='Outer update budget exhausted; not a convergence claim')


def managed_solve(config,state,raw,widths,data,nodes,d,extension,pair_cost,diagnostics):
    """Standard managed runner output, without Adam or stochastic batches."""
    from . import fiber_soft_winding as soft
    if not config.get('predicted_only_geometry') or extension is None:
        raise ValueError('Scaled L-BFGS currently supports the validated shared-only model')
    if config.get('learned_winding_sigma'):
        raise ValueError('Scaled L-BFGS does not support learned sigma')
    fixed=None
    if config.get('force_keep_source_ids'):
        sources=np.asarray(config['force_keep_source_ids'],dtype=np.int64)
        if not np.isin(sources,d['source_id']).all():raise ValueError('Missing forced-enabled source')
        fixed=np.isin(d['source_id'],sources)
        with torch.no_grad():state[torch.as_tensor(fixed,device=state.device),3]=100.
    if not extension.curve.ready:
        extension.curve.initialize(state,soft.decode_widths(raw,config).detach())
    if config.get('cavity_winding_sigma'):
        extension.update(state,soft.decode_widths(raw,config).detach(),data,0)
    root=Path(config['output_root']);root.mkdir(parents=True,exist_ok=True)
    measure_cuda=bool(config.get('measure_cuda_memory',False)) and state.is_cuda
    if measure_cuda:
        torch.cuda.synchronize(state.device)
        torch.cuda.reset_peak_memory_stats(state.device)
    started=time.monotonic();history=[]
    edge_count=len(data.get('i',[]))
    batch=int(config.get('lbfgs_edge_batch_size',max(1,edge_count)))
    if batch<1:raise ValueError('lbfgs_edge_batch_size must be positive')
    def parts():
        count=len(data['i'])
        for start in range(0,count,batch):
            edges=torch.arange(start,min(start+batch,count),device=state.device)
            value,terms=soft.objective(state,raw,widths,data,nodes,config,edges=edges,pair_cost=pair_cost,extension=extension)
            weight=len(edges)/count
            yield value*weight,{k:v*weight for k,v in terms.items()}
    def objective():
        if batch<edge_count:
            terms={}
            for _,piece in parts():
                for k,v in piece.items():terms[k]=terms.get(k,0)+v
            return sum(terms.values()),terms
        return soft.objective(state,raw,widths,data,nodes,config,pair_cost=pair_cost,extension=extension)
    def save(path):
        np.savez(path,state=state.detach().cpu().numpy(),widths=soft.decode_widths(raw,config).detach().cpu().numpy(),
            **extension.arrays(),**{k:d[k] for k in ('source_id','start','end','family')})
    def record(iteration):
        with torch.no_grad():
            value,terms=objective();p=state[:,3].sigmoid()
            row=dict(iteration=iteration,loss=float(value),seconds=time.monotonic()-started,
                enabled_percent=float(100*(p>=.5).float().mean()),mean_enabled_probability=float(p.mean()),
                width_factor=float((soft.decode_widths(raw,config)/widths).mean()),
                terms={k:float(v) for k,v in terms.items()})
            if diagnostics is not None:row.update(diagnostics(state,config))
            row.update(extension.diagnostics())
        history.append(row);atomic_write(root/'history.json',history,durable=False)
        print(json.dumps(row),flush=True)
        if config.get('save_diagnostic_checkpoints'):
            folder=root/'checkpoints';folder.mkdir(exist_ok=True);save(folder/f'{iteration:06d}.npz')
    record(0)
    def callback(iteration):
        if iteration%int(config.get('log_every',100))==0:record(iteration)
    result=uncertainty_joint(objective,state,raw,extension,int(config['steps']),data,config,callback,
                        optimize_rejection=not config.get('force_keep_fibers',False),
                        local_gap_scaling=config.get('lbfgs_local_gap_scaling',False),
                        direct_rejection=config.get('lbfgs_direct_rejection',False),fixed_enabled_mask=fixed,
                        objective_parts=parts if batch<edge_count else None,
                        frozen_fiber_mask=(np.isin(d['source_id'],config['freeze_source_ids'])
                            if config.get('freeze_source_ids') else None),
                        freeze_widths=bool(config.get('freeze_widths',False)))
    if history[-1]['iteration']!=result['iterations']:record(result['iterations'])
    save(root/'solution.npz')
    if measure_cuda:torch.cuda.synchronize(state.device)
    summary=dict(model='implicit_discrete_winding_v1',initial=history[0],final=history[-1],optimizer=result,optimizer_kind='scaled_lbfgs',
                 completed=True,seconds=time.monotonic()-started)
    if measure_cuda:
        summary['cuda_memory']=dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(state.device),
                                   peak_reserved_bytes=torch.cuda.max_memory_reserved(state.device),
                                   device_name=torch.cuda.get_device_name(state.device))
    atomic_write(root/'summary.json',summary,durable=False)
    print(json.dumps(dict(optimizer=result,final=history[-1])),flush=True)
    return summary
