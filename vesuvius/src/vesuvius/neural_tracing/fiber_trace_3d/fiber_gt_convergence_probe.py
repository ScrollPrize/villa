"""Full-objective small-GT diagnostics, without replacing the solver's losses."""
import json
import time
import argparse
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits
from lasagna.inference_provenance import atomic_write
from .fiber_gt_constraint_score import arrays
from .fiber_bucket_uv import BucketUV
from .fiber_gt_winding_reference import compare
from .fiber_density_sign_report import ordering
from . import fiber_soft_winding as soft, fiber_implicit_winding as density


def initial_order_labels(d,samples,state,leeway=.5):
    """Feasible integer labels from measured phases and perpendicular order.

    Initialization diagnostic only. No reference labels or enclosure targets.
    Return failure rather than silently relax a contradictory order cycle.
    """
    phases=[]
    for side,key in [('a','i'),('b','j')]:
        values=np.empty(len(d[key]))
        for f in range(len(state)):
            sample=samples['owner']==f;order=np.argsort(samples['ds'][sample])
            ds=samples['ds'][sample][order];p=samples['phase'][sample][order]
            rows=d[key]==f
            values[rows]=np.interp(d['ds_'+side][rows],ds,p)+state[f,1]-np.interp(0.,ds,p)
        phases.append(values)
    edges=[]
    for e,(i,j) in enumerate(zip(d['i'],d['j'])):
        if d['parallel'][e] or not d['sign'][e]:continue
        sign=d['sign'][e];a,b=(i,j) if sign>0 else (j,i)
        phase=sign*(phases[1][e]-phases[0][e])
        threshold=(1-sign*(int(d['family'][j])-int(d['family'][i])))*.5-leeway
        edges.append((int(a),int(b),int(np.ceil(threshold-phase))))
    labels=np.zeros(len(state),dtype=np.int64)
    for _ in range(len(state)):
        changed=False
        for a,b,lower in edges:
            if labels[b]<labels[a]+lower:labels[b]=labels[a]+lower;changed=True
        if not changed:break
    else:raise ValueError('Measured-phase perpendicular order has a positive cycle')
    labels+=int(np.rint(np.median(state[:,0]-labels)))
    return labels


@threadpool_limits.wrap(limits=1,user_api='blas')
def run(settings):
    if settings.get('devices') is not None and not settings.get('_managed_child',False):
        from .fiber_normal_sign import launch_tensor_solve
        request=dict(settings,gpu_manager=settings.get('gpu_manager',dict(
            exclusive=False,cpu_cores=4,vram_bytes_per_gpu=4*1024**3,label='GT-regression')))
        return launch_tensor_solve(request,__name__,Path(__file__))
    torch.set_num_threads(4)
    device='cuda:0' if settings.get('_managed_child',False) else 'cpu'
    root=Path(settings['output_root']);root.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((Path(settings['source'])/'config.json').read_text())
    cfg.update(settings.get('parameters',{}))
    if cfg.get('width_parameterization','softplus')!='softplus' and cfg.get('probe_optimizer','scaled_lbfgs')!='scaled_lbfgs':
        raise ValueError('Bounded width experiment requires scaled L-BFGS')
    d=arrays(cfg['tensor_cache']);samples=arrays(cfg['soft_samples'])
    if cfg.get('arc_prior_spacing_vx') is not None:
        from lasagna.inference_provenance import sha256_file
        samples=soft.prepare_samples(cfg,d,existing=samples)
        sample_path=root/'arc_prior_samples.npz'
        np.savez(sample_path,**samples)
        cfg.update(soft_samples=str(sample_path),samples_sha256=sha256_file(sample_path))
    data,nodes=soft.tensor_inputs(d,samples,cfg,device)
    initial=torch.tensor(arrays(cfg['initial_solution'])['widths'],dtype=torch.float64,device=device)
    reference=json.loads(Path(settings['reference']).read_text())
    out={}
    for trial in settings['trials']:
        saved=arrays(trial['checkpoint']);ext=BucketUV(d,cfg,device);ext.prepare_fields(d,samples)
        # This is a fresh optimizer comparison seeded with identical geometry,
        # not a controller resume. Sigma belongs to the trial configuration.
        if cfg.get('straight_fixed_length_geometry') and not bool(saved.get('straight_fixed_length_geometry',False)):
            # Explicit model comparison: retain initial N, anchor U/V and widths,
            # not the old model's independently trainable constraint knots.
            if trial.get('measured_phase_init') or trial.get('enclosure_width_init'):
                raise ValueError('Straight model comparison uses checkpoint anchors directly')
        else:
            ext.restore({k:v for k,v in saved.items() if k!='winding_sigma' and not k.startswith('cavity_')})
        if not ext.curve.ready:ext.curve.initialize(torch.tensor(saved['state'],device=device),torch.tensor(saved['widths'],device=device))
        if trial.get('measured_phase_init',False):
            # Diagnostic initialization only, from measured angular samples;
            # no reference winding labels, widths, or fitted GT coordinates.
            xy=ext.curve.coordinates().detach().clone()
            owner=ext.curve.owner.numpy();arcs=ext.curve.arc.numpy()
            for f in np.unique(owner):
                sample=samples['owner']==f;ds=samples['ds'][sample];ph=samples['phase'][sample]
                order=np.argsort(ds);query=(arcs[owner==f]-d['start'][f])*np.sign(d['node_ds'][f,-1])
                phase=np.interp(query,ds[order],ph[order])
                phase+=saved['state'][f,1]-np.interp(0.,ds[order],ph[order])
                xy[owner==f,0]=torch.tensor(phase)
            ext.curve.set_coordinates(xy)
        initial_state=saved['state'].copy()
        if trial.get('geometric_order_init',False):
            if not trial.get('measured_phase_init'):raise ValueError('Order initializer needs measured-phase geometry')
            initial_state[:,0]=initial_order_labels(d,samples,initial_state,cfg.get('sign_order_leeway',.5))
        width_scale=float(trial.get('initial_width_scale',1.))
        if not np.isfinite(width_scale) or width_scale<=0:
            raise ValueError('initial_width_scale must be finite and positive')
        # Width-only stress test: keep restored N/U/V geometry and the prior
        # unchanged. Do not reinitialize curves to hide the length mismatch.
        initial_widths=saved['widths']*width_scale
        if trial.get('enclosure_width_init',False):
            if not trial.get('geometric_order_init'):raise ValueError('Enclosure initializer needs measured-order labels')
            from .fiber_enclosure_width_init import fit_enclosure_widths
            from .fiber_geometry_cache import width_boundaries
            offsets=initial_state[:,0]
            phase0=[]
            for f in range(len(initial_state)):
                rows=samples['owner']==f;order=np.argsort(samples['ds'][rows])
                phase0.append(np.interp(0.,samples['ds'][rows][order],samples['phase'][rows][order]))
            phase0=np.asarray(phase0)
            # initial_order_labels used the initial U origin. Convert labels
            # to the measured phase origin used by the linear fit.
            origin_shift=initial_state[:,1]-phase0
            if not np.allclose(origin_shift,np.rint(origin_shift),atol=1e-5):
                raise ValueError('Initial and measured phase origins differ by a noninteger turn')
            offsets=offsets+np.rint(origin_shift)
            phase_q=offsets[samples['owner']]+samples['phase']
            active=np.arange(max(0,int(np.floor(phase_q.min()))-1),min(len(initial_widths),int(np.ceil(phase_q.max()))+2))
            gauge=int(active[0])
            initial_widths,anchors,qv,_=fit_enclosure_widths(d,samples,offsets,initial_widths,gauge,active)
            ww=torch.tensor(initial_widths);boundaries=width_boundaries(ww)
            xy=ext.curve.coordinates().detach().clone()
            for f in range(len(initial_state)):
                mask=ext.curve.owner==f
                ds=(ext.curve.arc[mask]-d['start'][f])*np.sign(d['node_ds'][f,-1])
                x=(boundaries[gauge]+anchors[f]+ds if d['family'][f]==0 else
                   soft.uv.q_to_u(torch.tensor(qv[f]),ww,boundaries).expand_as(ds))
                initial_state[f,0]=offsets[f]
                xy[mask,0]=soft.uv.u_to_q(x,ww,boundaries)-offsets[f]
                rows=samples['v_owner']==f
                v0=np.mean(samples['v_z'][rows]-(samples['v_ds'][rows] if d['family'][f] else 0))
                xy[mask,1]=v0+(ds if d['family'][f] else 0)
            ext.curve.set_coordinates(xy)
        state=torch.tensor(initial_state,dtype=torch.float64,device=device,requires_grad=True)
        raw=soft.encode_widths(torch.tensor(initial_widths,dtype=torch.float64,device=device),cfg).detach().requires_grad_(True)
        if cfg.get('cavity_winding_sigma'):
            ext.update(state,soft.decode_widths(raw,cfg),data,0)
        def objective():
            return soft.objective(state,raw,initial,data,nodes,cfg,pair_cost=density.pair_cost,extension=ext)
        def report():
            with torch.no_grad():
                loss,terms=objective();widths=soft.decode_widths(raw,cfg)
                s=dict(saved,**ext.arrays(),state=state.cpu().numpy(),widths=widths.cpu().numpy())
                ref=dict(reference,events=[e for e in reference['events'] if not d['parallel'][e['index']]])
                margin=ordering(density.map_state(state).cpu(),widths.cpu(),d,cfg,s).numpy()
                h,lp=density.hypotheses(state,ext.cavity.values,cfg.get('sigma_max',8))
                phase_error=ext.phase_prior_error(state,widths,nodes['phase'])
                phase_mean=float(phase_error.mean().sqrt()*360)
                return dict(loss=float(loss),terms={k:float(v) for k,v in terms.items()},
                    reference_perp=compare(d,s,ref)['error_windings'],
                    perp_sign_bad=int((margin[~d['parallel']]<0).sum()),
                    parallel_sign_bad=int((margin[d['parallel']]<0).sum()),
                    N=state[:,0].tolist(),labels=h[:,:,0].tolist(),probabilities=lp.exp().tolist(),
                    expected_phase_rms_degrees=phase_mean,widths=widths.tolist(),geometry=dict(ext.curve.last))
        result=dict(initial=report(),initial_width_scale=width_scale)
        # Actual per-term derivatives, including parameter scales and direction.
        loss,terms=objective();params=[state,raw]+list(ext.parameters())
        gradient={}
        for name,val in terms.items():
            if not val.requires_grad:continue
            g=torch.autograd.grad(val,params,retain_graph=True,allow_unused=True)
            gradient[name]=dict(N=g[0][:,0].tolist() if g[0] is not None else None,
                widths=g[1].tolist() if g[1] is not None else None,
                geometry_norm=float(sum(x.square().sum() for x in g[2:] if x is not None).sqrt())
                    if any(x is not None for x in g[2:]) else 0.)
        result['gradients']=gradient
        if settings.get('check_geometry_gradients',False):
            # Central differences of every objective term in deterministic
            # directions, with independent step sizes to expose cancellation.
            checks={};rng=torch.Generator().manual_seed(17)
            for name,param in [('widths',raw)]+list(ext.named_parameters()):
                direction=torch.randn(param.shape,generator=rng,dtype=param.dtype)
                direction/=direction.norm().clamp_min(1e-30)
                scale=100. if name=='widths' or name.endswith('.v') else .001
                direction*=scale;original=param.detach().clone()
                _,base_terms=objective();analytic={}
                for key,value in base_terms.items():
                    if not value.requires_grad:continue
                    g=torch.autograd.grad(value,param,retain_graph=True,allow_unused=True)[0]
                    analytic[key]=float((g*direction).sum()) if g is not None else 0.
                rows=[]
                for eps in (1e-2,1e-3,1e-4):
                    values=[]
                    for sign in (-1,1):
                        with torch.no_grad():
                            param.copy_(original+sign*eps*direction)
                            _,terms_at=objective()
                            values.append({k:float(v) for k,v in terms_at.items()})
                    rows.append(dict(epsilon=eps,finite_difference={k:(values[1][k]-values[0][k])/(2*eps) for k in analytic}))
                with torch.no_grad():param.copy_(original)
                checks[name]=dict(analytic=analytic,checks=rows)
            result['geometry_gradient_checks']=checks
            gaps=ext.curve.arc[1:]-ext.curve.arc[:-1]
            gaps=gaps[ext.curve.owner[1:]==ext.curve.owner[:-1]]
            result['segment_gaps']=dict(count=len(gaps),min=float(gaps.min()),median=float(gaps.median()),max=float(gaps.max()),below_one=int((gaps<1).sum()))
        # Independent finite difference of the full loss along each N coordinate.
        result['N_finite_difference']=[]
        for f in range(len(state)):
            old=float(state[f,0].detach());values=[]
            for step in (-1e-4,1e-4):
                with torch.no_grad():state[f,0]=old+step;values.append(float(objective()[0]))
            with torch.no_grad():state[f,0]=old
            result['N_finite_difference'].append((values[1]-values[0])/2e-4)
        started=time.monotonic()
        if trial.get('optimize') in ('joint','width'):
            from .fiber_scaled_lbfgs import uncertainty_joint
            history=[]
            def callback(iteration):
                if iteration%100==0:
                    r=report();history.append(dict(iteration=iteration,**r))
                    print(json.dumps(dict(trial=trial['name'],iteration=iteration,loss=r['loss'],reference=r['reference_perp'])),flush=True)
            kind=cfg.get('probe_optimizer','scaled_lbfgs')
            if kind=='adam':
                if trial['optimize']!='joint':raise ValueError('Adam comparison requires joint solve')
                # Same full objective, initial state, precision and geometry as
                # the L-BFGS probe. Only N in the redundant state is trainable;
                # trusted fibers remain active and U/V belong to curve nodes.
                mask=torch.zeros_like(state);mask[:,0]=1
                handle=state.register_hook(lambda g:g*mask)
                groups=ext.optimizer_groups()+[dict(params=[state],lr=cfg['lr']),
                                               dict(params=[raw],lr=cfg['width_lr'])]
                opt=torch.optim.Adam(groups)
                for iteration in range(1,trial.get('steps',1000)+1):
                    if cfg.get('cavity_winding_sigma'):
                        ext.update(state,soft.decode_widths(raw,cfg),data,iteration)
                    opt.zero_grad(set_to_none=True)
                    value,_=objective()
                    if not torch.isfinite(value):raise FloatingPointError('Nonfinite Adam objective')
                    value.backward();opt.step();callback(iteration)
                handle.remove()
                result['optimizer']=dict(kind='adam',iterations=iteration,
                    stopping_reason='step_limit',full_graph=True)
            elif kind=='scaled_lbfgs':
                result['optimizer']=uncertainty_joint(objective,state,raw,ext,trial.get('steps',1000),data,cfg,callback,
                                                optimize_width_only=trial['optimize']=='width')
            else:raise ValueError('Unknown probe_optimizer: '+kind)
            result['history']=history
        elif trial.get('optimize_N',True):
            fixed_xy=ext.curve.coordinates().detach().clone();fixed_raw=raw.detach().clone()
            def fun(x):
                with torch.no_grad():state[:,0]=torch.from_numpy(x)
                value,_=objective();g=torch.autograd.grad(value,state)[0][:,0]
                return float(value.detach()),g.detach().numpy().copy()
            fit=minimize(fun,state[:,0].detach().numpy().copy(),jac=True,method='L-BFGS-B',
                options=dict(maxiter=trial.get('steps',300),maxls=40,ftol=1e-12,gtol=1e-8))
            with torch.no_grad():state[:,0]=torch.from_numpy(fit.x)
            assert torch.equal(fixed_xy,ext.curve.coordinates().detach())
            assert torch.equal(fixed_raw,raw.detach())
            result['optimizer']=dict(success=bool(fit.success),message=str(fit.message),iterations=int(fit.nit),evaluations=int(fit.nfev))
        result['seconds']=time.monotonic()-started;result['final']=report()
        from .fiber_shared_curve import saved_segment_metrics
        result['final']['segments']=saved_segment_metrics(d,dict(saved,**ext.arrays(),
            state=state.detach().cpu().numpy(),widths=soft.decode_widths(raw,cfg).detach().cpu().numpy()),excluded_sources=(7,))
        dest=root/trial['name'];dest.mkdir(exist_ok=True)
        np.savez(dest/'solution.npz',**dict(saved,**ext.arrays(),state=state.detach().cpu().numpy(),widths=soft.decode_widths(raw,cfg).detach().cpu().numpy()))
        atomic_write(dest/'config.json',cfg,durable=False)
        atomic_write(dest/'report.json',result,durable=False)
        out[trial['name']]=result
        print(json.dumps(dict(trial=trial['name'],initial=result['initial']['loss'],final=result['final']['loss'],
            reference=result['final']['reference_perp'],seconds=result['seconds'],optimizer=result.get('optimizer'))),flush=True)
    atomic_write(root/'report.json',out,durable=False)
    return out


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--unmanaged',action='store_true')
    parser.add_argument('--devices')
    args=parser.parse_args()
    if not args.unmanaged:raise RuntimeError('Use the managed evaluation runner')
    settings=json.loads(args.config.read_text())
    if settings.get('check_geometry_gradients') or any(t.get('measured_phase_init') or
        t.get('geometric_order_init') or t.get('enclosure_width_init') or t.get('optimize')!='joint'
        for t in settings['trials']):
        raise ValueError('GPU GT harness currently supports joint regression trials only')
    result=run(dict(settings,_managed_child=True))
    atomic_write(Path(settings['output_root'])/'summary.json',result,durable=False)
