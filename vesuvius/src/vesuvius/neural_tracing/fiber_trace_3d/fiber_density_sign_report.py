"""Read-only MAP sign audit using the implicit density model's sheet ordering."""
import json
from pathlib import Path
import numpy as np
import torch
from lasagna.inference_provenance import atomic_write, sha256_file
from . import fiber_implicit_winding as density
from .fiber_winding_order import order_margin


from .fiber_winding_order import ordering


def gt_ordering(events, config):
    """Invert saved offset-corrected GT residual, then apply MAP sheet ordering."""
    parallel=np.array([e['parallel'] for e in events],dtype=bool)
    family=np.array([e['family_delta'] for e in events])
    displacement=np.array([e['residual']+e['signed_distance'] for e in events])-.5*family
    # The .5 above only undoes the historical serialized residual convention.
    # No measured distance or calibration enters the ordering decision itself.
    return order_margin(displacement,np.sign([e['normal_cosine'] for e in events]),
        family,parallel,config.get('sign_order_leeway',.5))


def run_gt(settings):
    results=[]
    for path in settings['gt_evaluations']:
        root=Path(path)
        events=json.loads((root/'gt_events.json').read_text())
        config=json.loads((root/'solve/config.json').read_text())
        saved=json.loads((root/'results.json').read_text())
        if saved['solution_sha256']!=sha256_file(root/'solve/solution.npz'):
            raise ValueError('Saved GT audit solution changed')
        order=gt_ordering(events,config)
        if settings.get('breakdown', False):
            from .fiber_gt_error_breakdown import run as run_breakdown
            run_breakdown(root, events, config, order, saved,
                          Path(settings['output_root']) / root.name)
        parallel=np.array([e['parallel'] for e in events],dtype=bool)
        enabled=np.array([e['enabled'] for e in events],dtype=bool)
        signed=np.array([e['normal_cosine']!=0 for e in events])
        rows=[]
        for cohort,keep in [('all',np.ones(len(events),bool)),('enabled',enabled),('disabled',~enabled)]:
            for kind,mask in [('parallel',parallel),('perpendicular',~parallel),('all',np.ones(len(events),bool))]:
                selected=keep&mask&signed;n=int(selected.sum());bad=selected&(order<0)
                rows.append(dict(cohort=cohort,kind=kind,events=n,violations=int(bad.sum()),
                    violation_percent=float(100*bad.sum()/n) if n else None,
                    ties=int((selected&(order==0)).sum())))
        results.append(dict(root=str(root),sign_weight=config.get('sign_violation_weight',0.),
            global_offset=saved['score']['gt_global_offset'],rows=rows,
            violations=[dict(e,signed_order=float(order[k])) for k,e in enumerate(events) if order[k]<0],
            hashes={str(p):sha256_file(p) for p in [root/'gt_events.json',root/'results.json',root/'solve/config.json',Path(__file__)]}))
    out=Path(settings['output_root']);out.mkdir(parents=True,exist_ok=True)
    atomic_write(out/'summary.json',dict(results=results,definition='Additional GT cross-event MAP sign audit; existing GT global offset; strict negative order, parallel ties separate'),durable=False)
    print(json.dumps([dict(weight=r['sign_weight'],offset=r['global_offset'],rows=r['rows']) for r in results]),flush=True)
    return results


def run(settings):
    if 'sweep_comparison' in settings:
        from .fiber_sign_rejection_report import compare
        return compare(**settings['sweep_comparison'])
    if 'gt_evaluations' in settings:return run_gt(settings)
    if 'cpu_threads' in settings:torch.set_num_threads(int(settings['cpu_threads']))
    root=Path(settings['solve_root']);out=Path(settings['output_root']);out.mkdir(parents=True,exist_ok=True)
    config=json.loads((root/'config.json').read_text())
    with np.load(root/'solution.npz') as f:s={k:f[k] for k in f.files}
    with np.load(config['tensor_cache']) as f:d={k:f[k] for k in f.files}
    for key in ('source_id','start','end','family'):
        if not np.array_equal(s[key],d[key]):raise ValueError('Mismatched identity')
    state=torch.as_tensor(s['state']);widths=torch.as_tensor(s['widths'])
    with torch.no_grad():
        sigma,bound=density.saved_sigma(state,s,config)
        hypotheses,lp=density.hypotheses(state,sigma,bound)
        chosen=lp.argmax(1);mapped=hypotheses[torch.arange(len(state)),chosen]
        confidence=lp.exp().amax(1)*state[:,3].sigmoid()
        order=ordering(mapped,widths,d,config,saved=s).numpy()
        endpoint_a,endpoint_b=state[d['i']],state[d['j']]
        da,db=d['ds_a'],d['ds_b']
        if config.get('shared_curve_geometry'):
            from .fiber_shared_curve import saved_points
            endpoint_a=endpoint_a.clone();endpoint_b=endpoint_b.clone()
            _,pa,va=saved_points(d,s,state.numpy(),widths.numpy(),d['i'],da)
            _,pb,vb=saved_points(d,s,state.numpy(),widths.numpy(),d['j'],db)
            if config.get('pair_predicted_winding',False):
                _,pa,_=saved_points(d,s,state.numpy(),widths.numpy(),d['i'],da,shared_only=True)
                _,pb,_=saved_points(d,s,state.numpy(),widths.numpy(),d['j'],db,shared_only=True)
            endpoint_a[:,1:3]=torch.as_tensor(np.column_stack((pa,va)))
            endpoint_b[:,1:3]=torch.as_tensor(np.column_stack((pb,vb)))
            da=np.zeros_like(da);db=np.zeros_like(db)
            if config.get('shared_curve_hypothesis_mapping') and not bool(s.get('shared_curve_integer_geometry',False)):
                endpoint_a,da=saved_points(d,s,state.numpy(),widths.numpy(),d['i'],d['ds_a'],as_mapping=True)
                endpoint_b,db=saved_points(d,s,state.numpy(),widths.numpy(),d['j'],d['ds_b'],as_mapping=True)
                endpoint_a=torch.as_tensor(endpoint_a,dtype=state.dtype)
                endpoint_b=torch.as_tensor(endpoint_b,dtype=state.dtype)
        masses=[]
        intermediate=None
        if config.get('intermediate_uv_geometry') or bool(s.get('shared_curve_integer_geometry',False)):
            from .fiber_intermediate_uv import restore_model
            intermediate=restore_model(d,s,config)
        batch=int(settings.get('batch_size',1024))
        if batch<1:raise ValueError('Sign audit batch must be positive')
        for start in range(0,len(d['i']),batch):
            edge=slice(start,start+batch);ii=d['i'][edge];jj=d['j'][edge];diag={}
            kwargs=dict(sigma_a=sigma[ii],sigma_b=sigma[jj]) if torch.is_tensor(sigma) else {}
            if intermediate is not None:
                kwargs['geometry']=intermediate.pair_geometry(state,widths,torch.arange(len(d['i']))[edge])
            density.pair_cost(endpoint_a[edge],endpoint_b[edge],torch.as_tensor(d['family'][ii]),
                torch.as_tensor(d['family'][jj]),torch.as_tensor(da[edge]),
                torch.as_tensor(db[edge]),widths,torch.as_tensor(d['sign'][edge]),
                torch.zeros(len(ii)),torch.as_tensor(d['parallel'][edge]).bool(),config,diagnostics=diag,**kwargs)
            masses.append(diag['wrong_probability'].numpy())
        wrong_probability=np.concatenate(masses)
        from .fiber_v_endpoint_offsets import fixed_geometry
        if config.get('shared_curve_geometry'):
            i,j=d['i'],d['j']
            if config.get('shared_curve_hypothesis_mapping'):
                _,pa,_=saved_points(d,s,mapped.numpy(),widths.numpy(),i,d['ds_a'])
                _,pb,_=saved_points(d,s,mapped.numpy(),widths.numpy(),j,d['ds_b'])
            separation=mapped[j,0]-mapped[i,0]+torch.floor(torch.as_tensor(pb-pa)+.5)
            meeting_mask=(confidence[i]>.5)&(confidence[j]>.5)&(separation==0)&torch.as_tensor(~d['parallel']&(d['family'][i]!=d['family'][j]))
        else:
            _,_,meeting_mask,_,_=fixed_geometry(d,state,widths,config,sigma=sigma,maximum_sigma=bound)
    signed=d['sign']!=0;i=d['i'];j=d['j'];enabled=s['state'][:,3]>=0
    confident=confidence.numpy()>.5
    cohorts={'all':np.ones(len(i),bool),'both_enabled':enabled[i]&enabled[j],
             'both_joint_above_half':confident[i]&confident[j],
             'at_least_one_disabled':~(enabled[i]&enabled[j])}
    rows=[]
    for name,cohort in cohorts.items():
        for kind,mask in [('all',np.ones(len(i),bool)),('parallel',d['parallel'].astype(bool)),('perpendicular',~d['parallel'].astype(bool))]:
            eligible=cohort&mask&signed;bad=eligible&(order<0);count=int(eligible.sum())
            sources=np.unique(np.r_[d['source_id'][i[bad]],d['source_id'][j[bad]]])
            rows.append(dict(cohort=name,kind=kind,signed_events=count,violations=int(bad.sum()),
                violation_percent=float(100*bad.sum()/count) if count else None,ties=int((eligible&(order==0)).sum()),
                incident_sources=len(sources),max_wrong_order=float(np.max(-order[bad],initial=0))))
            rows[-1].update(mean_wrong_probability=float(wrong_probability[eligible].mean()) if count else None,
                above_threshold_percent=float(100*(wrong_probability[eligible]>config.get('sign_probability_threshold',0.)).mean()) if count else None)
    bad=np.flatnonzero(signed&(order<0))
    events=[dict(event=int(e),source_a=int(d['source_id'][i[e]]),source_b=int(d['source_id'][j[e]]),
                 parallel=bool(d['parallel'][e]),signed_order=float(order[e]),both_enabled=bool(cohorts['both_enabled'][e]),
                 both_confident=bool(cohorts['both_joint_above_half'][e])) for e in bad]
    atomic_write(out/'violations.json',events,durable=False)
    result=dict(rows=rows,unsigned_events=int((~signed).sum()),
        sigma_mode='cavity' if config.get('cavity_winding_sigma') else 'learned' if config.get('learned_winding_sigma') else 'fixed',
        sigma_support_max=bound,
        deformation_meetings=int(meeting_mask.sum()),
        sign_violating_deformation_meetings=int((meeting_mask.numpy()&signed&(order<0)).sum()),
        definition='MAP continuous endpoint winding order; independent family-aware sign_order_leeway; zero boundary ties separate',
        current_loss='Signed Gaussian distance likelihood plus optional expected sign charge; no hard feasibility barrier',
        sign_violation_weight=config.get('sign_violation_weight',0.),
        sign_probability_threshold=config.get('sign_probability_threshold'),
        hashes={str(p):sha256_file(p) for p in [root/'config.json',root/'solution.npz',Path(config['tensor_cache']),Path(__file__),Path(density.__file__)]})
    atomic_write(out/'summary.json',result,durable=False);print(json.dumps(result),flush=True)
    return result
