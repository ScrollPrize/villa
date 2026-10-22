"""Fixed-whole-reference solve on a supplied dataset, with optional deformation."""
import json
from pathlib import Path
import numpy as np
import torch
from lasagna.inference_provenance import atomic_write, sha256_file
from . import fiber_soft_winding as soft, fiber_implicit_winding as density
from .fiber_gt_constraint_score import arrays, enabled_l1, gt_solve_width_metrics
from .fiber_anchored_dataset import fixed_reference_endpoints
from .fiber_selected_dataset_eval import remap_events, aligned_widths


def score_cohorts(residual, enabled, parallel, sign, order):
    """Report valid empty/rejected cohorts without inventing zero-error metrics."""
    score={}
    for name,kind in [('all',np.ones(len(enabled),bool)),('parallel',parallel),('perpendicular',~parallel)]:
        use=kind&enabled;error=np.abs(residual[use]);correct=(residual[use]>=-.5)&(residual[use]<.5)
        signed=use&(sign!=0)
        total=int(kind.sum());count=int(use.sum());signed_count=int(signed.sum())
        violations=int((signed&(order<0)).sum())
        score[name]=dict(total=total,enabled=count,coverage_percent=100*count/total if total else None,
            l1=float(error.mean()) if count else None,
            correct_percent=float(100*correct.mean()) if count else None,
            median=float(np.median(error)) if count else None,
            p95=float(np.quantile(error,.95)) if count else None,max=float(error.max()) if count else None,
            sign_violations=violations,signed_events=signed_count,
            sign_violation_percent=100*violations/signed_count if signed_count else None)
    score['postfit_offset_diagnostic']=(enabled_l1(residual,enabled,parallel)['gt_global_offset']
        if enabled.any() else None)
    return score


def run(settings):
    from .fiber_workflows import run as dispatch
    from .fiber_implicit_match_report import run as match_report
    from .fiber_v_endpoint_offsets import run_exact
    from .fiber_density_sign_report import run as sign_report
    root=Path(settings['output_root']);root.mkdir(parents=True,exist_ok=True)
    baseline=Path(settings['baseline_root']);gtroot=Path(settings['gt_root'])
    cfg=json.loads(Path(settings['base_config']).read_text())['fiber_implicit_winding']
    cfg.update(settings.get('parameters',{}))
    gcfg=json.loads((gtroot/'uv_whole/config.json').read_text())
    paths=[Path(settings['base_config']),baseline/'tensors.npz',baseline/'init/solution.npz',
           baseline/'init/samples.npz',baseline/'gt_events.json',baseline/'solve/solution.npz',
           gtroot/'uv_whole/config.json',gtroot/'uv_whole/solution.npz',Path(gcfg['tensor_cache']),Path(gcfg['afv'])]
    signature=dict(settings=settings,inputs={str(p):sha256_file(p) for p in paths})
    if (root/'inputs.json').exists() and json.loads((root/'inputs.json').read_text())!=signature:
        raise ValueError('Inpainting input changed')
    atomic_write(root/'inputs.json',signature,durable=False)
    if (root/'results.json').exists():return json.loads((root/'results.json').read_text())
    d=arrays(baseline/'tensors.npz');gs=arrays(gtroot/'uv_whole/solution.npz');gd=arrays(gcfg['tensor_cache'])
    events=remap_events(json.loads((baseline/'gt_events.json').read_text()),d)
    references=fixed_reference_endpoints(d,events,gd,gs,gcfg)
    refpath=root/'fixed_reference_events.npz'
    if not refpath.exists():np.savez(refpath,**references)
    init=root/'init';init.mkdir(exist_ok=True)
    original=arrays(baseline/'init/solution.npz');state=original['state'].copy()
    gauge=int(settings['initial_global_offset']);state[:,0]+=gauge
    # Shift interval labels with the integer gauge; preserve the radial shape.
    widths,_=aligned_widths(original['widths'],original['widths'],-gauge)
    # Shift initialization metadata with the gauge; it is not a width loss prior.
    d=dict(d,radial_initial_centers=state[:,0].copy(),radial_initial_widths=widths.copy())
    tensor_path=root/'tensors.npz'
    if not tensor_path.exists():np.savez(tensor_path,**d)
    if not (init/'solution.npz').exists():
        np.savez(init/'solution.npz',**dict(original,state=state,widths=widths))
    atomic_write(init/'config.json',dict(tensor_sha256=sha256_file(tensor_path),
        soft_samples=str(baseline/'init/samples.npz')),durable=False)
    cfg.update(tensor_cache=str(tensor_path),output_root=str(root/'solve'),
        initial_solution=str(init/'solution.npz'),fixed_reference_events=str(refpath),
        devices=settings.get('devices',cfg.get('devices','0')))
    cfg['gpu_manager']=dict(cfg['gpu_manager'],exclusive=False,label='fixed-GT-fiber-inpainting')
    atomic_write(root/'solve_config.json',{'fiber_implicit_winding':cfg},durable=False)
    atomic_write(root/'surface_inputs.json',dict(predicted_curves=str(Path(settings['dataset'])/'selected.afv'),
        tensors=cfg['tensor_cache'],solution=str(root/'solve/solution.npz'),
        deformed_solution=str(root/'deformed/solution.npz') if 'deformation_parameters' in settings else None,
        fixed_reference_curves=gcfg['afv'],fixed_reference_solution=str(gtroot/'uv_whole/solution.npz'),
        fixed_reference_config=str(gtroot/'uv_whole/config.json'),reference_endpoints=str(refpath),
        reference_fibers=len(gd['family']),reference_policy='whole curves, exact native GT layout, always active; never optimized or deformed'),durable=False)
    if not (root/'solve/summary.json').exists():dispatch({'fiber_implicit_winding':cfg})
    sol=arrays(root/'solve/solution.npz');st=sol['state'];w=sol['widths'];owner=references['owner']
    mapped=st.copy();mapped[:,0]=np.floor(mapped[:,0]+.5)
    with torch.no_grad():
        q,phase,v=soft.points(torch.as_tensor(mapped[owner],dtype=torch.float64),torch.as_tensor(d['family'][owner]),
            torch.as_tensor(references['ds']),torch.as_tensor(w,dtype=torch.float64))
    rs=references['state'];par=references['parallel'];fd=references['family']-d['family'][owner]
    residual=rs[:,0]+rs[:,1]-q.numpy()+cfg['perpendicular_offset']*(~par)*fd-references['sign']*references['distance']
    enabled=st[owner,3]>=0
    order=references['sign']*(rs[:,0]-mapped[owner,0]+np.floor(rs[:,1]-phase.numpy()+.5)+cfg['perpendicular_offset']*(~par)*fd)
    score=score_cohorts(residual,enabled,par,references['sign'],order)
    score['widths']=gt_solve_width_metrics(w,gs['widths'],np.floor(rs[:,0]+rs[:,1]).astype(int),0)
    atomic_write(root/'gt_events.json',[dict(e,enabled=bool(enabled[n]),residual=float(residual[n])) for n,e in enumerate(events)],durable=False)
    matches=match_report(dict(solve_roots=[str(root/'solve')],output_root=str(root/'matches'),cutoffs=[.5,.7]))
    internal=sign_report(dict(solve_root=str(root/'solve'),output_root=str(root/'internal_signs'),cpu_threads=2,batch_size=128))
    deformation=None
    if 'deformation_parameters' in settings:
        deform=deformation_config(json.loads((root/'solve/config.json').read_text()),settings,root)
        atomic_write(root/'deform_config.json',{'fiber_exact_deformation':deform},durable=False)
        deformation=(json.loads((root/'deformed/summary.json').read_text()) if (root/'deformed/summary.json').exists() else run_exact(deform))
    if any(sha256_file(p)!=signature['inputs'][str(p)] for p in paths):raise ValueError('Fixed input changed')
    result=dict(score=score,matches=matches,deformation=deformation,internal_signs=internal,
        final=json.loads((root/'solve/summary.json').read_text())['final'],
        fibers=len(st),enabled=int((st[:,3]>=0).sum()),fixed_references=len(gd['family']),reference_events=len(owner),
        baseline_and_GT_unchanged=True,initial_gauge_shift=gauge,
        solution_sha256=sha256_file(root/'solve/solution.npz'),
        interpretation='Supervised fixed-GT inpainting; no post-fit gauge correction, not held-out evaluation')
    atomic_write(root/'results.json',result,durable=False);print(json.dumps(result),flush=True);return result


def deformation_config(solved,settings,root):
    """Only explicit caller overrides supplement the solved model settings."""
    return dict(solved,**settings['deformation_parameters']) | dict(
        initial_solution=str(root/'solve/solution.npz'),output_root=str(root/'deformed'))
