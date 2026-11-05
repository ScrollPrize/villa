"""Add intact GT fibers, forced enabled by default, to a joint graph."""
import json
from pathlib import Path
import numpy as np
from lasagna.inference_provenance import atomic_write, sha256_file
from .fiber_gt_constraint_score import arrays
from .fiber_selected_dataset_eval import remap_events
from . import fiber_soft_winding as soft, fiber_uv_layout as uv


def combine(d, gd, events, config):
    """Reuse measured internal/cross constraints; do not import fitted GT UV. GT fibers keep their own UUIDs."""
    if len(d['links']) or len(gd['links']):raise ValueError('Expected intact unsplit fibers')
    n=len(d['family']);g=len(gd['family'])
    if len(np.unique(gd['source_id']))!=g:raise ValueError('GT fibers are split')
    mapped=gd['source_id'].copy()
    if np.isin(mapped,d['source_id']).any():
        raise ValueError('GT source IDs must be disjoint from ordinary fibers')
    piece_keys=('source_id','family','initial_k','start','end','phase','z','radius','lengths','node_ds','node_phase','node_z')
    out={k:np.concatenate((d[k],gd[k])) for k in piece_keys}
    out['source_index']=np.arange(n+g);out['ids']=out['source_id'].copy()
    out['earlier_labels']=np.r_[d['earlier_labels'],gd['earlier_labels']]
    out['width_initial']=d['width_initial'].copy()
    a=np.array([e['test_piece'] for e in events],dtype=np.int64)
    b=np.array([e['gt_piece'] for e in events],dtype=np.int64)
    if any(str(gd['source_id'][e['gt_piece']])!=e['gt_source'] for e in events):raise ValueError('GT cross-event identity mismatch')
    signed=np.array([e['signed_distance'] for e in events])
    cross=dict(i=a,j=b+n,
        ds_a=(np.array([e['test_arc'] for e in events])-d['start'][a])*np.sign(d['node_ds'][a,-1]),
        ds_b=(np.array([e['gt_arc'] for e in events])-gd['start'][b])*np.sign(gd['node_ds'][b,-1]),
        phase_a=np.array([e['test_angle'] for e in events]),phase_b=np.array([e['gt_angle'] for e in events]),
        phase_delta=np.array([e['phase_delta'] for e in events]),distance=np.abs(signed),sign=np.sign(signed),
        parallel=np.array([e['parallel'] for e in events],dtype=bool))
    for k in cross:
        out[k]=np.concatenate((d[k],gd[k]+n if k in ('i','j') else gd[k],cross[k]))
    out['event_ids']=np.arange(len(out['i']),dtype=np.int64)
    out['links']=np.empty((0,d['links'].shape[1]),dtype=d['links'].dtype)
    out['degree']=np.bincount(np.r_[out['i'],out['j']],minlength=n+g).astype(np.float32)
    centers,widths=uv.radial_initial_geometry(d,config)
    gc,_=uv.radial_initial_geometry(gd,config)
    out.update(radial_initial_centers=np.r_[centers,gc],radial_initial_widths=widths)
    out,filter_stats=soft.constraint_family_filter(out,True)
    out['degree']=np.bincount(np.r_[out['i'],out['j']],minlength=n+g).astype(np.float32)
    return out,mapped,filter_stats


def configure_gt_enablement(cfg,state,source_ids,gt_ids,forced=True):
    """Selectable GT uses ordinary finite logits and the same rejection cost."""
    cfg.pop('force_keep_fibers',None)
    cfg.pop('force_keep_source_ids',None)
    mask=np.isin(source_ids,gt_ids)
    if forced:
        state[mask,3]=100.
        cfg['force_keep_source_ids']=np.asarray(gt_ids).tolist()
    else:
        state[mask,3]=4.


def run(settings):
    if settings.get('report_only'):return report(settings)
    from .fiber_workflows import run as dispatch
    from vesuvius.neural_tracing.tracers.shared.annotations.fiber_store import fiber_volume,fibers_digest,read_polylines,write_fibers
    root=Path(settings['output_root']);root.mkdir(parents=True,exist_ok=True)
    baseline=Path(settings['baseline_root']);gt=Path(settings['gt_root'])
    cfg=json.loads(Path(settings['base_config']).read_text())['fiber_implicit_winding']
    cfg.update(settings.get('execution_parameters',{}))
    forced=settings.get('force_gt_enabled',True)
    gc=json.loads((gt/'uv_whole/config.json').read_text())
    if cfg.get('optimizer_kind')!='scaled_lbfgs':raise ValueError('Forced subset currently supported by scaled L-BFGS only')
    d=arrays(baseline/'tensors.npz');gd=arrays(gc['tensor_cache'])
    events=remap_events(json.loads((baseline/'gt_events.json').read_text()),d)
    merged,ids,filtered=combine(d,gd,events,cfg)
    paths=[Path(settings['base_config']),baseline/'tensors.npz',baseline/'gt_events.json',Path(gc['tensor_cache'])]
    signature=dict(settings=settings,inputs={str(p):sha256_file(p) for p in paths},gt_fibers=fibers_digest(gc['fibers']))
    if (root/'inputs.json').exists() and json.loads((root/'inputs.json').read_text())!=signature:raise ValueError('Changed joint GT inputs')
    atomic_write(root/'inputs.json',signature,durable=False)
    if (root/'results.json').exists():return json.loads((root/'results.json').read_text())
    volume=fiber_volume(cfg['fibers'])
    store=root/'fibers.vcobj'
    if not store.exists():
        curves=read_polylines(cfg['fibers'],d['source_id'],volume=volume)+read_polylines(gc['fibers'],volume=volume)
        write_fibers(store,curves,volume=volume)
    from .fiber_piece_split import split_graph
    cfg['fibers']=str(store)
    merged=split_graph(merged,cfg,ids if forced else ())
    tensor=root/'tensors.npz';np.savez(tensor,**merged)
    init=root/'init';init.mkdir(exist_ok=True)
    centers,widths=uv.radial_initial_geometry(merged,cfg)
    from .fiber_piece_split import initial_state
    state=initial_state(merged,centers,widths)
    configure_gt_enablement(cfg,state,merged['source_id'],ids,forced)
    np.savez(init/'solution.npz',state=state,widths=widths,**{k:merged[k] for k in ('source_id','start','end','family')})
    cfg.update(fibers=str(store),tensor_cache=str(tensor),initial_solution=str(init/'solution.npz'),
        output_root=str(root/'solve'),devices=settings.get('devices','0'))
    cfg['gpu_manager']=dict(cfg['gpu_manager'],exclusive=False,label='joint-GT-forced-enabled' if forced else 'joint-GT-selectable',
                            vram_bytes_per_gpu=settings.get('vram_bytes_per_gpu',32*1024**3))
    np.savez(init/'samples.npz',**soft.prepare_samples(cfg,merged))
    atomic_write(init/'config.json',dict(tensor_sha256=sha256_file(tensor),soft_samples=str(init/'samples.npz')),durable=False)
    atomic_write(root/'solve_config.json',{'fiber_implicit_winding':cfg},durable=False)
    audit=dict(test_fibers=len(d['family']),gt_fibers=len(gd['family']),gt_source_ids=ids.tolist(),
        cross_constraints=len(events),total_constraints=len(merged['i']),filter=filtered,
        initialization='fresh radial, no fitted GT coordinates',force_gt_enabled=forced,
        policy=('GT enabled fixed' if forced else 'GT rejection optimized like ordinary fibers')+'; all winding/geometry variables optimized; GT-assisted, not held-out')
    atomic_write(root/'dataset.json',audit,durable=False);print(json.dumps(audit),flush=True)
    if not (root/'solve/summary.json').exists():dispatch({'fiber_implicit_winding':cfg})
    result=arrays(root/'solve/solution.npz')
    logits=result['state'][np.isin(merged['source_id'],ids),3]
    if forced and not np.all(logits==100):raise ValueError('GT enabled state changed')
    from scipy.special import expit
    gt_rows=np.flatnonzero(np.isin(merged['source_id'],ids))
    gt_status=[dict(source_id=str(merged['source_id'][row]),start=float(merged['start'][row]),
                    end=float(merged['end'][row]),length_vx=float(merged['lengths'][row]),
                    enabled_probability=float(expit(logit)),enabled=bool(logit>=0))
               for row,logit in zip(gt_rows,logits)]
    result_report=dict(audit,gt_enablement=gt_status,solve=json.loads((root/'solve/summary.json').read_text()),solution_sha256=sha256_file(root/'solve/solution.npz'))
    atomic_write(root/'results.json',result_report,durable=False)
    if settings.get('report_after'):report(settings)
    return result_report


def report(settings):
    """Same frozen external GT metrics as the baseline, explicitly assisted."""
    from .fiber_gt_constraint_score import event_residuals,enabled_l1,gt_solve_width_metrics
    from .fiber_uv_arc_phase import event_arc_points
    from .fiber_density_sign_report import gt_ordering
    from .fiber_implicit_match_report import run as matches
    from .fiber_shared_curve import saved_segment_metrics
    root=Path(settings['output_root']);base=Path(settings['baseline_root']);gt=Path(settings['gt_root'])
    cfg=json.loads((root/'solve/config.json').read_text());gc=json.loads((gt/'uv_whole/config.json').read_text())
    d=arrays(cfg['tensor_cache']);sol=arrays(root/'solve/solution.npz')
    gd=arrays(gc['tensor_cache']);gs=arrays(gt/'uv_whole/solution.npz')
    events=remap_events(json.loads((base/'gt_events.json').read_text()),d)
    _,_,keep,pick=uv.hard_assignment(gs['state'],len(gs['widths']),gc,best_nonbroken=True)
    if not keep.all():raise ValueError('Disabled reference in scoring')
    mapped=sol['state'].copy();mapped[:,0]=np.floor(mapped[:,0]+.5)
    required=event_residuals(events,d,mapped,None,len(sol['widths']),gd,gs,pick,
        dict(cfg,soft_winding_model=True),gc,sol['widths'],saved_geometry=sol)
    enabled=sol['state'][np.array([e['test_piece'] for e in events]),3]>=0
    parallel=np.array([e['parallel'] for e in events],dtype=bool)
    score=enabled_l1(required,enabled,parallel)
    events=[dict(e,enabled=bool(enabled[n]),residual=float(required[n]-score['gt_global_offset'])) for n,e in enumerate(events)]
    order=gt_ordering(events,cfg);signs={}
    for name,kind in [('parallel',parallel),('perpendicular',~parallel)]:
        use=enabled&kind&np.array([e['normal_cosine']!=0 for e in events])
        signs[name]=dict(violations=int((use&(order<0)).sum()),events=int(use.sum()),percent=float(100*(use&(order<0)).sum()/use.sum()))
    q=event_arc_points(gd,gs['state'],pick,gs['widths'],np.array([e['gt_piece'] for e in events]),np.array([e['gt_arc'] for e in events]))
    score['gt_solve_widths']=gt_solve_width_metrics(sol['widths'],gs['widths'],np.floor(q).astype(int),score['gt_global_offset'])
    result=dict(score=score,signs=signs,segments=saved_segment_metrics(d,sol),
        matches=matches(dict(solve_roots=[str(root/'solve')],output_root=str(root/'matches'),cutoffs=[.5])),
        final=json.loads((root/'solve/summary.json').read_text())['final'],
        note='GT-assisted; scoring uses the same external frozen GT as baseline, not the jointly fitted GT positions')
    atomic_write(root/'comparison.json',result,durable=False)
    print(json.dumps(result),flush=True)
    return result
