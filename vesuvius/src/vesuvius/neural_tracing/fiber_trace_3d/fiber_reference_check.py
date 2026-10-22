"""Frozen consensus regression benchmark for the seven-fiber geometry solve.

No optimization of candidates. A single global affine alignment removes gauge
and low-order distortion; no per-fiber alignment or integer reassignment.
"""
import json
from pathlib import Path
import numpy as np
import torch
from lasagna.inference_provenance import atomic_write, sha256_file
from .fiber_gt_constraint_score import arrays
from .fiber_bucket_uv import BucketUV
from .fiber_bucket_uv_visualization import current_eligible_edges
from .fiber_affine_geometry import direction_plan
from .fiber_uv_layout import geometry_points


def stats(x):
    x=np.abs(np.asarray(x))
    return dict(mean=float(x.mean()),p95=float(np.quantile(x,.95)),max=float(x.max()))


def align(points,target,scale,mask=None):
    """One global affine Q,V transform. Keep unaligned/offset-only diagnostics."""
    if mask is None:mask=np.ones(len(points),dtype=bool)
    center=points[mask].mean(0)
    design=np.column_stack(((points-center)/scale,np.ones(len(points))))
    target_center=target[mask].mean(0)
    coeff=np.linalg.lstsq(design[mask],(target[mask]-target_center)/scale,rcond=None)[0]
    fitted=(design@coeff)*scale+target_center
    diff=(points-target)[mask]
    return fitted,coeff[:2],stats((diff-np.median(diff,axis=0))[:,0])


@torch.no_grad()
def load(root,excluded=()):
    root=Path(root);cfg=json.loads((root/'config.json').read_text())
    d=arrays(cfg['tensor_cache']);s=arrays(root/'solution.npz')
    cfg=dict(cfg,shared_curve_integer_geometry=bool(s.get('shared_curve_integer_geometry',False)))
    for key in ('source_id','start','end','family'):
        if not np.array_equal(s[key],d[key]):raise ValueError('Input identity mismatch: '+key)
    ext=BucketUV(d,cfg,'cpu')
    with np.load(root/'solution.npz') as checkpoint:ext.restore(checkpoint)
    ext.cavity.values=torch.as_tensor(s['winding_sigma'])
    state=torch.as_tensor(s['state']);widths=torch.as_tensor(s['widths'])
    _,lp=ext.local_hypotheses(state);col=lp.argmax(1)
    prob=(lp.exp().amax(1)*state[:,3].sigmoid()).numpy()
    index=torch.arange(len(ext.owner));columns=col[ext.owner]
    layers={}
    positions={'shared':ext.shared_positions(state,widths)}
    if ext.intermediate is not None:positions['intermediate']=ext.intermediate.physical(widths)
    if not ext.predicted_only:positions['bucket']=ext.positions()
    evaluation_layer='shared' if ext.predicted_only else 'bucket'
    for name,pos in positions.items():
        p=pos.double()[index,columns].numpy()
        # Existing physical-U -> continuous-winding conversion, no rounding.
        q,_=geometry_points(dict(family=np.zeros(len(p))),p[:,0],p[:,1],
                            np.r_[0,np.cumsum(s['widths'].astype(float))],s['widths'],
                            np.arange(len(p)),np.zeros(len(p)))
        layers[name]=np.column_stack((q,p[:,1]))
    p=positions[evaluation_layer].double()[index,columns].numpy()
    r,l,f,di,gap,_=direction_plan(ext.owner,ext.v_offset_arc,ext.curve.family,ext.curve.direction)
    r,l,f,di,gap=[x.numpy() for x in (r,l,f,di,gap)]
    delta=p[r]-p[l];length=np.linalg.norm(delta,axis=1)
    signed=delta[np.arange(len(f)),f]*di
    angle=np.degrees(np.arccos(np.clip(signed/np.maximum(length,1e-12),-1,1)))
    selected=~np.isin(d['source_id'][ext.owner.numpy()[r]],excluded)
    metrics={name:dict(length_percent=stats(100*(length[(f==fam)&selected]/gap[(f==fam)&selected]-1)),
                       angle_degrees=stats(angle[(f==fam)&selected]),reversals=int((signed[(f==fam)&selected]<=0).sum()))
             for name,fam in [('H',0),('V',1)]}
    return dict(root=str(root),layers=layers,evaluation_layer=evaluation_layer,p=p,metrics=metrics,prob=prob,
                events=current_eligible_edges(d,ext,state,widths,col,prob),
                a=ext.v_offset_a.numpy(),b=ext.v_offset_b.numpy(),
                identity=np.column_stack((ext.owner.numpy(),ext.v_offset_arc.numpy())),
                source_id=d['source_id'],family=d['family'],direction=ext.curve.direction.numpy(),
                solution_sha256=sha256_file(root/'solution.npz'),
                config_sha256=sha256_file(root/'config.json'))


def evaluate(candidate,ref,meta):
    for key in ('identity','source_id','family','direction','a','b'):
        if not np.array_equal(candidate[key],ref[key]):raise ValueError('Benchmark identity mismatch: '+key)
    target=ref['consensus'];p=candidate['layers'][candidate.get('evaluation_layer','bucket')]
    included=~np.isin(ref['source_id'],meta.get('excluded_source_ids',[]))
    mask=included[ref['identity'][:,0].astype(int)]
    aligned,matrix,offset=align(p,target,ref['scale'],mask)
    error=aligned-target
    matrix_error=float(np.max(np.abs(matrix-np.eye(2))))
    events=ref['events'];missing=np.setdiff1d(events,candidate['events'])
    # Always test the frozen cohort, even if a candidate disabled its meetings.
    cohort=np.union1d(events,candidate['events'])
    cohort=cohort[mask[candidate['a'][cohort]]&mask[candidate['b'][cohort]]]
    gaps=candidate['p'][candidate['b'][cohort]]-candidate['p'][candidate['a'][cohort]]
    checks=dict(winding=bool(np.abs(error[mask,0]).max()<=.1),
                vertical=bool(np.abs(error[mask,1]).max()<=meta['v_tolerance_vx']),
                global_distortion=bool(matrix_error<=meta['affine_tolerance']),
                meetings=bool(np.linalg.norm(gaps,axis=1).max()<1),
                cohort=len(missing)==0,confident=bool((candidate['prob'][included]>=.5).all()))
    for family in ('H','V'):
        for kind in ('length_percent','angle_degrees'):
            for metric,value in candidate['metrics'][family][kind].items():
                checks[f'{family}_{kind}_{metric}']=bool(value<=meta['limits'][family][kind][metric])
        checks[family+'_forward']=candidate['metrics'][family]['reversals']==0
    per_fiber=[]
    for owner,source in enumerate(ref['source_id']):
        fiber_mask=ref['identity'][:,0]==owner
        per_fiber.append(dict(source_id=int(source),family='HV'[int(ref['family'][owner])],
                              scored=bool(included[owner]),
                              winding_error=stats(error[fiber_mask,0]),vertical_error_vx=stats(error[fiber_mask,1])))
    return dict(root=candidate['root'],evaluation_layer=candidate.get('evaluation_layer','bucket'),passed=all(checks.values()),checks=checks,
                per_fiber=per_fiber,
                winding_error=stats(error[mask,0]),vertical_error_vx=stats(error[mask,1]),
                offset_only_winding_error=offset,affine_matrix=matrix.tolist(),
                affine_deviation=matrix_error,missing_events=missing.tolist(),
                meeting_u_v_max=np.abs(gaps).max(0).tolist(),meeting_norm=stats(np.linalg.norm(gaps,axis=1)),
                geometry=candidate['metrics'],solution_sha256=candidate['solution_sha256'])


def run(settings):
    torch.set_num_threads(4)
    bundle=Path(settings['reference']);output=Path(settings['output_root'])
    if settings.get('create'):
        if bundle.exists():raise FileExistsError('Reference is immutable: '+str(bundle))
        excluded=settings.get('excluded_source_ids',[])
        models=[load(p,excluded) for p in settings['references']]
        base=models[0]
        for model in models[1:]:
            for key in ('identity','source_id','family','direction','a','b','events'):
                if not np.array_equal(model[key],base[key]):raise ValueError('References disagree on '+key)
        mask=~np.isin(base['source_id'][base['identity'][:,0].astype(int)],excluded)
        target=base['layers']['bucket'];scale=np.maximum(np.ptp(target[mask],axis=0),1)
        aligned=[align(m['layers']['bucket'],target,scale,mask)[0] for m in models]
        consensus=np.median(aligned,axis=0)
        fits=[align(m['layers']['bucket'],consensus,scale,mask) for m in models]
        # Reference-derived V/global-affine envelopes, explicit and frozen.
        # Only one global transform; no local or per-fiber warps are permitted.
        meta=dict(version=2,excluded_source_ids=excluded,v_tolerance_vx=max(1.,1.05*max(np.abs(p-consensus)[mask,1].max() for p,_,_ in fits)),
                  affine_tolerance=max(.01,1.05*max(np.abs(a-np.eye(2)).max() for _,a,_ in fits)),
                  winding_tolerance=.1,meeting_tolerance_vx=1.,
                  references=[dict(root=m['root'],solution_sha256=m['solution_sha256'],
                                   config_sha256=m['config_sha256']) for m in models],limits={})
        for family in ('H','V'):
            meta['limits'][family]={kind:{metric:max(m['metrics'][family][kind][metric] for m in models)+1e-6
                for metric in ('mean','p95','max')} for kind in ('length_percent','angle_degrees')}
        ref={key:base[key] for key in ('identity','source_id','family','direction','a','b','events')}
        events=ref['events'];ref['events']=events[mask[ref['a'][events]]&mask[ref['b'][events]]]
        ref.update(consensus=consensus,scale=scale)
        validation=[evaluate(m,ref,meta) for m in models]
        # Preserve incompatible references as evidence, never silently relax
        # winding tolerance or drop a disagreeing fiber to manufacture a pass.
        meta['all_references_pass']=all(r['passed'] for r in validation)
        bundle.mkdir(parents=True)
        np.savez(bundle/'reference.npz',**ref)
        meta['reference_sha256']=sha256_file(bundle/'reference.npz')
        atomic_write(bundle/'manifest.json',meta,durable=False)
        atomic_write(bundle/'reference_validation.json',validation,durable=False)
    meta=json.loads((bundle/'manifest.json').read_text())
    if sha256_file(bundle/'reference.npz')!=meta['reference_sha256']:raise ValueError('Reference changed')
    ref=arrays(bundle/'reference.npz')
    rows=[evaluate(load(p,meta.get('excluded_source_ids',[])),ref,meta) for p in settings['candidates']]
    output.mkdir(parents=True,exist_ok=True)
    atomic_write(output/'report.json',dict(reference=str(bundle),thresholds=meta,rows=rows),durable=False)
    print(json.dumps(dict(thresholds=meta,rows=rows)),flush=True)
    return rows
