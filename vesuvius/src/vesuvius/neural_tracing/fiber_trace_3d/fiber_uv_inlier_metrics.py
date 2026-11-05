"""Fine geometry diagnostics conditioned on coarse-compatible selected buckets."""
import json
from pathlib import Path
import numpy as np


def measure(d,state,widths,config,offsets=None,cohort=None):
    from . import fiber_uv_layout as uv
    import torch
    if not config.get('bucket_shared_phase') or not config.get('bucket_position_split'):
        raise ValueError('Inlier diagnostics require shared-phase bucket/position split')
    geometry=uv.hard_geometry(d,state,widths,config,best_nonbroken=True)
    e=np.arange(len(d['i']));n=len(e);i=d['i'];j=d['j']
    _,_,coarse,active=uv.edge_readout(d,geometry,widths,e,config,state)
    sign=d['sign'];target=d['distance'];signed=sign!=0
    coarse_distance=np.where(signed,coarse*sign,np.abs(coarse))
    if config.get('integer_sheet_constraints',False):
        from .fiber_uv_integer_sheets import target as integer_target, margin
        fd=d['family'][j]-d['family'][i]
        target_k=integer_target(target,sign,fd,d['parallel'],config.get('perpendicular_layer_offset',.5))
        chosen=np.rint(coarse) if config.get('arc_phase_constraints',False) else coarse
        valid_sign=(~signed)|(chosen*sign>=margin(sign,fd,d['parallel']))
        current=active&valid_sign&signed&(chosen==target_k)
    else:
        valid_sign=(~signed)|(coarse*sign>=-config.get('sign_tolerance',1e-4))
        current=active&valid_sign&(np.abs(coarse_distance-target)<=.5)
    if offsets is not None and (offsets.shape[0]<2*n or offsets.shape[1]!=2 or not np.isfinite(offsets).all()):
        raise ValueError('Invalid logical endpoint offsets')
    _,_,_,_,u,v,b=geometry
    qa,va=uv.geometry_points(d,u,v,b,widths,i,d['ds_a'],None if offsets is None else offsets[:n])
    qb,vb=uv.geometry_points(d,u,v,b,widths,j,d['ds_b'],None if offsets is None else offsets[n:2*n])
    # Shared geometry, selected hypotheses only. In arc-phase mode these squared
    # residuals are geometric diagnostics, NOT the marginalized training loss.
    def t(x):return torch.as_tensor(x,dtype=torch.float64)
    delta,ordered,_,metric=uv.pair_geometry(t(qa[:,None]),t(qb[:,None]),t(va[:,None]),t(vb[:,None]),
        t(np.zeros_like(d['phase_delta']) if config.get('unified_winding_loss',False) else d['phase_delta']),
        t(d['family'][j]-d['family'][i]),t(sign),config)
    fine=np.where(signed,ordered.numpy().ravel(),np.abs(delta.numpy().ravel()))
    error=fine-target;u_loss=.5*(error/config.get('distance_sigma',.5))**2
    if config.get('integer_alignment_loss',False):
        fd=d['family'][j]-d['family'][i]
        integer_step=coarse-.5*fd
        error=qb-qa-integer_step
        evidence=np.exp(-.5*((coarse_distance-target)/config.get('distance_sigma',.5))**2)
        u_loss=evidence*.5*(error/config.get('distance_sigma',.5))**2
    if config.get('unified_winding_loss',False):
        from .fiber_uv_arc_phase import coarse_points
        pick=geometry[3]
        ca=coarse_points(d,state,pick,widths,i,d['ds_a'])
        cb=coarse_points(d,state,pick,widths,j,d['ds_b'])
        wi=widths[np.floor(ca).astype(int).clip(0,len(widths)-1)]
        wj=widths[np.floor(cb).astype(int).clip(0,len(widths)-1)]
        u_loss=u_loss*(.5*(wi+wj)/config['loss_reference_width'])**2
    if config.get('integer_sheet_constraints',False):
        error=qb-qa-(np.rint(coarse) if config.get('arc_phase_constraints',False) else coarse)
        u_loss=.5*(error/config.get('distance_sigma',.5))**2
    v_loss=np.minimum(((vb-va)/config.get('metric_scale',128.))**2,100.)
    weighted=config.get('distance_weight',1.)*config.get('position_distance_weight',.1)*u_loss+metric.numpy().ravel()
    def aggregate(mask):
        out={}
        for name,kind in [('parallel',d['parallel']),('perpendicular',~d['parallel'])]:
            sel=mask&kind;count=int(sel.sum())
            def avg(x):return float(np.mean(x[sel])) if count else None
            out[name]=dict(events=count,u_loss=avg(u_loss),v_loss=avg(v_loss),fine_loss=avg(weighted),
                u_error_l1=avg(np.abs(error)),v_rms_base=float(np.sqrt(avg((vb-va)**2))) if count else None,
                retained_percent=100*avg(active) if count else None,
                coarse_compatible_percent=100*avg(current) if count else None)
        rel=config.get('perpendicular_relative_weight',1.)
        vals=[out['parallel']['fine_loss'],out['perpendicular']['fine_loss']]
        out['balanced_fine_loss']=(vals[0]+rel*vals[1])/(1+rel) if all(v is not None for v in vals) else None
        return out
    frozen=current if cohort is None else np.asarray(cohort,dtype=bool)
    if frozen.shape!=current.shape:raise ValueError('Inlier cohort size mismatch')
    return dict(current=aggregate(current),fixed=aggregate(frozen),
        total_events=n,inlier_events=int(current.sum()),
        gate='retained endpoints, coarse sign valid, coarse target error <=0.5 winding',
        objective_kind='selected_geometric_residual_not_marginalized_loss' if config.get('arc_phase_constraints',False) else 'selected_fine_loss',
        readout='selected hypotheses; fine losses include offsets; no probability weighting'),current


def score_saved(config,cohort=None):
    with np.load(config['tensor_cache']) as archive:d={k:archive[k] for k in archive.files}
    with np.load(Path(config['output_root'])/'solution.npz') as s:
        if bool(config.get('event_uv_offsets'))!=('event_uv_offsets_base' in s):
            raise ValueError('Saved offset fields do not match configuration')
        for key in ('source_id','start','end','family'):
            if not np.array_equal(s[key],d[key]):raise ValueError('Solution identity mismatch: '+key)
        if 'event_uv_offsets_base' in s and not np.array_equal(s['event_ids'],d['event_ids']):
            raise ValueError('Offset event identity mismatch')
        return measure(d,s['state'],s['widths'],config,
            s['event_uv_offsets_base'] if 'event_uv_offsets_base' in s else None,cohort)


def run(settings):
    from lasagna.inference_provenance import atomic_write
    configs=[json.loads(Path(p).read_text())['fiber_uv_layout'] for p in settings['configs']]
    from .fiber_normal_sign import sha256_file
    if len({sha256_file(cfg['tensor_cache']) for cfg in configs})!=1:
        raise ValueError('Frozen cohort requires identical tensor/event ordering')
    _,cohort=score_saved(configs[0]);rows=[]
    for cfg in configs:
        metric,_=score_saved(cfg,cohort)
        rows.append(dict(output_root=cfg['output_root'],inlier=metric))
    root=Path(settings['output_root']);root.mkdir(parents=True,exist_ok=True)
    np.save(root/'cohort.npy',cohort)
    atomic_write(root/'results.json',rows,durable=False)
    print(json.dumps(rows),flush=True)
    return rows
