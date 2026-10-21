"""Shared whole-fiber tangent sign alignment, optionally rejecting conflicts."""
import argparse
import json
from pathlib import Path
import sqlite3
import time

import numpy as np
from lasagna.inference_provenance import atomic_write, sha256_file
from .fiber_normal_sign import sign_alignment_terms, launch_tensor_solve


def family_settings(config):
    family=config.get('family','H').upper()
    if family not in ('H','V'):raise ValueError('Direction family must be H or V')
    return family,0 if family=='H' else 1,('ccw','cw') if family=='H' else ('up','down')


def circulation_prior(points, axis, spacing=16., axis_half=8.):
    return direction_prior(points,axis,spacing,axis_half,family='H')


def direction_prior(points, axis, spacing=16., axis_half=8., family='H'):
    """Arc-weighted signed tangent agreement with circulation (H) or up (V).

    Positive axis follows increasing native Z. H uses axis x outward radius;
    V uses that local axis directly, including on-axis samples. Undefined H
    circulation contributes neither numerator nor denominator; unsupported =>0.
    """
    family=family_settings(dict(family=family))[0]
    if spacing<=0 or axis_half<=0:raise ValueError('Positive sampling parameters required')
    delta=np.diff(points,axis=0);length=np.linalg.norm(delta,axis=1);valid=length>1e-9
    if not valid.any():return 0.,0.
    delta=delta[valid];length=length[valid];start=points[:-1][valid]
    counts=np.maximum(1,np.ceil(length/spacing).astype(np.int64))
    segment=np.repeat(np.arange(len(length)),counts)
    local=np.arange(len(segment))-np.repeat(np.cumsum(counts)-counts,counts)
    t=(local+.5)/counts[segment]
    xyz=start[segment]+t[:,None]*delta[segment]
    axis=axis[np.argsort(axis[:,2],kind='stable')]
    if np.any(np.diff(axis[:,2])<=0):raise ValueError('Umbilicus Z must be unique')
    def center(z):return np.c_[np.interp(z,axis[:,2],axis[:,0]),np.interp(z,axis[:,2],axis[:,1]),z]
    tangent=center(xyz[:,2]+axis_half)-center(xyz[:,2]-axis_half)
    radial=xyz-center(xyz[:,2]);circulation=np.cross(tangent,radial) if family=='H' else tangent
    norm=np.linalg.norm(circulation,axis=1);ok=np.isfinite(norm)&(norm>1e-9)
    weights=length[segment]/counts[segment];den=weights[ok].sum()
    if not den:return 0.,0.
    forward=delta[segment]/length[segment,None]
    agreement=np.sum(forward[ok]*circulation[ok]/norm[ok,None],axis=1)
    return float(np.dot(weights[ok],agreement)/den),float(den/length.sum())


def _ro(path):return sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro',uri=True)


def inherit_h_labels(ids,labels,inherited):
    """Preserve H results without changing V membership or earlier exclusions."""
    if not np.array_equal(ids,inherited['ids']):raise ValueError('Inherited H IDs differ')
    new=inherited['newly_broken']
    if np.any(new&(labels!=0)):raise ValueError('Inherited rejection is not an H fiber')
    result=labels.copy();result[new]=2
    if not np.array_equal(result,inherited['labels']):raise ValueError('Inherited labels changed earlier stage')
    signs=inherited['direction_sign_ccw']
    if not np.isin(signs[result==0],[-1,1]).all() or np.any(signs[result!=0]):raise ValueError('Invalid inherited H signs')
    if not np.array_equal(inherited['direction_sign_cw'],-signs):raise ValueError('Invalid inherited H gauge')
    return result


def prepare(config):
    from .fiber_hv_stage import verify_feature_source
    from .fiber_normal_sign_visualization import load_umbilicus_frame
    from vesuvius.neural_tracing.tracers.shared.annotations.afv import read_afv_polylines
    from vesuvius.neural_tracing.tracers.shared.geometry import arclength,tangents_at
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    start=time.monotonic();cache=Path(config['tensor_cache']);stamp=cache.with_suffix('.json')
    family,label,_=family_settings(config)
    signature={k:sha256_file(config[k]) for k in ('database','afv','features','stage1','hv_solution','umbilicus')}
    signature.update(version=1,tangent_half=config.get('tangent_half',16.),prior_spacing=config.get('prior_spacing',16.),
                     axis_half=config.get('axis_half',8.),convention='axis_increasing_z_cross_outward_radius_ccw')
    if family=='V':signature.update(family='V',convention='local_umbilicus_axis_increasing_z_up')
    if config.get('inherited_h_solution'):
        if family!='V':raise ValueError('H inheritance is only for the V pass')
        signature['inherited_h_solution']=sha256_file(config['inherited_h_solution'])
    if any(not np.isfinite(signature[k]) or signature[k]<=0 for k in ('tangent_half','prior_spacing','axis_half')):
        raise ValueError('Sampling spans must be positive and finite')
    if cache.exists() and stamp.exists() and json.loads(stamp.read_text())['signature']==signature:return
    feature=json.loads(Path(config['features']).read_text());verify_feature_source(config['database'],feature)
    if feature['afv_sha256']!=signature['afv'] or feature['umbilicus_sha256']!=signature['umbilicus']:
        raise ValueError('Source/feature identity mismatch')
    previous=json.loads((Path(config['hv_solution']).parent/'summary.json').read_text())
    if previous['input']['signature']['database']!=signature['database'] or previous['input']['signature']['stage1']!=signature['stage1']:
        raise ValueError('Earlier stage identity mismatch')
    with np.load(config['hv_solution']) as h:ids=h['ids'].copy();labels=h['labels'].copy()
    with np.load(config['stage1']) as s:
        if not np.array_equal(ids,s['ids']) or np.any(s['broken']&(labels!=2)):raise ValueError('Stage1 exclusions changed')
    if not np.all(np.diff(ids)>0) or not np.isin(labels,[0,1,2]).all():raise ValueError('Invalid source labels')
    inherited={}
    if config.get('inherited_h_solution'):
        hpath=Path(config['inherited_h_solution'])
        hmeta=json.loads((hpath.parent/'summary.json').read_text())['input']['signature']
        for k in ('database','afv','features','stage1','hv_solution','umbilicus'):
            if hmeta[k]!=signature[k]:raise ValueError('Inherited H provenance mismatch: '+k)
        with np.load(hpath) as h:
            labels=inherit_h_labels(ids,labels,h)
            inherited={f'inherited_h_direction_sign_{g}':h[f'direction_sign_{g}'].copy() for g in ('ccw','cw')}
    active=np.flatnonzero(labels==label)
    if not len(active):raise ValueError('No retained family fibers')
    with _ro(config['afv']) as db:meta={k:json.loads(v) for k,v in db.execute('SELECT key,value FROM metadata')}
    axis=load_umbilicus_frame(config['umbilicus'],meta['root'])
    with _ro(config['database']) as db:
        rows=np.asarray(db.execute('SELECT id,fiber_a,fiber_b,arc_a,arc_b FROM constraints WHERE parallel_score>perpendicular_score ORDER BY id').fetchall(),dtype=float).reshape(-1,5)
    idx=np.searchsorted(ids,rows[:,1:3].astype(np.int64))
    if np.any(idx>=len(ids)) or not np.array_equal(ids[idx],rows[:,1:3]):raise ValueError('Unknown event fiber')
    mask=(labels[idx]==label).all(1);original_parallel=len(rows);rows=rows[mask];idx=idx[mask]
    remap=np.full(len(ids),-1,np.int64);remap[active]=np.arange(len(active));ends=remap[idx]
    count=len(rows);end_ids=ends.T.reshape(-1);arcs=rows[:,3:5].T.reshape(-1)
    order=np.argsort(end_ids,kind='stable');sorted_ids=end_ids[order]
    starts=np.searchsorted(sorted_ids,np.arange(len(active)));stops=np.searchsorted(sorted_ids,np.arange(len(active)),side='right')
    tangents=np.empty((2*count,3),np.float32);prior=np.zeros(len(active),np.float32)
    support=np.zeros(len(active),np.float32);lengths=np.zeros(len(active))
    geometry=read_afv_polylines(config['afv'],ids[active])
    print(f'{family}-direction prepare: {len(active):,} fibers; {count:,} parallel events',flush=True)
    for n,(row,xyz) in enumerate(geometry):
        if row['id']!=ids[active[n]]:raise ValueError('Geometry order mismatch')
        arc=arclength(xyz);lengths[n]=arc[-1];ep=order[starts[n]:stops[n]];ea=arcs[ep]
        if not np.isfinite(ea).all() or np.any(ea<0) or np.any(ea>arc[-1]+1e-5):raise ValueError('Invalid event arc')
        tangents[ep]=tangents_at(xyz,arc,ea,half=signature['tangent_half'])
        prior[n],support[n]=direction_prior(xyz,axis,signature['prior_spacing'],signature['axis_half'],family)
        if n and n%50000==0:print(f'{family}-direction prepare {n:,}/{len(active):,}',flush=True)
    na,nb=tangents.reshape(2,-1,3);valid=(np.linalg.norm(na,axis=1)>.99)&(np.linalg.norm(nb,axis=1)>.99)
    valid&=np.isfinite(na).all(1)&np.isfinite(nb).all(1)
    dot=np.clip(np.sum(na[valid]*nb[valid],axis=1),-1,1)
    i,j=ends[valid].T
    positive_weight=abs(dot)>0
    graph=coo_matrix((np.ones(int(positive_weight.sum())),(i[positive_weight],j[positive_weight])),shape=(len(active),len(active))).tocsr()
    components,component=connected_components(graph,directed=False)
    degree=np.bincount(np.r_[i[positive_weight],j[positive_weight]],minlength=len(active))
    comp_evidence=np.bincount(component,weights=lengths*abs(prior),minlength=components)
    comp_lengths=np.bincount(component,weights=lengths,minlength=components)
    cache.parent.mkdir(parents=True,exist_ok=True);tmp=cache.with_suffix('.partial.npz')
    np.savez(tmp,ids=ids,labels=labels,active=active,prior=prior,prior_support=support,lengths=lengths,
             i=i,j=j,dot=dot,event_ids=rows[valid,0].astype(np.int64),component=component,
             tangent_a=na[valid],tangent_b=nb[valid],**inherited)
    tmp.replace(cache)
    stats=dict(signature=signature,original_fibers=len(ids),retained_h=len(active),excluded_v=int((labels==1).sum()),
        excluded_earlier_broken=int((labels==2).sum()),original_parallel_events=original_parallel,
        retained_hh_parallel_candidates=count,degenerate_events=int((~valid).sum()),events=len(i),
        zero_dot_events=int((dot==0).sum()),pair_weight=float(abs(dot).sum(dtype=np.float64)),
        isolated_h=int((degree==0).sum()),components=int(components),
        weak_prior_components=int((comp_evidence/np.maximum(comp_lengths,1e-12)<.05).sum()),
        no_prior_h=int((support==0).sum()),weak_prior_h=int((abs(prior)<.05).sum()),seconds=time.monotonic()-start)
    if family=='V':
        for old,new in [('retained_h','retained_v'),('excluded_v','excluded_h'),('retained_hh_parallel_candidates','retained_vv_parallel_candidates'),('isolated_h','isolated_v'),('no_prior_h','no_prior_v'),('weak_prior_h','weak_prior_v')]:
            stats[new]=stats.pop(old)
        stats['excluded_h']=int((labels==0).sum())
    atomic_write(stamp,stats,durable=False);print(f'{family}-direction prepared '+json.dumps(stats),flush=True)


def objective(logits,prior,i,j,dot,prior_weight,entropy_weight):
    import torch
    pair,unary=sign_alignment_terms(logits,prior,i,j,dot)
    # Stable binary entropy, symmetric under global gauge reversal.
    p=logits.sigmoid()
    entropy=-(p*torch.nn.functional.logsigmoid(logits)+(1-p)*torch.nn.functional.logsigmoid(-logits)).mean()
    return pair+prior_weight*unary+entropy_weight*entropy,pair,unary,entropy


def breakage_objective(logits,broken,prior,i,j,dot,prior_weight,entropy_weight,penalty):
    from .fiber_hv_stage import probabilities
    p=probabilities(logits,broken)  # kept reverse, kept forward, broken
    keep=1-p[:,2]
    pair,_=sign_alignment_terms(logits,prior,i,j,dot,keep)
    unary=(keep*(1-prior*(2*logits.sigmoid()-1))*.5).mean()
    rejection=penalty*p[:,2].mean()
    entropy=-(p*p.clamp_min(1e-12).log()).sum(-1).mean()
    return pair+prior_weight*unary+rejection+entropy_weight*entropy,pair,unary,entropy


def breakage_readout(logits,broken,prior,i,j,dot,component,lengths):
    from scipy.special import expit
    p=expit(logits);q=expit(broken)
    joint=np.stack(((1-q)*(1-p),(1-q)*p,q),axis=1)
    rejected=joint.argmax(1)==2;keep=~rejected;edges=keep[i]&keep[j]
    sign=np.where(logits>=0,1,-1);wrong=(dot*sign[i]*sign[j]<0)&edges;w=abs(dot)
    stats=readout(logits,prior,i,j,dot,component,lengths)
    stats['all_original_h_sign_diagnostics']={k:stats.pop(k) for k in (
        'uncertain_percent','forward_percent','circulation_agreement',
        'length_weighted_circulation_agreement','weak_result_gauge_components')}
    stats.update(new_rejections=int(rejected.sum()),new_rejections_percent=100*float(rejected.mean()),
        retained_arc_percent=100*float(lengths[keep].sum()/lengths.sum()),
        retained_events=int(edges.sum()),retained_events_percent=100*float(edges.mean()) if len(edges) else None,
        disagreeing_events=int(wrong.sum()),
        disagreement_percent=100*float(wrong.sum()/edges.sum()) if edges.any() else None,
        disagreement_original_percent=100*float(wrong.mean()) if len(wrong) else None,
        weighted_disagreement_percent=100*float(w[wrong].sum(dtype=np.float64)/w[edges].sum(dtype=np.float64)) if w[edges].sum()>0 else None,
        weighted_disagreement_original_percent=100*float(w[wrong].sum(dtype=np.float64)/w.sum(dtype=np.float64)) if w.sum()>0 else None,
        ambiguous_joint_percent=100*float((joint.max(1)<.9).mean()))
    return stats,joint,rejected


def readout(logits,prior,i,j,dot,component,lengths):
    from scipy.special import expit
    sign=np.where(logits>=0,1,-1);wrong=dot*sign[i]*sign[j]<0;weight=abs(dot)
    count=int(component.max())+1 if len(component) else 0
    comp_prior=np.bincount(component,weights=lengths*prior*sign,minlength=count)
    comp_len=np.bincount(component,weights=lengths,minlength=count)
    return dict(events=len(dot),disagreeing_events=int(wrong.sum()),
        disagreement_percent=100*float(wrong.mean()) if len(dot) else None,
        weighted_disagreement_percent=100*float(weight[wrong].sum(dtype=np.float64)/weight.sum(dtype=np.float64)) if weight.sum()>0 else None,
        uncertain_percent=float(100*((expit(logits)>.1)&(expit(logits)<.9)).mean()),
        forward_percent=float(100*(sign==1).mean()),circulation_agreement=float(np.mean(prior*sign)),
        length_weighted_circulation_agreement=float(np.sum(lengths*prior*sign)/lengths.sum()),
        weak_result_gauge_components=int((abs(comp_prior)/np.maximum(comp_len,1e-12)<.05).sum()))


def label_metrics(metrics,family):
    if family=='V':
        if 'all_original_h_sign_diagnostics' in metrics:
            metrics['all_original_v_sign_diagnostics']=metrics.pop('all_original_h_sign_diagnostics')
        target=metrics.get('all_original_v_sign_diagnostics',metrics)
        for key in ('circulation_agreement','length_weighted_circulation_agreement'):
            if key in target:target[key.replace('circulation','axial')]=target.pop(key)
    return metrics


def solve(config):
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; no CPU training fallback')
    torch.set_num_threads(4);torch.manual_seed(config.get('seed',0));torch.use_deterministic_algorithms(True)
    with np.load(config['tensor_cache']) as src:d={k:src[k] for k in src.files}
    family,label,(positive,negative)=family_settings(config)
    if not np.array_equal(d['active'],np.flatnonzero(d['labels']==label)):raise ValueError('Wrong family tensor cache')
    a,i,j,dot=[torch.from_numpy(d[k]).to('cuda:0') for k in ('prior','i','j','dot')]
    initial=torch.where(a==0,torch.full_like(a,.001),2*a);u=torch.nn.Parameter(initial.clone())
    penalty=config.get('breakage_weight');b=None
    if penalty is not None:
        if not np.isfinite(penalty) or penalty<=0:raise ValueError('Positive finite breakage weight required')
        warm_meta=json.loads((Path(config['warm_start']).parent/'summary.json').read_text())
        evidence=json.loads(Path(config['tensor_cache']).with_suffix('.json').read_text())
        if warm_meta['input']['signature']!=evidence['signature']:raise ValueError('Warm-start evidence differs')
        with np.load(config['warm_start']) as saved:
            if not np.array_equal(saved['ids'],d['ids']) or not np.array_equal(saved['active'],d['active']):raise ValueError('Warm-start identity mismatch')
            # Preserve fitted confidence: softening signs invents pair error on
            # high-degree consistent fibers, encouraging spurious rejection.
            u=torch.nn.Parameter(torch.from_numpy(saved[f'sign_logit_{positive}'].copy()).to('cuda:0'))
        b=torch.nn.Parameter(torch.full_like(a,-4.))
    opt=torch.optim.Adam([u] if b is None else [u,b],lr=config.get('lr',.05));steps=int(config.get('steps',10000));every=int(config.get('log_every',1000))
    if steps<=0 or every<=0 or config.get('prior_weight',.001)<0 or config.get('entropy_weight',.001)<0:
        raise ValueError('Invalid optimizer settings')
    root=Path(config['output_root']);root.mkdir(parents=True,exist_ok=True)
    if (root/'summary.json').exists():raise FileExistsError('Completed direction solve exists')
    def metrics(x):return label_metrics(readout(x,d['prior'],d['i'],d['j'],d['dot'],d['component'],d['lengths']),family)
    baselines=dict(source_forward=metrics(np.ones(len(a))),umbilicus_only=metrics(initial.detach().cpu().numpy()))
    start=time.monotonic();history=[]
    for step in range(steps+1):
        entropy=config.get('entropy_weight',.001)*min(1.,max(0.,(step/steps-.5)/.3))
        if b is None:total,pair,prior,ent=objective(u,a,i,j,dot,config.get('prior_weight',.001),entropy)
        else:total,pair,prior,ent=breakage_objective(u,b,a,i,j,dot,config.get('prior_weight',.001),entropy,penalty)
        if step%every==0 or step==steps:
            torch.cuda.synchronize();values=u.detach().cpu().numpy()
            measured=metrics(values)
            if b is not None:
                broken_values=b.detach().cpu().numpy()
                measured,joint,rejected=breakage_readout(values,broken_values,d['prior'],d['i'],d['j'],d['dot'],d['component'],d['lengths'])
                label_metrics(measured,family)
            record=dict(step=step,seconds=time.monotonic()-start,loss=float(total.detach()),pair_loss=float(pair.detach()),
                        prior_loss=float(prior.detach()),entropy_weight=entropy,**measured)
            if not np.isfinite(record['loss']):raise RuntimeError('Nonfinite direction loss')
            history.append(record);atomic_write(root/'history.json',history,durable=False)
            print(f'{family}-direction '+json.dumps(record),flush=True)
            if step==steps:break
        opt.zero_grad(set_to_none=True);total.backward();opt.step()
    probability=u.sigmoid().detach().cpu().numpy();sign=np.where(values>=0,1,-1).astype(np.int8)
    full=np.zeros(len(d['ids']),np.int8);full[d['active']]=sign
    extra={}
    if b is not None:
        full[d['active'][rejected]]=0
        newly_broken=np.zeros(len(d['ids']),bool);newly_broken[d['active']]=rejected
        labels=d['labels'].copy();labels[newly_broken]=2
        extra=dict(newly_broken=newly_broken,broken=labels==2,labels=labels,
                   broken_probability=joint[:,2],broken_logit=broken_values,
                   **{f'joint_probability_{positive}':joint,f'joint_probability_{negative}':joint[:,[1,0,2]]})
    else:extra.update(labels=d['labels'],broken=d['labels']==2)
    extra.update({k:d[k] for k in d if k.startswith('inherited_h_direction_sign_')})
    aliases={f'direction_sign_{positive}':full,f'direction_sign_{negative}':-full,
             f'sign_probability_{positive}':probability,f'sign_probability_{negative}':1-probability,
             f'sign_logit_{positive}':values,f'sign_logit_{negative}':-values}
    convention='CCW viewed from positive-Z axis end towards negative' if family=='H' else 'UP along local positive-Z umbilicus axis'
    np.savez(root/'solution.npz',ids=d['ids'],active=d['active'],family=np.asarray(family),
             direction_sign_positive=full,direction_sign_negative=-full,component=d['component'],
             convention=np.asarray(convention+'; sign multiplies stored AFV tangent'),**aliases,**extra)
    final=history[-1];final['changed_vs_umbilicus_percent']=float(100*(sign!=np.where(d['prior']>=0,1,-1)).mean())
    # Independent CPU/gauge audit of the saved data.
    with np.load(root/'solution.npz') as saved:
        assert np.array_equal(saved[f'direction_sign_{negative}'],-saved[f'direction_sign_{positive}'])
        assert np.all(saved[f'direction_sign_{positive}'][d['labels']!=label]==0)
        opposite=saved[f'direction_sign_{negative}'][d['active']]
        assert int((d['dot']*opposite[d['i']]*opposite[d['j']]<0).sum())==final['disagreeing_events']
        np.testing.assert_allclose(saved[f'sign_probability_{positive}']+saved[f'sign_probability_{negative}'],1,atol=1e-7)
    summary=dict(config=config,input=json.loads(Path(config['tensor_cache']).with_suffix('.json').read_text()),
        baselines=baselines,final=final,gpu=torch.cuda.get_device_name(),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        new_rejections=int(rejected.sum()) if b is not None else 0,
        warm_start_sha256=sha256_file(config['warm_start']) if b is not None else None,
        interpretation=f'Whole-{family} tangent gauge alignment; rejection is model inconsistency, not proven anatomical breakage')
    atomic_write(root/'summary.json',summary,durable=False)


def solve_sweep(config):
    root=Path(config['output_root']);results=[]
    for penalty in config['breakage_weights']:
        child=root/f'penalty_{penalty:g}'
        cfg=dict(config,output_root=str(child),breakage_weight=penalty)
        cfg.pop('breakage_weights')
        solve(cfg)
        summary=json.loads((child/'summary.json').read_text())
        results.append(dict(penalty=penalty,output_root=str(child),**summary['final']))
    # Prefer actually conflict-free outcomes, then retain the most arc/fibers.
    ranked=sorted(results,key=lambda r:(r['disagreeing_events']!=0,r['disagreeing_events'],
                                       -r['retained_arc_percent'],r['new_rejections']))
    summary=dict(config=config,results=results,selected=ranked[0])
    atomic_write(root/'summary.json',summary,durable=False)
    print(f'{family_settings(config)[0]}-direction breakage sweep '+json.dumps(summary),flush=True)


def run(config):
    prepare(config)
    return launch_tensor_solve(config,__name__,Path(__file__))


def report(config):
    root=Path(config['output_root']);summary=json.loads((root/'summary.json').read_text());cfg=summary['config']
    if family_settings(cfg)[0]!='H':raise ValueError('Legacy sign-only report is H-only; use fiber_direction_review for V')
    with np.load(cfg['tensor_cache']) as archive:d={k:archive[k] for k in archive.files}
    with np.load(root/'solution.npz') as archive:s={k:archive[k] for k in archive.files}
    for key in ('database','afv','features','stage1','hv_solution','umbilicus'):
        if sha256_file(cfg[key])!=summary['input']['signature'][key]:raise ValueError('Changed source: '+key)
    if not np.array_equal(s['ids'],d['ids']) or not np.array_equal(s['active'],d['active']):raise ValueError('Saved source identity mismatch')
    if not np.all(np.isfinite(s['sign_logit_ccw'])):raise ValueError('Nonfinite solution')
    if not np.array_equal(s['direction_sign_cw'],-s['direction_sign_ccw']):raise ValueError('Bad CW gauge')
    if not np.array_equal(s['sign_logit_cw'],-s['sign_logit_ccw']):raise ValueError('Bad CW logits')
    if np.any(s['direction_sign_ccw'][d['labels']!=0]!=0):raise ValueError('Non-H direction assignment')
    sign=s['direction_sign_ccw'][d['active']]
    if not np.isin(sign,[-1,1]).all():raise ValueError('Unassigned H fiber')
    wrong=d['dot']*sign[d['i']]*sign[d['j']]<0
    if int(wrong.sum())!=summary['final']['disagreeing_events']:raise ValueError('Saved disagreement mismatch')
    np.testing.assert_allclose(s['sign_probability_cw']+s['sign_probability_ccw'],1,atol=1e-7)
    changed=int((sign!=np.where(d['prior']>=0,1,-1)).sum())
    rows=[('Original stored point order',summary['baselines']['source_forward']),
          ('Umbilicus only',summary['baselines']['umbilicus_only']),('Aligned H directions',summary['final'])]
    lines=['# H-fiber direction alignment results','',
        'Stage3 after inward normals and fixed H/V. No new rejection, cutting, or global U/V solve.',
        'All numbers concern retained H–H parallel events; disagreement is model consistency, not anatomical error.','',
        '| Method | Disagreeing events | Event disagreement | Weighted disagreement |',
        '|---|---:|---:|---:|']
    for label,m in rows:lines.append(f"|{label}|{m['disagreeing_events']:,}|{m['disagreement_percent']:.6f}%|{m['weighted_disagreement_percent']:.6f}%|")
    stats=summary['input'];final=summary['final']
    lines+=['',f"Retained H fibers: {stats['retained_h']:,}; parallel events: {stats['events']:,}; new rejections:0.",
        f"Changed directions vs umbilicus-only: {changed:,} ({100*changed/stats['retained_h']:.6f}%).",
        f"Uncertain probability in(0.1,0.9): {final['uncertain_percent']:.6f}%. This is optimizer certainty, not calibrated confidence.",
        f"Graph: {stats['components']:,} components, including {stats['isolated_h']:,} isolated H fibers.",
        f"Weak final circulation evidence: {final['weak_result_gauge_components']:,} components; independent gauges remain a limitation.",
        f"Preparation: {stats['seconds']:.2f}s; 10000full-batch Adam steps: {final['seconds']:.2f}s on {summary['gpu']}.",
        f"Peak CUDA tensors: {summary['peak_allocated_bytes']/1024**2:.2f}MiB. Shared managed GPU, one run, not an isolated speed study.",
        '',f"Output: `{root/'solution.npz'}`.",
        'CW directions are exact negatives of CCW; probabilities complemented and logits negated. Pair agreement is identical.',
        'Global assignment remains paused. Its prior experimental runs did not consume this new stage.']
    destination=Path(config['report']);destination.write_text('\n'.join(lines)+'\n')
    audit=dict(solution_sha256=sha256_file(root/'solution.npz'),tensor_cache_sha256=sha256_file(cfg['tensor_cache']),
        source_hashes_verified=True,disagreeing_events=int(wrong.sum()),changed_vs_umbilicus=changed,
        earlier_exclusions_preserved=True,cw_negation_verified=True)
    atomic_write(root/'audit.json',audit,durable=False)
    print('\n'.join(lines),flush=True);return dict(report=str(destination),audit=audit)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--unmanaged',action='store_true');parser.add_argument('--devices')
    args=parser.parse_args()
    if not args.unmanaged:raise RuntimeError('Use approved managed runner')
    config=json.loads(args.config.read_text())
    if config.get('baseline_first'):
        baseline=dict(config,output_root=str(Path(config['output_root'])/'sign_only'))
        baseline.pop('breakage_weights',None)
        solve(baseline)
        config['warm_start']=str(Path(baseline['output_root'])/'solution.npz')
    (solve_sweep if 'breakage_weights' in config else solve)(config)
