"""Saved density same-winding H/V match statistics at per-side probability cutoffs."""
import json
from pathlib import Path
import numpy as np
import torch
from lasagna.inference_provenance import atomic_write, sha256_file
from . import fiber_implicit_winding as density
from . import fiber_soft_winding as soft
from .fiber_soft_winding_report import stats


def candidates(h, v, dh, dv, widths, sigma, u_h=None, u_v=None, sigma_v=None, maximum_sigma=None, physical_u=False,points_h=None,points_v=None):
    hh, lh = density.hypotheses(h, sigma, maximum_sigma)
    vv, lv = density.hypotheses(v, sigma if sigma_v is None else sigma_v, maximum_sigma)
    qh, uh, vh = density.hypothesis_points(hh, torch.zeros(len(h), dtype=torch.long), dh, widths,physical_u)
    qv, uv, vvert = density.hypothesis_points(vv, torch.ones(len(v), dtype=torch.long), dv, widths,physical_u)
    if points_h is not None:
        qh,uh,vh=points_h(hh);qv,uv,vvert=points_v(vv)
    same = vv[:, None, :, 0]-hh[:, :, None, 0]+torch.floor(uv[:, None, :]-uh[:, :, None]+.5)==0
    ph = lh.exp()*h[:, 3:4].sigmoid()
    pv = lv.exp()*v[:, 3:4].sigmoid()
    boundaries=torch.cat((widths.new_zeros(1),widths.cumsum(0)))
    xh=soft.uv.q_to_u(qh,widths,boundaries)
    xv=soft.uv.q_to_u(qv,widths,boundaries)
    if u_h is not None:
        xh=xh+u_h[:,None];xv=xv+u_v[:,None]
        qh=soft.uv.u_to_q(xh,widths,boundaries)
        qv=soft.uv.u_to_q(xv,widths,boundaries)
    return dict(same=same,ph=ph,pv=pv,u=qv[:,None,:]-qh[:,:,None],
        x=xv[:,None,:]-xh[:,:,None],v=vvert[:,None,:]-vh[:,:,None])


def select(c, cutoff):
    valid=c['same']&(c['ph'][:,:,None]>=cutoff)&(c['pv'][:,None,:]>=cutoff)
    product=c['ph'][:,:,None]*c['pv'][:,None,:]
    flat=torch.where(valid,product,-torch.ones_like(product)).flatten(1)
    keep=valid.flatten(1).any(1)
    pick=flat.argmax(1)
    rows=torch.arange(len(keep))[keep]
    return dict(count=int(keep.sum()),candidate_pairs=int(valid.sum()),
        rows=rows.numpy(),probability=product.flatten(1)[rows,pick[keep]].numpy(),
        **{key:c[key].flatten(1)[rows,pick[keep]].numpy() for key in ('u','x','v')})


def run(settings):
    torch.set_num_threads(int(settings.get('threads',4)))
    cutoffs=settings.get('cutoffs',[.1,.25,.5,.7,.8,.9,.95,.99])
    if any(not 0 < c <= 1 for c in cutoffs):raise ValueError('Cutoffs must be in (0,1]')
    results=[];reference={};reference_hash=None
    for path in settings['solve_roots']:
        root=Path(path);cfg=json.loads((root/'config.json').read_text())
        if json.loads((root/'summary.json').read_text())['model']!='implicit_discrete_winding_v1':
            raise ValueError('Expected implicit density solution')
        with np.load(root/'solution.npz') as src:s={k:src[k] for k in src.files}
        with np.load(cfg['tensor_cache']) as src:d={k:src[k] for k in src.files}
        for key in ('source_id','start','end','family'):
            if not np.array_equal(s[key],d[key]):raise ValueError('Identity changed: '+key)
        if sha256_file(cfg['tensor_cache'])!=cfg['tensor_sha256']:raise ValueError('Input tensors changed')
        if settings.get('compare_first'):
            if reference_hash is None:reference_hash=cfg['tensor_sha256']
            if reference_hash!=cfg['tensor_sha256']:raise ValueError('Comparison requires identical event tensors')
        i,j=d['i'],d['j'];eligible=(d['family'][i]!=d['family'][j])&~d['parallel']
        edge=np.flatnonzero(eligible);i=i[edge];j=j[edge];reverse=d['family'][i]!=0
        h=np.where(reverse,j,i);v=np.where(reverse,i,j)
        dh=np.where(reverse,d['ds_b'][edge],d['ds_a'][edge])
        dv=np.where(reverse,d['ds_a'][edge],d['ds_b'][edge])
        state=torch.as_tensor(s['state'],dtype=torch.float64);widths=torch.as_tensor(s['widths'],dtype=torch.float64)
        sigma, bound = density.saved_sigma(state,s,cfg)
        sigma_h = sigma[h] if torch.is_tensor(sigma) else sigma
        sigma_args = dict(maximum_sigma=bound)
        if torch.is_tensor(sigma):sigma_args['sigma_v']=sigma[v]
        with torch.no_grad():
            sh,sv=state[h],state[v]
            if cfg.get('intermediate_uv_geometry'):
                from .fiber_intermediate_uv import restore_model
                ext=restore_model(d,s,cfg)
                ext.curve.add_plan('match_h',h,dh);ext.curve.add_plan('match_v',v,dv)
                sigma_args['points_h']=lambda hypotheses:ext.intermediate.points(ext,state,widths,'match_h',hypotheses)
                sigma_args['points_v']=lambda hypotheses:ext.intermediate.points(ext,state,widths,'match_v',hypotheses)
            if bool(s.get('shared_curve_integer_geometry',False)):
                if not cfg.get('intermediate_uv_geometry'):
                    from .fiber_shared_curve import restore_query_curve
                    curve=restore_query_curve(d,s)
                    curve.add_plan('match_h',h,dh);curve.add_plan('match_v',v,dv)
                    sigma_args['points_h']=lambda hypotheses:curve.hypothesis_queries(state,widths,curve.plans['match_h'],hypotheses[...,0])
                    sigma_args['points_v']=lambda hypotheses:curve.hypothesis_queries(state,widths,curve.plans['match_v'],hypotheses[...,0])
            elif cfg.get('shared_curve_hypothesis_mapping'):
                from .fiber_shared_curve import saved_points
                sh,dh=saved_points(d,s,state.numpy(),widths.numpy(),h,dh,as_mapping=True)
                sv,dv=saved_points(d,s,state.numpy(),widths.numpy(),v,dv,as_mapping=True)
                sh=torch.as_tensor(sh);sv=torch.as_tensor(sv);sigma_args['physical_u']=True
            elif cfg.get('shared_curve_geometry'):
                from .fiber_shared_curve import saved_points
                _,ph,vh=saved_points(d,s,state.numpy(),widths.numpy(),h,dh)
                _,pv,vv=saved_points(d,s,state.numpy(),widths.numpy(),v,dv)
                sh=sh.clone();sv=sv.clone()
                sh[:,1]=torch.as_tensor(ph);sh[:,2]=torch.as_tensor(vh)
                sv[:,1]=torch.as_tensor(pv);sv[:,2]=torch.as_tensor(vv)
                dh=np.zeros_like(dh);dv=np.zeros_like(dv)
            c=candidates(sh,sv,torch.as_tensor(dh),torch.as_tensor(dv),widths,sigma_h,**sigma_args)
            uncorrected_v = c['v'].clone()
            uncorrected_u,uncorrected_x=c['u'].clone(),c['x'].clone()
            if 'u_offsets' in s:
                oa=s['u_offsets'][s['v_offset_a'][edge]]
                ob=s['u_offsets'][s['v_offset_b'][edge]]
                hu=torch.as_tensor(np.where(reverse,ob,oa),dtype=torch.float64)
                vu=torch.as_tensor(np.where(reverse,oa,ob),dtype=torch.float64)
                c=candidates(sh,sv,torch.as_tensor(dh),torch.as_tensor(dv),widths,sigma_h,hu,vu,**sigma_args)
            if 'v_offsets' in s:
                offset_a=s['v_offsets'][s['v_offset_a'][edge]]
                offset_b=s['v_offsets'][s['v_offset_b'][edge]]
                delta=np.where(reverse,offset_a-offset_b,offset_b-offset_a)
                c['v']=c['v']+torch.as_tensor(delta)[:,None,None]
            rows=[]
            trained, _ = soft.constraint_family_filter(d, cfg.get('filter_constraint_families', False))
            # Eligible H/V perpendicular events are always retained; balance over
            # the actual training graph, while preserving original event IDs.
            trained_weights = soft.uv.constraint_class_weights(trained['parallel'],dict(cfg,constraint_class_balance=True))
            if cfg.get('filter_constraint_families', False):
                classweight = np.full(len(edge), trained_weights[~trained['parallel']][0]) if len(edge) else np.empty(0)
            else:
                classweight = trained_weights[edge]
            for cutoff in cutoffs:
                r=select(c,cutoff);weights=r['probability']*classweight[r['rows']]
                factor=cfg.get('same_sheet_match_multiplier',1.)
                losses=dict(u=cfg.get('phase_weight',1)*factor*weights*r['u']**2,
                    v=cfg.get('metric_weight',.02)*factor*weights*(r['v']/cfg.get('v_pair_scale',1000))**2)
                rows.append(dict(cutoff=cutoff,count=r['count'],candidate_pairs=r['candidate_pairs'],
                    coverage_percent=100*r['count']/len(edge) if len(edge) else 0,
                    u_windings=stats(r['u']),u_base_voxels=stats(r['x']),v_base_voxels=stats(r['v']),
                    losses={k:dict(selected_mean=float(a.mean()) if len(a) else None,
                        full_objective_contribution=float(a.sum()/len(trained['i']))) for k,a in losses.items()}))
                if 'v_offsets' in s:
                    raw=select(dict(c,v=uncorrected_v),cutoff)
                    rows[-1]['uncorrected_v_base_voxels']=stats(raw['v'])
                    # All-constraints mode corrects the original loss too.
                    # The additional term is not multiplied by same-sheet10.
                    existing_v=r['v'] if cfg.get('offset_application')=='all_constraints' else raw['v']
                    raw_loss=cfg.get('metric_weight',.02)*factor*weights*(existing_v/cfg.get('v_pair_scale',1000))**2
                    extra=cfg['v_offset_match_weight']*weights*(r['v']/1000)**2
                    for key,values in [('v',raw_loss),('v_offset_match',extra)]:
                        rows[-1]['losses'][key]=dict(selected_mean=float(values.mean()) if len(values) else None,
                            full_objective_contribution=float(values.sum()/len(trained['i'])))
                if 'u_offsets' in s:
                    raw=select(dict(c,u=uncorrected_u,x=uncorrected_x),cutoff)
                    rows[-1]['uncorrected_u_windings']=stats(raw['u'])
                    rows[-1]['uncorrected_u_base_voxels']=stats(raw['x'])
                    existing_u=(r['u']+.5)%1-.5 if cfg.get('offset_application')=='all_constraints' else raw['u']
                    raw_loss=cfg.get('phase_weight',1)*factor*weights*existing_u**2
                    extra=cfg['u_offset_match_weight']*weights*(r['x']/1000)**2
                    for key,values in [('u',raw_loss),('u_offset_match',extra)]:
                        rows[-1]['losses'][key]=dict(selected_mean=float(values.mean()) if len(values) else None,
                            full_objective_contribution=float(values.sum()/len(trained['i'])))
                if settings.get('compare_first'):
                    if cutoff not in reference:reference[cutoff]=r
                    ref=reference[cutoff]
                    common,ri,ci=np.intersect1d(ref['rows'],r['rows'],return_indices=True)
                    rows[-1]['common_with_reference']=dict(count=len(common),
                        lost=ref['count']-len(common),gained=r['count']-len(common),
                        reference_u=stats(ref['u'][ri]),current_u=stats(r['u'][ci]),
                        reference_v=stats(ref['v'][ri]),current_v=stats(r['v'][ci]))
        sigma_label = cfg['winding_sigma']
        if cfg.get('learned_winding_sigma') or cfg.get('cavity_winding_sigma'):
            sigma_label = dict(mode='cavity' if cfg.get('cavity_winding_sigma') else 'learned', initial=cfg['winding_sigma'],
                bounds=[cfg.get('sigma_min',.15),cfg.get('sigma_max',1.)],
                quantiles=np.quantile(s['winding_sigma'],[.05,.5,.95]).tolist())
        results.append(dict(solve_root=path,sigma=sigma_label,total_hv_perpendicular=len(edge),rows=rows,
            hashes={str(p):sha256_file(p) for p in (root/'solution.npz',root/'config.json',Path(cfg['tensor_cache']))}))
    output=Path(settings['output_root']);output.mkdir(parents=True,exist_ok=True)
    atomic_write(output/'results.json',dict(results=results),durable=False)
    lines=['# Same-winding H/V perpendicular match audit','',
        'Each side must independently meet the cutoff for P(enabled)*P(integer hypothesis).',
        'Same winding is phase-aligned at the constraint point, without H/V measurement calibration.',
        'Each event is counted once using its highest-product qualifying pair; no GT filtering.',
        'U is actual same-winding separation, V is V-minus-H in base voxels.','']
    def fmt(value):return '—' if value is None else f'{value:.5f}'
    for result in results:
        lines.extend([f"## sigma={result['sigma']}; {result['total_hv_perpendicular']} H/V perpendicular events",'',
            '| Cutoff | Count | Coverage | U L1 (turns) | U p95 | U max | U L1 (vx) | V L1 (vx) | V p95 | V max |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|'])
        for row in result['rows']:
            vals=[row[k][stat] if row[k] else None for k,stat in
                [('u_windings','l1'),('u_windings','p95_abs'),('u_windings','max_abs'),('u_base_voxels','l1'),
                 ('v_base_voxels','l1'),('v_base_voxels','p95_abs'),('v_base_voxels','max_abs')]]
            lines.append(f"| {row['cutoff']:.0%} | {row['count']} | {row['coverage_percent']:.2f}% | "+' | '.join(map(fmt,vals))+' |')
        lines.append('')
    (output/'report.md').write_text('\n'.join(lines))
    print(json.dumps(dict(output=str(output),results=results)),flush=True)
    return results
