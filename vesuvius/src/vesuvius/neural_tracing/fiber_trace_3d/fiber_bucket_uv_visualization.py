"""Saved absolute bucket positions versus their shared winding predictions."""
import csv
import json
from pathlib import Path

import numpy as np
import torch

from . import fiber_bucket_uv as bucket
from . import fiber_soft_winding as soft
from . import fiber_uv_visualization as vis
from .fiber_gt_constraint_score import arrays


def eligible_edges(d,labels,joint,carry):
    """The >=0.5 joint-confidence same-sheet perpendicular metric cohort."""
    i,j=d['i'],d['j']
    return np.flatnonzero((~d['parallel'].astype(bool)) & (d['family'][i]!=d['family'][j]) &
                         (joint[i]>=.5) & (joint[j]>=.5) & (labels[j]==labels[i]-carry))


def current_eligible_edges(d,ext,state,widths,column,joint):
    if not ext.config.get('bucket_uv_dynamic_same_sheet',False):
        labels=ext.labels[torch.arange(len(state)),column].numpy()
        return eligible_edges(d,labels,joint,ext.carry.numpy())
    edges=np.flatnonzero((~d['parallel'].astype(bool)) & (d['family'][d['i']]!=d['family'][d['j']]))
    with torch.no_grad():
        if ext.intermediate is not None and not ext.config.get('pair_predicted_winding',False):q=ext.intermediate.coordinates()[...,0]
        else:
            from .fiber_geometry_cache import width_boundaries
            q=soft.uv.u_to_q(ext.shared_positions(state,widths)[...,0],widths,width_boundaries(widths))
        i,j=d['i'][edges],d['j'][edges]
        qa=q[ext.v_offset_a[edges],column[i]][:,None]
        qb=q[ext.v_offset_b[edges],column[j]][:,None]
        rows,_,_=bucket.dynamic_same_sheet_pairs(qa,qb,torch.as_tensor(d['sign'][edges]),
            torch.as_tensor(d['family'][j]-d['family'][i]),ext.config.get('sign_order_leeway',.5))
    edges=edges[rows.numpy()]
    return edges[joint[d['i'][edges]]*joint[d['j'][edges]]>.5]


def geometry(d, saved, cfg, support=None, support_edges=None, individual=False, show_all=False):
    """Select a single saved MAP label per fiber; never optimize/rematch."""
    cfg=dict(cfg)
    cfg['shared_curve_integer_geometry']=bool(saved.get('shared_curve_integer_geometry',False))
    cfg.setdefault('bucket_uv_perpendicular_nodes_only',bool(saved.get('bucket_uv_perpendicular_nodes_only',False)))
    state=torch.as_tensor(saved['state']);widths=torch.as_tensor(saved['widths'])
    td,_=soft.tensor_inputs(d,{},cfg,'cpu')
    ext=bucket.BucketUV(d,cfg,'cpu');ext.restore(saved)
    # Restore queues resume state; read-only rendering uses saved sigma directly.
    ext.cavity.values=torch.as_tensor(saved['winding_sigma'])
    with torch.no_grad():
        hypotheses,logp=ext.local_hypotheses(state)
        column=logp.argmax(1)
        conditional=logp.exp().amax(1).numpy()
        enabled=state[:,3].sigmoid().numpy();joint=enabled*conditional
        nodes=torch.arange(len(ext.owner));cols=column[ext.owner]
        predicted=ext.shared_positions(state,widths)[nodes,cols].numpy()
        actual=predicted.copy() if ext.predicted_only else ext.positions()[nodes,cols].numpy()
    owner=ext.owner.numpy();arc=ext.v_offset_arc.numpy()
    offsets=actual-predicted
    labels=hypotheses[torch.arange(len(state)),column,0].numpy()
    edges=current_eligible_edges(d,ext,state,widths,column,joint)
    ext.visualization_edges=edges
    ext.visualization_candidate_edges=current_eligible_edges(d,ext,state,widths,column,np.ones_like(joint))
    if support is not None:
        i,j=d['i'],d['j']
        edges=np.flatnonzero((~d['parallel'].astype(bool)) & (d['family'][i]!=d['family'][j]) &
            (support[i]>.5) & (support[j]>.5) & (labels[j]==labels[i]-ext.carry.numpy()))
        edges=np.intersect1d(edges,support_edges)
    eligible_nodes=np.r_[ext.v_offset_a.numpy()[edges],ext.v_offset_b.numpy()[edges]]
    maximum=np.zeros(len(state))
    np.maximum.at(maximum,owner[eligible_nodes],np.abs(offsets[eligible_nodes]).max(1))
    ids=np.unique(owner[eligible_nodes])
    if support is not None:ids=np.flatnonzero(support>.5)
    if individual:ids=np.flatnonzero(joint>.5)
    if show_all:ids=np.arange(len(state))
    with torch.no_grad():
        ext.terms(state,widths,td,torch.arange(len(d['i'])))
    if support is None and not ext.predicted_only and len(edges)!=ext.last['bucket_uv_meetings_p0.5']:
        raise ValueError('Visualization cohort differs from metric cohort')
    before_x=[];before_y=[];after_x=[];after_y=[]
    if not len(ids):raise ValueError('No fibers pass the visualization support filter')
    for idx in ids:
        rows=np.flatnonzero(owner==idx)
        if not len(rows):
            # Isolated fibers have no local coordinates; use the shared curve.
            q=torch.floor(state[idx,0]+.5)+state[idx,1]
            with torch.no_grad():
                u=soft.uv.q_to_u(q,widths,torch.cat((widths.new_zeros(1),widths.cumsum(0)))).item()
            ds=d['node_ds'][idx,[0,2]]
            x=u+(d['family'][idx]==0)*ds;y=float(state[idx,2])+(d['family'][idx]==1)*ds
            before_x.append(x);before_y.append(y);after_x.append(x);after_y.append(y)
            continue
        knots=np.unique(np.r_[d['start'][idx],arc[rows],d['end'][idx]])
        # Shared UV is linear along each family; extend to original endpoints.
        ds=(knots-d['start'][idx])*np.sign(d['node_ds'][idx,-1])
        origin=predicted[rows[0]]-np.array([d['family'][idx]==0,d['family'][idx]==1])*float(ext.node_ds[rows[0]])
        if cfg.get('shared_curve_geometry'):
            from .fiber_shared_curve import saved_points
            mapped=saved['state'].copy();mapped[:,0]=labels
            q,_,vv=saved_points(d,saved,mapped,saved['widths'],np.full(len(knots),idx),ds,shared_only=True)
            with torch.no_grad():
                ww=torch.as_tensor(saved['widths'],dtype=torch.float64)
                xx=soft.uv.q_to_u(torch.as_tensor(q),ww,torch.cat((ww.new_zeros(1),ww.cumsum(0)))).numpy()
            xy=np.column_stack((xx,vv))
        else:
            xy=origin+ds[:,None]*np.array([d['family'][idx]==0,d['family'][idx]==1])
        delta=np.column_stack([np.interp(knots,arc[rows],offsets[rows,k]) for k in range(2)])
        before_x.append(xy[:,0]);before_y.append(xy[:,1])
        after_x.append(xy[:,0]+delta[:,0]);after_y.append(xy[:,1]+delta[:,1])
    before=(ids,before_x,before_y,joint[ids],conditional[ids],None)
    after=(ids,after_x,after_y,joint[ids],conditional[ids],None)
    return before,after,maximum,predicted,actual,ext,labels


def visible_connection_segments(d, geometry, owner, arc, excluded=()):
    """Every recorded edge between drawn curves, including uncertain/nonmeeting edges.

    Interpolate on the displayed polylines, including parallel edges with no
    learned perpendicular node. This is visibility, not a metric eligibility test.
    """
    ids,x,y=geometry[:3];lookup={int(i):n for n,i in enumerate(ids)}
    knots={int(i):np.unique(np.r_[d['start'][i],arc[owner==i],d['end'][i]]) for i in ids}
    excluded=set(excluded);result=[]
    for e,(i,j) in enumerate(zip(d['i'],d['j'])):
        if e in excluded or int(i) not in lookup or int(j) not in lookup:continue
        pair=[]
        for idx,ds in ((i,d['ds_a'][e]),(j,d['ds_b'][e])):
            n=lookup[int(idx)];a=d['start'][idx]+ds*np.sign(d['node_ds'][idx,-1])
            pair.append([float(np.interp(a,knots[int(idx)],x[n])),float(np.interp(a,knots[int(idx)],y[n]))])
        result.append(pair)
    return result


def run(config):
    torch.set_num_threads(4)
    root=Path(config['bucket_comparison']);out=Path(config['output_root']);out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((root/'config.json').read_text());d=arrays(cfg['tensor_cache'])
    support=None;support_report=None;support_edges=None
    with np.load(root/'solution.npz') as checkpoint:
        for key in ('source_id','start','end','family'):
            if not np.array_equal(checkpoint[key],d[key]):raise ValueError('Visualization identity mismatch')
        if config.get('attachment_history_window') is not None:
            if cfg.get('bucket_uv_dynamic_same_sheet',False):
                raise ValueError('Dynamic same-sheet history requires endpoint geometry history; use current-state visualization')
            from .fiber_winding_history import attachment_support
            window=int(config['attachment_history_window'])
            with np.load(root/'winding_history.npz') as history:
                if not np.array_equal(history['source_id'],d['source_id']):raise ValueError('Temporal history identity mismatch')
                support,edges,edge_support,counts=attachment_support(history,d,checkpoint['bucket_uv_carry'],cfg,window)
            support_edges=edges
            keep=support>.5;length=d['end']-d['start']
            support_report=dict(window=window,threshold=.5,fibers=int(keep.sum()),total_fibers=len(keep),
                fiber_percent=float(100*keep.mean()),length_percent=float(100*length[keep].sum()/length.sum()))
            with (out/'attachment_support.csv').open('w',newline='') as f:
                w=csv.writer(f);w.writerow(['source_id','piece_index','support','perp_constraints','visible'])
                w.writerows((int(s),i,float(support[i]),int(counts[i]),bool(keep[i])) for i,s in enumerate(d['source_id']))
            with (out/'attachment_edge_support.csv').open('w',newline='') as f:
                w=csv.writer(f);w.writerow(['edge','mean_activation']);w.writerows(zip(edges,edge_support))
        background_only=False
        try:
            before,after,maximum,predicted,actual,ext,labels=geometry(d,checkpoint,cfg,support,support_edges,
                show_all=config.get('show_all_fibers',False))
        except ValueError as error:
            if str(error)!='No fibers pass the visualization support filter' or not config.get('individual_probability_background',False):
                raise
            before,after,maximum,predicted,actual,ext,labels=geometry(d,checkpoint,cfg,individual=True)
            background_only=True
        backgrounds={}
        if config.get('individual_probability_background',False):
            bg_before,bg_after,_,_,_,bg_ext,bg_labels=geometry(d,checkpoint,cfg,individual=True)
            backgrounds={'predicted':bg_before,'bucket':bg_after}
            if bg_ext.intermediate is not None:
                backgrounds['intermediate']=intermediate_geometry(d,checkpoint,bg_ext,bg_labels,bg_before)[0]
        saved={k:checkpoint[k] for k in ('state','widths')}
    cohort='both joint probabilities >=50%' if support is None else f'mean last-{window} activation >50% per fiber, over perpendicular constraints with current joint same-sheet mass >50%'
    if cfg.get('bucket_uv_dynamic_same_sheet',False):
        cohort='current same-sheet endpoint hypotheses, permitted geometric order, joint pair mass >50%'
    if config.get('show_all_fibers',False):
        cohort='All fibers at MAP placement; all connections shown; metric eligibility unchanged'
    support_edge_set=None if support_edges is None else set(support_edges.tolist())
    geometries={'predicted':before,'bucket':after}
    points={'predicted':predicted,'bucket':actual}
    if ext.intermediate is not None:
        middle,midpoints=intermediate_geometry(d,saved,ext,labels,before)
        geometries={'predicted':before,'intermediate':middle,'bucket':after}
        points['intermediate']=midpoints
    ids=before[0];shown=set(ids.tolist())
    shown_joint=dict(zip(ids,before[3]))
    palette=[(27,108,168) if d['family'][i]==0 else (208,105,22) for i in ids]
    if background_only:
        palette=[(203,223,238) for i in ids]
    if config.get('show_all_fibers',False):
        palette=[(203,223,238) if shown_joint[i]<.5 else color for i,color in zip(ids,palette)]
    all_views=list(geometries.values())+list(backgrounds.values())
    xs=[x for g in all_views for x in g[1]];ys=[y for g in all_views for y in g[2]]
    bounds=[min(map(np.min,xs)),max(map(np.max,xs)),min(map(np.min,ys)),max(map(np.max,ys))]
    annotations={k:[] for k in geometries};connections={k:[] for k in geometries};records=[]
    for e,(i,j) in enumerate(zip(d['i'],d['j'])):
        if support_edge_set is not None and e not in support_edge_set:continue
        if i not in shown or j not in shown:continue
        if config.get('show_all_fibers',False) and min(shown_joint[i],shown_joint[j])<.5:continue
        a=int(ext.v_offset_a[e]);b=int(ext.v_offset_b[e])
        same=e in ext.visualization_edges
        if not same:continue
        da=np.abs(actual[a]-predicted[a]);db=np.abs(actual[b]-predicted[b])
        row=dict(edge=e,source_a=int(d['source_id'][i]),source_b=int(d['source_id'][j]),
                 same_sheet_map=bool(same),a_u=float(da[0]),a_v=float(da[1]),b_u=float(db[0]),b_v=float(db[1]))
        for name,xy in points.items():
            connections[name].append([xy[a].tolist(),xy[b].tolist()])
            error=np.abs(xy[b]-xy[a]) if same else None
            row[name+'_meeting_u']=float(error[0]) if same else None
            row[name+'_meeting_v']=float(error[1]) if same else None
            oa,ob=(np.abs(xy[a]-predicted[a]),np.abs(xy[b]-predicted[b])) if name=='intermediate' else (da,db)
            for node,offset,side in [(a,oa,'a'),(b,ob,'b')]:
                if offset.max()>100:
                    annotations[name].append(dict(uv=xy[node].tolist(),peer=xy[b if node==a else a].tolist(),
                        family=int(d['family'][i if node==a else j]),text=f'{offset[0]:.0f}/{offset[1]:.0f}'))
            if same and error.max()>100:
                annotations[name].append(dict(uv=((xy[a]+xy[b])*.5).tolist(),text=f'{error[0]:.0f}/{error[1]:.0f}'))
        records.append(row)
    context_connections={}
    for name,g in geometries.items():
        if name in backgrounds:
            bg=backgrounds[name];extra=[n for n,i in enumerate(bg[0]) if i not in shown]
            g=(np.r_[g[0],bg[0][extra]],list(g[1])+[bg[1][n] for n in extra],list(g[2])+[bg[2][n] for n in extra])
        excluded={r['edge'] for r in records}
        if not config.get('show_all_fibers',False):
            joint=np.zeros(len(d['family']));joint[before[0]]=before[3]
            if name in backgrounds:joint[backgrounds[name][0]]=backgrounds[name][3]
            allowed=set(ext.visualization_edges.tolist())
            if name in backgrounds:
                # Faint context never enters metrics or the solid active links.
                candidates=ext.visualization_candidate_edges
                candidates=candidates[(joint[d['i'][candidates]]>.5)&(joint[d['j'][candidates]]>.5)]
                allowed.update(candidates.tolist())
            excluded.update(set(range(len(d['i'])))-allowed)
        context_connections[name]=visible_connection_segments(d,g,ext.owner.numpy(),ext.v_offset_arc.numpy(),excluded=excluded)
    diagnostic_threshold=config.get('label_meeting_fibers_above_vx')
    if diagnostic_threshold is not None:
        threshold=float(diagnostic_threshold)
        if not np.isfinite(threshold) or threshold<0:
            raise ValueError('label_meeting_fibers_above_vx must be finite and nonnegative')
        flagged={}
        joint_by_id=dict(zip(ids,before[3]));conditional_by_id=dict(zip(ids,before[4]))
        for row in records:
            if max(row['bucket_meeting_u'],row['bucket_meeting_v'])<=threshold:continue
            e=row['edge']
            for i,j in [(int(d['i'][e]),int(d['j'][e])),(int(d['j'][e]),int(d['i'][e]))]:
                entry=flagged.setdefault(i,dict(u=0.,v=0.,partners=set(),count=0))
                entry['u']=max(entry['u'],row['bucket_meeting_u'])
                entry['v']=max(entry['v'],row['bucket_meeting_v'])
                entry['partners'].add(int(d['source_id'][j]));entry['count']+=1
        image_ids={i:n+1 for n,i in enumerate(sorted(flagged))}
        with (out/'image_fiber_ids.csv').open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(['image_id','source_id','piece_index'])
            writer.writerows((number,int(d['source_id'][i]),i) for i,number in image_ids.items())
        for name,g in geometries.items():
            for n,i in enumerate(ids):
                if i in flagged:
                    # Fiber IDs attach between nodes, not to a constraint base.
                    lengths=np.hypot(np.diff(g[1][n]),np.diff(g[2][n]))
                    middle=int(lengths.argmax())
                    annotations[name].append(dict(uv=[float((g[1][n][middle]+g[1][n][middle+1])*.5),float((g[2][n][middle]+g[2][n][middle+1])*.5)],
                        text=str(image_ids[i]),family=int(d['family'][i])))
        lines=['# Latest-best large meeting errors','',
            f'Solution: `{root}`. Selection: final bucket U or V meeting error > {threshold:g} base voxels, '
            f'same-sheet perpendicular MAP meeting; {cohort}.', '',
            'Short image IDs label only the affected fibers, consistently in all three views; original source IDs are retained in the table and image_fiber_ids.csv. Maxima below are over selected meetings; '
            'U and V maxima need not occur at the same meeting. Joint probability includes enabled probability.', '',
            '**Temporal data unavailable:** this run saved no per-iteration fiber states. Entry age, switches, '
            'and mean probability over the last 100 iterations cannot be reconstructed from aggregate history. '
            'A new replay would be a separate result, not the history of this saved solution.', '',
            '| Image ID | Source ID | Piece index | H/V | MAP label | Joint p | Conditional p | Max U/V (vx) | Meetings | Partner source IDs | Its since entry | Mean p last 100 |',
            '|---|---|---:|---|---:|---:|---:|---|---:|---|---|---|']
        for i,r in sorted(flagged.items(),key=lambda item:max(item[1]['u'],item[1]['v']),reverse=True):
            lines.append(f"| {image_ids[i]} | {int(d['source_id'][i])} | {i} | {'H' if d['family'][i]==0 else 'V'} | {int(labels[i])} | "
                         f"{joint_by_id[i]:.3f} | {conditional_by_id[i]:.3f} | {r['u']:.0f}/{r['v']:.0f} | "
                         f"{r['count']} | {', '.join(map(str,sorted(r['partners'])))} | unavailable | unavailable |")
        if (root/'winding_history.npz').exists():
            from .fiber_winding_history import summarize
            with np.load(root/'winding_history.npz') as history:
                if not np.array_equal(history['source_id'],d['source_id']):
                    raise ValueError('Temporal history identity mismatch')
                temporal=summarize(history,labels)
            lines[6]='Temporal data sampled after every update, including iteration 0. Entry age is the uninterrupted duration at the final MAP label with joint probability >=0.5. Last-100 statistics cover iterations 901–1000; probability is mass at the final label, even when it was not MAP. These are fiber-level eligibility statistics, not proof that each partner was simultaneously eligible.'
            lines[8]='| Image ID | Source ID | Piece | H/V | MAP | Joint p | Max U/V vx | Partner source IDs | Entry it | Age | Mean p100 | Min p100 | Visible/100 | On/off100 | MAP changes100 |'
            lines[9]='|'+'---|'*15
            lines=lines[:10]
            for i,r in sorted(flagged.items(),key=lambda item:max(item[1]['u'],item[1]['v']),reverse=True):
                t=temporal[i]
                lines.append(f"| {image_ids[i]} | {int(d['source_id'][i])} | {i} | {'H' if d['family'][i]==0 else 'V'} | {int(labels[i])} | "
                    f"{joint_by_id[i]:.3f} | {r['u']:.0f}/{r['v']:.0f} | {', '.join(map(str,sorted(r['partners'])))} | "
                    f"{t['entry']} | {t['age']} | {t['mean100']:.3f} | {t['min100']:.3f} | {t['visible100']} | {t['switches100']} | {t['map_switches100']} |")
            audit=[dict(image_id=image_ids[i],piece=i,source_id=int(d['source_id'][i]),joint=float(joint_by_id[i]),
                        max_u=r['u'],max_v=r['v'],partners=sorted(r['partners']),**temporal[i])
                   for i,r in flagged.items()]
            soft.uv.atomic_write(out/'meeting_history.json',dict(fibers=audit,
                count=len(audit),entered_last100=sum(r['age'] is not None and r['age']<100 for r in audit),
                switched_MAP_last100=sum(r['map_switches100']>0 for r in audit),
                toggled_visibility_last100=sum(r['switches100']>0 for r in audit)),durable=False)
        (out/'large_meeting_fibers.md').write_text('\n'.join(lines)+'\n')
    # Identity is independent of the optional outlier/history diagnostic.
    for name,g in geometries.items():
        for n,i in enumerate(g[0]):
            lengths=np.hypot(np.diff(g[1][n]),np.diff(g[2][n]))
            k=int(lengths.argmax()) if len(lengths) else 0
            j=min(k+1,len(g[1][n])-1)
            annotations[name].append(dict(uv=[float((g[1][n][k]+g[1][n][j])*.5),
                float((g[2][n][k]+g[2][n][j])*.5)],family=int(d['family'][i]),
                text=('H' if d['family'][i]==0 else 'V')+str(int(d['source_id'][i]))))
    common=dict(cfg,render_height=config.get('render_height',1000),raster_map_only=True,vector_pdf=True,
                diagnostic_label_layout=True,
                diagnostic_label_font_size=config.get('diagnostic_label_font_size',6),
                fiber_line_width=config.get('fiber_line_width',2),
                fiber_vector_line_width=config.get('fiber_vector_line_width',.65),_bounds=bounds,_colors=palette,
                _cohort_label=f'{0 if background_only else len(ids)} fibers in {len(records)} metric meetings; {cohort}',
                _legend='H blue; V orange. Black connections show meeting gaps. Numbers: integer U/V error; endpoints show displacement, midpoints show meeting mismatch (>100vx).')
    reports={}
    if backgrounds:common['_legend']='Pale blue: individual joint probability >50%. Dashed blue: additional same-sheet context links (not metrics). '+common['_legend']
    for name,g in geometries.items():
        if ext.predicted_only and name!='predicted':continue
        reports[name]=vis.render_layout(out/(name+'.svg'),name,d,saved['state'],saved['widths'],
                                       dict(common,_geometry=g,_background_geometry=backgrounds.get(name),_context_connections=context_connections[name],_annotations=annotations[name],_connections=connections[name]))
    with (out/'constraints.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]) if records else ['edge']);writer.writeheader();writer.writerows(records)
    with (out/'fibers.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['source_id','max_component_displacement_vx','displacement_over_200vx'])
        for i in ids:writer.writerow([int(d['source_id'][i]),maximum[i],bool(maximum[i]>200)])
    result=dict(views=reports,red_fibers=0,displacement_over_200vx_fibers=int((maximum[ids]>200).sum()),metric_meetings=len(records),
                metric_cohort_verified=not ext.predicted_only,metrics=ext.last,annotations={k:len(v) for k,v in annotations.items()},
                solution_sha256=soft.uv.sha256_file(root/'solution.npz'),
                notes='Only fibers participating in same-sheet perpendicular metric meetings, both joint probabilities>=0.5. Displacement maxima and labels use only those meeting endpoints. Connections drawn explicitly. H blue; V orange, without outlier color override. Saved MAP labels; opacity joint probability. Final voxel view: linear node-offset interpolation, constant endpoint extrapolation. Intermediate view, if present: actual winding-coordinate interpolation/extrapolation followed by width mapping. Integer U/V annotations at endpoints and meeting midpoints.')
    result['background_fibers']=len(backgrounds['bucket'][0]) if backgrounds else 0
    result['background_only']=background_only
    if ext.predicted_only:
        result['notes']='Saved predicted/shared geometry only; no intermediate or final bucket layer. Solid links require permitted same-sheet geometric order and joint pair probability >0.5. This display filter is stricter than the reported per-endpoint probability cutoff; visible meeting statistics describe only the drawn links.'
    if background_only:
        result['notes']='No fibers participate in eligible metric meetings. Pale-blue fibers are individually confident background only; dashed links are context, not metric meetings. Empty-cohort errors are null, not zero.'
    result['context_connections']=len(context_connections['bucket'])
    result['connection_visibility']=('All recorded constraints between drawn fibers; non-metric links are dashed light blue.'
        if config.get('show_all_fibers',False) else 'Metric same-sheet links are solid; additional same-sheet links between individually confident background fibers are dashed when background display is enabled.')
    def stats(values):
        v=np.asarray(values)
        return dict(mean=float(v.mean()),p95=float(np.percentile(v,95)),max=float(v.max())) if v.size else dict(mean=None,p95=None,max=None)
    result['visible_meeting_errors_vx']={name:{axis:stats([r[name+'_meeting_'+axis] for r in records]) for axis in ('u','v')} for name in points}
    result['visible_displacement_vx']={axis:stats([r[side+'_'+axis] for r in records for side in ('a','b')]) for axis in ('u','v')}
    ww=torch.as_tensor(saved['widths']);bb=torch.cat((ww.new_zeros(1),ww.cumsum(0)))
    qpoints={name:soft.uv.u_to_q(torch.as_tensor(xy[:,0]),ww,bb).numpy() for name,xy in points.items()}
    aa=np.array([int(ext.v_offset_a[r['edge']]) for r in records],dtype=int)
    ab=np.array([int(ext.v_offset_b[r['edge']]) for r in records],dtype=int)
    result['visible_meeting_u_windings']={name:stats(np.abs(q[aa]-q[ab])) for name,q in qpoints.items()}
    if support is not None:
        result['attachment_support']=support_report
        result['unfiltered_solver_metrics']=result.pop('metrics')
        result['notes']=cohort+'. Only final MAP same-sheet connections between retained fibers are measured/drawn; no optimization change. Opacity remains final joint probability.'
    soft.uv.atomic_write(out/'summary.json',result,durable=False)
    print(json.dumps(result),flush=True)
    return result


def intermediate_geometry(d,saved,ext,labels,before):
    """Evaluate the actual winding-local curve, then map it to voxel U.

    Use the solver's interpolation, not straight voxel chords across unequal
    winding widths. All three plots retain the same metric-filtered fibers.
    """
    state=torch.as_tensor(saved['state']);widths=torch.as_tensor(saved['widths'])
    ids=before[0];owner=ext.owner.numpy();arc=ext.v_offset_arc.numpy()
    arcs=[np.unique(np.r_[d['start'][i],arc[owner==i],d['end'][i]]) for i in ids]
    query=np.repeat(ids,[len(a) for a in arcs]);stops=np.cumsum([len(a) for a in arcs])
    ds=(np.concatenate(arcs)-d['start'][query])*np.sign(d['node_ds'][query,-1])
    ext.curve.add_plan('visualization',query,ds)
    mapped=state.clone();mapped[:,0]=torch.as_tensor(labels)
    with torch.no_grad():
        q,_,v=ext.intermediate.points(ext,state,widths,'visualization',mapped[query,None,:])
        bounds=torch.cat((widths.new_zeros(1),widths.cumsum(0)))
        x=soft.uv.q_to_u(q[:,0],widths,bounds).numpy();y=v[:,0].numpy()
        col=(torch.as_tensor(labels)[ext.owner]-ext.labels[ext.owner,0]).long()
        nodes=ext.intermediate.physical(widths)[torch.arange(len(ext.owner)),col].numpy()
    return (ids,np.split(x,stops[:-1]),np.split(y,stops[:-1]),*before[3:]),nodes
