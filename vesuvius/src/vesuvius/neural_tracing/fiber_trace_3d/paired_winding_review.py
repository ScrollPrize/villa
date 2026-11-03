"""Read-only comparison of realized winding placement in two saved solves."""
import csv
import json
from pathlib import Path
import numpy as np
import torch
from lasagna.inference_provenance import atomic_write,sha256_file
from .fiber_gt_constraint_score import arrays
from .fiber_shared_curve import saved_points
from .fiber_bucket_uv_visualization import current_eligible_edges
from .fiber_bucket_uv import BucketUV
from . import fiber_soft_winding as soft
from .fiber_uv_visualization import render_layout


def identity(d):
    keys=[(str(s),float(a),float(b),int(f)) for s,a,b,f in zip(d['source_id'],d['start'],d['end'],d['family'])]
    if len(set(keys))!=len(keys):raise ValueError('Duplicate fiber-piece identity')
    return {key:i for i,key in enumerate(keys)}


def winding_shift(delta):
    return int(np.floor(float(np.median(delta))+.5))


def prepared_geometry(d,s,c):
    """Existing saved-coordinate evaluator, queried once for all visible fibers."""
    state=torch.as_tensor(s['state']);widths=torch.as_tensor(s['widths'])
    ext=BucketUV(d,c,'cpu');ext.restore(s)
    ext.cavity.values=torch.as_tensor(s['winding_sigma'])
    with torch.no_grad():
        h,lp=ext.local_hypotheses(state);column=lp.argmax(1)
        conditional=lp.exp().amax(1).numpy();joint=state[:,3].sigmoid().numpy()*conditional
        labels=h[torch.arange(len(state)),column,0].numpy()
        ext.visualization_edges=current_eligible_edges(d,ext,state,widths,column,joint)
    ids=np.flatnonzero(joint>=.5);owner=ext.owner.numpy();arc=ext.v_offset_arc.numpy()
    knots=[np.unique(np.r_[d['start'][i],arc[owner==i],d['end'][i]]) for i in ids]
    count=[len(k) for k in knots];query=np.repeat(ids,count)
    ds=(np.concatenate(knots)-d['start'][query])*np.sign(d['node_ds'][query,-1])
    mapped=s['state'].copy();mapped[:,0]=labels
    q,_,v=saved_points(d,s,mapped,s['widths'],query,ds,shared_only=True)
    with torch.no_grad():
        x=soft.uv.q_to_u(torch.as_tensor(q),widths,torch.cat((widths.new_zeros(1),widths.cumsum(0)))).numpy()
    cuts=np.cumsum(count)[:-1]
    return (ids,np.split(x,cuts),np.split(v,cuts),joint[ids],conditional[ids],None),ext,labels,joint


def run(settings):
    torch.set_num_threads(4)
    out=Path(settings['output_root']);out.mkdir(parents=True,exist_ok=True)
    roots=[Path(settings[k]) for k in ('free','joint')]
    configs=[json.loads((r/'config.json').read_text()) for r in roots]
    data=[arrays(c['tensor_cache']) for c in configs]
    saved=[arrays(r/'solution.npz') for r in roots]
    maps=list(map(identity,data));common=sorted(maps[0].keys()&maps[1].keys())
    gs=[];extensions=[];labels=[];confidence=[]
    for d,s,c in zip(data,saved,configs):
        for key in ('source_id','start','end','family'):
            if not np.array_equal(d[key],s[key]):raise ValueError('Saved identity mismatch')
        g,ext,l,p=prepared_geometry(d,s,c)
        gs.append(g);extensions.append(ext);labels.append(l)
        confidence.append(p)
    eligible=[k for k in common if min(confidence[n][maps[n][k]] for n in (0,1))>=.5]
    if not eligible:raise ValueError('No matched fibers confident in both solves')
    # Fixed physical arc samples, including endpoints. Never compare N alone.
    arcs=[np.linspace(k[1],k[2],max(3,int(np.ceil((k[2]-k[1])/100))+1)) for k in eligible]
    counts=[len(a) for a in arcs];cuts=np.cumsum(counts)[:-1]
    qs=[]
    for n in (0,1):
        d=data[n];s=saved[n]
        ids=np.repeat([maps[n][k] for k in eligible],counts)
        ds=(np.concatenate(arcs)-d['start'][ids])*np.sign(d['node_ds'][ids,-1])
        state=s['state'].copy();state[:,0]=labels[n]
        q,_,_=saved_points(d,s,state,s['widths'],ids,ds,shared_only=True)
        qs.append(np.split(q,cuts))
    raw=[b-a for a,b in zip(*qs)]
    # Equal fiber weighting; remove only one integer global gauge, no per-fiber fit.
    gauge=winding_shift([np.median(x) for x in raw])
    records=[]
    for k,a,b,delta in zip(eligible,qs[0],qs[1],raw):
        delta=delta-gauge;step=winding_shift(delta)
        category='shifted' if step else ('partial' if np.max(np.abs(delta))>=.5 else 'stable')
        ia,ib=maps[0][k],maps[1][k]
        records.append(dict(source_id=k[0],start_arc=k[1],end_arc=k[2],family='HV'[k[3]],
            free_piece=ia,joint_piece=ib,free_probability=float(confidence[0][ia]),joint_probability=float(confidence[1][ib]),
            free_N_label=float(labels[0][ia]),joint_N_label=float(labels[1][ib]),
            free_mid_q=float(np.interp((k[1]+k[2])*.5,arcs[len(records)],a)),
            joint_mid_q_aligned=float(np.interp((k[1]+k[2])*.5,arcs[len(records)],b)-gauge),
            delta_median=float(np.median(delta)),delta_min=float(delta.min()),delta_max=float(delta.max()),
            winding_step=step,category=category,fraction_samples_over_half=float(np.mean(np.abs(delta)>=.5))))
    records.sort(key=lambda r:(r['category']=='stable',-abs(r['delta_median']),r['source_id']))
    flagged=[r for r in records if r['category']!='stable']
    for number,r in enumerate(flagged,1):r['image_id']=number
    for r in records:r.setdefault('image_id',None)
    with (out/'all_confident_fibers.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    with (out/'changed_fibers.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(flagged)
    summary=dict(free=str(roots[0]),joint=str(roots[1]),solution_sha256=[sha256_file(r/'solution.npz') for r in roots],
        common_fibers=len(common),confident_both=len(eligible),
        confident_free=len(gs[0][0]),confident_joint=len(gs[1][0]),
        confident_common_free_only=int(sum(confidence[0][maps[0][k]]>=.5 and confidence[1][maps[1][k]]<.5 for k in common)),
        confident_common_joint_only=int(sum(confidence[1][maps[1][k]]>=.5 and confidence[0][maps[0][k]]<.5 for k in common)),
        extra_joint_fibers=len(maps[1].keys()-maps[0].keys()),global_integer_offset_joint_minus_free=gauge,
        counts={cat:sum(r['category']==cat for r in records) for cat in ('stable','shifted','partial')},
        by_family={f:{cat:sum(r['category']==cat and r['family']==f for r in records) for cat in ('stable','shifted','partial')} for f in ('H','V')},
        step_histogram={str(i):sum(r['winding_step']==i for r in records) for i in sorted(set(r['winding_step'] for r in records))},
        definition='Joint enabled*MAP probability >=.5 in BOTH solves. Realized continuous winding evaluated at matched physical arcs (<=100vx spacing), after one equal-fiber-weighted median integer gauge. Shifted: nearest-integer median change !=0; partial: median unchanged but some sampled arc changes >=.5. Not a comparison of N labels alone.')
    atomic_write(out/'comparison.json',summary,durable=False)
    lines=['# Confident winding changes','',summary['definition'],'',
        f'Global joint-minus-free integer gauge removed: {gauge}. Confident in both: {len(eligible)}. '+
        f"Shifted: {summary['counts']['shifted']}; partial: {summary['counts']['partial']}; stable: {summary['counts']['stable']}.",'',
        'Dotted: changed winding; solid: unchanged or not comparable. H blue / V orange throughout. '+
        'IDs are identical across highlighted views. '+
        'Layouts retain their own fitted voxel coordinates and winding widths, with common plot bounds; only the analysis removes the integer gauge. '+
        'Continuous residual changes are shown, so a borderline half-turn change is not misrepresented as an exact whole-turn displacement.','',
        '|ID|Source|H/V|p free/joint|Median Δw|Min/max Δw|Step|Kind|','|---:|---:|---|---|---:|---|---:|---|']
    for r in flagged:
        lines.append(f"|{r['image_id']}|{r['source_id']}|{r['family']}|{r['free_probability']:.2f}/{r['joint_probability']:.2f}|{r['delta_median']:+.3f}|{r['delta_min']:+.3f}/{r['delta_max']:+.3f}|{r['winding_step']:+d}|{r['category']}|")
    (out/'changed_fibers.md').write_text('\n'.join(lines)+'\n')
    # Established renderer and normal visibility/link rules; no new raster engine.
    allx=[x for g in gs for x in g[1]];ally=[y for g in gs for y in g[2]]
    bounds=[min(map(np.min,allx)),max(map(np.max,allx)),min(map(np.min,ally)),max(map(np.max,ally))]
    for n,name in enumerate(('free','joint')):
        (out/name).mkdir(exist_ok=True)
        d=data[n];g=gs[n];ext=extensions[n]
        by_piece={r['free_piece' if n==0 else 'joint_piece']:r for r in flagged}
        colors=[];annotations=[];priority=[]
        for pos,i in enumerate(g[0]):
            r=by_piece.get(int(i));rgb=(27,108,168) if d['family'][i]==0 else (208,105,22)
            if r:
                middle=len(g[1][pos])//2
                annotations.append(dict(uv=[float(g[1][pos][middle]),float(g[2][pos][middle])],
                                        text=str(r['image_id']),family=int(d['family'][i])))
            colors.append(rgb);priority.append(bool(r))
        with torch.no_grad():
            state=torch.as_tensor(saved[n]['state']);widths=torch.as_tensor(saved[n]['widths'])
            _,lp=ext.local_hypotheses(state);col=lp.argmax(1)
            xy=ext.shared_positions(state,widths)[torch.arange(len(ext.owner)),col[ext.owner]].numpy()
        links=[[xy[int(ext.v_offset_a[e])].tolist(),xy[int(ext.v_offset_b[e])].tolist()] for e in ext.visualization_edges]
        with torch.no_grad():
            candidates=current_eligible_edges(d,ext,state,widths,col,np.ones(len(state)))
        active=set(ext.visualization_edges.tolist());p=confidence[n]
        from .fiber_density_sign_report import ordering
        mapped=state.clone();mapped[:,0]=torch.as_tensor(labels[n])
        with torch.no_grad():
            margin=ordering(mapped,widths,d,configs[n],saved=saved[n]).numpy()
        bad=(margin<0)&(d['sign']!=0)&(p[d['i']]>=.5)&(p[d['j']]>=.5)
        incident=set(np.r_[d['i'][bad],d['j'][bad]].tolist())
        underlays=[pos for pos,i in enumerate(g[0]) if int(i) in incident]
        atomic_write(out/name/'sign_underlays.json',dict(
            definition='Direct endpoints of internal parallel or perpendicular sign violations; both endpoints >=50% joint confidence; same MAP continuous-order audit as the report.',
            violated_constraints=int(bad.sum()),highlighted_fibers=len(underlays),
            pieces=sorted(incident),source_ids=[str(d['source_id'][i]) for i in sorted(incident)]),durable=False)
        context=[[xy[int(ext.v_offset_a[e])].tolist(),xy[int(ext.v_offset_b[e])].tolist()] for e in candidates
                 if e not in active and p[d['i'][e]]>=.5 and p[d['j'][e]]>=.5]
        render_layout(out/name/'predicted.svg',name+' — winding changes',d,saved[n]['state'],saved[n]['widths'],
            dict(configs[n],_geometry=g,_bounds=bounds,_colors=colors,_dotted=priority,_red_underlay_indices=underlays,_draw_order=np.argsort(priority,kind='stable'),
                 _annotations=annotations,_connections=links,_context_connections=context,render_height=2000,raster_map_only=True,vector_pdf=True,
                 diagnostic_label_layout=True,diagnostic_label_font_size=6,fiber_line_width=2,
                 _cohort_label='All individually confident fibers; dotted changes require >=50% in BOTH solves',
                 _legend='H blue / V orange; dotted: winding change. Red underlay: direct endpoint of a sign violation.'))
    atomic_write(out/'summary.json',dict(summary,completed=True),durable=False)
    print(json.dumps(summary),flush=True)
    return summary
