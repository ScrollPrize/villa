"""Read-only enclosure-component checks against fixed flow and surface references.

No component is re-solved. Contradictory components retain their explicitly
flagged spanning-tree audit layout. Offsets are L1-optimal translations only.
"""
import json
import time
from pathlib import Path
import numpy as np
from .fiber_min_flow_experiment import arrays, write
from .fiber_min_flow_benchmark import Paths
from .fiber_min_flow_full import PackedPaths
from .fiber_min_flow import fractional_targets, integer_targets
from .fiber_flow_reference_eval import sample_arcs, continuous_winding
from .fiber_reference_comparison import Surface, winding_at, weighted_median
from .fiber_curve_bank import CurveBank
from .fiber_normal_convention import inward_solution
from .fiber_uv_layout import signed_order
from .fiber_winding_order import order_margin
from .fiber_graph_store import FiberGraph


def cross_rows(graph, target_ids, reference_ids, included):
    """Orient UUID-preserving cross events as target → reference, either source order."""
    targets = {str(fid): i for i, fid in enumerate(target_ids)}
    references = {str(fid): i for i, fid in enumerate(reference_ids)}
    if targets.keys() & references.keys():
        raise ValueError('Cross graph target/reference fiber identities overlap')
    rows = []
    for k, (a, b) in enumerate(zip(graph.fiber_a, graph.fiber_b)):
        if a in targets and b in references:
            ia, ib, aa, ab = targets[a], references[b], graph.arc_a[k], graph.arc_b[k]
        elif b in targets and a in references:
            ia, ib, aa, ab = targets[b], references[a], graph.arc_b[k], graph.arc_a[k]
        else:
            raise ValueError('Not a bipartite cross graph')
        if included[ia]:
            rows.append((ia, ib, aa, ab, float(graph.parallel[k]), float(not graph.parallel[k]),
                         graph.distance[k], graph.distance[k]))
    return np.asarray(rows, dtype=float).reshape(-1, 8)


def stats(error, weight=None):
    error=np.asarray(error);ok=np.isfinite(error)
    e=abs(error[ok]);w=np.ones(len(e)) if weight is None else np.asarray(weight)[ok]
    return dict(count=len(e),l1=float(np.average(e,weights=w)) if len(e) else None,
        within_half_percent=float(100*np.average(e<=.5,weights=w)) if len(e) else None,
        max=float(e.max()) if len(e) else None)


def cross_metrics(local, target, rounded, sign, fd, par):
    result={}
    for name,mask in [('all',np.ones(len(local),bool)),('perpendicular',~par),('parallel',par)]:
        row=stats((local-target)[mask]);known=mask&(sign!=0)
        wrong=order_margin(local,sign,fd,par,.5)<-1e-8
        row.update(correct_step_percent=float(100*np.mean(abs(local[mask]-rounded[mask])<.5+1e-8)) if mask.any() else None,
                   sign_events=int(known.sum()),sign_violations=int((wrong&known).sum()),
                   sign_percent=float(100*np.mean(wrong[known])) if known.any() else None)
        result[name]=row
    return result


def integer_offset(values):
    """L1-optimal integer gauge; never introduce fractional phase/sign changes."""
    values=np.asarray(values)
    if not len(values):return None
    median=float(np.median(values));candidates=np.unique([np.floor(median),np.ceil(median)])
    return int(min(candidates,key=lambda x:(np.mean(abs(values-x)),abs(x),x)))


def run(config):
    start=time.monotonic();root=Path(config['output_root']);root.mkdir(parents=True,exist_ok=True)
    if (root/'request.json').exists() and json.loads((root/'request.json').read_text())!=config:
        raise ValueError('Reference evaluation configuration changed')
    write(root/'request.json',config)
    source=Path(config['enclosures']);cache=Path(config['cross_root'])
    d=arrays(source/'ordinary_tensors.npz');comp=arrays(source/'enclosure_components.npz')
    doc=json.loads((source/'enclosures.json').read_text())
    if not np.array_equal(comp['source_id'],d['source_id']):raise ValueError('Component identity mismatch')
    groups=[c for c in doc['components'] if c['fibers']>1]
    ids=d['source_id'];labels=comp['relative_winding'];cid=comp['component']
    included=np.isin(cid,[c['component'] for c in groups])
    paths=Paths(dict(fibers=str(source/'fibers.vcobj'),umbilicus=config['umbilicus']),d)
    irls=arrays(config['irls_solution']);refpaths=PackedPaths(config['irls_geometry'])
    selected=np.load(cache/'reference_ids.npy');n=len(ids)
    sign_records=json.loads((source/'sign_alignment.json').read_text())['fibers']
    normals=dict(ids=ids,normal_sign=np.array([r['normal_sign'] for r in sign_records]),broken=np.zeros(n,bool))
    if not np.array_equal([r['source_id'] for r in sign_records],ids):raise ValueError('Normal identity mismatch')
    newbank=CurveBank.from_normals(ids,cache/'target_normals',normals)
    refbank=CurveBank.from_normals(selected,cache/'reference_normals',inward_solution(arrays(config['reference_normal_solution'])))
    rows=cross_rows(FiberGraph(cache/'cross.vcobj'),ids,selected,included)
    ia=rows[:,0].astype(int);ib=rows[:,1].astype(int)
    at=np.searchsorted(irls['source_id'],selected[ib]);covered=at<len(irls['source_id'])
    covered&=irls['source_id'][np.minimum(at,len(irls['source_id'])-1)]==selected[ib]
    rows=rows[covered];ia=ia[covered];ib=ib[covered];at=at[covered]
    par=rows[:,4]>rows[:,5];fd=irls['family'][at].astype(int)-d['family'][ia].astype(int)
    family_ok=par==(fd==0)
    rows=rows[family_ok];ia=ia[family_ok];ib=ib[family_ok];at=at[family_ok];par=par[family_ok];fd=fd[family_ok]
    a,ta=paths.at_many(ids[ia],rows[:,2]);b,tb=refpaths.at_many(selected[ib],rows[:,3])
    if not np.array_equal(paths.seam.axis,refpaths.seam.axis) or paths.seam.angle!=refpaths.seam.angle:
        raise ValueError('Reference seam differs')
    connector=paths.seam.segments(a,b)
    na,va=newbank.sample_normals(ia,rows[:,2]);nb,vb=refbank.sample_normals(ib,rows[:,3])
    sign=signed_order(b-a,na,nb)
    target,valid=fractional_targets(np.where(par,rows[:,7],rows[:,6]),sign,fd,par,config.get('perpendicular_offset',.5))
    rounded,_=integer_targets(np.where(par,rows[:,7],rows[:,6]),sign,fd,par,config.get('perpendicular_offset',.5))
    valid&=va&vb
    local=irls['winding'][at]-labels[ia]+tb-ta-connector
    # Each connected reference graph has its own gauge. Require one fixed gauge.
    if len(np.unique(irls['component'][at[valid]]))!=1:raise ValueError('Cross events span independent IRLS gauges')
    ref_anchor=irls['winding'][at]+tb-ta-connector-target
    np.savez(root/'irls_events.npz',source_id=ids[ia],reference_source=selected[ib],component=cid[ia],
        local=local,target=target,rounded=rounded,sign=sign,parallel=par,valid=valid,reference_anchor=ref_anchor,family_delta=fd)
    print(json.dumps(dict(stage='irls_cross_ready',events=int(valid.sum()))),flush=True)
    # Reuse the established approximate surface projector, unchanged resolution.
    settings=config['spiral'];surface=Surface(settings['reference'],settings.get('xyz_scale',4),
        settings.get('surface_stride',2),settings.get('workers',32),settings.get('candidates',4))
    maximum=settings.get('max_distance',200)
    def project(xyz):
        rc,dist=surface.project(xyz);q=winding_at(rc[:,1],surface.x,surface.q)
        return np.where(dist<=maximum,q,np.nan),dist
    take=np.flatnonzero(included);length=np.array([paths.paths[str(ids[i])][0][-1] for i in take])
    owner,arc,weight,_=sample_arcs(length,settings.get('sample_spacing',200));owner=take[owner]
    xyz,turns=paths.at_many(ids[owner],arc)
    q=continuous_winding(paths.seam,xyz,turns,labels[owner]);truth,dist=project(xyz)
    np.savez(root/'spiral_samples.npz',source_id=ids[owner],component=cid[owner],arc=arc,
        weight=weight,fitted=q,reference=truth,distance=dist)
    # Internal ordinary constraints: reference step consistency and geometric signs.
    ea,eta=paths.ends('a');eb,etb=paths.ends('b');ec=paths.seam.segments(ea,eb)
    ra,da=project(ea);rb,db=project(eb)
    pa=continuous_winding(paths.seam,ea,eta,labels[d['i']]);pb=continuous_winding(paths.seam,eb,etb,labels[d['j']])
    local_internal=labels[d['j']]-labels[d['i']]+etb-eta-ec
    fdi=d['family'][d['j']].astype(int)-d['family'][d['i']].astype(int)
    wrong=order_margin(local_internal,d['sign'],fdi,d['parallel'],.5)<-1e-8
    inferred=np.full(n,np.nan)
    for i in take:
        use=valid&(ia==i)
        if use.any():inferred[i]=np.median(ref_anchor[use])
    results=[]
    for group in groups:
        c=group['component'];use=valid&(cid[ia]==c)
        offset=integer_offset((local-target)[use])
        real_offset=float(np.median((local-target)[use])) if use.any() else None
        scored=cross_metrics(local[use]-offset,target[use],rounded[use],sign[use],fd[use],par[use]) if offset is not None else None
        samples=cid[owner]==c;matched=samples&np.isfinite(truth)
        shift=weighted_median(truth[matched]-q[matched],weight[matched]) if matched.any() else None
        spiral=stats(q[matched]+shift-truth[matched],weight[matched]) if shift is not None else stats([])
        spiral.update(offset=shift,coverage_percent=float(100*weight[matched].sum()/weight[samples].sum()))
        internal=(cid[d['i']]==c)&(cid[d['j']]==c)
        internal_results={}
        for name,mask in [('perpendicular',internal&~d['parallel']),('parallel',internal&d['parallel'])]:
            known=mask&(d['sign']!=0)
            e=(labels[d['j']]-labels[d['i']])-(inferred[d['j']]-inferred[d['i']])
            internal_results[name]=dict(events=int(mask.sum()),
                spiral=stats(((pb-pa)-(rb-ra))[mask]),irls_induced=stats(e[mask]),
                layout_sign_events=int(known.sum()),layout_sign_violations=int((wrong&known).sum()))
            spiral_known=known&np.isfinite(ra)&np.isfinite(rb)
            spiral_wrong=order_margin(rb-ra,d['sign'],fdi,d['parallel'],.5)<-1e-8
            internal_results[name].update(spiral_sign_events=int(spiral_known.sum()),
                spiral_sign_violations=int((spiral_wrong&spiral_known).sum()),
                spiral_sign_percent=float(100*np.mean(spiral_wrong[spiral_known])) if spiral_known.any() else None)
        enclosures=[r for r in doc['relations'] if cid[r['h']]==c]
        enclosure_error=np.array([inferred[r['v']]-inferred[r['h']]-r['anchor_step'] for r in enclosures])
        result=dict(component=c,fibers=group['fibers'],conflicts=group['conflicting_relations'],
            irls_offset=offset,irls_cross=scored,spiral=spiral,internal=internal_results,
            irls_real_offset=real_offset,
            irls_real_offset_cross=cross_metrics(local[use]-real_offset,target[use],rounded[use],sign[use],fd[use],par[use]) if real_offset is not None else None,
            irls_enclosure_steps=stats(enclosure_error))
        results.append(result)
    report=dict(groups=results,seconds=time.monotonic()-start,
        protocol='Fixed enclosure audit layout; one L1-optimal integer offset per group for the integer IRLS realization, real offset for continuous spiral coordinates. No scale fit or per-fiber shifts. IRLS cross L1 uses fractional measured targets; correct step uses rounded measured target. Spiral L1 is arc-weighted pointwise winding error.',
        irls_solution_kind='saved integer winding realization; no new solve',
        caveat='Groups with conflicts use diagnostic spanning-tree potentials, not a consistent enclosure solution.')
    write(root/'results.json',report)
    lines=['# Enclosure groups versus fixed references','',report['protocol'],'',report['caveat'],'',
        '|Group|Fibers|Conflicts|Spiral L1|Spiral ≤0.5 %|Coverage %|IRLS perp n|L1|Correct %|Sign n/%|IRLS par n|L1|Correct %|Sign n/%|',
        '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    def fmt(v):return '—' if v is None else f'{v:.3f}'
    for r in sorted(results,key=lambda r:-r['fibers']):
        s=r['spiral'];p=r['irls_cross']['perpendicular'];a=r['irls_cross']['parallel']
        lines.append(f"|{r['component']}|{r['fibers']}|{r['conflicts']}|{fmt(s['l1'])}|{fmt(s['within_half_percent'])}|{fmt(s['coverage_percent'])}|{p['count']}|{fmt(p['l1'])}|{fmt(p['correct_step_percent'])}|{p['sign_violations']}/{fmt(p['sign_percent'])}|{a['count']}|{fmt(a['l1'])}|{fmt(a['correct_step_percent'])}|{a['sign_violations']}/{fmt(a['sign_percent'])}|")
    lines+=['','## Internal ordinary constraints','',
        'Pairwise differences are offset invariant. IRLS-induced values are per-fiber median anchor estimates from fixed IRLS cross constraints, not a new solve. Spiral values are projected at the ordinary constraint endpoints. Correct means difference within half a winding. Sign columns check each reference against the measured geometric order; the layout sign count checks the enclosure layout itself.',
        '', '|Group|Type|Events|Spiral L1 / correct %|IRLS-induced L1 / correct %|Spiral signs / tested|Layout signs / tested|',
        '|---:|---|---:|---:|---:|---:|---:|']
    for r in sorted(results,key=lambda r:-r['fibers']):
        for name,v in r['internal'].items():
            s=v['spiral'];i=v['irls_induced']
            lines.append(f"|{r['component']}|{name}|{v['events']}|{fmt(s['l1'])} / {fmt(s['within_half_percent'])}|{fmt(i['l1'])} / {fmt(i['within_half_percent'])}|{v['spiral_sign_violations']} / {v['spiral_sign_events']}|{v['layout_sign_violations']} / {v['layout_sign_events']}|")
    (root/'results.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(stage='complete',groups=len(results),seconds=report['seconds'])),flush=True)
    return report
