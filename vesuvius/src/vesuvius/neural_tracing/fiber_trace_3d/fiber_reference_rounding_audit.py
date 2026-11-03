"""Read-only reference/rounding ablation of saved solves; no optimization."""
import csv
import json
from pathlib import Path
import numpy as np
import torch
from lasagna.inference_provenance import atomic_write, sha256_file
from .fiber_gt_constraint_score import arrays, enabled_l1, event_residuals
from .fiber_uv_arc_phase import event_arc_points
from .fiber_winding_order import order_margin
from . import fiber_uv_layout as uv
from .fiber_selected_dataset_eval import remap_events


def read(path):
    return json.loads(Path(path).read_text())


def summarize(required, delta, events, enabled, integer=False):
    """Single integer gauge, selected using all enabled events, not per class."""
    par=np.array([e['parallel'] for e in events],bool)
    score=enabled_l1(required,enabled,par)
    shift=score['gt_global_offset']
    error=np.asarray(required)-shift
    rounded=np.floor(error+.5)
    sign=np.sign([e['normal_cosine'] for e in events])
    fd=np.array([e['family_delta'] for e in events])
    # Integer topology: outward H->V >=0, V->H >=1; parallels >=0.
    margin=order_margin(np.asarray(delta)-shift,sign,fd,par,0. if integer else .5)
    rows={}
    for name,kind in [('perpendicular',~par),('parallel',par),('all',np.ones(len(par),bool))]:
        use=enabled&kind;n=int(use.sum());signed=use&(sign!=0)
        rows[name]=dict(events=n,l1=float(abs(error[use]).mean()) if n else None,
            rounded_residual_l1=float(abs(rounded[use]).mean()) if n else None,
            correct_percent=float(100*(rounded[use]==0).mean()) if n else None,
            max_abs=float(abs(error[use]).max()) if n else None,
            sign_events=int(signed.sum()),sign_errors=int((signed&(margin<0)).sum()),
            sign_percent=float(100*(margin[signed]<0).mean()) if signed.any() else None)
    return dict(global_offset=shift,rows=rows),error


def run(settings):
    torch.set_num_threads(2)
    gt=Path(settings['gt_root']);gc=read(gt/'uv_whole/config.json')
    gd=arrays(gc['tensor_cache']);gs=arrays(gt/'uv_whole/solution.npz')
    _,_,keep,gpick=uv.hard_assignment(gs['state'],len(gs['widths']),gc,best_nonbroken=True)
    if not keep.all():raise ValueError('Reference contains disabled fibers')
    reference=read(settings['reference']);offsets=dict(reference['fiber_offsets'])
    rows=[];hashes={str(p):sha256_file(p) for p in [Path(settings['reference']),gt/'uv_whole/solution.npz',Path(__file__)]}
    out=Path(settings['output_root']);out.mkdir(parents=True,exist_ok=True)
    for spec in settings['solves']:
        root=Path(spec['root']);cfg=read(root/'solve/config.json')
        sol=arrays(root/'solve/solution.npz');d=arrays(cfg['tensor_cache'])
        event_path=Path(spec.get('events',root/'gt_events.json'))
        events=remap_events(read(event_path),d)
        for field in ('source_id','start','end','family'):
            if not np.array_equal(sol[field],d[field]):raise ValueError('Solution identity mismatch')
        for e in events:
            if d['source_id'][e['test_piece']]!=e['test_source']:raise ValueError('Event identity mismatch')
        state=sol['state'].copy();state[:,0]=np.floor(state[:,0]+.5)
        enabled=state[np.array([e['test_piece'] for e in events]),3]>=0
        oldq=event_arc_points(gd,gs['state'],gpick,gs['widths'],
            np.array([e['gt_piece'] for e in events]),np.array([e['gt_arc'] for e in events]))
        req=event_residuals(events,d,state,None,len(sol['widths']),gd,gs,gpick,
            dict(cfg,soft_winding_model=True),gc,sol['widths'],saved_geometry=sol)
        target=np.array([e['signed_distance']-.5*e['family_delta'] for e in events])
        q=oldq-target-req
        original,_=summarize(req,oldq-q,events,enabled)
        # Verify against serialized score before changing reference/cohort.
        saved=read(root/('comparison.json' if (root/'comparison.json').exists() else 'results.json'))['score']
        if abs(original['rows']['perpendicular']['l1']-saved['gt_perpendicular']['l1'])>1e-7:
            raise ValueError('Original GT L1 not reproduced')
        flowroot=Path(spec['flow_root'])
        with (flowroot/'gt_events.csv').open() as f:flow={int(e['event_id']):e for e in csv.DictReader(f)}
        def event_key(e):
            return (str(e['gt_source']),str(e['test_source']),round(float(e['gt_arc']),4),round(float(e['test_arc']),4))
        flow_geometry={event_key(e):e for e in flow.values()}
        if len(flow_geometry)!=len(flow):raise ValueError('Nonunique flow geometry key')
        indices=[i for i,e in enumerate(events) if e['gt_source'] in offsets and event_key(e) in flow_geometry]
        ev=[events[i] for i in indices];en=enabled[indices];q=q[indices];oldq=oldq[indices];target=target[indices]
        # Flow uses explicit polyline/connector seam crossings, not rounded angles.
        records=[flow_geometry[event_key(e)] for e in ev]
        for e,f in zip(ev,records):
            for k in ('gt_source','test_source'):
                if e[k]!=f[k]:raise ValueError('Flow event identity mismatch')
            for k in ('gt_arc','test_arc','signed_distance'):
                if abs(e[k]-float(f[k]))>1e-6:raise ValueError('Flow event geometry mismatch')
        connector=np.array([int(f['connector_crossings']) for f in records])
        measured=np.array([int(f['measured_integer_target']) for f in records])
        newq=np.array([offsets[e['gt_source']]+e['gt_angle'] for e in ev])
        newinteger=np.array([offsets[e['gt_source']]+int(np.floor(gd['node_phase'][e['gt_piece'],0]+.25))
            +int(f['gt_fiber_crossings']) for e,f in zip(ev,records)])
        results={};details=[]
        for name,gq,gi in [('old_fitted',oldq,np.floor(oldq+.25)),('enclosure',newq,newinteger)]:
            continuous=gq-q-target
            results[name+'_continuous'],err=summarize(continuous,gq-q,ev,en)
            # This is actual endpoint quantization in flow's fixed -quarter-turn
            # seam chart, with its integer measurement targets. Distinct from
            # merely rounding an already-computed error (also reported above).
            delta=gi-np.floor(q+.25)-connector
            results[name+'_integer'],ierr=summarize(delta-measured,delta,ev,en,integer=True)
            details.append((name,err,ierr))
        fm=read(flowroot/'metrics.json')['reference'];shift=fm['single_global_gauge']['offset']
        fr=np.array([int(f['integer_residual'])+int(f['gauge'])-shift for f in records])
        par=np.array([e['parallel'] for e in ev],bool)
        results['flow_same_enabled_cohort']={name:dict(events=int((en&mask).sum()),
            l1=float(abs(fr[en&mask]).mean()),
            sign_errors=sum(f['global_gauge_sign_violation']=='True' for f,u in zip(records,en&mask) if u))
            for name,mask in [('perpendicular',~par),('parallel',par)]}
        row=dict(name=spec['name'],root=str(root),original=original,comparison=results,
            matched_events=len(ev),enabled_matched_events=int(en.sum()),flow_total_events=len(flow),
            excluded_reference_events=sum(e['gt_source'] not in offsets for e in events),
            definition='Enabled mask unchanged; one global integer gauge. Continuous plus rounded residual diagnostic; integer endpoint chart uses flow seam/targets. No re-solving.')
        rows.append(row)
        for p in (root/'solve/solution.npz',root/'solve/config.json',event_path,flowroot/'gt_events.csv',flowroot/'metrics.json'):
            hashes[str(p)]=sha256_file(p)
        atomic_write(out/(spec['name']+'_events.json'),[dict(event_id=e['event_id'],enabled=bool(en[i]),
            predicted_q=float(q[i]),old_gt_q=float(oldq[i]),enclosure_gt_q=float(newq[i]),
            **{name+'_continuous_error':float(er[i]) for name,er,_ in details},
            **{name+'_integer_error':float(er[i]) for name,_,er in details}) for i,e in enumerate(ev)],durable=False)
        print(spec['name'],json.dumps({k:v['rows']['perpendicular'] for k,v in results.items() if 'rows' in v}),flush=True)
    atomic_write(out/'results.json',dict(results=rows,hashes=hashes),durable=False)
    report=['# Saved-solution reference and rounding audit','',
        'No solves or enabled masks changed. Original full-event scores are reproduced before changing protocols. Comparisons use reference IDs provided in the offset map and events present in both datasets, matched by source IDs and endpoint arc lengths. Compare counts only on matching cohorts.', '',
        'Fitted reference: '+str(gt/'uv_whole/solution.npz')+'. Alternative reference: '+str(settings['reference'])+'.', '',
        'Reference fiber offsets: '+json.dumps(offsets,sort_keys=True)+'.', '',
        'Continuous uses the existing continuous measured-distance target. Integer uses endpoint quantization floor(q+0.25) in the min-flow seam chart, its measured integer target and connector seam crossings. This is not merely rounding residuals and is not a re-solved integer assignment. Each protocol uses one global integer alignment, never per-component alignment. Correct means residual rounds to zero; sign checks use continuous half-turn leeway or integer topology respectively.', '',
        '|Solution|Reference/protocol|Class|Events|L1|Correct %|Sign errors|Sign %|Max|',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        for protocol,result in row['comparison'].items():
            if 'rows' not in result:continue
            for kind in ('perpendicular','parallel'):
                r=result['rows'][kind]
                report.append(f"|{row['name']}|{protocol}|{kind}|{r['events']}|{r['l1']:.3f}|{r['correct_percent']:.2f}|{r['sign_errors']}|{r['sign_percent']:.2f}|{r['max_abs']:.3f}|")
    report+=['','## Min-flow on each solution’s enabled event cohort','',
        'Supplied min-flow assignments, using their single global gauge. Cohort matching does not imply identical optimization protocols; consult the supplied run configurations.','',
        '|SGD cohort|Class|Events|Min-flow L1|Min-flow sign errors|','|---|---|---:|---:|---:|']
    for row in rows:
        for kind,r in row['comparison']['flow_same_enabled_cohort'].items():
            report.append(f"|{row['name']}|{kind}|{r['events']}|{r['l1']:.3f}|{r['sign_errors']}|")
    (out/'report.md').write_text('\n'.join(report)+'\n')
    return rows
