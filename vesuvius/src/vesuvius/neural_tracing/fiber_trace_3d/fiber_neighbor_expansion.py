"""Iterative GT-assisted neighbor inclusion; uses the existing full solver/losses."""
import json
from pathlib import Path
import time
import numpy as np
import torch
from lasagna.inference_provenance import atomic_write, sha256_file
from . import fiber_soft_winding as soft, fiber_uv_layout as uv
from .fiber_gt_constraint_score import arrays
from .fiber_anchored_dataset import induced_graph, one_hop, perpendicular_supported_graph, anchored_mask
from .fiber_selected_dataset_eval import remap_events
from .fiber_joint_gt_eval import combine
from .fiber_full_slab import clean_config
from .fiber_implicit_winding import hypotheses, saved_sigma
from .fiber_shared_curve import saved_points


def z_spans(xyz, z_range):
    """Exact arc intervals in a Z slab on a piecewise-linear physical fiber."""
    arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(xyz,axis=0),axis=1))]
    spans=[]
    for k,(a,b) in enumerate(zip(xyz[:-1,2],xyz[1:,2])):
        lo,hi=0.,1.
        if a==b:
            if not z_range[0]<=a<=z_range[1]:continue
        else:
            x,y=sorted(((z_range[0]-a)/(b-a),(z_range[1]-a)/(b-a)))
            lo=max(lo,x);hi=min(hi,y)
        if hi<=lo:continue
        left=arc[k]+lo*(arc[k+1]-arc[k]);right=arc[k]+hi*(arc[k+1]-arc[k])
        if spans and abs(spans[-1][1]-left)<1e-6:spans[-1][1]=right
        else:spans.append([left,right])
    return spans


def interval_overlap(a,b):
    return max(a[0],b[0])<=min(a[1],b[1])


def map_geometry(d,sol,config):
    state=sol['state'].copy()
    with torch.no_grad():
        st=torch.as_tensor(state)
        sigma,support=saved_sigma(st,sol,config)
        h,lp=hypotheses(st,sigma,support)
        pick=lp.argmax(1);labels=h[torch.arange(len(state)),pick,0].numpy()
        confidence=(lp.exp().amax(1)*torch.as_tensor(state[:,3]).sigmoid()).numpy()
    state[:,0]=labels
    def query(owner,arc):
        owner=np.asarray(owner,dtype=int);arc=np.asarray(arc)
        ds=(arc-d['start'][owner])*np.sign(d['node_ds'][owner,-1])
        q,_,v=saved_points(d,sol,state,sol['widths'],owner,ds,shared_only=True)
        return q,v
    return confidence,query


def select(d,sol,config,gt_ids,curves,z_range,confidence):
    """Whole fibers with MAP span in GT H range and an active path to GT."""
    probability,query=map_geometry(d,sol,config)
    gt=np.isin(d['source_id'],gt_ids)
    span_owners=[];arc_pairs=[]
    for i,sid in enumerate(d['source_id']):
        for lo,hi in z_spans(curves[int(sid)][1],z_range):
            lo=max(lo,d['start'][i]);hi=min(hi,d['end'][i])
            if hi>lo:span_owners.append(i);arc_pairs.append((lo,hi))
    owners=np.asarray(span_owners,int)
    if not len(owners):raise ValueError('No fiber intersects the Z slab')
    q,_=query(np.repeat(owners,2),np.asarray(arc_pairs).ravel());ranges=np.sort(q.reshape(-1,2),axis=1)
    h_ranges=ranges[gt[owners]&(d['family'][owners]==0)]
    if not len(h_ranges):raise ValueError('No GT H span in the Z slab')
    overlaps=np.zeros(len(ranges),bool)
    for lo,hi in h_ranges:overlaps|=(ranges[:,1]>=lo)&(ranges[:,0]<=hi)
    inside=np.zeros(len(gt),bool);inside[owners[overlaps]]=True
    active=((probability>=confidence)&inside)|gt
    active=anchored_mask(d,np.flatnonzero(gt),active)
    keep=active|gt
    records=[dict(source_id=int(s),probability=float(probability[i]),in_h_range=bool(inside[i]),
                  selected=bool(keep[i]),gt=bool(gt[i])) for i,s in enumerate(d['source_id'])]
    return set(map(int,d['source_id'][keep&~gt])),dict(gt_h_winding_ranges=h_ranges.tolist(),
        input_fibers=len(gt),selected_ordinary=int((keep&~gt).sum()),records=records)


def warm_state(d,old_d,old_sol,config):
    """Carry exact old anchors; initialize new neighbors using geometric votes."""
    lookup={int(s):i for i,s in enumerate(old_d['source_id'])}
    n=len(d['family']);state=np.column_stack((np.zeros(n),d['phase'],d['z'],np.full(n,4.))).astype(float)
    known=np.array([int(s) in lookup for s in d['source_id']])
    for i,s in enumerate(d['source_id']):
        if known[i]:
            j=lookup[int(s)];state[i]=old_sol['state'][j]
            state[i,1]=old_sol['straight_anchor_u'][j];state[i,2]=old_sol['straight_anchor_v'][j]
    _,query=map_geometry(old_d,old_sol,config)
    widths=torch.as_tensor(old_sol['widths'],dtype=torch.float64)
    boundaries=torch.cat((widths.new_zeros(1),widths.cumsum(0)))
    votes={}
    for reverse in (False,True):
        a,b=(d['j'],d['i']) if reverse else (d['i'],d['j'])
        da,db=(d['ds_b'],d['ds_a']) if reverse else (d['ds_a'],d['ds_b'])
        edges=np.flatnonzero(~known[a]&known[b]&(d['sign']!=0))
        if not len(edges):continue
        aa,bb=a[edges],b[edges]
        old_ids=np.array([lookup[int(s)] for s in d['source_id'][bb]])
        qb,vb=query(old_ids,d['start'][bb]+db[edges]*np.sign(d['node_ds'][bb,-1]))
        # Measurement only seeds the optimizer, never becomes a frozen assignment.
        delta=d['sign'][edges]*d['distance'][edges]*config.get('lasagna_winding_scale',1.)
        calibration=(~d['parallel'][edges])*(d['family'][d['j'][edges]]-d['family'][d['i'][edges]])*config.get('perpendicular_offset',.5)
        delta=delta-calibration
        qa=qb+(delta if reverse else -delta)
        with torch.no_grad():
            x=uv.q_to_u(torch.as_tensor(qa),widths,boundaries)-torch.as_tensor((d['family'][aa]==0)*da[edges])
            anchor=uv.u_to_q(x,widths,boundaries).numpy()
        va=vb-(d['family'][aa]==1)*da[edges]
        for i,q,v in zip(aa,anchor,va):votes.setdefault(int(i),[]).append((q-state[i,1],v))
    for i in np.flatnonzero(~known):
        if int(i) not in votes:raise ValueError('New fiber has no signed neighbor support: '+str(d['source_id'][i]))
        q,v=np.median(votes[int(i)],axis=0);state[i,0]=q;state[i,2]=v
    return state


def run(settings):
    if settings.get('finish_existing_after_round'):
        return finish_existing(settings)
    from .fiber_workflows import run as dispatch
    from .fiber_gt_constraint_score import run as extract_cross
    from vesuvius.neural_tracing.tracers.shared.annotations.afv import read_afv_polylines,write_afv
    torch.set_num_threads(4)
    started=time.monotonic();root=Path(settings['output_root']);root.mkdir(parents=True,exist_ok=True)
    seed=Path(settings['seed_solve']);cfg=clean_config(json.loads((seed/'config.json').read_text()))
    paths=[seed/'solution.npz',seed/'config.json',Path(settings['parent_tensors']),Path(settings['afv']),Path(__file__)]
    signature=dict(settings=settings,inputs={str(p):sha256_file(p) for p in paths})
    stamp=root/'inputs.json'
    if stamp.exists():
        prior=json.loads(stamp.read_text())
        old_inputs={k:v for k,v in prior['inputs'].items() if k!=str(Path(__file__))}
        new_inputs={k:v for k,v in signature['inputs'].items() if k!=str(Path(__file__))}
        if prior['settings']!=signature['settings'] or old_inputs!=new_inputs:
            raise ValueError('Expansion inputs changed; use a new directory')
        if prior!=signature:
            revisions=root/'resume_code_revisions.json'
            records=json.loads(revisions.read_text()) if revisions.exists() else []
            records.append(dict(previous=prior['inputs'][str(Path(__file__))],current=signature['inputs'][str(Path(__file__))]))
            atomic_write(revisions,records,durable=False)
    atomic_write(stamp,signature,durable=False)
    if (root/'summary.json').exists():return json.loads((root/'summary.json').read_text())
    parent=arrays(settings['parent_tensors'])
    parent,_=soft.constraint_family_filter(parent,True)
    if len(np.unique(parent['source_id']))!=len(parent['source_id']):raise ValueError('Whole-fiber parent required')
    with uv._ro(settings['afv']) as db:
        eligible=np.array([r[0] for r in db.execute('SELECT id FROM fibers WHERE max_z >= ? AND min_z <= ?',settings['z_range'])])
    # Cheap Z bounding selection only; exact physical slab spans are checked later.
    parent=induced_graph(parent,eligible,cfg)
    previous_d=arrays(cfg['tensor_cache']);previous=arrays(seed/'solution.npz')
    old_gt=np.asarray(cfg['force_keep_source_ids'])
    gtroot=Path(settings['gt_root']);gcfg=json.loads((gtroot/'uv_whole/config.json').read_text());gd=arrays(gcfg['tensor_cache'])
    if len(old_gt)!=len(gd['source_id']):raise ValueError('GT mapping mismatch')
    with uv._ro(settings['afv']) as db:max_id=db.execute('SELECT MAX(id) FROM fibers').fetchone()[0]
    gt_ids=np.arange(max_id+1,max_id+1+len(old_gt),dtype=np.int64)
    for a,b in zip(old_gt,gt_ids):previous_d['source_id'][previous_d['source_id']==a]=b
    previous['source_id']=previous_d['source_id'].copy()
    atomic_write(root/'gt_ids.json',dict(original=gd['source_id'].tolist(),old_solve=old_gt.tolist(),new=gt_ids.tolist()),durable=False)
    gt_curves=list(read_afv_polylines(gcfg['afv'],gd['source_id']))
    mapping=dict(zip(map(int,gd['source_id']),map(int,gt_ids)))
    curves={mapping[int(r['id'])]:(dict(r,id=mapping[int(r['id'])]),x) for r,x in gt_curves}
    def load_curves(ids):
        missing=[int(s) for s in ids if int(s) not in curves]
        for row,xyz in read_afv_polylines(settings['afv'],missing) if missing else []:curves[int(row['id'])]=(row,xyz)
    load_curves(previous_d['source_id'])
    selected,selection=select(previous_d,previous,cfg,gt_ids,curves,settings['z_range'],settings.get('confidence',.5))
    history=[];seen=[];stable=0

    def solve_graph(folder,sources,old_d,old_sol,frozen=None,cross_source=None):
        folder.mkdir(parents=True,exist_ok=True)
        if (folder/'solve/summary.json').exists():
            sc=json.loads((folder/'solve/config.json').read_text())
            cached_d=arrays(folder/'tensors.npz')
            load_curves(cached_d['source_id'])
            return cached_d,arrays(folder/'solve/solution.npz'),sc
        d=induced_graph(parent,sorted(sources),cfg)
        load_curves(sources)
        candidate_cfg=dict(cfg,afv=settings['afv'],tensor_cache=str(folder/'ordinary.npz'))
        np.savez(folder/'ordinary.npz',**d)
        atomic_write(folder/'candidate_config.json',{'fiber_uv_layout':candidate_cfg},durable=False)
        cross=folder/'cross'
        if cross_source is not None and not (cross/'cross_events.json').exists():
            cross.mkdir(exist_ok=True)
            cached=json.loads(Path(cross_source).read_text())
            cached['events']=remap_events(cached['events'],d)
            atomic_write(cross/'cross_events.json',cached,durable=False)
        if not (cross/'cross_events.json').exists():
            extract_cross(dict(configs=[str(folder/'candidate_config.json')],gt_root=str(gtroot),
                output_root=str(cross),extract_template=settings['extract_template'],orient_template=settings['orient_template'],
                threads=settings.get('threads',128),events_only=True,
                native_binary=settings.get('native_binary','vc_fiber_constraints')))
        events=remap_events(json.loads((cross/'cross_events.json').read_text())['events'],d)
        merged,_,_=combine(d,gd,events,cfg,gt_source_ids=gt_ids)
        merged,_=perpendicular_supported_graph(merged,cfg)
        if not np.isin(gt_ids,merged['source_id']).all():raise ValueError('GT lost perpendicular support')
        load_curves(merged['source_id'])
        afv=folder/'fibers.afv'
        if not afv.exists():write_afv(settings['afv'],afv,[curves[int(s)] for s in merged['source_id']],dict(source='iterative_GT_neighbor_expansion'))
        tensor=folder/'tensors.npz';np.savez(tensor,**merged)
        state=warm_state(merged,old_d,old_sol,cfg);state[np.isin(merged['source_id'],gt_ids),3]=100.
        init=folder/'init';init.mkdir(exist_ok=True)
        np.savez(init/'solution.npz',state=state,widths=old_sol['widths'],**{k:merged[k] for k in ('source_id','start','end','family')})
        config=dict(cfg,afv=str(afv),tensor_cache=str(tensor),initial_solution=str(init/'solution.npz'),
            output_root=str(folder/'solve'),force_keep_source_ids=gt_ids.tolist(),devices=settings.get('devices','0'))
        config.pop('freeze_source_ids',None);config.pop('freeze_widths',None)
        if frozen is not None:
            config.update(freeze_source_ids=np.intersect1d(list(frozen),merged['source_id']).tolist(),freeze_widths=True)
        config['gpu_manager']=dict(cfg['gpu_manager'],exclusive=False,label='iterative-GT-neighbor-expansion')
        np.savez(init/'samples.npz',**soft.prepare_samples(config,merged))
        atomic_write(init/'config.json',dict(tensor_sha256=sha256_file(tensor),soft_samples=str(init/'samples.npz')),durable=False)
        atomic_write(folder/'solve_config.json',{'fiber_implicit_winding':config},durable=False)
        dispatch({'fiber_implicit_winding':config})
        sol=arrays(folder/'solve/solution.npz')
        if not np.all(sol['state'][np.isin(merged['source_id'],gt_ids),3]==100):raise ValueError('GT was disabled')
        if frozen is not None:
            if not np.allclose(sol['widths'],old_sol['widths'],rtol=1e-12,atol=1e-8):raise ValueError('Conditional widths changed')
            lookup={int(s):i for i,s in enumerate(old_d['source_id'])}
            for i,s in enumerate(merged['source_id']):
                if int(s) not in frozen:continue
                j=lookup[int(s)]
                if not np.allclose(sol['state'][i,[0,3]],old_sol['state'][j,[0,3]],rtol=0,atol=1e-10):raise ValueError('Frozen state moved')
                for k in ('straight_anchor_u','straight_anchor_v'):
                    if not np.allclose(sol[k][i],old_sol[k][j],rtol=0,atol=1e-8):raise ValueError('Frozen geometry moved')
        return merged,sol,config

    termination='round_limit'
    for round_id in range(1,settings.get('max_rounds',12)+1):
        rr=root/f'round_{round_id:03d}';rr.mkdir(exist_ok=True)
        # Expand only via signed constraints from the currently retained set.
        active=np.isin(parent['source_id'],sorted(selected));signed=parent['sign']!=0
        edge=signed&(active[parent['i']]|active[parent['j']])
        candidates=set(map(int,np.r_[parent['source_id'][parent['i'][edge]],parent['source_id'][parent['j'][edge]]]))|selected
        print(json.dumps(dict(round=round_id,selected=len(selected),candidates=len(candidates),new_candidates=len(candidates-selected))),flush=True)
        if candidates-selected:
            cd,cs,cc=solve_graph(rr/'conditional',candidates,previous_d,previous,selected|set(map(int,gt_ids)))
        else:cd,cs,cc=previous_d,previous,cfg
        filtered,audit=select(cd,cs,cc,gt_ids,curves,settings['z_range'],settings.get('confidence',.5))
        atomic_write(rr/'conditional_selection.json',audit,durable=False)
        if not filtered:raise ValueError('No ordinary fibers survive selection')
        cross_source=rr/'conditional/cross/cross_events.json'
        nd,ns,nc=solve_graph(rr/'joint',filtered,cd,cs,
            cross_source=cross_source if cross_source.exists() else None)
        final,audit=select(nd,ns,nc,gt_ids,curves,settings['z_range'],settings.get('confidence',.5))
        atomic_write(rr/'selection.json',audit,durable=False)
        row=dict(round=round_id,before=len(selected),candidates=len(candidates),conditional_selected=len(filtered),
            after=len(final),added=sorted(final-selected),removed=sorted(selected-final),solve=str(rr/'joint/solve'),
            final=json.loads((rr/'joint/solve/summary.json').read_text())['final'])
        history.append(row);atomic_write(root/'history.json',history,durable=False)
        print(json.dumps({k:v for k,v in row.items() if k!='final'}),flush=True)
        stable=stable+1 if final==selected and filtered==final else 0
        fingerprint=tuple(sorted(final))
        if stable>=2:termination='membership_stable_two_rounds'
        elif fingerprint in seen and final!=selected:termination='membership_cycle'
        seen.append(tuple(sorted(selected)));selected=final;previous_d,previous,cfg=nd,ns,nc
        control=root/'stop_after_round.json'
        if control.exists() and round_id>=int(json.loads(control.read_text())['round']):
            termination='user_requested_round_stop'
        if termination!='round_limit':break
    # Ordinary-only export is compatible with the established selected dataset runner.
    export=induced_graph(parent,sorted(selected),cfg);np.savez(root/'tensors.npz',**export)
    path=root/'selected.afv'
    if not path.exists():write_afv(settings['afv'],path,[curves[s] for s in sorted(selected)],dict(source='iterative_GT_selected',gt_assisted=True))
    cross=root/'cross';cross.mkdir(exist_ok=True)
    final_events=json.loads((Path(history[-1]['solve']).parent/'cross/cross_events.json').read_text())['events']
    atomic_write(cross/'cross_events.json',dict(events=remap_events(final_events,export)),durable=False)
    summary=dict(termination=termination,converged=termination=='membership_stable_two_rounds',rounds=len(history),
        selected_sources=len(selected),solve=history[-1]['solve'],dataset=str(root),seconds=time.monotonic()-started,
        note='GT-assisted selection, not independent ground truth; whole source fibers retained when their slab span overlaps GT H winding ranges.')
    atomic_write(root/'summary.json',summary,durable=False);print(json.dumps(summary),flush=True)
    return summary


def finish_existing(settings):
    """Boundary-stop a pre-control coordinator, then export its saved rounds.

    Uses only the explicitly identified coordinator, never other GPU jobs.
    No solved data are deleted or recalculated.
    """
    import os
    import signal
    import subprocess
    pid=int(settings['coordinator_pid'])
    config_path=Path(settings['config_path']).resolve()
    original=json.loads(config_path.read_text())['fiber_neighbor_expansion']
    root=Path(original['output_root'])
    target=int(settings['finish_existing_after_round'])
    def alive():
        result=subprocess.run(['ps','-p',str(pid),'-o','args='],capture_output=True,text=True)
        command=result.stdout.strip()
        if result.returncode or not command:return False
        import shlex
        words=shlex.split(command)
        if len(words)!=5 or words[1:4]!=['-m','vesuvius.neural_tracing.fiber_trace_3d.fiber_workflows','--config'] or Path(words[4]).resolve()!=config_path:
            raise ValueError('Coordinator identity changed; refusing to signal PID')
        return True
    if not alive():raise ValueError('Requested coordinator is not running')
    atomic_write(root/'stop_after_round.json',dict(round=target,reason='user_requested'),durable=False)
    final=root/f'round_{target:03d}'/'joint/solve/summary.json'
    print(json.dumps(dict(waiting_for_round=target,coordinator_pid=pid)),flush=True)
    while not final.exists():
        if not alive():raise RuntimeError('Coordinator exited before requested round completed')
        time.sleep(.5)
    if alive():os.kill(pid,signal.SIGINT)
    deadline=time.monotonic()+60
    while alive():
        if time.monotonic()>deadline:raise RuntimeError('Coordinator did not stop; refusing concurrent resume')
        time.sleep(.5)
    print('Requested round saved; finalizing from completed checkpoints',flush=True)
    return run(original)
