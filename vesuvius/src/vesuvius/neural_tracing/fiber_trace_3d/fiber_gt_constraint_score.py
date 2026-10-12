"""Frozen-solution cross-constraint votes against intact reviewed GT fibers."""
import json
from pathlib import Path
import sqlite3

import numpy as np
from lasagna.inference_provenance import atomic_write, sha256_file
from . import fiber_uv_layout as uv


def arrays(path):
    with np.load(path) as src:return {k:src[k] for k in src.files}


def offset_votes(required):
    """One additive test-layout offset; ties round toward positive infinity."""
    required=np.asarray(required,dtype=float)
    if not len(required) or not np.isfinite(required).all():raise ValueError('No finite winding votes')
    votes=np.floor(required+.5).astype(np.int64)
    offsets,counts=np.unique(votes,return_counts=True)
    candidates=offsets[counts==counts.max()]
    offset=min(candidates,key=lambda x:(float(np.abs(required-x).mean()),abs(int(x)),int(x)))
    return int(offset),votes==offset,votes


def piece_at(d,source,arc):
    choices=np.flatnonzero((d['source_id']==source)&(d['start']<=arc)&(d['end']>=arc))
    # At a shared cut take the later piece, just as native preparation does.
    return int(choices[np.argmax(d['start'][choices])]) if len(choices) else None


def coarse_at(d,state,pick,piece,angle,k):
    anchor_turn=np.rint(float(d['node_phase'][piece,0])-float(d['phase'][piece]))
    return float(pick[piece]+angle-anchor_turn+state[piece,k+1])


def event_residuals(events,d,state,pick,k,gd,gs,gpick):
    """Required additive test winding correction, shared by both GT metrics."""
    return np.asarray([
        coarse_at(gd,gs['state'],gpick,e['gt_piece'],e['gt_angle'],len(gs['widths']))
        -coarse_at(d,state,pick,e['test_piece'],e['test_angle'],k)
        -e['phase_delta']+.5*e['family_delta']-e['signed_distance']
        for e in events],dtype=float)


def enabled_l1(required,enabled,parallel):
    """Equal enabled-event L1, with a single L1-optimal integer gauge."""
    required=np.asarray(required,dtype=float)
    enabled=np.asarray(enabled,dtype=bool);parallel=np.asarray(parallel,dtype=bool)
    if required.shape!=enabled.shape or parallel.shape!=enabled.shape:
        raise ValueError('Event masks must match residuals')
    values=np.sort(required[enabled])
    if not len(values) or not np.isfinite(required).all():
        raise ValueError('No finite enabled GT constraints')
    # Every real point in the median interval minimizes L1. If it contains
    # integers, choose the one closest to zero; otherwise test its neighbors.
    lo,hi=values[(len(values)-1)//2],values[len(values)//2]
    first,last=int(np.ceil(lo)),int(np.floor(hi))
    if first<=last:
        offset=min(max(0,first),last)
    else:
        offset=min((int(np.floor(lo)),int(np.ceil(hi))),
                   key=lambda x:(float(np.abs(values-x).mean()),abs(x),x))
    error=np.abs(required-offset)
    def part(mask):
        return dict(events=int(mask.sum()),l1=float(error[mask].mean()) if mask.any() else None)
    return dict(gt_enabled_l1=float(error[enabled].mean()),gt_global_offset=offset,
        gt_enabled_events=int(enabled.sum()),gt_total_events=len(required),
        gt_enabled_percent=float(100*enabled.mean()),gt_parallel=part(enabled&parallel),
        gt_perpendicular=part(enabled&~parallel),gt_disabled=part(~enabled))


class FrozenGTScorer:
    """Replay cached cross events without extracting geometry for each trial."""
    def __init__(self,spec,config):
        root=Path(spec['root']);gtroot=Path(spec['gt_root'])
        self.paths=[root/'inputs.json',root/spec['events'],gtroot/'uv_whole/config.json',
                    gtroot/'uv_whole/solution.npz',Path(config['tensor_cache'])]
        provenance=json.loads(self.paths[0].read_text())
        self.gcfg=json.loads(self.paths[2].read_text())
        self.paths.append(Path(self.gcfg['tensor_cache']))
        for key,path in [('tensors',self.paths[4]),('gt_tensors',self.paths[5]),
                         ('gt_solution',self.paths[3])]:
            if sha256_file(path)!=provenance[key]:raise ValueError('Changed GT scoring '+key)
        if not config.get('bucket_shared_phase') or not self.gcfg.get('bucket_shared_phase') or config['h_convention']!=self.gcfg['h_convention']:
            raise ValueError('GT scoring requires matching corrected coarse convention')
        self.d=arrays(self.paths[4]);self.gd=arrays(self.paths[5]);self.gs=arrays(self.paths[3])
        self.events=json.loads(self.paths[1].read_text())
        _,_,keep,self.gpick=uv.hard_assignment(self.gs['state'],len(self.gs['widths']),self.gcfg,best_nonbroken=True)
        if not keep.all():raise ValueError('Trusted GT must remain enabled')
        for e in self.events:
            if int(self.d['source_id'][e['test_piece']])!=e['test_source']:
                raise ValueError('Cached event piece identity mismatch')
            if int(self.gd['source_id'][e['gt_piece']])!=e['gt_source']:
                raise ValueError('Cached GT event piece identity mismatch')
        self.hashes={str(p):sha256_file(p) for p in self.paths}

    def score(self,solution,config):
        s=arrays(solution);k=len(s['widths'])
        for field in ('source_id','start','end','family'):
            if not np.array_equal(s[field],self.d[field]):raise ValueError('Changed scored solution '+field)
        _,_,keep,pick=uv.hard_assignment(s['state'],k,config,best_nonbroken=True)
        required=event_residuals(self.events,self.d,s['state'],pick,k,self.gd,self.gs,self.gpick)
        return enabled_l1(required,[keep[e['test_piece']] for e in self.events],
                          [e['parallel'] for e in self.events])


def run(settings):
    from .crop_benchmark import run as dispatch
    from .fiber_normal_convention import inward_solution
    from .fiber_normal_sign_visualization import load_umbilicus_frame
    from vesuvius.neural_tracing.fiber_follow.data.afv import read_afv_polylines,write_afv
    root=Path(settings['output_root']);root.mkdir(parents=True,exist_ok=True)
    configs=[json.loads(Path(p).read_text())['fiber_uv_layout'] for p in settings['configs']]
    cfg=configs[0];gtroot=Path(settings['gt_root'])
    gcfg=json.loads((gtroot/'uv_whole/config.json').read_text())
    d=arrays(cfg['tensor_cache']);gd=arrays(gcfg['tensor_cache'])
    gs=arrays(gtroot/'uv_whole/solution.npz')
    if not gcfg.get('bucket_shared_phase'):raise ValueError('GT must use corrected shared phase')
    if uv.violation_indices(gd,gs['state'],gs['widths'],gcfg).size:
        raise ValueError('GT has unresolved coarse ordering violations')
    for c in configs:
        if c['tensor_cache']!=cfg['tensor_cache'] or c['h_convention']!=gcfg['h_convention']:
            raise ValueError('Comparison must share test geometry and GT circulation convention')
        if not c.get('bucket_shared_phase'):raise ValueError('Corrected coarse phase required')
    signature=dict(version=1,configs={p:sha256_file(p) for p in settings['configs']},
                   tensors=sha256_file(cfg['tensor_cache']),gt_tensors=sha256_file(gcfg['tensor_cache']),
                   gt_solution=sha256_file(gtroot/'uv_whole/solution.npz'),
                   gt_normal=sha256_file(gcfg['stage1']),afv=sha256_file(cfg['afv']),
                   gt_afv=sha256_file(gcfg['afv']),umbilicus=sha256_file(cfg['umbilicus']),
                   extract_template=sha256_file(settings['extract_template']),
                   orient_template=sha256_file(settings['orient_template']))
    manifest=root/'inputs.json'
    if manifest.exists() and json.loads(manifest.read_text())!=signature:raise ValueError('Changed scoring inputs')
    atomic_write(manifest,signature,durable=False)
    test_curves=read_afv_polylines(cfg['afv'],np.unique(d['source_id']))
    gt_curves=read_afv_polylines(gcfg['afv'],np.unique(gd['source_id']))
    records=[];mapping={};curves={}
    for kind,items in [('test',test_curves),('gt',gt_curves)]:
        for row,xyz in items:
            cid=len(records)+1;mapping[cid]=(kind,int(row['id']));curves[cid]=xyz
            records.append((dict(row,id=cid),xyz))
    combined=root/'combined.afv'
    if not combined.exists():write_afv(cfg['afv'],combined,records,dict(source='gt_cross_constraint_score'))
    atomic_write(root/'fiber_ids.json',{str(k):v for k,v in mapping.items()},durable=False)
    for name,template in [('extract',settings['extract_template']),('orient',settings['orient_template'])]:
        native=json.loads(Path(template).read_text())
        native.update(afv=str(combined),full_volume=True,threads=settings.get('threads',128),
                      output=str(root/(name+'.json')),database=str(root/(name+'.sqlite')))
        if name=='orient':native['orient_database']=str(root/'extract.sqlite')
        path=root/(name+'_config.json');atomic_write(path,native,durable=False)
        if not Path(native['output']).exists():
            dispatch(dict(output_root=str(root/(name+'_runner')),reference_dir=str(gtroot),
                          input_files=[str(path)],commands=[dict(name=name,binary='vc_fiber_trace_chunk',
                                                             args=['afv-constraints',str(path)])]))
    if settings.get('prepare_only'):return dict(prepared=True,root=str(root))
    # Replay the EXACT frozen GT sweep, including its original event endpoints.
    # Adding new endpoints to a sweep can otherwise change its sign history.
    normal_root=root/'frozen_gt_normals'
    if not (normal_root/'index.json').exists():
        native=json.loads((gtroot/'orient_config.json').read_text())
        native.update(orient_database=gcfg['database'],normal_curves_output=str(normal_root),
                      database=str(root/'unused_curve_export.sqlite'),output=str(root/'frozen_gt_normals.json'))
        path=root/'frozen_gt_normals_config.json';atomic_write(path,native,durable=False)
        dispatch(dict(output_root=str(root/'frozen_gt_normals_runner'),reference_dir=str(gtroot),
                      input_files=[str(path)],commands=[dict(name='frozen_gt_normals',binary='vc_fiber_trace_chunk',
                                                         args=['afv-constraints',str(path)])]))
    with uv._ro(cfg['afv']) as db:metadata={k:json.loads(v) for k,v in db.execute('SELECT key,value FROM metadata')}
    axis=load_umbilicus_frame(cfg['umbilicus'],metadata['root'])
    angles={}
    for cid,xyz in curves.items():
        arc,_,angle,_=uv.fiber_geometry(xyz,0,axis)
        angles[cid]=(arc,angle*(1 if cfg['h_convention']=='ccw' else -1))
    gn=inward_solution(arrays(gcfg['stage1']));normal_sign=dict(zip(gn['ids'],gn['normal_sign']))
    from .fiber_uv_surface import CurveBank
    bank=CurveBank(gcfg['afv'],gd['source_id'],normal_root,gn)
    _,_,gkeep,gpick=uv.hard_assignment(gs['state'],len(gs['widths']),gcfg,best_nonbroken=True)
    if not gkeep.all():raise ValueError('GT must retain every fiber')
    events=[];excluded=dict(non_cross=0,outside_test_piece=0,invalid_gt_normal=0,tangent_connector=0)
    gauge_flips=0
    with sqlite3.connect(root/'orient.sqlite') as db:
        rows=db.execute('SELECT id,fiber_a,fiber_b,arc_a,arc_b,ax,ay,az,bx,by,bz,parallel_score,perpendicular_score,winding_distance,parallel_winding_distance,normal_a,normal_b FROM constraints ORDER BY id')
        for r in rows:
            ida,idb=int(r[1]),int(r[2]);ka,sa=mapping[ida];kb,sb=mapping[idb]
            if ka==kb:excluded['non_cross']+=1;continue
            if ka=='test':ti,gi,ts,gid,ta,ga,tx,gx,normal=ida,idb,sa,sb,r[3],r[4],r[5:8],r[8:11],r[16]
            else:ti,gi,ts,gid,ta,ga,tx,gx,normal=idb,ida,sb,sa,r[4],r[3],r[8:11],r[5:8],r[15]
            tp=piece_at(d,ts,ta);gp=piece_at(gd,gid,ga)
            if tp is None:excluded['outside_test_piece']+=1;continue
            if gp is None:raise ValueError('GT event outside intact curve')
            n=json.loads(normal)
            if not n['valid']:excluded['invalid_gt_normal']+=1;continue
            loc=int(np.searchsorted(bank.ids,gid))
            _,frozen,valid=bank.sample(np.array([loc]),np.array([ga]))
            raw=np.asarray(n['xyz']);agreement=float(raw@frozen[0])
            if not valid[0] or abs(agreement)<.5:excluded['invalid_gt_normal']+=1;continue
            inward=raw*(1 if agreement>=0 else -1)
            gauge_flips+=bool((agreement>=0)!=(normal_sign[gid]>=0))
            connector=np.asarray(gx)-tx;length=np.linalg.norm(connector)
            cosine=-float(np.dot(connector,inward))/max(length,1e-30)
            if length==0 or abs(cosine)<.1:excluded['tangent_connector']+=1;continue
            sign=1 if cosine>0 else -1
            at=float(np.interp(ta,*angles[ti]));ag=float(np.interp(ga,*angles[gi]))
            pd=(ag-at+.5)%1-.5;parallel=r[11]>r[12];distance=float(r[14] if parallel else r[13])
            events.append(dict(event_id=int(r[0]),test_source=ts,test_piece=tp,gt_source=gid,gt_piece=gp,
                test_arc=ta,gt_arc=ga,test_angle=at,gt_angle=ag,phase_delta=pd,
                family_delta=int(gd['family'][gp]-d['family'][tp]),parallel=bool(parallel),
                signed_distance=sign*distance,normal_cosine=cosine,
                class_matches_families=bool(parallel==(d['family'][tp]==gd['family'][gp]))))
    if not events:raise ValueError('No cross-constraint support')
    results=[]
    for c in configs:
        solution=Path(c['output_root'])/'solution.npz';s=arrays(solution);k=len(s['widths'])
        saved=json.loads((solution.parent/'summary.json').read_text())['config']
        for key in ('tensor_cache','afv','h_convention','winding_count','bucket_shared_phase',
                    'reference_initialization','reference_width_initialization'):
            if saved.get(key)!=c.get(key):raise ValueError('Saved solve config mismatch: '+key)
        if not np.isfinite(s['state']).all() or not np.isfinite(s['widths']).all():
            raise ValueError('Nonfinite saved solution')
        _,_,keep,pick=uv.hard_assignment(s['state'],k,c,best_nonbroken=True)
        if len(pick)!=len(d['source_id']):raise ValueError('Solution piece count mismatch')
        required=event_residuals(events,d,s['state'],pick,k,gd,gs,gpick)
        offset,correct,votes=offset_votes(required);required=np.asarray(required)
        enabled=np.asarray([keep[e['test_piece']] for e in events]);parallel=np.asarray([e['parallel'] for e in events])
        def stats(mask):
            n=int(mask.sum());return dict(constraints=n,correct=int(correct[mask].sum()),
                score_percent=float(100*correct[mask].mean()) if n else None)
        detail=[dict(e,disabled=bool(not enabled[n]),required_offset=float(required[n]),vote=int(votes[n]),
                     correct=bool(correct[n]),corrected_residual=float(required[n]-offset)) for n,e in enumerate(events)]
        name=Path(c['output_root']).name
        result=dict(name=name,solution=str(solution),solution_sha256=sha256_file(solution),
                    additive_global_offset=offset,**stats(np.ones(len(events),bool)),
                    mean_absolute_residual=float(np.abs(required-offset).mean()),
                    enabled=stats(enabled),disabled=stats(~enabled),parallel=stats(parallel),perpendicular=stats(~parallel),
                    by_gt={str(int(g)):stats(np.asarray([e['gt_source']==g for e in events])) for g in gd['source_id']},
                    excluded=excluded,class_disagreements=sum(not e['class_matches_families'] for e in events),
                    combined_sweep_gauge_corrections=gauge_flips,
                    gt_config_sha256=sha256_file(gtroot/'uv_whole/config.json'),
                    frozen_gt_normal_index_sha256=sha256_file(normal_root/'index.json'),
                    metric='equal event votes; nearest integer, ties toward +infinity; one additive test offset',
                    signing='GT endpoint inward normal only; abs cosine>=0.1',
                    disabled_policy='include best nonbroken hypothesis; retain disabled status')
        atomic_write(root/(name+'_events.json'),detail,durable=False)
        results.append(result);print(json.dumps(result),flush=True)
    atomic_write(root/'results.json',dict(inputs=signature,results=results),durable=False)
    return results
