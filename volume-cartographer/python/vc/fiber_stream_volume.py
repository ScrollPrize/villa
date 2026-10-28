"""Continuous CPU beam tracing with the neural full-volume coverage policy.

Seeds come from the shared native chunk seed generator. Native workers retain
their beam/width history across storage chunks. One coordinator batches collision
queries; it never performs beam sampling. Output tiles are atomic msgpack+zstd,
with seed provenance, widths, costs, stop reasons and endpoint links.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import threading
import time

import msgpack
import numpy as np
from numcodecs import Zstd

from vc import fiber_trace as ft
from vc.fiber_chunk_volume import configure_loader, open_inputs, job_keys, output_lock
from vesuvius.neural_tracing.tracers.fibers.tracing.volume_policy import (
    Owned, LivePaths, deferred_owner, novel_parts, tangents, runs, outside_ct_cut)


def atomic_json(path, value):
    tmp=path.with_suffix('.partial.json')
    tmp.write_text(json.dumps(value,indent=2)+'\n')
    tmp.replace(path)


def save_page(path, pieces):
    data=msgpack.packb(dict(version=1,coordinate_units='base_voxels',pieces=pieces),use_bin_type=True)
    tmp=path.with_suffix('.partial')
    tmp.write_bytes(Zstd(level=1).encode(data))
    tmp.replace(path)


def read_page(path):
    return msgpack.unpackb(Zstd().decode(path.read_bytes()),raw=False)


def uniform_path(points, widths, losses, spacing=8.):
    """Neural policy counts samples at its 8-base grid; beam commits are 16 base."""
    points=np.asarray(points); widths=np.asarray(widths); losses=np.asarray(losses)
    arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
    if arc[-1]<=0:
        return points[:1],widths[:1],losses[:1]
    at=np.r_[np.arange(0.,arc[-1],spacing),arc[-1]]
    return (np.stack([np.interp(at,arc,points[:,a]) for a in range(3)],1),
            np.interp(at,arc,widths),np.interp(at,arc,losses))


class NativeCT:
    """Use the regular native cache for the neural policy's CT-zero checks."""
    def __init__(self, locator, level=2):
        from vc.volume import Volume
        self.volume=Volume.open_url(locator)
        self.level=level-self.volume.base_scale_level
        if self.level<0 or not self.volume.has_scale_level(self.level):
            raise ValueError('CT requested level is unavailable')
        # Native coordinates are relative to the opened base, NOT selected level.
        self.scale=2.**self.volume.base_scale_level

    def outside(self, base_xyz):
        p=np.ascontiguousarray(np.asarray(base_xyz,dtype=np.float32).reshape(1,-1,3)/self.scale)
        if not p.shape[1]:
            return np.empty(0,bool)
        # Native nearest sampling applies the requested level itself.
        image, *_=self.volume.sample_coords(p,np.ones(p.shape[:2],bool),level=self.level,
            sampling='nearest',blocking=True,sampler_threads=1)
        return np.asarray(image).ravel()==0


class Stream:
    def __init__(self, cfg, out):
        self.cfg=cfg;self.out=out;self.options=cfg['stream']
        self.rows_limit=int(self.options.get('rows',128))
        if self.rows_limit<2 or self.rows_limit%2:
            raise ValueError('rows must be an even positive directional-trace count')
        self.seed_workers=int(self.options.get('seed_workers',8))
        self.grid=float(self.options.get('policy_grid_base',8))
        self.radius=2*self.grid
        self.spread=16*self.grid
        self.next_pair=0;self.next_fiber=0;self.epoch=0
        self.events=queue.Queue();self.cancel=threading.Event();self.control=ft.StreamControl()
        self.rows={};self.pairs={};self.tiles=[]
        self.pending_saves=[];self.counts=Counter();self.timings=Counter()
        self.begin=time.monotonic();self.start_cpu=time.process_time()
        self.last_progress=self.begin;self.peak_active=0
        base={k:v for k,v in cfg.items() if k!='stream'}
        configure_loader(base,self.rows_limit)
        self.field,self.normals,self.trace,self.chunk=open_inputs(base)
        if self.chunk.gpu_library or self.trace.parallel_threads!=1:
            raise ValueError('stream requires CPU seed construction and one native thread per trace')
        self.scale=self.field.trace_to_base_scale
        self.tolerance=float(self.trace.width_hard_relative_tolerance)
        if self.tolerance<=0:
            raise ValueError('stream requires the configured hard width tolerance')
        self.trace_config=self.trace.to_dict()
        self.steps=round(float(self.options.get('decision_base',64))/(self.trace.step_voxels*self.scale))
        if self.steps<1 or not math.isclose(self.steps*self.trace.step_voxels*self.scale,
                                          float(self.options.get('decision_base',64))):
            raise ValueError('collision cadence must be an integer number of beam steps')
        _,_,_,shape=ft.prediction_presence_storage(self.field)
        self.lo=np.array(self.options.get('box_base_xyz',[[0,0,0],list(shape[::-1])])[0],float)
        self.hi=np.array(self.options.get('box_base_xyz',[[0,0,0],list(shape[::-1])])[1],float)
        self.ct=NativeCT(self.options['ct_locator'],int(self.options.get('ct_level',2)))
        self.owned=Owned(self.chunk.core_base);self.live=LivePaths(self.radius)
        self.keys=job_keys(base,self.field,self.chunk)
        self.total_tiles=len(self.keys);self.cursor=0
        self.done=set()
        (out/'tiles').mkdir(exist_ok=True)
        identity=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
        record=out/'run.json'
        if record.exists():
            if json.loads(record.read_text()).get('output_schema'):
                raise ValueError('Python-reference is a legacy format control; cannot resume attributed native output')
            if json.loads(record.read_text())['identity']!=identity:
                raise ValueError('resume settings/source differ')
        else:
            atomic_json(record,dict(identity=identity,config=cfg,policy='neural-volume+width-v1',
                coordinates='base voxels',decision_base=self.steps*self.trace.step_voxels*self.scale))
        for page in sorted((out/'tiles').glob('*.msgpack.zst')):
            for p in read_page(page)['pieces']:
                points=np.asarray(p['points_base'])
                self.owned.add(points,p['id'],p['width_mean_base'])
                self.next_fiber=max(self.next_fiber,p['id']+1)
                self.counts['pieces']+=1
                self.counts['kept_base']+=float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum())
            self.done.add(page.name.removesuffix('.msgpack.zst'))
        progress=out/'progress.json'
        if progress.exists():
            self.next_fiber=max(self.next_fiber,json.loads(progress.read_text())['next_fiber_id'])
        self.owned.flush()
        self.counts['finished_tiles']=len(self.done)

    @staticmethod
    def name(key):
        return '_'.join(map(str,key))

    def prefetch(self, pool):
        limit=int(self.options.get('max_open',64))
        while len(self.tiles)<limit and self.cursor<len(self.keys):
            key=self.keys[self.cursor];self.cursor+=1
            name=self.name(key)
            if name in self.done:
                continue
            self.tiles.append(dict(key=key,name=name,
                future=pool.submit(ft.chunk_seeds,self.field,self.normals,key,self.chunk),
                pending=None,active=0,pieces=[],deferred={}))

    def admit(self, pool):
        # Same min-feed policy as the neural streamer, scaled for 128 rows.
        if (self.rows_limit-len(self.rows))//2 < min(int(self.options.get('min_feed',8)),self.rows_limit//4 or 1):
            return
        running=np.array([s['seed'] for s in self.pairs.values()]).reshape(-1,3)
        running_widths=np.array([s['seed_width'] for s in self.pairs.values()])
        running_ids=list(self.pairs)
        chosen=[];chosen_widths=[];chosen_ids=[]
        # Visit ready tiles only. A slow load cannot block tracing other tiles.
        for tile in self.tiles:
            if tile['pending'] is None:
                if not tile['future'].done():
                    continue
                tile['seeds']=tile['future'].result()
                tile['pending']=list(range(len(tile['seeds']['points_base'])))
                self.counts['seed_candidates']+=len(tile['pending'])
            d=tile['seeds'];later=[];pending=[]
            for k in tile['pending']:
                if tile['deferred'].get(k) in self.pairs:
                    later.append(k)
                else:
                    pending.append(k)
            # Batch covered-seed queries, not one KD query per seed.
            for begin in range(0,len(pending),128):
                part=pending[begin:begin+128]
                if len(self.rows)+2>self.rows_limit:
                    later+=pending[begin:];break
                owned=self.owned.owner(d['points_base'][part],d['directions'][part],self.radius,
                    widths=d['widths_base'][part],tolerance=self.tolerance)
                blocked=deferred_owner(d['points_base'][part],running,self.spread,
                    d['widths_base'][part],running_widths,self.tolerance)
                for k,own,wait in zip(part,owned,blocked):
                    if own>=0:
                        self.counts['covered_seeds']+=1;continue
                    if len(self.rows)+2>self.rows_limit:
                        later.append(k);continue
                    p=d['points_base'][k];w=d['widths_base'][k]
                    if np.any(p<self.lo) or np.any(p>=self.hi):
                        continue
                    if wait>=0:
                        tile['deferred'][k]=running_ids[wait];later.append(k);continue
                    chosen_block=deferred_owner([p],chosen,self.spread,[w],chosen_widths,self.tolerance)[0]
                    if chosen_block>=0:
                        tile['deferred'][k]=chosen_ids[chosen_block];later.append(k);continue
                    chosen.append(p);chosen_widths.append(w)
                    pair=self.next_pair;self.next_pair+=1;tile['active']+=1
                    chosen_ids.append(pair)
                    self.pairs[pair]=dict(seed=p,seed_width=w,tile=tile,halves={},direction=d['directions'][k])
                    for half,sign in enumerate((1.,-1.)):
                        row=pair*2+half
                        self.rows[row]=dict(pair=pair,half=half,prev=p.copy(),streak=0,outside_streak=0)
                        pool.submit(self.trace_one,row,p,d['directions'][k]*sign,w)
                    self.counts['started_pairs']+=1
            tile['pending']=sorted(later)  # retain the original strongest-seed order
            self.peak_active=max(self.peak_active,len(self.rows))
            if len(self.rows)+2>self.rows_limit:
                break

    def trace_one(self,row,point,direction,width):
        waited=0.;began=time.monotonic()
        try:
            cfg=ft.TraceConfig.from_dict(self.trace_config)
            result=ft.trace_stream(self.field,self.normals,point/self.scale,direction,
                float(self.options.get('maximum_base',64000))/self.scale,cfg,self.steps,self.control,
                width/self.scale,row)
            waited=result['wait_seconds']
            p,w,loss=uniform_path(np.asarray(result['points'])*self.scale,
                                 np.asarray(result['widths'])*self.scale,result['losses'],self.grid)
            outside=self.ct.outside(p)
            self.events.put(dict(kind='finished',row=row,points=p,widths=w,losses=loss,
                outside=outside,reason=result['reason'],seconds=time.monotonic()-began,wait=waited,
                width_count=result['width_count'],width_mean=result['width_mean']*self.scale))
        except BaseException as exc:
            self.events.put(dict(kind='error',error=exc,row=row))

    def check(self, events, outside):
        if not events:return
        started=time.monotonic();self.epoch+=1
        pos=np.array([e['point'] for e in events])*self.scale
        dirs=np.array([e['direction'] for e in events])
        phase=time.monotonic()
        widths=np.array([e['width'] for e in events])*self.scale
        pair=np.array([self.rows[e['row']]['pair'] for e in events])
        prev=np.array([self.rows[e['row']]['prev'] for e in events])
        near=(self.owned.owner(pos,dirs,self.radius,widths=widths,tolerance=self.tolerance)>=0)
        self.timings['check_owned_seconds']+=time.monotonic()-phase
        phase=time.monotonic()
        near|=self.live.hits(pos,dirs,pair,self.epoch,widths,self.tolerance)
        self.timings['check_live_query_seconds']+=time.monotonic()-phase
        phase=time.monotonic()
        self.live.add(prev,pos,pair,self.epoch,widths)
        self.timings['check_live_add_seconds']+=time.monotonic()-phase
        for i,e in enumerate(events):
            state=self.rows[e['row']];state['prev']=pos[i]
            state['streak']=state['streak']+1 if near[i] else 0
            state['outside_streak']=state['outside_streak']+1 if outside[i] else 0
            reason=None
            if np.any(pos[i]<self.lo-4*self.grid) or np.any(pos[i]>self.hi+4*self.grid):reason='bounds'
            elif state['outside_streak']>=2:reason='outside_ct'
            elif state['streak']>=3:reason='owned_or_live'
            if reason:state['reason']=reason;e['keep']=False
            self.control.respond(e['row'],e.get('keep',True))
        self.counts['collision_checks']+=len(events)
        self.timings['collision_seconds']+=time.monotonic()-started

    def finish(self,e):
        state=self.rows.pop(e['row']);pair=state['pair'];half=state['half']
        e['reason']=state.get('reason',e['reason'])
        self.counts['stop_'+e['reason']]+=1
        self.timings['worker_wall_seconds']+=e['seconds']
        self.timings['collision_wait_seconds']+=e['wait']
        info=self.pairs[pair];info['halves'][half]=e
        if len(info['halves'])<2:return
        del self.pairs[pair];self.live.drop([pair]);info['tile']['active']-=1
        fw,bw=info['halves'][0],info['halves'][1]
        self.counts['finished_pairs']+=1
        if len(fw['points'])+len(bw['points'])-1<48:
            return  # no clipping or duplicate trimming can make this long enough
        p=np.concatenate([bw['points'][::-1],fw['points'][1:]])
        w=np.concatenate([bw['widths'][::-1],fw['widths'][1:]])
        loss=np.concatenate([bw['losses'][::-1],fw['losses'][1:]])
        outside=np.r_[bw['outside'][::-1],fw['outside'][1:]]
        inside=np.all(p>=self.lo,1)&np.all(p<=self.hi,1)&~outside_ct_cut(outside,16)
        candidates=[]
        # Exactly the neural writer: evaluate every part before adding any of
        # this pair's new pieces to owned coverage.
        for a,b in runs(inside):
            if b-a<2:continue
            for s,t in novel_parts(p[a:b],self.owned,self.radius,24,16,48,
                                   w[a:b],self.tolerance):
                candidates.append((a+s,a+t))
        for a,b in candidates:
            piece=p[a:b];width=w[a:b];fid=self.next_fiber;self.next_fiber+=1
            links=self.owned.owner(piece[[0,-1]],tangents(piece)[[0,-1]],self.radius+self.grid,
                                   widths=width[[0,-1]],tolerance=self.tolerance)
            mean=float((fw['width_mean']*fw['width_count']+bw['width_mean']*bw['width_count']-
                        info['seed_width'])/(fw['width_count']+bw['width_count']-1))
            self.owned.add(piece,fid,mean)
            info['tile']['pieces'].append(dict(id=fid,points_base=piece.tolist(),
                width_mean_base=mean,width_history_base=width.tolist(),cumulative_cost=loss[a:b].tolist(),
                seed_base=info['seed'].tolist(),seed_direction=info['direction'].tolist(),
                reasons=[bw['reason'],fw['reason']],links=links.tolist()))
            self.counts['pieces']+=1
            self.counts['kept_base']+=float(np.linalg.norm(np.diff(piece,axis=0),axis=1).sum())

    def close_tiles(self,writers):
        for tile in list(self.tiles):
            if tile['pending']==[] and tile['active']==0:
                self.tiles.remove(tile)
                self.pending_saves.append((tile['name'],writers.submit(save_page,
                    self.out/'tiles'/(tile['name']+'.msgpack.zst'),tile['pieces'])))
        for name,future in list(self.pending_saves):
            if future.done():
                future.result();self.pending_saves.remove((name,future))
                self.done.add(name);self.counts['finished_tiles']+=1

    def progress(self,final=False):
        now=time.monotonic()
        elapsed=now-self.begin;cpu=time.process_time()-self.start_cpu
        last_cpu=getattr(self,'last_cpu',0.)
        report=dict(self.counts,**self.timings,wall_seconds=elapsed,cpu_seconds=cpu,
            mean_cpu_cores=cpu/max(elapsed,1e-9),active_directions=len(self.rows),
            interval_cpu_cores=(cpu-last_cpu)/max(now-self.last_progress,1e-9),
            peak_active=self.peak_active,open_tiles=len(self.tiles),total_tiles=self.total_tiles,
            next_fiber_id=self.next_fiber,complete=final)
        atomic_json(self.out/('summary.json' if final else 'progress.json'),report)
        print(json.dumps(report),flush=True);self.last_progress=now;self.last_cpu=cpu
        return report

    def run(self):
        with ThreadPoolExecutor(self.rows_limit) as traces, ThreadPoolExecutor(self.seed_workers) as seeds, \
             ThreadPoolExecutor(2) as writers, ThreadPoolExecutor(4) as ct_workers:
            ct_pending=[]
            try:
                while True:
                    self.prefetch(seeds)
                    t=time.monotonic();self.admit(traces);self.timings['admission_seconds']+=time.monotonic()-t
                    self.close_tiles(writers)
                    if not self.rows and not self.tiles and self.cursor>=len(self.keys):break
                    checks=self.control.poll(1);completed=[]
                    if checks:
                        pos=np.array([e['point'] for e in checks])*self.scale
                        ct_pending.append((checks,ct_workers.submit(self.ct.outside,pos),time.monotonic()))
                    while True:
                        try:e=self.events.get_nowait()
                        except queue.Empty:break
                        if e['kind']=='error':raise e['error']
                        completed.append(e)
                    # A cache miss in one batch must not stall unrelated traces.
                    # Each native row has at most one outstanding decision.
                    ready=[];ready_outside=[]
                    for batch,future,submitted in list(ct_pending):
                        if future.done():
                            ct_pending.remove((batch,future,submitted))
                            self.timings['check_ct_batch_latency_seconds']+=time.monotonic()-submitted
                            ready.extend(batch);ready_outside.append(future.result())
                    if ready:self.check(ready,np.concatenate(ready_outside))
                    t=time.monotonic()
                    for e in completed:self.finish(e)
                    self.timings['finalization_seconds']+=time.monotonic()-t
                    if time.monotonic()-self.last_progress>10:self.progress()
                for _,f in self.pending_saves:f.result()
                self.counts['finished_tiles']+=len(self.pending_saves);self.pending_saves=[]
            finally:
                self.cancel.set();self.control.cancel()
        return self.progress(True)


def run_native(cfg,out):
    """Metadata-only adapter: all scheduling and tracing lives in C++."""
    options=dict(cfg['stream'])
    options.setdefault('workers',options.get('rows',128))
    options.setdefault('active_fibers',10*options['workers'])
    base={k:v for k,v in cfg.items() if k!='stream'}
    configure_loader(base,options['workers'])
    field,normals,trace,chunks=open_inputs(base)
    if chunks.gpu_library or trace.parallel_threads!=1:
        raise ValueError('native streaming uses one shared CPU pool')
    _,_,_,shape=ft.prediction_presence_storage(field)
    options.setdefault('box_base_xyz',[[0,0,0],list(shape[::-1])])
    keys=job_keys(base,field,chunks)
    identity=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
    record=out/'run.json'
    if record.exists():
        saved=json.loads(record.read_text())
        if saved.get('output_schema')!='fiber-stream-attributes-v2':
            raise ValueError('old fiber output cannot be resumed; start a fresh attributes-v2 run')
        if saved['identity']!=identity:
            raise ValueError('resume settings/source differ')
    else:
        atomic_json(record,dict(identity=identity,config=cfg,policy='neural-volume+width-v1',
            output_schema='fiber-stream-attributes-v2',
            coordinates='base voxels',decision_base=options.get('decision_base',64)))
    atomic_json(out/'scheduler.json',dict(backend='native-cooperative-v1',options=options))
    return json.loads(ft.run_stream_volume(field,normals,trace,chunks,json.dumps(options),keys,str(out)))

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('config',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--python-reference',action='store_true',help='old coordinator, for matched diagnostics only')
    args=p.parse_args(argv);cfg=json.loads(args.config.read_text())
    args.output.mkdir(parents=True,exist_ok=True)
    with output_lock(args.output):
        if not args.python_reference:
            run_native(cfg,args.output)
            return 0
        stream=Stream(cfg,args.output)
        if os.environ.get('VC_STREAM_PROFILE'):
            import cProfile
            import pstats
            profile=cProfile.Profile()
            profile.runcall(stream.run)
            profile.dump_stats(str(args.output/'coordinator.prof'))
            pstats.Stats(profile).strip_dirs().sort_stats('tottime').print_stats(25)
        else:
            stream.run()
    return 0
