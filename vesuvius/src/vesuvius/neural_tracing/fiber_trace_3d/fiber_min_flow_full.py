"""Bounded parallel full-volume adapter for the shared certified flow solver.

Original object stores stay read-only. Exact polyline geometry is packed once;
disjoint workers stream constraint-ID ranges and combine only identical L1 arcs.
"""
import concurrent.futures
import json
import multiprocessing
import os
from pathlib import Path
import time

import numpy as np

from .fiber_min_flow import Seam,aggregate_l1_edges,solve
from .fiber_min_flow_benchmark import graph
from .fiber_min_flow_experiment import arrays,write
from .fiber_graph_store import FiberGraph, store_digest, write_winding_result, workflow_volume
from vesuvius.neural_tracing.tracers.shared.annotations.fiber_store import read_polylines, fiber_catalog, fiber_volume

KEY_DTYPE=np.dtype([('source','<i8'),('arc','<f8')])
PACKED_GEOMETRY_FORMAT=3
_worker=None


class PackedPaths:
    """Same endpoint evaluation as Paths, without per-fiber query loops."""
    def __init__(self,root,data=None,mode='r'):
        self.root=Path(root);self.data=data
        self.catalog=arrays(self.root/'catalog.npz')
        if int(self.catalog.get('format_version',0)) != PACKED_GEOMETRY_FORMAT or 'volume' not in self.catalog:
            raise ValueError('Packed geometry cache lacks a verified working frame; rebuild it from the vcobj source')
        self.volume = str(self.catalog['volume'].item())
        if not self.volume:
            raise ValueError('Packed geometry cache has an empty working volume; rebuild it')
        self.ids=self.catalog['source_id'];self.offset=self.catalog['point_offset']
        self.keys=np.load(self.root/'keys.npy',mmap_mode=mode)
        self.xyz=np.load(self.root/'xyz.npy',mmap_mode=mode)
        self.turns=np.load(self.root/'turns.npy',mmap_mode=mode)
        self.seam=Seam(self.catalog['axis'])

    def at_many(self,source,arc):
        source=np.asarray(source);arc=np.asarray(arc)
        fiber=np.searchsorted(self.ids,source)
        if np.any(fiber==len(self.ids)) or not np.array_equal(self.ids[fiber],source):raise ValueError('Unknown fiber')
        start=self.offset[fiber];end=self.offset[fiber+1]
        length=self.keys['arc'][end-1]
        if np.any(arc < -1e-4) or np.any(arc>length+np.maximum(1e-4,2e-7*length)):raise ValueError('Endpoint outside fiber')
        values=np.clip(arc,0,length)
        query=np.empty(len(source),KEY_DTYPE);query['source']=fiber;query['arc']=values
        j=np.minimum(np.searchsorted(self.keys,query,side='right')-1,end-2)
        if np.any(j<start):raise ValueError('Invalid packed arc search')
        a=self.keys['arc'][j];b=self.keys['arc'][j+1]
        p=self.xyz[j]+(self.xyz[j+1]-self.xyz[j])*((values-a)/np.maximum(b-a,1e-300))[:,None]
        return p,self.turns[j]+self.seam.segments(self.xyz[j],p)

    def ends(self,side):
        d=self.data;pick=d['i' if side=='a' else 'j']
        return self.at_many(self.ids[pick],d['ds_'+side])

    def at(self,source,arc):
        xyz,turns=self.at_many(np.array([source]),np.array([arc]))
        return xyz[0],int(turns[0])


def _init(config,stage):
    global _worker
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    paths=PackedPaths(Path(config['output_root'])/'geometry',mode='r+' if stage=='geometry' else 'r')
    _worker=(config,paths,None if stage=='geometry' else FiberGraph.cached(config['database'], Path(config['output_root'])/'constraint_columns'))


def geometry_job(job):
    config,paths,db=_worker
    first,last=job;root=Path(config['output_root'])/'geometry';marker=root/f'block_{first:08d}.json'
    if marker.exists():return json.loads(marker.read_text())
    started=time.monotonic()
    curves=read_polylines(config['fibers'],paths.ids[first:last],volume=paths.volume)
    for i,(row,xyz) in enumerate(curves,first):
        lo,hi=paths.offset[i:i+2]
        if len(xyz)!=hi-lo or row['id']!=paths.ids[i]:raise ValueError('Fiber catalog geometry mismatch')
        paths.xyz[lo:hi]=xyz
        paths.keys['source'][lo:hi]=i
        paths.keys['arc'][lo:hi]=np.r_[0,np.cumsum(np.linalg.norm(np.diff(xyz,axis=0),axis=1))]
        paths.turns[lo:hi]=paths.seam.polyline(xyz)
    result=dict(first=first,last=last,fibers=last-first,points=int(paths.offset[last]-paths.offset[first]),seconds=time.monotonic()-started)
    write(marker,result)
    return result


def event_batches(config,paths,db,first,last):
    """Shared memory-mapped graph columns; no per-worker database scan/JSON."""
    from .fiber_uv_layout import signed_order
    catalog=paths.catalog; ids=paths.ids
    for sl in db.batches(config.get('read_batch',32768),first,last):
        sa,sb=db.fiber_a[sl],db.fiber_b[sl]
        a=np.searchsorted(ids,sa); b=np.searchsorted(ids,sb)
        valid=(a<len(ids))&(b<len(ids))
        a=np.minimum(a,len(ids)-1); b=np.minimum(b,len(ids)-1)
        valid&=(ids[a]==sa)&(ids[b]==sb)&db.valid_a[sl]&db.valid_b[sl]
        par=db.parallel[sl]
        valid&=par==(catalog['family'][a]==catalog['family'][b])
        a=a[valid]; b=b[valid]; par=par[valid]
        sign=signed_order((db.point_b[sl]-db.point_a[sl])[valid],
            db.normal_a[sl][valid]*catalog['normal_sign'][a,None],
            db.normal_b[sl][valid]*catalog['normal_sign'][b,None])
        d=dict(i=a,j=b,ds_a=db.arc_a[sl][valid],ds_b=db.arc_b[sl][valid],family=catalog['family'],
               parallel=par,sign=sign,distance=np.where(par,db.parallel_distance[sl][valid],db.distance[sl][valid]),
               event_ids=db.event_ids[sl][valid])
        yield sl.stop-sl.start,d


def constraint_job(job):
    config,paths,db=_worker
    first,last=job;root=Path(config['output_root'])/'graph_parts';root.mkdir(exist_ok=True)
    marker=root/f'{first:010d}.json';out=root/f'{first:010d}.npz'
    if marker.exists() and out.exists():return json.loads(marker.read_text())
    started=time.monotonic();chunks=[];read=0;used=0;expanded=0
    for count,d in event_batches(config,paths,db,first,last):
        read+=count
        if not len(d['i']):continue
        paths.data=d
        g,transport=graph(config,d,paths)
        used+=int(transport['valid'].sum());expanded+=len(g['a'])
        chunks.append(aggregate_l1_edges(g['a'],g['b'],g['target'],g['weight']))
    keys=('a','b','target','weight')
    combined={k:np.concatenate([c[k] for c in chunks]) if chunks else np.empty(0,dtype=float if k=='weight' else np.int64) for k in keys}
    merged=aggregate_l1_edges(**combined)
    temporary=out.with_suffix('.partial.npz');np.savez(temporary,**merged);temporary.replace(out)
    result=dict(first=first,last=last,read=read,used=used,expanded=expanded,arcs=len(merged['a']),seconds=time.monotonic()-started)
    write(marker,result);return result


def parallel_stage(config,stage,jobs,worker):
    started=time.monotonic();results=[]
    write(Path(config['output_root'])/'progress.json',dict(stage=stage,completed=0,total=len(jobs),seconds=0))
    with concurrent.futures.ProcessPoolExecutor(max_workers=config.get('workers',32),
            mp_context=multiprocessing.get_context('spawn'),initializer=_init,initargs=(config,stage)) as pool:
        pending={pool.submit(worker,j):j for j in jobs}
        try:
            for future in concurrent.futures.as_completed(pending):
                results.append(future.result())
                elapsed=time.monotonic()-started
                progress=dict(stage=stage,completed=len(results),total=len(jobs),seconds=elapsed,
                    eta_seconds=elapsed/max(len(results),1)*(len(jobs)-len(results)))
                write(Path(config['output_root'])/'progress.json',progress)
                print(json.dumps(progress),flush=True)
        except BaseException as error:
            for future in pending:future.cancel()
            write(Path(config['output_root'])/'progress.json',dict(stage=stage,error=str(error),completed=len(results)))
            raise
    results.sort(key=lambda r:r['first'])
    write(Path(config['output_root'])/(stage+'_report.json'),dict(seconds=time.monotonic()-started,blocks=results))
    return results


def prepare(config):
    from .fiber_normal_convention import inward_solution
    from .fiber_normal_sign_visualization import load_umbilicus_frame
    root=Path(config['output_root']);geo=root/'geometry';geo.mkdir(parents=True,exist_ok=True)
    comparison=arrays(config['comparison']);normal=inward_solution(arrays(config['normal_solution']))
    if not np.array_equal(comparison['ids'],normal['ids']):raise ValueError('HV/normal identity differs')
    keep=comparison['labels']!=2
    ids=comparison['ids'][keep];family=comparison['labels'][keep];sign=normal['normal_sign'][keep]
    volume=workflow_volume(config)
    # Only point counts/identity come from the source binding. Coordinates,
    # arc lengths and seam transport are computed in the measured graph frame.
    rows=fiber_catalog(config['fibers'],fiber_volume(config['fibers']))
    catalog_ids=np.asarray([r['id'] for r in rows])
    p=np.searchsorted(catalog_ids,ids)
    if np.any(p>=len(rows)) or not np.array_equal(catalog_ids[p],ids):raise ValueError('Fiber identity differs')
    offsets=np.r_[0,np.cumsum(np.asarray([r['point_count'] for r in rows])[p])]
    catalog=dict(format_version=np.array(PACKED_GEOMETRY_FORMAT),volume=np.array(volume),source_id=ids,family=family,normal_sign=sign,point_offset=offsets,axis=load_umbilicus_frame(config['umbilicus'],volume))
    if not (geo/'catalog.npz').exists():np.savez(geo/'catalog.npz',**catalog)
    else:
        saved=arrays(geo/'catalog.npz')
        if any(k not in saved or not np.array_equal(v,saved[k]) for k,v in catalog.items()):raise ValueError('Geometry catalog changed; rebuild the cache')
    for name,dtype,shape in [('keys',KEY_DTYPE,(int(offsets[-1]),)),('xyz',np.float64,(int(offsets[-1]),3)),('turns',np.int64,(int(offsets[-1]),))]:
        path=geo/(name+'.npy')
        if not path.exists():
            array=np.lib.format.open_memmap(path,mode='w+',dtype=dtype,shape=shape);del array
    print(json.dumps(dict(stage='catalog',fibers=len(ids),excluded=int((~keep).sum()),points=int(offsets[-1]),
                          geometry_bytes=int(offsets[-1])*48)),flush=True)
    batch=config.get('geometry_batch',4096)
    jobs=[(i,min(i+batch,len(ids))) for i in range(0,len(ids),batch)]
    parallel_stage(config,'geometry',jobs,geometry_job)
    columns=FiberGraph.cached(config['database'],root/'constraint_columns')
    if not len(columns.event_ids): raise ValueError('Empty constraint graph')
    first,last=map(int,columns.event_ids[[0,-1]])
    batch=config.get('constraint_batch',500000)
    jobs=[(i,min(i+batch,last+1)) for i in range(first,last+1,batch)]
    parallel_stage(config,'constraints',jobs,constraint_job)
    return catalog


def run(config):
    root=Path(config['output_root']);root.mkdir(parents=True,exist_ok=True)
    if config.get('experiment_link'):
        alias=Path(config['experiment_link']);alias.parent.mkdir(parents=True,exist_ok=True)
        if alias.exists() and alias.resolve()!=root.resolve():raise ValueError('Experiment link points elsewhere')
        if not alias.exists():alias.symlink_to(root,target_is_directory=True)
    # Stat identities avoid hashing the 132GB immutable AFC on every resume.
    signature=dict(packed_geometry_format=PACKED_GEOMETRY_FORMAT,config=config,inputs={k:dict(path=str(Path(config[k]).resolve()),size=Path(config[k]).stat().st_size,
        mtime_ns=Path(config[k]).stat().st_mtime_ns) for k in ('comparison','normal_solution','umbilicus')})
    signature['stores']={k:store_digest(config[k]) for k in ('fibers','database')}
    manifest=root/'request.json'
    if manifest.exists() and json.loads(manifest.read_text())!=signature:raise ValueError('Full-flow inputs/configuration changed')
    write(manifest,signature)
    if (root/'summary.json').exists():return json.loads((root/'summary.json').read_text())
    started=time.monotonic();catalog=prepare(config)
    merged_path=root/'graph.npz'
    if merged_path.exists():g=arrays(merged_path)
    else:
        parts=[arrays(p) for p in sorted((root/'graph_parts').glob('*.npz')) if '.partial.' not in p.name]
        print(json.dumps(dict(stage='merge',parts=len(parts),input_arcs=sum(len(p['a']) for p in parts))),flush=True)
        combined={k:np.concatenate([p[k] for p in parts]) for k in ('a','b','target','weight')};del parts
        g=aggregate_l1_edges(**combined);del combined
        np.savez(merged_path,**g)
    if config.get('prepare_only',False):
        summary=dict(fibers=len(catalog['source_id']),arcs=len(g['a']),
                     seconds=time.monotonic()-started,gt_used=False,prepared_only=True)
        write(root/'summary.json',summary)
        write(root/'progress.json',dict(stage='prepared',**summary))
        print(json.dumps(summary),flush=True)
        return summary
    print(json.dumps(dict(stage='solve',fibers=len(catalog['source_id']),arcs=len(g['a']),
                          preparation_seconds=time.monotonic()-started)),flush=True)
    write(root/'progress.json',dict(stage='solve',fibers=len(catalog['source_id']),arcs=len(g['a'])))
    # The solver's own progress log is retained, including native HiGHS output.
    log=open(root/'solver.log','a',buffering=1);saved_fd=os.dup(1)
    try:
        os.dup2(log.fileno(),1)
        result=solve(len(catalog['source_id']),**g,log_progress=True)
    finally:
        os.dup2(saved_fd,1);os.close(saved_fd);log.close()
    np.savez(root/'solution.npz',source_id=catalog['source_id'],family=catalog['family'],
             **{k:v for k,v in result.items() if k!='certificate'})
    write_winding_result(root/'solution.vcobj',config['fibers'],catalog['source_id'],result['winding'],
                         method='min_cost_flow',family=catalog['family'],component=result['component'])
    summary=dict(fibers=len(catalog['source_id']),arcs=len(g['a']),certificate=result['certificate'],
                 seconds=time.monotonic()-started,gt_used=False)
    write(root/'summary.json',summary);write(root/'progress.json',dict(stage='complete',**summary))
    print(json.dumps(summary),flush=True);return summary
