"""Bounded parallel full-volume adapter for the shared certified flow solver.

Original AFV/AFC inputs stay read-only. Exact polyline geometry is packed once;
disjoint workers stream constraint-ID ranges and combine only identical L1 arcs.
"""
import concurrent.futures
import json
import multiprocessing
import os
from pathlib import Path
import sqlite3
import time

import numpy as np

from .fiber_min_flow import Seam,aggregate_l1_edges,solve
from .fiber_min_flow_benchmark import graph
from .fiber_min_flow_experiment import arrays,write
from .fiber_min_flow_selection import ro

KEY_DTYPE=np.dtype([('source','<i8'),('arc','<f8')])
_worker=None


class PackedPaths:
    """Same endpoint evaluation as Paths, without per-fiber query loops."""
    def __init__(self,root,data=None,mode='r'):
        self.root=Path(root);self.data=data
        self.catalog=arrays(self.root/'catalog.npz')
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
        query=np.empty(len(source),KEY_DTYPE);query['source']=source;query['arc']=values
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
    _worker=(config,paths,ro(config['afv'] if stage=='geometry' else config['database']))


def geometry_job(job):
    from vesuvius.neural_tracing.tracers.shared.annotations.afv import read_afv_polylines
    config,paths,db=_worker
    first,last=job;root=Path(config['output_root'])/'geometry';marker=root/f'block_{first:08d}.json'
    if marker.exists():return json.loads(marker.read_text())
    started=time.monotonic()
    curves=read_afv_polylines(config['afv'],paths.ids[first:last])
    for i,(row,xyz) in enumerate(curves,first):
        lo,hi=paths.offset[i:i+2]
        if len(xyz)!=hi-lo or row['id']!=paths.ids[i]:raise ValueError('AFV catalog geometry mismatch')
        paths.xyz[lo:hi]=xyz
        paths.keys['source'][lo:hi]=row['id']
        paths.keys['arc'][lo:hi]=np.r_[0,np.cumsum(np.linalg.norm(np.diff(xyz,axis=0),axis=1))]
        paths.turns[lo:hi]=paths.seam.polyline(xyz)
    result=dict(first=first,last=last,fibers=last-first,points=int(paths.offset[last]-paths.offset[first]),seconds=time.monotonic()-started)
    write(marker,result)
    return result


def event_batches(config,paths,db,first,last):
    """Shared bounded AFC decoding for flow and continuous winding adapters."""
    from .fiber_uv_layout import signed_order
    # SQLite parses normals in C. Cursor batches never retain Python objects for
    # the entire database. Disjoint primary-key ranges need no dedup set.
    normals=','.join(f"json_extract(normal_{side},'$.xyz[{axis}]')" for side in ('a','b') for axis in range(3))
    cursor=db.execute(f'''SELECT fiber_a,fiber_b,arc_a,arc_b,ax,ay,az,bx,by,bz,
        parallel_score,perpendicular_score,winding_distance,parallel_winding_distance,
        {normals},json_extract(normal_a,'$.valid'),json_extract(normal_b,'$.valid'),id
        FROM constraints WHERE id>=? AND id<? ORDER BY id''',(first,last))
    catalog=paths.catalog;ids=paths.ids
    while rows:=cursor.fetchmany(config.get('read_batch',32768)):
        values=np.asarray(rows,dtype=np.float64);read=len(values)
        a=np.searchsorted(ids,values[:,0]);b=np.searchsorted(ids,values[:,1])
        valid=(a<len(ids))&(b<len(ids));a=np.minimum(a,len(ids)-1);b=np.minimum(b,len(ids)-1)
        valid&=(ids[a]==values[:,0])&(ids[b]==values[:,1])&(values[:,20]==1)&(values[:,21]==1)
        par=values[:,10]>values[:,11]
        valid&=par==(catalog['family'][a]==catalog['family'][b])
        values=values[valid];a=a[valid];b=b[valid];par=par[valid]
        sign=signed_order(values[:,7:10]-values[:,4:7],
            values[:,14:17]*catalog['normal_sign'][a,None],values[:,17:20]*catalog['normal_sign'][b,None])
        d=dict(i=a,j=b,ds_a=values[:,2],ds_b=values[:,3],family=catalog['family'],
               parallel=par,sign=sign,distance=np.where(par,values[:,13],values[:,12]),
               event_ids=values[:,22].astype(np.int64))
        yield read,d


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
    with ro(config['afv']) as db:
        meta=json.loads(db.execute("SELECT value FROM metadata WHERE key='root'").fetchone()[0])
        rows=np.asarray(db.execute('SELECT id,point_count FROM fibers ORDER BY id').fetchall(),np.int64)
    p=np.searchsorted(rows[:,0],ids)
    if not np.array_equal(rows[p,0],ids):raise ValueError('AFV identity differs')
    offsets=np.r_[0,np.cumsum(rows[p,1])]
    catalog=dict(source_id=ids,family=family,normal_sign=sign,point_offset=offsets,axis=load_umbilicus_frame(config['umbilicus'],meta))
    if not (geo/'catalog.npz').exists():np.savez(geo/'catalog.npz',**catalog)
    else:
        saved=arrays(geo/'catalog.npz')
        if any(not np.array_equal(v,saved[k]) for k,v in catalog.items()):raise ValueError('Geometry catalog changed')
    for name,dtype,shape in [('keys',KEY_DTYPE,(int(offsets[-1]),)),('xyz',np.float64,(int(offsets[-1]),3)),('turns',np.int64,(int(offsets[-1]),))]:
        path=geo/(name+'.npy')
        if not path.exists():
            array=np.lib.format.open_memmap(path,mode='w+',dtype=dtype,shape=shape);del array
    print(json.dumps(dict(stage='catalog',fibers=len(ids),excluded=int((~keep).sum()),points=int(offsets[-1]),
                          geometry_bytes=int(offsets[-1])*48)),flush=True)
    batch=config.get('geometry_batch',4096)
    jobs=[(i,min(i+batch,len(ids))) for i in range(0,len(ids),batch)]
    parallel_stage(config,'geometry',jobs,geometry_job)
    with ro(config['database']) as db:first,last=db.execute('SELECT min(id),max(id) FROM constraints').fetchone()
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
    signature=dict(config=config,inputs={k:dict(path=str(Path(config[k]).resolve()),size=Path(config[k]).stat().st_size,
        mtime_ns=Path(config[k]).stat().st_mtime_ns) for k in ('afv','database','comparison','normal_solution','umbilicus')})
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
    summary=dict(fibers=len(catalog['source_id']),arcs=len(g['a']),certificate=result['certificate'],
                 seconds=time.monotonic()-started,gt_used=False)
    write(root/'summary.json',summary);write(root/'progress.json',dict(stage='complete',**summary))
    print(json.dumps(summary),flush=True);return summary
