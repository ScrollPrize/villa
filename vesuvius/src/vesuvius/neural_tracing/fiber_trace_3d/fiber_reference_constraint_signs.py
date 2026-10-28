"""Reference-anchored normal signs from the shared native cross constraints.

Reference gauges are frozen using the existing normal-curve exporter. The only
fit is an independent binary reduction per new fiber; no per-point SQL queries.
"""
import json
from pathlib import Path
import sqlite3
import time
import uuid
import numpy as np
from .fiber_min_flow_experiment import arrays, write
from .fiber_normal_convention import inward_solution
from .fiber_reference_signs import fit_sign, validate_reference


def reduce_evidence(ids, owner, dot, minimum_agreement=.5):
    index=np.searchsorted(ids,owner)
    if len(index) and (np.any(index>=len(ids)) or not np.array_equal(ids[index],owner)):
        raise ValueError('Unknown new fiber in normal evidence')
    total=np.bincount(index,weights=dot,minlength=len(ids))
    weight=np.bincount(index,weights=abs(dot),minlength=len(ids))
    count=np.bincount(index,minlength=len(ids))
    signs=np.sign(total).astype(np.int8)
    agreement=np.divide(abs(total),weight,out=np.zeros(len(ids)),where=weight>0)
    signs[agreement<minimum_agreement]=0
    wrong=np.bincount(index,weights=dot*np.sign(total[index])<0,minlength=len(ids))
    return [dict(source_id=int(fid),normal_sign=int(signs[i]),agreement=float(agreement[i]),
                 disagreement_fraction=float(wrong[i]/count[i]) if count[i] else 0.,
                 constraints=int(count[i])) for i,fid in enumerate(ids)]


def run(config):
    from .fiber_curve_bank import CurveBank
    from .fiber_min_flow_full import PackedPaths
    from .fiber_native_workflow import run as native_run
    from vesuvius.neural_tracing.tracers.shared.annotations.afv import read_afv_polylines, write_afv
    started=time.monotonic();root=Path(config['output_root']);root.mkdir(parents=True,exist_ok=True)
    ref=config['reference'];validate_reference(ref,config['afv'])
    request=root/'request.json'
    if request.exists() and json.loads(request.read_text())!=config:raise ValueError('Changed normal-transfer configuration')
    write(request,config)
    def native(name,cfg):
        path=root/(name+'_config.json');write(path,cfg)
        native_run(dict(output_root=str(root/'native'),native_binary=config['native_binary'],input_files=[str(path)],
                        commands=[dict(name=name,binary='vc_fiber_constraints',args=[str(path)])]))
    target=read_afv_polylines(config['afv']);ids=np.array([row['id'] for row,_ in target],np.int64)
    packed=PackedPaths(ref['geometry'])
    normal=inward_solution(arrays(ref['solution']))
    if not (root/'reference_ids.npy').exists():
        # Query the existing AFV spatial index per stored polyline block, not per
        # constraint point. Bounding boxes are conservative; native geometry
        # subsequently determines actual cross events. No nearest-fiber cap.
        with sqlite3.connect(Path(config['afv']).resolve().as_uri()+'?mode=ro',uri=True) as db:
            boxes=db.execute('SELECT min_x,max_x,min_y,max_y,min_z,max_z FROM block_bounds ORDER BY id').fetchall()
        candidates=set();radius=float(ref.get('maximum_distance',128))
        with sqlite3.connect(Path(ref['afv']).resolve().as_uri()+'?mode=ro',uri=True) as db:
            for k,bounds in enumerate(boxes):
                x0,x1,y0,y1,z0,z1=bounds
                rows=db.execute('''SELECT b.fiber_id FROM block_bounds r JOIN blocks b ON b.id=r.id
                    WHERE r.max_x>=? AND r.min_x<=? AND r.max_y>=? AND r.min_y<=?
                    AND r.max_z>=? AND r.min_z<=?''',(x0-radius,x1+radius,y0-radius,y1+radius,z0-radius,z1+radius))
                candidates.update(r[0] for r in rows)
                if k%1000==0:print(f'reference blocks {k}/{len(boxes)} candidates={len(candidates)}',flush=True)
        selected=np.intersect1d(list(candidates),packed.ids)
        selected=np.intersect1d(selected,normal['ids'][~normal['broken']])
        if not len(selected):raise ValueError('No supported reference neighbors')
        np.save(root/'reference_ids.npy',selected)
        write(root/'selection.json',dict(blocks=len(boxes),candidates=len(candidates),selected=len(selected),seconds=time.monotonic()-started))
    selected=np.load(root/'reference_ids.npy');n=len(ids)
    print(f'cross-normal graph: {n} new fibers, {len(selected)} reference neighbors',flush=True)
    if not (root/'combined.afv').exists():
        rows=[(dict(row,id=i+1),xyz) for i,(row,xyz) in enumerate(target)]
        at=np.searchsorted(packed.ids,selected)
        for i,p in enumerate(at):
            lo,hi=packed.offset[p:p+2]
            rows.append((dict(id=n+i+1,name=f'reference_{selected[i]}',family='unknown',annotation='{}'),packed.xyz[lo:hi]))
        write_afv(config['afv'],root/'combined.afv',rows,dict(uuid=str(uuid.uuid5(uuid.NAMESPACE_URL,str(root.resolve())))),keep_metadata=('frame','root'))
    # Export through the original native sweep, including original constraint
    # endpoints; every frozen endpoint is validated before the cache is published.
    for name,afv,database,subset in [('target',config['afv'],config['database'],None),
                                     ('reference',ref['afv'],ref['database'],selected.tolist())]:
        if (root/(name+'_normals')/'index.json').exists():continue
        cfg=json.loads(Path(config['orient_template']).read_text())
        cfg.update(afv=str(afv),orient_database=str(database),normal_curves_output=str(root/(name+'_normals')),
                   database=str(root/(name+'_unused.sqlite')),progress_file=str(root/(name+'_normal_progress.jsonl')),
                   output=str(root/(name+'_normals.json')),load_threads=16)
        if subset is not None:cfg['fiber_ids']=subset
        native(name+'_normals',cfg)
    if not (root/'cross.json').exists():
        cfg=json.loads(Path(config['extract_template']).read_text())
        cfg.update(afv=str(root/'combined.afv'),database=str(root/'cross.sqlite'),output=str(root/'cross.json'),
                   cross_query_prefix=n,load_threads=16,threads=64,progress_file=str(root/'cross_progress.jsonl'))
        native('cross',cfg)
    target_normal=dict(ids=ids,normal_sign=np.ones(n),broken=np.zeros(n,bool))
    newbank=CurveBank.from_normals(ids,root/'target_normals',target_normal)
    refbank=CurveBank.from_normals(selected,root/'reference_normals',normal)
    # One sequential read; source IDs were remapped with the new fibers first.
    with sqlite3.connect((root/'cross.sqlite').resolve().as_uri()+'?mode=ro',uri=True) as db:
        rows=np.asarray(db.execute('SELECT fiber_a,fiber_b,arc_a,arc_b,parallel_score,perpendicular_score FROM constraints ORDER BY id').fetchall(),float).reshape(-1,6)
    if len(rows) and (np.any(rows[:,0]>n) or np.any(rows[:,1]<=n)):
        raise ValueError('Native cross graph contains a within-source event')
    if not config.get('include_parallel',False):rows=rows[rows[:,5]>rows[:,4]]
    new=rows[:,0].astype(np.int64)-1; old=rows[:,1].astype(np.int64)-n-1
    a,va=newbank.sample_normals(new,rows[:,2]);b,vb=refbank.sample_normals(old,rows[:,3])
    dot=np.sum(a*b,axis=1);valid=va&vb&(abs(dot)>=config.get('minimum_abs_dot',.5))
    np.savez(root/'evidence.npz',source_id=ids[new[valid]],reference_source=selected[old[valid]],
             new_arc=rows[valid,2],reference_arc=rows[valid,3],dot=dot[valid])
    records=reduce_evidence(ids,ids[new[valid]],dot[valid],config.get('minimum_agreement',.5))
    result=dict(fibers=records,reference=ref,method='native_cross_constraints',
                reference_fibers=len(selected),constraints=len(rows),valid_constraints=int(valid.sum()),seconds=time.monotonic()-started)
    write(root/'result.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ('fibers','reference')}),flush=True)
    return records
