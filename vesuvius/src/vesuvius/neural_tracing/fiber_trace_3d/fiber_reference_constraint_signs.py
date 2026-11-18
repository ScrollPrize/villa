"""Reference-anchored normal signs from the shared native cross constraints.

Reference gauges are frozen using the existing normal-curve exporter. The only
fit is an independent binary reduction per new fiber; no per-point SQL queries.
"""
import json
from pathlib import Path
import time
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
    return [dict(source_id=str(fid),normal_sign=int(signs[i]),agreement=float(agreement[i]),
                 disagreement_fraction=float(wrong[i]/count[i]) if count[i] else 0.,
                 constraints=int(count[i])) for i,fid in enumerate(ids)]


def run(config):
    from .fiber_curve_bank import CurveBank
    from .fiber_min_flow_full import PackedPaths
    from .fiber_native_workflow import run as native_run
    from vesuvius.neural_tracing.tracers.shared.annotations.fiber_store import read_polylines,write_initial_fibers,fiber_volume,FiberBlockIndex
    from .fiber_graph_store import FiberGraph
    started=time.monotonic();root=Path(config['output_root']);root.mkdir(parents=True,exist_ok=True)
    ref=config['reference'];validate_reference(ref,config['fibers'])
    request=root/'request.json'
    if request.exists() and json.loads(request.read_text())!=config:raise ValueError('Changed normal-transfer configuration')
    write(request,config)
    def native(name,cfg):
        path=root/(name+'_config.json');write(path,cfg)
        native_run(dict(output_root=str(root/'native'),native_binary=config['native_binary'],input_files=[str(path)],
                        commands=[dict(name=name,binary='vc_fiber_constraints',args=[str(path)])]))
    target=read_polylines(config['fibers']);ids=np.array([row['id'] for row,_ in target],dtype='U36')
    packed=PackedPaths(ref['geometry'])
    normal=inward_solution(arrays(ref['solution']))
    if not (root/'reference_ids.npy').exists():
        candidates=set();radius=float(ref.get('maximum_distance',128))
        # Reuse the same ephemeral block-bounds index as region selection.
        index=FiberBlockIndex(ref['fibers'],packed.ids)
        boxes=[]
        for _,xyz in target:
            for first in range(0,len(xyz)-1,256):
                block=xyz[first:first+257];boxes.append([block.min(0),block.max(0)])
        for lo,hi in boxes:
            candidates.update(index.query(np.asarray([(lo+hi)/2]), (hi-lo)/2+radius))
        selected=np.intersect1d(list(candidates),packed.ids)
        selected=np.intersect1d(selected,normal['ids'][~normal['broken']])
        if not len(selected):raise ValueError('No supported reference neighbors')
        np.save(root/'reference_ids.npy',selected)
        write(root/'selection.json',dict(blocks=len(boxes),candidates=len(candidates),selected=len(selected),seconds=time.monotonic()-started))
    selected=np.load(root/'reference_ids.npy');n=len(ids)
    print(f'cross-normal graph: {n} new fibers, {len(selected)} reference neighbors',flush=True)
    if not (root/'combined.vcobj').exists():
        if np.intersect1d(ids,selected).size:raise ValueError('Target/reference UUIDs overlap')
        rows=list(target)
        at=np.searchsorted(packed.ids,selected)
        for i,p in enumerate(at):
            lo,hi=packed.offset[p:p+2]
            rows.append((dict(id=str(selected[i]),name='',family='unknown'),packed.xyz[lo:hi]))
        write_initial_fibers(root/'combined.vcobj',rows,volume=fiber_volume(config['fibers']),packed=True)
    # Export through the original native sweep, including original constraint
    # endpoints; every frozen endpoint is validated before the cache is published.
    for name,fibers,database,subset in [('target',config['fibers'],config['database'],None),
                                     ('reference',ref['fibers'],ref['database'],selected.tolist())]:
        if (root/(name+'_normals')/'index.json').exists():continue
        cfg=json.loads(Path(config['orient_template']).read_text())
        cfg.update(fibers=str(fibers),orient_database=str(database),normal_curves_output=str(root/(name+'_normals')),
                   database=str(root/(name+'_unused.vcobj')),progress_file=str(root/(name+'_normal_progress.jsonl')),
                   output=str(root/(name+'_normals.json')),load_threads=16)
        if subset is not None:cfg['fiber_ids']=subset
        native(name+'_normals',cfg)
    if not (root/'cross.json').exists():
        cfg=json.loads(Path(config['extract_template']).read_text())
        cfg.update(fibers=str(root/'combined.vcobj'),database=str(root/'cross.vcobj'),output=str(root/'cross.json'),
                   cross_query_fiber_ids=ids.tolist(),load_threads=16,threads=64,progress_file=str(root/'cross_progress.jsonl'))
        native('cross',cfg)
    target_normal=dict(ids=ids,normal_sign=np.ones(n),broken=np.zeros(n,bool))
    newbank=CurveBank.from_normals(ids,root/'target_normals',target_normal)
    refbank=CurveBank.from_normals(selected,root/'reference_normals',normal)
    graph=FiberGraph(root/'cross.vcobj')
    reverse=np.isin(graph.fiber_b,ids)
    new_ids=np.where(reverse,graph.fiber_b,graph.fiber_a)
    old_ids=np.where(reverse,graph.fiber_a,graph.fiber_b)
    if not np.isin(new_ids,ids).all() or not np.isin(old_ids,selected).all():
        raise ValueError('Native cross graph contains a within-source event')
    keep=np.ones(len(new_ids),bool) if config.get('include_parallel',False) else ~graph.parallel
    new=np.searchsorted(ids,new_ids[keep]);old=np.searchsorted(selected,old_ids[keep])
    rows=np.column_stack((new,old,np.where(reverse,graph.arc_b,graph.arc_a)[keep],
                           np.where(reverse,graph.arc_a,graph.arc_b)[keep]))
    a,va=newbank.sample_normals(new,rows[:,2]);b,vb=refbank.sample_normals(old,rows[:,3])
    dot=np.sum(a*b,axis=1);valid=va&vb&(abs(dot)>=config.get('minimum_abs_dot',.5))
    np.savez(root/'evidence.npz',source_id=ids[new[valid]],reference_source=selected[old[valid]],
             new_arc=rows[valid,2],reference_arc=rows[valid,3],dot=dot[valid])
    records=reduce_evidence(ids,ids[new[valid]],dot[valid],config.get('minimum_agreement',.5))
    result=dict(fibers=records,reference=ref,method='native_cross_constraints',
                reference_fibers=len(selected),constraints=len(rows),valid_constraints=int(valid.sum()),seconds=time.monotonic()-started)
    write(root/'result.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ('fibers','reference')}),flush=True)
    return records
