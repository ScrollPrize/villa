"""Perpendicular-only induced event-degree core of an existing full flow graph."""
from collections import deque
import json
from pathlib import Path
import time
import concurrent.futures
import multiprocessing
import numpy as np
from scipy.sparse import coo_matrix

from .fiber_min_flow_experiment import arrays,write
from .fiber_lemon import run_full


def event_core(nodes,a,b,weight,minimum,per_event_weight=1.):
    """Aggregated fractional-target weights sum to original event counts.

    Multiple events between the same fibers count separately; expanded target
    arcs do not. Recursively prune to guarantee the minimum in the final graph.
    """
    if minimum<1 or minimum!=int(minimum) or per_event_weight<=0:
        raise ValueError('Positive integer minimum and positive event weight required')
    if np.any(a==b):raise ValueError('Perpendicular self-constraint')
    adj=coo_matrix((np.r_[weight,weight]/per_event_weight,
        (np.r_[a,b],np.r_[b,a])),shape=(nodes,nodes)).tocsr()
    integer=np.rint(adj.data)
    if not np.allclose(adj.data,integer,rtol=0,atol=1e-6) or np.any(integer<1):
        raise ValueError('Arc weights do not reconstruct integral original event counts')
    adj.data=integer.astype(np.int64)
    degree=np.asarray(adj.sum(axis=1)).ravel().astype(np.int64);initial=degree.copy()
    queue=deque(map(int,np.flatnonzero(degree<minimum)));queued=degree<minimum
    keep=np.ones(nodes,bool)
    while queue:
        node=queue.popleft();keep[node]=False
        lo,hi=adj.indptr[node:node+2]
        neighbors=adj.indices[lo:hi];counts=adj.data[lo:hi]
        active=keep[neighbors];neighbors=neighbors[active];counts=counts[active]
        degree[neighbors]-=counts
        new=neighbors[(degree[neighbors]<minimum)&~queued[neighbors]]
        queued[new]=True;queue.extend(map(int,new))
    degree[~keep]=0
    return keep,initial,degree


def run(config):
    source=Path(config['source_root']);root=Path(config['output_root']);root.mkdir(parents=True,exist_ok=True)
    if (root/'request.json').exists():raise ValueError('Choose a fresh perpendicular experiment root')
    started=time.monotonic()
    original_request=json.loads((source/'request.json').read_text())
    weight=float(original_request['config'].get('perpendicular_relative_weight',1.))
    write(root/'request.json',dict(config=config,source_request=original_request,
        graph=dict(size=(source/'graph.npz').stat().st_size,mtime_ns=(source/'graph.npz').stat().st_mtime_ns)))
    if config.get('experiment_link'):Path(config['experiment_link']).symlink_to(root,target_is_directory=True)
    write(root/'progress.json',dict(stage='load_graph'))
    g=arrays(source/'graph.npz');catalog=arrays(source/'geometry/catalog.npz');family=catalog['family']
    par=family[g['a']]==family[g['b']]
    g={k:v[~par] for k,v in g.items()};del par
    minimum_length=float(config.get('minimum_length_base',0.))
    if not np.isfinite(minimum_length) or minimum_length<0:raise ValueError('Invalid minimum fiber length')
    eligible=np.ones(len(family),bool)
    if minimum_length>0:
        keys=np.load(source/'geometry/keys.npy',mmap_mode='r')
        lengths=keys['arc'][catalog['point_offset'][1:]-1]
        eligible=lengths>minimum_length
        length_mask=eligible[g['a']]&eligible[g['b']]
        g={k:v[length_mask] for k,v in g.items()};del length_mask
    write(root/'progress.json',dict(stage='prune',perpendicular_arcs=len(g['a'])))
    minimum=config.get('minimum_perpendicular_constraints',2)
    keep,initial,degree=event_core(len(family),g['a'],g['b'],g['weight'],minimum,weight)
    if not keep.any():raise ValueError('No fibers survive perpendicular core pruning')
    valid=keep[g['a']]&keep[g['b']]
    mapping=np.full(len(keep),-1,np.int64);mapping[keep]=np.arange(keep.sum())
    filtered={k:v[valid] for k,v in g.items()}
    filtered['a']=mapping[filtered['a']];filtered['b']=mapping[filtered['b']]
    report=dict(original_fibers=len(keep),retained_fibers=int(keep.sum()),
        minimum_length_base=minimum_length,length_eligible=int(eligible.sum()),
        removed_by_length=int(np.sum(~eligible)),
        eligible_initial_below_minimum=int(np.sum(eligible&(initial<minimum))),
        retained_percent=float(100*keep.mean()),initial_below_minimum=int(np.sum(initial<minimum)),
        additionally_pruned=int(np.sum(~keep & (initial>=minimum))),
        perpendicular_events_before=int(initial.sum()//2),perpendicular_events_after=int(degree.sum()//2),
        arcs_before=len(g['a']),arcs_after=len(filtered['a']),minimum_retained_degree=int(degree[keep].min()),
        degree_quantiles=np.quantile(degree[keep],[.1,.5,.9,1]).tolist(),
        definition='original accepted perpendicular events; recursive minimum-degree induced core before capacity quantization',
        seconds=time.monotonic()-started)
    np.savez(root/'graph.npz',**filtered)
    geo=root/'geometry';geo.mkdir()
    np.savez(geo/'catalog.npz',source_id=catalog['source_id'][keep],family=family[keep])
    np.savez(root/'selection.npz',source_id=catalog['source_id'],keep=keep,initial_degree=initial,retained_degree=degree)
    write(root/'selection.json',report);print(json.dumps(report),flush=True)
    del g,filtered,valid,mapping,catalog
    result=run_full(dict(source_root=str(root),output_root=str(root/'solve'),
        scale=config.get('scale',1),reference_mean=config['reference_mean'],progress_interval=config.get('progress_interval',20)))
    write(root/'progress.json',dict(stage='complete',selection=report,solve=result))
    return dict(selection=report,solve=result)


def _trial(config):
    from threadpoolctl import threadpool_limits
    from .fiber_flow_reference_eval import rescore
    with threadpool_limits(1):
        result=run(config)
        if config.get('reference_evaluation'):
            evaluation=dict(config['reference_evaluation'],solution=str(Path(config['output_root'])/'solve/solution.npz'),
                            output_root=str(Path(config['output_root'])/'reference'))
            result['reference']=rescore(evaluation)
        return result


def run_length_sweep(config):
    root=Path(config['output_root']);root.mkdir(parents=True,exist_ok=True)
    if (root/'request.json').exists():raise ValueError('Choose a fresh sweep root')
    write(root/'request.json',config)
    if config.get('experiment_link'):Path(config['experiment_link']).symlink_to(root,target_is_directory=True)
    trials=[]
    for length in config['lengths_base']:
        trial=dict(config['common'],minimum_length_base=length,output_root=str(root/f'length_{int(length)}'))
        trials.append(trial)
    results={};started=time.monotonic()
    write(root/'progress.json',dict(stage='running',completed=[],lengths=config['lengths_base']))
    with concurrent.futures.ProcessPoolExecutor(max_workers=config.get('workers',4),
            mp_context=multiprocessing.get_context('spawn')) as pool:
        futures={pool.submit(_trial,t):t['minimum_length_base'] for t in trials}
        for future in concurrent.futures.as_completed(futures):
            length=futures[future]
            try:results[str(length)]=future.result()
            except Exception as error:
                results[str(length)]=dict(error=str(error))
            write(root/'results.json',results)
            write(root/'progress.json',dict(stage='running',completed=list(results),seconds=time.monotonic()-started))
    write(root/'progress.json',dict(stage='complete',completed=list(results),seconds=time.monotonic()-started,
                                  failed=[k for k,v in results.items() if 'error' in v]))
    return results
