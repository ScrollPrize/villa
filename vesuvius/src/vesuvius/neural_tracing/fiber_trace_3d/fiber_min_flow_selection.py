"""GT-assisted whole-fiber neighborhood selection with the shared flow solver."""
import json
from pathlib import Path
import time

import numpy as np

from .fiber_min_flow_experiment import arrays, write
from vesuvius.neural_tracing.tracers.shared.annotations.fiber_store import (
    read_polylines, write_initial_fibers, fiber_catalog, fiber_volume, FiberBlockIndex)
from .fiber_graph_store import FiberGraph, workflow_volume


def prepare(config):
    root = Path(config['output_root']); root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    comparison = arrays(config['comparison'])
    ids, labels = comparison['ids'], comparison['labels']
    gtroot = Path(config['gt_root'])
    gcfg = json.loads((gtroot/'uv_whole/config.json').read_text())
    gd = arrays(gcfg['tensor_cache'])
    gt = read_polylines(gcfg['fibers'], gd['source_id'])
    volume = fiber_volume(config['fibers'])
    if workflow_volume(config)!=volume:
        raise ValueError('Neighborhood selection requires fibers imported in the measurement volume; transform the source before selection')
    if volume != fiber_volume(gcfg['fibers']):
        raise ValueError('GT/source coordinate frames differ')
    catalog = fiber_catalog(config['fibers'], volume)
    slab = np.asarray([r['id'] for r in catalog if r['bbox'] is not None
                       and r['bbox'][5]>=config['z_range'][0] and r['bbox'][2]<=config['z_range'][1]], dtype='U36')
    positions = np.searchsorted(ids, slab)
    if np.any(positions >= len(ids)) or not np.array_equal(ids[positions], slab):
        raise ValueError('Comparison/source identity mismatch')
    eligible = slab[labels[positions] != 2]
    np.save(root/'eligible_ids.npy', eligible)
    cache = root/'seeds.json'
    if cache.exists():
        selected = json.loads(cache.read_text())['selected_ids']
    else:
        spacing = config.get('seed_spacing', 64.)
        radius = config.get('maximum_distance', 128.) + spacing/2
        index = FiberBlockIndex(config['fibers'], eligible, volume=volume)
        selected = set(); sample_count = 0
        for row, xyz in gt:
            arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]
            at = np.linspace(0, arc[-1], max(2, int(np.ceil(arc[-1]/spacing))+1))
            samples = np.column_stack([np.interp(at, arc, xyz[:, k]) for k in range(3)])
            samples = samples[(samples[:,2]>=config['z_range'][0]-spacing)&(samples[:,2]<=config['z_range'][1]+spacing)]
            selected.update(index.query(samples,radius))
            sample_count += len(samples)
            print(json.dumps(dict(stage='seed_queries',gt_fiber=row['id'],samples=sample_count,
                                  candidates=len(selected),seconds=time.monotonic()-started)),flush=True)
        selected = sorted(selected)
        write(cache,dict(selected_ids=selected,slab_fibers=len(slab),eligible_fibers=len(eligible),
                         sample_count=sample_count,seconds=time.monotonic()-started))
    print(f'Preparing {len(selected)} eligible GT-neighbor fibers', flush=True)
    curves = read_polylines(config['fibers'], selected)
    source = np.array([r['id'] for r, _ in curves], dtype='U36')
    lengths = np.array([np.linalg.norm(np.diff(x, axis=0), axis=1).sum() for _, x in curves])
    tensor = root/'seed_tensors.npz'
    if not tensor.exists():
        np.savez(tensor, source_id=source, family=labels[np.searchsorted(ids, source)],
                 start=np.zeros(len(source)), end=lengths)
    fibers = root/'seed.vcobj'
    if not fibers.exists(): write_initial_fibers(fibers, curves, volume=volume, packed=True)
    cfg = dict(fibers=str(fibers), tensor_cache=str(tensor), umbilicus=config['umbilicus'], h_convention='ccw')
    write(root/'seed_config.json', {'fiber_uv_layout': cfg})
    from .fiber_gt_constraint_score import run as cross_run
    cross = root/'cross'
    if config.get('frozen_gt_cross'):
        from lasagna.inference_provenance import sha256_file
        prior = Path(config['frozen_gt_cross'])
        provenance = json.loads((prior/'inputs.json').read_text())
        from vesuvius.neural_tracing.tracers.shared.annotations.fiber_store import fibers_digest
        if fibers_digest(gcfg['fibers']) != provenance['gt_fibers']:
            raise ValueError('Frozen GT fiber snapshot mismatch')
        for key, path in [('gt_normal',gcfg['stage1']),
                          ('gt_tensors',gcfg['tensor_cache']), ('gt_solution',gtroot/'uv_whole/solution.npz')]:
            if sha256_file(path) != provenance[key]: raise ValueError('Frozen GT provenance mismatch: '+key)
        cross.mkdir(exist_ok=True)
        target = cross/'frozen_gt_normals'
        if not target.exists(): target.symlink_to((prior/'frozen_gt_normals').resolve(), target_is_directory=True)
    if not (cross/'cross_events.json').exists():
        cross_run(dict(configs=[str(root/'seed_config.json')], gt_root=config['gt_root'],
            output_root=str(cross), extract_template=config['extract_template'], orient_template=config['orient_template'],
            threads=config.get('threads', 64), events_only=True, native_binary=config['native_binary']))
    result = dict(stage='prepared', seed_fibers=len(source), eligible_fibers=len(eligible), seconds=time.monotonic()-started)
    write(root/'preparation.json', result); print(json.dumps(result), flush=True)
    return result


class EdgeStore:
    """Vectorized incident-edge index, built once; each event is retained once."""
    def __init__(self, config, eligible, labels, normal, groups=None):
        self.graph = FiberGraph(config['database'])
        self.eligible = set(map(str, eligible))
        self.labels, self.normal, self.groups = labels, normal, groups
        self.read, self.parts = set(), []
        self.seen = np.zeros(len(self.graph.event_ids), bool)
        endpoints = np.r_[self.graph.i,self.graph.j]
        self.order = np.argsort(endpoints,kind='stable')
        self.offset = np.r_[0,np.cumsum(np.bincount(endpoints,minlength=len(self.graph.ids)))]

    def load(self, sources):
        from .fiber_uv_layout import signed_order
        pending = sorted(set(map(str,sources))-self.read)
        if not pending:return
        indices = self.graph.indices(pending)
        positions = [self.order[self.offset[i]:self.offset[i+1]] for i in indices]
        events = np.unique(np.concatenate(positions)%len(self.seen)) if len(self.seen) else np.empty(0,np.int64)
        events = events[~self.seen[events]]
        self.seen[events] = True
        g = self.graph
        accepted = []
        for e in events:
            a,b = g.fiber_a[e],g.fiber_b[e]
            if a not in self.eligible or b not in self.eligible:continue
            if self.groups is not None and self.groups[a]==self.groups[b]:continue
            if g.parallel[e] != (self.labels[a]==self.labels[b]):continue
            if g.valid_a[e] and g.valid_b[e] and g.valid_connector[e]:accepted.append(e)
        use = np.asarray(accepted,np.int64)
        if len(use):
            a,b=g.fiber_a[use],g.fiber_b[use]
            signs=signed_order(g.connector[use],
                g.normal_a[use]*np.asarray([self.normal[x] for x in a])[:,None],
                g.normal_b[use]*np.asarray([self.normal[x] for x in b])[:,None])
            part=np.empty((len(use),7),object)
            part[:,0]=a;part[:,1]=b;part[:,2]=g.arc_a[use];part[:,3]=g.arc_b[use]
            part[:,4]=g.distance[use]
            part[:,5]=signs;part[:,6]=g.parallel[use]
            self.parts.append(part)
        self.read.update(pending)

    def edges(self):
        return np.concatenate(self.parts) if self.parts else np.empty((0,7),object)


def slab_winding_range(paths, source, winding, z_range):
    """Full in-slab polyline range, including exact boundary intersections."""
    arc, xyz, turns = paths.paths[str(source)]
    inside = (xyz[:,2]>=z_range[0]) & (xyz[:,2]<=z_range[1])
    relative = paths.seam.relative(xyz[inside])
    values = [winding+turns[inside]+((np.arctan2(relative[:,1],relative[:,0])-paths.seam.angle)%(2*np.pi))/(2*np.pi)]
    dz = np.diff(xyz[:,2])
    for z in z_range:
        hit = np.flatnonzero(((xyz[:-1,2]<z)&(xyz[1:,2]>z)) | ((xyz[:-1,2]>z)&(xyz[1:,2]<z)))
        if len(hit):
            s = arc[hit]+(z-xyz[hit,2])/dz[hit]*np.diff(arc)[hit]
            p,t = paths.at_many(np.full(len(s),source),s)
            rel = paths.seam.relative(p)
            values.append(winding+t+((np.arctan2(rel[:,1],rel[:,0])-paths.seam.angle)%(2*np.pi))/(2*np.pi))
    values = np.concatenate(values)
    return (float(values.min()),float(values.max())) if len(values) else (np.inf,-np.inf)


def select_range(d, solution, paths, gt_ids, z_range, graph):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    gt = np.isin(d['source_id'],gt_ids)
    ranges = np.array([slab_winding_range(paths,s,k,z_range)
                      for s,k in zip(d['source_id'],solution['winding'])])
    h = ranges[gt & (d['family']==0)]
    if not len(h) or not np.isfinite(h).all(): raise ValueError('Missing GT H slab ranges')
    inside = np.zeros(len(gt),bool)
    for lo,hi in h: inside |= (ranges[:,1]>=lo)&(ranges[:,0]<=hi)
    inside |= gt
    a,b = graph['a'],graph['b']
    valid = inside[a]&inside[b]
    adjacency = coo_matrix((np.ones(valid.sum()),(a[valid],b[valid])),shape=(len(gt),len(gt))).tocsr()
    _,components = connected_components(adjacency,directed=False)
    keep = inside & np.isin(components,components[gt])
    return keep, ranges, h


def run(config):
    from .fiber_min_flow import solve
    from .fiber_min_flow_benchmark import Paths, graph
    from .fiber_normal_convention import inward_solution
    root = Path(config['output_root'])
    root.mkdir(parents=True,exist_ok=True)
    request={k:v for k,v in config.items() if k!='finalize_only'}
    request_path=root/'selection_request.json'
    if request_path.exists() and json.loads(request_path.read_text())!=request:
        raise ValueError('Selection configuration changed; use a new output_root')
    write(request_path,request)
    if (root/'summary.json').exists():
        completed=json.loads((root/'summary.json').read_text())
        if completed.get('final_selection_stable'):
            for name in ('selected.vcobj','selected_with_gt.vcobj','selected_tensors.npz','selected_solution.npz'):
                if not (root/name).exists(): raise ValueError('Completed selection missing '+name)
            print(json.dumps(dict(stage='already_complete',source_fibers=completed['source_fibers'],
                                  gt_fibers=completed['gt_fibers'],output_root=str(root))),flush=True)
            return completed
    prepare(config)
    started = time.monotonic()
    comparison = arrays(config['comparison'])
    labels = dict(zip(map(str,comparison['ids']),map(int,comparison['labels'])))
    normal = inward_solution(arrays(config['normal_solution']))
    signs = dict(zip(map(str,normal['ids']),normal['normal_sign']))
    eligible = np.load(root/'eligible_ids.npy')
    store = EdgeStore(config,eligible,labels,signs)
    gcfg = json.loads((Path(config['gt_root'])/'uv_whole/config.json').read_text())
    gd = arrays(gcfg['tensor_cache'])
    gt_curves = read_polylines(gcfg['fibers'],gd['source_id'])
    gt_map = {str(s):str(s) for s in gd['source_id']}
    if set(gt_map) & set(map(str,comparison['ids'])):
        raise ValueError('GT and source fibers must be distinct UUIDs')
    gt_ids = np.array(list(gt_map.values()),dtype='U36')
    curves = {gt_map[str(row['id'])]:({**row,'id':gt_map[str(row['id'])]},xyz) for row,xyz in gt_curves}
    for i,s in enumerate(gd['source_id']): labels[gt_map[str(s)]]=int(gd['family'][i])
    extras = []
    for j in range(len(gd['i'])):
        a,b = gd['i'][j],gd['j'][j]
        extras.append([gt_map[str(gd['source_id'][a])],gt_map[str(gd['source_id'][b])],
            gd['start'][a]+gd['ds_a'][j]*np.sign(gd['node_ds'][a,-1]),
            gd['start'][b]+gd['ds_b'][j]*np.sign(gd['node_ds'][b,-1]),
            gd['distance'][j],gd['sign'][j],gd['parallel'][j]])
    cross = json.loads((root/'cross/cross_events.json').read_text())['events']
    for e in cross:
        if e['class_matches_families']:
            extras.append([e['test_source'],gt_map[e['gt_source']],e['test_arc'],e['gt_arc'],
                           abs(e['signed_distance']),np.sign(e['signed_distance']),e['parallel']])
    extras = np.array(extras,object).reshape(-1,7)
    seed = np.array(json.loads((root/'seeds.json').read_text())['selected_ids'],dtype='U36')
    current = np.union1d(seed,gt_ids)
    cache = {}; history = []; previous = None
    write(root/'gt_id_mapping.json',gt_map)

    def fit(ids, name):
        phase = time.monotonic()
        folder = root/name; folder.mkdir(exist_ok=True)
        missing = [str(s) for s in ids if str(s) not in curves]
        for row,xyz in read_polylines(config['fibers'],missing): curves[str(row['id'])]=(row,xyz)
        edges = np.concatenate((store.edges(),extras))
        edges = edges[np.isin(edges[:,0],ids)&np.isin(edges[:,1],ids)]
        lengths = np.array([np.linalg.norm(np.diff(curves[str(s)][1],axis=0),axis=1).sum() for s in ids])
        d = dict(source_id=ids, family=np.array([labels[str(s)] for s in ids]),start=np.zeros(len(ids)),
            end=lengths,node_ds=np.column_stack((np.zeros(len(ids)),lengths)),links=np.empty((0,3)),
            i=np.searchsorted(ids,edges[:,0]),j=np.searchsorted(ids,edges[:,1]),
            ds_a=edges[:,2].astype(float),ds_b=edges[:,3].astype(float),distance=edges[:,4].astype(float),sign=edges[:,5].astype(float),parallel=edges[:,6].astype(bool))
        paths = Paths(config,d,curves=[curves[str(s)] for s in ids if str(s) not in cache],cache=cache)
        g,transport = graph(config,d,paths)
        print(json.dumps(dict(stage='solve_start',name=name,fibers=len(ids),events=len(edges),flow_edges=len(g['a']),
                              preparation_seconds=time.monotonic()-phase)),flush=True)
        checkpoint = folder/'solution.npz'
        if checkpoint.exists():
            saved=arrays(folder/'tensors.npz')
            if any(not np.array_equal(saved[k],v) for k,v in d.items()): raise ValueError('Resume graph differs: '+name)
            solution=arrays(checkpoint); solution['certificate']=json.loads((folder/'certificate.json').read_text())
        else:
            solution=solve(len(ids),g['a'],g['b'],g['target'],g['weight'],g['one_sided'])
            np.savez(folder/'tensors.npz',**d)
            np.savez(checkpoint,**{k:v for k,v in solution.items() if k!='certificate'})
            write(folder/'certificate.json',solution['certificate'])
        keep,ranges,h=select_range(d,solution,paths,gt_ids,config['z_range'],g)
        local=solution['winding'][d['j']]-solution['winding'][d['i']]+transport['fiber_b']-transport['fiber_a']-transport['connector']
        margin=((~d['parallel'])&(d['sign']*(d['family'][d['j']]-d['family'][d['i']])<0)).astype(int)
        known=d['sign']!=0
        violation=known&(d['sign']*local<margin)
        metrics={}
        gt_event=np.isin(ids[d['i']],gt_ids)|np.isin(ids[d['j']],gt_ids)
        gt_internal=np.isin(ids[d['i']],gt_ids)&np.isin(ids[d['j']],gt_ids)
        for group,mask in [('all',np.ones(len(edges),bool)),('gt_connected',gt_event),('gt_internal',gt_internal)]:
            for kind,m in [('parallel',d['parallel']),('perpendicular',~d['parallel'])]:
                use=mask&m&known
                metrics[group+'_'+kind]=dict(events=int(use.sum()),sign_violations=int((violation&use).sum()),
                    measured_target_l1=float(np.mean(abs(local[use]-transport['measured_target'][use]))) if use.any() else None)
        report=dict(name=name,fibers=len(ids),events=len(edges),retained=int(keep.sum()),
            gt_h_ranges=h.tolist(),metrics=metrics,certificate=solution['certificate'],seconds=time.monotonic()-phase)
        write(folder/'report.json',report)
        np.savez(folder/'selection.npz',keep=keep,ranges=ranges)
        print(json.dumps(report),flush=True)
        return ids[keep],report,d,solution

    final = None
    resume_final = config.get('finalize_only',False)
    if resume_final:
        old_summary=json.loads((root/'summary.json').read_text())
        history=json.loads((root/'history.json').read_text())
        selected=arrays(root/'selected_tensors.npz')['source_id']
        store.load(selected[~np.isin(selected,gt_ids)])
        reason=old_summary['termination']
    for iteration in range(0 if resume_final else config.get('max_rounds',10)):
        store.load(current[~np.isin(current,gt_ids)])
        if iteration:
            incident=store.edges()
            incident=incident[np.isin(incident[:,0],current)|np.isin(incident[:,1],current)]
            current=np.union1d(current,incident[:,:2].astype('U36').ravel())
            store.load(current[~np.isin(current,gt_ids)])
        accepted,report,_,_=fit(current,f'round_{iteration:02d}_expanded')
        # Every filtered membership is jointly re-solved before accepting it.
        selected,filtered,d,solution=fit(accepted,f'round_{iteration:02d}_filtered')
        history.append(dict(round=iteration,expanded=report,filtered=filtered,
                            next_members=len(selected),seconds=time.monotonic()-started))
        write(root/'history.json',history)
        final=(d,solution)
        if previous is not None and np.array_equal(selected,previous) and np.array_equal(selected,accepted):
            reason='stable_membership'; break
        previous=selected.copy(); current=selected
    else:
        if not resume_final: reason='round_limit'
    # Monotone final cleanup: no re-additions, so this terminates in <= node_count
    # rounds and the exported graph satisfies its OWN solved selection criterion.
    cleanup=[]
    while final is None or not np.array_equal(final[0]['source_id'],selected):
        selected,report,d,solution=fit(selected,f'final_cleanup_{len(cleanup):02d}')
        final=(d,solution); cleanup.append(report)
        write(root/'cleanup.json',cleanup)
    d,solution=final
    np.savez(root/'selected_tensors.npz',**d)
    np.savez(root/'selected_solution.npz',**{k:v for k,v in solution.items() if k!='certificate'})
    exported=[({**curves[str(s)][0],'family':'H' if labels[str(s)]==0 else 'V'},curves[str(s)][1]) for s in d['source_id']]
    ordinary=[(row,xyz) for row,xyz in exported if row['id'] not in gt_ids]
    for filename in ('selected.vcobj','selected_with_gt.vcobj'):
        path=root/filename
        if path.exists(): path.replace(root/(filename+'.before_final_cleanup'))
    volume=fiber_volume(config['fibers'])
    write_initial_fibers(root/'selected.vcobj',ordinary,volume=volume,packed=True)
    write_initial_fibers(root/'selected_with_gt.vcobj',exported,volume=volume,packed=True)
    result=dict(termination=reason,rounds=len(history),source_fibers=len(ordinary),gt_fibers=len(gt_ids),
                seconds=time.monotonic()-started,last=history[-1],cleanup=cleanup,
                final_selection_stable=True)
    if resume_final: result['initial_run_seconds']=old_summary['seconds']
    write(root/'summary.json',result)
    print(json.dumps(result),flush=True)
    return result
