"""GT-assisted whole-fiber neighborhood selection with the shared flow solver."""
import json
from pathlib import Path
import sqlite3
import time

import numpy as np

from .fiber_min_flow_experiment import arrays, write
from vesuvius.neural_tracing.tracers.shared.annotations.afv import read_afv_polylines, write_afv


def ro(path):
    return sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)


def prepare(config):
    root = Path(config['output_root']); root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    comparison = arrays(config['comparison'])
    ids, labels = comparison['ids'], comparison['labels']
    gtroot = Path(config['gt_root'])
    gcfg = json.loads((gtroot/'uv_whole/config.json').read_text())
    gd = arrays(gcfg['tensor_cache'])
    gt = read_afv_polylines(gcfg['afv'], gd['source_id'])
    with ro(config['afv']) as db, ro(gcfg['afv']) as gdb:
        frame = json.loads(db.execute("SELECT value FROM metadata WHERE key='frame'").fetchone()[0])
        gframe = json.loads(gdb.execute("SELECT value FROM metadata WHERE key='frame'").fetchone()[0])
        if frame != gframe: raise ValueError('GT/source coordinate frames differ')
        slab = np.asarray([r[0] for r in db.execute(
            'SELECT id FROM fibers WHERE max_z>=? AND min_z<=? ORDER BY id', config['z_range'])], dtype=np.int64)
        positions = np.searchsorted(ids, slab)
        if not np.array_equal(ids[positions], slab): raise ValueError('Comparison/source identity mismatch')
        eligible = slab[labels[positions] != 2]
        np.save(root/'eligible_ids.npy', eligible)
        cache = root/'seeds.json'
        if cache.exists():
            selected = json.loads(cache.read_text())['selected_ids']
        else:
            spacing = config.get('seed_spacing', 64.)
            radius = config.get('maximum_distance', 128.) + spacing/2
            selected = set(); sample_count = 0
            for row, xyz in gt:
                # Original densely sampled vertices need not all be queried: use
                # uniform arc samples plus endpoints, preserving the distance pad.
                arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]
                at = np.linspace(0, arc[-1], max(2, int(np.ceil(arc[-1]/spacing))+1))
                samples = np.column_stack([np.interp(at, arc, xyz[:, a]) for a in range(3)])
                samples = samples[(samples[:, 2] >= config['z_range'][0]-spacing) & (samples[:, 2] <= config['z_range'][1]+spacing)]
                for point in samples:
                    lo, hi = point-radius, point+radius
                    rows = db.execute('''SELECT DISTINCT b.fiber_id FROM block_bounds r JOIN blocks b ON b.id=r.id
                        WHERE r.max_x>=? AND r.min_x<=? AND r.max_y>=? AND r.min_y<=?
                        AND r.max_z>=? AND r.min_z<=?''', tuple(np.column_stack((lo, hi)).ravel().tolist()))
                    selected.update(r[0] for r in rows)
                sample_count += len(samples)
                print(json.dumps(dict(stage='seed_queries', gt_fiber=int(row['id']), samples=sample_count,
                                      candidates=len(selected), seconds=time.monotonic()-started)), flush=True)
            selected = np.intersect1d(list(selected), eligible).tolist()
            write(cache, dict(selected_ids=selected, slab_fibers=len(slab), eligible_fibers=len(eligible),
                              sample_count=sample_count, seconds=time.monotonic()-started))
    print(f'Preparing {len(selected)} eligible GT-neighbor fibers', flush=True)
    curves = read_afv_polylines(config['afv'], selected)
    source = np.array([r['id'] for r, _ in curves], dtype=np.int64)
    lengths = np.array([np.linalg.norm(np.diff(x, axis=0), axis=1).sum() for _, x in curves])
    tensor = root/'seed_tensors.npz'
    if not tensor.exists():
        np.savez(tensor, source_id=source, family=labels[np.searchsorted(ids, source)],
                 start=np.zeros(len(source)), end=lengths)
    afv = root/'seed.afv'
    if not afv.exists(): write_afv(config['afv'], afv, curves, dict(source='GT-neighbor spatial seed'))
    cfg = dict(afv=str(afv), tensor_cache=str(tensor), umbilicus=config['umbilicus'], h_convention='ccw')
    write(root/'seed_config.json', {'fiber_uv_layout': cfg})
    from .fiber_gt_constraint_score import run as cross_run
    cross = root/'cross'
    if config.get('frozen_gt_cross'):
        from lasagna.inference_provenance import sha256_file
        prior = Path(config['frozen_gt_cross'])
        provenance = json.loads((prior/'inputs.json').read_text())
        for key, path in [('gt_afv',gcfg['afv']), ('gt_normal',gcfg['stage1']),
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
    """Indexed incident-edge reads; each source is read once, each event kept once."""
    def __init__(self, config, eligible, labels, normal, groups=None):
        self.db = ro(config['database'])
        self.eligible = set(map(int, eligible))
        self.labels = labels
        self.normal = normal
        self.groups = groups
        self.read = set()
        self.seen = set()
        self.parts = []

    def load(self, sources):
        from .fiber_uv_layout import signed_order
        pending = sorted(set(map(int, sources))-self.read)
        started = time.monotonic()
        for offset in range(0, len(pending), 128):
            batch = pending[offset:offset+128]
            placeholders = ','.join('?' for _ in batch)
            collected = []
            for side in ('a', 'b'):
                rows = self.db.execute(f'''SELECT id,fiber_a,fiber_b,arc_a,arc_b,
                    ax,ay,az,bx,by,bz,parallel_score,perpendicular_score,
                    winding_distance,parallel_winding_distance,normal_a,normal_b
                    FROM constraints WHERE fiber_{side} IN ({placeholders})''', batch)
                for row in rows:
                    eid,a,b = row[:3]
                    if eid in self.seen: continue
                    self.seen.add(eid)
                    if a not in self.eligible or b not in self.eligible: continue
                    if self.groups is not None and self.groups[a] == self.groups[b]: continue
                    parallel = row[11] > row[12]
                    if parallel != (self.labels[a] == self.labels[b]): continue
                    na,nb = json.loads(row[15]),json.loads(row[16])
                    if not na['valid'] or not nb['valid']: continue
                    collected.append((row,parallel,np.asarray(na['xyz'])*self.normal[a],
                                      np.asarray(nb['xyz'])*self.normal[b]))
            if collected:
                numeric = np.array([r[:15] for r,_,_,_ in collected], float)
                signs = signed_order(numeric[:,8:11]-numeric[:,5:8],
                    np.array([x[2] for x in collected]),np.array([x[3] for x in collected]))
                par = np.array([x[1] for x in collected])
                self.parts.append(np.column_stack((numeric[:,1:5],
                    np.where(par,numeric[:,14],numeric[:,13]),signs,par)))
            self.read.update(batch)
            if offset % 1024 == 0 or offset+128 >= len(pending):
                print(json.dumps(dict(stage='read_constraints', sources_done=min(offset+128,len(pending)),
                    sources_total=len(pending), unique_events=len(self.seen), seconds=time.monotonic()-started)),flush=True)

    def edges(self):
        return np.concatenate(self.parts) if self.parts else np.empty((0,7))


def slab_winding_range(paths, source, winding, z_range):
    """Full in-slab polyline range, including exact boundary intersections."""
    arc, xyz, turns = paths.paths[int(source)]
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
            for name in ('selected.afv','selected_with_gt.afv','selected_tensors.npz','selected_solution.npz'):
                if not (root/name).exists(): raise ValueError('Completed selection missing '+name)
            print(json.dumps(dict(stage='already_complete',source_fibers=completed['source_fibers'],
                                  gt_fibers=completed['gt_fibers'],output_root=str(root))),flush=True)
            return completed
    prepare(config)
    started = time.monotonic()
    comparison = arrays(config['comparison'])
    labels = dict(zip(map(int,comparison['ids']),map(int,comparison['labels'])))
    normal = inward_solution(arrays(config['normal_solution']))
    signs = dict(zip(map(int,normal['ids']),normal['normal_sign']))
    eligible = np.load(root/'eligible_ids.npy')
    store = EdgeStore(config,eligible,labels,signs)
    gcfg = json.loads((Path(config['gt_root'])/'uv_whole/config.json').read_text())
    gd = arrays(gcfg['tensor_cache'])
    gt_curves = read_afv_polylines(gcfg['afv'],gd['source_id'])
    gt_map = {int(s):int(comparison['ids'].max())+1+i for i,s in enumerate(gd['source_id'])}
    gt_ids = np.array(list(gt_map.values()),np.int64)
    curves = {gt_map[int(row['id'])]:({**row,'id':gt_map[int(row['id'])]},xyz) for row,xyz in gt_curves}
    for i,s in enumerate(gd['source_id']): labels[gt_map[int(s)]]=int(gd['family'][i])
    extras = []
    for j in range(len(gd['i'])):
        a,b = gd['i'][j],gd['j'][j]
        extras.append([gt_map[int(gd['source_id'][a])],gt_map[int(gd['source_id'][b])],
            gd['start'][a]+gd['ds_a'][j]*np.sign(gd['node_ds'][a,-1]),
            gd['start'][b]+gd['ds_b'][j]*np.sign(gd['node_ds'][b,-1]),
            gd['distance'][j],gd['sign'][j],gd['parallel'][j]])
    cross = json.loads((root/'cross/cross_events.json').read_text())['events']
    for e in cross:
        if e['class_matches_families']:
            extras.append([e['test_source'],gt_map[e['gt_source']],e['test_arc'],e['gt_arc'],
                           abs(e['signed_distance']),np.sign(e['signed_distance']),e['parallel']])
    extras = np.array(extras,float)
    seed = np.array(json.loads((root/'seeds.json').read_text())['selected_ids'],np.int64)
    current = np.union1d(seed,gt_ids)
    cache = {}; history = []; previous = None
    write(root/'gt_id_mapping.json',gt_map)

    def fit(ids, name):
        phase = time.monotonic()
        folder = root/name; folder.mkdir(exist_ok=True)
        missing = [int(s) for s in ids if int(s) not in curves]
        for row,xyz in read_afv_polylines(config['afv'],missing): curves[int(row['id'])]=(row,xyz)
        edges = np.concatenate((store.edges(),extras))
        edges = edges[np.isin(edges[:,0],ids)&np.isin(edges[:,1],ids)]
        lengths = np.array([np.linalg.norm(np.diff(curves[int(s)][1],axis=0),axis=1).sum() for s in ids])
        d = dict(source_id=ids, family=np.array([labels[int(s)] for s in ids]),start=np.zeros(len(ids)),
            end=lengths,node_ds=np.column_stack((np.zeros(len(ids)),lengths)),links=np.empty((0,3)),
            i=np.searchsorted(ids,edges[:,0]),j=np.searchsorted(ids,edges[:,1]),
            ds_a=edges[:,2],ds_b=edges[:,3],distance=edges[:,4],sign=edges[:,5],parallel=edges[:,6].astype(bool))
        paths = Paths(config,d,curves=[curves[int(s)] for s in ids if int(s) not in cache],cache=cache)
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
            current=np.union1d(current,incident[:,:2].astype(np.int64).ravel())
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
    exported=[({**curves[int(s)][0],'family':'H' if labels[int(s)]==0 else 'V'},curves[int(s)][1]) for s in d['source_id']]
    ordinary=[(row,xyz) for row,xyz in exported if row['id'] not in gt_ids]
    for filename in ('selected.afv','selected_with_gt.afv'):
        path=root/filename
        if path.exists(): path.replace(root/(filename+'.before_final_cleanup'))
    write_afv(config['afv'],root/'selected.afv',ordinary,dict(selection='GT-assisted min-cost-flow; whole original fibers'))
    write_afv(config['afv'],root/'selected_with_gt.afv',exported,
              dict(selection='GT-assisted min-cost-flow; remapped GT IDs in gt_id_mapping.json'))
    result=dict(termination=reason,rounds=len(history),source_fibers=len(ordinary),gt_fibers=len(gt_ids),
                seconds=time.monotonic()-started,last=history[-1],cleanup=cleanup,
                final_selection_stable=True)
    if resume_final: result['initial_run_seconds']=old_summary['seconds']
    write(root/'summary.json',result)
    store.db.close()
    print(json.dumps(result),flush=True)
    return result
