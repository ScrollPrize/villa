"""One-time migration of prediction-shaped archives into geometric replay states."""
from __future__ import annotations
import hashlib
import heapq
import json
from pathlib import Path
import numpy as np
from vesuvius.neural_tracing.fiber_follow.data import (
    OnPolicyStates, fiber_manifest, label_state, training_state_allowed,
)
from vesuvius.neural_tracing.fiber_follow.geometry import interp_at


def state_identity(row, geometry_hash):
    h = hashlib.sha256(geometry_hash.encode())
    for key in ('t','reverse','pos','frame','hist','hmask'):
        h.update(np.asarray(row[key],dtype='<f8').tobytes())
    return h.hexdigest()


def import_states(paths, fibers, cfg, band, volume, *, limit=20000, seed=0):
    """Discard predictions; preserve exact state, original-fiber identity and source.

    Stable hash-priority sampling keeps at most ``limit`` unique eligible states.
    Larger crops, dense targets and observed history are checked before storage.
    No old archive is modified or accepted by the v11 loader without migration.
    """
    if limit < 1:
        raise ValueError('State limit must be positive')
    manifest = fiber_manifest(fibers)
    lookup = {(f['name'],f['geometry_hash']):i for i,f in enumerate(manifest)}
    heap,seen,sources = [],set(),[]
    counts = dict(read=0,duplicate=0,holdout=0,unknown_fiber=0,eligible=0)
    for source_id,path in enumerate(paths):
        path = Path(path)
        if path.is_dir():
            metadata = json.loads((path/'metadata.json').read_text())
            arrays = {k:np.load(path/(k+'.npy'),mmap_mode='r') for k in OnPolicyStates.FIELDS}
        else:
            with np.load(path,allow_pickle=False) as z:
                metadata = json.loads(str(z['__metadata__'].item()))
                arrays = {k:z[k] for k in OnPolicyStates.FIELDS}
        if metadata['version'] not in (3,4,5):
            raise ValueError(f'Unsupported state format: {metadata["version"]}')
        provenance = metadata.get('provenance',{})
        if provenance.get('volume',{}).get('grid_scale',8.) != volume.grid_scale:
            raise ValueError('Replay coordinate scale differs from destination')
        sources.append(dict(path=str(path.resolve()),metadata=metadata))
        for j in range(len(arrays['pos'])):
            counts['read'] += 1
            old_fi = int(arrays['fiber_idx'][j])
            old_identity = metadata['fibers'][old_fi]
            fi = lookup.get((old_identity['name'],old_identity['geometry_hash']))
            if fi is None:
                counts['unknown_fiber'] += 1
                continue
            f = fibers[fi]
            row = {k:arrays[k][j] for k in OnPolicyStates.FIELDS}
            if not np.isfinite(row['t']) or not 0 <= row['t'] <= f.length:
                raise ValueError('Invalid original-fiber progress')
            item = label_state(f,row['pos'],row['frame'],row['hist'],row['hmask'],cfg,
                               t=float(row['t']),reverse=bool(row['reverse']),offtrack=bool(row['offtrack']))
            if not training_state_allowed(item,cfg.crop,band):
                counts['holdout'] += 1
                continue
            identity = state_identity(row,old_identity['geometry_hash'])
            if identity in seen:
                counts['duplicate'] += 1
                continue
            seen.add(identity)
            counts['eligible'] += 1
            row['fiber_idx'] = fi
            row['drift'] = np.nan if row['offtrack'] else float(np.linalg.norm(interp_at(f.points,f.s,np.array([row['t']]))[0]-row['pos']))
            row['source_cache'],row['source_row'] = source_id,j
            priority = int(hashlib.sha256(f'{seed}:{identity}'.encode()).hexdigest(),16)
            entry = (-priority,identity,{k:np.array(v,copy=True) for k,v in row.items()})
            if len(heap)<limit:
                heapq.heappush(heap,entry)
            elif entry[:2]>heap[0][:2]:
                heapq.heapreplace(heap,entry)
    if not heap:
        raise ValueError('No unique eligible training states in sources')
    rows = [row for _,_,row in sorted(heap,key=lambda e:e[1])]
    from dataclasses import asdict
    provenance = dict(import_sources=sources,selection_seed=seed,counts=counts,
                      crop=asdict(cfg.crop),volume=volume.to_dict(),format='geometry_only',step=-1)
    result = OnPolicyStates(manifest=manifest,provenance=provenance,
                                **{k:np.asarray([r[k] for r in rows]) for k in OnPolicyStates.FIELDS},
                                **{k:np.asarray([r[k] for r in rows]) for k in OnPolicyStates.OPTIONAL if k in rows[0]})
    result.validate_fibers(fibers)
    return result
