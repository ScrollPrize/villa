"""Scan crossing records around intact weakly connected patches with new fibers.

This is an enriched experiment, not a population-uniform sample. Geometry and
theta admission thresholds are unchanged. Source patches are never trimmed.
"""
import argparse
from collections import defaultdict
import itertools
import json
from pathlib import Path
import time

import numpy as np

from grow_track_graph import PackedTracks,CrossingCsr
from point_collection import _build_surface_patch_index
from spiral_helpers import load_patch_payload_chunk
from tifxyz import patch_from_payload


def cell_key(q):
    q=np.asarray(q,dtype=np.int64)
    return (q[...,2]*512+q[...,1])*512+q[...,0]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('outputs/winding-normal-neighborhood-10vox'))
    p.add_argument('--catalog',type=Path,default=Path('outputs/track-frontier/patch-catalog.json'))
    p.add_argument('--output',type=Path,default=Path('outputs/track-frontier/fiber-target-selection.json'))
    p.add_argument('--per-patch',type=int,default=20)
    p.add_argument('--seed',type=int,default=17092033)
    p.add_argument('--preflight',type=Path,help='Select around geometry-compatible fiber/patch candidates instead of only new-fiber targets')
    args=p.parse_args();started=time.monotonic();rng=np.random.default_rng(args.seed)
    raw=json.loads((args.root/'config.json').read_text());saved=json.loads((args.root/'pilot.json').read_text())
    catalog=json.loads(args.catalog.read_text());by_pid={r['patch']:r for r in catalog['patches']}
    weak={r['node'] for r in saved['pending'] if r['node'].startswith('patch:') and r['reason']=='insufficient_support'}
    target_nodes={e['b'] for e in saved['evidence'] if e['kind']=='contact' and e['b'] in weak and e['a'] not in saved['offsets']}
    if args.preflight:
        target_nodes={r['node'] for r in json.loads(args.preflight.read_text()) if r['decision']=='needs_track_witness'}
    pids=sorted({n[6:].rsplit(':',1)[0] for n in target_nodes})
    loaded=load_patch_payload_chunk(raw['patches_dir'],pids,7000,17500,saved['patch_preprocessing']['erode_cells_default'],io_threads=4,keep_largest_component=True)
    patches={pid:patch_from_payload(payload) for pid,payload,error,drop in loaded if payload is not None}
    if len(patches)!=len(pids):raise ValueError('Target geometry load failure')
    index,surface_ids=_build_surface_patch_index(patches,4.)
    cells=set()
    for pid in pids:
        lo=np.floor((np.array(by_pid[pid]['low'])-4.)/64).astype(int)
        hi=np.floor((np.array(by_pid[pid]['high'])+4.)/64).astype(int)
        if np.any(lo<0) or np.any(hi>=512):raise ValueError('Unexpected coordinate domain for spatial cells')
        cells.update(int(cell_key(q)) for q in itertools.product(*(range(a,b+1) for a,b in zip(lo,hi))))
    cells=np.array(sorted(cells),dtype=np.int64)
    tracks=PackedTracks('/home/sean/Desktop/spiral_dataset/tracks/2um_ds2_ps256_surf_v2.dbm.vctracks')
    csr=CrossingCsr('/home/sean/Desktop/spiral_dataset/tracks/2um_ds2_ps256_surf_v2.dbm.crossings.npz',tracks)
    selected=defaultdict(dict);prescreen=0;exact=0;eligible_count=0
    print(f'Scanning all crossings around {len(pids)} weak patches with unaccepted fiber contacts',flush=True)
    for first in range(0,tracks.track_count,100000):
        end=min(first+100000,tracks.track_count)
        begin,finish=int(csr.offsets[first]),int(csr.offsets[end])
        a=np.repeat(np.arange(first,end),np.diff(csr.offsets[first:end+1]))
        b=csr.partners[begin:finish];ia=csr.self_local[begin:finish];ib=csr.partner_local[begin:finish]
        lengtha=tracks.offsets[a+1]-tracks.offsets[a];lengthb=tracks.offsets[b+1]-tracks.offsets[b]
        longa=(ia>=10)&(lengtha-1-ia>=10);longb=(ib>=10)&(lengthb-1-ib>=10)
        keep=(a<b)&(longa|longb)&(ia>=3)&(ib>=3)&(lengtha-1-ia>=3)&(lengthb-1-ib>=3)
        a,b,ia,ib=[v[keep] for v in (a,b,ia,ib)]
        xyz=np.asarray(tracks.coordinates[tracks.offsets[a]+ia,::-1],float)
        q=cell_key(np.floor(xyz/64).astype(int));location=np.searchsorted(cells,q)
        near=(location<len(cells))&(cells[np.minimum(location,len(cells)-1)]==q)&(xyz[:,2]>=7200)&(xyz[:,2]<17300)
        a,b,ia,ib,xyz=[v[near] for v in (a,b,ia,ib,xyz)];prescreen+=len(xyz);eligible_count+=int(keep.sum())
        for start in range(0,len(xyz),10000):
            points=xyz[start:start+10000]
            offsets,surfaces,distances,ijs=index.locate_all_xyz_batch(np.ascontiguousarray(points),4.)
            rows=np.repeat(np.arange(len(points)),np.diff(offsets))+start
            priorities=rng.random(len(xyz[start:start+10000]))
            exact+=len(rows)
            for row,surface in zip(rows,surfaces):
                pid=surface_ids[int(surface)];cell=tuple(np.floor(xyz[row]/30).astype(int))
                score=float(priorities[row-start]);prior=selected[pid].get(cell)
                if prior is None or score<prior['priority']:
                    selected[pid][cell]=dict(priority=score,key=[int(a[row]),int(b[row]),int(ia[row]),int(ib[row])],center_xyz=xyz[row].tolist())
        if first%2000000==0:print(f'{end}/{tracks.track_count} tracks; {prescreen} prescreen crossings; {exact} target hits; {time.monotonic()-started:.1f}s',flush=True)
    pairs={};per_patch={}
    for pid in pids:
        chosen=[]
        for row in sorted(selected[pid].values(),key=lambda r:r['priority']):
            point=np.array(row['center_xyz'])
            if any(np.linalg.norm(point-np.array(r['center_xyz']))<30 for r in chosen):continue
            chosen.append(row)
            if len(chosen)>=args.per_patch:break
        per_patch[pid]=dict(spatial_cells=len(selected[pid]),selected=len(chosen))
        for row in chosen:
            key=tuple(row['key'])
            if key not in pairs:pairs[key]=dict(key=row['key'],center_xyz=row['center_xyz'],target_patches=[])
            pairs[key]['target_patches'].append(pid)
    output=dict(stats=dict(directed_records=len(csr.partners),canonical_arm_eligible=eligible_count,
        spatial_prescreen=prescreen,exact_target_hits=exact,target_patches=len(pids),targets_with_selected_crossings=sum(r['selected']>0 for r in per_patch.values()),
        selected_locations=len(pairs),elapsed_seconds=time.monotonic()-started),per_patch=per_patch,pairs=list(pairs.values()),
        settings=dict(seed=args.seed,per_patch=args.per_patch,preflight=str(args.preflight) if args.preflight else None),
        selection='enriched around intact fiber-connected patches; geometry-only selection')
    args.output.write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output['stats'],indent=2),flush=True)


if __name__=='__main__':main()
