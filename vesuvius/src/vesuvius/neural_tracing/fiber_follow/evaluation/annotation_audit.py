"""Offline annotation-proximity evidence for long Paris traces; not an identity oracle."""
import argparse
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from .seeds import matched_profile
from .summarize_long_audit import load_rows
from ..shared.experiment import write_json as write
from ..data.data import load_fibers
from ..shared.geometry import arclength


OTHER_WINDING = -1  # owner id of the original annotation's other windings
OTHER_WINDING_ARC = 32.  # arclength from the matched point beyond which an own-annotation point is another winding


def main():
    ap=argparse.ArgumentParser();ap.add_argument('directory');args=ap.parse_args()
    root=Path(args.directory)
    fibers=load_fibers('/mnt/raid_nvme/spiral_dataset_working/fibers',grid_scale=8.)
    print('Building annotation index',len(fibers),flush=True)
    points=np.concatenate([f.points for f in fibers]);owners=np.concatenate([np.full(len(f.points),i,np.int32) for i,f in enumerate(fibers)])
    vertex=np.concatenate([np.arange(len(f.points)) for f in fibers])
    tree=cKDTree(points);name_to_id={f.name:i for i,f in enumerate(fibers)}
    rows=[]
    for row in load_rows(root/'primary'):
        if row['source']!='paris4':continue
        data=np.load(root/'primary'/row['source']/row['cohort']/f"trace_{row['seed_index']:03d}.npz")
        path=data['path'];arc=arclength(path) if len(path)>1 else np.zeros(1)
        own=name_to_id[row['fiber_name']];f=fibers[own]
        # Arclength-matched distance to the original annotation (as strict scoring), so its other windings are off it.
        own_distance,own_arc=matched_profile(path,arc,f,row['t0'],row['sign'])
        distance,indices=tree.query(path,k=8,workers=4)
        ids=owners[indices]
        # The original fiber's other windings (more than OTHER_WINDING_ARC from the matched arc) are alternatives too.
        other_winding=(ids==own)&(np.abs(f.s[np.where(ids==own,vertex[indices],0)]-own_arc[:,None])>OTHER_WINDING_ARC)
        ids=np.where(other_winding,OTHER_WINDING,ids)
        # Only assess points outside the original annotation tolerance and before censored continuation.
        valid=(ids!=own)&(distance<=1.5)
        nearest=np.argmin(np.where(valid,distance,np.inf),axis=1)
        candidate=ids[np.arange(len(path)),nearest]
        supported=valid.any(1)&(own_distance>3)&(arc<=row['local']['local_scored_length'])
        begin=None;owner=None;events=[]
        for i in range(len(path)+1):
            now=int(candidate[i]) if i<len(path) and supported[i] else None
            if now!=owner:
                if owner is not None and arc[i-1]-arc[begin]>=32:
                    events.append(dict(start=float(arc[begin]),length=float(arc[i-1]-arc[begin]),
                                       neighbor=f.name+' (other winding)' if owner==OTHER_WINDING else fibers[owner].name))
                begin=i;owner=now
        failure_arc=row['t0']+row['sign']*row['followed']
        nearby_spans=[dict(start=span.start,end=span.end,mode=span.provenance.interp_mode) for span in f.spans if span.start-32<=failure_arc<=span.end+32]
        rows.append(dict(cohort=row['cohort'],seed_index=row['seed_index'],fiber_name=f.name,diverged=row['diverged'],
                         sustained_foreign_annotation_proximity=events,near_failure_spans=nearby_spans,
                         nearest_foldback_to_failure=min((abs(x-failure_arc) for x in f.foldbacks),default=None)))
    write(root/'annotation_proximity.json',dict(definition='At least 32 travel voxels within 1.5 of a single other annotation (or of the original annotation more than 32 voxels of arclength from the matched point: another winding), while >3 from the arclength-matched original, before locally censored endpoint. Proximity is evidence to review, not independently confirmed fiber identity.',
          index_fibers=len(fibers),rows=rows))
    print('Audited',len(rows),'traces; sustained alternative annotation proximity',sum(bool(r['sustained_foreign_annotation_proximity']) for r in rows),flush=True)


if __name__=='__main__':main()
