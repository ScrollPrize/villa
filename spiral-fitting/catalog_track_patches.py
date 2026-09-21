"""Read original patch geometry to build a conservative spatial search catalog.

Uses precisely the saved growth preprocessing, without candidate-dependent
cropping. Bounding boxes come from retained source quads, never metadata bbox
conventions. The catalog is an experiment artifact, not a modified dataset.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from spiral_helpers import load_patch_payload_chunk


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('outputs/winding-normal-neighborhood-10vox'))
    p.add_argument('--output',type=Path,default=Path('outputs/track-frontier/patch-catalog.json'))
    args=p.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
    config=json.loads((args.root/'config.json').read_text())
    report=json.loads((args.root/'pilot.json').read_text())
    entries=sorted(p.name for p in Path(config['patches_dir']).iterdir() if p.is_dir() and (p/'meta.json').is_file())
    zrange=tuple(map(int,config['z_range'].split(',')))
    records=[];dropped=[];started=time.monotonic()
    for start in range(0,len(entries),256):
        loaded=load_patch_payload_chunk(config['patches_dir'],entries[start:start+256],*zrange,
            report['patch_preprocessing']['erode_cells_default'],io_threads=4,keep_largest_component=True)
        for pid,payload,error,drop in loaded:
            if payload is None:
                dropped.append(dict(patch=pid,error=error,reason=drop));continue
            xyz=payload['zyxs'][...,::-1];q=payload['valid_quad_mask']
            keep=np.zeros(xyz.shape[:2],bool)
            for dr,dc in ((0,0),(0,1),(1,0),(1,1)):keep[dr:dr+q.shape[0],dc:dc+q.shape[1]] |= q
            v=xyz[keep]
            records.append(dict(patch=pid,low=v.min(0).tolist(),high=v.max(0).tolist(),quads=int(q.sum())))
        if start%2048==0:print(f'{min(start+256,len(entries))}/{len(entries)}; kept {len(records)}; {time.monotonic()-started:.1f}s',flush=True)
    result=dict(patches_dir=config['patches_dir'],z_range=zrange,preprocessing=report['patch_preprocessing'],
                source_entries=len(entries),patches=records,dropped=dropped,elapsed_seconds=time.monotonic()-started)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(f'Catalog complete: {len(records)} usable patches, {len(dropped)} excluded; {time.monotonic()-started:.1f}s',flush=True)


if __name__=='__main__':main()
