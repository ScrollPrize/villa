"""Prepare label-free per-piece mean/median/max density penalty inputs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from analyze_trace_cost_removal import load_steps, piece_costs


def densities(segments, begin, end):
    checked=piece_costs(segments,begin,end)
    lo=np.searchsorted(segments[:,1],begin,side='right')
    hi=np.searchsorted(segments[:,0],end,side='left')
    s=segments[lo:hi]
    lengths=np.minimum(s[:,1],end)-np.maximum(s[:,0],begin)
    density=s[:,2]/(s[:,1]-s[:,0])*s[:,4]
    order=np.argsort(density,kind='stable')
    index=np.searchsorted(np.cumsum(lengths[order]),lengths.sum()/2,side='left')
    return dict(mean=checked['mean_cost_per_prediction_voxel'],
                median=float(density[order[index]]),max=checked['max_cost_per_prediction_voxel'])


def multiplier(density):
    if not np.isfinite(density) or density < -1e-10:
        raise ValueError('invalid tracer density')
    return max(.1,.5/(.5+max(0.,density)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('steps',type=Path)
    p.add_argument('consistency',type=Path)
    p.add_argument('trace_artifact',type=Path)
    p.add_argument('output_dir',type=Path)
    args=p.parse_args()
    traces=load_steps(args.steps)
    with args.consistency.open() as stream:
        pieces=list(csv.DictReader(stream))
    output={mode:[] for mode in ['control','mean','median','max']}
    for r in pieces:
        trace=int(r['original_trace']); begin=float(r['begin_arc_base']); end=float(r['end_arc_base'])
        stats=densities(traces[trace],begin,end)
        for mode,rows in output.items():
            d=0. if mode=='control' else stats[mode]
            rows.append(dict(piece=int(r['piece']),original_trace=trace,begin_arc_base=begin,
                             end_arc_base=end,density=d,multiplier=multiplier(d)))
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for mode,rows in output.items():
        document=dict(version=1,statistic=mode,trace_artifact=str(args.trace_artifact.resolve()),
                      mapping='max(0.1, 0.5/(0.5+d))',density_units='loss_per_prediction_voxel',
                      steps_sha256=hashlib.sha256(args.steps.read_bytes()).hexdigest(),pieces=rows)
        (args.output_dir/(mode+'.json')).write_text(json.dumps(document,separators=(',',':'))+'\n')
        print(mode,len(rows),float(np.mean([r['multiplier'] for r in rows])))


if __name__=='__main__':
    main()
