"""Preserve held-out recovery decisions from the archived 50k EMA policy."""
import argparse
import json
from pathlib import Path
from vesuvius.neural_tracing.fiber_follow.experiment import read_manifest
import numpy as np
import torch
from evaluate_recovery import load_archived_baseline,ArchivedTracer
from vesuvius.neural_tracing.fiber_follow.collect import DecisionCollector
from vesuvius.neural_tracing.fiber_follow.data import SampleConfig,OnPolicyStates,ZBand,load_fibers,split_fibers,fiber_manifest
from vesuvius.neural_tracing.fiber_follow.trace import ModelTracer,TraceParams
from vesuvius.neural_tracing.fiber_follow.volume import FiberVolume


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--archive',required=True)
    ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--fibers',default='/mnt/raid_nvme/spiral_dataset_working/fibers')
    ap.add_argument('--device',default='cuda')
    ap.add_argument('--seeds',type=int,default=96)
    ap.add_argument('--max-len',type=float,default=1200)
    args=ap.parse_args(argv);torch.set_num_threads(4)
    if args.out.exists():raise FileExistsError(args.out)
    model,crop,nh,spec,ck=load_archived_baseline(args.checkpoint,args.archive,args.device)
    manifest=read_manifest(args.manifest)
    _,fibers=split_fibers(load_fibers(args.fibers,grid_scale=spec.grid_scale),ZBand(45000/8,48500/8))
    if manifest['fibers']!=fiber_manifest(fibers):raise ValueError('Frozen geometry changed')
    cfg=SampleConfig(crop=crop,n_history=nh,recent_history_points=model.cfg.recent_history_points,
                     n_future=model.cfg.n_future,future_step=model.cfg.future_step)
    tracer=ArchivedTracer(model,FiberVolume(spec),crop,nh,TraceParams(max_len=args.max_len,confidence=.7,explore_calls=8),device=args.device)
    rows=[];seeds=manifest['calibration'][:args.seeds]
    try:
        for i,s in enumerate(seeds):
            col=DecisionCollector(fibers[s['fiber']],s['fiber'],s['t'],s['sign'],cfg,stride=1)
            tracer.trace(np.asarray([s['pos']]),np.asarray([s['heading']]),on_decision=lambda _,state:col(state))
            rows.extend(col.finish());print(dict(seeds=i+1,states=len(rows)),flush=True)
    finally:tracer.close()
    bank=OnPolicyStates(manifest=fiber_manifest(fibers),provenance=dict(split='calibration',checkpoint=str(Path(args.checkpoint).resolve()),
        step=50000,weights='EMA',seed_manifest_sha256=manifest['sha256'],seeds=seeds,volume=spec.to_dict(),
        threshold=.7,max_len=args.max_len,explore_calls=8,full_manifest=len(seeds)==len(manifest['calibration'])),
        **{k:np.asarray([r[k] for r in rows]) for k in OnPolicyStates.FIELDS+tuple(OnPolicyStates.OPTIONAL)})
    args.out.parent.mkdir(parents=True,exist_ok=True);bank.save(args.out)
    print(dict(states=len(bank),departed=int(bank.offtrack.sum()),out=str(args.out)),flush=True)

if __name__=='__main__':main()
