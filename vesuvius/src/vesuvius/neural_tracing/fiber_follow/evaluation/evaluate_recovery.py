"""Evaluate fixed observed recovery states.

The model uses exactly the stored position, frame and observed history. Use
--limit for a CPU fixture smoke run.
"""
import argparse
import json
from pathlib import Path
import time
import hashlib
import numpy as np
import torch
from vesuvius.neural_tracing.fiber_follow.data.data import (
    OnPolicyStates,SampleConfig,ZBand,load_fibers,split_fibers,
)
from vesuvius.neural_tracing.fiber_follow.data.observations import FiberTracer, ObservationBuilder
from vesuvius.neural_tracing.fiber_follow.train.train import load_checkpoint
from vesuvius.neural_tracing.fiber_follow.data.volume import FiberVolume
from vesuvius.neural_tracing.fiber_follow.shared.experiment import jsonable
from vesuvius.neural_tracing.fiber_follow.evaluation.recovery_fixtures import recovery_counts, evaluate_recovery_states
from vesuvius.neural_tracing.fiber_follow.tracing.policy import DIAGNOSTIC_THRESHOLDS


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('checkpoint')
    ap.add_argument('--fixtures',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--fibers',default='/mnt/raid_nvme/spiral_dataset_working/fibers')
    ap.add_argument('--device',default='cuda')
    ap.add_argument('--limit',type=int,default=0)
    ap.add_argument('--thresholds',type=float,nargs='+',default=DIAGNOSTIC_THRESHOLDS)
    ap.add_argument('--recovery-length',type=float,default=32)
    ap.add_argument('--sampling-seed',type=int,default=0)
    args=ap.parse_args(argv);torch.set_num_threads(4)
    model,crop,nh,spec,ck=load_checkpoint(args.checkpoint,args.device)
    states=OnPolicyStates.load(args.fixtures)
    fibers=load_fibers(args.fibers,grid_scale=spec.grid_scale)
    _,fibers=split_fibers(fibers,ZBand(45000/spec.grid_scale,48500/spec.grid_scale));states.validate_fibers(fibers)
    if states.provenance.get('split')!='calibration': raise ValueError('Use held-out calibration recovery fixtures')
    sample=SampleConfig(crop=crop,n_history=nh,recent_history_points=model.cfg.recent_history_points,
                        n_future=model.cfg.n_future,future_step=model.cfg.future_step)
    from vesuvius.neural_tracing.fiber_follow.data.ct_normalization import prepare_normalization
    prepare_normalization(args.out.parent, [spec], known=ck['ct_normalization'])
    vol=FiberVolume(spec)
    count=len(states) if args.limit==0 else min(args.limit,len(states));start=time.monotonic()
    def progress(done, total):
        if done % 8 == 0:
            print(dict(states=done,total=total,seconds=time.monotonic()-start),flush=True)
    rows,predictions=evaluate_recovery_states(model,vol,states,fibers,sample,device=args.device,
        tracer_class=FiberTracer,
        batch_builder=ObservationBuilder(model.cfg),
        thresholds=args.thresholds,recovery_length=args.recovery_length,n_commit=ck.get('n_commit',8),
        tolerance=ck.get('tolerance',1.5),sampling_seed=args.sampling_seed,limit=args.limit,
        progress=progress)
    report=dict(checkpoint=str(Path(args.checkpoint).resolve()),checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        model_type=model.model_type,step=ck['step'],fixture_sha256=hashlib.sha256(args.fixtures.read_bytes()).hexdigest(),
        evaluated_states=count,full_fixture=count==len(states),recovery_length=args.recovery_length,seconds=time.monotonic()-start,
        sampling_seed=args.sampling_seed,
        thresholds={str(t):recovery_counts([r for r in rows if r['threshold']==t]) for t in args.thresholds},rows=rows)
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(report,indent=2,default=jsonable))
    np.savez(args.out.with_suffix('.npz'),points=np.asarray(predictions),state_indices=np.arange(count))
    print(dict(states=count,seconds=report['seconds'],out=str(args.out)),flush=True)

if __name__=='__main__':main()
