"""Evaluate fixed observed recovery states, including the archived 50k EMA baseline.

Both models use exactly the stored position, frame and observed history. The
baseline implementation is read from the preserved source archive; old weights
are never accepted by the v11 loader. Use --limit for a CPU fixture smoke run.
"""
import argparse
import json
from pathlib import Path
import sys
import tarfile
import time
import types
import hashlib
import numpy as np
import torch
from vesuvius.neural_tracing.fiber_follow.data import (
    OnPolicyStates,SampleConfig,ZBand,load_fibers,split_fibers,
)
from vesuvius.neural_tracing.fiber_follow.geometry import CropSpec
from vesuvius.neural_tracing.fiber_follow.model import prepare_model
from vesuvius.neural_tracing.fiber_follow.trace import ModelTracer
from vesuvius.neural_tracing.fiber_follow.train import load_checkpoint
from vesuvius.neural_tracing.fiber_follow.volume import FiberVolume,FiberVolumeSpec
from vesuvius.neural_tracing.fiber_follow.experiment import jsonable
from vesuvius.neural_tracing.fiber_follow.recovery import recovery_counts, evaluate_recovery_states


def load_archived_baseline(path,archive,device):
    # Trusted local source snapshot, not an alternative production interface.
    ck=torch.load(path,map_location=device,weights_only=False)
    if ck['architecture']!='future_flow_v10' or ck.get('step')!=50000:
        raise ValueError('Baseline must be the original 50k v10 checkpoint')
    module=types.ModuleType('_preserved_v10_baseline');sys.modules[module.__name__]=module
    with tarfile.open(archive) as saved:
        source=saved.extractfile('model.py').read().decode()
    exec(compile(source,str(archive)+'/model.py','exec'),module.__dict__)
    cfg=module.FollowNetConfig(**dict(ck['model_cfg'],deterministic_first=True))
    old=prepare_model(module.FollowNet(cfg),device).eval();old.load_state_dict(ck['ema'])
    class BaselineAdapter(torch.nn.Module):
        def __init__(self):
            super().__init__();self.original=old;self.cfg=cfg
            self.generator=torch.Generator(device=device).manual_seed(0)
        def forward(self,x,hist,hmask,**kw):
            # Legacy confidence uses support from its original random samples.
            # The archived tracer resets this once per trace call, as v10 did.
            out=self.original(x,hist,hmask,generator=self.generator)
            return dict(points=out['candidates'][:,0],confidence=out['confidence'][:,0],confidence_logits=out['confidence_logits'][:,0])
    return BaselineAdapter().eval(),CropSpec(**ck['crop']),ck['n_history'],FiberVolumeSpec(**ck['vol_spec']),ck


class ArchivedTracer(ModelTracer):
    def trace(self,*args,**kwargs):
        self.model.generator.manual_seed(self.p.seed)
        return super().trace(*args,**kwargs)


def main(argv=None, *, checkpoint_loader=None, model_tracer=None, batch_builder_factory=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('checkpoint')
    ap.add_argument('--fixtures',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--baseline-archive',type=Path)
    ap.add_argument('--fibers',default='/mnt/raid_nvme/spiral_dataset_working/fibers')
    ap.add_argument('--device',default='cuda')
    ap.add_argument('--limit',type=int,default=0)
    ap.add_argument('--thresholds',type=float,nargs='+',default=(.5,.85))
    ap.add_argument('--recovery-length',type=float,default=32)
    ap.add_argument('--sampling-seed',type=int,default=0)
    args=ap.parse_args(argv);torch.set_num_threads(4)
    if args.baseline_archive and batch_builder_factory is not None:
        ap.error('Use evaluate_recovery.py for the archived baseline')
    loaded=load_archived_baseline(args.checkpoint,args.baseline_archive,args.device) if args.baseline_archive else (checkpoint_loader or load_checkpoint)(args.checkpoint,args.device)
    model,crop,nh,spec,ck=loaded
    states=OnPolicyStates.load(args.fixtures)
    fibers=load_fibers(args.fibers,grid_scale=spec.grid_scale)
    _,fibers=split_fibers(fibers,ZBand(45000/spec.grid_scale,48500/spec.grid_scale));states.validate_fibers(fibers)
    if states.provenance.get('split')!='calibration': raise ValueError('Use held-out calibration recovery fixtures')
    sample=SampleConfig(crop=crop,n_history=nh,recent_history_points=model.cfg.recent_history_points,
                        n_future=model.cfg.n_future,future_step=model.cfg.future_step)
    vol=FiberVolume(spec)
    count=len(states) if args.limit==0 else min(args.limit,len(states));start=time.monotonic()
    def progress(done, total):
        if done % 8 == 0:
            print(dict(states=done,total=total,seconds=time.monotonic()-start),flush=True)
    rows,predictions=evaluate_recovery_states(model,vol,states,fibers,sample,device=args.device,
        tracer_class=ArchivedTracer if args.baseline_archive else (model_tracer or ModelTracer),
        batch_builder=batch_builder_factory(model.cfg) if batch_builder_factory else None,
        thresholds=args.thresholds,recovery_length=args.recovery_length,n_commit=ck.get('n_commit',8),
        tolerance=ck.get('tolerance',1.5),sampling_seed=args.sampling_seed,limit=args.limit,
        archived=bool(args.baseline_archive),progress=progress)
    report=dict(checkpoint=str(Path(args.checkpoint).resolve()),checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        architecture=ck['architecture'],step=ck['step'],fixture_sha256=hashlib.sha256(args.fixtures.read_bytes()).hexdigest(),
        evaluated_states=count,full_fixture=count==len(states),recovery_length=args.recovery_length,seconds=time.monotonic()-start,
        sampling_seed=args.sampling_seed,sampler_mode=getattr(model.cfg,'sampler_mode','zero'),
        thresholds={str(t):recovery_counts([r for r in rows if r['threshold']==t]) for t in args.thresholds},rows=rows)
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(report,indent=2,default=jsonable))
    np.savez(args.out.with_suffix('.npz'),points=np.asarray(predictions),state_indices=np.arange(count))
    print(dict(states=count,seconds=report['seconds'],out=str(args.out)),flush=True)

if __name__=='__main__':main()
