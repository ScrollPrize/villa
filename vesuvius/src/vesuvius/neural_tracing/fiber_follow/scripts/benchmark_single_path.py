"""Measure full-crop v11 training memory/throughput and complete tracing latency."""
import argparse
import copy
import json
from pathlib import Path
import time
from vesuvius.neural_tracing.fiber_follow.experiment import read_manifest
import numpy as np
import torch
from vesuvius.neural_tracing.fiber_follow.data import FollowDataset,SampleConfig,ZBand,load_fibers,split_fibers
from vesuvius.neural_tracing.fiber_follow.geometry import CropSpec
from vesuvius.neural_tracing.fiber_follow.model import ARCHITECTURE,FollowNet,FollowNetConfig,prepare_model
from vesuvius.neural_tracing.fiber_follow.train import optimizer_update
from vesuvius.neural_tracing.fiber_follow.trace import ModelTracer,TraceParams
from vesuvius.neural_tracing.fiber_follow.volume import FiberVolume,FiberVolumeSpec


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--fibers',default='/mnt/raid_nvme/spiral_dataset_working/fibers')
    ap.add_argument('--device',default='cuda')
    ap.add_argument('--microbatch',type=int,choices=(1,2),default=2)
    ap.add_argument('--flow-draws',type=int,default=64)
    ap.add_argument('--updates',type=int,default=3)
    args=ap.parse_args(argv)
    result=dict(architecture=ARCHITECTURE,device=args.device,microbatch=args.microbatch,
                effective_batch=8,flow_draws=args.flow_draws,crop=[176,96,96],passed=False)
    try:
        if not args.device.startswith('cuda') or not torch.cuda.is_available():
            raise RuntimeError('Preflight requires a working CUDA device; CPU smoke tests do not qualify')
        if args.updates<2: raise ValueError('Use at least two measured updates after warmup')
        torch.set_num_threads(4); torch.manual_seed(0); torch.backends.cudnn.benchmark=True
        manifest=read_manifest(args.manifest);spec=FiberVolumeSpec(**manifest['volume'])
        fibers=load_fibers(args.fibers,grid_scale=spec.grid_scale)
        train,val=split_fibers(fibers,ZBand(45000/spec.grid_scale,48500/spec.grid_scale))
        cfg=FollowNetConfig(flow_sigma=((1.,1.),)*16,flow_draws=args.flow_draws)
        sample=SampleConfig(crop=CropSpec(depth=176,width=96,behind=128,history_render='segments',history_sigma=.35))
        batch=next(iter(FollowDataset(train,spec,sample,ZBand(45000/8,48500/8),chunk=8)))
        micro=[{k:v[i:i+args.microbatch] for k,v in batch.items()} for i in range(0,8,args.microbatch)]
        model=prepare_model(FollowNet(cfg),args.device);ema=copy.deepcopy(model).requires_grad_(False)
        opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
        torch.cuda.reset_peak_memory_stats(args.device)
        times=[]
        for i in range(args.updates+1):
            torch.cuda.synchronize();start=time.perf_counter()
            optimizer_update(model,ema,opt,micro,2000+i,1e-3,device=args.device,compute_metrics=False)
            torch.cuda.synchronize()
            if i: times.append(time.perf_counter()-start)
        result.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(args.device),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(args.device),
                      update_seconds=times,samples_per_second=8/np.mean(times),gpu=torch.cuda.get_device_name(args.device))
        # End-to-end trace includes volume reading, rendering, all 8 velocity
        # evaluations and the final confidence evaluation, frame/loop checks.
        torch.backends.cudnn.benchmark=False
        tracer=ModelTracer(ema,FiberVolume(spec),sample.crop,128,
                           TraceParams(max_len=64,confidence=0.),device=args.device)
        seeds=manifest['monitor'][:1];decisions=[]
        try:
            torch.cuda.synchronize();start=time.perf_counter()
            paths,reasons=tracer.trace(np.array([s['pos'] for s in seeds]),np.array([s['heading'] for s in seeds]),
                on_decision=lambda i,state:decisions.append(state['travelled']))
            torch.cuda.synchronize(); elapsed=time.perf_counter()-start
            result.update(trace_seconds=elapsed,trace_decisions=len(decisions),trace_reasons=reasons,
                          seconds_per_decision=elapsed/max(1,len(decisions)),trace_points=[len(p) for p in paths],
                          trace_length=[float(np.linalg.norm(np.diff(p,axis=0),axis=-1).sum()) for p in paths],
                          velocity_evaluations_per_decision=8,final_confidence_evaluations=1)
        finally: tracer.close()
        result['passed']=True
    except Exception as exc:
        result['error']=f'{type(exc).__name__}: {exc}'
        if isinstance(exc,torch.cuda.OutOfMemoryError):
            result['next_action']='Rerun with --microbatch 1 (eight accumulation steps); retain crop and effective batch 8'
        raise
    finally:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(result,indent=2))
        print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__': main()
