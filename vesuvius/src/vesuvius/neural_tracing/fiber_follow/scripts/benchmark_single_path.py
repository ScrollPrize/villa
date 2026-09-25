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
from vesuvius.neural_tracing.fiber_follow.train import optimizer_update,compile_training_model
from vesuvius.neural_tracing.fiber_follow.trace import ModelTracer,TraceParams
from vesuvius.neural_tracing.fiber_follow.volume import FiberVolume,FiberVolumeSpec


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--fibers',default='/mnt/raid_nvme/spiral_dataset_working/fibers')
    ap.add_argument('--device',default='cuda')
    ap.add_argument('--microbatch',type=int,choices=(1,2),default=2)
    ap.add_argument('--cache-training-encoding',action=argparse.BooleanOptionalAction,default=True,
                    help='Measure encoder graph reuse (default: enabled), including its increased memory requirement')
    ap.add_argument('--compile',dest='compile_model',action=argparse.BooleanOptionalAction,default=True,
                    help='Compile CUDA training methods (default: enabled)')
    ap.add_argument('--flow-draws',type=int,default=64)
    ap.add_argument('--sampler-mode',choices=('zero','gaussian'),default='gaussian')
    ap.add_argument('--updates',type=int,default=3)
    ap.add_argument('--warmup',type=int,default=1)
    ap.add_argument('--conv-memory-format',choices=('contiguous','channels_last_3d'),
                    help='Override convolution layout for paired performance experiments')
    args=ap.parse_args(argv)
    result=dict(architecture=ARCHITECTURE,device=args.device,microbatch=args.microbatch,
                effective_batch=8,flow_draws=args.flow_draws,crop=[176,96,96],passed=False,
                cache_training_encoding=args.cache_training_encoding,compile_model=args.compile_model,sampler_mode=args.sampler_mode)
    try:
        if not args.device.startswith('cuda') or not torch.cuda.is_available():
            raise RuntimeError('Preflight requires a working CUDA device; CPU smoke tests do not qualify')
        if args.updates<2: raise ValueError('Use at least two measured updates after warmup')
        if args.warmup<1: raise ValueError('Use at least one warmup update')
        torch.set_num_threads(4); torch.manual_seed(0); torch.backends.cudnn.benchmark=True
        manifest=read_manifest(args.manifest);spec=FiberVolumeSpec(**manifest['volume'])
        fibers=load_fibers(args.fibers,grid_scale=spec.grid_scale)
        train,val=split_fibers(fibers,ZBand(45000/spec.grid_scale,48500/spec.grid_scale))
        cfg=FollowNetConfig(flow_sigma=((1.,1.),)*16,flow_draws=args.flow_draws,sampler_mode=args.sampler_mode)
        sample=SampleConfig(crop=CropSpec(depth=176,width=96,behind=128,history_render='segments',history_sigma=.35))
        batch=next(iter(FollowDataset(train,spec,sample,ZBand(45000/8,48500/8),chunk=8)))
        micro=[{k:v[i:i+args.microbatch] for k,v in batch.items()} for i in range(0,8,args.microbatch)]
        model=prepare_model(FollowNet(cfg),args.device)
        if args.conv_memory_format is not None:
            layout = torch.contiguous_format if args.conv_memory_format == 'contiguous' else torch.channels_last_3d
            torch.nn.utils.convert_conv3d_weight_memory_format(model,layout)
        result.update(conv_memory_format=('contiguous' if model.encoders[0][0].weight.is_contiguous()
                                          else 'channels_last_3d'),
                      torch_version=torch.__version__,cuda_version=torch.version.cuda,
                      cudnn_version=torch.backends.cudnn.version(),cpu_threads=torch.get_num_threads(),
                      warmup=args.warmup,updates=args.updates,seed=0,
                      precision='cuda_bfloat16_autocast',manifest=str(args.manifest),fibers=args.fibers)
        ema=copy.deepcopy(model).requires_grad_(False)
        opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
        if args.compile_model:
            print('Compiling CUDA training methods; warmup includes compilation.',flush=True)
            compile_training_model(model)
        torch.cuda.reset_peak_memory_stats(args.device)
        times=[]
        result['warmup_seconds']=[]
        for i in range(args.updates+args.warmup):
            torch.cuda.synchronize();start=time.perf_counter()
            optimizer_update(model,ema,opt,micro,2000+i,1e-3,device=args.device,compute_metrics=False,
                             cache_training_encoding=args.cache_training_encoding)
            torch.cuda.synchronize()
            elapsed=time.perf_counter()-start
            if i>=args.warmup: times.append(elapsed)
            else: result['warmup_seconds'].append(elapsed)
        result.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(args.device),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(args.device),
                      update_seconds=times,samples_per_second=8/np.mean(times),gpu=torch.cuda.get_device_name(args.device))
        result['update_seconds_summary']=dict(mean=float(np.mean(times)),median=float(np.median(times)),
                                              min=min(times),max=max(times),p95=float(np.percentile(times,95)))
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
            result['next_action']=('Rerun with --no-cache-training-encoding to release retained encoder graphs' if args.cache_training_encoding
                                   else 'Rerun with --microbatch 1 (eight accumulation steps); retain crop and effective batch 8')
        raise
    finally:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(result,indent=2))
        print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__': main()
