"""Compare v2/v3 compute with identical synthetic production-sized inputs.

Includes model forward/backward, clipping, AdamW and EMA for training. Excludes
volume I/O and data construction. Compiles both versions when --compile is set;
reports first-call/warm-up separately from steady-state wall-clock timings.
"""
import argparse
import copy
from dataclasses import replace
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch

from .model import build_model
from .train import load_checkpoint, compile_training_model, conv_memory_format, optimizer_update


def synthetic_batch(cfg, batch, observations):
    generator = torch.Generator().manual_seed(194)
    rand = lambda *shape: torch.rand(*shape,generator=generator)
    c, t, n = cfg.fine, observations, cfg.memory_patch_size
    positions = torch.zeros(batch,t,3)
    positions[...,2] = torch.arange(t)*cfg.memory_stride
    x = dict(fine=rand(batch,cfg.input_channels,c.depth,c.width,c.width),
             seed=torch.zeros(batch,1,3),seed_mask=torch.zeros(batch,1),seed_age=torch.ones(batch)*256,
             seed_tangent=torch.tensor([0.,0.,1.]).expand(batch,-1),
             memory_patches=rand(batch,t,cfg.input_channels,n,n,n),memory_mask=torch.ones(batch,t,dtype=torch.bool),
             memory_positions=positions,memory_frames=torch.eye(3).expand(batch,t,-1,-1).clone(),
             memory_seed_patch=rand(batch,cfg.input_channels,n,n,n),memory_seed_valid=torch.ones(batch,dtype=torch.bool),
             memory_seed_position=torch.zeros(batch,3),memory_seed_frame=torch.eye(3).expand(batch,-1,-1).clone())
    if cfg.memory_version == 3:
        x['route_frame'] = torch.eye(3).expand(batch,-1,-1).clone()
    hist = torch.zeros(batch,cfg.n_history,3)
    hist[...,2] = -torch.arange(1,cfg.n_history+1)
    q = 4*(cfg.n_future-1)+1
    candidates = torch.zeros(batch,2,cfg.n_future,3)
    candidates[...,2] = torch.arange(1,cfg.n_future+1)*cfg.future_step
    candidates[:,1,:,0] = 4
    labels = torch.zeros(batch,2,cfg.n_future); labels[:,0] = 1
    return dict(x=x,hist=hist,hmask=torch.ones(batch,cfg.n_history),dense_ab=torch.zeros(batch,q,2),
                dense_mask=torch.ones(batch,q),offtrack=torch.zeros(batch),endpoint_known=torch.zeros(batch),
                end_local=torch.zeros(batch,3),source=torch.zeros(batch),candidate_points=candidates,
                candidate_mask=torch.ones(batch,2,cfg.n_future,dtype=torch.bool),candidate_labels=labels,
                memory_target_identity=torch.ones(batch,t),memory_target_identity_mask=torch.ones(batch,t,dtype=torch.bool),
                memory_target_offset=torch.zeros(batch,t,3),memory_target_offset_mask=torch.ones(batch,t,dtype=torch.bool))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--device',default='cuda')
    ap.add_argument('--compile',action=argparse.BooleanOptionalAction,default=True)
    ap.add_argument('--batch',type=int,default=1)
    ap.add_argument('--repeats',type=int,default=10)
    ap.add_argument('--warmup',type=int,default=3)
    ap.add_argument('--threads',type=int,default=4)
    ap.add_argument('--versions',type=int,nargs='+',default=[2,3],choices=[2,3])
    ap.add_argument('--modes',nargs='+',default=['inference','training','training_sequence'],
                    choices=['inference','training','training_sequence'])
    args = ap.parse_args()
    if min(args.batch,args.repeats,args.warmup,args.threads) < 1:
        ap.error('Counts must be positive')
    torch.set_num_threads(args.threads)
    torch.manual_seed(123)
    old,*_ = load_checkpoint(args.checkpoint,'cpu')
    base_cfg,weights = old.cfg,old.state_dict()
    result = dict(torch=torch.__version__,device=args.device,
                  hardware=torch.cuda.get_device_name() if args.device.startswith('cuda') else 'CPU',
                  compile=args.compile,batch=args.batch,checkpoint=args.checkpoint,
                  precision='bf16 autocast, fp32 memory' if args.device.startswith('cuda') else 'fp32',
                  config=base_cfg.to_dict(),rows=[])
    sync = torch.cuda.synchronize if args.device.startswith('cuda') else lambda: None
    def save():
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(result,indent=2)+'\n')
    for version in args.versions:
        for mode in args.modes:
            if version == 2 and mode == 'training_sequence':
                continue
            cfg = replace(base_cfg,memory_version=version)
            model = build_model(cfg).to(args.device,memory_format=conv_memory_format(args.device))
            model.load_state_dict(weights,strict=version == 2)
            count = 1 if mode == 'inference' else cfg.memory_steps+1
            b = 1 if mode == 'inference' else args.batch
            batch = synthetic_batch(cfg,b,count)
            if mode == 'training_sequence':
                batch['route_sequence'] = synthetic_batch(cfg,b,count)
            ema = copy.deepcopy(model).requires_grad_(False).eval() if mode != 'inference' else None
            opt = torch.optim.AdamW(model.parameters(),lr=0.) if ema is not None else None
            model.train(mode != 'inference')
            wrapped = compile_training_model(model) if args.compile else model
            if mode == 'inference':
                x = {k:v.to(args.device) for k,v in batch['x'].items()}
                hist,hmask = batch['hist'].to(args.device),batch['hmask'].to(args.device)
                state = model.initial_memory(1,args.device)
                with torch.no_grad():
                    observed = model.recurrent_memory.observe(x,state)
                    state = {k:observed[k] for k in state}
                x['memory_seed_valid'].zero_()
                def step():
                    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=args.device.startswith('cuda')):
                        return wrapped(x,hist,hmask,memory=state)
            else:
                def step():
                    return optimizer_update(wrapped,ema,opt,[batch],1,0.,device=args.device,compute_metrics=False)
            print(f'Start v{version} {mode}',flush=True)
            sync(); started = time.perf_counter()
            for _ in range(args.warmup):
                last = step()
            sync(); warmup_seconds = time.perf_counter()-started
            diagnostics = {}
            if mode == 'inference':
                if not all(torch.isfinite(last[k]).all() for k in ('points','confidence')):
                    raise RuntimeError('Nonfinite inference output')
            else:
                diagnostics = {k:last[k] for k in ('loss','memory_grad_norm','rest_grad_norm')}
                if not all(np.isfinite(value) for value in diagnostics.values()):
                    raise RuntimeError(f'Nonfinite training result: {diagnostics}')
            if args.device.startswith('cuda'):
                torch.cuda.reset_peak_memory_stats()
            elapsed = []
            for _ in range(args.repeats):
                sync(); started = time.perf_counter(); step(); sync()
                elapsed.append((time.perf_counter()-started)*1000)
            row = dict(version=version,mode=mode,batch=b,observations=count,warmup_seconds=warmup_seconds,
                       mean_ms=float(np.mean(elapsed)),p50_ms=float(np.median(elapsed)),p95_ms=float(np.percentile(elapsed,95)),
                       ms_per_primary_sample=float(np.mean(elapsed)/b),samples=elapsed,
                       diagnostics=diagnostics,
                       peak_allocated_bytes=torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else None)
            result['rows'].append(row); save(); print(json.dumps(row),flush=True)
            del step,wrapped,model,ema,opt,batch
            gc.collect()
            if args.device.startswith('cuda'): torch.cuda.empty_cache()
    save()


if __name__ == '__main__':
    main()
