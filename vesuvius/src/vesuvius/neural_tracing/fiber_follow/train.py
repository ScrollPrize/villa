"""Supervised joint-flow fiber follower with asynchronous, dense-GT DAgger replay."""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from vesuvius.neural_tracing.fiber_follow.data import (
    FollowDataset, OnPolicyStates, SampleConfig, ZBand, load_fibers, split_fibers, DATA_POLICY, fiber_manifest,
)
from vesuvius.neural_tracing.fiber_follow.geometry import CropSpec
from vesuvius.neural_tracing.fiber_follow.model import ARCHITECTURE, FollowNet, FollowNetConfig, flow_targets, prepare_model, prior_mean
from vesuvius.neural_tracing.fiber_follow.online import OnlineCollector
from vesuvius.neural_tracing.fiber_follow.runloop import (
    RunLog, lr_at, prepare_run_dir,
    read_checkpoint as _read_checkpoint, save_checkpoint as _save_checkpoint,
)
from vesuvius.neural_tracing.fiber_follow.trace import DEFAULT_CONFIDENCE
from vesuvius.neural_tracing.fiber_follow.supervision import loss_fn, prefix_labels, refinement_metrics
from vesuvius.neural_tracing.fiber_follow.policy import DEFAULT_MAX_RECOVERY_DISTANCE, DEFAULT_N_COMMIT
from vesuvius.neural_tracing.fiber_follow.volume import FiberVolumeSpec
from vesuvius.neural_tracing.fiber_follow.training_log import format_training_log


def save_checkpoint(path, model, ema, vol_spec, sample_cfg, extra=None):
    _save_checkpoint(path, model, vol_spec, sample_cfg.crop, sample_cfg.n_history, ARCHITECTURE,
                     dict(extra or {}, ema=ema.state_dict(), sample_cfg=dataclasses.asdict(sample_cfg)))


def read_checkpoint(path, device='cuda'):
    return _read_checkpoint(path, ARCHITECTURE, device)


def load_checkpoint(path, device='cuda'):
    ck = read_checkpoint(path, device)
    model = prepare_model(FollowNet(FollowNetConfig(**ck['model_cfg'])), device)
    model.load_state_dict(ck['ema'])
    model.eval()
    return model, CropSpec(**ck['crop']), ck['n_history'], FiberVolumeSpec(**ck['vol_spec']), ck


@torch.no_grad()
def fit_flow_sigma(batches, cfg, states, progress=None):
    """Fit residual standard deviations from the training loader alone.

    Uses exactly the tokens the flow loss supervises: annotated, not departed,
    and not crop-censored (``flow_targets``).
    """
    count = torch.zeros(cfg.n_future, dtype=torch.float64)
    censored_count = torch.zeros_like(count)
    total = torch.zeros(cfg.n_future, 2, dtype=torch.float64)
    square = torch.zeros_like(total)
    seen = 0
    while seen < states:
        batch = next(batches)
        n = min(len(batch['hist']), states-seen)
        x1, token_mask, censored = flow_targets({k: batch[k][:n] for k in ('plane_ab', 'plane_mask', 'offtrack')}, cfg)
        mu = prior_mean(batch['hist'][:n], batch['hmask'][:n], cfg)[..., :2].double()
        known = token_mask.bool()
        residual = torch.where(known[..., None], x1[..., :2].double()-mu, 0.)
        if not torch.isfinite(residual).all():
            raise ValueError('Non-finite annotated residual during flow scale calibration')
        count += known.sum(0)
        censored_count += censored.sum(0)
        total += residual.sum(0)
        square += residual.square().sum(0)
        seen += n
        if progress is not None and (seen % 128 == 0 or seen == states):
            progress(seen)
    if (count < 2).any():
        raise ValueError('Flow scale calibration needs at least two known targets on every plane; '
                         'increase --flow-calibration-states')
    mean = total / count[:, None]
    std = (square / count[:, None] - mean.square()).clamp_min(0).sqrt()
    sigma = std.clamp_min(1.)
    return tuple(map(tuple, sigma.tolist())), dict(states=seen, known_counts=count.long().tolist(),
                                                  censored_counts=censored_count.long().tolist(),
                                                  residual_mean=mean.tolist(), residual_std=std.tolist())


@torch.no_grad()
def update_ema(ema, model, step, decay):
    """Update EMA once per optimizer update, ramping the decay in early updates.

    Without the ramp the average still carries 37% of the random initialization
    after 1,000 updates, which is what the first collector and diagnostics use.
    """
    effective_decay = min(decay, (1+step)/(10+step))
    for average, current in zip(ema.parameters(), model.parameters(), strict=True):
        average.lerp_(current.detach(), 1-effective_decay)
    for average, current in zip(ema.buffers(), model.buffers(), strict=True):
        average.copy_(current)


def training_rng_state():
    state = dict(torch=torch.get_rng_state(), numpy=np.random.get_state())
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state


def restore_training_rng(state):
    torch.set_rng_state(state['torch'].cpu())
    np.random.set_state(state['numpy'])
    if 'cuda' in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def resume_training(ck, model, ema, opt):
    """Restore weights, EMA, optimizer and RNG from a resumable checkpoint.

    Only ``ckpt_*.pt``/``last.pt`` written by training carry optimizer state;
    collector snapshots do not. Loader workers restart their own streams, so
    the sampled data sequence after a resume differs from an uninterrupted run.
    Returns the completed update count and replay samples seen so far.
    """
    if 'optimizer' not in ck:
        raise ValueError('Checkpoint holds no optimizer state; resume from ckpt_*.pt or last.pt of a run')
    model.load_state_dict(ck['model'])
    ema.load_state_dict(ck['ema'])
    opt.load_state_dict(ck['optimizer'])
    restore_training_rng(ck['rng'])
    return int(ck['step']), int(ck.get('replay_seen', 0))


def compile_training_model(model):
    """Compile the entry points training actually calls, preserving checkpoint keys.

    Call after copying the EMA and restoring any checkpoint. The EMA stays eager
    for tracing/diagnostics. Noise draws keep the eager RNG implementation, and
    backward runs outside autocast, matching optimizer_update.
    """
    import torch._functorch.config
    torch._functorch.config.backward_pass_autocast = 'off'
    for name in ('encode_conditioning', 'generate_training_curve', 'training_forward'):
        setattr(model,name,torch.compile(getattr(model,name),options={'fallback_random':True}))
    return model


def optimizer_update(model, ema, opt, batches, update, lr, *, device, tolerance=1.5, decay=.999, compute_metrics=True,
                     cache_training_encoding=True, n_commit=DEFAULT_N_COMMIT):
    """Accumulate microbatches, clip once, step once, then update EMA once.

    ``n_commit`` sets the confidence near-window and refinement metric width; it
    should equal the rollout commit limit.

    Encoding reuse retains each microbatch's encoder graph until its
    backward pass. It saves a forward pass at a substantial memory cost.
    """
    if cache_training_encoding and model.cfg.norm != 'group':
        raise ValueError('Encoding reuse requires group norm; batch norm updates state on each forward')
    for group in opt.param_groups:
        group['lr']=lr
    opt.zero_grad(set_to_none=True)
    total_loss=0.
    metrics={}
    sources=np.zeros(3,dtype=np.int64)
    strata=np.zeros((2,5),dtype=np.int64)
    total=sum(len(b['hist']) for b in batches)
    last=None
    # Label all detached generated curves first, so censoring and departures do
    # not change objective scaling with microbatch size. Caching retains encoder
    # graphs; disabling it retains only coordinates and re-encodes. Refinement
    # and labels stay detached, and the effective-batch denominators are unchanged.
    normalizers = dict(flow=0.,near=0.,full=0.)
    curves=[]
    encodings=[]
    refinement_steps=[]
    with torch.no_grad():
        for cpu in batches:
            x,hist,hmask = (cpu[k].to(device) for k in ('x','hist','hmask'))
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=str(device).startswith('cuda')):
                encoding_args={}
                if cache_training_encoding:
                    with torch.enable_grad():
                        encoding_args['encoding']=model.encode_conditioning(x.float(),hist,hmask)
                encodings.append(encoding_args)
                if compute_metrics:
                    points,steps=model.generate_training_curve(x.float(),hist,hmask,return_steps=True,**encoding_args)
                    refinement_steps.append(steps.float().cpu())
                    del steps
                else:
                    points=model.generate_training_curve(x.float(),hist,hmask,**encoding_args)
                points=points.float().cpu()
            curves.append(points)
            _,mask,_=prefix_labels(points,cpu,tolerance,model.cfg.max_recovery_distance)
            normalizers['near'] += mask[:,:n_commit].sum().item()
            normalizers['full'] += mask.sum().item()
            normalizers['flow'] += flow_targets(cpu,model.cfg)[1].sum().item()
        del x,hist,hmask,encoding_args
    if compute_metrics:
        targets={key:torch.cat([b[key] for b in batches]) for key in
                 ('plane_ab','plane_mask','offtrack','gt_history','gt_history_mask')}
        metrics['refinement']=refinement_metrics(torch.cat(refinement_steps),targets,model.cfg,n_commit)
        del refinement_steps,targets
    for cpu,points in zip(batches,curves):
        encoding_args=encodings.pop(0)
        batch={k:v.to(device) for k,v in cpu.items()}
        weight=len(batch['hist'])/total
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=str(device).startswith('cuda')):
            out=model.training_forward(batch['x'].float(),batch['hist'],batch['hmask'],batch,points.to(device),**encoding_args)
            loss,stats=loss_fn(out,batch,model.cfg,tolerance,update=update,compute_metrics=compute_metrics,normalizers=normalizers,
                               n_commit=n_commit)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Non-finite loss at update {update}')
        loss.backward()
        total_loss+=loss.detach().item()
        for key,value in stats.items():
            metrics[key]=metrics.get(key,0.)+value*(weight if key in ('confidence_coefficient','flow_known_fraction','flow_censored_fraction') else 1.)
        if 'source' in cpu:
            for source in range(3):
                sources[source]+=int((cpu['source']==source).sum())
            for source in (1,2):
                for band in range(5):
                    strata[source-1,band]+=int(((cpu['source']==source)&(cpu['stratum']==band)).sum())
        last={k:v.detach() for k,v in batch.items()}
        del out,loss,batch,encoding_args
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    opt.step()
    update_ema(ema,model,update,decay)
    metrics.update(fresh_fraction=sources[0]/total,fixed_fraction=sources[1]/total,recent_fraction=sources[2]/total,
                   replay_stratum_counts=strata.tolist(),replay_samples=int(sources[1:].sum()))
    return total_loss,metrics,last


def build_parser():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--fiber-zarrs',required=True)
    ap.add_argument('--fibers',required=True)
    ap.add_argument('--ct',required=True)
    ap.add_argument('--ct-level',type=int,default=1)
    ap.add_argument('--ct-grid-scale',type=float,default=8.)
    ap.add_argument('--name',required=True)
    ap.add_argument('--out-root',default=str(Path(__file__).parent/'output'))
    ap.add_argument('--device',default='cuda')
    ap.add_argument('--steps',type=int,default=50000)
    ap.add_argument('--batch',type=int,default=8,help='Effective batch per optimizer update')
    ap.add_argument('--microbatch',type=int,choices=(1,2),default=2)
    ap.add_argument('--cache-training-encoding',action=argparse.BooleanOptionalAction,default=True,
                    help='Reuse encoder graphs across training passes (default: enabled; higher GPU memory use)')
    ap.add_argument('--compile',dest='compile_model',action=argparse.BooleanOptionalAction,default=True,
                    help='Compile CUDA training methods (default: enabled; first update includes compilation)')
    ap.add_argument('--lr',type=float,default=1e-3)
    ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--worker-cache-gb',type=float,default=1.)
    ap.add_argument('--val-z',type=float,nargs=2,default=(45000.,48500.))
    ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--warmup',type=int,default=1000)
    ap.add_argument('--ema-decay',type=float,default=.999)
    ap.add_argument('--ckpt-every',type=int,default=1000)
    ap.add_argument('--log-every',type=int,default=50)
    ap.add_argument('--diag-every',type=int,default=1000)
    ap.add_argument('--diag-batch',type=int,default=1)
    ap.add_argument('--flow-draws',type=int,default=64)
    ap.add_argument('--flow-calibration-states',type=int,default=2048)
    ap.add_argument('--tolerance',type=float,default=1.5)
    ap.add_argument('--n-commit',type=int,default=DEFAULT_N_COMMIT,
                    help='Rollout commit limit; also the confidence near-window and refinement metric width')
    ap.add_argument('--fixed-bank',required=True)
    ap.add_argument('--onpolicy',nargs='*',default=[])
    ap.add_argument('--dagger-every',type=int,default=1000)
    ap.add_argument('--dagger-device')
    ap.add_argument('--dagger-seeds',type=int,default=64)
    ap.add_argument('--dagger-batch',type=int,default=1)
    ap.add_argument('--dagger-explore-calls',type=int,default=8)
    ap.add_argument('--dagger-trace-len',type=float,default=6000.)
    ap.add_argument('--replay-keep',type=int,default=4)
    ap.add_argument('--manifest',required=True,help='Frozen monitor, calibration and final seed manifest')
    ap.add_argument('--benchmark',required=True,help='Preflight JSON from scripts/benchmark_single_path.py')
    ap.add_argument('--resume',help='ckpt_*.pt or last.pt inside output/NAME; continues that run in place')
    # Intentionally no legacy fine-tune, sampling, ranking or teacher options.
    return ap


def main(argv=None):
    args=build_parser().parse_args(argv)
    if args.batch != 8 or args.batch % args.microbatch:
        raise ValueError('Keep effective batch 8; use microbatch 2 or 1')
    if min(args.steps,args.ckpt_every,args.log_every,args.replay_keep,args.diag_batch,args.flow_draws)<1:
        raise ValueError('Update counts, cadences and dimensions must be positive')
    if min(args.workers,args.diag_every,args.dagger_every,args.warmup)<0 or not 0<=args.ema_decay<1:
        raise ValueError('Invalid training settings')
    if str(args.device).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; full training requires a working GPU')
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    spec=FiberVolumeSpec(args.fiber_zarrs,ct_zarr=args.ct,ct_level=args.ct_level,ct_grid_scale=args.ct_grid_scale,inputs='ct+presence')
    if args.ct_level!=1 or args.ct_grid_scale!=8.:
        raise ValueError('This experiment requires CT level 1, ct_grid_scale=8')
    crop=CropSpec(depth=176,width=96,behind=128,history_render='segments',history_sigma=.35)
    sample_cfg=SampleConfig(crop=crop,history_jitter=0.,history_wobble=1.,angle_sigmas_deg=(2.,5.,10.))
    model_cfg=FollowNetConfig(flow_draws=args.flow_draws)
    compile_model=args.compile_model and torch.device(args.device).type=='cuda'
    benchmark=json.loads(Path(args.benchmark).read_text())
    if (benchmark.get('architecture')!=ARCHITECTURE or not benchmark.get('passed')
        or benchmark.get('microbatch')!=args.microbatch or benchmark.get('flow_draws')!=args.flow_draws
        or benchmark.get('cache_training_encoding',False)!=args.cache_training_encoding
        or benchmark.get('compile_model',False)!=compile_model
        or benchmark.get('crop')!=[176,96,96]):
        raise ValueError('A successful matching full-crop benchmark is required before training')
    fibers=load_fibers(args.fibers,grid_scale=spec.grid_scale)
    band=ZBand(*(v/spec.grid_scale for v in args.val_z))
    train_f,val_f=split_fibers(fibers,band)
    from vesuvius.neural_tracing.fiber_follow.experiment import read_manifest
    manifest=read_manifest(args.manifest)
    if manifest['fibers']!=fiber_manifest(val_f) or manifest['volume']!=spec.to_dict():
        raise ValueError('Frozen manifest does not match run geometry/volume')
    fixed=OnPolicyStates.load(args.fixed_bank)
    resume=read_checkpoint(args.resume,args.device) if args.resume else None
    out=prepare_run_dir(args.out_root,args.name,resume=resume is not None)
    published=out/'dagger'/'replay.json'
    if resume is not None:
        if Path(args.resume).resolve().parent!=out.resolve():
            raise ValueError('--resume must name a checkpoint inside the run directory given by --name')
        if resume.get('seed_manifest_sha256')!=manifest['sha256']:
            raise ValueError('Resumed checkpoint was trained against a different frozen manifest')
        # Continue from the caches the interrupted run had published, not the initial ones.
        replay_paths=json.loads(published.read_text()) if published.exists() else list(args.onpolicy)
    else:
        replay_paths=list(args.onpolicy)
    caches=[OnPolicyStates.load(p) for p in replay_paths]
    collector=OnlineCollector(out/'dagger',args.fibers,args.val_z,args.dagger_device or args.device,
                              every=args.dagger_every,max_seeds=args.dagger_seeds,batch=args.dagger_batch,
                              explore_calls=args.dagger_explore_calls,seed=args.seed,replay_keep=args.replay_keep,
                              initial=[c._dir for c in caches],trace_len=args.dagger_trace_len,confidence=.7,n_commit=args.n_commit)
    # A resumed run reseeds its loader workers so it does not replay the run's first states.
    ds=FollowDataset(train_f,spec,sample_cfg,band,chunk=args.microbatch,seed=args.seed+(resume['step'] if resume else 0),
                     cache_bytes=int(args.worker_cache_gb*(1<<30)),fixed=[fixed],onpolicy=caches,replay_index=str(collector.index))
    kwargs=dict(num_workers=args.workers,batch_size=None)
    if args.workers: kwargs.update(prefetch_factor=2,persistent_workers=True)
    loader=torch.utils.data.DataLoader(ds,**kwargs); it=iter(loader)
    if resume is None:
        print(f'Fitting residual scales from {args.flow_calibration_states} training states...',flush=True)
        model_cfg.flow_sigma,calibration=fit_flow_sigma(it,model_cfg,args.flow_calibration_states,
            progress=lambda seen:print(json.dumps(dict(calibration_states=seen,total=args.flow_calibration_states)),flush=True))
    else:
        model_cfg=FollowNetConfig(**resume['model_cfg']); calibration=resume['flow_calibration']
    model=prepare_model(FollowNet(model_cfg),args.device)
    ema=copy.deepcopy(model).requires_grad_(False).eval()
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    first=1; replay_seen=0
    if resume is not None:
        done,replay_seen=resume_training(resume,model,ema,opt); first=done+1
        del resume
    else: (out/'config.json').write_text(json.dumps(dict(vars(args),architecture=ARCHITECTURE,model_cfg=model_cfg.to_dict(),
         compile_model=compile_model,
         sample_cfg=dataclasses.asdict(sample_cfg),vol_spec=spec.to_dict(),flow_calibration=calibration,
         accumulation_steps=args.batch//args.microbatch,data_policy=DATA_POLICY,fiber_manifest=fiber_manifest(fibers),
         train=[f.name for f in train_f],val=[f.name for f in val_f],seed_manifest=manifest,
         cuda_channels_last_3d=(args.device.startswith('cuda') and
                               model.encoders[0][0].weight.is_contiguous(memory_format=torch.channels_last_3d)),
         cudnn_benchmark_training=args.device.startswith('cuda')),indent=2))
    if compile_model:
        print('Compiling CUDA training methods; the first update may take about a minute.',flush=True)
        compile_training_model(model)
    tracer=None
    seeds=manifest['monitor']
    if args.diag_every and seeds:
        from vesuvius.neural_tracing.fiber_follow.diag import plot_batch, plot_denoising, plot_curves, rollout_diag
        from vesuvius.neural_tracing.fiber_follow.trace import ModelTracer,TraceParams
        from vesuvius.neural_tracing.fiber_follow.volume import FiberVolume
        tracer=ModelTracer(ema,FiberVolume(spec,cache_bytes=2<<30),crop,128,TraceParams(n_commit=args.n_commit),device=args.device)
        (out/'images').mkdir(exist_ok=True)
    log=RunLog(out/'log.jsonl',formatter=format_training_log); start=time.monotonic()
    if first>1: log.record(dict(step=first,resumed_from=args.resume,replay_caches=len(caches),
                              compile_model=compile_model,cache_training_encoding=args.cache_training_encoding))
    try:
        for step in range(first,args.steps+1):
            event=collector.poll()
            if event: log.record(dict(step=step,**event))
            batches=[next(it) for _ in range(args.batch//args.microbatch)]
            log_step=step%args.log_every==0 or step==args.steps
            model.train(); torch.backends.cudnn.benchmark=args.device.startswith('cuda')
            lr=lr_at(step,args.lr,args.warmup,args.steps)
            loss,metrics,batch=optimizer_update(model,ema,opt,batches,step,lr,device=args.device,
                                                tolerance=args.tolerance,decay=args.ema_decay,compute_metrics=log_step,
                                                cache_training_encoding=args.cache_training_encoding,n_commit=args.n_commit)
            replay_seen+=metrics.pop('replay_samples')
            if log_step:
                log.record(dict(step=step,loss=loss,lr=lr,replay_samples_seen=replay_seen,
                                samples_per_second=(step-first+1)*args.batch/(time.monotonic()-start),**metrics))
            def save(path,resumable=False):
                extra=dict(step=step,tolerance=args.tolerance,n_commit=args.n_commit,seed=args.seed,ema_decay=args.ema_decay,ema_updates=step,
                           flow_calibration=calibration,seed_manifest_sha256=manifest['sha256'],replay_seen=replay_seen)
                if resumable:  # collector snapshots stay light; training checkpoints can continue the run
                    extra.update(optimizer=opt.state_dict(),rng=training_rng_state())
                save_checkpoint(path,model,ema,spec,sample_cfg,extra)
            if step<args.steps and collector.launch(step,save): log.record(dict(step=step,dagger_launched=True))
            if tracer is not None and step%args.diag_every==0:
                torch.backends.cudnn.benchmark=False
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=args.device.startswith('cuda')):
                    diagnostic=ema(batch['x'].float(),batch['hist'],batch['hmask'],return_steps=True)
                pred=diagnostic['points']; gt=torch.cat([batch['plane_ab'],pred[...,2:]],-1)
                plot_batch(batch['x'],pred,gt,batch['plane_mask'],crop,out/'images'/f'batch_{step:06d}.png',
                           batch['hist'],batch['hmask'],batch['gt_history'],batch['gt_history_mask'],
                           source=batch['source'],offtrack=batch['offtrack'],confidence=diagnostic['confidence'])
                plot_denoising(diagnostic['denoising_steps'],batch['hist'],batch['hmask'],out/'images'/f'denoising_{step:06d}.png')
                for threshold in (.5,.85):
                    tracer.p.confidence=threshold
                    summary=rollout_diag(tracer,val_f,seeds,out/'images'/f'rollout_{step:06d}_c{threshold}.png',batch=args.diag_batch)
                    log.record(dict(step=step,threshold=threshold,roll_coverage=summary['coverage_mean'],
                                    roll_precision=summary['length_precision'],roll_diverged=summary['diverged']))
                plot_curves(out/'log.jsonl',out/'curves.png')
            if step%args.ckpt_every==0 or step==args.steps:
                save(out/f'ckpt_{step:06d}.pt',resumable=True); save(out/'last.pt',resumable=True)
    finally:
        event=collector.close()
        if event: log.record(event)
        if tracer is not None: tracer.close()
        log.close()
    return str(out/'last.pt')

if __name__=='__main__':
    main()
