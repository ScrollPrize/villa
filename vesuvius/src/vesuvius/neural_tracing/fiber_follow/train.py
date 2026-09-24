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
from vesuvius.neural_tracing.fiber_follow.model import ARCHITECTURE, FollowNet, FollowNetConfig, prepare_model, prior_mean
from vesuvius.neural_tracing.fiber_follow.online import OnlineCollector
from vesuvius.neural_tracing.fiber_follow.runloop import (
    RunLog, lr_at, optimizer_step, prepare_run_dir,
    read_checkpoint as _read_checkpoint, save_checkpoint as _save_checkpoint,
)
from vesuvius.neural_tracing.fiber_follow.trace import DEFAULT_CONFIDENCE
from vesuvius.neural_tracing.fiber_follow.supervision import loss_fn, teacher_candidates
from vesuvius.neural_tracing.fiber_follow.policy import DEFAULT_MAX_RECOVERY_DISTANCE, choose_candidate
from vesuvius.neural_tracing.fiber_follow.volume import FiberVolumeSpec


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
def fit_flow_sigma(batches, cfg, states):
    """Fit masked residual standard deviations using only the training loader."""
    count = torch.zeros(cfg.n_future, dtype=torch.float64)
    total = torch.zeros(cfg.n_future, 2, dtype=torch.float64)
    square = torch.zeros_like(total)
    seen = 0
    while seen < states:
        batch = next(batches)
        n = min(len(batch['hist']), states-seen)
        mu = prior_mean(batch['hist'][:n], batch['hmask'][:n], cfg)[..., :2].double()
        known = batch['plane_mask'][:n].bool() & ~batch['offtrack'][:n, None].bool()
        residual = torch.where(known[..., None], batch['plane_ab'][:n].double()-mu, 0.)
        if not torch.isfinite(residual).all():
            raise ValueError('Non-finite annotated residual during flow scale calibration')
        count += known.sum(0)
        total += residual.sum(0)
        square += residual.square().sum(0)
        seen += n
    if (count < 2).any():
        raise ValueError('Flow scale calibration needs at least two known targets on every plane; '
                         'increase --flow-calibration-states')
    mean = total / count[:, None]
    std = (square / count[:, None] - mean.square()).clamp_min(0).sqrt()
    sigma = std.clamp_min(1.)
    return tuple(map(tuple, sigma.tolist())), dict(states=seen, known_counts=count.long().tolist(),
                                                  residual_mean=mean.tolist(), residual_std=std.tolist())


@torch.no_grad()
def update_ema(ema, model, step, decay):
    """Ramp averaging in early updates so initial collectors do not lag badly."""
    effective_decay = min(decay, (1+step)/(10+step))
    for average, current in zip(ema.parameters(), model.parameters(), strict=True):
        average.lerp_(current.detach(), 1-effective_decay)
    for average, current in zip(ema.buffers(), model.buffers(), strict=True):
        average.copy_(current)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--fiber-zarrs', required=True)
    ap.add_argument('--fibers', required=True)
    ap.add_argument('--ct')
    ap.add_argument('--ct-level', type=int, default=1)
    ap.add_argument('--ct-grid-scale', type=float, default=8., help='Base voxels per selected CT voxel; s1_ds2 level 0 uses 4')
    ap.add_argument('--inputs', choices=('fiber', 'fiber+ct', 'ct', 'ct+presence'))
    ap.add_argument('--name', required=True)
    ap.add_argument('--out-root', default=str(Path(__file__).parent/'output'))
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--steps', type=int, default=10000)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--worker-cache-gb', type=float, default=1.)
    ap.add_argument('--val-z', type=float, nargs=2, default=(45000., 48500.))
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--init', help='Exact current-architecture weights; starts a new optimizer')
    ap.add_argument('--warmup', type=int, default=1000)
    ap.add_argument('--ema-decay', type=float, default=.999)
    ap.add_argument('--ckpt-every', type=int, default=2000)
    ap.add_argument('--log-every', type=int, default=50)
    ap.add_argument('--diag-every', type=int, default=500, help='0 disables validation rollouts and images')
    ap.add_argument('--diag-seeds', type=int, default=16)
    ap.add_argument('--diag-batch', type=int, default=8, help='Maximum simultaneous diagnostic traces')
    ap.add_argument('--crop-depth', type=int, default=64)
    ap.add_argument('--crop-width', type=int, default=64)
    ap.add_argument('--crop-behind', type=int, default=16)
    ap.add_argument('--crop-spacing', type=float, default=1., help='Trace-grid voxels per crop sample')
    ap.add_argument('--gate-direction', action='store_true')
    ap.add_argument('--n-future', type=int, default=16)
    ap.add_argument('--future-step', type=float, default=2.)
    ap.add_argument('--n-history', type=int, default=128)
    ap.add_argument('--hist-points', type=int, default=32)
    ap.add_argument('--hist-stride', type=int, default=4)
    ap.add_argument('--recent-history-points', type=int, default=8, help='Dense observed past points conditioning the flow and scorer')
    ap.add_argument('--flow-layers', type=int, default=4, help='Transformer blocks of the future-path flow')
    ap.add_argument('--flow-heads', type=int, default=4)
    ap.add_argument('--flow-steps', type=int, default=4, help='Midpoint steps from the prior; two flow evaluations per step')
    ap.add_argument('--flow-samples', type=int, default=16, help='Future samples per state; every sample is scored')
    ap.add_argument('--flow-draws', type=int, default=32, help='(time, noise) draws per example in the flow loss')
    ap.add_argument('--flow-calibration-states', type=int, default=2048, help='Training states used once to fit residual scales')
    ap.add_argument('--flow-stencil-radius', type=float, default=2., help='3x3 flow patch pitch/radius in trace-grid voxels')
    ap.add_argument('--support-radius', type=float, default=1.5, help='RMS lateral radius for candidate sample support')
    ap.add_argument('--max-recovery-distance', type=float, default=DEFAULT_MAX_RECOVERY_DISTANCE,
                    help='Maximum origin-to-first-point connection length in trace-grid voxels')
    ap.add_argument('--flow-weight', type=float, default=1.)
    ap.add_argument('--widths', type=int, nargs='+', default=(24, 48, 96))
    ap.add_argument('--hidden', type=int, default=96)
    ap.add_argument('--norm', choices=('batch', 'group'), default='group')
    ap.add_argument('--lateral-sigmas', type=float, nargs=3, default=(.4, 1., 2.))
    ap.add_argument('--angle-sigmas', type=float, nargs=3, default=(4., 10., 20.))
    ap.add_argument('--history-render', choices=('points', 'segments'), default='points')
    ap.add_argument('--history-sigma', type=float, default=1., help='History Gaussian sigma in trace-grid voxels')
    ap.add_argument('--history-jitter', type=float, default=.25, help='Independent history point jitter; 0 keeps only smooth drift')
    ap.add_argument('--history-wobble', type=float, default=1.)
    ap.add_argument('--tolerance', type=float, default=1.5, help='Candidate correctness radius, grid voxels')
    ap.add_argument('--rank-weight', type=float, default=1.)
    ap.add_argument('--rank-temperature', type=float, default=20., help='Softmax temperature on candidate quality for the ranking target')
    ap.add_argument('--confidence-weight', type=float, default=1.)
    ap.add_argument('--confidence', type=float, default=DEFAULT_CONFIDENCE, help='Rollout confidence threshold; tune on validation')
    ap.add_argument('--onpolicy', nargs='+', default=[], help='Existing decision caches, oldest to newest')
    ap.add_argument('--onpolicy-prob', type=float, default=.3)
    ap.add_argument('--hard-prob', type=float, default=.2)
    ap.add_argument('--dagger-every', type=int, default=500, help='Snapshot cadence; 0 disables background collection')
    ap.add_argument('--dagger-device', help='Defaults to training device; may use another GPU or cpu')
    ap.add_argument('--dagger-seeds', type=int, default=64)
    ap.add_argument('--dagger-batch', type=int, default=8)
    ap.add_argument('--dagger-explore-calls', type=int, default=8)
    ap.add_argument('--dagger-trace-len', type=float, default=6000.)
    ap.add_argument('--replay-keep', type=int, default=4, help='Recent completed collections retained in active replay')
    args = ap.parse_args(argv)
    if min(args.steps, args.batch, args.ckpt_every, args.log_every, args.replay_keep, args.diag_batch) < 1:
        ap.error('Steps, batch, checkpoint/log intervals and replay-keep must be positive')
    if min(args.dagger_every, args.diag_every, args.workers) < 0 or args.dagger_batch < 1 or args.dagger_seeds < 1 or args.tolerance <= 0:
        ap.error('Invalid collection, diagnostic, worker, or tolerance settings')
    if args.n_history < args.hist_points*args.hist_stride:
        ap.error('n-history must cover hist-points * hist-stride')
    if args.recent_history_points < 1 or args.recent_history_points > args.hist_points*args.hist_stride:
        ap.error('recent-history-points must be positive and fit in history')
    if min(args.flow_layers, args.flow_heads, args.flow_steps, args.flow_draws, args.flow_samples) < 1:
        ap.error('Flow layers/heads/steps/draws/samples must be positive')
    if args.flow_calibration_states < 2 or not 0 <= args.ema_decay < 1 or args.warmup < 0:
        ap.error('Calibration needs at least two states, EMA decay in [0, 1), and nonnegative warmup')
    if not math.isfinite(args.support_radius) or args.support_radius <= 0 or args.flow_weight < 0:
        ap.error('support-radius must be positive and finite; flow-weight nonnegative')
    if not math.isfinite(args.flow_stencil_radius) or not 0 < args.flow_stencil_radius <= (args.crop_width-1)*args.crop_spacing/2:
        ap.error('flow-stencil-radius must be positive and fit inside the crop')
    if not math.isfinite(args.max_recovery_distance) or not 0 < args.future_step <= args.max_recovery_distance:
        ap.error('max-recovery-distance must be finite, positive, and at least future-step')
    if min(args.crop_spacing, args.ct_grid_scale) <= 0:
        ap.error('Voxel spacings must be positive')
    if not math.isfinite(args.history_sigma) or args.history_sigma <= 0 or not math.isfinite(args.history_jitter) or args.history_jitter < 0:
        ap.error('History sigma must be positive and jitter nonnegative')
    out = prepare_run_dir(args.out_root, args.name)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    spec = FiberVolumeSpec(args.fiber_zarrs, ct_zarr=args.ct, ct_level=args.ct_level, ct_grid_scale=args.ct_grid_scale,
                           inputs=args.inputs or ('fiber+ct' if args.ct else 'fiber'))
    crop = CropSpec(depth=args.crop_depth, width=args.crop_width, behind=args.crop_behind,
                    spacing=args.crop_spacing, gate_direction=args.gate_direction,
                    history_render=args.history_render, history_sigma=args.history_sigma)
    sample_cfg = SampleConfig(crop=crop, n_future=args.n_future, future_step=args.future_step, n_candidates=args.flow_samples,
                              n_history=args.n_history, recent_history_points=args.recent_history_points,
                              lateral_sigmas=tuple(args.lateral_sigmas),
                              angle_sigmas_deg=tuple(args.angle_sigmas), history_wobble=args.history_wobble, history_jitter=args.history_jitter)
    model_cfg = FollowNetConfig(in_channels={'fiber':8, 'fiber+ct':9, 'ct':2, 'ct+presence':3}[spec.mode],
                                depth=crop.depth, width=crop.width, behind=crop.behind,
                                spacing=crop.spacing, widths=tuple(args.widths), hidden=args.hidden,
                                n_future=args.n_future, future_step=args.future_step, hist_points=args.hist_points,
                                hist_stride=args.hist_stride,
                                recent_history_points=args.recent_history_points, norm=args.norm,
                                flow_layers=args.flow_layers, flow_heads=args.flow_heads, flow_steps=args.flow_steps,
                                flow_samples=args.flow_samples, flow_draws=args.flow_draws,
                                flow_stencil_radius=args.flow_stencil_radius, support_radius=args.support_radius,
                                max_recovery_distance=args.max_recovery_distance)
    fibers = load_fibers(args.fibers, grid_scale=spec.grid_scale)
    band = ZBand(args.val_z[0]/spec.grid_scale, args.val_z[1]/spec.grid_scale)
    train_f, val_f = split_fibers(fibers, band)
    caches = [OnPolicyStates.load(p) for p in args.onpolicy]
    for cache in caches:
        cache.validate_fibers(train_f)
    collector = OnlineCollector(out/'dagger', args.fibers, args.val_z, args.dagger_device or args.device,
                                every=args.dagger_every, max_seeds=args.dagger_seeds, batch=args.dagger_batch,
                                explore_calls=args.dagger_explore_calls, seed=args.seed, replay_keep=args.replay_keep,
                                initial=[c._dir for c in caches], trace_len=args.dagger_trace_len, confidence=args.confidence)
    chunk = min(16, args.batch)
    if args.batch % chunk:
        chunk = math.gcd(args.batch, chunk)
    ds = FollowDataset(train_f, spec, sample_cfg, band, chunk=chunk, seed=args.seed,
                       cache_bytes=int(args.worker_cache_gb*(1 << 30)), onpolicy=caches,
                       onpolicy_prob=args.onpolicy_prob, hard_prob=args.hard_prob, replay_index=str(collector.index))
    kwargs = dict(num_workers=args.workers, batch_size=None)
    if args.workers:
        kwargs.update(prefetch_factor=2, persistent_workers=True)
    dl = torch.utils.data.DataLoader(ds, **kwargs)
    it = iter(dl)
    initial = None
    ema_updates = 0
    if args.init:
        initial = read_checkpoint(args.init, args.device)
        model_cfg.flow_sigma = FollowNetConfig(**initial['model_cfg']).flow_sigma
        if (FollowNetConfig(**initial['model_cfg']) != model_cfg
                or initial['sample_cfg'] != dataclasses.asdict(sample_cfg)
                or FiberVolumeSpec(**initial['vol_spec']) != spec):
            raise ValueError('Initialization checkpoint configuration must exactly match this run')
        calibration = initial['flow_calibration']
        ema_updates = initial['ema_updates']
    else:
        print(f'Calibrating flow scales from {args.flow_calibration_states} training states...', flush=True)
        model_cfg.flow_sigma, calibration = fit_flow_sigma(it, model_cfg, args.flow_calibration_states)
        print(json.dumps(dict(flow_sigma=model_cfg.flow_sigma, flow_calibration=calibration)), flush=True)
    model = prepare_model(FollowNet(model_cfg), args.device)
    if initial is not None:
        model.load_state_dict(initial['model'])
    ema = copy.deepcopy(model).requires_grad_(False).eval()
    if initial is not None:
        ema.load_state_dict(initial['ema'])
    del initial
    (out/'config.json').write_text(json.dumps(dict(vars(args), architecture=ARCHITECTURE,
                                  model_cfg=model_cfg.to_dict(), flow_calibration=calibration,
                                  cuda_channels_last_3d=args.device.startswith('cuda'),
                                  cudnn_benchmark_training=args.device.startswith('cuda'),
                                  data_policy=DATA_POLICY, fiber_manifest=fiber_manifest(fibers),
                                  train=[f.name for f in train_f], val=[f.name for f in val_f]), indent=2))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    tracer = None
    if args.diag_every:
        from vesuvius.neural_tracing.fiber_follow.diag import plot_batch, plot_curves, rollout_diag
        from vesuvius.neural_tracing.fiber_follow.evaluate import make_seeds
        from vesuvius.neural_tracing.fiber_follow.trace import ModelTracer, TraceParams
        from vesuvius.neural_tracing.fiber_follow.volume import FiberVolume
        vol = FiberVolume(spec, cache_bytes=2 << 30)
        rng = np.random.default_rng(123)
        diag_fibers = [val_f[i] for i in rng.choice(len(val_f), min(args.diag_seeds, len(val_f)), replace=False)]
        seeds = make_seeds(diag_fibers, vol, per_fiber=1, seed=123)[::2][:args.diag_seeds]
        tracer = ModelTracer(ema, vol, crop, args.n_history, TraceParams(confidence=args.confidence), device=args.device)
        (out/'images').mkdir(exist_ok=True)
    print(json.dumps(dict(train_fibers=len(train_f), val_fibers=len(val_f), parameters=sum(p.numel() for p in model.parameters()))), flush=True)
    start = time.monotonic()
    log = RunLog(out/'log.jsonl')
    record = log.record
    replay_seen = 0
    try:
        for step in range(1, args.steps+1):
            event = collector.poll()
            if event:
                record(dict(step=step, **event))
            parts = [next(it) for _ in range(args.batch//chunk)]
            batch = {k: torch.cat([p[k] for p in parts]).to(args.device) for k in parts[0]}
            replay_seen += int((batch['source'] > 0).sum().item())
            lr = lr_at(step, args.lr, args.warmup, args.steps)
            model.train()
            # Fixed training shapes benefit from cuDNN's cached kernel search.
            # Rollouts below disable it because their active batch shrinks.
            torch.backends.cudnn.benchmark = args.device.startswith('cuda')
            extras = teacher_candidates(batch, model.cfg)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=args.device.startswith('cuda')):
                output = model(batch['x'].float(), batch['hist'], batch['hmask'], extras, targets=batch)
            loss, metrics = loss_fn(output, batch, model.cfg, args.tolerance, args.rank_weight,
                                    args.confidence_weight, args.flow_weight, args.rank_temperature,
                                    confidence_threshold=args.confidence)
            optimizer_step(model, opt, loss, step, lr)
            update_ema(ema, model, ema_updates+step, args.ema_decay)
            if step % args.log_every == 0 or step == args.steps:
                record(dict(step=step, loss=loss.item(), lr=lr, replay_samples_seen=replay_seen,
                            fresh_fraction=(batch['source'] == 0).float().mean().item(),
                            replay_fraction=(batch['source'] == 1).float().mean().item(),
                            hard_fraction=(batch['source'] == 2).float().mean().item(), samples_per_second=step*args.batch/(time.monotonic()-start), **metrics))
            save = lambda path: save_checkpoint(path, model, ema, spec, sample_cfg,
                                                dict(step=step, tolerance=args.tolerance, seed=args.seed,
                                                     ema_decay=args.ema_decay, ema_updates=ema_updates+step,
                                                     flow_calibration=calibration))
            if step < args.steps and collector.launch(step, save):
                record(dict(step=step, dagger_launched=True))
            if args.diag_every and step % args.diag_every == 0 and seeds:
                torch.backends.cudnn.benchmark = False
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=args.device.startswith('cuda')):
                    diagnostic = ema(batch['x'].float(), batch['hist'], batch['hmask'])
                ranks = diagnostic['ranks'][:, :model.cfg.flow_samples]
                chosen, _, _ = choose_candidate(
                    diagnostic['candidates'][:, :model.cfg.flow_samples], ranks,
                    diagnostic['confidence'][:, :model.cfg.flow_samples], args.confidence,
                    max_distance=model.cfg.max_recovery_distance)
                pred = diagnostic['candidates'][torch.arange(len(chosen), device=chosen.device), chosen]
                gt = torch.cat([batch['plane_ab'], pred[..., 2:]], -1)
                plot_batch(batch['x'], pred, gt, batch['plane_mask'], crop, out/'images'/f'batch_{step:06d}.png',
                           batch['hist'], batch['hmask'], batch['gt_history'], batch['gt_history_mask'],
                           source=batch['source'], offtrack=batch['offtrack'],
                           confidence=diagnostic['confidence'][torch.arange(len(chosen), device=chosen.device), chosen])
                summ = rollout_diag(tracer, diag_fibers, seeds, out/'images'/f'rollout_{step:06d}.png', batch=args.diag_batch)
                record(dict(step=step, roll_coverage=summ['coverage_mean'], roll_diverged=summ['diverged'],
                            roll_precision=summ['length_precision'], roll_unknown_fraction=summ['unknown_length_fraction']))
                plot_curves(out/'log.jsonl', out/'curves.png')
            if step % args.ckpt_every == 0 or step == args.steps:
                save(out/f'ckpt_{step:06d}.pt')
                save(out/'last.pt')
    finally:
        event = collector.close()
        if event:
            record(event)
        if tracer is not None:
            tracer.close()
        log.close()
    return str(out/'last.pt')


if __name__ == '__main__':
    main()
