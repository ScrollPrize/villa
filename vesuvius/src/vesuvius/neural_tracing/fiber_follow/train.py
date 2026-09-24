"""Supervised spatial fiber follower with asynchronous, dense-GT DAgger replay."""
from __future__ import annotations

import argparse
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
from vesuvius.neural_tracing.fiber_follow.model import ARCHITECTURE, FollowNet, FollowNetConfig, prepare_model
from vesuvius.neural_tracing.fiber_follow.online import OnlineCollector
from vesuvius.neural_tracing.fiber_follow.trace import DEFAULT_CONFIDENCE
from vesuvius.neural_tracing.fiber_follow.supervision import loss_fn, teacher_candidates
from vesuvius.neural_tracing.fiber_follow.volume import FiberVolumeSpec


def save_checkpoint(path, model, vol_spec, sample_cfg, extra=None):
    torch.save(dict(architecture=ARCHITECTURE, data_policy=DATA_POLICY, model=model.state_dict(),
                    model_cfg=model.cfg.to_dict(), crop=dataclasses.asdict(sample_cfg.crop),
                    n_history=sample_cfg.n_history, vol_spec=vol_spec.to_dict(), **(extra or {})), path)


def read_checkpoint(path, device='cuda'):
    ck = torch.load(path, map_location=device, weights_only=False)
    if ck['architecture'] != ARCHITECTURE or ck['data_policy'] != DATA_POLICY:
        raise ValueError('Checkpoint does not match the current architecture and supervision contract')
    return ck


def load_checkpoint(path, device='cuda'):
    ck = read_checkpoint(path, device)
    cfg = dict(ck['model_cfg'])
    cfg['widths'] = tuple(cfg['widths'])
    model = prepare_model(FollowNet(FollowNetConfig(**cfg)), device)
    model.load_state_dict(ck['model'])
    return model, CropSpec(**ck['crop']), ck['n_history'], FiberVolumeSpec(**ck['vol_spec']), ck


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--fiber-zarrs', required=True)
    ap.add_argument('--fibers', required=True)
    ap.add_argument('--ct')
    ap.add_argument('--ct-level', type=int, default=1)
    ap.add_argument('--ct-grid-scale', type=float, default=8., help='Base voxels per selected CT voxel; s1_ds2 level 0 uses 4')
    ap.add_argument('--inputs', choices=('fiber', 'fiber+ct', 'ct'))
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
    ap.add_argument('--allow-history-change', action='store_true', help='Allow --init to change only history rendering settings')
    ap.add_argument('--warmup', type=int, default=150)
    ap.add_argument('--ckpt-every', type=int, default=2000)
    ap.add_argument('--log-every', type=int, default=50)
    ap.add_argument('--diag-every', type=int, default=500, help='0 disables validation rollouts and images')
    ap.add_argument('--diag-seeds', type=int, default=16)
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
    ap.add_argument('--clean-points', type=int, default=8, help='Number of annotated past points to reconstruct with the current point')
    ap.add_argument('--clean-weight', type=float, default=.5)
    ap.add_argument('--heat-bins', type=int, default=61)
    ap.add_argument('--heat-spacing', type=float, default=1., help='Trace-grid voxels per lateral heatmap sample')
    ap.add_argument('--heatmap-target', choices=('planes', 'tube'), default='planes')
    ap.add_argument('--tube-sigma', type=float, default=.7, help='Gaussian tube sigma in trace-grid voxels')
    ap.add_argument('--n-candidates', type=int, default=4)
    ap.add_argument('--widths', type=int, nargs='+', default=(24, 48, 96))
    ap.add_argument('--hidden', type=int, default=96)
    ap.add_argument('--norm', choices=('batch', 'group'), default='batch')
    ap.add_argument('--lateral-sigmas', type=float, nargs=3, default=(.4, 1., 2.))
    ap.add_argument('--angle-sigmas', type=float, nargs=3, default=(4., 10., 20.))
    ap.add_argument('--history-render', choices=('points', 'segments'), default='points')
    ap.add_argument('--history-sigma', type=float, default=1., help='History Gaussian sigma in trace-grid voxels')
    ap.add_argument('--history-jitter', type=float, default=.25, help='Independent history point jitter; 0 keeps only smooth drift')
    ap.add_argument('--history-wobble', type=float, default=1.)
    ap.add_argument('--tolerance', type=float, default=1.5, help='Candidate correctness radius, grid voxels')
    ap.add_argument('--rank-weight', type=float, default=1.)
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
    if min(args.steps, args.batch, args.ckpt_every, args.log_every, args.replay_keep) < 1:
        ap.error('Steps, batch, checkpoint/log intervals and replay-keep must be positive')
    if min(args.dagger_every, args.diag_every, args.workers) < 0 or args.dagger_batch < 1 or args.dagger_seeds < 1 or args.tolerance <= 0:
        ap.error('Invalid collection, diagnostic, worker, or tolerance settings')
    if args.n_history < args.hist_points*args.hist_stride:
        ap.error('n-history must cover hist-points * hist-stride')
    if args.clean_points < 1 or args.clean_points > args.hist_points*args.hist_stride or args.clean_weight < 0:
        ap.error('clean-points must be positive, fit in history, and clean-weight must be nonnegative')
    if min(args.crop_spacing, args.heat_spacing, args.ct_grid_scale, args.tube_sigma) <= 0:
        ap.error('Voxel spacings and tube sigma must be positive')
    if not math.isfinite(args.history_sigma) or args.history_sigma <= 0 or not math.isfinite(args.history_jitter) or args.history_jitter < 0:
        ap.error('History sigma must be positive and jitter nonnegative')
    if args.allow_history_change and not args.init:
        ap.error('--allow-history-change requires --init')
    out = Path(args.out_root)/args.name
    if (out/'config.json').exists():
        raise FileExistsError(f'{out} already contains a run; use a new --name')
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    spec = FiberVolumeSpec(args.fiber_zarrs, ct_zarr=args.ct, ct_level=args.ct_level, ct_grid_scale=args.ct_grid_scale,
                           inputs=args.inputs or ('fiber+ct' if args.ct else 'fiber'))
    crop = CropSpec(depth=args.crop_depth, width=args.crop_width, behind=args.crop_behind,
                    spacing=args.crop_spacing, gate_direction=args.gate_direction,
                    history_render=args.history_render, history_sigma=args.history_sigma)
    sample_cfg = SampleConfig(crop=crop, n_future=args.n_future, future_step=args.future_step, n_candidates=args.n_candidates,
                              n_history=args.n_history, clean_points=args.clean_points,
                              lateral_sigmas=tuple(args.lateral_sigmas),
                              angle_sigmas_deg=tuple(args.angle_sigmas), history_wobble=args.history_wobble, history_jitter=args.history_jitter,
                              heatmap_target=args.heatmap_target, tube_sigma=args.tube_sigma)
    model_cfg = FollowNetConfig(in_channels={'fiber':8, 'fiber+ct':9, 'ct':2}[spec.mode],
                                depth=crop.depth, width=crop.width, behind=crop.behind,
                                spacing=crop.spacing, widths=tuple(args.widths), hidden=args.hidden,
                                n_future=args.n_future, future_step=args.future_step, hist_points=args.hist_points,
                                hist_stride=args.hist_stride, heat_bins=args.heat_bins, n_candidates=args.n_candidates,
                                clean_points=args.clean_points,
                                norm=args.norm, heat_spacing=args.heat_spacing,
                                heatmap_target=args.heatmap_target, tube_sigma=args.tube_sigma)
    model = prepare_model(FollowNet(model_cfg), args.device)
    if args.init:
        initial, initial_crop, nh, initial_spec, _ = load_checkpoint(args.init, args.device)
        if args.allow_history_change:
            initial_crop = dataclasses.replace(initial_crop, history_render=crop.history_render, history_sigma=crop.history_sigma)
        if initial.cfg != model_cfg or initial_crop != crop or nh != args.n_history or initial_spec != spec:
            raise ValueError('Initialization checkpoint configuration must exactly match this run')
        model.load_state_dict(initial.state_dict())
        del initial
    fibers = load_fibers(args.fibers, grid_scale=spec.grid_scale)
    band = ZBand(args.val_z[0]/spec.grid_scale, args.val_z[1]/spec.grid_scale)
    train_f, val_f = split_fibers(fibers, band)
    caches = [OnPolicyStates.load(p) for p in args.onpolicy]
    for cache in caches:
        cache.validate_fibers(train_f)
    (out/'config.json').write_text(json.dumps(dict(vars(args), architecture=ARCHITECTURE,
                                  cuda_channels_last_3d=args.device.startswith('cuda'),
                                  cudnn_benchmark_training=args.device.startswith('cuda'),
                                  data_policy=DATA_POLICY, fiber_manifest=fiber_manifest(fibers),
                                  train=[f.name for f in train_f], val=[f.name for f in val_f]), indent=2))
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
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    tracer = None
    if args.diag_every:
        from vesuvius.neural_tracing.fiber_follow.diag import plot_batch, plot_curves, plot_tube, rollout_diag
        from vesuvius.neural_tracing.fiber_follow.evaluate import make_seeds
        from vesuvius.neural_tracing.fiber_follow.trace import ModelTracer, TraceParams
        from vesuvius.neural_tracing.fiber_follow.volume import FiberVolume
        vol = FiberVolume(spec, cache_bytes=2 << 30)
        rng = np.random.default_rng(123)
        diag_fibers = [val_f[i] for i in rng.choice(len(val_f), min(args.diag_seeds, len(val_f)), replace=False)]
        seeds = make_seeds(diag_fibers, vol, per_fiber=1, seed=123)[::2][:args.diag_seeds]
        tracer = ModelTracer(model, vol, crop, args.n_history, TraceParams(confidence=args.confidence), device=args.device)
        (out/'images').mkdir(exist_ok=True)
    print(json.dumps(dict(train_fibers=len(train_f), val_fibers=len(val_f), parameters=sum(p.numel() for p in model.parameters()))), flush=True)
    it = iter(dl)
    start = time.monotonic()
    log = (out/'log.jsonl').open('a')
    def record(values):
        print(json.dumps(values), flush=True)
        log.write(json.dumps(values)+'\n')
        log.flush()
    replay_seen = 0
    try:
        for step in range(1, args.steps+1):
            event = collector.poll()
            if event:
                record(dict(step=step, **event))
            parts = [next(it) for _ in range(args.batch//chunk)]
            batch = {k: torch.cat([p[k] for p in parts]).to(args.device) for k in parts[0]}
            replay_seen += int((batch['source'] > 0).sum().item())
            lr = args.lr*min(1., step/max(1, args.warmup))*.5*(1+math.cos(math.pi*(step-1)/args.steps))
            for group in opt.param_groups:
                group['lr'] = lr
            model.train()
            # Fixed training shapes benefit from cuDNN's cached kernel search.
            # Rollouts below disable it because their active batch shrinks.
            torch.backends.cudnn.benchmark = args.device.startswith('cuda')
            extras = teacher_candidates(batch, model.cfg)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=args.device.startswith('cuda')):
                output = model(batch['x'].float(), batch['hist'], batch['hmask'], extras)
            loss, metrics = loss_fn(output, batch, model.cfg, args.tolerance, args.rank_weight,
                                    args.confidence_weight, args.clean_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite training loss at step {step}')
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            opt.step()
            if step % args.log_every == 0 or step == args.steps:
                record(dict(step=step, loss=loss.item(), lr=lr, replay_samples_seen=replay_seen,
                            fresh_fraction=(batch['source'] == 0).float().mean().item(),
                            replay_fraction=(batch['source'] == 1).float().mean().item(),
                            hard_fraction=(batch['source'] == 2).float().mean().item(), samples_per_second=step*args.batch/(time.monotonic()-start), **metrics))
            save = lambda path: save_checkpoint(path, model, spec, sample_cfg,
                                                dict(step=step, tolerance=args.tolerance, seed=args.seed))
            if step < args.steps and collector.launch(step, save):
                record(dict(step=step, dagger_launched=True))
            if args.diag_every and step % args.diag_every == 0 and seeds:
                torch.backends.cudnn.benchmark = False
                # Plot only model proposals, excluding GT candidates used to teach the scorer.
                ranks = output['ranks'][:, :model.cfg.n_candidates]
                chosen = ranks.argmax(-1)
                pred = output['candidates'][torch.arange(len(chosen), device=chosen.device), chosen]
                gt = torch.cat([batch['plane_ab'], pred[..., 2:]], -1)
                plot_batch(batch['x'], pred, gt, batch['plane_mask'], crop, out/'images'/f'batch_{step:06d}.png',
                           output['clean_history'], batch['clean_local'], batch['clean_mask'],
                           heat_half=model.plane_grid[:, -1, -1, 0].cpu().numpy(),
                           source=batch['source'], offtrack=batch['offtrack'],
                           confidence=output['confidence'][torch.arange(len(chosen), device=chosen.device), chosen])
                if args.heatmap_target == 'tube':
                    plot_tube(batch['x'], batch['tube_target'], batch['tube_mask'], output['tube_logits'],
                              out/'images'/f'tube_{step:06d}.png', offtrack=batch['offtrack'])
                summ = rollout_diag(tracer, diag_fibers, seeds, out/'images'/f'rollout_{step:06d}.png')
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
