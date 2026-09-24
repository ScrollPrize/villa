"""Train a re-ranker for the volume-cartographer fiber beam search.

The C++ beam (via ``vc.fiber_trace``) runs inside every loader worker over
controlled spans of training fibers, recording its prune-time candidate pools.
The model scores each pool from a CT crop and is supervised by dense GT.
Diagnostics report VC's own restart metric on held-out fibers with and without
the model in the loop.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from vesuvius.neural_tracing.fiber_follow.beam.data import BeamDataset
from vesuvius.neural_tracing.fiber_follow.beam.model import ARCHITECTURE, BeamRankNet, BeamNetConfig
from vesuvius.neural_tracing.fiber_follow.beam.native import BeamSpec, NativeBeam
from vesuvius.neural_tracing.fiber_follow.beam.states import BeamStateConfig
from vesuvius.neural_tracing.fiber_follow.beam.supervision import beam_loss
from vesuvius.neural_tracing.fiber_follow.data import DATA_POLICY, ZBand, fiber_manifest, load_fibers, split_fibers
from vesuvius.neural_tracing.fiber_follow.geometry import CropSpec
from vesuvius.neural_tracing.fiber_follow.model import prepare_model
from vesuvius.neural_tracing.fiber_follow.runloop import (
    RunLog, lr_at, optimizer_step, prepare_run_dir, read_checkpoint, save_checkpoint,
)
from vesuvius.neural_tracing.fiber_follow.volume import FiberVolumeSpec

STATE_KEYS = ('n_history', 'k_back', 'k_fwd', 'pool_size', 'tolerance', 'offtrack_distance', 'progress_slack',
              'heatmap_target', 'tube_sigma', 'lateral_sigmas', 'lateral_probs', 'angle_sigmas_deg', 'angle_probs')


def state_config_dict(cfg: BeamStateConfig) -> dict:
    return {k: getattr(cfg, k) for k in STATE_KEYS}


def load_beam_checkpoint(path, device='cuda'):
    """(model, BeamStateConfig, FiberVolumeSpec, BeamSpec, checkpoint dict)."""
    ck = read_checkpoint(path, ARCHITECTURE, device)
    cfg = dict(ck['model_cfg'])
    cfg['widths'] = tuple(cfg['widths'])
    model = prepare_model(BeamRankNet(BeamNetConfig(**cfg)), device)
    model.load_state_dict(ck['model'])
    state = {k: (tuple(v) if isinstance(v, list) else v) for k, v in ck['state_cfg'].items()}
    state_cfg = BeamStateConfig(crop=CropSpec(**ck['crop']), **state)
    return model, state_cfg, FiberVolumeSpec(**ck['vol_spec']), BeamSpec.from_dict(ck['beam_spec']), ck


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--fiber-zarrs', required=True, help='Presence zarr dir (seed/eval only in CT mode)')
    ap.add_argument('--fibers', required=True)
    ap.add_argument('--ct', required=True)
    ap.add_argument('--ct-level', type=int, default=0)
    ap.add_argument('--ct-grid-scale', type=float, default=4., help='Base voxels per CT voxel; s1_ds2 level 0 uses 4')
    ap.add_argument('--prediction-manifest', required=True, help='Lasagna fiber-prediction manifest the beam scores')
    ap.add_argument('--normal-manifest', help='Lasagna normal manifest (required by VC default smoothness)')
    ap.add_argument('--beam-cache-gb', type=float, default=.5)
    ap.add_argument('--scaledown-power', type=int, default=2, help='VC trace voxel = prediction voxel / 2**power')
    ap.add_argument('--beam-config', default='{}', help='JSON TraceConfig overrides (VC key names)')
    ap.add_argument('--hook-every-rounds', type=int, default=4)
    ap.add_argument('--pool-size', type=int, default=32)
    ap.add_argument('--k-back', type=int, default=16)
    ap.add_argument('--k-fwd', type=int, default=8)
    ap.add_argument('--n-history', type=int, default=128)
    ap.add_argument('--name', required=True)
    ap.add_argument('--out-root', default=str(Path(__file__).parent.parent / 'output'))
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--steps', type=int, default=10000)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--worker-cache-gb', type=float, default=1.)
    ap.add_argument('--val-z', type=float, nargs=2, default=(45000., 48500.))
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--warmup', type=int, default=150)
    ap.add_argument('--ckpt-every', type=int, default=500)
    ap.add_argument('--log-every', type=int, default=50)
    ap.add_argument('--diag-every', type=int, default=500, help='0 disables span diagnostics and images')
    ap.add_argument('--diag-fibers', type=int, default=8)
    ap.add_argument('--crop-depth', type=int, default=64)
    ap.add_argument('--crop-width', type=int, default=64)
    ap.add_argument('--crop-behind', type=int, default=16)
    ap.add_argument('--crop-spacing', type=float, default=.5)
    ap.add_argument('--history-render', choices=('points', 'segments'), default='segments')
    ap.add_argument('--history-sigma', type=float, default=.35)
    ap.add_argument('--widths', type=int, nargs='+', default=(24, 64, 128))
    ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--hist-points', type=int, default=32)
    ap.add_argument('--hist-stride', type=int, default=4)
    ap.add_argument('--norm', choices=('batch', 'group'), default='batch')
    ap.add_argument('--no-tube', action='store_true', help='Disable the auxiliary dense tube head')
    ap.add_argument('--tube-sigma', type=float, default=.35)
    ap.add_argument('--tolerance', type=float, default=1.5)
    ap.add_argument('--lateral-sigmas', type=float, nargs=3, default=(.4, 1., 2.))
    ap.add_argument('--angle-sigmas', type=float, nargs=3, default=(4., 10., 20.))
    ap.add_argument('--p-perturb', type=float, default=.5)
    ap.add_argument('--max-span', type=float, default=512., help='Longest random training span, grid voxels')
    ap.add_argument('--hard-prob', type=float, default=.5)
    ap.add_argument('--states-per-trace', type=int, default=12)
    ap.add_argument('--hard-spans', default='auto',
                    help="JSON cache of spans the hand beam fails on; 'auto' mines into output/hard_spans_*.json, "
                         "'none' disables")
    ap.add_argument('--hard-span-prob', type=float, default=.5, help='Probability of tracing a mined hard span')
    ap.add_argument('--mine-threads', type=int, default=0, help='C++ threads while mining hard spans (0 = OpenMP)')
    ap.add_argument('--rank-weight', type=float, default=1.)
    ap.add_argument('--onfiber-weight', type=float, default=1.)
    ap.add_argument('--prefix-weight', type=float, default=1.)
    ap.add_argument('--tube-weight', type=float, default=1.)
    ap.add_argument('--hook-mode', choices=('additive', 'replace'), default='additive')
    ap.add_argument('--hook-weight', type=float, default=1.)
    return ap


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)
    if min(args.steps, args.batch, args.ckpt_every, args.log_every, args.k_back, args.k_fwd, args.pool_size,
           args.hook_every_rounds, args.states_per_trace) < 1 or args.workers < 0 or args.diag_every < 0:
        ap.error('Steps, batch, intervals, path points, pool size, hook cadence and states must be positive')
    if args.n_history < args.hist_points * args.hist_stride or args.n_history < args.k_back:
        ap.error('n-history must cover hist-points * hist-stride and k-back')
    if min(args.crop_spacing, args.tolerance, args.tube_sigma, args.history_sigma) <= 0:
        ap.error('Spacings, tolerance and sigmas must be positive')
    if args.k_fwd > (args.crop_depth - args.crop_behind - 1) * args.crop_spacing:
        ap.error('k-fwd forward points must fit inside the crop')
    out = prepare_run_dir(args.out_root, args.name)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    spec = FiberVolumeSpec(args.fiber_zarrs, ct_zarr=args.ct, ct_level=args.ct_level, ct_grid_scale=args.ct_grid_scale,
                           inputs='ct')
    crop = CropSpec(depth=args.crop_depth, width=args.crop_width, behind=args.crop_behind, spacing=args.crop_spacing,
                    history_render=args.history_render, history_sigma=args.history_sigma)
    heatmap_target = 'planes' if args.no_tube else 'tube'
    state_cfg = BeamStateConfig(crop=crop, n_history=args.n_history, k_back=args.k_back, k_fwd=args.k_fwd,
                                pool_size=args.pool_size, tolerance=args.tolerance, heatmap_target=heatmap_target,
                                tube_sigma=args.tube_sigma, lateral_sigmas=tuple(args.lateral_sigmas),
                                angle_sigmas_deg=tuple(args.angle_sigmas))
    beam_spec = BeamSpec(args.prediction_manifest, args.normal_manifest, cache_bytes=int(args.beam_cache_gb * (1 << 30)),
                         scaledown_power=args.scaledown_power, config=json.loads(args.beam_config),
                         hook_every_rounds=args.hook_every_rounds, hook_pool_size=args.pool_size, parallel_threads=1)
    model_cfg = BeamNetConfig(in_channels=2, depth=crop.depth, width=crop.width, behind=crop.behind, spacing=crop.spacing,
                                widths=tuple(args.widths), hidden=args.hidden, n_future=2, future_step=1.,
                                hist_points=args.hist_points, hist_stride=args.hist_stride,
                                recent_history_points=min(8, args.hist_points * args.hist_stride),
                                norm=args.norm,
                                heatmap_target=heatmap_target, tube_sigma=args.tube_sigma)
    model = prepare_model(BeamRankNet(model_cfg), args.device)

    fibers = load_fibers(args.fibers, grid_scale=spec.grid_scale)
    band = ZBand(args.val_z[0] / spec.grid_scale, args.val_z[1] / spec.grid_scale)
    train_f, val_f = split_fibers(fibers, band)
    (out / 'config.json').write_text(json.dumps(dict(vars(args), architecture=ARCHITECTURE, data_policy=DATA_POLICY,
                                                     beam_spec=beam_spec.to_dict(), state_cfg=state_config_dict(state_cfg),
                                                     fiber_manifest=fiber_manifest(fibers),
                                                     train=[f.name for f in train_f], val=[f.name for f in val_f]),
                                                indent=2, default=str))
    chunk = min(8, args.batch)
    if args.batch % chunk:
        chunk = math.gcd(args.batch, chunk)
    hard_spans = None
    if args.hard_spans != 'none':
        from vesuvius.neural_tracing.fiber_follow.beam.mine import load_or_mine_hard_spans
        digest = hashlib.sha256(''.join(f.source_hash for f in train_f).encode()).hexdigest()[:12]
        cache = Path(args.out_root) / f'hard_spans_{digest}.json' if args.hard_spans == 'auto' else Path(args.hard_spans)
        miner = NativeBeam(dataclasses.replace(beam_spec, parallel_threads=args.mine_threads), spec.grid_scale)
        hard_spans = load_or_mine_hard_spans(cache, miner, train_f, spec.grid_scale,
                                             progress=lambda e: print(json.dumps(dict(mining=e)), flush=True))
        print(json.dumps(dict(hard_spans=sum(map(len, hard_spans.values())), hard_span_fibers=len(hard_spans),
                              cache=str(cache))), flush=True)
    ds = BeamDataset(train_f, spec, beam_spec, state_cfg, band, chunk=chunk, seed=args.seed,
                     cache_bytes=int(args.worker_cache_gb * (1 << 30)), p_perturb=args.p_perturb,
                     hard_prob=args.hard_prob, max_states_per_trace=args.states_per_trace,
                     hard_spans=hard_spans, hard_span_prob=args.hard_span_prob, max_span=args.max_span)
    kwargs = dict(num_workers=args.workers, batch_size=None)
    if args.workers:
        # The beam's readers keep process-global state: never fork after opening them (the mining
        # above ran in this process, so the forkserver context is required, not just preferred).
        kwargs.update(prefetch_factor=2, persistent_workers=True, multiprocessing_context='forkserver')
    dl = torch.utils.data.DataLoader(ds, **kwargs)
    it = iter(dl)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    diag = None
    if args.diag_every:
        from vesuvius.neural_tracing.fiber_follow.beam.diag import plot_pool, span_diag
        from vesuvius.neural_tracing.fiber_follow.beam.hook import ModelBeamHook
        from vesuvius.neural_tracing.fiber_follow.volume import FiberVolume
        rng = np.random.default_rng(123)
        diag_fibers = [val_f[i] for i in rng.choice(len(val_f), min(args.diag_fibers, len(val_f)), replace=False)]
        vol = FiberVolume(spec, cache_bytes=2 << 30)
        beam = NativeBeam(beam_spec, spec.grid_scale)
        hook = ModelBeamHook(model, vol, state_cfg, device=args.device, mode=args.hook_mode, weight=args.hook_weight)
        diag = dict(plot_pool=plot_pool, span_diag=span_diag, fibers=diag_fibers, beam=beam, hook=hook)
        (out / 'images').mkdir(exist_ok=True)
    print(json.dumps(dict(train_fibers=len(train_f), val_fibers=len(val_f),
                          parameters=sum(p.numel() for p in model.parameters()))), flush=True)
    log = RunLog(out / 'log.jsonl')
    start = time.monotonic()
    extra = lambda step: dict(step=step, seed=args.seed, beam_spec=beam_spec.to_dict(), state_cfg=state_config_dict(state_cfg),
                              hook_mode=args.hook_mode, hook_weight=args.hook_weight)
    try:
        for step in range(1, args.steps + 1):
            parts = [next(it) for _ in range(args.batch // chunk)]
            batch = {k: torch.cat([p[k] for p in parts]).to(args.device) for k in parts[0]}
            lr = lr_at(step, args.lr, args.warmup, args.steps)
            model.train()
            torch.backends.cudnn.benchmark = args.device.startswith('cuda')
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=args.device.startswith('cuda')):
                output = model(batch['x'].float(), batch['hist'], batch['hmask'], batch['candidates'],
                               batch['point_mask'], batch['hand_rel'])
            loss, metrics = beam_loss(output, batch, state_cfg.k_back, args.rank_weight, args.onfiber_weight,
                                      args.prefix_weight, 0. if args.no_tube else args.tube_weight)
            optimizer_step(model, opt, loss, step, lr)
            if step % args.log_every == 0 or step == args.steps:
                log.record(dict(step=step, loss=loss.item(), lr=lr,
                                hard_fraction=(batch['source'] == 2).float().mean().item(),
                                mined_fraction=(batch['source'] >= 1).float().mean().item(),
                                samples_per_second=step * args.batch / (time.monotonic() - start), **metrics))
            if diag and step % args.diag_every == 0:
                torch.backends.cudnn.benchmark = False
                diag['plot_pool'](batch, output, crop, out / 'images' / f'pool_{step:06d}.png', state_cfg.k_back)
                _, summary = diag['span_diag'](diag['beam'], diag['fibers'], spec.grid_scale, hook=diag['hook'])
                log.record(dict(step=step, **summary))
            if step % args.ckpt_every == 0 or step == args.steps:
                for path in (out / f'ckpt_{step:06d}.pt', out / 'last.pt'):
                    save_checkpoint(path, model, spec, crop, state_cfg.n_history, ARCHITECTURE, extra(step))
    finally:
        log.close()
    return str(out / 'last.pt')


if __name__ == '__main__':
    main()
