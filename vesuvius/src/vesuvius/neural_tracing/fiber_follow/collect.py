"""Collect exact policy decisions and relabel them with annotated dense geometry."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

import numpy as np
import torch

from vesuvius.neural_tracing.fiber_follow.data import (
    OnPolicyStates, SampleConfig, ZBand, load_fibers, split_fibers,
    fiber_manifest, label_state, training_state_allowed,
)
from vesuvius.neural_tracing.fiber_follow.evaluate import make_seeds
from vesuvius.neural_tracing.fiber_follow.geometry import arclength, tangent_at
from vesuvius.neural_tracing.fiber_follow.trace import DEFAULT_CONFIDENCE, DEFAULT_N_COMMIT, ModelTracer, TraceParams
from vesuvius.neural_tracing.fiber_follow.train import load_checkpoint
from vesuvius.neural_tracing.fiber_follow.volume import FiberVolume


class DecisionCollector:
    """Original-fiber, progress-bounded labeler, including pre-failure windows."""
    def __init__(self, fiber, fiber_idx, t0, sign, cfg, band=None, off_dist=3.5,
                 before=48., after=24., stride=16., max_states=192):
        self.fiber, self.fi, self.t, self.sign = fiber, fiber_idx, t0, sign
        self.cfg, self.band, self.off_dist = cfg, band, off_dist
        self.before, self.after, self.stride, self.max_states = before, after, stride, max_states
        self.rows, self.distances = [], []
        self.departed = None
        self.last_travelled = 0.
        self.boundary_crossed = False

    def __call__(self, state):
        f, sign = self.fiber, self.sign
        segment = state['last_segment']
        travelled = state['travelled']
        if self.band is not None and np.any((segment[:, 2] >= self.band.lo-48) &
                                            (segment[:, 2] < self.band.hi+48)):
            return False
        # Progress matching cannot jump to another winding or another fiber.
        progress = (f.s-self.t)*sign
        window = (progress >= -8) & (progress <= travelled-self.last_travelled+32)
        indices = np.flatnonzero(window)
        if not len(indices):
            return False
        near = f.points[indices]
        distances = np.linalg.norm(segment[:, None]-near[None], axis=-1)
        nearest = indices[distances.argmin(-1)]
        end_index = -1 if sign > 0 else 0
        endpoint = f.points[end_index]
        end_arc = f.length if sign > 0 else 0.
        tangent = tangent_at(f.points, f.s, end_arc)*sign
        residual = segment-endpoint
        beyond = residual @ tangent >= 0
        lateral = np.linalg.norm(residual-(residual @ tangent)[:, None]*tangent, axis=-1)
        # Crossing only counts after progressing near the actual annotation end.
        remaining = (end_arc-f.s[nearest])*sign
        crosses = beyond & (lateral <= self.off_dist) & (remaining <= max(8., travelled-self.last_travelled+2))
        if self.departed is None:
            bad = distances.min(-1) > self.off_dist
            crossing_idx = np.flatnonzero(crosses)
            bad_idx = np.flatnonzero(bad)
            crossing_first = len(crossing_idx) and (not len(bad_idx) or crossing_idx[0] <= bad_idx[0])
            if crossing_first:
                self.boundary_crossed = True
            if self.boundary_crossed and not f.endpoint_stop[0 if sign < 0 else 1]:
                return False  # never call unannotated continuation a failure
            if len(bad_idx) or self.boundary_crossed:
                first = int(bad_idx[0]) if len(bad_idx) else int(crossing_idx[0])
                self.departed = self.last_travelled+float(arclength(segment[:first+1])[-1])
            else:
                self.t = float(f.s[nearest[-1]])
        offtrack = self.departed is not None
        if offtrack and travelled-self.departed > self.after:
            return False
        hard = offtrack or state['would_stop'] or state['exploratory']
        if hard:
            for row, distance in zip(reversed(self.rows), reversed(self.distances)):
                if travelled-distance > self.before:
                    break
                row['hard'] = True
        item = label_state(f, state['pos'], state['frame'], state['hist'], state['hmask'], self.cfg,
                           t=self.t, reverse=sign < 0, offtrack=offtrack)
        if not training_state_allowed(item, self.cfg.crop, self.band):
            return False  # do not trace through held-out space and resume afterwards
        row = {k: state[k] for k in ('pos', 'frame', 'hist', 'hmask', 'exploratory')}
        # Current-position error against the matched GT point, in trace-grid
        # voxels; departed states have no correspondence. Lets replay stratify
        # on recoverable drift without relabeling every state at load time.
        drift = float('nan') if offtrack else float(np.linalg.norm(item['gt_history'][0]))
        row.update(source_cache=-1, source_row=len(self.rows), fiber_idx=self.fi, t=self.t, reverse=sign < 0, offtrack=offtrack, hard=hard, drift=drift)
        self.rows.append(row)
        self.distances.append(travelled)
        self.last_travelled = travelled
        return True

    def finish(self):
        # Thin ordinary states after retrospective hard-window marking.
        kept, last = [], -float('inf')
        for row, distance in zip(self.rows, self.distances):
            if row['hard'] or distance-last >= self.stride:
                kept.append(row)
                last = distance
        if len(kept) > self.max_states:
            hard = [r for r in kept if r['hard']]
            ordinary = [r for r in kept if not r['hard']]
            nh = min(len(hard), max(self.max_states//2, self.max_states-len(ordinary)))
            select = lambda rows, n: [rows[i] for i in np.linspace(0, len(rows)-1, n).astype(int)] if n else []
            kept = select(hard, nh)+select(ordinary, min(len(ordinary), self.max_states-nh))
        return kept


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--fibers', required=True)
    ap.add_argument('--fiber-zarrs')
    ap.add_argument('--val-z', type=float, nargs=2, default=(45000., 48500.))
    ap.add_argument('--seeds-per-fiber', type=int, default=2)
    ap.add_argument('--max-seeds', type=int, default=0, help='0 selects all eligible seeds')
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--trace-len', type=float, default=6000.)
    ap.add_argument('--off-dist', type=float, default=3.5)
    ap.add_argument('--before', type=float, default=48.)
    ap.add_argument('--after', type=float, default=24.)
    ap.add_argument('--stride', type=float, default=16.)
    ap.add_argument('--confidence', type=float, default=DEFAULT_CONFIDENCE)
    ap.add_argument('--n-commit', type=int, default=DEFAULT_N_COMMIT)
    ap.add_argument('--explore-calls', type=int, default=8)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=1)
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)
    model, crop, n_hist, spec, ck = load_checkpoint(args.checkpoint, args.device)
    cfg = SampleConfig(crop=crop, n_history=n_hist, recent_history_points=model.cfg.recent_history_points,
                       n_future=model.cfg.n_future, future_step=model.cfg.future_step)
    if args.fiber_zarrs:
        spec.fiber_zarr_dir = args.fiber_zarrs
    vol = FiberVolume(spec, cache_bytes=2 << 30)
    fibers = load_fibers(args.fibers, grid_scale=spec.grid_scale)
    band = ZBand(args.val_z[0]/spec.grid_scale, args.val_z[1]/spec.grid_scale)
    train_f, _ = split_fibers(fibers, band)
    # Limit volume reads as well as rollouts when collecting a bounded batch.
    rng = np.random.default_rng(args.seed)
    seeds = []
    for fi in rng.permutation(len(train_f)):
        seed_pool = make_seeds([train_f[fi]], vol, per_fiber=args.seeds_per_fiber,
                                min_presence=.8, seed=int(rng.integers(2**31)))
        for seed in seed_pool:
            seed['fiber'] = int(fi)
            if not band.lo-64 <= seed['pos'][2] < band.hi+64:
                seeds.append(seed)
        if args.max_seeds and len(seeds) >= args.max_seeds:
            seeds = seeds[:args.max_seeds]
            break
    tracer = ModelTracer(model, vol, crop, n_hist,
                         TraceParams(max_len=args.trace_len, confidence=args.confidence, explore_calls=args.explore_calls,
                                     n_commit=args.n_commit),
                         device=args.device)
    rows = []
    try:
        for offset in range(0, len(seeds), args.batch):
            chunk = seeds[offset:offset+args.batch]
            collectors = [DecisionCollector(train_f[s['fiber']], s['fiber'], s['t'], s['sign'], cfg, band,
                                           args.off_dist, args.before, args.after, args.stride) for s in chunk]
            tracer.trace(np.stack([s['pos'] for s in chunk]), np.stack([s['heading'] for s in chunk]),
                         on_decision=lambda i, state: collectors[i](state))
            for collector in collectors:
                rows.extend(collector.finish())
            print(json.dumps(dict(traces=offset+len(chunk), total=len(seeds), states=len(rows))), flush=True)
    finally:
        tracer.close()
    if not rows:
        raise ValueError('No eligible decision states; no cache was published')
    st = OnPolicyStates(manifest=fiber_manifest(train_f),
                        provenance=dict(checkpoint=os.path.abspath(args.checkpoint), step=ck.get('step'),
                                        model_cfg=model.cfg.to_dict(), crop=asdict(crop),
                                        volume=spec.to_dict(), collection=vars(args)),
                        **{key: np.asarray([row[key] for row in rows])
                           for key in OnPolicyStates.FIELDS + tuple(OnPolicyStates.OPTIONAL)})
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.stem+'.partial.npz')
    st.save(temp)
    # Pre-create mmap before publishing so multiple loader workers never race.
    OnPolicyStates.load(temp)
    temp_mmap = Path(str(temp)[:-4]+'_mmap_v5')
    final_mmap = Path(str(path)[:-4]+'_mmap_v5')
    os.replace(temp_mmap, final_mmap)
    os.replace(temp, path)
    print(json.dumps(dict(states=len(st), hard=int(st.hard.sum()), offtrack=int(st.offtrack.sum()), out=str(path))))
    return str(path)


if __name__ == '__main__':
    main()
