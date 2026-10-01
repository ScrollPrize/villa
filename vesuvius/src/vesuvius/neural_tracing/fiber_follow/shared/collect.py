"""Collect exact policy decisions and relabel them with annotated dense geometry."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

import numpy as np
import torch

from vesuvius.neural_tracing.fiber_follow.shared.data import (
    OnPolicyStates, SampleConfig, ZBand, load_fibers, split_fibers,
    fiber_manifest, label_state, training_state_allowed,
)
from vesuvius.neural_tracing.fiber_follow.shared.evaluate import make_seeds
from vesuvius.neural_tracing.fiber_follow.shared.geometry import arclength, tangent_at
from vesuvius.neural_tracing.fiber_follow.shared.trace import DEFAULT_CONFIDENCE, DEFAULT_N_COMMIT, ModelTracer, TraceParams
from vesuvius.neural_tracing.fiber_follow.shared.volume import FiberVolume
from vesuvius.neural_tracing.fiber_follow.shared.reference import SEED_FIELDS, SEED_DEFAULTS
from vesuvius.neural_tracing.fiber_follow.shared.heading import SEED_HEADING_POLICY, TRACE_HEADING_POLICY, FRAME_POLICY


class DecisionCollector:
    """Original-fiber, progress-bounded labeler, including pre-failure windows."""
    def __init__(self, fiber, fiber_idx, t0, sign, cfg, band=None, off_dist=3.5,
                 before=48., after=24., stride=16., max_states=192, additional_crops=(), bank_detector=None):
        self.fiber, self.fi, self.t, self.sign = fiber, fiber_idx, t0, sign
        self.cfg, self.band, self.off_dist = cfg, band, off_dist
        self.before, self.after, self.stride, self.max_states = before, after, stride, max_states
        self.additional_crops = tuple(additional_crops)
        self.rows, self.distances = [], []
        # Actual committed vertices, once per trace (before decision thinning).
        self.track = []
        self.departed = None
        self.last_travelled = 0.
        self.boundary_crossed = False
        self.bank_detector, self.bank_switch = bank_detector, None

    def observe_bank_segment(self, segment, travelled):
        """Attach first-contact evidence to the decisions preceding the switch."""
        if self.bank_detector is None or self.bank_switch is not None:
            return
        event = self.bank_detector.first_contact(self.fi, self.t, segment)
        if event is None:
            return
        start = travelled-float(arclength(segment)[-1])
        commit_rows = [j for j, distance in enumerate(self.distances) if distance <= start+1e-6]
        self.bank_switch = dict(switch_distance=start+event['distance'],
            switch_pos=event['pos'], switch_decision=commit_rows[-1] if commit_rows else 0,
            switch_bank_path=event['bank_path'], switch_bank_run=event['bank_run'])
        self.departed = (self.bank_switch['switch_distance'] if self.departed is None else
                         min(self.departed, self.bank_switch['switch_distance']))
        for row, distance in zip(reversed(self.rows), reversed(self.distances)):
            if self.bank_switch['switch_distance']-distance > self.before:
                break
            row.update(self.bank_switch, hard=True)
            if not row['offtrack']:
                row['failure_kind'] = 4  # geometry remains supervised before contact

    def observe_final_path(self, path):
        """A length-limited trace may commit a last segment with no next decision.

        Keep its causal pre-switch rows; never fabricate an unobserved endpoint.
        The caller excludes oracle-aborted traces, and holdout remains censored.
        """
        if len(path) >= len(self.track) and np.array_equal(np.asarray(self.track), np.asarray(path)[:len(self.track)]):
            self.track.extend(np.asarray(path)[len(self.track):].copy())
        if self.bank_detector is None or self.bank_switch is not None or not self.rows:
            return
        arc = arclength(path)
        if arc[-1] <= self.last_travelled+1e-6:
            return
        tail = np.concatenate((np.asarray(self.rows[-1]['pos'])[None],
                               np.asarray(path)[arc > self.last_travelled+1e-6]))
        if self.band is not None and np.any((tail[:,2] >= self.band.lo-48) & (tail[:,2] < self.band.hi+48)):
            return
        self.observe_bank_segment(tail, float(arc[-1]))

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
        self.observe_bank_segment(segment, travelled)
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
        if not all(training_state_allowed(item, crop, self.band)
                   for crop in (self.cfg.crop, *self.additional_crops)):
            return False  # do not trace through held-out space and resume afterwards
        row = {k: default(1)[0] for k, default in OnPolicyStates.OPTIONAL.items()}
        row.update({k: state[k] for k in ('pos', 'frame', 'hist', 'hmask', 'exploratory')})
        row.update({k: state[k] if k in state else SEED_DEFAULTS[k](1)[0] for k in SEED_FIELDS})
        # Current-position error against the matched GT point, in trace-grid
        # voxels; departed states have no correspondence. Lets replay stratify
        # on recoverable drift without relabeling every state at load time.
        drift = float('nan') if offtrack else float(np.linalg.norm(item['gt_history'][0]))
        row.update(source_cache=-1, source_row=len(self.rows), fiber_idx=self.fi, t=self.t, reverse=sign < 0, offtrack=offtrack, hard=hard, drift=drift)
        row['travelled'] = travelled
        row['heading_start'] = int(state.get('heading_start', 0))
        if self.bank_switch is not None:
            row.update(self.bank_switch, failure_kind=1)
        elif self.boundary_crossed:
            row['failure_kind'] = 3
        elif not offtrack and state['would_stop'] and item['plane_mask'][0] and not state.get('recovery_blocked', False):
            row['failure_kind'] = 2
        # last_segment is the actual committed polyline, including intermediate
        # vertices. It is also sufficient for callers without a full prefix.
        prefix = np.asarray(state.get('observed_path',
            list(self.track)+list(segment[1:] if self.track else segment)), dtype=np.float64)
        if self.track and not np.array_equal(np.asarray(self.track), prefix[:len(self.track)]):
            raise ValueError('Collected observed prefixes must extend the committed trace')
        self.track.extend(prefix[len(self.track):].copy())
        row['prefix_end'] = len(self.track)
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


def append_traces(collectors, rows, track):
    """Kept rows reference their own trace's earlier heads in the shared track."""
    for collector in collectors:
        for row in collector.finish():
            row.update(seq_start=len(track), seq_end=len(track)+row['prefix_end'])
            rows.append(row)
        track.extend(collector.track)


def track_arrays(track):
    return {'track_pos': np.asarray(track, np.float64).reshape(-1, 3)}


def main(argv=None, *, checkpoint_loader, tracer_class=ModelTracer, bank_loader=None, dataset_loader=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--fibers', required=True)
    ap.add_argument('--dataset-name', help='Source in the checkpoint dataset configuration')
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
    ap.add_argument('--n-commit', type=int, help='Default: checkpoint commit window')
    ap.add_argument('--explore-calls', type=int, default=8)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--failure-bank', action='append', default=[], help='Trusted bank for switch labeling; repeat for multiple banks')
    ap.add_argument('--bank-switch-tolerance', type=float, default=.75)
    ap.add_argument('--bank-own-tolerance', type=float, default=1.5)
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)
    model, crop, n_hist, spec, ck = checkpoint_loader(args.checkpoint, args.device)
    if args.n_commit is None:
        args.n_commit = ck.get('n_commit', DEFAULT_N_COMMIT)
    cfg = SampleConfig(crop=crop, n_history=n_hist, recent_history_points=model.cfg.recent_history_points,
                       n_future=model.cfg.n_future, future_step=model.cfg.future_step)
    if args.fiber_zarrs:
        spec.fiber_zarr_dir = args.fiber_zarrs
    if args.dataset_name:
        if dataset_loader is None:
            raise ValueError('This collector does not support named datasets')
        spec, train_f, band = dataset_loader(args, ck)
    else:
        fibers = load_fibers(args.fibers, grid_scale=spec.grid_scale)
        band = ZBand(args.val_z[0]/spec.grid_scale, args.val_z[1]/spec.grid_scale)
        train_f, _ = split_fibers(fibers, band)
    vol = FiberVolume(spec, cache_bytes=2 << 30)
    bank_detector = bank_loader(args, ck, train_f, band, spec) if bank_loader is not None else None
    # Limit volume reads as well as rollouts when collecting a bounded batch.
    rng = np.random.default_rng(args.seed)
    seeds = []
    for fi in rng.permutation(len(train_f)):
        seed_pool = make_seeds([train_f[fi]], vol, per_fiber=args.seeds_per_fiber,
                                seed=int(rng.integers(2**31)))
        for seed in seed_pool:
            seed['fiber'] = int(fi)
            if band is None or not band.lo-64 <= seed['pos'][2] < band.hi+64:
                seeds.append(seed)
        if args.max_seeds and len(seeds) >= args.max_seeds:
            seeds = seeds[:args.max_seeds]
            break
    tracer = tracer_class(model, vol, crop, n_hist,
                         TraceParams(max_len=args.trace_len, confidence=args.confidence, explore_calls=args.explore_calls,
                                     n_commit=args.n_commit, seed=args.seed),
                         device=args.device)
    rows, track = [], []
    try:
        for offset in range(0, len(seeds), args.batch):
            chunk = seeds[offset:offset+args.batch]
            collectors = [DecisionCollector(train_f[s['fiber']], s['fiber'], s['t'], s['sign'], cfg, band,
                                           args.off_dist, args.before, args.after, args.stride,
                                           additional_crops=getattr(tracer, 'additional_crops', ()),
                                           bank_detector=bank_detector) for s in chunk]
            paths, reasons = tracer.trace(np.stack([s['pos'] for s in chunk]), np.stack([s['heading'] for s in chunk]),
                                          on_decision=lambda i, state: collectors[i](state))
            for collector, path, reason in zip(collectors, paths, reasons):
                if reason != 'oracle':
                    collector.observe_final_path(path)
            append_traces(collectors, rows, track)
            print(json.dumps(dict(traces=offset+len(chunk), total=len(seeds), states=len(rows))), flush=True)
    finally:
        tracer.close()
    if not rows:
        raise ValueError('No eligible decision states; no cache was published')
    st = OnPolicyStates(manifest=fiber_manifest(train_f),
                        provenance=dict(checkpoint=os.path.abspath(args.checkpoint), step=ck.get('step'),
                                        seed_heading_policy=SEED_HEADING_POLICY, heading_policy=TRACE_HEADING_POLICY,frame_policy=FRAME_POLICY,
                                        model_cfg=model.cfg.to_dict(), crop=asdict(crop),
                                        sampler_mode=getattr(model.cfg, 'sampler_mode', 'zero'),
                                        volume=spec.to_dict(), collection=vars(args),
                                        failure_banks=([b.provenance() for b in bank_detector.banks]
                                                       if bank_detector is not None else [])),
                        **{key: np.asarray([row[key] for row in rows])
                           for key in OnPolicyStates.FIELDS + tuple(OnPolicyStates.OPTIONAL) + tuple(OnPolicyStates.ROW_TRACK)},
                        **track_arrays(track))
    # Fail in the collector before publishing a cache to live training workers.
    st.validate_fibers(train_f)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.stem+'.partial.npz')
    st.save(temp)
    # Pre-create mmap before publishing so multiple loader workers never race.
    OnPolicyStates.load(temp)
    temp_mmap = Path(str(temp)[:-4]+'_mmap_v8')
    final_mmap = Path(str(path)[:-4]+'_mmap_v8')
    os.replace(temp_mmap, final_mmap)
    os.replace(temp, path)
    print(json.dumps(dict(states=len(st), hard=int(st.hard.sum()), offtrack=int(st.offtrack.sum()), out=str(path))))
    return str(path)
