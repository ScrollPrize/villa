"""Collect exact policy decisions and label them under the shared state contract.

One collection traces one directed episode per distinct eligible fiber, visiting unseen
fibers before repeats and balancing directions over rounds through a saved coverage
cursor. A policy stop ends its episode; there is no forced exploration. Dense decisions
are retained before excursions, during recovery, at stops and at terminal failures;
ordinary following is thinned by travel.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time
import warnings

import numpy as np
import torch

from vesuvius.neural_tracing.fiber_follow.data.data import OnPolicyStates, SampleConfig, ZBand, load_fibers, split_fibers, fiber_manifest, label_state, training_state_allowed
from vesuvius.neural_tracing.fiber_follow.evaluation.seeds import directed_seed
from vesuvius.neural_tracing.fiber_follow.shared.geometry import arclength
from vesuvius.neural_tracing.fiber_follow.tracing.heading import SeedHeadingError, SEED_HEADING_POLICY, TRACE_HEADING_POLICY, FRAME_POLICY
from vesuvius.neural_tracing.fiber_follow.tracing.policy import checkpoint_policy
from vesuvius.neural_tracing.fiber_follow.shared.reference import SEED_FIELDS
from vesuvius.neural_tracing.fiber_follow.data.state_labels import DEPARTURE_DISTANCE, DEPARTURE_PATIENCE, REASON, RECOVERABLE, REPLAY_CLASSES, SUPERVISION, TERMINAL, TraceLabeler, replay_class
from vesuvius.neural_tracing.fiber_follow.tracing.trace import ModelTracer, TraceParams
from vesuvius.neural_tracing.fiber_follow.data.volume import FiberVolume


class DecisionCollector:
    """Labels one directed episode's decisions and keeps its event windows.

    An event is a contiguous run of excursion, recovery or terminal states, or a policy
    stop. Following rows within ``before`` voxels of an event onset become pre-excursion
    rows of that event. After the first terminal state at most ``after`` voxels are
    retained; collection then censors the episode (this never advances a stopped policy).
    """
    def __init__(self, fiber, fiber_idx, t0, sign, cfg, band=None, *, before=48., after=64., stride=16.,
                 max_states=192, additional_crops=(), bank_detector=None):
        if min(before, after, stride) < 0 or max_states < 1:
            raise ValueError('Retention windows must be nonnegative with a positive state cap')
        self.fiber, self.fi, self.sign = fiber, fiber_idx, sign
        self.cfg, self.band = cfg, band
        self.labeler = TraceLabeler(fiber, t0, sign, tolerance=cfg.label_tolerance,
                                    max_recovery_distance=cfg.max_recovery_distance,
                                    fiber_idx=fiber_idx, bank_detector=bank_detector)
        self.before, self.after, self.stride, self.max_states = before, after, stride, max_states
        self.additional_crops = tuple(additional_crops)
        self.rows, self.distances = [], []
        # Actual committed vertices, once per trace (before decision thinning).
        self.track = []
        self.terminal_at = None
        self.censored = None
        self.events, self.current_event = 0, None

    def in_band(self, segment):
        return self.band is not None and np.any((segment[:, 2] >= self.band.lo-48) & (segment[:, 2] < self.band.hi+48))

    def open_event(self, onset):
        """Start an event; following rows within ``before`` of its onset precede it."""
        event, self.events = self.events, self.events+1
        for row, distance in zip(reversed(self.rows), reversed(self.distances)):
            if onset-distance > self.before:
                break
            if row['event_id'] < 0:
                row.update(event_id=event, pre_excursion=True, hard=True)
        return event

    def __call__(self, state):
        segment = np.asarray(state['last_segment'])
        travelled = float(state['travelled'])
        if self.in_band(segment):
            self.censored = 'holdout'
            return False
        facts = self.labeler.observe(segment, travelled)
        item = label_state(self.fiber, state['pos'], state['frame'], state['hist'], state['hmask'], self.cfg,
                           t=facts['t'], reverse=self.sign < 0, trace=facts)
        if item['supervision_reason'] == REASON['unannotated']:
            self.censored = 'unannotated'  # never call unannotated continuation a failure
            return False
        if not all(training_state_allowed(item, crop, self.band) for crop in (self.cfg.crop, *self.additional_crops)):
            self.censored = 'holdout'  # do not trace through held-out space and resume afterwards
            return False
        terminal = item['supervision'] == TERMINAL
        if terminal and self.terminal_at is None:
            self.terminal_at = travelled
        if self.terminal_at is not None and travelled-self.terminal_at > self.after:
            self.censored = 'after_failure'
            return False
        displaced = terminal or facts['excursion'] or item['supervision'] == RECOVERABLE
        if displaced:
            if self.current_event is None:
                onset = min(travelled, facts['bad_run_start'] if np.isfinite(facts['bad_run_start']) else travelled,
                            facts['switch_distance'] if np.isfinite(facts['switch_distance']) else travelled)
                self.current_event = self.open_event(onset)
            event = self.current_event
        else:
            self.current_event = None
            event = -1
            if state['would_stop']:  # a stop is its own event, without a pre-excursion window
                event, self.events = self.events, self.events+1
        switch = self.labeler.switch or {}
        row = dict(fiber_idx=self.fi, t=facts['t'], reverse=self.sign < 0, travelled=travelled,
                   heading_start=int(state['heading_start']), source_row=len(self.rows),
                   **{k: state[k] for k in ('pos', 'frame', 'hist', 'hmask')},
                   **{k: state[k] for k in SEED_FIELDS},
                   **{k: item[k] for k in ('supervision', 'supervision_reason', 'geometry_valid', 'confidence_valid')},
                   **{k: facts[k] for k in ('match_distance', 'window_distance', 'match_valid', 'match_ambiguous',
                                            'switched', 'beyond_end', 'departure_distance', 'boundary_distance',
                                            'switch_distance', 'bad_run', 'bad_run_start')},
                   switch_pos=np.asarray(switch.get('switch_pos', np.full(3, np.nan)), np.float64),
                   switch_bank_path=str(switch.get('switch_bank_path', '')),
                   switch_bank_run=str(switch.get('switch_bank_run', '')),
                   would_stop=bool(state['would_stop']), n_commit=int(state['n_commit']),
                   proposal_points=np.asarray(state['points'], np.float32),
                   proposal_confidence=np.asarray(state['confidence'], np.float32),
                   event_id=event, pre_excursion=False, hard=bool(displaced or state['would_stop']))
        # last_segment is the actual committed polyline, including intermediate vertices,
        # so it extends the trace for callers that do not pass the whole observed prefix.
        prefix = np.asarray(state.get('observed_path', list(self.track)+list(segment[1:] if self.track else segment)),
                            dtype=np.float64)
        if self.track and not np.array_equal(np.asarray(self.track), prefix[:len(self.track)]):
            raise ValueError('Collected observed prefixes must extend the committed trace')
        self.track.extend(prefix[len(self.track):].copy())
        row['prefix_end'] = len(self.track)
        self.rows.append(row)
        self.distances.append(travelled)
        return True

    def observe_final_path(self, path):
        """A length-limited trace may commit a last segment with no next decision.

        Record its committed vertices and any event it starts, so the preceding rows
        are kept as pre-excursion; never fabricate an unobserved decision.
        """
        path = np.asarray(path)
        if len(path) >= len(self.track) and np.array_equal(np.asarray(self.track), path[:len(self.track)]):
            self.track.extend(path[len(self.track):].copy())
        if not self.rows or self.censored is not None:
            return
        arc = arclength(path)
        last = self.distances[-1]
        if arc[-1] <= last+1e-6:
            return
        tail = np.concatenate((np.asarray(self.rows[-1]['pos'])[None], path[arc > last+1e-6]))
        if self.in_band(tail):
            return
        facts = self.labeler.observe(tail, float(arc[-1]))
        if (facts['excursion'] or facts['switched'] or facts['match_distance'] > DEPARTURE_DISTANCE) and self.current_event is None:
            self.open_event(min(float(arc[-1]), facts['bad_run_start'] if np.isfinite(facts['bad_run_start']) else np.inf))

    def finish(self):
        """Event windows survive thinning; ordinary following is thinned by travel."""
        kept, last = [], -float('inf')
        for row, distance in zip(self.rows, self.distances):
            if row['hard'] or distance-last >= self.stride:
                kept.append(row)
                if not row['hard']:
                    last = distance
        if len(kept) > self.max_states:
            hard = [r for r in kept if r['hard']]
            ordinary = [r for r in kept if not r['hard']]
            nh = min(len(hard), max(self.max_states//2, self.max_states-len(ordinary)))
            select = lambda rows, n: [rows[i] for i in np.linspace(0, len(rows)-1, n).astype(int)] if n else []
            kept = sorted(select(hard, nh)+select(ordinary, min(len(ordinary), self.max_states-nh)),
                          key=lambda r: r['source_row'])
        for row in kept:
            row['replay_class'] = replay_class(row)
        return kept


def append_traces(collectors, rows, track):
    """Kept rows reference their own trace's earlier heads in the shared track."""
    for collector in collectors:
        episode = len({r['episode'] for r in rows})
        for row in collector.finish():
            row.update(seq_start=len(track), seq_end=len(track)+row.pop('prefix_end'), episode=episode)
            row.pop('pre_excursion')
            rows.append(row)
        track.extend(collector.track)


def track_arrays(track):
    return {'track_pos': np.asarray(track, np.float64).reshape(-1, 3)}


def collected_states(rows, track, fibers, provenance):
    """The replay cache for appended collector rows (``append_traces``)."""
    return OnPolicyStates(manifest=fiber_manifest(fibers), provenance=provenance,
                          **{key: np.asarray([row[key] for row in rows]) for key in OnPolicyStates.FIELDS},
                          **track_arrays(track))


class CoverageCursor:
    """Saved, deterministic fiber coverage across collections.

    Each epoch is a seeded permutation of all fibers; a collection continues where the
    previous one stopped, so unseen fibers come before repeats, and a fiber is never
    traced twice in one collection. Each fiber alternates direction across its visits
    (random on the first).
    """
    def __init__(self, fibers, seed, state=None, length_power=0.):
        self.digest = hashlib.sha256(json.dumps(fiber_manifest(fibers), sort_keys=True).encode()).hexdigest()
        self.count, self.seed = len(fibers), int(seed)
        if not np.isfinite(length_power) or length_power < 0:
            raise ValueError('Coverage length power must be finite and nonnegative')
        self.length_power = float(length_power)
        lengths = np.asarray(fibers.lengths if hasattr(fibers, 'lengths') else [f.length for f in fibers], np.float64)
        self.weights = lengths**self.length_power if self.length_power else None
        if state is None:
            state = dict(manifest=self.digest, seed=self.seed, epoch=0, position=0, order=self.permutation(0),
                         directions=[[0, 0] for _ in range(self.count)], length_power=self.length_power)
        if state['manifest'] != self.digest or len(state['directions']) != self.count or len(state['order']) != self.count:
            raise ValueError('Coverage cursor belongs to different fibers')
        if float(state.get('length_power', 0.)) != self.length_power:
            raise ValueError('Coverage cursor uses a different fiber length weighting')
        self.state = state

    @classmethod
    def load(cls, fibers, path, seed, length_power=0.):
        return cls(fibers, seed, json.loads(Path(path).read_text()) if path and Path(path).exists() else None,
                   length_power)

    def permutation(self, epoch):
        """Uniform, or weighted by length**power (sampling without replacement, Efraimidis-Spirakis)."""
        rng = np.random.default_rng([self.seed, epoch])
        if self.weights is None:
            return rng.permutation(self.count).tolist()
        keys = np.log(rng.random(self.count))/np.maximum(self.weights, 1e-300)
        return np.argsort(-keys, kind='stable').tolist()

    def take(self, rng, exclude=()):
        """Next (fiber, sign) in coverage order outside ``exclude``; advances the cursor."""
        state = self.state
        if state['position'] >= self.count:
            state.update(epoch=state['epoch']+1, position=0, order=self.permutation(state['epoch']+1))
        order, position = state['order'], state['position']
        j = next((k for k in range(position, self.count) if order[k] not in exclude), None)
        if j is None:
            return None
        order[position], order[j] = order[j], order[position]
        fi = int(order[position])
        state['position'] += 1
        negative, positive = state['directions'][fi]
        sign = -1. if negative < positive else 1. if positive < negative else float(rng.choice((-1., 1.)))
        state['directions'][fi][0 if sign < 0 else 1] += 1
        return fi, sign

    def save(self, path):
        temporary = Path(path).with_suffix('.partial.json')
        temporary.write_text(json.dumps(self.state))
        os.replace(temporary, path)


def select_seeds(fibers, vol, cursor, count, rng, band=None):
    """One directed seed per distinct fiber; skipped fibers keep explicit reasons."""
    seeds, skipped, tried = [], [], set()
    while len(seeds) < count and len(tried) < len(fibers):
        taken = cursor.take(rng, tried)
        if taken is None:
            break
        fi, sign = taken
        tried.add(fi)
        try:
            seed = directed_seed(fibers[fi], vol, rng, sign)
        except SeedHeadingError as error:
            skipped.append(dict(fiber=fi, reason='ct_heading', detail=str(error)))
            continue
        if seed is None:
            skipped.append(dict(fiber=fi, reason='too_short'))
        elif band is not None and band.lo-64 <= seed['pos'][2] < band.hi+64:
            skipped.append(dict(fiber=fi, reason='holdout'))
        else:
            seeds.append(dict(seed, fiber=int(fi)))
    return seeds, skipped


def supply_summary(rows, reasons, seconds, traced):
    """Collected class and event supply; reported with every collection."""
    episodes = {}
    for row in rows:
        episodes.setdefault(row['episode'], []).append(row)
    events = {name: len({(r['episode'], r['event_id']) for r in rows if r['replay_class'] == i})
              for i, name in enumerate(REPLAY_CLASSES)}
    return dict(states=len(rows), traces=len(reasons), fibers=len({r['fiber_idx'] for r in rows}),
                rows={name: sum(r['replay_class'] == i for r in rows) for i, name in enumerate(REPLAY_CLASSES)},
                events=events, supervision={name: sum(int(r['supervision']) == i for r in rows)
                                            for i, name in enumerate(SUPERVISION)},
                departed_traces=sum(any(np.isfinite(r['departure_distance']) for r in group) for group in episodes.values()),
                terminal_traces=sum(any(int(r['supervision']) == TERMINAL for r in group) for group in episodes.values()),
                stop_reasons={reason: reasons.count(reason) for reason in sorted(set(reasons))},
                traced_voxels=float(traced), seconds=float(seconds), voxels_per_second=float(traced/max(seconds, 1e-9)))


def main(argv=None, *, checkpoint_loader, tracer_class=ModelTracer, bank_loader=None, dataset_loader=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--fibers', required=True)
    ap.add_argument('--dataset-name', help='Source in the checkpoint dataset configuration')
    ap.add_argument('--fiber-zarrs')
    ap.add_argument('--val-z', type=float, nargs=2, default=(45000., 48500.))
    ap.add_argument('--fibers-per-collection', type=int, default=64,
                    help='Distinct fibers; one seed position and one directed episode each')
    ap.add_argument('--length-power', type=float, default=0.,
                    help='Coverage order favors fibers by length**power (0: uniform), still visiting unseen fibers first')
    ap.add_argument('--coverage-state', help='Coverage cursor to continue (JSON); the updated cursor is '
                                             'written next to --out')
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--forward-chunk', type=int, default=0, help='Rows per model forward call; 0 = whole batch')
    ap.add_argument('--trace-len', type=float, default=768.)
    ap.add_argument('--before', type=float, default=48., help='Dense decisions kept before an excursion')
    ap.add_argument('--after', type=float, default=64., help='Voxels kept after the first terminal failure')
    ap.add_argument('--stride', type=float, default=16., help='Travel between kept ordinary following decisions')
    ap.add_argument('--max-states', type=int, default=192,
                    help='Decisions kept per trace; with --stride 0 and a large cap every decision is kept (sequence '
                         'models train on consecutive decisions of a trace)')
    ap.add_argument('--confidence', type=float, help='Default: the checkpoint operating policy')
    ap.add_argument('--n-commit', type=int, help='Default: the checkpoint operating policy')
    ap.add_argument('--gate', choices=('full', 'prefix'), help='Commit gate. Default: the checkpoint operating policy')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--frame-checkpoint', help='Override the learned frame checkpoint recorded by the follower')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--bank-switch-tolerance', type=float, default=.75)
    ap.add_argument('--bank-own-tolerance', type=float, default=1.5)
    args = ap.parse_args(argv)
    if args.fibers_per_collection < 1 or args.batch < 1 or args.trace_len <= 0 or args.forward_chunk < 0:
        raise ValueError('Positive fibers, batch and trace length required')
    started = time.monotonic()
    if args.threads < 1:
        raise ValueError('Positive collector thread count required')
    torch.set_num_threads(args.threads)
    print(f'Collector PyTorch threads: {torch.get_num_threads()}', flush=True)
    model, crop, n_hist, spec, ck = checkpoint_loader(args.checkpoint, args.device)
    if args.frame_checkpoint:
        from vesuvius.neural_tracing.fiber_follow.tracing.crop_frames import bind_frame_checkpoint
        bind_frame_checkpoint(model.cfg, args.frame_checkpoint)
    policy = checkpoint_policy(ck, model.cfg, confidence=args.confidence, n_commit=args.n_commit, gate=args.gate)
    cfg = SampleConfig(crop=crop, n_history=n_hist, recent_history_points=model.cfg.recent_history_points,
                       n_future=model.cfg.n_future, future_step=model.cfg.future_step,
                       label_tolerance=float(ck.get('tolerance', 1.5)),
                       max_recovery_distance=model.cfg.max_recovery_distance)
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
    rng = np.random.default_rng(args.seed)
    cursor = CoverageCursor.load(train_f, args.coverage_state, args.seed, args.length_power)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        seeds, skipped = select_seeds(train_f, vol, cursor, args.fibers_per_collection, rng, band)
    tracer = tracer_class(model, vol, crop, n_hist,
                          TraceParams.from_policy(policy, max_len=args.trace_len, seed=args.seed,
                                                  forward_chunk=args.forward_chunk), device=args.device)
    rows, track, reasons, traced = [], [], [], 0.
    try:
        for offset in range(0, len(seeds), args.batch):
            chunk = seeds[offset:offset+args.batch]
            collectors = [DecisionCollector(train_f[s['fiber']], s['fiber'], s['t'], s['sign'], cfg, band,
                                            before=args.before, after=args.after, stride=args.stride,
                                            max_states=args.max_states, additional_crops=getattr(tracer, 'additional_crops', ()),
                                            bank_detector=bank_detector) for s in chunk]
            from vesuvius.neural_tracing.fiber_follow.tracing.crop_frames import trace_family_kwargs
            paths, chunk_reasons = tracer.trace(np.stack([s['pos'] for s in chunk]), np.stack([s['heading'] for s in chunk]),
                                                on_decision=lambda i, state: collectors[i](state),
                                                **trace_family_kwargs(tracer, [train_f[s['fiber']].tag for s in chunk]))
            for collector, path, reason in zip(collectors, paths, chunk_reasons):
                if reason != 'oracle':
                    collector.observe_final_path(path)
                traced += float(arclength(path)[-1])
            append_traces(collectors, rows, track)
            reasons.extend(c.censored or r for c, r in zip(collectors, chunk_reasons))
            print(json.dumps(dict(traces=offset+len(chunk), total=len(seeds), states=len(rows))), flush=True)
    finally:
        tracer.close()
    if not rows:
        raise ValueError('No eligible decision states; no cache was published')
    path = Path(args.out)
    supply = supply_summary(rows, reasons, time.monotonic()-started, traced)
    st = collected_states(rows, track, train_f,
                        dict(checkpoint=os.path.abspath(args.checkpoint), step=int(ck.get('step', 0)),
                                        cache_id=f'{path.stem}:{ck.get("step", 0)}:{args.seed}',
                                        operating_policy=policy.to_dict(),
                                        label_contract=dict(tolerance=cfg.label_tolerance,
                                                            max_recovery_distance=cfg.max_recovery_distance,
                                                            departure_distance=DEPARTURE_DISTANCE,
                                                            departure_patience=DEPARTURE_PATIENCE),
                                        seed_heading_policy=SEED_HEADING_POLICY,
                                        heading_policy=getattr(tracer, 'heading_policy', TRACE_HEADING_POLICY),
                                        frame_policy=getattr(tracer, 'frame_policy', FRAME_POLICY),
                                        model_cfg=model.cfg.to_dict(), crop=asdict(crop),
                                        volume=spec.to_dict(), collection=vars(args),
                                        coverage=dict(requested=args.fibers_per_collection, seeds=len(seeds),
                                                      skipped=skipped, epoch=cursor.state['epoch']),
                                        supply=supply,
                                        failure_banks=([b.provenance() for b in bank_detector.banks]
                                                       if bank_detector is not None else [])))
    # Fail in the collector before publishing a cache to live training workers.
    st.validate_fibers(train_f)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.stem+'.partial.npz')
    st.save(temp)
    # Pre-create mmap before publishing so multiple loader workers never race.
    OnPolicyStates.load(temp)
    os.replace(Path(str(temp)[:-4]+'_mmap'), Path(str(path)[:-4]+'_mmap'))
    os.replace(temp, path)
    cursor.save(path.with_suffix('.coverage.json'))
    print(json.dumps(dict(out=str(path), **supply)))
    return str(path)
