"""Recent training predictions become the next source-local supervised state.

Only small CPU geometry crosses the feedback queues. CT reads and labeling stay
in loader workers; no model or autograd graph is retained by a chain.
"""
import multiprocessing as mp
from queue import Empty, Full

import numpy as np
import torch

from ..shared.collect import DecisionCollector
from ..shared.data import label_state, make_sample
from ..shared.geometry import frame_from_heading
from ..shared.heading import FRAME_POLICY, SeedHeadingError, ct_frame, reframe_item
from ..shared.policy import commit_prefix
from ..shared.reference import SEED_FIELDS, observed_path
from ..shared.trace import TraceParams, advance_trace_path, trace_history


def preserve_live_metadata(batch):
    """Keep CPU NumPy geometry out of DataLoader's default tensor conversion.

    Observations/targets are already collated tensors. Geometry is ordinary
    pickle data, never tensor IPC storage or pinned image input.
    """
    return batch


class LiveContinuationSource:
    def __init__(self, *, steps=(4, 8), capacity=32, max_age=64, n_commit=16,
                 max_recovery_distance=6., switch_tolerance=.75, own_tolerance=1.5, stratified=False):
        self.steps, self.max_age = tuple(steps), max_age
        if len(self.steps) != 2 or not 1 <= self.steps[0] <= self.steps[1]:
            raise ValueError('Invalid live chain step range')
        self.stratified = stratified
        self.limit_bands = np.array_split(np.arange(self.steps[0], self.steps[1]+1), min(3, self.steps[1]-self.steps[0]+1))
        self.params = TraceParams(n_commit=n_commit)
        self.max_recovery_distance = max_recovery_distance
        self.switch_tolerance, self.own_tolerance = switch_tolerance, own_tolerance
        ctx = mp.get_context()
        self.chains, self.seeds = ctx.Queue(capacity), ctx.Queue(capacity)
        self.step = ctx.Value('q', 0)
        self.limit_cursor = ctx.Value('q', 0) if stratified else None
        self.detector = None

    def draw_limit(self, rng):
        if not self.stratified:
            return int(rng.integers(self.steps[0], self.steps[1]+1))
        # One counter per source shared by all loader workers: balance starts,
        # regardless of worker scheduling or how long preceding chains survive.
        with self.limit_cursor.get_lock():
            band = self.limit_bands[self.limit_cursor.value % len(self.limit_bands)]
            self.limit_cursor.value += 1
        return int(rng.integers(int(band[0]), int(band[-1])+1))

    def placeholder(self, dataset, rng, *, light=False):
        fi = int(rng.choice(len(dataset.fibers), p=dataset.weights))
        fiber = dataset.fibers[fi]
        t, reverse = float(rng.uniform(0, fiber.length)), bool(rng.integers(2))
        item = make_sample(fiber, t, reverse, dataset.cfg, rng, perturb=False)
        item.update(fiber_ref=(fi, t, reverse), source=0, source_step=-1, stratum=-1,
                    gt_unperturbed=True, live_requested=True, live_light_slot=light,
                    live_fallback=True)
        return dataset.prepare(item, rng)

    def metadata(self, item):
        # Clean sampled states bootstrap chains. Synthetic identity/wrong-turn
        # rows and old replay never seed the live continuation distribution.
        if not (item.get('gt_unperturbed') or item.get('live_continuation')):
            return None
        if item.get('offtrack', False):
            return None
        depth = int(item.get('live_depth', 0))
        limit = int(item.get('live_limit', self.steps[1]))
        if depth >= limit:
            return None
        path = observed_path(item)
        fi, traversal_t, reverse = item['fiber_ref']
        return dict(pos=np.asarray(item['pos']), frame=np.asarray(item['frame']),
                    observed_path=path, fiber_idx=fi, traversal_t=traversal_t, reverse=reverse,
                    heading_start=int(item.get('heading_start', 0)),
                    loop_start=int(item.get('live_loop_start', len(path)-1)),
                    travelled=float(item.get('live_travelled', 0.)), depth=depth, limit=limit,
                    **{key: item[key] for key in SEED_FIELDS})

    def publish(self, state):
        queue = self.chains if state['depth'] else self.seeds
        try:
            queue.put_nowait(state)
            return True
        except Full:
            return False

    def resolve(self, fallback, dataset, vol, rng):
        # Resolve at image-build time, after geometry lookahead. Planned batches
        # must not reserve stale chains many updates before they are consumed.
        for queue in (self.chains, self.seeds):
            for _ in range(16):
                try:
                    state = queue.get_nowait()
                except Empty:
                    break
                if self.step.value-state['source_step'] > self.max_age:
                    continue
                try:
                    item = self.advance(state, dataset, vol, rng)
                except SeedHeadingError:
                    # As with ordinary sampling, invalid CT orientation/context
                    # rejects this proposal. I/O failures still propagate.
                    continue
                if item is not None:
                    item['live_requested'] = True
                    item['live_light_slot'] = fallback.get('live_light_slot', False)
                    return item
        return fallback

    def advance(self, state, dataset, vol, rng):
        advanced, _ = advance_trace_path(state['observed_path'], state['frame'], state['points'],
            state['commit'], state['heading_start'], vol.shape, self.params,
            state['travelled'], state['loop_start'])
        if advanced is None:
            return None
        path = advanced['path']
        hist, mask = trace_history(path, dataset.cfg.n_history)
        frame = frame_from_heading(advanced['heading'], state['frame'][:, 0])
        fi, reverse = state['fiber_idx'], state['reverse']
        fiber = dataset.fibers[fi]
        t = fiber.length-state['traversal_t'] if reverse else state['traversal_t']
        if self.detector is None:
            from .bank_geometry import BankSwitchDetector
            builder = dataset.batch_builder
            banks = list({id(b): b for b in (getattr(builder, 'negative_bank', None),
                getattr(builder, 'near_negative_bank', None), getattr(builder, 'continuation_bank', None))
                if b is not None}.values())
            self.detector = BankSwitchDetector(banks, self.switch_tolerance, self.own_tolerance)
        oracle = DecisionCollector(fiber, fi, t, -1 if reverse else 1, dataset.cfg,
                                   dataset.exclude, bank_detector=self.detector)
        oracle.last_travelled = state['travelled']
        current = dict(pos=path[-1], frame=frame, hist=hist, hmask=mask,
            observed_path=path, last_segment=advanced['last_segment'],
            travelled=advanced['travelled'], heading_start=advanced['heading_start'],
            would_stop=False, exploratory=False,
            **{key: state[key] for key in SEED_FIELDS})
        current['seed_age'] += advanced['travelled']-state['travelled']
        if not oracle(current):
            return None
        row = oracle.rows[-1]
        item = label_state(fiber, current['pos'], frame, hist, mask, dataset.cfg,
                           t=oracle.t, reverse=reverse, offtrack=row['offtrack'])
        item.update({key: current[key] for key in SEED_FIELDS})
        item.update(observed_path=path, heading_start=advanced['heading_start'],
            fiber_ref=(fi, fiber.length-oracle.t if reverse else oracle.t, reverse),
            source=2, source_step=state['source_step'], stratum=4 if row['offtrack'] else 0,
            failure_kind=int(row.get('failure_kind', 0)),
            live_continuation=True, live_correct_continuation=not row['offtrack'],
            live_failure=bool(row['offtrack']), live_depth=state['depth']+1,
            live_limit=(self.draw_limit(rng) if state['depth'] == 0 else state['limit']),
            live_travelled=advanced['travelled'], live_loop_start=state['loop_start'])
        item = dataset.prepare(item, rng)
        if not dataset.state_allowed(item):
            return None
        # The unresolved-frame footprint covers all rolls and the CT tensor.
        dataset.prefetch_items([item], vol, required=True)
        diagnostics = {}
        frame = ct_frame(vol, current['pos'], advanced['heading'], state['frame'], diagnostics=diagnostics)
        reframe_item(item, frame)
        item.update(frame_policy=FRAME_POLICY, ct_frame_diagnostics=diagnostics)
        return item

    def close(self):
        for queue in (self.chains, self.seeds):
            queue.cancel_join_thread()
            queue.close()


class LiveContinuation:
    def __init__(self, dataset, *, steps, n_commit, max_recovery_distance,
                 switch_tolerance, own_tolerance, stratified=False):
        datasets = getattr(dataset, 'datasets', [dataset])
        self.sources = []
        for source in datasets:
            live = LiveContinuationSource(steps=steps, stratified=stratified, capacity=max(32, source.chunk*4),
                n_commit=n_commit, max_recovery_distance=max_recovery_distance,
                switch_tolerance=switch_tolerance, own_tolerance=own_tolerance)
            source.live_continuation = live
            self.sources.append(live)

    def set_step(self, step):
        for source in self.sources:
            source.step.value = step

    def feedback(self, cpu, output, step):
        states = cpu.get('_live_states')
        if states is None:
            return
        indices = [i for i, state in enumerate(states) if state is not None]
        if not indices:
            return
        ids = cpu.get('dataset_id', torch.zeros(len(states), dtype=torch.long))
        # Every batch comes from a single volume and therefore a single policy.
        source = self.sources[int(ids[0])]
        points = output['points'][indices].detach().float()
        confidence = output['confidence'][indices].detach().float()
        counts, _ = commit_prefix(points, confidence, source.params.confidence,
                                  source.params.n_commit, source.max_recovery_distance)
        points, counts = points.cpu().numpy(), counts.cpu().numpy()
        for i, proposal, count in zip(indices, points, counts):
            if count:
                source.publish(dict(states[i], points=proposal, commit=int(count), source_step=step))

    def close(self):
        for source in self.sources:
            source.close()
