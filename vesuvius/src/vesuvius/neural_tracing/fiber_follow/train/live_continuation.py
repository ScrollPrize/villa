"""Recent training predictions become the next source-local supervised state.

Only small CPU geometry crosses the feedback queues. CT reads and labeling stay
in loader workers; no model or autograd graph is retained by a chain. A chain
advances only through prefixes the operating policy accepts, continues through
following and recoverable states, and ends after retaining a terminal example.
Starts are restored recorded prefixes (observed path, seed reference, heading
state, correspondence and historical events) or fresh seed-only states.
"""
import multiprocessing as mp
import os
from queue import Empty, Full

import numpy as np
import torch

from vesuvius.neural_tracing.fiber_follow.data.data import SOURCE, label_state
from vesuvius.neural_tracing.fiber_follow.shared.geometry import frame_from_heading
from vesuvius.neural_tracing.fiber_follow.tracing.heading import FRAME_POLICY, SeedHeadingError, ct_frame, reframe_item
from vesuvius.neural_tracing.fiber_follow.tracing.policy import commit_prefix
from vesuvius.neural_tracing.fiber_follow.shared.reference import SEED_FIELDS, observed_path
from vesuvius.neural_tracing.fiber_follow.data.state_labels import FOLLOWING, RECOVERABLE, TERMINAL, TraceLabeler
from vesuvius.neural_tracing.fiber_follow.tracing.trace import TraceParams, advance_trace_path, trace_history

OUTCOMES = ('advanced', 'stale', 'censored', 'empty')


def preserve_live_metadata(batch):
    """Keep CPU NumPy geometry out of DataLoader's default tensor conversion.

    Observations/targets are already collated tensors. Geometry is ordinary
    pickle data, never tensor IPC storage or pinned image input.
    """
    return batch


def chain_memory(item, depth):
    """Chain identity and the decision records a chain's next state reads (decision memory).

    This state's own decision is recorded under key ``depth``; the trainer stores its
    encoder entry under (chain, depth) when the chain advances. A chain started from a
    recorded prefix carries simulated, unrecorded decisions for that prefix.
    """
    from vesuvius.neural_tracing.fiber_follow.shared.geometry import arclength
    from vesuvius.neural_tracing.fiber_follow.models.decision_memory import prune_decisions
    from vesuvius.neural_tracing.fiber_follow.data.decision_memory import simulated_decisions
    chain_id = item.get('live_chain_id')
    if chain_id is None:
        chain_id = int.from_bytes(os.urandom(7), 'little')
    path = observed_path(item)
    length = float(arclength(path)[-1])
    decisions = item.get('memory_decisions')
    if decisions is None:
        decisions = item.get('_simulated_decisions') or simulated_decisions(path, arclength(path))
    decisions = [d for d in decisions if d['travelled'] < length-1e-6]
    decisions = decisions+[dict(travelled=length, pos=np.asarray(item['pos']).copy(),
                                frame=np.asarray(item['frame']).copy(), key=int(depth))]
    keep = prune_decisions([d['travelled'] for d in decisions], length)
    return chain_id, [decisions[k] for k in keep]


class DecisionStore:
    """Encoder entries of live-chain decisions, on the training device, keyed by (chain, key).

    Entries are detached BF16 features recorded by the training forward of the decision
    itself. A chain's entries are pruned to its published decision records; chains not
    advanced for ``max_age`` updates are dropped (their next state would be stale anyway).
    """
    def __init__(self, max_age):
        self.max_age = max_age
        self.chains = {}

    def put(self, chain_id, key, features, keep, step):
        entries, _ = self.chains.get(chain_id, ({}, step))
        entries[key] = features.detach()
        self.chains[chain_id] = ({k: v for k, v in entries.items() if k in keep}, step)

    def gather(self, chains, keys, shape, device):
        """(B, SLOTS, ...) features for recorded slots and a found mask."""
        import torch
        found = torch.zeros(keys.shape, dtype=torch.bool)
        features = torch.zeros((*keys.shape, *shape), dtype=torch.bfloat16, device=device)
        for row, chain in enumerate(chains.tolist()):
            entries = self.chains.get(chain, ({}, 0))[0]
            for slot, key in enumerate(keys[row].tolist()):
                if key >= 0 and key in entries:
                    features[row, slot] = entries[key]
                    found[row, slot] = True
        return features, found

    def evict(self, step):
        self.chains = {c: v for c, v in self.chains.items() if step-v[1] <= self.max_age}

    def __len__(self):
        return sum(len(entries) for entries, _ in self.chains.values())


class LiveContinuationSource:
    """Per-source chain queues; chain limits are balanced over three bands of [min, max]."""
    def __init__(self, *, policy, steps=(12, 32), capacity=32, max_age=64,
                 switch_tolerance=.75, own_tolerance=1.5, step=None):
        self.steps, self.max_age = tuple(steps), max_age
        if len(self.steps) != 2 or not 1 <= self.steps[0] <= self.steps[1]:
            raise ValueError('Invalid live chain step range')
        self.limit_bands = np.array_split(np.arange(self.steps[0], self.steps[1]+1), min(3, self.steps[1]-self.steps[0]+1))
        self.policy = policy
        self.params = TraceParams.from_policy(policy)
        self.switch_tolerance, self.own_tolerance = switch_tolerance, own_tolerance
        ctx = mp.get_context()
        self.chains, self.seeds = ctx.Queue(capacity), ctx.Queue(capacity)
        self.step = ctx.Value('q', 0) if step is None else step
        self.limit_cursor = ctx.Value('q', 0)
        self.detector = None
        self.outcomes = dict.fromkeys(OUTCOMES, 0)

    def draw_limit(self, rng):
        # One counter per source shared by all loader workers: balance starts,
        # regardless of worker scheduling or how long preceding chains survive.
        with self.limit_cursor.get_lock():
            band = self.limit_bands[self.limit_cursor.value % len(self.limit_bands)]
            self.limit_cursor.value += 1
        return int(rng.integers(int(band[0]), int(band[-1])+1))

    def placeholder(self, dataset, rng, windows):
        """A chain start that is also a valid training item if no live state is ready."""
        item = dataset.live_start(rng, windows)
        item.update(live_requested=True)
        return item

    def metadata(self, item):
        """Chain state for an item the policy may continue from, else None."""
        if not (item.get('live_start') or item.get('live_continuation')):
            return None
        if item['supervision'] not in (FOLLOWING, RECOVERABLE) or not item['geometry_valid']:
            return None
        depth = int(item.get('live_depth', 0))
        limit = int(item.get('live_limit', self.steps[1]))
        if depth >= limit:
            return None
        fi, traversal_t, reverse = item['fiber_ref']
        labeler = item.get('labeler_state') or dict(
            t=float(item['trace_facts']['t']), last_travelled=float(item.get('travelled', 0.)), bad_run=0,
            bad_run_start=None, started=True, departure_distance=None, boundary_distance=None, switch=None)
        chain_id, decisions = chain_memory(item, depth)
        return dict(pos=np.asarray(item['pos']), frame=np.asarray(item['frame']),
                    chain_id=chain_id, memory_decisions=decisions,
                    observed_path=observed_path(item), fiber_idx=fi, reverse=reverse,
                    heading_start=int(item.get('heading_start', 0)),
                    loop_start=int(item['live_loop_start']),
                    travelled=float(labeler['last_travelled']), depth=depth, limit=limit, labeler=labeler,
                    start=item.get('live_start') or item.get('live_chain_start'),
                    **{key: item[key] for key in SEED_FIELDS})

    def publish(self, state):
        queue = self.chains if state['depth'] else self.seeds
        try:
            queue.put_nowait(state)
            return True
        except Full:
            return False

    def take_outcomes(self):
        outcomes, self.outcomes = self.outcomes, dict.fromkeys(OUTCOMES, 0)
        return outcomes

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
                    self.outcomes['stale'] += 1
                    continue
                try:
                    item = self.advance(state, dataset, vol, rng)
                except SeedHeadingError:
                    # As with ordinary sampling, invalid CT orientation/context
                    # rejects this proposal. I/O failures still propagate.
                    item = None
                if item is not None:
                    self.outcomes['advanced'] += 1
                    item.update(live_requested=True, task_requested=fallback['task_requested'],
                                task_delivered=fallback['task_requested'], task_fallback=0)
                    return item
                self.outcomes['censored'] += 1
        self.outcomes['empty'] += 1
        return fallback

    def advance(self, state, dataset, vol, rng):
        advanced, _ = advance_trace_path(state['observed_path'], state['frame'], state['points'],
            state['commit'], state['heading_start'], vol.shape, self.params,
            state['travelled'], state['loop_start'])
        if advanced is None:
            return None
        path = advanced['path']
        if dataset.exclude is not None and np.any((advanced['last_segment'][:, 2] >= dataset.exclude.lo-48)
                                                  & (advanced['last_segment'][:, 2] < dataset.exclude.hi+48)):
            return None
        hist, mask = trace_history(path, dataset.cfg.n_history)
        frame = frame_from_heading(advanced['heading'], state['frame'][:, 0])
        fi, reverse = state['fiber_idx'], state['reverse']
        fiber = dataset.fibers[fi]
        if self.detector is None:
            from vesuvius.neural_tracing.fiber_follow.data.bank_geometry import BankSwitchDetector
            builder = dataset.batch_builder
            banks = list({id(b): b for b in (getattr(builder, 'negative_bank', None),
                getattr(builder, 'near_negative_bank', None), getattr(builder, 'continuation_bank', None))
                if b is not None}.values())
            self.detector = BankSwitchDetector(banks, self.switch_tolerance, self.own_tolerance)
        labeler = TraceLabeler(fiber, state['labeler']['t'], -1 if reverse else 1,
                               tolerance=dataset.cfg.label_tolerance,
                               max_recovery_distance=dataset.cfg.max_recovery_distance,
                               fiber_idx=fi, bank_detector=self.detector, state=state['labeler'])
        facts = labeler.observe(advanced['last_segment'], advanced['travelled'])
        item = label_state(fiber, path[-1], frame, hist, mask, dataset.cfg,
                           t=facts['t'], reverse=reverse, trace=facts)
        if not item['confidence_valid']:
            return None  # censored: no correspondence, unannotated, ambiguous or unobservable
        # Unknown states with partial labels (an uncertified connection) are delivered but
        # end the chain, like terminal states: only following/recoverable states advance.
        terminal = item['supervision'] == TERMINAL
        item.update({key: state[key] for key in SEED_FIELDS})
        item['seed_age'] = state['seed_age']+advanced['travelled']-state['travelled']
        item.update(memory_decisions=state['memory_decisions'], live_chain_id=state['chain_id'])
        item.update(observed_path=path, heading_start=advanced['heading_start'],
            fiber_ref=(fi, fiber.length-facts['t'] if reverse else facts['t'], reverse),
            source=SOURCE['live'], source_step=state['source_step'], travelled=advanced['travelled'],
            live_continuation=True, live_terminal=terminal, live_depth=state['depth']+1,
            live_limit=(self.draw_limit(rng) if state['depth'] == 0 else state['limit']),
            live_travelled=advanced['travelled'], live_loop_start=state['loop_start'],
            live_chain_start=state['start'], labeler_state=labeler.state_dict())
        item = dataset.prepare(item, rng)
        if not dataset.state_allowed(item):
            return None
        # The unresolved-frame footprint covers all rolls and the CT tensor.
        dataset.prefetch_items([item], vol, required=True)
        from vesuvius.neural_tracing.fiber_follow.tracing.crop_frames import frame_predictor, orient_items
        item['fiber_family'] = fiber.tag
        predictor = frame_predictor(getattr(dataset.batch_builder, 'cfg', None))
        if predictor is None:
            diagnostics = {}
            frame = ct_frame(vol, item['pos'], advanced['heading'], state['frame'], diagnostics=diagnostics)
            reframe_item(item, frame)
            item.update(frame_policy=FRAME_POLICY, ct_frame_diagnostics=diagnostics)
        else:
            orient_items([item], vol, predictor, previous=[state['frame']])
        if hasattr(dataset.batch_builder, 'apply_roll'):
            dataset.batch_builder.apply_roll(item)
        return item

    def close(self):
        for queue in (self.chains, self.seeds):
            queue.cancel_join_thread()
            queue.close()


class LiveContinuation:
    def __init__(self, dataset, *, policy, steps, switch_tolerance, own_tolerance):
        datasets = getattr(dataset, 'datasets', [dataset])
        self.sources = []
        self.policy = policy
        for source in datasets:
            live = LiveContinuationSource(policy=policy, steps=steps, capacity=max(32, source.chunk*4),
                switch_tolerance=switch_tolerance, own_tolerance=own_tolerance, step=source.step)
            source.live_continuation = live
            self.sources.append(live)
        # Decision memory: generous age bound, since a hop may wait in loader queues.
        self.memory = DecisionStore(max_age=4*self.sources[0].max_age if self.sources else 256)

    def feedback(self, cpu, output, step):
        """Advance chains with the operating policy's commit on the trained prediction.

        ``output`` already went through same-position refinement and selection; a
        rejected decision publishes nothing, so a chain never advances past a stop.
        """
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
        counts, _ = commit_prefix(points, confidence, self.policy.confidence,
                                  self.policy.n_commit, self.policy.max_recovery_distance)
        points, counts = points.cpu().numpy(), counts.cpu().numpy()
        entries = output.get('memory_entry')
        for i, proposal, count in zip(indices, points, counts):
            if count:
                state = states[i]
                if entries is not None:
                    keep = {d['key'] for d in state['memory_decisions'] if d.get('key') is not None}
                    self.memory.put(state['chain_id'], state['memory_decisions'][-1]['key'], entries[i], keep, step)
                source.publish(dict(state, points=proposal, commit=int(count), source_step=step))
        if entries is not None:
            self.memory.evict(step)

    def attach_memory(self, x, shape):
        """Recorded chain entries (B, SLOTS, *shape) and which slots were found; ``shape`` is the
        model's ``cfg.memory_entry_shape``."""
        device = x['history_keys'].device
        features, found = self.memory.gather(x['history_chain'].cpu(), x['history_keys'].cpu(),
                                             tuple(shape), device)
        return features, found.to(device)

    def close(self):
        for source in self.sources:
            source.close()
