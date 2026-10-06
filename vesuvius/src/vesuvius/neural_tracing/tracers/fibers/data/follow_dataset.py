"""The training dataset: draws task-budgeted fresh, live, replay and synthetic decisions per source."""
from __future__ import annotations

import os
from collections import deque
import json

import numpy as np
import torch

from vesuvius.neural_tracing.tracers.shared.geometry import crop_local_grid
from vesuvius.neural_tracing.tracers.shared.crop_sampling.volume import CTVolume
from vesuvius.neural_tracing.tracers.shared.reference import SEED_FIELDS
from vesuvius.neural_tracing.tracers.fibers.data.episodes import episode_decisions, episode_tensors, mark_episode_identity
from vesuvius.neural_tracing.tracers.fibers.data.replay import EVENT_TABLE_SIZE, ReplayIndex, load_replay, replay_facts, replay_labeler_state, usable_replay
from vesuvius.neural_tracing.tracers.fibers.data.samples import STARTUP_CATEGORIES, make_sample
from vesuvius.neural_tracing.tracers.fibers.data.targets import label_state, resolve_trace_seed
from vesuvius.neural_tracing.tracers.fibers.data.tasks import FALLBACKS, SOURCE, TASK, TASKS, TaskBudget, stable_key


MAX_CT_FRAME_REJECTIONS = 64


class FollowDataset(torch.utils.data.IterableDataset):
    """Explicit task budget per batch slot; every draw is relabeled before crop I/O.

    Tasks follow ``TaskBudget``: fresh simulated traces (with startup and excursion
    allocations), live continuation (its own slots, never replay slots), five
    mutually exclusive DAgger replay classes and certified synthetic failures.
    Missing replay falls back to a fresh example of the corresponding kind; missing
    terminal replay may use certified synthetic failures within the configured cap.
    Fallbacks and deficits are recorded per item. Replay age ceilings and per-event
    draw caps are shared across loader workers through shared memory.
    """
    def __init__(self, fibers, vol_spec, cfg, chunk=2, seed=0,
                 cache_bytes=1 << 30, onpolicy=None,
                 window=256., pool_size=12, window_samples=192, replay_index=None, refresh_chunks=8,
                 batch_builder=None, budget=None, length_power=1.):
        import multiprocessing as mp
        self.budget = budget or TaskBudget()
        self.fibers, self.vol_spec, self.cfg = fibers, vol_spec, cfg
        self.chunk, self.seed, self.cache_bytes = chunk, seed, cache_bytes
        self.window, self.pool_size, self.window_samples = window, pool_size, window_samples
        self.replay_index, self.refresh_chunks = replay_index, refresh_chunks
        self.batch_builder = batch_builder
        self.live_continuation = None  # Optional trainer-to-worker prediction feedback.
        self.remote_prefetch = None  # Optional trainer-owned process queue client.
        self.episodes = None  # EpisodeSpec: plans of whole episodes (model 'sequence'); chunk then counts episodes
        self.remote_prefetch_lookahead = 0  # Planned microbatches per source/worker.
        # Shared by every loader worker: current update and per-event replay draw counts.
        self.step = mp.Value('q', 0)
        self.event_draws = mp.Array('i', EVENT_TABLE_SIZE)
        self._replay_paths = None
        self._set_replay(list(onpolicy or []))
        lengths = (fibers.lengths if hasattr(fibers, 'lengths')
                   else np.array([f.length for f in fibers]))
        if not len(lengths):
            raise ValueError('No training fibers')
        if not np.isfinite(length_power) or length_power < 0:
            raise ValueError('Fiber length power must be finite and nonnegative')
        # Power 1 draws uniformly over annotated arclength; larger powers favor long fibers.
        lengths = np.asarray(lengths, dtype=np.float64)
        if length_power != 1:
            lengths = lengths**length_power
        self.weights = lengths / lengths.sum()
        self.terminal_requests = self.synthetic_fallbacks = 0

    def set_step(self, step):
        self.step.value = int(step)

    def _set_replay(self, caches):
        if caches:
            caches = usable_replay(caches, self.fibers, self.cfg.n_history, self.vol_spec.grid_scale)
        self.onpolicy = caches
        self.index = ReplayIndex(caches)
        self._eligible = (None, frozenset())

    def eligible_caches(self):
        step = int(self.step.value)
        if self._eligible[0] != step:
            self._eligible = (step, frozenset(i for i, op in enumerate(self.onpolicy)
                                              if step-int(op.provenance['step']) <= self.budget.replay_max_age))
        return self._eligible[1]

    def claim(self, key):
        """Count one draw of an event against its shared cap; False once exhausted."""
        slot = key % EVENT_TABLE_SIZE
        with self.event_draws.get_lock():
            if self.event_draws[slot] >= self.budget.replay_event_cap:
                return False
            self.event_draws[slot] += 1
            return True

    def max_event_reuse(self):
        return int(max(self.event_draws[:])) if len(self.event_draws) else 0

    def refresh_replay(self):
        if self.replay_index is None or not os.path.exists(self.replay_index):
            return
        with open(self.replay_index) as fh:
            paths = json.load(fh)
        if paths != self._replay_paths:
            self._set_replay(load_replay(paths))
            self._replay_paths = paths

    def prepare(self, item, rng):
        """Let the batch builder attach geometry it reads later (e.g. path patches)."""
        if hasattr(self.batch_builder, 'prepare'):
            return self.batch_builder.prepare(item, self.fibers[item['fiber_ref'][0]], rng)
        return item

    # ----------------------------------------------------------------- task items

    def fresh_location(self, rng, windows):
        location = (self.batch_builder.fresh_location(rng)
                    if hasattr(self.batch_builder, 'fresh_location') else None)
        if location is not None:
            return location['fiber'], location['t'], location['reverse'], location['source']
        while len(windows) < self.pool_size:
            fi = rng.choice(len(self.fibers), p=self.weights)
            windows.append([fi, rng.uniform(0, self.fibers[fi].length), self.window_samples])
        wi = rng.integers(len(windows))
        fi, center, _ = windows[wi]
        f = self.fibers[fi]
        windows[wi][2] -= 1
        if windows[wi][2] <= 0:
            windows.pop(wi)
        rev = bool(rng.integers(2))
        if rng.random() < .1:
            t = max(0, f.length-rng.uniform(0, self.cfg.future_s[-1]*1.5))
        else:
            original_t = np.clip(center+rng.uniform(-self.window/2, self.window/2), 0, f.length)
            t = f.length-original_t if rev else original_t
        return int(fi), float(t), rev, 0

    def fresh_item(self, rng, windows, **options):
        """A simulated trace; ``options`` fix the startup category or force an excursion."""
        item = (self.batch_builder.replace_fresh(self.cfg, rng, **options)
                if hasattr(self.batch_builder, 'replace_fresh') else None)
        if item is None:
            fi, t, rev, location = self.fresh_location(rng, windows)
            item = make_sample(self.fibers[fi], t, rev, self.cfg, rng, **options)
            item.update(fiber_ref=(fi, t, rev), location_source=location)
            item = self.prepare(item, rng)
        item.update(source=SOURCE['fresh'], source_step=-1)
        return item

    def recovery_item(self, rng, windows):
        """Fresh substitute for missing recoverable replay: a displaced excursion head."""
        from vesuvius.neural_tracing.tracers.fibers.data.state_labels import RECOVERABLE
        for _ in range(8):
            item = self.fresh_item(rng, windows, startup=STARTUP_CATEGORIES.index('established'), excursion=True)
            if item['supervision'] == RECOVERABLE:
                return item
        return item

    def synthetic_item(self, rng):
        """Certified wrong continuation with visible original-fiber evidence, or None."""
        if not hasattr(self.batch_builder, 'synthetic_terminal'):
            return None
        for _ in range(3):
            item = self.batch_builder.synthetic_terminal(self.cfg, rng)
            if item is None:
                continue
            item = self.prepare(item, rng)
            if item.get('identity_evidence', False):
                return item
        return None

    def replay_draw(self, name, rng):
        """One relabeled row of a replay class, honoring the age ceiling and event cap."""
        eligible = self.eligible_caches()
        for _ in range(8):
            group = self.index.draw(name, rng, eligible)
            if group is None:
                return None
            if not self.claim(group[4]):
                continue
            ci, episode, event, rows, key = group
            item = self.replay_item(self.onpolicy[ci], int(rng.choice(rows)), rng)
            item.update(replay_event=key, replay_episode=stable_key(key, episode))
            return item
        return None

    def replay_item(self, op, j, rng):
        from vesuvius.neural_tracing.tracers.fibers.frames.ct import FRAME_POLICY
        fi, t, reverse = int(op.fiber_idx[j]), float(op.t[j]), bool(op.reverse[j])
        item = label_state(self.fibers[fi], op.pos[j], op.frame[j], op.hist[j], op.hmask[j], self.cfg,
                           t=t, reverse=reverse, trace=replay_facts(op, j, self.cfg))
        item.update(source=SOURCE['replay'], source_step=int(op.provenance['step']),
                    fiber_ref=(fi, self.fibers[fi].length-t if reverse else t, reverse),
                    replay_class=int(op.replay_class[j]), travelled=float(op.travelled[j]),
                    heading_start=int(op.heading_start[j]), frame_policy=op.provenance.get('frame_policy', FRAME_POLICY),
                    labeler_state=replay_labeler_state(op, j))
        item.update({k: getattr(op, k)[j] for k in SEED_FIELDS})
        item['observed_path'] = op.observed_prefix(j)
        return self.prepare(item, rng)

    def episode_items(self, rng, windows, index):
        """One episode (``self.episodes``) on a fresh fiber location: decisions in step order, each prepared like a
        fresh item; the episode shares one photometric draw (one scan) while frames and rolls stay per step."""
        spec = self.episodes
        for _ in range(1000):
            fi, _, reverse, location = self.fresh_location(rng, windows)
            length = self.fibers[fi].length
            steps = min(spec.steps, int(length//spec.commit))
            if steps < spec.min_steps:
                continue
            t0 = float(rng.uniform(0., length-steps*spec.commit))
            items, shared = [], None
            for step, (item, segment) in enumerate(episode_decisions(self.fibers[fi], t0, reverse, steps, spec.commit,
                                                                     self.cfg, rng)):
                item.update(fiber_ref=(fi, float(t0+step*spec.commit), reverse), location_source=location)
                item = self.prepare(item, rng)
                if shared is None:
                    shared = {key: item[key] for key in ('photometric', 'blur_sigma') if key in item}
                item.update(shared, source=SOURCE['fresh'], source_step=-1, task_requested=TASK['fresh'],
                            task_delivered=TASK['fresh'], task_fallback=0, episode_index=index, episode_step=step,
                            episode_supervised=step >= steps-spec.supervised, episode_segment=segment)
                items.append(item)
            return mark_episode_identity(items)
        raise ValueError('Could not draw a training episode: no fiber is long enough')

    def replay_episode_items(self, kind, rng, index):
        """One on-policy episode (``self.episodes``) from the replay caches: consecutive recorded decisions of one
        collected trace whose supervised (trailing) window contains a decision of replay class ``kind``. Each step is
        relabeled and prepared as a replay item; its committed segment is the recorded trace from its head to the next
        head (the last step's history token is never read, so its segment is its head alone). The collector keeps every
        decision for sequence models (``--max-states``, stride 0), so a trace's rows are consecutive decisions.
        Returns None when no eligible event is available."""
        spec = self.episodes
        eligible = self.eligible_caches()
        for _ in range(8):
            group = self.index.draw(kind, rng, eligible)
            if group is None:
                return None
            if not self.claim(group[4]):
                continue
            ci, episode, event, members, key = group
            op = self.onpolicy[ci]
            trace = np.flatnonzero(np.asarray(op.episode) == episode)
            trace = trace[np.argsort(np.asarray(op.source_row)[trace])]
            if np.any(np.diff(np.asarray(op.source_row)[trace]) != 1):
                raise ValueError('Replay episodes need every decision of a trace; collect with --max-states and stride 0')
            anchor = int(np.searchsorted(np.asarray(op.source_row)[trace], int(op.source_row[int(rng.choice(members))])))
            end = min(len(trace)-1, anchor+int(rng.integers(spec.supervised)))
            start = max(0, end-spec.steps+1)
            if end-start+1 < spec.min_steps:
                continue
            rows = trace[start:end+1]
            items, shared = [], None
            for step, j in enumerate(rows):
                item = self.replay_item(op, int(j), rng)
                if shared is None:
                    shared = {name: item[name] for name in ('photometric', 'blur_sigma') if name in item}
                if step+1 < len(rows):
                    segment = op.track_pos[int(op.seq_end[j])-1:int(op.seq_end[rows[step+1]])]
                else:
                    segment = np.asarray(op.pos[j])[None]
                item.update(shared, replay_event=key, replay_episode=stable_key(key, episode), episode_index=index,
                            episode_step=step, episode_supervised=step >= len(rows)-spec.supervised,
                            episode_segment=np.asarray(segment, np.float64))
                items.append(item)
            return mark_episode_identity(items)
        return None

    def episode_plan(self, rng, windows, index):
        """One training episode by the task budget: a simulated trace ('fresh') or an on-policy replay episode
        ('dagger_<class>', falling back to a fresh episode when none is eligible)."""
        name = TASKS[int(rng.choice(len(TASKS), p=self.budget.shares))]
        if name != 'fresh' and not name.startswith('dagger_'):
            raise ValueError(f'Episode training draws fresh or DAgger episodes only, not {name!r}')
        items = self.replay_episode_items(name[len('dagger_'):], rng, index) if name != 'fresh' else None
        delivered, fallback = (name, 'none') if items is not None else ('fresh', 'none' if name == 'fresh' else 'fresh')
        if items is None:
            items = self.episode_items(rng, windows, index)
        for item in items:
            item.update(task_requested=TASK[name], task_delivered=TASK[delivered], task_fallback=FALLBACKS.index(fallback))
        return items

    def live_start(self, rng, windows):
        """Chain starts: half recorded valid pre-excursion/recoverable prefixes, half seed-only."""
        from vesuvius.neural_tracing.tracers.fibers.data.state_labels import FOLLOWING, RECOVERABLE
        if rng.random() < .5:
            for name in ('pre_excursion', 'recoverable') if rng.random() < .5 else ('recoverable', 'pre_excursion'):
                item = self.replay_draw(name, rng)
                if item is not None and item['geometry_valid'] and item['supervision'] in (FOLLOWING, RECOVERABLE):
                    item.update(live_start='replay', live_loop_start=0)
                    return item
        item = self.fresh_item(rng, windows, startup=STARTUP_CATEGORIES.index('seed_only'))
        item.update(live_start='seed', live_loop_start=len(item['observed_path'])-1)
        return item

    def task_item(self, task, rng, windows):
        name, fallback, delivered = TASKS[task], 'none', TASKS[task]
        if name == 'fresh':
            item = self.fresh_item(rng, windows)
        elif name == 'live':
            item = (self.live_continuation.placeholder(self, rng, windows) if self.live_continuation is not None
                    else None)
            if item is None:
                item, fallback, delivered = self.fresh_item(rng, windows), 'no_live_state', 'fresh'
        elif name == 'synthetic_terminal':
            item = self.synthetic_item(rng)
            if item is None:
                item, fallback, delivered = self.fresh_item(rng, windows), 'fresh', 'fresh'
        else:
            kind = name[len('dagger_'):]
            item = self.replay_draw(kind, rng)
            if kind == 'terminal':
                self.terminal_requests += 1
                if item is None and self.synthetic_fallbacks < self.budget.terminal_fallback_cap*self.terminal_requests:
                    item = self.synthetic_item(rng)
                    if item is not None:
                        self.synthetic_fallbacks += 1
                        fallback, delivered = 'synthetic', 'synthetic_terminal'
            if item is None:
                fallback, delivered = 'fresh', 'fresh'
                item = (self.recovery_item(rng, windows) if kind == 'recoverable' else
                        self.fresh_item(rng, windows, startup=STARTUP_CATEGORIES.index('established')))
        item.update(task_requested=task, task_delivered=TASK[delivered], task_fallback=FALLBACKS.index(fallback))
        return item

    def prefetch_bounds(self, items, vol):
        if hasattr(self.batch_builder, 'prefetch_bounds'):
            return [bound for item in items for bound in self.batch_builder.prefetch_bounds(item,vol)]
        from vesuvius.neural_tracing.tracers.fibers.frames.ct import frame_prefetch_bounds
        return [bound for item in items for bound in frame_prefetch_bounds(item,self.cfg.crop,vol.input_scale)]

    def prefetch_items(self, items, vol, *, required=False):
        if self.remote_prefetch is None or not items:
            return
        # Whole lookahead windows are retained by the remote process instead of
        # sending disposable per-item hints into a potentially full queue.
        if not required and self.remote_prefetch_lookahead:
            return
        bounds = self.prefetch_bounds(items,vol)
        if required:
            self.remote_prefetch.ensure(vol.ct,bounds)
        else:
            self.remote_prefetch.submit(vol.ct,bounds)

    def resolve_seeds(self, items, vol, rng, windows):
        """CT seed headings; a seed without CT orientation is rejected and its task redrawn."""
        from vesuvius.neural_tracing.tracers.fibers.frames.ct import SeedHeadingError
        rejected = 0
        if self.episodes is not None:
            # A rejected seed redraws its whole episode: every decision of a trace shares the seed.
            groups = {}
            for item in items:
                groups.setdefault(item['episode_index'], []).append(item)
            resolved = []
            for index, group in groups.items():
                for _ in range(MAX_CT_FRAME_REJECTIONS):
                    try:
                        for item in group:
                            resolve_trace_seed(item, vol)
                        break
                    except SeedHeadingError:
                        rejected += 1
                        group = self.episode_items(rng, windows, index)
                        self.prefetch_items(group, vol, required=True)
                else:
                    raise SeedHeadingError('No CT-oriented replacement episode after repeated rejections')
                resolved.extend(group)
            items[:] = resolved
            return rejected
        for index, item in enumerate(items):
            for _ in range(MAX_CT_FRAME_REJECTIONS):
                try:
                    resolve_trace_seed(item, vol)
                    break
                except SeedHeadingError:
                    rejected += 1
                    replacement = self.task_item(item['task_requested'], rng, windows)
                    self.prefetch_items([replacement], vol, required=True)
                    item = items[index] = replacement
            else:
                raise SeedHeadingError('No CT-oriented replacement seed after repeated rejections')
        return rejected

    def __iter__(self):
        from vesuvius.neural_tracing.tracers.fibers.frames.ct import SeedHeadingError
        torch.set_num_threads(1)
        if self.remote_prefetch is not None:
            self.remote_prefetch.ensure_metadata(self.vol_spec)
            vol = CTVolume(self.vol_spec,cache_bytes=self.cache_bytes,cache_only=True)
        else:
            vol = CTVolume(self.vol_spec,cache_bytes=self.cache_bytes)
        grid = torch.from_numpy(crop_local_grid(self.cfg.crop)).float()
        lookahead = self.remote_prefetch_lookahead if self.remote_prefetch is not None else 0
        if lookahead < 0:
            raise ValueError('Remote prefetch lookahead must be nonnegative')
        windows = []
        plans, pending = self._iter_plans(vol, windows), deque()
        first = True
        worker = torch.utils.data.get_worker_info()
        live_rng = np.random.default_rng(np.random.SeedSequence([self.seed, 8123, 0 if worker is None else worker.id]))
        rejected = seed_rejections = 0
        while True:
            # Deliver the first batch promptly. Fill the deeper plan queue when
            # the loader asks for its next batch, overlapping the first update.
            while len(pending) < (1 if first else lookahead+1):
                plan = next(plans)
                bounds = self.prefetch_bounds(plan,vol) if lookahead else None
                pending.append((plan,bounds))
            first = False
            if lookahead:
                self.remote_prefetch.lookahead(vol.ct,
                    [bounds for _,bounds in pending],scope=id(self))
            items, bounds = pending.popleft()
            live_outcomes = None
            if self.live_continuation is not None:
                items = [self.live_continuation.resolve(item, self, vol, live_rng)
                         if item.get('live_requested') else item for item in items]
                live_outcomes = self.live_continuation.take_outcomes()
                # Resolved positions differ from the geometry lookahead plan.
                bounds = self.prefetch_bounds(items, vol) if lookahead else None
            if lookahead:
                self.remote_prefetch.ensure(vol.ct,bounds)
            else:
                self.prefetch_items(items,vol,required=True)
            seed_rejections += self.resolve_seeds(items, vol, live_rng, windows)
            try:
                if self.batch_builder is None:
                    raise ValueError('Training requires the common observation builder')
                batch = self.batch_builder(items, vol)
            except SeedHeadingError as exc:
                # Invalid crop or slab orientation context rejects the whole plan.
                # Weak CT orientation falls back in ct_frame; I/O errors propagate.
                rejected += 1
                if rejected >= MAX_CT_FRAME_REJECTIONS:
                    worker = torch.utils.data.get_worker_info()
                    raise SeedHeadingError(
                        f'Could not build a CT-oriented training batch after {rejected} consecutive rejected plans '
                        f'(source={getattr(self.vol_spec, "ct_zarr", "unknown")}, '
                        f'worker={worker.id if worker else 0}): {exc}') from exc
                continue
            if self.episodes is not None:
                batch.update(episode_tensors(items, self.episodes.commit))
            if self.live_continuation is not None:
                batch['_live_states'] = [self.live_continuation.metadata(item) for item in items]
                batch['live_depth'] = torch.tensor([item.get('live_depth', 0) for item in items])
                batch['live_limit'] = torch.tensor([item.get('live_limit', 0) for item in items])
                batch['live_travelled'] = torch.tensor([item.get('live_travelled', 0.) for item in items])
            # Row-aligned counters survive the usual tensor batch movers; totals sit in
            # row 0 so summing never multiplies them by the batch size. Zero counts are omitted.
            counters = dict(ct_frame_rejected_batches=rejected, ct_seed_rejections=seed_rejections,
                            replay_max_event_reuse=self.max_event_reuse() if self.onpolicy else 0)
            if live_outcomes is not None:
                counters.update({'live_'+name: value for name, value in live_outcomes.items()})
            for name, value in counters.items():
                if value:
                    batch[name] = torch.zeros(len(items), dtype=torch.int64)
                    batch[name][0] = value
            rejected = seed_rejections = 0
            yield batch

    def _iter_plans(self, vol, windows):
        """Ordered geometry/augmentation plans, without CT or dense image tensors."""
        info = torch.utils.data.get_worker_info()
        rng = np.random.default_rng(self.seed*1000 + (0 if info is None else info.id))
        chunks = 0
        while True:
            if chunks % self.refresh_chunks == 0:
                self.refresh_replay()
            chunks += 1
            items = []
            if self.episodes is not None:
                for index in range(self.chunk):
                    episode = self.episode_plan(rng, windows, index)
                    self.prefetch_items(episode, vol)
                    items.extend(episode)
                yield items
                continue
            for task in rng.choice(len(TASKS), size=self.chunk, p=self.budget.shares):
                item = self.task_item(int(task), rng, windows)
                self.prefetch_items([item], vol)
                items.append(item)
            if (self.remote_prefetch is not None and self.remote_prefetch_lookahead
                    and hasattr(self.batch_builder,'prepare_sampling_feedback')):
                self.batch_builder.prepare_sampling_feedback(items)
            yield items
