"""Training states for the beam re-ranker, produced by running the real beam.

Each sample runs the C++ tracer over one span of a training fiber (optionally
from a perturbed start) with an observing hook, records the candidate pools at
every hook round, and turns a subset of them into states with GT labels. The
beam never sees GT, so it wanders exactly as it does in production and the
recorded pools carry on-distribution mistakes.

Every line vertex between a fiber's first and last control point is user
verified, so span endpoints are arbitrary vertices of that trimmed line (the
loader already discards the exterior tails), not only the annotated control
points. The annotated control-point spans are still used for mining the spans
the hand beam fails on.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from scipy.spatial import cKDTree

from vesuvius.neural_tracing.fiber_follow.beam.native import BeamSpec, NativeBeam
from vesuvius.neural_tracing.fiber_follow.beam.states import BeamStateConfig, pool_state
from vesuvius.neural_tracing.fiber_follow.data import (
    FiberVolume, TracedFiber, ZBand, collate_with_volume, training_state_allowed,
)
from vesuvius.neural_tracing.fiber_follow.geometry import (
    crop_local_grid, frame_from_heading, normalize, tangent_at,
)

MIN_SPAN_GRID = 12.0  # VC3D traces spans of at least kMinimumTraceSteps steps (1 grid voxel each)
MAX_SPAN_GRID = 512.0

STACK_KEYS = ('candidates', 'point_mask', 'cand_mask', 'hand_loss', 'hand_rel', 'prefix_target', 'prefix_mask',
              'onfiber', 'label_mask', 'quality', 'offtrack', 'fwd_error', 'source')


def collate_beam(items, vol, crop, grid):
    """Crop inputs via the shared samplers, plus the stacked pool tensors."""
    out = collate_with_volume(items, vol, crop, grid)
    if 'tube_segments' in items[0]:
        from vesuvius.neural_tracing.fiber_follow.tube import render_tube
        tubes = [render_tube(it, crop, it['tube_sigma']) for it in items]
        out['tube_target'] = torch.from_numpy(np.stack([t[0] for t in tubes]))
        out['tube_mask'] = torch.from_numpy(np.stack([t[1] for t in tubes]))
    for key in STACK_KEYS:
        if key in items[0]:
            out[key] = torch.from_numpy(np.stack([np.asarray(it[key], np.float32) for it in items]))
    return out


def _mix(rng, sigmas, probs):
    return float(rng.choice(sigmas, p=probs))


def perturbed_start(fiber: TracedFiber, index: int, sign: float, cfg: BeamStateConfig, rng):
    """Start point/direction offset laterally and in heading, like ``data.make_sample``."""
    tau = tangent_at(fiber.points, fiber.s, float(fiber.s[index])) * sign
    fr = frame_from_heading(tau)
    lat = rng.normal(size=2) * _mix(rng, cfg.lateral_sigmas, cfg.lateral_probs)
    pos = fiber.points[index] + fr[:, 0] * lat[0] + fr[:, 1] * lat[1]
    ang = math.radians(_mix(rng, cfg.angle_sigmas_deg, cfg.angle_probs)) * rng.normal()
    axis_ang = rng.uniform(0, 2 * np.pi)
    ax = math.cos(axis_ang) * fr[:, 0] + math.sin(axis_ang) * fr[:, 1]
    return pos, normalize(math.cos(ang) * tau + math.sin(ang) * ax)


def sample_span(fiber: TracedFiber, rng, min_span=MIN_SPAN_GRID, max_span=MAX_SPAN_GRID):
    """Arclengths (start, end) of a random span anywhere on the trimmed line.

    Lengths are log-uniform between ``min_span`` and ``max_span`` (capped by the
    fiber), so short annotation-window spans and long ones are both frequent.
    """
    if fiber.length < min_span:
        return None
    length = float(np.exp(rng.uniform(np.log(min_span), np.log(min(max_span, fiber.length)))))
    start = float(rng.uniform(0., fiber.length - length))
    return start, start + length


def line_index(fiber: TracedFiber, arc: float) -> int:
    return int(np.abs(fiber.s - arc).argmin())


def record_pools(beam: NativeBeam, fiber: TracedFiber, start_index: int, target_index: int, *,
                 start_point=None, initial_direction=None):
    pools = []

    def observe(pool):
        pools.append(pool)
        return None
    result = beam.trace_span(fiber.points, start_index, target_index, start_point=start_point,
                             initial_direction=initial_direction, hook=observe)
    return pools, result


class BeamDataset(torch.utils.data.IterableDataset):
    """Yields collated chunks of labeled pool states from training fibers."""

    def __init__(self, fibers, vol_spec, beam_spec: BeamSpec, cfg: BeamStateConfig, exclude: ZBand | None, *,
                 chunk=8, seed=0, cache_bytes=1 << 30, p_perturb=0.5, hard_prob=0.5, max_states_per_trace=12,
                 hard_spans: dict | None = None, hard_span_prob=0.5, max_span=MAX_SPAN_GRID):
        self.fibers, self.vol_spec, self.beam_spec, self.cfg, self.exclude = fibers, vol_spec, beam_spec, cfg, exclude
        self.chunk, self.seed, self.cache_bytes = chunk, seed, cache_bytes
        self.p_perturb, self.hard_prob, self.max_span = p_perturb, hard_prob, max_span
        self.max_states = max_states_per_trace
        self.weights = np.array([f.length for f in fibers], float)
        self.weights /= self.weights.sum()
        # Spans the hand beam is known to fail on (see beam/mine.py), by fiber index.
        self.hard_spans = [(i, idx) for i, f in enumerate(fibers) for idx in (hard_spans or {}).get(f.name, ())
                           if idx < len(f.spans)]
        self.hard_span_prob = hard_span_prob if self.hard_spans else 0.0

    def pick_span(self, rng):
        """(fiber, (start_arc, end_arc) or None, from_hard_list)."""
        if rng.random() < self.hard_span_prob:
            i, idx = self.hard_spans[rng.integers(len(self.hard_spans))]
            span = self.fibers[i].spans[idx]
            return self.fibers[i], (span.start, span.end), True
        fiber = self.fibers[rng.choice(len(self.fibers), p=self.weights)]
        return fiber, sample_span(fiber, rng, max_span=self.max_span), False

    def states_for_trace(self, beam, rng, trees):
        """Run one span and return the selected labeled states (may be empty)."""
        fiber, span, mined = self.pick_span(rng)
        if span is None:
            return []
        sign = 1.0 if rng.random() < 0.5 else -1.0
        start_arc, target_arc = span if sign > 0 else span[::-1]
        start_index, target_index = line_index(fiber, start_arc), line_index(fiber, target_arc)
        if abs(target_index - start_index) < 2:
            return []
        start_point = initial_direction = None
        if rng.random() < self.p_perturb:
            start_point, initial_direction = perturbed_start(fiber, start_index, sign, self.cfg, rng)
        try:
            pools, _ = record_pools(beam, fiber, start_index, target_index,
                                    start_point=start_point, initial_direction=initial_direction)
        except ValueError:
            return []
        if not pools:
            return []
        if fiber.name not in trees:
            trees[fiber.name] = cKDTree(fiber.points)
        items = [pool_state(p, self.cfg, fiber, sign=sign, tree=trees[fiber.name], rng=rng) for p in pools]
        items = [it for it in items if training_state_allowed(it, self.cfg.crop, self.exclude)]
        hard = [it for it in items if it['onfiber'][0] == 0]
        easy = [it for it in items if it['onfiber'][0] != 0]
        rng.shuffle(hard)
        rng.shuffle(easy)
        n_hard = min(len(hard), int(round(self.max_states * self.hard_prob)) if easy else self.max_states)
        chosen = hard[:n_hard] + easy[:self.max_states - n_hard]
        for it in chosen:
            # source 2 marks hard states (hand tracer's choice is off-fiber), as in the follower's replay;
            # source 1 marks states from mined hard spans whose hand choice is still on-fiber.
            it['source'] = np.float32(2 if it['onfiber'][0] == 0 else (1 if mined else 0))
            it['source_step'] = np.float32(0)
        return chosen

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        worker = info.id if info else 0
        rng = np.random.default_rng(self.seed * 1000 + worker)
        torch.set_num_threads(1)
        vol = FiberVolume(self.vol_spec, cache_bytes=self.cache_bytes)
        beam = NativeBeam(self.beam_spec, self.vol_spec.grid_scale)
        grid = torch.from_numpy(crop_local_grid(self.cfg.crop)).float()
        trees = {}
        pending = []
        while True:
            attempts = 0
            while len(pending) < self.chunk:
                pending.extend(self.states_for_trace(beam, rng, trees))
                attempts += 1
                if attempts > 200 and not pending:
                    raise ValueError('Could not produce beam training states; check the beam spec and fibers')
            items, pending = pending[:self.chunk], pending[self.chunk:]
            yield collate_beam(items, vol, self.cfg.crop, grid)
