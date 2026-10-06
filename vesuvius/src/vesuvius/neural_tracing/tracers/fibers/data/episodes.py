"""Whole-trace episodes: a seeded simulated trace split into its per-step decisions."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from vesuvius.neural_tracing.tracers.shared.geometry import interp_at, traversal_curve
from vesuvius.neural_tracing.tracers.fibers.data.samples import STARTUP_CATEGORIES, SampleConfig, decision_on_path, excursion_offsets, trace_noise, trace_prefix_length


@dataclass(frozen=True)
class EpisodeSpec:
    """Consecutive decisions along one simulated trace, for whole-trace history (``memory='sequence'``)."""
    steps: int = 64  # decisions per episode (fewer on a fiber too short for them, never below min_steps)
    supervised: int = 16  # trailing decisions with a loss; earlier decisions only build history tokens
    commit: int = 12  # voxels between consecutive heads (the tracer's commit)
    min_steps: int = 4

    def __post_init__(self):
        if not (1 <= self.supervised <= self.steps and 2 <= self.min_steps <= self.steps and self.commit >= 1):
            raise ValueError('Episodes need 1 <= supervised <= steps, 2 <= min_steps <= steps and a positive commit')


def episode_decisions(fiber, t0, reverse, steps, commit, cfg: SampleConfig, rng: np.random.Generator):
    """``steps`` decisions on one simulated trace of this fiber, heads at traversal arclengths t0, t0+commit, ...

    The trace is drawn as ``simulated_trace`` draws a decision's observed path (startup prefix before t0, trace noise
    over the whole trace, optionally one excursion ending at the trace's end), then every head ending a prefix of it
    becomes a decision (``decision_on_path``). Returns (decision, committed segment) pairs; a step's segment is the
    observed path from its head to the next head (world xyz), the points that step commits. Requires
    t0 + steps*commit <= the fiber length.
    """
    p, s = traversal_curve(fiber, reverse)
    step = cfg.history_step
    category = int(rng.choice(len(STARTUP_CATEGORIES), p=cfg.startup_shares))
    prefix = int(trace_prefix_length(t0, category, rng)//step)*step
    per_commit = int(round(commit/step))
    count = int(round(prefix/step))+steps*per_commit
    arcs = t0-prefix+step*np.arange(count+1, dtype=np.float64)
    if arcs[-1] > s[-1]+1e-6:
        raise ValueError('Episode runs past the end of its fiber')
    noise, sigma = trace_noise(arcs, p, s, cfg, rng)
    path = interp_at(p, s, arcs)+noise
    details, offsets = {}, np.zeros_like(path)
    if (category == STARTUP_CATEGORIES.index('established') and arcs[-1]-arcs[0] >= 8.
            and rng.random() < cfg.excursion_probability):
        offsets, details = excursion_offsets(arcs, p, s, cfg, rng)
        path = path+offsets
    out = []
    for j in range(steps):
        i = int(round(prefix/step))+j*per_commit
        departed = details if np.abs(offsets[:i+1]).max() > 0 else {}
        decision = decision_on_path(fiber, float(arcs[i]), reverse, cfg, path[:i+1], arcs[:i+1], category, sigma, departed)
        out.append((decision, path[i:i+per_commit+1].copy()))
    return out


def loader_chunk(cfg, batch):
    """Units per loader item: ``batch`` decisions, or for sequence models one episode (the trainer merges ``batch``
    of them per microbatch, train/sequence.merge_episodes), so loader workers never hold whole episode microbatches
    in shared memory."""
    return 1 if cfg.model_type == 'sequence' else batch


def mark_episode_identity(items):
    """Episode steps in step order: a step whose episode has an earlier step on the original fiber reads that fiber
    through its history tokens, so its identity is observable (observations.identity_evidence)."""
    from vesuvius.neural_tracing.tracers.fibers.data.state_labels import DEPARTURE_DISTANCE
    seen = False
    for item in items:
        item['history_identity_evidence'] = seen
        seen = seen or float(item['match_distance']) <= DEPARTURE_DISTANCE
    return items


def episode_tensors(items, commit):
    """Row-aligned episode layout of a batch of episode decisions: episode and step indices, which rows are
    supervised, and each step's committed segment (world xyz, padded to the longest, at least commit+1 points) with
    its mask."""
    length = max([int(round(commit))+1]+[len(item['episode_segment']) for item in items])
    segment = np.zeros((len(items), length, 3), np.float32)
    mask = np.zeros((len(items), length), bool)
    for row, item in enumerate(items):
        points = np.asarray(item['episode_segment'])
        segment[row, :len(points)], mask[row, :len(points)] = points, True
    return dict(episode_index=torch.tensor([it['episode_index'] for it in items], dtype=torch.long),
                episode_step=torch.tensor([it['episode_step'] for it in items], dtype=torch.long),
                episode_supervised=torch.tensor([bool(it['episode_supervised']) for it in items]),
                episode_segment=torch.from_numpy(segment), episode_segment_mask=torch.from_numpy(mask))
