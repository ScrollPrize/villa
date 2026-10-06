"""Shared first-connection limits, the confidence-gated commit and the operating policy.

Commit gates: 'full' (default) commits ``n_commit`` points only when the proposal's full-horizon (last-plane)
survival confidence reaches the threshold, and nothing otherwise; 'prefix' (the earlier rule) commits the prefix
whose cumulative confidence stays above the threshold, up to ``n_commit`` points.
"""
from dataclasses import asdict, dataclass
import math

import torch


DEFAULT_CONFIDENCE = 0.4
DEFAULT_N_COMMIT = 8
DEFAULT_GATE = 'full'
GATES = ('full', 'prefix')
# Operating policies recorded before gates existed meant this (checkpoint_policy reads them unchanged).
LEGACY_CONFIDENCE, LEGACY_GATE = 0.5, 'prefix'
DEFAULT_MAX_RECOVERY_DISTANCE = 6.0
# Gate thresholds at which training logs, monitor rollouts, recovery studies
# and curve plots report. Calibration sweeps its own grid; nothing here is
# an operating point.
DIAGNOSTIC_THRESHOLDS = (.5,)


@dataclass(frozen=True)
class OperatingPolicy:
    """Resolved commit policy shared by live feedback, collection, evaluation and deployment.

    Same-position refinement runs up to ``refinement_steps`` extra attempts; the decision
    stops as soon as they are exhausted without an accepted prefix. These are run
    parameters of the single implementation, not selectors of different semantics.
    """
    confidence: float = DEFAULT_CONFIDENCE
    n_commit: int = DEFAULT_N_COMMIT
    max_recovery_distance: float = DEFAULT_MAX_RECOVERY_DISTANCE
    refinement_steps: int = 0
    gate: str = DEFAULT_GATE

    def __post_init__(self):
        if self.gate not in GATES:
            raise ValueError(f'Commit gate must be one of {GATES}')
        if not 0 <= self.confidence <= 1 or int(self.n_commit) != self.n_commit or self.n_commit < 1:
            raise ValueError('Operating policy needs a confidence in [0, 1] and a positive commit count')
        if not math.isfinite(self.max_recovery_distance) or self.max_recovery_distance <= 0:
            raise ValueError('Operating policy needs a finite positive recovery limit')
        if int(self.refinement_steps) != self.refinement_steps or self.refinement_steps < 0:
            raise ValueError('Refinement steps must be a nonnegative integer')

    def to_dict(self):
        return asdict(self)

    def validate_model(self, cfg):
        """Recovery limit and refinement belong to the model; the policy must agree."""
        if self.n_commit > cfg.n_future:
            raise ValueError(f'n_commit={self.n_commit} exceeds the model horizon n_future={cfg.n_future}')
        if self.max_recovery_distance != cfg.max_recovery_distance:
            raise ValueError('Operating policy recovery limit differs from the model')
        if self.refinement_steps != getattr(cfg, 'recurrent_refinement_steps', 0):
            raise ValueError('Operating policy refinement differs from the model')


def checkpoint_policy(checkpoint, cfg, *, confidence=None, n_commit=None, gate=None):
    """The checkpoint's recorded policy, optionally with an explicit threshold/commit/gate override.

    A policy recorded before commit gates existed used the prefix gate, and is read that way."""
    recorded = dict(checkpoint.get('operating_policy') or dict(
        confidence=LEGACY_CONFIDENCE, n_commit=checkpoint.get('n_commit', DEFAULT_N_COMMIT),
        max_recovery_distance=cfg.max_recovery_distance,
        refinement_steps=getattr(cfg, 'recurrent_refinement_steps', 0)))
    recorded.setdefault('gate', LEGACY_GATE)
    if confidence is not None:
        recorded['confidence'] = float(confidence)
    if n_commit is not None:
        recorded['n_commit'] = int(n_commit)
    if gate is not None:
        recorded['gate'] = gate
    policy = OperatingPolicy(**recorded)
    policy.validate_model(cfg)
    return policy


def recovery_allowed(points, max_distance=DEFAULT_MAX_RECOVERY_DISTANCE):
    """Bound the actual origin-to-first-point segment in trace-grid voxels.

    This permits a displaced start to recover, but never permits an unbounded
    jump. It is a geometric policy, independent of annotation availability.
    """
    first = points[..., 0, :].float()
    return (torch.isfinite(first).all(-1) & (first[..., 2] > 0)
            & (first.norm(dim=-1) <= max_distance))


def commit_prefix(points, confidence, threshold=DEFAULT_CONFIDENCE, n_commit=DEFAULT_N_COMMIT,
                  max_distance=DEFAULT_MAX_RECOVERY_DISTANCE):
    """Eligible prefix of a single curve; at most ``n_commit`` points per decision.

    ``n_commit`` may not exceed the generated horizon (``confidence.shape[-1]``).
    """
    horizon = confidence.shape[-1]
    if not 1 <= n_commit <= horizon:
        raise ValueError(f'n_commit must lie in [1, {horizon}]')
    allowed = recovery_allowed(points,max_distance)
    conf = confidence.float().cummin(-1).values
    count = (conf >= threshold).int().cumprod(-1).sum(-1).clamp(max=n_commit)
    return torch.where(allowed,count,0),allowed


def gate_horizon(cfg):
    """Planes whose confidence decides acceptance (``gate_plane``; the last plane for models without one)."""
    return getattr(cfg, 'gate_horizon', cfg.n_future)


def commit_count(points, confidence, threshold=DEFAULT_CONFIDENCE, n_commit=DEFAULT_N_COMMIT,
                 max_distance=DEFAULT_MAX_RECOVERY_DISTANCE, gate=DEFAULT_GATE, horizon=None):
    """Points to commit per proposal under ``gate`` (module docstring) and whether its first connection is allowed.

    ``horizon`` (a model's gate plane) limits the planes whose confidence decides; later planes are only predicted."""
    if horizon is not None:
        confidence = confidence[..., :horizon]
    if gate == 'prefix':
        return commit_prefix(points, confidence, threshold, n_commit, max_distance)
    if gate != 'full':
        raise ValueError(f'Commit gate must be one of {GATES}')
    if not 1 <= n_commit <= confidence.shape[-1]:
        raise ValueError(f'n_commit must lie in [1, {confidence.shape[-1]}]')
    allowed = recovery_allowed(points, max_distance)
    full = confidence.float().cummin(-1).values[..., -1] >= threshold
    return torch.where(allowed & full, n_commit, 0), allowed


def selection_window(n_commit, horizon, gate=DEFAULT_GATE):
    """Prefix length by which a model ranks its proposals: the commit for 'prefix', the full horizon for 'full'
    (where acceptance depends on the last plane)."""
    return horizon if gate == 'full' else n_commit


def select_candidate(points, confidence, n_commit, max_distance=DEFAULT_MAX_RECOVERY_DISTANCE,
                     *, stop_threshold=None):
    """Rank the commit prefix; optionally rescue a stop with an acceptable alternative.

    Preserve the original winner whenever it can advance. Otherwise choose the
    longest acceptable prefix, breaking ties by confidence at that prefix and
    then candidate order. Confidence and recovery limits remain unchanged.
    """
    conf = confidence.float().cummin(-1).values
    allowed = recovery_allowed(points, max_distance)
    horizon = min(n_commit, conf.shape[-1])
    selected = conf[..., horizon-1].masked_fill(~allowed, -torch.inf).argmax(-1)
    if stop_threshold is None:
        return selected
    counts, _ = commit_prefix(points, conf, stop_threshold, horizon, max_distance)
    longest = counts.max(-1, keepdim=True).values
    last = (counts-1).clamp_min(0)
    prefix_score = conf.gather(-1, last[..., None]).squeeze(-1)
    fallback = prefix_score.masked_fill((counts != longest) | (counts == 0), -torch.inf).argmax(-1)
    stopped = counts.gather(-1, selected[..., None]).squeeze(-1) == 0
    return torch.where(stopped & (longest.squeeze(-1) > 0), fallback, selected)
