"""Shared first-connection limits, confidence-gated prefix commit and the operating policy."""
from dataclasses import asdict, dataclass
import math

import torch


DEFAULT_CONFIDENCE = 0.5
DEFAULT_N_COMMIT = 8
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

    def __post_init__(self):
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


def checkpoint_policy(checkpoint, cfg, *, confidence=None, n_commit=None):
    """The checkpoint's recorded policy, optionally with an explicit threshold/commit override."""
    recorded = dict(checkpoint.get('operating_policy') or dict(
        confidence=DEFAULT_CONFIDENCE, n_commit=checkpoint.get('n_commit', DEFAULT_N_COMMIT),
        max_recovery_distance=cfg.max_recovery_distance,
        refinement_steps=getattr(cfg, 'recurrent_refinement_steps', 0)))
    if confidence is not None:
        recorded['confidence'] = float(confidence)
    if n_commit is not None:
        recorded['n_commit'] = int(n_commit)
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
