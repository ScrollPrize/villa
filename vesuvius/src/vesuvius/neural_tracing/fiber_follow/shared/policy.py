"""Shared first-connection limits and confidence-gated prefix commit."""
import torch


DEFAULT_CONFIDENCE = 0.7
DEFAULT_N_COMMIT = 8
DEFAULT_MAX_RECOVERY_DISTANCE = 6.0


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
