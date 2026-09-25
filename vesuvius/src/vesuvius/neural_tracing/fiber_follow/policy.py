"""Shared first-connection limits and confidence-gated prefix commit."""
import torch


DEFAULT_CONFIDENCE = 0.7
DEFAULT_MAX_RECOVERY_DISTANCE = 6.0


def recovery_allowed(points, max_distance=DEFAULT_MAX_RECOVERY_DISTANCE):
    """Bound the actual origin-to-first-point segment in trace-grid voxels.

    This permits a displaced start to recover, but never permits an unbounded
    jump. It is a geometric policy, independent of annotation availability.
    """
    first = points[..., 0, :].float()
    return (torch.isfinite(first).all(-1) & (first[..., 2] > 0)
            & (first.norm(dim=-1) <= max_distance))


def commit_prefix(points, confidence, threshold=DEFAULT_CONFIDENCE, n_commit=4,
                  max_distance=DEFAULT_MAX_RECOVERY_DISTANCE):
    """Eligible prefix of a single curve; at most four points per decision."""
    if not 1 <= n_commit <= 4:
        raise ValueError('n_commit must lie in [1, 4]')
    allowed = recovery_allowed(points,max_distance)
    conf = confidence.float().cummin(-1).values
    count = (conf >= threshold).int().cumprod(-1).sum(-1).clamp(max=n_commit)
    return torch.where(allowed,count,0),allowed
