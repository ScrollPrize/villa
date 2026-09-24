"""Shared first-connection limits and confidence-gated candidate choice."""
import torch


DEFAULT_CONFIDENCE = 0.7
DEFAULT_MAX_RECOVERY_DISTANCE = 6.0


def recovery_allowed(candidates, max_distance=DEFAULT_MAX_RECOVERY_DISTANCE):
    """Bound the actual origin-to-first-point segment in trace-grid voxels.

    This permits a displaced start to recover, but never permits an unbounded
    jump. It is a geometric policy, independent of annotation availability.
    """
    first = candidates[..., 0, :].float()
    return (torch.isfinite(first).all(-1) & (first[..., 2] > 0)
            & (first.norm(dim=-1) <= max_distance))


def choose_candidate(candidates, ranks, confidence, threshold=DEFAULT_CONFIDENCE,
                     n_commit=4, max_distance=DEFAULT_MAX_RECOVERY_DISTANCE):
    """Return chosen index, commit count, and recovery eligibility per candidate.

    Choose the highest rank among eligible open gates. If all gates are closed,
    retain the best eligible index for diagnostics / bounded exploration. If no
    recovery is eligible, the index is diagnostic only and commit is zero.
    """
    allowed = recovery_allowed(candidates, max_distance)
    conf = confidence.float().cummin(-1).values
    viable = allowed & (conf[..., 0] >= threshold)
    pool = torch.where(viable.any(-1, keepdim=True), viable, allowed)
    chosen = ranks.float().masked_fill(~pool, -torch.inf).argmax(-1)
    b = torch.arange(len(chosen), device=chosen.device)
    commit = (conf[b, chosen] >= threshold).sum(-1).clamp(max=n_commit)
    commit = torch.where(allowed[b, chosen], commit, 0)
    return chosen, commit, allowed
