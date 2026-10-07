"""Prefix correctness labels for a generated curve, shared by every tracer model."""
import torch
import torch.nn.functional as F
from vesuvius.neural_tracing.fiber_follow.tracing.policy import DEFAULT_MAX_RECOVERY_DISTANCE, recovery_allowed


@torch.no_grad()
def prefix_labels(points, batch, tolerance=1.5,
                     max_recovery_distance=DEFAULT_MAX_RECOVERY_DISTANCE, extra_failure=None, connector_failure=None):
    """Prefix agreement sampled every 0.25 forward voxel with the default config.

    A prefix is positive only when every sampled crossing is annotated and
    within tolerance. A known failure is negative even if later GT is missing.
    Unknown ends are censored. Tagged physical endpoints and terminal states
    supply explicit negatives. ``extra_failure`` (B, Q) marks further known
    failures, such as points on another fiber within tolerance, and
    ``connector_failure`` (B,) marks foreign contact on the origin-to-first-point
    connection. The connection must obey the same recovery distance limit as
    tracing. Within that limit, the first point may recover from a displaced
    origin; GT agreement begins at the first predicted plane. States without
    available confidence supervision are fully censored.
    """
    points = points.detach()
    B, K, _ = points.shape
    Q = batch['dense_ab'].shape[1]
    dense = F.interpolate(points[..., :2].transpose(1, 2),
                          size=Q, mode='linear', align_corners=True).transpose(1, 2).reshape(B, Q, 2)
    error = (dense-batch['dense_ab']).norm(dim=-1)
    known = batch['dense_mask'].bool()
    c = torch.linspace(0, 1, Q, device=points.device)
    forward = points[:, 0, 2, None]*(1-c)+points[:, -1, 2, None]*c
    beyond_end = batch['endpoint_known'][:, None].bool() & (forward > batch['end_local'][:, 2, None]+1e-4)
    beyond_end = beyond_end & ~known & (batch['end_local'][:, 2, None] >= 0)
    bad_recovery = ~recovery_allowed(points, max_recovery_distance)
    if connector_failure is not None:
        bad_recovery = bad_recovery | connector_failure.bool()
    failed = ((known & ((error > tolerance) | ~torch.isfinite(error))) | beyond_end
              | batch['terminal'][:, None].bool() | bad_recovery[..., None])
    if extra_failure is not None:
        failed = failed | extra_failure.bool()
    failure_prefix = failed.cumsum(-1) > 0
    known_prefix = (known | beyond_end).int().cummin(-1).values.bool()
    indices = torch.linspace(0, Q-1, K, device=points.device).round().long()
    target = (~failure_prefix[..., indices]).float()
    mask = (failure_prefix | known_prefix)[..., indices].float()*batch['confidence_valid'][:, None].float()
    valid_error = torch.where(known, error.nan_to_num(nan=8).clamp(max=8), 0.).sum(-1)/known.sum(-1).clamp(min=1)
    return target, mask, valid_error
