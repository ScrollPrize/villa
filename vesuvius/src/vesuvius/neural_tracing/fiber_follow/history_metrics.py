"""Measure observed trace drift on supplied, annotated history points."""
import torch

from vesuvius.neural_tracing.fiber_follow.model import history_tangent

TANGENT_POINTS = 6  # observed points, including the current one, in the diagnostic tangent fit


@torch.no_grad()
def observed_measurements(hist, hmask, target, target_mask, tangent_points=TANGENT_POINTS):
    """Per-state (value, valid) tensors; departed geometry is not scored.

    Error is measured only where both observed history and GT exist.
    These are aligned-arclength errors, not nearest-point distances that could
    silently credit a jump to a different winding.
    """
    n = target.shape[1] - 1
    observed = torch.cat([hist.new_zeros((len(hist), 1, 3)), hist[:, :n]], 1).float()
    supplied = torch.cat([hmask.new_ones((len(hist), 1)), hmask[:, :n]], 1).float()
    mask = supplied * target_mask
    observed_error = (observed - target).norm(dim=-1)
    counts = mask.sum(-1)
    mean = lambda value: (value * mask).sum(-1) / counts.clamp_min(1)
    result = dict(observed_current_error=(observed_error[:, 0], mask[:, 0] > 0),
                  observed_history_error=(mean(observed_error), counts > 0))
    truth, gt_valid = history_tangent(target, mask, tangent_points)
    before, old_valid = history_tangent(observed, mask, tangent_points)
    valid = gt_valid & old_valid
    angle = lambda direction: torch.rad2deg(torch.acos((direction * truth).sum(-1).clamp(-1, 1)))
    result.update(observed_tangent_error_deg=(angle(before), valid))
    return result


def summarize_history(measurements):
    result = {}
    for name, (value, valid) in measurements.items():
        count = valid.sum()
        result[name] = (value * valid).sum().item() / max(1, count.item())
        result[name + '_count'] = int(count.item())
    return result
