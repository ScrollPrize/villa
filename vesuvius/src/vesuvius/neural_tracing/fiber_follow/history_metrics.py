"""Compare observed and cleaned geometry on exactly the same valid points."""
import torch

from vesuvius.neural_tracing.fiber_follow.model import history_tangent

TANGENT_POINTS = 6  # corrected points, including the current one, in the diagnostic tangent fit


@torch.no_grad()
def cleaning_measurements(cleaned, hist, hmask, target, target_mask, tangent_points=TANGENT_POINTS):
    """Per-state (value, valid) tensors; departed states have correction sizes only.

    Improvement is measured only where both observed history and GT exist.
    These are aligned-arclength errors, not nearest-point distances that could
    silently credit a jump to a different winding.
    """
    n = cleaned.shape[1] - 1
    observed = torch.cat([hist.new_zeros((len(hist), 1, 3)), hist[:, :n]], 1).float()
    supplied = torch.cat([hmask.new_ones((len(hist), 1)), hmask[:, :n]], 1).float()
    mask = supplied * target_mask
    clean_error = (cleaned.float() - target).norm(dim=-1)
    observed_error = (observed - target).norm(dim=-1)
    counts = mask.sum(-1)
    mean = lambda value: (value * mask).sum(-1) / counts.clamp_min(1)
    old, new = mean(observed_error), mean(clean_error)
    result = dict(observed_current_error=(observed_error[:, 0], mask[:, 0] > 0),
                  clean_current_error=(clean_error[:, 0], mask[:, 0] > 0),
                  observed_history_error=(old, counts > 0), clean_history_error=(new, counts > 0),
                  clean_position_improved=((new < old).float(), counts > 0),
                  clean_correction_size=(((cleaned - observed).norm(dim=-1) * supplied).sum(-1)
                                         / supplied.sum(-1).clamp_min(1), supplied.sum(-1) > 0))
    truth, gt_valid = history_tangent(target, mask, tangent_points)
    before, old_valid = history_tangent(observed, mask, tangent_points)
    after, new_valid = history_tangent(cleaned, mask, tangent_points)
    valid = gt_valid & old_valid & new_valid
    angle = lambda direction: torch.rad2deg(torch.acos((direction * truth).sum(-1).clamp(-1, 1)))
    before_angle, after_angle = angle(before), angle(after)
    result.update(observed_tangent_error_deg=(before_angle, valid),
                  clean_tangent_error_deg=(after_angle, valid),
                  clean_tangent_improved=((after_angle < before_angle).float(), valid))
    return result


def summarize_cleaning(measurements):
    result = {}
    for name, (value, valid) in measurements.items():
        count = valid.sum()
        result[name] = (value * valid).sum().item() / max(1, count.item())
        result[name + '_count'] = int(count.item())
    return result
