"""The whole-set fiber satisfaction pass, run beside the patch one at the end
of a fit stage.

Over every point the fiber CC loss holds (fiber_cc_loss.FiberCCLoss: the
kept, in-window points of every fiber the gauge did not drop; no sample), with
the loss's own residual (FiberCCLoss.evaluate_points: the same sheet label n,
phase acceptance |rho| <= RHO_MAX and vertical-fiber offset), s in turns:

- the target of a point is c0 in the largest CC and, in every other CC,
  round(median s) over all that CC's accepted points, computed in this pass
  (whole_winding_targets, as the DT targets are; a CC without an accepted
  point has none);
- a point is satisfied when it is accepted, has a target and
  |s - target| <= SATISFIED_TURNS;
- a fiber is satisfied when all its points are; an other CC when all its
  fibers are.

It also reports the largest CC's health statistics (fiber_cc_loss.metrics)
exactly, over all its points rather than a sample, and how many other CCs'
cached DT targets (when the loss has them) differ from this pass's targets.

Outputs (write_fiber_satisfaction): the summary, a dict of scalars the caller
adds to satisfaction_metrics_<suffix>.json's summary; and
``fiber_satisfied_<suffix>.npz`` (numpy, compressed), one row per fiber:

    fiber_uuid           uint8 [F, 16]  the solve's fiber UUID bytes
                                        (uuid.UUID(bytes=row.tobytes()))
    component            int64 [F]      the fit's component (CC) id: the run's,
                                        or with the constraint cut
                                        (fiber_constraint_cut) the recomputed
                                        component's index
    largest              bool [F]       in the largest CC
    point_count          int32 [F]      points the loss holds
    accepted_count       int32 [F]      of which accepted
    satisfied_count      int32 [F]      of which satisfied
    satisfied_fraction   float32 [F]    satisfied_count / point_count
    satisfied            bool [F]       every point satisfied
    target               float32 [F]    the fiber's CC target (NaN: none)
    mean_offset          float32 [F]    mean signed s - target (turns) over its
                                        accepted points (NaN: none, or no target);
                                        s - c0 in the largest CC

with scalars ``run_uuid``, ``c0`` and ``satisfied_turns``.
"""
import dataclasses
import os

import numpy as np
import torch

from fiber_cc_loss import whole_winding_targets

# A point is satisfied within this many turns of its target (the health
# metrics' "off by more than half a turn" is its complement).
SATISFIED_TURNS = 0.5


@dataclasses.dataclass
class FiberSatisfaction:
    summary: dict
    fibers: dict
    run_uuid: str
    c0: int


def _ratio(numerator, denominator):
    return float(numerator) / max(float(denominator), 1.0)


@torch.no_grad()
def evaluate_fiber_satisfaction(loss, transform, dr):
    """The pass over ``loss`` (a FiberCCLoss) under ``transform`` and ``dr``
    (FiberSatisfaction). Consumes no random numbers."""
    device = loss.device
    point_count = torch.as_tensor(loss.fiber_point_count, dtype=torch.int64, device=device)
    num_fibers = int(point_count.numel())
    fiber_cc = torch.as_tensor(loss.fiber_cc, dtype=torch.int64, device=device)
    point_fiber = torch.repeat_interleave(torch.arange(num_fibers, device=device), point_count)
    point_cc = fiber_cc[point_fiber]
    s, ok = loss.evaluate_points(transform, dr, torch.arange(loss.num_points, device=device))
    largest_local = loss.largest_local

    targets, _ = whole_winding_targets(point_cc, s, ok, loss.num_cc)
    other_cc = ~loss.cc_is_largest
    other_has_target = torch.isfinite(targets) & other_cc
    if largest_local >= 0:
        targets[largest_local] = float(loss.c0)
    target = targets[point_cc]
    has_target = torch.isfinite(target)
    offset = s - torch.where(has_target, target, torch.zeros_like(target))
    measured = ok & has_target
    satisfied = measured & (offset.abs() <= SATISFIED_TURNS)

    def per_fiber(mask, weights=None):
        # float32 (MPS has no float64): exact for counts below 2^24.
        values = mask.to(torch.float32) if weights is None else torch.where(
            mask, weights, torch.zeros_like(weights))
        return torch.bincount(point_fiber, weights=values, minlength=num_fibers)

    fiber_accepted = per_fiber(ok).round().long()
    fiber_satisfied_count = per_fiber(satisfied).round().long()
    fiber_measured = per_fiber(measured)
    fiber_offset = per_fiber(measured, offset) / fiber_measured.clamp(min=1)
    fiber_offset = torch.where(fiber_measured > 0, fiber_offset,
                               torch.full_like(fiber_offset, float('nan')))
    fiber_satisfied = fiber_satisfied_count == point_count
    cc_unsatisfied = torch.bincount(fiber_cc[~fiber_satisfied], minlength=loss.num_cc)
    fiber_largest = fiber_cc == largest_local

    summary = {}
    total_points = int(loss.num_points)
    rejected = total_points - int(ok.sum())
    summary.update({
        'fiber_points': total_points,
        'fiber_rejected_points': rejected,
        'fiber_rejected_fraction': _ratio(rejected, total_points),
        'fiber_dropped_fibers': int(loss.gauge['dropped'].sum()),
        'fiber_c0': int(loss.c0),
    })
    for name, fibers in (('', torch.ones_like(fiber_largest)),
                         ('_largest_cc', fiber_largest),
                         ('_other_ccs', ~fiber_largest)):
        count = int(fibers.sum())
        good = fibers & fiber_satisfied
        points = int(point_count[fibers].sum())
        good_points = int(fiber_satisfied_count[fibers].sum())
        summary.update({
            f'satisfied_fibers{name}': int(good.sum()),
            f'total_fibers{name}': count,
            f'satisfied_fibers{name}_fraction': _ratio(good.sum(), count),
            f'satisfied_fibers{name}_point_weighted_fraction': _ratio(
                point_count[good].sum(), points),
            f'satisfied_fiber_points{name}': good_points,
            f'total_fiber_points{name}': points,
            f'satisfied_fiber_points{name}_fraction': _ratio(good_points, points),
        })
    other_ccs = int(other_cc.sum())
    good_ccs = int(((cc_unsatisfied == 0) & other_cc).sum())
    summary.update({
        'satisfied_fiber_other_ccs': good_ccs,
        'total_fiber_other_ccs': other_ccs,
        'satisfied_fiber_other_ccs_fraction': _ratio(good_ccs, other_ccs),
        'fiber_other_ccs_with_target': int(other_has_target.sum()),
        'fiber_other_ccs_nonzero_target': int((other_has_target & (targets != 0)).sum()),
    })
    if largest_local >= 0:
        in_largest = point_cc == largest_local
        largest_points = int(in_largest.sum())
        accepted_largest = in_largest & ok
        signed = (s - float(loss.c0))[accepted_largest]
        summary['fiber_largest_cc_points'] = largest_points
        summary['fiber_largest_cc_rejected_fraction'] = _ratio(
            largest_points - int(accepted_largest.sum()), largest_points)
        # Keys without a point to measure are left out, as in the health metrics.
        if signed.numel():
            error = signed.abs()
            summary.update({
                'fiber_largest_cc_median_offset': float(signed.median()),
                'fiber_largest_cc_mean_abs_offset': float(error.mean()),
                'fiber_largest_cc_off_fraction': _ratio(
                    (error > SATISFIED_TURNS).sum(), error.numel()),
            })
    if loss.targets is not None:
        cached = loss.targets
        present = torch.isfinite(cached) & other_cc
        differ = present & ~(torch.isfinite(targets) & (cached == targets))
        summary.update({
            'fiber_cached_targets_iteration': int(loss.targets_iteration),
            'fiber_cached_targets': int(present.sum()),
            'fiber_cached_targets_differing': int(differ.sum()),
        })

    fiber_target = targets[fiber_cc]
    count_np = point_count.cpu().numpy()
    satisfied_np = fiber_satisfied_count.cpu().numpy()
    fibers = {
        'fiber_uuid': np.ascontiguousarray(loss.fiber_uuid, dtype=np.uint8),
        'component': np.asarray(loss.cc_component, dtype=np.int64)[loss.fiber_cc],
        'largest': fiber_largest.cpu().numpy(),
        'point_count': count_np.astype(np.int32),
        'accepted_count': fiber_accepted.cpu().numpy().astype(np.int32),
        'satisfied_count': satisfied_np.astype(np.int32),
        'satisfied_fraction': (satisfied_np / np.maximum(count_np, 1)).astype(np.float32),
        'satisfied': fiber_satisfied.cpu().numpy(),
        'target': fiber_target.cpu().numpy().astype(np.float32),
        'mean_offset': fiber_offset.cpu().numpy().astype(np.float32),
    }
    return FiberSatisfaction(summary=summary, fibers=fibers, run_uuid=loss.run_uuid,
                             c0=int(loss.c0))


def fiber_satisfaction_lines(result):
    """The printed lines, in the patch satisfaction's style."""
    summary = result.summary
    lines = []
    for name, label in (('', ''), ('_largest_cc', ' (largest component)'),
                        ('_other_ccs', ' (other components)')):
        lines.append(
            f'satisfied_fibers{name} = {summary[f"satisfied_fibers{name}"]}/'
            f'{summary[f"total_fibers{name}"]} '
            f'({100 * summary[f"satisfied_fibers{name}_fraction"]:.1f}%; point-weighted '
            f'{100 * summary[f"satisfied_fibers{name}_point_weighted_fraction"]:.1f}%; '
            f'satisfied points {summary[f"satisfied_fiber_points{name}"]}/'
            f'{summary[f"total_fiber_points{name}"]} '
            f'{100 * summary[f"satisfied_fiber_points{name}_fraction"]:.1f}%){label}')
    lines.append(
        f'satisfied_fiber_other_ccs = {summary["satisfied_fiber_other_ccs"]}/{summary["total_fiber_other_ccs"]} '
        f'({100 * summary["satisfied_fiber_other_ccs_fraction"]:.1f}%) (other components, every '
        f'fiber satisfied; {summary["fiber_other_ccs_with_target"]} with a whole-winding '
        f'target, {summary["fiber_other_ccs_nonzero_target"]} of them nonzero)')
    parts = [f'rejected {100 * summary["fiber_rejected_fraction"]:.2f}% of '
             f'{summary["fiber_points"]} points']
    if 'fiber_largest_cc_points' in summary:
        parts.append(f'largest component rejected '
                     f'{100 * summary["fiber_largest_cc_rejected_fraction"]:.2f}%')
    if 'fiber_largest_cc_mean_abs_offset' in summary:
        parts.append(
            f'mean |s - c0| = {summary["fiber_largest_cc_mean_abs_offset"]:.3f} turns '
            f'(median s - c0 = {summary["fiber_largest_cc_median_offset"]:+.2f}), '
            f'{100 * summary["fiber_largest_cc_off_fraction"]:.1f}% off by more than half a turn')
    lines.append('fiber satisfaction (all points): ' + '; '.join(parts))
    if 'fiber_cached_targets' in summary:
        lines.append(
            f'fiber satisfaction: cached DT targets (step '
            f'{summary["fiber_cached_targets_iteration"]}): '
            f'{summary["fiber_cached_targets_differing"]} of '
            f'{summary["fiber_cached_targets"]} differ from this pass\'s')
    return lines


def write_fiber_satisfaction(result, out_path, suffix):
    """Write ``fiber_satisfied_<suffix>.npz``; returns its path."""
    path = os.path.join(out_path, f'fiber_satisfied_{suffix}.npz')
    np.savez_compressed(path, run_uuid=np.array(result.run_uuid or ''),
                        c0=np.array(result.c0, dtype=np.int64),
                        satisfied_turns=np.array(SATISFIED_TURNS), **result.fibers)
    return path
