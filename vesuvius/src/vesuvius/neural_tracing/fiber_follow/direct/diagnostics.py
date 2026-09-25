"""Additive decision diagnostics on the original fiber, including drift strata."""
import math

import torch
import torch.nn.functional as F

from vesuvius.neural_tracing.fiber_follow.policy import commit_prefix, recovery_allowed
from vesuvius.neural_tracing.fiber_follow.supervision import prefix_labels
from vesuvius.neural_tracing.fiber_follow.direct.supervision import (
    commit_window, dense_commit_mask, geometry_mask,
)


@torch.no_grad()
def decision_rows(output, batch, cfg, n_commit=None, tolerance=1.5, thresholds=(.5, .85)):
    """Small CPU records; no forward pass or random draws. Unknowns stay unknown."""
    output = {k: v.detach().float().cpu() for k, v in output.items()}
    batch = {k: v.detach().cpu() for k, v in batch.items() if isinstance(v, torch.Tensor)}
    window = commit_window(cfg, n_commit)
    points = output['points']
    labels, known, _ = prefix_labels(points, batch, tolerance, cfg.max_recovery_distance)
    annotated = batch['dense_mask'].bool() & ~batch['offtrack'][:, None].bool()
    observable = geometry_mask(batch, cfg)
    near = dense_commit_mask(observable.shape[1], cfg, window, points.device)
    mask = observable & near
    errors = {}
    for name, curve in (('final', points), ('initial', output.get('initial_points', points))):
        dense = F.interpolate(curve[..., :2].transpose(1, 2), size=mask.shape[1],
                              mode='linear', align_corners=True).transpose(1, 2)
        errors[name] = (dense-batch['dense_ab']).norm(dim=-1)
    allowed = recovery_allowed(points, cfg.max_recovery_distance)
    policies = {str(t): commit_prefix(points, output['confidence'], t, window, cfg.max_recovery_distance)[0]
                for t in thresholds}
    rows = []
    for i in range(len(points)):
        drift = None
        if 'gt_history' in batch and batch['gt_history_mask'][i, 0] > 0:
            value = float(batch['gt_history'][i, 0].norm())
            drift = value if math.isfinite(value) else None
        row = dict(drift=drift, departed=bool(batch['offtrack'][i]), states=1,
                   first_known=int(known[i, 0]), first_correct=int(known[i, 0]*labels[i, 0]),
                   commit_known=int(known[i, window-1]), commit_correct=int(known[i, window-1]*labels[i, window-1]),
                   positive_prefixes=int((labels[i]*known[i]).sum()), known_prefixes=int(known[i].sum()),
                   annotated_points=int(annotated[i].sum()), censored_points=int((annotated[i] & ~observable[i]).sum()),
                   recovery_blocked=int(not allowed[i]))
        for name, error in errors.items():
            valid = mask[i] & torch.isfinite(error[i])
            row[name+'_error_sum'] = float(error[i][valid].double().sum())
            row[name+'_error_count'] = int(valid.sum())
            row[name+'_nonfinite_count'] = int((mask[i] & ~torch.isfinite(error[i])).sum())
        comparable = bool(mask[i].any() and all(torch.isfinite(e[i][mask[i]]).all() for e in errors.values()))
        row['correction_comparable'] = int(comparable)
        delta = (errors['final'][i][mask[i]].mean()-errors['initial'][i][mask[i]].mean()) if comparable else 0.
        row['correction_improved'] = int(comparable and delta < -1e-6)
        row['correction_worsened'] = int(comparable and delta > 1e-6)
        for threshold, counts in policies.items():
            count = int(counts[i])
            assessed = count > 0 and bool(known[i, count-1])
            row['gate_'+threshold] = dict(
                false_stops=int(count == 0 and bool(known[i, 0]*labels[i, 0])),
                accepted_known=int(assessed),
                accepted_wrong=int(assessed and not labels[i, count-1]),
                accepted_unknown=int(count > 0 and not assessed),
                departed_continues=int(row['departed'] and count > 0))
        rows.append(row)
    return rows


def summarize_decisions(rows, n_commit):
    """Pool numerators/counts before dividing, including empty drift bands."""
    groups = {'all': rows}
    for name, lo, hi in (('<1', 0, 1), ('1-1.5', 1, 1.5), ('1.5-2', 1.5, 2),
                         ('2-3.5', 2, 3.5), ('>=3.5', 3.5, float('inf'))):
        groups[name] = [r for r in rows if not r['departed'] and r['drift'] is not None and lo <= r['drift'] < hi]
    groups['unknown'] = [r for r in rows if not r['departed'] and r['drift'] is None]
    groups['departed'] = [r for r in rows if r['departed']]

    def add(total, row):
        for key, value in row.items():
            if key in ('drift', 'departed'):
                continue
            if isinstance(value, dict):
                add(total.setdefault(key, {}), value)
            else:
                total[key] = total.get(key, 0)+value

    result = {}
    for name, members in groups.items():
        sums = {'states': 0}
        for row in members:
            add(sums, row)
        for stage in ('initial', 'final'):
            count = sums.get(stage+'_error_count', 0)
            sums[stage+'_error_mean'] = sums[stage+'_error_sum']/count if count else None
        result[name] = sums
    return dict(n_commit=n_commit, by_drift=result)
