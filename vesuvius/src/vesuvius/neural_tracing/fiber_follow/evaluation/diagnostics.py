"""Additive decision diagnostics on the original fiber, by state class and displacement."""
import math

import torch
import torch.nn.functional as F

from vesuvius.neural_tracing.fiber_follow.tracing.policy import DIAGNOSTIC_THRESHOLDS, commit_prefix, recovery_allowed
from vesuvius.neural_tracing.fiber_follow.data.labels import prefix_labels
from vesuvius.neural_tracing.fiber_follow.models.model import select_refinement
from vesuvius.neural_tracing.fiber_follow.data.state_labels import DISPLACEMENT_STRATA, SUPERVISION, TERMINAL, displacement_stratum
from vesuvius.neural_tracing.fiber_follow.train.supervision import commit_window, dense_commit_mask, geometry_mask


@torch.no_grad()
def decision_rows(output, batch, cfg, n_commit=None, tolerance=1.5, thresholds=DIAGNOSTIC_THRESHOLDS):
    """Small CPU records; no forward pass or random draws. Unknowns stay unknown."""
    output = {k: (v.detach().float() if v.is_floating_point() else v.detach()).cpu() for k, v in output.items()}
    batch = {k: v.detach().cpu() for k, v in batch.items() if isinstance(v, torch.Tensor)}
    window = commit_window(cfg, n_commit)
    output = select_refinement(output, cfg, n_commit=window)
    points = output['points']
    labels, known, _ = prefix_labels(points, batch, tolerance, cfg.max_recovery_distance)
    annotated = batch['dense_mask'].bool() & batch['geometry_valid'][:, None].bool()
    observable = geometry_mask(batch, cfg)
    near = dense_commit_mask(observable.shape[1], cfg, window, points.device)
    mask = observable & near
    errors = {}
    for name, curve in (('final', points), ('initial', output.get('initial_points', points))):
        dense = F.interpolate(curve[..., :2].transpose(1, 2), size=mask.shape[1],
                              mode='linear', align_corners=True).transpose(1, 2)
        errors[name] = (dense-batch['dense_ab']).norm(dim=-1)
    allowed = recovery_allowed(points, cfg.max_recovery_distance)
    policies = {}
    for threshold in thresholds:
        chosen = select_refinement(output, cfg, threshold, window)
        counts, _ = commit_prefix(chosen['points'], chosen['confidence'], threshold, window, cfg.max_recovery_distance)
        gate_labels, gate_known, _ = prefix_labels(chosen['points'], batch, tolerance, cfg.max_recovery_distance)
        policies[str(threshold)] = counts, gate_labels, gate_known
    rows = []
    for i in range(len(points)):
        distance = float(batch['match_distance'][i])
        row = dict(displacement=displacement_stratum(distance) if math.isfinite(distance) else 'unknown',
                   state=SUPERVISION[int(batch['supervision'][i])], states=1,
                   history_points=int(batch['hmask'][i].sum()),
                   first_confidence_sum=float(output['confidence'][i, 0]),
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
        for threshold, (counts, gate_labels, gate_known) in policies.items():
            count = int(counts[i])
            assessed = count > 0 and bool(gate_known[i, count-1])
            row['gate_'+threshold] = dict(
                false_stops=int(count == 0 and bool(gate_known[i, 0]*gate_labels[i, 0])),
                accepted_known=int(assessed),
                accepted_wrong=int(assessed and not gate_labels[i, count-1]),
                accepted_unknown=int(count > 0 and not assessed),
                terminal_continues=int(int(batch['supervision'][i]) == TERMINAL and count > 0))
        rows.append(row)
    return rows


def summarize_decisions(rows, n_commit):
    """Pool numerators/counts before dividing, including empty strata."""
    by_state = {'all': rows, **{name: [r for r in rows if r['state'] == name] for name in SUPERVISION}}
    by_displacement = {name: [r for r in rows if r['displacement'] == name]
                       for name in [s[0] for s in DISPLACEMENT_STRATA]+['unknown']}
    history_groups = {name: [r for r in rows if lo <= r['history_points'] <= hi]
                      for name, lo, hi in (('0', 0, 0), ('1-8', 1, 8), ('9-32', 9, 32), ('>32', 33, math.inf))}

    def add(total, row):
        for key, value in row.items():
            if key in ('displacement', 'state', 'history_points'):
                continue
            if isinstance(value, dict):
                add(total.setdefault(key, {}), value)
            else:
                total[key] = total.get(key, 0)+value

    def summarize(members):
        sums = {'states': 0}
        for row in members:
            add(sums, row)
        for stage in ('initial', 'final'):
            count = sums.get(stage+'_error_count', 0)
            sums[stage+'_error_mean'] = sums[stage+'_error_sum']/count if count else None
        sums['first_confidence_mean'] = sums.get('first_confidence_sum', 0)/len(members) if members else None
        return sums
    return dict(n_commit=n_commit, by_state={name: summarize(members) for name, members in by_state.items()},
                by_displacement={name: summarize(members) for name, members in by_displacement.items()},
                by_history={name: summarize(members) for name, members in history_groups.items()})
