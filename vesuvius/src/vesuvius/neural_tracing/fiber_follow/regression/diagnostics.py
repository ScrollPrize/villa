"""Additive decision diagnostics on the original fiber, including drift strata."""
import math

import torch
import torch.nn.functional as F

from vesuvius.neural_tracing.fiber_follow.shared.policy import DIAGNOSTIC_THRESHOLDS, commit_prefix, recovery_allowed
from vesuvius.neural_tracing.fiber_follow.shared.labels import prefix_labels
from vesuvius.neural_tracing.fiber_follow.regression.supervision import (
    commit_window, dense_commit_mask, geometry_mask,
)


@torch.no_grad()
def decision_rows(output, batch, cfg, n_commit=None, tolerance=1.5, thresholds=DIAGNOSTIC_THRESHOLDS):
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
    history_groups = {name: [r for r in rows if lo <= r['history_points'] <= hi]
                      for name, lo, hi in (('0', 0, 0), ('1-8', 1, 8), ('9-32', 9, 32), ('>32', 33, math.inf))}

    def add(total, row):
        for key, value in row.items():
            if key in ('drift', 'departed', 'history_points'):
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
    return dict(n_commit=n_commit, by_drift={name: summarize(members) for name, members in groups.items()},
                by_history={name: summarize(members) for name, members in history_groups.items()})



# History patch age bins (voxels behind the head); seed anchors are reported separately.
RANKING_BINS = (('4-16', 0, 16), ('17-48', 17, 48), ('49-128', 49, 128))


@torch.no_grad()
def identity_ranking(output, batch, cfg):
    """Does each on-fiber history patch prefer its fiber ahead over the negatives beside it?

    Counts per history-patch age bin and per positive forward distance, for
    diagnostics only; the model never receives a hand-computed similarity.
    """
    R, K = cfg.recent_patches, batch['positive_mask'].shape[1]
    M = batch['negative_mask'].shape[2]
    history = output['history_embedding'].float()
    on = (batch['patch_on_fiber']*output['patch_mask']).bool()
    query, support = output['query_embedding'].float(), output['query_support'].bool()
    positive, negative = query[:, :K], query[:, K:].reshape(len(query), K, M, -1)
    positive_ok = batch['positive_mask'].bool() & support[:, :K]
    negative_ok = batch['negative_mask'].bool() & support[:, K:].reshape(len(query), K, M)
    scored = positive_ok & negative_ok.any(-1)
    similarity = torch.einsum('bpe,bke->bpk', history, positive)
    competitor = torch.einsum('bpe,bkme->bpkm', history, negative).masked_fill(~negative_ok[:, None], -2.).amax(-1)
    correct = similarity > competitor
    valid = on[:, :, None] & scored[:, None]
    ages = torch.arange(1, R+1, device=query.device)*cfg.patch_every
    rows = {}
    for name, lo, hi in RANKING_BINS:
        member = valid[:, :R] & ((ages >= lo) & (ages <= hi))[None, :, None]
        rows[name] = (int((correct[:, :R] & member).sum()), int(member.sum()))
    anchors = valid[:, R:]
    rows['anchor'] = (int((correct[:, R:] & anchors).sum()), int(anchors.sum()))
    forward = batch['identity_points'][:, :K, 2]
    for name, lo, hi in (('ahead 1-8', 1, 8), ('ahead 8-20', 8, 20.01)):
        member = valid & ((forward >= lo) & (forward < hi))[:, None]
        rows[name] = (int((correct & member).sum()), int(member.sum()))
    return rows


def summarize_ranking(rows):
    total = {}
    for row in rows:
        for name, (correct, count) in row.items():
            a, b = total.get(name, (0, 0))
            total[name] = (a+correct, b+count)
    return {name: dict(correct=a, pairs=b, accuracy=a/b if b else None) for name, (a, b) in total.items()}
