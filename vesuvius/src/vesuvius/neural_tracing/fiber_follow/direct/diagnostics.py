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


def plot_judge_sequence(sequence, logits, path, *, audit=None, title='Observed CT judge input'):
    """Exact model CT with marker contours, raw scores and supervision masks."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    images = sequence['images'][0].detach().cpu().numpy()
    scores = logits[0].sigmoid().detach().cpu().numpy()
    targets = sequence.get('target')
    known = sequence.get('known')
    eligible = sequence.get('eligible')
    indices = np.linspace(0, len(images)-1, min(12,len(images))).astype(int)
    fig, axes = plt.subplots(3,len(indices),figsize=(2.3*len(indices),7),squeeze=False)
    lo, hi = np.quantile(images[:,:,0], [.01,.995])
    for col, i in enumerate(indices):
        for view in range(3):
            ax = axes[view,col]
            ax.imshow(images[i,view,0],cmap='gray',vmin=lo,vmax=max(hi,lo+1e-6),origin='lower')
            ax.contour(images[i,view,1],levels=[.5],colors=['cyan'],linewidths=.5)
            ax.set_xticks([]); ax.set_yticks([])
            if view == 0:
                label = '?' if known is None or not bool(known[0,i]) else str(int(targets[0,i]))
                support = bool(images[i,:,2].all())
                elig = bool(eligible[0,i]) if eligible is not None else support
                ax.set_title(f'{i}: p={scores[i]:.2f} GT={label}\nsupport={support} eligible={elig}',fontsize=8)
    if audit:
        title += f" | accepted={audit['accepted']:.1f}, endpoint={audit['endpoint']:.1f}"
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path,dpi=110)
    plt.close(fig)


def plot_judge_path(path, accepted, events, destination):
    """Observed retained/discarded geometry and operational GT onset bracket."""
    import matplotlib.pyplot as plt
    import numpy as np
    from ..events import truncate_path
    from ..geometry import arclength, interp_at
    path=np.asarray(path)
    kept=truncate_path(path,accepted)
    discarded=np.concatenate((kept[-1:],path[arclength(path)>accepted]))
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,(a,b) in zip(axes,((0,1),(0,2),(1,2))):
        ax.plot(kept[:,a]-path[0,a],kept[:,b]-path[0,b],color='green',label='accepted')
        ax.plot(discarded[:,a]-path[0,a],discarded[:,b]-path[0,b],color='red',label='discarded')
        if events.bracket is not None:
            points=interp_at(path,arclength(path),np.clip(events.bracket,0,arclength(path)[-1]))
            ax.scatter(points[:,a]-path[0,a],points[:,b]-path[0,b],marker='x',color='black',label='GT onset interval')
        ax.set_xlabel('xyz'[a]+' from seed');ax.set_ylabel('xyz'[b]+' from seed')
        ax.set_aspect('equal',adjustable='datalim')
    axes[0].legend()
    fig.suptitle(f'Observed path; accepted arc {accepted:g}; event {events.kind or "none/unknown"}')
    fig.tight_layout();fig.savefig(destination,dpi=110);plt.close(fig)
