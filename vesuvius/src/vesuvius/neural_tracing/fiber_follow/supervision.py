"""Dense prefix correctness of the one curve actually generated."""
import torch
import torch.nn.functional as F
from vesuvius.neural_tracing.fiber_follow.policy import DEFAULT_MAX_RECOVERY_DISTANCE, DEFAULT_N_COMMIT, recovery_allowed
from vesuvius.neural_tracing.fiber_follow.model import flow_targets


@torch.no_grad()
def refinement_metrics(steps, batch, cfg, n_commit=DEFAULT_N_COMMIT):
    """Commit-window crossing error for the initial curve and every midpoint update.

    Use the same observable GT mask at every step. Drift is the distance from
    the current origin to its original-fiber correspondence, not the nearest
    fiber. Departed states are excluded. Sums/counts support pooling log rows;
    nonfinite predictions are counted separately, never replaced with zero error.
    """
    near = min(n_commit, cfg.n_future)
    target, mask, _ = flow_targets(batch, cfg)
    known = mask[:, :near].bool()
    error = (steps[:, :, :near, :2].float()-target[:, None, :near, :2]).norm(dim=-1)
    finite = torch.isfinite(error)
    valid = known[:, None] & finite
    safe_error = torch.where(valid, error, 0.)
    current = batch['gt_history'][:, 0].float()
    drift = current.norm(dim=-1)
    drift_known = (batch['gt_history_mask'][:, 0] > 0) & torch.isfinite(drift)
    eligible = ~batch['offtrack'].bool()
    bands = {'all': eligible}
    for name, lo, hi in (('<1',0.,1.), ('1-1.5',1.,1.5), ('1.5-2',1.5,2.),
                         ('2-3.5',2.,3.5), ('>=3.5',3.5,float('inf'))):
        bands[name] = eligible & drift_known & (drift >= lo) & (drift < hi)
    bands['unknown'] = eligible & ~drift_known
    # Compare state means only when all known points are finite in both steps.
    state_error = safe_error.sum(-1)/known.sum(-1)[:, None].clamp_min(1)
    complete = (finite | ~known[:, None]).all(-1) & known.any(-1)[:, None]
    comparable = complete[:, 1:] & complete[:, :-1]
    delta = state_error[:, 1:]-state_error[:, :-1]
    result = {}
    for name, member in bands.items():
        selected = member[:, None, None]
        counts = (valid & selected).sum((0, 2))
        sums = torch.where(selected, safe_error, 0.).double().sum((0, 2))
        pairs = comparable & member[:, None]
        result[name] = dict(
            state_count=int(member.sum()),
            known_point_count=int((known & member[:, None]).sum()),
            point_count=counts.tolist(), error_sum=sums.tolist(),
            error_mean=[float(s/n) if n else None for s,n in zip(sums,counts)],
            nonfinite_point_count=(known[:, None] & ~finite & selected).sum((0, 2)).tolist(),
            comparison_count=pairs.sum(0).tolist(),
            improved_count=(pairs & (delta < -1e-6)).sum(0).tolist(),
            worsened_count=(pairs & (delta > 1e-6)).sum(0).tolist())
    return dict(first_n=near, departed_count=int((~eligible).sum()), by_drift=result)

@torch.no_grad()
def prefix_labels(points, batch, tolerance=1.5,
                     max_recovery_distance=DEFAULT_MAX_RECOVERY_DISTANCE):
    """Prefix agreement sampled every 0.25 forward voxel with the default config.

    A prefix is positive only when every sampled crossing is annotated and
    within tolerance. A known failure is negative even if later GT is missing.
    Unknown ends are censored. Tagged physical endpoints and already departed
    states supply explicit negatives. The origin-to-first-point connection must
    obey the same recovery distance limit as tracing. Within that limit, the
    first point may recover from a displaced origin; GT agreement begins there.
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
    failed = ((known & ((error > tolerance) | ~torch.isfinite(error))) | beyond_end
              | batch['offtrack'][:, None].bool() | bad_recovery[..., None])
    failure_prefix = failed.cumsum(-1) > 0
    known_prefix = (known | beyond_end).int().cummin(-1).values.bool()
    indices = torch.linspace(0, Q-1, K, device=points.device).round().long()
    target = (~failure_prefix[..., indices]).float()
    mask = (failure_prefix | known_prefix)[..., indices].float()
    valid_error = torch.where(known, error.nan_to_num(nan=8).clamp(max=8), 0.).sum(-1)/known.sum(-1).clamp(min=1)
    return target, mask, valid_error

def masked_bce(logits, target, mask, denominator=None):
    loss = F.binary_cross_entropy_with_logits(logits.float(), target, reduction='none')
    denominator = mask.sum() if denominator is None else mask.new_tensor(denominator)
    return (loss*mask).sum()/denominator.clamp_min(1)


def loss_fn(output, batch, cfg, tolerance=1.5, *, update=0, confidence_ramp=2000, compute_metrics=True, normalizers=None,
            n_commit=DEFAULT_N_COMMIT):
    """Flow loss plus confidence BCE, half over the commit window and half over the full horizon.

    ``n_commit`` is the rollout commit limit; the near term covers exactly the prefixes a
    decision can commit, clipped to the model horizon.
    """
    target, mask, error = prefix_labels(output['points'],batch,tolerance,cfg.max_recovery_distance)
    logits = output['confidence_logits']
    normalizers = normalizers or {}
    window = min(n_commit, logits.shape[1])
    near = masked_bce(logits[:,:window],target[:,:window],mask[:,:window],normalizers.get('near'))
    full = masked_bce(logits,target,mask,normalizers.get('full'))
    flow = output['flow_loss']
    if 'flow' in normalizers:
        flow = flow*output['flow_known_count']/max(1.,normalizers['flow'])
    confidence = .5*near+.5*full
    coefficient = min(1., max(0.,update/max(1,confidence_ramp)))
    total = flow+coefficient*confidence
    metrics = {}
    if compute_metrics:
        metrics = dict(flow=flow.item(), confidence_loss=confidence.item(),
                       confidence_commit=near.item(),confidence_all=full.item(),confidence_coefficient=coefficient,
                       commit_window=window,
                       flow_known_fraction=output['flow_known_fraction'].item(),
                       flow_censored_fraction=output['flow_censored_fraction'].item(),
                       commit_correct_count=(target[:,window-1]*mask[:,window-1]).sum().item(),
                       commit_known_count=mask[:,window-1].sum().item())
        for threshold in (.5,.85):
            eligible = recovery_allowed(output['points'],cfg.max_recovery_distance)
            open_gate = eligible & (output['confidence'][:,0]>=threshold)
            known = mask[:,0].bool()
            correct = target[:,0].bool()
            metrics.update({f'false_stop_count_{threshold}': (known & correct & ~open_gate).sum().item(),
                            f'correct_first_count_{threshold}': (known & correct).sum().item(),
                            f'departed_continue_count_{threshold}': (batch['offtrack'].bool() & open_gate).sum().item(),
                            f'departed_count_{threshold}': batch['offtrack'].sum().item()})
    return total,metrics
