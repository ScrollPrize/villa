"""Direct dense curve regression plus correctness of the produced prefix."""
import torch
import torch.nn.functional as F

from vesuvius.neural_tracing.fiber_follow.shared.labels import prefix_labels
from vesuvius.neural_tracing.fiber_follow.regression.model import feature_grid
from .survival_confidence import survival_loss


def geometry_mask(batch, cfg):
    annotated = batch['dense_mask'].bool() & ~batch['offtrack'][:, None].bool()
    if 'identity_observable' in batch:
        annotated &= batch['identity_observable'][:, None].bool()
    target = torch.where(annotated[..., None], batch['dense_ab'], 0.)
    observable = (target.abs().amax(-1) <= cfg.lateral_limit).int().cummin(-1).values.bool()
    return annotated & observable


def commit_window(cfg, n_commit):
    window = min(4, cfg.n_future) if n_commit is None else n_commit
    if not 1 <= window <= cfg.n_future:
        raise ValueError('Commit window must fit forecast')
    return window


def dense_commit_mask(count, cfg, n_commit, device):
    planes = torch.linspace(cfg.future_step, cfg.n_future*cfg.future_step, count, device=device)
    return planes <= n_commit*cfg.future_step+1e-6


def window_mean(values, mask, near):
    """Half commit-window, half full-horizon; empty windows contribute zero."""
    def mean(selected):
        return torch.where(selected, values, 0.).sum(-1)/selected.sum(-1).clamp_min(1)
    return .5*mean(mask & near)+.5*mean(mask)


def foreign_failures(points, batch, cfg, count):
    """Dense predictions in cells occupied by a validated neighboring path."""
    dense = F.interpolate(points.detach().float().transpose(1, 2), size=count, mode='linear',
                          align_corners=True).transpose(1, 2)
    foreign = batch['foreign']
    grid = feature_grid(dense, cfg.fine, foreign.shape[-3:])
    values = F.grid_sample(foreign[:, None].float(), grid[:, :, None, None], mode='nearest', align_corners=True)
    return values[:, 0, :, 0, 0] > .5


@torch.no_grad()
def candidate_targets(batch, cfg, tolerance):
    """Label supplied paths with the deployed dense first-failure contract.

    Called by the loader after foreign masks are built, before augmentation.
    Candidate support only censors; it never manufactures a positive prefix.
    """
    labels, known = [], []
    for points in batch['candidate_points'].unbind(1):
        foreign = foreign_failures(points, batch, cfg, batch['dense_mask'].shape[-1])
        target, mask, _ = prefix_labels(points, batch, tolerance, cfg.max_recovery_distance,
                                        extra_failure=foreign)
        labels.append(target)
        known.append(mask)
    mask = torch.stack(known, 1)*batch['candidate_mask']
    mask *= batch['identity_observable'][:, None, None]
    return torch.stack(labels, 1), mask


def weighted_state_sum(values, batch):
    """Keep the stream's fixed loss budget across chunks and endpoint replay."""
    return (values*batch.get('loss_weight', torch.ones_like(values))).sum()


@torch.no_grad()
def point_correctness(points, batch, cfg, tolerance, foreign=None):
    """Independent predicted-point counts, not cumulative prefix labels.

    Score every proposed point regardless of confidence. Unknown identity or
    missing/crop-censored annotation is excluded; known departures/endpoints
    and validated neighboring fibers are negatives. An earlier error does not
    invalidate a later point. No origin-to-first-point policy is applied here.
    """
    points = points.detach().float()
    indices = torch.linspace(0, batch['dense_ab'].shape[1]-1, points.shape[1],
                             device=points.device).round().long()
    target = batch.get('plane_ab', batch['dense_ab'][:, indices]).float()
    annotated = batch.get('plane_mask', batch['dense_mask'][:, indices]).bool()
    visible = target.abs().amax(-1) <= cfg.lateral_limit
    annotated = annotated & visible & torch.isfinite(target).all(-1)
    error = (points[..., :2]-target).norm(dim=-1)
    departed = batch['offtrack'].bool()[:, None]
    beyond_end = (batch['endpoint_known'].bool()[:, None]
                  & (batch['end_local'][:, 2, None] >= 0)
                  & (points[..., 2] > batch['end_local'][:, 2, None]+1e-4)
                  & ~batch.get('plane_mask', batch['dense_mask'][:, indices]).bool())
    neighbor = torch.zeros_like(annotated) if foreign is None else foreign[:, indices].bool()
    known = annotated | departed | beyond_end | neighbor
    if 'identity_observable' in batch:
        known &= batch['identity_observable'].bool()[:, None]
    correct = (known & annotated & (error <= tolerance) & torch.isfinite(points).all(-1)
               & ~departed & ~beyond_end & ~neighbor)
    return dict(point_correct_count=correct.sum(), point_wrong_count=(known & ~correct).sum(),
                point_unknown_count=(~known).sum())


def loss_terms(output, batch, cfg, tolerance=1.5, *, n_commit=None):
    """Return numerators/counts so effective-batch means are independent of microbatch.

    Unknown/crop-censored targets and departed states do not teach localization.
    Confirmed departures still teach rejection. Prefix correctness uses the
    same annotation semantics as the established tracer evaluation.
    """
    mask = geometry_mask(batch, cfg)
    window = commit_window(cfg, n_commit)
    near = dense_commit_mask(mask.shape[1], cfg, window, mask.device)
    predicted = F.interpolate(output['points'][..., :2].transpose(1, 2),
                              size=mask.shape[1], mode='linear', align_corners=True).transpose(1, 2)
    target = torch.where(mask[..., None], batch['dense_ab'], 0.)
    # Train every generated proposal, even when selection retains an earlier one.
    # The last attempted pass gets 75%; earlier passes share 25%. Survival averages attempts
    # so adding feedback iterations does not multiply the confidence loss weight.
    geometry_losses, confidence_losses = [], []
    for curve, hazards in zip(output['refinement_points'].unbind(1),
                              output['refinement_hazard_logits'].unbind(1)):
        dense = F.interpolate(curve[..., :2].transpose(1, 2), size=mask.shape[1],
                              mode='linear', align_corners=True).transpose(1, 2)
        error = F.smooth_l1_loss(dense, target, beta=1., reduction='none').mean(-1)
        geometry_losses.append(window_mean(error, mask, near))
        foreign = foreign_failures(curve, batch, cfg, mask.shape[1]) if 'foreign' in batch else None
        labels, known, _ = prefix_labels(curve, batch, tolerance, cfg.max_recovery_distance,
                                        extra_failure=foreign)
        if 'identity_observable' in batch:
            known = known*batch['identity_observable'][:, None]
        confidence_losses.append(survival_loss(hazards, labels, known)[0])
    attempts = output['refinement_mask'].bool()
    count = attempts.sum(1)
    last = count-1
    geometry_losses = torch.stack(geometry_losses, 1)
    final = geometry_losses.gather(1, last[:, None]).squeeze(1)
    earlier = attempts & (torch.arange(attempts.shape[1], device=mask.device)[None] < last[:, None])
    auxiliary = torch.where(earlier, geometry_losses, 0.).sum(1)/(count-1).clamp_min(1)
    geometry = torch.where(count > 1, .75*final+.25*auxiliary, final)
    confidence = torch.where(attempts, torch.stack(confidence_losses, 1), 0.).sum(1)/count
    labels, known, _ = prefix_labels(output['points'], batch, tolerance, cfg.max_recovery_distance)
    # Identity-aware labels: a point on a validated neighboring fiber is
    # wrong even within the distance tolerance. Distance metrics keep their names.
    identity = 'foreign' in batch
    supervised, supervised_known = labels, known
    extra = None
    if identity:
        extra = foreign_failures(output['points'], batch, cfg, mask.shape[1])
        supervised, supervised_known, _ = prefix_labels(output['points'], batch, tolerance,
                                                        cfg.max_recovery_distance, extra_failure=extra)
    if 'identity_observable' in batch:
        observed = batch['identity_observable'][:,None]
        known = known*observed
        supervised_known = supervised_known*observed
    _, confidence_valid = survival_loss(output['hazard_logits'], supervised, supervised_known)
    terms = dict(geometry_per_state=geometry,
                 confidence_per_state=confidence,
                 refinement_attempts_sum=count.sum(),
                 confidence_labeled_states=confidence_valid.any(-1).sum(),
                 confidence_departed_states=(confidence_valid.any(-1) & batch['offtrack'].bool()).sum(),
                 geometry_count=mask.sum(), confidence_count=supervised_known.sum(),
                 error_sum=torch.where(mask, (predicted-target).norm(dim=-1), 0.).sum(),
                 correct_count=(labels*known).sum())
    terms.update(point_correctness(output['points'], batch, cfg, tolerance, foreign=extra))
    if identity:
        terms.update(identity_correct_count=(supervised*supervised_known).sum(),
                     identity_flipped_count=(labels*known*(1-supervised)).sum())
    if 'candidate_confidence_logits' in output:
        mask = batch['candidate_mask'].bool()
        if 'identity_observable' in batch:
            mask = mask & batch['identity_observable'][:,None,None]
        labels = batch['candidate_labels'].float()
        per_candidate, mask = survival_loss(output['candidate_hazard_logits'], labels, mask)
        valid = mask.any(-1)
        terms['candidate_per_state'] = per_candidate.sum(-1)/valid.sum(-1).clamp_min(1)
        terms['candidate_states'] = valid.any(-1).sum()
        terms['candidate_intervals'] = mask.sum()
        failures = mask & (labels < .5)
        terms['candidate_late_failures'] = failures[..., 1:].sum()
        terms['candidate_first_failures'] = failures[..., 0].sum()
        terms['candidate_supervision_weight'] = weighted_state_sum(valid.any(-1).float(), batch)
    return terms
