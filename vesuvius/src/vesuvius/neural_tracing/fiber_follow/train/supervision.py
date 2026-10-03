"""Shared prediction and prefix-survival training objectives."""
import math

import torch
import torch.nn.functional as F

from vesuvius.neural_tracing.fiber_follow.data.labels import prefix_labels
from vesuvius.neural_tracing.fiber_follow.data.state_labels import RECOVERABLE, TERMINAL
from vesuvius.neural_tracing.fiber_follow.models.model import feature_grid
from vesuvius.neural_tracing.fiber_follow.models.survival_confidence import survival_loss

CONNECTOR_SPACING = .25  # same dense spatial resolution as the supervised path


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


def foreign_hits(points, batch, cfg):
    """Which (B, N, 3) crop-local points fall in cells of a validated neighboring path."""
    foreign = batch['foreign']
    grid = feature_grid(points.detach().float(), cfg.fine, foreign.shape[-3:])
    values = F.grid_sample(foreign[:, None].float(), grid[:, :, None, None], mode='nearest', align_corners=True)
    return values[:, 0, :, 0, 0] > .5


def foreign_failures(points, batch, cfg, count):
    """Dense predictions in cells occupied by a validated neighboring path."""
    dense = F.interpolate(points.detach().float().transpose(1, 2), size=count, mode='linear',
                          align_corners=True).transpose(1, 2)
    return foreign_hits(dense, batch, cfg)


def connector_failures(first, batch, cfg):
    """Foreign contact on the origin-to-first-point segment, excluding the origin itself.

    ``first`` is (B, 3). A safe endpoint does not make a connector safe if it crosses a
    known foreign fiber. Samples are ``CONNECTOR_SPACING`` apart along the longest
    allowed connection, so every committed connection is checked as densely as the path.
    """
    count = math.ceil(cfg.max_recovery_distance/CONNECTOR_SPACING)
    fraction = torch.arange(1, count+1, device=first.device, dtype=torch.float32)/count
    samples = first.detach().float()[:, None]*fraction[None, :, None]
    return foreign_hits(samples, batch, cfg).any(-1)


def geometry_mask(batch, cfg):
    """Certified original-fiber continuation inside the observable crop.

    Only states whose annotated first connection satisfies the commit limit carry
    geometry (``geometry_valid``); with a neighbor raster, a target connection that
    crosses a known foreign fiber is censored as well.
    """
    annotated = batch['dense_mask'].bool() & batch['geometry_valid'][:, None].bool()
    target = torch.where(annotated[..., None], batch['dense_ab'], 0.)
    observable = (target.abs().amax(-1) <= cfg.lateral_limit).int().cummin(-1).values.bool()
    mask = annotated & observable
    if 'foreign' in batch:
        mask = mask & ~target_connector_foreign(batch, cfg)[:, None]
    return mask


def target_connector_foreign(batch, cfg):
    first = torch.cat((batch['dense_ab'][:, 0], batch['dense_ab'].new_full((len(batch['dense_ab']), 1), cfg.future_step)), -1)
    return connector_failures(first, batch, cfg) & batch['geometry_valid'].bool()


def proposal_labels(points, batch, cfg, tolerance):
    """Prefix labels with the neighbor raster applied to the path and its connection."""
    extra = connector = None
    if 'foreign' in batch:
        extra = foreign_failures(points, batch, cfg, batch['dense_mask'].shape[1])
        connector = connector_failures(points[:, 0], batch, cfg)
    labels, known, error = prefix_labels(points, batch, tolerance, cfg.max_recovery_distance,
                                         extra_failure=extra, connector_failure=connector)
    return labels, known, error, extra


@torch.no_grad()
def point_correctness(points, batch, cfg, tolerance, foreign=None):
    """Independent predicted-point counts, not cumulative prefix labels.

    Score every proposed point regardless of confidence. Unavailable supervision or
    missing/crop-censored annotation is excluded; terminal states, known endpoints and
    validated neighboring fibers are negatives. An earlier error does not invalidate a
    later point. No origin-to-first-point policy is applied here.
    """
    points = points.detach().float()
    indices = torch.linspace(0, batch['dense_ab'].shape[1]-1, points.shape[1],
                             device=points.device).round().long()
    target = batch.get('plane_ab', batch['dense_ab'][:, indices]).float()
    annotated = batch.get('plane_mask', batch['dense_mask'][:, indices]).bool()
    visible = target.abs().amax(-1) <= cfg.lateral_limit
    annotated = annotated & visible & torch.isfinite(target).all(-1)
    error = (points[..., :2]-target).norm(dim=-1)
    terminal = batch['terminal'].bool()[:, None]
    beyond_end = (batch['endpoint_known'].bool()[:, None]
                  & (batch['end_local'][:, 2, None] >= 0)
                  & (points[..., 2] > batch['end_local'][:, 2, None]+1e-4)
                  & ~batch.get('plane_mask', batch['dense_mask'][:, indices]).bool())
    neighbor = torch.zeros_like(annotated) if foreign is None else foreign[:, indices].bool()
    known = (annotated | terminal | beyond_end | neighbor) & batch['confidence_valid'].bool()[:, None]
    correct = (known & annotated & (error <= tolerance) & torch.isfinite(points).all(-1)
               & ~terminal & ~beyond_end & ~neighbor)
    return dict(point_correct_count=correct.sum(), point_wrong_count=(known & ~correct).sum(),
                point_unknown_count=(~known).sum())


def loss_terms(output, batch, cfg, tolerance=1.5, *, n_commit=None):
    """Return numerators/counts so effective-batch means are independent of microbatch.

    Unknown/crop-censored targets and uncertified connections do not teach
    localization. Terminal states still teach rejection. Every generated proposal,
    including refinement attempts, receives confidence supervision under the same
    annotation, connection and neighbor semantics as the established evaluation.
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
        labels, known, _, _ = proposal_labels(curve, batch, cfg, tolerance)
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
    if cfg.model_type == 'flow_matching':
        geometry = output['flow_per_state']
    # Distance-only labels keep their names; identity-aware labels add the neighbor raster.
    labels, known, _ = prefix_labels(output['points'], batch, tolerance, cfg.max_recovery_distance)
    supervised, supervised_known, _, extra = proposal_labels(output['points'], batch, cfg, tolerance)
    _, confidence_valid = survival_loss(output['hazard_logits'], supervised, supervised_known)
    labeled = confidence_valid.any(-1)
    positive = (supervised*supervised_known).sum(-1)
    negative = ((1-supervised)*supervised_known).sum(-1)
    terms = dict(geometry_per_state=geometry,
                 confidence_per_state=confidence,
                 refinement_attempts_sum=count.sum(),
                 confidence_labeled_states=labeled.sum(),
                 confidence_terminal_states=(labeled & (batch['supervision'] == TERMINAL)).sum(),
                 confidence_recoverable_states=(labeled & (batch['supervision'] == RECOVERABLE)).sum(),
                 positive_targets_per_state=positive, negative_targets_per_state=negative,
                 geometry_states_per_state=mask.any(-1),
                 geometry_count=mask.sum(), confidence_count=supervised_known.sum(),
                 error_sum=torch.where(mask, (predicted-target).norm(dim=-1), 0.).sum(),
                 correct_count=(labels*known).sum())
    if 'foreign' in batch:
        terms.update(identity_correct_count=(supervised*supervised_known).sum(),
                     identity_flipped_count=(labels*known*(1-supervised)).sum(),
                     connector_rejected_targets=target_connector_foreign(batch, cfg).sum())
    terms.update(point_correctness(output['points'], batch, cfg, tolerance, foreign=extra))
    return terms
