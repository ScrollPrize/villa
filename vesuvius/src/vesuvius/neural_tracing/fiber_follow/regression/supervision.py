"""Direct dense curve regression plus correctness of the produced prefix."""
import torch
import torch.nn.functional as F

from vesuvius.neural_tracing.fiber_follow.shared.labels import prefix_labels
from vesuvius.neural_tracing.fiber_follow.regression.model import feature_grid


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


def identity_terms(output, batch, cfg, temperature=.1):
    """InfoNCE between pooled on-fiber recent history and appearance ahead.

    The anchor is the normalized mean over visible reference positions lying on the annotated
    fiber (at least two). A visible observed seed supplies the
    reference if recent history is insufficient; the two references are never
    added as separate losses. Each annotated positive must outscore only the
    validated centerline negatives beside it; it never
    has to reach similarity one. States without a scored positive contribute zero.
    """
    R = cfg.n_history
    on = batch['reference_on_fiber'][:, :R]*output['reference_mask'][:, :R]
    recent_ok = on.sum(1) >= 2
    seed_on = batch['reference_on_fiber'][:, R:]*output['reference_mask'][:, R:]
    seed_ok = ~recent_ok & (seed_on.sum(1) >= 1)
    recent = (output['reference_embedding'][:, :R].float()*on[..., None]).sum(1)
    seed = (output['reference_embedding'][:, R:].float()*seed_on[..., None]).sum(1)
    anchor = F.normalize(torch.where(seed_ok[:,None],seed,recent),dim=-1)
    K = batch['positive_mask'].shape[1]
    M = batch['negative_mask'].shape[2]
    query = output['query_embedding'].float()
    support = output['query_support'].bool()
    positive, negative = query[:, :K], query[:, K:].reshape(len(query), K, M, -1)
    positive_ok = batch['positive_mask'].bool() & support[:, :K]
    negative_ok = batch['negative_mask'].bool() & support[:, K:].reshape(len(query), K, M)
    valid = positive_ok & negative_ok.any(-1) & (recent_ok | seed_ok)[:, None]
    if 'identity_observable' in batch:
        valid &= batch['identity_observable'][:,None]
    positive_logit = (anchor[:, None]*positive).sum(-1)/temperature
    negative_logit = ((anchor[:, None, None]*negative).sum(-1)/temperature).masked_fill(~negative_ok, float('-inf'))
    loss = torch.logsumexp(torch.cat((positive_logit[..., None], negative_logit), -1), -1)-positive_logit
    loss = torch.where(valid, loss, 0.)
    return dict(identity_per_state=loss.sum(-1)/valid.sum(-1).clamp_min(1),
                identity_pair_valid=valid,identity_pair_loss=loss,
                identity_anchor_source=recent_ok.long()+2*seed_ok.long(),
                identity_count=valid.sum(), identity_states=valid.any(-1).sum(),
                identity_rank_correct=(valid & (positive_logit > negative_logit.amax(-1))).sum())


def loss_terms(output, batch, cfg, tolerance=1.5, *, n_commit=None, identity_temperature=.1):
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
    error = F.smooth_l1_loss(predicted, target, beta=1., reduction='none').mean(-1)
    geometry = window_mean(error, mask, near)
    initial_geometry = geometry
    if cfg.correction or cfg.recurrent_refinement_steps:
        initial = F.interpolate(output['initial_points'][..., :2].transpose(1, 2),
                                size=mask.shape[1], mode='linear', align_corners=True).transpose(1, 2)
        initial_error = F.smooth_l1_loss(initial, target, beta=1., reduction='none').mean(-1)
        initial_geometry = window_mean(initial_error, mask, near)
        # Supervise every earlier proposal without increasing the total loss
        # weight as refinement steps are added. The final curve gets 75%.
        auxiliary = initial_geometry
        if 'refinement_points' in output:
            earlier = output['refinement_points'][:, :-1]
            losses = []
            for curve in earlier.unbind(1):
                dense = F.interpolate(curve[..., :2].transpose(1, 2),
                    size=mask.shape[1], mode='linear', align_corners=True).transpose(1, 2)
                errors = F.smooth_l1_loss(dense, target, beta=1., reduction='none').mean(-1)
                losses.append(window_mean(errors, mask, near))
            auxiliary = torch.stack(losses).mean(0)
        geometry = .75*geometry+.25*auxiliary
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
    bce = F.binary_cross_entropy_with_logits(output['confidence_logits'], supervised, reduction='none')
    terms = dict(geometry_per_state=geometry,
                 confidence_per_state=window_mean(bce, supervised_known.bool(),
                                                  torch.arange(cfg.n_future, device=mask.device) < window),
                 geometry_count=mask.sum(), confidence_count=supervised_known.sum(),
                 error_sum=torch.where(mask, (predicted-target).norm(dim=-1), 0.).sum(),
                 correct_count=(labels*known).sum())
    terms.update(point_correctness(output['points'], batch, cfg, tolerance, foreign=extra))
    if identity:
        terms.update(identity_correct_count=(supervised*supervised_known).sum(),
                     identity_flipped_count=(labels*known*(1-supervised)).sum())
    if 'query_embedding' in output and 'positive_mask' in batch:
        terms.update(identity_terms(output, batch, cfg, identity_temperature))
    if 'candidate_confidence_logits' in output:
        mask = batch['candidate_mask'].bool()
        if 'identity_observable' in batch:
            mask = mask & batch['identity_observable'][:,None,None]
        labels = batch['candidate_labels'].float()
        bce = F.binary_cross_entropy_with_logits(output['candidate_confidence_logits'], labels, reduction='none')
        per_candidate = window_mean(bce, mask, torch.arange(cfg.n_future, device=mask.device) < window)
        valid = mask.any(-1)
        terms['candidate_per_state'] = per_candidate.sum(-1)/valid.sum(-1).clamp_min(1)
        terms['candidate_states'] = valid.any(-1).sum()
    if 'route_logits' in output:
        from .spatial_supervision import route_loss_terms
        terms.update(route_loss_terms(output,batch,cfg))
    return terms


def memory_probe_terms(output, batch):
    """Per-write departure and original-fiber offset probe on the recurrent memory.

    Targets are annotation-derived and never enter the writer. Each state
    averages over its labeled writes, so a long track weighs as much as a head.
    """
    probe = output['memory_probe'].float()
    identity_mask = batch['memory_target_identity_mask'].bool()
    target = batch['memory_target_identity'].float()
    bce = F.binary_cross_entropy_with_logits(probe[..., 0], target, reduction='none')
    identity = torch.where(identity_mask, bce, 0.).sum(-1)/identity_mask.sum(-1).clamp_min(1)
    offset_mask = batch['memory_target_offset_mask'].bool()
    offset_target = batch['memory_target_offset'].float()
    error = F.smooth_l1_loss(probe[..., 1:], offset_target, beta=1., reduction='none').sum(-1)
    offset = torch.where(offset_mask, error, 0.).sum(-1)/offset_mask.sum(-1).clamp_min(1)
    correct = ((probe[..., 0] > 0) == (target > .5)) & identity_mask
    departed = identity_mask & (target < .5)
    distance = (probe[..., 1:]-offset_target).norm(dim=-1)
    return dict(memory_identity_per_state=identity, memory_offset_per_state=offset,
                memory_identity_count=identity_mask.sum(), memory_identity_correct=correct.sum(),
                memory_departed_count=departed.sum(), memory_departed_correct=(correct & departed).sum(),
                memory_offset_count=offset_mask.sum(),
                memory_offset_error_sum=torch.where(offset_mask, distance, 0.).sum())
