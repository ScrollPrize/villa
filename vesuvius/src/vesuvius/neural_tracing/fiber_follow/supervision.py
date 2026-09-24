"""All proposal, ranking and continuation labels come from dense controlled GT."""
import torch
import torch.nn.functional as F


def coordinate_loss(candidates, cleaned, batch, cfg):
    """One whole-path winner, a shared near-term trunk, and supervised steps.

    Assignment is over the complete known trajectory, so it cannot stitch
    different modes together point by point. Unknown targets and departed
    states contribute no coordinate gradients. Teacher paths never enter this
    loss. No repulsion pushes modes away when the continuation is unambiguous.
    """
    paths = candidates[:, :cfg.n_candidates].float()
    B, M, K, _ = paths.shape
    known = batch['dense_mask'].float() * (1-batch['offtrack'])[:, None]
    Q = known.shape[1]
    dense = F.interpolate(paths[..., :2].reshape(B*M, K, 2).transpose(1, 2),
                          size=Q, mode='linear', align_corners=True).transpose(1, 2).reshape(B, M, Q, 2)
    error = F.smooth_l1_loss(dense, batch['dense_ab'][:, None].expand_as(dense), reduction='none').mean(-1)
    per_mode = (error*known[:, None]).sum(-1)/known.sum(-1, keepdim=True).clamp_min(1)
    valid = (known.sum(-1) > 0).float()
    winner = per_mode.detach().argmin(-1)
    chosen = per_mode[torch.arange(B, device=paths.device), winner]
    full = (chosen*valid).sum()/valid.sum().clamp_min(1)
    z = torch.linspace(cfg.future_step, K*cfg.future_step, Q, device=paths.device)
    early_mask = known * (z <= min(4, K)*cfg.future_step)[None]
    early_valid = (early_mask.sum(-1) > 0).float()
    trunk = (error*early_mask[:, None]).sum((1, 2))/(M*early_mask.sum(-1).clamp_min(1))
    trunk = (trunk*early_valid).sum()/early_valid.sum().clamp_min(1)
    path = paths[torch.arange(B, device=paths.device), winner, :, :2]
    pred = torch.cat([cleaned[:, :1, :2], path], 1)
    truth = torch.cat([batch['clean_local'][:, :1, :2], batch['plane_ab']], 1)
    supplied = torch.cat([batch['clean_mask'][:, :1], batch['plane_mask']], 1)
    step_mask = supplied[:, 1:]*supplied[:, :-1]*(1-batch['offtrack'])[:, None]
    step_error = F.smooth_l1_loss(pred[:, 1:]-pred[:, :-1], truth[:, 1:]-truth[:, :-1], reduction='none').mean(-1)
    steps = (step_error*step_mask).sum()/step_mask.sum().clamp_min(1)
    return full + .5*trunk + .25*steps, dict(coordinate_full=full.item(), coordinate_trunk=trunk.item(), coordinate_steps=steps.item())


def teacher_candidates(batch, cfg):
    """GT and perturbed paths bootstrap the scorer before proposals are useful.

    Proposals from the live network are always included separately. Padding at
    unknown planes has no labels, and these synthetic candidates are never used
    when reporting proposal recall or selected-candidate error.
    """
    ab = batch['plane_ab']
    B, K, _ = ab.shape
    forward = cfg.future_step*torch.arange(1, K+1, device=ab.device)
    gt = torch.cat([ab, forward[None, :, None].expand(B, -1, 1)], -1)
    offsets = torch.randn(B, 4, 1, 2, device=ab.device)
    offsets = offsets / offsets.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    offsets = offsets * torch.tensor([.5, 1.5, 3., 5.], device=ab.device)[None, :, None, None]
    ramp = torch.linspace(.3, 1., K, device=ab.device)[None, None, :, None]
    jitter = gt[:, None].expand(-1, 4, -1, -1).clone()
    jitter[..., :2] += offsets*ramp
    return torch.cat([gt[:, None], jitter, batch['replay_candidates']], 1)


def candidate_labels(candidates, batch, tolerance=1.5):
    """Prefix agreement sampled every 0.5 forward voxel (with default config).

    A prefix is positive only when every sampled crossing is annotated and
    within tolerance. A known failure is negative even if later GT is missing.
    Unknown ends are censored. Tagged physical endpoints and already departed
    states supply explicit negatives. The first prediction is a recovery point:
    prefixes begin there, allowing a perturbed state to return to its fiber.
    """
    B, M, K, _ = candidates.shape
    Q = batch['dense_ab'].shape[1]
    dense = F.interpolate(candidates[..., :2].reshape(B*M, K, 2).transpose(1, 2),
                          size=Q, mode='linear', align_corners=True).transpose(1, 2).reshape(B, M, Q, 2)
    error = (dense-batch['dense_ab'][:, None]).norm(dim=-1)
    known = batch['dense_mask'][:, None].bool().expand(-1, M, -1)
    c = torch.linspace(0, 1, Q, device=candidates.device)
    forward = candidates[..., 0, 2, None]*(1-c)+candidates[..., -1, 2, None]*c
    beyond_end = batch['endpoint_known'][:, None, None].bool() & (forward > batch['end_local'][:, None, 2, None]+1e-4)
    beyond_end = beyond_end & ~known & (batch['end_local'][:, None, 2, None] >= 0)
    failed = (known & (error > tolerance)) | beyond_end | batch['offtrack'][:, None, None].bool()
    failure_prefix = failed.cumsum(-1) > 0
    known_prefix = (known | beyond_end).int().cummin(-1).values.bool()
    indices = torch.linspace(0, Q-1, K, device=candidates.device).round().long()
    target = (~failure_prefix[..., indices]).float()
    mask = (failure_prefix | known_prefix)[..., indices].float()
    valid_error = (error.clamp(max=8)*known).sum(-1)/known.sum(-1).clamp(min=1)
    quality = (target*mask).sum(-1)/mask.sum(-1).clamp(min=1) - .1*valid_error
    return target, mask, quality, valid_error


def loss_fn(output, batch, cfg, tolerance=1.5, rank_weight=1., confidence_weight=1., clean_weight=.5,
            rank_temperature=20.):
    candidates = output['candidates']
    labels, mask, quality, error = candidate_labels(candidates.detach(), batch, tolerance)
    M = cfg.n_candidates
    candidate_valid = torch.ones_like(quality)
    # Synthetic candidates require an annotated first plane and a recoverable state.
    candidate_valid[:, M:M+5] = (batch['plane_mask'][:, :1] * (1-batch['offtrack'][:, None]))
    if candidate_valid.shape[1] > M+5:
        candidate_valid[:, M+5:] = batch['replay_valid']
    mask = mask*candidate_valid[..., None]
    bce = F.binary_cross_entropy_with_logits(output['confidence_logits'].float(), labels, reduction='none')
    confidence = (bce*mask).sum()/mask.sum().clamp(min=1)
    rank_valid = candidate_valid * (batch['dense_mask'].sum(-1) > 0)[:, None]
    rank_valid = rank_valid * (1-batch['offtrack'][:, None])
    # Quality differences among correct candidates are hundredths of a unit
    # (0.1 x mean error); a soft temperature makes the target near-uniform and
    # the loss pure noise once proposals are good. Keep the target peaked.
    target = (quality.detach()*rank_temperature).masked_fill(rank_valid == 0, -1e4).softmax(-1)
    logp = output['ranks'].float().masked_fill(rank_valid == 0, -1e4).log_softmax(-1)
    row_valid = rank_valid.any(-1).float()
    ranking = (-(target*logp).sum(-1)*row_valid).sum()/row_valid.sum().clamp(min=1)
    proposal, coordinate_metrics = coordinate_loss(candidates, output['clean_history'], batch, cfg)
    clean_error = F.smooth_l1_loss(output['clean_history'].float(), batch['clean_local'], reduction='none').mean(-1)
    supplied_history = torch.cat([batch['hmask'].new_ones((len(candidates), 1)), batch['hmask'][:, :cfg.clean_points]], 1)
    clean_mask = batch['clean_mask'] * supplied_history
    clean = (clean_error * clean_mask).sum() / clean_mask.sum().clamp(min=1)
    with torch.no_grad():
        chosen = output['ranks'][:, :M].argmax(-1)
        b = torch.arange(len(chosen), device=chosen.device)
        eligible = (batch['dense_mask'].sum(-1) > 0).float()*(1-batch['offtrack'])
        denominator = eligible.sum().clamp(min=1)
        oracle_error = error[:, :M].min(-1).values
        selected_error = error[b, chosen]
        # At the normal maximum commit horizon: defaults to 8 voxels.
        horizon = min(3, cfg.n_future-1)
        known = mask[:, :M, horizon].amin(-1)*eligible
        recall = (labels[:, :M, horizon].amax(-1)*known).sum()/known.sum().clamp(min=1)
        metrics = dict(proposal=proposal.item(), ranking=ranking.item(), confidence=confidence.item(),
                       oracle_error=(oracle_error*eligible).sum().item()/denominator.item(),
                       selected_error=(selected_error*eligible).sum().item()/denominator.item(),
                       oracle_recall=recall.item())
        metrics.update(coordinate_metrics)
        # How informative the ranking target is: entropy relative to a uniform
        # distribution over the valid candidates (1 = uninformative).
        n_valid = rank_valid.sum(-1).clamp(min=1)
        entropy = -(target*target.clamp(min=1e-12).log()).sum(-1)/n_valid.log().clamp(min=1e-6)
        metrics['rank_target_entropy'] = ((entropy*row_valid).sum()/row_valid.sum().clamp(min=1)).item()
        # Stop-gate calibration for the rank-selected proposal: false stops
        # (gate closed on a correct prefix) and false continues (gate open on a
        # wrong prefix), at the first point and the default commit horizon.
        conf = output['confidence'][b, chosen].float()
        for point, name in ((0, 'first'), (horizon, 'commit')):
            known = mask[b, chosen, point]*eligible
            positive = labels[b, chosen, point]*known
            negative = (1-labels[b, chosen, point])*known
            for thr in (0.3, 0.5, 0.7):
                closed = (conf[:, point] < thr).float()
                metrics[f'gate{thr:.1f}_{name}_false_stop'] = ((closed*positive).sum()/positive.sum().clamp(min=1)).item()
                metrics[f'gate{thr:.1f}_{name}_false_go'] = (((1-closed)*negative).sum()/negative.sum().clamp(min=1)).item()
            metrics[f'gate_{name}_negatives'] = (negative.sum()/known.sum().clamp(min=1)).item()
        current_mask = batch['clean_mask'][:, 0]
        current_error = (output['clean_history'][:, 0] - batch['clean_local'][:, 0]).norm(dim=-1)
        metrics['clean_loss'] = clean.item()
        metrics['clean_current_error'] = (current_error*current_mask).sum().item()/current_mask.sum().clamp(min=1).item()
        from vesuvius.neural_tracing.fiber_follow.history_metrics import cleaning_measurements, summarize_cleaning
        metrics.update(summarize_cleaning(cleaning_measurements(
            output['clean_history'], batch['hist'], batch['hmask'], batch['clean_local'],
            batch['clean_mask'], cfg.clean_tangent_points)))
        first_step = output['candidates'][:, :M, 0]
        metrics['candidate_first_step_max'] = first_step.norm(dim=-1).max().item()
        first_step = first_step[b, chosen]
        metrics['first_step_length'] = first_step.norm(dim=-1).mean().item()
        metrics['first_step_length_max'] = first_step.norm(dim=-1).max().item()
        metrics['candidate_endpoint_spread'] = output['candidates'][:, :M, -1, :2].std(dim=1, unbiased=False).norm(dim=-1).mean().item()
        first_known = batch['plane_mask'][:, 0]
        first_error = (first_step[:, :2] - batch['plane_ab'][:, 0]).norm(dim=-1)
        metrics['first_plane_error'] = (first_error*first_known).sum().item()/first_known.sum().clamp(min=1).item()
        truth = torch.cat([batch['clean_local'][:, :1, :2], batch['plane_ab']], 1)
        supplied = torch.cat([batch['clean_mask'][:, :1], batch['plane_mask']], 1)
        step_known = supplied[:, 1:]*supplied[:, :-1]*(1-batch['offtrack'])[:, None]
        outside_step = (truth[:, 1:]-truth[:, :-1]).norm(dim=-1) > cfg.max_lateral_slope*cfg.future_step
        metrics['target_step_limit_fraction'] = (outside_step*step_known).sum().item()/step_known.sum().clamp_min(1).item()
        # Report observability separately from annotation validity. Leaving the
        # crop is not a physical fiber endpoint and never removes dense GT.
        lateral = batch['dense_ab'].abs().amax(-1)
        known = batch['dense_mask']
        crop_half = (cfg.width-1)*cfg.spacing/2
        for name, boundary in [('target_crop_oob', crop_half), ('target_crop_edge', crop_half-3)]:
            metrics[name] = ((lateral > boundary)*known).sum().item()/known.sum().clamp(min=1).item()
    return proposal+rank_weight*ranking+confidence_weight*confidence+clean_weight*clean, metrics
