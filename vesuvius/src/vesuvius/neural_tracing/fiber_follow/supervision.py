"""All flow, ranking and continuation labels come from dense controlled GT."""
import torch
import torch.nn.functional as F


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


def loss_fn(output, batch, cfg, tolerance=1.5, rank_weight=1., confidence_weight=1., flow_weight=1.,
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
    # The joint polyline (anchor, cleaned past, future planes) is trained by
    # flow matching inside the model forward; the sampled candidates the
    # scorer sees carry no gradient back into the flow.
    flow = output['flow_loss']
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
        metrics = dict(flow=flow.item(), flow_past=output['flow_past'].item(), flow_future=output['flow_future'].item(),
                       flow_known_fraction=output['flow_known_fraction'].item(),
                       ranking=ranking.item(), confidence=confidence.item(),
                       oracle_error=(oracle_error*eligible).sum().item()/denominator.item(),
                       selected_error=(selected_error*eligible).sum().item()/denominator.item(),
                       oracle_recall=recall.item(),
                       candidate_support=output['candidate_support'].float().mean().item(),
                       selected_support=output['candidate_support'].float()[b, chosen.clamp(max=M-1)].mean().item())
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
        metrics['clean_current_error'] = (current_error*current_mask).sum().item()/current_mask.sum().clamp(min=1).item()
        from vesuvius.neural_tracing.fiber_follow.history_metrics import TANGENT_POINTS, cleaning_measurements, summarize_cleaning
        metrics.update(summarize_cleaning(cleaning_measurements(
            output['clean_history'], batch['hist'], batch['hmask'], batch['clean_local'],
            batch['clean_mask'], TANGENT_POINTS)))
        first_step = output['candidates'][:, :M, 0]
        metrics['candidate_first_step_max'] = first_step.norm(dim=-1).max().item()
        first_step = first_step[b, chosen]
        metrics['first_step_length'] = first_step.norm(dim=-1).mean().item()
        metrics['first_step_length_max'] = first_step.norm(dim=-1).max().item()
        metrics['candidate_endpoint_spread'] = output['candidates'][:, :M, -1, :2].std(dim=1, unbiased=False).norm(dim=-1).mean().item()
        first_known = batch['plane_mask'][:, 0]
        first_error = (first_step[:, :2] - batch['plane_ab'][:, 0]).norm(dim=-1)
        metrics['first_plane_error'] = (first_error*first_known).sum().item()/first_known.sum().clamp(min=1).item()
        # Report observability separately from annotation validity. Leaving the
        # crop is not a physical fiber endpoint and never removes dense GT.
        lateral = batch['dense_ab'].abs().amax(-1)
        known = batch['dense_mask']
        crop_half = (cfg.width-1)*cfg.spacing/2
        for name, boundary in [('target_crop_oob', crop_half), ('target_crop_edge', crop_half-3)]:
            metrics[name] = ((lateral > boundary)*known).sum().item()/known.sum().clamp(min=1).item()
    return flow_weight*flow+rank_weight*ranking+confidence_weight*confidence, metrics
