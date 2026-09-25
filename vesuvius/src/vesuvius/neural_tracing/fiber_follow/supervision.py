"""All flow, ranking and continuation labels come from dense controlled GT."""
import torch
import torch.nn.functional as F
from vesuvius.neural_tracing.fiber_follow.policy import (
    DEFAULT_CONFIDENCE, DEFAULT_MAX_RECOVERY_DISTANCE, choose_candidate, recovery_allowed,
)


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


def candidate_labels(candidates, batch, tolerance=1.5,
                     max_recovery_distance=DEFAULT_MAX_RECOVERY_DISTANCE):
    """Prefix agreement sampled every 0.5 forward voxel (with default config).

    A prefix is positive only when every sampled crossing is annotated and
    within tolerance. A known failure is negative even if later GT is missing.
    Unknown ends are censored. Tagged physical endpoints and already departed
    states supply explicit negatives. The origin-to-first-point connection must
    obey the same recovery distance limit as tracing. Within that limit, the
    first point may recover from a displaced origin; GT agreement begins there.
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
    bad_recovery = ~recovery_allowed(candidates, max_recovery_distance)
    failed = ((known & (error > tolerance)) | beyond_end
              | batch['offtrack'][:, None, None].bool() | bad_recovery[..., None])
    failure_prefix = failed.cumsum(-1) > 0
    known_prefix = (known | beyond_end).int().cummin(-1).values.bool()
    indices = torch.linspace(0, Q-1, K, device=candidates.device).round().long()
    target = (~failure_prefix[..., indices]).float()
    mask = (failure_prefix | known_prefix)[..., indices].float()
    valid_error = torch.where(known, error.nan_to_num(nan=8).clamp(max=8), 0.).sum(-1)/known.sum(-1).clamp(min=1)
    quality = (target*mask).sum(-1)/mask.sum(-1).clamp(min=1) - .1*valid_error
    return target, mask, quality, valid_error


def loss_fn(output, batch, cfg, tolerance=1.5, rank_weight=1., confidence_weight=1., flow_weight=1.,
            rank_temperature=20., confidence_threshold=DEFAULT_CONFIDENCE, n_commit=4, *, compute_metrics=True):
    """Training loss, optionally with diagnostic scalars for logging."""
    candidates = output['candidates']
    labels, mask, quality, error = candidate_labels(candidates.detach(), batch, tolerance, cfg.max_recovery_distance)
    M = cfg.flow_samples
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
    # Future coordinates are trained by flow matching inside the model forward;
    # the sampled candidates the scorer sees carry no gradient into the flow.
    flow = output['flow_loss']
    total = flow_weight*flow+rank_weight*ranking+confidence_weight*confidence
    if not compute_metrics:
        return total, {}
    with torch.no_grad():
        chosen, commit, allowed = choose_candidate(
            candidates[:, :M], output['ranks'][:, :M], output['confidence'][:, :M],
            confidence_threshold, n_commit, cfg.max_recovery_distance)
        b = torch.arange(len(chosen), device=chosen.device)
        eligible = (batch['dense_mask'].sum(-1) > 0).float()*(1-batch['offtrack'])
        denominator = eligible.sum().clamp(min=1)
        oracle_error = error[:, :M].min(-1).values
        selected_error = error[b, chosen]
        horizon = min(n_commit, cfg.n_future)-1
        positive = (labels[:, :M, horizon]*mask[:, :M, horizon]).amax(-1)
        known = ((positive > 0) | mask[:, :M, horizon].bool().all(-1)).float()*eligible
        recall = (positive*known).sum()/known.sum().clamp(min=1)
        mean_oracle_error = (oracle_error*eligible).sum()/denominator
        accepted = (commit > 0).float()*eligible
        metrics = dict(flow=flow.item(),
                       flow_known_fraction=output['flow_known_fraction'].item(),
                       flow_censored_fraction=output['flow_censored_fraction'].item(),
                       ranking=ranking.item(), confidence=confidence.item(),
                       oracle_error=mean_oracle_error.item(),
                       selected_error=(selected_error*eligible).sum().item()/denominator.item(),
                       oracle_recall=recall.item(),
                       oracle_recall_known_fraction=(known.sum()/denominator).item(),
                       selected_accept_fraction=(accepted.sum()/denominator).item(),
                       accepted_selected_error=((selected_error*accepted).sum()/accepted.sum().clamp(min=1)).item(),
                       selected_commit=commit.float().mean().item(),
                       candidate_support=output['candidate_support'][:, :M].float().mean().item(),
                       selected_support=output['candidate_support'].float()[b, chosen].mean().item())
        # How informative the ranking target is: entropy relative to a uniform
        # distribution over the valid candidates (1 = uninformative).
        n_valid = rank_valid.sum(-1).clamp(min=1)
        entropy = -(target*target.clamp(min=1e-12).log()).sum(-1)/n_valid.log().clamp(min=1e-6)
        metrics['rank_target_entropy'] = ((entropy*row_valid).sum()/row_valid.sum().clamp(min=1)).item()
        # Re-select at each threshold: changing the gate can change which
        # candidate the tracer chooses, not just whether that candidate stops.
        for thr in (0.3, 0.5, 0.7):
            gate_chosen, _, _ = choose_candidate(
                candidates[:, :M], output['ranks'][:, :M], output['confidence'][:, :M],
                thr, n_commit, cfg.max_recovery_distance)
            conf = output['confidence'][b, gate_chosen].float().cummin(-1).values
            for point, name in ((0, 'first'), (horizon, 'commit')):
                known = mask[b, gate_chosen, point]*eligible
                positive = labels[b, gate_chosen, point]*known
                negative = (1-labels[b, gate_chosen, point])*known
                closed = ((conf[:, point] < thr) | ~allowed[b, gate_chosen]).float()
                metrics[f'gate{thr:.1f}_{name}_false_stop'] = ((closed*positive).sum()/positive.sum().clamp(min=1)).item()
                metrics[f'gate{thr:.1f}_{name}_false_go'] = (((1-closed)*negative).sum()/negative.sum().clamp(min=1)).item()
        for point, name in ((0, 'first'), (horizon, 'commit')):
            known = mask[b, chosen, point]*eligible
            negative = (1-labels[b, chosen, point])*known
            metrics[f'gate_{name}_negatives'] = (negative.sum()/known.sum().clamp(min=1)).item()
        from vesuvius.neural_tracing.fiber_follow.history_metrics import observed_measurements, summarize_history
        metrics.update(summarize_history(observed_measurements(
            batch['hist'], batch['hmask'], batch['gt_history'], batch['gt_history_mask'])))
        first_step = output['candidates'][:, :M, 0]
        metrics['candidate_first_step_max'] = first_step.norm(dim=-1).max().item()
        metrics['candidate_recovery_reject_fraction'] = (~allowed).float().mean().item()
        metrics['recovery_blocked_fraction'] = (~allowed.any(-1)).float().mean().item()
        first_step = first_step[b, chosen]
        metrics['first_step_length'] = first_step.norm(dim=-1).mean().item()
        metrics['first_step_length_max'] = first_step.norm(dim=-1).max().item()
        metrics['candidate_endpoint_spread'] = output['candidates'][:, :M, -1, :2].std(dim=1, unbiased=False).norm(dim=-1).mean().item()
        first_known = batch['plane_mask'][:, 0]
        gt_first_length = (batch['plane_ab'][:, 0].square().sum(-1) + cfg.future_step**2).sqrt()
        metrics['target_recovery_reject_fraction'] = (
            ((gt_first_length > cfg.max_recovery_distance)*first_known).sum()/first_known.sum().clamp(min=1)).item()
        first_error = (first_step[:, :2] - batch['plane_ab'][:, 0]).norm(dim=-1)
        metrics['first_plane_error'] = (first_error*first_known).sum().item()/first_known.sum().clamp(min=1).item()
        # Report observability separately from annotation validity. Leaving the
        # crop is not a physical fiber endpoint and never removes dense GT.
        lateral = batch['dense_ab'].abs().amax(-1)
        known = batch['dense_mask']
        crop_half = (cfg.width-1)*cfg.spacing/2
        for name, boundary in [('target_crop_oob', crop_half), ('target_crop_edge', crop_half-3)]:
            metrics[name] = ((lateral > boundary)*known).sum().item()/known.sum().clamp(min=1).item()
    return total, metrics
