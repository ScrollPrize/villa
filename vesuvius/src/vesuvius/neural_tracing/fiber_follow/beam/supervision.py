"""Losses for the beam re-ranker. All labels come from dense controlled GT."""
from __future__ import annotations

import torch
import torch.nn.functional as F

def tube_loss(logits, target, mask):
    """Gaussian-value regression, balancing the 3-sigma neighborhood and background.

    Squared error retains the Gaussian itself as the optimum. Separate regional
    means keep the sparse tube from being overwhelmed by empty crop voxels.
    """
    error = (logits.float().sigmoid()-target.float()).square()
    near = (target >= 0.0111089965).float()*mask  # exp(-3**2/2)
    far = (target < 0.0111089965).float()*mask
    total = logits.sum()*0
    for weights in (near, far):
        count = weights.flatten(1).sum(-1)
        loss = (error*weights).flatten(1).sum(-1)/count.clamp(min=1)
        total = total + .5*(loss*(count > 0)).sum()/(count > 0).sum().clamp(min=1)
    return total


def beam_loss(output, batch, k_back: int, rank_weight=1., onfiber_weight=1., prefix_weight=1., tube_weight=1.):
    valid = batch['cand_mask'] > 0
    labeled = (batch['label_mask'] > 0) & valid
    ranks = output['ranks'].float()
    B, P = ranks.shape
    zero = ranks.sum() * 0

    # Listwise ranking toward the soft quality distribution over labeled candidates.
    rows = labeled.sum(-1) >= 2
    masked = ranks.masked_fill(~labeled, float('-inf'))
    target = (5 * batch['quality'].float()).masked_fill(~labeled, float('-inf')).softmax(-1)
    logp = masked.log_softmax(-1).masked_fill(~labeled, 0)
    ranking = (-(target * logp).sum(-1) * rows).sum() / rows.sum().clamp(min=1) if rows.any() else zero

    onfiber = F.binary_cross_entropy_with_logits(output['onfiber_logits'].float(), batch['onfiber'].float(), reduction='none')
    onfiber = (onfiber * labeled).sum() / labeled.sum().clamp(min=1)

    prefix_logits = output['prefix_logits'].float()[:, :, k_back + 1:]
    prefix_mask = batch['prefix_mask'].float() * valid[..., None]
    prefix = F.binary_cross_entropy_with_logits(prefix_logits, batch['prefix_target'].float(), reduction='none')
    prefix = (prefix * prefix_mask).sum() / prefix_mask.sum().clamp(min=1)

    total = rank_weight * ranking + onfiber_weight * onfiber + prefix_weight * prefix
    metrics = dict(ranking=ranking.item(), onfiber=onfiber.item(), prefix=prefix.item())
    if tube_weight > 0 and 'tube_logits' in output and 'tube_target' in batch:
        tube = tube_loss(output['tube_logits'], batch['tube_target'], batch['tube_mask'])
        total = total + tube_weight * tube
        metrics['tube'] = tube.item()

    with torch.no_grad():
        on = batch['onfiber'].float()
        # Pools are sorted by hand loss, so index 0 is the hand tracer's choice.
        hand_rows = labeled[:, 0]
        model_choice = ranks.masked_fill(~valid, float('-inf')).argmax(-1)
        model_rows = labeled[torch.arange(B, device=ranks.device), model_choice]
        oracle_rows = labeled.any(-1)
        metrics.update(
            hand_top1_onfiber=(on[:, 0] * hand_rows).sum().item() / max(1, hand_rows.sum().item()),
            model_top1_onfiber=(on[torch.arange(B, device=ranks.device), model_choice] * model_rows).sum().item()
            / max(1, model_rows.sum().item()),
            oracle_onfiber=((on * labeled).amax(-1) * oracle_rows).sum().item() / max(1, oracle_rows.sum().item()),
            onfiber_accuracy=((((output['onfiber_logits'] > 0).float() == on).float() * labeled).sum()
                              / labeled.sum().clamp(min=1)).item(),
            offtrack_fraction=batch['offtrack'].float().mean().item(),
            labeled_candidates=labeled.float().sum(-1).mean().item(),
        )
    return total, metrics
