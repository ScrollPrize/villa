"""Train the same step score used for pruning; censor unknown annotation ends."""
import torch
import torch.nn.functional as F


def beam_loss(output, batch, rank_weight=1., onfiber_weight=1.):
    logits = output['onfiber_logits'].float()
    labeled = (batch['label_mask'] > 0) & (batch['cand_mask'] > 0)
    zero = logits.sum()*0
    bce = F.binary_cross_entropy_with_logits(logits, batch['onfiber'].float(), reduction='none')
    onfiber = (bce*labeled).sum()/labeled.sum().clamp_min(1)
    # Select rows BEFORE softmax: an entirely censored row must never generate NaN.
    rows = labeled.sum(-1) >= 2
    ranking = zero
    if rows.any():
        mask = labeled[rows]
        logp = logits[rows].masked_fill(~mask, -torch.inf).log_softmax(-1).masked_fill(~mask, 0)
        target = (5*batch['quality'][rows]).masked_fill(~mask, -torch.inf).softmax(-1)
        ranking = -(target*logp).sum(-1).mean()
    total = rank_weight*ranking + onfiber_weight*onfiber
    with torch.no_grad():
        # Detached tensors: callers convert them only when logging, so training
        # steps do not synchronize with the GPU.
        eligible = labeled.any(-1)
        n_eligible = eligible.sum().clamp_min(1)
        choice = logits.masked_fill(~labeled, -torch.inf).argmax(-1)
        chosen = batch['onfiber'].gather(1, choice[:, None])[:, 0]
        metrics = dict(ranking=ranking.detach(), onfiber=onfiber.detach(),
                       model_top1_onfiber=(chosen*eligible).sum()/n_eligible,
                       oracle_onfiber=(batch['onfiber']*labeled).amax(-1).sum()/n_eligible,
                       labeled_candidates=labeled.sum(-1).float().mean())
    return total, metrics
