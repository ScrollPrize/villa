"""Causal segment survival: hazard logits to prefix confidence, and the first-failure loss."""
import torch
import torch.nn.functional as F


def survival_predictions(hazard_logits):
    """Return prefix logits/probabilities while keeping hazards explicit for loss.

    FP32 log-space accumulation avoids repeated low-precision products. Legacy
    diagnostics consume prefix logits, never the conditional hazard logits.
    """
    log_survival = F.logsigmoid(-hazard_logits.float()).cumsum(-1)
    # exp(log_survival) may round to exactly one for extreme negative hazards.
    # Keep the logit conversion finite without clipping the training hazards.
    safe = log_survival.clamp_max(-torch.finfo(log_survival.dtype).tiny)
    logits = log_survival-torch.log(-torch.expm1(safe))
    return logits, log_survival.exp()


def survival_loss(hazard_logits, prefix_labels, prefix_known):
    """First-failure negative log likelihood, or right-censored survival.

    Prefix labels establish survival through an endpoint, not independent
    point correctness. Once a prefix fails or becomes unknown, subsequent
    conditional hazards have no target. A later observed failure after a gap
    does not identify the interval of first failure; conservatively censor it.
    The likelihood sums intervals, then the caller averages states/candidates.
    """
    survived = prefix_known.bool() & (prefix_labels > .5)
    at_risk = torch.cat((torch.ones_like(survived[..., :1]),
                         survived.long().cumprod(-1)[..., :-1].bool()), -1)
    valid = at_risk & prefix_known.bool()
    failed = 1-prefix_labels.float()
    bce = F.binary_cross_entropy_with_logits(hazard_logits.float(), failed, reduction='none')
    return torch.where(valid, bce, 0.).sum(-1), valid
