"""Causal proposed-segment scoring with unrestricted access to observations."""
import torch
from torch import nn
import torch.nn.functional as F

from vesuvius.neural_tracing.fiber_follow.models.model import PathDecoderLayer


class SegmentSurvivalScorer(nn.Module):
    # Proposals occupy fixed forward planes. Four samples per incoming
    # segment match the default labels' quarter-forward-voxel resolution.
    samples_per_segment = 4

    def __init__(self, cfg):
        super().__init__()
        h = cfg.hidden
        evidence_width = cfg.path_evidence_width
        # Ordered image samples, start/end/displacement/length.
        width = self.samples_per_segment*evidence_width+10
        self.query = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, h), nn.SiLU())
        self.layers = nn.ModuleList(PathDecoderLayer(h, cfg.heads, 1024, dropout=0.,
            activation='gelu', batch_first=True, norm_first=True) for _ in range(cfg.scorer_layers))
        from vesuvius.neural_tracing.fiber_follow.models.history_slabs import HistoryAttention
        self.history_attention = HistoryAttention(h, cfg.heads)
        self.norm = nn.LayerNorm(h)
        self.failure = nn.Linear(h, 1)

    def segment_samples(self, points):
        """Every location depends only on this endpoint and its predecessor."""
        start = torch.cat((torch.zeros_like(points[:, :1]), points[:, :-1]), 1)
        fraction = torch.arange(1, self.samples_per_segment+1, device=points.device,
                                dtype=points.dtype)/self.samples_per_segment
        return start[:, :, None]+fraction[None, None, :, None]*(points-start)[:, :, None]

    def project_memory(self, memory, padding):
        return [layer.project_memory(memory) for layer in self.layers]

    def forward(self, spatial, points, projected, padding, history):
        if len(history) == 2:
            history = self.history_attention.project_memory(*history)
        start = torch.cat((torch.zeros_like(points[:, :1]), points[:, :-1]), 1)
        delta = points-start
        geometry = torch.cat((start, points, delta, delta.norm(dim=-1, keepdim=True)), -1)/16.
        query = self.query(torch.cat((spatial.flatten(2), geometry), -1))
        # Only the proposed path is causal. Cross-attention reads all observed
        # deep/fine-plane/reference/memory tokens; no generator hidden state enters here.
        k = points.shape[1]
        causal = torch.ones(k, k, device=points.device, dtype=torch.bool).triu(1)
        for layer, kv in zip(self.layers, projected):
            query = layer.forward_cached(query, kv, padding, causal_mask=causal,
                                         history=history, history_attention=self.history_attention)
        # Preserve precision before thresholding/ranking proposals. Casting the
        # logits after a BF16 projection cannot recover its rounding loss.
        with torch.autocast(query.device.type, enabled=False):
            return self.failure(self.norm(query.float())).squeeze(-1)


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
