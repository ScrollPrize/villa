"""Re-ranker over beam candidate paths, built on the fiber_follow spatial encoder."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from vesuvius.neural_tracing.fiber_follow.model import FollowNet, FollowNetConfig

ARCHITECTURE = 'beam_rerank_v2'


@dataclass
class BeamNetConfig(FollowNetConfig):
    heatmap_target: str = 'tube'
    tube_sigma: float = .35


class BeamRankNet(FollowNet):
    """Scores every candidate polyline of a pool from one oriented crop.

    Reuses ``FollowNet``'s UNet encoder, history conditioning and ranking /
    prefix heads, and owns an optional dense tube prediction head. Candidates are supplied by the beam, so the
    joint flow generator is unused. Each candidate token additionally sees
    the hand loss relative to the pool's best and its point validity.
    """

    def __init__(self, cfg: BeamNetConfig):
        super().__init__(cfg, create_flow=False)
        self.heat_head = nn.Conv3d(cfg.widths[0], 1, 1)
        del self.history_tokens
        del self.history_fusion
        del self.suffix_context
        del self.continuation_fusion
        w0 = cfg.widths[0]
        self.path_net = nn.Sequential(nn.Conv1d(w0 * 5 + 6 + 2, cfg.hidden, 3, padding=1), nn.SiLU(),
                                      nn.Conv1d(cfg.hidden, cfg.hidden, 3, padding=1), nn.SiLU())
        self.onfiber_head = nn.Linear(cfg.hidden, 1)

    def score_paths(self, features, candidates, point_mask, hand_rel):
        B, P, K, _ = candidates.shape
        grid = self.sampling_grid(candidates[..., None, :] + self.stencil)
        sampled = F.grid_sample(features.float(), grid.float(), align_corners=True, padding_mode='zeros')
        sampled = sampled.permute(0, 2, 1, 4, 3).reshape(B * P, -1, K)
        delta = torch.cat([candidates[:, :, 1:2] - candidates[:, :, :1], candidates[:, :, 1:] - candidates[:, :, :-1]], 2)
        geom = torch.cat([candidates / 32, delta / 4], -1).reshape(B * P, K, 6).transpose(1, 2)
        extra = torch.stack([hand_rel[..., None].expand(-1, -1, K) / 4, point_mask], -1).reshape(B * P, K, 2).transpose(1, 2)
        mask = point_mask.reshape(B * P, 1, K)
        tokens = self.path_net(torch.cat([sampled, geom.to(sampled.dtype), extra.to(sampled.dtype)], 1)) * mask
        pooled = tokens.sum(-1) / mask.sum(-1).clamp(min=1)
        ranks = self.rank_head(pooled).reshape(B, P)
        onfiber = self.onfiber_head(pooled).reshape(B, P)
        prefix, _ = self.prefix_context(tokens.transpose(1, 2).contiguous())
        prefix_logits = self.confidence_head(prefix.transpose(1, 2)).reshape(B, P, K)
        return ranks, onfiber, prefix_logits

    def forward(self, x, hist, hmask, candidates, point_mask, hand_rel):
        features, _ = self.encode(x, hist, hmask)
        out = {}
        if self.cfg.heatmap_target == 'tube':
            out['tube_logits'] = self.heat_head(features).float()[:, 0]
        ranks, onfiber, prefix = self.score_paths(features, candidates, point_mask, hand_rel)
        out.update(ranks=ranks.float(), onfiber_logits=onfiber.float(), prefix_logits=prefix.float())
        return out
