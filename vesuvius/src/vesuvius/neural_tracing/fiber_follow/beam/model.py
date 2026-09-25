"""Re-ranker over beam candidate paths, built on the fiber_follow spatial encoder."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from vesuvius.neural_tracing.fiber_follow.encoder import SpatialEncoder
from vesuvius.neural_tracing.fiber_follow.policy import DEFAULT_MAX_RECOVERY_DISTANCE

ARCHITECTURE = 'beam_rerank_v2'


@dataclass
class BeamEncoderConfig:
    in_channels: int = 8
    depth: int = 64
    width: int = 64
    behind: int = 16
    spacing: float = 1.0
    widths: tuple = (24, 48, 96)
    hidden: int = 96
    n_future: int = 16
    future_step: float = 2.0
    hist_points: int = 32
    hist_stride: int = 4
    recent_history_points: int = 8  # dense observed history used by the flow and scorer
    flow_layers: int = 4
    flow_heads: int = 4
    flow_steps: int = 4       # midpoint steps, two flow evaluations per step
    flow_samples: int = 16    # future-path samples drawn per state
    flow_draws: int = 16      # stratified (time, noise) draws against shared image features
    flow_sigma: tuple = ()   # fitted per-plane (a, b) residual std, trace-grid voxels
    flow_stencil_radius: float = 2.0  # 3x3 lateral patch pitch/radius, trace-grid voxels
    support_radius: float = 1.5  # RMS lateral distance on the commit window
    max_recovery_distance: float = DEFAULT_MAX_RECOVERY_DISTANCE  # origin to first point, trace-grid voxels
    norm: str = 'group'
    # Sample slot 0 is the ODE path integrated from y0 = 0 (the centre of the
    # noise) instead of a random draw, and the tracer commits that slot alone.
    # The other slots stay random draws so the support feature keeps its meaning.
    deterministic_first: bool = False

    def __post_init__(self):
        self.widths = tuple(self.widths)
        self.flow_sigma = tuple(tuple(row) for row in self.flow_sigma)
        if (not math.isfinite(self.max_recovery_distance) or self.max_recovery_distance <= 0
                or not 0 < self.future_step <= self.max_recovery_distance):
            raise ValueError('max_recovery_distance must be finite, positive, and at least future_step')

    def to_dict(self):
        return asdict(self)



@dataclass
class BeamNetConfig(BeamEncoderConfig):
    heatmap_target: str = 'tube'
    tube_sigma: float = .35


class BeamRankNet(SpatialEncoder):
    """Scores every candidate polyline of a pool from one oriented crop.

    Reuses the shared spatial encoder and owns its original ranking / prefix
    heads and optional dense tube prediction head. Candidates come from the beam. Each candidate token additionally sees
    the hand loss relative to the pool's best and its point validity.
    """

    def __init__(self, cfg: BeamNetConfig):
        super().__init__(cfg)
        self.heat_head = nn.Conv3d(cfg.widths[0], 1, 1)
        self.rank_head = nn.Linear(cfg.hidden, 1)
        self.prefix_context = nn.GRU(cfg.hidden, cfg.hidden, batch_first=True)
        self.confidence_head = nn.Conv1d(cfg.hidden, 1, 1)
        self.register_buffer('stencil', torch.tensor([[0.,0,0], [1.,0,0], [-1.,0,0], [0.,1,0], [0.,-1,0]])*cfg.spacing, persistent=False)
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
