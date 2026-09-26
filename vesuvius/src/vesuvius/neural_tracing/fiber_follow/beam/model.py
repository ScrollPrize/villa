"""Learned step costs for the native VC3D beam, from one CT resolution."""
from dataclasses import asdict, dataclass

import torch
from torch import nn
import torch.nn.functional as F

from vesuvius.neural_tracing.fiber_follow.encoder import SpatialEncoder

ARCHITECTURE = 'beam_step_cost_v3'


@dataclass
class BeamNetConfig:
    in_channels: int = 2
    depth: int = 192
    width: int = 96
    behind: int = 128
    spacing: float = 1.
    widths: tuple = (24, 48, 96)
    hidden: int = 96
    hist_points: int = 32
    hist_stride: int = 4
    norm: str = 'group'
    score_chunk: int = 256

    def __post_init__(self):
        self.widths = tuple(self.widths)
        if self.score_chunk < 1:
            raise ValueError('score_chunk must be positive')

    def to_dict(self):
        return asdict(self)


class BeamRankNet(SpatialEncoder):
    """Encode a frontier crop once and score every proposed step.

    Each proposal supplies its OWN recent history, parent point and new endpoint.
    A global spatial summary gives the scorer access to the entire forward crop,
    beyond the receptive field of the features sampled along the proposed path.
    No hand costs, GT inputs, coordinate decoder or dense tube objective.
    """
    # Measured on an RTX 5090 (bf16, 192x96x96, batch 2): NCDHW beats
    # channels_last_3d eager (112 vs 127 ms/update) and compiled (77 vs 140 ms).
    cuda_memory_format = torch.contiguous_format

    def __init__(self, cfg: BeamNetConfig):
        super().__init__(cfg)
        self.register_buffer('stencil', torch.tensor(
            [[0.,0,0], [1.,0,0], [-1.,0,0], [0.,1,0], [0.,-1,0]]) * cfg.spacing,
            persistent=False)
        self.path_net = nn.Sequential(nn.Conv1d(cfg.widths[0]*5 + 7, cfg.hidden, 3, padding=1),
                                      nn.SiLU(), nn.Conv1d(cfg.hidden, cfg.hidden, 3, padding=1), nn.SiLU())
        self.path_context = nn.GRU(cfg.hidden, cfg.hidden, batch_first=True)
        self.spatial_context = nn.Sequential(nn.Flatten(), nn.Linear(cfg.widths[-1]*4*3*3, cfg.hidden), nn.SiLU())
        self.cost_head = nn.Sequential(nn.Linear(cfg.hidden*2, cfg.hidden), nn.SiLU(), nn.Linear(cfg.hidden, 1))

    def encode_scene(self, x, hist, hmask):
        features, _, deep = self.encode(x, hist, hmask, return_deep=True)
        return features, self.spatial_context(F.adaptive_avg_pool3d(deep, (4, 3, 3)))

    def score_paths(self, features, context, candidates, point_mask):
        B, P, K, _ = candidates.shape
        grid = self.sampling_grid(candidates[..., None, :] + self.stencil)
        sampled = F.grid_sample(features.float(), grid.float(), align_corners=True, padding_mode='zeros')
        sampled = sampled.permute(0, 2, 1, 4, 3).reshape(B*P, -1, K)
        delta = torch.zeros_like(candidates)
        delta[:, :, 1:] = candidates[:, :, 1:] - candidates[:, :, :-1]
        delta[:, :, 1:] *= (point_mask[:, :, 1:]*point_mask[:, :, :-1])[..., None]
        geom = torch.cat([candidates/128, delta, point_mask[..., None]], -1)
        tokens = self.path_net(torch.cat([sampled, geom.reshape(B*P, K, 7).transpose(1, 2)], 1))
        # Skip padded history, rather than allowing GRU biases to invent it.
        tokens = tokens.transpose(1, 2)
        mask = point_mask.reshape(B*P, K)
        h = tokens.new_zeros(1, B*P, self.cfg.hidden)
        for k in range(K):
            _, new = self.path_context(tokens[:, k:k+1].contiguous(), h)
            h = torch.where(mask[:, k][None, :, None] > 0, new, h)
        global_context = context[:, None].expand(-1, P, -1).reshape(B*P, -1)
        logits = self.cost_head(torch.cat([h[0], global_context], -1)).reshape(B, P).float()
        return logits

    def forward(self, x, hist, hmask, candidates, point_mask):
        features, context = self.encode_scene(x, hist, hmask)
        logits = torch.cat([self.score_paths(features, context, candidates[:, a:a+self.cfg.score_chunk],
                                             point_mask[:, a:a+self.cfg.score_chunk])
                            for a in range(0, candidates.shape[1], self.cfg.score_chunk)], 1)
        return dict(onfiber_logits=logits, ranks=logits, step_cost=F.softplus(-logits))
