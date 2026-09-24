"""History-conditioned spatial proposal network and supervised path scorer."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from vesuvius.neural_tracing.fiber_follow.geometry import CropSpec, crop_local_grid

ARCHITECTURE = 'spatial_candidates_v3'


@dataclass
class FollowNetConfig:
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
    heat_bins: int = 61
    heat_spacing: float = 1.0
    n_candidates: int = 4
    peaks_per_plane: int = 8
    candidate_separation: float = 1.5
    curvature_penalty: float = 0.15
    norm: str = 'batch'
    heatmap_target: str = 'planes'
    tube_sigma: float = .7

    def to_dict(self):
        return asdict(self)


def prepare_model(model, device):
    """Use CUDA's efficient 3D convolution layout without changing precision."""
    model = model.to(device)
    if torch.device(device).type == 'cuda':
        nn.utils.convert_conv3d_weight_memory_format(model, torch.channels_last_3d)
    return model


def block(cin, cout, norm, stride=1):
    def normalization():
        if norm == 'batch':
            return nn.BatchNorm3d(cout)
        if norm == 'group':
            return nn.GroupNorm(math.gcd(8, cout), cout)
        raise ValueError('norm must be batch or group')
    return nn.Sequential(nn.Conv3d(cin, cout, 3, stride=stride, padding=1, bias=False),
                         normalization(), nn.SiLU(), nn.Conv3d(cout, cout, 3, padding=1, bias=False),
                         normalization(), nn.SiLU())


def diverse_topk(score, paths, count, separation):
    """Batched path-space suppression. Duplicate slots are allowed if fewer modes exist."""
    work = score.clone()
    selected = []
    batch = torch.arange(score.shape[0], device=score.device)
    for _ in range(count):
        idx = work.argmax(-1)
        selected.append(idx)
        chosen = paths[batch, idx]
        distance = (paths - chosen[:, None]).square().sum(-1).mean(-1)
        work = work.masked_fill(distance < separation**2, -1e9)
    return torch.stack(selected, 1)


@torch.no_grad()
def decode_candidates(logits, cfg):
    """Connect heatmap modes into coherent alternative paths; no rollout beam search."""
    B, K, nb, _ = logits.shape
    peaks = min(cfg.peaks_per_plane, nb * nb)
    lg = logits.float().reshape(B * K, 1, nb, nb)
    maxima = lg == F.max_pool2d(lg, 3, stride=1, padding=1)
    values, idx = lg.masked_fill(~maxima, -1e6).flatten(1).topk(peaks, -1)
    v, u = idx // nb, idx % nb
    # Refine each peak without averaging disconnected modes.
    offsets = torch.arange(-1, 2, device=lg.device)
    vv = (v[..., None] + offsets).clamp(0, nb-1)
    uu = (u[..., None] + offsets).clamp(0, nb-1)
    patch = lg[:, 0][torch.arange(B*K, device=lg.device)[:, None, None, None],
                     vv[..., :, None], uu[..., None, :]]
    weights = patch.flatten(-2).softmax(-1).reshape_as(patch)
    uv = torch.stack([(weights.sum(-2)*uu).sum(-1), (weights.sum(-1)*vv).sum(-1)], -1)
    uv = (uv.reshape(B, K, peaks, 2) - (nb-1)/2) * cfg.heat_spacing
    values = values.reshape(B, K, peaks) - logits.float().flatten(2).logsumexp(-1)[..., None]
    paths = uv[:, 0, :, None, :]
    score = values[:, 0] - .025 * uv[:, 0].square().sum(-1)
    batch = torch.arange(B, device=lg.device)[:, None]
    keep = diverse_topk(score, paths, cfg.n_candidates, cfg.candidate_separation)
    paths, score = paths[batch, keep], score[batch, keep]
    for k in range(1, K):
        last = paths[:, :, -1]
        previous = paths[:, :, -2] if k > 1 else last
        expected = last + (last - previous)
        penalty = (uv[:, k, None] - expected[:, :, None]).square().sum(-1)
        child_score = (score[..., None] + values[:, k, None] - cfg.curvature_penalty * penalty).flatten(1)
        children = torch.cat([paths[:, :, None].expand(-1, -1, peaks, -1, -1),
                              uv[:, k, None, :, None].expand(-1, paths.shape[1], -1, -1, -1)], -2)
        children = children.flatten(1, 2)
        keep = diverse_topk(child_score, children, cfg.n_candidates, cfg.candidate_separation)
        paths, score = children[batch, keep], child_score[batch, keep]
    forward = cfg.future_step * torch.arange(1, K+1, device=lg.device)
    return torch.cat([paths, forward[None, None, :, None].expand(B, cfg.n_candidates, -1, 1)], -1)


class FollowNet(nn.Module):
    def __init__(self, cfg: FollowNetConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.heatmap_target not in ('planes', 'tube') or cfg.tube_sigma <= 0:
            raise ValueError('Invalid heatmap target or tube width')
        if cfg.n_future < 2 or cfg.n_candidates < 1 or cfg.hist_points < 1:
            raise ValueError('At least two future planes, one candidate, and one history point are required')
        if cfg.n_candidates > cfg.peaks_per_plane or cfg.n_future * cfg.future_step > (cfg.depth-cfg.behind-1)*cfg.spacing:
            raise ValueError('Candidates/forecast horizon exceed the proposal configuration')
        self.crop = CropSpec(depth=cfg.depth, width=cfg.width, behind=cfg.behind,
                             spacing=cfg.spacing)
        if cfg.heat_bins < 3 or cfg.heat_spacing <= 0 or (cfg.heat_bins-1)*cfg.heat_spacing > (cfg.width-1)*cfg.spacing:
            raise ValueError('Heatmap lateral support must fit inside the crop')
        grid = torch.from_numpy(crop_local_grid(self.crop)).float()
        self.register_buffer('coordinates', grid.permute(3, 0, 1, 2)[None] / 32, persistent=False)
        w = cfg.widths
        self.encoders = nn.ModuleList([block(cfg.in_channels+3, w[0], cfg.norm)] +
                                     [block(a, b, cfg.norm, 2) for a, b in zip(w[:-1], w[1:])])
        self.decoders = nn.ModuleList([block(w[i+1]+w[i], w[i], cfg.norm) for i in range(len(w)-2, -1, -1)])
        self.history = nn.Sequential(nn.Linear(cfg.hist_points*4, cfg.hidden), nn.SiLU())
        self.condition = nn.ModuleList([nn.Linear(cfg.hidden, 2*c) for c in w])
        self.heat_head = nn.Conv3d(w[0], 1, 1)
        self.path_net = nn.Sequential(nn.Conv1d(w[0]*5+6, cfg.hidden, 3, padding=1), nn.SiLU(),
                                      nn.Conv1d(cfg.hidden, cfg.hidden, 3, padding=1), nn.SiLU())
        self.rank_head = nn.Linear(cfg.hidden, 1)
        # Prefix labels depend on every earlier candidate point. Carry that
        # information forward explicitly; local convolutions alone cannot do it.
        self.prefix_context = nn.GRU(cfg.hidden, cfg.hidden, batch_first=True)
        self.confidence_head = nn.Conv1d(cfg.hidden, 1, 1)
        side = (torch.arange(cfg.heat_bins)-(cfg.heat_bins-1)/2)*cfg.heat_spacing
        forward = cfg.future_step*torch.arange(1, cfg.n_future+1)
        c, v, u = torch.meshgrid(forward, side, side, indexing='ij')
        self.register_buffer('plane_grid', torch.stack([u, v, c], -1), persistent=False)
        self.register_buffer('stencil', torch.tensor([[0.,0,0], [1.,0,0], [-1.,0,0], [0.,1,0], [0.,-1,0]])*cfg.spacing, persistent=False)

    def sampling_grid(self, local):
        cfg = self.cfg
        half = (cfg.width-1)*cfg.spacing/2
        return torch.stack([local[..., 0]/half, local[..., 1]/half,
                            2*(local[..., 2]/cfg.spacing+cfg.behind)/(cfg.depth-1)-1], -1)

    def encode(self, x, hist, hmask):
        cfg = self.cfg
        if self.encoders[0][0].weight.is_contiguous(memory_format=torch.channels_last_3d):
            x = x.contiguous(memory_format=torch.channels_last_3d)
        h = hist[:, cfg.hist_stride-1::cfg.hist_stride][:, :cfg.hist_points]
        m = hmask[:, cfg.hist_stride-1::cfg.hist_stride][:, :cfg.hist_points]
        if h.shape[1] != cfg.hist_points:
            raise ValueError('History must cover hist_points * hist_stride')
        context = self.history(torch.cat([h*m[..., None]/32, m[..., None]], -1).flatten(1).to(x.dtype))
        features = torch.cat([x, self.coordinates.expand(len(x), -1, -1, -1, -1).to(x.dtype)], 1)
        skips = []
        for encoder, condition in zip(self.encoders, self.condition):
            features = encoder(features)
            gain, bias = condition(context).chunk(2, -1)
            features = features * (1 + .1*gain[..., None, None, None]) + bias[..., None, None, None]
            skips.append(features)
        for decoder, skip in zip(self.decoders, reversed(skips[:-1])):
            features = decoder(torch.cat([F.interpolate(features, size=skip.shape[-3:], mode='trilinear', align_corners=True), skip], 1))
        return features

    def score_candidates(self, features, candidates):
        B, M, K, _ = candidates.shape
        grid = self.sampling_grid(candidates[..., None, :] + self.stencil)
        sampled = F.grid_sample(features.float(), grid.float(), align_corners=True, padding_mode='zeros')
        # B,C,M,K,5 -> B*M,C*5,K
        sampled = sampled.permute(0, 2, 1, 4, 3).reshape(B*M, -1, K)
        delta = torch.cat([candidates[:, :, :1], candidates[:, :, 1:]-candidates[:, :, :-1]], 2)
        geom = torch.cat([candidates/32, delta/4], -1).reshape(B*M, K, 6).transpose(1, 2)
        tokens = self.path_net(torch.cat([sampled, geom], 1))
        ranks = self.rank_head(tokens.mean(-1)).reshape(B, M)
        prefix, _ = self.prefix_context(tokens.transpose(1, 2).contiguous())
        confidence_logits = self.confidence_head(prefix.transpose(1, 2)).reshape(B, M, K)
        return ranks, confidence_logits

    def forward(self, x, hist, hmask, extra_candidates=None):
        features = self.encode(x, hist, hmask)
        plane_grid = self.sampling_grid(self.plane_grid)[None].expand(len(x), -1, -1, -1, -1)
        tube = {}
        if self.cfg.heatmap_target == 'tube':
            tube_logits = self.heat_head(features).float()
            tube['tube_logits'] = tube_logits[:, 0]
            logits = F.grid_sample(tube_logits, plane_grid, align_corners=True).squeeze(1)
        else:
            planes = F.grid_sample(features.float(), plane_grid, align_corners=True)
            logits = self.heat_head(planes).squeeze(1)
        candidates = decode_candidates(logits, self.cfg)
        if extra_candidates is not None:
            candidates = torch.cat([candidates, extra_candidates.detach()], 1)
        ranks, confidence_logits = self.score_candidates(features, candidates)
        chosen = ranks.argmax(-1)
        return dict(**tube, points=candidates[torch.arange(len(x), device=x.device), chosen], candidates=candidates,
                    heatmap=logits, ranks=ranks, confidence_logits=confidence_logits,
                    confidence=confidence_logits.float().sigmoid().cummin(-1).values)
