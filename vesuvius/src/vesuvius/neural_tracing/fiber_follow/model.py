"""History-conditioned spatial proposal network and supervised path scorer."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from vesuvius.neural_tracing.fiber_follow.geometry import CropSpec, crop_local_grid

ARCHITECTURE = 'direct_paths_v7'


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
    clean_points: int = 8
    clean_tangent_points: int = 6  # includes the corrected current point
    n_candidates: int = 4
    max_history_correction: float = 4.0
    max_lateral_slope: float = 2.0
    path_sample_radius: float = 2.0
    norm: str = 'batch'

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


def history_tangent(points, mask, count):
    """Weighted line fit in arclength order, newest point first.

    Only the contiguous valid prefix is used. Nearer points get more weight;
    fewer than two distinct points supply no measured tangent.
    """
    count = min(count, points.shape[1])
    p = points[:, :count].float()
    valid = mask[:, :count].float().cumprod(-1)
    s = -torch.arange(count, device=p.device, dtype=p.dtype)
    w = valid / (1 + s.abs())
    p = torch.where(valid[..., None] > 0, p, 0.)
    total = w.sum(-1, keepdim=True).clamp_min(1)
    center = (w * s).sum(-1, keepdim=True) / total
    dp = p - (w[..., None] * p).sum(1, keepdim=True) / total[..., None]
    tangent = (w[..., None] * (s - center)[..., None] * dp).sum(1)
    measured = (valid.sum(-1) >= 2) & (tangent.norm(dim=-1) > 1e-6)
    forward = torch.zeros_like(tangent)
    forward[:, 2] = 1
    return torch.where(measured[:, None], F.normalize(tangent, dim=-1), forward), measured


def bounded_vector(value, limit):
    """Preserve small vectors exactly and cap their Euclidean magnitude."""
    return value / (value.norm(dim=-1, keepdim=True) / limit).clamp_min(1.)


class PathDecoder(nn.Module):
    """Differentiable, locally sampled paths rooted in the corrected history."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden
        self.mode_tokens = nn.Embedding(cfg.n_candidates, h)
        nn.init.normal_(self.mode_tokens.weight, std=.05)
        # Start with a central continuation and small directional alternatives.
        # These are learned parameters, not a fixed separation requirement.
        velocity = torch.zeros(cfg.n_candidates, 2)
        if cfg.n_candidates > 1:
            angles = torch.arange(cfg.n_candidates-1) * (2*math.pi/(cfg.n_candidates-1))
            velocity[1:] = .1*torch.stack([angles.cos(), angles.sin()], -1)
        self.mode_velocity = nn.Parameter(velocity)
        side = torch.tensor([-cfg.path_sample_radius, 0., cfg.path_sample_radius])
        v, u = torch.meshgrid(side, side, indexing='ij')
        self.register_buffer('stencil', torch.stack([u, v, torch.zeros_like(u)], -1).reshape(9, 3), persistent=False)
        self.cell = nn.GRUCell(cfg.widths[0]*9 + 6 + 2*h, h)
        self.velocity_head = nn.Linear(h, 2)
        nn.init.normal_(self.velocity_head.weight, std=.01)
        nn.init.zeros_(self.velocity_head.bias)

    def forward(self, features, context, cleaned, hmask, sampling_grid):
        cfg = self.cfg
        B, M = len(features), cfg.n_candidates
        valid = torch.cat([hmask.new_ones((B, 1)), hmask[:, :cfg.clean_points]], 1)
        tangent, measured = history_tangent(cleaned, valid, cfg.clean_tangent_points)
        slope = tangent[:, :2] / tangent[:, 2:].clamp_min(.25)
        slope = slope * (measured & (tangent[:, 2] > 0))[:, None]
        slope = bounded_vector(slope, cfg.max_lateral_slope)
        base = slope[:, None] + self.mode_velocity[None]
        velocity = bounded_vector(base, cfg.max_lateral_slope)
        history = context[0, :, None].expand(-1, M, -1)
        modes = self.mode_tokens.weight[None].expand(B, -1, -1)
        state = torch.tanh(history + modes).reshape(B*M, cfg.hidden)
        previous = cleaned[:, None, 0].expand(-1, M, -1)
        # Cast once: retaining a full float32 feature copy for each of 64
        # differentiable samples would multiply the activation memory.
        image_features = features.float()
        points = []
        for k in range(cfg.n_future):
            z = previous.new_full((B, M, 1), (k+1)*cfg.future_step)
            dz = z - previous[..., 2:]
            limit = cfg.max_lateral_slope * cfg.future_step
            expected = torch.cat([previous[..., :2] + bounded_vector(velocity*dz, limit), z], -1)
            grid = sampling_grid(expected[..., None, :] + self.stencil)[:, :, None]
            sampled = F.grid_sample(image_features, grid.float(), align_corners=True,
                                    padding_mode='zeros')[:, :, :, 0].permute(0, 2, 1, 3).flatten(2)
            phase = z / (cfg.n_future*cfg.future_step)
            tokens = torch.cat([sampled, expected/32, velocity/cfg.max_lateral_slope,
                                phase, history, modes], -1)
            state = self.cell(tokens.reshape(B*M, -1), state)
            velocity = bounded_vector(base + self.velocity_head(state).reshape(B, M, 2), cfg.max_lateral_slope)
            step = bounded_vector(velocity*dz, limit)
            previous = torch.cat([previous[..., :2] + step, z], -1)
            points.append(previous)
        return torch.stack(points, 2)


class FollowNet(nn.Module):
    def __init__(self, cfg: FollowNetConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.n_future < 2 or cfg.n_candidates < 1 or cfg.hist_points < 1:
            raise ValueError('At least two future planes, one candidate, and one history point are required')
        if cfg.clean_points < 1 or cfg.clean_points > cfg.hist_points * cfg.hist_stride:
            raise ValueError('clean_points must be positive and fit inside the supplied history')
        if cfg.clean_tangent_points < 2:
            raise ValueError('clean_tangent_points must be at least 2')
        if cfg.n_future * cfg.future_step > (cfg.depth-cfg.behind-1)*cfg.spacing:
            raise ValueError('Candidates/forecast horizon exceed the proposal configuration')
        self.crop = CropSpec(depth=cfg.depth, width=cfg.width, behind=cfg.behind,
                             spacing=cfg.spacing)
        if any(not math.isfinite(v) or v <= 0 for v in (cfg.max_history_correction, cfg.max_lateral_slope, cfg.path_sample_radius)):
            raise ValueError('History correction, lateral slope and sampling radius must be positive and finite')
        grid = torch.from_numpy(crop_local_grid(self.crop)).float()
        self.register_buffer('coordinates', grid.permute(3, 0, 1, 2)[None] / 32, persistent=False)
        w = cfg.widths
        self.encoders = nn.ModuleList([block(cfg.in_channels+3, w[0], cfg.norm)] +
                                     [block(a, b, cfg.norm, 2) for a, b in zip(w[:-1], w[1:])])
        self.decoders = nn.ModuleList([block(w[i+1]+w[i], w[i], cfg.norm) for i in range(len(w)-2, -1, -1)])
        self.history = nn.Sequential(nn.Linear(cfg.hist_points*4, cfg.hidden), nn.SiLU())
        self.clean_head = nn.Sequential(
            nn.Conv1d(w[0]+4, cfg.hidden, 3, padding=1), nn.SiLU(),
            nn.Conv1d(cfg.hidden, cfg.hidden, 3, padding=1), nn.SiLU(),
            nn.Conv1d(cfg.hidden, 3, 1))
        self.condition = nn.ModuleList([nn.Linear(cfg.hidden, 2*c) for c in w])
        self.proposal_decoder = PathDecoder(cfg)
        self.path_net = nn.Sequential(nn.Conv1d(w[0]*5+6, cfg.hidden, 3, padding=1), nn.SiLU(),
                                      nn.Conv1d(cfg.hidden, cfg.hidden, 3, padding=1), nn.SiLU())
        self.rank_head = nn.Linear(cfg.hidden, 1)
        # Prefix labels depend on every earlier candidate point. Carry that
        # information forward explicitly; local convolutions alone cannot do it.
        self.prefix_context = nn.GRU(cfg.hidden, cfg.hidden, batch_first=True)
        # Future evidence travels back to the first decision. Each candidate
        # has independent recurrent state, including synthetic training paths.
        self.suffix_context = nn.GRU(cfg.hidden, cfg.hidden, batch_first=True)
        self.continuation_fusion = nn.Sequential(nn.Linear(4*cfg.hidden, cfg.hidden), nn.SiLU())
        # Both paths remain visible: cleaning must not erase evidence of drift.
        self.history_tokens = nn.Sequential(nn.Linear(2*w[0]+10, cfg.hidden), nn.SiLU(),
                                            nn.Linear(cfg.hidden, cfg.hidden), nn.SiLU())
        self.history_fusion = nn.Sequential(nn.Linear(2*cfg.hidden, cfg.hidden), nn.SiLU())
        self.confidence_head = nn.Conv1d(cfg.hidden, 1, 1)
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

    def predict_clean_history(self, features, hist, hmask):
        count = self.cfg.clean_points
        observed = torch.cat([hist.new_zeros((len(hist), 1, 3)), hist[:, :count]], 1)
        valid = torch.cat([hmask.new_ones((len(hist), 1)), hmask[:, :count]], 1)
        observed = observed * valid[..., None]
        grid = self.sampling_grid(observed)[:, :, None, None]
        sampled = F.grid_sample(features.float(), grid.float(), align_corners=True,
                                padding_mode='zeros').squeeze(-1).squeeze(-1)
        tokens = torch.cat([sampled, (observed / 32).transpose(1, 2), valid[:, None]], 1)
        return observed + bounded_vector(self.clean_head(tokens).transpose(1, 2), self.cfg.max_history_correction)

    def encode_history(self, features, hist, hmask, clean_history):
        """Encode observed and cleaned history for both ranking and confidence.

        Read oldest to newest, skipping masked points without advancing the
        recurrent state. No future candidate or annotated history is supplied.
        """
        count = self.cfg.clean_points
        observed = torch.cat([hist.new_zeros((len(hist), 1, 3)), hist[:, :count]], 1)
        valid = torch.cat([hmask.new_ones((len(hist), 1)), hmask[:, :count]], 1)
        observed = torch.where(valid[..., None] > 0, observed, 0.)
        cleaned = torch.where(valid[..., None] > 0, clean_history, 0.)
        def sample(points):
            grid = self.sampling_grid(points)[:, :, None, None]
            return F.grid_sample(features.float(), grid.float(), align_corners=True,
                                 padding_mode='zeros').squeeze(-1).squeeze(-1).transpose(1, 2)
        tokens = self.history_tokens(torch.cat([sample(observed), sample(cleaned), observed/32,
                                               cleaned/32, (cleaned-observed)/4, valid[..., None]], -1))
        state = tokens.new_zeros((1, len(hist), self.cfg.hidden))
        for i in range(count, -1, -1):
            _, updated = self.prefix_context(tokens[:, i:i+1].contiguous(), state)
            state = torch.where(valid[None, :, i:i+1] > 0, updated, state)
        # Keep a short route from all supplied history, including its oldest
        # image observations, alongside the ordered recurrent representation.
        pooled = (tokens*valid[..., None]).sum(1)/valid.sum(1, keepdim=True).clamp_min(1)
        return self.history_fusion(torch.cat([state[0], pooled], -1))[None]

    def score_candidates(self, features, candidates, clean_history, hist, hmask):
        B, M, K, _ = candidates.shape
        grid = self.sampling_grid(candidates[..., None, :] + self.stencil)
        sampled = F.grid_sample(features.float(), grid.float(), align_corners=True, padding_mode='zeros')
        # B,C,M,K,5 -> B*M,C*5,K
        sampled = sampled.permute(0, 2, 1, 4, 3).reshape(B*M, -1, K)
        initial = candidates[:, :, :1] - clean_history[:, None, :1]
        delta = torch.cat([initial, candidates[:, :, 1:]-candidates[:, :, :-1]], 2)
        geom = torch.cat([candidates/32, delta/4], -1).reshape(B*M, K, 6).transpose(1, 2)
        tokens = self.path_net(torch.cat([sampled, geom], 1))
        context = self.encode_history(features, hist, hmask, clean_history)
        context = context[:, :, None].expand(-1, -1, M, -1).reshape(1, B*M, -1).contiguous()
        sequence = tokens.transpose(1, 2).contiguous()
        prefix, _ = self.prefix_context(sequence, context)
        suffix, _ = self.suffix_context(sequence.flip(1).contiguous())
        # Preserve explicit history at every future point, rather than relying
        # on the forward GRU to remember it for the entire horizon. The suffix
        # lets even the first prefix decision use the full proposed path. A
        # pooled future summary gives distant points a short gradient path;
        # they need not survive 64 recurrent updates to affect early decisions.
        future_summary = suffix.mean(1, keepdim=True).expand(-1, K, -1)
        joint = self.continuation_fusion(torch.cat([
            prefix, suffix.flip(1), future_summary,
            context[0, :, None].expand(-1, K, -1)], -1))
        ranks = self.rank_head(joint.mean(1)).reshape(B, M)
        confidence_logits = self.confidence_head(joint.transpose(1, 2)).reshape(B, M, K)
        return ranks, confidence_logits

    def forward(self, x, hist, hmask, extra_candidates=None):
        features = self.encode(x, hist, hmask)
        clean_history = self.predict_clean_history(features, hist, hmask)
        context = self.encode_history(features, hist, hmask, clean_history)
        candidates = self.proposal_decoder(features, context, clean_history, hmask, self.sampling_grid)
        if extra_candidates is not None:
            candidates = torch.cat([candidates, extra_candidates.detach()], 1)
        # Correctness targets are measured on these paths. Classification must
        # not move a path to make its own negative label easier to predict;
        # the direct coordinate objective trains the geometry instead.
        ranks, confidence_logits = self.score_candidates(features, candidates.detach(), clean_history, hist, hmask)
        chosen = ranks.argmax(-1)
        points = candidates[torch.arange(len(x), device=x.device), chosen]
        corrected_path = torch.cat([clean_history.flip(1), points], 1)
        return dict(clean_history=clean_history,
                    corrected_path=corrected_path, points=points, candidates=candidates,
                    ranks=ranks, confidence_logits=confidence_logits,
                    confidence=confidence_logits.float().sigmoid().cummin(-1).values)
