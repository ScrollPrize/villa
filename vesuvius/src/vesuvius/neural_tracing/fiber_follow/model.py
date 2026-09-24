"""Future-path flow matching conditioned on fixed observed trace history."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from vesuvius.neural_tracing.fiber_follow.geometry import CropSpec, crop_local_grid
from vesuvius.neural_tracing.fiber_follow.policy import DEFAULT_MAX_RECOVERY_DISTANCE

ARCHITECTURE = 'future_flow_v10'


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
    recent_history_points: int = 8  # dense observed history used by the flow and scorer
    flow_layers: int = 4
    flow_heads: int = 4
    flow_steps: int = 4       # midpoint steps, two flow evaluations per step
    flow_samples: int = 16    # future-path samples drawn per state
    flow_draws: int = 32      # stratified (time, noise) draws against shared image features
    flow_sigma: tuple = ()   # fitted per-plane (a, b) residual std, trace-grid voxels
    flow_stencil_radius: float = 2.0  # 3x3 lateral patch pitch/radius, trace-grid voxels
    support_radius: float = 1.5  # RMS lateral distance on the commit window
    max_recovery_distance: float = DEFAULT_MAX_RECOVERY_DISTANCE  # origin to first point, trace-grid voxels
    norm: str = 'group'

    def __post_init__(self):
        self.widths = tuple(self.widths)
        self.flow_sigma = tuple(tuple(row) for row in self.flow_sigma)
        if (not math.isfinite(self.max_recovery_distance) or self.max_recovery_distance <= 0
                or not 0 < self.future_step <= self.max_recovery_distance):
            raise ValueError('max_recovery_distance must be finite, positive, and at least future_step')

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


def time_embedding(t, dim):
    """Sinusoidal features of flow time ``t`` in [0, 1], shape (N, dim)."""
    half = dim // 2
    freqs = torch.exp(-math.log(1e4) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    angles = t.float()[:, None] * 1e3 * freqs[None]
    return torch.cat([angles.sin(), angles.cos()], -1)


def prior_mean(hist, hmask, cfg):
    """Extrapolate the observed tangent, keeping the observation patches in-crop."""
    tangent, measured = history_tangent(hist, hmask, cfg.recent_history_points)
    usable = measured & (tangent[:, 2] > 1e-3)
    slope = tangent[:, :2] / tangent[:, 2:].clamp_min(1e-3)
    slope = torch.where(usable[:, None], slope, 0.)
    planes = cfg.future_step * torch.arange(1, cfg.n_future+1, device=hist.device)
    half = (cfg.width-1)*cfg.spacing/2 - cfg.flow_stencil_radius
    lateral = (slope[:, None] * planes[None, :, None]).clamp(-half, half)
    return torch.cat([lateral, planes[None, :, None].expand(len(hist), -1, -1)], -1)


def candidate_support(candidates, samples, radius, window):
    """Local sample density for every path; generated paths occupy the first S slots.

    Leave out each generated path's own vote. Teachers and replay paths are
    evaluated against the same sample bank, without contributing votes to it.
    """
    B, S, _, _ = samples.shape
    key = candidates[:, :, :window, :2].reshape(B, candidates.shape[1], -1).float()
    bank = samples[:, :, :window, :2].reshape(B, S, -1).float()
    within = torch.cdist(key, bank) / math.sqrt(window) < radius
    within[:, :S] &= ~torch.eye(S, device=samples.device, dtype=torch.bool)[None]
    counts = within.float().sum(-1)
    denominator = counts.new_full((counts.shape[1],), S)
    denominator[:S] = max(1, S-1)
    return counts / denominator


class PathFlow(nn.Module):
    """Normalized lateral flow with fixed history and prior-mean image tokens."""
    def __init__(self, cfg, feature_channels):
        super().__init__()
        self.cfg = cfg
        sigma = torch.tensor(cfg.flow_sigma, dtype=torch.float32)
        if sigma.shape != (cfg.n_future, 2) or not torch.isfinite(sigma).all() or (sigma < 1).any():
            raise ValueError('flow_sigma must contain fitted (a, b) scales >= 1 for every future plane')
        half = (cfg.width-1)*cfg.spacing/2
        if not math.isfinite(cfg.flow_stencil_radius) or not 0 < cfg.flow_stencil_radius <= half:
            raise ValueError('flow_stencil_radius must be positive and fit inside the crop')
        self.register_buffer('sigma', sigma, persistent=False)
        h = cfg.hidden
        self.n_history = 1 + cfg.recent_history_points
        self.n_tokens = cfg.n_future
        self.n_fixed = self.n_history + self.n_tokens
        kind = torch.cat([torch.zeros(self.n_history), torch.ones(self.n_tokens),
                          torch.full((self.n_tokens,), 2)]).long()
        self.register_buffer('kind', kind, persistent=False)
        self.register_buffer('planes', cfg.future_step*torch.arange(1, cfg.n_future+1), persistent=False)
        stencil = torch.tensor([[a, b, 0.] for a in (-1., 0., 1.) for b in (-1., 0., 1.)])
        self.register_buffer('stencil', stencil*cfg.flow_stencil_radius, persistent=False)
        self.kind_embedding = nn.Embedding(3, h)
        self.position = nn.Embedding(len(kind), h)
        self.input = nn.Sequential(nn.Linear(feature_channels*9 + 5, h), nn.SiLU(), nn.Linear(h, h))
        self.time = nn.Sequential(nn.Linear(64, h), nn.SiLU(), nn.Linear(h, h))
        self.context = nn.Linear(h, h)
        layer = nn.TransformerEncoderLayer(h, cfg.flow_heads, 4*h, dropout=0., activation='gelu',
                                           batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, cfg.flow_layers, enable_nested_tensor=False)
        self.final = nn.LayerNorm(h)
        self.velocity = nn.Linear(h, 2)
        nn.init.normal_(self.velocity.weight, std=.01)
        nn.init.zeros_(self.velocity.bias)

    def prior(self, hist, hmask):
        return prior_mean(hist, hmask, self.cfg)

    def to_voxels(self, y, mu):
        lateral = mu[:, None, :, :2] + self.sigma*y
        return torch.cat([lateral, mu[:, None, :, 2:].expand(*y.shape[:-1], 1)], -1)

    def sample_features(self, features, coordinates, sampling_grid):
        B = len(coordinates)
        grid = sampling_grid(coordinates.reshape(B, -1, 1, 3) + self.stencil)
        sampled = F.grid_sample(features, grid[:, :, :, None], align_corners=True, padding_mode='zeros')
        return sampled[..., 0].permute(0, 2, 1, 3).reshape(*coordinates.shape[:-1], -1)

    def conditioning(self, features, hist, hmask, sampling_grid):
        """Sample static history/observation patches once per encoded state."""
        mu = self.prior(hist, hmask)
        observed = torch.cat([hist.new_zeros((len(hist), 1, 3)), hist[:, :self.n_history-1]], 1).float()
        supplied = torch.cat([hmask.new_ones((len(hist), 1)), hmask[:, :self.n_history-1]], 1).bool()
        observed = torch.where(supplied[..., None], observed, 0.)
        coordinates = torch.cat([observed, mu], 1)
        valid = torch.cat([supplied, torch.ones_like(mu[..., 0], dtype=torch.bool)], 1)
        return dict(mu=mu, coordinates=coordinates, valid=valid,
                    sampled=self.sample_features(features, coordinates, sampling_grid))

    def forward(self, features, context, y, t, fixed, sampling_grid, future_mask=None):
        """y (B, N, K, 2) is normalized; only feature queries use voxel coordinates."""
        B, N, K, _ = y.shape
        known = torch.ones(B, K, device=y.device, dtype=torch.bool) if future_mask is None else future_mask.bool()
        # Mask before conversion/sampling so arbitrary missing-target values are inert.
        y = torch.where(known[:, None, :, None], y, 0.)
        future = self.to_voxels(y.float(), fixed['mu'])
        future = torch.where(known[:, None, :, None], future, 0.)
        sampled = self.sample_features(features, future, sampling_grid)
        sampled = torch.cat([fixed['sampled'][:, None].expand(-1, N, -1, -1), sampled], 2)
        coordinates = torch.cat([fixed['coordinates'][:, None].expand(-1, N, -1, -1), future], 2)
        P = coordinates.shape[2]
        coordinates = coordinates.reshape(B*N, P, 3)
        t = t.reshape(B*N).float()
        observed_flag = (self.kind == 0).float()[None, :, None].expand(B*N, -1, -1)
        tokens = torch.cat([sampled.reshape(B*N, P, -1), coordinates/32, observed_flag,
                            t[:, None, None].expand(-1, P, 1)], -1)
        h = self.input(tokens) + self.kind_embedding(self.kind)[None] + self.position.weight[None]
        h = h + (self.time(time_embedding(t, 64)) + self.context(context[:, None].expand(B, N, -1).reshape(B*N, -1)))[:, None]
        valid = torch.cat([fixed['valid'], known], 1)
        padding = ~valid[:, None].expand(B, N, P).reshape(B*N, P)
        h = self.blocks(h, src_key_padding_mask=padding)
        return self.velocity(self.final(h[:, self.n_fixed:])).float().reshape(B, N, K, 2)


class FollowNet(nn.Module):
    def __init__(self, cfg: FollowNetConfig, *, create_flow=True):
        super().__init__()
        self.cfg = cfg
        if cfg.n_future < 2 or cfg.flow_samples < 1 or cfg.hist_points < 1:
            raise ValueError('At least two future planes, one candidate, and one history point are required')
        if not 1 <= cfg.recent_history_points <= cfg.hist_points * cfg.hist_stride:
            raise ValueError('recent_history_points must be positive and fit inside the supplied history')
        if cfg.n_future * cfg.future_step > (cfg.depth-cfg.behind-1)*cfg.spacing:
            raise ValueError('Candidates/forecast horizon exceed the proposal configuration')
        if min(cfg.flow_layers, cfg.flow_heads, cfg.flow_steps, cfg.flow_draws) < 1 or cfg.hidden % cfg.flow_heads:
            raise ValueError('Flow layers, heads, steps and draws must be positive; hidden must divide by heads')
        if any(not math.isfinite(v) or v <= 0 for v in (cfg.support_radius,)):
            raise ValueError('support_radius must be positive and finite')
        self.crop = CropSpec(depth=cfg.depth, width=cfg.width, behind=cfg.behind,
                             spacing=cfg.spacing)
        grid = torch.from_numpy(crop_local_grid(self.crop)).float()
        self.register_buffer('coordinates', grid.permute(3, 0, 1, 2)[None] / 32, persistent=False)
        w = cfg.widths
        self.encoders = nn.ModuleList([block(cfg.in_channels+3, w[0], cfg.norm)] +
                                     [block(a, b, cfg.norm, 2) for a, b in zip(w[:-1], w[1:])])
        self.decoders = nn.ModuleList([block(w[i+1]+w[i], w[i], cfg.norm) for i in range(len(w)-2, -1, -1)])
        self.history = nn.Sequential(nn.Linear(cfg.hist_points*4, cfg.hidden), nn.SiLU())
        self.condition = nn.ModuleList([nn.Linear(cfg.hidden, 2*c) for c in w])
        if create_flow:
            self.flow = PathFlow(cfg, w[0])
        self.path_net = nn.Sequential(nn.Conv1d(w[0]*5+7, cfg.hidden, 3, padding=1), nn.SiLU(),
                                      nn.Conv1d(cfg.hidden, cfg.hidden, 3, padding=1), nn.SiLU())
        self.rank_head = nn.Linear(cfg.hidden, 1)
        # Prefix labels depend on every earlier candidate point. Carry that
        # information forward explicitly; local convolutions alone cannot do it.
        self.prefix_context = nn.GRU(cfg.hidden, cfg.hidden, batch_first=True)
        # Future evidence travels back to the first decision. Each candidate
        # has independent recurrent state, including synthetic training paths.
        self.suffix_context = nn.GRU(cfg.hidden, cfg.hidden, batch_first=True)
        self.continuation_fusion = nn.Sequential(nn.Linear(4*cfg.hidden, cfg.hidden), nn.SiLU())
        self.history_tokens = nn.Sequential(nn.Linear(w[0]+4, cfg.hidden), nn.SiLU(),
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
        """Spatial features and the geometric-history context vector."""
        cfg = self.cfg
        if self.encoders[0][0].weight.is_contiguous(memory_format=torch.channels_last_3d):
            x = x.contiguous(memory_format=torch.channels_last_3d)
        h = hist[:, cfg.hist_stride-1::cfg.hist_stride][:, :cfg.hist_points]
        m = hmask[:, cfg.hist_stride-1::cfg.hist_stride][:, :cfg.hist_points]
        if h.shape[1] != cfg.hist_points:
            raise ValueError('History must cover hist_points * hist_stride')
        h = torch.where(m[..., None] > 0, h, 0.)
        context = self.history(torch.cat([h/32, m[..., None]], -1).flatten(1).to(x.dtype))
        features = torch.cat([x, self.coordinates.expand(len(x), -1, -1, -1, -1).to(x.dtype)], 1)
        skips = []
        for encoder, condition in zip(self.encoders, self.condition):
            features = encoder(features)
            gain, bias = condition(context).chunk(2, -1)
            features = features * (1 + .1*gain[..., None, None, None]) + bias[..., None, None, None]
            skips.append(features)
        for decoder, skip in zip(self.decoders, reversed(skips[:-1])):
            features = decoder(torch.cat([F.interpolate(features, size=skip.shape[-3:], mode='trilinear', align_corners=True), skip], 1))
        return features, context

    def future_targets(self, batch):
        """Annotated future crossings; departed states have no geometry targets."""
        ab = batch['plane_ab'].float()
        future = torch.cat([ab, self.flow.planes[None, :, None].expand(len(ab), -1, 1)], -1)
        return future, batch['plane_mask'].float()*(1-batch['offtrack'].float())[:, None]

    def flow_loss(self, features, context, hist, hmask, batch, generator=None, fixed=None):
        """Stratified flow matching on annotated normalized lateral residuals.

        Missing futures remain excluded as attention keys and from the loss.
        Observation tokens remain available on every requested plane.
        """
        cfg = self.cfg
        if fixed is None:
            fixed = self.flow.conditioning(features, hist, hmask, self.sampling_grid)
        x1, token_mask = self.future_targets(batch)
        mu = fixed['mu']
        B, P, _ = mu.shape
        D = cfg.flow_draws
        t = (torch.arange(D, device=mu.device)[None] +
             torch.rand(B, D, device=mu.device, generator=generator)) / D
        y0 = torch.randn(B, D, P, 2, device=mu.device, generator=generator)
        residual = torch.where(token_mask[..., None] > 0, x1[..., :2]-mu[..., :2], 0.)
        y1 = residual / self.flow.sigma
        target = torch.where(token_mask[:, None, :, None] > 0, y1[:, None], y0)
        yt = (1-t)[..., None, None]*y0 + t[..., None, None]*target
        velocity = self.flow(features, context, yt, t, fixed, self.sampling_grid, future_mask=token_mask)
        error = (velocity - (target-y0)).square()
        weight = token_mask[:, None, :, None].expand_as(error)
        loss = (error*weight).sum()/weight.sum().clamp_min(1)
        return dict(flow_loss=loss, flow_known_fraction=token_mask.mean().detach())

    @torch.no_grad()
    def sample(self, features, context, hist, hmask, n_samples=None, steps=None, generator=None, fixed=None):
        """Explicit midpoint integration in normalized coordinates; returns voxel-space paths."""
        cfg = self.cfg
        S = cfg.flow_samples if n_samples is None else n_samples
        T = cfg.flow_steps if steps is None else steps
        if min(S, T) < 1:
            raise ValueError('Sample count and midpoint steps must be positive')
        if fixed is None:
            fixed = self.flow.conditioning(features, hist, hmask, self.sampling_grid)
        mu = fixed['mu']
        B, P, _ = mu.shape
        y = torch.randn(B, S, P, 2, device=mu.device, generator=generator)
        for i in range(T):
            t = torch.full((B, S), i/T, device=mu.device)
            velocity = self.flow(features, context, y, t, fixed, self.sampling_grid)
            midpoint = y + velocity/(2*T)
            y = y + self.flow(features, context, midpoint, t+1/(2*T), fixed, self.sampling_grid)/T
        return self.flow.to_voxels(y, mu)

    def encode_history(self, features, hist, hmask):
        """Encode actual observed history for both ranking and confidence.

        Read oldest to newest, skipping masked points without advancing the
        recurrent state. No future candidate or annotated history is supplied.
        """
        count = self.cfg.recent_history_points
        observed = torch.cat([hist.new_zeros((len(hist), 1, 3)), hist[:, :count]], 1)
        valid = torch.cat([hmask.new_ones((len(hist), 1)), hmask[:, :count]], 1)
        observed = torch.where(valid[..., None] > 0, observed, 0.)
        grid = self.sampling_grid(observed)[:, :, None, None]
        sampled = F.grid_sample(features.float(), grid.float(), align_corners=True,
                                padding_mode='zeros').squeeze(-1).squeeze(-1).transpose(1, 2)
        tokens = self.history_tokens(torch.cat([sampled, observed/32, valid[..., None]], -1))
        state = tokens.new_zeros((1, len(hist), self.cfg.hidden))
        for i in range(count, -1, -1):
            _, updated = self.prefix_context(tokens[:, i:i+1].contiguous(), state)
            state = torch.where(valid[None, :, i:i+1] > 0, updated, state)
        # Keep a short route from all supplied history, including its oldest
        # image observations, alongside the ordered recurrent representation.
        pooled = (tokens*valid[..., None]).sum(1)/valid.sum(1, keepdim=True).clamp_min(1)
        return self.history_fusion(torch.cat([state[0], pooled], -1))[None]

    def score_candidates(self, features, candidates, hist, hmask, support):
        B, M, K, _ = candidates.shape
        grid = self.sampling_grid(candidates[..., None, :] + self.stencil)
        sampled = F.grid_sample(features.float(), grid.float(), align_corners=True, padding_mode='zeros')
        # B,C,M,K,5 -> B*M,C*5,K
        sampled = sampled.permute(0, 2, 1, 4, 3).reshape(B*M, -1, K)
        # The tracer connects to the actual current point (local origin).
        initial = candidates[:, :, :1]
        delta = torch.cat([initial, candidates[:, :, 1:]-candidates[:, :, :-1]], 2)
        geom = torch.cat([candidates/32, delta/4], -1).reshape(B*M, K, 6).transpose(1, 2)
        density = support.reshape(B*M, 1, 1).expand(-1, -1, K)
        tokens = self.path_net(torch.cat([sampled, geom, density], 1))
        context = self.encode_history(features, hist, hmask)
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

    def forward(self, x, hist, hmask, extra_candidates=None, targets=None, generator=None):
        """Generate future paths and score every sample, teacher and replay candidate.

        ``targets`` (the training batch) adds the flow-matching loss to the
        output. ``generator`` makes sampling reproducible. Candidates never
        receive gradients from the scorer: the flow is trained by its own loss.
        """
        cfg = self.cfg
        features, context = self.encode(x, hist, hmask)
        image = features.float()
        out = {}
        fixed = self.flow.conditioning(image, hist, hmask, self.sampling_grid)
        if targets is not None:
            out.update(self.flow_loss(image, context, hist, hmask, targets, generator, fixed))
        samples = self.sample(image, context, hist, hmask, generator=generator, fixed=fixed)
        candidates = samples
        if extra_candidates is not None:
            candidates = torch.cat([candidates, extra_candidates.detach()], 1)
        support = candidate_support(candidates, samples, cfg.support_radius, min(4, cfg.n_future))
        ranks, confidence_logits = self.score_candidates(image, candidates, hist, hmask, support)
        chosen = ranks.argmax(-1)
        points = candidates[torch.arange(len(x), device=x.device), chosen]
        out.update(samples=samples, candidate_support=support,
                   points=points, candidates=candidates,
                   ranks=ranks, confidence_logits=confidence_logits,
                   confidence=confidence_logits.float().sigmoid().cummin(-1).values)
        return out
