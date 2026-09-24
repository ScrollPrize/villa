"""History-conditioned joint flow-matching path generator and supervised path scorer."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from vesuvius.neural_tracing.fiber_follow.geometry import CropSpec, crop_local_grid

ARCHITECTURE = 'joint_flow_v8'


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
    n_candidates: int = 4
    flow_layers: int = 4
    flow_heads: int = 4
    flow_steps: int = 8       # Euler steps from the prior to a path sample
    flow_samples: int = 16    # joint polyline samples drawn per state
    flow_draws: int = 8       # (time, noise) draws per example in the training loss
    prior_scale: float = 4.0  # prior standard deviation, trace-grid voxels
    candidate_separation: float = 1.5  # RMS lateral distance on the commit window that separates modes
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
    """Weighted line fit in arclength order, newest point first (diagnostics only).

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


def select_candidates(paths, count, separation, window):
    """Pick ``count`` modes of a sample set by density peaks with suppression.

    Samples are compared by the RMS lateral distance over the first ``window``
    planes (the commit window). The densest unsuppressed sample is taken and
    everything within ``separation`` of it is suppressed; when all samples are
    suppressed, remaining slots take the sample farthest from the selection so
    every slot is a distinct real sample. Returns indices (B, count) and the
    support of each pick: the fraction of all samples within ``separation``.
    """
    B, S, K, _ = paths.shape
    if count > S:
        raise ValueError('Cannot select more candidates than samples')
    key = paths[:, :, :window, :2].reshape(B, S, -1).float()
    distance = torch.cdist(key, key) / math.sqrt(window)
    within = distance < separation
    support = within.float().mean(-1)
    suppressed = torch.zeros(B, S, dtype=torch.bool, device=paths.device)
    batch = torch.arange(B, device=paths.device)
    selected = []
    for _ in range(count):
        density = (within & ~suppressed[:, None, :]).sum(-1).float().masked_fill(suppressed, -1.)
        pick = density.argmax(-1)
        if selected:
            chosen = torch.stack(selected, 1)
            farthest = distance[batch[:, None], chosen].amin(1).argmax(-1)
            pick = torch.where(suppressed.all(-1), farthest, pick)
        suppressed |= within[batch, pick]
        selected.append(pick)
    index = torch.stack(selected, 1)
    return index, support[batch[:, None], index]


class PathFlow(nn.Module):
    """Velocity field over the joint polyline: anchor, cleaned past, future planes.

    Every token carries its current coordinate, its prior mean, image features
    sampled at the current coordinate, and whether an observation exists there.
    The forward coordinate of future tokens is fixed to its plane; the anchor
    and past tokens move in all three axes.
    """
    def __init__(self, cfg, feature_channels):
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden
        n = cfg.clean_points
        self.n_tokens = 1 + n + cfg.n_future
        kind = torch.zeros(self.n_tokens, dtype=torch.long)
        kind[1:1+n] = 1
        kind[1+n:] = 2
        self.register_buffer('kind', kind, persistent=False)
        free = torch.ones(self.n_tokens, 3)
        free[1+n:, 2] = 0
        self.register_buffer('free', free, persistent=False)
        self.register_buffer('planes', cfg.future_step*torch.arange(1, cfg.n_future+1), persistent=False)
        self.register_buffer('stencil', torch.tensor([[0., 0, 0], [1., 0, 0], [-1., 0, 0], [0., 1, 0], [0., -1, 0]])*cfg.spacing,
                             persistent=False)
        self.kind_embedding = nn.Embedding(3, h)
        self.position = nn.Embedding(self.n_tokens, h)
        self.input = nn.Sequential(nn.Linear(feature_channels*5 + 8, h), nn.SiLU(), nn.Linear(h, h))
        self.time = nn.Sequential(nn.Linear(64, h), nn.SiLU(), nn.Linear(h, h))
        self.context = nn.Linear(h, h)
        layer = nn.TransformerEncoderLayer(h, cfg.flow_heads, 4*h, dropout=0., activation='gelu',
                                           batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, cfg.flow_layers, enable_nested_tensor=False)
        self.final = nn.LayerNorm(h)
        self.velocity = nn.Linear(h, 3)
        # Small, not zero: a zero head would block every upstream gradient on the first step.
        nn.init.normal_(self.velocity.weight, std=.01)
        nn.init.zeros_(self.velocity.bias)

    def prior(self, hist, hmask):
        """Prior mean and observation flags: observed past where supplied, straight ahead otherwise."""
        cfg = self.cfg
        B = len(hist)
        n = cfg.clean_points
        observed = hist[:, :n].float()
        supplied = hmask[:, :n].float()
        default = torch.zeros_like(observed)
        default[..., 2] = -torch.arange(1, n+1, device=hist.device, dtype=torch.float32)
        past = torch.where(supplied[..., None] > 0, observed, default)
        future = torch.zeros(B, cfg.n_future, 3, device=hist.device)
        future[..., 2] = self.planes
        mu = torch.cat([torch.zeros(B, 1, 3, device=hist.device), past, future], 1)
        valid = torch.cat([torch.ones(B, 1, device=hist.device), supplied, torch.zeros(B, cfg.n_future, device=hist.device)], 1)
        return mu, valid

    def forward(self, features, context, x, t, mu, valid, sampling_grid):
        """features (B, C, D, H, W) float; x, mu (B, N, P, 3); t (B, N); valid (B, P); context (B, hidden)."""
        B, N, P, _ = x.shape
        grid = sampling_grid(x.reshape(B, N*P, 1, 3).float() + self.stencil)
        sampled = F.grid_sample(features, grid[:, :, :, None], align_corners=True, padding_mode='zeros')
        sampled = sampled[..., 0].permute(0, 2, 1, 3).reshape(B*N, P, -1)
        x = x.reshape(B*N, P, 3).float()
        mu = mu.expand(B, N, P, 3).reshape(B*N, P, 3).float()
        t = t.reshape(B*N).float()
        tokens = torch.cat([sampled, x/32, mu/32, valid[:, None].expand(B, N, P).reshape(B*N, P, 1).float(),
                            t[:, None, None].expand(-1, P, 1)], -1)
        h = self.input(tokens) + self.kind_embedding(self.kind)[None] + self.position.weight[None]
        h = h + (self.time(time_embedding(t, 64)) + self.context(context[:, None].expand(B, N, -1).reshape(B*N, -1)))[:, None]
        h = self.blocks(h)
        return (self.velocity(self.final(h)).float() * self.free).reshape(B, N, P, 3)


class FollowNet(nn.Module):
    def __init__(self, cfg: FollowNetConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.n_future < 2 or cfg.n_candidates < 1 or cfg.hist_points < 1:
            raise ValueError('At least two future planes, one candidate, and one history point are required')
        if cfg.clean_points < 1 or cfg.clean_points > cfg.hist_points * cfg.hist_stride:
            raise ValueError('clean_points must be positive and fit inside the supplied history')
        if cfg.n_future * cfg.future_step > (cfg.depth-cfg.behind-1)*cfg.spacing:
            raise ValueError('Candidates/forecast horizon exceed the proposal configuration')
        if min(cfg.flow_layers, cfg.flow_heads, cfg.flow_steps, cfg.flow_draws) < 1 or cfg.hidden % cfg.flow_heads:
            raise ValueError('Flow layers, heads, steps and draws must be positive; hidden must divide by heads')
        if cfg.flow_samples < cfg.n_candidates:
            raise ValueError('flow_samples must be at least n_candidates')
        if any(not math.isfinite(v) or v <= 0 for v in (cfg.prior_scale, cfg.candidate_separation)):
            raise ValueError('prior_scale and candidate_separation must be positive and finite')
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
        self.flow = PathFlow(cfg, w[0])
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
        """Spatial features and the geometric-history context vector."""
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
        return features, context

    def polyline_targets(self, batch):
        """Joint target polyline and its token mask from dense annotated geometry.

        Past tokens are targets only where observed history was supplied, so an
        absent history is never a reconstruction target. Unknown planes and
        departed states are masked; they leave no gradient.
        """
        n = self.cfg.clean_points
        clean = batch['clean_local'].float()
        supplied = torch.cat([batch['hmask'].new_ones((len(clean), 1)), batch['hmask'][:, :n]], 1).float()
        future = torch.cat([batch['plane_ab'].float(), self.flow.planes[None, :, None].expand(len(clean), -1, 1)], -1)
        x1 = torch.cat([clean, future], 1)
        mask = torch.cat([batch['clean_mask'].float()*supplied, batch['plane_mask'].float()], 1)
        return x1, mask*(1-batch['offtrack'].float())[:, None]

    def flow_loss(self, features, context, hist, hmask, batch, generator=None):
        """Conditional flow matching on the joint polyline, in units of the prior scale."""
        cfg = self.cfg
        mu, valid = self.flow.prior(hist, hmask)
        x1, token_mask = self.polyline_targets(batch)
        x1 = torch.where(token_mask[..., None] > 0, x1, mu)
        B, P, _ = mu.shape
        D = cfg.flow_draws
        t = torch.rand(B, D, device=mu.device, generator=generator)
        noise = torch.randn(B, D, P, 3, device=mu.device, generator=generator) * self.flow.free
        x0 = mu[:, None] + cfg.prior_scale*noise
        xt = (1-t)[..., None, None]*x0 + t[..., None, None]*x1[:, None]
        velocity = self.flow(features, context, xt, t, mu[:, None], valid, self.sampling_grid)
        error = (velocity - (x1[:, None]-x0)).square() / cfg.prior_scale**2
        weight = (token_mask[:, None, :, None]*self.flow.free).expand_as(error)
        past = torch.zeros(P, device=mu.device)
        past[:1+cfg.clean_points] = 1
        def mean(select):
            w = weight*select[None, None, :, None]
            return (error*w).sum()/w.sum().clamp_min(1)
        loss = mean(torch.ones_like(past))
        return dict(flow_loss=loss, flow_past=mean(past).detach(), flow_future=mean(1-past).detach(),
                    flow_known_fraction=token_mask.mean().detach())

    @torch.no_grad()
    def sample(self, features, context, hist, hmask, n_samples=None, steps=None, generator=None):
        """Euler integration of the flow from the prior; returns (B, S, P, 3) polylines."""
        cfg = self.cfg
        S = n_samples or cfg.flow_samples
        T = steps or cfg.flow_steps
        mu, valid = self.flow.prior(hist, hmask)
        B, P, _ = mu.shape
        x = mu[:, None] + cfg.prior_scale*torch.randn(B, S, P, 3, device=mu.device, generator=generator)*self.flow.free
        for i in range(T):
            t = torch.full((B, S), i/T, device=mu.device)
            x = x + self.flow(features, context, x, t, mu[:, None], valid, self.sampling_grid)/T
        return x

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

    def forward(self, x, hist, hmask, extra_candidates=None, targets=None, generator=None):
        """Generate joint polyline samples, select candidate modes, and score them.

        ``targets`` (the training batch) adds the flow-matching loss to the
        output. ``generator`` makes sampling reproducible. Candidates never
        receive gradients from the scorer: the flow is trained by its own loss.
        """
        cfg = self.cfg
        features, context = self.encode(x, hist, hmask)
        image = features.float()
        out = {}
        if targets is not None:
            out.update(self.flow_loss(image, context, hist, hmask, targets, generator))
        samples = self.sample(image, context, hist, hmask, generator=generator)
        n = cfg.clean_points
        clean_history = samples[:, :, :n+1].mean(1)
        futures = samples[:, :, n+1:]
        index, support = select_candidates(futures, cfg.n_candidates, cfg.candidate_separation, min(4, cfg.n_future))
        candidates = futures[torch.arange(len(x), device=x.device)[:, None], index]
        if extra_candidates is not None:
            candidates = torch.cat([candidates, extra_candidates.detach()], 1)
        ranks, confidence_logits = self.score_candidates(image, candidates, clean_history, hist, hmask)
        chosen = ranks.argmax(-1)
        points = candidates[torch.arange(len(x), device=x.device), chosen]
        corrected_path = torch.cat([clean_history.flip(1), points], 1)
        out.update(clean_history=clean_history, samples=samples, candidate_support=support,
                   corrected_path=corrected_path, points=points, candidates=candidates,
                   ranks=ranks, confidence_logits=confidence_logits,
                   confidence=confidence_logits.float().sigmoid().cummin(-1).values)
        return out
