"""Flow matching over lateral paths using the common observation and survival model."""
from dataclasses import dataclass
import math

import torch
from torch import nn

from .model import DirectConfig, ObservationFollower, PathDecoderLayer
from .survival_confidence import survival_predictions

ARCHITECTURE = 'aligned_path_flow_v1'


@dataclass
class FlowConfig(DirectConfig):
    model_type: str = 'flow_matching'
    recurrent_refinement_steps: int = 0
    flow_steps: int = 4
    flow_draws: int = 64
    flow_sigma: tuple = ()

    def __post_init__(self):
        super().__post_init__()
        self.flow_sigma = tuple(tuple(row) for row in self.flow_sigma)
        if self.model_type != 'flow_matching' or self.recurrent_refinement_steps != 0:
            raise ValueError('Flow uses midpoint integration, not regression retries')
        if min(self.flow_steps, self.flow_draws) < 1:
            raise ValueError('Positive flow steps and draws required')
        if self.flow_sigma and (len(self.flow_sigma) != self.n_future or any(
                len(row) != 2 or any(not math.isfinite(v) or v < 1 for v in row) for row in self.flow_sigma)):
            raise ValueError('Flow scales must contain two finite values >= 1 per future plane')

    @property
    def architecture(self):
        return ARCHITECTURE


def time_embedding(t, width=64):
    frequencies = torch.exp(-math.log(1e4)*torch.arange(width//2, device=t.device)/ (width//2))
    angles = t.float()[..., None]*1e3*frequencies
    return torch.cat((angles.sin(), angles.cos()), -1)


def flow_targets(batch, cfg):
    """Use the common geometry contract, sampled at the output planes."""
    from ..train.supervision import geometry_mask
    indices = torch.linspace(0, batch['dense_ab'].shape[1]-1, cfg.n_future,
                             device=batch['dense_ab'].device).round().long()
    mask = geometry_mask(batch, cfg)[:, indices]
    target = batch['dense_ab'][:, indices].float()
    return torch.where(mask[..., None], target, 0.), mask


def fit_flow_sigma(batches, cfg, states=2048):
    """Fit physical residual scales from training targets only."""
    count = torch.zeros(cfg.n_future, dtype=torch.float64)
    total = torch.zeros(cfg.n_future, 2, dtype=torch.float64)
    square = torch.zeros_like(total)
    seen = 0
    while seen < states:
        batch = next(batches)
        target, mask = flow_targets(batch, cfg)
        n = min(len(target), states-seen)
        target, mask = target[:n].double().cpu(), mask[:n].cpu()
        if not torch.isfinite(target).all():
            raise ValueError('Nonfinite flow calibration target')
        count += mask.sum(0)
        total += target.sum(0)
        square += target.square().sum(0)
        seen += n
    if (count < 2).any():
        raise ValueError('Flow calibration needs at least two known targets per plane; increase --flow-calibration-states')
    variance = (square/count[:, None]-(total/count[:, None]).square()).clamp_min(0)
    return tuple(map(tuple, variance.sqrt().clamp_min(1).tolist()))


class FlowFollower(ObservationFollower):
    """One deterministic tracing path; Gaussian/time draws only train velocity."""
    def __init__(self, cfg):
        super().__init__(cfg)
        if not cfg.flow_sigma:
            raise ValueError('Fit flow scales before constructing the flow model')
        h = cfg.hidden
        self.query = nn.Sequential(nn.Linear(h+4, h), nn.SiLU(), nn.Linear(h, h))
        self.time = nn.Sequential(nn.Linear(64, h), nn.SiLU(), nn.Linear(h, h))
        layer = PathDecoderLayer(h, cfg.heads, cfg.decoder_ffn, dropout=0., activation='gelu',
                                 batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, cfg.decoder_layers, norm=nn.LayerNorm(h))
        self.velocity = nn.Linear(h, 2)
        nn.init.normal_(self.velocity.weight, std=.01)
        nn.init.zeros_(self.velocity.bias)
        self.register_buffer('sigma', torch.tensor(cfg.flow_sigma, dtype=torch.float32), persistent=False)

    def to_points(self, y):
        z = self.planes.reshape(*([1]*(y.ndim-2)), -1, 1).expand(*y.shape[:-1], 1)
        return torch.cat((y*self.sigma, z), -1)

    def prepare_prediction(self, ctx, hist):
        memory, padding = self.decoder_memory(ctx)
        ctx['generator_projected'] = [layer.project_memory(memory) for layer in self.decoder.layers]
        return ctx['generator_projected'], padding

    def velocity_field(self, ctx, y, t, known=None):
        # Draws have independent self-attention, then share one observation K/V bank per state.
        b, draws, planes, _ = y.shape
        if known is None:
            known = torch.ones(b, planes, device=y.device, dtype=torch.bool)
        y = torch.where(known[:, None, :, None], y, 0.)
        points = self.to_points(y)
        evidence, support = self.sample_local(ctx['deep'].float(), points.reshape(b, draws*planes, 3))
        evidence = evidence.reshape(b, draws, planes, -1)
        forward = points[..., 2:]/(self.cfg.n_future*self.cfg.future_step)
        query = self.query(torch.cat((evidence.to(ctx['deep'].dtype), y, forward,
                                      support.reshape(b, draws, planes, 1)), -1))
        query = query+self.time(time_embedding(t))[:, :, None]
        padding = (~known)[:, None].expand(-1, draws, -1).reshape(b*draws, planes).clone()
        padding[:, -1] &= ~padding.all(-1)
        for layer, kv in zip(self.decoder.layers, ctx['generator_projected']):
            query = layer.forward_draws(query, kv, ctx['padding'], padding,
                history=ctx['history_projected'], history_attention=self.history_attention)
        return self.velocity(self.decoder.norm(query)).float()

    def generate(self, ctx, hist):
        b = len(hist)
        y = hist.new_zeros(b, 1, self.cfg.n_future, 2)
        curves = [self.to_points(y)[:, 0]]
        with torch.no_grad():
            for step in range(self.cfg.flow_steps):
                t = y.new_full((b, 1), step/self.cfg.flow_steps)
                v = self.velocity_field(ctx, y, t)
                y = y+self.velocity_field(ctx, y+v/(2*self.cfg.flow_steps),
                                           t+1/(2*self.cfg.flow_steps))/self.cfg.flow_steps
                curves.append(self.to_points(y)[:, 0])
        return curves

    def training_forward(self, x, hist, hmask, threshold, targets=None):
        ctx = self.context(x, hist, hmask)
        self.prepare_prediction(ctx, hist)
        curves = self.generate(ctx, hist)
        points = curves[-1]
        hazards = self.hazard_logits(ctx, points)
        logits, confidence = survival_predictions(hazards)
        result = self.proposal_output(curves[0], [points], [(hazards, logits, confidence)],
                                      [torch.ones(len(hist), device=hist.device, dtype=torch.bool)])
        result['solver_points'] = torch.stack(curves, 1)
        if targets is not None:
            target, known = flow_targets(targets, self.cfg)
            b, draws = len(hist), self.cfg.flow_draws
            noise = targets.get('flow_noise')
            if noise is None:
                noise = torch.randn(b, draws, self.cfg.n_future, 2, device=hist.device)
            times = targets.get('flow_times')
            if times is None:
                times = (torch.arange(draws, device=hist.device)[None]+torch.rand(b, draws, device=hist.device))/draws
            end = torch.where(known[:, None, :, None], (target/self.sigma)[:, None], noise)
            y = (1-times[..., None, None])*noise+times[..., None, None]*end
            predicted = self.velocity_field(ctx, y, times, known)
            error = (predicted-(end-noise)).square()
            mask = known[:, None, :, None].expand_as(error)
            result['flow_per_state'] = torch.where(mask, error, 0.).sum((1, 2, 3))/mask.sum((1, 2, 3)).clamp_min(1)
        return result

    def forward(self, x, hist, hmask, confidence_threshold=.5, n_commit=None):
        return self.select_prediction(self.training_forward(x, hist, hmask, confidence_threshold),
                                      confidence_threshold, n_commit)
