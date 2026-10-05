"""Flow matching over lateral paths using the common observation and survival model."""
from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from .model import CoordinateRegressionConfig, ObservationFollower, PathDecoderLayer
from .survival_confidence import survival_predictions



@dataclass
class FlowConfig(CoordinateRegressionConfig):
    model_type: str = 'flow_matching'
    recurrent_refinement_steps: int = 0
    flow_steps: int = 4
    flow_draws: int = 64
    flow_sigma: tuple = ()
    # Proposals beyond the zero-start path, from Gaussian starts (flow_sample_scale, in residual-scale
    # units). Training integrates and scores every proposal (the scorer learns sampled paths); tracing
    # takes a later proposal only when no earlier one is accepted (select_refinement). 0: one path.
    flow_samples: int = 0
    flow_sample_scale: float = 1.
    # 'input': the time embedding is added to the decoder queries once. 'adaln': it also modulates every
    # decoder branch and the output norm per draw (zero-initialized, so a warm start is exact).
    flow_time_conditioning: str = 'input'
    # Lower bound of the fitted residual scales (voxels), i.e. of the noise prior's width.
    flow_sigma_floor: float = 1.
    # Planes without a target in training: 'padded' leaves them out of self-attention; 'own_path' keeps
    # them, as tracing does, travelling from noise to this model's own zero-start path (no loss).
    flow_unknown_planes: str = 'padded'
    # Tracing-time choice among the proposals: 'retry' (later proposals only when no earlier one is
    # accepted) or 'best' (one ranking over all). flow_zero_start=False drops the zero-start path,
    # leaving only the Gaussian starts (needs flow_samples > 0).
    flow_selection: str = 'retry'
    flow_zero_start: bool = True
    # Tracing-time bar for Gaussian-start proposals (0: the gate's own threshold). A higher bar offsets the
    # optimism of taking the best of many noisy candidates: a sample plane counts as confident only above
    # it. Planes below the bar are pushed below the gate (so the tracer's commit check agrees); planes
    # above it keep their confidence, so eligible samples are ranked against the zero start as they are.
    flow_sample_threshold: float = 0.

    def __post_init__(self):
        super().__post_init__()
        self.flow_sigma = tuple(tuple(row) for row in self.flow_sigma)
        if self.model_type != 'flow_matching' or self.recurrent_refinement_steps != 0:
            raise ValueError('Flow uses midpoint integration, not regression retries')
        if min(self.flow_steps, self.flow_draws) < 1:
            raise ValueError('Positive flow steps and draws required')
        if type(self.flow_samples) is not int or self.flow_samples < 0 or not (
                math.isfinite(self.flow_sample_scale) and self.flow_sample_scale > 0):
            raise ValueError('Flow samples must be a nonnegative integer with a finite positive scale')
        if self.flow_time_conditioning not in ('input', 'adaln') or self.flow_unknown_planes not in ('padded', 'own_path'):
            raise ValueError("Flow time conditioning is 'input' or 'adaln'; unknown planes 'padded' or 'own_path'")
        if self.flow_selection not in ('retry', 'best') or (not self.flow_zero_start and not self.flow_samples):
            raise ValueError("Flow selection is 'retry' or 'best'; dropping the zero start needs flow samples")
        if not 0 <= self.flow_sample_threshold <= 1:
            raise ValueError('The flow sample threshold lies in [0, 1]')
        if not (math.isfinite(self.flow_sigma_floor) and self.flow_sigma_floor >= 1):
            raise ValueError('The flow scale floor must be finite and at least one voxel')
        if self.flow_sigma and (len(self.flow_sigma) != self.n_future or any(
                len(row) != 2 or any(not math.isfinite(v) or v < self.flow_sigma_floor for v in row)
                for row in self.flow_sigma)):
            raise ValueError('Flow scales must contain two finite values >= the scale floor per future plane')


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
    """Fit physical residual scales from training targets only, bounded below by the scale floor."""
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
    return tuple(map(tuple, variance.sqrt().clamp_min(cfg.flow_sigma_floor).tolist()))


class FlowFollower(ObservationFollower):
    """The zero-start path, optionally followed by sampled proposals; Gaussian/time draws train velocity."""
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
        if cfg.flow_time_conditioning == 'adaln':
            # Per layer: shift/scale/gate for four branches; output norm: shift/scale.
            self.time_modulation = nn.ModuleList(nn.Linear(h, 12*h) for _ in range(cfg.decoder_layers))
            self.output_modulation = nn.Linear(h, 2*h)
            for linear in (*self.time_modulation, self.output_modulation):
                nn.init.zeros_(linear.weight)
                nn.init.zeros_(linear.bias)
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
        time = self.time(time_embedding(t))
        query = query+time[:, :, None]
        padding = (~known)[:, None].expand(-1, draws, -1).reshape(b*draws, planes).clone()
        padding[:, -1] &= ~padding.all(-1)
        adaln = self.cfg.flow_time_conditioning == 'adaln'
        condition = F.silu(time) if adaln else None
        for index, (layer, kv) in enumerate(zip(self.decoder.layers, ctx['generator_projected'])):
            modulation = self.time_modulation[index](condition).unflatten(-1, (4, 3, self.cfg.hidden)) if adaln else None
            query = layer.forward_draws(query, kv, ctx['padding'], padding, history=ctx['history_projected'],
                                        history_attention=getattr(self, 'history_attention', None), modulation=modulation)
        output = self.decoder.norm(query)
        if adaln:
            shift, scale = self.output_modulation(condition)[:, :, None].chunk(2, -1)
            output = output*(1+scale)+shift
        return self.velocity(output).float()

    def generate(self, ctx, hist, start=None):
        """Midpoint integration from ``start`` (B, D, P, 2), by default the zero path (D=1).
        Returns the (B, D, P, 3) points at every solver step."""
        y = hist.new_zeros(len(hist), 1, self.cfg.n_future, 2) if start is None else start
        curves = [self.to_points(y)]
        with torch.no_grad():
            for step in range(self.cfg.flow_steps):
                t = y.new_full(y.shape[:2], step/self.cfg.flow_steps)
                v = self.velocity_field(ctx, y, t)
                y = y+self.velocity_field(ctx, y+v/(2*self.cfg.flow_steps),
                                           t+1/(2*self.cfg.flow_steps))/self.cfg.flow_steps
                curves.append(self.to_points(y))
        return curves

    def proposal_starts(self, hist, keys=None):
        """(B, D, P, 2): the zero path (unless flow_zero_start is off), then scaled Gaussian starts.

        ``keys`` (B,) int64, given by the tracer, seeds each row's starts from its own generator, so a
        decision's noise depends on its key only (not on batch neighbours, batch size or precision).
        Without keys (training) the global RNG draws them.
        """
        zero = hist.new_zeros(len(hist), 1, self.cfg.n_future, 2)
        if not self.cfg.flow_samples:
            return zero
        shape = (self.cfg.flow_samples, self.cfg.n_future, 2)
        if keys is None:
            noise = torch.randn(len(hist), *shape, device=hist.device)
        else:
            noise = torch.stack([torch.randn(*shape, device=hist.device,
                                             generator=torch.Generator(hist.device).manual_seed(int(key)))
                                 for key in keys.tolist()])
        noise = self.cfg.flow_sample_scale*noise.to(zero.dtype)
        return torch.cat((zero, noise), 1) if self.cfg.flow_zero_start else noise

    def training_forward(self, x, hist, hmask, threshold, targets=None):
        ctx = self.context(x, hist, hmask)
        self.prepare_prediction(ctx, hist)
        generated = self.generate(ctx, hist, self.proposal_starts(hist, x.get('flow_noise_keys')))
        curves = [curve[:, 0] for curve in generated]
        points = curves[-1]
        # Every proposal is scored, so the confidence loss trains the scorer on sampled paths too.
        proposals = list(generated[-1].unbind(1))
        scores = []
        for proposal in proposals:
            hazards = self.hazard_logits(ctx, proposal)
            scores.append((hazards, *survival_predictions(hazards)))
        valid = [torch.ones(len(hist), device=hist.device, dtype=torch.bool)]*len(proposals)
        result = self.proposal_output(curves[0], proposals, scores, valid)
        result['solver_points'] = torch.stack(curves, 1)
        # Same memory record and identity heads as coordinate regression, at the integrated path.
        if 'memory_entry' in ctx:
            result['memory_entry'] = ctx['memory_entry']
        result.update(self.identity_terms(ctx, x))
        result.update(self.verification_terms(ctx, x, points))
        if targets is not None:
            target, known = flow_targets(targets, self.cfg)
            b, draws = len(hist), self.cfg.flow_draws
            noise = targets.get('flow_noise')
            if noise is None:
                noise = torch.randn(b, draws, self.cfg.n_future, 2, device=hist.device)
            times = targets.get('flow_times')
            if times is None:
                times = (torch.arange(draws, device=hist.device)[None]+torch.rand(b, draws, device=hist.device))/draws
            if self.cfg.flow_unknown_planes == 'own_path':
                # Unknown planes stay in self-attention as at tracing, heading for the zero-start path.
                fill, attended = (points[..., :2]/self.sigma)[:, None].to(noise.dtype), None
            else:
                fill, attended = noise, known
            end = torch.where(known[:, None, :, None], (target/self.sigma)[:, None], fill)
            y = (1-times[..., None, None])*noise+times[..., None, None]*end
            predicted = self.velocity_field(ctx, y, times, attended)
            error = (predicted-(end-noise)).square()
            mask = known[:, None, :, None].expand_as(error)
            result['flow_per_state'] = torch.where(mask, error, 0.).sum((1, 2, 3))/mask.sum((1, 2, 3)).clamp_min(1)
        return result

    def select_prediction(self, output, confidence_threshold=.5, n_commit=None):
        from .model import select_refinement
        margin = self.cfg.flow_sample_threshold-confidence_threshold
        if self.cfg.flow_samples and margin > 0:
            first = 1 if self.cfg.flow_zero_start else 0
            confidence = output['refinement_confidence']
            samples = confidence[:, first:]
            barred = torch.where(samples >= self.cfg.flow_sample_threshold, samples, (samples-margin).clamp_min(0))
            output = dict(output, refinement_confidence=torch.cat((confidence[:, :first], barred), 1))
        return select_refinement(output, self.cfg, confidence_threshold, n_commit,
                                 retry=self.cfg.flow_selection == 'retry')

    def forward(self, x, hist, hmask, confidence_threshold=.5, n_commit=None):
        return self.select_prediction(self.training_forward(x, hist, hmask, confidence_threshold),
                                      confidence_threshold, n_commit)
