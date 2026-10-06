"""Flow matching over lateral paths: the generator mixin and its options (the 'flow' model, models/crop_transformer.py)."""
from dataclasses import dataclass
import math

import torch

from .model import select_refinement
from .survival_confidence import survival_predictions


@dataclass
class FlowOptions:
    """Flow fields mixed into the flow model's configuration."""
    flow_steps: int = 4
    flow_draws: int = 64
    flow_sigma: tuple = ()
    # Proposals beyond the zero-start path, from Gaussian starts (flow_sample_scale, in residual-scale
    # units). Training integrates and scores every proposal (the scorer learns sampled paths); tracing
    # takes a later proposal only when no earlier one is accepted (select_refinement). 0: one path.
    flow_samples: int = 4
    flow_sample_scale: float = 1.
    # Soft bound of the adaLN modulation: shift and scale M*tanh(x/M), branch gates tanh (Lumina-Next); 0 unbounded.
    # Unbounded, the time modulation drifted to scales of ~70 and training diverged; 4 lies above the healthy range.
    flow_modulation_bound: float = 4.
    # Lower bound of the fitted residual scales (voxels), i.e. of the noise prior's width.
    flow_sigma_floor: float = 3.
    # Pseudo-Huber loss on the velocity residual, per path point in residual-scale units: c²(√(1+d²/c²)−1), equal to
    # half the squared error below about flow_huber_c and linear above it, so outlying targets pull less.
    flow_huber_c: float = 1.

    def __post_init__(self):
        super().__post_init__()
        self.flow_sigma = tuple(tuple(row) for row in self.flow_sigma)
        if self.recurrent_refinement_steps != 0:
            raise ValueError('Flow uses midpoint integration, not regression retries')
        if min(self.flow_steps, self.flow_draws) < 1:
            raise ValueError('Positive flow steps and draws required')
        if type(self.flow_samples) is not int or self.flow_samples < 0 or not (
                math.isfinite(self.flow_sample_scale) and self.flow_sample_scale > 0):
            raise ValueError('Flow samples must be a nonnegative integer with a finite positive scale')
        if not (math.isfinite(self.flow_modulation_bound) and self.flow_modulation_bound >= 0):
            raise ValueError('The flow modulation bound must be finite and nonnegative (0: unbounded)')
        if not (math.isfinite(self.flow_sigma_floor) and self.flow_sigma_floor >= 1):
            raise ValueError('The flow scale floor must be finite and at least one voxel')
        if not (math.isfinite(self.flow_huber_c) and self.flow_huber_c > 0):
            raise ValueError('The pseudo-Huber scale must be finite and positive')
        if self.flow_sigma and (len(self.flow_sigma) != self.n_future or any(
                len(row) != 2 or any(not math.isfinite(v) or v < self.flow_sigma_floor for v in row)
                for row in self.flow_sigma)):
            raise ValueError('Flow scales must contain two finite values >= the scale floor per path plane')


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
    count = torch.zeros(len(cfg.path_plane_values), dtype=torch.float64)
    total = torch.zeros(len(cfg.path_plane_values), 2, dtype=torch.float64)
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
        raise ValueError('Flow calibration needs at least two known targets per plane; increase training.flow_calibration_states')
    variance = (square/count[:, None]-(total/count[:, None]).square()).clamp_min(0)
    return tuple(map(tuple, variance.sqrt().clamp_min(cfg.flow_sigma_floor).tolist()))


class FlowMatching:
    """The flow-matching generator over an observation model: the zero-start path, optionally followed by sampled
    proposals; Gaussian/time draws train velocity. The model provides ``planes`` (the forward coordinates of planes
    1..n_future: the proposal's planes for scorer, gate, tracer and labels), ``context``, ``velocity_field(ctx, y, t)``
    -> (B, D, P, 2), ``hazard_logits(ctx, points)`` and ``proposal_output``."""

    def init_flow(self, cfg):
        if not cfg.flow_sigma:
            raise ValueError('Fit flow scales before constructing the flow model')
        self.register_buffer('sigma', torch.tensor(cfg.flow_sigma, dtype=torch.float32), persistent=False)

    def to_points(self, y):
        z = self.planes.reshape(*([1]*(y.ndim-2)), -1, 1).expand(*y.shape[:-1], 1)
        return torch.cat((y*self.sigma, z), -1)

    def generate(self, ctx, hist, start=None):
        """Midpoint integration from ``start`` (B, D, P, 2), by default the zero path (D=1), without gradients.
        Returns the (B, D, P, 3) points at every solver step."""
        y = hist.new_zeros(len(hist), 1, len(self.planes), 2) if start is None else start
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
        """(B, D, P, 2): the zero path, then scaled Gaussian starts.

        ``keys`` (B,) int64, given by the tracer, seeds each row's starts from its own generator, so a
        decision's noise depends on its key only (not on batch neighbours, batch size or precision).
        Without keys (training) the global RNG draws them.
        """
        zero = hist.new_zeros(len(hist), 1, len(self.planes), 2)
        if not self.cfg.flow_samples:
            return zero
        shape = (self.cfg.flow_samples, len(self.planes), 2)
        if keys is None:
            noise = torch.randn(len(hist), *shape, device=hist.device)
        else:
            noise = torch.stack([torch.randn(*shape, device=hist.device,
                                             generator=torch.Generator(hist.device).manual_seed(int(key)))
                                 for key in keys.tolist()])
        noise = self.cfg.flow_sample_scale*noise.to(zero.dtype)
        return torch.cat((zero, noise), 1)

    def training_forward(self, x, hist, hmask, threshold, targets=None):
        ctx = self.context(x, hist, hmask)
        starts = self.proposal_starts(hist, x.get('flow_noise_keys'))
        generated = self.generate(ctx, hist, starts)
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
        if targets is not None:
            target, known = flow_targets(targets, self.cfg)
            b, draws = len(hist), self.cfg.flow_draws
            noise = targets.get('flow_noise')
            if noise is None:
                noise = torch.randn(b, draws, len(self.planes), 2, device=hist.device)
            times = targets.get('flow_times')
            if times is None:
                times = (torch.arange(draws, device=hist.device)[None]+torch.rand(b, draws, device=hist.device))/draws
            # Unknown planes stay in self-attention as at tracing, heading for the zero-start path.
            fill = (points[..., :2]/self.sigma)[:, None].to(noise.dtype)
            end = torch.where(known[:, None, :, None], (target/self.sigma)[:, None], fill)
            y = (1-times[..., None, None])*noise+times[..., None, None]*end
            predicted = self.velocity_field(ctx, y, times)
            residual = predicted-(end-noise)
            # Each coordinate carries its point's c²(√(1+d²/c²)−1) ≈ d²/2, so small residuals give half the squared
            # error and the pull per point is bounded by 2c beyond about c.
            c = self.cfg.flow_huber_c
            error = (c*c*((1+residual.square().sum(-1, keepdim=True)/(c*c)).sqrt()-1)).expand_as(residual)
            mask = known[:, None, :, None].expand_as(error)
            result['flow_per_state'] = torch.where(mask, error, 0.).sum((1, 2, 3))/mask.sum((1, 2, 3)).clamp_min(1)
        return result

    def select_prediction(self, output, confidence_threshold=.5, n_commit=None):
        return select_refinement(output, self.cfg, confidence_threshold, n_commit)

    def forward(self, x, hist, hmask, confidence_threshold=.5, n_commit=None):
        return self.select_prediction(self.training_forward(x, hist, hmask, confidence_threshold),
                                      confidence_threshold, n_commit)
