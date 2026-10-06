"""Frozen pre-cleanup flow follower ('flow_matching'), for inference and tracing with the v4-extension and v5 checkpoints.

This is the model of commit 0be84c237 (the code flow_v4_ext_paris4afv and flow_v5_scratch_huber_d144 were trained with),
reduced to what those checkpoints use: the shared patch-encoder context and segment survival scorer
(legacy/patch_model.py), the flow decoder (input or adaLN time conditioning) and the proposal selection. No memory,
identity, whole-crop, tube or regression paths, and no training. Parameter and buffer names are unchanged, so the
checkpoints' EMA weights load strictly. Nothing here is used by, or changes, the current models (models/); the current
tracer drives this model through its usual call: ``model(x, hist, hmask, confidence_threshold, n_commit)``.

    model, crop, n_history, spec, ck = load_checkpoint(PATH, device)
"""
from dataclasses import dataclass, fields

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from vesuvius.neural_tracing.fiber_follow.legacy.patch_model import (
    REQUIRED as PATCH_REQUIRED, LegacyPatchConfig, LegacyPatchFollower, check_recorded, decoder_layer, load_legacy,
    select_refinement)
from vesuvius.neural_tracing.fiber_follow.models.flow import time_embedding
from vesuvius.neural_tracing.fiber_follow.models.survival_confidence import survival_predictions
from vesuvius.neural_tracing.fiber_follow.tracing.policy import DEFAULT_CONFIDENCE

# Pre-cleanup model fields these checkpoints may hold only at these values (the code below implements no other).
REQUIRED = dict(PATCH_REQUIRED, model_type='flow_matching', recurrent_refinement_steps=0, query_scale=None)


@dataclass
class LegacyFlowConfig(LegacyPatchConfig):
    model_type: str = 'flow_matching'
    recurrent_refinement_steps: int = 0
    flow_steps: int = 4
    flow_sigma: tuple = ()
    flow_samples: int = 0
    flow_sample_scale: float = 1.
    flow_time_conditioning: str = 'input'
    flow_selection: str = 'retry'
    flow_zero_start: bool = True
    flow_sample_threshold: float = 0.

    def __post_init__(self):
        super().__post_init__()
        self.flow_sigma = tuple(tuple(row) for row in self.flow_sigma)
        if self.flow_time_conditioning not in ('input', 'adaln') or self.flow_selection not in ('retry', 'best'):
            raise ValueError("Legacy flow: time conditioning 'input'/'adaln', selection 'retry'/'best'")
        if len(self.flow_sigma) != self.n_future or not self.flow_zero_start and not self.flow_samples:
            raise ValueError('Legacy flow: one residual scale per plane; dropping the zero start needs samples')


def config_from_checkpoint(ck):
    """The legacy configuration of a pre-cleanup flow checkpoint, refusing features this code does not implement."""
    recorded = ck['model_cfg']
    check_recorded(recorded, REQUIRED, 'flow')
    kept = {f.name for f in fields(LegacyFlowConfig)}
    return LegacyFlowConfig(**{k: v for k, v in recorded.items() if k in kept})


def load_checkpoint(path, device='cuda'):
    """(EMA model in eval mode, crop, n_history, volume spec, checkpoint): the tracing CLI's checkpoint loader."""
    return load_legacy(path, device, lambda ck: LegacyFlowFollower(config_from_checkpoint(ck)))


class LegacyFlowFollower(LegacyPatchFollower):
    """Zero-start path plus optional Gaussian-start proposals, midpoint-integrated and scored; selection as tracing."""

    def __init__(self, cfg):
        super().__init__(cfg)
        h = cfg.hidden
        planes = cfg.future_step*np.arange(1, cfg.n_future+1, dtype=np.float64)  # as the original path_plane_values
        self.register_buffer('planes', torch.tensor(planes, dtype=torch.float32), persistent=False)
        self.register_buffer('sigma', torch.tensor(cfg.flow_sigma, dtype=torch.float32), persistent=False)
        self.plane_scale = cfg.n_future*cfg.future_step
        self.query = nn.Sequential(nn.Linear(h+4, h), nn.SiLU(), nn.Linear(h, h))
        self.time = nn.Sequential(nn.Linear(64, h), nn.SiLU(), nn.Linear(h, h))
        self.decoder = nn.TransformerDecoder(decoder_layer(cfg, cfg.decoder_ffn), cfg.decoder_layers, norm=nn.LayerNorm(h))
        self.velocity = nn.Linear(h, 2)
        if cfg.flow_time_conditioning == 'adaln':
            self.time_modulation = nn.ModuleList(nn.Linear(h, 12*h) for _ in range(cfg.decoder_layers))
            self.output_modulation = nn.Linear(h, 2*h)

    def context(self, x, hist, hmask):
        ctx = super().context(x, hist, hmask)
        ctx['generator_projected'] = [layer.project_memory(ctx['memory']) for layer in self.decoder.layers]
        return ctx

    def to_points(self, y):
        z = self.planes.reshape(*([1]*(y.ndim-2)), -1, 1).expand(*y.shape[:-1], 1)
        return torch.cat((y*self.sigma, z), -1)

    def velocity_field(self, ctx, y, t):
        b, draws, planes, _ = y.shape
        points = self.to_points(y)
        evidence, support = self.sample_local(ctx['deep'].float(), points.reshape(b, draws*planes, 3))
        query = self.query(torch.cat((evidence.reshape(b, draws, planes, -1).to(ctx['deep'].dtype), y,
                                      points[..., 2:]/self.plane_scale, support.reshape(b, draws, planes, 1)), -1))
        time = self.time(time_embedding(t))
        query = query+time[:, :, None]
        padding = torch.zeros(b*draws, planes, device=y.device, dtype=torch.bool)
        adaln = self.cfg.flow_time_conditioning == 'adaln'
        condition = F.silu(time) if adaln else None
        for index, (layer, kv) in enumerate(zip(self.decoder.layers, ctx['generator_projected'])):
            modulation = self.time_modulation[index](condition).unflatten(-1, (4, 3, self.cfg.hidden)) if adaln else None
            query = layer.forward_draws(query, kv, ctx['padding'], padding, modulation)
        output = self.decoder.norm(query)
        if adaln:
            shift, scale = self.output_modulation(condition)[:, :, None].chunk(2, -1)
            output = output*(1+scale)+shift
        return self.velocity(output).float()

    def generate(self, ctx, start):
        """Midpoint integration from ``start`` (B, D, P, 2): the (B, D, P, 3) points after the last step."""
        y = start
        with torch.no_grad():
            for step in range(self.cfg.flow_steps):
                t = y.new_full(y.shape[:2], step/self.cfg.flow_steps)
                v = self.velocity_field(ctx, y, t)
                y = y+self.velocity_field(ctx, y+v/(2*self.cfg.flow_steps), t+1/(2*self.cfg.flow_steps))/self.cfg.flow_steps
        return self.to_points(y)

    def proposal_starts(self, hist, keys=None):
        zero = hist.new_zeros(len(hist), 1, self.cfg.n_future, 2)
        if not self.cfg.flow_samples:
            return zero
        shape = (self.cfg.flow_samples, self.cfg.n_future, 2)
        if keys is None:
            noise = torch.randn(len(hist), *shape, device=hist.device)
        else:
            noise = torch.stack([torch.randn(*shape, device=hist.device, generator=torch.Generator(hist.device).manual_seed(int(key)))
                                 for key in keys.tolist()])
        noise = self.cfg.flow_sample_scale*noise.to(zero.dtype)
        return torch.cat((zero, noise), 1) if self.cfg.flow_zero_start else noise

    def proposals(self, x, hist, hmask):
        """Every proposal (zero start, then Gaussian starts), integrated and scored."""
        ctx = self.context(x, hist, hmask)
        starts = self.proposal_starts(hist, x.get('flow_noise_keys'))
        proposals = list(self.generate(ctx, starts).unbind(1))
        scores = [(hazards, *survival_predictions(hazards)) for hazards in (self.hazard_logits(ctx, p) for p in proposals)]
        valid = torch.ones(len(hist), device=hist.device, dtype=torch.bool)
        output = dict(initial_points=self.to_points(starts[:, 0]), refinement_points=torch.stack(proposals, 1),
                      refinement_mask=torch.stack([valid]*len(proposals), 1))
        output.update({'refinement_'+name: torch.stack([score[i] for score in scores], 1)
                       for i, name in enumerate(('hazard_logits', 'confidence_logits', 'confidence'))})
        return output

    def select_prediction(self, output, confidence_threshold=DEFAULT_CONFIDENCE, n_commit=None):
        """The tracer passes its threshold and commit window to models that define this (tracing/trace.py)."""
        return self.select_prediction(output, confidence_threshold, n_commit)

    def select_prediction(self, output, confidence_threshold=DEFAULT_CONFIDENCE, n_commit=None):
        """Proposal selection (sample bar, then retry or best). Its presence tells the tracer to pass the operating
        threshold and commit window."""
        margin = self.cfg.flow_sample_threshold-confidence_threshold
        if self.cfg.flow_samples and margin > 0:
            first = 1 if self.cfg.flow_zero_start else 0
            confidence = output['refinement_confidence']
            samples = confidence[:, first:]
            barred = torch.where(samples >= self.cfg.flow_sample_threshold, samples, (samples-margin).clamp_min(0))
            output = dict(output, refinement_confidence=torch.cat((confidence[:, :first], barred), 1))
        return select_refinement(output, self.cfg, confidence_threshold, n_commit, retry=self.cfg.flow_selection == 'retry')

    def forward(self, x, hist, hmask, confidence_threshold=DEFAULT_CONFIDENCE, n_commit=None):
        return self.select_prediction(self.proposals(x, hist, hmask), confidence_threshold, n_commit)
