"""Frozen pre-cleanup coordinate-regression follower ('coordinate_regression'), for inference and tracing only.

The patch-encoder regression model removed in the model cleanup (e.g. the mixed_ct_afv_forward35_v1 checkpoints),
reduced to what memory-free, future-plane checkpoints use: the shared patch-encoder context and segment survival scorer
(legacy/patch_model.py), the cross-attention path decoder, and recurrent refinement passes for proposals not yet
accepted through the gate plane. No training. Parameter and buffer names are unchanged, so the checkpoints' EMA weights
load strictly; nothing here is used by, or changes, the current models (models/).

    python -m vesuvius.neural_tracing.fiber_follow.legacy.infer --checkpoint CKPT --seed X,Y,Z --family H --out DIR
"""
from dataclasses import dataclass, fields

import torch
from torch import nn

from vesuvius.neural_tracing.fiber_follow.legacy.patch_model import (
    REQUIRED as PATCH_REQUIRED, LegacyPatchConfig, LegacyPatchFollower, check_recorded, decoder_layer, load_legacy,
    select_refinement)
from vesuvius.neural_tracing.fiber_follow.models.survival_confidence import survival_predictions
from vesuvius.neural_tracing.fiber_follow.tracing.policy import DEFAULT_CONFIDENCE, commit_prefix

REQUIRED = dict(PATCH_REQUIRED, model_type='coordinate_regression')


@dataclass
class LegacyCoordinateConfig(LegacyPatchConfig):
    model_type: str = 'coordinate_regression'
    query_scale: float | None = None  # forward-distance normalization of the decoder queries; None: n_future*future_step
    patch_radius: float = 1.
    recurrent_refinement_steps: int = 3

    @property
    def lateral_limit(self):
        return (self.fine.width-1)*self.fine.spacing/2-self.patch_radius


def config_from_checkpoint(ck):
    """The legacy configuration of a pre-cleanup regression checkpoint, refusing features this code does not implement."""
    recorded = ck['model_cfg']
    check_recorded(recorded, REQUIRED, 'coordinate regression')
    kept = {f.name for f in fields(LegacyCoordinateConfig)}
    return LegacyCoordinateConfig(**{k: v for k, v in recorded.items() if k in kept})


def load_checkpoint(path, device='cuda'):
    """(EMA model in eval mode, crop, n_history, volume spec, checkpoint): the tracing CLI's checkpoint loader."""
    return load_legacy(path, device, lambda ck: LegacyCoordinateFollower(config_from_checkpoint(ck)))


class LegacyCoordinateFollower(LegacyPatchFollower):
    """Plane coordinates from the path decoder, rescored and refined while not accepted through the gate plane."""

    def __init__(self, cfg):
        super().__init__(cfg)
        h = cfg.hidden
        self.register_buffer('planes', torch.arange(1, cfg.n_future+1).float()*cfg.future_step, persistent=False)
        self.query_scale = cfg.query_scale or cfg.n_future*cfg.future_step
        self.query = nn.Sequential(nn.Linear(h+1, h), nn.SiLU(), nn.Linear(h, h))
        self.decoder = nn.TransformerDecoder(decoder_layer(cfg, cfg.decoder_ffn), cfg.decoder_layers, norm=nn.LayerNorm(h))
        self.coordinates = nn.Linear(h, 2)
        if cfg.recurrent_refinement_steps:
            # Spatial evidence, previous coordinates, detached failure/survival.
            self.refinement_fusion = nn.Sequential(nn.Linear(h+1+3+2, h), nn.SiLU(), nn.Linear(h, h))
            self.refinement_stage = nn.Embedding(cfg.recurrent_refinement_steps, h)

    def decode(self, ctx, query):
        """Decoder states and plane points: lateral offsets bounded to the crop, the first within the recovery limit."""
        cfg = self.cfg
        for layer, kv in zip(self.decoder.layers, ctx['generator_projected']):
            query = layer.forward_cached(query, kv, ctx['padding'])
        decoded = self.decoder.norm(query)
        lateral = cfg.lateral_limit*torch.tanh(self.coordinates(decoded).float())
        first_limit = max(0., cfg.max_recovery_distance**2-cfg.future_step**2)**.5
        first = lateral[:, :1]
        first = first*(first_limit/first.norm(dim=-1, keepdim=True).clamp_min(1e-8)).clamp(max=1.)
        lateral = torch.cat((first, lateral[:, 1:]), 1)
        return decoded, torch.cat((lateral, self.planes[None, :, None].expand(len(decoded), -1, -1)), -1)

    def scored(self, ctx, points):
        hazards = self.hazard_logits(ctx, points)
        return (hazards, *survival_predictions(hazards))

    def forward(self, x, hist, hmask, confidence_threshold=DEFAULT_CONFIDENCE, n_commit=None):
        cfg = self.cfg
        ctx = self.context(x, hist, hmask)
        ctx['generator_projected'] = [layer.project_memory(ctx['memory']) for layer in self.decoder.layers]
        reference = hist.new_zeros(len(hist), cfg.n_future, 3)
        reference[..., 2] = self.planes
        patches = self.sample_local(ctx['deep'], reference)[0].to(ctx['deep'].dtype)
        decoded, points = self.decode(ctx, self.query(torch.cat((patches, reference[..., 2:]/self.query_scale), -1)))
        score = self.scored(ctx, points)
        refinements, scores = [points], [score]
        active = torch.ones(len(hist), device=hist.device, dtype=torch.bool)
        valid = [active]
        for stage in range(cfg.recurrent_refinement_steps):
            # An acceptance through the gate plane ends this row's retries; an accepted row keeps its last attempt.
            counts, _ = commit_prefix(points, score[2][..., :cfg.gate_horizon].detach(), confidence_threshold,
                                      cfg.gate_horizon, cfg.max_recovery_distance)
            active = active & (counts < cfg.gate_horizon)
            if not active.any():
                break
            feedback = torch.stack((score[0].sigmoid(), score[2]), -1).detach()
            refreshed = self.refinement_fusion(torch.cat((self.evidence(ctx, points), points/16., feedback), -1))
            new_decoded, new_points = self.decode(ctx, decoded+refreshed+self.refinement_stage.weight[stage].to(decoded.dtype))
            new_score = self.scored(ctx, new_points)
            decoded = torch.where(active[:, None, None], new_decoded, decoded)
            points = torch.where(active[:, None, None], new_points, points)
            score = tuple(torch.where(active[:, None], new, old) for new, old in zip(new_score, score))
            refinements.append(points)
            scores.append(score)
            valid.append(active)
        output = dict(initial_points=refinements[0], refinement_points=torch.stack(refinements, 1),
                      refinement_mask=torch.stack(valid, 1))
        output.update({'refinement_'+name: torch.stack([s[i] for s in scores], 1)
                       for i, name in enumerate(('hazard_logits', 'confidence_logits', 'confidence'))})
        return self.select_prediction(output, confidence_threshold, n_commit)

    def select_prediction(self, output, confidence_threshold=DEFAULT_CONFIDENCE, n_commit=None):
        """Proposal selection. Its presence tells the tracer to pass the operating threshold and commit window."""
        return select_refinement(output, self.cfg, confidence_threshold, n_commit)
