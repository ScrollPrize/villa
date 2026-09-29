"""Continuous trajectory prediction with direct access to identity memory.

One transformer decoder jointly predicts all future lateral coordinates. Every
layer reads image/history tokens, persistent slots, the immutable seed, and the
latest observation. Optional recurrent passes refine the curve with the same
decoder, refreshing spatial queries while reusing per-layer memory projections.
"""
from dataclasses import replace
import math

import torch
from torch import nn

from .feature_memory import FeatureMemory
from .model import DirectFollower, TRAJECTORY_MEMORY_ARCHITECTURE


class TrajectoryMemoryFollower(DirectFollower):
    def __init__(self, cfg):
        if cfg.memory_version != 4 or cfg.correction:
            raise ValueError('TrajectoryMemoryFollower requires memory v4 without correction')
        super().__init__(replace(cfg, memory_version=2, recurrent_refinement_steps=0))
        self.cfg, self.architecture = cfg, TRAJECTORY_MEMORY_ARCHITECTURE
        self.recurrent_memory = FeatureMemory(cfg, self.encoder.token_xyz)
        if cfg.recurrent_refinement_steps:
            width = 27*(cfg.channels+1)+cfg.hidden+1+3
            self.refinement_fusion = nn.Sequential(nn.Linear(width, cfg.hidden), nn.SiLU(),
                                                  nn.Linear(cfg.hidden, cfg.hidden))
            self.refinement_stage = nn.Embedding(cfg.recurrent_refinement_steps, cfg.hidden)
            self.refinement_delta = nn.Linear(cfg.hidden, 2)
            nn.init.zeros_(self.refinement_stage.weight)
            nn.init.zeros_(self.refinement_delta.weight)
            nn.init.zeros_(self.refinement_delta.bias)

    def context(self, x, hist, hmask):
        ctx = super().context(x, hist, hmask)
        if 'query_frame' in x:
            ctx['query_frame'] = x['query_frame']
        return ctx

    def forward(self, x, hist, hmask, queries=None, candidates=None, memory=None):
        # A cold, externally supplied history may have a remote seed. Encode
        # that seed crop once; streamed training starts at the seed itself.
        if 'feature_seed_x' in x:
            seed_x = x['feature_seed_x']
            empty = torch.zeros_like(hmask)
            seed_ctx = self.context(seed_x, torch.zeros_like(hist), empty)
            memory = self.recurrent_memory.observe_features(seed_ctx['deep'], seed_x, memory)
        ctx = self.context(x, hist, hmask)
        ctx['recurrent'] = self.recurrent_memory.observe_features(ctx['deep'], x, memory)
        out = self.predict(ctx, hist, candidates)
        out.update({'memory_'+k: v for k, v in ctx['recurrent'].items()})
        out.update(reference_embedding=ctx['reference_embedding'], reference_mask=ctx['reference_mask'])
        if queries is not None:
            from .model import sample_features
            import torch.nn.functional as F
            values, support = sample_features(ctx.get('fine_fp32', ctx['fine']), queries, self.cfg.fine)
            values = values.to(ctx['fine'].dtype)
            out.update(query_embedding=F.normalize(self.embedding(values), dim=-1), query_support=support)
        return out

    def predict(self, ctx, hist, candidates=None):
        cfg, state = self.cfg, ctx['recurrent']
        # Express retained spatial evidence in the actual crop frame, including roll.
        query_state = dict(state, frame=ctx.get('query_frame', state['frame']))
        identity, identity_padding = self.recurrent_memory.read_tokens(query_state)
        memory = torch.cat((ctx['memory'], identity.to(ctx['memory'].dtype)), 1)
        padding = torch.cat((ctx['padding'], identity_padding), 1)

        # These locations initialize feature queries, not output constraints.
        # Forward distance distinguishes the queries; self-attention couples
        # them and cross-attention can retrieve evidence anywhere in the crop.
        reference = hist.new_zeros(len(hist), cfg.n_future, 3)
        reference[..., 2] = self.planes
        query = self.query(self.query_features(ctx, reference))
        if cfg.recurrent_refinement_steps:
            # These tensors are shared only within this decision and remain attached.
            projected = [layer.project_memory(memory) for layer in self.decoder.layers]
            decoded = self.decode_cached(query, projected, padding)
        else:
            decoded = self.decoder(query, memory, memory_key_padding_mask=padding)
        lateral = cfg.lateral_limit*torch.tanh(self.coordinates(decoded).float())
        first_limit = math.sqrt(max(0., cfg.max_recovery_distance**2-cfg.future_step**2))
        first = lateral[:, :1]
        first = first*(first_limit/first.norm(dim=-1, keepdim=True).clamp_min(1e-8)).clamp(max=1.)
        lateral = torch.cat((first, lateral[:, 1:]), 1)
        points = torch.cat((lateral, reference[..., 2:]), -1)
        initial = points
        refinements = [points]
        for stage in range(cfg.recurrent_refinement_steps):
            evidence = self.evidence(ctx, points, 'refinement')
            refreshed = self.refinement_fusion(torch.cat((evidence, points/16.), -1))
            query = decoded+refreshed+self.refinement_stage.weight[stage].to(decoded.dtype)
            decoded = self.decode_cached(query, projected, padding)
            delta = self.refinement_delta(decoded).float().tanh()*(cfg.recurrent_refinement_limit/math.sqrt(2))
            lateral = (points[..., :2]+delta).clamp(-cfg.lateral_limit, cfg.lateral_limit)
            first = lateral[:, :1]
            first = first*(first_limit/first.norm(dim=-1, keepdim=True).clamp_min(1e-8)).clamp(max=1.)
            lateral = torch.cat((first, lateral[:, 1:]), 1)
            points = torch.cat((lateral, reference[..., 2:]), -1)
            refinements.append(points)
        logits = self.confidence_logits(ctx, decoded, points)
        out = dict(points=points, initial_points=initial, refinement_points=torch.stack(refinements, 1),
                   confidence_logits=logits, confidence=logits.sigmoid().cummin(-1).values)
        if candidates is not None:
            out['candidate_confidence_logits'] = torch.stack([
                self.confidence_logits(ctx, decoded, curve) for curve in candidates.unbind(1)], 1)
        return out

    def decode_cached(self, query, projected, padding):
        for layer, kv in zip(self.decoder.layers, projected):
            query = layer.forward_cached(query, kv, padding)
        return self.decoder.norm(query)
