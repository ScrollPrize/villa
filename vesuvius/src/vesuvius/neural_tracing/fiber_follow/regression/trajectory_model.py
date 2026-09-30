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
import torch.nn.functional as F

from .feature_memory import FeatureMemory
from .model import DirectFollower, TRAJECTORY_MEMORY_ARCHITECTURE


class TrajectoryMemoryFollower(DirectFollower):
    def __init__(self, cfg):
        if cfg.memory_version != 4 or cfg.correction:
            raise ValueError('TrajectoryMemoryFollower requires memory v4 without correction')
        super().__init__(replace(cfg, memory_version=2, recurrent_refinement_steps=0, feature_switch_crop_fraction=-1.))
        self.cfg, self.architecture = cfg, TRAJECTORY_MEMORY_ARCHITECTURE
        self.recurrent_memory = FeatureMemory(cfg, self.encoder.token_xyz)
        if cfg.feature_memory_revision == 2:
            from .detailed_memory import DetailedFeatureMemory
            self.recurrent_memory = DetailedFeatureMemory(cfg, self.encoder.token_xyz)
            self.encoder_memory_gate = nn.Parameter(torch.tensor(-4.))
            self.encoder_memory_projection = nn.Linear(cfg.hidden, cfg.hidden)
            width = 27*(cfg.channels+1)+cfg.hidden+1+3
            self.confidence_memory_query = nn.Sequential(nn.Linear(width, cfg.hidden), nn.LayerNorm(cfg.hidden))
            self.confidence_memory_attention = nn.MultiheadAttention(cfg.hidden, cfg.heads, dropout=0., batch_first=True)
            self.confidence_memory_head = nn.Sequential(nn.Linear(3*cfg.hidden, cfg.hidden), nn.SiLU(), nn.Linear(cfg.hidden, 1))
            nn.init.zeros_(self.confidence_memory_head[-1].weight)
            nn.init.zeros_(self.confidence_memory_head[-1].bias)
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
            memory, _ = self.observe_context(seed_ctx, seed_x, torch.zeros_like(hist), empty, memory)
        ctx = self.context(x, hist, hmask)
        ctx['recurrent'], observation = self.observe_context(ctx, x, hist, hmask, memory)
        out = self.predict(ctx, hist, candidates)
        if observation is not None:
            out.update(observation_tokens=observation[0], observation_xyz=observation[1], observation_valid=observation[2])
        out.update({'memory_'+k: v for k, v in ctx['recurrent'].items()})
        out.update(reference_embedding=ctx['reference_embedding'], reference_mask=ctx['reference_mask'])
        if queries is not None:
            from .model import sample_features
            import torch.nn.functional as F
            values, support = sample_features(ctx.get('fine_fp32', ctx['fine']), queries, self.cfg.fine)
            values = values.to(ctx['fine'].dtype)
            out.update(query_embedding=F.normalize(self.embedding(values), dim=-1), query_support=support)
        return out

    def observation_features(self, x, hist, hmask):
        """Re-encode a selected replay crop without decoding or reading memory."""
        ctx = self.context(x, hist, hmask)
        return self.recurrent_memory.extract(ctx['deep'], ctx.get('fine_fp32', ctx['fine']), hist, hmask)

    def observe_context(self, ctx, x, hist, hmask, memory):
        if self.cfg.feature_memory_revision == 1:
            return self.recurrent_memory.observe_features(ctx['deep'], x, memory), None
        observation = self.recurrent_memory.extract(ctx['deep'], ctx.get('fine_fp32', ctx['fine']), hist, hmask)
        state, retrieved = self.recurrent_memory.observe_tokens(*observation, x, memory)
        # A coarse spatial read broadcasts identity context to the deep lattice.
        # This avoids 39k queries over the entire historical cache. Stored
        # observation features above remain independent of this conditioning.
        n = self.recurrent_memory.coarse_count
        condition = self.encoder_memory_projection(retrieved[:, :n].to(ctx['deep'].dtype))
        condition = condition.transpose(1, 2).reshape(len(hist), self.cfg.hidden, *self.cfg.feature_memory_grid)
        condition = F.interpolate(condition, size=ctx['deep'].shape[-3:], mode='trilinear', align_corners=False)
        deep = ctx['deep']+self.encoder_memory_gate.sigmoid()*condition
        image_count = deep.shape[2]*deep.shape[3]*deep.shape[4]
        ctx['deep'] = deep
        if 'deep_fp32' in ctx:
            ctx['deep_fp32'] = deep.float()
        ctx['memory'] = torch.cat((deep.flatten(2).transpose(1, 2), ctx['memory'][:, image_count:]), 1)
        return state, observation

    def memory_confidence(self, ctx, spatial, points):
        query = self.confidence_memory_query(torch.cat((spatial, points/16.), -1))
        tokens = ctx['identity_tokens'].to(query.dtype)
        retrieved = self.confidence_memory_attention(query, tokens, tokens,
            key_padding_mask=ctx['identity_padding'], need_weights=False)[0]
        return self.confidence_memory_head(torch.cat((query, retrieved, query*retrieved), -1))[..., 0].float()

    def predict(self, ctx, hist, candidates=None):
        cfg = self.cfg
        memory, padding = self.decoder_memory(ctx)

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
        return self.finish_prediction(ctx, decoded, points, projected if cfg.recurrent_refinement_steps else None,
                                      padding, candidates)

    def decoder_memory(self, ctx):
        cfg, state = self.cfg, ctx['recurrent']
        # Express retained spatial evidence in the actual crop frame, including roll.
        query_state = dict(state, frame=ctx.get('query_frame', state['frame']))
        identity, identity_padding = self.recurrent_memory.read_tokens(query_state)
        if cfg.feature_memory_revision == 2:
            ctx.update(identity_tokens=identity, identity_padding=identity_padding)
        memory = torch.cat((ctx['memory'], identity.to(ctx['memory'].dtype)), 1)
        padding = torch.cat((ctx['padding'], identity_padding), 1)

        return memory, padding

    def finish_prediction(self, ctx, decoded, points, projected, padding, candidates=None):
        cfg = self.cfg
        first_limit = math.sqrt(max(0., cfg.max_recovery_distance**2-cfg.future_step**2))
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
            points = torch.cat((lateral, initial[..., 2:]), -1)
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
