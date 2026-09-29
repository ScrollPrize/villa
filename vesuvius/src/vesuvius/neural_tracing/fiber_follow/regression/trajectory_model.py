"""Continuous trajectory prediction with direct access to identity memory.

One transformer decoder jointly predicts all future lateral coordinates. Every
layer reads image/history tokens, persistent slots, the immutable seed, and the
latest observation. There is no lattice selection or coordinate refinement.
"""
from dataclasses import replace
import math

import torch

from .feature_memory import FeatureMemory
from .model import DirectFollower, TRAJECTORY_MEMORY_ARCHITECTURE


class TrajectoryMemoryFollower(DirectFollower):
    def __init__(self, cfg):
        if cfg.memory_version != 4 or cfg.correction:
            raise ValueError('TrajectoryMemoryFollower requires memory v4 without correction')
        super().__init__(replace(cfg, memory_version=2))
        self.cfg, self.architecture = cfg, TRAJECTORY_MEMORY_ARCHITECTURE
        self.recurrent_memory = FeatureMemory(cfg, self.encoder.token_xyz)

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
            values, support = sample_features(ctx['fine'], queries, self.cfg.fine)
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
        decoded = self.decoder(self.query(self.query_features(ctx, reference)), memory,
                               memory_key_padding_mask=padding)
        lateral = cfg.lateral_limit*torch.tanh(self.coordinates(decoded).float())
        first_limit = math.sqrt(max(0., cfg.max_recovery_distance**2-cfg.future_step**2))
        first = lateral[:, :1]
        first = first*(first_limit/first.norm(dim=-1, keepdim=True).clamp_min(1e-8)).clamp(max=1.)
        lateral = torch.cat((first, lateral[:, 1:]), 1)
        points = torch.cat((lateral, reference[..., 2:]), -1)
        logits = self.confidence_logits(ctx, decoded, points)
        out = dict(points=points, initial_points=points, refinement_points=points[:, None],
                   confidence_logits=logits, confidence=logits.sigmoid().cummin(-1).values)
        if candidates is not None:
            out['candidate_confidence_logits'] = torch.stack([
                self.confidence_logits(ctx, decoded, curve) for curve in candidates.unbind(1)], 1)
        return out
