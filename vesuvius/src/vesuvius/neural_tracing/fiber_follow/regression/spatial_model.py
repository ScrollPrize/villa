"""Memory v3: retrieve the original fiber spatially, then decode a connected route.

Route logits receive direct localization supervision. Discrete max-sum decoding
preserves modes; it never averages distinct fibers. The existing fine refiner
only moves within the selected lattice cell. No annotations enter this module.
"""
from dataclasses import replace
import math

import torch
from torch import nn
import torch.nn.functional as F

from .identity_memory import IdentityMemory
from .model import DirectFollower, SPATIAL_MEMORY_ARCHITECTURE, sample_features


def route_lattice(cfg):
    n = 2*math.ceil(cfg.lateral_limit/cfg.route_grid_step)+1
    axis = torch.linspace(-cfg.lateral_limit, cfg.lateral_limit, n)
    y, x = torch.meshgrid(axis, axis, indexing='ij')
    return torch.stack((x, y), -1).flatten(0, 1), n


def route_neighbors(n, radius):
    y, x = torch.meshgrid(torch.arange(n), torch.arange(n), indexing='ij')
    offsets = torch.cartesian_prod(torch.arange(-radius, radius+1), torch.arange(-radius, radius+1))
    yy, xx = y.flatten()[:, None]+offsets[:, 0], x.flatten()[:, None]+offsets[:, 1]
    valid = (yy >= 0) & (yy < n) & (xx >= 0) & (xx < n)
    return (yy.clamp(0,n-1)*n+xx.clamp(0,n-1)), valid, offsets.square().sum(-1).float()


@torch.no_grad()
def connected_route(logits, xy, neighbors, valid, costs, first_limit):
    """Max-sum lattice path; ties resolve deterministically by flattened order."""
    score = logits[:, 0].float().masked_fill(xy.square().sum(-1) > first_limit**2, -torch.inf)
    pointers = []
    for plane in range(1, logits.shape[1]):
        previous = score[:, neighbors]-costs
        previous = previous.masked_fill(~valid, -torch.inf)
        best, index = previous.max(-1)
        pointers.append(neighbors[None].expand(len(logits),-1,-1).gather(-1,index[...,None])[...,0])
        score = best+logits[:, plane].float()
    current = score.argmax(-1)
    path = [current]
    for pointer in reversed(pointers):
        current = pointer.gather(1, current[:, None])[:, 0]
        path.append(current)
    return torch.stack(path[::-1], 1)


class SpatialMemoryFollower(DirectFollower):
    def __init__(self, cfg):
        # Reuse the encoder, decoder, confidence and refiner implementations.
        super().__init__(replace(cfg, memory_version=2))
        self.cfg, self.architecture = cfg, SPATIAL_MEMORY_ARCHITECTURE
        self.recurrent_memory = IdentityMemory(cfg)
        self.route_query = nn.Sequential(nn.Linear(cfg.channels+3, cfg.hidden), nn.LayerNorm(cfg.hidden), nn.SiLU())
        self.route_attention = nn.MultiheadAttention(cfg.hidden, cfg.heads, dropout=0., batch_first=True)
        self.route_score = nn.Sequential(nn.Linear(3*cfg.hidden, cfg.hidden), nn.SiLU(), nn.Linear(cfg.hidden, 1))
        xy, self.route_width = route_lattice(cfg)
        neighbors, valid, costs = route_neighbors(self.route_width, cfg.route_transition_radius)
        self.register_buffer('route_xy', xy, persistent=False)
        self.register_buffer('route_neighbors', neighbors, persistent=False)
        self.register_buffer('route_neighbor_valid', valid, persistent=False)
        self.register_buffer('route_costs', costs*cfg.route_transition_cost, persistent=False)

    def context(self, x, hist, hmask):
        ctx = super().context(x,hist,hmask)
        if 'route_frame' in x:
            ctx['route_frame'] = x['route_frame']
        return ctx

    def predict(self, ctx, hist, candidates=None):
        cfg, b = self.cfg, len(hist)
        xy = self.route_xy.to(hist)
        grid = torch.cat((xy[None].expand(cfg.n_future,-1,-1),
                          self.planes[:,None,None].expand(-1,len(xy),1)), -1)
        points = grid.flatten(0,1)[None].expand(b,-1,-1)
        values, _ = sample_features(ctx['fine'], points, cfg.fine)
        query = self.route_query(torch.cat((values,points/16), -1))
        state = ctx['recurrent']
        # Memory patches use canonical tangent frames; the current crop can
        # have a different roll. Express seed pose in the actual query frame.
        query_state = dict(state,frame=ctx.get('route_frame',state['frame']))
        tokens, padding = self.recurrent_memory.read_tokens(query_state)
        retrieved = self.route_attention(query, tokens.to(query.dtype), tokens.to(query.dtype),
                                          key_padding_mask=padding, need_weights=False)[0]
        logits = self.route_score(torch.cat((query,retrieved,query*retrieved), -1))
        logits = logits.reshape(b,cfg.n_future,-1).float()
        first_limit = math.sqrt(max(0.,cfg.max_recovery_distance**2-cfg.future_step**2))
        indices = connected_route(logits,xy,self.route_neighbors,self.route_neighbor_valid,self.route_costs,first_limit)
        lateral = xy[indices]
        points = torch.cat((lateral,self.planes[None,:,None].expand(b,-1,-1)), -1)
        initial = points
        # The local decoder can use the recent observation, but identity retrieval
        # above has access only to persistent slots and the immutable seed.
        local_tokens = torch.cat((ctx['memory'],ctx['recurrent']['recent'].to(ctx['memory'].dtype)), 1)
        local_padding = F.pad(ctx['padding'],(0,8),value=False)
        decoded = self.decoder(self.query(self.query_features(ctx,points)),local_tokens,
                               memory_key_padding_mask=local_padding)
        refinements = [points]
        half_cell = cfg.lateral_limit/(self.route_width-1)
        for _ in range(cfg.correction_steps if cfg.correction else 0):
            evidence = self.evidence(ctx,points,'correction')
            correction = self.correction_head(torch.cat((decoded,evidence,points/16),-1)).float().tanh()
            lateral = lateral+correction*(cfg.correction_limit/math.sqrt(2))
            lateral = torch.maximum(torch.minimum(lateral,initial[...,:2]+half_cell),initial[...,:2]-half_cell)
            lateral = lateral.clamp(-cfg.lateral_limit,cfg.lateral_limit)
            first = lateral[:,:1]
            first = first*(first_limit/first.norm(dim=-1,keepdim=True).clamp_min(1e-8)).clamp(max=1.)
            lateral = torch.cat((first,lateral[:,1:]),1)
            points = torch.cat((lateral,initial[...,2:]),-1)
            refinements.append(points)
        # Probability in the selected cell's neighborhood exposes competing
        # spatial modes to the commit decision. It is not a calibrated guarantee.
        nearby = (xy[None,None]-xy[indices][:,:,None]).abs().amax(-1) <= 2*half_cell+1e-5
        support = (logits.softmax(-1)*nearby).sum(-1).clamp(1e-6,1-1e-6)
        raw_logits = self.confidence_logits(ctx,decoded,points)
        probability = (raw_logits.sigmoid()*support).clamp(1e-6,1-1e-6)
        confidence_logits = torch.logit(probability)
        out = dict(points=points,initial_points=initial,refinement_points=torch.stack(refinements,1),
                   confidence_logits=confidence_logits,confidence=probability.cummin(-1).values,
                   route_logits=logits,route_indices=indices,route_support=support)
        if candidates is not None:
            out['candidate_confidence_logits'] = torch.stack([
                self.confidence_logits(ctx,decoded,curve) for curve in candidates.unbind(1)],1)
        return out
