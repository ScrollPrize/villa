"""V5: continuous spatial candidates and a connected route.

The sampling lattice indexes predictions; each node has a learned subpixel
offset before search. Labels never enter proposal selection or route decoding.
Route node scores are dense proposal log probabilities alone; the earlier
shortlist identity head was removed after it changed 2.6% of plane selections
with no net accuracy effect.
"""
from dataclasses import replace
import math

import torch
from torch import nn
import torch.nn.functional as F

from .model import CANDIDATE_MEMORY_ARCHITECTURE, sample_features, TOKEN_STRIDE, TOKEN_OFFSET
from .trajectory_model import TrajectoryMemoryFollower


def proposal_lattice(cfg):
    n = 2*math.ceil(cfg.lateral_limit/cfg.proposal_step)+1
    axis = torch.linspace(-cfg.lateral_limit, cfg.lateral_limit, n)
    y, x = torch.meshgrid(axis, axis, indexing='ij')
    return torch.stack((x, y), -1).flatten(0, 1), n


@torch.no_grad()
def shortlist(scores, positions, count, suppression, first_limit):
    """Greedy spatial suppression at continuous locations, with stable argmax ties.

Only plane zero is reachability-filtered. Padding is explicit when suppression
exhausts a plane; it never silently reintroduces a duplicate node.
"""
    work = scores.detach().float().clone()
    first = work[:, :1].masked_fill(positions[:, :1, :, :2].square().sum(-1) > first_limit**2, -torch.inf)
    work = torch.cat((first, work[:, 1:]), 1)
    indices, masks = [], []
    for _ in range(count):
        best, index = work.max(-1)
        valid = torch.isfinite(best)
        center = positions.gather(2, index[..., None, None].expand(-1, -1, 1, 3))
        close = (positions[..., :2]-center[..., :2]).square().sum(-1) <= suppression**2
        work = work.masked_fill(close, -torch.inf)
        # Explicitly exclude the selected index even at zero suppression.
        work.scatter_(-1, index[..., None], -torch.inf)
        indices.append(index)
        masks.append(valid)
    return torch.stack(indices, -1), torch.stack(masks, -1)


def direction_alignment(tangent, tensor):
    """Sign-invariant alignment; zero/isotropic direction mixtures have no vote.

Tensor channels are uu,vv,ff,uv,uf,vf in the crop frame. Trace carries
availability. Frobenius anisotropy reduces the weight of ambiguous mixtures.
"""
    trace = tensor[..., :3].sum(-1).clamp_min(0.)
    norm2 = tensor[..., :3].square().sum(-1)+2*tensor[..., 3:].square().sum(-1)
    anisotropy = ((1.5*norm2-.5*trace.square()).clamp_min(0.)/(trace.square()+1e-8)).clamp(0., 1.)
    a, b, c = tangent.unbind(-1)
    quadratic = (tensor[..., 0]*a*a+tensor[..., 1]*b*b+tensor[..., 2]*c*c
                 +2*(tensor[..., 3]*a*b+tensor[..., 4]*a*c+tensor[..., 5]*b*c))
    return (1-quadratic/trace.clamp_min(1e-8)).clamp(0., 1.)*trace.clamp(max=1.)*anisotropy


@torch.no_grad()
def continuous_route(scores, positions, valid, directions, direction_cost, turn_cost, first_limit):
    """Second-order max-sum route over all adjacent-plane candidate pairs.

No hard slope/curvature cutoff or lattice adjacency. The first segment alone
has the existing recovery bound. Returns an explicit disconnected flag if no
complete route exists; its diagnostic fallback must not authorize continuation.
"""
    scores, positions, directions = scores.float(), positions.float(), directions.float()
    b, planes, k = scores.shape
    nodes = scores.masked_fill(~valid, -torch.inf)
    first_valid = positions[:, 0].square().sum(-1) <= first_limit**2
    head = F.normalize(positions[:, 0], dim=-1)
    first = nodes[:, 0].masked_fill(~first_valid, -torch.inf)
    first = first-direction_cost*direction_alignment(head, directions[:, 0])
    if planes == 1:
        connected = torch.isfinite(first).any(-1)
        return first.argmax(-1)[:, None], connected
    delta = positions[:, 1:, None]-positions[:, :-1, :, None]
    tangents = F.normalize(delta, dim=-1)  # B,T-1,previous,current,3
    edge_tensor = .5*(directions[:, 1:, None]+directions[:, :-1, :, None])
    edges = direction_cost*direction_alignment(tangents, edge_tensor)
    initial_turn = F.smooth_l1_loss(tangents[:, 0], head[:, :, None].expand_as(tangents[:, 0]),
                                  beta=.25, reduction='none').sum(-1)
    score = first[:, :, None]+nodes[:, 1, None]-edges[:, 0]-turn_cost*initial_turn
    pointers = []
    for t in range(2, planes):
        before = tangents[:, t-2, :, :, None]
        after = tangents[:, t-1, None]
        difference = before-after
        absolute = difference.abs()
        turn = torch.where(absolute < .25, .5*difference.square()/.25, absolute-.125).sum(-1)
        values = score[..., None]-turn_cost*turn
        best, pointer = values.max(1)
        score = best+nodes[:, t, None]-edges[:, t-1]
        pointers.append(pointer)
    connected = torch.isfinite(score.flatten(1)).any(-1)
    index = score.flatten(1).argmax(-1)
    previous, current = index//k, index % k
    path = [current, previous]
    for pointer in reversed(pointers):
        older = pointer.flatten(1).gather(1, (previous*k+current)[:, None])[:, 0]
        current, previous = previous, older
        path.append(older)
    selected = torch.stack(path[::-1], 1)
    fallback = nodes.argmax(-1)
    return torch.where(connected[:, None], selected, fallback), connected


class CandidateMemoryFollower(TrajectoryMemoryFollower):
    def __init__(self, cfg):
        if cfg.memory_version != 5:
            raise ValueError('CandidateMemoryFollower requires memory v5')
        super().__init__(replace(cfg, memory_version=4))
        self.cfg, self.architecture = cfg, CANDIDATE_MEMORY_ARCHITECTURE
        del self.coordinates
        width = cfg.channels+cfg.hidden+3
        self.proposal_head = nn.Sequential(nn.Linear(width, 64), nn.SiLU(), nn.Linear(64, 3))
        xy, self.proposal_width = proposal_lattice(cfg)
        self.proposal_spacing = 2*cfg.lateral_limit/(self.proposal_width-1)
        grid = torch.cat((xy[None].expand(cfg.n_future, -1, -1),
                          self.planes[:, None, None].expand(-1, len(xy), 1)), -1)
        self.register_buffer('proposal_grid', grid, persistent=False)

    def context(self, x, hist, hmask):
        ctx = super().context(x, hist, hmask)
        if self.cfg.direction_inputs:
            ctx['directions'] = x['fine'][:, 2:8]
        return ctx

    def spatial_features(self, ctx, points):
        fine, _ = sample_features(ctx.get('fine_fp32', ctx['fine']), points, self.cfg.fine)
        deep, _ = sample_features(ctx.get('deep_fp32', ctx['deep']), points, self.cfg.fine,
                                  TOKEN_STRIDE, TOKEN_OFFSET)
        return torch.cat((fine.to(ctx['fine'].dtype), deep.to(ctx['fine'].dtype),
                          (points/16.).to(ctx['fine'].dtype)), -1)

    def predict(self, ctx, hist, candidates=None):
        cfg, b = self.cfg, len(hist)
        memory, padding = self.decoder_memory(ctx)
        grid = self.proposal_grid[None].expand(b, -1, -1, -1)
        features = self.spatial_features(ctx, grid.flatten(1, 2))
        # Ranking thousands of sites in BF16 creates artificial score ties.
        with torch.autocast(device_type=features.device.type, enabled=False):
            proposed = self.proposal_head(features.float()).reshape(b, cfg.n_future, -1, 3)
        logits = proposed[..., 0]
        offsets = proposed[..., 1:].tanh()*(self.proposal_spacing/2)
        xy = (grid[..., :2]+offsets).clamp(-cfg.lateral_limit, cfg.lateral_limit)
        positions = torch.cat((xy, grid[..., 2:]), -1)
        first_limit = math.sqrt(max(0., cfg.max_recovery_distance**2-cfg.future_step**2))
        indices, valid = shortlist(logits, positions, cfg.proposal_candidates, cfg.proposal_suppression, first_limit)
        locations = positions.gather(2, indices[..., None].expand(-1, -1, -1, 3))
        flat = locations.detach().flatten(1, 2)
        node_scores = logits.log_softmax(-1).gather(-1, indices)
        if 'directions' in ctx:
            directions, _ = sample_features(ctx['directions'], flat, cfg.fine)
            directions = directions.reshape(b, cfg.n_future, cfg.proposal_candidates, 6)
        else:
            directions = locations.new_zeros(b, cfg.n_future, cfg.proposal_candidates, 6)
        route, connected = continuous_route(node_scores, locations, valid, directions,
            cfg.proposal_direction_cost, cfg.proposal_turn_cost, cfg.max_recovery_distance)
        points = locations.gather(2, route[..., None, None].expand(-1, -1, 1, 3))[:, :, 0]
        query = self.query(self.query_features(ctx, points))
        projected = [layer.project_memory(memory) for layer in self.decoder.layers] if cfg.recurrent_refinement_steps else None
        decoded = (self.decode_cached(query, projected, padding) if projected is not None else
                   self.decoder(query, memory, memory_key_padding_mask=padding))
        out = self.finish_prediction(ctx, decoded, points, projected, padding, candidates)
        out['confidence'] = out['confidence']*connected[:, None]
        out.update(proposal_logits=logits, proposal_offsets=offsets, proposal_grid=self.proposal_grid,
                   proposal_positions=locations, proposal_valid=valid,
                   proposal_route=route, proposal_connected=connected)
        return out


def initialize_candidate_model(source, cfg):
    """New-run initialization, with strict shared-weight accounting (no optimizer resume)."""
    if source.cfg.memory_version != 4 or source.cfg.feature_memory_revision != 2 or cfg.memory_version != 5:
        raise ValueError('V5 initialization requires feature-memory v4 revision 2')
    model = CandidateMemoryFollower(cfg).to(next(source.parameters()).device)
    old, target = source.state_dict(), model.state_dict()
    state = {}
    for key, value in old.items():
        if key.startswith('coordinates.'):
            continue
        if key == 'refinement_stage.weight':
            stages = target[key].clone()
            count = min(len(stages), len(value))
            stages[:count] = value[:count]
            state[key] = stages
        elif key not in target or target[key].shape != value.shape:
            raise ValueError(f'Incompatible shared parameter: {key}')
        else:
            state[key] = value
    missing = model.load_state_dict(state, strict=False)
    allowed = ('proposal_',)
    if not source.cfg.recurrent_refinement_steps:
        allowed += ('refinement_',)
    if missing.unexpected_keys or any(not k.startswith(allowed) for k in missing.missing_keys):
        raise ValueError(f'Unexpected migration mismatch: {missing}')
    return model
