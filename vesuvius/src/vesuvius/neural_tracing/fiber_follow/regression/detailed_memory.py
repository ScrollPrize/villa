"""Spatial observations and task-trained compression, without identity judgments."""
import math
import torch
from torch import nn
import torch.nn.functional as F

from .model import sample_features, crop_support, TOKEN_STRIDE, TOKEN_OFFSET


class DetailedFeatureMemory(nn.Module):
    def __init__(self, cfg, token_xyz):
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden
        self.initial_slots = nn.Parameter(torch.randn(cfg.memory_slots, h)*.02)
        self.motion = nn.Sequential(nn.Linear(12, h), nn.SiLU(), nn.Linear(h, h))
        self.write_norm = nn.LayerNorm(h)
        self.write_attention = nn.MultiheadAttention(h, cfg.heads, dropout=0., batch_first=True)
        self.proposal = nn.Linear(2*h, h)
        self.gate = nn.Linear(2*h, h)
        nn.init.constant_(self.gate.bias, -2.)
        self.role = nn.Parameter(torch.randn(2, h)*.02)
        self.count = math.prod(cfg.feature_memory_grid)
        self.feature_projection = nn.Sequential(nn.LayerNorm(cfg.hidden), nn.Linear(cfg.hidden, cfg.hidden))
        self.spatial = nn.Linear(4, cfg.hidden)
        xyz = token_xyz.reshape(*cfg.token_shape, 3).permute(3, 0, 1, 2)[None]
        self.register_buffer('local_xyz', F.adaptive_avg_pool3d(xyz, cfg.feature_memory_grid)
                             .flatten(2).transpose(1, 2)[0], persistent=False)
        self.coarse_count = self.count
        self.count += cfg.feature_detail_tokens
        width = 9*(cfg.channels+1)+cfg.hidden+1
        self.detail_projection = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, cfg.hidden))
        self.detail_role = nn.Parameter(torch.randn(cfg.hidden)*.02)
        self.prior_attention = nn.MultiheadAttention(cfg.hidden, cfg.heads, dropout=0., batch_first=True)
        self.prior_norm = nn.LayerNorm(cfg.hidden)
        self.write_position = nn.Linear(3, cfg.hidden, bias=False)
        stencil = torch.tensor([[a, b, 0.] for a in (-1., 0., 1.) for b in (-1., 0., 1.)])
        self.register_buffer('detail_stencil', stencil*cfg.patch_radius, persistent=False)

    def initial_state(self, batch, device):
        n, k, h = self.count, self.cfg.memory_steps, self.cfg.hidden
        return dict(slots=self.initial_slots.to(device).unsqueeze(0).expand(batch, -1, -1),
            anchor=torch.zeros(batch, n, h, device=device),
            anchor_xyz=torch.zeros(batch, n, 3, device=device),
            anchor_mask=torch.zeros(batch, n, device=device, dtype=torch.bool),
            anchor_valid=torch.zeros(batch, device=device, dtype=torch.bool),
            anchor_position=torch.zeros(batch, 3, device=device),
            anchor_frame=torch.eye(3, device=device).expand(batch, -1, -1),
            position=torch.zeros(batch, 3, device=device),
            frame=torch.eye(3, device=device).expand(batch, -1, -1),
            seen=torch.zeros(batch, device=device, dtype=torch.bool),
            cache=torch.zeros(batch, k, n, h, device=device),
            cache_xyz=torch.zeros(batch, k, n, 3, device=device),
            cache_frames=torch.eye(3, device=device).expand(batch, k, -1, -1).clone(),
            cache_valid=torch.zeros(batch, k, device=device, dtype=torch.bool),
            cache_mask=torch.zeros(batch, k, n, device=device, dtype=torch.bool),
            cache_age=torch.zeros(batch, k, device=device))

    def detail_points(self, hist, hmask):
        """Head plus evenly spaced visible *observed* history, with fixed padding."""
        b, n = len(hist), self.cfg.feature_detail_tokens
        valid = hmask.bool() & crop_support(hist, self.cfg.fine)
        count = valid.sum(1)
        order = torch.arange(hist.shape[1], device=hist.device)[None].expand(b, -1)
        order = order.masked_fill(~valid, hist.shape[1]).sort(1).values
        j = torch.arange(n-1, device=hist.device)[None].expand(b, -1)
        used = count.clamp(max=n-1)
        rank = (j*(count-1).clamp_min(0)[:, None]/(used-1).clamp_min(1)[:, None]).round().long()
        rank = rank.clamp(max=hist.shape[1]-1)
        index = order.gather(1, rank).clamp(max=hist.shape[1]-1)
        points = hist.gather(1, index[..., None].expand(-1, -1, 3))
        mask = j < used[:, None]
        points = torch.where(mask[..., None], points, 0.)
        return (torch.cat((hist.new_zeros(b, 1, 3), points), 1),
                torch.cat((torch.ones(b, 1, device=hist.device, dtype=torch.bool), mask), 1))

    def extract(self, deep, fine, hist, hmask):
        """Extract unconditioned evidence; never store a retrieved identity belief."""
        with torch.autocast(device_type=deep.device.type, enabled=False):
            coarse = F.adaptive_avg_pool3d(deep.float(), self.cfg.feature_memory_grid).flatten(2).transpose(1, 2)
            coarse = self.feature_projection(coarse)
            points, valid = self.detail_points(hist.float(), hmask)
            b, n = points.shape[:2]
            patches = points[:, :, None]+self.detail_stencil
            values, support = sample_features(fine, patches.reshape(b, n*9, 3), self.cfg.fine)
            local = torch.cat((values, support[..., None]), -1).reshape(b, n, -1)
            context, supported = sample_features(deep, points, self.cfg.fine, TOKEN_STRIDE, TOKEN_OFFSET)
            detail = self.detail_projection(torch.cat((local, context, supported[..., None]), -1))+self.detail_role
            detail = torch.where(valid[..., None], detail, 0.)
            xyz = torch.cat((self.local_xyz[None].expand(b, -1, -1), points), 1)
            mask = torch.cat((torch.ones(b, self.coarse_count, device=deep.device, dtype=torch.bool), valid), 1)
            return torch.cat((coarse, detail), 1), xyz, mask

    def read_tokens(self, state):
        """No crop-support mask: remote evidence remains readable until eviction."""
        frame, position = state['frame'], state['position']
        def spatial(xyz, ages):
            local = torch.einsum('bni,bij->bnj', xyz-position[:, None], frame)/16.
            local = local.sign()*torch.log1p(local.abs())
            return self.spatial(torch.cat((local, torch.log1p(ages)[..., None]), -1))
        anchor = state['anchor']+self.role[1]+spatial(state['anchor_xyz'],
                    state['cache_age'].new_zeros(state['anchor'].shape[:2]))
        anchor = anchor+self.relative_anchor(state['anchor_position'], state['anchor_frame'],
                    position, frame, state['anchor_valid'])[:, None]
        k, n = state['cache'].shape[1:3]
        cache = state['cache'].flatten(1, 2)+self.role[0]+spatial(state['cache_xyz'].flatten(1, 2),
                    state['cache_age'][:, :, None].expand(-1, -1, n).flatten(1, 2))
        rotation = frame[:, None].transpose(-1, -2) @ state['cache_frames']
        pose = torch.cat((rotation.new_zeros(len(frame), k, 3), rotation.flatten(-2)), -1)
        cache = cache+(self.motion(pose)[:, :, None].expand(-1, -1, n, -1)).flatten(1, 2)
        tokens = torch.cat((state['slots']+self.role[0], anchor, cache), 1)
        padding = torch.cat((torch.zeros(state['slots'].shape[:2], device=frame.device, dtype=torch.bool),
            ~state['anchor_mask'] | ~state['anchor_valid'][:, None],
            (~state['cache_mask'] | ~state['cache_valid'][:, :, None]).flatten(1, 2)), 1)
        return tokens, padding

    def relative_anchor(self, anchor_position, anchor_frame, position, frame, valid):
        delta = torch.einsum('...i,...ij->...j', anchor_position-position, frame)/16.
        delta = delta.sign()*torch.log1p(delta.abs())
        rotation = (frame.transpose(-1, -2) @ anchor_frame).flatten(-2)
        pose = torch.where(valid[..., None], torch.cat((delta, rotation), -1), 0.)
        return self.motion(pose)

    def write_inputs(self, encoded, pos, fr, previous, previous_frame, moved,
                     anchor, anchor_position, anchor_frame, anchor_valid):
        """Observation and seed tokens of each write (b×t×8×h), independent of the slots."""
        displacement = torch.einsum('bti,btij->btj', pos-previous, fr)/16.
        rotation = (fr.transpose(-1, -2) @ previous_frame).flatten(2)
        motion = torch.where(moved[..., None], torch.cat((displacement, rotation), -1), 0.)
        observations = encoded+self.motion(motion)[:, :, None]
        seed_pose = self.relative_anchor(anchor_position[:, None], anchor_frame[:, None], pos, fr,
                                         anchor_valid[:, None].expand(-1, pos.shape[1]))
        anchor_tokens = anchor[:, None]+self.role[1]+seed_pose[:, :, None]
        return observations, anchor_tokens

    def observe_tokens(self, tokens, local_xyz, valid, x, state=None):
        """Replayable small transition; all intervening writes stay differentiable."""
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            tokens, local_xyz = tokens.float(), local_xyz.float()
            state = self.initial_state(len(tokens), tokens.device) if state is None else state
            state = {k: v.float() if v.is_floating_point() else v for k, v in state.items()}
            position, frame = x['query_position'].float(), x['query_frame'].float()
            xyz = local_xyz @ frame.transpose(-1, -2)+position[:, None]
            # Read only the incoming state, expressed in today's crop frame.
            incoming, padding = self.read_tokens(dict(state, position=position, frame=frame))
            query = tokens+self.write_position(local_xyz/16.)
            retrieved = self.prior_attention(self.prior_norm(query), incoming, incoming,
                                             key_padding_mask=padding, need_weights=False)[0]
            retrieved = torch.where(valid[..., None], retrieved, 0.)
            seed = x['feature_seed_here'].bool() & ~state['anchor_valid']
            anchor = torch.where(seed[:, None, None], tokens, state['anchor'])
            anchor_xyz = torch.where(seed[:, None, None], xyz, state['anchor_xyz'])
            anchor_mask = torch.where(seed[:, None], valid, state['anchor_mask'])
            anchor_valid = seed | state['anchor_valid']
            anchor_position = torch.where(seed[:, None], position, state['anchor_position'])
            anchor_frame = torch.where(seed[:, None, None], frame, state['anchor_frame'])
            # Compress observations, not a retrieved interpretation of them.
            # The memory read above is used only by the current spatial encoder.
            observed, anchors = self.write_inputs(query[:, None],
                position[:, None], frame[:, None], state['position'][:, None], state['frame'][:, None],
                state['seen'][:, None], anchor, anchor_position, anchor_frame, anchor_valid)
            context = torch.cat((observed[:, 0], anchors[:, 0]), 1)
            write_padding = torch.cat((~valid, ~anchor_mask | ~anchor_valid[:, None]), 1)
            evidence = self.write_attention(self.write_norm(state['slots']), context, context,
                                            key_padding_mask=write_padding, need_weights=False)[0]
            pair = torch.cat((state['slots'], evidence), -1)
            slots = state['slots']+self.gate(pair).sigmoid()*(self.proposal(pair).tanh()-state['slots'])
            out = dict(slots=slots, anchor=anchor, anchor_xyz=anchor_xyz, anchor_mask=anchor_mask,
                anchor_valid=anchor_valid, anchor_position=anchor_position, anchor_frame=anchor_frame,
                position=position, frame=frame, seen=torch.ones_like(anchor_valid),
                cache=torch.cat((state['cache'][:, 1:], tokens[:, None]), 1),
                cache_xyz=torch.cat((state['cache_xyz'][:, 1:], xyz[:, None]), 1),
                cache_frames=torch.cat((state['cache_frames'][:, 1:], frame[:, None]), 1),
                cache_mask=torch.cat((state['cache_mask'][:, 1:], valid[:, None]), 1),
                cache_valid=torch.cat((state['cache_valid'][:, 1:], torch.ones_like(anchor_valid)[:, None]), 1),
                cache_age=torch.cat((state['cache_age'][:, 1:]+1, state['cache_age'].new_zeros(len(tokens), 1)), 1))
            if 'feature_active' in x:
                active = x['feature_active'].bool()
                for key, previous in state.items():
                    out[key] = torch.where(active.reshape(len(active), *(1,)*(previous.ndim-1)), out[key], previous)
            return out, retrieved
