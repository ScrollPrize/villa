"""Spatial observations and task-trained compression, without identity judgments."""
import torch
from torch import nn
import torch.nn.functional as F

from .feature_memory import FeatureMemory
from .model import sample_features, crop_support, TOKEN_STRIDE, TOKEN_OFFSET


class DetailedFeatureMemory(FeatureMemory):
    def __init__(self, cfg, token_xyz):
        super().__init__(cfg, token_xyz)
        del self.probe_norm, self.probe_attention, self.probe_head
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
        state = super().initial_state(batch, device)
        state.update(anchor_mask=torch.zeros(batch, self.count, device=device, dtype=torch.bool),
                     cache_mask=torch.zeros(batch, self.cfg.memory_steps, self.count, device=device, dtype=torch.bool))
        return state

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
        tokens, padding = super().read_tokens(state)
        s, n = self.cfg.memory_slots, self.count
        padding = torch.cat((padding[:, :s], ~state['anchor_mask'] | ~state['anchor_valid'][:, None],
                             (~state['cache_mask'] | ~state['cache_valid'][:, :, None]).flatten(1, 2)), 1)
        return tokens, padding

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
