"""Bounded spatial memory from main-encoder features; no historical image encoder."""
import math

import torch
from torch import nn
import torch.nn.functional as F

from .memory import LearnedMemory


class FeatureMemory(LearnedMemory):
    def __init__(self, cfg, token_xyz):
        super().__init__(cfg)
        # Reuse the tested admission probe and slot writer, not the patch encoder.
        del self.patch_encoder, self.patch_projection, self.patch_position
        del self.read_attention, self.read_norm
        self.count = math.prod(cfg.feature_memory_grid)
        self.feature_projection = nn.Sequential(nn.LayerNorm(cfg.hidden), nn.Linear(cfg.hidden, cfg.hidden))
        self.spatial = nn.Linear(4, cfg.hidden)
        xyz = token_xyz.reshape(*cfg.token_shape, 3).permute(3, 0, 1, 2)[None]
        self.register_buffer('local_xyz', F.adaptive_avg_pool3d(xyz, cfg.feature_memory_grid)
                             .flatten(2).transpose(1, 2)[0], persistent=False)

    def initial_state(self, batch, device):
        state = super().initial_state(batch, device)
        n, k, h = self.count, self.cfg.memory_steps, self.cfg.hidden
        state.update(anchor=torch.zeros(batch, n, h, device=device),
                     anchor_xyz=torch.zeros(batch, n, 3, device=device),
                     cache=torch.zeros(batch, k, n, h, device=device),
                     cache_xyz=torch.zeros(batch, k, n, 3, device=device),
                     cache_frames=torch.eye(3, device=device).expand(batch, k, -1, -1).clone(),
                     cache_valid=torch.zeros(batch, k, device=device, dtype=torch.bool),
                     cache_age=torch.zeros(batch, k, device=device))
        return state

    def observe_features(self, deep, x, state=None):
        # Slots and cached representations stay FP32, like the existing writer.
        with torch.autocast(device_type=deep.device.type, enabled=False):
            state = self.initial_state(len(deep), deep.device) if state is None else state
            state = {k: v.float() if v.is_floating_point() else v for k, v in state.items()}
            tokens = F.adaptive_avg_pool3d(deep.float(), self.cfg.feature_memory_grid).flatten(2).transpose(1, 2)
            tokens = self.feature_projection(tokens)
            position, frame = x['query_position'].float(), x['query_frame'].float()
            xyz = self.local_xyz[None] @ frame.transpose(-1, -2)+position[:, None]
            seed = x['feature_seed_here'].bool() & ~state['anchor_valid']
            anchor = torch.where(seed[:, None, None], tokens, state['anchor'])
            anchor_xyz = torch.where(seed[:, None, None], xyz, state['anchor_xyz'])
            anchor_valid = seed | state['anchor_valid']
            anchor_position = torch.where(seed[:, None], position, state['anchor_position'])
            anchor_frame = torch.where(seed[:, None, None], frame, state['anchor_frame'])
            observed, anchors = self.write_inputs(tokens[:, None], position[:, None], frame[:, None],
                state['position'][:, None], state['frame'][:, None], state['seen'][:, None],
                anchor, anchor_position, anchor_frame, anchor_valid)
            probe = self.probe(observed, state['slots'][:, None], anchors, anchor_valid)
            key, value = self.write_keys(torch.cat((observed, anchors), 2))
            proposed = self.write(state['slots'], key[:, 0], value[:, 0],
                                  torch.ones_like(anchor_valid), anchor_valid)
            slots = state['slots']+probe[:, 0, :1].sigmoid()[:, :, None]*(proposed-state['slots'])
            out = dict(slots=slots, anchor=anchor, anchor_xyz=anchor_xyz, anchor_valid=anchor_valid,
                anchor_position=anchor_position, anchor_frame=anchor_frame,
                position=position, frame=frame, seen=torch.ones_like(anchor_valid),
                cache=torch.cat((state['cache'][:, 1:], tokens[:, None]), 1),
                cache_xyz=torch.cat((state['cache_xyz'][:, 1:], xyz[:, None]), 1),
                cache_frames=torch.cat((state['cache_frames'][:, 1:], frame[:, None]), 1),
                cache_valid=torch.cat((state['cache_valid'][:, 1:], torch.ones_like(anchor_valid)[:, None]), 1),
                cache_age=torch.cat((state['cache_age'][:, 1:]+1, state['cache_age'].new_zeros(len(deep), 1)), 1),
                probe=probe)
            if 'feature_active' in x:
                active = x['feature_active'].bool()
                for key, previous in state.items():
                    if key != 'probe':
                        out[key] = torch.where(active.reshape(len(active), *(1,)*(previous.ndim-1)), out[key], previous)
            return out

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
            ~state['anchor_valid'][:, None].expand(-1, n),
            ~state['cache_valid'][:, :, None].expand(-1, -1, n).flatten(1, 2)), 1)
        return tokens, padding
