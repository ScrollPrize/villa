"""Differentiable observation memory; annotations never enter this module.

The seed is immutable within a trace. Adaptive slots learn read/write/retain
operations from later task losses. Historical patches are re-encoded with the
current weights during training; inference carries only a bounded tensor state.
"""
import torch
from torch import nn


class LearnedMemory(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden
        self.patch_encoder = nn.Sequential(
            nn.Conv3d(2, 16, 3, stride=2, padding=1), nn.GroupNorm(4, 16), nn.SiLU(),
            nn.Conv3d(16, 32, 3, stride=2, padding=1), nn.GroupNorm(8, 32), nn.SiLU(),
            nn.AdaptiveAvgPool3d((2, 2, 2)))
        self.patch_projection = nn.Linear(32, h)
        self.patch_position = nn.Parameter(torch.randn(8, h)*.02)
        self.initial_slots = nn.Parameter(torch.randn(cfg.memory_slots, h)*.02)
        self.motion = nn.Sequential(nn.Linear(12, h), nn.SiLU(), nn.Linear(h, h))
        self.write_norm = nn.LayerNorm(h)
        self.write_attention = nn.MultiheadAttention(h, cfg.heads, dropout=0., batch_first=True)
        self.proposal = nn.Linear(2*h, h)
        self.gate = nn.Linear(2*h, h)
        nn.init.constant_(self.gate.bias, -2.)
        self.read_norm = nn.LayerNorm(h)
        self.read_attention = nn.MultiheadAttention(h, cfg.heads, dropout=0., batch_first=True)
        self.role = nn.Parameter(torch.randn(2, h)*.02)  # adaptive versus original anchor

    def initial_state(self, batch, device):
        return dict(slots=self.initial_slots.to(device).unsqueeze(0).expand(batch, -1, -1),
                    anchor=torch.zeros(batch, 8, self.cfg.hidden, device=device),
                    anchor_valid=torch.zeros(batch, device=device, dtype=torch.bool),
                    anchor_position=torch.zeros(batch, 3, device=device),
                    anchor_frame=torch.eye(3, device=device).expand(batch, -1, -1),
                    position=torch.zeros(batch, 3, device=device),
                    frame=torch.eye(3, device=device).expand(batch, -1, -1),
                    seen=torch.zeros(batch, device=device, dtype=torch.bool))

    def encode(self, patches):
        return self.patch_projection(self.patch_encoder(patches).flatten(2).transpose(1, 2))+self.patch_position

    def observe(self, x, state=None):
        if 'memory_patches' not in x:
            raise ValueError('Memory model requires observed patch sequences from ObservationBuilder')
        patches = x['memory_patches']
        b, t = patches.shape[:2]
        valid = x['memory_mask'].bool()
        state = self.initial_state(b, patches.device) if state is None else state
        # Mask before encoding: padded NaNs must not poison attention/gradients.
        clean = torch.where(valid[..., None, None, None, None], patches, 0.)
        encoded = self.encode(clean.reshape(b*t, *patches.shape[2:])).reshape(b, t, 8, -1)
        seed_valid = x['memory_seed_valid'].bool() & ~state['anchor_valid'].bool()
        seed_patch = torch.where(seed_valid[:, None, None, None, None], x['memory_seed_patch'], 0.)
        seed = self.encode(seed_patch)
        anchor = torch.where(seed_valid[:, None, None], seed, state['anchor'])
        anchor_valid = state['anchor_valid'].bool() | seed_valid
        anchor_position = torch.where(seed_valid[:, None], x['memory_seed_position'], state['anchor_position'])
        anchor_frame = torch.where(seed_valid[:, None, None], x['memory_seed_frame'], state['anchor_frame'])
        slots, position, frame, seen = (state[k] for k in ('slots', 'position', 'frame', 'seen'))
        for j in range(t):
            active = valid[:, j]
            pos = torch.where(active[:, None], x['memory_positions'][:, j], position)
            fr = torch.where(active[:, None, None], x['memory_frames'][:, j], frame)
            displacement = torch.einsum('bi,bij->bj', pos-position, fr)/16.
            rotation = torch.bmm(fr.transpose(1, 2), frame).flatten(1)
            motion = torch.cat((displacement, rotation), -1)
            motion = torch.where((active & seen.bool())[:, None], motion, 0.)
            observations = encoded[:, j]+self.motion(motion)[:, None]
            seed_pose = self.relative_anchor(anchor_position, anchor_frame, pos, fr, anchor_valid)
            context = torch.cat((observations, anchor+self.role[1]+seed_pose[:, None]), 1)
            padding = torch.cat((torch.zeros(b, 8, device=patches.device, dtype=torch.bool),
                                 ~anchor_valid[:, None].expand(-1, 8)), 1)
            evidence = self.write_attention(self.write_norm(slots), context, context,
                                             key_padding_mask=padding, need_weights=False)[0]
            pair = torch.cat((slots, evidence), -1)
            gate = self.gate(pair).sigmoid()
            proposal = self.proposal(pair).tanh()
            updated = slots+gate*(proposal-slots)
            slots = torch.where(active[:, None, None], updated, slots)
            position, frame = pos, fr
            seen = seen.bool() | active
        return dict(slots=slots, anchor=anchor, anchor_valid=anchor_valid,
                    anchor_position=anchor_position, anchor_frame=anchor_frame,
                    position=position, frame=frame, seen=seen)

    def relative_anchor(self, anchor_position, anchor_frame, position, frame, valid):
        delta = torch.einsum('bi,bij->bj', anchor_position-position, frame)/16.
        delta = delta.sign()*torch.log1p(delta.abs())
        rotation = torch.bmm(frame.transpose(1, 2), anchor_frame).flatten(1)
        pose = torch.where(valid[:, None], torch.cat((delta, rotation), -1), 0.)
        return self.motion(pose)

    def read(self, decoded, state):
        slots = state['slots']+self.role[0]
        pose = self.relative_anchor(state['anchor_position'], state['anchor_frame'],
                                    state['position'], state['frame'], state['anchor_valid'])
        anchor = state['anchor']+self.role[1]+pose[:, None]
        tokens = torch.cat((slots, anchor), 1)
        padding = torch.cat((torch.zeros(slots.shape[:2], device=slots.device, dtype=torch.bool),
                             ~state['anchor_valid'][:, None].bool().expand(-1, 8)), 1)
        return decoded+self.read_attention(self.read_norm(decoded), tokens, tokens,
                                           key_padding_mask=padding, need_weights=False)[0]
