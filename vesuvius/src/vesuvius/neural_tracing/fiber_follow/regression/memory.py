"""Differentiable observation memory; annotations never enter this module.

The seed is immutable within a trace. Adaptive slots learn read/write/retain
operations from later task losses. Historical patches are re-encoded with the
current weights during training; inference carries only a bounded tensor state.
Observations older than ``memory_grad_steps`` are written without gradients
(burn-in), so long sequences build a realistic state at little training cost.
Version 2 adds a per-write probe, supervised only as an auxiliary target.
"""
import torch
from torch import nn

PROBE_OUTPUTS = 4  # on-original-fiber logit, offset to it in the observation frame


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
        if cfg.memory_version >= 2:
            # An identity read at initialization: warm-started decoders are unchanged.
            nn.init.zeros_(self.read_attention.out_proj.weight)
            nn.init.zeros_(self.read_attention.out_proj.bias)
            self.probe_norm = nn.LayerNorm(h)
            self.probe_attention = nn.MultiheadAttention(h, cfg.heads, dropout=0., batch_first=True)
            self.probe_head = nn.Sequential(nn.Linear(2*h, h), nn.SiLU(), nn.Linear(h, PROBE_OUTPUTS))

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

    def encode_sequence(self, patches):
        b, t = patches.shape[:2]
        return self.encode(patches.reshape(b*t, *patches.shape[2:])).reshape(b, t, 8, -1)

    def observe(self, x, state=None):
        """Write each observation in order; returns the new state (plus ``probe``, b×t×4)."""
        if 'memory_patches' not in x:
            raise ValueError('Memory model requires observed patch sequences from ObservationBuilder')
        patches = x['memory_patches']
        b, t = patches.shape[:2]
        valid = x['memory_mask'].bool()
        state = self.initial_state(b, patches.device) if state is None else state
        burn = max(0, t-self.cfg.memory_grad_steps-1)  # the head always backpropagates
        # Mask before encoding: padded NaNs must not poison attention/gradients.
        clean = torch.where(valid[..., None, None, None, None], patches, 0.)
        if burn:
            with torch.no_grad():
                early = self.encode_sequence(clean[:, :burn])
            encoded = torch.cat((early, self.encode_sequence(clean[:, burn:])), 1)
        else:
            encoded = self.encode_sequence(clean)
        seed_valid = x['memory_seed_valid'].bool() & ~state['anchor_valid'].bool()
        seed_patch = torch.where(seed_valid[:, None, None, None, None], x['memory_seed_patch'], 0.)
        seed = self.encode(seed_patch)
        anchor = torch.where(seed_valid[:, None, None], seed, state['anchor'])
        anchor_valid = state['anchor_valid'].bool() | seed_valid
        anchor_position = torch.where(seed_valid[:, None], x['memory_seed_position'], state['anchor_position'])
        anchor_frame = torch.where(seed_valid[:, None, None], x['memory_seed_frame'], state['anchor_frame'])
        anchor_state = (anchor, anchor_valid, anchor_position, anchor_frame)
        carry = tuple(state[k] for k in ('slots', 'position', 'frame', 'seen'))
        probes = []
        for j in range(t):
            inputs = (encoded[:, j], valid[:, j], x['memory_positions'][:, j], x['memory_frames'][:, j])
            if j < burn:
                with torch.no_grad():
                    carry, probe = self.write(inputs, carry, anchor_state)
            else:
                carry, probe = self.write(inputs, carry, anchor_state)
            probes.append(probe)
        slots, position, frame, seen = carry
        out = dict(slots=slots, anchor=anchor, anchor_valid=anchor_valid,
                   anchor_position=anchor_position, anchor_frame=anchor_frame,
                   position=position, frame=frame, seen=seen)
        if self.cfg.memory_version >= 2:
            out['probe'] = torch.stack(probes, 1)
        return out

    def write(self, inputs, carry, anchor_state):
        encoded, active, new_position, new_frame = inputs
        slots, position, frame, seen = carry
        anchor, anchor_valid, anchor_position, anchor_frame = anchor_state
        b = len(slots)
        pos = torch.where(active[:, None], new_position, position)
        fr = torch.where(active[:, None, None], new_frame, frame)
        displacement = torch.einsum('bi,bij->bj', pos-position, fr)/16.
        rotation = torch.bmm(fr.transpose(1, 2), frame).flatten(1)
        motion = torch.cat((displacement, rotation), -1)
        motion = torch.where((active & seen.bool())[:, None], motion, 0.)
        observations = encoded+self.motion(motion)[:, None]
        seed_pose = self.relative_anchor(anchor_position, anchor_frame, pos, fr, anchor_valid)
        anchor_tokens = anchor+self.role[1]+seed_pose[:, None]
        context = torch.cat((observations, anchor_tokens), 1)
        padding = torch.cat((torch.zeros(b, 8, device=slots.device, dtype=torch.bool),
                             ~anchor_valid[:, None].expand(-1, 8)), 1)
        evidence = self.write_attention(self.write_norm(slots), context, context,
                                         key_padding_mask=padding, need_weights=False)[0]
        pair = torch.cat((slots, evidence), -1)
        gate = self.gate(pair).sigmoid()
        proposal = self.proposal(pair).tanh()
        updated = slots+gate*(proposal-slots)
        slots = torch.where(active[:, None, None], updated, slots)
        probe = None
        if self.cfg.memory_version >= 2:
            # Judged from this observation and what memory now retains.
            query = observations.mean(1, keepdim=True)
            tokens = torch.cat((slots+self.role[0], anchor_tokens), 1)
            retained = torch.cat((torch.zeros(slots.shape[:2], device=slots.device, dtype=torch.bool),
                                  ~anchor_valid[:, None].expand(-1, 8)), 1)
            read = self.probe_attention(self.probe_norm(query), tokens, tokens,
                                        key_padding_mask=retained, need_weights=False)[0]
            probe = self.probe_head(torch.cat((query, read), -1))[:, 0]
        return (slots, pos, fr, seen.bool() | active), probe

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
