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
import torch.nn.functional as F

PROBE_OUTPUTS = 4  # on-original-fiber logit, offset to it in the observation frame


class LearnedMemory(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden
        self.patch_encoder = nn.Sequential(
            nn.Conv3d(cfg.input_channels, 16, 3, stride=2, padding=1), nn.GroupNorm(4, 16), nn.SiLU(),
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
        """Write observations in FP32, including under the caller's BF16 autocast.

        Long recurrent unrolls can amplify BF16 rounding into enormous writer
        gradients. Keep encoding, burn-in, writes and probes in FP32; the caller
        can still read the resulting memory and decode curves under autocast.
        """
        if 'memory_patches' not in x:
            raise ValueError('Memory model requires observed patch sequences from ObservationBuilder')
        with torch.autocast(device_type=x['memory_patches'].device.type, enabled=False):
            x = {k: v.float() if k.startswith('memory_') and v.is_floating_point() else v
                 for k, v in x.items()}
            if state is not None:
                state = {k: v.float() if v.is_floating_point() else v for k, v in state.items()}
            return self._observe(x, state)

    def prepare_sequence(self, x, state):
        """Shared encoding/poses for legacy and spatial-memory state transitions."""
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
        pos, fr, previous, previous_frame, moved, seen = self.poses(x, valid, state)
        return (state, valid, burn, encoded, anchor, anchor_valid, anchor_position,
                anchor_frame, pos, fr, previous, previous_frame, moved, seen)

    def _observe(self, x, state):
        """FP32 sequence unroll; returns state plus the per-write ``probe``."""
        (state, valid, burn, encoded, anchor, anchor_valid, anchor_position,
         anchor_frame, pos, fr, previous, previous_frame, moved, seen) = self.prepare_sequence(x, state)
        t = valid.shape[1]
        # Only the gated slot update is recurrent. Motion, seed pose and their
        # keys/values are formed for every write at once; the probe, which never
        # feeds back, reads the retained slots afterwards. Burn-in stays gradient-free.
        slots, probes = state['slots'], []
        for lo, hi in ((0, burn), (burn, t)):
            if lo == hi:
                continue
            # A Python branch specializes symbolic sequence lengths. Passing a
            # SymBool directly to set_grad_enabled breaks Dynamo recompilation.
            backprop = torch.is_grad_enabled()
            if lo < burn:
                backprop = False
            with torch.set_grad_enabled(backprop):
                span = slice(lo, hi)
                observations, anchor_tokens = self.write_inputs(
                    encoded[:, span], pos[:, span], fr[:, span], previous[:, span], previous_frame[:, span],
                    moved[:, span], anchor, anchor_position, anchor_frame, anchor_valid)
                key, value = self.write_keys(torch.cat((observations, anchor_tokens), 2))
                retained = []
                for j in range(hi-lo):
                    slots = self.write(slots, key[:, j], value[:, j], valid[:, lo+j], anchor_valid)
                    retained.append(slots)
                if self.cfg.memory_version >= 2:
                    probes.append(self.probe(observations, torch.stack(retained, 1), anchor_tokens, anchor_valid))
        out = dict(slots=slots, anchor=anchor, anchor_valid=anchor_valid,
                   anchor_position=anchor_position, anchor_frame=anchor_frame,
                   position=pos[:, -1], frame=fr[:, -1], seen=seen)
        if self.cfg.memory_version >= 2:
            out['probe'] = torch.cat(probes, 1)
        return out

    @staticmethod
    def poses(x, valid, state):
        """Carried pose after each write and before it; inactive writes keep the pose."""
        b, t = valid.shape
        # Latest active write at or before each step; index zero is the incoming state.
        latest = torch.where(valid, torch.arange(1, t+1, device=valid.device), 0).cummax(1).values
        before = torch.cat((torch.zeros_like(latest[:, :1]), latest[:, :-1]), 1)
        positions = torch.cat((state['position'][:, None],
                               torch.where(valid[..., None], x['memory_positions'], 0.)), 1)
        frames = torch.cat((state['frame'][:, None],
                            torch.where(valid[..., None, None], x['memory_frames'], 0.)), 1)
        def at(values, index):
            return values.gather(1, index.reshape(b, t, *(1,)*(values.dim()-2)).expand(-1, -1, *values.shape[2:]))
        seen = state['seen'].bool()[:, None] | (latest > 0)
        seen_before = torch.cat((state['seen'].bool()[:, None], seen[:, :-1]), 1)
        return (at(positions, latest), at(frames, latest), at(positions, before), at(frames, before),
                valid & seen_before, seen[:, -1])

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

    def write_keys(self, context):
        """Write-attention keys and values, b×t×heads×16×d, for all writes at once."""
        attention, h = self.write_attention, self.cfg.hidden
        kv = F.linear(context, attention.in_proj_weight[h:], attention.in_proj_bias[h:])
        return (v.unflatten(-1, (self.cfg.heads, -1)).transpose(-2, -3) for v in kv.chunk(2, -1))

    def write(self, slots, key, value, active, anchor_valid):
        """One gated update; equals ``write_attention`` over observation and seed tokens."""
        attention, h = self.write_attention, self.cfg.hidden
        query = F.linear(self.write_norm(slots), attention.in_proj_weight[:h], attention.in_proj_bias[:h])
        query = query.unflatten(-1, (self.cfg.heads, -1)).transpose(1, 2)
        visible = torch.cat((torch.ones_like(anchor_valid)[:, None].expand(-1, 8),
                             anchor_valid[:, None].expand(-1, 8)), 1)
        evidence = F.scaled_dot_product_attention(query, key, value, attn_mask=visible[:, None, None])
        evidence = attention.out_proj(evidence.transpose(1, 2).flatten(2))
        pair = torch.cat((slots, evidence), -1)
        gate = self.gate(pair).sigmoid()
        proposal = self.proposal(pair).tanh()
        updated = slots+gate*(proposal-slots)
        return torch.where(active[:, None, None], updated, slots)

    def probe(self, observations, slots, anchor_tokens, anchor_valid):
        """Judged from each observation and what memory retained after writing it (b×t×4)."""
        b, t = slots.shape[:2]
        query = observations.mean(2, keepdim=True)
        tokens = torch.cat((slots+self.role[0], anchor_tokens), 2).flatten(0, 1)
        retained = torch.cat((torch.zeros(b, slots.shape[2], device=slots.device, dtype=torch.bool),
                              ~anchor_valid[:, None].expand(-1, 8)), 1)
        read = self.probe_attention(self.probe_norm(query).flatten(0, 1), tokens, tokens,
                                    key_padding_mask=retained.repeat_interleave(t, 0), need_weights=False)[0]
        return self.probe_head(torch.cat((query, read.unflatten(0, (b, t))), -1))[:, :, 0]

    def relative_anchor(self, anchor_position, anchor_frame, position, frame, valid):
        delta = torch.einsum('...i,...ij->...j', anchor_position-position, frame)/16.
        delta = delta.sign()*torch.log1p(delta.abs())
        rotation = (frame.transpose(-1, -2) @ anchor_frame).flatten(-2)
        pose = torch.where(valid[..., None], torch.cat((delta, rotation), -1), 0.)
        return self.motion(pose)

    def read_tokens(self, state):
        slots = state['slots']+self.role[0]
        pose = self.relative_anchor(state['anchor_position'], state['anchor_frame'],
                                    state['position'], state['frame'], state['anchor_valid'])
        anchor = state['anchor']+self.role[1]+pose[:, None]
        tokens = torch.cat((slots, anchor), 1)
        padding = torch.cat((torch.zeros(slots.shape[:2], device=slots.device, dtype=torch.bool),
                             ~state['anchor_valid'][:, None].bool().expand(-1, 8)), 1)
        return tokens, padding

    def read(self, decoded, state):
        tokens, padding = self.read_tokens(state)
        return decoded+self.read_attention(self.read_norm(decoded), tokens, tokens,
                                           key_padding_mask=padding, need_weights=False)[0]
