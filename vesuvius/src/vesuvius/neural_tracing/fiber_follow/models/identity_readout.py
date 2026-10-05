"""Path-blind identity memory (``identity_objective='readout'``).

Appearance keys come from the encoder's image stem before the position embedding and the
rendered observed path are added, through a small translation-equivariant convolutional head
without position input. Nothing in a key says where the trace is, so "on my path" cannot
stand in for "on my fiber".

Each decision stores the keys of its own crop on a lateral grid around its head (KEY_GRID,
2-voxel spacing) as extra rows of its decision-memory entry. A key's value says where that
memory point was relative to that decision's trace axis (its lateral radius) plus the entry's
age and seed role: the drawn-path mask lives only in the value, as in STCN/XMem/SAM2. A current
location is read out by attention from its key to every memory key, then classified on/off the
original fiber from its own key and the readout. The current side carries no position, so the
readout must match appearance; memory points beside the old trace act as explicit neighbor
evidence.

Training uses the verifier's targets and loss (identity_verifier.py). Memory keys carry no
gradient (recorded or encoded without gradient, like decision-memory entries); the current
crop's keys, the key head, the readout and the stem train through the identity loss.
"""
import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

KEY_DIM = 64
KEY_LATERAL = np.arange(-12., 12.+1e-9, 2.)
KEY_PLANES = (-6., -2., 2., 6.)
KEY_GRID = np.array([(x, y, z) for z in KEY_PLANES for y in KEY_LATERAL for x in KEY_LATERAL], np.float32)
KEY_POINTS = len(KEY_GRID)
VALUE_DIM = 128
HEADS = 4
FEATURE_DIM = 128
ON_AXIS_SIGMA = 1.5  # radius scale of the soft on-trace indicator in a memory value


def key_rows(hidden):
    """Decision-memory entry rows holding one entry's packed keys at width ``hidden``."""
    return math.ceil(KEY_POINTS*KEY_DIM/hidden)


def pack_keys(keys, hidden):
    """(N, KEY_POINTS, KEY_DIM) -> (N, key_rows, hidden), zero padded."""
    flat = keys.flatten(1)
    flat = F.pad(flat, (0, key_rows(hidden)*hidden-flat.shape[1]))
    return flat.reshape(len(keys), -1, hidden)


def unpack_keys(rows):
    """(..., key_rows, hidden) -> (..., KEY_POINTS, KEY_DIM)."""
    flat = rows.flatten(-2)[..., :KEY_POINTS*KEY_DIM]
    return flat.reshape(*rows.shape[:-2], KEY_POINTS, KEY_DIM)


class KeyHead(nn.Module):
    """Stem tokens -> unit-scale appearance keys on the token lattice; no position, replicate padding."""
    def __init__(self, hidden, width=96):
        super().__init__()
        conv = lambda i, o, k: nn.Conv3d(i, o, k, padding=k//2, padding_mode='replicate')
        self.layers = nn.Sequential(
            conv(hidden, width, 1), nn.GroupNorm(8, width), nn.GELU(),
            conv(width, width, 3), nn.GroupNorm(8, width), nn.GELU(),
            conv(width, width, 3), nn.GroupNorm(8, width), nn.GELU(),
            conv(width, KEY_DIM, 1))

    def forward(self, stem):
        keys = self.layers(stem)
        return F.layer_norm(keys.movedim(1, -1), (KEY_DIM,)).movedim(-1, 1)


class IdentityReadout(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.key_head = KeyHead(cfg.hidden)
        self.query = nn.Linear(KEY_DIM, KEY_DIM)
        self.key = nn.Linear(KEY_DIM, KEY_DIM)
        self.value = nn.Linear(KEY_DIM, VALUE_DIM)
        # Radius/8 and a soft on-trace indicator; log age and seed role of the entry.
        self.placement = nn.Sequential(nn.Linear(2, VALUE_DIM), nn.SiLU(), nn.Linear(VALUE_DIM, VALUE_DIM))
        self.role = nn.Linear(2, VALUE_DIM)
        self.output = nn.Linear(VALUE_DIM, VALUE_DIM)
        self.head = nn.Sequential(nn.LayerNorm(KEY_DIM+VALUE_DIM), nn.Linear(KEY_DIM+VALUE_DIM, 256), nn.GELU(),
                                  nn.Linear(256, FEATURE_DIM))
        self.norm = nn.LayerNorm(FEATURE_DIM)
        self.logit = nn.Linear(FEATURE_DIM, 1)
        radius = np.hypot(KEY_GRID[:, 0], KEY_GRID[:, 1])
        placement = np.stack((radius/8., np.exp(-.5*(radius/ON_AXIS_SIGMA)**2)), -1).astype(np.float32)
        self.register_buffer('grid', torch.from_numpy(KEY_GRID), persistent=False)
        self.register_buffer('grid_placement', torch.from_numpy(placement), persistent=False)

    def memory(self, entry_keys, valid, pose):
        """Memory from entry keys (B, SLOTS, KEY_POINTS, KEY_DIM), slot validity and pose (B, SLOTS, 14).

        A key sampled outside its crop is exactly zero and is masked, like an invalid slot."""
        keys = entry_keys.float()
        usable = valid[..., None] & (keys.abs().sum(-1) > 0)
        # Masked keys still enter attention with zero weight, so their values must be finite.
        keys = torch.where(usable[..., None], keys, 0.)
        role = torch.where(valid[..., None], pose[..., 12:14].float(), 0.)
        return dict(identity_memory_keys=keys.flatten(1, 2).to(torch.bfloat16),
                    identity_memory_padding=~usable.flatten(1),
                    identity_memory_role=role)

    def project(self, memory):
        """Per-head K/V of one memory (no position of the current crop enters)."""
        keys = memory['identity_memory_keys'].float()
        padding = memory['identity_memory_padding']
        b, n, _ = keys.shape
        slots = memory['identity_memory_role'].shape[1]
        placement = self.placement(self.grid_placement)[None, None]  # (1, 1, KEY_POINTS, V)
        role = self.role(memory['identity_memory_role'].float())[:, :, None]  # (B, SLOTS, 1, V)
        values = self.value(keys)+(placement+role).reshape(b, slots*len(self.grid), VALUE_DIM)
        k = self.key(keys).reshape(b, n, HEADS, -1).transpose(1, 2)
        v = values.reshape(b, n, HEADS, -1).transpose(1, 2)
        empty = padding.all(-1)
        allowed = (~padding | empty[:, None])[:, None, None, :]
        return k, v, allowed, empty

    def forward(self, queries, projected):
        """(B, N, FEATURE_DIM) features and (B, N) logits for current keys (B, N, KEY_DIM)."""
        k, v, allowed, empty = projected
        b, n, _ = queries.shape
        # Attention in the ambient (autocast) precision: a dense field reads ~20k queries against
        # ~5k memory keys, which needs a fused kernel; the classification itself runs in FP32.
        q = self.query(queries).reshape(b, n, HEADS, -1).transpose(1, 2)
        read = F.scaled_dot_product_attention(q, k.to(q.dtype), v.to(q.dtype), attn_mask=allowed)
        read = self.output(read.transpose(1, 2).reshape(b, n, -1))
        with torch.autocast(queries.device.type, enabled=False):
            read = torch.where(empty[:, None, None], 0., read.float())
            features = self.head(torch.cat((queries.float(), read), -1))
            return features, self.logit(F.gelu(self.norm(features))).squeeze(-1)

    def entry_keys(self, keys, sample):
        """(N, KEY_POINTS, KEY_DIM) keys of one decision's crop at KEY_GRID around its head."""
        grid = self.grid[None].expand(len(keys), -1, -1)
        return sample(keys.float(), grid)[0]
