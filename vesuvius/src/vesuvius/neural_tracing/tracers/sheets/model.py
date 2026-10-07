"""The front model: a CT crop per front point and a grid of CT tokens along the front curve, read by one transformer.

Each point token is its CT crop embedding (a residual CNN of ResNet-D basic blocks,
``vesuvius.models.build.resblocks.BasicBlockD``, a 1x1 conv to ``point_bottleneck`` channels, then a Linear) plus
rotation-invariant geometry (position and frame in the front frame, place along the front, previous confidence).
The grid holds small CT crops along the curve (columns) for the current row and the last ``memory_rows`` rows, each
through its own residual CNN plus its geometry. One transformer runs full self-attention over the point and grid
tokens together, each kind with a learned type embedding; with ``fiber_tokens`` the fibers near the front join the
sequence as a third kind (``Tracer._fiber_tokens``), and with ``fiber_follow_mode`` a learned mode embedding (0 sheet
tracing, 1 following a fiber) is added to every token. The head predicts a displacement in the point frame, in units
of ``step``, relative to a straight advance, a confidence logit, and a tilt of the point's curve tangent along its
advance and normal axes; it starts at zero (straight advance, Catmull-Rom tangents, p = 0.5).
"""
from __future__ import annotations

import math
from functools import partial

import torch
from torch import nn

from vesuvius.models.build.resblocks import BasicBlockD
from vesuvius.neural_tracing.tracers.sheets.data.episodes import FIBER_CODE_DIM
from vesuvius.neural_tracing.tracers.models.weights import load_matching_weights
from vesuvius.neural_tracing.tracers.shared.compile import compile_options

POINT_FEATURES = 16
TOKEN_FEATURES = 17
# Fiber tokens: position and direction in the front frame (3 + 3), family one-hot (2), seed-fiber flag (1) and the
# fiber's instance code (data.episodes.FIBER_CODE_DIM).
FIBER_FEATURES = 3+3+2+1+FIBER_CODE_DIM


def _groups(channels: int) -> int:
    return 4 if channels % 4 == 0 else 1


class CropEncoder(nn.Module):
    """Crop CNN of ResNet-D basic blocks (one per stage: two 3x3x3 convs, GroupNorm(4), GELU, a pooled/projected
    skip), optionally a 1x1 conv to ``bottleneck`` channels, then flatten and Linear to ``dim``. Every crop size
    must be divisible by 2 per stride-2 stage (the skip pools by flooring, the conv by ceiling)."""

    def __init__(self, crop, channels, strides, dim: int, bottleneck: int = 0):
        super().__init__()
        norm = partial(nn.GroupNorm, 4)
        layers, c_in, shape = [], 1, [int(c) for c in crop]
        for c, stride in zip(channels, strides):
            layers.append(BasicBlockD(nn.Conv3d, c_in, c, 3, int(stride), conv_bias=True, norm_op=norm, nonlin=nn.GELU))
            if int(stride) == 2:
                shape = [s//2 for s in shape]
            c_in = c
        if bottleneck:
            layers += [nn.Conv3d(c_in, bottleneck, 1), nn.GroupNorm(_groups(bottleneck), bottleneck), nn.GELU()]
            c_in = bottleneck
        self.net = nn.Sequential(*layers)
        self.proj = nn.Linear(c_in*math.prod(shape), dim)

    def forward(self, x):
        return self.proj(self.net(x).flatten(1))


class Block(nn.Module):
    """Pre-norm self-attention and MLP."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.n_self = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        hidden = int(dim*mlp_ratio)
        self.n_mlp = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim))

    def forward(self, x, mask=None):
        """``mask`` (B, L) bool: tokens of ``x`` no token attends to (padding)."""
        h = self.n_self(x)
        x = x+self.self_attn(h, h, h, key_padding_mask=mask, need_weights=False)[0]
        return x+self.mlp(self.n_mlp(x))


class SheetTracer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        dim = cfg.dim
        self.encoder = CropEncoder(cfg.crop, cfg.channels, cfg.strides, dim, cfg.point_bottleneck)
        self.point_geometry = nn.Sequential(nn.Linear(POINT_FEATURES, dim), nn.GELU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList(Block(dim, cfg.heads, cfg.mlp_ratio, cfg.dropout) for _ in range(cfg.layers))
        self.token_type = nn.Parameter(torch.randn(2, dim)*.02)  # point / grid tokens
        self.token_encoder = CropEncoder(cfg.token_crop, cfg.token_channels, cfg.token_strides, dim)
        self.token_geometry = nn.Sequential(nn.Linear(TOKEN_FEATURES, dim), nn.GELU(), nn.Linear(dim, dim))
        if cfg.fiber_tokens:  # fiber points near the front as a third kind of token
            self.fiber_geometry = nn.Sequential(nn.Linear(FIBER_FEATURES, dim), nn.GELU(), nn.Linear(dim, dim))
            self.fiber_type = nn.Parameter(torch.randn(dim)*.02)
        if cfg.fiber_follow_mode:  # 0: sheet tracing, 1: following a fiber; zero, so it starts as the checkpoint
            self.mode_embed = nn.Parameter(torch.zeros(2, dim))
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 6)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, crops, point_geom, grid, mode=None, fibers=None):
        """crops (B, N, 1, Sn, St, Sa) -> displacement (B, N, 3) as (t, a, n), confidence logit (B, N), tokens
        (B, N, D), the tangent tilt (B, N, 2) as (a, n), and the current grid row's tokens (B, Q, D).

        ``grid``: ``crops`` (B, Q, 1, ...) along the current curve, ``geom`` (B, R, Q, F) for all rows oldest first
        (the current row last) and ``past`` (B, R-1, Q, D) stored row tokens or None. ``mode`` (B,) long: the tracing
        mode of each episode (``model.fiber_follow_mode``). ``fibers``: ``feat`` (B, K, FIBER_FEATURES) and ``pad``
        (B, K) of the fiber tokens (``model.fiber_tokens``; ``Tracer._fiber_tokens``).
        """
        B, N = crops.shape[:2]
        x = self.encoder(crops.flatten(0, 1)).view(B, N, -1)+self.point_geometry(point_geom)
        mode_token = self.mode_embed[mode] if mode is not None and self.cfg.fiber_follow_mode else None
        if mode_token is not None:
            x = x+mode_token[:, None]
        Q = grid['crops'].shape[1]
        row = self.token_encoder(grid['crops'].flatten(0, 1)).view(B, Q, -1)
        rows = row[:, None] if grid['past'] is None else torch.cat([grid['past'].to(row.dtype), row[:, None]], 1)
        g = rows+self.token_geometry(grid['geom'])
        if mode_token is not None:
            g = g+mode_token[:, None, None]
        seq = torch.cat([x+self.token_type[0], (g+self.token_type[1]).flatten(1, 2)], 1)  # points, then the grid
        mask = None
        if fibers is not None and self.cfg.fiber_tokens:  # fiber tokens last; padding masked out
            tokens = self.fiber_geometry(fibers['feat'])+self.fiber_type
            if mode_token is not None:
                tokens = tokens+mode_token[:, None]
            mask = torch.cat([fibers['pad'].new_zeros(B, seq.shape[1]), fibers['pad']], 1)
            seq = torch.cat([seq, tokens.to(seq.dtype)], 1)
        for block in self.blocks:
            seq = block(seq, mask=mask)
        x = seq[:, :N]
        out = self.head(self.norm(x))
        return out[..., :3], out[..., 3], x, out[..., 4:6], row


def compile_model(model: SheetTracer) -> None:
    """``train.compile_model``: torch.compile the model in place (parameter names, and so checkpoints, are
    unchanged). Inductor fuses the norm, activation and dtype-cast passes around the convolutions and the small
    transformer ops, which cuts both GPU time and kernel launches; same math, different rounding. One graph per
    grid-memory depth (and per grad mode for validation). Compiler options: ``shared.compile.compile_options``."""
    model.compile(**compile_options())


def build_model(cfg) -> SheetTracer:
    return SheetTracer(cfg.model)


OPTIONAL_MODULES = ('mode_', 'fiber_')  # modules a checkpoint without them starts fresh (``init_model_state``)


def init_model_state(model: SheetTracer, saved: dict) -> None:
    """``--init``: like ``load_model_state``, but the optional modules the checkpoint predates (``mode_embed``,
    ``fiber_*``) keep their fresh initialisation; any other missing or unexpected weight is still an error."""
    report = load_matching_weights(model, saved)  # models.weights, as the fiber tracer's --init
    if report['unexpected'] or report['mismatched'] or any(not k.startswith(OPTIONAL_MODULES) for k in report['fresh']):
        load_model_state(model, saved)  # the usual informative mismatch error
    if report['fresh']:
        print(f'new modules initialised fresh: {sorted({k.split(".")[0] for k in report["fresh"]})}', flush=True)


def load_model_state(model: SheetTracer, saved: dict) -> None:
    model.load_state_dict(saved)
