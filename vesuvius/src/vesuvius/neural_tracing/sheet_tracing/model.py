"""The front model: one shared crop encoder per point, then attention along the front and over past rows.

Each point token is its CT crop embedding plus rotation-invariant geometry (position and frame in the front
frame, place along the front, previous confidence). Blocks apply self-attention across the front points and cross-
attention to the last ``memory_rows`` rows of tokens; ``attention=False`` keeps only the per-point MLPs (the
independent-point ablation). With ``surface_tokens`` the memory is instead a 2D grid of CT tokens sampled along
the curve (columns) over the current and past rows, contextualised by axial row/column attention; each point
writes its crop token into the grid at its column and cross-attends to the whole grid. Optionally each column's
token also gets a long profile along the normal (``profile_crop``, the stack of neighbouring sheets), and the grid
memory holds rows at chosen lags plus the trace's first row (``memory_lags``, ``memory_anchor``). With
``joint_transformer``, one transformer instead runs full self-attention over the point and grid tokens together
(each with a learned type embedding), replacing the axial grid blocks and the point blocks. ``residual`` builds the
crop CNNs from ResNet-D basic blocks (``vesuvius.models.build.resblocks.BasicBlockD``), and ``point_bottleneck``
reduces the point CNN's channels with a 1x1 conv before its flattening Linear. The head predicts a displacement in the point frame, in units of ``step``,
relative to a straight advance, a confidence logit, and with ``tangents`` a tilt of the point's curve tangent
along its advance and normal axes; it starts at zero (straight advance, Catmull-Rom tangents, p = 0.5).
"""
from __future__ import annotations

import math
from functools import partial

import torch
from torch import nn

from vesuvius.models.build.resblocks import BasicBlockD
from vesuvius.neural_tracing.sheet_tracing.data.episodes import FIBER_CODE_DIM

POINT_FEATURES = 16
MEMORY_FEATURES = 8
TOKEN_FEATURES = 17
LAG_FEATURES = 2  # with model.memory_lags: log lag and anchor flag of each grid row
# Fiber tokens: position and direction in the front frame (3 + 3), family one-hot (2), seed-fiber flag (1) and the
# fiber's instance code (data.episodes.FIBER_CODE_DIM).
FIBER_FEATURES = 3+3+2+1+FIBER_CODE_DIM


def _groups(channels: int) -> int:
    return 4 if channels % 4 == 0 else 1


class CropEncoder(nn.Module):
    def __init__(self, crop, channels, strides, dim: int, in_channels: int = 1):
        super().__init__()
        layers, c_in, shape = [], in_channels, [int(c) for c in crop]
        for c, stride in zip(channels, strides):
            stride = int(stride)
            layers += [nn.Conv3d(c_in, c, 3, stride, 1), nn.GroupNorm(_groups(c), c), nn.GELU()]
            if stride == 2:
                shape = [(s+1)//2 for s in shape]
            c_in = c
        self.net = nn.Sequential(*layers)
        self.proj = nn.Linear(c_in*math.prod(shape), dim)

    def forward(self, x):
        return self.proj(self.net(x).flatten(1))


class ResidualCropEncoder(nn.Module):
    """Crop CNN of ResNet-D basic blocks (one per stage: two 3x3x3 convs, GroupNorm(4), GELU, a pooled/projected
    skip), optionally a 1x1 conv to ``bottleneck`` channels, then flatten and Linear to ``dim``. Every crop size
    must be divisible by 2 per stride-2 stage (the skip pools by flooring, the conv by ceiling)."""

    def __init__(self, crop, channels, strides, dim: int, in_channels: int = 1, bottleneck: int = 0):
        super().__init__()
        norm = partial(nn.GroupNorm, 4)
        layers, c_in, shape = [], in_channels, [int(c) for c in crop]
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


def crop_encoder(cfg, crop, channels, strides, dim: int, in_channels: int, bottleneck: int = 0) -> nn.Module:
    if cfg.residual:
        return ResidualCropEncoder(crop, channels, strides, dim, in_channels, bottleneck)
    if bottleneck:
        raise ValueError('model.point_bottleneck needs model.residual')
    return CropEncoder(crop, channels, strides, dim, in_channels)


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float, dropout: float, attention: bool, cross: bool = True):
        super().__init__()
        self.attention = attention
        if attention:
            self.n_self = nn.LayerNorm(dim)
            self.self_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        if attention and cross:
            self.n_cross, self.n_mem = nn.LayerNorm(dim), nn.LayerNorm(dim)
            self.cross_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        hidden = int(dim*mlp_ratio)
        self.n_mlp = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim))

    def forward(self, x, memory=None, mask=None):
        """``mask`` (B, L) bool: tokens of ``x`` no token attends to (padding)."""
        if self.attention:
            h = self.n_self(x)
            x = x+self.self_attn(h, h, h, key_padding_mask=mask, need_weights=False)[0]
            if memory is not None:
                m = self.n_mem(memory)
                x = x+self.cross_attn(self.n_cross(x), m, m, need_weights=False)[0]
        return x+self.mlp(self.n_mlp(x))


class AxialBlock(nn.Module):
    """Attention along each row (across curve columns), then along each column (across rows), then an MLP."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.n_row, self.n_col, self.n_mlp = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.row = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.col = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        hidden = int(dim*mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim))

    def forward(self, g):
        B, R, Q, D = g.shape
        h = self.n_row(g).reshape(B*R, Q, D)
        g = g+self.row(h, h, h, need_weights=False)[0].reshape(B, R, Q, D)
        h = self.n_col(g).transpose(1, 2).reshape(B*Q, R, D)
        g = g+self.col(h, h, h, need_weights=False)[0].reshape(B, Q, R, D).transpose(1, 2)
        return g+self.mlp(self.n_mlp(g))


class SheetTracer(nn.Module):
    def __init__(self, cfg, tangents: bool = False, in_channels: int = 1):
        super().__init__()
        self.cfg = cfg
        dim = cfg.dim
        self.encoder = crop_encoder(cfg, cfg.crop, cfg.channels, cfg.strides, dim, in_channels, cfg.point_bottleneck)
        self.point_geometry = nn.Sequential(nn.Linear(POINT_FEATURES, dim), nn.GELU(), nn.Linear(dim, dim))
        if not cfg.surface_tokens:  # the grid replaces memory over past point tokens
            self.memory_geometry = nn.Sequential(nn.Linear(MEMORY_FEATURES, dim), nn.GELU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList(Block(dim, cfg.heads, cfg.mlp_ratio, cfg.dropout, cfg.attention,
                                          cross=not cfg.joint_transformer) for _ in range(cfg.layers))
        if cfg.joint_transformer:  # learned point / grid token types for the single transformer
            self.token_type = nn.Parameter(torch.randn(2, dim)*.02)
        if cfg.context_crop:
            self.context_encoder = crop_encoder(cfg, cfg.context_crop, cfg.context_channels, cfg.context_strides, dim,
                                                in_channels)
            self.context_proj = nn.Linear(dim, dim)
            nn.init.zeros_(self.context_proj.weight)
            nn.init.zeros_(self.context_proj.bias)
        if cfg.surface_tokens:
            self.token_encoder = crop_encoder(cfg, cfg.token_crop, cfg.token_channels, cfg.token_strides, dim,
                                              in_channels)
            features = TOKEN_FEATURES+(LAG_FEATURES if cfg.memory_lags else 0)
            self.token_geometry = nn.Sequential(nn.Linear(features, dim), nn.GELU(), nn.Linear(dim, dim))
            if not cfg.joint_transformer:  # the points read the grid by cross-attention after axial blocks
                self.point_to_grid = nn.Linear(dim, dim)
                self.grid_blocks = nn.ModuleList(AxialBlock(dim, cfg.heads, cfg.mlp_ratio, cfg.dropout)
                                                 for _ in range(cfg.grid_layers))
            if cfg.profile_crop:
                self.profile_encoder = CropEncoder(cfg.profile_crop, cfg.profile_channels, cfg.profile_strides, dim,
                                                   in_channels)
                self.profile_proj = nn.Linear(dim, dim)
                nn.init.zeros_(self.profile_proj.weight)
                nn.init.zeros_(self.profile_proj.bias)
        if cfg.fiber_tokens:  # fiber points near the front as extra tokens of the joint transformer
            self.fiber_geometry = nn.Sequential(nn.Linear(FIBER_FEATURES, dim), nn.GELU(), nn.Linear(dim, dim))
            self.fiber_type = nn.Parameter(torch.randn(dim)*.02)
        if cfg.fiber_follow_mode:  # 0: sheet tracing, 1: following a fiber; zero, so it starts as the checkpoint
            self.mode_embed = nn.Parameter(torch.zeros(2, dim))
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 6 if tangents else 4)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, crops, point_geom, memory_tokens=None, memory_geom=None, grid=None, context=None, mode=None,
                fibers=None):
        """crops (B, N, 1, Sn, St, Sa) -> displacement (B, N, 3) as (t, a, n), confidence logit, tokens, the
        tangent tilt (B, N, 2) as (a, n) or None, and the current grid row's tokens (B, Q, D) or None.

        ``grid`` (surface tokens): ``crops`` (B, Q, 1, ...) along the current curve, ``geom`` (B, R, Q, F) for
        all rows oldest first (the current row last), ``past`` (B, R-1, Q, D) stored row tokens or None, and
        ``point_cols`` (N,) the front points' grid columns, and ``profile`` (B, Q, 1, ...) normal profiles at the
        current row's columns (with ``profile_crop``). ``mode`` (B,) long: the tracing mode of each episode
        (``model.fiber_follow_mode``), added to every token as a learned embedding. ``fibers``: ``feat`` (B, K,
        FIBER_FEATURES) and ``pad`` (B, K) of the fiber tokens (``model.fiber_tokens``; ``Tracer._fiber_tokens``).
        """
        B, N = crops.shape[:2]
        own = self.encoder(crops.flatten(0, 1)).view(B, N, -1)
        x = own+self.point_geometry(point_geom)
        mode_token = self.mode_embed[mode] if mode is not None and self.cfg.fiber_follow_mode else None
        if mode_token is not None:
            x = x+mode_token[:, None]
        if context is not None and self.cfg.context_crop:
            x = x+self.context_proj(self.context_encoder(context.flatten(0, 1)).view(B, N, -1))
        memory, row = None, None
        if grid is not None:
            Q = grid['crops'].shape[1]
            row = self.token_encoder(grid['crops'].flatten(0, 1)).view(B, Q, -1)
            if grid.get('profile') is not None:
                row = row+self.profile_proj(self.profile_encoder(grid['profile'].flatten(0, 1)).view(B, Q, -1))
            if not self.cfg.joint_transformer:
                row = row.index_add(1, grid['point_cols'], self.point_to_grid(own))
            rows = row[:, None] if grid['past'] is None else torch.cat([grid['past'].to(row.dtype), row[:, None]], 1)
            g = rows+self.token_geometry(grid['geom'])
            if mode_token is not None:
                g = g+mode_token[:, None, None]
            if self.cfg.joint_transformer:  # one sequence: the N point tokens, then every grid token
                seq = torch.cat([x+self.token_type[0], (g+self.token_type[1]).flatten(1, 2)], 1)
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
                return out[..., :3], out[..., 3], x, (out[..., 4:6] if out.shape[-1] == 6 else None), row
            for block in self.grid_blocks:
                g = block(g)
            memory = g.flatten(1, 2)
        elif memory_tokens is not None and self.cfg.attention:
            memory = memory_tokens+self.memory_geometry(memory_geom)
        for block in self.blocks:
            x = block(x, memory)
        out = self.head(self.norm(x))
        return out[..., :3], out[..., 3], x, (out[..., 4:6] if out.shape[-1] == 6 else None), row


class StraightAdvance(nn.Module):
    """Baseline: every point advances straight along its frame, always confident."""

    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim

    def forward(self, crops, point_geom, memory_tokens=None, memory_geom=None, grid=None, context=None, mode=None,
                fibers=None):
        B, N = crops.shape[:2]
        zeros = crops.new_zeros(B, N, 3)
        return zeros, crops.new_full((B, N), 10.), crops.new_zeros(B, N, self.dim), None, None


def compile_model(model: SheetTracer) -> None:
    """``train.compile_model``: torch.compile the model in place (parameter names, and so checkpoints, are
    unchanged). Inductor fuses the norm, activation and dtype-cast passes around the convolutions and the small
    transformer ops, which cuts both GPU time and kernel launches; same math, different rounding. One graph per
    grid-memory depth (and per grad mode for validation)."""
    model.compile(dynamic=False)


def build_model(cfg) -> SheetTracer:
    return SheetTracer(cfg.model, tangents=cfg.trace.curve == 'hermite', in_channels=1+len(cfg.data.extra_volumes))


def init_model_state(model: SheetTracer, saved: dict) -> None:
    """``--init``: like ``load_model_state``, but optional modules the checkpoint predates (``context_*``,
    ``profile_*``, ``mode_embed``, ``fiber_*``) keep their fresh initialisation, and first convs or linears that gained inputs
    (input channels from ``data.extra_volumes``, grid geometry features from ``model.memory_lags``) get zero weights
    for them, so the model starts out unchanged; any other missing or unexpected weight is still an error."""
    expected = model.state_dict()
    saved = dict(saved)
    for key, value in saved.items():  # weights that gained inputs: the new inputs start at zero
        want = expected.get(key)
        if (want is not None and value.ndim == want.ndim and value.ndim in (2, 5) and value.shape[1] < want.shape[1]
                and value.shape[0] == want.shape[0] and value.shape[2:] == want.shape[2:]):
            padded = torch.zeros_like(want)
            padded[:, :value.shape[1]] = value
            saved[key] = padded
            print(f'{key}: inputs {value.shape[1]} -> {want.shape[1]} (new ones zero)', flush=True)
    missing = [k for k in expected if k not in saved]
    if not missing:
        load_model_state(model, saved)
        return
    if any(not k.startswith(('context_', 'profile_', 'mode_', 'fiber_')) for k in missing):
        load_model_state(model, saved)  # raise the usual informative mismatch error
        return
    result = model.load_state_dict(saved, strict=False)
    if result.unexpected_keys:
        raise ValueError(f'Unexpected weights in the initial checkpoint: {result.unexpected_keys}')
    print(f'new modules initialised fresh: {sorted({k.split(".")[0] for k in missing})}', flush=True)


def load_model_state(model: SheetTracer, saved: dict) -> None:
    """Load current weights or weights saved before the front-point module names changed."""
    expected = model.state_dict()
    if saved.keys() == expected.keys():
        model.load_state_dict(saved)
        return
    if len(saved) != len(expected):
        model.load_state_dict(saved)  # raise the usual informative mismatch error
        return
    pairs = list(zip(saved.items(), expected.items()))
    renamed_roots = {'point_geometry', 'point_to_grid'}
    valid = all(
        old_value.shape == new_value.shape
        and (old_key == new_key or (
            new_key.split('.', 1)[0] in renamed_roots
            and old_key.partition('.')[2] == new_key.partition('.')[2]
            and old_key not in expected
        ))
        for (old_key, old_value), (new_key, new_value) in pairs
    )
    if not valid:
        model.load_state_dict(saved)  # raise the usual informative mismatch error
        return
    model.load_state_dict({new_key: old_value for (_, old_value), (new_key, _) in pairs})
