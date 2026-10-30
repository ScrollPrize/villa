"""The front model: a CT crop per front point and a grid of CT tokens along the front curve, read by one transformer.

Each point token is its CT crop embedding (``models.blocks.CropCNN``: residual ResNet-D stages with GroupNorm and
GELU, a 1x1 conv to ``point_bottleneck`` channels, then a Linear) plus
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

import torch
from torch import nn

from vesuvius.neural_tracing.tracers.sheets.data.episodes import FIBER_CODE_DIM
from vesuvius.neural_tracing.tracers.models.blocks import MLP, CropCNN, SelfAttentionLayer
from vesuvius.neural_tracing.tracers.models.weights import add_input_channels, load_matching_weights
from vesuvius.neural_tracing.tracers.shared.compile import compile_options

POINT_FEATURES = 16
TOKEN_FEATURES = 17
# Fiber tokens: position and direction in the front frame (3 + 3), family one-hot (2), seed-fiber flag (1) and the
# fiber's instance code (data.episodes.FIBER_CODE_DIM).
FIBER_FEATURES = 3+3+2+1+FIBER_CODE_DIM


def crop_encoder(crop, channels, strides, dim: int, bottleneck: int = 0, in_channels: int = 1) -> CropCNN:
    """One token per crop: a ResNet-D basic block per stage (two 3x3x3 convs, GroupNorm(4), GELU, a pooled/projected
    skip), optionally a 1x1 conv to ``bottleneck`` channels, then flatten and Linear to ``dim``. Every crop size
    must be divisible by 2 per stride-2 stage."""
    return CropCNN(channels, strides, (1,)*len(channels), norm='group', activation='gelu', conv_bias='all', head='crop',
                   dim=dim, crop=crop, bottleneck=bottleneck, in_channels=in_channels)


class SheetTracer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        dim = cfg.dim
        inputs = 2 if cfg.neighbor_channel else 1  # CT, and the neighbouring surface (model.neighbor_channel)
        self.encoder = crop_encoder(cfg.crop, cfg.channels, cfg.strides, dim, cfg.point_bottleneck, inputs)
        self.point_geometry = MLP(POINT_FEATURES, dim, nn.GELU)
        self.blocks = nn.ModuleList(SelfAttentionLayer(dim, cfg.heads, int(dim*cfg.mlp_ratio), cfg.qk_norm, cfg.qk_logit_scale)
                                    for _ in range(cfg.layers))
        self.token_type = nn.Parameter(torch.randn(2, dim)*.02)  # point / grid tokens
        self.token_encoder = crop_encoder(cfg.token_crop, cfg.token_channels, cfg.token_strides, dim, in_channels=inputs)
        self.token_geometry = MLP(TOKEN_FEATURES, dim, nn.GELU)
        if cfg.fiber_tokens:  # fiber points near the front as a third kind of token
            self.fiber_geometry = MLP(FIBER_FEATURES, dim, nn.GELU)
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
            seq = block(seq, mask)
        x = seq[:, :N]
        out = self.head(self.norm(x))
        return out[..., :3], out[..., 3], x, out[..., 4:6], row


def compile_model(model: SheetTracer) -> None:
    """``train.compile_model``: torch.compile the model in place (parameter names, and so checkpoints, are
    unchanged). Inductor fuses the norm, activation and dtype-cast passes around the convolutions and the small
    transformer ops, which cuts both GPU time and kernel launches; same math, different rounding. One graph per
    input shape: grid-memory depth, fiber tokens or none, batch size (training, neighbour rollouts, validation) and
    grad mode; with one front size these stay a few dozen, so the recompile limit is raised to ``RECOMPILE_LIMIT``
    (past the limit, new shapes would quietly run eager). Compiler options: ``shared.compile.compile_options``."""
    import torch._dynamo
    for name in ('recompile_limit', 'cache_size_limit'):
        if hasattr(torch._dynamo.config, name):
            setattr(torch._dynamo.config, name, max(getattr(torch._dynamo.config, name), RECOMPILE_LIMIT))
    model.compile(**compile_options())


RECOMPILE_LIMIT = 64


def build_model(cfg) -> SheetTracer:
    return SheetTracer(cfg.model)


# Transformer block parameter names before the blocks became models.blocks.SelfAttentionLayer (nn.MultiheadAttention,
# named LayerNorms, an MLP with a dropout layer); old checkpoints load renamed (``upgrade_state``).
LEGACY_BLOCK_NAMES = (('.n_self.', '.norm1.'), ('.self_attn.in_proj_', '.qkv.'), ('.self_attn.out_proj.', '.out.'),
                      ('.n_mlp.', '.norm2.'), ('.mlp.0.', '.ffn.0.'), ('.mlp.3.', '.ffn.2.'))
ENCODERS = ('encoder', 'token_encoder')


def legacy_state(saved: dict) -> bool:
    """Whether ``saved`` predates the shared transformer layer (its optimizer state does not match the model). The
    crop encoders' renaming (``_encoder_renames``) keeps parameter order and shapes, so it does not count."""
    return any(old in key for key in saved for old, _ in LEGACY_BLOCK_NAMES)


def _encoder_renames(saved: dict) -> dict:
    """Old -> new key prefixes of the crop encoders from before models.blocks.CropCNN (``net.K`` a residual stage
    while it has ``conv1``, then the bottleneck's conv and norm; ``proj`` the head)."""
    renames = {}
    for e in ENCODERS:
        stages = sorted({int(k.split('.')[2]) for k in saved if k.startswith(f'{e}.net.') and '.conv1.' in k})
        for i in stages:
            renames[f'{e}.net.{i}.'] = f'{e}.stages.{i}.blocks.0.'
        nxt = len(stages)
        renames[f'{e}.net.{nxt}.'] = f'{e}.bottleneck.0.'  # the 1x1 conv, its norm (the GELU has no weights)
        renames[f'{e}.net.{nxt+1}.'] = f'{e}.bottleneck.1.'
        renames[f'{e}.proj.'] = f'{e}.head.'
    return renames


def upgrade_state(saved: dict) -> dict:
    """``saved`` with weights of earlier layouts renamed (same tensors): transformer blocks of the earlier layer for
    SelfAttentionLayer (``in_proj`` holds q, k, v in the order ``qkv`` splits them), and the earlier crop encoders
    for models.blocks.CropCNN."""
    encoders = _encoder_renames(saved)

    def rename(key):
        for old, new in LEGACY_BLOCK_NAMES:
            if key.startswith('blocks.') and old in key:
                return key.replace(old, new)
        for old, new in encoders.items():
            if key.startswith(old):
                return new+key[len(old):]
        return key
    return {rename(k): v for k, v in saved.items()}


OPTIONAL_MODULES = ('mode_', 'fiber_')  # modules a checkpoint without them starts fresh (``init_model_state``)


def init_model_state(model: SheetTracer, saved: dict) -> None:
    """``--init``: like ``load_model_state``, but the optional modules the checkpoint predates (``mode_embed``,
    ``fiber_*``) keep their fresh initialisation, and input channels it lacks start at zero
    (``add_input_channels``); any other missing or unexpected weight is still an error."""
    saved = add_input_channels(model, upgrade_state(saved))
    report = load_matching_weights(model, saved)  # models.weights, as the fiber tracer's --init
    if report['unexpected'] or report['mismatched'] or any(not k.startswith(OPTIONAL_MODULES) for k in report['fresh']):
        load_model_state(model, saved)  # the usual informative mismatch error
    if report['fresh']:
        print(f'new modules initialised fresh: {sorted({k.split(".")[0] for k in report["fresh"]})}', flush=True)


def load_model_state(model: SheetTracer, saved: dict) -> None:
    model.load_state_dict(upgrade_state(saved))
