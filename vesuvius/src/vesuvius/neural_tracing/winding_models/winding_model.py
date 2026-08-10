"""Winding phase model over a ray-aligned 3-D slab.

The input is a single [H, W, L] slab sampled around a ray: transverse axes
H and W across the ray, the ray axis L at full volume resolution. A 3-D
conv encoder halves only the transverse axes, so every downstream feature
keeps per-sample localization along the ray. A short stem of plain
residual blocks mixes features locally, and a stack of axial attention
blocks does the long-range work: one attention over the ray axis per
column (relative position bias, as a sequence of length L) and one over
the transverse plane per ray position
(2-D relative bias). A light decoder restores one transverse level with a
skip connection, and every prediction head runs per column — a
ray-parallel line of voxels at the model's column stride — so the network
predicts winding structure everywhere in the slab rather than only on the
central ray. The central ray is only the sampling frame; it receives no
special readout or supervision.

Per column sample the model predicts:

- a monotone relative winding phase: softplus increments accumulated
  along the ray axis. Each increment is the exact winding-density integral
  over its segment -- the quantity fit_spiral's density loss consumes.
  Winding indices are globally consistent across a slab's columns, so the
  whole phase field shares one free offset, absorbed by the shift-invariant
  loss and the consumer's per-ray registration.
- a per-sample log-variance for those increments (heteroscedastic head),
  so the consumer can precision-weight registered observations instead
  of trusting every column equally.
- a crossing logit whose sigmoid peaks where the phase passes an integer.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

torch.set_float32_matmul_precision('high')


_KNOWN_CONFIG_KEYS = frozenset(
    {
        "encoder_channels",
        "trunk_dim",
        "trunk_stem_blocks",
        "axial_attention_blocks",
        "attention_heads",
        "max_relative_distance",
        "transverse_size",
        "decoder_dim",
        "crossing_head_kernel_size",
        "crossing_prior_prob",
        "phase_initial_increment",
        "density_log_variance_init",
    }
)


class ChannelLayerNorm(nn.LayerNorm):
    """Per-position LayerNorm over channels for channels-first conv features.

    Unlike GroupNorm, its statistics never pool over space, so zero-filled
    invalid regions of a slab cannot shift the normalization of valid voxels.
    """

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return super().forward(features.movedim(1, -1)).movedim(-1, 1)


def _norm(width: int) -> nn.Module:
    return ChannelLayerNorm(width)


class EncoderStage(nn.Module):
    """One transverse halving: strided 3x3x3 conv pair, ray axis untouched."""

    def __init__(self, in_channels: int, width: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(in_channels, width, 3, stride=(2, 2, 1), padding=1),
            _norm(width),
            nn.GELU(),
            nn.Conv3d(width, width, 3, padding=1),
            _norm(width),
            nn.GELU(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


class ResidualBlock(nn.Module):
    """Plain local residual refinement ahead of the attention stack.

    Long-range context along the ray is the attention blocks' job; the stem
    only gives them locally mixed features to attend over.
    """

    def __init__(self, width: int):
        super().__init__()
        self.conv1 = nn.Conv3d(width, width, 3, padding=1)
        self.norm1 = _norm(width)
        self.conv2 = nn.Conv3d(width, width, 3, padding=1)
        self.norm2 = _norm(width)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = self.conv1(features)
        residual = F.gelu(self.norm1(residual))
        residual = self.norm2(self.conv2(residual))
        return features + residual


class RelativeAttention1D(nn.Module):
    """Multi-head self-attention over one axis with relative position bias."""

    def __init__(self, dim: int, num_heads: int, max_distance: int):
        super().__init__()
        if dim % num_heads:
            raise ValueError("attention dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.max_distance = max_distance
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.relative_bias = nn.Parameter(
            torch.zeros(num_heads, 2 * max_distance + 1)
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, length, dim = tokens.shape
        qkv = (
            self.qkv(tokens)
            .reshape(batch, length, 3, self.num_heads, dim // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        positions = torch.arange(length, device=tokens.device)
        offsets = (positions[None, :] - positions[:, None]).clamp(
            -self.max_distance, self.max_distance
        )
        bias = self.relative_bias[:, offsets + self.max_distance]
        attended = F.scaled_dot_product_attention(
            qkv[0], qkv[1], qkv[2], attn_mask=bias.to(qkv.dtype)
        )
        return self.proj(attended.transpose(1, 2).reshape(batch, length, dim))


class TransverseAttention(nn.Module):
    """Self-attention over the H x W transverse plane at each ray position.

    The plane is small at the trunk resolution, so full plane attention is
    cheap and mixes both transverse axes in one hop; a factored 2-D
    relative bias keeps the geometry of the plane.
    """

    def __init__(self, dim: int, num_heads: int, max_distance: int):
        super().__init__()
        if dim % num_heads:
            raise ValueError("attention dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.max_distance = max_distance
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.relative_bias = nn.Parameter(
            torch.zeros(num_heads, 2 * max_distance + 1, 2 * max_distance + 1)
        )

    def _plane_bias(self, height: int, width: int, device) -> torch.Tensor:
        rows = torch.arange(height, device=device)
        cols = torch.arange(width, device=device)
        row_offsets = (rows[:, None] - rows[None, :]).clamp(
            -self.max_distance, self.max_distance
        ) + self.max_distance
        col_offsets = (cols[:, None] - cols[None, :]).clamp(
            -self.max_distance, self.max_distance
        ) + self.max_distance
        bias = self.relative_bias[
            :, row_offsets[:, None, :, None], col_offsets[None, :, None, :]
        ]
        return bias.reshape(self.num_heads, height * width, height * width)

    def forward(
        self, tokens: torch.Tensor, height: int, width: int
    ) -> torch.Tensor:
        """tokens [N, height * width, C] -> same shape."""
        batch, length, dim = tokens.shape
        qkv = (
            self.qkv(tokens)
            .reshape(batch, length, 3, self.num_heads, dim // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        bias = self._plane_bias(height, width, tokens.device)
        attended = F.scaled_dot_product_attention(
            qkv[0], qkv[1], qkv[2], attn_mask=bias[None].to(qkv.dtype)
        )
        return self.proj(attended.transpose(1, 2).reshape(batch, length, dim))


class AxialAttentionBlock(nn.Module):
    """Pre-LN block: ray-axis attention, transverse-plane attention, MLP.

    Operates channels-last on [B, H, W, L, C]. ``residual_std`` sets the
    initial scale of every residual branch's output projection
    (GPT-2/Megatron depth scaling) so the trunk stream stays well-scaled
    while every branch receives gradient from step zero.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        max_ray_distance: int,
        max_transverse_distance: int,
        residual_std: float,
    ):
        super().__init__()
        self.ray_norm = nn.LayerNorm(dim)
        self.ray_attention = RelativeAttention1D(dim, num_heads, max_ray_distance)
        self.transverse_norm = nn.LayerNorm(dim)
        self.transverse_attention = TransverseAttention(
            dim, num_heads, max_transverse_distance
        )
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim)
        )
        for projection in (
            self.ray_attention.proj,
            self.transverse_attention.proj,
            self.mlp[-1],
        ):
            nn.init.normal_(projection.weight, std=residual_std)
            nn.init.zeros_(projection.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, height, width, length, dim = tokens.shape

        columns = self.ray_norm(tokens).reshape(-1, length, dim)
        tokens = tokens + self.ray_attention(columns).reshape(tokens.shape)

        planes = (
            self.transverse_norm(tokens)
            .permute(0, 3, 1, 2, 4)
            .reshape(batch * length, height * width, dim)
        )
        planes = self.transverse_attention(planes, height, width)
        tokens = tokens + planes.reshape(
            batch, length, height, width, dim
        ).permute(0, 2, 3, 1, 4)

        return tokens + self.mlp(self.mlp_norm(tokens))


class WindingModel(nn.Module):
    def __init__(self, cfg: dict | None = None):
        super().__init__()
        cfg = dict(cfg or {})
        unknown = set(cfg) - _KNOWN_CONFIG_KEYS
        if unknown:
            raise ValueError(f"unknown model config keys: {sorted(unknown)}")
        channels = [int(width) for width in cfg.get("encoder_channels", (32, 64, 96))]
        if len(channels) < 2:
            raise ValueError("the decoder skip needs at least two encoder stages")
        trunk_dim = int(cfg.get("trunk_dim", 192))
        decoder_dim = int(cfg.get("decoder_dim", 96))
        num_heads = int(cfg.get("attention_heads", 6))
        max_ray_distance = int(cfg.get("max_relative_distance", 128))
        # The transverse relative bias always covers the whole trunk plane;
        # its extent follows from the slab's transverse size (injected by the
        # trainer) and the encoder's downsampling, never set by hand.
        transverse_size = int(cfg.get("transverse_size", 96))
        downsample = 2 ** len(channels)
        if transverse_size < downsample or transverse_size % downsample:
            raise ValueError(
                "transverse_size must be a positive multiple of the encoder's"
                f" total transverse downsampling ({downsample})"
            )
        max_transverse_distance = transverse_size // downsample - 1
        stem_blocks = int(cfg.get("trunk_stem_blocks", 2))
        if stem_blocks < 0:
            raise ValueError("trunk_stem_blocks must be non-negative")
        attention_blocks = int(cfg.get("axial_attention_blocks", 4))
        if attention_blocks < 1:
            raise ValueError("at least one axial attention block is required")

        # The image and validity channels enter together; each encoder stage
        # halves the transverse axes only.
        self.stages = nn.ModuleList()
        previous = 2
        for width in channels:
            self.stages.append(EncoderStage(previous, width))
            previous = width

        self.trunk_in = nn.Sequential(
            nn.Conv3d(channels[-1], trunk_dim, 1),
            _norm(trunk_dim),
            nn.GELU(),
        )
        self.stem_blocks = nn.ModuleList(
            ResidualBlock(trunk_dim) for _ in range(stem_blocks)
        )
        residual_std = 0.02 / math.sqrt(max(1, 3 * attention_blocks))
        self.attention_blocks = nn.ModuleList(
            AxialAttentionBlock(
                trunk_dim,
                num_heads,
                max_ray_distance,
                max_transverse_distance,
                residual_std,
            )
            for _ in range(attention_blocks)
        )

        self.decoder = nn.Sequential(
            nn.Conv3d(trunk_dim + channels[-2], decoder_dim, 3, padding=1),
            _norm(decoder_dim),
            nn.GELU(),
            nn.Conv3d(decoder_dim, decoder_dim, 3, padding=1),
            _norm(decoder_dim),
            nn.GELU(),
        )

        self.phase_head = nn.Conv3d(decoder_dim, 1, 1)
        self.log_variance_head = nn.Conv3d(decoder_dim, 1, 1)
        crossing_kernel = int(cfg.get("crossing_head_kernel_size", 5))
        if crossing_kernel <= 0 or crossing_kernel % 2 == 0:
            raise ValueError("crossing_head_kernel_size must be a positive odd integer")
        self.crossing_head = nn.Conv3d(
            decoder_dim,
            1,
            kernel_size=(1, 1, crossing_kernel),
            padding=(0, 0, crossing_kernel // 2),
        )

        crossing_prior = cfg.get("crossing_prior_prob")
        if crossing_prior is not None:
            crossing_prior = float(crossing_prior)
            if not 0.0 < crossing_prior < 1.0:
                raise ValueError("crossing_prior_prob must be between zero and one")
            nn.init.constant_(
                self.crossing_head.bias,
                math.log(crossing_prior / (1.0 - crossing_prior)),
            )

        phase_increment = cfg.get("phase_initial_increment")
        if phase_increment is not None:
            phase_increment = float(phase_increment)
            if phase_increment <= 0.0:
                raise ValueError("phase_initial_increment must be positive")
            nn.init.zeros_(self.phase_head.weight)
            nn.init.constant_(
                self.phase_head.bias,
                math.log(math.expm1(phase_increment)),
            )

        # Start with a moderate predicted variance: confident enough that the
        # density NLL has teeth from early on, loose enough that early large
        # residuals don't dominate the total loss.
        nn.init.zeros_(self.log_variance_head.weight)
        nn.init.constant_(
            self.log_variance_head.bias,
            float(cfg.get("density_log_variance_init", -4.0)),
        )

    @property
    def column_stride(self) -> int:
        """Input voxels per output column on each transverse axis."""
        return 2 ** (len(self.stages) - 1)

    def forward(
        self, slab_image: torch.Tensor, slab_valid: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """slab_image [B, H, W, L] raw intensities, slab_valid same-shape bool.

        H and W are the transverse axes (must be divisible by twice the
        column stride), L is the ray axis. Outputs are per column at the
        column stride: [B, H / stride, W / stride, L].
        """
        image = slab_image.float()
        valid = slab_valid.float()

        # Per-slab standardization over valid voxels; invalid voxels are
        # zeroed. The validity channel rides along so the model can tell
        # true darkness from unsampled space.
        mass = valid.sum(dim=(1, 2, 3)).clamp_min(1.0)
        mean = (image * valid).sum(dim=(1, 2, 3)) / mass
        centered = (image - mean[:, None, None, None]) * valid
        std = (centered.square().sum(dim=(1, 2, 3)) / mass).sqrt()
        image = centered / (std[:, None, None, None] + 1e-6)

        features = torch.stack([image, valid], dim=1)
        skips = []
        for stage in self.stages:
            features = stage(features)
            skips.append(features)

        features = self.trunk_in(features)
        for block in self.stem_blocks:
            features = block(features)

        tokens = features.permute(0, 2, 3, 4, 1)
        for block in self.attention_blocks:
            tokens = block(tokens)
        features = tokens.permute(0, 4, 1, 2, 3)

        features = F.interpolate(features, scale_factor=(2, 2, 1), mode="nearest")
        features = self.decoder(torch.cat([features, skips[-2]], dim=1))

        phase_increments = F.softplus(
            self.phase_head(features).squeeze(1).float()
        )
        phase = phase_increments.cumsum(dim=-1)
        return {
            "phase": phase,
            "phase_increments": phase_increments,
            "density_log_variance": self.log_variance_head(features).squeeze(1),
            "crossing_logits": self.crossing_head(features).squeeze(1),
        }
