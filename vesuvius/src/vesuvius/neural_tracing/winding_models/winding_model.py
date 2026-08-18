"""Winding phase model over a ray-aligned 3-D slab.

The input is a single [H, W, L] slab sampled around a ray: transverse axes
H and W across the ray, the ray axis L at full volume resolution. A 3-D
conv encoder halves only the transverse axes, so every downstream feature
keeps per-sample localization along the ray. A short stem of plain
residual blocks mixes features locally, and a stack of axial attention
blocks does the long-range work: one attention over the ray axis per
column (relative position bias, as a sequence of length L) and one over
the transverse plane per ray position
(2-D relative bias). A light decoder restores ``decoder_levels`` transverse
levels with skip connections (each level halves the column stride at 4x
the decoder cost; the trunk is untouched), optionally followed by a
sub-pixel head (``head_upsample``): each head predicts s^2 values per
column, pixel-shuffled onto an s-times-finer transverse grid — learned
interpolation from the coarse features, cheaper but weaker than a real
decoder level. ``full_resolution_head`` provides a cheaper native-resolution
alternative: it upsamples the stride-4 decoder once, concatenates the
stride-2 encoder skip, and uses two 1x1 projections plus a 2x sub-pixel
shuffle. Every prediction head runs per column — a
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
- optionally (``use_variance_head``, kept for older checkpoints) a
  per-sample log-variance for those increments (heteroscedastic head),
  so the consumer can precision-weight registered observations instead
  of trusting every column equally.
- optionally (``use_crossing_head``, kept for older checkpoints) a
  crossing logit whose sigmoid peaks where the phase passes an integer.
  Without the head, crossings decode as the integer passages of the
  monotone phase itself.
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
        "decoder_levels",
        "head_upsample",
        "full_resolution_head",
        "use_crossing_head",
        "use_variance_head",
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

        decoder_levels = int(cfg.get("decoder_levels", 1))
        if not 1 <= decoder_levels <= len(channels) - 1:
            raise ValueError(
                "decoder_levels must be between 1 and one less than the "
                "number of encoder stages"
            )
        self.decoder = nn.Sequential(
            nn.Conv3d(trunk_dim + channels[-2], decoder_dim, 3, padding=1),
            _norm(decoder_dim),
            nn.GELU(),
            nn.Conv3d(decoder_dim, decoder_dim, 3, padding=1),
            _norm(decoder_dim),
            nn.GELU(),
        )
        # Additional transverse restoration levels consuming the earlier
        # encoder skips; kept out of ``decoder`` so level-1 checkpoints warm
        # start the shared level unchanged.
        self.extra_decoders = nn.ModuleList(
            nn.Sequential(
                nn.Conv3d(
                    decoder_dim + channels[-2 - level], decoder_dim, 3,
                    padding=1,
                ),
                _norm(decoder_dim),
                nn.GELU(),
                nn.Conv3d(decoder_dim, decoder_dim, 3, padding=1),
                _norm(decoder_dim),
                nn.GELU(),
            )
            for level in range(1, decoder_levels)
        )

        # Sub-pixel (pixel shuffle) head upsampling: each head predicts
        # head_upsample^2 values per column, rearranged onto a finer
        # transverse grid. Cheap learned interpolation from the coarse
        # features, versus a real decoder level's skip-fed resolution.
        head_upsample = int(cfg.get("head_upsample", 1))
        base_stride = 2 ** (len(channels) - decoder_levels)
        if head_upsample < 1 or head_upsample & (head_upsample - 1):
            raise ValueError("head_upsample must be a positive power of two")
        if head_upsample > base_stride:
            raise ValueError(
                "head_upsample cannot push the column stride below one voxel"
            )
        self.head_upsample = head_upsample
        head_channels = head_upsample**2

        # Native output without a wide stride-2 decoder. The stride-4 decoder
        # is interpolated to the first encoder skip, fused through a narrow
        # 1x1 projection, and four logits are shuffled to the voxel grid.
        self.full_resolution_head_enabled = bool(
            cfg.get("full_resolution_head", False)
        )
        if self.full_resolution_head_enabled:
            if decoder_levels != 1 or base_stride != 4 or len(channels) < 3:
                raise ValueError(
                    "full_resolution_head requires three encoder stages and "
                    "decoder_levels=1"
                )
            if head_upsample != 1:
                raise ValueError(
                    "full_resolution_head cannot be combined with head_upsample"
                )
            if bool(cfg.get("use_crossing_head", True)) or bool(
                cfg.get("use_variance_head", True)
            ):
                raise ValueError(
                    "full_resolution_head currently requires the crossing "
                    "and variance heads to be disabled"
                )
            detail_dim = channels[0]
            self.full_resolution_head = nn.Sequential(
                nn.Conv3d(decoder_dim + channels[0], detail_dim, 1),
                _norm(detail_dim),
                nn.GELU(),
                nn.Conv3d(detail_dim, 4, 1),
            )
            self.phase_head = None
        else:
            self.full_resolution_head = None
            self.phase_head = nn.Conv3d(decoder_dim, head_channels, 1)
        # The heteroscedastic variance head is likewise optional (the
        # default keeps older checkpoints loading); without it the density
        # loss falls back to a fixed-scale Huber and the consumer treats
        # increments as homoscedastic.
        if bool(cfg.get("use_variance_head", True)):
            self.log_variance_head = nn.Conv3d(decoder_dim, head_channels, 1)
        else:
            self.log_variance_head = None
        # The crossing head is retained for older checkpoints (the default
        # keeps their saved configs loading); without it, crossings decode
        # as the integer passages of the monotone phase, which are
        # duplicate-free by construction.
        if bool(cfg.get("use_crossing_head", True)):
            crossing_kernel = int(cfg.get("crossing_head_kernel_size", 5))
            if crossing_kernel <= 0 or crossing_kernel % 2 == 0:
                raise ValueError(
                    "crossing_head_kernel_size must be a positive odd integer"
                )
            self.crossing_head = nn.Conv3d(
                decoder_dim,
                head_channels,
                kernel_size=(1, 1, crossing_kernel),
                padding=(0, 0, crossing_kernel // 2),
            )

            crossing_prior = cfg.get("crossing_prior_prob")
            if crossing_prior is not None:
                crossing_prior = float(crossing_prior)
                if not 0.0 < crossing_prior < 1.0:
                    raise ValueError(
                        "crossing_prior_prob must be between zero and one"
                    )
                nn.init.constant_(
                    self.crossing_head.bias,
                    math.log(crossing_prior / (1.0 - crossing_prior)),
                )
        else:
            self.crossing_head = None

        phase_increment = cfg.get("phase_initial_increment")
        if phase_increment is not None:
            phase_increment = float(phase_increment)
            if phase_increment <= 0.0:
                raise ValueError("phase_initial_increment must be positive")
            phase_output = (
                self.full_resolution_head[-1]
                if self.full_resolution_head is not None
                else self.phase_head
            )
            nn.init.zeros_(phase_output.weight)
            nn.init.constant_(
                phase_output.bias,
                math.log(math.expm1(phase_increment)),
            )

        # Start with a moderate predicted variance: confident enough that the
        # density NLL has teeth from early on, loose enough that early large
        # residuals don't dominate the total loss.
        if self.log_variance_head is not None:
            nn.init.zeros_(self.log_variance_head.weight)
            nn.init.constant_(
                self.log_variance_head.bias,
                float(cfg.get("density_log_variance_init", -4.0)),
            )

    @property
    def column_stride(self) -> int:
        """Input voxels per output column on each transverse axis."""
        if self.full_resolution_head_enabled:
            return 1
        decoder_levels = 1 + len(self.extra_decoders)
        return 2 ** (len(self.stages) - decoder_levels) // self.head_upsample

    @staticmethod
    def _pixel_shuffle_output(out: torch.Tensor, scale: int) -> torch.Tensor:
        """Pixel-shuffle head channels onto the transverse spatial axes."""
        if scale == 1:
            return out.squeeze(1)
        batch, _, height, width, length = out.shape
        out = out.reshape(batch, scale, scale, height, width, length)
        out = out.permute(0, 3, 1, 4, 2, 5)
        return out.reshape(batch, height * scale, width * scale, length)

    def _head_output(self, head: nn.Module, features: torch.Tensor) -> torch.Tensor:
        """Apply a prediction head and pixel-shuffle it transversely.

        With ``head_upsample`` s, the head's s^2 channels rearrange onto an
        s-times-finer transverse grid: channel s_i * s + s_j lands at
        transverse offset (s_i, s_j) within each column's footprint.
        """
        return self._pixel_shuffle_output(head(features), self.head_upsample)

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

        # Trilinear rather than nearest upsampling: nearest duplicates one
        # trunk vector across a token's whole transverse footprint, so any
        # disagreement between adjacent tokens surfaces as a phase shelf
        # exactly on the token boundary (the two 3x3 decoder convs cannot
        # hide the seam). The ray axis keeps scale 1, so only the transverse
        # axes interpolate.
        features = F.interpolate(
            features, scale_factor=(2, 2, 1), mode="trilinear", align_corners=False
        )
        features = self.decoder(torch.cat([features, skips[-2]], dim=1))
        for level, decoder in enumerate(self.extra_decoders):
            features = F.interpolate(
                features, scale_factor=(2, 2, 1), mode="trilinear",
                align_corners=False,
            )
            features = decoder(
                torch.cat([features, skips[-3 - level]], dim=1)
            )

        if self.full_resolution_head is None:
            phase_logits = self._head_output(self.phase_head, features)
        else:
            fine_features = F.interpolate(
                features,
                size=skips[0].shape[2:],
                mode="trilinear",
                align_corners=False,
            )
            phase_logits = self._pixel_shuffle_output(
                self.full_resolution_head(
                    torch.cat([fine_features, skips[0]], dim=1)
                ),
                2,
            )

        phase_increments = F.softplus(phase_logits.float())
        phase = phase_increments.cumsum(dim=-1)
        output = {
            "phase": phase,
            "phase_increments": phase_increments,
        }
        if self.log_variance_head is not None:
            output["density_log_variance"] = self._head_output(
                self.log_variance_head, features
            )
        if self.crossing_head is not None:
            output["crossing_logits"] = self._head_output(
                self.crossing_head, features
            )
        return output
