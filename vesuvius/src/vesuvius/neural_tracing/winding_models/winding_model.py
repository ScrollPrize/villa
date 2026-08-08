"""Winding phase model over intersecting transverse planes.

A shared-weight 2-D conv encoder collapses each plane's height while keeping
the ray axis at full resolution, the planes are fused symmetrically (their
order and orientation around the ray is arbitrary), and a 1-D transformer
with relative position bias runs along the ray. Per ray sample the model
predicts a monotone relative winding phase (non-negative increments
accumulated along the ray; the free offset is absorbed by the shift-invariant
loss and the consumer's per-ray registration) and a crossing logit whose
sigmoid peaks where the phase passes an integer.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _norm(width: int) -> nn.GroupNorm:
    return nn.GroupNorm(min(8, width), width)


class PlaneEncoder(nn.Module):
    """2-D encoder that halves plane height per stage at fixed ray stride."""

    def __init__(self, channels: list[int]):
        super().__init__()
        stages = []
        previous = 2  # intensity + validity
        for width in channels:
            stages.append(
                nn.Sequential(
                    nn.Conv2d(previous, width, 3, stride=(2, 1), padding=1),
                    _norm(width),
                    nn.GELU(),
                    nn.Conv2d(width, width, 3, padding=1),
                    _norm(width),
                    nn.GELU(),
                )
            )
            previous = width
        self.stages = nn.Sequential(*stages)

    def forward(self, planes: torch.Tensor) -> torch.Tensor:
        """[N, 2, H, W] -> [N, 3C, W] via center-row, mean, and max pooling.

        The ray lies on the planes' center row; pooled rows supply context
        from the surrounding sheets.
        """
        features = self.stages(planes)
        center = features[:, :, features.shape[2] // 2]
        return torch.cat([center, features.mean(dim=2), features.amax(dim=2)], dim=1)


class RayAttention(nn.Module):
    """Multi-head self-attention along the ray with relative position bias."""

    def __init__(self, dim: int, num_heads: int, max_distance: int):
        super().__init__()
        if dim % num_heads:
            raise ValueError("transformer dim must be divisible by num_heads")
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


class RayTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_distance: int):
        super().__init__()
        self.attention_norm = nn.LayerNorm(dim)
        self.attention = RayAttention(dim, num_heads, max_distance)
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim)
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = tokens + self.attention(self.attention_norm(tokens))
        return tokens + self.mlp(self.mlp_norm(tokens))


class WindingModel(nn.Module):
    def __init__(self, cfg: dict | None = None):
        super().__init__()
        cfg = dict(cfg or {})
        channels = [int(width) for width in cfg.get("encoder_channels", (32, 64, 96, 128))]
        dim = int(cfg.get("transformer_dim", 192))
        self.monotone_phase = bool(cfg.get("monotone_phase", True))
        self.encoder = PlaneEncoder(channels)
        self.fuse = nn.Linear(3 * channels[-1], dim)
        self.blocks = nn.ModuleList(
            RayTransformerBlock(
                dim,
                int(cfg.get("transformer_heads", 6)),
                int(cfg.get("max_relative_distance", 128)),
            )
            for _ in range(int(cfg.get("transformer_layers", 4)))
        )
        self.output_norm = nn.LayerNorm(dim)
        self.phase_head = nn.Linear(dim, 1)
        self.crossing_head = nn.Linear(dim, 1)

    def forward(
        self, plane_images: torch.Tensor, plane_valid: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """plane_images [B, P, H, W] raw intensities, plane_valid same-shape bool."""
        batch, num_planes, height, width = plane_images.shape
        images = plane_images.float().reshape(-1, 1, height, width)
        valid = plane_valid.reshape(-1, 1, height, width).float()

        # Per-plane standardization over valid pixels; invalid pixels are
        # zeroed and flagged through the validity channel instead.
        mass = valid.sum(dim=(2, 3)).clamp_min(1.0)
        mean = (images * valid).sum(dim=(2, 3)) / mass
        centered = (images - mean[:, :, None, None]) * valid
        std = (centered.square().sum(dim=(2, 3)) / mass).sqrt()
        images = centered / (std[:, :, None, None] + 1e-6)

        features = self.encoder(torch.cat([images, valid], dim=1))
        features = features.reshape(batch, num_planes, -1, width).mean(dim=1)
        tokens = self.fuse(features.transpose(1, 2))
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.output_norm(tokens)

        phase = self.phase_head(tokens).squeeze(-1)
        if self.monotone_phase:
            phase = F.softplus(phase.float()).cumsum(dim=-1)
        return {
            "phase": phase,
            "crossing_logits": self.crossing_head(tokens).squeeze(-1),
        }
