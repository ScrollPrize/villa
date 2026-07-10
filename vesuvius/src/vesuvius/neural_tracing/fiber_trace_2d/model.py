from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class FiberStripDirectionModelConfig:
    in_channels: int = 1
    hidden_channels: int = 64
    depth: int = 10


def _group_norm_groups(channels: int, preferred_groups: int = 8) -> int:
    groups = min(int(preferred_groups), int(channels))
    while groups > 1 and int(channels) % groups != 0:
        groups -= 1
    return groups


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = _group_norm_groups(channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.act = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        residual = image
        out = self.conv1(image)
        out = self.norm1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.norm2(out)
        return self.act(out + residual)


class FiberStripDirectionNet(nn.Module):
    """V0 2D ResNet that predicts Lasagna two-channel direction codes."""

    def __init__(self, config: FiberStripDirectionModelConfig | None = None) -> None:
        super().__init__()
        cfg = FiberStripDirectionModelConfig() if config is None else config
        if cfg.in_channels <= 0:
            raise ValueError("in_channels must be > 0")
        if cfg.hidden_channels <= 0:
            raise ValueError("hidden_channels must be > 0")
        if cfg.depth <= 0:
            raise ValueError("depth must be > 0")
        hidden = int(cfg.hidden_channels)
        groups = _group_norm_groups(hidden)
        self.input = nn.Sequential(
            nn.Conv2d(int(cfg.in_channels), hidden, kernel_size=3, padding=1),
            nn.GroupNorm(groups, hidden),
            nn.SiLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[_ResidualBlock(hidden) for _ in range(int(cfg.depth))])
        self.output = nn.Conv2d(hidden, 2, kernel_size=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4:
            raise ValueError("image must have shape N,C,H,W")
        return torch.sigmoid(self.output(self.blocks(self.input(image))))
