from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class FiberStripDirectionModelConfig:
    in_channels: int = 1
    hidden_channels: int = 32
    depth: int = 5


class FiberStripDirectionNet(nn.Module):
    """Small V0 2D CNN that predicts Lasagna two-channel direction codes."""

    def __init__(self, config: FiberStripDirectionModelConfig | None = None) -> None:
        super().__init__()
        cfg = FiberStripDirectionModelConfig() if config is None else config
        if cfg.in_channels <= 0:
            raise ValueError("in_channels must be > 0")
        if cfg.hidden_channels <= 0:
            raise ValueError("hidden_channels must be > 0")
        if cfg.depth <= 0:
            raise ValueError("depth must be > 0")
        layers: list[nn.Module] = []
        channels = int(cfg.in_channels)
        for _ in range(int(cfg.depth)):
            layers.extend(
                [
                    nn.Conv2d(channels, int(cfg.hidden_channels), kernel_size=3, padding=1),
                    nn.GroupNorm(1, int(cfg.hidden_channels)),
                    nn.SiLU(inplace=True),
                ]
            )
            channels = int(cfg.hidden_channels)
        layers.append(nn.Conv2d(channels, 2, kernel_size=1))
        self.net = nn.Sequential(*layers)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4:
            raise ValueError("image must have shape N,C,H,W")
        return torch.sigmoid(self.net(image))
