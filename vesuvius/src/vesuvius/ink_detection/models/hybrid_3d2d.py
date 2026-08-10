"""Local depth-aware feature fusion for a two-dimensional segmentation body."""

from __future__ import annotations

import torch
from torch import nn


class LocalDepthFusionStem(nn.Module):
    """Map one BCZYX image channel to attention-weighted sum and max features."""

    def __init__(self, *, channels: int = 16) -> None:
        super().__init__()
        channels = int(channels)
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels!r}")
        self.channels = channels
        self.features = nn.Sequential(
            nn.Conv3d(1, channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(channels, affine=True),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(channels, affine=True),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True),
        )
        self.attention_logits = nn.Conv3d(channels, 1, kernel_size=1)
        nn.init.zeros_(self.attention_logits.weight)
        nn.init.zeros_(self.attention_logits.bias)

    def forward(self, image_BCZYX: torch.Tensor) -> torch.Tensor:
        features_BCZYX = self.features(image_BCZYX)
        weights_B1ZYX = torch.softmax(
            self.attention_logits(features_BCZYX).float(), dim=2
        )
        attention_pooled_BCYX = (
            features_BCZYX * weights_B1ZYX.to(features_BCZYX.dtype)
        ).sum(dim=2)
        max_pooled_BCYX = features_BCZYX.amax(dim=2)
        return torch.cat([attention_pooled_BCYX, max_pooled_BCYX], dim=1)


class Local3DStem2DUNet(nn.Module):
    """Fuse a fixed-depth BCZYX image before applying a 2D segmentation body."""

    def __init__(
        self,
        network: nn.Module,
        *,
        input_depth: int,
        stem_channels: int = 16,
    ) -> None:
        super().__init__()
        self.network = network
        self.input_depth = int(input_depth)
        if self.input_depth <= 0:
            raise ValueError(f"input_depth must be positive, got {input_depth!r}")
        self.depth_fusion = LocalDepthFusionStem(channels=stem_channels)

    @property
    def shared_encoder(self):
        return self.network.shared_encoder

    def forward(self, image_BCZYX: torch.Tensor):
        if image_BCZYX.ndim != 5:
            raise ValueError(
                "3D-stem/2D-UNet input must have shape "
                f"[batch, channel, z, y, x], got {tuple(image_BCZYX.shape)}"
            )
        if int(image_BCZYX.shape[1]) != 1:
            raise ValueError(
                "3D-stem/2D-UNet requires one source image channel, "
                f"got {int(image_BCZYX.shape[1])}"
            )
        if int(image_BCZYX.shape[2]) != self.input_depth:
            raise ValueError(
                f"3D-stem/2D-UNet input depth mismatch: expected "
                f"{self.input_depth}, got {int(image_BCZYX.shape[2])}"
            )
        return self.network(self.depth_fusion(image_BCZYX))
