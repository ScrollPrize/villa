from __future__ import annotations

import torch
from torch import nn


class LocalDepthFusionStem(nn.Module):
    """Learn local 3D features, then fuse depth into stable 2D channels."""

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

    @property
    def output_channels(self) -> int:
        return 2 * self.channels

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.features(image)
        # Keep the depth softmax stable under autocast. It begins as a depth
        # mean, while the parallel max branch preserves strong local features.
        weights = torch.softmax(self.attention_logits(features).float(), dim=2)
        attention_pooled = (features * weights.to(features.dtype)).sum(dim=2)
        max_pooled = features.amax(dim=2)
        return torch.cat([attention_pooled, max_pooled], dim=1)


class Local3DStem2DUNet(nn.Module):
    """Thin 3D Z-locality front end followed by the standard 2D nnU-Net body."""

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

    def forward(self, image: torch.Tensor):
        if image.ndim != 5:
            raise ValueError(
                "3D-stem/2D-UNet input must have shape [batch, channel, z, y, x], "
                f"got {tuple(image.shape)}"
            )
        if int(image.shape[1]) != 1:
            raise ValueError(
                "3D-stem/2D-UNet requires one source image channel, "
                f"got {int(image.shape[1])}"
            )
        if int(image.shape[2]) != self.input_depth:
            raise ValueError(
                f"3D-stem/2D-UNet input depth mismatch: expected {self.input_depth}, "
                f"got {int(image.shape[2])}"
            )
        return self.network(self.depth_fusion(image))
