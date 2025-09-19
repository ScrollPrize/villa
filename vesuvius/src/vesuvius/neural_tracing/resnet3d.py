import torch
import torch.nn as nn
import torch.nn.functional as F


class SeparableConv3d(nn.Module):
    """
    Factorized 3D convolution: sequential convs along depth, height, width axes.
    Replaces a full 3x3x3 with 3x1x1 + 1x3x1 + 1x1x3.
    """
    def __init__(self, in_channels, out_channels, stride=1, bias=False):
        super().__init__()
        self.conv_d = nn.Conv3d(in_channels, out_channels, kernel_size=(3,1,1),
                                stride=(stride,1,1), padding=(1,0,0), bias=bias)
        self.conv_h = nn.Conv3d(out_channels, out_channels, kernel_size=(1,3,1),
                                stride=(1,stride,1), padding=(0,1,0), bias=bias)
        self.conv_w = nn.Conv3d(out_channels, out_channels, kernel_size=(1,1,3),
                                stride=(1,1,stride), padding=(0,0,1), bias=bias)

    def forward(self, x):
        x = self.conv_d(x)
        x = self.conv_h(x)
        x = self.conv_w(x)
        return x


class ResidualBlock3D(nn.Module):
    """
    Residual block with two 3D convolution layers.
    Uses GroupNorm + SiLU.
    """
    def __init__(self, in_channels, out_channels, stride=1, groups=8):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(groups, out_channels)
        self.act1 = nn.SiLU(inplace=True)

        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(groups, out_channels)

        # Skip connection (identity or 1x1 conv if shape mismatch)
        if in_channels != out_channels or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.GroupNorm(groups, out_channels),
            )
        else:
            self.downsample = None

        self.act2 = nn.SiLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.act2(out)
        return out


class ResNet3DEncoder(nn.Module):
    """
    3D ResNet encoder
    - channels: list of channel sizes per stage
    - blocks: list of block counts per stage
    - groups: number of groups for GroupNorm
    """
    def __init__(self, in_channels=1, channels=[32, 64, 96, 128],
                 blocks=[2, 2, 2, 2], groups=8):
        super().__init__()
        assert len(channels) == len(blocks), "channels and blocks must match in length"

        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, channels[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(groups, channels[0]),
            nn.SiLU(inplace=True),
        )

        stages = []
        in_ch = channels[0]
        for stage_idx, (ch, num_blocks) in enumerate(zip(channels, blocks)):
            stage_blocks = []
            for block_idx in range(num_blocks):
                stride = 2 if (block_idx == 0 and stage_idx > 0) else 1
                stage_blocks.append(ResidualBlock3D(in_ch, ch, stride=stride, groups=groups))
                in_ch = ch
            stages.append(nn.Sequential(*stage_blocks))
        self.stages = nn.ModuleList(stages)

    def forward(self, x):
        x = self.stem(x)
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return x

