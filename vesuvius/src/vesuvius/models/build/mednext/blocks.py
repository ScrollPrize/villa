"""
3D MedNeXt blocks adapted from the official MIC-DKFZ MedNeXt repository.

This file intentionally keeps the stage-1 implementation close to upstream
MedNeXt v1 semantics while restricting support to 3D.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm3D(nn.Module):
    """channels_first LayerNorm variant used by upstream MedNeXt blocks."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"LayerNorm3D expects [B, C, D, H, W], got {tuple(x.shape)}")
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None, None] * x + self.bias[:, None, None, None]


def _make_norm(norm_type: str, num_channels: int) -> nn.Module:
    norm_type = str(norm_type).strip().lower()
    if norm_type == "group":
        return nn.GroupNorm(num_groups=num_channels, num_channels=num_channels)
    if norm_type == "layer":
        return LayerNorm3D(num_channels)
    if norm_type == "instance":
        return nn.InstanceNorm3d(num_channels, affine=True)
    raise ValueError(f"Unsupported MedNeXt norm_type {norm_type!r}")


class MedNeXtBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        exp_r: int = 4,
        kernel_size: int = 7,
        do_res: bool = True,
        norm_type: str = "group",
        n_groups: int | None = None,
        grn: bool = False,
    ):
        super().__init__()
        self.do_res = bool(do_res)
        self.grn = bool(grn)

        self.conv1 = nn.Conv3d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=in_channels if n_groups is None else int(n_groups),
        )
        self.norm = _make_norm(norm_type, in_channels)
        self.conv2 = nn.Conv3d(
            in_channels=in_channels,
            out_channels=exp_r * in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.act = nn.GELU()
        self.conv3 = nn.Conv3d(
            in_channels=exp_r * in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        if self.grn:
            expanded = exp_r * in_channels
            self.grn_beta = nn.Parameter(torch.zeros(1, expanded, 1, 1, 1), requires_grad=True)
            self.grn_gamma = nn.Parameter(torch.zeros(1, expanded, 1, 1, 1), requires_grad=True)

    def forward(self, x: torch.Tensor, dummy_tensor: torch.Tensor | None = None) -> torch.Tensor:
        del dummy_tensor
        residual = x
        x = self.conv1(x)
        x = self.act(self.conv2(self.norm(x)))
        if self.grn:
            gx = torch.norm(x, p=2, dim=(-3, -2, -1), keepdim=True)
            nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
            x = self.grn_gamma * (x * nx) + self.grn_beta + x
        x = self.conv3(x)
        if self.do_res:
            x = residual + x
        return x


class MedNeXtDownBlock3D(MedNeXtBlock3D):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        exp_r: int = 4,
        kernel_size: int = 7,
        do_res: bool = False,
        norm_type: str = "group",
        grn: bool = False,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            exp_r=exp_r,
            kernel_size=kernel_size,
            do_res=False,
            norm_type=norm_type,
            grn=grn,
        )
        self.resample_do_res = bool(do_res)
        if self.resample_do_res:
            self.res_conv = nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=2,
            )
        self.conv1 = nn.Conv3d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size // 2,
            groups=in_channels,
        )

    def forward(self, x: torch.Tensor, dummy_tensor: torch.Tensor | None = None) -> torch.Tensor:
        y = super().forward(x, dummy_tensor=dummy_tensor)
        if self.resample_do_res:
            y = y + self.res_conv(x)
        return y


class MedNeXtUpBlock3D(MedNeXtBlock3D):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        exp_r: int = 4,
        kernel_size: int = 7,
        do_res: bool = False,
        norm_type: str = "group",
        grn: bool = False,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            exp_r=exp_r,
            kernel_size=kernel_size,
            do_res=False,
            norm_type=norm_type,
            grn=grn,
        )
        self.resample_do_res = bool(do_res)
        if self.resample_do_res:
            self.res_conv = nn.ConvTranspose3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=2,
            )
        self.conv1 = nn.ConvTranspose3d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size // 2,
            groups=in_channels,
        )

    def forward(self, x: torch.Tensor, dummy_tensor: torch.Tensor | None = None) -> torch.Tensor:
        y = super().forward(x, dummy_tensor=dummy_tensor)
        y = F.pad(y, (1, 0, 1, 0, 1, 0))
        if self.resample_do_res:
            res = F.pad(self.res_conv(x), (1, 0, 1, 0, 1, 0))
            y = y + res
        return y


class OutBlock3D(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.conv_out = nn.ConvTranspose3d(in_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor, dummy_tensor: torch.Tensor | None = None) -> torch.Tensor:
        del dummy_tensor
        return self.conv_out(x)
