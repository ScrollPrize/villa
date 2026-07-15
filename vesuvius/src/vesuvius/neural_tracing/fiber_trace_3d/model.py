from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from vesuvius.neural_tracing.fiber_trace.model import (
    _derive_features_per_stage,
    _derive_unet_strides,
)
from vesuvius.neural_tracing.nets.vesuvius_unet3d import Vesuvius3dUnetModel


@dataclass(frozen=True)
class FiberTrace3DModelConfig:
    input_channels: int = 1
    output_channels: int = 7
    features_per_stage: tuple[int, ...] = (16, 32, 64, 128)
    strides: tuple[tuple[int, int, int], ...] | None = None
    decoder_upsample_mode: str = "pixelshuffle"
    squeeze_excitation: bool = False


class FiberTrace3DNet(nn.Module):
    """3D U-Net with Lasagna 3x2 direction and fiber-presence outputs."""

    def __init__(self, config: FiberTrace3DModelConfig | None = None) -> None:
        super().__init__()
        cfg = FiberTrace3DModelConfig() if config is None else config
        if cfg.input_channels <= 0:
            raise ValueError("input_channels must be > 0")
        if cfg.output_channels < 7:
            raise ValueError("output_channels must contain 6 direction + 1 presence channels")
        if not cfg.features_per_stage:
            raise ValueError("features_per_stage must not be empty")
        strides = cfg.strides
        if strides is None:
            strides = ((1, 1, 1),) + ((2, 2, 2),) * (
                len(cfg.features_per_stage) - 1
            )
        backbone_config = {
            "features_per_stage": [int(v) for v in cfg.features_per_stage],
            "strides": [list(map(int, stride)) for stride in strides],
            "time_emb_dim": 0,
            "squeeze_excitation": bool(cfg.squeeze_excitation),
            "decoder_upsample_mode": str(cfg.decoder_upsample_mode),
            "keep_inactive_deep_supervision_layers": False,
            "normalization": "none",
        }
        self.net = Vesuvius3dUnetModel(
            int(cfg.input_channels),
            int(cfg.output_channels),
            {"model_config": backbone_config},
        )
        self.output_channels = int(cfg.output_channels)

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        if volume.ndim != 5:
            raise ValueError("volume must have shape B,C,D,H,W")
        raw = self.net(volume)
        return torch.sigmoid(raw)


def direction_output(output: torch.Tensor) -> torch.Tensor:
    if output.ndim != 5 or int(output.shape[1]) < 6:
        raise ValueError("model output must have shape B,C,D,H,W with >= 6 channels")
    return output[:, :6]


def presence_output(output: torch.Tensor) -> torch.Tensor:
    if output.ndim != 5 or int(output.shape[1]) < 7:
        raise ValueError("model output must have shape B,C,D,H,W with >= 7 channels")
    return output[:, 6:7]


def build_fiber_trace_3d_model(config: dict[str, Any]) -> FiberTrace3DNet:
    model_cfg = dict(config.get("model_3d", config.get("model", {})))
    features_per_stage = _derive_features_per_stage(model_cfg)
    strides = _derive_unet_strides(
        features_per_stage,
        model_cfg,
        crop_size=config.get("patch_shape_zyx", config.get("crop_size")),
    )
    return FiberTrace3DNet(
        FiberTrace3DModelConfig(
            input_channels=int(model_cfg.get("input_channels", 1)),
            output_channels=int(model_cfg.get("output_channels", 7)),
            features_per_stage=features_per_stage,
            strides=strides,
            decoder_upsample_mode=str(
                model_cfg.get("decoder_upsample_mode", "pixelshuffle")
            ),
            squeeze_excitation=bool(model_cfg.get("squeeze_excitation", False)),
        )
    )
