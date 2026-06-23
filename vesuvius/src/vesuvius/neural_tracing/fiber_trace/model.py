from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from vesuvius.neural_tracing.nets.vesuvius_unet3d import Vesuvius3dUnetModel


class DirectionConditionedFiberTraceModel(nn.Module):
    """3D U-Net feature backbone with a direction-conditioned dense head."""

    def __init__(
        self,
        *,
        input_channels: int = 1,
        conditioned_feature_channels: int = 64,
        backbone_channels: int | None = None,
        embedding_dim: int = 16,
        features_per_stage: tuple[int, ...] = (16, 32, 64, 128, 256, 512, 1024),
        strides: tuple[tuple[int, int, int], ...] | None = None,
        head_channels: int = 64,
        decoder_upsample_mode: str = "pixelshuffle",
    ) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        if backbone_channels is not None:
            conditioned_feature_channels = int(backbone_channels)
        self.conditioned_feature_channels = int(conditioned_feature_channels)
        self.backbone_channels = self.conditioned_feature_channels
        if strides is None:
            strides = ((1, 1, 1),) + ((2, 2, 2),) * max(
                0, len(features_per_stage) - 1
            )
        backbone_config = {
            "features_per_stage": [int(v) for v in features_per_stage],
            "strides": [list(map(int, stride)) for stride in strides],
            "time_emb_dim": 0,
            "squeeze_excitation": False,
            "decoder_upsample_mode": str(decoder_upsample_mode),
        }
        self.backbone = Vesuvius3dUnetModel(
            int(input_channels),
            self.conditioned_feature_channels,
            {"model_config": backbone_config},
        )
        self.head = nn.Sequential(
            nn.Conv3d(
                self.conditioned_feature_channels + 3,
                int(head_channels),
                kernel_size=1,
                bias=True,
            ),
            nn.SiLU(inplace=True),
            nn.Conv3d(
                int(head_channels), self.embedding_dim + 3, kernel_size=1, bias=True
            ),
        )

    def _apply_head_to_vectors(self, values: Tensor) -> Tensor:
        first = self.head[0]
        second = self.head[2]
        hidden = F.linear(values, first.weight.flatten(1), first.bias)
        hidden = F.silu(hidden, inplace=False)
        return F.linear(hidden, second.weight.flatten(1), second.bias)

    def forward(
        self,
        volume: Tensor,
        cond_fw_xyz: Tensor,
        sample_indices: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if volume.ndim != 5:
            raise ValueError(
                f"volume must have shape [B, C, D, H, W], got {tuple(volume.shape)}"
            )
        if cond_fw_xyz.shape != (volume.shape[0], 3):
            raise ValueError(
                f"cond_fw_xyz must have shape [B, 3], got {tuple(cond_fw_xyz.shape)}"
            )

        cond_fw_xyz = F.normalize(cond_fw_xyz.to(dtype=volume.dtype), dim=1)
        features = self.backbone(volume)
        if tuple(features.shape[2:]) != tuple(volume.shape[2:]):
            features = F.interpolate(
                features, size=volume.shape[2:], mode="trilinear", align_corners=False
            )

        if sample_indices is not None:
            if sample_indices.ndim != 1:
                raise ValueError(
                    "sample_indices must be a 1D flattened [B, D, H, W] index tensor"
                )
            sample_indices = sample_indices.to(device=features.device, dtype=torch.long)
            spatial_size = int(features.shape[2] * features.shape[3] * features.shape[4])
            flat_features = features.permute(0, 2, 3, 4, 1).reshape(
                -1, features.shape[1]
            )
            sampled_features = flat_features[sample_indices]
            batch_indices = torch.div(
                sample_indices, spatial_size, rounding_mode="floor"
            )
            sampled_cond = cond_fw_xyz[batch_indices]
            raw = self._apply_head_to_vectors(
                torch.cat([sampled_features, sampled_cond], dim=1)
            )
            embedding = F.normalize(raw[:, : self.embedding_dim], dim=1)
            fw = F.normalize(raw[:, self.embedding_dim : self.embedding_dim + 3], dim=1)
            return {
                "embedding": embedding,
                "fw": fw,
                "sample_indices": sample_indices,
            }

        cond = cond_fw_xyz.view(volume.shape[0], 3, 1, 1, 1)
        cond = cond.expand(-1, -1, *features.shape[2:])
        raw = self.head(torch.cat([features, cond], dim=1))
        embedding = F.normalize(raw[:, : self.embedding_dim], dim=1)
        fw = F.normalize(raw[:, self.embedding_dim : self.embedding_dim + 3], dim=1)
        return {"embedding": embedding, "fw": fw}


def _derive_features_per_stage(model_cfg: dict[str, Any]) -> tuple[int, ...]:
    if "features_per_stage" in model_cfg:
        features = tuple(int(v) for v in model_cfg["features_per_stage"])
        if not features:
            raise ValueError("model.features_per_stage must not be empty")
        if any(value <= 0 for value in features):
            raise ValueError("model.features_per_stage values must be positive")
        return features

    base = int(model_cfg.get("unet_base_channels", 16))
    depth = int(model_cfg.get("unet_depth", 3))
    if base <= 0:
        raise ValueError(f"model.unet_base_channels must be positive, got {base}")
    if depth <= 0:
        raise ValueError(f"model.unet_depth must be positive, got {depth}")
    return tuple(base * (2**stage) for stage in range(depth))


def _derive_unet_strides(
    features_per_stage: tuple[int, ...],
    model_cfg: dict[str, Any],
    crop_size: Any = None,
) -> tuple[tuple[int, int, int], ...]:
    if "strides" in model_cfg:
        strides = tuple(tuple(int(v) for v in stride) for stride in model_cfg["strides"])
        if len(strides) != len(features_per_stage):
            raise ValueError(
                "model.strides must have one stride per U-Net stage: "
                f"got {len(strides)} for {len(features_per_stage)} stages"
            )
        if any(len(stride) != 3 for stride in strides):
            raise ValueError("model.strides entries must be length-3")
        if any(any(value <= 0 for value in stride) for stride in strides):
            raise ValueError("model.strides values must be positive")
        return strides
    no_downsample_stages = 1
    if crop_size is not None:
        if isinstance(crop_size, int):
            min_crop = int(crop_size)
        elif isinstance(crop_size, (list, tuple)) and crop_size:
            min_crop = min(int(v) for v in crop_size)
        else:
            raise ValueError(f"crop_size must be an int or sequence, got {crop_size!r}")
        if min_crop <= 0:
            raise ValueError(f"crop_size values must be positive, got {crop_size!r}")
        max_downsamples = max(0, int(min_crop).bit_length() - 2)
        no_downsample_stages = max(
            1, len(features_per_stage) - int(max_downsamples)
        )
    no_downsample_stages = min(len(features_per_stage), no_downsample_stages)
    return ((1, 1, 1),) * no_downsample_stages + ((2, 2, 2),) * (
        len(features_per_stage) - no_downsample_stages
    )


def build_fiber_trace_model(
    config: dict[str, Any],
) -> DirectionConditionedFiberTraceModel:
    model_cfg = dict(config.get("model", config))
    features_per_stage = _derive_features_per_stage(model_cfg)
    strides = _derive_unet_strides(
        features_per_stage,
        model_cfg,
        crop_size=config.get("crop_size"),
    )
    conditioned_feature_channels = int(
        model_cfg.get(
            "conditioned_feature_channels",
            model_cfg.get("backbone_channels", 64),
        )
    )
    if conditioned_feature_channels <= 0:
        raise ValueError(
            "model.conditioned_feature_channels/backbone_channels must be positive"
        )
    return DirectionConditionedFiberTraceModel(
        input_channels=int(model_cfg.get("input_channels", 1)),
        conditioned_feature_channels=conditioned_feature_channels,
        embedding_dim=int(model_cfg.get("embedding_dim", 16)),
        features_per_stage=features_per_stage,
        strides=strides,
        head_channels=int(model_cfg.get("head_channels", 64)),
        decoder_upsample_mode=str(model_cfg.get("decoder_upsample_mode", "pixelshuffle")),
    )
