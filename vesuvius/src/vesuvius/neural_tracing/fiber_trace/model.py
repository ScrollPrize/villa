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
        backbone_channels: int = 16,
        embedding_dim: int = 16,
        features_per_stage: tuple[int, ...] = (16, 32),
        head_channels: int = 32,
    ) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.backbone_channels = int(backbone_channels)
        self.backbone = Vesuvius3dUnetModel(
            int(input_channels),
            self.backbone_channels,
            {
                "model_config": {
                    "features_per_stage": [int(v) for v in features_per_stage],
                    "time_emb_dim": 0,
                    "squeeze_excitation": False,
                }
            },
        )
        self.head = nn.Sequential(
            nn.Conv3d(
                self.backbone_channels + 6, int(head_channels), kernel_size=1, bias=True
            ),
            nn.SiLU(inplace=True),
            nn.Conv3d(
                int(head_channels), self.embedding_dim + 6, kernel_size=1, bias=True
            ),
        )

    def forward(
        self, volume: Tensor, cond_fw_xyz: Tensor, cond_up_xyz: Tensor
    ) -> dict[str, Tensor]:
        if volume.ndim != 5:
            raise ValueError(
                f"volume must have shape [B, C, D, H, W], got {tuple(volume.shape)}"
            )
        if cond_fw_xyz.shape != (volume.shape[0], 3):
            raise ValueError(
                f"cond_fw_xyz must have shape [B, 3], got {tuple(cond_fw_xyz.shape)}"
            )
        if cond_up_xyz.shape != (volume.shape[0], 3):
            raise ValueError(
                f"cond_up_xyz must have shape [B, 3], got {tuple(cond_up_xyz.shape)}"
            )

        cond_fw_xyz = F.normalize(cond_fw_xyz.to(dtype=volume.dtype), dim=1)
        cond_up_xyz = F.normalize(cond_up_xyz.to(dtype=volume.dtype), dim=1)
        features = self.backbone(volume)
        if tuple(features.shape[2:]) != tuple(volume.shape[2:]):
            features = F.interpolate(
                features, size=volume.shape[2:], mode="trilinear", align_corners=False
            )

        cond = torch.cat([cond_fw_xyz, cond_up_xyz], dim=1).view(
            volume.shape[0], 6, 1, 1, 1
        )
        cond = cond.expand(-1, -1, *features.shape[2:])
        raw = self.head(torch.cat([features, cond], dim=1))
        embedding = F.normalize(raw[:, : self.embedding_dim], dim=1)
        fw = F.normalize(raw[:, self.embedding_dim : self.embedding_dim + 3], dim=1)
        up = F.normalize(raw[:, self.embedding_dim + 3 : self.embedding_dim + 6], dim=1)
        return {"embedding": embedding, "fw": fw, "up": up}


def build_fiber_trace_model(
    config: dict[str, Any],
) -> DirectionConditionedFiberTraceModel:
    model_cfg = dict(config.get("model", config))
    return DirectionConditionedFiberTraceModel(
        input_channels=int(model_cfg.get("input_channels", 1)),
        backbone_channels=int(model_cfg.get("backbone_channels", 16)),
        embedding_dim=int(model_cfg.get("embedding_dim", 16)),
        features_per_stage=tuple(
            int(v) for v in model_cfg.get("features_per_stage", (16, 32))
        ),
        head_channels=int(model_cfg.get("head_channels", 32)),
    )
