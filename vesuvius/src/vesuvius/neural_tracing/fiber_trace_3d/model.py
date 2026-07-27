from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from vesuvius.neural_tracing.fiber_trace_3d.direction import (
    decode_lasagna_direction_3x2_analytic,
    encode_lasagna_direction_3x2,
)
from vesuvius.neural_tracing.fiber_trace.model import (
    _derive_features_per_stage,
    _derive_unet_strides,
)
from vesuvius.neural_tracing.nets.vesuvius_unet3d import Vesuvius3dUnetModel


@dataclass(frozen=True)
class FiberTrace3DModelConfig:
    input_channels: int = 1
    output_channels: int = 7
    direction_branch_count: int = 1
    conditioned_decoder_enabled: bool = False
    conditioned_latent_channels: int = 64
    conditioned_decoder_hidden_channels: int = 64
    conditioned_decoder_layers: int = 3
    features_per_stage: tuple[int, ...] = (16, 32, 64, 128)
    strides: tuple[tuple[int, int, int], ...] | None = None
    decoder_upsample_mode: str = "pixelshuffle"
    squeeze_excitation: bool = False
    normalization: str = "batch"


class FiberTrace3DNet(nn.Module):
    """3D U-Net with Lasagna 3x2 direction and fiber-presence outputs."""

    def __init__(self, config: FiberTrace3DModelConfig | None = None) -> None:
        super().__init__()
        cfg = FiberTrace3DModelConfig() if config is None else config
        if cfg.input_channels <= 0:
            raise ValueError("input_channels must be > 0")
        if not cfg.features_per_stage:
            raise ValueError("features_per_stage must not be empty")
        self.conditioned_decoder_enabled = bool(cfg.conditioned_decoder_enabled)
        if self.conditioned_decoder_enabled:
            if int(cfg.output_channels) != 7:
                raise ValueError(
                    "conditioned decoder mode emits exactly 7 output channels per query"
                )
            if int(cfg.direction_branch_count) != 1:
                raise ValueError(
                    "conditioned decoder mode does not use free direction branches; "
                    "set direction_branch_count to 1 or omit it"
                )
            if int(cfg.conditioned_latent_channels) <= 0:
                raise ValueError("conditioned_latent_channels must be > 0")
            if int(cfg.conditioned_decoder_hidden_channels) <= 0:
                raise ValueError("conditioned_decoder_hidden_channels must be > 0")
            if int(cfg.conditioned_decoder_layers) <= 0:
                raise ValueError("conditioned_decoder_layers must be > 0")
        else:
            if cfg.direction_branch_count <= 0:
                raise ValueError("direction_branch_count must be > 0")
            if cfg.output_channels != int(cfg.direction_branch_count) * 7:
                raise ValueError(
                    "output_channels must equal direction_branch_count * "
                    "(6 direction + 1 presence channels)"
                )
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
            "normalization": str(cfg.normalization),
        }
        backbone_out_channels = (
            int(cfg.conditioned_latent_channels)
            if self.conditioned_decoder_enabled
            else int(cfg.output_channels)
        )
        self.net = Vesuvius3dUnetModel(
            int(cfg.input_channels),
            backbone_out_channels,
            {"model_config": backbone_config},
        )
        self.output_channels = int(cfg.output_channels)
        self.direction_branch_count = int(cfg.direction_branch_count)
        self.conditioned_latent_channels = int(cfg.conditioned_latent_channels)
        self.conditioned_decoder_hidden_channels = int(
            cfg.conditioned_decoder_hidden_channels
        )
        self.conditioned_decoder_layers = int(cfg.conditioned_decoder_layers)
        self.conditioned_decoder: nn.Sequential | None = None
        if self.conditioned_decoder_enabled:
            self.conditioned_decoder = self._make_conditioned_decoder(cfg)

    @staticmethod
    def _make_conditioned_decoder(cfg: FiberTrace3DModelConfig) -> nn.Sequential:
        layers: list[nn.Module] = []
        in_channels = int(cfg.conditioned_latent_channels) + 6
        hidden_channels = int(cfg.conditioned_decoder_hidden_channels)
        layer_count = int(cfg.conditioned_decoder_layers)
        for _index in range(max(0, layer_count - 1)):
            layers.append(nn.Conv3d(in_channels, hidden_channels, kernel_size=1))
            layers.append(nn.ReLU(inplace=True))
            in_channels = hidden_channels
        layers.append(nn.Conv3d(in_channels, 7, kernel_size=1))
        return nn.Sequential(*layers)

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        if volume.ndim != 5:
            raise ValueError("volume must have shape B,C,D,H,W")
        if self.conditioned_decoder_enabled:
            latent = self.encode_volume(volume)
            query = torch.zeros(
                (
                    int(latent.shape[0]),
                    6,
                    int(latent.shape[2]),
                    int(latent.shape[3]),
                    int(latent.shape[4]),
                ),
                dtype=latent.dtype,
                device=latent.device,
            )
            return self.decode_conditioned_latent(latent, query)
        raw = self.net(volume)
        return torch.sigmoid(raw)

    def encode_volume(self, volume: torch.Tensor) -> torch.Tensor:
        if not self.conditioned_decoder_enabled:
            raise RuntimeError("encode_volume is only available in conditioned decoder mode")
        if volume.ndim != 5:
            raise ValueError("volume must have shape B,C,D,H,W")
        return self.net(volume)

    def decode_conditioned_latent(
        self,
        latent: torch.Tensor,
        query: torch.Tensor,
    ) -> torch.Tensor:
        if not self.conditioned_decoder_enabled or self.conditioned_decoder is None:
            raise RuntimeError(
                "decode_conditioned_latent is only available in conditioned decoder mode"
            )
        if latent.ndim != 5:
            raise ValueError("latent must have shape B,C,D,H,W")
        query_volume = self._query_to_volume(query, latent)
        raw = self.conditioned_decoder(torch.cat([latent, query_volume], dim=1))
        return torch.sigmoid(raw)

    def decode_conditioned_points(
        self,
        latent: torch.Tensor,
        indices_bzyx: torch.Tensor,
        query_n6: torch.Tensor,
    ) -> torch.Tensor:
        if not self.conditioned_decoder_enabled or self.conditioned_decoder is None:
            raise RuntimeError(
                "decode_conditioned_points is only available in conditioned decoder mode"
            )
        if latent.ndim != 5:
            raise ValueError("latent must have shape B,C,D,H,W")
        if indices_bzyx.ndim != 2 or int(indices_bzyx.shape[1]) != 4:
            raise ValueError("indices_bzyx must have shape N,4")
        if query_n6.ndim != 2 or int(query_n6.shape[1]) != 6:
            raise ValueError("query_n6 must have shape N,6")
        indices = indices_bzyx.to(dtype=torch.long, device=latent.device)
        query = query_n6.to(dtype=latent.dtype, device=latent.device)
        features = latent[
            indices[:, 0],
            :,
            indices[:, 1],
            indices[:, 2],
            indices[:, 3],
        ]
        raw = self.conditioned_decoder(
            torch.cat([features, query], dim=1).view(int(indices.shape[0]), -1, 1, 1, 1)
        )
        return torch.sigmoid(raw[:, :, 0, 0, 0])

    def forward_conditioned(self, volume: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        latent = self.encode_volume(volume)
        return self.decode_conditioned_latent(latent, query)

    def forward_recurrent(
        self,
        volume: torch.Tensor,
        *,
        steps: int = 2,
        detach_query: bool = True,
    ) -> torch.Tensor:
        if not self.conditioned_decoder_enabled:
            if int(steps) != 1:
                raise RuntimeError("recurrent forward requires conditioned decoder mode")
            return self.forward(volume).unsqueeze(1)
        step_count = int(steps)
        if step_count <= 0:
            raise ValueError("steps must be > 0")
        latent = self.encode_volume(volume)
        query = torch.zeros(
            (
                int(latent.shape[0]),
                6,
                int(latent.shape[2]),
                int(latent.shape[3]),
                int(latent.shape[4]),
            ),
            dtype=latent.dtype,
            device=latent.device,
        )
        outputs: list[torch.Tensor] = []
        for _step in range(step_count):
            output = self.decode_conditioned_latent(latent, query)
            outputs.append(output)
            dirs_bdhw6 = output[:, :6].permute(0, 2, 3, 4, 1).contiguous()
            axis_xyz = decode_lasagna_direction_3x2_analytic(dirs_bdhw6)
            query_bdhw6 = encode_lasagna_direction_3x2(axis_xyz)
            if detach_query:
                query_bdhw6 = query_bdhw6.detach()
            query = query_bdhw6.permute(0, 4, 1, 2, 3).contiguous()
        return torch.stack(outputs, dim=1)

    def forward_recurrent_grouped(
        self,
        volume: torch.Tensor,
        *,
        steps: int = 2,
        detach_query: bool = True,
    ) -> torch.Tensor:
        outputs = self.forward_recurrent(
            volume,
            steps=int(steps),
            detach_query=bool(detach_query),
        )
        batch, step_count, channels, depth, height, width = (int(v) for v in outputs.shape)
        return outputs.reshape(batch, step_count * channels, depth, height, width)

    @staticmethod
    def _query_to_volume(query: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        if query.ndim == 2:
            if int(query.shape[0]) != int(latent.shape[0]) or int(query.shape[1]) != 6:
                raise ValueError("query must have shape B,6 when passed as a vector")
            return query.to(dtype=latent.dtype, device=latent.device).view(
                int(latent.shape[0]),
                6,
                1,
                1,
                1,
            ).expand(-1, -1, int(latent.shape[2]), int(latent.shape[3]), int(latent.shape[4]))
        if query.ndim != 5:
            raise ValueError("query must have shape B,6 or B,6,D,H,W")
        if (
            int(query.shape[0]) != int(latent.shape[0])
            or int(query.shape[1]) != 6
            or tuple(int(v) for v in query.shape[2:]) != tuple(int(v) for v in latent.shape[2:])
        ):
            raise ValueError("query volume must have shape B,6,D,H,W matching latent")
        return query.to(dtype=latent.dtype, device=latent.device)


def direction_output(output: torch.Tensor) -> torch.Tensor:
    if output.ndim != 5 or int(output.shape[1]) < 6:
        raise ValueError("model output must have shape B,C,D,H,W with >= 6 channels")
    return output[:, :6]


def presence_output(output: torch.Tensor) -> torch.Tensor:
    if output.ndim != 5 or int(output.shape[1]) < 7:
        raise ValueError("model output must have shape B,C,D,H,W with >= 7 channels")
    return output[:, 6:7]


def direction_outputs(output: torch.Tensor) -> torch.Tensor:
    if output.ndim != 5:
        raise ValueError("model output must have shape B,C,D,H,W")
    channels = int(output.shape[1])
    if channels < 7 or channels % 7 != 0:
        raise ValueError("model output channels must be a positive multiple of 7")
    branch_count = channels // 7
    dirs = []
    for branch in range(branch_count):
        start = branch * 7
        dirs.append(output[:, start : start + 6])
    return torch.stack(dirs, dim=1)


def presence_outputs(output: torch.Tensor) -> torch.Tensor:
    if output.ndim != 5:
        raise ValueError("model output must have shape B,C,D,H,W")
    channels = int(output.shape[1])
    if channels < 7 or channels % 7 != 0:
        raise ValueError("model output channels must be a positive multiple of 7")
    branch_count = channels // 7
    presences = []
    for branch in range(branch_count):
        presences.append(output[:, branch * 7 + 6 : branch * 7 + 7])
    return torch.stack(presences, dim=1)


def build_fiber_trace_3d_model(config: dict[str, Any]) -> FiberTrace3DNet:
    model_cfg = dict(config.get("model_3d", config.get("model", {})))
    features_per_stage = _derive_features_per_stage(model_cfg)
    strides = _derive_unet_strides(
        features_per_stage,
        model_cfg,
        crop_size=config.get("patch_shape_zyx", config.get("crop_size")),
    )
    if "output_channels" in model_cfg:
        output_channels = int(model_cfg["output_channels"])
        if "direction_branch_count" in model_cfg:
            direction_branch_count = int(model_cfg["direction_branch_count"])
        else:
            if output_channels % 7 != 0:
                raise ValueError("model_3d.output_channels must be a multiple of 7")
            direction_branch_count = output_channels // 7
    else:
        direction_branch_count = int(model_cfg.get("direction_branch_count", 1))
        output_channels = direction_branch_count * 7
    return FiberTrace3DNet(
        FiberTrace3DModelConfig(
            input_channels=int(model_cfg.get("input_channels", 1)),
            output_channels=output_channels,
            direction_branch_count=direction_branch_count,
            conditioned_decoder_enabled=bool(
                model_cfg.get("conditioned_decoder_enabled", False)
            ),
            conditioned_latent_channels=int(
                model_cfg.get("conditioned_latent_channels", 64)
            ),
            conditioned_decoder_hidden_channels=int(
                model_cfg.get("conditioned_decoder_hidden_channels", 64)
            ),
            conditioned_decoder_layers=int(
                model_cfg.get("conditioned_decoder_layers", 3)
            ),
            features_per_stage=features_per_stage,
            strides=strides,
            decoder_upsample_mode=str(
                model_cfg.get("decoder_upsample_mode", "pixelshuffle")
            ),
            squeeze_excitation=bool(model_cfg.get("squeeze_excitation", False)),
            normalization=str(model_cfg.get("normalization", "batch")),
        )
    )
