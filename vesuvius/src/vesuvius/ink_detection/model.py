"""Construction of the supported ink segmentation model families."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from vesuvius.ink_detection.config import InkConfig
from vesuvius.ink_detection.hybrid_3d2d import Local3DStem2DUNet


def _build_network(config_dict: dict, *, op_dims: int) -> nn.Module:
    from vesuvius.models.build.build_network_from_config import NetworkFromConfig

    model_config = dict(config_dict.get("model_config") or {})
    crop_size = config_dict["crop_size"]
    manager = SimpleNamespace(
        model_config=model_config,
        train_patch_size=tuple(crop_size),
        train_batch_size=int(config_dict.get("batch_size", 1)),
        in_channels=int(config_dict["in_channels"]),
        model_name=config_dict.get("model_name", "ink_det"),
        autoconfigure=bool(
            model_config.get(
                "autoconfigure", config_dict.get("autoconfigure", True)
            )
        ),
        spacing=model_config.get("spacing", [1, 1, 1]),
        targets=config_dict["targets"],
        enable_deep_supervision=bool(
            config_dict.get("enable_deep_supervision", False)
        ),
        op_dims=int(op_dims),
    )
    model = NetworkFromConfig(manager)
    if manager.enable_deep_supervision and hasattr(model, "task_decoders"):
        for decoder in model.task_decoders.values():
            if hasattr(decoder, "deep_supervision"):
                decoder.deep_supervision = True
    return model


class SliceChannel2DModel(nn.Module):
    """Represent the Z planes of a one-channel BCZYX image as 2D channels."""

    def __init__(self, network: nn.Module, *, input_depth: int) -> None:
        super().__init__()
        self.network = network
        self.input_depth = int(input_depth)
        if self.input_depth <= 0:
            raise ValueError(f"input_depth must be positive, got {input_depth!r}")

    @property
    def shared_encoder(self):
        return self.network.shared_encoder

    def forward(self, image_BCZYX: torch.Tensor):
        if image_BCZYX.ndim != 5:
            raise ValueError(
                "2.5D input must have shape [batch, channel, z, y, x], "
                f"got {tuple(image_BCZYX.shape)}"
            )
        if int(image_BCZYX.shape[1]) != 1:
            raise ValueError(
                "2.5D slice-channel conversion requires one source image "
                f"channel, got {int(image_BCZYX.shape[1])}"
            )
        if int(image_BCZYX.shape[2]) != self.input_depth:
            raise ValueError(
                f"2.5D input depth mismatch: expected {self.input_depth}, "
                f"got {int(image_BCZYX.shape[2])}"
            )
        return self.network(image_BCZYX[:, 0])


def _base_build_mapping(config: InkConfig) -> dict:
    return {
        "crop_size": list(config.model.crop_size),
        "patch_size": list(config.model.crop_size),
        "batch_size": config.model.batch_size,
        "in_channels": config.model.in_channels,
        "model_name": config.model.model_name,
        "autoconfigure": config.model.autoconfigure,
        "enable_deep_supervision": config.model.enable_deep_supervision,
        "model_config": config.model.model_settings_mapping(),
        "targets": {
            name: target.to_mapping()
            for name, target in config.targets.items()
        },
    }


def _make_2p5d_model(config: InkConfig) -> SliceChannel2DModel:
    input_depth = config.model.crop_size[0]
    network_config = _base_build_mapping(config)
    network_config["crop_size"] = list(config.model.crop_size[1:])
    network_config["patch_size"] = list(network_config["crop_size"])
    network_config["in_channels"] = input_depth

    model_config = dict(network_config.get("model_config") or {})
    model_config.pop("input_pad_depth_to", None)
    model_config["z_projection_mode"] = "none"
    spacing = model_config.get("spacing", [1, 1, 1])
    if not isinstance(spacing, (list, tuple)) or len(spacing) not in {2, 3}:
        raise ValueError(
            f"2.5D model spacing must have two or three axes, got {spacing!r}"
        )
    model_config["spacing"] = [float(value) for value in spacing[-2:]]
    network_config["model_config"] = model_config

    targets = {}
    for name, target in network_config["targets"].items():
        target_config = dict(target)
        target_config["z_projection_mode"] = "none"
        if isinstance(target_config.get("z_projection"), dict):
            target_config["z_projection"] = {
                **target_config["z_projection"],
                "mode": "none",
            }
        targets[name] = target_config
    network_config["targets"] = targets
    return SliceChannel2DModel(
        _build_network(network_config, op_dims=2),
        input_depth=input_depth,
    )


def _make_3d_stem_2d_model(config: InkConfig) -> Local3DStem2DUNet:
    input_depth = config.model.crop_size[0]
    network_config = _base_build_mapping(config)
    source_model_config = dict(network_config.get("model_config") or {})
    source_model_config.pop("stem_channels", None)
    stem_channels = config.model.stem_channels
    source_model_config.pop("input_pad_depth_to", None)
    source_model_config["z_projection_mode"] = "none"
    spacing = source_model_config.get("spacing", [1, 1, 1])
    if not isinstance(spacing, (list, tuple)) or len(spacing) not in {2, 3}:
        raise ValueError(
            "3D-stem/2D-UNet spacing must have two or three axes, "
            f"got {spacing!r}"
        )
    source_model_config["spacing"] = [float(value) for value in spacing[-2:]]

    network_config["crop_size"] = list(config.model.crop_size[1:])
    network_config["patch_size"] = list(network_config["crop_size"])
    network_config["in_channels"] = 2 * stem_channels
    network_config["model_config"] = source_model_config
    network_config["targets"] = {
        name: {**dict(target), "z_projection_mode": "none"}
        for name, target in network_config["targets"].items()
    }
    return Local3DStem2DUNet(
        _build_network(network_config, op_dims=2),
        input_depth=input_depth,
        stem_channels=stem_channels,
    )


def make_model(config: InkConfig) -> nn.Module:
    """Construct the model selected by a validated ink configuration."""

    model_type = config.model.model_type
    if model_type in {"vesuvius_unet", "unet"}:
        return _build_network(_base_build_mapping(config), op_dims=3)
    if model_type in {"vesuvius_unet_2p5d", "unet_2p5d"}:
        return _make_2p5d_model(config)
    if model_type in {
        "vesuvius_unet_3d_stem_2d",
        "unet_3d_stem_2d",
    }:
        return _make_3d_stem_2d_model(config)
    raise ValueError(f"Unsupported model_type {model_type!r}")
