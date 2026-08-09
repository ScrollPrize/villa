from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from koine_machines.models.hybrid_3d2d import Local3DStem2DUNet
from koine_machines.models.resnet3d import ResNet3DSegmentationModel


def _config_dict_to_mgr(config_dict, *, op_dims=3):
    model_config = dict(config_dict.get("model_config", {}) or {})
    targets = config_dict.get("targets")

    mgr = SimpleNamespace()
    mgr.model_config = model_config
    crop_size = config_dict["crop_size"]
    if isinstance(crop_size, (list, tuple)):
        mgr.train_patch_size = tuple(crop_size)
    else:
        mgr.train_patch_size = (crop_size, crop_size, crop_size)
    mgr.train_batch_size = int(config_dict.get("batch_size", 1))
    mgr.in_channels = int(config_dict["in_channels"])
    mgr.model_name = config_dict.get("model_name", "ink_det")
    mgr.autoconfigure = bool(
        model_config.get("autoconfigure", config_dict.get("autoconfigure", True))
    )
    mgr.spacing = model_config.get("spacing", [1, 1, 1])
    mgr.targets = targets
    mgr.enable_deep_supervision = bool(config_dict.get("enable_deep_supervision", False))
    mgr.op_dims = int(op_dims)
    return mgr


def build_network_from_config_dict(config_dict, *, op_dims=3):
    from vesuvius.models.build.build_network_from_config import NetworkFromConfig

    mgr = _config_dict_to_mgr(config_dict, op_dims=op_dims)
    model = NetworkFromConfig(mgr)
    if getattr(mgr, "enable_deep_supervision", False) and hasattr(model, "task_decoders"):
        for decoder in model.task_decoders.values():
            if hasattr(decoder, "deep_supervision"):
                decoder.deep_supervision = True
    return model


class SliceChannel2DModel(nn.Module):
    """Run a 2D network with the input Z planes represented as channels."""

    def __init__(self, network: nn.Module, *, input_depth: int) -> None:
        super().__init__()
        self.network = network
        self.input_depth = int(input_depth)
        if self.input_depth <= 0:
            raise ValueError(f"input_depth must be positive, got {input_depth!r}")

    @property
    def shared_encoder(self):
        return self.network.shared_encoder

    def forward(self, image: torch.Tensor):
        if image.ndim != 5:
            raise ValueError(
                "2.5D input must have shape [batch, channel, z, y, x], "
                f"got {tuple(image.shape)}"
            )
        if int(image.shape[1]) != 1:
            raise ValueError(
                "2.5D slice-channel conversion requires one source image channel, "
                f"got {int(image.shape[1])}"
            )
        if int(image.shape[2]) != self.input_depth:
            raise ValueError(
                f"2.5D input depth mismatch: expected {self.input_depth}, "
                f"got {int(image.shape[2])}"
            )
        return self.network(image[:, 0])


def _make_2p5d_model(config):
    crop_size = config.get("crop_size", config.get("patch_size"))
    if not isinstance(crop_size, (list, tuple)) or len(crop_size) != 3:
        raise ValueError(
            "vesuvius_unet_2p5d requires crop_size or patch_size [z, y, x], "
            f"got {crop_size!r}"
        )
    input_depth = int(crop_size[0])
    network_config = dict(config)
    network_config["crop_size"] = [int(crop_size[1]), int(crop_size[2])]
    network_config["patch_size"] = list(network_config["crop_size"])
    network_config["in_channels"] = input_depth

    model_config = dict(network_config.get("model_config") or {})
    model_config.pop("input_pad_depth_to", None)
    model_config["z_projection_mode"] = "none"
    spacing = model_config.get("spacing", [1, 1, 1])
    if not isinstance(spacing, (list, tuple)) or len(spacing) not in {2, 3}:
        raise ValueError(f"2.5D model spacing must have two or three axes, got {spacing!r}")
    model_config["spacing"] = [float(value) for value in spacing[-2:]]
    network_config["model_config"] = model_config

    targets = {}
    for name, target in (network_config.get("targets") or {}).items():
        target_config = dict(target)
        target_config["z_projection_mode"] = "none"
        if isinstance(target_config.get("z_projection"), dict):
            target_config["z_projection"] = {
                **target_config["z_projection"],
                "mode": "none",
            }
        targets[name] = target_config
    network_config["targets"] = targets

    network = build_network_from_config_dict(network_config, op_dims=2)
    return SliceChannel2DModel(network, input_depth=input_depth)


def _make_3d_stem_2d_model(config):
    crop_size = config.get("crop_size", config.get("patch_size"))
    if not isinstance(crop_size, (list, tuple)) or len(crop_size) != 3:
        raise ValueError(
            "vesuvius_unet_3d_stem_2d requires crop_size or patch_size "
            f"[z, y, x], got {crop_size!r}"
        )
    input_depth = int(crop_size[0])
    source_model_config = dict(config.get("model_config") or {})
    stem_channels = int(source_model_config.pop("stem_channels", 16))
    source_model_config.pop("input_pad_depth_to", None)
    source_model_config["z_projection_mode"] = "none"
    spacing = source_model_config.get("spacing", [1, 1, 1])
    if not isinstance(spacing, (list, tuple)) or len(spacing) not in {2, 3}:
        raise ValueError(
            f"3D-stem/2D-UNet spacing must have two or three axes, got {spacing!r}"
        )
    source_model_config["spacing"] = [float(value) for value in spacing[-2:]]

    network_config = dict(config)
    network_config["crop_size"] = [int(crop_size[1]), int(crop_size[2])]
    network_config["patch_size"] = list(network_config["crop_size"])
    network_config["in_channels"] = 2 * stem_channels
    network_config["model_config"] = source_model_config
    network_config["targets"] = {
        name: {**dict(target), "z_projection_mode": "none"}
        for name, target in (network_config.get("targets") or {}).items()
    }
    network = build_network_from_config_dict(network_config, op_dims=2)
    return Local3DStem2DUNet(
        network,
        input_depth=input_depth,
        stem_channels=stem_channels,
    )


def _infer_resnet3d_depth(model_type, model_config):
    normalized = str(model_type).strip().lower()
    if normalized.startswith("resnet3d-"):
        return int(normalized.split("-", 1)[1])
    return int(
        model_config.get(
            "depth",
            model_config.get(
                "model_depth",
                model_config.get("resnet3d_model_depth", 50),
            ),
        )
    )


def _make_resnet3d_model(config):
    model_config = dict(config.get("model_config") or {})
    model_type = str(config.get("model_type", "")).strip().lower()
    backbone_pretrained_path = model_config.get("backbone_pretrained_path")
    if backbone_pretrained_path is not None:
        backbone_pretrained_path = str(Path(backbone_pretrained_path))

    return ResNet3DSegmentationModel(
        depth=_infer_resnet3d_depth(model_type, model_config),
        norm=str(model_config.get("norm", "batch")),
        group_norm_groups=int(model_config.get("group_norm_groups", 32)),
        pretrained=bool(model_config.get("pretrained", True)),
        backbone_pretrained_path=backbone_pretrained_path,
    )


def make_model(config):
    model_type = str(config.get("model_type", "")).strip().lower()

    if model_type in {"vesuvius_unet", "unet"}:
        return build_network_from_config_dict(config)
    if model_type in {"vesuvius_unet_2p5d", "unet_2p5d"}:
        return _make_2p5d_model(config)
    if model_type in {"vesuvius_unet_3d_stem_2d", "unet_3d_stem_2d"}:
        return _make_3d_stem_2d_model(config)

    if model_type == "resnet3d" or model_type.startswith("resnet3d-"):
        return _make_resnet3d_model(config)

    raise RuntimeError(
        f"unexpected model_type={config.get('model_type')!r}; expected "
        "'vesuvius_unet', 'vesuvius_unet_2p5d', "
        "'vesuvius_unet_3d_stem_2d', or 'resnet3d'"
    )
