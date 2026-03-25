from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from koine_machines.models.resnet3d import ResNet3DSegmentationModel


def _config_dict_to_mgr(config_dict):
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
    mgr.op_dims = 3
    return mgr


def build_network_from_config_dict(config_dict):
    from vesuvius.models.build.build_network_from_config import NetworkFromConfig

    mgr = _config_dict_to_mgr(config_dict)
    model = NetworkFromConfig(mgr)
    if getattr(mgr, "enable_deep_supervision", False) and hasattr(model, "task_decoders"):
        for decoder in model.task_decoders.values():
            if hasattr(decoder, "deep_supervision"):
                decoder.deep_supervision = True
    return model


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

    if model_type == "resnet3d" or model_type.startswith("resnet3d-"):
        return _make_resnet3d_model(config)

    raise RuntimeError(
        f"unexpected model_type={config.get('model_type')!r}, should be 'vesuvius_unet' or 'resnet3d'"
    )
