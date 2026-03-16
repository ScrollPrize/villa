from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model, list_models


DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)


def normalize_for_backbone(images: torch.Tensor) -> torch.Tensor:
    images = images.repeat(1, 3, 1, 1)
    mean = images.new_tensor(DEFAULT_MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(DEFAULT_STD).view(1, 3, 1, 1)
    return (images - mean) / std


class TinyEmbedder(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PCAEmbedder(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int) -> None:
        super().__init__()
        self.net = nn.Linear(input_dim, embedding_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class InkPatchEmbedder(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        backbone_dim: int,
        embedding_dim: int,
        hidden_dim: int,
        dropout: float,
        head: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = TinyEmbedder(backbone_dim, embedding_dim, hidden_dim, dropout) if head is None else head

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        embeddings = self.head(features)
        return embeddings, features


def extract_backbone_features(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "forward_features"):
        features = model.forward_features(images)
    else:
        features = model(images)

    if isinstance(features, dict):
        for key in ("x_norm_clstoken", "cls_token", "x_cls", "pooled", "embedding"):
            if key in features:
                features = features[key]
                break
        else:
            if "x_norm_patchtokens" in features:
                features = features["x_norm_patchtokens"].mean(dim=1)
            elif "last_hidden_state" in features:
                features = features["last_hidden_state"][:, 0]
            else:
                first_value = next(iter(features.values()))
                features = first_value

    if isinstance(features, (list, tuple)):
        features = features[0]

    if features.ndim == 4:
        features = F.adaptive_avg_pool2d(features, output_size=1).flatten(1)
    elif features.ndim == 3:
        features = features[:, 0]
    elif features.ndim != 2:
        raise ValueError(f"Unsupported feature shape from backbone: {tuple(features.shape)!r}")
    return features


class BackboneAdapter(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return extract_backbone_features(self.model, images)


def freeze_module_parameters(module: nn.Module) -> nn.Module:
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


def load_checkpoint_state(checkpoint_path: Path) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model", "teacher", "student"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint format at {checkpoint_path}")
    cleaned = {}
    prefixes = ("module.", "backbone.", "model.", "encoder.")
    for key, value in checkpoint.items():
        if not isinstance(value, torch.Tensor):
            continue
        new_key = key
        stripped = True
        while stripped:
            stripped = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
                    stripped = True
        cleaned[new_key] = value
    return cleaned


def _build_timm_backbone(backbone_name: str, checkpoint_path: Path | None, image_size: int) -> nn.Module:
    kwargs: dict[str, Any] = {"pretrained": checkpoint_path is None}

    try:
        return create_model(backbone_name, num_classes=0, img_size=image_size, **kwargs)
    except TypeError:
        return create_model(backbone_name, num_classes=0, **kwargs)
    except RuntimeError as exc:
        suggestions = sorted(name for name in list_models(f"*{backbone_name.split('.')[0]}*") if backbone_name.split(".")[0] in name)
        raise RuntimeError(
            f"Failed to create backbone {backbone_name!r}. "
            f"Example matching timm models: {suggestions[:20]!r}"
        ) from exc


def create_backbone(
    backbone_name: str,
    checkpoint_path: Path | None,
    image_size: int,
    device: torch.device,
) -> tuple[nn.Module, int]:
    return create_backbone_with_mode(backbone_name, checkpoint_path, image_size, device, freeze_backbone=True)


def create_backbone_with_mode(
    backbone_name: str,
    checkpoint_path: Path | None,
    image_size: int,
    device: torch.device,
    *,
    freeze_backbone: bool,
) -> tuple[nn.Module, int]:
    model = _build_timm_backbone(backbone_name, checkpoint_path, image_size)

    if checkpoint_path is not None:
        state_dict = load_checkpoint_state(checkpoint_path)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"checkpoint missing keys ({len(missing)}): {missing[:8]}")
        if unexpected:
            print(f"checkpoint unexpected keys ({len(unexpected)}): {unexpected[:8]}")

    backbone = BackboneAdapter(model).to(device)
    if freeze_backbone:
        freeze_module_parameters(backbone)
        backbone.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, 3, image_size, image_size, device=device)
        dim = int(backbone(dummy).shape[-1])
    return backbone, dim


def create_frozen_dino_backbone(
    backbone_name: str,
    checkpoint_path: Path | None,
    image_size: int,
    device: torch.device,
) -> tuple[nn.Module, int]:
    return create_backbone_with_mode(
        backbone_name,
        checkpoint_path,
        image_size,
        device,
        freeze_backbone=True,
    )


def resolve_checkpoint_config(checkpoint_path: Path) -> tuple[Path, dict[str, Any]]:
    checkpoint_path = checkpoint_path.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("config") if isinstance(checkpoint, dict) else None
    if isinstance(config, dict):
        return checkpoint_path, config

    config_path = checkpoint_path.parent / "config.json"
    if not config_path.exists():
        raise ValueError(f"Could not find config in checkpoint or at {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected JSON object at {config_path}, got {type(config).__name__}")
    return checkpoint_path, config


def build_embedder_from_config(
    config: dict[str, Any],
    *,
    device: torch.device,
    checkpoint_path: Path | None = None,
    freeze_parameters: bool | None = None,
) -> tuple[InkPatchEmbedder, dict[str, Any]]:
    checkpoint_method = str(config.get("adaptation_method", "trained")).strip().lower()
    backbone_name = str(config["backbone_name"])
    backbone_checkpoint = config.get("backbone_checkpoint")
    crop_size = int(config["crop_size"])
    embedding_dim = int(config.get("embedding_dim", 0) or 0)
    hidden_dim = int(config.get("hidden_dim", embedding_dim or 0) or 0)
    dropout = float(config.get("dropout", 0.0 if checkpoint_method in {"pca", "plain"} else 0.1))
    freeze_backbone = bool(config.get("freeze_backbone", True))

    backbone_checkpoint_path = Path(backbone_checkpoint).expanduser().resolve() if backbone_checkpoint else None
    backbone, backbone_dim = create_backbone_with_mode(
        backbone_name,
        backbone_checkpoint_path,
        crop_size,
        device,
        freeze_backbone=freeze_backbone,
    )

    if checkpoint_method == "plain":
        model = InkPatchEmbedder(
            backbone=backbone,
            backbone_dim=backbone_dim,
            embedding_dim=backbone_dim,
            hidden_dim=backbone_dim,
            dropout=0.0,
            head=nn.Identity(),
        ).to(device)
        runtime_config = {
            **config,
            "crop_size": crop_size,
            "embedding_dim": backbone_dim,
            "hidden_dim": backbone_dim,
            "dropout": 0.0,
            "freeze_backbone": freeze_backbone,
            "adaptation_method": checkpoint_method,
        }
    elif checkpoint_method == "pca":
        model = InkPatchEmbedder(
            backbone=backbone,
            backbone_dim=backbone_dim,
            embedding_dim=embedding_dim,
            hidden_dim=max(1, embedding_dim),
            dropout=0.0,
            head=PCAEmbedder(backbone_dim, embedding_dim),
        ).to(device)
        runtime_config = {
            **config,
            "crop_size": crop_size,
            "freeze_backbone": freeze_backbone,
            "adaptation_method": checkpoint_method,
        }
    else:
        model = InkPatchEmbedder(
            backbone=backbone,
            backbone_dim=backbone_dim,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        ).to(device)
        runtime_config = {
            **config,
            "crop_size": crop_size,
            "freeze_backbone": freeze_backbone,
            "adaptation_method": checkpoint_method,
        }

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state = checkpoint.get("model") if isinstance(checkpoint, dict) else None
        if not isinstance(state, dict):
            raise ValueError(f"Checkpoint {checkpoint_path} does not contain a saved model state")
        if checkpoint_method == "plain":
            pass
        elif checkpoint_method == "pca":
            missing, unexpected = model.head.load_state_dict(state, strict=True)
            if missing or unexpected:
                raise ValueError(f"Unexpected PCA head state mismatch: missing={missing}, unexpected={unexpected}")
        elif freeze_backbone:
            missing, unexpected = model.head.load_state_dict(state, strict=True)
            if missing or unexpected:
                raise ValueError(f"Unexpected head state mismatch: missing={missing}, unexpected={unexpected}")
        else:
            missing, unexpected = model.load_state_dict(state, strict=True)
            if missing or unexpected:
                raise ValueError(f"Unexpected model state mismatch: missing={missing}, unexpected={unexpected}")

    if freeze_parameters:
        freeze_module_parameters(model)
    model.eval()
    return model, runtime_config


def load_adapted_model(
    checkpoint_path: Path,
    device: torch.device,
    *,
    freeze_parameters: bool = False,
) -> tuple[InkPatchEmbedder, dict[str, Any]]:
    checkpoint_path, config = resolve_checkpoint_config(checkpoint_path)
    return build_embedder_from_config(
        config,
        device=device,
        checkpoint_path=checkpoint_path,
        freeze_parameters=freeze_parameters,
    )
