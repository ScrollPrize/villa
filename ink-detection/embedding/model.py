from __future__ import annotations

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
        with torch.no_grad():
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


class FrozenBackbone(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return extract_backbone_features(self.model, images)

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


def create_frozen_dino_backbone(
    backbone_name: str,
    checkpoint_path: Path | None,
    image_size: int,
    device: torch.device,
) -> tuple[FrozenBackbone, int]:
    kwargs: dict[str, Any] = {"img_size": image_size}
    if checkpoint_path is None:
        kwargs["pretrained"] = True
    else:
        kwargs["pretrained"] = False

    try:
        model = create_model(backbone_name, num_classes=0, **kwargs)
    except RuntimeError as exc:
        available = sorted(name for name in list_models("*dino*") if "dino" in name)
        raise RuntimeError(
            f"Failed to create backbone {backbone_name!r}. "
            f"Available timm DINO-like models: {available[:20]!r}"
        ) from exc

    if checkpoint_path is not None:
        state_dict = load_checkpoint_state(checkpoint_path)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"checkpoint missing keys ({len(missing)}): {missing[:8]}")
        if unexpected:
            print(f"checkpoint unexpected keys ({len(unexpected)}): {unexpected[:8]}")

    frozen = FrozenBackbone(model).to(device)
    with torch.no_grad():
        dummy = torch.zeros(1, 3, image_size, image_size, device=device)
        dim = int(frozen(dummy).shape[-1])
    return frozen, dim
