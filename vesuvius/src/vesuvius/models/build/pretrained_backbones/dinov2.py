import json
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

from vesuvius.models.build.transformers.patch_encode_decode import (
    LayerNormNd,
    PatchDecode,
    PixelShuffle3D,
    _compute_patch_decode_plan,
)

from .dinovol_2_builder import build_dinovol_2_backbone


_DINO_V2_PRETRAINED = {
    "dinov2_ps8": {
        "repo_id": "scrollprize/dinov2_ps8",
        "checkpoint": "dinov2_ps8.pt",
        "config": "config.json",
    },
    "dinov2_ps16_crop256": {
        "repo_id": "scrollprize/dinov2_ps16_crop256",
        "checkpoint": "dinov2_ps16_crop256.pt",
        "config": "config_ps256.json",
    },
}


class Dinov2Backbone(nn.Module):
    def __init__(self, backbone, input_channels: int):
        super().__init__()
        self.backbone = backbone
        self.patch_embed_size = tuple(int(v) for v in backbone.patch_size)
        self.ndim = len(self.patch_embed_size)
        self.embed_dim = int(backbone.embed_dim)
        self.input_channels = int(input_channels)
        self.output_channels = [self.embed_dim]
        self.strides = [self.patch_embed_size]

    def forward(self, x):
        spatial = tuple(int(v) for v in x.shape[2:])
        if len(spatial) != self.ndim:
            raise ValueError(
                f"DINOv2 backbone expected {self.ndim} spatial dims, got input shape {tuple(x.shape)}"
            )
        if any(size % patch != 0 for size, patch in zip(spatial, self.patch_embed_size)):
            raise ValueError(
                f"Input shape {spatial} must be divisible by patch size {self.patch_embed_size}"
            )
        tokens = self.backbone(x)["x_norm_patchtokens"]
        grid = tuple(size // patch for size, patch in zip(spatial, self.patch_embed_size))
        features = tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[-1], *grid).contiguous()
        return [features]


class MinimalDinov2Decoder(nn.Module):
    def __init__(self, encoder, num_classes):
        super().__init__()
        conv = nn.Conv2d if encoder.ndim == 2 else nn.Conv3d
        conv_t = nn.ConvTranspose2d if encoder.ndim == 2 else nn.ConvTranspose3d
        self.project = conv(encoder.embed_dim, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.upsample = None
        if any(step > 1 for step in encoder.patch_embed_size):
            self.upsample = conv_t(
                num_classes,
                num_classes,
                kernel_size=encoder.patch_embed_size,
                stride=encoder.patch_embed_size,
                bias=True,
            )

    def forward(self, features, input_image=None):
        x = features[0] if isinstance(features, list) else features
        x = self.project(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


class PrimusPatchDecodeDinov2Decoder(nn.Module):
    def __init__(self, encoder, num_classes):
        super().__init__()
        self.patch_decoder = PatchDecode(
            patch_size=encoder.patch_embed_size,
            embed_dim=encoder.embed_dim,
            out_channels=num_classes,
            norm=LayerNormNd,
            activation=nn.GELU,
        )

    def forward(self, features, input_image=None):
        x = features[0] if isinstance(features, list) else features
        return self.patch_decoder(x)


class PixelShuffleConvDinov2Decoder(nn.Module):
    def __init__(self, encoder, num_classes):
        super().__init__()
        if encoder.ndim != 3:
            raise ValueError(
                f"pixelshuffle_conv currently only supports 3D pretrained backbones, got ndim={encoder.ndim}"
            )

        num_stages, strides, channels = _compute_patch_decode_plan(
            encoder.patch_embed_size,
            embed_dim=encoder.embed_dim,
            out_channels=num_classes,
        )
        conv = nn.Conv3d
        final_hidden_channels = max(
            8,
            channels[-2] if num_stages > 1 else min(64, max(8, encoder.embed_dim // 8)),
        )
        detail_channels = min(16, final_hidden_channels)
        stages = []
        for stage_idx in range(num_stages):
            next_channels = channels[stage_idx + 1] if stage_idx < num_stages - 1 else final_hidden_channels
            scale_factors = tuple(int(v) for v in strides[stage_idx])
            expansion_channels = int(next_channels) * int(torch.tensor(scale_factors).prod().item())
            stage_ops = [
                conv(channels[stage_idx], expansion_channels, kernel_size=1, stride=1, padding=0, bias=True),
                PixelShuffle3D(scale_factors),
                conv(next_channels, next_channels, kernel_size=3, stride=1, padding=1, bias=True),
                nn.GroupNorm(_resolve_group_norm_groups(next_channels), next_channels),
                nn.GELU(),
            ]
            stages.append(nn.Sequential(*stage_ops))
        self.decode = nn.Sequential(*stages)
        self.final_input_detail = nn.Sequential(
            conv(encoder.input_channels, detail_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(_resolve_group_norm_groups(detail_channels), detail_channels),
            nn.GELU(),
        )
        self.final_refine = nn.Sequential(
            conv(final_hidden_channels + detail_channels, final_hidden_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(_resolve_group_norm_groups(final_hidden_channels), final_hidden_channels),
            nn.GELU(),
            conv(final_hidden_channels, final_hidden_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(_resolve_group_norm_groups(final_hidden_channels), final_hidden_channels),
            nn.GELU(),
            conv(final_hidden_channels, num_classes, kernel_size=1, stride=1, padding=0, bias=True),
        )

    def forward(self, features, input_image=None):
        x = features[0] if isinstance(features, list) else features
        x = self.decode(x)
        if input_image is None:
            raise ValueError("pixelshuffle_conv decoder requires the original input_image for final refinement")
        if tuple(input_image.shape[2:]) != tuple(x.shape[2:]):
            raise ValueError(
                f"pixelshuffle_conv expected input_image spatial shape {tuple(x.shape[2:])}, "
                f"got {tuple(input_image.shape[2:])}"
            )
        detail = self.final_input_detail(input_image)
        x = torch.cat([x, detail], dim=1)
        return self.final_refine(x)


def _resolve_group_norm_groups(num_channels: int, max_groups: int = 8) -> int:
    if num_channels < 1:
        raise ValueError(f"GroupNorm requires num_channels >= 1, got {num_channels}")
    return max(1, math.gcd(int(num_channels), int(max_groups)))


def _load_model_config(spec, input_channels):
    checkpoint_path = Path(hf_hub_download(repo_id=spec["repo_id"], filename=spec["checkpoint"]))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = _resolve_checkpoint_config(spec, checkpoint_path, checkpoint)
    model_config = dict(config.get("model") or {})
    dataset_config = config.get("dataset") or {}
    if "global_crops_size" not in model_config and "global_crop_size" in dataset_config:
        model_config["global_crops_size"] = dataset_config["global_crop_size"]
    if "local_crops_size" not in model_config and "local_crop_size" in dataset_config:
        model_config["local_crops_size"] = dataset_config["local_crop_size"]
    model_config["input_channels"] = int(input_channels)
    return model_config, checkpoint_path, checkpoint


def _load_json_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_local_checkpoint_config(
    checkpoint_path: Path,
    checkpoint,
    *,
    config_path: Optional[str | Path] = None,
):
    embedded_config = checkpoint.get("config")
    if isinstance(embedded_config, dict) and isinstance(embedded_config.get("model"), dict):
        return embedded_config

    candidate_paths: list[Path] = []
    if config_path is not None:
        candidate_paths.append(Path(config_path).expanduser())
    candidate_paths.append(checkpoint_path.with_name("config.json"))
    candidate_paths.append(checkpoint_path.with_suffix(".json"))

    seen = set()
    for candidate in candidate_paths:
        resolved = candidate.resolve() if candidate.exists() else candidate
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return _load_json_config(candidate)

    raise ValueError(
        "Local Dinovol checkpoint does not contain an embedded config and no JSON config was found. "
        f"checkpoint={checkpoint_path}"
    )


def _resolve_checkpoint_config(spec, checkpoint_path, checkpoint):
    embedded_config = checkpoint.get("config")
    if isinstance(embedded_config, dict) and isinstance(embedded_config.get("model"), dict):
        return embedded_config

    sidecar_path = checkpoint_path.with_name("config.json")
    if sidecar_path.exists():
        return _load_json_config(sidecar_path)

    config_path = hf_hub_download(repo_id=spec["repo_id"], filename=spec["config"])
    return _load_json_config(config_path)


def _load_teacher_backbone_state(checkpoint):
    if isinstance(checkpoint, (str, Path)):
        checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=False)
    for branch_name in ("teacher", "student"):
        branch_state = checkpoint.get(branch_name)
        if not isinstance(branch_state, dict):
            continue
        backbone_state = {
            key.replace("backbone.", "", 1): value
            for key, value in branch_state.items()
            if key.startswith("backbone.")
        }
        if backbone_state:
            return backbone_state
    raise ValueError("Checkpoint does not contain teacher.backbone or student.backbone weights")


def _load_local_model_config(checkpoint_path, input_channels, *, config_path=None):
    checkpoint_path = Path(checkpoint_path).expanduser()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Local Dinovol checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = _resolve_local_checkpoint_config(checkpoint_path, checkpoint, config_path=config_path)
    model_config = dict(config.get("model") or {})
    dataset_config = config.get("dataset") or {}
    if "global_crops_size" not in model_config and "global_crop_size" in dataset_config:
        model_config["global_crops_size"] = dataset_config["global_crop_size"]
    if "local_crops_size" not in model_config and "local_crop_size" in dataset_config:
        model_config["local_crops_size"] = dataset_config["local_crop_size"]
    model_config["input_channels"] = int(input_channels)
    return model_config, checkpoint_path, checkpoint


def build_dinov2_backbone(name, input_channels, input_shape, *, config_path=None):
    del input_shape
    if name in _DINO_V2_PRETRAINED:
        spec = _DINO_V2_PRETRAINED[name]
        model_config, checkpoint_path, checkpoint = _load_model_config(spec, input_channels)
    else:
        model_config, checkpoint_path, checkpoint = _load_local_model_config(
            name,
            input_channels,
            config_path=config_path,
        )
    backbone = build_dinovol_2_backbone(model_config)
    backbone.load_pretrained_weights(_load_teacher_backbone_state(checkpoint), unchunk=True)
    return Dinov2Backbone(backbone, input_channels=input_channels)


def build_dinov2_decoder(name, encoder, num_classes):
    if name == "minimal":
        return MinimalDinov2Decoder(encoder, num_classes)
    if name == "primus_patch_decode":
        return PrimusPatchDecodeDinov2Decoder(encoder, num_classes)
    if name == "pixelshuffle_conv":
        return PixelShuffleConvDinov2Decoder(encoder, num_classes)
    raise ValueError(
        f"Unknown pretrained_decoder_type {name!r}. Expected one of: minimal, primus_patch_decode, pixelshuffle_conv"
    )
