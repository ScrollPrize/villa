import json
from pathlib import Path

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
import yaml

from vesuvius.models.build.transformers.patch_encode_decode import LayerNormNd, PatchDecode

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
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.patch_embed_size = tuple(int(v) for v in backbone.patch_size)
        self.ndim = len(self.patch_embed_size)
        self.embed_dim = int(backbone.embed_dim)
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

    def forward(self, features):
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

    def forward(self, features):
        x = features[0] if isinstance(features, list) else features
        return self.patch_decoder(x)


def _load_model_config(spec, input_channels):
    checkpoint_path, checkpoint = _resolve_checkpoint_path_and_payload(spec)
    config = _resolve_checkpoint_config(spec, checkpoint_path, checkpoint)
    model_config, dataset_config = _extract_model_and_dataset_config(config)
    if "global_crops_size" not in model_config and "global_crop_size" in dataset_config:
        model_config["global_crops_size"] = dataset_config["global_crop_size"]
    if "local_crops_size" not in model_config and "local_crop_size" in dataset_config:
        model_config["local_crops_size"] = dataset_config["local_crop_size"]
    model_config["input_channels"] = int(input_channels)
    return model_config, checkpoint_path, checkpoint


def _resolve_checkpoint_path_and_payload(spec):
    if "checkpoint_path" in spec:
        checkpoint_path = Path(spec["checkpoint_path"]).expanduser().resolve()
    else:
        checkpoint_path = Path(hf_hub_download(repo_id=spec["repo_id"], filename=spec["checkpoint"]))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return checkpoint_path, checkpoint


def _extract_model_and_dataset_config(config):
    if not isinstance(config, dict):
        return {}, {}
    if isinstance(config.get("model"), dict):
        return dict(config["model"]), dict(config.get("dataset") or {})
    return dict(config), {}


def _load_config_file(path):
    with open(path, "r", encoding="utf-8") as f:
        if path.suffix.lower() == ".json":
            return json.load(f)
        if path.suffix.lower() in {".yaml", ".yml"}:
            return yaml.safe_load(f)
    raise ValueError(f"Unsupported config file extension for DINOv2 checkpoint config: {path}")


def _candidate_local_config_paths(checkpoint_path):
    candidates = [
        checkpoint_path.with_name("config.json"),
        checkpoint_path.with_name("config.yaml"),
        checkpoint_path.with_name("config.yml"),
        checkpoint_path.with_suffix(".json"),
        checkpoint_path.with_suffix(".yaml"),
        checkpoint_path.with_suffix(".yml"),
    ]
    unique_candidates = []
    seen = set()
    for candidate in candidates:
        candidate_str = str(candidate)
        if candidate_str in seen:
            continue
        seen.add(candidate_str)
        unique_candidates.append(candidate)
    return unique_candidates


def _resolve_checkpoint_config(spec, checkpoint_path, checkpoint):
    embedded_config = checkpoint.get("config")
    if isinstance(embedded_config, dict):
        return embedded_config

    for sidecar_path in _candidate_local_config_paths(checkpoint_path):
        if sidecar_path.exists():
            return _load_config_file(sidecar_path)

    if "repo_id" in spec:
        config_path = Path(hf_hub_download(repo_id=spec["repo_id"], filename=spec["config"]))
        return _load_config_file(config_path)

    candidate_paths = ", ".join(str(path) for path in _candidate_local_config_paths(checkpoint_path))
    raise ValueError(
        f"Could not resolve DINOv2 config for local checkpoint '{checkpoint_path}'. "
        f"Expected an embedded 'config' dict or one of: {candidate_paths}"
    )


def _extract_backbone_state(candidate_state):
    if not isinstance(candidate_state, dict):
        return None

    for prefix in (
        "backbone.",
        "module.backbone.",
        "_orig_mod.backbone.",
        "module._orig_mod.backbone.",
    ):
        backbone_state = {
            key.replace(prefix, "", 1): value
            for key, value in candidate_state.items()
            if key.startswith(prefix)
        }
        if backbone_state:
            return backbone_state

    nested_state = candidate_state.get("state_dict")
    if isinstance(nested_state, dict):
        return _extract_backbone_state(nested_state)

    if all(isinstance(key, str) for key in candidate_state):
        likely_backbone_keys = {"cls_token", "mask_token", "pos_embed", "patch_embed.proj.weight"}
        if any(key in candidate_state for key in likely_backbone_keys):
            return dict(candidate_state)

    return None


def _load_teacher_backbone_state(checkpoint):
    if isinstance(checkpoint, (str, Path)):
        checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=False)
    for branch_name in ("teacher", "student"):
        backbone_state = _extract_backbone_state(checkpoint.get(branch_name))
        if backbone_state:
            return backbone_state
    direct_backbone_state = _extract_backbone_state(checkpoint)
    if direct_backbone_state:
        return direct_backbone_state
    raise ValueError("Checkpoint does not contain teacher.backbone or student.backbone weights")


def build_dinov2_backbone(name, input_channels, input_shape):
    del input_shape
    spec = _DINO_V2_PRETRAINED.get(name)
    if spec is None:
        checkpoint_path = Path(name).expanduser()
        if checkpoint_path.exists():
            if checkpoint_path.is_dir():
                raise ValueError(
                    f"Local pretrained_backbone must point to a checkpoint file, not a directory: {checkpoint_path}"
                )
            spec = {"checkpoint_path": str(checkpoint_path)}
        else:
            raise ValueError(
                f"Unknown pretrained_backbone {name!r}. Expected one of: "
                f"{', '.join(sorted(_DINO_V2_PRETRAINED))}, or a local checkpoint path"
            )
    model_config, checkpoint_path, checkpoint = _load_model_config(spec, input_channels)
    backbone = build_dinovol_2_backbone(model_config)
    backbone.load_pretrained_weights(_load_teacher_backbone_state(checkpoint), unchunk=True)
    return Dinov2Backbone(backbone)


def build_dinov2_decoder(name, encoder, num_classes):
    if name == "minimal":
        return MinimalDinov2Decoder(encoder, num_classes)
    if name == "primus_patch_decode":
        return PrimusPatchDecodeDinov2Decoder(encoder, num_classes)
    raise ValueError(
        f"Unknown pretrained_decoder_type {name!r}. Expected one of: minimal, primus_patch_decode"
    )
