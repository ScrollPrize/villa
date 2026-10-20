# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in LICENSES/Apache-2.0.txt.
#
# Modified for Dinovol's three-dimensional training pipeline.

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Any, Mapping, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import trunc_normal_
from torch.nn.utils.parametrizations import weight_norm
from torch.utils.checkpoint import checkpoint

from dinovol_2.model.dinov2_eva import Eva, EvaWithChunking
from dinovol_2.ops.point_embeddings import sample_grid_points
from dinovol_2.model.rope import MixedRopePositionEmbedding, RopePositionEmbedding


_BACKBONE_DEFAULTS = {
    "attention_type": "full",
    "attention_backend": None,
    "input_channels": 1,
    "global_crops_size": (256, 256, 256),
    "local_crops_size": None,
    "embed_dim": 864,
    "patch_size": (16, 16, 16),
    "embedding_type": "default",
    "deeper_embed_batch_chunk_size": 1,
    "deeper_embed_stage_channels": None,
    "deeper_embed_stage_strides": None,
    "deeper_embed_stem_channels": None,
    "deeper_embed_stem_type": None,
    "deeper_embed_depth_per_level": None,
    "deeper_embed_use_stem": True,
    "deeper_embed_final_projection": True,
    "deeper_embed_channels_last": False,
    "depth": 24,
    "num_heads": 16,
    "qkv_bias": True,
    "qkv_fused": False,
    "mlp_ratio": 8.0 / 3.0,
    "swiglu_mlp": True,
    "scale_mlp": True,
    "scale_attn_inner": False,
    "proj_drop_rate": 0.0,
    "attn_drop_rate": 0.0,
    "drop_path_rate": 0.0,
    "drop_path_uniform": False,
    "init_values": None,
    "use_abs_pos_emb": False,
    "use_rot_pos_emb": True,
    "num_reg_tokens": 4,
    "untie_cls_and_patch_norms": False,
    "untie_global_and_local_cls_norm": False,
    "grad_checkpointing": False,
    "block_chunks": 0,
}

_BACKBONE_DEFAULTS_V2 = {
    "attention_type": "full",
    "attention_backend": None,
    "input_channels": 1,
    "global_crops_size": (256, 256, 256),
    "local_crops_size": None,
    "embed_dim": 864,
    "patch_size": (16, 16, 16),
    "embedding_type": "default",
    "deeper_embed_batch_chunk_size": 1,
    "deeper_embed_stage_channels": None,
    "deeper_embed_stage_strides": None,
    "deeper_embed_stem_channels": None,
    "deeper_embed_stem_type": None,
    "deeper_embed_depth_per_level": None,
    "deeper_embed_use_stem": True,
    "deeper_embed_final_projection": True,
    "deeper_embed_channels_last": False,
    "depth": 24,
    "num_heads": 16,
    "qkv_bias": True,
    "qkv_fused": True,
    "mlp_ratio": 8.0 / 3.0,
    "swiglu_mlp": True,
    "scale_mlp": True,
    "scale_attn_inner": False,
    "proj_drop_rate": 0.0,
    "attn_drop_rate": 0.0,
    "drop_path_rate": 0.3,
    "drop_path_uniform": False,
    "init_values": 1e-5,
    "use_abs_pos_emb": False,
    "use_rot_pos_emb": True,
    "num_reg_tokens": 4,
    "untie_cls_and_patch_norms": False,
    "untie_global_and_local_cls_norm": False,
    "grad_checkpointing": False,
    "block_chunks": 0,
}

_BACKBONE_DEFAULTS_V3 = {
    **_BACKBONE_DEFAULTS_V2,
    "qkv_bias": False,
    # Effective SwiGLU width: DINOv3's ffn_ratio=3 includes a 2/3 adjustment.
    "mlp_ratio": 2.0,
    "scale_mlp": False,
    "untie_global_and_local_cls_norm": True,
}

_ROPE_KWARG_CONFIG_KEYS = {
    "base": "rope_base",
    "min_period": "rope_min_period",
    "max_period": "rope_max_period",
    "normalize_coords": "rope_normalize_coords",
    "shift_coords": "rope_shift_coords",
    "jitter_coords": "rope_jitter_coords",
    "rescale_coords": "rope_rescale_coords",
    "dtype": "rope_dtype",
}
_ROPE_KWARG_DEFAULTS = {
    "base": 100.0,
    "normalize_coords": "separate",
    "rescale_coords": 2.0,
}
_ROPE_KWARG_FALLBACK_KEYS = {
    "base": "pos_embed_rope_base",
    "min_period": "pos_embed_rope_min_period",
    "max_period": "pos_embed_rope_max_period",
    "normalize_coords": "pos_embed_rope_normalize_coords",
    "shift_coords": "pos_embed_rope_shift_coords",
    "jitter_coords": "pos_embed_rope_jitter_coords",
    "rescale_coords": "pos_embed_rope_rescale_coords",
    "dtype": "pos_embed_rope_dtype",
}
_ROPE_DTYPE_ALIASES = {
    "fp32": torch.float32,
    "float32": torch.float32,
    "fp16": torch.float16,
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
}
_ROPE_IMPL_ALIASES = {
    "axial": RopePositionEmbedding,
    "default": RopePositionEmbedding,
    "mixed": MixedRopePositionEmbedding,
    "mixed_learnable": MixedRopePositionEmbedding,
}
_HEAD_DEFAULTS = {
    "hidden_dim": 2048,
    "bottleneck_dim": 256,
    "nlayers": 3,
    "use_bn": False,
    "norm_last_layer": False,
}
_HEAD_PREFIX_DEFAULTS = {
    "dino": {
        "bottleneck_dim": 384,
    },
    "ibot": {},
}
_HEAD_OUT_DIM_DEFAULTS = {
    "dino": 131072,
    "ibot": 131072,
}


def _resolve_backbone_defaults(config: Mapping[str, Any]) -> Mapping[str, Any]:
    model_type = config.get("model_type")
    if isinstance(model_type, str) and model_type.strip().lower() == "v3":
        return _BACKBONE_DEFAULTS_V3
    if isinstance(model_type, str) and model_type.strip().lower() == "v2":
        return _BACKBONE_DEFAULTS_V2
    return _BACKBONE_DEFAULTS


def _resolve_default_rope_type(config: Mapping[str, Any]) -> str:
    model_type = config.get("model_type")
    if isinstance(model_type, str) and model_type.strip().lower() == "v2":
        return "mixed"
    return "axial"


def _as_3tuple(value: int | Tuple[int, int, int]) -> Tuple[int, int, int]:
    if isinstance(value, int):
        return (value, value, value)
    result = tuple(int(v) for v in value)
    if len(result) != 3:
        raise ValueError(f"expected 3 values and got {len(result)}: {result}")
    return result


def _config_value(
    config: Mapping[str, Any],
    key: str,
    default: Any,
    *,
    fallback_key: Optional[str] = None,
) -> Any:
    if key in config:
        return config[key]
    if fallback_key is not None and fallback_key in config:
        return config[fallback_key]
    return default


def _resolve_rope_dtype(value: Any) -> Any:
    if value is None or isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _ROPE_DTYPE_ALIASES:
            return _ROPE_DTYPE_ALIASES[normalized]
    raise ValueError(f"unsupported rope dtype value: {value!r}")


def _resolve_rope_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    rope_kwargs = dict(config.get("rope_kwargs") or {})
    for rope_key, config_key in _ROPE_KWARG_CONFIG_KEYS.items():
        if rope_key in rope_kwargs:
            continue
        fallback_key = _ROPE_KWARG_FALLBACK_KEYS[rope_key]
        if config_key in config or fallback_key in config:
            rope_kwargs[rope_key] = _config_value(config, config_key, None, fallback_key=fallback_key)

    for rope_key, default_value in _ROPE_KWARG_DEFAULTS.items():
        rope_kwargs.setdefault(rope_key, default_value)

    if "dtype" in rope_kwargs:
        rope_kwargs["dtype"] = _resolve_rope_dtype(rope_kwargs["dtype"])

    return rope_kwargs


def _resolve_rope_impl(
    config: Mapping[str, Any],
    rope_kwargs: Mapping[str, Any],
) -> tuple[type[nn.Module], dict[str, Any]]:
    resolved_kwargs = dict(rope_kwargs)
    rope_type = _config_value(config, "rope_type", None, fallback_key="pos_embed_rope_type")
    if rope_type is None:
        rope_type = resolved_kwargs.pop("type", resolved_kwargs.pop("rope_type", _resolve_default_rope_type(config)))

    if isinstance(rope_type, str):
        normalized = rope_type.strip().lower()
        if normalized not in _ROPE_IMPL_ALIASES:
            raise ValueError(
                f"unsupported rope_type={rope_type!r}; expected one of {sorted(_ROPE_IMPL_ALIASES)}"
            )
        return _ROPE_IMPL_ALIASES[normalized], resolved_kwargs

    if isinstance(rope_type, type) and issubclass(rope_type, nn.Module):
        return rope_type, resolved_kwargs

    raise ValueError(f"unsupported rope_type value: {rope_type!r}")


def _materialize_backbone_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if config.get("deeper_embed_patch_chunk_size") is not None:
        raise ValueError("Spatial CNN chunking is no longer supported; use deeper_embed_batch_chunk_size instead.")
    backbone_defaults = _resolve_backbone_defaults(config)
    backbone_config = {key: config.get(key, default) for key, default in backbone_defaults.items()}
    if "num_reg_tokens" not in config and "num_register_tokens" in config:
        backbone_config["num_reg_tokens"] = int(config["num_register_tokens"])

    rope_kwargs = _resolve_rope_kwargs(config)
    rope_impl, rope_kwargs = _resolve_rope_impl(config, rope_kwargs)
    rope_type = next(
        name
        for name, impl_cls in _ROPE_IMPL_ALIASES.items()
        if impl_cls is rope_impl
    )

    global_crops_size = _as_3tuple(backbone_config["global_crops_size"])
    local_crop_value = backbone_config["local_crops_size"] or global_crops_size
    local_crops_size = _as_3tuple(local_crop_value)

    materialized = dict(backbone_config)
    materialized.update(
        {
            "global_crops_size": global_crops_size,
            "local_crops_size": local_crops_size,
            "patch_size": _as_3tuple(backbone_config["patch_size"]),
            "rope_type": rope_type,
            "rope_kwargs": dict(rope_kwargs),
        }
    )
    if "model_type" in config:
        materialized["model_type"] = config["model_type"]
    return materialized


def _get_vit_lr_decay_rate(
    name: str,
    *,
    lr_decay_rate: float,
    num_layers: int,
    chunked_blocks: bool,
) -> float:
    layer_id = num_layers + 1
    if name.startswith("backbone"):
        if any(
            token in name
            for token in (
                ".pos_embed",
                ".patch_embed",
                ".down_projection",
                ".mask_token",
                ".cls_token",
                ".reg_token",
            )
        ):
            layer_id = 0
        elif ".blocks." in name and ".residual." not in name:
            block_path = name[name.find(".blocks.") + 1 :].split(".")
            if chunked_blocks and len(block_path) > 2 and block_path[2].isdigit():
                layer_id = int(block_path[2]) + 1
            elif len(block_path) > 1 and block_path[1].isdigit():
                layer_id = int(block_path[1]) + 1

    return lr_decay_rate ** (num_layers + 1 - layer_id)


def _fuse_params_groups(
    all_params_groups: list[dict[str, Any]],
    *,
    keys: tuple[str, ...] = ("lr_multiplier", "wd_multiplier", "is_last_layer"),
) -> list[dict[str, Any]]:
    fused: dict[str, dict[str, Any]] = defaultdict(lambda: {"params": []})
    for group in all_params_groups:
        identifier = "_".join(f"{key}={group[key]}" for key in keys)
        fused_group = fused[identifier]
        for key in keys:
            fused_group[key] = group[key]
        fused_group["params"].append(group["params"])
    return list(fused.values())


def _weight_norm_scale_parameter(module: nn.Module) -> nn.Parameter:
    return module.parametrizations.weight.original0


def _upgrade_weight_norm_state_dict_keys(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    upgraded = dict(state_dict)
    for key in list(upgraded):
        if key.endswith(".last_layer.weight_g"):
            upgraded[key.replace(".last_layer.weight_g", ".last_layer.parametrizations.weight.original0")] = upgraded.pop(key)
        elif key.endswith(".last_layer.weight_v"):
            upgraded[key.replace(".last_layer.weight_v", ".last_layer.parametrizations.weight.original1")] = upgraded.pop(key)
    return upgraded


class DINOHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        nlayers: int = 3,
        use_bn: bool = False,
        norm_last_layer: bool = False,
    ) -> None:
        super().__init__()
        if nlayers < 1:
            raise ValueError(f"DINO head needs at least one layer, got nlayers={nlayers}")

        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())

            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())

            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)

        self.apply(self._init_weights)
        self.last_layer = weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        last_layer_scale = _weight_norm_scale_parameter(self.last_layer)
        last_layer_scale.data.fill_(1.0)
        if norm_last_layer:
            last_layer_scale.requires_grad = False

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor, apply_last_layer: bool = True) -> torch.Tensor:
        x = self.mlp(x)
        eps = 1e-6 if x.dtype == torch.float16 else 1e-12
        x = F.normalize(x, dim=-1, p=2, eps=eps)
        if not apply_last_layer:
            return x
        return self.last_layer(x)


class PointDenseHead(nn.Module):
    """Per-point embedding via a 3D top-down FPN refinement neck and factored linear readouts.

    Coarse ViT patch tokens are progressively upsampled in 3D and fused with CNN skip
    feature maps using 3x3x3 residual convolutions so high-level semantics spatially gate
    fine local boundaries before point interpolation. A shared trunk feature h(x) then
    projects into two separate linear readouts:
      - `dense`: object instance subspace (sheets and fibers, within-crop)
      - `ink`: semantic ink subspace (cross-view and cross-batch)
    """

    def __init__(
        self,
        token_dim: int,
        skip_channels: Tuple[int, ...],
        hidden_dim: int = 256,
        out_dim: int = 64,
        ink_dim: int = 32,
        fpn_dim: Optional[int] = None,
    ):
        super().__init__()
        self.token_dim = int(token_dim)
        self.skip_channels = tuple(int(channels) for channels in skip_channels)
        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(out_dim)
        self.ink_dim = int(ink_dim)
        resolved_fpn_dim = int(fpn_dim) if fpn_dim is not None else max(16, min(64, self.hidden_dim // 4))
        self.fpn_dim = resolved_fpn_dim

        # Top-down 3D FPN over the 16^3 -> 32^3 CNN skip grids so 128^3 and 64^3 grids
        # are only indexed by sample_grid_points and freed before iBOT backward.
        self._fpn_skip_start = self.fpn_skip_start(len(self.skip_channels))
        fpn_skips = self.skip_channels[self._fpn_skip_start:]
        self.fpn_in = nn.Conv3d(self.token_dim, self.fpn_dim, kernel_size=1)
        self.fpn_laterals = nn.ModuleList(
            nn.Conv3d(ch, self.fpn_dim, kernel_size=1) for ch in reversed(fpn_skips)
        )
        self.fpn_refines = nn.ModuleList(
            nn.Sequential(
                nn.Conv3d(self.fpn_dim, self.fpn_dim, kernel_size=3, padding=1, groups=max(1, self.fpn_dim // 8)),
                nn.GELU(),
                nn.Conv3d(self.fpn_dim, self.fpn_dim, kernel_size=1),
            )
            for _ in fpn_skips
        )
        self.fpn_norm = nn.LayerNorm(self.fpn_dim)

        # Point-sampled CNN skips + token + refined 3D FPN feature -> shared trunk h(x).
        self.skip_norms = nn.ModuleList(nn.LayerNorm(channels) for channels in self.skip_channels)
        in_features = self.token_dim + sum(self.skip_channels) + self.fpn_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_features, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.token_shortcut = nn.Linear(self.token_dim, self.hidden_dim)
        self.trunk_norm = nn.LayerNorm(self.hidden_dim)

        # Separate linear readouts so sheet identity and ink presence never collide.
        self.obj_proj = nn.Linear(self.hidden_dim, self.out_dim)
        self.ink_proj = nn.Linear(self.hidden_dim, self.ink_dim)

    @staticmethod
    def fpn_skip_start(num_skips: int) -> int:
        """Index of the first (finest) CNN skip level the FPN refines; finer ones are only sampled."""
        if num_skips > 2:
            return 2
        return 1 if num_skips == 2 else 0

    def _sample_fpn_points(
        self,
        token_grid: torch.Tensor,
        skips: list[torch.Tensor],
        point_coordinates: torch.Tensor,
        input_shape: Tuple[int, ...],
    ) -> torch.Tensor:
        """Run top-down 3D FPN refinement and sample the target point coordinates (no checkpointing)."""
        dtype = token_grid.dtype
        coarser_skips = skips[self._fpn_skip_start:]
        x = self.fpn_in(token_grid.permute(0, 4, 1, 2, 3).contiguous())
        for skip, lateral, refine in zip(reversed(coarser_skips), self.fpn_laterals, self.fpn_refines):
            target_size = skip.shape[2:]
            if x.shape[2:] != target_size:
                x = F.interpolate(x, size=target_size, mode="trilinear", align_corners=False)
            x = x + lateral(skip.to(dtype))
            x = x + refine(x)
        fpn_grid = x.permute(0, 2, 3, 4, 1).contiguous()
        return sample_grid_points(fpn_grid, point_coordinates, input_shape).to(dtype)

    def forward_points(
        self,
        token_grid: torch.Tensor,
        skips: list[torch.Tensor],
        point_coordinates: torch.Tensor,
        input_shape: Tuple[int, ...],
        skip_points: Optional[list[torch.Tensor]] = None,
    ) -> dict[str, torch.Tensor]:
        """skip_points: the skips already sampled at point_coordinates (the backbone's
        embed_skip_points); skips then only needs the FPN levels."""
        dtype = token_grid.dtype
        tokens = sample_grid_points(token_grid, point_coordinates, input_shape)
        fpn_pts = self.fpn_norm(self._sample_fpn_points(token_grid, skips, point_coordinates, input_shape))
        if skip_points is None:
            skip_points = [
                sample_grid_points(level.permute(0, 2, 3, 4, 1), point_coordinates, input_shape) for level in skips
            ]
        skip_parts = [norm(points.to(dtype)) for norm, points in zip(self.skip_norms, skip_points)]
        tokens_cast = tokens.to(dtype)
        trunk = self.trunk_norm(
            self.mlp(torch.cat([tokens_cast, *skip_parts, fpn_pts], dim=-1)) + self.token_shortcut(tokens_cast)
        )
        return {
            "dense": F.normalize(self.obj_proj(trunk).float(), dim=-1),
            "ink": F.normalize(self.ink_proj(trunk).float(), dim=-1),
            "tokens": tokens,
        }

    def forward(self, tokens: torch.Tensor, skips: torch.Tensor) -> torch.Tensor:
        """Fallback point-only forward when pre-sampled tokens and skip features are passed."""
        dtype = tokens.dtype
        parts = [tokens] + [
            norm(level) for norm, level in zip(self.skip_norms, skips.split(self.skip_channels, dim=-1))
        ]
        zeros_fpn = tokens.new_zeros(*tokens.shape[:-1], self.fpn_dim)
        trunk = self.trunk_norm(
            self.mlp(torch.cat([*parts, zeros_fpn], dim=-1)) + self.token_shortcut(tokens.to(dtype))
        )
        return F.normalize(self.obj_proj(trunk).float(), dim=-1)


class DinoVitStudentTeacher(nn.Module):
    """Minimal 3D DINO-style student/teacher wrapper around the local EVA backbone."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.config = dict(config)
        student_backbone = self._build_backbone(self.config)
        # dense_head: point heads for the object losses (see pretrain.object_losses).
        self.dense_config = dict(self.config.get("dense_head") or {})
        self.dense_enabled = bool(self.dense_config.get("enabled", False))
        if self.dense_enabled:
            student_backbone.return_embed_skips = True
            student_backbone.embed_skip_grid_start = PointDenseHead.fpn_skip_start(
                len(student_backbone.down_projection.skip_channels)
            )
        teacher_backbone = deepcopy(student_backbone)

        student_modules = {
            "backbone": student_backbone,
            "dino_head": self._build_head("dino"),
            "ibot_head": self._build_head("ibot", fallback_prefix="dino"),
        }
        teacher_modules = {
            "backbone": teacher_backbone,
            "dino_head": self._build_head("dino"),
            "ibot_head": self._build_head("ibot", fallback_prefix="dino"),
        }
        if self.dense_enabled:
            for modules in (student_modules, teacher_modules):
                modules.update(self._build_dense_heads(student_backbone))

        self.student = nn.ModuleDict(student_modules)
        self.teacher = nn.ModuleDict(teacher_modules)

        pretrained_weights = self.config.get("pretrained_weights")
        if pretrained_weights:
            self.load_pretrained_weights(
                pretrained_weights,
                backbone_only=bool(self.config.get("pretrained_backbone_only", True)),
                unchunk=bool(self.config.get("pretrained_unchunk", False)),
            )

        self.synchronize_teacher_from_student()
        self._freeze_teacher()

    @staticmethod
    def _build_backbone(config: Mapping[str, Any]) -> nn.Module:
        backbone_config = _materialize_backbone_config(config)
        rope_impl, rope_kwargs = _resolve_rope_impl(config, backbone_config["rope_kwargs"])
        global_crops_size = _as_3tuple(backbone_config["global_crops_size"])
        local_crops_size = _as_3tuple(backbone_config["local_crops_size"])
        block_chunks = int(backbone_config["block_chunks"])
        backbone_cls = EvaWithChunking if block_chunks > 0 else Eva
        kwargs = dict(backbone_config)
        kwargs.update(
            {
                "global_crops_size": global_crops_size,
                "local_crops_size": local_crops_size,
                "patch_size": _as_3tuple(backbone_config["patch_size"]),
                "rope_impl": rope_impl,
                "rope_kwargs": rope_kwargs,
            }
        )
        kwargs.pop("block_chunks")
        kwargs.pop("model_type", None)
        kwargs.pop("rope_type", None)
        if backbone_cls is EvaWithChunking:
            kwargs["block_chunks"] = block_chunks
        return backbone_cls(**kwargs)

    def _build_head(self, prefix: str, fallback_prefix: Optional[str] = None) -> DINOHead:
        backbone_defaults = _resolve_backbone_defaults(self.config)
        prefix_defaults = {**_HEAD_DEFAULTS, **_HEAD_PREFIX_DEFAULTS.get(prefix, {})}
        kwargs = {
            suffix: _config_value(
                self.config,
                f"{prefix}_head_{suffix}",
                prefix_defaults[suffix],
                fallback_key=f"{fallback_prefix}_head_{suffix}" if fallback_prefix else None,
            )
            for suffix in _HEAD_DEFAULTS
        }
        out_dim = _config_value(
            self.config,
            f"{prefix}_out_dim",
            self.config.get(
                "dino_out_dim",
                _HEAD_OUT_DIM_DEFAULTS.get(fallback_prefix, _HEAD_OUT_DIM_DEFAULTS["dino"])
                if fallback_prefix
                else _HEAD_OUT_DIM_DEFAULTS[prefix],
            ),
            fallback_key=f"{fallback_prefix}_out_dim" if fallback_prefix else None,
        )
        return DINOHead(
            in_dim=int(self.config.get("embed_dim", backbone_defaults["embed_dim"])),
            out_dim=int(out_dim),
            **kwargs,
        )

    def _build_dense_heads(self, backbone: nn.Module) -> dict[str, nn.Module]:
        token_dim = int(backbone.embed_dim)
        hidden_dim = int(self.dense_config.get("hidden_dim", 256))
        out_dim = int(self.dense_config.get("out_dim", 64))
        ink_dim = int(self.dense_config.get("ink_dim", min(32, out_dim)))
        fpn_dim = self.dense_config.get("fpn_dim")
        return {
            "dense_head": PointDenseHead(
                token_dim,
                backbone.down_projection.skip_channels,
                hidden_dim=hidden_dim,
                out_dim=out_dim,
                ink_dim=ink_dim,
                fpn_dim=int(fpn_dim) if fpn_dim is not None else None,
            ),
            "region_head": nn.Sequential(
                nn.Linear(token_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, int(self.dense_config.get("region_dim", 128))),
            ),
        }

    def _freeze_teacher(self) -> None:
        for parameter in self.teacher.parameters():
            parameter.requires_grad = False

    def train(self, mode: bool = True) -> "DinoVitStudentTeacher":
        super().train(mode)
        self.teacher.eval()
        return self

    def synchronize_teacher_from_student(self) -> None:
        self.teacher.load_state_dict(self.student.state_dict(), strict=True)

    def get_params_groups(
        self,
        *,
        lr_decay_rate: float = 1.0,
        patch_embed_lr_mult: float = 1.0,
    ) -> list[dict[str, Any]]:
        no_decay_names: set[str] = set()
        for module_name, module in self.student.named_modules():
            if module is self.student or not hasattr(module, "no_weight_decay"):
                continue
            module_prefix = f"{module_name}." if module_name else ""
            no_decay_names.update(module_prefix + name for name in module.no_weight_decay())

        backbone = self.student.backbone
        backbone_defaults = _resolve_backbone_defaults(self.config)
        num_layers = int(self.config.get("depth", backbone_defaults["depth"]))
        chunked_blocks = bool(getattr(backbone, "chunked_blocks", False))

        params_groups: list[dict[str, Any]] = []
        for name, parameter in self.student.named_parameters():
            if not parameter.requires_grad:
                continue

            group = {
                "params": parameter,
                "is_last_layer": "last_layer" in name,
                "lr_multiplier": _get_vit_lr_decay_rate(
                    name,
                    lr_decay_rate=lr_decay_rate,
                    num_layers=num_layers,
                    chunked_blocks=chunked_blocks,
                ),
                "wd_multiplier": 1.0,
            }

            if (
                name in no_decay_names
                or name.endswith(".bias")
                or "norm" in name
                or "gamma" in name
            ):
                group["wd_multiplier"] = 0.0

            if "patch_embed" in name or "down_projection" in name:
                group["lr_multiplier"] *= patch_embed_lr_mult

            params_groups.append(group)

        return _fuse_params_groups(params_groups)

    def load_pretrained_weights(self, checkpoint_path: str, *, backbone_only: bool = True, unchunk: bool = False) -> None:
        if backbone_only:
            self.student.backbone.load_pretrained_weights(checkpoint_path, backbone_only=True, unchunk=unchunk)
            self.synchronize_teacher_from_student()
            self._freeze_teacher()
            return

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "teacher" in checkpoint:
            state_dict = checkpoint["teacher"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
        missing, unexpected = self.student.load_state_dict(_upgrade_weight_norm_state_dict_keys(state_dict), strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected keys in DINO checkpoint: {unexpected}")
        if missing:
            raise RuntimeError(f"Missing keys while loading DINO checkpoint: {missing}")
        self.synchronize_teacher_from_student()
        self._freeze_teacher()

    @torch.no_grad()
    def update_teacher(self, momentum: float) -> None:
        if not 0.0 <= momentum <= 1.0:
            raise ValueError(f"EMA momentum must be in [0, 1], got {momentum}")
        teacher_params = [parameter.data for parameter in self.teacher.parameters()]
        student_params = [parameter.data for parameter in self.student.parameters()]
        torch._foreach_mul_(teacher_params, momentum)
        torch._foreach_add_(teacher_params, student_params, alpha=1.0 - momentum)

    @staticmethod
    def _apply_head(head: nn.Module, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim == 2:
            return head(tokens)
        leading_shape = tokens.shape[:-1]
        projections = head(tokens.reshape(-1, tokens.shape[-1]))
        return projections.reshape(*leading_shape, projections.shape[-1])

    @staticmethod
    def select_masked_tokens(
        tokens: torch.Tensor,
        mask_indices_list: torch.Tensor,
        n_masked_patches: Optional[int] = None,
    ) -> torch.Tensor:
        masked_tokens = torch.index_select(
            tokens.flatten(0, 1),
            dim=0,
            index=mask_indices_list,
        )
        if n_masked_patches is not None:
            masked_tokens = masked_tokens[:n_masked_patches]
        return masked_tokens

    @staticmethod
    def select_masked_patch_tokens(
        patch_tokens: torch.Tensor,
        mask_indices_list: torch.Tensor,
        n_masked_patches: Optional[int] = None,
    ) -> torch.Tensor:
        return DinoVitStudentTeacher.select_masked_tokens(
            patch_tokens,
            mask_indices_list=mask_indices_list,
            n_masked_patches=n_masked_patches,
        )

    def project_cls_tokens(self, branch: nn.ModuleDict, cls_tokens: torch.Tensor) -> torch.Tensor:
        return self._apply_head(branch.dino_head, cls_tokens)

    def project_patch_tokens(self, branch: nn.ModuleDict, patch_tokens: torch.Tensor) -> torch.Tensor:
        return self._apply_head(branch.ibot_head, patch_tokens)

    def project_masked_patch_tokens(
        self,
        branch: nn.ModuleDict,
        patch_tokens: torch.Tensor,
        mask_indices_list: torch.Tensor,
        n_masked_patches: Optional[int] = None,
    ) -> torch.Tensor:
        masked_tokens = self.select_masked_patch_tokens(
            patch_tokens,
            mask_indices_list=mask_indices_list,
            n_masked_patches=n_masked_patches,
        )
        return self.project_patch_tokens(branch, masked_tokens)

    def project_global_cls_and_masked_patch_tokens(
        self,
        branch: nn.ModuleDict,
        cls_tokens: torch.Tensor,
        patch_tokens: torch.Tensor,
        mask_indices_list: torch.Tensor,
        n_masked_patches: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cls_projections = self.project_cls_tokens(branch, cls_tokens)
        patch_projections = self.project_masked_patch_tokens(
            branch,
            patch_tokens,
            mask_indices_list=mask_indices_list,
            n_masked_patches=n_masked_patches,
        )
        return cls_projections, patch_projections

    def _structured_global_outputs(
        self,
        branch: nn.ModuleDict,
        global_outputs: Mapping[str, torch.Tensor],
        mask_indices_list: torch.Tensor,
        n_masked_patches: Optional[int],
        masked_patch_features: bool,
        masked_source_tokens: Optional[torch.Tensor] = None,
    ) -> dict[str, Mapping[str, torch.Tensor] | torch.Tensor]:
        """masked_source_tokens: tokens to select the masked ones from, if not the patch tokens."""
        global_outputs = dict(global_outputs)
        cls_projections = self.project_cls_tokens(branch, global_outputs["cls_tokens"])
        outputs: dict[str, Mapping[str, torch.Tensor] | torch.Tensor] = {
            "global": global_outputs,
            "global_cls_projections": cls_projections,
        }
        masked_tokens = self.select_masked_patch_tokens(
            global_outputs["patch_tokens"] if masked_source_tokens is None else masked_source_tokens,
            mask_indices_list=mask_indices_list,
            n_masked_patches=n_masked_patches,
        )
        if masked_patch_features:
            # Normalized iBOT bottleneck features and the last-layer weight; the loss
            # forms the [n_masked, out_dim] logits chunk by chunk. Reading the weight
            # here keeps its parameters inside the (DDP) forward pass.
            outputs["global_masked_patch_features"] = branch.ibot_head(masked_tokens, apply_last_layer=False)
            outputs["ibot_last_layer_weight"] = branch.ibot_head.last_layer.weight
        else:
            outputs["global_masked_patch_projections"] = self.project_patch_tokens(branch, masked_tokens)
        return outputs

    def _format_branch_outputs(
        self,
        branch: nn.ModuleDict,
        backbone_outputs: Mapping[str, torch.Tensor],
        project_cls_tokens: bool = True,
        project_patch_tokens: bool = False,
    ) -> dict[str, torch.Tensor]:
        cls_tokens = backbone_outputs["x_norm_clstoken"]
        if cls_tokens is None:
            raise RuntimeError("DINO requires a backbone that returns class tokens.")

        outputs: dict[str, torch.Tensor] = {
            "cls_tokens": cls_tokens,
            "patch_tokens": backbone_outputs["x_norm_patchtokens"],
        }
        for key in ("embed_skips", "embed_skip_points"):
            if key in backbone_outputs:
                outputs[key] = backbone_outputs[key]
        if project_cls_tokens:
            outputs["cls_projections"] = self.project_cls_tokens(branch, cls_tokens)
        if project_patch_tokens:
            outputs["patch_projections"] = self.project_patch_tokens(branch, backbone_outputs["x_norm_patchtokens"])
        return outputs

    def _forward_branch(
        self,
        branch: nn.ModuleDict,
        x: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
        project_cls_tokens: bool = True,
        project_patch_tokens: bool = False,
        *,
        view_kind: str = "global",
        point_coordinates: Optional[torch.Tensor] = None,
    ) -> Mapping[str, torch.Tensor] | list[dict[str, torch.Tensor]]:
        """point_coordinates: dense-head points; the backbone then samples its CNN skips there."""
        backbone_outputs = branch.backbone(
            x, masks=masks, is_training=self.training, view_kind=view_kind, point_coordinates=point_coordinates
        )
        if isinstance(backbone_outputs, list):
            return [
                self._format_branch_outputs(
                    branch,
                    output,
                    project_cls_tokens=project_cls_tokens,
                    project_patch_tokens=project_patch_tokens,
                )
                for output in backbone_outputs
            ]
        return self._format_branch_outputs(
            branch,
            backbone_outputs,
            project_cls_tokens=project_cls_tokens,
            project_patch_tokens=project_patch_tokens,
        )

    @staticmethod
    def _cross_view_tokens(
        branch: nn.ModuleDict,
        patch_tokens: torch.Tensor,
        coordinates: torch.Tensor,
        rows: torch.Tensor,
        input_shape: Tuple[int, ...],
    ) -> torch.Tensor:
        """Patch tokens of view rows[r] interpolated at coordinates[r] (input voxels), [rows, P, C]."""
        token_shape = tuple(size // patch for size, patch in zip(input_shape, branch.backbone.patch_size))
        token_grid = patch_tokens.reshape(patch_tokens.shape[0], *token_shape, patch_tokens.shape[-1])
        # Coordinates outside the view (tokens without a counterpart, left out of the
        # loss by the trainer) are clamped to the border by the sampler.
        return sample_grid_points(token_grid[rows], coordinates.float(), input_shape).to(patch_tokens.dtype)

    def _point_features(
        self,
        branch: nn.ModuleDict,
        outputs: dict[str, Any],
        point_coordinates: torch.Tensor,
        input_shape: Tuple[int, ...],
    ) -> dict[str, torch.Tensor]:
        """Dense/ink embeddings and interpolated tokens at [views*batch, P, 3] view coordinates.

        The embedding skips are dropped from `outputs` afterwards so their memory is
        released (the teacher's) or left to autograd (the student's).
        """
        skips = outputs.pop("embed_skips")
        skip_points = outputs.pop("embed_skip_points", None)
        patch_tokens = outputs["patch_tokens"]
        token_shape = tuple(size // patch for size, patch in zip(input_shape, branch.backbone.patch_size))
        token_grid = patch_tokens.reshape(patch_tokens.shape[0], *token_shape, patch_tokens.shape[-1])
        return branch.dense_head.forward_points(token_grid, skips, point_coordinates, input_shape, skip_points)

    def forward(
        self,
        student_input: torch.Tensor,
        teacher_input: Optional[torch.Tensor] = None,
        student_masks: Optional[torch.Tensor] = None,
        teacher_masks: Optional[torch.Tensor] = None,
        local_student_input: Optional[torch.Tensor] = None,
        mask_indices_list: Optional[torch.Tensor] = None,
        n_masked_patches: Optional[int] = None,
        *,
        return_teacher: bool = True,
        project_student_patch_tokens: bool = False,
        project_teacher_patch_tokens: bool = False,
        masked_patch_features: bool = False,
        point_coordinates: Optional[torch.Tensor] = None,
        teacher_target_coordinates: Optional[torch.Tensor] = None,
        teacher_target_rows: Optional[torch.Tensor] = None,
    ) -> dict[str, Mapping[str, torch.Tensor] | dict[str, Mapping[str, torch.Tensor] | torch.Tensor]]:
        """teacher_target_coordinates [rows, tokens, 3] and teacher_target_rows [rows]: when
        given, the teacher's iBOT features for masked token t of row r are read at
        teacher_target_coordinates[r, t] in the view of row teacher_target_rows[r]
        (cross-view targets) instead of at the same token of the same view."""
        project_cls_tokens = mask_indices_list is None
        input_shape = tuple(int(size) for size in student_input.shape[2:])
        # The teacher runs first: its no-grad intermediates are then freed before the
        # student's saved activations accumulate, which lowers peak memory.
        teacher_result = None
        if return_teacher:
            teacher_source = student_input if teacher_input is None else teacher_input
            with torch.no_grad():
                teacher_outputs = self._forward_branch(
                    self.teacher,
                    teacher_source,
                    masks=teacher_masks,
                    project_cls_tokens=project_cls_tokens,
                    project_patch_tokens=project_teacher_patch_tokens and mask_indices_list is None,
                    view_kind="global",
                    point_coordinates=point_coordinates,
                )
                teacher_points = None
                if point_coordinates is not None and "embed_skips" in teacher_outputs:
                    teacher_points = self._point_features(self.teacher, teacher_outputs, point_coordinates, input_shape)
                teacher_outputs.pop("embed_skips", None)
                teacher_outputs.pop("embed_skip_points", None)
                if mask_indices_list is None:
                    teacher_result = teacher_outputs
                else:
                    target_tokens = None
                    if teacher_target_coordinates is not None:
                        target_tokens = self._cross_view_tokens(
                            self.teacher, teacher_outputs["patch_tokens"], teacher_target_coordinates,
                            teacher_target_rows, input_shape,
                        )
                    teacher_result = self._structured_global_outputs(
                        self.teacher,
                        teacher_outputs,
                        mask_indices_list,
                        n_masked_patches,
                        masked_patch_features,
                        masked_source_tokens=target_tokens,
                    )
        student_outputs = self._forward_branch(
            self.student,
            student_input,
            masks=student_masks,
            project_cls_tokens=project_cls_tokens,
            project_patch_tokens=project_student_patch_tokens and mask_indices_list is None,
            view_kind="global",
            point_coordinates=point_coordinates,
        )
        student_points = None
        if point_coordinates is not None and "embed_skips" in student_outputs:
            student_points = self._point_features(self.student, student_outputs, point_coordinates, input_shape)
        student_outputs.pop("embed_skips", None)
        student_outputs.pop("embed_skip_points", None)
        if mask_indices_list is None:
            outputs: dict[str, Mapping[str, torch.Tensor] | dict[str, Mapping[str, torch.Tensor] | torch.Tensor]] = {
                "student": student_outputs
            }
        else:
            structured_student_outputs = self._structured_global_outputs(
                self.student,
                student_outputs,
                mask_indices_list,
                n_masked_patches,
                masked_patch_features,
            )
            if local_student_input is not None:
                structured_student_outputs["local"] = self._forward_branch(
                    self.student,
                    local_student_input,
                    masks=None,
                    project_cls_tokens=True,
                    view_kind="local",
                )
            outputs = {"student": structured_student_outputs}

        if teacher_result is not None:
            outputs["teacher"] = teacher_result
        if student_points is not None:
            outputs["student"]["object_points"] = student_points
        if teacher_result is not None and teacher_points is not None:
            outputs["teacher"]["object_points"] = teacher_points
        return outputs
