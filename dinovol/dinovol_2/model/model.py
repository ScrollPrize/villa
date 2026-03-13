from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Any, Mapping, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import trunc_normal_
from torch.nn.utils.parametrizations import weight_norm

from dinovol_2.model.dinov2_eva import Eva, EvaWithChunking


_BACKBONE_DEFAULTS = {
    "input_channels": 1,
    "global_crops_size": (256, 256, 256),
    "local_crops_size": None,
    "embed_dim": 864,
    "patch_size": (16, 16, 16),
    "embedding_type": "default",
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
    "use_abs_pos_emb": True,
    "use_rot_pos_emb": True,
    "num_reg_tokens": 4,
    "grad_checkpointing": False,
    "block_chunks": 0,
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        eps = 1e-6 if x.dtype == torch.float16 else 1e-12
        x = F.normalize(x, dim=-1, p=2, eps=eps)
        return self.last_layer(x)


class DinoVitStudentTeacher(nn.Module):
    """Minimal 3D DINO-style student/teacher wrapper around the local EVA backbone."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.config = dict(config)
        student_backbone = self._build_backbone(self.config)
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
        backbone_config = {key: config.get(key, default) for key, default in _BACKBONE_DEFAULTS.items()}
        if "num_reg_tokens" not in config and "num_register_tokens" in config:
            backbone_config["num_reg_tokens"] = int(config["num_register_tokens"])
        global_crops_size = _as_3tuple(backbone_config["global_crops_size"])
        local_crop_value = backbone_config["local_crops_size"] or global_crops_size
        local_crops_size = _as_3tuple(local_crop_value)
        block_chunks = int(backbone_config["block_chunks"])
        backbone_cls = EvaWithChunking if block_chunks > 0 else Eva
        kwargs = dict(backbone_config)
        kwargs.update(
            {
                "global_crops_size": global_crops_size,
                "local_crops_size": local_crops_size,
                "patch_size": _as_3tuple(backbone_config["patch_size"]),
            }
        )
        kwargs.pop("block_chunks")
        if backbone_cls is EvaWithChunking:
            kwargs["block_chunks"] = block_chunks
        return backbone_cls(**kwargs)

    def _build_head(self, prefix: str, fallback_prefix: Optional[str] = None) -> DINOHead:
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
            in_dim=int(self.config.get("embed_dim", _BACKBONE_DEFAULTS["embed_dim"])),
            out_dim=int(out_dim),
            **kwargs,
        )

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
        num_layers = int(self.config.get("depth", _BACKBONE_DEFAULTS["depth"]))
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

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
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
        for student_param, teacher_param in zip(self.student.parameters(), self.teacher.parameters()):
            teacher_param.data.mul_(momentum).add_(student_param.data, alpha=1.0 - momentum)

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
        patch_support = backbone_outputs.get("x_patch_support")
        if patch_support is not None:
            outputs["patch_token_support"] = patch_support
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
    ) -> Mapping[str, torch.Tensor] | list[dict[str, torch.Tensor]]:
        backbone_outputs = branch.backbone(x, masks=masks, is_training=self.training)
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
    ) -> dict[str, Mapping[str, torch.Tensor] | dict[str, Mapping[str, torch.Tensor] | torch.Tensor]]:
        project_cls_tokens = mask_indices_list is None
        student_outputs = self._forward_branch(
            self.student,
            student_input,
            masks=student_masks,
            project_cls_tokens=project_cls_tokens,
            project_patch_tokens=project_student_patch_tokens and mask_indices_list is None,
        )
        if mask_indices_list is None:
            outputs: dict[str, Mapping[str, torch.Tensor] | dict[str, Mapping[str, torch.Tensor] | torch.Tensor]] = {
                "student": student_outputs
            }
        else:
            student_global = dict(student_outputs)
            student_global_cls, student_patch = self.project_global_cls_and_masked_patch_tokens(
                self.student,
                student_global["cls_tokens"],
                student_global["patch_tokens"],
                mask_indices_list,
                n_masked_patches=n_masked_patches,
            )
            structured_student_outputs: dict[str, Mapping[str, torch.Tensor] | torch.Tensor] = {
                "global": student_global,
                "global_cls_projections": student_global_cls,
                "global_masked_patch_projections": student_patch,
            }
            if "patch_token_support" in student_global:
                structured_student_outputs["global_masked_patch_support"] = self.select_masked_tokens(
                    student_global["patch_token_support"],
                    mask_indices_list=mask_indices_list,
                    n_masked_patches=n_masked_patches,
                )
            if local_student_input is not None:
                structured_student_outputs["local"] = self._forward_branch(
                    self.student,
                    local_student_input,
                    masks=None,
                    project_cls_tokens=True,
                )
            outputs = {"student": structured_student_outputs}

        if return_teacher:
            teacher_source = student_input if teacher_input is None else teacher_input
            with torch.no_grad():
                teacher_outputs = self._forward_branch(
                    self.teacher,
                    teacher_source,
                    masks=teacher_masks,
                    project_cls_tokens=project_cls_tokens,
                    project_patch_tokens=project_teacher_patch_tokens and mask_indices_list is None,
                )
                if mask_indices_list is None:
                    outputs["teacher"] = teacher_outputs
                else:
                    teacher_global = dict(teacher_outputs)
                    teacher_global_cls, teacher_patch = self.project_global_cls_and_masked_patch_tokens(
                        self.teacher,
                        teacher_global["cls_tokens"],
                        teacher_global["patch_tokens"],
                        mask_indices_list,
                        n_masked_patches=n_masked_patches,
                    )
                    teacher_structured_outputs: dict[str, Mapping[str, torch.Tensor] | torch.Tensor] = {
                        "global": teacher_global,
                        "global_cls_projections": teacher_global_cls,
                        "global_masked_patch_projections": teacher_patch,
                    }
                    if "patch_token_support" in teacher_global:
                        teacher_structured_outputs["global_masked_patch_support"] = self.select_masked_tokens(
                            teacher_global["patch_token_support"],
                            mask_indices_list=mask_indices_list,
                            n_masked_patches=n_masked_patches,
                        )
                    outputs["teacher"] = teacher_structured_outputs
        return outputs
