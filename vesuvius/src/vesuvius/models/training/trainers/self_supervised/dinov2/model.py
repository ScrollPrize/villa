from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from vesuvius.models.build.transformers.dinov2_eva import Eva, EvaWithChunking


_BACKBONE_DEFAULTS = {
    "input_channels": 1,
    "global_crops_size": (256, 256, 256),
    "local_crops_size": None,
    "embed_dim": 864,
    "patch_size": (8, 8, 8),
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
    "num_reg_tokens": 0,
    "grad_checkpointing": False,
    "block_chunks": 0,
}
_HEAD_DEFAULTS = {
    "hidden_dim": 2048,
    "bottleneck_dim": 256,
    "nlayers": 3,
    "use_bn": False,
    "norm_last_layer": True,
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


class DINOHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        nlayers: int = 3,
        use_bn: bool = False,
        norm_last_layer: bool = True,
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

        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1.0)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        return self.last_layer(x)


class DinoVitStudentTeacher(nn.Module):
    """Minimal 3D DINO-style student/teacher wrapper around the local EVA backbone."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.config = dict(config)
        self.ibot_separate_head = bool(self.config.get("ibot_separate_head", False))
        student_backbone = self._build_backbone(self.config)
        teacher_backbone = deepcopy(student_backbone)

        student_modules = {
            "backbone": student_backbone,
            "dino_head": self._build_head("dino"),
        }
        teacher_modules = {
            "backbone": teacher_backbone,
            "dino_head": self._build_head("dino"),
        }
        if self.ibot_separate_head:
            student_modules["ibot_head"] = self._build_head("ibot", fallback_prefix="dino")
            teacher_modules["ibot_head"] = self._build_head("ibot", fallback_prefix="dino")

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
        kwargs = {
            suffix: _config_value(
                self.config,
                f"{prefix}_head_{suffix}",
                default,
                fallback_key=f"{fallback_prefix}_head_{suffix}" if fallback_prefix else None,
            )
            for suffix, default in _HEAD_DEFAULTS.items()
        }
        out_dim = _config_value(
            self.config,
            f"{prefix}_out_dim",
            self.config.get("dino_out_dim", 65536),
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
        missing, unexpected = self.student.load_state_dict(state_dict, strict=False)
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
    def select_masked_patch_tokens(
        patch_tokens: torch.Tensor,
        mask_indices_list: torch.Tensor,
        n_masked_patches: Optional[int] = None,
    ) -> torch.Tensor:
        masked_tokens = torch.index_select(
            patch_tokens.flatten(0, 1),
            dim=0,
            index=mask_indices_list,
        )
        if n_masked_patches is not None:
            masked_tokens = masked_tokens[:n_masked_patches]
        return masked_tokens

    @staticmethod
    def _patch_head(branch: nn.ModuleDict) -> nn.Module:
        return branch.ibot_head if "ibot_head" in branch else branch.dino_head

    def project_patch_tokens(self, branch: nn.ModuleDict, patch_tokens: torch.Tensor) -> torch.Tensor:
        return self._apply_head(self._patch_head(branch), patch_tokens)

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

    def _format_branch_outputs(
        self,
        branch: nn.ModuleDict,
        backbone_outputs: Mapping[str, torch.Tensor],
        project_patch_tokens: bool = False,
    ) -> dict[str, torch.Tensor]:
        cls_tokens = backbone_outputs["x_norm_clstoken"]
        if cls_tokens is None:
            raise RuntimeError("DINO requires a backbone that returns class tokens.")

        outputs: dict[str, torch.Tensor] = {
            "cls_tokens": cls_tokens,
            "cls_projections": self._apply_head(branch.dino_head, cls_tokens),
            "patch_tokens": backbone_outputs["x_norm_patchtokens"],
        }
        if project_patch_tokens:
            outputs["patch_projections"] = self.project_patch_tokens(branch, backbone_outputs["x_norm_patchtokens"])
        return outputs

    def _forward_branch(
        self,
        branch: nn.ModuleDict,
        x: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
        project_patch_tokens: bool = False,
    ) -> Mapping[str, torch.Tensor] | list[dict[str, torch.Tensor]]:
        backbone_outputs = branch.backbone(x, masks=masks, is_training=self.training)
        if isinstance(backbone_outputs, list):
            return [
                self._format_branch_outputs(branch, output, project_patch_tokens=project_patch_tokens)
                for output in backbone_outputs
            ]
        return self._format_branch_outputs(branch, backbone_outputs, project_patch_tokens=project_patch_tokens)

    def forward(
        self,
        student_input: torch.Tensor,
        teacher_input: Optional[torch.Tensor] = None,
        student_masks: Optional[torch.Tensor] = None,
        teacher_masks: Optional[torch.Tensor] = None,
        *,
        return_teacher: bool = True,
        project_student_patch_tokens: bool = False,
        project_teacher_patch_tokens: bool = False,
    ) -> dict[str, Mapping[str, torch.Tensor]]:
        outputs: dict[str, Mapping[str, torch.Tensor]] = {
            "student": self._forward_branch(
                self.student,
                student_input,
                masks=student_masks,
                project_patch_tokens=project_student_patch_tokens,
            )
        }

        if return_teacher:
            teacher_source = student_input if teacher_input is None else teacher_input
            with torch.no_grad():
                outputs["teacher"] = self._forward_branch(
                    self.teacher,
                    teacher_source,
                    masks=teacher_masks,
                    project_patch_tokens=project_teacher_patch_tokens,
                )
        return outputs
