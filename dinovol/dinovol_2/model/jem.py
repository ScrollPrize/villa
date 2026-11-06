"""JEM student/teacher: CLS-free backbone, per-depth categorical heads, structural heads.

Categorical heads (one per alignment depth, with EMA teacher copies):
    D -> Linear(hidden) -> GELU -> Linear(hidden) -> GELU -> Linear(bottleneck)
      -> L2 normalize -> Linear(K, bias=False)
Structural heads (student only, one per structural target depth, all reading the
final student features): the same MLP to `structural_dim`, then L2 normalize.

Categorical inputs at every depth are the raw tap normalized once by the backbone's
shared patch LayerNorm (the final depth reuses the backbone's normalized output).
Structural targets are raw teacher taps (layer 0 = patch embedding) passed through a
non-affine LayerNorm and L2 normalization by the loss.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from dinovol_2.model.backbone import build_backbone
from dinovol_2.model.model import DINOHead, ema_update_, vit_parameter_groups


def _apply_head(head: nn.Module, tokens: torch.Tensor, use_checkpoint: bool) -> torch.Tensor:
    if use_checkpoint and torch.is_grad_enabled():
        return checkpoint(head, tokens, use_reentrant=False)
    return head(tokens)


class JemStudentTeacher(nn.Module):
    def __init__(self, backbone_config: Mapping[str, Any], heads: Mapping[str, Any]) -> None:
        super().__init__()
        self.heads_config = dict(heads)
        self.alignment_depths = tuple(int(d) for d in heads["alignment_depths"])
        self.structural_depths = tuple(int(d) for d in heads["structural_depths"])
        self.checkpoint_heads = bool(heads.get("checkpoint_heads", False))
        backbone = build_backbone(backbone_config)
        if backbone.num_class_tokens:
            raise ValueError("JEM uses a backbone without a class token (backbone.class_token=false)")
        if backbone.cls_norm is not None or backbone.local_cls_norm is not None:
            raise ValueError("JEM disables CLS-only output norms (untie_* = false)")
        depth = backbone.depth_in_blocks
        if self.alignment_depths[-1] != depth or any(d < 1 or d > depth for d in self.alignment_depths):
            raise ValueError(f"alignment depths must lie in [1, {depth}] and end at the final block, "
                             f"got {self.alignment_depths}")
        if any(d < 0 or d >= depth for d in self.structural_depths):
            raise ValueError(f"structural target depths must lie in [0, {depth}), got {self.structural_depths}")
        embed_dim = int(backbone.embed_dim)
        categorical = nn.ModuleList(
            DINOHead(embed_dim, int(heads["categories"]), hidden_dim=int(heads["hidden_dim"]),
                     bottleneck_dim=int(heads["bottleneck_dim"]), nlayers=int(heads["nlayers"]),
                     last_layer_type=str(heads["last_layer_type"]))
            for _ in self.alignment_depths
        )
        structural = nn.ModuleList(
            DINOHead(embed_dim, 0, hidden_dim=int(heads["hidden_dim"]),
                     bottleneck_dim=int(heads["structural_dim"]), nlayers=int(heads["nlayers"]),
                     last_layer_type="none")
            for _ in self.structural_depths
        )
        self.student = nn.ModuleDict({"backbone": backbone, "categorical_heads": categorical,
                                      "structural_heads": structural})
        self.teacher = nn.ModuleDict({"backbone": deepcopy(backbone), "categorical_heads": deepcopy(categorical)})
        self.synchronize_teacher_from_student()
        for parameter in self.teacher.parameters():
            parameter.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher.eval()  # no drop-path or other stochastic layers in the teacher
        return self

    def _teacher_pairs(self):
        student = dict(self.student.named_parameters())
        student_buffers = dict(self.student.named_buffers())
        pairs = [(parameter, student[name]) for name, parameter in self.teacher.named_parameters()]
        pairs += [(buffer, student_buffers[name]) for name, buffer in self.teacher.named_buffers()
                  if buffer.is_floating_point()]
        return pairs

    @torch.no_grad()
    def synchronize_teacher_from_student(self) -> None:
        for teacher, student in self._teacher_pairs():
            teacher.copy_(student)

    @torch.no_grad()
    def update_teacher(self, momentum: float) -> None:
        """Parameter (and floating buffer) EMA, including the categorical final matrices:
        W_teacher <- m W_teacher + (1 - m) W_student, with no row renormalization."""
        pairs = self._teacher_pairs()
        ema_update_([teacher.data for teacher, _ in pairs], [student.data for _, student in pairs], momentum)

    def get_params_groups(self, *, lr_decay_rate: float = 1.0, patch_embed_lr_mult: float = 1.0):
        backbone = self.student.backbone
        return vit_parameter_groups(self.student, num_layers=backbone.depth_in_blocks,
                                    chunked_blocks=bool(getattr(backbone, "chunked_blocks", False)),
                                    lr_decay_rate=lr_decay_rate, patch_embed_lr_mult=patch_embed_lr_mult)

    # ------------------------------------------------------------------ branches

    def _categorical_inputs(self, backbone, features) -> list[torch.Tensor]:
        """Shared patch LayerNorm applied once per raw tap; the final tap is already normalized."""
        final = self.alignment_depths[-1]
        return [features["x_norm_patchtokens"] if depth == final else backbone.norm(features["taps"][depth])
                for depth in self.alignment_depths]

    def student_forward(self, crops: torch.Tensor, masks: torch.Tensor, *, view_kind: str,
                        structural: bool = True) -> dict[str, Any]:
        """Student logits per alignment depth [B, N, K] and structural outputs per target depth [B, N, C]."""
        backbone = self.student.backbone
        taps = [d for d in self.alignment_depths[:-1]]
        # The processor already zero-filled masked cells with vc_reader before the stem.
        features = backbone.forward_features(crops, masks, view_kind=view_kind, taps=taps, mask_voxels=False)
        inputs = self._categorical_inputs(backbone, features)
        logits = {depth: _apply_head(head, tokens, self.checkpoint_heads)
                  for depth, head, tokens in zip(self.alignment_depths, self.student.categorical_heads, inputs)}
        outputs = {"logits": logits}
        if structural:
            final = features["x_norm_patchtokens"]
            outputs["structural"] = {
                depth: _apply_head(head, final, self.checkpoint_heads)
                for depth, head in zip(self.structural_depths, self.student.structural_heads)
            }
        return outputs

    def forward(self, global_crops, global_masks, local_crops, local_masks, *, structural: bool = True):
        """One DDP forward per source chunk covers every student parameter."""
        return {
            "global": self.student_forward(global_crops, global_masks, view_kind="global", structural=structural),
            "local": self.student_forward(local_crops, local_masks, view_kind="local", structural=structural),
        }

    @torch.no_grad()
    def teacher_forward(self, crops: torch.Tensor) -> dict[str, Any]:
        """Unmasked teacher: categorical logits per alignment depth and raw structural target taps."""
        backbone = self.teacher.backbone
        taps = sorted(set(self.alignment_depths[:-1]) | set(self.structural_depths))
        features = backbone.forward_features(crops, None, view_kind="global", taps=taps)
        inputs = self._categorical_inputs(backbone, features)
        return {
            "logits": {depth: head(tokens) for depth, head, tokens
                       in zip(self.alignment_depths, self.teacher.categorical_heads, inputs)},
            "structural": {depth: features["taps"][depth] for depth in self.structural_depths},
        }


def head_parameter_count(head: nn.Module) -> int:
    return sum(parameter.numel() for parameter in head.parameters())


def check_depths(depths: Sequence[int]) -> tuple[int, ...]:
    return tuple(sorted({int(depth) for depth in depths}))
