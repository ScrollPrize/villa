"""Losses tailored for surface-frame regression."""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


class SurfaceFrameMultiTermLoss(nn.Module):
    """Weighted combination of directional, frame-alignment, and orthogonality losses.

    The loss operates on 9-channel tensors that encode a 3×3 surface frame per voxel:
    the first three channels correspond to ``t_u`` (x/y/z components), the next three
    to ``t_v``, and the final three to ``n``. All reductions are masked so that only
    voxels with valid labels contribute to the objective.
    """

    def __init__(
        self,
        *,
        lambda_direction: float = 1.0,
        lambda_frame: float = 1.0,
        lambda_orthogonality: float = 0.1,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if lambda_direction < 0 or lambda_frame < 0 or lambda_orthogonality < 0:
            raise ValueError("SurfaceFrameMultiTermLoss weights must be non-negative.")

        self.lambda_direction = float(lambda_direction)
        self.lambda_frame = float(lambda_frame)
        self.lambda_orthogonality = float(lambda_orthogonality)
        self.eps = float(eps)

        self.register_buffer("_identity", torch.eye(3, dtype=torch.float32), persistent=False)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if prediction.shape != target.shape:
            raise ValueError(
                f"Prediction/target shape mismatch: {prediction.shape} vs {target.shape}"
            )
        if prediction.size(1) != 9:
            raise ValueError("Surface-frame tensors must have 9 channels (3 vectors × 3 components).")

        bsz, _, *spatial = prediction.shape
        spatial_size = math.prod(spatial) if spatial else 1

        pred_flat = prediction.view(bsz, 9, spatial_size)
        target_flat = target.view(bsz, 9, spatial_size)

        mask_flat = self._build_mask(target_flat, mask)
        valid_voxels = mask_flat.sum().item()
        if valid_voxels <= 0:
            raise RuntimeError("Surface frame loss received an empty mask; no labeled voxels present.")

        pred_frame = pred_flat.view(bsz, 3, 3, spatial_size)
        target_frame = target_flat.view(bsz, 3, 3, spatial_size)

        pred_norm = F.normalize(pred_frame, p=2, dim=2, eps=self.eps)
        target_norm = F.normalize(target_frame, p=2, dim=2, eps=self.eps)

        direction_loss = self._direction_term(pred_norm, target_norm, mask_flat)
        frame_loss = self._frame_alignment_term(pred_norm, target_norm, mask_flat)
        orthogonality_loss = self._orthogonality_term(pred_norm, mask_flat)

        total_loss = (
            self.lambda_direction * direction_loss
            + self.lambda_frame * frame_loss
            + self.lambda_orthogonality * orthogonality_loss
        )

        components = {
            "direction": direction_loss.detach(),
            "frame": frame_loss.detach(),
            "orthogonality": orthogonality_loss.detach(),
        }

        return total_loss, components

    # ------------------------------------------------------------------------------------------
    # Internal helpers

    def _build_mask(
        self,
        target_flat: torch.Tensor,
        supplied_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        bsz = target_flat.size(0)
        spatial_size = target_flat.size(-1)

        if supplied_mask is None:
            mask = (target_flat.abs().sum(dim=1, keepdim=True) > self.eps).float()
        else:
            mask = supplied_mask.float()
            if mask.dim() == target_flat.dim() - 1:
                mask = mask.unsqueeze(1)
            if mask.size(1) == 1:
                mask = mask.expand(-1, target_flat.size(1), *mask.shape[2:])
            mask = mask.view(bsz, target_flat.size(1), spatial_size)
            mask = mask.max(dim=1, keepdim=True)[0]

        return mask.view(bsz, spatial_size)

    def _direction_term(
        self,
        pred_norm: torch.Tensor,
        target_norm: torch.Tensor,
        mask_flat: torch.Tensor,
    ) -> torch.Tensor:
        # pred_norm/target_norm: (B, 3, 3, S)
        cos = (pred_norm * target_norm).sum(dim=2)  # (B, 3, S)
        dir_err = 1.0 - cos
        weighted = dir_err * mask_flat.unsqueeze(1)
        denom = (mask_flat.sum() * 3.0).clamp_min(self.eps)
        return weighted.sum() / denom

    def _frame_alignment_term(
        self,
        pred_norm: torch.Tensor,
        target_norm: torch.Tensor,
        mask_flat: torch.Tensor,
    ) -> torch.Tensor:
        # Reorder to (B, S, 3, 3) where last dim indexes frame vectors
        pred_mats = pred_norm.permute(0, 3, 2, 1)  # (B, S, 3, 3)
        target_mats = target_norm.permute(0, 3, 2, 1)

        valid_mask = mask_flat > 0.5  # (B, S)
        if not torch.any(valid_mask):
            raise RuntimeError("Surface frame loss received an empty valid mask after preprocessing.")

        pred_valid = pred_mats[valid_mask]
        target_valid = target_mats[valid_mask]

        identity = self._identity.to(pred_norm.device, pred_norm.dtype)
        alignment = torch.matmul(pred_valid.transpose(1, 2), target_valid)
        delta = alignment - identity
        return delta.pow(2).sum(dim=(1, 2)).mean()

    def _orthogonality_term(
        self,
        pred_norm: torch.Tensor,
        mask_flat: torch.Tensor,
    ) -> torch.Tensor:
        # pred_norm: (B, 3, 3, S)
        u = pred_norm[:, 0].permute(0, 2, 1)  # (B, S, 3)
        v = pred_norm[:, 1].permute(0, 2, 1)
        n_vec = pred_norm[:, 2].permute(0, 2, 1)

        cross = torch.cross(u, v, dim=2)
        cross_mismatch = (cross - n_vec).pow(2).sum(dim=2)

        uv_dot = (u * v).sum(dim=2).pow(2)
        un_dot = (u * n_vec).sum(dim=2).pow(2)
        vn_dot = (v * n_vec).sum(dim=2).pow(2)

        ortho = cross_mismatch + uv_dot + un_dot + vn_dot
        denom = mask_flat.sum().clamp_min(self.eps)
        return (ortho * mask_flat).sum() / denom


__all__ = ["SurfaceFrameMultiTermLoss"]
