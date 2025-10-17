from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch

from .base_metric import BaseMetric


@dataclass(frozen=True)
class AffinityMetricConfig:
    """Configuration bundle for affinity evaluation."""

    apply_sigmoid: bool = True
    invert: bool = False
    eps: float = 1e-6
    name: str = "affinity"


class AffinityStatsMetric(BaseMetric):
    """
    Metric for Mutex Watershed affinity heads.

    Computes Sørensen–Dice on continuous affinity predictions (mirroring the
    training objective) together with a mean absolute error summary. All
    statistics are reported channel-wise and summarised with mean/min/max to
    make the metric readable with tens of channels.
    """

    def __init__(self, *, config: Optional[AffinityMetricConfig] = None) -> None:
        cfg = config or AffinityMetricConfig()
        super().__init__(cfg.name)
        self._cfg = cfg

    def compute(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        *,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        pred_tensor = self._prepare_tensor(pred)
        gt_tensor = self._prepare_tensor(gt, match_dtype=pred_tensor.dtype)

        if mask is not None:
            mask_tensor = self._prepare_tensor(mask, match_dtype=pred_tensor.dtype, allow_bool=True)
            mask_tensor = mask_tensor.clamp_min(0.0).clamp_max(1.0)
        else:
            mask_tensor = None

        if self._cfg.apply_sigmoid:
            pred_tensor = pred_tensor.sigmoid()

        if self._cfg.invert:
            pred_tensor = 1.0 - pred_tensor
            gt_tensor = 1.0 - gt_tensor

        # Ensure tensors share shape: [B, C, ...]
        if pred_tensor.shape != gt_tensor.shape:
            raise ValueError(
                f"Prediction and ground truth must share the same shape. "
                f"Got pred={tuple(pred_tensor.shape)}, gt={tuple(gt_tensor.shape)}."
            )

        if mask_tensor is not None:
            if mask_tensor.shape != pred_tensor.shape:
                raise ValueError(
                    "Mask must have the same shape as prediction/ground truth "
                    f"after broadcasting. Got mask={tuple(mask_tensor.shape)}, "
                    f"pred={tuple(pred_tensor.shape)}."
                )

        batch, channels = pred_tensor.shape[:2]
        flat_dim = batch, channels, -1

        pred_flat = pred_tensor.reshape(*flat_dim)
        gt_flat = gt_tensor.reshape(*flat_dim)

        if mask_tensor is not None:
            mask_flat = mask_tensor.reshape(*flat_dim)
            pred_flat = pred_flat * mask_flat
            gt_flat = gt_flat * mask_flat
        else:
            mask_flat = None

        dice_per_channel = self._dice_per_channel(pred_flat, gt_flat, mask_flat)
        mae_per_channel = self._mae_per_channel(pred_flat, gt_flat, mask_flat)
        coverage = mask_tensor.mean().item() if mask_tensor is not None else 1.0

        return {
            "dice_mean": dice_per_channel.mean().item(),
            "dice_median": dice_per_channel.median().item(),
            "dice_min": dice_per_channel.min().item(),
            "dice_max": dice_per_channel.max().item(),
            "dice_std": dice_per_channel.std(unbiased=False).item(),
            "mae_mean": mae_per_channel.mean().item(),
            "mask_coverage": coverage,
        }

    def _prepare_tensor(
        self,
        tensor: torch.Tensor,
        *,
        match_dtype: Optional[torch.dtype] = None,
        allow_bool: bool = False,
    ) -> torch.Tensor:
        if isinstance(tensor, (list, tuple)):
            tensor = tensor[0]

        if tensor.dtype == torch.bfloat16:
            tensor = tensor.float()
        elif tensor.dtype == torch.bool:
            if not allow_bool:
                tensor = tensor.to(torch.float32)
            else:
                tensor = tensor.to(torch.float32)

        if match_dtype is not None and tensor.dtype != match_dtype:
            tensor = tensor.to(match_dtype)

        if tensor.ndim < 2:
            raise ValueError(
                "Expected tensors with at least [batch, channels, ...] dimensions, "
                f"got tensor with shape {tuple(tensor.shape)}."
            )

        if tensor.device.type != "cpu":
            tensor = tensor.detach().cpu()
        else:
            tensor = tensor.detach()

        return tensor

    def _dice_per_channel(
        self,
        pred_flat: torch.Tensor,
        gt_flat: torch.Tensor,
        mask_flat: Optional[torch.Tensor],
    ) -> torch.Tensor:
        eps = self._cfg.eps
        intersection = (pred_flat * gt_flat).sum(dim=-1)
        pred_sq = (pred_flat * pred_flat).sum(dim=-1)
        gt_sq = (gt_flat * gt_flat).sum(dim=-1)

        dice = (2.0 * intersection + eps) / (pred_sq + gt_sq + eps)

        if mask_flat is not None:
            valid = mask_flat.sum(dim=-1) > 0
            dice = dice.masked_fill(~valid, float("nan"))

        return self._mean_ignore_nan(dice, dim=0)

    def _mae_per_channel(
        self,
        pred_flat: torch.Tensor,
        gt_flat: torch.Tensor,
        mask_flat: Optional[torch.Tensor],
    ) -> torch.Tensor:
        abs_diff = (pred_flat - gt_flat).abs()

        if mask_flat is not None:
            weights = mask_flat.sum(dim=-1)
            mae = abs_diff.sum(dim=-1) / weights.clamp_min(1.0)
            mae = mae.masked_fill(weights <= 0, float("nan"))
            return self._mean_ignore_nan(mae, dim=0)
        else:
            return abs_diff.mean(dim=-1).mean(dim=0)

    @staticmethod
    def _mean_ignore_nan(tensor: torch.Tensor, dim: int) -> torch.Tensor:
        if not torch.isnan(tensor).any():
            return tensor.mean(dim=dim)

        mask = ~torch.isnan(tensor)
        safe_sum = tensor.masked_fill(~mask, 0.0).sum(dim=dim)
        counts = mask.sum(dim=dim).clamp_min(1)
        return safe_sum / counts
