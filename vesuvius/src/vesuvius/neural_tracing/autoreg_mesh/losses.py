from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from vesuvius.neural_tracing.autoreg_mesh.serialization import IGNORE_INDEX
from vesuvius.neural_tracing.datasets.common import voxelize_surface_grid


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    mask = mask.to(dtype=values.dtype)
    denom = mask.sum().clamp(min=1.0)
    return (values * mask).sum() / denom


def _coarse_pointer_loss(outputs: dict, batch: dict) -> Tensor:
    logits = outputs["coarse_logits"]
    targets = batch["target_coarse_ids"]
    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=IGNORE_INDEX,
        reduction="none",
    ).reshape_as(targets)
    return _masked_mean(loss, batch["target_supervision_mask"])


def _offset_bin_loss(outputs: dict, batch: dict, offset_num_bins: tuple[int, int, int]) -> Tensor:
    logits = outputs["offset_logits"]
    targets = batch["target_offset_bins"]
    mask = batch["target_supervision_mask"]
    total = logits.new_zeros(())
    for axis, bins in enumerate(offset_num_bins):
        axis_logits = logits[:, :, axis, :bins]
        axis_targets = targets[:, :, axis]
        axis_loss = F.cross_entropy(
            axis_logits.reshape(-1, bins),
            axis_targets.reshape(-1),
            ignore_index=IGNORE_INDEX,
            reduction="none",
        ).reshape_as(axis_targets)
        total = total + _masked_mean(axis_loss, mask)
    return total / float(len(offset_num_bins))


def _stop_loss(outputs: dict, batch: dict) -> Tensor:
    logits = outputs["stop_logits"]
    targets = batch["target_stop"]
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return _masked_mean(loss, batch["target_supervision_mask"])


def _position_refine_loss(outputs: dict, batch: dict, *, loss_type: str) -> Tensor:
    if str(loss_type) != "huber":
        raise ValueError(f"Unsupported position_refine_loss={loss_type!r}")
    target_residual = batch["target_xyz"] - batch["target_bin_center_xyz"]
    pred_residual = outputs["pred_refine_residual"]
    per_token = F.smooth_l1_loss(pred_residual, target_residual, reduction="none").mean(dim=-1)
    return _masked_mean(per_token, batch["target_supervision_mask"])


def _occupancy_metric(outputs: dict, batch: dict) -> Tensor:
    pred_xyz = outputs.get("pred_xyz_refined", outputs["pred_xyz"]).detach().cpu()
    target_mask = batch["target_supervision_mask"].detach().cpu()
    volume = batch["volume"]
    device = volume.device
    losses = []
    for batch_idx in range(pred_xyz.shape[0]):
        count = int(target_mask[batch_idx].sum().item())
        if count <= 0:
            continue
        grid_shape = tuple(int(v) for v in batch["target_grid_shape"][batch_idx].tolist())
        pred_grid = outputs["pred_xyz"][batch_idx, :count].detach().cpu().numpy()
        pred_grid = pred_grid.reshape(grid_shape[0], grid_shape[1], 3)
        target_grid = batch["target_grid_local"][batch_idx].detach().cpu().numpy()
        crop_shape = tuple(int(v) for v in volume.shape[-3:])
        pred_vox = torch.from_numpy(voxelize_surface_grid(pred_grid.astype("float32"), crop_shape)).to(device=device)
        target_vox = torch.from_numpy(voxelize_surface_grid(target_grid.astype("float32"), crop_shape)).to(device=device)
        losses.append(F.binary_cross_entropy(pred_vox.clamp(1e-6, 1.0 - 1e-6), target_vox))
    if not losses:
        return torch.zeros((), device=device)
    return torch.stack(losses).mean()


def compute_autoreg_mesh_losses(
    outputs: dict,
    batch: dict,
    *,
    offset_num_bins: tuple[int, int, int],
    occupancy_loss_weight: float = 0.0,
    position_refine_weight_active: float = 0.0,
    position_refine_loss_type: str = "huber",
) -> dict[str, Tensor]:
    coarse_loss = _coarse_pointer_loss(outputs, batch)
    offset_loss = _offset_bin_loss(outputs, batch, offset_num_bins=offset_num_bins)
    stop_loss = _stop_loss(outputs, batch)
    total_loss = coarse_loss + offset_loss + stop_loss
    refine_loss = total_loss.new_zeros(())
    if float(position_refine_weight_active) > 0.0:
        refine_loss = _position_refine_loss(outputs, batch, loss_type=position_refine_loss_type)
        total_loss = total_loss + float(position_refine_weight_active) * refine_loss

    occupancy_metric = total_loss.new_zeros(())
    if float(occupancy_loss_weight) > 0.0:
        # This metric is intentionally detached/non-differentiable; keep it out
        # of the optimized objective until a differentiable rasterizer exists.
        occupancy_metric = _occupancy_metric(outputs, batch)

    return {
        "loss": total_loss,
        "coarse_loss": coarse_loss,
        "offset_loss": offset_loss,
        "stop_loss": stop_loss,
        "refine_loss": refine_loss,
        "refine_loss_weight_active": total_loss.new_tensor(float(position_refine_weight_active)),
        "occupancy_metric": occupancy_metric,
    }
