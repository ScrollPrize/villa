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


def _build_distance_aware_coarse_targets(
    target_coarse_ids: Tensor,
    supervision_mask: Tensor,
    *,
    coarse_grid_shape: tuple[int, int, int],
    radius: int,
    sigma: float,
) -> tuple[Tensor, Tensor]:
    gz, gy, gx = [int(v) for v in coarse_grid_shape]
    device = target_coarse_ids.device
    dtype = torch.float32
    offsets_1d = torch.arange(-int(radius), int(radius) + 1, device=device, dtype=dtype)
    offset_grid = torch.stack(torch.meshgrid(offsets_1d, offsets_1d, offsets_1d, indexing="ij"), dim=-1).reshape(-1, 3)
    k = int(offset_grid.shape[0])

    safe_ids = torch.where(supervision_mask, target_coarse_ids, torch.zeros_like(target_coarse_ids))
    safe_ids = safe_ids.to(torch.long)
    z = safe_ids // (gy * gx)
    rem = safe_ids % (gy * gx)
    y = rem // gx
    x = rem % gx
    gt_coords = torch.stack([z, y, x], dim=-1).to(dtype=dtype)

    neighbor_coords = gt_coords.unsqueeze(-2) + offset_grid.view(1, 1, k, 3)
    valid = supervision_mask.unsqueeze(-1).expand(-1, -1, k).clone()
    valid &= neighbor_coords[..., 0] >= 0
    valid &= neighbor_coords[..., 0] < gz
    valid &= neighbor_coords[..., 1] >= 0
    valid &= neighbor_coords[..., 1] < gy
    valid &= neighbor_coords[..., 2] >= 0
    valid &= neighbor_coords[..., 2] < gx

    dist2 = (offset_grid ** 2).sum(dim=-1)
    weights = torch.exp(-dist2 / (2.0 * float(sigma) * float(sigma))).view(1, 1, k).expand_as(valid.to(dtype))
    weights = torch.where(valid, weights, torch.zeros_like(weights))
    denom = weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    weights = weights / denom

    neighbor_coords_long = neighbor_coords.to(torch.long)
    neighbor_ids = (
        neighbor_coords_long[..., 0] * (gy * gx) +
        neighbor_coords_long[..., 1] * gx +
        neighbor_coords_long[..., 2]
    )
    neighbor_ids = torch.where(valid, neighbor_ids, torch.zeros_like(neighbor_ids))
    return neighbor_ids, weights


def _coarse_target_entropy(target_probs: Tensor, supervision_mask: Tensor) -> Tensor:
    entropy = -(target_probs * torch.log(target_probs.clamp(min=1e-8))).sum(dim=-1)
    return _masked_mean(entropy, supervision_mask)


def _hard_coarse_pointer_loss(outputs: dict, batch: dict) -> Tensor:
    logits = outputs["coarse_logits"]
    targets = batch["target_coarse_ids"]
    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=IGNORE_INDEX,
        reduction="none",
    ).reshape_as(targets)
    return _masked_mean(loss, batch["target_supervision_mask"])


def _coarse_pointer_loss(
    outputs: dict,
    batch: dict,
    *,
    distance_aware_enabled: bool,
    distance_aware_radius: int,
    distance_aware_sigma: float,
    distance_aware_loss_type: str,
) -> tuple[Tensor, Tensor]:
    if not bool(distance_aware_enabled):
        return _hard_coarse_pointer_loss(outputs, batch), outputs["coarse_logits"].new_zeros(())
    if str(distance_aware_loss_type) != "soft_ce":
        raise ValueError(f"Unsupported distance_aware_coarse_target_loss={distance_aware_loss_type!r}")

    logits = outputs["coarse_logits"]
    supervision_mask = batch["target_supervision_mask"]
    neighbor_ids, target_probs = _build_distance_aware_coarse_targets(
        batch["target_coarse_ids"],
        supervision_mask,
        coarse_grid_shape=tuple(int(v) for v in outputs["coarse_grid_shape"]),
        radius=int(distance_aware_radius),
        sigma=float(distance_aware_sigma),
    )
    log_probs = F.log_softmax(logits, dim=-1)
    gathered_log_probs = torch.gather(log_probs, dim=-1, index=neighbor_ids)
    per_token_loss = -(target_probs * gathered_log_probs).sum(dim=-1)
    coarse_loss = _masked_mean(per_token_loss, supervision_mask)
    coarse_target_entropy = _coarse_target_entropy(target_probs, supervision_mask)
    return coarse_loss, coarse_target_entropy


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


def _sequence_to_grid_torch(sequence: Tensor, *, grid_shape: tuple[int, int], direction: str) -> Tensor:
    h, w = int(grid_shape[0]), int(grid_shape[1])
    expected = int(h * w)
    if int(sequence.shape[0]) != expected:
        raise ValueError(f"sequence length {sequence.shape[0]} does not match grid_shape {grid_shape!r}")

    flat_to_seq = torch.empty((expected,), device=sequence.device, dtype=torch.long)
    cursor = 0
    if direction in {"left", "right"}:
        strip_order = range(w) if direction == "left" else range(w - 1, -1, -1)
        for col_idx in strip_order:
            for row_idx in range(h):
                flat_to_seq[row_idx * w + col_idx] = cursor
                cursor += 1
    elif direction in {"up", "down"}:
        strip_order = range(h) if direction == "up" else range(h - 1, -1, -1)
        for row_idx in strip_order:
            for col_idx in range(w):
                flat_to_seq[row_idx * w + col_idx] = cursor
                cursor += 1
    else:
        raise ValueError(f"unsupported direction {direction!r}")
    return sequence.index_select(0, flat_to_seq).reshape(h, w, *sequence.shape[1:])


def _quad_gram_entries(grid_xyz: Tensor) -> Tensor:
    u = grid_xyz[:-1, 1:, :] - grid_xyz[:-1, :-1, :]
    v = grid_xyz[1:, :-1, :] - grid_xyz[:-1, :-1, :]
    uu = (u * u).sum(dim=-1)
    uv = (u * v).sum(dim=-1)
    vv = (v * v).sum(dim=-1)
    return torch.stack([uu, uv, vv], dim=-1)


def _geometry_metric_loss(
    outputs: dict,
    batch: dict,
    *,
    loss_type: str,
    include_refine_residual: bool,
) -> Tensor:
    if str(loss_type) != "huber":
        raise ValueError(f"Unsupported geometry_metric_loss={loss_type!r}")

    pred_xyz_soft = outputs["pred_xyz_soft"]
    if not bool(include_refine_residual):
        pred_xyz_soft = pred_xyz_soft - outputs["pred_refine_residual"]

    target_xyz = batch["target_xyz"]
    target_valid_mask = batch["target_valid_mask"]
    target_lengths = batch["target_lengths"]
    losses = []
    for batch_idx, direction in enumerate(batch["direction"]):
        target_len = int(target_lengths[batch_idx].item())
        if target_len <= 0:
            continue
        grid_shape = tuple(int(v) for v in batch["target_grid_shape"][batch_idx].tolist())
        if min(grid_shape) < 2:
            continue

        pred_grid = _sequence_to_grid_torch(pred_xyz_soft[batch_idx, :target_len], grid_shape=grid_shape, direction=direction)
        target_grid = _sequence_to_grid_torch(target_xyz[batch_idx, :target_len], grid_shape=grid_shape, direction=direction)
        valid_grid = _sequence_to_grid_torch(
            target_valid_mask[batch_idx, :target_len],
            grid_shape=grid_shape,
            direction=direction,
        ).bool()
        quad_mask = valid_grid[:-1, :-1] & valid_grid[:-1, 1:] & valid_grid[1:, :-1] & valid_grid[1:, 1:]
        if not bool(quad_mask.any()):
            continue

        pred_gram = _quad_gram_entries(pred_grid)
        target_gram = _quad_gram_entries(target_grid)
        per_quad = F.smooth_l1_loss(pred_gram, target_gram, reduction="none").mean(dim=-1)
        losses.append(_masked_mean(per_quad, quad_mask))

    if not losses:
        return pred_xyz_soft.new_zeros(())
    return torch.stack(losses).mean()


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
        pred_grid = pred_xyz[batch_idx, :count].numpy()
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
    offset_loss_weight_active: float = 1.0,
    position_refine_weight_active: float = 0.0,
    position_refine_loss_type: str = "huber",
    geometry_metric_weight_active: float = 0.0,
    geometry_metric_loss_type: str = "huber",
    distance_aware_coarse_targets_enabled: bool = True,
    distance_aware_coarse_target_radius: int = 1,
    distance_aware_coarse_target_sigma: float = 1.0,
    distance_aware_coarse_target_loss: str = "soft_ce",
) -> dict[str, Tensor]:
    coarse_loss, coarse_target_entropy = _coarse_pointer_loss(
        outputs,
        batch,
        distance_aware_enabled=distance_aware_coarse_targets_enabled,
        distance_aware_radius=distance_aware_coarse_target_radius,
        distance_aware_sigma=distance_aware_coarse_target_sigma,
        distance_aware_loss_type=distance_aware_coarse_target_loss,
    )
    offset_loss = _offset_bin_loss(outputs, batch, offset_num_bins=offset_num_bins)
    stop_loss = _stop_loss(outputs, batch)
    total_loss = coarse_loss + float(offset_loss_weight_active) * offset_loss + stop_loss
    coarse_excess_nll = coarse_loss - coarse_target_entropy
    refine_loss = total_loss.new_zeros(())
    if float(position_refine_weight_active) > 0.0:
        refine_loss = _position_refine_loss(outputs, batch, loss_type=position_refine_loss_type)
        total_loss = total_loss + float(position_refine_weight_active) * refine_loss

    geometry_metric_loss = total_loss.new_zeros(())
    if float(geometry_metric_weight_active) > 0.0:
        geometry_metric_loss = _geometry_metric_loss(
            outputs,
            batch,
            loss_type=geometry_metric_loss_type,
            include_refine_residual=float(position_refine_weight_active) > 0.0,
        )
        total_loss = total_loss + float(geometry_metric_weight_active) * geometry_metric_loss

    occupancy_metric = total_loss.new_zeros(())
    if float(occupancy_loss_weight) > 0.0:
        # This metric is intentionally detached/non-differentiable; keep it out
        # of the optimized objective until a differentiable rasterizer exists.
        occupancy_metric = _occupancy_metric(outputs, batch)

    return {
        "loss": total_loss,
        "coarse_loss": coarse_loss,
        "coarse_target_entropy": coarse_target_entropy,
        "coarse_excess_nll": coarse_excess_nll,
        "offset_loss": offset_loss,
        "offset_loss_weight_active": total_loss.new_tensor(float(offset_loss_weight_active)),
        "stop_loss": stop_loss,
        "refine_loss": refine_loss,
        "refine_loss_weight_active": total_loss.new_tensor(float(position_refine_weight_active)),
        "geometry_metric_loss": geometry_metric_loss,
        "geometry_metric_weight_active": total_loss.new_tensor(float(geometry_metric_weight_active)),
        "occupancy_metric": occupancy_metric,
    }
