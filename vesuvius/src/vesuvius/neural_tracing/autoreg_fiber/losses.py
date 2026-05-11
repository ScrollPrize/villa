from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from vesuvius.neural_tracing.autoreg_fiber.serialization import IGNORE_INDEX


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    mask = mask.to(dtype=values.dtype)
    denom = mask.sum().clamp(min=1.0)
    safe_values = torch.where(mask > 0, torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0), torch.zeros_like(values))
    return safe_values.sum() / denom


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

    safe_ids = torch.where(supervision_mask, target_coarse_ids, torch.zeros_like(target_coarse_ids)).to(torch.long)
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
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    neighbor_coords_long = neighbor_coords.to(torch.long)
    neighbor_ids = (
        neighbor_coords_long[..., 0] * (gy * gx)
        + neighbor_coords_long[..., 1] * gx
        + neighbor_coords_long[..., 2]
    )
    neighbor_ids = torch.where(valid, neighbor_ids, torch.zeros_like(neighbor_ids))
    return neighbor_ids, weights


def _build_distance_aware_axis_targets(
    axis_target: Tensor,
    supervision_mask: Tensor,
    *,
    axis_size: int,
    radius: int,
    sigma: float,
) -> tuple[Tensor, Tensor]:
    device = axis_target.device
    dtype = torch.float32
    offsets = torch.arange(-int(radius), int(radius) + 1, device=device, dtype=torch.long)
    safe_target = torch.where(supervision_mask, axis_target, torch.zeros_like(axis_target)).to(torch.long)
    neighbor_ids = safe_target.unsqueeze(-1) + offsets.view(1, 1, -1)
    valid = supervision_mask.unsqueeze(-1).expand_as(neighbor_ids).clone()
    valid &= neighbor_ids >= 0
    valid &= neighbor_ids < int(axis_size)
    dist2 = offsets.to(dtype=dtype).square()
    weights = torch.exp(-dist2 / (2.0 * float(sigma) * float(sigma))).view(1, 1, -1)
    weights = weights.expand_as(valid.to(dtype=dtype))
    weights = torch.where(valid, weights, torch.zeros_like(weights))
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
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


def _hard_axis_pointer_loss(axis_logits: Tensor, axis_target: Tensor, supervision_mask: Tensor) -> Tensor:
    loss = F.cross_entropy(
        axis_logits.reshape(-1, axis_logits.shape[-1]),
        axis_target.reshape(-1),
        ignore_index=IGNORE_INDEX,
        reduction="none",
    ).reshape_as(axis_target)
    return _masked_mean(loss, supervision_mask)


def _unflatten_coarse_axis_ids(coarse_ids: Tensor, *, coarse_grid_shape: tuple[int, int, int]) -> dict[str, Tensor]:
    _, gy, gx = [int(v) for v in coarse_grid_shape]
    safe = torch.where(coarse_ids >= 0, coarse_ids, torch.zeros_like(coarse_ids)).to(torch.long)
    z = safe // (gy * gx)
    rem = safe % (gy * gx)
    y = rem // gx
    x = rem % gx
    ignore = torch.full_like(coarse_ids, IGNORE_INDEX)
    return {
        "z": torch.where(coarse_ids >= 0, z, ignore),
        "y": torch.where(coarse_ids >= 0, y, ignore),
        "x": torch.where(coarse_ids >= 0, x, ignore),
    }


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
    gathered_log_probs = torch.where(target_probs > 0, gathered_log_probs, torch.zeros_like(gathered_log_probs))
    per_token_loss = -(target_probs * gathered_log_probs).sum(dim=-1)
    coarse_loss = _masked_mean(per_token_loss, supervision_mask)
    coarse_target_entropy = _coarse_target_entropy(target_probs, supervision_mask)
    return coarse_loss, coarse_target_entropy


def _factorized_coarse_pointer_loss(
    outputs: dict,
    batch: dict,
    *,
    distance_aware_enabled: bool,
    distance_aware_radius: int,
    distance_aware_sigma: float,
    distance_aware_loss_type: str,
) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
    if outputs.get("coarse_axis_logits") is None:
        raise ValueError("axis_factorized coarse mode requires coarse_axis_logits")
    if bool(distance_aware_enabled) and str(distance_aware_loss_type) != "soft_ce":
        raise ValueError(f"Unsupported distance_aware_coarse_target_loss={distance_aware_loss_type!r}")

    supervision_mask = batch["target_supervision_mask"]
    coarse_grid_shape = tuple(int(v) for v in outputs["coarse_grid_shape"])
    axis_targets = _unflatten_coarse_axis_ids(batch["target_coarse_ids"], coarse_grid_shape=coarse_grid_shape)
    axis_sizes = {"z": int(coarse_grid_shape[0]), "y": int(coarse_grid_shape[1]), "x": int(coarse_grid_shape[2])}
    axis_losses: dict[str, Tensor] = {}
    axis_entropies: dict[str, Tensor] = {}

    for axis_name in ("z", "y", "x"):
        axis_logits = outputs["coarse_axis_logits"][axis_name]
        axis_target = axis_targets[axis_name]
        if not bool(distance_aware_enabled):
            axis_losses[axis_name] = _hard_axis_pointer_loss(axis_logits, axis_target, supervision_mask)
            axis_entropies[axis_name] = axis_logits.new_zeros(())
            continue
        neighbor_ids, target_probs = _build_distance_aware_axis_targets(
            axis_target,
            supervision_mask,
            axis_size=axis_sizes[axis_name],
            radius=int(distance_aware_radius),
            sigma=float(distance_aware_sigma),
        )
        log_probs = F.log_softmax(axis_logits, dim=-1)
        gathered_log_probs = torch.gather(log_probs, dim=-1, index=neighbor_ids)
        gathered_log_probs = torch.where(target_probs > 0, gathered_log_probs, torch.zeros_like(gathered_log_probs))
        per_token_loss = -(target_probs * gathered_log_probs).sum(dim=-1)
        axis_losses[axis_name] = _masked_mean(per_token_loss, supervision_mask)
        axis_entropies[axis_name] = _coarse_target_entropy(target_probs, supervision_mask)

    coarse_loss = torch.stack([axis_losses["z"], axis_losses["y"], axis_losses["x"]]).mean()
    coarse_target_entropy = torch.stack([axis_entropies["z"], axis_entropies["y"], axis_entropies["x"]]).mean()
    return coarse_loss, coarse_target_entropy, axis_losses


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
    loss = F.binary_cross_entropy_with_logits(outputs["stop_logits"], batch["target_stop"], reduction="none")
    return _masked_mean(loss, batch["target_supervision_mask"])


def _position_refine_loss(outputs: dict, batch: dict, *, loss_type: str) -> Tensor:
    if str(loss_type) != "huber":
        raise ValueError(f"Unsupported position_refine_loss={loss_type!r}")
    target_residual = batch["target_xyz"] - batch["target_bin_center_xyz"]
    pred_residual = outputs["pred_refine_residual"]
    per_token = F.smooth_l1_loss(pred_residual, target_residual, reduction="none").mean(dim=-1)
    return _masked_mean(per_token, batch["target_supervision_mask"])


def _prediction_finite_mask(pred_xyz_sequence: Tensor) -> Tensor:
    return torch.isfinite(pred_xyz_sequence).all(dim=-1)


def _soft_geometry_xyz(outputs: dict, *, include_refine_residual: bool) -> Tensor:
    if "pred_xyz_soft" in outputs:
        pred = outputs["pred_xyz_soft"]
    elif "pred_xyz_refined" in outputs:
        pred = outputs["pred_xyz_refined"]
    else:
        pred = outputs["pred_xyz"]
    if bool(include_refine_residual):
        return pred
    if "pred_refine_residual" not in outputs:
        return pred
    return pred - outputs["pred_refine_residual"]


def _xyz_huber_loss(pred_xyz_sequence: Tensor, batch: dict) -> Tensor:
    mask = batch["target_supervision_mask"] & _prediction_finite_mask(pred_xyz_sequence)
    per_token = F.smooth_l1_loss(pred_xyz_sequence, batch["target_xyz"], reduction="none").mean(dim=-1)
    return _masked_mean(per_token, mask)


def _segment_vector_huber_loss(pred_xyz_sequence: Tensor, batch: dict) -> Tensor:
    if pred_xyz_sequence.shape[1] <= 1:
        return pred_xyz_sequence.new_zeros(())
    pred_delta = pred_xyz_sequence[:, 1:, :] - pred_xyz_sequence[:, :-1, :]
    target_delta = batch["target_xyz"][:, 1:, :] - batch["target_xyz"][:, :-1, :]
    mask = batch["target_supervision_mask"][:, 1:] & batch["target_supervision_mask"][:, :-1]
    mask &= _prediction_finite_mask(pred_xyz_sequence[:, 1:, :])
    mask &= _prediction_finite_mask(pred_xyz_sequence[:, :-1, :])
    per_token = F.smooth_l1_loss(pred_delta, target_delta, reduction="none").mean(dim=-1)
    return _masked_mean(per_token, mask)


def _tube_radius_huber_loss(pred_xyz_sequence: Tensor, batch: dict) -> Tensor:
    target = batch["target_xyz"]
    if target.shape[1] < 2:
        return pred_xyz_sequence.new_zeros(())
    error = pred_xyz_sequence - target
    tangent = torch.zeros_like(target)
    tangent[:, 1:-1, :] = (target[:, 2:, :] - target[:, :-2, :]) * 0.5
    tangent[:, 0, :] = target[:, 1, :] - target[:, 0, :]
    tangent[:, -1, :] = target[:, -1, :] - target[:, -2, :]
    tangent_hat = tangent / tangent.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    parallel = (error * tangent_hat).sum(dim=-1, keepdim=True)
    perp = error - parallel * tangent_hat
    perp_mag = perp.norm(dim=-1)
    sup = batch["target_supervision_mask"]
    mask = sup & _prediction_finite_mask(pred_xyz_sequence)
    per_token = F.smooth_l1_loss(perp_mag, torch.zeros_like(perp_mag), reduction="none")
    return _masked_mean(per_token, mask)


def _straightness_huber_loss(pred_xyz_sequence: Tensor, batch: dict) -> Tensor:
    """Curvature-difference penalty (option C): penalize how the predicted
    second-difference deviates from the target's second-difference, so that
    matching a curved target is unbiased while wobble away from the target's
    smooth-curvature pattern is penalized. Config key remains
    ``straightness_loss_*`` for compatibility, but the regularizer no longer
    biases toward globally-straight predictions."""
    if pred_xyz_sequence.shape[1] <= 2:
        return pred_xyz_sequence.new_zeros(())
    target = batch["target_xyz"]
    pred_curvature = (
        pred_xyz_sequence[:, 2:, :]
        - 2.0 * pred_xyz_sequence[:, 1:-1, :]
        + pred_xyz_sequence[:, :-2, :]
    )
    target_curvature = (
        target[:, 2:, :]
        - 2.0 * target[:, 1:-1, :]
        + target[:, :-2, :]
    )
    diff = pred_curvature - target_curvature
    sup = batch["target_supervision_mask"]
    mask = sup[:, 2:] & sup[:, 1:-1] & sup[:, :-2]
    mask &= _prediction_finite_mask(pred_xyz_sequence[:, 2:, :])
    mask &= _prediction_finite_mask(pred_xyz_sequence[:, 1:-1, :])
    mask &= _prediction_finite_mask(pred_xyz_sequence[:, :-2, :])
    per_token = F.smooth_l1_loss(
        diff, torch.zeros_like(diff), reduction="none"
    ).mean(dim=-1)
    return _masked_mean(per_token, mask)


def _l1_xyz_metric(pred_xyz_sequence: Tensor, batch: dict) -> Tensor:
    mask = batch["target_supervision_mask"] & _prediction_finite_mask(pred_xyz_sequence)
    per_token = (pred_xyz_sequence - batch["target_xyz"]).abs().mean(dim=-1)
    return _masked_mean(per_token, mask)


def _coarse_accuracy_metrics(outputs: dict, batch: dict) -> dict[str, Tensor]:
    mask = batch["target_supervision_mask"]
    pred_coarse_ids = outputs["pred_coarse_ids"]
    target_coarse_ids = batch["target_coarse_ids"]
    coarse_grid_shape = tuple(int(v) for v in outputs["coarse_grid_shape"])
    pred_axis = outputs.get("pred_coarse_axis_ids") or _unflatten_coarse_axis_ids(
        pred_coarse_ids,
        coarse_grid_shape=coarse_grid_shape,
    )
    target_axis = _unflatten_coarse_axis_ids(target_coarse_ids, coarse_grid_shape=coarse_grid_shape)
    return {
        "coarse_exact_acc": _masked_mean((pred_coarse_ids == target_coarse_ids).to(dtype=torch.float32), mask),
        "coarse_axis_acc_z": _masked_mean((pred_axis["z"] == target_axis["z"]).to(dtype=torch.float32), mask),
        "coarse_axis_acc_y": _masked_mean((pred_axis["y"] == target_axis["y"]).to(dtype=torch.float32), mask),
        "coarse_axis_acc_x": _masked_mean((pred_axis["x"] == target_axis["x"]).to(dtype=torch.float32), mask),
    }


def _pred_oob_fraction(pred_xyz_sequence: Tensor, batch: dict) -> Tensor:
    if "volume" not in batch:
        return pred_xyz_sequence.new_zeros(())
    mask = batch.get("target_mask", batch["target_supervision_mask"]).to(torch.bool)
    crop_shape = tuple(int(v) for v in batch["volume"].shape[-3:])
    max_coord = torch.tensor(crop_shape, device=pred_xyz_sequence.device, dtype=pred_xyz_sequence.dtype) - 1e-4
    finite = _prediction_finite_mask(pred_xyz_sequence)
    oob = finite & (((pred_xyz_sequence < 0.0) | (pred_xyz_sequence > max_coord.view(1, 1, 3))).any(dim=-1))
    return _masked_mean(oob.to(dtype=pred_xyz_sequence.dtype), mask)


def compute_autoreg_fiber_losses(
    outputs: dict,
    batch: dict,
    *,
    offset_num_bins: tuple[int, int, int],
    offset_loss_weight_active: float = 1.0,
    position_refine_weight_active: float = 0.0,
    position_refine_loss_type: str = "huber",
    xyz_soft_loss_weight_active: float = 0.0,
    xyz_soft_loss_type: str = "huber",
    segment_vector_loss_weight_active: float = 0.0,
    segment_vector_loss_type: str = "huber",
    straightness_loss_weight_active: float = 0.0,
    straightness_loss_type: str = "huber",
    tube_radius_loss_weight_active: float = 0.0,
    tube_radius_loss_type: str = "huber",
    distance_aware_coarse_targets_enabled: bool = True,
    distance_aware_coarse_target_radius: int = 1,
    distance_aware_coarse_target_sigma: float = 1.0,
    distance_aware_coarse_target_loss: str = "soft_ce",
) -> dict[str, Tensor]:
    axis_loss_metrics = {
        "z": batch["target_xyz"].new_zeros(()),
        "y": batch["target_xyz"].new_zeros(()),
        "x": batch["target_xyz"].new_zeros(()),
    }
    if str(outputs.get("coarse_prediction_mode", "joint_pointer")) == "axis_factorized":
        coarse_loss, coarse_target_entropy, axis_loss_metrics = _factorized_coarse_pointer_loss(
            outputs,
            batch,
            distance_aware_enabled=distance_aware_coarse_targets_enabled,
            distance_aware_radius=distance_aware_coarse_target_radius,
            distance_aware_sigma=distance_aware_coarse_target_sigma,
            distance_aware_loss_type=distance_aware_coarse_target_loss,
        )
    else:
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
    include_refine_residual = float(position_refine_weight_active) > 0.0
    soft_xyz_for_loss = _soft_geometry_xyz(outputs, include_refine_residual=include_refine_residual)
    if str(xyz_soft_loss_type) != "huber":
        raise ValueError(f"Unsupported xyz_soft_loss={xyz_soft_loss_type!r}")
    xyz_soft_loss = _xyz_huber_loss(soft_xyz_for_loss, batch)
    refine_loss = total_loss.new_zeros(())
    if float(position_refine_weight_active) > 0.0:
        refine_loss = _position_refine_loss(outputs, batch, loss_type=position_refine_loss_type)
        total_loss = total_loss + float(position_refine_weight_active) * refine_loss
    if float(xyz_soft_loss_weight_active) > 0.0:
        total_loss = total_loss + float(xyz_soft_loss_weight_active) * xyz_soft_loss
    segment_vector_loss = total_loss.new_zeros(())
    if float(segment_vector_loss_weight_active) > 0.0:
        if str(segment_vector_loss_type) != "huber":
            raise ValueError(f"Unsupported segment_vector_loss={segment_vector_loss_type!r}")
        segment_vector_loss = _segment_vector_huber_loss(soft_xyz_for_loss, batch)
        total_loss = total_loss + float(segment_vector_loss_weight_active) * segment_vector_loss
    straightness_loss = total_loss.new_zeros(())
    if float(straightness_loss_weight_active) > 0.0:
        if str(straightness_loss_type) != "huber":
            raise ValueError(f"Unsupported straightness_loss={straightness_loss_type!r}")
        straightness_loss = _straightness_huber_loss(soft_xyz_for_loss, batch)
        total_loss = total_loss + float(straightness_loss_weight_active) * straightness_loss
    tube_radius_loss = total_loss.new_zeros(())
    if float(tube_radius_loss_weight_active) > 0.0:
        if str(tube_radius_loss_type) != "huber":
            raise ValueError(f"Unsupported tube_radius_loss={tube_radius_loss_type!r}")
        tube_radius_loss = _tube_radius_huber_loss(soft_xyz_for_loss, batch)
        total_loss = total_loss + float(tube_radius_loss_weight_active) * tube_radius_loss

    pred_xyz_refined = outputs.get("pred_xyz_refined", outputs["pred_xyz"])
    coarse_acc_metrics = _coarse_accuracy_metrics(outputs, batch)
    stop_pred = torch.sigmoid(outputs["stop_logits"]) >= 0.5
    stop_target = batch["target_stop"] >= 0.5
    stop_acc = _masked_mean((stop_pred == stop_target).to(dtype=torch.float32), batch["target_supervision_mask"])
    return {
        "loss": total_loss,
        "coarse_loss": coarse_loss,
        "coarse_target_entropy": coarse_target_entropy,
        "coarse_excess_nll": coarse_loss - coarse_target_entropy,
        "coarse_exact_acc": coarse_acc_metrics["coarse_exact_acc"],
        "coarse_axis_acc_z": coarse_acc_metrics["coarse_axis_acc_z"],
        "coarse_axis_acc_y": coarse_acc_metrics["coarse_axis_acc_y"],
        "coarse_axis_acc_x": coarse_acc_metrics["coarse_axis_acc_x"],
        "coarse_axis_loss_z": axis_loss_metrics["z"],
        "coarse_axis_loss_y": axis_loss_metrics["y"],
        "coarse_axis_loss_x": axis_loss_metrics["x"],
        "offset_loss": offset_loss,
        "offset_loss_weight_active": total_loss.new_tensor(float(offset_loss_weight_active)),
        "stop_loss": stop_loss,
        "stop_acc": stop_acc,
        "refine_loss": refine_loss,
        "refine_loss_weight_active": total_loss.new_tensor(float(position_refine_weight_active)),
        "xyz_soft_loss": xyz_soft_loss,
        "xyz_soft_loss_weight_active": total_loss.new_tensor(float(xyz_soft_loss_weight_active)),
        "xyz_l1_soft": _l1_xyz_metric(soft_xyz_for_loss.detach(), batch),
        "xyz_l1_refined": _l1_xyz_metric(pred_xyz_refined.detach(), batch),
        "segment_vector_loss": segment_vector_loss,
        "segment_vector_loss_weight_active": total_loss.new_tensor(float(segment_vector_loss_weight_active)),
        "straightness_loss": straightness_loss,
        "straightness_loss_weight_active": total_loss.new_tensor(float(straightness_loss_weight_active)),
        "tube_radius_loss": tube_radius_loss,
        "tube_radius_loss_weight_active": total_loss.new_tensor(float(tube_radius_loss_weight_active)),
        "pred_oob_fraction_refined": _pred_oob_fraction(pred_xyz_refined.detach(), batch),
    }


__all__ = [
    "compute_autoreg_fiber_losses",
]
