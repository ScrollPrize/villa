from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from vesuvius.neural_tracing.autoreg_mesh.serialization import IGNORE_INDEX
from vesuvius.neural_tracing.datasets.common import voxelize_surface_grid


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


def _build_distance_aware_axis_targets(
    target_axis_ids: Tensor,
    supervision_mask: Tensor,
    *,
    axis_size: int,
    radius: int,
    sigma: float,
) -> tuple[Tensor, Tensor]:
    device = target_axis_ids.device
    dtype = torch.float32
    offsets = torch.arange(-int(radius), int(radius) + 1, device=device, dtype=dtype)
    k = int(offsets.shape[0])
    safe_ids = torch.where(supervision_mask, target_axis_ids, torch.zeros_like(target_axis_ids)).to(torch.long)
    neighbor_ids = safe_ids.unsqueeze(-1) + offsets.to(dtype=torch.long).view(1, 1, k)
    valid = supervision_mask.unsqueeze(-1).expand(-1, -1, k).clone()
    valid &= neighbor_ids >= 0
    valid &= neighbor_ids < int(axis_size)
    weights = torch.exp(-(offsets ** 2) / (2.0 * float(sigma) * float(sigma))).view(1, 1, k).expand_as(valid.to(dtype))
    weights = torch.where(valid, weights, torch.zeros_like(weights))
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    neighbor_ids = torch.where(valid, neighbor_ids, torch.zeros_like(neighbor_ids))
    return neighbor_ids, weights


def _coarse_target_entropy(target_probs: Tensor, supervision_mask: Tensor) -> Tensor:
    entropy = -(target_probs * torch.log(target_probs.clamp(min=1e-8))).sum(dim=-1)
    return _masked_mean(entropy, supervision_mask)


def _unflatten_coarse_axis_ids(target_coarse_ids: Tensor, *, coarse_grid_shape: tuple[int, int, int]) -> dict[str, Tensor]:
    _gz, gy, gx = [int(v) for v in coarse_grid_shape]
    coarse = target_coarse_ids.clamp(min=0)
    z = coarse // (gy * gx)
    rem = coarse % (gy * gx)
    y = rem // gx
    x = rem % gx
    return {"z": z, "y": y, "x": x}


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


def _hard_axis_pointer_loss(axis_logits: Tensor, axis_targets: Tensor, supervision_mask: Tensor) -> Tensor:
    axis_loss = F.cross_entropy(
        axis_logits.reshape(-1, axis_logits.shape[-1]),
        axis_targets.reshape(-1),
        ignore_index=IGNORE_INDEX,
        reduction="none",
    ).reshape_as(axis_targets)
    return _masked_mean(axis_loss, supervision_mask)


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
    constraint_mask = outputs.get("coarse_constraint_joint_valid_mask")
    if constraint_mask is not None:
        neighbor_allowed = torch.gather(constraint_mask.to(torch.bool), dim=-1, index=neighbor_ids)
        target_probs = torch.where(neighbor_allowed, target_probs, torch.zeros_like(target_probs))
        target_probs = target_probs / target_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
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
    axis_sizes = {
        "z": int(coarse_grid_shape[0]),
        "y": int(coarse_grid_shape[1]),
        "x": int(coarse_grid_shape[2]),
    }
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
        axis_valid_masks = outputs.get("coarse_constraint_axis_valid_masks")
        if axis_valid_masks is not None and axis_valid_masks.get(axis_name) is not None:
            neighbor_allowed = torch.gather(axis_valid_masks[axis_name].to(torch.bool), dim=-1, index=neighbor_ids)
            target_probs = torch.where(neighbor_allowed, target_probs, torch.zeros_like(target_probs))
            target_probs = target_probs / target_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        log_probs = F.log_softmax(axis_logits, dim=-1)
        gathered_log_probs = torch.gather(log_probs, dim=-1, index=neighbor_ids)
        gathered_log_probs = torch.where(target_probs > 0, gathered_log_probs, torch.zeros_like(gathered_log_probs))
        per_token_loss = -(target_probs * gathered_log_probs).sum(dim=-1)
        axis_losses[axis_name] = _masked_mean(per_token_loss, supervision_mask)
        axis_entropies[axis_name] = _coarse_target_entropy(target_probs, supervision_mask)

    coarse_loss = torch.stack([axis_losses["z"], axis_losses["y"], axis_losses["x"]]).mean()
    coarse_target_entropy = torch.stack([axis_entropies["z"], axis_entropies["y"], axis_entropies["x"]]).mean()
    return coarse_loss, coarse_target_entropy, axis_losses


def _coarse_accuracy_metrics(outputs: dict, batch: dict) -> dict[str, Tensor]:
    if "pred_coarse_ids" not in outputs:
        zeros = batch["target_xyz"].new_zeros(())
        return {
            "coarse_exact_acc": zeros,
            "coarse_axis_acc_z": zeros,
            "coarse_axis_acc_y": zeros,
            "coarse_axis_acc_x": zeros,
        }
    mask = batch["target_supervision_mask"]
    pred_coarse_ids = outputs["pred_coarse_ids"]
    target_coarse_ids = batch["target_coarse_ids"]
    exact_acc = _masked_mean((pred_coarse_ids == target_coarse_ids).to(dtype=torch.float32), mask)
    coarse_grid_shape = tuple(int(v) for v in outputs["coarse_grid_shape"])
    target_axis = _unflatten_coarse_axis_ids(target_coarse_ids, coarse_grid_shape=coarse_grid_shape)
    pred_axis = outputs["pred_coarse_axis_ids"]
    return {
        "coarse_exact_acc": exact_acc,
        "coarse_axis_acc_z": _masked_mean((pred_axis["z"] == target_axis["z"]).to(dtype=torch.float32), mask),
        "coarse_axis_acc_y": _masked_mean((pred_axis["y"] == target_axis["y"]).to(dtype=torch.float32), mask),
        "coarse_axis_acc_x": _masked_mean((pred_axis["x"] == target_axis["x"]).to(dtype=torch.float32), mask),
    }


def _offset_bin_loss(
    outputs: dict,
    batch: dict,
    offset_num_bins: tuple[int, int, int],
    *,
    distance_aware_enabled: bool = False,
    distance_aware_radius: int = 1,
    distance_aware_sigma: float = 0.75,
) -> Tensor:
    logits = outputs["offset_logits"]
    targets = batch["target_offset_bins"]
    mask = batch["target_supervision_mask"]
    total = logits.new_zeros(())
    for axis, bins in enumerate(offset_num_bins):
        axis_logits = logits[:, :, axis, :bins]
        axis_targets = targets[:, :, axis]
        if not bool(distance_aware_enabled):
            axis_loss = F.cross_entropy(
                axis_logits.reshape(-1, bins),
                axis_targets.reshape(-1),
                ignore_index=IGNORE_INDEX,
                reduction="none",
            ).reshape_as(axis_targets)
            total = total + _masked_mean(axis_loss, mask)
        else:
            neighbor_ids, target_probs = _build_distance_aware_axis_targets(
                axis_targets,
                mask,
                axis_size=int(bins),
                radius=int(distance_aware_radius),
                sigma=float(distance_aware_sigma),
            )
            log_probs = F.log_softmax(axis_logits, dim=-1)
            gathered_log_probs = torch.gather(log_probs, dim=-1, index=neighbor_ids)
            gathered_log_probs = torch.where(target_probs > 0, gathered_log_probs, torch.zeros_like(gathered_log_probs))
            per_token_loss = -(target_probs * gathered_log_probs).sum(dim=-1)
            total = total + _masked_mean(per_token_loss, mask)
    return total / float(len(offset_num_bins))


def _offset_accuracy_metrics(outputs: dict, batch: dict, offset_num_bins: tuple[int, int, int]) -> dict[str, Tensor]:
    logits = outputs["offset_logits"]
    targets = batch["target_offset_bins"]
    mask = batch["target_supervision_mask"]
    pred_bins = []
    bin_errors = []
    for axis, bins in enumerate(offset_num_bins):
        pred = logits[:, :, axis, :bins].argmax(dim=-1)
        pred_bins.append(pred)
        bin_errors.append((pred - targets[:, :, axis]).abs().to(dtype=torch.float32))
    exact_match = torch.ones_like(mask, dtype=torch.bool)
    for axis in range(len(offset_num_bins)):
        exact_match = exact_match & (pred_bins[axis] == targets[:, :, axis])
    mean_bin_error = torch.stack(bin_errors).mean(dim=0)
    return {
        "offset_exact_acc": _masked_mean(exact_match.to(dtype=torch.float32), mask),
        "offset_mean_bin_error": _masked_mean(mean_bin_error, mask),
    }


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


def _soft_geometry_xyz(outputs: dict, *, include_refine_residual: bool) -> Tensor:
    if "pred_xyz_soft" in outputs:
        pred_xyz_soft = outputs["pred_xyz_soft"]
    elif "pred_xyz_refined" in outputs:
        pred_xyz_soft = outputs["pred_xyz_refined"]
    else:
        pred_xyz_soft = outputs["pred_xyz"]
    if bool(include_refine_residual):
        return pred_xyz_soft
    if "pred_refine_residual" not in outputs:
        return pred_xyz_soft
    return pred_xyz_soft - outputs["pred_refine_residual"]


def _geometry_batch_available(batch: dict) -> bool:
    return (
        "target_lengths" in batch and
        "target_grid_shape" in batch and
        "target_valid_mask" in batch and
        "target_grid_local" in batch and
        "conditioning_grid_local" in batch and
        "direction" in batch
    )


def _as_grid_tensor(value, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    if torch.is_tensor(value):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(value, device=device, dtype=dtype)


def _prediction_finite_mask(pred_xyz_sequence: Tensor) -> Tensor:
    return torch.isfinite(pred_xyz_sequence).all(dim=-1)


def _sequence_token_mask(pred_xyz_sequence: Tensor, batch: dict) -> Tensor:
    if "target_mask" in batch:
        return batch["target_mask"].to(device=pred_xyz_sequence.device, dtype=torch.bool)
    return torch.ones(pred_xyz_sequence.shape[:2], device=pred_xyz_sequence.device, dtype=torch.bool)


def _crop_max_coord(pred_xyz_sequence: Tensor, batch: dict) -> Tensor:
    if "volume" not in batch:
        raise KeyError("batch does not contain volume")
    crop_shape = tuple(int(v) for v in batch["volume"].shape[-3:])
    return torch.tensor(crop_shape, device=pred_xyz_sequence.device, dtype=pred_xyz_sequence.dtype) - 1e-4


def _invalid_vertex_fraction_from_sequence(pred_xyz_sequence: Tensor, batch: dict) -> Tensor:
    mask = _sequence_token_mask(pred_xyz_sequence, batch)
    invalid = ~_prediction_finite_mask(pred_xyz_sequence)
    return _masked_mean(invalid.to(dtype=pred_xyz_sequence.dtype), mask)


def _pred_oob_fraction_from_sequence(pred_xyz_sequence: Tensor, batch: dict) -> Tensor:
    if "volume" not in batch:
        return pred_xyz_sequence.new_zeros(())
    mask = _sequence_token_mask(pred_xyz_sequence, batch)
    finite = _prediction_finite_mask(pred_xyz_sequence)
    max_coord = _crop_max_coord(pred_xyz_sequence, batch)
    oob = finite & (((pred_xyz_sequence < 0.0) | (pred_xyz_sequence > max_coord.view(1, 1, 3))).any(dim=-1))
    return _masked_mean(oob.to(dtype=pred_xyz_sequence.dtype), mask)


def _boundary_touch_fraction_from_sequence(pred_xyz_sequence: Tensor, batch: dict) -> Tensor:
    if "volume" not in batch:
        return pred_xyz_sequence.new_zeros(())
    mask = _sequence_token_mask(pred_xyz_sequence, batch)
    finite = _prediction_finite_mask(pred_xyz_sequence)
    max_coord = _crop_max_coord(pred_xyz_sequence, batch)
    touches = finite & (((pred_xyz_sequence <= 0.0) | (pred_xyz_sequence >= max_coord.view(1, 1, 3))).any(dim=-1))
    sample_mask = mask.any(dim=-1)
    sample_touch = torch.where(sample_mask, touches.any(dim=-1), torch.zeros_like(sample_mask))
    return _masked_mean(sample_touch.to(dtype=pred_xyz_sequence.dtype), sample_mask)


def _l1_xyz_metric(pred_xyz_sequence: Tensor, batch: dict) -> Tensor:
    if "target_xyz" not in batch or "target_supervision_mask" not in batch:
        return pred_xyz_sequence.new_zeros(())
    mask = batch["target_supervision_mask"] & _prediction_finite_mask(pred_xyz_sequence)
    per_token = (pred_xyz_sequence - batch["target_xyz"]).abs().mean(dim=-1)
    return _masked_mean(per_token, mask)


def _boundary_loss(outputs: dict, batch: dict) -> Tensor:
    if "volume" not in batch:
        return outputs.get("pred_xyz_refined", outputs["pred_xyz"]).new_zeros(())
    pred_xyz = outputs.get("pred_xyz_refined", outputs["pred_xyz"])
    mask = batch.get("target_supervision_mask", _sequence_token_mask(pred_xyz, batch)) & _prediction_finite_mask(pred_xyz)
    max_coord = _crop_max_coord(pred_xyz, batch)
    per_axis = F.softplus(-pred_xyz) + F.softplus(pred_xyz - max_coord.view(1, 1, 3))
    per_token = per_axis.sum(dim=-1)
    return _masked_mean(per_token, mask)


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


def _merge_grids_torch(cond_grid: Tensor, cont_grid: Tensor, *, direction: str) -> Tensor:
    if direction == "left":
        return torch.cat([cond_grid, cont_grid], dim=1)
    if direction == "right":
        return torch.cat([cont_grid, cond_grid], dim=1)
    if direction == "up":
        return torch.cat([cond_grid, cont_grid], dim=0)
    if direction == "down":
        return torch.cat([cont_grid, cond_grid], dim=0)
    raise ValueError(f"unsupported direction {direction!r}")


def _merge_region_masks_torch(cond_mask: Tensor, cont_mask: Tensor, *, direction: str) -> Tensor:
    if direction == "left":
        return torch.cat([cond_mask, cont_mask], dim=1)
    if direction == "right":
        return torch.cat([cont_mask, cond_mask], dim=1)
    if direction == "up":
        return torch.cat([cond_mask, cont_mask], dim=0)
    if direction == "down":
        return torch.cat([cont_mask, cond_mask], dim=0)
    raise ValueError(f"unsupported direction {direction!r}")


def _iter_geometry_examples(pred_xyz_sequence: Tensor, batch: dict):
    if not _geometry_batch_available(batch):
        return
    for batch_idx, direction in enumerate(batch["direction"]):
        target_len = int(batch["target_lengths"][batch_idx].item())
        if target_len <= 0:
            continue
        grid_shape = tuple(int(v) for v in batch["target_grid_shape"][batch_idx].tolist())
        pred_grid = _sequence_to_grid_torch(pred_xyz_sequence[batch_idx, :target_len], grid_shape=grid_shape, direction=direction)
        target_grid = _as_grid_tensor(
            batch["target_grid_local"][batch_idx],
            device=pred_grid.device,
            dtype=pred_grid.dtype,
        )
        cond_grid = _as_grid_tensor(
            batch["conditioning_grid_local"][batch_idx],
            device=pred_grid.device,
            dtype=pred_grid.dtype,
        )
        target_valid_grid = _sequence_to_grid_torch(
            batch["target_valid_mask"][batch_idx, :target_len],
            grid_shape=grid_shape,
            direction=direction,
        ).bool()
        yield {
            "batch_idx": batch_idx,
            "direction": str(direction),
            "grid_shape": grid_shape,
            "pred_grid": pred_grid,
            "target_grid": target_grid,
            "conditioning_grid": cond_grid,
            "target_valid_grid": target_valid_grid,
        }


def _paired_seam_bands(example: dict, *, band_width: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    direction = str(example["direction"])
    pred_grid = example["pred_grid"]
    target_grid = example["target_grid"]
    cond_grid = example["conditioning_grid"]
    target_valid_grid = example["target_valid_grid"]
    cond_valid = torch.isfinite(cond_grid).all(dim=-1)
    pred_valid = torch.isfinite(pred_grid).all(dim=-1)
    band = int(band_width)

    if direction == "left":
        band = min(band, int(cond_grid.shape[1]), int(pred_grid.shape[1]))
        cond_band = torch.flip(cond_grid[:, -band:, :], dims=(1,))
        pred_band = pred_grid[:, :band, :]
        target_band = target_grid[:, :band, :]
        valid_band = torch.flip(cond_valid[:, -band:], dims=(1,)) & pred_valid[:, :band] & target_valid_grid[:, :band]
    elif direction == "right":
        band = min(band, int(cond_grid.shape[1]), int(pred_grid.shape[1]))
        cond_band = cond_grid[:, :band, :]
        pred_band = torch.flip(pred_grid[:, -band:, :], dims=(1,))
        target_band = torch.flip(target_grid[:, -band:, :], dims=(1,))
        valid_band = cond_valid[:, :band] & torch.flip(pred_valid[:, -band:], dims=(1,)) & torch.flip(target_valid_grid[:, -band:], dims=(1,))
    elif direction == "up":
        band = min(band, int(cond_grid.shape[0]), int(pred_grid.shape[0]))
        cond_band = torch.flip(cond_grid[-band:, :, :], dims=(0,))
        pred_band = pred_grid[:band, :, :]
        target_band = target_grid[:band, :, :]
        valid_band = torch.flip(cond_valid[-band:, :], dims=(0,)) & pred_valid[:band, :] & target_valid_grid[:band, :]
    elif direction == "down":
        band = min(band, int(cond_grid.shape[0]), int(pred_grid.shape[0]))
        cond_band = cond_grid[:band, :, :]
        pred_band = torch.flip(pred_grid[-band:, :, :], dims=(0,))
        target_band = torch.flip(target_grid[-band:, :, :], dims=(0,))
        valid_band = cond_valid[:band, :] & torch.flip(pred_valid[-band:, :], dims=(0,)) & torch.flip(target_valid_grid[-band:, :], dims=(0,))
    else:
        raise ValueError(f"unsupported direction {direction!r}")
    return cond_band, pred_band, target_band, valid_band


def _first_strip_wrong_side_rate_from_sequence(pred_xyz_sequence: Tensor, batch: dict) -> Tensor:
    if not _geometry_batch_available(batch):
        return pred_xyz_sequence.new_zeros(())
    values = []
    for example in _iter_geometry_examples(pred_xyz_sequence, batch):
        direction = str(example["direction"])
        pred_grid = example["pred_grid"]
        cond_grid = example["conditioning_grid"]
        pred_valid = torch.isfinite(pred_grid).all(dim=-1)
        cond_valid = torch.isfinite(cond_grid).all(dim=-1)
        if direction == "left":
            frontier = cond_grid[:, -1, :]
            interior = cond_grid[:, -2, :] if int(cond_grid.shape[1]) > 1 else torch.full_like(frontier, float("nan"))
            pred_first = pred_grid[:, 0, :]
            valid = cond_valid[:, -1] & pred_valid[:, 0]
            if int(cond_grid.shape[1]) > 1:
                valid &= cond_valid[:, -2]
            default_dir = frontier.new_tensor([0.0, 0.0, 1.0])
        elif direction == "right":
            frontier = cond_grid[:, 0, :]
            interior = cond_grid[:, 1, :] if int(cond_grid.shape[1]) > 1 else torch.full_like(frontier, float("nan"))
            pred_first = pred_grid[:, -1, :]
            valid = cond_valid[:, 0] & pred_valid[:, -1]
            if int(cond_grid.shape[1]) > 1:
                valid &= cond_valid[:, 1]
            default_dir = frontier.new_tensor([0.0, 0.0, -1.0])
        elif direction == "up":
            frontier = cond_grid[-1, :, :]
            interior = cond_grid[-2, :, :] if int(cond_grid.shape[0]) > 1 else torch.full_like(frontier, float("nan"))
            pred_first = pred_grid[0, :, :]
            valid = cond_valid[-1, :] & pred_valid[0, :]
            if int(cond_grid.shape[0]) > 1:
                valid &= cond_valid[-2, :]
            default_dir = frontier.new_tensor([0.0, 1.0, 0.0])
        elif direction == "down":
            frontier = cond_grid[0, :, :]
            interior = cond_grid[1, :, :] if int(cond_grid.shape[0]) > 1 else torch.full_like(frontier, float("nan"))
            pred_first = pred_grid[-1, :, :]
            valid = cond_valid[0, :] & pred_valid[-1, :]
            if int(cond_grid.shape[0]) > 1:
                valid &= cond_valid[1, :]
            default_dir = frontier.new_tensor([0.0, -1.0, 0.0])
        else:
            raise ValueError(f"unsupported direction {direction!r}")

        outward = frontier - interior
        step = torch.linalg.norm(outward, dim=-1)
        dir_unit = torch.where(
            (step > 1e-6).unsqueeze(-1),
            outward / step.clamp(min=1e-6).unsqueeze(-1),
            default_dir.view(1, 3).expand_as(frontier),
        )
        step = torch.where(step > 1e-6, step, torch.ones_like(step))
        displacement = pred_first - frontier
        signed = (displacement * dir_unit).sum(dim=-1)
        wrong = signed < (-0.1 * step)
        if bool(valid.any()):
            values.append(_masked_mean(wrong.to(dtype=pred_xyz_sequence.dtype), valid))
    if not values:
        return pred_xyz_sequence.new_zeros(())
    return torch.stack(values).mean()


def _seam_edge_error_from_sequence(pred_xyz_sequence: Tensor, batch: dict, *, band_width: int) -> Tensor:
    if not _geometry_batch_available(batch):
        return pred_xyz_sequence.new_zeros(())
    losses = []
    for example in _iter_geometry_examples(pred_xyz_sequence, batch):
        cond_band, pred_band, target_band, valid_band = _paired_seam_bands(example, band_width=int(band_width))
        edge_pred = pred_band - cond_band
        edge_target = target_band - cond_band
        per_vertex = (edge_pred - edge_target).abs().mean(dim=-1)
        if bool(valid_band.any()):
            losses.append(_masked_mean(per_vertex, valid_band))
    if not losses:
        return pred_xyz_sequence.new_zeros(())
    return torch.stack(losses).mean()


def _seam_edge_loss_from_sequence(pred_xyz_sequence: Tensor, batch: dict, *, band_width: int, loss_type: str) -> Tensor:
    if str(loss_type) != "edge_huber":
        raise ValueError(f"Unsupported seam_loss={loss_type!r}")
    if not _geometry_batch_available(batch):
        return pred_xyz_sequence.new_zeros(())
    losses = []
    for example in _iter_geometry_examples(pred_xyz_sequence, batch):
        cond_band, pred_band, target_band, valid_band = _paired_seam_bands(example, band_width=int(band_width))
        edge_pred = pred_band - cond_band
        edge_target = target_band - cond_band
        per_vertex = F.smooth_l1_loss(edge_pred, edge_target, reduction="none").mean(dim=-1)
        if bool(valid_band.any()):
            losses.append(_masked_mean(per_vertex, valid_band))
    if not losses:
        return pred_xyz_sequence.new_zeros(())
    return torch.stack(losses).mean()


def _triangle_det_ratio(
    pred_p0: Tensor,
    pred_p1: Tensor,
    pred_p2: Tensor,
    target_p0: Tensor,
    target_p1: Tensor,
    target_p2: Tensor,
    *,
    eps: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    target_e1 = target_p1 - target_p0
    target_e2 = target_p2 - target_p0
    pred_e1 = pred_p1 - pred_p0
    pred_e2 = pred_p2 - pred_p0

    len1 = torch.linalg.norm(target_e1, dim=-1)
    u = target_e1 / len1.clamp(min=eps).unsqueeze(-1)
    proj = (target_e2 * u).sum(dim=-1)
    v_temp = target_e2 - proj.unsqueeze(-1) * u
    len2 = torch.linalg.norm(v_temp, dim=-1)
    v = v_temp / len2.clamp(min=eps).unsqueeze(-1)

    p11 = (pred_e1 * u).sum(dim=-1)
    p21 = (pred_e1 * v).sum(dim=-1)
    p12 = (pred_e2 * u).sum(dim=-1)
    p22 = (pred_e2 * v).sum(dim=-1)
    det_p = p11 * p22 - p12 * p21
    det_q = len1 * len2
    valid = (len1 > eps) & (len2 > eps)
    return det_p / det_q.clamp(min=eps), valid


def _triangle_det_ratio_from_sequence(pred_xyz_sequence: Tensor, batch: dict) -> tuple[list[Tensor], list[Tensor]]:
    dets: list[Tensor] = []
    masks: list[Tensor] = []
    if not _geometry_batch_available(batch):
        return dets, masks
    for example in _iter_geometry_examples(pred_xyz_sequence, batch):
        direction = str(example["direction"])
        pred_grid = example["pred_grid"]
        target_grid = example["target_grid"]
        cond_grid = example["conditioning_grid"]
        target_valid_grid = example["target_valid_grid"] & torch.isfinite(target_grid).all(dim=-1)
        cond_valid = torch.isfinite(cond_grid).all(dim=-1)
        pred_valid = torch.isfinite(pred_grid).all(dim=-1)

        full_pred = _merge_grids_torch(cond_grid, pred_grid, direction=direction)
        full_target = _merge_grids_torch(cond_grid, target_grid, direction=direction)
        full_valid = _merge_region_masks_torch(cond_valid, pred_valid & target_valid_grid, direction=direction)
        pred_region = _merge_region_masks_torch(
            torch.zeros_like(cond_valid, dtype=torch.bool),
            torch.ones_like(target_valid_grid, dtype=torch.bool),
            direction=direction,
        )

        p00 = full_pred[:-1, :-1, :]
        p01 = full_pred[:-1, 1:, :]
        p10 = full_pred[1:, :-1, :]
        p11 = full_pred[1:, 1:, :]
        t00 = full_target[:-1, :-1, :]
        t01 = full_target[:-1, 1:, :]
        t10 = full_target[1:, :-1, :]
        t11 = full_target[1:, 1:, :]

        valid_a = full_valid[:-1, :-1] & full_valid[:-1, 1:] & full_valid[1:, :-1]
        valid_b = full_valid[1:, 1:] & full_valid[1:, :-1] & full_valid[:-1, 1:]
        region_a = pred_region[:-1, :-1] | pred_region[:-1, 1:] | pred_region[1:, :-1]
        region_b = pred_region[1:, 1:] | pred_region[1:, :-1] | pred_region[:-1, 1:]

        det_a, nondeg_a = _triangle_det_ratio(p00, p01, p10, t00, t01, t10)
        det_b, nondeg_b = _triangle_det_ratio(p11, p10, p01, t11, t10, t01)
        dets.extend([det_a, det_b])
        masks.extend([valid_a & region_a & nondeg_a, valid_b & region_b & nondeg_b])
    return dets, masks


def _triangle_flip_rate_from_sequence(pred_xyz_sequence: Tensor, batch: dict) -> Tensor:
    dets, masks = _triangle_det_ratio_from_sequence(pred_xyz_sequence, batch)
    if not dets:
        return pred_xyz_sequence.new_zeros(())
    values = []
    for det, mask in zip(dets, masks, strict=True):
        if bool(mask.any()):
            values.append(_masked_mean((det <= 0.0).to(dtype=det.dtype), mask))
    if not values:
        return pred_xyz_sequence.new_zeros(())
    return torch.stack(values).mean()


def _triangle_barrier_loss_from_sequence(pred_xyz_sequence: Tensor, batch: dict, *, margin: float) -> Tensor:
    dets, masks = _triangle_det_ratio_from_sequence(pred_xyz_sequence, batch)
    if not dets:
        return pred_xyz_sequence.new_zeros(())
    values = []
    for det, mask in zip(dets, masks, strict=True):
        if bool(mask.any()):
            values.append(_masked_mean(F.softplus(float(margin) - det), mask))
    if not values:
        return pred_xyz_sequence.new_zeros(())
    return torch.stack(values).mean()


def _quad_gram_entries(grid_xyz: Tensor) -> Tensor:
    u = grid_xyz[:-1, 1:, :] - grid_xyz[:-1, :-1, :]
    v = grid_xyz[1:, :-1, :] - grid_xyz[:-1, :-1, :]
    uu = (u * u).sum(dim=-1)
    uv = (u * v).sum(dim=-1)
    vv = (v * v).sum(dim=-1)
    return torch.stack([uu, uv, vv], dim=-1)


def _geometry_metric_loss_from_sequence(pred_xyz_sequence: Tensor, batch: dict, *, loss_type: str) -> Tensor:
    if str(loss_type) != "huber":
        raise ValueError(f"Unsupported geometry_metric_loss={loss_type!r}")
    if not _geometry_batch_available(batch):
        return pred_xyz_sequence.new_zeros(())

    losses = []
    for example in _iter_geometry_examples(pred_xyz_sequence, batch):
        if min(example["grid_shape"]) < 2:
            continue
        pred_grid = example["pred_grid"]
        target_grid = example["target_grid"]
        valid_grid = example["target_valid_grid"] & torch.isfinite(pred_grid).all(dim=-1) & torch.isfinite(target_grid).all(dim=-1)
        quad_mask = valid_grid[:-1, :-1] & valid_grid[:-1, 1:] & valid_grid[1:, :-1] & valid_grid[1:, 1:]
        if not bool(quad_mask.any()):
            continue

        pred_gram = _quad_gram_entries(pred_grid)
        target_gram = _quad_gram_entries(target_grid)
        per_quad = F.smooth_l1_loss(pred_gram, target_gram, reduction="none").mean(dim=-1)
        losses.append(_masked_mean(per_quad, quad_mask))

    if not losses:
        return pred_xyz_sequence.new_zeros(())
    return torch.stack(losses).mean()


def _geometry_metric_loss(
    outputs: dict,
    batch: dict,
    *,
    loss_type: str,
    include_refine_residual: bool,
) -> Tensor:
    return _geometry_metric_loss_from_sequence(
        _soft_geometry_xyz(outputs, include_refine_residual=include_refine_residual),
        batch,
        loss_type=loss_type,
    )


def _triangle_gram_matrix(p0: Tensor, p1: Tensor, p2: Tensor) -> Tensor:
    e1 = p1 - p0
    e2 = p2 - p0
    g11 = (e1 * e1).sum(dim=-1)
    g12 = (e1 * e2).sum(dim=-1)
    g22 = (e2 * e2).sum(dim=-1)
    row0 = torch.stack([g11, g12], dim=-1)
    row1 = torch.stack([g12, g22], dim=-1)
    return torch.stack([row0, row1], dim=-2)


def _regularize_2x2_gram(gram: Tensor, *, eps: float) -> Tensor:
    gram = 0.5 * (gram + gram.transpose(-1, -2))
    gram = torch.nan_to_num(gram, nan=0.0, posinf=0.0, neginf=0.0)
    eye = torch.eye(2, device=gram.device, dtype=gram.dtype)
    return gram + float(eps) * eye


def _inverse_2x2(matrix: Tensor, *, det_floor: float) -> Tensor:
    a = matrix[..., 0, 0]
    b = matrix[..., 0, 1]
    c = matrix[..., 1, 0]
    d = matrix[..., 1, 1]
    det = a * d - b * c
    safe_det = det.clamp(min=float(det_floor))
    inv = torch.stack(
        [
            torch.stack([d / safe_det, -b / safe_det], dim=-1),
            torch.stack([-c / safe_det, a / safe_det], dim=-1),
        ],
        dim=-2,
    )
    return torch.nan_to_num(inv, nan=0.0, posinf=0.0, neginf=0.0)


def _symmetric_dirichlet_from_gram(pred_gram: Tensor, target_gram: Tensor, *, eps: float = 1e-4) -> Tensor:
    pred_gram = pred_gram.float()
    target_gram = target_gram.float()
    pred_reg = _regularize_2x2_gram(pred_gram, eps=eps)
    target_reg = _regularize_2x2_gram(target_gram, eps=eps)
    det_floor = float(eps) * float(eps)
    target_inv = _inverse_2x2(target_reg, det_floor=det_floor)
    pred_inv = _inverse_2x2(pred_reg, det_floor=det_floor)
    forward = target_inv @ pred_reg
    backward = pred_inv @ target_reg
    forward_trace = torch.diagonal(forward, dim1=-2, dim2=-1).sum(dim=-1)
    backward_trace = torch.diagonal(backward, dim1=-2, dim2=-1).sum(dim=-1)
    energy = 0.25 * (forward_trace + backward_trace) - 1.0
    return torch.clamp_min(torch.nan_to_num(energy, nan=0.0, posinf=1e6, neginf=0.0), 0.0)


def _geometry_sd_loss(
    outputs: dict,
    batch: dict,
    *,
    include_refine_residual: bool,
) -> Tensor:
    return _geometry_sd_loss_from_sequence(
        _soft_geometry_xyz(outputs, include_refine_residual=include_refine_residual),
        batch,
    )


def _geometry_sd_loss_from_sequence(pred_xyz_sequence: Tensor, batch: dict) -> Tensor:
    if not _geometry_batch_available(batch):
        return pred_xyz_sequence.new_zeros(())
    losses = []
    for example in _iter_geometry_examples(pred_xyz_sequence, batch):
        if min(example["grid_shape"]) < 2:
            continue
        pred_grid = example["pred_grid"]
        target_grid = example["target_grid"]
        valid_grid = example["target_valid_grid"] & torch.isfinite(pred_grid).all(dim=-1) & torch.isfinite(target_grid).all(dim=-1)

        p00 = pred_grid[:-1, :-1, :]
        p01 = pred_grid[:-1, 1:, :]
        p10 = pred_grid[1:, :-1, :]
        p11 = pred_grid[1:, 1:, :]
        t00 = target_grid[:-1, :-1, :]
        t01 = target_grid[:-1, 1:, :]
        t10 = target_grid[1:, :-1, :]
        t11 = target_grid[1:, 1:, :]

        tri_a_mask = valid_grid[:-1, :-1] & valid_grid[:-1, 1:] & valid_grid[1:, :-1]
        tri_b_mask = valid_grid[1:, 1:] & valid_grid[1:, :-1] & valid_grid[:-1, 1:]

        if bool(tri_a_mask.any()):
            pred_gram_a = _triangle_gram_matrix(p00, p01, p10)
            target_gram_a = _triangle_gram_matrix(t00, t01, t10)
            losses.append(_masked_mean(_symmetric_dirichlet_from_gram(pred_gram_a, target_gram_a), tri_a_mask))
        if bool(tri_b_mask.any()):
            pred_gram_b = _triangle_gram_matrix(p11, p10, p01)
            target_gram_b = _triangle_gram_matrix(t11, t10, t01)
            losses.append(_masked_mean(_symmetric_dirichlet_from_gram(pred_gram_b, target_gram_b), tri_b_mask))

    if not losses:
        return pred_xyz_sequence.new_zeros(())
    return torch.stack(losses).mean()


def _occupancy_metric(outputs: dict, batch: dict) -> Tensor:
    pred_xyz = outputs.get("pred_xyz_refined", outputs["pred_xyz"]).detach().cpu()
    volume = batch["volume"]
    device = volume.device
    losses = []
    for batch_idx in range(pred_xyz.shape[0]):
        grid_shape = tuple(int(v) for v in batch["target_grid_shape"][batch_idx].tolist())
        count = int(grid_shape[0] * grid_shape[1])
        if count <= 0:
            continue
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
    xyz_soft_loss_weight_active: float = 0.0,
    xyz_soft_loss_type: str = "huber",
    seam_loss_weight_active: float = 0.0,
    seam_loss_type: str = "edge_huber",
    seam_band_width: int = 1,
    triangle_barrier_weight_active: float = 0.0,
    triangle_barrier_margin: float = 0.05,
    boundary_loss_weight_active: float = 0.0,
    geometry_metric_weight_active: float = 0.0,
    geometry_metric_loss_type: str = "huber",
    geometry_sd_weight_active: float = 0.0,
    distance_aware_coarse_targets_enabled: bool = True,
    distance_aware_coarse_target_radius: int = 1,
    distance_aware_coarse_target_sigma: float = 1.0,
    distance_aware_coarse_target_loss: str = "soft_ce",
    distance_aware_offset_targets_enabled: bool = False,
    distance_aware_offset_target_radius: int = 1,
    distance_aware_offset_target_sigma: float = 0.75,
) -> dict[str, Tensor]:
    coarse_prediction_mode = str(outputs.get("coarse_prediction_mode", "joint_pointer"))
    axis_loss_metrics = {
        "z": batch["target_xyz"].new_zeros(()),
        "y": batch["target_xyz"].new_zeros(()),
        "x": batch["target_xyz"].new_zeros(()),
    }
    if coarse_prediction_mode == "axis_factorized":
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
    offset_loss = _offset_bin_loss(
        outputs, batch, offset_num_bins=offset_num_bins,
        distance_aware_enabled=bool(distance_aware_offset_targets_enabled),
        distance_aware_radius=int(distance_aware_offset_target_radius),
        distance_aware_sigma=float(distance_aware_offset_target_sigma),
    )
    stop_loss = _stop_loss(outputs, batch)
    total_loss = coarse_loss + float(offset_loss_weight_active) * offset_loss + stop_loss
    coarse_excess_nll = coarse_loss - coarse_target_entropy
    include_refine_residual = float(position_refine_weight_active) > 0.0
    soft_xyz_for_loss = _soft_geometry_xyz(outputs, include_refine_residual=include_refine_residual)
    if str(xyz_soft_loss_type) != "huber":
        raise ValueError(f"Unsupported xyz_soft_loss={xyz_soft_loss_type!r}")
    xyz_soft_mask = batch["target_supervision_mask"] & _prediction_finite_mask(soft_xyz_for_loss)
    per_token_xyz_soft = F.smooth_l1_loss(soft_xyz_for_loss, batch["target_xyz"], reduction="none").mean(dim=-1)
    xyz_soft_loss = _masked_mean(per_token_xyz_soft, xyz_soft_mask)
    refine_loss = total_loss.new_zeros(())
    if float(position_refine_weight_active) > 0.0:
        refine_loss = _position_refine_loss(outputs, batch, loss_type=position_refine_loss_type)
        total_loss = total_loss + float(position_refine_weight_active) * refine_loss

    if float(xyz_soft_loss_weight_active) > 0.0:
        total_loss = total_loss + float(xyz_soft_loss_weight_active) * xyz_soft_loss

    seam_loss = total_loss.new_zeros(())
    seam_edge_error_soft = _seam_edge_error_from_sequence(soft_xyz_for_loss.detach(), batch, band_width=int(seam_band_width))
    if float(seam_loss_weight_active) > 0.0:
        seam_loss = _seam_edge_loss_from_sequence(
            soft_xyz_for_loss,
            batch,
            band_width=int(seam_band_width),
            loss_type=seam_loss_type,
        )
        total_loss = total_loss + float(seam_loss_weight_active) * seam_loss

    triangle_barrier_loss = total_loss.new_zeros(())
    triangle_flip_rate_soft = _triangle_flip_rate_from_sequence(soft_xyz_for_loss.detach(), batch)
    if float(triangle_barrier_weight_active) > 0.0:
        triangle_barrier_loss = _triangle_barrier_loss_from_sequence(
            soft_xyz_for_loss,
            batch,
            margin=float(triangle_barrier_margin),
        )
        total_loss = total_loss + float(triangle_barrier_weight_active) * triangle_barrier_loss

    boundary_loss = _boundary_loss(outputs, batch)

    geometry_metric_loss = total_loss.new_zeros(())
    if float(geometry_metric_weight_active) > 0.0:
        geometry_metric_loss = _geometry_metric_loss(
            outputs,
            batch,
            loss_type=geometry_metric_loss_type,
            include_refine_residual=include_refine_residual,
        )
        total_loss = total_loss + float(geometry_metric_weight_active) * geometry_metric_loss

    geometry_sd_loss = total_loss.new_zeros(())
    if float(geometry_sd_weight_active) > 0.0:
        geometry_sd_loss = _geometry_sd_loss(
            outputs,
            batch,
            include_refine_residual=include_refine_residual,
        )
        total_loss = total_loss + float(geometry_sd_weight_active) * geometry_sd_loss

    pred_xyz_refined = outputs.get("pred_xyz_refined", outputs["pred_xyz"])
    coarse_acc_metrics = _coarse_accuracy_metrics(outputs, batch)
    offset_acc_metrics = _offset_accuracy_metrics(outputs, batch, offset_num_bins=offset_num_bins)
    xyz_l1_soft = _l1_xyz_metric(soft_xyz_for_loss.detach(), batch)
    xyz_l1_refined = _l1_xyz_metric(pred_xyz_refined.detach(), batch)
    first_strip_wrong_side_rate_refined = _first_strip_wrong_side_rate_from_sequence(pred_xyz_refined.detach(), batch)
    pred_oob_fraction_refined = _pred_oob_fraction_from_sequence(pred_xyz_refined.detach(), batch)
    invalid_vertex_fraction_refined = _invalid_vertex_fraction_from_sequence(pred_xyz_refined.detach(), batch)
    boundary_touch_fraction_refined = _boundary_touch_fraction_from_sequence(pred_xyz_refined.detach(), batch)
    seam_edge_error_refined = _seam_edge_error_from_sequence(pred_xyz_refined.detach(), batch, band_width=int(seam_band_width))
    triangle_flip_rate_refined = _triangle_flip_rate_from_sequence(pred_xyz_refined.detach(), batch)
    geometry_metric_refined = _geometry_metric_loss_from_sequence(pred_xyz_refined.detach(), batch, loss_type=geometry_metric_loss_type)
    geometry_sd_refined = _geometry_sd_loss_from_sequence(pred_xyz_refined.detach(), batch)

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
        "coarse_z_loss": axis_loss_metrics["z"],
        "coarse_y_loss": axis_loss_metrics["y"],
        "coarse_x_loss": axis_loss_metrics["x"],
        "coarse_exact_acc": coarse_acc_metrics["coarse_exact_acc"],
        "coarse_axis_acc_z": coarse_acc_metrics["coarse_axis_acc_z"],
        "coarse_axis_acc_y": coarse_acc_metrics["coarse_axis_acc_y"],
        "coarse_axis_acc_x": coarse_acc_metrics["coarse_axis_acc_x"],
        "coarse_constraint_keep_fraction": outputs.get("coarse_constraint_keep_fraction", total_loss.new_zeros(())),
        "coarse_constraint_empty_rate": outputs.get("coarse_constraint_empty_rate", total_loss.new_zeros(())),
        "coarse_constraint_target_outside_rate": outputs.get("coarse_constraint_target_outside_rate", total_loss.new_zeros(())),
        "offset_loss": offset_loss,
        "offset_loss_weight_active": total_loss.new_tensor(float(offset_loss_weight_active)),
        "offset_exact_acc": offset_acc_metrics["offset_exact_acc"],
        "offset_mean_bin_error": offset_acc_metrics["offset_mean_bin_error"],
        "stop_loss": stop_loss,
        "refine_loss": refine_loss,
        "refine_loss_weight_active": total_loss.new_tensor(float(position_refine_weight_active)),
        "xyz_soft_loss": xyz_soft_loss,
        "xyz_soft_loss_weight_active": total_loss.new_tensor(float(xyz_soft_loss_weight_active)),
        "xyz_l1_soft": xyz_l1_soft,
        "xyz_l1_refined": xyz_l1_refined,
        "seam_loss": seam_loss,
        "seam_loss_weight_active": total_loss.new_tensor(float(seam_loss_weight_active)),
        "seam_edge_error_soft": seam_edge_error_soft,
        "seam_edge_error_refined": seam_edge_error_refined,
        "triangle_barrier_loss": triangle_barrier_loss,
        "triangle_barrier_weight_active": total_loss.new_tensor(float(triangle_barrier_weight_active)),
        "boundary_loss": boundary_loss,
        "boundary_loss_weight_active": total_loss.new_zeros(()),
        "triangle_flip_rate_soft": triangle_flip_rate_soft,
        "triangle_flip_rate_refined": triangle_flip_rate_refined,
        "first_strip_wrong_side_rate_refined": first_strip_wrong_side_rate_refined,
        "pred_oob_fraction_refined": pred_oob_fraction_refined,
        "teacher_forced_pred_oob_fraction": pred_oob_fraction_refined,
        "teacher_forced_invalid_vertex_fraction": invalid_vertex_fraction_refined,
        "teacher_forced_boundary_touch_fraction": boundary_touch_fraction_refined,
        "geometry_metric_loss": geometry_metric_loss,
        "geometry_metric_weight_active": total_loss.new_tensor(float(geometry_metric_weight_active)),
        "geometry_metric_refined": geometry_metric_refined,
        "geometry_sd_loss": geometry_sd_loss,
        "geometry_sd_weight_active": total_loss.new_tensor(float(geometry_sd_weight_active)),
        "geometry_sd_refined": geometry_sd_refined,
        "occupancy_metric": occupancy_metric,
    }
