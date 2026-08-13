from __future__ import annotations

import math
from types import SimpleNamespace

import torch

import model as fit_model


_DIR_EPS = 1.0e-6


def _edge_length(diff: torch.Tensor) -> torch.Tensor:
	return torch.sqrt((diff * diff).sum(dim=-1, keepdim=True) + 1e-8)


def _unit_directions(diff: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
	length = torch.linalg.vector_norm(diff, dim=-1, keepdim=True)
	valid = length > _DIR_EPS
	unit = torch.where(valid, diff / length.clamp_min(1.0e-12), torch.zeros_like(diff))
	return unit, valid


def _normalize_direction(direction_sum: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
	length = torch.linalg.vector_norm(direction_sum, dim=-1, keepdim=True)
	valid = length > _DIR_EPS
	direction = torch.where(valid, direction_sum / length.clamp_min(1.0e-12), torch.zeros_like(direction_sum))
	return direction.detach(), valid


def _h_edge_directions(diff_h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
	"""Local H-edge directions for quads, shape (D, Hm-1, Wm-1, 3)."""
	unit, valid = _unit_directions(diff_h)
	direction_sum = unit[:, :, :-1, :].clone()
	valid_count = valid[:, :, :-1, :].to(dtype=unit.dtype)

	direction_sum = direction_sum + unit[:, :, 1:, :]
	valid_count = valid_count + valid[:, :, 1:, :].to(dtype=unit.dtype)
	if int(diff_h.shape[2]) > 2:
		direction_sum[:, :, 1:, :] = direction_sum[:, :, 1:, :] + unit[:, :, :-2, :]
		valid_count[:, :, 1:, :] = valid_count[:, :, 1:, :] + valid[:, :, :-2, :].to(dtype=unit.dtype)

	direction_sum = torch.where(valid_count > 0.0, direction_sum, torch.zeros_like(direction_sum))
	return _normalize_direction(direction_sum)


def _w_edge_directions(diff_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
	"""Local W-edge directions for quads, shape (D, Hm-1, Wm-1, 3)."""
	unit, valid = _unit_directions(diff_w)
	direction_sum = unit[:, :-1, :, :].clone()
	valid_count = valid[:, :-1, :, :].to(dtype=unit.dtype)

	direction_sum = direction_sum + unit[:, 1:, :, :]
	valid_count = valid_count + valid[:, 1:, :, :].to(dtype=unit.dtype)
	if int(diff_w.shape[1]) > 2:
		direction_sum[:, 1:, :, :] = direction_sum[:, 1:, :, :] + unit[:, :-2, :, :]
		valid_count[:, 1:, :, :] = valid_count[:, 1:, :, :] + valid[:, :-2, :, :].to(dtype=unit.dtype)

	direction_sum = torch.where(valid_count > 0.0, direction_sum, torch.zeros_like(direction_sum))
	return _normalize_direction(direction_sum)


def _directional_step_penalty(
	diff: torch.Tensor,
	target: float | torch.Tensor,
	direction: torch.Tensor,
	direction_valid: torch.Tensor,
) -> torch.Tensor:
	if torch.is_tensor(target):
		target_t = target.to(device=diff.device, dtype=diff.dtype)
	else:
		target_t = diff.new_tensor(float(target))
	target_t = target_t.clamp_min(1.0e-12)
	length = _edge_length(diff)
	projected = (diff * direction).sum(dim=-1, keepdim=True).abs()
	short = torch.relu(target_t - projected).square() / target_t.square()
	long = torch.relu(length - target_t).square() / target_t.square()
	directional = short + long

	rel = (length - target_t) / target_t
	fallback = rel * rel
	return torch.where(direction_valid, directional, fallback)


def _offset_slices(size: int, offset: int) -> tuple[slice, slice]:
	if offset >= 0:
		return slice(0, size - offset), slice(offset, size)
	return slice(-offset, size), slice(0, size + offset)


def _offset_average(length: torch.Tensor, offsets: tuple[tuple[int, int], ...]) -> tuple[torch.Tensor, torch.Tensor]:
	out = torch.zeros_like(length)
	count = torch.zeros_like(length)
	H = int(length.shape[1])
	W = int(length.shape[2])
	for dh, dw in offsets:
		if abs(int(dh)) >= H or abs(int(dw)) >= W:
			continue
		src_h, dst_h = _offset_slices(H, int(dh))
		src_w, dst_w = _offset_slices(W, int(dw))
		out[:, dst_h, dst_w, :] = out[:, dst_h, dst_w, :] + length[:, src_h, src_w, :]
		count[:, dst_h, dst_w, :] = count[:, dst_h, dst_w, :] + 1.0
	return out, count


def _all_direction_local_average(edge_data: dict[str, dict[str, torch.Tensor]]) -> torch.Tensor:
	"""Differentiable 5-tap local average of equivalent step across all directions."""
	offsets = {
		"h": ((0, 0), (-1, 0), (1, 0), (-2, 0), (2, 0)),
		"w": ((0, 0), (0, -1), (0, 1), (0, -2), (0, 2)),
		"diag": ((0, 0), (-1, -1), (1, 1), (-2, -2), (2, 2)),
		"anti_diag": ((0, 0), (-1, 1), (1, -1), (-2, 2), (2, -2)),
	}
	total = torch.zeros_like(edge_data["h"]["length"])
	count = torch.zeros_like(total)
	for name in ("h", "w", "diag", "anti_diag"):
		equiv_length = edge_data[name]["length"] / edge_data[name]["equiv_scale"]
		part, part_count = _offset_average(equiv_length, offsets[name])
		total = total + part
		count = count + part_count
	return total / count.clamp_min(1.0)


def _step_regularizer_edge_data(xyz: torch.Tensor) -> dict[str, dict[str, torch.Tensor]]:
	diff_h_full = xyz[:, 1:, :, :] - xyz[:, :-1, :, :]
	dir_h, valid_h = _h_edge_directions(diff_h_full)
	diff_h = diff_h_full[:, :, :-1, :]

	diff_w_full = xyz[:, :, 1:, :] - xyz[:, :, :-1, :]
	dir_w, valid_w = _w_edge_directions(diff_w_full)
	diff_w = diff_w_full[:, :-1, :, :]

	diff_d1 = xyz[:, 1:, 1:, :] - xyz[:, :-1, :-1, :]
	dir_d1, valid_d1 = _unit_directions(diff_d1)

	diff_d2 = xyz[:, 1:, :-1, :] - xyz[:, :-1, 1:, :]
	dir_d2, valid_d2 = _unit_directions(diff_d2)

	return {
		"h": {
			"diff": diff_h,
			"direction": dir_h,
			"valid": valid_h,
			"length": _edge_length(diff_h),
			"equiv_scale": xyz.new_tensor(1.0),
		},
		"w": {
			"diff": diff_w,
			"direction": dir_w,
			"valid": valid_w,
			"length": _edge_length(diff_w),
			"equiv_scale": xyz.new_tensor(1.0),
		},
		"diag": {
			"diff": diff_d1,
			"direction": dir_d1.detach(),
			"valid": valid_d1,
			"length": _edge_length(diff_d1),
			"equiv_scale": xyz.new_tensor(math.sqrt(2.0)),
		},
		"anti_diag": {
			"diff": diff_d2,
			"direction": dir_d2.detach(),
			"valid": valid_d2,
			"length": _edge_length(diff_d2),
			"equiv_scale": xyz.new_tensor(math.sqrt(2.0)),
		},
	}


def _step_regularizer_targets(
	edge_data: dict[str, dict[str, torch.Tensor]],
	*,
	mesh_step: float,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
	local_avg_step = _all_direction_local_average(edge_data)
	smooth_targets = {
		name: local_avg_step * edge_data[name]["equiv_scale"]
		for name in ("h", "w", "diag", "anti_diag")
	}
	equiv_lengths = [
		edge_data[name]["length"] / edge_data[name]["equiv_scale"]
		for name in ("h", "w", "diag", "anti_diag")
	]
	global_avg_step = torch.cat([v.reshape(-1) for v in equiv_lengths], dim=0).mean()
	global_scale = edge_data["h"]["length"].new_tensor(float(mesh_step)) / global_avg_step.detach().clamp_min(1.0e-12)
	avg_targets = {name: target.detach() * global_scale for name, target in smooth_targets.items()}
	return smooth_targets, avg_targets, global_avg_step


def _step_regularizer_maps_from_targets(
	edge_data: dict[str, dict[str, torch.Tensor]],
	targets: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
	penalties = []
	for name in ("h", "w", "diag", "anti_diag"):
		data = edge_data[name]
		penalties.append(_directional_step_penalty(
			data["diff"],
			targets[name],
			data["direction"],
			data["valid"],
		))
	lm = sum(penalties) * 0.25
	lm = lm.permute(0, 3, 1, 2)
	mask = torch.ones_like(lm)
	return lm, mask


def step_regularizer_loss_maps(*, res: fit_model.FitResult3D) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
	"""Return grouped smooth/average step maps computed from one edge bundle."""
	edge_data = _step_regularizer_edge_data(res.xyz_lr)
	smooth_targets, avg_targets, _global_avg_step = _step_regularizer_targets(
		edge_data,
		mesh_step=float(res.params.mesh_step),
	)
	smooth_lm, smooth_mask = _step_regularizer_maps_from_targets(edge_data, smooth_targets)
	avg_lm, avg_mask = _step_regularizer_maps_from_targets(edge_data, avg_targets)
	return {
		"smooth_step": (smooth_lm, smooth_mask),
		"avg_step": (avg_lm, avg_mask),
	}


def step_regularizer_loss(
	*,
	res: fit_model.FitResult3D,
) -> dict[str, tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]]:
	"""Grouped local smooth-step and global average-step regularizer."""
	maps = step_regularizer_loss_maps(res=res)
	return {
		name: (lm.mean(), (lm,), (mask,))
		for name, (lm, mask) in maps.items()
	}


def step_loss_analysis(xyz: torch.Tensor, *, mesh_step: float) -> dict[str, float]:
	"""Return step-loss and edge-length stats for an uncropped mesh/shell grid.

	`loss` mirrors `step_loss()` on the supplied grid. The step stats combine H,
	W, and both diagonal edges, with diagonal lengths divided by sqrt(2) so they
	estimate the equivalent mesh step.
	"""
	if xyz.ndim == 3:
		xyz_b = xyz.unsqueeze(0)
	elif xyz.ndim == 4:
		xyz_b = xyz
	else:
		raise ValueError(f"xyz must have shape (H, W, 3) or (D, H, W, 3), got {tuple(xyz.shape)}")
	if int(xyz_b.shape[-1]) != 3:
		raise ValueError(f"xyz last dimension must be 3, got {tuple(xyz_b.shape)}")
	if int(xyz_b.shape[1]) < 2 or int(xyz_b.shape[2]) < 2:
		raise ValueError(f"step analysis requires H>=2 and W>=2, got {tuple(xyz_b.shape)}")
	mesh_step_i = max(1, int(round(float(mesh_step))))
	params = fit_model.ModelParams3D(
		mesh_step=mesh_step_i,
		winding_step=mesh_step_i,
		subsample_mesh=1,
		subsample_winding=1,
		scaledown=1.0,
		z_step_eff=1,
		volume_extent=None,
		pyramid_d=False,
	)
	res = SimpleNamespace(xyz_lr=xyz_b, params=params)
	with torch.no_grad():
		lm, _mask = step_loss_maps(res=res)  # type: ignore[arg-type]
		diff_h = xyz_b[:, 1:, :, :] - xyz_b[:, :-1, :, :]
		diff_w = xyz_b[:, :, 1:, :] - xyz_b[:, :, :-1, :]
		diff_d1 = xyz_b[:, 1:, 1:, :] - xyz_b[:, :-1, :-1, :]
		diff_d2 = xyz_b[:, 1:, :-1, :] - xyz_b[:, :-1, 1:, :]
		h_vals = torch.linalg.vector_norm(diff_h, dim=-1)
		w_vals = torch.linalg.vector_norm(diff_w, dim=-1)
		d1_vals = torch.linalg.vector_norm(diff_d1, dim=-1) / math.sqrt(2.0)
		d2_vals = torch.linalg.vector_norm(diff_d2, dim=-1) / math.sqrt(2.0)
		step_vals = torch.cat(
			[
				h_vals.reshape(-1),
				w_vals.reshape(-1),
				d1_vals.reshape(-1),
				d2_vals.reshape(-1),
			],
			dim=0,
		)
		h_max = float(h_vals.amax().detach().cpu())
		w_max = float(w_vals.amax().detach().cpu())
		d1_max = float(d1_vals.amax().detach().cpu())
		d2_max = float(d2_vals.amax().detach().cpu())
		max_kind = "h"
		max_value = h_max
		if w_max > max_value:
			max_kind = "w"
			max_value = w_max
		if d1_max > max_value:
			max_kind = "diag"
			max_value = d1_max
		if d2_max > max_value:
			max_kind = "anti_diag"
			max_value = d2_max
		return {
			"loss": float(lm.mean().detach().cpu()),
			"step_min": float(step_vals.amin().detach().cpu()),
			"step_avg": float(step_vals.mean().detach().cpu()),
			"step_med": float(step_vals.median().detach().cpu()),
			"step_max": float(step_vals.amax().detach().cpu()),
			"h_avg": float(h_vals.mean().detach().cpu()),
			"w_avg": float(w_vals.mean().detach().cpu()),
			"diag_avg": float(torch.cat([d1_vals.reshape(-1), d2_vals.reshape(-1)], dim=0).mean().detach().cpu()),
			"h_max": h_max,
			"w_max": w_max,
			"diag_max": max(d1_max, d2_max),
			"max_kind": max_kind,
			"max_value": max_value,
			"target": float(mesh_step_i),
		}


def step_loss_maps(*, res: fit_model.FitResult3D) -> tuple[torch.Tensor, torch.Tensor]:
	"""Return step-size squared penalty (relative), shape (D, 1, Hm-1, Wm-1).

	Checks four directions: H, W, and both diagonals.
	Short edges expand along the local mesh edge direction only, while long
	edges contract in full 3D. Diagonal target = mesh_step * sqrt(2).

	Returns (loss_map, mask) both (D, 1, Hm-1, Wm-1).
	"""
	xyz = res.xyz_lr  # (D, Hm, Wm, 3)
	t = float(res.params.mesh_step)
	t_diag = t * math.sqrt(2.0)

	# H direction: (D, Hm-1, Wm, 1) -> crop W to (D, Hm-1, Wm-1, 1)
	diff_h = xyz[:, 1:, :, :] - xyz[:, :-1, :, :]
	dir_h, valid_h = _h_edge_directions(diff_h)
	pen_h = _directional_step_penalty(diff_h[:, :, :-1, :], t, dir_h, valid_h)

	# W direction: (D, Hm, Wm-1, 1) -> crop H to (D, Hm-1, Wm-1, 1)
	diff_w = xyz[:, :, 1:, :] - xyz[:, :, :-1, :]
	dir_w, valid_w = _w_edge_directions(diff_w)
	pen_w = _directional_step_penalty(diff_w[:, :-1, :, :], t, dir_w, valid_w)

	# Diagonal (H+1, W+1): (D, Hm-1, Wm-1, 1)
	diff_d1 = xyz[:, 1:, 1:, :] - xyz[:, :-1, :-1, :]
	dir_d1, valid_d1 = _unit_directions(diff_d1)
	pen_d1 = _directional_step_penalty(diff_d1, t_diag, dir_d1.detach(), valid_d1)

	# Anti-diagonal (H+1, W-1): (D, Hm-1, Wm-1, 1)
	diff_d2 = xyz[:, 1:, :-1, :] - xyz[:, :-1, 1:, :]
	dir_d2, valid_d2 = _unit_directions(diff_d2)
	pen_d2 = _directional_step_penalty(diff_d2, t_diag, dir_d2.detach(), valid_d2)

	# Average all four, permute to (D, 1, Hm-1, Wm-1)
	lm = (pen_h + pen_w + pen_d1 + pen_d2) * 0.25
	lm = lm.permute(0, 3, 1, 2)
	mask = torch.ones_like(lm)
	return lm, mask


def step_loss(*, res: fit_model.FitResult3D) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
	"""Penalize mesh edge lengths deviating from mesh_step (relative)."""
	lm, mask = step_loss_maps(res=res)
	return lm.mean(), (lm,), (mask,)
