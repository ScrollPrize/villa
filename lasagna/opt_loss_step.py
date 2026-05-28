from __future__ import annotations

import math

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
	target: float,
	direction: torch.Tensor,
	direction_valid: torch.Tensor,
) -> torch.Tensor:
	target_t = diff.new_tensor(float(target)).clamp_min(1.0e-12)
	length = _edge_length(diff)
	projected = (diff * direction).sum(dim=-1, keepdim=True).abs()
	short = torch.relu(target_t - projected).square() / target_t.square()
	long = torch.relu(length - target_t).square() / target_t.square()
	directional = short + long

	rel = (length - target_t) / target_t
	fallback = rel * rel
	return torch.where(direction_valid, directional, fallback)


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
