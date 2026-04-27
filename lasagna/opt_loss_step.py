from __future__ import annotations

import math

import torch

import model as fit_model


def step_loss_maps(*, res: fit_model.FitResult3D) -> tuple[torch.Tensor, torch.Tensor]:
	"""Return step-size squared penalty (relative), shape (D, 1, Hm-1, Wm-1).

	Checks four directions: H, W, and both diagonals.
	Penalizes deviation from the target mesh_step in fullres voxels
	(diagonal target = mesh_step * sqrt(2)).

	Returns (loss_map, mask) both (D, 1, Hm-1, Wm-1).
	"""
	xyz = res.xyz_lr  # (D, Hm, Wm, 3)
	t = float(res.params.mesh_step)
	t_diag = t * math.sqrt(2.0)

	def _rel_sq(diff: torch.Tensor, target: float) -> torch.Tensor:
		"""Relative squared penalty for edge lengths."""
		length = torch.sqrt((diff * diff).sum(dim=-1, keepdim=True) + 1e-8)  # (..., 1)
		rel = (length - target) / (target + 1e-12)
		return rel * rel

	# H direction: (D, Hm-1, Wm, 1) -> crop W to (D, Hm-1, Wm-1, 1)
	diff_h = xyz[:, 1:, :, :] - xyz[:, :-1, :, :]
	pen_h = _rel_sq(diff_h[:, :, :-1, :], t)

	# W direction: (D, Hm, Wm-1, 1) -> crop H to (D, Hm-1, Wm-1, 1)
	diff_w = xyz[:, :, 1:, :] - xyz[:, :, :-1, :]
	pen_w = _rel_sq(diff_w[:, :-1, :, :], t)

	# Diagonal (H+1, W+1): (D, Hm-1, Wm-1, 1)
	diff_d1 = xyz[:, 1:, 1:, :] - xyz[:, :-1, :-1, :]
	pen_d1 = _rel_sq(diff_d1, t_diag)

	# Anti-diagonal (H+1, W-1): (D, Hm-1, Wm-1, 1)
	diff_d2 = xyz[:, 1:, :-1, :] - xyz[:, :-1, 1:, :]
	pen_d2 = _rel_sq(diff_d2, t_diag)

	# Average all four, permute to (D, 1, Hm-1, Wm-1)
	lm = (pen_h + pen_w + pen_d1 + pen_d2) * 0.25
	lm = lm.permute(0, 3, 1, 2)
	mask = torch.ones_like(lm)
	return lm, mask


def step_loss(*, res: fit_model.FitResult3D) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
	"""Penalize mesh edge lengths deviating from mesh_step (relative)."""
	lm, mask = step_loss_maps(res=res)
	return lm.mean(), (lm,), (mask,)
