from __future__ import annotations

import math

import torch

import model as fit_model

_THRESHOLD_RAD = math.radians(60.0)  # activate when bend angle > 60° from flat
_COS_THRESHOLD = math.cos(math.pi - _THRESHOLD_RAD)  # cos(120°) = -0.5


def bend_loss(*, res: fit_model.FitResult3D) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
	"""Penalize sharp bends in the mesh where adjacent edges form angles > 60° from flat.

	For each triplet of consecutive vertices (along H, W, and both diagonals),
	compute the angle at the middle vertex. Flat = 180° = cos(-1).
	Threshold at 120° (60° from flat). Only active when
	cos(angle) > cos(120°) = -0.5.
	"""
	xyz = res.xyz_lr  # (D, Hm, Wm, 3)

	def _bend_penalty(e1: torch.Tensor, e2: torch.Tensor) -> torch.Tensor:
		"""Compute bend penalty for edge pairs. e1, e2: (..., 3).

		cos(angle) between -e1 and e2 (consecutive edges).
		Flat → cos = -1. Sharp bend → cos → +1.
		Penalty = max(0, cos - threshold)².
		"""
		e1n = e1 * torch.rsqrt((e1 * e1).sum(dim=-1, keepdim=True) + 1e-12)
		e2n = e2 * torch.rsqrt((e2 * e2).sum(dim=-1, keepdim=True) + 1e-12)
		# angle at middle vertex: between -e1 (incoming) and e2 (outgoing)
		cos_angle = (-e1n * e2n).sum(dim=-1, keepdim=True)  # (..., 1)
		# flat = cos(-1), threshold at cos(120°) = -0.5
		excess = (cos_angle - _COS_THRESHOLD).clamp(min=0)
		return excess * excess

	# H-direction triplets: p[i-1], p[i], p[i+1]
	e1_h = xyz[:, 1:-1, :, :] - xyz[:, :-2, :, :]   # edge from i-1 to i
	e2_h = xyz[:, 2:, :, :] - xyz[:, 1:-1, :, :]     # edge from i to i+1
	pen_h = _bend_penalty(e1_h, e2_h)                 # (D, Hm-2, Wm, 1)

	# W-direction triplets
	e1_w = xyz[:, :, 1:-1, :] - xyz[:, :, :-2, :]
	e2_w = xyz[:, :, 2:, :] - xyz[:, :, 1:-1, :]
	pen_w = _bend_penalty(e1_w, e2_w)                 # (D, Hm, Wm-2, 1)

	# Diagonal triplets on the non-periodic interior, matching cyl_bend's two
	# diagonal directions without cylindrical width wrapping.
	mid = xyz[:, 1:-1, 1:-1, :]
	e1_d0 = mid - xyz[:, :-2, :-2, :]
	e2_d0 = xyz[:, 2:, 2:, :] - mid
	pen_d0 = _bend_penalty(e1_d0, e2_d0)              # (D, Hm-2, Wm-2, 1)
	e1_d1 = mid - xyz[:, :-2, 2:, :]
	e2_d1 = xyz[:, 2:, :-2, :] - mid
	pen_d1 = _bend_penalty(e1_d1, e2_d1)              # (D, Hm-2, Wm-2, 1)

	# Crop to common interior and average
	pen_h_crop = pen_h[:, :, 1:-1, :]                 # (D, Hm-2, Wm-2, 1)
	pen_w_crop = pen_w[:, 1:-1, :, :]                 # (D, Hm-2, Wm-2, 1)
	lm = (0.25 * (pen_h_crop + pen_w_crop + pen_d0 + pen_d1)).permute(0, 3, 1, 2)

	mask = torch.ones_like(lm)
	return lm.mean(), (lm,), (mask,)
