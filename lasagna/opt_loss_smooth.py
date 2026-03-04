from __future__ import annotations

import torch

import model as fit_model


def smooth_loss(*, res: fit_model.FitResult3D) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
	"""Penalize mesh vertices deviating from the average of their H and W neighbors.

	For each interior vertex, the target is the average of its 4 neighbors
	(up, down, left, right along the mesh H and W axes — NOT winding direction).
	"""
	xyz = res.xyz_lr  # (D, Hm, Wm, 3)

	# Interior vertices: [1:-1] in both H and W
	center = xyz[:, 1:-1, 1:-1, :]  # (D, Hm-2, Wm-2, 3)
	avg = 0.25 * (
		xyz[:, :-2, 1:-1, :] +  # up
		xyz[:, 2:, 1:-1, :] +   # down
		xyz[:, 1:-1, :-2, :] +  # left
		xyz[:, 1:-1, 2:, :]     # right
	)
	diff = center - avg
	lm = (diff * diff).sum(dim=-1, keepdim=True)  # (D, Hm-2, Wm-2, 1)
	lm = lm.permute(0, 3, 1, 2)  # (D, 1, Hm-2, Wm-2)
	mask = torch.ones_like(lm)
	return lm.mean(), (lm,), (mask,)
