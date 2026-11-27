from __future__ import annotations

import torch

import mesh_loss_metal
import model as fit_model


def smooth_loss(*, res: fit_model.FitResult3D) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
	"""Penalize mesh vertices deviating from the average of their H and W neighbors.

	For each interior vertex, the target is the average of its 4 neighbors
	(up, down, left, right along the mesh H and W axes — NOT winding direction).
	"""
	xyz = res.xyz_lr  # (D, Hm, Wm, 3)
	if mesh_loss_metal.enabled(xyz, 3):
		# one fused dispatch each way on MPS; the PyTorch code below stays the reference
		lm = mesh_loss_metal.smooth_loss_map(xyz, float(res.params.mesh_step))
		return lm.mean(), (lm,), (torch.ones_like(lm),)
	step_sq = float(res.params.mesh_step) ** 2

	# Interior vertices: [1:-1] in both H and W
	center = xyz[:, 1:-1, 1:-1, :]  # (D, Hm-2, Wm-2, 3)
	avg = 0.25 * (
		xyz[:, :-2, 1:-1, :] +  # up
		xyz[:, 2:, 1:-1, :] +   # down
		xyz[:, 1:-1, :-2, :] +  # left
		xyz[:, 1:-1, 2:, :]     # right
	)
	diff = center - avg
	lm = (diff * diff).sum(dim=-1, keepdim=True) / step_sq  # (D, Hm-2, Wm-2, 1)
	lm = lm.permute(0, 3, 1, 2)  # (D, 1, Hm-2, Wm-2)
	mask = torch.ones_like(lm)
	return lm.mean(), (lm,), (mask,)
