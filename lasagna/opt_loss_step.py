from __future__ import annotations

import torch

import model as fit_model


def step_loss_maps(*, res: fit_model.FitResult3D) -> torch.Tensor:
	"""Return height-step squared penalty (relative), shape (D, 1, Hm-1, Wm).

	Computes distance between adjacent mesh points along the height (H) axis
	and penalizes deviation from the target mesh_step in fullres voxels.
	"""
	# res.xyz_lr: (D, Hm, Wm, 3)
	diff = res.xyz_lr[:, 1:, :, :] - res.xyz_lr[:, :-1, :, :]  # (D, Hm-1, Wm, 3)
	len_v = torch.sqrt((diff * diff).sum(dim=-1, keepdim=True) + 1e-8)  # (D, Hm-1, Wm, 1)
	len_v = len_v.permute(0, 3, 1, 2)  # (D, 1, Hm-1, Wm)
	t_v = float(res.params.mesh_step)
	rel = (len_v - t_v) / (t_v + 1e-12)
	return rel * rel


def step_loss(*, res: fit_model.FitResult3D) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
	"""Penalize height mesh edge lengths deviating from mesh_step (relative)."""
	lm = step_loss_maps(res=res)
	mask = torch.ones_like(lm)
	return lm.mean(), (lm,), (mask,)
