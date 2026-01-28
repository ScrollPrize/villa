from __future__ import annotations

import torch

import model as fit_model



def step_loss_maps(*, res: fit_model.FitResult) -> torch.Tensor:
	"""Return vertical step squared penalty (relative), shape (1,1,H-1,W)."""
	x = res.xy_lr[..., 0].unsqueeze(1)
	y = res.xy_lr[..., 1].unsqueeze(1)

	dx_v = x[:, :, 1:, :] - x[:, :, :-1, :]
	dy_v = y[:, :, 1:, :] - y[:, :, :-1, :]
	len_v = torch.sqrt(dx_v * dx_v + dy_v * dy_v + 1e-8)

	t_v = float(res.mesh_step_px)
	eps = 1e-12
	rel = (len_v - t_v) / (t_v + eps)
	return rel * rel


def step_loss(*, res: fit_model.FitResult) -> torch.Tensor:
	"""Penalize vertical mesh edge lengths deviating from the configured step size (relative)."""
	step_v = step_loss_maps(res=res)
	return step_v.mean()
