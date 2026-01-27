from __future__ import annotations

import torch

import model as fit_model


def step_loss_maps(*, res: fit_model.FitResult) -> tuple[torch.Tensor, torch.Tensor]:
	"""Return (step_h, step_v) squared penalties in pixel units.

	- step_h: (1,1,H,W-1) for horizontal edges
	- step_v: (1,1,H-1,W) for vertical edges
	"""
	x = res.xy_lr[..., 0].unsqueeze(1)
	y = res.xy_lr[..., 1].unsqueeze(1)

	dx_h = x[:, :, :, 1:] - x[:, :, :, :-1]
	dy_h = y[:, :, :, 1:] - y[:, :, :, :-1]
	len_h = torch.sqrt(dx_h * dx_h + dy_h * dy_h + 1e-8)

	dx_v = x[:, :, 1:, :] - x[:, :, :-1, :]
	dy_v = y[:, :, 1:, :] - y[:, :, :-1, :]
	len_v = torch.sqrt(dx_v * dx_v + dy_v * dy_v + 1e-8)

	t_h = float(res.winding_step_px)
	t_v = float(res.mesh_step_px)
	step_h = (len_h - t_h) * (len_h - t_h)
	step_v = (len_v - t_v) * (len_v - t_v)
	return step_h, step_v


def step_loss(*, res: fit_model.FitResult) -> torch.Tensor:
	"""Penalize mesh edge lengths deviating from the configured pixel step sizes."""
	step_h, step_v = step_loss_maps(res=res)
	return step_h.mean() + step_v.mean()
