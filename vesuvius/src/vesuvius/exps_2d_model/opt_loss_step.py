from __future__ import annotations

import torch


def step_loss_maps(*, model) -> tuple[torch.Tensor, torch.Tensor]:
	"""Return (step_h, step_v) squared penalties in pixel units.

	- step_h: (1,1,H,W-1) for horizontal edges
	- step_v: (1,1,H-1,W) for vertical edges
	"""
	x, y = model.grid_xy()
	wi = float(max(2, int(model.init.w_img)) - 1)
	hi = float(max(2, int(model.init.h_img)) - 1)
	x = (x + 1.0) * (0.5 * wi)
	y = (y + 1.0) * (0.5 * hi)

	dx_h = x[:, :, :, 1:] - x[:, :, :, :-1]
	dy_h = y[:, :, :, 1:] - y[:, :, :, :-1]
	len_h = torch.sqrt(dx_h * dx_h + dy_h * dy_h + 1e-8)

	dx_v = x[:, :, 1:, :] - x[:, :, :-1, :]
	dy_v = y[:, :, 1:, :] - y[:, :, :-1, :]
	len_v = torch.sqrt(dx_v * dx_v + dy_v * dy_v + 1e-8)

	t_h = float(model.init.winding_step_px)
	t_v = float(model.init.mesh_step_px)
	step_h = (len_h - t_h) * (len_h - t_h)
	step_v = (len_v - t_v) * (len_v - t_v)
	return step_h, step_v


def step_loss(*, model) -> torch.Tensor:
	"""Penalize mesh edge lengths deviating from the configured pixel step sizes."""
	step_h, step_v = step_loss_maps(model=model)
	return step_h.mean() + step_v.mean()
