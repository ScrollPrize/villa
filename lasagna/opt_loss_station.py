from __future__ import annotations

import torch

import model as fit_model
from opt_loss_dir import _vertex_normals


# Module-level state set by set_station_ref(), persists for rest of stage.
_center_ref: torch.Tensor | None = None  # (3,) snapshot


def reset_station_ref() -> None:
	"""Clear station-keeping reference (e.g. at start of new stage)."""
	global _center_ref
	_center_ref = None


def _gaussian_center(res: fit_model.FitResult3D) -> torch.Tensor:
	"""Gaussian-weighted center-of-mass of xyz_lr. Returns (3,)."""
	D, Hm, Wm, _ = res.xyz_lr.shape
	dev = res.xyz_lr.device

	# Separable Gaussian weights — tight, ~3-5 central quads
	sig = 1.5
	cd = torch.arange(D, device=dev, dtype=torch.float32) - (D - 1) / 2
	ch = torch.arange(Hm, device=dev, dtype=torch.float32) - (Hm - 1) / 2
	cw = torch.arange(Wm, device=dev, dtype=torch.float32) - (Wm - 1) / 2
	gd = torch.exp(-0.5 * (cd / sig) ** 2)   # (D,)
	gh = torch.exp(-0.5 * (ch / sig) ** 2)   # (Hm,)
	gw = torch.exp(-0.5 * (cw / sig) ** 2)   # (Wm,)
	gauss = gd[:, None, None] * gh[None, :, None] * gw[None, None, :]  # (D, Hm, Wm)

	w = gauss * res.mask_lr.squeeze(1)  # (D, Hm, Wm)
	wsum = w.sum().clamp(min=1e-8)
	center = (res.xyz_lr * w.unsqueeze(-1)).sum(dim=(0, 1, 2)) / wsum  # (3,)
	return center


def set_station_ref(*, res: fit_model.FitResult3D) -> None:
	"""Snapshot current center as reference (call at stage start)."""
	global _center_ref
	with torch.no_grad():
		_center_ref = _gaussian_center(res).detach().clone()


def station_loss(*, res: fit_model.FitResult3D) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
	"""Loss penalizing drift of mesh center-of-mass from stage-start reference."""
	if _center_ref is None:
		z = torch.zeros((), device=res.xyz_lr.device)
		ones = torch.ones(1, 1, 1, 1, device=res.xyz_lr.device)
		return z, (z.view(1, 1, 1, 1),), (ones,)
	center = _gaussian_center(res)
	delta = center - _center_ref  # (3,)

	# Project displacement onto tangent plane (allow normal movement, penalize tangential drift)
	normals = _vertex_normals(res.xyz_lr.detach())  # (D, Hm, Wm, 3)
	D, Hm, Wm, _ = res.xyz_lr.shape
	sig = 1.5
	dev = res.xyz_lr.device
	cd = torch.arange(D, device=dev, dtype=torch.float32) - (D - 1) / 2
	ch = torch.arange(Hm, device=dev, dtype=torch.float32) - (Hm - 1) / 2
	cw = torch.arange(Wm, device=dev, dtype=torch.float32) - (Wm - 1) / 2
	gd = torch.exp(-0.5 * (cd / sig) ** 2)
	gh = torch.exp(-0.5 * (ch / sig) ** 2)
	gw = torch.exp(-0.5 * (cw / sig) ** 2)
	gauss = gd[:, None, None] * gh[None, :, None] * gw[None, None, :]
	w = gauss * res.mask_lr.squeeze(1)
	avg_normal = (normals * w.unsqueeze(-1)).sum(dim=(0, 1, 2))
	avg_normal = avg_normal / avg_normal.norm().clamp(min=1e-8)  # unit normal

	delta_tangential = delta - (delta * avg_normal).sum() * avg_normal
	loss = (delta_tangential ** 2).sum()
	# Dummy loss map/mask for the visualization pipeline
	D, Hm, Wm = res.xyz_lr.shape[:3]
	lm = loss.detach().expand(D, 1, Hm, Wm)
	mask = res.mask_lr
	return loss, (lm,), (mask,)
