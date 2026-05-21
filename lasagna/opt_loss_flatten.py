from __future__ import annotations

import torch

import model as fit_model


_sdir_eps = 1.0e-8
_orient_min_det = 0.0
_last_stats: dict[str, float] = {}
_prev_point_mask: torch.Tensor | None = None


def configure(
	*,
	sdir_eps: float | None = None,
	orient_min_det: float | None = None,
	reset_history: bool = True,
) -> None:
	global _sdir_eps, _orient_min_det, _last_stats, _prev_point_mask
	if sdir_eps is not None:
		_sdir_eps = max(1.0e-12, float(sdir_eps))
	if orient_min_det is not None:
		_orient_min_det = float(orient_min_det)
	if reset_history:
		_last_stats = {}
		_prev_point_mask = None


def last_stats() -> dict[str, float]:
	return dict(_last_stats)


def flatten_sdir_loss(
	*,
	res: fit_model.FitResult3D,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
	"""Symmetric Dirichlet energy for the flatten inverse map output surface.

	The surface samples live in fullres coordinates.  The 2D output grid domain
	uses the measured average spacing of the source tifxyz grid, so flattening
	does not impose global scaling from metadata.
	"""
	global _last_stats, _prev_point_mask
	xyz = res.flatten_xyz
	point_mask = res.flatten_point_mask
	quad_mask = res.flatten_quad_mask
	if xyz is None or point_mask is None or quad_mask is None:
		raise RuntimeError("flatten_sdir requires flatten forward artifacts")
	if xyz.ndim != 4 or int(xyz.shape[0]) != 1 or int(xyz.shape[-1]) != 3:
		raise RuntimeError(f"flatten_xyz must have shape (1,H,W,3), got {tuple(xyz.shape)}")
	if int(xyz.shape[1]) < 2 or int(xyz.shape[2]) < 2:
		zero = xyz.sum() * 0.0
		return zero, (zero.reshape(1, 1, 1, 1),), (zero.reshape(1, 1, 1, 1),)

	p00 = xyz[:, :-1, :-1]
	p10 = xyz[:, 1:, :-1]
	p01 = xyz[:, :-1, 1:]
	p11 = xyz[:, 1:, 1:]
	if res.flatten_target_step is None:
		domain_step = torch.tensor(float(res.params.mesh_step), device=xyz.device, dtype=xyz.dtype)
	else:
		domain_step = res.flatten_target_step.to(device=xyz.device, dtype=xyz.dtype)
	domain_step = domain_step.clamp_min(1.0e-12)
	du = 0.5 * ((p10 - p00) + (p11 - p01)) / domain_step
	dv = 0.5 * ((p01 - p00) + (p11 - p10)) / domain_step

	a = (du * du).sum(dim=-1)
	b = (du * dv).sum(dim=-1)
	c = (dv * dv).sum(dim=-1)
	det = (a * c - b * b).clamp_min(float(_sdir_eps))
	tr_g = a + c
	tr_inv = (a + c) / det
	lm = torch.nan_to_num(tr_g + tr_inv - 4.0, nan=0.0, posinf=1.0e12, neginf=0.0)
	mask = quad_mask.to(device=lm.device, dtype=lm.dtype)
	wsum = mask.sum()
	if bool((wsum > 0).detach().cpu()):
		loss = (lm * mask).sum() / wsum
	else:
		loss = (lm * 0.0).sum()

	with torch.no_grad():
		pm = point_mask.to(dtype=torch.bool)
		qm = quad_mask.to(dtype=torch.bool)
		prev_pm = _prev_point_mask
		if prev_pm is None or tuple(prev_pm.shape) != tuple(pm.shape):
			valid_to_invalid = torch.zeros_like(pm)
			invalid_to_valid = torch.zeros_like(pm)
			no_new_loss = loss.detach()
		else:
			prev_pm = prev_pm.to(device=pm.device, dtype=torch.bool)
			valid_to_invalid = prev_pm & ~pm
			invalid_to_valid = ~prev_pm & pm
			stable_pm = pm & prev_pm
			if int(stable_pm.shape[0]) > 1 and int(stable_pm.shape[1]) > 1:
				stable_qm = (
					stable_pm[:-1, :-1] &
					stable_pm[1:, :-1] &
					stable_pm[:-1, 1:] &
					stable_pm[1:, 1:]
				).unsqueeze(0) & qm
			else:
				stable_qm = torch.zeros_like(qm)
			stable_mask = stable_qm.to(device=lm.device, dtype=lm.dtype)
			stable_wsum = stable_mask.sum()
			if bool((stable_wsum > 0).detach().cpu()):
				no_new_loss = (lm * stable_mask).sum() / stable_wsum
			else:
				no_new_loss = loss.detach()
		_last_stats = {
			"flatten_point_valid": float(pm.float().mean().detach().cpu()),
			"flatten_quad_valid": float(qm.float().mean().detach().cpu()) if qm.numel() else 0.0,
			"flatten_tgt_step": float(domain_step.detach().cpu()),
			"flatten_valid_to_invalid": float(valid_to_invalid.float().mean().detach().cpu()),
			"flatten_invalid_to_valid": float(invalid_to_valid.float().mean().detach().cpu()),
			"flatten_sdir_no_new": float(no_new_loss.detach().cpu()),
		}
		_prev_point_mask = pm.detach().clone()
	return loss, (lm.unsqueeze(1),), (mask.unsqueeze(1),)


def flatten_map_step_loss(
	*,
	res: fit_model.FitResult3D,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
	"""Regularize the inverse map to advance one source-grid cell per output step."""
	map_yx = res.flatten_map
	if map_yx is None:
		raise RuntimeError("flatten_map_step requires flatten_map")
	if map_yx.ndim != 3 or int(map_yx.shape[-1]) != 2:
		raise RuntimeError(f"flatten_map must have shape (H,W,2), got {tuple(map_yx.shape)}")

	H, W = int(map_yx.shape[0]), int(map_yx.shape[1])
	maps: list[torch.Tensor] = []
	masks: list[torch.Tensor] = []
	sum_loss = map_yx.sum() * 0.0
	sum_weight = map_yx.new_zeros(())

	def _accumulate(lm: torch.Tensor) -> None:
		nonlocal sum_loss, sum_weight
		mask = torch.isfinite(lm)
		mask_f = mask.to(dtype=lm.dtype)
		maps.append(torch.nan_to_num(lm, nan=0.0, posinf=1.0e12, neginf=0.0).unsqueeze(0).unsqueeze(1))
		masks.append(mask_f.unsqueeze(0).unsqueeze(1))
		sum_loss = sum_loss + (maps[-1].squeeze(0).squeeze(0) * mask_f).sum()
		sum_weight = sum_weight + mask_f.sum()

	if H > 1:
		target_y = torch.tensor([1.0, 0.0], device=map_yx.device, dtype=map_yx.dtype)
		dy = map_yx[1:, :] - map_yx[:-1, :] - target_y
		_accumulate((dy * dy).sum(dim=-1))
	if W > 1:
		target_x = torch.tensor([0.0, 1.0], device=map_yx.device, dtype=map_yx.dtype)
		dx = map_yx[:, 1:] - map_yx[:, :-1] - target_x
		_accumulate((dx * dx).sum(dim=-1))

	if bool((sum_weight > 0).detach().cpu()):
		loss = sum_loss / sum_weight
	else:
		loss = map_yx.sum() * 0.0
		zero = loss.reshape(1, 1, 1, 1)
		maps = [zero]
		masks = [zero]
	return loss, tuple(maps), tuple(masks)


def flatten_avg_offset_loss(
	*,
	res: fit_model.FitResult3D,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
	"""Keep the mean inverse-map offset fixed over the init-valid source area."""
	global _last_stats
	map_yx = res.flatten_map
	mask = res.flatten_avg_offset_mask
	target = res.flatten_initial_avg_offset
	if map_yx is None or mask is None or target is None:
		raise RuntimeError("flatten_avg_offset requires flatten map offset anchor artifacts")
	if map_yx.ndim != 3 or int(map_yx.shape[-1]) != 2:
		raise RuntimeError(f"flatten_map must have shape (H,W,2), got {tuple(map_yx.shape)}")
	if tuple(mask.shape) != tuple(map_yx.shape[:2]):
		raise RuntimeError(f"flatten_avg_offset_mask shape {tuple(mask.shape)} does not match map {tuple(map_yx.shape[:2])}")
	if target.numel() != 2:
		raise RuntimeError(f"flatten_initial_avg_offset must have 2 values, got {tuple(target.shape)}")

	identity = fit_model.Model3D._identity_flatten_map(
		h=int(map_yx.shape[0]),
		w=int(map_yx.shape[1]),
		device=map_yx.device,
		dtype=map_yx.dtype,
	)
	mask_f = mask.to(device=map_yx.device, dtype=map_yx.dtype)
	weight = mask_f.sum()
	if bool((weight > 0).detach().cpu()):
		avg_offset = ((map_yx - identity) * mask_f.unsqueeze(-1)).sum(dim=(0, 1)) / weight
		diff = avg_offset - target.to(device=map_yx.device, dtype=map_yx.dtype).reshape(2)
		loss = (diff * diff).sum()
	else:
		avg_offset = map_yx.sum(dim=(0, 1)) * 0.0
		diff = avg_offset
		loss = map_yx.sum() * 0.0
	lm = torch.nan_to_num(
		((map_yx - identity) - target.to(device=map_yx.device, dtype=map_yx.dtype).reshape(1, 1, 2)).square().sum(dim=-1),
		nan=0.0,
		posinf=1.0e12,
		neginf=0.0,
	)
	with torch.no_grad():
		_last_stats = {
			**_last_stats,
			"flatten_avg_offset_norm": float(torch.linalg.vector_norm(diff).detach().cpu()),
		}
	return loss, (lm.unsqueeze(0).unsqueeze(1),), (mask_f.unsqueeze(0).unsqueeze(1),)


def flatten_orient_loss(
	*,
	res: fit_model.FitResult3D,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
	"""Source-map signed-area hinge for fold-in prevention."""
	global _last_stats
	map_yx = res.flatten_map
	if map_yx is None:
		raise RuntimeError("flatten_orient requires flatten_map")
	if map_yx.ndim != 3 or int(map_yx.shape[-1]) != 2:
		raise RuntimeError(f"flatten_map must have shape (H,W,2), got {tuple(map_yx.shape)}")

	H, W = int(map_yx.shape[0]), int(map_yx.shape[1])
	if H < 2 or W < 2:
		zero = map_yx.sum() * 0.0
		return zero, (zero.reshape(1, 1, 1, 1),), (zero.reshape(1, 1, 1, 1),)

	m00 = map_yx[:-1, :-1]
	m10 = map_yx[1:, :-1]
	m01 = map_yx[:-1, 1:]
	m11 = map_yx[1:, 1:]
	dy = 0.5 * ((m10 - m00) + (m11 - m01))
	dx = 0.5 * ((m01 - m00) + (m11 - m10))
	det = dy[..., 0] * dx[..., 1] - dy[..., 1] * dx[..., 0]
	min_det = torch.tensor(float(_orient_min_det), device=map_yx.device, dtype=map_yx.dtype)
	lm = torch.nan_to_num(torch.relu(min_det - det) ** 2, nan=0.0, posinf=1.0e12, neginf=0.0)
	active = torch.isfinite(det) & (det < min_det)
	mask = active.to(dtype=lm.dtype)
	loss = (lm * mask).sum()

	with torch.no_grad():
		valid_det = det[torch.isfinite(det)]
		if valid_det.numel():
			fold_frac = float((valid_det <= 0.0).float().mean().detach().cpu())
			lowdet_frac = float((valid_det < min_det).float().mean().detach().cpu())
			min_det_val = float(valid_det.min().detach().cpu())
			mean_det_val = float(valid_det.mean().detach().cpu())
		else:
			fold_frac = 0.0
			lowdet_frac = 0.0
			min_det_val = 0.0
			mean_det_val = 0.0
		_last_stats = {
			**_last_stats,
			"flatten_orient_fold_frac": fold_frac,
			"flatten_orient_lowdet_frac": lowdet_frac,
			"flatten_orient_min_det": min_det_val,
			"flatten_orient_mean_det": mean_det_val,
		}
	return loss, (lm.unsqueeze(0).unsqueeze(1),), (mask.unsqueeze(0).unsqueeze(1),)
