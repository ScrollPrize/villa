from __future__ import annotations

import math

import torch

import model as fit_model

_last_stats: dict[str, float] = {}
_candidate_terms: dict[str, torch.Tensor] = {}
_candidate_normal_stats: dict[str, torch.Tensor] = {}
_sample_cache_key: int | None = None
_sample_cache_value: tuple[torch.Tensor | None, torch.Tensor | None] | None = None


def last_stats() -> dict[str, float]:
	return dict(_last_stats)


def reset_candidate_terms() -> None:
	global _candidate_terms, _candidate_normal_stats, _sample_cache_key, _sample_cache_value, _last_stats
	_candidate_terms = {}
	_candidate_normal_stats = {}
	_sample_cache_key = None
	_sample_cache_value = None
	_last_stats = {}


def _active_terms(weights: dict[str, float]) -> dict[str, float]:
	return {
		name: float(weights.get(name, 0.0))
		for name in ("cyl_normal", "cyl_center")
		if float(weights.get(name, 0.0)) != 0.0 and name in _candidate_terms
	}


def _candidate_totals(weights: dict[str, float]) -> tuple[torch.Tensor | None, dict[str, float]]:
	active = _active_terms(weights)
	if not active:
		return None, {}
	ref = next(iter(_candidate_terms.values()))
	total = torch.zeros_like(ref)
	for name, weight in active.items():
		total = total + weight * _candidate_terms[name]
	return total, active


def top_candidates(weights: dict[str, float], *, limit: int) -> list[dict[str, float | int]]:
	total, active = _candidate_totals(weights)
	if total is None:
		return []
	valid = torch.isfinite(total)
	if not bool(valid.any().detach().cpu()):
		return []
	total_valid = torch.where(valid, total, torch.full_like(total, float("inf")))
	k = min(max(0, int(limit)), int(valid.sum().detach().cpu()))
	if k == 0:
		return []
	values, indices = torch.topk(total_valid, k=k, largest=False)
	out: list[dict[str, float | int]] = []
	for rank, (value_t, idx_t) in enumerate(zip(values, indices, strict=True), start=1):
		idx = int(idx_t.detach().cpu())
		row: dict[str, float | int] = {
			"rank": rank,
			"idx": idx,
			"cyl_min": float(value_t.detach().cpu()),
		}
		for name in active:
			row[name] = float(_candidate_terms[name][idx_t].detach().cpu())
		for name, stat in _candidate_normal_stats.items():
			row[name] = float(stat[idx_t].detach().cpu())
		out.append(row)
	return out


def top_candidate_indices(weights: dict[str, float], *, limit: int) -> list[int]:
	return [int(row["idx"]) for row in top_candidates(weights, limit=limit)]


def display_stats(weights: dict[str, float]) -> tuple[int | None, float | None, dict[str, float]]:
	total, active = _candidate_totals(weights)
	if total is None:
		return None, None, {}
	valid = torch.isfinite(total)
	if not bool(valid.any().detach().cpu()):
		return None, None, {"cyl_min": float("inf")}
	total_valid = torch.where(valid, total, torch.full_like(total, float("inf")))
	best_t = torch.argmin(total_valid)
	best = int(best_t.detach().cpu())
	out = {"cyl_min": float(total_valid[best_t].detach().cpu())}
	for name in active:
		out[name] = float(_candidate_terms[name][best_t].detach().cpu())
	for name, values in _candidate_normal_stats.items():
		out[name] = float(values[best_t].detach().cpu())
	max_values = _candidate_normal_stats.get("cyl_nerr_max")
	if max_values is not None:
		max_valid = torch.isfinite(max_values)
		if bool(max_valid.any().detach().cpu()):
			out["cyl_nerr_max_min"] = float(max_values[max_valid].amin().detach().cpu())
		else:
			out["cyl_nerr_max_min"] = float("inf")
	return best, out["cyl_min"], out


def cyl_normal_prefetch_items_for_result(*, res: fit_model.FitResult3D) -> dict[str, torch.Tensor]:
	if res.cyl_xyz is None or res.cyl_count <= 0:
		return {}
	pts = res.cyl_xyz.detach()
	return {"grad_mag": pts, "nx": pts, "ny": pts}


def _zero_loss(res: fit_model.FitResult3D) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
	zero = res.xyz_lr.sum() * 0.0
	return zero, (zero.view(1, 1, 1, 1),), (zero.view(1, 1, 1, 1),)


def _sample_gt(res: fit_model.FitResult3D) -> tuple[torch.Tensor | None, torch.Tensor | None]:
	global _sample_cache_key, _sample_cache_value
	key = id(res)
	if _sample_cache_key == key and _sample_cache_value is not None:
		return _sample_cache_value
	if res.cyl_xyz is None or res.cyl_normals is None or res.cyl_count <= 0:
		return None, None
	sampled = res.data.grid_sample_fullres(
		res.cyl_xyz.detach(),
		diff=False,
		channels={"grad_mag", "nx", "ny"},
	)
	_sample_cache_key = key
	_sample_cache_value = (sampled.normal_3d, sampled.grad_mag)
	return _sample_cache_value


def _cyl_dims(res: fit_model.FitResult3D) -> tuple[int, int, int, int, int]:
	N = int(res.cyl_count)
	S = max(1, int(getattr(res, "cyl_sample_count", 1)))
	D = max(1, int(res.xyz_lr.shape[0]))
	H = int(res.cyl_xyz.shape[1])
	W = int(res.cyl_xyz.shape[2])
	return N, S, D, H, W


def _candidate_mean(lm: torch.Tensor, mask: torch.Tensor, *, res: fit_model.FitResult3D) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	N, S, D, H, W = _cyl_dims(res)
	lm_c = lm.reshape(N, S, D, H, W)
	mask_c = mask.reshape(N, S, D, H, W) * _mesh_distance_falloff(res=res, dtype=lm.dtype, device=lm.device)
	wsum = mask_c.sum(dim=(1, 2, 3, 4))
	err = (lm_c * mask_c).sum(dim=(1, 2, 3, 4)) / wsum.clamp(min=1.0)
	err = torch.where(wsum > 0.0, err, torch.full_like(err, float("inf")))
	return err, lm_c, mask_c


def _mesh_distance_falloff(*, res: fit_model.FitResult3D, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
	N, S, D, H, W = _cyl_dims(res)
	with torch.no_grad():
		h_idx = torch.arange(H, device=device, dtype=dtype)
		w_idx = torch.arange(W, device=device, dtype=dtype)
		h_center = float(H // 2)
		w_center = float(W // 2)
		h_edge = max(h_center, float(H - 1) - h_center, 1.0)
		w_edge = max(w_center, float(W - 1) - w_center, 1.0)
		h_dist = ((h_idx - h_center).abs() / h_edge).clamp(max=1.0)
		w_dist = ((w_idx - w_center).abs() / w_edge).clamp(max=1.0)
		grid_dist = torch.maximum(h_dist.view(1, H, 1), w_dist.view(1, 1, W))
		floor = 0.1
		falloff = floor + (1.0 - floor) * (1.0 - grid_dist).pow(2.0)
		return falloff.view(1, 1, 1, H, W).expand(N, S, D, H, W)


def _register_normal_angle_stats(dot: torch.Tensor, mask: torch.Tensor, *, res: fit_model.FitResult3D) -> None:
	N, S, D, H, W = _cyl_dims(res)
	with torch.no_grad():
		dot_c = dot.detach().reshape(N, S, D, H, W).clamp(min=-1.0, max=1.0)
		mask_f = mask.detach().reshape(N, S, D, H, W) * _mesh_distance_falloff(res=res, dtype=dot_c.dtype, device=dot_c.device)
		mask_c = mask_f > 0.0
		angle = torch.acos(dot_c) * (180.0 / math.pi)
		wsum = mask_f.sum(dim=(1, 2, 3, 4))
		avg = (angle * mask_f).sum(dim=(1, 2, 3, 4)) / wsum.clamp(min=1.0)
		avg = torch.where(wsum > 0.0, avg, torch.full_like(avg, float("inf")))
		inf = torch.full_like(angle, float("inf"))
		ninf = torch.full_like(angle, float("-inf"))
		min_v = torch.where(mask_c, angle, inf).amin(dim=(1, 2, 3, 4))
		max_v = torch.where(mask_c, angle, ninf).amax(dim=(1, 2, 3, 4))
		max_v = torch.where(wsum > 0.0, max_v, torch.full_like(max_v, float("inf")))
		_candidate_normal_stats["cyl_nerr_avg"] = avg.detach()
		_candidate_normal_stats["cyl_nerr_min"] = min_v.detach()
		_candidate_normal_stats["cyl_nerr_max"] = max_v.detach()


def _register_candidate_term(name: str, err: torch.Tensor) -> torch.Tensor:
	_candidate_terms[name] = err.detach()
	valid = torch.isfinite(err)
	if bool(valid.any().detach().cpu()):
		return err[valid].mean()
	return err.new_zeros(()) + err[~valid].new_zeros(()).sum() * 0.0


def _register_shell_normal_angle_stats(dot_abs: torch.Tensor, mask: torch.Tensor) -> None:
	with torch.no_grad():
		dot_abs = dot_abs.detach().clamp(min=0.0, max=1.0)
		mask_f = mask.detach().to(dtype=dot_abs.dtype)
		mask_c = mask_f > 0.0
		angle = torch.acos(dot_abs) * (180.0 / math.pi)
		wsum = mask_f.sum()
		avg = (angle * mask_f).sum() / wsum.clamp(min=1.0)
		avg = torch.where(wsum > 0.0, avg, torch.full_like(avg, float("inf")))
		inf = torch.full_like(angle, float("inf"))
		ninf = torch.full_like(angle, float("-inf"))
		min_v = torch.where(mask_c, angle, inf).amin()
		max_v = torch.where(mask_c, angle, ninf).amax()
		max_v = torch.where(wsum > 0.0, max_v, torch.full_like(max_v, float("inf")))
		_candidate_normal_stats["cyl_nerr_avg"] = avg.detach().view(1)
		_candidate_normal_stats["cyl_nerr_min"] = min_v.detach().view(1)
		_candidate_normal_stats["cyl_nerr_max"] = max_v.detach().view(1)


def _unit_xy(v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
	length = v.norm(dim=-1).clamp(min=1.0e-8)
	return v / length.unsqueeze(-1), length


def _signed_xy_normal_alignment(
	*,
	res: fit_model.FitResult3D,
	target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
	cyl_xy, _cyl_xy_len = _unit_xy(res.cyl_normals[..., :2])
	with torch.no_grad():
		xyz = res.cyl_xyz.detach()
		target_xy_raw = target[..., :2].detach()
		target_xy, target_xy_len = _unit_xy(target_xy_raw)
		umb_xy = res.data.umbilicus_xy_at_z(xyz[..., 2])
		radial_xy, radial_len = _unit_xy(xyz[..., :2] - umb_xy)
		target_dot_radial = (target_xy * radial_xy).sum(dim=-1)
		flip = target_dot_radial < 0.0
		target_oriented = torch.where(flip.unsqueeze(-1), -target_xy, target_xy)
		radial_weight = (target_oriented * radial_xy).sum(dim=-1).clamp(min=0.0, max=1.0)
		in_plane_weight = target_xy_raw.norm(dim=-1).clamp(min=0.0, max=1.0)
		valid = (target_xy_len > 1.0e-7) & (radial_len > 1.0e-7)
		weight = radial_weight * in_plane_weight * valid.to(dtype=target.dtype)
	dot = (cyl_xy * target_oriented).sum(dim=-1).clamp(min=-1.0, max=1.0)
	return dot, weight


def cyl_normal_loss(*, res: fit_model.FitResult3D) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
	"""Signed xy normal alignment for analytic cylinder candidates.

	GT normal samples are intentionally non-differentiable; gradients flow only
	through the analytic cylinder normals. The GT normal is oriented away from
	the umbilicus in the local xy slice before comparison.
	"""
	if res.cyl_xyz is None or res.cyl_normals is None or res.cyl_count <= 0:
		return _zero_loss(res)

	target, grad_mag = _sample_gt(res)
	if target is None or grad_mag is None:
		return _zero_loss(res)

	if bool(getattr(res, "cyl_shell_mode", False)):
		normal = res.cyl_normals
		normal = normal / normal.norm(dim=-1, keepdim=True).clamp(min=1.0e-8)
		dot = (normal * target).sum(dim=-1).clamp(min=-1.0, max=1.0)
		dot_abs = dot.abs()
		lm = 1.0 - dot_abs * dot_abs
		mask = (grad_mag.squeeze(0).squeeze(0) > 0.0).to(dtype=lm.dtype)
		wsum = mask.sum()
		err = (lm * mask).sum() / wsum.clamp(min=1.0)
		err = torch.where(wsum > 0.0, err, torch.full_like(err, float("inf")))
		_register_shell_normal_angle_stats(dot_abs, mask)
		loss = _register_candidate_term("cyl_normal", err.view(1))
		return loss, (lm.unsqueeze(0).unsqueeze(0),), (mask.unsqueeze(0).unsqueeze(0),)

	dot, normal_weight = _signed_xy_normal_alignment(res=res, target=target)
	lm = 1.0 - dot
	mask = (grad_mag.squeeze(0).squeeze(0) > 0.0).to(dtype=lm.dtype) * normal_weight.to(dtype=lm.dtype)

	_register_normal_angle_stats(dot, mask, res=res)
	err, lm_c, mask_c = _candidate_mean(lm, mask, res=res)
	loss = _register_candidate_term("cyl_normal", err)
	return loss, (lm_c,), (mask_c,)


def cyl_center_loss(*, res: fit_model.FitResult3D) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
	"""Distance from sampled GT normal rays to each candidate cylinder axis."""
	if res.cyl_xyz is None or res.cyl_centers is None or res.cyl_axes is None or res.cyl_count <= 0:
		return _zero_loss(res)

	target, grad_mag = _sample_gt(res)
	if target is None or grad_mag is None:
		return _zero_loss(res)

	N, S, D, H, W = _cyl_dims(res)
	xyz = res.cyl_xyz.reshape(N, S, D, H, W, 3)
	ray_dir = target.reshape(N, S, D, H, W, 3).detach()
	axis = res.cyl_axes.view(N, 1, 1, 1, 1, 3)
	center = res.cyl_centers.view(N, 1, 1, 1, 1, 3)
	delta = center - xyz
	cross_dir = torch.cross(ray_dir, axis.expand_as(ray_dir), dim=-1)
	cross_sq = (cross_dir * cross_dir).sum(dim=-1)
	line_num = ((delta * cross_dir).sum(dim=-1) ** 2)
	line_dist_sq = line_num / cross_sq.clamp(min=1.0e-8)
	parallel = torch.cross(delta, axis.expand_as(delta), dim=-1)
	parallel_dist_sq = (parallel * parallel).sum(dim=-1)
	dist_sq = torch.where(cross_sq > 1.0e-6, line_dist_sq, parallel_dist_sq)
	axis_delta = delta - (delta * axis).sum(dim=-1, keepdim=True) * axis
	radius_sq = (axis_delta * axis_delta).sum(dim=-1).clamp(min=1.0)
	lm = dist_sq / radius_sq
	mask = (grad_mag.squeeze(0).squeeze(0) > 0.0).to(dtype=lm.dtype).reshape(N, S, D, H, W)
	mask = mask * ((ray_dir * ray_dir).sum(dim=-1) > 1.0e-6).to(dtype=lm.dtype)

	err, lm_c, mask_c = _candidate_mean(lm, mask, res=res)
	loss = _register_candidate_term("cyl_center", err)
	return loss, (lm_c,), (mask_c,)
