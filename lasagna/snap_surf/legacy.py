from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
import time

import torch
import torch.nn.functional as F

import model as fit_model
import opt_loss_station
import opt_loss_winding_density

from .config import SnapSurfConfig
from .state import _DirectionState, _SurfaceState
from .tensor import *

def _empty_grow_stats() -> dict[str, int | float]:
	return {
		"drop": 0,
		"ring": 0,
		"sup": 0,
		"tgt": 0,
		"dist": 0,
		"grid": 0,
		"ori": 0,
		"new": 0,
		"local": 0,
		"brute": 0,
		"front": 0,
		"brute_on": 0,
		"tested": 0,
		"gerr_n": 0,
		"gerr_sum": 0.0,
		"gerr_max": 0.0,
		"raw_map_n": 0,
		"raw_map_min": float("inf"),
		"raw_map_sum": 0.0,
		"raw_map_max": 0.0,
		"in_map_n": 0,
		"in_map_min": float("inf"),
		"in_map_sum": 0.0,
		"in_map_max": 0.0,
	}

def _add_grow_stats(dst: dict[str, int | float], src: dict[str, int | float]) -> None:
	for k in ("drop", "ring", "sup", "tgt", "dist", "grid", "ori", "new", "local", "brute", "front", "brute_on", "tested", "gerr_n", "gerr_sum", "raw_map_n", "raw_map_sum", "in_map_n", "in_map_sum"):
		dst[k] = dst.get(k, 0) + src.get(k, 0)
	dst["gerr_max"] = max(float(dst.get("gerr_max", 0.0)), float(src.get("gerr_max", 0.0)))
	dst["raw_map_min"] = min(float(dst.get("raw_map_min", float("inf"))), float(src.get("raw_map_min", float("inf"))))
	dst["raw_map_max"] = max(float(dst.get("raw_map_max", 0.0)), float(src.get("raw_map_max", 0.0)))
	dst["in_map_min"] = min(float(dst.get("in_map_min", float("inf"))), float(src.get("in_map_min", float("inf"))))
	dst["in_map_max"] = max(float(dst.get("in_map_max", 0.0)), float(src.get("in_map_max", 0.0)))

def _grow_direction(
	state: _DirectionState,
	*,
	source_xyz: torch.Tensor,
	source_valid: torch.Tensor,
	target_xyz: torch.Tensor,
	target_valid: torch.Tensor,
	normal_xyz: torch.Tensor,
	normal_from_source: bool,
	cfg: SnapSurfConfig,
) -> dict[str, int | float]:
	stats = _empty_grow_stats()
	if state.map is None or state.valid is None or state.source_shape is None or state.target_shape is None:
		return stats
	valid_b, map_b, source_valid_b = _batched_source_views(state, source_valid)
	base_valid = valid_b.clone()
	if int(base_valid.sum().detach().cpu()) == 0:
		return stats
	candidate_mask = _neighbor4_mask(base_valid) & source_valid_b & ~base_valid
	candidate_mask &= _seed_source_limit_mask(state, base_valid, radius=cfg.seed_radius)
	stats["ring"] = int(candidate_mask.sum().detach().cpu())
	if not bool(candidate_mask.any().detach().cpu()):
		return stats

	candidate_mask &= _neighbor4_mask(base_valid)
	stats["sup"] = int(candidate_mask.sum().detach().cpu())
	if not bool(candidate_mask.any().detach().cpu()):
		return stats

	cand_bidx = candidate_mask.nonzero(as_tuple=False)
	_, support_count, _ = _direct_predict_candidates_batched(
		state,
		valid_b=base_valid,
		map_b=map_b,
		candidate_bidx=cand_bidx,
		radius=cfg.affine_radius,
	)
	support_ok = support_count >= 1
	if not bool(support_ok.any().detach().cpu()):
		return stats
	cand_bidx = cand_bidx[support_ok]
	C = int(cand_bidx.shape[0])
	if state.source_rank == 3:
		source_idx = cand_bidx
	else:
		source_idx = cand_bidx[:, 1:]
	source_pos = _points_at_indices(source_xyz, source_idx)

	bases = _all_valid_target_quad_bases(target_valid.bool())
	if int(bases.shape[0]) == 0:
		return stats
	p00, p10, p01, p11 = _quad_corners_batched(target_xyz, bases)
	K = int(bases.shape[0])
	if normal_from_source:
		ref_normal = _points_at_indices(normal_xyz, source_idx)
		ref_normal = F.normalize(ref_normal, dim=-1, eps=1.0e-8)[:, None, :].expand(C, K, -1)
	else:
		ref_normal = _quad_average_normal_batched(normal_xyz, bases)
		ref_normal = ref_normal.unsqueeze(0).expand(C, K, -1)
	base_coord = bases.to(dtype=map_b.dtype).unsqueeze(0).expand(C, K, -1)
	coord, line_score, normal_abs = _closest_point_on_quad_along_normal_batched(
		source_pos[:, None, :],
		ref_normal,
		p00.unsqueeze(0),
		p10.unsqueeze(0),
		p01.unsqueeze(0),
		p11.unsqueeze(0),
		base_coord,
	)
	valid_choice = (
		torch.isfinite(source_pos).all(dim=-1)[:, None] &
		torch.isfinite(line_score) &
		torch.isfinite(normal_abs) &
		torch.isfinite(coord).all(dim=-1)
	)
	line_valid = torch.where(valid_choice, line_score, torch.full_like(line_score, float("inf")))
	best_line = line_valid.min(dim=1).values
	has_choice = torch.isfinite(best_line)
	if not bool(has_choice.any().detach().cpu()):
		return stats
	close_line = valid_choice & (line_score <= (best_line[:, None] + 1.0e-9))
	normal_tiebreak = torch.where(close_line, normal_abs, torch.full_like(normal_abs, float("inf")))
	best_k = torch.argmin(normal_tiebreak, dim=1)
	best_coord = coord[torch.arange(C, device=coord.device), best_k]
	accepted_mask = has_choice & torch.isfinite(best_coord).all(dim=-1)
	accepted_n = int(accepted_mask.sum().detach().cpu())
	stats["tgt"] = accepted_n
	stats["dist"] = accepted_n
	stats["grid"] = accepted_n
	stats["ori"] = accepted_n
	stats["new"] = accepted_n
	if accepted_n <= 0:
		return stats

	map_out = map_b.clone()
	valid_out = base_valid.clone()
	acc_bidx = cand_bidx[accepted_mask]
	map_out[acc_bidx[:, 0], acc_bidx[:, 1], acc_bidx[:, 2]] = best_coord[accepted_mask].to(dtype=map_b.dtype)
	valid_out[acc_bidx[:, 0], acc_bidx[:, 1], acc_bidx[:, 2]] = True
	_write_batched_state(state, valid_out, map_out)
	return stats

def _grow_until_stalled_direction(
	state: _DirectionState,
	*,
	source_xyz: torch.Tensor,
	source_valid: torch.Tensor,
	target_xyz: torch.Tensor,
	target_valid: torch.Tensor,
	normal_xyz: torch.Tensor,
	normal_from_source: bool,
	cfg: SnapSurfConfig,
	max_iters: int,
) -> tuple[int, dict[str, int | float]]:
	total = _empty_grow_stats()
	attempts = 0
	for _ in range(max(0, int(max_iters))):
		grow = _grow_direction(
			state,
			source_xyz=source_xyz,
			source_valid=source_valid,
			target_xyz=target_xyz,
			target_valid=target_valid,
			normal_xyz=normal_xyz,
			normal_from_source=normal_from_source,
			cfg=cfg,
		)
		attempts += 1
		_add_grow_stats(total, grow)
		if int(grow.get("new", 0)) <= 0:
			break
	return attempts, total

def _closest_external_seed_surface(
	*,
	seed: torch.Tensor,
	ext_xyz: torch.Tensor,
	ext_valid: torch.Tensor,
	ext_quad_valid: torch.Tensor,
	chunk_quads: int = 262144,
) -> tuple[tuple[int, int] | None, torch.Tensor | None, float]:
	"""Closest point on any valid external tifxyz quad to the seed."""
	if ext_valid.numel() == 0 or not bool(ext_valid.any().detach().cpu()):
		return None, None, float("inf")
	if ext_xyz.ndim != 3 or int(ext_xyz.shape[-1]) != 3:
		return None, None, float("inf")
	H, W, _ = ext_xyz.shape
	if H < 2 or W < 2 or ext_quad_valid.numel() == 0 or not bool(ext_quad_valid.any().detach().cpu()):
		pts = ext_xyz[ext_valid & torch.isfinite(ext_xyz).all(dim=-1)]
		if pts.numel() == 0:
			return None, None, float("inf")
		dist2 = (pts - seed.view(1, 3)).square().sum(dim=-1)
		best = int(torch.argmin(dist2).detach().cpu())
		return None, pts[best].detach(), math.sqrt(float(dist2[best].detach().cpu()))

	valid_ids = ext_quad_valid.reshape(-1).nonzero(as_tuple=False).flatten()
	if valid_ids.numel() == 0:
		return None, None, float("inf")
	Wq = W - 1
	rows_all = torch.div(valid_ids, Wq, rounding_mode="floor")
	cols_all = valid_ids - rows_all * Wq
	best = torch.full((), float("inf"), device=ext_xyz.device, dtype=ext_xyz.dtype)
	best_hw: tuple[int, int] | None = None
	best_point: torch.Tensor | None = None
	chunk = max(1, int(chunk_quads))
	for start in range(0, int(valid_ids.numel()), chunk):
		end = min(start + chunk, int(valid_ids.numel()))
		rows = rows_all[start:end]
		cols = cols_all[start:end]
		p00 = ext_xyz[rows, cols]
		p10 = ext_xyz[rows + 1, cols]
		p01 = ext_xyz[rows, cols + 1]
		p11 = ext_xyz[rows + 1, cols + 1]
		finite = (
			torch.isfinite(p00).all(dim=-1) &
			torch.isfinite(p10).all(dim=-1) &
			torch.isfinite(p01).all(dim=-1) &
			torch.isfinite(p11).all(dim=-1)
		)
		if not bool(finite.any().detach().cpu()):
			continue
		rows_f = rows[finite]
		cols_f = cols[finite]
		p00 = p00[finite]
		p10 = p10[finite]
		p01 = p01[finite]
		p11 = p11[finite]
		cp0, _ = opt_loss_station._closest_points_on_triangles(seed, p00, p10, p11)
		cp1, _ = opt_loss_station._closest_points_on_triangles(seed, p00, p11, p01)
		d20 = (cp0 - seed.view(1, 3)).square().sum(dim=-1)
		d21 = (cp1 - seed.view(1, 3)).square().sum(dim=-1)
		use_first = d20 <= d21
		d2 = torch.where(use_first, d20, d21)
		local = int(torch.argmin(d2).detach().cpu())
		local_best = d2[local]
		if float(local_best.detach().cpu()) < float(best.detach().cpu()):
			best = local_best
			best_hw = (int(rows_f[local].detach().cpu()), int(cols_f[local].detach().cpu()))
			best_point = (cp0 if bool(use_first[local].detach().cpu()) else cp1)[local].detach()
	if not bool(torch.isfinite(best).detach().cpu()):
		return None, None, float("inf")
	return best_hw, best_point, math.sqrt(float(best.detach().cpu()))

def _closest_model_surface_quad(
	*,
	point: torch.Tensor,
	model_xyz: torch.Tensor,
	model_valid: torch.Tensor,
) -> tuple[tuple[int, int, int] | None, float]:
	D, H, W, _ = model_xyz.shape
	if H < 2 or W < 2:
		return None, float("inf")
	quad_valid = (
		model_valid[:, :-1, :-1] &
		model_valid[:, 1:, :-1] &
		model_valid[:, :-1, 1:] &
		model_valid[:, 1:, 1:]
	)
	valid_ids = quad_valid.reshape(-1).nonzero(as_tuple=False).flatten()
	if valid_ids.numel() == 0:
		return None, float("inf")
	Hq, Wq = H - 1, W - 1
	d = torch.div(valid_ids, Hq * Wq, rounding_mode="floor")
	rem = valid_ids - d * Hq * Wq
	h = torch.div(rem, Wq, rounding_mode="floor")
	w = rem - h * Wq
	p00 = model_xyz[d, h, w]
	p10 = model_xyz[d, h + 1, w]
	p01 = model_xyz[d, h, w + 1]
	p11 = model_xyz[d, h + 1, w + 1]
	cp0, _ = opt_loss_station._closest_points_on_triangles(point, p00, p10, p11)
	cp1, _ = opt_loss_station._closest_points_on_triangles(point, p00, p11, p01)
	d2 = torch.minimum(
		(cp0 - point.view(1, 3)).square().sum(dim=-1),
		(cp1 - point.view(1, 3)).square().sum(dim=-1),
	)
	best = int(torch.argmin(d2).detach().cpu())
	return (
		int(d[best].detach().cpu()),
		int(h[best].detach().cpu()),
		int(w[best].detach().cpu()),
	), math.sqrt(float(d2[best].detach().cpu()))

def _closest_point_uv_on_model_quad(
	*,
	point: torch.Tensor,
	model_xyz: torch.Tensor,
	model_quad: tuple[int, int, int],
) -> tuple[torch.Tensor, torch.Tensor, float]:
	d, h, w = model_quad
	p00 = model_xyz[d, h, w].view(1, 3)
	p10 = model_xyz[d, h + 1, w].view(1, 3)
	p01 = model_xyz[d, h, w + 1].view(1, 3)
	p11 = model_xyz[d, h + 1, w + 1].view(1, 3)
	cp0, bary0 = opt_loss_station._closest_points_on_triangles(point, p00, p10, p11)
	cp1, bary1 = opt_loss_station._closest_points_on_triangles(point, p00, p11, p01)
	d20 = (cp0 - point.view(1, 3)).square().sum(dim=-1)
	d21 = (cp1 - point.view(1, 3)).square().sum(dim=-1)
	if float(d20[0].detach().cpu()) <= float(d21[0].detach().cpu()):
		b = bary0[0]
		uv = torch.stack([
			torch.as_tensor(float(h), device=point.device, dtype=point.dtype) + b[1] + b[2],
			torch.as_tensor(float(w), device=point.device, dtype=point.dtype) + b[2],
		], dim=0)
		return cp0[0].detach(), uv.detach(), math.sqrt(float(d20[0].detach().cpu()))
	b = bary1[0]
	uv = torch.stack([
		torch.as_tensor(float(h), device=point.device, dtype=point.dtype) + b[1],
		torch.as_tensor(float(w), device=point.device, dtype=point.dtype) + b[1] + b[2],
	], dim=0)
	return cp1[0].detach(), uv.detach(), math.sqrt(float(d21[0].detach().cpu()))

def _map_init_seed_quad_uv_for_points(
	points: torch.Tensor,
	*,
	ext_xyz: torch.Tensor,
	model_xyz: torch.Tensor,
	ext_quad: tuple[int, int],
	model_quad: tuple[int, int, int],
	transform: tuple[tuple[int, int], ...],
	ext_anchor: torch.Tensor,
	model_anchor_uv: torch.Tensor,
	eps: float = 1.0e-8,
) -> tuple[torch.Tensor, torch.Tensor, str | None]:
	out = torch.full((*points.shape[:-1], 2), float("nan"), device=points.device, dtype=points.dtype)
	ok = torch.zeros(points.shape[:-1], device=points.device, dtype=torch.bool)
	if points.numel() == 0:
		return out, ok, None
	eh, ew = ext_quad
	d, mh, mw = model_quad
	ext_pts = torch.stack([ext_xyz[eh + th, ew + tw] for th, tw in transform], dim=0)
	model_pts = torch.stack([model_xyz[d, mh + sh, mw + sw] for sh, sw in _CORNERS_2D], dim=0)
	if not bool(torch.isfinite(ext_pts).all().detach().cpu()):
		return out, ok, "non-finite external seed quad"
	if not bool(torch.isfinite(model_pts).all().detach().cpu()):
		return out, ok, "non-finite model seed quad"
	if not bool(torch.isfinite(ext_anchor).all().detach().cpu()):
		return out, ok, "non-finite external seed anchor"
	if not bool(torch.isfinite(model_anchor_uv).all().detach().cpu()):
		return out, ok, "non-finite model seed anchor"

	ext_h = ext_pts[1] - ext_pts[0]
	ext_w = ext_pts[2] - ext_pts[0]
	model_h = model_pts[1] - model_pts[0]
	model_w = model_pts[2] - model_pts[0]
	ext_h_len = ext_h.norm()
	ext_w_len = ext_w.norm()
	model_h_len = model_h.norm()
	model_w_len = model_w.norm()
	lengths = torch.stack([ext_h_len, ext_w_len, model_h_len, model_w_len])
	if not bool(torch.isfinite(lengths).all().detach().cpu()):
		return out, ok, "non-finite seed quad edge length"
	if float(lengths.min().detach().cpu()) <= float(eps):
		return out, ok, "degenerate seed quad edge"

	ext_basis = torch.stack([ext_h / ext_h_len, ext_w / ext_w_len], dim=1)
	model_unit = torch.stack([model_h / model_h_len, model_w / model_w_len], dim=1)
	model_edges = torch.stack([model_h, model_w], dim=1)
	flat = points.reshape(-1, 3)
	point_ok = torch.isfinite(flat).all(dim=-1)
	if not bool(point_ok.any().detach().cpu()):
		return out, ok, None
	rel = (flat[point_ok] - ext_anchor.view(1, 3)).transpose(0, 1)
	try:
		ext_coeff = torch.linalg.lstsq(ext_basis, rel).solution.transpose(0, 1)
		model_disp = (ext_coeff @ model_unit.transpose(0, 1)).transpose(0, 1)
		uv_delta = torch.linalg.lstsq(model_edges, model_disp).solution.transpose(0, 1)
	except RuntimeError as exc:
		return out, ok, f"seed quad solve failed: {exc}"
	uv = model_anchor_uv.view(1, 2) + uv_delta
	local_ok = torch.isfinite(uv).all(dim=-1)
	flat_out = out.reshape(-1, 2)
	flat_ok = ok.reshape(-1)
	idx = point_ok.nonzero(as_tuple=False).flatten()
	flat_out[idx] = torch.where(local_ok.unsqueeze(-1), uv, flat_out[idx])
	flat_ok[idx] = local_ok
	return out, ok, None

def _choose_seed_transform(
	*,
	model_xyz: torch.Tensor,
	ext_xyz: torch.Tensor,
	model_quad: tuple[int, int, int],
	ext_quad: tuple[int, int],
	cfg: SnapSurfConfig,
) -> tuple[tuple[tuple[int, int], ...], int]:
	transforms = [_dihedral_transforms()[0]] if cfg.orientation in {"identity", "none"} else _dihedral_transforms()
	d, mh, mw = model_quad
	eh, ew = ext_quad
	model_pts = torch.stack([model_xyz[d, mh + sh, mw + sw] for sh, sw in _CORNERS_2D], dim=0)
	model_norm = _normalized_seed_quad(model_pts)
	best = transforms[0]
	best_score = float("inf")
	for transform in transforms:
		ext_pts = torch.stack([ext_xyz[eh + th, ew + tw] for th, tw in transform], dim=0)
		ext_norm = _normalized_seed_quad(ext_pts)
		score = float((model_norm - ext_norm).norm(dim=-1).sum().detach().cpu())
		if score < best_score:
			best_score = score
			best = transform
	return best, _transform_det_sign(best)

def _huber(residual: torch.Tensor, *, delta: float) -> torch.Tensor:
	abs_r = residual.abs()
	d = float(delta)
	return torch.where(abs_r <= d, 0.5 * residual.square(), d * (abs_r - 0.5 * d))

def _huber_grad(residual: torch.Tensor, *, delta: float) -> torch.Tensor:
	d = float(delta)
	return residual.clamp(min=-d, max=d)

def _empty_residual_stats() -> dict[str, float]:
	return {"n": 0.0, "sum": 0.0, "abs_sum": 0.0, "abs_max": 0.0, "toward_sum": 0.0}

def _residual_stats(raw_residual: torch.Tensor, scaled_residual: torch.Tensor, *, delta: float) -> dict[str, float]:
	if raw_residual.numel() == 0:
		return _empty_residual_stats()
	finite = torch.isfinite(raw_residual) & torch.isfinite(scaled_residual)
	if not bool(finite.any().detach().cpu()):
		return _empty_residual_stats()
	raw = raw_residual.detach()[finite]
	scaled = scaled_residual.detach()[finite]
	abs_r = raw.abs()
	grad_scaled = _huber_grad(scaled, delta=delta)
	# Positive means the gradient-descent update points toward the matched proxy plane.
	toward = grad_scaled * scaled
	return {
		"n": float(raw.numel()),
		"sum": float(raw.sum().cpu()),
		"abs_sum": float(abs_r.sum().cpu()),
		"abs_max": float(abs_r.max().cpu()),
		"toward_sum": float(toward.sum().cpu()),
	}

def _seed_model_sheet_mask(
	*,
	model_valid: torch.Tensor,
	model_normals: torch.Tensor,
	seed_quad: tuple[int, int, int] | None,
) -> torch.Tensor:
	if seed_quad is None:
		return torch.zeros_like(model_valid, dtype=torch.bool)
	D = int(model_valid.shape[0])
	d0 = int(seed_quad[0])
	dd = torch.arange(D, device=model_valid.device).view(D, 1, 1)
	normal_ok = (
		torch.isfinite(model_normals).all(dim=-1) &
		(model_normals.norm(dim=-1) > 1.0e-8)
	)
	return model_valid.bool() & normal_ok & (dd == d0)

def _normalized_model_to_ext_map(map_b: torch.Tensor) -> torch.Tensor:
	D, H, W = (int(v) for v in map_b.shape[:3])
	source_hw = _source_hw_grid(n=D, h=H, w=W, device=map_b.device, dtype=map_b.dtype)
	return map_b - source_hw

def _neighbor_min_mapping_distance(
	*,
	center_valid: torch.Tensor,
	neighbor_valid: torch.Tensor,
	norm_map: torch.Tensor,
	normal_dist: torch.Tensor | None = None,
	max_normal_ratio: float | None = None,
	normal_distance_floor: float = 10.0,
) -> torch.Tensor:
	D, H, W = (int(v) for v in center_valid.shape)
	if H == 0 or W == 0:
		return torch.empty(D, H, W, device=norm_map.device, dtype=norm_map.dtype)
	finite_map = torch.isfinite(norm_map).all(dim=-1)
	center_ok = center_valid.bool() & finite_map
	neighbor_ok = neighbor_valid.bool() & finite_map
	ratio_patch_ok = None
	if normal_dist is not None and max_normal_ratio is not None:
		normal_dist = normal_dist.to(device=norm_map.device, dtype=norm_map.dtype).abs()
		finite_dist = torch.isfinite(normal_dist)
		center_ok = center_ok & finite_dist
		neighbor_ok = neighbor_ok & finite_dist
		dist_safe = torch.where(neighbor_ok, normal_dist, torch.zeros_like(normal_dist))
		dist_patch = F.unfold(
			dist_safe.unsqueeze(1),
			kernel_size=3,
			padding=1,
		).transpose(1, 2).reshape(D, H, W, 9)
		floor = max(1.0e-6, float(normal_distance_floor))
		center_dist = normal_dist.clamp_min(floor).unsqueeze(-1)
		neighbor_dist = dist_patch.clamp_min(floor)
		ratio = torch.maximum(center_dist / neighbor_dist, neighbor_dist / center_dist)
		ratio_patch_ok = torch.isfinite(ratio) & (ratio <= float(max_normal_ratio))
	map_safe = torch.where(neighbor_ok.unsqueeze(-1), norm_map, torch.zeros_like(norm_map))
	map_patch = F.unfold(
		map_safe.permute(0, 3, 1, 2),
		kernel_size=3,
		padding=1,
	).transpose(1, 2).reshape(D, H, W, 2, 9)
	valid_patch = F.unfold(
		neighbor_ok.to(dtype=norm_map.dtype).unsqueeze(1),
		kernel_size=3,
		padding=1,
	).transpose(1, 2).reshape(D, H, W, 9) > 0.0
	valid_patch[..., 4] = False
	if ratio_patch_ok is not None:
		ratio_patch_ok[..., 4] = False
		valid_patch = valid_patch & ratio_patch_ok
	dist = (map_patch - norm_map.unsqueeze(-1)).square().sum(dim=3).sqrt()
	dist = torch.where(
		center_ok.unsqueeze(-1) & valid_patch,
		dist,
		torch.full_like(dist, float("inf")),
	)
	return dist.min(dim=-1).values

def _mapping_distance_stats(
	*,
	center_valid: torch.Tensor,
	neighbor_valid: torch.Tensor,
	norm_map: torch.Tensor,
) -> dict[str, int | float]:
	dist = _neighbor_min_mapping_distance(
		center_valid=center_valid,
		neighbor_valid=neighbor_valid,
		norm_map=norm_map,
	)
	finite = center_valid.bool() & torch.isfinite(dist)
	if not bool(finite.any().detach().cpu()):
		return {"n": 0, "min": float("inf"), "sum": 0.0, "max": 0.0}
	vals = dist[finite].detach()
	return {
		"n": int(vals.numel()),
		"min": float(vals.min().cpu()),
		"sum": float(vals.sum().cpu()),
		"max": float(vals.max().cpu()),
	}

def _seed_quad_corner_mask(
	shape: tuple[int, int, int],
	seed_quad: tuple[int, int, int],
	*,
	device: torch.device,
) -> torch.Tensor:
	D, H, W = (int(v) for v in shape)
	mask = torch.zeros(D, H, W, device=device, dtype=torch.bool)
	d0, h0, w0 = (int(v) for v in seed_quad)
	if d0 < 0 or d0 >= D:
		return mask
	h1 = min(H, h0 + 2)
	w1 = min(W, w0 + 2)
	h0 = max(0, h0)
	w0 = max(0, w0)
	if h0 < h1 and w0 < w1:
		mask[d0, h0:h1, w0:w1] = True
	return mask

def _closest_seed_source_mask(
	*,
	raw_valid: torch.Tensor,
	model_xyz: torch.Tensor,
	seed: torch.Tensor,
) -> torch.Tensor:
	mask = torch.zeros_like(raw_valid, dtype=torch.bool)
	if not bool(raw_valid.any().detach().cpu()):
		return mask
	finite = raw_valid.bool() & torch.isfinite(model_xyz).all(dim=-1)
	if not bool(finite.any().detach().cpu()):
		return mask
	dist2 = (model_xyz - seed.view(1, 1, 1, 3)).square().sum(dim=-1)
	dist2 = torch.where(finite, dist2, torch.full_like(dist2, float("inf")))
	best_flat = torch.argmin(dist2.reshape(-1))
	best_dist = dist2.reshape(-1)[best_flat]
	if not bool(torch.isfinite(best_dist).detach().cpu()):
		return mask
	mask.reshape(-1)[best_flat] = True
	return mask

def _seeded_mapping_inlier_filter(
	*,
	raw_valid: torch.Tensor,
	raw_map: torch.Tensor,
	seed_quad: tuple[int, int, int] | None = None,
	initial_inlier: torch.Tensor | None = None,
	max_distance: float,
	normal_dist: torch.Tensor | None = None,
	max_normal_ratio: float | None = None,
	normal_distance_floor: float = 10.0,
) -> tuple[torch.Tensor, dict[str, int | float]]:
	norm_map = _normalized_model_to_ext_map(raw_map)
	raw_valid = raw_valid.bool() & torch.isfinite(norm_map).all(dim=-1)
	if normal_dist is not None:
		normal_dist = normal_dist.to(device=raw_valid.device, dtype=raw_map.dtype).abs()
		raw_valid = raw_valid & torch.isfinite(normal_dist)
	if initial_inlier is not None:
		inlier = raw_valid & initial_inlier.to(device=raw_valid.device).bool()
	elif seed_quad is not None:
		seed_mask = _seed_quad_corner_mask(
			tuple(int(v) for v in raw_valid.shape),
			seed_quad,
			device=raw_valid.device,
		)
		inlier = raw_valid & seed_mask
	else:
		inlier = torch.zeros_like(raw_valid)
	D, H, W = (int(v) for v in raw_valid.shape)
	if bool(inlier.any().detach().cpu()):
		for _ in range(max(1, D + H + W)):
			candidate = raw_valid & ~inlier
			if not bool(candidate.any().detach().cpu()):
				break
			min_dist = _neighbor_min_mapping_distance(
				center_valid=candidate,
				neighbor_valid=inlier,
				norm_map=norm_map,
				normal_dist=normal_dist,
				max_normal_ratio=max_normal_ratio,
				normal_distance_floor=normal_distance_floor,
			)
			new_inlier = candidate & (min_dist < float(max_distance))
			if not bool(new_inlier.any().detach().cpu()):
				break
			inlier = inlier | new_inlier
	return inlier, {}

def _valid_ext_quad_bases(ext_valid: torch.Tensor, ext_quad_valid: torch.Tensor) -> torch.Tensor:
	H, W = int(ext_valid.shape[0]), int(ext_valid.shape[1])
	if H < 2 or W < 2:
		return torch.empty(0, 2, device=ext_valid.device, dtype=torch.long)
	corner_quad_valid = (
		ext_valid[:-1, :-1].bool() &
		ext_valid[1:, :-1].bool() &
		ext_valid[:-1, 1:].bool() &
		ext_valid[1:, 1:].bool()
	)
	if tuple(ext_quad_valid.shape) == tuple(corner_quad_valid.shape):
		corner_quad_valid = corner_quad_valid & ext_quad_valid.bool()
	return corner_quad_valid.nonzero(as_tuple=False)

def _ext_quad_bases_around_coords(
	coords: torch.Tensor,
	shape: tuple[int, int],
	*,
	search_ring: int,
) -> tuple[torch.Tensor, torch.Tensor]:
	C = int(coords.shape[0])
	H, W = (int(v) for v in shape)
	device = coords.device
	if C == 0 or H < 2 or W < 2:
		return (
			torch.empty(C, 0, 2, device=device, dtype=torch.long),
			torch.zeros(C, 0, device=device, dtype=torch.bool),
		)
	r = max(0, int(search_ring))
	offs = torch.arange(-r, r + 1, device=device, dtype=torch.long)
	off_h, off_w = torch.meshgrid(offs, offs, indexing="ij")
	hw_offsets = torch.stack([off_h.reshape(-1), off_w.reshape(-1)], dim=-1)
	K = int(hw_offsets.shape[0])
	finite = torch.isfinite(coords).all(dim=-1)
	safe_coords = torch.where(torch.isfinite(coords), coords, torch.zeros_like(coords))
	base_hw = torch.floor(safe_coords).to(dtype=torch.long)
	base_hw = torch.stack(
		[
			base_hw[:, 0].clamp(0, H - 2),
			base_hw[:, 1].clamp(0, W - 2),
		],
		dim=-1,
	)
	bases = base_hw[:, None, :] + hw_offsets.view(1, K, 2)
	in_bounds = (
		finite[:, None] &
		(bases[..., 0] >= 0) &
		(bases[..., 0] <= H - 2) &
		(bases[..., 1] >= 0) &
		(bases[..., 1] <= W - 2)
	)
	return bases, in_bounds

def _ext_quad_valid_at_bases(
	ext_valid: torch.Tensor,
	ext_quad_valid: torch.Tensor,
	bases: torch.Tensor,
	in_bounds: torch.Tensor,
) -> torch.Tensor:
	if bases.numel() == 0:
		return torch.zeros(bases.shape[:-1], device=ext_valid.device, dtype=torch.bool)
	H, W = int(ext_valid.shape[0]), int(ext_valid.shape[1])
	h = bases[..., 0].clamp(0, max(0, H - 2))
	w = bases[..., 1].clamp(0, max(0, W - 2))
	ok = (
		ext_valid[h, w] &
		ext_valid[h + 1, w] &
		ext_valid[h, w + 1] &
		ext_valid[h + 1, w + 1]
	)
	if tuple(ext_quad_valid.shape) == (max(0, H - 1), max(0, W - 1)):
		ok = ok & ext_quad_valid[h, w]
	return ok & in_bounds

def _intersect_ray_quad_candidates(
	*,
	source_pos: torch.Tensor,
	source_normals: torch.Tensor,
	bases: torch.Tensor,
	p00: torch.Tensor,
	p10: torch.Tensor,
	p01: torch.Tensor,
	p11: torch.Tensor,
	cfg: SnapSurfConfig,
	candidate_valid: torch.Tensor | None = None,
	hint_u: torch.Tensor | None = None,
	hint_v: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int | float]]:
	C = int(source_pos.shape[0])
	device = source_pos.device
	dtype = source_pos.dtype
	coords_empty = torch.full((C, 2), float("nan"), device=device, dtype=dtype)
	accepted_empty = torch.zeros(C, device=device, dtype=torch.bool)
	stats: dict[str, int | float] = {
		"target_hit": 0,
		"distance_hit": 0,
		"accepted": 0,
		"tested": 0,
		"line_err_sum": 0.0,
		"line_err_max": 0.0,
	}
	if C == 0:
		return coords_empty, accepted_empty, stats
	if bases.ndim != 3:
		raise ValueError(f"expected per-source candidate bases, got shape {tuple(bases.shape)}")
	K = int(bases.shape[1])
	stats["tested"] = C * K
	if K == 0:
		return coords_empty, accepted_empty, stats
	if candidate_valid is None:
		candidate_valid = torch.ones(C, K, device=device, dtype=torch.bool)
	else:
		candidate_valid = candidate_valid.bool()
	if hint_u is None:
		hint_u = torch.full((C, K), 0.5, device=device, dtype=dtype)
	if hint_v is None:
		hint_v = torch.full((C, K), 0.5, device=device, dtype=dtype)

	n = F.normalize(source_normals, dim=-1, eps=1.0e-8)
	O = source_pos[:, None, :]
	N = n[:, None, :]
	u, v = opt_loss_winding_density.ray_bilinear_intersect_refined(
		O,
		N,
		p00,
		p10,
		p01,
		p11,
		hint_u,
		hint_v,
		passes=2,
	)
	a = p10 - p00
	b = p01 - p00
	c = p11 - p10 - p01 + p00
	q = p00 + u.unsqueeze(-1) * a + v.unsqueeze(-1) * b + (u * v).unsqueeze(-1) * c
	delta = q - O
	signed = (delta * N).sum(dim=-1)
	abs_signed = signed.abs()
	line_delta = delta - signed.unsqueeze(-1) * N
	line_err = line_delta.norm(dim=-1)
	finite = (
		candidate_valid &
		torch.isfinite(source_pos).all(dim=-1)[:, None] &
		torch.isfinite(n).all(dim=-1)[:, None] &
		(n.norm(dim=-1) > 1.0e-8)[:, None] &
		torch.isfinite(q).all(dim=-1) &
		torch.isfinite(u) &
		torch.isfinite(v) &
		torch.isfinite(abs_signed) &
		torch.isfinite(line_err)
	)
	uv_tol = 1.0e-4
	uv_ok = (
		(u >= -uv_tol) & (u <= 1.0 + uv_tol) &
		(v >= -uv_tol) & (v <= 1.0 + uv_tol)
	)
	hit = finite & uv_ok
	has_hit = hit.any(dim=1)
	stats["target_hit"] = int(has_hit.sum().detach().cpu())
	if not bool(has_hit.any().detach().cpu()):
		return coords_empty, accepted_empty, stats

	line_hit = hit & (line_err <= float(cfg.ray_residual))
	has_line_hit = line_hit.any(dim=1)
	if not bool(has_line_hit.any().detach().cpu()):
		return coords_empty, accepted_empty, stats

	score = torch.where(line_hit, abs_signed, torch.full_like(abs_signed, float("inf")))
	best_k = torch.argmin(score, dim=1)
	row = torch.arange(C, device=device)
	best_dist = score[row, best_k]
	best_line_err = line_err[row, best_k]
	accepted = has_line_hit & torch.isfinite(best_dist)
	stats["distance_hit"] = int(accepted.sum().detach().cpu())
	stats["accepted"] = int(accepted.sum().detach().cpu())
	best_bases = bases[row, best_k].to(dtype=dtype)
	best_u = u[row, best_k].clamp(0.0, 1.0)
	best_v = v[row, best_k].clamp(0.0, 1.0)
	coords = best_bases + torch.stack([best_u, best_v], dim=-1)
	coords = torch.where(accepted.unsqueeze(-1), coords, torch.full_like(coords, float("nan")))
	if bool(accepted.any().detach().cpu()):
		err = best_line_err[accepted].detach()
		stats["line_err_sum"] = float(err.sum().cpu())
		stats["line_err_max"] = float(err.max().cpu())
	return coords, accepted, stats

def _intersect_model_points_with_ext_surface_chunk(
	*,
	source_pos: torch.Tensor,
	source_normals: torch.Tensor,
	bases: torch.Tensor,
	p00: torch.Tensor,
	p10: torch.Tensor,
	p01: torch.Tensor,
	p11: torch.Tensor,
	cfg: SnapSurfConfig,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int | float]]:
	C = int(source_pos.shape[0])
	device = source_pos.device
	dtype = source_pos.dtype
	coords_empty = torch.full((C, 2), float("nan"), device=device, dtype=dtype)
	accepted_empty = torch.zeros(C, device=device, dtype=torch.bool)
	stats: dict[str, int | float] = {
		"target_hit": 0,
		"distance_hit": 0,
		"accepted": 0,
		"tested": 0,
		"line_err_sum": 0.0,
		"line_err_max": 0.0,
	}
	if C == 0:
		return coords_empty, accepted_empty, stats

	K = int(bases.shape[0])
	stats["tested"] = C * K
	if K == 0:
		return coords_empty, accepted_empty, stats
	return _intersect_ray_quad_candidates(
		source_pos=source_pos,
		source_normals=source_normals,
		bases=bases.view(1, K, 2).expand(C, K, 2),
		p00=p00.view(1, K, 3).expand(C, K, 3),
		p10=p10.view(1, K, 3).expand(C, K, 3),
		p01=p01.view(1, K, 3).expand(C, K, 3),
		p11=p11.view(1, K, 3).expand(C, K, 3),
		cfg=cfg,
	)

def _intersect_model_points_with_ext_surface_near_coords(
	*,
	source_pos: torch.Tensor,
	source_normals: torch.Tensor,
	pred_coords: torch.Tensor,
	ext_xyz: torch.Tensor,
	ext_valid: torch.Tensor,
	ext_quad_valid: torch.Tensor,
	cfg: SnapSurfConfig,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int | float]]:
	C = int(source_pos.shape[0])
	device = source_pos.device
	dtype = source_pos.dtype
	coords_empty = torch.full((C, 2), float("nan"), device=device, dtype=dtype)
	accepted_empty = torch.zeros(C, device=device, dtype=torch.bool)
	if C == 0:
		return coords_empty, accepted_empty, {
			"target_hit": 0,
			"distance_hit": 0,
			"accepted": 0,
			"tested": 0,
			"line_err_sum": 0.0,
			"line_err_max": 0.0,
		}
	H, W = int(ext_xyz.shape[0]), int(ext_xyz.shape[1])
	if H < 2 or W < 2:
		return coords_empty, accepted_empty, {
			"target_hit": 0,
			"distance_hit": 0,
			"accepted": 0,
			"tested": 0,
			"line_err_sum": 0.0,
			"line_err_max": 0.0,
		}
	stats: dict[str, int | float] = {
		"target_hit": 0,
		"distance_hit": 0,
		"accepted": 0,
		"tested": C,
		"line_err_sum": 0.0,
		"line_err_max": 0.0,
	}
	n_all = F.normalize(source_normals, dim=-1, eps=1.0e-8)
	finite_pred = torch.isfinite(pred_coords).all(dim=-1)
	source_ok_all = (
		torch.isfinite(source_pos).all(dim=-1) &
		torch.isfinite(n_all).all(dim=-1) &
		(n_all.norm(dim=-1) > 1.0e-8)
	)
	pred_coord_ok = _quad_valid_at_coords(
		ext_valid.bool(),
		pred_coords,
		tuple(int(v) for v in ext_xyz.shape[:2]),
	) & finite_pred
	if tuple(ext_quad_valid.shape) == (max(0, H - 1), max(0, W - 1)):
		pred_bases, pred_in_bounds = _ext_quad_bases_around_coords(
			pred_coords,
			tuple(int(v) for v in ext_xyz.shape[:2]),
			search_ring=0,
		)
		pred_coord_ok &= _ext_quad_valid_at_bases(
			ext_valid.bool(),
			ext_quad_valid.bool(),
			pred_bases,
			pred_in_bounds,
		).reshape(C)
	safe_pred_for_sample = torch.where(torch.isfinite(pred_coords), pred_coords, torch.zeros_like(pred_coords))
	q_pred = _sample_surface_grid(ext_xyz, safe_pred_for_sample)
	delta_pred = q_pred - source_pos
	signed_pred = (delta_pred * n_all).sum(dim=-1)
	line_err_pred = (delta_pred - signed_pred.unsqueeze(-1) * n_all).norm(dim=-1)
	pred_accept = (
		source_ok_all &
		pred_coord_ok &
		torch.isfinite(q_pred).all(dim=-1) &
		torch.isfinite(line_err_pred) &
		(line_err_pred <= float(cfg.ray_residual))
	)
	if bool(pred_accept.any().detach().cpu()):
		coords_empty[pred_accept] = pred_coords[pred_accept]
		accepted_empty[pred_accept] = True
		err = line_err_pred[pred_accept].detach()
		stats["target_hit"] += int(pred_accept.sum().detach().cpu())
		stats["distance_hit"] += int(pred_accept.sum().detach().cpu())
		stats["accepted"] += int(pred_accept.sum().detach().cpu())
		stats["line_err_sum"] += float(err.sum().cpu())
		stats["line_err_max"] = max(float(stats["line_err_max"]), float(err.max().cpu()))
	remaining = ~pred_accept
	if not bool(remaining.any().detach().cpu()):
		return coords_empty, accepted_empty, stats

	out_coords = coords_empty
	out_accepted = accepted_empty
	rem_idx = remaining.nonzero(as_tuple=False).flatten()
	source_pos = source_pos[rem_idx]
	source_normals = source_normals[rem_idx]
	pred_coords = pred_coords[rem_idx]
	C = int(source_pos.shape[0])
	stats["tested"] += 2 * C
	finite_pred = torch.isfinite(pred_coords).all(dim=-1)
	safe_pred = torch.where(torch.isfinite(pred_coords), pred_coords, torch.zeros_like(pred_coords))
	base0_h = torch.floor(safe_pred[:, 0].clamp(0.0, float(H - 1))).clamp(0, H - 2).long()
	base0_w = torch.floor(safe_pred[:, 1].clamp(0.0, float(W - 1))).clamp(0, W - 2).long()
	base0 = torch.stack([base0_h, base0_w], dim=-1)
	frac0_h = (safe_pred[:, 0] - base0_h.to(dtype=dtype)).clamp(0.0, 1.0)
	frac0_w = (safe_pred[:, 1] - base0_w.to(dtype=dtype)).clamp(0.0, 1.0)
	base0_valid = _ext_quad_valid_at_bases(
		ext_valid.bool(),
		ext_quad_valid.bool(),
		base0.view(C, 1, 2),
		finite_pred.view(C, 1),
	).view(C)

	n = F.normalize(source_normals, dim=-1, eps=1.0e-8)
	source_ok = (
		torch.isfinite(source_pos).all(dim=-1) &
		torch.isfinite(n).all(dim=-1) &
		(n.norm(dim=-1) > 1.0e-8)
	)
	p00, p10, p01, p11 = _quad_corners_batched(ext_xyz, base0)
	u1, v1 = fit_model.Model3D._ray_bilinear_intersect(
		source_pos,
		n,
		p00,
		p10,
		p01,
		p11,
		frac0_h,
		frac0_w,
	)
	pass1_valid = source_ok & base0_valid & torch.isfinite(u1) & torch.isfinite(v1)
	coord1_h_raw = base0_h.to(dtype=dtype) + u1
	coord1_w_raw = base0_w.to(dtype=dtype) + v1
	coord1_h = torch.where(pass1_valid, coord1_h_raw, torch.zeros_like(coord1_h_raw)).clamp(0.0, float(H - 1))
	coord1_w = torch.where(pass1_valid, coord1_w_raw, torch.zeros_like(coord1_w_raw)).clamp(0.0, float(W - 1))
	base1_h = torch.floor(coord1_h).clamp(0, H - 2).long()
	base1_w = torch.floor(coord1_w).clamp(0, W - 2).long()
	base1 = torch.stack([base1_h, base1_w], dim=-1)
	frac1_h = coord1_h - base1_h.to(dtype=dtype)
	frac1_w = coord1_w - base1_w.to(dtype=dtype)
	base1_valid = _ext_quad_valid_at_bases(
		ext_valid.bool(),
		ext_quad_valid.bool(),
		base1.view(C, 1, 2),
		pass1_valid.view(C, 1),
	).view(C)

	p00, p10, p01, p11 = _quad_corners_batched(ext_xyz, base1)
	u2, v2 = fit_model.Model3D._ray_bilinear_intersect(
		source_pos,
		n,
		p00,
		p10,
		p01,
		p11,
		frac1_h,
		frac1_w,
	)
	a = p10 - p00
	b = p01 - p00
	c = p11 - p10 - p01 + p00
	q = p00 + u2.unsqueeze(-1) * a + v2.unsqueeze(-1) * b + (u2 * v2).unsqueeze(-1) * c
	delta = q - source_pos
	signed = (delta * n).sum(dim=-1)
	line_delta = delta - signed.unsqueeze(-1) * n
	line_err = line_delta.norm(dim=-1)
	uv_tol = 1.0e-4
	uv_ok = (
		(u2 >= -uv_tol) & (u2 <= 1.0 + uv_tol) &
		(v2 >= -uv_tol) & (v2 <= 1.0 + uv_tol)
	)
	target_hit = (
		source_ok &
		pass1_valid &
		base1_valid &
		torch.isfinite(q).all(dim=-1) &
		torch.isfinite(u2) &
		torch.isfinite(v2) &
		torch.isfinite(line_err) &
		uv_ok
	)
	stats["target_hit"] += int(target_hit.sum().detach().cpu())
	accepted = target_hit & (line_err <= float(cfg.ray_residual))
	stats["distance_hit"] += int(accepted.sum().detach().cpu())
	stats["accepted"] += int(accepted.sum().detach().cpu())
	coords = base1.to(dtype=dtype) + torch.stack([u2.clamp(0.0, 1.0), v2.clamp(0.0, 1.0)], dim=-1)
	coords = torch.where(accepted.unsqueeze(-1), coords, torch.full_like(coords, float("nan")))
	if bool(accepted.any().detach().cpu()):
		err = line_err[accepted].detach()
		stats["line_err_sum"] += float(err.sum().cpu())
		stats["line_err_max"] = max(float(stats["line_err_max"]), float(err.max().cpu()))
	out_coords[rem_idx] = coords
	out_accepted[rem_idx] = accepted
	return out_coords, out_accepted, stats

def _intersect_model_points_with_ext_surface(
	*,
	source_pos: torch.Tensor,
	source_normals: torch.Tensor,
	ext_xyz: torch.Tensor,
	ext_valid: torch.Tensor,
	ext_quad_valid: torch.Tensor,
	cfg: SnapSurfConfig,
	pair_chunk_limit: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int | float]]:
	C = int(source_pos.shape[0])
	device = source_pos.device
	dtype = source_pos.dtype
	coords_all = torch.full((C, 2), float("nan"), device=device, dtype=dtype)
	accepted_all = torch.zeros(C, device=device, dtype=torch.bool)
	stats: dict[str, int | float] = {
		"target_hit": 0,
		"distance_hit": 0,
		"accepted": 0,
		"tested": 0,
		"line_err_sum": 0.0,
		"line_err_max": 0.0,
	}
	if C == 0:
		return coords_all, accepted_all, stats

	bases = _valid_ext_quad_bases(ext_valid, ext_quad_valid)
	K = int(bases.shape[0])
	stats["tested"] = C * K
	if K == 0:
		return coords_all, accepted_all, stats
	p00, p10, p01, p11 = _quad_corners_batched(ext_xyz, bases)
	pair_limit = int(cfg.brute_pair_chunk_limit if pair_chunk_limit is None else pair_chunk_limit)
	chunk = max(1, min(C, max(1, pair_limit) // max(1, K)))
	for start in range(0, C, chunk):
		end = min(C, start + chunk)
		coords, accepted, part = _intersect_model_points_with_ext_surface_chunk(
			source_pos=source_pos[start:end],
			source_normals=source_normals[start:end],
			bases=bases,
			p00=p00,
			p10=p10,
			p01=p01,
			p11=p11,
			cfg=cfg,
		)
		coords_all[start:end] = coords
		accepted_all[start:end] = accepted
		stats["target_hit"] += int(part.get("target_hit", 0))
		stats["distance_hit"] += int(part.get("distance_hit", 0))
		stats["accepted"] += int(part.get("accepted", 0))
		stats["line_err_sum"] += float(part.get("line_err_sum", 0.0))
		stats["line_err_max"] = max(float(stats["line_err_max"]), float(part.get("line_err_max", 0.0)))
	return coords_all, accepted_all, stats

def _intersect_model_points_with_ext_surface_incremental(
	*,
	source_pos: torch.Tensor,
	source_normals: torch.Tensor,
	prev_coords: torch.Tensor | None,
	allow_brute: bool,
	brute_source_mask: torch.Tensor | None,
	ext_xyz: torch.Tensor,
	ext_valid: torch.Tensor,
	ext_quad_valid: torch.Tensor,
	cfg: SnapSurfConfig,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int | float]]:
	C = int(source_pos.shape[0])
	device = source_pos.device
	dtype = source_pos.dtype
	coords_all = torch.full((C, 2), float("nan"), device=device, dtype=dtype)
	accepted_all = torch.zeros(C, device=device, dtype=torch.bool)
	stats: dict[str, int | float] = {
		"target_hit": 0,
		"distance_hit": 0,
		"accepted": 0,
		"tested": 0,
		"line_err_sum": 0.0,
		"line_err_max": 0.0,
		"local_accepted": 0,
		"brute_sources": 0,
		"brute_front": 0,
		"brute_allowed": int(bool(allow_brute)),
	}
	if C == 0:
		return coords_all, accepted_all, stats

	prev_ok = torch.zeros(C, device=device, dtype=torch.bool)
	if prev_coords is not None and prev_coords.shape == coords_all.shape:
		prev_ok = torch.isfinite(prev_coords).all(dim=-1)
	if bool(prev_ok.any().detach().cpu()):
		local_idx = prev_ok.nonzero(as_tuple=False).flatten()
		coords, accepted, part = _intersect_model_points_with_ext_surface_near_coords(
			source_pos=source_pos[local_idx],
			source_normals=source_normals[local_idx],
			pred_coords=prev_coords[local_idx],
			ext_xyz=ext_xyz,
			ext_valid=ext_valid,
			ext_quad_valid=ext_quad_valid,
			cfg=cfg,
		)
		coords_all[local_idx] = coords
		accepted_all[local_idx] = accepted
		stats["target_hit"] += int(part.get("target_hit", 0))
		stats["distance_hit"] += int(part.get("distance_hit", 0))
		stats["accepted"] += int(part.get("accepted", 0))
		stats["tested"] += int(part.get("tested", 0))
		stats["line_err_sum"] += float(part.get("line_err_sum", 0.0))
		stats["line_err_max"] = max(float(stats["line_err_max"]), float(part.get("line_err_max", 0.0)))
		stats["local_accepted"] = int(part.get("accepted", 0))

	if brute_source_mask is None:
		brute_mask = torch.ones(C, device=device, dtype=torch.bool)
	else:
		brute_mask = brute_source_mask.to(device=device).bool()
		if tuple(brute_mask.shape) != (C,):
			raise ValueError(f"expected brute_source_mask shape {(C,)}, got {tuple(brute_mask.shape)}")
	stats["brute_front"] = int(brute_mask.sum().detach().cpu())
	remaining = (~accepted_all) & brute_mask
	if not bool(allow_brute):
		remaining = remaining & torch.zeros_like(remaining)
	stats["brute_sources"] = int(remaining.sum().detach().cpu())
	if bool(remaining.any().detach().cpu()):
		rem_idx = remaining.nonzero(as_tuple=False).flatten()
		coords, accepted, part = _intersect_model_points_with_ext_surface(
			source_pos=source_pos[rem_idx],
			source_normals=source_normals[rem_idx],
			ext_xyz=ext_xyz,
			ext_valid=ext_valid,
			ext_quad_valid=ext_quad_valid,
			cfg=cfg,
		)
		coords_all[rem_idx] = coords
		accepted_all[rem_idx] = accepted
		stats["target_hit"] += int(part.get("target_hit", 0))
		stats["distance_hit"] += int(part.get("distance_hit", 0))
		stats["accepted"] += int(part.get("accepted", 0))
		stats["tested"] += int(part.get("tested", 0))
		stats["line_err_sum"] += float(part.get("line_err_sum", 0.0))
		stats["line_err_max"] = max(float(stats["line_err_max"]), float(part.get("line_err_max", 0.0)))
	return coords_all, accepted_all, stats

def _rebuild_model_to_ext_rays(
	state: _SurfaceState,
	*,
	model_xyz_det: torch.Tensor,
	model_valid: torch.Tensor,
	model_normals_det: torch.Tensor,
	ext_xyz: torch.Tensor,
	ext_valid: torch.Tensor,
	ext_quad_valid: torch.Tensor,
	cfg: SnapSurfConfig,
	seed_xyz: tuple[float, float, float],
) -> tuple[bool, float, dict[str, int | float], int]:
	stats = _empty_grow_stats()
	stats["tested"] = 0
	stats["line_err_sum"] = 0.0
	stats["line_err_max"] = 0.0
	seed = torch.tensor(seed_xyz, device=model_xyz_det.device, dtype=model_xyz_det.dtype)
	seed_quad, seed_model_dist = _closest_model_surface_quad(
		point=seed,
		model_xyz=model_xyz_det,
		model_valid=model_valid,
	)
	state.model_to_ext.seed_base_idx = seed_quad
	state.model_to_ext.orientation_sign = 1
	if seed_quad is None:
		_clear_direction_state(state.model_to_ext)
		_clear_direction_state(state.ext_to_model)
		state.ext_to_model.seed_base_idx = None
		return False, float("inf"), stats, 0

	source_mask = _seed_model_sheet_mask(
		model_valid=model_valid,
		model_normals=model_normals_det,
		seed_quad=seed_quad,
	)
	source_idx = source_mask.nonzero(as_tuple=False)
	source_possible = int(source_idx.shape[0])
	stats["ring"] = source_possible
	stats["sup"] = source_possible
	if source_possible == 0:
		_clear_direction_state(state.model_to_ext)
		_clear_direction_state(state.ext_to_model)
		state.ext_to_model.seed_base_idx = None
		return True, seed_model_dist, stats, 0

	prev_coords = None
	prev_valid_full = None
	if state.model_to_ext.map is not None and state.model_to_ext.valid is not None:
		prev_valid_full = state.model_to_ext.valid.clone()
		prev_coords = state.model_to_ext.map[source_idx[:, 0], source_idx[:, 1], source_idx[:, 2]].clone()
	brute_front_full = _brute_source_front_mask(
		prev_valid=prev_valid_full,
		source_mask=source_mask,
		seed_quad=seed_quad,
		radius=cfg.brute_boundary_radius,
	)
	brute_source_mask = brute_front_full[source_idx[:, 0], source_idx[:, 1], source_idx[:, 2]]
	allow_brute = (int(state.snap_eval_count) % int(cfg.brute_interval)) == 0

	_clear_direction_state(state.model_to_ext)
	_clear_direction_state(state.ext_to_model)
	state.ext_to_model.seed_base_idx = None

	source_pos = _points_at_indices(model_xyz_det, source_idx)
	source_normals = _points_at_indices(model_normals_det, source_idx)
	coords, accepted, ray_stats = _intersect_model_points_with_ext_surface_incremental(
		source_pos=source_pos,
		source_normals=source_normals,
		prev_coords=prev_coords,
		allow_brute=allow_brute,
		brute_source_mask=brute_source_mask,
		ext_xyz=ext_xyz,
		ext_valid=ext_valid,
		ext_quad_valid=ext_quad_valid,
		cfg=cfg,
	)
	stats["tgt"] = int(ray_stats.get("target_hit", 0))
	raw_accepted = int(ray_stats.get("accepted", 0))
	stats["dist"] = raw_accepted
	stats["tested"] = int(ray_stats.get("tested", 0))
	stats["local"] = int(ray_stats.get("local_accepted", 0))
	stats["brute"] = int(ray_stats.get("brute_sources", 0))
	stats["front"] = int(ray_stats.get("brute_front", 0))
	stats["brute_on"] = int(ray_stats.get("brute_allowed", 0))
	stats["gerr_n"] = raw_accepted
	stats["gerr_sum"] = float(ray_stats.get("line_err_sum", 0.0))
	stats["gerr_max"] = float(ray_stats.get("line_err_max", 0.0))
	if state.model_to_ext.map is not None and state.model_to_ext.valid is not None and bool(accepted.any().detach().cpu()):
		raw_map = torch.full_like(state.model_to_ext.map, float("nan"))
		raw_valid = torch.zeros_like(state.model_to_ext.valid)
		raw_normal_dist = torch.full(raw_valid.shape, float("nan"), device=model_xyz_det.device, dtype=model_xyz_det.dtype)
		if prev_coords is not None:
			prev_finite = torch.isfinite(prev_coords).all(dim=-1)
			if bool(prev_finite.any().detach().cpu()):
				prev_idx = source_idx[prev_finite]
				raw_map[prev_idx[:, 0], prev_idx[:, 1], prev_idx[:, 2]] = prev_coords[prev_finite].to(dtype=raw_map.dtype)
		acc_idx = source_idx[accepted]
		raw_map[acc_idx[:, 0], acc_idx[:, 1], acc_idx[:, 2]] = coords[accepted].to(dtype=raw_map.dtype)
		raw_valid[acc_idx[:, 0], acc_idx[:, 1], acc_idx[:, 2]] = True
		acc_target = _sample_surface_grid(ext_xyz, coords[accepted]).detach()
		acc_normals = F.normalize(source_normals[accepted], dim=-1, eps=1.0e-8)
		acc_normal_dist = ((source_pos[accepted] - acc_target) * acc_normals).sum(dim=-1).abs()
		raw_normal_dist[acc_idx[:, 0], acc_idx[:, 1], acc_idx[:, 2]] = acc_normal_dist.to(dtype=raw_normal_dist.dtype)
		initial_inlier = _closest_seed_source_mask(
			raw_valid=raw_valid,
			model_xyz=model_xyz_det,
			seed=seed,
		)
		inlier_valid, map_stats = _seeded_mapping_inlier_filter(
			raw_valid=raw_valid,
			raw_map=raw_map,
			initial_inlier=initial_inlier,
			max_distance=cfg.map_inlier_distance,
			normal_dist=raw_normal_dist,
			max_normal_ratio=cfg.inlier_normal_distance_ratio,
			normal_distance_floor=cfg.inlier_normal_distance_floor,
		)
		final_n = int(inlier_valid.sum().detach().cpu())
		stats["drop"] = max(0, raw_accepted - final_n)
		stats["grid"] = final_n
		stats["ori"] = final_n
		stats["new"] = final_n
		for k, v in map_stats.items():
			stats[k] = v
		state.model_to_ext.map[:] = raw_map
		state.model_to_ext.valid[:] = inlier_valid
	else:
		if state.model_to_ext.map is not None and prev_coords is not None:
			state.model_to_ext.map[:] = float("nan")
			state.model_to_ext.map[source_idx[:, 0], source_idx[:, 1], source_idx[:, 2]] = prev_coords
		if state.model_to_ext.valid is not None:
			state.model_to_ext.valid[:] = False
		stats["grid"] = 0
		stats["ori"] = 0
		stats["new"] = 0
	state.snap_eval_count += 1
	return True, seed_model_dist, stats, source_possible

def _direction_loss_model_ray_to_ext(
	state: _DirectionState,
	*,
	model_xyz: torch.Tensor,
	model_normals: torch.Tensor,
	ext_xyz: torch.Tensor,
	ext_valid: torch.Tensor,
	cfg: SnapSurfConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, dict[str, float]]:
	device = model_xyz.device
	dtype = model_xyz.dtype
	lm = torch.zeros(model_xyz.shape[:3], device=device, dtype=dtype)
	mask = torch.zeros(model_xyz.shape[:3], device=device, dtype=dtype)
	if state.map is None or state.valid is None or state.count() == 0:
		z = model_xyz.new_zeros(())
		return z, lm.unsqueeze(1), mask.unsqueeze(1), 0, _empty_residual_stats()
	idx = state.valid.nonzero(as_tuple=False)
	all_coords = state.map[state.valid]
	coord_ok = torch.isfinite(all_coords).all(dim=-1) & _quad_valid_at_coords(
		ext_valid.bool(),
		all_coords,
		tuple(int(v) for v in ext_xyz.shape[:2]),
	)
	if not bool(coord_ok.any().detach().cpu()):
		return model_xyz.new_zeros(()), lm.unsqueeze(1), mask.unsqueeze(1), 0, _empty_residual_stats()
	idx = idx[coord_ok]
	coords = all_coords[coord_ok]
	src = _points_at_indices(model_xyz, idx)
	tgt = _sample_surface_grid(ext_xyz, coords).detach()
	n_raw = _points_at_indices(model_normals.detach(), idx)
	n = F.normalize(n_raw, dim=-1, eps=1.0e-8)
	raw_residual = ((src - tgt) * n).sum(dim=-1)
	residual = raw_residual / cfg.distance_scale
	values = _huber(residual, delta=cfg.huber_delta / cfg.distance_scale)
	finite = (
		torch.isfinite(src).all(dim=-1) &
		torch.isfinite(tgt).all(dim=-1) &
		torch.isfinite(n_raw).all(dim=-1) &
		torch.isfinite(n).all(dim=-1) &
		(n.norm(dim=-1) > 1.0e-8) &
		torch.isfinite(raw_residual) &
		torch.isfinite(values)
	)
	values_f = values[finite]
	loss = values_f.mean() if values_f.numel() else model_xyz.new_zeros(())
	if bool(finite.any().detach().cpu()):
		lm_idx = idx[finite]
		lm[lm_idx[:, 0], lm_idx[:, 1], lm_idx[:, 2]] = values_f.detach()
		mask[lm_idx[:, 0], lm_idx[:, 1], lm_idx[:, 2]] = 1.0
	stats = _residual_stats(raw_residual, residual, delta=cfg.huber_delta / cfg.distance_scale)
	return loss, lm.unsqueeze(1), mask.unsqueeze(1), int(values_f.numel()), stats

__all__ = [name for name in globals() if not name.startswith('__')]
