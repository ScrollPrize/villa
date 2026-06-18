from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

import dense_batch_flow
import model as fit_model
from opt_loss_dir import _vertex_normals
from opt_loss_station import _intersect_single_quad
from progress_table import (
	ProgressColumn,
	format_progress_value,
	print_progress_legend,
	progress_header,
	progress_row,
	progress_widths,
)

_INNER_FACTOR = 0.0  # penalty reduction for points inside the predicted surface
_flow_gate_cfg: dict | None = None
_flow_gate_stage: str = "stage"
_flow_gate_seed_xyz: tuple[float, float, float] | None = None
_flow_gate_out_dir: Path | None = None
_flow_gate_debug_counts: dict[str, int] = {}
_flow_gate_last_stats: dict[str, float] = {}
_flow_gate_last_timing: dict[str, float] = {}
_flow_gate_last_channels: dict | None = None
_flow_gate_seed_hw_cache: tuple[int, int, float, float] | None = None
_flow_gate_jpg_warned: bool = False
_atlas_snap_report_legend_printed: set[tuple[str, int]] = set()
_pred_dt_normal_source: str = "model"


def flow_gate_prefetch_points(
	*,
	data,
	xyz_hr: torch.Tensor,
	xyz_lr: torch.Tensor | None = None,
	cfg: dict | None,
) -> torch.Tensor | None:
	"""Extra pred-dt sample positions used by the flow-gate render."""
	if cfg is None or not bool(cfg.get("enabled", False)):
		return None
	extra: list[torch.Tensor] = []
	r = int(max(0, int(cfg.get("pred_dt_pool_radius", 0))))
	if r > 0:
		step_scale = float(cfg.get("pred_dt_pool_step_scale", 0.5))
		xyz0 = xyz_hr.detach()
		offsets = torch.tensor(
			[(dx, dy, dz)
			 for dz in range(-r, r + 1)
			 for dy in range(-r, r + 1)
			 for dx in range(-r, r + 1)],
			device=xyz_hr.device,
			dtype=xyz_hr.dtype,
		)
		spacing = torch.tensor(
			data._spacing_for("pred_dt"),
			device=xyz0.device,
			dtype=xyz0.dtype,
		)
		offsets = offsets * spacing.view(1, 3) * step_scale
		extra.append((xyz0.unsqueeze(0) + offsets.view(-1, 1, 1, 1, 3)).reshape(-1, 3))

	if not extra:
		return None
	return torch.cat(extra, dim=0).view(1, 1, -1, 3)


def _seed_gt_normal(*, seed_xyz: torch.Tensor, res: fit_model.FitResult3D) -> torch.Tensor:
	query = seed_xyz.view(1, 1, 1, 3)
	sampled = res.data.grid_sample_fullres(query, channels={"nx", "ny"})
	nx = sampled.nx.squeeze()
	ny = sampled.ny.squeeze()
	nz = (1.0 - nx * nx - ny * ny).clamp(min=0.0).sqrt()
	n = torch.stack([nx, ny, nz]).to(device=seed_xyz.device, dtype=seed_xyz.dtype)
	return n / n.norm().clamp(min=1e-8)


def _seed_surface_intersection_xy(
	*,
	xyz_img: torch.Tensor,
	seed_xyz: torch.Tensor,
	n_gt: torch.Tensor,
) -> tuple[float, float] | None:
	"""Intersect seed + t*n_gt with the current 2D rendered surface.

	Returns image-space (x, y), using the high-resolution model render.
	"""
	if xyz_img.ndim != 3 or xyz_img.shape[-1] != 3:
		return None
	H, W, _ = xyz_img.shape
	if H < 2 or W < 2:
		return None

	M00 = xyz_img[:-1, :-1]
	M10 = xyz_img[1:, :-1]
	M01 = xyz_img[:-1, 1:]
	M11 = xyz_img[1:, 1:]
	frac_h = torch.tensor(0.5, device=xyz_img.device, dtype=xyz_img.dtype)
	frac_w = torch.tensor(0.5, device=xyz_img.device, dtype=xyz_img.dtype)

	u, v = fit_model.Model3D._ray_bilinear_intersect(
		seed_xyz, n_gt, M00, M10, M01, M11, frac_h, frac_w)
	valid = (
		torch.isfinite(u) & torch.isfinite(v) &
		(u >= -1.0e-4) & (u <= 1.0 + 1.0e-4) &
		(v >= -1.0e-4) & (v <= 1.0 + 1.0e-4)
	)
	p = (
		(1.0 - u).unsqueeze(-1) * (1.0 - v).unsqueeze(-1) * M00 +
		u.unsqueeze(-1) * (1.0 - v).unsqueeze(-1) * M10 +
		(1.0 - u).unsqueeze(-1) * v.unsqueeze(-1) * M01 +
		u.unsqueeze(-1) * v.unsqueeze(-1) * M11
	)
	delta = p - seed_xyz.view(1, 1, 3)
	t = (delta * n_gt.view(1, 1, 3)).sum(dim=-1)
	residual = (delta - t.unsqueeze(-1) * n_gt.view(1, 1, 3)).norm(dim=-1)
	score = residual * 1000.0 + t.abs()
	score = score.masked_fill(~valid, float("inf"))
	best_flat_t = torch.argmin(score)
	best_score = score.reshape(-1)[best_flat_t]
	if not torch.isfinite(best_score).detach().cpu().item():
		return None
	best_flat = int(best_flat_t.detach().cpu())
	y = best_flat // (W - 1)
	x = best_flat % (W - 1)
	return (
		float((torch.as_tensor(float(x), device=xyz_img.device, dtype=xyz_img.dtype) + v[y, x]).detach().cpu()),
		float((torch.as_tensor(float(y), device=xyz_img.device, dtype=xyz_img.dtype) + u[y, x]).detach().cpu()),
	)


def _seed_surface_intersection_xy_from_cache(
	*,
	xyz_img: torch.Tensor,
	seed_xyz: torch.Tensor,
	n_gt: torch.Tensor,
	h_frac: float,
	w_frac: float,
) -> tuple[float, float] | None:
	"""One station-style cached ray/surface update on the high-res render grid."""
	if xyz_img.ndim != 3 or xyz_img.shape[-1] != 3:
		return None
	H, W, _ = xyz_img.shape
	if H < 2 or W < 2:
		return None
	if not (np.isfinite(h_frac) and np.isfinite(w_frac)):
		return None

	row = max(0, min(int(h_frac), H - 2))
	col = max(0, min(int(w_frac), W - 2))
	frac_h = h_frac - float(row)
	frac_w = w_frac - float(col)

	try:
		u1, v1, _ = _intersect_single_quad(
			seed_xyz, n_gt,
			xyz_img[row, col],
			xyz_img[row + 1, col],
			xyz_img[row, col + 1],
			xyz_img[row + 1, col + 1],
			frac_h, frac_w,
		)
	except Exception:
		return None
	if not (np.isfinite(u1) and np.isfinite(v1)):
		return None

	new_h = max(0.0, min(float(H - 2), float(row) + float(u1)))
	new_w = max(0.0, min(float(W - 2), float(col) + float(v1)))
	new_row = max(0, min(int(new_h), H - 2))
	new_col = max(0, min(int(new_w), W - 2))
	new_frac_h = new_h - float(new_row)
	new_frac_w = new_w - float(new_col)

	try:
		u2, v2, _ = _intersect_single_quad(
			seed_xyz, n_gt,
			xyz_img[new_row, new_col],
			xyz_img[new_row + 1, new_col],
			xyz_img[new_row, new_col + 1],
			xyz_img[new_row + 1, new_col + 1],
			new_frac_h, new_frac_w,
		)
	except Exception:
		return None
	if not (np.isfinite(u2) and np.isfinite(v2)):
		return None
	if u2 < 0.0 or u2 > 1.0 or v2 < 0.0 or v2 > 1.0:
		return None

	return float(new_col) + float(v2), float(new_row) + float(u2)


def _corr_point_source_xy(
	*,
	res: fit_model.FitResult3D,
	xyz_img: torch.Tensor,
	cfg: dict,
	sub_h: int,
	sub_w: int,
	layer_index: int | None = None,
	winding_value: float | int | None = None,
) -> tuple[np.ndarray, dict[str, float], dict[str, np.ndarray]]:
	"""Project nearby correction points to rendered-surface pixels for flow seeds."""
	stats: dict[str, float] = {
		"pred_dt_corr_seed_candidates": 0.0,
		"pred_dt_corr_seed_valid": 0.0,
	}
	def empty_debug(surface_distance: float | None = None) -> dict[str, np.ndarray]:
		debug = {
			"xy": np.zeros((0, 2), dtype=np.int32),
			"distance": np.zeros((0,), dtype=np.float32),
			"valid": np.zeros((0,), dtype=np.bool_),
		}
		if surface_distance is not None:
			debug["surface_distance"] = np.asarray([surface_distance], dtype=np.float32)
		return debug

	if not bool(cfg.get("corr_seed_enabled", True)):
		return np.zeros((0, 2), dtype=np.int32), stats, empty_debug()
	corr = res.data.corr_points
	if corr is None or corr.points_xyz_winda.shape[0] == 0:
		return np.zeros((0, 2), dtype=np.int32), stats, empty_debug()
	points_xyz_winda = corr.points_xyz_winda
	if winding_value is not None:
		winda = points_xyz_winda[:, 3].detach()
		tol = float(cfg.get("corr_seed_winding_tolerance", 1.0e-4))
		target = torch.as_tensor(float(winding_value), device=winda.device, dtype=winda.dtype)
		match = torch.isfinite(winda) & torch.isclose(winda, target, rtol=0.0, atol=tol)
		stats["pred_dt_corr_seed_winding_candidates"] = float(int(match.sum().detach().cpu()))
		if not bool(match.any().detach().cpu()):
			return np.zeros((0, 2), dtype=np.int32), stats, empty_debug()
		points_xyz_winda = points_xyz_winda[match]
	if xyz_img.ndim != 3 or xyz_img.shape[-1] != 3:
		return np.zeros((0, 2), dtype=np.int32), stats, empty_debug()
	He, We, _ = xyz_img.shape
	if He < 2 or We < 2:
		return np.zeros((0, 2), dtype=np.int32), stats, empty_debug()

	default_distance = max(2.0, 1.5 * float(max(1, sub_h, sub_w)))
	max_distance = float(cfg.get("corr_seed_surface_distance", default_distance))
	stats["pred_dt_corr_seed_surface_distance"] = max_distance
	if max_distance < 0.0:
		return np.zeros((0, 2), dtype=np.int32), stats, empty_debug(max_distance)

	pts = points_xyz_winda[:, :3].to(
		device=xyz_img.device,
		dtype=xyz_img.dtype,
	).detach()
	finite = torch.isfinite(pts).all(dim=-1)
	if not finite.any():
		stats["pred_dt_corr_seed_candidates"] = float(int(pts.shape[0]))
		return np.zeros((0, 2), dtype=np.int32), stats, empty_debug(max_distance)
	pts = pts[finite]
	stats["pred_dt_corr_seed_candidates"] = float(int(pts.shape[0]))

	surface = xyz_img.detach().reshape(-1, 3)
	if surface.shape[0] == 0:
		return np.zeros((0, 2), dtype=np.int32), stats, empty_debug(max_distance)
	point_chunk = max(1, int(cfg.get("corr_seed_point_chunk", 32)))
	surface_chunk = max(1024, int(cfg.get("corr_seed_surface_chunk", 65536)))
	best_dist2_all: list[torch.Tensor] = []
	best_index_all: list[torch.Tensor] = []
	for p0 in range(0, int(pts.shape[0]), point_chunk):
		p = pts[p0:p0 + point_chunk]
		best_dist2 = torch.full(
			(p.shape[0],),
			float("inf"),
			device=pts.device,
			dtype=pts.dtype,
		)
		best_index = torch.zeros((p.shape[0],), device=pts.device, dtype=torch.long)
		for s0 in range(0, int(surface.shape[0]), surface_chunk):
			s = surface[s0:s0 + surface_chunk]
			d2 = ((p[:, None, :] - s[None, :, :]) ** 2).sum(dim=-1)
			chunk_dist2, chunk_index = d2.min(dim=1)
			update = chunk_dist2 < best_dist2
			best_dist2 = torch.where(update, chunk_dist2, best_dist2)
			best_index = torch.where(update, chunk_index + int(s0), best_index)
		best_dist2_all.append(best_dist2)
		best_index_all.append(best_index)
	best_dist2 = torch.cat(best_dist2_all, dim=0)
	best_index = torch.cat(best_index_all, dim=0)
	best_dist = best_dist2.clamp_min(0.0).sqrt()
	valid = torch.isfinite(best_dist) & (best_dist <= max_distance)
	valid_count = int(valid.sum().detach().cpu())
	stats["pred_dt_corr_seed_valid"] = float(valid_count)
	flat = best_index.detach().cpu().numpy().astype(np.int64, copy=False)
	y = (flat // int(We)).clip(0, int(He) - 1)
	x = (flat % int(We)).clip(0, int(We) - 1)
	all_xy = np.stack([x, y], axis=1).astype(np.int32, copy=False)
	valid_np = valid.detach().cpu().numpy().astype(np.bool_, copy=False)
	distance_np = best_dist.detach().cpu().numpy().astype(np.float32, copy=False)
	debug = {
		"xy": all_xy,
		"distance": distance_np,
		"valid": valid_np,
		"surface_distance": np.asarray([max_distance], dtype=np.float32),
	}
	if valid_count <= 0:
		stats["pred_dt_corr_seed_unique"] = 0.0
		return np.zeros((0, 2), dtype=np.int32), stats, debug
	valid_dist = best_dist[valid]
	stats["pred_dt_corr_seed_mean_distance"] = float(valid_dist.mean().detach().cpu())
	stats["pred_dt_corr_seed_max_distance"] = float(valid_dist.max().detach().cpu())
	xy = all_xy[valid_np]
	if xy.shape[0] > 1:
		xy = np.unique(xy, axis=0)
	stats["pred_dt_corr_seed_unique"] = float(int(xy.shape[0]))
	return xy, stats, debug


def _bilinear_at_points(
	grid: torch.Tensor,
	row: torch.Tensor,
	col: torch.Tensor,
	frac_h: torch.Tensor,
	frac_w: torch.Tensor,
) -> torch.Tensor:
	v00 = grid[row, col]
	v10 = grid[row + 1, col]
	v01 = grid[row, col + 1]
	v11 = grid[row + 1, col + 1]
	return (
		v00 * (1.0 - frac_h).unsqueeze(-1) * (1.0 - frac_w).unsqueeze(-1)
		+ v10 * frac_h.unsqueeze(-1) * (1.0 - frac_w).unsqueeze(-1)
		+ v01 * (1.0 - frac_h).unsqueeze(-1) * frac_w.unsqueeze(-1)
		+ v11 * frac_h.unsqueeze(-1) * frac_w.unsqueeze(-1)
	)


def _atlas_snap_source_xy(
	*,
	res: fit_model.FitResult3D,
	xyz_img: torch.Tensor,
	pred_img: np.ndarray,
	cfg: dict,
	sub_h: int,
	sub_w: int,
	layer_index: int,
) -> tuple[np.ndarray, dict[str, float], dict[str, object]]:
	"""Project atlas pred-snap anchors to rendered-surface pixels for flow seeds."""
	stats: dict[str, float] = {
		"pred_dt_atlas_snap_seed_candidates": 0.0,
		"pred_dt_atlas_snap_seed_target_accepted": 0.0,
		"pred_dt_atlas_snap_seed_pred_dt_accepted": 0.0,
		"pred_dt_atlas_snap_seed_unique": 0.0,
	}
	def empty_debug(surface_distance: float | None = None, min_inlier: float | None = None) -> dict[str, object]:
		debug = {
			"xy": np.zeros((0, 2), dtype=np.int32),
			"distance": np.zeros((0,), dtype=np.float32),
			"inlier": np.zeros((0,), dtype=np.float32),
			"valid": np.zeros((0,), dtype=np.bool_),
			"records": [],
		}
		if surface_distance is not None:
			debug["surface_distance"] = np.asarray([surface_distance], dtype=np.float32)
		if min_inlier is not None:
			debug["min_inlier"] = np.asarray([min_inlier], dtype=np.float32)
		return debug

	if not bool(cfg.get("atlas_snap_seed_enabled", False)):
		return np.zeros((0, 2), dtype=np.int32), stats, empty_debug()
	lines = getattr(res.data, "atlas_lines", None)
	if lines is None:
		return np.zeros((0, 2), dtype=np.int32), stats, empty_debug()
	K = int(getattr(lines, "target_xyz").shape[0])
	if K <= 0:
		return np.zeros((0, 2), dtype=np.int32), stats, empty_debug()
	is_snap = getattr(lines, "is_snap_point", None)
	if is_snap is None:
		return np.zeros((0, 2), dtype=np.int32), stats, empty_debug()
	snap = is_snap.to(device=xyz_img.device, dtype=torch.bool).view(K)
	if not bool(snap.any().detach().cpu()):
		return np.zeros((0, 2), dtype=np.int32), stats, empty_debug()
	if res.normals is None:
		raise RuntimeError("pred_dt_flow_gate atlas_snap_seed_enabled requires model normals")
	if xyz_img.ndim != 3 or xyz_img.shape[-1] != 3:
		return np.zeros((0, 2), dtype=np.int32), stats, empty_debug()
	if pred_img.ndim != 2:
		raise RuntimeError(f"pred_dt_flow_gate expected 2D pred_dt image for atlas snap seeds, got {tuple(pred_img.shape)}")
	He, We, _ = xyz_img.shape
	if He <= 0 or We <= 0:
		return np.zeros((0, 2), dtype=np.int32), stats, empty_debug()
	d = int(layer_index)
	if d < 0 or d >= int(res.xyz_lr.shape[0]):
		return np.zeros((0, 2), dtype=np.int32), stats, empty_debug()
	Hm = int(res.xyz_lr.shape[1])
	Wm = int(res.xyz_lr.shape[2])
	if Hm < 2 or Wm < 2:
		return np.zeros((0, 2), dtype=np.int32), stats, empty_debug()

	default_distance = max(2.0, 1.5 * float(max(1, sub_h, sub_w)))
	max_distance = float(cfg.get(
		"atlas_snap_seed_target_distance",
		cfg.get("corr_seed_surface_distance", default_distance),
	))
	pull_cfg = cfg.get("anticipatory_pull", None)
	pull_cfg = pull_cfg if isinstance(pull_cfg, dict) else {}
	inlier_zero = float(pull_cfg.get("inlier_zero", 80.0))
	inlier_one = float(pull_cfg.get("inlier_one", 120.0))
	if inlier_one <= inlier_zero:
		raise ValueError("pred_dt_flow_gate atlas snap seeding requires inlier_one > inlier_zero")
	min_inlier = float(cfg.get("atlas_snap_seed_min_inlier", 0.5))
	stats["pred_dt_atlas_snap_seed_target_distance"] = max_distance
	stats["pred_dt_atlas_snap_seed_min_inlier"] = min_inlier
	if max_distance < 0.0:
		return np.zeros((0, 2), dtype=np.int32), stats, empty_debug(max_distance, min_inlier)

	idx = torch.nonzero(snap, as_tuple=False).flatten()
	stats["pred_dt_atlas_snap_seed_candidates"] = float(int(idx.numel()))
	target = lines.target_xyz.to(device=xyz_img.device, dtype=xyz_img.dtype).index_select(0, idx)
	h = lines.model_h.to(device=xyz_img.device, dtype=torch.float32).index_select(0, idx)
	w = lines.model_w.to(device=xyz_img.device, dtype=torch.float32).index_select(0, idx)
	finite = torch.isfinite(target).all(dim=-1) & torch.isfinite(h) & torch.isfinite(w)
	hc = h.clamp(0.0, float(Hm - 1))
	wc = w.clamp(0.0, float(Wm - 1))
	row = torch.floor(hc).to(dtype=torch.long).clamp(0, Hm - 2)
	col = torch.floor(wc).to(dtype=torch.long).clamp(0, Wm - 2)
	frac_h = (hc - row.to(dtype=torch.float32)).to(device=xyz_img.device, dtype=xyz_img.dtype).clamp(0.0, 1.0)
	frac_w = (wc - col.to(dtype=torch.float32)).to(device=xyz_img.device, dtype=xyz_img.dtype).clamp(0.0, 1.0)

	n0 = _bilinear_at_points(res.normals[d].detach().to(device=xyz_img.device, dtype=xyz_img.dtype), row, col, frac_h, frac_w)
	n_norm = n0.norm(dim=-1, keepdim=True)
	n0 = n0 / n_norm.clamp_min(1.0e-8)
	n_valid = torch.isfinite(n0).all(dim=-1) & (n_norm.squeeze(-1) > 1.0e-8)
	h_hr = (h.to(device=xyz_img.device, dtype=xyz_img.dtype) * float(sub_h)).clamp(0.0, float(He - 1))
	w_hr = (w.to(device=xyz_img.device, dtype=xyz_img.dtype) * float(sub_w)).clamp(0.0, float(We - 1))
	row_hr = torch.floor(h_hr).to(dtype=torch.long).clamp(0, He - 2)
	col_hr = torch.floor(w_hr).to(dtype=torch.long).clamp(0, We - 2)
	frac_h_hr = (h_hr - row_hr.to(dtype=xyz_img.dtype)).clamp(0.0, 1.0)
	frac_w_hr = (w_hr - col_hr.to(dtype=xyz_img.dtype)).clamp(0.0, 1.0)
	M00 = xyz_img[row_hr, col_hr]
	M10 = xyz_img[row_hr + 1, col_hr]
	M01 = xyz_img[row_hr, col_hr + 1]
	M11 = xyz_img[row_hr + 1, col_hr + 1]
	u1, v1 = fit_model.Model3D._ray_bilinear_intersect(target, n0, M00, M10, M01, M11, frac_h_hr, frac_w_hr)
	finite1 = torch.isfinite(u1) & torch.isfinite(v1)
	step_h = torch.where(u1 < 0.0, torch.full_like(row_hr, -1), torch.where(u1 > 1.0, torch.ones_like(row_hr), torch.zeros_like(row_hr)))
	step_w = torch.where(v1 < 0.0, torch.full_like(col_hr, -1), torch.where(v1 > 1.0, torch.ones_like(col_hr), torch.zeros_like(col_hr)))
	row2 = torch.where(finite1, (row_hr + step_h).clamp(0, He - 2), row_hr)
	col2 = torch.where(finite1, (col_hr + step_w).clamp(0, We - 2), col_hr)
	frac_h1 = torch.where(finite1, u1.clamp(0.0, 1.0), frac_h_hr)
	frac_w1 = torch.where(finite1, v1.clamp(0.0, 1.0), frac_w_hr)
	Q00 = xyz_img[row2, col2]
	Q10 = xyz_img[row2 + 1, col2]
	Q01 = xyz_img[row2, col2 + 1]
	Q11 = xyz_img[row2 + 1, col2 + 1]
	u2, v2 = fit_model.Model3D._ray_bilinear_intersect(target, n0, Q00, Q10, Q01, Q11, frac_h1, frac_w1)
	in_quad = torch.isfinite(u2) & torch.isfinite(v2) & (u2 >= 0.0) & (u2 <= 1.0) & (v2 >= 0.0) & (v2 <= 1.0)
	u = u2.clamp(0.0, 1.0)
	v = v2.clamp(0.0, 1.0)
	hit = (
		Q00 * (1.0 - u).unsqueeze(-1) * (1.0 - v).unsqueeze(-1)
		+ Q10 * u.unsqueeze(-1) * (1.0 - v).unsqueeze(-1)
		+ Q01 * (1.0 - u).unsqueeze(-1) * v.unsqueeze(-1)
		+ Q11 * u.unsqueeze(-1) * v.unsqueeze(-1)
	)
	distance = (target - hit).norm(dim=-1)
	hit_finite = torch.isfinite(hit).all(dim=-1)
	distance_finite = torch.isfinite(distance)
	within_distance = distance <= max_distance
	target_valid = finite & n_valid & in_quad & hit_finite & distance_finite & within_distance
	stats["pred_dt_atlas_snap_seed_target_accepted"] = float(int(target_valid.sum().detach().cpu()))
	h_cont = row2.to(dtype=xyz_img.dtype) + u
	w_cont = col2.to(dtype=xyz_img.dtype) + v
	source_y = torch.round(h_cont).to(dtype=torch.long).clamp(0, He - 1)
	source_x = torch.round(w_cont).to(dtype=torch.long).clamp(0, We - 1)
	pred_t = torch.as_tensor(pred_img, device=xyz_img.device, dtype=xyz_img.dtype)
	pred_value = pred_t[source_y, source_x]
	inlier = ((pred_value - inlier_zero) / (inlier_one - inlier_zero)).clamp(0.0, 1.0)
	pred_valid = inlier >= min_inlier
	accepted = target_valid & pred_valid
	stats["pred_dt_atlas_snap_seed_pred_dt_accepted"] = float(int(accepted.sum().detach().cpu()))
	if bool(target_valid.any().detach().cpu()):
		valid_dist = distance[target_valid]
		stats["pred_dt_atlas_snap_seed_mean_distance"] = float(valid_dist.mean().detach().cpu())
		stats["pred_dt_atlas_snap_seed_max_distance"] = float(valid_dist.max().detach().cpu())

	all_xy = torch.stack([source_x, source_y], dim=1).detach().cpu().numpy().astype(np.int32, copy=False)
	sample_indices = idx.detach().cpu().numpy().astype(np.int64, copy=False)
	source_indices_raw = tuple(int(v) for v in (getattr(lines, "source_indices", ()) or ()))
	object_ids_raw = tuple(str(v) for v in (getattr(lines, "object_ids", ()) or ()))
	target_cpu = target.detach().cpu().numpy().astype(np.float32, copy=False)
	hit_cpu = hit.detach().cpu().numpy().astype(np.float32, copy=False)
	model_h_cpu = h.detach().cpu().numpy().astype(np.float32, copy=False)
	model_w_cpu = w.detach().cpu().numpy().astype(np.float32, copy=False)
	distance_cpu = distance.detach().cpu().numpy().astype(np.float32, copy=False)
	pred_value_cpu = pred_value.detach().cpu().numpy().astype(np.float32, copy=False)
	inlier_cpu = inlier.detach().cpu().numpy().astype(np.float32, copy=False)
	finite_cpu = finite.detach().cpu().numpy().astype(np.bool_, copy=False)
	n_valid_cpu = n_valid.detach().cpu().numpy().astype(np.bool_, copy=False)
	in_quad_cpu = in_quad.detach().cpu().numpy().astype(np.bool_, copy=False)
	hit_finite_cpu = hit_finite.detach().cpu().numpy().astype(np.bool_, copy=False)
	distance_finite_cpu = distance_finite.detach().cpu().numpy().astype(np.bool_, copy=False)
	within_distance_cpu = within_distance.detach().cpu().numpy().astype(np.bool_, copy=False)
	target_valid_cpu = target_valid.detach().cpu().numpy().astype(np.bool_, copy=False)
	pred_valid_cpu = pred_valid.detach().cpu().numpy().astype(np.bool_, copy=False)
	accepted_cpu = accepted.detach().cpu().numpy().astype(np.bool_, copy=False)
	def float_or_none(value: object) -> float | None:
		try:
			out = float(value)
		except (TypeError, ValueError):
			return None
		return out if np.isfinite(out) else None

	def xyz_or_none(values: np.ndarray) -> list[float | None]:
		return [float_or_none(v) for v in np.asarray(values).reshape(3)]

	records: list[dict[str, object]] = []
	for rec_i, sample_index in enumerate(sample_indices.tolist()):
		reason = "accepted"
		if not bool(finite_cpu[rec_i]):
			reason = "nonfinite_target_or_model_coord"
		elif not bool(n_valid_cpu[rec_i]):
			reason = "invalid_model_normal"
		elif not bool(in_quad_cpu[rec_i]):
			reason = "no_surface_intersection_near_model_coord"
		elif not bool(hit_finite_cpu[rec_i]):
			reason = "nonfinite_surface_hit"
		elif not bool(distance_finite_cpu[rec_i]):
			reason = "nonfinite_surface_distance"
		elif not bool(within_distance_cpu[rec_i]):
			reason = "surface_distance_too_large"
		elif not bool(pred_valid_cpu[rec_i]):
			reason = "pred_dt_inlier_too_low"
		source_index = (
			source_indices_raw[int(sample_index)]
			if 0 <= int(sample_index) < len(source_indices_raw)
			else int(sample_index)
		)
		object_id = (
			object_ids_raw[int(sample_index)]
			if 0 <= int(sample_index) < len(object_ids_raw)
			else "unknown"
		)
		records.append({
			"sample_index": int(sample_index),
			"object_id": object_id,
			"source_index": int(source_index),
			"model_h": float_or_none(model_h_cpu[rec_i]),
			"model_w": float_or_none(model_w_cpu[rec_i]),
			"target_xyz": xyz_or_none(target_cpu[rec_i]),
			"surface_hit_xyz": xyz_or_none(hit_cpu[rec_i]),
			"source_xy": [int(all_xy[rec_i, 0]), int(all_xy[rec_i, 1])],
			"surface_distance": float_or_none(distance_cpu[rec_i]),
			"surface_distance_limit": float(max_distance),
			"pred_dt_value": float_or_none(pred_value_cpu[rec_i]),
			"inlier": float_or_none(inlier_cpu[rec_i]),
			"min_inlier": float(min_inlier),
			"target_valid": bool(target_valid_cpu[rec_i]),
			"pred_valid": bool(pred_valid_cpu[rec_i]),
			"accepted": bool(accepted_cpu[rec_i]),
			"reason": reason,
		})
	debug = {
		"xy": all_xy,
		"distance": distance_cpu,
		"inlier": inlier_cpu,
		"valid": accepted_cpu,
		"surface_distance": np.asarray([max_distance], dtype=np.float32),
		"min_inlier": np.asarray([min_inlier], dtype=np.float32),
		"records": records,
	}
	if not bool(accepted.any().detach().cpu()):
		return np.zeros((0, 2), dtype=np.int32), stats, debug
	xy = all_xy[debug["valid"]]
	if xy.shape[0] > 1:
		xy = np.unique(xy, axis=0)
	stats["pred_dt_atlas_snap_seed_unique"] = float(int(xy.shape[0]))
	return xy, stats, debug


def _sample_pred_dt_max3d(
	*,
	res: fit_model.FitResult3D,
	xyz_hr: torch.Tensor,
	layer_index: int,
	radius: int,
	step_scale: float,
) -> torch.Tensor:
	"""Pred-dt render for flow gating, max-pooled in 3D around surface samples."""
	r = int(max(0, radius))
	if r <= 0:
		pred_hr = res.data_s.pred_dt.squeeze(0).squeeze(0)
		if pred_hr.ndim == 2:
			if int(layer_index) != 0:
				raise RuntimeError(f"pred_dt_flow_gate cannot sample layer {int(layer_index)} from 2D pred_dt render")
			return pred_hr
		if pred_hr.ndim != 3:
			raise RuntimeError(f"pred_dt_flow_gate expected pred_dt render shape (D,H,W), got {tuple(pred_hr.shape)}")
		if not (0 <= int(layer_index) < int(pred_hr.shape[0])):
			raise RuntimeError(
				f"pred_dt_flow_gate layer {int(layer_index)} outside pred_dt render depth {int(pred_hr.shape[0])}"
			)
		return pred_hr[int(layer_index)]

	offsets = torch.tensor(
		[(dx, dy, dz)
		 for dz in range(-r, r + 1)
		 for dy in range(-r, r + 1)
		 for dx in range(-r, r + 1)],
		device=xyz_hr.device,
		dtype=xyz_hr.dtype,
	)
	spacing = torch.tensor(
		res.data._spacing_for("pred_dt"),
		device=xyz_hr.device,
		dtype=xyz_hr.dtype,
	)
	offsets = offsets * spacing.view(1, 3) * float(step_scale)
	query = xyz_hr.unsqueeze(0) + offsets.view(-1, 1, 1, 3)
	sampled = res.data.grid_sample_fullres(query, channels={"pred_dt"}).pred_dt
	if sampled is None:
		raise RuntimeError("pred_dt_flow_gate requires pred_dt to be loaded")
	pred = sampled.squeeze(0).squeeze(0)
	if pred.ndim != 3:
		raise RuntimeError(f"pred_dt_flow_gate expected pooled pred_dt shape (N,H,W), got {tuple(pred.shape)}")
	return pred.amax(dim=0)


def _pred_dt_loss_sample_xyz(res: fit_model.FitResult3D) -> torch.Tensor:
	"""Exact LR positions used by pred_dt_loss for differentiable pred-dt sampling."""
	n = _pred_dt_projection_normals(res)
	proj_len = (res.xyz_lr * n).sum(dim=-1, keepdim=True)
	xyz_normal = proj_len * n
	xyz_tangential = res.xyz_lr - xyz_normal
	return xyz_normal + xyz_tangential.detach()


def _neighbor_candidate_indices(*, Hm: int, Wm: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
	"""Directed 8-neighbor root->tip pairs on one LR winding grid."""
	tip_hs: list[torch.Tensor] = []
	tip_ws: list[torch.Tensor] = []
	root_hs: list[torch.Tensor] = []
	root_ws: list[torch.Tensor] = []
	for dh in (-1, 0, 1):
		for dw in (-1, 0, 1):
			if dh == 0 and dw == 0:
				continue
			h0 = max(0, -dh)
			h1 = Hm - max(0, dh)
			w0 = max(0, -dw)
			w1 = Wm - max(0, dw)
			if h1 <= h0 or w1 <= w0:
				continue
			h, w = torch.meshgrid(
				torch.arange(h0, h1, device=device, dtype=torch.long),
				torch.arange(w0, w1, device=device, dtype=torch.long),
				indexing="ij",
			)
			tip_hs.append(h.reshape(-1))
			tip_ws.append(w.reshape(-1))
			root_hs.append((h + dh).reshape(-1))
			root_ws.append((w + dw).reshape(-1))
	return (
		torch.cat(tip_hs, dim=0),
		torch.cat(tip_ws, dim=0),
		torch.cat(root_hs, dim=0),
		torch.cat(root_ws, dim=0),
	)


def _anticipatory_pull_cfg(cfg: dict | None) -> dict | None:
	if not isinstance(cfg, dict):
		return None
	pull_cfg = cfg.get("anticipatory_pull", None)
	if not isinstance(pull_cfg, dict) or not bool(pull_cfg.get("enabled", False)):
		return None
	return pull_cfg


def _anticipatory_normal_offset_factors(*, cfg: dict, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
	steps = max(1, int(cfg.get("search_steps", 21)))
	if steps % 2 == 0:
		steps += 1
	deg = float(cfg.get("search_angle_degrees", 60.0))
	rad = np.deg2rad(max(0.0, deg))
	return torch.linspace(-float(np.tan(rad)), float(np.tan(rad)), steps, device=device, dtype=dtype)


def _anticipatory_reference_step(
	*,
	cfg: dict,
	device: torch.device,
	dtype: torch.dtype,
	params: fit_model.ModelParams3D | None = None,
	xyz0: torch.Tensor | None = None,
) -> torch.Tensor:
	if "search_step_length" in cfg:
		return torch.tensor(float(cfg["search_step_length"]), device=device, dtype=dtype).clamp_min(1e-6)
	if params is not None:
		return torch.tensor(float(params.mesh_step), device=device, dtype=dtype).clamp_min(1e-6)
	if xyz0 is not None:
		dh = (xyz0[1:] - xyz0[:-1]).norm(dim=-1).median() if xyz0.shape[0] > 1 else torch.tensor(1.0, device=device, dtype=dtype)
		dw = (xyz0[:, 1:] - xyz0[:, :-1]).norm(dim=-1).median() if xyz0.shape[1] > 1 else torch.tensor(1.0, device=device, dtype=dtype)
		return (0.5 * (dh + dw)).to(device=device, dtype=dtype).clamp_min(1e-6)
	return torch.tensor(1.0, device=device, dtype=dtype)


def flow_gate_prefetch_points_for_result(
	*,
	res: fit_model.FitResult3D,
	cfg: dict | None,
) -> torch.Tensor | None:
	items = flow_gate_prefetch_items_for_result(res=res, cfg=cfg)
	return items.get("pred_dt")


def pred_dt_prefetch_items_for_result(*, res: fit_model.FitResult3D) -> dict[str, torch.Tensor]:
	"""Exact pred-dt sample positions used by `pred_dt_loss`."""
	return {"pred_dt": _pred_dt_loss_sample_xyz(res).detach().reshape(1, 1, -1, 3)}


def flow_gate_prefetch_items_for_result(
	*,
	res: fit_model.FitResult3D,
	cfg: dict | None,
) -> dict[str, torch.Tensor]:
	"""Exact channel sample positions for active flow-gate losses after model forward."""
	out: dict[str, torch.Tensor] = {}
	if cfg is None or not bool(cfg.get("enabled", False)):
		return out
	xyz_hr = res.xyz_hr.detach()
	device = xyz_hr.device
	dtype = xyz_hr.dtype

	if _flow_gate_seed_xyz is not None:
		seed = torch.tensor(_flow_gate_seed_xyz, device=device, dtype=dtype).view(1, 1, 1, 3)
		out["nx"] = seed
		out["ny"] = seed

	loss_xyz = _pred_dt_loss_sample_xyz(res).detach().reshape(1, 1, -1, 3)
	out["pred_dt"] = loss_xyz

	r = int(max(0, int(cfg.get("pred_dt_pool_radius", 0))))
	if r > 0:
		step_scale = float(cfg.get("pred_dt_pool_step_scale", 0.5))
		offsets = torch.tensor(
			[(dx, dy, dz)
			 for dz in range(-r, r + 1)
			 for dy in range(-r, r + 1)
			 for dx in range(-r, r + 1)],
			device=device,
			dtype=dtype,
		)
		spacing = torch.tensor(res.data._spacing_for("pred_dt"), device=device, dtype=dtype)
		offsets = offsets * spacing.view(1, 3) * step_scale
		pool_points = (xyz_hr.unsqueeze(0) + offsets.view(-1, 1, 1, 1, 3)).reshape(1, 1, -1, 3)
		out["pred_dt"] = torch.cat([out["pred_dt"], pool_points], dim=2)

	# Anticipatory pull depends on the current flow gate. It is intentionally not
	# prefetched here; `_score_anticipatory_pull_candidates` filters by the known
	# gate and prefetches/samples the remaining candidates in bounded chunks.
	return out


def _pred_dt_projection_normals(res: fit_model.FitResult3D) -> torch.Tensor:
	if _pred_dt_normal_source == "model":
		n = _vertex_normals(res.xyz_lr.detach())
	elif _pred_dt_normal_source == "gt":
		if res.gt_normal_lr is None:
			raise RuntimeError("pred_dt normal_source='gt' requires gt_normal_lr")
		n = res.gt_normal_lr.detach().to(device=res.xyz_lr.device, dtype=res.xyz_lr.dtype)
	else:
		raise RuntimeError(f"unsupported pred_dt normal_source: {_pred_dt_normal_source!r}")
	return n / n.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def _tip_normals_from_result(
	*,
	res: fit_model.FitResult3D,
	tip_h: torch.Tensor,
	tip_w: torch.Tensor,
	layer_index: int = 0,
) -> torch.Tensor:
	n = _pred_dt_projection_normals(res)[int(layer_index), tip_h, tip_w].detach()
	return n / n.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def _empty_anticipatory_pull_candidates(
	*,
	device: torch.device,
	dtype: torch.dtype,
	samples_n: int,
	stats: dict[str, float],
	layer_index: int = 0,
) -> dict:
	return {
		"candidate_idx": torch.empty(0, device=device, dtype=torch.long),
		"layer": torch.empty(0, device=device, dtype=torch.long),
		"tip_h": torch.empty(0, device=device, dtype=torch.long),
		"tip_w": torch.empty(0, device=device, dtype=torch.long),
		"root_h": torch.empty(0, device=device, dtype=torch.long),
		"root_w": torch.empty(0, device=device, dtype=torch.long),
		"target_xyz": torch.empty((0, 3), device=device, dtype=dtype),
		"prefix": torch.empty(0, device=device, dtype=dtype),
		"inliers": torch.empty((0, samples_n), device=device, dtype=dtype),
		"offset": torch.empty(0, device=device, dtype=dtype),
		"_stats": stats,
	}


def _prefetch_sparse_pred_dt_chunks(*, res: fit_model.FitResult3D, query: torch.Tensor) -> int:
	sparse_caches = getattr(res.data, "sparse_caches", None)
	if not sparse_caches:
		return 0
	pred_dt_caches = [
		cache for cache in sparse_caches.values()
		if "pred_dt" in set(getattr(cache, "channels", ()))
	]
	if not pred_dt_caches:
		return 0
	for cache in pred_dt_caches:
		spacing = res.data._spacing_for(cache.channels[0])
		cache.prefetch(query, res.data.origin_fullres, spacing)
	for cache in pred_dt_caches:
		cache.sync()
	return len(pred_dt_caches)


def _score_anticipatory_pull_candidates(
	*,
	res: fit_model.FitResult3D,
	cfg: dict,
	flow_weight: torch.Tensor | None = None,
	mask_lr: torch.Tensor | None = None,
	layer_index: int = 0,
) -> dict | None:
	"""Fit one-step root->tip candidates that can actually use the known gate."""
	d = int(layer_index)
	if d < 0 or d >= int(res.xyz_lr.shape[0]):
		return None
	xyz0 = res.xyz_lr[d].detach()
	Hm, Wm = int(xyz0.shape[0]), int(xyz0.shape[1])
	if Hm <= 1 or Wm <= 1:
		return None
	samples_n = max(2, int(cfg.get("samples", 8)))
	inlier_zero = float(cfg.get("inlier_zero", 80.0))
	inlier_one = float(cfg.get("inlier_one", 120.0))
	if inlier_one <= inlier_zero:
		raise ValueError("anticipatory_pull requires inlier_one > inlier_zero")
	chunk_candidates = max(256, int(cfg.get("chunk_candidates", 65536)))
	tip_h, tip_w, root_h, root_w = _neighbor_candidate_indices(Hm=Hm, Wm=Wm, device=xyz0.device)
	total_candidates = int(tip_h.numel())
	stats: dict[str, float] = {
		"total_candidates": float(total_candidates),
		"gate_candidates": float(total_candidates),
		"scored_candidates": 0.0,
		"active_batches": 0.0,
		"chunk_candidates": float(chunk_candidates),
		"query_samples": 0.0,
		"sparse_prefetch_batches": 0.0,
	}
	if flow_weight is not None:
		root_weight = flow_weight[0, 0, root_h, root_w].detach()
		tip_weight = flow_weight[0, 0, tip_h, tip_w].detach()
		if mask_lr is not None:
			tip_mask = mask_lr[0, 0, tip_h, tip_w].detach()
		else:
			tip_mask = torch.ones_like(tip_weight)
		gate_candidate = (root_weight > 0.0) & (tip_weight < 1.0) & (root_weight > tip_weight) & (tip_mask > 0.0)
		stats["gate_candidates"] = float(gate_candidate.sum().detach().cpu())
		if not bool(gate_candidate.any().detach().cpu()):
			return _empty_anticipatory_pull_candidates(
				device=xyz0.device,
				dtype=xyz0.dtype,
				samples_n=samples_n,
				stats=stats,
				layer_index=d,
			)
		tip_h = tip_h[gate_candidate]
		tip_w = tip_w[gate_candidate]
		root_h = root_h[gate_candidate]
		root_w = root_w[gate_candidate]
	n_candidates = int(tip_h.numel())
	if n_candidates <= 0:
		return _empty_anticipatory_pull_candidates(
			device=xyz0.device,
			dtype=xyz0.dtype,
			samples_n=samples_n,
			stats=stats,
			layer_index=d,
		)
	offset_factors = _anticipatory_normal_offset_factors(cfg=cfg, device=xyz0.device, dtype=xyz0.dtype)
	ref_step = _anticipatory_reference_step(cfg=cfg, device=xyz0.device, dtype=xyz0.dtype, params=res.params)
	if offset_factors.ndim != 1:
		raise RuntimeError(f"anticipatory_pull expected 1D offset factors, got {tuple(offset_factors.shape)}")
	offset_count = int(offset_factors.numel())
	t = torch.linspace(0.0, 1.0, samples_n, device=xyz0.device, dtype=xyz0.dtype).view(1, 1, samples_n, 1)
	targets: list[torch.Tensor] = []
	prefixes: list[torch.Tensor] = []
	best_inliers: list[torch.Tensor] = []
	best_offsets_all: list[torch.Tensor] = []
	batches = 0
	sparse_prefetch_batches = 0
	with torch.no_grad():
		for c0 in range(0, n_candidates, chunk_candidates):
			c1 = min(n_candidates, c0 + chunk_candidates)
			th = tip_h[c0:c1]
			tw = tip_w[c0:c1]
			rh = root_h[c0:c1]
			rw = root_w[c0:c1]
			root = xyz0[rh, rw]
			tip = xyz0[th, tw]
			line_vec = tip - root
			n = _tip_normals_from_result(res=res, tip_h=th, tip_w=tw, layer_index=d)
			offsets = ref_step * offset_factors
			target_vec = line_vec.view(-1, 1, 3) + offsets.view(1, -1, 1) * n.view(-1, 1, 3)
			query = root.view(-1, 1, 1, 3) + t * target_vec.view(c1 - c0, offset_count, 1, 3)
			flat_query = query.reshape(1, 1, -1, 3)
			if _prefetch_sparse_pred_dt_chunks(res=res, query=flat_query) > 0:
				sparse_prefetch_batches += 1
			sampled = res.data.grid_sample_fullres(flat_query, channels={"pred_dt"}).pred_dt
			if sampled is None:
				raise RuntimeError("anticipatory_pull requires pred_dt to be loaded")
			pred = sampled.reshape(c1 - c0, offset_count, samples_n)
			inlier = ((pred - inlier_zero) / (inlier_one - inlier_zero)).clamp(0.0, 1.0)
			prefix = inlier.cumprod(dim=2).mean(dim=2)
			best_score, best_offset_idx = prefix.max(dim=1)
			best_offsets = offsets[best_offset_idx]
			targets.append(tip + best_offsets.unsqueeze(-1) * n)
			prefixes.append(best_score)
			best_inliers.append(inlier[torch.arange(c1 - c0, device=xyz0.device), best_offset_idx])
			best_offsets_all.append(best_offsets)
			batches += 1
	stats.update({
		"scored_candidates": float(n_candidates),
		"active_batches": float(batches),
		"query_samples": float(n_candidates * offset_count * samples_n),
		"sparse_prefetch_batches": float(sparse_prefetch_batches),
	})
	candidate_idx = torch.arange(n_candidates, device=xyz0.device, dtype=torch.long)
	return {
		"candidate_idx": candidate_idx,
		"layer": torch.full((n_candidates,), d, device=xyz0.device, dtype=torch.long),
		"tip_h": tip_h,
		"tip_w": tip_w,
		"root_h": root_h,
		"root_w": root_w,
		"target_xyz": torch.cat(targets, dim=0),
		"prefix": torch.cat(prefixes, dim=0),
		"inliers": torch.cat(best_inliers, dim=0),
		"offset": torch.cat(best_offsets_all, dim=0),
		"_stats": stats,
	}


def _activate_anticipatory_pull(
	*,
	candidates: dict | None,
	flow_weight: torch.Tensor,
	mask_lr: torch.Tensor,
	cfg: dict,
	weight_scale: float = 1.0,
) -> dict | None:
	if candidates is None:
		return None
	tip_h = candidates["tip_h"]
	tip_w = candidates["tip_w"]
	root_h = candidates["root_h"]
	root_w = candidates["root_w"]
	prefix = candidates["prefix"].detach()
	root_weight = flow_weight[0, 0, root_h, root_w].detach()
	tip_weight = flow_weight[0, 0, tip_h, tip_w].detach()
	tip_mask = mask_lr[0, 0, tip_h, tip_w].detach()
	active = (tip_weight < 1.0) & (root_weight > 0.0) & (root_weight > tip_weight) & (prefix > 0.0) & (tip_mask > 0.0)
	if not bool(active.any().detach().cpu()):
		return {
			"candidate_idx": candidates["candidate_idx"][:0],
			"layer": candidates.get("layer", candidates["tip_h"])[:0],
			"tip_h": tip_h[:0],
			"tip_w": tip_w[:0],
			"root_h": root_h[:0],
			"root_w": root_w[:0],
			"target_xyz": candidates["target_xyz"][:0],
			"candidate_weight": prefix[:0],
			"prefix": prefix[:0],
			"root_weight": root_weight[:0],
		}
	loss_weight = float(cfg.get("loss_weight", 1.0))
	candidate_weight = root_weight[active] * prefix[active] * tip_mask[active] * loss_weight * float(weight_scale)
	return {
		"candidate_idx": candidates["candidate_idx"][active],
		"layer": candidates.get("layer", torch.zeros_like(candidates["candidate_idx"]))[active],
		"tip_h": tip_h[active],
		"tip_w": tip_w[active],
		"root_h": root_h[active],
		"root_w": root_w[active],
		"target_xyz": candidates["target_xyz"][active].detach(),
		"candidate_weight": candidate_weight.detach(),
		"prefix": prefix[active].detach(),
		"root_weight": root_weight[active].detach(),
	}


def _anticipatory_pull_loss_map(*, res: fit_model.FitResult3D, pull: dict | None) -> torch.Tensor:
	out = torch.zeros_like(res.mask_lr)
	if pull is None or int(pull["tip_h"].numel()) == 0:
		return out
	layer = pull.get("layer", torch.zeros_like(pull["tip_h"]))
	layer = layer.to(device=res.xyz_lr.device, dtype=torch.long)
	tip_h = pull["tip_h"].to(device=res.xyz_lr.device, dtype=torch.long)
	tip_w = pull["tip_w"].to(device=res.xyz_lr.device, dtype=torch.long)
	live_tip = res.xyz_lr[layer, tip_h, tip_w]
	target = pull["target_xyz"].to(device=live_tip.device, dtype=live_tip.dtype)
	candidate_weight = pull["candidate_weight"].to(device=live_tip.device, dtype=live_tip.dtype)
	per_candidate = F.smooth_l1_loss(live_tip, target, reduction="none").sum(dim=-1)
	idx = layer * int(res.xyz_lr.shape[1]) * int(res.xyz_lr.shape[2]) + tip_h * int(res.xyz_lr.shape[2]) + tip_w
	out_flat = out.reshape(-1)
	out_flat.index_add_(0, idx, per_candidate * candidate_weight)
	return out


def _anticipatory_pull_debug_lr(
	*,
	pull: dict | None,
	Hm: int,
	Wm: int,
	device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	weight = torch.zeros(Hm, Wm, device=device, dtype=torch.float32)
	prefix = torch.zeros(Hm, Wm, device=device, dtype=torch.float32)
	root_weight = torch.zeros(Hm, Wm, device=device, dtype=torch.float32)
	if pull is None or int(pull["tip_h"].numel()) == 0:
		return weight, prefix, root_weight
	idx = pull["tip_h"] * Wm + pull["tip_w"]
	flat_weight = weight.reshape(-1)
	flat_prefix = prefix.reshape(-1)
	flat_root_weight = root_weight.reshape(-1)
	flat_weight.index_add_(0, idx, pull["candidate_weight"].to(device=device, dtype=torch.float32))
	flat_prefix.index_add_(0, idx, pull["prefix"].to(device=device, dtype=torch.float32))
	flat_root_weight.index_add_(0, idx, pull["root_weight"].to(device=device, dtype=torch.float32))
	count = torch.zeros(Hm * Wm, device=device, dtype=torch.float32)
	count.index_add_(0, idx, torch.ones_like(idx, dtype=torch.float32))
	den = count.clamp_min(1.0)
	flat_prefix /= den
	flat_root_weight /= den
	return weight, prefix, root_weight


def _anticipatory_pull_overlay(
	*,
	pull: dict | None,
	height: int,
	width: int,
	sub_h: int,
	sub_w: int,
) -> np.ndarray | None:
	if pull is None or int(pull["tip_h"].numel()) == 0:
		return None
	img = np.zeros((height, width, 3), dtype=np.float32)
	try:
		import cv2
	except Exception:
		cv2 = None
	tip_h = pull["tip_h"].detach().cpu().numpy()
	tip_w = pull["tip_w"].detach().cpu().numpy()
	root_h = pull["root_h"].detach().cpu().numpy()
	root_w = pull["root_w"].detach().cpu().numpy()
	candidate_weight = pull["candidate_weight"].detach().cpu().numpy()
	if candidate_weight.size == 0:
		return img
	scale = float(candidate_weight.max()) if float(candidate_weight.max()) > 0.0 else 1.0
	for th, tw, rh, rw, cw in zip(tip_h, tip_w, root_h, root_w, candidate_weight):
		x0 = int(round(float(rw) * float(sub_w)))
		y0 = int(round(float(rh) * float(sub_h)))
		x1 = int(round(float(tw) * float(sub_w)))
		y1 = int(round(float(th) * float(sub_h)))
		x0 = max(0, min(width - 1, x0))
		x1 = max(0, min(width - 1, x1))
		y0 = max(0, min(height - 1, y0))
		y1 = max(0, min(height - 1, y1))
		color = (0.0, 255.0 * min(1.0, float(cw) / scale), 255.0)
		if cv2 is not None:
			cv2.line(img, (x0, y0), (x1, y1), color, 1, lineType=cv2.LINE_AA)
		else:
			steps = max(abs(x1 - x0), abs(y1 - y0), 1)
			for i in range(steps + 1):
				a = i / steps
				x = int(round((1.0 - a) * x0 + a * x1))
				y = int(round((1.0 - a) * y0 + a * y1))
				img[y, x] = color
	return img


def _anticipatory_debug_points(cfg: dict) -> list[tuple[int, int]]:
	points = cfg.get("debug_points", None)
	if points is None:
		points = cfg.get("debug_fit_points", None)
	if not isinstance(points, (list, tuple)):
		return []
	out: list[tuple[int, int]] = []
	for p in points:
		if not isinstance(p, (list, tuple)) or len(p) < 2:
			continue
		try:
			out.append((int(p[0]), int(p[1])))
		except Exception:
			continue
	return out


def _anticipatory_debug_roi_center(cfg: dict) -> tuple[float, float, float] | None:
	center = cfg.get("debug_roi_center_xyz", None)
	if center is None:
		center = cfg.get("debug_center_xyz", None)
	if not isinstance(center, (list, tuple)) or len(center) < 3:
		return None
	try:
		return (float(center[0]), float(center[1]), float(center[2]))
	except Exception:
		return None


def _write_anticipatory_fit_debug_mosaic(
	*,
	res: fit_model.FitResult3D,
	cfg: dict,
	candidates: dict | None,
	pull: dict | None,
	flow_weight: torch.Tensor | None,
	stage_name: str,
	debug_index: int,
	out_dir: Path | None,
) -> None:
	points = _anticipatory_debug_points(cfg)
	roi_center = _anticipatory_debug_roi_center(cfg)
	if out_dir is None or (not points and roi_center is None) or candidates is None:
		return
	try:
		import cv2
	except Exception as exc:
		print(f"[pred_dt_flow_gate] anticipatory fit debug skipped: cv2 import failed: {exc}", flush=True)
		return
	xyz0 = res.xyz_lr[0].detach()
	Hm, Wm = int(xyz0.shape[0]), int(xyz0.shape[1])
	if Hm <= 1 or Wm <= 1:
		return
	normals = _vertex_normals(res.xyz_lr.detach())[0]
	up = max(1, int(cfg.get("debug_slice_upsample", 8)))
	slice_w = max(24, int(cfg.get("debug_slice_width", 96)))
	slice_h = max(16, int(cfg.get("debug_slice_height", 48)))
	inlier_zero = float(cfg.get("inlier_zero", 80.0))
	inlier_one = float(cfg.get("inlier_one", 120.0))
	if inlier_one <= inlier_zero:
		return
	active_idx = None
	if pull is not None and int(pull["candidate_idx"].numel()) > 0:
		active_idx = pull["candidate_idx"].detach()

	def _candidate_for_point(h: int, w: int) -> int | None:
		if h < 0 or h >= Hm or w < 0 or w >= Wm:
			return None
		tip_match = (candidates["tip_h"] == h) & (candidates["tip_w"] == w)
		if active_idx is not None and int(active_idx.numel()) > 0:
			active_match = tip_match[active_idx]
			if bool(active_match.any().detach().cpu()):
				sub_idx = active_idx[active_match]
				prefix = candidates["prefix"][sub_idx]
				return int(sub_idx[int(torch.argmax(prefix).detach().cpu())].detach().cpu())
		if not bool(tip_match.any().detach().cpu()):
			return None
		sub_idx = tip_match.nonzero(as_tuple=True)[0]
		prefix = candidates["prefix"][sub_idx]
		return int(sub_idx[int(torch.argmax(prefix).detach().cpu())].detach().cpu())

	debug_candidates: list[tuple[int | None, str]] = []
	for tip_h, tip_w in points:
		cand_i = _candidate_for_point(tip_h, tip_w)
		debug_candidates.append((cand_i, f"tip=({tip_h},{tip_w})"))

	if roi_center is not None and flow_weight is not None:
		k = max(1, int(cfg.get("debug_roi_k", 8)))
		root_threshold = float(cfg.get("debug_roi_root_min", 0.5))
		tip_threshold = float(cfg.get("debug_roi_tip_max", 0.5))
		center_t = torch.tensor(roi_center, device=xyz0.device, dtype=xyz0.dtype)
		rh = candidates["root_h"]
		rw = candidates["root_w"]
		th = candidates["tip_h"]
		tw = candidates["tip_w"]
		root_weight = flow_weight[0, 0, rh, rw].detach()
		tip_weight = flow_weight[0, 0, th, tw].detach()
		eligible = (root_weight > root_threshold) & (tip_weight < tip_threshold)
		if bool(eligible.any().detach().cpu()):
			dist2 = ((xyz0[rh, rw] - center_t.view(1, 3)) ** 2).sum(dim=-1)
			dist2 = dist2.masked_fill(~eligible, float("inf"))
			n_pick = min(k, int(eligible.sum().detach().cpu()))
			picked = torch.topk(-dist2, k=n_pick, largest=True).indices
			for cand_t in picked:
				cand_i = int(cand_t.detach().cpu())
				d = float(dist2[cand_i].sqrt().detach().cpu())
				debug_candidates.append((cand_i, f"roi d={d:.1f}"))

	tiles: list[np.ndarray] = []
	xs_unit = torch.linspace(-0.25, 1.25, slice_w, device=xyz0.device, dtype=xyz0.dtype)
	ys_unit = torch.linspace(-1.0, 1.0, slice_h, device=xyz0.device, dtype=xyz0.dtype)
	def draw_text_bg(img: np.ndarray, text: str, org: tuple[int, int], font_scale: float, color: tuple[int, int, int]) -> None:
		font = cv2.FONT_HERSHEY_SIMPLEX
		thickness = 1
		(size_w, size_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
		x, y = org
		x0 = max(0, x - 2)
		y0 = max(0, y - size_h - 3)
		x1 = min(img.shape[1] - 1, x + size_w + 2)
		y1 = min(img.shape[0] - 1, y + baseline + 3)
		cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 0), -1)
		cv2.putText(img, text, org, font, font_scale, color, thickness, cv2.LINE_AA)

	for cand_i, label in debug_candidates:
		if cand_i is None:
			tile = np.zeros((slice_h * up, slice_w * up, 3), dtype=np.uint8)
			draw_text_bg(tile, f"{label} no cand", (6, 18), 0.45, (255, 255, 255))
			tiles.append(tile)
			continue
		rh = int(candidates["root_h"][cand_i].detach().cpu())
		rw = int(candidates["root_w"][cand_i].detach().cpu())
		th = int(candidates["tip_h"][cand_i].detach().cpu())
		tw = int(candidates["tip_w"][cand_i].detach().cpu())
		root = xyz0[rh, rw]
		tip = xyz0[th, tw]
		line = tip - root
		line_len = line.norm().clamp_min(1e-6)
		line_dir = line / line_len
		normal = _tip_normals_from_result(
			res=res,
			tip_h=torch.tensor([th], device=xyz0.device, dtype=torch.long),
			tip_w=torch.tensor([tw], device=xyz0.device, dtype=torch.long),
		)[0]
		normal = normal / normal.norm().clamp_min(1e-6)
		offset_factors = _anticipatory_normal_offset_factors(cfg=cfg, device=xyz0.device, dtype=xyz0.dtype)
		ref_step = _anticipatory_reference_step(cfg=cfg, device=xyz0.device, dtype=xyz0.dtype, params=res.params)
		search_radius = float((ref_step * offset_factors.abs().max()).detach().cpu()) if offset_factors.numel() > 0 else float(ref_step.detach().cpu())
		search_radius = max(search_radius, 1.0e-6)
		xs = xs_unit * line_len
		ys = ys_unit * search_radius
		grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
		query = root.view(1, 1, 3) + grid_x.unsqueeze(-1) * line_dir.view(1, 1, 3) + grid_y.unsqueeze(-1) * normal.view(1, 1, 3)
		sampled = res.data.grid_sample_fullres(
			query.view(1, slice_h, slice_w, 3),
			channels={"pred_dt"},
		).pred_dt
		if sampled is None:
			continue
		pred = sampled.squeeze(0).squeeze(0).squeeze(0)
		gray = ((pred - inlier_zero) / (inlier_one - inlier_zero)).clamp(0.0, 1.0)
		img = (gray.detach().cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
		rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
		rgb = cv2.resize(rgb, (slice_w * up, slice_h * up), interpolation=cv2.INTER_NEAREST)
		y0 = int(round((0.0 + 1.0) * 0.5 * (slice_h - 1) * up))
		best_offset = float(candidates["offset"][cand_i].detach().cpu()) if "offset" in candidates else 0.0
		x_root = int(round((0.0 + 0.25) / 1.5 * (slice_w - 1) * up))
		x_tip = int(round((1.0 + 0.25) / 1.5 * (slice_w - 1) * up))
		y_tip_offset = int(round(((best_offset / search_radius) + 1.0) * 0.5 * (slice_h - 1) * up))
		cv2.line(rgb, (x_root, y0), (x_tip, y0), (140, 140, 140), 1, cv2.LINE_AA)
		cv2.circle(rgb, (x_root, y0), 3, (255, 128, 0), -1, cv2.LINE_AA)
		cv2.circle(rgb, (x_tip, y0), 3, (255, 0, 128), -1, cv2.LINE_AA)
		draw_text_bg(rgb, "R0", (max(0, x_root - 8), max(10, y0 - 6)), 0.28, (255, 128, 0))
		draw_text_bg(rgb, "T0", (min(rgb.shape[1] - 20, x_tip + 3), max(10, y0 - 6)), 0.28, (255, 0, 128))
		cv2.line(rgb, (x_root, y0), (x_tip, y_tip_offset), (0, 255, 255), 1, cv2.LINE_AA)
		inliers = candidates["inliers"][cand_i].detach().cpu().numpy()
		baseline_scores: list[float] | None = None
		if 0 <= y0 // up < gray.shape[0]:
			baseline_scores = []
		for si, score in enumerate(inliers):
			a = si / max(1, len(inliers) - 1)
			xp = int(round(((a + 0.25) / 1.5) * (slice_w - 1) * up))
			yp = int(round(((a * best_offset / search_radius) + 1.0) * 0.5 * (slice_h - 1) * up))
			cv2.circle(rgb, (xp, yp), 2, (0, 0, 255), -1, cv2.LINE_AA)
			if si < 8:
				draw_text_bg(rgb, f"{score:.2f}", (max(0, xp - 14), min(rgb.shape[0] - 4, yp + 16 + (si % 2) * 12)), 0.28, (255, 255, 255))
			if baseline_scores is not None:
				xg = max(0, min(gray.shape[1] - 1, int(round(xp / up))))
				yg = max(0, min(gray.shape[0] - 1, int(round(y0 / up))))
				baseline_scores.append(float(gray[yg, xg].detach().cpu()))
				cv2.circle(rgb, (xp, y0), 2, (255, 180, 0), -1, cv2.LINE_AA)
				if si < 8:
					draw_text_bg(rgb, f"{baseline_scores[-1]:.2f}", (max(0, xp - 14), max(10, y0 - 10 - (si % 2) * 12)), 0.28, (255, 180, 0))
		prefix = float(candidates["prefix"][cand_i].detach().cpu())
		draw_text_bg(rgb, f"{label} tip=({th},{tw}) root=({rh},{rw})", (6, 12), 0.35, (255, 255, 255))
		eq_angle = np.rad2deg(np.arctan2(best_offset, float(ref_step.detach().cpu())))
		draw_text_bg(rgb, f"line={prefix:.3f} off={best_offset:.1f} eq={eq_angle:.1f}", (6, 26), 0.35, (0, 255, 255))
		cv2.line(rgb, (0, y0), (rgb.shape[1] - 1, y0), (96, 96, 96), 1, cv2.LINE_AA)
		tiles.append(rgb)
	if not tiles:
		return
	n = len(tiles)
	cols = max(1, int(np.ceil(np.sqrt(2.0 * n))))
	rows = int(np.ceil(n / cols))
	tile_h, tile_w = tiles[0].shape[:2]
	mosaic = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)
	for i, tile in enumerate(tiles):
		r = i // cols
		c = i % cols
		mosaic[r * tile_h:(r + 1) * tile_h, c * tile_w:(c + 1) * tile_w] = tile
	out_dir.mkdir(parents=True, exist_ok=True)
	for suffix in (f"{stage_name}_{debug_index:06d}", stage_name):
		path = out_dir / f"pred_dt_flow_gate_{suffix}_anticipatory_fit_points.jpg"
		cv2.imwrite(str(path), mosaic)


def configure_flow_gate(
	*,
	cfg: dict | None,
	stage_name: str,
	seed_xyz: tuple[float, float, float] | None,
	out_dir: str | None,
	capture_channels: bool = False,
) -> None:
	global _flow_gate_cfg, _flow_gate_stage, _flow_gate_seed_xyz, _flow_gate_out_dir, _flow_gate_debug_counts, _flow_gate_last_stats, _flow_gate_last_channels, _flow_gate_seed_hw_cache, _atlas_snap_report_legend_printed
	_flow_gate_last_stats = {}
	_flow_gate_last_channels = None
	_flow_gate_seed_hw_cache = None
	_atlas_snap_report_legend_printed = set()
	if not isinstance(cfg, dict) or not bool(cfg.get("enabled", False)):
		_flow_gate_cfg = None
		_flow_gate_stage = str(stage_name)
		_flow_gate_seed_xyz = seed_xyz
		_flow_gate_out_dir = Path(out_dir) if out_dir else None
		return
	_flow_gate_cfg = dict(cfg)
	_flow_gate_cfg["_capture_tifxyz_channels"] = bool(capture_channels)
	_flow_gate_stage = str(stage_name)
	_flow_gate_seed_xyz = seed_xyz
	debug_out_dir = _flow_gate_cfg.get("debug_out_dir", None)
	_flow_gate_out_dir = Path(debug_out_dir) if debug_out_dir else (Path(out_dir) if out_dir else None)
	_flow_gate_debug_counts[str(stage_name)] = 0


def _debug_interval(cfg: dict, *names: str) -> int:
	for name in names:
		if name in cfg:
			return max(0, int(cfg.get(name, 0)))
	return 0


def configure_pred_dt(*, normal_source: str | None = None) -> None:
	global _pred_dt_normal_source
	src = "model" if normal_source is None else str(normal_source)
	if src not in {"model", "gt"}:
		raise ValueError("pred_dt normal_source must be 'model' or 'gt'")
	_pred_dt_normal_source = src


def _xy_key(value: object) -> tuple[int, int] | None:
	try:
		x, y = value  # type: ignore[misc]
	except Exception:
		return None
	try:
		return int(x), int(y)
	except (TypeError, ValueError):
		return None


def _unique_xy_rows(*arrays: np.ndarray) -> np.ndarray:
	out: list[tuple[int, int]] = []
	seen: set[tuple[int, int]] = set()
	for arr in arrays:
		xy = np.asarray(arr, dtype=np.int32).reshape(-1, 2)
		for x, y in xy:
			key = (int(x), int(y))
			if key in seen:
				continue
			seen.add(key)
			out.append(key)
	if not out:
		return np.zeros((0, 2), dtype=np.int32)
	return np.asarray(out, dtype=np.int32)


_ATLAS_SNAP_REPORT_COLUMNS = (
	ProgressColumn("idx", "idx", "atlas snap sample index", min_width=3),
	ProgressColumn("src", "src", "source line point index", min_width=4),
	ProgressColumn("x", "x", "render-space x pixel", min_width=3),
	ProgressColumn("y", "y", "render-space y pixel", min_width=3),
	ProgressColumn("ok", "ok", "accepted as a flow seed candidate", min_width=2),
	ProgressColumn(
		"use",
		"use",
		"flow source use: pri primary, ext extra, acc accepted but not used, no rejected",
		min_width=3,
	),
	ProgressColumn("why", "why", "accept/reject reason", min_width=6),
	ProgressColumn("dist", "dist", "surface projection distance", min_width=7),
	ProgressColumn("dt", "dt", "pred-dt value at seed pixel", min_width=5),
	ProgressColumn("inl", "inl", "pred-dt inlier score", min_width=5),
	ProgressColumn("area", "area", "flow component pixel count under this source", min_width=5),
	ProgressColumn("obj", "obj", "object alias; see alias lines above", min_width=3),
)


def _atlas_snap_report_float(value: object) -> str:
	if value is None:
		return "-"
	try:
		v = float(value)
	except (TypeError, ValueError):
		return "-"
	if not np.isfinite(v):
		return "-"
	return format_progress_value(v)


def _atlas_snap_report_bool(value: object) -> str:
	return "Y" if bool(value) else "N"


def _atlas_snap_report_use(value: object) -> str:
	raw = str(value)
	return {
		"primary": "pri",
		"extra": "ext",
		"accepted_not_passed_to_flow": "acc",
		"not_used": "no",
	}.get(raw, raw[:3] if raw else "-")


def _atlas_snap_report_obj(value: object) -> str:
	raw = str(value) if value is not None else "-"
	if not raw or raw == "None":
		return "-"
	return Path(raw).name


def _atlas_snap_report_row_values(
	rec: dict[str, object],
	*,
	object_aliases: dict[str, str],
) -> dict[str, str]:
	xy = _xy_key(rec.get("source_xy"))
	obj_name = _atlas_snap_report_obj(rec.get("object_id"))
	return {
		"idx": str(rec.get("sample_index", "-")),
		"src": str(rec.get("source_index", "-")),
		"x": "-" if xy is None else str(int(xy[0])),
		"y": "-" if xy is None else str(int(xy[1])),
		"ok": _atlas_snap_report_bool(rec.get("accepted", False)),
		"use": _atlas_snap_report_use(rec.get("flow_source")),
		"why": str(rec.get("reason", "-"))[:16] or "-",
		"dist": _atlas_snap_report_float(rec.get("surface_distance")),
		"dt": _atlas_snap_report_float(rec.get("pred_dt_value")),
		"inl": _atlas_snap_report_float(rec.get("inlier")),
		"area": _atlas_snap_report_float(rec.get("seed_area")),
		"obj": object_aliases.get(obj_name, obj_name),
	}


def _flow_component_area_at_xy(
	labels: np.ndarray | None,
	xy: tuple[int, int] | None,
) -> tuple[int | None, int | None]:
	if labels is None or xy is None:
		return None, None
	arr = np.asarray(labels)
	if arr.ndim != 2:
		return None, None
	x, y = int(xy[0]), int(xy[1])
	if y < 0 or x < 0 or y >= int(arr.shape[0]) or x >= int(arr.shape[1]):
		return None, None
	label = int(round(float(arr[y, x])))
	if label <= 0:
		return label, 0
	return label, int(np.count_nonzero(np.rint(arr).astype(np.int64, copy=False) == label))


def _print_atlas_snap_seed_report_table(
	*,
	stage_name: str,
	layer_index: int,
	winding: int,
	records: list[dict[str, object]],
	accepted_count: int,
	used_count: int,
	flow_status: str,
) -> None:
	global _atlas_snap_report_legend_printed
	prefix = "[pred_dt_flow_gate]"
	print(
		f"{prefix} {stage_name}: atlas snap seeds "
		f"layer={layer_index} winding={winding} samples={len(records)} "
		f"accepted={accepted_count} used={used_count} flow={flow_status}",
		flush=True,
	)
	legend_key = (str(stage_name), int(layer_index))
	if legend_key not in _atlas_snap_report_legend_printed:
		print_progress_legend(
			prefix=prefix,
			items=[(col.label, col.description) for col in _ATLAS_SNAP_REPORT_COLUMNS],
			cols_per_row=2,
		)
		_atlas_snap_report_legend_printed.add(legend_key)
	object_names = sorted({
		_atlas_snap_report_obj(rec.get("object_id"))
		for rec in records
		if _atlas_snap_report_obj(rec.get("object_id")) != "-"
	})
	object_aliases = {name: f"o{i}" for i, name in enumerate(object_names)}
	for name in object_names:
		print(f"{prefix} atlas snap obj {object_aliases[name]}={name}", flush=True)
	row_values = [
		_atlas_snap_report_row_values(rec, object_aliases=object_aliases)
		for rec in records
	]
	widths = progress_widths(_ATLAS_SNAP_REPORT_COLUMNS, {})
	for values in row_values:
		for col in _ATLAS_SNAP_REPORT_COLUMNS:
			widths[col.key] = max(int(widths[col.key]), len(values.get(col.key, "")))
	print(f"{prefix} {progress_header(_ATLAS_SNAP_REPORT_COLUMNS, widths)}", flush=True)
	for values in row_values:
		print(f"{prefix} {progress_row(_ATLAS_SNAP_REPORT_COLUMNS, widths, values)}", flush=True)


def _write_atlas_snap_seed_report(
	*,
	stage_name: str,
	debug_index: int,
	layer_index: int,
	winding: int,
	atlas_seed_debug: dict[str, object] | None,
	primary_xy: tuple[int, int] | None,
	extra_source_xy: np.ndarray,
	flow_status: str,
	flow_source_value: int | None = None,
	source_component_mask_hr: np.ndarray | None = None,
	out_dir: Path | None,
) -> None:
	if not atlas_seed_debug:
		return
	records_raw = atlas_seed_debug.get("records", [])
	if not isinstance(records_raw, list):
		return
	primary_key = None if primary_xy is None else (int(primary_xy[0]), int(primary_xy[1]))
	extra_keys = {
		(int(x), int(y))
		for x, y in np.asarray(extra_source_xy, dtype=np.int32).reshape(-1, 2)
	}
	records: list[dict[str, object]] = []
	accepted_count = 0
	used_count = 0
	for item in records_raw:
		if not isinstance(item, dict):
			continue
		rec = dict(item)
		xy = _xy_key(rec.get("source_xy"))
		accepted = bool(rec.get("accepted", False))
		if accepted:
			accepted_count += 1
		flow_source = "not_used"
		if accepted and xy is not None:
			if primary_key is not None and xy == primary_key:
				flow_source = "primary"
			elif xy in extra_keys:
				flow_source = "extra"
			else:
				flow_source = "accepted_not_passed_to_flow"
		if flow_source in {"primary", "extra"}:
			used_count += 1
		rec["flow_source"] = flow_source
		rec["flow_status"] = flow_status
		if flow_source_value is not None:
			rec["flow_source_value"] = int(flow_source_value)
		label, area = _flow_component_area_at_xy(source_component_mask_hr, xy)
		if label is not None:
			rec["flow_component_label"] = int(label)
		if area is not None:
			rec["seed_area"] = int(area)
		records.append(rec)
	if not records:
		return

	_print_atlas_snap_seed_report_table(
		stage_name=str(stage_name),
		layer_index=int(layer_index),
		winding=int(winding),
		records=records,
		accepted_count=int(accepted_count),
		used_count=int(used_count),
		flow_status=str(flow_status),
	)


def flow_gate_last_stats() -> dict[str, float]:
	return dict(_flow_gate_last_stats)


def flow_gate_last_timing() -> dict[str, float]:
	return dict(_flow_gate_last_timing)


def flow_gate_last_channels() -> dict | None:
	if _flow_gate_last_channels is None:
		return None
	out = dict(_flow_gate_last_channels)
	for name in ("flow_gate_local_contrast", "flow_gate_component_normalized"):
		value = out.get(name)
		if isinstance(value, torch.Tensor):
			out[name] = value.detach().cpu().clone()
	return out


def _normalize_positive_debug_image(image: np.ndarray | None) -> np.ndarray | None:
	if image is None:
		return None
	arr = np.asarray(image, dtype=np.float32)
	if arr.ndim != 2:
		return None
	finite = np.isfinite(arr)
	positive = finite & (arr > 0.0)
	out = np.zeros_like(arr, dtype=np.float32)
	if not positive.any():
		return out
	scale = float(arr[positive].max())
	if scale <= 0.0:
		return out
	out[finite] = np.clip(np.maximum(arr[finite], 0.0) / scale, 0.0, 1.0)
	return out


def _flow_seed_overlay_panel(
	base_u8: np.ndarray,
	*,
	source_xy: tuple[int, int] | None,
	extra_source_xy: np.ndarray | None = None,
	corr_seed_debug: dict[str, np.ndarray] | None,
	source_edge_mask: np.ndarray | None,
	flow_metadata: dict | None,
) -> np.ndarray:
	panel = np.repeat(np.asarray(base_u8, dtype=np.uint8)[..., None], 3, axis=2)
	if panel.ndim != 3 or panel.shape[2] != 3:
		return panel
	H, W, _ = panel.shape

	try:
		import cv2
	except Exception:
		cv2 = None
	pil_image = None
	pil_draw = None
	if cv2 is None:
		try:
			from PIL import Image, ImageDraw
			pil_image = Image.fromarray(panel, mode="RGB")
			pil_draw = ImageDraw.Draw(pil_image)
		except Exception:
			pil_image = None
			pil_draw = None

	def put_text(text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
		if cv2 is not None:
			cv2.putText(
				panel,
				text,
				(max(0, min(W - 1, x)), max(8, min(H - 1, y))),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.35,
				color,
				1,
				cv2.LINE_AA,
			)
		elif pil_draw is not None:
			pil_draw.text((max(0, min(W - 1, x)), max(0, min(H - 1, y - 8))), text, fill=color)

	def draw_circle(x: int, y: int, color: tuple[int, int, int], *, radius: int = 3) -> None:
		if x < 0 or y < 0 or x >= W or y >= H:
			return
		if cv2 is not None:
			cv2.circle(panel, (x, y), radius, color, 1, cv2.LINE_AA)
		elif pil_draw is not None:
			pil_draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color)
		else:
			for yy in range(max(0, y - radius), min(H, y + radius + 1)):
				for xx in range(max(0, x - radius), min(W, x + radius + 1)):
					if (xx - x) * (xx - x) + (yy - y) * (yy - y) <= radius * radius:
						panel[yy, xx] = color

	def draw_cross(x: int, y: int, color: tuple[int, int, int]) -> None:
		if x < 0 or y < 0 or x >= W or y >= H:
			return
		if cv2 is not None:
			cv2.drawMarker(
				panel,
				(x, y),
				color,
				markerType=cv2.MARKER_CROSS,
				markerSize=11,
				thickness=1,
				line_type=cv2.LINE_AA,
			)
		elif pil_draw is not None:
			pil_draw.line((x, max(0, y - 5), x, min(H - 1, y + 5)), fill=color)
			pil_draw.line((max(0, x - 5), y, min(W - 1, x + 5), y), fill=color)
		else:
			panel[max(0, y - 5):min(H, y + 6), x] = color
			panel[y, max(0, x - 5):min(W, x + 6)] = color

	if source_edge_mask is not None:
		mask = np.asarray(source_edge_mask, dtype=np.float32)
		if mask.shape == (H, W):
			panel[mask > 0.0] = (255, 64, 255)
			if pil_image is not None:
				from PIL import Image, ImageDraw
				pil_image = Image.fromarray(panel, mode="RGB")
				pil_draw = ImageDraw.Draw(pil_image)

	if isinstance(flow_metadata, dict):
		accepted = int(flow_metadata.get("accepted_source_count", 0))
		requested = int(flow_metadata.get("extra_source_count", 0)) + 1
		source_edges = int(flow_metadata.get("source_edge_count", 0))
		seeded_nodes = int(flow_metadata.get("seeded_node_count", 0))
		put_text(
			f"c++ src {accepted}/{requested} edges {source_edges} nodes {seeded_nodes}",
			4,
			28,
			(255, 64, 255),
		)

	if corr_seed_debug:
		xy = np.asarray(corr_seed_debug.get("xy", np.zeros((0, 2), dtype=np.int32)), dtype=np.int32)
		distance = np.asarray(corr_seed_debug.get("distance", np.zeros((0,), dtype=np.float32)), dtype=np.float32)
		valid = np.asarray(corr_seed_debug.get("valid", np.zeros((0,), dtype=np.bool_)), dtype=np.bool_)
		n = min(int(xy.shape[0]) if xy.ndim == 2 else 0, int(distance.shape[0]), int(valid.shape[0]))
		label_limit = min(n, 64)
		for i in range(n):
			x = int(xy[i, 0])
			y = int(xy[i, 1])
			color = (80, 255, 80) if bool(valid[i]) else (255, 96, 32)
			draw_circle(x, y, color, radius=3)
			if i < label_limit:
				put_text(f"{float(distance[i]):.1f}", x + 4, y - 4, color)
		threshold_arr = corr_seed_debug.get("surface_distance")
		if threshold_arr is not None and np.asarray(threshold_arr).size > 0:
			threshold = float(np.asarray(threshold_arr, dtype=np.float32).reshape(-1)[0])
			if np.isfinite(threshold):
				put_text(f"corr <= {threshold:.1f}", 4, 14, (255, 255, 255))

	if source_xy is not None:
		draw_cross(int(source_xy[0]), int(source_xy[1]), (64, 224, 255))
		put_text("src", int(source_xy[0]) + 6, int(source_xy[1]) + 12, (64, 224, 255))
	if extra_source_xy is not None:
		extra_xy = np.asarray(extra_source_xy, dtype=np.int32).reshape(-1, 2)
		for x, y in extra_xy:
			draw_cross(int(x), int(y), (255, 224, 64))
	if pil_image is not None:
		panel = np.asarray(pil_image, dtype=np.uint8)
	return panel


def _write_flow_gate_debug(
	*,
	stage_name: str,
	debug_index: int,
	pred_u8: np.ndarray,
	flow_hr: np.ndarray | None,
	smooth_grid_flow: np.ndarray | None,
	gate_basis_hr: np.ndarray | None,
	graph_edge_flow_rgb: np.ndarray | None,
	island_obstacle_factor_rgb: np.ndarray | None,
	island_removed_mask_hr: np.ndarray | None,
	island_flow_passability_rgb: np.ndarray | None,
	island_propagated_edge_flow_rgb: np.ndarray | None,
	island_bonus_edge_flow_rgb: np.ndarray | None,
	island_tree_dense_no_backtrack_hr: np.ndarray | None,
	island_tree_dense_greedy_ascent_hr: np.ndarray | None,
	source_edge_mask_hr: np.ndarray | None,
	source_component_mask_hr: np.ndarray | None,
	weight_hr: np.ndarray | None,
	out_dir: Path | None,
	pull_weight_hr: np.ndarray | None = None,
	pull_prefix_hr: np.ndarray | None = None,
	pull_root_weight_hr: np.ndarray | None = None,
	pull_overlay_rgb: np.ndarray | None = None,
) -> None:
	if out_dir is None:
		return
	try:
		import tifffile
	except Exception as exc:
		print(f"[pred_dt_flow_gate] debug write skipped: tifffile import failed: {exc}", flush=True)
		return
	out_dir.mkdir(parents=True, exist_ok=True)
	pred_raw_u8 = pred_u8.astype(np.uint8, copy=True)
	flow = (
		np.zeros_like(pred_raw_u8, dtype=np.float32)
		if flow_hr is None
		else np.asarray(flow_hr, dtype=np.float32)
	)
	grid_flow = (
		None
		if smooth_grid_flow is None
		else np.asarray(smooth_grid_flow, dtype=np.float32)
	)
	gate_basis = _normalize_positive_debug_image(gate_basis_hr)
	graph_flow = (
		None
		if graph_edge_flow_rgb is None
		else np.asarray(graph_edge_flow_rgb, dtype=np.float32)
	)
	island_obstacle = (
		None
		if island_obstacle_factor_rgb is None
		else np.asarray(island_obstacle_factor_rgb, dtype=np.float32)
	)
	island_removed = (
		None
		if island_removed_mask_hr is None
		else np.asarray(island_removed_mask_hr, dtype=np.float32)
	)
	island_passability = (
		None
		if island_flow_passability_rgb is None
		else np.asarray(island_flow_passability_rgb, dtype=np.float32)
	)
	island_propagated = (
		None
		if island_propagated_edge_flow_rgb is None
		else np.asarray(island_propagated_edge_flow_rgb, dtype=np.float32)
	)
	island_bonus = (
		None
		if island_bonus_edge_flow_rgb is None
		else np.asarray(island_bonus_edge_flow_rgb, dtype=np.float32)
	)
	island_dense_no_backtrack = (
		None
		if island_tree_dense_no_backtrack_hr is None
		else np.asarray(island_tree_dense_no_backtrack_hr, dtype=np.float32)
	)
	island_dense_greedy = (
		None
		if island_tree_dense_greedy_ascent_hr is None
		else np.asarray(island_tree_dense_greedy_ascent_hr, dtype=np.float32)
	)
	source_edges = (
		None
		if source_edge_mask_hr is None
		else np.asarray(source_edge_mask_hr, dtype=np.float32)
	)
	source_components = (
		None
		if source_component_mask_hr is None
		else np.asarray(source_component_mask_hr, dtype=np.float32)
	)
	weights = (
		np.zeros_like(pred_raw_u8, dtype=np.float32)
		if weight_hr is None
		else np.asarray(weight_hr, dtype=np.float32)
	)
	pull_weight = None if pull_weight_hr is None else np.asarray(pull_weight_hr, dtype=np.float32)
	pull_prefix = None if pull_prefix_hr is None else np.asarray(pull_prefix_hr, dtype=np.float32)
	pull_root_weight = None if pull_root_weight_hr is None else np.asarray(pull_root_weight_hr, dtype=np.float32)
	pull_overlay = None if pull_overlay_rgb is None else np.asarray(pull_overlay_rgb, dtype=np.float32)
	def write_named_layer(tw, image: np.ndarray, *, name: str) -> None:
		arr = np.asarray(image)
		if arr.ndim == 2:
			arr = np.repeat(arr.astype(np.float32, copy=False)[..., None], 3, axis=2)
		elif arr.ndim == 3 and arr.shape[2] == 3:
			arr = arr.astype(np.float32, copy=False)
		else:
			raise ValueError(f"flow gate debug layer {name!r} has unsupported shape {arr.shape}")
		tw.write(
			arr,
			photometric="rgb",
			metadata=None,
			extratags=[(285, "s", 0, name, False)],  # TIFFTAG_PAGENAME
		)
	for suffix in (f"{stage_name}_{debug_index:06d}", stage_name):
		layers_path = out_dir / f"pred_dt_flow_gate_{suffix}_layers.tif"
		with tifffile.TiffWriter(str(layers_path)) as tw:
			write_named_layer(tw, pred_raw_u8, name="pred_dt")
			write_named_layer(tw, flow, name="raw_flow_bilinear")
			if grid_flow is not None:
				write_named_layer(tw, grid_flow, name="smooth_grid_flow")
			if gate_basis is not None:
				write_named_layer(tw, gate_basis, name="gate_basis_flow_normalized")
			if graph_flow is not None:
				write_named_layer(tw, graph_flow, name="graph_edge_flow")
			if island_obstacle is not None:
				write_named_layer(tw, island_obstacle, name="island_obstacle_factor")
			if island_removed is not None:
				write_named_layer(tw, island_removed, name="island_removed_mask")
			if island_passability is not None:
				write_named_layer(tw, island_passability, name="island_flow_passability")
			if island_propagated is not None:
				write_named_layer(tw, island_propagated, name="island_propagated_edge_flow")
			if island_bonus is not None:
				write_named_layer(tw, island_bonus, name="island_bonus_edge_flow")
			if island_dense_no_backtrack is not None:
				write_named_layer(
					tw,
					island_dense_no_backtrack,
					name="island_tree_dense_flow_no_backtrack",
				)
			if island_dense_greedy is not None:
				write_named_layer(
					tw,
					island_dense_greedy,
					name="island_tree_dense_flow_greedy_ascent",
				)
			if source_edges is not None:
				write_named_layer(tw, source_edges, name="graph_source_edges")
			if source_components is not None:
				write_named_layer(tw, source_components, name="source_components")
			write_named_layer(tw, weights, name="flow_gate_weight")
			if pull_weight is not None:
				write_named_layer(tw, pull_weight, name="anticipatory_pull_weight")
			if pull_prefix is not None:
				write_named_layer(tw, pull_prefix, name="anticipatory_pull_prefix")
			if pull_root_weight is not None:
				write_named_layer(tw, pull_root_weight, name="anticipatory_pull_root_weight")
			if pull_overlay is not None:
				write_named_layer(tw, pull_overlay, name="anticipatory_pull_overlay")
		tifffile.imwrite(str(out_dir / f"pred_dt_flow_gate_{suffix}_pred_dt.tif"), pred_raw_u8)
		tifffile.imwrite(str(out_dir / f"pred_dt_flow_gate_{suffix}_raw_flow.tif"), flow)
		if grid_flow is not None:
			tifffile.imwrite(str(out_dir / f"pred_dt_flow_gate_{suffix}_smooth_grid_flow.tif"), grid_flow)
		if gate_basis is not None:
			tifffile.imwrite(
				str(out_dir / f"pred_dt_flow_gate_{suffix}_gate_basis_flow_normalized.tif"),
				gate_basis,
			)
		if graph_flow is not None:
			tifffile.imwrite(str(out_dir / f"pred_dt_flow_gate_{suffix}_graph_edge_flow.tif"), graph_flow)
		if island_obstacle is not None:
			tifffile.imwrite(
				str(out_dir / f"pred_dt_flow_gate_{suffix}_island_obstacle_factor.tif"),
				island_obstacle,
			)
		if island_removed is not None:
			tifffile.imwrite(
				str(out_dir / f"pred_dt_flow_gate_{suffix}_island_removed_mask.tif"),
				island_removed,
			)
		if island_passability is not None:
			tifffile.imwrite(
				str(out_dir / f"pred_dt_flow_gate_{suffix}_island_flow_passability.tif"),
				island_passability,
			)
		if island_propagated is not None:
			tifffile.imwrite(
				str(out_dir / f"pred_dt_flow_gate_{suffix}_island_propagated_edge_flow.tif"),
				island_propagated,
			)
		if island_bonus is not None:
			tifffile.imwrite(
				str(out_dir / f"pred_dt_flow_gate_{suffix}_island_bonus_edge_flow.tif"),
				island_bonus,
			)
		if island_dense_no_backtrack is not None:
			tifffile.imwrite(
				str(out_dir / f"pred_dt_flow_gate_{suffix}_island_tree_dense_flow_no_backtrack.tif"),
				island_dense_no_backtrack,
			)
		if island_dense_greedy is not None:
			tifffile.imwrite(
				str(out_dir / f"pred_dt_flow_gate_{suffix}_island_tree_dense_flow_greedy_ascent.tif"),
				island_dense_greedy,
			)
		if source_edges is not None:
			tifffile.imwrite(
				str(out_dir / f"pred_dt_flow_gate_{suffix}_graph_source_edges.tif"),
				source_edges,
			)
		if source_components is not None:
			tifffile.imwrite(
				str(out_dir / f"pred_dt_flow_gate_{suffix}_source_components.tif"),
				source_components,
			)
		tifffile.imwrite(str(out_dir / f"pred_dt_flow_gate_{suffix}_weight.tif"), weights)
		if pull_weight is not None:
			tifffile.imwrite(str(out_dir / f"pred_dt_flow_gate_{suffix}_anticipatory_pull_weight.tif"), pull_weight)
		if pull_prefix is not None:
			tifffile.imwrite(str(out_dir / f"pred_dt_flow_gate_{suffix}_anticipatory_pull_prefix.tif"), pull_prefix)
		if pull_root_weight is not None:
			tifffile.imwrite(str(out_dir / f"pred_dt_flow_gate_{suffix}_anticipatory_pull_root_weight.tif"), pull_root_weight)
		if pull_overlay is not None:
			tifffile.imwrite(str(out_dir / f"pred_dt_flow_gate_{suffix}_anticipatory_pull_overlay.tif"), pull_overlay)


def _write_flow_gate_weight_jpg(
	*,
	stage_name: str,
	debug_index: int,
	pred_u8: np.ndarray,
	gate_weight_hr: np.ndarray,
	greedy_direct_flow_hr: np.ndarray | None,
	source_edge_mask_hr: np.ndarray | None,
	source_component_mask_hr: np.ndarray | None,
	source_xy: tuple[int, int] | None,
	extra_source_xy: np.ndarray | None = None,
	corr_seed_debug: dict[str, np.ndarray] | None,
	flow_metadata: dict | None,
	out_dir: Path | None,
) -> None:
	global _flow_gate_jpg_warned
	if out_dir is None:
		return
	pred = np.asarray(pred_u8)
	gate_weight = np.asarray(gate_weight_hr, dtype=np.float32)
	if pred.ndim != 2 or gate_weight.ndim != 2:
		return
	if pred.shape != gate_weight.shape:
		return
	gate_vis = gate_weight.clip(0.0, 1.0)
	greedy_direct_vis = _normalize_positive_debug_image(greedy_direct_flow_hr)
	if greedy_direct_vis is None or greedy_direct_vis.shape != pred.shape:
		greedy_direct_vis = np.zeros_like(gate_weight, dtype=np.float32)
	threshold_basis_vis = (pred.astype(np.uint8) >= 110).astype(np.uint8) * 255
	source_component_vis = None
	if source_component_mask_hr is not None:
		source_component = np.asarray(source_component_mask_hr, dtype=np.float32)
		if source_component.shape == pred.shape:
			source_component_vis = (source_component > 0.0).astype(np.uint8) * 255
	pred_vis = ((pred.astype(np.float32) - 80.0) / (127.0 - 80.0)).clip(0.0, 1.0)
	right_panel = _flow_seed_overlay_panel(
		(greedy_direct_vis * 255.0 + 0.5).astype(np.uint8),
		source_xy=source_xy,
		extra_source_xy=extra_source_xy,
		corr_seed_debug=corr_seed_debug,
		source_edge_mask=source_edge_mask_hr,
		flow_metadata=flow_metadata,
	)
	def gray_to_rgb(gray: np.ndarray) -> np.ndarray:
		return np.repeat(np.asarray(gray, dtype=np.uint8)[..., None], 3, axis=2)
	panels = [
		gray_to_rgb(threshold_basis_vis),
	]
	if source_component_vis is not None:
		panels.append(gray_to_rgb(source_component_vis))
	panels.extend([
		gray_to_rgb((pred_vis * 255.0 + 0.5).astype(np.uint8)),
		gray_to_rgb((gate_vis * 255.0 + 0.5).astype(np.uint8)),
		right_panel,
	])
	img = np.concatenate(panels, axis=1)
	jpg_dir = out_dir / "pred_dt_flow_gate_weight_jpg"
	try:
		jpg_dir.mkdir(parents=True, exist_ok=True)
		if str(stage_name) == str(_flow_gate_stage):
			path = jpg_dir / f"vis_{debug_index:06d}.jpg"
		else:
			safe_stage = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(stage_name))
			path = jpg_dir / f"vis_{safe_stage}_{debug_index:06d}.jpg"
		try:
			import cv2
			if not cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR)):
				raise RuntimeError("cv2.imwrite returned false")
		except Exception:
			from PIL import Image
			Image.fromarray(img, mode="RGB").save(str(path), quality=95)
	except Exception as exc:
		if not _flow_gate_jpg_warned:
			print(f"[pred_dt_flow_gate] jpg weight write skipped: {exc}", flush=True)
			_flow_gate_jpg_warned = True


def _flow_gate_weight(res: fit_model.FitResult3D) -> torch.Tensor | tuple[torch.Tensor, dict | None] | None:
	global _flow_gate_last_stats, _flow_gate_last_timing, _flow_gate_last_channels, _flow_gate_seed_hw_cache
	_flow_gate_last_stats = {}
	_flow_gate_last_timing = {}
	_flow_gate_last_channels = None
	cfg = _flow_gate_cfg
	if cfg is None:
		return None
	if res.data_s.pred_dt is None:
		raise RuntimeError("pred_dt_flow_gate requires pred_dt to be loaded")

	gate_factor = float(cfg.get("gate_factor", 1.0))
	backtrack_distance = float(cfg.get("backtrack_distance", 10.0))
	local_boost = float(cfg.get("local_boost", 1.0))
	pred_dt_pool_radius = int(cfg.get("pred_dt_pool_radius", 0))
	pred_dt_pool_step_scale = float(cfg.get("pred_dt_pool_step_scale", 0.5))
	capture_tifxyz_channels = bool(cfg.get("_capture_tifxyz_channels", False))
	pull_cfg = _anticipatory_pull_cfg(cfg)
	if not 0.0 <= gate_factor <= 1.0:
		raise ValueError("pred_dt_flow_gate requires gate_factor in [0, 1]")
	if not 0.0 <= local_boost <= 1.0:
		raise ValueError("pred_dt_flow_gate requires local_boost in [0, 1]")
	debug = bool(cfg.get("debug", True))
	debug_index = 0
	if debug:
		debug_index = _flow_gate_debug_counts.get(_flow_gate_stage, 0)
		_flow_gate_debug_counts[_flow_gate_stage] = debug_index + 1
	debug_layer_interval = _debug_interval(cfg, "debug_layer_interval")
	debug_jpg_interval = _debug_interval(cfg, "debug_vis_interval", "debug_jpg_interval")
	write_layer_debug = (
		debug and debug_layer_interval > 0 and (debug_index % debug_layer_interval) == 0
	)
	write_jpg_debug = (
		debug and debug_jpg_interval > 0 and (debug_index % debug_jpg_interval) == 0
	)
	write_atlas_snap_seed_report = (
		debug
		and bool(cfg.get("atlas_snap_seed_report", True))
		and (write_layer_debug or write_jpg_debug or bool(cfg.get("atlas_snap_seed_report_every_step", False)))
	)
	return_flow_debug = write_layer_debug or write_jpg_debug
	timing: dict[str, float] = {}
	def mark(label: str) -> float:
		return time.perf_counter()

	def done(label: str, t0: float) -> None:
		timing[label] = timing.get(label, 0.0) + (time.perf_counter() - t0)

	def publish_timing() -> None:
		global _flow_gate_last_timing
		_flow_gate_last_timing = dict(timing)

	with torch.no_grad():
		D = int(res.xyz_lr.shape[0])
		Hm = int(res.xyz_lr.shape[1])
		Wm = int(res.xyz_lr.shape[2])
		depth_windings_raw = getattr(res.params, "depth_windings", None)
		if isinstance(depth_windings_raw, (list, tuple)) and len(depth_windings_raw) == D:
			depth_windings = tuple(depth_windings_raw)
		else:
			depth_windings = tuple(range(D))
		sub_h = int(res.params.subsample_mesh)
		sub_w = int(res.params.subsample_winding)
		grid_step = max(1, int(round(0.5 * (sub_h + sub_w))))
		yy, xx = np.meshgrid(
			np.arange(Hm, dtype=np.float32) * float(sub_h),
			np.arange(Wm, dtype=np.float32) * float(sub_w),
			indexing="ij",
		)
		query_xy = np.stack([xx, yy], axis=-1).reshape(-1, 2)
		grid_y, grid_x = torch.meshgrid(
			torch.arange(Hm, device=res.xyz_lr.device, dtype=torch.float32),
			torch.arange(Wm, device=res.xyz_lr.device, dtype=torch.float32),
			indexing="ij",
		)
		weight = torch.full(
			(D, 1, Hm, Wm),
			float(1.0 - gate_factor),
			device=res.xyz_lr.device,
			dtype=torch.float32,
		)
		gate_weight = torch.zeros_like(weight)
		component_local = torch.zeros((D, Hm, Wm), device=res.xyz_lr.device, dtype=torch.float32)
		component_normalized = torch.zeros((D, Hm, Wm), device=res.xyz_lr.device, dtype=torch.float32)
		stats: dict[str, float] = {
			"pred_dt_flow_layers": float(D),
			"pred_dt_flow_layers_with_seed": 0.0,
			"pred_dt_flow_layers_with_corr": 0.0,
			"pred_dt_flow_layers_with_atlas_snap": 0.0,
			"pred_dt_flow_layers_with_flow": 0.0,
			"pred_dt_flow_layers_no_sources": 0.0,
			"pred_dt_corr_seed_cxx_accepted": 0.0,
			"pred_dt_flow_source_edges": 0.0,
			"pred_dt_flow_seeded_nodes": 0.0,
		}

		def layer_stage_name(d: int, winding: float | int) -> str:
			if D == 1:
				return _flow_gate_stage
			try:
				w_text = str(int(winding)) if float(winding).is_integer() else f"{float(winding):g}"
			except Exception:
				w_text = str(winding)
			w_text = w_text.replace("-", "m").replace(".", "p")
			return f"{_flow_gate_stage}_layer{d:03d}_w{w_text}"

		def unpack_flow_outputs(flow_outputs):
			if return_flow_debug:
				if capture_tifxyz_channels:
					(
						query_flow,
						dense_flow,
						smooth_grid_flow,
						gate_basis_flow,
						graph_edge_flow_rgb,
						island_obstacle_factor_rgb,
						island_removed_mask_hr,
						island_flow_passability_rgb,
						island_propagated_edge_flow_rgb,
						island_bonus_edge_flow_rgb,
						island_tree_dense_no_backtrack_hr,
						island_tree_dense_greedy_ascent_hr,
						source_edge_mask_hr,
						source_component_mask_hr,
						flow_components,
						flow_metadata,
					) = flow_outputs
				else:
					(
						query_flow,
						dense_flow,
						smooth_grid_flow,
						gate_basis_flow,
						graph_edge_flow_rgb,
						island_obstacle_factor_rgb,
						island_removed_mask_hr,
						island_flow_passability_rgb,
						island_propagated_edge_flow_rgb,
						island_bonus_edge_flow_rgb,
						island_tree_dense_no_backtrack_hr,
						island_tree_dense_greedy_ascent_hr,
						source_edge_mask_hr,
						source_component_mask_hr,
						flow_metadata,
					) = flow_outputs
					flow_components = None
			else:
				if capture_tifxyz_channels:
					query_flow, dense_flow, flow_components, flow_metadata = flow_outputs
				else:
					query_flow, dense_flow, flow_metadata = flow_outputs
					flow_components = None
				smooth_grid_flow = None
				gate_basis_flow = None
				graph_edge_flow_rgb = None
				island_obstacle_factor_rgb = None
				island_removed_mask_hr = None
				island_flow_passability_rgb = None
				island_propagated_edge_flow_rgb = None
				island_bonus_edge_flow_rgb = None
				island_tree_dense_no_backtrack_hr = None
				island_tree_dense_greedy_ascent_hr = None
				source_edge_mask_hr = None
				source_component_mask_hr = None
			return {
				"query_flow": query_flow,
				"dense_flow": dense_flow,
				"smooth_grid_flow": smooth_grid_flow,
				"gate_basis_flow": gate_basis_flow,
				"graph_edge_flow_rgb": graph_edge_flow_rgb,
				"island_obstacle_factor_rgb": island_obstacle_factor_rgb,
				"island_removed_mask_hr": island_removed_mask_hr,
				"island_flow_passability_rgb": island_flow_passability_rgb,
				"island_propagated_edge_flow_rgb": island_propagated_edge_flow_rgb,
				"island_bonus_edge_flow_rgb": island_bonus_edge_flow_rgb,
				"island_tree_dense_no_backtrack_hr": island_tree_dense_no_backtrack_hr,
				"island_tree_dense_greedy_ascent_hr": island_tree_dense_greedy_ascent_hr,
				"source_edge_mask_hr": source_edge_mask_hr,
				"source_component_mask_hr": source_component_mask_hr,
				"flow_components": flow_components,
				"flow_metadata": flow_metadata,
			}

		last_pull_debug: tuple[dict | None, dict | None, torch.Tensor, str, int] | None = None
		for d, winding in enumerate(depth_windings):
			xyz_hr = res.xyz_hr[d].detach()
			stage_name = layer_stage_name(d, winding)
			stats[f"pred_dt_layer_{d}_winding"] = float(winding)

			_t = mark("flow_sampling")
			pred_hr = _sample_pred_dt_max3d(
				res=res,
				xyz_hr=xyz_hr,
				layer_index=d,
				radius=pred_dt_pool_radius,
				step_scale=pred_dt_pool_step_scale,
			)
			pred_img = pred_hr.detach().clamp(0, 255).round().to(torch.uint8).cpu().numpy()
			done("flow_sampling", _t)
			He, We = pred_img.shape

			_t = mark("flow_sampling")
			source_x: int | None = None
			source_y: int | None = None
			seed_area = None
			seed_present = _flow_gate_seed_xyz is not None and float(winding) == 0.0
			if seed_present:
				seed = torch.tensor(_flow_gate_seed_xyz, device=xyz_hr.device, dtype=xyz_hr.dtype)
				n_gt = _seed_gt_normal(seed_xyz=seed, res=res)
				source_xy_f = None
				if _flow_gate_seed_hw_cache is not None:
					cache_h, cache_w, cache_y, cache_x = _flow_gate_seed_hw_cache
					if cache_h == He and cache_w == We:
						source_xy_f = _seed_surface_intersection_xy_from_cache(
							xyz_img=xyz_hr,
							seed_xyz=seed,
							n_gt=n_gt,
							h_frac=cache_y,
							w_frac=cache_x,
						)
				if source_xy_f is None:
					source_xy_f = _seed_surface_intersection_xy(
						xyz_img=xyz_hr,
						seed_xyz=seed,
						n_gt=n_gt,
					)
				if source_xy_f is None:
					dist2 = ((xyz_hr - seed.view(1, 1, 3)) ** 2).sum(dim=-1)
					source_flat = int(torch.argmin(dist2).detach().cpu())
					source_y = source_flat // We
					source_x = source_flat % We
					_flow_gate_seed_hw_cache = (He, We, float(source_y), float(source_x))
				else:
					_flow_gate_seed_hw_cache = (He, We, float(source_xy_f[1]), float(source_xy_f[0]))
					source_x = int(round(source_xy_f[0]))
					source_y = int(round(source_xy_f[1]))
					source_x = max(0, min(We - 1, source_x))
					source_y = max(0, min(He - 1, source_y))
				source_grid_y = float(source_y) / float(max(1, sub_h))
				source_grid_x = float(source_x) / float(max(1, sub_w))
				seed_area = (
					(grid_y - source_grid_y).square() + (grid_x - source_grid_x).square()
				).le(4.0).view(1, 1, Hm, Wm)
			done("flow_sampling", _t)

			_t = mark("flow_sampling")
			corr_source_xy, corr_seed_stats, corr_seed_debug = _corr_point_source_xy(
				res=res,
				xyz_img=xyz_hr,
				cfg=cfg,
				sub_h=sub_h,
				sub_w=sub_w,
				layer_index=d,
				winding_value=winding,
			)
			corr_count = int(corr_source_xy.shape[0])
			atlas_source_xy, atlas_seed_stats, atlas_seed_debug = _atlas_snap_source_xy(
				res=res,
				xyz_img=xyz_hr,
				pred_img=pred_img,
				cfg=cfg,
				sub_h=sub_h,
				sub_w=sub_w,
				layer_index=d,
			)
			atlas_count = int(atlas_source_xy.shape[0])
			if seed_present:
				stats["pred_dt_flow_layers_with_seed"] += 1.0
			if corr_count > 0:
				stats["pred_dt_flow_layers_with_corr"] += 1.0
			if atlas_count > 0:
				stats["pred_dt_flow_layers_with_atlas_snap"] += 1.0
			stats[f"pred_dt_layer_{d}_seed_sources"] = 1.0 if seed_present else 0.0
			stats[f"pred_dt_layer_{d}_corr_sources"] = float(corr_count)
			stats[f"pred_dt_layer_{d}_atlas_snap_sources"] = float(atlas_count)
			for key, value in corr_seed_stats.items():
				value_f = float(value)
				stats[f"pred_dt_layer_{d}_{key}"] = float(value)
				if key.endswith(("candidates", "valid", "unique")):
					stats[key] = stats.get(key, 0.0) + value_f
				else:
					stats[key] = max(stats.get(key, value_f), value_f)
			for key, value in atlas_seed_stats.items():
				value_f = float(value)
				stats[f"pred_dt_layer_{d}_{key}"] = value_f
				if key.endswith(("candidates", "accepted", "unique")):
					stats[key] = stats.get(key, 0.0) + value_f
				else:
					stats[key] = max(stats.get(key, value_f), value_f)

			primary_xy: tuple[int, int] | None = None
			explicit_source_xy = _unique_xy_rows(corr_source_xy, atlas_source_xy)
			seed_source_xy = (
				np.asarray([[int(source_x), int(source_y)]], dtype=np.int32)
				if source_x is not None and source_y is not None
				else np.zeros((0, 2), dtype=np.int32)
			)
			all_source_xy = _unique_xy_rows(explicit_source_xy, seed_source_xy)
			if explicit_source_xy.shape[0] > 0:
				primary_xy = (int(explicit_source_xy[0, 0]), int(explicit_source_xy[0, 1]))
				extra_source_xy = all_source_xy[1:]
			elif seed_source_xy.shape[0] > 0:
				primary_xy = (int(seed_source_xy[0, 0]), int(seed_source_xy[0, 1]))
				extra_source_xy = np.zeros((0, 2), dtype=np.int32)
			else:
				extra_source_xy = np.zeros((0, 2), dtype=np.int32)
			done("flow_sampling", _t)

			if primary_xy is None:
				stats["pred_dt_flow_layers_no_sources"] += 1.0
				stats[f"pred_dt_layer_{d}_flow_called"] = 0.0
				if write_atlas_snap_seed_report:
					_write_atlas_snap_seed_report(
						stage_name=stage_name,
						debug_index=debug_index,
						layer_index=d,
						winding=int(winding),
						atlas_seed_debug=atlas_seed_debug,
						primary_xy=primary_xy,
						extra_source_xy=extra_source_xy,
						flow_status="no_sources",
						source_component_mask_hr=None,
						out_dir=_flow_gate_out_dir,
					)
				if write_layer_debug:
					_t = mark("write_layer_debug")
					weight_hr = F.interpolate(
						weight[d:d + 1],
						size=(He, We),
						mode="bilinear",
						align_corners=True,
					)[0, 0].detach().cpu().numpy().astype(np.float32)
					_write_flow_gate_debug(
						stage_name=stage_name,
						debug_index=debug_index,
						pred_u8=pred_img,
						flow_hr=None,
						smooth_grid_flow=None,
						gate_basis_hr=None,
						graph_edge_flow_rgb=None,
						island_obstacle_factor_rgb=None,
						island_removed_mask_hr=None,
						island_flow_passability_rgb=None,
						island_propagated_edge_flow_rgb=None,
						island_bonus_edge_flow_rgb=None,
						island_tree_dense_no_backtrack_hr=None,
						island_tree_dense_greedy_ascent_hr=None,
						source_edge_mask_hr=None,
						source_component_mask_hr=None,
						weight_hr=weight_hr,
						out_dir=_flow_gate_out_dir,
					)
					done("write_layer_debug", _t)
				continue

			stats[f"pred_dt_layer_{d}_flow_called"] = 1.0
			def _compute_flow_outputs_for_layer():
				_t_flow = mark("flow_calc")
				try:
					return dense_batch_flow.compute_flow_grid(
						pred_img,
						source_xy=primary_xy,
						extra_source_xy=extra_source_xy,
						query_xy=query_xy,
						verbose=False,
						return_debug=return_flow_debug,
						return_metadata=True,
						return_components=capture_tifxyz_channels,
						grid_step=grid_step,
						backtrack_distance=backtrack_distance,
						local_boost=local_boost,
					)
				finally:
					done("flow_calc", _t_flow)

			try:
				flow = unpack_flow_outputs(_compute_flow_outputs_for_layer())
				stats["pred_dt_flow_layers_with_flow"] += 1.0
				if write_atlas_snap_seed_report:
					_write_atlas_snap_seed_report(
						stage_name=stage_name,
						debug_index=debug_index,
						layer_index=d,
						winding=int(winding),
						atlas_seed_debug=atlas_seed_debug,
						primary_xy=primary_xy,
						extra_source_xy=extra_source_xy,
						flow_status="flow_called",
						source_component_mask_hr=flow["source_component_mask_hr"],
						out_dir=_flow_gate_out_dir,
					)
			except RuntimeError as exc:
				message = str(exc)
				source_value = int(pred_img[primary_xy[1], primary_xy[0]]) if 0 <= primary_xy[1] < He and 0 <= primary_xy[0] < We else -1
				if "white distance domain" not in message:
					raise
				if seed_area is not None:
					gate_weight[d:d + 1] = seed_area.to(device=gate_weight.device, dtype=torch.float32)
					weight[d:d + 1] = gate_factor * gate_weight[d:d + 1] + (1.0 - gate_factor)
				print(
					f"[pred_dt_flow_gate] {stage_name}: skipped flow "
					f"(source outside C++ flow domain, value={source_value})",
					flush=True,
				)
				if write_atlas_snap_seed_report:
					_write_atlas_snap_seed_report(
						stage_name=stage_name,
						debug_index=debug_index,
						layer_index=d,
						winding=int(winding),
						atlas_seed_debug=atlas_seed_debug,
						primary_xy=primary_xy,
						extra_source_xy=extra_source_xy,
						flow_status="flow_skipped_source_outside_domain",
						flow_source_value=source_value,
						source_component_mask_hr=None,
						out_dir=_flow_gate_out_dir,
					)
				if write_layer_debug:
					_t = mark("write_layer_debug")
					weight_hr = F.interpolate(
						weight[d:d + 1],
						size=(He, We),
						mode="bilinear",
						align_corners=True,
					)[0, 0].detach().cpu().numpy().astype(np.float32)
					_write_flow_gate_debug(
						stage_name=stage_name,
						debug_index=debug_index,
						pred_u8=pred_img,
						flow_hr=None,
						smooth_grid_flow=None,
						gate_basis_hr=None,
						graph_edge_flow_rgb=None,
						island_obstacle_factor_rgb=None,
						island_removed_mask_hr=None,
						island_flow_passability_rgb=None,
						island_propagated_edge_flow_rgb=None,
						island_bonus_edge_flow_rgb=None,
						island_tree_dense_no_backtrack_hr=None,
						island_tree_dense_greedy_ascent_hr=None,
						source_edge_mask_hr=None,
						source_component_mask_hr=None,
						weight_hr=weight_hr,
						out_dir=_flow_gate_out_dir,
					)
					done("write_layer_debug", _t)
				if write_jpg_debug:
					_t = mark("write_weight_jpg")
					gate_weight_hr = F.interpolate(
						gate_weight[d:d + 1],
						size=(He, We),
						mode="nearest",
					)[0, 0].detach().cpu().numpy().astype(np.float32)
					_write_flow_gate_weight_jpg(
						stage_name=stage_name,
						debug_index=debug_index,
						pred_u8=pred_img,
						gate_weight_hr=gate_weight_hr,
						greedy_direct_flow_hr=None,
						source_edge_mask_hr=None,
						source_component_mask_hr=None,
						source_xy=primary_xy,
						extra_source_xy=extra_source_xy,
						corr_seed_debug=corr_seed_debug,
						flow_metadata=None,
						out_dir=_flow_gate_out_dir,
					)
					done("write_weight_jpg", _t)
				continue

			_t = mark("compute_weight")
			flow_lr = torch.as_tensor(
				flow["query_flow"].reshape(Hm, Wm),
				device=res.xyz_lr.device,
				dtype=torch.float32,
			).view(1, 1, Hm, Wm)
			gate_weight[d:d + 1] = flow_lr.clamp(0.0, 1.0)
			weight[d:d + 1] = gate_factor * gate_weight[d:d + 1] + (1.0 - gate_factor)
			flow_components = flow["flow_components"]
			if capture_tifxyz_channels and isinstance(flow_components, dict):
				local_arr = flow_components.get("flow_gate_local_contrast")
				normalized_arr = flow_components.get("flow_gate_component_normalized")
				if local_arr is not None and normalized_arr is not None:
					component_local[d] = torch.as_tensor(
						np.asarray(local_arr, dtype=np.float32).reshape(Hm, Wm),
						device=res.xyz_lr.device,
						dtype=torch.float32,
					).clamp(0.0, 1.0)
					component_normalized[d] = torch.as_tensor(
						np.asarray(normalized_arr, dtype=np.float32).reshape(Hm, Wm),
						device=res.xyz_lr.device,
						dtype=torch.float32,
					).clamp(0.0, 1.0)
			flow_metadata = flow["flow_metadata"]
			if isinstance(flow_metadata, dict):
				stats["pred_dt_corr_seed_cxx_accepted"] += float(flow_metadata.get("accepted_source_count", 0))
				stats["pred_dt_flow_source_edges"] += float(flow_metadata.get("source_edge_count", 0))
				stats["pred_dt_flow_seeded_nodes"] += float(flow_metadata.get("seeded_node_count", 0))
				stats[f"pred_dt_layer_{d}_cxx_accepted_sources"] = float(flow_metadata.get("accepted_source_count", 0))
				stats[f"pred_dt_layer_{d}_flow_source_edges"] = float(flow_metadata.get("source_edge_count", 0))
				stats[f"pred_dt_layer_{d}_flow_seeded_nodes"] = float(flow_metadata.get("seeded_node_count", 0))
			done("compute_weight", _t)

			gate_basis_hr = None
			if flow["gate_basis_flow"] is not None and (write_layer_debug or write_jpg_debug):
				gate_basis_hr = np.asarray(flow["gate_basis_flow"], dtype=np.float32)
			if write_layer_debug:
				_t = mark("write_layer_debug")
				flow_hr = F.interpolate(
					gate_weight[d:d + 1],
					size=(He, We),
					mode="bilinear",
					align_corners=True,
				)[0, 0].detach().cpu().numpy().astype(np.float32)
				weight_hr = F.interpolate(
					weight[d:d + 1],
					size=(He, We),
					mode="bilinear",
					align_corners=True,
				)[0, 0].detach().cpu().numpy().astype(np.float32)
				pull_weight_hr = None
				pull_prefix_hr = None
				pull_root_weight_hr = None
				pull_overlay_rgb = None
				if D == 1 and pull_cfg is not None:
					pull_candidates_debug = _score_anticipatory_pull_candidates(
						res=res,
						cfg=pull_cfg,
						flow_weight=gate_weight[d:d + 1],
						mask_lr=res.mask_lr[d:d + 1],
						layer_index=d,
					)
					pull_debug = _activate_anticipatory_pull(
						candidates=pull_candidates_debug,
						flow_weight=gate_weight[d:d + 1],
						mask_lr=res.mask_lr[d:d + 1],
						cfg=pull_cfg,
						weight_scale=gate_factor,
					)
					pull_weight_lr, pull_prefix_lr, pull_root_weight_lr = _anticipatory_pull_debug_lr(
						pull=pull_debug,
						Hm=Hm,
						Wm=Wm,
						device=res.xyz_lr.device,
					)
					pull_weight_hr = pull_weight_lr.detach().cpu().numpy().astype(np.float32)
					pull_prefix_hr = pull_prefix_lr.detach().cpu().numpy().astype(np.float32)
					pull_root_weight_hr = pull_root_weight_lr.detach().cpu().numpy().astype(np.float32)
					pull_overlay_rgb = _anticipatory_pull_overlay(
						pull=pull_debug,
						height=He,
						width=We,
						sub_h=sub_h,
						sub_w=sub_w,
					)
				_write_flow_gate_debug(
					stage_name=stage_name,
					debug_index=debug_index,
					pred_u8=pred_img,
					flow_hr=flow_hr,
					smooth_grid_flow=flow["smooth_grid_flow"],
					gate_basis_hr=gate_basis_hr,
					graph_edge_flow_rgb=flow["graph_edge_flow_rgb"],
					island_obstacle_factor_rgb=flow["island_obstacle_factor_rgb"],
					island_removed_mask_hr=flow["island_removed_mask_hr"],
					island_flow_passability_rgb=flow["island_flow_passability_rgb"],
					island_propagated_edge_flow_rgb=flow["island_propagated_edge_flow_rgb"],
					island_bonus_edge_flow_rgb=flow["island_bonus_edge_flow_rgb"],
					island_tree_dense_no_backtrack_hr=flow["island_tree_dense_no_backtrack_hr"],
					island_tree_dense_greedy_ascent_hr=flow["island_tree_dense_greedy_ascent_hr"],
					source_edge_mask_hr=flow["source_edge_mask_hr"],
					source_component_mask_hr=flow["source_component_mask_hr"],
					weight_hr=weight_hr,
					pull_weight_hr=pull_weight_hr,
					pull_prefix_hr=pull_prefix_hr,
					pull_root_weight_hr=pull_root_weight_hr,
					pull_overlay_rgb=pull_overlay_rgb,
					out_dir=_flow_gate_out_dir,
				)
				done("write_layer_debug", _t)
			if write_jpg_debug:
				_t = mark("write_weight_jpg")
				gate_weight_hr = np.asarray(flow["dense_flow"], dtype=np.float32)
				if gate_weight_hr.shape != pred_img.shape:
					gate_weight_hr = F.interpolate(
						gate_weight[d:d + 1],
						size=(He, We),
						mode="nearest",
					)[0, 0].detach().cpu().numpy().astype(np.float32)
				_write_flow_gate_weight_jpg(
					stage_name=stage_name,
					debug_index=debug_index,
					pred_u8=pred_img,
					gate_weight_hr=gate_weight_hr,
					greedy_direct_flow_hr=gate_basis_hr,
					source_edge_mask_hr=flow["source_edge_mask_hr"],
					source_component_mask_hr=flow["source_component_mask_hr"],
					source_xy=primary_xy,
					extra_source_xy=extra_source_xy,
					corr_seed_debug=corr_seed_debug,
					flow_metadata=flow_metadata,
					out_dir=_flow_gate_out_dir,
				)
				done("write_weight_jpg", _t)

		_t = mark("compute_weight")
		valid = res.mask_lr > 0.0
		valid_count = max(1.0, float(valid.sum().detach().cpu()))
		stats.update({
			"pred_dt_gate_gt0": float(((gate_weight > 0.0) & valid).sum().detach().cpu()) / valid_count,
			"pred_dt_gate_gt01": float(((gate_weight > 0.1) & valid).sum().detach().cpu()) / valid_count,
			"pred_dt_gate_gt05": float(((gate_weight > 0.5) & valid).sum().detach().cpu()) / valid_count,
			"pred_dt_gate_eq1": float(((gate_weight >= 1.0) & valid).sum().detach().cpu()) / valid_count,
			"pred_dt_gate_n_gt0": float(((gate_weight > 0.0) & valid).sum().detach().cpu()) / valid_count,
			"pred_dt_gate_n_gt01": float(((gate_weight > 0.1) & valid).sum().detach().cpu()) / valid_count,
			"pred_dt_gate_n_gt05": float(((gate_weight > 0.5) & valid).sum().detach().cpu()) / valid_count,
		})
		done("compute_weight", _t)

		if capture_tifxyz_channels:
			_flow_gate_last_channels = {
				"version": 1,
				"stage_name": str(_flow_gate_stage),
				"mesh_shape_dhw": [int(D), int(Hm), int(Wm)],
				"flow_gate_local_contrast": component_local.detach().cpu(),
				"flow_gate_component_normalized": component_normalized.detach().cpu(),
				"source_config": {
					"local_boost": float(local_boost),
					"backtrack_distance": float(backtrack_distance),
					"gate_factor": float(gate_factor),
					"grid_step": int(grid_step),
				},
			}

		pull = None
		if pull_cfg is not None:
			_t_pull = mark("anticipatory_pull")
			pull_parts: list[dict] = []
			total_candidates = 0
			gate_candidates = 0
			scored_candidates = 0
			batches = 0
			chunk_candidates = 0
			query_samples = 0
			sparse_prefetch_batches = 0
			for d in range(D):
				candidates_d = _score_anticipatory_pull_candidates(
					res=res,
					cfg=pull_cfg,
					flow_weight=gate_weight[d:d + 1],
					mask_lr=res.mask_lr[d:d + 1],
					layer_index=d,
				)
				pull_d = _activate_anticipatory_pull(
					candidates=candidates_d,
					flow_weight=gate_weight[d:d + 1],
					mask_lr=res.mask_lr[d:d + 1],
					cfg=pull_cfg,
					weight_scale=gate_factor,
				)
				if pull_d is not None:
					pull_parts.append(pull_d)
				pull_stats = candidates_d.get("_stats", {}) if candidates_d is not None else {}
				total_candidates += int(pull_stats.get("total_candidates", 0.0))
				gate_candidates += int(pull_stats.get("gate_candidates", 0.0))
				scored_candidates += int(pull_stats.get("scored_candidates", 0.0))
				batches += int(pull_stats.get("active_batches", 0.0))
				chunk_candidates = max(chunk_candidates, int(pull_stats.get("chunk_candidates", 0.0)))
				query_samples += int(pull_stats.get("query_samples", 0.0))
				sparse_prefetch_batches += int(pull_stats.get("sparse_prefetch_batches", 0.0))
				if D == 1:
					last_pull_debug = (candidates_d, pull_d, gate_weight[d:d + 1], _flow_gate_stage, d)
			if pull_parts:
				first = pull_parts[0]
				pull = {}
				for key, value in first.items():
					if isinstance(value, torch.Tensor):
						pull[key] = torch.cat([p[key] for p in pull_parts], dim=0)
			active_count = 0 if pull is None else int(pull["tip_h"].numel())
			den_total = float(max(1, total_candidates))
			stats.update({
				"pred_dt_pull_gate_frac": float(gate_candidates) / den_total,
				"pred_dt_pull_scored_frac": float(scored_candidates) / den_total,
				"pred_dt_pull_active_frac": float(active_count) / den_total,
				"pred_dt_pull_batches": float(batches),
				"pred_dt_pull_samples_m": float(query_samples) / 1.0e6,
				"pred_dt_pull_weight_mean": (
					float(pull["candidate_weight"].mean().detach().cpu()) if active_count > 0 else 0.0
				),
				"pred_dt_pull_prefix_mean": (
					float(pull["prefix"].mean().detach().cpu()) if active_count > 0 else 0.0
				),
			})
			done("anticipatory_pull", _t_pull)
			if bool(pull_cfg.get("print_stats", False)):
				print(
					f"[pred_dt_flow_gate] {_flow_gate_stage}: anticipatory pull "
					f"total={total_candidates} gate={gate_candidates} scored={scored_candidates} "
					f"active={active_count} batches={batches} chunk={chunk_candidates} "
					f"samples={query_samples} sparse_prefetch_batches={sparse_prefetch_batches}",
					flush=True,
				)
			if write_layer_debug and last_pull_debug is not None:
				candidates_dbg, pull_dbg, flow_weight_dbg, stage_name_dbg, _d_dbg = last_pull_debug
				_write_anticipatory_fit_debug_mosaic(
					res=res,
					cfg=pull_cfg,
					candidates=candidates_dbg,
					pull=pull_dbg,
					flow_weight=flow_weight_dbg,
					stage_name=stage_name_dbg,
					debug_index=debug_index,
					out_dir=_flow_gate_out_dir,
				)
		else:
			stats.update({
				"pred_dt_pull_gate_frac": 0.0,
				"pred_dt_pull_scored_frac": 0.0,
				"pred_dt_pull_active_frac": 0.0,
				"pred_dt_pull_batches": 0.0,
				"pred_dt_pull_samples_m": 0.0,
				"pred_dt_pull_weight_mean": 0.0,
				"pred_dt_pull_prefix_mean": 0.0,
			})

		_flow_gate_last_stats = stats
		publish_timing()
		return weight, pull


def pred_dt_loss(*, res: fit_model.FitResult3D) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
	"""Pred-DT loss: clamped outside L2 plus inside L1 pushing mesh into prediction.

	Encoding: outside=[80,127], inside=[128,175], boundary at 127.5.
	lm_out = clamp(127 - raw, min=0)^2    — active outside, zero inside
	lm_in  = clamp(255 - raw, max=127)    — active inside, constant (no grad) outside
	lm = lm_out + 0.25 * lm_in
	"""
	# Project gradients onto surface normal only (prevents tangential crimping)
	xyz = _pred_dt_loss_sample_xyz(res)

	# Sample pred_dt using common sampling with per-channel spacing and diff gradients
	sampled = res.data.grid_sample_fullres(xyz, diff=True, channels={"pred_dt"})
	sampled_raw = sampled.pred_dt.squeeze(0).permute(1, 0, 2, 3)  # (D, 1, Hm, Wm)

	lm_out_l1 = (127.0 - sampled_raw).clamp(min=0)   # outside: 1–47, inside: 0 (no grad)
	lm_out = lm_out_l1 * lm_out_l1
	lm_in = (255.0 - sampled_raw).clamp(max=127.0)    # inside: 80–127, outside: 127 (constant, no grad)
	lm = lm_out + _INNER_FACTOR * lm_in

	mask = res.mask_lr
	flow_result = _flow_gate_weight(res)
	pull = None
	if isinstance(flow_result, tuple):
		flow_gate, pull = flow_result
	else:
		flow_gate = flow_result
	wsum = mask.sum()
	if flow_gate is not None:
		pull_lm = _anticipatory_pull_loss_map(res=res, pull=pull)
		loss = (lm * mask * flow_gate + pull_lm).mean()
	else:
		if float(wsum) > 0.0:
			loss = (lm * mask).sum() / wsum
		else:
			loss = lm.mean()
	if flow_gate is not None:
		return loss, (lm, pull_lm), (mask * flow_gate, torch.ones_like(pull_lm))
	return loss, (lm,), (mask,)
