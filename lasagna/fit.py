import argparse
import copy
import dataclasses
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

import cli_data
import cli_json
import cli_model
import cli_opt
import fit_data
import model
import opt_loss_corr
import opt_loss_dir
import opt_loss_pred_dt
import opt_loss_step
import optimizer
import volume_scale


_SHELL_STEP_ANALYSIS_ENABLED = False


def _stage_start(label: str) -> float:
	return 0.0


def _stage_done(label: str, t0: float) -> None:
	return None


def _truthy_config_bool(value: object) -> bool:
	if isinstance(value, bool):
		return value
	if isinstance(value, (int, float)):
		return bool(value)
	if isinstance(value, str):
		return value.strip().lower() in {"1", "true", "yes", "on"}
	return False


@dataclass(frozen=True)
class InitGrowConfig:
	enabled: bool = False
	axis: str = "z"
	initial_depth: int = 1
	target_depth_from: str = "depth"
	order: tuple[str, ...] = ("up",)
	step: int = 1


def _raw_init_grow_initial_depth(args_cfg: dict | None) -> int:
	if not isinstance(args_cfg, dict):
		return 1
	raw = args_cfg.get("init-grow", args_cfg.get("init_grow"))
	if not isinstance(raw, dict):
		return 1
	return max(1, int(raw.get("initial_depth", raw.get("initial-depth", 1))))


def _raw_init_grow_enabled(args_cfg: dict | None) -> bool:
	if not isinstance(args_cfg, dict):
		return False
	raw = args_cfg.get("init-grow", args_cfg.get("init_grow"))
	if raw is None:
		return False
	if not isinstance(raw, dict):
		return True
	return _truthy_config_bool(raw.get("enabled", True))


def _init_grow_stage_depth_delta(cfg: dict) -> int:
	def _grow_value(grow: object, keys: tuple[str, ...]) -> int:
		if not isinstance(grow, dict):
			return 0
		total = 0
		for key in keys:
			if key in grow:
				total += max(0, int(grow[key]))
		return total

	def _walk(stages: object) -> int:
		if not isinstance(stages, list):
			return 0
		total = 0
		for stage in stages:
			if not isinstance(stage, dict):
				continue
			name = str(stage.get("name", ""))
			if name == "expand-z":
				total += _grow_value(
					stage.get("grow"),
					("d_pos", "d-pos", "z_pos", "z-pos", "depth_pos", "depth-pos", "up"),
				)
				total += _grow_value(
					stage.get("grow"),
					("d_neg", "d-neg", "z_neg", "z-neg", "depth_neg", "depth-neg", "down"),
				)
			total += _walk(stage.get("stages"))
		return total

	return _walk(cfg.get("stages"))


def _parse_init_grow_config(args_cfg: dict | None, *, target_depth: int) -> InitGrowConfig:
	if not isinstance(args_cfg, dict):
		return InitGrowConfig(enabled=False)
	raw = args_cfg.get("init-grow", args_cfg.get("init_grow"))
	if raw is None:
		return InitGrowConfig(enabled=False)
	if not isinstance(raw, dict):
		raise ValueError("args.init-grow must be an object")
	enabled = _truthy_config_bool(raw.get("enabled", True))
	axis = str(raw.get("axis", "z")).strip().lower()
	if axis not in {"z", "d", "depth"}:
		raise ValueError("args.init-grow.axis must be 'z'")
	initial_depth = int(raw.get("initial_depth", raw.get("initial-depth", 1)))
	if initial_depth < 1:
		raise ValueError("args.init-grow.initial_depth must be >= 1")
	if initial_depth > int(target_depth):
		raise ValueError("args.init-grow.initial_depth must be <= args.depth")
	target_depth_from = str(raw.get("target_depth_from", raw.get("target-depth-from", "depth"))).strip().lower()
	if target_depth_from != "depth":
		raise ValueError("args.init-grow.target_depth_from currently only supports 'depth'")
	order_raw = raw.get("order", ["up"])
	if isinstance(order_raw, str):
		order = (order_raw.strip().lower(),)
	elif isinstance(order_raw, list):
		order = tuple(str(v).strip().lower() for v in order_raw)
	else:
		raise ValueError("args.init-grow.order must be a string or list")
	for item in order:
		if item not in {"up", "down"}:
			raise ValueError("args.init-grow.order entries must be 'up' or 'down'")
	step = int(raw.get("step", 1))
	if step < 1:
		raise ValueError("args.init-grow.step must be >= 1")
	return InitGrowConfig(
		enabled=enabled,
		axis="z",
		initial_depth=initial_depth,
		target_depth_from=target_depth_from,
		order=order,
		step=step,
	)


def _require_torch_device_available(device: torch.device) -> None:
	if device.type != "cuda":
		return
	if not torch.cuda.is_available():
		raise RuntimeError(
			"CUDA device was requested, but PyTorch cannot access an NVIDIA GPU. "
			"Expose the NVIDIA driver/device nodes to this process (for example "
			"/dev/nvidia*, Docker --gpus all, or the equivalent sandbox GPU "
			"passthrough). Refusing to continue because falling back to CPU would "
			"make fit smoke/perf runs misleading."
		)
	count = int(torch.cuda.device_count())
	if device.index is not None and int(device.index) >= count:
		raise RuntimeError(
			f"CUDA device {device} was requested, but PyTorch reports only "
			f"{count} visible CUDA device(s)."
		)


def _grid_center(mdl: "model.Model3D") -> torch.Tensor:
	"""Bilinear center of the model grid — matches (Hm-1)/2, (Wm-1)/2 in station loss."""
	xyz = mdl._grid_xyz()  # (D, Hm, Wm, 3)
	Hm, Wm = xyz.shape[1], xyz.shape[2]
	h_mid, w_mid = (Hm - 1) / 2.0, (Wm - 1) / 2.0
	h0, w0 = int(h_mid), int(w_mid)
	h1, w1 = min(h0 + 1, Hm - 1), min(w0 + 1, Wm - 1)
	fh, fw = h_mid - h0, w_mid - w0
	return ((1 - fh) * (1 - fw) * xyz[0, h0, w0]
	      + fh * (1 - fw) * xyz[0, h1, w0]
	      + (1 - fh) * fw * xyz[0, h0, w1]
	      + fh * fw * xyz[0, h1, w1])


def _optimization_seed_xyz(
	*,
	model_init: str,
	config_seed: tuple[float, float, float] | None,
	mdl: "model.Model3D",
) -> tuple[float, float, float] | None:
	"""Return the station seed used during optimization."""
	if model_init in {"ext", "model"}:
		center_pt = _grid_center(mdl)
		return (float(center_pt[0]), float(center_pt[1]), float(center_pt[2]))
	return config_seed


@dataclasses.dataclass(frozen=True)
class _CorrPointRoiProjection:
	row_index: int
	point_id: int
	collection_id: int
	xyz: tuple[float, float, float]
	h: float
	w: float
	distance: float
	direction_sign: int = 0


@dataclasses.dataclass(frozen=True)
class _CorrPointRoiInit:
	surface: object
	closest: object
	effective_seed: tuple[float, float, float]
	model_w: float
	model_h: float
	projections: list[_CorrPointRoiProjection]
	skipped: list[dict]
	payload: dict


def _unwrap_w_near(w: float, anchor_w: float, width: int) -> float:
	width_f = float(max(1, int(width)))
	dw = math.fmod(float(w) - float(anchor_w) + 0.5 * width_f, width_f)
	if dw < 0.0:
		dw += width_f
	dw -= 0.5 * width_f
	return float(anchor_w) + dw


def _trim_shell_for_anchor(surface, closest, *, source_step: float):
	from init_shell_index import trim_shell_surface_rows_by_quality

	trimmed_surface, trim_top, trim_bottom = trim_shell_surface_rows_by_quality(
		surface,
		target_step=source_step,
		lo_ratio=0.5,
		hi_ratio=2.0,
	)
	if trim_top or trim_bottom:
		if not (float(trim_top) <= float(closest.h) <= float(trim_top + trimmed_surface.xyz_wrapped.shape[0] - 1)):
			raise ValueError(
				f"corr-point-roi source row trim removed closest seed row: "
				f"h={closest.h:.3f} trim_top={trim_top} kept_h={int(trimmed_surface.xyz_wrapped.shape[0])}"
			)
		closest = dataclasses.replace(
			closest,
			h=float(closest.h) - float(trim_top),
			quad_row=max(0, int(closest.quad_row) - int(trim_top)),
		)
		surface = trimmed_surface
	return surface, closest, int(trim_top), int(trim_bottom)


def _projection_skip_reasons(skipped: list[dict]) -> dict[str, int]:
	reasons: dict[str, int] = {}
	for item in skipped:
		reason = str(item.get("reason", "unknown"))
		reasons[reason] = reasons.get(reason, 0) + 1
	return dict(sorted(reasons.items()))


def _sample_corr_point_roi_normals(
	*,
	data: fit_data.FitData3D,
	corr_points: fit_data.CorrPoints3D,
	device: torch.device,
) -> torch.Tensor:
	points = corr_points.points_xyz_winda[:, :3].to(device=device, dtype=torch.float32)
	if points.numel() <= 0:
		return points.reshape(0, 3)
	sample_xyz = points.reshape(1, 1, int(points.shape[0]), 3)
	if data.sparse_caches:
		for cache in data.sparse_caches.values():
			if not ({"nx", "ny"} & set(cache.channels)):
				continue
			cache.prefetch(sample_xyz, data.origin_fullres, data._spacing_for(cache.channels[0]))
		for cache in data.sparse_caches.values():
			if {"nx", "ny"} & set(cache.channels):
				cache.sync()
	sampled = data.grid_sample_fullres(sample_xyz, channels={"nx", "ny"})
	normals = sampled.normal_3d
	if normals is None:
		raise ValueError("corr-point-roi requires nx and ny channels to sample corr point normals")
	normals = normals.reshape(int(points.shape[0]), 3)
	return normals / (normals.norm(dim=-1, keepdim=True) + 1.0e-8)


def _line_project_points_to_selected_shell(
	surface,
	points_xyz: torch.Tensor,
	normals_xyz: torch.Tensor,
	*,
	anchor_hw: tuple[float, float],
	device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
	points = points_xyz.to(device=device, dtype=torch.float32)
	normals = normals_xyz.to(device=device, dtype=torch.float32)
	K = int(points.shape[0])
	if K <= 0:
		empty_f = torch.empty(0, device=device, dtype=torch.float32)
		empty_b = torch.empty(0, device=device, dtype=torch.bool)
		empty_i = torch.empty(0, device=device, dtype=torch.int64)
		return empty_b, empty_f, empty_f, empty_i
	if points.shape != normals.shape or points.shape[-1] != 3:
		raise ValueError(
			f"corr-point-roi line projection expects points/normals shape (K,3), "
			f"got points={tuple(points.shape)} normals={tuple(normals.shape)}"
		)
	shell = surface.xyz_wrapped.to(device=device, dtype=torch.float32)
	H = int(shell.shape[0])
	W = int(surface.unique_w)
	if H < 2 or W < 1:
		raise ValueError(f"corr-point-roi selected shell is too small: shape={tuple(shell.shape[:2])} unique_w={W}")
	p00 = shell[:-1, :W].reshape(-1, 3).contiguous()
	p10 = shell[1:, :W].reshape(-1, 3).contiguous()
	p01 = shell[:-1, 1:W + 1].reshape(-1, 3).contiguous()
	p11 = shell[1:, 1:W + 1].reshape(-1, 3).contiguous()
	Q = int(p00.shape[0])
	row_ids = torch.arange(H - 1, device=device, dtype=torch.float32).view(H - 1, 1).expand(H - 1, W).reshape(-1)
	col_ids = torch.arange(W, device=device, dtype=torch.float32).view(1, W).expand(H - 1, W).reshape(-1)
	best_score = torch.full((K,), float("inf"), device=device, dtype=torch.float32)
	best_h = torch.full((K,), float("nan"), device=device, dtype=torch.float32)
	best_w = torch.full((K,), float("nan"), device=device, dtype=torch.float32)
	best_sign = torch.zeros((K,), device=device, dtype=torch.int64)
	anchor_h, anchor_w = (float(anchor_hw[0]), float(anchor_hw[1]))
	width_f = float(max(1, W))
	batch_quads = max(1, min(32768, 2_000_000 // max(1, K)))
	for q0 in range(0, Q, batch_quads):
		q1 = min(q0 + batch_quads, Q)
		B = int(q1 - q0)
		M00 = p00[q0:q1].view(1, B, 3).expand(K, B, 3)
		M10 = p10[q0:q1].view(1, B, 3).expand(K, B, 3)
		M01 = p01[q0:q1].view(1, B, 3).expand(K, B, 3)
		M11 = p11[q0:q1].view(1, B, 3).expand(K, B, 3)
		O = points.view(K, 1, 3).expand(K, B, 3)
		frac_h = torch.full((K, B), 0.5, device=device, dtype=torch.float32)
		frac_w = torch.full((K, B), 0.5, device=device, dtype=torch.float32)
		rows = row_ids[q0:q1].view(1, B)
		cols = col_ids[q0:q1].view(1, B)
		for sign in (1, -1):
			direction = normals.view(K, 1, 3).expand(K, B, 3) * float(sign)
			u, v = model.Model3D._ray_bilinear_intersect(O, direction, M00, M10, M01, M11, frac_h, frac_w)
			hit = M00 * (1.0 - u.unsqueeze(-1)) * (1.0 - v.unsqueeze(-1))
			hit = hit + M10 * u.unsqueeze(-1) * (1.0 - v.unsqueeze(-1))
			hit = hit + M01 * (1.0 - u.unsqueeze(-1)) * v.unsqueeze(-1)
			hit = hit + M11 * u.unsqueeze(-1) * v.unsqueeze(-1)
			ray_t = ((hit - O) * direction).sum(dim=-1)
			h = rows + u
			w = torch.remainder(cols + v, width_f)
			dw = torch.remainder(w - anchor_w + 0.5 * width_f, width_f) - 0.5 * width_f
			score = (h - anchor_h) ** 2 + dw ** 2
			valid = (
				torch.isfinite(u) & torch.isfinite(v) &
				torch.isfinite(score) & torch.isfinite(ray_t) &
				(u >= 0.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0) &
				(ray_t >= -1.0e-4)
			)
			score = torch.where(valid, score, torch.full_like(score, float("inf")))
			chunk_score, chunk_idx = score.min(dim=1)
			better = chunk_score < best_score
			if bool(better.any().detach().cpu()):
				best_score = torch.where(better, chunk_score, best_score)
				best_h = torch.where(better, h.gather(1, chunk_idx.view(K, 1)).squeeze(1), best_h)
				best_w = torch.where(better, w.gather(1, chunk_idx.view(K, 1)).squeeze(1), best_w)
				best_sign = torch.where(
					better,
					torch.full_like(best_sign, int(sign)),
					best_sign,
				)
	valid = torch.isfinite(best_score)
	return valid, best_h, best_w, best_sign


def _project_corr_points_to_shell(
	surface,
	corr_points: fit_data.CorrPoints3D,
	normals_xyz: torch.Tensor,
	*,
	anchor_hw: tuple[float, float],
	device: torch.device,
) -> tuple[list[_CorrPointRoiProjection], list[dict]]:
	points = corr_points.points_xyz_winda[:, :3].detach().cpu()
	cols = corr_points.collection_idx.detach().cpu()
	pids = corr_points.point_ids.detach().cpu()
	valid, h_t, w_t, sign_t = _line_project_points_to_selected_shell(
		surface,
		corr_points.points_xyz_winda[:, :3],
		normals_xyz,
		anchor_hw=anchor_hw,
		device=device,
	)
	valid_cpu = valid.detach().cpu()
	h_cpu = h_t.detach().cpu()
	w_cpu = w_t.detach().cpu()
	sign_cpu = sign_t.detach().cpu()
	projections: list[_CorrPointRoiProjection] = []
	skipped: list[dict] = []
	for i in range(int(points.shape[0])):
		xyz = tuple(float(v) for v in points[i].tolist())
		if not bool(valid_cpu[i].item()):
			skipped.append({
				"row_index": int(i),
				"point_id": int(pids[i].item()),
				"collection_id": int(cols[i].item()),
				"reason": "no_line_shell_intersection",
			})
			continue
		dh = float(h_cpu[i].item()) - float(anchor_hw[0])
		dw = _unwrap_w_near(float(w_cpu[i].item()), float(anchor_hw[1]), int(surface.unique_w)) - float(anchor_hw[1])
		projections.append(_CorrPointRoiProjection(
			row_index=int(i),
			point_id=int(pids[i].item()),
			collection_id=int(cols[i].item()),
			xyz=xyz,
			h=float(h_cpu[i].item()),
			w=float(w_cpu[i].item()),
			distance=float(math.sqrt(dh * dh + dw * dw)),
			direction_sign=int(sign_cpu[i].item()),
		))
	return projections, skipped


def _derive_corr_point_roi_init(
	*,
	shell_index,
	corr_points: fit_data.CorrPoints3D,
	normals_xyz: torch.Tensor,
	mesh_step: float,
	init_margin_grid_points: int,
	device: torch.device,
):
	if corr_points.points_xyz_winda.shape[0] <= 0:
		raise ValueError("corr-point-roi requires nonempty corr_points")
	points = corr_points.points_xyz_winda[:, :3].to(device=device, dtype=torch.float32)
	mean_xyz = points.mean(dim=0)
	initial_idx = int(torch.argmin(((points - mean_xyz.view(1, 3)) ** 2).sum(dim=-1)).detach().cpu())
	initial_seed = tuple(float(v) for v in points[initial_idx].detach().cpu().tolist())

	def _select_from_seed(seed_xyz: tuple[float, float, float]):
		closest = shell_index.closest_point(seed_xyz, device=device)
		surface = shell_index.surfaces[closest.shell_index]
		source_step = float(surface.source_step) if surface.source_step is not None else float(mesh_step)
		surface, closest, trim_top, trim_bottom = _trim_shell_for_anchor(
			surface,
			closest,
			source_step=source_step,
		)
		projections, skipped = _project_corr_points_to_shell(
			surface,
			corr_points,
			normals_xyz,
			anchor_hw=(float(closest.h), float(closest.w)),
			device=device,
		)
		return closest, surface, projections, skipped, trim_top, trim_bottom

	closest0, surface0, projections0, skipped0, trim_top0, trim_bottom0 = _select_from_seed(initial_seed)
	if not projections0:
		raise ValueError("corr-point-roi could not project any corr_points onto the initial shell")
	width0 = int(surface0.unique_w)
	h_avg = sum(float(p.h) for p in projections0) / float(len(projections0))
	w_unwrapped0 = [_unwrap_w_near(float(p.w), float(closest0.w), width0) for p in projections0]
	w_avg = sum(w_unwrapped0) / float(len(w_unwrapped0))
	recenter_projection = min(
		zip(projections0, w_unwrapped0),
		key=lambda item: (float(item[0].h) - h_avg) ** 2 + (float(item[1]) - w_avg) ** 2,
	)[0]

	closest, surface, projections, skipped, trim_top, trim_bottom = _select_from_seed(recenter_projection.xyz)
	if not projections:
		raise ValueError("corr-point-roi could not project any corr_points onto the final shell")
	width = int(surface.unique_w)
	w_unwrapped = [_unwrap_w_near(float(p.w), float(closest.w), width) for p in projections]
	h_vals = [float(p.h) for p in projections]
	h_span = max(h_vals) - min(h_vals)
	w_span = max(w_unwrapped) - min(w_unwrapped)
	margin_vx = float(max(0, int(init_margin_grid_points))) * float(mesh_step)
	model_h = max(float(mesh_step), h_span * float(mesh_step) + 2.0 * margin_vx)
	model_w = max(float(mesh_step), w_span * float(mesh_step) + 2.0 * margin_vx)
	payload = {
		"mode": "corr-point-roi",
		"projection_mode": "normal-line",
		"init_margin_grid_points": int(init_margin_grid_points),
		"effective_seed": [float(v) for v in recenter_projection.xyz],
		"initial_seed": [float(v) for v in initial_seed],
		"initial_shell_id": str(closest0.shell_id),
		"final_shell_id": str(closest.shell_id),
		"initial_trim_top": int(trim_top0),
		"initial_trim_bottom": int(trim_bottom0),
		"final_trim_top": int(trim_top),
		"final_trim_bottom": int(trim_bottom),
		"model_w": float(model_w),
		"model_h": float(model_h),
		"model_w_unit": "voxels",
		"windings": 1,
		"parsed_point_count": int(corr_points.points_xyz_winda.shape[0]),
		"initial_usable_point_count": int(len(projections0)),
		"initial_skipped_point_count": int(len(skipped0)),
		"initial_skipped_reasons": _projection_skip_reasons(skipped0),
		"usable_point_count": int(len(projections)),
		"skipped_point_count": int(len(skipped)),
		"skipped_reasons": _projection_skip_reasons(skipped),
		"skipped_points": skipped,
		"projected_points": [
			{
				"row_index": int(p.row_index),
				"point_id": int(p.point_id),
				"collection_id": int(p.collection_id),
				"h": float(p.h),
				"w": float(p.w),
				"direction_sign": int(p.direction_sign),
				"grid_distance_to_anchor": float(p.distance),
			}
			for p in projections
		],
		"final_anchor_h": float(closest.h),
		"final_anchor_w": float(closest.w),
		"projected_h_span_grid": float(h_span),
		"projected_w_span_grid": float(w_span),
		"projected_h_span_vx": float(h_span * float(mesh_step)),
		"projected_w_span_vx": float(w_span * float(mesh_step)),
	}
	return _CorrPointRoiInit(
		surface=surface,
		closest=closest,
		effective_seed=recenter_projection.xyz,
		model_w=float(model_w),
		model_h=float(model_h),
		projections=projections,
		skipped=skipped,
		payload=payload,
	)


def _corr_point_roi_mask_from_results(
	corr_results: dict | None,
	*,
	shape: tuple[int, int],
	radius: int,
	device: torch.device,
) -> tuple[torch.Tensor, dict]:
	if not isinstance(corr_results, dict):
		raise ValueError("corr-point-roi output mask requires _corr_points_results_ in the checkpoint")
	points = corr_results.get("points_list", None)
	if not isinstance(points, list):
		raw_points = corr_results.get("points", {})
		points = list(raw_points.values()) if isinstance(raw_points, dict) else []
	H, W = int(shape[0]), int(shape[1])
	seed = torch.zeros((1, 1, H, W), device=device, dtype=torch.float32)
	usable = 0
	skipped: dict[str, int] = {}
	for point in points:
		if not isinstance(point, dict):
			continue
		if not bool(point.get("valid", False)):
			skipped["point_invalid"] = skipped.get("point_invalid", 0) + 1
			continue
		locations = point.get("model_locations", [])
		if not isinstance(locations, list) or not locations:
			skipped["missing_model_locations"] = skipped.get("missing_model_locations", 0) + 1
			continue
		point_used = False
		for loc in locations:
			if not isinstance(loc, dict):
				continue
			try:
				d = int(loc.get("d"))
				h = float(loc.get("h"))
				w = float(loc.get("w"))
			except (TypeError, ValueError):
				skipped["bad_location_value"] = skipped.get("bad_location_value", 0) + 1
				continue
			if d != 0:
				skipped["nonzero_depth_location"] = skipped.get("nonzero_depth_location", 0) + 1
				continue
			if not (math.isfinite(h) and math.isfinite(w)):
				skipped["nonfinite_location"] = skipped.get("nonfinite_location", 0) + 1
				continue
			h0 = math.floor(h)
			h1 = math.ceil(h)
			w0 = math.floor(w)
			w1 = math.ceil(w)
			for hh in {h0, h1}:
				for ww in {w0, w1}:
					if 0 <= hh < H and 0 <= ww < W:
						seed[0, 0, int(hh), int(ww)] = 1.0
						point_used = True
		if point_used:
			usable += 1
	if usable <= 0:
		raise ValueError(f"corr-point-roi output mask has no usable final corr projections; skipped={skipped}")
	r = max(0, int(radius))
	if r > 0:
		mask = F.max_pool2d(seed, kernel_size=2 * r + 1, stride=1, padding=r) > 0.0
	else:
		mask = seed > 0.0
	debug = {
		"usable_point_count": int(usable),
		"seed_vertex_count": int(seed.sum().detach().cpu().item()),
		"dilated_vertex_count": int(mask.sum().detach().cpu().item()),
		"skipped_reasons": dict(sorted(skipped.items())),
	}
	return mask[0, 0], debug


def _first_cylinder_stage_model_step(stages: list[optimizer.Stage]) -> float | None:
	for stage in stages:
		if stage.global_opt is None:
			continue
		if "cyl_params" not in stage.global_opt.params:
			continue
		args = stage.global_opt.args or {}
		value = args.get(optimizer.CYLINDER_STAGE_STEP_ARG)
		if value is None:
			return None
		value_f = float(value)
		return value_f if value_f > 0.0 else None
	return None


def _apply_cylinder_prepare_model_step(mdl: "model.Model3D", model_step: float | None) -> None:
	if model_step is None:
		return
	step = float(model_step)
	if hasattr(mdl, "cyl_shell_width_target_step"):
		mdl.cyl_shell_width_target_step = step
	if hasattr(mdl, "cyl_shell_current_width_step"):
		mdl.cyl_shell_current_width_step = step
	if hasattr(mdl, "cyl_shell_z_step"):
		mdl.cyl_shell_z_step = step
	if hasattr(mdl, "cyl_shell_current_height_step"):
		mdl.cyl_shell_current_height_step = step


def _require_manifest_init_shell_dir(prep_params: dict) -> str:
	value = prep_params.get("init_shell_dir", None)
	if value is None:
		raise ValueError("shell-dir-crop init requires .lasagna.json key 'init_shell_dir'")
	if not isinstance(value, str) or not value.strip():
		raise ValueError("shell-dir-crop init requires non-empty string .lasagna.json key 'init_shell_dir'")
	return str(value)


def _parse_corr_points(obj: dict, device: torch.device) -> fit_data.CorrPoints3D | None:
	"""Parse a VC3D corr_points collections dict into CorrPoints3D."""
	cols = obj.get("collections", {})
	print(f"[fit] _parse_corr_points: {len(cols) if isinstance(cols, dict) else 0} collections in input", flush=True)
	if not isinstance(cols, dict):
		print(f"[fit] _parse_corr_points: collections is not a dict: {type(cols).__name__}", flush=True)
		return None
	rows: list[list[float]] = []
	cids: list[int] = []
	pids: list[int] = []
	abs_flags: list[bool] = []
	for _cid, col in cols.items():
		if not isinstance(col, dict):
			print(f"[fit] _parse_corr_points: col {_cid} is not a dict", flush=True)
			continue
		md = col.get("metadata", {})
		if not isinstance(md, dict):
			md = {}
		is_abs = bool(md.get("winding_is_absolute", True))
		pts = col.get("points", {})
		if not isinstance(pts, dict):
			continue
		try:
			cid_i = int(_cid)
		except Exception:
			cid_i = -1
		n_pts = 0
		for _pid, pd in pts.items():
			if not isinstance(pd, dict):
				continue
			pv = pd.get("p", None)
			if not isinstance(pv, (list, tuple)) or len(pv) < 3:
				continue
			wa = pd.get("wind_a", None)
			if wa is None:
				print(f"[fit] WARNING: corr point {_pid} in collection {_cid} has no wind_a, skipping")
				continue
			try:
				pid_i = int(_pid)
			except Exception:
				pid_i = -1
			rows.append([float(pv[0]), float(pv[1]), float(pv[2]), float(wa)])
			cids.append(cid_i)
			pids.append(pid_i)
			abs_flags.append(is_abs)
			n_pts += 1
		print(f"[fit] _parse_corr_points: col {_cid}: {n_pts} points, "
			  f"absolute={is_abs}", flush=True)
	if not rows:
		print(f"[fit] _parse_corr_points: no valid points found after parsing", flush=True)
		return None
	pts_t = torch.tensor(rows, dtype=torch.float32, device=device)
	col_t = torch.tensor(cids, dtype=torch.int64, device=device)
	pid_t = torch.tensor(pids, dtype=torch.int64, device=device)
	abs_t = torch.tensor(abs_flags, dtype=torch.bool, device=device)
	n_abs = int(abs_t.sum().item())
	print(f"[fit] loaded {pts_t.shape[0]} corr_points from config "
		  f"({len(set(cids))} collections, {n_abs} absolute, {len(rows) - n_abs} relative)")
	return fit_data.CorrPoints3D(points_xyz_winda=pts_t, collection_idx=col_t,
								 point_ids=pid_t, is_absolute=abs_t)


def _build_parser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(
		prog="fit.py",
		description="3D fit entrypoint",
	)
	cli_data.add_args(p)
	cli_model.add_args(p)
	cli_opt.add_args(p)
	p.add_argument("--out-dir", default=None, help="Output directory for snapshots and debug")
	p.add_argument("--tifxyz-init", default=None, help="Initialize model from tifxyz directory instead of model.pt or new model")
	p.add_argument("--model-init", choices=("seed", "ext", "model", "flatten"), default="seed",
		help="Initial model source: seed creates a new model, ext uses --tifxyz-init, model uses --model-input, flatten optimizes one external tifxyz inverse map")
	p.add_argument("--self-map-init", choices=("off", "multi_wrap_full", "multi_wrap_d"), default="off",
		help="Initialize snap_surf_map from the model itself instead of external_surfaces")
	p.add_argument("--flatten-solver", choices=("torch", "inverse", "forward"), default="torch",
		help="Flatten solver variant for model-init=flatten: torch/inverse keeps the existing inverse-map Adam path; forward optimizes source-vertex UVs and inverts at export")
	p.add_argument("--approval-inpaint", action=argparse.BooleanOptionalAction, default=False,
		help="Use selected approval mask/tifxyz data to inpaint the seed-region setup")
	p.add_argument("--approval-inpaint-corr-spacing", type=float, default=None,
		help="Correction point spacing for approval inpaint (default: --mesh-step)")
	p.add_argument("--approval-inpaint-padding-frac", type=float, default=0.25,
		help="Per-side model extent padding for approval inpaint")
	p.add_argument("--approval-inpaint-output-mask", action=argparse.BooleanOptionalAction, default=False,
		help="Store selected approval-inpaint cell as an export mask")
	p.add_argument("--approval-inpaint-output-mask-dilate", type=int, default=3,
		help="Output-mask dilation radius in exported mesh vertices")
	p.add_argument("--approval-inpaint-tifxyz", default=None, help=argparse.SUPPRESS)
	p.add_argument("--tifxyz-flow-gate-channels", action=argparse.BooleanOptionalAction, default=False,
		help="Store flow-gate component maps in the final checkpoint for tifxyz export")
	p.add_argument("--progress", action="store_true", default=False,
		help="Print machine-readable PROGRESS lines to stdout")
	return p


def _fit_config_args_from_namespace(args: argparse.Namespace) -> dict[str, object]:
	out: dict[str, object] = {}
	for k, v in vars(args).items():
		if k == "windings":
			continue
		out[k.replace("_", "-")] = v
	return out


def _validate_self_map_init_args(
	*,
	self_map_init: str,
	model_init: str,
	init_mode: str,
	model_depth: int,
	model_w: float | None,
	model_w_unit: str,
) -> str:
	mode = str(self_map_init if self_map_init is not None else "off").strip().lower().replace("-", "_")
	if mode not in {"off", "multi_wrap_full", "multi_wrap_d"}:
		raise ValueError(f"invalid self-map-init '{self_map_init}' (expected off, multi_wrap_full, or multi_wrap_d)")
	if mode == "off":
		return mode
	if str(model_init).strip().lower() != "seed":
		raise ValueError("self-map-init requires args.model-init=seed")
	if str(init_mode).strip().lower() != "shell-dir-crop":
		raise ValueError("self-map-init requires args.init-mode=shell-dir-crop")
	if mode == "multi_wrap_full":
		if int(model_depth) != 1:
			raise ValueError("self-map-init=multi_wrap_full requires args.depth=1")
	if mode == "multi_wrap_d":
		if int(model_depth) <= 1:
			raise ValueError("self-map-init=multi_wrap_d requires args.depth > 1")
	unit = str(model_w_unit).strip().lower()
	if unit == "wraps":
		_validate_self_map_width_contract(mode=mode, model_w_wraps=None if model_w is None else float(model_w))
	return mode


def _validate_self_map_width_contract(*, mode: str, model_w_wraps: float | None) -> None:
	mode_i = str(mode if mode is not None else "off").strip().lower().replace("-", "_")
	if mode_i == "off":
		return
	wraps = 0.0 if model_w_wraps is None else float(model_w_wraps)
	if mode_i == "multi_wrap_full":
		if wraps <= 1.0:
			raise ValueError("self-map-init=multi_wrap_full requires args.model-w > 1.0 wraps")
	elif mode_i == "multi_wrap_d":
		if not (0.0 < wraps < 1.0):
			raise ValueError("self-map-init=multi_wrap_d requires 0 < args.model-w < 1.0 wraps")


def _dummy_flatten_data() -> fit_data.FitData3D:
	return fit_data.FitData3D(
		cos=None,
		grad_mag=None,
		nx=None,
		ny=None,
		pred_dt=None,
		corr_points=None,
		winding_volume=None,
		origin_fullres=(0.0, 0.0, 0.0),
		spacing=(1.0, 1.0, 1.0),
		channel_spacing=None,
		_vol_size=(1, 1, 1),
		sparse_caches=None,
	)


def _mesh_step_from_tifxyz_meta(meta: dict, fallback: int) -> int:
	scale = meta.get("scale") if isinstance(meta, dict) else None
	if isinstance(scale, list) and scale and float(scale[0]) > 0.0:
		return max(1, int(round(1.0 / float(scale[0]))))
	return max(1, int(fallback))


def _scale_from_tifxyz_meta(meta: dict, mesh_step: int) -> float:
	scale = meta.get("scale") if isinstance(meta, dict) else None
	if isinstance(scale, list) and scale and float(scale[0]) > 0.0:
		return float(scale[0])
	return 1.0 / float(max(1, int(mesh_step)))


def _shape_list(shape: tuple[int, int, int] | None) -> list[int] | None:
	return None if shape is None else [int(v) for v in shape]


def _source_shape_from_tifxyz_meta(meta: dict, fallback_shape_zyx: tuple[int, int, int] | None) -> tuple[int, int, int] | None:
	return volume_scale.tifxyz_source_shape(meta, fallback_shape_zyx)


def _tifxyz_scale_to_base(
	meta: dict,
	*,
	base_shape_zyx: tuple[int, int, int] | None,
	request_shape_zyx: tuple[int, int, int] | None,
	path_label: str,
) -> volume_scale.CoordinateScale:
	source_shape = _source_shape_from_tifxyz_meta(meta, request_shape_zyx)
	return volume_scale.coordinate_scale_to_base(
		base_shape_zyx=base_shape_zyx,
		source_shape_zyx=source_shape,
		source_name=f"{path_label}.base_shape_zyx",
	)


def _load_scaled_tifxyz(
	path: str,
	*,
	device: torch.device,
	base_shape_zyx: tuple[int, int, int] | None,
	request_shape_zyx: tuple[int, int, int] | None,
	path_label: str,
):
	from tifxyz_io import load_tifxyz
	xyz, valid, meta = load_tifxyz(path, device=device)
	scale = _tifxyz_scale_to_base(
		meta,
		base_shape_zyx=base_shape_zyx,
		request_shape_zyx=request_shape_zyx,
		path_label=path_label,
	)
	xyz = volume_scale.scale_tifxyz_tensor(xyz, valid, scale.factor)
	if not scale.is_identity:
		print(
			f"[fit] scaled {path_label} coordinates by {scale.factor:.9g} "
			f"from source_shape={_shape_list(scale.source_shape_zyx)} "
			f"to base_shape={_shape_list(scale.base_shape_zyx)}",
			flush=True,
		)
	return xyz, valid, meta, scale


def _scaled_approval_tifxyz_path(
	path: str,
	*,
	tmp_parent: Path | None,
	base_shape_zyx: tuple[int, int, int] | None,
	request_shape_zyx: tuple[int, int, int] | None,
) -> str:
	meta = volume_scale.read_tifxyz_meta(path)
	scale = _tifxyz_scale_to_base(
		meta,
		base_shape_zyx=base_shape_zyx,
		request_shape_zyx=request_shape_zyx,
		path_label="approval-inpaint tifxyz",
	)
	if scale.is_identity:
		return path
	parent = tmp_parent if tmp_parent is not None else Path(path).parent
	parent.mkdir(parents=True, exist_ok=True)
	dst = parent / "approval_inpaint_base_scale.tifxyz"
	volume_scale.copy_scaled_tifxyz_dir(
		path,
		dst,
		factor=scale.factor,
		base_shape_zyx=base_shape_zyx,
	)
	print(
		f"[fit] approval-inpaint tifxyz scaled by {scale.factor:.9g} "
		f"from source_shape={_shape_list(scale.source_shape_zyx)} "
		f"to base_shape={_shape_list(scale.base_shape_zyx)}",
		flush=True,
	)
	return str(dst)


def _save_flatten_model(path: str, *, mdl: model.Model3D, data: fit_data.FitData3D, fit_config: dict) -> None:
	st = dict(mdl.state_dict())
	for k in [k for k in st if k.startswith("mesh_ms.")]:
		del st[k]
	with torch.no_grad():
		map_yx, xyz, point_mask, _quad_mask = mdl._flatten_sample_current()
		sentinel = torch.full_like(xyz, -1.0)
		xyz = torch.where(point_mask.unsqueeze(0).unsqueeze(-1), xyz, sentinel)
		st["mesh_flat"] = xyz.permute(3, 0, 1, 2).detach().cpu()
		st["flatten_map_flat"] = map_yx.detach().cpu()
		st["flatten_point_mask"] = point_mask.detach().cpu()
	params = asdict(mdl.params)
	if fit_config.get("lasagna_base_shape_zyx") is not None:
		params["lasagna_base_shape_zyx"] = list(fit_config["lasagna_base_shape_zyx"])
	st["_model_params_"] = params
	st["_fit_config_"] = fit_config
	torch.save(st, path)


def _export_flatten_result(
	*,
	mdl: model.Model3D,
	data: fit_data.FitData3D,
	out_dir: Path,
	scale: float,
	voxel_size_um: float | None,
	fit_config: dict,
	model_source: Path | None,
) -> None:
	import numpy as np
	import fit2tifxyz

	out_dir.mkdir(parents=True, exist_ok=True)
	with torch.no_grad():
		_map_yx, xyz, point_mask, _quad_mask = mdl._flatten_sample_current()
	xyz_np = xyz[0].detach().cpu().numpy().astype(np.float32, copy=False)
	mask_np = point_mask.detach().cpu().numpy().astype(bool, copy=False)
	x = np.where(mask_np, xyz_np[..., 0], -1.0).astype(np.float32, copy=False)
	y = np.where(mask_np, xyz_np[..., 1], -1.0).astype(np.float32, copy=False)
	z = np.where(mask_np, xyz_np[..., 2], -1.0).astype(np.float32, copy=False)
	mesh_step = 1.0 / float(scale) if float(scale) > 0.0 else float(mdl.params.mesh_step)
	area = fit2tifxyz._get_area(x, y, z, mesh_step, voxel_size_um)
	fit2tifxyz._write_tifxyz(
		out_dir=out_dir / "flatten.tifxyz",
		x=x,
		y=y,
		z=z,
		scale=scale,
		model_source=model_source,
		fit_config=fit_config,
		area=area,
		base_shape_zyx=volume_scale.parse_shape_zyx(
			fit_config.get("lasagna_base_shape_zyx"), name="lasagna_base_shape_zyx"),
		lasagna_base_shape_zyx=volume_scale.parse_shape_zyx(
			fit_config.get("lasagna_base_shape_zyx"), name="lasagna_base_shape_zyx"),
	)
	fit2tifxyz._print_area(area)


def _run_flatten_mode(
	*,
	cfg: dict,
	fit_config: dict,
	args: argparse.Namespace,
	model_cfg: cli_model.ModelConfig,
	opt_cfg: cli_opt.OptConfig,
	progress_enabled: bool,
	out_dir: str | None,
) -> int:
	ext_surfaces_cfg = cfg.get("external_surfaces", None)
	if not isinstance(ext_surfaces_cfg, list) or len(ext_surfaces_cfg) != 1:
		raise ValueError("model-init=flatten requires exactly one external_surfaces entry")
	ext0 = ext_surfaces_cfg[0]
	if not isinstance(ext0, dict) or not ext0.get("path"):
		raise ValueError("model-init=flatten external_surfaces[0] requires path")
	if getattr(args, "tifxyz_init", None):
		raise ValueError("model-init=flatten uses external_surfaces[0], not --tifxyz-init")
	if model_cfg.model_input is not None:
		raise ValueError("model-init=flatten must not set --model-input")

	device = torch.device(str(getattr(args, "device", "cuda")))
	from tifxyz_io import load_tifxyz
	xyz, valid, meta = load_tifxyz(str(ext0["path"]), device=device)
	mesh_step = _mesh_step_from_tifxyz_meta(meta, model_cfg.mesh_step)
	scale = _scale_from_tifxyz_meta(meta, mesh_step)

	stage_cfg = copy.deepcopy(cfg)
	for key in ("external_surfaces", "tifxyz", "voxel_size_um", "corr_points"):
		stage_cfg.pop(key, None)
	stages = optimizer.load_stages_cfg(stage_cfg, init_mode=None)
	flatten_args: dict[str, object] = {}
	if isinstance(cfg.get("args"), dict):
		flatten_args.update(cfg["args"])
	if stages:
		flatten_args.update(stages[0].global_opt.args or {})
	flatten_solver_raw = flatten_args.get(
		"flatten_solver",
		flatten_args.get("flatten-solver", getattr(args, "flatten_solver", "torch")),
	)
	flatten_direction = model.Model3D._normalize_flatten_direction(str(flatten_solver_raw))
	flatten_output_margin = float(flatten_args.get(
		"flatten_output_margin",
		flatten_args.get("flatten_forward_output_margin", 0.10),
	))
	filter_source_angles = _truthy_config_bool(flatten_args.get("flatten_filter_source_angles", True))
	filter_angle_deg = float(flatten_args.get("flatten_filter_angle_deg", 90.0))
	filter_radius = int(flatten_args.get("flatten_filter_radius", 2))
	mdl = model.Model3D.from_flatten_tifxyz_crop(
		xyz,
		valid,
		device=device,
		mesh_step=mesh_step,
		winding_step=model_cfg.winding_step,
		subsample_mesh=model_cfg.subsample_mesh,
		subsample_winding=model_cfg.subsample_winding,
		flatten_filter_source_angles=filter_source_angles,
		flatten_filter_angle_deg=filter_angle_deg,
		flatten_filter_radius=filter_radius,
		flatten_direction=flatten_direction,
		flatten_output_margin=flatten_output_margin,
	)
	data = _dummy_flatten_data()

	print("data: flatten-only (no volume input)")
	print("model:", model_cfg)
	print("opt:", opt_cfg)
	print(
		f"[fit] model-init=flatten solver={flatten_direction} source={ext0['path']} "
		f"shape={tuple(xyz.shape)} valid={int(valid.sum())}/{valid.numel()} "
		f"model_shape={mdl.mesh_h}x{mdl.mesh_w} "
		f"mesh_step={mesh_step} target_step={float(mdl.flatten_target_step.detach().cpu()):.6g}",
		flush=True,
	)
	filter_stats = getattr(mdl, "flatten_source_filter_stats", {})
	if filter_source_angles:
		print(
			f"[fit] flatten source angle filter: angle>{filter_angle_deg:.4g} radius={max(0, filter_radius)} "
			f"bad_pairs={int(filter_stats.get('bad_pairs', 0.0))} "
			f"bad_cells={int(filter_stats.get('bad_cells', 0.0))} "
			f"dilated={int(filter_stats.get('bad_cells_dilated', 0.0))} "
			f"cell_valid={int(filter_stats.get('cell_valid_after', 0.0))}/"
			f"{int(filter_stats.get('cell_valid_before', 0.0))}",
			flush=True,
		)

	def _snapshot(*, stage: str, step: int, loss: float, data, res=None) -> None:
		if out_dir is None:
			return
		out = Path(out_dir)
		out.mkdir(parents=True, exist_ok=True)
		snaps = out / "model_snapshots"
		snaps.mkdir(parents=True, exist_ok=True)
		_save_flatten_model(str(snaps / f"model_{stage}_{step:06d}.pt"), mdl=mdl, data=data, fit_config=fit_config)

	def _progress(*, step: int, total: int, loss: float, **_kw: object) -> None:
		if progress_enabled:
			print(f"PROGRESS {step} {total} {loss:.6f}", flush=True)

	with torch.no_grad():
		map_yx, xyz0, point_mask, quad_mask = mdl._flatten_sample_current()
		print(
			f"initial flatten: map_shape={tuple(map_yx.shape)} "
			f"point_valid={int(point_mask.sum())}/{point_mask.numel()} "
			f"quad_valid={int(quad_mask.sum())}/{quad_mask.numel()}",
			flush=True,
		)

	optimizer.optimize(
		model=mdl,
		data=data,
		stages=stages,
		snapshot_interval=opt_cfg.snapshot_interval,
		snapshot_fn=_snapshot,
		progress_fn=_progress,
		ensure_data_fn=None,
		seed_xyz=None,
		out_dir=out_dir,
	)

	if device.type == "cuda":
		peak_gb = torch.cuda.max_memory_allocated(device) / 2**30
		print(f"[fit] peak GPU memory: {peak_gb:.2f} GiB", flush=True)

	model_out: str | None = model_cfg.model_output
	if model_out is not None:
		_save_flatten_model(str(model_out), mdl=mdl, data=data, fit_config=fit_config)
		print(f"[fit] saved model to {model_out}")
	if out_dir is not None:
		out = Path(out_dir)
		out.mkdir(parents=True, exist_ok=True)
		final_path = out / "model_final.pt"
		_save_flatten_model(str(final_path), mdl=mdl, data=data, fit_config=fit_config)
		model_source = Path(model_out) if model_out is not None else final_path
		_export_flatten_result(
			mdl=mdl,
			data=data,
			out_dir=out / "tifxyz",
			scale=scale,
			voxel_size_um=(None if cfg.get("voxel_size_um") is None else float(cfg.get("voxel_size_um"))),
			fit_config=fit_config,
			model_source=model_source,
		)
	return 0


def main(argv: list[str] | None = None) -> int:
	if argv is None:
		argv = sys.argv[1:]

	_t_fit_total = _stage_start("total")
	_t = _stage_start("parse_config")
	parser = _build_parser()
	cfg_paths, argv_rest = cli_json.split_cfg_argv(argv)
	cfg_paths = [str(x) for x in cfg_paths]
	cfg = cli_json.merge_cfgs(cfg_paths)
	fit_config = copy.deepcopy(cfg)
	cli_json.apply_defaults_from_cfg_args(parser, cfg)
	args = parser.parse_args(argv_rest or [])

	model_cfg = cli_model.from_args(args)
	requested_model_depth = int(model_cfg.depth)
	args_cfg = cfg.get("args") if isinstance(cfg.get("args"), dict) else None
	grow_enabled = _raw_init_grow_enabled(args_cfg)
	grow_initial_depth = _raw_init_grow_initial_depth(args_cfg) if grow_enabled else requested_model_depth
	grow_stage_depth_delta = _init_grow_stage_depth_delta(cfg) if grow_enabled else 0
	final_model_depth = max(requested_model_depth, grow_initial_depth + grow_stage_depth_delta)
	if final_model_depth != requested_model_depth:
		setattr(args, "depth", final_model_depth)
		model_cfg = dataclasses.replace(model_cfg, depth=final_model_depth)
	init_grow_cfg = _parse_init_grow_config(
		args_cfg,
		target_depth=final_model_depth,
	)
	# Merge final parsed args into fit_config so checkpoint has all values.
	# depth is canonical; legacy args.windings is accepted only as an input alias.
	fit_config.setdefault("args", {}).update(_fit_config_args_from_namespace(args))
	fit_config["args"].pop("windings", None)
	opt_cfg = cli_opt.from_args(args)
	progress_enabled = bool(args.progress)
	_out_dir = args.out_dir
	_stage_done("parse_config", _t)

	model_init = str(getattr(args, "model_init", "seed")).strip().lower()
	self_map_init = str(getattr(args, "self_map_init", "off")).strip().lower().replace("-", "_")
	if model_init not in {"seed", "ext", "model", "flatten"}:
		raise ValueError(f"invalid model-init '{model_init}' (expected seed, ext, model, or flatten)")
	if model_init == "flatten":
		return _run_flatten_mode(
			cfg=cfg,
			fit_config=fit_config,
			args=args,
			model_cfg=model_cfg,
			opt_cfg=opt_cfg,
			progress_enabled=progress_enabled,
			out_dir=_out_dir,
		)

	data_cfg = cli_data.from_args(args)
	if init_grow_cfg.enabled:
		if model_init != "seed" or str(model_cfg.init_mode).strip().lower() != "shell-dir-crop":
			raise ValueError("args.init-grow requires args.model-init=seed and args.init-mode=shell-dir-crop")
		if self_map_init != "multi_wrap_d":
			raise ValueError("args.init-grow currently requires args.self-map-init=multi_wrap_d")
		print(
			f"[fit] init-grow enabled: initial_depth={init_grow_cfg.initial_depth} "
			f"target_depth={final_model_depth} order={list(init_grow_cfg.order)} step={init_grow_cfg.step}",
			flush=True,
		)
	print("data:", data_cfg)
	print("model:", model_cfg)
	print("opt:", opt_cfg)

	device = torch.device(data_cfg.device)
	_require_torch_device_available(device)
	init_mode = str(model_cfg.init_mode).strip().lower()
	self_map_init = _validate_self_map_init_args(
		self_map_init=self_map_init,
		model_init=model_init,
		init_mode=init_mode,
		model_depth=int(model_cfg.depth),
		model_w=data_cfg.model_w,
		model_w_unit=data_cfg.model_w_unit,
	)
	self_map_model_w_wraps: float | None = (
		None if data_cfg.model_w is None else float(data_cfg.model_w)
	) if data_cfg.model_w_unit == "wraps" else None
	if init_mode == "shell-dir-crop" and model_init != "seed":
		raise ValueError("init-mode=shell-dir-crop requires args.model-init=seed")
	if init_mode == "shell-dir-crop" and "init_shell_dir" in cfg:
		raise ValueError("do not set top-level config key 'init_shell_dir'; shell-dir-crop reads it from --input .lasagna.json")

	# Probe preprocessed data for scaledown and volume extent (in base/VC3D coords)
	_t = _stage_start("probe_preprocessed_data")
	prep_params = fit_data.get_preprocessed_params(str(data_cfg.input))
	source_to_base = float(prep_params.get("source_to_base", 1.0))
	lasagna_base_shape_zyx = volume_scale.parse_shape_zyx(
		prep_params.get("base_shape_zyx"), name="lasagna_base_shape_zyx")
	vc3d_volume_shape_zyx = volume_scale.parse_shape_zyx(
		cfg.get("vc3d_volume_shape_zyx"), name="vc3d_volume_shape_zyx")
	request_scale = volume_scale.coordinate_scale_to_base(
		base_shape_zyx=lasagna_base_shape_zyx,
		source_shape_zyx=vc3d_volume_shape_zyx,
		source_name="vc3d_volume_shape_zyx",
	)
	fit_config["lasagna_base_shape_zyx"] = _shape_list(lasagna_base_shape_zyx)
	if vc3d_volume_shape_zyx is not None:
		fit_config["vc3d_volume_shape_zyx"] = _shape_list(vc3d_volume_shape_zyx)
	if not request_scale.is_identity:
		print(
			f"[fit] VC3D coordinate import scale={request_scale.factor:.9g} "
			f"vc3d_shape={_shape_list(vc3d_volume_shape_zyx)} "
			f"lasagna_base_shape={_shape_list(lasagna_base_shape_zyx)}",
			flush=True,
		)
	if data_cfg.seed is not None:
		scaled_seed = tuple(float(v) for v in volume_scale.scale_xyz_point(data_cfg.seed, request_scale.factor)[:3])
		data_cfg = dataclasses.replace(data_cfg, seed=scaled_seed)
		fit_config.setdefault("args", {})["seed"] = [float(v) for v in scaled_seed]
	if isinstance(cfg.get("corr_points"), dict):
		scaled_corr = volume_scale.scale_corr_points_json(cfg["corr_points"], request_scale.factor)
		cfg["corr_points"] = scaled_corr
		fit_config["corr_points"] = copy.deepcopy(scaled_corr)
	# Model scaledown in base coords = channel_scaledown * source_to_base
	scaledown = float(prep_params["scaledown"]) * source_to_base
	volume_extent_fullres = prep_params.get("volume_extent_fullres")
	print(f"[fit] scaledown={scaledown} (source_sd={prep_params['scaledown']} "
		  f"source_to_base={source_to_base}) volume_extent={volume_extent_fullres}", flush=True)
	_stage_done("probe_preprocessed_data", _t)

	# Approval inpaint is a seed-mode preprocessor: VC3D sends the selected
	# tifxyz plus approval/d channels, then this step derives corr points,
	# a centered effective seed, and extents before the configured init runs.
	_t = _stage_start("approval_inpaint")
	approval_inpaint_enabled = _truthy_config_bool(getattr(args, "approval_inpaint", False))
	approval_inpaint_output_mask_enabled = _truthy_config_bool(getattr(args, "approval_inpaint_output_mask", False))
	if approval_inpaint_output_mask_enabled and not approval_inpaint_enabled:
		raise ValueError("approval-inpaint-output-mask requires approval-inpaint=true")
	approval_inpaint_output_mask: dict | None = None
	if approval_inpaint_enabled:
		if model_init != "seed":
			raise ValueError("approval-inpaint requires args.model-init=seed")
		if data_cfg.seed is None:
			raise ValueError("approval-inpaint requires args.seed")
		approval_tifxyz = getattr(args, "approval_inpaint_tifxyz", None)
		if not approval_tifxyz:
			raise ValueError("approval-inpaint requires service arg approval-inpaint-tifxyz")
		from approval_inpaint import build_approval_inpaint
		approval_tifxyz = _scaled_approval_tifxyz_path(
			str(approval_tifxyz),
			tmp_parent=None,
			base_shape_zyx=lasagna_base_shape_zyx,
			request_shape_zyx=vc3d_volume_shape_zyx,
		)

		result = build_approval_inpaint(
			tifxyz_path=str(approval_tifxyz),
			seed=tuple(float(v) for v in data_cfg.seed),
			mesh_step=float(model_cfg.mesh_step),
			corr_spacing=getattr(args, "approval_inpaint_corr_spacing", None),
			padding_frac=getattr(args, "approval_inpaint_padding_frac", 0.25),
			existing_corr_points=cfg.get("corr_points") if isinstance(cfg.get("corr_points"), dict) else None,
			output_mask=approval_inpaint_output_mask_enabled,
			output_mask_dilate=int(getattr(args, "approval_inpaint_output_mask_dilate", 3)),
		)
		approval_inpaint_output_mask = result.output_mask
		data_cfg = dataclasses.replace(
			data_cfg,
			seed=result.seed,
			model_w=result.model_w,
			model_w_unit="voxels",
			model_h=result.model_h,
		)
		cfg["corr_points"] = result.corr_points
		fit_config["corr_points"] = copy.deepcopy(result.corr_points)
		fit_config.setdefault("args", {}).update({
			"seed": [float(v) for v in result.seed],
			"model-w": int(result.model_w),
			"model-w-unit": "voxels",
			"model-h": int(result.model_h),
			"approval-inpaint": True,
			"approval-inpaint-output-mask": bool(approval_inpaint_output_mask_enabled),
			"approval-inpaint-output-mask-dilate": int(getattr(args, "approval_inpaint_output_mask_dilate", 3)),
		})
		if approval_inpaint_output_mask is not None:
			fit_config["args"]["approval-inpaint-output-mask-source"] = str(
				approval_inpaint_output_mask.get("source", "corr_points")
			)
			fit_config["args"]["approval-inpaint-output-mask-corr-collections"] = [
				int(v) for v in approval_inpaint_output_mask.get("corr_collection_ids", [])
			]
		print(
			f"[fit] approval-inpaint: points={result.point_count} "
			f"component={result.component_size} skeleton={result.skeleton_size} "
			f"bounds={result.index_bounds} source_step={result.source_mesh_step:.3f} "
			f"seed=({result.seed[0]:.1f},{result.seed[1]:.1f},{result.seed[2]:.1f}) "
			f"model_w={result.model_w} model_h={result.model_h}",
			flush=True,
		)
	_stage_done("approval_inpaint", _t)

	_t = _stage_start("corr_point_roi_init")
	corr_point_roi_enabled = _truthy_config_bool(getattr(args, "corr_point_roi", False))
	corr_point_roi_init: _CorrPointRoiInit | None = None
	corr_point_roi_shell_index = None
	if corr_point_roi_enabled:
		if model_init != "seed":
			raise ValueError("corr-point-roi requires args.model-init=seed")
		if init_mode != "shell-dir-crop":
			raise ValueError("corr-point-roi requires args.init-mode=shell-dir-crop")
		corr_points_obj_for_roi = cfg.get("corr_points")
		if not isinstance(corr_points_obj_for_roi, dict):
			raise ValueError("corr-point-roi requires nonempty corr_points")
		corr_points_3d_for_roi = _parse_corr_points(corr_points_obj_for_roi, device)
		if corr_points_3d_for_roi is None or corr_points_3d_for_roi.points_xyz_winda.shape[0] <= 0:
			raise ValueError("corr-point-roi requires nonempty corr_points")
		from init_shell_index import InitShellIndex
		init_shell_dir = _require_manifest_init_shell_dir(prep_params)
		corr_point_roi_shell_index = InitShellIndex.from_directory(init_shell_dir)
		if device.type == "cuda":
			corr_point_roi_normal_data = fit_data.load_3d_streaming(
				path=str(data_cfg.input),
				device=device,
				sparse_prefetch_backend=data_cfg.sparse_prefetch_backend,
				skip_channels={"cos", "pred_dt"},
			)
		else:
			corr_point_roi_normal_data = fit_data.load_3d(
				path=str(data_cfg.input),
				device=device,
				cuda_gridsample=(device.type == "cuda" and bool(data_cfg.cuda_gridsample)),
				skip_channels={"cos", "pred_dt"},
			)
		corr_point_roi_normals = _sample_corr_point_roi_normals(
			data=corr_point_roi_normal_data,
			corr_points=corr_points_3d_for_roi,
			device=device,
		)
		corr_point_roi_init = _derive_corr_point_roi_init(
			shell_index=corr_point_roi_shell_index,
			corr_points=corr_points_3d_for_roi,
			normals_xyz=corr_point_roi_normals,
			mesh_step=float(model_cfg.mesh_step),
			init_margin_grid_points=int(data_cfg.corr_point_roi_init_margin),
			device=device,
		)
		corr_point_roi_init.payload["output_radius_grid_points"] = int(data_cfg.corr_point_roi_output_radius)
		data_cfg = dataclasses.replace(
			data_cfg,
			seed=corr_point_roi_init.effective_seed,
			model_w=corr_point_roi_init.model_w,
			model_w_unit="voxels",
			model_h=corr_point_roi_init.model_h,
			windings=1,
		)
		model_cfg = dataclasses.replace(model_cfg, depth=1, pyramid_d=False)
		fit_config.setdefault("args", {}).update({
			"corr-point-roi": True,
			"corr-point-roi-init-margin": int(data_cfg.corr_point_roi_init_margin),
			"corr-point-roi-output-radius": int(data_cfg.corr_point_roi_output_radius),
			"seed": [float(v) for v in corr_point_roi_init.effective_seed],
			"model-w": float(corr_point_roi_init.model_w),
			"model-w-unit": "voxels",
			"model-h": float(corr_point_roi_init.model_h),
			"windings": 1,
			"depth": 1,
			"pyramid-d": False,
		})
		fit_config["_corr_point_roi_init_"] = copy.deepcopy(corr_point_roi_init.payload)
		print(
			f"[fit] corr-point-roi: shell={corr_point_roi_init.payload['final_shell_id']} "
			f"line_hits={corr_point_roi_init.payload['usable_point_count']}/"
			f"{corr_point_roi_init.payload['parsed_point_count']} "
			f"skipped={corr_point_roi_init.payload['skipped_reasons']} "
			f"seed=({corr_point_roi_init.effective_seed[0]:.1f},"
			f"{corr_point_roi_init.effective_seed[1]:.1f},"
			f"{corr_point_roi_init.effective_seed[2]:.1f}) "
			f"model_w={corr_point_roi_init.model_w:.1f} model_h={corr_point_roi_init.model_h:.1f}",
			flush=True,
		)
	_stage_done("corr_point_roi_init", _t)

	# --- Init from seed (new model only) ---
	_t = _stage_start("derive_initial_model_params")
	if model_init == "seed":
		if getattr(args, "tifxyz_init", None):
			raise ValueError("model-init=seed must not set --tifxyz-init; tifxyz can only be used as external_surfaces")
		if model_cfg.model_input is not None:
			raise ValueError("model-init=seed must not set --model-input")
		missing_seed = []
		if data_cfg.seed is None:
			missing_seed.append("--seed")
		if data_cfg.model_h is None:
			missing_seed.append("--model-h")
		if missing_seed:
			raise ValueError(f"model-init=seed requires {', '.join(missing_seed)}")
	elif model_init == "ext":
		if not getattr(args, "tifxyz_init", None):
			raise ValueError("model-init=ext requires --tifxyz-init")
		if model_cfg.model_input is not None:
			raise ValueError("model-init=ext must not set --model-input")
	elif model_init == "model":
		if model_cfg.model_input is None:
			raise ValueError("model-init=model requires --model-input")
		if getattr(args, "tifxyz_init", None):
			raise ValueError("model-init=model must not set --tifxyz-init; tifxyz can only be used as external_surfaces")

	if model_init == "seed" and data_cfg.seed is not None:
		model_cfg = dataclasses.replace(model_cfg, z_center=float(data_cfg.seed[2]))
		label = "shell-dir-crop" if init_mode == "shell-dir-crop" else "cylinder_seed"
		print(f"[fit] {label} from seed: x={float(data_cfg.seed[0]):.1f} "
			  f"y={float(data_cfg.seed[1]):.1f} z={float(data_cfg.seed[2]):.1f}",
			  flush=True)

	# --- Size mesh from model_h only for the umbilicus tube experiment ---
	if model_init == "seed" and init_mode == "cylinder_seed" and data_cfg.model_h is not None:
		tube_z_step = 1000.0
		auto_mesh_w = 20
		auto_mesh_h = max(2, int(math.ceil(float(data_cfg.model_h) / tube_z_step)) + 1)
		auto_depth = 1
		actual_z_step = float(data_cfg.model_h) / float(max(1, auto_mesh_h - 1))

		model_cfg = dataclasses.replace(model_cfg, depth=auto_depth, mesh_h=auto_mesh_h, mesh_w=auto_mesh_w)
		print(f"[fit] model size: depth={auto_depth} mesh_h={auto_mesh_h} mesh_w={auto_mesh_w} "
			  f"z_step={actual_z_step:.1f} z_step_target={tube_z_step:.1f} "
			  f"(umbilicus tube search grid; final mesh bake uses model-w/model-h/mesh-step)", flush=True)
	_stage_done("derive_initial_model_params", _t)

	tifxyz_init = getattr(args, "tifxyz_init", None)

	# --- Construct / load model (before data, so we can compute bbox) ---
	_t = _stage_start("construct_model")
	if model_init == "ext":
		from tifxyz_io import surface_step_stats
		xyz_init, valid_init, _meta_init, _scale_init = _load_scaled_tifxyz(
			str(tifxyz_init),
			device=device,
			base_shape_zyx=lasagna_base_shape_zyx,
			request_shape_zyx=vc3d_volume_shape_zyx,
			path_label="tifxyz-init",
		)
		_step_h, _step_w, _step_diag, step_avg = surface_step_stats(xyz_init, valid_init)
		mesh_step_init = model_cfg.mesh_step
		if math.isfinite(step_avg) and step_avg > 0.0:
			mesh_step_init = max(1, int(round(step_avg)))
		mdl = model.Model3D.from_tifxyz_crop(
			xyz_init,
			valid_init,
			device=device,
			mesh_step=mesh_step_init,
			winding_step=model_cfg.winding_step,
			subsample_mesh=model_cfg.subsample_mesh,
			subsample_winding=model_cfg.subsample_winding,
		)
		print(f"[fit] initialized from tifxyz: {tifxyz_init}", flush=True)
	elif model_init == "seed":
		if init_mode == "shell-dir-crop":
			print("[fit] model-init=seed/init-mode=shell-dir-crop: constructing model from init shells", flush=True)
			from init_shell_index import (
				InitShellIndex,
				crop_shell_surface,
				shell_quality_analysis,
				trim_shell_surface_rows_by_quality,
			)
			init_shell_dir = _require_manifest_init_shell_dir(prep_params)
			if corr_point_roi_init is not None:
				shell_index = corr_point_roi_shell_index if corr_point_roi_shell_index is not None else InitShellIndex.from_directory(init_shell_dir)
				closest = corr_point_roi_init.closest
				surface = corr_point_roi_init.surface
			else:
				shell_index = InitShellIndex.from_directory(init_shell_dir)
				closest = shell_index.closest_point(tuple(float(v) for v in data_cfg.seed), device=device)
				surface = shell_index.surfaces[closest.shell_index]
			source_step = float(surface.source_step) if surface.source_step is not None else float(model_cfg.mesh_step)
			selected_shell = surface.xyz_wrapped[:, :surface.unique_w].to(device=device, dtype=torch.float32)
			print(
				f"[fit] shell-dir-crop closest shell before crop: "
				f"id={closest.shell_id} path={surface.path} "
				f"source_step={source_step:.3f} "
				f"source_shape={int(surface.xyz_wrapped.shape[0])}x{int(surface.xyz_wrapped.shape[1])} "
				f"unique_shape={int(selected_shell.shape[0])}x{int(selected_shell.shape[1])} "
				f"quad=({closest.quad_row},{closest.quad_col}) tri={closest.triangle_id} "
				f"h={closest.h:.3f} w={closest.w:.3f} dist={closest.distance:.3f}",
				flush=True,
			)
			source_quality = shell_quality_analysis(selected_shell, target_step=source_step)
			print(
				f"[fit] shell-dir-crop source-shell quality before row trim: "
				f"target_step={source_quality['target_step']:.3f} target_area={source_quality['target_area']:.3f} "
				f"h=({source_quality['h_min']:.3f},{source_quality['h_med']:.3f},{source_quality['h_max']:.3f}) "
				f"w_top=({source_quality['w_top_min']:.3f},{source_quality['w_top_med']:.3f},{source_quality['w_top_max']:.3f}) "
				f"w_bottom=({source_quality['w_bottom_min']:.3f},{source_quality['w_bottom_med']:.3f},{source_quality['w_bottom_max']:.3f}) "
				f"diag_main=({source_quality['diag_main_min']:.3f},{source_quality['diag_main_med']:.3f},{source_quality['diag_main_max']:.3f}) "
				f"diag_anti=({source_quality['diag_anti_min']:.3f},{source_quality['diag_anti_med']:.3f},{source_quality['diag_anti_max']:.3f}) "
				f"area=({source_quality['area_min']:.3f},{source_quality['area_med']:.3f},{source_quality['area_max']:.3f}) "
				f"area_sqrt=({source_quality['area_sqrt_min']:.3f},{source_quality['area_sqrt_med']:.3f},{source_quality['area_sqrt_max']:.3f})",
				flush=True,
			)
			if corr_point_roi_init is None:
				trimmed_surface, trim_top, trim_bottom = trim_shell_surface_rows_by_quality(
					surface,
					target_step=source_step,
					lo_ratio=0.5,
					hi_ratio=2.0,
				)
				if trim_top or trim_bottom:
					if not (float(trim_top) <= float(closest.h) <= float(trim_top + trimmed_surface.xyz_wrapped.shape[0] - 1)):
						raise ValueError(
							f"shell-dir-crop source row trim removed closest seed row: "
							f"h={closest.h:.3f} trim_top={trim_top} kept_h={int(trimmed_surface.xyz_wrapped.shape[0])}"
						)
					closest = dataclasses.replace(
						closest,
						h=float(closest.h) - float(trim_top),
						quad_row=max(0, int(closest.quad_row) - int(trim_top)),
					)
					surface = trimmed_surface
					selected_shell = surface.xyz_wrapped[:, :surface.unique_w].to(device=device, dtype=torch.float32)
					trim_quality = shell_quality_analysis(selected_shell, target_step=source_step)
					print(
						f"[fit] shell-dir-crop source-shell row trim: "
						f"trim_top={trim_top} trim_bottom={trim_bottom} "
						f"kept_shape={int(surface.xyz_wrapped.shape[0])}x{int(surface.xyz_wrapped.shape[1])} "
						f"adjusted_h={closest.h:.3f} "
						f"h=({trim_quality['h_min']:.3f},{trim_quality['h_med']:.3f},{trim_quality['h_max']:.3f}) "
						f"w_top=({trim_quality['w_top_min']:.3f},{trim_quality['w_top_med']:.3f},{trim_quality['w_top_max']:.3f}) "
						f"w_bottom=({trim_quality['w_bottom_min']:.3f},{trim_quality['w_bottom_med']:.3f},{trim_quality['w_bottom_max']:.3f}) "
						f"diag_main=({trim_quality['diag_main_min']:.3f},{trim_quality['diag_main_med']:.3f},{trim_quality['diag_main_max']:.3f}) "
						f"diag_anti=({trim_quality['diag_anti_min']:.3f},{trim_quality['diag_anti_med']:.3f},{trim_quality['diag_anti_max']:.3f}) "
						f"area=({trim_quality['area_min']:.3f},{trim_quality['area_med']:.3f},{trim_quality['area_max']:.3f}) "
						f"area_sqrt=({trim_quality['area_sqrt_min']:.3f},{trim_quality['area_sqrt_med']:.3f},{trim_quality['area_sqrt_max']:.3f})",
						flush=True,
					)
			if _SHELL_STEP_ANALYSIS_ENABLED:
				step_stats = opt_loss_step.step_loss_analysis(selected_shell, mesh_step=source_step)
				print(
					f"[fit] shell-dir-crop selected-shell step analysis before crop: "
					f"loss={step_stats['loss']:.6g} target={step_stats['target']:.3f} "
					f"step_min={step_stats['step_min']:.3f} step_avg={step_stats['step_avg']:.3f} "
					f"step_med={step_stats['step_med']:.3f} step_max={step_stats['step_max']:.3f} "
					f"h_avg={step_stats['h_avg']:.3f} w_avg={step_stats['w_avg']:.3f} "
					f"diag_avg={step_stats['diag_avg']:.3f} "
					f"h_max={step_stats['h_max']:.3f} w_max={step_stats['w_max']:.3f} "
					f"diag_max={step_stats['diag_max']:.3f} max_kind={step_stats['max_kind']}",
					flush=True,
				)
			crop_xyz, crop_valid, crop_info = crop_shell_surface(
				surface,
				closest,
				seed=tuple(float(v) for v in data_cfg.seed),
				model_w=float(data_cfg.model_w) if data_cfg.model_w is not None else 0.0,
				model_h=float(data_cfg.model_h),
				model_w_unit=data_cfg.model_w_unit,
				mesh_step=float(model_cfg.mesh_step),
				device=device,
			)
			if self_map_init != "off":
				self_map_model_w_wraps = float(crop_info.requested_width_wraps)
				_validate_self_map_width_contract(
					mode=self_map_init,
					model_w_wraps=self_map_model_w_wraps,
				)
			if _SHELL_STEP_ANALYSIS_ENABLED:
				crop_step_stats = opt_loss_step.step_loss_analysis(crop_xyz, mesh_step=float(model_cfg.mesh_step))
				print(
					f"[fit] shell-dir-crop resampled-crop step analysis: "
					f"loss={crop_step_stats['loss']:.6g} target={crop_step_stats['target']:.3f} "
					f"step_min={crop_step_stats['step_min']:.3f} step_avg={crop_step_stats['step_avg']:.3f} "
					f"step_med={crop_step_stats['step_med']:.3f} step_max={crop_step_stats['step_max']:.3f} "
					f"h_avg={crop_step_stats['h_avg']:.3f} w_avg={crop_step_stats['w_avg']:.3f} "
					f"diag_avg={crop_step_stats['diag_avg']:.3f} "
					f"h_max={crop_step_stats['h_max']:.3f} w_max={crop_step_stats['w_max']:.3f} "
					f"diag_max={crop_step_stats['diag_max']:.3f} max_kind={crop_step_stats['max_kind']}",
					flush=True,
				)
			mdl = model.Model3D.from_tifxyz_crop(
				crop_xyz,
				crop_valid,
				device=device,
				mesh_step=model_cfg.mesh_step,
				winding_step=model_cfg.winding_step,
				subsample_mesh=model_cfg.subsample_mesh,
				subsample_winding=model_cfg.subsample_winding,
				depth=(
					init_grow_cfg.initial_depth
					if init_grow_cfg.enabled
					else (model_cfg.depth if self_map_init == "multi_wrap_d" else 1)
				),
			)
			mdl.params = dataclasses.replace(
				mdl.params,
				scaledown=scaledown,
				z_step_eff=int(round(scaledown)),
				volume_extent=None,
				model_w=(None if data_cfg.model_w is None else float(data_cfg.model_w)),
				model_h=float(data_cfg.model_h),
			)
			if _SHELL_STEP_ANALYSIS_ENABLED:
				model_step_stats = opt_loss_step.step_loss_analysis(mdl._grid_xyz().detach(), mesh_step=float(model_cfg.mesh_step))
				print(
					f"[fit] shell-dir-crop model-init step analysis: "
					f"loss={model_step_stats['loss']:.6g} target={model_step_stats['target']:.3f} "
					f"step_min={model_step_stats['step_min']:.3f} step_avg={model_step_stats['step_avg']:.3f} "
					f"step_med={model_step_stats['step_med']:.3f} step_max={model_step_stats['step_max']:.3f} "
					f"h_avg={model_step_stats['h_avg']:.3f} w_avg={model_step_stats['w_avg']:.3f} "
					f"diag_avg={model_step_stats['diag_avg']:.3f} "
					f"h_max={model_step_stats['h_max']:.3f} w_max={model_step_stats['w_max']:.3f} "
					f"diag_max={model_step_stats['diag_max']:.3f} max_kind={model_step_stats['max_kind']}",
					flush=True,
				)
			print(
				f"[fit] shell-dir-crop selected {closest.shell_id}: "
				f"quad=({closest.quad_row},{closest.quad_col}) tri={closest.triangle_id} "
				f"h={closest.h:.3f} w={closest.w:.3f} "
				f"dist={closest.distance:.3f} "
				f"crop={crop_info.mesh_h}x{crop_info.mesh_w} "
				f"requested_h={crop_info.requested_mesh_h} "
				f"dropped_h={crop_info.requested_mesh_h - crop_info.mesh_h} "
				f"dropped_h_low={crop_info.height_dropped_low} "
				f"dropped_h_high={crop_info.height_dropped_high} "
				f"source={crop_info.source_h}x{crop_info.source_w} "
				f"full_width={crop_info.full_width} "
				f"requested_width_wraps={crop_info.requested_width_wraps:.6g}",
				flush=True,
			)
		elif init_mode == "cylinder_seed":
			print(f"[fit] model-init=seed: constructing model from seed", flush=True)
			mdl = model.Model3D(
				device=device,
				depth=model_cfg.depth,
				mesh_h=model_cfg.mesh_h,
				mesh_w=model_cfg.mesh_w,
				mesh_step=model_cfg.mesh_step,
				winding_step=model_cfg.winding_step,
				subsample_mesh=model_cfg.subsample_mesh,
				subsample_winding=model_cfg.subsample_winding,
				scaledown=scaledown,
				z_step_eff=int(round(scaledown)),
				z_center=model_cfg.z_center,
				init_mode=model_cfg.init_mode,
				volume_extent=None,
				pyramid_d=model_cfg.pyramid_d,
			)
			mdl.init_cylinder_seed(
				seed=tuple(float(v) for v in data_cfg.seed),
				model_w=float(data_cfg.model_w) if data_cfg.model_w is not None else 0.0,
				model_h=float(data_cfg.model_h),
				volume_extent_fullres=volume_extent_fullres,
			)
		else:
			raise ValueError(f"unsupported init-mode for model-init=seed: {init_mode}")
	else:
		print(f"[fit] model-init=model: loading checkpoint {model_cfg.model_input}", flush=True)
		st = torch.load(model_cfg.model_input, map_location=device, weights_only=False)
		mdl = model.Model3D.from_checkpoint(st, device=device)

	print(f"Model3D: depth={mdl.depth} mesh_h={mdl.mesh_h} mesh_w={mdl.mesh_w} "
		  f"cylinder_enabled={getattr(mdl, 'cylinder_enabled', False)}")
	_stage_done("construct_model", _t)

	# Load external reference surfaces
	_t = _stage_start("load_external_surfaces")
	ext_surfaces_cfg = cfg.pop("external_surfaces", None)
	if isinstance(ext_surfaces_cfg, list) and ext_surfaces_cfg:
		if len(ext_surfaces_cfg) != 1:
			raise ValueError(
				f"external_surfaces currently requires exactly one entry, got {len(ext_surfaces_cfg)}")
		from tifxyz_io import surface_step_stats
		for es in ext_surfaces_cfg:
			es_path = str(es["path"])
			es_offset = float(es.get("offset", 1.0))
			xyz_ext, valid_ext, meta_ext, es_scale = _load_scaled_tifxyz(
				es_path,
				device=device,
				base_shape_zyx=lasagna_base_shape_zyx,
				request_shape_zyx=vc3d_volume_shape_zyx,
				path_label=f"external surface {es_path}",
			)
			idx = mdl.add_external_surface(xyz_ext, valid=valid_ext, offset=es_offset)
			meta_ext_base = volume_scale.scale_tifxyz_meta(
				meta_ext,
				es_scale.factor,
				base_shape_zyx=lasagna_base_shape_zyx,
				lasagna_base_shape_zyx=lasagna_base_shape_zyx,
			)
			scale = meta_ext_base.get("scale") if isinstance(meta_ext_base, dict) else None
			meta_step = float("nan")
			if isinstance(scale, list) and scale and float(scale[0]) > 0.0:
				meta_step = 1.0 / float(scale[0])
			step_h, step_w, step_diag, step_avg = surface_step_stats(xyz_ext, valid_ext)
			ratio = step_avg / max(1.0e-8, float(mdl.params.mesh_step))
			print(f"[fit] external surface {idx}: path={es_path} offset={es_offset} "
				  f"shape={tuple(xyz_ext.shape)} valid={int(valid_ext.sum())}/{valid_ext.numel()} "
				  f"meta_step={meta_step:.3f} step_h={step_h:.3f} step_w={step_w:.3f} step_diag={step_diag:.3f} "
				  f"step_avg={step_avg:.3f} model_step={float(mdl.params.mesh_step):.3f} "
				  f"step_ratio={ratio:.3f}", flush=True)
	_stage_done("load_external_surfaces", _t)

	# Parse correction points from config (injected by VC3D)
	_t = _stage_start("parse_corr_points")
	corr_points_obj = cfg.pop("corr_points", None)
	corr_points_3d: fit_data.CorrPoints3D | None = None
	if isinstance(corr_points_obj, dict):
		corr_points_3d = _parse_corr_points(corr_points_obj, device)
	else:
		print(f"[fit] corr_points: not found in config (type={type(corr_points_obj).__name__})", flush=True)
	_stage_done("parse_corr_points", _t)

	# Strip non-stage keys before parsing stages
	_t = _stage_start("load_optimizer_stages")
	cfg.pop("args", None)
	cfg.pop("voxel_size_um", None)
	cfg.pop("external_surfaces", None)
	cfg.pop("tifxyz", None)
	stages = optimizer.load_stages_cfg(
		cfg,
		init_mode=model_cfg.init_mode if model_init == "seed" else None,
	)
	print("[fit] optimizer stages:", flush=True)
	for i, st in enumerate(stages):
		if st.children:
			print(
				f"[fit]   stage{i} name={st.name!r} expand wrapper children={len(st.children)} grow={st.grow}",
				flush=True,
			)
			for j, child in enumerate(st.children):
				if child.global_opt is None:
					print(f"[fit]     child{j} name={child.name!r} wrapper", flush=True)
					continue
				args_snap_map = child.global_opt.args.get("snap_surf_map") if isinstance(child.global_opt.args, dict) else None
				print(
					f"[fit]     child{j} name={child.name!r} steps={child.global_opt.steps} "
					f"snap_surf_map_eff={child.global_opt.eff.get('snap_surf_map', 0.0):.6g} "
					f"snap_surf_map_args={args_snap_map}",
					flush=True,
				)
			continue
		if st.global_opt is None:
			print(f"[fit]   stage{i} name={st.name!r} empty wrapper", flush=True)
			continue
		args_snap_map = st.global_opt.args.get("snap_surf_map") if isinstance(st.global_opt.args, dict) else None
		print(
			f"[fit]   stage{i} name={st.name!r} steps={st.global_opt.steps} "
			f"snap_surf_map_eff={st.global_opt.eff.get('snap_surf_map', 0.0):.6g} "
			f"snap_surf_map_args={args_snap_map}",
			flush=True,
		)
	_stage_done("load_optimizer_stages", _t)

	# --- Streaming data loader ---
	def _streaming_skip_channels(needed_channels: set[str]) -> set[str]:
		optional = {"cos", "pred_dt"}
		return optional - set(needed_channels)

	def _streaming_loaded_channels(d: fit_data.FitData3D) -> set[str]:
		if not d.sparse_caches:
			return set()
		return {
			ch
			for cache in d.sparse_caches.values()
			for ch in cache.channels
		}

	def _load_streaming(needed_channels: set[str]) -> fit_data.FitData3D:
		d = fit_data.load_3d_streaming(
			path=str(data_cfg.input),
			device=device,
			sparse_prefetch_backend=data_cfg.sparse_prefetch_backend,
			skip_channels=_streaming_skip_channels(needed_channels),
		)
		Z, Y, X = d.size
		# Volume extent covers the full zarr volume
		sx, sy, sz = d.spacing
		volume_extent = (
			d.origin_fullres[0],
			d.origin_fullres[1],
			d.origin_fullres[2],
			d.origin_fullres[0] + (X - 1) * sx,
			d.origin_fullres[1] + (Y - 1) * sy,
			d.origin_fullres[2] + (Z - 1) * sz,
		)
		mdl.params = dataclasses.replace(mdl.params, volume_extent=volume_extent)
		if corr_points_3d is not None:
			d = dataclasses.replace(d, corr_points=corr_points_3d)
		if data_cfg.winding_volume is not None:
			wv_t, wv_min, wv_max = fit_data.load_winding_volume(
				path=data_cfg.winding_volume, device=device,
				crop=None, downscale=scaledown)
			d = dataclasses.replace(d, winding_volume=wv_t,
						winding_min=wv_min, winding_max=wv_max)
		return d

	def _ensure_data(data: fit_data.FitData3D | None, needed_channels: set[str]) -> fit_data.FitData3D:
		if data is None:
			return _load_streaming(needed_channels)
		loaded = _streaming_loaded_channels(data)
		required = {"grad_mag", "nx", "ny"} | set(needed_channels)
		if not required.issubset(loaded) or (loaded & {"cos", "pred_dt"}) != (required & {"cos", "pred_dt"}):
			d = _load_streaming(needed_channels)
			if data.corr_points is not None:
				d = dataclasses.replace(d, corr_points=data.corr_points)
			if data.winding_volume is not None:
				d = dataclasses.replace(
					d,
					winding_volume=data.winding_volume,
					winding_min=data.winding_min,
					winding_max=data.winding_max,
				)
			return d
		# Streaming covers full volume — no border checks needed
		return data

	_t = _stage_start("load_data")
	data = _ensure_data(None, set())
	_stage_done("load_data", _t)

	if getattr(mdl, "cylinder_enabled", False) and hasattr(mdl, "prepare_umbilicus_tube_init"):
		_apply_cylinder_prepare_model_step(mdl, _first_cylinder_stage_model_step(stages))
		_t = _stage_start("prepare_umbilicus_tube_init")
		mdl.prepare_umbilicus_tube_init(data)
		_stage_done("prepare_umbilicus_tube_init", _t)

	# Print loaded data summary
	Z, Y, X = data.size
	if data.sparse_caches:
		_cache_table_bytes = sum(c.chunk_table.nbytes for c in data.sparse_caches.values())
		print(f"[fit] data (streaming): vol_size=({Z},{Y},{X}) origin={data.origin_fullres} "
			  f"spacing={data.spacing} groups={list(data.sparse_caches.keys())} "
			  f"table_mem={_cache_table_bytes / 2**20:.1f} MiB", flush=True)
	else:
		_data_bytes = sum(t.nbytes for t in [data.cos, data.grad_mag, data.nx, data.ny] if t is not None)
		if data.pred_dt is not None:
			_data_bytes += data.pred_dt.nbytes
		if data.winding_volume is not None:
			_data_bytes += data.winding_volume.nbytes
		print(f"[fit] data: size=({Z},{Y},{X}) origin={data.origin_fullres} spacing={data.spacing} "
			  f"pred_dt={data.pred_dt is not None} winding_volume={data.winding_volume is not None} "
			  f"corr_points={data.corr_points is not None} "
			  f"mem={_data_bytes / 2**30:.2f} GiB", flush=True)

	# Print initial mesh stats
	_t = _stage_start("initial_mesh_stats")
	with torch.no_grad():
		xyz = mdl._grid_xyz()
		mn = xyz.amin(dim=(0, 1, 2)).cpu().numpy().tolist()
		mx = xyz.amax(dim=(0, 1, 2)).cpu().numpy().tolist()
		mean = xyz.mean(dim=(0, 1, 2)).cpu().numpy().tolist()
		print(f"initial mesh: mean={[round(v, 1) for v in mean]} "
			  f"min={[round(v, 1) for v in mn]} max={[round(v, 1) for v in mx]}")
	_stage_done("initial_mesh_stats", _t)

	tifxyz_flow_gate_channels = bool(getattr(args, "tifxyz_flow_gate_channels", False))
	last_flow_gate_channels_payload: dict | None = None

	def _save_model(
		path: str,
		*,
		apply_corr_point_roi_mask: bool = True,
		flow_gate_channels_payload: dict | None = None,
	) -> None:
		st = dict(mdl.state_dict())
		# Store flat mesh instead of pyramid levels
		ms_keys = [k for k in st if k.startswith("mesh_ms.")]
		for k in ms_keys:
			del st[k]
		st.pop("cyl_params", None)
		if getattr(mdl, "cylinder_enabled", False) and getattr(mdl, "cyl_shell_mode", False):
			st.pop("conn_offsets", None)
			st.pop("amp", None)
			st.pop("bias", None)
		elif getattr(mdl, "cylinder_enabled", False) and "conn_offsets" in st:
			st["conn_offsets"] = torch.zeros_like(st["conn_offsets"])
		with torch.no_grad():
			mesh_flat = mdl.mesh_flat_for_save(data=data)
		params = asdict(mdl.params)
		if lasagna_base_shape_zyx is not None:
			params["lasagna_base_shape_zyx"] = [int(v) for v in lasagna_base_shape_zyx]
		st["_model_params_"] = params
		st["_fit_config_"] = fit_config
		corr_results = opt_loss_corr.get_last_results()
		if corr_results is not None:
			st["_corr_points_results_"] = corr_results
		if corr_point_roi_init is not None:
			payload = copy.deepcopy(corr_point_roi_init.payload)
			payload["output_radius_grid_points"] = int(data_cfg.corr_point_roi_output_radius)
			if apply_corr_point_roi_mask:
				if int(mesh_flat.shape[1]) != 1:
					raise ValueError(f"corr-point-roi requires final mesh depth 1, got {int(mesh_flat.shape[1])}")
				mask, mask_debug = _corr_point_roi_mask_from_results(
					corr_results,
					shape=(int(mesh_flat.shape[2]), int(mesh_flat.shape[3])),
					radius=int(data_cfg.corr_point_roi_output_radius),
					device=mesh_flat.device,
				)
				sentinel = torch.full_like(mesh_flat, -1.0)
				mesh_flat = torch.where(mask.view(1, 1, int(mesh_flat.shape[2]), int(mesh_flat.shape[3])), mesh_flat, sentinel)
				payload["output_mask"] = mask_debug
				print(
					f"[fit] corr-point-roi output mask: usable={mask_debug['usable_point_count']} "
					f"seed_vertices={mask_debug['seed_vertex_count']} "
					f"dilated_vertices={mask_debug['dilated_vertex_count']} "
					f"radius={payload['output_radius_grid_points']}",
					flush=True,
				)
			st["_corr_point_roi_"] = payload
		st["mesh_flat"] = mesh_flat
		if approval_inpaint_output_mask is not None:
			st["_approval_inpaint_output_mask_"] = copy.deepcopy(approval_inpaint_output_mask)
			print(
				"[fit] saving approval-inpaint output mask "
				f"collections={approval_inpaint_output_mask.get('corr_collection_ids', [])} "
				f"dilate={approval_inpaint_output_mask.get('dilation_radius')} "
				f"corr_results_saved={corr_results is not None}",
				flush=True,
			)
			if corr_results is None:
				print(
					"[fit] WARNING: approval-inpaint output mask was requested, but no "
					"corr point results were produced; fit2tifxyz cannot project the mask",
					flush=True,
				)
		if flow_gate_channels_payload is not None:
			st["_flow_gate_channels_"] = copy.deepcopy(flow_gate_channels_payload)
		# Store winding volume auto-offset if computed
		from opt_loss_winding_volume import _winding_offset, _winding_direction
		if _winding_offset is not None:
			st["_winding_offset_"] = _winding_offset
			st["_winding_direction_"] = _winding_direction
		torch.save(st, path)

	def _snapshot(*, stage: str, step: int, loss: float, data, res=None) -> None:
		nonlocal last_flow_gate_channels_payload
		if tifxyz_flow_gate_channels and res is not None:
			payload = opt_loss_pred_dt.flow_gate_last_channels()
			if payload is not None:
				last_flow_gate_channels_payload = payload
		if _out_dir is not None:
			out = Path(_out_dir)
			out.mkdir(parents=True, exist_ok=True)
			snaps = out / "model_snapshots"
			snaps.mkdir(parents=True, exist_ok=True)
			_save_model(str(snaps / f"model_{stage}_{step:06d}.pt"), apply_corr_point_roi_mask=False)

	def _progress(*, step: int, total: int, loss: float, **_kw: object) -> None:
		if progress_enabled:
			print(f"PROGRESS {step} {total} {loss:.6f}", flush=True)

	opt_loss_dir.set_mask_zero_normals(opt_cfg.normal_mask_zero)

	# Run optimization
	_t = _stage_start("prepare_optimization")
	config_seed = tuple(float(v) for v in data_cfg.seed) if data_cfg.seed is not None else None
	seed_xyz = _optimization_seed_xyz(model_init=model_init, config_seed=config_seed, mdl=mdl)
	if model_init == "ext":
		print(f"[fit] tifxyz seed: ({seed_xyz[0]:.0f}, {seed_xyz[1]:.0f}, {seed_xyz[2]:.0f})",
			  flush=True)
	elif model_init == "model":
		print(f"[fit] checkpoint seed (grid center): ({seed_xyz[0]:.0f}, {seed_xyz[1]:.0f}, {seed_xyz[2]:.0f})",
			  flush=True)
	_stage_done("prepare_optimization", _t)
	_t = _stage_start("optimizer")
	init_grow_runtime = asdict(init_grow_cfg) if init_grow_cfg.enabled else None
	if init_grow_runtime is not None:
		init_grow_runtime["target_depth"] = final_model_depth
	optimizer.optimize(
		model=mdl,
		data=data,
		stages=stages,
		snapshot_interval=opt_cfg.snapshot_interval,
		snapshot_fn=_snapshot,
		progress_fn=_progress,
		ensure_data_fn=_ensure_data,
		seed_xyz=seed_xyz,
		out_dir=_out_dir,
		capture_flow_gate_channels=tifxyz_flow_gate_channels,
		self_map_init=self_map_init,
		self_map_model_w_wraps=self_map_model_w_wraps,
		init_grow=init_grow_runtime,
	)
	_stage_done("optimizer", _t)

	if device.type == "cuda":
		peak_gb = torch.cuda.max_memory_allocated(device) / 2**30
		print(f"[fit] peak GPU memory: {peak_gb:.2f} GiB", flush=True)

	# Save final model
	if model_cfg.model_output is not None:
		_t = _stage_start("save_model_output")
		_save_model(str(model_cfg.model_output), flow_gate_channels_payload=last_flow_gate_channels_payload)
		print(f"[fit] saved model to {model_cfg.model_output}")
		_stage_done("save_model_output", _t)

	# Save snapshot
	if _out_dir is not None:
		_t = _stage_start("save_final_snapshot")
		out = Path(_out_dir)
		out.mkdir(parents=True, exist_ok=True)
		_save_model(str(out / "model_final.pt"), flow_gate_channels_payload=last_flow_gate_channels_payload)
		_stage_done("save_final_snapshot", _t)

	# Export tifxyz
	model_out = model_cfg.model_output
	if model_out is None and _out_dir is not None:
		model_out = str(Path(_out_dir) / "model_final.pt")
	if model_out is not None and _out_dir is not None:
		_t = _stage_start("export_tifxyz")
		import fit2tifxyz
		export_dir = str(Path(_out_dir) / "tifxyz")
		tifxyz_argv = ["--input", str(model_out), "--output", export_dir]
		if getattr(mdl, "cyl_shell_completed", None):
			tifxyz_argv.append("--single-segment")
		voxel_size_um = cfg.get("voxel_size_um")
		if voxel_size_um is not None:
			tifxyz_argv += ["--voxel-size-um", str(float(voxel_size_um))]
		fit2tifxyz.main(tifxyz_argv)
		_stage_done("export_tifxyz", _t)

	_stage_done("total", _t_fit_total)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
