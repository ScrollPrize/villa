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

from .config import SnapSurfConfig, _map_init_log, _parse_map_init_config, _safe_frac, _snap_surf_log
from .state import _SurfaceState
from .legacy import _closest_external_seed_surface, _direction_loss_model_ray_to_ext, _rebuild_model_to_ext_rays
from .map_pyramid import _map_init_empty_stats
from .map_growth import _run_map_init_for_surface, _run_map_init_interleaved_for_surface
from .map_objective import _map_init_surface_normal_loss
from .debug_obj import (
	_debug_obj_iter_dir,
	_debug_write_map_init_objs,
	_debug_write_snap_objs,
	_surface_records_from_res,
	set_debug_step as _set_debug_obj_step,
)

_cfg = SnapSurfConfig()
_active = False
_seed_xyz: tuple[float, float, float] | None = None
_states: list[_SurfaceState] = []
_last_stats: dict[str, float] = {}
_debug_step: int | None = None
_debug_label: str | None = None
_stage_label: str | None = None
_stage_steps: int | None = None

def reset_state() -> None:
	global _states, _last_stats, _debug_step, _debug_label, _stage_label, _stage_steps
	_states = []
	_last_stats = {}
	_debug_step = None
	_debug_label = None
	_stage_label = None
	_stage_steps = None
	_set_debug_obj_step(None)

def configure_snap_surf(
	*,
	cfg: dict | None = None,
	seed_xyz: tuple[float, float, float] | None = None,
	active: bool = False,
	stage_label: str | None = None,
	stage_steps: int | None = None,
) -> None:
	"""Configure the runtime snap-surface loss state for the current stage."""
	global _cfg, _active, _seed_xyz, _stage_label, _stage_steps
	raw = dict(cfg or {})
	bad = sorted(set(raw.keys()) - set(SnapSurfConfig.__dataclass_fields__.keys()))
	if bad:
		raise ValueError(f"snap_surf args: unknown key(s): {bad}")
	default_cfg = SnapSurfConfig()
	map_init_cfg = _parse_map_init_config(raw.get("map_init", default_cfg.map_init))
	debug_obj_raw = raw.get("debug_obj_dir", default_cfg.debug_obj_dir)
	if isinstance(debug_obj_raw, bool):
		debug_obj_dir = "snap_surf_objs" if debug_obj_raw else None
	else:
		debug_obj_dir = None if debug_obj_raw in {None, ""} else str(debug_obj_raw)
	_cfg = SnapSurfConfig(
		init_distance=float(raw.get("init_distance", default_cfg.init_distance)),
		point_distance=float(raw.get("point_distance", default_cfg.point_distance)),
		grid_error=float(raw.get("grid_error", default_cfg.grid_error)),
		affine_radius=max(1, int(raw.get("affine_radius", default_cfg.affine_radius))),
		search_ring=max(0, int(raw.get("search_ring", default_cfg.search_ring))),
		seed_radius=int(raw.get("seed_radius", default_cfg.seed_radius)),
		map_inlier_distance=float(raw.get("map_inlier_distance", default_cfg.map_inlier_distance)),
		inlier_normal_distance_ratio=float(raw.get("inlier_normal_distance_ratio", default_cfg.inlier_normal_distance_ratio)),
		inlier_normal_distance_floor=float(raw.get("inlier_normal_distance_floor", default_cfg.inlier_normal_distance_floor)),
		ray_residual=float(raw.get("ray_residual", default_cfg.ray_residual)),
		brute_interval=max(1, int(raw.get("brute_interval", default_cfg.brute_interval))),
		brute_boundary_radius=max(0, int(raw.get("brute_boundary_radius", default_cfg.brute_boundary_radius))),
		brute_pair_chunk_limit=max(1, int(raw.get("brute_pair_chunk_limit", default_cfg.brute_pair_chunk_limit))),
		huber_delta=float(raw.get("huber_delta", default_cfg.huber_delta)),
		distance_scale=max(1.0e-8, float(raw.get("distance_scale", default_cfg.distance_scale))),
		w_to_ext=float(raw.get("w_to_ext", default_cfg.w_to_ext)),
		orientation=str(raw.get("orientation", default_cfg.orientation)).strip().lower(),
		debug_obj_dir=debug_obj_dir,
		debug_obj_interval=max(1, int(raw.get("debug_obj_interval", default_cfg.debug_obj_interval))),
		map_init=map_init_cfg,
	)
	if _cfg.init_distance < 0.0:
		raise ValueError("snap_surf args.init_distance must be >= 0")
	if _cfg.point_distance < 0.0:
		raise ValueError("snap_surf args.point_distance must be >= 0")
	if _cfg.grid_error < 0.0:
		raise ValueError("snap_surf args.grid_error must be >= 0")
	if _cfg.seed_radius < 0:
		raise ValueError("snap_surf args.seed_radius must be >= 0")
	if _cfg.map_inlier_distance <= 0.0:
		raise ValueError("snap_surf args.map_inlier_distance must be > 0")
	if _cfg.inlier_normal_distance_ratio < 1.0:
		raise ValueError("snap_surf args.inlier_normal_distance_ratio must be >= 1")
	if _cfg.inlier_normal_distance_floor < 0.0:
		raise ValueError("snap_surf args.inlier_normal_distance_floor must be >= 0")
	if _cfg.ray_residual < 0.0:
		raise ValueError("snap_surf args.ray_residual must be >= 0")
	if _cfg.huber_delta <= 0.0:
		raise ValueError("snap_surf args.huber_delta must be > 0")
	if _cfg.w_to_ext < 0.0:
		raise ValueError("snap_surf direction weight must be >= 0")
	if _cfg.orientation not in {"auto", "identity", "none"}:
		raise ValueError("snap_surf args.orientation must be 'auto', 'identity', or 'none'")
	if active and seed_xyz is None:
		raise ValueError("snap_surf requires args.seed")
	_active = bool(active)
	_seed_xyz = None if seed_xyz is None else tuple(float(v) for v in seed_xyz)
	_stage_label = None if stage_label is None else str(stage_label)
	_stage_steps = None if stage_steps is None else max(0, int(stage_steps))
	_snap_surf_log(
		"configured "
		f"stage={_stage_label!r} "
		f"active={int(_active)} "
		f"map_init={int(_cfg.map_init.enabled)} "
		f"debug_obj_dir={_cfg.debug_obj_dir!r} "
		f"debug_obj_interval={_cfg.debug_obj_interval} "
		f"stage_steps={_stage_steps} "
		f"seed={_seed_xyz}"
	)
	if _active and _cfg.map_init.enabled:
		_map_init_log(
			"enabled "
			f"surface_loss={int(_cfg.map_init.surface_loss)} "
			f"initial_iters={_cfg.map_init.initial_iters} "
			f"update_interval={_cfg.map_init.update_interval} "
			f"update_global_opt_iters={_cfg.map_init.update_global_opt_iters} "
			f"tracking_opt_iters={_cfg.map_init.tracking_opt_iters} "
			f"first_global_opt_iters={_cfg.map_init.first_global_opt_iters} "
			f"last_global_opt_iters={_cfg.map_init.last_global_opt_iters} "
			f"subdiv={_cfg.map_init.subdiv} "
			f"iters={_cfg.map_init.iters} "
			f"seed_opt_iters={_cfg.map_init.seed_opt_iters} "
			f"candidate_opt_iters={_cfg.map_init.candidate_opt_iters} "
			f"candidate_lr={_cfg.map_init.candidate_lr} "
			f"fringe_opt_iters={_cfg.map_init.fringe_opt_iters} "
			f"fringe_lr={_cfg.map_init.fringe_lr} "
			f"grow_opt_iters={_cfg.map_init.grow_opt_iters} "
			f"global_opt_interval={_cfg.map_init.global_opt_interval} "
			f"progress_interval={_cfg.map_init.progress_interval} "
			f"progress_mode={_cfg.map_init.progress_mode!r} "
			f"no_progress_iters={_cfg.map_init.no_progress_iters} "
			f"scale_levels={_cfg.map_init.scale_levels} "
			f"scale_factor={_cfg.map_init.scale_factor} "
			f"min_scale_level={_cfg.map_init.min_scale_level} "
			f"dense_opt={int(_cfg.map_init.dense_opt)} "
			f"dense_reg_radius={_cfg.map_init.dense_reg_radius} "
			f"w_dense_prior={_cfg.map_init.w_dense_prior} "
			f"w_metric_smooth={_cfg.map_init.w_metric_smooth} "
			f"w_area_smooth={_cfg.map_init.w_area_smooth} "
			f"max_sample_distance={_cfg.map_init.max_sample_distance} "
			f"max_sample_angle_deg={_cfg.map_init.max_sample_angle_deg} "
			f"sample_angle_step_fraction={_cfg.map_init.sample_angle_step_fraction} "
			f"max_step_neighbor_ratio={_cfg.map_init.max_step_neighbor_ratio} "
			f"repair_max_blocks={_cfg.map_init.repair_max_blocks} "
			f"repair_opt_iters={_cfg.map_init.repair_opt_iters} "
			f"repair_lr_mult={_cfg.map_init.repair_lr_mult} "
			f"repair_w_jac_mult={_cfg.map_init.repair_w_jac_mult} "
			f"lr={_cfg.map_init.lr} "
			f"seed_radius={_cfg.map_init.seed_radius} "
			f"edge_init_radius={_cfg.map_init.edge_init_radius} "
			f"jac_margin={_cfg.map_init.jac_margin} "
			f"fixture_export_dir={_cfg.map_init.fixture_export_dir!r} "
			f"fixture_export_once={int(_cfg.map_init.fixture_export_once)} "
			f"fixture_export_objs={int(_cfg.map_init.fixture_export_objs)}"
		)
		for state in _states:
			state.reset_map_init()

def last_stats() -> dict[str, float]:
	return dict(_last_stats)

def set_debug_step(step: int | None, *, label: str | None = None) -> None:
	global _debug_step, _debug_label
	_debug_step = None if step is None else int(step)
	_debug_label = None if label is None else str(label)
	_set_debug_obj_step(_debug_step, label=_debug_label)

def snap_surf_loss(*, res: fit_model.FitResult3D) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
	"""Stateful external-surface snapping loss with explicit grown correspondences."""
	global _last_stats
	cfg = _cfg
	device = res.xyz_lr.device
	dtype = res.xyz_lr.dtype
	if not _active:
		z = res.xyz_lr.sum() * 0.0
		return z, (z.reshape(1, 1, 1, 1),), (z.reshape(1, 1, 1, 1),)
	if _seed_xyz is None:
		raise RuntimeError("snap_surf requires args.seed")
	records = _surface_records_from_res(res)
	if not records:
		raise RuntimeError("snap_surf requires at least one external_surfaces entry")

	if res.normals is None:
		raise RuntimeError("snap_surf requires model normals")
	model_xyz_det = res.xyz_lr.detach()
	model_normals_det = res.normals.detach()
	model_valid = torch.isfinite(model_xyz_det).all(dim=-1)

	while len(_states) < len(records):
		_states.append(_SurfaceState())

	if cfg.map_init.enabled:
		total = res.xyz_lr.sum() * 0.0
		total_weight = 0.0
		lm_accum = torch.zeros(res.xyz_lr.shape[:3], device=device, dtype=dtype).unsqueeze(1)
		mask_accum = torch.zeros_like(lm_accum)
		stats = _map_init_empty_stats()
		stats["snaps_sdist"] = float("inf")
		stats["snaps_sext"] = float("inf")
		avg_keys = {
			"snaps_map_loss", "snaps_map_dist", "snaps_map_vec", "snaps_map_norm",
			"snaps_map_smooth", "snaps_map_bend", "snaps_map_jac",
			"snaps_map_jmin", "snaps_map_prior", "snaps_map_reg",
			"snaps_map_jbad", "snaps_map_jbadf",
			"snaps_map_samples", "snaps_map_uvbad", "snaps_map_model_bad", "snaps_map_step_bad",
			"snaps_map_nsign", "snaps_map_scales",
			"snaps_map_surf", "snaps_map_surf_n", "snaps_map_surf_avg",
			"snaps_map_surf_abs", "snaps_map_surf_max",
		}
		for k in avg_keys:
			stats[k] = 0.0
		for si, (ext_xyz, ext_valid, ext_normals, ext_quad_valid) in enumerate(records):
			state = _states[si]
			ext_xyz = ext_xyz.to(device=device, dtype=dtype).detach()
			ext_valid = ext_valid.to(device=device).bool()
			ext_normals = ext_normals.to(device=device, dtype=dtype).detach()
			ext_quad_valid = ext_quad_valid.to(device=device).bool()
			ext_valid = ext_valid & torch.isfinite(ext_xyz).all(dim=-1)
			if int(ext_valid.shape[0]) > 1 and int(ext_valid.shape[1]) > 1:
				corner_quad_valid = (
					ext_valid[:-1, :-1] &
					ext_valid[1:, :-1] &
					ext_valid[:-1, 1:] &
					ext_valid[1:, 1:]
				)
				if tuple(ext_quad_valid.shape) == tuple(corner_quad_valid.shape):
					ext_quad_valid = ext_quad_valid & corner_quad_valid
				else:
					ext_quad_valid = corner_quad_valid
			state.ensure(
				model_shape=tuple(int(v) for v in res.xyz_lr.shape[:3]),
				ext_shape=tuple(int(v) for v in ext_xyz.shape[:2]),
				device=device,
				dtype=dtype,
			)
			if cfg.map_init.surface_loss:
				map_stats = _run_map_init_interleaved_for_surface(
					state,
					model_xyz=model_xyz_det,
					model_valid=model_valid,
					model_normals=model_normals_det,
					ext_xyz=ext_xyz,
					ext_valid=ext_valid,
					ext_normals=ext_normals,
					ext_quad_valid=ext_quad_valid,
					cfg=cfg,
					seed_xyz=_seed_xyz,
					surface_index=si,
					surface_count=len(records),
					debug_step=_debug_step,
					stage_steps=_stage_steps,
				)
			else:
				map_stats = _run_map_init_for_surface(
					state,
					model_xyz=model_xyz_det,
					model_valid=model_valid,
					model_normals=model_normals_det,
					ext_xyz=ext_xyz,
					ext_valid=ext_valid,
					ext_normals=ext_normals,
					ext_quad_valid=ext_quad_valid,
					cfg=cfg,
					seed_xyz=_seed_xyz,
					surface_index=si,
					surface_count=len(records),
				)
			stats["snaps_sdist"] = min(stats["snaps_sdist"], float(map_stats.get("snaps_sdist", float("inf"))))
			stats["snaps_sext"] = min(stats["snaps_sext"], float(map_stats.get("snaps_sext", float("inf"))))
			for k, v in map_stats.items():
				if k in {"snaps_sdist", "snaps_sext"}:
					continue
				stats[k] = float(stats.get(k, 0.0)) + float(v)
			if cfg.map_init.surface_loss and state.map_init.uv is not None and state.map_init.active_quad is not None and state.map_init.model_depth is not None:
				l_map, lm_map, mask_map, loss_stats = _map_init_surface_normal_loss(
					uv_full=state.map_init.uv,
					active_quad=state.map_init.active_quad,
					ext_pos=ext_xyz,
					ext_normals=ext_normals,
					ext_valid=ext_valid,
					ext_quad_valid=ext_quad_valid,
					ext_coords=state.map_init.ext_coords,
					model_xyz=res.xyz_lr,
					model_valid=model_valid,
					model_normals=model_normals_det,
					model_depth=int(state.map_init.model_depth),
					cfg=cfg,
				)
				for k, v in loss_stats.items():
					stats[k] = float(stats.get(k, 0.0)) + float(v)
				if float(loss_stats.get("snaps_map_surf_n", 0.0)) > 0.0:
					total = total + l_map
					total_weight += 1.0
					lm_accum = lm_accum + lm_map
					mask_accum = (mask_accum + mask_map).clamp(max=1.0)
			_debug_write_map_init_objs(
				cfg=cfg,
				surface_index=si,
				surface_count=len(records),
				model_xyz=model_xyz_det,
				model_valid=model_valid,
				ext_xyz=ext_xyz,
				ext_valid=ext_valid,
				state=state,
			)
		stats["snaps_seed"] = _safe_frac(stats["snaps_seed"], len(records))
		for k in avg_keys:
			stats[k] = _safe_frac(stats.get(k, 0.0), len(records))
		if total_weight > 0.0:
			total = total / total_weight
		_last_stats = stats
		return total, (lm_accum,), (mask_accum,)

	if _debug_step == 0:
		_snap_surf_log(
			"map_init disabled for this active snap_surf stage; "
			f"stage={_stage_label!r}; using legacy correspondence snapping"
		)

	total = res.xyz_lr.new_zeros(())
	total_weight = 0.0
	lm_accum = torch.zeros(res.xyz_lr.shape[:3], device=device, dtype=dtype).unsqueeze(1)
	mask_accum = torch.zeros_like(lm_accum)
	stats = {
		"snaps_seed": 0.0,
		"snaps_sdist": float("inf"),
		"snaps_sext": float("inf"),
		"snaps_m2e": 0.0,
		"snaps_local": 0.0,
		"snaps_brute": 0.0,
		"snaps_front": 0.0,
		"snaps_brute_on": 0.0,
		"_snaps_tested": 0.0,
		"_snaps_gerr_n": 0.0,
		"_snaps_gerr_sum": 0.0,
		"_snaps_gerr_max": 0.0,
		"_snaps_res_n": 0.0,
		"_snaps_res_sum": 0.0,
		"_snaps_res_abs_sum": 0.0,
		"_snaps_res_abs_max": 0.0,
		"_snaps_toward_sum": 0.0,
	}
	total_model_possible = 0

	for si, (ext_xyz, ext_valid, ext_normals, ext_quad_valid) in enumerate(records):
		state = _states[si]
		ext_xyz = ext_xyz.to(device=device, dtype=dtype).detach()
		ext_valid = ext_valid.to(device=device).bool()
		ext_normals = ext_normals.to(device=device, dtype=dtype).detach()
		ext_quad_valid = ext_quad_valid.to(device=device).bool()
		ext_valid = ext_valid & torch.isfinite(ext_xyz).all(dim=-1)
		if int(ext_valid.shape[0]) > 1 and int(ext_valid.shape[1]) > 1:
			corner_quad_valid = (
				ext_valid[:-1, :-1] &
				ext_valid[1:, :-1] &
				ext_valid[:-1, 1:] &
				ext_valid[1:, 1:]
			)
			if tuple(ext_quad_valid.shape) == tuple(corner_quad_valid.shape):
				ext_quad_valid = ext_quad_valid & corner_quad_valid
			else:
				ext_quad_valid = corner_quad_valid
		state.ensure(
			model_shape=tuple(int(v) for v in res.xyz_lr.shape[:3]),
			ext_shape=tuple(int(v) for v in ext_xyz.shape[:2]),
			device=device,
			dtype=dtype,
		)

		with torch.no_grad():
			seed = torch.tensor(_seed_xyz, device=device, dtype=dtype)
			_ext_seed_hw, _ext_seed_point, seed_ext_dist = _closest_external_seed_surface(
				seed=seed,
				ext_xyz=ext_xyz,
				ext_valid=ext_valid,
				ext_quad_valid=ext_quad_valid,
			)
			seed_inserted, seed_dist, grow_m2e, source_possible = _rebuild_model_to_ext_rays(
				state=state,
				model_xyz_det=model_xyz_det,
				model_valid=model_valid,
				model_normals_det=model_normals_det,
				ext_xyz=ext_xyz,
				ext_valid=ext_valid,
				ext_quad_valid=ext_quad_valid,
				cfg=cfg,
				seed_xyz=_seed_xyz,
			)
			total_model_possible += int(source_possible)
			stats["snaps_sdist"] = min(stats["snaps_sdist"], float(seed_dist))
			stats["snaps_sext"] = min(stats["snaps_sext"], float(seed_ext_dist))
			if seed_inserted:
				stats["snaps_seed"] += 1.0
			for k in ("local", "brute", "front", "brute_on"):
				stats[f"snaps_{k}"] += float(grow_m2e.get(k, 0))
			stats["_snaps_tested"] += float(grow_m2e.get("tested", 0))
			stats["_snaps_gerr_n"] += float(grow_m2e.get("gerr_n", 0))
			stats["_snaps_gerr_sum"] += float(grow_m2e.get("gerr_sum", 0.0))
			stats["_snaps_gerr_max"] = max(
				stats["_snaps_gerr_max"],
				float(grow_m2e.get("gerr_max", 0.0)),
			)
			_debug_write_snap_objs(
				cfg=cfg,
				surface_index=si,
				surface_count=len(records),
				model_xyz=model_xyz_det,
				model_valid=model_valid,
				ext_xyz=ext_xyz,
				ext_valid=ext_valid,
				state=state,
			)

		l_m2e, lm_m2e, mask_m2e, n_m2e, rs_m2e = _direction_loss_model_ray_to_ext(
			state.model_to_ext,
			model_xyz=res.xyz_lr,
			model_normals=model_normals_det,
			ext_xyz=ext_xyz,
			ext_valid=ext_valid,
			cfg=cfg,
		)
		stats["_snaps_res_n"] += float(rs_m2e.get("n", 0.0))
		stats["_snaps_res_sum"] += float(rs_m2e.get("sum", 0.0))
		stats["_snaps_res_abs_sum"] += float(rs_m2e.get("abs_sum", 0.0))
		stats["_snaps_res_abs_max"] = max(stats["_snaps_res_abs_max"], float(rs_m2e.get("abs_max", 0.0)))
		stats["_snaps_toward_sum"] += float(rs_m2e.get("toward_sum", 0.0))
		stats["snaps_m2e"] += float(n_m2e)
		w_m2e = cfg.w_to_ext if n_m2e > 0 else 0.0
		if w_m2e > 0.0:
			total = total + l_m2e
			total_weight += 1.0
		lm_accum = lm_accum + lm_m2e
		mask_accum = (mask_accum + mask_m2e).clamp(max=1.0)

	if total_weight > 0.0:
		total = total / total_weight
	if not bool(torch.isfinite(total.detach()).all().cpu()):
		raise RuntimeError(f"snap_surf produced non-finite loss: stats={stats}")
	stats["snaps_m2e"] = _safe_frac(stats["snaps_m2e"], total_model_possible)
	stats["snaps_seed"] = _safe_frac(stats["snaps_seed"], len(records))
	stats["snaps_brute_on"] = _safe_frac(stats["snaps_brute_on"], len(records))
	tested = stats.pop("_snaps_tested")
	gerr_n = stats.pop("_snaps_gerr_n")
	gerr_sum = stats.pop("_snaps_gerr_sum")
	gerr_max = stats.pop("_snaps_gerr_max")
	res_n = stats.pop("_snaps_res_n")
	res_sum = stats.pop("_snaps_res_sum")
	res_abs_sum = stats.pop("_snaps_res_abs_sum")
	res_abs_max = stats.pop("_snaps_res_abs_max")
	toward_sum = stats.pop("_snaps_toward_sum")
	stats["snaps_gerr_avg"] = _safe_frac(gerr_sum, gerr_n)
	stats["snaps_gerr_max"] = float(gerr_max)
	stats["snaps_ravg"] = _safe_frac(res_sum, res_n)
	stats["snaps_rabs"] = _safe_frac(res_abs_sum, res_n)
	stats["snaps_rmax"] = float(res_abs_max)
	stats["snaps_tow"] = _safe_frac(toward_sum, res_n)
	stats["snaps_pairs_m"] = _safe_frac(tested, 1_000_000.0)
	_last_stats = stats
	return total, (lm_accum,), (mask_accum,)

__all__ = [name for name in globals() if not name.startswith('__')]
