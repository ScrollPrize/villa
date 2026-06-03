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

@dataclass(frozen=True)
class SnapSurfMapInitConfig:
	enabled: bool = False
	surface_loss: bool = False
	initial_iters: int = 1000
	update_interval: int = 100
	update_global_opt_iters: int = 50
	tracking_opt_iters: int = 1
	first_global_opt_iters: int = 50
	last_global_opt_iters: int = 50
	subdiv: int = 4
	iters: int = 1000
	seed_opt_iters: int = 100
	candidate_opt_iters: int = 10
	candidate_lr: float = 0.05
	fringe_opt_iters: int = 10
	fringe_lr: float = 0.05
	grow_opt_iters: int = 100
	global_opt_interval: int = 10
	progress_interval: int = 100
	progress_mode: str = "block"
	no_progress_iters: int = 1000
	scale_levels: int = 1
	scale_factor: int = 2
	min_scale_level: int = 0
	dense_opt: bool = False
	dense_reg_radius: int = 2
	w_dense_prior: float = 0.001
	repair_max_blocks: int = 3  # 0 means no repair-block cap; repair may consume the remaining iters budget.
	repair_opt_iters: int = 0
	repair_lr_mult: float = 0.25
	repair_w_jac_mult: float = 10.0
	lr: float = 0.05
	seed_radius: int = 1
	edge_init_radius: int = 2
	w_dist: float = 1.0
	w_vec_normal: float = 1.0
	w_surface_normal: float = 1.0
	z_lift_enabled: bool = True
	z_lift_refine_enabled: bool = False
	z_lift_norm_xy_min: float = 0.1
	w_z_lift: float = 10.0
	z_lift_huber_delta: float = math.pi / 4.0
	w_smooth: float = 0.05
	w_bend: float = 0.01
	w_jac: float = 1.0
	w_metric_smooth: float = 0.05
	w_area_smooth: float = 0.02
	angle_dist_mult: float = 9.0
	max_sample_distance: float = 1000.0
	max_sample_angle_deg: float = 45.0
	sample_angle_step_fraction: float = 0.1
	max_step_neighbor_ratio: float = 10.0
	ext_mesh_health_filter: bool = True
	ext_mesh_health_max_edge_ratio: float = 2.0
	ext_mesh_health_max_area_ratio: float = 3.0
	ext_mesh_health_min_area_ratio: float = 0.75
	ext_mesh_health_max_aspect_ratio: float = 2.0
	ext_mesh_health_max_diag_ratio: float = 2.0
	ext_mesh_health_min_triangle_normal_dot: float = 0.25
	ext_mesh_health_min_normal_dot: float = 0.0
	ext_mesh_health_reject_radius: int = 4
	jac_margin: float = 0.05
	compile_objective: bool = False
	compile_objective_mode: str | None = None
	fixture_export_dir: str | None = None
	fixture_export_once: bool = True
	fixture_export_objs: bool = True

@dataclass(frozen=True)
class SnapSurfConfig:
	init_distance: float = 50.0
	point_distance: float = 25.0
	grid_error: float = 0.75
	affine_radius: int = 2
	search_ring: int = 1
	seed_radius: int = 4
	map_inlier_distance: float = 8.0
	inlier_normal_distance_ratio: float = 1.5
	inlier_normal_distance_floor: float = 10.0
	ray_residual: float = 1.0e-2
	brute_interval: int = 10
	brute_boundary_radius: int = 10
	brute_pair_chunk_limit: int = 8_000_000
	huber_delta: float = 5.0
	distance_scale: float = 1.0
	w_to_ext: float = 1.0
	orientation: str = "auto"
	debug_obj_dir: str | None = None
	debug_obj_interval: int = 1
	map_init: SnapSurfMapInitConfig = field(default_factory=SnapSurfMapInitConfig)

def _parse_map_init_config(raw: object) -> SnapSurfMapInitConfig:
	defaults = SnapSurfMapInitConfig()
	if raw is None:
		return defaults
	if isinstance(raw, SnapSurfMapInitConfig):
		cfg = raw
	elif isinstance(raw, dict):
		raw_cfg = dict(raw)
		for alias in ("minscale", "min_scale"):
			if alias in raw_cfg:
				if "min_scale_level" in raw_cfg:
					raise ValueError("snap_surf args.map_init: use only one of min_scale_level, min_scale, minscale")
				raw_cfg["min_scale_level"] = raw_cfg.pop(alias)
		if "map_z_lift" in raw_cfg:
			raise ValueError("snap_surf args.map_init: use map_turn instead of map_z_lift")
		if "w_z_lift" in raw_cfg:
			raise ValueError("snap_surf args.map_init: use map_turn instead of w_z_lift")
		if "map_turn" in raw_cfg:
			raw_cfg["w_z_lift"] = raw_cfg.pop("map_turn")
		bad = sorted(set(raw_cfg.keys()) - set(SnapSurfMapInitConfig.__dataclass_fields__.keys()))
		if bad:
			raise ValueError(f"snap_surf args.map_init: unknown key(s): {bad}")
		cfg = SnapSurfMapInitConfig(
			enabled=bool(raw_cfg.get("enabled", defaults.enabled)),
			surface_loss=bool(raw_cfg.get("surface_loss", defaults.surface_loss)),
			initial_iters=max(0, int(raw_cfg.get("initial_iters", defaults.initial_iters))),
			update_interval=max(1, int(raw_cfg.get("update_interval", defaults.update_interval))),
			update_global_opt_iters=max(0, int(raw_cfg.get("update_global_opt_iters", defaults.update_global_opt_iters))),
			tracking_opt_iters=max(0, int(raw_cfg.get("tracking_opt_iters", defaults.tracking_opt_iters))),
			first_global_opt_iters=max(0, int(raw_cfg.get("first_global_opt_iters", defaults.first_global_opt_iters))),
			last_global_opt_iters=max(0, int(raw_cfg.get("last_global_opt_iters", defaults.last_global_opt_iters))),
			subdiv=max(1, int(raw_cfg.get("subdiv", defaults.subdiv))),
			iters=max(0, int(raw_cfg.get("iters", defaults.iters))),
			seed_opt_iters=max(0, int(raw_cfg.get("seed_opt_iters", defaults.seed_opt_iters))),
			candidate_opt_iters=max(0, int(raw_cfg.get("candidate_opt_iters", defaults.candidate_opt_iters))),
			candidate_lr=float(raw_cfg.get("candidate_lr", defaults.candidate_lr)),
			fringe_opt_iters=max(0, int(raw_cfg.get("fringe_opt_iters", defaults.fringe_opt_iters))),
			fringe_lr=float(raw_cfg.get("fringe_lr", defaults.fringe_lr)),
			grow_opt_iters=max(0, int(raw_cfg.get("grow_opt_iters", defaults.grow_opt_iters))),
			global_opt_interval=max(1, int(raw_cfg.get("global_opt_interval", defaults.global_opt_interval))),
			progress_interval=max(100, int(raw_cfg.get("progress_interval", defaults.progress_interval))),
			progress_mode=str(raw_cfg.get("progress_mode", defaults.progress_mode)).lower(),
			no_progress_iters=max(0, int(raw_cfg.get("no_progress_iters", defaults.no_progress_iters))),
			scale_levels=max(1, int(raw_cfg.get("scale_levels", defaults.scale_levels))),
			scale_factor=max(1, int(raw_cfg.get("scale_factor", defaults.scale_factor))),
			min_scale_level=max(0, int(raw_cfg.get("min_scale_level", defaults.min_scale_level))),
			dense_opt=bool(raw_cfg.get("dense_opt", defaults.dense_opt)),
			dense_reg_radius=max(0, int(raw_cfg.get("dense_reg_radius", defaults.dense_reg_radius))),
			w_dense_prior=float(raw_cfg.get("w_dense_prior", defaults.w_dense_prior)),
			repair_max_blocks=max(0, int(raw_cfg.get("repair_max_blocks", defaults.repair_max_blocks))),
			repair_opt_iters=max(0, int(raw_cfg.get("repair_opt_iters", defaults.repair_opt_iters))),
			repair_lr_mult=float(raw_cfg.get("repair_lr_mult", defaults.repair_lr_mult)),
			repair_w_jac_mult=float(raw_cfg.get("repair_w_jac_mult", defaults.repair_w_jac_mult)),
			lr=float(raw_cfg.get("lr", defaults.lr)),
			seed_radius=max(0, int(raw_cfg.get("seed_radius", defaults.seed_radius))),
			edge_init_radius=max(1, int(raw_cfg.get("edge_init_radius", defaults.edge_init_radius))),
			w_dist=float(raw_cfg.get("w_dist", defaults.w_dist)),
			w_vec_normal=float(raw_cfg.get("w_vec_normal", defaults.w_vec_normal)),
			w_surface_normal=float(raw_cfg.get("w_surface_normal", defaults.w_surface_normal)),
			z_lift_enabled=bool(raw_cfg.get("z_lift_enabled", defaults.z_lift_enabled)),
			z_lift_refine_enabled=bool(raw_cfg.get("z_lift_refine_enabled", defaults.z_lift_refine_enabled)),
			z_lift_norm_xy_min=float(raw_cfg.get("z_lift_norm_xy_min", defaults.z_lift_norm_xy_min)),
			w_z_lift=float(raw_cfg.get("w_z_lift", defaults.w_z_lift)),
			z_lift_huber_delta=float(raw_cfg.get("z_lift_huber_delta", defaults.z_lift_huber_delta)),
			w_smooth=float(raw_cfg.get("w_smooth", defaults.w_smooth)),
			w_bend=float(raw_cfg.get("w_bend", defaults.w_bend)),
			w_jac=float(raw_cfg.get("w_jac", defaults.w_jac)),
			w_metric_smooth=float(raw_cfg.get("w_metric_smooth", defaults.w_metric_smooth)),
			w_area_smooth=float(raw_cfg.get("w_area_smooth", defaults.w_area_smooth)),
			angle_dist_mult=float(raw_cfg.get("angle_dist_mult", defaults.angle_dist_mult)),
			max_sample_distance=float(raw_cfg.get("max_sample_distance", defaults.max_sample_distance)),
			max_sample_angle_deg=float(raw_cfg.get("max_sample_angle_deg", defaults.max_sample_angle_deg)),
			sample_angle_step_fraction=float(raw_cfg.get("sample_angle_step_fraction", defaults.sample_angle_step_fraction)),
			max_step_neighbor_ratio=float(raw_cfg.get("max_step_neighbor_ratio", defaults.max_step_neighbor_ratio)),
			ext_mesh_health_filter=bool(raw_cfg.get("ext_mesh_health_filter", defaults.ext_mesh_health_filter)),
			ext_mesh_health_max_edge_ratio=float(raw_cfg.get("ext_mesh_health_max_edge_ratio", defaults.ext_mesh_health_max_edge_ratio)),
			ext_mesh_health_max_area_ratio=float(raw_cfg.get("ext_mesh_health_max_area_ratio", defaults.ext_mesh_health_max_area_ratio)),
			ext_mesh_health_min_area_ratio=float(raw_cfg.get("ext_mesh_health_min_area_ratio", defaults.ext_mesh_health_min_area_ratio)),
			ext_mesh_health_max_aspect_ratio=float(raw_cfg.get("ext_mesh_health_max_aspect_ratio", defaults.ext_mesh_health_max_aspect_ratio)),
			ext_mesh_health_max_diag_ratio=float(raw_cfg.get("ext_mesh_health_max_diag_ratio", defaults.ext_mesh_health_max_diag_ratio)),
			ext_mesh_health_min_triangle_normal_dot=float(raw_cfg.get("ext_mesh_health_min_triangle_normal_dot", defaults.ext_mesh_health_min_triangle_normal_dot)),
			ext_mesh_health_min_normal_dot=float(raw_cfg.get("ext_mesh_health_min_normal_dot", defaults.ext_mesh_health_min_normal_dot)),
			ext_mesh_health_reject_radius=max(0, int(raw_cfg.get("ext_mesh_health_reject_radius", defaults.ext_mesh_health_reject_radius))),
			jac_margin=float(raw_cfg.get("jac_margin", defaults.jac_margin)),
			compile_objective=bool(raw_cfg.get("compile_objective", defaults.compile_objective)),
			compile_objective_mode=(
				None
				if raw_cfg.get("compile_objective_mode", defaults.compile_objective_mode) in {None, "", "default"}
				else str(raw_cfg.get("compile_objective_mode", defaults.compile_objective_mode))
			),
			fixture_export_dir=(
				None
				if raw_cfg.get("fixture_export_dir", defaults.fixture_export_dir) in {None, ""}
				else str(raw_cfg.get("fixture_export_dir", defaults.fixture_export_dir))
			),
			fixture_export_once=bool(raw_cfg.get("fixture_export_once", defaults.fixture_export_once)),
			fixture_export_objs=bool(raw_cfg.get("fixture_export_objs", defaults.fixture_export_objs)),
		)
	else:
		raise ValueError("snap_surf args.map_init must be an object or null")
	if cfg.lr <= 0.0:
		raise ValueError("snap_surf args.map_init.lr must be > 0")
	if cfg.candidate_lr <= 0.0:
		raise ValueError("snap_surf args.map_init.candidate_lr must be > 0")
	if cfg.fringe_lr <= 0.0:
		raise ValueError("snap_surf args.map_init.fringe_lr must be > 0")
	if cfg.repair_lr_mult <= 0.0:
		raise ValueError("snap_surf args.map_init.repair_lr_mult must be > 0")
	if cfg.repair_w_jac_mult < 0.0:
		raise ValueError("snap_surf args.map_init.repair_w_jac_mult must be >= 0")
	for name in (
		"w_dist", "w_vec_normal", "w_surface_normal", "w_smooth", "w_bend", "w_jac",
		"w_metric_smooth", "w_area_smooth", "w_dense_prior", "angle_dist_mult",
		"w_z_lift",
	):
		if float(getattr(cfg, name)) < 0.0:
			raise ValueError(f"snap_surf args.map_init.{name} must be >= 0")
	if cfg.z_lift_norm_xy_min < 0.0:
		raise ValueError("snap_surf args.map_init.z_lift_norm_xy_min must be >= 0")
	if cfg.z_lift_huber_delta <= 0.0:
		raise ValueError("snap_surf args.map_init.z_lift_huber_delta must be > 0")
	if cfg.max_sample_distance < 0.0:
		raise ValueError("snap_surf args.map_init.max_sample_distance must be >= 0")
	if cfg.max_sample_angle_deg < 0.0 or cfg.max_sample_angle_deg > 180.0:
		raise ValueError("snap_surf args.map_init.max_sample_angle_deg must be in [0, 180]")
	if cfg.sample_angle_step_fraction < 0.0:
		raise ValueError("snap_surf args.map_init.sample_angle_step_fraction must be >= 0")
	if cfg.max_step_neighbor_ratio < 0.0:
		raise ValueError("snap_surf args.map_init.max_step_neighbor_ratio must be >= 0")
	if cfg.ext_mesh_health_max_edge_ratio < 0.0:
		raise ValueError("snap_surf args.map_init.ext_mesh_health_max_edge_ratio must be >= 0")
	if cfg.ext_mesh_health_max_area_ratio < 0.0:
		raise ValueError("snap_surf args.map_init.ext_mesh_health_max_area_ratio must be >= 0")
	if cfg.ext_mesh_health_min_area_ratio < 0.0:
		raise ValueError("snap_surf args.map_init.ext_mesh_health_min_area_ratio must be >= 0")
	if cfg.ext_mesh_health_max_aspect_ratio < 0.0:
		raise ValueError("snap_surf args.map_init.ext_mesh_health_max_aspect_ratio must be >= 0")
	if cfg.ext_mesh_health_max_diag_ratio < 0.0:
		raise ValueError("snap_surf args.map_init.ext_mesh_health_max_diag_ratio must be >= 0")
	if cfg.ext_mesh_health_min_triangle_normal_dot < -1.0 or cfg.ext_mesh_health_min_triangle_normal_dot > 1.0:
		raise ValueError("snap_surf args.map_init.ext_mesh_health_min_triangle_normal_dot must be in [-1, 1]")
	if cfg.ext_mesh_health_min_normal_dot < 0.0 or cfg.ext_mesh_health_min_normal_dot > 1.0:
		raise ValueError("snap_surf args.map_init.ext_mesh_health_min_normal_dot must be in [0, 1]")
	if cfg.jac_margin < 0.0:
		raise ValueError("snap_surf args.map_init.jac_margin must be >= 0")
	if cfg.progress_mode not in ("block", "periodic", "both", "none"):
		raise ValueError("snap_surf args.map_init.progress_mode must be one of block, periodic, both, none")
	if cfg.compile_objective_mode not in (None, "reduce-overhead", "max-autotune"):
		raise ValueError("snap_surf args.map_init.compile_objective_mode must be one of default, reduce-overhead, max-autotune")
	if int(cfg.scale_levels) > 1 and int(cfg.scale_factor) != 2:
		raise ValueError("snap_surf args.map_init.scale_factor must be 2 when scale_levels > 1")
	return cfg

def _safe_frac(n: int | float, d: int | float) -> float:
	den = float(d)
	if den <= 0.0:
		return 0.0
	return float(n) / den

def _snap_surf_log(message: str) -> None:
	print(f"[snap_surf] {message}", flush=True)

def _map_init_log(message: str) -> None:
	print(f"[snap_surf.map_init] {message}", flush=True)

__all__ = [name for name in globals() if not name.startswith('__')]
