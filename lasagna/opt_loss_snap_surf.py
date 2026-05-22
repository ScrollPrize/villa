from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
import time

import torch
import torch.nn.functional as F

import model as fit_model
import opt_loss_winding_density
import opt_loss_station


@dataclass(frozen=True)
class SnapSurfMapInitConfig:
	enabled: bool = False
	subdiv: int = 4
	iters: int = 1000
	seed_opt_iters: int = 100
	candidate_opt_iters: int = 10
	candidate_lr: float = 0.05
	fringe_opt_iters: int = 10
	fringe_lr: float = 0.05
	grow_opt_iters: int = 100
	global_opt_interval: int = 1
	progress_interval: int = 100
	progress_mode: str = "block"
	scale_levels: int = 1
	scale_factor: int = 2
	min_scale_level: int = 0
	coarse_revisit_blocks: int = 0
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
	jac_margin: float = 0.05


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
	w_to_model: float = 1.0
	orientation: str = "auto"
	debug_obj_dir: str | None = None
	debug_obj_interval: int = 1
	map_init: SnapSurfMapInitConfig = field(default_factory=SnapSurfMapInitConfig)


_cfg = SnapSurfConfig()
_active = False
_seed_xyz: tuple[float, float, float] | None = None
_states: list["_SurfaceState"] = []
_last_stats: dict[str, float] = {}
_debug_step: int | None = None
_debug_label: str | None = None
_stage_label: str | None = None


def reset_state() -> None:
	global _states, _last_stats, _debug_step, _debug_label, _stage_label
	_states = []
	_last_stats = {}
	_debug_step = None
	_debug_label = None
	_stage_label = None


def configure_snap_surf(
	*,
	cfg: dict | None = None,
	seed_xyz: tuple[float, float, float] | None = None,
	active: bool = False,
	stage_label: str | None = None,
) -> None:
	"""Configure the runtime snap-surface loss state for the current stage."""
	global _cfg, _active, _seed_xyz, _stage_label
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
		w_to_model=float(raw.get("w_to_model", default_cfg.w_to_model)),
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
	if _cfg.w_to_ext < 0.0 or _cfg.w_to_model < 0.0:
		raise ValueError("snap_surf direction weights must be >= 0")
	if _cfg.orientation not in {"auto", "identity", "none"}:
		raise ValueError("snap_surf args.orientation must be 'auto', 'identity', or 'none'")
	if active and seed_xyz is None:
		raise ValueError("snap_surf requires args.seed")
	_active = bool(active)
	_seed_xyz = None if seed_xyz is None else tuple(float(v) for v in seed_xyz)
	_stage_label = None if stage_label is None else str(stage_label)
	_snap_surf_log(
		"configured "
		f"stage={_stage_label!r} "
		f"active={int(_active)} "
		f"map_init={int(_cfg.map_init.enabled)} "
		f"debug_obj_dir={_cfg.debug_obj_dir!r} "
		f"debug_obj_interval={_cfg.debug_obj_interval} "
		f"seed={_seed_xyz}"
	)
	if _active and _cfg.map_init.enabled:
		_map_init_log(
			"enabled "
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
			f"scale_levels={_cfg.map_init.scale_levels} "
			f"scale_factor={_cfg.map_init.scale_factor} "
			f"min_scale_level={_cfg.map_init.min_scale_level} "
			f"coarse_revisit_blocks={_cfg.map_init.coarse_revisit_blocks} "
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
			f"jac_margin={_cfg.map_init.jac_margin}"
		)
		for state in _states:
			state.reset_map_init()


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
		bad = sorted(set(raw_cfg.keys()) - set(SnapSurfMapInitConfig.__dataclass_fields__.keys()))
		if bad:
			raise ValueError(f"snap_surf args.map_init: unknown key(s): {bad}")
		cfg = SnapSurfMapInitConfig(
			enabled=bool(raw_cfg.get("enabled", defaults.enabled)),
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
			scale_levels=max(1, int(raw_cfg.get("scale_levels", defaults.scale_levels))),
			scale_factor=max(1, int(raw_cfg.get("scale_factor", defaults.scale_factor))),
			min_scale_level=max(0, int(raw_cfg.get("min_scale_level", defaults.min_scale_level))),
			coarse_revisit_blocks=max(0, int(raw_cfg.get("coarse_revisit_blocks", defaults.coarse_revisit_blocks))),
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
			jac_margin=float(raw_cfg.get("jac_margin", defaults.jac_margin)),
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
	):
		if float(getattr(cfg, name)) < 0.0:
			raise ValueError(f"snap_surf args.map_init.{name} must be >= 0")
	if cfg.max_sample_distance < 0.0:
		raise ValueError("snap_surf args.map_init.max_sample_distance must be >= 0")
	if cfg.max_sample_angle_deg < 0.0 or cfg.max_sample_angle_deg > 180.0:
		raise ValueError("snap_surf args.map_init.max_sample_angle_deg must be in [0, 180]")
	if cfg.sample_angle_step_fraction < 0.0:
		raise ValueError("snap_surf args.map_init.sample_angle_step_fraction must be >= 0")
	if cfg.max_step_neighbor_ratio < 0.0:
		raise ValueError("snap_surf args.map_init.max_step_neighbor_ratio must be >= 0")
	if cfg.jac_margin < 0.0:
		raise ValueError("snap_surf args.map_init.jac_margin must be >= 0")
	if cfg.progress_mode not in ("block", "periodic", "both", "none"):
		raise ValueError("snap_surf args.map_init.progress_mode must be one of block, periodic, both, none")
	if int(cfg.scale_levels) > 1 and int(cfg.scale_factor) != 2:
		raise ValueError("snap_surf args.map_init.scale_factor must be 2 when scale_levels > 1")
	return cfg


def last_stats() -> dict[str, float]:
	return dict(_last_stats)


def set_debug_step(step: int | None, *, label: str | None = None) -> None:
	global _debug_step, _debug_label
	_debug_step = None if step is None else int(step)
	_debug_label = None if label is None else str(label)


def _safe_frac(n: int | float, d: int | float) -> float:
	den = float(d)
	if den <= 0.0:
		return 0.0
	return float(n) / den


def _snap_surf_log(message: str) -> None:
	print(f"[snap_surf] {message}", flush=True)


def _map_init_log(message: str) -> None:
	print(f"[snap_surf.map_init] {message}", flush=True)


class _DirectionState:
	def __init__(self, *, source_rank: int, target_rank: int) -> None:
		self.source_rank = int(source_rank)
		self.target_rank = int(target_rank)
		self.source_shape: tuple[int, ...] | None = None
		self.target_shape: tuple[int, ...] | None = None
		self.map: torch.Tensor | None = None
		self.valid: torch.Tensor | None = None
		self.orientation_sign: int = 1
		self.seed_base_idx: tuple[int, ...] | None = None

	def ensure(
		self,
		*,
		source_shape: tuple[int, ...],
		target_shape: tuple[int, ...],
		device: torch.device,
		dtype: torch.dtype,
	) -> None:
		if (
			self.map is not None
			and self.valid is not None
			and self.source_shape == source_shape
			and self.target_shape == target_shape
			and self.map.device == device
			and self.map.dtype == dtype
		):
			return
		self.source_shape = tuple(int(v) for v in source_shape)
		self.target_shape = tuple(int(v) for v in target_shape)
		self.map = torch.full((*self.source_shape, self.target_rank), float("nan"), device=device, dtype=dtype)
		self.valid = torch.zeros(self.source_shape, device=device, dtype=torch.bool)
		self.orientation_sign = 1
		self.seed_base_idx = None

	def count(self) -> int:
		if self.valid is None:
			return 0
		return int(self.valid.sum().detach().cpu())


class _MapInitState:
	def __init__(self) -> None:
		self.done: bool = False
		self.active_quad: torch.Tensor | None = None
		self.blocked_quad: torch.Tensor | None = None
		self.scale_active_quads: list[torch.Tensor | None] = []
		self.scale_blocked_quads: list[torch.Tensor | None] = []
		self.uv: torch.Tensor | None = None
		self.uv_guess: torch.Tensor | None = None
		self.ext_pos: torch.Tensor | None = None
		self.ext_normals: torch.Tensor | None = None
		self.ext_valid: torch.Tensor | None = None
		self.ext_quad_valid: torch.Tensor | None = None
		self.ext_coords: torch.Tensor | None = None
		self.uv_pyramid: list[torch.Tensor] | None = None
		self.scale_level: int = 0
		self.target_scale_level: int = 0
		self.scale_strides: list[int] = [1]
		self.model_depth: int | None = None
		self.seed_ext_sample_hw: tuple[int, int] | None = None
		self.seed_model_quad: tuple[int, int, int] | None = None
		self.orientation_sign: int = 1
		self.normal_sign: int = 1
		self.total_iters: int = 0
		self.grow_steps: int = 0
		self.opt_blocks: int = 0
		self.global_opt_blocks: int = 0
		self.rim_only_blocks: int = 0
		self.rim_problem_blocks: int = 0
		self.rim_blocks_since_global_opt: int = 0
		self.repair_blocks: int = 0
		self.added_total: int = 0
		self.sparse_pruned_total: int = 0
		self.add_sample_loss_sum: float = 0.0
		self.add_sample_weight: float = 0.0
		self.add_bad_samples: float = 0.0
		self.add_total_samples: float = 0.0
		self.add_success_quads: float = 0.0
		self.add_total_quads: float = 0.0
		self.fringe_sample_loss_sum: float = 0.0
		self.fringe_sample_weight: float = 0.0
		self.fringe_bad_samples: float = 0.0
		self.fringe_total_samples: float = 0.0
		self.fringe_success_quads: float = 0.0
		self.fringe_total_quads: float = 0.0
		self.interval_add_sample_loss_sum: float = 0.0
		self.interval_add_sample_weight: float = 0.0
		self.interval_add_bad_samples: float = 0.0
		self.interval_add_total_samples: float = 0.0
		self.interval_add_success_quads: float = 0.0
		self.interval_add_total_quads: float = 0.0
		self.interval_fringe_sample_loss_sum: float = 0.0
		self.interval_fringe_sample_weight: float = 0.0
		self.interval_fringe_bad_samples: float = 0.0
		self.interval_fringe_total_samples: float = 0.0
		self.interval_fringe_success_quads: float = 0.0
		self.interval_fringe_total_quads: float = 0.0
		self.fringe_debug_rows: int = 0
		self.scale_levels_used: int = 1
		self.progress_rows: int = 0
		self.progress_last_time: float | None = None
		self.progress_last_iter: int = 0
		self.blocked_last_revisit_iter: int = 0
		self.coarse_revisit_last_block: int = 0
		self.stats: dict[str, float] = {}

	def reset(self) -> None:
		self.__init__()

	@property
	def active(self) -> torch.Tensor | None:
		return self.active_quad

	@active.setter
	def active(self, value: torch.Tensor | None) -> None:
		self.active_quad = value

	@property
	def uv_ext_vertex(self) -> torch.Tensor | None:
		return self.uv

	@uv_ext_vertex.setter
	def uv_ext_vertex(self, value: torch.Tensor | None) -> None:
		self.uv = value

	def active_count(self) -> int:
		if self.active_quad is None:
			return 0
		return int(self.active_quad.sum().detach().cpu())

	def current_stride(self) -> int:
		if 0 <= int(self.scale_level) < len(self.scale_strides):
			return int(self.scale_strides[int(self.scale_level)])
		return 1


class _SurfaceState:
	def __init__(self) -> None:
		self.model_to_ext = _DirectionState(source_rank=3, target_rank=2)
		self.ext_to_model = _DirectionState(source_rank=2, target_rank=3)
		self.map_init = _MapInitState()
		self.ext_seed_hw: tuple[int, int] | None = None
		self.seed_ext_distance: float | None = None
		self.seed_ext_key: tuple[float, float, float] | None = None
		self.seed_ext_point_xyz: tuple[float, float, float] | None = None
		self.snap_eval_count: int = 0

	def reset_map_init(self) -> None:
		self.map_init.reset()

	def ensure(
		self,
		*,
		model_shape: tuple[int, int, int],
		ext_shape: tuple[int, int],
		device: torch.device,
		dtype: torch.dtype,
	) -> None:
		old_model_shape = self.model_to_ext.source_shape
		old_ext_shape = self.model_to_ext.target_shape
		self.model_to_ext.ensure(
			source_shape=model_shape,
			target_shape=ext_shape,
			device=device,
			dtype=dtype,
		)
		self.ext_to_model.ensure(
			source_shape=ext_shape,
			target_shape=model_shape,
			device=device,
			dtype=dtype,
		)
		if old_model_shape != model_shape or old_ext_shape != ext_shape:
			self.ext_seed_hw = None
			self.seed_ext_distance = None
			self.seed_ext_key = None
			self.seed_ext_point_xyz = None
			self.snap_eval_count = 0
			self.reset_map_init()


_CORNERS_2D = ((0, 0), (1, 0), (0, 1), (1, 1))


def _dihedral_transforms() -> list[tuple[tuple[int, int], ...]]:
	return [
		tuple((h, w) for h, w in _CORNERS_2D),
		tuple((w, 1 - h) for h, w in _CORNERS_2D),
		tuple((1 - h, 1 - w) for h, w in _CORNERS_2D),
		tuple((1 - w, h) for h, w in _CORNERS_2D),
		tuple((1 - h, w) for h, w in _CORNERS_2D),
		tuple((h, 1 - w) for h, w in _CORNERS_2D),
		tuple((w, h) for h, w in _CORNERS_2D),
		tuple((1 - w, 1 - h) for h, w in _CORNERS_2D),
	]


def _transform_det_sign(transform: tuple[tuple[int, int], ...]) -> int:
	t00 = torch.tensor(transform[0], dtype=torch.float32)
	t10 = torch.tensor(transform[1], dtype=torch.float32)
	t01 = torch.tensor(transform[2], dtype=torch.float32)
	a = t10 - t00
	b = t01 - t00
	det = float(a[0] * b[1] - a[1] * b[0])
	return 1 if det >= 0.0 else -1


def _normalized_seed_quad(points: torch.Tensor) -> torch.Tensor:
	centered = points - points.mean(dim=0, keepdim=True)
	scale = centered.square().sum(dim=-1).mean().sqrt()
	if not bool(torch.isfinite(scale).detach().cpu()) or float(scale.detach().cpu()) <= 1.0e-8:
		return centered
	return centered / scale


def _source_hw_from_index(idx: tuple[int, ...], *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
	return torch.tensor([float(idx[-2]), float(idx[-1])], device=device, dtype=dtype)


def _coord_in_bounds(coord: torch.Tensor, shape: tuple[int, ...]) -> bool:
	if not bool(torch.isfinite(coord).all().detach().cpu()):
		return False
	for i, size in enumerate(shape):
		v = float(coord[i].detach().cpu())
		if v < 0.0 or v > float(size - 1):
			return False
	return True


def _sample_grid2d(grid: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
	H, W = int(grid.shape[0]), int(grid.shape[1])
	if H < 2 or W < 2:
		return torch.full((*coords.shape[:-1], int(grid.shape[-1])), float("nan"), device=grid.device, dtype=grid.dtype)
	h = coords[..., 0].clamp(0.0, float(H - 1))
	w = coords[..., 1].clamp(0.0, float(W - 1))
	h0 = torch.floor(h).clamp(0, H - 2).long()
	w0 = torch.floor(w).clamp(0, W - 2).long()
	h1 = h0 + 1
	w1 = w0 + 1
	fh = (h - h0.to(dtype=grid.dtype)).unsqueeze(-1)
	fw = (w - w0.to(dtype=grid.dtype)).unsqueeze(-1)
	v00 = grid[h0, w0]
	v10 = grid[h1, w0]
	v01 = grid[h0, w1]
	v11 = grid[h1, w1]
	return (1 - fh) * (1 - fw) * v00 + fh * (1 - fw) * v10 + (1 - fh) * fw * v01 + fh * fw * v11


def _sample_grid3d(grid: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
	D, H, W = int(grid.shape[0]), int(grid.shape[1]), int(grid.shape[2])
	if H < 2 or W < 2:
		return torch.full((*coords.shape[:-1], int(grid.shape[-1])), float("nan"), device=grid.device, dtype=grid.dtype)
	d = coords[..., 0].clamp(0.0, float(D - 1))
	if D < 2:
		return _sample_grid2d(grid[0], coords[..., 1:])
	h = coords[..., 1].clamp(0.0, float(H - 1))
	w = coords[..., 2].clamp(0.0, float(W - 1))
	d0 = torch.floor(d).clamp(0, D - 2).long()
	d1 = d0 + 1
	fd = (d - d0.to(dtype=grid.dtype)).unsqueeze(-1)
	h0 = torch.floor(h).clamp(0, H - 2).long()
	w0 = torch.floor(w).clamp(0, W - 2).long()
	h1 = h0 + 1
	w1 = w0 + 1
	fh = (h - h0.to(dtype=grid.dtype)).unsqueeze(-1)
	fw = (w - w0.to(dtype=grid.dtype)).unsqueeze(-1)
	v000 = grid[d0, h0, w0]
	v010 = grid[d0, h1, w0]
	v001 = grid[d0, h0, w1]
	v011 = grid[d0, h1, w1]
	v100 = grid[d1, h0, w0]
	v110 = grid[d1, h1, w0]
	v101 = grid[d1, h0, w1]
	v111 = grid[d1, h1, w1]
	p0 = (1 - fh) * (1 - fw) * v000 + fh * (1 - fw) * v010 + (1 - fh) * fw * v001 + fh * fw * v011
	p1 = (1 - fh) * (1 - fw) * v100 + fh * (1 - fw) * v110 + (1 - fh) * fw * v101 + fh * fw * v111
	return (1 - fd) * p0 + fd * p1


def _sample_grid(grid: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
	if grid.ndim == 3:
		return _sample_grid2d(grid, coords)
	if grid.ndim == 4:
		return _sample_grid3d(grid, coords)
	raise ValueError(f"expected 2D/3D grid with vector channel, got shape {tuple(grid.shape)}")


def _sample_surface_grid(grid: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
	if grid.ndim == 3:
		return _sample_grid2d(grid, coords)
	if grid.ndim != 4:
		raise ValueError(f"expected 2D/3D surface grid with vector channel, got shape {tuple(grid.shape)}")
	D, H, W, C = int(grid.shape[0]), int(grid.shape[1]), int(grid.shape[2]), int(grid.shape[3])
	if H < 2 or W < 2:
		return torch.full((*coords.shape[:-1], C), float("nan"), device=grid.device, dtype=grid.dtype)
	flat = coords.reshape(-1, 3)
	d = torch.round(flat[:, 0]).clamp(0, D - 1).long()
	h = flat[:, 1].clamp(0.0, float(H - 1))
	w = flat[:, 2].clamp(0.0, float(W - 1))
	h0 = torch.floor(h).clamp(0, H - 2).long()
	w0 = torch.floor(w).clamp(0, W - 2).long()
	h1 = h0 + 1
	w1 = w0 + 1
	fh = (h - h0.to(dtype=grid.dtype)).unsqueeze(-1)
	fw = (w - w0.to(dtype=grid.dtype)).unsqueeze(-1)
	v00 = grid[d, h0, w0]
	v10 = grid[d, h1, w0]
	v01 = grid[d, h0, w1]
	v11 = grid[d, h1, w1]
	out = (1 - fh) * (1 - fw) * v00 + fh * (1 - fw) * v10 + (1 - fh) * fw * v01 + fh * fw * v11
	return out.reshape(*coords.shape[:-1], C)


def _points_at_indices(grid: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
	if idx.numel() == 0:
		return torch.empty(0, int(grid.shape[-1]), device=grid.device, dtype=grid.dtype)
	if idx.shape[1] == 2:
		return grid[idx[:, 0], idx[:, 1]]
	return grid[idx[:, 0], idx[:, 1], idx[:, 2]]


def _batched_source_views(
	state: _DirectionState,
	source_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	if state.map is None or state.valid is None:
		raise RuntimeError("snap_surf direction state is not initialized")
	if state.source_rank == 3:
		return state.valid, state.map, source_valid.bool()
	return state.valid.unsqueeze(0), state.map.unsqueeze(0), source_valid.bool().unsqueeze(0)


def _write_batched_state(state: _DirectionState, valid_b: torch.Tensor, map_b: torch.Tensor) -> None:
	if state.map is None or state.valid is None:
		return
	valid_b = valid_b & torch.isfinite(map_b).all(dim=-1)
	if state.source_rank == 3:
		state.valid[:] = valid_b
		state.map[:] = map_b
	else:
		state.valid[:] = valid_b[0]
		state.map[:] = map_b[0]
	bad_valid = state.valid & ~torch.isfinite(state.map).all(dim=-1)
	if bool(bad_valid.any().detach().cpu()):
		state.valid[bad_valid] = False
	state.map[~state.valid] = float("nan")


def _source_hw_grid(*, n: int, h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
	hh = torch.arange(h, device=device, dtype=dtype).view(1, h, 1).expand(n, h, w)
	ww = torch.arange(w, device=device, dtype=dtype).view(1, 1, w).expand(n, h, w)
	return torch.stack([hh, ww], dim=-1)


def _quad_valid_at_coords(valid: torch.Tensor, coords: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
	if coords.numel() == 0:
		return torch.zeros(coords.shape[:-1], device=valid.device, dtype=torch.bool)
	flat = coords.reshape(-1, coords.shape[-1])
	finite = torch.isfinite(flat).all(dim=-1)
	safe_flat = torch.where(torch.isfinite(flat), flat, torch.zeros_like(flat))
	if len(shape) == 2:
		H, W = int(shape[0]), int(shape[1])
		if H < 2 or W < 2:
			return torch.zeros(coords.shape[:-1], device=valid.device, dtype=torch.bool)
		h = safe_flat[:, 0]
		w = safe_flat[:, 1]
		in_bounds = finite & (h >= 0.0) & (h <= float(H - 1)) & (w >= 0.0) & (w <= float(W - 1))
		h0 = torch.floor(h.clamp(0.0, float(H - 1))).clamp(0, H - 2).long()
		w0 = torch.floor(w.clamp(0.0, float(W - 1))).clamp(0, W - 2).long()
		ok = (
			valid[h0, w0] &
			valid[h0 + 1, w0] &
			valid[h0, w0 + 1] &
			valid[h0 + 1, w0 + 1]
		) & in_bounds
		return ok.reshape(coords.shape[:-1])

	D, H, W = int(shape[0]), int(shape[1]), int(shape[2])
	if H < 2 or W < 2:
		return torch.zeros(coords.shape[:-1], device=valid.device, dtype=torch.bool)
	d = safe_flat[:, 0]
	h = safe_flat[:, 1]
	w = safe_flat[:, 2]
	in_bounds = (
		finite &
		(d >= 0.0) & (d <= float(D - 1)) &
		(h >= 0.0) & (h <= float(H - 1)) &
		(w >= 0.0) & (w <= float(W - 1))
	)
	di = torch.round(d.clamp(0.0, float(D - 1))).long()
	h0 = torch.floor(h.clamp(0.0, float(H - 1))).clamp(0, H - 2).long()
	w0 = torch.floor(w.clamp(0.0, float(W - 1))).clamp(0, W - 2).long()
	ok = (
		valid[di, h0, w0] &
		valid[di, h0 + 1, w0] &
		valid[di, h0, w0 + 1] &
		valid[di, h0 + 1, w0 + 1]
	) & in_bounds
	return ok.reshape(coords.shape[:-1])


def _neighbor4_mask(valid_b: torch.Tensor) -> torch.Tensor:
	n, h, w = valid_b.shape
	if h == 0 or w == 0:
		return torch.zeros_like(valid_b)
	x = valid_b.to(dtype=torch.float32).reshape(n, 1, h, w)
	k = torch.tensor(
		[[[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]],
		device=valid_b.device,
		dtype=torch.float32,
	).unsqueeze(0)
	return F.conv2d(x, k, padding=1).reshape(n, h, w) > 0.0


def _neighbor8_mask(valid_b: torch.Tensor) -> torch.Tensor:
	n, h, w = valid_b.shape
	if h == 0 or w == 0:
		return torch.zeros_like(valid_b)
	x = valid_b.to(dtype=torch.float32).reshape(n, 1, h, w)
	k = torch.ones(1, 1, 3, 3, device=valid_b.device, dtype=torch.float32)
	k[..., 1, 1] = 0.0
	return F.conv2d(x, k, padding=1).reshape(n, h, w) > 0.0


def _dilate_mask_2d(mask_b: torch.Tensor, *, radius: int) -> torch.Tensor:
	r = max(0, int(radius))
	if r == 0:
		return mask_b.bool()
	if mask_b.numel() == 0:
		return mask_b.bool()
	x = mask_b.to(dtype=torch.float32).unsqueeze(1)
	return F.max_pool2d(x, kernel_size=2 * r + 1, stride=1, padding=r).squeeze(1) > 0.0


def _brute_source_front_mask(
	*,
	prev_valid: torch.Tensor | None,
	source_mask: torch.Tensor,
	seed_quad: tuple[int, int, int] | None,
	radius: int,
) -> torch.Tensor:
	source_b = source_mask.bool()
	if prev_valid is not None and bool((prev_valid.bool() & source_b).any().detach().cpu()):
		inlier = prev_valid.bool() & source_b
		boundary = inlier & _neighbor8_mask(source_b & ~inlier)
		if not bool(boundary.any().detach().cpu()):
			boundary = inlier
	else:
		boundary = _seed_quad_corner_mask(
			tuple(int(v) for v in source_b.shape),
			seed_quad,
			device=source_b.device,
		) & source_b
		if not bool(boundary.any().detach().cpu()):
			boundary = source_b
	return _dilate_mask_2d(boundary, radius=radius) & source_b


def _seed_source_limit_mask(state: _DirectionState, valid_b: torch.Tensor, *, radius: int) -> torch.Tensor:
	if state.seed_base_idx is None:
		return torch.ones_like(valid_b, dtype=torch.bool)
	r = int(radius)
	_, h, w = valid_b.shape
	device = valid_b.device
	hh = torch.arange(h, device=device).view(1, h, 1)
	ww = torch.arange(w, device=device).view(1, 1, w)
	if state.source_rank == 3:
		d0, h0, w0 = (int(v) for v in state.seed_base_idx)
		dd = torch.arange(valid_b.shape[0], device=device).view(-1, 1, 1)
		return (
			(dd == d0) &
			(hh >= h0 - r) &
			(hh <= h0 + 1 + r) &
			(ww >= w0 - r) &
			(ww <= w0 + 1 + r)
		)
	h0, w0 = (int(v) for v in state.seed_base_idx)
	return (
		(hh >= h0 - r) &
		(hh <= h0 + 1 + r) &
		(ww >= w0 - r) &
		(ww <= w0 + 1 + r)
	).expand_as(valid_b)


def _local_affine_predict_batched(
	state: _DirectionState,
	*,
	valid_b: torch.Tensor,
	map_b: torch.Tensor,
	radius: int,
	exclude_self: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	"""Predict target coords for every source corner from local accepted supports."""
	n, h, w = valid_b.shape
	rank = state.target_rank
	device = map_b.device
	dtype = map_b.dtype
	if h == 0 or w == 0:
		empty_pred = torch.empty(n, h, w, rank, device=device, dtype=dtype)
		empty_count = torch.empty(n, h, w, device=device, dtype=torch.long)
		empty_det = torch.empty(n, h, w, device=device, dtype=dtype)
		return empty_pred, empty_count, empty_det

	k_size = 2 * int(radius) + 1
	k_count = k_size * k_size
	source_hw = _source_hw_grid(n=n, h=h, w=w, device=device, dtype=dtype)
	support_valid = valid_b.to(dtype=dtype)
	target_safe = torch.where(valid_b.unsqueeze(-1), map_b, torch.zeros_like(map_b))

	src_patch = F.unfold(
		source_hw.permute(0, 3, 1, 2),
		kernel_size=k_size,
		padding=int(radius),
	).transpose(1, 2).reshape(n, h * w, 2, k_count).transpose(2, 3)
	tgt_patch = F.unfold(
		target_safe.permute(0, 3, 1, 2),
		kernel_size=k_size,
		padding=int(radius),
	).transpose(1, 2).reshape(n, h * w, rank, k_count).transpose(2, 3)
	w_patch = F.unfold(
		support_valid.unsqueeze(1),
		kernel_size=k_size,
		padding=int(radius),
	).transpose(1, 2).reshape(n, h * w, k_count)
	if exclude_self:
		w_patch[..., k_count // 2] = 0.0
	count = w_patch.sum(dim=-1).to(dtype=torch.long)

	ones = torch.ones(n, h * w, k_count, 1, device=device, dtype=dtype)
	A = torch.cat([src_patch, ones], dim=-1)
	Aw = A * w_patch.unsqueeze(-1)
	ATA = torch.einsum("nlki,nlkj->nlij", Aw, A)
	ATY = torch.einsum("nlki,nlkr->nlir", Aw, tgt_patch)
	eye = torch.eye(3, device=device, dtype=dtype).view(1, 1, 3, 3)
	try:
		sol = torch.linalg.solve(ATA + eye * 1.0e-4, ATY)
	except RuntimeError:
		sol = torch.linalg.pinv(ATA + eye * 1.0e-4) @ ATY
	query = torch.cat([
		source_hw.reshape(n, h * w, 2),
		torch.ones(n, h * w, 1, device=device, dtype=dtype),
	], dim=-1)
	pred = torch.einsum("nli,nlir->nlr", query, sol).reshape(n, h, w, rank)
	if rank >= 2:
		det = (
			sol[:, :, 0, -2] * sol[:, :, 1, -1] -
			sol[:, :, 0, -1] * sol[:, :, 1, -2]
		).reshape(n, h, w)
	else:
		det = torch.ones(n, h, w, device=device, dtype=dtype)
	return pred, count.reshape(n, h, w), det


def _first_support_predict_batched(
	state: _DirectionState,
	*,
	valid_b: torch.Tensor,
	map_b: torch.Tensor,
	radius: int,
) -> tuple[torch.Tensor, torch.Tensor]:
	n, h, w = valid_b.shape
	rank = state.target_rank
	device = map_b.device
	dtype = map_b.dtype
	if h == 0 or w == 0:
		return (
			torch.empty(n, h, w, rank, device=device, dtype=dtype),
			torch.empty(n, h, w, device=device, dtype=torch.long),
		)

	k_size = 2 * int(radius) + 1
	k_count = k_size * k_size
	source_hw = _source_hw_grid(n=n, h=h, w=w, device=device, dtype=dtype)
	target_safe = torch.where(valid_b.unsqueeze(-1), map_b, torch.zeros_like(map_b))

	src_patch = F.unfold(
		source_hw.permute(0, 3, 1, 2),
		kernel_size=k_size,
		padding=int(radius),
	).transpose(1, 2).reshape(n, h * w, 2, k_count).transpose(2, 3)
	tgt_patch = F.unfold(
		target_safe.permute(0, 3, 1, 2),
		kernel_size=k_size,
		padding=int(radius),
	).transpose(1, 2).reshape(n, h * w, rank, k_count).transpose(2, 3)
	w_patch = F.unfold(
		valid_b.to(dtype=dtype).unsqueeze(1),
		kernel_size=k_size,
		padding=int(radius),
	).transpose(1, 2).reshape(n, h * w, k_count)

	count = w_patch.sum(dim=-1).to(dtype=torch.long)
	first = torch.argmax(w_patch, dim=-1)
	first_src = torch.gather(
		src_patch,
		2,
		first.view(n, h * w, 1, 1).expand(n, h * w, 1, 2),
	).squeeze(2)
	first_tgt = torch.gather(
		tgt_patch,
		2,
		first.view(n, h * w, 1, 1).expand(n, h * w, 1, rank),
	).squeeze(2)
	query_hw = source_hw.reshape(n, h * w, 2)
	pred = first_tgt.clone()
	pred[..., -2:] = pred[..., -2:] + (query_hw - first_src)
	return pred.reshape(n, h, w, rank), count.reshape(n, h, w)


def _direct_predict_candidates_batched(
	state: _DirectionState,
	*,
	valid_b: torch.Tensor,
	map_b: torch.Tensor,
	candidate_bidx: torch.Tensor,
	radius: int,
	single_neighbor_transform: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	C = int(candidate_bidx.shape[0])
	rank = state.target_rank
	device = map_b.device
	dtype = map_b.dtype
	if C == 0:
		return (
			torch.empty(0, rank, device=device, dtype=dtype),
			torch.empty(0, device=device, dtype=torch.long),
			torch.empty(0, rank, device=device, dtype=dtype),
		)
	n, h, w = valid_b.shape
	k_size = 2 * int(radius) + 1
	k_count = k_size * k_size
	source_hw = _source_hw_grid(n=n, h=h, w=w, device=device, dtype=dtype)
	target_safe = torch.where(valid_b.unsqueeze(-1), map_b, torch.zeros_like(map_b))

	src_patch_all = F.unfold(
		source_hw.permute(0, 3, 1, 2),
		kernel_size=k_size,
		padding=int(radius),
	).transpose(1, 2).reshape(n, h * w, 2, k_count).transpose(2, 3)
	tgt_patch_all = F.unfold(
		target_safe.permute(0, 3, 1, 2),
		kernel_size=k_size,
		padding=int(radius),
	).transpose(1, 2).reshape(n, h * w, rank, k_count).transpose(2, 3)
	w_patch_all = F.unfold(
		valid_b.to(dtype=dtype).unsqueeze(1),
		kernel_size=k_size,
		padding=int(radius),
	).transpose(1, 2).reshape(n, h * w, k_count)

	batch = candidate_bidx[:, 0]
	linear = candidate_bidx[:, 1] * w + candidate_bidx[:, 2]
	src_patch = src_patch_all[batch, linear]
	tgt_patch = tgt_patch_all[batch, linear]
	w_patch = w_patch_all[batch, linear]
	count = w_patch.sum(dim=-1).to(dtype=torch.long)

	A = torch.cat([src_patch, torch.ones(C, k_count, 1, device=device, dtype=dtype)], dim=-1)
	Aw = A * w_patch.unsqueeze(-1)
	ATA = torch.einsum("cki,ckj->cij", Aw, A)
	ATY = torch.einsum("cki,ckr->cir", Aw, tgt_patch)
	eye = torch.eye(3, device=device, dtype=dtype).view(1, 3, 3)
	try:
		sol = torch.linalg.solve(ATA + eye * 1.0e-4, ATY)
	except RuntimeError:
		sol = torch.linalg.pinv(ATA + eye * 1.0e-4) @ ATY
	query_hw = candidate_bidx[:, 1:].to(dtype=dtype)
	query = torch.cat([query_hw, torch.ones(C, 1, device=device, dtype=dtype)], dim=-1)
	affine_pred = torch.einsum("ci,cir->cr", query, sol)

	source_dist2 = (src_patch - query_hw[:, None, :]).square().sum(dim=-1)
	source_dist2 = torch.where(w_patch > 0.0, source_dist2, torch.full_like(source_dist2, float("inf")))
	nearest = torch.argmin(source_dist2, dim=-1)
	nearest_src = torch.gather(
		src_patch,
		1,
		nearest.view(C, 1, 1).expand(C, 1, 2),
	).squeeze(1)
	nearest_tgt = torch.gather(
		tgt_patch,
		1,
		nearest.view(C, 1, 1).expand(C, 1, rank),
	).squeeze(1)
	if single_neighbor_transform is None:
		nearest_step_pred = nearest_tgt.clone()
		nearest_step_pred[..., -2:] = nearest_step_pred[..., -2:] + (query_hw - nearest_src)
	else:
		step = single_neighbor_transform.to(device=device, dtype=dtype)
		if tuple(step.shape) != (2, rank):
			raise ValueError(f"single_neighbor_transform must have shape {(2, rank)}, got {tuple(step.shape)}")
		nearest_step_pred = nearest_tgt + (query_hw - nearest_src) @ step
	if single_neighbor_transform is None:
		use_step = count == 1
	else:
		use_step = count < 3
	pred = torch.where(use_step.unsqueeze(-1), nearest_step_pred, affine_pred)
	return pred, count, nearest_tgt


def _bool_at_index(mask: torch.Tensor, idx: tuple[int, ...]) -> bool:
	return bool(mask[idx].detach().cpu())


def _set_correspondence(state: _DirectionState, source_idx: tuple[int, ...], target_coord: torch.Tensor) -> None:
	if state.map is None or state.valid is None:
		return
	coord = target_coord.to(device=state.map.device, dtype=state.map.dtype)
	if not bool(torch.isfinite(coord).all().detach().cpu()):
		state.valid[source_idx] = False
		state.map[source_idx] = float("nan")
		return
	state.map[source_idx] = coord
	state.valid[source_idx] = True


def _svd_rank_2d(points: torch.Tensor) -> int:
	if points.shape[0] < 2:
		return 0
	centered = points - points.mean(dim=0, keepdim=True)
	try:
		s = torch.linalg.svdvals(centered)
	except RuntimeError:
		return 0
	return int((s > 1.0e-6).sum().detach().cpu())


def _similarity_predict(
	source: torch.Tensor,
	target: torch.Tensor,
	query: torch.Tensor,
	*,
	orientation_sign: int,
) -> torch.Tensor | None:
	n = int(source.shape[0])
	if n < 2:
		return None
	best_i, best_j = 0, 1
	best_len = -1.0
	for i in range(n):
		for j in range(i + 1, n):
			l2 = float((source[j] - source[i]).square().sum().detach().cpu())
			if l2 > best_len:
				best_i, best_j = i, j
				best_len = l2
	if best_len <= 1.0e-12:
		return None
	s0 = source[best_i]
	s1 = source[best_j]
	t0 = target[best_i]
	t1 = target[best_j]
	e = s1 - s0
	f = t1 - t0
	p = torch.stack([-e[1], e[0]])
	q = query - s0
	len2 = (e * e).sum().clamp(min=1.0e-12)
	a = (q * e).sum() / len2
	b = (q * p).sum() / len2
	pred = t0 + a * f
	orient = 1.0 if int(orientation_sign) >= 0 else -1.0
	if target.shape[1] == 2:
		f_perp = torch.stack([-f[1], f[0]])
		pred = pred + b * orient * f_perp
	else:
		f_hw = f[-2:]
		f_perp_hw = torch.stack([-f_hw[1], f_hw[0]])
		if float(f_perp_hw.square().sum().detach().cpu()) <= 1.0e-12:
			return None
		pred = pred.clone()
		pred[-2:] = pred[-2:] + b * orient * f_perp_hw
	return pred


def _predict_target_coord(
	source: torch.Tensor,
	target: torch.Tensor,
	query: torch.Tensor,
	*,
	orientation_sign: int,
) -> torch.Tensor | None:
	if int(source.shape[0]) < 2:
		return None
	if int(source.shape[0]) >= 3 and _svd_rank_2d(source) >= 2:
		S = torch.cat([source, torch.ones(source.shape[0], 1, device=source.device, dtype=source.dtype)], dim=1)
		q = torch.cat([query, query.new_ones(1)], dim=0)
		try:
			sol = torch.linalg.lstsq(S, target).solution
		except RuntimeError:
			sol = torch.linalg.pinv(S) @ target
		return q @ sol
	return _similarity_predict(source, target, query, orientation_sign=orientation_sign)


def _affine_orientation_pass(
	count: torch.Tensor,
	det: torch.Tensor,
	*,
	orientation_sign: int,
	eps: float = 1.0e-4,
) -> torch.Tensor:
	expected = 1.0 if int(orientation_sign) >= 0 else -1.0
	confident = (count >= 3) & torch.isfinite(det) & (det.abs() > float(eps))
	return (~confident) | ((det * expected) >= 0.0)


def _affine_det_sign_from_points(
	source_hw: torch.Tensor,
	target_coord: torch.Tensor,
	*,
	fallback: int,
	eps: float = 1.0e-6,
) -> int:
	target_hw = target_coord[:, -2:]
	finite = torch.isfinite(source_hw).all(dim=-1) & torch.isfinite(target_hw).all(dim=-1)
	if int(finite.sum().detach().cpu()) < 3:
		return 1 if int(fallback) >= 0 else -1
	source_hw = source_hw[finite]
	target_hw = target_hw[finite]
	A = torch.cat([source_hw, torch.ones(source_hw.shape[0], 1, device=source_hw.device, dtype=source_hw.dtype)], dim=1)
	try:
		sol = torch.linalg.lstsq(A, target_hw).solution
	except RuntimeError:
		sol = torch.linalg.pinv(A) @ target_hw
	det = sol[0, 0] * sol[1, 1] - sol[0, 1] * sol[1, 0]
	if not bool(torch.isfinite(det).detach().cpu()) or abs(float(det.detach().cpu())) <= float(eps):
		return 1 if int(fallback) >= 0 else -1
	return 1 if float(det.detach().cpu()) >= 0.0 else -1


def _clear_direction_state(state: _DirectionState) -> None:
	if state.map is None or state.valid is None:
		return
	state.valid[:] = False
	state.map[:] = float("nan")


def _target_quad_bases_around_batched(
	pred: torch.Tensor,
	target_shape: tuple[int, ...],
	*,
	search_ring: int,
) -> tuple[torch.Tensor, torch.Tensor]:
	r = max(1, int(search_ring))
	device = pred.device
	C = int(pred.shape[0])
	rank = len(target_shape)
	if rank not in {2, 3}:
		raise ValueError(f"expected 2D/3D target shape, got {target_shape}")
	if C == 0 or int(target_shape[-2]) < 2 or int(target_shape[-1]) < 2:
		return (
			torch.empty(C, 0, rank, device=device, dtype=torch.long),
			torch.zeros(C, 0, device=device, dtype=torch.bool),
		)
	offs = torch.arange(-r, r + 1, device=device, dtype=torch.long)
	off_h, off_w = torch.meshgrid(offs, offs, indexing="ij")
	hw_offsets = torch.stack([off_h.reshape(-1), off_w.reshape(-1)], dim=-1)
	K = int(hw_offsets.shape[0])
	finite = torch.isfinite(pred).all(dim=-1)
	safe_pred = torch.where(torch.isfinite(pred), pred, torch.zeros_like(pred))
	center_hw = torch.round(safe_pred[:, -2:]).to(dtype=torch.long)
	base_hw = center_hw[:, None, :] + hw_offsets.view(1, K, 2)
	H, W = int(target_shape[-2]), int(target_shape[-1])
	in_bounds = (
		finite[:, None] &
		(base_hw[..., 0] >= 0) &
		(base_hw[..., 0] <= H - 2) &
		(base_hw[..., 1] >= 0) &
		(base_hw[..., 1] <= W - 2)
	)
	if rank == 2:
		return base_hw, in_bounds
	D = int(target_shape[0])
	base_d = torch.round(safe_pred[:, 0]).to(dtype=torch.long).view(C, 1, 1).expand(C, K, 1)
	in_bounds = in_bounds & (base_d[..., 0] >= 0) & (base_d[..., 0] < D)
	return torch.cat([base_d, base_hw], dim=-1), in_bounds


def _all_valid_target_quad_bases(valid: torch.Tensor) -> torch.Tensor:
	"""Return every target quad whose four corners are valid."""
	if valid.ndim == 2:
		H, W = int(valid.shape[0]), int(valid.shape[1])
		if H < 2 or W < 2:
			return torch.empty(0, 2, device=valid.device, dtype=torch.long)
		quad_ok = (
			valid[:-1, :-1].bool() &
			valid[1:, :-1].bool() &
			valid[:-1, 1:].bool() &
			valid[1:, 1:].bool()
		)
		return quad_ok.nonzero(as_tuple=False)
	if valid.ndim == 3:
		D, H, W = int(valid.shape[0]), int(valid.shape[1]), int(valid.shape[2])
		if D < 1 or H < 2 or W < 2:
			return torch.empty(0, 3, device=valid.device, dtype=torch.long)
		quad_ok = (
			valid[:, :-1, :-1].bool() &
			valid[:, 1:, :-1].bool() &
			valid[:, :-1, 1:].bool() &
			valid[:, 1:, 1:].bool()
		)
		return quad_ok.nonzero(as_tuple=False)
	raise ValueError(f"expected 2D/3D validity mask, got shape {tuple(valid.shape)}")


def _quad_valid_at_bases(valid: torch.Tensor, bases: torch.Tensor, in_bounds: torch.Tensor) -> torch.Tensor:
	if bases.numel() == 0:
		return torch.zeros(bases.shape[:-1], device=valid.device, dtype=torch.bool)
	if bases.shape[-1] == 2:
		H, W = int(valid.shape[0]), int(valid.shape[1])
		h = bases[..., 0].clamp(0, max(0, H - 2))
		w = bases[..., 1].clamp(0, max(0, W - 2))
		ok = valid[h, w] & valid[h + 1, w] & valid[h, w + 1] & valid[h + 1, w + 1]
		return ok & in_bounds
	D, H, W = int(valid.shape[0]), int(valid.shape[1]), int(valid.shape[2])
	d = bases[..., 0].clamp(0, max(0, D - 1))
	h = bases[..., 1].clamp(0, max(0, H - 2))
	w = bases[..., 2].clamp(0, max(0, W - 2))
	ok = (
		valid[d, h, w] &
		valid[d, h + 1, w] &
		valid[d, h, w + 1] &
		valid[d, h + 1, w + 1]
	)
	return ok & in_bounds


def _quad_corners_batched(grid: torch.Tensor, bases: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
	if bases.numel() == 0:
		shape = (*bases.shape[:-1], int(grid.shape[-1]))
		empty = torch.empty(shape, device=grid.device, dtype=grid.dtype)
		return empty, empty, empty, empty
	if bases.shape[-1] == 2:
		H, W = int(grid.shape[0]), int(grid.shape[1])
		h = bases[..., 0].clamp(0, max(0, H - 2))
		w = bases[..., 1].clamp(0, max(0, W - 2))
		return grid[h, w], grid[h + 1, w], grid[h, w + 1], grid[h + 1, w + 1]
	D, H, W = int(grid.shape[0]), int(grid.shape[1]), int(grid.shape[2])
	d = bases[..., 0].clamp(0, max(0, D - 1))
	h = bases[..., 1].clamp(0, max(0, H - 2))
	w = bases[..., 2].clamp(0, max(0, W - 2))
	return grid[d, h, w], grid[d, h + 1, w], grid[d, h, w + 1], grid[d, h + 1, w + 1]


def _quad_average_normal_batched(normals: torch.Tensor, bases: torch.Tensor) -> torch.Tensor:
	p00, p10, p01, p11 = _quad_corners_batched(normals, bases)
	return F.normalize((p00 + p10 + p01 + p11) * 0.25, dim=-1, eps=1.0e-8)


def _closest_points_on_triangles_batched(
	p: torch.Tensor,
	a: torch.Tensor,
	b: torch.Tensor,
	c: torch.Tensor,
	*,
	eps: float = 1.0e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
	ab = b - a
	ac = c - a
	ap = p - a

	def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
		return (x * y).sum(dim=-1)

	d1 = dot(ab, ap)
	d2 = dot(ac, ap)
	bp = p - b
	d3 = dot(ab, bp)
	d4 = dot(ac, bp)
	cp = p - c
	d5 = dot(ab, cp)
	d6 = dot(ac, cp)

	vc = d1 * d4 - d3 * d2
	vb = d5 * d2 - d1 * d6
	va = d3 * d6 - d5 * d4

	denom = (va + vb + vc).clamp_min(eps)
	face_v = vb / denom
	face_w = vc / denom
	closest = a + face_v.unsqueeze(-1) * ab + face_w.unsqueeze(-1) * ac
	bary = torch.stack((1.0 - face_v - face_w, face_v, face_w), dim=-1)

	one_a = torch.zeros_like(bary)
	one_a[..., 0] = 1.0
	one_b = torch.zeros_like(bary)
	one_b[..., 1] = 1.0
	one_c = torch.zeros_like(bary)
	one_c[..., 2] = 1.0

	mask_a = (d1 <= 0.0) & (d2 <= 0.0)
	closest = torch.where(mask_a.unsqueeze(-1), a, closest)
	bary = torch.where(mask_a.unsqueeze(-1), one_a, bary)

	mask_b = (d3 >= 0.0) & (d4 <= d3)
	closest = torch.where(mask_b.unsqueeze(-1), b, closest)
	bary = torch.where(mask_b.unsqueeze(-1), one_b, bary)

	mask_ab = (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)
	ab_v = d1 / (d1 - d3).clamp_min(eps)
	closest_ab = a + ab_v.unsqueeze(-1) * ab
	bary_ab = torch.stack((1.0 - ab_v, ab_v, torch.zeros_like(ab_v)), dim=-1)
	closest = torch.where(mask_ab.unsqueeze(-1), closest_ab, closest)
	bary = torch.where(mask_ab.unsqueeze(-1), bary_ab, bary)

	mask_c = (d6 >= 0.0) & (d5 <= d6)
	closest = torch.where(mask_c.unsqueeze(-1), c, closest)
	bary = torch.where(mask_c.unsqueeze(-1), one_c, bary)

	mask_ac = (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)
	ac_w = d2 / (d2 - d6).clamp_min(eps)
	closest_ac = a + ac_w.unsqueeze(-1) * ac
	bary_ac = torch.stack((1.0 - ac_w, torch.zeros_like(ac_w), ac_w), dim=-1)
	closest = torch.where(mask_ac.unsqueeze(-1), closest_ac, closest)
	bary = torch.where(mask_ac.unsqueeze(-1), bary_ac, bary)

	mask_bc = (va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0)
	bc_w = (d4 - d3) / ((d4 - d3) + (d5 - d6)).clamp_min(eps)
	closest_bc = b + bc_w.unsqueeze(-1) * (c - b)
	bary_bc = torch.stack((torch.zeros_like(bc_w), 1.0 - bc_w, bc_w), dim=-1)
	closest = torch.where(mask_bc.unsqueeze(-1), closest_bc, closest)
	bary = torch.where(mask_bc.unsqueeze(-1), bary_bc, bary)

	return closest, bary


def _closest_point_on_quad_along_normal_batched(
	point: torch.Tensor,
	normal: torch.Tensor,
	p00: torch.Tensor,
	p10: torch.Tensor,
	p01: torch.Tensor,
	p11: torch.Tensor,
	base_coord: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	n = F.normalize(normal, dim=-1, eps=1.0e-8)
	normal_ok = torch.isfinite(n).all(dim=-1) & (n.norm(dim=-1) > 1.0e-8)

	def project(x: torch.Tensor) -> torch.Tensor:
		return x - (x * n).sum(dim=-1, keepdim=True) * n

	pp = project(point)
	q00 = project(p00)
	q10 = project(p10)
	q01 = project(p01)
	q11 = project(p11)
	cp0, bary0 = _closest_points_on_triangles_batched(pp, q00, q10, q11)
	cp1, bary1 = _closest_points_on_triangles_batched(pp, q00, q11, q01)
	line0 = (cp0 - pp).square().sum(dim=-1)
	line1 = (cp1 - pp).square().sum(dim=-1)
	use_first = line0 <= line1

	coord_h0 = bary0[..., 1] + bary0[..., 2]
	coord_w0 = bary0[..., 2]
	q0 = bary0[..., 0:1] * p00 + bary0[..., 1:2] * p10 + bary0[..., 2:3] * p11
	coord_h1 = bary1[..., 1]
	coord_w1 = bary1[..., 1] + bary1[..., 2]
	q1 = bary1[..., 0:1] * p00 + bary1[..., 1:2] * p11 + bary1[..., 2:3] * p01

	coord_h = torch.where(use_first, coord_h0, coord_h1)
	coord_w = torch.where(use_first, coord_w0, coord_w1)
	q = torch.where(use_first.unsqueeze(-1), q0, q1)
	line = torch.where(use_first, line0, line1)
	normal_abs = ((point - q) * n).sum(dim=-1).abs()
	coord = base_coord.to(dtype=point.dtype).expand(*coord_h.shape, int(base_coord.shape[-1])).clone()
	coord[..., -2] = coord[..., -2] + coord_h
	coord[..., -1] = coord[..., -1] + coord_w
	finite = (
		normal_ok &
		torch.isfinite(point).all(dim=-1) &
		torch.isfinite(p00).all(dim=-1) &
		torch.isfinite(p10).all(dim=-1) &
		torch.isfinite(p01).all(dim=-1) &
		torch.isfinite(p11).all(dim=-1) &
		torch.isfinite(coord).all(dim=-1) &
		torch.isfinite(line) &
		torch.isfinite(normal_abs)
	)
	line = torch.where(finite, line, torch.full_like(line, float("inf")))
	normal_abs = torch.where(finite, normal_abs, torch.full_like(normal_abs, float("inf")))
	coord = torch.where(finite.unsqueeze(-1), coord, base_coord.to(dtype=point.dtype))
	return coord, line, normal_abs


def _revalidate_direction(
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
	start_count = state.count()
	out: dict[str, int | float] = {"drop": 0, "pgrid": 0, "perr_n": 0, "perr_sum": 0.0, "perr_max": 0.0}
	if state.map is None or state.valid is None or state.target_shape is None:
		return out
	if start_count == 0:
		return out
	valid_b, map_b, source_valid_b = _batched_source_views(state, source_valid)
	new_valid_b = valid_b.clone()
	if not bool(new_valid_b.any().detach().cpu()):
		return out

	coords = map_b[new_valid_b]
	source_idx_b = new_valid_b.nonzero(as_tuple=False)
	source_idx = (
		source_idx_b
		if state.source_rank == 3
		else source_idx_b[:, 1:]
	)
	target_ok = _quad_valid_at_coords(target_valid.bool(), coords, state.target_shape)

	src_pos = _points_at_indices(source_xyz, source_idx)
	tgt_pos = _sample_surface_grid(target_xyz, coords)
	if normal_from_source:
		n = _points_at_indices(normal_xyz, source_idx)
	else:
		n = _sample_surface_grid(normal_xyz, coords)
	n = F.normalize(n, dim=-1, eps=1.0e-8)
	dist = ((src_pos - tgt_pos) * n).sum(dim=-1).abs()
	geom_ok = (
		torch.isfinite(src_pos).all(dim=-1) &
		torch.isfinite(tgt_pos).all(dim=-1) &
		torch.isfinite(n).all(dim=-1) &
		(n.norm(dim=-1) > 1.0e-8) &
		(dist <= float(cfg.point_distance))
	)
	source_ok = source_valid_b[new_valid_b]
	keep_flat = source_ok & target_ok & geom_ok
	new_valid_b[source_idx_b[:, 0], source_idx_b[:, 1], source_idx_b[:, 2]] = keep_flat

	pred, count, det = _local_affine_predict_batched(
		state,
		valid_b=new_valid_b,
		map_b=map_b,
		radius=cfg.affine_radius,
		exclude_self=True,
	)
	check = new_valid_b & (count >= 2)
	if bool(check.any().detach().cpu()):
		grid_err = (pred - map_b).norm(dim=-1)
		grid_pass = check & (grid_err <= float(cfg.grid_error))
		if cfg.orientation == "none":
			orient_pass = torch.ones_like(grid_pass)
		else:
			orient_pass = _affine_orientation_pass(count, det, orientation_sign=state.orientation_sign)
		grid_fail = check & ~grid_pass
		if bool(grid_fail.any().detach().cpu()):
			fail_vals = grid_err[grid_fail]
			out["pgrid"] = int(fail_vals.numel())
			out["perr_n"] = int(fail_vals.numel())
			out["perr_sum"] = float(fail_vals.sum().detach().cpu())
			out["perr_max"] = float(fail_vals.max().detach().cpu())
		new_valid_b &= (~check) | (grid_pass & orient_pass)

	after_count = int(new_valid_b.sum().detach().cpu())
	_write_batched_state(state, new_valid_b, map_b)
	out["drop"] = max(0, int(start_count) - after_count)
	return out


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


def _try_seed_reinsert(
	state: _SurfaceState,
	*,
	model_xyz: torch.Tensor,
	model_valid: torch.Tensor,
	ext_xyz: torch.Tensor,
	ext_valid: torch.Tensor,
	ext_quad_valid: torch.Tensor,
	cfg: SnapSurfConfig,
	seed_xyz: tuple[float, float, float],
) -> tuple[bool, float, float]:
	seed = torch.tensor(seed_xyz, device=ext_xyz.device, dtype=ext_xyz.dtype)
	seed_key = tuple(float(v) for v in seed_xyz)
	if state.seed_ext_distance is None or state.seed_ext_key != seed_key:
		ext_seed_hw, ext_seed_point, ext_seed_dist = _closest_external_seed_surface(
			seed=seed,
			ext_xyz=ext_xyz,
			ext_valid=ext_valid,
			ext_quad_valid=ext_quad_valid,
		)
		state.ext_seed_hw = ext_seed_hw
		state.seed_ext_distance = ext_seed_dist
		state.seed_ext_key = seed_key
		state.seed_ext_point_xyz = (
			None if ext_seed_point is None
			else tuple(float(v) for v in ext_seed_point.detach().cpu().tolist())
		)
	if state.ext_seed_hw is None or state.seed_ext_point_xyz is None:
		return False, float("inf"), state.seed_ext_distance
	eh, ew = state.ext_seed_hw
	if eh < 0 or ew < 0 or eh >= int(ext_quad_valid.shape[0]) or ew >= int(ext_quad_valid.shape[1]):
		state.ext_seed_hw = None
		return False, float("inf"), state.seed_ext_distance
	if not _bool_at_index(ext_quad_valid, (eh, ew)):
		state.ext_seed_hw = None
		return False, float("inf"), state.seed_ext_distance
	ext_seed_point_t = torch.tensor(state.seed_ext_point_xyz, device=model_xyz.device, dtype=model_xyz.dtype)
	model_quad, surface_dist = _closest_model_surface_quad(
		point=ext_seed_point_t,
		model_xyz=model_xyz,
		model_valid=model_valid,
	)
	if model_quad is None or surface_dist > cfg.init_distance:
		return False, surface_dist, state.seed_ext_distance
	d, mh, mw = model_quad
	model_quad_valid = (
		_bool_at_index(model_valid, (d, mh, mw)) and
		_bool_at_index(model_valid, (d, mh + 1, mw)) and
		_bool_at_index(model_valid, (d, mh, mw + 1)) and
		_bool_at_index(model_valid, (d, mh + 1, mw + 1))
	)
	if not model_quad_valid:
		return False, surface_dist, state.seed_ext_distance
	transform, fallback_sign = _choose_seed_transform(
		model_xyz=model_xyz,
		ext_xyz=ext_xyz,
		model_quad=model_quad,
		ext_quad=(eh, ew),
		cfg=cfg,
	)
	ext_base = torch.tensor([float(eh), float(ew)], device=model_xyz.device, dtype=model_xyz.dtype)
	model_base = torch.tensor([float(d), float(mh), float(mw)], device=model_xyz.device, dtype=model_xyz.dtype)
	model_to_ext_seed: dict[tuple[int, int], torch.Tensor] = {}
	ext_to_model_seed: dict[tuple[int, int], torch.Tensor] = {}
	for i, (sh, sw) in enumerate(_CORNERS_2D):
		th, tw = transform[i]
		model_to_ext_seed[(sh, sw)] = ext_base + torch.tensor([float(th), float(tw)], device=model_xyz.device, dtype=model_xyz.dtype)
		ext_to_model_seed[(th, tw)] = model_base + torch.tensor([0.0, float(sh), float(sw)], device=model_xyz.device, dtype=model_xyz.dtype)
	seed_source_hw = torch.tensor(_CORNERS_2D, device=model_xyz.device, dtype=model_xyz.dtype)
	m2e_targets = torch.stack([model_to_ext_seed[(sh, sw)] for sh, sw in _CORNERS_2D], dim=0)
	e2m_targets = torch.stack([ext_to_model_seed[(sh, sw)] for sh, sw in _CORNERS_2D], dim=0)
	m2e_sign = 1 if cfg.orientation in {"identity", "none"} else _affine_det_sign_from_points(
		seed_source_hw,
		m2e_targets,
		fallback=fallback_sign,
	)
	e2m_sign = 1 if cfg.orientation in {"identity", "none"} else _affine_det_sign_from_points(
		seed_source_hw,
		e2m_targets,
		fallback=fallback_sign,
	)
	for i, (sh, sw) in enumerate(_CORNERS_2D):
		model_idx = (d, mh + sh, mw + sw)
		if not _bool_at_index(model_valid, model_idx):
			continue
		_set_correspondence(state.model_to_ext, model_idx, model_to_ext_seed[(sh, sw)])
	for th, tw in _CORNERS_2D:
		ext_idx = (eh + th, ew + tw)
		if not _bool_at_index(ext_valid, ext_idx):
			continue
		_set_correspondence(state.ext_to_model, ext_idx, ext_to_model_seed[(th, tw)])
	state.model_to_ext.seed_base_idx = model_quad
	state.ext_to_model.seed_base_idx = (eh, ew)
	state.model_to_ext.orientation_sign = m2e_sign
	state.ext_to_model.orientation_sign = e2m_sign
	return True, surface_dist, state.seed_ext_distance


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


def _direction_loss_model_to_ext(
	state: _DirectionState,
	*,
	model_xyz: torch.Tensor,
	ext_xyz: torch.Tensor,
	ext_valid: torch.Tensor,
	ext_normals: torch.Tensor,
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
	coord_ok = torch.isfinite(all_coords).all(dim=-1) & _quad_valid_at_coords(ext_valid.bool(), all_coords, tuple(int(v) for v in ext_xyz.shape[:2]))
	if not bool(coord_ok.any().detach().cpu()):
		return model_xyz.new_zeros(()), lm.unsqueeze(1), mask.unsqueeze(1), 0, _empty_residual_stats()
	idx = idx[coord_ok]
	coords = all_coords[coord_ok]
	src = _points_at_indices(model_xyz, idx)
	tgt = _sample_surface_grid(ext_xyz, coords).detach()
	n_raw = _sample_surface_grid(ext_normals.detach(), coords)
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


def _direction_loss_ext_to_model(
	state: _DirectionState,
	*,
	model_xyz: torch.Tensor,
	model_valid: torch.Tensor,
	ext_normals: torch.Tensor,
	ext_xyz: torch.Tensor,
	cfg: SnapSurfConfig,
) -> tuple[torch.Tensor, int, dict[str, float]]:
	if state.map is None or state.valid is None or state.count() == 0:
		return model_xyz.new_zeros(()), 0, _empty_residual_stats()
	idx = state.valid.nonzero(as_tuple=False)
	all_coords = state.map[state.valid]
	coord_ok = torch.isfinite(all_coords).all(dim=-1) & _quad_valid_at_coords(model_valid.bool(), all_coords, tuple(int(v) for v in model_xyz.shape[:3]))
	if not bool(coord_ok.any().detach().cpu()):
		return model_xyz.new_zeros(()), 0, _empty_residual_stats()
	idx = idx[coord_ok]
	coords = all_coords[coord_ok]
	src = _points_at_indices(ext_xyz, idx).detach()
	tgt = _sample_surface_grid(model_xyz, coords)
	n_raw = _points_at_indices(ext_normals.detach(), idx)
	n = F.normalize(n_raw, dim=-1, eps=1.0e-8)
	raw_residual = ((tgt - src) * n).sum(dim=-1)
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
	stats = _residual_stats(raw_residual, residual, delta=cfg.huber_delta / cfg.distance_scale)
	return loss, int(values_f.numel()), stats


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


def _map_init_empty_stats() -> dict[str, float]:
	return {
		"snaps_seed": 0.0,
		"snaps_sdist": float("inf"),
		"snaps_sext": float("inf"),
		"snaps_m2e": 0.0,
		"snaps_map_active": 0.0,
		"snaps_map_init": 0.0,
		"snaps_map_added": 0.0,
		"snaps_map_blocked": 0.0,
		"snaps_map_sparse": 0.0,
		"snaps_map_iters": 0.0,
		"snaps_map_blocks": 0.0,
		"snaps_map_grow": 0.0,
		"snaps_map_global": 0.0,
		"snaps_map_rim": 0.0,
		"snaps_map_rim_problem": 0.0,
		"snaps_map_add_loss": 0.0,
		"snaps_map_add_bad_frac": 0.0,
		"snaps_map_add_success_frac": 0.0,
		"snaps_map_fringe_loss": 0.0,
		"snaps_map_fringe_bad_frac": 0.0,
		"snaps_map_fringe_success_frac": 0.0,
		"snaps_map_loss": 0.0,
		"snaps_map_dist": 0.0,
		"snaps_map_vec": 0.0,
		"snaps_map_norm": 0.0,
		"snaps_map_smooth": 0.0,
		"snaps_map_bend": 0.0,
		"snaps_map_jac": 0.0,
		"snaps_map_smooth_fwd": 0.0,
		"snaps_map_bend_fwd": 0.0,
		"snaps_map_jac_fwd": 0.0,
		"snaps_map_metric_smooth": 0.0,
		"snaps_map_area_smooth": 0.0,
		"snaps_map_smooth_rev": 0.0,
		"snaps_map_bend_rev": 0.0,
		"snaps_map_jac_rev": 0.0,
		"snaps_map_jinv_min": 0.0,
		"snaps_map_jinv_bad": 0.0,
		"snaps_map_jmin": 0.0,
		"snaps_map_prior": 0.0,
		"snaps_map_reg": 0.0,
		"snaps_map_jbad": 0.0,
		"snaps_map_jbadf": 0.0,
		"snaps_map_samples": 0.0,
		"snaps_map_uvbad": 0.0,
		"snaps_map_model_bad": 0.0,
		"snaps_map_step_bad": 0.0,
		"snaps_map_nsign": 1.0,
		"snaps_map_scales": 1.0,
		"snaps_map_repair": 0.0,
	}


def _map_init_sample_external_surface(
	*,
	ext_xyz: torch.Tensor,
	ext_normals: torch.Tensor,
	ext_valid: torch.Tensor,
	ext_quad_valid: torch.Tensor,
	subdiv: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
	H, W = int(ext_xyz.shape[0]), int(ext_xyz.shape[1])
	s = max(1, int(subdiv))
	device = ext_xyz.device
	dtype = ext_xyz.dtype
	if H < 2 or W < 2:
		coords = torch.empty(0, 0, 2, device=device, dtype=dtype)
		pos = torch.empty(0, 0, 3, device=device, dtype=dtype)
		normals = torch.empty(0, 0, 3, device=device, dtype=dtype)
		valid = torch.zeros(0, 0, device=device, dtype=torch.bool)
		return coords, pos, normals, valid
	Hs = (H - 1) * s
	Ws = (W - 1) * s
	hh = (torch.arange(Hs, device=device, dtype=dtype) + 0.5) / float(s)
	ww = (torch.arange(Ws, device=device, dtype=dtype) + 0.5) / float(s)
	grid_h, grid_w = torch.meshgrid(hh, ww, indexing="ij")
	coords = torch.stack([grid_h, grid_w], dim=-1)
	base_h = torch.floor(grid_h).clamp(0, H - 2).long()
	base_w = torch.floor(grid_w).clamp(0, W - 2).long()
	corner_quad_valid = (
		ext_valid[:-1, :-1].bool() &
		ext_valid[1:, :-1].bool() &
		ext_valid[:-1, 1:].bool() &
		ext_valid[1:, 1:].bool()
	)
	if tuple(ext_quad_valid.shape) == tuple(corner_quad_valid.shape):
		corner_quad_valid = corner_quad_valid & ext_quad_valid.bool()
	quad_ok = corner_quad_valid[base_h, base_w]
	pos = _sample_surface_grid(ext_xyz, coords)
	n_raw = _sample_surface_grid(ext_normals, coords)
	normals = F.normalize(n_raw, dim=-1, eps=1.0e-8)
	valid = (
		quad_ok &
		torch.isfinite(pos).all(dim=-1) &
		torch.isfinite(n_raw).all(dim=-1) &
		torch.isfinite(normals).all(dim=-1) &
		(normals.norm(dim=-1) > 1.0e-8)
	)
	return coords, pos, normals, valid


def _map_init_external_vertex_coords(ext_xyz: torch.Tensor) -> torch.Tensor:
	H, W = int(ext_xyz.shape[0]), int(ext_xyz.shape[1])
	hh = torch.arange(H, device=ext_xyz.device, dtype=ext_xyz.dtype).view(H, 1).expand(H, W)
	ww = torch.arange(W, device=ext_xyz.device, dtype=ext_xyz.dtype).view(1, W).expand(H, W)
	return torch.stack([hh, ww], dim=-1)


def _map_init_dyadic_strides(
	h_ext: int,
	w_ext: int,
	*,
	requested_levels: int,
	scale_factor: int = 2,
) -> list[int]:
	levels = max(1, int(requested_levels))
	if levels > 1 and int(scale_factor) != 2:
		raise ValueError("map-init dyadic pyramid requires scale_factor=2 when scale_levels > 1")
	qh = max(0, int(h_ext) - 1)
	qw = max(0, int(w_ext) - 1)
	out: list[int] = []
	for level in range(levels):
		stride = 1 << int(level)
		if qh % stride != 0 or qw % stride != 0:
			break
		out.append(stride)
	if not out:
		out.append(1)
	return out


def _map_init_dyadic_level_shape(h_ext: int, w_ext: int, level: int) -> tuple[int, int]:
	stride = 1 << int(level)
	qh = max(0, int(h_ext) - 1)
	qw = max(0, int(w_ext) - 1)
	if qh % stride != 0 or qw % stride != 0:
		raise ValueError(f"external quad grid {(qh, qw)} is not divisible by dyadic stride {stride}")
	return qh // stride + 1, qw // stride + 1


def _map_init_dyadic_level_coords(
	ext_xyz: torch.Tensor,
	level: int,
) -> torch.Tensor:
	H, W = _map_init_dyadic_level_shape(int(ext_xyz.shape[0]), int(ext_xyz.shape[1]), int(level))
	stride = 1 << int(level)
	hh = (torch.arange(H, device=ext_xyz.device, dtype=ext_xyz.dtype) * float(stride)).view(H, 1).expand(H, W)
	ww = (torch.arange(W, device=ext_xyz.device, dtype=ext_xyz.dtype) * float(stride)).view(1, W).expand(H, W)
	return torch.stack([hh, ww], dim=-1)


def _map_init_dyadic_level_quad_valid(
	ext_valid: torch.Tensor,
	ext_quad_valid: torch.Tensor | None,
	level: int,
) -> torch.Tensor:
	full = _map_init_external_quad_valid(ext_valid, ext_quad_valid)
	stride = 1 << int(level)
	if stride == 1:
		return full
	QH, QW = int(full.shape[0]), int(full.shape[1])
	if QH == 0 or QW == 0:
		return torch.zeros(QH // stride, QW // stride, device=ext_valid.device, dtype=torch.bool)
	if QH % stride != 0 or QW % stride != 0:
		raise ValueError(f"external quad grid {(QH, QW)} is not divisible by dyadic stride {stride}")
	x = full.to(dtype=torch.float32).reshape(1, 1, QH, QW)
	pooled = F.avg_pool2d(x, kernel_size=stride, stride=stride).reshape(QH // stride, QW // stride)
	return pooled >= 1.0


def _map_init_dyadic_level_vertex_valid(ext_valid: torch.Tensor, level: int) -> torch.Tensor:
	stride = 1 << int(level)
	return ext_valid[::stride, ::stride].bool()


def _map_init_external_quad_valid(
	ext_valid: torch.Tensor,
	ext_quad_valid: torch.Tensor | None,
) -> torch.Tensor:
	H, W = int(ext_valid.shape[0]), int(ext_valid.shape[1])
	if H < 2 or W < 2:
		return torch.zeros(max(0, H - 1), max(0, W - 1), device=ext_valid.device, dtype=torch.bool)
	out = (
		ext_valid[:-1, :-1].bool() &
		ext_valid[1:, :-1].bool() &
		ext_valid[:-1, 1:].bool() &
		ext_valid[1:, 1:].bool()
	)
	if ext_quad_valid is not None and tuple(ext_quad_valid.shape) == tuple(out.shape):
		out = out & ext_quad_valid.bool()
	return out


def _map_init_active_vertex_mask(active_quad: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
	H, W = int(shape[0]), int(shape[1])
	out = torch.zeros(H, W, device=active_quad.device, dtype=torch.bool)
	if active_quad.numel() == 0:
		return out
	q = active_quad.bool()
	out[:-1, :-1] |= q
	out[1:, :-1] |= q
	out[:-1, 1:] |= q
	out[1:, 1:] |= q
	return out


def _map_init_quad_corner_all(mask: torch.Tensor) -> torch.Tensor:
	H, W = int(mask.shape[0]), int(mask.shape[1])
	if H < 2 or W < 2:
		return torch.zeros(max(0, H - 1), max(0, W - 1), device=mask.device, dtype=torch.bool)
	return mask[:-1, :-1].bool() & mask[1:, :-1].bool() & mask[:-1, 1:].bool() & mask[1:, 1:].bool()


def _map_init_quad_offsets(*, subdiv: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
	s = max(1, int(subdiv))
	v = (torch.arange(s, device=device, dtype=dtype) + 0.5) / float(s)
	oh, ow = torch.meshgrid(v, v, indexing="ij")
	return torch.stack([oh.reshape(-1), ow.reshape(-1)], dim=-1)


def _map_init_bilerp_quad(
	v00: torch.Tensor,
	v10: torch.Tensor,
	v01: torch.Tensor,
	v11: torch.Tensor,
	offsets: torch.Tensor,
) -> torch.Tensor:
	fh = offsets[:, 0].view(1, -1, *([1] * (v00.ndim - 1)))
	fw = offsets[:, 1].view(1, -1, *([1] * (v00.ndim - 1)))
	return (
		(1.0 - fh) * (1.0 - fw) * v00.unsqueeze(1) +
		fh * (1.0 - fw) * v10.unsqueeze(1) +
		(1.0 - fh) * fw * v01.unsqueeze(1) +
		fh * fw * v11.unsqueeze(1)
	)


def _map_init_ext_quad_valid_at_coords(
	ext_quad_valid: torch.Tensor | None,
	coords: torch.Tensor,
	shape: tuple[int, int],
) -> torch.Tensor:
	if coords.numel() == 0:
		return torch.zeros(coords.shape[:-1], device=coords.device, dtype=torch.bool)
	H, W = int(shape[0]), int(shape[1])
	if H < 2 or W < 2:
		return torch.zeros(coords.shape[:-1], device=coords.device, dtype=torch.bool)
	if ext_quad_valid is None or tuple(ext_quad_valid.shape) != (H - 1, W - 1):
		return torch.ones(coords.shape[:-1], device=coords.device, dtype=torch.bool)
	flat = coords.reshape(-1, 2)
	finite = torch.isfinite(flat).all(dim=-1)
	safe = torch.where(torch.isfinite(flat), flat, torch.zeros_like(flat))
	h = safe[:, 0]
	w = safe[:, 1]
	in_bounds = finite & (h >= 0.0) & (h <= float(H - 1)) & (w >= 0.0) & (w <= float(W - 1))
	h0 = torch.floor(h.clamp(0.0, float(H - 1))).clamp(0, H - 2).long()
	w0 = torch.floor(w.clamp(0.0, float(W - 1))).clamp(0, W - 2).long()
	ok = ext_quad_valid.bool()[h0, w0] & in_bounds
	return ok.reshape(coords.shape[:-1])


def _map_init_level_external_tensors(
	*,
	ext_pos: torch.Tensor,
	ext_normals: torch.Tensor,
	ext_valid: torch.Tensor,
	ext_quad_valid: torch.Tensor | None,
	ext_coords: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
	if ext_coords is None:
		return ext_pos, ext_normals, ext_valid.bool(), ext_quad_valid
	safe = torch.where(torch.isfinite(ext_coords), ext_coords, torch.zeros_like(ext_coords))
	pos = _sample_surface_grid(ext_pos, safe)
	n_raw = _sample_surface_grid(ext_normals, safe)
	normals = F.normalize(n_raw, dim=-1, eps=1.0e-8)
	valid = (
		torch.isfinite(ext_coords).all(dim=-1) &
		_quad_valid_at_coords(ext_valid.bool(), safe, tuple(int(v) for v in ext_valid.shape)) &
		_map_init_ext_quad_valid_at_coords(ext_quad_valid, safe, tuple(int(v) for v in ext_valid.shape)) &
		torch.isfinite(pos).all(dim=-1) &
		torch.isfinite(n_raw).all(dim=-1) &
		torch.isfinite(normals).all(dim=-1) &
		(normals.norm(dim=-1) > 1.0e-8)
	)
	level_quad_valid = _map_init_quad_corner_all(valid)
	return pos, normals, valid, level_quad_valid


def _map_init_quad_sample_tensors(
	*,
	uv_full: torch.Tensor,
	ext_pos: torch.Tensor,
	ext_normals: torch.Tensor,
	ext_valid: torch.Tensor,
	ext_quad_valid: torch.Tensor | None,
	ext_coords: torch.Tensor | None = None,
	quad_hw: torch.Tensor,
	subdiv: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
	Q = int(quad_hw.shape[0])
	s = max(1, int(subdiv))
	S = s * s
	if Q == 0:
		return (
			uv_full.new_empty(0, S, 2),
			ext_pos.new_empty(0, S, 3),
			ext_normals.new_empty(0, S, 3),
			torch.zeros(0, S, device=uv_full.device, dtype=torch.bool),
			torch.zeros(0, device=uv_full.device, dtype=torch.bool),
		)
	qh = quad_hw[:, 0]
	qw = quad_hw[:, 1]
	offsets = _map_init_quad_offsets(subdiv=s, device=uv_full.device, dtype=uv_full.dtype)
	uv_samples = _map_init_bilerp_quad(
		uv_full[qh, qw],
		uv_full[qh + 1, qw],
		uv_full[qh, qw + 1],
		uv_full[qh + 1, qw + 1],
		offsets,
	)
	if ext_coords is None:
		ext_samples = _map_init_bilerp_quad(
			ext_pos[qh, qw],
			ext_pos[qh + 1, qw],
			ext_pos[qh, qw + 1],
			ext_pos[qh + 1, qw + 1],
			offsets.to(dtype=ext_pos.dtype),
		)
		n_raw = _map_init_bilerp_quad(
			ext_normals[qh, qw],
			ext_normals[qh + 1, qw],
			ext_normals[qh, qw + 1],
			ext_normals[qh + 1, qw + 1],
			offsets.to(dtype=ext_normals.dtype),
		)
		quad_valid = _map_init_external_quad_valid(ext_valid, ext_quad_valid)[qh, qw]
	else:
		sample_coords = _map_init_bilerp_quad(
			ext_coords[qh, qw],
			ext_coords[qh + 1, qw],
			ext_coords[qh, qw + 1],
			ext_coords[qh + 1, qw + 1],
			offsets.to(dtype=ext_coords.dtype),
		)
		safe_coords = torch.where(torch.isfinite(sample_coords), sample_coords, torch.zeros_like(sample_coords))
		ext_samples = _sample_surface_grid(ext_pos, safe_coords)
		n_raw = _sample_surface_grid(ext_normals, safe_coords)
		quad_valid = (
			torch.isfinite(sample_coords).all(dim=-1) &
			_quad_valid_at_coords(ext_valid.bool(), safe_coords, tuple(int(v) for v in ext_valid.shape)) &
			_map_init_ext_quad_valid_at_coords(ext_quad_valid, safe_coords, tuple(int(v) for v in ext_valid.shape))
		).all(dim=1)
	n_samples = F.normalize(n_raw, dim=-1, eps=1.0e-8)
	quad_uv_ok = (
		torch.isfinite(uv_full[qh, qw]).all(dim=-1) &
		torch.isfinite(uv_full[qh + 1, qw]).all(dim=-1) &
		torch.isfinite(uv_full[qh, qw + 1]).all(dim=-1) &
		torch.isfinite(uv_full[qh + 1, qw + 1]).all(dim=-1)
	)
	sample_ext_ok = (
		quad_valid.unsqueeze(-1) &
		torch.isfinite(ext_samples).all(dim=-1) &
		torch.isfinite(n_raw).all(dim=-1) &
		torch.isfinite(n_samples).all(dim=-1) &
		(n_samples.norm(dim=-1) > 1.0e-8)
	)
	return uv_samples, ext_samples, n_samples, sample_ext_ok, quad_uv_ok


def _map_init_model_samples_ok(
	uv_samples: torch.Tensor,
	*,
	model_valid: torch.Tensor,
	model_normals: torch.Tensor,
	depth: int,
) -> torch.Tensor:
	if uv_samples.numel() == 0:
		return torch.zeros(uv_samples.shape[:-1], device=uv_samples.device, dtype=torch.bool)
	flat_ok = _map_init_model_coord_ok(
		uv_samples.reshape(-1, 2),
		model_valid=model_valid,
		model_normals=model_normals,
		depth=int(depth),
	)
	return flat_ok.reshape(uv_samples.shape[:-1])


def _map_init_allow_partial_model_samples(scale_level: int) -> bool:
	return int(scale_level) > 0


def _map_init_mean_quad_edge_length(corners: torch.Tensor, corner_valid: torch.Tensor) -> torch.Tensor:
	if corners.numel() == 0:
		return torch.zeros(corners.shape[:1], device=corners.device, dtype=corners.dtype)
	edges = torch.stack([
		corners[:, 1] - corners[:, 0],
		corners[:, 3] - corners[:, 2],
		corners[:, 2] - corners[:, 0],
		corners[:, 3] - corners[:, 1],
	], dim=1)
	valid = torch.stack([
		corner_valid[:, 1] & corner_valid[:, 0],
		corner_valid[:, 3] & corner_valid[:, 2],
		corner_valid[:, 2] & corner_valid[:, 0],
		corner_valid[:, 3] & corner_valid[:, 1],
	], dim=1)
	length = edges.norm(dim=-1)
	valid = valid & torch.isfinite(edges).all(dim=-1) & torch.isfinite(length) & (length > 1.0e-8)
	count = valid.to(dtype=corners.dtype).sum(dim=1)
	total = torch.where(valid, length, torch.zeros_like(length)).sum(dim=1)
	return torch.where(count > 0.0, total / count.clamp_min(1.0), torch.zeros_like(total))


def _map_init_ext_quad_corner_positions(
	*,
	ext_pos: torch.Tensor,
	ext_valid: torch.Tensor,
	ext_coords: torch.Tensor | None,
	quad_hw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
	Q = int(quad_hw.shape[0])
	if Q == 0:
		return (
			ext_pos.new_empty(0, 4, 3),
			torch.zeros(0, 4, device=ext_pos.device, dtype=torch.bool),
		)
	qh = quad_hw[:, 0]
	qw = quad_hw[:, 1]
	if ext_coords is None:
		corners = torch.stack([
			ext_pos[qh, qw],
			ext_pos[qh + 1, qw],
			ext_pos[qh, qw + 1],
			ext_pos[qh + 1, qw + 1],
		], dim=1)
		corner_valid = torch.stack([
			ext_valid[qh, qw],
			ext_valid[qh + 1, qw],
			ext_valid[qh, qw + 1],
			ext_valid[qh + 1, qw + 1],
		], dim=1).bool()
	else:
		coords = torch.stack([
			ext_coords[qh, qw],
			ext_coords[qh + 1, qw],
			ext_coords[qh, qw + 1],
			ext_coords[qh + 1, qw + 1],
		], dim=1)
		safe = torch.where(torch.isfinite(coords), coords, torch.zeros_like(coords))
		corners = _sample_surface_grid(ext_pos, safe)
		corner_valid = (
			torch.isfinite(coords).all(dim=-1) &
			_quad_valid_at_coords(ext_valid.bool(), safe, tuple(int(v) for v in ext_valid.shape))
		)
	corner_valid = corner_valid & torch.isfinite(corners).all(dim=-1)
	return corners, corner_valid


def _map_init_model_quad_corner_positions(
	*,
	uv_full: torch.Tensor,
	quad_hw: torch.Tensor,
	model_xyz: torch.Tensor,
	model_valid: torch.Tensor,
	model_depth: int,
) -> tuple[torch.Tensor, torch.Tensor]:
	Q = int(quad_hw.shape[0])
	if Q == 0:
		return (
			model_xyz.new_empty(0, 4, 3),
			torch.zeros(0, 4, device=model_xyz.device, dtype=torch.bool),
		)
	qh = quad_hw[:, 0]
	qw = quad_hw[:, 1]
	uv_corners = torch.stack([
		uv_full[qh, qw],
		uv_full[qh + 1, qw],
		uv_full[qh, qw + 1],
		uv_full[qh + 1, qw + 1],
	], dim=1)
	coords = _map_init_coords3(uv_corners, depth=int(model_depth))
	safe = torch.where(torch.isfinite(coords), coords, torch.zeros_like(coords))
	corners = _sample_surface_grid(model_xyz, safe)
	corner_valid = (
		torch.isfinite(coords).all(dim=-1) &
		_quad_valid_at_coords(model_valid.bool(), safe, tuple(int(v) for v in model_valid.shape)) &
		torch.isfinite(corners).all(dim=-1)
	)
	return corners, corner_valid


def _map_init_quad_physical_step_lengths(
	*,
	uv_full: torch.Tensor,
	quad_hw: torch.Tensor,
	ext_pos: torch.Tensor,
	ext_valid: torch.Tensor,
	ext_coords: torch.Tensor | None,
	model_xyz: torch.Tensor,
	model_valid: torch.Tensor,
	model_depth: int,
) -> tuple[torch.Tensor, torch.Tensor]:
	ext_corners, ext_corner_valid = _map_init_ext_quad_corner_positions(
		ext_pos=ext_pos,
		ext_valid=ext_valid,
		ext_coords=ext_coords,
		quad_hw=quad_hw,
	)
	model_corners, model_corner_valid = _map_init_model_quad_corner_positions(
		uv_full=uv_full,
		quad_hw=quad_hw,
		model_xyz=model_xyz,
		model_valid=model_valid,
		model_depth=int(model_depth),
	)
	return (
		_map_init_mean_quad_edge_length(ext_corners, ext_corner_valid),
		_map_init_mean_quad_edge_length(model_corners, model_corner_valid),
	)


def _map_init_connection_cos_min(
	distance: torch.Tensor,
	opposing_step: torch.Tensor | None,
	*,
	cfg: SnapSurfMapInitConfig,
) -> torch.Tensor:
	base_angle = math.radians(max(0.0, min(180.0, float(cfg.max_sample_angle_deg))))
	if opposing_step is None or float(cfg.sample_angle_step_fraction) <= 0.0:
		return torch.full_like(distance, math.cos(base_angle))
	step = torch.where(
		torch.isfinite(opposing_step),
		opposing_step.clamp_min(0.0),
		torch.zeros_like(opposing_step),
	).to(device=distance.device, dtype=distance.dtype)
	extra = torch.atan2(
		float(cfg.sample_angle_step_fraction) * step,
		distance.clamp_min(1.0e-6),
	)
	cap = math.pi if base_angle > (math.pi / 2.0) else (math.pi / 2.0)
	allowed = (base_angle + extra).clamp(max=cap)
	return torch.cos(allowed)


def _map_init_sample_geometry_limit_ok(
	*,
	p_ext: torch.Tensor,
	n_ext: torch.Tensor,
	p_model: torch.Tensor,
	n_model_raw: torch.Tensor,
	normal_sign: int | None,
	cfg: SnapSurfMapInitConfig,
	ext_step: torch.Tensor | None = None,
	model_step: torch.Tensor | None = None,
) -> torch.Tensor:
	if p_ext.numel() == 0:
		return torch.zeros(p_ext.shape[:-1], device=p_ext.device, dtype=torch.bool)
	n_ext_base = F.normalize(n_ext, dim=-1, eps=1.0e-8)
	n_model = F.normalize(n_model_raw, dim=-1, eps=1.0e-8)
	v = p_model - p_ext
	d = v.norm(dim=-1)
	ok = (
		torch.isfinite(p_ext).all(dim=-1) &
		torch.isfinite(p_model).all(dim=-1) &
		torch.isfinite(n_ext).all(dim=-1) &
		torch.isfinite(n_ext_base).all(dim=-1) &
		(n_ext_base.norm(dim=-1) > 1.0e-8) &
		torch.isfinite(n_model_raw).all(dim=-1) &
		torch.isfinite(n_model).all(dim=-1) &
		(n_model.norm(dim=-1) > 1.0e-8) &
		torch.isfinite(d)
	)
	max_dist = float(cfg.max_sample_distance)
	if max_dist > 0.0:
		ok = ok & (d <= max_dist)
	max_angle = float(cfg.max_sample_angle_deg)
	if max_angle < 180.0:
		cos_min = math.cos(math.radians(max(0.0, max_angle)))
		near_zero = d <= 1.0
		u = v / d.clamp_min(1.0e-8).unsqueeze(-1)
		cos_min_ext = _map_init_connection_cos_min(d, model_step, cfg=cfg)
		cos_min_model = _map_init_connection_cos_min(d, ext_step, cfg=cfg)
		signs = (1.0, -1.0) if normal_sign is None else (1.0 if int(normal_sign) >= 0 else -1.0,)
		angle_ok = torch.zeros_like(ok)
		for sign in signs:
			n_ext_s = n_ext_base * float(sign)
			c_ext = (u * n_ext_s).sum(dim=-1)
			c_model = (u * n_model).sum(dim=-1)
			c_norm = (n_ext_s * n_model).sum(dim=-1)
			angle_ok = angle_ok | (
				torch.isfinite(c_ext) &
				torch.isfinite(c_model) &
				torch.isfinite(c_norm) &
				torch.isfinite(cos_min_ext) &
				torch.isfinite(cos_min_model) &
				((near_zero | (c_ext >= cos_min_ext)) & (near_zero | (c_model >= cos_min_model)) & (c_norm >= cos_min))
			)
		ok = ok & angle_ok
	return ok


def _map_init_candidate_quad_samples_ok(
	*,
	uv_full: torch.Tensor,
	quad_hw: torch.Tensor,
	ext_pos: torch.Tensor,
	ext_normals: torch.Tensor,
	ext_valid: torch.Tensor,
	ext_quad_valid: torch.Tensor | None,
	ext_coords: torch.Tensor | None = None,
	model_xyz: torch.Tensor | None = None,
	model_valid: torch.Tensor,
	model_normals: torch.Tensor,
	model_depth: int,
	normal_sign: int | None = 1,
	cfg: SnapSurfConfig,
	allow_partial_model_samples: bool = False,
	enforce_sample_limits: bool = True,
) -> torch.Tensor:
	uv_samples, p_ext, n_ext, ext_ok, quad_uv_ok = _map_init_quad_sample_tensors(
		uv_full=uv_full,
		ext_pos=ext_pos,
		ext_normals=ext_normals,
		ext_valid=ext_valid,
		ext_quad_valid=ext_quad_valid,
		ext_coords=ext_coords,
		quad_hw=quad_hw,
		subdiv=int(cfg.map_init.subdiv),
	)
	model_ok = _map_init_model_samples_ok(
		uv_samples,
		model_valid=model_valid,
		model_normals=model_normals,
		depth=int(model_depth),
	)
	if bool(enforce_sample_limits) and model_xyz is not None:
		coords3 = _map_init_coords3(uv_samples, depth=int(model_depth))
		safe_coords = torch.where(torch.isfinite(coords3), coords3, torch.zeros_like(coords3))
		p_model = _sample_surface_grid(model_xyz, safe_coords)
		n_model_raw = _sample_surface_grid(model_normals, safe_coords)
		ext_step, model_step = _map_init_quad_physical_step_lengths(
			uv_full=uv_full,
			quad_hw=quad_hw,
			ext_pos=ext_pos,
			ext_valid=ext_valid,
			ext_coords=ext_coords,
			model_xyz=model_xyz,
			model_valid=model_valid,
			model_depth=int(model_depth),
		)
		model_ok = model_ok & _map_init_sample_geometry_limit_ok(
			p_ext=p_ext,
			n_ext=n_ext,
			p_model=p_model,
			n_model_raw=n_model_raw,
			normal_sign=normal_sign,
			cfg=cfg.map_init,
			ext_step=ext_step[:, None].expand_as(model_ok),
			model_step=model_step[:, None].expand_as(model_ok),
		)
	base_ok = ext_ok & torch.isfinite(uv_samples).all(dim=-1)
	if bool(allow_partial_model_samples):
		return quad_uv_ok & base_ok.all(dim=1) & (base_ok & model_ok).any(dim=1)
	return quad_uv_ok & (base_ok & model_ok).all(dim=1)


def _map_init_coords3(uv: torch.Tensor, *, depth: int) -> torch.Tensor:
	d = torch.full((*uv.shape[:-1], 1), float(depth), device=uv.device, dtype=uv.dtype)
	return torch.cat([d, uv], dim=-1)


def _map_init_model_coord_ok(
	uv: torch.Tensor,
	*,
	model_valid: torch.Tensor,
	model_normals: torch.Tensor,
	depth: int,
) -> torch.Tensor:
	if uv.numel() == 0:
		return torch.zeros(uv.shape[:-1], device=uv.device, dtype=torch.bool)
	coords3 = _map_init_coords3(uv, depth=depth)
	finite = torch.isfinite(coords3).all(dim=-1)
	safe_coords = torch.where(torch.isfinite(coords3), coords3, torch.zeros_like(coords3))
	coord_ok = _quad_valid_at_coords(
		model_valid.bool(),
		safe_coords,
		tuple(int(v) for v in model_valid.shape),
	) & finite
	n_raw = _sample_surface_grid(model_normals, safe_coords)
	n = F.normalize(n_raw, dim=-1, eps=1.0e-8)
	return (
		coord_ok &
		torch.isfinite(n_raw).all(dim=-1) &
		torch.isfinite(n).all(dim=-1) &
		(n.norm(dim=-1) > 1.0e-8)
	)


def _map_init_clamp_uv(uv: torch.Tensor, *, model_h: int, model_w: int) -> torch.Tensor:
	h = uv[..., 0].clamp(0.0, float(max(0, int(model_h) - 1)))
	w = uv[..., 1].clamp(0.0, float(max(0, int(model_w) - 1)))
	return torch.stack([h, w], dim=-1)


def _map_init_scale_shapes(h: int, w: int, *, levels: int, factor: int) -> list[tuple[int, int]]:
	out: list[tuple[int, int]] = [(max(1, int(h)), max(1, int(w)))]
	step = max(2, int(factor))
	for _ in range(1, max(1, int(levels))):
		prev_h, prev_w = out[-1]
		if prev_h <= 1 and prev_w <= 1:
			break
		out.append((max(1, (prev_h + step - 1) // step), max(1, (prev_w + step - 1) // step)))
	return out


def _map_init_integrate_uv_pyr(pyr: torch.nn.ParameterList) -> torch.Tensor:
	v = pyr[-1]
	for d in reversed(pyr[:-1]):
		v = F.interpolate(v, size=(int(d.shape[2]), int(d.shape[3])), mode="bilinear", align_corners=True) + d
	return v.permute(0, 2, 3, 1).squeeze(0).contiguous()


def _map_init_make_zero_uv_pyramid(
	*,
	ext_xyz: torch.Tensor,
	strides: list[int],
	dtype: torch.dtype,
) -> list[torch.Tensor]:
	pyr: list[torch.Tensor] = []
	for level, _stride in enumerate(strides):
		H, W = _map_init_dyadic_level_shape(int(ext_xyz.shape[0]), int(ext_xyz.shape[1]), int(level))
		pyr.append(torch.zeros(1, 2, H, W, device=ext_xyz.device, dtype=dtype))
	return pyr


def _map_init_integrate_dyadic_uv_pyramid(
	pyr: list[torch.Tensor],
	*,
	active_level: int = 0,
) -> torch.Tensor:
	if not pyr:
		raise ValueError("empty map-init UV pyramid")
	level = int(active_level)
	if level < 0 or level >= len(pyr):
		raise ValueError(f"active_level {level} outside pyramid with {len(pyr)} levels")
	v = pyr[-1]
	for i in range(len(pyr) - 2, level - 1, -1):
		v = F.interpolate(v, size=(int(pyr[i].shape[2]), int(pyr[i].shape[3])), mode="bilinear", align_corners=True) + pyr[i]
	return v.permute(0, 2, 3, 1).squeeze(0).contiguous()


def _map_init_integrate_dyadic_uv_to_nchw(
	pyr: list[torch.Tensor],
	*,
	active_level: int,
) -> torch.Tensor:
	return _map_init_integrate_dyadic_uv_pyramid(pyr, active_level=active_level).permute(2, 0, 1).unsqueeze(0).contiguous()


def _map_init_coarser_dyadic_uv_nchw(
	pyr: list[torch.Tensor],
	*,
	level: int,
) -> torch.Tensor:
	if not pyr:
		raise ValueError("empty map-init UV pyramid")
	level_i = int(level)
	if level_i == len(pyr) - 1:
		return torch.zeros_like(pyr[level_i])
	coarse = _map_init_integrate_dyadic_uv_to_nchw(pyr, active_level=level_i + 1)
	return F.interpolate(coarse, size=(int(pyr[level_i].shape[2]), int(pyr[level_i].shape[3])), mode="bilinear", align_corners=True)


def _map_init_sync_current_uv_to_pyramid(state: _MapInitState) -> None:
	if state.uv_pyramid is None or state.uv is None:
		return
	level = int(state.scale_level)
	if level < 0 or level >= len(state.uv_pyramid):
		return
	current = _map_init_integrate_dyadic_uv_pyramid(state.uv_pyramid, active_level=level)
	valid = torch.isfinite(state.uv).all(dim=-1)
	updated = torch.where(valid.unsqueeze(-1), state.uv.detach(), current.detach())
	coarse_up = _map_init_coarser_dyadic_uv_nchw(state.uv_pyramid, level=level)
	state.uv_pyramid[level] = updated.permute(2, 0, 1).unsqueeze(0).contiguous().detach() - coarse_up.detach()


def _map_init_mask_current_uv(state: _MapInitState, uv: torch.Tensor, cfg: SnapSurfConfig) -> torch.Tensor:
	if bool(cfg.map_init.dense_opt):
		return uv
	active = state.active_quad
	if active is None:
		return uv
	active_vertex = _map_init_active_vertex_mask(active, tuple(int(v) for v in uv.shape[:2]))
	return torch.where(active_vertex.unsqueeze(-1), uv, torch.full_like(uv, float("nan")))


def _map_init_refresh_current_uv_from_pyramid(state: _MapInitState, cfg: SnapSurfConfig) -> None:
	if state.uv_pyramid is None:
		return
	uv = _map_init_integrate_dyadic_uv_pyramid(state.uv_pyramid, active_level=int(state.scale_level)).detach()
	state.uv = _map_init_mask_current_uv(state, uv, cfg)


def _map_init_set_current_level_external_coords(state: _MapInitState) -> None:
	if state.ext_pos is None:
		state.ext_coords = None
		return
	state.ext_coords = _map_init_dyadic_level_coords(state.ext_pos, int(state.scale_level)).detach()


def _map_init_repeat_quads_to_finer(active_quad: torch.Tensor) -> torch.Tensor:
	if active_quad.numel() == 0:
		return torch.zeros(
			int(active_quad.shape[0]) * 2,
			int(active_quad.shape[1]) * 2,
			device=active_quad.device,
			dtype=torch.bool,
		)
	return active_quad.bool().repeat_interleave(2, dim=0).repeat_interleave(2, dim=1)


def _map_init_full_blocks_to_coarser(active_quad: torch.Tensor) -> torch.Tensor:
	QH, QW = int(active_quad.shape[0]), int(active_quad.shape[1])
	PH, PW = QH // 2, QW // 2
	if PH == 0 or PW == 0:
		return torch.zeros(PH, PW, device=active_quad.device, dtype=torch.bool)
	block = active_quad[:PH * 2, :PW * 2].bool().reshape(PH, 2, PW, 2)
	return block.all(dim=3).all(dim=1)


def _map_init_level_quad_zero_from_pyramid(mi: _MapInitState, level: int) -> torch.Tensor | None:
	if mi.uv_pyramid is None:
		return None
	level_i = int(level)
	if level_i < 0 or level_i >= len(mi.uv_pyramid):
		return None
	p = mi.uv_pyramid[level_i]
	H, W = int(p.shape[2]), int(p.shape[3])
	return torch.zeros(max(0, H - 1), max(0, W - 1), device=p.device, dtype=torch.bool)


def _map_init_ensure_scale_masks(mi: _MapInitState) -> None:
	if mi.uv_pyramid is None:
		return
	if len(mi.scale_active_quads) == len(mi.uv_pyramid) and len(mi.scale_blocked_quads) == len(mi.uv_pyramid):
		return
	mi.scale_active_quads = []
	mi.scale_blocked_quads = []
	for level in range(len(mi.uv_pyramid)):
		z = _map_init_level_quad_zero_from_pyramid(mi, level)
		if z is None:
			mi.scale_active_quads.append(None)
			mi.scale_blocked_quads.append(None)
		else:
			mi.scale_active_quads.append(z)
			mi.scale_blocked_quads.append(torch.zeros_like(z))


def _map_init_store_current_scale_masks(mi: _MapInitState) -> None:
	if mi.active_quad is None:
		return
	_map_init_ensure_scale_masks(mi)
	level = int(mi.scale_level)
	if level < 0 or level >= len(mi.scale_active_quads):
		return
	active = mi.active_quad.detach().bool().clone()
	mi.scale_active_quads[level] = active
	blocked = mi.blocked_quad
	if blocked is None or tuple(blocked.shape) != tuple(active.shape):
		mi.scale_blocked_quads[level] = torch.zeros_like(active)
	else:
		mi.scale_blocked_quads[level] = blocked.detach().bool().clone()


def _map_init_zero_residual_for_new_quads(
	mi: _MapInitState,
	level: int,
	new_quad: torch.Tensor,
	existing_quad: torch.Tensor | None,
) -> None:
	if mi.uv_pyramid is None:
		return
	level_i = int(level)
	if level_i < 0 or level_i >= len(mi.uv_pyramid):
		return
	p = mi.uv_pyramid[level_i]
	H, W = int(p.shape[2]), int(p.shape[3])
	expected = (max(0, H - 1), max(0, W - 1))
	if tuple(int(v) for v in new_quad.shape) != expected:
		return
	new_vertex = _map_init_active_vertex_mask(new_quad.bool(), (H, W))
	if existing_quad is not None and tuple(int(v) for v in existing_quad.shape) == expected:
		existing_vertex = _map_init_active_vertex_mask(existing_quad.bool(), (H, W))
		new_vertex = new_vertex & ~existing_vertex
	if not bool(new_vertex.any().detach().cpu()):
		return
	mask = new_vertex.to(device=p.device).view(1, 1, H, W)
	mi.uv_pyramid[level_i] = torch.where(mask, torch.zeros_like(p), p).detach()


def _map_init_promote_full_active_to_coarser(mi: _MapInitState, *, from_level: int, to_level: int) -> None:
	_map_init_store_current_scale_masks(mi)
	_map_init_ensure_scale_masks(mi)
	if not mi.scale_active_quads:
		return
	start = max(0, int(from_level))
	end = min(int(to_level), len(mi.scale_active_quads) - 1)
	for level in range(start + 1, end + 1):
		finer = mi.scale_active_quads[level - 1]
		if finer is None:
			promoted = _map_init_level_quad_zero_from_pyramid(mi, level)
		else:
			promoted = _map_init_full_blocks_to_coarser(finer)
		expected = _map_init_level_quad_zero_from_pyramid(mi, level)
		if promoted is None or expected is None:
			continue
		if tuple(promoted.shape) != tuple(expected.shape):
			fixed = torch.zeros_like(expected)
			h = min(int(fixed.shape[0]), int(promoted.shape[0]))
			w = min(int(fixed.shape[1]), int(promoted.shape[1]))
			if h > 0 and w > 0:
				fixed[:h, :w] = promoted[:h, :w]
			promoted = fixed
		existing = mi.scale_active_quads[level]
		if existing is None or tuple(existing.shape) != tuple(promoted.shape):
			existing = torch.zeros_like(promoted, dtype=torch.bool)
		new_quad = promoted.bool() & ~existing.bool()
		_map_init_zero_residual_for_new_quads(mi, level, new_quad, existing)
		mi.scale_active_quads[level] = promoted.detach().bool().clone()
		mi.scale_blocked_quads[level] = torch.zeros_like(promoted, dtype=torch.bool)


def _map_init_switch_to_scale(
	state: _SurfaceState,
	cfg: SnapSurfConfig,
	level: int,
	*,
	reset_blocked: bool,
) -> bool:
	mi = state.map_init
	if mi.uv_pyramid is None:
		return False
	_map_init_store_current_scale_masks(mi)
	_map_init_ensure_scale_masks(mi)
	level_i = int(level)
	if level_i < 0 or level_i >= len(mi.uv_pyramid):
		return False
	active = mi.scale_active_quads[level_i]
	if active is None:
		active = _map_init_level_quad_zero_from_pyramid(mi, level_i)
	if active is None:
		return False
	mi.scale_level = level_i
	mi.active_quad = active.detach().bool().clone()
	blocked = mi.scale_blocked_quads[level_i] if level_i < len(mi.scale_blocked_quads) else None
	if reset_blocked or blocked is None or tuple(blocked.shape) != tuple(active.shape):
		mi.blocked_quad = torch.zeros_like(active, dtype=torch.bool)
	else:
		mi.blocked_quad = blocked.detach().bool().clone()
	mi.blocked_last_revisit_iter = int(mi.total_iters)
	mi.rim_blocks_since_global_opt = 0
	_map_init_set_current_level_external_coords(mi)
	_map_init_refresh_current_uv_from_pyramid(mi, cfg)
	return True


def _map_init_transition_to_finer(state: _SurfaceState, cfg: SnapSurfConfig) -> bool:
	mi = state.map_init
	if mi.uv_pyramid is None or mi.active_quad is None or int(mi.scale_level) <= 0:
		return False
	_map_init_sync_current_uv_to_pyramid(mi)
	old_level = int(mi.scale_level)
	old_stride = mi.current_stride()
	_map_init_store_current_scale_masks(mi)
	new_level = old_level - 1
	repeated = _map_init_repeat_quads_to_finer(mi.active_quad)
	_map_init_ensure_scale_masks(mi)
	existing = mi.scale_active_quads[new_level] if new_level < len(mi.scale_active_quads) else None
	existing_active = torch.zeros_like(repeated, dtype=torch.bool)
	if existing is not None and tuple(existing.shape) == tuple(repeated.shape):
		existing_active = existing.bool()
	new_quad = repeated.bool() & ~existing_active
	_map_init_zero_residual_for_new_quads(mi, new_level, new_quad, existing_active)
	if existing is not None and tuple(existing.shape) == tuple(repeated.shape):
		repeated = repeated | existing_active
	mi.scale_active_quads[new_level] = repeated.detach().clone()
	mi.scale_blocked_quads[new_level] = torch.zeros_like(repeated, dtype=torch.bool)
	if not _map_init_switch_to_scale(state, cfg, new_level, reset_blocked=True):
		return False
	_map_init_log(
		"scale transition "
		f"level={old_level}->{mi.scale_level} "
		f"stride={old_stride}->{mi.current_stride()} "
		f"active={mi.active_count()} "
		f"uv_shape={tuple(int(v) for v in mi.uv.shape[:2]) if mi.uv is not None else None}"
	)
	return True


def _map_init_source_to_uv_transform(
	uv: torch.Tensor,
	active_vertices: torch.Tensor,
	*,
	eps: float = 1.0e-6,
) -> torch.Tensor | None:
	valid = active_vertices.bool() & torch.isfinite(uv).all(dim=-1)
	if int(valid.sum().detach().cpu()) < 3:
		return None
	hw = valid.nonzero(as_tuple=False).to(device=uv.device, dtype=uv.dtype)
	if _svd_rank_2d(hw) < 2:
		return None
	target = uv[valid]
	A = torch.cat([hw, torch.ones(hw.shape[0], 1, device=uv.device, dtype=uv.dtype)], dim=1)
	try:
		sol = torch.linalg.lstsq(A, target).solution
	except RuntimeError:
		return None
	step = sol[:2, :]
	if not bool(torch.isfinite(step).all().detach().cpu()):
		return None
	if float(step.norm(dim=-1).min().detach().cpu()) <= float(eps):
		return None
	return step.detach()


def _map_init_finalize_dyadic_state(state: _SurfaceState, cfg: SnapSurfConfig) -> None:
	mi = state.map_init
	if mi.uv_pyramid is None or mi.active_quad is None:
		return
	target_level = max(0, min(int(mi.target_scale_level), len(mi.uv_pyramid) - 1))
	if int(mi.scale_level) < target_level:
		target_level = int(mi.scale_level)
	_map_init_sync_current_uv_to_pyramid(mi)
	while int(mi.scale_level) > target_level:
		if not _map_init_transition_to_finer(state, cfg):
			break
		_map_init_sync_current_uv_to_pyramid(mi)
	uv_full = _map_init_integrate_dyadic_uv_pyramid(mi.uv_pyramid, active_level=target_level).detach()
	mi.scale_level = target_level
	_map_init_set_current_level_external_coords(mi)
	mi.uv = _map_init_mask_current_uv(mi, uv_full, cfg)
	if mi.blocked_quad is None or tuple(mi.blocked_quad.shape) != tuple(mi.active_quad.shape):
		mi.blocked_quad = torch.zeros_like(mi.active_quad, dtype=torch.bool)
	_map_init_store_current_scale_masks(mi)


def _map_init_uv_pyr_from_masked(
	uv: torch.Tensor,
	valid: torch.Tensor,
	*,
	levels: int,
	factor: int,
) -> torch.nn.ParameterList:
	if uv.ndim != 3 or int(uv.shape[-1]) != 2:
		raise ValueError("map-init uv must be (H,W,2)")
	H, W = int(uv.shape[0]), int(uv.shape[1])
	shapes = _map_init_scale_shapes(H, W, levels=levels, factor=factor)
	valid = valid.bool() & torch.isfinite(uv).all(dim=-1)
	valid_nchw = valid.to(dtype=uv.dtype).view(1, 1, H, W)
	target0 = torch.where(valid.unsqueeze(-1), uv, torch.zeros_like(uv)).permute(2, 0, 1).unsqueeze(0).contiguous()
	targets: list[torch.Tensor] = [target0]
	valids: list[torch.Tensor] = [valid_nchw]
	for h_t, w_t in shapes[1:]:
		prev_valid = valids[-1]
		data_down = F.interpolate(targets[-1] * prev_valid, size=(int(h_t), int(w_t)), mode="bilinear", align_corners=True)
		valid_down = F.interpolate(prev_valid, size=(int(h_t), int(w_t)), mode="bilinear", align_corners=True)
		target = data_down / valid_down.clamp_min(1.0e-6)
		valid_mask = (valid_down > 0.01).to(dtype=uv.dtype)
		targets.append(target)
		valids.append(valid_mask)

	residuals: list[torch.Tensor] = [torch.empty(0, device=uv.device, dtype=uv.dtype)] * len(targets)
	recon = targets[-1]
	residuals[-1] = targets[-1]
	for i in range(len(targets) - 2, -1, -1):
		up = F.interpolate(recon, size=(int(targets[i].shape[2]), int(targets[i].shape[3])), mode="bilinear", align_corners=True)
		residuals[i] = (targets[i] - up) * valids[i]
		recon = up + residuals[i]

	out = torch.nn.ParameterList()
	for r in residuals:
		out.append(torch.nn.Parameter(r.detach().clone()))
	return out


def _map_init_uv_pyr_from_dense(
	uv: torch.Tensor,
	*,
	levels: int,
	factor: int,
) -> torch.nn.ParameterList:
	if uv.ndim != 3 or int(uv.shape[-1]) != 2:
		raise ValueError("map-init dense uv must be (H,W,2)")
	H, W = int(uv.shape[0]), int(uv.shape[1])
	shapes = _map_init_scale_shapes(H, W, levels=levels, factor=factor)
	targets: list[torch.Tensor] = [uv.permute(2, 0, 1).unsqueeze(0).contiguous()]
	for h_t, w_t in shapes[1:]:
		targets.append(F.interpolate(targets[-1], size=(int(h_t), int(w_t)), mode="bilinear", align_corners=True))

	residuals: list[torch.Tensor] = [torch.empty(0, device=uv.device, dtype=uv.dtype)] * len(targets)
	recon = targets[-1]
	residuals[-1] = targets[-1]
	for i in range(len(targets) - 2, -1, -1):
		up = F.interpolate(recon, size=(int(targets[i].shape[2]), int(targets[i].shape[3])), mode="bilinear", align_corners=True)
		residuals[i] = targets[i] - up
		recon = up + residuals[i]

	out = torch.nn.ParameterList()
	for r in residuals:
		out.append(torch.nn.Parameter(r.detach().clone()))
	return out


def _map_init_scalespace_inpaint_uv(
	uv: torch.Tensor,
	active: torch.Tensor,
	*,
	cfg: SnapSurfMapInitConfig,
	model_h: int | None = None,
	model_w: int | None = None,
) -> torch.Tensor:
	valid = active.bool() & torch.isfinite(uv).all(dim=-1)
	if uv.numel() == 0 or not bool(valid.any().detach().cpu()):
		return uv.detach().clone()
	pyr = _map_init_uv_pyr_from_masked(
		uv,
		valid,
		levels=int(cfg.scale_levels),
		factor=int(cfg.scale_factor),
	)
	with torch.no_grad():
		out = _map_init_integrate_uv_pyr(pyr).detach()
		if model_h is not None and model_w is not None:
			out = _map_init_clamp_uv(out, model_h=int(model_h), model_w=int(model_w))
		out = torch.where(torch.isfinite(out), out, torch.zeros_like(out))
		out = torch.where(valid.unsqueeze(-1), uv.detach(), out)
		return out


def _map_init_refresh_uv_guess(
	state: _SurfaceState,
	*,
	model_valid: torch.Tensor,
	cfg: SnapSurfConfig,
) -> None:
	if state.map_init.uv is None or state.map_init.active_quad is None:
		return
	H, W = int(model_valid.shape[1]), int(model_valid.shape[2])
	uv_finite = torch.isfinite(state.map_init.uv).all(dim=-1)
	active_vertex = _map_init_active_vertex_mask(state.map_init.active_quad, tuple(int(v) for v in state.map_init.uv.shape[:2]))
	if not cfg.map_init.dense_opt:
		state.map_init.uv_guess = None
		return
	if cfg.map_init.dense_opt and bool(uv_finite.all().detach().cpu()):
		state.map_init.uv_guess = _map_init_clamp_uv(
			state.map_init.uv.detach(),
			model_h=H,
			model_w=W,
		)
		return
	if int(cfg.map_init.scale_levels) <= 1:
		state.map_init.uv_guess = None
		return
	state.map_init.uv_guess = _map_init_scalespace_inpaint_uv(
		state.map_init.uv,
		active_vertex,
		cfg=cfg.map_init,
		model_h=H,
		model_w=W,
	).detach()


def _map_init_dense_seed_uv(
	state: _SurfaceState,
	*,
	model_valid: torch.Tensor,
	cfg: SnapSurfConfig,
) -> torch.Tensor:
	if state.map_init.uv is None or state.map_init.active_quad is None:
		raise RuntimeError("map-init dense seed requested before initialization")
	H, W = int(model_valid.shape[1]), int(model_valid.shape[2])
	active_vertex = _map_init_active_vertex_mask(state.map_init.active_quad, tuple(int(v) for v in state.map_init.uv.shape[:2]))
	if state.map_init.uv_guess is not None and tuple(state.map_init.uv_guess.shape[:2]) == tuple(state.map_init.uv.shape[:2]):
		seed = state.map_init.uv_guess.detach().clone()
	else:
		seed = _map_init_scalespace_inpaint_uv(
			state.map_init.uv,
			active_vertex,
			cfg=cfg.map_init,
			model_h=H,
			model_w=W,
		)
	active_finite = active_vertex & torch.isfinite(state.map_init.uv).all(dim=-1)
	seed = torch.where(active_finite.unsqueeze(-1), state.map_init.uv.detach(), seed)
	seed = torch.where(torch.isfinite(seed), seed, torch.zeros_like(seed))
	return _map_init_clamp_uv(seed, model_h=H, model_w=W)


def _map_init_distance_multiplier(
	c_ext: torch.Tensor,
	c_model: torch.Tensor,
	cfg: SnapSurfMapInitConfig,
) -> torch.Tensor:
	def angle(c: torch.Tensor) -> torch.Tensor:
		clamped = c.clamp(0.0, 1.0)
		near_one = clamped >= (1.0 - 1.0e-7)
		safe = clamped.clamp(max=1.0 - 1.0e-7)
		return torch.where(near_one, torch.zeros_like(clamped), torch.acos(safe))

	a_ext = angle(c_ext)
	a_model = angle(c_model)
	angle_sum = ((a_ext + a_model) / (math.pi / 2.0)).clamp(0.0, 2.0)
	return 1.0 + float(cfg.angle_dist_mult) * angle_sum.square()


def _map_init_jacobian_values(
	uv: torch.Tensor,
	active_quad: torch.Tensor,
	*,
	orientation_sign: int,
) -> torch.Tensor:
	H, W = int(uv.shape[0]), int(uv.shape[1])
	if H < 2 or W < 2 or active_quad.numel() == 0:
		return torch.empty(0, device=uv.device, dtype=uv.dtype)
	finite_uv = torch.isfinite(uv).all(dim=-1)
	cell = active_quad.bool() & _map_init_quad_corner_all(finite_uv)
	if not bool(cell.any().detach().cpu()):
		return torch.empty(0, device=uv.device, dtype=uv.dtype)
	p00 = uv[:-1, :-1]
	p10 = uv[1:, :-1]
	p01 = uv[:-1, 1:]
	p11 = uv[1:, 1:]
	dh0 = p10 - p00
	dh1 = p11 - p01
	dw0 = p01 - p00
	dw1 = p11 - p10
	dets = torch.stack([
		dh0[..., 0] * dw0[..., 1] - dh0[..., 1] * dw0[..., 0],
		dh0[..., 0] * dw1[..., 1] - dh0[..., 1] * dw1[..., 0],
		dh1[..., 0] * dw0[..., 1] - dh1[..., 1] * dw0[..., 0],
		dh1[..., 0] * dw1[..., 1] - dh1[..., 1] * dw1[..., 0],
	], dim=-1)
	finite = cell.unsqueeze(-1) & torch.isfinite(dets)
	if not bool(finite.any().detach().cpu()):
		return torch.empty(0, device=uv.device, dtype=uv.dtype)
	sign = 1.0 if int(orientation_sign) >= 0 else -1.0
	return (dets * sign)[finite]


def _map_init_jacobian_bad_quad_mask(
	uv: torch.Tensor,
	active_quad: torch.Tensor,
	*,
	orientation_sign: int,
	jac_margin: float,
) -> torch.Tensor:
	if active_quad.numel() == 0:
		return torch.zeros_like(active_quad, dtype=torch.bool)
	H, W = int(uv.shape[0]), int(uv.shape[1])
	bad = torch.zeros_like(active_quad, dtype=torch.bool)
	if H < 2 or W < 2:
		return bad
	finite_uv = torch.isfinite(uv).all(dim=-1)
	cell = active_quad.bool() & _map_init_quad_corner_all(finite_uv)
	if not bool(cell.any().detach().cpu()):
		return bad
	p00 = uv[:-1, :-1]
	p10 = uv[1:, :-1]
	p01 = uv[:-1, 1:]
	p11 = uv[1:, 1:]
	dh0 = p10 - p00
	dh1 = p11 - p01
	dw0 = p01 - p00
	dw1 = p11 - p10
	dets = torch.stack([
		dh0[..., 0] * dw0[..., 1] - dh0[..., 1] * dw0[..., 0],
		dh0[..., 0] * dw1[..., 1] - dh0[..., 1] * dw1[..., 0],
		dh1[..., 0] * dw0[..., 1] - dh1[..., 1] * dw0[..., 0],
		dh1[..., 0] * dw1[..., 1] - dh1[..., 1] * dw1[..., 0],
	], dim=-1)
	sign = 1.0 if int(orientation_sign) >= 0 else -1.0
	finite = cell.unsqueeze(-1) & torch.isfinite(dets)
	bad = cell & (~finite.all(dim=-1) | ((dets * sign) < float(jac_margin)).any(dim=-1))
	return bad


def _map_init_inverse_jacobian_bad_quad_mask(
	uv: torch.Tensor,
	active_quad: torch.Tensor,
	*,
	orientation_sign: int,
	jac_margin: float,
) -> torch.Tensor:
	if active_quad.numel() == 0:
		return torch.zeros_like(active_quad, dtype=torch.bool)
	H, W = int(uv.shape[0]), int(uv.shape[1])
	bad = torch.zeros_like(active_quad, dtype=torch.bool)
	if H < 2 or W < 2:
		return bad
	finite_uv = torch.isfinite(uv).all(dim=-1)
	cell = active_quad.bool() & _map_init_quad_corner_all(finite_uv)
	if not bool(cell.any().detach().cpu()):
		return bad
	dh = uv[1:, :-1] - uv[:-1, :-1]
	dw = uv[:-1, 1:] - uv[:-1, :-1]
	det = dh[..., 0] * dw[..., 1] - dh[..., 1] * dw[..., 0]
	finite = cell & torch.isfinite(dh).all(dim=-1) & torch.isfinite(dw).all(dim=-1) & torch.isfinite(det)
	sign = 1.0 if int(orientation_sign) >= 0 else -1.0
	det_signed = det * sign
	eps = max(1.0e-3, 0.1 * float(jac_margin))
	inv_det = torch.where(det_signed > eps, det_signed.clamp_min(eps).reciprocal(), torch.zeros_like(det_signed))
	bad = cell & (~finite | (inv_det < float(jac_margin)))
	return bad


def _map_init_jacobian_penalty(
	uv: torch.Tensor,
	active_quad: torch.Tensor,
	*,
	orientation_sign: int,
	jac_margin: float,
) -> torch.Tensor:
	values = _map_init_jacobian_values(uv, active_quad, orientation_sign=orientation_sign)
	if values.numel() == 0:
		finite = uv[torch.isfinite(uv)]
		if finite.numel():
			return finite.sum() * 0.0
		return torch.zeros((), device=uv.device, dtype=uv.dtype)
	return F.relu(float(jac_margin) - values).square().mean()


def _map_init_inverse_regularization_terms(
	uv: torch.Tensor,
	active_quad: torch.Tensor,
	*,
	orientation_sign: int,
	jac_margin: float,
) -> dict[str, torch.Tensor]:
	z = uv[torch.isfinite(uv)].sum() * 0.0 if uv.numel() else torch.zeros((), device=uv.device, dtype=uv.dtype)
	H, W = int(uv.shape[0]), int(uv.shape[1])
	if H < 2 or W < 2 or active_quad.numel() == 0:
		return {
			"smooth": z,
			"bend": z,
			"jac": z,
			"jac_min": z,
			"jac_bad": torch.tensor(0.0, device=uv.device, dtype=uv.dtype),
		}

	finite_uv = torch.isfinite(uv).all(dim=-1)
	cell = active_quad.bool() & _map_init_quad_corner_all(finite_uv)
	dh = uv[1:, :-1] - uv[:-1, :-1]
	dw = uv[:-1, 1:] - uv[:-1, :-1]
	det = dh[..., 0] * dw[..., 1] - dh[..., 1] * dw[..., 0]
	finite = cell & torch.isfinite(dh).all(dim=-1) & torch.isfinite(dw).all(dim=-1) & torch.isfinite(det)
	if not bool(finite.any().detach().cpu()):
		return {
			"smooth": z,
			"bend": z,
			"jac": z,
			"jac_min": z,
			"jac_bad": torch.tensor(0.0, device=uv.device, dtype=uv.dtype),
		}

	sign = 1.0 if int(orientation_sign) >= 0 else -1.0
	det_signed = det * sign
	det_v = det_signed[finite]
	dh_v = dh[finite]
	dw_v = dw[finite]
	eps = max(1.0e-3, 0.1 * float(jac_margin))
	safe_det = det_v.clamp_min(eps)
	fro2 = dh_v.square().sum(dim=-1) + dw_v.square().sum(dim=-1)
	# This is the Frobenius norm of d(source)/d(model). Identity maps to 1,
	# matching the forward smooth term's identity scale.
	smooth_rev = (0.5 * fro2 / safe_det.square()).mean()
	det_for_recip = det_v.clamp_min(eps)
	inv_det = torch.where(det_v > eps, det_for_recip.reciprocal(), torch.zeros_like(det_v))
	jac_rev = F.relu(float(jac_margin) - inv_det).square().mean()
	jac_inv_min = inv_det.min()
	jac_inv_bad = (inv_det < float(jac_margin)).sum()

	raw_safe_det = safe_det * sign
	inv_j = torch.zeros((*det.shape, 2, 2), device=uv.device, dtype=uv.dtype)
	inv_j_finite = torch.stack([
		torch.stack([dw_v[:, 1] / raw_safe_det, -dw_v[:, 0] / raw_safe_det], dim=-1),
		torch.stack([-dh_v[:, 1] / raw_safe_det, dh_v[:, 0] / raw_safe_det], dim=-1),
	], dim=-2)
	inv_j[finite] = inv_j_finite
	bend_vals: list[torch.Tensor] = []
	if int(inv_j.shape[0]) > 1:
		m = finite[1:, :] & finite[:-1, :]
		dj = inv_j[1:, :] - inv_j[:-1, :]
		if bool(m.any().detach().cpu()):
			bend_vals.append(dj.square().sum(dim=(-1, -2))[m])
	if int(inv_j.shape[1]) > 1:
		m = finite[:, 1:] & finite[:, :-1]
		dj = inv_j[:, 1:] - inv_j[:, :-1]
		if bool(m.any().detach().cpu()):
			bend_vals.append(dj.square().sum(dim=(-1, -2))[m])
	bend_rev = torch.cat(bend_vals).mean() if bend_vals else z
	return {
		"smooth": smooth_rev,
		"bend": bend_rev,
		"jac": jac_rev,
		"jac_min": jac_inv_min,
		"jac_bad": torch.tensor(float(int(jac_inv_bad.detach().cpu())), device=uv.device, dtype=uv.dtype),
	}


def _map_init_mean_square_diffs(
	pairs: list[tuple[torch.Tensor, torch.Tensor]],
	z: torch.Tensor,
) -> torch.Tensor:
	total = z
	count = torch.zeros((), device=z.device, dtype=z.dtype)
	for diff, mask in pairs:
		if diff.numel() == 0:
			continue
		finite = mask.bool() & torch.isfinite(diff)
		total = total + torch.where(finite, diff.square(), torch.zeros_like(diff)).sum()
		count = count + finite.to(dtype=z.dtype).sum()
	return torch.where(count > 0.0, total / count.clamp_min(1.0), z)


def _map_init_model_metric_positions(
	uv: torch.Tensor,
	*,
	model_xyz: torch.Tensor,
	model_valid: torch.Tensor | None,
	model_depth: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
	finite_uv = torch.isfinite(uv).all(dim=-1)
	if model_xyz.ndim == 4:
		if model_depth is None:
			return uv, finite_uv
		coords = _map_init_coords3(uv, depth=int(model_depth))
		safe_coords = torch.where(torch.isfinite(coords), coords, torch.zeros_like(coords))
		pos = _sample_surface_grid(model_xyz, safe_coords)
		valid = torch.isfinite(coords).all(dim=-1) & torch.isfinite(pos).all(dim=-1)
		if model_valid is not None:
			valid = valid & _quad_valid_at_coords(
				model_valid.bool(),
				safe_coords,
				tuple(int(v) for v in model_valid.shape),
			)
		return pos, valid
	if model_xyz.ndim == 3:
		safe_coords = torch.where(torch.isfinite(uv), uv, torch.zeros_like(uv))
		pos = _sample_surface_grid(model_xyz, safe_coords)
		valid = finite_uv & torch.isfinite(pos).all(dim=-1)
		if model_valid is not None:
			valid = valid & _quad_valid_at_coords(
				model_valid.bool(),
				safe_coords,
				tuple(int(v) for v in model_valid.shape),
			)
		return pos, valid
	return uv, finite_uv


def _map_init_long_step_mask(length: torch.Tensor, valid: torch.Tensor, *, max_ratio: float) -> torch.Tensor:
	if length.numel() == 0 or float(max_ratio) <= 0.0:
		return torch.zeros_like(valid, dtype=torch.bool)
	H, W = int(length.shape[0]), int(length.shape[1])
	if H == 0 or W == 0:
		return torch.zeros_like(valid, dtype=torch.bool)
	length_safe = torch.where(valid.bool() & torch.isfinite(length), length, torch.zeros_like(length))
	len_patch = F.unfold(length_safe.reshape(1, 1, H, W), kernel_size=3, padding=1).reshape(1, 9, H, W)[0]
	valid_patch = F.unfold(valid.to(dtype=length.dtype).reshape(1, 1, H, W), kernel_size=3, padding=1).reshape(1, 9, H, W)[0] > 0.0
	valid_patch[4] = False
	inf = torch.full_like(len_patch, float("inf"))
	neighbor_min = torch.where(valid_patch, len_patch, inf).min(dim=0).values
	has_neighbor = torch.isfinite(neighbor_min)
	return (
		valid.bool() &
		has_neighbor &
		torch.isfinite(length) &
		(length > neighbor_min.clamp_min(1.0e-6) * float(max_ratio))
	)


def _map_init_step_neighbor_bad_quad_mask(
	uv: torch.Tensor,
	active_quad: torch.Tensor,
	*,
	model_xyz: torch.Tensor,
	model_valid: torch.Tensor,
	model_depth: int,
	max_ratio: float,
) -> torch.Tensor:
	active = active_quad.bool()
	if active.numel() == 0 or float(max_ratio) <= 0.0:
		return torch.zeros_like(active, dtype=torch.bool)
	H, W = int(uv.shape[0]), int(uv.shape[1])
	if H < 2 or W < 2:
		return torch.zeros_like(active, dtype=torch.bool)
	metric_pos, metric_valid = _map_init_model_metric_positions(
		uv,
		model_xyz=model_xyz,
		model_valid=model_valid,
		model_depth=int(model_depth),
	)
	metric_safe = torch.where(metric_valid.unsqueeze(-1), metric_pos, torch.zeros_like(metric_pos))
	edge_h_active = torch.zeros(H - 1, W, device=uv.device, dtype=torch.bool)
	edge_h_active[:, :-1] |= active
	edge_h_active[:, 1:] |= active
	length_h = (metric_safe[1:, :] - metric_safe[:-1, :]).norm(dim=-1)
	valid_h = (
		edge_h_active &
		metric_valid[1:, :] &
		metric_valid[:-1, :] &
		torch.isfinite(length_h)
	)
	bad_h = _map_init_long_step_mask(length_h, valid_h, max_ratio=float(max_ratio))

	edge_w_active = torch.zeros(H, W - 1, device=uv.device, dtype=torch.bool)
	edge_w_active[:-1, :] |= active
	edge_w_active[1:, :] |= active
	length_w = (metric_safe[:, 1:] - metric_safe[:, :-1]).norm(dim=-1)
	valid_w = (
		edge_w_active &
		metric_valid[:, 1:] &
		metric_valid[:, :-1] &
		torch.isfinite(length_w)
	)
	bad_w = _map_init_long_step_mask(length_w, valid_w, max_ratio=float(max_ratio))

	bad_quad = torch.zeros_like(active, dtype=torch.bool)
	bad_quad |= bad_h[:, :-1] | bad_h[:, 1:]
	bad_quad |= bad_w[:-1, :] | bad_w[1:, :]
	return active & bad_quad


def _map_init_forward_smooth_bend_terms(
	field: torch.Tensor,
	vertex_valid: torch.Tensor,
	reg_quad: torch.Tensor,
	z: torch.Tensor,
) -> dict[str, torch.Tensor]:
	H, W = int(field.shape[0]), int(field.shape[1])
	field_safe = torch.where(vertex_valid.bool().unsqueeze(-1), field, torch.zeros_like(field))
	smooth_vals: list[torch.Tensor] = []
	if H > 1:
		edge = torch.zeros(H - 1, W, device=field.device, dtype=torch.bool)
		if reg_quad.numel():
			edge[:, :-1] |= reg_quad
			edge[:, 1:] |= reg_quad
		m = edge & vertex_valid[1:, :] & vertex_valid[:-1, :]
		dv = field_safe[1:, :] - field_safe[:-1, :]
		finite = m & torch.isfinite(dv).all(dim=-1)
		if bool(finite.any().detach().cpu()):
			smooth_vals.append(dv.square().sum(dim=-1)[finite])
	if W > 1:
		edge = torch.zeros(H, W - 1, device=field.device, dtype=torch.bool)
		if reg_quad.numel():
			edge[:-1, :] |= reg_quad
			edge[1:, :] |= reg_quad
		m = edge & vertex_valid[:, 1:] & vertex_valid[:, :-1]
		dv = field_safe[:, 1:] - field_safe[:, :-1]
		finite = m & torch.isfinite(dv).all(dim=-1)
		if bool(finite.any().detach().cpu()):
			smooth_vals.append(dv.square().sum(dim=-1)[finite])
	smooth = torch.cat(smooth_vals).mean() if smooth_vals else z

	if H > 2 and W > 2:
		m = (
			vertex_valid[1:-1, 1:-1] &
			vertex_valid[:-2, 1:-1] &
			vertex_valid[2:, 1:-1] &
			vertex_valid[1:-1, :-2] &
			vertex_valid[1:-1, 2:]
		)
		lap = (
			field_safe[:-2, 1:-1] +
			field_safe[2:, 1:-1] +
			field_safe[1:-1, :-2] +
			field_safe[1:-1, 2:] -
			4.0 * field_safe[1:-1, 1:-1]
		)
		finite = m & torch.isfinite(lap).all(dim=-1)
		bend = lap.square().sum(dim=-1)[finite].mean() if bool(finite.any().detach().cpu()) else z
	else:
		bend = z
	return {"smooth": smooth, "bend": bend}


def _map_init_reference_edge_square(
	ext_pos: torch.Tensor,
	finite_ext: torch.Tensor,
	reg_quad: torch.Tensor,
	z: torch.Tensor,
) -> torch.Tensor:
	H, W = int(ext_pos.shape[0]), int(ext_pos.shape[1])
	values: list[torch.Tensor] = []
	if H > 1:
		edge = torch.zeros(H - 1, W, device=ext_pos.device, dtype=torch.bool)
		if reg_quad.numel():
			edge[:, :-1] |= reg_quad
			edge[:, 1:] |= reg_quad
		dv = ext_pos[1:, :] - ext_pos[:-1, :]
		valid = edge & finite_ext[1:, :] & finite_ext[:-1, :] & torch.isfinite(dv).all(dim=-1)
		if bool(valid.any().detach().cpu()):
			values.append(dv.square().sum(dim=-1)[valid])
	if W > 1:
		edge = torch.zeros(H, W - 1, device=ext_pos.device, dtype=torch.bool)
		if reg_quad.numel():
			edge[:-1, :] |= reg_quad
			edge[1:, :] |= reg_quad
		dv = ext_pos[:, 1:] - ext_pos[:, :-1]
		valid = edge & finite_ext[:, 1:] & finite_ext[:, :-1] & torch.isfinite(dv).all(dim=-1)
		if bool(valid.any().detach().cpu()):
			values.append(dv.square().sum(dim=-1)[valid])
	if not values:
		return torch.ones((), device=z.device, dtype=z.dtype)
	return torch.cat(values).mean().clamp_min(1.0e-6)


def _map_init_local_evenness_terms(
	uv: torch.Tensor,
	ext_pos: torch.Tensor,
	active_quad: torch.Tensor,
	*,
	metric_pos: torch.Tensor | None = None,
	metric_valid: torch.Tensor | None = None,
	model_xyz: torch.Tensor | None = None,
	model_valid: torch.Tensor | None = None,
	model_depth: int | None = None,
	eps: float = 1.0e-6,
) -> dict[str, torch.Tensor]:
	z = uv[torch.isfinite(uv)].sum() * 0.0 if uv.numel() else torch.zeros((), device=uv.device, dtype=uv.dtype)
	H, W = int(uv.shape[0]), int(uv.shape[1])
	if H < 2 or W < 2 or active_quad.numel() == 0:
		return {"metric_smooth": z, "area_smooth": z}

	finite_uv = torch.isfinite(uv).all(dim=-1)
	finite_ext = torch.isfinite(ext_pos).all(dim=-1)
	if metric_pos is None or metric_valid is None:
		if model_xyz is not None:
			metric_pos, metric_valid = _map_init_model_metric_positions(
				uv,
				model_xyz=model_xyz,
				model_valid=model_valid,
				model_depth=model_depth,
			)
		else:
			metric_pos = uv
			metric_valid = finite_uv
	quad = active_quad.bool() & _map_init_quad_corner_all(metric_valid) & _map_init_quad_corner_all(finite_ext)
	safe_metric = torch.where(metric_valid.unsqueeze(-1), metric_pos, torch.zeros_like(metric_pos))
	safe_ext = torch.where(finite_ext.unsqueeze(-1), ext_pos, torch.zeros_like(ext_pos))
	eps_t = torch.tensor(float(eps), device=uv.device, dtype=uv.dtype)

	edge_h = torch.zeros(H - 1, W, device=uv.device, dtype=torch.bool)
	edge_h[:, :-1] |= quad
	edge_h[:, 1:] |= quad
	duv_h = safe_metric[1:, :] - safe_metric[:-1, :]
	dext_h = safe_ext[1:, :] - safe_ext[:-1, :]
	uv_len_h = duv_h.norm(dim=-1)
	ext_len_h = dext_h.norm(dim=-1)
	valid_h = (
		edge_h &
		metric_valid[1:, :] & metric_valid[:-1, :] &
		finite_ext[1:, :] & finite_ext[:-1, :] &
		torch.isfinite(uv_len_h) & torch.isfinite(ext_len_h)
	)
	scale_h = torch.log((uv_len_h + eps_t) / (ext_len_h + eps_t))

	edge_w = torch.zeros(H, W - 1, device=uv.device, dtype=torch.bool)
	edge_w[:-1, :] |= quad
	edge_w[1:, :] |= quad
	duv_w = safe_metric[:, 1:] - safe_metric[:, :-1]
	dext_w = safe_ext[:, 1:] - safe_ext[:, :-1]
	uv_len_w = duv_w.norm(dim=-1)
	ext_len_w = dext_w.norm(dim=-1)
	valid_w = (
		edge_w &
		metric_valid[:, 1:] & metric_valid[:, :-1] &
		finite_ext[:, 1:] & finite_ext[:, :-1] &
		torch.isfinite(uv_len_w) & torch.isfinite(ext_len_w)
	)
	scale_w = torch.log((uv_len_w + eps_t) / (ext_len_w + eps_t))

	metric_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
	if int(scale_h.shape[0]) > 1:
		metric_pairs.append((scale_h[1:, :] - scale_h[:-1, :], valid_h[1:, :] & valid_h[:-1, :]))
	if int(scale_h.shape[1]) > 1:
		metric_pairs.append((scale_h[:, 1:] - scale_h[:, :-1], valid_h[:, 1:] & valid_h[:, :-1]))
	if int(scale_w.shape[0]) > 1:
		metric_pairs.append((scale_w[1:, :] - scale_w[:-1, :], valid_w[1:, :] & valid_w[:-1, :]))
	if int(scale_w.shape[1]) > 1:
		metric_pairs.append((scale_w[:, 1:] - scale_w[:, :-1], valid_w[:, 1:] & valid_w[:, :-1]))
	metric_smooth = _map_init_mean_square_diffs(metric_pairs, z)

	def cross2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
		return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

	p00 = safe_metric[:-1, :-1]
	p10 = safe_metric[1:, :-1]
	p01 = safe_metric[:-1, 1:]
	p11 = safe_metric[1:, 1:]
	if int(safe_metric.shape[-1]) == 2:
		uv_area = 0.5 * cross2(p10 - p00, p01 - p00).abs() + 0.5 * cross2(p11 - p10, p11 - p01).abs()
	else:
		uv_area = (
			0.5 * torch.cross(p10 - p00, p01 - p00, dim=-1).norm(dim=-1) +
			0.5 * torch.cross(p11 - p10, p11 - p01, dim=-1).norm(dim=-1)
		)

	e00 = safe_ext[:-1, :-1]
	e10 = safe_ext[1:, :-1]
	e01 = safe_ext[:-1, 1:]
	e11 = safe_ext[1:, 1:]
	ext_area = (
		0.5 * torch.cross(e10 - e00, e01 - e00, dim=-1).norm(dim=-1) +
		0.5 * torch.cross(e11 - e10, e11 - e01, dim=-1).norm(dim=-1)
	)
	area_valid = quad & torch.isfinite(uv_area) & torch.isfinite(ext_area)
	area_scale = torch.log((uv_area + eps_t) / (ext_area + eps_t))
	area_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
	if int(area_scale.shape[0]) > 1:
		area_pairs.append((area_scale[1:, :] - area_scale[:-1, :], area_valid[1:, :] & area_valid[:-1, :]))
	if int(area_scale.shape[1]) > 1:
		area_pairs.append((area_scale[:, 1:] - area_scale[:, :-1], area_valid[:, 1:] & area_valid[:, :-1]))
	area_smooth = _map_init_mean_square_diffs(area_pairs, z)

	return {"metric_smooth": metric_smooth, "area_smooth": area_smooth}


def _map_init_local_jacobian_pass(
	uv: torch.Tensor,
	active_quad: torch.Tensor,
	*,
	h: int,
	w: int,
	orientation_sign: int,
	jac_margin: float,
) -> bool:
	QH, QW = int(active_quad.shape[0]), int(active_quad.shape[1])
	if QH == 0 or QW == 0:
		return True
	for bh in range(max(0, int(h) - 1), min(QH, int(h) + 2)):
		for bw in range(max(0, int(w) - 1), min(QW, int(w) + 2)):
			if not bool(active_quad[bh, bw].detach().cpu()):
				continue
			cell = torch.zeros_like(active_quad, dtype=torch.bool)
			cell[bh, bw] = True
			vals = _map_init_jacobian_values(uv, cell, orientation_sign=orientation_sign)
			if vals.numel() == 0:
				return False
			if float(vals.min().detach().cpu()) < float(jac_margin):
				return False
			if float(jac_margin) > 0.0 and float(vals.max().detach().cpu()) > 1.0 / float(jac_margin):
				return False
	return True


def _map_init_regularization_masks(
	*,
	active_quad: torch.Tensor,
	ext_valid: torch.Tensor,
	ext_quad_valid: torch.Tensor | None,
	uv_finite: torch.Tensor,
	cfg: SnapSurfMapInitConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
	active_vertex = _map_init_active_vertex_mask(active_quad, tuple(int(v) for v in uv_finite.shape))
	if not bool(cfg.dense_opt):
		vertex = active_vertex & ext_valid.bool() & uv_finite
		quad = active_quad.bool() & _map_init_external_quad_valid(ext_valid, ext_quad_valid) & _map_init_quad_corner_all(uv_finite)
		return vertex, quad
	band = _dilate_mask_2d(
		active_vertex.unsqueeze(0),
		radius=int(cfg.dense_reg_radius),
	)[0]
	vertex = band & ext_valid.bool() & uv_finite
	quad = _map_init_quad_corner_all(vertex) & _map_init_external_quad_valid(ext_valid, ext_quad_valid)
	return vertex, quad


def _map_init_objective(
	*,
	uv_full: torch.Tensor,
	active_quad: torch.Tensor,
	ext_pos: torch.Tensor,
	ext_normals: torch.Tensor,
	ext_valid: torch.Tensor,
	ext_quad_valid: torch.Tensor | None = None,
	ext_coords: torch.Tensor | None = None,
	model_xyz: torch.Tensor,
	model_valid: torch.Tensor,
	model_normals: torch.Tensor,
	model_depth: int,
	normal_sign: int,
	orientation_sign: int,
	cfg: SnapSurfConfig,
	w_jac_mult: float = 1.0,
	uv_prior: torch.Tensor | None = None,
	allow_partial_model_samples: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
	mi = cfg.map_init
	z = uv_full[torch.isfinite(uv_full)].sum() * 0.0 if uv_full.numel() else model_xyz.sum() * 0.0
	ext_vertex_pos, _ext_vertex_normals, ext_vertex_valid, ext_level_quad_valid = _map_init_level_external_tensors(
		ext_pos=ext_pos,
		ext_normals=ext_normals,
		ext_valid=ext_valid,
		ext_quad_valid=ext_quad_valid,
		ext_coords=ext_coords,
	)
	uv_finite = torch.isfinite(uv_full).all(dim=-1)
	active_quad = active_quad.bool()
	active_count = int(active_quad.sum().detach().cpu())
	quad_uv_ok_grid = active_quad & _map_init_quad_corner_all(uv_finite)
	active_bad_count = int((active_quad & ~quad_uv_ok_grid).sum().detach().cpu())
	finite_count = 0
	model_bad_count = 0
	sample_total_count = 0
	sample_bad_count = 0
	sample_valid_count = 0
	sample_loss = z
	sample_bad_frac = z
	sample_quad_ok_grid = torch.zeros_like(active_quad, dtype=torch.bool)
	quad_hw = active_quad.nonzero(as_tuple=False)
	if int(quad_hw.shape[0]) > 0:
		uv_samples, p_ext, n_ext_raw, sample_ext_ok, quad_uv_ok = _map_init_quad_sample_tensors(
			uv_full=uv_full,
			ext_pos=ext_pos,
			ext_normals=ext_normals,
			ext_valid=ext_valid,
			ext_quad_valid=ext_quad_valid,
			ext_coords=ext_coords,
			quad_hw=quad_hw,
			subdiv=int(mi.subdiv),
		)
		Q, S = int(uv_samples.shape[0]), int(uv_samples.shape[1])
		uv = uv_samples.reshape(Q * S, 2)
		coords3 = _map_init_coords3(uv, depth=model_depth)
		safe_coords = torch.where(torch.isfinite(coords3), coords3, torch.zeros_like(coords3))
		p_ext_f = p_ext.reshape(Q * S, 3)
		n_ext_raw_f = n_ext_raw.reshape(Q * S, 3)
		n_ext = F.normalize(n_ext_raw_f, dim=-1, eps=1.0e-8) * (1.0 if int(normal_sign) >= 0 else -1.0)
		p_model = _sample_surface_grid(model_xyz, safe_coords)
		n_model_raw = _sample_surface_grid(model_normals, safe_coords)
		n_model = F.normalize(n_model_raw, dim=-1, eps=1.0e-8)
		ext_step_q, model_step_q = _map_init_quad_physical_step_lengths(
			uv_full=uv_full,
			quad_hw=quad_hw,
			ext_pos=ext_pos,
			ext_valid=ext_valid,
			ext_coords=ext_coords,
			model_xyz=model_xyz,
			model_valid=model_valid,
			model_depth=int(model_depth),
		)
		ext_step_f = ext_step_q[:, None].expand(Q, S).reshape(Q * S)
		model_step_f = model_step_q[:, None].expand(Q, S).reshape(Q * S)
		coord_ok = _quad_valid_at_coords(
			model_valid.bool(),
			safe_coords,
			tuple(int(v) for v in model_valid.shape),
		)
		v = p_model - p_ext_f
		d = v.norm(dim=-1)
		u = v / d.clamp_min(1.0e-8).unsqueeze(-1)
		c_ext = (u * n_ext).sum(dim=-1)
		c_model = (u * n_model).sum(dim=-1)
		c_norm = (n_ext * n_model).sum(dim=-1)
		sample_limit_ok = _map_init_sample_geometry_limit_ok(
			p_ext=p_ext_f,
			n_ext=n_ext_raw_f,
			p_model=p_model,
			n_model_raw=n_model_raw,
			normal_sign=normal_sign,
			cfg=mi,
			ext_step=ext_step_f,
			model_step=model_step_f,
		)
		dist_mult = _map_init_distance_multiplier(c_ext, c_model, mi)
		dist_values = _huber(d, delta=cfg.huber_delta) * dist_mult
		vec_values = (1.0 - c_ext) + (1.0 - c_model)
		norm_values = 1.0 - c_norm
		base_finite = (
			sample_ext_ok.reshape(Q * S) &
			quad_uv_ok[:, None].expand(Q, S).reshape(Q * S) &
			torch.isfinite(uv).all(dim=-1) &
			torch.isfinite(p_ext_f).all(dim=-1) &
			torch.isfinite(n_ext_raw_f).all(dim=-1) &
			torch.isfinite(n_ext).all(dim=-1) &
			(n_ext.norm(dim=-1) > 1.0e-8)
		)
		model_finite = (
			coord_ok &
			torch.isfinite(p_model).all(dim=-1) &
			torch.isfinite(n_model_raw).all(dim=-1) &
			torch.isfinite(n_model).all(dim=-1) &
			(n_model.norm(dim=-1) > 1.0e-8) &
			torch.isfinite(dist_values) &
			torch.isfinite(vec_values) &
			torch.isfinite(norm_values)
		)
		finite = base_finite & model_finite
		limited_finite = finite & sample_limit_ok
		finite_qs = finite.reshape(Q, S)
		limited_finite_qs = limited_finite.reshape(Q, S)
		base_finite_qs = base_finite.reshape(Q, S)
		if bool(allow_partial_model_samples):
			loss_quad = base_finite_qs.all(dim=1) & finite_qs.any(dim=1)
			valid_quad = base_finite_qs.all(dim=1) & limited_finite_qs.any(dim=1)
		else:
			loss_quad = finite_qs.all(dim=1)
			valid_quad = limited_finite_qs.all(dim=1)
		sample_quad_ok_grid[quad_hw[:, 0], quad_hw[:, 1]] = valid_quad
		sample_total_count = Q * S
		sample_valid_count = int(finite.sum().detach().cpu())
		sample_bad_count = int((~limited_finite).sum().detach().cpu())
		sample_bad_frac = torch.tensor(
			float(sample_bad_count) / float(max(1, sample_total_count)),
			device=uv_full.device,
			dtype=uv_full.dtype,
		)
		if bool(finite.any().detach().cpu()):
			sample_values = (
				float(mi.w_dist) * dist_values +
				float(mi.w_vec_normal) * vec_values +
				float(mi.w_surface_normal) * norm_values
			)
			sample_loss = sample_values[finite].mean()
		if bool(loss_quad.any().detach().cpu()):
			loss_sample = finite_qs & loss_quad.unsqueeze(1)
			loss_count = loss_sample.to(dtype=uv_full.dtype).sum(dim=1).clamp_min(1.0)
			finite_count = int(loss_sample.sum().detach().cpu())
			model_bad_count = int((~valid_quad).sum().detach().cpu())
			dist_grid = dist_values.reshape(Q, S)
			vec_grid = vec_values.reshape(Q, S)
			norm_grid = norm_values.reshape(Q, S)
			d_grid = d.reshape(Q, S)
			dist_q_all = torch.where(loss_sample, dist_grid, dist_grid.new_zeros(Q, S)).sum(dim=1) / loss_count
			vec_q_all = torch.where(loss_sample, vec_grid, vec_grid.new_zeros(Q, S)).sum(dim=1) / loss_count
			norm_q_all = torch.where(loss_sample, norm_grid, norm_grid.new_zeros(Q, S)).sum(dim=1) / loss_count
			d_q_all = torch.where(loss_sample, d_grid, d_grid.new_zeros(Q, S)).sum(dim=1) / loss_count
			dist_q = dist_q_all[loss_quad]
			vec_q = vec_q_all[loss_quad]
			norm_q = norm_q_all[loss_quad]
			d_q = d_q_all[loss_quad]
			dist_loss = dist_q.mean()
			vec_loss = vec_q.mean()
			norm_loss = norm_q.mean()
			dist_avg = d_q.mean()
		else:
			model_bad_count = active_count
			dist_loss = z
			vec_loss = z
			norm_loss = z
			dist_avg = z
	else:
		dist_loss = z
		vec_loss = z
		norm_loss = z
		dist_avg = z

	reg_finite, reg_quad = _map_init_regularization_masks(
		active_quad=active_quad,
		ext_valid=ext_vertex_valid,
		ext_quad_valid=ext_level_quad_valid,
		uv_finite=uv_finite,
		cfg=mi,
	)
	reg_count = int(reg_finite.sum().detach().cpu())
	uv_safe = torch.where(reg_finite.unsqueeze(-1), uv_full, torch.zeros_like(uv_full))
	model_metric_pos, model_metric_valid = _map_init_model_metric_positions(
		uv_safe,
		model_xyz=model_xyz,
		model_valid=model_valid,
		model_depth=model_depth,
	)
	model_metric_valid = model_metric_valid & reg_finite
	model_metric_safe = torch.where(
		model_metric_valid.unsqueeze(-1),
		model_metric_pos,
		torch.zeros_like(model_metric_pos),
	)
	uv_fwd_terms = _map_init_forward_smooth_bend_terms(uv_safe, reg_finite, reg_quad, z)
	model_raw_fwd_terms = _map_init_forward_smooth_bend_terms(model_metric_safe, model_metric_valid, reg_quad, z)
	physical_ref2 = _map_init_reference_edge_square(
		ext_vertex_pos,
		torch.isfinite(ext_vertex_pos).all(dim=-1) & ext_vertex_valid,
		reg_quad,
		z,
	)
	smooth_uv_fwd_loss = uv_fwd_terms["smooth"]
	bend_uv_fwd_loss = uv_fwd_terms["bend"]
	smooth_model_fwd_loss = model_raw_fwd_terms["smooth"] / physical_ref2
	bend_model_fwd_loss = model_raw_fwd_terms["bend"] / physical_ref2
	smooth_fwd_loss = smooth_uv_fwd_loss + smooth_model_fwd_loss
	bend_fwd_loss = bend_uv_fwd_loss + bend_model_fwd_loss

	jac_fwd_loss = _map_init_jacobian_penalty(
		uv_safe,
		reg_quad,
		orientation_sign=orientation_sign,
		jac_margin=mi.jac_margin,
	)
	inv_terms = _map_init_inverse_regularization_terms(
		uv_safe,
		reg_quad,
		orientation_sign=orientation_sign,
		jac_margin=mi.jac_margin,
	)
	even_terms = _map_init_local_evenness_terms(
		uv_safe,
		ext_vertex_pos,
		reg_quad,
		metric_pos=model_metric_pos,
		metric_valid=model_metric_valid,
	)
	metric_smooth_loss = even_terms["metric_smooth"]
	area_smooth_loss = even_terms["area_smooth"]
	smooth_rev_loss = inv_terms["smooth"]
	bend_rev_loss = inv_terms["bend"]
	jac_rev_loss = inv_terms["jac"]
	smooth_loss = smooth_fwd_loss + smooth_rev_loss
	bend_loss = bend_fwd_loss + bend_rev_loss
	jac_loss = jac_fwd_loss + jac_rev_loss
	jac_vals = _map_init_jacobian_values(uv_safe, reg_quad, orientation_sign=orientation_sign)
	jac_min = jac_vals.min() if jac_vals.numel() else z
	if jac_vals.numel():
		jac_bad = jac_vals < float(mi.jac_margin)
		jac_bad_count = int(jac_bad.sum().detach().cpu())
		jac_bad_frac = float(jac_bad_count) / float(max(1, int(jac_vals.numel())))
	else:
		jac_bad_count = 0
		jac_bad_frac = 0.0
	jac_bad_quad_grid = _map_init_jacobian_bad_quad_mask(
		uv_safe,
		reg_quad,
		orientation_sign=orientation_sign,
		jac_margin=mi.jac_margin,
	)
	jac_inv_bad_quad_grid = _map_init_inverse_jacobian_bad_quad_mask(
		uv_safe,
		reg_quad,
		orientation_sign=orientation_sign,
		jac_margin=mi.jac_margin,
	)
	step_bad_quad_grid = _map_init_step_neighbor_bad_quad_mask(
		uv_safe,
		reg_quad,
		model_xyz=model_xyz,
		model_valid=model_valid,
		model_depth=int(model_depth),
		max_ratio=float(mi.max_step_neighbor_ratio),
	)
	quad_success_grid = (
		active_quad &
		quad_uv_ok_grid &
		sample_quad_ok_grid &
		~jac_bad_quad_grid &
		~jac_inv_bad_quad_grid &
		~step_bad_quad_grid
	)
	quad_success_count = int(quad_success_grid.sum().detach().cpu())
	quad_success_frac = torch.tensor(
		float(quad_success_count) / float(max(1, active_count)),
		device=uv_full.device,
		dtype=uv_full.dtype,
	)
	if bool(mi.dense_opt) and uv_prior is not None:
		prior_finite = reg_finite & torch.isfinite(uv_prior).all(dim=-1)
		if bool(prior_finite.any().detach().cpu()):
			prior_loss = (uv_full - uv_prior).square().sum(dim=-1)[prior_finite].mean()
		else:
			prior_loss = z
	else:
		prior_loss = z
	loss = (
		float(mi.w_dist) * dist_loss +
		float(mi.w_vec_normal) * vec_loss +
		float(mi.w_surface_normal) * norm_loss +
		float(mi.w_smooth) * smooth_loss +
		float(mi.w_bend) * bend_loss +
		float(mi.w_jac) * float(w_jac_mult) * jac_loss +
		float(mi.w_metric_smooth) * metric_smooth_loss +
		float(mi.w_area_smooth) * area_smooth_loss +
		float(mi.w_dense_prior) * prior_loss
	)
	return loss, {
		"loss": loss.detach(),
		"dist": dist_loss.detach(),
		"vec": vec_loss.detach(),
		"norm": norm_loss.detach(),
		"smooth": smooth_loss.detach(),
		"bend": bend_loss.detach(),
		"jac": jac_loss.detach(),
		"smooth_fwd": smooth_fwd_loss.detach(),
		"bend_fwd": bend_fwd_loss.detach(),
		"smooth_uv_fwd": smooth_uv_fwd_loss.detach(),
		"bend_uv_fwd": bend_uv_fwd_loss.detach(),
		"smooth_model_fwd": smooth_model_fwd_loss.detach(),
		"bend_model_fwd": bend_model_fwd_loss.detach(),
		"jac_fwd": jac_fwd_loss.detach(),
		"metric_smooth": metric_smooth_loss.detach(),
		"area_smooth": area_smooth_loss.detach(),
		"smooth_rev": smooth_rev_loss.detach(),
		"bend_rev": bend_rev_loss.detach(),
		"jac_rev": jac_rev_loss.detach(),
		"jac_min": jac_min.detach(),
		"jac_inv_min": inv_terms["jac_min"].detach(),
		"prior": prior_loss.detach(),
		"dist_avg": dist_avg.detach(),
		"active": torch.tensor(float(active_count), device=uv_full.device, dtype=uv_full.dtype),
		"reg": torch.tensor(float(reg_count), device=uv_full.device, dtype=uv_full.dtype),
		"samples": torch.tensor(float(finite_count), device=uv_full.device, dtype=uv_full.dtype),
		"sample_loss": sample_loss.detach(),
		"sample_total": torch.tensor(float(sample_total_count), device=uv_full.device, dtype=uv_full.dtype),
		"sample_valid": torch.tensor(float(sample_valid_count), device=uv_full.device, dtype=uv_full.dtype),
		"sample_bad": torch.tensor(float(sample_bad_count), device=uv_full.device, dtype=uv_full.dtype),
		"sample_bad_frac": sample_bad_frac.detach(),
		"quad_total": torch.tensor(float(active_count), device=uv_full.device, dtype=uv_full.dtype),
		"quad_success": torch.tensor(float(quad_success_count), device=uv_full.device, dtype=uv_full.dtype),
		"quad_success_frac": quad_success_frac.detach(),
		"uv_bad": torch.tensor(float(active_bad_count), device=uv_full.device, dtype=uv_full.dtype),
		"model_bad": torch.tensor(float(model_bad_count), device=uv_full.device, dtype=uv_full.dtype),
		"jac_bad": torch.tensor(float(jac_bad_count), device=uv_full.device, dtype=uv_full.dtype),
		"jac_bad_frac": torch.tensor(float(jac_bad_frac), device=uv_full.device, dtype=uv_full.dtype),
		"jac_bad_quad": torch.tensor(float(int(jac_bad_quad_grid.sum().detach().cpu())), device=uv_full.device, dtype=uv_full.dtype),
		"jac_inv_bad": inv_terms["jac_bad"].detach(),
		"jac_inv_bad_quad": torch.tensor(float(int(jac_inv_bad_quad_grid.sum().detach().cpu())), device=uv_full.device, dtype=uv_full.dtype),
		"step_bad_quad": torch.tensor(float(int(step_bad_quad_grid.sum().detach().cpu())), device=uv_full.device, dtype=uv_full.dtype),
	}


def _map_init_estimate_normal_sign(
	*,
	active_quad: torch.Tensor,
	uv: torch.Tensor,
	ext_pos: torch.Tensor,
	ext_normals: torch.Tensor,
	ext_valid: torch.Tensor,
	ext_quad_valid: torch.Tensor | None,
	ext_coords: torch.Tensor | None = None,
	model_normals: torch.Tensor,
	model_depth: int,
	subdiv: int,
) -> int:
	quad_hw = active_quad.bool().nonzero(as_tuple=False)
	if int(quad_hw.shape[0]) == 0:
		return 1
	uv_samples, _p_ext, n_ext_samples, sample_ext_ok, quad_uv_ok = _map_init_quad_sample_tensors(
		uv_full=uv,
		ext_pos=ext_pos,
		ext_normals=ext_normals,
		ext_valid=ext_valid,
		ext_quad_valid=ext_quad_valid,
		ext_coords=ext_coords,
		quad_hw=quad_hw,
		subdiv=int(subdiv),
	)
	mask = sample_ext_ok & quad_uv_ok.unsqueeze(-1) & torch.isfinite(uv_samples).all(dim=-1)
	if not bool(mask.any().detach().cpu()):
		return 1
	uv_flat = uv_samples[mask]
	coords3 = _map_init_coords3(uv_flat, depth=model_depth)
	safe_coords = torch.where(torch.isfinite(coords3), coords3, torch.zeros_like(coords3))
	n_ext = F.normalize(n_ext_samples[mask], dim=-1, eps=1.0e-8)
	n_model = F.normalize(_sample_surface_grid(model_normals, safe_coords), dim=-1, eps=1.0e-8)
	ok = (
		torch.isfinite(n_ext).all(dim=-1) &
		torch.isfinite(n_model).all(dim=-1) &
		(n_ext.norm(dim=-1) > 1.0e-8) &
		(n_model.norm(dim=-1) > 1.0e-8)
	)
	if not bool(ok.any().detach().cpu()):
		return 1
	mean_dot = (n_ext[ok] * n_model[ok]).sum(dim=-1).mean()
	return 1 if float(mean_dot.detach().cpu()) >= 0.0 else -1


def _map_init_seed_state(
	state: _SurfaceState,
	*,
	model_xyz: torch.Tensor,
	model_valid: torch.Tensor,
	model_normals: torch.Tensor,
	ext_xyz: torch.Tensor,
	ext_valid: torch.Tensor,
	ext_normals: torch.Tensor,
	ext_quad_valid: torch.Tensor,
	cfg: SnapSurfConfig,
	seed_xyz: tuple[float, float, float],
) -> tuple[bool, float, float, int]:
	mi = cfg.map_init
	H_ext, W_ext = int(ext_xyz.shape[0]), int(ext_xyz.shape[1])
	state.map_init.ext_pos = ext_xyz.detach()
	state.map_init.ext_normals = ext_normals.detach()
	state.map_init.ext_valid = ext_valid.detach()
	state.map_init.ext_quad_valid = ext_quad_valid.detach()
	strides = _map_init_dyadic_strides(
		H_ext,
		W_ext,
		requested_levels=int(mi.scale_levels),
		scale_factor=int(mi.scale_factor),
	)
	if len(strides) != int(mi.scale_levels):
		_map_init_log(
			"dyadic levels clamped "
			f"requested={int(mi.scale_levels)} "
			f"usable={len(strides)} "
			f"quad_shape={(max(0, H_ext - 1), max(0, W_ext - 1))}"
		)
	start_level = len(strides) - 1
	target_level = max(0, min(int(mi.min_scale_level), start_level))
	if target_level != int(mi.min_scale_level):
		_map_init_log(
			"min scale clamped "
			f"requested={int(mi.min_scale_level)} "
			f"usable_max={start_level} "
			f"target={target_level}"
		)
	state.map_init.scale_strides = strides
	state.map_init.target_scale_level = target_level
	state.map_init.scale_levels_used = start_level - target_level + 1
	state.map_init.scale_level = start_level
	state.map_init.uv_pyramid = _map_init_make_zero_uv_pyramid(
		ext_xyz=ext_xyz,
		strides=strides,
		dtype=model_xyz.dtype,
	)
	_map_init_ensure_scale_masks(state.map_init)
	_map_init_set_current_level_external_coords(state.map_init)
	level_H, level_W = _map_init_dyadic_level_shape(H_ext, W_ext, int(state.map_init.scale_level))
	state.map_init.active_quad = torch.zeros(max(0, level_H - 1), max(0, level_W - 1), device=model_xyz.device, dtype=torch.bool)
	state.map_init.blocked_quad = torch.zeros_like(state.map_init.active_quad)
	state.map_init.uv = torch.full((level_H, level_W, 2), float("nan"), device=model_xyz.device, dtype=model_xyz.dtype)
	_map_init_store_current_scale_masks(state.map_init)

	seed = torch.tensor(seed_xyz, device=model_xyz.device, dtype=model_xyz.dtype)
	ext_seed_hw, ext_seed_point, seed_ext_dist = _closest_external_seed_surface(
		seed=seed,
		ext_xyz=ext_xyz,
		ext_valid=ext_valid,
		ext_quad_valid=ext_quad_valid,
	)
	if ext_seed_hw is None or ext_seed_point is None:
		return False, float("inf"), float("inf"), 0
	eh, ew = ext_seed_hw

	model_quad, seed_model_dist = _closest_model_surface_quad(
		point=seed,
		model_xyz=model_xyz,
		model_valid=model_valid,
	)
	if model_quad is None:
		return False, float("inf"), seed_ext_dist, 0
	model_seed_point, model_seed_uv, seed_model_dist = _closest_point_uv_on_model_quad(
		point=seed,
		model_xyz=model_xyz,
		model_quad=model_quad,
	)
	transform, transform_sign = _choose_seed_transform(
		model_xyz=model_xyz,
		ext_xyz=ext_xyz,
		model_quad=model_quad,
		ext_quad=(eh, ew),
		cfg=cfg,
	)
	d, mh, mw = model_quad
	vertex_coords = state.map_init.ext_coords
	if vertex_coords is None:
		return False, float("inf"), seed_ext_dist, 0
	ext_level_pos, _ext_level_normals, ext_level_valid, _ext_level_quad_valid = _map_init_level_external_tensors(
		ext_pos=ext_xyz,
		ext_normals=ext_normals,
		ext_valid=ext_valid,
		ext_quad_valid=ext_quad_valid,
		ext_coords=state.map_init.ext_coords,
	)
	uv_all, uv_ok, uv_reason = _map_init_seed_quad_uv_for_points(
		ext_level_pos,
		ext_xyz=ext_xyz,
		model_xyz=model_xyz,
		ext_quad=(eh, ew),
		model_quad=model_quad,
		transform=transform,
		ext_anchor=ext_seed_point.to(device=model_xyz.device, dtype=model_xyz.dtype),
		model_anchor_uv=model_seed_uv.to(device=model_xyz.device, dtype=model_xyz.dtype),
	)
	uv_ok = uv_ok & ext_level_valid
	if not bool(uv_ok.any().detach().cpu()):
		_map_init_log(
			"seed quad uv failed "
			f"reason={uv_reason or 'no valid current-level vertices'} "
			f"ext_quad={(eh, ew)} "
			f"model_quad={model_quad}"
		)
		return False, float(seed_model_dist), seed_ext_dist, 0
	uv_all = torch.where(uv_ok.unsqueeze(-1), uv_all, torch.full_like(uv_all, float("nan")))
	_map_init_log(
		"seed quad uv "
		f"vertices={int(uv_ok.sum().detach().cpu())}/{level_H * level_W} "
		f"ext_quad={(eh, ew)} "
		f"model_quad={model_quad} "
		f"model_anchor_uv=({float(model_seed_uv[0].detach().cpu()):.6g},{float(model_seed_uv[1].detach().cpu()):.6g}) "
		f"seed_model_point=({float(model_seed_point[0].detach().cpu()):.6g},{float(model_seed_point[1].detach().cpu()):.6g},{float(model_seed_point[2].detach().cpu()):.6g})"
	)
	if state.map_init.uv_pyramid is not None:
		seed_level = torch.where(torch.isfinite(uv_all), uv_all, torch.zeros_like(uv_all))
		state.map_init.uv_pyramid[int(state.map_init.scale_level)] = seed_level.permute(2, 0, 1).unsqueeze(0).contiguous().detach()
	orientation_sign = 1 if cfg.orientation in {"identity", "none"} else int(transform_sign)
	r = max(0, int(mi.seed_radius))
	level = int(state.map_init.scale_level)
	stride = int(state.map_init.current_stride())
	allow_partial_model_samples = _map_init_allow_partial_model_samples(level)
	quad_valid = _map_init_dyadic_level_quad_valid(ext_valid, ext_quad_valid, level)
	QH, QW = int(quad_valid.shape[0]), int(quad_valid.shape[1])
	eh_level = max(0, min(max(0, QH - 1), int(eh) // max(1, stride)))
	ew_level = max(0, min(max(0, QW - 1), int(ew) // max(1, stride)))
	hh = torch.arange(QH, device=model_xyz.device).view(QH, 1)
	ww = torch.arange(QW, device=model_xyz.device).view(1, QW)
	active_quad = (
		quad_valid &
		(hh >= eh_level - r) &
		(hh <= eh_level + r) &
		(ww >= ew_level - r) &
		(ww <= ew_level + r)
	)
	cand_hw = active_quad.nonzero(as_tuple=False)
	if int(cand_hw.shape[0]) > 0:
		ok_quad = _map_init_candidate_quad_samples_ok(
			uv_full=uv_all,
			quad_hw=cand_hw,
			ext_pos=ext_xyz,
			ext_normals=ext_normals,
			ext_valid=ext_valid,
			ext_quad_valid=ext_quad_valid,
			ext_coords=state.map_init.ext_coords,
			model_xyz=model_xyz,
			model_valid=model_valid,
			model_normals=model_normals,
			model_depth=int(d),
			normal_sign=None,
			cfg=cfg,
			allow_partial_model_samples=allow_partial_model_samples,
			enforce_sample_limits=False,
		)
		active_quad[cand_hw[:, 0], cand_hw[:, 1]] = ok_quad
	if not bool(active_quad.any().detach().cpu()) and 0 <= eh_level < QH and 0 <= ew_level < QW:
		seed_quad_hw = torch.tensor([[eh_level, ew_level]], device=model_xyz.device, dtype=torch.long)
		seed_ok = bool(_map_init_candidate_quad_samples_ok(
			uv_full=uv_all,
			quad_hw=seed_quad_hw,
			ext_pos=ext_xyz,
			ext_normals=ext_normals,
			ext_valid=ext_valid,
			ext_quad_valid=ext_quad_valid,
			ext_coords=state.map_init.ext_coords,
			model_xyz=model_xyz,
			model_valid=model_valid,
			model_normals=model_normals,
			model_depth=int(d),
			normal_sign=None,
			cfg=cfg,
			allow_partial_model_samples=allow_partial_model_samples,
			enforce_sample_limits=False,
		)[0].detach().cpu())
		active_quad[eh_level, ew_level] = seed_ok
	if not allow_partial_model_samples and bool(active_quad.any().detach().cpu()):
		active_quad = active_quad & _map_init_quad_corner_all(_map_init_model_coord_ok(
			uv_all,
			model_valid=model_valid,
			model_normals=model_normals,
			depth=d,
		))
	if not bool(active_quad.any().detach().cpu()):
		return False, float(seed_model_dist), seed_ext_dist, 0
	state.map_init.active_quad = active_quad.detach()
	state.map_init.blocked_quad = torch.zeros_like(active_quad, dtype=torch.bool)
	state.map_init.uv = _map_init_mask_current_uv(state.map_init, uv_all.detach(), cfg)
	_map_init_sync_current_uv_to_pyramid(state.map_init)
	_map_init_refresh_current_uv_from_pyramid(state.map_init, cfg)
	_map_init_store_current_scale_masks(state.map_init)
	state.map_init.model_depth = int(d)
	state.map_init.seed_ext_sample_hw = (eh, ew)
	state.map_init.seed_model_quad = model_quad
	state.map_init.orientation_sign = int(orientation_sign)
	state.map_init.normal_sign = _map_init_estimate_normal_sign(
		active_quad=active_quad,
		uv=uv_all,
		ext_pos=ext_xyz,
		ext_normals=ext_normals,
		ext_valid=ext_valid,
		ext_quad_valid=ext_quad_valid,
		ext_coords=state.map_init.ext_coords,
		model_normals=model_normals,
		model_depth=int(d),
		subdiv=int(mi.subdiv),
	)
	init_count = int(active_quad.sum().detach().cpu())
	state.map_init.added_total = init_count
	_map_init_refresh_uv_guess(state, model_valid=model_valid, cfg=cfg)
	return init_count > 0, float(seed_model_dist), seed_ext_dist, init_count


def _map_init_grow_once(
	state: _SurfaceState,
	*,
	model_xyz: torch.Tensor,
	model_valid: torch.Tensor,
	model_normals: torch.Tensor,
	cfg: SnapSurfConfig,
) -> int:
	mi = cfg.map_init
	active = state.map_init.active_quad
	uv = state.map_init.uv
	ext_pos = state.map_init.ext_pos
	ext_normals = state.map_init.ext_normals
	ext_valid = state.map_init.ext_valid
	ext_quad_valid = state.map_init.ext_quad_valid
	depth = state.map_init.model_depth
	if active is None or uv is None or ext_pos is None or ext_normals is None or ext_valid is None or depth is None:
		return 0
	if not bool(active.any().detach().cpu()):
		return 0
	quad_valid = _map_init_dyadic_level_quad_valid(ext_valid, ext_quad_valid, int(state.map_init.scale_level))
	allow_partial_model_samples = _map_init_allow_partial_model_samples(int(state.map_init.scale_level))
	blocked = state.map_init.blocked_quad
	if blocked is None or tuple(blocked.shape) != tuple(active.shape):
		blocked = torch.zeros_like(active, dtype=torch.bool)
	revisit_interval = max(1, int(cfg.map_init.progress_interval))
	if (
		int(state.map_init.total_iters) - int(state.map_init.blocked_last_revisit_iter) >= revisit_interval and
		bool(blocked.any().detach().cpu())
	):
		blocked = torch.zeros_like(blocked, dtype=torch.bool)
		state.map_init.blocked_quad = blocked
		state.map_init.blocked_last_revisit_iter = int(state.map_init.total_iters)
	candidate = _neighbor8_mask(active.unsqueeze(0))[0] & ~active & ~blocked.bool() & quad_valid
	if not bool(candidate.any().detach().cpu()):
		return 0
	cand_hw = candidate.nonzero(as_tuple=False)
	active_vertices = _map_init_active_vertex_mask(active, tuple(int(v) for v in uv.shape[:2])) & torch.isfinite(uv).all(dim=-1)
	cand_vertices = _map_init_active_vertex_mask(candidate, tuple(int(v) for v in uv.shape[:2])) & ~active_vertices
	pred_grid = uv.clone()
	pred_ok_grid = active_vertices.clone()
	single_neighbor_transform = _map_init_source_to_uv_transform(uv, active_vertices)
	if bool(cand_vertices.any().detach().cpu()):
		vert_hw = cand_vertices.nonzero(as_tuple=False)
		vert_bidx = torch.cat([
			torch.zeros(vert_hw.shape[0], 1, device=vert_hw.device, dtype=torch.long),
			vert_hw,
		], dim=1)
		tmp_state = _DirectionState(source_rank=2, target_rank=2)
		pred, count, _nearest = _direct_predict_candidates_batched(
			tmp_state,
			valid_b=active_vertices.unsqueeze(0),
			map_b=uv.unsqueeze(0),
			candidate_bidx=vert_bidx,
			radius=max(1, int(mi.edge_init_radius)),
			single_neighbor_transform=single_neighbor_transform,
		)
		local_ok = (count >= 1) & torch.isfinite(pred).all(dim=-1)
		if not allow_partial_model_samples:
			local_ok = local_ok & _map_init_model_coord_ok(
				pred,
				model_valid=model_valid,
				model_normals=model_normals,
				depth=int(depth),
			)
		if single_neighbor_transform is None:
			local_ok = local_ok & (count >= 3)
		if bool(mi.dense_opt) and state.map_init.uv_guess is not None and tuple(state.map_init.uv_guess.shape[:2]) == tuple(uv.shape[:2]):
			guess = state.map_init.uv_guess[vert_hw[:, 0], vert_hw[:, 1]]
			guess_ok = torch.isfinite(guess).all(dim=-1)
			if not allow_partial_model_samples:
				guess_ok = guess_ok & _map_init_model_coord_ok(
					guess,
				model_valid=model_valid,
				model_normals=model_normals,
					depth=int(depth),
			)
			pred = torch.where(local_ok.unsqueeze(-1), pred, torch.where(guess_ok.unsqueeze(-1), guess, pred))
			local_ok = local_ok | guess_ok
		if bool(local_ok.any().detach().cpu()):
			ok_hw = vert_hw[local_ok]
			pred_grid[ok_hw[:, 0], ok_hw[:, 1]] = pred[local_ok].detach()
			pred_ok_grid[ok_hw[:, 0], ok_hw[:, 1]] = True
	candidate_seed = candidate & _map_init_quad_corner_all(pred_ok_grid & torch.isfinite(pred_grid).all(dim=-1))
	if bool(candidate_seed.any().detach().cpu()):
		_map_init_log_fringe_debug(
			state=state.map_init,
			phase="cinit",
			block=state.map_init.opt_blocks + 1,
			iter_idx=state.map_init.total_iters,
			uv_full=pred_grid,
			active_quad=candidate_seed,
			ext_pos=ext_pos,
			ext_normals=ext_normals,
			ext_valid=ext_valid,
			ext_quad_valid=ext_quad_valid,
			model_xyz=model_xyz,
			model_valid=model_valid,
			model_normals=model_normals,
			model_depth=int(depth),
			cfg=cfg,
		)
	candidate_possible = candidate_seed
	candidate_sample_reject = torch.zeros_like(candidate_seed, dtype=torch.bool)
	if bool(candidate_possible.any().detach().cpu()):
		possible_hw = candidate_possible.nonzero(as_tuple=False)
		possible_ok = _map_init_candidate_quad_samples_ok(
			uv_full=pred_grid,
			quad_hw=possible_hw,
			ext_pos=ext_pos,
			ext_normals=ext_normals,
			ext_valid=ext_valid,
			ext_quad_valid=ext_quad_valid,
			ext_coords=state.map_init.ext_coords,
			model_xyz=model_xyz,
			model_valid=model_valid,
			model_normals=model_normals,
			model_depth=int(depth),
			normal_sign=state.map_init.normal_sign,
			cfg=cfg,
			allow_partial_model_samples=allow_partial_model_samples,
			enforce_sample_limits=False,
		)
		filtered = torch.zeros_like(candidate_possible, dtype=torch.bool)
		filtered[possible_hw[:, 0], possible_hw[:, 1]] = possible_ok
		if bool((~possible_ok).any().detach().cpu()):
			bad_hw = possible_hw[~possible_ok]
			candidate_sample_reject[bad_hw[:, 0], bad_hw[:, 1]] = True
		candidate_possible = filtered
	if (
		not bool(mi.dense_opt) and int(mi.candidate_opt_iters) > 0 and
		bool(candidate_possible.any().detach().cpu())
	):
		candidate_opt_vertices = (
			_map_init_active_vertex_mask(candidate_possible, tuple(int(v) for v in uv.shape[:2])) &
			~active_vertices &
			pred_ok_grid &
			torch.isfinite(pred_grid).all(dim=-1)
		)
		if bool(candidate_opt_vertices.any().detach().cpu()):
			pred_grid, _prefit_terms = _map_init_optimize_vertex_mask(
				state,
				base_uv=pred_grid,
				active_quad=candidate_possible,
				opt_vertex_mask=candidate_opt_vertices,
				model_xyz=model_xyz,
				model_valid=model_valid,
				model_normals=model_normals,
				cfg=cfg,
				steps=int(mi.candidate_opt_iters),
				mode="add",
				lr=float(mi.candidate_lr),
			)
			pred_finite = torch.isfinite(pred_grid).all(dim=-1)
			if allow_partial_model_samples:
				pred_ok_grid = (active_vertices & torch.isfinite(uv).all(dim=-1)) | pred_finite
			else:
				pred_ok_grid = (
					(active_vertices & torch.isfinite(uv).all(dim=-1)) |
					(
						pred_finite &
						_map_init_model_coord_ok(
							pred_grid,
							model_valid=model_valid,
							model_normals=model_normals,
							depth=int(depth),
						)
					)
				)
			candidate_possible = candidate & _map_init_quad_corner_all(pred_ok_grid) & ~candidate_sample_reject
			_map_init_log_fringe_debug(
				state=state.map_init,
				phase="cand",
				block=state.map_init.opt_blocks,
				iter_idx=state.map_init.total_iters,
				uv_full=pred_grid,
				active_quad=candidate_possible,
				ext_pos=ext_pos,
				ext_normals=ext_normals,
				ext_valid=ext_valid,
				ext_quad_valid=ext_quad_valid,
				model_xyz=model_xyz,
				model_valid=model_valid,
				model_normals=model_normals,
				model_depth=int(depth),
				cfg=cfg,
			)
	active_new = active.clone()
	blocked_new = blocked.clone() | candidate_sample_reject
	uv_new = uv.clone()
	added = 0
	rejected = 0
	for i in range(int(cand_hw.shape[0])):
		h = int(cand_hw[i, 0].detach().cpu())
		w = int(cand_hw[i, 1].detach().cpu())
		if bool(active_new[h, w].detach().cpu()):
			continue
		if not bool(candidate_possible[h, w].detach().cpu()):
			continue
		corners = (
			(h, w),
			(h + 1, w),
			(h, w + 1),
			(h + 1, w + 1),
		)
		prev_uv = [uv_new[ch, cw].clone() for ch, cw in corners]
		proposed = [uv_new[ch, cw] if bool(torch.isfinite(uv_new[ch, cw]).all().detach().cpu()) else pred_grid[ch, cw] for ch, cw in corners]
		if not all(bool(torch.isfinite(p).all().detach().cpu()) for p in proposed):
			continue
		if not all(bool(pred_ok_grid[ch, cw].detach().cpu()) or bool(torch.isfinite(uv_new[ch, cw]).all().detach().cpu()) for ch, cw in corners):
			continue
		for (ch, cw), p in zip(corners, proposed, strict=False):
			uv_new[ch, cw] = p.detach()
		active_new[h, w] = True
		samples_ok = bool(_map_init_candidate_quad_samples_ok(
			uv_full=uv_new,
			quad_hw=cand_hw[i:i + 1],
			ext_pos=ext_pos,
			ext_normals=ext_normals,
			ext_valid=ext_valid,
			ext_quad_valid=ext_quad_valid,
			ext_coords=state.map_init.ext_coords,
			model_xyz=model_xyz,
			model_valid=model_valid,
			model_normals=model_normals,
			model_depth=int(depth),
			normal_sign=state.map_init.normal_sign,
			cfg=cfg,
			allow_partial_model_samples=allow_partial_model_samples,
		)[0].detach().cpu())
		jac_ok = _map_init_local_jacobian_pass(
			uv_new,
			active_new,
			h=h,
			w=w,
			orientation_sign=state.map_init.orientation_sign,
			jac_margin=mi.jac_margin,
		)
		step_bad = _map_init_step_neighbor_bad_quad_mask(
			uv_new,
			active_new,
			model_xyz=model_xyz,
			model_valid=model_valid,
			model_depth=int(depth),
			max_ratio=float(mi.max_step_neighbor_ratio),
		)
		step_ok = not bool(step_bad[h, w].detach().cpu())
		if samples_ok and jac_ok and step_ok:
			added += 1
			blocked_new[h, w] = False
		else:
			active_new[h, w] = False
			blocked_new[h, w] = True
			rejected += 1
			active_vertices_after = _map_init_active_vertex_mask(active_new, tuple(int(v) for v in uv_new.shape[:2]))
			for (ch, cw), prev in zip(corners, prev_uv, strict=False):
				if bool(mi.dense_opt) or bool(active_vertices_after[ch, cw].detach().cpu()):
					uv_new[ch, cw] = prev
				else:
					uv_new[ch, cw] = float("nan")
	if rejected > 0:
		active_new, uv_new, blocked_new, _sparse_count = _map_init_apply_sparse_quad_cleanup(
			state,
			active_new,
			uv_new,
			blocked_new,
			cfg=cfg,
		)
	new_quad = active_new.bool() & ~active.bool()
	if (
		not bool(mi.dense_opt) and int(mi.fringe_opt_iters) > 0 and
		bool(new_quad.any().detach().cpu())
	):
		fringe_vertices = (
			_map_init_active_vertex_mask(new_quad, tuple(int(v) for v in uv_new.shape[:2])) &
			~active_vertices &
			torch.isfinite(uv_new).all(dim=-1)
		)
		if bool(fringe_vertices.any().detach().cpu()):
			uv_new, _fringe_terms = _map_init_optimize_vertex_mask(
				state,
				base_uv=uv_new,
				active_quad=new_quad,
				opt_vertex_mask=fringe_vertices,
				model_xyz=model_xyz,
				model_valid=model_valid,
				model_normals=model_normals,
				cfg=cfg,
				steps=int(mi.fringe_opt_iters),
				mode="fringe",
				lr=float(mi.fringe_lr),
			)
			_map_init_log_fringe_debug(
				state=state.map_init,
				phase="fringe",
				block=state.map_init.opt_blocks,
				iter_idx=state.map_init.total_iters,
				uv_full=uv_new,
				active_quad=new_quad,
				ext_pos=ext_pos,
				ext_normals=ext_normals,
				ext_valid=ext_valid,
				ext_quad_valid=ext_quad_valid,
				model_xyz=model_xyz,
				model_valid=model_valid,
				model_normals=model_normals,
				model_depth=int(depth),
				cfg=cfg,
			)
			active_new, uv_new, blocked_new, _fringe_sample_reject, _fringe_fold_reject, _fringe_sparse = _map_init_reject_bad_new_quads(
				state,
				active_before=active,
				active_quad=active_new,
				uv=uv_new,
				blocked_quad=blocked_new,
				model_xyz=model_xyz,
				model_valid=model_valid,
				model_normals=model_normals,
				cfg=cfg,
			)
	added = int((active_new.bool() & ~active.bool()).sum().detach().cpu())
	state.map_init.active_quad = active_new
	state.map_init.blocked_quad = blocked_new
	state.map_init.uv = uv_new
	_map_init_sync_current_uv_to_pyramid(state.map_init)
	_map_init_refresh_current_uv_from_pyramid(state.map_init, cfg)
	_map_init_store_current_scale_masks(state.map_init)
	state.map_init.added_total += added
	if added > 0:
		state.map_init.grow_steps += 1
		_map_init_refresh_uv_guess(state, model_valid=model_valid, cfg=cfg)
	return added


def _map_init_term_float(terms: dict[str, torch.Tensor], key: str) -> float:
	v = terms.get(key)
	if v is None:
		return 0.0
	return float(v.detach().cpu())


def _map_init_accumulate_phase_stats(state: _MapInitState, phase: str, terms: dict[str, torch.Tensor]) -> None:
	valid = max(0.0, _map_init_term_float(terms, "sample_valid"))
	total = max(0.0, _map_init_term_float(terms, "sample_total"))
	bad = max(0.0, _map_init_term_float(terms, "sample_bad"))
	loss = _map_init_term_float(terms, "sample_loss")
	quad_success = max(0.0, _map_init_term_float(terms, "quad_success"))
	quad_total = max(0.0, _map_init_term_float(terms, "quad_total"))
	if phase == "add":
		state.add_sample_loss_sum += loss * valid
		state.add_sample_weight += valid
		state.add_bad_samples += bad
		state.add_total_samples += total
		state.add_success_quads += quad_success
		state.add_total_quads += quad_total
		state.interval_add_sample_loss_sum += loss * valid
		state.interval_add_sample_weight += valid
		state.interval_add_bad_samples += bad
		state.interval_add_total_samples += total
		state.interval_add_success_quads += quad_success
		state.interval_add_total_quads += quad_total
	elif phase == "fringe":
		state.fringe_sample_loss_sum += loss * valid
		state.fringe_sample_weight += valid
		state.fringe_bad_samples += bad
		state.fringe_total_samples += total
		state.fringe_success_quads += quad_success
		state.fringe_total_quads += quad_total
		state.interval_fringe_sample_loss_sum += loss * valid
		state.interval_fringe_sample_weight += valid
		state.interval_fringe_bad_samples += bad
		state.interval_fringe_total_samples += total
		state.interval_fringe_success_quads += quad_success
		state.interval_fringe_total_quads += quad_total


def _map_init_interval_phase_stats(state: _MapInitState, phase: str) -> tuple[float, float]:
	if phase == "add":
		loss_sum = state.interval_add_sample_loss_sum
		weight = state.interval_add_sample_weight
		quad_success = state.interval_add_success_quads
		quad_total = state.interval_add_total_quads
	elif phase == "fringe":
		loss_sum = state.interval_fringe_sample_loss_sum
		weight = state.interval_fringe_sample_weight
		quad_success = state.interval_fringe_success_quads
		quad_total = state.interval_fringe_total_quads
	else:
		return 0.0, 0.0
	loss = float(loss_sum) / float(max(1.0, weight))
	success = float(quad_success) / float(max(1.0, quad_total))
	if quad_total <= 0.0:
		success = 0.0
	return loss, success


def _map_init_reset_interval_phase_stats(state: _MapInitState) -> None:
	state.interval_add_sample_loss_sum = 0.0
	state.interval_add_sample_weight = 0.0
	state.interval_add_bad_samples = 0.0
	state.interval_add_total_samples = 0.0
	state.interval_add_success_quads = 0.0
	state.interval_add_total_quads = 0.0
	state.interval_fringe_sample_loss_sum = 0.0
	state.interval_fringe_sample_weight = 0.0
	state.interval_fringe_bad_samples = 0.0
	state.interval_fringe_total_samples = 0.0
	state.interval_fringe_success_quads = 0.0
	state.interval_fringe_total_quads = 0.0


def _map_init_block_progress_enabled(cfg: SnapSurfConfig) -> bool:
	return str(cfg.map_init.progress_mode) in ("block", "both")


def _map_init_periodic_progress_enabled(cfg: SnapSurfConfig) -> bool:
	return str(cfg.map_init.progress_mode) in ("periodic", "both")


def _map_init_log_fringe_debug(
	*,
	state: _MapInitState,
	phase: str,
	block: int,
	iter_idx: int,
	uv_full: torch.Tensor,
	active_quad: torch.Tensor,
	ext_pos: torch.Tensor,
	ext_normals: torch.Tensor,
	ext_valid: torch.Tensor,
	ext_quad_valid: torch.Tensor | None,
	model_xyz: torch.Tensor,
	model_valid: torch.Tensor,
	model_normals: torch.Tensor,
	model_depth: int,
	cfg: SnapSurfConfig,
) -> None:
	if not _map_init_block_progress_enabled(cfg):
		return
	if not bool(active_quad.any().detach().cpu()):
		return
	with torch.no_grad():
		_, terms = _map_init_objective(
			uv_full=uv_full,
			active_quad=active_quad,
			ext_pos=ext_pos,
			ext_normals=ext_normals,
			ext_valid=ext_valid,
			ext_quad_valid=ext_quad_valid,
			ext_coords=state.ext_coords,
			model_xyz=model_xyz,
			model_valid=model_valid,
			model_normals=model_normals,
			model_depth=int(model_depth),
			normal_sign=state.normal_sign,
			orientation_sign=state.orientation_sign,
			cfg=cfg,
			allow_partial_model_samples=_map_init_allow_partial_model_samples(int(state.scale_level)),
		)
	if int(state.fringe_debug_rows) % 20 == 0:
		_map_init_log("map-init block columns")
		print(
			f"{'map':>3s} {'ph':>6s} {'sc':>2s} {'st':>3s} {'res':>7s} {'blk':>3s} {'it':>9s} {'quad':>5s} "
			f"{'smp':>6s} {'succ':>6s} {'sloss':>7s} {'badq':>11s} "
			f"{'loss':>7s} {'dst':>7s} {'vec':>7s} {'nrm':>7s} "
			f"{'smo':>7s} {'bnd':>7s} {'jac':>7s} {'met':>7s} "
			f"{'ar':>7s} {'sr':>7s} {'br':>7s} {'jr':>7s} "
			f"{'jbad':>5s} {'jmin':>6s} {'rmin':>6s}",
			flush=True,
		)
	success = _map_init_term_float(terms, "quad_success_frac")
	bad = (
		f"{_map_init_term_float(terms, 'uv_bad'):.0f}/"
		f"{_map_init_term_float(terms, 'model_bad'):.0f}/"
		f"{_map_init_term_float(terms, 'jac_bad_quad'):.0f}/"
		f"{_map_init_term_float(terms, 'jac_inv_bad_quad'):.0f}/"
		f"{_map_init_term_float(terms, 'step_bad_quad'):.0f}"
	)
	res_label = f"{int(uv_full.shape[0])}x{int(uv_full.shape[1])}"
	print(
		f"{'map':>3s} {str(phase)[:6]:>6s} {int(state.scale_level):2d} {int(state.current_stride()):3d} {res_label:>7s} {int(block):3d} "
		f"{int(iter_idx):4d}/{int(cfg.map_init.iters):<4d} "
		f"{int(active_quad.sum().detach().cpu()):5d} "
		f"{_map_init_term_float(terms, 'sample_valid'):6.0f} "
		f"{_map_init_fmt_val(success):>6s} "
		f"{_map_init_fmt_val(_map_init_term_float(terms, 'sample_loss')):>7s} "
		f"{bad:>11s} "
		f"{_map_init_fmt_val(_map_init_term_float(terms, 'loss')):>7s} "
		f"{_map_init_fmt_val(_map_init_term_float(terms, 'dist')):>7s} "
		f"{_map_init_fmt_val(_map_init_term_float(terms, 'vec')):>7s} "
		f"{_map_init_fmt_val(_map_init_term_float(terms, 'norm')):>7s} "
		f"{_map_init_fmt_val(_map_init_term_float(terms, 'smooth')):>7s} "
		f"{_map_init_fmt_val(_map_init_term_float(terms, 'bend')):>7s} "
		f"{_map_init_fmt_val(_map_init_term_float(terms, 'jac')):>7s} "
		f"{_map_init_fmt_val(_map_init_term_float(terms, 'metric_smooth')):>7s} "
		f"{_map_init_fmt_val(_map_init_term_float(terms, 'area_smooth')):>7s} "
		f"{_map_init_fmt_val(_map_init_term_float(terms, 'smooth_rev')):>7s} "
		f"{_map_init_fmt_val(_map_init_term_float(terms, 'bend_rev')):>7s} "
		f"{_map_init_fmt_val(_map_init_term_float(terms, 'jac_rev')):>7s} "
		f"{_map_init_term_float(terms, 'jac_bad'):5.0f} "
		f"{_map_init_fmt_val(_map_init_term_float(terms, 'jac_min')):>6s} "
		f"{_map_init_fmt_val(_map_init_term_float(terms, 'jac_inv_min')):>6s}",
		flush=True,
	)
	state.fringe_debug_rows += 1


def _map_init_fmt_val(v: float) -> str:
	av = abs(float(v))
	if av != 0.0 and (av >= 1000.0 or av < 1.0e-3):
		return f"{float(v):.1e}"
	if av < 10.0:
		return f"{float(v):.4f}"
	if av < 100.0:
		return f"{float(v):.3f}"
	return f"{float(v):.1f}"


def _map_init_print_progress_legend() -> None:
	items = (
		("mi", "map init"),
		("ph", "add/fringe/grow"),
		("sc", "scale level"),
		("st", "dyadic stride"),
		("res", "scale vertices"),
		("blk", "opt block"),
		("it", "iter/total"),
		("it/s", "opt it/s"),
		("act", "active quads"),
		("blkq", "blocked quads"),
		("spr", "sparse pruned"),
		("smp", "valid samples"),
		("bad", "uv/model/jac/rjac/step"),
		("aloss", "add sample loss"),
		("asuc", "add success frac"),
		("floss", "fringe sample loss"),
		("fsuc", "fringe success frac"),
		("loss", "objective"),
		("dist", "distance"),
		("metr", "model edge scale"),
		("area", "model area scale"),
		("jmin", "min jac"),
		("rmin", "min rev jac"),
	)
	_map_init_log("progress columns")
	key_w = max(len(k) for k, _v in items)
	desc_w = max(len(v) for _k, v in items)
	cell_w = key_w + 3 + desc_w
	header_cell = f"{'col':<{key_w}} : {'meaning':<{desc_w}}"
	header = " | ".join(f"{header_cell:<{cell_w}}" for _ in range(3))
	print(f"  {header}", flush=True)
	for i in range(0, len(items), 3):
		cells = [f"{k:<{key_w}} : {v:<{desc_w}}" for k, v in items[i:i + 3]]
		while len(cells) < 3:
			cells.append(" " * cell_w)
		row = " | ".join(cells)
		print(f"  {row}", flush=True)


def _map_init_log_progress(
	*,
	state: _MapInitState,
	mode: str,
	block: int,
	iter_idx: int,
	iter_total: int,
	active_count: int,
	terms: dict[str, torch.Tensor],
) -> None:
	now = time.monotonic()
	if state.progress_last_time is None:
		it_s = 0.0
	else:
		dt = max(1.0e-9, now - float(state.progress_last_time))
		it_s = float(int(iter_idx) - int(state.progress_last_iter)) / dt
	state.progress_last_time = now
	state.progress_last_iter = int(iter_idx)
	blocked_count = int(state.blocked_quad.sum().detach().cpu()) if state.blocked_quad is not None else 0
	if int(state.progress_rows) % 20 == 0:
		_map_init_print_progress_legend()
		print(
			f"{'mi':>2s} {'ph':>6s} {'sc':>2s} {'st':>3s} {'res':>7s} {'blk':>3s} {'it':>9s} {'it/s':>6s} "
			f"{'act':>5s} {'blkq':>5s} {'spr':>5s} {'smp':>6s} {'bad':>11s} "
			f"{'aloss':>7s} {'asuc':>6s} {'floss':>7s} {'fsuc':>6s} "
			f"{'loss':>7s} {'dist':>7s} {'metr':>7s} {'area':>7s} "
			f"{'jmin':>6s} {'rmin':>6s}",
			flush=True,
		)
	bad = (
		f"{_map_init_term_float(terms, 'uv_bad'):.0f}/"
		f"{_map_init_term_float(terms, 'model_bad'):.0f}/"
		f"{_map_init_term_float(terms, 'jac_bad'):.0f}/"
		f"{_map_init_term_float(terms, 'jac_inv_bad'):.0f}/"
		f"{_map_init_term_float(terms, 'step_bad_quad'):.0f}"
	)
	add_loss, add_success = _map_init_interval_phase_stats(state, "add")
	fringe_loss, fringe_success = _map_init_interval_phase_stats(state, "fringe")
	res_label = "?"
	if state.uv is not None:
		res_label = f"{int(state.uv.shape[0])}x{int(state.uv.shape[1])}"
	print(
		f"{'mi':>2s} {str(mode)[:6]:>6s} {int(state.scale_level):2d} {int(state.current_stride()):3d} {res_label:>7s} {int(block):3d} "
		f"{int(iter_idx):4d}/{int(iter_total):<4d} "
		f"{_map_init_fmt_val(it_s):>6s} "
		f"{int(active_count):5d} "
		f"{blocked_count:5d} "
		f"{int(state.sparse_pruned_total):5d} "
		f"{_map_init_term_float(terms, 'samples'):6.0f} "
		f"{bad:>11s} "
		f"{_map_init_fmt_val(add_loss):>7s} "
		f"{_map_init_fmt_val(add_success):>6s} "
		f"{_map_init_fmt_val(fringe_loss):>7s} "
		f"{_map_init_fmt_val(fringe_success):>6s} "
		f"{_map_init_fmt_val(_map_init_term_float(terms, 'loss')):>7s} "
		f"{_map_init_fmt_val(_map_init_term_float(terms, 'dist')):>7s} "
		f"{_map_init_fmt_val(_map_init_term_float(terms, 'metric_smooth')):>7s} "
		f"{_map_init_fmt_val(_map_init_term_float(terms, 'area_smooth')):>7s} "
		f"{_map_init_fmt_val(_map_init_term_float(terms, 'jac_min')):>6s} "
		f"{_map_init_fmt_val(_map_init_term_float(terms, 'jac_inv_min')):>6s}",
		flush=True,
	)
	state.progress_rows += 1
	_map_init_reset_interval_phase_stats(state)


def _map_init_folded_quad_mask(
	uv: torch.Tensor,
	active_quad: torch.Tensor,
	*,
	orientation_sign: int,
) -> torch.Tensor:
	active = active_quad.bool()
	if active.numel() == 0:
		return active.clone()
	finite_uv = torch.isfinite(uv).all(dim=-1)
	cell = active & _map_init_quad_corner_all(finite_uv)
	bad = active & ~cell
	if not bool(cell.any().detach().cpu()):
		return bad
	p00 = uv[:-1, :-1]
	p10 = uv[1:, :-1]
	p01 = uv[:-1, 1:]
	p11 = uv[1:, 1:]
	dh0 = p10 - p00
	dh1 = p11 - p01
	dw0 = p01 - p00
	dw1 = p11 - p10
	dets = torch.stack([
		dh0[..., 0] * dw0[..., 1] - dh0[..., 1] * dw0[..., 0],
		dh0[..., 0] * dw1[..., 1] - dh0[..., 1] * dw1[..., 0],
		dh1[..., 0] * dw0[..., 1] - dh1[..., 1] * dw0[..., 0],
		dh1[..., 0] * dw1[..., 1] - dh1[..., 1] * dw1[..., 0],
	], dim=-1)
	sign = 1.0 if int(orientation_sign) >= 0 else -1.0
	dets = dets * sign
	dets_finite = torch.isfinite(dets).all(dim=-1)
	det_min = torch.where(dets_finite, dets.min(dim=-1).values, torch.full_like(dets[..., 0], float("-inf")))
	return bad | (active & cell & (det_min <= 0.0))


def _map_init_sparse_quad_mask(
	active_quad: torch.Tensor,
	*,
	min_neighbors: int = 3,
	seed_mask: torch.Tensor | None = None,
) -> torch.Tensor:
	active = active_quad.bool()
	out = torch.zeros_like(active, dtype=torch.bool)
	if active.numel() == 0 or int(min_neighbors) <= 0:
		return out
	keep = active.clone()
	seed = torch.zeros_like(active, dtype=torch.bool)
	if seed_mask is not None and tuple(seed_mask.shape) == tuple(active.shape):
		seed = seed_mask.bool()
	H, W = int(keep.shape[0]), int(keep.shape[1])
	k = torch.ones(1, 1, 3, 3, device=keep.device, dtype=torch.float32)
	k[..., 1, 1] = 0.0
	while bool(keep.any().detach().cpu()):
		count = F.conv2d(keep.to(dtype=torch.float32).reshape(1, 1, H, W), k, padding=1).reshape(H, W)
		remove = keep & (count < float(min_neighbors))
		if seed_mask is not None:
			hole_touch = _neighbor8_mask((seed | out).unsqueeze(0))[0]
			remove = remove & hole_touch
		if not bool(remove.any().detach().cpu()):
			break
		out |= remove
		keep &= ~remove
	return out


def _map_init_apply_sparse_quad_cleanup(
	state: _SurfaceState,
	active_quad: torch.Tensor,
	uv: torch.Tensor,
	blocked_quad: torch.Tensor,
	*,
	cfg: SnapSurfConfig,
	seed_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
	sparse_bad = _map_init_sparse_quad_mask(active_quad, min_neighbors=3, seed_mask=seed_mask)
	sparse_count = int(sparse_bad.sum().detach().cpu())
	if sparse_count <= 0:
		return active_quad, uv, blocked_quad, 0
	active_new = active_quad.bool() & ~sparse_bad
	if not bool(active_new.any().detach().cpu()):
		return active_quad, uv, blocked_quad, 0
	blocked_new = blocked_quad.bool() | sparse_bad
	uv_new = uv
	if not bool(cfg.map_init.dense_opt):
		active_vertices = _map_init_active_vertex_mask(active_new, tuple(int(v) for v in uv.shape[:2]))
		uv_new = torch.where(active_vertices.unsqueeze(-1), uv, torch.full_like(uv, float("nan")))
	state.map_init.sparse_pruned_total += sparse_count
	return active_new, uv_new, blocked_new, sparse_count


def _map_init_prune_bad_active_quads(
	state: _SurfaceState,
	*,
	model_xyz: torch.Tensor,
	model_valid: torch.Tensor,
	model_normals: torch.Tensor,
	cfg: SnapSurfConfig,
) -> tuple[int, int, int]:
	active = state.map_init.active_quad
	uv = state.map_init.uv
	ext_pos = state.map_init.ext_pos
	ext_normals = state.map_init.ext_normals
	ext_valid = state.map_init.ext_valid
	ext_quad_valid = state.map_init.ext_quad_valid
	depth = state.map_init.model_depth
	if active is None or uv is None or ext_pos is None or ext_normals is None or ext_valid is None or depth is None:
		return 0, 0, 0
	if not bool(active.any().detach().cpu()):
		return 0, 0, 0
	sample_bad = torch.zeros_like(active, dtype=torch.bool)
	quad_hw = active.bool().nonzero(as_tuple=False)
	allow_partial_model_samples = _map_init_allow_partial_model_samples(int(state.map_init.scale_level))
	if int(quad_hw.shape[0]) > 0:
		ok = _map_init_candidate_quad_samples_ok(
			uv_full=uv,
			quad_hw=quad_hw,
			ext_pos=ext_pos,
			ext_normals=ext_normals,
			ext_valid=ext_valid,
			ext_quad_valid=ext_quad_valid,
			ext_coords=state.map_init.ext_coords,
			model_xyz=model_xyz,
			model_valid=model_valid,
			model_normals=model_normals,
			model_depth=int(depth),
			normal_sign=state.map_init.normal_sign,
			cfg=cfg,
			allow_partial_model_samples=allow_partial_model_samples,
		)
		sample_bad[quad_hw[:, 0], quad_hw[:, 1]] = ~ok
	step_bad = _map_init_step_neighbor_bad_quad_mask(
		uv,
		active,
		model_xyz=model_xyz,
		model_valid=model_valid,
		model_depth=int(depth),
		max_ratio=float(cfg.map_init.max_step_neighbor_ratio),
	)
	folded_bad = _map_init_folded_quad_mask(
		uv,
		active,
		orientation_sign=state.map_init.orientation_sign,
	)
	bad = active.bool() & (sample_bad | step_bad | folded_bad)
	bad_count = int(bad.sum().detach().cpu())
	if bad_count <= 0:
		return 0, 0, 0
	active_new = active.bool() & ~bad
	blocked = state.map_init.blocked_quad
	if blocked is None or tuple(blocked.shape) != tuple(active.shape):
		blocked = torch.zeros_like(active, dtype=torch.bool)
	blocked = blocked.bool() | bad
	active_new, uv_new, blocked, sparse_count = _map_init_apply_sparse_quad_cleanup(
		state,
		active_new,
		uv,
		blocked,
		cfg=cfg,
		seed_mask=bad,
	)
	state.map_init.blocked_quad = blocked
	state.map_init.active_quad = active_new
	if not bool(cfg.map_init.dense_opt):
		active_vertices = _map_init_active_vertex_mask(active_new, tuple(int(v) for v in uv.shape[:2]))
		state.map_init.uv = torch.where(active_vertices.unsqueeze(-1), uv_new, torch.full_like(uv_new, float("nan")))
	else:
		state.map_init.uv = uv_new
	_map_init_sync_current_uv_to_pyramid(state.map_init)
	_map_init_refresh_current_uv_from_pyramid(state.map_init, cfg)
	_map_init_store_current_scale_masks(state.map_init)
	sample_count = int((bad & (sample_bad | step_bad)).sum().detach().cpu())
	fold_count = int((bad & folded_bad).sum().detach().cpu())
	_map_init_refresh_uv_guess(state, model_valid=model_valid, cfg=cfg)
	return sample_count, fold_count, sparse_count


def _map_init_reject_bad_new_quads(
	state: _SurfaceState,
	*,
	active_before: torch.Tensor,
	active_quad: torch.Tensor,
	uv: torch.Tensor,
	blocked_quad: torch.Tensor,
	model_xyz: torch.Tensor,
	model_valid: torch.Tensor,
	model_normals: torch.Tensor,
	cfg: SnapSurfConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, int]:
	ext_pos = state.map_init.ext_pos
	ext_normals = state.map_init.ext_normals
	ext_valid = state.map_init.ext_valid
	ext_quad_valid = state.map_init.ext_quad_valid
	depth = state.map_init.model_depth
	if ext_pos is None or ext_normals is None or ext_valid is None or depth is None:
		return active_quad, uv, blocked_quad, 0, 0, 0
	new_quad = active_quad.bool() & ~active_before.bool()
	if not bool(new_quad.any().detach().cpu()):
		return active_quad, uv, blocked_quad, 0, 0, 0
	sample_bad = torch.zeros_like(active_quad, dtype=torch.bool)
	quad_hw = new_quad.nonzero(as_tuple=False)
	allow_partial_model_samples = _map_init_allow_partial_model_samples(int(state.map_init.scale_level))
	if int(quad_hw.shape[0]) > 0:
		ok = _map_init_candidate_quad_samples_ok(
			uv_full=uv,
			quad_hw=quad_hw,
			ext_pos=ext_pos,
			ext_normals=ext_normals,
			ext_valid=ext_valid,
			ext_quad_valid=ext_quad_valid,
			ext_coords=state.map_init.ext_coords,
			model_xyz=model_xyz,
			model_valid=model_valid,
			model_normals=model_normals,
			model_depth=int(depth),
			normal_sign=state.map_init.normal_sign,
			cfg=cfg,
			allow_partial_model_samples=allow_partial_model_samples,
		)
		sample_bad[quad_hw[:, 0], quad_hw[:, 1]] = ~ok
	step_bad = _map_init_step_neighbor_bad_quad_mask(
		uv,
		active_quad,
		model_xyz=model_xyz,
		model_valid=model_valid,
		model_depth=int(depth),
		max_ratio=float(cfg.map_init.max_step_neighbor_ratio),
	) & new_quad
	folded_bad = _map_init_folded_quad_mask(
		uv,
		active_quad,
		orientation_sign=state.map_init.orientation_sign,
	) & new_quad
	bad = new_quad & (sample_bad | step_bad | folded_bad)
	if not bool(bad.any().detach().cpu()):
		return active_quad, uv, blocked_quad, 0, 0, 0
	active_new = active_quad.bool() & ~bad
	blocked_new = blocked_quad.bool() | bad
	active_new, uv_new, blocked_new, sparse_count = _map_init_apply_sparse_quad_cleanup(
		state,
		active_new,
		uv,
		blocked_new,
		cfg=cfg,
		seed_mask=bad,
	)
	if not bool(cfg.map_init.dense_opt):
		active_vertices = _map_init_active_vertex_mask(active_new, tuple(int(v) for v in uv.shape[:2]))
		uv_new = torch.where(active_vertices.unsqueeze(-1), uv_new, torch.full_like(uv_new, float("nan")))
	sample_count = int((bad & (sample_bad | step_bad)).sum().detach().cpu())
	fold_count = int((bad & folded_bad).sum().detach().cpu())
	return active_new, uv_new, blocked_new, sample_count, fold_count, sparse_count


def _map_init_optimize_vertex_mask(
	state: _SurfaceState,
	*,
	base_uv: torch.Tensor,
	active_quad: torch.Tensor,
	opt_vertex_mask: torch.Tensor,
	model_xyz: torch.Tensor,
	model_valid: torch.Tensor,
	model_normals: torch.Tensor,
	cfg: SnapSurfConfig,
	steps: int,
	mode: str,
	lr: float,
	w_jac_mult: float = 1.0,
	commit: bool = False,
	refresh_guess: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
	ext_pos = state.map_init.ext_pos
	ext_normals = state.map_init.ext_normals
	ext_valid = state.map_init.ext_valid
	ext_quad_valid = state.map_init.ext_quad_valid
	depth = state.map_init.model_depth
	active_quad = active_quad.bool()
	opt_vertex_mask = opt_vertex_mask.bool() & torch.isfinite(base_uv).all(dim=-1)
	if (
		ext_pos is None or ext_normals is None or ext_valid is None or depth is None or
		not bool(active_quad.any().detach().cpu())
	):
		z = model_xyz.sum() * 0.0
		return base_uv.detach(), {
			"loss": z.detach(), "dist": z.detach(), "vec": z.detach(), "norm": z.detach(),
			"smooth": z.detach(), "bend": z.detach(), "jac": z.detach(),
			"smooth_fwd": z.detach(), "bend_fwd": z.detach(),
			"smooth_uv_fwd": z.detach(), "bend_uv_fwd": z.detach(),
			"smooth_model_fwd": z.detach(), "bend_model_fwd": z.detach(),
			"metric_smooth": z.detach(), "area_smooth": z.detach(),
			"jac_min": z.detach(), "jac_bad": z.detach(), "jac_bad_frac": z.detach(),
			"model_bad": z.detach(),
			"sample_loss": z.detach(), "sample_total": z.detach(), "sample_valid": z.detach(),
			"sample_bad": z.detach(), "sample_bad_frac": z.detach(),
			"quad_total": z.detach(), "quad_success": z.detach(), "quad_success_frac": z.detach(),
			"jac_bad_quad": z.detach(), "jac_inv_bad_quad": z.detach(), "step_bad_quad": z.detach(),
			"completed": z.detach(), "requested": z.detach(),
		}
	remaining = max(0, int(cfg.map_init.iters) - int(state.map_init.total_iters))
	requested_steps = min(max(0, int(steps)), remaining)
	allow_partial_model_samples = _map_init_allow_partial_model_samples(int(state.map_init.scale_level))
	if requested_steps <= 0 or not bool(opt_vertex_mask.any().detach().cpu()):
		_, terms = _map_init_objective(
			uv_full=base_uv.detach(),
			active_quad=active_quad,
			ext_pos=ext_pos,
			ext_normals=ext_normals,
			ext_valid=ext_valid,
			ext_quad_valid=ext_quad_valid,
			ext_coords=state.map_init.ext_coords,
			model_xyz=model_xyz,
			model_valid=model_valid,
			model_normals=model_normals,
			model_depth=int(depth),
			normal_sign=state.map_init.normal_sign,
			orientation_sign=state.map_init.orientation_sign,
			cfg=cfg,
			w_jac_mult=w_jac_mult,
			allow_partial_model_samples=allow_partial_model_samples,
		)
		terms = dict(terms)
		z = model_xyz.sum() * 0.0
		terms["completed"] = z.detach()
		terms["requested"] = torch.tensor(float(requested_steps), device=model_xyz.device, dtype=model_xyz.dtype)
		return base_uv.detach(), terms
	H, W = int(model_valid.shape[1]), int(model_valid.shape[2])
	base = base_uv.detach().clone()
	param = torch.nn.Parameter(base[opt_vertex_mask].detach().clone())
	opt = torch.optim.Adam([param], lr=float(lr))

	def current_uv_full() -> torch.Tensor:
		out = base.clone()
		out[opt_vertex_mask] = param
		return out

	periodic_progress = _map_init_periodic_progress_enabled(cfg)
	progress_interval = max(100, int(cfg.map_init.progress_interval))
	if periodic_progress and state.map_init.progress_last_time is None:
		state.map_init.progress_last_time = time.monotonic()
		state.map_init.progress_last_iter = int(state.map_init.total_iters)
	active_count = int(active_quad.sum().detach().cpu())
	last_terms: dict[str, torch.Tensor] | None = None
	completed = 0
	for local_iter in range(1, requested_steps + 1):
		opt.zero_grad(set_to_none=True)
		uv_full = current_uv_full()
		loss, terms = _map_init_objective(
			uv_full=uv_full,
			active_quad=active_quad,
			ext_pos=ext_pos,
			ext_normals=ext_normals,
			ext_valid=ext_valid,
			ext_quad_valid=ext_quad_valid,
			ext_coords=state.map_init.ext_coords,
			model_xyz=model_xyz,
			model_valid=model_valid,
			model_normals=model_normals,
			model_depth=int(depth),
			normal_sign=state.map_init.normal_sign,
			orientation_sign=state.map_init.orientation_sign,
			cfg=cfg,
			w_jac_mult=w_jac_mult,
			allow_partial_model_samples=allow_partial_model_samples,
		)
		if not bool(torch.isfinite(loss).detach().cpu()):
			_map_init_log(
				"opt nonfinite "
				f"mode={mode} "
				f"block={state.map_init.opt_blocks + 1} "
				f"iter={state.map_init.total_iters + local_iter}/{cfg.map_init.iters} "
				f"active={active_count} "
				f"uv_bad={float(terms.get('uv_bad', torch.zeros(())).detach().cpu()):.0f} "
				f"samples={float(terms.get('samples', torch.zeros(())).detach().cpu()):.0f} "
				f"loss={float(loss.detach().cpu())}"
			)
			break
		loss.backward()
		if param.grad is not None and not bool(torch.isfinite(param.grad).all().detach().cpu()):
			_map_init_log(
				"opt nonfinite_grad "
				f"mode={mode} "
				f"block={state.map_init.opt_blocks + 1} "
				f"iter={state.map_init.total_iters + local_iter}/{cfg.map_init.iters} "
				f"active={active_count}"
			)
			break
		prev_param = param.detach().clone()
		opt.step()
		with torch.no_grad():
			param[:, 0].clamp_(0.0, float(max(0, H - 1)))
			param[:, 1].clamp_(0.0, float(max(0, W - 1)))
			if not bool(torch.isfinite(param).all().detach().cpu()):
				param.copy_(prev_param)
				_map_init_log(
					"opt nonfinite_param "
					f"mode={mode} "
					f"block={state.map_init.opt_blocks + 1} "
					f"iter={state.map_init.total_iters + local_iter}/{cfg.map_init.iters} "
					f"active={active_count}"
				)
				break
		last_terms = terms
		completed += 1
		if periodic_progress:
			global_iter = state.map_init.total_iters + local_iter
			if global_iter % progress_interval == 0:
				_map_init_log_progress(
				state=state.map_init,
					mode=mode,
					block=state.map_init.opt_blocks + 1,
					iter_idx=global_iter,
					iter_total=int(cfg.map_init.iters),
					active_count=active_count,
					terms=terms,
				)
	with torch.no_grad():
		uv_full = current_uv_full().detach()
	state.map_init.total_iters += int(completed)
	state.map_init.opt_blocks += 1
	_, last_terms_eval = _map_init_objective(
		uv_full=uv_full,
		active_quad=active_quad,
		ext_pos=ext_pos,
		ext_normals=ext_normals,
		ext_valid=ext_valid,
		ext_quad_valid=ext_quad_valid,
		ext_coords=state.map_init.ext_coords,
		model_xyz=model_xyz,
		model_valid=model_valid,
		model_normals=model_normals,
		model_depth=int(depth),
		normal_sign=state.map_init.normal_sign,
		orientation_sign=state.map_init.orientation_sign,
		cfg=cfg,
		w_jac_mult=w_jac_mult,
		allow_partial_model_samples=allow_partial_model_samples,
	)
	last_terms = dict(last_terms_eval if last_terms is None else last_terms_eval)
	last_terms["completed"] = torch.tensor(float(completed), device=model_xyz.device, dtype=model_xyz.dtype)
	last_terms["requested"] = torch.tensor(float(requested_steps), device=model_xyz.device, dtype=model_xyz.dtype)
	if mode in ("add", "fringe"):
		_map_init_accumulate_phase_stats(state.map_init, mode, last_terms)
	if commit:
		state.map_init.uv = uv_full.detach()
		_map_init_sync_current_uv_to_pyramid(state.map_init)
		_map_init_refresh_current_uv_from_pyramid(state.map_init, cfg)
		_map_init_store_current_scale_masks(state.map_init)
		if refresh_guess:
			_map_init_refresh_uv_guess(state, model_valid=model_valid, cfg=cfg)
	return uv_full.detach(), last_terms


def _map_init_optimize_block(
	state: _SurfaceState,
	*,
	model_xyz: torch.Tensor,
	model_valid: torch.Tensor,
	model_normals: torch.Tensor,
	cfg: SnapSurfConfig,
	steps: int,
	mode: str = "grow",
	lr_mult: float = 1.0,
	w_jac_mult: float = 1.0,
) -> dict[str, torch.Tensor]:
	active = state.map_init.active_quad
	uv = state.map_init.uv
	ext_pos = state.map_init.ext_pos
	ext_normals = state.map_init.ext_normals
	ext_valid = state.map_init.ext_valid
	ext_quad_valid = state.map_init.ext_quad_valid
	depth = state.map_init.model_depth
	if (
		active is None or uv is None or ext_pos is None or ext_normals is None or
		ext_valid is None or depth is None or int(steps) <= 0 or
		not bool(active.any().detach().cpu())
	):
		z = model_xyz.sum() * 0.0
		return {
			"loss": z.detach(), "dist": z.detach(), "vec": z.detach(), "norm": z.detach(),
			"smooth": z.detach(), "bend": z.detach(), "jac": z.detach(),
			"smooth_fwd": z.detach(), "bend_fwd": z.detach(),
			"smooth_uv_fwd": z.detach(), "bend_uv_fwd": z.detach(),
			"smooth_model_fwd": z.detach(), "bend_model_fwd": z.detach(),
			"metric_smooth": z.detach(), "area_smooth": z.detach(),
			"jac_min": z.detach(), "jac_bad": z.detach(), "jac_bad_frac": z.detach(),
			"model_bad": z.detach(),
			"sample_loss": z.detach(), "sample_total": z.detach(), "sample_valid": z.detach(),
			"sample_bad": z.detach(), "sample_bad_frac": z.detach(),
			"quad_total": z.detach(), "quad_success": z.detach(), "quad_success_frac": z.detach(),
			"jac_bad_quad": z.detach(), "jac_inv_bad_quad": z.detach(), "step_bad_quad": z.detach(),
		}
	H, W = int(model_valid.shape[1]), int(model_valid.shape[2])
	dense_mode = bool(cfg.map_init.dense_opt)
	allow_partial_model_samples = _map_init_allow_partial_model_samples(int(state.map_init.scale_level))
	lr = float(cfg.map_init.lr) * float(lr_mult)
	base_uv = uv.detach().clone()
	active_vertices = _map_init_active_vertex_mask(active, tuple(int(v) for v in uv.shape[:2])) & torch.isfinite(uv).all(dim=-1)
	uv_prior: torch.Tensor | None = None
	if dense_mode:
		dense_seed = _map_init_dense_seed_uv(state, model_valid=model_valid, cfg=cfg)
		uv_prior = dense_seed.detach().clone()
		param = torch.nn.Parameter(dense_seed.detach().clone())
		opt_params = [param]
	else:
		param = torch.nn.Parameter(uv[active_vertices].detach().clone())
		opt_params = [param]
	opt = torch.optim.Adam(opt_params, lr=lr)

	def current_uv_full() -> torch.Tensor:
		if dense_mode:
			return _map_init_clamp_uv(param, model_h=H, model_w=W)
		if param is None:
			raise RuntimeError("map-init active optimizer parameter missing")
		out = base_uv.clone()
		out[active_vertices] = param
		return out

	last_terms: dict[str, torch.Tensor] | None = None
	requested_steps = int(steps)
	periodic_progress = _map_init_periodic_progress_enabled(cfg)
	progress_interval = max(100, int(cfg.map_init.progress_interval))
	if periodic_progress and state.map_init.progress_last_time is None:
		state.map_init.progress_last_time = time.monotonic()
		state.map_init.progress_last_iter = int(state.map_init.total_iters)
	completed = 0
	for local_iter in range(1, requested_steps + 1):
		opt.zero_grad(set_to_none=True)
		uv_full = current_uv_full()
		loss, terms = _map_init_objective(
			uv_full=uv_full,
			active_quad=active,
			ext_pos=ext_pos,
			ext_normals=ext_normals,
			ext_valid=ext_valid,
			ext_quad_valid=ext_quad_valid,
			ext_coords=state.map_init.ext_coords,
			model_xyz=model_xyz,
			model_valid=model_valid,
			model_normals=model_normals,
			model_depth=int(depth),
			normal_sign=state.map_init.normal_sign,
			orientation_sign=state.map_init.orientation_sign,
			cfg=cfg,
			w_jac_mult=w_jac_mult,
			uv_prior=uv_prior,
			allow_partial_model_samples=allow_partial_model_samples,
		)
		if not bool(torch.isfinite(loss).detach().cpu()):
			_map_init_log(
				"opt nonfinite "
				f"mode={mode} "
				f"block={state.map_init.opt_blocks + 1} "
				f"iter={state.map_init.total_iters + local_iter}/{cfg.map_init.iters} "
				f"active={state.map_init.active_count()} "
				f"uv_bad={float(terms.get('uv_bad', torch.zeros(())).detach().cpu()):.0f} "
				f"samples={float(terms.get('samples', torch.zeros(())).detach().cpu()):.0f} "
				f"loss={float(loss.detach().cpu())}"
			)
			break
		loss.backward()
		grad_finite = True
		for p in opt_params:
			if p.grad is not None and not bool(torch.isfinite(p.grad).all().detach().cpu()):
				grad_finite = False
				break
		if not grad_finite:
			_map_init_log(
				"opt nonfinite_grad "
				f"mode={mode} "
				f"block={state.map_init.opt_blocks + 1} "
				f"iter={state.map_init.total_iters + local_iter}/{cfg.map_init.iters} "
				f"active={state.map_init.active_count()}"
			)
			break
		prev_params = [p.detach().clone() for p in opt_params]
		opt.step()
		with torch.no_grad():
			if not dense_mode:
				if param is None:
					raise RuntimeError("map-init active optimizer parameter missing")
				param[:, 0].clamp_(0.0, float(max(0, H - 1)))
				param[:, 1].clamp_(0.0, float(max(0, W - 1)))
			param_finite = all(bool(torch.isfinite(p).all().detach().cpu()) for p in opt_params)
			if not param_finite:
				for p, prev in zip(opt_params, prev_params, strict=False):
					p.copy_(prev)
				_map_init_log(
					"opt nonfinite_param "
					f"mode={mode} "
					f"block={state.map_init.opt_blocks + 1} "
					f"iter={state.map_init.total_iters + local_iter}/{cfg.map_init.iters} "
					f"active={state.map_init.active_count()}"
				)
				break
		last_terms = terms
		completed += 1
		if periodic_progress:
			global_iter = state.map_init.total_iters + local_iter
			if global_iter % progress_interval == 0:
				_map_init_log_progress(
				state=state.map_init,
					mode=mode,
					block=state.map_init.opt_blocks + 1,
					iter_idx=global_iter,
					iter_total=int(cfg.map_init.iters),
					active_count=state.map_init.active_count(),
					terms=terms,
				)
	with torch.no_grad():
		state.map_init.uv = current_uv_full().detach()
	_map_init_sync_current_uv_to_pyramid(state.map_init)
	_map_init_refresh_current_uv_from_pyramid(state.map_init, cfg)
	_map_init_store_current_scale_masks(state.map_init)
	state.map_init.total_iters += int(completed)
	state.map_init.opt_blocks += 1
	_map_init_refresh_uv_guess(state, model_valid=model_valid, cfg=cfg)
	if state.map_init.uv is not None:
		uv_full = state.map_init.uv
	else:
		uv_full = current_uv_full().detach()
	_, last_terms = _map_init_objective(
		uv_full=uv_full,
		active_quad=active,
		ext_pos=ext_pos,
		ext_normals=ext_normals,
		ext_valid=ext_valid,
		ext_quad_valid=ext_quad_valid,
		ext_coords=state.map_init.ext_coords,
		model_xyz=model_xyz,
		model_valid=model_valid,
		model_normals=model_normals,
		model_depth=int(depth),
		normal_sign=state.map_init.normal_sign,
		orientation_sign=state.map_init.orientation_sign,
		cfg=cfg,
		w_jac_mult=w_jac_mult,
		uv_prior=uv_prior,
		allow_partial_model_samples=allow_partial_model_samples,
	)
	last_terms = dict(last_terms)
	last_terms["completed"] = torch.tensor(float(completed), device=model_xyz.device, dtype=model_xyz.dtype)
	last_terms["requested"] = torch.tensor(float(requested_steps), device=model_xyz.device, dtype=model_xyz.dtype)
	return last_terms


def _map_init_eval_terms_for_state(
	state: _SurfaceState,
	*,
	model_xyz: torch.Tensor,
	model_valid: torch.Tensor,
	model_normals: torch.Tensor,
	cfg: SnapSurfConfig,
) -> dict[str, torch.Tensor]:
	if (
		state.map_init.active_quad is None or state.map_init.uv is None or
		state.map_init.ext_pos is None or state.map_init.ext_normals is None or
		state.map_init.ext_valid is None or state.map_init.model_depth is None
	):
		return {}
	_, terms = _map_init_objective(
		uv_full=state.map_init.uv,
		active_quad=state.map_init.active_quad,
		ext_pos=state.map_init.ext_pos,
		ext_normals=state.map_init.ext_normals,
		ext_valid=state.map_init.ext_valid,
		ext_quad_valid=state.map_init.ext_quad_valid,
		ext_coords=state.map_init.ext_coords,
		model_xyz=model_xyz,
		model_valid=model_valid,
		model_normals=model_normals,
		model_depth=int(state.map_init.model_depth),
		normal_sign=state.map_init.normal_sign,
		orientation_sign=state.map_init.orientation_sign,
		cfg=cfg,
		allow_partial_model_samples=_map_init_allow_partial_model_samples(int(state.map_init.scale_level)),
	)
	return dict(terms)


def _map_init_needs_repair(terms: dict[str, torch.Tensor]) -> bool:
	jac_bad = _map_init_term_float(terms, "jac_bad") > 0.0
	jac_flipped = _map_init_term_float(terms, "jac_min") <= 0.0
	return jac_bad and jac_flipped


def _map_init_terms_need_global_opt(terms: dict[str, torch.Tensor]) -> bool:
	if not terms:
		return True
	loss = terms.get("loss")
	if loss is None or not bool(torch.isfinite(loss).all().detach().cpu()):
		return True
	if _map_init_term_float(terms, "uv_bad") > 0.0:
		return True
	if _map_init_term_float(terms, "model_bad") > 0.0:
		return True
	if _map_init_term_float(terms, "step_bad_quad") > 0.0:
		return True
	if _map_init_term_float(terms, "jac_bad") > 0.0 and _map_init_term_float(terms, "jac_min") <= 0.0:
		return True
	if _map_init_term_float(terms, "jac_inv_bad_quad") > 0.0:
		return True
	return False


def _map_init_should_run_global_opt(
	state: _SurfaceState,
	cfg: SnapSurfConfig,
	*,
	added: int,
	pruned_sample: int,
	pruned_fold: int,
	pruned_sparse: int,
	terms: dict[str, torch.Tensor],
) -> tuple[bool, str]:
	interval = max(1, int(cfg.map_init.global_opt_interval))
	if interval <= 1:
		return True, "interval"
	if int(added) <= 0:
		return True, "stall"
	if int(pruned_sample) > 0 or int(pruned_fold) > 0 or int(pruned_sparse) > 0:
		return True, "rim_prune"
	if _map_init_terms_need_global_opt(terms):
		return True, "rim_problem"
	if int(state.map_init.rim_blocks_since_global_opt) + 1 >= interval:
		return True, "interval"
	return False, "rim_ok"


def _map_init_try_coarser_revisit(
	state: _SurfaceState,
	*,
	model_xyz: torch.Tensor,
	model_valid: torch.Tensor,
	model_normals: torch.Tensor,
	cfg: SnapSurfConfig,
) -> int:
	mi = state.map_init
	interval = int(cfg.map_init.coarse_revisit_blocks)
	if interval <= 0 or mi.uv_pyramid is None or mi.active_quad is None:
		return 0
	current_level = int(mi.scale_level)
	max_level = len(mi.uv_pyramid) - 1
	if current_level >= max_level:
		return 0
	if int(mi.opt_blocks) - int(mi.coarse_revisit_last_block) < interval:
		return 0
	mi.coarse_revisit_last_block = int(mi.opt_blocks)
	original_level = current_level
	level = original_level + 1
	_map_init_promote_full_active_to_coarser(mi, from_level=original_level, to_level=level)
	active = mi.scale_active_quads[level] if level < len(mi.scale_active_quads) else None
	if active is None or not bool(active.any().detach().cpu()):
		return 0
	if not _map_init_switch_to_scale(state, cfg, level, reset_blocked=True):
		_map_init_switch_to_scale(state, cfg, original_level, reset_blocked=False)
		return 0
	active = mi.active_quad
	before = int(active.sum().detach().cpu()) if active is not None else 0
	added = _map_init_grow_once(
		state,
		model_xyz=model_xyz,
		model_valid=model_valid,
		model_normals=model_normals,
		cfg=cfg,
	)
	if added > 0:
		_map_init_log(
			"coarse revisit "
			f"level={level} "
			f"from={original_level} "
			f"active={before}->{state.map_init.active_count()} "
			f"added={added} "
			f"block={mi.opt_blocks}"
		)
		return int(added)
	_map_init_store_current_scale_masks(mi)
	_map_init_switch_to_scale(state, cfg, original_level, reset_blocked=False)
	return 0


def _map_init_repair_block_steps(cfg: SnapSurfConfig) -> int:
	steps = int(cfg.map_init.repair_opt_iters)
	if steps <= 0:
		steps = int(cfg.map_init.grow_opt_iters)
	return max(0, steps)


def _map_init_repair_block_allowed(cfg: SnapSurfConfig, completed_repair_blocks: int) -> bool:
	cap = int(cfg.map_init.repair_max_blocks)
	return cap <= 0 or int(completed_repair_blocks) < cap


def _debug_write_map_init_scale_objs(
	*,
	cfg: SnapSurfConfig,
	surface_index: int,
	surface_count: int,
	model_xyz: torch.Tensor,
	model_valid: torch.Tensor,
	ext_xyz: torch.Tensor,
	ext_valid: torch.Tensor,
	state: _SurfaceState,
) -> None:
	if _debug_obj_iter_dir(cfg) is None:
		return
	mi = state.map_init
	if mi.uv is None:
		return
	level = int(mi.scale_level)
	snapshot = f"scale_l{level:02d}"
	_debug_write_map_init_objs(
		cfg=cfg,
		surface_index=surface_index,
		surface_count=surface_count,
		model_xyz=model_xyz,
		model_valid=model_valid,
		ext_xyz=ext_xyz,
		ext_valid=ext_valid,
		state=state,
		snapshot_name=snapshot,
	)


def _run_map_init_for_surface(
	state: _SurfaceState,
	*,
	model_xyz: torch.Tensor,
	model_valid: torch.Tensor,
	model_normals: torch.Tensor,
	ext_xyz: torch.Tensor,
	ext_valid: torch.Tensor,
	ext_normals: torch.Tensor,
	ext_quad_valid: torch.Tensor,
	cfg: SnapSurfConfig,
	seed_xyz: tuple[float, float, float],
	surface_index: int = 0,
	surface_count: int = 1,
) -> dict[str, float]:
	if state.map_init.done:
		_map_init_log(
			"reuse "
			f"surface_active={state.map_init.active_count()} "
			f"iters={state.map_init.total_iters} "
			f"normal_sign={state.map_init.normal_sign}"
		)
		return dict(state.map_init.stats)
	stats = _map_init_empty_stats()
	_map_init_log(
		"start "
		f"model_shape={tuple(int(v) for v in model_xyz.shape[:3])} "
		f"ext_shape={tuple(int(v) for v in ext_xyz.shape[:2])} "
		f"valid_ext={int(ext_valid.sum().detach().cpu())} "
		f"valid_ext_quads={int(ext_quad_valid.sum().detach().cpu()) if ext_quad_valid.numel() else 0}"
	)
	with torch.enable_grad():
		with torch.no_grad():
			ok, seed_model_dist, seed_ext_dist, init_count = _map_init_seed_state(
				state,
				model_xyz=model_xyz,
				model_valid=model_valid,
				model_normals=model_normals,
				ext_xyz=ext_xyz,
				ext_valid=ext_valid,
				ext_normals=ext_normals,
				ext_quad_valid=ext_quad_valid,
				cfg=cfg,
				seed_xyz=seed_xyz,
			)
		stats["snaps_sdist"] = float(seed_model_dist)
		stats["snaps_sext"] = float(seed_ext_dist)
		stats["snaps_seed"] = 1.0 if ok else 0.0
		stats["snaps_map_init"] = float(init_count)
		_map_init_log(
			"seed "
			f"ok={int(ok)} "
			f"seed_model_dist={seed_model_dist:.6g} "
			f"seed_ext_dist={seed_ext_dist:.6g} "
			f"init_active={init_count} "
			f"seed_ext_sample={state.map_init.seed_ext_sample_hw} "
			f"seed_model_quad={state.map_init.seed_model_quad} "
			f"model_depth={state.map_init.model_depth} "
			f"orientation_sign={state.map_init.orientation_sign} "
			f"normal_sign={state.map_init.normal_sign}"
		)
		if ok:
			last_terms: dict[str, torch.Tensor] = {}
			seed_block = min(
				int(cfg.map_init.seed_opt_iters),
				int(cfg.map_init.iters) - int(state.map_init.total_iters),
			)
			seed_opt_complete = True
			if seed_block > 0:
				if (
					state.map_init.active_quad is not None and state.map_init.uv is not None and
					state.map_init.ext_pos is not None and state.map_init.ext_normals is not None and
					state.map_init.ext_valid is not None and state.map_init.model_depth is not None
				):
					_map_init_log_fringe_debug(
						state=state.map_init,
						phase="seed0",
						block=state.map_init.opt_blocks + 1,
						iter_idx=state.map_init.total_iters,
						uv_full=state.map_init.uv,
						active_quad=state.map_init.active_quad,
						ext_pos=state.map_init.ext_pos,
						ext_normals=state.map_init.ext_normals,
						ext_valid=state.map_init.ext_valid,
						ext_quad_valid=state.map_init.ext_quad_valid,
						model_xyz=model_xyz,
						model_valid=model_valid,
						model_normals=model_normals,
						model_depth=int(state.map_init.model_depth),
						cfg=cfg,
					)
				last_terms = _map_init_optimize_block(
					state,
					model_xyz=model_xyz,
					model_valid=model_valid,
					model_normals=model_normals,
					cfg=cfg,
					steps=seed_block,
					mode="seed",
				)
				if (
					state.map_init.active_quad is not None and state.map_init.uv is not None and
					state.map_init.ext_pos is not None and state.map_init.ext_normals is not None and
					state.map_init.ext_valid is not None and state.map_init.model_depth is not None
				):
					_map_init_log_fringe_debug(
						state=state.map_init,
						phase="seed",
						block=state.map_init.opt_blocks,
						iter_idx=state.map_init.total_iters,
						uv_full=state.map_init.uv,
						active_quad=state.map_init.active_quad,
						ext_pos=state.map_init.ext_pos,
						ext_normals=state.map_init.ext_normals,
						ext_valid=state.map_init.ext_valid,
						ext_quad_valid=state.map_init.ext_quad_valid,
						model_xyz=model_xyz,
						model_valid=model_valid,
						model_normals=model_normals,
						model_depth=int(state.map_init.model_depth),
						cfg=cfg,
					)
				completed_seed = int(float(last_terms.get("completed", torch.zeros(())).detach().cpu()))
				if completed_seed < seed_block:
					seed_opt_complete = False
				if seed_opt_complete:
					pruned_sample, pruned_fold, pruned_sparse = _map_init_prune_bad_active_quads(
						state,
						model_xyz=model_xyz,
						model_valid=model_valid,
						model_normals=model_normals,
						cfg=cfg,
					)
					if pruned_sample > 0 or pruned_fold > 0 or pruned_sparse > 0:
						last_terms = _map_init_eval_terms_for_state(
							state,
							model_xyz=model_xyz,
							model_valid=model_valid,
							model_normals=model_normals,
							cfg=cfg,
						)
						_map_init_log(
							"seed prune "
							f"level={state.map_init.scale_level} "
							f"sample={pruned_sample} "
							f"fold={pruned_fold} "
							f"sparse={pruned_sparse} "
							f"active={state.map_init.active_count()}"
						)
					if state.map_init.active_count() <= 0:
						seed_opt_complete = False
			while seed_opt_complete and state.map_init.total_iters < int(cfg.map_init.iters):
				added = _map_init_grow_once(
					state,
					model_xyz=model_xyz,
					model_valid=model_valid,
					model_normals=model_normals,
					cfg=cfg,
				)
				block = min(
					int(cfg.map_init.grow_opt_iters),
					int(cfg.map_init.iters) - int(state.map_init.total_iters),
				)
				pruned_sample = 0
				pruned_fold = 0
				pruned_sparse = 0
				run_global = False
				global_reason = "none"
				if block > 0:
					if int(cfg.map_init.global_opt_interval) <= 1:
						run_global = True
						global_reason = "interval"
					else:
						if state.map_init.active_count() > 0:
							last_terms = _map_init_eval_terms_for_state(
								state,
								model_xyz=model_xyz,
								model_valid=model_valid,
								model_normals=model_normals,
								cfg=cfg,
							)
						run_global, global_reason = _map_init_should_run_global_opt(
							state,
							cfg,
							added=added,
							pruned_sample=pruned_sample,
							pruned_fold=pruned_fold,
							pruned_sparse=pruned_sparse,
							terms=last_terms,
						)
				if run_global:
					if global_reason in ("rim_prune", "rim_problem"):
						state.map_init.rim_problem_blocks += 1
					if (
						state.map_init.active_quad is not None and state.map_init.uv is not None and
						state.map_init.ext_pos is not None and state.map_init.ext_normals is not None and
						state.map_init.ext_valid is not None and state.map_init.model_depth is not None
					):
						_map_init_log_fringe_debug(
							state=state.map_init,
							phase="grow0",
							block=state.map_init.opt_blocks + 1,
							iter_idx=state.map_init.total_iters,
							uv_full=state.map_init.uv,
							active_quad=state.map_init.active_quad,
							ext_pos=state.map_init.ext_pos,
							ext_normals=state.map_init.ext_normals,
							ext_valid=state.map_init.ext_valid,
							ext_quad_valid=state.map_init.ext_quad_valid,
							model_xyz=model_xyz,
							model_valid=model_valid,
							model_normals=model_normals,
							model_depth=int(state.map_init.model_depth),
							cfg=cfg,
						)
					last_terms = _map_init_optimize_block(
						state,
						model_xyz=model_xyz,
						model_valid=model_valid,
						model_normals=model_normals,
						cfg=cfg,
						steps=block,
					)
					state.map_init.global_opt_blocks += 1
					state.map_init.rim_blocks_since_global_opt = 0
					if (
						state.map_init.active_quad is not None and state.map_init.uv is not None and
						state.map_init.ext_pos is not None and state.map_init.ext_normals is not None and
						state.map_init.ext_valid is not None and state.map_init.model_depth is not None
					):
						_map_init_log_fringe_debug(
							state=state.map_init,
							phase="grow",
							block=state.map_init.opt_blocks,
							iter_idx=state.map_init.total_iters,
							uv_full=state.map_init.uv,
							active_quad=state.map_init.active_quad,
							ext_pos=state.map_init.ext_pos,
							ext_normals=state.map_init.ext_normals,
							ext_valid=state.map_init.ext_valid,
							ext_quad_valid=state.map_init.ext_quad_valid,
							model_xyz=model_xyz,
							model_valid=model_valid,
							model_normals=model_normals,
							model_depth=int(state.map_init.model_depth),
							cfg=cfg,
						)
					completed = int(float(last_terms.get("completed", torch.zeros(())).detach().cpu()))
					pruned_sample_after, pruned_fold_after, pruned_sparse_after = _map_init_prune_bad_active_quads(
						state,
						model_xyz=model_xyz,
						model_valid=model_valid,
						model_normals=model_normals,
						cfg=cfg,
					)
					if pruned_sample_after > 0 or pruned_fold_after > 0 or pruned_sparse_after > 0:
						last_terms = _map_init_eval_terms_for_state(
							state,
							model_xyz=model_xyz,
							model_valid=model_valid,
							model_normals=model_normals,
							cfg=cfg,
						)
					if completed < block:
						break
					repair_local_blocks = 0
					repair_block_cap = int(cfg.map_init.repair_max_blocks)
					repair_block_cap_label = "unlimited" if repair_block_cap <= 0 else str(repair_block_cap)
					while (
						_map_init_needs_repair(last_terms) and
						_map_init_repair_block_allowed(cfg, repair_local_blocks) and
						state.map_init.total_iters < int(cfg.map_init.iters)
					):
						repair_block = min(
							_map_init_repair_block_steps(cfg),
							int(cfg.map_init.iters) - int(state.map_init.total_iters),
						)
						if repair_block <= 0:
							break
						repair_local_blocks += 1
						last_terms = _map_init_optimize_block(
							state,
							model_xyz=model_xyz,
							model_valid=model_valid,
							model_normals=model_normals,
							cfg=cfg,
							steps=repair_block,
							mode="repair",
							lr_mult=float(cfg.map_init.repair_lr_mult),
							w_jac_mult=float(cfg.map_init.repair_w_jac_mult),
						)
						state.map_init.repair_blocks += 1
						completed_repair = int(float(last_terms.get("completed", torch.zeros(())).detach().cpu()))
						pruned_sample, pruned_fold, pruned_sparse = _map_init_prune_bad_active_quads(
							state,
							model_xyz=model_xyz,
							model_valid=model_valid,
							model_normals=model_normals,
							cfg=cfg,
						)
						if pruned_sample > 0 or pruned_fold > 0 or pruned_sparse > 0:
							last_terms = _map_init_eval_terms_for_state(
								state,
								model_xyz=model_xyz,
								model_valid=model_valid,
								model_normals=model_normals,
								cfg=cfg,
							)
						if completed_repair < repair_block:
							break
					if _map_init_needs_repair(last_terms):
						_map_init_log(
							"repair_unresolved "
							f"iters={state.map_init.total_iters}/{cfg.map_init.iters} "
							f"active={state.map_init.active_count()} "
							f"uv_bad={_map_init_term_float(last_terms, 'uv_bad'):.0f} "
							f"model_bad={_map_init_term_float(last_terms, 'model_bad'):.0f} "
							f"jac_bad={_map_init_term_float(last_terms, 'jac_bad'):.0f} "
							f"jac_min={_map_init_term_float(last_terms, 'jac_min'):.6g} "
							f"repair_block_cap={repair_block_cap_label} "
							"continue_growth=1"
						)
				elif added > 0:
					state.map_init.rim_only_blocks += 1
					state.map_init.rim_blocks_since_global_opt += 1
				if added <= 0 or block <= 0:
					if int(state.map_init.scale_level) > int(state.map_init.target_scale_level):
						_debug_write_map_init_scale_objs(
							cfg=cfg,
							surface_index=surface_index,
							surface_count=surface_count,
							model_xyz=model_xyz,
							model_valid=model_valid,
							ext_xyz=ext_xyz,
							ext_valid=ext_valid,
							state=state,
						)
					if (
						int(state.map_init.scale_level) > int(state.map_init.target_scale_level) and
						_map_init_transition_to_finer(state, cfg)
					):
						pruned_sample, pruned_fold, pruned_sparse = _map_init_prune_bad_active_quads(
							state,
							model_xyz=model_xyz,
							model_valid=model_valid,
							model_normals=model_normals,
							cfg=cfg,
						)
						last_terms = _map_init_eval_terms_for_state(
							state,
							model_xyz=model_xyz,
							model_valid=model_valid,
							model_normals=model_normals,
							cfg=cfg,
						)
						_map_init_refresh_uv_guess(state, model_valid=model_valid, cfg=cfg)
						_map_init_log(
							"scale prune "
							f"level={state.map_init.scale_level} "
							f"sample={pruned_sample} "
							f"fold={pruned_fold} "
							f"sparse={pruned_sparse} "
							f"active={state.map_init.active_count()}"
						)
						continue
					break
			while int(state.map_init.scale_level) > int(state.map_init.target_scale_level):
				_debug_write_map_init_scale_objs(
					cfg=cfg,
					surface_index=surface_index,
					surface_count=surface_count,
					model_xyz=model_xyz,
					model_valid=model_valid,
					ext_xyz=ext_xyz,
					ext_valid=ext_valid,
					state=state,
				)
				if not _map_init_transition_to_finer(state, cfg):
					break
				_map_init_sync_current_uv_to_pyramid(state.map_init)
			_map_init_finalize_dyadic_state(state, cfg)
			_debug_write_map_init_scale_objs(
				cfg=cfg,
				surface_index=surface_index,
				surface_count=surface_count,
				model_xyz=model_xyz,
				model_valid=model_valid,
				ext_xyz=ext_xyz,
				ext_valid=ext_valid,
				state=state,
			)
			if not last_terms and state.map_init.active_quad is not None and state.map_init.uv is not None:
				_, last_terms = _map_init_objective(
					uv_full=state.map_init.uv,
					active_quad=state.map_init.active_quad,
					ext_pos=state.map_init.ext_pos,
					ext_normals=state.map_init.ext_normals,
					ext_valid=state.map_init.ext_valid,
					ext_quad_valid=state.map_init.ext_quad_valid,
					ext_coords=state.map_init.ext_coords,
					model_xyz=model_xyz,
					model_valid=model_valid,
					model_normals=model_normals,
					model_depth=int(state.map_init.model_depth),
					normal_sign=state.map_init.normal_sign,
					orientation_sign=state.map_init.orientation_sign,
					cfg=cfg,
					allow_partial_model_samples=_map_init_allow_partial_model_samples(int(state.map_init.scale_level)),
				)
			elif state.map_init.active_quad is not None and state.map_init.uv is not None:
				_, last_terms = _map_init_objective(
					uv_full=state.map_init.uv,
					active_quad=state.map_init.active_quad,
					ext_pos=state.map_init.ext_pos,
					ext_normals=state.map_init.ext_normals,
					ext_valid=state.map_init.ext_valid,
					ext_quad_valid=state.map_init.ext_quad_valid,
					ext_coords=state.map_init.ext_coords,
					model_xyz=model_xyz,
					model_valid=model_valid,
					model_normals=model_normals,
					model_depth=int(state.map_init.model_depth),
					normal_sign=state.map_init.normal_sign,
					orientation_sign=state.map_init.orientation_sign,
					cfg=cfg,
					allow_partial_model_samples=_map_init_allow_partial_model_samples(int(state.map_init.scale_level)),
				)
			for key, stat_key in (
				("loss", "snaps_map_loss"),
				("dist", "snaps_map_dist"),
				("vec", "snaps_map_vec"),
				("norm", "snaps_map_norm"),
				("smooth", "snaps_map_smooth"),
				("bend", "snaps_map_bend"),
				("jac", "snaps_map_jac"),
				("smooth_fwd", "snaps_map_smooth_fwd"),
				("bend_fwd", "snaps_map_bend_fwd"),
				("jac_fwd", "snaps_map_jac_fwd"),
				("metric_smooth", "snaps_map_metric_smooth"),
				("area_smooth", "snaps_map_area_smooth"),
				("smooth_rev", "snaps_map_smooth_rev"),
				("bend_rev", "snaps_map_bend_rev"),
				("jac_rev", "snaps_map_jac_rev"),
				("jac_min", "snaps_map_jmin"),
				("jac_inv_min", "snaps_map_jinv_min"),
				("prior", "snaps_map_prior"),
				("reg", "snaps_map_reg"),
				("jac_bad", "snaps_map_jbad"),
				("jac_bad_frac", "snaps_map_jbadf"),
				("jac_inv_bad", "snaps_map_jinv_bad"),
				("samples", "snaps_map_samples"),
				("uv_bad", "snaps_map_uvbad"),
				("model_bad", "snaps_map_model_bad"),
				("step_bad_quad", "snaps_map_step_bad"),
			):
				if key in last_terms:
					stats[stat_key] = float(last_terms[key].detach().cpu())
	stats["snaps_map_active"] = float(state.map_init.active_count())
	stats["snaps_map_added"] = float(state.map_init.added_total)
	stats["snaps_map_blocked"] = float(int(state.map_init.blocked_quad.sum().detach().cpu())) if state.map_init.blocked_quad is not None else 0.0
	stats["snaps_map_sparse"] = float(state.map_init.sparse_pruned_total)
	stats["snaps_map_iters"] = float(state.map_init.total_iters)
	stats["snaps_map_blocks"] = float(state.map_init.opt_blocks)
	stats["snaps_map_grow"] = float(state.map_init.grow_steps)
	stats["snaps_map_global"] = float(state.map_init.global_opt_blocks)
	stats["snaps_map_rim"] = float(state.map_init.rim_only_blocks)
	stats["snaps_map_rim_problem"] = float(state.map_init.rim_problem_blocks)
	stats["snaps_map_add_loss"] = (
		float(state.map_init.add_sample_loss_sum) / float(max(1.0, state.map_init.add_sample_weight))
	)
	stats["snaps_map_add_bad_frac"] = (
		float(state.map_init.add_bad_samples) / float(max(1.0, state.map_init.add_total_samples))
	)
	stats["snaps_map_add_success_frac"] = (
		float(state.map_init.add_success_quads) / float(max(1.0, state.map_init.add_total_quads))
		if state.map_init.add_total_quads > 0.0 else 0.0
	)
	stats["snaps_map_fringe_loss"] = (
		float(state.map_init.fringe_sample_loss_sum) / float(max(1.0, state.map_init.fringe_sample_weight))
	)
	stats["snaps_map_fringe_bad_frac"] = (
		float(state.map_init.fringe_bad_samples) / float(max(1.0, state.map_init.fringe_total_samples))
	)
	stats["snaps_map_fringe_success_frac"] = (
		float(state.map_init.fringe_success_quads) / float(max(1.0, state.map_init.fringe_total_quads))
		if state.map_init.fringe_total_quads > 0.0 else 0.0
	)
	stats["snaps_map_nsign"] = float(state.map_init.normal_sign)
	stats["snaps_map_scales"] = float(state.map_init.scale_levels_used)
	stats["snaps_map_repair"] = float(state.map_init.repair_blocks)
	state.map_init.done = True
	state.map_init.stats = stats
	_map_init_log(
		"done "
		f"active={stats['snaps_map_active']:.0f} "
		f"added_total={stats['snaps_map_added']:.0f} "
		f"blocked={stats['snaps_map_blocked']:.0f} "
		f"sparse_pruned={stats['snaps_map_sparse']:.0f} "
		f"iters={stats['snaps_map_iters']:.0f} "
		f"grow_steps={stats['snaps_map_grow']:.0f} "
		f"global_blocks={stats['snaps_map_global']:.0f} "
		f"rim_only_blocks={stats['snaps_map_rim']:.0f} "
		f"repair_blocks={stats['snaps_map_repair']:.0f} "
		f"jac_bad={stats['snaps_map_jbad']:.0f} "
		f"rjac_bad={stats['snaps_map_jinv_bad']:.0f} "
		f"model_bad={stats['snaps_map_model_bad']:.0f} "
		f"loss={stats['snaps_map_loss']:.6g} "
		f"normal_sign={state.map_init.normal_sign}"
	)
	return dict(stats)


def _surface_records_from_res(res: fit_model.FitResult3D) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
	records = getattr(res, "ext_surfaces", None)
	if records is not None:
		return list(records)
	out = []
	if res.ext_conn is None:
		return out
	for item in res.ext_conn:
		ext_xyz = item[2][0].detach()
		ext_normals = item[3][0].detach()
		corner_valid = (
			torch.isfinite(ext_xyz).all(dim=-1) &
			torch.isfinite(ext_normals).all(dim=-1) &
			(ext_normals.norm(dim=-1) > 1.0e-8)
		)
		if len(item) >= 7:
			quad_valid = item[6][0, 0].bool().detach()
		elif int(ext_xyz.shape[0]) > 1 and int(ext_xyz.shape[1]) > 1:
			quad_valid = (
				corner_valid[:-1, :-1] &
				corner_valid[1:, :-1] &
				corner_valid[:-1, 1:] &
				corner_valid[1:, 1:]
			)
		else:
			quad_valid = torch.zeros(0, 0, device=ext_xyz.device, dtype=torch.bool)
		out.append((ext_xyz, corner_valid.detach(), ext_normals, quad_valid))
	return out


def _debug_obj_safe_label(label: str | None) -> str:
	raw = "snap" if label is None or not str(label).strip() else str(label).strip()
	return "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in raw)


def _debug_obj_iter_dir(cfg: SnapSurfConfig) -> Path | None:
	if not cfg.debug_obj_dir or _debug_step is None:
		return None
	if int(_debug_step) % max(1, int(cfg.debug_obj_interval)) != 0:
		return None
	label = _debug_obj_safe_label(_debug_label)
	return Path(cfg.debug_obj_dir) / f"{label}_step{int(_debug_step):06d}"


def _write_obj_mesh_2d(path: Path, xyz: torch.Tensor, valid: torch.Tensor) -> None:
	xyz_cpu = xyz.detach().cpu()
	valid_cpu = (valid.detach().cpu().bool() & torch.isfinite(xyz_cpu).all(dim=-1))
	H, W = int(xyz_cpu.shape[0]), int(xyz_cpu.shape[1])
	vid: dict[tuple[int, int], int] = {}
	lines: list[str] = ["# snap_surf external surface\n"]
	next_id = 1
	for h in range(H):
		for w in range(W):
			if not bool(valid_cpu[h, w]):
				continue
			p = xyz_cpu[h, w].tolist()
			vid[(h, w)] = next_id
			next_id += 1
			lines.append(f"v {p[0]:.9g} {p[1]:.9g} {p[2]:.9g}\n")
	for h in range(max(0, H - 1)):
		for w in range(max(0, W - 1)):
			keys = ((h, w), (h + 1, w), (h + 1, w + 1), (h, w + 1))
			if all(k in vid for k in keys):
				lines.append("f " + " ".join(str(vid[k]) for k in keys) + "\n")
	path.write_text("".join(lines), encoding="utf-8")


def _write_obj_mesh_3d_surfaces(path: Path, xyz: torch.Tensor, valid: torch.Tensor) -> None:
	xyz_cpu = xyz.detach().cpu()
	valid_cpu = (valid.detach().cpu().bool() & torch.isfinite(xyz_cpu).all(dim=-1))
	D, H, W = int(xyz_cpu.shape[0]), int(xyz_cpu.shape[1]), int(xyz_cpu.shape[2])
	vid: dict[tuple[int, int, int], int] = {}
	lines: list[str] = ["# snap_surf model surface\n"]
	next_id = 1
	for d in range(D):
		lines.append(f"o model_d{d:03d}\n")
		for h in range(H):
			for w in range(W):
				if not bool(valid_cpu[d, h, w]):
					continue
				p = xyz_cpu[d, h, w].tolist()
				vid[(d, h, w)] = next_id
				next_id += 1
				lines.append(f"v {p[0]:.9g} {p[1]:.9g} {p[2]:.9g}\n")
		for h in range(max(0, H - 1)):
			for w in range(max(0, W - 1)):
				keys = ((d, h, w), (d, h + 1, w), (d, h + 1, w + 1), (d, h, w + 1))
				if all(k in vid for k in keys):
					lines.append("f " + " ".join(str(vid[k]) for k in keys) + "\n")
	path.write_text("".join(lines), encoding="utf-8")


def _write_obj_lines(path: Path, a: torch.Tensor, b: torch.Tensor, *, label: str) -> None:
	a_cpu = a.detach().cpu()
	b_cpu = b.detach().cpu()
	finite = torch.isfinite(a_cpu).all(dim=-1) & torch.isfinite(b_cpu).all(dim=-1)
	lines: list[str] = [f"# snap_surf {label}\n", f"o {label}\n"]
	vid = 1
	for p0, p1, ok in zip(a_cpu, b_cpu, finite, strict=False):
		if not bool(ok):
			continue
		q0 = p0.tolist()
		q1 = p1.tolist()
		lines.append(f"v {q0[0]:.9g} {q0[1]:.9g} {q0[2]:.9g}\n")
		lines.append(f"v {q1[0]:.9g} {q1[1]:.9g} {q1[2]:.9g}\n")
		lines.append(f"l {vid} {vid + 1}\n")
		vid += 2
	path.write_text("".join(lines), encoding="utf-8")


def _write_obj_points(path: Path, points: torch.Tensor, valid: torch.Tensor, *, label: str) -> None:
	points_cpu = points.detach().cpu()
	valid_cpu = valid.detach().cpu().bool() & torch.isfinite(points_cpu).all(dim=-1)
	lines: list[str] = [f"# snap_surf {label}\n", f"o {label}\n"]
	for p, ok in zip(points_cpu.reshape(-1, 3), valid_cpu.reshape(-1), strict=False):
		if not bool(ok):
			continue
		q = p.tolist()
		lines.append(f"v {q[0]:.9g} {q[1]:.9g} {q[2]:.9g}\n")
	path.write_text("".join(lines), encoding="utf-8")


def _debug_write_snap_objs(
	*,
	cfg: SnapSurfConfig,
	surface_index: int,
	surface_count: int,
	model_xyz: torch.Tensor,
	model_valid: torch.Tensor,
	ext_xyz: torch.Tensor,
	ext_valid: torch.Tensor,
	state: _SurfaceState,
) -> None:
	iter_dir = _debug_obj_iter_dir(cfg)
	if iter_dir is None:
		return
	iter_dir.mkdir(parents=True, exist_ok=True)
	prefix = "" if int(surface_count) == 1 else f"surf{int(surface_index):03d}_"
	_write_obj_mesh_2d(iter_dir / f"{prefix}ext_surface.obj", ext_xyz, ext_valid)
	_write_obj_mesh_3d_surfaces(iter_dir / f"{prefix}model_surface.obj", model_xyz, model_valid)

	if state.model_to_ext.map is not None and state.model_to_ext.valid is not None and state.model_to_ext.count() > 0:
		idx = state.model_to_ext.valid.nonzero(as_tuple=False)
		coords = state.model_to_ext.map[state.model_to_ext.valid]
		ok = torch.isfinite(coords).all(dim=-1) & _quad_valid_at_coords(ext_valid.bool(), coords, tuple(int(v) for v in ext_xyz.shape[:2]))
		if bool(ok.any().detach().cpu()):
			src = _points_at_indices(model_xyz, idx[ok])
			tgt = _sample_surface_grid(ext_xyz, coords[ok])
			_write_obj_lines(iter_dir / f"{prefix}corr_model_to_ext.obj", src, tgt, label="corr_model_to_ext")
		else:
			_write_obj_lines(iter_dir / f"{prefix}corr_model_to_ext.obj", model_xyz.new_empty(0, 3), model_xyz.new_empty(0, 3), label="corr_model_to_ext")
	else:
		_write_obj_lines(iter_dir / f"{prefix}corr_model_to_ext.obj", model_xyz.new_empty(0, 3), model_xyz.new_empty(0, 3), label="corr_model_to_ext")

	if state.ext_to_model.map is not None and state.ext_to_model.valid is not None and state.ext_to_model.count() > 0:
		idx = state.ext_to_model.valid.nonzero(as_tuple=False)
		coords = state.ext_to_model.map[state.ext_to_model.valid]
		ok = torch.isfinite(coords).all(dim=-1) & _quad_valid_at_coords(model_valid.bool(), coords, tuple(int(v) for v in model_xyz.shape[:3]))
		if bool(ok.any().detach().cpu()):
			src = _points_at_indices(ext_xyz, idx[ok])
			tgt = _sample_surface_grid(model_xyz, coords[ok])
			_write_obj_lines(iter_dir / f"{prefix}corr_ext_to_model.obj", src, tgt, label="corr_ext_to_model")
		else:
			_write_obj_lines(iter_dir / f"{prefix}corr_ext_to_model.obj", model_xyz.new_empty(0, 3), model_xyz.new_empty(0, 3), label="corr_ext_to_model")
	else:
		_write_obj_lines(iter_dir / f"{prefix}corr_ext_to_model.obj", model_xyz.new_empty(0, 3), model_xyz.new_empty(0, 3), label="corr_ext_to_model")


def _debug_write_map_init_objs(
	*,
	cfg: SnapSurfConfig,
	surface_index: int,
	surface_count: int,
	model_xyz: torch.Tensor,
	model_valid: torch.Tensor,
	ext_xyz: torch.Tensor,
	ext_valid: torch.Tensor,
	state: _SurfaceState,
	snapshot_name: str | None = None,
) -> None:
	iter_dir = _debug_obj_iter_dir(cfg)
	if iter_dir is None:
		_map_init_log(
			"obj skip "
			f"debug_obj_dir={cfg.debug_obj_dir!r} "
			f"debug_step={_debug_step} "
			f"debug_obj_interval={cfg.debug_obj_interval}"
		)
		return
	if snapshot_name is not None:
		iter_dir = iter_dir / "map_init_scales" / _debug_obj_safe_label(snapshot_name)
	iter_dir.mkdir(parents=True, exist_ok=True)
	_map_init_log(f"obj write dir={iter_dir}")
	prefix = "" if int(surface_count) == 1 else f"surf{int(surface_index):03d}_"

	def paths(name: str) -> list[Path]:
		out = [iter_dir / f"{prefix}{name}"]
		if int(surface_count) > 1 and int(surface_index) == 0:
			out.append(iter_dir / name)
		return out

	def write_mesh_2d(name: str, xyz: torch.Tensor, valid: torch.Tensor) -> None:
		for path in paths(name):
			_write_obj_mesh_2d(path, xyz, valid)

	def write_mesh_3d(name: str, xyz: torch.Tensor, valid: torch.Tensor) -> None:
		for path in paths(name):
			_write_obj_mesh_3d_surfaces(path, xyz, valid)

	def write_lines(name: str, a: torch.Tensor, b: torch.Tensor, *, label: str) -> None:
		for path in paths(name):
			_write_obj_lines(path, a, b, label=label)

	def write_points(name: str, points: torch.Tensor, valid: torch.Tensor, *, label: str) -> None:
		for path in paths(name):
			_write_obj_points(path, points, valid, label=label)

	write_mesh_2d("ext_surface.obj", ext_xyz, ext_valid)
	write_mesh_3d("model_surface.obj", model_xyz, model_valid)
	mi = state.map_init
	empty = model_xyz.new_empty(0, 3)
	write_lines("corr_model_to_ext.obj", empty, empty, label="map_init_no_corr_model_to_ext")
	write_lines("corr_ext_to_model.obj", empty, empty, label="map_init_no_corr_ext_to_model")
	if (
		mi.active_quad is None or mi.uv is None or mi.ext_pos is None or
		mi.ext_normals is None or mi.ext_valid is None or
		mi.model_depth is None or mi.active_count() <= 0
	):
		write_mesh_2d("map_mapped_surface.obj", model_xyz.new_empty(0, 0, 3), torch.zeros(0, 0, device=model_xyz.device, dtype=torch.bool))
		write_lines("map_ext_to_model.obj", empty, empty, label="map_ext_to_model")
		write_points("map_active_mask.obj", empty, torch.zeros(0, device=model_xyz.device, dtype=torch.bool), label="map_active_mask")
		return
	s = max(1, int(cfg.map_init.subdiv))
	H_ext, W_ext = int(mi.uv.shape[0]), int(mi.uv.shape[1])
	Hs, Ws = max(0, H_ext - 1) * s, max(0, W_ext - 1) * s
	mapped_grid = torch.full((Hs, Ws, 3), float("nan"), device=model_xyz.device, dtype=model_xyz.dtype)
	ext_grid = torch.full((Hs, Ws, 3), float("nan"), device=model_xyz.device, dtype=model_xyz.dtype)
	ok = torch.zeros(Hs, Ws, device=model_xyz.device, dtype=torch.bool)
	quad_hw = mi.active_quad.bool().nonzero(as_tuple=False)
	if int(quad_hw.shape[0]) > 0:
		uv_samples, ext_samples, _n_ext, sample_ext_ok, quad_uv_ok = _map_init_quad_sample_tensors(
			uv_full=mi.uv,
			ext_pos=mi.ext_pos,
			ext_normals=mi.ext_normals,
			ext_valid=mi.ext_valid,
			ext_quad_valid=mi.ext_quad_valid,
			ext_coords=mi.ext_coords,
			quad_hw=quad_hw,
			subdiv=s,
		)
		coords3 = _map_init_coords3(uv_samples, depth=int(mi.model_depth))
		safe_coords = torch.where(torch.isfinite(coords3), coords3, torch.zeros_like(coords3))
		mapped = _sample_surface_grid(model_xyz, safe_coords)
		model_ok = _quad_valid_at_coords(model_valid.bool(), safe_coords, tuple(int(v) for v in model_xyz.shape[:3]))
		sample_ok = (
			sample_ext_ok &
			quad_uv_ok.unsqueeze(-1) &
			torch.isfinite(uv_samples).all(dim=-1) &
			model_ok &
			torch.isfinite(mapped).all(dim=-1)
		)
		sample_idx = torch.arange(s * s, device=model_xyz.device, dtype=torch.long)
		rows = quad_hw[:, 0:1] * s + (sample_idx // s).view(1, -1)
		cols = quad_hw[:, 1:2] * s + (sample_idx % s).view(1, -1)
		mapped_grid[rows, cols] = torch.where(sample_ok.unsqueeze(-1), mapped, torch.full_like(mapped, float("nan")))
		ext_grid[rows, cols] = torch.where(sample_ok.unsqueeze(-1), ext_samples, torch.full_like(ext_samples, float("nan")))
		ok[rows, cols] = sample_ok
	write_mesh_2d("map_mapped_surface.obj", mapped_grid, ok)
	if bool(ok.any().detach().cpu()):
		write_lines("map_ext_to_model.obj", ext_grid[ok], mapped_grid[ok], label="map_ext_to_model")
	else:
		write_lines("map_ext_to_model.obj", empty, empty, label="map_ext_to_model")
	write_points("map_active_mask.obj", ext_grid, ok, label="map_active_mask")
	_map_init_log(
		"obj wrote "
		f"prefix={prefix!r} "
		f"snapshot={snapshot_name!r} "
		f"active={int(mi.active_quad.sum().detach().cpu())} "
		f"uv_finite={int(torch.isfinite(mi.uv).all(dim=-1).sum().detach().cpu())} "
		f"model_ok={int(ok.sum().detach().cpu())} "
		f"mapped={int(ok.sum().detach().cpu())} "
		"files=map_ext_to_model.obj,map_mapped_surface.obj,map_active_mask.obj"
	)


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
		_map_init_log(
			"loss call "
			f"stage={_stage_label!r} "
			f"surfaces={len(records)} "
			f"debug_step={_debug_step} "
			f"debug_label={_debug_label!r} "
			f"debug_dir={_debug_obj_iter_dir(cfg)}"
		)
		z = res.xyz_lr.sum() * 0.0
		lm_zero = torch.zeros(res.xyz_lr.shape[:3], device=device, dtype=dtype).unsqueeze(1)
		mask_zero = torch.zeros_like(lm_zero)
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
		_last_stats = stats
		return z, (lm_zero,), (mask_zero,)

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
