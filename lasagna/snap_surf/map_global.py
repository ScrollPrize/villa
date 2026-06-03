from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import json
import math
from pathlib import Path
import time
from typing import Any

import torch
from progress_table import (
	ProgressColumn,
	format_progress_value,
	print_progress_legend,
	progress_header,
	progress_row,
	progress_widths,
)
from .config import SnapSurfConfig, SnapSurfMapInitConfig, _parse_map_init_config
from .debug_obj import _debug_obj_safe_label, _write_obj_lines, _write_obj_mesh_2d, _write_obj_mesh_3d_surfaces, _write_obj_points
from .legacy import (
	_closest_external_seed_surface,
	_closest_model_surface_quad,
	_closest_point_uv_on_model_quad,
	_huber,
)
from .map_fixture_io import MapFixture, _float_tif, _write_json, compare_map_tensors, export_map_fixture, load_map_fixture
from .map_objective import (
	_map_init_distance_multiplier,
	_map_init_lifted_z_heading_field,
	_map_init_objective,
	_map_init_sample_scalar_quad_field,
)
from .map_pyramid import (
	_map_init_coords3,
	_map_init_dyadic_level_shape,
	_map_init_dyadic_strides,
	_map_init_integrate_dyadic_uv_pyramid,
	_map_init_uv_pyr_from_dense,
)
from .state import _SurfaceState
from .tensor import _quad_valid_at_coords, _sample_surface_grid


_PRINTED_PROGRESS_LEGENDS: set[str] = set()
_CacheStats = dict[str, float]
_NO_Z_LIFT = object()


def _sync_timing_device(*values: Any) -> None:
	for value in values:
		if torch.is_tensor(value) and value.is_cuda:
			torch.cuda.synchronize(value.device)
			return


@dataclass(frozen=True)
class _LrAutoscaleConfig:
	enabled: bool = False
	window: int = 10
	eps: float = 1.0e-12
	min_scale: float | None = None
	max_scale: float | None = None


class _LrAutoscaleState:
	def __init__(self, cfg: _LrAutoscaleConfig) -> None:
		self.cfg = cfg
		self.losses: list[torch.Tensor] = []
		self.last_scale: torch.Tensor | None = None

	def scale_for_loss(self, loss: torch.Tensor) -> torch.Tensor:
		cfg = self.cfg
		value = loss.detach()
		if value.ndim != 0:
			value = value.mean()
		value = torch.where(
			torch.isfinite(value),
			value.abs().clamp_min(float(cfg.eps)),
			value.new_tensor(1.0),
		)
		self.losses.append(value)
		if len(self.losses) > int(cfg.window):
			del self.losses[:len(self.losses) - int(cfg.window)]
		max_loss = torch.stack(self.losses).max().clamp_min(float(cfg.eps))
		scale = value.new_tensor(1.0) / max_loss
		if cfg.min_scale is not None or cfg.max_scale is not None:
			min_scale = -float("inf") if cfg.min_scale is None else float(cfg.min_scale)
			max_scale = float("inf") if cfg.max_scale is None else float(cfg.max_scale)
			scale = scale.clamp(min=min_scale, max=max_scale)
		self.last_scale = scale
		return scale


def _print_progress_legend_once(*, prefix: str, items: list[tuple[str, str]]) -> None:
	if prefix in _PRINTED_PROGRESS_LEGENDS:
		return
	_PRINTED_PROGRESS_LEGENDS.add(prefix)
	print_progress_legend(prefix=prefix, items=items)


def _truthy_arg(value: Any) -> bool:
	if isinstance(value, bool):
		return value
	if value is None:
		return False
	if isinstance(value, (int, float)):
		return value != 0
	return str(value).strip().lower() not in {"", "0", "false", "no", "off"}


def _obj_outputs_enabled(cfg: GlobalMapConfig, stage: GlobalMapStageConfig | None = None) -> bool:
	keys = ("write_objs", "debug_objs", "write_debug_objs", "write_stage_objs")
	if stage is not None:
		for key in keys:
			if key in stage.args:
				return _truthy_arg(stage.args.get(key))
	for key in keys:
		if key in cfg.base:
			return _truthy_arg(cfg.base.get(key))
	return False


@dataclass(frozen=True)
class GlobalMapStageConfig:
	name: str = ""
	steps: int = 0
	lr: float = 0.05
	params: tuple[str, ...] = ()
	min_scaledown: int = 0
	w_fac: float | dict[str, float] = 1.0
	args: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GlobalMapConfig:
	base: dict[str, Any] = field(default_factory=dict)
	stages: tuple[GlobalMapStageConfig, ...] = ()


@dataclass(frozen=True)
class SeedQuadAffineInitResult:
	affine: torch.Tensor
	sign: int
	ext_quad: tuple[int, int]
	model_quad: tuple[int, int, int]
	sampled_count: int
	hit_count: int
	kept_count: int
	rejected_far_count: int
	model_quad_count: int
	ext_step_h: float
	ext_step_w: float
	model_step_h: float
	model_step_w: float
	model_radius_h: int
	model_radius_w: int
	seed_uv_rmse: float
	seed_uv_max: float
	seed_valid_count: int
	seed_loss: float
	seed_dist: float
	seed_dist_avg: float
	seed_vec: float
	seed_norm: float
	seed_turn: float
	seed_turn_smp: int


def _lr_warmup_steps(args: dict[str, Any] | None) -> int:
	args = args or {}
	raw = args.get("lr_warmup_steps", args.get("warmup_steps", 0))
	return max(0, int(raw))


def _lr_warmup_factor(*, step1: int, warmup_steps: int) -> float:
	warmup_steps = max(0, int(warmup_steps))
	if warmup_steps <= 0:
		return 1.0
	return min(1.0, max(0.0, float(step1) / float(warmup_steps)))


def _optional_positive_float(value: Any, *, name: str) -> float | None:
	if value is None:
		return None
	out = float(value)
	if out <= 0.0:
		raise ValueError(f"{name} must be > 0")
	return out


def _lr_autoscale_config(args: dict[str, Any] | None) -> _LrAutoscaleConfig:
	args = args or {}
	raw = args.get("lr_autoscale", args.get("lr_auto_scale", False))
	raw_cfg: dict[str, Any] = {}
	if isinstance(raw, dict):
		enabled = _truthy_arg(raw.get("enabled", True))
		raw_cfg.update(raw)
	else:
		enabled = _truthy_arg(raw)
	window = int(raw_cfg.get(
		"window",
		raw_cfg.get("steps", args.get("lr_autoscale_window", args.get("lr_autoscale_steps", 10))),
	))
	if window <= 0:
		raise ValueError("lr_autoscale_window must be > 0")
	eps = float(raw_cfg.get("eps", args.get("lr_autoscale_eps", 1.0e-12)))
	if eps <= 0.0:
		raise ValueError("lr_autoscale_eps must be > 0")
	min_scale = _optional_positive_float(
		raw_cfg.get("min_scale", args.get("lr_autoscale_min_scale", None)),
		name="lr_autoscale_min_scale",
	)
	max_scale = _optional_positive_float(
		raw_cfg.get("max_scale", args.get("lr_autoscale_max_scale", None)),
		name="lr_autoscale_max_scale",
	)
	if min_scale is not None and max_scale is not None and min_scale > max_scale:
		raise ValueError("lr_autoscale_min_scale must be <= lr_autoscale_max_scale")
	return _LrAutoscaleConfig(
		enabled=enabled,
		window=window,
		eps=eps,
		min_scale=min_scale,
		max_scale=max_scale,
	)


def _capture_optimizer_target_lrs(opt: torch.optim.Optimizer) -> None:
	for group in opt.param_groups:
		if "_target_lr" not in group:
			group["_target_lr"] = float(group.get("lr", 0.0))


def _apply_optimizer_lr_warmup(opt: torch.optim.Optimizer, *, step1: int, warmup_steps: int) -> None:
	if int(warmup_steps) <= 0:
		return
	scale = _lr_warmup_factor(step1=int(step1), warmup_steps=int(warmup_steps))
	for group in opt.param_groups:
		if "_target_lr" not in group:
			group["_target_lr"] = float(group.get("lr", 0.0))
		target_lr = float(group["_target_lr"])
		group["lr"] = target_lr * scale


def _make_lr_autoscale_state(args: dict[str, Any] | None) -> _LrAutoscaleState | None:
	cfg = _lr_autoscale_config(args)
	return _LrAutoscaleState(cfg) if cfg.enabled else None


def _apply_optimizer_lr_schedule(
	opt: torch.optim.Optimizer,
	*,
	step1: int,
	warmup_steps: int,
	autoscale: _LrAutoscaleState | None,
	loss: torch.Tensor,
) -> None:
	warmup = _lr_warmup_factor(step1=int(step1), warmup_steps=int(warmup_steps))
	auto_scale = autoscale.scale_for_loss(loss) if autoscale is not None else None
	for group in opt.param_groups:
		if "_target_lr" not in group:
			group["_target_lr"] = float(group.get("lr", 0.0))
		target_lr = float(group["_target_lr"])
		lr = target_lr * warmup
		group["lr"] = auto_scale * lr if auto_scale is not None else lr


def _optimizer_lr_for_display(opt: torch.optim.Optimizer | None, fallback: float) -> float:
	if opt is None or not opt.param_groups:
		return float(fallback)
	return float(opt.param_groups[0].get("lr", fallback))


def _lr_autoscale_stats(prefix: str, autoscale: _LrAutoscaleState | None) -> dict[str, float]:
	if autoscale is None or autoscale.last_scale is None:
		return {}
	return {
		f"{prefix}_lr_autoscale": float(autoscale.last_scale.detach().cpu()),
		f"{prefix}_lr_autoscale_window": float(len(autoscale.losses)),
	}


def _public_stage_param(name: str) -> str:
	return {"affine": "map_surf_affine", "map_uv_ms": "map_surf_ms"}.get(str(name), str(name))


def _public_stage_params(params: tuple[str, ...]) -> tuple[str, ...]:
	return tuple(_public_stage_param(p) for p in params)


def _stage_param_label(params: tuple[str, ...], *, fallback: str) -> str:
	public = _public_stage_params(params)
	return "_".join(public) if public else fallback


def _stage_w_fac_label(w_fac: float | dict[str, float]) -> str:
	if isinstance(w_fac, dict):
		return json.dumps({str(k): float(v) for k, v in sorted(w_fac.items())}, sort_keys=True, separators=(",", ":"))
	return format_progress_value(float(w_fac))


def _canonical_stage_params(params: tuple[str, ...], *, normal_lasagna: bool = False) -> tuple[str, ...]:
	out: list[str] = []
	replacements = {
		"affine": "map_surf_affine",
		"map_affine": "map_surf_affine",
		"map_uv_ms": "map_surf_ms",
	}
	for p in params:
		name = str(p)
		if name in replacements:
			raise ValueError(f"global map stage params: use '{replacements[name]}' instead of '{name}'")
		if name == "map_surf_affine":
			name = "affine"
		elif name == "map_surf_ms":
			name = "map_uv_ms"
		out.append(name)
	bad = sorted(set(out) - {"affine", "map_uv_ms"})
	if bad:
		raise ValueError(f"global map stage params: unknown name(s): {bad}")
	if "affine" in out and "map_uv_ms" in out:
		raise ValueError("global map stage params: map_surf_affine and map_surf_ms must be optimized in separate stages")
	return tuple(out)


_STAGE_W_FAC_KEYS = {
	"dist": "dist",
	"map_dist": "dist",
	"vec": "vec",
	"map_vec_normal": "vec",
	"norm": "norm",
	"map_surface_normal": "norm",
	"turn": "turn",
	"map_z_lift": "turn",
	"smooth": "smooth",
	"map_smooth": "smooth",
	"bend": "bend",
	"map_bend": "bend",
	"jac": "jac",
	"map_jac": "jac",
	"metric_smooth": "metric_smooth",
	"map_metric_smooth": "metric_smooth",
	"area_smooth": "area_smooth",
	"map_area_smooth": "area_smooth",
	"prior": "prior",
	"map_dense_prior": "prior",
	"map_station_t": "map_station_t",
	"w_station_t": "map_station_t",
}


def _canonical_stage_w_fac(raw: Any, *, index: int) -> float | dict[str, float]:
	if raw is None:
		return 1.0
	if isinstance(raw, dict):
		out: dict[str, float] = {}
		bad = sorted(set(str(k) for k in raw.keys()) - set(_STAGE_W_FAC_KEYS.keys()))
		if bad:
			raise ValueError(f"global map stage {index} w_fac: unknown term(s): {bad}")
		for k, v in raw.items():
			if v is None:
				continue
			out[_STAGE_W_FAC_KEYS[str(k)]] = float(v)
		return out
	return float(raw)


def parse_global_map_stage_item(item: dict[str, Any], *, index: int = 0, normal_lasagna: bool = False) -> GlobalMapStageConfig:
	if not isinstance(item, dict):
		raise ValueError(f"global map stage {index} must be an object")
	params = item.get("params", ())
	if isinstance(params, str):
		params_t = (params,)
	elif isinstance(params, list):
		params_t = tuple(str(v) for v in params)
	else:
		raise ValueError(f"global map stage {index} params must be a string or list")
	params_t = _canonical_stage_params(params_t, normal_lasagna=normal_lasagna)
	args = item.get("args", {})
	if args is None:
		args = {}
	if not isinstance(args, dict):
		raise ValueError(f"global map stage {index} args must be an object")
	w_fac = _canonical_stage_w_fac(item.get("w_fac", 1.0), index=index)
	return GlobalMapStageConfig(
		name=str(item.get("name", item.get("kind", ""))),
		steps=max(0, int(item.get("steps", 0))),
		lr=float(item.get("lr", 0.05)),
		params=params_t,
		min_scaledown=max(0, int(item.get("min_scaledown", 0))),
		w_fac=w_fac,
		args=dict(args),
	)


class AffineMapModel(torch.nn.Module):
	def __init__(
		self,
		*,
		ext_shape: tuple[int, int],
		device: torch.device,
		dtype: torch.dtype,
		initial: torch.Tensor | None = None,
	) -> None:
		super().__init__()
		H, W = int(ext_shape[0]), int(ext_shape[1])
		self.ext_shape = (H, W)
		hh = torch.arange(H, device=device, dtype=dtype).view(H, 1).expand(H, W)
		ww = torch.arange(W, device=device, dtype=dtype).view(1, W).expand(H, W)
		self.register_buffer("ext_hw", torch.stack([hh, ww], dim=-1).contiguous())
		if initial is None:
			initial = torch.tensor(
				[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
				device=device,
				dtype=dtype,
			)
		self.affine = torch.nn.Parameter(initial.to(device=device, dtype=dtype).clone())

	def forward(self) -> torch.Tensor:
		hw = self.ext_hw
		return (
			hw[..., 0:1] * self.affine[:, 0] +
			hw[..., 1:2] * self.affine[:, 1] +
			self.affine[:, 2]
		)

	def eval_at(self, hw: torch.Tensor) -> torch.Tensor:
		return (
			hw[..., 0:1] * self.affine[:, 0] +
			hw[..., 1:2] * self.affine[:, 1] +
			self.affine[:, 2]
		)


class GlobalMapModel(torch.nn.Module):
	def __init__(
		self,
		full_uv: torch.Tensor,
		*,
		levels: int,
		factor: int = 2,
		preserve_batch: bool | None = None,
	) -> None:
		super().__init__()
		self.preserve_batch = bool(full_uv.ndim == 4) if preserve_batch is None else bool(preserve_batch)
		self.map_uv_ms = _map_init_uv_pyr_from_dense(full_uv.detach(), levels=int(levels), factor=int(factor))

	def forward(self, *, active_level: int = 0) -> torch.Tensor:
		return _map_init_integrate_dyadic_uv_pyramid(
			list(self.map_uv_ms),
			active_level=int(active_level),
			preserve_batch=bool(self.preserve_batch),
		)


SELF_MAP_MODES = ("off", "multi_wrap_full", "multi_wrap_d")
SELF_MAP_DIRECTIONS = ("out", "in")


def normalize_self_map_init(value: object) -> str:
	mode = str(value if value is not None else "off").strip().lower().replace("-", "_")
	if mode not in SELF_MAP_MODES:
		raise ValueError(f"invalid self-map-init '{value}' (expected off, multi_wrap_full, or multi_wrap_d)")
	return mode


def self_map_pair_depths(mode: str, direction: str, depth: int) -> tuple[list[int], list[int]]:
	mode_i = normalize_self_map_init(mode)
	direction_i = str(direction).strip().lower()
	if direction_i not in SELF_MAP_DIRECTIONS:
		raise ValueError(f"invalid self-map direction '{direction}' (expected out or in)")
	D = int(depth)
	if mode_i == "off":
		return [], []
	if mode_i == "multi_wrap_full":
		if D != 1:
			raise ValueError("self-map-init=multi_wrap_full requires model depth D=1")
		return [0], [0]
	if D <= 1:
		raise ValueError("self-map-init=multi_wrap_d requires model depth D>1")
	if direction_i == "out":
		return list(range(0, D - 1)), list(range(1, D))
	return list(range(1, D)), list(range(0, D - 1))


def self_map_signed_offset(mode: str, direction: str, offset: float) -> float:
	mode_i = normalize_self_map_init(mode)
	if mode_i == "off":
		return 0.0
	direction_i = str(direction).strip().lower()
	if direction_i not in SELF_MAP_DIRECTIONS:
		raise ValueError(f"invalid self-map direction '{direction}' (expected out or in)")
	sign = 1.0 if direction_i == "out" else -1.0
	return sign * float(offset)


def self_map_initial_uv(
	*,
	mode: str,
	direction: str,
	depth: int,
	height: int,
	width: int,
	model_w_wraps: float | None = None,
	device: torch.device,
	dtype: torch.dtype,
) -> torch.Tensor:
	src, _dst = self_map_pair_depths(mode, direction, int(depth))
	N = len(src)
	H = int(height)
	W = int(width)
	if N <= 0:
		return torch.empty(0, H, W, 2, device=device, dtype=dtype)
	hh = torch.arange(H, device=device, dtype=dtype).view(1, H, 1).expand(N, H, W)
	ww = torch.arange(W, device=device, dtype=dtype).view(1, 1, W).expand(N, H, W)
	uv = torch.stack([hh, ww], dim=-1).contiguous()
	if normalize_self_map_init(mode) == "multi_wrap_full":
		wraps = 1.0 if model_w_wraps is None else float(model_w_wraps)
		if not math.isfinite(wraps) or wraps <= 0.0:
			raise ValueError("self-map-init=multi_wrap_full requires positive model_w_wraps")
		wrap_cols = float(max(1, W - 1)) / wraps
		sign = 1.0 if str(direction).strip().lower() == "out" else -1.0
		uv[..., 1] = uv[..., 1] + sign * wrap_cols
	return uv


def self_map_active_quads(
	*,
	mode: str,
	direction: str,
	model_valid: torch.Tensor,
	uv: torch.Tensor,
) -> torch.Tensor:
	if uv.ndim != 4 or int(uv.shape[-1]) != 2:
		raise ValueError("self-map uv must be (N,H,W,2)")
	src, dst = self_map_pair_depths(mode, direction, int(model_valid.shape[0]))
	N, H, W = int(uv.shape[0]), int(uv.shape[1]), int(uv.shape[2])
	if len(src) != N:
		raise ValueError(f"self-map uv batch {N} does not match topology batch {len(src)}")
	if H < 2 or W < 2:
		return torch.zeros(N, max(0, H - 1), max(0, W - 1), device=uv.device, dtype=torch.bool)
	full_h, full_w = int(model_valid.shape[1]), int(model_valid.shape[2])
	source_valid = model_valid[torch.tensor(src, device=model_valid.device, dtype=torch.long)].to(device=uv.device).bool()
	source_quad_full = (
		source_valid[:, :-1, :-1] &
		source_valid[:, 1:, :-1] &
		source_valid[:, :-1, 1:] &
		source_valid[:, 1:, 1:]
	)
	if (H, W) == (full_h, full_w):
		source_quad = source_quad_full
	else:
		level = 0
		for candidate in range(1, 32):
			try:
				if _map_init_dyadic_level_shape(full_h, full_w, candidate) == (H, W):
					level = candidate
					break
			except ValueError:
				break
		if level <= 0:
			raise ValueError(f"self-map uv shape {(H, W)} is not a dyadic level of model shape {(full_h, full_w)}")
		stride = 1 << level
		QH, QW = int(source_quad_full.shape[1]), int(source_quad_full.shape[2])
		if QH % stride != 0 or QW % stride != 0:
			raise ValueError(f"self-map source quad grid {(QH, QW)} is not divisible by dyadic stride {stride}")
		pooled = torch.nn.functional.avg_pool2d(
			source_quad_full.to(dtype=uv.dtype).unsqueeze(1),
			kernel_size=stride,
			stride=stride,
		).squeeze(1)
		source_quad = pooled >= 1.0
	target_depth = torch.tensor(dst, device=uv.device, dtype=uv.dtype).view(N, 1, 1, 1).expand(N, H, W, 1)
	coords = torch.cat([target_depth, uv], dim=-1)
	target_ok = _quad_valid_at_coords(model_valid.bool(), coords, tuple(int(v) for v in model_valid.shape))
	target_quad = target_ok[:, :-1, :-1] & target_ok[:, 1:, :-1] & target_ok[:, :-1, 1:] & target_ok[:, 1:, 1:]
	return source_quad & target_quad


def _combine_term_dicts(term_dicts: list[dict[str, torch.Tensor]], weights: list[torch.Tensor], z: torch.Tensor) -> dict[str, torch.Tensor]:
	if not term_dicts:
		return {"loss": z.detach()}
	keys = sorted(set().union(*(d.keys() for d in term_dicts)))
	total_w = z
	for w in weights:
		total_w = total_w + w
	out: dict[str, torch.Tensor] = {}
	for key in keys:
		acc = z
		for terms, w in zip(term_dicts, weights):
			acc = acc + terms.get(key, z).to(device=z.device, dtype=z.dtype) * w.to(device=z.device, dtype=z.dtype)
		out[key] = (acc / total_w.clamp_min(1.0)).detach()
	return out


def _self_map_objective_for_uv(
	*,
	uv: torch.Tensor,
	mode: str,
	direction: str,
	model_xyz: torch.Tensor,
	model_normals: torch.Tensor,
	model_valid: torch.Tensor,
	cfg: SnapSurfConfig,
	level: int,
	need_stats: bool = True,
	active_quad: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
	if uv.ndim != 4 or int(uv.shape[-1]) != 2:
		raise ValueError("self-map objective uv must be (N,H,W,2)")
	src, dst = self_map_pair_depths(mode, direction, int(model_xyz.shape[0]))
	if len(src) != int(uv.shape[0]):
		raise ValueError(f"self-map uv batch {int(uv.shape[0])} does not match topology batch {len(src)}")
	active = active_quad
	if active is None:
		active = self_map_active_quads(mode=mode, direction=direction, model_valid=model_valid, uv=uv)
	z = model_xyz.sum() * 0.0 + uv.sum() * 0.0
	losses: list[torch.Tensor] = []
	terms_list: list[dict[str, torch.Tensor]] = []
	weights: list[torch.Tensor] = []
	for i, (src_d, dst_d) in enumerate(zip(src, dst)):
		ext_coords = None
		if tuple(int(v) for v in uv.shape[1:3]) != tuple(int(v) for v in model_xyz.shape[1:3]):
			ext_coords = _level_coords(tuple(int(v) for v in model_xyz.shape[1:3]), int(level), uv[i])
		loss_i, terms_i = _map_init_objective(
			uv_full=uv[i],
			active_quad=active[i],
			ext_pos=model_xyz[int(src_d)].detach(),
			ext_normals=model_normals[int(src_d)].detach(),
			ext_valid=model_valid[int(src_d)].detach().bool(),
			ext_quad_valid=None,
			ext_coords=ext_coords,
			model_xyz=model_xyz,
			model_valid=model_valid,
			model_normals=model_normals,
			model_depth=int(dst_d),
			sign=1,
			cfg=cfg,
			allow_partial_model_samples=True,
			need_stats=bool(need_stats),
			ext_z_lift_theta=None,
			ext_z_lift_valid=None,
			model_z_lift_theta=None,
			model_z_lift_valid=None,
		)
		w = active[i].to(dtype=uv.dtype).sum().clamp_min(1.0)
		losses.append(loss_i * w)
		terms_list.append(terms_i)
		weights.append(w)
	if not losses:
		return z, {"loss": z.detach()}
	total_w = z
	total = z
	for loss_i, w in zip(losses, weights):
		total = total + loss_i
		total_w = total_w + w
	return total / total_w.clamp_min(1.0), _combine_term_dicts(terms_list, weights, z)


def parse_global_map_config(path: str | Path) -> GlobalMapConfig:
	raw = json.loads(Path(path).read_text(encoding="utf-8"))
	if not isinstance(raw, dict):
		raise ValueError("global map config must be an object")
	base = raw.get("base", {})
	if not isinstance(base, dict):
		raise ValueError("global map config base must be an object")
	stages_raw = raw.get("stages", [])
	if not isinstance(stages_raw, list):
		raise ValueError("global map config stages must be a list")
	stages: list[GlobalMapStageConfig] = []
	for i, item in enumerate(stages_raw):
		stages.append(parse_global_map_stage_item(item, index=i))
	return GlobalMapConfig(base=dict(base), stages=tuple(stages))


def snap_surf_config_from_global_config(cfg: GlobalMapConfig, stage: GlobalMapStageConfig | None = None) -> SnapSurfConfig:
	raw = dict(cfg.base)
	stage_args = dict(stage.args) if stage is not None else {}
	raw_map = dict(raw.get("map_init", {}))
	raw_map.update(stage_args.get("map_init", {}))
	if "subdiv" in stage_args:
		raw_map["subdiv"] = stage_args["subdiv"]
	for key, value in stage_args.items():
		if key == "map_init":
			continue
		if key in SnapSurfMapInitConfig.__dataclass_fields__:
			raw_map[key] = value
	raw["map_init"] = raw_map
	map_cfg = _parse_map_init_config(raw_map)
	defaults = SnapSurfConfig()
	kwargs: dict[str, Any] = {}
	for name in SnapSurfConfig.__dataclass_fields__:
		if name == "map_init":
			continue
		kwargs[name] = raw.get(name, getattr(defaults, name))
	kwargs["map_init"] = map_cfg
	return SnapSurfConfig(**kwargs)


def _full_active_quad(fixture: MapFixture) -> torch.Tensor:
	return (
		fixture.ext_valid[:-1, :-1].bool() &
		fixture.ext_valid[1:, :-1].bool() &
		fixture.ext_valid[:-1, 1:].bool() &
		fixture.ext_valid[1:, 1:].bool() &
		fixture.ext_quad_valid.bool()
	)


def _external_quad_base_valid(
	ext_xyz: torch.Tensor,
	ext_valid: torch.Tensor,
	ext_quad_valid: torch.Tensor,
) -> torch.Tensor:
	finite = torch.isfinite(ext_xyz).all(dim=-1)
	return (
		ext_valid[:-1, :-1].bool() &
		ext_valid[1:, :-1].bool() &
		ext_valid[:-1, 1:].bool() &
		ext_valid[1:, 1:].bool() &
		finite[:-1, :-1] &
		finite[1:, :-1] &
		finite[:-1, 1:] &
		finite[1:, 1:] &
		ext_quad_valid.bool()
	)

def _model_quad_base_valid(model_xyz: torch.Tensor, model_valid: torch.Tensor) -> torch.Tensor:
	D, H, W = (int(model_xyz.shape[0]), int(model_xyz.shape[1]), int(model_xyz.shape[2]))
	if H < 2 or W < 2:
		return torch.zeros(D, max(0, H - 1), max(0, W - 1), device=model_xyz.device, dtype=torch.bool)
	finite = torch.isfinite(model_xyz).all(dim=-1)
	v = model_valid.bool() & finite
	return v[:, :-1, :-1] & v[:, 1:, :-1] & v[:, :-1, 1:] & v[:, 1:, 1:]

def _fixture_seed_ext_quad(fixture: MapFixture) -> tuple[int, int] | None:
	raw_init = fixture.metadata.get("seed_quad_init")
	if isinstance(raw_init, dict):
		raw_ext = raw_init.get("ext_quad")
		if isinstance(raw_ext, (list, tuple)) and len(raw_ext) == 2:
			return int(raw_ext[0]), int(raw_ext[1])
	raw_ext = fixture.metadata.get("seed_ext_sample_hw")
	if isinstance(raw_ext, (list, tuple)) and len(raw_ext) >= 2:
		return int(raw_ext[0]), int(raw_ext[1])
	raw_seed = fixture.metadata.get("seed_xyz")
	if isinstance(raw_seed, (list, tuple)) and len(raw_seed) == 3:
		seed = torch.tensor([float(raw_seed[0]), float(raw_seed[1]), float(raw_seed[2])], device=fixture.ext_xyz.device, dtype=fixture.ext_xyz.dtype)
		quad, _point, _dist = _closest_external_seed_surface(
			seed=seed,
			ext_xyz=fixture.ext_xyz,
			ext_valid=fixture.ext_valid.bool(),
			ext_quad_valid=fixture.ext_quad_valid.bool(),
		)
		return None if quad is None else (int(quad[0]), int(quad[1]))
	return None

def _fixture_seed_model_quad(fixture: MapFixture) -> tuple[int, int, int] | None:
	raw_init = fixture.metadata.get("seed_quad_init")
	if isinstance(raw_init, dict):
		raw_model = raw_init.get("model_quad")
		if isinstance(raw_model, (list, tuple)) and len(raw_model) == 3:
			return int(raw_model[0]), int(raw_model[1]), int(raw_model[2])
	raw_model = fixture.metadata.get("seed_model_quad")
	if isinstance(raw_model, (list, tuple)) and len(raw_model) == 3:
		return int(raw_model[0]), int(raw_model[1]), int(raw_model[2])
	raw_seed = fixture.metadata.get("seed_xyz")
	if isinstance(raw_seed, (list, tuple)) and len(raw_seed) == 3:
		seed = torch.tensor([float(raw_seed[0]), float(raw_seed[1]), float(raw_seed[2])], device=fixture.model_xyz.device, dtype=fixture.model_xyz.dtype)
		quad, _dist = _closest_model_surface_quad(
			point=seed,
			model_xyz=fixture.model_xyz,
			model_valid=fixture.model_valid.bool(),
		)
		return None if quad is None else (int(quad[0]), int(quad[1]), int(quad[2]))
	return None

def _tensor_cache_token(t: torch.Tensor) -> tuple[Any, ...]:
	return (
		str(t.device),
		str(t.dtype),
		tuple(int(v) for v in t.shape),
		tuple(int(v) for v in t.stride()),
		int(t.storage_offset()),
		int(t.untyped_storage().data_ptr()),
		int(getattr(t, "_version", 0)),
	)

def _external_health_cfg_token(cfg: SnapSurfMapInitConfig) -> tuple[Any, ...]:
	return (
		float(cfg.ext_mesh_health_max_edge_ratio),
		float(cfg.ext_mesh_health_max_area_ratio),
		float(cfg.ext_mesh_health_min_area_ratio),
		float(cfg.ext_mesh_health_max_aspect_ratio),
		float(cfg.ext_mesh_health_max_diag_ratio),
		float(cfg.ext_mesh_health_min_triangle_normal_dot),
		float(cfg.ext_mesh_health_min_normal_dot),
		int(cfg.ext_mesh_health_reject_radius),
	)

def _external_health_cache_key(
	fixture: MapFixture,
	cfg: SnapSurfMapInitConfig,
	*,
	external_surface_index: int,
) -> tuple[Any, ...]:
	seed_raw = fixture.metadata.get("seed_ext_sample_hw")
	if isinstance(seed_raw, (list, tuple)) and len(seed_raw) == 2:
		seed_ext = (int(seed_raw[0]), int(seed_raw[1]))
	else:
		seed_ext = None
	seed_xyz_raw = fixture.metadata.get("seed_xyz")
	if isinstance(seed_xyz_raw, (list, tuple)) and len(seed_xyz_raw) == 3:
		seed_xyz = tuple(float(v) for v in seed_xyz_raw)
	else:
		seed_xyz = None
	return (
		int(external_surface_index),
		_external_health_cfg_token(cfg),
		seed_ext,
		seed_xyz,
	)

def _map_init_z_lift_for_fixture(
	fixture: MapFixture,
	cfg: SnapSurfConfig,
	*,
	sign: int | None = None,
	external_surface_index: int | None = None,
	mesh_epoch: int | None = None,
	external_cache: dict[tuple[Any, ...], dict[str, Any] | None] | None = None,
	model_cache: dict[tuple[Any, ...], dict[str, Any] | None] | None = None,
	cache_stats: _CacheStats | None = None,
) -> dict[str, Any] | None:
	t_total = time.perf_counter()
	mi = cfg.map_init
	if not bool(mi.z_lift_enabled) or float(mi.w_z_lift) <= 0.0:
		return None
	seed_ext = _fixture_seed_ext_quad(fixture)
	seed_model = _fixture_seed_model_quad(fixture)
	if seed_ext is None or seed_model is None:
		return None
	sign_i = int(fixture.metadata.get("sign", 1) or 1) if sign is None else int(sign)
	norm_xy_min = float(mi.z_lift_norm_xy_min)

	def _bump(name: str) -> None:
		if cache_stats is not None:
			cache_stats[name] = int(cache_stats.get(name, 0)) + 1

	ext_key = None
	if external_surface_index is not None:
		ext_key = (
			int(external_surface_index),
			_external_health_cache_key(fixture, mi, external_surface_index=int(external_surface_index)),
			seed_ext,
			norm_xy_min,
		)
	ext_part: dict[str, Any] | None
	t_ext = time.perf_counter()
	if external_cache is not None and ext_key is not None and ext_key in external_cache:
		_bump("zext_hit")
		ext_part = external_cache[ext_key]
	else:
		if external_cache is not None and ext_key is not None:
			_bump("zext_miss")
		ext_base = _external_quad_base_valid(fixture.ext_xyz, fixture.ext_valid.bool(), fixture.ext_quad_valid.bool())
		ext_theta, ext_valid, ext_stats = _map_init_lifted_z_heading_field(
			fixture.ext_normals,
			ext_base,
			seed_ext,
			norm_xy_min=norm_xy_min,
			sign=1,
		)
		ext_part = None if float(ext_stats["valid"]) <= 0.0 else {
			"theta_lifted": ext_theta.detach(),
			"valid": ext_valid.detach(),
			"stats": dict(ext_stats),
		}
		if external_cache is not None and ext_key is not None:
			external_cache[ext_key] = ext_part
	ext_ms = 1000.0 * (time.perf_counter() - t_ext)

	model_key = None
	if mesh_epoch is not None:
		# The model z-rotation field is valid only for the fixed map-init
		# mesh/normals represented by this epoch. Rebuild it if geometry moves.
		model_key = (
			int(mesh_epoch),
			sign_i,
			int(fixture.metadata.get("model_depth", 0) or 0),
			seed_model,
			norm_xy_min,
			tuple(int(v) for v in fixture.model_normals.shape),
			str(fixture.model_normals.device),
			str(fixture.model_normals.dtype),
		)
	model_part: dict[str, Any] | None
	t_model = time.perf_counter()
	if model_cache is not None and model_key is not None and model_key in model_cache:
		_bump("zmdl_hit")
		model_part = model_cache[model_key]
	else:
		if model_cache is not None and model_key is not None:
			_bump("zmdl_miss")
		model_base = _model_quad_base_valid(fixture.model_xyz, fixture.model_valid.bool())
		model_theta, model_valid, model_stats = _map_init_lifted_z_heading_field(
			fixture.model_normals,
			model_base,
			seed_model,
			norm_xy_min=norm_xy_min,
			sign=sign_i,
		)
		model_part = None if float(model_stats["valid"]) <= 0.0 else {
			"theta_lifted": model_theta.detach(),
			"valid": model_valid.detach(),
			"stats": dict(model_stats),
		}
		if model_cache is not None and model_key is not None:
			model_cache[model_key] = model_part
	model_ms = 1000.0 * (time.perf_counter() - t_model)

	if ext_part is None or model_part is None:
		return None
	total_ms = 1000.0 * (time.perf_counter() - t_total)
	out = {
		"ext_theta_lifted": ext_part["theta_lifted"],
		"ext_valid": ext_part["valid"],
		"model_theta_lifted": model_part["theta_lifted"],
		"model_valid": model_part["valid"],
		"stats": {
			"ext_valid": float(ext_part["stats"]["valid"]),
			"ext_invalid": float(ext_part["stats"]["invalid"]),
			"ext_unreachable": float(ext_part["stats"]["unreachable"]),
			"model_valid": float(model_part["stats"]["valid"]),
			"model_invalid": float(model_part["stats"]["invalid"]),
			"model_unreachable": float(model_part["stats"]["unreachable"]),
			"ext_ms": float(ext_ms),
			"model_ms": float(model_ms),
			"total_ms": float(total_ms),
		},
	}
	if cache_stats is not None:
		cache_stats["zext_ms"] = float(cache_stats.get("zext_ms", 0.0)) + float(ext_ms)
		cache_stats["zmdl_ms"] = float(cache_stats.get("zmdl_ms", 0.0)) + float(model_ms)
		cache_stats["z_lift_ms"] = float(cache_stats.get("z_lift_ms", 0.0)) + float(total_ms)
	return out


def _dilate_quad_mask_euclidean(mask: torch.Tensor, radius: int) -> torch.Tensor:
	r = max(0, int(radius))
	if r <= 0 or int(mask.numel()) == 0:
		return mask.bool()
	H, W = int(mask.shape[0]), int(mask.shape[1])
	src = mask.bool()
	out = torch.zeros_like(src)
	for dh in range(-r, r + 1):
		for dw in range(-r, r + 1):
			if dh * dh + dw * dw > r * r:
				continue
			src_h0 = max(0, -dh)
			src_h1 = H - max(0, dh)
			src_w0 = max(0, -dw)
			src_w1 = W - max(0, dw)
			if src_h0 >= src_h1 or src_w0 >= src_w1:
				continue
			dst_h0 = max(0, dh)
			dst_h1 = H - max(0, -dh)
			dst_w0 = max(0, dw)
			dst_w1 = W - max(0, -dw)
			out[dst_h0:dst_h1, dst_w0:dst_w1] |= src[src_h0:src_h1, src_w0:src_w1]
	return out


def _external_quad_health_filter(
	*,
	ext_xyz: torch.Tensor,
	ext_valid: torch.Tensor,
	ext_quad_valid: torch.Tensor,
	ext_normals: torch.Tensor | None,
	cfg: SnapSurfMapInitConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
	base = _external_quad_base_valid(ext_xyz, ext_valid, ext_quad_valid)
	empty_stats = {
		"quads_input": float(int(base.sum().detach().cpu())),
		"quads_kept": 0.0,
		"quads_rejected": float(int(base.sum().detach().cpu())),
		"median_edge": 0.0,
		"median_area": 0.0,
		"max_edge": 0.0,
		"max_area": 0.0,
	}
	if not bool(base.any().detach().cpu()):
		return ext_quad_valid.bool() & base, empty_stats

	p00 = ext_xyz[:-1, :-1]
	p10 = ext_xyz[1:, :-1]
	p01 = ext_xyz[:-1, 1:]
	p11 = ext_xyz[1:, 1:]
	e0 = (p10 - p00).norm(dim=-1)
	e1 = (p11 - p01).norm(dim=-1)
	e2 = (p01 - p00).norm(dim=-1)
	e3 = (p11 - p10).norm(dim=-1)
	edges = torch.stack([e0, e1, e2, e3], dim=-1)
	edge_min = edges.min(dim=-1).values
	edge_max = edges.max(dim=-1).values
	d0 = (p11 - p00).norm(dim=-1)
	d1 = (p10 - p01).norm(dim=-1)
	diag_min = torch.minimum(d0, d1)
	diag_max = torch.maximum(d0, d1)
	tri0 = torch.linalg.cross(p10 - p00, p01 - p00, dim=-1)
	tri1 = torch.linalg.cross(p01 - p11, p10 - p11, dim=-1)
	area0 = tri0.norm(dim=-1) * 0.5
	area1 = tri1.norm(dim=-1) * 0.5
	area = area0 + area1
	tri_dot = (
		torch.nn.functional.normalize(tri0, dim=-1, eps=1.0e-8) *
		torch.nn.functional.normalize(tri1, dim=-1, eps=1.0e-8)
	).sum(dim=-1)
	finite_metrics = (
		torch.isfinite(edges).all(dim=-1) &
		torch.isfinite(d0) &
		torch.isfinite(d1) &
		torch.isfinite(area) &
		torch.isfinite(tri_dot)
	)
	positive = finite_metrics & (edge_min > 0.0) & (diag_min > 0.0) & (area > 0.0)
	usable = base & positive
	if not bool(usable.any().detach().cpu()):
		stats = dict(empty_stats)
		stats["quads_rejected"] = float(int(base.sum().detach().cpu()))
		return ext_quad_valid.bool() & usable, stats

	edge_ref = edges[usable.unsqueeze(-1).expand_as(edges)].median()
	area_ref = area[usable].median()
	eps = torch.finfo(ext_xyz.dtype).eps if ext_xyz.dtype.is_floating_point else 1.0e-12
	edge_ref_safe = torch.clamp(edge_ref, min=float(eps))
	area_ref_safe = torch.clamp(area_ref, min=float(eps))
	aspect = edge_max / torch.clamp(edge_min, min=float(eps))
	diag_ratio = diag_max / torch.clamp(diag_min, min=float(eps))
	healthy = usable
	if float(cfg.ext_mesh_health_max_aspect_ratio) > 0.0:
		healthy = healthy & (aspect <= float(cfg.ext_mesh_health_max_aspect_ratio))
	if float(cfg.ext_mesh_health_max_diag_ratio) > 0.0:
		healthy = healthy & (diag_ratio <= float(cfg.ext_mesh_health_max_diag_ratio))
	if float(cfg.ext_mesh_health_max_edge_ratio) > 0.0:
		healthy = healthy & (edge_max <= edge_ref_safe * float(cfg.ext_mesh_health_max_edge_ratio))
	if float(cfg.ext_mesh_health_max_area_ratio) > 0.0:
		healthy = healthy & (area <= area_ref_safe * float(cfg.ext_mesh_health_max_area_ratio))
	if float(cfg.ext_mesh_health_min_area_ratio) > 0.0:
		healthy = healthy & (area >= area_ref_safe * float(cfg.ext_mesh_health_min_area_ratio))
	if float(cfg.ext_mesh_health_min_triangle_normal_dot) > -1.0:
		healthy = healthy & (tri_dot >= float(cfg.ext_mesh_health_min_triangle_normal_dot))
	if ext_normals is not None and float(cfg.ext_mesh_health_min_normal_dot) > 0.0:
		n_quad = torch.nn.functional.normalize(
			torch.linalg.cross(p10 - p00, p01 - p00, dim=-1),
			dim=-1,
			eps=1.0e-8,
		)
		n_avg = (
			ext_normals[:-1, :-1] +
			ext_normals[1:, :-1] +
			ext_normals[:-1, 1:] +
			ext_normals[1:, 1:]
		) * 0.25
		n_avg = torch.nn.functional.normalize(n_avg, dim=-1, eps=1.0e-8)
		normal_dot = (n_quad * n_avg).sum(dim=-1).abs()
		healthy = healthy & torch.isfinite(normal_dot) & (normal_dot >= float(cfg.ext_mesh_health_min_normal_dot))
	else:
		normal_dot = torch.zeros_like(area)
	stats = {
		"quads_input": float(int(base.sum().detach().cpu())),
		"quads_kept": float(int(healthy.sum().detach().cpu())),
		"quads_rejected": float(int((base & ~healthy).sum().detach().cpu())),
		"median_edge": float(edge_ref.detach().cpu()),
		"median_area": float(area_ref.detach().cpu()),
		"max_edge": float(edge_max[base].max().detach().cpu()),
		"max_area": float(area[base].max().detach().cpu()),
		"max_aspect": float(aspect[base].max().detach().cpu()),
		"max_diag_ratio": float(diag_ratio[base].max().detach().cpu()),
		"min_triangle_normal_dot": float(tri_dot[base].min().detach().cpu()),
		"min_normal_dot": float(normal_dot[base].min().detach().cpu()) if ext_normals is not None and float(cfg.ext_mesh_health_min_normal_dot) > 0.0 else 0.0,
	}
	initial_rejected = base & ~healthy
	reject_radius = max(0, int(cfg.ext_mesh_health_reject_radius))
	padding_rejected = torch.zeros_like(base)
	if reject_radius > 0 and bool(initial_rejected.any().detach().cpu()):
		padding_rejected = _dilate_quad_mask_euclidean(initial_rejected, reject_radius) & base & ~initial_rejected
		healthy = healthy & ~padding_rejected
		stats["quads_rejected_initial"] = float(int(initial_rejected.sum().detach().cpu()))
		stats["quads_rejected_padding"] = float(int(padding_rejected.sum().detach().cpu()))
		stats["quads_reject_radius"] = float(reject_radius)
		stats["quads_kept"] = float(int(healthy.sum().detach().cpu()))
		stats["quads_rejected"] = float(int((base & ~healthy).sum().detach().cpu()))
	else:
		stats["quads_rejected_initial"] = float(int(initial_rejected.sum().detach().cpu()))
		stats["quads_rejected_padding"] = 0.0
		stats["quads_reject_radius"] = float(reject_radius)
	return ext_quad_valid.bool() & healthy, stats


def _apply_external_quad_health_filter(
	fixture: MapFixture,
	cfg: SnapSurfConfig,
	*,
	label: str,
	external_surface_index: int | None = None,
	cache: dict[tuple[Any, ...], tuple[torch.Tensor, dict[str, float], list[int] | None]] | None = None,
	cache_stats: _CacheStats | None = None,
) -> MapFixture:
	mi = cfg.map_init
	if not bool(mi.ext_mesh_health_filter):
		return fixture
	metadata = dict(fixture.metadata)
	cache_key = (
		_external_health_cache_key(fixture, mi, external_surface_index=int(external_surface_index))
		if external_surface_index is not None else None
	)
	cached = cache.get(cache_key) if cache is not None and cache_key is not None else None
	cache_hit = cached is not None
	if cached is None:
		if cache is not None and cache_key is not None and cache_stats is not None:
			cache_stats["health_miss"] = int(cache_stats.get("health_miss", 0)) + 1
		filtered_quad, stats = _external_quad_health_filter(
			ext_xyz=fixture.ext_xyz,
			ext_valid=fixture.ext_valid,
			ext_quad_valid=fixture.ext_quad_valid,
			ext_normals=fixture.ext_normals,
			cfg=mi,
		)
		moved_seed: list[int] | None = None
		seed_raw = metadata.get("seed_ext_sample_hw")
		seed_valid = False
		if isinstance(seed_raw, (list, tuple)) and len(seed_raw) == 2:
			h, w = int(seed_raw[0]), int(seed_raw[1])
			if 0 <= h < int(filtered_quad.shape[0]) and 0 <= w < int(filtered_quad.shape[1]):
				seed_valid = bool(filtered_quad[h, w].detach().cpu())
		if not seed_valid and metadata.get("seed_xyz") is not None:
			seed_vals = metadata.get("seed_xyz")
			if isinstance(seed_vals, (list, tuple)) and len(seed_vals) == 3:
				seed = torch.tensor(seed_vals, device=fixture.ext_xyz.device, dtype=fixture.ext_xyz.dtype)
				ext_quad, _point, _dist = _closest_external_seed_surface(
					seed=seed,
					ext_xyz=fixture.ext_xyz,
					ext_valid=fixture.ext_valid,
					ext_quad_valid=filtered_quad,
				)
				if ext_quad is not None:
					moved_seed = [int(ext_quad[0]), int(ext_quad[1])]
		if cache is not None and cache_key is not None:
			cache[cache_key] = (filtered_quad.detach(), dict(stats), moved_seed)
	else:
		if cache_stats is not None:
			cache_stats["health_hit"] = int(cache_stats.get("health_hit", 0)) + 1
		filtered_quad, stats, moved_seed = cached
		stats = dict(stats)
		filtered_quad = filtered_quad.to(device=fixture.ext_quad_valid.device, dtype=torch.bool)
	metadata["ext_mesh_health"] = stats
	if moved_seed is not None:
		metadata["seed_ext_sample_hw"] = [int(moved_seed[0]), int(moved_seed[1])]
	before = int(stats["quads_input"])
	after = int(stats["quads_kept"])
	rejected = int(stats["quads_rejected"])
	if rejected > 0 and not cache_hit:
		print(
			"[snap_surf.map_global] external quad health filter "
			f"{label} kept={after}/{before} rejected={rejected} "
			f"median_edge={stats['median_edge']:.6g} max_edge={stats['max_edge']:.6g} "
			f"median_area={stats['median_area']:.6g} max_area={stats['max_area']:.6g} "
			f"max_aspect={stats.get('max_aspect', 0.0):.6g} max_diag={stats.get('max_diag_ratio', 0.0):.6g} "
			f"min_tri_dot={stats.get('min_triangle_normal_dot', 0.0):.6g} "
			f"pad_r={int(stats.get('quads_reject_radius', 0.0))} pad_rejected={int(stats.get('quads_rejected_padding', 0.0))}",
			flush=True,
		)
	if moved_seed is not None and rejected > 0:
		print(
			f"[snap_surf.map_global] external quad health filter {label} moved seed_ext_sample_hw={metadata['seed_ext_sample_hw']}",
			flush=True,
		)
	return replace(fixture, metadata=metadata, ext_quad_valid=filtered_quad)


def _max_supported_level(ext_shape: tuple[int, int], requested: int) -> int:
	strides = _map_init_dyadic_strides(
		int(ext_shape[0]),
		int(ext_shape[1]),
		requested_levels=max(1, int(requested) + 1),
		scale_factor=2,
	)
	return max(0, len(strides) - 1)


def _level_active_quad(active_full: torch.Tensor, level: int) -> torch.Tensor:
	level_i = int(level)
	if level_i <= 0:
		return active_full.bool()
	stride = 1 << level_i
	QH, QW = int(active_full.shape[0]), int(active_full.shape[1])
	H, W = QH // stride, QW // stride
	if H <= 0 or W <= 0:
		return torch.zeros(max(0, H), max(0, W), device=active_full.device, dtype=torch.bool)
	block = active_full[:H * stride, :W * stride].bool().reshape(H, stride, W, stride)
	return block.all(dim=3).all(dim=1)


def _seed_ext_hw(metadata: dict[str, Any], ext_shape: tuple[int, int], *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
	raw = metadata.get("seed_ext_sample_hw")
	if isinstance(raw, (list, tuple)) and len(raw) == 2:
		h, w = float(raw[0]), float(raw[1])
	else:
		h = (float(ext_shape[0]) - 1.0) * 0.5
		w = (float(ext_shape[1]) - 1.0) * 0.5
	return torch.tensor([h, w], device=device, dtype=dtype)


def _seed_model_uv(fixture: MapFixture, seed_ext_hw: torch.Tensor) -> torch.Tensor:
	raw_uv = fixture.metadata.get("seed_model_uv")
	if isinstance(raw_uv, (list, tuple)) and len(raw_uv) == 2:
		return torch.tensor([float(raw_uv[0]), float(raw_uv[1])], device=fixture.model_xyz.device, dtype=fixture.model_xyz.dtype)
	raw_quad = fixture.metadata.get("seed_model_quad")
	raw_seed = fixture.metadata.get("seed_xyz")
	if isinstance(raw_quad, (list, tuple)) and len(raw_quad) == 3 and isinstance(raw_seed, (list, tuple)) and len(raw_seed) == 3:
		try:
			model_quad = (int(raw_quad[0]), int(raw_quad[1]), int(raw_quad[2]))
			seed = torch.tensor([float(raw_seed[0]), float(raw_seed[1]), float(raw_seed[2])], device=fixture.model_xyz.device, dtype=fixture.model_xyz.dtype)
			_model_point, uv, _dist = _closest_point_uv_on_model_quad(point=seed, model_xyz=fixture.model_xyz, model_quad=model_quad)
			return uv.to(device=fixture.model_xyz.device, dtype=fixture.model_xyz.dtype)
		except (RuntimeError, ValueError, TypeError, IndexError):
			pass
	return seed_ext_hw.to(device=fixture.model_xyz.device, dtype=fixture.model_xyz.dtype)


def _affine_from_linear(seed_ext_hw: torch.Tensor, seed_model_uv: torch.Tensor, linear: torch.Tensor) -> torch.Tensor:
	offset = seed_model_uv - linear @ seed_ext_hw
	return torch.cat([linear, offset.view(2, 1)], dim=1)


def _affine_multistart_cfg(cfg: GlobalMapConfig, stage: GlobalMapStageConfig) -> dict[str, Any]:
	base_raw = cfg.base.get("affine_multistart", {})
	if not isinstance(base_raw, dict):
		base_raw = {}
	stage_raw = stage.args.get("affine_multistart", {})
	if not isinstance(stage_raw, dict):
		stage_raw = {}
	out = dict(base_raw)
	out.update(stage_raw)
	return out


def _affine_multistart_candidates(
	*,
	seed_ext_hw: torch.Tensor,
	seed_model_uv: torch.Tensor,
	rot_deg: list[float],
	scales: list[float],
) -> list[tuple[int | str, float | None, float | None, torch.Tensor]]:
	candidates: list[tuple[int | str, float | None, float | None, torch.Tensor]] = []
	device = seed_ext_hw.device
	dtype = seed_ext_hw.dtype
	idx = 0
	for scale in scales:
		for deg in rot_deg:
			rad = math.radians(float(deg))
			c = math.cos(rad)
			s = math.sin(rad)
			linear = torch.tensor(
				[[float(scale) * c, -float(scale) * s], [float(scale) * s, float(scale) * c]],
				device=device,
				dtype=dtype,
			)
			candidates.append((idx, float(deg), float(scale), _affine_from_linear(seed_ext_hw, seed_model_uv, linear)))
			idx += 1
	return candidates


def _affine_seed_grid_cfg(stage: GlobalMapStageConfig) -> dict[str, Any]:
	init_raw = stage.args.get("affine_seed_quad_init", stage.args.get("seed_quad_affine", {}))
	grid_raw: Any = {}
	if isinstance(init_raw, dict):
		grid_raw = init_raw.get("grid_search", init_raw.get("grid", init_raw.get("candidate_grid", {})))
	stage_raw = stage.args.get("affine_seed_grid", stage.args.get("affine_seed_quad_grid", {}))
	cfg: dict[str, Any] = {
		"enabled": False,
		"rot_deg": [-10.0, -5.0, 0.0, 5.0, 10.0],
		"scales": [0.75, 0.9, 1.0, 1.1, 1.25],
	}
	for raw in (grid_raw, stage_raw):
		if isinstance(raw, bool):
			cfg["enabled"] = bool(raw)
		elif isinstance(raw, dict):
			values = dict(raw)
			if "rotations_deg" in values and "rot_deg" not in values:
				values["rot_deg"] = values["rotations_deg"]
			cfg.update(values)
	return cfg


def _float_list(raw: Any, *, name: str) -> list[float]:
	if isinstance(raw, (int, float)):
		return [float(raw)]
	if not isinstance(raw, list):
		raise ValueError(f"{name} must be a number or list")
	return [float(v) for v in raw]


def _affine_seed_grid_candidates(
	*,
	base_affine: torch.Tensor,
	seed_ext_hw: torch.Tensor,
	rot_deg: list[float],
	scales: list[float],
) -> list[tuple[int | str, float | None, float | None, torch.Tensor]]:
	device = base_affine.device
	dtype = base_affine.dtype
	base = base_affine.to(device=device, dtype=dtype)
	seed_hw = seed_ext_hw.to(device=device, dtype=dtype)
	seed_uv = base[:, :2] @ seed_hw + base[:, 2]
	candidates: list[tuple[int | str, float | None, float | None, torch.Tensor]] = [("seedq", None, None, base.detach().clone())]
	idx = 0
	for scale in scales:
		for deg in rot_deg:
			if abs(float(deg)) <= 1.0e-12 and abs(float(scale) - 1.0) <= 1.0e-12:
				continue
			rad = math.radians(float(deg))
			c = math.cos(rad)
			s = math.sin(rad)
			perturb = torch.tensor(
				[[float(scale) * c, -float(scale) * s], [float(scale) * s, float(scale) * c]],
				device=device,
				dtype=dtype,
			)
			linear = perturb @ base[:, :2]
			candidates.append((idx, float(deg), float(scale), _affine_from_linear(seed_hw, seed_uv, linear)))
			idx += 1
	return candidates


def _valid_ext_quad(fixture: MapFixture, h: int, w: int) -> bool:
	Hq, Wq = int(fixture.ext_quad_valid.shape[0]), int(fixture.ext_quad_valid.shape[1])
	if h < 0 or w < 0 or h >= Hq or w >= Wq:
		return False
	if not bool(fixture.ext_quad_valid[h, w].detach().cpu()):
		return False
	verts_ok = (
		fixture.ext_valid[h, w] &
		fixture.ext_valid[h + 1, w] &
		fixture.ext_valid[h, w + 1] &
		fixture.ext_valid[h + 1, w + 1]
	)
	if not bool(verts_ok.detach().cpu()):
		return False
	pts = torch.stack([
		fixture.ext_xyz[h, w],
		fixture.ext_xyz[h + 1, w],
		fixture.ext_xyz[h, w + 1],
		fixture.ext_xyz[h + 1, w + 1],
	], dim=0)
	return bool(torch.isfinite(pts).all().detach().cpu())


def _seed_ext_quad(fixture: MapFixture, seed_hw: torch.Tensor) -> tuple[int, int] | None:
	raw_seed = fixture.metadata.get("seed_xyz")
	if isinstance(raw_seed, (list, tuple)) and len(raw_seed) == 3:
		seed = torch.tensor([float(raw_seed[0]), float(raw_seed[1]), float(raw_seed[2])], device=fixture.ext_xyz.device, dtype=fixture.ext_xyz.dtype)
		quad, _point, _dist = _closest_external_seed_surface(
			seed=seed,
			ext_xyz=fixture.ext_xyz,
			ext_valid=fixture.ext_valid.bool(),
			ext_quad_valid=fixture.ext_quad_valid.bool(),
		)
		if quad is not None:
			return quad
	Hq, Wq = int(fixture.ext_quad_valid.shape[0]), int(fixture.ext_quad_valid.shape[1])
	h = max(0, min(int(round(float(seed_hw[0].detach().cpu()))), Hq - 1))
	w = max(0, min(int(round(float(seed_hw[1].detach().cpu()))), Wq - 1))
	if _valid_ext_quad(fixture, h, w):
		return h, w
	return None


def _quad2_corners(grid: torch.Tensor, quad: tuple[int, int]) -> torch.Tensor:
	h, w = (int(v) for v in quad)
	return torch.stack([
		grid[h, w],
		grid[h + 1, w],
		grid[h, w + 1],
		grid[h + 1, w + 1],
	], dim=0)


def _quad3_corners(grid: torch.Tensor, quad: tuple[int, int, int]) -> torch.Tensor:
	d, h, w = (int(v) for v in quad)
	return torch.stack([
		grid[d, h, w],
		grid[d, h + 1, w],
		grid[d, h, w + 1],
		grid[d, h + 1, w + 1],
	], dim=0)


def _quad_edge_lengths(corners: torch.Tensor) -> tuple[float, float] | None:
	if not bool(torch.isfinite(corners).all().detach().cpu()):
		return None
	h_edges = torch.stack([corners[1] - corners[0], corners[3] - corners[2]], dim=0)
	w_edges = torch.stack([corners[2] - corners[0], corners[3] - corners[1]], dim=0)
	h = h_edges.norm(dim=-1).mean()
	w = w_edges.norm(dim=-1).mean()
	if not bool(torch.isfinite(h).detach().cpu()) or not bool(torch.isfinite(w).detach().cpu()):
		return None
	if float(h.detach().cpu()) <= 1.0e-8 or float(w.detach().cpu()) <= 1.0e-8:
		return None
	return float(h.detach().cpu()), float(w.detach().cpu())


def _quad_geom_normal(corners: torch.Tensor) -> torch.Tensor | None:
	if not bool(torch.isfinite(corners).all().detach().cpu()):
		return None
	h_axis = 0.5 * ((corners[1] - corners[0]) + (corners[3] - corners[2]))
	w_axis = 0.5 * ((corners[2] - corners[0]) + (corners[3] - corners[1]))
	n = torch.linalg.cross(h_axis, w_axis, dim=0)
	n_norm = n.norm()
	if not bool(torch.isfinite(n_norm).detach().cpu()) or float(n_norm.detach().cpu()) <= 1.0e-8:
		return None
	return n / n_norm


def _sign_from_dot(a: torch.Tensor | None, b: torch.Tensor | None, *, fallback: int = 1) -> int:
	if a is None or b is None:
		return 1 if int(fallback) >= 0 else -1
	dot = (a * b).sum()
	if not bool(torch.isfinite(dot).detach().cpu()):
		return 1 if int(fallback) >= 0 else -1
	return 1 if float(dot.detach().cpu()) >= 0.0 else -1


def _model_seed_patch_quads(
	*,
	model_valid: torch.Tensor,
	model_quad: tuple[int, int, int],
	radius_h: int,
	radius_w: int,
) -> torch.Tensor:
	d, h, w = (int(v) for v in model_quad)
	D, H, W = (int(v) for v in model_valid.shape)
	if d < 0 or d >= D or H < 2 or W < 2:
		return torch.empty(0, 3, device=model_valid.device, dtype=torch.long)
	h0 = max(0, h - max(0, int(radius_h)))
	h1 = min(H - 2, h + max(0, int(radius_h)))
	w0 = max(0, w - max(0, int(radius_w)))
	w1 = min(W - 2, w + max(0, int(radius_w)))
	if h0 > h1 or w0 > w1:
		return torch.empty(0, 3, device=model_valid.device, dtype=torch.long)
	quad_valid = (
		model_valid[d, h0:h1 + 2, w0:w1 + 2][:-1, :-1]
		& model_valid[d, h0:h1 + 2, w0:w1 + 2][1:, :-1]
		& model_valid[d, h0:h1 + 2, w0:w1 + 2][:-1, 1:]
		& model_valid[d, h0:h1 + 2, w0:w1 + 2][1:, 1:]
	)
	local = quad_valid.nonzero(as_tuple=False)
	if local.numel() == 0:
		return torch.empty(0, 3, device=model_valid.device, dtype=torch.long)
	out = torch.empty(local.shape[0], 3, device=model_valid.device, dtype=torch.long)
	out[:, 0] = int(d)
	out[:, 1] = local[:, 0] + int(h0)
	out[:, 2] = local[:, 1] + int(w0)
	return out


def _model_patch_triangles(
	*,
	model_xyz: torch.Tensor,
	patch_quads: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
	if patch_quads.numel() == 0:
		empty_tri = torch.empty(0, 3, 3, device=model_xyz.device, dtype=model_xyz.dtype)
		empty_uv = torch.empty(0, 3, 2, device=model_xyz.device, dtype=model_xyz.dtype)
		return empty_tri, empty_uv
	d = patch_quads[:, 0]
	h = patch_quads[:, 1]
	w = patch_quads[:, 2]
	p00 = model_xyz[d, h, w]
	p10 = model_xyz[d, h + 1, w]
	p01 = model_xyz[d, h, w + 1]
	p11 = model_xyz[d, h + 1, w + 1]
	tri = torch.cat([
		torch.stack([p00, p10, p11], dim=1),
		torch.stack([p00, p11, p01], dim=1),
	], dim=0)
	h_f = h.to(dtype=model_xyz.dtype)
	w_f = w.to(dtype=model_xyz.dtype)
	uv00 = torch.stack([h_f, w_f], dim=-1)
	uv10 = torch.stack([h_f + 1.0, w_f], dim=-1)
	uv01 = torch.stack([h_f, w_f + 1.0], dim=-1)
	uv11 = torch.stack([h_f + 1.0, w_f + 1.0], dim=-1)
	uv = torch.cat([
		torch.stack([uv00, uv10, uv11], dim=1),
		torch.stack([uv00, uv11, uv01], dim=1),
	], dim=0)
	finite = torch.isfinite(tri).all(dim=(1, 2))
	return tri[finite], uv[finite]


def _ray_triangle_intersections(
	origin: torch.Tensor,
	direction: torch.Tensor,
	tri: torch.Tensor,
	tri_uv: torch.Tensor,
	*,
	eps: float = 1.0e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
	if tri.numel() == 0:
		return (
			torch.empty(0, device=origin.device, dtype=origin.dtype),
			torch.empty(0, 2, device=origin.device, dtype=origin.dtype),
		)
	v0 = tri[:, 0]
	v1 = tri[:, 1]
	v2 = tri[:, 2]
	e1 = v1 - v0
	e2 = v2 - v0
	pvec = torch.linalg.cross(direction.view(1, 3).expand_as(e2), e2, dim=-1)
	det = (e1 * pvec).sum(dim=-1)
	det_ok = det.abs() > float(eps)
	inv_det = torch.where(det_ok, det.reciprocal(), torch.zeros_like(det))
	tvec = origin.view(1, 3) - v0
	u = (tvec * pvec).sum(dim=-1) * inv_det
	qvec = torch.linalg.cross(tvec, e1, dim=-1)
	v = (direction.view(1, 3) * qvec).sum(dim=-1) * inv_det
	t = (e2 * qvec).sum(dim=-1) * inv_det
	ok = (
		det_ok &
		torch.isfinite(t) &
		torch.isfinite(u) &
		torch.isfinite(v) &
		(t >= -float(eps)) &
		(u >= -float(eps)) &
		(v >= -float(eps)) &
		((u + v) <= 1.0 + float(eps))
	)
	if not bool(ok.any().detach().cpu()):
		return (
			torch.empty(0, device=origin.device, dtype=origin.dtype),
			torch.empty(0, 2, device=origin.device, dtype=origin.dtype),
		)
	w = 1.0 - u - v
	bary = torch.stack([w, u, v], dim=-1)
	uv = (bary.unsqueeze(-1) * tri_uv).sum(dim=1)
	return t[ok], uv[ok]


def _seed_quad_sample_grid(
	*,
	ext_quad: tuple[int, int],
	ext_xyz: torch.Tensor,
	ext_normals: torch.Tensor,
	samples: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	n = max(2, int(samples))
	h, w = (int(v) for v in ext_quad)
	lin = torch.linspace(0.0, 1.0, n, device=ext_xyz.device, dtype=ext_xyz.dtype)
	fh, fw = torch.meshgrid(lin, lin, indexing="ij")
	hw = torch.stack([fh + float(h), fw + float(w)], dim=-1).reshape(n * n, 2)
	p00 = ext_xyz[h, w]
	p10 = ext_xyz[h + 1, w]
	p01 = ext_xyz[h, w + 1]
	p11 = ext_xyz[h + 1, w + 1]
	n00 = ext_normals[h, w]
	n10 = ext_normals[h + 1, w]
	n01 = ext_normals[h, w + 1]
	n11 = ext_normals[h + 1, w + 1]
	fh_f = fh.reshape(n * n, 1)
	fw_f = fw.reshape(n * n, 1)
	points = (
		(1.0 - fh_f) * (1.0 - fw_f) * p00 +
		fh_f * (1.0 - fw_f) * p10 +
		(1.0 - fh_f) * fw_f * p01 +
		fh_f * fw_f * p11
	)
	normals = (
		(1.0 - fh_f) * (1.0 - fw_f) * n00 +
		fh_f * (1.0 - fw_f) * n10 +
		(1.0 - fh_f) * fw_f * n01 +
		fh_f * fw_f * n11
	)
	normals = torch.nn.functional.normalize(normals, dim=-1, eps=1.0e-8)
	return hw, points, normals


def _seed_quad_affine_cfg(raw: Any) -> dict[str, Any]:
	if isinstance(raw, dict):
		return dict(raw)
	return {}


def _seed_quad_affine_sample_terms(
	*,
	affine: torch.Tensor,
	ext_hw: torch.Tensor,
	target_uv: torch.Tensor,
	ext_points: torch.Tensor,
	ext_normals: torch.Tensor,
	fixture: MapFixture,
	stage_cfg: SnapSurfConfig,
	sign: int,
) -> dict[str, float]:
	x = torch.cat(
		[ext_hw.to(device=affine.device, dtype=affine.dtype), torch.ones((int(ext_hw.shape[0]), 1), device=affine.device, dtype=affine.dtype)],
		dim=1,
	)
	pred_uv = x @ affine.transpose(0, 1)
	target_uv = target_uv.to(device=pred_uv.device, dtype=pred_uv.dtype)
	uv_err = pred_uv - target_uv
	uv_finite = torch.isfinite(uv_err).all(dim=-1)
	if bool(uv_finite.any().detach().cpu()):
		uv_norm = uv_err[uv_finite].norm(dim=-1)
		uv_rmse = math.sqrt(float(uv_norm.square().mean().detach().cpu()))
		uv_max = float(uv_norm.max().detach().cpu())
	else:
		uv_rmse = float("nan")
		uv_max = float("nan")
	depth = int(fixture.metadata.get("model_depth", 0) or 0)
	coords3 = _map_init_coords3(pred_uv, depth=depth)
	safe_coords = torch.where(torch.isfinite(coords3), coords3, torch.zeros_like(coords3))
	p_model = _sample_surface_grid(fixture.model_xyz, safe_coords)
	n_model_raw = _sample_surface_grid(fixture.model_normals, safe_coords)
	sign_f = 1.0 if int(sign) >= 0 else -1.0
	n_ext = torch.nn.functional.normalize(ext_normals.to(device=pred_uv.device, dtype=pred_uv.dtype), dim=-1, eps=1.0e-8)
	n_model = torch.nn.functional.normalize(n_model_raw, dim=-1, eps=1.0e-8) * sign_f
	coord_ok = _quad_valid_at_coords(fixture.model_valid.bool(), safe_coords, tuple(int(v) for v in fixture.model_valid.shape))
	p_ext = ext_points.to(device=pred_uv.device, dtype=pred_uv.dtype)
	v = p_model - p_ext
	d = v.norm(dim=-1)
	u = v / d.clamp_min(1.0e-8).unsqueeze(-1)
	c_ext = (u * n_ext).sum(dim=-1).abs()
	c_model = (u * n_model).sum(dim=-1).abs()
	c_norm = (n_ext * n_model).sum(dim=-1)
	dist_values = _huber(d, delta=stage_cfg.huber_delta) * _map_init_distance_multiplier(c_ext, c_model, stage_cfg.map_init)
	vec_values = (1.0 - c_ext) + (1.0 - c_model)
	norm_values = 1.0 - c_norm
	z_lift = _map_init_z_lift_for_fixture(fixture, stage_cfg, sign=sign)
	turn_values = torch.zeros_like(norm_values)
	turn_valid = torch.ones_like(coord_ok, dtype=torch.bool)
	if z_lift is not None:
		quad_hw = torch.floor(ext_hw.to(device=affine.device, dtype=affine.dtype)).long()
		ext_theta = z_lift["ext_theta_lifted"].to(device=affine.device, dtype=affine.dtype)
		ext_valid = z_lift["ext_valid"].to(device=affine.device).bool()
		quad_hw[:, 0] = quad_hw[:, 0].clamp(0, max(0, int(ext_theta.shape[0]) - 1))
		quad_hw[:, 1] = quad_hw[:, 1].clamp(0, max(0, int(ext_theta.shape[1]) - 1))
		model_theta, model_theta_valid = _map_init_sample_scalar_quad_field(
			z_lift["model_theta_lifted"].to(device=affine.device, dtype=affine.dtype),
			z_lift["model_valid"].to(device=affine.device).bool(),
			safe_coords,
			tuple(int(v) for v in z_lift["model_valid"].shape),
		)
		ext_theta_sample = ext_theta[quad_hw[:, 0], quad_hw[:, 1]]
		turn_valid = (
			ext_valid[quad_hw[:, 0], quad_hw[:, 1]] &
			model_theta_valid &
			torch.isfinite(ext_theta_sample) &
			torch.isfinite(model_theta)
		)
		turn_values = _huber(ext_theta_sample - model_theta, delta=float(stage_cfg.map_init.z_lift_huber_delta))
	valid = (
		coord_ok &
		turn_valid &
		torch.isfinite(p_ext).all(dim=-1) &
		torch.isfinite(p_model).all(dim=-1) &
		torch.isfinite(n_ext).all(dim=-1) &
		torch.isfinite(n_model_raw).all(dim=-1) &
		torch.isfinite(n_model).all(dim=-1) &
		(n_ext.norm(dim=-1) > 1.0e-8) &
		(n_model.norm(dim=-1) > 1.0e-8) &
		torch.isfinite(dist_values) &
		torch.isfinite(vec_values) &
		torch.isfinite(norm_values) &
		torch.isfinite(turn_values)
	)
	if not bool(valid.any().detach().cpu()):
		return {
			"uv_rmse": uv_rmse,
			"uv_max": uv_max,
			"valid": 0.0,
			"loss": float("nan"),
			"dist": float("nan"),
			"dist_avg": float("nan"),
			"vec": float("nan"),
			"norm": float("nan"),
			"turn": float("nan"),
			"turn_smp": 0.0,
		}
	dist = float(dist_values[valid].mean().detach().cpu())
	vec = float(vec_values[valid].mean().detach().cpu())
	norm = float(norm_values[valid].mean().detach().cpu())
	turn = float(turn_values[valid].mean().detach().cpu())
	loss = (
		float(stage_cfg.map_init.w_dist) * dist +
		float(stage_cfg.map_init.w_vec_normal) * vec +
		float(stage_cfg.map_init.w_surface_normal) * norm +
		float(stage_cfg.map_init.w_z_lift) * turn
	)
	return {
		"uv_rmse": uv_rmse,
		"uv_max": uv_max,
		"valid": float(int(valid.sum().detach().cpu())),
		"loss": loss,
		"dist": dist,
		"dist_avg": float(d[valid].mean().detach().cpu()),
		"vec": vec,
		"norm": norm,
		"turn": turn,
		"turn_smp": float(int(valid.sum().detach().cpu())),
	}


def _seed_quad_affine_alignment_sign(
	*,
	affine: torch.Tensor,
	ext_hw: torch.Tensor,
	ext_normals: torch.Tensor,
	fixture: MapFixture,
	fallback: int = 1,
) -> int:
	x = torch.cat(
		[ext_hw.to(device=affine.device, dtype=affine.dtype), torch.ones((int(ext_hw.shape[0]), 1), device=affine.device, dtype=affine.dtype)],
		dim=1,
	)
	pred_uv = x @ affine.transpose(0, 1)
	depth = int(fixture.metadata.get("model_depth", 0) or 0)
	coords3 = _map_init_coords3(pred_uv, depth=depth)
	safe_coords = torch.where(torch.isfinite(coords3), coords3, torch.zeros_like(coords3))
	n_ext = torch.nn.functional.normalize(ext_normals.to(device=pred_uv.device, dtype=pred_uv.dtype), dim=-1, eps=1.0e-8)
	n_model_raw = _sample_surface_grid(fixture.model_normals, safe_coords)
	n_model = torch.nn.functional.normalize(n_model_raw, dim=-1, eps=1.0e-8)
	coord_ok = _quad_valid_at_coords(fixture.model_valid.bool(), safe_coords, tuple(int(v) for v in fixture.model_valid.shape))
	valid = (
		coord_ok &
		torch.isfinite(pred_uv).all(dim=-1) &
		torch.isfinite(n_ext).all(dim=-1) &
		torch.isfinite(n_model_raw).all(dim=-1) &
		torch.isfinite(n_model).all(dim=-1) &
		(n_ext.norm(dim=-1) > 1.0e-8) &
		(n_model.norm(dim=-1) > 1.0e-8)
	)
	if not bool(valid.any().detach().cpu()):
		return 1 if int(fallback) >= 0 else -1
	mean_dot = (n_ext[valid] * n_model[valid]).sum(dim=-1).mean()
	if not bool(torch.isfinite(mean_dot).detach().cpu()):
		return 1 if int(fallback) >= 0 else -1
	return 1 if float(mean_dot.detach().cpu()) >= 0.0 else -1


def _affine_uv_field(
	affine: torch.Tensor,
	ext_shape: tuple[int, int],
) -> torch.Tensor:
	H, W = int(ext_shape[0]), int(ext_shape[1])
	hh = torch.arange(H, device=affine.device, dtype=affine.dtype).view(H, 1).expand(H, W)
	ww = torch.arange(W, device=affine.device, dtype=affine.dtype).view(1, W).expand(H, W)
	return (
		hh.unsqueeze(-1) * affine[:, 0] +
		ww.unsqueeze(-1) * affine[:, 1] +
		affine[:, 2]
	)


def _seed_quad_expansion_radii(
	active: torch.Tensor,
	seed_ext_quad: tuple[int, int],
) -> list[int]:
	coords = active.bool().nonzero(as_tuple=False)
	if int(coords.shape[0]) == 0:
		return [0]
	seed_h, seed_w = (int(v) for v in seed_ext_quad)
	dh = (coords[:, 0] - seed_h).to(dtype=torch.float32)
	dw = (coords[:, 1] - seed_w).to(dtype=torch.float32)
	max_radius = int(math.ceil(float(torch.sqrt(dh.square() + dw.square()).max().detach().cpu())))
	radii = [0]
	r = 8
	while r < max_radius:
		radii.append(r)
		r *= 2
	if max_radius > 0:
		radii.append(max(r, max_radius))
	return radii


def _seed_quad_expansion_reopt_radii(
	active: torch.Tensor,
	seed_ext_quad: tuple[int, int],
) -> list[int]:
	radii = _seed_quad_expansion_radii(active, seed_ext_quad)
	return radii[1:] if len(radii) > 1 else radii


def _seed_quad_expansion_active_mask(
	active: torch.Tensor,
	seed_ext_quad: tuple[int, int],
	radius: int,
) -> torch.Tensor:
	QH, QW = int(active.shape[0]), int(active.shape[1])
	seed_h, seed_w = (int(v) for v in seed_ext_quad)
	hh = torch.arange(QH, device=active.device).view(QH, 1).expand(QH, QW)
	ww = torch.arange(QW, device=active.device).view(1, QW).expand(QH, QW)
	dist2 = (hh - seed_h).square() + (ww - seed_w).square()
	near = dist2 <= int(radius) * int(radius)
	return active.bool() & near

def _seed_quad_expansion_crop_slices(
	active: torch.Tensor,
	seed_ext_quad: tuple[int, int],
	radius: int,
	cfg: SnapSurfMapInitConfig,
) -> tuple[slice, slice, slice, slice]:
	QH, QW = int(active.shape[0]), int(active.shape[1])
	seed_h, seed_w = (int(v) for v in seed_ext_quad)
	pad = max(0, int(cfg.dense_reg_radius)) if bool(cfg.dense_opt) else 0
	r = max(0, int(radius)) + pad
	h0 = max(0, seed_h - r)
	h1 = min(QH - 1, seed_h + r)
	w0 = max(0, seed_w - r)
	w1 = min(QW - 1, seed_w + r)
	return (
		slice(h0, h1 + 2),
		slice(w0, w1 + 2),
		slice(h0, h1 + 1),
		slice(w0, w1 + 1),
	)


def _objective_for_active_uv(
	*,
	uv: torch.Tensor,
	active_quad: torch.Tensor,
	fixture: MapFixture,
	cfg: SnapSurfConfig,
	need_stats: bool = True,
	active_quad_crop: tuple[slice, slice, slice, slice] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
	sign_i = int(fixture.metadata.get("sign", 1) or 1)
	z_lift = _map_init_z_lift_for_fixture(fixture, cfg, sign=sign_i)
	return _map_init_objective(
		uv_full=uv,
		active_quad=active_quad,
		ext_pos=fixture.ext_xyz,
		ext_normals=fixture.ext_normals,
		ext_valid=fixture.ext_valid,
		ext_quad_valid=fixture.ext_quad_valid,
		ext_coords=None,
		model_xyz=fixture.model_xyz,
		model_valid=fixture.model_valid,
		model_normals=fixture.model_normals,
		model_depth=int(fixture.metadata.get("model_depth", 0) or 0),
		sign=sign_i,
		cfg=cfg,
		allow_partial_model_samples=True,
		need_stats=bool(need_stats),
		crop_active_quad=True,
		active_quad_crop=active_quad_crop,
		ext_z_lift_theta=None if z_lift is None else z_lift["ext_theta_lifted"],
		ext_z_lift_valid=None if z_lift is None else z_lift["ext_valid"],
		model_z_lift_theta=None if z_lift is None else z_lift["model_theta_lifted"],
		model_z_lift_valid=None if z_lift is None else z_lift["model_valid"],
	)


def _affine_seed_quad_expansion_row_from_terms(
	*,
	radius: int,
	active: torch.Tensor,
	loss: torch.Tensor,
	terms: dict[str, torch.Tensor],
	station: torch.Tensor | None = None,
) -> dict[str, float]:
	row = {
		"radius": float(radius),
		"quads": float(int(active.sum().detach().cpu())),
		"loss": float(loss.detach().cpu()),
		"dist": _global_term_value(terms, "dist"),
		"dist_avg": _global_term_value(terms, "dist_avg"),
		"vec": _global_term_value(terms, "vec"),
		"norm": _global_term_value(terms, "norm"),
		"turn": _global_term_value(terms, "turn"),
		"smooth": _global_term_value(terms, "smooth"),
		"jac": _global_term_value(terms, "jac"),
		"samples": _global_term_value(terms, "samples"),
		"sample_bad": _global_term_value(terms, "sample_bad"),
		"quad_success": _global_term_value(terms, "quad_success"),
	}
	if station is not None:
		row["station"] = float(station.detach().cpu())
	return row


def _affine_seed_quad_expansion_rows(
	*,
	affine: torch.Tensor,
	seed_ext_quad: tuple[int, int],
	fixture: MapFixture,
	stage_cfg: SnapSurfConfig,
	sign: int | None = None,
	timing_prefix: str | None = None,
) -> list[dict[str, float]]:
	t_total = time.perf_counter()
	active_full = _full_active_quad(fixture)
	uv = _affine_uv_field(affine, tuple(int(v) for v in fixture.ext_xyz.shape[:2]))
	sign_i = int(fixture.metadata.get("sign", 1) or 1) if sign is None else int(sign)
	old_sign = fixture.metadata.get("sign")
	fixture.metadata["sign"] = sign_i
	rows: list[dict[str, float]] = []
	radii = _seed_quad_expansion_radii(active_full, seed_ext_quad)
	if timing_prefix is not None:
		print(
			f"{timing_prefix} start radii={len(radii)} full_quads={int(active_full.sum().detach().cpu())}",
			flush=True,
		)
	try:
		for radius in radii:
			_sync_timing_device(active_full)
			t_radius = time.perf_counter()
			active = _seed_quad_expansion_active_mask(active_full, seed_ext_quad, radius)
			crop = _seed_quad_expansion_crop_slices(active_full, seed_ext_quad, radius, stage_cfg.map_init)
			_sync_timing_device(active)
			mask_ms = 1000.0 * (time.perf_counter() - t_radius)
			t_obj = time.perf_counter()
			loss, terms = _objective_for_active_uv(
				uv=uv,
				active_quad=active,
				fixture=fixture,
				cfg=stage_cfg,
				active_quad_crop=crop,
			)
			_sync_timing_device(loss)
			obj_ms = 1000.0 * (time.perf_counter() - t_obj)
			rows.append(_affine_seed_quad_expansion_row_from_terms(
				radius=radius,
				active=active,
				loss=loss,
				terms=terms,
			))
			if timing_prefix is not None:
				total_ms = 1000.0 * (time.perf_counter() - t_radius)
				vert_h, vert_w, quad_h, quad_w = crop
				print(
					f"{timing_prefix} rad={int(radius)} quads={int(active.sum().detach().cpu())} "
					f"quad_crop=({quad_h.start}:{quad_h.stop},{quad_w.start}:{quad_w.stop}) "
					f"vert_crop=({vert_h.start}:{vert_h.stop},{vert_w.start}:{vert_w.stop}) "
					f"mask={mask_ms:.3f}ms objective={obj_ms:.3f}ms total={total_ms:.3f}ms",
					flush=True,
				)
	finally:
		if old_sign is None:
			fixture.metadata.pop("sign", None)
		else:
			fixture.metadata["sign"] = old_sign
	if timing_prefix is not None:
		print(
			f"{timing_prefix} done total={1000.0 * (time.perf_counter() - t_total):.3f}ms",
			flush=True,
		)
	return rows


_SEED_EXPANSION_COLUMNS = (
	ProgressColumn("radius", "rad", "Euclidean radius in external quad-grid coordinates; 0 is just the seed quad", min_width=5),
	ProgressColumn("quads", "quads", "active external quads in radius", min_width=7),
	ProgressColumn("loss", "loss", "map objective loss with optimizer terms", min_width=7),
	ProgressColumn("dist", "dist", "map distance loss", min_width=7),
	ProgressColumn("dist_avg", "avgd", "mean connection distance", min_width=7),
	ProgressColumn("vec", "vec", "vector-normal loss", min_width=7),
	ProgressColumn("norm", "nrm", "surface-normal loss", min_width=7),
	ProgressColumn("turn", "turn", "lifted z-heading loss", min_width=7),
	ProgressColumn("smooth", "smo", "smoothness loss", min_width=7),
	ProgressColumn("jac", "jac", "jacobian loss", min_width=7),
	ProgressColumn("samples", "smp", "objective samples", min_width=7),
	ProgressColumn("sample_bad", "bad", "samples rejected by objective limits", min_width=7),
	ProgressColumn("quad_success", "okq", "quads passing objective checks", min_width=7),
)


def _print_affine_seed_quad_expansion_diagnostic(
	*,
	affine: torch.Tensor,
	seed_ext_quad: tuple[int, int],
	fixture: MapFixture,
	stage_cfg: SnapSurfConfig,
	sign: int | None = None,
) -> None:
	rows = _affine_seed_quad_expansion_rows(
		affine=affine,
		seed_ext_quad=seed_ext_quad,
		fixture=fixture,
		stage_cfg=stage_cfg,
		sign=sign,
		timing_prefix="[snap_surf.map_global] affine seed quad expansion loss timing",
	)
	widths = progress_widths(
		_SEED_EXPANSION_COLUMNS,
		{
			"radius": "1000000",
			"quads": "100000000",
			"loss": "-1.0e+99",
			"dist": "-1.0e+99",
			"dist_avg": "-1.0e+99",
			"vec": "-1.0e+99",
			"norm": "-1.0e+99",
			"turn": "-1.0e+99",
			"smooth": "-1.0e+99",
			"jac": "-1.0e+99",
			"samples": "100000000",
			"sample_bad": "100000000",
			"quad_success": "100000000",
		},
	)
	_print_progress_legend_once(
		prefix="[snap_surf.map_global] affine seed quad expansion loss",
		items=[(col.label, col.description) for col in _SEED_EXPANSION_COLUMNS],
	)
	print(progress_header(_SEED_EXPANSION_COLUMNS, widths), flush=True)
	for row in rows:
		values = {
			"radius": str(int(row["radius"])),
			"quads": str(int(row["quads"])),
			"loss": format_progress_value(float(row["loss"])),
			"dist": format_progress_value(float(row["dist"])),
			"dist_avg": format_progress_value(float(row["dist_avg"])),
			"vec": format_progress_value(float(row["vec"])),
			"norm": format_progress_value(float(row["norm"])),
			"turn": format_progress_value(float(row["turn"])),
			"smooth": format_progress_value(float(row["smooth"])),
			"jac": format_progress_value(float(row["jac"])),
			"samples": str(int(row["samples"])),
			"sample_bad": str(int(row["sample_bad"])),
			"quad_success": str(int(row["quad_success"])),
		}
		print(progress_row(_SEED_EXPANSION_COLUMNS, widths, values), flush=True)


def _run_affine_seed_quad_expansion_reopt(
	*,
	affine: AffineMapModel,
	fixture: MapFixture,
	stage_cfg: SnapSurfConfig,
	stage: GlobalMapStageConfig | None = None,
	seed_ext_quad: tuple[int, int],
	seed_hw: torch.Tensor,
	station_target: torch.Tensor,
	w_station: float,
	steps: int,
	lr: float,
	status_interval: int,
	lr_warmup_steps: int,
	stage_idx: int,
	progress_widths_run: dict[str, int] | None = None,
	progress_row_idx: int = 0,
	debug_obj_root: Path | None = None,
	cancel_fn=None,
) -> tuple[list[dict[str, float]], int]:
	steps_i = max(0, int(steps))
	if steps_i <= 0:
		return [], 0
	lr_warmup_steps_i = max(0, int(lr_warmup_steps))
	status_interval_i = max(0, int(status_interval))
	active_full = _full_active_quad(fixture)
	radii = _seed_quad_expansion_reopt_radii(active_full, seed_ext_quad)
	rows: list[dict[str, float]] = []
	progress_rows = 0
	mi = stage_cfg.map_init
	params = "map_surf_affine"
	stage_name = "-" if stage is None or not stage.name else stage.name
	stage_lr = float("nan") if stage is None else float(stage.lr)
	w_fac = "-" if stage is None else _stage_w_fac_label(stage.w_fac)
	print(
		"[snap_surf.map_global] affine seed quad expansion reopt opts "
		f"stg={int(stage_idx)} name={stage_name} params={params} optimizer=Adam "
		f"stage_lr={stage_lr:.6g} lr={float(lr):.6g} steps_per_radius={steps_i} "
		f"lr_warmup_steps={lr_warmup_steps_i} status_interval={status_interval_i} radii={len(radii)} "
		f"start_radius={int(radii[0]) if radii else 0} max_radius={int(radii[-1]) if radii else 0} "
		f"w_fac={w_fac} "
		"weights="
		f"dist:{float(mi.w_dist):.6g},vec:{float(mi.w_vec_normal):.6g},norm:{float(mi.w_surface_normal):.6g},"
		f"turn:{float(mi.w_z_lift):.6g},"
		f"smooth:{float(mi.w_smooth):.6g},bend:{float(mi.w_bend):.6g},jac:{float(mi.w_jac):.6g},"
		f"metric:{float(mi.w_metric_smooth):.6g},area:{float(mi.w_area_smooth):.6g},"
		f"prior:{float(mi.w_dense_prior):.6g},station:{float(w_station):.6g}",
		flush=True,
	)
	if debug_obj_root is not None:
		print(
			f"[snap_surf.map_global] affine seed quad expansion debug objs dir={debug_obj_root}",
			flush=True,
		)
	def opt_lr(opt: torch.optim.Optimizer | None) -> float:
		return _optimizer_lr_for_display(opt, float(lr))
	def write_debug_map(radius: int, phase: str, step_count: int, active: torch.Tensor, row: dict[str, float]) -> None:
		if debug_obj_root is None:
			return
		out = debug_obj_root / f"rad_{int(radius):06d}_{_debug_obj_safe_label(phase)}"
		t_write = time.perf_counter()
		print(
			f"[snap_surf.map_global] affine seed quad expansion debug obj start "
			f"radius={int(radius)} phase={phase} dir={out}",
			flush=True,
		)
		_write_map_objs(
			out,
			uv=affine().detach(),
			fixture=fixture,
			meta={
				"phase": str(phase),
				"radius": int(radius),
				"steps": int(step_count),
				"active_quads": int(active.sum().detach().cpu()),
				"lr": float(row.get("lr", lr)),
				"loss": float(row["loss"]),
				"dist": float(row["dist"]),
				"dist_avg": float(row["dist_avg"]),
				"vec": float(row["vec"]),
				"norm": float(row["norm"]),
				"smooth": float(row["smooth"]),
				"jac": float(row["jac"]),
				"samples": int(row["samples"]),
				"sample_bad": int(row["sample_bad"]),
				"quad_success": int(row["quad_success"]),
			},
			active_quad=active,
		)
		print(
			f"[snap_surf.map_global] affine seed quad expansion debug obj "
			f"radius={int(radius)} phase={phase} write={1000.0 * (time.perf_counter() - t_write):.3f}ms",
			flush=True,
		)
	def eval_row(
		radius: int,
		active: torch.Tensor,
		crop: tuple[slice, slice, slice, slice],
		step_count: int,
		init_loss: float,
	) -> dict[str, float]:
		with torch.no_grad():
			uv = affine()
			loss, terms = _objective_for_active_uv(
				uv=uv,
				active_quad=active,
				fixture=fixture,
				cfg=stage_cfg,
				active_quad_crop=crop,
			)
			station_raw = loss.new_zeros(())
			if float(w_station) > 0.0:
				station_raw = _station_loss(uv, seed_hw, station_target)
				loss = loss + float(w_station) * station_raw
			row = _affine_seed_quad_expansion_row_from_terms(
				radius=radius,
				active=active,
				loss=loss,
				terms=terms,
				station=station_raw,
			)
		row["iters"] = float(step_count)
		row["init_loss"] = float(init_loss)
		row["loss_gain"] = float(init_loss) - float(row["loss"])
		row["lr"] = float(lr)
		return row
	def eval_progress(
		radius: int,
		active: torch.Tensor,
		crop: tuple[slice, slice, slice, slice],
		step_count: int,
		init_loss: float,
	) -> tuple[float, dict[str, torch.Tensor], dict[str, float]]:
		with torch.no_grad():
			uv = affine()
			loss, terms = _objective_for_active_uv(
				uv=uv,
				active_quad=active,
				fixture=fixture,
				cfg=stage_cfg,
				active_quad_crop=crop,
			)
			station_raw = loss.new_zeros(())
			if float(w_station) > 0.0:
				station_raw = _station_loss(uv, seed_hw, station_target)
				loss = loss + float(w_station) * station_raw
			progress_terms = dict(terms)
			progress_terms["station"] = station_raw.detach()
			err = _fixture_mapping_error(uv.detach(), fixture)
		return float(loss.detach().cpu()), progress_terms, err
	for radius in radii:
		if cancel_fn is not None:
			cancel_fn()
		active = _seed_quad_expansion_active_mask(active_full, seed_ext_quad, radius)
		crop = _seed_quad_expansion_crop_slices(active_full, seed_ext_quad, radius, stage_cfg.map_init)
		init_row = eval_row(radius, active, crop, 0, 0.0)
		init_loss = float(init_row["loss"])
		init_row["init_loss"] = init_loss
		init_row["loss_gain"] = 0.0
		rows.append(init_row)
		write_debug_map(radius, "init", 0, active, init_row)
		if progress_widths_run is not None:
			report_loss, report_terms, report_err = eval_progress(radius, active, crop, 0, init_loss)
			_print_global_progress(
				row_idx=int(progress_row_idx) + progress_rows,
				widths=progress_widths_run,
				stage_idx=stage_idx,
				iter_label=f"grow-r{int(radius)}:0/{steps_i}",
				lr=float(lr),
				level=0,
				loss=report_loss,
				terms=report_terms,
				it_s=None,
				err=report_err,
			)
			progress_rows += 1
		opt = torch.optim.Adam([affine.affine], lr=float(lr))
		_capture_optimizer_target_lrs(opt)
		lr_autoscale = _make_lr_autoscale_state(stage.args if stage is not None else None)
		for step in range(steps_i):
			if cancel_fn is not None:
				cancel_fn()
			opt.zero_grad(set_to_none=True)
			uv = affine()
			loss, _terms = _objective_for_active_uv(
				uv=uv,
				active_quad=active,
				fixture=fixture,
				cfg=stage_cfg,
				need_stats=False,
				active_quad_crop=crop,
			)
			if float(w_station) > 0.0:
				loss = loss + float(w_station) * _station_loss(uv, seed_hw, station_target)
			loss.backward()
			step1 = step + 1
			_apply_optimizer_lr_schedule(
				opt,
				step1=step1,
				warmup_steps=lr_warmup_steps_i,
				autoscale=lr_autoscale,
				loss=loss,
			)
			opt.step()
			status_due = (
				step == 0 or
				step1 == steps_i or
				(status_interval_i > 0 and (step1 % status_interval_i) == 0)
			)
			if status_due:
				step_lr = opt_lr(opt)
				row = eval_row(radius, active, crop, step1, init_loss)
				row["lr"] = float(step_lr)
				rows.append(row)
				if progress_widths_run is not None:
					report_loss, report_terms, report_err = eval_progress(radius, active, crop, step1, init_loss)
					_print_global_progress(
						row_idx=int(progress_row_idx) + progress_rows,
						widths=progress_widths_run,
						stage_idx=stage_idx,
						iter_label=f"grow-r{int(radius)}:{step1}/{steps_i}",
						lr=float(step_lr),
						level=0,
						loss=report_loss,
						terms=report_terms,
						it_s=None,
						err=report_err,
					)
					progress_rows += 1
		final_row = eval_row(radius, active, crop, steps_i, init_loss)
		final_row["lr"] = opt_lr(opt)
		write_debug_map(radius, "final", steps_i, active, final_row)
	return rows, progress_rows


def _write_affine_seed_initial_debug_radius(
	*,
	debug_obj_root: Path | None,
	affine_tensor: torch.Tensor,
	fixture: MapFixture,
	stage_cfg: SnapSurfConfig,
	seed_ext_quad: tuple[int, int],
	radius: int = 128,
) -> None:
	if debug_obj_root is None:
		return
	radius_i = max(0, int(radius))
	t_total = time.perf_counter()
	print(
		f"[snap_surf.map_global] affine seed quad initial debug objs start radius={radius_i} dir={debug_obj_root}",
		flush=True,
	)
	_sync_timing_device(affine_tensor)
	t_uv = time.perf_counter()
	uv = _affine_uv_field(affine_tensor, tuple(int(v) for v in fixture.ext_xyz.shape[:2]))
	_sync_timing_device(uv)
	uv_ms = 1000.0 * (time.perf_counter() - t_uv)
	t_mask = time.perf_counter()
	active_full = _full_active_quad(fixture)
	active = _seed_quad_expansion_active_mask(active_full, seed_ext_quad, radius_i)
	crop = _seed_quad_expansion_crop_slices(active_full, seed_ext_quad, radius_i, stage_cfg.map_init)
	_sync_timing_device(active)
	mask_ms = 1000.0 * (time.perf_counter() - t_mask)
	t_obj = time.perf_counter()
	loss, terms = _objective_for_active_uv(
		uv=uv,
		active_quad=active,
		fixture=fixture,
		cfg=stage_cfg,
		active_quad_crop=crop,
	)
	_sync_timing_device(loss)
	obj_ms = 1000.0 * (time.perf_counter() - t_obj)
	row = _affine_seed_quad_expansion_row_from_terms(
		radius=radius_i,
		active=active,
		loss=loss,
		terms=terms,
	)
	out = debug_obj_root / f"rad_{radius_i:06d}_initial_filtered"
	t_write = time.perf_counter()
	_write_map_objs(
		out,
		uv=uv,
		fixture=fixture,
		meta={
			"phase": "initial_filtered",
			"radius": radius_i,
			"active_quads": int(active.sum().detach().cpu()),
			"loss": float(row["loss"]),
			"dist": float(row["dist"]),
			"dist_avg": float(row["dist_avg"]),
			"vec": float(row["vec"]),
			"norm": float(row["norm"]),
			"smooth": float(row["smooth"]),
			"jac": float(row["jac"]),
			"samples": int(row["samples"]),
			"sample_bad": int(row["sample_bad"]),
			"quad_success": int(row["quad_success"]),
			"ext_mesh_health": fixture.metadata.get("ext_mesh_health"),
		},
		active_quad=active,
	)
	write_ms = 1000.0 * (time.perf_counter() - t_write)
	total_ms = 1000.0 * (time.perf_counter() - t_total)
	print(
		f"[snap_surf.map_global] affine seed quad initial debug objs radius={radius_i} dir={out} "
		f"uv={uv_ms:.3f}ms mask={mask_ms:.3f}ms objective={obj_ms:.3f}ms "
		f"write={write_ms:.3f}ms total={total_ms:.3f}ms",
		flush=True,
	)


def _seed_quad_affine_init_result(
	*,
	fixture: MapFixture,
	stage_cfg: SnapSurfConfig,
	seed_hw: torch.Tensor,
	seed_model_uv: torch.Tensor,
	raw: Any = None,
) -> SeedQuadAffineInitResult | None:
	cfg = _seed_quad_affine_cfg(raw)
	samples = max(2, int(cfg.get("samples", cfg.get("sample_grid", 16))))
	max_distance = float(cfg.get("max_ray_distance", cfg.get("max_intersection_distance", 100.0)))
	safety_margin = max(0, int(cfg.get("model_patch_margin", cfg.get("safety_margin", 2))))
	raw_quad = fixture.metadata.get("seed_model_quad")
	raw_seed = fixture.metadata.get("seed_xyz")
	if not (isinstance(raw_quad, (list, tuple)) and len(raw_quad) == 3 and isinstance(raw_seed, (list, tuple)) and len(raw_seed) == 3):
		return None
	model_quad = (int(raw_quad[0]), int(raw_quad[1]), int(raw_quad[2]))
	seed = torch.tensor([float(raw_seed[0]), float(raw_seed[1]), float(raw_seed[2])], device=fixture.ext_xyz.device, dtype=fixture.ext_xyz.dtype)
	seed_ext_quad, seed_ext_point, _seed_ext_dist = _closest_external_seed_surface(
		seed=seed,
		ext_xyz=fixture.ext_xyz,
		ext_valid=fixture.ext_valid.bool(),
		ext_quad_valid=fixture.ext_quad_valid.bool(),
	)
	if seed_ext_quad is None or seed_ext_point is None:
		return None
	if _seed_ext_quad(fixture, seed_hw) is None:
		return None
	ext_corners = _quad2_corners(fixture.ext_xyz, seed_ext_quad)
	model_corners = _quad3_corners(fixture.model_xyz, model_quad)
	ext_steps = _quad_edge_lengths(ext_corners)
	model_steps = _quad_edge_lengths(model_corners)
	ext_normal = _quad_geom_normal(ext_corners)
	model_normal = _quad_geom_normal(model_corners)
	if ext_steps is None or model_steps is None or ext_normal is None or model_normal is None:
		return None
	ext_h, ext_w = ext_steps
	model_h, model_w = model_steps
	radius_h = max(1, int(math.ceil(ext_h / max(model_h, 1.0e-8)))) + safety_margin
	radius_w = max(1, int(math.ceil(ext_w / max(model_w, 1.0e-8)))) + safety_margin
	patch_quads = _model_seed_patch_quads(
		model_valid=fixture.model_valid.bool(),
		model_quad=model_quad,
		radius_h=radius_h,
		radius_w=radius_w,
	)
	tri, tri_uv = _model_patch_triangles(model_xyz=fixture.model_xyz, patch_quads=patch_quads)
	if tri.numel() == 0:
		return None
	ext_hw, ext_points, ext_sample_normals = _seed_quad_sample_grid(
		ext_quad=seed_ext_quad,
		ext_xyz=fixture.ext_xyz,
		ext_normals=fixture.ext_normals,
		samples=samples,
	)
	raw_dirs = ext_sample_normals
	kept_hw: list[torch.Tensor] = []
	kept_uv: list[torch.Tensor] = []
	kept_points: list[torch.Tensor] = []
	kept_normals: list[torch.Tensor] = []
	hit_count = 0
	rejected_far = 0
	for i in range(int(ext_points.shape[0])):
		p = ext_points[i]
		direction = raw_dirs[i]
		if (
			not bool(torch.isfinite(p).all().detach().cpu()) or
			not bool(torch.isfinite(direction).all().detach().cpu()) or
			float(direction.norm().detach().cpu()) <= 1.0e-8
		):
			continue
		direction = torch.nn.functional.normalize(direction, dim=0, eps=1.0e-8)
		t_fwd, uv_fwd = _ray_triangle_intersections(p, direction, tri, tri_uv)
		t_back, uv_back = _ray_triangle_intersections(p, -direction, tri, tri_uv)
		t = torch.cat([t_fwd, t_back], dim=0)
		uv = torch.cat([uv_fwd, uv_back], dim=0)
		if t.numel() == 0:
			continue
		dist = t.abs()
		best = int(torch.argmin(dist).detach().cpu())
		best_dist = float(dist[best].detach().cpu())
		hit_count += 1
		if best_dist > max_distance:
			rejected_far += 1
			continue
		best_uv = uv[best]
		if not bool(torch.isfinite(best_uv).all().detach().cpu()):
			continue
		kept_hw.append(ext_hw[i])
		kept_uv.append(best_uv)
		kept_points.append(p)
		kept_normals.append(ext_sample_normals[i])
	if len(kept_hw) < 3:
		print(
			f"[snap_surf.map_global] affine seed quad ray init unavailable "
			f"ext_quad={seed_ext_quad} model_quad={model_quad} samples={samples * samples} "
			f"hits={hit_count} kept={len(kept_hw)} rejected_far={rejected_far} "
			f"model_quads={int(patch_quads.shape[0])}",
			flush=True,
		)
		return None
	x_hw = torch.stack(kept_hw, dim=0)
	y_uv = torch.stack(kept_uv, dim=0)
	y_points = torch.stack(kept_points, dim=0)
	y_normals = torch.stack(kept_normals, dim=0)
	x = torch.cat([x_hw, torch.ones((len(kept_hw), 1), device=x_hw.device, dtype=x_hw.dtype)], dim=1)
	try:
		sol = torch.linalg.lstsq(x, y_uv).solution
	except RuntimeError:
		return None
	affine = sol.transpose(0, 1).contiguous()
	if not bool(torch.isfinite(affine).all().detach().cpu()):
		return None
	sign = _seed_quad_affine_alignment_sign(
		affine=affine,
		ext_hw=x_hw,
		ext_normals=y_normals,
		fixture=fixture,
	)
	sample_terms = _seed_quad_affine_sample_terms(
		affine=affine,
		ext_hw=x_hw,
		target_uv=y_uv,
		ext_points=y_points,
		ext_normals=y_normals,
		fixture=fixture,
		stage_cfg=stage_cfg,
		sign=sign,
	)
	print(
		f"[snap_surf.map_global] affine seed quad ray init "
		f"ext_quad={seed_ext_quad} model_quad={model_quad} "
		f"sign={sign} sign_semantics=model_normal_alignment "
		f"ext_step=({ext_h:.6g},{ext_w:.6g}) model_step=({model_h:.6g},{model_w:.6g}) "
		f"model_radius=({radius_h},{radius_w}) model_quads={int(patch_quads.shape[0])} "
		f"samples={samples * samples} hits={hit_count} kept={len(kept_hw)} rejected_far={rejected_far}",
		flush=True,
	)
	print(
		f"[snap_surf.map_global] affine seed quad fitted matrix "
		f"a00={float(affine[0, 0].detach().cpu()):.9g} "
		f"a01={float(affine[0, 1].detach().cpu()):.9g} "
		f"a02={float(affine[0, 2].detach().cpu()):.9g} "
		f"a10={float(affine[1, 0].detach().cpu()):.9g} "
		f"a11={float(affine[1, 1].detach().cpu()):.9g} "
		f"a12={float(affine[1, 2].detach().cpu()):.9g}",
		flush=True,
	)
	print(
		f"[snap_surf.map_global] affine seed quad 16x16 loss "
		f"valid={int(sample_terms['valid'])}/{len(kept_hw)} "
		f"loss={sample_terms['loss']:.6g} "
		f"dist={sample_terms['dist']:.6g} dist_avg={sample_terms['dist_avg']:.6g} "
		f"vec={sample_terms['vec']:.6g} norm={sample_terms['norm']:.6g} "
		f"turn={sample_terms['turn']:.6g} turn_smp={int(sample_terms['turn_smp'])} "
		f"uv_rmse={sample_terms['uv_rmse']:.6g} uv_max={sample_terms['uv_max']:.6g}",
		flush=True,
	)
	if bool(cfg.get("expansion_loss_diag", False)):
		_print_affine_seed_quad_expansion_diagnostic(
			affine=affine,
			seed_ext_quad=seed_ext_quad,
			fixture=fixture,
			stage_cfg=stage_cfg,
			sign=sign,
		)
	return SeedQuadAffineInitResult(
		affine=affine,
		sign=sign,
		ext_quad=seed_ext_quad,
		model_quad=model_quad,
		sampled_count=samples * samples,
		hit_count=hit_count,
		kept_count=len(kept_hw),
		rejected_far_count=rejected_far,
		model_quad_count=int(patch_quads.shape[0]),
		ext_step_h=ext_h,
		ext_step_w=ext_w,
		model_step_h=model_h,
		model_step_w=model_w,
		model_radius_h=radius_h,
		model_radius_w=radius_w,
		seed_uv_rmse=float(sample_terms["uv_rmse"]),
		seed_uv_max=float(sample_terms["uv_max"]),
		seed_valid_count=int(sample_terms["valid"]),
		seed_loss=float(sample_terms["loss"]),
		seed_dist=float(sample_terms["dist"]),
		seed_dist_avg=float(sample_terms["dist_avg"]),
		seed_vec=float(sample_terms["vec"]),
		seed_norm=float(sample_terms["norm"]),
		seed_turn=float(sample_terms["turn"]),
		seed_turn_smp=int(sample_terms["turn_smp"]),
	)


def _apply_seed_quad_init_metadata(fixture: MapFixture, result: SeedQuadAffineInitResult) -> None:
	fixture.metadata["sign"] = int(result.sign)
	fixture.metadata["seed_quad_init"] = {
		"ext_quad": [int(v) for v in result.ext_quad],
		"model_quad": [int(v) for v in result.model_quad],
		"sign": int(result.sign),
		"sign_semantics": "model_normal_alignment",
		"samples": int(result.sampled_count),
		"hits": int(result.hit_count),
		"kept": int(result.kept_count),
		"rejected_far": int(result.rejected_far_count),
		"model_quads": int(result.model_quad_count),
		"ext_step_h": float(result.ext_step_h),
		"ext_step_w": float(result.ext_step_w),
		"model_step_h": float(result.model_step_h),
		"model_step_w": float(result.model_step_w),
		"model_radius_h": int(result.model_radius_h),
		"model_radius_w": int(result.model_radius_w),
		"seed_loss": float(result.seed_loss),
		"seed_dist": float(result.seed_dist),
		"seed_dist_avg": float(result.seed_dist_avg),
		"seed_vec": float(result.seed_vec),
		"seed_norm": float(result.seed_norm),
		"seed_turn": float(result.seed_turn),
		"seed_turn_smp": int(result.seed_turn_smp),
		"seed_uv_rmse": float(result.seed_uv_rmse),
		"seed_uv_max": float(result.seed_uv_max),
		"seed_valid": int(result.seed_valid_count),
	}


def _affine_from_seed_ext_quads(
	*,
	fixture: MapFixture,
	stage_cfg: SnapSurfConfig,
	seed_hw: torch.Tensor,
	seed_model_uv: torch.Tensor,
	raw: Any = None,
) -> torch.Tensor | None:
	result = _seed_quad_affine_init_result(
		fixture=fixture,
		stage_cfg=stage_cfg,
		seed_hw=seed_hw,
		seed_model_uv=seed_model_uv,
		raw=raw,
	)
	if result is None:
		return None
	return result.affine


def _is_affine_init_scan(stage: GlobalMapStageConfig) -> bool:
	return stage.name in {"affine_init_scan", "affine_multistart_scan", "affine_init_multistart"}


def _is_affine_seed_quad_init(stage: GlobalMapStageConfig) -> bool:
	return stage.name in {"affine_seed_quad_init", "seed_quad_affine_init", "seed2q"}


def _score_affine_tensor(
	affine_tensor: torch.Tensor,
	*,
	affine_model: AffineMapModel,
	fixture: MapFixture,
	stage_cfg: SnapSurfConfig,
	seed_hw: torch.Tensor,
	station_target: torch.Tensor,
	w_station: float,
) -> tuple[float, dict[str, torch.Tensor], dict[str, float], torch.Tensor]:
	with torch.no_grad():
		affine_model.affine.copy_(affine_tensor)
	uv = affine_model()
	loss, terms = _objective_for_uv(uv=uv, fixture=fixture, cfg=stage_cfg, level=0)
	station_raw = loss.new_zeros(())
	if float(w_station) > 0.0:
		station_raw = _station_loss(uv, seed_hw, station_target)
		loss = loss + float(w_station) * station_raw
	terms = dict(terms)
	terms["station"] = station_raw.detach()
	err = _fixture_mapping_error(uv.detach(), fixture)
	return float(loss.detach().cpu()), terms, err, affine_model.affine.detach().clone()


def _optimize_affine_candidate(
	*,
	candidate: torch.Tensor,
	affine_model: AffineMapModel,
	fixture: MapFixture,
	stage_cfg: SnapSurfConfig,
	seed_hw: torch.Tensor,
	station_target: torch.Tensor,
	w_station: float,
	steps: int,
	lr: float,
	cancel_fn=None,
) -> tuple[float, torch.Tensor]:
	with torch.no_grad():
		affine_model.affine.copy_(candidate)
	opt = torch.optim.Adam([affine_model.affine], lr=float(lr)) if int(steps) > 0 else None
	last_loss = float("inf")
	stage_active_quad = _level_active_quad(_full_active_quad(fixture), 0)
	for _ in range(max(1, int(steps))):
		if cancel_fn is not None:
			cancel_fn()
		if opt is not None:
			opt.zero_grad(set_to_none=True)
		uv = affine_model()
		loss, _terms = _objective_for_uv(
			uv=uv,
			fixture=fixture,
			cfg=stage_cfg,
			level=0,
			need_stats=False,
			active_quad=stage_active_quad,
		)
		if float(w_station) > 0.0:
			loss = loss + float(w_station) * _station_loss(uv, seed_hw, station_target)
		last_loss = float(loss.detach().cpu())
		if opt is None:
			break
		loss.backward()
		opt.step()
	return last_loss, affine_model.affine.detach().clone()


_AFFINE_DIAG_COLUMNS = (
	ProgressColumn("idx", "idx", "candidate index", min_width=3),
	ProgressColumn("rot", "rot", "initial rotation degrees", min_width=7),
	ProgressColumn("scl", "scl", "initial scale", min_width=6),
	ProgressColumn("isdet", "isdet", "initial determinant scale", min_width=7),
	ProgressColumn("ish", "ish", "initial h-axis scale", min_width=7),
	ProgressColumn("isw", "isw", "initial w-axis scale", min_width=7),
	ProgressColumn("irot", "irot", "initial affine rotation degrees", min_width=7),
	ProgressColumn("sdet", "sdet", "affine determinant scale", min_width=7),
	ProgressColumn("sh", "sh", "affine h-axis scale", min_width=7),
	ProgressColumn("sw", "sw", "affine w-axis scale", min_width=7),
	ProgressColumn("arot", "arot", "affine rotation degrees", min_width=7),
	ProgressColumn("iloss", "iloss", "initial objective", min_width=7),
	ProgressColumn("loss", "loss", "final objective", min_width=7),
	ProgressColumn("dist", "dst", "distance loss", min_width=7),
	ProgressColumn("vec", "vec", "vector-normal loss", min_width=7),
	ProgressColumn("norm", "nrm", "surface-normal loss", min_width=7),
	ProgressColumn("turn", "turn", "lifted z-heading loss", min_width=7),
	ProgressColumn("smooth", "smo", "uv smooth loss", min_width=7),
	ProgressColumn("bend", "bnd", "uv bend loss", min_width=7),
	ProgressColumn("jac", "jac", "jacobian loss", min_width=7),
	ProgressColumn("metric_smooth", "met", "model metric loss", min_width=7),
	ProgressColumn("area_smooth", "ar", "external physical area loss", min_width=7),
	ProgressColumn("prior", "pri", "dense prior loss", min_width=7),
	ProgressColumn("station", "stat", "station loss", min_width=7),
	ProgressColumn("avgd", "avgd", "avg fixture model quad distance", min_width=7),
	ProgressColumn("maxd", "maxd", "max fixture model quad distance", min_width=7),
	ProgressColumn("smp", "smp", "objective samples", min_width=6),
	ProgressColumn("i00", "i00", "initial affine h-from-h", min_width=8),
	ProgressColumn("i01", "i01", "initial affine h-from-w", min_width=8),
	ProgressColumn("i02", "i02", "initial affine h offset", min_width=8),
	ProgressColumn("i10", "i10", "initial affine w-from-h", min_width=8),
	ProgressColumn("i11", "i11", "initial affine w-from-w", min_width=8),
	ProgressColumn("i12", "i12", "initial affine w offset", min_width=8),
	ProgressColumn("a00", "a00", "affine h-from-h", min_width=8),
	ProgressColumn("a01", "a01", "affine h-from-w", min_width=8),
	ProgressColumn("a02", "a02", "affine h offset", min_width=8),
	ProgressColumn("a10", "a10", "affine w-from-h", min_width=8),
	ProgressColumn("a11", "a11", "affine w-from-w", min_width=8),
	ProgressColumn("a12", "a12", "affine w offset", min_width=8),
)


def _affine_diag_widths() -> dict[str, int]:
	values = {col.key: "-1.0e+99" for col in _AFFINE_DIAG_COLUMNS}
	values["idx"] = "1000000"
	values["rot"] = "-180.000"
	values["scl"] = "100.000"
	values["irot"] = "-180.000"
	values["arot"] = "-180.000"
	values["smp"] = "1000000"
	for key in ("i00", "i01", "i02", "i10", "i11", "i12"):
		values[key] = "-1.0e+99"
	return progress_widths(_AFFINE_DIAG_COLUMNS, values)


def _affine_summary(affine_tensor: torch.Tensor | None) -> dict[str, float] | None:
	if affine_tensor is None:
		return None
	aff = affine_tensor.detach().cpu()
	a00 = float(aff[0, 0])
	a01 = float(aff[0, 1])
	a10 = float(aff[1, 0])
	a11 = float(aff[1, 1])
	det = a00 * a11 - a01 * a10
	return {
		"sdet": math.copysign(math.sqrt(abs(det)), det) if math.isfinite(det) else float("nan"),
		"sh": math.hypot(a00, a10),
		"sw": math.hypot(a01, a11),
		"rot": math.degrees(math.atan2(a10, a00)),
	}


def _affine_diag_values(
	*,
	idx: int | str,
	rot: float | None,
	scale: float | None,
	initial_loss: float | None,
	final_loss: float,
	terms: dict[str, torch.Tensor],
	err: dict[str, float],
	initial_affine_tensor: torch.Tensor | None,
	affine_tensor: torch.Tensor,
) -> dict[str, str]:
	aff = affine_tensor.detach().cpu()
	init_aff = initial_affine_tensor.detach().cpu() if initial_affine_tensor is not None else None
	init_summary = _affine_summary(init_aff)
	summary = _affine_summary(aff)
	def init_value(i: int, j: int) -> str:
		return "" if init_aff is None else format_progress_value(float(init_aff[i, j]))
	def summary_value(values: dict[str, float] | None, key: str) -> str:
		return "" if values is None else format_progress_value(float(values[key]))
	return {
		"idx": str(idx),
		"rot": "" if rot is None else format_progress_value(float(rot)),
		"scl": "" if scale is None else format_progress_value(float(scale)),
		"isdet": summary_value(init_summary, "sdet"),
		"ish": summary_value(init_summary, "sh"),
		"isw": summary_value(init_summary, "sw"),
		"irot": summary_value(init_summary, "rot"),
		"sdet": summary_value(summary, "sdet"),
		"sh": summary_value(summary, "sh"),
		"sw": summary_value(summary, "sw"),
		"arot": summary_value(summary, "rot"),
		"iloss": "" if initial_loss is None else format_progress_value(float(initial_loss)),
		"loss": format_progress_value(float(final_loss)),
		"dist": format_progress_value(_global_term_value(terms, "dist")),
		"vec": format_progress_value(_global_term_value(terms, "vec")),
		"norm": format_progress_value(_global_term_value(terms, "norm")),
		"turn": format_progress_value(_global_term_value(terms, "turn")),
		"smooth": format_progress_value(_global_term_value(terms, "smooth")),
		"bend": format_progress_value(_global_term_value(terms, "bend")),
		"jac": format_progress_value(_global_term_value(terms, "jac")),
		"metric_smooth": format_progress_value(_global_term_value(terms, "metric_smooth")),
		"area_smooth": format_progress_value(_global_term_value(terms, "area_smooth")),
		"prior": format_progress_value(_global_term_value(terms, "prior")),
		"station": format_progress_value(_global_term_value(terms, "station")),
		"avgd": format_progress_value(float(err["avg_model_quad_distance"])),
		"maxd": format_progress_value(float(err["max_model_quad_distance"])),
		"smp": str(int(err["mapping_error_samples"])),
		"i00": init_value(0, 0),
		"i01": init_value(0, 1),
		"i02": init_value(0, 2),
		"i10": init_value(1, 0),
		"i11": init_value(1, 1),
		"i12": init_value(1, 2),
		"a00": format_progress_value(float(aff[0, 0])),
		"a01": format_progress_value(float(aff[0, 1])),
		"a02": format_progress_value(float(aff[0, 2])),
		"a10": format_progress_value(float(aff[1, 0])),
		"a11": format_progress_value(float(aff[1, 1])),
		"a12": format_progress_value(float(aff[1, 2])),
	}


def _print_affine_diag_header(label: str, widths: dict[str, int]) -> None:
	_print_progress_legend_once(
		prefix=f"[snap_surf.map_global] {label}",
		items=[(col.label, col.description) for col in _AFFINE_DIAG_COLUMNS],
	)
	print(progress_header(_AFFINE_DIAG_COLUMNS, widths), flush=True)


def _print_affine_diag_row(widths: dict[str, int], values: dict[str, str]) -> None:
	print(progress_row(_AFFINE_DIAG_COLUMNS, widths, values), flush=True)


def _select_affine_seed_grid_candidate(
	*,
	base_affine: torch.Tensor,
	stage: GlobalMapStageConfig,
	affine_model: AffineMapModel,
	fixture: MapFixture,
	stage_cfg: SnapSurfConfig,
	seed_hw: torch.Tensor,
	station_target: torch.Tensor,
	w_station: float,
	cancel_fn=None,
) -> tuple[torch.Tensor, float]:
	cfg = _affine_seed_grid_cfg(stage)
	if not bool(cfg.get("enabled", True)):
		return base_affine.detach().clone(), float("nan")
	rot_deg = _float_list(cfg.get("rot_deg", cfg.get("rotations_deg", [-10.0, -5.0, 0.0, 5.0, 10.0])), name="affine_seed_grid rot_deg")
	scales = _float_list(cfg.get("scales", [0.75, 0.9, 1.0, 1.1, 1.25]), name="affine_seed_grid scales")
	candidates = _affine_seed_grid_candidates(
		base_affine=base_affine,
		seed_ext_hw=seed_hw,
		rot_deg=rot_deg,
		scales=scales,
	)
	if not candidates:
		return base_affine.detach().clone(), float("nan")
	base_loss, _base_terms, _base_err, base_scored = _score_affine_tensor(
		base_affine,
		affine_model=affine_model,
		fixture=fixture,
		stage_cfg=stage_cfg,
		seed_hw=seed_hw,
		station_target=station_target,
		w_station=w_station,
	)
	best_loss = base_loss if math.isfinite(base_loss) else float("inf")
	best_idx: int | str = "seedq"
	best_affine = base_scored
	widths = _affine_diag_widths()
	_print_affine_diag_header("affine seed quad grid", widths)
	for cand_idx, rot, scale, cand in candidates:
		if cancel_fn is not None:
			cancel_fn()
		loss, terms, err, scored_affine = _score_affine_tensor(
			cand,
			affine_model=affine_model,
			fixture=fixture,
			stage_cfg=stage_cfg,
			seed_hw=seed_hw,
			station_target=station_target,
			w_station=w_station,
		)
		_print_affine_diag_row(
			widths,
			_affine_diag_values(
				idx=cand_idx,
				rot=rot,
				scale=scale,
				initial_loss=base_loss,
				final_loss=loss,
				terms=terms,
				err=err,
				initial_affine_tensor=base_affine,
				affine_tensor=scored_affine,
			),
		)
		if math.isfinite(loss) and loss < best_loss:
			best_loss = loss
			best_idx = cand_idx
			best_affine = scored_affine
	with torch.no_grad():
		affine_model.affine.copy_(best_affine)
	print(
		f"[snap_surf.map_global] affine seed quad grid candidates={len(candidates)} "
		f"best_idx={best_idx} base_loss={base_loss:.6g} best_loss={best_loss:.6g}",
		flush=True,
	)
	return best_affine.detach().clone(), best_loss


def _fit_reference_affine(fixture: MapFixture) -> tuple[torch.Tensor, float] | None:
	ref = fixture.reference_uv
	valid = torch.isfinite(ref).all(dim=-1)
	if int(valid.sum().detach().cpu()) < 3:
		return None
	H, W = int(ref.shape[0]), int(ref.shape[1])
	hh = torch.arange(H, device=ref.device, dtype=ref.dtype).view(H, 1).expand(H, W)
	ww = torch.arange(W, device=ref.device, dtype=ref.dtype).view(1, W).expand(H, W)
	ones = torch.ones_like(hh)
	x = torch.stack([hh, ww, ones], dim=-1)[valid]
	y = ref[valid]
	try:
		sol = torch.linalg.lstsq(x, y).solution
	except RuntimeError:
		return None
	affine = sol.transpose(0, 1).contiguous()
	pred = x @ sol
	uv_mse = float((pred - y).square().sum(dim=-1).mean().detach().cpu())
	return affine, uv_mse


def _print_reference_affine_diagnostic(
	*,
	affine: AffineMapModel,
	fixture: MapFixture,
	stage_cfg: SnapSurfConfig,
	seed_hw: torch.Tensor,
	station_target: torch.Tensor,
) -> None:
	fit = _fit_reference_affine(fixture)
	if fit is None:
		print("[snap_surf.map_global] affine fixture reference unavailable: fewer than 3 finite reference samples", flush=True)
		return
	ref_affine, uv_mse = fit
	original = affine.affine.detach().clone()
	try:
		final_loss, terms, err, scored_affine = _score_affine_tensor(
			ref_affine,
			affine_model=affine,
			fixture=fixture,
			stage_cfg=stage_cfg,
			seed_hw=seed_hw,
			station_target=station_target,
			w_station=0.0,
		)
	finally:
		with torch.no_grad():
			affine.affine.copy_(original)
	widths = _affine_diag_widths()
	print(f"[snap_surf.map_global] affine fixture reference uv_mse={uv_mse:.6g}", flush=True)
	_print_affine_diag_header("affine fixture reference", widths)
	_print_affine_diag_row(
		widths,
		_affine_diag_values(
			idx="ref",
			rot=None,
			scale=None,
			initial_loss=None,
			final_loss=final_loss,
			terms=terms,
			err=err,
			initial_affine_tensor=None,
			affine_tensor=scored_affine,
		),
	)


def _run_affine_multistart(
	*,
	cfg_global: GlobalMapConfig,
	stage: GlobalMapStageConfig,
	affine: AffineMapModel,
	fixture: MapFixture,
	stage_cfg: SnapSurfConfig,
	seed_hw: torch.Tensor,
	station_target: torch.Tensor,
	w_station: float,
	cancel_fn=None,
) -> SeedQuadAffineInitResult | None:
	cfg = _affine_multistart_cfg(cfg_global, stage)
	if not bool(cfg.get("enabled", False)):
		return None
	rot_raw = cfg.get("rot_deg", cfg.get("rotations_deg", [-30.0, -15.0, 0.0, 15.0, 30.0]))
	scale_raw = cfg.get("scales", [0.75, 1.0, 1.25])
	if not isinstance(rot_raw, list) or not isinstance(scale_raw, list):
		raise ValueError("affine_multistart rot_deg and scales must be lists")
	rot_deg = [float(v) for v in rot_raw]
	scales = [float(v) for v in scale_raw]
	steps = max(0, int(cfg.get("steps", 25)))
	lr = float(cfg.get("lr", stage.lr))
	seed_model_uv = _seed_model_uv(fixture, seed_hw)
	candidates = _affine_multistart_candidates(
		seed_ext_hw=seed_hw,
		seed_model_uv=seed_model_uv,
		rot_deg=rot_deg,
		scales=scales,
	)
	seed_quad_cfg = cfg.get("seed_quad_affine", True)
	seed_quad_result: SeedQuadAffineInitResult | None = None
	if isinstance(seed_quad_cfg, dict):
		seed_quad_enabled = bool(seed_quad_cfg.get("enabled", True))
	else:
		seed_quad_enabled = bool(seed_quad_cfg)
	if seed_quad_enabled:
		seed_quad_result = _seed_quad_affine_init_result(
			fixture=fixture,
			stage_cfg=stage_cfg,
			seed_hw=seed_hw,
			seed_model_uv=seed_model_uv,
			raw=seed_quad_cfg,
		)
		if seed_quad_result is not None:
			_apply_seed_quad_init_metadata(fixture, seed_quad_result)
			candidates.insert(0, ("seedq", None, None, seed_quad_result.affine))
	if not candidates:
		return seed_quad_result
	start_affine = affine.affine.detach().clone()
	best_loss = float("inf")
	best_affine = start_affine
	widths = _affine_diag_widths()
	_print_affine_diag_header("affine multistart", widths)
	for cand_idx, rot, scale, cand in candidates:
		if cancel_fn is not None:
			cancel_fn()
		initial_loss, _initial_terms, _initial_err, _initial_affine = _score_affine_tensor(
			cand,
			affine_model=affine,
			fixture=fixture,
			stage_cfg=stage_cfg,
			seed_hw=seed_hw,
			station_target=station_target,
			w_station=w_station,
		)
		loss, value = _optimize_affine_candidate(
			candidate=cand,
			affine_model=affine,
			fixture=fixture,
			stage_cfg=stage_cfg,
			seed_hw=seed_hw,
			station_target=station_target,
			w_station=w_station,
			steps=steps,
			lr=lr,
			cancel_fn=cancel_fn,
		)
		final_loss, final_terms, final_err, final_affine = _score_affine_tensor(
			value,
			affine_model=affine,
			fixture=fixture,
			stage_cfg=stage_cfg,
			seed_hw=seed_hw,
			station_target=station_target,
			w_station=w_station,
		)
		_print_affine_diag_row(
			widths,
			_affine_diag_values(
				idx=cand_idx,
				rot=rot,
				scale=scale,
				initial_loss=initial_loss,
				final_loss=final_loss,
				terms=final_terms,
				err=final_err,
				initial_affine_tensor=cand,
				affine_tensor=final_affine,
			),
		)
		loss = final_loss
		value = final_affine
		if math.isfinite(loss) and loss < best_loss:
			best_loss = loss
			best_affine = value
	with torch.no_grad():
		affine.affine.copy_(best_affine)
	print(
		f"[snap_surf.map_global] affine multistart candidates={len(candidates)} "
		f"steps={steps} best_loss={best_loss:.6g}",
		flush=True,
	)
	return seed_quad_result


def _prepare_affine_seed_quad_candidate(
	*,
	stage: GlobalMapStageConfig,
	affine: AffineMapModel,
	fixture: MapFixture,
	stage_cfg: SnapSurfConfig,
	seed_hw: torch.Tensor,
	station_target: torch.Tensor,
	w_station: float,
	raw: Any,
	lr: float,
	stage_idx: int,
	progress_widths_run: dict[str, int] | None,
	progress_row_idx: int = 0,
	debug_obj_root: Path | None = None,
	cancel_fn=None,
) -> tuple[SeedQuadAffineInitResult | None, torch.Tensor | None, float, int]:
	if isinstance(raw, dict) and not bool(raw.get("enabled", True)):
		return None, None, float("nan"), 0
	if not isinstance(raw, dict) and not bool(raw):
		return None, None, float("nan"), 0
	t_prepare = time.perf_counter()
	reopt_cfg = raw if isinstance(raw, dict) else {}
	reopt_enabled = bool(reopt_cfg.get("expansion_reopt", reopt_cfg.get("grow_reopt", False)))
	reopt_steps = max(0, int(reopt_cfg.get("expansion_reopt_steps", reopt_cfg.get("grow_reopt_steps", 100))))
	reopt_lr = float(reopt_cfg.get("expansion_reopt_lr", reopt_cfg.get("grow_reopt_lr", lr)))
	status_interval = max(0, int(reopt_cfg.get(
		"status_interval",
		reopt_cfg.get("debug_print_interval", stage.args.get("status_interval", stage.args.get("debug_print_interval", 100))),
	)))
	lr_warmup_steps = _lr_warmup_steps(stage.args)
	t_seed = time.perf_counter()
	seed_result = _seed_quad_affine_init_result(
		fixture=fixture,
		stage_cfg=stage_cfg,
		seed_hw=seed_hw,
		seed_model_uv=_seed_model_uv(fixture, seed_hw),
		raw=raw,
	)
	print(
		f"[snap_surf.map_global] affine seed quad prepare seed_fit={1000.0 * (time.perf_counter() - t_seed):.3f}ms",
		flush=True,
	)
	if seed_result is None:
		print("[snap_surf.map_global] affine seed quad init unavailable", flush=True)
		return None, None, float("nan"), 0
	_apply_seed_quad_init_metadata(fixture, seed_result)
	candidate = seed_result.affine
	_write_affine_seed_initial_debug_radius(
		debug_obj_root=debug_obj_root,
		affine_tensor=candidate,
		fixture=fixture,
		stage_cfg=stage_cfg,
		seed_ext_quad=seed_result.ext_quad,
		radius=128,
	)
	progress_rows = 0
	if reopt_enabled:
		t_reopt = time.perf_counter()
		with torch.no_grad():
			affine.affine.copy_(candidate)
		_reopt_rows, progress_rows = _run_affine_seed_quad_expansion_reopt(
			affine=affine,
			fixture=fixture,
			stage_cfg=stage_cfg,
			stage=stage,
			seed_ext_quad=seed_result.ext_quad,
			seed_hw=seed_hw,
			station_target=station_target,
			w_station=w_station,
			steps=reopt_steps,
			lr=reopt_lr,
			status_interval=status_interval,
			lr_warmup_steps=lr_warmup_steps,
			stage_idx=stage_idx,
			progress_widths_run=progress_widths_run,
			progress_row_idx=progress_row_idx,
			debug_obj_root=debug_obj_root,
			cancel_fn=cancel_fn,
		)
		candidate = affine.affine.detach().clone()
		print(
			f"[snap_surf.map_global] affine seed quad prepare expansion_reopt={1000.0 * (time.perf_counter() - t_reopt):.3f}ms",
			flush=True,
		)
	t_score = time.perf_counter()
	seed_loss, _initial_terms, _initial_err, _initial_affine = _score_affine_tensor(
		candidate,
		affine_model=affine,
		fixture=fixture,
		stage_cfg=stage_cfg,
		seed_hw=seed_hw,
		station_target=station_target,
		w_station=w_station,
	)
	print(
		f"[snap_surf.map_global] affine seed quad prepare score={1000.0 * (time.perf_counter() - t_score):.3f}ms",
		flush=True,
	)
	t_grid = time.perf_counter()
	candidate, initial_loss = _select_affine_seed_grid_candidate(
		base_affine=candidate,
		stage=stage,
		affine_model=affine,
		fixture=fixture,
		stage_cfg=stage_cfg,
		seed_hw=seed_hw,
		station_target=station_target,
		w_station=w_station,
		cancel_fn=cancel_fn,
	)
	print(
		f"[snap_surf.map_global] affine seed quad prepare grid={1000.0 * (time.perf_counter() - t_grid):.3f}ms "
		f"total={1000.0 * (time.perf_counter() - t_prepare):.3f}ms",
		flush=True,
	)
	if not math.isfinite(initial_loss):
		initial_loss = seed_loss
	with torch.no_grad():
		affine.affine.copy_(candidate)
	return seed_result, candidate.detach().clone(), initial_loss, progress_rows


def _run_affine_seed_quad_init(
	*,
	stage_idx: int,
	stage: GlobalMapStageConfig,
	affine: AffineMapModel,
	fixture: MapFixture,
	stage_cfg: SnapSurfConfig,
	seed_hw: torch.Tensor,
	station_target: torch.Tensor,
	w_station: float,
	progress_widths_run: dict[str, int],
	progress_row_idx: int,
	out_root: Path | None = None,
	write_objs: bool = False,
) -> int:
	raw = stage.args.get("affine_seed_quad_init", stage.args.get("seed_quad_affine", {}))
	if isinstance(raw, dict):
		if not bool(raw.get("enabled", True)):
			return 0
		steps = max(0, int(raw.get("steps", stage.steps)))
		lr = float(raw.get("lr", stage.lr))
	else:
		if not bool(raw):
			return 0
		steps = max(0, int(stage.steps))
		lr = float(stage.lr)
	debug_obj_root = None
	if out_root is not None and bool(write_objs):
		label = stage.name or _stage_param_label(stage.params, fallback="affine_seed_quad_init")
		debug_obj_root = Path(out_root) / "objs" / f"stage_{int(stage_idx):03d}_{_debug_obj_safe_label(label)}" / "expansion_reopt"
	seed_result, candidate, initial_loss, prep_progress_rows = _prepare_affine_seed_quad_candidate(
		stage=stage,
		affine=affine,
		fixture=fixture,
		stage_cfg=stage_cfg,
		seed_hw=seed_hw,
		station_target=station_target,
		w_station=w_station,
		raw=raw,
		lr=lr,
		stage_idx=stage_idx,
		progress_widths_run=progress_widths_run,
		progress_row_idx=progress_row_idx,
		debug_obj_root=debug_obj_root,
	)
	if seed_result is None or candidate is None:
		return 0
	opt = torch.optim.Adam([affine.affine], lr=float(lr)) if steps > 0 else None
	if opt is not None:
		_capture_optimizer_target_lrs(opt)
	lr_autoscale = _make_lr_autoscale_state(stage.args)
	status_interval = max(0, int(stage.args.get("status_interval", stage.args.get("debug_print_interval", 100))))
	lr_warmup_steps = _lr_warmup_steps(stage.args)
	last_status_time: float | None = None
	last_status_step = 0
	progress_rows = int(prep_progress_rows)
	stage_active_quad = _level_active_quad(_full_active_quad(fixture), 0)
	with torch.no_grad():
		report_loss, report_terms, report_err = _global_progress_state(
			uv=affine(),
			fixture=fixture,
			cfg=stage_cfg,
			seed_hw=seed_hw,
			station_target=station_target,
			w_station=w_station,
			active_quad=stage_active_quad,
		)
	_print_global_progress(
		row_idx=int(progress_row_idx) + progress_rows,
		widths=progress_widths_run,
		stage_idx=stage_idx,
		iter_label=f"0/{steps}",
		lr=float(lr),
		level=0,
		loss=report_loss,
		terms=report_terms,
		it_s=None,
		err=report_err,
	)
	progress_rows += 1
	for step in range(steps):
		if opt is not None:
			opt.zero_grad(set_to_none=True)
		uv = affine()
		loss, _terms = _objective_for_uv(
			uv=uv,
			fixture=fixture,
			cfg=stage_cfg,
			level=0,
			need_stats=False,
			active_quad=stage_active_quad,
		)
		station_raw = loss.new_zeros(())
		if float(w_station) > 0.0:
			station_raw = _station_loss(uv, seed_hw, station_target)
			loss = loss + float(w_station) * station_raw
		if opt is not None:
			loss.backward()
			_apply_optimizer_lr_schedule(
				opt,
				step1=step + 1,
				warmup_steps=lr_warmup_steps,
				autoscale=lr_autoscale,
				loss=loss,
			)
			opt.step()
		step1 = step + 1
		status_due = (
			step == 0 or
			step1 == steps or
			(status_interval > 0 and (step1 % status_interval) == 0)
		)
		if status_due:
			with torch.no_grad():
				uv_after = affine()
				report_loss, report_terms, err = _global_progress_state(
					uv=uv_after,
					fixture=fixture,
					cfg=stage_cfg,
					seed_hw=seed_hw,
					station_target=station_target,
					w_station=w_station,
					active_quad=stage_active_quad,
				)
			now = time.monotonic()
			it_s = None
			if last_status_time is not None:
				it_s = float(step1 - last_status_step) / max(1.0e-9, now - last_status_time)
			last_status_time = now
			last_status_step = step1
			_print_global_progress(
				row_idx=int(progress_row_idx) + progress_rows,
				widths=progress_widths_run,
				stage_idx=stage_idx,
				iter_label=f"{step1}/{steps}",
				lr=_optimizer_lr_for_display(opt, float(lr)),
				level=0,
				loss=report_loss,
				terms=report_terms,
				it_s=it_s,
				err=err,
			)
			progress_rows += 1
	value = affine.affine.detach().clone()
	final_loss, final_terms, final_err, final_affine = _score_affine_tensor(
		value,
		affine_model=affine,
		fixture=fixture,
		stage_cfg=stage_cfg,
		seed_hw=seed_hw,
		station_target=station_target,
		w_station=w_station,
	)
	widths = _affine_diag_widths()
	_print_affine_diag_header("affine seed quad init", widths)
	_print_affine_diag_row(
		widths,
		_affine_diag_values(
			idx="seedq",
			rot=None,
			scale=None,
			initial_loss=initial_loss,
			final_loss=final_loss,
			terms=final_terms,
			err=final_err,
			initial_affine_tensor=candidate,
			affine_tensor=final_affine,
		),
	)
	with torch.no_grad():
		affine.affine.copy_(final_affine)
	print(
		f"[snap_surf.map_global] affine seed quad init steps={steps} loss={final_loss:.6g}",
		flush=True,
	)
	return progress_rows


def _stage_loss_cfg(base_cfg: SnapSurfConfig, stage: GlobalMapStageConfig) -> SnapSurfConfig:
	mi = base_cfg.map_init
	if isinstance(stage.w_fac, dict):
		weights = {str(k): float(v) for k, v in stage.w_fac.items()}
		bad = sorted(set(weights.keys()) - set(_STAGE_W_FAC_KEYS.values()))
		if bad:
			raise ValueError(f"global map stage '{stage.name}' w_fac: unknown term(s): {bad}")
		return replace(
			base_cfg,
			map_init=replace(
				mi,
				w_dist=float(mi.w_dist) * weights.get("dist", 1.0),
				w_vec_normal=float(mi.w_vec_normal) * weights.get("vec", 1.0),
				w_surface_normal=float(mi.w_surface_normal) * weights.get("norm", 1.0),
				w_z_lift=float(mi.w_z_lift) * weights.get("turn", 1.0),
				w_smooth=float(mi.w_smooth) * weights.get("smooth", 1.0),
				w_bend=float(mi.w_bend) * weights.get("bend", 1.0),
				w_jac=float(mi.w_jac) * weights.get("jac", 1.0),
				w_metric_smooth=float(mi.w_metric_smooth) * weights.get("metric_smooth", 1.0),
				w_area_smooth=float(mi.w_area_smooth) * weights.get("area_smooth", 1.0),
				w_dense_prior=float(mi.w_dense_prior) * weights.get("prior", 1.0),
			),
		)
	scale = float(stage.w_fac)
	return replace(
		base_cfg,
		map_init=replace(
			mi,
			w_dist=float(mi.w_dist) * scale,
			w_vec_normal=float(mi.w_vec_normal) * scale,
			w_surface_normal=float(mi.w_surface_normal) * scale,
			w_z_lift=float(mi.w_z_lift) * scale,
			w_smooth=float(mi.w_smooth) * scale,
			w_bend=float(mi.w_bend) * scale,
			w_jac=float(mi.w_jac) * scale,
			w_metric_smooth=float(mi.w_metric_smooth) * scale,
			w_area_smooth=float(mi.w_area_smooth) * scale,
			w_dense_prior=float(mi.w_dense_prior) * scale,
		),
	)


def _stage_station_weight(cfg_global: GlobalMapConfig, stage: GlobalMapStageConfig) -> float:
	base = float(stage.args.get("map_station_t", stage.args.get("w_station_t", cfg_global.base.get("map_station_t", 0.0))))
	if isinstance(stage.w_fac, dict):
		return base * float(stage.w_fac.get("map_station_t", 1.0))
	return base * float(stage.w_fac)


def _station_loss(uv: torch.Tensor, seed_ext_hw: torch.Tensor, target_uv: torch.Tensor) -> torch.Tensor:
	H, W = int(uv.shape[0]), int(uv.shape[1])
	coords = seed_ext_hw.view(1, 2)
	h = coords[:, 0].clamp(0.0, float(max(0, H - 1)))
	w = coords[:, 1].clamp(0.0, float(max(0, W - 1)))
	h0 = torch.floor(h).clamp(0, max(0, H - 2)).long()
	w0 = torch.floor(w).clamp(0, max(0, W - 2)).long()
	h1 = (h0 + 1).clamp(max=max(0, H - 1))
	w1 = (w0 + 1).clamp(max=max(0, W - 1))
	fh = (h - h0.to(dtype=uv.dtype)).view(1, 1)
	fw = (w - w0.to(dtype=uv.dtype)).view(1, 1)
	value = (
		(1.0 - fh) * (1.0 - fw) * uv[h0, w0] +
		fh * (1.0 - fw) * uv[h1, w0] +
		(1.0 - fh) * fw * uv[h0, w1] +
		fh * fw * uv[h1, w1]
	).view(2)
	return (value - target_uv).square().mean()


def _level_seed_hw(seed_ext_hw: torch.Tensor, level: int) -> torch.Tensor:
	stride = float(1 << max(0, int(level)))
	return seed_ext_hw / stride


def _affine_uv_for_level(affine: AffineMapModel, ext_shape: tuple[int, int], level: int) -> torch.Tensor:
	level_i = max(0, int(level))
	if level_i == 0:
		return affine()
	coords = _level_coords(ext_shape, level_i, affine.affine)
	return affine.eval_at(coords)


def _stage_objective_level(stage: GlobalMapStageConfig, train_level: int, ext_shape: tuple[int, int]) -> int:
	args = stage.args if isinstance(stage.args, dict) else {}
	if "objective_min_scaledown" in args:
		level = max(0, int(args["objective_min_scaledown"]))
	elif "sample_min_scaledown" in args:
		level = max(0, int(args["sample_min_scaledown"]))
	elif (
		_truthy_arg(args.get("use_min_scaledown_sampling", False)) or
		_truthy_arg(args.get("sample_at_min_scaledown", False)) or
		_truthy_arg(args.get("coarse_sampling", False))
	):
		level = max(0, int(train_level))
	else:
		level = 0
	return min(level, _max_supported_level(ext_shape, level))


def _objective_for_uv(
	*,
	uv: torch.Tensor,
	fixture: MapFixture,
	cfg: SnapSurfConfig,
	level: int,
	z_lift: dict[str, Any] | None | object = _NO_Z_LIFT,
	need_stats: bool = True,
	active_quad: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
	active = _level_active_quad(_full_active_quad(fixture), int(level)) if active_quad is None else active_quad
	ext_coords = None if int(level) == 0 else _level_coords(fixture.ext_xyz.shape[:2], int(level), uv)
	sign_i = int(fixture.metadata.get("sign", 1) or 1)
	if z_lift is _NO_Z_LIFT:
		z_lift = _map_init_z_lift_for_fixture(fixture, cfg, sign=sign_i)
	z_lift_d = None if z_lift is _NO_Z_LIFT else z_lift
	ext_z_lift_theta = None if z_lift_d is None else z_lift_d["ext_theta_lifted"]
	ext_z_lift_valid = None if z_lift_d is None else z_lift_d["ext_valid"]
	return _map_init_objective(
		uv_full=uv,
		active_quad=active,
		ext_pos=fixture.ext_xyz,
		ext_normals=fixture.ext_normals,
		ext_valid=fixture.ext_valid,
		ext_quad_valid=fixture.ext_quad_valid,
		ext_coords=ext_coords,
		model_xyz=fixture.model_xyz,
		model_valid=fixture.model_valid,
		model_normals=fixture.model_normals,
		model_depth=int(fixture.metadata.get("model_depth", 0) or 0),
		sign=sign_i,
		cfg=cfg,
		allow_partial_model_samples=True,
		need_stats=bool(need_stats),
		ext_z_lift_theta=ext_z_lift_theta,
		ext_z_lift_valid=ext_z_lift_valid,
		model_z_lift_theta=None if z_lift_d is None else z_lift_d["model_theta_lifted"],
		model_z_lift_valid=None if z_lift_d is None else z_lift_d["model_valid"],
	)


def _uv_model_positions(
	uv: torch.Tensor,
	fixture: MapFixture,
) -> tuple[torch.Tensor, torch.Tensor]:
	depth = int(fixture.metadata.get("model_depth", 0) or 0)
	d = torch.full((*uv.shape[:-1], 1), float(depth), device=uv.device, dtype=uv.dtype)
	coords = torch.cat([d, uv], dim=-1)
	finite = torch.isfinite(coords).all(dim=-1)
	valid = finite & _quad_valid_at_coords(fixture.model_valid.bool(), coords, tuple(int(v) for v in fixture.model_valid.shape))
	safe_coords = torch.where(torch.isfinite(coords), coords, torch.zeros_like(coords))
	pos = _sample_surface_grid(fixture.model_xyz, safe_coords)
	valid = valid & torch.isfinite(pos).all(dim=-1)
	return pos, valid


def _fixture_mapping_error(uv: torch.Tensor, fixture: MapFixture) -> dict[str, float]:
	pos, valid = _uv_model_positions(uv, fixture)
	if tuple(fixture.reference_uv.shape[:2]) == tuple(uv.shape[:2]):
		reference_uv = fixture.reference_uv.to(device=uv.device, dtype=uv.dtype)
	else:
		level = 0
		for candidate in range(1, 32):
			try:
				if _map_init_dyadic_level_shape(
					int(fixture.reference_uv.shape[0]),
					int(fixture.reference_uv.shape[1]),
					candidate,
				) == tuple(int(v) for v in uv.shape[:2]):
					level = candidate
					break
			except ValueError:
				break
		if level > 0:
			coords = _level_coords(tuple(int(v) for v in fixture.reference_uv.shape[:2]), level, uv)
			reference_uv = _sample_surface_grid(
				fixture.reference_uv.to(device=uv.device, dtype=uv.dtype),
				coords,
			)
		else:
			reference_uv = fixture.reference_uv.to(device=uv.device, dtype=uv.dtype)
	ref_pos, ref_valid = _uv_model_positions(reference_uv, fixture)
	if tuple(valid.shape) != tuple(ref_valid.shape):
		return {
			"avg_model_quad_distance": 0.0,
			"max_model_quad_distance": 0.0,
			"mapping_error_samples": 0.0,
		}
	common = valid & ref_valid
	if not bool(common.any().detach().cpu()):
		return {
			"avg_model_quad_distance": 0.0,
			"max_model_quad_distance": 0.0,
			"mapping_error_samples": 0.0,
		}
	dist = (pos[common] - ref_pos[common]).norm(dim=-1)
	return {
		"avg_model_quad_distance": float(dist.mean().detach().cpu()),
		"max_model_quad_distance": float(dist.max().detach().cpu()),
		"mapping_error_samples": float(int(dist.numel())),
	}


def _vertex_mask_from_quad_mask(active_quad: torch.Tensor, shape: torch.Size | tuple[int, int]) -> torch.Tensor:
	H, W = int(shape[0]), int(shape[1])
	mask = torch.zeros((H, W), device=active_quad.device, dtype=torch.bool)
	if int(active_quad.numel()) == 0:
		return mask
	active = active_quad.bool()
	mask[:-1, :-1] |= active
	mask[1:, :-1] |= active
	mask[:-1, 1:] |= active
	mask[1:, 1:] |= active
	return mask


def _model_vertex_mask_for_uv(
	uv: torch.Tensor,
	ok: torch.Tensor,
	fixture: MapFixture,
) -> torch.Tensor:
	mask = torch.zeros_like(fixture.model_valid.bool())
	if not bool(ok.any().detach().cpu()):
		return mask
	D, H, W = (int(v) for v in fixture.model_valid.shape)
	if H <= 0 or W <= 0 or D <= 0:
		return mask
	depth = max(0, min(D - 1, int(fixture.metadata.get("model_depth", 0) or 0)))
	coords = uv[ok]
	finite = torch.isfinite(coords).all(dim=-1)
	if not bool(finite.any().detach().cpu()):
		return mask
	coords = coords[finite]
	if H == 1:
		h0 = torch.zeros(coords.shape[0], device=coords.device, dtype=torch.long)
		h1 = h0
	else:
		h0 = torch.floor(coords[:, 0]).clamp(0, H - 2).long()
		h1 = h0 + 1
	if W == 1:
		w0 = torch.zeros(coords.shape[0], device=coords.device, dtype=torch.long)
		w1 = w0
	else:
		w0 = torch.floor(coords[:, 1]).clamp(0, W - 2).long()
		w1 = w0 + 1
	mask[depth, h0, w0] = True
	mask[depth, h1, w0] = True
	mask[depth, h0, w1] = True
	mask[depth, h1, w1] = True
	return mask


def _write_stage_objs(
	out_root: Path,
	*,
	stage_idx: int,
	stage: GlobalMapStageConfig,
	uv: torch.Tensor,
	fixture: MapFixture,
) -> None:
	label = stage.name or _stage_param_label(stage.params, fallback="noop")
	out = out_root / "objs" / f"stage_{int(stage_idx):03d}_{_debug_obj_safe_label(label)}"
	_write_map_objs(
		out,
		uv=uv,
		fixture=fixture,
		meta={
			"stage": int(stage_idx),
			"name": stage.name,
			"params": list(_public_stage_params(stage.params)),
		},
	)


def _write_map_objs(
	out: Path,
	*,
	uv: torch.Tensor,
	fixture: MapFixture,
	meta: dict[str, Any],
	active_quad: torch.Tensor | None = None,
) -> None:
	out.mkdir(parents=True, exist_ok=True)
	model_pos, model_ok = _uv_model_positions(uv, fixture)
	ext_ok = fixture.ext_valid.bool() & torch.isfinite(fixture.ext_xyz).all(dim=-1)
	ext_vertex_mask = None
	if active_quad is not None:
		ext_vertex_mask = _vertex_mask_from_quad_mask(active_quad.to(device=fixture.ext_xyz.device), fixture.ext_xyz.shape[:2])
		ext_ok = ext_ok & ext_vertex_mask
	ok = model_ok & ext_ok
	ext_valid_debug = fixture.ext_valid if ext_vertex_mask is None else (fixture.ext_valid.bool() & ext_vertex_mask)
	_write_obj_mesh_2d(out / "ext_surface.obj", fixture.ext_xyz, ext_valid_debug)
	model_valid_debug = fixture.model_valid
	if active_quad is not None:
		model_valid_debug = fixture.model_valid.bool() & _model_vertex_mask_for_uv(uv, ok, fixture)
	_write_obj_mesh_3d_surfaces(out / "model_surface.obj", fixture.model_xyz, model_valid_debug)
	if bool(ok.any().detach().cpu()):
		_write_obj_lines(out / "map_ext_to_model.obj", fixture.ext_xyz[ok], model_pos[ok], label="global_map_ext_to_model")
	else:
		empty = fixture.model_xyz.new_empty(0, 3)
		_write_obj_lines(out / "map_ext_to_model.obj", empty, empty, label="global_map_ext_to_model")
	worst_count = 0
	if bool(ok.any().detach().cpu()):
		map_dist = (model_pos[ok] - fixture.ext_xyz[ok]).norm(dim=-1)
		k = max(1, int(math.ceil(float(map_dist.numel()) * 0.01)))
		_dist_vals, dist_idx = torch.topk(map_dist, k=min(k, int(map_dist.numel())), largest=True)
		ok_ids = ok.reshape(-1).nonzero(as_tuple=False).flatten()
		worst_ids = ok_ids[dist_idx]
		worst_mask = torch.zeros_like(ok.reshape(-1))
		worst_mask[worst_ids] = True
		worst_mask = worst_mask.view_as(ok)
		worst_count = int(worst_mask.sum().detach().cpu())
		_write_obj_lines(
			out / "map_ext_to_model_worst_1pct.obj",
			fixture.ext_xyz[worst_mask],
			model_pos[worst_mask],
			label="global_map_ext_to_model_worst_1pct",
		)
	else:
		empty = fixture.model_xyz.new_empty(0, 3)
		_write_obj_lines(
			out / "map_ext_to_model_worst_1pct.obj",
			empty,
			empty,
			label="global_map_ext_to_model_worst_1pct",
		)
	_write_obj_points(out / "map_valid_ext_points.obj", fixture.ext_xyz, ok, label="global_map_valid_ext_points")
	_write_json(
		out / "meta.json",
		{
				**meta,
				"map_ext_to_model": "map_ext_to_model.obj",
				"map_ext_to_model_worst_1pct": "map_ext_to_model_worst_1pct.obj",
				"valid_vectors": int(ok.sum().detach().cpu()),
				"worst_1pct_vectors": worst_count,
				**_fixture_mapping_error(uv, fixture),
			},
		)


def _level_coords(ext_shape: torch.Size | tuple[int, int], level: int, uv: torch.Tensor) -> torch.Tensor:
	H, W = _map_init_dyadic_level_shape(int(ext_shape[0]), int(ext_shape[1]), int(level))
	stride = 1 << int(level)
	hh = (torch.arange(H, device=uv.device, dtype=uv.dtype) * float(stride)).view(H, 1).expand(H, W)
	ww = (torch.arange(W, device=uv.device, dtype=uv.dtype) * float(stride)).view(1, W).expand(H, W)
	return torch.stack([hh, ww], dim=-1)


_GLOBAL_PROGRESS_COLUMNS = (
	ProgressColumn("stg", "stg", "stage index"),
	ProgressColumn("it", "it", "stage step"),
	ProgressColumn("lr", "lr", "optimizer learning rate", min_width=7),
	ProgressColumn("lvl", "lvl", "minimum trained pyramid level"),
	ProgressColumn("loss", "loss", "objective", min_width=7),
	ProgressColumn("dist", "dst", "distance loss", min_width=7),
	ProgressColumn("vec", "vec", "vector-normal loss", min_width=7),
	ProgressColumn("norm", "nrm", "surface-normal loss", min_width=7),
	ProgressColumn("turn", "turn", "lifted z-heading loss", min_width=7),
	ProgressColumn("smooth", "smo", "uv smooth loss", min_width=7),
	ProgressColumn("bend", "bnd", "uv bend loss", min_width=7),
	ProgressColumn("jac", "jac", "jacobian loss", min_width=7),
	ProgressColumn("metric_smooth", "met", "model metric loss", min_width=7),
	ProgressColumn("area_smooth", "ar", "external physical area loss", min_width=7),
	ProgressColumn("prior", "pri", "dense prior loss", min_width=7),
	ProgressColumn("station", "stat", "station loss", min_width=7),
	ProgressColumn("it_s", "it/s", "optimizer it/s", min_width=6),
	ProgressColumn("avgd", "avgd", "avg fixture model quad distance", min_width=7),
	ProgressColumn("maxd", "maxd", "max fixture model quad distance", min_width=7),
	ProgressColumn("smp", "smp", "objective samples", min_width=6),
	ProgressColumn("sample_bad", "bad", "objective samples rejected by limits or validity", min_width=6),
	ProgressColumn("uv_bad", "uvb", "active quads with non-finite UV corners", min_width=5),
	ProgressColumn("model_bad", "mbd", "active quads rejected by model/sample checks", min_width=5),
)


def _global_progress_widths(cfg: GlobalMapConfig) -> dict[str, int]:
	values: dict[str, str] = {}
	stage_count = max(1, len(cfg.stages))
	max_steps = max((int(stage.steps) for stage in cfg.stages), default=1)
	values["stg"] = str(stage_count - 1)
	values["it"] = f"grow-r1000000:{max_steps}/{max_steps}"
	values["lr"] = "-1.0e+99"
	values["lvl"] = str(max((int(stage.min_scaledown) for stage in cfg.stages), default=0))
	values["loss"] = "-1.0e+99"
	for key in ("dist", "vec", "norm", "turn", "smooth", "bend", "jac", "metric_smooth", "area_smooth", "prior", "station"):
		values[key] = "-1.0e+99"
	values["it_s"] = "-1.0e+99"
	values["avgd"] = "-1.0e+99"
	values["maxd"] = "-1.0e+99"
	values["smp"] = "1000000"
	values["sample_bad"] = "1000000"
	values["uv_bad"] = "1000000"
	values["model_bad"] = "1000000"
	return progress_widths(_GLOBAL_PROGRESS_COLUMNS, values)


def _global_term_value(terms: dict[str, torch.Tensor], key: str) -> float:
	v = terms.get(key)
	if v is None:
		return 0.0
	return float(v.detach().cpu())


def _print_global_progress(
	*,
	row_idx: int,
	widths: dict[str, int],
	stage_idx: int,
	iter_label: str,
	lr: float | None,
	level: int,
	loss: float,
	terms: dict[str, torch.Tensor],
	it_s: float | None,
	err: dict[str, float],
) -> None:
	values = {
		"stg": str(int(stage_idx)),
		"it": str(iter_label),
		"lr": format_progress_value(float(lr)) if lr is not None else "",
		"lvl": str(int(level)),
		"loss": format_progress_value(float(loss)),
		"dist": format_progress_value(_global_term_value(terms, "dist")),
		"vec": format_progress_value(_global_term_value(terms, "vec")),
		"norm": format_progress_value(_global_term_value(terms, "norm")),
		"turn": format_progress_value(_global_term_value(terms, "turn")),
		"smooth": format_progress_value(_global_term_value(terms, "smooth")),
		"bend": format_progress_value(_global_term_value(terms, "bend")),
		"jac": format_progress_value(_global_term_value(terms, "jac")),
		"metric_smooth": format_progress_value(_global_term_value(terms, "metric_smooth")),
		"area_smooth": format_progress_value(_global_term_value(terms, "area_smooth")),
		"prior": format_progress_value(_global_term_value(terms, "prior")),
		"station": format_progress_value(_global_term_value(terms, "station")),
		"it_s": format_progress_value(float(it_s)) if it_s is not None else "",
		"avgd": format_progress_value(float(err["avg_model_quad_distance"])),
		"maxd": format_progress_value(float(err["max_model_quad_distance"])),
		"smp": str(int(_global_term_value(terms, "samples"))),
		"sample_bad": str(int(_global_term_value(terms, "sample_bad"))),
		"uv_bad": str(int(_global_term_value(terms, "uv_bad"))),
		"model_bad": str(int(_global_term_value(terms, "model_bad"))),
	}
	if int(row_idx) % 20 == 0:
		_print_progress_legend_once(
			prefix="[snap_surf.map_global]",
			items=[(col.label, col.description) for col in _GLOBAL_PROGRESS_COLUMNS],
		)
		print(progress_header(_GLOBAL_PROGRESS_COLUMNS, widths), flush=True)
	print(progress_row(_GLOBAL_PROGRESS_COLUMNS, widths, values), flush=True)


def _global_progress_state(
	*,
	uv: torch.Tensor,
	fixture: MapFixture,
	cfg: SnapSurfConfig,
	level: int = 0,
	seed_hw: torch.Tensor,
	station_target: torch.Tensor,
	w_station: float,
	z_lift: dict[str, Any] | None | object = _NO_Z_LIFT,
	active_quad: torch.Tensor | None = None,
) -> tuple[float, dict[str, torch.Tensor], dict[str, float]]:
	loss, terms = _objective_for_uv(
		uv=uv,
		fixture=fixture,
		cfg=cfg,
		level=int(level),
		z_lift=z_lift,
		active_quad=active_quad,
	)
	station_raw = loss.new_zeros(())
	if float(w_station) > 0.0:
		station_raw = _station_loss(uv, _level_seed_hw(seed_hw, int(level)), station_target)
		loss = loss + float(w_station) * station_raw
	progress_terms = dict(terms)
	progress_terms["station"] = station_raw.detach()
	err = _fixture_mapping_error(uv.detach(), fixture)
	return float(loss.detach().cpu()), progress_terms, err


def _nan_reference_uv(ext_shape: tuple[int, int], *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
	return torch.full((int(ext_shape[0]), int(ext_shape[1]), 2), float("nan"), device=device, dtype=dtype)


def _fixture_from_live_tensors(
	*,
	model_xyz: torch.Tensor,
	model_normals: torch.Tensor,
	model_valid: torch.Tensor,
	ext_xyz: torch.Tensor,
	ext_valid: torch.Tensor,
	ext_normals: torch.Tensor,
	ext_quad_valid: torch.Tensor,
	seed_xyz: tuple[float, float, float] | None,
	sign: int = 1,
) -> MapFixture:
	device = model_xyz.device
	dtype = model_xyz.dtype
	metadata: dict[str, Any] = {
		"model_depth": 0,
		"sign": int(sign),
	}
	if seed_xyz is not None:
		seed = torch.tensor(seed_xyz, device=device, dtype=dtype)
		ext_quad, _ext_point, _ext_dist = _closest_external_seed_surface(
			seed=seed,
			ext_xyz=ext_xyz,
			ext_valid=ext_valid.bool(),
			ext_quad_valid=ext_quad_valid.bool(),
		)
		model_quad, _model_dist = _closest_model_surface_quad(
			point=seed,
			model_xyz=model_xyz,
			model_valid=model_valid.bool(),
		)
		if ext_quad is not None:
			metadata["seed_ext_sample_hw"] = [int(ext_quad[0]), int(ext_quad[1])]
		if model_quad is not None:
			metadata["seed_model_quad"] = [int(model_quad[0]), int(model_quad[1]), int(model_quad[2])]
		metadata["seed_xyz"] = [float(v) for v in seed_xyz]
	return MapFixture(
		root=Path("."),
		metadata=metadata,
		ext_xyz=ext_xyz.detach(),
		ext_valid=ext_valid.detach().bool(),
		ext_quad_valid=ext_quad_valid.detach().bool(),
		ext_normals=ext_normals.detach(),
		model_xyz=model_xyz.detach(),
		model_valid=model_valid.detach().bool(),
		model_normals=model_normals.detach(),
		reference_uv=_nan_reference_uv(tuple(int(v) for v in ext_xyz.shape[:2]), device=device, dtype=dtype),
		reference_active_quad=torch.zeros(
			max(0, int(ext_xyz.shape[0]) - 1),
			max(0, int(ext_xyz.shape[1]) - 1),
			device=device,
			dtype=torch.bool,
		),
		reference_blocked_quad=torch.zeros(
			max(0, int(ext_xyz.shape[0]) - 1),
			max(0, int(ext_xyz.shape[1]) - 1),
			device=device,
			dtype=torch.bool,
		),
	)


def _map_fixture_export_spec(stage: GlobalMapStageConfig) -> dict[str, Any] | None:
	raw = stage.args.get("export_fixture", stage.args.get("fixture_export", None))
	if raw is None:
		raw_dir = stage.args.get("fixture_export_dir", stage.args.get("map_fixture_export_dir", None))
		if raw_dir in (None, "", False):
			return None
		raw = {"dir": raw_dir}
	if raw in (False, None, ""):
		return None
	if raw is True:
		raw = {"dir": stage.args.get("fixture_export_dir", "map_fixture")}
	if isinstance(raw, str):
		raw = {"dir": raw}
	if not isinstance(raw, dict):
		raise ValueError("map fixture export config must be an object, string, bool, or null")
	out_dir = raw.get("dir", raw.get("out_dir", raw.get("path", stage.args.get("fixture_export_dir", None))))
	if out_dir in (None, "", False):
		raise ValueError("map fixture export requires 'dir' or fixture_export_dir")
	once = _truthy_arg(raw.get("once", stage.args.get("fixture_export_once", True)))
	export_objs = _truthy_arg(raw.get("objs", raw.get("export_objs", stage.args.get("fixture_export_objs", False))))
	write_geometry = _truthy_arg(raw.get("geometry", raw.get("write_geometry", True)))
	return {
		"dir": str(out_dir),
		"once": bool(once),
		"objs": bool(export_objs),
		"geometry": bool(write_geometry),
		"tag": str(raw.get("tag", raw.get("name", ""))),
	}


def _surface_state_for_fixture_export(
	*,
	fixture: MapFixture,
	uv: torch.Tensor,
	active_quad: torch.Tensor,
	sign: int,
) -> _SurfaceState:
	state = _SurfaceState()
	state.ensure(
		model_shape=tuple(int(v) for v in fixture.model_xyz.shape[:3]),
		ext_shape=tuple(int(v) for v in fixture.ext_xyz.shape[:2]),
		device=fixture.model_xyz.device,
		dtype=fixture.model_xyz.dtype,
	)
	mi = state.map_init
	mi.uv = uv.detach()
	mi.active_quad = active_quad.detach().bool()
	mi.blocked_quad = torch.zeros_like(mi.active_quad)
	mi.ext_pos = fixture.ext_xyz.detach()
	mi.ext_normals = fixture.ext_normals.detach()
	mi.ext_valid = fixture.ext_valid.detach().bool()
	mi.ext_quad_valid = fixture.ext_quad_valid.detach().bool()
	mi.model_depth = int(fixture.metadata.get("model_depth", 0) or 0)
	mi.sign = int(sign)
	seed_hw = fixture.metadata.get("seed_ext_sample_hw")
	if isinstance(seed_hw, (list, tuple)) and len(seed_hw) >= 2:
		mi.seed_ext_sample_hw = (int(seed_hw[0]), int(seed_hw[1]))
		state.ext_seed_hw = mi.seed_ext_sample_hw
	seed_model = fixture.metadata.get("seed_model_quad")
	if isinstance(seed_model, (list, tuple)) and len(seed_model) >= 3:
		mi.seed_model_quad = (int(seed_model[0]), int(seed_model[1]), int(seed_model[2]))
	mi.scale_level = 0
	mi.target_scale_level = 0
	mi.scale_strides = [1]
	mi.scale_levels_used = 1
	mi.done = True
	return state


class SelfMapRuntime:
	"""Persistent batched self-map optimizer for init-shell winding pairs."""

	def __init__(
		self,
		*,
		mode: str,
		direction: str,
		model_w_wraps: float | None = None,
		base: dict[str, Any] | None = None,
	) -> None:
		self.mode = normalize_self_map_init(mode)
		self.direction = str(direction).strip().lower()
		if self.direction not in SELF_MAP_DIRECTIONS:
			raise ValueError(f"invalid self-map direction '{direction}'")
		self.model_w_wraps = None if model_w_wraps is None else float(model_w_wraps)
		self.cfg_global = GlobalMapConfig(base=dict(base or {}), stages=())
		self.global_model: GlobalMapModel | None = None
		self.optimizer: torch.optim.Optimizer | None = None
		self.optimizer_key: tuple[tuple[str, ...], int, int, float, int] | None = None
		self.lr_autoscale: _LrAutoscaleState | None = None
		self.last: dict[str, float] = {}
		self.steps_run = 0
		self._snap_loss_mode_printed: set[tuple[str, float]] = set()

	def _ensure_model(self, model_xyz: torch.Tensor, base_cfg: SnapSurfConfig, stage: GlobalMapStageConfig) -> None:
		if self.global_model is not None:
			return
		D, H, W = int(model_xyz.shape[0]), int(model_xyz.shape[1]), int(model_xyz.shape[2])
		initial_uv = self.initial_uv(depth=D, height=H, width=W, device=model_xyz.device, dtype=model_xyz.dtype)
		levels = _max_supported_level(
			(H, W),
			max(int(stage.min_scaledown), int(base_cfg.map_init.scale_levels) - 1),
		) + 1
		self.global_model = GlobalMapModel(initial_uv.detach(), levels=levels, factor=2, preserve_batch=True)

	def initial_uv(
		self,
		*,
		depth: int,
		height: int,
		width: int,
		device: torch.device,
		dtype: torch.dtype,
	) -> torch.Tensor:
		return self_map_initial_uv(
			mode=self.mode,
			direction=self.direction,
			depth=int(depth),
			height=int(height),
			width=int(width),
			model_w_wraps=self.model_w_wraps,
			device=device,
			dtype=dtype,
		)

	def _uv(self, *, active_level: int = 0) -> torch.Tensor:
		if self.global_model is None:
			raise RuntimeError("self-map runtime is not initialized")
		return self.global_model(active_level=int(active_level))

	def _params_for_stage(self, stage: GlobalMapStageConfig) -> tuple[list[torch.nn.Parameter], int]:
		if self.global_model is None:
			raise RuntimeError("self-map runtime is not initialized")
		params: list[torch.nn.Parameter] = []
		level = 0
		if "map_uv_ms" in stage.params or "affine" in stage.params:
			level = min(max(0, int(stage.min_scaledown)), len(self.global_model.map_uv_ms) - 1)
			params.extend(list(self.global_model.map_uv_ms.parameters())[level:])
		return params, level

	def _stats_from_terms(self, loss: torch.Tensor, terms: dict[str, torch.Tensor]) -> dict[str, float]:
		stats = {
			"snaps_map_loss": float(loss.detach().cpu()),
			"snaps_map_runtime_steps": float(self.steps_run),
			"snaps_map_samples": float(_global_term_value(terms, "samples")),
		}
		for term_key, stat_key in (
			("dist", "snaps_map_dist"),
			("vec", "snaps_map_vec"),
			("norm", "snaps_map_norm"),
			("turn", "snaps_map_turn"),
			("turn_smp", "snaps_map_turn_smp"),
			("smooth", "snaps_map_smooth"),
			("bend", "snaps_map_bend"),
			("jac", "snaps_map_jac"),
			("metric_smooth", "snaps_map_metric_smooth"),
			("area_smooth", "snaps_map_area_smooth"),
			("prior", "snaps_map_prior"),
			("sample_total", "snaps_map_sample_total"),
			("sample_valid", "snaps_map_sample_valid"),
			("sample_bad", "snaps_map_sample_bad"),
			("uv_bad", "snaps_map_uvbad"),
			("model_bad", "snaps_map_model_bad"),
			("loss_finite", "snaps_map_loss_finite"),
		):
			stats[stat_key] = float(_global_term_value(terms, term_key))
		return stats

	def run_stage(
		self,
		*,
		stage: GlobalMapStageConfig,
		model_xyz: torch.Tensor,
		model_normals: torch.Tensor,
		model_valid: torch.Tensor,
		persistent_optimizer: bool = False,
		status_fn=None,
		cancel_fn=None,
		auto_stop_fn=None,
		**_unused,
	) -> dict[str, float]:
		base_cfg = snap_surf_config_from_global_config(self.cfg_global, stage)
		stage_cfg = _stage_loss_cfg(base_cfg, stage)
		self._ensure_model(model_xyz, base_cfg, stage)
		params, train_level = self._params_for_stage(stage)
		sample_level = _stage_objective_level(stage, train_level, tuple(int(v) for v in model_xyz.shape[1:3]))
		status_interval = max(0, int(stage.args.get("status_interval", stage.args.get("debug_print_interval", 100))))
		lr_warmup_steps = _lr_warmup_steps(stage.args)
		lr_autoscale_cfg = _lr_autoscale_config(stage.args)
		auto_loss_history: list[float] = []
		steps_completed = 0
		auto_stopped = False
		lr_autoscale: _LrAutoscaleState | None = None
		opt: torch.optim.Optimizer | None = None
		if params and int(stage.steps) > 0:
			key = (tuple(stage.params), int(train_level), int(sample_level), float(stage.lr), int(stage.min_scaledown))
			if (not persistent_optimizer) or self.optimizer is None or self.optimizer_key != key:
				self.optimizer = torch.optim.Adam(params, lr=float(stage.lr))
				_capture_optimizer_target_lrs(self.optimizer)
				self.optimizer_key = key
				self.lr_autoscale = _LrAutoscaleState(lr_autoscale_cfg) if lr_autoscale_cfg.enabled else None
			opt = self.optimizer
			assert opt is not None
			lr_autoscale = self.lr_autoscale if persistent_optimizer else (_LrAutoscaleState(lr_autoscale_cfg) if lr_autoscale_cfg.enabled else None)
			for step in range(int(stage.steps)):
				if cancel_fn is not None:
					cancel_fn()
				opt.zero_grad(set_to_none=True)
				uv = self._uv(active_level=int(sample_level))
				status_due = (
					status_fn is not None and (
						step == 0 or step == int(stage.steps) - 1 or (status_interval > 0 and (step % status_interval) == 0)
					)
				)
				active = self_map_active_quads(mode=self.mode, direction=self.direction, model_valid=model_valid, uv=uv)
				loss, terms = _self_map_objective_for_uv(
					uv=uv,
					mode=self.mode,
					direction=self.direction,
					model_xyz=model_xyz,
					model_normals=model_normals,
					model_valid=model_valid,
					cfg=stage_cfg,
					level=int(sample_level),
					need_stats=status_due,
					active_quad=active,
				)
				_apply_optimizer_lr_schedule(
					opt,
					step1=(self.steps_run + 1 if persistent_optimizer else step + 1),
					warmup_steps=lr_warmup_steps,
					autoscale=lr_autoscale,
					loss=loss,
				)
				if status_due:
					status_stats = self._stats_from_terms(loss, terms)
					status_stats["snaps_map_lr"] = _optimizer_lr_for_display(opt, float(stage.lr))
					status_stats.update(_lr_autoscale_stats("snaps_map", lr_autoscale))
					status_fn(step=step, total=int(stage.steps), stats=status_stats)
				loss.backward()
				opt.step()
				self.steps_run += 1
				steps_completed = step + 1
				if auto_stop_fn is not None and steps_completed > lr_warmup_steps:
					auto_loss_history.append(float(loss.detach().cpu()))
					auto_stopped = bool(auto_stop_fn(history=auto_loss_history, step=steps_completed - lr_warmup_steps))
				if auto_stopped:
					break
			if not persistent_optimizer:
				self.optimizer = None
				self.optimizer_key = None
				self.lr_autoscale = None
		with torch.no_grad():
			uv = self._uv(active_level=int(sample_level))
			active = self_map_active_quads(mode=self.mode, direction=self.direction, model_valid=model_valid, uv=uv)
			loss, terms = _self_map_objective_for_uv(
				uv=uv,
				mode=self.mode,
				direction=self.direction,
				model_xyz=model_xyz,
				model_normals=model_normals,
				model_valid=model_valid,
				cfg=stage_cfg,
				level=int(sample_level),
				need_stats=True,
				active_quad=active,
			)
		stats = self._stats_from_terms(loss, terms)
		stats["snaps_map_lr"] = _optimizer_lr_for_display(opt if params and int(stage.steps) > 0 else None, float(stage.lr))
		stats["snaps_map_stage_steps"] = float(steps_completed)
		stats["snaps_map_auto_stopped"] = 1.0 if auto_stopped else 0.0
		stats.update(_lr_autoscale_stats("snaps_map", lr_autoscale))
		self.last = stats
		return stats

	def _snap_inputs(
		self,
		model_xyz: torch.Tensor,
		model_normals: torch.Tensor,
		model_valid: torch.Tensor,
	) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
		self._ensure_model(model_xyz, snap_surf_config_from_global_config(self.cfg_global), GlobalMapStageConfig(params=("map_uv_ms",)))
		uv = self._uv().detach()
		src, dst = self_map_pair_depths(self.mode, self.direction, int(model_xyz.shape[0]))
		src_t = torch.tensor(src, device=model_xyz.device, dtype=torch.long)
		dst_t = torch.tensor(dst, device=model_xyz.device, dtype=torch.long)
		source_xyz = model_xyz[src_t].detach()
		source_normals = model_normals[src_t].detach()
		source_valid = model_valid[src_t].detach().bool()
		active = self_map_active_quads(mode=self.mode, direction=self.direction, model_valid=model_valid, uv=uv)
		active_vertex = torch.zeros_like(source_valid, dtype=torch.bool)
		if active.numel():
			active_vertex[:, :-1, :-1] |= active
			active_vertex[:, 1:, :-1] |= active
			active_vertex[:, :-1, 1:] |= active
			active_vertex[:, 1:, 1:] |= active
		return uv, dst_t, source_xyz, source_normals, source_valid & active_vertex, active

	def snap_loss_prefetch_items(
		self,
		*,
		model_xyz: torch.Tensor,
		model_normals: torch.Tensor,
		model_valid: torch.Tensor,
		offset: float = 1.0,
		data=None,
		strip_samples: int = 5,
		**_unused,
	) -> dict[str, torch.Tensor]:
		if data is None:
			return {}
		uv, dst_t, source_xyz, _source_normals, source_valid, _active = self._snap_inputs(model_xyz, model_normals, model_valid)
		target_depth = dst_t.to(dtype=uv.dtype).view(-1, 1, 1, 1).expand(*uv.shape[:-1], 1)
		coords = torch.cat([target_depth, uv], dim=-1)
		target_ok = _quad_valid_at_coords(model_valid.bool(), coords, tuple(int(v) for v in model_valid.shape))
		valid = source_valid & target_ok & torch.isfinite(coords).all(dim=-1) & torch.isfinite(source_xyz).all(dim=-1)
		if not bool(valid.any().detach().cpu()):
			return {}
		safe_coords = torch.where(torch.isfinite(coords), coords, torch.zeros_like(coords))
		model_pos = _sample_surface_grid(model_xyz, safe_coords)
		sample_valid = valid & torch.isfinite(model_pos).all(dim=-1)
		if not bool(sample_valid.any().detach().cpu()):
			return {}
		t = torch.linspace(0.0, 1.0, max(2, int(strip_samples)), device=model_xyz.device, dtype=model_xyz.dtype)
		source_valid_xyz = source_xyz[sample_valid]
		target_valid_xyz = model_pos.detach()[sample_valid]
		strip = source_valid_xyz.unsqueeze(-2) + t.view(1, -1, 1) * (target_valid_xyz - source_valid_xyz).unsqueeze(-2)
		return {"grad_mag": strip.reshape(1, 1, -1, 3).contiguous()}

	def snap_loss(
		self,
		*,
		model_xyz: torch.Tensor,
		model_normals: torch.Tensor,
		model_valid: torch.Tensor,
		offset: float = 1.0,
		data=None,
		strip_samples: int = 5,
		**_unused,
	) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], dict[str, float]]:
		offset_f = float(offset)
		signed_offset = self_map_signed_offset(self.mode, self.direction, offset_f)
		mode_key = ("self_winding", offset_f, signed_offset)
		if mode_key not in self._snap_loss_mode_printed:
			print(
				f"[snap_surf.map_global] self snap loss mode={self.mode}/{self.direction} "
				f"offset={offset_f:.6g} signed_offset={signed_offset:.6g}",
				flush=True,
			)
			self._snap_loss_mode_printed.add(mode_key)
		uv, dst_t, source_xyz, _source_normals, source_valid, _active = self._snap_inputs(model_xyz, model_normals, model_valid)
		target_depth = dst_t.to(dtype=uv.dtype).view(-1, 1, 1, 1).expand(*uv.shape[:-1], 1)
		coords = torch.cat([target_depth, uv], dim=-1)
		target_ok = _quad_valid_at_coords(model_valid.bool(), coords, tuple(int(v) for v in model_valid.shape))
		valid = source_valid & target_ok & torch.isfinite(coords).all(dim=-1) & torch.isfinite(source_xyz).all(dim=-1)
		safe_coords = torch.where(torch.isfinite(coords), coords, torch.zeros_like(coords))
		model_pos = _sample_surface_grid(model_xyz, safe_coords)
		model_n = torch.nn.functional.normalize(_sample_surface_grid(model_normals.detach(), safe_coords), dim=-1, eps=1.0e-8)
		valid = valid & torch.isfinite(model_pos).all(dim=-1) & torch.isfinite(model_n).all(dim=-1)
		z = model_xyz.sum() * 0.0
		lm = torch.zeros(model_xyz.shape[:3], device=model_xyz.device, dtype=model_xyz.dtype).unsqueeze(1)
		mask = torch.zeros_like(lm)
		if not bool(valid.any().detach().cpu()):
			return z, (lm,), (mask,), {"snaps_map_snap": 0.0, "snaps_map_snap_abs": 0.0, "snaps_map_snap_max": 0.0, "snaps_map_snap_samples": 0.0}
		signed_vox = ((model_pos - source_xyz.detach()) * model_n.detach()).sum(dim=-1)
		if data is None:
			residual = signed_vox - signed_vox.new_tensor(signed_offset)
			valid_final = valid
		else:
			strip_samples_i = max(2, int(strip_samples))
			sample_valid = valid & torch.isfinite(model_pos).all(dim=-1) & torch.isfinite(model_n).all(dim=-1)
			with torch.no_grad():
				t = torch.linspace(0.0, 1.0, strip_samples_i, device=model_xyz.device, dtype=model_xyz.dtype)
				origin = torch.tensor(data.origin_fullres, device=model_xyz.device, dtype=model_xyz.dtype)
				spacing = torch.tensor(data._spacing_for("grad_mag"), device=model_xyz.device, dtype=model_xyz.dtype)
				sentinel = origin - 64.0 * spacing
				source_safe = torch.where(sample_valid.unsqueeze(-1), source_xyz.detach(), sentinel.view(1, 1, 1, 3))
				target_safe = torch.where(sample_valid.unsqueeze(-1), model_pos.detach(), sentinel.view(1, 1, 1, 3))
				diff = target_safe - source_safe
				strip = source_safe.unsqueeze(-2) + t.view(1, 1, 1, strip_samples_i, 1) * diff.unsqueeze(-2)
				N, H, W = int(source_xyz.shape[0]), int(source_xyz.shape[1]), int(source_xyz.shape[2])
				sampled = data.grid_sample_fullres(strip.reshape(1, N * H, W * strip_samples_i, 3), channels={"grad_mag"})
				if sampled.grad_mag is None:
					raise RuntimeError("self snap_surf_map offset mode requires grad_mag samples")
				mag = sampled.grad_mag.detach().squeeze(0).squeeze(0).reshape(N, H, W, strip_samples_i)
				strip_valid = (mag > 0.0).all(dim=-1)
				mean_grad = mag.mean(dim=-1)
				strip_len = diff.square().sum(dim=-1).sqrt()
				int_sign = torch.sign(((model_pos.detach() - source_xyz.detach()) * model_n.detach()).sum(dim=-1))
				signed_windings = int_sign * strip_len * mean_grad
				winding_err = signed_windings - signed_vox.new_tensor(signed_offset)
			valid_final = sample_valid & strip_valid
			normal_residual = signed_vox * mean_grad
			residual = normal_residual + (winding_err - normal_residual).detach()
		vals = residual[valid_final]
		if vals.numel() == 0:
			return z, (lm,), (mask,), {"snaps_map_snap": 0.0, "snaps_map_snap_abs": 0.0, "snaps_map_snap_max": 0.0, "snaps_map_snap_samples": 0.0}
		loss = vals.square().mean()
		stats = {
			"snaps_map_snap": float(loss.detach().cpu()),
			"snaps_map_snap_abs": float(vals.detach().abs().mean().cpu()),
			"snaps_map_snap_max": float(vals.detach().abs().max().cpu()),
			"snaps_map_snap_samples": float(int(vals.numel())),
		}
		return loss, (lm,), (mask,), stats


class BoundarySelfMapRuntime(SelfMapRuntime):
	"""Self-map runtime for grow boundaries with one fixed and one trainable surface."""

	def __init__(
		self,
		*,
		mode: str,
		direction: str,
		fixed_xyz: torch.Tensor,
		fixed_normals: torch.Tensor,
		fixed_valid: torch.Tensor,
		model_w_wraps: float | None = None,
		base: dict[str, Any] | None = None,
	) -> None:
		super().__init__(
			mode=mode,
			direction=direction,
			model_w_wraps=model_w_wraps,
			base=base,
		)
		if self.mode != "multi_wrap_d":
			raise ValueError("boundary self-map grow currently requires self-map-init=multi_wrap_d")
		self.fixed_xyz = fixed_xyz.detach()
		self.fixed_normals = fixed_normals.detach()
		self.fixed_valid = fixed_valid.detach().bool()

	def _fixed_tensors_for(self, model_xyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		device = model_xyz.device
		dtype = model_xyz.dtype
		return (
			self.fixed_xyz.to(device=device, dtype=dtype),
			self.fixed_normals.to(device=device, dtype=dtype),
			self.fixed_valid.to(device=device).bool(),
		)

	def _ensure_model(self, model_xyz: torch.Tensor, base_cfg: SnapSurfConfig, stage: GlobalMapStageConfig) -> None:
		if self.global_model is not None:
			return
		if int(model_xyz.shape[0]) != 1:
			raise ValueError(f"boundary self-map grow expects temporary model depth 1, got {int(model_xyz.shape[0])}")
		H, W = int(model_xyz.shape[1]), int(model_xyz.shape[2])
		initial_uv = self.initial_uv(depth=2, height=H, width=W, device=model_xyz.device, dtype=model_xyz.dtype)
		levels = _max_supported_level(
			(H, W),
			max(int(stage.min_scaledown), int(base_cfg.map_init.scale_levels) - 1),
		) + 1
		self.global_model = GlobalMapModel(initial_uv.detach(), levels=levels, factor=2, preserve_batch=True)

	def _stacked_boundary(
		self,
		*,
		model_xyz: torch.Tensor,
		model_normals: torch.Tensor,
		model_valid: torch.Tensor,
	) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		if int(model_xyz.shape[0]) != 1:
			raise ValueError(f"boundary self-map grow expects temporary model depth 1, got {int(model_xyz.shape[0])}")
		fixed_xyz, fixed_normals, fixed_valid = self._fixed_tensors_for(model_xyz)
		return (
			torch.stack([fixed_xyz, model_xyz[0]], dim=0),
			torch.stack([fixed_normals, model_normals[0]], dim=0),
			torch.stack([fixed_valid, model_valid[0].bool()], dim=0),
		)

	def run_stage(
		self,
		*,
		stage: GlobalMapStageConfig,
		model_xyz: torch.Tensor,
		model_normals: torch.Tensor,
		model_valid: torch.Tensor,
		persistent_optimizer: bool = False,
		status_fn=None,
		cancel_fn=None,
		auto_stop_fn=None,
		**_unused,
	) -> dict[str, float]:
		base_cfg = snap_surf_config_from_global_config(self.cfg_global, stage)
		stage_cfg = _stage_loss_cfg(base_cfg, stage)
		self._ensure_model(model_xyz, base_cfg, stage)
		params, train_level = self._params_for_stage(stage)
		sample_level = _stage_objective_level(stage, train_level, tuple(int(v) for v in model_xyz.shape[1:3]))
		status_interval = max(0, int(stage.args.get("status_interval", stage.args.get("debug_print_interval", 100))))
		lr_warmup_steps = _lr_warmup_steps(stage.args)
		lr_autoscale_cfg = _lr_autoscale_config(stage.args)
		auto_loss_history: list[float] = []
		steps_completed = 0
		auto_stopped = False
		lr_autoscale: _LrAutoscaleState | None = None
		opt: torch.optim.Optimizer | None = None
		stacked_xyz, stacked_normals, stacked_valid = self._stacked_boundary(
			model_xyz=model_xyz,
			model_normals=model_normals,
			model_valid=model_valid,
		)
		if params and int(stage.steps) > 0:
			key = (tuple(stage.params), int(train_level), int(sample_level), float(stage.lr), int(stage.min_scaledown))
			if (not persistent_optimizer) or self.optimizer is None or self.optimizer_key != key:
				self.optimizer = torch.optim.Adam(params, lr=float(stage.lr))
				_capture_optimizer_target_lrs(self.optimizer)
				self.optimizer_key = key
				self.lr_autoscale = _LrAutoscaleState(lr_autoscale_cfg) if lr_autoscale_cfg.enabled else None
			opt = self.optimizer
			assert opt is not None
			lr_autoscale = self.lr_autoscale if persistent_optimizer else (_LrAutoscaleState(lr_autoscale_cfg) if lr_autoscale_cfg.enabled else None)
			for step in range(int(stage.steps)):
				if cancel_fn is not None:
					cancel_fn()
				opt.zero_grad(set_to_none=True)
				uv = self._uv(active_level=int(sample_level))
				status_due = (
					status_fn is not None and (
						step == 0 or step == int(stage.steps) - 1 or (status_interval > 0 and (step % status_interval) == 0)
					)
				)
				active = self_map_active_quads(mode=self.mode, direction=self.direction, model_valid=stacked_valid, uv=uv)
				loss, terms = _self_map_objective_for_uv(
					uv=uv,
					mode=self.mode,
					direction=self.direction,
					model_xyz=stacked_xyz,
					model_normals=stacked_normals,
					model_valid=stacked_valid,
					cfg=stage_cfg,
					level=int(sample_level),
					need_stats=status_due,
					active_quad=active,
				)
				_apply_optimizer_lr_schedule(
					opt,
					step1=(self.steps_run + 1 if persistent_optimizer else step + 1),
					warmup_steps=lr_warmup_steps,
					autoscale=lr_autoscale,
					loss=loss,
				)
				if status_due:
					status_stats = self._stats_from_terms(loss, terms)
					status_stats["snaps_map_lr"] = _optimizer_lr_for_display(opt, float(stage.lr))
					status_stats.update(_lr_autoscale_stats("snaps_map", lr_autoscale))
					status_fn(step=step, total=int(stage.steps), stats=status_stats)
				loss.backward()
				opt.step()
				self.steps_run += 1
				steps_completed = step + 1
				if auto_stop_fn is not None and steps_completed > lr_warmup_steps:
					auto_loss_history.append(float(loss.detach().cpu()))
					auto_stopped = bool(auto_stop_fn(history=auto_loss_history, step=steps_completed - lr_warmup_steps))
				if auto_stopped:
					break
			if not persistent_optimizer:
				self.optimizer = None
				self.optimizer_key = None
				self.lr_autoscale = None
		with torch.no_grad():
			uv = self._uv(active_level=int(sample_level))
			active = self_map_active_quads(mode=self.mode, direction=self.direction, model_valid=stacked_valid, uv=uv)
			loss, terms = _self_map_objective_for_uv(
				uv=uv,
				mode=self.mode,
				direction=self.direction,
				model_xyz=stacked_xyz,
				model_normals=stacked_normals,
				model_valid=stacked_valid,
				cfg=stage_cfg,
				level=int(sample_level),
				need_stats=True,
				active_quad=active,
			)
		stats = self._stats_from_terms(loss, terms)
		stats["snaps_map_lr"] = _optimizer_lr_for_display(opt if params and int(stage.steps) > 0 else None, float(stage.lr))
		stats["snaps_map_stage_steps"] = float(steps_completed)
		stats["snaps_map_auto_stopped"] = 1.0 if auto_stopped else 0.0
		stats.update(_lr_autoscale_stats("snaps_map", lr_autoscale))
		self.last = stats
		return stats

	def _snap_inputs(
		self,
		model_xyz: torch.Tensor,
		model_normals: torch.Tensor,
		model_valid: torch.Tensor,
	) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]:
		self._ensure_model(model_xyz, snap_surf_config_from_global_config(self.cfg_global), GlobalMapStageConfig(params=("map_uv_ms",)))
		uv = self._uv().detach()
		fixed_xyz, fixed_normals, fixed_valid = self._fixed_tensors_for(model_xyz)
		if self.direction == "out":
			source_xyz = fixed_xyz.detach().unsqueeze(0)
			source_valid = fixed_valid.detach().bool().unsqueeze(0)
			target_xyz = model_xyz[0].unsqueeze(0)
			target_normals = model_normals[0].unsqueeze(0)
			target_valid = model_valid[0].bool().unsqueeze(0)
			source_trainable = False
		else:
			source_xyz = model_xyz[0].unsqueeze(0)
			source_valid = model_valid[0].bool().unsqueeze(0)
			target_xyz = fixed_xyz.detach().unsqueeze(0)
			target_normals = fixed_normals.detach().unsqueeze(0)
			target_valid = fixed_valid.detach().bool().unsqueeze(0)
			source_trainable = True
		stacked_valid = torch.cat([fixed_valid.unsqueeze(0), model_valid[0].bool().unsqueeze(0)], dim=0)
		active = self_map_active_quads(mode=self.mode, direction=self.direction, model_valid=stacked_valid, uv=uv)
		active_vertex = torch.zeros_like(source_valid, dtype=torch.bool)
		if active.numel():
			active_vertex[:, :-1, :-1] |= active
			active_vertex[:, 1:, :-1] |= active
			active_vertex[:, :-1, 1:] |= active
			active_vertex[:, 1:, 1:] |= active
		return uv, target_xyz, target_normals, target_valid, source_xyz, source_valid & active_vertex, active, source_trainable

	def snap_loss_prefetch_items(
		self,
		*,
		model_xyz: torch.Tensor,
		model_normals: torch.Tensor,
		model_valid: torch.Tensor,
		offset: float = 1.0,
		data=None,
		strip_samples: int = 5,
		**_unused,
	) -> dict[str, torch.Tensor]:
		if data is None:
			return {}
		uv, target_xyz, _target_normals, target_valid, source_xyz, source_valid, _active, _source_trainable = self._snap_inputs(
			model_xyz,
			model_normals,
			model_valid,
		)
		depth = torch.zeros((*uv.shape[:-1], 1), device=uv.device, dtype=uv.dtype)
		coords = torch.cat([depth, uv], dim=-1)
		target_ok = _quad_valid_at_coords(target_valid.bool(), coords, tuple(int(v) for v in target_valid.shape))
		valid = source_valid & target_ok & torch.isfinite(coords).all(dim=-1) & torch.isfinite(source_xyz).all(dim=-1)
		if not bool(valid.any().detach().cpu()):
			return {}
		safe_coords = torch.where(torch.isfinite(coords), coords, torch.zeros_like(coords))
		target_pos = _sample_surface_grid(target_xyz, safe_coords)
		sample_valid = valid & torch.isfinite(target_pos).all(dim=-1)
		if not bool(sample_valid.any().detach().cpu()):
			return {}
		t = torch.linspace(0.0, 1.0, max(2, int(strip_samples)), device=model_xyz.device, dtype=model_xyz.dtype)
		source_valid_xyz = source_xyz.detach()[sample_valid]
		target_valid_xyz = target_pos.detach()[sample_valid]
		strip = source_valid_xyz.unsqueeze(-2) + t.view(1, -1, 1) * (target_valid_xyz - source_valid_xyz).unsqueeze(-2)
		return {"grad_mag": strip.reshape(1, 1, -1, 3).contiguous()}

	def snap_loss(
		self,
		*,
		model_xyz: torch.Tensor,
		model_normals: torch.Tensor,
		model_valid: torch.Tensor,
		offset: float = 1.0,
		data=None,
		strip_samples: int = 5,
		**_unused,
	) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], dict[str, float]]:
		offset_f = float(offset)
		signed_offset = self_map_signed_offset(self.mode, self.direction, offset_f)
		mode_key = ("boundary_self_winding", offset_f, signed_offset)
		if mode_key not in self._snap_loss_mode_printed:
			print(
				f"[snap_surf.map_global] boundary self snap loss mode={self.mode}/{self.direction} "
				f"offset={offset_f:.6g} signed_offset={signed_offset:.6g}",
				flush=True,
			)
			self._snap_loss_mode_printed.add(mode_key)
		uv, target_xyz, target_normals, target_valid, source_xyz, source_valid, _active, source_trainable = self._snap_inputs(
			model_xyz,
			model_normals,
			model_valid,
		)
		depth = torch.zeros((*uv.shape[:-1], 1), device=uv.device, dtype=uv.dtype)
		coords = torch.cat([depth, uv], dim=-1)
		target_ok = _quad_valid_at_coords(target_valid.bool(), coords, tuple(int(v) for v in target_valid.shape))
		valid = source_valid & target_ok & torch.isfinite(coords).all(dim=-1) & torch.isfinite(source_xyz).all(dim=-1)
		safe_coords = torch.where(torch.isfinite(coords), coords, torch.zeros_like(coords))
		target_pos = _sample_surface_grid(target_xyz, safe_coords)
		target_n = torch.nn.functional.normalize(_sample_surface_grid(target_normals.detach(), safe_coords), dim=-1, eps=1.0e-8)
		valid = valid & torch.isfinite(target_pos).all(dim=-1) & torch.isfinite(target_n).all(dim=-1)
		z = model_xyz.sum() * 0.0
		lm = torch.zeros(model_xyz.shape[:3], device=model_xyz.device, dtype=model_xyz.dtype).unsqueeze(1)
		mask = torch.zeros_like(lm)
		if not bool(valid.any().detach().cpu()):
			return z, (lm,), (mask,), {"snaps_map_snap": 0.0, "snaps_map_snap_abs": 0.0, "snaps_map_snap_max": 0.0, "snaps_map_snap_samples": 0.0}
		source_for_residual = source_xyz if source_trainable else source_xyz.detach()
		target_for_residual = target_pos.detach() if source_trainable else target_pos
		signed_vox = ((target_for_residual - source_for_residual) * target_n.detach()).sum(dim=-1)
		if data is None:
			residual = signed_vox - signed_vox.new_tensor(signed_offset)
			valid_final = valid
		else:
			strip_samples_i = max(2, int(strip_samples))
			sample_valid = valid & torch.isfinite(target_pos).all(dim=-1) & torch.isfinite(target_n).all(dim=-1)
			with torch.no_grad():
				t = torch.linspace(0.0, 1.0, strip_samples_i, device=model_xyz.device, dtype=model_xyz.dtype)
				origin = torch.tensor(data.origin_fullres, device=model_xyz.device, dtype=model_xyz.dtype)
				spacing = torch.tensor(data._spacing_for("grad_mag"), device=model_xyz.device, dtype=model_xyz.dtype)
				sentinel = origin - 64.0 * spacing
				source_safe = torch.where(sample_valid.unsqueeze(-1), source_xyz.detach(), sentinel.view(1, 1, 1, 3))
				target_safe = torch.where(sample_valid.unsqueeze(-1), target_pos.detach(), sentinel.view(1, 1, 1, 3))
				diff = target_safe - source_safe
				strip = source_safe.unsqueeze(-2) + t.view(1, 1, 1, strip_samples_i, 1) * diff.unsqueeze(-2)
				N, H, W = int(source_xyz.shape[0]), int(source_xyz.shape[1]), int(source_xyz.shape[2])
				sampled = data.grid_sample_fullres(strip.reshape(1, N * H, W * strip_samples_i, 3), channels={"grad_mag"})
				if sampled.grad_mag is None:
					raise RuntimeError("boundary self snap_surf_map offset mode requires grad_mag samples")
				mag = sampled.grad_mag.detach().squeeze(0).squeeze(0).reshape(N, H, W, strip_samples_i)
				strip_valid = (mag > 0.0).all(dim=-1)
				mean_grad = mag.mean(dim=-1)
				strip_len = diff.square().sum(dim=-1).sqrt()
				int_sign = torch.sign(((target_pos.detach() - source_xyz.detach()) * target_n.detach()).sum(dim=-1))
				signed_windings = int_sign * strip_len * mean_grad
				winding_err = signed_windings - signed_vox.new_tensor(signed_offset)
			valid_final = sample_valid & strip_valid
			normal_residual = signed_vox * mean_grad
			residual = normal_residual + (winding_err - normal_residual).detach()
		vals = residual[valid_final]
		if vals.numel() == 0:
			return z, (lm,), (mask,), {"snaps_map_snap": 0.0, "snaps_map_snap_abs": 0.0, "snaps_map_snap_max": 0.0, "snaps_map_snap_samples": 0.0}
		loss = vals.square().mean()
		stats = {
			"snaps_map_snap": float(loss.detach().cpu()),
			"snaps_map_snap_abs": float(vals.detach().abs().mean().cpu()),
			"snaps_map_snap_max": float(vals.detach().abs().max().cpu()),
			"snaps_map_snap_samples": float(int(vals.numel())),
		}
		return loss, (lm,), (mask,), stats


class GlobalMapRuntime:
	"""Persistent global rectangular map optimizer for live snap-surf tensors."""

	def __init__(self, *, base: dict[str, Any] | None = None, seed_xyz: tuple[float, float, float] | None = None) -> None:
		self.cfg_global = GlobalMapConfig(base=dict(base or {}), stages=())
		self.seed_xyz = seed_xyz
		self.affine: AffineMapModel | None = None
		self.global_model: GlobalMapModel | None = None
		self.optimizer: torch.optim.Optimizer | None = None
		self.optimizer_key: tuple[tuple[str, ...], int, float, int] | None = None
		self.lr_autoscale: _LrAutoscaleState | None = None
		self.station_target: torch.Tensor | None = None
		self.last: dict[str, float] = {}
		self.steps_run = 0
		self._snap_loss_mode_printed: set[tuple[str, float]] = set()
		self.sign = 1
		self._external_health_cache: dict[tuple[Any, ...], tuple[torch.Tensor, dict[str, float], list[int] | None]] = {}
		self._z_lift_external_cache: dict[tuple[Any, ...], dict[str, Any] | None] = {}
		self._z_lift_model_cache: dict[tuple[Any, ...], dict[str, Any] | None] = {}
		self._fixture_exports_done: set[str] = set()

	def _z_lift_for_stage(
		self,
		fixture: MapFixture,
		cfg: SnapSurfConfig,
		*,
		external_surface_index: int,
		mesh_epoch: int,
		cache_stats: _CacheStats,
	) -> dict[str, Any] | None:
		return _map_init_z_lift_for_fixture(
			fixture,
			cfg,
			sign=int(fixture.metadata.get("sign", self.sign) or self.sign),
			external_surface_index=int(external_surface_index),
			mesh_epoch=int(mesh_epoch),
			external_cache=self._z_lift_external_cache,
			model_cache=self._z_lift_model_cache,
			cache_stats=cache_stats,
		)

	def _ensure_models(self, fixture: MapFixture, base_cfg: SnapSurfConfig, stage: GlobalMapStageConfig) -> None:
		device = fixture.model_xyz.device
		dtype = fixture.model_xyz.dtype
		seed_hw = _seed_ext_hw(fixture.metadata, tuple(int(v) for v in fixture.ext_xyz.shape[:2]), device=device, dtype=dtype)
		if self.affine is None:
			seed_uv = _seed_model_uv(fixture, seed_hw)
			initial_affine = _affine_from_linear(seed_hw, seed_uv, torch.eye(2, device=device, dtype=dtype))
			self.affine = AffineMapModel(
				ext_shape=tuple(int(v) for v in fixture.ext_xyz.shape[:2]),
				device=device,
				dtype=dtype,
				initial=initial_affine,
			)
			self.station_target = self.affine.eval_at(seed_hw).detach()
		if "map_uv_ms" in stage.params and self.global_model is None:
			levels = _max_supported_level(
				tuple(int(v) for v in fixture.ext_xyz.shape[:2]),
				max(int(stage.min_scaledown), int(base_cfg.map_init.scale_levels) - 1),
			) + 1
			self.station_target = self.affine.eval_at(seed_hw).detach()
			self.global_model = GlobalMapModel(self.affine().detach(), levels=levels, factor=2)

	def _uv(self, *, active_level: int = 0) -> torch.Tensor:
		if self.global_model is not None:
			return self.global_model(active_level=int(active_level))
		if self.affine is None:
			raise RuntimeError("global map runtime is not initialized")
		return _affine_uv_for_level(self.affine, tuple(int(v) for v in self.affine.ext_shape), int(active_level))

	def _params_for_stage(self, stage: GlobalMapStageConfig) -> tuple[list[torch.nn.Parameter], int]:
		params: list[torch.nn.Parameter] = []
		level = 0
		if "affine" in stage.params:
			if self.affine is None:
				raise RuntimeError("affine map model is not initialized")
			params.append(self.affine.affine)
		if "map_uv_ms" in stage.params:
			if self.global_model is None:
				raise RuntimeError("global map model is not initialized")
			level = min(max(0, int(stage.min_scaledown)), len(self.global_model.map_uv_ms) - 1)
			params.extend(list(self.global_model.map_uv_ms.parameters())[level:])
		return params, level

	def run_stage(
		self,
		*,
		stage: GlobalMapStageConfig,
		model_xyz: torch.Tensor,
		model_normals: torch.Tensor,
		model_valid: torch.Tensor,
		ext_xyz: torch.Tensor,
		ext_valid: torch.Tensor,
		ext_normals: torch.Tensor,
		ext_quad_valid: torch.Tensor,
		external_surface_index: int = 0,
		mesh_epoch: int = 0,
		persistent_optimizer: bool = False,
		status_fn=None,
		cancel_fn=None,
		auto_stop_fn=None,
	) -> dict[str, float]:
		cache_stats: _CacheStats = {
			"zext_hit": 0,
			"zext_miss": 0,
			"zmdl_hit": 0,
			"zmdl_miss": 0,
			"health_hit": 0,
			"health_miss": 0,
		}
		startup_timing = _truthy_arg(stage.args.get("startup_timing", stage.args.get("opt_timing", True)))
		startup_health_s = 0.0
		startup_z_lift_s = 0.0
		startup_initial_eval_s = 0.0
		fixture = _fixture_from_live_tensors(
			model_xyz=model_xyz,
			model_normals=model_normals,
			model_valid=model_valid,
			ext_xyz=ext_xyz,
			ext_valid=ext_valid,
			ext_normals=ext_normals,
			ext_quad_valid=ext_quad_valid,
			seed_xyz=self.seed_xyz,
			sign=self.sign,
		)
		base_cfg = snap_surf_config_from_global_config(self.cfg_global, stage)
		_t_startup = time.perf_counter()
		fixture = _apply_external_quad_health_filter(
			fixture,
			base_cfg,
			label="runtime",
			external_surface_index=int(external_surface_index),
			cache=self._external_health_cache,
			cache_stats=cache_stats,
		)
		startup_health_s = time.perf_counter() - _t_startup
		cache_stats["health_ms"] = float(cache_stats.get("health_ms", 0.0)) + 1000.0 * float(startup_health_s)
		stage_cfg = _stage_loss_cfg(base_cfg, stage)
		disable_z_lift = _truthy_arg(stage.args.get("disable_z_lift", stage.args.get("disable_turn", False)))
		if disable_z_lift:
			stage_cfg = replace(
				stage_cfg,
				map_init=replace(stage_cfg.map_init, z_lift_enabled=False, w_z_lift=0.0),
			)
		self._ensure_models(fixture, base_cfg, stage)
		assert self.affine is not None
		seed_hw = _seed_ext_hw(fixture.metadata, tuple(int(v) for v in fixture.ext_xyz.shape[:2]), device=fixture.model_xyz.device, dtype=fixture.model_xyz.dtype)
		station_target = self.station_target if self.station_target is not None else self.affine.eval_at(seed_hw).detach()
		w_station = _stage_station_weight(self.cfg_global, stage)
		if _is_affine_seed_quad_init(stage):
			raw_seed = stage.args.get("affine_seed_quad_init", stage.args.get("seed_quad_affine", {}))
			runtime_progress_widths = _global_progress_widths(GlobalMapConfig(base=self.cfg_global.base, stages=(stage,)))
			debug_obj_root = None
			debug_obj_dir = stage.args.get("debug_obj_dir", None)
			if debug_obj_dir is not None and debug_obj_dir is not False:
				if isinstance(debug_obj_dir, bool):
					debug_root = Path("snap_surf_objs")
				else:
					debug_root = Path(str(debug_obj_dir))
				label = stage.name or _stage_param_label(stage.params, fallback="affine_seed_quad_init")
				debug_obj_root = debug_root / f"map_global_{_debug_obj_safe_label(label)}" / "expansion_reopt"
			seed_result, _candidate, _initial_loss, _prep_progress_rows = _prepare_affine_seed_quad_candidate(
				stage=stage,
				affine=self.affine,
				fixture=fixture,
				stage_cfg=stage_cfg,
				seed_hw=seed_hw,
				station_target=station_target,
				w_station=w_station,
				raw=raw_seed,
				lr=float(stage.lr),
				stage_idx=0,
				progress_widths_run=runtime_progress_widths,
				progress_row_idx=0,
				debug_obj_root=debug_obj_root,
				cancel_fn=cancel_fn,
			)
			if seed_result is not None:
				self.sign = int(seed_result.sign)
		if _is_affine_init_scan(stage):
			seed_result = _run_affine_multistart(
				cfg_global=self.cfg_global,
				stage=stage,
				affine=self.affine,
				fixture=fixture,
				stage_cfg=stage_cfg,
				seed_hw=seed_hw,
				station_target=station_target,
				w_station=w_station,
				cancel_fn=cancel_fn,
			)
			if seed_result is not None:
				self.sign = int(seed_result.sign)
		fixture.metadata["sign"] = int(self.sign)
		_t_startup = time.perf_counter()
		stage_z_lift = None if disable_z_lift else self._z_lift_for_stage(
			fixture,
			stage_cfg,
			external_surface_index=int(external_surface_index),
			mesh_epoch=int(mesh_epoch),
			cache_stats=cache_stats,
		)
		startup_z_lift_s = time.perf_counter() - _t_startup
		params, train_level = self._params_for_stage(stage)
		sample_level = _stage_objective_level(stage, train_level, tuple(int(v) for v in fixture.ext_xyz.shape[:2]))
		stage_active_quad = _level_active_quad(_full_active_quad(fixture), int(sample_level))
		startup_initial_eval_s = 0.0
		startup_timing_printed = False
		def _maybe_print_startup_timing() -> None:
			nonlocal startup_timing_printed
			if startup_timing_printed or not startup_timing:
				return
			startup_timing_printed = True
			label = stage.name or _stage_param_label(stage.params, fallback="map_stage")
			print(
				f"[snap_surf.map_global] {label}: startup "
				f"mesh_filter={1000.0 * startup_health_s:.3f}ms "
				f"turn={1000.0 * startup_z_lift_s:.3f}ms "
				f"turn_ext={float(cache_stats.get('zext_ms', 0.0)):.3f}ms "
				f"turn_model={float(cache_stats.get('zmdl_ms', 0.0)):.3f}ms "
				f"initial_eval={1000.0 * startup_initial_eval_s:.3f}ms "
				f"health_hit={cache_stats['health_hit']} health_miss={cache_stats['health_miss']} "
				f"zext_hit={cache_stats['zext_hit']} zext_miss={cache_stats['zext_miss']} "
				f"zmdl_hit={cache_stats['zmdl_hit']} zmdl_miss={cache_stats['zmdl_miss']}",
				flush=True,
			)

		def _stats_from_eval(loss_f: float, terms: dict[str, torch.Tensor], err: dict[str, float]) -> dict[str, float]:
			stats = {
				"snaps_map_loss": float(loss_f),
				"snaps_map_samples": float(_global_term_value(terms, "samples")),
				"snaps_map_runtime_steps": float(self.steps_run),
				"snaps_map_zext_cache_hit": float(cache_stats["zext_hit"]),
				"snaps_map_zext_cache_miss": float(cache_stats["zext_miss"]),
				"snaps_map_zmdl_cache_hit": float(cache_stats["zmdl_hit"]),
				"snaps_map_zmdl_cache_miss": float(cache_stats["zmdl_miss"]),
				"snaps_map_health_cache_hit": float(cache_stats["health_hit"]),
				"snaps_map_health_cache_miss": float(cache_stats["health_miss"]),
				"snaps_map_startup_health_ms": 1000.0 * float(startup_health_s),
				"snaps_map_startup_z_lift_ms": 1000.0 * float(startup_z_lift_s),
				"snaps_map_startup_mesh_filter_ms": 1000.0 * float(startup_health_s),
				"snaps_map_startup_turn_ms": 1000.0 * float(startup_z_lift_s),
				"snaps_map_startup_turn_ext_ms": float(cache_stats.get("zext_ms", 0.0)),
				"snaps_map_startup_turn_model_ms": float(cache_stats.get("zmdl_ms", 0.0)),
				"snaps_map_startup_initial_eval_ms": 1000.0 * float(startup_initial_eval_s),
			}
			for term_key, stat_key in (
				("dist", "snaps_map_dist"),
				("vec", "snaps_map_vec"),
				("norm", "snaps_map_norm"),
				("turn", "snaps_map_turn"),
				("turn_smp", "snaps_map_turn_smp"),
				("smooth", "snaps_map_smooth"),
				("bend", "snaps_map_bend"),
				("jac", "snaps_map_jac"),
				("metric_smooth", "snaps_map_metric_smooth"),
				("area_smooth", "snaps_map_area_smooth"),
				("prior", "snaps_map_prior"),
				("sample_total", "snaps_map_sample_total"),
				("sample_valid", "snaps_map_sample_valid"),
				("sample_base", "snaps_map_sample_base"),
				("sample_model", "snaps_map_sample_model"),
				("sample_limit", "snaps_map_sample_limit"),
				("sample_bad", "snaps_map_sample_bad"),
				("turn_valid", "snaps_map_turn_valid"),
				("loss_quad", "snaps_map_loss_quad"),
				("valid_quad", "snaps_map_valid_quad"),
				("uv_bad", "snaps_map_uvbad"),
				("model_bad", "snaps_map_model_bad"),
				("loss_finite", "snaps_map_loss_finite"),
			):
				stats[stat_key] = float(_global_term_value(terms, term_key))
			if stage_z_lift is not None:
				zs = stage_z_lift["stats"]
				stats["snaps_map_zext_bad"] = float(zs["ext_invalid"])
				stats["snaps_map_zext_unr"] = float(zs["ext_unreachable"])
				stats["snaps_map_zmdl_bad"] = float(zs["model_invalid"])
				stats["snaps_map_zmdl_unr"] = float(zs["model_unreachable"])
				stats["snaps_map_zext_ms"] = float(zs.get("ext_ms", 0.0))
				stats["snaps_map_zmdl_ms"] = float(zs.get("model_ms", 0.0))
				stats["snaps_map_zlift_ms"] = float(zs.get("total_ms", 0.0))
			if int(err["mapping_error_samples"]) > 0:
				stats["snaps_map_avg"] = float(err["avg_model_quad_distance"])
				stats["snaps_map_max"] = float(err["max_model_quad_distance"])
			return stats

		def _stats_for_current_uv() -> dict[str, float]:
			with torch.no_grad():
				loss_f, terms, err = _global_progress_state(
					uv=self._uv(active_level=int(sample_level)),
					fixture=fixture,
					cfg=stage_cfg,
					level=int(sample_level),
					seed_hw=seed_hw,
					station_target=station_target,
					w_station=w_station,
					z_lift=stage_z_lift,
					active_quad=stage_active_quad,
				)
			return _stats_from_eval(loss_f, terms, err)

		status_interval = max(0, int(stage.args.get("status_interval", stage.args.get("debug_print_interval", 100))))
		lr_warmup_steps = _lr_warmup_steps(stage.args)
		lr_autoscale_cfg = _lr_autoscale_config(stage.args)
		auto_loss_history: list[float] = []
		steps_completed = 0
		auto_stopped = False
		lr_autoscale: _LrAutoscaleState | None = None
		opt: torch.optim.Optimizer | None = None
		if params and int(stage.steps) > 0:
			key = (tuple(stage.params), int(train_level), int(sample_level), float(stage.lr), int(stage.min_scaledown))
			if (not persistent_optimizer) or self.optimizer is None or self.optimizer_key != key:
				self.optimizer = torch.optim.Adam(params, lr=float(stage.lr))
				_capture_optimizer_target_lrs(self.optimizer)
				self.optimizer_key = key
				self.lr_autoscale = _LrAutoscaleState(lr_autoscale_cfg) if lr_autoscale_cfg.enabled else None
			opt = self.optimizer
			assert opt is not None
			if persistent_optimizer:
				lr_autoscale = getattr(self, "lr_autoscale", None)
				if lr_autoscale_cfg.enabled and lr_autoscale is None:
					lr_autoscale = _LrAutoscaleState(lr_autoscale_cfg)
					self.lr_autoscale = lr_autoscale
				elif lr_autoscale is not None and lr_autoscale.cfg != lr_autoscale_cfg:
					lr_autoscale = _LrAutoscaleState(lr_autoscale_cfg) if lr_autoscale_cfg.enabled else None
					self.lr_autoscale = lr_autoscale
			else:
				lr_autoscale = _LrAutoscaleState(lr_autoscale_cfg) if lr_autoscale_cfg.enabled else None
			for _ in range(int(stage.steps)):
				if cancel_fn is not None:
					cancel_fn()
				opt.zero_grad(set_to_none=True)
				uv = self._uv(active_level=int(sample_level))
				step1 = _ + 1
				status_due = (
					status_fn is not None and (
						_ == 0 or
						_ == int(stage.steps) - 1 or
						(status_interval > 0 and (_ % status_interval) == 0)
					)
				)
				_t_initial_eval = time.perf_counter() if _ == 0 else None
				loss, terms = _objective_for_uv(
					uv=uv,
					fixture=fixture,
					cfg=stage_cfg,
					level=int(sample_level),
					z_lift=stage_z_lift,
					need_stats=status_due,
					active_quad=stage_active_quad,
				)
				if w_station > 0.0:
					station_raw = _station_loss(uv, _level_seed_hw(seed_hw, int(sample_level)), station_target)
					loss = loss + w_station * station_raw
				else:
					station_raw = loss.new_zeros(())
				if _t_initial_eval is not None:
					startup_initial_eval_s = time.perf_counter() - _t_initial_eval
					_maybe_print_startup_timing()
				warmup_step1 = self.steps_run + 1 if persistent_optimizer else _ + 1
				_apply_optimizer_lr_schedule(
					opt,
					step1=warmup_step1,
					warmup_steps=lr_warmup_steps,
					autoscale=lr_autoscale,
					loss=loss,
				)
				if status_due:
					progress_terms = dict(terms)
					progress_terms["station"] = station_raw.detach()
					err = _fixture_mapping_error(uv.detach(), fixture)
					status_stats = _stats_from_eval(float(loss.detach().cpu()), progress_terms, err)
					status_stats["snaps_map_lr"] = _optimizer_lr_for_display(opt, float(stage.lr))
					status_stats.update(_lr_autoscale_stats("snaps_map", lr_autoscale))
					status_fn(step=_, total=int(stage.steps), stats=status_stats)
				loss.backward()
				opt.step()
				if cancel_fn is not None:
					cancel_fn()
				self.steps_run += 1
				steps_completed = step1
				if auto_stop_fn is not None and step1 > lr_warmup_steps:
					auto_step = step1 - lr_warmup_steps
					auto_loss_history.append(float(loss.detach().cpu()))
					auto_stopped = bool(auto_stop_fn(history=auto_loss_history, step=auto_step))
				if auto_stopped:
					break
			if not persistent_optimizer:
				self.optimizer = None
				self.optimizer_key = None
				self.lr_autoscale = None
		else:
			_t_startup = time.perf_counter()
			initial_stats = _stats_for_current_uv()
			initial_stats["snaps_map_lr"] = float(stage.lr)
			startup_initial_eval_s = time.perf_counter() - _t_startup
			initial_stats["snaps_map_startup_initial_eval_ms"] = 1000.0 * float(startup_initial_eval_s)
			_maybe_print_startup_timing()
			if status_fn is not None:
				status_fn(step=0, total=int(stage.steps), stats=initial_stats)
		stats = _stats_for_current_uv()
		stats["snaps_map_lr"] = _optimizer_lr_for_display(opt if params and int(stage.steps) > 0 else None, float(stage.lr))
		stats["snaps_map_stage_steps"] = float(steps_completed)
		stats["snaps_map_auto_stopped"] = 1.0 if auto_stopped else 0.0
		stats.update(_lr_autoscale_stats("snaps_map", lr_autoscale))
		self.last = stats
		debug_obj_dir = stage.args.get("debug_obj_dir", None)
		if debug_obj_dir:
			if isinstance(debug_obj_dir, bool):
				debug_root = Path("snap_surf_objs")
			else:
				debug_root = Path(str(debug_obj_dir))
			label = stage.name or _stage_param_label(stage.params, fallback="map_stage")
			_write_map_objs(
				debug_root / f"map_global_{_debug_obj_safe_label(label)}",
				uv=self._uv(active_level=0).detach(),
				fixture=fixture,
				meta={
					"name": label,
					"params": list(_public_stage_params(stage.params)),
					"steps": int(stage.steps),
					"persistent_optimizer": bool(persistent_optimizer),
					**stats,
				},
			)
		export_spec = _map_fixture_export_spec(stage)
		if export_spec is not None:
			export_dir = Path(str(export_spec["dir"]))
			export_key = str(export_dir.resolve()) if export_dir.is_absolute() else str(export_dir)
			if (not bool(export_spec["once"])) or export_key not in self._fixture_exports_done:
				final_uv = self._uv(active_level=0).detach()
				state = _surface_state_for_fixture_export(
					fixture=fixture,
					uv=final_uv,
					active_quad=_full_active_quad(fixture),
					sign=int(self.sign),
				)
				export_meta = export_map_fixture(
					export_dir,
					cfg=stage_cfg,
					state=state,
					model_xyz=fixture.model_xyz,
					model_valid=fixture.model_valid,
					model_normals=fixture.model_normals,
					ext_xyz=fixture.ext_xyz,
					ext_valid=fixture.ext_valid,
					ext_normals=fixture.ext_normals,
					ext_quad_valid=fixture.ext_quad_valid,
					seed_xyz=tuple(float(v) for v in fixture.metadata.get("seed_xyz", (0.0, 0.0, 0.0))),
					surface_index=int(external_surface_index),
					surface_count=1,
					step=int(self.steps_run),
					stats=stats,
					export_objs=bool(export_spec["objs"]),
					write_geometry=bool(export_spec["geometry"]),
				)
				self._fixture_exports_done.add(export_key)
				stats["snaps_map_fixture_exported"] = 1.0
				stats["snaps_map_fixture_export_active_quads"] = float(export_meta.get("map_counts", {}).get("active_quads", 0))
				print(
					f"[snap_surf.map_global] exported map fixture dir={export_dir} "
					f"stage={stage.name or _stage_param_label(stage.params, fallback='map_stage')} "
					f"step={int(self.steps_run)}",
					flush=True,
				)
			else:
				stats["snaps_map_fixture_exported"] = 0.0
		return stats

	def snap_loss_prefetch_items(
		self,
		*,
		model_xyz: torch.Tensor,
		model_normals: torch.Tensor,
		model_valid: torch.Tensor,
		ext_xyz: torch.Tensor,
		ext_valid: torch.Tensor,
		ext_normals: torch.Tensor,
		ext_quad_valid: torch.Tensor,
		offset: float = 0.0,
		data=None,
		strip_samples: int = 5,
	) -> dict[str, torch.Tensor]:
		if float(offset) == 0.0:
			return {}
		if data is None:
			return {}
		if self.affine is None:
			fixture = _fixture_from_live_tensors(
				model_xyz=model_xyz,
				model_normals=model_normals,
				model_valid=model_valid,
				ext_xyz=ext_xyz,
				ext_valid=ext_valid,
				ext_normals=ext_normals,
				ext_quad_valid=ext_quad_valid,
				seed_xyz=self.seed_xyz,
				sign=self.sign,
			)
			base_cfg = snap_surf_config_from_global_config(self.cfg_global)
			self._ensure_models(fixture, base_cfg, GlobalMapStageConfig(params=("affine",)))
		with torch.no_grad():
			uv = self._uv().detach()
			depth = torch.zeros((*uv.shape[:-1], 1), device=uv.device, dtype=uv.dtype)
			coords = torch.cat([depth, uv], dim=-1)
			model_ok = _quad_valid_at_coords(model_valid.bool(), coords, tuple(int(v) for v in model_valid.shape))
			ext_ok = ext_valid.bool() & torch.isfinite(ext_xyz).all(dim=-1)
			finite = torch.isfinite(coords).all(dim=-1)
			valid = finite & ext_ok & model_ok
			if not bool(valid.any().detach().cpu()):
				return {}
			safe_coords = torch.where(torch.isfinite(coords), coords, torch.zeros_like(coords))
			model_pos = _sample_surface_grid(model_xyz, safe_coords)
			model_n = torch.nn.functional.normalize(
				_sample_surface_grid(model_normals.detach(), safe_coords),
				dim=-1,
				eps=1.0e-8,
			)
			model_n = model_n * (1.0 if int(self.sign) >= 0 else -1.0)
			sample_valid = (
				valid &
				torch.isfinite(ext_xyz).all(dim=-1) &
				torch.isfinite(model_pos).all(dim=-1) &
				torch.isfinite(model_n).all(dim=-1)
			)
			if not bool(sample_valid.any().detach().cpu()):
				return {}
			strip_samples_i = max(2, int(strip_samples))
			t = torch.linspace(0.0, 1.0, strip_samples_i, device=model_xyz.device, dtype=model_xyz.dtype)
			ext_xyz_valid = ext_xyz.detach()[sample_valid]
			model_pos_valid = model_pos.detach()[sample_valid]
			diff = model_pos_valid - ext_xyz_valid
			strip = ext_xyz_valid.unsqueeze(-2) + t.view(1, strip_samples_i, 1) * diff.unsqueeze(-2)
			return {"grad_mag": strip.reshape(1, 1, -1, 3).contiguous()}

	def snap_loss(
		self,
		*,
		model_xyz: torch.Tensor,
		model_normals: torch.Tensor,
		model_valid: torch.Tensor,
		ext_xyz: torch.Tensor,
		ext_valid: torch.Tensor,
		ext_normals: torch.Tensor,
		ext_quad_valid: torch.Tensor,
		offset: float = 0.0,
		data=None,
		strip_samples: int = 5,
	) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], dict[str, float]]:
		offset_f = float(offset)
		offset_mode = "voxel" if offset_f == 0.0 else "winding"
		mode_key = (offset_mode, offset_f)
		if mode_key not in self._snap_loss_mode_printed:
			print(
				f"[snap_surf.map_global] snap loss offset_mode={offset_mode} offset={offset_f:.6g}",
				flush=True,
			)
			self._snap_loss_mode_printed.add(mode_key)
		if self.affine is None:
			fixture = _fixture_from_live_tensors(
				model_xyz=model_xyz,
				model_normals=model_normals,
				model_valid=model_valid,
				ext_xyz=ext_xyz,
				ext_valid=ext_valid,
				ext_normals=ext_normals,
				ext_quad_valid=ext_quad_valid,
				seed_xyz=self.seed_xyz,
				sign=self.sign,
			)
			base_cfg = snap_surf_config_from_global_config(self.cfg_global)
			self._ensure_models(fixture, base_cfg, GlobalMapStageConfig(params=("affine",)))
		uv = self._uv().detach()
		depth = torch.zeros((*uv.shape[:-1], 1), device=uv.device, dtype=uv.dtype)
		coords = torch.cat([depth, uv], dim=-1)
		model_ok = _quad_valid_at_coords(model_valid.bool(), coords, tuple(int(v) for v in model_valid.shape))
		ext_ok = ext_valid.bool() & torch.isfinite(ext_xyz).all(dim=-1)
		finite = torch.isfinite(coords).all(dim=-1)
		valid = finite & ext_ok & model_ok
		safe_coords = torch.where(torch.isfinite(coords), coords, torch.zeros_like(coords))
		model_pos = _sample_surface_grid(model_xyz, safe_coords)
		model_n = torch.nn.functional.normalize(
			_sample_surface_grid(model_normals.detach(), safe_coords),
			dim=-1,
			eps=1.0e-8,
		)
		model_n = model_n * (1.0 if int(self.sign) >= 0 else -1.0)
		valid = valid & torch.isfinite(model_pos).all(dim=-1) & torch.isfinite(model_n).all(dim=-1)
		z = model_xyz.sum() * 0.0
		lm = torch.zeros(model_xyz.shape[:3], device=model_xyz.device, dtype=model_xyz.dtype).unsqueeze(1)
		mask = torch.zeros_like(lm)
		if not bool(valid.any().detach().cpu()):
			return z, (lm,), (mask,), {
				"snaps_map_snap": 0.0,
				"snaps_map_snap_abs": 0.0,
				"snaps_map_snap_max": 0.0,
				"snaps_map_snap_samples": 0.0,
			}
		signed_vox = ((model_pos - ext_xyz.detach()) * model_n.detach()).sum(dim=-1)
		if offset_f == 0.0:
			residual = signed_vox
		else:
			if data is None:
				raise RuntimeError("snap_surf_map offset mode requires volume data with grad_mag")
			strip_samples_i = max(2, int(strip_samples))
			sample_valid = (
				valid &
				torch.isfinite(ext_xyz).all(dim=-1) &
				torch.isfinite(model_pos).all(dim=-1) &
				torch.isfinite(model_n).all(dim=-1)
			)
			with torch.no_grad():
				t = torch.linspace(0.0, 1.0, strip_samples_i, device=model_xyz.device, dtype=model_xyz.dtype)
				origin = torch.tensor(data.origin_fullres, device=model_xyz.device, dtype=model_xyz.dtype)
				spacing = torch.tensor(data._spacing_for("grad_mag"), device=model_xyz.device, dtype=model_xyz.dtype)
				sentinel = origin - 64.0 * spacing
				ext_xyz_safe = torch.where(sample_valid.unsqueeze(-1), ext_xyz.detach(), sentinel.view(1, 1, 3))
				model_pos_safe = torch.where(sample_valid.unsqueeze(-1), model_pos.detach(), sentinel.view(1, 1, 3))
				diff = model_pos_safe - ext_xyz_safe
				strip = ext_xyz_safe.unsqueeze(-2) + (
					t.view(*((1,) * (ext_xyz.ndim - 1)), strip_samples_i, 1) * diff.unsqueeze(-2)
				)
				H, W = int(ext_xyz.shape[0]), int(ext_xyz.shape[1])
				strip_flat = strip.reshape(1, H, W * strip_samples_i, 3)
				sampled = data.grid_sample_fullres(strip_flat, channels={"grad_mag"})
				if sampled.grad_mag is None:
					raise RuntimeError("snap_surf_map offset mode requires grad_mag samples")
				mag = sampled.grad_mag.detach().squeeze(0).squeeze(0).reshape(1, H, W, strip_samples_i).squeeze(0)
				strip_valid = (mag > 0.0).all(dim=-1)
				mean_grad = mag.mean(dim=-1)
				strip_len = diff.square().sum(dim=-1).sqrt()
				int_sign = torch.sign(((model_pos.detach() - ext_xyz.detach()) * model_n.detach()).sum(dim=-1))
				signed_windings = int_sign * strip_len * mean_grad
				winding_err = signed_windings - signed_vox.new_tensor(offset_f)
			valid = sample_valid & strip_valid
			normal_residual = signed_vox * mean_grad
			residual = normal_residual + (winding_err - normal_residual).detach()
		vals = residual[valid]
		if vals.numel() == 0:
			return z, (lm,), (mask,), {
				"snaps_map_snap": 0.0,
				"snaps_map_snap_abs": 0.0,
				"snaps_map_snap_max": 0.0,
				"snaps_map_snap_samples": 0.0,
			}
		loss = vals.square().mean()
		stats = {
			"snaps_map_snap": float(loss.detach().cpu()),
			"snaps_map_snap_abs": float(vals.detach().abs().mean().cpu()),
			"snaps_map_snap_max": float(vals.detach().abs().max().cpu()),
			"snaps_map_snap_samples": float(int(vals.numel())),
		}
		return loss, (lm,), (mask,), stats


def optimize_fixture(
	fixture_dir: str | Path,
	config_path: str | Path,
	*,
	out_dir: str | Path,
	device: torch.device | str = "cpu",
) -> dict[str, Any]:
	_PRINTED_PROGRESS_LEGENDS.clear()
	device_t = torch.device(str(device))
	fixture = load_map_fixture(fixture_dir, device=device_t)
	cfg_global = parse_global_map_config(config_path)
	base_cfg = snap_surf_config_from_global_config(cfg_global)
	fixture = _apply_external_quad_health_filter(fixture, base_cfg, label="fixture")
	fixture_z_lift_external_cache: dict[tuple[Any, ...], dict[str, Any] | None] = {}
	fixture_z_lift_model_cache: dict[tuple[Any, ...], dict[str, Any] | None] = {}
	dtype = fixture.model_xyz.dtype
	seed_hw = _seed_ext_hw(fixture.metadata, tuple(int(v) for v in fixture.ext_xyz.shape[:2]), device=device_t, dtype=dtype)
	seed_uv = _seed_model_uv(fixture, seed_hw)
	initial_affine = _affine_from_linear(seed_hw, seed_uv, torch.eye(2, device=device_t, dtype=dtype))
	affine = AffineMapModel(
		ext_shape=tuple(int(v) for v in fixture.ext_xyz.shape[:2]),
		device=device_t,
		dtype=dtype,
		initial=initial_affine,
	)
	global_model: GlobalMapModel | None = None
	station_target = affine.eval_at(seed_hw).detach()
	history: list[dict[str, Any]] = []
	full_active = _full_active_quad(fixture)
	out = Path(out_dir)
	out.mkdir(parents=True, exist_ok=True)
	progress_rows = 0
	progress_widths_run = _global_progress_widths(cfg_global)
	first_affine_stage = next((stage for stage in cfg_global.stages if "affine" in stage.params), None)
	if first_affine_stage is not None:
		_print_reference_affine_diagnostic(
			affine=affine,
			fixture=fixture,
			stage_cfg=_stage_loss_cfg(snap_surf_config_from_global_config(cfg_global, first_affine_stage), first_affine_stage),
			seed_hw=seed_hw,
			station_target=station_target,
		)

	for stage_idx, stage in enumerate(cfg_global.stages):
		stage_cfg = _stage_loss_cfg(snap_surf_config_from_global_config(cfg_global, stage), stage)
		w_station = _stage_station_weight(cfg_global, stage)
		if _is_affine_seed_quad_init(stage):
			write_stage_objs = _obj_outputs_enabled(cfg_global, stage)
			progress_rows += _run_affine_seed_quad_init(
				stage_idx=stage_idx,
				stage=stage,
				affine=affine,
				fixture=fixture,
				stage_cfg=stage_cfg,
				seed_hw=seed_hw,
				station_target=station_target,
				w_station=w_station,
				progress_widths_run=progress_widths_run,
				progress_row_idx=progress_rows,
				out_root=out,
				write_objs=write_stage_objs,
			)
			stage_uv = affine().detach()
			err = _fixture_mapping_error(stage_uv, fixture)
			history.append({
				"stage": stage_idx,
				"name": stage.name,
				"step": int(stage.steps),
				"params": list(_public_stage_params(stage.params)),
				"train_min_level": 0,
				**err,
			})
			if write_stage_objs:
				_write_stage_objs(out, stage_idx=stage_idx, stage=stage, uv=stage_uv, fixture=fixture)
			continue
		if _is_affine_init_scan(stage):
			_run_affine_multistart(
				cfg_global=cfg_global,
				stage=stage,
				affine=affine,
				fixture=fixture,
				stage_cfg=stage_cfg,
				seed_hw=seed_hw,
				station_target=station_target,
				w_station=w_station,
			)
			stage_uv = affine().detach()
			err = _fixture_mapping_error(stage_uv, fixture)
			history.append({
				"stage": stage_idx,
				"name": stage.name,
				"step": 0,
				"params": list(_public_stage_params(stage.params)),
				"train_min_level": 0,
				**err,
			})
			if _obj_outputs_enabled(cfg_global, stage):
				_write_stage_objs(out, stage_idx=stage_idx, stage=stage, uv=stage_uv, fixture=fixture)
			continue
		params: list[torch.nn.Parameter] = []
		max_level = _max_supported_level(tuple(int(v) for v in fixture.ext_xyz.shape[:2]), int(stage.min_scaledown))
		level = min(int(stage.min_scaledown), max_level)
		if "affine" in stage.params:
			params.append(affine.affine)
		if "map_uv_ms" in stage.params:
			if global_model is None:
				levels = _max_supported_level(tuple(int(v) for v in fixture.ext_xyz.shape[:2]), max(int(stage.min_scaledown), int(base_cfg.map_init.scale_levels) - 1)) + 1
				station_target = affine.eval_at(seed_hw).detach()
				global_model = GlobalMapModel(affine().detach(), levels=levels, factor=2)
			level = min(level, len(global_model.map_uv_ms) - 1)
			params.extend(list(global_model.map_uv_ms.parameters())[level:])
		if not params or int(stage.steps) <= 0:
			continue
		if "affine" in stage.params:
			_run_affine_multistart(
				cfg_global=cfg_global,
				stage=stage,
				affine=affine,
				fixture=fixture,
				stage_cfg=stage_cfg,
				seed_hw=seed_hw,
				station_target=station_target,
				w_station=w_station,
			)
		stage_cache_stats: _CacheStats = {}
		stage_z_lift = _map_init_z_lift_for_fixture(
			fixture,
			stage_cfg,
			sign=int(fixture.metadata.get("sign", 1) or 1),
			external_surface_index=0,
			mesh_epoch=0,
			external_cache=fixture_z_lift_external_cache,
			model_cache=fixture_z_lift_model_cache,
			cache_stats=stage_cache_stats,
		)
		sample_level = _stage_objective_level(stage, level, tuple(int(v) for v in fixture.ext_xyz.shape[:2]))
		stage_active_quad = _level_active_quad(_full_active_quad(fixture), int(sample_level))
		opt = torch.optim.Adam(params, lr=float(stage.lr))
		_capture_optimizer_target_lrs(opt)
		status_interval = max(0, int(stage.args.get("status_interval", stage.args.get("debug_print_interval", 100))))
		lr_warmup_steps = _lr_warmup_steps(stage.args)
		lr_autoscale = _make_lr_autoscale_state(stage.args)
		last_status_time: float | None = None
		last_status_step = 0
		with torch.no_grad():
			if "map_uv_ms" in stage.params and global_model is not None:
				uv_report = global_model(active_level=int(sample_level))
			else:
				uv_report = _affine_uv_for_level(affine, tuple(int(v) for v in fixture.ext_xyz.shape[:2]), int(sample_level))
			report_loss, report_terms, report_err = _global_progress_state(
				uv=uv_report,
				fixture=fixture,
				cfg=stage_cfg,
				level=int(sample_level),
				seed_hw=seed_hw,
				station_target=station_target,
				w_station=w_station,
				z_lift=stage_z_lift,
				active_quad=stage_active_quad,
			)
		_print_global_progress(
			row_idx=progress_rows,
			widths=progress_widths_run,
			stage_idx=stage_idx,
			iter_label=f"0/{int(stage.steps)}",
			lr=float(stage.lr),
			level=level,
			loss=report_loss,
			terms=report_terms,
			it_s=None,
			err=report_err,
		)
		progress_rows += 1
		for step in range(int(stage.steps)):
			opt.zero_grad(set_to_none=True)
			if "map_uv_ms" in stage.params and global_model is not None:
				uv = global_model(active_level=int(sample_level))
			else:
				uv = _affine_uv_for_level(affine, tuple(int(v) for v in fixture.ext_xyz.shape[:2]), int(sample_level))
			loss, _terms = _objective_for_uv(
				uv=uv,
				fixture=fixture,
				cfg=stage_cfg,
				level=int(sample_level),
				z_lift=stage_z_lift,
				need_stats=False,
				active_quad=stage_active_quad,
			)
			station_raw = loss.new_zeros(())
			if w_station > 0.0:
				station_raw = _station_loss(uv, _level_seed_hw(seed_hw, int(sample_level)), station_target)
				loss = loss + w_station * station_raw
			loss.backward()
			_apply_optimizer_lr_schedule(
				opt,
				step1=step + 1,
				warmup_steps=lr_warmup_steps,
				autoscale=lr_autoscale,
				loss=loss,
			)
			opt.step()
			with torch.no_grad():
				if "map_uv_ms" in stage.params and global_model is not None:
					uv_after = global_model(active_level=int(sample_level))
				else:
					uv_after = _affine_uv_for_level(affine, tuple(int(v) for v in fixture.ext_xyz.shape[:2]), int(sample_level))
			step1 = step + 1
			status_due = (
				step == 0 or
				step1 == int(stage.steps) or
				(status_interval > 0 and (step1 % status_interval) == 0)
			)
			if status_due:
				report_loss, report_terms, err = _global_progress_state(
					uv=uv_after,
					fixture=fixture,
					cfg=stage_cfg,
					level=int(sample_level),
					seed_hw=seed_hw,
					station_target=station_target,
					w_station=w_station,
					z_lift=stage_z_lift,
					active_quad=stage_active_quad,
				)
				now = time.monotonic()
				it_s = None
				if last_status_time is not None:
					it_s = float(step1 - last_status_step) / max(1.0e-9, now - last_status_time)
				last_status_time = now
				last_status_step = step1
				_print_global_progress(
					row_idx=progress_rows,
					widths=progress_widths_run,
					stage_idx=stage_idx,
					iter_label=f"{step1}/{int(stage.steps)}",
					lr=_optimizer_lr_for_display(opt, float(stage.lr)),
					level=level,
					loss=report_loss,
					terms=report_terms,
					it_s=it_s,
					err=err,
				)
				progress_rows += 1
				if step == int(stage.steps) - 1:
					report_loss, report_terms, err = _global_progress_state(
						uv=uv_after,
						fixture=fixture,
						cfg=stage_cfg,
						level=int(sample_level),
						seed_hw=seed_hw,
						station_target=station_target,
						w_station=w_station,
						z_lift=stage_z_lift,
						active_quad=stage_active_quad,
					)
				history.append({
					"stage": stage_idx,
					"step": step,
					"params": list(_public_stage_params(stage.params)),
					"train_min_level": level,
					"loss": report_loss,
					**err,
					"terms": {k: float(v.detach().cpu()) for k, v in report_terms.items() if v.ndim == 0},
				})
		if "map_uv_ms" in stage.params and global_model is not None:
			stage_uv = global_model(active_level=0).detach()
		else:
			stage_uv = affine().detach()
		if _obj_outputs_enabled(cfg_global, stage):
			_write_stage_objs(out, stage_idx=stage_idx, stage=stage, uv=stage_uv, fixture=fixture)

	final_uv = global_model(active_level=0).detach() if global_model is not None else affine().detach()
	if _obj_outputs_enabled(cfg_global):
		_write_map_objs(
			out / "objs" / "final",
			uv=final_uv,
			fixture=fixture,
			meta={
				"name": "final",
				"params": ["map_surf_ms"] if global_model is not None else ["map_surf_affine"],
				"stages_completed": len(cfg_global.stages),
			},
		)
	_float_tif(out / "model_x.tif", final_uv[..., 1])
	_float_tif(out / "model_y.tif", final_uv[..., 0])
	meta = {
		"kind": "snap_surf_global_map",
		"fixture_dir": str(Path(fixture_dir)),
		"config_path": str(Path(config_path)),
		"ext_shape": [int(v) for v in fixture.ext_xyz.shape[:2]],
		"model_shape": [int(v) for v in fixture.model_xyz.shape[:3]],
		"active_quads": int(full_active.sum().detach().cpu()),
		"affine": affine.affine.detach().cpu(),
		"station_seed_ext_hw": seed_hw.detach().cpu(),
		"station_target_uv": station_target.detach().cpu(),
		"sign": int(fixture.metadata.get("sign", 1) or 1),
		"sign_semantics": "model_normal_alignment",
		"seed_quad_init": fixture.metadata.get("seed_quad_init"),
		"stages": [asdict(stage) for stage in cfg_global.stages],
	}
	_write_json(out / "meta.json", meta)
	metrics = _global_metrics(final_uv, fixture)
	metrics.update(_fixture_mapping_error(final_uv, fixture))
	metrics["history"] = history
	metrics["fixture_dir"] = str(Path(fixture_dir))
	metrics["out_dir"] = str(out)
	_write_json(out / "metrics.json", metrics)
	return metrics


def _global_metrics(uv: torch.Tensor, fixture: MapFixture) -> dict[str, Any]:
	finite = torch.isfinite(uv).all(dim=-1)
	ref_finite = torch.isfinite(fixture.reference_uv).all(dim=-1)
	common = finite & ref_finite
	if bool(common.any().detach().cpu()):
		d = uv[common] - fixture.reference_uv[common]
		abs_d = d.abs()
		l2 = d.square().sum(dim=-1).sqrt()
		max_abs = abs_d.max(dim=0).values
		mean_abs = abs_d.mean(dim=0)
		rms = d.square().mean(dim=0).sqrt()
		max_l2 = l2.max()
	else:
		max_abs = uv.new_zeros(2)
		mean_abs = uv.new_zeros(2)
		rms = uv.new_zeros(2)
		max_l2 = uv.new_zeros(())
	return {
		"finite_vertices": int(finite.sum().detach().cpu()),
		"reference_finite_vertices": int(ref_finite.sum().detach().cpu()),
		"common_vertices": int(common.sum().detach().cpu()),
		"model_y_max_abs_delta": float(max_abs[0].detach().cpu()),
		"model_x_max_abs_delta": float(max_abs[1].detach().cpu()),
		"model_y_mean_abs_delta": float(mean_abs[0].detach().cpu()),
		"model_x_mean_abs_delta": float(mean_abs[1].detach().cpu()),
		"model_y_rms_delta": float(rms[0].detach().cpu()),
		"model_x_rms_delta": float(rms[1].detach().cpu()),
		"model_l2_max_delta": float(max_l2.detach().cpu()),
	}


def _reference_map_tensors(
	reference_dir: str | Path | None,
	fixture: MapFixture,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
	if reference_dir is None:
		return fixture.reference_uv, fixture.reference_active_quad, fixture.reference_blocked_quad, str(fixture.root)
	root = Path(reference_dir)
	if (root / "fixture.json").exists():
		ref_fixture = load_map_fixture(root, device=fixture.reference_uv.device)
		return ref_fixture.reference_uv, ref_fixture.reference_active_quad, ref_fixture.reference_blocked_quad, str(root)
	map_root = root / "map" if (root / "map" / "model_x.tif").exists() else root
	import tifffile

	model_x = torch.as_tensor(tifffile.imread(str(map_root / "model_x.tif")), device=fixture.reference_uv.device, dtype=fixture.reference_uv.dtype)
	model_y = torch.as_tensor(tifffile.imread(str(map_root / "model_y.tif")), device=fixture.reference_uv.device, dtype=fixture.reference_uv.dtype)
	reference_uv = torch.stack([model_y, model_x], dim=-1)
	active_path = map_root / "active_quad.tif"
	blocked_path = map_root / "blocked_quad.tif"
	if active_path.exists():
		reference_active_quad = torch.as_tensor(tifffile.imread(str(active_path)), device=fixture.reference_uv.device).bool()
	else:
		reference_active_quad = fixture.reference_active_quad
	if blocked_path.exists():
		reference_blocked_quad = torch.as_tensor(tifffile.imread(str(blocked_path)), device=fixture.reference_uv.device).bool()
	else:
		reference_blocked_quad = torch.zeros_like(reference_active_quad)
	return reference_uv, reference_active_quad, reference_blocked_quad, str(root)


def _benchmark_device_metadata(device: torch.device) -> dict[str, Any]:
	meta: dict[str, Any] = {
		"device": str(device),
		"torch_version": str(torch.__version__),
		"cuda_available": bool(torch.cuda.is_available()),
	}
	if device.type == "cuda":
		idx = 0 if device.index is None else int(device.index)
		meta.update({
			"cuda_device_index": idx,
			"cuda_device_name": torch.cuda.get_device_name(idx),
			"cuda_version": torch.version.cuda,
		})
	return meta


def benchmark_fixture(
	fixture_dir: str | Path,
	config_path: str | Path,
	*,
	out_dir: str | Path,
	device: torch.device | str = "cpu",
	reference_dir: str | Path | None = None,
	max_model_abs_delta: float = 1.0,
	max_model_l2_delta: float = 1.0,
	require_mask_equal: bool = True,
	profile_components: bool = False,
	profile_repeats: int = 3,
	profile_stage: str | int | None = None,
	profiler_trace: str | Path | None = None,
) -> dict[str, Any]:
	device_t = torch.device(str(device))
	out = Path(out_dir)
	out.mkdir(parents=True, exist_ok=True)
	if device_t.type == "cuda":
		torch.cuda.synchronize(device_t)
	start = time.perf_counter()
	metrics = optimize_fixture(fixture_dir, config_path, out_dir=out, device=device_t)
	if device_t.type == "cuda":
		torch.cuda.synchronize(device_t)
	elapsed_s = time.perf_counter() - start
	fixture = load_map_fixture(fixture_dir, device=device_t)
	fixture = _apply_external_quad_health_filter(fixture, snap_surf_config_from_global_config(parse_global_map_config(config_path)), label="benchmark")
	import tifffile

	model_x = torch.as_tensor(tifffile.imread(str(out / "model_x.tif")), device=device_t, dtype=fixture.reference_uv.dtype)
	model_y = torch.as_tensor(tifffile.imread(str(out / "model_y.tif")), device=device_t, dtype=fixture.reference_uv.dtype)
	rerun_uv = torch.stack([model_y, model_x], dim=-1)
	rerun_active = _full_active_quad(fixture)
	rerun_blocked = torch.zeros_like(rerun_active)
	ref_uv, ref_active, ref_blocked, ref_label = _reference_map_tensors(reference_dir, fixture)
	compare = compare_map_tensors(
		reference_uv=ref_uv.to(device=device_t, dtype=rerun_uv.dtype),
		reference_active_quad=ref_active.to(device=device_t).bool(),
		reference_blocked_quad=ref_blocked.to(device=device_t).bool(),
		rerun_uv=rerun_uv,
		rerun_active_quad=rerun_active,
		rerun_blocked_quad=rerun_blocked,
	)
	thresholds = {
		"max_model_abs_delta": float(max_model_abs_delta),
		"max_model_l2_delta": float(max_model_l2_delta),
		"require_mask_equal": bool(require_mask_equal),
	}
	max_abs_observed = max(float(compare["model_y_max_abs_delta"]), float(compare["model_x_max_abs_delta"]))
	passed = (
		max_abs_observed <= float(max_model_abs_delta) and
		float(compare["model_l2_max_delta"]) <= float(max_model_l2_delta) and
		((not bool(require_mask_equal)) or (bool(compare["active_quad_equal"]) and bool(compare["blocked_quad_equal"])))
	)
	profile_rows: list[dict[str, Any]] = []
	if bool(profile_components):
		profile_rows = profile_fixture_components(
			fixture_dir,
			config_path,
			out_dir=out,
			device=device_t,
			repeats=int(profile_repeats),
			stage=profile_stage,
			profiler_trace=profiler_trace,
		)
	result = {
		"kind": "snap_surf_map_benchmark",
		"fixture_dir": str(Path(fixture_dir)),
		"config_path": str(Path(config_path)),
		"out_dir": str(out),
		"reference_dir": ref_label,
		"elapsed_s": float(elapsed_s),
		"device": _benchmark_device_metadata(device_t),
		"thresholds": thresholds,
		"passed": bool(passed),
		"status": "pass" if bool(passed) else "fail",
		"optimizer_metrics": metrics,
		"map_deltas": compare,
		"profile_components": profile_rows,
	}
	_write_json(out / "benchmark.json", result)
	return result


_PROFILE_COMPONENT_WEIGHTS: dict[str, dict[str, float]] = {
	"dist": {"w_dist": 1.0},
	"vec": {"w_vec_normal": 1.0},
	"norm": {"w_surface_normal": 1.0},
	"turn": {"w_z_lift": 1.0},
	"smooth": {"w_smooth": 1.0},
	"bend": {"w_bend": 1.0},
	"jac": {"w_jac": 1.0},
	"metric_smooth": {"w_metric_smooth": 1.0},
	"area_smooth": {"w_area_smooth": 1.0},
	"prior": {"w_dense_prior": 1.0},
}


def _profile_component_cfg(stage_cfg: SnapSurfConfig, component: str) -> SnapSurfConfig:
	if component == "all":
		return stage_cfg
	zeroed = {
		"w_dist": 0.0,
		"w_vec_normal": 0.0,
		"w_surface_normal": 0.0,
		"w_z_lift": 0.0,
		"w_smooth": 0.0,
		"w_bend": 0.0,
		"w_jac": 0.0,
		"w_metric_smooth": 0.0,
		"w_area_smooth": 0.0,
		"w_dense_prior": 0.0,
	}
	zeroed.update(_PROFILE_COMPONENT_WEIGHTS[component])
	if component != "turn":
		zeroed["z_lift_enabled"] = bool(stage_cfg.map_init.z_lift_enabled)
	return replace(stage_cfg, map_init=replace(stage_cfg.map_init, **zeroed))


def _profile_stage_selected(stage: GlobalMapStageConfig, stage_idx: int, selector: str | int | None) -> bool:
	if selector is None:
		return True
	if isinstance(selector, int):
		return int(stage_idx) == int(selector)
	raw = str(selector)
	if raw.isdigit():
		return int(stage_idx) == int(raw)
	return str(stage.name) == raw


def _snapshot_parameters(params: list[torch.nn.Parameter]) -> list[torch.Tensor]:
	return [p.detach().clone() for p in params]


def _assert_parameters_unchanged(params: list[torch.nn.Parameter], before: list[torch.Tensor]) -> None:
	for i, (p, old) in enumerate(zip(params, before)):
		if not bool(torch.equal(p.detach(), old)):
			raise RuntimeError(f"profile changed optimizer parameter {i}")


def _run_fixture_stage_training_step_loop(
	*,
	stage: GlobalMapStageConfig,
	stage_cfg: SnapSurfConfig,
	fixture: MapFixture,
	affine: AffineMapModel,
	global_model: GlobalMapModel | None,
	level: int,
	sample_level: int,
	stage_active_quad: torch.Tensor,
	stage_z_lift: dict[str, Any] | None | object,
	params: list[torch.nn.Parameter],
	seed_hw: torch.Tensor,
	station_target: torch.Tensor,
	w_station: float,
) -> None:
	opt = torch.optim.Adam(params, lr=float(stage.lr))
	_capture_optimizer_target_lrs(opt)
	lr_warmup_steps = _lr_warmup_steps(stage.args)
	lr_autoscale = _make_lr_autoscale_state(stage.args)
	for step in range(int(stage.steps)):
		opt.zero_grad(set_to_none=True)
		if "map_uv_ms" in stage.params and global_model is not None:
			uv = global_model(active_level=int(sample_level))
		else:
			uv = _affine_uv_for_level(affine, tuple(int(v) for v in fixture.ext_xyz.shape[:2]), int(sample_level))
		loss, _terms = _objective_for_uv(
			uv=uv,
			fixture=fixture,
			cfg=stage_cfg,
			level=int(sample_level),
			z_lift=stage_z_lift,
			need_stats=False,
			active_quad=stage_active_quad,
		)
		if float(w_station) > 0.0:
			loss = loss + float(w_station) * _station_loss(uv, _level_seed_hw(seed_hw, int(sample_level)), station_target)
		loss.backward()
		_apply_optimizer_lr_schedule(
			opt,
			step1=step + 1,
			warmup_steps=lr_warmup_steps,
			autoscale=lr_autoscale,
			loss=loss,
		)
		opt.step()


def _time_profile_component(
	*,
	component: str,
	repeats: int,
	uv_fn: Any,
	params: list[torch.nn.Parameter],
	fixture: MapFixture,
	cfg: SnapSurfConfig,
	level: int,
	z_lift: dict[str, Any] | None | object,
	active_quad: torch.Tensor,
) -> dict[str, Any]:
	times: list[float] = []
	loss_value = 0.0
	for _ in range(max(1, int(repeats))):
		for p in params:
			p.grad = None
		if fixture.model_xyz.is_cuda:
			torch.cuda.synchronize(fixture.model_xyz.device)
		start = time.perf_counter()
		uv = uv_fn()
		loss, terms = _objective_for_uv(
			uv=uv,
			fixture=fixture,
			cfg=cfg,
			level=int(level),
			z_lift=z_lift,
			need_stats=False,
			active_quad=active_quad,
		)
		loss.backward()
		if fixture.model_xyz.is_cuda:
			torch.cuda.synchronize(fixture.model_xyz.device)
		times.append(time.perf_counter() - start)
		loss_value = float(terms["loss"].detach().cpu())
	for p in params:
		p.grad = None
	times_sorted = sorted(times)
	n = len(times_sorted)
	return {
		"component": component,
		"repeats": int(n),
		"loss": float(loss_value),
		"mean_s": float(sum(times) / max(1, len(times))),
		"min_s": float(times_sorted[0]),
		"median_s": float(times_sorted[n // 2]),
		"max_s": float(times_sorted[-1]),
	}


def _write_profile_trace(
	*,
	path: str | Path,
	uv_fn: Any,
	params: list[torch.nn.Parameter],
	fixture: MapFixture,
	cfg: SnapSurfConfig,
	level: int,
	z_lift: dict[str, Any] | None | object,
	active_quad: torch.Tensor,
) -> None:
	trace_path = Path(path)
	trace_path.parent.mkdir(parents=True, exist_ok=True)
	for p in params:
		p.grad = None
	activities = [torch.profiler.ProfilerActivity.CPU]
	if fixture.model_xyz.is_cuda:
		activities.append(torch.profiler.ProfilerActivity.CUDA)
		torch.cuda.synchronize(fixture.model_xyz.device)
	with torch.profiler.profile(activities=activities, record_shapes=False, profile_memory=False) as prof:
		uv = uv_fn()
		loss, _terms = _objective_for_uv(
			uv=uv,
			fixture=fixture,
			cfg=cfg,
			level=int(level),
			z_lift=z_lift,
			need_stats=False,
			active_quad=active_quad,
		)
		loss.backward()
		if fixture.model_xyz.is_cuda:
			torch.cuda.synchronize(fixture.model_xyz.device)
	prof.export_chrome_trace(str(trace_path))
	for p in params:
		p.grad = None


def profile_fixture_components(
	fixture_dir: str | Path,
	config_path: str | Path,
	*,
	out_dir: str | Path,
	device: torch.device | str = "cpu",
	repeats: int = 3,
	stage: str | int | None = None,
	profiler_trace: str | Path | None = None,
) -> list[dict[str, Any]]:
	_PRINTED_PROGRESS_LEGENDS.clear()
	device_t = torch.device(str(device))
	fixture = load_map_fixture(fixture_dir, device=device_t)
	cfg_global = parse_global_map_config(config_path)
	base_cfg = snap_surf_config_from_global_config(cfg_global)
	fixture = _apply_external_quad_health_filter(fixture, base_cfg, label="profile")
	fixture_z_lift_external_cache: dict[tuple[Any, ...], dict[str, Any] | None] = {}
	fixture_z_lift_model_cache: dict[tuple[Any, ...], dict[str, Any] | None] = {}
	dtype = fixture.model_xyz.dtype
	seed_hw = _seed_ext_hw(fixture.metadata, tuple(int(v) for v in fixture.ext_xyz.shape[:2]), device=device_t, dtype=dtype)
	seed_uv = _seed_model_uv(fixture, seed_hw)
	initial_affine = _affine_from_linear(seed_hw, seed_uv, torch.eye(2, device=device_t, dtype=dtype))
	affine = AffineMapModel(
		ext_shape=tuple(int(v) for v in fixture.ext_xyz.shape[:2]),
		device=device_t,
		dtype=dtype,
		initial=initial_affine,
	)
	global_model: GlobalMapModel | None = None
	station_target = affine.eval_at(seed_hw).detach()
	for stage_idx, stage_cfg_raw in enumerate(cfg_global.stages):
		stage_cfg = _stage_loss_cfg(snap_surf_config_from_global_config(cfg_global, stage_cfg_raw), stage_cfg_raw)
		w_station = _stage_station_weight(cfg_global, stage_cfg_raw)
		if _is_affine_seed_quad_init(stage_cfg_raw):
			_run_affine_seed_quad_init(
				stage_idx=stage_idx,
				stage=stage_cfg_raw,
				affine=affine,
				fixture=fixture,
				stage_cfg=stage_cfg,
				seed_hw=seed_hw,
				station_target=station_target,
				w_station=w_station,
				progress_widths_run=_global_progress_widths(cfg_global),
				progress_row_idx=0,
				out_root=Path(out_dir),
				write_objs=False,
			)
			continue
		if _is_affine_init_scan(stage_cfg_raw):
			_run_affine_multistart(
				cfg_global=cfg_global,
				stage=stage_cfg_raw,
				affine=affine,
				fixture=fixture,
				stage_cfg=stage_cfg,
				seed_hw=seed_hw,
				station_target=station_target,
				w_station=w_station,
			)
			continue
		params: list[torch.nn.Parameter] = []
		max_level = _max_supported_level(tuple(int(v) for v in fixture.ext_xyz.shape[:2]), int(stage_cfg_raw.min_scaledown))
		level = min(int(stage_cfg_raw.min_scaledown), max_level)
		if "affine" in stage_cfg_raw.params:
			params.append(affine.affine)
		if "map_uv_ms" in stage_cfg_raw.params:
			if global_model is None:
				levels = _max_supported_level(tuple(int(v) for v in fixture.ext_xyz.shape[:2]), max(int(stage_cfg_raw.min_scaledown), int(base_cfg.map_init.scale_levels) - 1)) + 1
				station_target = affine.eval_at(seed_hw).detach()
				global_model = GlobalMapModel(affine().detach(), levels=levels, factor=2)
			level = min(level, len(global_model.map_uv_ms) - 1)
			params.extend(list(global_model.map_uv_ms.parameters())[level:])
		if not params:
			continue
		if "affine" in stage_cfg_raw.params:
			_run_affine_multistart(
				cfg_global=cfg_global,
				stage=stage_cfg_raw,
				affine=affine,
				fixture=fixture,
				stage_cfg=stage_cfg,
				seed_hw=seed_hw,
				station_target=station_target,
				w_station=w_station,
			)
		stage_z_lift = _map_init_z_lift_for_fixture(
			fixture,
			stage_cfg,
			sign=int(fixture.metadata.get("sign", 1) or 1),
			external_surface_index=0,
			mesh_epoch=0,
			external_cache=fixture_z_lift_external_cache,
			model_cache=fixture_z_lift_model_cache,
		)
		sample_level = _stage_objective_level(stage_cfg_raw, level, tuple(int(v) for v in fixture.ext_xyz.shape[:2]))
		stage_active_quad = _level_active_quad(_full_active_quad(fixture), int(sample_level))
		if "map_uv_ms" in stage_cfg_raw.params and global_model is not None and _profile_stage_selected(stage_cfg_raw, stage_idx, stage):
			profile_params = list(global_model.map_uv_ms.parameters())[level:]
			before = _snapshot_parameters([affine.affine] + list(global_model.map_uv_ms.parameters()))
			uv_fn = lambda: global_model(active_level=int(sample_level))
			rows: list[dict[str, Any]] = []
			components = ["all"] + list(_PROFILE_COMPONENT_WEIGHTS.keys())
			for component in components:
				cfg_i = _profile_component_cfg(stage_cfg, component)
				rows.append(_time_profile_component(
					component=component,
					repeats=int(repeats),
					uv_fn=uv_fn,
					params=profile_params,
					fixture=fixture,
					cfg=cfg_i,
					level=int(sample_level),
					z_lift=stage_z_lift,
					active_quad=stage_active_quad,
				))
			if profiler_trace is not None:
				_write_profile_trace(
					path=profiler_trace,
					uv_fn=uv_fn,
					params=profile_params,
					fixture=fixture,
					cfg=stage_cfg,
					level=int(sample_level),
					z_lift=stage_z_lift,
					active_quad=stage_active_quad,
				)
			_assert_parameters_unchanged([affine.affine] + list(global_model.map_uv_ms.parameters()), before)
			out = Path(out_dir)
			out.mkdir(parents=True, exist_ok=True)
			payload = {
				"kind": "snap_surf_map_component_profile",
				"fixture_dir": str(Path(fixture_dir)),
				"config_path": str(Path(config_path)),
				"stage": int(stage_idx),
				"stage_name": str(stage_cfg_raw.name),
				"sample_level": int(sample_level),
				"repeats": int(repeats),
				"rows": rows,
				"profiler_trace": None if profiler_trace is None else str(profiler_trace),
			}
			_write_json(out / "profile_components.json", payload)
			return rows
		_run_fixture_stage_training_step_loop(
			stage=stage_cfg_raw,
			stage_cfg=stage_cfg,
			fixture=fixture,
			affine=affine,
			global_model=global_model,
			level=level,
			sample_level=sample_level,
			stage_active_quad=stage_active_quad,
			stage_z_lift=stage_z_lift,
			params=params,
			seed_hw=seed_hw,
			station_target=station_target,
			w_station=w_station,
		)
	raise ValueError("no map_surf_ms stage matched component profile selector")


__all__ = [name for name in globals() if not name.startswith("__")]
