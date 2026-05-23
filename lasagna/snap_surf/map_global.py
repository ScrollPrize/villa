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
from .legacy import _closest_point_uv_on_model_quad
from .map_fixture_io import MapFixture, _float_tif, _write_json, load_map_fixture
from .map_objective import _map_init_objective
from .map_pyramid import (
	_map_init_dyadic_level_shape,
	_map_init_dyadic_strides,
	_map_init_integrate_dyadic_uv_pyramid,
	_map_init_uv_pyr_from_dense,
)
from .tensor import _quad_valid_at_coords, _sample_surface_grid


@dataclass(frozen=True)
class GlobalMapStageConfig:
	steps: int = 0
	lr: float = 0.05
	params: tuple[str, ...] = ()
	min_scaledown: int = 0
	w_fac: float = 1.0
	args: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GlobalMapConfig:
	base: dict[str, Any] = field(default_factory=dict)
	stages: tuple[GlobalMapStageConfig, ...] = ()


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
	) -> None:
		super().__init__()
		self.map_uv_ms = _map_init_uv_pyr_from_dense(full_uv.detach(), levels=int(levels), factor=int(factor))

	def forward(self, *, active_level: int = 0) -> torch.Tensor:
		return _map_init_integrate_dyadic_uv_pyramid(list(self.map_uv_ms), active_level=int(active_level))


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
		if not isinstance(item, dict):
			raise ValueError(f"global map stage {i} must be an object")
		params = item.get("params", ())
		if isinstance(params, str):
			params_t = (params,)
		elif isinstance(params, list):
			params_t = tuple(str(v) for v in params)
		else:
			raise ValueError(f"global map stage {i} params must be a string or list")
		args = item.get("args", {})
		if args is None:
			args = {}
		if not isinstance(args, dict):
			raise ValueError(f"global map stage {i} args must be an object")
		stages.append(
			GlobalMapStageConfig(
				steps=max(0, int(item.get("steps", 0))),
				lr=float(item.get("lr", 0.05)),
				params=params_t,
				min_scaledown=max(0, int(item.get("min_scaledown", 0))),
				w_fac=float(item.get("w_fac", 1.0)),
				args=dict(args),
			)
		)
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
) -> list[tuple[int, float, float, torch.Tensor]]:
	candidates: list[tuple[int, float, float, torch.Tensor]] = []
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
) -> tuple[float, torch.Tensor]:
	with torch.no_grad():
		affine_model.affine.copy_(candidate)
	opt = torch.optim.Adam([affine_model.affine], lr=float(lr)) if int(steps) > 0 else None
	last_loss = float("inf")
	for _ in range(max(1, int(steps))):
		if opt is not None:
			opt.zero_grad(set_to_none=True)
		uv = affine_model()
		loss, _terms = _objective_for_uv(uv=uv, fixture=fixture, cfg=stage_cfg, level=0)
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
	ProgressColumn("iloss", "iloss", "initial objective", min_width=7),
	ProgressColumn("loss", "loss", "final objective", min_width=7),
	ProgressColumn("dist", "dst", "distance loss", min_width=7),
	ProgressColumn("vec", "vec", "vector-normal loss", min_width=7),
	ProgressColumn("norm", "nrm", "surface-normal loss", min_width=7),
	ProgressColumn("smooth", "smo", "uv smooth loss", min_width=7),
	ProgressColumn("bend", "bnd", "uv bend loss", min_width=7),
	ProgressColumn("jac", "jac", "jacobian loss", min_width=7),
	ProgressColumn("metric_smooth", "met", "model metric loss", min_width=7),
	ProgressColumn("area_smooth", "ar", "external physical area loss", min_width=7),
	ProgressColumn("prior", "pri", "dense prior loss", min_width=7),
	ProgressColumn("station", "stat", "station loss", min_width=7),
	ProgressColumn("avgd", "avgd", "avg fixture model quad distance", min_width=7),
	ProgressColumn("maxd", "maxd", "max fixture model quad distance", min_width=7),
	ProgressColumn("smp", "smp", "fixture comparison samples", min_width=6),
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
	values["smp"] = "1000000"
	return progress_widths(_AFFINE_DIAG_COLUMNS, values)


def _affine_diag_values(
	*,
	idx: int | str,
	rot: float | None,
	scale: float | None,
	initial_loss: float | None,
	final_loss: float,
	terms: dict[str, torch.Tensor],
	err: dict[str, float],
	affine_tensor: torch.Tensor,
) -> dict[str, str]:
	aff = affine_tensor.detach().cpu()
	return {
		"idx": str(idx),
		"rot": "" if rot is None else format_progress_value(float(rot)),
		"scl": "" if scale is None else format_progress_value(float(scale)),
		"iloss": "" if initial_loss is None else format_progress_value(float(initial_loss)),
		"loss": format_progress_value(float(final_loss)),
		"dist": format_progress_value(_global_term_value(terms, "dist")),
		"vec": format_progress_value(_global_term_value(terms, "vec")),
		"norm": format_progress_value(_global_term_value(terms, "norm")),
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
		"a00": format_progress_value(float(aff[0, 0])),
		"a01": format_progress_value(float(aff[0, 1])),
		"a02": format_progress_value(float(aff[0, 2])),
		"a10": format_progress_value(float(aff[1, 0])),
		"a11": format_progress_value(float(aff[1, 1])),
		"a12": format_progress_value(float(aff[1, 2])),
	}


def _print_affine_diag_header(label: str, widths: dict[str, int]) -> None:
	print_progress_legend(
		prefix=f"[snap_surf.map_global] {label}",
		items=[(col.label, col.description) for col in _AFFINE_DIAG_COLUMNS],
	)
	print(progress_header(_AFFINE_DIAG_COLUMNS, widths), flush=True)


def _print_affine_diag_row(widths: dict[str, int], values: dict[str, str]) -> None:
	print(progress_row(_AFFINE_DIAG_COLUMNS, widths, values), flush=True)


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
) -> None:
	cfg = _affine_multistart_cfg(cfg_global, stage)
	if not bool(cfg.get("enabled", False)):
		return
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
	if not candidates:
		return
	start_affine = affine.affine.detach().clone()
	best_loss = float("inf")
	best_affine = start_affine
	widths = _affine_diag_widths()
	_print_affine_diag_header("affine multistart", widths)
	for cand_idx, rot, scale, cand in candidates:
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


def _stage_loss_cfg(base_cfg: SnapSurfConfig, stage: GlobalMapStageConfig) -> SnapSurfConfig:
	mi = base_cfg.map_init
	scale = float(stage.w_fac)
	return replace(
		base_cfg,
		map_init=replace(
			mi,
			w_dist=float(mi.w_dist) * scale,
			w_vec_normal=float(mi.w_vec_normal) * scale,
			w_surface_normal=float(mi.w_surface_normal) * scale,
			w_smooth=float(mi.w_smooth) * scale,
			w_bend=float(mi.w_bend) * scale,
			w_jac=float(mi.w_jac) * scale,
			w_metric_smooth=float(mi.w_metric_smooth) * scale,
			w_area_smooth=float(mi.w_area_smooth) * scale,
		),
	)


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


def _objective_for_uv(
	*,
	uv: torch.Tensor,
	fixture: MapFixture,
	cfg: SnapSurfConfig,
	level: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
	active = _level_active_quad(_full_active_quad(fixture), int(level))
	ext_coords = None if int(level) == 0 else _level_coords(fixture.ext_xyz.shape[:2], int(level), uv)
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
		normal_sign=int(fixture.metadata.get("normal_sign", 1) or 1),
		orientation_sign=int(fixture.metadata.get("orientation_sign", 1) or 1),
		cfg=cfg,
		allow_partial_model_samples=True,
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
	ref_pos, ref_valid = _uv_model_positions(fixture.reference_uv.to(device=uv.device, dtype=uv.dtype), fixture)
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


def _write_stage_objs(
	out_root: Path,
	*,
	stage_idx: int,
	stage: GlobalMapStageConfig,
	uv: torch.Tensor,
	fixture: MapFixture,
) -> None:
	label = "_".join(stage.params) if stage.params else "noop"
	out = out_root / "objs" / f"stage_{int(stage_idx):03d}_{_debug_obj_safe_label(label)}"
	out.mkdir(parents=True, exist_ok=True)
	_write_obj_mesh_2d(out / "ext_surface.obj", fixture.ext_xyz, fixture.ext_valid)
	_write_obj_mesh_3d_surfaces(out / "model_surface.obj", fixture.model_xyz, fixture.model_valid)
	model_pos, model_ok = _uv_model_positions(uv, fixture)
	ext_ok = fixture.ext_valid.bool() & torch.isfinite(fixture.ext_xyz).all(dim=-1)
	ok = model_ok & ext_ok
	if bool(ok.any().detach().cpu()):
		_write_obj_lines(out / "map_ext_to_model.obj", fixture.ext_xyz[ok], model_pos[ok], label="global_map_ext_to_model")
	else:
		empty = fixture.model_xyz.new_empty(0, 3)
		_write_obj_lines(out / "map_ext_to_model.obj", empty, empty, label="global_map_ext_to_model")
	_write_obj_points(out / "map_valid_ext_points.obj", fixture.ext_xyz, ok, label="global_map_valid_ext_points")
	_write_json(
		out / "meta.json",
		{
			"stage": int(stage_idx),
			"params": list(stage.params),
			"map_ext_to_model": "map_ext_to_model.obj",
			"valid_vectors": int(ok.sum().detach().cpu()),
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
	ProgressColumn("params", "params", "optimized params", min_width=6),
	ProgressColumn("lvl", "lvl", "minimum trained pyramid level"),
	ProgressColumn("loss", "loss", "objective", min_width=7),
	ProgressColumn("dist", "dst", "distance loss", min_width=7),
	ProgressColumn("vec", "vec", "vector-normal loss", min_width=7),
	ProgressColumn("norm", "nrm", "surface-normal loss", min_width=7),
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
	ProgressColumn("smp", "smp", "fixture comparison samples", min_width=6),
)


def _global_progress_widths(cfg: GlobalMapConfig) -> dict[str, int]:
	values: dict[str, str] = {}
	stage_count = max(1, len(cfg.stages))
	max_steps = max((int(stage.steps) for stage in cfg.stages), default=1)
	values["stg"] = str(stage_count - 1)
	values["it"] = f"{max_steps}/{max_steps}"
	values["params"] = max((",".join(stage.params) or "-" for stage in cfg.stages), key=len, default="-")
	values["lvl"] = str(max((int(stage.min_scaledown) for stage in cfg.stages), default=0))
	values["loss"] = "-1.0e+99"
	for key in ("dist", "vec", "norm", "smooth", "bend", "jac", "metric_smooth", "area_smooth", "prior", "station"):
		values[key] = "-1.0e+99"
	values["it_s"] = "-1.0e+99"
	values["avgd"] = "-1.0e+99"
	values["maxd"] = "-1.0e+99"
	values["smp"] = "1000000"
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
	params: tuple[str, ...],
	level: int,
	loss: float,
	terms: dict[str, torch.Tensor],
	it_s: float | None,
	err: dict[str, float],
) -> None:
	values = {
		"stg": str(int(stage_idx)),
		"it": str(iter_label),
		"params": ",".join(params) or "-",
		"lvl": str(int(level)),
		"loss": format_progress_value(float(loss)),
		"dist": format_progress_value(_global_term_value(terms, "dist")),
		"vec": format_progress_value(_global_term_value(terms, "vec")),
		"norm": format_progress_value(_global_term_value(terms, "norm")),
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
		"smp": str(int(err["mapping_error_samples"])),
	}
	if int(row_idx) % 20 == 0:
		print_progress_legend(
			prefix="[snap_surf.map_global]",
			items=[(col.label, col.description) for col in _GLOBAL_PROGRESS_COLUMNS],
		)
		print(progress_header(_GLOBAL_PROGRESS_COLUMNS, widths), flush=True)
	print(progress_row(_GLOBAL_PROGRESS_COLUMNS, widths, values), flush=True)


def optimize_fixture(
	fixture_dir: str | Path,
	config_path: str | Path,
	*,
	out_dir: str | Path,
	device: torch.device | str = "cpu",
) -> dict[str, Any]:
	device_t = torch.device(str(device))
	fixture = load_map_fixture(fixture_dir, device=device_t)
	cfg_global = parse_global_map_config(config_path)
	base_cfg = snap_surf_config_from_global_config(cfg_global)
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
		w_station = float(stage.args.get("map_station_t", stage.args.get("w_station_t", cfg_global.base.get("map_station_t", 0.0))))
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
		opt = torch.optim.Adam(params, lr=float(stage.lr))
		status_interval = max(0, int(stage.args.get("status_interval", stage.args.get("debug_print_interval", 100))))
		last_status_time: float | None = None
		last_status_step = 0
		for step in range(int(stage.steps)):
			opt.zero_grad(set_to_none=True)
			if "map_uv_ms" in stage.params and global_model is not None:
				uv = global_model(active_level=0)
			else:
				uv = affine()
			loss, terms = _objective_for_uv(uv=uv, fixture=fixture, cfg=stage_cfg, level=0)
			station_raw = loss.new_zeros(())
			if w_station > 0.0:
				station_raw = _station_loss(uv, seed_hw, station_target)
				loss = loss + w_station * station_raw
			loss.backward()
			opt.step()
			progress_terms = dict(terms)
			progress_terms["station"] = station_raw.detach()
			with torch.no_grad():
				if "map_uv_ms" in stage.params and global_model is not None:
					uv_after = global_model(active_level=0)
				else:
					uv_after = affine()
			step1 = step + 1
			status_due = (
				step == 0 or
				step1 == int(stage.steps) or
				(status_interval > 0 and (step1 % status_interval) == 0)
			)
			if status_due:
				err = _fixture_mapping_error(uv_after.detach(), fixture)
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
					params=stage.params,
					level=level,
					loss=float(loss.detach().cpu()),
					terms=progress_terms,
					it_s=it_s,
					err=err,
				)
				progress_rows += 1
			if step == int(stage.steps) - 1:
				err = _fixture_mapping_error(uv_after.detach(), fixture)
				history.append({
					"stage": stage_idx,
					"step": step,
					"params": list(stage.params),
					"train_min_level": level,
					"loss": float(loss.detach().cpu()),
					**err,
					"terms": {k: float(v.detach().cpu()) for k, v in progress_terms.items() if v.ndim == 0},
				})
		if "map_uv_ms" in stage.params and global_model is not None:
			stage_uv = global_model(active_level=0).detach()
		else:
			stage_uv = affine().detach()
		_write_stage_objs(out, stage_idx=stage_idx, stage=stage, uv=stage_uv, fixture=fixture)

	final_uv = global_model(active_level=0).detach() if global_model is not None else affine().detach()
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


__all__ = [name for name in globals() if not name.startswith("__")]
