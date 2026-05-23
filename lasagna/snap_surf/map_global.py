from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import json
from pathlib import Path
from typing import Any

import torch
from .config import SnapSurfConfig, SnapSurfMapInitConfig, _parse_map_init_config
from .debug_obj import _debug_obj_safe_label, _write_obj_lines, _write_obj_mesh_2d, _write_obj_mesh_3d_surfaces, _write_obj_points
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
	affine = AffineMapModel(ext_shape=tuple(int(v) for v in fixture.ext_xyz.shape[:2]), device=device_t, dtype=dtype)
	global_model: GlobalMapModel | None = None
	seed_hw = _seed_ext_hw(fixture.metadata, tuple(int(v) for v in fixture.ext_xyz.shape[:2]), device=device_t, dtype=dtype)
	station_target = affine.eval_at(seed_hw).detach()
	history: list[dict[str, Any]] = []
	full_active = _full_active_quad(fixture)
	out = Path(out_dir)
	out.mkdir(parents=True, exist_ok=True)
	print("stage step params train_min loss avg_map_dist max_map_dist samples", flush=True)

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
		opt = torch.optim.Adam(params, lr=float(stage.lr))
		for step in range(int(stage.steps)):
			opt.zero_grad(set_to_none=True)
			if "map_uv_ms" in stage.params and global_model is not None:
				uv = global_model(active_level=0)
			else:
				uv = affine()
			loss, terms = _objective_for_uv(uv=uv, fixture=fixture, cfg=stage_cfg, level=0)
			w_station = float(stage.args.get("map_station_t", stage.args.get("w_station_t", cfg_global.base.get("map_station_t", 0.0))))
			if w_station > 0.0:
				loss = loss + w_station * _station_loss(uv, seed_hw, station_target)
			loss.backward()
			opt.step()
			err = _fixture_mapping_error(uv.detach(), fixture)
			print(
				f"{stage_idx:5d} {step:4d} {','.join(stage.params) or '-':>10s} "
				f"{level:9d} {float(loss.detach().cpu()):.9g} "
				f"{err['avg_model_quad_distance']:.9g} "
				f"{err['max_model_quad_distance']:.9g} "
				f"{int(err['mapping_error_samples'])}",
				flush=True,
			)
			if step == int(stage.steps) - 1:
				history.append({
					"stage": stage_idx,
					"step": step,
					"params": list(stage.params),
					"train_min_level": level,
					"loss": float(loss.detach().cpu()),
					**err,
					"terms": {k: float(v.detach().cpu()) for k, v in terms.items() if v.ndim == 0},
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
