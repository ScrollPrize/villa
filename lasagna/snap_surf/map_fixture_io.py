from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
import torch

from .config import SnapSurfConfig, _parse_map_init_config
from .debug_obj import _write_map_init_objs_to_dir
from .map_pyramid import _map_init_active_vertex_mask, _map_init_external_quad_valid
from .state import _SurfaceState

MAP_FIXTURE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class MapFixture:
	root: Path
	metadata: dict[str, Any]
	ext_xyz: torch.Tensor
	ext_valid: torch.Tensor
	ext_quad_valid: torch.Tensor
	ext_normals: torch.Tensor
	model_xyz: torch.Tensor
	model_valid: torch.Tensor
	model_normals: torch.Tensor
	reference_uv: torch.Tensor
	reference_active_quad: torch.Tensor
	reference_blocked_quad: torch.Tensor


def _jsonable(value: Any) -> Any:
	if is_dataclass(value):
		return _jsonable(asdict(value))
	if isinstance(value, Path):
		return str(value)
	if isinstance(value, torch.Tensor):
		return _jsonable(value.detach().cpu().tolist())
	if isinstance(value, np.ndarray):
		return _jsonable(value.tolist())
	if isinstance(value, np.generic):
		return _jsonable(value.item())
	if isinstance(value, dict):
		return {str(k): _jsonable(v) for k, v in value.items()}
	if isinstance(value, (list, tuple)):
		return [_jsonable(v) for v in value]
	if isinstance(value, bool) or value is None or isinstance(value, str):
		return value
	if isinstance(value, int):
		return int(value)
	if isinstance(value, float):
		return float(value) if math.isfinite(float(value)) else None
	return str(value)


def _write_json(path: Path, data: dict[str, Any]) -> None:
	path.write_text(json.dumps(_jsonable(data), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _float_tif(path: Path, tensor: torch.Tensor) -> None:
	arr = tensor.detach().cpu().numpy().astype(np.float32, copy=False)
	tifffile.imwrite(str(path), arr, photometric="minisblack")


def _mask_tif(path: Path, tensor: torch.Tensor) -> None:
	arr = tensor.detach().cpu().bool().numpy().astype(np.uint8, copy=False)
	tifffile.imwrite(str(path), arr, photometric="minisblack")


def _read_float_tif(path: Path, *, device: torch.device | str) -> torch.Tensor:
	arr = np.array(tifffile.imread(str(path)), dtype=np.float32, copy=True)
	return torch.from_numpy(arr).to(device=device)


def _read_mask_tif(path: Path, *, device: torch.device | str) -> torch.Tensor:
	arr = np.array(tifffile.imread(str(path)), copy=True)
	return torch.from_numpy(arr != 0).to(device=device)


def _write_vector_dir(path: Path, tensor: torch.Tensor, valid: torch.Tensor, *, meta: dict[str, Any]) -> None:
	if int(tensor.shape[-1]) != 3:
		raise ValueError(f"expected vector tensor with final dimension 3, got {tuple(tensor.shape)}")
	path.mkdir(parents=True, exist_ok=True)
	_float_tif(path / "x.tif", tensor[..., 0])
	_float_tif(path / "y.tif", tensor[..., 1])
	_float_tif(path / "z.tif", tensor[..., 2])
	_mask_tif(path / "valid.tif", valid)
	_write_json(path / "meta.json", meta)


def _read_vector_dir(path: Path, *, device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
	x = _read_float_tif(path / "x.tif", device=device)
	y = _read_float_tif(path / "y.tif", device=device)
	z = _read_float_tif(path / "z.tif", device=device)
	if tuple(x.shape) != tuple(y.shape) or tuple(x.shape) != tuple(z.shape):
		raise ValueError(f"fixture vector shape mismatch in {path}: x={tuple(x.shape)} y={tuple(y.shape)} z={tuple(z.shape)}")
	valid = _read_mask_tif(path / "valid.tif", device=device)
	meta_path = path / "meta.json"
	meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
	return torch.stack([x, y, z], dim=-1), valid.bool(), meta


def map_fixture_surface_dir(root: str | Path, surface_index: int, surface_count: int) -> Path:
	base = Path(root)
	if int(surface_count) <= 1:
		return base
	return base / f"surf{int(surface_index):03d}"


def map_tensors_from_state(state: _SurfaceState) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
	mi = state.map_init
	if mi.uv is None or mi.active_quad is None:
		raise RuntimeError("map fixture export requires initialized map-init uv and active_quad")
	uv = mi.uv.detach()
	active_quad = mi.active_quad.detach().bool()
	if mi.blocked_quad is None:
		blocked_quad = torch.zeros_like(active_quad)
	else:
		blocked_quad = mi.blocked_quad.detach().bool()
	active_vertex = _map_init_active_vertex_mask(active_quad, tuple(int(v) for v in uv.shape[:2]))
	finite_vertex = active_vertex & torch.isfinite(uv).all(dim=-1)
	uv_out = torch.where(finite_vertex.unsqueeze(-1), uv, torch.full_like(uv, float("nan")))
	return uv_out, active_quad, blocked_quad, finite_vertex


def write_map_outputs(path: str | Path, state: _SurfaceState, *, meta: dict[str, Any] | None = None) -> dict[str, int]:
	out = Path(path)
	out.mkdir(parents=True, exist_ok=True)
	uv, active_quad, blocked_quad, active_vertex = map_tensors_from_state(state)
	_float_tif(out / "model_x.tif", uv[..., 1])
	_float_tif(out / "model_y.tif", uv[..., 0])
	_mask_tif(out / "active_quad.tif", active_quad)
	_mask_tif(out / "blocked_quad.tif", blocked_quad)
	if meta is not None:
		_write_json(out / "meta.json", meta)
	return {
		"active_vertices": int(active_vertex.sum().detach().cpu()),
		"active_quads": int(active_quad.sum().detach().cpu()),
		"blocked_quads": int(blocked_quad.sum().detach().cpu()),
	}


def fixture_metadata(
	*,
	cfg: SnapSurfConfig,
	state: _SurfaceState,
	model_xyz: torch.Tensor,
	ext_xyz: torch.Tensor,
	seed_xyz: tuple[float, float, float],
	surface_index: int,
	surface_count: int,
	step: int | None,
	stats: dict[str, float] | None,
) -> dict[str, Any]:
	mi = state.map_init
	return {
		"schema_version": MAP_FIXTURE_SCHEMA_VERSION,
		"kind": "snap_surf_map_fixture",
		"seed_xyz": [float(v) for v in seed_xyz],
		"step": None if step is None else int(step),
		"surface_index": int(surface_index),
		"surface_count": int(surface_count),
		"ext_shape": [int(v) for v in ext_xyz.shape[:2]],
		"model_shape": [int(v) for v in model_xyz.shape[:3]],
		"model_depth": None if mi.model_depth is None else int(mi.model_depth),
		"snap_surf_config": asdict(cfg),
		"map_init_config": asdict(cfg.map_init),
		"stats": dict(stats or {}),
		"sign": int(mi.sign),
		"sign_semantics": "model_normal_alignment",
		"seed_ext_sample_hw": None if mi.seed_ext_sample_hw is None else [int(v) for v in mi.seed_ext_sample_hw],
		"seed_model_quad": None if mi.seed_model_quad is None else [int(v) for v in mi.seed_model_quad],
		"seed_model_distance": float(mi.seed_model_distance),
		"seed_ext_distance": float(mi.seed_ext_distance),
		"scale": {
			"scale_level": int(mi.scale_level),
			"target_scale_level": int(mi.target_scale_level),
			"scale_strides": [int(v) for v in mi.scale_strides],
			"scale_levels_used": int(mi.scale_levels_used),
			"current_stride": int(mi.current_stride()),
		},
	}


def export_map_fixture(
	out_dir: str | Path,
	*,
	cfg: SnapSurfConfig,
	state: _SurfaceState,
	model_xyz: torch.Tensor,
	model_valid: torch.Tensor,
	model_normals: torch.Tensor,
	ext_xyz: torch.Tensor,
	ext_valid: torch.Tensor,
	ext_normals: torch.Tensor,
	ext_quad_valid: torch.Tensor | None,
	seed_xyz: tuple[float, float, float],
	surface_index: int,
	surface_count: int,
	step: int | None = None,
	stats: dict[str, float] | None = None,
	export_objs: bool = True,
	write_geometry: bool = True,
) -> dict[str, Any]:
	out = Path(out_dir)
	out.mkdir(parents=True, exist_ok=True)
	meta = fixture_metadata(
		cfg=cfg,
		state=state,
		model_xyz=model_xyz,
		ext_xyz=ext_xyz,
		seed_xyz=seed_xyz,
		surface_index=surface_index,
		surface_count=surface_count,
		step=step,
		stats=stats,
	)
	map_counts = write_map_outputs(out / "map", state, meta={"model_depth": meta["model_depth"]})
	meta["map_counts"] = map_counts
	if write_geometry:
		ext_quad = (
			_map_init_external_quad_valid(ext_valid.bool(), ext_quad_valid)
			if ext_quad_valid is not None else
			_map_init_external_quad_valid(ext_valid.bool(), None)
		)
		_write_vector_dir(
			out / "ext_surface",
			ext_xyz.detach(),
			ext_valid.detach().bool(),
			meta={"kind": "external_surface", "shape": [int(v) for v in ext_xyz.shape[:2]]},
		)
		_mask_tif(out / "ext_surface" / "quad_valid.tif", ext_quad)
		_write_vector_dir(
			out / "model_stack",
			model_xyz.detach(),
			model_valid.detach().bool(),
			meta={"kind": "model_stack", "shape": [int(v) for v in model_xyz.shape[:3]]},
		)
		_write_vector_dir(
			out / "ext_normals",
			ext_normals.detach(),
			(ext_valid.detach().bool() & torch.isfinite(ext_normals.detach()).all(dim=-1)),
			meta={"kind": "external_normals", "shape": [int(v) for v in ext_normals.shape[:2]]},
		)
		_write_vector_dir(
			out / "model_normals",
			model_normals.detach(),
			(model_valid.detach().bool() & torch.isfinite(model_normals.detach()).all(dim=-1)),
			meta={"kind": "model_normals", "shape": [int(v) for v in model_normals.shape[:3]]},
		)
	if export_objs:
		_write_map_init_objs_to_dir(
			out_dir=out / "objs",
			cfg=cfg,
			surface_index=surface_index,
			surface_count=surface_count,
			model_xyz=model_xyz,
			model_valid=model_valid,
			ext_xyz=ext_xyz,
			ext_valid=ext_valid,
			state=state,
		)
	_write_json(out / "fixture.json", meta)
	return meta


def load_map_fixture(path: str | Path, *, device: torch.device | str = "cpu") -> MapFixture:
	root = Path(path)
	meta = json.loads((root / "fixture.json").read_text(encoding="utf-8"))
	ext_xyz, ext_valid, _ext_meta = _read_vector_dir(root / "ext_surface", device=device)
	quad_path = root / "ext_surface" / "quad_valid.tif"
	ext_quad_valid = (
		_read_mask_tif(quad_path, device=device)
		if quad_path.exists() else
		_map_init_external_quad_valid(ext_valid.bool(), None)
	)
	model_xyz, model_valid, _model_meta = _read_vector_dir(root / "model_stack", device=device)
	ext_normals, _ext_normals_valid, _ = _read_vector_dir(root / "ext_normals", device=device)
	model_normals, _model_normals_valid, _ = _read_vector_dir(root / "model_normals", device=device)
	model_x = _read_float_tif(root / "map" / "model_x.tif", device=device)
	model_y = _read_float_tif(root / "map" / "model_y.tif", device=device)
	if tuple(model_x.shape) != tuple(model_y.shape):
		raise ValueError(f"fixture map shape mismatch: model_x={tuple(model_x.shape)} model_y={tuple(model_y.shape)}")
	reference_uv = torch.stack([model_y, model_x], dim=-1)
	reference_active_quad = _read_mask_tif(root / "map" / "active_quad.tif", device=device)
	reference_blocked_quad = _read_mask_tif(root / "map" / "blocked_quad.tif", device=device)
	return MapFixture(
		root=root,
		metadata=meta,
		ext_xyz=ext_xyz,
		ext_valid=ext_valid.bool(),
		ext_quad_valid=ext_quad_valid.bool(),
		ext_normals=ext_normals,
		model_xyz=model_xyz,
		model_valid=model_valid.bool(),
		model_normals=model_normals,
		reference_uv=reference_uv,
		reference_active_quad=reference_active_quad.bool(),
		reference_blocked_quad=reference_blocked_quad.bool(),
	)


def snap_surf_config_from_fixture(metadata: dict[str, Any]) -> SnapSurfConfig:
	raw = dict(metadata.get("snap_surf_config") or {})
	raw_map = raw.get("map_init", metadata.get("map_init_config") or {})
	map_cfg = _parse_map_init_config(raw_map)
	defaults = SnapSurfConfig()
	kwargs: dict[str, Any] = {}
	for name in SnapSurfConfig.__dataclass_fields__:
		if name == "map_init":
			continue
		kwargs[name] = raw.get(name, getattr(defaults, name))
	kwargs["map_init"] = map_cfg
	return SnapSurfConfig(**kwargs)


def compare_map_tensors(
	*,
	reference_uv: torch.Tensor,
	reference_active_quad: torch.Tensor,
	reference_blocked_quad: torch.Tensor,
	rerun_uv: torch.Tensor,
	rerun_active_quad: torch.Tensor,
	rerun_blocked_quad: torch.Tensor,
) -> dict[str, Any]:
	if tuple(reference_uv.shape) != tuple(rerun_uv.shape):
		raise ValueError(f"map uv shape mismatch: reference={tuple(reference_uv.shape)} rerun={tuple(rerun_uv.shape)}")
	if tuple(reference_active_quad.shape) != tuple(rerun_active_quad.shape):
		raise ValueError(
			f"active quad shape mismatch: reference={tuple(reference_active_quad.shape)} rerun={tuple(rerun_active_quad.shape)}"
		)
	ref_vertex = _map_init_active_vertex_mask(reference_active_quad.bool(), tuple(int(v) for v in reference_uv.shape[:2]))
	rerun_vertex = _map_init_active_vertex_mask(rerun_active_quad.bool(), tuple(int(v) for v in rerun_uv.shape[:2]))
	ref_finite = ref_vertex & torch.isfinite(reference_uv).all(dim=-1)
	rerun_finite = rerun_vertex & torch.isfinite(rerun_uv).all(dim=-1)
	common = ref_finite & rerun_finite
	delta = rerun_uv - reference_uv
	if bool(common.any().detach().cpu()):
		d_common = delta[common]
		abs_common = d_common.abs()
		l2_common = d_common.square().sum(dim=-1).sqrt()
		max_abs = abs_common.max(dim=0).values
		mean_abs = abs_common.mean(dim=0)
		rms = d_common.square().mean(dim=0).sqrt()
		max_l2 = l2_common.max()
	else:
		max_abs = reference_uv.new_zeros(2)
		mean_abs = reference_uv.new_zeros(2)
		rms = reference_uv.new_zeros(2)
		max_l2 = reference_uv.new_zeros(())
	active_diff = reference_active_quad.bool() ^ rerun_active_quad.bool()
	blocked_diff = reference_blocked_quad.bool() ^ rerun_blocked_quad.bool()
	return {
		"reference_active_quads": int(reference_active_quad.bool().sum().detach().cpu()),
		"rerun_active_quads": int(rerun_active_quad.bool().sum().detach().cpu()),
		"active_quad_diff": int(active_diff.sum().detach().cpu()),
		"active_quad_equal": bool(not bool(active_diff.any().detach().cpu())),
		"reference_blocked_quads": int(reference_blocked_quad.bool().sum().detach().cpu()),
		"rerun_blocked_quads": int(rerun_blocked_quad.bool().sum().detach().cpu()),
		"blocked_quad_diff": int(blocked_diff.sum().detach().cpu()),
		"blocked_quad_equal": bool(not bool(blocked_diff.any().detach().cpu())),
		"reference_finite_vertices": int(ref_finite.sum().detach().cpu()),
		"rerun_finite_vertices": int(rerun_finite.sum().detach().cpu()),
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
