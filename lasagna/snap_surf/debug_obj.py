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

from .config import SnapSurfConfig, _map_init_log
from .state import _SurfaceState
from .tensor import *
from .map_pyramid import _map_init_quad_sample_tensors, _map_init_coords3

_debug_step: int | None = None
_debug_label: str | None = None

def set_debug_step(step: int | None, *, label: str | None = None) -> None:
	global _debug_step, _debug_label
	_debug_step = None if step is None else int(step)
	_debug_label = None if label is None else str(label)

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

__all__ = [name for name in globals() if not name.startswith('__')]
