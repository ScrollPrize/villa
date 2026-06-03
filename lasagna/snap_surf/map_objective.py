from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
import time
from typing import Any

import torch
import torch.nn.functional as F

import model as fit_model
import opt_loss_station
import opt_loss_winding_density

from .config import SnapSurfConfig, SnapSurfMapInitConfig
from .tensor import *
from .legacy import _huber
from .map_pyramid import *

@dataclass(frozen=True)
class _MapInitSurfaceSamplePlan:
	grid_shape: tuple[int, ...]
	coords_shape: tuple[int, ...]
	in_bounds: torch.Tensor
	idx: tuple[torch.Tensor, ...]
	weights: tuple[torch.Tensor, ...]

@dataclass(frozen=True)
class _MapInitEvennessExternalContext:
	finite_ext: torch.Tensor
	static_quad: torch.Tensor
	ext_len_h: torch.Tensor | None = None
	ext_len_w: torch.Tensor | None = None
	ext_len_h_valid: torch.Tensor | None = None
	ext_len_w_valid: torch.Tensor | None = None
	metric_h_pair_down: torch.Tensor | None = None
	metric_h_pair_right: torch.Tensor | None = None
	metric_w_pair_down: torch.Tensor | None = None
	metric_w_pair_right: torch.Tensor | None = None
	ext_area: torch.Tensor | None = None
	ext_area_valid: torch.Tensor | None = None
	area_pair_down: torch.Tensor | None = None
	area_pair_right: torch.Tensor | None = None

_MAP_INIT_COMPILED_FN_CACHE: dict[tuple[Any, ...], Any] = {}

def _map_init_tensor_compile_signature(t: torch.Tensor) -> tuple[Any, ...]:
	return (
		str(t.device),
		str(t.dtype),
		tuple(int(v) for v in t.shape),
		bool(t.requires_grad),
	)

def _map_init_compile_signature(args: tuple[Any, ...]) -> tuple[Any, ...]:
	sig: list[tuple[Any, ...]] = []
	for value in args:
		if torch.is_tensor(value):
			sig.append(_map_init_tensor_compile_signature(value))
	return tuple(sig)

def _map_init_compile_enabled(cfg: SnapSurfMapInitConfig, *, need_stats: bool) -> bool:
	return bool(cfg.compile_objective) and not bool(need_stats) and hasattr(torch, "compile")

def _map_init_call_compiled(
	name: str,
	fn,
	args: tuple[Any, ...],
	*,
	mode: str | None = None,
	fullgraph: bool = False,
	dynamic: bool = False,
):
	tensor_sig = _map_init_compile_signature(args)
	if not tensor_sig:
		return fn(*args)
	device = tensor_sig[0][0]
	dtype = tensor_sig[0][1]
	key = (
		str(name),
		device,
		dtype,
		tensor_sig,
		None if mode in {None, "", "default"} else str(mode),
		bool(fullgraph),
		bool(dynamic),
	)
	compiled = _MAP_INIT_COMPILED_FN_CACHE.get(key)
	if compiled is None:
		compile_kwargs: dict[str, Any] = {
			"fullgraph": bool(fullgraph),
			"dynamic": bool(dynamic),
		}
		if mode not in {None, "", "default"}:
			compile_kwargs["mode"] = str(mode)
		compiled = torch.compile(fn, **compile_kwargs)
		_MAP_INIT_COMPILED_FN_CACHE[key] = compiled
	return compiled(*args)

def _principal_angle_delta(to_angle: torch.Tensor, from_angle: torch.Tensor) -> torch.Tensor:
	two_pi = 2.0 * math.pi
	return torch.remainder(to_angle - from_angle + math.pi, two_pi) - math.pi

def _map_init_quad_normal_headings(
	normals: torch.Tensor,
	*,
	sign: int = 1,
) -> torch.Tensor:
	sign_f = 1.0 if int(sign) >= 0 else -1.0
	n = normals * sign_f
	if n.ndim == 3:
		q = 0.25 * (n[:-1, :-1] + n[1:, :-1] + n[:-1, 1:] + n[1:, 1:])
	elif n.ndim == 4:
		q = 0.25 * (n[:, :-1, :-1] + n[:, 1:, :-1] + n[:, :-1, 1:] + n[:, 1:, 1:])
	else:
		raise ValueError(f"expected 2D/3D normal grid, got shape {tuple(normals.shape)}")
	return torch.atan2(q[..., 1], q[..., 0])

def _map_init_lifted_z_vertex_heading_field(
	normals: torch.Tensor,
	vertex_valid: torch.Tensor,
	seed_vertex: tuple[int, ...],
	*,
	norm_xy_min: float,
	sign: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
	if vertex_valid.numel() == 0:
		theta_empty = torch.full_like(vertex_valid, float("nan"), dtype=normals.dtype)
		return theta_empty, vertex_valid.bool(), {"valid": 0.0, "invalid": 0.0, "unreachable": 0.0}
	if normals.ndim not in (3, 4):
		raise ValueError(f"expected 2D/3D normal grid, got shape {tuple(normals.shape)}")
	if tuple(normals.shape[:-1]) != tuple(vertex_valid.shape):
		raise ValueError(f"normal/valid shape mismatch: normals={tuple(normals.shape)} valid={tuple(vertex_valid.shape)}")
	sign_f = 1.0 if int(sign) >= 0 else -1.0
	n = normals * sign_f
	phi = torch.atan2(n[..., 1], n[..., 0])
	xy_norm = n[..., :2].norm(dim=-1)
	base_valid = vertex_valid.bool()
	valid = (
		base_valid &
		torch.isfinite(n).all(dim=-1) &
		torch.isfinite(phi) &
		torch.isfinite(xy_norm) &
		(xy_norm >= float(norm_xy_min))
	)
	reached = torch.zeros_like(valid, dtype=torch.bool)
	theta = torch.full_like(phi, float("nan"))
	shape = tuple(int(v) for v in valid.shape)
	if len(seed_vertex) != len(shape) or any(int(seed_vertex[i]) < 0 or int(seed_vertex[i]) >= shape[i] for i in range(len(shape))):
		invalid = int((base_valid & ~valid).sum().detach().cpu())
		valid_count = int(valid.sum().detach().cpu())
		return theta.to(device=normals.device, dtype=normals.dtype), reached.to(device=normals.device), {"valid": 0.0, "invalid": float(invalid), "unreachable": float(valid_count)}
	seed = tuple(int(v) for v in seed_vertex)
	if not bool(valid[seed].detach().cpu()):
		invalid = int((base_valid & ~valid).sum().detach().cpu())
		valid_count = int(valid.sum().detach().cpu())
		return theta.to(device=normals.device, dtype=normals.dtype), reached.to(device=normals.device), {"valid": 0.0, "invalid": float(invalid), "unreachable": float(valid_count)}

	def _shift(src: torch.Tensor, *, axis: int, step: int, fill: float | bool) -> torch.Tensor:
		if src.dtype == torch.bool:
			out = torch.full_like(src, bool(fill))
		else:
			out = torch.full_like(src, float(fill))
		dst_slice = [slice(None)] * src.ndim
		src_slice = [slice(None)] * src.ndim
		if step > 0:
			dst_slice[axis] = slice(1, None)
			src_slice[axis] = slice(None, -1)
		else:
			dst_slice[axis] = slice(None, -1)
			src_slice[axis] = slice(1, None)
		out[tuple(dst_slice)] = src[tuple(src_slice)]
		return out

	reached[seed] = True
	theta[seed] = 0.0
	frontier = torch.zeros_like(valid, dtype=torch.bool)
	frontier[seed] = True
	directions = (
		(len(shape) - 2, -1),
		(len(shape) - 2, 1),
		(len(shape) - 1, -1),
		(len(shape) - 1, 1),
	)
	while bool(frontier.any().detach().cpu()):
		next_frontier = torch.zeros_like(frontier, dtype=torch.bool)
		accepted = torch.zeros_like(frontier, dtype=torch.bool)
		next_theta = theta.clone()
		for axis, step in directions:
			source_frontier = _shift(frontier, axis=axis, step=step, fill=False)
			source_phi = _shift(phi, axis=axis, step=step, fill=float("nan"))
			source_theta = _shift(theta, axis=axis, step=step, fill=float("nan"))
			propose = source_frontier & valid & ~reached & ~accepted
			proposed_theta = source_theta + _principal_angle_delta(phi, source_phi)
			next_theta = torch.where(propose, proposed_theta, next_theta)
			accepted = accepted | propose
			next_frontier = next_frontier | propose
		if not bool(next_frontier.any().detach().cpu()):
			break
		theta = next_theta
		reached = reached | next_frontier
		frontier = next_frontier
	invalid = int((base_valid & ~valid).sum().detach().cpu())
	reachable = int(reached.sum().detach().cpu())
	valid_count = int(valid.sum().detach().cpu())
	theta = torch.where(reached, theta, torch.full_like(theta, float("nan")))
	return theta.to(device=normals.device, dtype=normals.dtype), reached.to(device=normals.device), {
		"valid": float(reachable),
		"invalid": float(invalid),
		"unreachable": float(max(0, valid_count - reachable)),
	}

def _map_init_lifted_z_heading_field(
	normals: torch.Tensor,
	base_quad_valid: torch.Tensor,
	seed_quad: tuple[int, ...],
	*,
	norm_xy_min: float,
	sign: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
	if base_quad_valid.numel() == 0:
		theta_empty = torch.full_like(base_quad_valid, float("nan"), dtype=normals.dtype)
		return theta_empty, base_quad_valid.bool(), {"valid": 0.0, "invalid": 0.0, "unreachable": 0.0}
	sign_f = 1.0 if int(sign) >= 0 else -1.0
	n = normals * sign_f
	if n.ndim == 3:
		q = 0.25 * (n[:-1, :-1] + n[1:, :-1] + n[:-1, 1:] + n[1:, 1:])
	elif n.ndim == 4:
		q = 0.25 * (n[:, :-1, :-1] + n[:, 1:, :-1] + n[:, :-1, 1:] + n[:, 1:, 1:])
	else:
		raise ValueError(f"expected 2D/3D normal grid, got shape {tuple(normals.shape)}")
	phi = torch.atan2(q[..., 1], q[..., 0])
	xy_norm = q[..., :2].norm(dim=-1)
	valid = (
		base_quad_valid.bool() &
		torch.isfinite(q).all(dim=-1) &
		torch.isfinite(phi) &
		torch.isfinite(xy_norm) &
		(xy_norm >= float(norm_xy_min))
	)
	reached = torch.zeros_like(valid, dtype=torch.bool)
	theta = torch.full_like(phi, float("nan"))
	shape = tuple(int(v) for v in valid.shape)
	if len(seed_quad) != len(shape) or any(int(seed_quad[i]) < 0 or int(seed_quad[i]) >= shape[i] for i in range(len(shape))):
		invalid = int((base_quad_valid.bool() & ~valid).sum().detach().cpu())
		reachable = int(reached.sum().detach().cpu())
		valid_count = int(valid.sum().detach().cpu())
		return theta.to(device=normals.device, dtype=normals.dtype), reached.to(device=normals.device), {"valid": float(reachable), "invalid": float(invalid), "unreachable": float(valid_count)}
	seed = tuple(int(v) for v in seed_quad)
	if not bool(valid[seed].detach().cpu()):
		invalid = int((base_quad_valid.bool() & ~valid).sum().detach().cpu())
		valid_count = int(valid.sum().detach().cpu())
		return theta.to(device=normals.device, dtype=normals.dtype), reached.to(device=normals.device), {"valid": 0.0, "invalid": float(invalid), "unreachable": float(valid_count)}

	def _shift(src: torch.Tensor, *, axis: int, step: int, fill: float | bool) -> torch.Tensor:
		if src.dtype == torch.bool:
			out = torch.full_like(src, bool(fill))
		else:
			out = torch.full_like(src, float(fill))
		dst_slice = [slice(None)] * src.ndim
		src_slice = [slice(None)] * src.ndim
		if step > 0:
			dst_slice[axis] = slice(1, None)
			src_slice[axis] = slice(None, -1)
		else:
			dst_slice[axis] = slice(None, -1)
			src_slice[axis] = slice(1, None)
		out[tuple(dst_slice)] = src[tuple(src_slice)]
		return out

	reached[seed] = True
	theta[seed] = 0.0
	frontier = torch.zeros_like(valid, dtype=torch.bool)
	frontier[seed] = True
	directions = (
		(len(shape) - 2, -1),
		(len(shape) - 2, 1),
		(len(shape) - 1, -1),
		(len(shape) - 1, 1),
	)
	while bool(frontier.any().detach().cpu()):
		next_frontier = torch.zeros_like(frontier, dtype=torch.bool)
		accepted = torch.zeros_like(frontier, dtype=torch.bool)
		next_theta = theta.clone()
		for axis, step in directions:
			source_frontier = _shift(frontier, axis=axis, step=step, fill=False)
			source_phi = _shift(phi, axis=axis, step=step, fill=float("nan"))
			source_theta = _shift(theta, axis=axis, step=step, fill=float("nan"))
			propose = source_frontier & valid & ~reached & ~accepted
			proposed_theta = source_theta + _principal_angle_delta(phi, source_phi)
			next_theta = torch.where(propose, proposed_theta, next_theta)
			accepted = accepted | propose
			next_frontier = next_frontier | propose
		if not bool(next_frontier.any().detach().cpu()):
			break
		theta = next_theta
		reached = reached | next_frontier
		frontier = next_frontier
	invalid = int((base_quad_valid.bool() & ~valid).sum().detach().cpu())
	reachable = int(reached.sum().detach().cpu())
	valid_count = int(valid.sum().detach().cpu())
	theta = torch.where(reached, theta, torch.full_like(theta, float("nan")))
	return theta.to(device=normals.device, dtype=normals.dtype), reached.to(device=normals.device), {
		"valid": float(reachable),
		"invalid": float(invalid),
		"unreachable": float(max(0, valid_count - reachable)),
	}

def _map_init_lifted_z_heading_branches(
	normals: torch.Tensor,
	base_quad_valid: torch.Tensor,
	seed_quad: tuple[int, ...],
	*,
	norm_xy_min: float,
	sign: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
	theta, valid, stats = _map_init_lifted_z_heading_field(
		normals,
		base_quad_valid,
		seed_quad,
		norm_xy_min=norm_xy_min,
		sign=sign,
	)
	phi = _map_init_quad_normal_headings(normals, sign=sign).to(device=theta.device, dtype=theta.dtype)
	two_pi = 2.0 * math.pi
	k = torch.where(valid.bool(), torch.round((theta - phi) / two_pi), torch.zeros_like(theta))
	return k, valid, stats

def _map_init_valid_field_values(field: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
	valid_b = valid.to(device=field.device).bool()
	while valid_b.ndim < field.ndim:
		valid_b = valid_b.unsqueeze(-1)
	return torch.where(valid_b & torch.isfinite(field), field, torch.zeros_like(field))

def _map_init_tensor_cache_key(tensor: torch.Tensor) -> tuple[Any, ...]:
	return (
		id(tensor),
		int(tensor.data_ptr()) if tensor.numel() else 0,
		tuple(int(v) for v in tensor.shape),
		tuple(int(v) for v in tensor.stride()),
		str(tensor.dtype),
		str(tensor.device),
		int(tensor.storage_offset()),
		int(getattr(tensor, "_version", 0)),
		bool(tensor.requires_grad),
	)

def _map_init_cache_get(cache: dict[tuple[Any, ...], Any] | None, key: tuple[Any, ...]) -> Any:
	if cache is None:
		return None
	return cache.get(key)

def _map_init_cache_put(cache: dict[tuple[Any, ...], Any] | None, key: tuple[Any, ...], value: Any) -> Any:
	if cache is None:
		return value
	if len(cache) > 64:
		cache.clear()
	cache[key] = value
	return value

def _map_init_packed_pos_norm_values(
	pos: torch.Tensor,
	normals: torch.Tensor,
	valid: torch.Tensor,
) -> torch.Tensor:
	return torch.cat([
		_map_init_valid_field_values(pos, valid),
		_map_init_valid_field_values(normals, valid),
	], dim=-1)

def _map_init_packed_model_vertex_values(
	pos: torch.Tensor,
	normals: torch.Tensor,
	valid: torch.Tensor,
	theta: torch.Tensor | None = None,
	theta_valid: torch.Tensor | None = None,
) -> torch.Tensor:
	packed = _map_init_packed_pos_norm_values(pos, normals, valid)
	if theta is None or theta_valid is None:
		return packed
	theta_valid_b = theta_valid.to(device=theta.device).bool() & torch.isfinite(theta)
	theta_safe = torch.where(theta_valid_b, theta, torch.zeros_like(theta))
	return torch.cat([packed, theta_safe.unsqueeze(-1), theta_valid_b.to(dtype=theta.dtype).unsqueeze(-1)], dim=-1)

def _map_init_split_packed_model_vertex_sample(packed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
	if int(packed.shape[-1]) == 6:
		p, n = _map_init_split_packed_pos_norm(packed)
		return p, n, None, None
	if int(packed.shape[-1]) == 8:
		return packed[..., :3], packed[..., 3:6], packed[..., 6], packed[..., 7]
	raise ValueError(f"expected packed model vertex tensor with 6 or 8 channels, got shape {tuple(packed.shape)}")

def _map_init_packed_pos_norm_cache_key(
	kind: str,
	pos: torch.Tensor,
	normals: torch.Tensor,
	valid: torch.Tensor,
	*,
	prefix: tuple[Any, ...] = (),
) -> tuple[Any, ...]:
	return (
		"packed_pos_norm",
		str(kind),
		*tuple(prefix),
		_map_init_tensor_cache_key(pos),
		_map_init_tensor_cache_key(normals),
		_map_init_tensor_cache_key(valid),
	)

def _map_init_cached_packed_pos_norm_values(
	*,
	kind: str,
	pos: torch.Tensor,
	normals: torch.Tensor,
	valid: torch.Tensor,
	cache: dict[tuple[Any, ...], Any] | None = None,
	prefix: tuple[Any, ...] = (),
) -> torch.Tensor:
	key = _map_init_packed_pos_norm_cache_key(kind, pos, normals, valid, prefix=prefix)
	cached = _map_init_cache_get(cache, key)
	if torch.is_tensor(cached):
		return cached
	packed = _map_init_packed_pos_norm_values(pos, normals, valid)
	if pos.requires_grad or normals.requires_grad:
		return packed
	return _map_init_cache_put(cache, key, packed)

def _map_init_split_packed_pos_norm(packed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
	if int(packed.shape[-1]) != 6:
		raise ValueError(f"expected packed pos/norm tensor with 6 channels, got shape {tuple(packed.shape)}")
	return packed[..., :3], packed[..., 3:6]

def _map_init_surface_sample_plan(
	coords: torch.Tensor,
	grid_shape: tuple[int, ...],
	*,
	quad_scalar_bounds: bool = False,
) -> _MapInitSurfaceSamplePlan:
	if len(grid_shape) == 3:
		D, H, W = int(grid_shape[0]), int(grid_shape[1]), int(grid_shape[2])
		flat = coords.reshape(-1, 3)
		if D <= 0 or H <= 0 or W <= 0:
			empty_i = torch.zeros(flat.shape[0], device=coords.device, dtype=torch.long)
			zero = torch.zeros(flat.shape[0], device=coords.device, dtype=coords.dtype)
			return _MapInitSurfaceSamplePlan(tuple(grid_shape), tuple(coords.shape[:-1]), torch.zeros(flat.shape[0], device=coords.device, dtype=torch.bool), (empty_i, empty_i, empty_i, empty_i, empty_i), (zero, zero))
		finite = torch.isfinite(flat).all(dim=-1)
		safe = torch.where(torch.isfinite(flat), flat, torch.zeros_like(flat))
		d = safe[:, 0]
		h = safe[:, 1]
		w = safe[:, 2]
		h_max_bound = float(H) if bool(quad_scalar_bounds) else float(H - 1)
		w_max_bound = float(W) if bool(quad_scalar_bounds) else float(W - 1)
		in_bounds = finite & (d >= 0.0) & (d <= float(D - 1)) & (h >= 0.0) & (h <= h_max_bound) & (w >= 0.0) & (w <= w_max_bound)
		di = torch.round(d.clamp(0.0, float(max(0, D - 1)))).long()
		hc = h.clamp(0.0, float(max(0, H - 1)))
		wc = w.clamp(0.0, float(max(0, W - 1)))
		if H <= 1:
			h0 = h1 = torch.zeros_like(hc, dtype=torch.long)
			fh = torch.zeros_like(hc)
		else:
			h0 = torch.floor(hc).clamp(0, H - 2).long()
			h1 = h0 + 1
			fh = hc - h0.to(dtype=coords.dtype)
		if W <= 1:
			w0 = w1 = torch.zeros_like(wc, dtype=torch.long)
			fw = torch.zeros_like(wc)
		else:
			w0 = torch.floor(wc).clamp(0, W - 2).long()
			w1 = w0 + 1
			fw = wc - w0.to(dtype=coords.dtype)
		return _MapInitSurfaceSamplePlan(tuple(grid_shape), tuple(coords.shape[:-1]), in_bounds, (di, h0, h1, w0, w1), (fh, fw))
	if len(grid_shape) == 2:
		H, W = int(grid_shape[0]), int(grid_shape[1])
		flat = coords.reshape(-1, 2)
		if H <= 0 or W <= 0:
			empty_i = torch.zeros(flat.shape[0], device=coords.device, dtype=torch.long)
			zero = torch.zeros(flat.shape[0], device=coords.device, dtype=coords.dtype)
			return _MapInitSurfaceSamplePlan(tuple(grid_shape), tuple(coords.shape[:-1]), torch.zeros(flat.shape[0], device=coords.device, dtype=torch.bool), (empty_i, empty_i, empty_i, empty_i), (zero, zero))
		finite = torch.isfinite(flat).all(dim=-1)
		safe = torch.where(torch.isfinite(flat), flat, torch.zeros_like(flat))
		h = safe[:, 0]
		w = safe[:, 1]
		h_max_bound = float(H) if bool(quad_scalar_bounds) else float(H - 1)
		w_max_bound = float(W) if bool(quad_scalar_bounds) else float(W - 1)
		in_bounds = finite & (h >= 0.0) & (h <= h_max_bound) & (w >= 0.0) & (w <= w_max_bound)
		hc = h.clamp(0.0, float(max(0, H - 1)))
		wc = w.clamp(0.0, float(max(0, W - 1)))
		if H <= 1:
			h0 = h1 = torch.zeros_like(hc, dtype=torch.long)
			fh = torch.zeros_like(hc)
		else:
			h0 = torch.floor(hc).clamp(0, H - 2).long()
			h1 = h0 + 1
			fh = hc - h0.to(dtype=coords.dtype)
		if W <= 1:
			w0 = w1 = torch.zeros_like(wc, dtype=torch.long)
			fw = torch.zeros_like(wc)
		else:
			w0 = torch.floor(wc).clamp(0, W - 2).long()
			w1 = w0 + 1
			fw = wc - w0.to(dtype=coords.dtype)
		return _MapInitSurfaceSamplePlan(tuple(grid_shape), tuple(coords.shape[:-1]), in_bounds, (h0, h1, w0, w1), (fh, fw))
	raise ValueError(f"expected 2D/3D sample grid shape, got {grid_shape}")

def _map_init_sample_surface_grid_plan(grid: torch.Tensor, plan: _MapInitSurfaceSamplePlan) -> torch.Tensor:
	if grid.ndim == 3:
		H, W, C = int(grid.shape[0]), int(grid.shape[1]), int(grid.shape[2])
		if tuple(plan.grid_shape) != (H, W) or H < 1 or W < 1:
			return torch.full((*plan.coords_shape, C), float("nan"), device=grid.device, dtype=grid.dtype)
		h0, h1, w0, w1 = plan.idx
		fh, fw = (v.to(device=grid.device, dtype=grid.dtype).unsqueeze(-1) for v in plan.weights)
		out = (
			(1.0 - fh) * (1.0 - fw) * grid[h0, w0] +
			fh * (1.0 - fw) * grid[h1, w0] +
			(1.0 - fh) * fw * grid[h0, w1] +
			fh * fw * grid[h1, w1]
		)
		return out.reshape(*plan.coords_shape, C)
	if grid.ndim != 4:
		raise ValueError(f"expected 2D/3D surface grid with vector channel, got shape {tuple(grid.shape)}")
	D, H, W, C = int(grid.shape[0]), int(grid.shape[1]), int(grid.shape[2]), int(grid.shape[3])
	if tuple(plan.grid_shape) != (D, H, W) or D < 1 or H < 1 or W < 1:
		return torch.full((*plan.coords_shape, C), float("nan"), device=grid.device, dtype=grid.dtype)
	di, h0, h1, w0, w1 = plan.idx
	fh, fw = (v.to(device=grid.device, dtype=grid.dtype).unsqueeze(-1) for v in plan.weights)
	out = (
		(1.0 - fh) * (1.0 - fw) * grid[di, h0, w0] +
		fh * (1.0 - fw) * grid[di, h1, w0] +
		(1.0 - fh) * fw * grid[di, h0, w1] +
		fh * fw * grid[di, h1, w1]
	)
	return out.reshape(*plan.coords_shape, C)

def _map_init_sample_valid_plan(valid: torch.Tensor, plan: _MapInitSurfaceSamplePlan) -> torch.Tensor:
	valid_b = valid.bool()
	if valid_b.ndim == 2:
		H, W = int(valid_b.shape[0]), int(valid_b.shape[1])
		if tuple(plan.grid_shape) != (H, W) or H < 1 or W < 1:
			return torch.zeros(plan.coords_shape, device=valid.device, dtype=torch.bool)
		h0, h1, w0, w1 = plan.idx
		ok = valid_b[h0, w0] & valid_b[h1, w0] & valid_b[h0, w1] & valid_b[h1, w1] & plan.in_bounds.to(device=valid.device)
		return ok.reshape(plan.coords_shape)
	if valid_b.ndim != 3:
		raise ValueError(f"expected 2D/3D validity grid, got shape {tuple(valid.shape)}")
	D, H, W = int(valid_b.shape[0]), int(valid_b.shape[1]), int(valid_b.shape[2])
	if tuple(plan.grid_shape) != (D, H, W) or D < 1 or H < 1 or W < 1:
		return torch.zeros(plan.coords_shape, device=valid.device, dtype=torch.bool)
	di, h0, h1, w0, w1 = plan.idx
	ok = valid_b[di, h0, w0] & valid_b[di, h1, w0] & valid_b[di, h0, w1] & valid_b[di, h1, w1] & plan.in_bounds.to(device=valid.device)
	return ok.reshape(plan.coords_shape)

def _map_init_sample_scalar_plan(field: torch.Tensor, valid: torch.Tensor, plan: _MapInitSurfaceSamplePlan) -> tuple[torch.Tensor, torch.Tensor]:
	valid_field = valid.to(device=field.device).bool() & torch.isfinite(field)
	field_safe = torch.where(valid_field, field, torch.zeros_like(field))
	if field.ndim == 2:
		sampled = _map_init_sample_surface_grid_plan(field_safe.unsqueeze(-1), plan).squeeze(-1)
	elif field.ndim == 3:
		sampled = _map_init_sample_surface_grid_plan(field_safe.unsqueeze(-1), plan).squeeze(-1)
	else:
		raise ValueError(f"expected 2D/3D scalar field, got shape {tuple(field.shape)}")
	ok = _map_init_sample_valid_plan(valid_field, plan) & torch.isfinite(sampled)
	return sampled, ok

def _map_init_sample_surface_grid_with_valid(
	grid: torch.Tensor,
	valid: torch.Tensor,
	coords: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
	if grid.ndim == 3:
		H, W, C = int(grid.shape[0]), int(grid.shape[1]), int(grid.shape[2])
		if H < 2 or W < 2:
			return (
				torch.full((*coords.shape[:-1], C), float("nan"), device=grid.device, dtype=grid.dtype),
				torch.zeros(coords.shape[:-1], device=valid.device, dtype=torch.bool),
			)
		flat = coords.reshape(-1, 2)
		finite = torch.isfinite(flat).all(dim=-1)
		safe = torch.where(torch.isfinite(flat), flat, torch.zeros_like(flat))
		h = safe[:, 0]
		w = safe[:, 1]
		in_bounds = finite & (h >= 0.0) & (h <= float(H - 1)) & (w >= 0.0) & (w <= float(W - 1))
		hc = h.clamp(0.0, float(H - 1))
		wc = w.clamp(0.0, float(W - 1))
		h0 = torch.floor(hc).clamp(0, H - 2).long()
		w0 = torch.floor(wc).clamp(0, W - 2).long()
		h1 = h0 + 1
		w1 = w0 + 1
		fh = (hc - h0.to(dtype=grid.dtype)).unsqueeze(-1)
		fw = (wc - w0.to(dtype=grid.dtype)).unsqueeze(-1)
		out = (
			(1.0 - fh) * (1.0 - fw) * grid[h0, w0] +
			fh * (1.0 - fw) * grid[h1, w0] +
			(1.0 - fh) * fw * grid[h0, w1] +
			fh * fw * grid[h1, w1]
		)
		ok = (
			valid[h0, w0].bool() &
			valid[h1, w0].bool() &
			valid[h0, w1].bool() &
			valid[h1, w1].bool() &
			in_bounds
		)
		return out.reshape(*coords.shape[:-1], C), ok.reshape(coords.shape[:-1])
	if grid.ndim != 4:
		raise ValueError(f"expected 2D/3D surface grid with vector channel, got shape {tuple(grid.shape)}")
	D, H, W, C = int(grid.shape[0]), int(grid.shape[1]), int(grid.shape[2]), int(grid.shape[3])
	if H < 2 or W < 2:
		return (
			torch.full((*coords.shape[:-1], C), float("nan"), device=grid.device, dtype=grid.dtype),
			torch.zeros(coords.shape[:-1], device=valid.device, dtype=torch.bool),
		)
	flat = coords.reshape(-1, 3)
	finite = torch.isfinite(flat).all(dim=-1)
	safe = torch.where(torch.isfinite(flat), flat, torch.zeros_like(flat))
	d = safe[:, 0]
	h = safe[:, 1]
	w = safe[:, 2]
	in_bounds = (
		finite &
		(d >= 0.0) & (d <= float(D - 1)) &
		(h >= 0.0) & (h <= float(H - 1)) &
		(w >= 0.0) & (w <= float(W - 1))
	)
	di = torch.round(d.clamp(0.0, float(D - 1))).long()
	hc = h.clamp(0.0, float(H - 1))
	wc = w.clamp(0.0, float(W - 1))
	h0 = torch.floor(hc).clamp(0, H - 2).long()
	w0 = torch.floor(wc).clamp(0, W - 2).long()
	h1 = h0 + 1
	w1 = w0 + 1
	fh = (hc - h0.to(dtype=grid.dtype)).unsqueeze(-1)
	fw = (wc - w0.to(dtype=grid.dtype)).unsqueeze(-1)
	out = (
		(1.0 - fh) * (1.0 - fw) * grid[di, h0, w0] +
		fh * (1.0 - fw) * grid[di, h1, w0] +
		(1.0 - fh) * fw * grid[di, h0, w1] +
		fh * fw * grid[di, h1, w1]
	)
	ok = (
		valid[di, h0, w0].bool() &
		valid[di, h1, w0].bool() &
		valid[di, h0, w1].bool() &
		valid[di, h1, w1].bool() &
		in_bounds
	)
	return out.reshape(*coords.shape[:-1], C), ok.reshape(coords.shape[:-1])

def _map_init_sample_scalar_quad_field(
	field: torch.Tensor,
	valid: torch.Tensor,
	coords3: torch.Tensor,
	shape: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
	if coords3.numel() == 0:
		return (
			field.new_empty(coords3.shape[:-1]),
			torch.zeros(coords3.shape[:-1], device=coords3.device, dtype=torch.bool),
		)
	if len(shape) not in (2, 3):
		raise ValueError(f"expected 2D/3D scalar field shape, got {shape}")
	if any(int(v) <= 0 for v in shape):
		return (
			field.new_zeros(coords3.shape[:-1]),
			torch.zeros(coords3.shape[:-1], device=coords3.device, dtype=torch.bool),
		)
	plan = _map_init_surface_sample_plan(coords3, tuple(int(v) for v in shape), quad_scalar_bounds=True)
	return _map_init_sample_scalar_plan(field, valid.to(device=field.device).bool(), plan)

def _map_init_z_lift_turn_values(
	*,
	active_quad: torch.Tensor,
	ext_theta_lifted: torch.Tensor | None,
	ext_valid: torch.Tensor | None,
	ext_theta_samples: torch.Tensor | None = None,
	ext_sample_valid: torch.Tensor | None = None,
	model_theta_lifted: torch.Tensor | None = None,
	model_valid: torch.Tensor | None = None,
	model_theta_samples: torch.Tensor | None = None,
	model_sample_valid: torch.Tensor | None = None,
	coords3: torch.Tensor,
	cfg: SnapSurfMapInitConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
	if (
		not bool(cfg.z_lift_enabled)
		or ext_theta_lifted is None
		or ext_valid is None
		or model_theta_lifted is None
		or model_valid is None
	):
		return coords3.new_zeros(coords3.shape[:-1]), torch.zeros(coords3.shape[:-1], device=coords3.device, dtype=torch.bool)
	active_sample = active_quad.to(device=coords3.device).bool().unsqueeze(-1).expand(coords3.shape[:-1])
	if ext_theta_samples is not None and ext_sample_valid is not None:
		ext_theta = ext_theta_samples.to(device=coords3.device, dtype=coords3.dtype)
		ext_ok = active_sample & ext_sample_valid.to(device=coords3.device).bool() & torch.isfinite(ext_theta)
	elif tuple(int(v) for v in ext_theta_lifted.shape) == tuple(int(v) for v in active_quad.shape):
		# Compatibility-only path for legacy quad z-lift fixtures.
		ext_theta = ext_theta_lifted.to(device=coords3.device, dtype=coords3.dtype).unsqueeze(-1).expand(coords3.shape[:-1])
		ext_ok = (
			active_sample &
			ext_valid.to(device=coords3.device).bool().unsqueeze(-1).expand(coords3.shape[:-1]) &
			torch.isfinite(ext_theta)
		)
	else:
		ext_theta = coords3.new_zeros(coords3.shape[:-1])
		ext_ok = torch.zeros(coords3.shape[:-1], device=coords3.device, dtype=torch.bool)
	ext_theta = torch.where(ext_ok, ext_theta, torch.zeros_like(ext_theta))
	if model_theta_samples is not None and model_sample_valid is not None:
		model_theta = model_theta_samples.to(device=coords3.device, dtype=coords3.dtype)
		model_ok = model_sample_valid.to(device=coords3.device).bool()
	else:
		model_theta = coords3.new_zeros(coords3.shape[:-1])
		model_ok = torch.zeros(coords3.shape[:-1], device=coords3.device, dtype=torch.bool)
	valid = (
		ext_ok &
		model_ok &
		torch.isfinite(model_theta)
	)
	residual = torch.where(valid, ext_theta - model_theta, torch.zeros_like(model_theta))
	values = _huber(residual, delta=float(cfg.z_lift_huber_delta))
	return values, valid & torch.isfinite(values)


def _map_init_sample_external_quad_scalar_field(
	field: torch.Tensor,
	valid: torch.Tensor,
	coords2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
	if coords2.numel() == 0:
		return coords2.new_zeros(coords2.shape[:-1]), torch.zeros(coords2.shape[:-1], device=coords2.device, dtype=torch.bool)
	H, W = int(field.shape[0]), int(field.shape[1])
	if H <= 0 or W <= 0:
		return coords2.new_zeros(coords2.shape[:-1]), torch.zeros(coords2.shape[:-1], device=coords2.device, dtype=torch.bool)
	plan = _map_init_surface_sample_plan(coords2, tuple(int(v) for v in valid.shape))
	return _map_init_sample_scalar_plan(
		field.to(device=coords2.device, dtype=coords2.dtype),
		valid.to(device=coords2.device).bool(),
		plan,
	)

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

def _map_init_sample_geometry_limit_ok_precomputed(
	*,
	p_ext: torch.Tensor,
	n_ext_raw: torch.Tensor,
	n_ext: torch.Tensor,
	p_model: torch.Tensor,
	n_model_raw: torch.Tensor,
	n_model: torch.Tensor,
	d: torch.Tensor,
	c_ext: torch.Tensor,
	c_model: torch.Tensor,
	c_norm: torch.Tensor,
	cfg: SnapSurfMapInitConfig,
	ext_step: torch.Tensor | None = None,
	model_step: torch.Tensor | None = None,
) -> torch.Tensor:
	if p_ext.numel() == 0:
		return torch.zeros(p_ext.shape[:-1], device=p_ext.device, dtype=torch.bool)
	ok = (
		torch.isfinite(p_ext).all(dim=-1) &
		torch.isfinite(p_model).all(dim=-1) &
		torch.isfinite(n_ext_raw).all(dim=-1) &
		torch.isfinite(n_ext).all(dim=-1) &
		(n_ext.norm(dim=-1) > 1.0e-8) &
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
		near_zero = d <= 1.0
		cos_min_ext = _map_init_connection_cos_min(d, model_step, cfg=cfg)
		cos_min_model = _map_init_connection_cos_min(d, ext_step, cfg=cfg)
		angle_ok = (
			torch.isfinite(c_ext) &
			torch.isfinite(c_model) &
			torch.isfinite(c_norm) &
			torch.isfinite(cos_min_ext) &
			torch.isfinite(cos_min_model) &
			((near_zero | (c_ext >= cos_min_ext)) & (near_zero | (c_model >= cos_min_model)))
		)
		ok = ok & angle_ok
	return ok

def _map_init_jacobian_values(
	uv: torch.Tensor,
	active_quad: torch.Tensor,
) -> torch.Tensor:
	H, W = int(uv.shape[0]), int(uv.shape[1])
	if H < 2 or W < 2 or active_quad.numel() == 0:
		return torch.empty(0, device=uv.device, dtype=uv.dtype)
	finite_uv = torch.isfinite(uv).all(dim=-1)
	cell = active_quad.bool() & _map_init_quad_corner_all(finite_uv)
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
	return dets[finite]

def _map_init_masked_mean_values(
	pairs: list[tuple[torch.Tensor, torch.Tensor]],
	z: torch.Tensor,
) -> torch.Tensor:
	total = z
	count = torch.zeros((), device=z.device, dtype=z.dtype)
	for values, mask in pairs:
		if values.numel() == 0:
			continue
		finite = mask.bool() & torch.isfinite(values)
		total = total + torch.where(finite, values, torch.zeros_like(values)).sum()
		count = count + finite.to(dtype=z.dtype).sum()
	return torch.where(count > 0.0, total / count.clamp_min(1.0), z)

def _map_init_prior_loss(
	uv: torch.Tensor,
	uv_prior: torch.Tensor,
	reg_finite: torch.Tensor,
	z: torch.Tensor,
) -> torch.Tensor:
	return _map_init_masked_mean_values([
		((uv - uv_prior).square().sum(dim=-1), reg_finite.bool() & torch.isfinite(uv_prior).all(dim=-1))
	], z)

def _map_init_jacobian_bad_quad_mask(
	uv: torch.Tensor,
	active_quad: torch.Tensor,
	*,
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
	finite = cell.unsqueeze(-1) & torch.isfinite(dets)
	bad = cell & (~finite.all(dim=-1) | (dets < float(jac_margin)).any(dim=-1))
	return bad

def _map_init_inverse_jacobian_bad_quad_mask(
	uv: torch.Tensor,
	active_quad: torch.Tensor,
	*,
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
	det_signed = det
	eps = max(1.0e-3, 0.1 * float(jac_margin))
	inv_det = torch.where(det_signed > eps, det_signed.clamp_min(eps).reciprocal(), torch.zeros_like(det_signed))
	bad = cell & (~finite | (inv_det < float(jac_margin)))
	return bad

def _map_init_jacobian_penalty(
	uv: torch.Tensor,
	active_quad: torch.Tensor,
	*,
	jac_margin: float,
) -> torch.Tensor:
	z = uv[torch.isfinite(uv)].sum() * 0.0 if uv.numel() else torch.zeros((), device=uv.device, dtype=uv.dtype)
	H, W = int(uv.shape[0]), int(uv.shape[1])
	if H < 2 or W < 2 or active_quad.numel() == 0:
		return z
	finite_uv = torch.isfinite(uv).all(dim=-1)
	cell = active_quad.bool() & _map_init_quad_corner_all(finite_uv)
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
	values = F.relu(float(jac_margin) - dets).square()
	return _map_init_masked_mean_values([(values, finite)], z)

def _map_init_jacobian_penalty_tensor(
	uv: torch.Tensor,
	active_quad: torch.Tensor,
	jac_margin: float,
) -> torch.Tensor:
	return _map_init_jacobian_penalty(uv, active_quad, jac_margin=float(jac_margin))

def _map_init_inverse_regularization_terms(
	uv: torch.Tensor,
	active_quad: torch.Tensor,
	*,
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
	det_signed = det
	eps = max(1.0e-3, 0.1 * float(jac_margin))
	safe_det = det_signed.clamp_min(eps)
	fro2 = dh.square().sum(dim=-1) + dw.square().sum(dim=-1)
	# This is the Frobenius norm of d(source)/d(model). Identity maps to 1,
	# matching the forward smooth term's identity scale.
	smooth_rev = _map_init_masked_mean_values([(0.5 * fro2 / safe_det.square(), finite)], z)
	inv_det = torch.where(det_signed > eps, safe_det.reciprocal(), torch.zeros_like(det_signed))
	jac_rev = _map_init_masked_mean_values([(F.relu(float(jac_margin) - inv_det).square(), finite)], z)
	inf = torch.full_like(inv_det, float("inf"))
	jac_inv_min_raw = torch.where(finite, inv_det, inf).min()
	finite_count = finite.to(dtype=uv.dtype).sum()
	jac_inv_min = torch.where(finite_count > 0.0, jac_inv_min_raw, z)
	jac_inv_bad = (finite & (inv_det < float(jac_margin))).to(dtype=uv.dtype).sum()

	raw_safe_det = safe_det
	inv_j = torch.zeros((*det.shape, 2, 2), device=uv.device, dtype=uv.dtype)
	inv_j_finite = torch.stack([
		torch.stack([dw[..., 1] / raw_safe_det, -dw[..., 0] / raw_safe_det], dim=-1),
		torch.stack([-dh[..., 1] / raw_safe_det, dh[..., 0] / raw_safe_det], dim=-1),
	], dim=-2)
	inv_j = torch.where(finite.unsqueeze(-1).unsqueeze(-1), inv_j_finite, inv_j)
	bend_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
	if int(inv_j.shape[0]) > 1:
		m = finite[1:, :] & finite[:-1, :]
		dj = inv_j[1:, :] - inv_j[:-1, :]
		bend_pairs.append((dj.square().sum(dim=(-1, -2)), m))
	if int(inv_j.shape[1]) > 1:
		m = finite[:, 1:] & finite[:, :-1]
		dj = inv_j[:, 1:] - inv_j[:, :-1]
		bend_pairs.append((dj.square().sum(dim=(-1, -2)), m))
	bend_rev = _map_init_masked_mean_values(bend_pairs, z)
	return {
		"smooth": smooth_rev,
		"bend": bend_rev,
		"jac": jac_rev,
		"jac_min": jac_inv_min,
		"jac_bad": jac_inv_bad,
	}

def _map_init_inverse_regularization_terms_tensor(
	uv: torch.Tensor,
	active_quad: torch.Tensor,
	jac_margin: float,
) -> dict[str, torch.Tensor]:
	return _map_init_inverse_regularization_terms(uv, active_quad, jac_margin=float(jac_margin))

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
	model_xyz_safe: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
	finite_uv = torch.isfinite(uv).all(dim=-1)
	if model_xyz.ndim == 4:
		if model_depth is None:
			return uv, finite_uv
		coords = _map_init_coords3(uv, depth=int(model_depth))
		safe_coords = torch.where(torch.isfinite(coords), coords, torch.zeros_like(coords))
		model_source = model_xyz if model_xyz_safe is None and model_valid is None else model_xyz_safe
		if model_source is None:
			model_source = _map_init_valid_field_values(model_xyz, model_valid)
		pos = _sample_surface_grid(model_source, safe_coords)
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
		model_source = model_xyz if model_xyz_safe is None and model_valid is None else model_xyz_safe
		if model_source is None:
			model_source = _map_init_valid_field_values(model_xyz, model_valid)
		pos = _sample_surface_grid(model_source, safe_coords)
		valid = finite_uv & torch.isfinite(pos).all(dim=-1)
		if model_valid is not None:
			valid = valid & _quad_valid_at_coords(
				model_valid.bool(),
				safe_coords,
				tuple(int(v) for v in model_valid.shape),
			)
		return pos, valid
	return uv, finite_uv

def _map_init_model_metric_positions_masked(
	uv: torch.Tensor,
	active_vertex: torch.Tensor,
	*,
	model_xyz: torch.Tensor,
	model_valid: torch.Tensor | None,
	model_depth: int | None,
	model_xyz_safe: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
	mask = active_vertex.to(device=uv.device).bool() & torch.isfinite(uv).all(dim=-1)
	out_channels = 3 if model_xyz.ndim in (3, 4) else int(uv.shape[-1])
	pos = torch.zeros((*uv.shape[:2], out_channels), device=uv.device, dtype=uv.dtype)
	valid = torch.zeros(uv.shape[:2], device=uv.device, dtype=torch.bool)
	if not bool(mask.any().detach().cpu()):
		return pos, valid
	uv_sel = uv[mask]
	pos_sel, valid_sel = _map_init_model_metric_positions(
		uv_sel,
		model_xyz=model_xyz,
		model_valid=model_valid,
		model_depth=model_depth,
		model_xyz_safe=model_xyz_safe,
	)
	pos[mask] = pos_sel
	valid[mask] = valid_sel
	return pos, valid

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
	metric_pos: torch.Tensor | None = None,
	metric_valid: torch.Tensor | None = None,
) -> torch.Tensor:
	active = active_quad.bool()
	if active.numel() == 0 or float(max_ratio) <= 0.0:
		return torch.zeros_like(active, dtype=torch.bool)
	H, W = int(uv.shape[0]), int(uv.shape[1])
	if H < 2 or W < 2:
		return torch.zeros_like(active, dtype=torch.bool)
	if metric_pos is None or metric_valid is None:
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
	smooth_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
	if H > 1:
		edge = torch.zeros(H - 1, W, device=field.device, dtype=torch.bool)
		if reg_quad.numel():
			edge[:, :-1] |= reg_quad
			edge[:, 1:] |= reg_quad
		m = edge & vertex_valid[1:, :] & vertex_valid[:-1, :]
		dv = field_safe[1:, :] - field_safe[:-1, :]
		finite = m & torch.isfinite(dv).all(dim=-1)
		smooth_pairs.append((dv.square().sum(dim=-1), finite))
	if W > 1:
		edge = torch.zeros(H, W - 1, device=field.device, dtype=torch.bool)
		if reg_quad.numel():
			edge[:-1, :] |= reg_quad
			edge[1:, :] |= reg_quad
		m = edge & vertex_valid[:, 1:] & vertex_valid[:, :-1]
		dv = field_safe[:, 1:] - field_safe[:, :-1]
		finite = m & torch.isfinite(dv).all(dim=-1)
		smooth_pairs.append((dv.square().sum(dim=-1), finite))
	smooth = _map_init_masked_mean_values(smooth_pairs, z)

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
		bend = _map_init_masked_mean_values([(lap.square().sum(dim=-1), finite)], z)
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
	pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
	if H > 1:
		edge = torch.zeros(H - 1, W, device=ext_pos.device, dtype=torch.bool)
		if reg_quad.numel():
			edge[:, :-1] |= reg_quad
			edge[:, 1:] |= reg_quad
		dv = ext_pos[1:, :] - ext_pos[:-1, :]
		valid = edge & finite_ext[1:, :] & finite_ext[:-1, :] & torch.isfinite(dv).all(dim=-1)
		pairs.append((dv.square().sum(dim=-1), valid))
	if W > 1:
		edge = torch.zeros(H, W - 1, device=ext_pos.device, dtype=torch.bool)
		if reg_quad.numel():
			edge[:-1, :] |= reg_quad
			edge[1:, :] |= reg_quad
		dv = ext_pos[:, 1:] - ext_pos[:, :-1]
		valid = edge & finite_ext[:, 1:] & finite_ext[:, :-1] & torch.isfinite(dv).all(dim=-1)
		pairs.append((dv.square().sum(dim=-1), valid))
	mean = _map_init_masked_mean_values(pairs, torch.ones((), device=z.device, dtype=z.dtype))
	return mean.clamp_min(1.0e-6)

def _map_init_reg_physical_ref_cache_key(
	*,
	ext_pos: torch.Tensor,
	ext_valid: torch.Tensor,
	ext_quad_valid: torch.Tensor | None,
	ext_coords: torch.Tensor | None,
	active_quad: torch.Tensor,
	cfg: SnapSurfMapInitConfig,
	prefix: tuple[Any, ...] = (),
	external_static_cache_key: Any | None = None,
) -> tuple[Any, ...]:
	return (
		"reg_physical_ref",
		*tuple(prefix),
		external_static_cache_key,
		bool(cfg.dense_opt),
		int(cfg.dense_reg_radius),
		_map_init_tensor_cache_key(ext_pos),
		_map_init_tensor_cache_key(ext_valid),
		None if ext_quad_valid is None else _map_init_tensor_cache_key(ext_quad_valid),
		None if ext_coords is None else _map_init_tensor_cache_key(ext_coords),
		_map_init_tensor_cache_key(active_quad),
	)

def _map_init_cached_reg_physical_ref(
	*,
	ext_pos: torch.Tensor,
	finite_ext: torch.Tensor,
	reg_quad: torch.Tensor,
	z: torch.Tensor,
	cache: dict[tuple[Any, ...], Any] | None,
	key: tuple[Any, ...],
) -> torch.Tensor:
	cached = _map_init_cache_get(cache, key)
	if torch.is_tensor(cached):
		return cached
	return _map_init_cache_put(cache, key, _map_init_reference_edge_square(ext_pos, finite_ext, reg_quad, z))

def _map_init_evenness_external_context_cache_key(
	*,
	ext_pos: torch.Tensor,
	active_quad: torch.Tensor,
	prefix: tuple[Any, ...] = (),
	external_static_cache_key: Any | None = None,
) -> tuple[Any, ...]:
	return (
		"reg_evenness_external",
		*tuple(prefix),
		external_static_cache_key,
		_map_init_tensor_cache_key(ext_pos),
		_map_init_tensor_cache_key(active_quad),
	)

def _map_init_signed_reciprocal_scale(
	model_value: torch.Tensor,
	ext_value: torch.Tensor,
	eps_t: torch.Tensor,
) -> torch.Tensor:
	model_safe = model_value + eps_t
	ext_safe = ext_value + eps_t
	return 0.5 * ((model_safe / ext_safe) - (ext_safe / model_safe))

def _map_init_evenness_context_has(
	ctx: _MapInitEvennessExternalContext,
	*,
	need_metric: bool,
	need_area: bool,
) -> bool:
	if bool(need_metric) and (
		ctx.ext_len_h is None or ctx.ext_len_w is None or
		ctx.ext_len_h_valid is None or ctx.ext_len_w_valid is None or
		ctx.metric_h_pair_down is None or ctx.metric_h_pair_right is None or
		ctx.metric_w_pair_down is None or ctx.metric_w_pair_right is None
	):
		return False
	if bool(need_area) and (
		ctx.ext_area is None or ctx.ext_area_valid is None or
		ctx.area_pair_down is None or ctx.area_pair_right is None
	):
		return False
	return True

def _map_init_build_evenness_external_context(
	ext_pos: torch.Tensor,
	active_quad: torch.Tensor,
	*,
	need_metric: bool,
	need_area: bool,
	base: _MapInitEvennessExternalContext | None = None,
) -> _MapInitEvennessExternalContext:
	H, W = int(ext_pos.shape[0]), int(ext_pos.shape[1])
	finite_ext = base.finite_ext if base is not None else torch.isfinite(ext_pos).all(dim=-1)
	static_quad = base.static_quad if base is not None else (active_quad.bool() & _map_init_quad_corner_all(finite_ext))
	safe_ext = None

	ext_len_h = base.ext_len_h if base is not None else None
	ext_len_w = base.ext_len_w if base is not None else None
	ext_len_h_valid = base.ext_len_h_valid if base is not None else None
	ext_len_w_valid = base.ext_len_w_valid if base is not None else None
	metric_h_pair_down = base.metric_h_pair_down if base is not None else None
	metric_h_pair_right = base.metric_h_pair_right if base is not None else None
	metric_w_pair_down = base.metric_w_pair_down if base is not None else None
	metric_w_pair_right = base.metric_w_pair_right if base is not None else None
	if bool(need_metric) and (
		ext_len_h is None or ext_len_w is None or
		ext_len_h_valid is None or ext_len_w_valid is None or
		metric_h_pair_down is None or metric_h_pair_right is None or
		metric_w_pair_down is None or metric_w_pair_right is None
	):
		safe_ext = torch.where(finite_ext.unsqueeze(-1), ext_pos, torch.zeros_like(ext_pos))
		edge_h = torch.zeros(H - 1, W, device=ext_pos.device, dtype=torch.bool)
		edge_h[:, :-1] |= static_quad
		edge_h[:, 1:] |= static_quad
		dext_h = safe_ext[1:, :] - safe_ext[:-1, :]
		ext_len_h = dext_h.norm(dim=-1)
		ext_len_h_valid = (
			edge_h &
			finite_ext[1:, :] & finite_ext[:-1, :] &
			torch.isfinite(dext_h).all(dim=-1) &
			torch.isfinite(ext_len_h)
		)
		metric_h_pair_down = ext_len_h_valid[1:, :] & ext_len_h_valid[:-1, :] if H > 2 else torch.zeros((0, W), device=ext_pos.device, dtype=torch.bool)
		metric_h_pair_right = ext_len_h_valid[:, 1:] & ext_len_h_valid[:, :-1] if W > 1 else torch.zeros((H - 1, 0), device=ext_pos.device, dtype=torch.bool)

		edge_w = torch.zeros(H, W - 1, device=ext_pos.device, dtype=torch.bool)
		edge_w[:-1, :] |= static_quad
		edge_w[1:, :] |= static_quad
		dext_w = safe_ext[:, 1:] - safe_ext[:, :-1]
		ext_len_w = dext_w.norm(dim=-1)
		ext_len_w_valid = (
			edge_w &
			finite_ext[:, 1:] & finite_ext[:, :-1] &
			torch.isfinite(dext_w).all(dim=-1) &
			torch.isfinite(ext_len_w)
		)
		metric_w_pair_down = ext_len_w_valid[1:, :] & ext_len_w_valid[:-1, :] if H > 1 else torch.zeros((0, W - 1), device=ext_pos.device, dtype=torch.bool)
		metric_w_pair_right = ext_len_w_valid[:, 1:] & ext_len_w_valid[:, :-1] if W > 2 else torch.zeros((H, 0), device=ext_pos.device, dtype=torch.bool)

	ext_area = base.ext_area if base is not None else None
	ext_area_valid = base.ext_area_valid if base is not None else None
	area_pair_down = base.area_pair_down if base is not None else None
	area_pair_right = base.area_pair_right if base is not None else None
	if bool(need_area) and (ext_area is None or ext_area_valid is None or area_pair_down is None or area_pair_right is None):
		if safe_ext is None:
			safe_ext = torch.where(finite_ext.unsqueeze(-1), ext_pos, torch.zeros_like(ext_pos))
		e00 = safe_ext[:-1, :-1]
		e10 = safe_ext[1:, :-1]
		e01 = safe_ext[:-1, 1:]
		e11 = safe_ext[1:, 1:]
		if int(safe_ext.shape[-1]) == 2:
			def cross2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
				return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
			ext_area = 0.5 * cross2(e10 - e00, e01 - e00).abs() + 0.5 * cross2(e11 - e10, e11 - e01).abs()
		else:
			ext_area = (
				0.5 * torch.cross(e10 - e00, e01 - e00, dim=-1).norm(dim=-1) +
				0.5 * torch.cross(e11 - e10, e11 - e01, dim=-1).norm(dim=-1)
			)
		ext_area_valid = static_quad & torch.isfinite(ext_area)
		area_pair_down = ext_area_valid[1:, :] & ext_area_valid[:-1, :] if H > 2 else torch.zeros((0, W - 1), device=ext_pos.device, dtype=torch.bool)
		area_pair_right = ext_area_valid[:, 1:] & ext_area_valid[:, :-1] if W > 2 else torch.zeros((H - 1, 0), device=ext_pos.device, dtype=torch.bool)

	return _MapInitEvennessExternalContext(
		finite_ext=finite_ext,
		static_quad=static_quad,
		ext_len_h=ext_len_h,
		ext_len_w=ext_len_w,
		ext_len_h_valid=ext_len_h_valid,
		ext_len_w_valid=ext_len_w_valid,
		metric_h_pair_down=metric_h_pair_down,
		metric_h_pair_right=metric_h_pair_right,
		metric_w_pair_down=metric_w_pair_down,
		metric_w_pair_right=metric_w_pair_right,
		ext_area=ext_area,
		ext_area_valid=ext_area_valid,
		area_pair_down=area_pair_down,
		area_pair_right=area_pair_right,
	)

def _map_init_cached_evenness_external_context(
	*,
	ext_pos: torch.Tensor,
	active_quad: torch.Tensor,
	need_metric: bool,
	need_area: bool,
	cache: dict[tuple[Any, ...], Any] | None,
	key: tuple[Any, ...],
) -> _MapInitEvennessExternalContext:
	cached = _map_init_cache_get(cache, key)
	if isinstance(cached, _MapInitEvennessExternalContext):
		if _map_init_evenness_context_has(cached, need_metric=need_metric, need_area=need_area):
			return cached
		ctx = _map_init_build_evenness_external_context(
			ext_pos,
			active_quad,
			need_metric=need_metric,
			need_area=need_area,
			base=cached,
		)
	else:
		ctx = _map_init_build_evenness_external_context(
			ext_pos,
			active_quad,
			need_metric=need_metric,
			need_area=need_area,
		)
	return _map_init_cache_put(cache, key, ctx)

def _map_init_local_metric_evenness_term(
	duv_h: torch.Tensor,
	duv_w: torch.Tensor,
	finite_metric: torch.Tensor,
	static_quad: torch.Tensor,
	ext_len_h: torch.Tensor,
	ext_len_w: torch.Tensor,
	ext_len_h_valid: torch.Tensor,
	ext_len_w_valid: torch.Tensor,
	metric_h_pair_down: torch.Tensor,
	metric_h_pair_right: torch.Tensor,
	metric_w_pair_down: torch.Tensor,
	metric_w_pair_right: torch.Tensor,
	z: torch.Tensor,
	eps_t: torch.Tensor,
) -> torch.Tensor:
	H, W = int(finite_metric.shape[0]), int(finite_metric.shape[1])
	metric_quad = static_quad.bool() & _map_init_quad_corner_all(finite_metric)
	edge_h = torch.zeros(H - 1, W, device=duv_h.device, dtype=torch.bool)
	edge_h[:, :-1] |= metric_quad
	edge_h[:, 1:] |= metric_quad
	uv_len_h = duv_h.norm(dim=-1)
	valid_h = (
		edge_h &
		ext_len_h_valid.bool() &
		finite_metric[1:, :] & finite_metric[:-1, :] &
		torch.isfinite(duv_h).all(dim=-1) &
		torch.isfinite(uv_len_h)
	)
	scale_h = _map_init_signed_reciprocal_scale(uv_len_h, ext_len_h.to(device=duv_h.device, dtype=duv_h.dtype), eps_t)

	edge_w = torch.zeros(H, W - 1, device=duv_w.device, dtype=torch.bool)
	edge_w[:-1, :] |= metric_quad
	edge_w[1:, :] |= metric_quad
	uv_len_w = duv_w.norm(dim=-1)
	valid_w = (
		edge_w &
		ext_len_w_valid.bool() &
		finite_metric[:, 1:] & finite_metric[:, :-1] &
		torch.isfinite(duv_w).all(dim=-1) &
		torch.isfinite(uv_len_w)
	)
	scale_w = _map_init_signed_reciprocal_scale(uv_len_w, ext_len_w.to(device=duv_w.device, dtype=duv_w.dtype), eps_t)

	metric_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
	if int(scale_h.shape[0]) > 1:
		metric_pairs.append((scale_h[1:, :] - scale_h[:-1, :], metric_h_pair_down.bool() & valid_h[1:, :] & valid_h[:-1, :]))
	if int(scale_h.shape[1]) > 1:
		metric_pairs.append((scale_h[:, 1:] - scale_h[:, :-1], metric_h_pair_right.bool() & valid_h[:, 1:] & valid_h[:, :-1]))
	if int(scale_w.shape[0]) > 1:
		metric_pairs.append((scale_w[1:, :] - scale_w[:-1, :], metric_w_pair_down.bool() & valid_w[1:, :] & valid_w[:-1, :]))
	if int(scale_w.shape[1]) > 1:
		metric_pairs.append((scale_w[:, 1:] - scale_w[:, :-1], metric_w_pair_right.bool() & valid_w[:, 1:] & valid_w[:, :-1]))
	return _map_init_mean_square_diffs(metric_pairs, z)

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
	need_metric: bool = True,
	need_area: bool = True,
	external_context: _MapInitEvennessExternalContext | None = None,
	cache: dict[tuple[Any, ...], Any] | None = None,
	cache_key_prefix: tuple[Any, ...] = (),
	external_static_cache_key: Any | None = None,
	external_active_quad: torch.Tensor | None = None,
	compile_objective: bool = False,
	compile_mode: str | None = None,
) -> dict[str, torch.Tensor]:
	z = uv[torch.isfinite(uv)].sum() * 0.0 if uv.numel() else torch.zeros((), device=uv.device, dtype=uv.dtype)
	H, W = int(uv.shape[0]), int(uv.shape[1])
	if H < 2 or W < 2 or active_quad.numel() == 0 or (not bool(need_metric) and not bool(need_area)):
		return {"metric_smooth": z, "area_smooth": z}

	context_quad = active_quad if external_active_quad is None else external_active_quad
	if external_context is None:
		ctx_key = _map_init_evenness_external_context_cache_key(
			ext_pos=ext_pos,
			active_quad=context_quad,
			prefix=cache_key_prefix,
			external_static_cache_key=external_static_cache_key,
		)
		external_context = _map_init_cached_evenness_external_context(
			ext_pos=ext_pos,
			active_quad=context_quad,
			need_metric=bool(need_metric),
			need_area=bool(need_area),
			cache=cache,
			key=ctx_key,
		)
	elif not _map_init_evenness_context_has(external_context, need_metric=bool(need_metric), need_area=bool(need_area)):
		external_context = _map_init_build_evenness_external_context(
			ext_pos,
			context_quad,
			need_metric=bool(need_metric),
			need_area=bool(need_area),
			base=external_context,
		)

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
			metric_valid = torch.isfinite(uv).all(dim=-1)
	finite_metric = metric_valid.bool() & torch.isfinite(metric_pos).all(dim=-1)
	safe_metric = torch.where(finite_metric.unsqueeze(-1), metric_pos, torch.zeros_like(metric_pos))
	eps_t = torch.tensor(float(eps), device=uv.device, dtype=uv.dtype)
	duv_h = safe_metric[1:, :] - safe_metric[:-1, :]
	duv_w = safe_metric[:, 1:] - safe_metric[:, :-1]

	if bool(need_metric):
		assert external_context.ext_len_h is not None and external_context.ext_len_w is not None
		assert external_context.ext_len_h_valid is not None and external_context.ext_len_w_valid is not None
		assert external_context.metric_h_pair_down is not None and external_context.metric_h_pair_right is not None
		assert external_context.metric_w_pair_down is not None and external_context.metric_w_pair_right is not None
		metric_args = (
			duv_h,
			duv_w,
			finite_metric,
			external_context.static_quad,
			external_context.ext_len_h,
			external_context.ext_len_w,
			external_context.ext_len_h_valid,
			external_context.ext_len_w_valid,
			external_context.metric_h_pair_down,
			external_context.metric_h_pair_right,
			external_context.metric_w_pair_down,
			external_context.metric_w_pair_right,
			z,
			eps_t,
		)
		if bool(compile_objective):
			metric_smooth = _map_init_call_compiled(
				"metric_evenness",
				_map_init_local_metric_evenness_term,
				metric_args,
				mode=compile_mode,
			)
		else:
			metric_smooth = _map_init_local_metric_evenness_term(*metric_args)
	else:
		metric_smooth = z

	if bool(need_area):
		assert external_context.ext_area is not None and external_context.ext_area_valid is not None
		assert external_context.area_pair_down is not None and external_context.area_pair_right is not None
		if int(safe_metric.shape[-1]) == 2:
			def cross2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
				return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
			uv_area = 0.5 * cross2(duv_h[:, :-1], duv_w[:-1, :]).abs() + 0.5 * cross2(duv_w[1:, :], duv_h[:, 1:]).abs()
		else:
			uv_area = (
				0.5 * torch.cross(duv_h[:, :-1], duv_w[:-1, :], dim=-1).norm(dim=-1) +
				0.5 * torch.cross(duv_w[1:, :], duv_h[:, 1:], dim=-1).norm(dim=-1)
			)
		area_valid = (
			external_context.ext_area_valid &
			_map_init_quad_corner_all(finite_metric) &
			torch.isfinite(uv_area)
		)
		area_scale = _map_init_signed_reciprocal_scale(uv_area, external_context.ext_area.to(device=uv.device, dtype=uv.dtype), eps_t)
		area_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
		if int(area_scale.shape[0]) > 1:
			area_pairs.append((area_scale[1:, :] - area_scale[:-1, :], external_context.area_pair_down & area_valid[1:, :] & area_valid[:-1, :]))
		if int(area_scale.shape[1]) > 1:
			area_pairs.append((area_scale[:, 1:] - area_scale[:, :-1], external_context.area_pair_right & area_valid[:, 1:] & area_valid[:, :-1]))
		area_smooth = _map_init_mean_square_diffs(area_pairs, z)
	else:
		area_smooth = z

	return {"metric_smooth": metric_smooth, "area_smooth": area_smooth}

def _map_init_local_jacobian_pass(
	uv: torch.Tensor,
	active_quad: torch.Tensor,
	*,
	h: int,
	w: int,
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
			vals = _map_init_jacobian_values(uv, cell)
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

def _map_init_static_regularization_quad_mask(
	*,
	active_quad: torch.Tensor,
	ext_valid: torch.Tensor,
	ext_quad_valid: torch.Tensor | None,
	vertex_shape: tuple[int, int],
	cfg: SnapSurfMapInitConfig,
) -> torch.Tensor:
	if not bool(cfg.dense_opt):
		return active_quad.bool() & _map_init_external_quad_valid(ext_valid, ext_quad_valid)
	active_vertex = _map_init_active_vertex_mask(active_quad, tuple(int(v) for v in vertex_shape))
	band = _dilate_mask_2d(
		active_vertex.unsqueeze(0),
		radius=int(cfg.dense_reg_radius),
	)[0]
	vertex = band & ext_valid.bool()
	return _map_init_quad_corner_all(vertex) & _map_init_external_quad_valid(ext_valid, ext_quad_valid)

def _map_init_static_reg_quad_cache_key(
	*,
	active_quad: torch.Tensor,
	ext_valid: torch.Tensor,
	ext_quad_valid: torch.Tensor | None,
	vertex_shape: tuple[int, int],
	cfg: SnapSurfMapInitConfig,
	prefix: tuple[Any, ...] = (),
	external_static_cache_key: Any | None = None,
) -> tuple[Any, ...]:
	return (
		"reg_evenness_static_quad",
		*tuple(prefix),
		external_static_cache_key,
		bool(cfg.dense_opt),
		int(cfg.dense_reg_radius),
		tuple(int(v) for v in vertex_shape),
		_map_init_tensor_cache_key(active_quad),
		_map_init_tensor_cache_key(ext_valid),
		None if ext_quad_valid is None else _map_init_tensor_cache_key(ext_quad_valid),
	)

def _map_init_cached_static_regularization_quad_mask(
	*,
	active_quad: torch.Tensor,
	ext_valid: torch.Tensor,
	ext_quad_valid: torch.Tensor | None,
	vertex_shape: tuple[int, int],
	cfg: SnapSurfMapInitConfig,
	cache: dict[tuple[Any, ...], Any] | None,
	key: tuple[Any, ...],
) -> torch.Tensor:
	cached = _map_init_cache_get(cache, key)
	if torch.is_tensor(cached):
		return cached
	return _map_init_cache_put(cache, key, _map_init_static_regularization_quad_mask(
		active_quad=active_quad,
		ext_valid=ext_valid,
		ext_quad_valid=ext_quad_valid,
		vertex_shape=vertex_shape,
		cfg=cfg,
	))

def _map_init_dense_bilerp_quad(grid: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
	H, W = int(grid.shape[0]), int(grid.shape[1])
	S = int(offsets.shape[0])
	if H < 2 or W < 2:
		return grid.new_empty(max(0, H - 1), max(0, W - 1), S, *grid.shape[2:])
	fh = offsets[:, 0].view(1, 1, S, *([1] * (grid.ndim - 2)))
	fw = offsets[:, 1].view(1, 1, S, *([1] * (grid.ndim - 2)))
	v00 = grid[:-1, :-1].unsqueeze(2)
	v10 = grid[1:, :-1].unsqueeze(2)
	v01 = grid[:-1, 1:].unsqueeze(2)
	v11 = grid[1:, 1:].unsqueeze(2)
	return (
		(1.0 - fh) * (1.0 - fw) * v00 +
		fh * (1.0 - fw) * v10 +
		(1.0 - fh) * fw * v01 +
		fh * fw * v11
	)

def _map_init_dense_quad_external_sample_cache_key(
	*,
	ext_pos: torch.Tensor,
	ext_normals: torch.Tensor,
	ext_valid: torch.Tensor,
	ext_quad_valid: torch.Tensor | None,
	ext_coords: torch.Tensor | None,
	ext_theta: torch.Tensor | None,
	ext_theta_valid: torch.Tensor | None,
	subdiv: int,
	prefix: tuple[Any, ...] = (),
	ext_coords_key: Any | None = None,
) -> tuple[Any, ...]:
	return (
		"dense_quad_external_samples",
		*tuple(prefix),
		int(subdiv),
		_map_init_tensor_cache_key(ext_pos),
		_map_init_tensor_cache_key(ext_normals),
		_map_init_tensor_cache_key(ext_valid),
		None if ext_quad_valid is None else _map_init_tensor_cache_key(ext_quad_valid),
		ext_coords_key if ext_coords_key is not None else (None if ext_coords is None else _map_init_tensor_cache_key(ext_coords)),
		None if ext_theta is None else _map_init_tensor_cache_key(ext_theta),
		None if ext_theta_valid is None else _map_init_tensor_cache_key(ext_theta_valid),
	)

def _map_init_dense_quad_external_sample_tensors(
	*,
	uv_full: torch.Tensor,
	ext_pos: torch.Tensor,
	ext_normals: torch.Tensor,
	ext_valid: torch.Tensor,
	ext_quad_valid: torch.Tensor | None,
	ext_coords: torch.Tensor | None,
	ext_theta: torch.Tensor | None = None,
	ext_theta_valid: torch.Tensor | None = None,
	subdiv: int,
	profile_blocks: dict[str, list[float]] | None = None,
	cache: dict[tuple[Any, ...], Any] | None = None,
	cache_key_prefix: tuple[Any, ...] = (),
	ext_coords_cache_key: Any | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
	def _timed(name: str, fn):
		if profile_blocks is None:
			return fn()
		if uv_full.is_cuda:
			torch.cuda.synchronize(uv_full.device)
		start = time.perf_counter()
		out = fn()
		if uv_full.is_cuda:
			torch.cuda.synchronize(uv_full.device)
		profile_blocks.setdefault(name, []).append(time.perf_counter() - start)
		return out

	key = _map_init_dense_quad_external_sample_cache_key(
		ext_pos=ext_pos,
		ext_normals=ext_normals,
		ext_valid=ext_valid,
		ext_quad_valid=ext_quad_valid,
		ext_coords=ext_coords,
		ext_theta=ext_theta,
		ext_theta_valid=ext_theta_valid,
		subdiv=int(subdiv),
		prefix=cache_key_prefix,
		ext_coords_key=ext_coords_cache_key,
	)
	cached = _map_init_cache_get(cache, key)
	if cached is not None:
		if profile_blocks is not None:
			profile_blocks.setdefault("sample_ext_cached", []).append(0.0)
		return cached

	s = max(1, int(subdiv))
	offsets = _map_init_quad_offsets(subdiv=s, device=uv_full.device, dtype=uv_full.dtype)
	ext_pos_norm_safe = _timed("sample_ext_packed_source_prepare", lambda: _map_init_cached_packed_pos_norm_values(
		kind="external",
		pos=ext_pos,
		normals=ext_normals,
		valid=ext_valid,
		cache=cache,
		prefix=cache_key_prefix,
	))
	if ext_coords is None:
		if ext_theta is not None and ext_theta_valid is not None:
			theta_valid = ext_theta_valid.to(device=ext_theta.device).bool() & torch.isfinite(ext_theta)
			theta_safe = torch.where(theta_valid, ext_theta, torch.full_like(ext_theta, float("nan")))
			ext_source = torch.cat([ext_pos_norm_safe, theta_safe.unsqueeze(-1)], dim=-1)
		else:
			ext_source = ext_pos_norm_safe
		ext_samples_packed = _timed("sample_ext_packed_bilerp", lambda: _map_init_dense_bilerp_quad(
			ext_source,
			offsets.to(dtype=ext_source.dtype),
		))
		quad_ext_valid = _map_init_external_quad_valid(ext_valid, ext_quad_valid)
	else:
		sample_coords = _timed("sample_ext_coord_bilerp", lambda: _map_init_dense_bilerp_quad(ext_coords, offsets.to(dtype=ext_coords.dtype)))
		safe_coords = torch.where(torch.isfinite(sample_coords), sample_coords, torch.zeros_like(sample_coords))
		ext_samples_packed = _timed("sample_ext_packed_grid", lambda: _sample_surface_grid(ext_pos_norm_safe, safe_coords))
		sample_coord_ok = (
			torch.isfinite(sample_coords).all(dim=-1) &
			_quad_valid_at_coords(ext_valid.bool(), safe_coords, tuple(int(v) for v in ext_valid.shape)) &
			_map_init_ext_quad_valid_at_coords(ext_quad_valid, safe_coords, tuple(int(v) for v in ext_valid.shape))
		)
		quad_ext_valid = sample_coord_ok.all(dim=-1)
	if int(ext_samples_packed.shape[-1]) == 7:
		ext_samples = ext_samples_packed[..., :3]
		n_raw = ext_samples_packed[..., 3:6]
		ext_theta_samples = ext_samples_packed[..., 6]
		ext_theta_sample_valid = quad_ext_valid.unsqueeze(-1) & torch.isfinite(ext_theta_samples)
	else:
		ext_samples, n_raw = _map_init_split_packed_pos_norm(ext_samples_packed)
		ext_theta_samples = None
		ext_theta_sample_valid = None
	n_samples = F.normalize(n_raw, dim=-1, eps=1.0e-8)
	sample_ext_ok = (
		quad_ext_valid.unsqueeze(-1) &
		torch.isfinite(ext_samples).all(dim=-1) &
		torch.isfinite(n_raw).all(dim=-1) &
		torch.isfinite(n_samples).all(dim=-1) &
		(n_samples.norm(dim=-1) > 1.0e-8)
	)
	return _map_init_cache_put(cache, key, (ext_samples, n_samples, sample_ext_ok, ext_theta_samples, ext_theta_sample_valid))

def _map_init_dense_quad_sample_tensors(
	*,
	uv_full: torch.Tensor,
	ext_pos: torch.Tensor,
	ext_normals: torch.Tensor,
	ext_valid: torch.Tensor,
	ext_quad_valid: torch.Tensor | None,
	ext_coords: torch.Tensor | None = None,
	ext_theta: torch.Tensor | None = None,
	ext_theta_valid: torch.Tensor | None = None,
	subdiv: int,
	profile_blocks: dict[str, list[float]] | None = None,
	cache: dict[tuple[Any, ...], Any] | None = None,
	cache_key_prefix: tuple[Any, ...] = (),
	ext_coords_cache_key: Any | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
	def _timed(name: str, fn):
		if profile_blocks is None:
			return fn()
		if uv_full.is_cuda:
			torch.cuda.synchronize(uv_full.device)
		start = time.perf_counter()
		out = fn()
		if uv_full.is_cuda:
			torch.cuda.synchronize(uv_full.device)
		profile_blocks.setdefault(name, []).append(time.perf_counter() - start)
		return out

	s = max(1, int(subdiv))
	offsets = _map_init_quad_offsets(subdiv=s, device=uv_full.device, dtype=uv_full.dtype)
	uv_samples = _timed("sample_uv_bilerp", lambda: _map_init_dense_bilerp_quad(uv_full, offsets))
	ext_samples, n_samples, sample_ext_ok, ext_theta_samples, ext_theta_sample_valid = _map_init_dense_quad_external_sample_tensors(
		uv_full=uv_full,
		ext_pos=ext_pos,
		ext_normals=ext_normals,
		ext_valid=ext_valid,
		ext_quad_valid=ext_quad_valid,
		ext_coords=ext_coords,
		ext_theta=ext_theta,
		ext_theta_valid=ext_theta_valid,
		subdiv=int(subdiv),
		profile_blocks=profile_blocks,
		cache=cache,
		cache_key_prefix=cache_key_prefix,
		ext_coords_cache_key=ext_coords_cache_key,
	)
	quad_uv_ok = _map_init_quad_corner_all(torch.isfinite(uv_full).all(dim=-1))
	return uv_samples, ext_samples, n_samples, sample_ext_ok, quad_uv_ok, ext_theta_samples, ext_theta_sample_valid

def _map_init_dense_mean_quad_edge_length(corners: torch.Tensor, corner_valid: torch.Tensor) -> torch.Tensor:
	if corners.numel() == 0:
		return torch.zeros(corners.shape[:2], device=corners.device, dtype=corners.dtype)
	edges = torch.stack([
		corners[..., 1, :] - corners[..., 0, :],
		corners[..., 3, :] - corners[..., 2, :],
		corners[..., 2, :] - corners[..., 0, :],
		corners[..., 3, :] - corners[..., 1, :],
	], dim=-2)
	valid = torch.stack([
		corner_valid[..., 1] & corner_valid[..., 0],
		corner_valid[..., 3] & corner_valid[..., 2],
		corner_valid[..., 2] & corner_valid[..., 0],
		corner_valid[..., 3] & corner_valid[..., 1],
	], dim=-1)
	length = edges.norm(dim=-1)
	valid = valid & torch.isfinite(edges).all(dim=-1) & torch.isfinite(length) & (length > 1.0e-8)
	count = valid.to(dtype=corners.dtype).sum(dim=-1)
	total = torch.where(valid, length, torch.zeros_like(length)).sum(dim=-1)
	return torch.where(count > 0.0, total / count.clamp_min(1.0), torch.zeros_like(total))

def _map_init_dense_quad_external_physical_step_lengths(
	*,
	ext_pos: torch.Tensor,
	ext_valid: torch.Tensor,
	ext_coords: torch.Tensor | None,
) -> torch.Tensor:
	if ext_coords is None:
		ext_pos_safe = _map_init_valid_field_values(ext_pos, ext_valid)
		ext_corners = torch.stack([
			ext_pos_safe[:-1, :-1],
			ext_pos_safe[1:, :-1],
			ext_pos_safe[:-1, 1:],
			ext_pos_safe[1:, 1:],
		], dim=-2)
		ext_corner_valid = torch.stack([
			ext_valid[:-1, :-1],
			ext_valid[1:, :-1],
			ext_valid[:-1, 1:],
			ext_valid[1:, 1:],
		], dim=-1).bool()
	else:
		coords = torch.stack([
			ext_coords[:-1, :-1],
			ext_coords[1:, :-1],
			ext_coords[:-1, 1:],
			ext_coords[1:, 1:],
		], dim=-2)
		safe = torch.where(torch.isfinite(coords), coords, torch.zeros_like(coords))
		ext_corners = _sample_surface_grid(_map_init_valid_field_values(ext_pos, ext_valid), safe)
		ext_corner_valid = (
			torch.isfinite(coords).all(dim=-1) &
			_quad_valid_at_coords(ext_valid.bool(), safe, tuple(int(v) for v in ext_valid.shape))
		)
	ext_corner_valid = ext_corner_valid & torch.isfinite(ext_corners).all(dim=-1)
	return _map_init_dense_mean_quad_edge_length(ext_corners, ext_corner_valid)

def _map_init_dense_quad_external_step_cache_key(
	*,
	ext_pos: torch.Tensor,
	ext_valid: torch.Tensor,
	ext_coords: torch.Tensor | None,
	prefix: tuple[Any, ...] = (),
	ext_coords_key: Any | None = None,
) -> tuple[Any, ...]:
	return (
		"dense_quad_external_step_lengths",
		*tuple(prefix),
		_map_init_tensor_cache_key(ext_pos),
		_map_init_tensor_cache_key(ext_valid),
		ext_coords_key if ext_coords_key is not None else (None if ext_coords is None else _map_init_tensor_cache_key(ext_coords)),
	)

def _map_init_cached_dense_quad_external_physical_step_lengths(
	*,
	ext_pos: torch.Tensor,
	ext_valid: torch.Tensor,
	ext_coords: torch.Tensor | None,
	cache: dict[tuple[Any, ...], Any] | None = None,
	cache_key_prefix: tuple[Any, ...] = (),
	ext_coords_cache_key: Any | None = None,
) -> torch.Tensor:
	key = _map_init_dense_quad_external_step_cache_key(
		ext_pos=ext_pos,
		ext_valid=ext_valid,
		ext_coords=ext_coords,
		prefix=cache_key_prefix,
		ext_coords_key=ext_coords_cache_key,
	)
	cached = _map_init_cache_get(cache, key)
	if torch.is_tensor(cached):
		return cached
	return _map_init_cache_put(cache, key, _map_init_dense_quad_external_physical_step_lengths(
		ext_pos=ext_pos,
		ext_valid=ext_valid,
		ext_coords=ext_coords,
	))

def _map_init_dense_quad_model_physical_step_lengths(
	*,
	uv_full: torch.Tensor,
	model_xyz: torch.Tensor,
	model_valid: torch.Tensor,
	model_depth: int,
	model_xyz_safe: torch.Tensor | None = None,
) -> torch.Tensor:
	uv_corners = torch.stack([
		uv_full[:-1, :-1],
		uv_full[1:, :-1],
		uv_full[:-1, 1:],
		uv_full[1:, 1:],
	], dim=-2)
	coords3 = _map_init_coords3(uv_corners, depth=int(model_depth))
	safe3 = torch.where(torch.isfinite(coords3), coords3, torch.zeros_like(coords3))
	model_source = _map_init_valid_field_values(model_xyz, model_valid) if model_xyz_safe is None else model_xyz_safe
	model_corners = _sample_surface_grid(model_source, safe3)
	model_corner_valid = (
		torch.isfinite(coords3).all(dim=-1) &
		_quad_valid_at_coords(model_valid.bool(), safe3, tuple(int(v) for v in model_valid.shape)) &
		torch.isfinite(model_corners).all(dim=-1)
	)
	return _map_init_dense_mean_quad_edge_length(model_corners, model_corner_valid)

def _map_init_dense_quad_model_physical_step_lengths_from_metric(
	model_metric_pos: torch.Tensor,
	model_metric_valid: torch.Tensor,
) -> torch.Tensor:
	if model_metric_pos.numel() == 0:
		return torch.zeros(model_metric_pos.shape[:2], device=model_metric_pos.device, dtype=model_metric_pos.dtype)
	corners = torch.stack([
		model_metric_pos[:-1, :-1],
		model_metric_pos[1:, :-1],
		model_metric_pos[:-1, 1:],
		model_metric_pos[1:, 1:],
	], dim=-2)
	corner_valid = torch.stack([
		model_metric_valid[:-1, :-1],
		model_metric_valid[1:, :-1],
		model_metric_valid[:-1, 1:],
		model_metric_valid[1:, 1:],
	], dim=-1).bool()
	return _map_init_dense_mean_quad_edge_length(corners, corner_valid)

def _map_init_dense_quad_physical_step_lengths(
	*,
	uv_full: torch.Tensor,
	ext_pos: torch.Tensor,
	ext_valid: torch.Tensor,
	ext_coords: torch.Tensor | None,
	model_xyz: torch.Tensor,
	model_valid: torch.Tensor,
	model_depth: int,
) -> tuple[torch.Tensor, torch.Tensor]:
	return (
		_map_init_dense_quad_external_physical_step_lengths(
			ext_pos=ext_pos,
			ext_valid=ext_valid,
			ext_coords=ext_coords,
		),
		_map_init_dense_quad_model_physical_step_lengths(
			uv_full=uv_full,
			model_xyz=model_xyz,
			model_valid=model_valid,
			model_depth=int(model_depth),
		),
	)

def _map_init_reduce_sample_terms_tensor(
	active_quad: torch.Tensor,
	quad_uv_ok: torch.Tensor,
	uv: torch.Tensor,
	p_ext: torch.Tensor,
	n_ext_raw: torch.Tensor,
	n_ext: torch.Tensor,
	coord_ok: torch.Tensor,
	p_model: torch.Tensor,
	n_model_raw: torch.Tensor,
	n_model: torch.Tensor,
	dist_values: torch.Tensor,
	vec_values: torch.Tensor,
	norm_values: torch.Tensor,
	turn_values: torch.Tensor,
	turn_valid: torch.Tensor,
	sample_ext_ok: torch.Tensor,
	sample_limit_ok: torch.Tensor,
	allow_partial_model_samples: bool,
	need_stats: bool,
	z_lift_active: bool,
	z_lift_stats_active: bool,
	w_dist: float,
	w_vec_normal: float,
	w_surface_normal: float,
	w_z_lift: float,
	z: torch.Tensor,
	d: torch.Tensor,
) -> dict[str, torch.Tensor]:
	Hq, Wq, S = int(uv.shape[0]), int(uv.shape[1]), int(uv.shape[2])
	active_sample = active_quad.unsqueeze(-1).expand(Hq, Wq, S)
	base_finite = (
		active_sample &
		sample_ext_ok &
		quad_uv_ok.unsqueeze(-1).expand(Hq, Wq, S) &
		torch.isfinite(uv).all(dim=-1) &
		torch.isfinite(p_ext).all(dim=-1) &
		torch.isfinite(n_ext_raw).all(dim=-1) &
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
	if bool(z_lift_active):
		model_finite = model_finite & turn_valid
	finite = base_finite & model_finite
	limited_finite = finite & sample_limit_ok
	if bool(allow_partial_model_samples):
		loss_quad = active_quad & base_finite.all(dim=-1) & finite.any(dim=-1)
		valid_quad = active_quad & base_finite.all(dim=-1) & limited_finite.any(dim=-1)
	else:
		loss_quad = active_quad & finite.all(dim=-1)
		valid_quad = active_quad & limited_finite.all(dim=-1)

	sample_total_count_t = active_sample.to(dtype=z.dtype).sum()
	sample_valid_count_t = z.new_zeros(())
	sample_bad_count_t = z.new_zeros(())
	sample_base_count_t = z.new_zeros(())
	sample_model_count_t = z.new_zeros(())
	sample_limit_count_t = z.new_zeros(())
	turn_valid_count_t = z.new_zeros(())
	sample_bad_frac = z
	sample_loss = z
	if bool(need_stats):
		sample_valid_count_t = finite.to(dtype=z.dtype).sum()
		sample_bad_count_t = (active_sample & ~limited_finite).to(dtype=z.dtype).sum()
		sample_base_count_t = base_finite.to(dtype=z.dtype).sum()
		sample_model_count_t = (active_sample & model_finite).to(dtype=z.dtype).sum()
		sample_limit_count_t = limited_finite.to(dtype=z.dtype).sum()
		turn_valid_count_t = (active_sample & turn_valid).to(dtype=z.dtype).sum()
		sample_bad_frac = sample_bad_count_t / sample_total_count_t.clamp_min(1.0)
		sample_values = (
			float(w_dist) * dist_values +
			float(w_vec_normal) * vec_values +
			float(w_surface_normal) * norm_values +
			float(w_z_lift) * turn_values
		)
		sample_loss = _map_init_masked_mean_values([(sample_values, finite)], z)

	loss_sample = finite & loss_quad.unsqueeze(-1)
	loss_count = loss_sample.to(dtype=z.dtype).sum(dim=-1).clamp_min(1.0)
	finite_count_t = loss_sample.to(dtype=z.dtype).sum() if bool(need_stats) else z.new_zeros(())
	model_bad_count_t = (active_quad & ~valid_quad).to(dtype=z.dtype).sum() if bool(need_stats) else z.new_zeros(())
	loss_quad_count_t = loss_quad.to(dtype=z.dtype).sum() if bool(need_stats) else z.new_zeros(())
	valid_quad_count_t = valid_quad.to(dtype=z.dtype).sum() if bool(need_stats) else z.new_zeros(())
	turn_sample_count_t = finite_count_t if bool(need_stats) and bool(z_lift_stats_active) else z.new_zeros(())

	dist_q_all = torch.where(loss_sample, dist_values, dist_values.new_zeros(Hq, Wq, S)).sum(dim=-1) / loss_count
	vec_q_all = torch.where(loss_sample, vec_values, vec_values.new_zeros(Hq, Wq, S)).sum(dim=-1) / loss_count
	norm_q_all = torch.where(loss_sample, norm_values, norm_values.new_zeros(Hq, Wq, S)).sum(dim=-1) / loss_count
	turn_q_all = torch.where(loss_sample, turn_values, turn_values.new_zeros(Hq, Wq, S)).sum(dim=-1) / loss_count
	d_q_all = torch.where(loss_sample, d, d.new_zeros(Hq, Wq, S)).sum(dim=-1) / loss_count
	dist_loss = _map_init_masked_mean_values([(dist_q_all, loss_quad)], z)
	vec_loss = _map_init_masked_mean_values([(vec_q_all, loss_quad)], z)
	norm_loss = _map_init_masked_mean_values([(norm_q_all, loss_quad)], z)
	turn_loss = _map_init_masked_mean_values([(turn_q_all, loss_quad)], z)
	dist_avg = _map_init_masked_mean_values([(d_q_all, loss_quad)], z)
	return {
		"finite": finite,
		"loss_quad": loss_quad,
		"valid_quad": valid_quad,
		"dist_loss": dist_loss,
		"vec_loss": vec_loss,
		"norm_loss": norm_loss,
		"turn_loss": turn_loss,
		"dist_avg": dist_avg,
		"sample_total_count": sample_total_count_t,
		"sample_valid_count": sample_valid_count_t,
		"sample_bad_count": sample_bad_count_t,
		"sample_base_count": sample_base_count_t,
		"sample_model_count": sample_model_count_t,
		"sample_limit_count": sample_limit_count_t,
		"turn_valid_count": turn_valid_count_t,
		"sample_bad_frac": sample_bad_frac,
		"sample_loss": sample_loss,
		"finite_count": finite_count_t,
		"model_bad_count": model_bad_count_t,
		"loss_quad_count": loss_quad_count_t,
		"valid_quad_count": valid_quad_count_t,
		"turn_sample_count": turn_sample_count_t,
	}

def _map_init_reduce_sample_terms(
	*,
	active_quad: torch.Tensor,
	quad_uv_ok: torch.Tensor,
	uv: torch.Tensor,
	p_ext: torch.Tensor,
	n_ext_raw: torch.Tensor,
	n_ext: torch.Tensor,
	coord_ok: torch.Tensor,
	p_model: torch.Tensor,
	n_model_raw: torch.Tensor,
	n_model: torch.Tensor,
	dist_values: torch.Tensor,
	vec_values: torch.Tensor,
	norm_values: torch.Tensor,
	turn_values: torch.Tensor,
	turn_valid: torch.Tensor,
	sample_ext_ok: torch.Tensor,
	sample_limit_ok: torch.Tensor,
	allow_partial_model_samples: bool,
	need_stats: bool,
	z_lift_active: bool,
	z_lift_stats_active: bool,
	mi: SnapSurfMapInitConfig,
	z: torch.Tensor,
	d: torch.Tensor,
) -> dict[str, torch.Tensor]:
	return _map_init_reduce_sample_terms_tensor(
		active_quad,
		quad_uv_ok,
		uv,
		p_ext,
		n_ext_raw,
		n_ext,
		coord_ok,
		p_model,
		n_model_raw,
		n_model,
		dist_values,
		vec_values,
		norm_values,
		turn_values,
		turn_valid,
		sample_ext_ok,
		sample_limit_ok,
		bool(allow_partial_model_samples),
		bool(need_stats),
		bool(z_lift_active),
		bool(z_lift_stats_active),
		float(mi.w_dist),
		float(mi.w_vec_normal),
		float(mi.w_surface_normal),
		float(mi.w_z_lift),
		z,
		d,
	)

def _map_init_active_quad_crop_slices(
	active_quad: torch.Tensor,
	cfg: SnapSurfMapInitConfig,
) -> tuple[slice, slice, slice, slice] | None:
	if active_quad.numel() == 0:
		return None
	active_hw = active_quad.bool().nonzero(as_tuple=False)
	if int(active_hw.shape[0]) == 0:
		return None
	QH, QW = int(active_quad.shape[0]), int(active_quad.shape[1])
	pad = max(0, int(cfg.dense_reg_radius)) if bool(cfg.dense_opt) else 0
	h0 = max(0, int(active_hw[:, 0].min().detach().cpu()) - pad)
	h1 = min(QH - 1, int(active_hw[:, 0].max().detach().cpu()) + pad)
	w0 = max(0, int(active_hw[:, 1].min().detach().cpu()) - pad)
	w1 = min(QW - 1, int(active_hw[:, 1].max().detach().cpu()) + pad)
	return (
		slice(h0, h1 + 2),
		slice(w0, w1 + 2),
		slice(h0, h1 + 1),
		slice(w0, w1 + 1),
	)

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
	sign: int,
	cfg: SnapSurfConfig,
	w_jac_mult: float = 1.0,
	uv_prior: torch.Tensor | None = None,
	allow_partial_model_samples: bool = False,
	need_stats: bool = True,
	crop_active_quad: bool = False,
	active_quad_crop: tuple[slice, slice, slice, slice] | None = None,
	ext_z_lift_theta: torch.Tensor | None = None,
	ext_z_lift_valid: torch.Tensor | None = None,
	model_z_lift_theta: torch.Tensor | None = None,
	model_z_lift_valid: torch.Tensor | None = None,
	profile_blocks: dict[str, list[float]] | None = None,
	runtime_cache: dict[tuple[Any, ...], Any] | None = None,
	cache_key_prefix: tuple[Any, ...] = (),
	external_static_cache_key: Any | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
	mi = cfg.map_init
	z = uv_full[torch.isfinite(uv_full)].sum() * 0.0 if uv_full.numel() else model_xyz.sum() * 0.0
	def _timed(name: str, fn):
		if profile_blocks is None:
			return fn()
		if uv_full.is_cuda:
			torch.cuda.synchronize(uv_full.device)
		start = time.perf_counter()
		out = fn()
		if uv_full.is_cuda:
			torch.cuda.synchronize(uv_full.device)
		profile_blocks.setdefault(name, []).append(time.perf_counter() - start)
		return out

	train_fast = not bool(need_stats)
	sample_dist_active = (not train_fast) or float(mi.w_dist) > 0.0
	sample_vec_active = (not train_fast) or float(mi.w_vec_normal) > 0.0
	sample_norm_active = (not train_fast) or float(mi.w_surface_normal) > 0.0
	turn_term_active = (not train_fast) or (bool(mi.z_lift_enabled) and float(mi.w_z_lift) > 0.0)
	sample_terms_active = sample_dist_active or sample_vec_active or sample_norm_active or turn_term_active
	smooth_term_active = (not train_fast) or float(mi.w_smooth) > 0.0
	bend_term_active = (not train_fast) or float(mi.w_bend) > 0.0
	jac_term_active = (not train_fast) or (float(mi.w_jac) * float(w_jac_mult)) > 0.0
	metric_term_active = (not train_fast) or float(mi.w_metric_smooth) > 0.0
	area_term_active = (not train_fast) or float(mi.w_area_smooth) > 0.0
	prior_term_active = (not train_fast) or (bool(mi.dense_opt) and float(mi.w_dense_prior) > 0.0 and uv_prior is not None)
	compile_objective = _map_init_compile_enabled(mi, need_stats=bool(need_stats))
	compile_mode = mi.compile_objective_mode
	sample_step_limits_need_physical = float(mi.max_sample_angle_deg) < 180.0 and float(mi.sample_angle_step_fraction) > 0.0
	model_reg_metric_needed = smooth_term_active or bend_term_active or metric_term_active or area_term_active
	reg_terms_active = (
		smooth_term_active or bend_term_active or jac_term_active or
		metric_term_active or area_term_active or prior_term_active or bool(need_stats)
	)
	active_quad = active_quad.bool()
	original_quad_shape = tuple(int(v) for v in active_quad.shape)
	original_vertex_shape = tuple(int(v) for v in uv_full.shape[:2])
	crop = active_quad_crop
	if crop is None and bool(crop_active_quad):
		crop = _map_init_active_quad_crop_slices(active_quad, mi)
	if crop is not None:
		vh, vw, qh, qw = crop
		uv_full = uv_full[vh, vw]
		active_quad = active_quad[qh, qw]
		if uv_prior is not None:
			uv_prior = uv_prior[vh, vw]
		if ext_coords is None:
			ext_pos = ext_pos[vh, vw]
			ext_normals = ext_normals[vh, vw]
			ext_valid = ext_valid[vh, vw]
			if ext_quad_valid is not None and tuple(ext_quad_valid.shape) == original_quad_shape:
				ext_quad_valid = ext_quad_valid[qh, qw]
			if ext_z_lift_theta is not None:
				if tuple(ext_z_lift_theta.shape) == original_vertex_shape:
					ext_z_lift_theta = ext_z_lift_theta[vh, vw]
				elif tuple(ext_z_lift_theta.shape) == original_quad_shape:
					ext_z_lift_theta = ext_z_lift_theta[qh, qw]
			if ext_z_lift_valid is not None:
				if tuple(ext_z_lift_valid.shape) == original_vertex_shape:
					ext_z_lift_valid = ext_z_lift_valid[vh, vw]
				elif tuple(ext_z_lift_valid.shape) == original_quad_shape:
					ext_z_lift_valid = ext_z_lift_valid[qh, qw]
		else:
			ext_coords = ext_coords[vh, vw]
	ext_vertex_pos, _ext_vertex_normals, ext_vertex_valid, ext_level_quad_valid = _map_init_level_external_tensors(
		ext_pos=ext_pos,
		ext_normals=ext_normals,
		ext_valid=ext_valid,
		ext_quad_valid=ext_quad_valid,
		ext_coords=ext_coords,
	)
	uv_finite = torch.isfinite(uv_full).all(dim=-1)
	active_count_t = active_quad.to(dtype=uv_full.dtype).sum() if bool(need_stats) else uv_full.new_zeros(())
	quad_uv_ok_grid = active_quad & _map_init_quad_corner_all(uv_finite) if bool(need_stats) else None
	active_bad_count_t = (
		(active_quad & ~quad_uv_ok_grid).to(dtype=uv_full.dtype).sum()
		if quad_uv_ok_grid is not None else
		uv_full.new_zeros(())
	)
	finite_count_t = uv_full.new_zeros(())
	model_bad_count_t = uv_full.new_zeros(())
	sample_total_count_t = uv_full.new_zeros(())
	sample_bad_count_t = uv_full.new_zeros(())
	sample_valid_count_t = uv_full.new_zeros(())
	sample_loss = z
	sample_bad_frac = z
	sample_quad_ok_grid = torch.zeros_like(active_quad, dtype=torch.bool) if bool(need_stats) else None
	sample_base_count_t = uv_full.new_zeros(())
	sample_model_count_t = uv_full.new_zeros(())
	sample_limit_count_t = uv_full.new_zeros(())
	loss_quad_count_t = uv_full.new_zeros(())
	valid_quad_count_t = uv_full.new_zeros(())
	turn_loss = z
	turn_sample_count_t = uv_full.new_zeros(())
	turn_valid_count_t = uv_full.new_zeros(())
	model_metric_pos_cached: torch.Tensor | None = None
	model_metric_valid_cached: torch.Tensor | None = None
	if sample_terms_active and active_quad.numel() > 0 and bool(active_quad.any().detach().cpu()):
		z_lift_fields_available = (
			bool(mi.z_lift_enabled)
			and ext_z_lift_theta is not None
			and ext_z_lift_valid is not None
			and model_z_lift_theta is not None
			and model_z_lift_valid is not None
		)
		pack_ext_theta = turn_term_active and z_lift_fields_available and ext_coords is None
		uv_samples, p_ext, n_ext_raw, sample_ext_ok, quad_uv_ok, ext_theta_samples, ext_theta_sample_valid = _map_init_dense_quad_sample_tensors(
			uv_full=uv_full,
			ext_pos=ext_pos,
			ext_normals=ext_normals,
			ext_valid=ext_valid,
			ext_quad_valid=ext_quad_valid,
			ext_coords=ext_coords,
			ext_theta=ext_z_lift_theta if pack_ext_theta else None,
			ext_theta_valid=ext_z_lift_valid if pack_ext_theta else None,
			subdiv=int(mi.subdiv),
			profile_blocks=profile_blocks,
			cache=runtime_cache,
			cache_key_prefix=cache_key_prefix,
			ext_coords_cache_key=external_static_cache_key,
		)
		Hq, Wq, S = int(uv_samples.shape[0]), int(uv_samples.shape[1]), int(uv_samples.shape[2])
		uv = uv_samples
		coords3 = _map_init_coords3(uv, depth=model_depth)
		safe_coords = torch.where(torch.isfinite(coords3), coords3, torch.zeros_like(coords3))
		p_ext_f = p_ext
		n_ext_raw_f = n_ext_raw
		sign_f = 1.0 if int(sign) >= 0 else -1.0
		n_ext = F.normalize(n_ext_raw_f, dim=-1, eps=1.0e-8)

		def _sample_model_context():
			if turn_term_active and z_lift_fields_available:
				model_source_i = _map_init_packed_model_vertex_values(
					model_xyz,
					model_normals,
					model_valid,
					model_z_lift_theta.to(device=model_xyz.device, dtype=model_xyz.dtype),
					model_z_lift_valid.to(device=model_xyz.device).bool(),
				)
			else:
				model_source_i = _map_init_cached_packed_pos_norm_values(
					kind="model",
					pos=model_xyz,
					normals=model_normals,
					valid=model_valid,
					cache=runtime_cache,
					prefix=cache_key_prefix,
				)
			model_plan_i = _map_init_surface_sample_plan(safe_coords, tuple(int(v) for v in model_valid.shape))
			return (
				model_source_i,
				_map_init_sample_surface_grid_plan(model_source_i, model_plan_i),
				_map_init_sample_valid_plan(model_valid.bool(), model_plan_i),
				model_plan_i,
			)

		model_source, model_sample_packed, coord_ok, _model_sample_plan = _timed("sample_model_context", _sample_model_context)
		model_xyz_safe = model_source[..., :3]
		p_model, n_model_raw, model_theta_samples, model_theta_valid_samples = _map_init_split_packed_model_vertex_sample(model_sample_packed)
		model_theta_sample_valid = None
		if model_theta_samples is not None and model_theta_valid_samples is not None:
			model_theta_sample_valid = coord_ok & torch.isfinite(model_theta_samples) & (model_theta_valid_samples >= (1.0 - 1.0e-6))
		n_model = F.normalize(n_model_raw, dim=-1, eps=1.0e-8) * sign_f
		if sample_step_limits_need_physical:
			ext_step_q = _timed("ext_metric_steps", lambda: _map_init_cached_dense_quad_external_physical_step_lengths(
				ext_pos=ext_pos,
				ext_valid=ext_valid,
				ext_coords=ext_coords,
				cache=runtime_cache,
				cache_key_prefix=cache_key_prefix,
				ext_coords_cache_key=external_static_cache_key,
			))
			def _sample_model_metric_steps():
				model_metric_pos_i, model_metric_valid_i = _map_init_model_metric_positions(
					uv_full,
					model_xyz=model_xyz,
					model_valid=model_valid,
					model_depth=int(model_depth),
					model_xyz_safe=model_xyz_safe,
				)
				return (
					model_metric_pos_i,
					model_metric_valid_i,
					_map_init_dense_quad_model_physical_step_lengths_from_metric(
						model_metric_pos_i,
						model_metric_valid_i,
					),
				)

			model_metric_pos_cached, model_metric_valid_cached, model_step_q = _timed("sample_model_metric_steps", _sample_model_metric_steps)
			ext_step_f = ext_step_q.unsqueeze(-1).expand(Hq, Wq, S)
			model_step_f = model_step_q.unsqueeze(-1).expand(Hq, Wq, S)
		else:
			ext_step_f = None
			model_step_f = None

		def _sample_dot_product_tensors():
			v_i = p_model - p_ext_f
			d_i = v_i.norm(dim=-1)
			u_i = v_i / d_i.clamp_min(1.0e-8).unsqueeze(-1)
			return (
				v_i,
				d_i,
				u_i,
				(u_i * n_ext).sum(dim=-1).abs(),
				(u_i * n_model).sum(dim=-1).abs(),
				(n_ext * n_model).sum(dim=-1),
			)

		v, d, u, c_ext, c_model, c_norm = _timed("dot_products", _sample_dot_product_tensors)
		sample_limit_ok = _timed("geometry_limit", lambda: _map_init_sample_geometry_limit_ok_precomputed(
			p_ext=p_ext_f,
			n_ext_raw=n_ext_raw_f,
			n_ext=n_ext,
			p_model=p_model,
			n_model_raw=n_model_raw,
			n_model=n_model,
			d=d,
			c_ext=c_ext,
			c_model=c_model,
			c_norm=c_norm,
			cfg=mi,
			ext_step=ext_step_f,
			model_step=model_step_f,
		))
		dist_mult = _timed("dist_multiplier", lambda: _map_init_distance_multiplier(c_ext, c_model, mi))
		dist_values, vec_values, norm_values = _timed("loss_values", lambda: (
			_huber(d, delta=cfg.huber_delta) * dist_mult,
			(1.0 - c_ext) + (1.0 - c_model),
			1.0 - c_norm,
		))
		if turn_term_active and z_lift_fields_available and ext_coords is not None:
			offsets = _map_init_quad_offsets(subdiv=int(mi.subdiv), device=uv_full.device, dtype=uv_full.dtype)

			def _sample_ext_turn_field():
				sample_ext_coords = _map_init_dense_bilerp_quad(ext_coords, offsets.to(dtype=ext_coords.dtype))
				return _map_init_sample_external_quad_scalar_field(
					ext_z_lift_theta,
					ext_z_lift_valid,
					sample_ext_coords,
				)

			ext_theta_samples, ext_theta_sample_valid = _timed("sample_turn_ext_field", _sample_ext_turn_field)
		if turn_term_active:
			turn_values, turn_valid = _timed("turn_values", lambda: _map_init_z_lift_turn_values(
				active_quad=active_quad,
				ext_theta_lifted=ext_z_lift_theta,
				ext_valid=ext_z_lift_valid,
				ext_theta_samples=ext_theta_samples,
				ext_sample_valid=ext_theta_sample_valid,
				model_theta_lifted=model_z_lift_theta,
				model_valid=model_z_lift_valid,
				model_theta_samples=model_theta_samples,
				model_sample_valid=model_theta_sample_valid,
				coords3=safe_coords,
				cfg=mi,
			))
		else:
			turn_values = z.expand(Hq, Wq, S)
			turn_valid = torch.zeros((Hq, Wq, S), device=uv_full.device, dtype=torch.bool)
		z_lift_active = turn_term_active and z_lift_fields_available
		z_lift_stats_active = z_lift_fields_available
		sample_reduce_args = (
			active_quad,
			quad_uv_ok,
			uv,
			p_ext_f,
			n_ext_raw_f,
			n_ext,
			coord_ok,
			p_model,
			n_model_raw,
			n_model,
			dist_values,
			vec_values,
			norm_values,
			turn_values,
			turn_valid,
			sample_ext_ok,
			sample_limit_ok,
			bool(allow_partial_model_samples),
			bool(need_stats),
			bool(z_lift_active),
			bool(z_lift_stats_active),
			float(mi.w_dist),
			float(mi.w_vec_normal),
			float(mi.w_surface_normal),
			float(mi.w_z_lift),
			z,
			d,
		)
		if bool(compile_objective):
			sample_reduced = _timed("sample_reduce", lambda: _map_init_call_compiled(
				"sample_reduce",
				_map_init_reduce_sample_terms_tensor,
				sample_reduce_args,
				mode=compile_mode,
			))
		else:
			sample_reduced = _timed("sample_reduce", lambda: _map_init_reduce_sample_terms_tensor(*sample_reduce_args))
		finite = sample_reduced["finite"]
		loss_quad = sample_reduced["loss_quad"]
		valid_quad = sample_reduced["valid_quad"]
		if sample_quad_ok_grid is not None:
			sample_quad_ok_grid = valid_quad
		dist_loss = sample_reduced["dist_loss"]
		vec_loss = sample_reduced["vec_loss"]
		norm_loss = sample_reduced["norm_loss"]
		turn_loss = sample_reduced["turn_loss"]
		dist_avg = sample_reduced["dist_avg"]
		sample_total_count_t = sample_reduced["sample_total_count"]
		sample_valid_count_t = sample_reduced["sample_valid_count"]
		sample_bad_count_t = sample_reduced["sample_bad_count"]
		sample_base_count_t = sample_reduced["sample_base_count"]
		sample_model_count_t = sample_reduced["sample_model_count"]
		sample_limit_count_t = sample_reduced["sample_limit_count"]
		turn_valid_count_t = sample_reduced["turn_valid_count"]
		sample_bad_frac = sample_reduced["sample_bad_frac"]
		sample_loss = sample_reduced["sample_loss"]
		finite_count_t = sample_reduced["finite_count"]
		model_bad_count_t = sample_reduced["model_bad_count"]
		loss_quad_count_t = sample_reduced["loss_quad_count"]
		valid_quad_count_t = sample_reduced["valid_quad_count"]
		turn_sample_count_t = sample_reduced["turn_sample_count"]
		if bool(need_stats):
			model_bad_count_t = torch.where(
				loss_quad.to(dtype=uv_full.dtype).sum() > 0.0,
				model_bad_count_t,
				active_count_t,
			)
	else:
		dist_loss = z
		vec_loss = z
		norm_loss = z
		dist_avg = z

	if reg_terms_active:
		reg_finite, reg_quad = _timed("reg_masks", lambda: _map_init_regularization_masks(
			active_quad=active_quad,
			ext_valid=ext_vertex_valid,
			ext_quad_valid=ext_level_quad_valid,
			uv_finite=uv_finite,
			cfg=mi,
		))
		uv_safe = _timed("reg_uv_safe", lambda: torch.where(reg_finite.unsqueeze(-1), uv_full, torch.zeros_like(uv_full)))
	else:
		reg_finite = torch.zeros_like(uv_finite, dtype=torch.bool)
		reg_quad = torch.zeros_like(active_quad, dtype=torch.bool)
		uv_safe = torch.where(uv_finite.unsqueeze(-1), uv_full, torch.zeros_like(uv_full))
	reg_count_t = reg_finite.to(dtype=uv_full.dtype).sum() if bool(need_stats) else uv_full.new_zeros(())
	if model_reg_metric_needed:
		if model_metric_pos_cached is None or model_metric_valid_cached is None:
			model_metric_pos, model_metric_valid = _timed("reg_model_metric_sample", lambda: _map_init_model_metric_positions_masked(
				uv_safe,
				reg_finite,
				model_xyz=model_xyz,
				model_valid=model_valid,
				model_depth=model_depth,
			))
		else:
			model_metric_pos, model_metric_valid = _timed("reg_model_metric_reuse", lambda: (
				model_metric_pos_cached,
				model_metric_valid_cached,
			))
		model_metric_valid = model_metric_valid & reg_finite
	else:
		model_metric_pos = torch.zeros((*uv_full.shape[:2], 3), device=uv_full.device, dtype=uv_full.dtype)
		model_metric_valid = torch.zeros_like(uv_finite, dtype=torch.bool)
	model_metric_safe = torch.where(
		model_metric_valid.unsqueeze(-1),
		model_metric_pos,
		torch.zeros_like(model_metric_pos),
	)
	if smooth_term_active or bend_term_active:
		if bool(compile_objective):
			uv_fwd_terms = _timed("reg_uv_smooth_bend_fwd", lambda: _map_init_call_compiled(
				"forward_smooth_bend",
				_map_init_forward_smooth_bend_terms,
				(uv_safe, reg_finite, reg_quad, z),
				mode=compile_mode,
			))
			model_raw_fwd_terms = _timed("reg_model_smooth_bend_fwd", lambda: _map_init_call_compiled(
				"forward_smooth_bend",
				_map_init_forward_smooth_bend_terms,
				(model_metric_safe, model_metric_valid, reg_quad, z),
				mode=compile_mode,
			))
		else:
			uv_fwd_terms = _timed("reg_uv_smooth_bend_fwd", lambda: _map_init_forward_smooth_bend_terms(uv_safe, reg_finite, reg_quad, z))
			model_raw_fwd_terms = _timed("reg_model_smooth_bend_fwd", lambda: _map_init_forward_smooth_bend_terms(model_metric_safe, model_metric_valid, reg_quad, z))
		physical_ref_key = _map_init_reg_physical_ref_cache_key(
			ext_pos=ext_pos,
			ext_valid=ext_valid,
			ext_quad_valid=ext_quad_valid,
			ext_coords=ext_coords,
			active_quad=active_quad,
			cfg=mi,
			prefix=cache_key_prefix,
			external_static_cache_key=external_static_cache_key,
		)
		physical_ref2 = _timed("reg_physical_ref", lambda: _map_init_cached_reg_physical_ref(
			ext_pos=ext_vertex_pos,
			finite_ext=torch.isfinite(ext_vertex_pos).all(dim=-1) & ext_vertex_valid,
			reg_quad=reg_quad,
			z=z,
			cache=runtime_cache,
			key=physical_ref_key,
		))
		smooth_uv_fwd_loss = uv_fwd_terms["smooth"] if smooth_term_active else z
		bend_uv_fwd_loss = uv_fwd_terms["bend"] if bend_term_active else z
		smooth_model_fwd_loss = model_raw_fwd_terms["smooth"] / physical_ref2 if smooth_term_active else z
		bend_model_fwd_loss = model_raw_fwd_terms["bend"] / physical_ref2 if bend_term_active else z
	else:
		smooth_uv_fwd_loss = z
		bend_uv_fwd_loss = z
		smooth_model_fwd_loss = z
		bend_model_fwd_loss = z
	smooth_fwd_loss = smooth_uv_fwd_loss + smooth_model_fwd_loss
	bend_fwd_loss = bend_uv_fwd_loss + bend_model_fwd_loss

	if jac_term_active:
		if bool(compile_objective):
			jac_fwd_loss = _timed("reg_jac_fwd", lambda: _map_init_call_compiled(
				"jacobian_penalty",
				_map_init_jacobian_penalty_tensor,
				(uv_safe, reg_quad, float(mi.jac_margin)),
				mode=compile_mode,
			))
		else:
			jac_fwd_loss = _timed("reg_jac_fwd", lambda: _map_init_jacobian_penalty(
				uv_safe,
				reg_quad,
				jac_margin=mi.jac_margin,
			))
	else:
		jac_fwd_loss = z
	if smooth_term_active or bend_term_active or jac_term_active or bool(need_stats):
		if bool(compile_objective):
			inv_terms = _timed("reg_inverse_terms", lambda: _map_init_call_compiled(
				"inverse_regularization",
				_map_init_inverse_regularization_terms_tensor,
				(uv_safe, reg_quad, float(mi.jac_margin)),
				mode=compile_mode,
			))
		else:
			inv_terms = _timed("reg_inverse_terms", lambda: _map_init_inverse_regularization_terms(
				uv_safe,
				reg_quad,
				jac_margin=mi.jac_margin,
			))
	else:
		inv_terms = {"smooth": z, "bend": z, "jac": z, "jac_min": z, "jac_bad": z}
	if metric_term_active or area_term_active:
		def _evenness_terms():
			even_static_quad_key = _map_init_static_reg_quad_cache_key(
				active_quad=active_quad,
				ext_valid=ext_vertex_valid,
				ext_quad_valid=ext_level_quad_valid,
				vertex_shape=tuple(int(v) for v in uv_full.shape[:2]),
				cfg=mi,
				prefix=cache_key_prefix,
				external_static_cache_key=external_static_cache_key,
			)
			even_static_quad = _map_init_cached_static_regularization_quad_mask(
				active_quad=active_quad,
				ext_valid=ext_vertex_valid,
				ext_quad_valid=ext_level_quad_valid,
				vertex_shape=tuple(int(v) for v in uv_full.shape[:2]),
				cfg=mi,
				cache=runtime_cache,
				key=even_static_quad_key,
			)
			return _map_init_local_evenness_terms(
				uv_safe,
				ext_vertex_pos,
				reg_quad,
				metric_pos=model_metric_pos,
				metric_valid=model_metric_valid,
				need_metric=metric_term_active,
				need_area=area_term_active,
				cache=runtime_cache,
				cache_key_prefix=cache_key_prefix,
				external_static_cache_key=external_static_cache_key,
				external_active_quad=even_static_quad,
				compile_objective=bool(compile_objective),
				compile_mode=compile_mode,
			)

		even_terms = _timed("reg_evenness_terms", _evenness_terms)
	else:
		even_terms = {"metric_smooth": z, "area_smooth": z}
	metric_smooth_loss = even_terms["metric_smooth"] if metric_term_active else z
	area_smooth_loss = even_terms["area_smooth"] if area_term_active else z
	smooth_rev_loss = inv_terms["smooth"] if smooth_term_active else z
	bend_rev_loss = inv_terms["bend"] if bend_term_active else z
	jac_rev_loss = inv_terms["jac"] if jac_term_active else z
	smooth_loss = smooth_fwd_loss + smooth_rev_loss
	bend_loss = bend_fwd_loss + bend_rev_loss
	jac_loss = jac_fwd_loss + jac_rev_loss
	jac_min = z
	jac_bad_count_t = uv_full.new_zeros(())
	jac_bad_frac = z
	jac_bad_quad_count_t = uv_full.new_zeros(())
	jac_inv_bad_quad_count_t = uv_full.new_zeros(())
	step_bad_quad_count_t = uv_full.new_zeros(())
	quad_success_count_t = uv_full.new_zeros(())
	quad_success_frac = z
	if bool(need_stats):
		jac_vals = _map_init_jacobian_values(uv_safe, reg_quad)
		jac_min = jac_vals.min() if jac_vals.numel() else z
		if jac_vals.numel():
			jac_bad = jac_vals < float(mi.jac_margin)
			jac_bad_count_t = jac_bad.to(dtype=uv_full.dtype).sum()
			jac_bad_frac = jac_bad_count_t / float(max(1, int(jac_vals.numel())))
		jac_bad_quad_grid = _map_init_jacobian_bad_quad_mask(
			uv_safe,
			reg_quad,
			jac_margin=mi.jac_margin,
		)
		jac_inv_bad_quad_grid = _map_init_inverse_jacobian_bad_quad_mask(
			uv_safe,
			reg_quad,
			jac_margin=mi.jac_margin,
		)
		step_bad_quad_grid = _map_init_step_neighbor_bad_quad_mask(
			uv_safe,
			reg_quad,
			model_xyz=model_xyz,
			model_valid=model_valid,
			model_depth=int(model_depth),
			max_ratio=float(mi.max_step_neighbor_ratio),
			metric_pos=model_metric_pos if model_reg_metric_needed else None,
			metric_valid=model_metric_valid if model_reg_metric_needed else None,
		)
		assert quad_uv_ok_grid is not None
		assert sample_quad_ok_grid is not None
		quad_success_grid = (
			active_quad &
			quad_uv_ok_grid &
			sample_quad_ok_grid &
			~jac_bad_quad_grid &
			~jac_inv_bad_quad_grid &
			~step_bad_quad_grid
		)
		quad_success_count_t = quad_success_grid.to(dtype=uv_full.dtype).sum()
		quad_success_frac = quad_success_count_t / active_count_t.clamp_min(1.0)
		jac_bad_quad_count_t = jac_bad_quad_grid.to(dtype=uv_full.dtype).sum()
		jac_inv_bad_quad_count_t = jac_inv_bad_quad_grid.to(dtype=uv_full.dtype).sum()
		step_bad_quad_count_t = step_bad_quad_grid.to(dtype=uv_full.dtype).sum()
	if prior_term_active and bool(mi.dense_opt) and uv_prior is not None:
		if bool(compile_objective):
			prior_loss = _timed("reg_prior", lambda: _map_init_call_compiled(
				"prior_loss",
				_map_init_prior_loss,
				(uv_full, uv_prior, reg_finite, z),
				mode=compile_mode,
			))
		else:
			prior_loss = _timed("reg_prior", lambda: _map_init_prior_loss(uv_full, uv_prior, reg_finite, z))
	else:
		prior_loss = z
	loss = (
		float(mi.w_dist) * dist_loss +
		float(mi.w_vec_normal) * vec_loss +
		float(mi.w_surface_normal) * norm_loss +
		float(mi.w_z_lift) * turn_loss +
		float(mi.w_smooth) * smooth_loss +
		float(mi.w_bend) * bend_loss +
		float(mi.w_jac) * float(w_jac_mult) * jac_loss +
		float(mi.w_metric_smooth) * metric_smooth_loss +
		float(mi.w_area_smooth) * area_smooth_loss +
		float(mi.w_dense_prior) * prior_loss
	)
	loss_finite_t = torch.isfinite(loss.detach()).to(dtype=uv_full.dtype)
	if not bool(need_stats):
		return loss, {
			"loss": loss.detach(),
			"dist": dist_loss.detach(),
			"vec": vec_loss.detach(),
			"norm": norm_loss.detach(),
			"turn": turn_loss.detach(),
			"smooth": smooth_loss.detach(),
			"bend": bend_loss.detach(),
			"jac": jac_loss.detach(),
			"metric_smooth": metric_smooth_loss.detach(),
			"area_smooth": area_smooth_loss.detach(),
			"prior": prior_loss.detach(),
		}
	return loss, {
		"loss": loss.detach(),
		"dist": dist_loss.detach(),
		"vec": vec_loss.detach(),
		"norm": norm_loss.detach(),
		"turn": turn_loss.detach(),
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
		"active": active_count_t.detach(),
		"reg": reg_count_t.detach(),
		"samples": finite_count_t.detach(),
		"turn_smp": turn_sample_count_t.detach(),
		"sample_loss": sample_loss.detach(),
		"sample_total": sample_total_count_t.detach(),
		"sample_valid": sample_valid_count_t.detach(),
		"sample_base": sample_base_count_t.detach(),
		"sample_model": sample_model_count_t.detach(),
		"sample_limit": sample_limit_count_t.detach(),
		"sample_bad": sample_bad_count_t.detach(),
		"sample_bad_frac": sample_bad_frac.detach(),
		"turn_valid": turn_valid_count_t.detach(),
		"loss_quad": loss_quad_count_t.detach(),
		"valid_quad": valid_quad_count_t.detach(),
		"loss_finite": loss_finite_t.detach(),
		"quad_total": active_count_t.detach(),
		"quad_success": quad_success_count_t.detach(),
		"quad_success_frac": quad_success_frac.detach(),
		"uv_bad": active_bad_count_t.detach(),
		"model_bad": model_bad_count_t.detach(),
		"jac_bad": jac_bad_count_t.detach(),
		"jac_bad_frac": jac_bad_frac.detach(),
		"jac_bad_quad": jac_bad_quad_count_t.detach(),
		"jac_inv_bad": inv_terms["jac_bad"].detach(),
		"jac_inv_bad_quad": jac_inv_bad_quad_count_t.detach(),
		"step_bad_quad": step_bad_quad_count_t.detach(),
	}

def _map_init_surface_normal_loss(
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
	cfg: SnapSurfConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
	device = model_xyz.device
	dtype = model_xyz.dtype
	lm = torch.zeros(model_xyz.shape[:3], device=device, dtype=dtype)
	lm_count = torch.zeros_like(lm)
	mask = torch.zeros_like(lm)
	empty_stats = {
		"snaps_map_surf": 0.0,
		"snaps_map_surf_n": 0.0,
		"snaps_map_surf_avg": 0.0,
		"snaps_map_surf_abs": 0.0,
		"snaps_map_surf_max": 0.0,
	}
	z = model_xyz.sum() * 0.0
	if uv_full.numel() == 0 or active_quad.numel() == 0:
		return z, lm.unsqueeze(1), mask.unsqueeze(1), empty_stats
	active_quad = active_quad.bool()
	quad_hw = active_quad.nonzero(as_tuple=False)
	if int(quad_hw.shape[0]) == 0:
		return z, lm.unsqueeze(1), mask.unsqueeze(1), empty_stats
	uv_samples, p_ext, _n_ext, sample_ext_ok, quad_uv_ok = _map_init_quad_sample_tensors(
		uv_full=uv_full.detach(),
		ext_pos=ext_pos.detach(),
		ext_normals=ext_normals.detach(),
		ext_valid=ext_valid,
		ext_quad_valid=ext_quad_valid,
		ext_coords=ext_coords,
		quad_hw=quad_hw,
		subdiv=int(cfg.map_init.subdiv),
	)
	Q, S = int(uv_samples.shape[0]), int(uv_samples.shape[1])
	uv = uv_samples.reshape(Q * S, 2).detach()
	coords3 = _map_init_coords3(uv, depth=int(model_depth)).detach()
	safe_coords = torch.where(torch.isfinite(coords3), coords3, torch.zeros_like(coords3))
	p_ext_f = p_ext.reshape(Q * S, 3).detach()
	p_model = _sample_surface_grid(model_xyz, safe_coords)
	n_model_raw = _sample_surface_grid(model_normals.detach(), safe_coords)
	n_model = F.normalize(n_model_raw, dim=-1, eps=1.0e-8)
	coord_ok = _quad_valid_at_coords(
		model_valid.bool(),
		safe_coords,
		tuple(int(v) for v in model_valid.shape),
	)
	raw_residual = ((p_model - p_ext_f) * n_model).sum(dim=-1)
	scaled_residual = raw_residual / float(cfg.distance_scale)
	values = _huber(scaled_residual, delta=float(cfg.huber_delta) / float(cfg.distance_scale))
	finite = (
		sample_ext_ok.reshape(Q * S) &
		quad_uv_ok[:, None].expand(Q, S).reshape(Q * S) &
		torch.isfinite(uv).all(dim=-1) &
		coord_ok &
		torch.isfinite(p_ext_f).all(dim=-1) &
		torch.isfinite(p_model).all(dim=-1) &
		torch.isfinite(n_model_raw).all(dim=-1) &
		torch.isfinite(n_model).all(dim=-1) &
		(n_model.norm(dim=-1) > 1.0e-8) &
		torch.isfinite(raw_residual) &
		torch.isfinite(values)
	)
	if not bool(finite.any().detach().cpu()):
		return z, lm.unsqueeze(1), mask.unsqueeze(1), empty_stats
	values_f = values[finite]
	raw_f = raw_residual[finite]
	loss = values_f.mean()
	coords_f = safe_coords[finite]
	D, H, W = (int(model_xyz.shape[0]), int(model_xyz.shape[1]), int(model_xyz.shape[2]))
	idx_d = torch.round(coords_f[:, 0]).clamp(0, max(0, D - 1)).long()
	idx_h = torch.round(coords_f[:, 1]).clamp(0, max(0, H - 1)).long()
	idx_w = torch.round(coords_f[:, 2]).clamp(0, max(0, W - 1)).long()
	lm.index_put_((idx_d, idx_h, idx_w), values_f.detach(), accumulate=True)
	lm_count.index_put_((idx_d, idx_h, idx_w), torch.ones_like(values_f.detach()), accumulate=True)
	mask = lm_count > 0.0
	lm = torch.where(mask, lm / lm_count.clamp_min(1.0), lm)
	stats = {
		"snaps_map_surf": float(loss.detach().cpu()),
		"snaps_map_surf_n": float(values_f.numel()),
		"snaps_map_surf_avg": float(raw_f.mean().detach().cpu()),
		"snaps_map_surf_abs": float(raw_f.abs().mean().detach().cpu()),
		"snaps_map_surf_max": float(raw_f.abs().max().detach().cpu()),
	}
	return loss, lm.unsqueeze(1), mask.to(dtype=dtype).unsqueeze(1), stats

__all__ = [name for name in globals() if not name.startswith('__')]
