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

from .config import SnapSurfConfig, SnapSurfMapInitConfig, _map_init_log
from .state import _MapInitState, _SurfaceState, _DirectionState
from .tensor import *

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
		"snaps_map_turn": 0.0,
		"snaps_map_turn_smp": 0.0,
		"snaps_map_zext_bad": 0.0,
		"snaps_map_zext_unr": 0.0,
		"snaps_map_zmdl_bad": 0.0,
		"snaps_map_zmdl_unr": 0.0,
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
		"snaps_map_surf": 0.0,
		"snaps_map_surf_n": 0.0,
		"snaps_map_surf_avg": 0.0,
		"snaps_map_surf_abs": 0.0,
		"snaps_map_surf_max": 0.0,
		"snaps_map_nsign": 1.0,
		"snaps_map_scales": 1.0,
		"snaps_map_repair": 0.0,
	}

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
	sign: int,
	cfg: SnapSurfMapInitConfig,
	ext_step: torch.Tensor | None = None,
	model_step: torch.Tensor | None = None,
) -> torch.Tensor:
	if p_ext.numel() == 0:
		return torch.zeros(p_ext.shape[:-1], device=p_ext.device, dtype=torch.bool)
	sign_f = 1.0 if int(sign) >= 0 else -1.0
	n_ext_base = F.normalize(n_ext, dim=-1, eps=1.0e-8)
	n_model = F.normalize(n_model_raw, dim=-1, eps=1.0e-8) * sign_f
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
		near_zero = d <= 1.0
		u = v / d.clamp_min(1.0e-8).unsqueeze(-1)
		cos_min_ext = _map_init_connection_cos_min(d, model_step, cfg=cfg)
		cos_min_model = _map_init_connection_cos_min(d, ext_step, cfg=cfg)
		angle_ok = torch.zeros_like(ok)
		c_ext = (u * n_ext_base).sum(dim=-1).abs()
		c_model = (u * n_model).sum(dim=-1).abs()
		c_norm = (n_ext_base * n_model).sum(dim=-1)
		angle_ok = angle_ok | (
			torch.isfinite(c_ext) &
			torch.isfinite(c_model) &
			torch.isfinite(c_norm) &
			torch.isfinite(cos_min_ext) &
			torch.isfinite(cos_min_model) &
			((near_zero | (c_ext >= cos_min_ext)) & (near_zero | (c_model >= cos_min_model)))
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
	sign: int = 1,
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
			sign=sign,
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
	preserve_batch: bool = False,
) -> torch.Tensor:
	if not pyr:
		raise ValueError("empty map-init UV pyramid")
	level = int(active_level)
	if level < 0 or level >= len(pyr):
		raise ValueError(f"active_level {level} outside pyramid with {len(pyr)} levels")
	v = pyr[-1]
	for i in range(len(pyr) - 2, level - 1, -1):
		v = F.interpolate(v, size=(int(pyr[i].shape[2]), int(pyr[i].shape[3])), mode="bilinear", align_corners=True) + pyr[i]
	out = v.permute(0, 2, 3, 1).contiguous()
	return out if bool(preserve_batch) else out.squeeze(0).contiguous()

def _map_init_integrate_dyadic_uv_to_nchw(
	pyr: list[torch.Tensor],
	*,
	active_level: int,
) -> torch.Tensor:
	out = _map_init_integrate_dyadic_uv_pyramid(pyr, active_level=active_level, preserve_batch=True)
	return out.permute(0, 3, 1, 2).contiguous()

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
	if uv.ndim == 3 and int(uv.shape[-1]) == 2:
		uv_n = uv.unsqueeze(0)
	elif uv.ndim == 4 and int(uv.shape[-1]) == 2:
		uv_n = uv
	else:
		raise ValueError("map-init dense uv must be (H,W,2) or (N,H,W,2)")
	H, W = int(uv_n.shape[1]), int(uv_n.shape[2])
	shapes = _map_init_scale_shapes(H, W, levels=levels, factor=factor)
	targets: list[torch.Tensor] = [uv_n.permute(0, 3, 1, 2).contiguous()]
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

__all__ = [name for name in globals() if not name.startswith('__')]
