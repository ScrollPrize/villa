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

from .config import SnapSurfConfig, SnapSurfMapInitConfig
from .tensor import *
from .legacy import _huber
from .map_pyramid import *

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
