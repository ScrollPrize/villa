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
from .state import _MapInitState, _SurfaceState, _DirectionState
from .tensor import *
from .legacy import *
from .map_pyramid import *
from .map_objective import *
from .debug_obj import _debug_obj_iter_dir, _debug_write_map_init_objs

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

__all__ = [name for name in globals() if not name.startswith('__')]
