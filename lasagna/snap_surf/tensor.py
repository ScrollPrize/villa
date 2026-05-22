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

from .config import SnapSurfConfig
from .state import _DirectionState

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

__all__ = [name for name in globals() if not name.startswith('__')]
