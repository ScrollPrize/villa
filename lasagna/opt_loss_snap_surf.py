from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import torch
import torch.nn.functional as F

import model as fit_model
import opt_loss_winding_density
import opt_loss_station


@dataclass(frozen=True)
class SnapSurfConfig:
	init_distance: float = 50.0
	point_distance: float = 25.0
	grid_error: float = 0.75
	affine_radius: int = 2
	search_ring: int = 1
	seed_radius: int = 4
	map_inlier_distance: float = 8.0
	inlier_normal_distance_ratio: float = 1.5
	inlier_normal_distance_floor: float = 10.0
	ray_residual: float = 1.0e-2
	brute_interval: int = 10
	brute_boundary_radius: int = 10
	brute_pair_chunk_limit: int = 8_000_000
	huber_delta: float = 5.0
	distance_scale: float = 1.0
	w_to_ext: float = 1.0
	w_to_model: float = 1.0
	orientation: str = "auto"
	debug_obj_dir: str | None = None
	debug_obj_interval: int = 1


_cfg = SnapSurfConfig()
_active = False
_seed_xyz: tuple[float, float, float] | None = None
_states: list["_SurfaceState"] = []
_last_stats: dict[str, float] = {}
_debug_step: int | None = None
_debug_label: str | None = None


def reset_state() -> None:
	global _states, _last_stats, _debug_step, _debug_label
	_states = []
	_last_stats = {}
	_debug_step = None
	_debug_label = None


def configure_snap_surf(
	*,
	cfg: dict | None = None,
	seed_xyz: tuple[float, float, float] | None = None,
	active: bool = False,
) -> None:
	"""Configure the runtime snap-surface loss state for the current stage."""
	global _cfg, _active, _seed_xyz
	raw = dict(cfg or {})
	bad = sorted(set(raw.keys()) - set(SnapSurfConfig.__dataclass_fields__.keys()))
	if bad:
		raise ValueError(f"snap_surf args: unknown key(s): {bad}")
	_cfg = SnapSurfConfig(
		init_distance=float(raw.get("init_distance", SnapSurfConfig.init_distance)),
		point_distance=float(raw.get("point_distance", SnapSurfConfig.point_distance)),
		grid_error=float(raw.get("grid_error", SnapSurfConfig.grid_error)),
		affine_radius=max(1, int(raw.get("affine_radius", SnapSurfConfig.affine_radius))),
		search_ring=max(0, int(raw.get("search_ring", SnapSurfConfig.search_ring))),
		seed_radius=int(raw.get("seed_radius", SnapSurfConfig.seed_radius)),
		map_inlier_distance=float(raw.get("map_inlier_distance", SnapSurfConfig.map_inlier_distance)),
		inlier_normal_distance_ratio=float(raw.get("inlier_normal_distance_ratio", SnapSurfConfig.inlier_normal_distance_ratio)),
		inlier_normal_distance_floor=float(raw.get("inlier_normal_distance_floor", SnapSurfConfig.inlier_normal_distance_floor)),
		ray_residual=float(raw.get("ray_residual", SnapSurfConfig.ray_residual)),
		brute_interval=max(1, int(raw.get("brute_interval", SnapSurfConfig.brute_interval))),
		brute_boundary_radius=max(0, int(raw.get("brute_boundary_radius", SnapSurfConfig.brute_boundary_radius))),
		brute_pair_chunk_limit=max(1, int(raw.get("brute_pair_chunk_limit", SnapSurfConfig.brute_pair_chunk_limit))),
		huber_delta=float(raw.get("huber_delta", SnapSurfConfig.huber_delta)),
		distance_scale=max(1.0e-8, float(raw.get("distance_scale", SnapSurfConfig.distance_scale))),
		w_to_ext=float(raw.get("w_to_ext", SnapSurfConfig.w_to_ext)),
		w_to_model=float(raw.get("w_to_model", SnapSurfConfig.w_to_model)),
		orientation=str(raw.get("orientation", SnapSurfConfig.orientation)).strip().lower(),
		debug_obj_dir=None if raw.get("debug_obj_dir", SnapSurfConfig.debug_obj_dir) in {None, ""} else str(raw.get("debug_obj_dir")),
		debug_obj_interval=max(1, int(raw.get("debug_obj_interval", SnapSurfConfig.debug_obj_interval))),
	)
	if _cfg.init_distance < 0.0:
		raise ValueError("snap_surf args.init_distance must be >= 0")
	if _cfg.point_distance < 0.0:
		raise ValueError("snap_surf args.point_distance must be >= 0")
	if _cfg.grid_error < 0.0:
		raise ValueError("snap_surf args.grid_error must be >= 0")
	if _cfg.seed_radius < 0:
		raise ValueError("snap_surf args.seed_radius must be >= 0")
	if _cfg.map_inlier_distance <= 0.0:
		raise ValueError("snap_surf args.map_inlier_distance must be > 0")
	if _cfg.inlier_normal_distance_ratio < 1.0:
		raise ValueError("snap_surf args.inlier_normal_distance_ratio must be >= 1")
	if _cfg.inlier_normal_distance_floor < 0.0:
		raise ValueError("snap_surf args.inlier_normal_distance_floor must be >= 0")
	if _cfg.ray_residual < 0.0:
		raise ValueError("snap_surf args.ray_residual must be >= 0")
	if _cfg.huber_delta <= 0.0:
		raise ValueError("snap_surf args.huber_delta must be > 0")
	if _cfg.w_to_ext < 0.0 or _cfg.w_to_model < 0.0:
		raise ValueError("snap_surf direction weights must be >= 0")
	if _cfg.orientation not in {"auto", "identity", "none"}:
		raise ValueError("snap_surf args.orientation must be 'auto', 'identity', or 'none'")
	if active and seed_xyz is None:
		raise ValueError("snap_surf requires args.seed")
	_active = bool(active)
	_seed_xyz = None if seed_xyz is None else tuple(float(v) for v in seed_xyz)


def last_stats() -> dict[str, float]:
	return dict(_last_stats)


def set_debug_step(step: int | None, *, label: str | None = None) -> None:
	global _debug_step, _debug_label
	_debug_step = None if step is None else int(step)
	_debug_label = None if label is None else str(label)


def _safe_frac(n: int | float, d: int | float) -> float:
	den = float(d)
	if den <= 0.0:
		return 0.0
	return float(n) / den


class _DirectionState:
	def __init__(self, *, source_rank: int, target_rank: int) -> None:
		self.source_rank = int(source_rank)
		self.target_rank = int(target_rank)
		self.source_shape: tuple[int, ...] | None = None
		self.target_shape: tuple[int, ...] | None = None
		self.map: torch.Tensor | None = None
		self.valid: torch.Tensor | None = None
		self.orientation_sign: int = 1
		self.seed_base_idx: tuple[int, ...] | None = None

	def ensure(
		self,
		*,
		source_shape: tuple[int, ...],
		target_shape: tuple[int, ...],
		device: torch.device,
		dtype: torch.dtype,
	) -> None:
		if (
			self.map is not None
			and self.valid is not None
			and self.source_shape == source_shape
			and self.target_shape == target_shape
			and self.map.device == device
			and self.map.dtype == dtype
		):
			return
		self.source_shape = tuple(int(v) for v in source_shape)
		self.target_shape = tuple(int(v) for v in target_shape)
		self.map = torch.full((*self.source_shape, self.target_rank), float("nan"), device=device, dtype=dtype)
		self.valid = torch.zeros(self.source_shape, device=device, dtype=torch.bool)
		self.orientation_sign = 1
		self.seed_base_idx = None

	def count(self) -> int:
		if self.valid is None:
			return 0
		return int(self.valid.sum().detach().cpu())


class _SurfaceState:
	def __init__(self) -> None:
		self.model_to_ext = _DirectionState(source_rank=3, target_rank=2)
		self.ext_to_model = _DirectionState(source_rank=2, target_rank=3)
		self.ext_seed_hw: tuple[int, int] | None = None
		self.seed_ext_distance: float | None = None
		self.seed_ext_key: tuple[float, float, float] | None = None
		self.seed_ext_point_xyz: tuple[float, float, float] | None = None
		self.snap_eval_count: int = 0

	def ensure(
		self,
		*,
		model_shape: tuple[int, int, int],
		ext_shape: tuple[int, int],
		device: torch.device,
		dtype: torch.dtype,
	) -> None:
		old_ext_shape = self.model_to_ext.target_shape
		self.model_to_ext.ensure(
			source_shape=model_shape,
			target_shape=ext_shape,
			device=device,
			dtype=dtype,
		)
		self.ext_to_model.ensure(
			source_shape=ext_shape,
			target_shape=model_shape,
			device=device,
			dtype=dtype,
		)
		if old_ext_shape != ext_shape:
			self.ext_seed_hw = None
			self.seed_ext_distance = None
			self.seed_ext_key = None
			self.seed_ext_point_xyz = None
			self.snap_eval_count = 0


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


def _sample_grid3d(grid: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
	D, H, W = int(grid.shape[0]), int(grid.shape[1]), int(grid.shape[2])
	if H < 2 or W < 2:
		return torch.full((*coords.shape[:-1], int(grid.shape[-1])), float("nan"), device=grid.device, dtype=grid.dtype)
	d = coords[..., 0].clamp(0.0, float(D - 1))
	if D < 2:
		return _sample_grid2d(grid[0], coords[..., 1:])
	h = coords[..., 1].clamp(0.0, float(H - 1))
	w = coords[..., 2].clamp(0.0, float(W - 1))
	d0 = torch.floor(d).clamp(0, D - 2).long()
	d1 = d0 + 1
	fd = (d - d0.to(dtype=grid.dtype)).unsqueeze(-1)
	h0 = torch.floor(h).clamp(0, H - 2).long()
	w0 = torch.floor(w).clamp(0, W - 2).long()
	h1 = h0 + 1
	w1 = w0 + 1
	fh = (h - h0.to(dtype=grid.dtype)).unsqueeze(-1)
	fw = (w - w0.to(dtype=grid.dtype)).unsqueeze(-1)
	v000 = grid[d0, h0, w0]
	v010 = grid[d0, h1, w0]
	v001 = grid[d0, h0, w1]
	v011 = grid[d0, h1, w1]
	v100 = grid[d1, h0, w0]
	v110 = grid[d1, h1, w0]
	v101 = grid[d1, h0, w1]
	v111 = grid[d1, h1, w1]
	p0 = (1 - fh) * (1 - fw) * v000 + fh * (1 - fw) * v010 + (1 - fh) * fw * v001 + fh * fw * v011
	p1 = (1 - fh) * (1 - fw) * v100 + fh * (1 - fw) * v110 + (1 - fh) * fw * v101 + fh * fw * v111
	return (1 - fd) * p0 + fd * p1


def _sample_grid(grid: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
	if grid.ndim == 3:
		return _sample_grid2d(grid, coords)
	if grid.ndim == 4:
		return _sample_grid3d(grid, coords)
	raise ValueError(f"expected 2D/3D grid with vector channel, got shape {tuple(grid.shape)}")


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
	nearest_step_pred = nearest_tgt.clone()
	nearest_step_pred[..., -2:] = nearest_step_pred[..., -2:] + (query_hw - nearest_src)
	pred = torch.where((count == 1).unsqueeze(-1), nearest_step_pred, affine_pred)
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


def _empty_grow_stats() -> dict[str, int | float]:
	return {
		"drop": 0,
		"ring": 0,
		"sup": 0,
		"tgt": 0,
		"dist": 0,
		"grid": 0,
		"ori": 0,
		"new": 0,
		"local": 0,
		"brute": 0,
		"front": 0,
		"brute_on": 0,
		"tested": 0,
		"gerr_n": 0,
		"gerr_sum": 0.0,
		"gerr_max": 0.0,
		"raw_map_n": 0,
		"raw_map_min": float("inf"),
		"raw_map_sum": 0.0,
		"raw_map_max": 0.0,
		"in_map_n": 0,
		"in_map_min": float("inf"),
		"in_map_sum": 0.0,
		"in_map_max": 0.0,
	}


def _add_grow_stats(dst: dict[str, int | float], src: dict[str, int | float]) -> None:
	for k in ("drop", "ring", "sup", "tgt", "dist", "grid", "ori", "new", "local", "brute", "front", "brute_on", "tested", "gerr_n", "gerr_sum", "raw_map_n", "raw_map_sum", "in_map_n", "in_map_sum"):
		dst[k] = dst.get(k, 0) + src.get(k, 0)
	dst["gerr_max"] = max(float(dst.get("gerr_max", 0.0)), float(src.get("gerr_max", 0.0)))
	dst["raw_map_min"] = min(float(dst.get("raw_map_min", float("inf"))), float(src.get("raw_map_min", float("inf"))))
	dst["raw_map_max"] = max(float(dst.get("raw_map_max", 0.0)), float(src.get("raw_map_max", 0.0)))
	dst["in_map_min"] = min(float(dst.get("in_map_min", float("inf"))), float(src.get("in_map_min", float("inf"))))
	dst["in_map_max"] = max(float(dst.get("in_map_max", 0.0)), float(src.get("in_map_max", 0.0)))


def _grow_direction(
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
	stats = _empty_grow_stats()
	if state.map is None or state.valid is None or state.source_shape is None or state.target_shape is None:
		return stats
	valid_b, map_b, source_valid_b = _batched_source_views(state, source_valid)
	base_valid = valid_b.clone()
	if int(base_valid.sum().detach().cpu()) == 0:
		return stats
	candidate_mask = _neighbor4_mask(base_valid) & source_valid_b & ~base_valid
	candidate_mask &= _seed_source_limit_mask(state, base_valid, radius=cfg.seed_radius)
	stats["ring"] = int(candidate_mask.sum().detach().cpu())
	if not bool(candidate_mask.any().detach().cpu()):
		return stats

	candidate_mask &= _neighbor4_mask(base_valid)
	stats["sup"] = int(candidate_mask.sum().detach().cpu())
	if not bool(candidate_mask.any().detach().cpu()):
		return stats

	cand_bidx = candidate_mask.nonzero(as_tuple=False)
	_, support_count, _ = _direct_predict_candidates_batched(
		state,
		valid_b=base_valid,
		map_b=map_b,
		candidate_bidx=cand_bidx,
		radius=cfg.affine_radius,
	)
	support_ok = support_count >= 1
	if not bool(support_ok.any().detach().cpu()):
		return stats
	cand_bidx = cand_bidx[support_ok]
	C = int(cand_bidx.shape[0])
	if state.source_rank == 3:
		source_idx = cand_bidx
	else:
		source_idx = cand_bidx[:, 1:]
	source_pos = _points_at_indices(source_xyz, source_idx)

	bases = _all_valid_target_quad_bases(target_valid.bool())
	if int(bases.shape[0]) == 0:
		return stats
	p00, p10, p01, p11 = _quad_corners_batched(target_xyz, bases)
	K = int(bases.shape[0])
	if normal_from_source:
		ref_normal = _points_at_indices(normal_xyz, source_idx)
		ref_normal = F.normalize(ref_normal, dim=-1, eps=1.0e-8)[:, None, :].expand(C, K, -1)
	else:
		ref_normal = _quad_average_normal_batched(normal_xyz, bases)
		ref_normal = ref_normal.unsqueeze(0).expand(C, K, -1)
	base_coord = bases.to(dtype=map_b.dtype).unsqueeze(0).expand(C, K, -1)
	coord, line_score, normal_abs = _closest_point_on_quad_along_normal_batched(
		source_pos[:, None, :],
		ref_normal,
		p00.unsqueeze(0),
		p10.unsqueeze(0),
		p01.unsqueeze(0),
		p11.unsqueeze(0),
		base_coord,
	)
	valid_choice = (
		torch.isfinite(source_pos).all(dim=-1)[:, None] &
		torch.isfinite(line_score) &
		torch.isfinite(normal_abs) &
		torch.isfinite(coord).all(dim=-1)
	)
	line_valid = torch.where(valid_choice, line_score, torch.full_like(line_score, float("inf")))
	best_line = line_valid.min(dim=1).values
	has_choice = torch.isfinite(best_line)
	if not bool(has_choice.any().detach().cpu()):
		return stats
	close_line = valid_choice & (line_score <= (best_line[:, None] + 1.0e-9))
	normal_tiebreak = torch.where(close_line, normal_abs, torch.full_like(normal_abs, float("inf")))
	best_k = torch.argmin(normal_tiebreak, dim=1)
	best_coord = coord[torch.arange(C, device=coord.device), best_k]
	accepted_mask = has_choice & torch.isfinite(best_coord).all(dim=-1)
	accepted_n = int(accepted_mask.sum().detach().cpu())
	stats["tgt"] = accepted_n
	stats["dist"] = accepted_n
	stats["grid"] = accepted_n
	stats["ori"] = accepted_n
	stats["new"] = accepted_n
	if accepted_n <= 0:
		return stats

	map_out = map_b.clone()
	valid_out = base_valid.clone()
	acc_bidx = cand_bidx[accepted_mask]
	map_out[acc_bidx[:, 0], acc_bidx[:, 1], acc_bidx[:, 2]] = best_coord[accepted_mask].to(dtype=map_b.dtype)
	valid_out[acc_bidx[:, 0], acc_bidx[:, 1], acc_bidx[:, 2]] = True
	_write_batched_state(state, valid_out, map_out)
	return stats


def _grow_until_stalled_direction(
	state: _DirectionState,
	*,
	source_xyz: torch.Tensor,
	source_valid: torch.Tensor,
	target_xyz: torch.Tensor,
	target_valid: torch.Tensor,
	normal_xyz: torch.Tensor,
	normal_from_source: bool,
	cfg: SnapSurfConfig,
	max_iters: int,
) -> tuple[int, dict[str, int | float]]:
	total = _empty_grow_stats()
	attempts = 0
	for _ in range(max(0, int(max_iters))):
		grow = _grow_direction(
			state,
			source_xyz=source_xyz,
			source_valid=source_valid,
			target_xyz=target_xyz,
			target_valid=target_valid,
			normal_xyz=normal_xyz,
			normal_from_source=normal_from_source,
			cfg=cfg,
		)
		attempts += 1
		_add_grow_stats(total, grow)
		if int(grow.get("new", 0)) <= 0:
			break
	return attempts, total


def _closest_external_seed_surface(
	*,
	seed: torch.Tensor,
	ext_xyz: torch.Tensor,
	ext_valid: torch.Tensor,
	ext_quad_valid: torch.Tensor,
	chunk_quads: int = 262144,
) -> tuple[tuple[int, int] | None, torch.Tensor | None, float]:
	"""Closest point on any valid external tifxyz quad to the seed."""
	if ext_valid.numel() == 0 or not bool(ext_valid.any().detach().cpu()):
		return None, None, float("inf")
	if ext_xyz.ndim != 3 or int(ext_xyz.shape[-1]) != 3:
		return None, None, float("inf")
	H, W, _ = ext_xyz.shape
	if H < 2 or W < 2 or ext_quad_valid.numel() == 0 or not bool(ext_quad_valid.any().detach().cpu()):
		pts = ext_xyz[ext_valid & torch.isfinite(ext_xyz).all(dim=-1)]
		if pts.numel() == 0:
			return None, None, float("inf")
		dist2 = (pts - seed.view(1, 3)).square().sum(dim=-1)
		best = int(torch.argmin(dist2).detach().cpu())
		return None, pts[best].detach(), math.sqrt(float(dist2[best].detach().cpu()))

	valid_ids = ext_quad_valid.reshape(-1).nonzero(as_tuple=False).flatten()
	if valid_ids.numel() == 0:
		return None, None, float("inf")
	Wq = W - 1
	rows_all = torch.div(valid_ids, Wq, rounding_mode="floor")
	cols_all = valid_ids - rows_all * Wq
	best = torch.full((), float("inf"), device=ext_xyz.device, dtype=ext_xyz.dtype)
	best_hw: tuple[int, int] | None = None
	best_point: torch.Tensor | None = None
	chunk = max(1, int(chunk_quads))
	for start in range(0, int(valid_ids.numel()), chunk):
		end = min(start + chunk, int(valid_ids.numel()))
		rows = rows_all[start:end]
		cols = cols_all[start:end]
		p00 = ext_xyz[rows, cols]
		p10 = ext_xyz[rows + 1, cols]
		p01 = ext_xyz[rows, cols + 1]
		p11 = ext_xyz[rows + 1, cols + 1]
		finite = (
			torch.isfinite(p00).all(dim=-1) &
			torch.isfinite(p10).all(dim=-1) &
			torch.isfinite(p01).all(dim=-1) &
			torch.isfinite(p11).all(dim=-1)
		)
		if not bool(finite.any().detach().cpu()):
			continue
		rows_f = rows[finite]
		cols_f = cols[finite]
		p00 = p00[finite]
		p10 = p10[finite]
		p01 = p01[finite]
		p11 = p11[finite]
		cp0, _ = opt_loss_station._closest_points_on_triangles(seed, p00, p10, p11)
		cp1, _ = opt_loss_station._closest_points_on_triangles(seed, p00, p11, p01)
		d20 = (cp0 - seed.view(1, 3)).square().sum(dim=-1)
		d21 = (cp1 - seed.view(1, 3)).square().sum(dim=-1)
		use_first = d20 <= d21
		d2 = torch.where(use_first, d20, d21)
		local = int(torch.argmin(d2).detach().cpu())
		local_best = d2[local]
		if float(local_best.detach().cpu()) < float(best.detach().cpu()):
			best = local_best
			best_hw = (int(rows_f[local].detach().cpu()), int(cols_f[local].detach().cpu()))
			best_point = (cp0 if bool(use_first[local].detach().cpu()) else cp1)[local].detach()
	if not bool(torch.isfinite(best).detach().cpu()):
		return None, None, float("inf")
	return best_hw, best_point, math.sqrt(float(best.detach().cpu()))


def _closest_model_surface_quad(
	*,
	point: torch.Tensor,
	model_xyz: torch.Tensor,
	model_valid: torch.Tensor,
) -> tuple[tuple[int, int, int] | None, float]:
	D, H, W, _ = model_xyz.shape
	if H < 2 or W < 2:
		return None, float("inf")
	quad_valid = (
		model_valid[:, :-1, :-1] &
		model_valid[:, 1:, :-1] &
		model_valid[:, :-1, 1:] &
		model_valid[:, 1:, 1:]
	)
	valid_ids = quad_valid.reshape(-1).nonzero(as_tuple=False).flatten()
	if valid_ids.numel() == 0:
		return None, float("inf")
	Hq, Wq = H - 1, W - 1
	d = torch.div(valid_ids, Hq * Wq, rounding_mode="floor")
	rem = valid_ids - d * Hq * Wq
	h = torch.div(rem, Wq, rounding_mode="floor")
	w = rem - h * Wq
	p00 = model_xyz[d, h, w]
	p10 = model_xyz[d, h + 1, w]
	p01 = model_xyz[d, h, w + 1]
	p11 = model_xyz[d, h + 1, w + 1]
	cp0, _ = opt_loss_station._closest_points_on_triangles(point, p00, p10, p11)
	cp1, _ = opt_loss_station._closest_points_on_triangles(point, p00, p11, p01)
	d2 = torch.minimum(
		(cp0 - point.view(1, 3)).square().sum(dim=-1),
		(cp1 - point.view(1, 3)).square().sum(dim=-1),
	)
	best = int(torch.argmin(d2).detach().cpu())
	return (
		int(d[best].detach().cpu()),
		int(h[best].detach().cpu()),
		int(w[best].detach().cpu()),
	), math.sqrt(float(d2[best].detach().cpu()))


def _choose_seed_transform(
	*,
	model_xyz: torch.Tensor,
	ext_xyz: torch.Tensor,
	model_quad: tuple[int, int, int],
	ext_quad: tuple[int, int],
	cfg: SnapSurfConfig,
) -> tuple[tuple[tuple[int, int], ...], int]:
	transforms = [_dihedral_transforms()[0]] if cfg.orientation in {"identity", "none"} else _dihedral_transforms()
	d, mh, mw = model_quad
	eh, ew = ext_quad
	model_pts = torch.stack([model_xyz[d, mh + sh, mw + sw] for sh, sw in _CORNERS_2D], dim=0)
	model_norm = _normalized_seed_quad(model_pts)
	best = transforms[0]
	best_score = float("inf")
	for transform in transforms:
		ext_pts = torch.stack([ext_xyz[eh + th, ew + tw] for th, tw in transform], dim=0)
		ext_norm = _normalized_seed_quad(ext_pts)
		score = float((model_norm - ext_norm).norm(dim=-1).sum().detach().cpu())
		if score < best_score:
			best_score = score
			best = transform
	return best, _transform_det_sign(best)


def _try_seed_reinsert(
	state: _SurfaceState,
	*,
	model_xyz: torch.Tensor,
	model_valid: torch.Tensor,
	ext_xyz: torch.Tensor,
	ext_valid: torch.Tensor,
	ext_quad_valid: torch.Tensor,
	cfg: SnapSurfConfig,
	seed_xyz: tuple[float, float, float],
) -> tuple[bool, float, float]:
	seed = torch.tensor(seed_xyz, device=ext_xyz.device, dtype=ext_xyz.dtype)
	seed_key = tuple(float(v) for v in seed_xyz)
	if state.seed_ext_distance is None or state.seed_ext_key != seed_key:
		ext_seed_hw, ext_seed_point, ext_seed_dist = _closest_external_seed_surface(
			seed=seed,
			ext_xyz=ext_xyz,
			ext_valid=ext_valid,
			ext_quad_valid=ext_quad_valid,
		)
		state.ext_seed_hw = ext_seed_hw
		state.seed_ext_distance = ext_seed_dist
		state.seed_ext_key = seed_key
		state.seed_ext_point_xyz = (
			None if ext_seed_point is None
			else tuple(float(v) for v in ext_seed_point.detach().cpu().tolist())
		)
	if state.ext_seed_hw is None or state.seed_ext_point_xyz is None:
		return False, float("inf"), state.seed_ext_distance
	eh, ew = state.ext_seed_hw
	if eh < 0 or ew < 0 or eh >= int(ext_quad_valid.shape[0]) or ew >= int(ext_quad_valid.shape[1]):
		state.ext_seed_hw = None
		return False, float("inf"), state.seed_ext_distance
	if not _bool_at_index(ext_quad_valid, (eh, ew)):
		state.ext_seed_hw = None
		return False, float("inf"), state.seed_ext_distance
	ext_seed_point_t = torch.tensor(state.seed_ext_point_xyz, device=model_xyz.device, dtype=model_xyz.dtype)
	model_quad, surface_dist = _closest_model_surface_quad(
		point=ext_seed_point_t,
		model_xyz=model_xyz,
		model_valid=model_valid,
	)
	if model_quad is None or surface_dist > cfg.init_distance:
		return False, surface_dist, state.seed_ext_distance
	d, mh, mw = model_quad
	model_quad_valid = (
		_bool_at_index(model_valid, (d, mh, mw)) and
		_bool_at_index(model_valid, (d, mh + 1, mw)) and
		_bool_at_index(model_valid, (d, mh, mw + 1)) and
		_bool_at_index(model_valid, (d, mh + 1, mw + 1))
	)
	if not model_quad_valid:
		return False, surface_dist, state.seed_ext_distance
	transform, fallback_sign = _choose_seed_transform(
		model_xyz=model_xyz,
		ext_xyz=ext_xyz,
		model_quad=model_quad,
		ext_quad=(eh, ew),
		cfg=cfg,
	)
	ext_base = torch.tensor([float(eh), float(ew)], device=model_xyz.device, dtype=model_xyz.dtype)
	model_base = torch.tensor([float(d), float(mh), float(mw)], device=model_xyz.device, dtype=model_xyz.dtype)
	model_to_ext_seed: dict[tuple[int, int], torch.Tensor] = {}
	ext_to_model_seed: dict[tuple[int, int], torch.Tensor] = {}
	for i, (sh, sw) in enumerate(_CORNERS_2D):
		th, tw = transform[i]
		model_to_ext_seed[(sh, sw)] = ext_base + torch.tensor([float(th), float(tw)], device=model_xyz.device, dtype=model_xyz.dtype)
		ext_to_model_seed[(th, tw)] = model_base + torch.tensor([0.0, float(sh), float(sw)], device=model_xyz.device, dtype=model_xyz.dtype)
	seed_source_hw = torch.tensor(_CORNERS_2D, device=model_xyz.device, dtype=model_xyz.dtype)
	m2e_targets = torch.stack([model_to_ext_seed[(sh, sw)] for sh, sw in _CORNERS_2D], dim=0)
	e2m_targets = torch.stack([ext_to_model_seed[(sh, sw)] for sh, sw in _CORNERS_2D], dim=0)
	m2e_sign = 1 if cfg.orientation in {"identity", "none"} else _affine_det_sign_from_points(
		seed_source_hw,
		m2e_targets,
		fallback=fallback_sign,
	)
	e2m_sign = 1 if cfg.orientation in {"identity", "none"} else _affine_det_sign_from_points(
		seed_source_hw,
		e2m_targets,
		fallback=fallback_sign,
	)
	for i, (sh, sw) in enumerate(_CORNERS_2D):
		model_idx = (d, mh + sh, mw + sw)
		if not _bool_at_index(model_valid, model_idx):
			continue
		_set_correspondence(state.model_to_ext, model_idx, model_to_ext_seed[(sh, sw)])
	for th, tw in _CORNERS_2D:
		ext_idx = (eh + th, ew + tw)
		if not _bool_at_index(ext_valid, ext_idx):
			continue
		_set_correspondence(state.ext_to_model, ext_idx, ext_to_model_seed[(th, tw)])
	state.model_to_ext.seed_base_idx = model_quad
	state.ext_to_model.seed_base_idx = (eh, ew)
	state.model_to_ext.orientation_sign = m2e_sign
	state.ext_to_model.orientation_sign = e2m_sign
	return True, surface_dist, state.seed_ext_distance


def _huber(residual: torch.Tensor, *, delta: float) -> torch.Tensor:
	abs_r = residual.abs()
	d = float(delta)
	return torch.where(abs_r <= d, 0.5 * residual.square(), d * (abs_r - 0.5 * d))


def _huber_grad(residual: torch.Tensor, *, delta: float) -> torch.Tensor:
	d = float(delta)
	return residual.clamp(min=-d, max=d)


def _empty_residual_stats() -> dict[str, float]:
	return {"n": 0.0, "sum": 0.0, "abs_sum": 0.0, "abs_max": 0.0, "toward_sum": 0.0}


def _residual_stats(raw_residual: torch.Tensor, scaled_residual: torch.Tensor, *, delta: float) -> dict[str, float]:
	if raw_residual.numel() == 0:
		return _empty_residual_stats()
	finite = torch.isfinite(raw_residual) & torch.isfinite(scaled_residual)
	if not bool(finite.any().detach().cpu()):
		return _empty_residual_stats()
	raw = raw_residual.detach()[finite]
	scaled = scaled_residual.detach()[finite]
	abs_r = raw.abs()
	grad_scaled = _huber_grad(scaled, delta=delta)
	# Positive means the gradient-descent update points toward the matched proxy plane.
	toward = grad_scaled * scaled
	return {
		"n": float(raw.numel()),
		"sum": float(raw.sum().cpu()),
		"abs_sum": float(abs_r.sum().cpu()),
		"abs_max": float(abs_r.max().cpu()),
		"toward_sum": float(toward.sum().cpu()),
	}


def _direction_loss_model_to_ext(
	state: _DirectionState,
	*,
	model_xyz: torch.Tensor,
	ext_xyz: torch.Tensor,
	ext_valid: torch.Tensor,
	ext_normals: torch.Tensor,
	cfg: SnapSurfConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, dict[str, float]]:
	device = model_xyz.device
	dtype = model_xyz.dtype
	lm = torch.zeros(model_xyz.shape[:3], device=device, dtype=dtype)
	mask = torch.zeros(model_xyz.shape[:3], device=device, dtype=dtype)
	if state.map is None or state.valid is None or state.count() == 0:
		z = model_xyz.new_zeros(())
		return z, lm.unsqueeze(1), mask.unsqueeze(1), 0, _empty_residual_stats()
	idx = state.valid.nonzero(as_tuple=False)
	all_coords = state.map[state.valid]
	coord_ok = torch.isfinite(all_coords).all(dim=-1) & _quad_valid_at_coords(ext_valid.bool(), all_coords, tuple(int(v) for v in ext_xyz.shape[:2]))
	if not bool(coord_ok.any().detach().cpu()):
		return model_xyz.new_zeros(()), lm.unsqueeze(1), mask.unsqueeze(1), 0, _empty_residual_stats()
	idx = idx[coord_ok]
	coords = all_coords[coord_ok]
	src = _points_at_indices(model_xyz, idx)
	tgt = _sample_surface_grid(ext_xyz, coords).detach()
	n_raw = _sample_surface_grid(ext_normals.detach(), coords)
	n = F.normalize(n_raw, dim=-1, eps=1.0e-8)
	raw_residual = ((src - tgt) * n).sum(dim=-1)
	residual = raw_residual / cfg.distance_scale
	values = _huber(residual, delta=cfg.huber_delta / cfg.distance_scale)
	finite = (
		torch.isfinite(src).all(dim=-1) &
		torch.isfinite(tgt).all(dim=-1) &
		torch.isfinite(n_raw).all(dim=-1) &
		torch.isfinite(n).all(dim=-1) &
		(n.norm(dim=-1) > 1.0e-8) &
		torch.isfinite(raw_residual) &
		torch.isfinite(values)
	)
	values_f = values[finite]
	loss = values_f.mean() if values_f.numel() else model_xyz.new_zeros(())
	if bool(finite.any().detach().cpu()):
		lm_idx = idx[finite]
		lm[lm_idx[:, 0], lm_idx[:, 1], lm_idx[:, 2]] = values_f.detach()
		mask[lm_idx[:, 0], lm_idx[:, 1], lm_idx[:, 2]] = 1.0
	stats = _residual_stats(raw_residual, residual, delta=cfg.huber_delta / cfg.distance_scale)
	return loss, lm.unsqueeze(1), mask.unsqueeze(1), int(values_f.numel()), stats


def _direction_loss_ext_to_model(
	state: _DirectionState,
	*,
	model_xyz: torch.Tensor,
	model_valid: torch.Tensor,
	ext_normals: torch.Tensor,
	ext_xyz: torch.Tensor,
	cfg: SnapSurfConfig,
) -> tuple[torch.Tensor, int, dict[str, float]]:
	if state.map is None or state.valid is None or state.count() == 0:
		return model_xyz.new_zeros(()), 0, _empty_residual_stats()
	idx = state.valid.nonzero(as_tuple=False)
	all_coords = state.map[state.valid]
	coord_ok = torch.isfinite(all_coords).all(dim=-1) & _quad_valid_at_coords(model_valid.bool(), all_coords, tuple(int(v) for v in model_xyz.shape[:3]))
	if not bool(coord_ok.any().detach().cpu()):
		return model_xyz.new_zeros(()), 0, _empty_residual_stats()
	idx = idx[coord_ok]
	coords = all_coords[coord_ok]
	src = _points_at_indices(ext_xyz, idx).detach()
	tgt = _sample_surface_grid(model_xyz, coords)
	n_raw = _points_at_indices(ext_normals.detach(), idx)
	n = F.normalize(n_raw, dim=-1, eps=1.0e-8)
	raw_residual = ((tgt - src) * n).sum(dim=-1)
	residual = raw_residual / cfg.distance_scale
	values = _huber(residual, delta=cfg.huber_delta / cfg.distance_scale)
	finite = (
		torch.isfinite(src).all(dim=-1) &
		torch.isfinite(tgt).all(dim=-1) &
		torch.isfinite(n_raw).all(dim=-1) &
		torch.isfinite(n).all(dim=-1) &
		(n.norm(dim=-1) > 1.0e-8) &
		torch.isfinite(raw_residual) &
		torch.isfinite(values)
	)
	values_f = values[finite]
	loss = values_f.mean() if values_f.numel() else model_xyz.new_zeros(())
	stats = _residual_stats(raw_residual, residual, delta=cfg.huber_delta / cfg.distance_scale)
	return loss, int(values_f.numel()), stats


def _seed_model_sheet_mask(
	*,
	model_valid: torch.Tensor,
	model_normals: torch.Tensor,
	seed_quad: tuple[int, int, int] | None,
) -> torch.Tensor:
	if seed_quad is None:
		return torch.zeros_like(model_valid, dtype=torch.bool)
	D = int(model_valid.shape[0])
	d0 = int(seed_quad[0])
	dd = torch.arange(D, device=model_valid.device).view(D, 1, 1)
	normal_ok = (
		torch.isfinite(model_normals).all(dim=-1) &
		(model_normals.norm(dim=-1) > 1.0e-8)
	)
	return model_valid.bool() & normal_ok & (dd == d0)


def _normalized_model_to_ext_map(map_b: torch.Tensor) -> torch.Tensor:
	D, H, W = (int(v) for v in map_b.shape[:3])
	source_hw = _source_hw_grid(n=D, h=H, w=W, device=map_b.device, dtype=map_b.dtype)
	return map_b - source_hw


def _neighbor_min_mapping_distance(
	*,
	center_valid: torch.Tensor,
	neighbor_valid: torch.Tensor,
	norm_map: torch.Tensor,
	normal_dist: torch.Tensor | None = None,
	max_normal_ratio: float | None = None,
	normal_distance_floor: float = 10.0,
) -> torch.Tensor:
	D, H, W = (int(v) for v in center_valid.shape)
	if H == 0 or W == 0:
		return torch.empty(D, H, W, device=norm_map.device, dtype=norm_map.dtype)
	finite_map = torch.isfinite(norm_map).all(dim=-1)
	center_ok = center_valid.bool() & finite_map
	neighbor_ok = neighbor_valid.bool() & finite_map
	ratio_patch_ok = None
	if normal_dist is not None and max_normal_ratio is not None:
		normal_dist = normal_dist.to(device=norm_map.device, dtype=norm_map.dtype).abs()
		finite_dist = torch.isfinite(normal_dist)
		center_ok = center_ok & finite_dist
		neighbor_ok = neighbor_ok & finite_dist
		dist_safe = torch.where(neighbor_ok, normal_dist, torch.zeros_like(normal_dist))
		dist_patch = F.unfold(
			dist_safe.unsqueeze(1),
			kernel_size=3,
			padding=1,
		).transpose(1, 2).reshape(D, H, W, 9)
		floor = max(1.0e-6, float(normal_distance_floor))
		center_dist = normal_dist.clamp_min(floor).unsqueeze(-1)
		neighbor_dist = dist_patch.clamp_min(floor)
		ratio = torch.maximum(center_dist / neighbor_dist, neighbor_dist / center_dist)
		ratio_patch_ok = torch.isfinite(ratio) & (ratio <= float(max_normal_ratio))
	map_safe = torch.where(neighbor_ok.unsqueeze(-1), norm_map, torch.zeros_like(norm_map))
	map_patch = F.unfold(
		map_safe.permute(0, 3, 1, 2),
		kernel_size=3,
		padding=1,
	).transpose(1, 2).reshape(D, H, W, 2, 9)
	valid_patch = F.unfold(
		neighbor_ok.to(dtype=norm_map.dtype).unsqueeze(1),
		kernel_size=3,
		padding=1,
	).transpose(1, 2).reshape(D, H, W, 9) > 0.0
	valid_patch[..., 4] = False
	if ratio_patch_ok is not None:
		ratio_patch_ok[..., 4] = False
		valid_patch = valid_patch & ratio_patch_ok
	dist = (map_patch - norm_map.unsqueeze(-1)).square().sum(dim=3).sqrt()
	dist = torch.where(
		center_ok.unsqueeze(-1) & valid_patch,
		dist,
		torch.full_like(dist, float("inf")),
	)
	return dist.min(dim=-1).values


def _mapping_distance_stats(
	*,
	center_valid: torch.Tensor,
	neighbor_valid: torch.Tensor,
	norm_map: torch.Tensor,
) -> dict[str, int | float]:
	dist = _neighbor_min_mapping_distance(
		center_valid=center_valid,
		neighbor_valid=neighbor_valid,
		norm_map=norm_map,
	)
	finite = center_valid.bool() & torch.isfinite(dist)
	if not bool(finite.any().detach().cpu()):
		return {"n": 0, "min": float("inf"), "sum": 0.0, "max": 0.0}
	vals = dist[finite].detach()
	return {
		"n": int(vals.numel()),
		"min": float(vals.min().cpu()),
		"sum": float(vals.sum().cpu()),
		"max": float(vals.max().cpu()),
	}


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


def _closest_seed_source_mask(
	*,
	raw_valid: torch.Tensor,
	model_xyz: torch.Tensor,
	seed: torch.Tensor,
) -> torch.Tensor:
	mask = torch.zeros_like(raw_valid, dtype=torch.bool)
	if not bool(raw_valid.any().detach().cpu()):
		return mask
	finite = raw_valid.bool() & torch.isfinite(model_xyz).all(dim=-1)
	if not bool(finite.any().detach().cpu()):
		return mask
	dist2 = (model_xyz - seed.view(1, 1, 1, 3)).square().sum(dim=-1)
	dist2 = torch.where(finite, dist2, torch.full_like(dist2, float("inf")))
	best_flat = torch.argmin(dist2.reshape(-1))
	best_dist = dist2.reshape(-1)[best_flat]
	if not bool(torch.isfinite(best_dist).detach().cpu()):
		return mask
	mask.reshape(-1)[best_flat] = True
	return mask


def _seeded_mapping_inlier_filter(
	*,
	raw_valid: torch.Tensor,
	raw_map: torch.Tensor,
	seed_quad: tuple[int, int, int] | None = None,
	initial_inlier: torch.Tensor | None = None,
	max_distance: float,
	normal_dist: torch.Tensor | None = None,
	max_normal_ratio: float | None = None,
	normal_distance_floor: float = 10.0,
) -> tuple[torch.Tensor, dict[str, int | float]]:
	norm_map = _normalized_model_to_ext_map(raw_map)
	raw_valid = raw_valid.bool() & torch.isfinite(norm_map).all(dim=-1)
	if normal_dist is not None:
		normal_dist = normal_dist.to(device=raw_valid.device, dtype=raw_map.dtype).abs()
		raw_valid = raw_valid & torch.isfinite(normal_dist)
	if initial_inlier is not None:
		inlier = raw_valid & initial_inlier.to(device=raw_valid.device).bool()
	elif seed_quad is not None:
		seed_mask = _seed_quad_corner_mask(
			tuple(int(v) for v in raw_valid.shape),
			seed_quad,
			device=raw_valid.device,
		)
		inlier = raw_valid & seed_mask
	else:
		inlier = torch.zeros_like(raw_valid)
	D, H, W = (int(v) for v in raw_valid.shape)
	if bool(inlier.any().detach().cpu()):
		for _ in range(max(1, D + H + W)):
			candidate = raw_valid & ~inlier
			if not bool(candidate.any().detach().cpu()):
				break
			min_dist = _neighbor_min_mapping_distance(
				center_valid=candidate,
				neighbor_valid=inlier,
				norm_map=norm_map,
				normal_dist=normal_dist,
				max_normal_ratio=max_normal_ratio,
				normal_distance_floor=normal_distance_floor,
			)
			new_inlier = candidate & (min_dist < float(max_distance))
			if not bool(new_inlier.any().detach().cpu()):
				break
			inlier = inlier | new_inlier
	return inlier, {}


def _valid_ext_quad_bases(ext_valid: torch.Tensor, ext_quad_valid: torch.Tensor) -> torch.Tensor:
	H, W = int(ext_valid.shape[0]), int(ext_valid.shape[1])
	if H < 2 or W < 2:
		return torch.empty(0, 2, device=ext_valid.device, dtype=torch.long)
	corner_quad_valid = (
		ext_valid[:-1, :-1].bool() &
		ext_valid[1:, :-1].bool() &
		ext_valid[:-1, 1:].bool() &
		ext_valid[1:, 1:].bool()
	)
	if tuple(ext_quad_valid.shape) == tuple(corner_quad_valid.shape):
		corner_quad_valid = corner_quad_valid & ext_quad_valid.bool()
	return corner_quad_valid.nonzero(as_tuple=False)


def _ext_quad_bases_around_coords(
	coords: torch.Tensor,
	shape: tuple[int, int],
	*,
	search_ring: int,
) -> tuple[torch.Tensor, torch.Tensor]:
	C = int(coords.shape[0])
	H, W = (int(v) for v in shape)
	device = coords.device
	if C == 0 or H < 2 or W < 2:
		return (
			torch.empty(C, 0, 2, device=device, dtype=torch.long),
			torch.zeros(C, 0, device=device, dtype=torch.bool),
		)
	r = max(0, int(search_ring))
	offs = torch.arange(-r, r + 1, device=device, dtype=torch.long)
	off_h, off_w = torch.meshgrid(offs, offs, indexing="ij")
	hw_offsets = torch.stack([off_h.reshape(-1), off_w.reshape(-1)], dim=-1)
	K = int(hw_offsets.shape[0])
	finite = torch.isfinite(coords).all(dim=-1)
	safe_coords = torch.where(torch.isfinite(coords), coords, torch.zeros_like(coords))
	base_hw = torch.floor(safe_coords).to(dtype=torch.long)
	base_hw = torch.stack(
		[
			base_hw[:, 0].clamp(0, H - 2),
			base_hw[:, 1].clamp(0, W - 2),
		],
		dim=-1,
	)
	bases = base_hw[:, None, :] + hw_offsets.view(1, K, 2)
	in_bounds = (
		finite[:, None] &
		(bases[..., 0] >= 0) &
		(bases[..., 0] <= H - 2) &
		(bases[..., 1] >= 0) &
		(bases[..., 1] <= W - 2)
	)
	return bases, in_bounds


def _ext_quad_valid_at_bases(
	ext_valid: torch.Tensor,
	ext_quad_valid: torch.Tensor,
	bases: torch.Tensor,
	in_bounds: torch.Tensor,
) -> torch.Tensor:
	if bases.numel() == 0:
		return torch.zeros(bases.shape[:-1], device=ext_valid.device, dtype=torch.bool)
	H, W = int(ext_valid.shape[0]), int(ext_valid.shape[1])
	h = bases[..., 0].clamp(0, max(0, H - 2))
	w = bases[..., 1].clamp(0, max(0, W - 2))
	ok = (
		ext_valid[h, w] &
		ext_valid[h + 1, w] &
		ext_valid[h, w + 1] &
		ext_valid[h + 1, w + 1]
	)
	if tuple(ext_quad_valid.shape) == (max(0, H - 1), max(0, W - 1)):
		ok = ok & ext_quad_valid[h, w]
	return ok & in_bounds


def _intersect_ray_quad_candidates(
	*,
	source_pos: torch.Tensor,
	source_normals: torch.Tensor,
	bases: torch.Tensor,
	p00: torch.Tensor,
	p10: torch.Tensor,
	p01: torch.Tensor,
	p11: torch.Tensor,
	cfg: SnapSurfConfig,
	candidate_valid: torch.Tensor | None = None,
	hint_u: torch.Tensor | None = None,
	hint_v: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int | float]]:
	C = int(source_pos.shape[0])
	device = source_pos.device
	dtype = source_pos.dtype
	coords_empty = torch.full((C, 2), float("nan"), device=device, dtype=dtype)
	accepted_empty = torch.zeros(C, device=device, dtype=torch.bool)
	stats: dict[str, int | float] = {
		"target_hit": 0,
		"distance_hit": 0,
		"accepted": 0,
		"tested": 0,
		"line_err_sum": 0.0,
		"line_err_max": 0.0,
	}
	if C == 0:
		return coords_empty, accepted_empty, stats
	if bases.ndim != 3:
		raise ValueError(f"expected per-source candidate bases, got shape {tuple(bases.shape)}")
	K = int(bases.shape[1])
	stats["tested"] = C * K
	if K == 0:
		return coords_empty, accepted_empty, stats
	if candidate_valid is None:
		candidate_valid = torch.ones(C, K, device=device, dtype=torch.bool)
	else:
		candidate_valid = candidate_valid.bool()
	if hint_u is None:
		hint_u = torch.full((C, K), 0.5, device=device, dtype=dtype)
	if hint_v is None:
		hint_v = torch.full((C, K), 0.5, device=device, dtype=dtype)

	n = F.normalize(source_normals, dim=-1, eps=1.0e-8)
	O = source_pos[:, None, :]
	N = n[:, None, :]
	u, v = opt_loss_winding_density.ray_bilinear_intersect_refined(
		O,
		N,
		p00,
		p10,
		p01,
		p11,
		hint_u,
		hint_v,
		passes=2,
	)
	a = p10 - p00
	b = p01 - p00
	c = p11 - p10 - p01 + p00
	q = p00 + u.unsqueeze(-1) * a + v.unsqueeze(-1) * b + (u * v).unsqueeze(-1) * c
	delta = q - O
	signed = (delta * N).sum(dim=-1)
	abs_signed = signed.abs()
	line_delta = delta - signed.unsqueeze(-1) * N
	line_err = line_delta.norm(dim=-1)
	finite = (
		candidate_valid &
		torch.isfinite(source_pos).all(dim=-1)[:, None] &
		torch.isfinite(n).all(dim=-1)[:, None] &
		(n.norm(dim=-1) > 1.0e-8)[:, None] &
		torch.isfinite(q).all(dim=-1) &
		torch.isfinite(u) &
		torch.isfinite(v) &
		torch.isfinite(abs_signed) &
		torch.isfinite(line_err)
	)
	uv_tol = 1.0e-4
	uv_ok = (
		(u >= -uv_tol) & (u <= 1.0 + uv_tol) &
		(v >= -uv_tol) & (v <= 1.0 + uv_tol)
	)
	hit = finite & uv_ok
	has_hit = hit.any(dim=1)
	stats["target_hit"] = int(has_hit.sum().detach().cpu())
	if not bool(has_hit.any().detach().cpu()):
		return coords_empty, accepted_empty, stats

	line_hit = hit & (line_err <= float(cfg.ray_residual))
	has_line_hit = line_hit.any(dim=1)
	if not bool(has_line_hit.any().detach().cpu()):
		return coords_empty, accepted_empty, stats

	score = torch.where(line_hit, abs_signed, torch.full_like(abs_signed, float("inf")))
	best_k = torch.argmin(score, dim=1)
	row = torch.arange(C, device=device)
	best_dist = score[row, best_k]
	best_line_err = line_err[row, best_k]
	accepted = has_line_hit & torch.isfinite(best_dist)
	stats["distance_hit"] = int(accepted.sum().detach().cpu())
	stats["accepted"] = int(accepted.sum().detach().cpu())
	best_bases = bases[row, best_k].to(dtype=dtype)
	best_u = u[row, best_k].clamp(0.0, 1.0)
	best_v = v[row, best_k].clamp(0.0, 1.0)
	coords = best_bases + torch.stack([best_u, best_v], dim=-1)
	coords = torch.where(accepted.unsqueeze(-1), coords, torch.full_like(coords, float("nan")))
	if bool(accepted.any().detach().cpu()):
		err = best_line_err[accepted].detach()
		stats["line_err_sum"] = float(err.sum().cpu())
		stats["line_err_max"] = float(err.max().cpu())
	return coords, accepted, stats


def _intersect_model_points_with_ext_surface_chunk(
	*,
	source_pos: torch.Tensor,
	source_normals: torch.Tensor,
	bases: torch.Tensor,
	p00: torch.Tensor,
	p10: torch.Tensor,
	p01: torch.Tensor,
	p11: torch.Tensor,
	cfg: SnapSurfConfig,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int | float]]:
	C = int(source_pos.shape[0])
	device = source_pos.device
	dtype = source_pos.dtype
	coords_empty = torch.full((C, 2), float("nan"), device=device, dtype=dtype)
	accepted_empty = torch.zeros(C, device=device, dtype=torch.bool)
	stats: dict[str, int | float] = {
		"target_hit": 0,
		"distance_hit": 0,
		"accepted": 0,
		"tested": 0,
		"line_err_sum": 0.0,
		"line_err_max": 0.0,
	}
	if C == 0:
		return coords_empty, accepted_empty, stats

	K = int(bases.shape[0])
	stats["tested"] = C * K
	if K == 0:
		return coords_empty, accepted_empty, stats
	return _intersect_ray_quad_candidates(
		source_pos=source_pos,
		source_normals=source_normals,
		bases=bases.view(1, K, 2).expand(C, K, 2),
		p00=p00.view(1, K, 3).expand(C, K, 3),
		p10=p10.view(1, K, 3).expand(C, K, 3),
		p01=p01.view(1, K, 3).expand(C, K, 3),
		p11=p11.view(1, K, 3).expand(C, K, 3),
		cfg=cfg,
	)


def _intersect_model_points_with_ext_surface_near_coords(
	*,
	source_pos: torch.Tensor,
	source_normals: torch.Tensor,
	pred_coords: torch.Tensor,
	ext_xyz: torch.Tensor,
	ext_valid: torch.Tensor,
	ext_quad_valid: torch.Tensor,
	cfg: SnapSurfConfig,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int | float]]:
	C = int(source_pos.shape[0])
	device = source_pos.device
	dtype = source_pos.dtype
	coords_empty = torch.full((C, 2), float("nan"), device=device, dtype=dtype)
	accepted_empty = torch.zeros(C, device=device, dtype=torch.bool)
	if C == 0:
		return coords_empty, accepted_empty, {
			"target_hit": 0,
			"distance_hit": 0,
			"accepted": 0,
			"tested": 0,
			"line_err_sum": 0.0,
			"line_err_max": 0.0,
		}
	H, W = int(ext_xyz.shape[0]), int(ext_xyz.shape[1])
	if H < 2 or W < 2:
		return coords_empty, accepted_empty, {
			"target_hit": 0,
			"distance_hit": 0,
			"accepted": 0,
			"tested": 0,
			"line_err_sum": 0.0,
			"line_err_max": 0.0,
		}
	stats: dict[str, int | float] = {
		"target_hit": 0,
		"distance_hit": 0,
		"accepted": 0,
		"tested": C,
		"line_err_sum": 0.0,
		"line_err_max": 0.0,
	}
	n_all = F.normalize(source_normals, dim=-1, eps=1.0e-8)
	finite_pred = torch.isfinite(pred_coords).all(dim=-1)
	source_ok_all = (
		torch.isfinite(source_pos).all(dim=-1) &
		torch.isfinite(n_all).all(dim=-1) &
		(n_all.norm(dim=-1) > 1.0e-8)
	)
	pred_coord_ok = _quad_valid_at_coords(
		ext_valid.bool(),
		pred_coords,
		tuple(int(v) for v in ext_xyz.shape[:2]),
	) & finite_pred
	if tuple(ext_quad_valid.shape) == (max(0, H - 1), max(0, W - 1)):
		pred_bases, pred_in_bounds = _ext_quad_bases_around_coords(
			pred_coords,
			tuple(int(v) for v in ext_xyz.shape[:2]),
			search_ring=0,
		)
		pred_coord_ok &= _ext_quad_valid_at_bases(
			ext_valid.bool(),
			ext_quad_valid.bool(),
			pred_bases,
			pred_in_bounds,
		).reshape(C)
	safe_pred_for_sample = torch.where(torch.isfinite(pred_coords), pred_coords, torch.zeros_like(pred_coords))
	q_pred = _sample_surface_grid(ext_xyz, safe_pred_for_sample)
	delta_pred = q_pred - source_pos
	signed_pred = (delta_pred * n_all).sum(dim=-1)
	line_err_pred = (delta_pred - signed_pred.unsqueeze(-1) * n_all).norm(dim=-1)
	pred_accept = (
		source_ok_all &
		pred_coord_ok &
		torch.isfinite(q_pred).all(dim=-1) &
		torch.isfinite(line_err_pred) &
		(line_err_pred <= float(cfg.ray_residual))
	)
	if bool(pred_accept.any().detach().cpu()):
		coords_empty[pred_accept] = pred_coords[pred_accept]
		accepted_empty[pred_accept] = True
		err = line_err_pred[pred_accept].detach()
		stats["target_hit"] += int(pred_accept.sum().detach().cpu())
		stats["distance_hit"] += int(pred_accept.sum().detach().cpu())
		stats["accepted"] += int(pred_accept.sum().detach().cpu())
		stats["line_err_sum"] += float(err.sum().cpu())
		stats["line_err_max"] = max(float(stats["line_err_max"]), float(err.max().cpu()))
	remaining = ~pred_accept
	if not bool(remaining.any().detach().cpu()):
		return coords_empty, accepted_empty, stats

	out_coords = coords_empty
	out_accepted = accepted_empty
	rem_idx = remaining.nonzero(as_tuple=False).flatten()
	source_pos = source_pos[rem_idx]
	source_normals = source_normals[rem_idx]
	pred_coords = pred_coords[rem_idx]
	C = int(source_pos.shape[0])
	stats["tested"] += 2 * C
	finite_pred = torch.isfinite(pred_coords).all(dim=-1)
	safe_pred = torch.where(torch.isfinite(pred_coords), pred_coords, torch.zeros_like(pred_coords))
	base0_h = torch.floor(safe_pred[:, 0].clamp(0.0, float(H - 1))).clamp(0, H - 2).long()
	base0_w = torch.floor(safe_pred[:, 1].clamp(0.0, float(W - 1))).clamp(0, W - 2).long()
	base0 = torch.stack([base0_h, base0_w], dim=-1)
	frac0_h = (safe_pred[:, 0] - base0_h.to(dtype=dtype)).clamp(0.0, 1.0)
	frac0_w = (safe_pred[:, 1] - base0_w.to(dtype=dtype)).clamp(0.0, 1.0)
	base0_valid = _ext_quad_valid_at_bases(
		ext_valid.bool(),
		ext_quad_valid.bool(),
		base0.view(C, 1, 2),
		finite_pred.view(C, 1),
	).view(C)

	n = F.normalize(source_normals, dim=-1, eps=1.0e-8)
	source_ok = (
		torch.isfinite(source_pos).all(dim=-1) &
		torch.isfinite(n).all(dim=-1) &
		(n.norm(dim=-1) > 1.0e-8)
	)
	p00, p10, p01, p11 = _quad_corners_batched(ext_xyz, base0)
	u1, v1 = fit_model.Model3D._ray_bilinear_intersect(
		source_pos,
		n,
		p00,
		p10,
		p01,
		p11,
		frac0_h,
		frac0_w,
	)
	pass1_valid = source_ok & base0_valid & torch.isfinite(u1) & torch.isfinite(v1)
	coord1_h_raw = base0_h.to(dtype=dtype) + u1
	coord1_w_raw = base0_w.to(dtype=dtype) + v1
	coord1_h = torch.where(pass1_valid, coord1_h_raw, torch.zeros_like(coord1_h_raw)).clamp(0.0, float(H - 1))
	coord1_w = torch.where(pass1_valid, coord1_w_raw, torch.zeros_like(coord1_w_raw)).clamp(0.0, float(W - 1))
	base1_h = torch.floor(coord1_h).clamp(0, H - 2).long()
	base1_w = torch.floor(coord1_w).clamp(0, W - 2).long()
	base1 = torch.stack([base1_h, base1_w], dim=-1)
	frac1_h = coord1_h - base1_h.to(dtype=dtype)
	frac1_w = coord1_w - base1_w.to(dtype=dtype)
	base1_valid = _ext_quad_valid_at_bases(
		ext_valid.bool(),
		ext_quad_valid.bool(),
		base1.view(C, 1, 2),
		pass1_valid.view(C, 1),
	).view(C)

	p00, p10, p01, p11 = _quad_corners_batched(ext_xyz, base1)
	u2, v2 = fit_model.Model3D._ray_bilinear_intersect(
		source_pos,
		n,
		p00,
		p10,
		p01,
		p11,
		frac1_h,
		frac1_w,
	)
	a = p10 - p00
	b = p01 - p00
	c = p11 - p10 - p01 + p00
	q = p00 + u2.unsqueeze(-1) * a + v2.unsqueeze(-1) * b + (u2 * v2).unsqueeze(-1) * c
	delta = q - source_pos
	signed = (delta * n).sum(dim=-1)
	line_delta = delta - signed.unsqueeze(-1) * n
	line_err = line_delta.norm(dim=-1)
	uv_tol = 1.0e-4
	uv_ok = (
		(u2 >= -uv_tol) & (u2 <= 1.0 + uv_tol) &
		(v2 >= -uv_tol) & (v2 <= 1.0 + uv_tol)
	)
	target_hit = (
		source_ok &
		pass1_valid &
		base1_valid &
		torch.isfinite(q).all(dim=-1) &
		torch.isfinite(u2) &
		torch.isfinite(v2) &
		torch.isfinite(line_err) &
		uv_ok
	)
	stats["target_hit"] += int(target_hit.sum().detach().cpu())
	accepted = target_hit & (line_err <= float(cfg.ray_residual))
	stats["distance_hit"] += int(accepted.sum().detach().cpu())
	stats["accepted"] += int(accepted.sum().detach().cpu())
	coords = base1.to(dtype=dtype) + torch.stack([u2.clamp(0.0, 1.0), v2.clamp(0.0, 1.0)], dim=-1)
	coords = torch.where(accepted.unsqueeze(-1), coords, torch.full_like(coords, float("nan")))
	if bool(accepted.any().detach().cpu()):
		err = line_err[accepted].detach()
		stats["line_err_sum"] += float(err.sum().cpu())
		stats["line_err_max"] = max(float(stats["line_err_max"]), float(err.max().cpu()))
	out_coords[rem_idx] = coords
	out_accepted[rem_idx] = accepted
	return out_coords, out_accepted, stats


def _intersect_model_points_with_ext_surface(
	*,
	source_pos: torch.Tensor,
	source_normals: torch.Tensor,
	ext_xyz: torch.Tensor,
	ext_valid: torch.Tensor,
	ext_quad_valid: torch.Tensor,
	cfg: SnapSurfConfig,
	pair_chunk_limit: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int | float]]:
	C = int(source_pos.shape[0])
	device = source_pos.device
	dtype = source_pos.dtype
	coords_all = torch.full((C, 2), float("nan"), device=device, dtype=dtype)
	accepted_all = torch.zeros(C, device=device, dtype=torch.bool)
	stats: dict[str, int | float] = {
		"target_hit": 0,
		"distance_hit": 0,
		"accepted": 0,
		"tested": 0,
		"line_err_sum": 0.0,
		"line_err_max": 0.0,
	}
	if C == 0:
		return coords_all, accepted_all, stats

	bases = _valid_ext_quad_bases(ext_valid, ext_quad_valid)
	K = int(bases.shape[0])
	stats["tested"] = C * K
	if K == 0:
		return coords_all, accepted_all, stats
	p00, p10, p01, p11 = _quad_corners_batched(ext_xyz, bases)
	pair_limit = int(cfg.brute_pair_chunk_limit if pair_chunk_limit is None else pair_chunk_limit)
	chunk = max(1, min(C, max(1, pair_limit) // max(1, K)))
	for start in range(0, C, chunk):
		end = min(C, start + chunk)
		coords, accepted, part = _intersect_model_points_with_ext_surface_chunk(
			source_pos=source_pos[start:end],
			source_normals=source_normals[start:end],
			bases=bases,
			p00=p00,
			p10=p10,
			p01=p01,
			p11=p11,
			cfg=cfg,
		)
		coords_all[start:end] = coords
		accepted_all[start:end] = accepted
		stats["target_hit"] += int(part.get("target_hit", 0))
		stats["distance_hit"] += int(part.get("distance_hit", 0))
		stats["accepted"] += int(part.get("accepted", 0))
		stats["line_err_sum"] += float(part.get("line_err_sum", 0.0))
		stats["line_err_max"] = max(float(stats["line_err_max"]), float(part.get("line_err_max", 0.0)))
	return coords_all, accepted_all, stats


def _intersect_model_points_with_ext_surface_incremental(
	*,
	source_pos: torch.Tensor,
	source_normals: torch.Tensor,
	prev_coords: torch.Tensor | None,
	allow_brute: bool,
	brute_source_mask: torch.Tensor | None,
	ext_xyz: torch.Tensor,
	ext_valid: torch.Tensor,
	ext_quad_valid: torch.Tensor,
	cfg: SnapSurfConfig,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int | float]]:
	C = int(source_pos.shape[0])
	device = source_pos.device
	dtype = source_pos.dtype
	coords_all = torch.full((C, 2), float("nan"), device=device, dtype=dtype)
	accepted_all = torch.zeros(C, device=device, dtype=torch.bool)
	stats: dict[str, int | float] = {
		"target_hit": 0,
		"distance_hit": 0,
		"accepted": 0,
		"tested": 0,
		"line_err_sum": 0.0,
		"line_err_max": 0.0,
		"local_accepted": 0,
		"brute_sources": 0,
		"brute_front": 0,
		"brute_allowed": int(bool(allow_brute)),
	}
	if C == 0:
		return coords_all, accepted_all, stats

	prev_ok = torch.zeros(C, device=device, dtype=torch.bool)
	if prev_coords is not None and prev_coords.shape == coords_all.shape:
		prev_ok = torch.isfinite(prev_coords).all(dim=-1)
	if bool(prev_ok.any().detach().cpu()):
		local_idx = prev_ok.nonzero(as_tuple=False).flatten()
		coords, accepted, part = _intersect_model_points_with_ext_surface_near_coords(
			source_pos=source_pos[local_idx],
			source_normals=source_normals[local_idx],
			pred_coords=prev_coords[local_idx],
			ext_xyz=ext_xyz,
			ext_valid=ext_valid,
			ext_quad_valid=ext_quad_valid,
			cfg=cfg,
		)
		coords_all[local_idx] = coords
		accepted_all[local_idx] = accepted
		stats["target_hit"] += int(part.get("target_hit", 0))
		stats["distance_hit"] += int(part.get("distance_hit", 0))
		stats["accepted"] += int(part.get("accepted", 0))
		stats["tested"] += int(part.get("tested", 0))
		stats["line_err_sum"] += float(part.get("line_err_sum", 0.0))
		stats["line_err_max"] = max(float(stats["line_err_max"]), float(part.get("line_err_max", 0.0)))
		stats["local_accepted"] = int(part.get("accepted", 0))

	if brute_source_mask is None:
		brute_mask = torch.ones(C, device=device, dtype=torch.bool)
	else:
		brute_mask = brute_source_mask.to(device=device).bool()
		if tuple(brute_mask.shape) != (C,):
			raise ValueError(f"expected brute_source_mask shape {(C,)}, got {tuple(brute_mask.shape)}")
	stats["brute_front"] = int(brute_mask.sum().detach().cpu())
	remaining = (~accepted_all) & brute_mask
	if not bool(allow_brute):
		remaining = remaining & torch.zeros_like(remaining)
	stats["brute_sources"] = int(remaining.sum().detach().cpu())
	if bool(remaining.any().detach().cpu()):
		rem_idx = remaining.nonzero(as_tuple=False).flatten()
		coords, accepted, part = _intersect_model_points_with_ext_surface(
			source_pos=source_pos[rem_idx],
			source_normals=source_normals[rem_idx],
			ext_xyz=ext_xyz,
			ext_valid=ext_valid,
			ext_quad_valid=ext_quad_valid,
			cfg=cfg,
		)
		coords_all[rem_idx] = coords
		accepted_all[rem_idx] = accepted
		stats["target_hit"] += int(part.get("target_hit", 0))
		stats["distance_hit"] += int(part.get("distance_hit", 0))
		stats["accepted"] += int(part.get("accepted", 0))
		stats["tested"] += int(part.get("tested", 0))
		stats["line_err_sum"] += float(part.get("line_err_sum", 0.0))
		stats["line_err_max"] = max(float(stats["line_err_max"]), float(part.get("line_err_max", 0.0)))
	return coords_all, accepted_all, stats


def _rebuild_model_to_ext_rays(
	state: _SurfaceState,
	*,
	model_xyz_det: torch.Tensor,
	model_valid: torch.Tensor,
	model_normals_det: torch.Tensor,
	ext_xyz: torch.Tensor,
	ext_valid: torch.Tensor,
	ext_quad_valid: torch.Tensor,
	cfg: SnapSurfConfig,
	seed_xyz: tuple[float, float, float],
) -> tuple[bool, float, dict[str, int | float], int]:
	stats = _empty_grow_stats()
	stats["tested"] = 0
	stats["line_err_sum"] = 0.0
	stats["line_err_max"] = 0.0
	seed = torch.tensor(seed_xyz, device=model_xyz_det.device, dtype=model_xyz_det.dtype)
	seed_quad, seed_model_dist = _closest_model_surface_quad(
		point=seed,
		model_xyz=model_xyz_det,
		model_valid=model_valid,
	)
	state.model_to_ext.seed_base_idx = seed_quad
	state.model_to_ext.orientation_sign = 1
	if seed_quad is None:
		_clear_direction_state(state.model_to_ext)
		_clear_direction_state(state.ext_to_model)
		state.ext_to_model.seed_base_idx = None
		return False, float("inf"), stats, 0

	source_mask = _seed_model_sheet_mask(
		model_valid=model_valid,
		model_normals=model_normals_det,
		seed_quad=seed_quad,
	)
	source_idx = source_mask.nonzero(as_tuple=False)
	source_possible = int(source_idx.shape[0])
	stats["ring"] = source_possible
	stats["sup"] = source_possible
	if source_possible == 0:
		_clear_direction_state(state.model_to_ext)
		_clear_direction_state(state.ext_to_model)
		state.ext_to_model.seed_base_idx = None
		return True, seed_model_dist, stats, 0

	prev_coords = None
	prev_valid_full = None
	if state.model_to_ext.map is not None and state.model_to_ext.valid is not None:
		prev_valid_full = state.model_to_ext.valid.clone()
		prev_coords = state.model_to_ext.map[source_idx[:, 0], source_idx[:, 1], source_idx[:, 2]].clone()
	brute_front_full = _brute_source_front_mask(
		prev_valid=prev_valid_full,
		source_mask=source_mask,
		seed_quad=seed_quad,
		radius=cfg.brute_boundary_radius,
	)
	brute_source_mask = brute_front_full[source_idx[:, 0], source_idx[:, 1], source_idx[:, 2]]
	allow_brute = (int(state.snap_eval_count) % int(cfg.brute_interval)) == 0

	_clear_direction_state(state.model_to_ext)
	_clear_direction_state(state.ext_to_model)
	state.ext_to_model.seed_base_idx = None

	source_pos = _points_at_indices(model_xyz_det, source_idx)
	source_normals = _points_at_indices(model_normals_det, source_idx)
	coords, accepted, ray_stats = _intersect_model_points_with_ext_surface_incremental(
		source_pos=source_pos,
		source_normals=source_normals,
		prev_coords=prev_coords,
		allow_brute=allow_brute,
		brute_source_mask=brute_source_mask,
		ext_xyz=ext_xyz,
		ext_valid=ext_valid,
		ext_quad_valid=ext_quad_valid,
		cfg=cfg,
	)
	stats["tgt"] = int(ray_stats.get("target_hit", 0))
	raw_accepted = int(ray_stats.get("accepted", 0))
	stats["dist"] = raw_accepted
	stats["tested"] = int(ray_stats.get("tested", 0))
	stats["local"] = int(ray_stats.get("local_accepted", 0))
	stats["brute"] = int(ray_stats.get("brute_sources", 0))
	stats["front"] = int(ray_stats.get("brute_front", 0))
	stats["brute_on"] = int(ray_stats.get("brute_allowed", 0))
	stats["gerr_n"] = raw_accepted
	stats["gerr_sum"] = float(ray_stats.get("line_err_sum", 0.0))
	stats["gerr_max"] = float(ray_stats.get("line_err_max", 0.0))
	if state.model_to_ext.map is not None and state.model_to_ext.valid is not None and bool(accepted.any().detach().cpu()):
		raw_map = torch.full_like(state.model_to_ext.map, float("nan"))
		raw_valid = torch.zeros_like(state.model_to_ext.valid)
		raw_normal_dist = torch.full(raw_valid.shape, float("nan"), device=model_xyz_det.device, dtype=model_xyz_det.dtype)
		if prev_coords is not None:
			prev_finite = torch.isfinite(prev_coords).all(dim=-1)
			if bool(prev_finite.any().detach().cpu()):
				prev_idx = source_idx[prev_finite]
				raw_map[prev_idx[:, 0], prev_idx[:, 1], prev_idx[:, 2]] = prev_coords[prev_finite].to(dtype=raw_map.dtype)
		acc_idx = source_idx[accepted]
		raw_map[acc_idx[:, 0], acc_idx[:, 1], acc_idx[:, 2]] = coords[accepted].to(dtype=raw_map.dtype)
		raw_valid[acc_idx[:, 0], acc_idx[:, 1], acc_idx[:, 2]] = True
		acc_target = _sample_surface_grid(ext_xyz, coords[accepted]).detach()
		acc_normals = F.normalize(source_normals[accepted], dim=-1, eps=1.0e-8)
		acc_normal_dist = ((source_pos[accepted] - acc_target) * acc_normals).sum(dim=-1).abs()
		raw_normal_dist[acc_idx[:, 0], acc_idx[:, 1], acc_idx[:, 2]] = acc_normal_dist.to(dtype=raw_normal_dist.dtype)
		initial_inlier = _closest_seed_source_mask(
			raw_valid=raw_valid,
			model_xyz=model_xyz_det,
			seed=seed,
		)
		inlier_valid, map_stats = _seeded_mapping_inlier_filter(
			raw_valid=raw_valid,
			raw_map=raw_map,
			initial_inlier=initial_inlier,
			max_distance=cfg.map_inlier_distance,
			normal_dist=raw_normal_dist,
			max_normal_ratio=cfg.inlier_normal_distance_ratio,
			normal_distance_floor=cfg.inlier_normal_distance_floor,
		)
		final_n = int(inlier_valid.sum().detach().cpu())
		stats["drop"] = max(0, raw_accepted - final_n)
		stats["grid"] = final_n
		stats["ori"] = final_n
		stats["new"] = final_n
		for k, v in map_stats.items():
			stats[k] = v
		state.model_to_ext.map[:] = raw_map
		state.model_to_ext.valid[:] = inlier_valid
	else:
		if state.model_to_ext.map is not None and prev_coords is not None:
			state.model_to_ext.map[:] = float("nan")
			state.model_to_ext.map[source_idx[:, 0], source_idx[:, 1], source_idx[:, 2]] = prev_coords
		if state.model_to_ext.valid is not None:
			state.model_to_ext.valid[:] = False
		stats["grid"] = 0
		stats["ori"] = 0
		stats["new"] = 0
	state.snap_eval_count += 1
	return True, seed_model_dist, stats, source_possible


def _direction_loss_model_ray_to_ext(
	state: _DirectionState,
	*,
	model_xyz: torch.Tensor,
	model_normals: torch.Tensor,
	ext_xyz: torch.Tensor,
	ext_valid: torch.Tensor,
	cfg: SnapSurfConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, dict[str, float]]:
	device = model_xyz.device
	dtype = model_xyz.dtype
	lm = torch.zeros(model_xyz.shape[:3], device=device, dtype=dtype)
	mask = torch.zeros(model_xyz.shape[:3], device=device, dtype=dtype)
	if state.map is None or state.valid is None or state.count() == 0:
		z = model_xyz.new_zeros(())
		return z, lm.unsqueeze(1), mask.unsqueeze(1), 0, _empty_residual_stats()
	idx = state.valid.nonzero(as_tuple=False)
	all_coords = state.map[state.valid]
	coord_ok = torch.isfinite(all_coords).all(dim=-1) & _quad_valid_at_coords(
		ext_valid.bool(),
		all_coords,
		tuple(int(v) for v in ext_xyz.shape[:2]),
	)
	if not bool(coord_ok.any().detach().cpu()):
		return model_xyz.new_zeros(()), lm.unsqueeze(1), mask.unsqueeze(1), 0, _empty_residual_stats()
	idx = idx[coord_ok]
	coords = all_coords[coord_ok]
	src = _points_at_indices(model_xyz, idx)
	tgt = _sample_surface_grid(ext_xyz, coords).detach()
	n_raw = _points_at_indices(model_normals.detach(), idx)
	n = F.normalize(n_raw, dim=-1, eps=1.0e-8)
	raw_residual = ((src - tgt) * n).sum(dim=-1)
	residual = raw_residual / cfg.distance_scale
	values = _huber(residual, delta=cfg.huber_delta / cfg.distance_scale)
	finite = (
		torch.isfinite(src).all(dim=-1) &
		torch.isfinite(tgt).all(dim=-1) &
		torch.isfinite(n_raw).all(dim=-1) &
		torch.isfinite(n).all(dim=-1) &
		(n.norm(dim=-1) > 1.0e-8) &
		torch.isfinite(raw_residual) &
		torch.isfinite(values)
	)
	values_f = values[finite]
	loss = values_f.mean() if values_f.numel() else model_xyz.new_zeros(())
	if bool(finite.any().detach().cpu()):
		lm_idx = idx[finite]
		lm[lm_idx[:, 0], lm_idx[:, 1], lm_idx[:, 2]] = values_f.detach()
		mask[lm_idx[:, 0], lm_idx[:, 1], lm_idx[:, 2]] = 1.0
	stats = _residual_stats(raw_residual, residual, delta=cfg.huber_delta / cfg.distance_scale)
	return loss, lm.unsqueeze(1), mask.unsqueeze(1), int(values_f.numel()), stats


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


def snap_surf_loss(*, res: fit_model.FitResult3D) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
	"""Stateful external-surface snapping loss with explicit grown correspondences."""
	global _last_stats
	cfg = _cfg
	device = res.xyz_lr.device
	dtype = res.xyz_lr.dtype
	if not _active:
		z = res.xyz_lr.sum() * 0.0
		return z, (z.reshape(1, 1, 1, 1),), (z.reshape(1, 1, 1, 1),)
	if _seed_xyz is None:
		raise RuntimeError("snap_surf requires args.seed")
	records = _surface_records_from_res(res)
	if not records:
		raise RuntimeError("snap_surf requires at least one external_surfaces entry")

	if res.normals is None:
		raise RuntimeError("snap_surf requires model normals")
	model_xyz_det = res.xyz_lr.detach()
	model_normals_det = res.normals.detach()
	model_valid = torch.isfinite(model_xyz_det).all(dim=-1)

	while len(_states) < len(records):
		_states.append(_SurfaceState())

	total = res.xyz_lr.new_zeros(())
	total_weight = 0.0
	lm_accum = torch.zeros(res.xyz_lr.shape[:3], device=device, dtype=dtype).unsqueeze(1)
	mask_accum = torch.zeros_like(lm_accum)
	stats = {
		"snaps_seed": 0.0,
		"snaps_sdist": float("inf"),
		"snaps_sext": float("inf"),
		"snaps_m2e": 0.0,
		"snaps_local": 0.0,
		"snaps_brute": 0.0,
		"snaps_front": 0.0,
		"snaps_brute_on": 0.0,
		"_snaps_tested": 0.0,
		"_snaps_gerr_n": 0.0,
		"_snaps_gerr_sum": 0.0,
		"_snaps_gerr_max": 0.0,
		"_snaps_res_n": 0.0,
		"_snaps_res_sum": 0.0,
		"_snaps_res_abs_sum": 0.0,
		"_snaps_res_abs_max": 0.0,
		"_snaps_toward_sum": 0.0,
	}
	total_model_possible = 0

	for si, (ext_xyz, ext_valid, ext_normals, ext_quad_valid) in enumerate(records):
		state = _states[si]
		ext_xyz = ext_xyz.to(device=device, dtype=dtype).detach()
		ext_valid = ext_valid.to(device=device).bool()
		ext_normals = ext_normals.to(device=device, dtype=dtype).detach()
		ext_quad_valid = ext_quad_valid.to(device=device).bool()
		ext_valid = ext_valid & torch.isfinite(ext_xyz).all(dim=-1)
		if int(ext_valid.shape[0]) > 1 and int(ext_valid.shape[1]) > 1:
			corner_quad_valid = (
				ext_valid[:-1, :-1] &
				ext_valid[1:, :-1] &
				ext_valid[:-1, 1:] &
				ext_valid[1:, 1:]
			)
			if tuple(ext_quad_valid.shape) == tuple(corner_quad_valid.shape):
				ext_quad_valid = ext_quad_valid & corner_quad_valid
			else:
				ext_quad_valid = corner_quad_valid
		state.ensure(
			model_shape=tuple(int(v) for v in res.xyz_lr.shape[:3]),
			ext_shape=tuple(int(v) for v in ext_xyz.shape[:2]),
			device=device,
			dtype=dtype,
		)

		with torch.no_grad():
			seed = torch.tensor(_seed_xyz, device=device, dtype=dtype)
			_ext_seed_hw, _ext_seed_point, seed_ext_dist = _closest_external_seed_surface(
				seed=seed,
				ext_xyz=ext_xyz,
				ext_valid=ext_valid,
				ext_quad_valid=ext_quad_valid,
			)
			seed_inserted, seed_dist, grow_m2e, source_possible = _rebuild_model_to_ext_rays(
				state=state,
				model_xyz_det=model_xyz_det,
				model_valid=model_valid,
				model_normals_det=model_normals_det,
				ext_xyz=ext_xyz,
				ext_valid=ext_valid,
				ext_quad_valid=ext_quad_valid,
				cfg=cfg,
				seed_xyz=_seed_xyz,
			)
			total_model_possible += int(source_possible)
			stats["snaps_sdist"] = min(stats["snaps_sdist"], float(seed_dist))
			stats["snaps_sext"] = min(stats["snaps_sext"], float(seed_ext_dist))
			if seed_inserted:
				stats["snaps_seed"] += 1.0
			for k in ("local", "brute", "front", "brute_on"):
				stats[f"snaps_{k}"] += float(grow_m2e.get(k, 0))
			stats["_snaps_tested"] += float(grow_m2e.get("tested", 0))
			stats["_snaps_gerr_n"] += float(grow_m2e.get("gerr_n", 0))
			stats["_snaps_gerr_sum"] += float(grow_m2e.get("gerr_sum", 0.0))
			stats["_snaps_gerr_max"] = max(
				stats["_snaps_gerr_max"],
				float(grow_m2e.get("gerr_max", 0.0)),
			)
			_debug_write_snap_objs(
				cfg=cfg,
				surface_index=si,
				surface_count=len(records),
				model_xyz=model_xyz_det,
				model_valid=model_valid,
				ext_xyz=ext_xyz,
				ext_valid=ext_valid,
				state=state,
			)

		l_m2e, lm_m2e, mask_m2e, n_m2e, rs_m2e = _direction_loss_model_ray_to_ext(
			state.model_to_ext,
			model_xyz=res.xyz_lr,
			model_normals=model_normals_det,
			ext_xyz=ext_xyz,
			ext_valid=ext_valid,
			cfg=cfg,
		)
		stats["_snaps_res_n"] += float(rs_m2e.get("n", 0.0))
		stats["_snaps_res_sum"] += float(rs_m2e.get("sum", 0.0))
		stats["_snaps_res_abs_sum"] += float(rs_m2e.get("abs_sum", 0.0))
		stats["_snaps_res_abs_max"] = max(stats["_snaps_res_abs_max"], float(rs_m2e.get("abs_max", 0.0)))
		stats["_snaps_toward_sum"] += float(rs_m2e.get("toward_sum", 0.0))
		stats["snaps_m2e"] += float(n_m2e)
		w_m2e = cfg.w_to_ext if n_m2e > 0 else 0.0
		if w_m2e > 0.0:
			total = total + l_m2e
			total_weight += 1.0
		lm_accum = lm_accum + lm_m2e
		mask_accum = (mask_accum + mask_m2e).clamp(max=1.0)

	if total_weight > 0.0:
		total = total / total_weight
	if not bool(torch.isfinite(total.detach()).all().cpu()):
		raise RuntimeError(f"snap_surf produced non-finite loss: stats={stats}")
	stats["snaps_m2e"] = _safe_frac(stats["snaps_m2e"], total_model_possible)
	stats["snaps_seed"] = _safe_frac(stats["snaps_seed"], len(records))
	stats["snaps_brute_on"] = _safe_frac(stats["snaps_brute_on"], len(records))
	tested = stats.pop("_snaps_tested")
	gerr_n = stats.pop("_snaps_gerr_n")
	gerr_sum = stats.pop("_snaps_gerr_sum")
	gerr_max = stats.pop("_snaps_gerr_max")
	res_n = stats.pop("_snaps_res_n")
	res_sum = stats.pop("_snaps_res_sum")
	res_abs_sum = stats.pop("_snaps_res_abs_sum")
	res_abs_max = stats.pop("_snaps_res_abs_max")
	toward_sum = stats.pop("_snaps_toward_sum")
	stats["snaps_gerr_avg"] = _safe_frac(gerr_sum, gerr_n)
	stats["snaps_gerr_max"] = float(gerr_max)
	stats["snaps_ravg"] = _safe_frac(res_sum, res_n)
	stats["snaps_rabs"] = _safe_frac(res_abs_sum, res_n)
	stats["snaps_rmax"] = float(res_abs_max)
	stats["snaps_tow"] = _safe_frac(toward_sum, res_n)
	stats["snaps_pairs_m"] = _safe_frac(tested, 1_000_000.0)
	_last_stats = stats
	return total, (lm_accum,), (mask_accum,)
