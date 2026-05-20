from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

import model as fit_model
import opt_loss_station


@dataclass(frozen=True)
class SnapSurfConfig:
	init_distance: float = 50.0
	point_distance: float = 25.0
	grid_error: float = 0.75
	affine_radius: int = 2
	search_ring: int = 1
	huber_delta: float = 5.0
	distance_scale: float = 1.0
	w_to_ext: float = 1.0
	w_to_model: float = 1.0
	orientation: str = "auto"


_cfg = SnapSurfConfig()
_active = False
_seed_xyz: tuple[float, float, float] | None = None
_states: list["_SurfaceState"] = []
_last_stats: dict[str, float] = {}


def reset_state() -> None:
	global _states, _last_stats
	_states = []
	_last_stats = {}


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
		huber_delta=float(raw.get("huber_delta", SnapSurfConfig.huber_delta)),
		distance_scale=max(1.0e-8, float(raw.get("distance_scale", SnapSurfConfig.distance_scale))),
		w_to_ext=float(raw.get("w_to_ext", SnapSurfConfig.w_to_ext)),
		w_to_model=float(raw.get("w_to_model", SnapSurfConfig.w_to_model)),
		orientation=str(raw.get("orientation", SnapSurfConfig.orientation)).strip().lower(),
	)
	if _cfg.init_distance < 0.0:
		raise ValueError("snap_surf args.init_distance must be >= 0")
	if _cfg.point_distance < 0.0:
		raise ValueError("snap_surf args.point_distance must be >= 0")
	if _cfg.grid_error < 0.0:
		raise ValueError("snap_surf args.grid_error must be >= 0")
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
	if state.source_rank == 3:
		state.valid[:] = valid_b
		state.map[:] = map_b
	else:
		state.valid[:] = valid_b[0]
		state.map[:] = map_b[0]
	state.map[~state.valid] = float("nan")


def _source_hw_grid(*, n: int, h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
	hh = torch.arange(h, device=device, dtype=dtype).view(1, h, 1).expand(n, h, w)
	ww = torch.arange(w, device=device, dtype=dtype).view(1, 1, w).expand(n, h, w)
	return torch.stack([hh, ww], dim=-1)


def _linear_ids(coords: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
	if len(shape) == 2:
		return coords[..., 0] * int(shape[1]) + coords[..., 1]
	return (coords[..., 0] * int(shape[1]) + coords[..., 1]) * int(shape[2]) + coords[..., 2]


def _gather_bool_at_coords(mask: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
	if coords.numel() == 0:
		return torch.zeros(coords.shape[:-1], device=mask.device, dtype=torch.bool)
	if coords.shape[-1] == 2:
		return mask[coords[..., 0], coords[..., 1]]
	return mask[coords[..., 0], coords[..., 1], coords[..., 2]]


def _gather_points_at_coords(grid: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
	if coords.numel() == 0:
		return torch.empty(*coords.shape[:-1], int(grid.shape[-1]), device=grid.device, dtype=grid.dtype)
	if coords.shape[-1] == 2:
		return grid[coords[..., 0], coords[..., 1]]
	return grid[coords[..., 0], coords[..., 1], coords[..., 2]]


def _quad_valid_at_coords(valid: torch.Tensor, coords: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
	if coords.numel() == 0:
		return torch.zeros(coords.shape[:-1], device=valid.device, dtype=torch.bool)
	flat = coords.reshape(-1, coords.shape[-1])
	finite = torch.isfinite(flat).all(dim=-1)
	if len(shape) == 2:
		H, W = int(shape[0]), int(shape[1])
		if H < 2 or W < 2:
			return torch.zeros(coords.shape[:-1], device=valid.device, dtype=torch.bool)
		h = flat[:, 0]
		w = flat[:, 1]
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
	d = flat[:, 0]
	h = flat[:, 1]
	w = flat[:, 2]
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


def _state_indices_from_batched(flat_idx: torch.Tensor, *, state: _DirectionState, h: int, w: int) -> torch.Tensor:
	n_idx = flat_idx // (h * w)
	rem = flat_idx - n_idx * h * w
	h_idx = rem // w
	w_idx = rem - h_idx * w
	if state.source_rank == 3:
		return torch.stack([n_idx, h_idx, w_idx], dim=-1)
	return torch.stack([h_idx, w_idx], dim=-1)


def _target_occupied_mask(state: _DirectionState, valid_b: torch.Tensor) -> torch.Tensor:
	if state.map is None or state.target_shape is None:
		raise RuntimeError("snap_surf direction state is not initialized")
	total = math.prod(state.target_shape)
	occupied = torch.zeros(total, device=state.map.device, dtype=torch.bool)
	if not bool(valid_b.any().detach().cpu()):
		return occupied
	map_b = state.map if state.source_rank == 3 else state.map.unsqueeze(0)
	coords = torch.round(map_b[valid_b]).to(dtype=torch.long)
	in_bounds = torch.ones(coords.shape[0], device=coords.device, dtype=torch.bool)
	for axis, size in enumerate(state.target_shape):
		in_bounds &= (coords[:, axis] >= 0) & (coords[:, axis] < int(size))
	if bool(in_bounds.any().detach().cpu()):
		occupied[_linear_ids(coords[in_bounds], state.target_shape)] = True
	return occupied


def _bool_at_index(mask: torch.Tensor, idx: tuple[int, ...]) -> bool:
	return bool(mask[idx].detach().cpu())


def _target_key(coord: torch.Tensor) -> tuple[int, ...]:
	return tuple(int(round(float(v))) for v in coord.detach().cpu().tolist())


def _used_target_keys(state: _DirectionState, valid_mask: torch.Tensor | None = None) -> set[tuple[int, ...]]:
	if state.map is None or state.valid is None:
		return set()
	use = state.valid if valid_mask is None else valid_mask
	out: set[tuple[int, ...]] = set()
	for idx in use.nonzero(as_tuple=False):
		coord = state.map[tuple(int(v) for v in idx.tolist())]
		if bool(torch.isfinite(coord).all().detach().cpu()):
			out.add(_target_key(coord))
	return out


def _clear_target_key(state: _DirectionState, key: tuple[int, ...]) -> None:
	if state.map is None or state.valid is None:
		return
	for idx in state.valid.nonzero(as_tuple=False):
		idx_t = tuple(int(v) for v in idx.tolist())
		if _target_key(state.map[idx_t]) == key:
			state.valid[idx_t] = False
			state.map[idx_t] = float("nan")


def _set_correspondence(state: _DirectionState, source_idx: tuple[int, ...], target_coord: torch.Tensor) -> None:
	if state.map is None or state.valid is None:
		return
	state.map[source_idx] = target_coord.to(device=state.map.device, dtype=state.map.dtype)
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


def _gather_supports(
	state: _DirectionState,
	idx: tuple[int, ...],
	*,
	radius: int,
	valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
	if state.map is None or state.valid is None:
		raise RuntimeError("snap_surf direction state is not initialized")
	valid = state.valid if valid_mask is None else valid_mask
	device = state.map.device
	dtype = state.map.dtype
	source_pts: list[torch.Tensor] = []
	target_pts: list[torch.Tensor] = []
	if state.source_rank == 3:
		d, h, w = idx
		_, H, W = state.source_shape
		for hh in range(max(0, h - radius), min(H - 1, h + radius) + 1):
			for ww in range(max(0, w - radius), min(W - 1, w + radius) + 1):
				j = (d, hh, ww)
				if j == idx or not _bool_at_index(valid, j):
					continue
				source_pts.append(torch.tensor([float(hh), float(ww)], device=device, dtype=dtype))
				target_pts.append(state.map[j])
	else:
		h, w = idx
		H, W = state.source_shape
		for hh in range(max(0, h - radius), min(H - 1, h + radius) + 1):
			for ww in range(max(0, w - radius), min(W - 1, w + radius) + 1):
				j = (hh, ww)
				if j == idx or not _bool_at_index(valid, j):
					continue
				source_pts.append(torch.tensor([float(hh), float(ww)], device=device, dtype=dtype))
				target_pts.append(state.map[j])
	if not source_pts:
		return (
			torch.empty(0, 2, device=device, dtype=dtype),
			torch.empty(0, state.target_rank, device=device, dtype=dtype),
		)
	return torch.stack(source_pts, dim=0), torch.stack(target_pts, dim=0)


def _orientation_ok(
	source_support: torch.Tensor,
	target_support: torch.Tensor,
	query_source_hw: torch.Tensor,
	query_target: torch.Tensor,
	*,
	orientation_sign: int,
) -> bool:
	if int(source_support.shape[0]) < 2:
		return True
	expected = 1 if int(orientation_sign) >= 0 else -1
	target_query_hw = query_target[-2:]
	target_hw = target_support[:, -2:]
	for i in range(int(source_support.shape[0])):
		for j in range(i + 1, int(source_support.shape[0])):
			a = source_support[i] - query_source_hw
			b = source_support[j] - query_source_hw
			ta = target_hw[i] - target_query_hw
			tb = target_hw[j] - target_query_hw
			det_s = a[0] * b[1] - a[1] * b[0]
			det_t = ta[0] * tb[1] - ta[1] * tb[0]
			if abs(float(det_s.detach().cpu())) <= 1.0e-6 or abs(float(det_t.detach().cpu())) <= 1.0e-6:
				continue
			got = 1 if float((det_s * det_t).detach().cpu()) >= 0.0 else -1
			if got != expected:
				return False
	return True


def _target_search_coords(
	pred: torch.Tensor,
	target_shape: tuple[int, ...],
	*,
	search_ring: int,
) -> list[tuple[int, ...]]:
	centers = [int(round(float(v))) for v in pred.detach().cpu().tolist()]
	ranges = []
	for c, size in zip(centers, target_shape):
		lo = max(0, c - search_ring)
		hi = min(int(size) - 1, c + search_ring)
		ranges.append(range(lo, hi + 1))
	if len(ranges) == 2:
		return [(h, w) for h in ranges[0] for w in ranges[1]]
	return [(d, h, w) for d in ranges[0] for h in ranges[1] for w in ranges[2]]


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
		orient_ok = torch.ones_like(check)
		if cfg.orientation != "none":
			expected = 1.0 if int(state.orientation_sign) >= 0 else -1.0
			orient_ok = (det * expected) >= -1.0e-6
		grid_fail = check & (grid_err > float(cfg.grid_error))
		if bool(grid_fail.any().detach().cpu()):
			fail_vals = grid_err[grid_fail]
			out["pgrid"] = int(fail_vals.numel())
			out["perr_n"] = int(fail_vals.numel())
			out["perr_sum"] = float(fail_vals.sum().detach().cpu())
			out["perr_max"] = float(fail_vals.max().detach().cpu())
		new_valid_b &= (~check) | ((grid_err <= float(cfg.grid_error)) & orient_ok)

	after_count = int(new_valid_b.sum().detach().cpu())
	_write_batched_state(state, new_valid_b, map_b)
	out["drop"] = max(0, int(start_count) - after_count)
	return out


def _empty_grow_stats() -> dict[str, int | float]:
	return {
		"ring": 0,
		"sup": 0,
		"tgt": 0,
		"dist": 0,
		"grid": 0,
		"ori": 0,
		"new": 0,
		"gerr_n": 0,
		"gerr_sum": 0.0,
		"gerr_max": 0.0,
	}


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
	n, h, w = base_valid.shape
	candidate_mask = _neighbor4_mask(base_valid) & source_valid_b & ~base_valid
	stats["ring"] = int(candidate_mask.sum().detach().cpu())
	if not bool(candidate_mask.any().detach().cpu()):
		return stats

	pred, count, det = _local_affine_predict_batched(
		state,
		valid_b=base_valid,
		map_b=map_b,
		radius=cfg.affine_radius,
		exclude_self=False,
	)
	candidate_mask &= count >= 2
	stats["sup"] = int(candidate_mask.sum().detach().cpu())
	if not bool(candidate_mask.any().detach().cpu()):
		return stats

	flat_mask = candidate_mask.reshape(-1)
	flat_ids = flat_mask.nonzero(as_tuple=False).squeeze(1)
	source_idx = _state_indices_from_batched(flat_ids, state=state, h=h, w=w)
	pred_flat = pred.reshape(-1, state.target_rank)[flat_ids]
	det_flat = det.reshape(-1)[flat_ids]

	offsets = []
	r = int(cfg.search_ring)
	if state.target_rank == 2:
		for dh in range(-r, r + 1):
			for dw in range(-r, r + 1):
				offsets.append((float(dh), float(dw)))
	else:
		for dh in range(-r, r + 1):
			for dw in range(-r, r + 1):
				offsets.append((0.0, float(dh), float(dw)))
	offset_t = torch.tensor(offsets, device=map_b.device, dtype=map_b.dtype)
	search_coords = pred_flat.unsqueeze(1) + offset_t.view(1, -1, state.target_rank)

	target_ok = _quad_valid_at_coords(target_valid.bool(), search_coords, state.target_shape)
	stats["tgt"] = int(target_ok.any(dim=1).sum().detach().cpu())

	grid_err = (search_coords - pred_flat.unsqueeze(1)).norm(dim=-1)
	target_pos = _sample_surface_grid(target_xyz, search_coords)
	source_pos = _points_at_indices(source_xyz, source_idx).unsqueeze(1)
	if normal_from_source:
		n = _points_at_indices(normal_xyz, source_idx).unsqueeze(1).expand_as(target_pos)
	else:
		n = _sample_surface_grid(normal_xyz, search_coords)
	n = F.normalize(n, dim=-1, eps=1.0e-8)
	dist = ((source_pos - target_pos) * n).sum(dim=-1).abs()
	target_geom_ok = (
		torch.isfinite(target_pos).all(dim=-1) &
		torch.isfinite(n).all(dim=-1) &
		(n.norm(dim=-1) > 1.0e-8)
	)
	dist_ok = target_ok & target_geom_ok & (dist <= float(cfg.point_distance))
	stats["dist"] = int(dist_ok.any(dim=1).sum().detach().cpu())
	best_dist_gerr = torch.where(dist_ok, grid_err, torch.full_like(grid_err, float("inf"))).min(dim=1).values
	finite_best = torch.isfinite(best_dist_gerr)
	if bool(finite_best.any().detach().cpu()):
		gerr_vals = best_dist_gerr[finite_best]
		stats["gerr_n"] = int(gerr_vals.numel())
		stats["gerr_sum"] = float(gerr_vals.sum().detach().cpu())
		stats["gerr_max"] = float(gerr_vals.max().detach().cpu())
	grid_ok = dist_ok & (grid_err <= float(cfg.grid_error))
	stats["grid"] = int(grid_ok.any(dim=1).sum().detach().cpu())
	ok = grid_ok
	if cfg.orientation != "none":
		expected = 1.0 if int(state.orientation_sign) >= 0 else -1.0
		ok = ok & ((det_flat * expected) >= -1.0e-6).unsqueeze(1)
	stats["ori"] = int(ok.any(dim=1).sum().detach().cpu())
	score = torch.where(ok, dist + grid_err, torch.full_like(dist, float("inf")))
	best_score, best_pos = score.min(dim=1)
	accepted = torch.isfinite(best_score)
	if not bool(accepted.any().detach().cpu()):
		return stats

	acc_source_idx = source_idx[accepted]
	acc_target_coords = search_coords[accepted, best_pos[accepted]]

	stats["new"] = int(acc_source_idx.shape[0])
	if acc_source_idx.numel() == 0:
		return stats
	map_out = map_b.clone()
	valid_out = base_valid.clone()
	if state.source_rank == 3:
		map_out[acc_source_idx[:, 0], acc_source_idx[:, 1], acc_source_idx[:, 2]] = acc_target_coords.to(dtype=map_b.dtype)
		valid_out[acc_source_idx[:, 0], acc_source_idx[:, 1], acc_source_idx[:, 2]] = True
	else:
		map_out[0, acc_source_idx[:, 0], acc_source_idx[:, 1]] = acc_target_coords.to(dtype=map_b.dtype)
		valid_out[0, acc_source_idx[:, 0], acc_source_idx[:, 1]] = True
	_write_batched_state(state, valid_out, map_out)
	return stats


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


def _closest_point_on_quad_coord(
	point: torch.Tensor,
	p00: torch.Tensor,
	p10: torch.Tensor,
	p01: torch.Tensor,
	p11: torch.Tensor,
	base_coord: torch.Tensor,
) -> torch.Tensor:
	cp0, bary0 = opt_loss_station._closest_points_on_triangles(
		point,
		p00.view(1, 3),
		p10.view(1, 3),
		p11.view(1, 3),
	)
	cp1, bary1 = opt_loss_station._closest_points_on_triangles(
		point,
		p00.view(1, 3),
		p11.view(1, 3),
		p01.view(1, 3),
	)
	d0 = (cp0[0] - point).square().sum()
	d1 = (cp1[0] - point).square().sum()
	if float(d0.detach().cpu()) <= float(d1.detach().cpu()):
		bary = bary0[0]
		h_frac = bary[1] + bary[2]
		w_frac = bary[2]
	else:
		bary = bary1[0]
		h_frac = bary[1]
		w_frac = bary[1] + bary[2]
	coord = base_coord.clone()
	coord[-2] = coord[-2] + h_frac
	coord[-1] = coord[-1] + w_frac
	return coord


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
	best = transforms[0]
	best_score = float("inf")
	for transform in transforms:
		score = 0.0
		for i, (sh, sw) in enumerate(_CORNERS_2D):
			th, tw = transform[i]
			mp = model_xyz[d, mh + sh, mw + sw]
			ep = ext_xyz[eh + th, ew + tw]
			score += float((mp - ep).norm().detach().cpu())
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
	ext_base = torch.tensor([float(eh), float(ew)], device=model_xyz.device, dtype=model_xyz.dtype)
	model_base = torch.tensor([float(d), float(mh), float(mw)], device=model_xyz.device, dtype=model_xyz.dtype)
	model_to_ext_seed: dict[tuple[int, int], torch.Tensor] = {}
	ext_to_model_seed: dict[tuple[int, int], torch.Tensor] = {}
	for sh, sw in _CORNERS_2D:
		model_point = model_xyz[d, mh + sh, mw + sw]
		model_to_ext_seed[(sh, sw)] = _closest_point_on_quad_coord(
			model_point,
			ext_xyz[eh, ew],
			ext_xyz[eh + 1, ew],
			ext_xyz[eh, ew + 1],
			ext_xyz[eh + 1, ew + 1],
			ext_base,
		)
	for th, tw in _CORNERS_2D:
		ext_point = ext_xyz[eh + th, ew + tw]
		ext_to_model_seed[(th, tw)] = _closest_point_on_quad_coord(
			ext_point,
			model_xyz[d, mh, mw],
			model_xyz[d, mh + 1, mw],
			model_xyz[d, mh, mw + 1],
			model_xyz[d, mh + 1, mw + 1],
			model_base,
		)
	a = model_to_ext_seed[(1, 0)] - model_to_ext_seed[(0, 0)]
	b = model_to_ext_seed[(0, 1)] - model_to_ext_seed[(0, 0)]
	det = float((a[0] * b[1] - a[1] * b[0]).detach().cpu())
	det_sign = 1 if det >= 0.0 else -1
	state.model_to_ext.orientation_sign = det_sign
	state.ext_to_model.orientation_sign = det_sign
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
	return True, surface_dist, state.seed_ext_distance


def _huber(residual: torch.Tensor, *, delta: float) -> torch.Tensor:
	abs_r = residual.abs()
	d = float(delta)
	return torch.where(abs_r <= d, 0.5 * residual.square(), d * (abs_r - 0.5 * d))


def _direction_loss_model_to_ext(
	state: _DirectionState,
	*,
	model_xyz: torch.Tensor,
	model_normals: torch.Tensor,
	ext_xyz: torch.Tensor,
	cfg: SnapSurfConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
	device = model_xyz.device
	dtype = model_xyz.dtype
	lm = torch.zeros(model_xyz.shape[:3], device=device, dtype=dtype)
	mask = torch.zeros(model_xyz.shape[:3], device=device, dtype=dtype)
	if state.map is None or state.valid is None or state.count() == 0:
		z = model_xyz.sum() * 0.0
		return z, lm.unsqueeze(1), mask.unsqueeze(1), 0
	idx = state.valid.nonzero(as_tuple=False)
	src = _points_at_indices(model_xyz, idx)
	coords = state.map[state.valid]
	tgt = _sample_surface_grid(ext_xyz, coords).detach()
	n = F.normalize(_points_at_indices(model_normals.detach(), idx), dim=-1, eps=1.0e-8)
	residual = ((src - tgt) * n).sum(dim=-1) / cfg.distance_scale
	values = _huber(residual, delta=cfg.huber_delta / cfg.distance_scale)
	loss = values.mean() if values.numel() else model_xyz.sum() * 0.0
	lm[state.valid] = values.detach()
	mask[state.valid] = 1.0
	return loss, lm.unsqueeze(1), mask.unsqueeze(1), int(values.numel())


def _direction_loss_ext_to_model(
	state: _DirectionState,
	*,
	model_xyz: torch.Tensor,
	model_normals: torch.Tensor,
	ext_xyz: torch.Tensor,
	cfg: SnapSurfConfig,
) -> tuple[torch.Tensor, int]:
	if state.map is None or state.valid is None or state.count() == 0:
		return model_xyz.sum() * 0.0, 0
	idx = state.valid.nonzero(as_tuple=False)
	src = _points_at_indices(ext_xyz, idx).detach()
	coords = state.map[state.valid]
	tgt = _sample_surface_grid(model_xyz, coords)
	n = F.normalize(_sample_surface_grid(model_normals.detach(), coords), dim=-1, eps=1.0e-8)
	residual = ((tgt - src) * n).sum(dim=-1) / cfg.distance_scale
	values = _huber(residual, delta=cfg.huber_delta / cfg.distance_scale)
	loss = values.mean() if values.numel() else model_xyz.sum() * 0.0
	return loss, int(values.numel())


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

	model_xyz_det = res.xyz_lr.detach()
	model_normals = res.normals
	if model_normals is None:
		model_normals = fit_model.Model3D._vertex_normals(res.xyz_lr).detach()
	model_valid = torch.isfinite(model_xyz_det).all(dim=-1)

	while len(_states) < len(records):
		_states.append(_SurfaceState())

	total = res.xyz_lr.sum() * 0.0
	total_weight = 0.0
	lm_accum = torch.zeros(res.xyz_lr.shape[:3], device=device, dtype=dtype).unsqueeze(1)
	mask_accum = torch.zeros_like(lm_accum)
	stats = {
		"snaps_seed": 0.0,
		"snaps_sdist": float("inf"),
		"snaps_sext": float("inf"),
		"snaps_m2e": 0.0,
		"snaps_e2m": 0.0,
		"snaps_drop": 0.0,
		"snaps_ring": 0.0,
		"snaps_sup": 0.0,
		"snaps_tgt": 0.0,
		"snaps_dist": 0.0,
		"snaps_grid": 0.0,
		"snaps_ori": 0.0,
		"snaps_new": 0.0,
		"snaps_pgrid": 0.0,
		"_snaps_gerr_n": 0.0,
		"_snaps_gerr_sum": 0.0,
		"_snaps_gerr_max": 0.0,
		"_snaps_perr_n": 0.0,
		"_snaps_perr_sum": 0.0,
		"_snaps_perr_max": 0.0,
	}
	total_model_possible = 0
	total_ext_possible = 0

	for si, (ext_xyz, ext_valid, ext_normals, ext_quad_valid) in enumerate(records):
		state = _states[si]
		ext_xyz = ext_xyz.to(device=device, dtype=dtype).detach()
		ext_valid = ext_valid.to(device=device).bool()
		ext_normals = ext_normals.to(device=device, dtype=dtype).detach()
		ext_quad_valid = ext_quad_valid.to(device=device).bool()
		state.ensure(
			model_shape=tuple(int(v) for v in res.xyz_lr.shape[:3]),
			ext_shape=tuple(int(v) for v in ext_xyz.shape[:2]),
			device=device,
			dtype=dtype,
		)
		total_model_possible += int(model_valid.sum().detach().cpu())
		total_ext_possible += int(ext_valid.sum().detach().cpu())

		with torch.no_grad():
			rv_m2e = _revalidate_direction(
				state.model_to_ext,
				source_xyz=model_xyz_det,
				source_valid=model_valid,
				target_xyz=ext_xyz,
				target_valid=ext_valid,
				normal_xyz=model_normals.detach(),
				normal_from_source=True,
				cfg=cfg,
			)
			rv_e2m = _revalidate_direction(
				state.ext_to_model,
				source_xyz=ext_xyz,
				source_valid=ext_valid,
				target_xyz=model_xyz_det,
				target_valid=model_valid,
				normal_xyz=model_normals.detach(),
				normal_from_source=False,
				cfg=cfg,
			)
			stats["snaps_drop"] += float(rv_m2e.get("drop", 0) + rv_e2m.get("drop", 0))
			stats["snaps_pgrid"] += float(rv_m2e.get("pgrid", 0) + rv_e2m.get("pgrid", 0))
			stats["_snaps_perr_n"] += float(rv_m2e.get("perr_n", 0) + rv_e2m.get("perr_n", 0))
			stats["_snaps_perr_sum"] += float(rv_m2e.get("perr_sum", 0.0) + rv_e2m.get("perr_sum", 0.0))
			stats["_snaps_perr_max"] = max(
				stats["_snaps_perr_max"],
				float(rv_m2e.get("perr_max", 0.0)),
				float(rv_e2m.get("perr_max", 0.0)),
			)
			seed_inserted, seed_dist, seed_ext_dist = _try_seed_reinsert(
				state,
				model_xyz=model_xyz_det,
				model_valid=model_valid,
				ext_xyz=ext_xyz,
				ext_valid=ext_valid,
				ext_quad_valid=ext_quad_valid,
				cfg=cfg,
				seed_xyz=_seed_xyz,
			)
			stats["snaps_sdist"] = min(stats["snaps_sdist"], float(seed_dist))
			stats["snaps_sext"] = min(stats["snaps_sext"], float(seed_ext_dist))
			if seed_inserted:
				stats["snaps_seed"] += 1.0
			grow_m2e = _grow_direction(
				state.model_to_ext,
				source_xyz=model_xyz_det,
				source_valid=model_valid,
				target_xyz=ext_xyz,
				target_valid=ext_valid,
				normal_xyz=model_normals.detach(),
				normal_from_source=True,
				cfg=cfg,
			)
			grow_e2m = _grow_direction(
				state.ext_to_model,
				source_xyz=ext_xyz,
				source_valid=ext_valid,
				target_xyz=model_xyz_det,
				target_valid=model_valid,
				normal_xyz=model_normals.detach(),
				normal_from_source=False,
				cfg=cfg,
			)
			for k in ("ring", "sup", "tgt", "dist", "grid", "ori", "new"):
				stats[f"snaps_{k}"] += float(grow_m2e.get(k, 0) + grow_e2m.get(k, 0))
			stats["_snaps_gerr_n"] += float(grow_m2e.get("gerr_n", 0) + grow_e2m.get("gerr_n", 0))
			stats["_snaps_gerr_sum"] += float(grow_m2e.get("gerr_sum", 0.0) + grow_e2m.get("gerr_sum", 0.0))
			stats["_snaps_gerr_max"] = max(
				stats["_snaps_gerr_max"],
				float(grow_m2e.get("gerr_max", 0.0)),
				float(grow_e2m.get("gerr_max", 0.0)),
			)

		l_m2e, lm_m2e, mask_m2e, n_m2e = _direction_loss_model_to_ext(
			state.model_to_ext,
			model_xyz=res.xyz_lr,
			model_normals=model_normals,
			ext_xyz=ext_xyz,
			cfg=cfg,
		)
		l_e2m, n_e2m = _direction_loss_ext_to_model(
			state.ext_to_model,
			model_xyz=res.xyz_lr,
			model_normals=model_normals,
			ext_xyz=ext_xyz,
			cfg=cfg,
		)
		stats["snaps_m2e"] += float(n_m2e)
		stats["snaps_e2m"] += float(n_e2m)
		w_m2e = cfg.w_to_ext if n_m2e > 0 else 0.0
		w_e2m = cfg.w_to_model if n_e2m > 0 else 0.0
		w_sum = w_m2e + w_e2m
		if w_sum > 0.0:
			total = total + (w_m2e * l_m2e + w_e2m * l_e2m) / w_sum
			total_weight += 1.0
		lm_accum = lm_accum + lm_m2e
		mask_accum = (mask_accum + mask_m2e).clamp(max=1.0)

	if total_weight > 0.0:
		total = total / total_weight
	stats["snaps_m2e"] = _safe_frac(stats["snaps_m2e"], total_model_possible)
	stats["snaps_e2m"] = _safe_frac(stats["snaps_e2m"], total_ext_possible)
	stats["snaps_seed"] = _safe_frac(stats["snaps_seed"], len(records))
	gerr_n = stats.pop("_snaps_gerr_n")
	gerr_sum = stats.pop("_snaps_gerr_sum")
	gerr_max = stats.pop("_snaps_gerr_max")
	perr_n = stats.pop("_snaps_perr_n")
	perr_sum = stats.pop("_snaps_perr_sum")
	perr_max = stats.pop("_snaps_perr_max")
	stats["snaps_gerr_avg"] = _safe_frac(gerr_sum, gerr_n)
	stats["snaps_gerr_max"] = float(gerr_max)
	stats["snaps_perr_avg"] = _safe_frac(perr_sum, perr_n)
	stats["snaps_perr_max"] = float(perr_max)
	_last_stats = stats
	return total, (lm_accum,), (mask_accum,)
