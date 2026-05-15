from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import numpy as np
import torch
import tifffile


WRAP_ATOL = 1.0e-3
WRAP_RTOL = 1.0e-5


@dataclass(frozen=True)
class InitShellSurface:
	shell_id: str
	path: Path
	xyz_wrapped: torch.Tensor
	unique_w: int


@dataclass(frozen=True)
class ShellClosestPoint:
	shell_id: str
	shell_index: int
	shell_path: Path
	quad_row: int
	quad_col: int
	triangle_id: int
	barycentric: tuple[float, float, float]
	closest_xyz: tuple[float, float, float]
	distance: float
	h: float
	w: float


@dataclass(frozen=True)
class ShellCropInfo:
	full_width: bool
	anchor_h: float
	anchor_w: float
	anchor_arc: float
	circumference: float
	mesh_h: int
	mesh_w: int


def _load_wrapped_tifxyz(path: Path) -> torch.Tensor:
	if not path.is_dir():
		raise ValueError(f"init shell path is not a directory: {path}")
	x_path = path / "x.tif"
	y_path = path / "y.tif"
	z_path = path / "z.tif"
	if not x_path.exists() or not y_path.exists() or not z_path.exists():
		raise ValueError(f"init shell is missing x.tif/y.tif/z.tif: {path}")
	x = tifffile.imread(str(x_path)).astype(np.float32)
	y = tifffile.imread(str(y_path)).astype(np.float32)
	z = tifffile.imread(str(z_path)).astype(np.float32)
	if x.shape != y.shape or x.shape != z.shape:
		raise ValueError(f"init shell tifxyz shape mismatch in {path}: x={x.shape} y={y.shape} z={z.shape}")
	xyz_np = np.stack([x, y, z], axis=-1)
	if xyz_np.ndim != 3 or xyz_np.shape[-1] != 3:
		raise ValueError(f"init shell xyz must have shape (H, W, 3), got {xyz_np.shape} in {path}")
	if xyz_np.shape[0] < 2 or xyz_np.shape[1] < 4:
		raise ValueError(f"init shell must be a wrapped grid with H>=2 and W>=4, got {xyz_np.shape[:2]} in {path}")
	if np.any(~np.isfinite(xyz_np)):
		raise ValueError(f"init shell contains non-finite vertices: {path}")
	valid = np.all(xyz_np != -1.0, axis=-1)
	if not bool(np.all(valid)):
		raise ValueError(f"init shell contains invalid (-1,-1,-1) tifxyz vertices: {path}")
	if not np.allclose(xyz_np[:, 0], xyz_np[:, -1], atol=WRAP_ATOL, rtol=WRAP_RTOL):
		raise ValueError(f"init shell is not explicitly wrapped: first and last columns differ in {path}")
	return torch.from_numpy(xyz_np).to(dtype=torch.float32).contiguous()


def _quad_span(corners: torch.Tensor) -> torch.Tensor:
	p00 = corners[:, 0]
	p10 = corners[:, 1]
	p11 = corners[:, 2]
	p01 = corners[:, 3]
	spans = torch.stack([
		(p00 - p10).norm(dim=-1),
		(p00 - p11).norm(dim=-1),
		(p00 - p01).norm(dim=-1),
		(p10 - p11).norm(dim=-1),
		(p10 - p01).norm(dim=-1),
		(p11 - p01).norm(dim=-1),
	], dim=0)
	return spans.max()


def _closest_points_on_triangles(
	point: torch.Tensor,
	a: torch.Tensor,
	b: torch.Tensor,
	c: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
	"""Return closest points and barycentric coordinates for point vs triangles.

	Barycentric coordinates are ordered as (a, b, c). The implementation is the
	region test from Real-Time Collision Detection, vectorized over triangles.
	"""
	if a.numel() == 0:
		empty = torch.empty(0, 3, device=point.device, dtype=point.dtype)
		return empty, empty
	eps = torch.finfo(a.dtype).eps
	p = point.view(1, 3).to(device=a.device, dtype=a.dtype)
	ab = b - a
	ac = c - a
	ap = p - a
	d1 = (ab * ap).sum(dim=-1)
	d2 = (ac * ap).sum(dim=-1)

	bp = p - b
	d3 = (ab * bp).sum(dim=-1)
	d4 = (ac * bp).sum(dim=-1)

	cp = p - c
	d5 = (ab * cp).sum(dim=-1)
	d6 = (ac * cp).sum(dim=-1)

	n = int(a.shape[0])
	bary = torch.zeros(n, 3, device=a.device, dtype=a.dtype)
	assigned = torch.zeros(n, device=a.device, dtype=torch.bool)

	def assign(mask: torch.Tensor, values: torch.Tensor) -> None:
		nonlocal assigned
		m = mask & ~assigned
		if bool(m.any().detach().cpu()):
			bary[m] = values[m]
			assigned |= m

	values = torch.zeros_like(bary)
	values[:, 0] = 1.0
	assign((d1 <= 0.0) & (d2 <= 0.0), values)

	values = torch.zeros_like(bary)
	values[:, 1] = 1.0
	assign((d3 >= 0.0) & (d4 <= d3), values)

	vc = d1 * d4 - d3 * d2
	v = d1 / (d1 - d3).clamp(min=eps)
	values = torch.zeros_like(bary)
	values[:, 0] = 1.0 - v
	values[:, 1] = v
	assign((vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0), values)

	values = torch.zeros_like(bary)
	values[:, 2] = 1.0
	assign((d6 >= 0.0) & (d5 <= d6), values)

	vb = d5 * d2 - d1 * d6
	w = d2 / (d2 - d6).clamp(min=eps)
	values = torch.zeros_like(bary)
	values[:, 0] = 1.0 - w
	values[:, 2] = w
	assign((vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0), values)

	va = d3 * d6 - d5 * d4
	w = (d4 - d3) / ((d4 - d3) + (d5 - d6)).clamp(min=eps)
	values = torch.zeros_like(bary)
	values[:, 1] = 1.0 - w
	values[:, 2] = w
	assign((va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0), values)

	inside = ~assigned
	denom = (va + vb + vc).clamp(min=eps)
	values = torch.zeros_like(bary)
	values[:, 1] = vb / denom
	values[:, 2] = vc / denom
	values[:, 0] = 1.0 - values[:, 1] - values[:, 2]
	assign(inside, values)

	closest = bary[:, 0:1] * a + bary[:, 1:2] * b + bary[:, 2:3] * c
	return closest, bary


class InitShellIndex:
	def __init__(self, surfaces: list[InitShellSurface]) -> None:
		if not surfaces:
			raise ValueError("init shell directory contains no shell_*.tifxyz directories")
		try:
			from scipy.spatial import cKDTree
		except ImportError as exc:
			raise ValueError("shell-dir-crop init requires scipy.spatial.cKDTree") from exc

		self.surfaces = surfaces
		self._shell_widths = np.asarray([int(s.unique_w) for s in surfaces], dtype=np.int64)
		self._shell_heights = np.asarray([int(s.xyz_wrapped.shape[0]) for s in surfaces], dtype=np.int64)
		vertex_xyz: list[np.ndarray] = []
		vertex_shell: list[np.ndarray] = []
		vertex_row: list[np.ndarray] = []
		vertex_col: list[np.ndarray] = []
		quad_corners: list[torch.Tensor] = []
		quad_shell: list[np.ndarray] = []
		quad_row: list[np.ndarray] = []
		quad_col: list[np.ndarray] = []
		self._quad_starts: list[int] = []
		max_span = 0.0

		for shell_i, surf in enumerate(surfaces):
			xyz = surf.xyz_wrapped
			h, w_wrapped, _ = xyz.shape
			w_unique = int(surf.unique_w)
			rows, cols = np.meshgrid(np.arange(h), np.arange(w_wrapped), indexing="ij")
			vertex_xyz.append(xyz.reshape(-1, 3).numpy())
			vertex_shell.append(np.full(h * w_wrapped, shell_i, dtype=np.int64))
			vertex_row.append(rows.reshape(-1).astype(np.int64))
			vertex_col.append(cols.reshape(-1).astype(np.int64))

			p00 = xyz[:-1, :w_unique]
			p10 = xyz[1:, :w_unique]
			p11 = xyz[1:, 1:w_unique + 1]
			p01 = xyz[:-1, 1:w_unique + 1]
			corners = torch.stack([p00, p10, p11, p01], dim=2).reshape(-1, 4, 3).contiguous()
			self._quad_starts.append(sum(int(q.shape[0]) for q in quad_corners))
			quad_corners.append(corners)
			q_count = int(corners.shape[0])
			q_rows, q_cols = np.meshgrid(np.arange(h - 1), np.arange(w_unique), indexing="ij")
			quad_shell.append(np.full(q_count, shell_i, dtype=np.int64))
			quad_row.append(q_rows.reshape(-1).astype(np.int64))
			quad_col.append(q_cols.reshape(-1).astype(np.int64))
			max_span = max(max_span, float(_quad_span(corners).detach().cpu()))

		self.vertex_xyz = np.concatenate(vertex_xyz, axis=0)
		self.vertex_shell = np.concatenate(vertex_shell, axis=0)
		self.vertex_row = np.concatenate(vertex_row, axis=0)
		self.vertex_col = np.concatenate(vertex_col, axis=0)
		self.quad_corners = torch.cat(quad_corners, dim=0).contiguous()
		self.quad_shell = np.concatenate(quad_shell, axis=0)
		self.quad_row = np.concatenate(quad_row, axis=0)
		self.quad_col = np.concatenate(quad_col, axis=0)
		self._quad_starts_np = np.asarray(self._quad_starts, dtype=np.int64)
		self.max_quad_span = float(max_span)
		if self.vertex_xyz.shape[0] == 0 or self.quad_corners.shape[0] == 0:
			raise ValueError("init shell directory produced no lookup vertices/quads")
		self.tree = cKDTree(self.vertex_xyz)

	@classmethod
	def from_directory(cls, init_shell_dir: str | Path) -> "InitShellIndex":
		root = Path(init_shell_dir)
		if not root.is_dir():
			raise ValueError(f"init_shell_dir does not exist or is not a directory: {root}")
		paths = [p for p in sorted(root.glob("shell_*.tifxyz")) if p.is_dir()]
		if not paths:
			raise ValueError(f"init_shell_dir contains no shell_*.tifxyz directories: {root}")
		surfaces = []
		for path in paths:
			xyz = _load_wrapped_tifxyz(path)
			surfaces.append(InitShellSurface(
				shell_id=path.name,
				path=path,
				xyz_wrapped=xyz,
				unique_w=int(xyz.shape[1]) - 1,
			))
		return cls(surfaces)

	def _candidate_quads_from_vertices(self, vertex_indices: list[int]) -> np.ndarray:
		if not vertex_indices:
			return np.empty(0, dtype=np.int64)
		idx = np.asarray(vertex_indices, dtype=np.int64)
		shell_i = self.vertex_shell[idx]
		row = self.vertex_row[idx]
		widths = self._shell_widths[shell_i]
		heights = self._shell_heights[shell_i]
		col_unique = np.mod(self.vertex_col[idx], widths)
		quad_ids: list[np.ndarray] = []
		for row_off in (-1, 0):
			q_row = row + row_off
			for col_off in (-1, 0):
				q_col = np.mod(col_unique + col_off, widths)
				mask = (q_row >= 0) & (q_row < heights - 1)
				if not bool(mask.any()):
					continue
				starts = self._quad_starts_np[shell_i[mask]]
				quad_ids.append(starts + q_row[mask] * widths[mask] + q_col[mask])
		if not quad_ids:
			return np.empty(0, dtype=np.int64)
		return np.unique(np.concatenate(quad_ids, axis=0))

	def closest_point(
		self,
		seed: tuple[float, float, float],
		*,
		device: torch.device | str = "cpu",
		batch_quads: int = 65536,
	) -> ShellClosestPoint:
		seed_np = np.asarray(seed, dtype=np.float64)
		if seed_np.shape != (3,) or not np.all(np.isfinite(seed_np)):
			raise ValueError(f"seed must be three finite coordinates, got {seed}")
		d_vertex, _idx = self.tree.query(seed_np, k=1)
		radius = float(d_vertex) + float(self.max_quad_span)
		vertex_indices = self.tree.query_ball_point(seed_np, r=radius)
		candidate_ids = self._candidate_quads_from_vertices(vertex_indices)
		if candidate_ids.size == 0:
			raise ValueError("shell-dir-crop lookup found no candidate quads")

		dev = torch.device(device)
		seed_t = torch.tensor(seed_np, device=dev, dtype=torch.float32)
		best_dist_sq = float("inf")
		best_quad_global = -1
		best_triangle = -1
		best_bary: tuple[float, float, float] | None = None
		best_xyz: tuple[float, float, float] | None = None

		candidate_ids = np.sort(candidate_ids)
		for start in range(0, int(candidate_ids.size), int(batch_quads)):
			batch_np = candidate_ids[start:start + int(batch_quads)]
			batch_idx = torch.from_numpy(batch_np).to(dtype=torch.long)
			quad = self.quad_corners.index_select(0, batch_idx).to(device=dev, dtype=torch.float32)
			for tri_id, (a, b, c) in enumerate(((quad[:, 0], quad[:, 1], quad[:, 2]), (quad[:, 0], quad[:, 2], quad[:, 3]))):
				closest, bary = _closest_points_on_triangles(seed_t, a, b, c)
				dist_sq = ((closest - seed_t.view(1, 3)) ** 2).sum(dim=-1)
				local_i = int(torch.argmin(dist_sq).detach().cpu())
				dist_sq_f = float(dist_sq[local_i].detach().cpu())
				if dist_sq_f < best_dist_sq:
					best_dist_sq = dist_sq_f
					best_quad_global = int(batch_np[local_i])
					best_triangle = int(tri_id)
					best_bary = tuple(float(v) for v in bary[local_i].detach().cpu().tolist())
					best_xyz = tuple(float(v) for v in closest[local_i].detach().cpu().tolist())

		if best_quad_global < 0 or best_bary is None or best_xyz is None:
			raise ValueError("shell-dir-crop lookup failed to select a closest point")
		shell_i = int(self.quad_shell[best_quad_global])
		row = int(self.quad_row[best_quad_global])
		col = int(self.quad_col[best_quad_global])
		a_bary, b_bary, c_bary = best_bary
		if best_triangle == 0:
			h = float(row) + b_bary + c_bary
			w = float(col) + c_bary
		else:
			h = float(row) + b_bary
			w = float(col) + b_bary + c_bary
		w = math.fmod(w, float(self.surfaces[shell_i].unique_w))
		if w < 0.0:
			w += float(self.surfaces[shell_i].unique_w)
		return ShellClosestPoint(
			shell_id=self.surfaces[shell_i].shell_id,
			shell_index=shell_i,
			shell_path=self.surfaces[shell_i].path,
			quad_row=row,
			quad_col=col,
			triangle_id=best_triangle,
			barycentric=best_bary,
			closest_xyz=best_xyz,
			distance=math.sqrt(max(0.0, best_dist_sq)),
			h=h,
			w=w,
		)


def _row_cumulative_lengths(row: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	edges = (torch.roll(row, shifts=-1, dims=0) - row).norm(dim=-1)
	cumulative = torch.cat([
		torch.zeros(1, device=row.device, dtype=row.dtype),
		edges.cumsum(dim=0),
	], dim=0)
	circumference = cumulative[-1].clamp(min=1.0e-8)
	return cumulative, edges, circumference


def _sample_periodic_row_by_arclength(row: torch.Tensor, distances: torch.Tensor) -> torch.Tensor:
	cumulative, edges, circumference = _row_cumulative_lengths(row)
	s = torch.remainder(distances.to(device=row.device, dtype=row.dtype), circumference)
	idx_hi = torch.searchsorted(cumulative.contiguous(), s.contiguous(), right=True)
	idx_hi = idx_hi.clamp(min=1, max=int(row.shape[0]))
	idx0 = idx_hi - 1
	idx1 = torch.remainder(idx0 + 1, int(row.shape[0]))
	seg_len = edges.index_select(0, idx0.reshape(-1)).reshape_as(s).clamp(min=1.0e-8)
	s0 = cumulative.index_select(0, idx0.reshape(-1)).reshape_as(s)
	frac = ((s - s0) / seg_len).clamp(min=0.0, max=1.0).unsqueeze(-1)
	p0 = row.index_select(0, idx0.reshape(-1)).reshape(*s.shape, int(row.shape[-1]))
	p1 = row.index_select(0, idx1.reshape(-1)).reshape(*s.shape, int(row.shape[-1]))
	return p0 + frac * (p1 - p0)


def _row_arclength_at_param(row: torch.Tensor, w: float) -> torch.Tensor:
	w_unique = int(row.shape[0])
	cumulative, edges, _circumference = _row_cumulative_lengths(row)
	w_t = torch.remainder(torch.tensor(float(w), device=row.device, dtype=row.dtype), float(w_unique))
	idx0 = int(torch.floor(w_t).detach().cpu())
	frac = w_t - float(idx0)
	return cumulative[idx0] + frac * edges[idx0]


def _sample_periodic_row_by_param(row: torch.Tensor, w: float) -> torch.Tensor:
	w_unique = int(row.shape[0])
	w_t = torch.remainder(torch.tensor(float(w), device=row.device, dtype=row.dtype), float(w_unique))
	idx0 = int(torch.floor(w_t).detach().cpu())
	idx1 = (idx0 + 1) % w_unique
	frac = w_t - float(idx0)
	return row[idx0] + frac * (row[idx1] - row[idx0])


def _interpolate_shell_row(shell: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
	h_clamped = h.clamp(min=0.0, max=float(int(shell.shape[0]) - 1))
	idx0 = torch.floor(h_clamped).to(dtype=torch.long).clamp(max=int(shell.shape[0]) - 2)
	idx1 = idx0 + 1
	frac = (h_clamped - idx0.to(dtype=h.dtype)).view(-1, 1, 1)
	row0 = shell.index_select(0, idx0.reshape(-1))
	row1 = shell.index_select(0, idx1.reshape(-1))
	rows = row0 + frac * (row1 - row0)
	return rows.reshape(*h.shape, int(shell.shape[1]), 3)


def _height_targets_from_anchor(
	shell: torch.Tensor,
	*,
	anchor_h: float,
	anchor_w: float,
	height_offsets: torch.Tensor,
) -> torch.Tensor:
	curve = torch.stack([
		_sample_periodic_row_by_param(shell[row_i], anchor_w)
		for row_i in range(int(shell.shape[0]))
	], dim=0)
	seg = (curve[1:] - curve[:-1]).norm(dim=-1)
	cumulative = torch.cat([seg.new_zeros(1), seg.cumsum(dim=0)], dim=0)
	total = float(cumulative[-1].detach().cpu())
	if total <= 1.0e-8:
		raise ValueError("init shell height curve has zero length at crop anchor")
	h_anchor = torch.tensor(float(anchor_h), device=shell.device, dtype=shell.dtype).clamp(
		min=0.0,
		max=float(int(shell.shape[0]) - 1),
	)
	h0 = int(torch.floor(h_anchor).detach().cpu())
	h0 = min(max(0, h0), int(shell.shape[0]) - 2)
	h_frac = h_anchor - float(h0)
	s_anchor = cumulative[h0] + h_frac * (cumulative[h0 + 1] - cumulative[h0])
	target_s = (s_anchor + height_offsets.to(device=shell.device, dtype=shell.dtype)).clamp(
		min=0.0,
		max=cumulative[-1],
	)
	idx_hi = torch.searchsorted(cumulative.contiguous(), target_s.contiguous(), right=False)
	idx_hi = idx_hi.clamp(min=1, max=int(shell.shape[0]) - 1)
	idx0 = idx_hi - 1
	s0 = cumulative.index_select(0, idx0.reshape(-1)).reshape_as(target_s)
	s1 = cumulative.index_select(0, idx_hi.reshape(-1)).reshape_as(target_s)
	frac = ((target_s - s0) / (s1 - s0).clamp(min=1.0e-8)).clamp(min=0.0, max=1.0)
	return idx0.to(dtype=shell.dtype) + frac


def crop_shell_surface(
	surface: InitShellSurface,
	closest: ShellClosestPoint,
	*,
	seed: tuple[float, float, float],
	model_w: float | None,
	model_h: float,
	mesh_step: float,
	device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, ShellCropInfo]:
	if closest.shell_id != surface.shell_id:
		raise ValueError(f"closest point shell_id {closest.shell_id!r} does not match surface {surface.shell_id!r}")
	seed_np = np.asarray(seed, dtype=np.float64)
	if seed_np.shape != (3,) or not np.all(np.isfinite(seed_np)):
		raise ValueError(f"seed must be three finite coordinates, got {seed}")
	step = float(mesh_step)
	if step <= 0.0:
		raise ValueError(f"mesh_step must be > 0 for shell-dir-crop, got {mesh_step}")
	if float(model_h) <= 0.0:
		raise ValueError(f"model_h must be > 0 for shell-dir-crop, got {model_h}")
	dev = torch.device(device)
	shell = surface.xyz_wrapped[:, :surface.unique_w].to(device=dev, dtype=torch.float32).contiguous()
	h_count = max(2, int(math.ceil(float(model_h) / step)) + 1)
	height_offsets = torch.linspace(
		-0.5 * float(model_h),
		0.5 * float(model_h),
		h_count,
		device=dev,
		dtype=torch.float32,
	)
	target_h = _height_targets_from_anchor(
		shell,
		anchor_h=float(closest.h),
		anchor_w=float(closest.w),
		height_offsets=height_offsets,
	)
	anchor_row = _interpolate_shell_row(
		shell,
		torch.tensor([float(closest.h)], device=dev, dtype=torch.float32),
	)[0]
	anchor_arc = _row_arclength_at_param(anchor_row, float(closest.w))
	_cumulative, _edges, circ_anchor = _row_cumulative_lengths(anchor_row)
	circ_anchor_f = float(circ_anchor.detach().cpu())
	model_w_f = 0.0 if model_w is None else float(model_w)
	full_width = model_w_f <= 0.0 or model_w_f >= circ_anchor_f - step
	if full_width:
		w_count = max(3, int(math.ceil(circ_anchor_f / step)))
		base_offsets = None
	else:
		w_count = max(2, int(math.ceil(model_w_f / step)) + 1)
		base_offsets = torch.linspace(
			-0.5 * model_w_f,
			0.5 * model_w_f,
			w_count,
			device=dev,
			dtype=torch.float32,
		)
	anchor_phase = anchor_arc / circ_anchor
	rows = _interpolate_shell_row(shell, target_h)
	out_rows = []
	for row in rows:
		_row_cum, _row_edges, row_circ = _row_cumulative_lengths(row)
		center_arc = anchor_phase.to(device=dev, dtype=torch.float32) * row_circ
		if full_width:
			row_step = row_circ / float(w_count)
			width_offsets = (torch.arange(w_count, device=dev, dtype=torch.float32) - float(w_count - 1) * 0.5) * row_step
		else:
			width_offsets = base_offsets
		out_rows.append(_sample_periodic_row_by_arclength(row, center_arc + width_offsets))
	crop = torch.stack(out_rows, dim=0).contiguous()
	h_mid = float(int(crop.shape[0]) - 1) * 0.5
	w_mid = float(int(crop.shape[1]) - 1) * 0.5
	h0 = int(math.floor(h_mid))
	w0 = int(math.floor(w_mid))
	h1 = min(h0 + 1, int(crop.shape[0]) - 1)
	w1 = min(w0 + 1, int(crop.shape[1]) - 1)
	fh = torch.tensor(h_mid - float(h0), device=dev, dtype=crop.dtype)
	fw = torch.tensor(w_mid - float(w0), device=dev, dtype=crop.dtype)
	center = (
		(1.0 - fh) * (1.0 - fw) * crop[h0, w0]
		+ fh * (1.0 - fw) * crop[h1, w0]
		+ (1.0 - fh) * fw * crop[h0, w1]
		+ fh * fw * crop[h1, w1]
	)
	anchor_xyz = torch.tensor(closest.closest_xyz, device=dev, dtype=crop.dtype)
	crop = crop + (anchor_xyz - center).view(1, 1, 3)
	valid = torch.ones(crop.shape[:2], device=dev, dtype=torch.bool)
	info = ShellCropInfo(
		full_width=bool(full_width),
		anchor_h=float(closest.h),
		anchor_w=float(closest.w),
		anchor_arc=float(anchor_arc.detach().cpu()),
		circumference=circ_anchor_f,
		mesh_h=int(crop.shape[0]),
		mesh_w=int(crop.shape[1]),
	)
	return crop, valid, info
