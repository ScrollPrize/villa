from __future__ import annotations

import copy
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import tifffile


@dataclass(frozen=True)
class ApprovalInpaintResult:
	seed: tuple[float, float, float]
	model_w: int
	model_h: int
	corr_points: dict
	point_count: int
	component_size: int
	skeleton_size: int
	index_bounds: tuple[int, int, int, int]  # row_min, row_max, col_min, col_max
	source_mesh_step: float


def _as_2d_nonzero(arr: np.ndarray, *, name: str) -> np.ndarray:
	a = np.asarray(arr)
	if a.ndim == 2:
		return a != 0
	if a.ndim == 3 and a.shape[-1] in (3, 4):
		return np.any(a != 0, axis=-1)
	if a.ndim == 3:
		return np.any(a != 0, axis=0)
	raise ValueError(f"{name} must be a 2D or RGB/RGBA image, got shape {a.shape}")


def load_approval_mask(path: str | Path, *, expected_shape: tuple[int, int] | None = None) -> np.ndarray:
	"""Load approval.tif as a binary mask. Any nonzero sample is approved."""
	p = Path(path)
	if not p.exists():
		raise ValueError(f"missing approval mask: {p}")
	mask = _as_2d_nonzero(tifffile.imread(str(p)), name=str(p))
	if expected_shape is not None and tuple(mask.shape) != tuple(expected_shape):
		raise ValueError(f"approval mask shape mismatch: got {mask.shape}, expected {expected_shape}")
	return mask.astype(bool, copy=False)


def _load_tifxyz_arrays(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
	p = Path(path)
	if not p.is_dir():
		raise ValueError(f"approval inpaint tifxyz path is not a directory: {p}")
	required = ["x.tif", "y.tif", "z.tif", "meta.json", "approval.tif"]
	missing = [name for name in required if not (p / name).exists()]
	if missing:
		raise ValueError(f"approval inpaint tifxyz is missing required file(s): {', '.join(missing)}")

	x = tifffile.imread(str(p / "x.tif")).astype(np.float32)
	y = tifffile.imread(str(p / "y.tif")).astype(np.float32)
	z = tifffile.imread(str(p / "z.tif")).astype(np.float32)
	if x.shape != y.shape or x.shape != z.shape:
		raise ValueError(f"tifxyz shape mismatch: x={x.shape} y={y.shape} z={z.shape}")
	if x.ndim != 2:
		raise ValueError(f"tifxyz coordinate images must be 2D, got {x.shape}")
	xyz = np.stack([x, y, z], axis=-1)
	valid = np.isfinite(xyz).all(axis=-1) & np.all(xyz != -1.0, axis=-1)

	d_path = p / "d.tif"
	if d_path.exists():
		d = tifffile.imread(str(d_path)).astype(np.float32)
		if d.ndim == 3:
			if d.shape[-1] in (3, 4):
				d = d[..., 0]
			else:
				d = d[0]
		if tuple(d.shape) != tuple(x.shape):
			raise ValueError(f"d.tif shape mismatch: got {d.shape}, expected {x.shape}")
	else:
		print("[approval_inpaint] d.tif not found; generated corr point wind_a defaults to 0.0", flush=True)
		d = np.zeros_like(x, dtype=np.float32)

	meta: dict = {}
	try:
		import json
		meta = json.loads((p / "meta.json").read_text(encoding="utf-8"))
	except Exception:
		meta = {}

	return xyz, valid, d, meta


def _source_mesh_step(meta: dict, fallback: float) -> float:
	scale = meta.get("scale") if isinstance(meta, dict) else None
	if isinstance(scale, list) and scale:
		try:
			s0 = float(scale[0])
			if math.isfinite(s0) and s0 > 0.0:
				return max(1.0, 1.0 / s0)
		except (TypeError, ValueError):
			pass
	return max(1.0, float(fallback))


def _neighbors4(r: int, c: int, h: int, w: int) -> Iterable[tuple[int, int]]:
	if r > 0:
		yield r - 1, c
	if r + 1 < h:
		yield r + 1, c
	if c > 0:
		yield r, c - 1
	if c + 1 < w:
		yield r, c + 1


def _neighbors8(r: int, c: int, h: int, w: int) -> Iterable[tuple[int, int]]:
	for dr in (-1, 0, 1):
		for dc in (-1, 0, 1):
			if dr == 0 and dc == 0:
				continue
			rr = r + dr
			cc = c + dc
			if 0 <= rr < h and 0 <= cc < w:
				yield rr, cc


def _connected_components(mask: np.ndarray, *, connectivity: int = 4) -> tuple[np.ndarray, int, list[int]]:
	if mask.ndim != 2:
		raise ValueError(f"connected component mask must be 2D, got {mask.shape}")
	h, w = mask.shape
	labels = np.zeros((h, w), dtype=np.int32)
	sizes: list[int] = [0]
	next_label = 0
	neighbors = _neighbors8 if connectivity == 8 else _neighbors4
	for r in range(h):
		for c in range(w):
			if not mask[r, c] or labels[r, c] != 0:
				continue
			next_label += 1
			q: deque[tuple[int, int]] = deque([(r, c)])
			labels[r, c] = next_label
			size = 0
			while q:
				rr, cc = q.popleft()
				size += 1
				for nr, nc in neighbors(rr, cc, h, w):
					if mask[nr, nc] and labels[nr, nc] == 0:
						labels[nr, nc] = next_label
						q.append((nr, nc))
			sizes.append(size)
	return labels, next_label, sizes


def find_unapproved_components(approval: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, int, list[int]]:
	"""Return connected valid, unapproved regions.

	The complement is restricted to valid tifxyz vertices; invalid vertices and
	image edges act as borders for approval-inpaint selection.
	"""
	if approval.shape != valid.shape:
		raise ValueError(f"approval/valid shape mismatch: {approval.shape} vs {valid.shape}")
	return _connected_components((~approval.astype(bool)) & valid.astype(bool), connectivity=4)


def nearest_valid_index(xyz: np.ndarray, valid: np.ndarray, seed: tuple[float, float, float]) -> tuple[int, int]:
	if xyz.ndim != 3 or xyz.shape[-1] != 3:
		raise ValueError(f"xyz must have shape (H, W, 3), got {xyz.shape}")
	if valid.shape != xyz.shape[:2]:
		raise ValueError(f"valid shape mismatch: {valid.shape} vs {xyz.shape[:2]}")
	seed_np = np.asarray(seed, dtype=np.float64)
	if seed_np.shape != (3,) or not np.isfinite(seed_np).all():
		raise ValueError(f"approval inpaint seed must contain three finite values, got {seed}")

	flat_xyz = xyz.reshape(-1, 3)
	flat_valid = valid.reshape(-1)
	best_i = -1
	best_d2 = float("inf")
	chunk = 1_000_000
	for start in range(0, flat_xyz.shape[0], chunk):
		end = min(start + chunk, flat_xyz.shape[0])
		mask = flat_valid[start:end]
		if not bool(mask.any()):
			continue
		coords = flat_xyz[start:end][mask].astype(np.float64, copy=False)
		d2 = np.sum((coords - seed_np.reshape(1, 3)) ** 2, axis=1)
		local = int(np.argmin(d2))
		if float(d2[local]) < best_d2:
			valid_indices = np.flatnonzero(mask)
			best_i = start + int(valid_indices[local])
			best_d2 = float(d2[local])
	if best_i < 0:
		raise ValueError("approval inpaint source tifxyz contains no valid vertices")
	h, w = valid.shape
	return divmod(best_i, w)


def select_component_for_seed(
	labels: np.ndarray,
	seed_index: tuple[int, int],
) -> tuple[int, np.ndarray]:
	r, c = seed_index
	if r < 0 or r >= labels.shape[0] or c < 0 or c >= labels.shape[1]:
		raise ValueError(f"approval inpaint projected seed index out of bounds: {seed_index}")
	label = int(labels[r, c])
	if label <= 0:
		raise ValueError(
			"approval inpaint seed does not project into an enclosed unapproved region "
			f"(grid row={r} col={c})"
		)
	return label, labels == label


def _dilate4(mask: np.ndarray) -> np.ndarray:
	out = mask.copy()
	out[1:] |= mask[:-1]
	out[:-1] |= mask[1:]
	out[:, 1:] |= mask[:, :-1]
	out[:, :-1] |= mask[:, 1:]
	return out


def enclosing_approval_mask(component: np.ndarray, approval: np.ndarray, valid: np.ndarray) -> np.ndarray:
	if component.shape != approval.shape or approval.shape != valid.shape:
		raise ValueError("component, approval, and valid masks must have identical shapes")
	adjacent = _dilate4(component) & ~component & approval.astype(bool) & valid.astype(bool)
	if not bool(adjacent.any()):
		raise ValueError("approval inpaint region has no adjacent approved boundary")
	return adjacent


def _zhang_suen_skeleton(mask: np.ndarray) -> np.ndarray:
	img = mask.astype(np.uint8, copy=True)
	if img.ndim != 2:
		raise ValueError(f"skeleton mask must be 2D, got {img.shape}")
	if min(img.shape) < 3:
		return img.astype(bool)
	changed = True
	while changed:
		changed = False
		for step in (0, 1):
			to_delete: list[tuple[int, int]] = []
			for r in range(1, img.shape[0] - 1):
				for c in range(1, img.shape[1] - 1):
					if img[r, c] == 0:
						continue
					p2 = img[r - 1, c]
					p3 = img[r - 1, c + 1]
					p4 = img[r, c + 1]
					p5 = img[r + 1, c + 1]
					p6 = img[r + 1, c]
					p7 = img[r + 1, c - 1]
					p8 = img[r, c - 1]
					p9 = img[r - 1, c - 1]
					ns = int(p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9)
					if ns < 2 or ns > 6:
						continue
					seq = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
					transitions = sum(1 for i in range(8) if seq[i] == 0 and seq[i + 1] == 1)
					if transitions != 1:
						continue
					if step == 0:
						if p2 * p4 * p6 != 0 or p4 * p6 * p8 != 0:
							continue
					else:
						if p2 * p4 * p8 != 0 or p2 * p6 * p8 != 0:
							continue
					to_delete.append((r, c))
			if to_delete:
				changed = True
				for r, c in to_delete:
					img[r, c] = 0
	return img.astype(bool)


def skeletonize_boundary(mask: np.ndarray) -> np.ndarray:
	try:
		from skimage.morphology import skeletonize  # type: ignore
		return np.asarray(skeletonize(mask.astype(bool)), dtype=bool)
	except Exception:
		return _zhang_suen_skeleton(mask)


def _order_component(coords: list[tuple[int, int]], shape: tuple[int, int]) -> list[tuple[int, int]]:
	coord_set = set(coords)
	degree: dict[tuple[int, int], int] = {}
	for p in coords:
		degree[p] = sum((n in coord_set) for n in _neighbors8(p[0], p[1], shape[0], shape[1]))
	unvisited = set(coords)
	order: list[tuple[int, int]] = []
	while unvisited:
		endpoints = sorted(p for p in unvisited if degree.get(p, 0) <= 1)
		start = endpoints[0] if endpoints else min(unvisited)
		stack = [start]
		while stack:
			p = stack.pop()
			if p not in unvisited:
				continue
			unvisited.remove(p)
			order.append(p)
			neigh = [n for n in _neighbors8(p[0], p[1], shape[0], shape[1]) if n in unvisited]
			neigh.sort(key=lambda q: (degree.get(q, 0), q[0], q[1]), reverse=True)
			stack.extend(neigh)
	return order


def ordered_skeleton_points(skeleton: np.ndarray) -> list[tuple[int, int]]:
	labels, n, _sizes = _connected_components(skeleton.astype(bool), connectivity=8)
	ordered: list[tuple[int, int]] = []
	for label in range(1, n + 1):
		coords_np = np.argwhere(labels == label)
		if coords_np.size == 0:
			continue
		coords = [(int(r), int(c)) for r, c in coords_np]
		ordered.extend(_order_component(coords, skeleton.shape))
	return ordered


def sample_skeleton_points(
	skeleton: np.ndarray,
	*,
	spacing_px: float,
) -> list[tuple[int, int]]:
	ordered = ordered_skeleton_points(skeleton)
	if not ordered:
		return []
	spacing = max(1.0, float(spacing_px))
	selected: list[tuple[int, int]] = []
	last: tuple[int, int] | None = None
	for p in ordered:
		if last is None:
			selected.append(p)
			last = p
			continue
		dist = math.hypot(float(p[0] - last[0]), float(p[1] - last[1]))
		if dist + 1.0e-6 >= spacing:
			selected.append(p)
			last = p
	if not selected:
		selected.append(ordered[0])
	return selected


def _sample_xyz_at_index_center(
	xyz: np.ndarray,
	valid: np.ndarray,
	row: float,
	col: float,
) -> tuple[float, float, float]:
	h, w = valid.shape
	r0 = int(math.floor(row))
	c0 = int(math.floor(col))
	r0 = max(0, min(r0, h - 1))
	c0 = max(0, min(c0, w - 1))
	r1 = min(r0 + 1, h - 1)
	c1 = min(c0 + 1, w - 1)
	fr = max(0.0, min(float(row) - r0, 1.0))
	fc = max(0.0, min(float(col) - c0, 1.0))
	corners = [(r0, c0), (r1, c0), (r0, c1), (r1, c1)]
	if all(bool(valid[r, c]) for r, c in corners):
		p00 = xyz[r0, c0].astype(np.float64)
		p10 = xyz[r1, c0].astype(np.float64)
		p01 = xyz[r0, c1].astype(np.float64)
		p11 = xyz[r1, c1].astype(np.float64)
		p = (
			(1.0 - fr) * (1.0 - fc) * p00
			+ fr * (1.0 - fc) * p10
			+ (1.0 - fr) * fc * p01
			+ fr * fc * p11
		)
		return float(p[0]), float(p[1]), float(p[2])

	valid_rc = np.argwhere(valid)
	if valid_rc.size == 0:
		raise ValueError("approval inpaint source tifxyz contains no valid vertices")
	d2 = (valid_rc[:, 0].astype(np.float64) - float(row)) ** 2 + (
		valid_rc[:, 1].astype(np.float64) - float(col)
	) ** 2
	rr, cc = (int(v) for v in valid_rc[int(np.argmin(d2))])
	return float(xyz[rr, cc, 0]), float(xyz[rr, cc, 1]), float(xyz[rr, cc, 2])


def _next_collection_id(corr_points: dict) -> str:
	cols = corr_points.get("collections", {})
	max_id = -1
	if isinstance(cols, dict):
		for key in cols.keys():
			try:
				max_id = max(max_id, int(key))
			except (TypeError, ValueError):
				continue
	return str(max_id + 1)


def _merge_generated_corr_points(
	existing: dict | None,
	coords: list[tuple[int, int]],
	xyz: np.ndarray,
	d: np.ndarray,
) -> dict:
	out = copy.deepcopy(existing) if isinstance(existing, dict) else {}
	cols = out.setdefault("collections", {})
	if not isinstance(cols, dict):
		cols = {}
		out["collections"] = cols
	cid = _next_collection_id(out)
	points = {}
	for pid, (r, c) in enumerate(coords):
		points[str(pid)] = {
			"p": [
				float(xyz[r, c, 0]),
				float(xyz[r, c, 1]),
				float(xyz[r, c, 2]),
			],
			"wind_a": float(d[r, c]),
		}
	cols[cid] = {
		"name": "approval_inpaint",
		"metadata": {"winding_is_absolute": True},
		"points": points,
	}
	return out


def build_approval_inpaint(
	*,
	tifxyz_path: str | Path,
	seed: tuple[float, float, float],
	mesh_step: float,
	corr_spacing: float | None = None,
	padding_frac: float | None = 0.25,
	existing_model_w: int | float | None = None,
	existing_model_h: int | float | None = None,
	existing_corr_points: dict | None = None,
) -> ApprovalInpaintResult:
	xyz, valid, d, meta = _load_tifxyz_arrays(tifxyz_path)
	approval = load_approval_mask(Path(tifxyz_path) / "approval.tif", expected_shape=valid.shape)
	d_valid = np.isfinite(d) & (d >= 0.0)
	valid = valid & d_valid
	labels, _n_components, sizes = find_unapproved_components(approval, valid)
	seed_index = nearest_valid_index(xyz, valid, seed)
	component_label, component = select_component_for_seed(labels, seed_index)
	boundary_source = enclosing_approval_mask(component, approval, valid)
	skeleton = skeletonize_boundary(boundary_source)
	if not bool(skeleton.any()):
		skeleton = boundary_source
	source_step = _source_mesh_step(meta, mesh_step)
	spacing = float(corr_spacing) if corr_spacing is not None else float(mesh_step)
	if not math.isfinite(spacing) or spacing <= 0.0:
		raise ValueError(f"approval inpaint corr spacing must be > 0, got {corr_spacing}")
	spacing_px = max(1.0, spacing / source_step)
	candidate_coords = sample_skeleton_points(skeleton, spacing_px=spacing_px)
	coords = [
		(r, c)
		for r, c in candidate_coords
		if bool(valid[r, c]) and bool(np.isfinite(xyz[r, c]).all()) and bool(np.isfinite(d[r, c]))
	]
	if not coords:
		raise ValueError("approval inpaint generated no valid skeleton correction points")

	rows = np.asarray([p[0] for p in coords], dtype=np.float64)
	cols = np.asarray([p[1] for p in coords], dtype=np.float64)
	rmin = int(rows.min())
	rmax = int(rows.max())
	cmin = int(cols.min())
	cmax = int(cols.max())
	center_r = 0.5 * (float(rmin) + float(rmax))
	center_c = 0.5 * (float(cmin) + float(cmax))
	new_seed = _sample_xyz_at_index_center(xyz, valid, center_r, center_c)

	pad = 0.25 if padding_frac is None else float(padding_frac)
	if not math.isfinite(pad) or pad < 0.0:
		raise ValueError(f"approval inpaint padding fraction must be >= 0, got {padding_frac}")
	pad_mul = 1.0 + 2.0 * pad
	width = max(float(mesh_step), float(cmax - cmin) * source_step) * pad_mul
	height = max(float(mesh_step), float(rmax - rmin) * source_step) * pad_mul
	if existing_model_w is not None:
		width = max(width, float(existing_model_w))
	if existing_model_h is not None:
		height = max(height, float(existing_model_h))
	model_w = max(1, int(math.ceil(width)))
	model_h = max(1, int(math.ceil(height)))
	corr_points = _merge_generated_corr_points(existing_corr_points, coords, xyz, d)
	return ApprovalInpaintResult(
		seed=new_seed,
		model_w=model_w,
		model_h=model_h,
		corr_points=corr_points,
		point_count=len(coords),
		component_size=int(sizes[component_label]) if component_label < len(sizes) else int(component.sum()),
		skeleton_size=int(skeleton.sum()),
		index_bounds=(rmin, rmax, cmin, cmax),
		source_mesh_step=float(source_step),
	)
