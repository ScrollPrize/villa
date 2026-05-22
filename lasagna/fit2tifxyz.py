from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile
import torch

import cli_json
import model


@dataclass(frozen=True)
class ExportConfig:
	input: str
	output: str
	prefix: str = "winding_"
	device: str = "cpu"
	single_segment: bool = False
	copy_model: bool = False
	output_name: str | None = None
	voxel_size_um: float | None = None


def _valid_xyz_mask(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
	return (
		np.isfinite(x)
		& np.isfinite(y)
		& np.isfinite(z)
		& ~((x == -1.0) & (y == -1.0) & (z == -1.0))
	)


def _build_parser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(description="Export 3D fit model as tifxyz surfaces (one per winding/depth)")
	cli_json.add_args(p)
	g = p.add_argument_group("io")
	g.add_argument("--input", required=True, help="Model checkpoint (.pt)")
	g.add_argument("--output", required=True, help="Output directory")
	g.add_argument("--prefix", default="winding_", help="Output tifxyz directory prefix")
	g.add_argument("--single-segment", action="store_true", default=False,
		help="Export all windings into a single tifxyz")
	g.add_argument("--copy-model", action="store_true", default=False,
		help="Copy model checkpoint instead of symlink")
	g.add_argument("--output-name", default=None, help="Override tifxyz directory name")
	g.add_argument("--voxel-size-um", type=float, default=None,
		help="Voxel size in micrometers (for area calculation)")
	return p


def _get_area(x: np.ndarray, y: np.ndarray, z: np.ndarray,
			  step_size: float, voxel_size_um: float | None) -> dict:
	"""Compute surface area from a tifxyz mesh grid.

	Counts valid quads (all 4 corners finite and not the -1/-1/-1 sentinel) × step_size².
	Returns dict with area_vx2 and optionally area_cm2.
	"""
	valid = _valid_xyz_mask(x, y, z)
	valid_quads = valid[:-1, :-1] & valid[:-1, 1:] & valid[1:, :-1] & valid[1:, 1:]
	area_vx2 = int(valid_quads.sum()) * step_size ** 2
	result = {"area_vx2": area_vx2}
	if voxel_size_um is not None:
		result["area_cm2"] = area_vx2 * voxel_size_um ** 2 / 1e8
	return result


def _print_area(area: dict) -> None:
	parts = [f"area_vx2={area['area_vx2']:.0f}"]
	if "area_cm2" in area:
		parts.append(f"area_cm2={area['area_cm2']:.4f}")
	print(f"[fit2tifxyz] {' '.join(parts)}", flush=True)


def _bbox_for_xyz(x: np.ndarray, y: np.ndarray, z: np.ndarray, *, out_dir: Path) -> list[list[float]]:
	valid = _valid_xyz_mask(x, y, z)
	if not bool(valid.any()):
		print(f"[fit2tifxyz] WARNING: no valid vertices in {out_dir}; writing invalid bbox", flush=True)
		return [[-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0]]
	return [
		[float(np.nanmin(x[valid])), float(np.nanmin(y[valid])), float(np.nanmin(z[valid]))],
		[float(np.nanmax(x[valid])), float(np.nanmax(y[valid])), float(np.nanmax(z[valid]))],
	]


def _closest_point_triangle_barycentric(
	p: np.ndarray,
	a: np.ndarray,
	b: np.ndarray,
	c: np.ndarray,
) -> tuple[np.ndarray, tuple[float, float, float]]:
	ab = b - a
	ac = c - a
	ap = p - a
	d1 = float(np.dot(ab, ap))
	d2 = float(np.dot(ac, ap))
	if d1 <= 0.0 and d2 <= 0.0:
		return a, (1.0, 0.0, 0.0)

	bp = p - b
	d3 = float(np.dot(ab, bp))
	d4 = float(np.dot(ac, bp))
	if d3 >= 0.0 and d4 <= d3:
		return b, (0.0, 1.0, 0.0)

	vc = d1 * d4 - d3 * d2
	if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
		v = d1 / (d1 - d3)
		return a + v * ab, (1.0 - v, v, 0.0)

	cp = p - c
	d5 = float(np.dot(ab, cp))
	d6 = float(np.dot(ac, cp))
	if d6 >= 0.0 and d5 <= d6:
		return c, (0.0, 0.0, 1.0)

	vb = d5 * d2 - d1 * d6
	if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
		w = d2 / (d2 - d6)
		return a + w * ac, (1.0 - w, 0.0, w)

	va = d3 * d6 - d5 * d4
	if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
		w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
		return b + w * (c - b), (0.0, 1.0 - w, w)

	denom = 1.0 / (va + vb + vc)
	v = vb * denom
	w = vc * denom
	u = 1.0 - v - w
	return u * a + v * b + w * c, (u, v, w)


def _candidate_quads_for_point(
	xyz: np.ndarray,
	valid: np.ndarray,
	point: np.ndarray,
	*,
	nearest_vertices: int = 16,
) -> list[tuple[int, int]]:
	valid_rc = np.argwhere(valid)
	if valid_rc.size == 0:
		return []
	valid_xyz = xyz[valid]
	d2 = np.sum((valid_xyz.astype(np.float64, copy=False) - point.reshape(1, 3)) ** 2, axis=1)
	k = min(int(nearest_vertices), int(d2.shape[0]))
	if k <= 0:
		return []
	if k < d2.shape[0]:
		nearest = np.argpartition(d2, k - 1)[:k]
	else:
		nearest = np.arange(d2.shape[0])

	h, w = valid.shape
	quads: set[tuple[int, int]] = set()
	for idx in nearest:
		r, c = (int(v) for v in valid_rc[int(idx)])
		for qr in (r - 1, r):
			for qc in (c - 1, c):
				if qr < 0 or qc < 0 or qr + 1 >= h or qc + 1 >= w:
					continue
				if bool(valid[qr, qc] and valid[qr + 1, qc] and valid[qr, qc + 1] and valid[qr + 1, qc + 1]):
					quads.add((qr, qc))
	if quads:
		return sorted(quads)

	valid_quads = valid[:-1, :-1] & valid[1:, :-1] & valid[:-1, 1:] & valid[1:, 1:]
	quad_rc = np.argwhere(valid_quads)
	if quad_rc.size == 0:
		return []
	centers = (
		xyz[:-1, :-1][valid_quads].astype(np.float64)
		+ xyz[1:, :-1][valid_quads].astype(np.float64)
		+ xyz[:-1, 1:][valid_quads].astype(np.float64)
		+ xyz[1:, 1:][valid_quads].astype(np.float64)
	) * 0.25
	center_d2 = np.sum((centers - point.reshape(1, 3)) ** 2, axis=1)
	kq = min(32, int(center_d2.shape[0]))
	nearest_q = np.argpartition(center_d2, kq - 1)[:kq] if kq < center_d2.shape[0] else np.arange(center_d2.shape[0])
	return [(int(quad_rc[i, 0]), int(quad_rc[i, 1])) for i in nearest_q]


def _project_point_to_mesh_grid(
	point_xyz: np.ndarray,
	x: np.ndarray,
	y: np.ndarray,
	z: np.ndarray,
) -> tuple[float, float]:
	xyz = np.stack([x, y, z], axis=-1).astype(np.float64, copy=False)
	valid = _valid_xyz_mask(x, y, z)
	p = np.asarray(point_xyz, dtype=np.float64)
	if p.shape != (3,) or not np.isfinite(p).all():
		raise ValueError(f"approval output mask contour point must be finite xyz, got {point_xyz}")

	best_d2 = float("inf")
	best_rc: tuple[float, float] | None = None
	for r, c in _candidate_quads_for_point(xyz, valid, p):
		p00 = xyz[r, c]
		p10 = xyz[r + 1, c]
		p01 = xyz[r, c + 1]
		p11 = xyz[r + 1, c + 1]
		for tri in (0, 1):
			if tri == 0:
				closest, (_u, v, w) = _closest_point_triangle_barycentric(p, p00, p10, p11)
				row = float(r) + v + w
				col = float(c) + w
			else:
				closest, (_u, v, w) = _closest_point_triangle_barycentric(p, p00, p11, p01)
				row = float(r) + v
				col = float(c) + v + w
			d2 = float(np.sum((closest - p) ** 2))
			if d2 < best_d2:
				best_d2 = d2
				best_rc = (row, col)
	if best_rc is None:
		raise ValueError("approval output mask could not project contour point onto final mesh")
	return best_rc


def _project_contour_xyz_to_grid(
	contour_xyz: list,
	x: np.ndarray,
	y: np.ndarray,
	z: np.ndarray,
) -> list[tuple[float, float]]:
	return [
		_project_point_to_mesh_grid(np.asarray(point, dtype=np.float64), x, y, z)
		for point in contour_xyz
	]


def _mark_segment_vertices(
	mask: np.ndarray,
	p0: tuple[float, float],
	p1: tuple[float, float],
	*,
	eps: float = 1.0e-6,
) -> None:
	r0, c0 = p0
	r1, c1 = p1
	rmin = max(0, int(math.floor(min(r0, r1) - eps)))
	rmax = min(mask.shape[0] - 1, int(math.ceil(max(r0, r1) + eps)))
	cmin = max(0, int(math.floor(min(c0, c1) - eps)))
	cmax = min(mask.shape[1] - 1, int(math.ceil(max(c0, c1) + eps)))
	seg = np.asarray([r1 - r0, c1 - c0], dtype=np.float64)
	seg_len2 = float(np.dot(seg, seg))
	for r in range(rmin, rmax + 1):
		for c in range(cmin, cmax + 1):
			v = np.asarray([float(r) - r0, float(c) - c0], dtype=np.float64)
			if seg_len2 <= eps:
				dist2 = float(np.dot(v, v))
			else:
				t = max(0.0, min(1.0, float(np.dot(v, seg) / seg_len2)))
				d = v - t * seg
				dist2 = float(np.dot(d, d))
			if dist2 <= eps * eps:
				mask[r, c] = True


def _rasterize_contours(
	contours_rc: list[list[tuple[float, float]]],
	shape: tuple[int, int],
) -> np.ndarray:
	h, w = (int(shape[0]), int(shape[1]))
	out = np.zeros((h, w), dtype=bool)
	if h <= 0 or w <= 0:
		return out

	for r in range(h):
		y = float(r)
		xints: list[float] = []
		for contour in contours_rc:
			if len(contour) < 3:
				continue
			for i, (r0, c0) in enumerate(contour):
				r1, c1 = contour[(i + 1) % len(contour)]
				if (r0 > y) != (r1 > y):
					t = (y - r0) / (r1 - r0)
					xints.append(float(c0 + t * (c1 - c0)))
		xints.sort()
		for i in range(0, len(xints) - 1, 2):
			c0 = int(math.ceil(min(xints[i], xints[i + 1]) - 1.0e-6))
			c1 = int(math.floor(max(xints[i], xints[i + 1]) + 1.0e-6))
			if c1 < 0 or c0 >= w:
				continue
			out[r, max(0, c0):min(w, c1 + 1)] = True

	for contour in contours_rc:
		if len(contour) < 2:
			continue
		for i, p0 in enumerate(contour):
			_mark_segment_vertices(out, p0, contour[(i + 1) % len(contour)])
	return out


def _dilate_chebyshev(mask: np.ndarray, radius: int) -> np.ndarray:
	r = int(radius)
	if r <= 0:
		return mask.astype(bool, copy=True)
	src = mask.astype(bool, copy=False)
	out = src.copy()
	h, w = src.shape
	for dr in range(-r, r + 1):
		rs0 = max(0, -dr)
		rs1 = min(h, h - dr)
		rd0 = max(0, dr)
		rd1 = min(h, h + dr)
		for dc in range(-r, r + 1):
			cs0 = max(0, -dc)
			cs1 = min(w, w - dc)
			cd0 = max(0, dc)
			cd1 = min(w, w + dc)
			out[rd0:rd1, cd0:cd1] |= src[rs0:rs1, cs0:cs1]
	return out


def _approval_output_mask_for_layer(
	payload: dict,
	x: np.ndarray,
	y: np.ndarray,
	z: np.ndarray,
) -> np.ndarray:
	contours = payload.get("contours_xyz", payload.get("contours", []))
	if not isinstance(contours, list) or not contours:
		raise ValueError("approval output mask payload has no contours_xyz")
	projected = [
		_project_contour_xyz_to_grid(contour, x, y, z)
		for contour in contours
		if isinstance(contour, list) and len(contour) >= 3
	]
	if not projected:
		raise ValueError("approval output mask payload has no usable contours")
	mask = _rasterize_contours(projected, x.shape)
	return _dilate_chebyshev(mask, int(payload.get("dilation_radius", payload.get("dilate", 0))))


def _apply_output_vertex_mask(
	x: np.ndarray,
	y: np.ndarray,
	z: np.ndarray,
	d: np.ndarray | None,
	mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
	if mask.shape != x.shape:
		raise ValueError(f"output mask shape mismatch: {mask.shape} vs {x.shape}")
	x_out = x.astype(np.float32, copy=True)
	y_out = y.astype(np.float32, copy=True)
	z_out = z.astype(np.float32, copy=True)
	x_out[~mask] = -1.0
	y_out[~mask] = -1.0
	z_out[~mask] = -1.0
	d_out = None
	if d is not None:
		d_out = d.astype(np.float32, copy=True)
		d_out[~mask] = -1.0
	return x_out, y_out, z_out, d_out


def _write_tifxyz(*, out_dir: Path, x: np.ndarray, y: np.ndarray, z: np.ndarray,
				  scale: float, d: np.ndarray | None = None,
				  model_source: Path | None = None,
				  copy_model: bool = False, fit_config: dict | None = None,
				  area: dict | None = None,
				  components: list[list[int]] | None = None) -> None:
	out_dir.mkdir(parents=True, exist_ok=True)
	if x.shape != y.shape or x.shape != z.shape:
		raise ValueError("x/y/z must have identical shapes")
	if x.ndim != 2:
		raise ValueError("x/y/z must be 2D")

	xf = x.astype(np.float32, copy=False)
	yf = y.astype(np.float32, copy=False)
	zf = z.astype(np.float32, copy=False)

	meta = {
		"uuid": str(out_dir.name),
		"type": "seg",
		"format": "tifxyz",
		"scale": [float(scale), float(scale)],
		"bbox": _bbox_for_xyz(xf, yf, zf, out_dir=out_dir),
	}
	if components is not None:
		meta["components"] = components
	if area is not None:
		meta.update(area)
	if model_source is not None:
		meta["model_source"] = str(model_source)
	if fit_config is not None:
		meta["fit_config"] = fit_config
	(out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
	tifffile.imwrite(str(out_dir / "x.tif"), xf, compression="lzw")
	tifffile.imwrite(str(out_dir / "y.tif"), yf, compression="lzw")
	tifffile.imwrite(str(out_dir / "z.tif"), zf, compression="lzw")
	if d is not None:
		tifffile.imwrite(str(out_dir / "d.tif"), d.astype(np.float32, copy=False), compression="lzw")

	if model_source is not None:
		dest = out_dir / "model.pt"
		if dest.is_symlink() or dest.exists():
			dest.unlink()
		if copy_model:
			shutil.copy2(str(model_source.resolve()), str(dest))
		else:
			dest.symlink_to(model_source.resolve())


def main(argv: list[str] | None = None) -> int:
	parser = _build_parser()
	args = cli_json.parse_args(parser, argv)
	cfg = ExportConfig(
		input=str(args.input),
		output=str(args.output),
		prefix=str(args.prefix),
		single_segment=bool(args.single_segment),
		copy_model=bool(args.copy_model),
		output_name=None if args.output_name in (None, "") else str(args.output_name),
		voxel_size_um=args.voxel_size_um,
	)

	dev = torch.device(cfg.device)
	st = torch.load(cfg.input, map_location=dev, weights_only=False)
	if not isinstance(st, dict):
		raise ValueError("expected a state_dict checkpoint")
	model_params = st.get("_model_params_", None)
	if not isinstance(model_params, dict):
		model_params = None
	fit_config = st.get("_fit_config_", None)
	if not isinstance(fit_config, dict):
		fit_config = None
	corr_points_results = st.get("_corr_points_results_", None)
	if not isinstance(corr_points_results, dict):
		corr_points_results = None
	approval_output_mask = st.get("_approval_inpaint_output_mask_", None)
	if not isinstance(approval_output_mask, dict):
		approval_output_mask = None

	# Reconstruct mesh (3, D, Hm, Wm) — pyramid stores full xyz positions
	mdl = model.Model3D.from_checkpoint(st, device=dev)
	mesh = mdl.mesh_coarse()

	_, D, Hm, Wm = (int(v) for v in mesh.shape)
	mesh_np = mesh.detach().cpu().numpy()  # (3, D, Hm, Wm)

	mesh_step = 100
	if model_params is not None:
		mesh_step = int(model_params.get("mesh_step", 100))
	xy_step_fullres = float(mesh_step)
	meta_scale = 1.0 / xy_step_fullres

	out_base = Path(cfg.output)
	out_base.mkdir(parents=True, exist_ok=True)

	BORDER_W = 2

	print(f"[fit2tifxyz] exporting D={D} Hm={Hm} Wm={Wm}, mesh already in fullres coords"
		  f", voxel_size_um={cfg.voxel_size_um}")
	if approval_output_mask is not None:
		n_contours = len(approval_output_mask.get("contours_xyz", []))
		radius = int(approval_output_mask.get("dilation_radius", approval_output_mask.get("dilate", 0)))
		print(f"[fit2tifxyz] approval-inpaint output mask: contours={n_contours} dilate={radius}", flush=True)

	if cfg.single_segment:
		# Combine all depth layers horizontally
		total_w = Wm * D + max(0, D - 1) * BORDER_W
		x_all = np.full((Hm, total_w), -1.0, dtype=np.float32)
		y_all = np.full((Hm, total_w), -1.0, dtype=np.float32)
		z_all = np.full((Hm, total_w), -1.0, dtype=np.float32)
		d_all = np.full((Hm, total_w), -1.0, dtype=np.float32)

		col = 0
		components: list[list[int]] = []
		for d in range(D):
			x_layer = mesh_np[0, d]  # (Hm, Wm)
			y_layer = mesh_np[1, d]
			z_layer = mesh_np[2, d]
			d_layer = np.full((Hm, Wm), float(d), dtype=np.float32)
			if approval_output_mask is not None:
				mask = _approval_output_mask_for_layer(approval_output_mask, x_layer, y_layer, z_layer)
				x_layer, y_layer, z_layer, d_layer = _apply_output_vertex_mask(
					x_layer, y_layer, z_layer, d_layer, mask
				)
			else:
				valid_layer = _valid_xyz_mask(x_layer, y_layer, z_layer)
				d_layer = np.where(valid_layer, float(d), -1.0).astype(np.float32)
			x_all[:, col:col + Wm] = x_layer
			y_all[:, col:col + Wm] = y_layer
			z_all[:, col:col + Wm] = z_layer
			d_all[:, col:col + Wm] = d_layer
			components.append([col, col + Wm])
			col += Wm + BORDER_W

		seg_name = cfg.output_name if cfg.output_name else f"{cfg.prefix}.tifxyz"
		out_dir = out_base / seg_name
		area = _get_area(x_all, y_all, z_all, xy_step_fullres, cfg.voxel_size_um)
		_write_tifxyz(out_dir=out_dir, x=x_all, y=y_all, z=z_all, d=d_all, scale=meta_scale,
					  model_source=Path(cfg.input), copy_model=cfg.copy_model, fit_config=fit_config,
					  area=area, components=components if D > 1 else None)
		_print_area(area)
		if model_params is not None:
			(out_dir / "model_params.json").write_text(json.dumps(model_params, indent=2) + "\n", encoding="utf-8")
		if corr_points_results is not None:
			(out_dir / "corr_points_results.json").write_text(json.dumps(corr_points_results, indent=2) + "\n", encoding="utf-8")
	else:
		# One tifxyz per depth layer (winding)
		total_area = {"area_vx2": 0.0}
		if cfg.voxel_size_um is not None:
			total_area["area_cm2"] = 0.0
		for d in range(D):
			x = mesh_np[0, d]  # (Hm, Wm) already in fullres
			y = mesh_np[1, d]
			z = mesh_np[2, d]
			if approval_output_mask is not None:
				mask = _approval_output_mask_for_layer(approval_output_mask, x, y, z)
				x, y, z, _ = _apply_output_vertex_mask(x, y, z, None, mask)
			valid = _valid_xyz_mask(x, y, z)
			d_layer = np.where(valid, float(d), -1.0).astype(np.float32)
			area = _get_area(x, y, z, xy_step_fullres, cfg.voxel_size_um)
			total_area["area_vx2"] += area["area_vx2"]
			if "area_cm2" in area:
				total_area["area_cm2"] += area["area_cm2"]
			out_dir = out_base / f"{cfg.prefix}{d:04d}.tifxyz"
			_write_tifxyz(out_dir=out_dir, x=x, y=y, z=z, d=d_layer, scale=meta_scale,
						  model_source=Path(cfg.input), copy_model=cfg.copy_model, fit_config=fit_config,
						  area=area)
			if model_params is not None:
				(out_dir / "model_params.json").write_text(json.dumps(model_params, indent=2) + "\n", encoding="utf-8")
			if corr_points_results is not None:
				(out_dir / "corr_points_results.json").write_text(json.dumps(corr_points_results, indent=2) + "\n", encoding="utf-8")
		_print_area(total_area)

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
