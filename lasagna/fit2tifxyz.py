from __future__ import annotations

import argparse
import json
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

	Counts valid quads (all 4 corners finite and != -1) × step_size².
	Returns dict with area_vx2 and optionally area_cm2.
	"""
	valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
	valid &= ~((x == -1.0) & (y == -1.0) & (z == -1.0))
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
	valid = np.isfinite(xf) & np.isfinite(yf) & np.isfinite(zf)
	valid &= ~((xf == -1.0) & (yf == -1.0) & (zf == -1.0))
	if np.any(valid):
		bbox = [
			[float(np.nanmin(xf[valid])), float(np.nanmin(yf[valid])), float(np.nanmin(zf[valid]))],
			[float(np.nanmax(xf[valid])), float(np.nanmax(yf[valid])), float(np.nanmax(zf[valid]))],
		]
	else:
		bbox = [[-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0]]

	meta = {
		"uuid": str(out_dir.name),
		"type": "seg",
		"format": "tifxyz",
		"scale": [float(scale), float(scale)],
		"bbox": bbox,
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


def _as_numpy_float32(value: object, *, name: str) -> np.ndarray:
	if isinstance(value, torch.Tensor):
		return value.detach().cpu().numpy().astype(np.float32, copy=False)
	arr = np.asarray(value, dtype=np.float32)
	if arr.size == 0:
		raise ValueError(f"flatten checkpoint field {name!r} is empty")
	return arr


def _export_flatten_checkpoint(
	*,
	st: dict,
	cfg: ExportConfig,
	model_params: dict | None,
	fit_config: dict | None,
) -> int:
	map_yx = _as_numpy_float32(st["flatten_map_flat"], name="flatten_map_flat")
	if map_yx.ndim != 3 or map_yx.shape[-1] != 2:
		raise ValueError("flatten_map_flat must have shape (H, W, 2)")

	mesh = st.get("mesh_flat")
	if mesh is None:
		dev = torch.device(cfg.device)
		mdl = model.Model3D.from_checkpoint(st, device=dev)
		mesh_np = mdl.mesh_coarse().detach().cpu().numpy().astype(np.float32, copy=False)
	else:
		mesh_np = _as_numpy_float32(mesh, name="mesh_flat")
	if mesh_np.ndim != 4 or mesh_np.shape[0] != 3 or mesh_np.shape[1] != 1:
		raise ValueError("flatten mesh_flat must have shape (3, 1, H, W)")

	x = mesh_np[0, 0].astype(np.float32, copy=False)
	y = mesh_np[1, 0].astype(np.float32, copy=False)
	z = mesh_np[2, 0].astype(np.float32, copy=False)
	if x.shape != tuple(map_yx.shape[:2]):
		raise ValueError("flatten_map_flat shape does not match mesh_flat shape")

	point_mask = st.get("flatten_point_mask")
	if point_mask is not None:
		if isinstance(point_mask, torch.Tensor):
			mask_np = point_mask.detach().cpu().numpy().astype(bool, copy=False)
		else:
			mask_np = np.asarray(point_mask, dtype=bool)
		if mask_np.shape != x.shape:
			raise ValueError("flatten_point_mask shape does not match mesh_flat shape")
		x = np.where(mask_np, x, -1.0).astype(np.float32, copy=False)
		y = np.where(mask_np, y, -1.0).astype(np.float32, copy=False)
		z = np.where(mask_np, z, -1.0).astype(np.float32, copy=False)

	mesh_step = 100
	if model_params is not None:
		mesh_step = int(model_params.get("mesh_step", 100))
	xy_step_fullres = float(mesh_step)
	meta_scale = 1.0 / xy_step_fullres

	out_base = Path(cfg.output)
	out_base.mkdir(parents=True, exist_ok=True)

	valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
	valid &= ~((x == -1.0) & (y == -1.0) & (z == -1.0))
	d = np.where(valid, 0.0, -1.0).astype(np.float32, copy=False)
	seg_name = cfg.output_name if cfg.output_name else f"{cfg.prefix}0000.tifxyz"
	out_dir = out_base / seg_name
	area = _get_area(x, y, z, xy_step_fullres, cfg.voxel_size_um)
	_write_tifxyz(
		out_dir=out_dir,
		x=x,
		y=y,
		z=z,
		d=d,
		scale=meta_scale,
		model_source=Path(cfg.input),
		copy_model=cfg.copy_model,
		fit_config=fit_config,
		area=area,
	)
	if model_params is not None:
		(out_dir / "model_params.json").write_text(json.dumps(model_params, indent=2) + "\n", encoding="utf-8")
	_print_area(area)
	print(f"[fit2tifxyz] exported {out_dir.name}", flush=True)
	return 0


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
	if "flatten_map_flat" in st:
		return _export_flatten_checkpoint(
			st=st,
			cfg=cfg,
			model_params=model_params,
			fit_config=fit_config,
		)

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
			x_all[:, col:col + Wm] = mesh_np[0, d]  # (Hm, Wm)
			y_all[:, col:col + Wm] = mesh_np[1, d]
			z_all[:, col:col + Wm] = mesh_np[2, d]
			d_all[:, col:col + Wm] = float(d)
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
			valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
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
