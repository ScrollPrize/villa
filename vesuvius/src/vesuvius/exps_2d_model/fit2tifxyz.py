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

import torch.nn.functional as F

import cli_json


@dataclass(frozen=True)
class ExportConfig:
	input: str
	output: str
	prefix: str = "winding_"
	device: str = "cpu"
	downscale: float = 4.0
	offset_x: float = 0.0
	offset_y: float = 0.0
	offset_z: int = 0
	# Defaults follow the requested convention:
	# - scale unit is the grid-step (meta.scale=0.1)
	# - z-step is 1 and we only output every 10th slice
	z0: int = 0
	z_step: int = 10
	grid_step: int = 10
	scale: float = 0.1
	single_segment: bool = False
	copy_model: bool = False
	output_name: str | None = None


def _build_parser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(description="Export fit model grid as tifxyz surfaces (one per winding)")
	cli_json.add_args(p)
	g = p.add_argument_group("io")
	g.add_argument("--input", required=True, help="Model checkpoint (.pt) produced by fit")
	g.add_argument("--output", required=True, help="Output directory (will contain one tifxyz dir per winding)")
	g.add_argument("--prefix", default="winding_", help="Output tifxyz directory prefix (default: winding_)")
	g.add_argument("--downscale", type=float, default=4.0, help="Fit-time downscale; x/y are multiplied by this")
	g.add_argument(
		"--offset",
		type=float,
		nargs=3,
		default=(0.0, 0.0, 0.0),
		help="Offsets (x y z) in original voxel/pixel units.",
	)
	g.add_argument(
		"--single-segment",
		action="store_true",
		default=False,
		help="Export all windings into a single tifxyz (horizontally, separated by 2 invalid-point border)",
	)
	g.add_argument(
		"--copy-model",
		action="store_true",
		default=False,
		help="Copy the model checkpoint into the tifxyz dir instead of creating a symlink",
	)
	g.add_argument(
		"--output-name",
		default=None,
		help="Override the tifxyz directory name (e.g. 'my_segment_v002.tifxyz')",
	)
	return p


def _integrate_param_pyramid(src: list[torch.Tensor]) -> torch.Tensor:
	v = src[-1]
	for d in reversed(src[:-1]):
		up = F.interpolate(v, scale_factor=2.0, mode="bilinear", align_corners=True)
		up = up[:, :, : int(d.shape[2]), : int(d.shape[3])]
		v = up + d
	return v


def _mesh_coarse_from_state_dict(st: dict) -> torch.Tensor:
	keys = sorted(k for k in st.keys() if k.startswith("mesh_ms.") and k.split(".")[-1].isdigit())
	if not keys:
		raise ValueError("checkpoint missing mesh_ms.* tensors")
	# state_dict indices are 0..n_scales-1; we want them in order
	idx_keys = sorted(((int(k.split(".")[-1]), k) for k in keys), key=lambda t: t[0])
	pyr = [st[k].detach() for _i, k in idx_keys]
	return _integrate_param_pyramid(pyr)


def _apply_global_transform_from_state_dict(*, uv: torch.Tensor, st: dict) -> torch.Tensor:
	# uv: (N,2,H,W) in pixel units; apply rotation/scale around mesh center.
	theta = st.get("theta", torch.zeros((), dtype=torch.float32, device=uv.device)).detach().to(device=uv.device, dtype=torch.float32)
	winding_scale = st.get("winding_scale", torch.ones((), dtype=torch.float32, device=uv.device)).detach().to(device=uv.device, dtype=torch.float32)

	u = uv[:, 0:1]
	v = uv[:, 1:2]
	u = winding_scale * u

	min_u = torch.amin(u)
	max_u = torch.amax(u)
	min_v = torch.amin(v)
	max_v = torch.amax(v)
	xc = 0.5 * (min_u + max_u)
	yc = 0.5 * (min_v + max_v)

	c = torch.cos(theta)
	s = torch.sin(theta)
	x = xc + c * (u - xc) - s * (v - yc)
	y = yc + s * (u - xc) + c * (v - yc)
	return torch.cat([x, y], dim=1)


def _write_tifxyz(*, out_dir: Path, x: np.ndarray, y: np.ndarray, z: np.ndarray, scale: float, model_source: Path | None = None, copy_model: bool = False, fit_config: dict | None = None) -> None:
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
		"bbox": [
			[float(np.min(xf)), float(np.min(yf)), float(np.min(zf))],
			[float(np.max(xf)), float(np.max(yf)), float(np.max(zf))],
		],
	}
	if model_source is not None:
		meta["model_source"] = str(model_source)
	else:
		meta.pop("model_source", None)
	if fit_config is not None:
		meta["fit_config"] = fit_config
	(out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
	tifffile.imwrite(str(out_dir / "x.tif"), xf, compression="lzw")
	tifffile.imwrite(str(out_dir / "y.tif"), yf, compression="lzw")
	tifffile.imwrite(str(out_dir / "z.tif"), zf, compression="lzw")

	# Create model.pt symlink or copy pointing to the source checkpoint
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
	raw_argv = list(sys.argv[1:] if argv is None else argv)
	offset_explicit = "--offset" in raw_argv
	args = cli_json.parse_args(parser, argv)
	base = {
		"input": str(args.input),
		"output": str(args.output),
		"prefix": str(args.prefix),
		"downscale": float(args.downscale),
		"offset_x": float(args.offset[0]),
		"offset_y": float(args.offset[1]),
		"offset_z": int(round(float(args.offset[2]))),
		"single_segment": bool(args.single_segment),
		"copy_model": bool(args.copy_model),
		"output_name": None if args.output_name in (None, "") else str(args.output_name),
	}
	cfg = ExportConfig(**base)
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

	if model_params is not None:
		c6_full = model_params.get("crop_fullres_xyzwhd", None)
		c6_model = model_params.get("crop_xyzwhd", None)
		c6 = None
		crop_key = "none"
		if isinstance(c6_full, (list, tuple)) and len(c6_full) == 6:
			c6 = c6_full
			crop_key = "crop_fullres_xyzwhd"
		elif isinstance(c6_model, (list, tuple)) and len(c6_model) == 6:
			c6 = c6_model
			crop_key = "crop_xyzwhd"
		# Margin from expanded data: mesh model coords are shifted by margin.
		# Subtract margin (in full-res) from offset so conversion is correct.
		margin_modelpx = model_params.get("data_margin_modelpx", (0.0, 0.0))
		if not isinstance(margin_modelpx, (list, tuple)) or len(margin_modelpx) != 2:
			margin_modelpx = (0.0, 0.0)
		if c6 is not None:
			x0c, y0c, _wc, _hc, z0c, _d = (int(v) for v in c6)
			if not bool(offset_explicit):
				if crop_key == "crop_fullres_xyzwhd":
					base["offset_x"] = float(x0c) - float(margin_modelpx[0]) * float(base["downscale"])
					base["offset_y"] = float(y0c) - float(margin_modelpx[1]) * float(base["downscale"])
				else:
					ds = float(base["downscale"])
					if ds <= 0.0:
						ds = 1.0
					base["offset_x"] = float(x0c) * ds - float(margin_modelpx[0]) * ds
					base["offset_y"] = float(y0c) * ds - float(margin_modelpx[1]) * ds
				base["offset_z"] = 0
			base["z0"] = int(z0c)
		if "z_step_vx" in model_params:
			base["z_step"] = max(1, int(model_params["z_step_vx"]))
		cfg = ExportConfig(**base)

	crop_bounds_fullres: tuple[float, float, float, float] | None = None
	if model_params is not None:
		c6_full = model_params.get("crop_fullres_xyzwhd", None)
		c6_model = model_params.get("crop_xyzwhd", None)
		# When data has margins, use expanded data bounds for clipping
		# (the valid channel already handled masking during fit).
		data_sz = model_params.get("data_size_modelpx", (0, 0))
		if not isinstance(data_sz, (list, tuple)) or len(data_sz) != 2:
			data_sz = (0, 0)
		margin_modelpx = model_params.get("data_margin_modelpx", (0.0, 0.0))
		if not isinstance(margin_modelpx, (list, tuple)) or len(margin_modelpx) != 2:
			margin_modelpx = (0.0, 0.0)
		ds_cb = float(cfg.downscale) if float(cfg.downscale) > 0.0 else 1.0
		if int(data_sz[0]) > 0 and int(data_sz[1]) > 0:
			# Use full data extent for crop bounds
			x0 = float(cfg.offset_x)
			y0 = float(cfg.offset_y)
			x1 = x0 + float(max(0, int(data_sz[1]) - 1)) * ds_cb
			y1 = y0 + float(max(0, int(data_sz[0]) - 1)) * ds_cb
			crop_bounds_fullres = (x0, y0, x1, y1)
		elif isinstance(c6_full, (list, tuple)) and len(c6_full) == 6:
			x0c, y0c, wc, hc, _z0c, _d = (int(v) for v in c6_full)
			x0 = float(x0c)
			y0 = float(y0c)
			x1 = x0 + float(max(0, int(wc) - 1))
			y1 = y0 + float(max(0, int(hc) - 1))
			crop_bounds_fullres = (x0, y0, x1, y1)
		elif isinstance(c6_model, (list, tuple)) and len(c6_model) == 6:
			x0c, y0c, wc, hc, _z0c, _d = (int(v) for v in c6_model)
			x0 = float(x0c) * ds_cb
			y0 = float(y0c) * ds_cb
			x1 = x0 + float(max(0, int(wc) - 1)) * ds_cb
			y1 = y0 + float(max(0, int(hc) - 1)) * ds_cb
			crop_bounds_fullres = (x0, y0, x1, y1)

	offset_src = "cli --offset" if bool(offset_explicit) else "model crop"
	crop_dbg = None if model_params is None else {
		"crop_fullres_xyzwhd": model_params.get("crop_fullres_xyzwhd", None),
		"crop_xyzwhd": model_params.get("crop_xyzwhd", None),
	}
	print(
		"[fit2tifxyz] using offsets",
		{
			"source": offset_src,
			"offset_x": float(cfg.offset_x),
			"offset_y": float(cfg.offset_y),
			"offset_z": int(cfg.offset_z),
			"downscale": float(cfg.downscale),
			"crop_from_checkpoint": crop_dbg,
		},
	)

	mesh_uv = _mesh_coarse_from_state_dict(st).to(device=dev, dtype=torch.float32)
	xy = _apply_global_transform_from_state_dict(uv=mesh_uv, st=st)
	# Determine export z stride.
	k = max(1, int(round(float(cfg.grid_step) / float(max(1, int(cfg.z_step))))))
	idx_z = list(range(0, int(xy.shape[0]), int(k)))
	if not idx_z:
		idx_z = [0]

	xy_lr = xy.permute(0, 2, 3, 1).detach().cpu().numpy()  # (N,Hm,Wm,2)
	xy_lr = xy_lr * float(cfg.downscale)
	xy_lr[..., 0] += float(cfg.offset_x)
	xy_lr[..., 1] += float(cfg.offset_y)

	out_base = Path(cfg.output)
	out_base.mkdir(parents=True, exist_ok=True)

	# Convention:
	# - Height is Z.
	# - Width is mesh-y (across surfaces).
	# - Create one tifxyz per winding (mesh-x column).
	n, hm, wm, _c2 = (int(v) for v in xy_lr.shape)
	idx_z_a = np.asarray(idx_z, dtype=np.int64)

	# z coordinates are exported in full-resolution units.
	# Model z_step is in fit-space; multiply by XY downscale for effective full-res z stride.
	z_step_fullres = int(round(float(cfg.z_step) * float(cfg.downscale)))
	if z_step_fullres <= 0:
		raise ValueError(f"invalid effective full-res z-step: z_step={cfg.z_step}, downscale={cfg.downscale}, z_step_fullres={z_step_fullres}")
	z_vals = np.asarray([cfg.z0 + int(cfg.offset_z) + zi * int(z_step_fullres) for zi in idx_z], dtype=np.float32)
	z_grid = z_vals.reshape(-1, 1).repeat(hm, axis=1)

	meta_scale = float(cfg.scale) * float(cfg.downscale)
	BORDER_W = 2  # invalid-point border width between windings in single-segment mode

	def _apply_crop_mask(x: np.ndarray, y: np.ndarray, z_use: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
		mask = None
		if crop_bounds_fullres is not None:
			x0, y0, x1, y1 = crop_bounds_fullres
			v = np.isfinite(x) & np.isfinite(y) & (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
			mask = (v.astype(np.uint8) * 255)
			if np.any(~v):
				x = x.copy()
				y = y.copy()
				z_use = z_use.copy()
				x[~v] = -1.0
				y[~v] = -1.0
				z_use[~v] = -1.0
		return x, y, z_use, mask

	if cfg.single_segment:
		# Combine all windings horizontally into one tifxyz, separated by
		# BORDER_W columns of invalid (-1) points.
		nz = len(idx_z)
		total_w = wm * hm + max(0, wm - 1) * BORDER_W
		x_all = np.full((nz, total_w), -1.0, dtype=np.float32)
		y_all = np.full((nz, total_w), -1.0, dtype=np.float32)
		z_all = np.full((nz, total_w), -1.0, dtype=np.float32)
		mask_all = np.full((nz, total_w), 0, dtype=np.uint8) if crop_bounds_fullres is not None else None

		col = 0
		for wi in range(wm):
			x_w = xy_lr[idx_z_a, :, wi, 0]
			y_w = xy_lr[idx_z_a, :, wi, 1]
			x_w, y_w, z_w, mask_w = _apply_crop_mask(x_w, y_w, z_grid)
			x_all[:, col:col + hm] = x_w
			y_all[:, col:col + hm] = y_w
			z_all[:, col:col + hm] = z_w
			if mask_all is not None and mask_w is not None:
				mask_all[:, col:col + hm] = mask_w
			col += hm + BORDER_W  # skip border columns (already -1)

		seg_name = cfg.output_name if cfg.output_name else f"{cfg.prefix}.tifxyz"
		out_dir = out_base / seg_name
		_write_tifxyz(out_dir=out_dir, x=x_all, y=y_all, z=z_all, scale=meta_scale, model_source=Path(cfg.input), copy_model=cfg.copy_model, fit_config=fit_config)
		if model_params is not None:
			(out_dir / "model_params.json").write_text(json.dumps(model_params, indent=2) + "\n", encoding="utf-8")
		if mask_all is not None:
			tifffile.imwrite(str(out_dir / "mask.tif"), mask_all, compression="lzw")
	else:
		for wi in range(wm):
			x = xy_lr[idx_z_a, :, wi, 0]
			y = xy_lr[idx_z_a, :, wi, 1]
			x, y, z_use, mask = _apply_crop_mask(x, y, z_grid)
			out_dir = out_base / f"{cfg.prefix}{wi:04d}.tifxyz"
			_write_tifxyz(out_dir=out_dir, x=x, y=y, z=z_use, scale=meta_scale, model_source=Path(cfg.input), copy_model=cfg.copy_model, fit_config=fit_config)
			if model_params is not None:
				(out_dir / "model_params.json").write_text(json.dumps(model_params, indent=2) + "\n", encoding="utf-8")
			if mask is not None:
				tifffile.imwrite(str(out_dir / "mask.tif"), mask, compression="lzw")

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
