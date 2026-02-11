from __future__ import annotations

import argparse
import json
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


def _write_tifxyz(*, out_dir: Path, x: np.ndarray, y: np.ndarray, z: np.ndarray, scale: float) -> None:
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
	(out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
	tifffile.imwrite(str(out_dir / "x.tif"), xf, compression="lzw")
	tifffile.imwrite(str(out_dir / "y.tif"), yf, compression="lzw")
	tifffile.imwrite(str(out_dir / "z.tif"), zf, compression="lzw")


def main(argv: list[str] | None = None) -> int:
	parser = _build_parser()
	args = cli_json.parse_args(parser, argv)
	base = {
		"input": str(args.input),
		"output": str(args.output),
		"prefix": str(args.prefix),
		"downscale": float(args.downscale),
		"offset_x": float(args.offset[0]),
		"offset_y": float(args.offset[1]),
		"offset_z": int(round(float(args.offset[2]))),
	}
	cfg = ExportConfig(**base)
	dev = torch.device(cfg.device)

	st = torch.load(cfg.input, map_location=dev)
	if not isinstance(st, dict):
		raise ValueError("expected a state_dict checkpoint")
	model_params = st.get("_model_params_", None)
	if not isinstance(model_params, dict):
		model_params = None

	if model_params is not None:
		c6 = model_params.get("crop_xyzwhd", None)
		if isinstance(c6, (list, tuple)) and len(c6) == 6:
			_x0c, _y0c, _wc, _hc, z0c, _d = (int(v) for v in c6)
			base["z0"] = int(z0c)
		if "z_step_vx" in model_params:
			base["z_step"] = max(1, int(model_params["z_step_vx"]))
		cfg = ExportConfig(**base)

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

	z_vals = np.asarray([cfg.z0 + int(cfg.offset_z) + zi * int(cfg.z_step) for zi in idx_z], dtype=np.float32)
	z_grid = z_vals.reshape(-1, 1).repeat(hm, axis=1)

	meta_scale = float(cfg.scale) * float(cfg.downscale)
	for wi in range(wm):
		x = xy_lr[idx_z_a, :, wi, 0]
		y = xy_lr[idx_z_a, :, wi, 1] - 256
		z_use = z_grid
		mask = None
		if model_params is not None:
			c6 = model_params.get("crop_xyzwhd", None)
			if isinstance(c6, (list, tuple)) and len(c6) == 6:
				_x0c, _y0c, wc, hc, _z0c, _d = (int(v) for v in c6)
				x0 = float(cfg.offset_x)
				y0 = float(cfg.offset_y)
				x1 = x0 + float(max(0, int(wc) - 1))
				y1 = y0 + float(max(0, int(hc) - 1))
				v = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
				mask = (v.astype(np.uint8) * 255)
				if np.any(~v):
					x = x.copy()
					y = y.copy()
					z_use = z_grid.copy()
					x[~v] = -1.0
					y[~v] = -1.0
					z_use[~v] = -1.0
		out_dir = out_base / f"{cfg.prefix}{wi:04d}.tifxyz"
		_write_tifxyz(out_dir=out_dir, x=x, y=y, z=z_use, scale=meta_scale)
		if model_params is not None:
			(out_dir / "model_params.json").write_text(json.dumps(model_params, indent=2) + "\n", encoding="utf-8")
		if mask is not None:
			tifffile.imwrite(str(out_dir / "mask.tif"), mask, compression="lzw")

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
