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
	single_segment: bool = False
	copy_model: bool = False
	output_name: str | None = None


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
	return p


def _integrate_pyramid_3d(src: list[torch.Tensor]) -> torch.Tensor:
	"""Integrate residual pyramid -> (C, D, H, W)."""
	v = src[-1]
	for d in reversed(src[:-1]):
		up = F.interpolate(v.unsqueeze(0), scale_factor=2.0, mode='trilinear', align_corners=True).squeeze(0)
		up = up[:, :int(d.shape[1]), :int(d.shape[2]), :int(d.shape[3])]
		v = up + d
	return v


def _mesh_coarse_from_state_dict(st: dict) -> torch.Tensor:
	"""Reconstruct mesh from checkpoint. Returns (3, D, Hm, Wm) in fullres coords."""
	keys = sorted(k for k in st.keys() if k.startswith("mesh_ms.") and k.split(".")[-1].isdigit())
	if not keys:
		raise ValueError("checkpoint missing mesh_ms.* tensors")
	idx_keys = sorted(((int(k.split(".")[-1]), k) for k in keys), key=lambda t: t[0])
	pyr = [st[k].detach() for _i, k in idx_keys]
	return _integrate_pyramid_3d(pyr)


def _write_tifxyz(*, out_dir: Path, x: np.ndarray, y: np.ndarray, z: np.ndarray,
				  scale: float, model_source: Path | None = None,
				  copy_model: bool = False, fit_config: dict | None = None) -> None:
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
			[float(np.nanmin(xf)), float(np.nanmin(yf)), float(np.nanmin(zf))],
			[float(np.nanmax(xf)), float(np.nanmax(yf)), float(np.nanmax(zf))],
		],
	}
	if model_source is not None:
		meta["model_source"] = str(model_source)
	if fit_config is not None:
		meta["fit_config"] = fit_config
	(out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
	tifffile.imwrite(str(out_dir / "x.tif"), xf, compression="lzw")
	tifffile.imwrite(str(out_dir / "y.tif"), yf, compression="lzw")
	tifffile.imwrite(str(out_dir / "z.tif"), zf, compression="lzw")

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

	# Reconstruct mesh (3, D, Hm, Wm) — pyramid stores full xyz positions
	mesh = _mesh_coarse_from_state_dict(st)

	_, D, Hm, Wm = (int(v) for v in mesh.shape)
	mesh_np = mesh.detach().cpu().numpy()  # (3, D, Hm, Wm)

	mesh_step = 16
	if model_params is not None:
		mesh_step = int(model_params.get("mesh_step", 16))
	xy_step_fullres = float(mesh_step)
	meta_scale = 1.0 / xy_step_fullres

	out_base = Path(cfg.output)
	out_base.mkdir(parents=True, exist_ok=True)

	BORDER_W = 2

	print(f"[fit2tifxyz] exporting D={D} Hm={Hm} Wm={Wm}, mesh already in fullres coords")

	if cfg.single_segment:
		# Combine all depth layers horizontally
		total_w = Wm * D + max(0, D - 1) * BORDER_W
		x_all = np.full((Hm, total_w), -1.0, dtype=np.float32)
		y_all = np.full((Hm, total_w), -1.0, dtype=np.float32)
		z_all = np.full((Hm, total_w), -1.0, dtype=np.float32)

		col = 0
		for d in range(D):
			x_all[:, col:col + Wm] = mesh_np[0, d]  # (Hm, Wm)
			y_all[:, col:col + Wm] = mesh_np[1, d]
			z_all[:, col:col + Wm] = mesh_np[2, d]
			col += Wm + BORDER_W

		seg_name = cfg.output_name if cfg.output_name else f"{cfg.prefix}.tifxyz"
		out_dir = out_base / seg_name
		_write_tifxyz(out_dir=out_dir, x=x_all, y=y_all, z=z_all, scale=meta_scale,
					  model_source=Path(cfg.input), copy_model=cfg.copy_model, fit_config=fit_config)
		if model_params is not None:
			(out_dir / "model_params.json").write_text(json.dumps(model_params, indent=2) + "\n", encoding="utf-8")
	else:
		# One tifxyz per depth layer (winding)
		for d in range(D):
			x = mesh_np[0, d]  # (Hm, Wm) already in fullres
			y = mesh_np[1, d]
			z = mesh_np[2, d]
			out_dir = out_base / f"{cfg.prefix}{d:04d}.tifxyz"
			_write_tifxyz(out_dir=out_dir, x=x, y=y, z=z, scale=meta_scale,
						  model_source=Path(cfg.input), copy_model=cfg.copy_model, fit_config=fit_config)
			if model_params is not None:
				(out_dir / "model_params.json").write_text(json.dumps(model_params, indent=2) + "\n", encoding="utf-8")

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
