from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch

import fit_data


def _load_model_params_from_checkpoint(path: str) -> dict | None:
	st = torch.load(path, map_location="cpu")
	if not isinstance(st, dict):
		return None
	mp = st.get("_model_params_", None)
	if not isinstance(mp, dict):
		return None
	return mp


@dataclass(frozen=True)
class DataConfig:
	input: str
	unet_checkpoint: str | None
	unet_layer: int | None
	unet_z: int | None
	z_step: int
	unet_tile_size: int
	unet_overlap: int
	unet_border: int
	unet_group: str | None
	device: str
	downscale: float
	crop: tuple[int, int, int, int] | None
	grad_mag_blur_sigma: float
	dir_blur_sigma: float


def add_args(p: argparse.ArgumentParser) -> None:
	g = p.add_argument_group("data")
	g.add_argument("--input", default=None)
	g.add_argument("--unet-checkpoint", default=None)
	g.add_argument("--unet-layer", type=int, default=None)
	g.add_argument("--unet-z", type=int, default=None)
	g.add_argument("--z-step", type=int, default=1)
	g.add_argument("--unet-tile-size", type=int, default=2048)
	g.add_argument("--unet-overlap", type=int, default=128)
	g.add_argument("--unet-border", type=int, default=32)
	g.add_argument("--unet-group", default=None)
	g.add_argument("--device", default="cuda")
	g.add_argument("--downscale", type=float, default=4.0)
	g.add_argument("--crop", type=int, nargs=4, default=None)
	g.add_argument("--grad-mag-blur-sigma", type=float, default=0.0)
	g.add_argument("--dir-blur-sigma", type=float, default=0.0)


def from_args(args: argparse.Namespace) -> DataConfig:
	if args.input in (None, ""):
		raise ValueError("missing --input (can be provided via JSON config args)")
	crop = tuple(int(v) for v in args.crop) if args.crop is not None else None
	unet_z = None if args.unet_z is None else int(args.unet_z)
	z_step = max(1, int(args.z_step))
	if (crop is None or unet_z is None or int(args.z_step) == 1) and args.unet_checkpoint is not None:
		mp = _load_model_params_from_checkpoint(str(args.unet_checkpoint))
		if mp is not None:
			c6 = mp.get("crop_xyzwhd", None)
			if isinstance(c6, (list, tuple)) and len(c6) == 6:
				x, y, w, h, z0, d = (int(v) for v in c6)
				if crop is None:
					crop = (x, y, w, h)
				if unet_z is None:
					unet_z = int(z0)
				if args.z_size is None:
					args.z_size = int(d)
			if int(args.z_step) == 1 and ("z_step_vx" in mp):
				z_step = max(1, int(mp["z_step_vx"]))
	return DataConfig(
		input=str(args.input),
		unet_checkpoint=None if args.unet_checkpoint in (None, "") else str(args.unet_checkpoint),
		unet_layer=None if args.unet_layer is None else int(args.unet_layer),
		unet_z=unet_z,
		z_step=z_step,
		unet_tile_size=int(args.unet_tile_size),
		unet_overlap=int(args.unet_overlap),
		unet_border=int(args.unet_border),
		unet_group=None if args.unet_group in (None, "") else str(args.unet_group),
		device=str(args.device),
		downscale=float(args.downscale),
		crop=crop,
		grad_mag_blur_sigma=float(getattr(args, "grad_mag_blur_sigma", 0.0) or 0.0),
		dir_blur_sigma=float(getattr(args, "dir_blur_sigma", 0.0) or 0.0),
	)


def load_fit_data(cfg: DataConfig, *, z_size: int = 1, out_dir_base: str | None = None) -> fit_data.FitData:
	return fit_data.load(
		path=cfg.input,
		device=torch.device(cfg.device),
		downscale=cfg.downscale,
		crop=cfg.crop,
		unet_checkpoint=cfg.unet_checkpoint,
		unet_layer=cfg.unet_layer,
		unet_z=cfg.unet_z,
		z_size=max(1, int(z_size)),
		z_step=int(cfg.z_step),
		unet_tile_size=cfg.unet_tile_size,
		unet_overlap=cfg.unet_overlap,
		unet_border=cfg.unet_border,
		unet_group=cfg.unet_group,
		unet_out_dir_base=out_dir_base,
		grad_mag_blur_sigma=float(cfg.grad_mag_blur_sigma),
		dir_blur_sigma=float(cfg.dir_blur_sigma),
	)
