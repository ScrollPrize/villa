from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch

import fit_data


@dataclass(frozen=True)
class DataConfig:
	input: str
	device: str
	downscale: float
	crop: tuple[int, int, int, int, int, int] | None  # (x, y, z, w, h, d) fullres
	bbox: tuple[int, int, int, int, int] | None        # (cx, cy, cz, w, h) fullres seed
	z_size: int | None                                  # z extent in fullres voxels
	cuda_gridsample: bool                               # use custom CUDA uint8 grid_sample kernel


def add_args(p: argparse.ArgumentParser) -> None:
	g = p.add_argument_group("data")
	g.add_argument("--input", default=None)
	g.add_argument("--device", default="cuda")
	g.add_argument("--downscale", type=float, default=4.0)
	g.add_argument("--crop", type=int, nargs=6, default=None,
		metavar=("X", "Y", "Z", "W", "H", "D"),
		help="3D volume crop in fullres voxels: x y z w h d")
	g.add_argument("--bbox", type=int, nargs=5, default=None,
		metavar=("CX", "CY", "CZ", "W", "H"),
		help="Seed bbox: center + XY size in fullres voxels")
	g.add_argument("--z-size", type=int, default=None,
		help="Z extent in fullres voxels (used with --bbox)")
	g.add_argument("--cuda-gridsample", type=int, default=1,
		help="Use custom CUDA uint8 grid_sample kernel (1=yes, 0=fallback to PyTorch F.grid_sample)")


def from_args(args: argparse.Namespace) -> DataConfig:
	if args.input in (None, ""):
		raise ValueError("missing --input (can be provided via JSON config args)")
	crop = None
	if args.crop is not None:
		crop = tuple(int(v) for v in args.crop)
		if len(crop) != 6:
			raise ValueError("--crop requires exactly 6 values: x y z w h d")
	bbox = None
	if getattr(args, "bbox", None) is not None:
		bbox = tuple(int(v) for v in args.bbox)
		if len(bbox) != 5:
			raise ValueError("--bbox requires exactly 5 values: cx cy cz w h")
	z_size = None if getattr(args, "z_size", None) is None else int(args.z_size)
	return DataConfig(
		input=str(args.input),
		device=str(args.device),
		downscale=float(args.downscale),
		crop=crop,
		bbox=bbox,
		z_size=z_size,
		cuda_gridsample=bool(int(getattr(args, "cuda_gridsample", 1))),
	)


def load_fit_data(cfg: DataConfig) -> fit_data.FitData3D:
	return fit_data.load_3d(
		path=cfg.input,
		device=torch.device(cfg.device),
		downscale=cfg.downscale,
		crop=cfg.crop,
		cuda_gridsample=cfg.cuda_gridsample,
	)
