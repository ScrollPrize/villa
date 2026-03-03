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


def add_args(p: argparse.ArgumentParser) -> None:
	g = p.add_argument_group("data")
	g.add_argument("--input", default=None)
	g.add_argument("--device", default="cuda")
	g.add_argument("--downscale", type=float, default=4.0)
	g.add_argument("--crop", type=int, nargs=6, default=None,
		metavar=("X", "Y", "Z", "W", "H", "D"),
		help="3D volume crop in fullres voxels: x y z w h d")


def from_args(args: argparse.Namespace) -> DataConfig:
	if args.input in (None, ""):
		raise ValueError("missing --input (can be provided via JSON config args)")
	crop = None
	if args.crop is not None:
		crop = tuple(int(v) for v in args.crop)
		if len(crop) != 6:
			raise ValueError("--crop requires exactly 6 values: x y z w h d")
	return DataConfig(
		input=str(args.input),
		device=str(args.device),
		downscale=float(args.downscale),
		crop=crop,
	)


def load_fit_data(cfg: DataConfig) -> fit_data.FitData3D:
	return fit_data.load_3d(
		path=cfg.input,
		device=torch.device(cfg.device),
		downscale=cfg.downscale,
		crop=cfg.crop,
	)
