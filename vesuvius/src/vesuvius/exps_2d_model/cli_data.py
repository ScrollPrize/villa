from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch

import fit_data


@dataclass(frozen=True)
class DataConfig:
	input: str
	unet_checkpoint: str | None
	unet_layer: int | None
	device: str
	downscale: float
	crop: tuple[int, int, int, int] | None


def add_args(p: argparse.ArgumentParser) -> None:
	g = p.add_argument_group("data")
	g.add_argument("--input", required=True)
	g.add_argument("--unet-checkpoint", default=None)
	g.add_argument("--unet-layer", type=int, default=None)
	g.add_argument("--device", default="cuda")
	g.add_argument("--downscale", type=float, default=4.0)
	g.add_argument("--crop", type=int, nargs=4, default=None)


def from_args(args: argparse.Namespace) -> DataConfig:
	return DataConfig(
		input=str(args.input),
		unet_checkpoint=None if args.unet_checkpoint in (None, "") else str(args.unet_checkpoint),
		unet_layer=None if args.unet_layer is None else int(args.unet_layer),
		device=str(args.device),
		downscale=float(args.downscale),
		crop=tuple(int(v) for v in args.crop) if args.crop is not None else None,
	)


def load_fit_data(cfg: DataConfig) -> fit_data.FitData:
	return fit_data.load(
		path=cfg.input,
		device=torch.device(cfg.device),
		downscale=cfg.downscale,
		crop=cfg.crop,
	)
