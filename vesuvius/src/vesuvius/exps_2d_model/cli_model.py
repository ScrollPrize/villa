from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
	mesh_step_px: int
	winding_step_px: int
	subsample_mesh: int
	subsample_winding: int
	init_size_frac: float


def add_args(p: argparse.ArgumentParser) -> None:
	g = p.add_argument_group("model")
	g.add_argument("--mesh-step", type=int, default=16)
	g.add_argument("--winding-step", type=int, default=16)
	g.add_argument("--subsample-mesh", type=int, default=4)
	g.add_argument("--subsample-winding", type=int, default=4)
	g.add_argument("--init-size-frac", type=float, default=2.0)


def from_args(args: argparse.Namespace) -> ModelConfig:
	return ModelConfig(
		mesh_step_px=int(args.mesh_step),
		winding_step_px=int(args.winding_step),
		subsample_mesh=int(args.subsample_mesh),
		subsample_winding=int(args.subsample_winding),
		init_size_frac=float(args.init_size_frac),
	)
