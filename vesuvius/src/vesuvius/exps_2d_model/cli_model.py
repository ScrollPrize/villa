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
	init_size_frac_h: float | None
	init_size_frac_v: float | None
	model_input: str | None
	model_output: str | None


def add_args(p: argparse.ArgumentParser) -> None:
	g = p.add_argument_group("model")
	g.add_argument("--mesh-step", type=int, default=16)
	g.add_argument("--winding-step", type=int, default=16)
	g.add_argument("--subsample-mesh", type=int, default=4)
	g.add_argument("--subsample-winding", type=int, default=4)
	g.add_argument("--init-size-frac", type=float, default=2.0)
	g.add_argument("--init-size-frac-h", type=float, default=None)
	g.add_argument("--init-size-frac-v", type=float, default=None)
	g.add_argument("--model-input", default=None)
	g.add_argument("--model-output", default=None)


def from_args(args: argparse.Namespace) -> ModelConfig:
	return ModelConfig(
		mesh_step_px=int(args.mesh_step),
		winding_step_px=int(args.winding_step),
		subsample_mesh=int(args.subsample_mesh),
		subsample_winding=int(args.subsample_winding),
		init_size_frac=float(args.init_size_frac),
		init_size_frac_h=None if args.init_size_frac_h is None else float(args.init_size_frac_h),
		init_size_frac_v=None if args.init_size_frac_v is None else float(args.init_size_frac_v),
		model_input=None if args.model_input is None else str(args.model_input),
		model_output=None if args.model_output is None else str(args.model_output),
	)
