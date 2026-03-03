from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
	mesh_step: int           # fullres voxels
	winding_step: int        # fullres voxels
	subsample_mesh: int
	subsample_winding: int
	depth: int               # number of windings
	mesh_h: int              # mesh grid height
	mesh_w: int              # mesh grid width
	arc_cx: float            # arc center x (fullres)
	arc_cy: float            # arc center y (fullres)
	arc_radius: float        # arc radius (fullres)
	arc_angle0: float        # start angle (radians)
	arc_angle1: float        # end angle (radians)
	z_center: float          # scroll height center (fullres)
	pyramid_d: bool
	model_input: str | None
	model_output: str | None


def add_args(p: argparse.ArgumentParser) -> None:
	g = p.add_argument_group("model")
	g.add_argument("--mesh-step", type=int, default=16)
	g.add_argument("--winding-step", type=int, default=25)
	g.add_argument("--subsample-mesh", type=int, default=4)
	g.add_argument("--subsample-winding", type=int, default=4)
	g.add_argument("--depth", type=int, default=3)
	g.add_argument("--mesh-h", type=int, default=32)
	g.add_argument("--mesh-w", type=int, default=32)
	g.add_argument("--arc-cx", type=float, default=0.0)
	g.add_argument("--arc-cy", type=float, default=0.0)
	g.add_argument("--arc-radius", type=float, default=1000.0)
	g.add_argument("--arc-angle-start", type=float, default=-0.5)
	g.add_argument("--arc-angle-end", type=float, default=0.5)
	g.add_argument("--z-center", type=float, default=0.0)
	g.add_argument("--pyramid-d", action="store_true", default=False)
	g.add_argument("--model-input", default=None)
	g.add_argument("--model-output", default=None)


def from_args(args: argparse.Namespace) -> ModelConfig:
	return ModelConfig(
		mesh_step=int(args.mesh_step),
		winding_step=int(args.winding_step),
		subsample_mesh=int(args.subsample_mesh),
		subsample_winding=int(args.subsample_winding),
		depth=max(1, int(args.depth)),
		mesh_h=max(2, int(args.mesh_h)),
		mesh_w=max(2, int(args.mesh_w)),
		arc_cx=float(args.arc_cx),
		arc_cy=float(args.arc_cy),
		arc_radius=float(args.arc_radius),
		arc_angle0=float(args.arc_angle_start),
		arc_angle1=float(args.arc_angle_end),
		z_center=float(args.z_center),
		pyramid_d=bool(args.pyramid_d),
		model_input=None if args.model_input is None else str(args.model_input),
		model_output=None if args.model_output is None else str(args.model_output),
	)
