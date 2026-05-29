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
	init_mode: str           # "cylinder_seed" or "shell-dir-crop"
	z_center: float          # scroll height center (fullres)
	pyramid_d: bool
	model_input: str | None
	model_output: str | None


def add_args(p: argparse.ArgumentParser) -> None:
	g = p.add_argument_group("model")
	g.add_argument("--mesh-step", type=int, default=100)
	g.add_argument("--winding-step", type=int, default=25)
	g.add_argument("--subsample-mesh", type=int, default=4)
	g.add_argument("--subsample-winding", type=int, default=4)
	g.add_argument("--depth", type=int, default=None,
		help="Number of windings / model depth layers")
	g.add_argument("--windings", type=int, default=None,
		help="Alias for --depth")
	g.add_argument("--mesh-h", type=int, default=32)
	g.add_argument("--mesh-w", type=int, default=32)
	g.add_argument("--init-mode", default="cylinder_seed", choices=["cylinder_seed", "shell-dir-crop"])
	g.add_argument("--z-center", type=float, default=0.0)
	g.add_argument("--pyramid-d", action=argparse.BooleanOptionalAction, default=True)
	g.add_argument("--model-input", default=None)
	g.add_argument("--model-output", default=None)


def _resolved_depth(args: argparse.Namespace) -> int:
	depth_raw = getattr(args, "depth", None)
	windings_raw = getattr(args, "windings", None)
	depth = None if depth_raw is None else int(depth_raw)
	windings = None if windings_raw is None else int(windings_raw)
	if depth is not None and windings is not None and depth != windings:
		raise ValueError("args.depth and args.windings must match")
	resolved = depth if depth is not None else windings
	if resolved is None:
		resolved = 3
	resolved = max(1, int(resolved))
	setattr(args, "depth", resolved)
	return resolved


def from_args(args: argparse.Namespace) -> ModelConfig:
	depth = _resolved_depth(args)
	return ModelConfig(
		mesh_step=int(args.mesh_step),
		winding_step=int(args.winding_step),
		subsample_mesh=int(args.subsample_mesh),
		subsample_winding=int(args.subsample_winding),
		depth=depth,
		mesh_h=max(2, int(args.mesh_h)),
		mesh_w=max(2, int(args.mesh_w)),
		init_mode=str(args.init_mode),
		z_center=float(args.z_center),
		pyramid_d=bool(args.pyramid_d),
		model_input=None if args.model_input is None else str(args.model_input),
		model_output=None if args.model_output is None else str(args.model_output),
	)
