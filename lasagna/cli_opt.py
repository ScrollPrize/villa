from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class OptConfig:
	device: str
	snapshot_interval: int
	corr_mode: str              # "legacy", "snap", "winding"
	normal_mask_zero: bool


def add_args(p: argparse.ArgumentParser) -> None:
	g = p.add_argument_group("opt")
	g.add_argument("--snapshot-interval", type=int, default=1000)
	g.add_argument("--corr-mode", type=str, default="winding",
		choices=["legacy", "snap", "winding"],
		help="Corr point mode: legacy, snap, or winding (default)")
	g.add_argument("--corr-snap", type=int, default=None,
		help=argparse.SUPPRESS)  # backward compat: 0=legacy, 1=snap
	g.add_argument("--normal-mask-zero", type=int, default=0,
		help="Mask out zero-normal voxels (nx=ny=0) from normal loss. Use with label-derived data.")
	# device arg is provided by data part for now


def from_args(args: argparse.Namespace) -> OptConfig:
	corr_mode = getattr(args, "corr_mode", "winding")
	corr_snap_val = getattr(args, "corr_snap", None)
	if corr_snap_val is not None:
		corr_mode = "snap" if int(corr_snap_val) else "legacy"
	return OptConfig(
		device=str(args.device),
		snapshot_interval=int(args.snapshot_interval),
		corr_mode=corr_mode,
		normal_mask_zero=bool(int(getattr(args, "normal_mask_zero", 0))),
	)
