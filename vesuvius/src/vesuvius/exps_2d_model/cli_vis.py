from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class VisConfig:
	out_dir: str | None
	scale: int


def add_args(p: argparse.ArgumentParser) -> None:
	g = p.add_argument_group("vis")
	g.add_argument("--out-dir", default=None, help="Vis/debug output dir (omit to skip all debug output)")
	g.add_argument("--vis-scale", type=int, default=4)


def from_args(args: argparse.Namespace) -> VisConfig:
	od = args.out_dir
	return VisConfig(
		out_dir=None if od is None or str(od) == "" else str(od),
		scale=max(1, int(args.vis_scale)),
	)
