from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class VisConfig:
	out_dir: str
	scale: int


def add_args(p: argparse.ArgumentParser) -> None:
	g = p.add_argument_group("vis")
	g.add_argument("--out-dir", default="res")
	g.add_argument("--vis-scale", type=int, default=4)


def from_args(args: argparse.Namespace) -> VisConfig:
	return VisConfig(
		out_dir=str(args.out_dir),
		scale=max(1, int(args.vis_scale)),
	)
