from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class OptConfig:
	device: str
	snapshot_interval: int
	corr_snap: bool


def add_args(p: argparse.ArgumentParser) -> None:
	g = p.add_argument_group("opt")
	g.add_argument("--snapshot-interval", type=int, default=1000)
	g.add_argument("--corr-snap", type=int, default=1,
		help="Corr point mode: 1=snap-to-surface (default), 0=legacy winding observation")
	# device arg is provided by data part for now


def from_args(args: argparse.Namespace) -> OptConfig:
	return OptConfig(
		device=str(args.device),
		snapshot_interval=int(args.snapshot_interval),
		corr_snap=bool(int(getattr(args, "corr_snap", 1))),
	)
