from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class OptConfig:
	stages_json: str
	device: str
	snapshot_interval: int


def add_args(p: argparse.ArgumentParser) -> None:
	g = p.add_argument_group("opt")
	g.add_argument("--stages-json", default="default_stages.json")
	g.add_argument("--snapshot-interval", type=int, default=1000)
	# device arg is provided by data part for now


def from_args(args: argparse.Namespace) -> OptConfig:
	return OptConfig(
		stages_json=str(args.stages_json),
		device=str(args.device),
		snapshot_interval=int(args.snapshot_interval),
	)
