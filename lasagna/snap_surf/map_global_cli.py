from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .map_global import optimize_fixture


def _build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Global rectangular snap-surf map optimizer")
	sub = parser.add_subparsers(dest="cmd", required=True)
	opt = sub.add_parser("optimize-fixture", help="Optimize a global rectangular map from a saved fixture")
	opt.add_argument("fixture_dir", help="Fixture directory containing fixture.json")
	opt.add_argument("config", help="Global optimizer JSON config")
	opt.add_argument("--out", required=True, help="Output directory")
	opt.add_argument("--device", default="auto", help="Torch device; default auto uses cuda when available, else cpu")
	return parser


def _resolve_device(raw: str) -> torch.device:
	if str(raw).lower() == "auto":
		return torch.device("cuda" if torch.cuda.is_available() else "cpu")
	return torch.device(str(raw))


def main(argv: list[str] | None = None) -> int:
	parser = _build_parser()
	args = parser.parse_args(argv)
	if args.cmd == "optimize-fixture":
		device = _resolve_device(str(args.device))
		print(f"[map_global_cli] device={device}", flush=True)
		metrics = optimize_fixture(
			args.fixture_dir,
			args.config,
			out_dir=Path(args.out),
			device=device,
		)
		print(
			"[map_global_cli] "
			f"common_vertices={metrics['common_vertices']} "
			f"model_x_max_abs_delta={metrics['model_x_max_abs_delta']:.9g} "
			f"model_y_max_abs_delta={metrics['model_y_max_abs_delta']:.9g}",
			flush=True,
		)
		return 0
	parser.error(f"unknown command: {args.cmd}")
	return 2


if __name__ == "__main__":
	raise SystemExit(main())
