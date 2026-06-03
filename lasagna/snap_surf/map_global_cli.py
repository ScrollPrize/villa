from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .map_global import benchmark_fixture, optimize_fixture


def _build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Global rectangular snap-surf map optimizer")
	sub = parser.add_subparsers(dest="cmd", required=True)
	opt = sub.add_parser("optimize-fixture", help="Optimize a global rectangular map from a saved fixture")
	opt.add_argument("fixture_dir", help="Fixture directory containing fixture.json")
	opt.add_argument("config", help="Global optimizer JSON config")
	opt.add_argument("--out", required=True, help="Output directory")
	opt.add_argument("--device", default="auto", help="Torch device; default auto uses cuda when available, else cpu")
	bench = sub.add_parser("benchmark-fixture", help="Optimize a fixture and compare the final map against a reference")
	bench.add_argument("fixture_dir", help="Fixture directory containing fixture.json")
	bench.add_argument("config", help="Global optimizer JSON config")
	bench.add_argument("--out", required=True, help="Output directory")
	bench.add_argument("--device", default="auto", help="Torch device; default auto uses cuda when available, else cpu")
	bench.add_argument("--reference-dir", default=None, help="Reference fixture directory or map directory; defaults to fixture map")
	bench.add_argument("--max-model-abs-delta", type=float, default=1.0, help="Maximum allowed per-axis absolute UV delta")
	bench.add_argument("--max-model-l2-delta", type=float, default=1.0, help="Maximum allowed per-vertex L2 UV delta")
	bench.add_argument("--allow-mask-diff", action="store_true", help="Do not fail when active/blocked masks differ")
	bench.add_argument("--profile-components", action="store_true", help="Time all and single map objective components without optimizer steps")
	bench.add_argument("--profile-repeats", type=int, default=3, help="Component profiling repeats per component")
	bench.add_argument("--profile-stage", default=None, help="Map stage index or name to profile; defaults to first map_surf_ms stage")
	bench.add_argument("--profiler-trace", default=None, help="Optional profiler trace output path for component profiling")
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
	if args.cmd == "benchmark-fixture":
		device = _resolve_device(str(args.device))
		print(f"[map_global_cli] device={device}", flush=True)
		result = benchmark_fixture(
			args.fixture_dir,
			args.config,
			out_dir=Path(args.out),
			device=device,
			reference_dir=args.reference_dir,
			max_model_abs_delta=float(args.max_model_abs_delta),
			max_model_l2_delta=float(args.max_model_l2_delta),
			require_mask_equal=not bool(args.allow_mask_diff),
			profile_components=bool(args.profile_components),
			profile_repeats=int(args.profile_repeats),
			profile_stage=args.profile_stage,
			profiler_trace=args.profiler_trace,
		)
		deltas = result["map_deltas"]
		print(
			"[map_global_cli] "
			f"status={result['status']} "
			f"elapsed_s={result['elapsed_s']:.6g} "
			f"common_vertices={deltas['common_vertices']} "
			f"model_l2_max_delta={deltas['model_l2_max_delta']:.9g} "
			f"active_quad_diff={deltas['active_quad_diff']} "
			f"blocked_quad_diff={deltas['blocked_quad_diff']}",
			flush=True,
		)
		return 0 if bool(result["passed"]) else 1
	parser.error(f"unknown command: {args.cmd}")
	return 2


if __name__ == "__main__":
	raise SystemExit(main())
