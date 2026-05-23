from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from .map_fixture_io import (
	_write_json,
	compare_map_tensors,
	export_map_fixture,
	load_map_fixture,
	map_tensors_from_state,
	snap_surf_config_from_fixture,
)
from .map_growth import _run_map_init_for_surface
from .state import _SurfaceState


def _build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Snap-surf map fixture utilities")
	sub = parser.add_subparsers(dest="cmd", required=True)
	compare = sub.add_parser("compare-reference", help="Rerun map-init and compare against a saved reference fixture")
	compare.add_argument("fixture_dir", help="Fixture directory containing fixture.json")
	compare.add_argument("--out", required=True, help="Output directory for rerun map and metrics")
	compare.add_argument("--device", default="cpu", help="Torch device for the rerun")
	compare.add_argument("--objs", action="store_true", default=False, help="Also write the rerun OBJ visualization set")
	return parser


def _seed_xyz(metadata: dict[str, Any]) -> tuple[float, float, float]:
	raw = metadata.get("seed_xyz")
	if not isinstance(raw, (list, tuple)) or len(raw) != 3:
		raise ValueError("fixture metadata is missing seed_xyz")
	return float(raw[0]), float(raw[1]), float(raw[2])


def _compare_reference(args: argparse.Namespace) -> int:
	device = torch.device(str(args.device))
	fixture = load_map_fixture(args.fixture_dir, device=device)
	cfg = snap_surf_config_from_fixture(fixture.metadata)
	state = _SurfaceState()
	state.ensure(
		model_shape=tuple(int(v) for v in fixture.model_xyz.shape[:3]),
		ext_shape=tuple(int(v) for v in fixture.ext_xyz.shape[:2]),
		device=device,
		dtype=fixture.model_xyz.dtype,
	)
	stats = _run_map_init_for_surface(
		state,
		model_xyz=fixture.model_xyz,
		model_valid=fixture.model_valid,
		model_normals=fixture.model_normals,
		ext_xyz=fixture.ext_xyz,
		ext_valid=fixture.ext_valid,
		ext_normals=fixture.ext_normals,
		ext_quad_valid=fixture.ext_quad_valid,
		cfg=cfg,
		seed_xyz=_seed_xyz(fixture.metadata),
		surface_index=int(fixture.metadata.get("surface_index", 0)),
		surface_count=int(fixture.metadata.get("surface_count", 1)),
	)
	out = Path(args.out)
	out.mkdir(parents=True, exist_ok=True)
	export_map_fixture(
		out,
		cfg=cfg,
		state=state,
		model_xyz=fixture.model_xyz,
		model_valid=fixture.model_valid,
		model_normals=fixture.model_normals,
		ext_xyz=fixture.ext_xyz,
		ext_valid=fixture.ext_valid,
		ext_normals=fixture.ext_normals,
		ext_quad_valid=fixture.ext_quad_valid,
		seed_xyz=_seed_xyz(fixture.metadata),
		surface_index=int(fixture.metadata.get("surface_index", 0)),
		surface_count=int(fixture.metadata.get("surface_count", 1)),
		step=fixture.metadata.get("step"),
		stats=stats,
		export_objs=bool(args.objs),
		write_geometry=False,
	)
	rerun_uv, rerun_active_quad, rerun_blocked_quad, _active_vertex = map_tensors_from_state(state)
	metrics = compare_map_tensors(
		reference_uv=fixture.reference_uv,
		reference_active_quad=fixture.reference_active_quad,
		reference_blocked_quad=fixture.reference_blocked_quad,
		rerun_uv=rerun_uv,
		rerun_active_quad=rerun_active_quad,
		rerun_blocked_quad=rerun_blocked_quad,
	)
	metrics["fixture_dir"] = str(Path(args.fixture_dir))
	metrics["out_dir"] = str(out)
	metrics["rerun_stats"] = stats
	_write_json(out / "metrics.json", metrics)
	print(
		"[map_fixture_cli] "
		f"active_diff={metrics['active_quad_diff']} "
		f"blocked_diff={metrics['blocked_quad_diff']} "
		f"model_x_max_abs_delta={metrics['model_x_max_abs_delta']:.9g} "
		f"model_y_max_abs_delta={metrics['model_y_max_abs_delta']:.9g}",
		flush=True,
	)
	return 0


def main(argv: list[str] | None = None) -> int:
	parser = _build_parser()
	args = parser.parse_args(argv)
	if args.cmd == "compare-reference":
		return _compare_reference(args)
	parser.error(f"unknown command: {args.cmd}")
	return 2


if __name__ == "__main__":
	raise SystemExit(main())
