import argparse
import copy
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch

import cli_data
import cli_json
import cli_model
import cli_opt
import fit_data
import model
import optimizer


def _build_parser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(
		prog="fit.py",
		description="3D fit entrypoint",
	)
	cli_data.add_args(p)
	cli_model.add_args(p)
	cli_opt.add_args(p)
	p.add_argument("--out-dir", default=None, help="Output directory for snapshots and debug")
	p.add_argument("--progress", action="store_true", default=False,
		help="Print machine-readable PROGRESS lines to stdout")
	return p


def main(argv: list[str] | None = None) -> int:
	if argv is None:
		argv = sys.argv[1:]

	parser = _build_parser()
	cfg_paths, argv_rest = cli_json.split_cfg_argv(argv)
	cfg_paths = [str(x) for x in cfg_paths]
	cfg = cli_json.merge_cfgs(cfg_paths)
	fit_config = copy.deepcopy(cfg)
	cli_json.apply_defaults_from_cfg_args(parser, cfg)
	args = parser.parse_args(argv_rest or [])

	data_cfg = cli_data.from_args(args)
	model_cfg = cli_model.from_args(args)
	opt_cfg = cli_opt.from_args(args)
	progress_enabled = bool(args.progress)
	_out_dir = args.out_dir

	print("data:", data_cfg)
	print("model:", model_cfg)
	print("opt:", opt_cfg)

	device = torch.device(data_cfg.device)

	# Probe preprocessed zarr for z_step_eff
	prep_params = fit_data.get_preprocessed_params(str(data_cfg.input))
	z_step_eff = 1
	if prep_params is not None:
		z_step_eff = int(prep_params["z_step_eff"])
		print(f"[fit] zarr z_step_eff={z_step_eff}", flush=True)

	# Load 3D data
	data = cli_data.load_fit_data(data_cfg)

	# Determine volume extent from data
	Z, Y, X = data.size
	volume_extent = (
		data.origin_fullres[0],
		data.origin_fullres[1],
		data.origin_fullres[2],
		data.origin_fullres[0] + (X - 1) * data.spacing[0],
		data.origin_fullres[1] + (Y - 1) * data.spacing[1],
		data.origin_fullres[2] + (Z - 1) * data.spacing[2],
	)

	# Create or load model
	model_params_in: dict | None = None
	if model_cfg.model_input is not None:
		st_in = torch.load(model_cfg.model_input, map_location="cpu")
		if isinstance(st_in, dict) and isinstance(st_in.get("_model_params_", None), dict):
			model_params_in = st_in["_model_params_"]

	if model_cfg.model_input is not None and model_params_in is not None:
		# Load checkpoint — derive model shape from state dict
		st_cpu = torch.load(model_cfg.model_input, map_location="cpu")
		mesh0 = st_cpu.get("mesh_ms.0", None)
		if mesh0 is None or not hasattr(mesh0, "shape"):
			raise ValueError("model_input missing mesh_ms.0")
		_c, D, H, W = (int(v) for v in mesh0.shape)

		mdl = model.Model3D(
			device=device,
			depth=D,
			mesh_h=H,
			mesh_w=W,
			mesh_step=int(model_params_in.get("mesh_step", model_cfg.mesh_step)),
			winding_step=int(model_params_in.get("winding_step", model_cfg.winding_step)),
			subsample_mesh=int(model_params_in.get("subsample_mesh", model_cfg.subsample_mesh)),
			subsample_winding=int(model_params_in.get("subsample_winding", model_cfg.subsample_winding)),
			scaledown=float(model_params_in.get("scaledown", data_cfg.downscale)),
			z_step_eff=int(model_params_in.get("z_step_eff", z_step_eff)),
			z_center=float(model_cfg.z_center),
			arc_cx=float(model_cfg.arc_cx),
			arc_cy=float(model_cfg.arc_cy),
			arc_radius=float(model_cfg.arc_radius),
			arc_angle0=float(model_cfg.arc_angle0),
			arc_angle1=float(model_cfg.arc_angle1),
			volume_extent=volume_extent,
		)

		st = torch.load(model_cfg.model_input, map_location=device)
		miss, unexp = mdl.load_state_dict_compat(st, strict=False)
		if unexp:
			print("state_dict: unexpected keys:", sorted(unexp))
		if miss:
			print("state_dict: missing keys:", sorted(miss))
	else:
		# Create new model from CLI args
		mdl = model.Model3D(
			device=device,
			depth=model_cfg.depth,
			mesh_h=model_cfg.mesh_h,
			mesh_w=model_cfg.mesh_w,
			mesh_step=model_cfg.mesh_step,
			winding_step=model_cfg.winding_step,
			subsample_mesh=model_cfg.subsample_mesh,
			subsample_winding=model_cfg.subsample_winding,
			scaledown=data_cfg.downscale,
			z_step_eff=z_step_eff,
			z_center=model_cfg.z_center,
			arc_cx=model_cfg.arc_cx,
			arc_cy=model_cfg.arc_cy,
			arc_radius=model_cfg.arc_radius,
			arc_angle0=model_cfg.arc_angle0,
			arc_angle1=model_cfg.arc_angle1,
			volume_extent=volume_extent,
		)

	print(f"Model3D: depth={mdl.depth} mesh_h={mdl.mesh_h} mesh_w={mdl.mesh_w} "
		  f"arc_enabled={mdl.arc_enabled}")

	# Print initial mesh stats
	with torch.no_grad():
		xyz0 = mdl._grid_xyz()
		mn = xyz0.amin(dim=(0, 1, 2)).cpu().numpy().tolist()
		mx = xyz0.amax(dim=(0, 1, 2)).cpu().numpy().tolist()
		mean = xyz0.mean(dim=(0, 1, 2)).cpu().numpy().tolist()
		print(f"initial mesh: mean={[round(v, 1) for v in mean]} "
			  f"min={[round(v, 1) for v in mn]} max={[round(v, 1) for v in mx]}")

	# Parse stages
	stages = optimizer.load_stages_cfg(cfg)

	def _save_model(path: str) -> None:
		if mdl.arc_enabled:
			mdl.bake_arc_into_mesh()
		st = dict(mdl.state_dict())
		st["_model_params_"] = asdict(mdl.params)
		st["_fit_config_"] = fit_config
		torch.save(st, path)

	def _snapshot(*, stage: str, step: int, loss: float, data, res=None) -> None:
		if _out_dir is not None:
			out = Path(_out_dir)
			out.mkdir(parents=True, exist_ok=True)
			snaps = out / "model_snapshots"
			snaps.mkdir(parents=True, exist_ok=True)
			_save_model(str(snaps / f"model_{stage}_{step:06d}.pt"))

	def _progress(*, step: int, total: int, loss: float, **_kw: object) -> None:
		if progress_enabled:
			print(f"PROGRESS {step} {total} {loss:.6f}", flush=True)

	# Run optimization
	optimizer.optimize(
		model=mdl,
		data=data,
		stages=stages,
		snapshot_interval=opt_cfg.snapshot_interval,
		snapshot_fn=_snapshot,
		progress_fn=_progress,
	)

	# Save final model
	if model_cfg.model_output is not None:
		_save_model(str(model_cfg.model_output))
		print(f"[fit] saved model to {model_cfg.model_output}")

	# Save snapshot
	if _out_dir is not None:
		out = Path(_out_dir)
		out.mkdir(parents=True, exist_ok=True)
		_save_model(str(out / "model_final.pt"))

	# Export tifxyz
	model_out = model_cfg.model_output
	if model_out is None and _out_dir is not None:
		model_out = str(Path(_out_dir) / "model_final.pt")
	if model_out is not None and _out_dir is not None:
		import fit2tifxyz
		export_dir = str(Path(_out_dir) / "tifxyz")
		fit2tifxyz.main(["--input", str(model_out), "--output", export_dir])

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
