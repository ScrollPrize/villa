import argparse
import copy
import dataclasses
import math
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


def _arc_params_from_bbox(
	bbox: tuple[int, int, int, int, int],
	z_size: int | None,
	volume_extent_fullres: tuple[int, int, int],
) -> dict:
	"""Derive arc params from bbox seed point.

	bbox: (cx, cy, cz, w, h) in fullres voxels — seed center + XY extent
	volume_extent_fullres: (X_total, Y_total, Z_total) in fullres voxels
	Returns dict with keys matching ModelConfig arc fields.
	"""
	seed_cx, seed_cy, seed_cz = float(bbox[0]), float(bbox[1]), float(bbox[2])
	seed_w, seed_h = float(bbox[3]), float(bbox[4])
	vol_x, vol_y, _vol_z = volume_extent_fullres

	# Volume center XY = estimate of scroll axis
	arc_cx = vol_x / 2.0
	arc_cy = vol_y / 2.0

	# Arc radius = distance from volume center to seed point
	dx = seed_cx - arc_cx
	dy = seed_cy - arc_cy
	arc_radius = math.sqrt(dx * dx + dy * dy)
	arc_radius = max(arc_radius, 100.0)

	# Seed angle
	seed_angle = math.atan2(dy, dx)

	# Angular half-extent to cover bbox
	half_angle = max(seed_w, seed_h) / (2.0 * arc_radius)
	half_angle = max(half_angle, 0.05)

	return {
		"arc_cx": arc_cx,
		"arc_cy": arc_cy,
		"arc_radius": arc_radius,
		"arc_angle0": seed_angle - half_angle,
		"arc_angle1": seed_angle + half_angle,
		"z_center": seed_cz,
	}


def _auto_crop(
	mesh_bbox: tuple[float, float, float, float, float, float],
	volume_extent_fullres: tuple[int, int, int],
	margin: float = 3.0,
) -> tuple[int, int, int, int, int, int]:
	"""Compute crop = margin × mesh extent, centered on mesh, clamped to volume.

	Returns (x0, y0, z0, w, h, d) in fullres voxels.
	"""
	x_min, y_min, z_min, x_max, y_max, z_max = mesh_bbox
	vol_x, vol_y, vol_z = volume_extent_fullres

	cx = (x_min + x_max) / 2.0
	cy = (y_min + y_max) / 2.0
	cz = (z_min + z_max) / 2.0

	ex = max((x_max - x_min) * margin / 2.0, 100.0)
	ey = max((y_max - y_min) * margin / 2.0, 100.0)
	ez = max((z_max - z_min) * margin / 2.0, 100.0)

	x0 = max(0, int(cx - ex))
	y0 = max(0, int(cy - ey))
	z0 = max(0, int(cz - ez))
	x1 = min(vol_x, int(cx + ex))
	y1 = min(vol_y, int(cy + ey))
	z1 = min(vol_z, int(cz + ez))

	return (x0, y0, z0, x1 - x0, y1 - y0, z1 - z0)


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

	# Probe preprocessed zarr for scaledown and volume extent
	prep_params = fit_data.get_preprocessed_params(str(data_cfg.input))
	scaledown = data_cfg.downscale
	volume_extent_fullres = None
	if prep_params is not None:
		scaledown = float(prep_params["scaledown"])
		volume_extent_fullres = prep_params.get("volume_extent_fullres")
		print(f"[fit] zarr scaledown={scaledown} volume_extent={volume_extent_fullres}", flush=True)

	# --- Arc init from bbox (new model only) ---
	is_new_model = model_cfg.model_input is None
	if is_new_model and data_cfg.bbox is not None and volume_extent_fullres is not None:
		arc = _arc_params_from_bbox(data_cfg.bbox, data_cfg.z_size, volume_extent_fullres)
		model_cfg = dataclasses.replace(model_cfg, **arc)
		print(f"[fit] arc from bbox: cx={arc['arc_cx']:.1f} cy={arc['arc_cy']:.1f} "
			  f"r={arc['arc_radius']:.1f} a0={arc['arc_angle0']:.3f} a1={arc['arc_angle1']:.3f} "
			  f"z={arc['z_center']:.1f}", flush=True)

	# --- Auto-size mesh from bbox (new model only) ---
	if is_new_model and data_cfg.bbox is not None:
		seed_w, seed_h = float(data_cfg.bbox[3]), float(data_cfg.bbox[4])

		# Radial extent → depth (number of windings)
		radial_extent = min(seed_w, seed_h)
		auto_depth = max(1, int(radial_extent / model_cfg.winding_step))

		# Z extent → mesh_h
		z_ext = float(data_cfg.z_size) if data_cfg.z_size is not None else (model_cfg.mesh_step * (model_cfg.mesh_h - 1))
		auto_mesh_h = max(2, int(z_ext / model_cfg.mesh_step) + 1)

		# Angular extent → mesh_w (match resolution to mesh_step)
		arc_length = model_cfg.arc_radius * (model_cfg.arc_angle1 - model_cfg.arc_angle0)
		auto_mesh_w = max(2, int(arc_length / model_cfg.mesh_step) + 1)

		model_cfg = dataclasses.replace(model_cfg, depth=auto_depth, mesh_h=auto_mesh_h, mesh_w=auto_mesh_w)
		print(f"[fit] auto-sized: depth={auto_depth} mesh_h={auto_mesh_h} mesh_w={auto_mesh_w}", flush=True)

	# --- Construct / load model (before data, so we can compute bbox) ---
	if is_new_model:
		mdl = model.Model3D(
			device=device,
			depth=model_cfg.depth,
			mesh_h=model_cfg.mesh_h,
			mesh_w=model_cfg.mesh_w,
			mesh_step=model_cfg.mesh_step,
			winding_step=model_cfg.winding_step,
			subsample_mesh=model_cfg.subsample_mesh,
			subsample_winding=model_cfg.subsample_winding,
			scaledown=scaledown,
			z_step_eff=int(round(scaledown)),
			z_center=model_cfg.z_center,
			arc_cx=model_cfg.arc_cx,
			arc_cy=model_cfg.arc_cy,
			arc_radius=model_cfg.arc_radius,
			arc_angle0=model_cfg.arc_angle0,
			arc_angle1=model_cfg.arc_angle1,
			volume_extent=None,
			pyramid_d=model_cfg.pyramid_d,
		)
	else:
		st = torch.load(model_cfg.model_input, map_location=device, weights_only=False)
		mdl = model.Model3D.from_checkpoint(st, device=device)

	print(f"Model3D: depth={mdl.depth} mesh_h={mdl.mesh_h} mesh_w={mdl.mesh_w} "
		  f"arc_enabled={mdl.arc_enabled}")

	# --- Data loading (with auto-crop, blur, and reload support) ---
	def _load_data() -> fit_data.FitData3D:
		with torch.no_grad():
			xyz = mdl._grid_xyz()  # (D, Hm, Wm, 3)
			mesh_bbox = (float(xyz[..., 0].min()), float(xyz[..., 1].min()), float(xyz[..., 2].min()),
						 float(xyz[..., 0].max()), float(xyz[..., 1].max()), float(xyz[..., 2].max()))
		print(f"[fit] mesh bbox: "
			  f"min=({mesh_bbox[0]:.0f},{mesh_bbox[1]:.0f},{mesh_bbox[2]:.0f}) "
			  f"max=({mesh_bbox[3]:.0f},{mesh_bbox[4]:.0f},{mesh_bbox[5]:.0f})", flush=True)
		if volume_extent_fullres is not None:
			auto_crop = _auto_crop(mesh_bbox, volume_extent_fullres, margin=3.0)
			print(f"[fit] auto-crop: x={auto_crop[0]} y={auto_crop[1]} z={auto_crop[2]} "
				  f"w={auto_crop[3]} h={auto_crop[4]} d={auto_crop[5]}", flush=True)
			cfg = dataclasses.replace(data_cfg, crop=auto_crop)
		else:
			cfg = data_cfg
		d = cli_data.load_fit_data(cfg)
		fit_data.blur_3d(d, sigma=2.0)
		print("[fit] blurred data sigma=2.0", flush=True)
		Z, Y, X = d.size
		volume_extent = (
			d.origin_fullres[0],
			d.origin_fullres[1],
			d.origin_fullres[2],
			d.origin_fullres[0] + (X - 1) * d.spacing[0],
			d.origin_fullres[1] + (Y - 1) * d.spacing[1],
			d.origin_fullres[2] + (Z - 1) * d.spacing[2],
		)
		mdl.params = dataclasses.replace(mdl.params, volume_extent=volume_extent)
		return d

	data = _load_data()

	# Print initial mesh stats
	with torch.no_grad():
		xyz = mdl._grid_xyz()
		mn = xyz.amin(dim=(0, 1, 2)).cpu().numpy().tolist()
		mx = xyz.amax(dim=(0, 1, 2)).cpu().numpy().tolist()
		mean = xyz.mean(dim=(0, 1, 2)).cpu().numpy().tolist()
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
		load_data_fn=_load_data,
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
