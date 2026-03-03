import argparse
import copy
import dataclasses
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F

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


def _compute_mesh_bbox_from_arc(
	*,
	arc_cx: float, arc_cy: float, arc_radius: float,
	arc_angle0: float, arc_angle1: float,
	z_center: float, mesh_step: int, mesh_h: int,
	depth: int, winding_step: int,
) -> tuple[float, float, float, float, float, float]:
	"""Compute (x_min, y_min, z_min, x_max, y_max, z_max) in fullres voxels."""
	# Z extent
	h_extent = mesh_step * (mesh_h - 1)
	z_min = z_center - h_extent / 2.0
	z_max = z_center + h_extent / 2.0

	# Radius range across depth layers
	r_min = arc_radius - (depth - 1) / 2.0 * winding_step
	r_max = arc_radius + (depth - 1) / 2.0 * winding_step

	# Find trig extrema over angle range
	cos_vals = [math.cos(arc_angle0), math.cos(arc_angle1)]
	sin_vals = [math.sin(arc_angle0), math.sin(arc_angle1)]
	for k in range(-4, 5):
		t = k * math.pi
		if arc_angle0 <= t <= arc_angle1:
			cos_vals.append(math.cos(t))
		t_pos = k * math.pi + math.pi / 2.0
		if arc_angle0 <= t_pos <= arc_angle1:
			sin_vals.append(math.sin(t_pos))
		t_neg = k * math.pi - math.pi / 2.0
		if arc_angle0 <= t_neg <= arc_angle1:
			sin_vals.append(math.sin(t_neg))

	cos_mn, cos_mx = min(cos_vals), max(cos_vals)
	sin_mn, sin_mx = min(sin_vals), max(sin_vals)

	x_candidates = [arc_cx + r * c for r in [r_min, r_max] for c in [cos_mn, cos_mx]]
	y_candidates = [arc_cy + r * s for r in [r_min, r_max] for s in [sin_mn, sin_mx]]

	return (min(x_candidates), min(y_candidates), z_min,
			max(x_candidates), max(y_candidates), z_max)


def _mesh_bbox_from_state_dict(state_dict: dict) -> tuple[float, float, float, float, float, float] | None:
	"""Extract mesh bbox from a saved model's state dict (arc already baked)."""
	ms_keys = sorted(k for k in state_dict if k.startswith("mesh_ms."))
	if not ms_keys:
		return None
	tensors = [state_dict[k] for k in ms_keys]
	# Integrate pyramid: coarsest to finest
	v = tensors[-1]
	for d in reversed(tensors[:-1]):
		up = F.interpolate(v.unsqueeze(0), scale_factor=2.0,
						   mode='trilinear', align_corners=True).squeeze(0)
		v = up[:, :d.shape[1], :d.shape[2], :d.shape[3]] + d
	# v is (3, D, H, W) in fullres coordinates
	return (float(v[0].min()), float(v[1].min()), float(v[2].min()),
			float(v[0].max()), float(v[1].max()), float(v[2].max()))


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

	# --- Auto-crop from mesh bbox ---
	if data_cfg.crop is None and volume_extent_fullres is not None:
		mesh_bbox = None
		if not is_new_model and model_cfg.model_input is not None:
			# Loaded model: peek state dict to get mesh bbox
			st_peek = torch.load(model_cfg.model_input, map_location="cpu", weights_only=False)
			mesh_bbox = _mesh_bbox_from_state_dict(st_peek)
			del st_peek
			if mesh_bbox is not None:
				print(f"[fit] mesh bbox from checkpoint: "
					  f"min=({mesh_bbox[0]:.0f},{mesh_bbox[1]:.0f},{mesh_bbox[2]:.0f}) "
					  f"max=({mesh_bbox[3]:.0f},{mesh_bbox[4]:.0f},{mesh_bbox[5]:.0f})", flush=True)
		elif is_new_model:
			mesh_bbox = _compute_mesh_bbox_from_arc(
				arc_cx=model_cfg.arc_cx, arc_cy=model_cfg.arc_cy,
				arc_radius=model_cfg.arc_radius,
				arc_angle0=model_cfg.arc_angle0, arc_angle1=model_cfg.arc_angle1,
				z_center=model_cfg.z_center,
				mesh_step=model_cfg.mesh_step, mesh_h=model_cfg.mesh_h,
				depth=model_cfg.depth, winding_step=model_cfg.winding_step,
			)
			print(f"[fit] mesh bbox from arc: "
				  f"min=({mesh_bbox[0]:.0f},{mesh_bbox[1]:.0f},{mesh_bbox[2]:.0f}) "
				  f"max=({mesh_bbox[3]:.0f},{mesh_bbox[4]:.0f},{mesh_bbox[5]:.0f})", flush=True)
		if mesh_bbox is not None:
			auto_crop = _auto_crop(mesh_bbox, volume_extent_fullres, margin=3.0)
			print(f"[fit] auto-crop: x={auto_crop[0]} y={auto_crop[1]} z={auto_crop[2]} "
				  f"w={auto_crop[3]} h={auto_crop[4]} d={auto_crop[5]}", flush=True)
			data_cfg = dataclasses.replace(data_cfg, crop=auto_crop)

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
			scaledown=float(model_params_in.get("scaledown", scaledown)),
			z_step_eff=int(round(float(model_params_in.get("scaledown", scaledown)))),
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
		mdl.arc_enabled = False  # saved models have baked arcs
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
			scaledown=scaledown,
			z_step_eff=int(round(scaledown)),
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
