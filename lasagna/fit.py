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
import opt_loss_corr
import opt_loss_dir
import optimizer


def _arc_params_from_seed(
	seed: tuple[int, int, int],
	model_w: int,
	volume_extent_fullres: tuple[int, int, int],
) -> dict:
	"""Derive arc params from seed point and model width.

	seed: (cx, cy, cz) in fullres voxels
	model_w: model width in fullres voxels (circumferential extent)
	volume_extent_fullres: (X_total, Y_total, Z_total) in fullres voxels
	Returns dict with keys matching ModelConfig arc fields.
	"""
	seed_cx, seed_cy, seed_cz = float(seed[0]), float(seed[1]), float(seed[2])
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

	# Angular half-extent from model width
	half_angle = float(model_w) / (2.0 * arc_radius)
	half_angle = max(half_angle, 0.05)

	return {
		"arc_cx": arc_cx,
		"arc_cy": arc_cy,
		"arc_radius": arc_radius,
		"arc_angle0": seed_angle - half_angle,
		"arc_angle1": seed_angle + half_angle,
		"z_center": seed_cz,
	}


def _straight_params_from_seed(
	seed: tuple[int, int, int],
	model_w: int,
) -> dict:
	"""Derive straight params from seed point and model width.

	The line is centered at the seed XY, oriented at angle 0 (along X),
	with half-width = model_w / 2. The optimizer will adjust angle.
	"""
	return {
		"straight_cx": float(seed[0]),
		"straight_cy": float(seed[1]),
		"straight_angle": 0.0,
		"straight_half_w": float(model_w) / 2.0,
		"z_center": float(seed[2]),
	}


def _parse_corr_points(obj: dict, device: torch.device) -> fit_data.CorrPoints3D | None:
	"""Parse a VC3D corr_points collections dict into CorrPoints3D."""
	cols = obj.get("collections", {})
	print(f"[fit] _parse_corr_points: {len(cols) if isinstance(cols, dict) else 0} collections in input", flush=True)
	if not isinstance(cols, dict):
		print(f"[fit] _parse_corr_points: collections is not a dict: {type(cols).__name__}", flush=True)
		return None
	rows: list[list[float]] = []
	cids: list[int] = []
	pids: list[int] = []
	for _cid, col in cols.items():
		if not isinstance(col, dict):
			print(f"[fit] _parse_corr_points: col {_cid} is not a dict", flush=True)
			continue
		md = col.get("metadata", {})
		if not isinstance(md, dict):
			md = {}
		if bool(md.get("winding_is_absolute", True)):
			n_pts = len(col.get("points", {})) if isinstance(col.get("points"), dict) else 0
			print(f"[fit] _parse_corr_points: skipping collection {_cid} "
				  f"(winding_is_absolute={md.get('winding_is_absolute')}, {n_pts} points)", flush=True)
			continue
		pts = col.get("points", {})
		if not isinstance(pts, dict):
			continue
		try:
			cid_i = int(_cid)
		except Exception:
			cid_i = -1
		for _pid, pd in pts.items():
			if not isinstance(pd, dict):
				continue
			pv = pd.get("p", None)
			if not isinstance(pv, (list, tuple)) or len(pv) < 3:
				continue
			wa = pd.get("wind_a", None)
			if wa is None:
				raise ValueError(f"[fit] corr point {_pid} in collection {_cid} has no wind_a "
								 f"— d.tif was not available when this point was placed")
			try:
				pid_i = int(_pid)
			except Exception:
				pid_i = -1
			rows.append([float(pv[0]), float(pv[1]), float(pv[2]), float(wa)])
			cids.append(cid_i)
			pids.append(pid_i)
	if not rows:
		print(f"[fit] _parse_corr_points: no valid points found after parsing", flush=True)
		return None
	pts_t = torch.tensor(rows, dtype=torch.float32, device=device)
	col_t = torch.tensor(cids, dtype=torch.int64, device=device)
	pid_t = torch.tensor(pids, dtype=torch.int64, device=device)
	print(f"[fit] loaded {pts_t.shape[0]} corr_points from config "
		  f"({len(set(cids))} collections)")
	return fit_data.CorrPoints3D(points_xyz_winda=pts_t, collection_idx=col_t,
								 point_ids=pid_t)


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
	# Merge final parsed args into fit_config so checkpoint has all values
	fit_config.setdefault("args", {}).update(
		{k.replace("_", "-"): v for k, v in vars(args).items()})

	data_cfg = cli_data.from_args(args)
	model_cfg = cli_model.from_args(args)
	opt_cfg = cli_opt.from_args(args)
	progress_enabled = bool(args.progress)
	_out_dir = args.out_dir

	print("data:", data_cfg)
	print("model:", model_cfg)
	print("opt:", opt_cfg)

	device = torch.device(data_cfg.device)

	# Probe preprocessed data for scaledown and volume extent (in base/VC3D coords)
	prep_params = fit_data.get_preprocessed_params(str(data_cfg.input))
	source_to_base = float(prep_params.get("source_to_base", 1.0))
	# Model scaledown in base coords = channel_scaledown * source_to_base
	scaledown = float(prep_params["scaledown"]) * source_to_base
	volume_extent_fullres = prep_params.get("volume_extent_fullres")
	print(f"[fit] scaledown={scaledown} (source_sd={prep_params['scaledown']} "
		  f"source_to_base={source_to_base}) volume_extent={volume_extent_fullres}", flush=True)

	# --- Init from seed (new model only) ---
	is_new_model = model_cfg.model_input is None
	if is_new_model and data_cfg.seed is not None and data_cfg.model_w is not None:
		if model_cfg.init_mode == "straight":
			sp = _straight_params_from_seed(data_cfg.seed, data_cfg.model_w)
			model_cfg = dataclasses.replace(model_cfg, **sp)
			print(f"[fit] straight from seed: cx={sp['straight_cx']:.1f} cy={sp['straight_cy']:.1f} "
				  f"angle={sp['straight_angle']:.3f} half_w={sp['straight_half_w']:.1f} "
				  f"z={sp['z_center']:.1f}", flush=True)
		elif volume_extent_fullres is not None:
			arc = _arc_params_from_seed(data_cfg.seed, data_cfg.model_w, volume_extent_fullres)
			model_cfg = dataclasses.replace(model_cfg, **arc)
			print(f"[fit] arc from seed: cx={arc['arc_cx']:.1f} cy={arc['arc_cy']:.1f} "
				  f"r={arc['arc_radius']:.1f} a0={arc['arc_angle0']:.3f} a1={arc['arc_angle1']:.3f} "
				  f"z={arc['z_center']:.1f}", flush=True)

	# --- Size mesh from model_w, model_h, windings (new model only) ---
	if is_new_model and data_cfg.model_w is not None and data_cfg.model_h is not None and data_cfg.windings is not None:
		auto_mesh_w = max(2, int(data_cfg.model_w / model_cfg.mesh_step) + 1)
		auto_mesh_h = max(2, int(data_cfg.model_h / model_cfg.mesh_step) + 1)
		auto_depth = max(1, data_cfg.windings)

		model_cfg = dataclasses.replace(model_cfg, depth=auto_depth, mesh_h=auto_mesh_h, mesh_w=auto_mesh_w)
		print(f"[fit] model size: depth={auto_depth} mesh_h={auto_mesh_h} mesh_w={auto_mesh_w}", flush=True)

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
			straight_cx=model_cfg.straight_cx,
			straight_cy=model_cfg.straight_cy,
			straight_angle=model_cfg.straight_angle,
			straight_half_w=model_cfg.straight_half_w,
			init_mode=model_cfg.init_mode,
			volume_extent=None,
			pyramid_d=model_cfg.pyramid_d,
		)
	else:
		st = torch.load(model_cfg.model_input, map_location=device, weights_only=False)
		mdl = model.Model3D.from_checkpoint(st, device=device)

	print(f"Model3D: depth={mdl.depth} mesh_h={mdl.mesh_h} mesh_w={mdl.mesh_w} "
		  f"arc_enabled={mdl.arc_enabled}")

	# Parse correction points from config (injected by VC3D)
	corr_points_obj = cfg.pop("corr_points", None)
	corr_points_3d: fit_data.CorrPoints3D | None = None
	if isinstance(corr_points_obj, dict):
		corr_points_3d = _parse_corr_points(corr_points_obj, device)
	else:
		print(f"[fit] corr_points: not found in config (type={type(corr_points_obj).__name__})", flush=True)

	# Parse stages (before data loading so we know which channels to skip)
	stages = optimizer.load_stages_cfg(cfg)

	_skip_channels: set[str] = set()
	_any_data = any(s.global_opt.eff.get("data", 0) > 0 or s.global_opt.eff.get("data_plain", 0) > 0
					for s in stages)
	_any_pred_dt = any(s.global_opt.eff.get("pred_dt", 0) > 0 for s in stages)
	if not _any_data:
		_skip_channels.add("cos")
	if not _any_pred_dt:
		_skip_channels.add("pred_dt")
	if _skip_channels:
		print(f"[fit] skipping channels: {sorted(_skip_channels)}", flush=True)

	# --- Data loading (with auto-crop, blur, and reload support) ---
	def _load_data() -> fit_data.FitData3D:
		d = fit_data.load_3d_for_model(
			path=str(data_cfg.input), device=device, model=mdl,
			cuda_gridsample=data_cfg.cuda_gridsample,
			erode_valid_mask=data_cfg.erode_valid_mask,
			skip_channels=_skip_channels,
		)
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
		if corr_points_3d is not None:
			d = dataclasses.replace(d, corr_points=corr_points_3d)
		if data_cfg.winding_volume is not None:
			ox, oy, oz = d.origin_fullres
			sx, sy, sz = d.spacing
			wv_crop = (int(ox), int(oy), int(oz),
					   int(X * sx), int(Y * sy), int(Z * sz))
			wv_t, wv_min, wv_max = fit_data.load_winding_volume(
				path=data_cfg.winding_volume, device=device,
				crop=wv_crop, downscale=scaledown)
			d = dataclasses.replace(d, winding_volume=wv_t,
						winding_min=wv_min, winding_max=wv_max)
		return d

	data = _load_data()

	# Print loaded data summary
	Z, Y, X = data.size
	_data_bytes = sum(t.nbytes for t in [data.cos, data.grad_mag, data.nx, data.ny] if t is not None)
	if data.pred_dt is not None:
		_data_bytes += data.pred_dt.nbytes
	if data.winding_volume is not None:
		_data_bytes += data.winding_volume.nbytes
	print(f"[fit] data: size=({Z},{Y},{X}) origin={data.origin_fullres} spacing={data.spacing} "
		  f"pred_dt={data.pred_dt is not None} winding_volume={data.winding_volume is not None} "
		  f"corr_points={data.corr_points is not None} "
		  f"mem={_data_bytes / 2**30:.2f} GiB", flush=True)

	# Print initial mesh stats
	with torch.no_grad():
		xyz = mdl._grid_xyz()
		mn = xyz.amin(dim=(0, 1, 2)).cpu().numpy().tolist()
		mx = xyz.amax(dim=(0, 1, 2)).cpu().numpy().tolist()
		mean = xyz.mean(dim=(0, 1, 2)).cpu().numpy().tolist()
		print(f"initial mesh: mean={[round(v, 1) for v in mean]} "
			  f"min={[round(v, 1) for v in mn]} max={[round(v, 1) for v in mx]}")

	def _save_model(path: str) -> None:
		if mdl.arc_enabled:
			mdl.bake_arc_into_mesh()
		if mdl.straight_enabled:
			mdl.bake_straight_into_mesh()
		st = dict(mdl.state_dict())
		# Store flat mesh instead of pyramid levels
		ms_keys = [k for k in st if k.startswith("mesh_ms.")]
		for k in ms_keys:
			del st[k]
		with torch.no_grad():
			st["mesh_flat"] = mdl.mesh_coarse().detach().clone()
		st["_model_params_"] = asdict(mdl.params)
		st["_fit_config_"] = fit_config
		corr_results = opt_loss_corr.get_last_results()
		if corr_results is not None:
			st["_corr_points_results_"] = corr_results
		# Store winding volume auto-offset if computed
		from opt_loss_winding_volume import _winding_offset, _winding_direction
		if _winding_offset is not None:
			st["_winding_offset_"] = _winding_offset
			st["_winding_direction_"] = _winding_direction
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

	# Configure corr snap mode
	opt_loss_corr.set_snap_mode(opt_cfg.corr_snap)
	opt_loss_dir.set_mask_zero_normals(opt_cfg.normal_mask_zero)

	# Run optimization
	seed_xyz = tuple(float(v) for v in data_cfg.seed) if data_cfg.seed is not None else None
	optimizer.optimize(
		model=mdl,
		data=data,
		stages=stages,
		snapshot_interval=opt_cfg.snapshot_interval,
		snapshot_fn=_snapshot,
		progress_fn=_progress,
		load_data_fn=_load_data,
		volume_extent_fullres=volume_extent_fullres,
		seed_xyz=seed_xyz,
	)

	if device.type == "cuda":
		peak_gb = torch.cuda.max_memory_allocated(device) / 2**30
		print(f"[fit] peak GPU memory: {peak_gb:.2f} GiB", flush=True)

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
		tifxyz_argv = ["--input", str(model_out), "--output", export_dir]
		voxel_size_um = cfg.get("voxel_size_um")
		if voxel_size_um is not None:
			tifxyz_argv += ["--voxel-size-um", str(float(voxel_size_um))]
		fit2tifxyz.main(tifxyz_argv)

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
