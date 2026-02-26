import argparse
import sys
from pathlib import Path

import cli_data
import cli_model
import cli_opt
import cli_vis
import fit_data
import model
import optimizer
import point_constraints
import torch
import vis
from dataclasses import asdict
from dataclasses import replace
import json

import cli_json


def _build_parser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(
		prog="fit.py",
		description="2D fit entrypoint (CLI composition)",
	)
	cli_data.add_args(p)
	cli_model.add_args(p)
	cli_opt.add_args(p)
	cli_vis.add_args(p)
	point_constraints.add_args(p)
	p.add_argument("--progress", action="store_true", default=False,
		help="Print machine-readable PROGRESS lines to stdout")
	return p


def main(argv: list[str] | None = None) -> int:
	if argv is None:
		argv = sys.argv[1:]

	# Keep --points <file.json> out of generic cfg json-path detection.
	argv_cfg_scan: list[str] = []
	points_argv: list[str] = []
	i = 0
	while i < len(argv):
		a = str(argv[i])
		if a == "--points" and (i + 1) < len(argv):
			points_argv.extend([a, str(argv[i + 1])])
			i += 2
			continue
		argv_cfg_scan.append(a)
		i += 1
	parser = _build_parser()
	cfg_paths, argv_rest = cli_json.split_cfg_argv(argv_cfg_scan)
	cfg_paths = [str(x) for x in cfg_paths]
	cfg = cli_json.merge_cfgs(cfg_paths)
	import copy
	fit_config = copy.deepcopy(cfg)  # snapshot before pops; saved into checkpoint
	cli_json.apply_defaults_from_cfg_args(parser, cfg)
	args = parser.parse_args((argv_rest or []) + points_argv)

	data_cfg = cli_data.from_args(args)
	model_cfg = cli_model.from_args(args)
	# --bbox implies init_size_frac=1.0 so model exactly covers the bounding box
	if getattr(args, "bbox", None) is not None:
		model_cfg = replace(model_cfg, init_size_frac=1.0, init_size_frac_h=None, init_size_frac_v=None)
	opt_cfg = cli_opt.from_args(args)
	vis_cfg = cli_vis.from_args(args)
	progress_enabled = bool(args.progress)
	points_cfg = point_constraints.from_args(args)
	points_tensor, points_collection_idx, points_ids = point_constraints.load_points_tensor(points_cfg)
	# Merge inline corr_points from config JSON (e.g. sent by VC3D)
	corr_points_obj = cfg.pop("corr_points", None)
	if isinstance(corr_points_obj, dict):
		inline_pts, inline_cids, inline_pids = point_constraints.load_points_from_collections_dict(corr_points_obj)
		if inline_pts.shape[0] > 0:
			print(f"[fit] loaded {inline_pts.shape[0]} inline corr_points from config")
			points_tensor = torch.cat([points_tensor, inline_pts], dim=0)
			points_collection_idx = torch.cat([points_collection_idx, inline_cids], dim=0)
			points_ids = torch.cat([points_ids, inline_pids], dim=0)
	point_constraints.print_points_tensor(points_tensor)

	print("data:", data_cfg)
	print("model:", model_cfg)
	print("opt:", opt_cfg)
	print("vis:", vis_cfg)

	model_params_in: dict | None = None
	z_size_use = int(model_cfg.z_size)
	z_size_from_state = 0
	mesh_step_use = int(model_cfg.mesh_step_px)
	winding_step_use = int(model_cfg.winding_step_px)
	subsample_mesh_use = int(model_cfg.subsample_mesh)
	subsample_winding_use = int(model_cfg.subsample_winding)
	z_step_use = int(data_cfg.z_step)
	if model_cfg.model_input is not None:
		st_in = torch.load(model_cfg.model_input, map_location="cpu")
		if isinstance(st_in, dict) and isinstance(st_in.get("_model_params_", None), dict):
			model_params_in = st_in["_model_params_"]
		if isinstance(st_in, dict):
			for k in ("amp", "bias", "mesh_ms.0", "conn_offset_ms.0"):
				t = st_in.get(k, None)
				if t is None or not hasattr(t, "shape"):
					continue
				try:
					z_size_from_state = max(1, int(t.shape[0]))
					z_size_use = int(z_size_from_state)
					break
				except Exception:
					continue
		if model_params_in is not None:
			if "mesh_step_px" in model_params_in:
				mesh_step_use = int(model_params_in["mesh_step_px"])
			if "winding_step_px" in model_params_in:
				winding_step_use = int(model_params_in["winding_step_px"])
			if "subsample_mesh" in model_params_in:
				subsample_mesh_use = int(model_params_in["subsample_mesh"])
			if "subsample_winding" in model_params_in:
				subsample_winding_use = int(model_params_in["subsample_winding"])
			if int(z_step_use) == 1 and ("z_step_vx" in model_params_in):
				z_step_use = max(1, int(model_params_in["z_step_vx"]))
			c6 = model_params_in.get("crop_fullres_xyzwhd", None)
			if isinstance(c6, (list, tuple)) and len(c6) == 6:
				x, y, w, h, z0, d = (int(v) for v in c6)
				if data_cfg.crop is None:
					data_cfg = replace(data_cfg, crop=(x, y, w, h))
				if data_cfg.unet_z is None:
					data_cfg = replace(data_cfg, unet_z=int(z0))
				z_size_use = max(1, int(d))
				data_cfg = replace(data_cfg, z_step=int(z_step_use))
			elif "z_size" in model_params_in:
				z_size_use = max(1, int(model_params_in["z_size"]))
		if int(z_size_from_state) > 0:
			z_size_use = max(int(z_size_use), int(z_size_from_state))

	# Probe preprocessed zarr for z_step_eff (full-res spacing between consecutive slices).
	# The preprocessed zarr already has z_step*scaledown stepping baked in, so each
	# consecutive zarr slice maps 1:1 to a model z-plane.
	prep_params = fit_data.get_preprocessed_params(str(data_cfg.input))
	if prep_params is not None:
		zarr_z_step = int(prep_params["z_step"])
		zarr_z_step_eff = int(prep_params["z_step_eff"])
		print(f"[fit] zarr z_step={zarr_z_step} z_step_eff={zarr_z_step_eff}", flush=True)
		# The model's z_step_vx must match the zarr's z_step so that dz in
		# z-normal loss and export z-coordinates are correct.
		if int(z_step_use) != zarr_z_step:
			print(f"[fit] z_step override: {z_step_use} -> {zarr_z_step} (from zarr)", flush=True)
			z_step_use = zarr_z_step
			data_cfg = replace(data_cfg, z_step=int(z_step_use))

	# When loading a checkpoint with a preprocessed zarr, the checkpoint's z_size
	# may have been computed with a different z_step_eff. Recompute z_size to
	# maintain the same full-res depth using the zarr's actual z_step_eff.
	if prep_params is not None and model_params_in is not None:
		zarr_z_step_eff = int(prep_params["z_step_eff"])
		ckpt_z_step_vx = int(model_params_in.get("z_step_vx", 1))
		ckpt_sd = float(model_params_in.get("scaledown", data_cfg.downscale))
		ckpt_z_step_eff = max(1, int(round(ckpt_z_step_vx * ckpt_sd)))
		# Infer what full-res depth the checkpoint intended
		original_fullres = int(z_size_use) * ckpt_z_step_eff
		correct_z_size = max(1, int(round(original_fullres / zarr_z_step_eff)))
		if z_size_use != correct_z_size:
			print(f"[fit] z_size corrected: {z_size_use} -> {correct_z_size} "
				  f"(ckpt z_step_eff={ckpt_z_step_eff} vs zarr={zarr_z_step_eff})", flush=True)
			z_size_use = correct_z_size

	# --bbox: args.z_size is depth in full-res voxels; convert to number of z-slices.
	# Always use args.z_size (the authoritative full-res depth from the GUI/CLI),
	# even when loading a checkpoint whose z_size may be from a run with wrong z_step.
	if getattr(args, "bbox", None) is not None:
		z_size_fullres = max(1, int(args.z_size))
		if prep_params is not None:
			z_step_eff = int(prep_params["z_step_eff"])
		else:
			z_step_eff = max(1, int(z_step_use) * int(round(data_cfg.downscale)))
		z_size_new = max(1, int(round(z_size_fullres / z_step_eff)))
		z_size_use = z_size_new
		print(f"[fit] bbox z_size: {z_size_fullres} fullres -> {z_size_use} slices (z_step_eff={z_step_eff})", flush=True)

	# Effective full-res z-spacing per working z-plane.  Preprocessed zarrs
	# bake z_step*downscale into each slice; raw OME-Zarr slices are loaded
	# at raw z_step spacing (downscale only affects x/y).
	if prep_params is not None:
		_z_step_fullres = int(prep_params["z_step_eff"])
	else:
		_z_step_fullres = int(z_step_use)
	points_tensor_work = point_constraints.to_working_coords(
		points_xyz_winda=points_tensor,
		downscale=float(data_cfg.downscale),
		crop_xywh=data_cfg.crop,
		z0=data_cfg.unet_z,
		z_step=int(_z_step_fullres),
		z_size=int(z_size_use),
	)
	print(f"[fit] point coord mapping: z0={data_cfg.unet_z} z_step_fullres={_z_step_fullres} "
		  f"ds={data_cfg.downscale} crop={data_cfg.crop}")
	print("[point_constraints] points_xyz_winda_work", points_tensor_work)
	points_all = torch.empty((0, 4), dtype=torch.float32)
	idx_left = torch.empty((0, 3), dtype=torch.int64)
	valid_left = torch.empty((0,), dtype=torch.bool)
	min_dist_left = torch.empty((0,), dtype=torch.float32)
	idx_right = torch.empty((0, 3), dtype=torch.int64)
	valid_right = torch.empty((0,), dtype=torch.bool)
	min_dist_right = torch.empty((0,), dtype=torch.float32)
	z_hi = torch.empty((0,), dtype=torch.int64)
	z_frac = torch.empty((0,), dtype=torch.float32)
	winding_obs = torch.empty((0,), dtype=torch.float32)
	winding_avg = torch.empty((0,), dtype=torch.float32)
	winding_err = torch.empty((0,), dtype=torch.float32)

	device = torch.device(data_cfg.device)
	crop_xyzwhd = None
	if data_cfg.crop is not None and data_cfg.unet_z is not None:
		x, y, w, h = (int(v) for v in data_cfg.crop)
		crop_xyzwhd = (x, y, w, h, int(data_cfg.unet_z), int(z_size_use))

	if model_cfg.model_input is not None and model_params_in is not None:
		# Derive model init size from checkpoint tensor shapes (ignore init_size_frac unless creating a new model).
		st_cpu = torch.load(model_cfg.model_input, map_location="cpu")
		mesh0 = st_cpu.get("mesh_ms.0", None) if isinstance(st_cpu, dict) else None
		if mesh0 is None or not hasattr(mesh0, "shape"):
			raise ValueError("model_input missing mesh_ms.0")
		gh = int(mesh0.shape[2])
		gw = int(mesh0.shape[3])
		init = model.ModelInit(
			init_size_frac=1.0,
			init_size_frac_h=None,
			init_size_frac_v=None,
			mesh_step_px=int(mesh_step_use),
			winding_step_px=int(winding_step_use),
			mesh_h=int(gh),
			mesh_w=int(gw),
		)
		mdl = model.Model2D(
			init=init,
			device=device,
			z_size=int(z_size_use),
			subsample_mesh=int(subsample_mesh_use),
			subsample_winding=int(subsample_winding_use),
			z_step_vx=int(z_step_use),
			scaledown=float(data_cfg.downscale),
			crop_xyzwhd=crop_xyzwhd,
		)
	else:
		mdl = None
	if model_cfg.model_input is not None:
		st = torch.load(model_cfg.model_input, map_location=device)
		# Truncate z-dimension if checkpoint z_size != model z_size (e.g. after
		# bbox z_size correction due to z_step change).
		_ckpt_z = None
		for _ck in ("mesh_ms.0", "amp", "bias"):
			_ct = st.get(_ck)
			if isinstance(_ct, torch.Tensor) and _ct.dim() >= 1 and _ct.shape[0] > 0:
				_ckpt_z = int(_ct.shape[0])
				break
		if _ckpt_z is not None and _ckpt_z != int(z_size_use):
			_nz = int(z_size_use)
			if _nz < _ckpt_z:
				print(f"[fit] truncating checkpoint z-planes: {_ckpt_z} -> {_nz} (keeping first {_nz})", flush=True)
				for _ck in list(st.keys()):
					_cv = st[_ck]
					if isinstance(_cv, torch.Tensor) and _cv.dim() >= 1 and int(_cv.shape[0]) == _ckpt_z:
						st[_ck] = _cv[:_nz].contiguous()
			else:
				print(f"[fit] WARNING: model z_size ({_nz}) > checkpoint z ({_ckpt_z})", flush=True)
		miss, unexp = mdl.load_state_dict_compat(st, strict=False)
		if unexp:
			print("state_dict: unexpected keys:", sorted(unexp))
		if miss:
			print("state_dict: missing keys:", sorted(miss))
		with torch.no_grad():
			xy0 = mdl.mesh_coarse().detach()
			mean_xy = xy0.mean(dim=(0, 2, 3)).to(dtype=torch.float32).cpu().numpy().tolist()
			min_xy = xy0.amin(dim=(0, 2, 3)).to(dtype=torch.float32).cpu().numpy().tolist()
			max_xy = xy0.amax(dim=(0, 2, 3)).to(dtype=torch.float32).cpu().numpy().tolist()
			print(f"loaded mesh_coarse: mean_xy={mean_xy} min_xy={min_xy} max_xy={max_xy}")
	if mdl is not None:
		print("model_init:", mdl.init)
		print("mesh:", mdl.mesh_h, mdl.mesh_w)
	if int(points_tensor_work.shape[0]) > 0 and mdl is not None:
		print("[point_constraints] starting closest segment search")
		with torch.no_grad():
			xy_lr0 = mdl._grid_xy()
			xy_conn0 = mdl._xy_conn_px(xy_lr=xy_lr0)
		(points_all,
		 idx_left, valid_left, min_dist_left,
		 idx_right, valid_right, min_dist_right,
		 z_hi, z_frac) = point_constraints.closest_conn_segment_indices(points_xyz_winda=points_tensor_work, xy_conn=xy_conn0)
		n_valid = int(valid_left.sum().item())
		n_interp = int((z_frac > 0.0).sum().item())
		print(f"[point_constraints] {int(points_all.shape[0])} pts: {n_valid} valid, {n_interp} z-interpolated")
		for pi in range(int(points_all.shape[0])):
			pf = points_tensor[pi]  # original fullres
			pw = points_all[pi]     # working coords (after margin shift)
			z_lo_i = int(idx_left[pi, 0].item())
			z_hi_i = int(z_hi[pi].item())
			frac_i = z_frac[pi].item()
			print(f"  pt{pi}: fullres=({pf[0]:.0f},{pf[1]:.0f},{pf[2]:.0f}) "
				  f"work=({pw[0]:.1f},{pw[1]:.1f},{pw[2]:.2f}) "
				  f"z_lo={z_lo_i} z_hi={z_hi_i} z_frac={frac_i:.3f}")

	_z0_fullres = int(data_cfg.unet_z) if data_cfg.unet_z is not None else 0
	_out_dir = vis_cfg.out_dir  # None means skip all debug/vis output
	# Corr point vis always gets a directory when points exist.
	# Falls back to cwd (the fit_service working directory).
	_corr_out_dir: str | None = None
	if int(points_all.shape[0]) > 0:
		if _out_dir is not None:
			_corr_out_dir = _out_dir
		else:
			import os
			_corr_out_dir = os.getcwd()
		print(f"[fit] corr_out_dir={_corr_out_dir}")
	data = cli_data.load_fit_data(data_cfg, z_size=int(z_size_use), out_dir_base=_out_dir)
	data = fit_data.FitData(
		cos=data.cos,
		grad_mag=data.grad_mag,
		dir0=data.dir0,
		dir1=data.dir1,
		valid=data.valid,
		dir0_y=data.dir0_y,
		dir1_y=data.dir1_y,
		dir0_x=data.dir0_x,
		dir1_x=data.dir1_x,
		downscale=float(data.downscale),
		data_margin_xy=data.data_margin_xy,
		constraints=fit_data.ConstraintsData(
			points=fit_data.PointConstraintsData(
				points_xyz_winda=points_all,
				collection_idx=points_collection_idx,
				idx_left=idx_left,
				valid_left=valid_left,
				idx_right=idx_right,
				valid_right=valid_right,
				z_hi=z_hi,
				z_frac=z_frac,
			)
		),
	)
	device = data.cos.device
	_margin_xy = tuple(float(v) for v in data.data_margin_xy)
	_data_size = tuple(int(v) for v in data.size)  # (h, w) in model pixels
	# Shift point constraints from crop-relative to data-space model pixels.
	# to_working_coords maps to (abs - crop_origin) / ds, but with margins
	# the crop origin sits at (margin_x, margin_y) in model pixel space.
	if _margin_xy[0] != 0.0 or _margin_xy[1] != 0.0:
		if int(points_all.shape[0]) > 0:
			points_all[:, 0] += _margin_xy[0]
			points_all[:, 1] += _margin_xy[1]
			print(f"[fit] shifted {int(points_all.shape[0])} constraint points by margin ({_margin_xy[0]:.1f}, {_margin_xy[1]:.1f})")
		if int(points_tensor_work.shape[0]) > 0:
			points_tensor_work[:, 0] += _margin_xy[0]
			points_tensor_work[:, 1] += _margin_xy[1]
	# For checkpoint models: translate mesh by the delta between new and old margins.
	# A checkpoint saved with margins already has them baked into the mesh coords.
	if mdl is not None:
		old_margin = (0.0, 0.0)
		if model_params_in is not None:
			old_m = model_params_in.get("data_margin_modelpx", (0.0, 0.0))
			if isinstance(old_m, (list, tuple)) and len(old_m) == 2:
				old_margin = (float(old_m[0]), float(old_m[1]))
		delta_x = _margin_xy[0] - old_margin[0]
		delta_y = _margin_xy[1] - old_margin[1]
		updates = {"data_size_modelpx": _data_size, "data_margin_modelpx": _margin_xy}
		if delta_x != 0.0 or delta_y != 0.0:
			with torch.no_grad():
				mdl.mesh_ms[-1].data[:, 0, :, :] += delta_x
				mdl.mesh_ms[-1].data[:, 1, :, :] += delta_y
			print(f"[fit] translated checkpoint mesh by margin delta ({delta_x:.1f}, {delta_y:.1f})"
				  f" (old={old_margin}, new={_margin_xy})")
		else:
			print(f"[fit] checkpoint margin matches data margin ({_margin_xy[0]:.1f}, {_margin_xy[1]:.1f}), no translation")
		mdl.params = replace(mdl.params, **updates)
	if mdl is None:
		mdl = model.Model2D.from_fit_data(
			data=data,
			mesh_step_px=int(mesh_step_use),
			winding_step_px=int(winding_step_use),
			init_size_frac=model_cfg.init_size_frac,
			init_size_frac_h=model_cfg.init_size_frac_h,
			init_size_frac_v=model_cfg.init_size_frac_v,
			z_size=int(z_size_use),
			z_step_vx=int(z_step_use),
			scaledown=float(data_cfg.downscale),
			device=device,
			subsample_mesh=int(subsample_mesh_use),
			subsample_winding=int(subsample_winding_use),
			crop_xyzwhd=crop_xyzwhd,
			data_margin_modelpx=_margin_xy,
			data_size_modelpx=_data_size,
		)
		print("model_init:", mdl.init)
		print("mesh:", mdl.mesh_h, mdl.mesh_w)
	if int(points_all.shape[0]) == 0 and int(points_tensor_work.shape[0]) > 0:
		# Deferred setup: early block was skipped because mdl wasn't created yet.
		# points_tensor_work is already margin-shifted at this point.
		print("[point_constraints] deferred closest segment search (model created from scratch)")
		with torch.no_grad():
			xy_lr0 = mdl._grid_xy()
			xy_conn0 = mdl._xy_conn_px(xy_lr=xy_lr0)
		(points_all,
		 idx_left, valid_left, min_dist_left,
		 idx_right, valid_right, min_dist_right,
		 z_hi, z_frac) = point_constraints.closest_conn_segment_indices(points_xyz_winda=points_tensor_work, xy_conn=xy_conn0)
		n_valid = int(valid_left.sum().item())
		n_interp = int((z_frac > 0.0).sum().item())
		print(f"[point_constraints] {int(points_all.shape[0])} pts: {n_valid} valid, {n_interp} z-interpolated")
		for pi in range(int(points_all.shape[0])):
			pf = points_tensor[pi]  # original fullres
			pw = points_all[pi]     # working coords (margin-shifted)
			z_lo_i = int(idx_left[pi, 0].item())
			z_hi_i = int(z_hi[pi].item())
			frac_i = z_frac[pi].item()
			print(f"  pt{pi}: fullres=({pf[0]:.0f},{pf[1]:.0f},{pf[2]:.0f}) "
				  f"work=({pw[0]:.1f},{pw[1]:.1f},{pw[2]:.2f}) "
				  f"z_lo={z_lo_i} z_hi={z_hi_i} z_frac={frac_i:.3f}")
		if _out_dir is not None:
			_corr_out_dir = _out_dir
		else:
			import os
			_corr_out_dir = os.getcwd()
		print(f"[fit] corr_out_dir={_corr_out_dir}")
		data = replace(data, constraints=fit_data.ConstraintsData(
			points=fit_data.PointConstraintsData(
				points_xyz_winda=points_all,
				collection_idx=points_collection_idx,
				idx_left=idx_left,
				valid_left=valid_left,
				idx_right=idx_right,
				valid_right=valid_right,
				z_hi=z_hi,
				z_frac=z_frac,
			)
		))
	def _build_corr_points_results(*, obs, avg, err, pt_ids, collection_idx):
		"""Build JSON-serializable dict of per-point winding results."""
		import math
		result = {"points": {}, "collection_avgs": {}}
		n = int(obs.shape[0]) if obs.numel() > 0 else 0
		for i in range(n):
			pid = int(pt_ids[i].item()) if i < int(pt_ids.shape[0]) else -1
			o = float(obs[i].item())
			e = float(err[i].item())
			cid = int(collection_idx[i].item()) if i < int(collection_idx.shape[0]) else -1
			entry = {"collection_id": cid}
			if math.isfinite(o):
				entry["winding_obs"] = round(o, 6)
			else:
				entry["winding_obs"] = None
			if math.isfinite(e):
				entry["winding_err"] = round(e, 6)
			else:
				entry["winding_err"] = None
			result["points"][str(pid)] = entry
		# Per-collection averages
		if avg.numel() > 0 and collection_idx.numel() > 0:
			import math as _m
			uc = torch.unique(collection_idx)
			for cid_t in uc.tolist():
				cid_int = int(cid_t)
				mask = (collection_idx == cid_int)
				avg_vals = avg[mask]
				if avg_vals.numel() > 0:
					v = float(avg_vals[0].item())
					if _m.isfinite(v):
						result["collection_avgs"][str(cid_int)] = round(v, 6)
		return result

	_last_corr_results: list[dict | None] = [None]  # mutable container for nested function access

	if int(points_all.shape[0]) > 0:
		with torch.no_grad():
			xy_lr_corr = mdl._grid_xy()
			xy_conn_corr = mdl._xy_conn_px(xy_lr=xy_lr_corr)
			(pts_corr,
			 idx_l_corr, ok_l_corr, _d_l,
			 idx_r_corr, ok_r_corr, _d_r,
			 z_hi_corr, z_frac_corr) = point_constraints.closest_conn_segment_indices(
				points_xyz_winda=points_all,
				xy_conn=xy_conn_corr,
			)
			_wobs0, winding_avg, winding_err = point_constraints.winding_observed_and_error(
				points_xyz_winda=pts_corr,
				collection_idx=points_collection_idx,
				xy_conn=xy_conn_corr,
				idx_left=idx_l_corr,
				valid_left=ok_l_corr,
				idx_right=idx_r_corr,
				valid_right=ok_r_corr,
				z_hi=z_hi_corr,
				z_frac=z_frac_corr,
			)
		_last_corr_results[0] = _build_corr_points_results(
			obs=_wobs0, avg=winding_avg, err=winding_err,
			pt_ids=points_ids, collection_idx=points_collection_idx)
		if _corr_out_dir is not None:
			vis.save_corr_points(
				data=data,
				xy_lr=xy_lr_corr,
				xy_conn=xy_conn_corr,
				points_xyz_winda=pts_corr,
				idx_left=idx_l_corr,
				valid_left=ok_l_corr,
				idx_right=idx_r_corr,
				valid_right=ok_r_corr,
				winding_avg=winding_avg,
				winding_err=winding_err,
				postfix="init",
				out_dir=_corr_out_dir,
				scale=vis_cfg.scale,
				z0_fullres=_z0_fullres,
				z_step_fullres=_z_step_fullres,
			)

	if _out_dir is not None:
		vis.save(model=mdl, data=data, postfix="init", out_dir=_out_dir, scale=vis_cfg.scale)
		mdl.save_tiff(data=data, path=f"{_out_dir}/raw_init.tif")
	stages = optimizer.load_stages_cfg(cfg)

	def _save_model_snapshot(*, stage: str, step: int) -> None:
		if _out_dir is None:
			return
		out = Path(_out_dir)
		out.mkdir(parents=True, exist_ok=True)
		out_snap = out / "model_snapshots"
		out_snap.mkdir(parents=True, exist_ok=True)
		p = out_snap / f"model_{stage}_{step:06d}.pt"
		st = dict(mdl.state_dict())
		st["_model_params_"] = asdict(mdl.params)
		st["_fit_config_"] = fit_config
		if _last_corr_results[0] is not None:
			st["_corr_points_results_"] = _last_corr_results[0]
		torch.save(st, str(p))

	def _save_model_output_final() -> None:
		if model_cfg.model_output is None:
			return
		st = dict(mdl.state_dict())
		st["_model_params_"] = asdict(mdl.params)
		st["_fit_config_"] = fit_config
		if _last_corr_results[0] is not None:
			st["_corr_points_results_"] = _last_corr_results[0]
		torch.save(st, str(model_cfg.model_output))

	_save_model_snapshot(stage="init", step=0)
	def _snapshot(*, stage: str, step: int, loss: float, data, res=None, vis_losses=None) -> None:
		if _corr_out_dir is not None and int(points_all.shape[0]) > 0:
			if res is not None:
				xy_lr_corr = res.xy_lr
				xy_conn_corr = res.xy_conn
			else:
				with torch.no_grad():
					xy_lr_corr = mdl._grid_xy()
					xy_conn_corr = mdl._xy_conn_px(xy_lr=xy_lr_corr)
			(pts_corr,
			 idx_l_corr, ok_l_corr, _d_l,
			 idx_r_corr, ok_r_corr, _d_r,
			 z_hi_corr, z_frac_corr) = point_constraints.closest_conn_segment_indices(
				points_xyz_winda=points_all,
				xy_conn=xy_conn_corr,
			)
			_wobs, wavg, werr = point_constraints.winding_observed_and_error(
				points_xyz_winda=pts_corr,
				collection_idx=points_collection_idx,
				xy_conn=xy_conn_corr,
				idx_left=idx_l_corr,
				valid_left=ok_l_corr,
				idx_right=idx_r_corr,
				valid_right=ok_r_corr,
				z_hi=z_hi_corr,
				z_frac=z_frac_corr,
			)
			_last_corr_results[0] = _build_corr_points_results(
				obs=_wobs, avg=wavg, err=werr,
				pt_ids=points_ids, collection_idx=points_collection_idx)
			vis.save_corr_points(
				data=data,
				xy_lr=xy_lr_corr,
				xy_conn=xy_conn_corr,
				points_xyz_winda=pts_corr,
				idx_left=idx_l_corr,
				valid_left=ok_l_corr,
				idx_right=idx_r_corr,
				valid_right=ok_r_corr,
				winding_avg=wavg,
				winding_err=werr,
				postfix=f"{stage}_{step:06d}",
				out_dir=_corr_out_dir,
				scale=vis_cfg.scale,
				z0_fullres=_z0_fullres,
				z_step_fullres=_z_step_fullres,
			)
		if _out_dir is not None:
			vis.save(
				model=mdl,
				data=data,
				res=res,
				vis_losses=vis_losses,
				postfix=f"{stage}_{step:06d}",
				out_dir=_out_dir,
				scale=vis_cfg.scale,
			)
			mdl.save_tiff(data=data, path=f"{_out_dir}/raw_{stage}_{step:06d}.tif")
		_save_model_snapshot(stage=stage, step=step)

	def _corr_snapshot(*, stage: str, step: int, data, res) -> None:
		"""Save corr-point debug images (lightweight, every 100 steps)."""
		if _corr_out_dir is None or int(points_all.shape[0]) <= 0:
			return
		with torch.no_grad():
			xy_lr_corr = res.xy_lr
			xy_conn_corr = res.xy_conn
			(pts_corr,
			 idx_l_corr, ok_l_corr, _d_l,
			 idx_r_corr, ok_r_corr, _d_r,
			 z_hi_corr, z_frac_corr) = point_constraints.closest_conn_segment_indices(
				points_xyz_winda=points_all,
				xy_conn=xy_conn_corr,
			)
			_wobs, wavg, werr = point_constraints.winding_observed_and_error(
				points_xyz_winda=pts_corr,
				collection_idx=points_collection_idx,
				xy_conn=xy_conn_corr,
				idx_left=idx_l_corr,
				valid_left=ok_l_corr,
				idx_right=idx_r_corr,
				valid_right=ok_r_corr,
				z_hi=z_hi_corr,
				z_frac=z_frac_corr,
			)
			_last_corr_results[0] = _build_corr_points_results(
				obs=_wobs, avg=wavg, err=werr,
				pt_ids=points_ids, collection_idx=points_collection_idx)
		vis.save_corr_points(
			data=data,
			xy_lr=xy_lr_corr,
			xy_conn=xy_conn_corr,
			points_xyz_winda=pts_corr,
			idx_left=idx_l_corr,
			valid_left=ok_l_corr,
			idx_right=idx_r_corr,
			valid_right=ok_r_corr,
			winding_avg=wavg,
			winding_err=werr,
			postfix=f"{stage}_{step:06d}",
			out_dir=_corr_out_dir,
			scale=vis_cfg.scale,
			z0_fullres=_z0_fullres,
			z_step_fullres=_z_step_fullres,
		)

	def _progress(*, step: int, total: int, loss: float, **_kw: object) -> None:
		if progress_enabled:
			print(f"PROGRESS {step} {total} {loss:.6f}", flush=True)

	data = optimizer.optimize(
		model=mdl,
		data=data,
		data_cfg=data_cfg,
		data_out_dir_base=_out_dir,
		stages=stages,
		snapshot_interval=opt_cfg.snapshot_interval,
		snapshot_fn=_snapshot,
		corr_snapshot_fn=_corr_snapshot,
		progress_fn=_progress,
	)
	if _corr_out_dir is not None and int(points_all.shape[0]) > 0:
		with torch.no_grad():
			xy_lr_corr = mdl._grid_xy()
			xy_conn_corr = mdl._xy_conn_px(xy_lr=xy_lr_corr)
			(pts_corr,
			 idx_l_corr, ok_l_corr, _d_l,
			 idx_r_corr, ok_r_corr, _d_r,
			 z_hi_corr, z_frac_corr) = point_constraints.closest_conn_segment_indices(
				points_xyz_winda=points_all,
				xy_conn=xy_conn_corr,
			)
			_wobsf, wavgf, werrf = point_constraints.winding_observed_and_error(
				points_xyz_winda=pts_corr,
				collection_idx=points_collection_idx,
				xy_conn=xy_conn_corr,
				idx_left=idx_l_corr,
				valid_left=ok_l_corr,
				idx_right=idx_r_corr,
				valid_right=ok_r_corr,
				z_hi=z_hi_corr,
				z_frac=z_frac_corr,
			)
		_last_corr_results[0] = _build_corr_points_results(
			obs=_wobsf, avg=wavgf, err=werrf,
			pt_ids=points_ids, collection_idx=points_collection_idx)
		vis.save_corr_points(
			data=data,
			xy_lr=xy_lr_corr,
			xy_conn=xy_conn_corr,
			points_xyz_winda=pts_corr,
			idx_left=idx_l_corr,
			valid_left=ok_l_corr,
			idx_right=idx_r_corr,
			valid_right=ok_r_corr,
			winding_avg=wavgf,
			winding_err=werrf,
			postfix="final",
			out_dir=_corr_out_dir,
			scale=vis_cfg.scale,
			z0_fullres=_z0_fullres,
			z_step_fullres=_z_step_fullres,
		)
	if _out_dir is not None:
		vis.save(model=mdl, data=data, postfix="final", out_dir=_out_dir, scale=vis_cfg.scale)
		mdl.save_tiff(data=data, path=f"{_out_dir}/raw_final.tif")
	_save_model_snapshot(stage="final", step=0)
	_save_model_output_final()
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
