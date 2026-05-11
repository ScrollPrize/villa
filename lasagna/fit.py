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


def _stage_start(label: str) -> float:
	return 0.0


def _stage_done(label: str, t0: float) -> None:
	return None


def _grid_center(mdl: "model.Model3D") -> torch.Tensor:
	"""Bilinear center of the model grid — matches (Hm-1)/2, (Wm-1)/2 in station loss."""
	xyz = mdl._grid_xyz()  # (D, Hm, Wm, 3)
	Hm, Wm = xyz.shape[1], xyz.shape[2]
	h_mid, w_mid = (Hm - 1) / 2.0, (Wm - 1) / 2.0
	h0, w0 = int(h_mid), int(w_mid)
	h1, w1 = min(h0 + 1, Hm - 1), min(w0 + 1, Wm - 1)
	fh, fw = h_mid - h0, w_mid - w0
	return ((1 - fh) * (1 - fw) * xyz[0, h0, w0]
	      + fh * (1 - fw) * xyz[0, h1, w0]
	      + (1 - fh) * fw * xyz[0, h0, w1]
	      + fh * fw * xyz[0, h1, w1])


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
	abs_flags: list[bool] = []
	for _cid, col in cols.items():
		if not isinstance(col, dict):
			print(f"[fit] _parse_corr_points: col {_cid} is not a dict", flush=True)
			continue
		md = col.get("metadata", {})
		if not isinstance(md, dict):
			md = {}
		is_abs = bool(md.get("winding_is_absolute", True))
		pts = col.get("points", {})
		if not isinstance(pts, dict):
			continue
		try:
			cid_i = int(_cid)
		except Exception:
			cid_i = -1
		n_pts = 0
		for _pid, pd in pts.items():
			if not isinstance(pd, dict):
				continue
			pv = pd.get("p", None)
			if not isinstance(pv, (list, tuple)) or len(pv) < 3:
				continue
			wa = pd.get("wind_a", None)
			if wa is None:
				print(f"[fit] WARNING: corr point {_pid} in collection {_cid} has no wind_a, skipping")
				continue
			try:
				pid_i = int(_pid)
			except Exception:
				pid_i = -1
			rows.append([float(pv[0]), float(pv[1]), float(pv[2]), float(wa)])
			cids.append(cid_i)
			pids.append(pid_i)
			abs_flags.append(is_abs)
			n_pts += 1
		print(f"[fit] _parse_corr_points: col {_cid}: {n_pts} points, "
			  f"absolute={is_abs}", flush=True)
	if not rows:
		print(f"[fit] _parse_corr_points: no valid points found after parsing", flush=True)
		return None
	pts_t = torch.tensor(rows, dtype=torch.float32, device=device)
	col_t = torch.tensor(cids, dtype=torch.int64, device=device)
	pid_t = torch.tensor(pids, dtype=torch.int64, device=device)
	abs_t = torch.tensor(abs_flags, dtype=torch.bool, device=device)
	n_abs = int(abs_t.sum().item())
	print(f"[fit] loaded {pts_t.shape[0]} corr_points from config "
		  f"({len(set(cids))} collections, {n_abs} absolute, {len(rows) - n_abs} relative)")
	return fit_data.CorrPoints3D(points_xyz_winda=pts_t, collection_idx=col_t,
								 point_ids=pid_t, is_absolute=abs_t)


def _build_parser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(
		prog="fit.py",
		description="3D fit entrypoint",
	)
	cli_data.add_args(p)
	cli_model.add_args(p)
	cli_opt.add_args(p)
	p.add_argument("--out-dir", default=None, help="Output directory for snapshots and debug")
	p.add_argument("--tifxyz-init", default=None, help="Initialize model from tifxyz directory instead of model.pt or new model")
	p.add_argument("--model-init", choices=("seed", "ext", "model"), default="seed",
		help="Initial model source: seed creates a new model, ext uses --tifxyz-init, model uses --model-input")
	p.add_argument("--window-size", type=int, default=None,
		help="Window size in fullres voxels for windowed tifxyz optimization (0 or omit = no windowing)")
	p.add_argument("--window-overlap", type=int, default=0,
		help="Overlap between windows in fullres voxels")
	p.add_argument("--progress", action="store_true", default=False,
		help="Print machine-readable PROGRESS lines to stdout")
	return p


def _compute_window_grid(
	H: int, W: int, mesh_step: int, window_size: int, overlap: int,
) -> list[tuple[int, int, int, int]]:
	"""Compute window tiles over a (H, W) vertex grid.

	window_size and overlap are in fullres voxels.
	Returns list of (h0, h1, w0, w1) in vertex indices.
	"""
	if overlap >= window_size:
		raise ValueError(f"overlap ({overlap}) must be less than window_size ({window_size})")
	win_verts = window_size // mesh_step + 1
	overlap_verts = overlap // mesh_step
	stride = max(1, win_verts - overlap_verts)
	windows = []
	h = 0
	while h < H:
		h1 = min(h + win_verts, H)
		w = 0
		while w < W:
			w1 = min(w + win_verts, W)
			windows.append((h, h1, w, w1))
			if w1 == W:
				break
			w += stride
		if h1 == H:
			break
		h += stride
	return windows


def main(argv: list[str] | None = None) -> int:
	if argv is None:
		argv = sys.argv[1:]

	_t_fit_total = _stage_start("total")
	_t = _stage_start("parse_config")
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
	_stage_done("parse_config", _t)

	print("data:", data_cfg)
	print("model:", model_cfg)
	print("opt:", opt_cfg)

	device = torch.device(data_cfg.device)
	model_init = str(getattr(args, "model_init", "seed")).strip().lower()
	if model_init not in {"seed", "ext", "model"}:
		raise ValueError(f"invalid model-init '{model_init}' (expected seed, ext, or model)")

	# Probe preprocessed data for scaledown and volume extent (in base/VC3D coords)
	_t = _stage_start("probe_preprocessed_data")
	prep_params = fit_data.get_preprocessed_params(str(data_cfg.input))
	source_to_base = float(prep_params.get("source_to_base", 1.0))
	# Model scaledown in base coords = channel_scaledown * source_to_base
	scaledown = float(prep_params["scaledown"]) * source_to_base
	volume_extent_fullres = prep_params.get("volume_extent_fullres")
	print(f"[fit] scaledown={scaledown} (source_sd={prep_params['scaledown']} "
		  f"source_to_base={source_to_base}) volume_extent={volume_extent_fullres}", flush=True)
	_stage_done("probe_preprocessed_data", _t)

	# --- Init from seed (new model only) ---
	_t = _stage_start("derive_initial_model_params")
	if model_init == "seed":
		if getattr(args, "tifxyz_init", None):
			raise ValueError("model-init=seed must not set --tifxyz-init; tifxyz can only be used as external_surfaces")
		if model_cfg.model_input is not None:
			raise ValueError("model-init=seed must not set --model-input")
		missing_seed = []
		if data_cfg.seed is None:
			missing_seed.append("--seed")
		if data_cfg.model_h is None:
			missing_seed.append("--model-h")
		if missing_seed:
			raise ValueError(f"model-init=seed requires {', '.join(missing_seed)}")
	elif model_init == "ext":
		if not getattr(args, "tifxyz_init", None):
			raise ValueError("model-init=ext requires --tifxyz-init")
		if model_cfg.model_input is not None:
			raise ValueError("model-init=ext must not set --model-input")
	elif model_init == "model":
		if model_cfg.model_input is None:
			raise ValueError("model-init=model requires --model-input")
		if getattr(args, "tifxyz_init", None):
			raise ValueError("model-init=model must not set --tifxyz-init; tifxyz can only be used as external_surfaces")

	if model_init == "seed" and data_cfg.seed is not None:
		model_cfg = dataclasses.replace(model_cfg, z_center=float(data_cfg.seed[2]))
		print(f"[fit] cylinder_seed from seed: x={float(data_cfg.seed[0]):.1f} "
			  f"y={float(data_cfg.seed[1]):.1f} z={float(data_cfg.seed[2]):.1f}",
			  flush=True)

	# --- Size mesh from model_h only for the umbilicus tube experiment ---
	if model_init == "seed" and data_cfg.model_h is not None:
		tube_z_step = 1000.0
		auto_mesh_w = 20
		auto_mesh_h = max(2, int(math.ceil(float(data_cfg.model_h) / tube_z_step)) + 1)
		auto_depth = 1
		actual_z_step = float(data_cfg.model_h) / float(max(1, auto_mesh_h - 1))

		model_cfg = dataclasses.replace(model_cfg, depth=auto_depth, mesh_h=auto_mesh_h, mesh_w=auto_mesh_w)
		print(f"[fit] model size: depth={auto_depth} mesh_h={auto_mesh_h} mesh_w={auto_mesh_w} "
			  f"z_step={actual_z_step:.1f} z_step_target={tube_z_step:.1f} "
			  f"(umbilicus tube search grid; final mesh bake uses model-w/model-h/mesh-step)", flush=True)
	_stage_done("derive_initial_model_params", _t)

	# --- Windowed tifxyz mode ---
	tifxyz_init = getattr(args, "tifxyz_init", None)
	window_size = getattr(args, "window_size", None) or 0
	window_overlap = getattr(args, "window_overlap", 0)
	if model_init != "ext" and window_size > 0:
		raise ValueError("windowed optimization currently requires model-init=ext")

	if model_init == "ext" and tifxyz_init and window_size > 0:
		from tifxyz_io import load_tifxyz
		import fit2tifxyz as _f2t
		import json as _json

		# Load full tifxyz to CPU (save GPU mem)
		full_xyz, full_valid, full_meta = load_tifxyz(tifxyz_init, device="cpu")
		H_full, W_full, _ = full_xyz.shape
		mesh_step = model_cfg.mesh_step
		scale = full_meta.get("scale")
		if scale is not None and isinstance(scale, list) and len(scale) >= 1 and float(scale[0]) > 0:
			mesh_step = max(1, int(round(1.0 / float(scale[0]))))

		# Get offset from external_surfaces config
		ext_surfaces_cfg = cfg.pop("external_surfaces", None)
		offset_val = 1.0
		if isinstance(ext_surfaces_cfg, list) and ext_surfaces_cfg:
			offset_val = float(ext_surfaces_cfg[0].get("offset", 1.0))
		ext_margin = max(4, int(2 * abs(offset_val) / mesh_step) + 2)

		# Parse stages and channel skipping (shared across windows)
		cfg.pop("corr_points", None)
		cfg.pop("args", None)
		cfg.pop("voxel_size_um", None)
		cfg.pop("external_surfaces", None)
		cfg.pop("tifxyz_data", None)
		cfg.pop("offset_value", None)
		stages = optimizer.load_stages_cfg(
			cfg,
			init_mode=model_cfg.init_mode if model_init == "seed" else None,
		)

		windows = _compute_window_grid(H_full, W_full, mesh_step, window_size, window_overlap)
		n_windows = len(windows)
		overlap_verts = window_overlap // mesh_step
		print(f"[fit] windowed mode: {n_windows} windows, window_size={window_size} "
			  f"overlap={window_overlap} mesh_step={mesh_step} grid={H_full}x{W_full}",
			  flush=True)

		# Output directory for window tifxyz exports
		output_dir = model_cfg.model_output
		if output_dir is not None:
			output_dir = str(Path(output_dir).parent)
		elif _out_dir is not None:
			output_dir = _out_dir
		else:
			raise ValueError("windowed mode requires --model-output or --out-dir")
		Path(output_dir).mkdir(parents=True, exist_ok=True)

		voxel_size_um = fit_config.get("voxel_size_um")

		for wi, (h0, h1, w0, w1) in enumerate(windows):
			print(f"\n[fit] === window {wi+1}/{n_windows}: rows [{h0}:{h1}], cols [{w0}:{w1}] "
				  f"({h1-h0}x{w1-w0} verts) ===", flush=True)

			# Crop tifxyz to window
			crop_xyz = full_xyz[h0:h1, w0:w1].to(device)
			crop_valid = full_valid[h0:h1, w0:w1].to(device)

			# Create model from crop
			mdl = model.Model3D.from_tifxyz_crop(
				crop_xyz, crop_valid, device=device, mesh_step=mesh_step,
				winding_step=model_cfg.winding_step,
				subsample_mesh=model_cfg.subsample_mesh,
				subsample_winding=model_cfg.subsample_winding,
			)

			# Crop external surface with margin for ray intersection at boundaries
			eh0 = max(0, h0 - ext_margin)
			eh1 = min(H_full, h1 + ext_margin)
			ew0 = max(0, w0 - ext_margin)
			ew1 = min(W_full, w1 + ext_margin)
			ext_xyz = full_xyz[eh0:eh1, ew0:ew1].to(device)
			ext_valid = full_valid[eh0:eh1, ew0:ew1].to(device)
			ext_idx = mdl.add_external_surface(ext_xyz, valid=ext_valid, offset=offset_val)
			# ext→model mapping: ext corner r → model grid r + h_off
			# ext grid 0 = fullres eh0, model grid 0 = fullres h0
			# so model_h = r + (eh0 - h0)
			mdl._ext_conn_offsets[ext_idx][0] = float(eh0 - h0)
			mdl._ext_conn_offsets[ext_idx][1] = float(ew0 - w0)

			# Streaming data loader for this window
			def _streaming_skip_channels(needed_channels: set[str]) -> set[str]:
				optional = {"cos", "pred_dt"}
				return optional - set(needed_channels)

			def _streaming_loaded_channels(d: fit_data.FitData3D) -> set[str]:
				if not d.sparse_caches:
					return set()
				return {
					ch
					for cache in d.sparse_caches.values()
					for ch in cache.channels
				}

			def _load_streaming_win(needed_channels: set[str]) -> fit_data.FitData3D:
				d = fit_data.load_3d_streaming(
					path=str(data_cfg.input),
					device=device,
					sparse_prefetch_backend=data_cfg.sparse_prefetch_backend,
					skip_channels=_streaming_skip_channels(needed_channels),
				)
				Z, Y, X = d.size
				sx, sy, sz = d.spacing
				volume_extent = (
					d.origin_fullres[0], d.origin_fullres[1], d.origin_fullres[2],
					d.origin_fullres[0] + (X - 1) * sx,
					d.origin_fullres[1] + (Y - 1) * sy,
					d.origin_fullres[2] + (Z - 1) * sz,
				)
				mdl.params = dataclasses.replace(mdl.params, volume_extent=volume_extent)
				return d

			def _ensure_data_win(data: fit_data.FitData3D | None, needed_channels: set[str]) -> fit_data.FitData3D:
				if data is None:
					return _load_streaming_win(needed_channels)
				loaded = _streaming_loaded_channels(data)
				required = {"grad_mag", "nx", "ny"} | set(needed_channels)
				if not required.issubset(loaded) or (loaded & {"cos", "pred_dt"}) != (required & {"cos", "pred_dt"}):
					return _load_streaming_win(needed_channels)
				return data

			data = _ensure_data_win(None, set())

			# Progress wrapper: prefix window index, scale overall progress
			def _make_progress(wi_=wi, n_=n_windows):
				def _progress_win(*, step: int, total: int, loss: float, **kw: object) -> None:
					if progress_enabled:
						inner = float(kw.get("overall_progress", 0.0))
						overall = (wi_ + inner) / n_
						stage_name = kw.get("stage_name", "")
						print(f"PROGRESS {step} {total} {loss:.6f} win={wi_+1}/{n_} "
							  f"overall={overall:.3f} {stage_name}", flush=True)
				return _progress_win

			opt_loss_dir.set_mask_zero_normals(opt_cfg.normal_mask_zero)

			# Seed from center of model grid (matches h_mid/w_mid in station loss)
			center_pt = _grid_center(mdl)
			win_seed = (float(center_pt[0]), float(center_pt[1]), float(center_pt[2]))
			print(f"[fit] window seed: ({win_seed[0]:.0f}, {win_seed[1]:.0f}, {win_seed[2]:.0f})",
				  flush=True)

			optimizer.optimize(
				model=mdl,
				data=data,
				stages=stages,
				snapshot_interval=0,
				snapshot_fn=lambda **kw: None,
				progress_fn=_make_progress(),
				ensure_data_fn=_ensure_data_win,
				seed_xyz=win_seed,
				out_dir=str(Path(output_dir) / f"window_{wi:04d}"),
			)

			# Export this window's tifxyz
			mesh = mdl.mesh_coarse()  # (3, 1, Hm, Wm)
			mesh_np = mesh.detach().cpu().numpy()
			Hm, Wm = mesh_np.shape[2], mesh_np.shape[3]
			x_out = mesh_np[0, 0]  # (Hm, Wm)
			y_out = mesh_np[1, 0]
			z_out = mesh_np[2, 0]
			meta_scale = 1.0 / float(mesh_step)

			win_name = f"window_{wi:04d}.tifxyz"
			win_dir = Path(output_dir) / win_name
			area = _f2t._get_area(x_out, y_out, z_out, float(mesh_step),
								  float(voxel_size_um) if voxel_size_um else None)
			_f2t._write_tifxyz(
				out_dir=win_dir, x=x_out, y=y_out, z=z_out,
				scale=meta_scale, area=area,
			)
			# Add window metadata to meta.json
			meta_path = win_dir / "meta.json"
			meta = _json.loads(meta_path.read_text(encoding="utf-8"))
			meta["window_index"] = wi
			meta["window_origin_verts"] = [h0, w0]
			meta["window_size_verts"] = [h1 - h0, w1 - w0]
			meta["source_grid_size_verts"] = [H_full, W_full]
			meta["overlap_verts"] = overlap_verts
			meta_path.write_text(_json.dumps(meta, indent=2) + "\n", encoding="utf-8")

			_f2t._print_area(area)
			print(f"[fit] exported {win_name}", flush=True)

			# Free GPU memory before next window
			del mdl, data, crop_xyz, crop_valid, ext_xyz, ext_valid, mesh, mesh_np
			if device.type == "cuda":
				torch.cuda.empty_cache()

		print(f"\n[fit] windowed mode complete: {n_windows} windows exported to {output_dir}",
			  flush=True)
		return 0

	# --- Construct / load model (before data, so we can compute bbox) ---
	_t = _stage_start("construct_model")
	if model_init == "ext":
		mdl = model.Model3D.from_tifxyz(
			tifxyz_init, device=device,
			mesh_step=model_cfg.mesh_step,
			winding_step=model_cfg.winding_step,
			subsample_mesh=model_cfg.subsample_mesh,
			subsample_winding=model_cfg.subsample_winding,
		)
		print(f"[fit] initialized from tifxyz: {tifxyz_init}", flush=True)
	elif model_init == "seed":
		print(f"[fit] model-init=seed: constructing model from seed", flush=True)
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
			init_mode=model_cfg.init_mode,
			volume_extent=None,
			pyramid_d=model_cfg.pyramid_d,
		)
		mdl.init_cylinder_seed(
			seed=tuple(float(v) for v in data_cfg.seed),
			model_w=float(data_cfg.model_w) if data_cfg.model_w is not None else 0.0,
			model_h=float(data_cfg.model_h),
			volume_extent_fullres=volume_extent_fullres,
		)
	else:
		print(f"[fit] model-init=model: loading checkpoint {model_cfg.model_input}", flush=True)
		st = torch.load(model_cfg.model_input, map_location=device, weights_only=False)
		mdl = model.Model3D.from_checkpoint(st, device=device)

	print(f"Model3D: depth={mdl.depth} mesh_h={mdl.mesh_h} mesh_w={mdl.mesh_w} "
		  f"cylinder_enabled={getattr(mdl, 'cylinder_enabled', False)}")
	_stage_done("construct_model", _t)

	# Load external reference surfaces
	_t = _stage_start("load_external_surfaces")
	ext_surfaces_cfg = cfg.pop("external_surfaces", None)
	if isinstance(ext_surfaces_cfg, list) and ext_surfaces_cfg:
		from tifxyz_io import load_tifxyz
		for es in ext_surfaces_cfg:
			es_path = str(es["path"])
			es_offset = float(es.get("offset", 1.0))
			xyz_ext, valid_ext, meta_ext = load_tifxyz(es_path, device=device)
			idx = mdl.add_external_surface(xyz_ext, valid=valid_ext, offset=es_offset)
			print(f"[fit] external surface {idx}: path={es_path} offset={es_offset} "
				  f"shape={tuple(xyz_ext.shape)} valid={int(valid_ext.sum())}/{valid_ext.numel()}", flush=True)
	_stage_done("load_external_surfaces", _t)

	# Parse correction points from config (injected by VC3D)
	_t = _stage_start("parse_corr_points")
	corr_points_obj = cfg.pop("corr_points", None)
	corr_points_3d: fit_data.CorrPoints3D | None = None
	if isinstance(corr_points_obj, dict):
		corr_points_3d = _parse_corr_points(corr_points_obj, device)
	else:
		print(f"[fit] corr_points: not found in config (type={type(corr_points_obj).__name__})", flush=True)
	_stage_done("parse_corr_points", _t)

	# Strip non-stage keys before parsing stages
	_t = _stage_start("load_optimizer_stages")
	cfg.pop("args", None)
	cfg.pop("voxel_size_um", None)
	cfg.pop("external_surfaces", None)
	cfg.pop("tifxyz_data", None)
	cfg.pop("offset_value", None)
	stages = optimizer.load_stages_cfg(
		cfg,
		init_mode=model_cfg.init_mode if model_init == "seed" else None,
	)
	_stage_done("load_optimizer_stages", _t)

	# --- Streaming data loader ---
	def _streaming_skip_channels(needed_channels: set[str]) -> set[str]:
		optional = {"cos", "pred_dt"}
		return optional - set(needed_channels)

	def _streaming_loaded_channels(d: fit_data.FitData3D) -> set[str]:
		if not d.sparse_caches:
			return set()
		return {
			ch
			for cache in d.sparse_caches.values()
			for ch in cache.channels
		}

	def _load_streaming(needed_channels: set[str]) -> fit_data.FitData3D:
		d = fit_data.load_3d_streaming(
			path=str(data_cfg.input),
			device=device,
			sparse_prefetch_backend=data_cfg.sparse_prefetch_backend,
			skip_channels=_streaming_skip_channels(needed_channels),
		)
		Z, Y, X = d.size
		# Volume extent covers the full zarr volume
		sx, sy, sz = d.spacing
		volume_extent = (
			d.origin_fullres[0],
			d.origin_fullres[1],
			d.origin_fullres[2],
			d.origin_fullres[0] + (X - 1) * sx,
			d.origin_fullres[1] + (Y - 1) * sy,
			d.origin_fullres[2] + (Z - 1) * sz,
		)
		mdl.params = dataclasses.replace(mdl.params, volume_extent=volume_extent)
		if corr_points_3d is not None:
			d = dataclasses.replace(d, corr_points=corr_points_3d)
		if data_cfg.winding_volume is not None:
			wv_t, wv_min, wv_max = fit_data.load_winding_volume(
				path=data_cfg.winding_volume, device=device,
				crop=None, downscale=scaledown)
			d = dataclasses.replace(d, winding_volume=wv_t,
						winding_min=wv_min, winding_max=wv_max)
		return d

	def _ensure_data(data: fit_data.FitData3D | None, needed_channels: set[str]) -> fit_data.FitData3D:
		if data is None:
			return _load_streaming(needed_channels)
		loaded = _streaming_loaded_channels(data)
		required = {"grad_mag", "nx", "ny"} | set(needed_channels)
		if not required.issubset(loaded) or (loaded & {"cos", "pred_dt"}) != (required & {"cos", "pred_dt"}):
			d = _load_streaming(needed_channels)
			if data.corr_points is not None:
				d = dataclasses.replace(d, corr_points=data.corr_points)
			if data.winding_volume is not None:
				d = dataclasses.replace(
					d,
					winding_volume=data.winding_volume,
					winding_min=data.winding_min,
					winding_max=data.winding_max,
				)
			return d
		# Streaming covers full volume — no border checks needed
		return data

	_t = _stage_start("load_data")
	data = _ensure_data(None, set())
	_stage_done("load_data", _t)

	if getattr(mdl, "cylinder_enabled", False) and hasattr(mdl, "prepare_umbilicus_tube_init"):
		_t = _stage_start("prepare_umbilicus_tube_init")
		mdl.prepare_umbilicus_tube_init(data)
		_stage_done("prepare_umbilicus_tube_init", _t)

	# Print loaded data summary
	Z, Y, X = data.size
	if data.sparse_caches:
		_cache_table_bytes = sum(c.chunk_table.nbytes for c in data.sparse_caches.values())
		print(f"[fit] data (streaming): vol_size=({Z},{Y},{X}) origin={data.origin_fullres} "
			  f"spacing={data.spacing} groups={list(data.sparse_caches.keys())} "
			  f"table_mem={_cache_table_bytes / 2**20:.1f} MiB", flush=True)
	else:
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
	_t = _stage_start("initial_mesh_stats")
	with torch.no_grad():
		xyz = mdl._grid_xyz()
		mn = xyz.amin(dim=(0, 1, 2)).cpu().numpy().tolist()
		mx = xyz.amax(dim=(0, 1, 2)).cpu().numpy().tolist()
		mean = xyz.mean(dim=(0, 1, 2)).cpu().numpy().tolist()
		print(f"initial mesh: mean={[round(v, 1) for v in mean]} "
			  f"min={[round(v, 1) for v in mn]} max={[round(v, 1) for v in mx]}")
	_stage_done("initial_mesh_stats", _t)

	def _save_model(path: str) -> None:
		st = dict(mdl.state_dict())
		# Store flat mesh instead of pyramid levels
		ms_keys = [k for k in st if k.startswith("mesh_ms.")]
		for k in ms_keys:
			del st[k]
		st.pop("cyl_params", None)
		if getattr(mdl, "cylinder_enabled", False) and getattr(mdl, "cyl_shell_mode", False):
			st.pop("conn_offsets", None)
			st.pop("amp", None)
			st.pop("bias", None)
		elif getattr(mdl, "cylinder_enabled", False) and "conn_offsets" in st:
			st["conn_offsets"] = torch.zeros_like(st["conn_offsets"])
		with torch.no_grad():
			st["mesh_flat"] = mdl.mesh_flat_for_save(data=data)
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

	opt_loss_dir.set_mask_zero_normals(opt_cfg.normal_mask_zero)

	# Run optimization
	_t = _stage_start("prepare_optimization")
	seed_xyz = tuple(float(v) for v in data_cfg.seed) if data_cfg.seed is not None else None
	# tifxyz init: always use model grid center as seed (overrides CLI seed)
	if model_init == "ext":
		center_pt = _grid_center(mdl)
		seed_xyz = (float(center_pt[0]), float(center_pt[1]), float(center_pt[2]))
		print(f"[fit] tifxyz seed: ({seed_xyz[0]:.0f}, {seed_xyz[1]:.0f}, {seed_xyz[2]:.0f})",
			  flush=True)
	# Re-optimize from checkpoint: derive seed from model grid center
	if seed_xyz is None and model_init == "model":
		center_pt = _grid_center(mdl)
		seed_xyz = (float(center_pt[0]), float(center_pt[1]), float(center_pt[2]))
		print(f"[fit] checkpoint seed (grid center): ({seed_xyz[0]:.0f}, {seed_xyz[1]:.0f}, {seed_xyz[2]:.0f})",
			  flush=True)
	_stage_done("prepare_optimization", _t)
	_t = _stage_start("optimizer")
	optimizer.optimize(
		model=mdl,
		data=data,
		stages=stages,
		snapshot_interval=opt_cfg.snapshot_interval,
		snapshot_fn=_snapshot,
		progress_fn=_progress,
		ensure_data_fn=_ensure_data,
		seed_xyz=seed_xyz,
		out_dir=_out_dir,
	)
	_stage_done("optimizer", _t)

	if device.type == "cuda":
		peak_gb = torch.cuda.max_memory_allocated(device) / 2**30
		print(f"[fit] peak GPU memory: {peak_gb:.2f} GiB", flush=True)

	# Save final model
	if model_cfg.model_output is not None:
		_t = _stage_start("save_model_output")
		_save_model(str(model_cfg.model_output))
		print(f"[fit] saved model to {model_cfg.model_output}")
		_stage_done("save_model_output", _t)

	# Save snapshot
	if _out_dir is not None:
		_t = _stage_start("save_final_snapshot")
		out = Path(_out_dir)
		out.mkdir(parents=True, exist_ok=True)
		_save_model(str(out / "model_final.pt"))
		_stage_done("save_final_snapshot", _t)

	# Export tifxyz
	model_out = model_cfg.model_output
	if model_out is None and _out_dir is not None:
		model_out = str(Path(_out_dir) / "model_final.pt")
	if model_out is not None and _out_dir is not None:
		_t = _stage_start("export_tifxyz")
		import fit2tifxyz
		export_dir = str(Path(_out_dir) / "tifxyz")
		tifxyz_argv = ["--input", str(model_out), "--output", export_dir]
		if getattr(mdl, "cyl_shell_completed", None):
			tifxyz_argv.append("--single-segment")
		voxel_size_um = cfg.get("voxel_size_um")
		if voxel_size_um is not None:
			tifxyz_argv += ["--voxel-size-um", str(float(voxel_size_um))]
		fit2tifxyz.main(tifxyz_argv)
		_stage_done("export_tifxyz", _t)

	_stage_done("total", _t_fit_total)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
