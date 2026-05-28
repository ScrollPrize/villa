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
import opt_loss_step
import optimizer


def _stage_start(label: str) -> float:
	return 0.0


def _stage_done(label: str, t0: float) -> None:
	return None


def _truthy_config_bool(value: object) -> bool:
	if isinstance(value, bool):
		return value
	if isinstance(value, (int, float)):
		return bool(value)
	if isinstance(value, str):
		return value.strip().lower() in {"1", "true", "yes", "on"}
	return False


def _require_torch_device_available(device: torch.device) -> None:
	if device.type != "cuda":
		return
	if not torch.cuda.is_available():
		raise RuntimeError(
			"CUDA device was requested, but PyTorch cannot access an NVIDIA GPU. "
			"Expose the NVIDIA driver/device nodes to this process (for example "
			"/dev/nvidia*, Docker --gpus all, or the equivalent sandbox GPU "
			"passthrough). Refusing to continue because falling back to CPU would "
			"make fit smoke/perf runs misleading."
		)
	count = int(torch.cuda.device_count())
	if device.index is not None and int(device.index) >= count:
		raise RuntimeError(
			f"CUDA device {device} was requested, but PyTorch reports only "
			f"{count} visible CUDA device(s)."
		)


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


def _optimization_seed_xyz(
	*,
	model_init: str,
	config_seed: tuple[float, float, float] | None,
	mdl: "model.Model3D",
) -> tuple[float, float, float] | None:
	"""Return the station seed used during optimization."""
	if model_init in {"ext", "model"}:
		center_pt = _grid_center(mdl)
		return (float(center_pt[0]), float(center_pt[1]), float(center_pt[2]))
	return config_seed


def _first_cylinder_stage_model_step(stages: list[optimizer.Stage]) -> float | None:
	for stage in stages:
		if "cyl_params" not in stage.global_opt.params:
			continue
		args = stage.global_opt.args or {}
		value = args.get(optimizer.CYLINDER_STAGE_STEP_ARG)
		if value is None:
			return None
		value_f = float(value)
		return value_f if value_f > 0.0 else None
	return None


def _apply_cylinder_prepare_model_step(mdl: "model.Model3D", model_step: float | None) -> None:
	if model_step is None:
		return
	step = float(model_step)
	if hasattr(mdl, "cyl_shell_width_target_step"):
		mdl.cyl_shell_width_target_step = step
	if hasattr(mdl, "cyl_shell_current_width_step"):
		mdl.cyl_shell_current_width_step = step
	if hasattr(mdl, "cyl_shell_z_step"):
		mdl.cyl_shell_z_step = step
	if hasattr(mdl, "cyl_shell_current_height_step"):
		mdl.cyl_shell_current_height_step = step


def _require_manifest_init_shell_dir(prep_params: dict) -> str:
	value = prep_params.get("init_shell_dir", None)
	if value is None:
		raise ValueError("shell-dir-crop init requires .lasagna.json key 'init_shell_dir'")
	if not isinstance(value, str) or not value.strip():
		raise ValueError("shell-dir-crop init requires non-empty string .lasagna.json key 'init_shell_dir'")
	return str(value)


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
	p.add_argument("--model-init", choices=("seed", "ext", "model", "flatten"), default="seed",
		help="Initial model source: seed creates a new model, ext uses --tifxyz-init, model uses --model-input, flatten optimizes one external tifxyz inverse map")
	p.add_argument("--flatten-solver", choices=("torch", "inverse", "forward"), default="torch",
		help="Flatten solver variant for model-init=flatten: torch/inverse keeps the existing inverse-map Adam path; forward optimizes source-vertex UVs and inverts at export")
	p.add_argument("--approval-inpaint", action=argparse.BooleanOptionalAction, default=False,
		help="Use selected approval mask/tifxyz data to inpaint the seed-region setup")
	p.add_argument("--approval-inpaint-corr-spacing", type=float, default=None,
		help="Correction point spacing for approval inpaint (default: --mesh-step)")
	p.add_argument("--approval-inpaint-padding-frac", type=float, default=0.25,
		help="Per-side model extent padding for approval inpaint")
	p.add_argument("--approval-inpaint-output-mask", action=argparse.BooleanOptionalAction, default=False,
		help="Store selected approval-inpaint cell as an export mask")
	p.add_argument("--approval-inpaint-output-mask-dilate", type=int, default=3,
		help="Output-mask dilation radius in exported mesh vertices")
	p.add_argument("--approval-inpaint-tifxyz", default=None, help=argparse.SUPPRESS)
	p.add_argument("--progress", action="store_true", default=False,
		help="Print machine-readable PROGRESS lines to stdout")
	return p


def _dummy_flatten_data() -> fit_data.FitData3D:
	return fit_data.FitData3D(
		cos=None,
		grad_mag=None,
		nx=None,
		ny=None,
		pred_dt=None,
		corr_points=None,
		winding_volume=None,
		origin_fullres=(0.0, 0.0, 0.0),
		spacing=(1.0, 1.0, 1.0),
		channel_spacing=None,
		_vol_size=(1, 1, 1),
		sparse_caches=None,
	)


def _mesh_step_from_tifxyz_meta(meta: dict, fallback: int) -> int:
	scale = meta.get("scale") if isinstance(meta, dict) else None
	if isinstance(scale, list) and scale and float(scale[0]) > 0.0:
		return max(1, int(round(1.0 / float(scale[0]))))
	return max(1, int(fallback))


def _scale_from_tifxyz_meta(meta: dict, mesh_step: int) -> float:
	scale = meta.get("scale") if isinstance(meta, dict) else None
	if isinstance(scale, list) and scale and float(scale[0]) > 0.0:
		return float(scale[0])
	return 1.0 / float(max(1, int(mesh_step)))


def _save_flatten_model(path: str, *, mdl: model.Model3D, data: fit_data.FitData3D, fit_config: dict) -> None:
	st = dict(mdl.state_dict())
	for k in [k for k in st if k.startswith("mesh_ms.")]:
		del st[k]
	with torch.no_grad():
		map_yx, xyz, point_mask, _quad_mask = mdl._flatten_sample_current()
		sentinel = torch.full_like(xyz, -1.0)
		xyz = torch.where(point_mask.unsqueeze(0).unsqueeze(-1), xyz, sentinel)
		st["mesh_flat"] = xyz.permute(3, 0, 1, 2).detach().cpu()
		st["flatten_map_flat"] = map_yx.detach().cpu()
		st["flatten_point_mask"] = point_mask.detach().cpu()
	st["_model_params_"] = asdict(mdl.params)
	st["_fit_config_"] = fit_config
	torch.save(st, path)


def _export_flatten_result(
	*,
	mdl: model.Model3D,
	data: fit_data.FitData3D,
	out_dir: Path,
	scale: float,
	voxel_size_um: float | None,
	fit_config: dict,
	model_source: Path | None,
) -> None:
	import numpy as np
	import fit2tifxyz

	out_dir.mkdir(parents=True, exist_ok=True)
	with torch.no_grad():
		_map_yx, xyz, point_mask, _quad_mask = mdl._flatten_sample_current()
	xyz_np = xyz[0].detach().cpu().numpy().astype(np.float32, copy=False)
	mask_np = point_mask.detach().cpu().numpy().astype(bool, copy=False)
	x = np.where(mask_np, xyz_np[..., 0], -1.0).astype(np.float32, copy=False)
	y = np.where(mask_np, xyz_np[..., 1], -1.0).astype(np.float32, copy=False)
	z = np.where(mask_np, xyz_np[..., 2], -1.0).astype(np.float32, copy=False)
	mesh_step = 1.0 / float(scale) if float(scale) > 0.0 else float(mdl.params.mesh_step)
	area = fit2tifxyz._get_area(x, y, z, mesh_step, voxel_size_um)
	fit2tifxyz._write_tifxyz(
		out_dir=out_dir / "flatten.tifxyz",
		x=x,
		y=y,
		z=z,
		scale=scale,
		model_source=model_source,
		fit_config=fit_config,
		area=area,
	)
	fit2tifxyz._print_area(area)


def _run_flatten_mode(
	*,
	cfg: dict,
	fit_config: dict,
	args: argparse.Namespace,
	model_cfg: cli_model.ModelConfig,
	opt_cfg: cli_opt.OptConfig,
	progress_enabled: bool,
	out_dir: str | None,
) -> int:
	ext_surfaces_cfg = cfg.get("external_surfaces", None)
	if not isinstance(ext_surfaces_cfg, list) or len(ext_surfaces_cfg) != 1:
		raise ValueError("model-init=flatten requires exactly one external_surfaces entry")
	ext0 = ext_surfaces_cfg[0]
	if not isinstance(ext0, dict) or not ext0.get("path"):
		raise ValueError("model-init=flatten external_surfaces[0] requires path")
	if getattr(args, "tifxyz_init", None):
		raise ValueError("model-init=flatten uses external_surfaces[0], not --tifxyz-init")
	if model_cfg.model_input is not None:
		raise ValueError("model-init=flatten must not set --model-input")

	device = torch.device(str(getattr(args, "device", "cuda")))
	from tifxyz_io import load_tifxyz
	xyz, valid, meta = load_tifxyz(str(ext0["path"]), device=device)
	mesh_step = _mesh_step_from_tifxyz_meta(meta, model_cfg.mesh_step)
	scale = _scale_from_tifxyz_meta(meta, mesh_step)

	stage_cfg = copy.deepcopy(cfg)
	for key in ("external_surfaces", "tifxyz", "voxel_size_um", "corr_points"):
		stage_cfg.pop(key, None)
	stages = optimizer.load_stages_cfg(stage_cfg, init_mode=None)
	flatten_args: dict[str, object] = {}
	if isinstance(cfg.get("args"), dict):
		flatten_args.update(cfg["args"])
	if stages:
		flatten_args.update(stages[0].global_opt.args or {})
	flatten_solver_raw = flatten_args.get(
		"flatten_solver",
		flatten_args.get("flatten-solver", getattr(args, "flatten_solver", "torch")),
	)
	flatten_direction = model.Model3D._normalize_flatten_direction(str(flatten_solver_raw))
	flatten_output_margin = float(flatten_args.get(
		"flatten_output_margin",
		flatten_args.get("flatten_forward_output_margin", 0.10),
	))
	filter_source_angles = _truthy_config_bool(flatten_args.get("flatten_filter_source_angles", True))
	filter_angle_deg = float(flatten_args.get("flatten_filter_angle_deg", 90.0))
	filter_radius = int(flatten_args.get("flatten_filter_radius", 2))
	mdl = model.Model3D.from_flatten_tifxyz_crop(
		xyz,
		valid,
		device=device,
		mesh_step=mesh_step,
		winding_step=model_cfg.winding_step,
		subsample_mesh=model_cfg.subsample_mesh,
		subsample_winding=model_cfg.subsample_winding,
		flatten_filter_source_angles=filter_source_angles,
		flatten_filter_angle_deg=filter_angle_deg,
		flatten_filter_radius=filter_radius,
		flatten_direction=flatten_direction,
		flatten_output_margin=flatten_output_margin,
	)
	data = _dummy_flatten_data()

	print("data: flatten-only (no volume input)")
	print("model:", model_cfg)
	print("opt:", opt_cfg)
	print(
		f"[fit] model-init=flatten solver={flatten_direction} source={ext0['path']} "
		f"shape={tuple(xyz.shape)} valid={int(valid.sum())}/{valid.numel()} "
		f"model_shape={mdl.mesh_h}x{mdl.mesh_w} "
		f"mesh_step={mesh_step} target_step={float(mdl.flatten_target_step.detach().cpu()):.6g}",
		flush=True,
	)
	filter_stats = getattr(mdl, "flatten_source_filter_stats", {})
	if filter_source_angles:
		print(
			f"[fit] flatten source angle filter: angle>{filter_angle_deg:.4g} radius={max(0, filter_radius)} "
			f"bad_pairs={int(filter_stats.get('bad_pairs', 0.0))} "
			f"bad_cells={int(filter_stats.get('bad_cells', 0.0))} "
			f"dilated={int(filter_stats.get('bad_cells_dilated', 0.0))} "
			f"cell_valid={int(filter_stats.get('cell_valid_after', 0.0))}/"
			f"{int(filter_stats.get('cell_valid_before', 0.0))}",
			flush=True,
		)

	def _snapshot(*, stage: str, step: int, loss: float, data, res=None) -> None:
		if out_dir is None:
			return
		out = Path(out_dir)
		out.mkdir(parents=True, exist_ok=True)
		snaps = out / "model_snapshots"
		snaps.mkdir(parents=True, exist_ok=True)
		_save_flatten_model(str(snaps / f"model_{stage}_{step:06d}.pt"), mdl=mdl, data=data, fit_config=fit_config)

	def _progress(*, step: int, total: int, loss: float, **_kw: object) -> None:
		if progress_enabled:
			print(f"PROGRESS {step} {total} {loss:.6f}", flush=True)

	with torch.no_grad():
		map_yx, xyz0, point_mask, quad_mask = mdl._flatten_sample_current()
		print(
			f"initial flatten: map_shape={tuple(map_yx.shape)} "
			f"point_valid={int(point_mask.sum())}/{point_mask.numel()} "
			f"quad_valid={int(quad_mask.sum())}/{quad_mask.numel()}",
			flush=True,
		)

	optimizer.optimize(
		model=mdl,
		data=data,
		stages=stages,
		snapshot_interval=opt_cfg.snapshot_interval,
		snapshot_fn=_snapshot,
		progress_fn=_progress,
		ensure_data_fn=None,
		seed_xyz=None,
		out_dir=out_dir,
	)

	if device.type == "cuda":
		peak_gb = torch.cuda.max_memory_allocated(device) / 2**30
		print(f"[fit] peak GPU memory: {peak_gb:.2f} GiB", flush=True)

	model_out: str | None = model_cfg.model_output
	if model_out is not None:
		_save_flatten_model(str(model_out), mdl=mdl, data=data, fit_config=fit_config)
		print(f"[fit] saved model to {model_out}")
	if out_dir is not None:
		out = Path(out_dir)
		out.mkdir(parents=True, exist_ok=True)
		final_path = out / "model_final.pt"
		_save_flatten_model(str(final_path), mdl=mdl, data=data, fit_config=fit_config)
		model_source = Path(model_out) if model_out is not None else final_path
		_export_flatten_result(
			mdl=mdl,
			data=data,
			out_dir=out / "tifxyz",
			scale=scale,
			voxel_size_um=(None if cfg.get("voxel_size_um") is None else float(cfg.get("voxel_size_um"))),
			fit_config=fit_config,
			model_source=model_source,
		)
	return 0


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

	model_cfg = cli_model.from_args(args)
	opt_cfg = cli_opt.from_args(args)
	progress_enabled = bool(args.progress)
	_out_dir = args.out_dir
	_stage_done("parse_config", _t)

	model_init = str(getattr(args, "model_init", "seed")).strip().lower()
	if model_init not in {"seed", "ext", "model", "flatten"}:
		raise ValueError(f"invalid model-init '{model_init}' (expected seed, ext, model, or flatten)")
	if model_init == "flatten":
		return _run_flatten_mode(
			cfg=cfg,
			fit_config=fit_config,
			args=args,
			model_cfg=model_cfg,
			opt_cfg=opt_cfg,
			progress_enabled=progress_enabled,
			out_dir=_out_dir,
		)

	data_cfg = cli_data.from_args(args)
	print("data:", data_cfg)
	print("model:", model_cfg)
	print("opt:", opt_cfg)

	device = torch.device(data_cfg.device)
	_require_torch_device_available(device)
	init_mode = str(model_cfg.init_mode).strip().lower()
	if init_mode == "shell-dir-crop" and model_init != "seed":
		raise ValueError("init-mode=shell-dir-crop requires args.model-init=seed")
	if init_mode == "shell-dir-crop" and "init_shell_dir" in cfg:
		raise ValueError("do not set top-level config key 'init_shell_dir'; shell-dir-crop reads it from --input .lasagna.json")

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

	# Approval inpaint is a seed-mode preprocessor: VC3D sends the selected
	# tifxyz plus approval/d channels, then this step derives corr points,
	# a centered effective seed, and extents before the configured init runs.
	_t = _stage_start("approval_inpaint")
	approval_inpaint_enabled = _truthy_config_bool(getattr(args, "approval_inpaint", False))
	approval_inpaint_output_mask_enabled = _truthy_config_bool(getattr(args, "approval_inpaint_output_mask", False))
	if approval_inpaint_output_mask_enabled and not approval_inpaint_enabled:
		raise ValueError("approval-inpaint-output-mask requires approval-inpaint=true")
	approval_inpaint_output_mask: dict | None = None
	if approval_inpaint_enabled:
		if model_init != "seed":
			raise ValueError("approval-inpaint requires args.model-init=seed")
		if data_cfg.seed is None:
			raise ValueError("approval-inpaint requires args.seed")
		approval_tifxyz = getattr(args, "approval_inpaint_tifxyz", None)
		if not approval_tifxyz:
			raise ValueError("approval-inpaint requires service arg approval-inpaint-tifxyz")
		from approval_inpaint import build_approval_inpaint

		result = build_approval_inpaint(
			tifxyz_path=str(approval_tifxyz),
			seed=tuple(float(v) for v in data_cfg.seed),
			mesh_step=float(model_cfg.mesh_step),
			corr_spacing=getattr(args, "approval_inpaint_corr_spacing", None),
			padding_frac=getattr(args, "approval_inpaint_padding_frac", 0.25),
			existing_corr_points=cfg.get("corr_points") if isinstance(cfg.get("corr_points"), dict) else None,
			output_mask=approval_inpaint_output_mask_enabled,
			output_mask_dilate=int(getattr(args, "approval_inpaint_output_mask_dilate", 3)),
		)
		approval_inpaint_output_mask = result.output_mask
		data_cfg = dataclasses.replace(
			data_cfg,
			seed=result.seed,
			model_w=result.model_w,
			model_h=result.model_h,
		)
		cfg["corr_points"] = result.corr_points
		fit_config["corr_points"] = copy.deepcopy(result.corr_points)
		fit_config.setdefault("args", {}).update({
			"seed": [float(v) for v in result.seed],
			"model-w": int(result.model_w),
			"model-h": int(result.model_h),
			"approval-inpaint": True,
			"approval-inpaint-output-mask": bool(approval_inpaint_output_mask_enabled),
			"approval-inpaint-output-mask-dilate": int(getattr(args, "approval_inpaint_output_mask_dilate", 3)),
		})
		if approval_inpaint_output_mask is not None:
			fit_config["args"]["approval-inpaint-output-mask-source"] = str(
				approval_inpaint_output_mask.get("source", "corr_points")
			)
			fit_config["args"]["approval-inpaint-output-mask-corr-collections"] = [
				int(v) for v in approval_inpaint_output_mask.get("corr_collection_ids", [])
			]
		print(
			f"[fit] approval-inpaint: points={result.point_count} "
			f"component={result.component_size} skeleton={result.skeleton_size} "
			f"bounds={result.index_bounds} source_step={result.source_mesh_step:.3f} "
			f"seed=({result.seed[0]:.1f},{result.seed[1]:.1f},{result.seed[2]:.1f}) "
			f"model_w={result.model_w} model_h={result.model_h}",
			flush=True,
		)
	_stage_done("approval_inpaint", _t)

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
		label = "shell-dir-crop" if init_mode == "shell-dir-crop" else "cylinder_seed"
		print(f"[fit] {label} from seed: x={float(data_cfg.seed[0]):.1f} "
			  f"y={float(data_cfg.seed[1]):.1f} z={float(data_cfg.seed[2]):.1f}",
			  flush=True)

	# --- Size mesh from model_h only for the umbilicus tube experiment ---
	if model_init == "seed" and init_mode == "cylinder_seed" and data_cfg.model_h is not None:
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

	tifxyz_init = getattr(args, "tifxyz_init", None)

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
		if init_mode == "shell-dir-crop":
			print("[fit] model-init=seed/init-mode=shell-dir-crop: constructing model from init shells", flush=True)
			from init_shell_index import (
				InitShellIndex,
				crop_shell_surface,
				shell_quality_analysis,
				trim_shell_surface_rows_by_quality,
			)
			init_shell_dir = _require_manifest_init_shell_dir(prep_params)
			shell_index = InitShellIndex.from_directory(init_shell_dir)
			closest = shell_index.closest_point(tuple(float(v) for v in data_cfg.seed), device=device)
			surface = shell_index.surfaces[closest.shell_index]
			source_step = float(surface.source_step) if surface.source_step is not None else float(model_cfg.mesh_step)
			selected_shell = surface.xyz_wrapped[:, :surface.unique_w].to(device=device, dtype=torch.float32)
			print(
				f"[fit] shell-dir-crop closest shell before crop: "
				f"id={closest.shell_id} path={surface.path} "
				f"source_step={source_step:.3f} "
				f"source_shape={int(surface.xyz_wrapped.shape[0])}x{int(surface.xyz_wrapped.shape[1])} "
				f"unique_shape={int(selected_shell.shape[0])}x{int(selected_shell.shape[1])} "
				f"quad=({closest.quad_row},{closest.quad_col}) tri={closest.triangle_id} "
				f"h={closest.h:.3f} w={closest.w:.3f} dist={closest.distance:.3f}",
				flush=True,
			)
			source_quality = shell_quality_analysis(selected_shell, target_step=source_step)
			print(
				f"[fit] shell-dir-crop source-shell quality before row trim: "
				f"target_step={source_quality['target_step']:.3f} target_area={source_quality['target_area']:.3f} "
				f"h=({source_quality['h_min']:.3f},{source_quality['h_med']:.3f},{source_quality['h_max']:.3f}) "
				f"w_top=({source_quality['w_top_min']:.3f},{source_quality['w_top_med']:.3f},{source_quality['w_top_max']:.3f}) "
				f"w_bottom=({source_quality['w_bottom_min']:.3f},{source_quality['w_bottom_med']:.3f},{source_quality['w_bottom_max']:.3f}) "
				f"diag_main=({source_quality['diag_main_min']:.3f},{source_quality['diag_main_med']:.3f},{source_quality['diag_main_max']:.3f}) "
				f"diag_anti=({source_quality['diag_anti_min']:.3f},{source_quality['diag_anti_med']:.3f},{source_quality['diag_anti_max']:.3f}) "
				f"area=({source_quality['area_min']:.3f},{source_quality['area_med']:.3f},{source_quality['area_max']:.3f}) "
				f"area_sqrt=({source_quality['area_sqrt_min']:.3f},{source_quality['area_sqrt_med']:.3f},{source_quality['area_sqrt_max']:.3f})",
				flush=True,
			)
			trimmed_surface, trim_top, trim_bottom = trim_shell_surface_rows_by_quality(
				surface,
				target_step=source_step,
				lo_ratio=0.67,
				hi_ratio=1.5,
			)
			if trim_top or trim_bottom:
				if not (float(trim_top) <= float(closest.h) <= float(trim_top + trimmed_surface.xyz_wrapped.shape[0] - 1)):
					raise ValueError(
						f"shell-dir-crop source row trim removed closest seed row: "
						f"h={closest.h:.3f} trim_top={trim_top} kept_h={int(trimmed_surface.xyz_wrapped.shape[0])}"
					)
				closest = dataclasses.replace(
					closest,
					h=float(closest.h) - float(trim_top),
					quad_row=max(0, int(closest.quad_row) - int(trim_top)),
				)
				surface = trimmed_surface
				selected_shell = surface.xyz_wrapped[:, :surface.unique_w].to(device=device, dtype=torch.float32)
				trim_quality = shell_quality_analysis(selected_shell, target_step=source_step)
				print(
					f"[fit] shell-dir-crop source-shell row trim: "
					f"trim_top={trim_top} trim_bottom={trim_bottom} "
					f"kept_shape={int(surface.xyz_wrapped.shape[0])}x{int(surface.xyz_wrapped.shape[1])} "
					f"adjusted_h={closest.h:.3f} "
					f"h=({trim_quality['h_min']:.3f},{trim_quality['h_med']:.3f},{trim_quality['h_max']:.3f}) "
					f"w_top=({trim_quality['w_top_min']:.3f},{trim_quality['w_top_med']:.3f},{trim_quality['w_top_max']:.3f}) "
					f"w_bottom=({trim_quality['w_bottom_min']:.3f},{trim_quality['w_bottom_med']:.3f},{trim_quality['w_bottom_max']:.3f}) "
					f"diag_main=({trim_quality['diag_main_min']:.3f},{trim_quality['diag_main_med']:.3f},{trim_quality['diag_main_max']:.3f}) "
					f"diag_anti=({trim_quality['diag_anti_min']:.3f},{trim_quality['diag_anti_med']:.3f},{trim_quality['diag_anti_max']:.3f}) "
					f"area=({trim_quality['area_min']:.3f},{trim_quality['area_med']:.3f},{trim_quality['area_max']:.3f}) "
					f"area_sqrt=({trim_quality['area_sqrt_min']:.3f},{trim_quality['area_sqrt_med']:.3f},{trim_quality['area_sqrt_max']:.3f})",
					flush=True,
				)
			step_stats = opt_loss_step.step_loss_analysis(selected_shell, mesh_step=source_step)
			print(
				f"[fit] shell-dir-crop selected-shell step analysis before crop: "
				f"loss={step_stats['loss']:.6g} target={step_stats['target']:.3f} "
				f"step_min={step_stats['step_min']:.3f} step_avg={step_stats['step_avg']:.3f} "
				f"step_med={step_stats['step_med']:.3f} step_max={step_stats['step_max']:.3f} "
				f"h_avg={step_stats['h_avg']:.3f} w_avg={step_stats['w_avg']:.3f} "
				f"diag_avg={step_stats['diag_avg']:.3f} "
				f"h_max={step_stats['h_max']:.3f} w_max={step_stats['w_max']:.3f} "
				f"diag_max={step_stats['diag_max']:.3f} max_kind={step_stats['max_kind']}",
				flush=True,
			)
			crop_xyz, crop_valid, crop_info = crop_shell_surface(
				surface,
				closest,
				seed=tuple(float(v) for v in data_cfg.seed),
				model_w=float(data_cfg.model_w) if data_cfg.model_w is not None else 0.0,
				model_h=float(data_cfg.model_h),
				model_w_unit=data_cfg.model_w_unit,
				mesh_step=float(model_cfg.mesh_step),
				device=device,
			)
			crop_step_stats = opt_loss_step.step_loss_analysis(crop_xyz, mesh_step=float(model_cfg.mesh_step))
			print(
				f"[fit] shell-dir-crop resampled-crop step analysis: "
				f"loss={crop_step_stats['loss']:.6g} target={crop_step_stats['target']:.3f} "
				f"step_min={crop_step_stats['step_min']:.3f} step_avg={crop_step_stats['step_avg']:.3f} "
				f"step_med={crop_step_stats['step_med']:.3f} step_max={crop_step_stats['step_max']:.3f} "
				f"h_avg={crop_step_stats['h_avg']:.3f} w_avg={crop_step_stats['w_avg']:.3f} "
				f"diag_avg={crop_step_stats['diag_avg']:.3f} "
				f"h_max={crop_step_stats['h_max']:.3f} w_max={crop_step_stats['w_max']:.3f} "
				f"diag_max={crop_step_stats['diag_max']:.3f} max_kind={crop_step_stats['max_kind']}",
				flush=True,
			)
			mdl = model.Model3D.from_tifxyz_crop(
				crop_xyz,
				crop_valid,
				device=device,
				mesh_step=model_cfg.mesh_step,
				winding_step=model_cfg.winding_step,
				subsample_mesh=model_cfg.subsample_mesh,
				subsample_winding=model_cfg.subsample_winding,
			)
			mdl.params = dataclasses.replace(
				mdl.params,
				scaledown=scaledown,
				z_step_eff=int(round(scaledown)),
				volume_extent=None,
				model_w=(None if data_cfg.model_w is None else float(data_cfg.model_w)),
				model_h=float(data_cfg.model_h),
			)
			model_step_stats = opt_loss_step.step_loss_analysis(mdl._grid_xyz().detach(), mesh_step=float(model_cfg.mesh_step))
			print(
				f"[fit] shell-dir-crop model-init step analysis: "
				f"loss={model_step_stats['loss']:.6g} target={model_step_stats['target']:.3f} "
				f"step_min={model_step_stats['step_min']:.3f} step_avg={model_step_stats['step_avg']:.3f} "
				f"step_med={model_step_stats['step_med']:.3f} step_max={model_step_stats['step_max']:.3f} "
				f"h_avg={model_step_stats['h_avg']:.3f} w_avg={model_step_stats['w_avg']:.3f} "
				f"diag_avg={model_step_stats['diag_avg']:.3f} "
				f"h_max={model_step_stats['h_max']:.3f} w_max={model_step_stats['w_max']:.3f} "
				f"diag_max={model_step_stats['diag_max']:.3f} max_kind={model_step_stats['max_kind']}",
				flush=True,
			)
			print(
				f"[fit] shell-dir-crop selected {closest.shell_id}: "
				f"quad=({closest.quad_row},{closest.quad_col}) tri={closest.triangle_id} "
				f"h={closest.h:.3f} w={closest.w:.3f} "
				f"dist={closest.distance:.3f} "
				f"crop={crop_info.mesh_h}x{crop_info.mesh_w} "
				f"requested_h={crop_info.requested_mesh_h} "
				f"dropped_h={crop_info.requested_mesh_h - crop_info.mesh_h} "
				f"source={crop_info.source_h}x{crop_info.source_w} "
				f"full_width={crop_info.full_width}",
				flush=True,
			)
		elif init_mode == "cylinder_seed":
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
			raise ValueError(f"unsupported init-mode for model-init=seed: {init_mode}")
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
		from tifxyz_io import load_tifxyz, surface_step_stats
		for es in ext_surfaces_cfg:
			es_path = str(es["path"])
			es_offset = float(es.get("offset", 1.0))
			xyz_ext, valid_ext, meta_ext = load_tifxyz(es_path, device=device)
			idx = mdl.add_external_surface(xyz_ext, valid=valid_ext, offset=es_offset)
			scale = meta_ext.get("scale") if isinstance(meta_ext, dict) else None
			meta_step = float("nan")
			if isinstance(scale, list) and scale and float(scale[0]) > 0.0:
				meta_step = 1.0 / float(scale[0])
			step_h, step_w, step_diag, step_avg = surface_step_stats(xyz_ext, valid_ext)
			ratio = step_avg / max(1.0e-8, float(mdl.params.mesh_step))
			print(f"[fit] external surface {idx}: path={es_path} offset={es_offset} "
				  f"shape={tuple(xyz_ext.shape)} valid={int(valid_ext.sum())}/{valid_ext.numel()} "
				  f"meta_step={meta_step:.3f} step_h={step_h:.3f} step_w={step_w:.3f} step_diag={step_diag:.3f} "
				  f"step_avg={step_avg:.3f} model_step={float(mdl.params.mesh_step):.3f} "
				  f"step_ratio={ratio:.3f}", flush=True)
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
	cfg.pop("tifxyz", None)
	stages = optimizer.load_stages_cfg(
		cfg,
		init_mode=model_cfg.init_mode if model_init == "seed" else None,
	)
	print("[fit] optimizer stages:", flush=True)
	for i, st in enumerate(stages):
		args_snap_map = st.global_opt.args.get("snap_surf_map") if isinstance(st.global_opt.args, dict) else None
		print(
			f"[fit]   stage{i} name={st.name!r} steps={st.global_opt.steps} "
			f"snap_surf_map_eff={st.global_opt.eff.get('snap_surf_map', 0.0):.6g} "
			f"snap_surf_map_args={args_snap_map}",
			flush=True,
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
		_apply_cylinder_prepare_model_step(mdl, _first_cylinder_stage_model_step(stages))
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
		if approval_inpaint_output_mask is not None:
			st["_approval_inpaint_output_mask_"] = copy.deepcopy(approval_inpaint_output_mask)
			print(
				"[fit] saving approval-inpaint output mask "
				f"collections={approval_inpaint_output_mask.get('corr_collection_ids', [])} "
				f"dilate={approval_inpaint_output_mask.get('dilation_radius')} "
				f"corr_results_saved={corr_results is not None}",
				flush=True,
			)
			if corr_results is None:
				print(
					"[fit] WARNING: approval-inpaint output mask was requested, but no "
					"corr point results were produced; fit2tifxyz cannot project the mask",
					flush=True,
				)
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
	config_seed = tuple(float(v) for v in data_cfg.seed) if data_cfg.seed is not None else None
	seed_xyz = _optimization_seed_xyz(model_init=model_init, config_seed=config_seed, mdl=mdl)
	if model_init == "ext":
		print(f"[fit] tifxyz seed: ({seed_xyz[0]:.0f}, {seed_xyz[1]:.0f}, {seed_xyz[2]:.0f})",
			  flush=True)
	elif model_init == "model":
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
