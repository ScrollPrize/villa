from __future__ import annotations

import json
import time
from dataclasses import dataclass

import torch

import cli_data
import fit_data
import opt_loss_data
import opt_loss_dir
import opt_loss_pred_dt
import opt_loss_step
import opt_loss_smooth
import opt_loss_winding_density
import opt_loss_corr
import opt_loss_winding_volume
import opt_loss_station
import opt_loss_bend


def _require_consumed_dict(*, where: str, cfg: dict) -> None:
	if cfg:
		bad = sorted(cfg.keys())
		print(f"WARNING stages_json: {where}: unknown key(s): {bad}")


@dataclass(frozen=True)
class OptSettings:
	steps: int
	lr: float | list[float]
	params: list[str]
	min_scaledown: int
	default_mul: float | None
	w_fac: dict | None
	eff: dict[str, float]
	args: dict | None = None


@dataclass(frozen=True)
class Stage:
	name: str
	global_opt: OptSettings


def _stage_to_modifiers(
	base: dict[str, float],
	prev_eff: dict[str, float] | None,
	default_mul: float | None,
	w_fac: dict | None,
) -> tuple[dict[str, float], dict[str, float]]:
	if prev_eff is None:
		prev_eff = {k: float(v) for k, v in base.items()}
	if default_mul is None and w_fac is None:
		eff = dict(prev_eff)
	else:
		eff = dict(prev_eff)
		if default_mul is not None:
			for name in base.keys():
				if w_fac is None or name not in w_fac:
					eff[name] = float(base[name]) * float(default_mul)
		if w_fac is not None:
			for k, v in w_fac.items():
				if v is None:
					continue
				eff[str(k)] = float(base.get(str(k), 0.0)) * float(v)

	mods: dict[str, float] = {}
	for name, val in eff.items():
		b = float(base.get(name, 0.0))
		mods[name] = (float(val) / b) if b != 0.0 else 0.0
	return eff, mods


def _need_term(name: str, stage_eff: dict[str, float]) -> float:
	return float(stage_eff.get(name, 0.0))


def _parse_opt_settings(
	*,
	stage_name: str,
	opt_cfg: dict,
	base: dict[str, float],
	prev_eff: dict[str, float] | None,
) -> OptSettings:
	opt_cfg = dict(opt_cfg)
	steps = max(0, int(opt_cfg.get("steps", 0)))
	lr_raw = opt_cfg.get("lr", 1e-3)
	if isinstance(lr_raw, list):
		if not lr_raw:
			raise ValueError(f"stages_json: stage '{stage_name}' opt.lr: must be a number or a non-empty list")
		lr: float | list[float] = [float(v) for v in lr_raw]
	else:
		lr = float(lr_raw)
	params = opt_cfg.get("params", [])
	if not isinstance(params, list):
		params = []
	params = [str(p) for p in params]
	valid = {"mesh_ms", "amp", "bias",
			 "arc_cx", "arc_cy", "arc_radius", "arc_angle0", "arc_angle1",
			 "straight_cx", "straight_cy", "straight_angle", "straight_half_w"}
	bad_params = sorted(set(params) - valid)
	if bad_params:
		raise ValueError(f"stages_json: stage '{stage_name}' opt.params: unknown name(s): {bad_params}")
	min_scaledown = max(0, int(opt_cfg.get("min_scaledown", 0)))
	default_mul = opt_cfg.get("default_mul", None)
	w_fac = opt_cfg.get("w_fac", None)
	args_raw = opt_cfg.get("args", None)
	# Back-compat: translate old "auto_offset": true → args dict
	if args_raw is None and opt_cfg.get("auto_offset", False):
		args_raw = {"winding_offset_autocrop": True}
	if args_raw is not None and not isinstance(args_raw, dict):
		raise ValueError(f"stages_json: stage '{stage_name}' opt 'args' must be an object or null")
	args = dict(args_raw) if args_raw else {}
	opt_cfg.pop("steps", None)
	opt_cfg.pop("lr", None)
	opt_cfg.pop("params", None)
	opt_cfg.pop("min_scaledown", None)
	opt_cfg.pop("default_mul", None)
	opt_cfg.pop("w_fac", None)
	opt_cfg.pop("auto_offset", None)
	opt_cfg.pop("args", None)
	_require_consumed_dict(where=f"stage '{stage_name}' opt", cfg=opt_cfg)
	if default_mul is not None:
		default_mul = float(default_mul)
	if w_fac is not None and not isinstance(w_fac, dict):
		raise ValueError(f"stages_json: stage '{stage_name}' opt 'w_fac' must be an object or null")
	if isinstance(w_fac, dict):
		bad_terms = sorted(set(str(k) for k in w_fac.keys()) - set(base.keys()))
		if bad_terms:
			raise ValueError(f"stages_json: stage '{stage_name}' opt.w_fac: unknown term(s): {bad_terms}")
	eff, _mods = _stage_to_modifiers(base, prev_eff, default_mul, w_fac)
	return OptSettings(
		steps=steps,
		lr=lr,
		params=params,
		min_scaledown=min_scaledown,
		default_mul=default_mul,
		w_fac=w_fac,
		eff=eff,
		args=args,
	)


lambda_global: dict[str, float] = {
	"normal": 1.0,
	"step": 0.0,
	"smooth": 0.0,
	"winding_density": 0.0,
	"data": 0.0,
	"data_plain": 0.0,
	"pred_dt": 0.0,
	"corr": 0.0,
	"winding_vol": 0.0,
	"station_n": 0.0,
	"station_t": 0.0,
	"bend": 0.0,
	"ext_offset": 0.0,
}


def load_stages_cfg(cfg: dict) -> list[Stage]:
	cfg = dict(cfg)
	base = dict(lambda_global)
	base_cfg = cfg.pop("base", None)
	if isinstance(base_cfg, dict):
		bad_base = sorted(set(str(k) for k in base_cfg.keys()) - set(base.keys()))
		if bad_base:
			raise ValueError(f"stages_json: base: unknown term(s): {bad_base}")
		for k, v in base_cfg.items():
			base[str(k)] = float(v)

	stages_cfg = cfg.pop("stages", None)
	if not isinstance(stages_cfg, list) or not stages_cfg:
		raise ValueError("stages_json: expected a non-empty list in key 'stages'")
	_require_consumed_dict(where="top-level", cfg=cfg)

	out: list[Stage] = []
	for s in stages_cfg:
		if not isinstance(s, dict):
			raise ValueError("stages_json: each stage must be an object")
		s = dict(s)
		name = str(s.pop("name", ""))
		global_opt_cfg = s.pop("global_opt", None)
		if global_opt_cfg is None:
			global_opt_cfg = dict(s)
			s.clear()
		_require_consumed_dict(where=f"stage '{name}'", cfg=s)
		if not isinstance(global_opt_cfg, dict):
			raise ValueError(f"stages_json: stage '{name}' field 'global_opt' must be an object")
		global_opt = _parse_opt_settings(stage_name=name, opt_cfg=global_opt_cfg, base=base, prev_eff=None)
		out.append(Stage(name=name, global_opt=global_opt))
	return out


def load_stages(path: str) -> list[Stage]:
	with open(path, "r", encoding="utf-8") as f:
		cfg = json.load(f)
		if not isinstance(cfg, dict):
			raise ValueError("stages_json: expected an object")
		return load_stages_cfg(cfg)


def total_steps_for_stages(stages: list[Stage]) -> int:
	total = 0
	for stage in stages:
		total += max(0, stage.global_opt.steps)
	return total


def _lr_last(lr: float | list[float]) -> float:
	if isinstance(lr, list):
		return float(lr[-1])
	return float(lr)


def _lr_scalespace(*, lr: float | list[float], scale_i: int) -> float:
	if not isinstance(lr, list):
		return float(lr)
	if not lr:
		return 0.0
	idx = -1 - int(scale_i)
	if -len(lr) <= idx < 0:
		return float(lr[idx])
	return float(lr[0])


def check_data_bounds(model, data: fit_data.FitData3D, margin: float = 100.0,
					  volume_extent_fullres: tuple[int, int, int] | None = None) -> bool:
	"""Return True if any mesh vertex is within `margin` fullres voxels of the data border.

	Skips edges where the loaded data already reaches the volume boundary.
	"""
	with torch.no_grad():
		xyz = model._grid_xyz()  # (D, Hm, Wm, 3)
		mesh_min = [float(xyz[..., i].min()) for i in range(3)]
		mesh_max = [float(xyz[..., i].max()) for i in range(3)]
	Z, Y, X = data.size
	# Data extent in fullres: (x, y, z)
	data_min = list(data.origin_fullres)
	data_max = [
		data.origin_fullres[0] + (X - 1) * data.spacing[0],
		data.origin_fullres[1] + (Y - 1) * data.spacing[1],
		data.origin_fullres[2] + (Z - 1) * data.spacing[2],
	]
	# Full volume max per axis (x, y, z)
	if volume_extent_fullres is not None:
		vol_max = [float(volume_extent_fullres[0]),
				   float(volume_extent_fullres[1]),
				   float(volume_extent_fullres[2])]
	else:
		vol_max = None
	for i in range(3):
		# Near min edge — but skip if data already starts at volume origin
		if data_min[i] > 0 and mesh_min[i] - data_min[i] < margin:
			return True
		# Near max edge — but skip if data already reaches volume edge
		at_vol_max = vol_max is not None and data_max[i] >= vol_max[i] - data.spacing[i]
		if not at_vol_max and data_max[i] - mesh_max[i] < margin:
			return True
	return False


def optimize(
	*,
	model,
	data: fit_data.FitData3D,
	stages: list[Stage],
	snapshot_interval: int,
	snapshot_fn,
	progress_fn=None,
	ensure_data_fn=None,
	seed_xyz: tuple[float, float, float] | None = None,
) -> fit_data.FitData3D:
	opt_loss_corr.reset_state()

	terms = {
		"step": {"loss": opt_loss_step.step_loss},
		"smooth": {"loss": opt_loss_smooth.smooth_loss},
		"winding_density": {"loss": opt_loss_winding_density.winding_density_loss, "min_depth": 2},
		"normal": {"loss": opt_loss_dir.normal_loss},
		"data": {"loss": opt_loss_data.data_loss},
		"data_plain": {"loss": opt_loss_data.data_plain_loss},
		"pred_dt": {"loss": opt_loss_pred_dt.pred_dt_loss},
		"corr": {"loss": opt_loss_corr.corr_winding_loss},
		"winding_vol": {"loss": opt_loss_winding_volume.winding_volume_loss},
		"station": {"loss": opt_loss_station.station_loss, "sub": ["station_n", "station_t"]},
		"bend": {"loss": opt_loss_bend.bend_loss},
		"ext_offset": {"loss": opt_loss_winding_density.ext_offset_loss},
	}

	_corr_start_printed = [False]

	def _run_opt(*, si: int, label: str, stage: Stage, opt_cfg: OptSettings, data: fit_data.FitData3D) -> fit_data.FitData3D:
		print(f"[optimizer] {label}: params={opt_cfg.params} steps={opt_cfg.steps} "
			  f"lr={opt_cfg.lr} min_scaledown={opt_cfg.min_scaledown}", flush=True)
		if opt_cfg.steps <= 0:
			return data

		# Configure corr Phase D Gaussian-splat σ (default 1.0; 7×7 vertex neighborhood).
		corr_splat_sigma = float(opt_cfg.args.get("corr_splat_sigma", 1.0)) if opt_cfg.args else 1.0
		opt_loss_corr.set_splat_sigma(corr_splat_sigma)

		# If arc/straight params not in optimized set, bake into mesh
		arc_params_set = {"arc_cx", "arc_cy", "arc_radius", "arc_angle0", "arc_angle1"}
		if not arc_params_set.intersection(opt_cfg.params):
			if hasattr(model, "arc_enabled") and model.arc_enabled:
				model.bake_arc_into_mesh()
		straight_params_set = {"straight_cx", "straight_cy", "straight_angle", "straight_half_w"}
		if not straight_params_set.intersection(opt_cfg.params):
			if hasattr(model, "straight_enabled") and model.straight_enabled:
				model.bake_straight_into_mesh()

		all_params = model.opt_params()
		param_groups: list[dict] = []
		for name in opt_cfg.params:
			group = all_params.get(name, [])
			if name in {"mesh_ms"}:
				k0 = max(0, int(opt_cfg.min_scaledown))
				for pi, p in enumerate(group):
					if pi < k0:
						continue
					param_groups.append({"params": [p], "lr": _lr_scalespace(lr=opt_cfg.lr, scale_i=pi)})
			else:
				lr_last = _lr_last(opt_cfg.lr)
				for p in group:
					param_groups.append({"params": [p], "lr": lr_last})
		if not param_groups:
			return data
		opt = torch.optim.Adam(param_groups)

		# winding_offset_autocrop: compute offset/direction then crop invalid depth layers
		if opt_cfg.args and opt_cfg.args.get("winding_offset_autocrop") and _need_term("winding_vol", opt_cfg.eff) > 0:
			with torch.no_grad():
				res_ao = model(data)
			ao_offset, ao_dir = opt_loss_winding_volume.compute_auto_offset(res=res_ao)
			print(f"[optimizer] auto_offset: offset={ao_offset}, direction={ao_dir}", flush=True)
			d_lo, d_hi = opt_loss_winding_volume.compute_depth_crop_range(
				ao_offset, ao_dir, model.depth, data.winding_volume,
				winding_min=data.winding_min, winding_max=data.winding_max,
			)
			if d_lo != 0 or d_hi != model.depth:
				model.crop_depth(d_lo, d_hi)
				# Update winding offset to account for removed leading layers
				opt_loss_winding_volume._winding_offset = ao_offset + d_lo * ao_dir
				print(f"[optimizer] adjusted offset after crop: {opt_loss_winding_volume._winding_offset}", flush=True)
				# Rebuild optimizer param groups since model shape changed
				all_params = model.opt_params()
				param_groups = []
				for name in opt_cfg.params:
					group = all_params.get(name, [])
					if name in {"mesh_ms"}:
						k0 = max(0, int(opt_cfg.min_scaledown))
						for pi, p in enumerate(group):
							if pi < k0:
								continue
							param_groups.append({"params": [p], "lr": _lr_scalespace(lr=opt_cfg.lr, scale_i=pi)})
					else:
						lr_last = _lr_last(opt_cfg.lr)
						for p in group:
							param_groups.append({"params": [p], "lr": lr_last})
				if not param_groups:
					return data
				opt = torch.optim.Adam(param_groups)

		_status_rows = 0

		def _print_status(*, step_label: str, loss_val: float, tv: dict[str, float], pv: dict[str, float],
						  its: float | None = None) -> None:
			nonlocal _status_rows
			tv_keys = sorted(tv.keys())
			pv_keys = sorted(pv.keys())
			cols = tv_keys + [f"p:{k}" for k in pv_keys]
			if _status_rows % 20 == 0:
				hdr = f"{'step':>20s}  {'loss':>8s}  {'it/s':>6s}"
				for c in cols:
					hdr += f"  {c:>10s}"
				print(hdr)
			_status_rows += 1
			its_str = f"{its:6.1f}" if its is not None else f"{'':>6s}"
			row = f"{step_label:>20s}  {loss_val:8.4f}  {its_str}"
			for k in tv_keys:
				row += f"  {tv[k]:10.4f}"
			for k in pv_keys:
				row += f"  {pv[k]:10.4f}"
			print(row)

		# Ensure data covers mesh and has all channels needed by this stage
		_needed_channels: set[str] = set()
		if _need_term("pred_dt", opt_cfg.eff) > 0:
			_needed_channels.add("pred_dt")
		if _need_term("data", opt_cfg.eff) > 0 or _need_term("data_plain", opt_cfg.eff) > 0:
			_needed_channels.add("cos")
		if ensure_data_fn is not None:
			data = ensure_data_fn(data, _needed_channels)

		# Initial evaluation
		def _eval_terms(res_, eff_):
			"""Evaluate all loss terms, handling both single and multi-loss returns."""
			total = torch.zeros((), device=next(model.parameters()).device, dtype=torch.float32)
			tv: dict[str, float] = {}
			D = res_.xyz_lr.shape[0]
			for name, t in terms.items():
				min_d = t.get("min_depth", 1)
				if D < min_d:
					continue
				sub_names = t.get("sub")
				if sub_names:
					# Multi-loss: check if any sub-term has weight
					if not any(_need_term(s, eff_) > 0 for s in sub_names):
						continue
				else:
					if _need_term(name, eff_) == 0.0:
						continue
				result = t["loss"](res=res_)
				if isinstance(result, dict):
					for sub_name, (lv, lms, masks) in result.items():
						w = _need_term(sub_name, eff_)
						if w == 0.0:
							continue
						tv[sub_name] = float(lv.detach().cpu())
						total = total + w * lv
				else:
					lv, lms, masks = result
					w = _need_term(name, eff_)
					tv[name] = float(lv.detach().cpu())
					total = total + w * lv
			return total, tv

		# Streaming mode: filter caches to only those needed by this stage
		# grad_mag/nx/ny are always needed; cos and pred_dt are conditional
		_active_caches = []
		if data.sparse_caches:
			_always_needed = {"grad_mag", "nx", "ny"}
			_stage_channels = _always_needed | _needed_channels
			for _cache in data.sparse_caches.values():
				if _stage_channels & set(_cache.channels):
					_active_caches.append(_cache)

		# Initial prefetch for streaming mode
		if _active_caches:
			with torch.no_grad():
				_xyz_lr_pf = model._grid_xyz()
				_xyz_hr_pf = model._grid_xyz_hr(_xyz_lr_pf)
			for _cache in _active_caches:
				_sp = data._spacing_for(_cache.channels[0])
				_cache.prefetch(_xyz_hr_pf, data.origin_fullres, _sp)
			# Also prefetch chunks for corr points (static positions, loaded once)
			if data.corr_points is not None and data.corr_points.points_xyz_winda.shape[0] > 0:
				_corr_xyz = data.corr_points.points_xyz_winda[:, :3].to(
					device=next(model.parameters()).device, dtype=torch.float32)
				for _cache in _active_caches:
					_sp = data._spacing_for(_cache.channels[0])
					_cache.prefetch(_corr_xyz, data.origin_fullres, _sp)
			for _cache in _active_caches:
				_cache.sync()

		# Station-keeping: set seed point anchor (once, on first stage that uses it)
		# Must be AFTER prefetch+sync so grid_sample_fullres can read loaded chunks.
		if (_need_term("station_n", opt_cfg.eff) > 0 or _need_term("station_t", opt_cfg.eff) > 0) and seed_xyz is not None:
			dev = next(model.parameters()).device
			seed_t = torch.tensor(list(seed_xyz), device=dev, dtype=torch.float32)
			opt_loss_station.set_seed(seed_t, data, Hm=model.mesh_h, Wm=model.mesh_w, D=model.depth)

		with torch.no_grad():
			res0 = model(data)
			loss0, term_vals0 = _eval_terms(res0, opt_cfg.eff)
			param_vals0: dict[str, float] = {}
			for k, vs in all_params.items():
				if len(vs) == 1 and vs[0].numel() == 1:
					param_vals0[k] = float(vs[0].detach().cpu())
			term_vals0 = {k: round(v, 4) for k, v in term_vals0.items()}
			param_vals0 = {k: round(v, 4) for k, v in param_vals0.items()}
			_print_status(step_label=f"{label} 0/{opt_cfg.steps}", loss_val=loss0.item(), tv=term_vals0, pv=param_vals0)
			# Print corr detail after initial eval (first stage only)
			if not _corr_start_printed[0] and "corr" in term_vals0:
				opt_loss_corr.print_detail("START")
				_corr_start_printed[0] = True
		snapshot_fn(stage=label, step=0, loss=float(loss0.detach().cpu()), data=data, res=res0)

		max_steps = opt_cfg.steps
		_t_wall_start = time.perf_counter()
		_t_steps_acc = 0
		loss = loss0

		for step in range(max_steps):
			# Sync: wait for chunks loaded by last prefetch
			if _active_caches:
				for _cache in _active_caches:
					_cache.sync()

			if fit_data.CHUNK_STATS_ENABLED:
				fit_data._chunk_stats.begin_iteration()
			res = model(data)
			loss, term_vals = _eval_terms(res, opt_cfg.eff)
			if fit_data.CHUNK_STATS_ENABLED:
				fit_data._chunk_stats.end_iteration()

			opt.zero_grad(set_to_none=True)
			loss.backward()
			opt.step()
			model.update_conn_offsets()
			model.update_ext_conn_offsets()

			# Prefetch: predict next iteration's chunks from updated mesh
			if _active_caches:
				with torch.no_grad():
					_xyz_lr_pf = model._grid_xyz()
					_xyz_hr_pf = model._grid_xyz_hr(_xyz_lr_pf)
				for _cache in _active_caches:
					_sp = data._spacing_for(_cache.channels[0])
					_cache.prefetch(_xyz_hr_pf, data.origin_fullres, _sp)
				for _cache in _active_caches:
					_cache.end_iteration()
			_t_steps_acc += 1
			_done_steps[0] += 1

			step1 = step + 1
			_stage_progress = step1 / max_steps if max_steps > 0 else 1.0
			_overall_progress = (si + _stage_progress) / _num_stages if _num_stages > 0 else 1.0

			if progress_fn is not None:
				progress_fn(
					step=_done_steps[0], total=_total_steps, loss=float(loss.detach().cpu()),
					stage_progress=_stage_progress, overall_progress=_overall_progress,
					stage_name=stage.name,
				)

			if step == 0 or step1 == max_steps or (step1 % 100) == 0:
				param_vals: dict[str, float] = {}
				for k, vs in all_params.items():
					if len(vs) == 1 and vs[0].numel() == 1:
						param_vals[k] = float(vs[0].detach().cpu())
				term_vals = {k: round(v, 4) for k, v in term_vals.items()}
				param_vals = {k: round(v, 4) for k, v in param_vals.items()}
				_t_wall_now = time.perf_counter()
				_t_wall_elapsed = _t_wall_now - _t_wall_start
				_its = _t_steps_acc / _t_wall_elapsed if _t_wall_elapsed > 0 else None
				_print_status(step_label=f"{label} {step1}/{opt_cfg.steps}",
							  loss_val=loss.item(), tv=term_vals, pv=param_vals, its=_its)
				_t_steps_acc = 0
				_t_wall_start = _t_wall_now

			if ensure_data_fn is not None and (step1 % 100) == 0:
				data = ensure_data_fn(data, _needed_channels)

			if snap_int > 0 and (step1 % snap_int) == 0:
				snapshot_fn(stage=label, step=step1, loss=float(loss.detach().cpu()), data=data, res=res)

		snapshot_fn(stage=label, step=max_steps, loss=float(loss.detach().cpu()), data=data, res=res)
		return data

	snap_int = int(snapshot_interval)
	if snap_int < 0:
		snap_int = 0

	_total_steps = total_steps_for_stages(stages)
	_done_steps = [0]
	_num_stages = len(stages)

	# Debug: show corr status
	_corr_terms = ("corr",)
	has_corr_pts = data.corr_points is not None and data.corr_points.points_xyz_winda.shape[0] > 0
	corr_weights = {t: [(_need_term(t, s.global_opt.eff), s.name) for s in stages if s.global_opt.steps > 0]
					for t in _corr_terms}
	active_corr = {t: ws for t, ws in corr_weights.items() if any(w > 0 for w, _ in ws)}
	print(f"[optimizer] corr_points={has_corr_pts} active_corr_terms={list(active_corr.keys())}", flush=True)
	if has_corr_pts:
		cp = data.corr_points
		n = cp.points_xyz_winda.shape[0]
		print(f"[optimizer] {n} corr points", flush=True)
		if not active_corr:
			print(f"[optimizer] WARNING: corr points loaded but no corr weight > 0 in any stage!", flush=True)

	for si, stage in enumerate(stages):
		if stage.global_opt.steps > 0:
			data = _run_opt(si=si, label=f"stage{si}", stage=stage, opt_cfg=stage.global_opt, data=data)

	# Print sparse cache summary
	if data.sparse_caches:
		for _cache in data.sparse_caches.values():
			_cache.print_summary()

	if active_corr:
		opt_loss_corr.print_detail("END")
		opt_loss_corr.print_summary()
	elif has_corr_pts:
		print("[optimizer] corr points present but no corr weight > 0, no corr loss computed", flush=True)

	return data
