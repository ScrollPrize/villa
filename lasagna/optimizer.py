from __future__ import annotations

import json
import os
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
import opt_loss_cyl


def _debug_cuda_sync(label: str) -> None:
	if os.environ.get("LASAGNA_SYNC_DEBUG", "0") == "0":
		return
	if torch.cuda.is_available():
		try:
			torch.cuda.synchronize()
		except RuntimeError as exc:
			raise RuntimeError(f"CUDA failure after {label}") from exc


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
	valid = {"mesh_ms", "amp", "bias", "cyl_params"}
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
	if "cyl_params" in params:
		if params != ["cyl_params"]:
			raise ValueError(f"stages_json: stage '{stage_name}' opt.params: cyl_params must be optimized alone")
		if float(eff.get("cyl_normal", 0.0)) == 0.0 and float(eff.get("cyl_center", 0.0)) == 0.0:
			raise ValueError(f"stages_json: stage '{stage_name}' with cyl_params requires a nonzero cylinder loss")
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
	"cyl_normal": 0.0,
	"cyl_center": 0.0,
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
		mul = 3 if "cyl_params" in stage.global_opt.params else 1
		total += max(0, stage.global_opt.steps) * mul
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
	out_dir: str | None = None,
) -> fit_data.FitData3D:
	opt_loss_corr.reset_state()

	def _stage_start(name: str) -> float:
		return 0.0

	def _stage_done(name: str, t0: float) -> None:
		return None

	def _timing_cuda_sync() -> None:
		if torch.cuda.is_available():
			torch.cuda.synchronize()

	def _truthy(value) -> bool:
		if isinstance(value, bool):
			return value
		if value is None:
			return False
		if isinstance(value, (int, float)):
			return value != 0
		return str(value).strip().lower() not in {"", "0", "false", "no", "off"}

	def _flow_timing_enabled(cfg) -> bool:
		if _truthy(os.environ.get("LASAGNA_FLOW_TIMING")):
			return True
		if not isinstance(cfg, dict):
			return False
		return _truthy(cfg.get("profile_cuda_timing", False))

	class _FlowTimingWindow:
		def __init__(self, *, interval: int = 100) -> None:
			self.interval = max(1, int(interval))
			self.count = 0
			self.acc = {
				"total": 0.0,
				"io_prefetch": 0.0,
				"flow_sampling": 0.0,
				"flow_calc": 0.0,
				"opt_step": 0.0,
				"model_forward": 0.0,
				"loss_eval": 0.0,
			}

		def add(self, key: str, seconds: float) -> None:
			self.acc[key] = self.acc.get(key, 0.0) + max(0.0, float(seconds))

		def finish_iter(self, *, label: str, step1: int, max_steps: int) -> None:
			self.count += 1
			if (step1 % self.interval) != 0 and step1 != max_steps:
				return
			if self.count <= 0:
				return
			total = max(1.0e-12, self.acc.get("total", 0.0))
			io_prefetch = self.acc.get("io_prefetch", 0.0)
			flow_sampling = self.acc.get("flow_sampling", 0.0)
			flow_calc = self.acc.get("flow_calc", 0.0)
			opt_step = self.acc.get("opt_step", 0.0)
			measured = io_prefetch + flow_sampling + flow_calc + opt_step
			other = max(0.0, total - measured)
			rows = [
				("io/prefetch", io_prefetch),
				("flow sampling", flow_sampling),
				("flow calc", flow_calc),
				("opt step", opt_step),
				("other", other),
			]
			print(f"[flow_timing] {label} {step1}/{max_steps} over {self.count} iters", flush=True)
			print(f"{'part':<16s} {'runtime_%':>9s} {'ms/it':>10s}", flush=True)
			for name, seconds in rows:
				pct = 100.0 * seconds / total
				ms_it = 1000.0 * seconds / float(self.count)
				print(f"{name:<16s} {pct:9.2f} {ms_it:10.2f}", flush=True)
			self.count = 0
			for key in list(self.acc.keys()):
				self.acc[key] = 0.0

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
		"cyl_normal": {"loss": opt_loss_cyl.cyl_normal_loss},
		"cyl_center": {"loss": opt_loss_cyl.cyl_center_loss},
	}

	_corr_start_printed = [False]

	def _run_opt(*, si: int, label: str, stage: Stage, opt_cfg: OptSettings, data: fit_data.FitData3D) -> fit_data.FitData3D:
		_t_stage_total = _stage_start(f"{label}.total")
		print(f"[optimizer] {label}: params={opt_cfg.params} steps={opt_cfg.steps} "
			  f"lr={opt_cfg.lr} min_scaledown={opt_cfg.min_scaledown}", flush=True)
		is_cyl_stage = "cyl_params" in opt_cfg.params
		if opt_cfg.steps <= 0 and not is_cyl_stage:
			return data
		stage_eff = (
			{
				"cyl_normal": float(opt_cfg.eff.get("cyl_normal", 0.0)),
				"cyl_center": float(opt_cfg.eff.get("cyl_center", 0.0)),
			}
			if is_cyl_stage else opt_cfg.eff
		)
		stage_uses_cyl_loss = (
			_need_term("cyl_normal", stage_eff) > 0 or
			_need_term("cyl_center", stage_eff) > 0
		)
		stage_args = opt_cfg.args or {}
		status_interval_raw = stage_args.get("status_interval", stage_args.get("debug_print_interval", 100))
		status_interval = max(0, int(status_interval_raw))

		# Configure corr Phase D Gaussian-splat σ (default 1.0; 7×7 vertex neighborhood).
		_t = _stage_start(f"{label}.configure_losses")
		corr_splat_sigma = float(opt_cfg.args.get("corr_splat_sigma", 1.0)) if opt_cfg.args else 1.0
		opt_loss_corr.set_splat_sigma(corr_splat_sigma)
		pred_dt_flow_gate_cfg = opt_cfg.args.get("pred_dt_flow_gate") if opt_cfg.args else None
		pred_dt_normal_source = (opt_cfg.args or {}).get("pred_dt_normal_source", None)
		if pred_dt_normal_source is None and isinstance(pred_dt_flow_gate_cfg, dict):
			pred_dt_normal_source = pred_dt_flow_gate_cfg.get("normal_source", None)
		opt_loss_pred_dt.configure_pred_dt(normal_source=pred_dt_normal_source)
		opt_loss_pred_dt.configure_flow_gate(
			cfg=pred_dt_flow_gate_cfg if _need_term("pred_dt", stage_eff) > 0 else None,
			stage_name=stage.name or label,
			seed_xyz=seed_xyz,
			out_dir=out_dir,
		)
		_stage_done(f"{label}.configure_losses", _t)

		# Once cylinder initialization is done, convert only the best candidate
		# to the regular mesh before any mesh-space optimization.
		_t = _stage_start(f"{label}.prepare_model_params")
		if not is_cyl_stage and getattr(model, "cylinder_enabled", False):
			model.bake_cylinder_into_mesh(data)
		_stage_done(f"{label}.prepare_model_params", _t)

		def _make_param_groups() -> tuple[dict[str, list], list[dict]]:
			all_params_ = model.opt_params()
			param_groups_: list[dict] = []
			for name in opt_cfg.params:
				group = all_params_.get(name, [])
				if name in {"mesh_ms"}:
					k0 = max(0, int(opt_cfg.min_scaledown))
					for pi, p in enumerate(group):
						if pi < k0:
							continue
						param_groups_.append({"params": [p], "lr": _lr_scalespace(lr=opt_cfg.lr, scale_i=pi)})
				else:
					lr_last = _lr_last(opt_cfg.lr)
					for p in group:
						param_groups_.append({"params": [p], "lr": lr_last})
			return all_params_, param_groups_

		_t = _stage_start(f"{label}.build_optimizer")
		all_params, param_groups = _make_param_groups()
		if not param_groups:
			return data
		opt = torch.optim.Adam(param_groups)
		_stage_done(f"{label}.build_optimizer", _t)

		# winding_offset_autocrop: compute offset/direction then crop invalid depth layers
		if opt_cfg.args and opt_cfg.args.get("winding_offset_autocrop") and _need_term("winding_vol", stage_eff) > 0:
			_t = _stage_start(f"{label}.winding_offset_autocrop")
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
				all_params, param_groups = _make_param_groups()
				if not param_groups:
					return data
				opt = torch.optim.Adam(param_groups)
			_stage_done(f"{label}.winding_offset_autocrop", _t)

		_status_rows = 0

		def _print_status(*, step_label: str, loss_val: float, tv: dict[str, float], pv: dict[str, float],
						  its: float | None = None) -> None:
			nonlocal _status_rows
			label_map = {
				"pred_dt_gate_gt0": "g>0",
				"pred_dt_gate_gt01": "g>.1",
				"pred_dt_gate_gt05": "g>.5",
				"pred_dt_gate_eq1": "g=1",
				"pred_dt_gate_n_gt0": "n>0",
				"pred_dt_gate_n_gt01": "n>.1",
				"pred_dt_gate_n_gt05": "n>.5",
				"pred_dt_pull_active_frac": "pull%",
				"pred_dt_pull_prefix_mean": "pullpre",
				"pred_dt_pull_weight_mean": "pullw",
			}
			key_order = {
				"pred_dt_gate_gt0": 100,
				"pred_dt_gate_gt01": 101,
				"pred_dt_gate_gt05": 102,
				"pred_dt_gate_eq1": 103,
				"pred_dt_gate_n_gt0": 104,
				"pred_dt_gate_n_gt01": 105,
				"pred_dt_gate_n_gt05": 106,
				"pred_dt_pull_active_frac": 107,
				"pred_dt_pull_prefix_mean": 108,
				"pred_dt_pull_weight_mean": 109,
			}
			def _sort_key(k: str) -> tuple[int, str]:
				return (key_order.get(k, 0), k)
			def _display_key(k: str) -> str:
				return label_map.get(k, k)
			def _fmt_val(k: str, v: float) -> str:
				av = abs(v)
				if av != 0.0 and (av >= 1000.0 or av < 1.0e-3):
					return f"{v:.1e}"
				if av < 10.0:
					return f"{v:.4f}"
				if av < 100.0:
					return f"{v:.3f}"
				return f"{v:.1f}"
			tv_keys = sorted(tv.keys(), key=_sort_key)
			pv_keys = sorted(pv.keys())
			cols = tv_keys + [f"p:{k}" for k in pv_keys]
			values = {k: _fmt_val(k, tv[k]) for k in tv_keys}
			values.update({f"p:{k}": _fmt_val(f"p:{k}", pv[k]) for k in pv_keys})
			widths = {k: max(len(_display_key(k)), len(values[k]), 5) for k in cols}
			if _status_rows % 20 == 0:
				hdr = f"{'step':>16s} {'loss':>8s} {'it/s':>5s}"
				for c in cols:
					hdr += f" {_display_key(c):>{widths[c]}s}"
				print(hdr)
			_status_rows += 1
			its_str = f"{its:5.1f}" if its is not None else f"{'':>5s}"
			row = f"{step_label:>16s} {loss_val:8.4f} {its_str}"
			for k in tv_keys:
				row += f" {values[k]:>{widths[k]}s}"
			for k in pv_keys:
				pk = f"p:{k}"
				row += f" {values[pk]:>{widths[pk]}s}"
			print(row)

		def _print_cylinder_rough_top(rows: list[dict[str, float | int]], *, keep_n: int) -> None:
			before = int(getattr(model, "cyl_params").shape[0])
			print(f"[optimizer] {label}: rough cylinder candidates={before}, keep={keep_n}", flush=True)
			if not rows:
				print(f"[optimizer] {label}: no finite rough cylinder candidates", flush=True)
				return
			params = model.cyl_params.detach().cpu()
			show_center = any("cyl_center" in row for row in rows)
			header = (
				f"{'rank':>4s} {'idx':>5s} {'score':>10s} {'normal':>10s}"
				+ (f" {'center':>10s}" if show_center else "")
				+ f" {'n_avg':>8s} {'n_max':>8s} {'r':>9s} {'ratio':>7s} {'seed':>8s} {'roll':>8s}"
			)
			print(header, flush=True)
			for row in rows:
				idx = int(row["idx"])
				p = params[idx]
				k = float(p[1])
				den = max(1.0e-6, 1.0 - k)
				ratio = (1.0 + k) / den
				line = (
					f"{int(row['rank']):4d} {idx:5d} {float(row['cyl_min']):10.4g} "
					f"{float(row.get('cyl_normal', float('nan'))):10.4g} "
				)
				if show_center:
					line += f"{float(row.get('cyl_center', float('nan'))):10.4g} "
				line += (
					f"{float(row.get('cyl_nerr_avg', float('nan'))):8.3f} "
					f"{float(row.get('cyl_nerr_max', float('nan'))):8.3f} "
					f"{float(p[0]):9.2f} {ratio:7.3f} {float(p[2]):8.3f} {float(p[5]):8.3f}"
				)
				print(line, flush=True)

		def _prune_cylinder_candidates_after_initial_eval() -> bool:
			if not (stage_uses_cyl_loss and is_cyl_stage and getattr(model, "cylinder_enabled", False)):
				return False
			keep_n = 16
			top_rows = opt_loss_cyl.top_candidates(stage_eff, limit=10)
			_print_cylinder_rough_top(top_rows, keep_n=keep_n)
			top_indices = opt_loss_cyl.top_candidate_indices(stage_eff, limit=keep_n)
			before = int(model.cyl_params.shape[0])
			if not top_indices or before <= len(top_indices):
				return False
			kept = model.keep_cylinder_candidates(top_indices)
			print(f"[optimizer] {label}: pruned rough cylinder candidates {before} -> {kept}", flush=True)
			return True

		# Ensure data covers mesh and has all channels needed by this stage
		_needed_channels: set[str] = set()
		if _need_term("pred_dt", stage_eff) > 0:
			_needed_channels.add("pred_dt")
		if _need_term("data", stage_eff) > 0 or _need_term("data_plain", stage_eff) > 0:
			_needed_channels.add("cos")
		if ensure_data_fn is not None:
			_t = _stage_start(f"{label}.ensure_data")
			data = ensure_data_fn(data, _needed_channels)
			_stage_done(f"{label}.ensure_data", _t)

		def _prefetch_loss_points_for_result(res_) -> None:
			if not _active_caches:
				return
			with torch.no_grad():
				_loss_prefetch_items = opt_loss_pred_dt.flow_gate_prefetch_items_for_result(
					res=res_,
					cfg=pred_dt_flow_gate_cfg,
				)
				if stage_uses_cyl_loss:
					_cyl_items = opt_loss_cyl.cyl_normal_prefetch_items_for_result(res=res_)
					for _ch, _pts in _cyl_items.items():
						if _ch in _loss_prefetch_items:
							_loss_prefetch_items[_ch] = torch.cat(
								[_loss_prefetch_items[_ch].reshape(1, 1, -1, 3), _pts.reshape(1, 1, -1, 3)],
								dim=2,
							)
						else:
							_loss_prefetch_items[_ch] = _pts
				if _need_term("ext_offset", stage_eff) > 0:
					_ext_offset_items = opt_loss_winding_density.ext_offset_prefetch_items_for_result(res=res_)
					for _ch, _pts in _ext_offset_items.items():
						if _ch in _loss_prefetch_items:
							_loss_prefetch_items[_ch] = torch.cat(
								[_loss_prefetch_items[_ch].reshape(1, 1, -1, 3), _pts.reshape(1, 1, -1, 3)],
								dim=2,
							)
						else:
							_loss_prefetch_items[_ch] = _pts
			if not _loss_prefetch_items:
				return
			for _cache in _active_caches:
				points = [
					_loss_prefetch_items[ch].reshape(1, 1, -1, 3)
					for ch in _cache.channels
					if ch in _loss_prefetch_items
				]
				if points:
					_pf = torch.cat(points, dim=2) if len(points) > 1 else points[0]
					_sp = data._spacing_for(_cache.channels[0])
					_cache.prefetch(_pf, data.origin_fullres, _sp)
			for _cache in _active_caches:
				if any(ch in _loss_prefetch_items for ch in _cache.channels):
					_cache.sync()

		# Initial evaluation
		def _eval_terms(res_, eff_, *, profile_label: str | None = None):
			"""Evaluate all loss terms, handling both single and multi-loss returns."""
			total = torch.zeros((), device=next(model.parameters()).device, dtype=torch.float32)
			tv: dict[str, float] = {}
			if stage_uses_cyl_loss:
				opt_loss_cyl.reset_candidate_terms()
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
				_t_loss = _stage_start(f"{profile_label}.{name}") if profile_label is not None else None
				result = t["loss"](res=res_)
				_debug_cuda_sync(f"{profile_label}.{name}" if profile_label is not None else name)
				if _t_loss is not None:
					_stage_done(f"{profile_label}.{name}", _t_loss)
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
					if name == "pred_dt":
						tv.update(opt_loss_pred_dt.flow_gate_last_stats())
					total = total + w * lv
			display_loss: float | None = None
			if stage_uses_cyl_loss:
				best_idx, display_loss, display_tv = opt_loss_cyl.display_stats(eff_)
				if best_idx is not None and hasattr(model, "set_best_cylinder_index"):
					model.set_best_cylinder_index(best_idx)
				if display_tv:
					tv.update(display_tv)
			return total, tv, display_loss

		# Streaming mode: filter caches to only those needed by this stage
		# grad_mag/nx/ny are always needed; cos and pred_dt are conditional
		_active_caches = []
		if data.sparse_caches:
			_always_needed = {"grad_mag", "nx", "ny"}
			_stage_channels = _always_needed | _needed_channels
			for _cache in data.sparse_caches.values():
				if _stage_channels & set(_cache.channels):
					_active_caches.append(_cache)
			_active_channels = {
				ch
				for _cache in _active_caches
				for ch in _cache.channels
			}
			_unwanted_optional = (_active_channels & {"cos", "pred_dt"}) - _needed_channels
			if _unwanted_optional:
				raise RuntimeError(
					f"{label}: streaming cache has optional channel(s) not needed by this stage: "
					f"{sorted(_unwanted_optional)}; needed={sorted(_needed_channels)}"
				)

		if is_cyl_stage and bool(getattr(model, "cyl_shell_mode", False)):
			if hasattr(model, "prepare_umbilicus_tube_init"):
				model.prepare_umbilicus_tube_init(data)
			shell_count = int(getattr(model, "cyl_shell_target_count", 3))
			max_steps = int(opt_cfg.steps)

			def _prefetch_shell_model_points() -> None:
				if not _active_caches:
					return
				with torch.no_grad():
					_xyz_lr_pf = model._grid_xyz()
					_xyz_hr_pf = model._grid_xyz_hr(_xyz_lr_pf)
				for _cache in _active_caches:
					_sp = data._spacing_for(_cache.channels[0])
					_cache.prefetch(_xyz_hr_pf, data.origin_fullres, _sp)
					_cache.prefetch(_xyz_lr_pf, data.origin_fullres, _sp)
				for _cache in _active_caches:
					_cache.sync()

			def _shell_param_values() -> dict[str, float]:
				if not hasattr(model, "_shell_offset_stats"):
					return {}
				avg, mn, mx = model._shell_offset_stats()
				return {"off_avg": avg, "off_min": mn, "off_max": mx}

			for shell_i in range(shell_count):
				if hasattr(model, "begin_cylinder_shell"):
					model.begin_cylinder_shell(shell_i, data)
				all_params, param_groups = _make_param_groups()
				if not param_groups:
					return data
				opt = torch.optim.Adam(param_groups)
				shell_label = f"{label}.shell{shell_i + 1}"

				_t = _stage_start(f"{shell_label}.initial_eval")
				_prefetch_shell_model_points()
				with torch.no_grad():
					res0 = model(data)
					_prefetch_loss_points_for_result(res0)
					loss0, term_vals0, display_loss0 = _eval_terms(
						res0, stage_eff, profile_label=f"{shell_label}.initial_eval.loss")
				term_vals0 = {k: round(v, 4) for k, v in term_vals0.items()}
				_print_status(
					step_label=f"{shell_label} 0/{max_steps}",
					loss_val=float(display_loss0) if display_loss0 is not None else loss0.item(),
					tv=term_vals0,
					pv=_shell_param_values(),
				)
				snapshot_fn(stage=shell_label.replace(".", "_"), step=0,
							loss=float(loss0.detach().cpu()), data=data, res=res0)
				_stage_done(f"{shell_label}.initial_eval", _t)

				loss = loss0
				display_loss = display_loss0
				res = res0
				_t_wall_start = time.perf_counter()
				_t_steps_acc = 0
				for step in range(max_steps):
					if _active_caches:
						for _cache in _active_caches:
							_cache.sync()
					if fit_data.CHUNK_STATS_ENABLED:
						fit_data._chunk_stats.begin_iteration()
					_prefetch_shell_model_points()
					res = model(data)
					_prefetch_loss_points_for_result(res)
					loss, term_vals, display_loss = _eval_terms(res, stage_eff)
					if fit_data.CHUNK_STATS_ENABLED:
						fit_data._chunk_stats.end_iteration()

					opt.zero_grad(set_to_none=True)
					loss.backward()
					opt.step()

					if _active_caches:
						_prefetch_shell_model_points()
						for _cache in _active_caches:
							_cache.end_iteration()

					step1 = step + 1
					_t_steps_acc += 1
					_done_steps[0] += 1
					_stage_progress = step1 / max_steps if max_steps > 0 else 1.0
					_overall_progress = (
						(si + (shell_i + _stage_progress) / max(1, shell_count)) / _num_stages
						if _num_stages > 0 else 1.0
					)
					if progress_fn is not None:
						progress_fn(
							step=_done_steps[0], total=_total_steps,
							loss=float(display_loss) if display_loss is not None else float(loss.detach().cpu()),
							stage_progress=_stage_progress, overall_progress=_overall_progress,
							stage_name=stage.name,
						)
					if step == 0 or step1 == max_steps or (status_interval > 0 and (step1 % status_interval) == 0):
						term_vals = {k: round(v, 4) for k, v in term_vals.items()}
						_t_wall_now = time.perf_counter()
						_t_wall_elapsed = _t_wall_now - _t_wall_start
						_its = _t_steps_acc / _t_wall_elapsed if _t_wall_elapsed > 0 else None
						_print_status(
							step_label=f"{shell_label} {step1}/{max_steps}",
							loss_val=float(display_loss) if display_loss is not None else loss.item(),
							tv=term_vals,
							pv=_shell_param_values(),
							its=_its,
						)
						_t_steps_acc = 0
						_t_wall_start = _t_wall_now
					if snap_int > 0 and (step1 % snap_int) == 0:
						snapshot_fn(stage=shell_label.replace(".", "_"), step=step1,
									loss=float(loss.detach().cpu()), data=data, res=res)

				if hasattr(model, "complete_current_cylinder_shell"):
					model.complete_current_cylinder_shell(data)
				snapshot_fn(stage=shell_label.replace(".", "_"), step=max_steps,
							loss=float(loss.detach().cpu()), data=data, res=res)

			_stage_done(f"{label}.total", _t_stage_total)
			return data

		# Initial prefetch for streaming mode
		if _active_caches:
			_t = _stage_start(f"{label}.initial_prefetch")
			with torch.no_grad():
				_xyz_lr_pf = model._grid_xyz()
				_xyz_hr_pf = model._grid_xyz_hr(_xyz_lr_pf)
				_pred_dt_extra_pf = opt_loss_pred_dt.flow_gate_prefetch_points(
					data=data,
					xyz_hr=_xyz_hr_pf,
					xyz_lr=_xyz_lr_pf,
					cfg=pred_dt_flow_gate_cfg,
				)
				_cyl_pf = None
				if stage_uses_cyl_loss and getattr(model, "cylinder_enabled", False):
					_cyl_pf, _ = model.cylinder_samples()
					_cyl_pf = _cyl_pf.detach()
			for _cache in _active_caches:
				_sp = data._spacing_for(_cache.channels[0])
				_cache.prefetch(_xyz_hr_pf, data.origin_fullres, _sp)
				if _pred_dt_extra_pf is not None and "pred_dt" in _cache.channels:
					_cache.prefetch(_pred_dt_extra_pf, data.origin_fullres, _sp)
				if _cyl_pf is not None and {"grad_mag", "nx", "ny"} & set(_cache.channels):
					_cache.prefetch(_cyl_pf, data.origin_fullres, _sp)
			# Also prefetch chunks for corr points (static positions, loaded once)
			if data.corr_points is not None and data.corr_points.points_xyz_winda.shape[0] > 0:
				_corr_xyz = data.corr_points.points_xyz_winda[:, :3].to(
					device=next(model.parameters()).device, dtype=torch.float32)
				for _cache in _active_caches:
					_sp = data._spacing_for(_cache.channels[0])
					_cache.prefetch(_corr_xyz, data.origin_fullres, _sp)
			for _cache in _active_caches:
				_cache.sync()
			_stage_done(f"{label}.initial_prefetch", _t)

		# Station-keeping: set seed point anchor (once, on first stage that uses it)
		# Must be AFTER prefetch+sync so grid_sample_fullres can read loaded chunks.
		if (_need_term("station_n", stage_eff) > 0 or _need_term("station_t", stage_eff) > 0) and seed_xyz is not None:
			_t = _stage_start(f"{label}.station_seed")
			dev = next(model.parameters()).device
			seed_t = torch.tensor(list(seed_xyz), device=dev, dtype=torch.float32)
			opt_loss_station.set_seed(seed_t, data, Hm=model.mesh_h, Wm=model.mesh_w, D=model.depth)
			_stage_done(f"{label}.station_seed", _t)

		_t = _stage_start(f"{label}.initial_eval")
		with torch.no_grad():
			_t_forward = _stage_start(f"{label}.initial_eval.model_forward")
			res0 = model(data)
			_debug_cuda_sync(f"{label}.initial_eval.model_forward")
			_stage_done(f"{label}.initial_eval.model_forward", _t_forward)
			_t_loss_prefetch = _stage_start(f"{label}.initial_eval.loss_prefetch")
			_prefetch_loss_points_for_result(res0)
			_debug_cuda_sync(f"{label}.initial_eval.loss_prefetch")
			_stage_done(f"{label}.initial_eval.loss_prefetch", _t_loss_prefetch)
			_t_terms = _stage_start(f"{label}.initial_eval.loss_terms")
			loss0, term_vals0, display_loss0 = _eval_terms(
				res0, stage_eff, profile_label=f"{label}.initial_eval.loss")
			_stage_done(f"{label}.initial_eval.loss_terms", _t_terms)
			_t_prune = _stage_start(f"{label}.initial_eval.cylinder_prune")
			if _prune_cylinder_candidates_after_initial_eval():
				all_params, param_groups = _make_param_groups()
				if not param_groups:
					return data
				opt = torch.optim.Adam(param_groups)
			_stage_done(f"{label}.initial_eval.cylinder_prune", _t_prune)
			_t_params = _stage_start(f"{label}.initial_eval.param_values")
			param_vals0: dict[str, float] = {}
			for k, vs in all_params.items():
				if len(vs) == 1 and vs[0].numel() == 1:
					param_vals0[k] = float(vs[0].detach().cpu())
			_stage_done(f"{label}.initial_eval.param_values", _t_params)
			_t_status = _stage_start(f"{label}.initial_eval.status_print")
			term_vals0 = {k: round(v, 4) for k, v in term_vals0.items()}
			param_vals0 = {k: round(v, 4) for k, v in param_vals0.items()}
			_print_status(
				step_label=f"{label} 0/{opt_cfg.steps}",
				loss_val=float(display_loss0) if display_loss0 is not None else loss0.item(),
				tv=term_vals0,
				pv=param_vals0,
			)
			_stage_done(f"{label}.initial_eval.status_print", _t_status)
			# Print corr detail after initial eval (first stage only)
			if not _corr_start_printed[0] and "corr" in term_vals0:
				opt_loss_corr.print_detail("START")
				_corr_start_printed[0] = True
		_stage_done(f"{label}.initial_eval", _t)
		_t = _stage_start(f"{label}.initial_snapshot")
		snapshot_fn(stage=label, step=0, loss=float(loss0.detach().cpu()), data=data, res=res0)
		_stage_done(f"{label}.initial_snapshot", _t)

		max_steps = opt_cfg.steps
		_t_wall_start = time.perf_counter()
		_t_steps_acc = 0
		loss = loss0
		display_loss = display_loss0
		res = res0
		_flow_timing = None
		if (
			pred_dt_flow_gate_cfg is not None
			and bool(pred_dt_flow_gate_cfg.get("enabled", False))
			and _need_term("pred_dt", stage_eff) > 0
			and _flow_timing_enabled(pred_dt_flow_gate_cfg)
		):
			_flow_timing = _FlowTimingWindow(interval=100)

		for step in range(max_steps):
			_t_iter = time.perf_counter()
			# Sync: wait for chunks loaded by last prefetch
			_t_io = time.perf_counter()
			if _active_caches:
				for _cache in _active_caches:
					_cache.sync()
			if _flow_timing is not None:
				_flow_timing.add("io_prefetch", time.perf_counter() - _t_io)

			if fit_data.CHUNK_STATS_ENABLED:
				fit_data._chunk_stats.begin_iteration()
			_t_forward = time.perf_counter()
			res = model(data)
			_debug_cuda_sync(f"{label}.{step + 1}.model_forward")
			if _flow_timing is not None:
				_timing_cuda_sync()
				_flow_timing.add("model_forward", time.perf_counter() - _t_forward)
			_t_io = time.perf_counter()
			_prefetch_loss_points_for_result(res)
			_debug_cuda_sync(f"{label}.{step + 1}.loss_prefetch")
			if _flow_timing is not None:
				_flow_timing.add("io_prefetch", time.perf_counter() - _t_io)
			_t_loss_eval = time.perf_counter()
			loss, term_vals, display_loss = _eval_terms(res, stage_eff)
			if _flow_timing is not None:
				_timing_cuda_sync()
				_flow_timing.add("loss_eval", time.perf_counter() - _t_loss_eval)
				_flow_parts = opt_loss_pred_dt.flow_gate_last_timing()
				_flow_timing.add("flow_sampling", _flow_parts.get("flow_sampling", 0.0))
				_flow_timing.add("flow_calc", _flow_parts.get("flow_calc", 0.0))
			if fit_data.CHUNK_STATS_ENABLED:
				fit_data._chunk_stats.end_iteration()

			_t_opt = time.perf_counter()
			opt.zero_grad(set_to_none=True)
			loss.backward()
			opt.step()
			model.update_conn_offsets()
			model.update_ext_conn_offsets()
			if _flow_timing is not None:
				_timing_cuda_sync()
				_flow_timing.add("opt_step", time.perf_counter() - _t_opt)

			# Prefetch: predict next iteration's chunks from updated mesh
			_t_io = time.perf_counter()
			if _active_caches:
				with torch.no_grad():
					_xyz_lr_pf = model._grid_xyz()
					_xyz_hr_pf = model._grid_xyz_hr(_xyz_lr_pf)
					_pred_dt_extra_pf = opt_loss_pred_dt.flow_gate_prefetch_points(
						data=data,
						xyz_hr=_xyz_hr_pf,
						xyz_lr=_xyz_lr_pf,
						cfg=pred_dt_flow_gate_cfg,
					)
					_cyl_pf = None
					if stage_uses_cyl_loss and getattr(model, "cylinder_enabled", False):
						_cyl_pf, _ = model.cylinder_samples()
						_cyl_pf = _cyl_pf.detach()
				for _cache in _active_caches:
					_sp = data._spacing_for(_cache.channels[0])
					_cache.prefetch(_xyz_hr_pf, data.origin_fullres, _sp)
					if _pred_dt_extra_pf is not None and "pred_dt" in _cache.channels:
						_cache.prefetch(_pred_dt_extra_pf, data.origin_fullres, _sp)
					if _cyl_pf is not None and {"grad_mag", "nx", "ny"} & set(_cache.channels):
						_cache.prefetch(_cyl_pf, data.origin_fullres, _sp)
				for _cache in _active_caches:
					_cache.end_iteration()
			if _flow_timing is not None:
				_flow_timing.add("io_prefetch", time.perf_counter() - _t_io)
			_t_steps_acc += 1
			_done_steps[0] += 1

			step1 = step + 1
			_stage_progress = step1 / max_steps if max_steps > 0 else 1.0
			_overall_progress = (si + _stage_progress) / _num_stages if _num_stages > 0 else 1.0

			if progress_fn is not None:
				progress_fn(
					step=_done_steps[0], total=_total_steps,
					loss=float(display_loss) if display_loss is not None else float(loss.detach().cpu()),
					stage_progress=_stage_progress, overall_progress=_overall_progress,
					stage_name=stage.name,
				)

			if step == 0 or step1 == max_steps or (status_interval > 0 and (step1 % status_interval) == 0):
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
							  loss_val=float(display_loss) if display_loss is not None else loss.item(),
							  tv=term_vals, pv=param_vals, its=_its)
				_t_steps_acc = 0
				_t_wall_start = _t_wall_now

			if ensure_data_fn is not None and (step1 % 100) == 0:
				_t_io = time.perf_counter()
				data = ensure_data_fn(data, _needed_channels)
				if _flow_timing is not None:
					_flow_timing.add("io_prefetch", time.perf_counter() - _t_io)

			if _flow_timing is not None:
				_flow_timing.add("total", time.perf_counter() - _t_iter)
				_flow_timing.finish_iter(label=label, step1=step1, max_steps=max_steps)

			if snap_int > 0 and (step1 % snap_int) == 0:
				snapshot_fn(stage=label, step=step1, loss=float(loss.detach().cpu()), data=data, res=res)

		_t = _stage_start(f"{label}.final_snapshot")
		snapshot_fn(stage=label, step=max_steps, loss=float(loss.detach().cpu()), data=data, res=res)
		_stage_done(f"{label}.final_snapshot", _t)
		_stage_done(f"{label}.total", _t_stage_total)
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
		run_zero_step_cyl = "cyl_params" in stage.global_opt.params
		if stage.global_opt.steps > 0 or run_zero_step_cyl:
			data = _run_opt(si=si, label=f"stage{si}", stage=stage, opt_cfg=stage.global_opt, data=data)
			if active_corr:
				opt_loss_corr.print_detail(f"stage{si} END")
				opt_loss_corr.print_summary()

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
