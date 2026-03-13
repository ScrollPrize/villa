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
	valid = {"mesh_ms", "conn_offset_ms", "amp", "bias",
			 "arc_cx", "arc_cy", "arc_radius", "arc_angle0", "arc_angle1",
			 "straight_cx", "straight_cy", "straight_angle", "straight_half_w"}
	bad_params = sorted(set(params) - valid)
	if bad_params:
		raise ValueError(f"stages_json: stage '{stage_name}' opt.params: unknown name(s): {bad_params}")
	min_scaledown = max(0, int(opt_cfg.get("min_scaledown", 0)))
	default_mul = opt_cfg.get("default_mul", None)
	w_fac = opt_cfg.get("w_fac", None)
	opt_cfg.pop("steps", None)
	opt_cfg.pop("lr", None)
	opt_cfg.pop("params", None)
	opt_cfg.pop("min_scaledown", None)
	opt_cfg.pop("default_mul", None)
	opt_cfg.pop("w_fac", None)
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
	)


lambda_global: dict[str, float] = {
	"normal": 1.0,
	"step": 0.0,
	"smooth": 0.0,
	"winding_density": 0.0,
	"normal": 0.0,
	"data": 0.0,
	"data_plain": 0.0,
	"pred_dt": 0.0,
	"corr": 0.0,
	"winding_vol": 0.0,
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
	load_data_fn=None,
	volume_extent_fullres: tuple[int, int, int] | None = None,
) -> fit_data.FitData3D:

	terms = {
		"step": {"loss": opt_loss_step.step_loss},
		"smooth": {"loss": opt_loss_smooth.smooth_loss},
		"winding_density": {"loss": opt_loss_winding_density.winding_density_loss},
		"normal": {"loss": opt_loss_dir.normal_loss},
		"data": {"loss": opt_loss_data.data_loss},
		"data_plain": {"loss": opt_loss_data.data_plain_loss},
		"pred_dt": {"loss": opt_loss_pred_dt.pred_dt_loss},
		"corr": {"loss": opt_loss_corr.corr_loss},
		"winding_vol": {"loss": opt_loss_winding_volume.winding_volume_loss},
	}

	def _run_opt(*, si: int, label: str, stage: Stage, opt_cfg: OptSettings, data: fit_data.FitData3D) -> fit_data.FitData3D:
		if opt_cfg.steps <= 0:
			return data

		# If arc/straight params not in optimized set, bake into mesh
		arc_params_set = {"arc_cx", "arc_cy", "arc_radius", "arc_angle0", "arc_angle1"}
		if not arc_params_set.intersection(opt_cfg.params):
			if hasattr(model, "arc_enabled") and model.arc_enabled:
				model.bake_arc_into_mesh()
		straight_params_set = {"straight_cx", "straight_cy", "straight_angle", "straight_half_w"}
		if not straight_params_set.intersection(opt_cfg.params):
			if hasattr(model, "straight_enabled") and model.straight_enabled:
				model.bake_straight_into_mesh()

		# If min_scaledown > 0, reconstruct pyramid and zero fine levels
		if "mesh_ms" in opt_cfg.params and opt_cfg.min_scaledown > 0:
			k0 = opt_cfg.min_scaledown
			with torch.no_grad():
				flat = model._integrate_pyramid_3d(model.mesh_ms, pyramid_d=model.pyramid_d)
				model.mesh_ms = model._construct_pyramid_from_flat_3d(
					flat, len(model.mesh_ms), pyramid_d=model.pyramid_d)
				for pi in range(min(k0, len(model.mesh_ms))):
					model.mesh_ms[pi].zero_()

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
			return
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

		# Initial evaluation
		with torch.no_grad():
			res0 = model(data)
			loss0 = torch.zeros((), device=data.cos.device, dtype=data.cos.dtype)
			term_vals0: dict[str, float] = {}
			for name, t in terms.items():
				w = _need_term(name, opt_cfg.eff)
				if w == 0.0:
					continue
				lv, lms, masks = t["loss"](res=res0)
				term_vals0[name] = float(lv.detach().cpu())
				loss0 = loss0 + w * lv
			param_vals0: dict[str, float] = {}
			for k, vs in all_params.items():
				if len(vs) == 1 and vs[0].numel() == 1:
					param_vals0[k] = float(vs[0].detach().cpu())
			term_vals0 = {k: round(v, 4) for k, v in term_vals0.items()}
			param_vals0 = {k: round(v, 4) for k, v in param_vals0.items()}
			_print_status(step_label=f"{label} 0/{opt_cfg.steps}", loss_val=loss0.item(), tv=term_vals0, pv=param_vals0)
		snapshot_fn(stage=label, step=0, loss=float(loss0.detach().cpu()), data=data, res=res0)

		max_steps = opt_cfg.steps
		_t_wall_start = time.perf_counter()
		_t_steps_acc = 0
		loss = loss0

		for step in range(max_steps):
			res = model(data)
			loss = torch.zeros((), device=data.cos.device, dtype=data.cos.dtype)
			term_vals: dict[str, float] = {}
			for name, t in terms.items():
				w = _need_term(name, opt_cfg.eff)
				if w == 0.0:
					continue
				lv, lms, masks = t["loss"](res=res)
				term_vals[name] = float(lv.detach().cpu())
				loss = loss + w * lv

			opt.zero_grad(set_to_none=True)
			loss.backward()
			opt.step()
			model.update_conn_offsets()
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

			if load_data_fn is not None and (step1 % 100) == 0 and check_data_bounds(model, data, volume_extent_fullres=volume_extent_fullres):
				print(f"[optimizer] mesh near data border at step {step1}, reloading data", flush=True)
				data = load_data_fn()

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
	has_corr_pts = data.corr_points is not None and data.corr_points.points_xyz_winda.shape[0] > 0
	corr_weights = [(_need_term("corr", s.global_opt.eff), s.name) for s in stages if s.global_opt.steps > 0]
	print(f"[optimizer] corr_points={has_corr_pts}"
		  f" snap_mode={opt_loss_corr._snap_mode}"
		  f" corr_weights={corr_weights}", flush=True)
	if has_corr_pts:
		cp = data.corr_points
		n = cp.points_xyz_winda.shape[0]
		print(f"[optimizer] {n} corr points", flush=True)
		if all(w == 0.0 for w, _ in corr_weights):
			print(f"[optimizer] WARNING: corr points loaded but corr weight is 0 in all stages!", flush=True)

	for si, stage in enumerate(stages):
		if stage.global_opt.steps > 0:
			data = _run_opt(si=si, label=f"stage{si}", stage=stage, opt_cfg=stage.global_opt, data=data)

	if any(_need_term("corr", stage.global_opt.eff) > 0 for stage in stages if stage.global_opt.steps > 0):
		opt_loss_corr.print_summary()
	elif has_corr_pts:
		print("[optimizer] corr points present but corr weight=0, no corr loss computed", flush=True)

	return data
