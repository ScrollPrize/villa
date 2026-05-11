from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

import torch

import cli_data
import fit_data
import model as fit_model
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
		cyl_loss_names = (
			"cyl_normal", "cyl_center", "cyl_smooth", "cyl_z_smooth", "cyl_step",
			"cyl_radial_mean", "cyl_bend", "cyl_conn_mesh", "cyl_conn_gt", "cyl_base_mesh", "cyl_base_gt",
		)
		if not any(float(eff.get(name, 0.0)) != 0.0 for name in cyl_loss_names):
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
	"cyl_smooth": 0.0,
	"cyl_z_smooth": 0.0,
	"cyl_step": 0.0,
	"cyl_radial_mean": 0.0,
	"cyl_bend": 0.0,
	"cyl_conn_mesh": 0.0,
	"cyl_conn_gt": 0.0,
	"cyl_base_mesh": 0.0,
	"cyl_base_gt": 0.0,
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

	def _opt_timing_enabled(stage_args_: dict) -> bool:
		if _truthy(os.environ.get("LASAGNA_OPT_TIMING")):
			return True
		if not isinstance(stage_args_, dict):
			return False
		return _truthy(stage_args_.get("profile_opt", False))

	def _opt_timing_interval(stage_args_: dict, *, fallback: int) -> int:
		raw = os.environ.get("LASAGNA_OPT_TIMING_INTERVAL", None)
		if raw is None and isinstance(stage_args_, dict):
			raw = stage_args_.get("profile_interval", stage_args_.get("profile_opt_interval", None))
		if raw is None:
			raw = fallback
		return max(1, int(raw))

	def _opt_timing_sync_cuda(stage_args_: dict) -> bool:
		raw = os.environ.get("LASAGNA_OPT_TIMING_SYNC_CUDA", None)
		if raw is None and isinstance(stage_args_, dict):
			raw = stage_args_.get("profile_cuda_sync", True)
		return _truthy(raw)

	def _is_cyl_stage(stage_: Stage) -> bool:
		return "cyl_params" in stage_.global_opt.params

	def _next_stage_is_cyl(si_: int) -> bool:
		return si_ + 1 < len(stages) and _is_cyl_stage(stages[si_ + 1])

	def _shell_pass_count(shell_i_: int) -> int:
		if hasattr(model, "cylinder_shell_pass_count"):
			return max(1, int(model.cylinder_shell_pass_count(int(shell_i_))))
		return 1

	def _scheduled_total_steps() -> int:
		if not bool(getattr(model, "cyl_shell_mode", False)):
			return total_steps_for_stages(stages)
		shell_count_ = max(0, int(getattr(model, "cyl_shell_target_count", 0)))
		next_shell_ = 0
		total = 0
		for si_, stage_ in enumerate(stages):
			steps_ = max(0, int(stage_.global_opt.steps))
			if _is_cyl_stage(stage_):
				if next_shell_ >= shell_count_:
					continue
				run_shells_ = 1 if _next_stage_is_cyl(si_) else shell_count_ - next_shell_
				run_shells_ = max(0, min(run_shells_, shell_count_ - next_shell_))
				total += steps_ * sum(_shell_pass_count(shell_i_) for shell_i_ in range(next_shell_, next_shell_ + run_shells_))
				next_shell_ += run_shells_
			else:
				total += steps_
		return total

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

	class _OptTimingWindow:
		_ORDER = [
			"cache_sync",
			"model_point_prefetch",
			"model_forward",
			"loss_prefetch",
			"loss_eval",
			"chunk_stats",
			"zero_grad",
			"backward",
			"optimizer_step",
			"model_updates",
			"next_prefetch",
			"progress",
			"status",
			"ensure_data",
			"snapshot",
		]

		def __init__(self, *, interval: int = 100, sync_cuda: bool = True) -> None:
			self.interval = max(1, int(interval))
			self.sync_cuda = bool(sync_cuda)
			self.count = 0
			self.acc: dict[str, float] = {"total": 0.0}

		def sync(self) -> None:
			if self.sync_cuda:
				_timing_cuda_sync()

		def add(self, key: str, seconds: float) -> None:
			self.acc[key] = self.acc.get(key, 0.0) + max(0.0, float(seconds))

		def finish_iter(self, *, label: str, step1: int, max_steps: int) -> None:
			self.count += 1
			if (step1 % self.interval) != 0 and step1 != max_steps:
				return
			if self.count <= 0:
				return
			total = max(1.0e-12, self.acc.get("total", 0.0))
			primary_keys = [k for k in self._ORDER if self.acc.get(k, 0.0) > 0.0]
			extra_primary = sorted(
				k for k, v in self.acc.items()
				if k not in {"total", *primary_keys} and not k.startswith("loss:") and v > 0.0
			)
			loss_keys = sorted(k for k, v in self.acc.items() if k.startswith("loss:") and v > 0.0)
			measured = sum(self.acc.get(k, 0.0) for k in primary_keys + extra_primary)
			other = max(0.0, total - measured)
			print(
				f"[opt_timing] {label} {step1}/{max_steps} over {self.count} iters "
				f"sync_cuda={int(self.sync_cuda)}",
				flush=True,
			)
			print(f"{'part':<24s} {'runtime_%':>9s} {'ms/it':>10s} {'total_s':>10s}", flush=True)
			for key in primary_keys + extra_primary:
				seconds = self.acc.get(key, 0.0)
				print(
					f"{key:<24s} {100.0 * seconds / total:9.2f} "
					f"{1000.0 * seconds / float(self.count):10.2f} {seconds:10.3f}",
					flush=True,
				)
			if other > 0.0:
				print(
					f"{'other':<24s} {100.0 * other / total:9.2f} "
					f"{1000.0 * other / float(self.count):10.2f} {other:10.3f}",
					flush=True,
				)
			if loss_keys:
				print("[opt_timing] loss term breakdown is inside loss_eval", flush=True)
				for key in loss_keys:
					seconds = self.acc.get(key, 0.0)
					print(
						f"{key:<24s} {100.0 * seconds / total:9.2f} "
						f"{1000.0 * seconds / float(self.count):10.2f} {seconds:10.3f}",
						flush=True,
					)
			self.count = 0
			for key in list(self.acc.keys()):
				self.acc[key] = 0.0

	Needs = fit_model.ModelForwardNeeds
	terms = {
		"step": {"loss": opt_loss_step.step_loss, "needs": Needs()},
		"smooth": {"loss": opt_loss_smooth.smooth_loss, "needs": Needs()},
		"winding_density": {
			"loss": opt_loss_winding_density.winding_density_loss,
			"min_depth": 2,
			"needs": Needs(xyz_hr=True, mesh_conn=True),
		},
		"normal": {
			"loss": opt_loss_dir.normal_loss,
			"needs": Needs(
				lr_data_channels=frozenset({"grad_mag", "nx", "ny"}),
				lr_prefetch_channels=frozenset({"grad_mag", "nx", "ny"}),
			),
		},
		"data": {
			"loss": opt_loss_data.data_loss,
			"needs": Needs(
				xyz_hr=True,
				xyz_hr_grad=True,
				hr_data_channels=frozenset({"grad_mag"}),
				hr_prefetch_channels=frozenset({"grad_mag"}),
				hr_prefetch_grad_channels=frozenset({"cos"}),
				target=True,
			),
		},
		"data_plain": {
			"loss": opt_loss_data.data_plain_loss,
			"needs": Needs(
				xyz_hr=True,
				xyz_hr_grad=True,
				hr_data_channels=frozenset({"grad_mag"}),
				hr_prefetch_channels=frozenset({"grad_mag"}),
				hr_prefetch_grad_channels=frozenset({"cos"}),
				target=True,
			),
		},
		"pred_dt": {
			"loss": opt_loss_pred_dt.pred_dt_loss,
			"needs": Needs(
				xyz_hr=True,
				hr_data_channels=frozenset({"pred_dt"}),
				hr_prefetch_channels=frozenset({"pred_dt"}),
				lr_data_channels=frozenset({"grad_mag"}),
				lr_prefetch_channels=frozenset({"grad_mag"}),
				prefetch_pred_dt_loss=True,
			),
		},
		"corr": {
			"loss": opt_loss_corr.corr_winding_loss,
			"needs": Needs(mesh_normals=True, prefetch_corr_points=True),
		},
		"winding_vol": {
			"loss": opt_loss_winding_volume.winding_volume_loss,
			"needs": Needs(
				lr_data_channels=frozenset({"grad_mag"}),
				lr_prefetch_channels=frozenset({"grad_mag"}),
			),
		},
		"station": {
			"loss": opt_loss_station.station_loss,
			"sub": ["station_n", "station_t"],
			"needs": Needs(
				lr_data_channels=frozenset({"grad_mag"}),
				lr_prefetch_channels=frozenset({"grad_mag"}),
			),
		},
		"bend": {"loss": opt_loss_bend.bend_loss, "needs": Needs()},
		"ext_offset": {
			"loss": opt_loss_winding_density.ext_offset_loss,
			"needs": Needs(ext_conn=True, prefetch_ext_offset=True),
		},
		"cyl_normal": {
			"loss": opt_loss_cyl.cyl_normal_loss,
			"needs": Needs(
				cyl_samples=True,
				cyl_normals=True,
				cyl_shell_fields=True,
				prefetch_cyl_gt_normals=True,
			),
		},
		"cyl_center": {
			"loss": opt_loss_cyl.cyl_center_loss,
			"needs": Needs(
				cyl_samples=True,
				cyl_normals=True,
				cyl_centers_axes=True,
				prefetch_cyl_gt_normals=True,
			),
		},
		"cyl_smooth": {"loss": opt_loss_cyl.cyl_smooth_loss, "needs": Needs(cyl_samples=True)},
		"cyl_z_smooth": {"loss": opt_loss_cyl.cyl_z_smooth_loss, "needs": Needs(cyl_samples=True)},
		"cyl_step": {"loss": opt_loss_cyl.cyl_step_loss, "needs": Needs(cyl_samples=True)},
		"cyl_radial_mean": {
			"loss": opt_loss_cyl.cyl_radial_mean_loss,
			"needs": Needs(cyl_samples=True, cyl_shell_fields=True),
		},
		"cyl_bend": {"loss": opt_loss_cyl.cyl_bend_loss, "needs": Needs(cyl_samples=True)},
		"cyl_conn_mesh": {
			"loss": opt_loss_cyl.cyl_conn_mesh_loss,
			"needs": Needs(cyl_samples=True, cyl_shell_fields=True),
		},
		"cyl_conn_gt": {
			"loss": opt_loss_cyl.cyl_conn_gt_loss,
			"needs": Needs(cyl_samples=True, cyl_shell_fields=True, prefetch_cyl_gt_normals=True),
		},
		"cyl_base_mesh": {
			"loss": opt_loss_cyl.cyl_base_mesh_loss,
			"needs": Needs(cyl_samples=True, cyl_shell_fields=True),
		},
		"cyl_base_gt": {
			"loss": opt_loss_cyl.cyl_base_gt_loss,
			"needs": Needs(cyl_samples=True, cyl_shell_fields=True, prefetch_cyl_gt_normals=True),
		},
	}

	_corr_start_printed = [False]

	def _is_term_active(name: str, t: dict, eff: dict[str, float]) -> bool:
		sub_names = t.get("sub")
		if sub_names:
			return any(_need_term(s, eff) > 0 for s in sub_names)
		return _need_term(name, eff) > 0

	def _needs_for_eff(
		eff: dict[str, float],
		*,
		pred_dt_flow_gate_cfg_: dict | None,
		pred_dt_normal_source_: object,
	) -> fit_model.ModelForwardNeeds:
		needs = Needs()
		for name, t in terms.items():
			if not _is_term_active(name, t, eff):
				continue
			if name.startswith("cyl_") and not bool(getattr(model, "cylinder_enabled", False)):
				continue
			needs = needs.merged(t.get("needs", Needs()))
			if name == "pred_dt":
				if str(pred_dt_normal_source_ or "model").strip().lower() == "gt":
					needs = needs.merged(Needs(
						lr_data_channels=frozenset({"grad_mag", "nx", "ny"}),
						lr_prefetch_channels=frozenset({"grad_mag", "nx", "ny"}),
					))
				if isinstance(pred_dt_flow_gate_cfg_, dict) and bool(pred_dt_flow_gate_cfg_.get("enabled", False)):
					needs = needs.merged(Needs(
						xyz_hr=True,
						hr_data_channels=frozenset({"pred_dt"}),
						hr_prefetch_channels=frozenset({"pred_dt"}),
						prefetch_pred_dt_flow=True,
					))
		return needs

	def _prefetch_grad_summary(needs: fit_model.ModelForwardNeeds) -> str:
		grad_channels, nograd_channels = needs.prefetch_channels_by_position_grad()
		return (
			f"prefetch_grad_channels={sorted(grad_channels)} "
			f"prefetch_nograd_channels={sorted(nograd_channels)}"
		)

	def _missing_loss_fields(
		*,
		name: str,
		required: fit_model.ModelForwardNeeds,
		res_: fit_model.FitResult3D,
	) -> list[str]:
		missing: list[str] = []
		if required.xyz_hr and res_.xyz_hr is None:
			missing.append("xyz_hr")
		if required.hr_data_channels:
			if res_.data_s is None:
				missing.append(f"data_s[{','.join(sorted(required.hr_data_channels))}]")
			else:
				for ch in sorted(required.hr_data_channels):
					if getattr(res_.data_s, ch, None) is None:
						missing.append(f"data_s.{ch}")
		if required.lr_data_channels:
			if res_.data_lr is None:
				missing.append(f"data_lr[{','.join(sorted(required.lr_data_channels))}]")
			else:
				for ch in sorted(required.lr_data_channels):
					if getattr(res_.data_lr, ch, None) is None:
						missing.append(f"data_lr.{ch}")
		if required.target and (res_.target_plain is None or res_.target_mod is None):
			missing.append("target_plain/target_mod")
		if required.mesh_conn:
			if res_.xy_conn is None:
				missing.append("xy_conn")
			if res_.mask_conn is None:
				missing.append("mask_conn")
			if res_.sign_conn is None:
				missing.append("sign_conn")
		if required.mesh_normals and res_.normals is None:
			missing.append("normals")
		if required.ext_conn and res_.ext_conn is None:
			missing.append("ext_conn")
		cyl_active = bool(getattr(model, "cylinder_enabled", False))
		if cyl_active:
			if required.cyl_samples and (res_.cyl_xyz is None or res_.cyl_count <= 0):
				missing.append("cyl_xyz")
			if required.cyl_normals and res_.cyl_normals is None:
				missing.append("cyl_normals")
			if required.cyl_centers_axes and not bool(getattr(res_, "cyl_shell_mode", False)):
				if res_.cyl_centers is None:
					missing.append("cyl_centers")
				if res_.cyl_axes is None:
					missing.append("cyl_axes")
			if required.cyl_shell_fields and bool(getattr(res_, "cyl_shell_mode", False)):
				if res_.cyl_shell_delta_xy is None:
					missing.append("cyl_shell_delta_xy")
		if missing:
			return [f"{name}: {field}" for field in missing]
		return missing

	def _add_prefetch_items(
		dst: dict[str, torch.Tensor],
		src: dict[str, torch.Tensor] | None,
	) -> None:
		if not src:
			return
		for ch, pts in src.items():
			if ch in dst:
				dst[ch] = torch.cat(
					[dst[ch].reshape(1, 1, -1, 3), pts.reshape(1, 1, -1, 3)],
					dim=2,
				)
			else:
				dst[ch] = pts

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
				"cyl_smooth": float(opt_cfg.eff.get("cyl_smooth", 0.0)),
				"cyl_z_smooth": float(opt_cfg.eff.get("cyl_z_smooth", 0.0)),
				"cyl_step": float(opt_cfg.eff.get("cyl_step", 0.0)),
				"cyl_radial_mean": float(opt_cfg.eff.get("cyl_radial_mean", 0.0)),
				"cyl_bend": float(opt_cfg.eff.get("cyl_bend", 0.0)),
				"cyl_conn_mesh": float(opt_cfg.eff.get("cyl_conn_mesh", 0.0)),
				"cyl_conn_gt": float(opt_cfg.eff.get("cyl_conn_gt", 0.0)),
				"cyl_base_mesh": float(opt_cfg.eff.get("cyl_base_mesh", 0.0)),
				"cyl_base_gt": float(opt_cfg.eff.get("cyl_base_gt", 0.0)),
			}
			if is_cyl_stage else opt_cfg.eff
		)
		stage_uses_cyl_loss = (
			_need_term("cyl_normal", stage_eff) > 0 or
			_need_term("cyl_center", stage_eff) > 0 or
			_need_term("cyl_smooth", stage_eff) > 0 or
			_need_term("cyl_z_smooth", stage_eff) > 0 or
			_need_term("cyl_step", stage_eff) > 0 or
			_need_term("cyl_radial_mean", stage_eff) > 0 or
			_need_term("cyl_bend", stage_eff) > 0 or
			_need_term("cyl_conn_mesh", stage_eff) > 0 or
			_need_term("cyl_conn_gt", stage_eff) > 0 or
			_need_term("cyl_base_mesh", stage_eff) > 0 or
			_need_term("cyl_base_gt", stage_eff) > 0
		)
		stage_args = opt_cfg.args or {}
		status_interval_raw = stage_args.get("status_interval", stage_args.get("debug_print_interval", 100))
		status_interval = max(0, int(status_interval_raw))
		opt_timing_enabled = _opt_timing_enabled(stage_args)
		opt_timing_interval = _opt_timing_interval(stage_args, fallback=max(1, status_interval or 100))
		opt_timing_sync = _opt_timing_sync_cuda(stage_args)
		if opt_timing_enabled:
			print(
				f"[optimizer] {label}: opt timing enabled interval={opt_timing_interval} "
				f"sync_cuda={int(opt_timing_sync)}",
				flush=True,
			)

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
		_compile_cyl_normal_raw = os.environ.get(
			"LASAGNA_COMPILE_CYL_NORMAL",
			stage_args.get("compile_cyl_normal", False),
		)
		_compile_cyl_normal = _truthy(_compile_cyl_normal_raw)
		_compile_cyl_normal_backend = os.environ.get(
			"LASAGNA_COMPILE_CYL_NORMAL_BACKEND",
			stage_args.get("compile_cyl_normal_backend", None),
		)
		_compile_cyl_normal_mode = os.environ.get(
			"LASAGNA_COMPILE_CYL_NORMAL_MODE",
			stage_args.get("compile_cyl_normal_mode", None),
		)
		_compile_cyl_normal_dynamic = _truthy(os.environ.get(
			"LASAGNA_COMPILE_CYL_NORMAL_DYNAMIC",
			stage_args.get("compile_cyl_normal_dynamic", False),
		))
		_compile_cyl_normal_fullgraph = _truthy(os.environ.get(
			"LASAGNA_COMPILE_CYL_NORMAL_FULLGRAPH",
			stage_args.get("compile_cyl_normal_fullgraph", False),
		))
		opt_loss_cyl.configure_compile(
			shell_normal=_compile_cyl_normal,
			backend=_compile_cyl_normal_backend,
			mode=_compile_cyl_normal_mode,
			dynamic=_compile_cyl_normal_dynamic,
			fullgraph=_compile_cyl_normal_fullgraph,
		)
		if _compile_cyl_normal:
			_details = []
			if _compile_cyl_normal_backend:
				_details.append(f"backend={_compile_cyl_normal_backend}")
			if _compile_cyl_normal_mode:
				_details.append(f"mode={_compile_cyl_normal_mode}")
			if _compile_cyl_normal_dynamic:
				_details.append("dynamic=1")
			if _compile_cyl_normal_fullgraph:
				_details.append("fullgraph=1")
			_detail_str = " " + " ".join(_details) if _details else ""
			print(f"[optimizer] {label}: compile_cyl_normal=1{_detail_str}", flush=True)
		_stage_done(f"{label}.configure_losses", _t)

		# Once cylinder initialization is done, convert only the best candidate
		# to the regular mesh before any mesh-space optimization.
		_t = _stage_start(f"{label}.prepare_model_params")
		if not is_cyl_stage and getattr(model, "cylinder_enabled", False):
			model.bake_cylinder_into_mesh(data)
		_stage_done(f"{label}.prepare_model_params", _t)

		stage_needs = _needs_for_eff(
			stage_eff,
			pred_dt_flow_gate_cfg_=pred_dt_flow_gate_cfg,
			pred_dt_normal_source_=pred_dt_normal_source,
		)
		print(
			f"[optimizer] {label}: forward_needs={stage_needs.summary()} "
			f"{_prefetch_grad_summary(stage_needs)}",
			flush=True,
		)

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
		_status_step_width = max(16, len(f"{label} {max(0, opt_cfg.steps)}/{max(0, opt_cfg.steps)}") + 2)

		def _print_status(*, step_label: str, loss_val: float, tv: dict[str, float], pv: dict[str, float],
						  its: float | None = None, force_header: bool = False) -> None:
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
			if force_header or _status_rows % 20 == 0:
				hdr = f"{'step':>{_status_step_width}s} {'loss':>8s} {'it/s':>5s}"
				for c in cols:
					hdr += f" {_display_key(c):>{widths[c]}s}"
				print(hdr)
			_status_rows += 1
			its_str = f"{its:5.1f}" if its is not None else f"{'':>5s}"
			row = f"{step_label:>{_status_step_width}s} {loss_val:8.4f} {its_str}"
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

		# Ensure streaming data has all optional channels needed by this stage.
		_needed_channels: set[str] = set(stage_needs.prefetch_channels()) & {"cos", "pred_dt"}
		if ensure_data_fn is not None:
			_t = _stage_start(f"{label}.ensure_data")
			data = ensure_data_fn(data, _needed_channels)
			_stage_done(f"{label}.ensure_data", _t)

		def _prefetch_model_points(needs_: fit_model.ModelForwardNeeds, *, sync: bool = True) -> None:
			if not _active_caches:
				return
			with torch.no_grad():
				_xyz_lr_pf = model._grid_xyz()
				_need_hr_pf = bool(
					needs_.xyz_hr or needs_.hr_data_channels or needs_.hr_prefetch_channels
					or needs_.target or needs_.prefetch_pred_dt_flow
				)
				_xyz_hr_pf = model._grid_xyz_hr(_xyz_lr_pf) if _need_hr_pf else None
				_pred_dt_extra_pf = (
					opt_loss_pred_dt.flow_gate_prefetch_points(
						data=data,
						xyz_hr=_xyz_hr_pf,
						xyz_lr=_xyz_lr_pf,
						cfg=pred_dt_flow_gate_cfg,
					)
					if needs_.prefetch_pred_dt_flow and _xyz_hr_pf is not None else None
				)
				_cyl_pf = None
				if (
					needs_.prefetch_cyl_gt_normals
					and getattr(model, "cylinder_enabled", False)
					and not bool(getattr(model, "cyl_shell_mode", False))
				):
					_cyl_pf, _ = model.cylinder_samples()
					_cyl_pf = _cyl_pf.detach()
				_corr_xyz = None
				if (
					needs_.prefetch_corr_points
					and data.corr_points is not None
					and data.corr_points.points_xyz_winda.shape[0] > 0
				):
					_corr_xyz = data.corr_points.points_xyz_winda[:, :3].to(
						device=next(model.parameters()).device,
						dtype=torch.float32,
					)
			_hr_channels = set(needs_.hr_data_channels) | set(needs_.hr_prefetch_channels)
			_lr_channels = set(needs_.lr_data_channels) | set(needs_.lr_prefetch_channels)
			for _cache in _active_caches:
				_cache_channels = set(_cache.channels)
				_sp = data._spacing_for(_cache.channels[0])
				if _xyz_hr_pf is not None and (_cache_channels & _hr_channels):
					_cache.prefetch(_xyz_hr_pf, data.origin_fullres, _sp)
				if _cache_channels & _lr_channels:
					_cache.prefetch(_xyz_lr_pf, data.origin_fullres, _sp)
				if _pred_dt_extra_pf is not None and "pred_dt" in _cache_channels:
					_cache.prefetch(_pred_dt_extra_pf, data.origin_fullres, _sp)
				if _cyl_pf is not None and ({"grad_mag", "nx", "ny"} & _cache_channels):
					_cache.prefetch(_cyl_pf, data.origin_fullres, _sp)
				if _corr_xyz is not None and ({"grad_mag", "nx", "ny"} & _cache_channels):
					_cache.prefetch(_corr_xyz, data.origin_fullres, _sp)
			if sync:
				for _cache in _active_caches:
					_cache.sync()

		def _prefetch_loss_points_for_result(res_, needs_: fit_model.ModelForwardNeeds) -> None:
			if not _active_caches:
				return
			with torch.no_grad():
				_loss_prefetch_items: dict[str, torch.Tensor] = {}
				if needs_.prefetch_pred_dt_loss:
					_add_prefetch_items(
						_loss_prefetch_items,
						opt_loss_pred_dt.pred_dt_prefetch_items_for_result(res=res_),
					)
				if needs_.prefetch_pred_dt_flow:
					_add_prefetch_items(
						_loss_prefetch_items,
						opt_loss_pred_dt.flow_gate_prefetch_items_for_result(
							res=res_,
							cfg=pred_dt_flow_gate_cfg,
						),
					)
				if needs_.prefetch_cyl_gt_normals:
					_add_prefetch_items(
						_loss_prefetch_items,
						opt_loss_cyl.cyl_normal_prefetch_items_for_result(res=res_),
					)
				if needs_.prefetch_ext_offset:
					_add_prefetch_items(
						_loss_prefetch_items,
						opt_loss_winding_density.ext_offset_prefetch_items_for_result(res=res_),
					)
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
		def _eval_terms(res_, eff_, *, profile_label: str | None = None, timing: _OptTimingWindow | None = None):
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
				missing = _missing_loss_fields(
					name=name,
					required=t.get("needs", Needs()),
					res_=res_,
				)
				if missing:
					raise RuntimeError(
						f"{profile_label or label}: active loss '{name}' missing forward artifact(s): "
						f"{', '.join(missing)}"
					)
				_t_loss = _stage_start(f"{profile_label}.{name}") if profile_label is not None else None
				_t_loss_wall = time.perf_counter() if timing is not None else None
				result = t["loss"](res=res_)
				_debug_cuda_sync(f"{profile_label}.{name}" if profile_label is not None else name)
				if timing is not None and _t_loss_wall is not None:
					timing.sync()
					timing.add(f"loss:{name}", time.perf_counter() - _t_loss_wall)
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
			if stage_uses_cyl_loss and not bool(getattr(res_, "cyl_shell_mode", False)):
				best_idx, display_loss, display_tv = opt_loss_cyl.display_stats(eff_)
				if best_idx is not None and hasattr(model, "set_best_cylinder_index"):
					model.set_best_cylinder_index(best_idx)
				if display_tv:
					tv.update(display_tv)
			return total, tv, display_loss

		# Streaming mode: filter caches to only those requested by active losses.
		_active_caches = []
		if data.sparse_caches:
			_stage_channels = set(stage_needs.prefetch_channels())
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
			start_shell = min(max(0, len(getattr(model, "cyl_shell_completed", []))), shell_count)
			if start_shell >= shell_count:
				_stage_done(f"{label}.total", _t_stage_total)
				return data
			if _next_stage_is_cyl(si):
				shell_indices = range(start_shell, min(start_shell + 1, shell_count))
			else:
				shell_indices = range(start_shell, shell_count)
			max_steps = int(opt_cfg.steps)
			_shell_status_labels: list[str] = []
			for shell_i in shell_indices:
				if shell_i <= 0:
					_shell_status_labels.append(f"{label}.shell{shell_i + 1}")
				else:
					_shell_status_labels.append(f"{label}.shell{shell_i + 1}.grow")
					if bool(getattr(model, "cyl_shell_optimize_resampled", False)):
						_shell_status_labels.append(f"{label}.shell{shell_i + 1}.resampled")
			shell_start = start_shell + 1
			shell_end = start_shell + len(shell_indices)
			pass_mode = (
				"zero-step generate/evaluate only"
				if max_steps <= 0
				else f"optimize {max_steps} steps per pass"
			)
			print(
				f"[optimizer] {label}: shell schedule shells={shell_start}-{shell_end}/{shell_count} "
				f"passes={len(_shell_status_labels)} mode={pass_mode}",
				flush=True,
			)
			for _shell_label in _shell_status_labels:
				print(f"[optimizer] {label}: queued {_shell_label}", flush=True)
			_status_step_width = max(
				_status_step_width,
				max(
					len(f"{shell_label} {max_steps}/{max_steps}") + 2
					for shell_label in _shell_status_labels
				),
			)

			def _prefetch_shell_model_points(needs_: fit_model.ModelForwardNeeds) -> None:
				_prefetch_model_points(needs_)

			def _shell_dbg_values() -> dict[str, float]:
				if not hasattr(model, "_shell_width_step_stats"):
					return {}
				_avg, mn, mx = model._shell_width_step_stats()
				return {
					"wstep_tgt_vx": float(getattr(model, "cyl_shell_current_width_step", 0.0)),
					"wstep_min_vx": mn,
					"wstep_max_vx": mx,
				}

			for shell_i in shell_indices:
				if hasattr(model, "begin_cylinder_shell"):
					model.begin_cylinder_shell(shell_i, data)
				shell_eff = dict(stage_eff)
				for _conn_term in ("cyl_conn_mesh", "cyl_conn_gt", "cyl_base_mesh", "cyl_base_gt"):
					shell_eff[_conn_term] = 0.0
				_base_wstep = float(getattr(model, "cyl_shell_width_target_step", 0.0))
				_grow_wstep = _base_wstep * max(1.0, float(getattr(model, "cyl_shell_growth_factor", 1.5)))
				if shell_i <= 0:
					shell_passes = [(f"{label}.shell{shell_i + 1}", shell_eff, False, _base_wstep, _base_wstep)]
				else:
					shell_passes = [
						(f"{label}.shell{shell_i + 1}.grow", shell_eff, False, _base_wstep, _grow_wstep),
					]
					if bool(getattr(model, "cyl_shell_optimize_resampled", False)):
						shell_passes.append(
							(f"{label}.shell{shell_i + 1}.resampled", shell_eff, True, _base_wstep, _base_wstep)
						)

				for shell_label, pass_eff, needs_resample, wstep_start, wstep_end in shell_passes:
					pass_needs = _needs_for_eff(
						pass_eff,
						pred_dt_flow_gate_cfg_=pred_dt_flow_gate_cfg,
						pred_dt_normal_source_=pred_dt_normal_source,
					)
					print(
						f"[optimizer] {shell_label}: forward_needs={pass_needs.summary()} "
						f"{_prefetch_grad_summary(pass_needs)}",
						flush=True,
					)
					if needs_resample:
						if not hasattr(model, "resample_current_cylinder_shell_width_for_growth"):
							raise RuntimeError("shell grow pass requested, but model cannot resample cylinder shell width")
						model.resample_current_cylinder_shell_width_for_growth(data)
					model.cyl_shell_current_width_step = float(wstep_start)
					action = "evaluate only" if max_steps <= 0 else f"optimize {max_steps} steps"
					if abs(float(wstep_end) - float(wstep_start)) > 1.0e-6 and max_steps > 0:
						action = f"{action}, ramp wstep {float(wstep_start):.1f}->{float(wstep_end):.1f}"
					else:
						action = f"{action}, wstep {float(wstep_start):.1f}"
					print(f"[optimizer] {shell_label}: {action}", flush=True)
					all_params, param_groups = _make_param_groups()
					if not param_groups:
						return data
					opt = torch.optim.Adam(param_groups)

					_t = _stage_start(f"{shell_label}.initial_eval")
					_prefetch_shell_model_points(pass_needs)
					with torch.no_grad():
						res0 = model(data, needs=pass_needs)
						_prefetch_loss_points_for_result(res0, pass_needs)
						loss0, term_vals0, display_loss0 = _eval_terms(
							res0, pass_eff, profile_label=f"{shell_label}.initial_eval.loss")
					term_vals0 = {k: round(v, 4) for k, v in term_vals0.items()}
					_print_status(
						step_label=f"{shell_label} 0/{max_steps}",
						loss_val=float(display_loss0) if display_loss0 is not None else loss0.item(),
						tv=term_vals0,
						pv=_shell_dbg_values(),
						force_header=True,
					)
					snapshot_fn(stage=shell_label.replace(".", "_"), step=0,
								loss=float(loss0.detach().cpu()), data=data, res=res0)
					_stage_done(f"{shell_label}.initial_eval", _t)

					loss = loss0
					display_loss = display_loss0
					res = res0
					_t_wall_start = time.perf_counter()
					_t_steps_acc = 0
					step1 = 0
					_opt_timing = (
						_OptTimingWindow(interval=opt_timing_interval, sync_cuda=opt_timing_sync)
						if opt_timing_enabled else None
					)
					for step in range(max_steps):
						_t_iter = time.perf_counter()
						if max_steps > 0:
							_alpha = float(step + 1) / float(max_steps)
							model.cyl_shell_current_width_step = float(wstep_start) + _alpha * (float(wstep_end) - float(wstep_start))

						_t_part = time.perf_counter()
						if _active_caches:
							for _cache in _active_caches:
								_cache.sync()
						if _opt_timing is not None:
							_opt_timing.add("cache_sync", time.perf_counter() - _t_part)

						_t_part = time.perf_counter()
						if fit_data.CHUNK_STATS_ENABLED:
							fit_data._chunk_stats.begin_iteration()
						if _opt_timing is not None:
							_opt_timing.add("chunk_stats", time.perf_counter() - _t_part)

						_t_part = time.perf_counter()
						_prefetch_shell_model_points(pass_needs)
						if _opt_timing is not None:
							_opt_timing.sync()
							_opt_timing.add("model_point_prefetch", time.perf_counter() - _t_part)

						_t_part = time.perf_counter()
						res = model(data, needs=pass_needs)
						if _opt_timing is not None:
							_opt_timing.sync()
							_opt_timing.add("model_forward", time.perf_counter() - _t_part)

						_t_part = time.perf_counter()
						_prefetch_loss_points_for_result(res, pass_needs)
						if _opt_timing is not None:
							_opt_timing.sync()
							_opt_timing.add("loss_prefetch", time.perf_counter() - _t_part)

						_t_part = time.perf_counter()
						loss, term_vals, display_loss = _eval_terms(res, pass_eff, timing=_opt_timing)
						if _opt_timing is not None:
							_opt_timing.sync()
							_opt_timing.add("loss_eval", time.perf_counter() - _t_part)

						_t_part = time.perf_counter()
						if fit_data.CHUNK_STATS_ENABLED:
							fit_data._chunk_stats.end_iteration()
						if _opt_timing is not None:
							_opt_timing.add("chunk_stats", time.perf_counter() - _t_part)

						_t_part = time.perf_counter()
						opt.zero_grad(set_to_none=True)
						if _opt_timing is not None:
							_opt_timing.add("zero_grad", time.perf_counter() - _t_part)

						_t_part = time.perf_counter()
						loss.backward()
						if _opt_timing is not None:
							_opt_timing.sync()
							_opt_timing.add("backward", time.perf_counter() - _t_part)

						_t_part = time.perf_counter()
						opt.step()
						if _opt_timing is not None:
							_opt_timing.sync()
							_opt_timing.add("optimizer_step", time.perf_counter() - _t_part)

						_t_part = time.perf_counter()
						if _active_caches:
							_prefetch_shell_model_points(pass_needs)
							for _cache in _active_caches:
								_cache.end_iteration()
						if _opt_timing is not None:
							_opt_timing.add("next_prefetch", time.perf_counter() - _t_part)

						step1 = step + 1
						_t_steps_acc += 1
						_done_steps[0] += 1
						_stage_progress = step1 / max_steps if max_steps > 0 else 1.0
						_overall_progress = (
							min(1.0, _done_steps[0] / max(1, _total_steps))
							if _total_steps > 0 else 1.0
						)

						_t_part = time.perf_counter()
						if progress_fn is not None:
							progress_fn(
								step=_done_steps[0], total=_total_steps,
								loss=float(display_loss) if display_loss is not None else float(loss.detach().cpu()),
								stage_progress=_stage_progress, overall_progress=_overall_progress,
								stage_name=stage.name,
							)
						if _opt_timing is not None:
							_opt_timing.add("progress", time.perf_counter() - _t_part)

						_t_part = time.perf_counter()
						if step == 0 or step1 == max_steps or (status_interval > 0 and (step1 % status_interval) == 0):
							term_vals = {k: round(v, 4) for k, v in term_vals.items()}
							_t_wall_now = time.perf_counter()
							_t_wall_elapsed = _t_wall_now - _t_wall_start
							_its = _t_steps_acc / _t_wall_elapsed if _t_wall_elapsed > 0 else None
							_print_status(
								step_label=f"{shell_label} {step1}/{max_steps}",
								loss_val=float(display_loss) if display_loss is not None else loss.item(),
								tv=term_vals,
								pv=_shell_dbg_values(),
								its=_its,
							)
							_t_steps_acc = 0
							_t_wall_start = _t_wall_now
						if _opt_timing is not None:
							_opt_timing.add("status", time.perf_counter() - _t_part)
							_opt_timing.add("total", time.perf_counter() - _t_iter)
							_opt_timing.finish_iter(label=shell_label, step1=step1, max_steps=max_steps)
					if snap_int > 0 and (step1 % snap_int) == 0:
						snapshot_fn(stage=shell_label.replace(".", "_"), step=step1,
									loss=float(loss.detach().cpu()), data=data, res=res)

					snapshot_fn(stage=shell_label.replace(".", "_"), step=max_steps,
								loss=float(loss.detach().cpu()), data=data, res=res)
					if (
						shell_i > 0
						and not needs_resample
						and not bool(getattr(model, "cyl_shell_optimize_resampled", False))
					):
						if not hasattr(model, "resample_current_cylinder_shell_width_for_growth"):
							raise RuntimeError("shell grow pass requested, but model cannot resample cylinder shell width")
						model.resample_current_cylinder_shell_width_for_growth(data)
						print(f"[optimizer] {shell_label}: resampled without reopt", flush=True)

				if hasattr(model, "complete_current_cylinder_shell"):
					model.complete_current_cylinder_shell(data)
					print(
						f"[optimizer] {label}: completed shells="
						f"{len(getattr(model, 'cyl_shell_completed', []))}/{shell_count}",
						flush=True,
					)

			_stage_done(f"{label}.total", _t_stage_total)
			return data

		# Initial prefetch for streaming mode
		if _active_caches:
			_t = _stage_start(f"{label}.initial_prefetch")
			_prefetch_model_points(stage_needs)
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
			res0 = model(data, needs=stage_needs)
			_debug_cuda_sync(f"{label}.initial_eval.model_forward")
			_stage_done(f"{label}.initial_eval.model_forward", _t_forward)
			_t_loss_prefetch = _stage_start(f"{label}.initial_eval.loss_prefetch")
			_prefetch_loss_points_for_result(res0, stage_needs)
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
				force_header=True,
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
		_opt_timing = (
			_OptTimingWindow(interval=opt_timing_interval, sync_cuda=opt_timing_sync)
			if opt_timing_enabled else None
		)

		for step in range(max_steps):
			_t_iter = time.perf_counter()
			# Sync: wait for chunks loaded by last prefetch
			_t_io = time.perf_counter()
			if _active_caches:
				for _cache in _active_caches:
					_cache.sync()
			if _flow_timing is not None:
				_flow_timing.add("io_prefetch", time.perf_counter() - _t_io)
			if _opt_timing is not None:
				_opt_timing.add("cache_sync", time.perf_counter() - _t_io)

			_t_part = time.perf_counter()
			if fit_data.CHUNK_STATS_ENABLED:
				fit_data._chunk_stats.begin_iteration()
			if _opt_timing is not None:
				_opt_timing.add("chunk_stats", time.perf_counter() - _t_part)

			_t_forward = time.perf_counter()
			res = model(data, needs=stage_needs)
			_debug_cuda_sync(f"{label}.{step + 1}.model_forward")
			if _flow_timing is not None:
				_timing_cuda_sync()
				_flow_timing.add("model_forward", time.perf_counter() - _t_forward)
			if _opt_timing is not None:
				_opt_timing.sync()
				_opt_timing.add("model_forward", time.perf_counter() - _t_forward)

			_t_io = time.perf_counter()
			_prefetch_loss_points_for_result(res, stage_needs)
			_debug_cuda_sync(f"{label}.{step + 1}.loss_prefetch")
			if _flow_timing is not None:
				_flow_timing.add("io_prefetch", time.perf_counter() - _t_io)
			if _opt_timing is not None:
				_opt_timing.sync()
				_opt_timing.add("loss_prefetch", time.perf_counter() - _t_io)

			_t_loss_eval = time.perf_counter()
			loss, term_vals, display_loss = _eval_terms(res, stage_eff, timing=_opt_timing)
			if _flow_timing is not None:
				_timing_cuda_sync()
				_flow_timing.add("loss_eval", time.perf_counter() - _t_loss_eval)
				_flow_parts = opt_loss_pred_dt.flow_gate_last_timing()
				_flow_timing.add("flow_sampling", _flow_parts.get("flow_sampling", 0.0))
				_flow_timing.add("flow_calc", _flow_parts.get("flow_calc", 0.0))
			if _opt_timing is not None:
				_opt_timing.sync()
				_opt_timing.add("loss_eval", time.perf_counter() - _t_loss_eval)

			_t_part = time.perf_counter()
			if fit_data.CHUNK_STATS_ENABLED:
				fit_data._chunk_stats.end_iteration()
			if _opt_timing is not None:
				_opt_timing.add("chunk_stats", time.perf_counter() - _t_part)

			_t_opt = time.perf_counter()
			_t_part = time.perf_counter()
			opt.zero_grad(set_to_none=True)
			if _opt_timing is not None:
				_opt_timing.add("zero_grad", time.perf_counter() - _t_part)
			_t_part = time.perf_counter()
			loss.backward()
			if _opt_timing is not None:
				_opt_timing.sync()
				_opt_timing.add("backward", time.perf_counter() - _t_part)
			_t_part = time.perf_counter()
			opt.step()
			if _opt_timing is not None:
				_opt_timing.sync()
				_opt_timing.add("optimizer_step", time.perf_counter() - _t_part)
			_t_part = time.perf_counter()
			model.update_conn_offsets()
			model.update_ext_conn_offsets()
			if _flow_timing is not None:
				_timing_cuda_sync()
				_flow_timing.add("opt_step", time.perf_counter() - _t_opt)
			if _opt_timing is not None:
				_opt_timing.sync()
				_opt_timing.add("model_updates", time.perf_counter() - _t_part)

			# Prefetch: predict next iteration's chunks from updated mesh
			_t_io = time.perf_counter()
			if _active_caches:
				_prefetch_model_points(stage_needs, sync=False)
				for _cache in _active_caches:
					_cache.end_iteration()
			if _flow_timing is not None:
				_flow_timing.add("io_prefetch", time.perf_counter() - _t_io)
			if _opt_timing is not None:
				_opt_timing.add("next_prefetch", time.perf_counter() - _t_io)

			_t_steps_acc += 1
			_done_steps[0] += 1
			step1 = step + 1
			_stage_progress = step1 / max_steps if max_steps > 0 else 1.0
			_overall_progress = (si + _stage_progress) / _num_stages if _num_stages > 0 else 1.0

			_t_part = time.perf_counter()
			if progress_fn is not None:
				progress_fn(
					step=_done_steps[0], total=_total_steps,
					loss=float(display_loss) if display_loss is not None else float(loss.detach().cpu()),
					stage_progress=_stage_progress, overall_progress=_overall_progress,
					stage_name=stage.name,
				)
			if _opt_timing is not None:
				_opt_timing.add("progress", time.perf_counter() - _t_part)

			_t_part = time.perf_counter()
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
			if _opt_timing is not None:
				_opt_timing.add("status", time.perf_counter() - _t_part)

			if ensure_data_fn is not None and (step1 % 100) == 0:
				_t_io = time.perf_counter()
				data = ensure_data_fn(data, _needed_channels)
				if _flow_timing is not None:
					_flow_timing.add("io_prefetch", time.perf_counter() - _t_io)
				if _opt_timing is not None:
					_opt_timing.add("ensure_data", time.perf_counter() - _t_io)

			if snap_int > 0 and (step1 % snap_int) == 0:
				_t_part = time.perf_counter()
				snapshot_fn(stage=label, step=step1, loss=float(loss.detach().cpu()), data=data, res=res)
				if _opt_timing is not None:
					_opt_timing.add("snapshot", time.perf_counter() - _t_part)

			if _flow_timing is not None:
				_flow_timing.add("total", time.perf_counter() - _t_iter)
				_flow_timing.finish_iter(label=label, step1=step1, max_steps=max_steps)
			if _opt_timing is not None:
				_opt_timing.add("total", time.perf_counter() - _t_iter)
				_opt_timing.finish_iter(label=label, step1=step1, max_steps=max_steps)

		_t = _stage_start(f"{label}.final_snapshot")
		snapshot_fn(stage=label, step=max_steps, loss=float(loss.detach().cpu()), data=data, res=res)
		_stage_done(f"{label}.final_snapshot", _t)
		_stage_done(f"{label}.total", _t_stage_total)
		return data

	snap_int = int(snapshot_interval)
	if snap_int < 0:
		snap_int = 0

	_total_steps = _scheduled_total_steps()
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
