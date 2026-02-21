from __future__ import annotations

import json
from dataclasses import dataclass, replace

import torch

import cli_data
import fit_data
import opt_loss_dir
import opt_loss_data
import opt_loss_geom
import opt_loss_gradmag
import opt_loss_mod
import opt_loss_step
import opt_loss_corr
import mask_schedule
import point_constraints


def _require_consumed_dict(*, where: str, cfg: dict) -> None:
	if cfg:
		bad = sorted(cfg.keys())
		# raise ValueError(f"stages_json: {where}: unknown key(s): {bad}")
		print(f"WARNING stages_json: {where}: unknown key(s): {bad}")


@dataclass(frozen=True)
class OptSettings:
	steps: int
	lr: float | list[float]
	params: list[str]
	min_scaledown: int
	opt_window: int
	default_mul: float | None
	w_fac: dict | None
	eff: dict[str, float]


@dataclass(frozen=True)
class Stage:
	name: str
	grow: dict | None
	masks: list[dict] | None
	global_opt: OptSettings
	local_opt: list[OptSettings] | None


@dataclass(frozen=True)
class VisLossCollection:
	loss_maps: dict[str, tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]]

	def add_or_update(self, *, name: str, maps: tuple[torch.Tensor, ...], masks: tuple[torch.Tensor, ...]) -> "VisLossCollection":
		if len(maps) == 0:
			return self
		loss_maps = dict(self.loss_maps)
		loss_maps[name] = (maps, masks)
		return VisLossCollection(loss_maps=loss_maps)


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
	"""Return effective weight for a term; 0.0 means 'skip this term'."""
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
		try:
			lr: float | list[float] = [float(v) for v in lr_raw]
		except Exception as e:
			raise ValueError(f"stages_json: stage '{stage_name}' opt.lr: invalid list") from e
	else:
		lr = float(lr_raw)
	params = opt_cfg.get("params", [])
	if not isinstance(params, list):
		params = []
	params = [str(p) for p in params]
	bad_params = sorted(set(params) - {"theta", "winding_scale", "mesh_ms", "conn_offset_ms", "amp_ms", "bias_ms", "amp", "bias"})
	if bad_params:
		raise ValueError(f"stages_json: stage '{stage_name}' opt.params: unknown name(s): {bad_params}")
	min_scaledown = max(0, int(opt_cfg.get("min_scaledown", 0)))
	opt_window = max(1, int(opt_cfg.get("opt_window", 1)))
	default_mul = opt_cfg.get("default_mul", None)
	w_fac = opt_cfg.get("w_fac", None)
	opt_cfg.pop("steps", None)
	opt_cfg.pop("lr", None)
	opt_cfg.pop("params", None)
	opt_cfg.pop("min_scaledown", None)
	opt_cfg.pop("opt_window", None)
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
		opt_window=opt_window,
		default_mul=default_mul,
		w_fac=w_fac,
		eff=eff,
	)



def _lr_last(lr: float | list[float]) -> float:
	if isinstance(lr, list):
		return float(lr[-1])
	return float(lr)


def _lr_scalespace(*, lr: float | list[float], scale_i: int) -> float:
	"""Return learning rate for a scalespace param at index `scale_i` (0=highest res).

	If `lr` is a list, the last element applies to highest-res (scale_i=0), then backwards.
	If length doesn't match, missing rates fall back to the first element.
	"""
	if not isinstance(lr, list):
		return float(lr)
	if not lr:
		return 0.0
	idx = -1 - int(scale_i)
	if -len(lr) <= idx < 0:
		return float(lr[idx])
	return float(lr[0])


def load_stages_cfg(cfg: dict) -> list[Stage]:
	cfg = dict(cfg)

	lambda_global: dict[str, float] = {
			"dir_v": 1.0,
			"dir_conn": 1.0,
			"data": 0.0,
			"data_plain": 0.0,
			"data_grad": 0.0,
			"contr": 0.0,
			"mod_smooth_y": 0.0,
			"step": 0.0,
			"gradmag": 0.0,
			"mean_pos": 0.0,
			"smooth_x": 0.0,
			"smooth_y": 0.0,
			"meshoff_sy": 0.0,
			"conn_sy_l": 0.0,
			"conn_sy_r": 0.0,
			"angle": 0.0,
			"y_straight": 0.0,
			"z_straight": 0.0,
			"z_normal": 0.0,
			"corr_winding": 0.0,
	}
	base_cfg = cfg.pop("base", None)
	if isinstance(base_cfg, dict):
		bad_base = sorted(set(str(k) for k in base_cfg.keys()) - set(lambda_global.keys()))
		if bad_base:
			raise ValueError(f"stages_json: base: unknown term(s): {bad_base}")
		for k, v in base_cfg.items():
			lambda_global[str(k)] = float(v)

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
		grow = s.pop("grow", None)
		masks = s.pop("masks", None)
		if masks is not None and not isinstance(masks, list):
			raise ValueError(f"stages_json: stage '{name}' field 'masks' must be a list or null")
		if isinstance(masks, list):
			masks = [m for m in masks if isinstance(m, dict)]
		if grow is not None and not isinstance(grow, dict):
			raise ValueError(f"stages_json: stage '{name}' field 'grow' must be an object or null")
		if isinstance(grow, dict):
			g = dict(grow)
			directions = g.pop("directions", [])
			generations = g.pop("generations", 0)
			grow_steps = g.pop("steps", 0)
			_require_consumed_dict(where=f"stage '{name}' grow", cfg=g)
			grow = {"directions": directions, "generations": generations, "steps": grow_steps}

		global_opt_cfg = s.pop("global_opt", None)
		local_opt_cfg = s.pop("local_opt", None)
		if global_opt_cfg is None and local_opt_cfg is None:
			# Back-compat: treat the stage itself as global_opt.
			global_opt_cfg = dict(s)
			local_opt_cfg = None
			s.clear()
			_require_consumed_dict(where=f"stage '{name}'", cfg=s)
		else:
			_require_consumed_dict(where=f"stage '{name}'", cfg=s)

		if not isinstance(global_opt_cfg, dict):
			raise ValueError(f"stages_json: stage '{name}' field 'global_opt' must be an object")
		if local_opt_cfg is not None and not isinstance(local_opt_cfg, (dict, list)):
			raise ValueError(f"stages_json: stage '{name}' field 'local_opt' must be an object, list, or null")

		global_opt = _parse_opt_settings(stage_name=name, opt_cfg=global_opt_cfg, base=lambda_global, prev_eff=None)
		local_opt = None
		if local_opt_cfg is not None:
			if isinstance(local_opt_cfg, dict):
				local_opt_cfg = [local_opt_cfg]
			if not isinstance(local_opt_cfg, list) or not local_opt_cfg:
				raise ValueError(f"stages_json: stage '{name}' field 'local_opt' must be a non-empty list, object, or null")
			local_opt = []
			for li, opt_cfg in enumerate(local_opt_cfg):
				if not isinstance(opt_cfg, dict):
					raise ValueError(f"stages_json: stage '{name}' field 'local_opt[{li}]' must be an object")
				local_opt.append(
					_parse_opt_settings(stage_name=name, opt_cfg=opt_cfg, base=lambda_global, prev_eff=None)
				)

		out.append(Stage(name=name, grow=grow, masks=masks, global_opt=global_opt, local_opt=local_opt))
	return out


def load_stages(path: str) -> list[Stage]:
	with open(path, "r", encoding="utf-8") as f:
		cfg = json.load(f)
		if not isinstance(cfg, dict):
			raise ValueError("stages_json: expected an object")
		return load_stages_cfg(cfg)


def total_steps_for_stages(stages: list[Stage]) -> int:
	"""Compute the total number of optimization steps across all stages."""
	total = 0
	for stage in stages:
		if stage.grow is None:
			total += max(0, stage.global_opt.steps)
		else:
			generations = max(0, int(stage.grow.get("generations", 0)))
			local_opts = stage.local_opt if stage.local_opt is not None else [stage.global_opt]
			for _gi in range(generations):
				total += max(0, stage.global_opt.steps)
				for lo in local_opts:
					total += max(0, lo.steps)
	return total


def optimize(
	*,
	model,
	data: fit_data.FitData,
	data_cfg: cli_data.DataConfig | None = None,
	data_out_dir_base: str | None = None,
	stages: list[Stage],
	snapshot_interval: int,
	snapshot_fn,
	progress_fn=None,
) -> fit_data.FitData:
	data_z0 = data_cfg.unet_z if data_cfg is not None else None
	def _masked_mean_per_z(*, lm: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
		"""Return per-batch masked mean for arbitrary-shaped loss maps.

		- If lm is scalar, returns (1,).
		- Otherwise treats dim0 as batch and averages over remaining dims.
		- mask is broadcast to lm.
		"""
		if lm.ndim == 0:
			return lm.view(1)
		if int(lm.shape[0]) == 0:
			return lm.new_zeros((0,))
		m = mask.to(dtype=lm.dtype)
		mm = m.expand_as(lm)
		n = int(lm.shape[0])
		wsum = mm.reshape(n, -1).sum(dim=1)
		lsum = (lm * mm).reshape(n, -1).sum(dim=1)
		fallback = lm.reshape(n, -1).mean(dim=1)
		return torch.where(wsum > 0.0, lsum / wsum, fallback)

	def _print_losses_per_z(*, label: str, res, eff: dict[str, float], mean_pos_xy: torch.Tensor | None) -> None:
		pts_c0 = data.constraints.points if data.constraints is not None else None
		def _corr_pts_for_res() -> fit_data.PointConstraintsData | None:
			if pts_c0 is None:
				return None
			pts_all, idx_left, valid_left, _d_l, idx_right, valid_right, _d_r = point_constraints.closest_conn_segment_indices(
				points_xyz_winda=pts_c0.points_xyz_winda,
				xy_conn=res.xy_conn,
			)
			return fit_data.PointConstraintsData(
				points_xyz_winda=pts_all,
				collection_idx=pts_c0.collection_idx,
				idx_left=idx_left,
				valid_left=valid_left,
				idx_right=idx_right,
				valid_right=valid_right,
			)
		term_to_maps = {
			"data": lambda: opt_loss_data.data_loss_map(res=res),
			"data_plain": lambda: opt_loss_data.data_plain_loss_map(res=res),
			"data_grad": lambda: opt_loss_data.data_grad_loss_map(res=res),
			"dir_v": lambda: opt_loss_dir.dir_v_loss_maps(res=res),
			"gradmag": lambda: opt_loss_gradmag.gradmag_period_loss_map(res=res),
			"mod_smooth_y": lambda: opt_loss_mod.mod_smooth_y_loss_map(res=res),
			"smooth_x": lambda: opt_loss_geom.smooth_x_loss_map(res=res),
			"smooth_y": lambda: opt_loss_geom.smooth_y_loss_map(res=res),
			"meshoff_sy": lambda: opt_loss_geom.meshoff_smooth_y_loss_map(res=res),
			"conn_sy_l": lambda: opt_loss_geom.conn_y_smooth_l_loss_map(res=res),
			"conn_sy_r": lambda: opt_loss_geom.conn_y_smooth_r_loss_map(res=res),
			"angle": lambda: opt_loss_geom.angle_symmetry_loss_map(res=res),
			"y_straight": lambda: opt_loss_geom.y_straight_loss_map(res=res),
			"z_straight": lambda: opt_loss_geom.z_straight_loss_map(res=res),
			"z_normal": lambda: opt_loss_dir.z_normal_loss_maps(res=res),
			"corr_winding": lambda: (lambda lv, lms, ms: (lms[0], ms[0]))(*opt_loss_corr.corr_winding_loss(res=res, pts_c=_corr_pts_for_res())),
			"step": lambda: (lambda lm: (lm, torch.ones_like(lm)))(opt_loss_step.step_loss_maps(res=res)),
		}
		if mean_pos_xy is not None:
			term_to_maps["mean_pos"] = lambda: opt_loss_geom.mean_pos_loss_map(res=res, target_xy=mean_pos_xy)

		out: dict[str, list[float]] = {}
		for name, w in eff.items():
			if float(w) == 0.0:
				continue
			if name == "dir_conn":
				lm_l, lm_r, mask_l, mask_r = opt_loss_dir.dir_conn_loss_maps(res=res)
				l = _masked_mean_per_z(lm=lm_l, mask=mask_l)
				r = _masked_mean_per_z(lm=lm_r, mask=mask_r)
				out[name] = [float(x) for x in (0.5 * (l + r)).detach().cpu().tolist()]
				continue
			fn = term_to_maps.get(name)
			if fn is None:
				continue
			lm, mask = fn()
			out[name] = [float(x) for x in _masked_mean_per_z(lm=lm, mask=mask).detach().cpu().tolist()]
		if not out:
			return
		names = sorted(out.keys())
		n_z = max(len(v) for v in out.values())
		# Print as table: rows=z-slices, cols=loss terms
		hdr = f"{'z':>4s}"
		for k in names:
			hdr += f"  {k:>10s}"
		print(f"{label} losses_per_z:")
		print(hdr)
		for zi in range(n_z):
			row = f"{zi:4d}"
			for k in names:
				v = out[k][zi] if zi < len(out[k]) else 0.0
				row += f"  {v:10.6f}"
			print(row)
	def _run_opt(*, si: int, label: str, stage: Stage, opt_cfg: OptSettings, keep_only_grown_z: bool = False) -> None:
		if opt_cfg.steps <= 0:
			return
		# If the stage does not optimize any global transform params, bake the current
		# global transform into the mesh and disable it.
		if not ("theta" in opt_cfg.params or "winding_scale" in opt_cfg.params):
			if hasattr(model, "global_transform_enabled") and bool(model.global_transform_enabled):
				model.bake_global_transform_into_mesh()

		all_params = model.opt_params()
		hooks: list[torch.utils.hooks.RemovableHandle] = []
		cm_lr = model.const_mask_lr
		if cm_lr is not None:
			if cm_lr.ndim != 4 or int(cm_lr.shape[1]) != 1:
				raise ValueError("const_mask_lr must be (N,1,Hm,Wm)")
			cm_lr = cm_lr.detach().to(device=data.cos.device, dtype=torch.float32)
			if int(cm_lr.shape[0]) != int(data.cos.shape[0]):
				raise ValueError("const_mask_lr batch must match data batch")
			keep_lr = (1.0 - cm_lr)

		ins_z = model._last_grow_insert_z if bool(keep_only_grown_z) else None
		z_keep = None
		if ins_z is not None:
			ins_z = int(ins_z)
			n = int(data.cos.shape[0])
			if not (0 <= ins_z < n):
				raise ValueError("grow z index out of range")
			z_keep = torch.zeros(n, 1, 1, 1, device=data.cos.device, dtype=torch.float32)
			z_keep[ins_z, 0, 0, 0] = 1.0
		param_groups: list[dict] = []
		for name in opt_cfg.params:
			if name in {"theta", "winding_scale"}:
				if hasattr(model, "global_transform_enabled") and not bool(model.global_transform_enabled):
					raise ValueError(f"opt params include '{name}' but global transform is disabled")
			group = all_params.get(name, [])
			if name in {"mesh_ms", "conn_offset_ms"}:
				if cm_lr is not None:
					if not group:
						continue
					p0 = group[0]
					if p0.shape[0] != keep_lr.shape[0] or p0.shape[-2:] != keep_lr.shape[-2:]:
						raise ValueError("const_mask_lr must match mesh_ms[0]/conn_offset_ms[0] spatial shape")
					m = keep_lr.to(dtype=p0.dtype)
					if int(p0.shape[1]) != 1:
						m = m.expand(int(m.shape[0]), int(p0.shape[1]), int(p0.shape[2]), int(p0.shape[3]))
					hooks.append(p0.register_hook(lambda g, mm=m: g * mm))
					param_groups.append({"params": [p0], "lr": _lr_scalespace(lr=opt_cfg.lr, scale_i=0)})
				else:
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
		if z_keep is not None:
			for g in param_groups:
				for p in g.get("params", []):
					if p.ndim >= 1 and int(p.shape[0]) == int(z_keep.shape[0]):
						m = z_keep.to(dtype=p.dtype).view(int(p.shape[0]), *([1] * (int(p.ndim) - 1)))
						hooks.append(p.register_hook(lambda g, mm=m: g * mm))
		opt = torch.optim.Adam(param_groups)

		def _loss3(*, loss_fn, res) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
			lv, lms, masks = loss_fn(res=res)
			return lv, tuple(lms), tuple(masks)
		mean_pos_xy = None
		if _need_term("mean_pos", opt_cfg.eff) != 0.0:
			with torch.no_grad():
				res_init = model(data)
				mean_pos_xy = res_init.xy_lr.mean(dim=(0, 1, 2))
		pts_c0 = data.constraints.points if data.constraints is not None else None
		def _corr_pts_for_res(res) -> fit_data.PointConstraintsData | None:
			if pts_c0 is None:
				return None
			pts_all, idx_left, valid_left, _d_l, idx_right, valid_right, _d_r = point_constraints.closest_conn_segment_indices(
				points_xyz_winda=pts_c0.points_xyz_winda,
				xy_conn=res.xy_conn,
			)
			return fit_data.PointConstraintsData(
				points_xyz_winda=pts_all,
				collection_idx=pts_c0.collection_idx,
				idx_left=idx_left,
				valid_left=valid_left,
				idx_right=idx_right,
				valid_right=valid_right,
			)
		terms = {
			"dir_v": {"loss": opt_loss_dir.dir_v_loss},
			"dir_conn": {"loss": opt_loss_dir.dir_conn_loss},
			"data": {"loss": opt_loss_data.data_loss},
			"data_plain": {"loss": opt_loss_data.data_plain_loss},
			"data_grad": {"loss": opt_loss_data.data_grad_loss},
			"contr": {"loss": opt_loss_mod.contr_loss},
			"mod_smooth_y": {"loss": opt_loss_mod.mod_smooth_y_loss},
			"step": {"loss": opt_loss_step.step_loss},
			"gradmag": {"loss": opt_loss_gradmag.gradmag_period_loss},
			"mean_pos": {"loss": lambda *, res: opt_loss_geom.mean_pos_loss(res=res, target_xy=mean_pos_xy)},
			"smooth_x": {"loss": opt_loss_geom.smooth_x_loss},
			"smooth_y": {"loss": opt_loss_geom.smooth_y_loss},
			"meshoff_sy": {"loss": opt_loss_geom.meshoff_smooth_y_loss},
			"conn_sy_l": {"loss": opt_loss_geom.conn_y_smooth_l_loss},
			"conn_sy_r": {"loss": opt_loss_geom.conn_y_smooth_r_loss},
			"angle": {"loss": opt_loss_geom.angle_symmetry_loss},
			"y_straight": {"loss": opt_loss_geom.y_straight_loss},
			"z_straight": {"loss": opt_loss_geom.z_straight_loss},
			"z_normal": {"loss": opt_loss_dir.z_normal_loss},
			"corr_winding": {"loss": lambda *, res: opt_loss_corr.corr_winding_loss(res=res, pts_c=_corr_pts_for_res(res))},
		}
		_status_rows_since_header = 0

		def _print_status(*, step_label: str, loss_val: float, tv: dict[str, float], pv: dict[str, float]) -> None:
			nonlocal _status_rows_since_header
			tv_keys = sorted(tv.keys())
			pv_keys = sorted(pv.keys())
			cols = tv_keys + [f"p:{k}" for k in pv_keys]
			if _status_rows_since_header % 20 == 0:
				hdr = f"{'step':>20s}  {'loss':>8s}"
				for c in cols:
					hdr += f"  {c:>10s}"
				print(hdr)
			_status_rows_since_header += 1
			row = f"{step_label:>20s}  {loss_val:8.4f}"
			for k in tv_keys:
				row += f"  {tv[k]:10.4f}"
			for k in pv_keys:
				row += f"  {pv[k]:10.4f}"
			print(row)

		with torch.no_grad():
			res0 = model(data)
			if stage.masks is not None:
				model.update_ema(xy_lr=res0.xy_lr, xy_conn=res0.xy_conn)
				stage_img_masks, stage_img_masks_losses = mask_schedule.build_stage_img_masks(
					model=model,
					it=0,
					masks=stage.masks,
				)
				res0 = replace(res0, _stage_img_masks=stage_img_masks, _stage_img_masks_losses=stage_img_masks_losses)
			loss0 = torch.zeros((), device=data.cos.device, dtype=data.cos.dtype)
			term_vals0: dict[str, float] = {}
			vis_losses0 = VisLossCollection(loss_maps={})
			for name, t in terms.items():
				w = _need_term(name, opt_cfg.eff)
				if w == 0.0:
					continue
				lv, lms, masks = _loss3(loss_fn=t["loss"], res=res0)
				vis_losses0 = vis_losses0.add_or_update(name=name, maps=lms, masks=masks)
				term_vals0[name] = float(lv.detach().cpu())
				loss0 = loss0 + w * lv
			param_vals0: dict[str, float] = {}
			for k, vs in all_params.items():
				if len(vs) == 1 and vs[0].numel() == 1:
					param_vals0[k] = float(vs[0].detach().cpu())
			term_vals0 = {k: round(v, 4) for k, v in term_vals0.items()}
			param_vals0 = {k: round(v, 4) for k, v in param_vals0.items()}
			_print_status(step_label=f"{label} 0/{opt_cfg.steps}", loss_val=loss0.item(), tv=term_vals0, pv=param_vals0)
			_print_losses_per_z(label=f"{label} step 0/{opt_cfg.steps}", res=res0, eff=opt_cfg.eff, mean_pos_xy=mean_pos_xy)
		snapshot_fn(stage=label, step=0, loss=float(loss0.detach().cpu()), data=data, res=res0, vis_losses=vis_losses0)

		for step in range(opt_cfg.steps):
			res = model(data)
			model.update_ema(xy_lr=res.xy_lr, xy_conn=res.xy_conn)
			if stage.masks is not None:
				with torch.no_grad():
					stage_img_masks, stage_img_masks_losses = mask_schedule.build_stage_img_masks(
						model=model,
						it=int(step),
						masks=stage.masks,
					)
				res = replace(res, _stage_img_masks=stage_img_masks, _stage_img_masks_losses=stage_img_masks_losses)
			loss = torch.zeros((), device=data.cos.device, dtype=data.cos.dtype)
			term_vals: dict[str, float] = {}
			vis_losses = VisLossCollection(loss_maps={})
			for name, t in terms.items():
				w = _need_term(name, opt_cfg.eff)
				if w == 0.0:
					continue
				lv, lms, masks = _loss3(loss_fn=t["loss"], res=res)
				vis_losses = vis_losses.add_or_update(name=name, maps=lms, masks=masks)
				term_vals[name] = float(lv.detach().cpu())
				loss = loss + w * lv
			opt.zero_grad(set_to_none=True)
			loss.backward()
			opt.step()
			_done_steps[0] += 1
			if progress_fn is not None:
				progress_fn(step=_done_steps[0], total=_total_steps, loss=float(loss.detach().cpu()))

			step1 = step + 1
			if step == 0 or step1 == opt_cfg.steps or (step1 % 100) == 0:
				param_vals: dict[str, float] = {}
				for k, vs in all_params.items():
					if len(vs) == 1 and vs[0].numel() == 1:
						param_vals[k] = float(vs[0].detach().cpu())
				term_vals = {k: round(v, 4) for k, v in term_vals.items()}
				param_vals = {k: round(v, 4) for k, v in param_vals.items()}
				_print_status(step_label=f"{label} {step1}/{opt_cfg.steps}", loss_val=loss.item(), tv=term_vals, pv=param_vals)

			if snap_int > 0 and (step1 % snap_int) == 0:
				snapshot_fn(stage=label, step=step1, loss=float(loss.detach().cpu()), data=data, res=res, vis_losses=vis_losses)

		snapshot_fn(stage=label, step=opt_cfg.steps, loss=float(loss.detach().cpu()), data=data, res=res, vis_losses=vis_losses)
		with torch.no_grad():
			_print_losses_per_z(label=f"{label} step {opt_cfg.steps}/{opt_cfg.steps}", res=res, eff=opt_cfg.eff, mean_pos_xy=mean_pos_xy)
		for h in hooks:
			h.remove()

	snap_int = int(snapshot_interval)
	if snap_int < 0:
		snap_int = 0

	_total_steps = total_steps_for_stages(stages)
	_done_steps = [0]  # mutable counter

	for si, stage in enumerate(stages):
		if stage.grow is None:
			if stage.global_opt.steps > 0:
				_run_opt(si=si, label=f"stage{si}", stage=stage, opt_cfg=stage.global_opt)
			continue
		grow = stage.grow
		directions = grow.get("directions", [])
		if directions is None:
			directions = []
		if not isinstance(directions, list):
			raise ValueError(f"stages_json: stage '{stage.name}' grow.directions must be a list")
		generations = max(0, int(grow.get("generations", 0)))
		grow_steps = max(0, int(grow.get("steps", 0)))
		local_opts = stage.local_opt if stage.local_opt is not None else [stage.global_opt]
		for gi in range(generations):
			model.grow(directions=[str(d) for d in directions], steps=grow_steps)
			if model._last_grow_insert_z is not None:
				if int(data.cos.shape[0]) != int(model.z_size):
					if data_cfg is None:
						raise ValueError("grow fw/bw requires passing data_cfg")
					if data_z0 is None:
						raise ValueError("grow fw/bw requires --unet-z")
					data, data_z0 = fit_data.grow_z_from_omezarr_unet(
						data=data,
						cfg=data_cfg,
						unet_z0=int(data_z0),
						new_z_size=int(model.z_size),
						insert_z=int(model._last_grow_insert_z),
						out_dir_base=data_out_dir_base,
					)
			stage_g = f"stage{si}_grow{gi:04d}"
			snapshot_fn(stage=stage_g, step=0, loss=0.0, data=data)
			if stage.global_opt.steps > 0:
				_run_opt(si=si, label=f"{stage_g}_global", stage=stage, opt_cfg=stage.global_opt, keep_only_grown_z=False)
			if not local_opts or all(int(o.steps) <= 0 for o in local_opts):
				continue
			ins = model._last_grow_insert_lr
			if ins is None:
				if model._last_grow_insert_z is not None:
					model.const_mask_lr = None
					for li, opt_cfg in enumerate(local_opts):
						_run_opt(si=si, label=f"{stage_g}_local{li}", stage=stage, opt_cfg=opt_cfg, keep_only_grown_z=True)
					continue
				raise RuntimeError("grow: missing insertion rect")
			py0, px0, ho, wo = ins
			hm = int(model.mesh_h)
			wm = int(model.mesh_w)
			y0 = int(py0)
			x0 = int(px0)
			y1 = int(py0 + ho)
			x1 = int(px0 + wo)
			win = max(0, int(local_opt.opt_window) - 1)
			n = int(data.cos.shape[0])
			cm = torch.zeros(n, 1, hm, wm, device=data.cos.device, dtype=torch.float32)
			cm[:, :, y0:y1, x0:x1] = 1.0
			cm[:, :, y0:min(y1, y0 + win), x0:x1] = 0.0
			cm[:, :, max(y0, y1 - win):y1, x0:x1] = 0.0
			cm[:, :, y0:y1, x0:min(x1, x0 + win)] = 0.0
			cm[:, :, y0:y1, max(x0, x1 - win):x1] = 0.0
			model.const_mask_lr = cm
			for li, opt_cfg in enumerate(local_opts):
				_run_opt(si=si, label=f"{stage_g}_local{li}", stage=stage, opt_cfg=opt_cfg, keep_only_grown_z=True)
	model.const_mask_lr = None
	return data
