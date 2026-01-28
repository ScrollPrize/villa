from __future__ import annotations

import json
from dataclasses import dataclass

import torch

import fit_data
import opt_loss_dir
import opt_loss_geom
import opt_loss_gradmag
import opt_loss_step


@dataclass(frozen=True)
class OptSettings:
	steps: int
	lr: float
	params: list[str]
	min_scaledown: int
	default_mul: float | None
	w_fac: dict | None
	eff: dict[str, float]


@dataclass(frozen=True)
class Stage:
	name: str
	grow: dict | None
	global_opt: OptSettings
	local_opt: OptSettings | None


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
				if isinstance(v, dict) and "abs" in v:
					eff[str(k)] = float(v["abs"])
				else:
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
	steps = max(0, int(opt_cfg.get("steps", 0)))
	lr = float(opt_cfg.get("lr", 1e-3))
	params = opt_cfg.get("params", [])
	if not isinstance(params, list):
		params = []
	params = [str(p) for p in params]
	min_scaledown = max(0, int(opt_cfg.get("min_scaledown", 0)))
	default_mul = opt_cfg.get("default_mul", None)
	w_fac = opt_cfg.get("w_fac", None)
	if default_mul is not None:
		default_mul = float(default_mul)
	if w_fac is not None and not isinstance(w_fac, dict):
		raise ValueError(f"stages_json: stage '{stage_name}' opt 'w_fac' must be an object or null")
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


def load_stages(path: str) -> list[Stage]:
	with open(path, "r", encoding="utf-8") as f:
		cfg = json.load(f)

		lambda_global: dict[str, float] = {
			"dir_unet": 1.0,
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
		}
	base_cfg = cfg.get("base", None)
	if isinstance(base_cfg, dict):
		for k, v in base_cfg.items():
			lambda_global[str(k)] = float(v)

	stages_cfg = cfg.get("stages", None)
	if not isinstance(stages_cfg, list) or not stages_cfg:
		raise ValueError("stages_json: expected a non-empty list in key 'stages'")

	out: list[Stage] = []
	prev_eff: dict[str, float] | None = None
	for s in stages_cfg:
		if not isinstance(s, dict):
			raise ValueError("stages_json: each stage must be an object")
		name = str(s.get("name", ""))
		grow = s.get("grow", None)
		if grow is not None and not isinstance(grow, dict):
			raise ValueError(f"stages_json: stage '{name}' field 'grow' must be an object or null")

		global_opt_cfg = s.get("global_opt", None)
		local_opt_cfg = s.get("local_opt", None)
		if global_opt_cfg is None and local_opt_cfg is None:
			# Back-compat: treat the stage itself as global_opt.
			global_opt_cfg = dict(s)
			global_opt_cfg.pop("grow", None)
			global_opt_cfg.pop("global_opt", None)
			global_opt_cfg.pop("local_opt", None)
			local_opt_cfg = None

		if not isinstance(global_opt_cfg, dict):
			raise ValueError(f"stages_json: stage '{name}' field 'global_opt' must be an object")
		if local_opt_cfg is not None and not isinstance(local_opt_cfg, dict):
			raise ValueError(f"stages_json: stage '{name}' field 'local_opt' must be an object or null")

		global_opt = _parse_opt_settings(stage_name=name, opt_cfg=global_opt_cfg, base=lambda_global, prev_eff=prev_eff)
		prev_eff = global_opt.eff
		local_opt = None
		if local_opt_cfg is not None:
			local_opt = _parse_opt_settings(stage_name=name, opt_cfg=local_opt_cfg, base=lambda_global, prev_eff=prev_eff)

		out.append(Stage(name=name, grow=grow, global_opt=global_opt, local_opt=local_opt))
	return out


def optimize(
	*,
	model,
	data: fit_data.FitData,
	stages: list[Stage],
	snapshot_interval: int,
	snapshot_fn,
) -> None:
	def _run_opt(*, si: int, label: str, opt_cfg: OptSettings) -> None:
		if opt_cfg.steps <= 0:
			return

		all_params = model.opt_params()
		params: list[torch.nn.Parameter] = []
		for name in opt_cfg.params:
			group = all_params.get(name, [])
			if name in {"mesh_ms", "conn_offset_ms"}:
				k0 = max(0, int(opt_cfg.min_scaledown))
				params.extend(group[k0:])
			else:
				params.extend(group)
		if not params:
			return
		opt = torch.optim.Adam(params, lr=opt_cfg.lr)
		mean_pos_xy = None
		if _need_term("mean_pos", opt_cfg.eff) != 0.0:
			with torch.no_grad():
				res_init = model(data)
				mean_pos_xy = res_init.xy_lr.mean(dim=(0, 1, 2))
		terms = {
			"dir_unet": {"loss": opt_loss_dir.direction_loss},
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
		}
		with torch.no_grad():
			res0 = model(data)
			loss0 = torch.zeros((), device=data.cos.device, dtype=data.cos.dtype)
			term_vals0: dict[str, float] = {}
			for name, t in terms.items():
				w = _need_term(name, opt_cfg.eff)
				if w == 0.0:
					continue
				loss_fn = t["loss"]
				lv = loss_fn(res=res0)
				term_vals0[name] = float(lv.detach().cpu())
				loss0 = loss0 + w * lv
			param_vals0: dict[str, float] = {}
			for k, vs in all_params.items():
				if len(vs) == 1 and vs[0].numel() == 1:
					param_vals0[k] = float(vs[0].detach().cpu())
			term_vals0 = {k: round(v, 4) for k, v in term_vals0.items()}
			param_vals0 = {k: round(v, 4) for k, v in param_vals0.items()}
			print(f"{label} step 0/{opt_cfg.steps}: loss={loss0.item():.4f} terms={term_vals0} params={param_vals0}")
		snapshot_fn(stage=label, step=0, loss=float(loss0.detach().cpu()))

		for step in range(opt_cfg.steps):
			res = model(data)
			loss = torch.zeros((), device=data.cos.device, dtype=data.cos.dtype)
			term_vals: dict[str, float] = {}
			for name, t in terms.items():
				w = _need_term(name, opt_cfg.eff)
				if w == 0.0:
					continue
				loss_fn = t["loss"]
				lv = loss_fn(res=res)
				term_vals[name] = float(lv.detach().cpu())
				loss = loss + w * lv
			opt.zero_grad(set_to_none=True)
			loss.backward()
			opt.step()

			step1 = step + 1
			if step == 0 or step1 == opt_cfg.steps or (step1 % 100) == 0:
				param_vals: dict[str, float] = {}
				for k, vs in all_params.items():
					if len(vs) == 1 and vs[0].numel() == 1:
						param_vals[k] = float(vs[0].detach().cpu())
				term_vals = {k: round(v, 4) for k, v in term_vals.items()}
				param_vals = {k: round(v, 4) for k, v in param_vals.items()}
				print(f"{label} step {step1}/{opt_cfg.steps}: loss={loss.item():.4f} terms={term_vals} params={param_vals}")

			if snap_int > 0 and (step1 % snap_int) == 0:
				snapshot_fn(stage=label, step=step1, loss=float(loss.detach().cpu()))

		snapshot_fn(stage=label, step=opt_cfg.steps, loss=float(loss.detach().cpu()))

	snap_int = int(snapshot_interval)
	if snap_int < 0:
		snap_int = 0

	for si, stage in enumerate(stages):
		if stage.global_opt.steps > 0:
			_run_opt(si=si, label=f"stage{si}", opt_cfg=stage.global_opt)
		if stage.grow is None:
			continue
		grow = stage.grow
		directions = grow.get("directions", [])
		if directions is None:
			directions = []
		if not isinstance(directions, list):
			raise ValueError(f"stages_json: stage '{stage.name}' grow.directions must be a list")
		generations = max(0, int(grow.get("generations", 0)))
		grow_steps = max(0, int(grow.get("steps", 0)))
		local_opt = stage.local_opt if stage.local_opt is not None else stage.global_opt
		for gi in range(generations):
			model.grow(directions=[str(d) for d in directions], steps=grow_steps)
			snapshot_fn(stage=f"stage{si}_grow{gi}", step=0, loss=0.0)
			if local_opt.steps <= 0:
				continue
			ins = getattr(model, "_last_grow_insert_lr", None)
			if ins is None:
				raise RuntimeError("grow: missing insertion rect")
			py0, px0, ho, wo = ins
			cm = torch.zeros(1, 1, int(model.mesh_h), int(model.mesh_w), device=data.cos.device, dtype=torch.float32)
			cm[:, :, py0:py0 + ho, px0:px0 + wo] = 1.0
			model.const_mask_lr = cm
			_run_opt(si=si, label=f"stage{si}_grow{gi}", opt_cfg=local_opt)
		model.const_mask_lr = None
