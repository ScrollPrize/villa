from __future__ import annotations

import json
from dataclasses import dataclass

import torch

import fit_data
import opt_loss_dir
import opt_loss_step


@dataclass(frozen=True)
class Stage:
	name: str
	steps: int
	lr: float
	params: list[str]
	min_scaledown: int
	eff: dict[str, float]


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


def load_stages(path: str) -> list[Stage]:
	with open(path, "r", encoding="utf-8") as f:
		cfg = json.load(f)

	lambda_global: dict[str, float] = {"dir_unet": 1.0, "step": 0.0}
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
		steps = max(0, int(s.get("steps", 0)))
		lr = float(s.get("lr", 1e-3))
		params = s.get("params", [])
		if not isinstance(params, list):
			params = []
		params = [str(p) for p in params]
		min_scaledown = max(0, int(s.get("min_scaledown", 0)))
		default_mul = s.get("default_mul", None)
		w_fac = s.get("w_fac", None)
		if default_mul is not None:
			default_mul = float(default_mul)
		if w_fac is not None and not isinstance(w_fac, dict):
			raise ValueError(f"stages_json: stage '{name}' field 'w_fac' must be an object or null")
		eff, _mods = _stage_to_modifiers(lambda_global, prev_eff, default_mul, w_fac)
		prev_eff = eff
		out.append(Stage(name=name, steps=steps, lr=lr, params=params, min_scaledown=min_scaledown, eff=eff))
	return out


def optimize(
	*,
	model,
	data: fit_data.FitData,
	stages: list[Stage],
	snapshot_interval: int,
	snapshot_fn,
) -> None:
	snap_int = int(snapshot_interval)
	if snap_int < 0:
		snap_int = 0

	for si, stage in enumerate(stages):
		if stage.steps <= 0:
			continue

		all_params = model.opt_params()
		params: list[torch.nn.Parameter] = []
		for name in stage.params:
			group = all_params.get(name, [])
			if name == "offset_ms":
				k0 = max(0, int(stage.min_scaledown))
				params.extend(group[k0:])
			else:
				params.extend(group)
		if not params:
			continue
		opt = torch.optim.Adam(params, lr=stage.lr)
		terms = {
			"dir_unet": {"loss": opt_loss_dir.direction_loss},
			"step": {"loss": opt_loss_step.step_loss},
		}
		with torch.no_grad():
			res0 = model(data)
			loss0 = torch.zeros((), device=data.cos.device, dtype=data.cos.dtype)
			term_vals0: dict[str, float] = {}
			for name, t in terms.items():
				w = _need_term(name, stage.eff)
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
			print(f"stage{si} step 0/{stage.steps}: loss={loss0.item():.6f} terms={term_vals0} params={param_vals0}")
		snapshot_fn(stage=f"stage{si}", step=0, loss=float(loss0.detach().cpu()))

		for step in range(stage.steps):
			res = model(data)
			loss = torch.zeros((), device=data.cos.device, dtype=data.cos.dtype)
			term_vals: dict[str, float] = {}
			for name, t in terms.items():
				w = _need_term(name, stage.eff)
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
			if step == 0 or step1 == stage.steps or (step1 % 100) == 0:
				param_vals: dict[str, float] = {}
				for k, vs in all_params.items():
					if len(vs) == 1 and vs[0].numel() == 1:
						param_vals[k] = float(vs[0].detach().cpu())
				print(f"stage{si} step {step1}/{stage.steps}: loss={loss.item():.6f} terms={term_vals} params={param_vals}")

			if snap_int > 0 and (step1 % snap_int) == 0:
				snapshot_fn(stage=f"stage{si}", step=step1, loss=float(loss.detach().cpu()))

		snapshot_fn(stage=f"stage{si}", step=stage.steps, loss=float(loss.detach().cpu()))
