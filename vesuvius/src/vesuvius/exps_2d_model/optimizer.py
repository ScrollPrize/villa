from __future__ import annotations

import json
from dataclasses import dataclass

import torch

import fit_data
import opt_loss_dir


@dataclass(frozen=True)
class Stage:
	name: str
	steps: int
	lr: float
	params: list[str]
	min_scaledown: int


def load_stages(path: str) -> list[Stage]:
	with open(path, "r", encoding="utf-8") as f:
		cfg = json.load(f)

	stages_cfg = cfg.get("stages", None)
	if not isinstance(stages_cfg, list) or not stages_cfg:
		raise ValueError("stages_json: expected a non-empty list in key 'stages'")

	out: list[Stage] = []
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
		out.append(Stage(name=name, steps=steps, lr=lr, params=params, min_scaledown=min_scaledown))
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
		with torch.no_grad():
			loss0 = opt_loss_dir.direction_loss(model=model, data=data)
			param_vals0: dict[str, float] = {}
			for k, vs in all_params.items():
				if len(vs) == 1 and vs[0].numel() == 1:
					param_vals0[k] = float(vs[0].detach().cpu())
			print(f"stage{si} step 0/{stage.steps}: loss={loss0.item():.6f} params={param_vals0}")
		snapshot_fn(stage=f"stage{si}", step=0, loss=float(loss0.detach().cpu()))

		for step in range(stage.steps):
			loss = opt_loss_dir.direction_loss(model=model, data=data)
			opt.zero_grad(set_to_none=True)
			loss.backward()
			opt.step()

			step1 = step + 1
			if step == 0 or step1 == stage.steps or (step1 % 100) == 0:
				param_vals: dict[str, float] = {}
				for k, vs in all_params.items():
					if len(vs) == 1 and vs[0].numel() == 1:
						param_vals[k] = float(vs[0].detach().cpu())
				print(f"stage{si} step {step1}/{stage.steps}: loss={loss.item():.6f} params={param_vals}")

			if snap_int > 0 and (step1 % snap_int) == 0:
				snapshot_fn(stage=f"stage{si}", step=step1, loss=float(loss.detach().cpu()))

		snapshot_fn(stage=f"stage{si}", step=stage.steps, loss=float(loss.detach().cpu()))
