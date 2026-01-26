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
		out.append(Stage(name=name, steps=steps, lr=lr))
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

	for stage in stages:
		if stage.steps <= 0:
			continue

		opt = torch.optim.Adam([model.theta], lr=stage.lr)
		with torch.no_grad():
			loss0 = opt_loss_dir.direction_loss(model=model, data=data)
			theta_val0 = float(model.theta.detach().cpu())
			print(f"{stage.name} step 0/{stage.steps}: loss={loss0.item():.6f} theta={theta_val0:.6f}")
		snapshot_fn(stage=stage.name, step=0, loss=float(loss0.detach().cpu()))

		for step in range(stage.steps):
			loss = opt_loss_dir.direction_loss(model=model, data=data)
			opt.zero_grad(set_to_none=True)
			loss.backward()
			opt.step()

			step1 = step + 1
			if step == 0 or step1 == stage.steps or (step1 % 100) == 0:
				theta_val = float(model.theta.detach().cpu())
				print(f"{stage.name} step {step1}/{stage.steps}: loss={loss.item():.6f} theta={theta_val:.6f}")

			if snap_int > 0 and (step1 % snap_int) == 0:
				snapshot_fn(stage=stage.name, step=step1, loss=float(loss.detach().cpu()))

		snapshot_fn(stage=stage.name, step=stage.steps, loss=float(loss.detach().cpu()))
