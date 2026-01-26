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


def optimize_rotation_only(*, model, data: fit_data.FitData, stages: list[Stage]) -> None:
	def _noop(*, stage: str, step: int) -> None:
		return

	snapshot_interval = 0
	snapshot_fn = _noop
	return _optimize_rotation_only_inner(
		model=model,
		data=data,
		stages=stages,
		snapshot_interval=snapshot_interval,
		snapshot_fn=snapshot_fn,
	)


def _optimize_rotation_only_inner(
	*,
	model,
	data: fit_data.FitData,
	stages: list[Stage],
	snapshot_interval: int,
	snapshot_fn,
) -> None:
	snap_int = int(snapshot_interval)
	for stage in stages:
		if stage.steps <= 0:
			continue
		opt = torch.optim.Adam([model.theta], lr=stage.lr)
		for step in range(stage.steps):
			loss = opt_loss_dir.direction_loss(model=model, data=data)
			opt.zero_grad(set_to_none=True)
			loss.backward()
			opt.step()
			if snap_int > 0 and ((step + 1) % snap_int == 0 or step == stage.steps - 1):
				snapshot_fn(stage=stage.name, step=step + 1)
			if step == 0 or step == stage.steps - 1 or (step + 1) % 100 == 0:
				theta_val = float(model.theta.detach().cpu())
				print(f"{stage.name} step {step + 1}/{stage.steps}: loss={loss.item():.6f} theta={theta_val:.6f}")


def optimize_rotation_only_with_snapshots(
	*,
	model,
	data: fit_data.FitData,
	stages: list[Stage],
	snapshot_interval: int,
	snapshot_fn,
) -> None:
	return _optimize_rotation_only_inner(
		model=model,
		data=data,
		stages=stages,
		snapshot_interval=snapshot_interval,
		snapshot_fn=snapshot_fn,
	)
