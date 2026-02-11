import argparse
import sys
from pathlib import Path

import cli_data
import cli_model
import cli_opt
import cli_vis
import model
import optimizer
import torch
import vis
from dataclasses import asdict
from dataclasses import replace
import json

import cli_json


def _build_parser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(
		prog="fit.py",
		description="2D fit entrypoint (CLI composition)",
	)
	cli_data.add_args(p)
	cli_model.add_args(p)
	cli_opt.add_args(p)
	cli_vis.add_args(p)
	return p


def main(argv: list[str] | None = None) -> int:
	if argv is None:
		argv = sys.argv[1:]
	parser = _build_parser()
	cfg_paths, argv_rest = cli_json.split_cfg_argv(argv)
	cfg_paths = [str(x) for x in cfg_paths]
	cfg = cli_json.merge_cfgs(cfg_paths)
	cli_json.apply_defaults_from_cfg_args(parser, cfg)
	args = parser.parse_args(argv_rest)

	data_cfg = cli_data.from_args(args)
	model_cfg = cli_model.from_args(args)
	opt_cfg = cli_opt.from_args(args)
	vis_cfg = cli_vis.from_args(args)

	print("data:", data_cfg)
	print("model:", model_cfg)
	print("opt:", opt_cfg)
	print("vis:", vis_cfg)

	model_params_in: dict | None = None
	z_size_use = int(model_cfg.z_size)
	mesh_step_use = int(model_cfg.mesh_step_px)
	winding_step_use = int(model_cfg.winding_step_px)
	subsample_mesh_use = int(model_cfg.subsample_mesh)
	subsample_winding_use = int(model_cfg.subsample_winding)
	z_step_use = int(data_cfg.z_step)
	if model_cfg.model_input is not None:
		st_in = torch.load(model_cfg.model_input, map_location="cpu")
		if isinstance(st_in, dict) and isinstance(st_in.get("_model_params_", None), dict):
			model_params_in = st_in["_model_params_"]
		if isinstance(st_in, dict) and ("amp" in st_in) and hasattr(st_in["amp"], "shape"):
			try:
				z_size_use = max(1, int(st_in["amp"].shape[0]))
			except Exception:
				pass
		if model_params_in is not None:
			if "mesh_step_px" in model_params_in:
				mesh_step_use = int(model_params_in["mesh_step_px"])
			if "winding_step_px" in model_params_in:
				winding_step_use = int(model_params_in["winding_step_px"])
			if "subsample_mesh" in model_params_in:
				subsample_mesh_use = int(model_params_in["subsample_mesh"])
			if "subsample_winding" in model_params_in:
				subsample_winding_use = int(model_params_in["subsample_winding"])
			if int(z_step_use) == 1 and ("z_step_vx" in model_params_in):
				z_step_use = max(1, int(model_params_in["z_step_vx"]))
			c6 = model_params_in.get("crop_xyzwhd", None)
			if isinstance(c6, (list, tuple)) and len(c6) == 6:
				x, y, w, h, z0, _d = (int(v) for v in c6)
				crop_use = data_cfg.crop if data_cfg.crop is not None else (x, y, w, h)
				unet_z_use = data_cfg.unet_z if data_cfg.unet_z is not None else int(z0)
				data_cfg = replace(data_cfg, crop=crop_use, unet_z=unet_z_use, z_step=int(z_step_use))

	if model_params_in is not None:
		print("model_params_in:\n" + json.dumps(model_params_in, indent=2, sort_keys=True))

	data = cli_data.load_fit_data(data_cfg, z_size=int(z_size_use), out_dir_base=vis_cfg.out_dir)
	device = data.cos.device
	crop_xyzwhd = None
	if data_cfg.crop is not None and data_cfg.unet_z is not None:
		x, y, w, h = (int(v) for v in data_cfg.crop)
		crop_xyzwhd = (x, y, w, h, int(data_cfg.unet_z), int(z_size_use))

	if model_cfg.model_input is not None and model_params_in is not None:
		# Derive model init size from checkpoint tensor shapes (ignore init_size_frac unless creating a new model).
		st_cpu = torch.load(model_cfg.model_input, map_location="cpu")
		mesh0 = st_cpu.get("mesh_ms.0", None) if isinstance(st_cpu, dict) else None
		if mesh0 is None or not hasattr(mesh0, "shape"):
			raise ValueError("model_input missing mesh_ms.0")
		gh = int(mesh0.shape[2])
		gw = int(mesh0.shape[3])
		init = model.ModelInit(
			init_size_frac=1.0,
			init_size_frac_h=None,
			init_size_frac_v=None,
			mesh_step_px=int(mesh_step_use),
			winding_step_px=int(winding_step_use),
			mesh_h=int(gh),
			mesh_w=int(gw),
		)
		mdl = model.Model2D(
			init=init,
			device=device,
			z_size=int(z_size_use),
			subsample_mesh=int(subsample_mesh_use),
			subsample_winding=int(subsample_winding_use),
			z_step_vx=int(z_step_use),
			scaledown=float(data_cfg.downscale),
			crop_xyzwhd=crop_xyzwhd,
		)
	else:
		mdl = model.Model2D.from_fit_data(
			data=data,
			mesh_step_px=int(mesh_step_use),
			winding_step_px=int(winding_step_use),
			init_size_frac=model_cfg.init_size_frac,
			init_size_frac_h=model_cfg.init_size_frac_h,
			init_size_frac_v=model_cfg.init_size_frac_v,
			z_size=int(z_size_use),
			z_step_vx=int(z_step_use),
			scaledown=float(data_cfg.downscale),
			device=device,
			subsample_mesh=int(subsample_mesh_use),
			subsample_winding=int(subsample_winding_use),
			crop_xyzwhd=crop_xyzwhd,
		)
	if model_cfg.model_input is not None:
		st = torch.load(model_cfg.model_input, map_location=device)
		miss, unexp = mdl.load_state_dict_compat(st, strict=False)
		if unexp:
			print("state_dict: unexpected keys:", sorted(unexp))
		if miss:
			print("state_dict: missing keys:", sorted(miss))
		with torch.no_grad():
			xy0 = mdl.mesh_coarse().detach()
			mean_xy = xy0.mean(dim=(0, 2, 3)).to(dtype=torch.float32).cpu().numpy().tolist()
			min_xy = xy0.amin(dim=(0, 2, 3)).to(dtype=torch.float32).cpu().numpy().tolist()
			max_xy = xy0.amax(dim=(0, 2, 3)).to(dtype=torch.float32).cpu().numpy().tolist()
			print(f"loaded mesh_coarse: mean_xy={mean_xy} min_xy={min_xy} max_xy={max_xy}")
	print("model_init:", mdl.init)
	print("mesh:", mdl.mesh_h, mdl.mesh_w)

	vis.save(model=mdl, data=data, postfix="init", out_dir=vis_cfg.out_dir, scale=vis_cfg.scale)
	mdl.save_tiff(data=data, path=f"{vis_cfg.out_dir}/raw_init.tif")
	stages = optimizer.load_stages_cfg(cfg)
	def _save_model_snapshot(*, stage: str, step: int) -> None:
		out = Path(vis_cfg.out_dir)
		out.mkdir(parents=True, exist_ok=True)
		out_snap = out / "model_snapshots"
		out_snap.mkdir(parents=True, exist_ok=True)
		p = out_snap / f"model_{stage}_{step:06d}.pt"
		st = dict(mdl.state_dict())
		st["_model_params_"] = asdict(mdl.params)
		torch.save(st, str(p))

	def _save_model_output_final() -> None:
		if model_cfg.model_output is None:
			return
		st = dict(mdl.state_dict())
		st["_model_params_"] = asdict(mdl.params)
		torch.save(st, str(model_cfg.model_output))

	_save_model_snapshot(stage="init", step=0)
	def _snapshot(*, stage: str, step: int, loss: float, data) -> None:
		vis.save(
			model=mdl,
			data=data,
			postfix=f"{stage}_{step:06d}",
			out_dir=vis_cfg.out_dir,
			scale=vis_cfg.scale,
		)
		mdl.save_tiff(data=data, path=f"{vis_cfg.out_dir}/raw_{stage}_{step:06d}.tif")
		_save_model_snapshot(stage=stage, step=step)

	data = optimizer.optimize(
		model=mdl,
		data=data,
		data_cfg=data_cfg,
		data_out_dir_base=vis_cfg.out_dir,
		stages=stages,
		snapshot_interval=opt_cfg.snapshot_interval,
		snapshot_fn=_snapshot,
	)
	vis.save(model=mdl, data=data, postfix="final", out_dir=vis_cfg.out_dir, scale=vis_cfg.scale)
	mdl.save_tiff(data=data, path=f"{vis_cfg.out_dir}/raw_final.tif")
	_save_model_snapshot(stage="final", step=0)
	_save_model_output_final()
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
