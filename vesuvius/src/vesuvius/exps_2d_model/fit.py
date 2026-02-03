import argparse
from pathlib import Path

import cli_data
import cli_model
import cli_opt
import cli_vis
import model
import optimizer
import torch
import vis


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
	parser = _build_parser()
	args = parser.parse_args(argv)

	data_cfg = cli_data.from_args(args)
	model_cfg = cli_model.from_args(args)
	opt_cfg = cli_opt.from_args(args)
	vis_cfg = cli_vis.from_args(args)

	print("data:", data_cfg)
	print("model:", model_cfg)
	print("opt:", opt_cfg)
	print("vis:", vis_cfg)

	data = cli_data.load_fit_data(data_cfg, z_size=model_cfg.z_size, out_dir_base=vis_cfg.out_dir)
	device = data.cos.device
	mdl = model.Model2D.from_fit_data(
		data=data,
		mesh_step_px=model_cfg.mesh_step_px,
		winding_step_px=model_cfg.winding_step_px,
		init_size_frac=model_cfg.init_size_frac,
		init_size_frac_h=model_cfg.init_size_frac_h,
		init_size_frac_v=model_cfg.init_size_frac_v,
		z_size=model_cfg.z_size,
		device=device,
		subsample_mesh=model_cfg.subsample_mesh,
		subsample_winding=model_cfg.subsample_winding,
	)
	if model_cfg.model_input is not None:
		st = torch.load(model_cfg.model_input, map_location=device)
		miss, unexp = mdl.load_state_dict_compat(st, strict=False)
		if unexp:
			print("state_dict: unexpected keys:", sorted(unexp))
		if miss:
			print("state_dict: missing keys:", sorted(miss))
	print("model_init:", mdl.init)
	print("mesh:", mdl.mesh_h, mdl.mesh_w)

	vis.save(model=mdl, data=data, postfix="init", out_dir=vis_cfg.out_dir, scale=vis_cfg.scale)
	mdl.save_tiff(data=data, path=f"{vis_cfg.out_dir}/raw_init.tif")
	stages = optimizer.load_stages(opt_cfg.stages_json)
	def _save_model_snapshot(*, stage: str, step: int) -> None:
		out = Path(vis_cfg.out_dir)
		out.mkdir(parents=True, exist_ok=True)
		out_snap = out / "model_snapshots"
		out_snap.mkdir(parents=True, exist_ok=True)
		p = out_snap / f"model_{stage}_{step:06d}.pt"
		torch.save(mdl.state_dict(), str(p))

	def _save_model_output_final() -> None:
		if model_cfg.model_output is None:
			return
		torch.save(mdl.state_dict(), str(model_cfg.model_output))

	_save_model_snapshot(stage="init", step=0)
	def _snapshot(*, stage: str, step: int, loss: float) -> None:
		vis.save(
			model=mdl,
			data=data,
			postfix=f"{stage}_{step:06d}",
			out_dir=vis_cfg.out_dir,
			scale=vis_cfg.scale,
		)
		mdl.save_tiff(data=data, path=f"{vis_cfg.out_dir}/raw_{stage}_{step:06d}.tif")
		_save_model_snapshot(stage=stage, step=step)

	optimizer.optimize(
		model=mdl,
		data=data,
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
