from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

import fit2tifxyz
import fit_data
import model
import optimizer
import tifxyz_io


CYLINDER_STAGE_NAMES = ("cyl_init", "cyl_grow", "cyl_grow_refine")
START_SHELL_WRAP_ATOL = 1.0e-3
START_SHELL_WRAP_RTOL = 1.0e-6


def _positive_int(value: str) -> int:
	out = int(value)
	if out <= 0:
		raise argparse.ArgumentTypeError(f"must be > 0, got {value}")
	return out


def _build_parser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(
		prog="cyl_init_tool.py",
		description="Run only lasagna cylinder shell initialization/growth and export each shell as tifxyz.",
	)
	p.add_argument("config", help="Stage config JSON; only cyl_init/cyl_grow/cyl_grow_refine are used")
	p.add_argument("--input", default=None,
				   help="Lasagna volume manifest (.lasagna.json); overrides config args.input")
	p.add_argument("--out-dir", required=True,
				   help="Output directory for shell_XXXX.tifxyz and interm_shell_XXXX.tifxyz exports")
	p.add_argument("--device", default=None,
				   help="Torch device; overrides config args.device, default cuda")
	p.add_argument("--sparse-prefetch-backend", choices=("tensorstore", "python-zarr"), default=None,
				   help="Streaming prefetch backend; overrides config args.sparse-prefetch-backend")
	p.add_argument("--z-range", type=float, nargs=2, default=None,
				   metavar=("Z0", "Z1"), help="Fullres z extent for the cylinder shell")
	p.add_argument("--start-shell", default=None,
				   help="Existing wrapped .tifxyz shell to continue growing from")
	p.add_argument("--shell-count", type=_positive_int, default=None,
				   help="Override total cylinder shell count, including the initial/start shell")
	return p


def _cfg_arg(cfg: dict, key: str, default: object = None) -> object:
	args = cfg.get("args", {})
	if not isinstance(args, dict):
		return default
	if key in args:
		return args[key]
	alt = key.replace("-", "_")
	if alt in args:
		return args[alt]
	return default


def _filter_cylinder_config(cfg: dict, *, start_shell: bool = False) -> dict:
	"""Return a stages config containing only cylinder shell progression stages."""
	if not isinstance(cfg, dict):
		raise ValueError("config must be a JSON object")
	stages_raw = cfg.get("stages", None)
	if not isinstance(stages_raw, list):
		raise ValueError("config must contain a 'stages' list")
	keep = set(CYLINDER_STAGE_NAMES)
	out: dict = {}
	if "base" in cfg:
		out["base"] = copy.deepcopy(cfg["base"])
	filtered = []
	for stage in stages_raw:
		if not isinstance(stage, dict):
			continue
		name = str(stage.get("name", ""))
		if name not in keep:
			continue
		if start_shell and name == "cyl_init":
			continue
		filtered.append(copy.deepcopy(stage))
	out["stages"] = filtered
	return out


def _apply_shell_count_override(stage_cfg: dict, shell_count: int | None) -> None:
	if shell_count is None:
		return
	for stage in stage_cfg.get("stages", []):
		if not isinstance(stage, dict):
			continue
		args = stage.setdefault("args", {})
		if not isinstance(args, dict):
			args = {}
			stage["args"] = args
		args["cyl_max_shells"] = int(shell_count)


def _has_stage(stage_cfg: dict, name: str) -> bool:
	return any(isinstance(stage, dict) and str(stage.get("name", "")) == name
			   for stage in stage_cfg.get("stages", []))


def _derive_z_range_from_umbilicus(data: fit_data.FitData3D) -> tuple[float, float]:
	points = data.umbilicus_points
	if points is None or points.numel() == 0:
		raise ValueError("cannot derive --z-range: FitData3D has no umbilicus control points")
	z = points.detach().to(device="cpu", dtype=torch.float32)[:, 2]
	z0 = float(z.min())
	z1 = float(z.max())
	span = z1 - z0
	if not math.isfinite(span) or span <= 0.0:
		raise ValueError(f"cannot derive --z-range from degenerate umbilicus z span: {z0:.3f}..{z1:.3f}")
	pad = 0.1 * span
	return z0 - pad, z1 + pad


def _normalize_z_range(z_range: tuple[float, float] | list[float]) -> tuple[float, float]:
	z0 = float(z_range[0])
	z1 = float(z_range[1])
	if not (math.isfinite(z0) and math.isfinite(z1)):
		raise ValueError(f"invalid --z-range: {z0} {z1}")
	if z1 <= z0:
		raise ValueError(f"invalid --z-range: Z1 must be greater than Z0, got {z0:.3f} {z1:.3f}")
	return z0, z1


def _umbilicus_seed_at_z(data: fit_data.FitData3D, *, z: float, device: torch.device) -> tuple[float, float, float]:
	z_t = torch.tensor([float(z)], device=device, dtype=torch.float32)
	xy = data.umbilicus_xy_at_z(z_t)[0].detach().to(device="cpu")
	return float(xy[0]), float(xy[1]), float(z)


def _first_cylinder_stage_model_step(stages: list[optimizer.Stage]) -> float | None:
	for stage in stages:
		if "cyl_params" not in stage.global_opt.params:
			continue
		args = stage.global_opt.args or {}
		value = args.get(optimizer.CYLINDER_STAGE_STEP_ARG)
		if value is None:
			return None
		value_f = float(value)
		return value_f if value_f > 0.0 else None
	return None


def _apply_cylinder_prepare_model_step(mdl: model.Model3D, model_step: float | None) -> None:
	if model_step is None:
		return
	step = float(model_step)
	if hasattr(mdl, "cyl_shell_width_target_step"):
		mdl.cyl_shell_width_target_step = step
	if hasattr(mdl, "cyl_shell_current_width_step"):
		mdl.cyl_shell_current_width_step = step
	if hasattr(mdl, "cyl_shell_z_step"):
		mdl.cyl_shell_z_step = step
	if hasattr(mdl, "cyl_shell_current_height_step"):
		mdl.cyl_shell_current_height_step = step


def _set_model_volume_extent(mdl: model.Model3D, data: fit_data.FitData3D) -> None:
	Z, Y, X = data.size
	sx, sy, sz = data.spacing
	volume_extent = (
		data.origin_fullres[0],
		data.origin_fullres[1],
		data.origin_fullres[2],
		data.origin_fullres[0] + (X - 1) * sx,
		data.origin_fullres[1] + (Y - 1) * sy,
		data.origin_fullres[2] + (Z - 1) * sz,
	)
	mdl.params = dataclasses.replace(mdl.params, volume_extent=volume_extent)


def _streaming_skip_channels(needed_channels: set[str]) -> set[str]:
	optional = {"cos", "pred_dt"}
	return optional - set(needed_channels)


def _streaming_loaded_channels(data: fit_data.FitData3D) -> set[str]:
	if not data.sparse_caches:
		return set()
	return {
		ch
		for cache in data.sparse_caches.values()
		for ch in cache.channels
	}


def _wrap_shell_numpy(shell_xyz: torch.Tensor | np.ndarray) -> np.ndarray:
	if isinstance(shell_xyz, torch.Tensor):
		arr = shell_xyz.detach().to(device="cpu", dtype=torch.float32).numpy()
	else:
		arr = np.asarray(shell_xyz, dtype=np.float32)
	if arr.ndim != 3 or arr.shape[-1] != 3:
		raise ValueError(f"shell_xyz must have shape (H, W, 3), got {arr.shape}")
	if arr.shape[0] < 2 or arr.shape[1] < 3:
		raise ValueError(f"shell_xyz requires H>=2 and W>=3, got {arr.shape[:2]}")
	return np.concatenate([arr, arr[:, :1, :]], axis=1)


def _export_shell_tifxyz(
	*,
	out_dir: Path,
	shell_xyz: torch.Tensor | np.ndarray,
	scale: float,
	stage_label: str | None = None,
	shell_index: int | None = None,
) -> None:
	wrapped = _wrap_shell_numpy(shell_xyz)
	x = wrapped[..., 0]
	y = wrapped[..., 1]
	z = wrapped[..., 2]
	area = fit2tifxyz._get_area(x, y, z, 1.0 / float(scale), None)
	fit2tifxyz._write_tifxyz(out_dir=out_dir, x=x, y=y, z=z, scale=float(scale), area=area)
	if stage_label is not None or shell_index is not None:
		meta_path = out_dir / "meta.json"
		meta = json.loads(meta_path.read_text(encoding="utf-8"))
		if stage_label is not None:
			meta["cylinder_stage"] = str(stage_label)
		if shell_index is not None:
			meta["cylinder_shell_index"] = int(shell_index)
		meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def _strip_wrapped_start_shell(
	xyz: torch.Tensor,
	valid: torch.Tensor,
	*,
	atol: float = START_SHELL_WRAP_ATOL,
	rtol: float = START_SHELL_WRAP_RTOL,
) -> torch.Tensor:
	if xyz.ndim != 3 or int(xyz.shape[-1]) != 3:
		raise ValueError(f"start shell xyz must have shape (H, W, 3), got {tuple(xyz.shape)}")
	if valid.shape != xyz.shape[:2]:
		raise ValueError(f"start shell validity shape mismatch: xyz={tuple(xyz.shape)} valid={tuple(valid.shape)}")
	if not bool(valid.all().detach().cpu()):
		raise ValueError("start shell must not contain invalid tifxyz vertices")
	if int(xyz.shape[0]) < 2 or int(xyz.shape[1]) < 4:
		raise ValueError(f"start shell must be a wrapped cylinder grid with H>=2 and W>=4, got {tuple(xyz.shape[:2])}")
	if not torch.allclose(xyz[:, 0], xyz[:, -1], atol=float(atol), rtol=float(rtol)):
		raise ValueError("start shell is not explicitly wrapped: first and last columns do not match")
	return xyz[:, :-1].contiguous()


def _load_start_shell(path: str | Path, *, device: torch.device) -> tuple[torch.Tensor, dict]:
	xyz, valid, meta = tifxyz_io.load_tifxyz(path, device=device)
	return _strip_wrapped_start_shell(xyz, valid), meta


def _seed_model_from_completed_shell(
	mdl: model.Model3D,
	*,
	shell_xyz: torch.Tensor,
	seed_xyz: tuple[float, float, float],
	z_center: float,
) -> None:
	device = next(mdl.parameters()).device
	shell = shell_xyz.detach().to(device=device, dtype=torch.float32).contiguous()
	z_min = float(shell[..., 2].amin().detach().cpu())
	z_max = float(shell[..., 2].amax().detach().cpu())
	model_h = max(1.0, z_max - z_min)
	with torch.no_grad():
		mdl.cyl_seed_xyz = torch.tensor(seed_xyz, device=device, dtype=torch.float32)
	mdl.params = dataclasses.replace(mdl.params, model_w=None, model_h=model_h)
	mdl.cyl_shell_mode = True
	mdl.cylinder_enabled = True
	mdl.init_mode = "cylinder_seed"
	mdl.cyl_shell_seed_z = float(seed_xyz[2])
	mdl.cyl_shell_z_center_target = float(z_center)
	mdl.z_center = float(z_center)
	mdl.cyl_shell_model_h = model_h
	mdl.cyl_shell_z = None
	mdl.cyl_shell_base = None
	mdl.cyl_shell_dirs = None
	mdl.cyl_shell_completed = [shell]
	mdl.cyl_shell_current_index = 0
	mdl.cyl_shell_active = False
	mdl.cyl_shell_search_direction = 1
	mdl.cyl_shell_search_initial_class = None
	mdl.cyl_shell_search_last_class = None
	mdl.cyl_shell_search_initial_signed_distance = None
	mdl.cyl_shell_search_last_signed_distance = None
	mdl.cyl_shell_search_crossed = False
	mdl.cyl_shell_search_done = False


class _ShellExporter:
	def __init__(self, *, out_dir: Path, scale: float) -> None:
		self.out_dir = out_dir
		self.scale = float(scale)
		self.count = 0

	@staticmethod
	def _prefix_for_stage(stage_label: str) -> str:
		if ".cyl_grow_shell" in stage_label and ".cyl_grow_refine_shell" not in stage_label:
			return "interm_shell"
		return "shell"

	def __call__(self, *, shell_index: int, shell_xyz: torch.Tensor, stage_label: str, data: fit_data.FitData3D) -> None:
		self.count += 1
		prefix = self._prefix_for_stage(str(stage_label))
		ordinal = int(shell_index) + 1 if shell_index is not None else self.count
		shell_dir = self.out_dir / f"{prefix}_{ordinal:04d}.tifxyz"
		_export_shell_tifxyz(
			out_dir=shell_dir,
			shell_xyz=shell_xyz,
			scale=self.scale,
			stage_label=stage_label,
			shell_index=shell_index,
		)
		print(f"[cyl_init_tool] exported {shell_dir}", flush=True)


def _construct_model(
	*,
	device: torch.device,
	scaledown: float,
	z_center: float,
	model_step: float,
) -> model.Model3D:
	step = max(1, int(round(float(model_step))))
	return model.Model3D(
		device=device,
		depth=1,
		mesh_h=2,
		mesh_w=3,
		mesh_step=step,
		winding_step=step,
		subsample_mesh=1,
		subsample_winding=1,
		scaledown=scaledown,
		z_step_eff=int(round(scaledown)),
		z_center=z_center,
		init_mode="cylinder_seed",
		volume_extent=None,
		pyramid_d=False,
	)


def main(argv: list[str] | None = None) -> int:
	if argv is None:
		argv = sys.argv[1:]
	parser = _build_parser()
	args = parser.parse_args(argv)
	cfg_path = Path(args.config)
	cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
	if not isinstance(cfg, dict):
		raise ValueError(f"stage config must contain a JSON object: {cfg_path}")

	input_path = args.input if args.input not in (None, "") else _cfg_arg(cfg, "input")
	if input_path in (None, ""):
		raise ValueError("missing --input (lasagna volume manifest .lasagna.json)")
	device_name = args.device if args.device not in (None, "") else _cfg_arg(cfg, "device", "cuda")
	prefetch_backend = (
		args.sparse_prefetch_backend
		if args.sparse_prefetch_backend not in (None, "")
		else _cfg_arg(cfg, "sparse-prefetch-backend", "tensorstore")
	)
	prefetch_backend = str(prefetch_backend)
	if prefetch_backend not in {"tensorstore", "python-zarr"}:
		raise ValueError(
			f"invalid sparse prefetch backend {prefetch_backend!r}; "
			"expected 'tensorstore' or 'python-zarr'"
		)

	device = torch.device(str(device_name))
	out_dir = Path(args.out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)

	stage_cfg = _filter_cylinder_config(cfg, start_shell=args.start_shell is not None)
	_apply_shell_count_override(stage_cfg, args.shell_count)
	if args.start_shell is not None and not _has_stage(stage_cfg, "cyl_grow"):
		stages: list[optimizer.Stage] = []
	else:
		if not stage_cfg.get("stages"):
			raise ValueError("config contains no cylinder progression stages")
		stages = optimizer.load_stages_cfg(
			stage_cfg,
			init_mode="" if args.start_shell is not None else "cylinder_seed",
		)

	stage_model_step = _first_cylinder_stage_model_step(stages) if stages else None
	model_step = stage_model_step if stage_model_step is not None else 100.0

	prep_params = fit_data.get_preprocessed_params(str(input_path))
	source_to_base = float(prep_params.get("source_to_base", 1.0))
	scaledown = float(prep_params["scaledown"]) * source_to_base
	volume_extent_fullres = prep_params.get("volume_extent_fullres")
	print(f"[cyl_init_tool] scaledown={scaledown} volume_extent={volume_extent_fullres}", flush=True)

	def _load_streaming(needed_channels: set[str]) -> fit_data.FitData3D:
		d = fit_data.load_3d_streaming(
			path=str(input_path),
			device=device,
			sparse_prefetch_backend=prefetch_backend,
			skip_channels=_streaming_skip_channels(needed_channels),
		)
		_set_model_volume_extent(mdl, d)
		return d

	def _ensure_data(data: fit_data.FitData3D | None, needed_channels: set[str]) -> fit_data.FitData3D:
		if data is None:
			return _load_streaming(needed_channels)
		loaded = _streaming_loaded_channels(data)
		required = {"grad_mag", "nx", "ny"} | set(needed_channels)
		if not required.issubset(loaded) or (loaded & {"cos", "pred_dt"}) != (required & {"cos", "pred_dt"}):
			return _load_streaming(needed_channels)
		return data

	initial_data = fit_data.load_3d_streaming(
		path=str(input_path),
		device=device,
		sparse_prefetch_backend=prefetch_backend,
		skip_channels=_streaming_skip_channels(set()),
	)

	if args.z_range is None:
		if args.start_shell is None:
			z0, z1 = _derive_z_range_from_umbilicus(initial_data)
		else:
			z0 = z1 = 0.0
	else:
		z0, z1 = _normalize_z_range(args.z_range)

	if args.start_shell is None:
		z_center = 0.5 * (z0 + z1)
		seed_xyz = _umbilicus_seed_at_z(initial_data, z=z_center, device=device)
		mdl = _construct_model(device=device, scaledown=scaledown, z_center=z_center, model_step=model_step)
		_set_model_volume_extent(mdl, initial_data)
		mdl.init_cylinder_seed(
			seed=seed_xyz,
			model_w=0.0,
			model_h=float(z1 - z0),
			volume_extent_fullres=volume_extent_fullres,
			exact_z_range=(z0, z1),
		)
		_apply_cylinder_prepare_model_step(mdl, stage_model_step)
		mdl.prepare_umbilicus_tube_init(initial_data)
		print(f"[cyl_init_tool] fresh cylinder z_range=[{z0:.3f}, {z1:.3f}] seed={seed_xyz}", flush=True)
	else:
		shell_xyz, shell_meta = _load_start_shell(args.start_shell, device=device)
		if args.z_range is None:
			z0 = float(shell_xyz[..., 2].amin().detach().cpu())
			z1 = float(shell_xyz[..., 2].amax().detach().cpu())
		z_center = 0.5 * (z0 + z1)
		seed_xyz = _umbilicus_seed_at_z(initial_data, z=z_center, device=device)
		mdl = _construct_model(device=device, scaledown=scaledown, z_center=z_center, model_step=model_step)
		_set_model_volume_extent(mdl, initial_data)
		_seed_model_from_completed_shell(mdl, shell_xyz=shell_xyz, seed_xyz=seed_xyz, z_center=z_center)
		scale = shell_meta.get("scale")
		if scale is not None and isinstance(scale, list) and scale and float(scale[0]) > 0.0:
			shell_step = max(1, int(round(1.0 / float(scale[0]))))
			mdl.params = dataclasses.replace(mdl.params, mesh_step=shell_step)
			_apply_cylinder_prepare_model_step(mdl, float(shell_step))
		_apply_cylinder_prepare_model_step(mdl, stage_model_step)
		print(f"[cyl_init_tool] continuing from {args.start_shell} seed={seed_xyz}", flush=True)

	export_step = max(1.0, float(getattr(mdl, "cyl_shell_width_target_step", mdl.params.mesh_step)))
	export_scale = 1.0 / export_step
	exporter = _ShellExporter(out_dir=out_dir, scale=export_scale)

	if not stages:
		if not getattr(mdl, "cyl_shell_completed", None):
			raise RuntimeError("no cylinder shell available to export")
		exporter(
			shell_index=0,
			shell_xyz=mdl.cyl_shell_completed[-1],
			stage_label="start_shell",
			data=initial_data,
		)
		return 0

	data = initial_data
	data = optimizer.optimize(
		model=mdl,
		data=data,
		stages=stages,
		snapshot_interval=0,
		snapshot_fn=lambda **_: None,
		progress_fn=None,
		ensure_data_fn=_ensure_data,
		seed_xyz=seed_xyz,
		out_dir=str(out_dir),
		cylinder_shell_callback=exporter,
	)
	_ = data
	if exporter.count == 0 and getattr(mdl, "cyl_shell_completed", None):
		exporter(
			shell_index=len(mdl.cyl_shell_completed) - 1,
			shell_xyz=mdl.cyl_shell_completed[-1],
			stage_label="final_shell",
			data=initial_data,
		)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
