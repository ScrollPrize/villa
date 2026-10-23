from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, replace

import torch

import cli_data
import fit_data
import model as fit_model
import cyl_sdf_volume
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
import opt_loss_snap_surf
import opt_loss_atlas_line
from snap_surf import map_global as snap_surf_map_global
import opt_loss_flatten
from flatten_clamped_adam import FlattenClampedAdam
from . import timing as optimizer_timing
from .reporting import StageReporter
from .config import (
	OptSettings,
	Stage,
	CYLINDER_SEED_INIT_STAGE_ROLES,
	CYLINDER_STAGE_STEP_ARG,
	CYLINDER_OUTPUT_ALL_SHELLS_ARGS,
	CYLINDER_MAX_SEARCH_SHELLS_ARGS,
	CYLINDER_GROW_DIRECTION_ARG,
	CYLINDER_OUTSIDE_GRID_STEP_ARG,
	CYLINDER_OUTSIDE_SAMPLE_FACTOR_ARG,
	CYLINDER_OUTSIDE_THREADS_ARG,
	CYLINDER_OUTSIDE_CHUNK_SIZE_ARG,
	CYLINDER_OUTSIDE_DEEP_INTERP_CHUNKS_ARG,
	CYLINDER_OUTSIDE_DEEP_BLEND_CHUNKS_ARG,
	CYLINDER_REFINE_MAX_IFRAC_ARGS,
	DEFAULT_CYLINDER_REFINE_MAX_IFRAC,
	MODEL_INTERNAL_PARAM,
	normalize_cylinder_grow_direction,
	_parse_opt_settings,
	total_steps_for_stages,
	_lr_last,
	_global_map_stage_from_opt_settings,
	_resolve_snap_surf_map_fixture_export_args,
	_truthy,
)


@dataclass(frozen=True)
class CylinderGrowWidthTarget:
	width_count: int
	circumference: float
	width_step: float


def cylinder_grow_width_target(
	*,
	reference_width_count: int,
	reference_circumference: float,
	shell_index: int,
	grow_factor: float,
	direction: int,
) -> CylinderGrowWidthTarget:
	ref_w = max(3, int(reference_width_count))
	ref_circ = max(1.0e-6, float(reference_circumference))
	idx = max(0, int(shell_index))
	factor = max(1.0, float(grow_factor))
	scale = factor ** idx
	if int(direction) < 0:
		scale = 1.0 / scale
	target_circ = ref_circ * scale
	target_w_f = float(ref_w) * scale
	target_w = max(3, int(math.floor(target_w_f + 0.5)))
	return CylinderGrowWidthTarget(
		width_count=target_w,
		circumference=target_circ,
		width_step=target_circ / float(target_w),
	)


def _cyl_outside_mode_for_direction(direction: int) -> str:
	return (
		cyl_sdf_volume.CYL_OUTSIDE_MODE_OUTSIDE
		if int(direction) < 0
		else cyl_sdf_volume.CYL_OUTSIDE_MODE_INSIDE
	)


def _need_term(name: str, stage_eff: dict[str, float]) -> float:
	return float(stage_eff.get(name, 0.0))


def _lr_scalespace(*, lr: float | list[float], scale_i: int) -> float:
	if not isinstance(lr, list):
		return float(lr)
	if not lr:
		return 0.0
	idx = -1 - int(scale_i)
	if -len(lr) <= idx < 0:
		return float(lr[idx])
	return float(lr[0])


def _steps_label(opt_cfg: OptSettings) -> str:
	if opt_cfg.steps_auto:
		return f"auto:{int(opt_cfg.steps)}"
	return str(int(opt_cfg.steps))


def _auto_steps_window(args: dict | None) -> int:
	args = args or {}
	return max(1, int(args.get("auto_steps_window", 100)))


def _auto_steps_min(args: dict | None, *, window: int) -> int:
	args = args or {}
	return max(1, int(args.get("auto_steps_min", 2 * int(window))))


def _auto_steps_rel_threshold(args: dict | None) -> float:
	args = args or {}
	raw = args.get("auto_steps_rel_threshold", args.get("auto_steps_rel_tol", 1.0e-4))
	return max(0.0, float(raw))


def _auto_steps_relative_improvement(history: list[float], *, window: int) -> float:
	if len(history) <= int(window):
		return math.inf
	before = history[:-int(window)]
	recent = history[-int(window):]
	if not before or not recent:
		return math.inf
	best_before = min(float(v) for v in before)
	best_recent = min(float(v) for v in recent)
	return (best_before - best_recent) / max(abs(best_before), 1.0e-12)


def _auto_steps_should_stop(history: list[float], *, window: int, rel_threshold: float) -> bool:
	return _auto_steps_relative_improvement(history, window=window) < float(rel_threshold)


def _lr_warmup_steps(args: dict | None) -> int:
	args = args or {}
	raw = args.get("lr_warmup_steps", args.get("warmup_steps", 0))
	return max(0, int(raw))


def _lr_warmup_factor(*, step1: int, warmup_steps: int) -> float:
	warmup_steps = max(0, int(warmup_steps))
	if warmup_steps <= 0:
		return 1.0
	return min(1.0, max(0.0, float(step1) / float(warmup_steps)))


def _capture_optimizer_target_lrs(opt: torch.optim.Optimizer) -> None:
	for group in opt.param_groups:
		group.setdefault("_target_lr", float(group.get("lr", 0.0)))


def _apply_optimizer_lr_warmup(opt: torch.optim.Optimizer, *, step1: int, warmup_steps: int) -> None:
	if int(warmup_steps) <= 0:
		return
	scale = _lr_warmup_factor(step1=int(step1), warmup_steps=int(warmup_steps))
	for group in opt.param_groups:
		target_lr = float(group.setdefault("_target_lr", float(group.get("lr", 0.0))))
		group["lr"] = target_lr * scale


def _flatten_max_update_base(args: dict | None) -> float:
	if args is None:
		return 0.1
	return float(args.get("flatten_max_update", 0.1))


def _clamp_flatten_map_ms_update(
	params: list[torch.nn.Parameter],
	before: list[torch.Tensor],
	*,
	base_step: float,
) -> None:
	base = float(base_step)
	if base <= 0.0:
		return
	with torch.no_grad():
		for scale_i, (p, prev) in enumerate(zip(params, before)):
			if p.shape != prev.shape:
				continue
			max_step = base * (2.0 ** int(scale_i))
			delta = p - prev
			if delta.ndim >= 1 and int(delta.shape[0]) == 2:
				norm = torch.linalg.vector_norm(delta, dim=0, keepdim=True)
				scale = (float(max_step) / norm.clamp_min(1.0e-12)).clamp_max(1.0)
				p.copy_(prev + delta * scale)
			else:
				p.copy_(prev + delta.clamp(min=-float(max_step), max=float(max_step)))


def _renorm_forward_flatten_map_step(model) -> dict[str, float]:
	"""Rescale a forward flatten UV map so avg physical voxels per UV pixel hits target."""
	if not bool(getattr(model, "flatten_enabled", False)):
		return {}
	if str(getattr(model, "flatten_direction", "inverse")).strip().lower() != "forward":
		return {}
	if not hasattr(model, "flatten_map") or not hasattr(model, "flatten_map_ms"):
		return {}
	with torch.no_grad():
		uv = model.flatten_map()
		xyz = getattr(model, "flatten_source_xyz", None)
		valid = getattr(model, "flatten_source_valid", None)
		cell_valid = getattr(model, "flatten_source_cell_valid", None)
		target_t = getattr(model, "flatten_target_step", None)
		if xyz is None or valid is None or cell_valid is None or target_t is None:
			return {}
		if uv.ndim != 3 or xyz.ndim != 3 or int(uv.shape[-1]) != 2 or int(xyz.shape[-1]) != 3:
			return {}
		if tuple(uv.shape[:2]) != tuple(xyz.shape[:2]) or tuple(valid.shape) != tuple(uv.shape[:2]):
			return {}
		if tuple(cell_valid.shape) != (max(0, int(uv.shape[0]) - 1), max(0, int(uv.shape[1]) - 1)):
			return {}
		target_step = float(target_t.detach().cpu())
		if not math.isfinite(target_step) or target_step <= 0.0:
			return {}
		valid_t = (
			valid.to(device=uv.device, dtype=torch.bool)
			& torch.isfinite(uv).all(dim=-1)
			& torch.isfinite(xyz).all(dim=-1)
		)
		row_edge_valid, col_edge_valid, diag00_valid, diag01_valid = opt_loss_flatten._retained_source_edge_masks(
			valid_t,
			cell_valid.to(device=uv.device, dtype=torch.bool),
		)
		sum_ratio = uv.new_zeros(())
		sum_weight = uv.new_zeros(())

		def _accumulate(delta_uv: torch.Tensor, delta_xyz: torch.Tensor, mask: torch.Tensor) -> None:
			nonlocal sum_ratio, sum_weight
			uv_len = torch.linalg.vector_norm(delta_uv, dim=-1)
			phys_len = torch.linalg.vector_norm(delta_xyz, dim=-1)
			ok = mask & torch.isfinite(uv_len) & torch.isfinite(phys_len) & (uv_len > 1.0e-8) & (phys_len > 0.0)
			mask_f = ok.to(dtype=uv.dtype)
			sum_ratio = sum_ratio + ((phys_len / uv_len) * mask_f).sum()
			sum_weight = sum_weight + mask_f.sum()

		if int(uv.shape[0]) > 1:
			_accumulate(
				uv[1:, :] - uv[:-1, :],
				xyz[1:, :] - xyz[:-1, :],
				row_edge_valid & valid_t[1:, :] & valid_t[:-1, :],
			)
		if int(uv.shape[1]) > 1:
			_accumulate(
				uv[:, 1:] - uv[:, :-1],
				xyz[:, 1:] - xyz[:, :-1],
				col_edge_valid & valid_t[:, 1:] & valid_t[:, :-1],
			)
		if int(uv.shape[0]) > 1 and int(uv.shape[1]) > 1:
			_accumulate(
				uv[1:, 1:] - uv[:-1, :-1],
				xyz[1:, 1:] - xyz[:-1, :-1],
				diag00_valid & valid_t[1:, 1:] & valid_t[:-1, :-1],
			)
			_accumulate(
				uv[1:, :-1] - uv[:-1, 1:],
				xyz[1:, :-1] - xyz[:-1, 1:],
				diag01_valid & valid_t[1:, :-1] & valid_t[:-1, 1:],
			)
		if not bool((sum_weight > 0).detach().cpu()):
			return {}
		current_step = float((sum_ratio / sum_weight.clamp_min(1.0)).detach().cpu())
		if not math.isfinite(current_step) or current_step <= 0.0:
			return {}
		scale = current_step / target_step
		if not math.isfinite(scale) or scale <= 0.0:
			return {}
		if abs(scale - 1.0) <= 1.0e-6:
			return {
				"flatten_renorm_step_before": current_step,
				"flatten_renorm_step_after": current_step,
				"flatten_renorm_scale": 1.0,
			}
		center_mask = valid_t
		center_weight = center_mask.to(dtype=uv.dtype).sum()
		if not bool((center_weight > 0).detach().cpu()):
			return {}
		center = (uv * center_mask.unsqueeze(-1).to(dtype=uv.dtype)).sum(dim=(0, 1)) / center_weight.clamp_min(1.0)
		renorm_uv = center + (uv - center) * float(scale)
		flat = renorm_uv.permute(2, 0, 1).unsqueeze(1).contiguous()
		new_pyr = fit_model.Model3D._construct_pyramid_from_flat_3d(
			flat,
			len(model.flatten_map_ms),
			pyramid_d=False,
		)
		if len(new_pyr) != len(model.flatten_map_ms):
			return {}
		for dst, src in zip(model.flatten_map_ms, new_pyr, strict=True):
			if tuple(dst.shape) != tuple(src.shape):
				return {}
		for dst, src in zip(model.flatten_map_ms, new_pyr, strict=True):
			dst.copy_(src.to(device=dst.device, dtype=dst.dtype))
		return {
			"flatten_renorm_step_before": current_step,
			"flatten_renorm_step_after": target_step,
			"flatten_renorm_scale": float(scale),
			"flatten_grid_step_avg": target_step,
		}


def _clear_optimizer_state_for_params(opt: torch.optim.Optimizer, params: list[torch.nn.Parameter]) -> None:
	for p in params:
		opt.state.pop(p, None)


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
	cancel_fn=None,
	ensure_data_fn=None,
	seed_xyz: tuple[float, float, float] | None = None,
	out_dir: str | None = None,
	capture_flow_gate_channels: bool = False,
	cylinder_shell_callback=None,
	self_map_init: str = "off",
	self_map_model_w_wraps: float | None = None,
	init_grow: dict | None = None,
	snap_surf_map_state: dict | None = None,
	require_snap_surf_map_state: bool = False,
	snap_surf_boundary: dict | None = None,
) -> fit_data.FitData3D:
	_optimize_t0 = time.perf_counter()
	opt_loss_corr.reset_state()
	opt_loss_snap_surf.reset_state()
	opt_loss_atlas_line.reset_state()
	_snap_global_runtime: snap_surf_map_global.GlobalMapRuntime | None = None
	_snap_self_runtimes: dict[str, snap_surf_map_global.SelfMapRuntime] = {}
	_loaded_snap_surf_map_state = snap_surf_map_state if isinstance(snap_surf_map_state, dict) else None
	_snap_surf_boundary = snap_surf_boundary if isinstance(snap_surf_boundary, dict) else None
	_snap_global_offset_debug_printed = False
	_map_forward_needs = fit_model.ModelForwardNeeds(mesh_normals=True, ext_surfaces=True)
	snap_global_mesh_epoch = 0
	self_map_mode = snap_surf_map_global.normalize_self_map_init(self_map_init)

	def _runtime_map_model_state(runtime) -> dict | None:
		global_model = getattr(runtime, "global_model", None)
		if global_model is None:
			return None
		return {
			"preserve_batch": bool(getattr(global_model, "preserve_batch", False)),
			"map_uv_ms": [p.detach().cpu() for p in list(getattr(global_model, "map_uv_ms", []))],
		}

	def _restore_runtime_map_model(
		runtime,
		state: dict,
		*,
		expected_mode: str | None = None,
		expected_direction: str | None = None,
	) -> None:
		if expected_mode is not None and str(state.get("mode", expected_mode)).replace("-", "_") != str(expected_mode):
			raise ValueError(f"snap-surf map checkpoint mode mismatch: expected {expected_mode}, got {state.get('mode')}")
		if expected_direction is not None and str(state.get("direction", expected_direction)) != str(expected_direction):
			raise ValueError(
				f"snap-surf map checkpoint direction mismatch: expected {expected_direction}, got {state.get('direction')}"
			)
		tensors = state.get("map_uv_ms")
		if not isinstance(tensors, list) or not tensors:
			raise ValueError("snap-surf map checkpoint is missing map_uv_ms tensors")
		device = next(model.parameters()).device
		params = []
		for t in tensors:
			if not torch.is_tensor(t):
				raise ValueError("snap-surf map checkpoint contains non-tensor map_uv_ms entries")
			params.append(t.to(device=device, dtype=torch.float32).detach().clone())
		first = params[0]
		if first.ndim == 4 and int(first.shape[1]) == 2:
			dense0 = first.permute(0, 2, 3, 1).contiguous()
		elif first.ndim == 4 and int(first.shape[-1]) == 2:
			dense0 = first
		elif first.ndim == 3 and int(first.shape[-1]) == 2:
			dense0 = first
		else:
			raise ValueError(f"snap-surf map checkpoint has unsupported map_uv_ms[0] shape {tuple(first.shape)}")
		global_model = snap_surf_map_global.GlobalMapModel(
			dense0,
			levels=1,
			factor=2,
			preserve_batch=bool(state.get("preserve_batch", first.ndim == 4)),
		)
		global_model.map_uv_ms = torch.nn.ParameterList([torch.nn.Parameter(p) for p in params])
		runtime.global_model = global_model
		if state.get("model_w_wraps") is not None:
			runtime.model_w_wraps = float(state["model_w_wraps"])
		runtime.steps_run = int(state.get("steps_run", getattr(runtime, "steps_run", 0)))

	def _current_snap_surf_map_state() -> dict | None:
		self_maps: dict[str, dict] = {}
		for direction, runtime in sorted(_snap_self_runtimes.items()):
			model_state = _runtime_map_model_state(runtime)
			if model_state is None:
				continue
			model_state.update({
				"mode": runtime.mode,
				"direction": direction,
				"model_w_wraps": runtime.model_w_wraps,
				"steps_run": int(getattr(runtime, "steps_run", 0)),
			})
			self_maps[direction] = model_state
		out: dict[str, object] = {"version": 1, "self_map_init": self_map_mode}
		if self_maps:
			out["self_maps"] = self_maps
		global_state = _runtime_map_model_state(_snap_global_runtime) if _snap_global_runtime is not None else None
		if global_state is not None:
			out["global_map"] = global_state
		return out if ("self_maps" in out or "global_map" in out) else None

	def _publish_snap_surf_map_state() -> None:
		state = _current_snap_surf_map_state()
		if state is not None:
			setattr(model, "_snap_surf_map_state_for_save", state)

	def _identity_self_map_state_for_model(direction: str) -> dict:
		with torch.no_grad():
			model_xyz = model._grid_xyz().detach()
		runtime = snap_surf_map_global.SelfMapRuntime(
			mode=self_map_mode,
			direction=direction,
			model_w_wraps=self_map_model_w_wraps,
		)
		runtime._ensure_model(
			model_xyz,
			snap_surf_map_global.snap_surf_config_from_global_config(runtime.cfg_global),
			snap_surf_map_global.GlobalMapStageConfig(params=("map_uv_ms",)),
		)
		state = _runtime_map_model_state(runtime)
		if state is None:
			raise RuntimeError(f"failed to initialize identity self map state for direction {direction!r}")
		state.update({
			"mode": self_map_mode,
			"direction": direction,
			"model_w_wraps": self_map_model_w_wraps,
			"steps_run": 0,
		})
		return state

	def _state_level_batches(state: dict) -> list[torch.Tensor]:
		raw = state.get("map_uv_ms")
		if not isinstance(raw, list) or not raw:
			raise RuntimeError("snap-surf map state is missing map_uv_ms tensors")
		out: list[torch.Tensor] = []
		for t in raw:
			if not torch.is_tensor(t):
				raise RuntimeError("snap-surf map state contains non-tensor map_uv_ms entry")
			tt = t.detach().cpu()
			if tt.ndim == 3:
				tt = tt.unsqueeze(0)
			if tt.ndim != 4 or (int(tt.shape[-1]) != 2 and int(tt.shape[1]) != 2):
				raise RuntimeError(f"snap-surf map tensor must be batchable UV data, got {tuple(tt.shape)}")
			out.append(tt)
		return out

	def _replace_self_map_batch(base_state: dict, source_state: dict, *, batch_index: int) -> dict:
		base_levels = _state_level_batches(base_state)
		source_levels = _state_level_batches(source_state)
		if len(base_levels) != len(source_levels):
			raise RuntimeError(
				f"cannot fuse snap-surf map levels: self={len(base_levels)} source={len(source_levels)}"
			)
		fused_levels: list[torch.Tensor] = []
		for bi, si in zip(base_levels, source_levels):
			if int(si.shape[0]) != 1:
				raise RuntimeError(f"expand-z boundary map must have one batch, got {int(si.shape[0])}")
			if tuple(bi.shape[1:]) != tuple(si.shape[1:]):
				raise RuntimeError(
					f"cannot fuse snap-surf map level shape self={tuple(bi.shape)} source={tuple(si.shape)}"
				)
			if not (0 <= int(batch_index) < int(bi.shape[0])):
				raise RuntimeError(
					f"self-map batch index {int(batch_index)} outside tensor batch {int(bi.shape[0])}"
				)
			bf = bi.clone()
			bf[int(batch_index)] = si[0]
			fused_levels.append(bf)
		out = dict(base_state)
		out["preserve_batch"] = True
		out["map_uv_ms"] = fused_levels
		return out

	def _copy_self_map_existing_batches(base_state: dict, old_state: dict | None) -> dict:
		if old_state is None:
			return base_state
		base_levels = _state_level_batches(base_state)
		old_levels = _state_level_batches(old_state)
		if len(base_levels) != len(old_levels):
			raise RuntimeError(
				f"cannot preserve snap-surf map levels: new={len(base_levels)} old={len(old_levels)}"
			)
		fused_levels: list[torch.Tensor] = []
		for bi, oi in zip(base_levels, old_levels):
			if tuple(bi.shape[1:]) != tuple(oi.shape[1:]):
				raise RuntimeError(
					f"cannot preserve snap-surf map level shape new={tuple(bi.shape)} old={tuple(oi.shape)}"
				)
			if int(oi.shape[0]) > int(bi.shape[0]):
				raise RuntimeError(
					f"old snap-surf map batch {int(oi.shape[0])} exceeds new batch {int(bi.shape[0])}"
				)
			bf = bi.clone()
			if int(oi.shape[0]) > 0:
				bf[:int(oi.shape[0])] = oi
			fused_levels.append(bf)
		out = dict(base_state)
		out["preserve_batch"] = True
		out["map_uv_ms"] = fused_levels
		return out

	def _fuse_expand_z_snap_surf_maps_append(
		*,
		old_state: dict | None,
		temp_state: dict | None,
		old_depth: int,
		add_depth: int,
	) -> None:
		if self_map_mode == "off":
			return
		if add_depth != 1:
			raise NotImplementedError("expand-z map fusion currently requires init-grow.step=1")
		if temp_state is None or not isinstance(temp_state.get("self_maps"), dict):
			raise RuntimeError("expand-z map fusion requires optimized temporary boundary snap-surf self_maps")
		if "global_map" in temp_state:
			raise RuntimeError("expand-z map fusion no longer accepts temporary snap-surf global_map state")
		temp_self = temp_state["self_maps"]
		missing = [
			d for d in ("out", "in")
			if not isinstance(temp_self.get(d), dict)
		]
		if missing:
			raise RuntimeError(f"expand-z map fusion requires boundary snap-surf self maps: missing {missing}")
		old_self = old_state.get("self_maps", {}) if isinstance(old_state, dict) else {}
		if old_self is not None and not isinstance(old_self, dict):
			raise RuntimeError("existing snap-surf self map state is malformed")
		boundary_batch = int(old_depth) - 1
		self_maps: dict[str, dict] = {}
		for direction in ("out", "in"):
			old_direction_state = (
				old_self.get(direction)
				if isinstance(old_self, dict) and isinstance(old_self.get(direction), dict)
				else None
			)
			base_state = _copy_self_map_existing_batches(
				_identity_self_map_state_for_model(direction),
				old_direction_state,
			)
			boundary_state = temp_self[direction]
			base_state = _replace_self_map_batch(
				base_state,
				boundary_state,
				batch_index=boundary_batch,
			)
			base_state["steps_run"] = int(boundary_state.get("steps_run", 0))
			base_state.update({
				"mode": self_map_mode,
				"direction": direction,
				"model_w_wraps": self_map_model_w_wraps,
			})
			self_maps[direction] = base_state
		setattr(model, "_snap_surf_map_state_for_save", {
			"version": 1,
			"self_map_init": self_map_mode,
			"self_maps": self_maps,
		})

	if require_snap_surf_map_state and self_map_mode != "off":
		if _loaded_snap_surf_map_state is None or not isinstance(_loaded_snap_surf_map_state.get("self_maps"), dict):
			raise ValueError("self-map reopt requires checkpoint snap-surf self maps")
		missing = [d for d in ("out", "in") if d not in _loaded_snap_surf_map_state["self_maps"]]
		if missing:
			raise ValueError(f"self-map reopt checkpoint is missing snap-surf self maps: {missing}")

	def _bump_snap_global_mesh_epoch() -> None:
		nonlocal snap_global_mesh_epoch
		snap_global_mesh_epoch += 1

	def _unpack_ext_surface_record(record):
		if len(record) < 4:
			raise RuntimeError("external surface record must have at least 4 fields")
		offset = float(record[4]) if len(record) >= 5 else 0.0
		return record[0], record[1], record[2], record[3], offset

	def _print_snap_global_offset_debug(records) -> None:
		nonlocal _snap_global_offset_debug_printed
		if _snap_global_offset_debug_printed:
			return
		offsets = [float(record[4]) if len(record) >= 5 else 0.0 for record in records]
		print(
			"[snap_surf.map_global] external surface offsets at first map call: "
			f"configured={offsets} used_by_map_optimizer=not_applied "
			"interpretation=correspondence_only; snap_surf_map loss applies nonzero offsets as winding residuals",
			flush=True,
		)
		_snap_global_offset_debug_printed = True

	def _snap_global_model_offset_text() -> str:
		offsets = [float(v) for v in getattr(model, "_ext_offsets", [])]
		if not offsets:
			return "none"
		if len(offsets) == 1:
			return f"{offsets[0]:.6g}"
		return "[" + ",".join(f"{v:.6g}" for v in offsets) + "]"

	def _snap_global_runtime_for(stage_args: dict | None = None) -> snap_surf_map_global.GlobalMapRuntime:
		nonlocal _snap_global_runtime
		if _snap_global_runtime is None:
			base = {}
			if isinstance(stage_args, dict):
				if isinstance(stage_args.get("map_global"), dict):
					base = dict(stage_args.get("map_global"))
			_snap_global_runtime = snap_surf_map_global.GlobalMapRuntime(base=base, seed_xyz=seed_xyz)
		return _snap_global_runtime

	def _snap_self_runtime_for(direction: str, stage_args: dict | None = None) -> snap_surf_map_global.SelfMapRuntime:
		direction_i = str(direction).strip().lower()
		runtime = _snap_self_runtimes.get(direction_i)
		if runtime is None:
			base = {}
			if isinstance(stage_args, dict) and isinstance(stage_args.get("map_global"), dict):
				base = dict(stage_args.get("map_global"))
			if _snap_surf_boundary is not None:
				missing = [
					k for k in ("fixed_xyz", "fixed_normals", "fixed_valid")
					if k not in _snap_surf_boundary
				]
				if missing:
					raise ValueError(f"boundary self-map grow is missing fixed tensors: {missing}")
				runtime = snap_surf_map_global.BoundarySelfMapRuntime(
					mode=self_map_mode,
					direction=direction_i,
					model_w_wraps=self_map_model_w_wraps,
					base=base,
					fixed_xyz=_snap_surf_boundary["fixed_xyz"],
					fixed_normals=_snap_surf_boundary["fixed_normals"],
					fixed_valid=_snap_surf_boundary["fixed_valid"],
				)
			else:
				runtime = snap_surf_map_global.SelfMapRuntime(
					mode=self_map_mode,
					direction=direction_i,
					model_w_wraps=self_map_model_w_wraps,
					base=base,
				)
			if _loaded_snap_surf_map_state is not None:
				self_maps = _loaded_snap_surf_map_state.get("self_maps", {})
				if isinstance(self_maps, dict) and direction_i in self_maps:
					_restore_runtime_map_model(
						runtime,
						self_maps[direction_i],
						expected_mode=self_map_mode,
						expected_direction=direction_i,
					)
				elif require_snap_surf_map_state and self_map_mode != "off":
					raise ValueError(f"self-map reopt checkpoint is missing snap-surf self map '{direction_i}'")
			_snap_self_runtimes[direction_i] = runtime
		return runtime

	def _self_map_directions(stage_args: dict | None = None) -> list[str]:
		if self_map_mode == "off":
			return []
		raw = (stage_args or {}).get("self_map_direction", (stage_args or {}).get("self-map-direction", "both"))
		mode = str(raw).strip().lower()
		if mode in {"both", "all"}:
			return ["out", "in"]
		if mode in {"out", "in"}:
			return [mode]
		raise ValueError(f"invalid self_map_direction {raw!r} (expected out, in, or both)")

	def _run_snap_global_map_stage(
		*,
		stage: snap_surf_map_global.GlobalMapStageConfig,
		res,
		stage_args: dict | None,
		persistent_optimizer: bool,
		status_fn=None,
		auto_stop_fn=None,
	) -> dict[str, float]:
		if self_map_mode != "off":
			if res.normals is None:
				raise RuntimeError("self snap_surf global map optimizer requires model normals")
			all_stats: dict[str, float] = {}
			for direction in _self_map_directions(stage_args):
				stats = _snap_self_runtime_for(direction, stage_args).run_stage(
					stage=stage,
					model_xyz=res.xyz_lr,
					model_normals=res.normals,
					model_valid=torch.isfinite(res.xyz_lr).all(dim=-1),
					persistent_optimizer=persistent_optimizer,
					status_fn=status_fn,
					cancel_fn=cancel_fn,
					auto_stop_fn=auto_stop_fn,
				)
				for k, v in stats.items():
					all_stats[k] = all_stats.get(k, 0.0) + float(v) / float(max(1, len(_self_map_directions(stage_args))))
			_publish_snap_surf_map_state()
			return all_stats
		records = getattr(res, "ext_surfaces", None)
		if not records:
			raise RuntimeError("snap_surf global map optimizer requires external_surfaces")
		if res.normals is None:
			raise RuntimeError("snap_surf global map optimizer requires model normals")
		_print_snap_global_offset_debug(records)
		ext_xyz, ext_valid, ext_normals, ext_quad_valid, _offset = _unpack_ext_surface_record(records[0])
		runtime = _snap_global_runtime_for(stage_args)
		stats = runtime.run_stage(
			stage=stage,
			model_xyz=res.xyz_lr,
			model_normals=res.normals,
			model_valid=torch.isfinite(res.xyz_lr).all(dim=-1),
			ext_xyz=ext_xyz,
			ext_valid=ext_valid,
			ext_normals=ext_normals,
			ext_quad_valid=ext_quad_valid,
			external_surface_index=0,
			mesh_epoch=snap_global_mesh_epoch,
			persistent_optimizer=persistent_optimizer,
			status_fn=status_fn,
			cancel_fn=cancel_fn,
			auto_stop_fn=auto_stop_fn,
		)
		_publish_snap_surf_map_state()
		return stats

	def _compact_snap_global_map_stats(stats: dict[str, float]) -> dict[str, float]:
		return {
			k: float(stats[k])
			for k in ("snaps_map_loss", "snaps_map_dist", "snaps_map_vec", "snaps_map_norm", "snaps_map_turn", "snaps_map_turn_smp")
			if k in stats
		}

	def _snap_global_map_loss(*, res) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
		if self_map_mode != "off":
			if res.normals is None:
				raise RuntimeError("self snap_surf_map requires model normals")
			loss_total = res.xyz_lr.sum() * 0.0
			lms_all: tuple[torch.Tensor, ...] = ()
			masks_all: tuple[torch.Tensor, ...] = ()
			stats_acc: dict[str, float] = {}
			directions = _self_map_directions()
			for direction in directions:
				loss, lms, masks, stats = _snap_self_runtime_for(direction).snap_loss(
					model_xyz=res.xyz_lr,
					model_normals=res.normals,
					model_valid=torch.isfinite(res.xyz_lr).all(dim=-1),
					offset=1.0,
					data=res.data,
					strip_samples=max(2, int(res.params.subsample_mesh) + 1),
				)
				loss_total = loss_total + loss / float(max(1, len(directions)))
				lms_all = lms_all + tuple(lms)
				masks_all = masks_all + tuple(masks)
				for k, v in stats.items():
					stats_acc[k] = stats_acc.get(k, 0.0) + float(v) / float(max(1, len(directions)))
			opt_loss_snap_surf.update_last_stats(stats_acc)
			_publish_snap_surf_map_state()
			return loss_total, lms_all, masks_all
		records = getattr(res, "ext_surfaces", None)
		if not records:
			raise RuntimeError("snap_surf_map requires external_surfaces")
		if res.normals is None:
			raise RuntimeError("snap_surf_map requires model normals")
		ext_xyz, ext_valid, ext_normals, ext_quad_valid, offset = _unpack_ext_surface_record(records[0])
		loss, lms, masks, stats = _snap_global_runtime_for().snap_loss(
			model_xyz=res.xyz_lr,
			model_normals=res.normals,
			model_valid=torch.isfinite(res.xyz_lr).all(dim=-1),
			ext_xyz=ext_xyz,
			ext_valid=ext_valid,
			ext_normals=ext_normals,
			ext_quad_valid=ext_quad_valid,
			offset=offset,
			data=res.data,
			strip_samples=max(2, int(res.params.subsample_mesh) + 1),
		)
		opt_loss_snap_surf.update_last_stats(stats)
		_publish_snap_surf_map_state()
		return loss, lms, masks

	def _stage_start(name: str) -> float:
		return 0.0

	def _stage_done(name: str, t0: float) -> None:
		return None

	def _is_cyl_stage(stage_: Stage) -> bool:
		return stage_.global_opt is not None and "cyl_params" in stage_.global_opt.params

	def _cyl_stage_width_target_step(opt_cfg_: OptSettings) -> float | None:
		args = opt_cfg_.args if isinstance(opt_cfg_.args, dict) else {}
		if CYLINDER_STAGE_STEP_ARG not in args or args[CYLINDER_STAGE_STEP_ARG] is None:
			return None
		value = float(args[CYLINDER_STAGE_STEP_ARG])
		if value <= 0.0:
			raise ValueError(f"cylinder stage arg '{CYLINDER_STAGE_STEP_ARG}' must be > 0, got {value}")
		return value

	def _cyl_stage_output_all_shells() -> bool:
		for stage_ in stages:
			if stage_.global_opt is None:
				continue
			args = stage_.global_opt.args if isinstance(stage_.global_opt.args, dict) else {}
			for key in CYLINDER_OUTPUT_ALL_SHELLS_ARGS:
				if key in args and _truthy(args.get(key)):
					return True
		return False

	def _cyl_stage_max_search_shells(default: int) -> int:
		for stage_ in stages:
			if stage_.global_opt is None:
				continue
			args = stage_.global_opt.args if isinstance(stage_.global_opt.args, dict) else {}
			for key in CYLINDER_MAX_SEARCH_SHELLS_ARGS:
				if key not in args or args.get(key) is None:
					continue
				value = int(args.get(key))
				if value <= 0:
					raise ValueError(f"cylinder stage arg '{key}' must be > 0, got {value}")
				return value
		return int(default)

	def _cyl_stage_grow_direction(opt_cfg_: OptSettings) -> int:
		args = opt_cfg_.args if isinstance(opt_cfg_.args, dict) else {}
		return normalize_cylinder_grow_direction(args.get(CYLINDER_GROW_DIRECTION_ARG, "outward"))

	def _cyl_outside_enabled(eff_: dict[str, float]) -> bool:
		return _need_term("cyl_outside", eff_) > 0.0

	def _cyl_outside_grid_step(stage_args_: dict) -> float:
		value = stage_args_.get(CYLINDER_OUTSIDE_GRID_STEP_ARG, cyl_sdf_volume.DEFAULT_CYL_OUTSIDE_GRID_STEP)
		value = float(value)
		if value <= 0.0:
			raise ValueError(f"cylinder stage arg '{CYLINDER_OUTSIDE_GRID_STEP_ARG}' must be > 0, got {value}")
		return value

	def _cyl_outside_sample_factor(stage_args_: dict) -> int:
		value = int(stage_args_.get(CYLINDER_OUTSIDE_SAMPLE_FACTOR_ARG, 2))
		if value <= 0:
			raise ValueError(f"cylinder stage arg '{CYLINDER_OUTSIDE_SAMPLE_FACTOR_ARG}' must be > 0, got {value}")
		return value

	def _cyl_outside_threads(stage_args_: dict) -> int:
		raw = stage_args_.get(CYLINDER_OUTSIDE_THREADS_ARG, os.environ.get("LASAGNA_CYL_OUTSIDE_THREADS", 0))
		value = int(raw)
		if value < 0:
			raise ValueError(f"cylinder stage arg '{CYLINDER_OUTSIDE_THREADS_ARG}' must be >= 0, got {value}")
		return value

	def _cyl_outside_chunk_size(stage_args_: dict) -> int:
		return int(stage_args_.get(
			CYLINDER_OUTSIDE_CHUNK_SIZE_ARG,
			cyl_sdf_volume.DEFAULT_CYL_OUTSIDE_CHUNK_SIZE,
		))

	def _cyl_outside_deep_interp_chunks(stage_args_: dict) -> float:
		return float(stage_args_.get(
			CYLINDER_OUTSIDE_DEEP_INTERP_CHUNKS_ARG,
			cyl_sdf_volume.DEFAULT_CYL_OUTSIDE_DEEP_INTERP_CHUNKS,
		))

	def _cyl_outside_deep_blend_chunks(stage_args_: dict) -> float:
		value = float(stage_args_.get(
			CYLINDER_OUTSIDE_DEEP_BLEND_CHUNKS_ARG,
			cyl_sdf_volume.DEFAULT_CYL_OUTSIDE_DEEP_BLEND_CHUNKS,
		))
		if value < 0.0:
			raise ValueError(f"cylinder stage arg '{CYLINDER_OUTSIDE_DEEP_BLEND_CHUNKS_ARG}' must be >= 0, got {value}")
		return value

	def _clear_cyl_outside_field() -> None:
		if hasattr(model, "clear_cyl_outside_volume"):
			model.clear_cyl_outside_volume()
			return
		setattr(model, "cyl_outside_volume", None)
		setattr(model, "cyl_outside_origin", None)
		setattr(model, "cyl_outside_spacing", None)
		setattr(model, "cyl_outside_shape", None)
		setattr(model, "cyl_outside_depth_max", 0.0)
		setattr(model, "cyl_outside_model_step", None)

	def _set_cyl_outside_field(field: cyl_sdf_volume.CylOutsideVolume, *, sample_factor: int, model_step: float) -> None:
		if hasattr(model, "set_cyl_outside_volume"):
			model.set_cyl_outside_volume(field, sample_factor=sample_factor, model_step=model_step)
			return
		setattr(model, "cyl_outside_volume", field.volume.detach().contiguous())
		setattr(model, "cyl_outside_origin", tuple(float(v) for v in field.origin))
		setattr(model, "cyl_outside_spacing", tuple(float(v) for v in field.spacing))
		setattr(model, "cyl_outside_shape", tuple(int(v) for v in field.shape))
		setattr(model, "cyl_outside_depth_max", float(field.depth_max))
		setattr(model, "cyl_outside_sample_factor", max(1, int(sample_factor)))
		setattr(model, "cyl_outside_model_step", max(1.0e-6, float(model_step)))

	def _configure_cyl_outside_step(*, sample_factor: int, model_step: float) -> None:
		setattr(model, "cyl_outside_sample_factor", max(1, int(sample_factor)))
		setattr(model, "cyl_outside_model_step", max(1.0e-6, float(model_step)))

	def _build_cyl_outside_from_completed(
		*,
		eff_: dict[str, float],
		stage_args_: dict,
		model_step: float,
		label_: str,
		direction: int,
	) -> None:
		if not _cyl_outside_enabled(eff_):
			return
		shells = getattr(model, "cyl_shell_completed", None)
		if not shells:
			_clear_cyl_outside_field()
			return
		device = next(model.parameters()).device
		grid_step = _cyl_outside_grid_step(stage_args_)
		sample_factor = _cyl_outside_sample_factor(stage_args_)
		threads = _cyl_outside_threads(stage_args_)
		chunk_size = _cyl_outside_chunk_size(stage_args_)
		deep_interp_chunks = _cyl_outside_deep_interp_chunks(stage_args_)
		deep_blend_chunks = _cyl_outside_deep_blend_chunks(stage_args_)
		mode = _cyl_outside_mode_for_direction(direction)
		depth_cap = cyl_sdf_volume.CYL_OUTSIDE_BARRIER_DEPTH_MAX
		_bbox = cyl_sdf_volume.default_shell_bbox(shells[-1], grid_step=grid_step)
		_origin, _shape = cyl_sdf_volume.shape_for_bbox(_bbox, grid_step=grid_step)
		_voxels = int(_shape[0]) * int(_shape[1]) * int(_shape[2])
		print(
			f"[optimizer] {label_}: building cyl_outside previous-shell field "
			f"mode={mode} shape={_shape} voxels={_voxels} grid_step={grid_step:.1f} "
			f"bbox_padding={depth_cap:.1f} depth_cap={depth_cap:.1f} "
			f"threads={'auto' if threads == 0 else threads} chunk_size={chunk_size} "
			f"deep_interp_chunks={deep_interp_chunks:g} deep_blend_chunks={deep_blend_chunks:g}; "
			f"first run may compile the libigl extension",
			flush=True,
		)
		field = cyl_sdf_volume.build_previous_shell_violation_depth_volume(
			shells[-1].detach(),
			mode=mode,
			grid_step=grid_step,
			device=device,
			progress_label=label_,
			threads=threads,
			chunk_size=chunk_size,
			deep_interp_chunks=deep_interp_chunks,
			deep_blend_chunks=deep_blend_chunks,
		)
		_set_cyl_outside_field(field, sample_factor=sample_factor, model_step=model_step)
		print(
			f"[optimizer] {label_}: cyl_outside field shape={field.shape} "
			f"mode={mode} "
			f"origin=({field.origin[0]:.1f},{field.origin[1]:.1f},{field.origin[2]:.1f}) "
			f"grid_step={grid_step:.1f} depth_max={field.depth_max:.3f}",
			flush=True,
		)

	def _next_stage_is_cyl(si_: int) -> bool:
		return si_ + 1 < len(stages) and _is_cyl_stage(stages[si_ + 1])

	def _shell_pass_count(shell_i_: int) -> int:
		if hasattr(model, "cylinder_shell_pass_count"):
			return max(1, int(model.cylinder_shell_pass_count(int(shell_i_))))
		return 1

	def _scheduled_total_steps() -> int:
		if not bool(getattr(model, "cyl_shell_mode", False)):
			return total_steps_for_stages(stages)
		total = 0
		for stage_ in stages:
			if stage_.global_opt is None:
				continue
			steps_ = max(0, int(stage_.global_opt.steps))
			total += steps_
		return total

	_cyl_output_all_shells = _cyl_stage_output_all_shells()
	if _cyl_output_all_shells:
		setattr(model, "cyl_shell_output_all_shells", True)

	def _collapse_cylinder_shells_to_last() -> None:
		if _cyl_output_all_shells:
			return
		if getattr(model, "cyl_shell_completed", None):
			model.cyl_shell_completed = [model.cyl_shell_completed[-1]]
			model.cyl_shell_current_index = 0

	Needs = fit_model.ModelForwardNeeds
	terms = {
		"step": {"loss": opt_loss_step.step_loss, "needs": Needs()},
		"step_regularizer": {
			"loss": opt_loss_step.step_regularizer_loss,
			"sub": ["smooth_step", "avg_step"],
			"needs": Needs(),
		},
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
		"snap_surf_map": {
			"loss": _snap_global_map_loss,
			"needs": Needs(
				mesh_normals=True,
				ext_surfaces=(self_map_mode == "off"),
				lr_prefetch_channels=frozenset({"grad_mag"}),
				prefetch_snap_surf_map=True,
			),
		},
		"atlas_line": {
			"loss": opt_loss_atlas_line.atlas_line_loss,
			"sub": ["atlas_line", "atlas_line_control", "atlas_line_other", "atlas_line_snap"],
			"needs": Needs(mesh_normals=True),
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
		"cyl_z_center": {"loss": opt_loss_cyl.cyl_z_center_loss, "needs": Needs(cyl_samples=True)},
		"cyl_step_push": {
			"loss": opt_loss_cyl.cyl_step_push_loss,
			"needs": Needs(cyl_samples=True, cyl_shell_fields=True, prefetch_cyl_grad_mask=True),
		},
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
		"cyl_outside": {
			"loss": opt_loss_cyl.cyl_outside_loss,
			"needs": Needs(cyl_samples=True, cyl_shell_fields=True, prefetch_cyl_grad_mask=True),
		},
		"flatten_sdir": {
			"loss": opt_loss_flatten.flatten_sdir_loss,
			"needs": Needs(flatten=True),
		},
		"flatten_map_step": {
			"loss": opt_loss_flatten.flatten_map_step_loss,
			"needs": Needs(flatten=True),
		},
		"flatten_edge_step": {
			"loss": opt_loss_flatten.flatten_edge_step_loss,
			"needs": Needs(flatten=True),
		},
		"flatten_avg_offset": {
			"loss": opt_loss_flatten.flatten_avg_offset_loss,
			"needs": Needs(flatten=True),
		},
		"flatten_orient": {
			"loss": opt_loss_flatten.flatten_orient_loss,
			"needs": Needs(flatten=True),
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
					if bool(pred_dt_flow_gate_cfg_.get("atlas_snap_seed_enabled", False)):
						needs = needs.merged(Needs(mesh_normals=True))
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
		if required.ext_surfaces and res_.ext_surfaces is None:
			missing.append("ext_surfaces")
		if required.flatten:
			if res_.flatten_map is None:
				missing.append("flatten_map")
			if res_.flatten_xyz is None:
				missing.append("flatten_xyz")
			if res_.flatten_point_mask is None:
				missing.append("flatten_point_mask")
			if res_.flatten_quad_mask is None:
				missing.append("flatten_quad_mask")
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
				if res_.cyl_shell_delta_xyz is None:
					missing.append("cyl_shell_delta_xyz")
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

	def _stage_eff_for_opt(*, is_cyl_stage_: bool, opt_cfg_: OptSettings) -> dict[str, float]:
		if not is_cyl_stage_:
			return opt_cfg_.eff
		return {
			"cyl_normal": float(opt_cfg_.eff.get("cyl_normal", 0.0)),
			"cyl_center": float(opt_cfg_.eff.get("cyl_center", 0.0)),
			"cyl_smooth": float(opt_cfg_.eff.get("cyl_smooth", 0.0)),
			"cyl_z_smooth": float(opt_cfg_.eff.get("cyl_z_smooth", 0.0)),
			"cyl_z_center": float(opt_cfg_.eff.get("cyl_z_center", 0.0)),
			"cyl_step_push": float(opt_cfg_.eff.get("cyl_step_push", 0.0)),
			"cyl_step": float(opt_cfg_.eff.get("cyl_step", 0.0)),
			"cyl_radial_mean": float(opt_cfg_.eff.get("cyl_radial_mean", 0.0)),
			"cyl_bend": float(opt_cfg_.eff.get("cyl_bend", 0.0)),
			"cyl_conn_mesh": float(opt_cfg_.eff.get("cyl_conn_mesh", 0.0)),
			"cyl_conn_gt": float(opt_cfg_.eff.get("cyl_conn_gt", 0.0)),
			"cyl_base_mesh": float(opt_cfg_.eff.get("cyl_base_mesh", 0.0)),
			"cyl_base_gt": float(opt_cfg_.eff.get("cyl_base_gt", 0.0)),
			"cyl_outside": float(opt_cfg_.eff.get("cyl_outside", 0.0)),
		}

	def _run_opt(*, si: int, label: str, stage: Stage, opt_cfg: OptSettings, data: fit_data.FitData3D) -> fit_data.FitData3D:
		_t_stage_total = _stage_start(f"{label}.total")
		is_cyl_stage = "cyl_params" in opt_cfg.params
		is_cyl_shelling_stage = is_cyl_stage and stage.name in CYLINDER_SEED_INIT_STAGE_ROLES
		if bool(getattr(model, "cyl_shell_abort", False)):
			_stage_done(f"{label}.total", _t_stage_total)
			return data
		if not is_cyl_shelling_stage:
			print(f"[optimizer] {label}: params={opt_cfg.params} steps={_steps_label(opt_cfg)} "
				  f"lr={opt_cfg.lr} min_scaledown={opt_cfg.min_scaledown}", flush=True)
		if opt_cfg.steps <= 0 and not is_cyl_stage:
			return data
		stage_eff = _stage_eff_for_opt(is_cyl_stage_=is_cyl_stage, opt_cfg_=opt_cfg)
		if is_cyl_stage and stage.name != "cyl_grow":
			stage_eff["cyl_step_push"] = 0.0
		stage_uses_cyl_loss = (
			_need_term("cyl_normal", stage_eff) > 0 or
			_need_term("cyl_center", stage_eff) > 0 or
			_need_term("cyl_smooth", stage_eff) > 0 or
			_need_term("cyl_z_smooth", stage_eff) > 0 or
			_need_term("cyl_z_center", stage_eff) > 0 or
			_need_term("cyl_step_push", stage_eff) > 0 or
			_need_term("cyl_step", stage_eff) > 0 or
			_need_term("cyl_radial_mean", stage_eff) > 0 or
			_need_term("cyl_bend", stage_eff) > 0 or
			_need_term("cyl_conn_mesh", stage_eff) > 0 or
			_need_term("cyl_conn_gt", stage_eff) > 0 or
			_need_term("cyl_base_mesh", stage_eff) > 0 or
			_need_term("cyl_base_gt", stage_eff) > 0 or
			_need_term("cyl_outside", stage_eff) > 0
		)
		stage_args = opt_cfg.args or {}
		atlas_debug_objs = bool(stage_args.get("atlas_debug_objs", stage_args.get("atlas-debug-objs", False)))
		atlas_debug_obj_interval = max(1, int(stage_args.get(
			"atlas_debug_obj_interval",
			stage_args.get("atlas-debug-obj-interval", 1),
		)))
		status_interval_raw = stage_args.get("status_interval", stage_args.get("debug_print_interval", 100))
		status_interval = max(0, int(status_interval_raw))
		steps_label = _steps_label(opt_cfg)
		lr_warmup_steps = _lr_warmup_steps(stage_args)
		auto_window = _auto_steps_window(stage_args) if opt_cfg.steps_auto else 0
		auto_min = _auto_steps_min(stage_args, window=auto_window) if opt_cfg.steps_auto else 0
		auto_rel_threshold = _auto_steps_rel_threshold(stage_args) if opt_cfg.steps_auto else 0.0
		opt_timing_enabled = optimizer_timing.opt_timing_enabled(stage_args)
		opt_timing_interval = optimizer_timing.opt_timing_interval(stage_args, fallback=max(1, status_interval or 100))
		opt_timing_sync = optimizer_timing.opt_timing_sync_cuda(stage_args)
		flatten_max_update = (
			_flatten_max_update_base(stage_args)
			if "map_flatten_ms" in opt_cfg.params and bool(getattr(model, "flatten_enabled", False))
			else 0.0
		)
		flatten_renorm_interval = (
			max(0, int(stage_args.get("flatten_renorm_interval", 0)))
			if "map_flatten_ms" in opt_cfg.params and bool(getattr(model, "flatten_enabled", False))
			else 0
		)
		fused_flatten_adam_clamp = _truthy(os.environ.get(
			"LASAGNA_FUSED_FLATTEN_ADAM_CLAMP",
			stage_args.get("fused_flatten_adam_clamp", False),
		))
		if opt_timing_enabled and not is_cyl_shelling_stage:
			print(
				f"[optimizer] {label}: opt timing enabled interval={opt_timing_interval} "
				f"sync_cuda={int(opt_timing_sync)}",
				flush=True,
			)
		if opt_cfg.steps_auto and not is_cyl_shelling_stage:
			print(
				f"[optimizer] {label}: auto steps max={opt_cfg.steps} window={auto_window} "
				f"min={auto_min} rel_threshold={auto_rel_threshold:g}",
				flush=True,
			)
		if lr_warmup_steps > 0 and not is_cyl_shelling_stage:
			print(
				f"[optimizer] {label}: lr warmup steps={lr_warmup_steps}",
				flush=True,
			)

		if opt_cfg.kind == "map":
			if self_map_mode == "off" and not getattr(model, "_ext_surfaces", None):
				raise ValueError("snap_surf global map stages require external_surfaces")
			map_stage = _global_map_stage_from_opt_settings(name=stage.name, opt_cfg=opt_cfg, args=stage_args)
			map_stage = replace(
				map_stage,
				args=_resolve_snap_surf_map_fixture_export_args(map_stage.args, out_dir=out_dir),
			)
			_map_status_printer = snap_surf_map_global.MapRuntimeStatusPrinter(label=label, total_steps=max(0, opt_cfg.steps))
			_map_auto_stop_info: dict[str, float] = {}

			def _map_auto_stop_fn(*, history: list[float], step: int) -> bool:
				if int(step) < int(auto_min):
					return False
				rel_improvement = _auto_steps_relative_improvement(history, window=auto_window)
				if rel_improvement >= float(auto_rel_threshold):
					return False
				_map_auto_stop_info["post_warmup_step"] = float(step)
				_map_auto_stop_info["rel_improvement"] = float(rel_improvement)
				return True

			def _print_map_status(*, step: int, total: int, stats: dict[str, float]) -> None:
				_map_status_printer.print(
					step=int(step),
					total=int(total),
					stats=stats,
					fallback_lr=float(opt_cfg.lr if isinstance(opt_cfg.lr, (int, float)) else _lr_last(opt_cfg.lr)),
				)
				if progress_fn is not None:
					_stage_total = max(1, int(total))
					_stage_step = min(max(0, int(step)), _stage_total)
					_scheduled_stage_steps = max(0, int(opt_cfg.steps))
					_done = _done_steps[0] + min(_stage_step, _scheduled_stage_steps)
					progress_fn(
						step=_done,
						total=_total_steps,
						loss=float(stats.get("snaps_map_loss", 0.0)),
						stage_progress=_stage_step / _stage_total,
						overall_progress=(
							(si + (_stage_step / _stage_total)) / _num_stages
							if _num_stages > 0 else 1.0
						),
						stage_name=stage.name,
					)

			res_map = model(data, needs=_map_forward_needs)
			stats = _run_snap_global_map_stage(
				stage=map_stage,
				res=res_map,
				stage_args=stage_args,
				persistent_optimizer=False,
				status_fn=_print_map_status,
				auto_stop_fn=_map_auto_stop_fn if opt_cfg.steps_auto else None,
			)
			opt_loss_snap_surf.update_last_stats(stats)
			stage_steps_done = int(stats.get("snaps_map_stage_steps", opt_cfg.steps))
			stage_steps_done = min(max(0, stage_steps_done), max(0, int(opt_cfg.steps)))
			_done_steps[0] += stage_steps_done
			if progress_fn is not None:
				progress_fn(
					step=_done_steps[0],
					total=_total_steps,
					loss=float(stats.get("snaps_map_loss", 0.0)),
					stage_progress=1.0,
					overall_progress=(
						(si + 1) / _num_stages if _num_stages > 0 else 1.0
					),
					stage_name=stage.name,
				)
			if opt_cfg.steps_auto and _map_auto_stop_info:
				print(
					f"[optimizer] {label}: auto steps stopped at "
					f"{stage_steps_done}/{opt_cfg.steps} "
					f"post_warmup={int(_map_auto_stop_info['post_warmup_step'])} "
					f"rel_improvement_{auto_window}={_map_auto_stop_info['rel_improvement']:.6g} "
					f"< {auto_rel_threshold:g}",
					flush=True,
				)
			print(
				f"[optimizer] {label}: snap_surf_global_map "
				f"loss={stats.get('snaps_map_loss', 0.0):.6g} "
				f"turn={stats.get('snaps_map_turn', 0.0):.6g} "
				f"turn_smp={stats.get('snaps_map_turn_smp', 0.0):.0f} "
				f"samples={stats.get('snaps_map_samples', 0.0):.0f}",
				flush=True,
			)
			_stage_done(f"{label}.total", _t_stage_total)
			return data

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
			capture_channels=bool(capture_flow_gate_channels),
		)
		snap_surf_map_args = stage_args.get("snap_surf_map")
		if snap_surf_map_args is not None and not isinstance(snap_surf_map_args, dict):
			raise ValueError(f"stage '{stage.name}' opt.args.snap_surf_map must be an object")
		if isinstance(snap_surf_map_args, dict):
			snap_surf_map_args = dict(snap_surf_map_args)

		snap_surf_map_opt_stage: snap_surf_map_global.GlobalMapStageConfig | None = None
		raw_map_opt = None
		if isinstance(snap_surf_map_args, dict) and "map_opt" in snap_surf_map_args:
			raw_map_opt = snap_surf_map_args.get("map_opt")
		if raw_map_opt is not None:
			if not isinstance(raw_map_opt, dict):
				raise ValueError(f"stage '{stage.name}' opt.args.snap_surf_map.map_opt must be an object or null")
			raw_map_opt_cfg = dict(raw_map_opt)
			raw_map_name = str(raw_map_opt_cfg.pop("name", raw_map_opt_cfg.pop("kind", "snap_surf_map.map_opt")))
			map_opt_cfg = _parse_opt_settings(
				stage_name=f"{stage.name}.snap_surf_map.map_opt",
				opt_cfg=raw_map_opt_cfg,
				base=opt_cfg.base_eff,
			)
			if map_opt_cfg.kind != "map":
				raise ValueError(f"stage '{stage.name}' opt.args.snap_surf_map.map_opt params must be map params")
			snap_surf_map_opt_stage = _global_map_stage_from_opt_settings(
				name=raw_map_name,
				opt_cfg=map_opt_cfg,
				args=map_opt_cfg.args or {},
			)
			snap_surf_map_opt_stage = replace(
				snap_surf_map_opt_stage,
				args=_resolve_snap_surf_map_fixture_export_args(snap_surf_map_opt_stage.args, out_dir=out_dir),
			)

		snap_surf_map_weight = _need_term("snap_surf_map", stage_eff)
		snap_surf_global_map_mode = snap_surf_map_opt_stage is not None or snap_surf_map_weight > 0.0
		print(
			f"[optimizer] {label}: snap_surf_map_weight={snap_surf_map_weight:.6g} "
			f"snap_surf_map_offset={_snap_global_model_offset_text() if snap_surf_global_map_mode else 'n/a'} "
			f"snap_surf_map_offset_mode={'auto' if snap_surf_global_map_mode else 'n/a'}",
			flush=True,
		)

		_flatten_diagnostics = _truthy(os.environ.get(
			"LASAGNA_FLATTEN_DIAGNOSTICS",
			stage_args.get("flatten_diagnostics", True),
		))
		opt_loss_flatten.configure(
			sdir_eps=float(stage_args.get("flatten_sdir_eps", 1.0e-8)),
			orient_min_det=float(stage_args.get("flatten_orient_min_det", 1.0e-2)),
			order_margin=float(stage_args.get("flatten_order_margin", 0.05)),
			edge_step_global_scale=float(stage_args.get("flatten_edge_step_global_scale", 1.0)),
			diagnostics=_flatten_diagnostics,
		)
		if _need_term("flatten_edge_step", stage_eff) > 0 and not is_cyl_shelling_stage:
			print(
				f"[optimizer] {label}: flatten_edge_step_global_scale="
				f"{float(stage_args.get('flatten_edge_step_global_scale', 1.0)):g}",
				flush=True,
			)
		_compile_flatten = _truthy(os.environ.get(
			"LASAGNA_COMPILE_FLATTEN",
			stage_args.get("compile_flatten", False),
		))
		_compile_flatten_backend = os.environ.get(
			"LASAGNA_COMPILE_FLATTEN_BACKEND",
			stage_args.get("compile_flatten_backend", None),
		)
		_compile_flatten_mode = os.environ.get(
			"LASAGNA_COMPILE_FLATTEN_MODE",
			stage_args.get("compile_flatten_mode", None),
		)
		_compile_flatten_dynamic = _truthy(os.environ.get(
			"LASAGNA_COMPILE_FLATTEN_DYNAMIC",
			stage_args.get("compile_flatten_dynamic", False),
		))
		_compile_flatten_fullgraph = _truthy(os.environ.get(
			"LASAGNA_COMPILE_FLATTEN_FULLGRAPH",
			stage_args.get("compile_flatten_fullgraph", False),
		))
		_compile_flatten_combined = _truthy(os.environ.get(
			"LASAGNA_COMPILE_FLATTEN_COMBINED",
			stage_args.get("compile_flatten_combined", True),
		))
		opt_loss_flatten.configure_compile(
			enabled=_compile_flatten,
			backend=_compile_flatten_backend,
			mode=_compile_flatten_mode,
			dynamic=_compile_flatten_dynamic,
			fullgraph=_compile_flatten_fullgraph,
		)
		if not _flatten_diagnostics and "map_flatten_ms" in opt_cfg.params:
			print(f"[optimizer] {label}: flatten_diagnostics=0", flush=True)
		if _compile_flatten and "map_flatten_ms" in opt_cfg.params:
			_details = []
			if _compile_flatten_backend:
				_details.append(f"backend={_compile_flatten_backend}")
			if _compile_flatten_mode:
				_details.append(f"mode={_compile_flatten_mode}")
			if _compile_flatten_dynamic:
				_details.append("dynamic=1")
			if _compile_flatten_fullgraph:
				_details.append("fullgraph=1")
			if (
				_compile_flatten_combined
				and not _flatten_diagnostics
				and str(getattr(model, "flatten_direction", "inverse")) == "forward"
			):
				_details.append("combined=1")
			_detail_str = " " + " ".join(_details) if _details else ""
			print(f"[optimizer] {label}: compile_flatten=1{_detail_str}", flush=True)
		_flatten_combined_names = (
			"flatten_sdir",
			"flatten_map_step",
			"flatten_avg_offset",
			"flatten_orient",
		)
		_flatten_combined_weights = None
		_flatten_combined_step_loss = "map"
		if (
			_compile_flatten
			and _compile_flatten_combined
			and not _flatten_diagnostics
			and "map_flatten_ms" in opt_cfg.params
			and str(getattr(model, "flatten_direction", "inverse")) == "forward"
		):
			_flatten_map_combined_names = (
				"flatten_sdir",
				"flatten_map_step",
				"flatten_avg_offset",
				"flatten_orient",
			)
			_flatten_edge_combined_names = (
				"flatten_sdir",
				"flatten_edge_step",
				"flatten_avg_offset",
				"flatten_orient",
			)
			_map_step_weight = _need_term("flatten_map_step", stage_eff)
			_edge_step_weight = _need_term("flatten_edge_step", stage_eff)
			if _edge_step_weight != 0.0 and _map_step_weight == 0.0:
				_flatten_combined_names = _flatten_edge_combined_names
				_flatten_combined_step_loss = "edge"
			elif _map_step_weight != 0.0 and _edge_step_weight == 0.0:
				_flatten_combined_names = _flatten_map_combined_names
				_flatten_combined_step_loss = "map"
			_flatten_weight_values = tuple(_need_term(name, stage_eff) for name in _flatten_combined_names)
			if all(weight != 0.0 for weight in _flatten_weight_values) and not (
				_map_step_weight != 0.0 and _edge_step_weight != 0.0
			):
				_flatten_combined_weights = torch.tensor(
					_flatten_weight_values,
					device=next(model.parameters()).device,
					dtype=torch.float32,
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
		if _compile_cyl_normal and not is_cyl_shelling_stage:
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

		if bool(getattr(model, "cyl_shell_mode", False)) and stage.name not in CYLINDER_SEED_INIT_STAGE_ROLES:
			if not bool(getattr(model, "cyl_shell_search_done", False)):
				if not getattr(model, "cyl_shell_completed", None):
					raise RuntimeError(f"{label}: cylinder shell progression has no completed shell")
				_collapse_cylinder_shells_to_last()
				model.cyl_shell_search_done = True

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
		if snap_surf_map_opt_stage is not None:
			stage_needs = stage_needs.merged(Needs(
				mesh_normals=True,
				ext_surfaces=(self_map_mode == "off"),
			))
		if not is_cyl_shelling_stage:
			print(
				f"[optimizer] {label}: forward_needs={stage_needs.summary()} "
				f"{_prefetch_grad_summary(stage_needs)}",
				flush=True,
			)

		def _make_param_groups(opt_settings: OptSettings | None = None) -> tuple[dict[str, list], list[dict]]:
			settings = opt_settings if opt_settings is not None else opt_cfg
			all_params_ = model.opt_params()
			param_groups_: list[dict] = []
			for name in settings.params:
				internal_name = MODEL_INTERNAL_PARAM.get(name, name)
				group = all_params_.get(internal_name, [])
				if name in {"mesh_ms", "map_flatten_ms"}:
					k0 = max(0, int(settings.min_scaledown))
					for pi, p in enumerate(group):
						if pi < k0:
							continue
						param_group = {
							"params": [p],
							"lr": _lr_scalespace(lr=settings.lr, scale_i=pi),
						}
						if name == "map_flatten_ms":
							param_group["_flatten_scale_i"] = pi
						param_groups_.append(param_group)
				elif name == "cyl_params" and bool(getattr(model, "cyl_shell_mode", False)):
					scale_count = 0
					if hasattr(model, "cyl_param_scale_count"):
						scale_count = max(0, int(model.cyl_param_scale_count()))
					k0 = max(0, int(settings.min_scaledown))
					for pi, p in enumerate(group):
						if pi < scale_count:
							if pi < k0:
								continue
							param_groups_.append({"params": [p], "lr": _lr_scalespace(lr=settings.lr, scale_i=pi)})
						else:
							param_groups_.append({"params": [p], "lr": _lr_last(settings.lr)})
				else:
					lr_last = _lr_last(settings.lr)
					for p in group:
						param_groups_.append({"params": [p], "lr": lr_last})
			return all_params_, param_groups_

		def _make_optimizer(
			param_groups_: list[dict],
			opt_settings: OptSettings | None = None,
		) -> torch.optim.Optimizer:
			settings = opt_settings if opt_settings is not None else opt_cfg
			if (
				fused_flatten_adam_clamp
				and flatten_max_update > 0.0
				and settings.params == ["map_flatten_ms"]
			):
				return FlattenClampedAdam(param_groups_, base_step=flatten_max_update)
			return torch.optim.Adam(param_groups_)

		_t = _stage_start(f"{label}.build_optimizer")
		all_params, param_groups = _make_param_groups()
		if not param_groups:
			return data
		opt = _make_optimizer(param_groups)
		_capture_optimizer_target_lrs(opt)
		if flatten_max_update > 0.0 and not is_cyl_shelling_stage:
			print(
				f"[optimizer] {label}: flatten_max_update={flatten_max_update:g} "
				f"(per-scale cap doubles at each coarser level)",
				flush=True,
			)
		if isinstance(opt, FlattenClampedAdam) and not is_cyl_shelling_stage:
			print(
				f"[optimizer] {label}: fused_flatten_adam_clamp=1 "
				f"(Triton CUDA with PyTorch fallback)",
				flush=True,
			)
		if flatten_renorm_interval > 0 and not is_cyl_shelling_stage:
			print(
				f"[optimizer] {label}: flatten_renorm_interval={flatten_renorm_interval}",
				flush=True,
			)
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
				opt = _make_optimizer(param_groups)
				_capture_optimizer_target_lrs(opt)
		_stage_done(f"{label}.winding_offset_autocrop", _t)

		reporter = StageReporter(
			stage_args=stage_args,
			step_width=max(16, len(f"{label} {max(0, opt_cfg.steps)}/{steps_label}") + 2),
		)

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
			if bool(getattr(model, "cyl_shell_mode", False)):
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
				_mesh_conn_pf = (
					model.mesh_conn_prefetch_points(_xyz_lr_pf)
					if needs_.mesh_conn and hasattr(model, "mesh_conn_prefetch_points") else ()
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
				if _mesh_conn_pf and "grad_mag" in _cache_channels:
					for _pf in _mesh_conn_pf:
						_cache.prefetch(_pf, data.origin_fullres, _sp)
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
			_prefetched_channels: set[str] = set()
			with torch.no_grad():
				_loss_prefetch_items: dict[str, torch.Tensor] = {}
				if needs_.mesh_conn:
					optimizer_timing.log_cuda_memory(f"{label}.loss_prefetch.winding_density.begin")
					for _i, _pf in enumerate(opt_loss_winding_density.winding_density_prefetch_grad_mag_batches_for_result(res=res_)):
						optimizer_timing.log_cuda_memory(
							f"{label}.loss_prefetch.winding_density.batch{_i}.ready"
						)
						for _cache in _active_caches:
							_cache_channels = set(_cache.channels)
							if "grad_mag" not in _cache_channels:
								continue
							_sp = data._spacing_for(_cache.channels[0])
							_cache.prefetch(_pf, data.origin_fullres, _sp)
							_prefetched_channels.add("grad_mag")
						optimizer_timing.log_cuda_memory(
							f"{label}.loss_prefetch.winding_density.batch{_i}.prefetched"
						)
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
				if needs_.prefetch_cyl_grad_mask:
					_add_prefetch_items(
						_loss_prefetch_items,
						opt_loss_cyl.cyl_step_push_prefetch_items_for_result(res=res_),
					)
				if needs_.prefetch_ext_offset:
					_add_prefetch_items(
						_loss_prefetch_items,
						opt_loss_winding_density.ext_offset_prefetch_items_for_result(res=res_),
					)
				if needs_.prefetch_snap_surf_map:
					if self_map_mode != "off" and res_.normals is not None:
						for direction in _self_map_directions():
							_add_prefetch_items(
								_loss_prefetch_items,
								_snap_self_runtime_for(direction).snap_loss_prefetch_items(
									model_xyz=res_.xyz_lr,
									model_normals=res_.normals,
									model_valid=torch.isfinite(res_.xyz_lr).all(dim=-1),
									offset=1.0,
									data=res_.data,
									strip_samples=max(2, int(res_.params.subsample_mesh) + 1),
								),
							)
					else:
						records = getattr(res_, "ext_surfaces", None)
						if records and res_.normals is not None:
							ext_xyz, ext_valid, ext_normals, ext_quad_valid, offset = _unpack_ext_surface_record(records[0])
							_add_prefetch_items(
								_loss_prefetch_items,
								_snap_global_runtime_for().snap_loss_prefetch_items(
									model_xyz=res_.xyz_lr,
									model_normals=res_.normals,
									model_valid=torch.isfinite(res_.xyz_lr).all(dim=-1),
									ext_xyz=ext_xyz,
									ext_valid=ext_valid,
									ext_normals=ext_normals,
									ext_quad_valid=ext_quad_valid,
									offset=offset,
									data=res_.data,
									strip_samples=max(2, int(res_.params.subsample_mesh) + 1),
								),
							)
			if not _loss_prefetch_items and not _prefetched_channels:
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
					_prefetched_channels.update(ch for ch in _cache.channels if ch in _loss_prefetch_items)
			for _cache in _active_caches:
				if any(ch in _prefetched_channels for ch in _cache.channels):
					_cache.sync()

		# Initial evaluation
		def _eval_terms(res_, eff_, *, profile_label: str | None = None,
						timing: optimizer_timing.OptTimingWindow | None = None,
						atlas_debug_step: int | None = None,
						report_flatten: bool = True):
			"""Evaluate all loss terms, handling both single and multi-loss returns."""
			total = torch.zeros((), device=next(model.parameters()).device, dtype=torch.float32)
			tv: dict[str, float] = {}
			if stage_uses_cyl_loss:
				opt_loss_cyl.reset_candidate_terms()
			D = res_.xyz_lr.shape[0]
			if _flatten_combined_weights is not None:
				for name in _flatten_combined_names:
					t = terms[name]
					missing = _missing_loss_fields(
						name=name,
						required=t.get("needs", Needs()),
						res_=res_,
					)
					if missing:
						raise RuntimeError(
							f"{profile_label or label}: active combined flatten loss missing "
							f"artifact(s) for '{name}': {', '.join(missing)}"
						)
				loss_label = f"{profile_label or label}.flatten_combined"
				optimizer_timing.log_cuda_memory(f"{loss_label}.before")
				_t_loss = _stage_start(loss_label) if profile_label is not None else None
				_t_loss_wall = time.perf_counter() if timing is not None else None
				combined_parts = opt_loss_flatten.flatten_combined_loss_parts(
					res=res_,
					weights=_flatten_combined_weights,
					step_loss=_flatten_combined_step_loss,
				)
				combined_loss = combined_parts[0]
				optimizer_timing.debug_cuda_sync(loss_label)
				optimizer_timing.log_cuda_memory(f"{loss_label}.after_raw")
				if timing is not None and _t_loss_wall is not None:
					timing.sync()
					timing.add("loss:flatten_combined", time.perf_counter() - _t_loss_wall)
				if _t_loss is not None:
					_stage_done(loss_label, _t_loss)
				total = total + combined_loss
				if combined_loss.device.type == "cpu":
					# CPU scalar reads do not require device-to-host synchronization.
					for i, name in enumerate(_flatten_combined_names):
						lv = combined_parts[i + 1]
						tv[name] = float(lv.detach().cpu())
						tv[f"{name}_contrib"] = float((lv * _flatten_combined_weights[i]).detach().cpu())
				elif report_flatten:
					component_tensor = torch.stack([part.detach() for part in combined_parts[1:]])
					values, contributions = torch.stack((
						component_tensor, component_tensor * _flatten_combined_weights,
					)).cpu().tolist()
					for name, value, contribution in zip(_flatten_combined_names, values, contributions, strict=True):
						tv[name] = value
						tv[f"{name}_contrib"] = contribution
			for name, t in terms.items():
				if _flatten_combined_weights is not None and name in _flatten_combined_names:
					continue
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
				loss_label = f"{profile_label or label}.{name}"
				optimizer_timing.log_cuda_memory(f"{loss_label}.before")
				_t_loss = _stage_start(f"{profile_label}.{name}") if profile_label is not None else None
				_t_loss_wall = time.perf_counter() if timing is not None else None
				if name == "atlas_line":
					write_atlas_debug = (
						atlas_debug_objs
						and atlas_debug_step is not None
						and (int(atlas_debug_step) % atlas_debug_obj_interval) == 0
					)
					result = t["loss"](res=res_, stage_eff=eff_, debug_payload=write_atlas_debug)
					if write_atlas_debug:
						opt_loss_atlas_line.write_debug_objs(
							stage=stage.name,
							step=int(atlas_debug_step),
							interval=1,
						)
				else:
					result = t["loss"](res=res_)
				optimizer_timing.debug_cuda_sync(f"{profile_label}.{name}" if profile_label is not None else name)
				optimizer_timing.log_cuda_memory(f"{loss_label}.after_raw")
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
					if name == "atlas_line":
						tv.update(opt_loss_atlas_line.last_stats())
				else:
					lv, lms, masks = result
					w = _need_term(name, eff_)
					is_flatten_term = name in (
						"flatten_sdir",
						"flatten_map_step",
						"flatten_edge_step",
						"flatten_avg_offset",
						"flatten_orient",
					)
					tv[name] = float(lv.detach().cpu())
					if is_flatten_term:
						tv[f"{name}_contrib"] = float((lv * w).detach().cpu())
					if name == "pred_dt":
						tv.update(opt_loss_pred_dt.flow_gate_last_stats())
					if name == "cyl_outside":
						tv.update(opt_loss_cyl.last_stats())
					if name == "snap_surf_map":
						tv.update(opt_loss_snap_surf.last_stats())
					if name == "atlas_line":
						tv.update(opt_loss_atlas_line.last_stats())
					if _flatten_diagnostics and name in ("flatten_sdir", "flatten_avg_offset", "flatten_orient"):
						tv.update(opt_loss_flatten.last_stats())
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
			role = str(stage.name)
			max_steps = int(opt_cfg.steps)
			default_max_search_shells = max(1, int(getattr(model, "cyl_shell_search_max_shells", 16)))
			max_search_shells = max(1, _cyl_stage_max_search_shells(default_max_search_shells))
			_stage_wstep = _cyl_stage_width_target_step(opt_cfg)
			_prev_stage_wstep = float(getattr(model, "cyl_shell_width_target_step", 0.0))
			_base_wstep = float(
				_stage_wstep
				if _stage_wstep is not None else getattr(model, "cyl_shell_width_target_step", 0.0)
			)
			if hasattr(model, "cyl_shell_width_target_step"):
				model.cyl_shell_width_target_step = _base_wstep
			if _base_wstep > 0.0 and hasattr(model, "cyl_shell_z_step"):
				model.cyl_shell_z_step = _base_wstep
			if _base_wstep > 0.0 and hasattr(model, "cyl_shell_current_height_step"):
				model.cyl_shell_current_height_step = _base_wstep
			if hasattr(model, "prepare_umbilicus_tube_init"):
				model.prepare_umbilicus_tube_init(data)

			def _prefetch_shell_model_points(needs_: fit_model.ModelForwardNeeds) -> None:
				_prefetch_model_points(needs_)

			def _shell_width_count() -> int:
				if hasattr(model, "current_cylinder_shell_xyz"):
					try:
						return int(model.current_cylinder_shell_xyz().shape[1])
					except Exception:
						pass
				offsets = getattr(model, "cyl_shell_w_offsets", None)
				if offsets is not None and hasattr(offsets, "shape") and len(offsets.shape) >= 2:
					return int(offsets.shape[1])
				return 0

			def _shell_dbg_values(res_: fit_model.FitResult3D | None = None) -> dict[str, float]:
				if not hasattr(model, "_shell_width_step_stats"):
					return {}
				_width_stats = (
					opt_loss_cyl.cyl_shell_width_edge_stats(res=res_)
					if res_ is not None else None
				)
				if _width_stats is None:
					_avg = model._shell_width_step_stats()[0]
					_iavg = math.nan
					_ifrac = 0.0
				else:
					_avg = float(_width_stats["valid_avg_vx"])
					_iavg = float(_width_stats["invalid_avg_vx"])
					_ifrac = float(_width_stats["invalid_frac"])
				_havg = (
					model._shell_height_step_stats()[0]
					if hasattr(model, "_shell_height_step_stats") else 0.0
				)
				w_count = max(0, _shell_width_count())
				tgt = float(getattr(model, "cyl_shell_current_width_step", 0.0))
				h_tgt = float(getattr(model, "cyl_shell_z_step", getattr(model, "cyl_shell_current_height_step", tgt)))
				out = {
					"bend_max_deg": float(model._shell_bend_max_degrees())
						if hasattr(model, "_shell_bend_max_degrees") else 0.0,
					"hstep_avg_vx": float(_havg),
					"hstep_tgt_vx": h_tgt,
					"wstep_avg_vx": float(_avg),
					"wstep_invalid_avg_vx": float(_iavg),
					"wstep_invalid_frac": float(_ifrac),
					"wstep_tgt_vx": tgt,
				}
				if w_count > 0:
					out["wcirc_avg_vx"] = float(w_count) * float(_avg)
					out["wcirc_tgt_vx"] = float(w_count) * tgt
				return out

			def _pass_eff_for_role(*, keep_radial_mean: bool = True,
								   eff: dict[str, float] | None = None) -> dict[str, float]:
				pass_eff = dict(stage_eff if eff is None else eff)
				for _conn_term in ("cyl_conn_mesh", "cyl_conn_gt", "cyl_base_mesh", "cyl_base_gt"):
					pass_eff[_conn_term] = 0.0
				if role != "cyl_grow":
					pass_eff["cyl_step_push"] = 0.0
				if not keep_radial_mean:
					pass_eff["cyl_radial_mean"] = 0.0
				return pass_eff

			def _abort_after_shell_error(shell_label: str, err: Exception, *, keep_active: bool) -> None:
				print(
					f"[optimizer] ERROR {shell_label}: {err}; "
					f"stopping remaining stages and outputting optimized shells.",
					flush=True,
				)
				model.cyl_shell_search_done = True
				setattr(model, "cyl_shell_abort", True)
				if not keep_active and hasattr(model, "cyl_shell_active"):
					model.cyl_shell_active = False

			def _actual_width_step_avg(*, fallback: float) -> float:
				if hasattr(model, "_shell_width_step_stats"):
					return max(1.0, float(model._shell_width_step_stats()[0]))
				return max(1.0, float(fallback))

			def _cyl_grow_factor() -> float:
				args = opt_cfg.args or {}
				for key in ("cyl_grow_factor", "grow_factor", "cyl_shell_growth_factor"):
					if key in args:
						return max(1.0, float(args[key]))
				return 1.5

			def _cyl_refine_max_ifrac(refine_opt: OptSettings) -> float | None:
				args = refine_opt.args or {}
				raw = DEFAULT_CYLINDER_REFINE_MAX_IFRAC
				for key in CYLINDER_REFINE_MAX_IFRAC_ARGS:
					if key in args:
						raw = args[key]
						break
				if raw is None:
					return None
				value = float(raw)
				if value < 0.0:
					return None
				return value

			def _run_shell_pass(shell_label: str, pass_eff: dict[str, float], *,
								wstep_start: float, wstep_end: float,
								pass_opt_cfg: OptSettings | None = None,
								pass_steps: int | None = None,
								model_step: float | None = None,
								max_resamples: int | None = None,
								allow_resample: bool = True,
								resample_after_linear_grow: bool = False,
								resample_width_count: int | None = None,
								resample_width_step: float | None = None,
								status_label: str | None = None,
								shell_no: int | None = None,
								suppress_initial_status: bool = False) -> dict[str, object]:
				nonlocal data, all_params, param_groups, opt
				display_label = status_label or shell_label
				pass_settings = pass_opt_cfg if pass_opt_cfg is not None else opt_cfg
				pass_max_steps = max(0, int(max_steps if pass_steps is None else pass_steps))
				if shell_no is not None:
					reporter.ensure_step_width(len("1000000") + 2)
				else:
					reporter.ensure_step_width(len(f"{display_label} {pass_max_steps}/{pass_max_steps}") + 2)
				pass_needs = _needs_for_eff(
					pass_eff,
					pred_dt_flow_gate_cfg_=pred_dt_flow_gate_cfg,
					pred_dt_normal_source_=pred_dt_normal_source,
				)
				if _cyl_outside_enabled(pass_eff):
					_configure_cyl_outside_step(
						sample_factor=_cyl_outside_sample_factor(pass_settings.args or {}),
						model_step=float(model_step if model_step is not None else wstep_start),
					)
				model.cyl_shell_current_width_step = float(wstep_start)
				all_params, param_groups = _make_param_groups(pass_settings)
				if not param_groups:
					raise RuntimeError(f"{shell_label}: no cylinder parameters available to optimize")
				opt = _make_optimizer(param_groups, pass_settings)
				_capture_optimizer_target_lrs(opt)
				resample_count = 0
				resampled_this_pass = False
				step1 = 0

				def _error_result(err: Exception) -> dict[str, object]:
					return {
						"seed_hit": False,
						"metrics": None,
						"resamples": resample_count,
						"resampled": resampled_this_pass,
						"error": err,
						"keep_active": step1 > 0,
					}

				def _resample_shell_width_to_model_step() -> bool:
					nonlocal resample_count, resampled_this_pass
					if not allow_resample:
						return True
					if resample_width_count is None and model_step is None:
						return True
					if max_resamples is not None and resample_count >= max_resamples:
						print(
							f"[optimizer] ERROR {shell_label}: cylinder shell pass hit "
							f"resample cap {max_resamples}; outputting completed shells.",
							flush=True,
						)
						return False
					if resample_width_count is not None:
						if not hasattr(model, "resample_current_cylinder_shell_width_to_count"):
							raise RuntimeError(f"{shell_label}: model cannot resample cylinder shell width to target count")
						model.resample_current_cylinder_shell_width_to_count(
							data,
							int(resample_width_count),
							target_step=resample_width_step,
						)
						_bump_snap_global_mesh_epoch()
					else:
						model_step_ref = max(1.0, float(model_step))
						if not hasattr(model, "resample_current_cylinder_shell_width_to_step"):
							raise RuntimeError(f"{shell_label}: model cannot resample cylinder shell width to model-step")
						model.resample_current_cylinder_shell_width_to_step(data, model_step_ref)
						_bump_snap_global_mesh_epoch()
					resample_count += 1
					resampled_this_pass = True
					return True

				_t = _stage_start(f"{shell_label}.initial_eval")
				_prefetch_shell_model_points(pass_needs)
				with torch.no_grad():
					res0 = model(data, needs=pass_needs)
					_prefetch_loss_points_for_result(res0, pass_needs)
					loss0, term_vals0, display_loss0 = _eval_terms(
						res0, pass_eff, profile_label=f"{shell_label}.initial_eval.loss")
				if bool(getattr(model, "flatten_enabled", False)):
					term_vals0 = {
						**term_vals0,
						**opt_loss_flatten.current_grid_step_stats(res0),
					}
				term_vals0 = {k: round(v, 4) for k, v in term_vals0.items()}
				if not suppress_initial_status:
					reporter.print_status(
						step_label=(
							"0" if shell_no is not None
							else f"{display_label} 0/{pass_max_steps}"
						),
						loss_val=float(display_loss0) if display_loss0 is not None else loss0.item(),
						tv=term_vals0,
						pv=_shell_dbg_values(res0),
						force_header=True,
						shell_no=shell_no,
					)
				if snap_int > 0:
					snapshot_fn(stage=shell_label.replace(".", "_"), step=0,
								loss=float(loss0.detach().cpu()), data=data, res=res0)
				_stage_done(f"{shell_label}.initial_eval", _t)

				loss = loss0
				display_loss = display_loss0
				res = res0
				_t_wall_start = time.perf_counter()
				_t_steps_acc = 0
				_opt_timing = (
					optimizer_timing.OptTimingWindow(interval=opt_timing_interval, sync_cuda=opt_timing_sync)
					if opt_timing_enabled else None
				)
				step = 0
				while step < pass_max_steps:
					_t_iter = time.perf_counter()
					if pass_max_steps > 0:
						_alpha = float(step + 1) / float(pass_max_steps)
						model.cyl_shell_current_width_step = (
							float(wstep_start) + _alpha * (float(wstep_end) - float(wstep_start))
						)

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
					_apply_optimizer_lr_warmup(opt, step1=step + 1, warmup_steps=lr_warmup_steps)
					opt.step()
					_bump_snap_global_mesh_epoch()
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
					_stage_progress = step1 / pass_max_steps if pass_max_steps > 0 else 1.0
					_overall_progress = (
						(si + _stage_progress) / _num_stages if _num_stages > 0 else 1.0
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
					if shell_no is not None:
						_status_due = status_interval > 0 and step1 > 0 and (step1 % status_interval) == 0
					else:
						_status_due = (
							step == 0 or
							step1 == pass_max_steps or
							(status_interval > 0 and (step1 % status_interval) == 0)
						)
					if _status_due:
						if bool(getattr(model, "flatten_enabled", False)):
							term_vals = {
								**term_vals,
								**opt_loss_flatten.current_grid_step_stats(res),
							}
						term_vals = {k: round(v, 4) for k, v in term_vals.items()}
						_t_wall_now = time.perf_counter()
						_t_wall_elapsed = _t_wall_now - _t_wall_start
						_its = _t_steps_acc / _t_wall_elapsed if _t_wall_elapsed > 0 else None
						reporter.print_status(
							step_label=(
								str(step1) if shell_no is not None
								else f"{display_label} {step1}/{pass_max_steps}"
							),
							loss_val=float(display_loss) if display_loss is not None else loss.item(),
							tv=term_vals,
							pv=_shell_dbg_values(res),
							its=_its,
							shell_no=shell_no,
						)
						_t_steps_acc = 0
						_t_wall_start = _t_wall_now
					if _opt_timing is not None:
						_opt_timing.add("status", time.perf_counter() - _t_part)
						_opt_timing.add("total", time.perf_counter() - _t_iter)
						_opt_timing.finish_iter(
							label=shell_label,
							step1=step1,
							max_steps=pass_max_steps,
						)
					step += 1
				if (
					resample_after_linear_grow
					and step1 > 0
					and not resampled_this_pass
					and not _resample_shell_width_to_model_step()
				):
					return _error_result(RuntimeError(f"{shell_label}: failed to resample cylinder shell width after grow"))
				if snap_int > 0 and (step1 % snap_int) == 0:
					snapshot_fn(stage=shell_label.replace(".", "_"), step=step1,
								loss=float(loss.detach().cpu()), data=data, res=res)

				snapshot_fn(stage=shell_label.replace(".", "_"), step=step1,
							loss=float(loss.detach().cpu()), data=data, res=res)
				debug_values = _shell_dbg_values(res)
				return {
					"seed_hit": False,
					"metrics": debug_values,
					"resamples": resample_count,
					"resampled": resampled_this_pass,
					"error": None,
					"keep_active": False,
					"debug": debug_values,
				}

			def _grow_refine_eff(refine_opt: OptSettings) -> dict[str, float]:
				refine_eff = _stage_eff_for_opt(is_cyl_stage_=True, opt_cfg_=refine_opt)
				refine_eff = _pass_eff_for_role(eff=refine_eff)
				refine_eff["cyl_step_push"] = 0.0
				return refine_eff

			def _emit_cylinder_shell_callback(stage_label: str) -> None:
				if cylinder_shell_callback is None:
					return
				shells = getattr(model, "cyl_shell_completed", None)
				if not shells:
					return
				shell_index = len(shells) - 1
				shell_xyz = shells[-1].detach().clone()
				cylinder_shell_callback(
					shell_index=shell_index,
					shell_xyz=shell_xyz,
					stage_label=stage_label,
					data=data,
				)

			def _run_grow_refine_pass(
				*,
				refine_label: str,
				refine_opt: OptSettings,
				shell_no: int | None,
			) -> dict[str, object]:
				if int(refine_opt.steps) <= 0:
					return {"seed_hit": False, "metrics": None, "resamples": 0, "resampled": False, "error": None}
				if not getattr(model, "cyl_shell_completed", None):
					raise RuntimeError(f"{refine_label}: cyl_grow_refine requires a completed grow shell")
				if not hasattr(model, "begin_cylinder_shell_refine"):
					raise RuntimeError(f"{refine_label}: model does not support cylinder shell grow-refine")
				model.begin_cylinder_shell_refine(data)
				refine_wstep = float(getattr(model, "cyl_shell_width_target_step", 0.0))
				if refine_wstep <= 0.0:
					refine_wstep = _actual_width_step_avg(fallback=_base_wstep)
				if hasattr(model, "cyl_shell_width_target_step"):
					model.cyl_shell_width_target_step = float(refine_wstep)
				result_ = _run_shell_pass(
					refine_label,
					_grow_refine_eff(refine_opt),
					wstep_start=refine_wstep,
					wstep_end=refine_wstep,
					pass_opt_cfg=refine_opt,
					pass_steps=int(refine_opt.steps),
					model_step=refine_wstep,
					allow_resample=False,
					shell_no=shell_no,
					suppress_initial_status=False,
				)
				if result_.get("error") is None and hasattr(model, "complete_current_cylinder_shell"):
					model.complete_current_cylinder_shell(data)
					_bump_snap_global_mesh_epoch()
					_emit_cylinder_shell_callback(refine_label)
					max_ifrac = _cyl_refine_max_ifrac(refine_opt)
					debug_values = result_.get("debug")
					if max_ifrac is not None and isinstance(debug_values, dict):
						ifrac = float(debug_values.get("wstep_invalid_frac", 0.0))
						if math.isfinite(ifrac) and ifrac > float(max_ifrac):
							print(
								f"[optimizer] {refine_label}: stopping cylinder shell growth because "
								f"ifrac={ifrac:.4f} exceeds cyl_refine_max_ifrac={float(max_ifrac):.4f}; "
								f"too many wrapped width edges have an endpoint outside the grad_mag mask.",
								flush=True,
							)
							model.cyl_shell_search_done = True
							setattr(model, "cyl_shell_abort", True)
				if hasattr(model, "cyl_shell_width_target_step"):
					model.cyl_shell_width_target_step = float(_base_wstep)
				return result_

			if role == "cyl_init":
				if bool(getattr(model, "cyl_shell_search_done", False)):
					_stage_done(f"{label}.total", _t_stage_total)
					return data
				if getattr(model, "cyl_shell_completed", None) and hasattr(model, "begin_cylinder_shell_refine"):
					model.begin_cylinder_shell_refine(data)
				elif hasattr(model, "begin_cylinder_shell"):
					model.begin_cylinder_shell(0, data, direction=1)
				if _cyl_outside_enabled(stage_eff):
					_clear_cyl_outside_field()
				_run_shell_pass(f"{label}.cyl_init", _pass_eff_for_role(),
								wstep_start=_base_wstep, wstep_end=_base_wstep,
								shell_no=1)
				if hasattr(model, "complete_current_cylinder_shell"):
					model.complete_current_cylinder_shell(data)
					_bump_snap_global_mesh_epoch()
					_emit_cylinder_shell_callback(f"{label}.cyl_init")
				if _cyl_init_only:
					_stage_done(f"{label}.total", _t_stage_total)
					return data
				_stage_done(f"{label}.total", _t_stage_total)
				return data

			if role == "cyl_grow":
				if bool(getattr(model, "cyl_shell_search_done", False)):
					_stage_done(f"{label}.total", _t_stage_total)
					return data
				if not getattr(model, "cyl_shell_completed", None):
					raise RuntimeError(f"{label}: cyl_grow requires cyl_init to complete a first shell")
				direction = _cyl_stage_grow_direction(opt_cfg)
				model.cyl_shell_search_direction = int(direction)
				grow_refine_opt = (
					stages[si + 1].global_opt
					if si + 1 < len(stages) and stages[si + 1].name == "cyl_grow_refine"
					else None
				)
				grow_factor = _cyl_grow_factor()
				if hasattr(model, "cyl_shell_growth_factor"):
					model.cyl_shell_growth_factor = float(grow_factor)
				reference_width_raw = getattr(model, "cyl_grow_reference_width_count", None)
				reference_circ_raw = getattr(model, "cyl_grow_reference_circumference", None)
				if reference_width_raw is None or reference_circ_raw is None:
					raise RuntimeError(
						f"{label}: cyl_grow requires stored grow reference fields "
						"cyl_grow_reference_width_count and cyl_grow_reference_circumference"
					)
				reference_width_count = int(reference_width_raw)
				reference_circumference = float(reference_circ_raw)
				if reference_width_count < 3 or not math.isfinite(reference_circumference) or reference_circumference <= 0.0:
					raise RuntimeError(
						f"{label}: invalid cylinder grow reference "
						f"reference_w={reference_width_count} reference_circ={reference_circumference}"
					)
				reference_step = reference_circumference / float(reference_width_count)
				print(
					f"[optimizer] {label}: cylinder grow reference "
					f"reference_w={reference_width_count} "
					f"reference_step={reference_step:.6g} "
					f"reference_circ={reference_circumference:.6g}",
					flush=True,
				)
				result: dict[str, object] = {"seed_hit": False, "metrics": None, "resamples": 0, "resampled": False, "error": None}
				while True:
					shell_i = len(getattr(model, "cyl_shell_completed", []))
					if shell_i >= max_search_shells:
						print(
							f"[optimizer] {label}: cylinder shell growth reached "
							f"shell cap {max_search_shells}; outputting completed shells.",
							flush=True,
						)
						break
					prev_width_target = cylinder_grow_width_target(
						reference_width_count=reference_width_count,
						reference_circumference=reference_circumference,
						shell_index=shell_i - 1,
						grow_factor=grow_factor,
						direction=direction,
					)
					grow_width_target = cylinder_grow_width_target(
						reference_width_count=reference_width_count,
						reference_circumference=reference_circumference,
						shell_index=shell_i,
						grow_factor=grow_factor,
						direction=direction,
					)
					if hasattr(model, "begin_cylinder_shell"):
						direction_label = "inward" if direction < 0 else "outward"
						print(
							f"[optimizer] {label}: adding cylinder shell "
							f"{shell_i + 1}/{max_search_shells} "
							f"direction={direction_label} "
							f"wstep_start={prev_width_target.width_step:.6g} "
							f"wstep_end={grow_width_target.width_step:.6g} "
							f"target_w={grow_width_target.width_count} "
							f"target_circ={grow_width_target.circumference:.6g} "
							f"grow_factor={grow_factor:.6g}",
							flush=True,
						)
						grow_shell_label = f"{label}.cyl_grow_shell{shell_i + 1}"
						grow_pass_eff = _pass_eff_for_role()
						_build_cyl_outside_from_completed(
							eff_=grow_pass_eff,
							stage_args_=stage_args,
							model_step=_base_wstep,
							label_=grow_shell_label,
							direction=direction,
						)
						model.begin_cylinder_shell(shell_i, data, direction=direction)
					else:
						grow_shell_label = f"{label}.cyl_grow_shell{shell_i + 1}"
						grow_pass_eff = _pass_eff_for_role()
					result = _run_shell_pass(
						grow_shell_label,
						grow_pass_eff,
						wstep_start=prev_width_target.width_step,
						wstep_end=grow_width_target.width_step,
						model_step=_base_wstep,
						max_resamples=1,
						allow_resample=True,
						resample_after_linear_grow=True,
						resample_width_count=grow_width_target.width_count,
						resample_width_step=grow_width_target.width_step,
						shell_no=shell_i + 1,
						suppress_initial_status=False,
					)
					if result.get("error") is not None:
						_abort_after_shell_error(
							f"{label}.cyl_grow_shell{shell_i + 1}",
							result["error"],
							keep_active=bool(result.get("keep_active", False)),
						)
						break
					if hasattr(model, "complete_current_cylinder_shell"):
						model.complete_current_cylinder_shell(data)
						_bump_snap_global_mesh_epoch()
						_emit_cylinder_shell_callback(f"{label}.cyl_grow_shell{shell_i + 1}")
					if grow_refine_opt is not None:
						result = _run_grow_refine_pass(
							refine_label=f"{label}.cyl_grow_refine_shell{shell_i + 1}",
							refine_opt=grow_refine_opt,
							shell_no=shell_i + 1,
						)
						if result.get("error") is not None:
							_abort_after_shell_error(
								f"{label}.cyl_grow_refine_shell{shell_i + 1}",
								result["error"],
								keep_active=bool(result.get("keep_active", False)),
							)
							break
						if bool(getattr(model, "cyl_shell_abort", False)) or bool(getattr(model, "cyl_shell_search_done", False)):
							break
				if bool(result.get("error") is not None):
					_stage_done(f"{label}.total", _t_stage_total)
					return data
				_collapse_cylinder_shells_to_last()
				model.cyl_shell_search_done = True
				_stage_done(f"{label}.total", _t_stage_total)
				return data

			if role == "cyl_grow_refine":
				_stage_done(f"{label}.total", _t_stage_total)
				return data

			if not getattr(model, "cyl_shell_completed", None):
				raise RuntimeError(f"{label}: cylinder shell stage requires a completed progression shell")
			stage_model_step = _base_wstep
			if stage_model_step <= 0.0:
				stage_model_step = float(getattr(model, "cyl_shell_width_target_step", 1.0))
			prev_model_step = _prev_stage_wstep if _prev_stage_wstep > 0.0 else float(stage_model_step)
			if hasattr(model, "cyl_shell_width_target_step"):
				model.cyl_shell_width_target_step = float(stage_model_step)
			if not hasattr(model, "begin_cylinder_shell_refine"):
				raise RuntimeError(f"{label}: model does not support cylinder shell stage")
			model.begin_cylinder_shell_refine(data)
			if abs(float(stage_model_step) - float(prev_model_step)) > 1.0e-6:
				if hasattr(model, "resample_current_cylinder_shell_height_to_step"):
					model.resample_current_cylinder_shell_height_to_step(data, float(stage_model_step))
					_bump_snap_global_mesh_epoch()
				if not hasattr(model, "resample_current_cylinder_shell_width_to_step"):
					raise RuntimeError(f"{label}: model cannot resample cylinder shell width to model-step")
				model.resample_current_cylinder_shell_width_to_step(data, float(stage_model_step))
				_bump_snap_global_mesh_epoch()
			shell_stage_eff = dict(stage_eff)
			shell_stage_eff["cyl_radial_mean"] = 0.0
			shell_stage_eff["cyl_step_push"] = 0.0
			_run_shell_pass(
				f"{label}.{stage.name}",
				shell_stage_eff,
				wstep_start=stage_model_step,
				wstep_end=stage_model_step,
				model_step=stage_model_step,
				allow_resample=False,
			)
			if hasattr(model, "complete_current_cylinder_shell"):
				model.complete_current_cylinder_shell(data)
				_bump_snap_global_mesh_epoch()
				_emit_cylinder_shell_callback(f"{label}.{stage.name}")
			_collapse_cylinder_shells_to_last()
			_stage_done(f"{label}.total", _t_stage_total)
			return data

		# Initial prefetch for streaming mode
		if _active_caches:
			_t = _stage_start(f"{label}.initial_prefetch")
			optimizer_timing.log_cuda_memory(f"{label}.initial_prefetch.before")
			_prefetch_model_points(stage_needs)
			optimizer_timing.log_cuda_memory(f"{label}.initial_prefetch.after")
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
			optimizer_timing.log_cuda_memory(f"{label}.initial_eval.model_forward.before")
			res0 = model(data, needs=stage_needs)
			optimizer_timing.debug_cuda_sync(f"{label}.initial_eval.model_forward")
			optimizer_timing.log_cuda_memory(f"{label}.initial_eval.model_forward.after")
			_stage_done(f"{label}.initial_eval.model_forward", _t_forward)
			_t_loss_prefetch = _stage_start(f"{label}.initial_eval.loss_prefetch")
			optimizer_timing.log_cuda_memory(f"{label}.initial_eval.loss_prefetch.before")
			_prefetch_loss_points_for_result(res0, stage_needs)
			optimizer_timing.debug_cuda_sync(f"{label}.initial_eval.loss_prefetch")
			optimizer_timing.log_cuda_memory(f"{label}.initial_eval.loss_prefetch.after")
			_stage_done(f"{label}.initial_eval.loss_prefetch", _t_loss_prefetch)
			_t_terms = _stage_start(f"{label}.initial_eval.loss_terms")
			optimizer_timing.log_cuda_memory(f"{label}.initial_eval.loss_terms.before")
			opt_loss_snap_surf.set_debug_step(0, label=f"{label}_initial")
			loss0, term_vals0, display_loss0 = _eval_terms(
				res0, stage_eff, profile_label=f"{label}.initial_eval.loss", atlas_debug_step=0)
			optimizer_timing.log_cuda_memory(f"{label}.initial_eval.loss_terms.after")
			_stage_done(f"{label}.initial_eval.loss_terms", _t_terms)
			if snap_surf_map_opt_stage is not None:
				map_initial_stage = snap_surf_map_global.GlobalMapStageConfig(
					name=f"{snap_surf_map_opt_stage.name or 'map_opt'}_initial",
					steps=0,
					lr=snap_surf_map_opt_stage.lr,
					params=snap_surf_map_opt_stage.params,
					min_scaledown=snap_surf_map_opt_stage.min_scaledown,
					w_fac=snap_surf_map_opt_stage.w_fac,
					args={**dict(snap_surf_map_opt_stage.args), "disable_z_lift": True},
				)
				map_initial_stats = _run_snap_global_map_stage(
					stage=map_initial_stage,
					res=res0,
					stage_args=stage_args,
					persistent_optimizer=True,
				)
				term_vals0.update(_compact_snap_global_map_stats(map_initial_stats))
			_t_prune = _stage_start(f"{label}.initial_eval.cylinder_prune")
			if _prune_cylinder_candidates_after_initial_eval():
				all_params, param_groups = _make_param_groups()
				if not param_groups:
					return data
				opt = torch.optim.Adam(param_groups)
				_capture_optimizer_target_lrs(opt)
			_stage_done(f"{label}.initial_eval.cylinder_prune", _t_prune)
			_t_params = _stage_start(f"{label}.initial_eval.param_values")
			param_vals0: dict[str, float] = {}
			for k, vs in all_params.items():
				if len(vs) == 1 and vs[0].numel() == 1:
					param_vals0[k] = float(vs[0].detach().cpu())
			_stage_done(f"{label}.initial_eval.param_values", _t_params)
			_t_status = _stage_start(f"{label}.initial_eval.status_print")
			if bool(getattr(model, "flatten_enabled", False)):
				term_vals0 = {
					**term_vals0,
					**opt_loss_flatten.current_grid_step_stats(res0),
				}
			term_vals0 = {k: round(v, 4) for k, v in term_vals0.items()}
			param_vals0 = {k: round(v, 4) for k, v in param_vals0.items()}
			reporter.print_status(
				step_label=f"{label} 0/{steps_label}",
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
		if snap_int > 0:
			_t = _stage_start(f"{label}.initial_snapshot")
			snapshot_fn(stage=label, step=0, loss=float(loss0.detach().cpu()), data=data, res=res0)
			_stage_done(f"{label}.initial_snapshot", _t)

		max_steps = opt_cfg.steps
		final_step = 0
		auto_loss_history = [float(loss0.detach().cpu())] if opt_cfg.steps_auto and lr_warmup_steps <= 0 else []
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
			and optimizer_timing.flow_timing_enabled(pred_dt_flow_gate_cfg)
		):
			_flow_timing = optimizer_timing.FlowTimingWindow(interval=100)
		_opt_timing = (
			optimizer_timing.OptTimingWindow(interval=opt_timing_interval, sync_cuda=opt_timing_sync)
			if opt_timing_enabled else None
		)
		if max_steps > 0:
			# The first iteration immediately recomputes both values. Keeping the
			# no-grad initial FitResult alive retains its full integrated UV grid.
			loss = None
			res = None
			del loss0, res0

		for step in range(max_steps):
			_t_iter = time.perf_counter()
			auto_stop = False
			status_due = (
				step == 0 or step + 1 == max_steps
				or (status_interval > 0 and (step + 1) % status_interval == 0)
			)
			optimizer_timing.log_cuda_memory(f"{label}.{step + 1}.iter.begin")
			# Sync: wait for chunks loaded by last prefetch
			_t_io = time.perf_counter()
			if _active_caches:
				for _cache in _active_caches:
					_cache.sync()
			optimizer_timing.log_cuda_memory(f"{label}.{step + 1}.cache_sync.after")
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
			optimizer_timing.log_cuda_memory(f"{label}.{step + 1}.model_forward.before")
			res = model(data, needs=stage_needs)
			optimizer_timing.debug_cuda_sync(f"{label}.{step + 1}.model_forward")
			optimizer_timing.log_cuda_memory(f"{label}.{step + 1}.model_forward.after")
			if _flow_timing is not None:
				optimizer_timing.synchronize_cuda()
				_flow_timing.add("model_forward", time.perf_counter() - _t_forward)
			if _opt_timing is not None:
				_opt_timing.sync()
				_opt_timing.add("model_forward", time.perf_counter() - _t_forward)

			_t_io = time.perf_counter()
			optimizer_timing.log_cuda_memory(f"{label}.{step + 1}.loss_prefetch.before")
			_prefetch_loss_points_for_result(res, stage_needs)
			optimizer_timing.debug_cuda_sync(f"{label}.{step + 1}.loss_prefetch")
			optimizer_timing.log_cuda_memory(f"{label}.{step + 1}.loss_prefetch.after")
			if _flow_timing is not None:
				_flow_timing.add("io_prefetch", time.perf_counter() - _t_io)
			if _opt_timing is not None:
				_opt_timing.sync()
				_opt_timing.add("loss_prefetch", time.perf_counter() - _t_io)

			_t_loss_eval = time.perf_counter()
			opt_loss_snap_surf.set_debug_step(step + 1, label=label)
			optimizer_timing.log_cuda_memory(f"{label}.{step + 1}.loss_eval.before")
			# Automatic stopping can request a final report on any iteration.
			loss, term_vals, display_loss = _eval_terms(
				res, stage_eff, timing=_opt_timing, atlas_debug_step=step + 1,
				report_flatten=status_due or opt_cfg.steps_auto)
			optimizer_timing.log_cuda_memory(f"{label}.{step + 1}.loss_eval.after")
			if _flow_timing is not None:
				optimizer_timing.synchronize_cuda()
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
			optimizer_timing.log_cuda_memory(f"{label}.{step + 1}.zero_grad.before")
			opt.zero_grad(set_to_none=True)
			optimizer_timing.log_cuda_memory(f"{label}.{step + 1}.zero_grad.after")
			if _opt_timing is not None:
				_opt_timing.add("zero_grad", time.perf_counter() - _t_part)
			_t_part = time.perf_counter()
			optimizer_timing.log_cuda_memory(f"{label}.{step + 1}.backward.before")
			loss.backward()
			optimizer_timing.log_cuda_memory(f"{label}.{step + 1}.backward.after")
			if _opt_timing is not None:
				_opt_timing.sync()
				_opt_timing.add("backward", time.perf_counter() - _t_part)
			_t_part = time.perf_counter()
			_flatten_update_before = None
			_flatten_update_params = all_params.get("flatten_map_ms", [])
			if (
				flatten_max_update > 0.0
				and _flatten_update_params
				and not isinstance(opt, FlattenClampedAdam)
			):
				_flatten_update_before = [p.detach().clone() for p in _flatten_update_params]
			_apply_optimizer_lr_warmup(opt, step1=step + 1, warmup_steps=lr_warmup_steps)
			optimizer_timing.log_cuda_memory(f"{label}.{step + 1}.opt_step.before")
			opt.step()
			optimizer_timing.log_cuda_memory(f"{label}.{step + 1}.opt_step.after")
			_bump_snap_global_mesh_epoch()
			if _flatten_update_before is not None:
				_clamp_flatten_map_ms_update(
					_flatten_update_params,
					_flatten_update_before,
					base_step=flatten_max_update,
				)
			step1 = step + 1
			if flatten_renorm_interval > 0 and (step1 % flatten_renorm_interval) == 0:
				_renorm_stats = _renorm_forward_flatten_map_step(model)
				if _renorm_stats:
					_clear_optimizer_state_for_params(opt, _flatten_update_params)
					if "flatten_grid_step_avg" in _renorm_stats:
						term_vals["flatten_grid_step_avg"] = _renorm_stats["flatten_grid_step_avg"]
					print(
						f"[optimizer] {label}: flatten renorm step={step1} "
						f"avg_step={_renorm_stats['flatten_renorm_step_before']:.6g} "
						f"target={_renorm_stats['flatten_renorm_step_after']:.6g} "
						f"uv_scale={_renorm_stats['flatten_renorm_scale']:.6g}",
						flush=True,
					)
			if _opt_timing is not None:
				_opt_timing.sync()
				_opt_timing.add("optimizer_step", time.perf_counter() - _t_part)
			_t_part = time.perf_counter()
			model.update_conn_offsets()
			model.update_ext_conn_offsets()
			if snap_surf_map_opt_stage is not None:
				with torch.no_grad():
					res_map_after = model(data, needs=_map_forward_needs)
				map_args = dict(snap_surf_map_opt_stage.args)
				map_args["startup_timing"] = False
				map_args["disable_z_lift"] = True
				if step + 1 != max_steps:
					map_args.pop("debug_obj_dir", None)
				map_stage_for_step = snap_surf_map_global.GlobalMapStageConfig(
					name=snap_surf_map_opt_stage.name,
					steps=snap_surf_map_opt_stage.steps,
					lr=snap_surf_map_opt_stage.lr,
					params=snap_surf_map_opt_stage.params,
					min_scaledown=snap_surf_map_opt_stage.min_scaledown,
					w_fac=snap_surf_map_opt_stage.w_fac,
					args=map_args,
				)
				map_stats = _run_snap_global_map_stage(
					stage=map_stage_for_step,
					res=res_map_after,
					stage_args=stage_args,
					persistent_optimizer=True,
				)
				opt_loss_snap_surf.update_last_stats(map_stats)
				term_vals.update(_compact_snap_global_map_stats(map_stats))
			if _flow_timing is not None:
				optimizer_timing.synchronize_cuda()
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
			final_step = step1
			_stage_progress = step1 / max_steps if max_steps > 0 else 1.0
			_overall_progress = (
				(si + _stage_progress) / _num_stages if _num_stages > 0 else 1.0
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

			if opt_cfg.steps_auto and step1 > lr_warmup_steps:
				auto_step = step1 - lr_warmup_steps
				auto_loss_history.append(float(loss.detach().cpu()))
				if auto_step >= auto_min and _auto_steps_should_stop(
					auto_loss_history,
					window=auto_window,
					rel_threshold=auto_rel_threshold,
				):
					auto_stop = True

			_t_part = time.perf_counter()
			if status_due or auto_stop:
				if bool(getattr(model, "flatten_enabled", False)):
					_renorm_grid_step = term_vals.get("flatten_grid_step_avg")
					term_vals = {
						**term_vals,
						**opt_loss_flatten.current_grid_step_stats(res),
					}
					if _renorm_grid_step is not None:
						term_vals["flatten_grid_step_avg"] = _renorm_grid_step
				param_vals: dict[str, float] = {}
				for k, vs in all_params.items():
					if len(vs) == 1 and vs[0].numel() == 1:
						param_vals[k] = float(vs[0].detach().cpu())
				term_vals = {k: round(v, 4) for k, v in term_vals.items()}
				param_vals = {k: round(v, 4) for k, v in param_vals.items()}
				_t_wall_now = time.perf_counter()
				_t_wall_elapsed = _t_wall_now - _t_wall_start
				_its = _t_steps_acc / _t_wall_elapsed if _t_wall_elapsed > 0 else None
				reporter.print_status(step_label=f"{label} {step1}/{steps_label}",
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
			if auto_stop:
				rel_improvement = _auto_steps_relative_improvement(auto_loss_history, window=auto_window)
				print(
					f"[optimizer] {label}: auto steps stopped at {step1}/{max_steps} "
					f"rel_improvement_{auto_window}={rel_improvement:.6g} "
					f"< {auto_rel_threshold:g}",
					flush=True,
				)
				break

		_t = _stage_start(f"{label}.final_snapshot")
		snapshot_fn(stage=label, step=final_step, loss=float(loss.detach().cpu()), data=data, res=res)
		_stage_done(f"{label}.final_snapshot", _t)
		_stage_done(f"{label}.total", _t_stage_total)
		return data

	snap_int = int(snapshot_interval)
	if snap_int < 0:
		snap_int = 0

	_total_steps = _scheduled_total_steps()
	_done_steps = [0]
	_num_stages = len(stages)
	_cyl_init_only = (
		bool(getattr(model, "cyl_shell_mode", False))
		and any(stage.name == "cyl_init" for stage in stages if stage.global_opt is not None)
		and not any(stage.name == "cyl_grow" for stage in stages if stage.global_opt is not None)
	)

	# Debug: show corr status
	_corr_terms = ("corr",)
	has_corr_pts = data.corr_points is not None and data.corr_points.points_xyz_winda.shape[0] > 0
	corr_weights = {t: [(_need_term(t, s.global_opt.eff), s.name) for s in stages if s.global_opt is not None and s.global_opt.steps > 0]
					for t in _corr_terms}
	active_corr = {t: ws for t, ws in corr_weights.items() if any(w > 0 for w, _ in ws)}
	print(f"[optimizer] corr_points={has_corr_pts} active_corr_terms={list(active_corr.keys())}", flush=True)
	if has_corr_pts:
		cp = data.corr_points
		n = cp.points_xyz_winda.shape[0]
		print(f"[optimizer] {n} corr points", flush=True)
		if not active_corr:
			print(f"[optimizer] WARNING: corr points loaded but no corr weight > 0 in any stage!", flush=True)

	def _fuse_expand_z_append(
		temp_model: fit_model.Model3D,
		*,
		old_snap_surf_map_state: dict | None,
		temp_snap_surf_map_state: dict | None,
		add_depth: int,
	) -> None:
		with torch.no_grad():
			old_depth = int(model.depth)
			old_flat = fit_model.Model3D._integrate_pyramid_3d(model.mesh_ms, pyramid_d=model.pyramid_d).detach()
			new_flat = fit_model.Model3D._integrate_pyramid_3d(temp_model.mesh_ms, pyramid_d=temp_model.pyramid_d).detach()
			if tuple(old_flat.shape[2:]) != tuple(new_flat.shape[2:]):
				raise RuntimeError(
					f"expand-z fuse requires matching H/W, got old={tuple(old_flat.shape)} new={tuple(new_flat.shape)}"
				)
			flat = torch.cat([old_flat, new_flat], dim=1).contiguous()
			amp = torch.cat([model.amp.detach(), temp_model.amp.detach()], dim=0).contiguous()
			bias = torch.cat([model.bias.detach(), temp_model.bias.detach()], dim=0).contiguous()
			depth_windings = tuple(model.params.depth_windings) + tuple(temp_model.params.depth_windings)
			if len(depth_windings) != int(flat.shape[1]):
				raise RuntimeError(
					f"expand-z fuse depth_windings length {len(depth_windings)} "
					f"must match fused depth {int(flat.shape[1])}"
				)
		n_scales = len(model.mesh_ms)
		model.depth = int(flat.shape[1])
		model.pyramid_d = bool(model.params.pyramid_d) and model.depth > 1
		model.params = replace(
			model.params,
			pyramid_d=model.pyramid_d,
			depth_windings=depth_windings,
		)
		model.mesh_ms = fit_model.Model3D._construct_pyramid_from_flat_3d(
			flat,
			n_scales,
			pyramid_d=model.pyramid_d,
		)
		model.conn_offsets = torch.zeros(
			4,
			model.depth,
			model.mesh_h,
			model.mesh_w,
			device=flat.device,
			dtype=torch.float32,
		)
		model.amp = torch.nn.Parameter(amp.to(device=flat.device, dtype=torch.float32))
		model.bias = torch.nn.Parameter(bias.to(device=flat.device, dtype=torch.float32))
		_fuse_expand_z_snap_surf_maps_append(
			old_state=old_snap_surf_map_state,
			temp_state=temp_snap_surf_map_state,
			old_depth=old_depth,
			add_depth=int(add_depth),
		)
		print(
			f"[optimizer] expand-z fuse append: depth={model.depth} "
			f"depth_windings={list(model.params.depth_windings)}",
			flush=True,
		)

	def _run_expand_z_stage(*, si: int, stage: Stage, data: fit_data.FitData3D) -> fit_data.FitData3D:
		if init_grow is None:
			raise ValueError("expand-z requires args.init-grow")
		order = tuple(str(v).strip().lower() for v in init_grow.get("order", ("up",)))
		if order != ("up",):
			raise NotImplementedError("expand-z currently implements init-grow order ['up'] only")
		target_depth = int(init_grow.get("target_depth", model.depth))
		step_depth = max(1, int(init_grow.get("step", 1)))
		if target_depth <= model.depth:
			print(f"[optimizer] expand-z: target_depth={target_depth} already reached", flush=True)
			return data
		while model.depth < target_depth:
			add_depth = min(step_depth, target_depth - model.depth)
			if add_depth != 1 and self_map_mode != "off":
				raise NotImplementedError("expand-z self-map grow currently requires init-grow.step=1")
			with torch.no_grad():
				ref_res = model(data, needs=fit_model.ModelForwardNeeds(mesh_normals=True))
				if ref_res.normals is None:
					raise RuntimeError("expand-z boundary self-map grow requires reference model normals")
				ref_xyz = ref_res.xyz_lr.detach()[-1].contiguous()
				ref_normals = ref_res.normals.detach()[-1].contiguous()
				ref_valid = torch.isfinite(ref_xyz).all(dim=-1)
			print(
				f"[optimizer] expand-z: building temporary +{add_depth} winding model "
				f"from reference depth {model.depth - 1} toward target_depth={target_depth}",
				flush=True,
			)
			temp_model = fit_model.Model3D.from_tifxyz_crop(
				ref_xyz,
				ref_valid,
				device=ref_xyz.device,
				mesh_step=model.params.mesh_step,
				winding_step=model.params.winding_step,
				subsample_mesh=model.params.subsample_mesh,
				subsample_winding=model.params.subsample_winding,
				depth=add_depth,
			)
			temp_model.params = replace(
				temp_model.params,
				scaledown=model.params.scaledown,
				z_step_eff=model.params.z_step_eff,
				volume_extent=model.params.volume_extent,
				model_w=model.params.model_w,
				model_h=model.params.model_h,
				depth_windings=tuple(
					int(model.params.depth_windings[-1]) + i + 1
					for i in range(add_depth)
				),
			)
			with torch.no_grad():
				temp_model.amp.copy_(model.amp.detach()[-1:].expand_as(temp_model.amp))
				temp_model.bias.copy_(model.bias.detach()[-1:].expand_as(temp_model.bias))
			old_snap_surf_map_state = getattr(model, "_snap_surf_map_state_for_save", None)
			data = optimize(
				model=temp_model,
				data=data,
				stages=list(stage.children),
				snapshot_interval=0,
				snapshot_fn=lambda **_kw: None,
				progress_fn=progress_fn,
				cancel_fn=cancel_fn,
				ensure_data_fn=ensure_data_fn,
				seed_xyz=seed_xyz,
				out_dir=out_dir,
				self_map_init=self_map_mode,
				self_map_model_w_wraps=self_map_model_w_wraps,
				init_grow=None,
				snap_surf_boundary={
					"fixed_xyz": ref_xyz,
					"fixed_normals": ref_normals,
					"fixed_valid": ref_valid,
				},
			)
			temp_snap_surf_map_state = getattr(temp_model, "_snap_surf_map_state_for_save", None)
			_fuse_expand_z_append(
				temp_model,
				old_snap_surf_map_state=old_snap_surf_map_state if isinstance(old_snap_surf_map_state, dict) else None,
				temp_snap_surf_map_state=temp_snap_surf_map_state if isinstance(temp_snap_surf_map_state, dict) else None,
				add_depth=add_depth,
			)
		return data

	for si, stage in enumerate(stages):
		if stage.children:
			if stage.name == "expand-z":
				data = _run_expand_z_stage(si=si, stage=stage, data=data)
				continue
			raise NotImplementedError(f"unsupported wrapper stage '{stage.name}'")
		if stage.global_opt is None:
			continue
		run_zero_step_cyl = "cyl_params" in stage.global_opt.params
		if stage.global_opt.steps > 0 or run_zero_step_cyl:
			_stage_wall_t0 = time.perf_counter()
			data = _run_opt(si=si, label=f"stage{si}", stage=stage, opt_cfg=stage.global_opt, data=data)
			if active_corr:
				opt_loss_corr.print_detail(f"stage{si} END")
				opt_loss_corr.print_summary()
			quiet_shell_stage = (
				"cyl_params" in stage.global_opt.params and
				stage.name in CYLINDER_SEED_INIT_STAGE_ROLES
			)
			if not quiet_shell_stage:
				print(
					f"[optimizer] stage{si} '{stage.name}' complete in "
					f"{optimizer_timing.format_duration(time.perf_counter() - _stage_wall_t0)}",
					flush=True,
				)
			if _cyl_init_only and stage.name == "cyl_init":
				if hasattr(model, "cyl_shell_search_done"):
					model.cyl_shell_search_done = True
				print(
					"[optimizer] cylinder_seed: no cyl_grow stage; outputting cyl_init shell only",
					flush=True,
				)
				break
			if bool(getattr(model, "cyl_shell_abort", False)):
				break

	# Print sparse cache summary
	if data.sparse_caches:
		for _cache in data.sparse_caches.values():
			_cache.print_summary()

	if active_corr:
		opt_loss_corr.print_detail("END")
		opt_loss_corr.print_summary()
	elif has_corr_pts:
		print("[optimizer] corr points present but no corr weight > 0, no corr loss computed", flush=True)

	_publish_snap_surf_map_state()
	print(f"[optimizer] total optimize time: {optimizer_timing.format_duration(time.perf_counter() - _optimize_t0)}", flush=True)
	return data
