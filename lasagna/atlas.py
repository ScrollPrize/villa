from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

import fit_data
import model
from tifxyz_io import load_tifxyz, surface_step_stats


@dataclass(frozen=True)
class AtlasInit:
	model: model.Model3D
	atlas_lines: fit_data.AtlasLines3D
	metadata: dict[str, Any]


def _load_json_path(path: str | Path, *, label: str) -> dict[str, Any]:
	p = Path(path)
	if not p.is_file():
		raise ValueError(f"{label} path is not a file: {p}")
	obj = json.loads(p.read_text(encoding="utf-8"))
	if not isinstance(obj, dict):
		raise ValueError(f"{label} must be a JSON object: {p}")
	return obj


def load_vc3d_fiber_line(path: str | Path) -> list[tuple[float, float, float]]:
	obj = _load_json_path(path, label="vc3d_fiber")
	if obj.get("type", "vc3d_fiber") != "vc3d_fiber":
		raise ValueError(f"unsupported fiber JSON type at {path}: {obj.get('type')!r}")
	points = obj.get("line_points")
	if not isinstance(points, list):
		raise ValueError(f"vc3d_fiber requires line_points[]: {path}")
	out: list[tuple[float, float, float]] = []
	for i, item in enumerate(points):
		if not isinstance(item, list) or len(item) != 3:
			raise ValueError(f"vc3d_fiber line_points[{i}] must be [x,y,z]")
		p = tuple(float(v) for v in item)
		if not all(math.isfinite(v) for v in p):
			raise ValueError(f"vc3d_fiber line_points[{i}] contains non-finite values")
		out.append(p)
	return out


def load_vc3d_fiber_control_points(path: str | Path) -> list[tuple[float, float, float]]:
	obj = _load_json_path(path, label="vc3d_fiber")
	if obj.get("type", "vc3d_fiber") != "vc3d_fiber":
		raise ValueError(f"unsupported fiber JSON type at {path}: {obj.get('type')!r}")
	points = obj.get("control_points", [])
	if points is None:
		points = []
	if not isinstance(points, list):
		raise ValueError(f"vc3d_fiber control_points must be a list: {path}")
	out: list[tuple[float, float, float]] = []
	for i, item in enumerate(points):
		if not isinstance(item, list) or len(item) != 3:
			raise ValueError(f"vc3d_fiber control_points[{i}] must be [x,y,z]")
		p = tuple(float(v) for v in item)
		if not all(math.isfinite(v) for v in p):
			raise ValueError(f"vc3d_fiber control_points[{i}] contains non-finite values")
		out.append(p)
	return out


def load_vc3d_atlas_fiber_mapping(path: str | Path) -> dict[str, Any]:
	obj = _load_json_path(path, label="vc3d_atlas_fiber_mapping")
	if obj.get("type", "vc3d_atlas_fiber_mapping") != "vc3d_atlas_fiber_mapping":
		raise ValueError(f"unsupported mapping JSON type at {path}: {obj.get('type')!r}")
	if int(obj.get("version", 0)) != 4:
		raise ValueError(
			f"vc3d_atlas_fiber_mapping {path} has obsolete or missing version "
			f"{obj.get('version')!r}; rebuild required"
		)
	anchors = obj.get("line_anchors")
	if not isinstance(anchors, list):
		raise ValueError(f"vc3d_atlas_fiber_mapping requires line_anchors[]: {path}")
	return obj


def load_vc3d_atlas_pred_snap_points(path: str | Path) -> list[dict[str, tuple[float, float, float]]]:
	obj = _load_json_path(path, label="vc3d_atlas_pred_snap_points")
	if obj.get("type", "vc3d_atlas_pred_snap_points") != "vc3d_atlas_pred_snap_points":
		raise ValueError(f"unsupported pred-snap JSON type at {path}: {obj.get('type')!r}")
	if int(obj.get("version", 0)) != 1:
		raise ValueError(f"unsupported pred-snap JSON version at {path}: {obj.get('version')!r}")
	entries = obj.get("entries", {})
	if entries is None:
		return []
	if not isinstance(entries, dict):
		raise ValueError(f"vc3d_atlas_pred_snap_points entries must be an object: {path}")
	out: list[dict[str, tuple[float, float, float]]] = []
	for key, item in entries.items():
		if not isinstance(item, dict):
			raise ValueError(f"pred-snap entry {key!r} must be an object: {path}")
		control = item.get("control_point")
		snap = item.get("pred_snap_point")
		if not (isinstance(control, list) and len(control) == 3):
			raise ValueError(f"pred-snap entry {key!r} requires control_point [x,y,z]: {path}")
		if snap is None:
			continue
		if not (isinstance(snap, list) and len(snap) == 3):
			raise ValueError(f"pred-snap entry {key!r} pred_snap_point must be [x,y,z] or null: {path}")
		control_p = tuple(float(v) for v in control)
		snap_p = tuple(float(v) for v in snap)
		if not (all(math.isfinite(v) for v in control_p) and all(math.isfinite(v) for v in snap_p)):
			continue
		out.append({"control_point": control_p, "pred_snap_point": snap_p})
	return out


def _validate_wrapped_base(xyz: torch.Tensor, valid: torch.Tensor) -> int:
	if xyz.ndim != 3 or int(xyz.shape[-1]) != 3:
		raise ValueError(f"atlas base shell must have shape (H,W,3), got {tuple(xyz.shape)}")
	H, W, _ = xyz.shape
	if H < 2 or W < 2:
		raise ValueError(f"atlas base shell is too small: {tuple(xyz.shape)}")
	both = valid[:, 0] & valid[:, W - 1]
	if bool(both.any().detach().cpu()):
		delta = (xyz[:, 0] - xyz[:, W - 1]).norm(dim=-1)
		max_delta = float(delta[both].amax().detach().cpu())
		if max_delta > 1.0e-3:
			raise ValueError(f"atlas base first/last columns do not match; max_delta={max_delta:.6g}")
	return W - 1


def _atlas_winding_for_column(atlas_u: float, period_columns: int, zero_winding_column: int) -> int:
	if period_columns <= 0 or not math.isfinite(atlas_u):
		return 0
	period = float(period_columns)
	return int(math.floor((float(atlas_u) - float(zero_winding_column)) / period))


def _margin_vx_to_grid_units(margin_vx: int, step_vx: float) -> int:
	if math.isfinite(step_vx) and step_vx > 0.0:
		return max(0, int(math.ceil(float(margin_vx) / float(step_vx))))
	return max(0, int(margin_vx))


def _crop_wrapped_base_shell(
	xyz: torch.Tensor,
	valid: torch.Tensor,
	*,
	row_start: int,
	row_end: int,
	column_start: int,
	column_end: int,
) -> tuple[torch.Tensor, torch.Tensor]:
	period_columns = int(xyz.shape[1]) - 1
	if period_columns <= 0:
		raise ValueError("atlas base shell has invalid horizontal period")
	row_start = int(row_start)
	row_end = int(row_end)
	column_start = int(column_start)
	column_end = int(column_end)
	if row_end <= row_start:
		raise ValueError(f"atlas vertical crop is empty: [{row_start}, {row_end})")
	if column_end <= column_start:
		raise ValueError(f"atlas horizontal crop is empty: [{column_start}, {column_end})")
	rows_xyz = xyz[row_start:row_end]
	rows_valid = valid[row_start:row_end]
	width = column_end - column_start
	cols = torch.arange(width, device=xyz.device, dtype=torch.long).add(column_start).remainder(period_columns)
	return rows_xyz.index_select(1, cols).contiguous(), rows_valid.index_select(1, cols).contiguous()


def _surface_normals(xyz: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
	H, W, _ = xyz.shape
	h_prev = xyz[torch.arange(H, device=xyz.device).sub(1).clamp(0, H - 1)]
	h_next = xyz[torch.arange(H, device=xyz.device).add(1).clamp(0, H - 1)]
	w_prev = xyz[:, torch.arange(W, device=xyz.device).sub(1).clamp(0, W - 1)]
	w_next = xyz[:, torch.arange(W, device=xyz.device).add(1).clamp(0, W - 1)]
	dh = h_next - h_prev
	dw = w_next - w_prev
	n = torch.linalg.cross(dw, dh, dim=-1)
	length = torch.linalg.vector_norm(n, dim=-1, keepdim=True)
	n = torch.where(length > 1.0e-8, n / length.clamp_min(1.0e-12), torch.zeros_like(n))
	return torch.where(valid.unsqueeze(-1), n, torch.zeros_like(n))


def _bilinear_sample_grid(values: torch.Tensor, h: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
	H, W = int(values.shape[0]), int(values.shape[1])
	hc = h.clamp(0.0, float(max(0, H - 1)))
	wc = w.clamp(0.0, float(max(0, W - 1)))
	h0 = torch.floor(hc).to(dtype=torch.long).clamp(0, H - 1)
	w0 = torch.floor(wc).to(dtype=torch.long).clamp(0, W - 1)
	h1 = (h0 + 1).clamp(0, H - 1)
	w1 = (w0 + 1).clamp(0, W - 1)
	fh = (hc - h0.to(dtype=hc.dtype)).unsqueeze(-1)
	fw = (wc - w0.to(dtype=wc.dtype)).unsqueeze(-1)
	v00 = values[h0, w0]
	v10 = values[h1, w0]
	v01 = values[h0, w1]
	v11 = values[h1, w1]
	return (
		v00 * (1.0 - fh) * (1.0 - fw)
		+ v10 * fh * (1.0 - fw)
		+ v01 * (1.0 - fh) * fw
		+ v11 * fh * fw
	)


def _bilinear_sample_valid(valid: torch.Tensor, h: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
	H, W = int(valid.shape[0]), int(valid.shape[1])
	hc = h.clamp(0.0, float(max(0, H - 1)))
	wc = w.clamp(0.0, float(max(0, W - 1)))
	h0 = torch.floor(hc).to(dtype=torch.long).clamp(0, H - 1)
	w0 = torch.floor(wc).to(dtype=torch.long).clamp(0, W - 1)
	h1 = (h0 + 1).clamp(0, H - 1)
	w1 = (w0 + 1).clamp(0, W - 1)
	return valid[h0, w0] & valid[h1, w0] & valid[h0, w1] & valid[h1, w1]


def _resample_axis_count(count: int, source_step_vx: float, target_step_vx: float) -> int:
	if count <= 1:
		return max(1, int(count))
	if not (math.isfinite(source_step_vx) and source_step_vx > 0.0):
		return int(count)
	if not (math.isfinite(target_step_vx) and target_step_vx > 0.0):
		return int(count)
	source_grid_step = float(target_step_vx) / float(source_step_vx)
	if not (math.isfinite(source_grid_step) and source_grid_step > 0.0):
		return int(count)
	return max(2, int(math.ceil(float(count - 1) / source_grid_step)) + 1)


def _resample_base_to_mesh_step(
	xyz: torch.Tensor,
	valid: torch.Tensor,
	*,
	mesh_step: int,
) -> tuple[torch.Tensor, torch.Tensor, float, float, dict[str, Any]]:
	source_h, source_w, _source_diag, source_avg = surface_step_stats(xyz, valid)
	target_step = float(max(1, int(mesh_step)))
	out_h = _resample_axis_count(int(xyz.shape[0]), source_h, target_step)
	out_w = _resample_axis_count(int(xyz.shape[1]), source_w, target_step)
	if out_h == int(xyz.shape[0]) and out_w == int(xyz.shape[1]):
		row_scale = 1.0
		col_scale = 1.0
		resampled_xyz = xyz
		resampled_valid = valid
	else:
		row_positions = torch.linspace(0.0, float(int(xyz.shape[0]) - 1), out_h, device=xyz.device, dtype=torch.float32)
		col_positions = torch.linspace(0.0, float(int(xyz.shape[1]) - 1), out_w, device=xyz.device, dtype=torch.float32)
		h_grid, w_grid = torch.meshgrid(row_positions, col_positions, indexing="ij")
		resampled_xyz = _bilinear_sample_grid(xyz, h_grid.reshape(-1), w_grid.reshape(-1)).reshape(out_h, out_w, 3)
		resampled_valid = _bilinear_sample_valid(valid, h_grid.reshape(-1), w_grid.reshape(-1)).reshape(out_h, out_w)
		row_scale = float(out_h - 1) / float(max(1, int(xyz.shape[0]) - 1))
		col_scale = float(out_w - 1) / float(max(1, int(xyz.shape[1]) - 1))
	actual_h, actual_w, actual_diag, actual_avg = surface_step_stats(resampled_xyz, resampled_valid)
	meta = {
		"requested_mesh_step": int(max(1, int(mesh_step))),
		"source_mesh_step_h": float(source_h),
		"source_mesh_step_w": float(source_w),
		"source_mesh_step_avg": float(source_avg),
		"resampled_mesh_step_h": float(actual_h),
		"resampled_mesh_step_w": float(actual_w),
		"resampled_mesh_step_diag": float(actual_diag),
		"resampled_mesh_step_avg": float(actual_avg),
		"resampled_source_rows": int(xyz.shape[0]),
		"resampled_source_columns": int(xyz.shape[1]),
		"resampled_rows": int(out_h),
		"resampled_columns": int(out_w),
		"resampled_row_index_scale": float(row_scale),
		"resampled_column_index_scale": float(col_scale),
	}
	return resampled_xyz.contiguous(), resampled_valid.contiguous(), row_scale, col_scale, meta


def _anchor_atlas_uv(anchor: dict[str, Any]) -> tuple[float, float]:
	atlas = anchor.get("atlas")
	if isinstance(atlas, list) and len(atlas) == 2:
		return float(atlas[0]), float(atlas[1])
	return float(anchor.get("atlasU")), float(anchor.get("atlasV"))


def _anchor_world(anchor: dict[str, Any]) -> tuple[float, float, float] | None:
	world = anchor.get("world")
	if isinstance(world, list) and len(world) == 3:
		out = tuple(float(v) for v in world)
		if all(math.isfinite(v) for v in out):
			return out
	return None


def _line_anchor_target(
	line_points: list[tuple[float, float, float]],
	source_index: int,
) -> tuple[float, float, float]:
	if 0 <= source_index < len(line_points):
		return line_points[source_index]
	raise ValueError(
		f"line anchor source_index {source_index} is outside fiber line_points "
		f"range [0,{len(line_points)})"
	)


def _same_point(
	a: tuple[float, float, float],
	b: tuple[float, float, float],
	*,
	tol: float = 1.0e-6,
) -> bool:
	return all(abs(float(a[i]) - float(b[i])) <= tol for i in range(3))


def _control_point_line_indices(
	control_points: list[tuple[float, float, float]],
	line_points: list[tuple[float, float, float]],
	*,
	label: str,
) -> dict[int, int]:
	line_cursor = 0
	out: dict[int, int] = {}
	for control_i, control in enumerate(control_points):
		match_i: int | None = None
		while line_cursor < len(line_points):
			if _same_point(control, line_points[line_cursor]):
				match_i = line_cursor
				line_cursor += 1
				break
			line_cursor += 1
		if match_i is None:
			raise ValueError(
				f"{label} control_points[{control_i}] is not an ordered subset "
				"of line_points; rebuild required"
			)
		out[match_i] = control_i
	return out


def build_atlas_init(
	atlas_obj: dict[str, Any],
	*,
	device: torch.device,
	mesh_step: int,
	winding_step: int,
) -> AtlasInit:
	if atlas_obj.get("type") != "lasagna_atlas" or int(atlas_obj.get("version", 0)) != 1:
		raise ValueError("atlas config must be type=lasagna_atlas version=1")
	base = atlas_obj.get("base")
	if not isinstance(base, dict) or not base.get("path"):
		raise ValueError("atlas config requires resolved base.path")
	metadata = atlas_obj.get("metadata")
	if not isinstance(metadata, dict):
		metadata = {}
	zero_col = int(metadata.get("zero_winding_column", 0))

	base_xyz, base_valid, _base_meta = load_tifxyz(str(base["path"]), device=device)
	period_columns = _validate_wrapped_base(base_xyz, base_valid)
	if int(metadata.get("period_columns", period_columns)) != period_columns:
		raise ValueError(
			f"atlas period_columns mismatch: metadata={metadata.get('period_columns')} "
			f"base={period_columns}"
		)

	line_objects: dict[str, str] = {}
	objects = atlas_obj.get("objects", {})
	if isinstance(objects, dict):
		for item in objects.get("line", []) if isinstance(objects.get("line", []), list) else []:
			if not isinstance(item, dict) or not item.get("path"):
				continue
			line_id = str(item.get("id") or item.get("ref", {}).get("name") or "")
			if line_id:
				line_objects[line_id] = str(item["path"])

	mapped_rows: list[float] = []
	control_rows: list[float] = []
	control_actual_us: list[float] = []
	targets: list[tuple[float, float, float]] = []
	object_ids: list[str] = []
	source_indices: list[int] = []
	actual_us: list[float] = []
	is_control_points: list[bool] = []
	is_snap_points: list[bool] = []

	maps = atlas_obj.get("maps", [])
	if not isinstance(maps, list):
		raise ValueError("atlas config maps must be a list")
	for map_i, map_entry in enumerate(maps):
		if not isinstance(map_entry, dict) or str(map_entry.get("object_type", "line")) != "line":
			continue
		map_path = map_entry.get("map_path")
		if not map_path:
			raise ValueError(f"atlas map {map_i} requires resolved map_path")
		object_id = str(map_entry.get("object_id") or map_entry.get("object_ref", {}).get("name") or "")
		line_path = str(map_entry.get("object_path") or line_objects.get(object_id) or "")
		line_points = load_vc3d_fiber_line(line_path) if line_path else []
		control_points = load_vc3d_fiber_control_points(line_path) if line_path else []
		mapping = load_vc3d_atlas_fiber_mapping(str(map_path))
		control_line_to_row = _control_point_line_indices(
			control_points,
			line_points,
			label=f"atlas map {map_i} fiber {line_path or object_id}",
		)
		winding_offset = int(map_entry.get("winding_offset", 0))
		raw_control_anchors = mapping.get("control_anchors", [])
		if raw_control_anchors is None:
			raw_control_anchors = []
		if not isinstance(raw_control_anchors, list):
			raise ValueError(f"atlas map {map_i} control_anchors must be a list: {map_path}")

		control_samples: list[tuple[int, float, float, tuple[float, float, float]]] = []
		for anchor_i, anchor in enumerate(raw_control_anchors):
			if not isinstance(anchor, dict):
				raise ValueError(f"atlas map {map_i} control_anchors[{anchor_i}] must be an object: {map_path}")
			source_index = int(anchor.get("source_index", -1))
			if source_index < 0:
				raise ValueError(f"atlas map {map_i} control_anchors[{anchor_i}] has invalid source_index: {map_path}")
			atlas_u, atlas_v = _anchor_atlas_uv(anchor)
			if not (math.isfinite(atlas_u) and math.isfinite(atlas_v)):
				raise ValueError(f"atlas map {map_i} control_anchors[{anchor_i}] has non-finite atlas coordinates: {map_path}")
			actual_u = atlas_u + float(winding_offset * period_columns)
			target = _anchor_world(anchor)
			if target is None or not all(math.isfinite(v) for v in target):
				raise ValueError(f"atlas map {map_i} control_anchors[{anchor_i}] requires finite world: {map_path}")
			control_row = control_line_to_row.get(source_index)
			if control_row is None:
				raise ValueError(
					f"atlas map {map_i} control_anchors[{anchor_i}] source_index {source_index} "
					"does not identify a fiber control point line_points index; rebuild required"
				)
			control_target = control_points[control_row]
			if not _same_point(target, control_target):
				raise ValueError(
					f"atlas map {map_i} control_anchors[{anchor_i}] world does not match "
					f"fiber control_points[{control_row}]; rebuild required"
				)
			control_samples.append((source_index, float(atlas_v), float(actual_u), target))

		if not control_samples:
			raise ValueError(f"atlas map {map_i} has no usable control_anchors: {map_path}")
		control_line_indices = [source_index for source_index, _atlas_v, _actual_u, _target in control_samples]
		control_span_start = min(control_line_indices)
		control_span_end = max(control_line_indices)
		control_keys = {(object_id, line_index) for line_index in control_line_indices}

		for source_index, atlas_v, actual_u, target in control_samples:
			actual_us.append(float(actual_u))
			mapped_rows.append(atlas_v)
			control_rows.append(atlas_v)
			control_actual_us.append(float(actual_u))
			targets.append(target)
			object_ids.append(object_id)
			source_indices.append(source_index)
			is_control_points.append(True)
			is_snap_points.append(False)

		pred_snap_path = map_entry.get("pred_snap_path")
		if pred_snap_path:
			control_by_point: list[tuple[tuple[float, float, float], int, float, float]] = [
				(target, source_index, atlas_v, actual_u)
				for source_index, atlas_v, actual_u, target in control_samples
			]
			for snap in load_vc3d_atlas_pred_snap_points(str(pred_snap_path)):
				control = snap["control_point"]
				match: tuple[int, float, float] | None = None
				for target, source_index, atlas_v, actual_u in control_by_point:
					if _same_point(control, target):
						match = (source_index, atlas_v, actual_u)
						break
				if match is None:
					raise ValueError(
						f"pred-snap entry control_point {control} does not match a control anchor: {pred_snap_path}"
					)
				source_index, atlas_v, actual_u = match
				actual_us.append(float(actual_u))
				mapped_rows.append(float(atlas_v))
				targets.append(snap["pred_snap_point"])
				object_ids.append(object_id)
				source_indices.append(source_index)
				is_control_points.append(False)
				is_snap_points.append(True)

		for anchor in mapping.get("line_anchors", []):
			if not isinstance(anchor, dict):
				continue
			source_index = int(anchor.get("source_index", -1))
			if source_index < control_span_start or source_index > control_span_end:
				continue
			if (object_id, source_index) in control_keys:
				continue
			atlas_u, atlas_v = _anchor_atlas_uv(anchor)
			if not (math.isfinite(atlas_u) and math.isfinite(atlas_v)):
				continue
			actual_u = atlas_u + float(winding_offset * period_columns)
			target = _line_anchor_target(line_points, source_index)
			if not all(math.isfinite(v) for v in target):
				continue
			actual_us.append(float(actual_u))
			mapped_rows.append(float(atlas_v))
			targets.append(target)
			object_ids.append(object_id)
			source_indices.append(source_index)
			is_control_points.append(False)
			is_snap_points.append(False)

	if not targets:
		raise ValueError("atlas config produced no usable line anchors")
	left_w = min(_atlas_winding_for_column(u, period_columns, zero_col) for u in control_actual_us)
	right_w = max(_atlas_winding_for_column(u, period_columns, zero_col) for u in control_actual_us)
	unwrap_count = max(1, right_w - left_w + 1)
	start_column = zero_col + left_w * period_columns
	init_margin_vx = 4000
	base_step_h, base_step_w, _base_step_diag, _base_step_avg = surface_step_stats(base_xyz, base_valid)
	init_margin_rows = _margin_vx_to_grid_units(init_margin_vx, base_step_h)
	init_margin_columns = _margin_vx_to_grid_units(init_margin_vx, base_step_w)
	base_shell_height = int(base_xyz.shape[0])
	crop_row_start = max(0, int(math.floor(min(control_rows))) - init_margin_rows)
	crop_row_end = min(base_shell_height, int(math.ceil(max(control_rows))) + init_margin_rows)
	crop_column_start = int(math.floor(min(control_actual_us))) - init_margin_columns
	crop_column_end = int(math.ceil(max(control_actual_us))) + init_margin_columns
	atlas_u_offset = float(crop_column_start)
	mapped_rows = [float(v) - float(crop_row_start) for v in mapped_rows]
	mapped_cols = [float(u) - atlas_u_offset for u in actual_us]

	cropped_xyz, cropped_valid = _crop_wrapped_base_shell(
		base_xyz,
		base_valid,
		row_start=crop_row_start,
		row_end=crop_row_end,
		column_start=crop_column_start,
		column_end=crop_column_end,
	)
	unwrapped_xyz, unwrapped_valid, row_index_scale, col_index_scale, resample_meta = _resample_base_to_mesh_step(
		cropped_xyz,
		cropped_valid,
		mesh_step=mesh_step,
	)
	mapped_rows = [float(v) * row_index_scale for v in mapped_rows]
	mapped_cols = [float(v) * col_index_scale for v in mapped_cols]
	atlas_winding_model_ranges = tuple(
		(
			int(winding),
			float((zero_col + winding * period_columns - atlas_u_offset) * col_index_scale),
			float((zero_col + (winding + 1) * period_columns - atlas_u_offset) * col_index_scale),
		)
		for winding in range(left_w, right_w + 1)
	)
	normals_grid = _surface_normals(unwrapped_xyz, unwrapped_valid)
	h_t = torch.tensor(mapped_rows, device=device, dtype=torch.float32)
	w_t = torch.tensor(mapped_cols, device=device, dtype=torch.float32)
	n_t = _bilinear_sample_grid(normals_grid, h_t, w_t)
	n_len = torch.linalg.vector_norm(n_t, dim=-1, keepdim=True)
	n_t = torch.where(n_len > 1.0e-8, n_t / n_len.clamp_min(1.0e-12), torch.zeros_like(n_t))
	target_t = torch.tensor(targets, device=device, dtype=torch.float32)
	lines = fit_data.AtlasLines3D(
		target_xyz=target_t,
		normal_xyz=n_t,
		model_h=h_t,
		model_w=w_t,
		object_ids=tuple(object_ids),
		source_indices=tuple(source_indices),
		is_control_point=torch.tensor(is_control_points, device=device, dtype=torch.bool),
		is_snap_point=torch.tensor(is_snap_points, device=device, dtype=torch.bool),
		atlas_winding_model_ranges=atlas_winding_model_ranges,
	)

	mdl = model.Model3D.from_tifxyz_crop(
		unwrapped_xyz,
		unwrapped_valid,
		device=device,
		mesh_step=max(1, int(mesh_step)),
		winding_step=winding_step,
		subsample_mesh=1,
		subsample_winding=1,
		depth=1,
	)
	meta = {
		"period_columns": int(period_columns),
		"leftmost_winding": int(left_w),
		"rightmost_winding": int(right_w),
		"unwrap_count": int(unwrap_count),
		"start_column": int(start_column),
		"init_margin_vx": int(init_margin_vx),
		"init_margin_rows": int(init_margin_rows),
		"init_margin_columns": int(init_margin_columns),
		"crop_row_start": int(crop_row_start),
		"crop_row_end": int(crop_row_end),
		"crop_column_start": int(crop_column_start),
		"crop_column_end": int(crop_column_end),
		"atlas_u_offset": float(atlas_u_offset),
		"line_sample_count": int(target_t.shape[0]),
		"control_point_sample_count": int(sum(1 for v in is_control_points if v)),
		"pred_snap_sample_count": int(sum(1 for v in is_snap_points if v)),
		"other_line_point_sample_count": int(sum(1 for is_ctl, is_snap in zip(is_control_points, is_snap_points) if not is_ctl and not is_snap)),
	}
	meta.update(resample_meta)
	return AtlasInit(model=mdl, atlas_lines=lines, metadata=meta)
