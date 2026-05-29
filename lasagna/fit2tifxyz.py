from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile
import torch

import cli_json
import model
import volume_scale


@dataclass(frozen=True)
class ExportConfig:
	input: str
	output: str
	prefix: str = "winding_"
	device: str = "cpu"
	single_segment: bool = False
	copy_model: bool = False
	output_name: str | None = None
	voxel_size_um: float | None = None
	target_volume_shape_zyx: tuple[int, int, int] | None = None
	flow_gate_channels: str = "auto"


def _valid_xyz_mask(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
	return (
		np.isfinite(x)
		& np.isfinite(y)
		& np.isfinite(z)
		& ~((x == -1.0) & (y == -1.0) & (z == -1.0))
	)


def _build_parser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(description="Export 3D fit model as tifxyz surfaces (one per winding/depth)")
	cli_json.add_args(p)
	g = p.add_argument_group("io")
	g.add_argument("--input", required=True, help="Model checkpoint (.pt)")
	g.add_argument("--output", required=True, help="Output directory")
	g.add_argument("--prefix", default="winding_", help="Output tifxyz directory prefix")
	g.add_argument("--single-segment", action="store_true", default=False,
		help="Export all windings into a single tifxyz")
	g.add_argument("--copy-model", action="store_true", default=False,
		help="Copy model checkpoint instead of symlink")
	g.add_argument("--output-name", default=None, help="Override tifxyz directory name")
	g.add_argument("--voxel-size-um", type=float, default=None,
		help="Voxel size in micrometers (for area calculation)")
	g.add_argument("--target-volume-shape-zyx", type=int, nargs=3, metavar=("Z", "Y", "X"), default=None,
		help="Export coordinates into this VC3D target volume coordinate scale")
	g.add_argument("--flow-gate-channels", choices=("auto", "on", "off"), default="auto",
		help="Export captured flow-gate component channels when available")
	return p


def _get_area(x: np.ndarray, y: np.ndarray, z: np.ndarray,
			  step_size: float, voxel_size_um: float | None) -> dict:
	"""Compute surface area from a tifxyz mesh grid.

	Counts valid quads (all 4 corners finite and not the -1/-1/-1 sentinel) × step_size².
	Returns dict with area_vx2 and optionally area_cm2.
	"""
	valid = _valid_xyz_mask(x, y, z)
	valid_quads = valid[:-1, :-1] & valid[:-1, 1:] & valid[1:, :-1] & valid[1:, 1:]
	area_vx2 = int(valid_quads.sum()) * step_size ** 2
	result = {"area_vx2": area_vx2}
	if voxel_size_um is not None:
		result["area_cm2"] = area_vx2 * voxel_size_um ** 2 / 1e8
	return result


def _print_area(area: dict) -> None:
	parts = [f"area_vx2={area['area_vx2']:.0f}"]
	if "area_cm2" in area:
		parts.append(f"area_cm2={area['area_cm2']:.4f}")
	print(f"[fit2tifxyz] {' '.join(parts)}", flush=True)


def _bbox_for_xyz(x: np.ndarray, y: np.ndarray, z: np.ndarray, *, out_dir: Path) -> list[list[float]]:
	valid = _valid_xyz_mask(x, y, z)
	if not bool(valid.any()):
		print(f"[fit2tifxyz] WARNING: no valid vertices in {out_dir}; writing invalid bbox", flush=True)
		return [[-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0]]
	return [
		[float(np.nanmin(x[valid])), float(np.nanmin(y[valid])), float(np.nanmin(z[valid]))],
		[float(np.nanmax(x[valid])), float(np.nanmax(y[valid])), float(np.nanmax(z[valid]))],
	]


def _shape_from_model_params(model_params: dict | None, key: str) -> tuple[int, int, int] | None:
	if not isinstance(model_params, dict):
		return None
	return volume_scale.parse_shape_zyx(model_params.get(key), name=f"_model_params_.{key}")


def _export_coordinate_scale(
	*,
	model_params: dict | None,
	target_volume_shape_zyx: tuple[int, int, int] | None,
) -> tuple[float, tuple[int, int, int] | None, tuple[int, int, int] | None]:
	lasagna_base_shape = _shape_from_model_params(model_params, "lasagna_base_shape_zyx")
	if target_volume_shape_zyx is None:
		return 1.0, lasagna_base_shape, lasagna_base_shape
	if lasagna_base_shape is None:
		print(
			"[fit2tifxyz] WARNING: checkpoint has no lasagna_base_shape_zyx; "
			"export target volume shape is recorded without coordinate scaling",
			flush=True,
		)
		return 1.0, None, target_volume_shape_zyx
	scale = volume_scale.coordinate_scale_between_shapes(
		from_shape_zyx=lasagna_base_shape,
		to_shape_zyx=target_volume_shape_zyx,
		from_name="lasagna_base_shape_zyx",
		to_name="target_volume_shape_zyx",
	)
	if not scale.is_identity:
		print(
			f"[fit2tifxyz] export coordinate scale={scale.factor:.9g} "
			f"lasagna_base_shape={list(lasagna_base_shape)} "
			f"target_shape={list(target_volume_shape_zyx)}",
			flush=True,
		)
	return scale.factor, lasagna_base_shape, target_volume_shape_zyx


def _scaled_xyz_for_export(
	x: np.ndarray,
	y: np.ndarray,
	z: np.ndarray,
	factor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	return volume_scale.scale_tifxyz_arrays(x, y, z, factor)


def _mark_segment_vertices(
	mask: np.ndarray,
	p0: tuple[float, float],
	p1: tuple[float, float],
	*,
	eps: float = 1.0e-6,
) -> None:
	r0, c0 = p0
	r1, c1 = p1
	rmin = max(0, int(math.floor(min(r0, r1) - eps)))
	rmax = min(mask.shape[0] - 1, int(math.ceil(max(r0, r1) + eps)))
	cmin = max(0, int(math.floor(min(c0, c1) - eps)))
	cmax = min(mask.shape[1] - 1, int(math.ceil(max(c0, c1) + eps)))
	seg = np.asarray([r1 - r0, c1 - c0], dtype=np.float64)
	seg_len2 = float(np.dot(seg, seg))
	for r in range(rmin, rmax + 1):
		for c in range(cmin, cmax + 1):
			v = np.asarray([float(r) - r0, float(c) - c0], dtype=np.float64)
			if seg_len2 <= eps:
				dist2 = float(np.dot(v, v))
			else:
				t = max(0.0, min(1.0, float(np.dot(v, seg) / seg_len2)))
				d = v - t * seg
				dist2 = float(np.dot(d, d))
			if dist2 <= eps * eps:
				mask[r, c] = True


def _rasterize_contours(
	contours_rc: list[list[tuple[float, float]]],
	shape: tuple[int, int],
) -> np.ndarray:
	h, w = (int(shape[0]), int(shape[1]))
	out = np.zeros((h, w), dtype=bool)
	if h <= 0 or w <= 0:
		return out

	for r in range(h):
		y = float(r)
		xints: list[float] = []
		for contour in contours_rc:
			if len(contour) < 3:
				continue
			for i, (r0, c0) in enumerate(contour):
				r1, c1 = contour[(i + 1) % len(contour)]
				if (r0 > y) != (r1 > y):
					t = (y - r0) / (r1 - r0)
					xints.append(float(c0 + t * (c1 - c0)))
		xints.sort()
		for i in range(0, len(xints) - 1, 2):
			c0 = int(math.ceil(min(xints[i], xints[i + 1]) - 1.0e-6))
			c1 = int(math.floor(max(xints[i], xints[i + 1]) + 1.0e-6))
			if c1 < 0 or c0 >= w:
				continue
			out[r, max(0, c0):min(w, c1 + 1)] = True

	for contour in contours_rc:
		if len(contour) < 2:
			continue
		for i, p0 in enumerate(contour):
			_mark_segment_vertices(out, p0, contour[(i + 1) % len(contour)])
	return out


def _dilate_chebyshev(mask: np.ndarray, radius: int) -> np.ndarray:
	r = int(radius)
	if r <= 0:
		return mask.astype(bool, copy=True)
	src = mask.astype(bool, copy=False)
	out = src.copy()
	h, w = src.shape
	for dr in range(-r, r + 1):
		rs0 = max(0, -dr)
		rs1 = min(h, h - dr)
		rd0 = max(0, dr)
		rd1 = min(h, h + dr)
		for dc in range(-r, r + 1):
			cs0 = max(0, -dc)
			cs1 = min(w, w - dc)
			cd0 = max(0, dc)
			cd1 = min(w, w + dc)
			out[rd0:rd1, cd0:cd1] |= src[rs0:rs1, cs0:cs1]
	return out


def _mask_bbox(mask: np.ndarray) -> list[int] | None:
	rc = np.argwhere(mask.astype(bool, copy=False))
	if rc.size == 0:
		return None
	rmin, cmin = (int(v) for v in rc.min(axis=0))
	rmax, cmax = (int(v) for v in rc.max(axis=0))
	return [rmin, rmax, cmin, cmax]


def _points_bbox(points: list[tuple[float, float]]) -> list[float] | None:
	if not points:
		return None
	arr = np.asarray(points, dtype=np.float64)
	return [
		round(float(np.min(arr[:, 0])), 3),
		round(float(np.max(arr[:, 0])), 3),
		round(float(np.min(arr[:, 1])), 3),
		round(float(np.max(arr[:, 1])), 3),
	]


def _corr_points_result_list(corr_results: dict) -> list[dict]:
	points_list = corr_results.get("points_list", None)
	if isinstance(points_list, list):
		return [p for p in points_list if isinstance(p, dict)]
	points = corr_results.get("points", None)
	if not isinstance(points, dict):
		return []
	return [
		p for _key, p in sorted(
			points.items(),
			key=lambda item: (0, int(item[0])) if str(item[0]).isdigit() else (1, str(item[0])),
		)
		if isinstance(p, dict)
	]


def _corr_points_result_lookup(corr_results: dict) -> dict[tuple[int, int], dict]:
	lookup: dict[tuple[int, int], dict] = {}
	for point in _corr_points_result_list(corr_results):
		try:
			cid = int(point.get("collection_id"))
			pid = int(point.get("point_id"))
		except (TypeError, ValueError):
			continue
		lookup[(cid, pid)] = point
	return lookup


def _approval_mask_corr_collection_ids(payload: dict, fit_config: dict | None) -> set[int]:
	ids_raw = payload.get("corr_collection_ids", [])
	ids: set[int] = set()
	if isinstance(ids_raw, list):
		for value in ids_raw:
			try:
				ids.add(int(value))
			except (TypeError, ValueError):
				continue
	if ids:
		return ids
	if not isinstance(fit_config, dict):
		return ids
	corr_points = fit_config.get("corr_points", {})
	if not isinstance(corr_points, dict):
		return ids
	collections = corr_points.get("collections", {})
	if not isinstance(collections, dict):
		return ids
	for cid, collection in collections.items():
		if isinstance(collection, dict) and collection.get("name") == "approval_inpaint":
			try:
				ids.add(int(cid))
			except (TypeError, ValueError):
				continue
	return ids


def _approval_mask_corr_contours(payload: dict, collection_ids: set[int]) -> list[tuple[int, list[int]]]:
	raw = payload.get("corr_contours", [])
	contours: list[tuple[int, list[int]]] = []
	if not isinstance(raw, list):
		return contours
	for item in raw:
		if not isinstance(item, dict):
			continue
		try:
			cid = int(item.get("collection_id"))
		except (TypeError, ValueError):
			continue
		if cid not in collection_ids:
			continue
		point_ids_raw = item.get("point_ids", [])
		if not isinstance(point_ids_raw, list):
			continue
		point_ids = []
		for value in point_ids_raw:
			try:
				point_ids.append(int(value))
			except (TypeError, ValueError):
				continue
		if point_ids:
			contours.append((cid, point_ids))
	return contours


def _corr_point_model_location_for_layer_reason(
	point: dict,
	layer_index: int,
	shape: tuple[int, int],
) -> tuple[tuple[float, float] | None, str]:
	if not bool(point.get("valid", False)):
		return None, "point_invalid"
	err = point.get("winding_err", None)
	if err is None:
		return None, "missing_winding_err"
	try:
		err_f = float(err)
	except (TypeError, ValueError):
		return None, "bad_winding_err"
	if not math.isfinite(err_f):
		return None, "nonfinite_winding_err"
	locations = point.get("model_locations", [])
	if not isinstance(locations, list):
		return None, "missing_model_locations"
	best: tuple[float, float, float] | None = None
	saw_layer = False
	saw_bad_value = False
	saw_bad_weight = False
	saw_out_of_bounds = False
	hmax = float(shape[0] - 1)
	wmax = float(shape[1] - 1)
	for loc in locations:
		if not isinstance(loc, dict):
			saw_bad_value = True
			continue
		try:
			d = int(loc.get("d"))
			h = float(loc.get("h"))
			w = float(loc.get("w"))
			weight = float(loc.get("weight", 1.0))
			residual = float(loc.get("residual"))
		except (TypeError, ValueError):
			saw_bad_value = True
			continue
		if d != int(layer_index):
			continue
		saw_layer = True
		if not (
			math.isfinite(h)
			and math.isfinite(w)
			and math.isfinite(weight)
			and math.isfinite(residual)
		):
			saw_bad_value = True
			continue
		if weight <= 0.0:
			saw_bad_weight = True
			continue
		if h < 0.0 or w < 0.0 or h > hmax or w > wmax:
			saw_out_of_bounds = True
			continue
		if best is None or weight > best[0]:
			best = (weight, h, w)
	if best is None:
		if not locations:
			return None, "empty_model_locations"
		if not saw_layer:
			return None, "no_location_on_layer"
		if saw_out_of_bounds:
			return None, "location_out_of_bounds"
		if saw_bad_weight:
			return None, "nonpositive_location_weight"
		if saw_bad_value:
			return None, "bad_location_value"
		return None, "no_usable_location"
	return (best[1], best[2]), "ok"


def _corr_point_model_location_for_layer(
	point: dict,
	layer_index: int,
	shape: tuple[int, int],
) -> tuple[float, float] | None:
	loc, _reason = _corr_point_model_location_for_layer_reason(point, layer_index, shape)
	return loc


def _approval_output_mask_for_layer_with_debug(
	payload: dict,
	x: np.ndarray,
	y: np.ndarray,
	z: np.ndarray,
	*,
	layer_index: int,
	corr_results: dict | None,
	fit_config: dict | None,
) -> tuple[np.ndarray, np.ndarray, dict]:
	debug: dict = {
		"layer_index": int(layer_index),
		"mesh_shape": [int(x.shape[0]), int(x.shape[1])],
		"mesh_vertices": int(x.size),
		"payload_version": payload.get("version"),
		"payload_source": payload.get("source"),
		"dilation_radius": int(payload.get("dilation_radius", payload.get("dilate", 0))),
	}
	if not isinstance(corr_results, dict):
		raise ValueError("approval output mask requires _corr_points_results_ in the checkpoint")
	collection_ids = _approval_mask_corr_collection_ids(payload, fit_config)
	if not collection_ids:
		raise ValueError("approval output mask has no approval-inpaint corr collection id")
	debug["collection_ids"] = sorted(int(v) for v in collection_ids)
	points = []
	contours_rc: list[list[tuple[float, float]]] = []
	all_results = _corr_points_result_list(corr_results)
	lookup = _corr_points_result_lookup(corr_results)
	payload_contours = _approval_mask_corr_contours(payload, collection_ids)
	debug["corr_points_total"] = len(all_results)
	debug["payload_contours"] = len(payload_contours)
	n_seen = 0
	skip_reasons: dict[str, int] = {}
	sample_points: list[list[float]] = []
	if payload_contours:
		for cid, point_ids in payload_contours:
			contour_points: list[tuple[float, float]] = []
			for pid in point_ids:
				point = lookup.get((cid, pid))
				if point is None:
					skip_reasons["missing_point_id"] = skip_reasons.get("missing_point_id", 0) + 1
					continue
				n_seen += 1
				loc, reason = _corr_point_model_location_for_layer_reason(point, int(layer_index), x.shape)
				if loc is None:
					skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
					continue
				contour_points.append(loc)
				points.append(loc)
				if len(sample_points) < 12:
					sample_points.append([round(float(loc[0]), 3), round(float(loc[1]), 3)])
			if len(contour_points) >= 3:
				contours_rc.append(contour_points)
	else:
		for point in all_results:
			try:
				cid = int(point.get("collection_id"))
			except (TypeError, ValueError):
				continue
			if cid not in collection_ids:
				continue
			n_seen += 1
			loc, reason = _corr_point_model_location_for_layer_reason(point, int(layer_index), x.shape)
			if loc is None:
				skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
				continue
			points.append(loc)
			if len(sample_points) < 12:
				sample_points.append([round(float(loc[0]), 3), round(float(loc[1]), 3)])
		if len(points) >= 3:
			contours_rc.append(points)
	n_skipped = sum(skip_reasons.values())
	debug["corr_points_matching_collection"] = int(n_seen)
	debug["corr_points_usable"] = int(len(points))
	debug["usable_contours"] = int(len(contours_rc))
	debug["corr_points_skipped"] = int(n_skipped)
	debug["skip_reasons"] = dict(sorted(skip_reasons.items()))
	debug["usable_points_bbox_rc"] = _points_bbox(points)
	debug["usable_points_sample_rc"] = sample_points
	if n_skipped:
		print(
			f"[fit2tifxyz] approval output mask skipped {n_skipped}/{n_seen} "
			f"corr point(s) on layer {layer_index}: {debug['skip_reasons']}",
			flush=True,
		)
	if not contours_rc:
		print(
			f"[fit2tifxyz] WARNING: approval output mask has only {len(points)} usable "
			f"corr point(s) and {len(contours_rc)} drawable contour(s) on layer {layer_index}; masking the layer out",
			flush=True,
		)
		raw = np.zeros(x.shape, dtype=bool)
		debug["raw_true_vertices"] = 0
		debug["raw_bbox_rc"] = None
		debug["dilated_true_vertices"] = 0
		debug["dilated_bbox_rc"] = None
		return raw, raw.copy(), debug
	raw = _rasterize_contours(contours_rc, x.shape)
	mask = _dilate_chebyshev(raw, debug["dilation_radius"])
	debug["raw_true_vertices"] = int(raw.sum())
	debug["raw_bbox_rc"] = _mask_bbox(raw)
	debug["dilated_true_vertices"] = int(mask.sum())
	debug["dilated_bbox_rc"] = _mask_bbox(mask)
	if mask.size > 0:
		debug["raw_fraction"] = round(float(raw.sum()) / float(mask.size), 6)
		debug["dilated_fraction"] = round(float(mask.sum()) / float(mask.size), 6)
	if bool(mask.all()):
		print(
			f"[fit2tifxyz] WARNING: approval output mask layer {layer_index} keeps every vertex "
			f"(usable={len(points)} raw={int(raw.sum())} dilated={int(mask.sum())}/{mask.size})",
			flush=True,
		)
	return mask, raw, debug


def _approval_output_mask_for_layer(
	payload: dict,
	x: np.ndarray,
	y: np.ndarray,
	z: np.ndarray,
	*,
	layer_index: int,
	corr_results: dict | None,
	fit_config: dict | None,
) -> np.ndarray:
	mask, _raw, _debug = _approval_output_mask_for_layer_with_debug(
		payload, x, y, z,
		layer_index=layer_index, corr_results=corr_results, fit_config=fit_config,
	)
	return mask


def _apply_output_vertex_mask(
	x: np.ndarray,
	y: np.ndarray,
	z: np.ndarray,
	d: np.ndarray | None,
	mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
	if mask.shape != x.shape:
		raise ValueError(f"output mask shape mismatch: {mask.shape} vs {x.shape}")
	x_out = x.astype(np.float32, copy=True)
	y_out = y.astype(np.float32, copy=True)
	z_out = z.astype(np.float32, copy=True)
	x_out[~mask] = -1.0
	y_out[~mask] = -1.0
	z_out[~mask] = -1.0
	d_out = None
	if d is not None:
		d_out = d.astype(np.float32, copy=True)
		d_out[~mask] = -1.0
	return x_out, y_out, z_out, d_out


def _as_channel_numpy(value: object, *, name: str) -> np.ndarray:
	if isinstance(value, torch.Tensor):
		return value.detach().cpu().numpy().astype(np.float32, copy=False)
	return np.asarray(value, dtype=np.float32)


def _fit_config_flow_gate_channels_enabled(fit_config: dict | None) -> bool:
	if not isinstance(fit_config, dict):
		return False
	args = fit_config.get("args", {})
	if not isinstance(args, dict):
		return False
	return bool(args.get("tifxyz-flow-gate-channels", False))


def _flow_gate_source_config(payload: dict | None) -> dict:
	if not isinstance(payload, dict):
		return {}
	source = payload.get("source_config", {})
	return dict(source) if isinstance(source, dict) else {}


def _flow_gate_extra_channel_specs(payload: dict | None) -> list[dict]:
	source_config = _flow_gate_source_config(payload)
	return [
		{
			"name": "flow_gate_local_contrast",
			"file": "flow_gate_local_contrast.tif",
			"dtype": "float32",
			"range": [0.0, 1.0],
			"source": "pred_dt_flow_gate.local_contrast",
			"source_config": source_config,
		},
		{
			"name": "flow_gate_component_normalized",
			"file": "flow_gate_component_normalized.tif",
			"dtype": "float32",
			"range": [0.0, 1.0],
			"source": "pred_dt_flow_gate.component_normalized",
			"source_config": source_config,
		},
	]


def _flow_gate_payload_for_export(
	st: dict,
	fit_config: dict | None,
	mode: str,
) -> dict | None:
	mode = str(mode)
	if mode == "off":
		return None
	payload = st.get("_flow_gate_channels_", None)
	if not isinstance(payload, dict):
		if mode == "on":
			raise ValueError("flow-gate channel export requested, but checkpoint has no _flow_gate_channels_ payload")
		return None
	if mode == "auto" and not _fit_config_flow_gate_channels_enabled(fit_config):
		return None
	return payload


def _flow_gate_channel_arrays(payload: dict | None, *, shape_dhw: tuple[int, int, int]) -> dict[str, np.ndarray] | None:
	if payload is None:
		return None
	D, H, W = (int(v) for v in shape_dhw)
	channels: dict[str, np.ndarray] = {}
	for name in ("flow_gate_local_contrast", "flow_gate_component_normalized"):
		if name not in payload:
			raise ValueError(f"flow-gate channel payload missing {name!r}")
		arr = _as_channel_numpy(payload[name], name=name)
		if arr.shape == (H, W):
			arr = arr.reshape(1, H, W)
		if arr.shape != (D, H, W):
			raise ValueError(
				f"flow-gate channel {name!r} shape {arr.shape} does not match exported mesh shape {(D, H, W)}"
			)
		channels[name] = np.clip(arr.astype(np.float32, copy=False), 0.0, 1.0)
	return channels


def _corr_results_debug_summary(corr_results: dict | None) -> dict:
	if not isinstance(corr_results, dict):
		return {"present": False}
	points = _corr_points_result_list(corr_results)
	model_location_count = 0
	valid_count = 0
	collections: dict[str, int] = {}
	for point in points:
		try:
			cid = str(int(point.get("collection_id")))
		except (TypeError, ValueError):
			cid = "invalid"
		collections[cid] = collections.get(cid, 0) + 1
		if bool(point.get("valid", False)):
			valid_count += 1
		locations = point.get("model_locations", [])
		if isinstance(locations, list):
			model_location_count += sum(1 for loc in locations if isinstance(loc, dict))
	return {
		"present": True,
		"points_total": len(points),
		"points_valid": int(valid_count),
		"model_locations_total": int(model_location_count),
		"collections": dict(sorted(collections.items())),
	}


def _fit_config_mask_enabled(fit_config: dict | None) -> bool:
	if not isinstance(fit_config, dict):
		return False
	args = fit_config.get("args", {})
	if not isinstance(args, dict):
		return False
	return bool(args.get("approval-inpaint-output-mask", False))


def _write_mask_debug_artifacts(
	out_dir: Path,
	debug: dict,
	*,
	raw_mask: np.ndarray | None = None,
	mask: np.ndarray | None = None,
) -> None:
	(out_dir / "approval_inpaint_output_mask_debug.json").write_text(
		json.dumps(debug, indent=2) + "\n",
		encoding="utf-8",
	)
	if raw_mask is not None:
		tifffile.imwrite(
			str(out_dir / "approval_inpaint_output_mask_raw.tif"),
			raw_mask.astype(np.uint8) * 255,
			compression="lzw",
		)
	if mask is not None:
		tifffile.imwrite(
			str(out_dir / "approval_inpaint_output_mask_dilated.tif"),
			mask.astype(np.uint8) * 255,
			compression="lzw",
		)


def _print_mask_debug(debug: dict) -> None:
	print(
		"[fit2tifxyz] approval mask "
		f"layer={debug.get('layer_index')} "
		f"collections={debug.get('collection_ids')} "
		f"points={debug.get('corr_points_usable')}/{debug.get('corr_points_matching_collection')} "
		f"skipped={debug.get('corr_points_skipped')} "
		f"raw={debug.get('raw_true_vertices')}/{debug.get('mesh_vertices')} "
		f"dilated={debug.get('dilated_true_vertices')}/{debug.get('mesh_vertices')} "
		f"valid={debug.get('valid_vertices_before')}->{debug.get('valid_vertices_after')} "
		f"points_bbox={debug.get('usable_points_bbox_rc')} "
		f"mask_bbox={debug.get('dilated_bbox_rc')}",
		flush=True,
	)


def _write_tifxyz(*, out_dir: Path, x: np.ndarray, y: np.ndarray, z: np.ndarray,
				  scale: float, d: np.ndarray | None = None,
				  model_source: Path | None = None,
				  copy_model: bool = False, fit_config: dict | None = None,
				  job_spec: dict | None = None,
				  area: dict | None = None,
				  components: list[list[int]] | None = None,
				  base_shape_zyx: tuple[int, int, int] | None = None,
				  lasagna_base_shape_zyx: tuple[int, int, int] | None = None,
				  extra_channels: dict[str, np.ndarray] | None = None,
				  extra_channel_specs: list[dict] | None = None) -> None:
	out_dir.mkdir(parents=True, exist_ok=True)
	if x.shape != y.shape or x.shape != z.shape:
		raise ValueError("x/y/z must have identical shapes")
	if x.ndim != 2:
		raise ValueError("x/y/z must be 2D")

	xf = x.astype(np.float32, copy=False)
	yf = y.astype(np.float32, copy=False)
	zf = z.astype(np.float32, copy=False)
	valid = np.isfinite(xf) & np.isfinite(yf) & np.isfinite(zf)
	valid &= ~((xf == -1.0) & (yf == -1.0) & (zf == -1.0))
	if np.any(valid):
		bbox = [
			[float(np.nanmin(xf[valid])), float(np.nanmin(yf[valid])), float(np.nanmin(zf[valid]))],
			[float(np.nanmax(xf[valid])), float(np.nanmax(yf[valid])), float(np.nanmax(zf[valid]))],
		]
	else:
		bbox = [[-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0]]

	meta = {
		"uuid": str(out_dir.name),
		"type": "seg",
		"format": "tifxyz",
		"scale": [float(scale), float(scale)],
		"bbox": _bbox_for_xyz(xf, yf, zf, out_dir=out_dir),
	}
	if base_shape_zyx is not None:
		meta["base_shape_zyx"] = [int(v) for v in base_shape_zyx]
	if lasagna_base_shape_zyx is not None:
		meta["lasagna_base_shape_zyx"] = [int(v) for v in lasagna_base_shape_zyx]
	if components is not None:
		meta["components"] = components
	if area is not None:
		meta.update(area)
	if model_source is not None:
		meta["model_source"] = str(model_source)
	if fit_config is not None:
		meta["fit_config"] = fit_config
	if job_spec is not None:
		meta["lasagna_job"] = job_spec
	if extra_channels:
		meta["extra_channels"] = extra_channel_specs or [
			{
				"name": name,
				"file": f"{name}.tif",
				"dtype": "float32",
				"range": [0.0, 1.0],
			}
			for name in sorted(extra_channels)
		]
	(out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
	tifffile.imwrite(str(out_dir / "x.tif"), xf, compression=None)
	tifffile.imwrite(str(out_dir / "y.tif"), yf, compression=None)
	tifffile.imwrite(str(out_dir / "z.tif"), zf, compression=None)
	if d is not None:
		tifffile.imwrite(str(out_dir / "d.tif"), d.astype(np.float32, copy=False), compression=None)
	if extra_channels:
		spec_by_name = {
			str(spec.get("name")): spec
			for spec in (extra_channel_specs or [])
			if isinstance(spec, dict) and spec.get("name") is not None
		}
		for name, values in extra_channels.items():
			arr = np.asarray(values, dtype=np.float32)
			if arr.shape != xf.shape:
				raise ValueError(f"extra channel {name!r} shape mismatch: {arr.shape} vs {xf.shape}")
			arr = np.clip(arr, 0.0, 1.0).astype(np.float32, copy=False)
			arr = np.where(valid, arr, 0.0).astype(np.float32, copy=False)
			spec = spec_by_name.get(name, {})
			file_name = str(spec.get("file", f"{name}.tif"))
			tifffile.imwrite(str(out_dir / file_name), arr, compression=None)

	if model_source is not None:
		dest = out_dir / "model.pt"
		if dest.is_symlink() or dest.exists():
			dest.unlink()
		if copy_model:
			shutil.copy2(str(model_source.resolve()), str(dest))
		else:
			dest.symlink_to(model_source.resolve())


def _as_numpy_float32(value: object, *, name: str) -> np.ndarray:
	if isinstance(value, torch.Tensor):
		return value.detach().cpu().numpy().astype(np.float32, copy=False)
	arr = np.asarray(value, dtype=np.float32)
	if arr.size == 0:
		raise ValueError(f"flatten checkpoint field {name!r} is empty")
	return arr


def _export_flatten_checkpoint(
	*,
	st: dict,
	cfg: ExportConfig,
	model_params: dict | None,
	fit_config: dict | None,
	job_spec: dict | None,
	export_factor: float = 1.0,
	lasagna_base_shape_zyx: tuple[int, int, int] | None = None,
	output_base_shape_zyx: tuple[int, int, int] | None = None,
) -> int:
	map_yx = _as_numpy_float32(st["flatten_map_flat"], name="flatten_map_flat")
	if map_yx.ndim != 3 or map_yx.shape[-1] != 2:
		raise ValueError("flatten_map_flat must have shape (H, W, 2)")

	mesh = st.get("mesh_flat")
	if mesh is None:
		dev = torch.device(cfg.device)
		mdl = model.Model3D.from_checkpoint(st, device=dev)
		mesh_np = mdl.mesh_coarse().detach().cpu().numpy().astype(np.float32, copy=False)
	else:
		mesh_np = _as_numpy_float32(mesh, name="mesh_flat")
	if mesh_np.ndim != 4 or mesh_np.shape[0] != 3 or mesh_np.shape[1] != 1:
		raise ValueError("flatten mesh_flat must have shape (3, 1, H, W)")

	x = mesh_np[0, 0].astype(np.float32, copy=False)
	y = mesh_np[1, 0].astype(np.float32, copy=False)
	z = mesh_np[2, 0].astype(np.float32, copy=False)
	if x.shape != tuple(map_yx.shape[:2]):
		raise ValueError("flatten_map_flat shape does not match mesh_flat shape")

	point_mask = st.get("flatten_point_mask")
	if point_mask is not None:
		if isinstance(point_mask, torch.Tensor):
			mask_np = point_mask.detach().cpu().numpy().astype(bool, copy=False)
		else:
			mask_np = np.asarray(point_mask, dtype=bool)
		if mask_np.shape != x.shape:
			raise ValueError("flatten_point_mask shape does not match mesh_flat shape")
		x = np.where(mask_np, x, -1.0).astype(np.float32, copy=False)
		y = np.where(mask_np, y, -1.0).astype(np.float32, copy=False)
		z = np.where(mask_np, z, -1.0).astype(np.float32, copy=False)

	mesh_step = 100
	if model_params is not None:
		mesh_step = int(model_params.get("mesh_step", 100))
	xy_step_fullres = float(mesh_step)
	xy_step_export = xy_step_fullres * float(export_factor)
	meta_scale = 1.0 / xy_step_export

	out_base = Path(cfg.output)
	out_base.mkdir(parents=True, exist_ok=True)

	valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
	valid &= ~((x == -1.0) & (y == -1.0) & (z == -1.0))
	d = np.where(valid, 0.0, -1.0).astype(np.float32, copy=False)
	seg_name = cfg.output_name if cfg.output_name else f"{cfg.prefix}0000.tifxyz"
	out_dir = out_base / seg_name
	x, y, z = _scaled_xyz_for_export(x, y, z, export_factor)
	area = _get_area(x, y, z, xy_step_export, cfg.voxel_size_um)
	_write_tifxyz(
		out_dir=out_dir,
		x=x,
		y=y,
		z=z,
		d=d,
		scale=meta_scale,
		model_source=Path(cfg.input),
		copy_model=cfg.copy_model,
		fit_config=fit_config,
		job_spec=job_spec,
		area=area,
		base_shape_zyx=output_base_shape_zyx,
		lasagna_base_shape_zyx=lasagna_base_shape_zyx,
	)
	if model_params is not None:
		(out_dir / "model_params.json").write_text(json.dumps(model_params, indent=2) + "\n", encoding="utf-8")
	_print_area(area)
	print(f"[fit2tifxyz] exported {out_dir.name}", flush=True)
	return 0


def main(argv: list[str] | None = None, *, cancel_fn=None) -> int:
	def _check_cancel() -> None:
		if cancel_fn is not None:
			cancel_fn()

	parser = _build_parser()
	args = cli_json.parse_args(parser, argv)
	cfg = ExportConfig(
		input=str(args.input),
		output=str(args.output),
		prefix=str(args.prefix),
		single_segment=bool(args.single_segment),
		copy_model=bool(args.copy_model),
		output_name=None if args.output_name in (None, "") else str(args.output_name),
		voxel_size_um=args.voxel_size_um,
		target_volume_shape_zyx=(
			None if args.target_volume_shape_zyx is None
			else tuple(int(v) for v in args.target_volume_shape_zyx)
		),
		flow_gate_channels=str(args.flow_gate_channels),
	)

	dev = torch.device(cfg.device)
	_check_cancel()
	st = torch.load(cfg.input, map_location=dev, weights_only=False)
	_check_cancel()
	if not isinstance(st, dict):
		raise ValueError("expected a state_dict checkpoint")
	model_params = st.get("_model_params_", None)
	if not isinstance(model_params, dict):
		model_params = None
	fit_config = st.get("_fit_config_", None)
	if not isinstance(fit_config, dict):
		fit_config = None
	job_spec = st.get("_job_spec_", None)
	if not isinstance(job_spec, dict):
		job_spec = None
	corr_points_results = st.get("_corr_points_results_", None)
	if not isinstance(corr_points_results, dict):
		corr_points_results = None
	approval_output_mask = st.get("_approval_inpaint_output_mask_", None)
	if not isinstance(approval_output_mask, dict):
		approval_output_mask = None
	flow_gate_payload = _flow_gate_payload_for_export(st, fit_config, cfg.flow_gate_channels)
	export_factor, lasagna_base_shape_zyx, output_base_shape_zyx = _export_coordinate_scale(
		model_params=model_params,
		target_volume_shape_zyx=cfg.target_volume_shape_zyx,
	)
	if "flatten_map_flat" in st:
		_check_cancel()
		return _export_flatten_checkpoint(
		st=st,
		cfg=cfg,
		model_params=model_params,
		fit_config=fit_config,
		job_spec=job_spec,
		export_factor=export_factor,
		lasagna_base_shape_zyx=lasagna_base_shape_zyx,
		output_base_shape_zyx=output_base_shape_zyx,
	)

	# Reconstruct mesh (3, D, Hm, Wm) — pyramid stores full xyz positions
	_check_cancel()
	mdl = model.Model3D.from_checkpoint(st, device=dev)
	mesh = mdl.mesh_coarse()
	_check_cancel()

	_, D, Hm, Wm = (int(v) for v in mesh.shape)
	mesh_np = mesh.detach().cpu().numpy()  # (3, D, Hm, Wm)
	flow_gate_channels = _flow_gate_channel_arrays(
		flow_gate_payload,
		shape_dhw=(D, Hm, Wm),
	)
	flow_gate_channel_specs = (
		_flow_gate_extra_channel_specs(flow_gate_payload)
		if flow_gate_channels is not None
		else None
	)

	mesh_step = 100
	if model_params is not None:
		mesh_step = int(model_params.get("mesh_step", 100))
	xy_step_fullres = float(mesh_step)
	xy_step_export = xy_step_fullres * float(export_factor)
	meta_scale = 1.0 / xy_step_export

	out_base = Path(cfg.output)
	out_base.mkdir(parents=True, exist_ok=True)

	BORDER_W = 2

	print(f"[fit2tifxyz] exporting D={D} Hm={Hm} Wm={Wm}, mesh stored in Lasagna base coords"
		  f", voxel_size_um={cfg.voxel_size_um}")
	if approval_output_mask is not None:
		collection_ids = approval_output_mask.get("corr_collection_ids", [])
		radius = int(approval_output_mask.get("dilation_radius", approval_output_mask.get("dilate", 0)))
		corr_summary = _corr_results_debug_summary(corr_points_results)
		print(
			f"[fit2tifxyz] approval-inpaint output mask: source=corr_points "
			f"collections={collection_ids} dilate={radius} corr_results={corr_summary}",
			flush=True,
		)
	elif _fit_config_mask_enabled(fit_config):
		print(
			"[fit2tifxyz] WARNING: fit_config says approval-inpaint output masking was enabled, "
			"but checkpoint has no _approval_inpaint_output_mask_ payload; export is unmasked",
			flush=True,
		)

	if cfg.single_segment:
		# Combine all depth layers horizontally
		total_w = Wm * D + max(0, D - 1) * BORDER_W
		x_all = np.full((Hm, total_w), -1.0, dtype=np.float32)
		y_all = np.full((Hm, total_w), -1.0, dtype=np.float32)
		z_all = np.full((Hm, total_w), -1.0, dtype=np.float32)
		d_all = np.full((Hm, total_w), -1.0, dtype=np.float32)
		extra_all: dict[str, np.ndarray] | None = None
		if flow_gate_channels is not None:
			extra_all = {
				name: np.zeros((Hm, total_w), dtype=np.float32)
				for name in flow_gate_channels
			}

		col = 0
		components: list[list[int]] = []
		mask_debug_layers: list[dict] = []
		raw_mask_all = np.zeros((Hm, total_w), dtype=bool)
		mask_all = np.zeros((Hm, total_w), dtype=bool)
		for d in range(D):
			_check_cancel()
			x_layer = mesh_np[0, d]  # (Hm, Wm)
			y_layer = mesh_np[1, d]
			z_layer = mesh_np[2, d]
			d_layer = np.full((Hm, Wm), float(d), dtype=np.float32)
			if approval_output_mask is not None:
				valid_before = int(_valid_xyz_mask(x_layer, y_layer, z_layer).sum())
				mask, raw_mask, mask_debug = _approval_output_mask_for_layer_with_debug(
					approval_output_mask, x_layer, y_layer, z_layer,
					layer_index=d, corr_results=corr_points_results, fit_config=fit_config,
				)
				x_layer, y_layer, z_layer, d_layer = _apply_output_vertex_mask(
					x_layer, y_layer, z_layer, d_layer, mask
				)
				mask_debug["valid_vertices_before"] = valid_before
				mask_debug["valid_vertices_after"] = int(_valid_xyz_mask(x_layer, y_layer, z_layer).sum())
				mask_debug["single_segment_col_range"] = [int(col), int(col + Wm)]
				_print_mask_debug(mask_debug)
				mask_debug_layers.append(mask_debug)
				raw_mask_all[:, col:col + Wm] = raw_mask
				mask_all[:, col:col + Wm] = mask
			else:
				valid_layer = _valid_xyz_mask(x_layer, y_layer, z_layer)
				d_layer = np.where(valid_layer, float(d), -1.0).astype(np.float32)
			x_all[:, col:col + Wm] = x_layer
			y_all[:, col:col + Wm] = y_layer
			z_all[:, col:col + Wm] = z_layer
			d_all[:, col:col + Wm] = d_layer
			if extra_all is not None and flow_gate_channels is not None:
				valid_layer = _valid_xyz_mask(x_layer, y_layer, z_layer)
				for name, arr in flow_gate_channels.items():
					ch_layer = np.clip(arr[d], 0.0, 1.0).astype(np.float32, copy=False)
					extra_all[name][:, col:col + Wm] = np.where(valid_layer, ch_layer, 0.0)
			components.append([col, col + Wm])
			col += Wm + BORDER_W

		_check_cancel()
		seg_name = cfg.output_name if cfg.output_name else f"{cfg.prefix}.tifxyz"
		out_dir = out_base / seg_name
		x_all, y_all, z_all = _scaled_xyz_for_export(x_all, y_all, z_all, export_factor)
		area = _get_area(x_all, y_all, z_all, xy_step_export, cfg.voxel_size_um)
		_write_tifxyz(out_dir=out_dir, x=x_all, y=y_all, z=z_all, d=d_all, scale=meta_scale,
					  model_source=Path(cfg.input), copy_model=cfg.copy_model, fit_config=fit_config,
					  job_spec=job_spec,
					  area=area, components=components if D > 1 else None,
					  base_shape_zyx=output_base_shape_zyx,
					  lasagna_base_shape_zyx=lasagna_base_shape_zyx,
					  extra_channels=extra_all,
					  extra_channel_specs=flow_gate_channel_specs)
		if approval_output_mask is not None:
			_write_mask_debug_artifacts(
				out_dir,
				{
					"mode": "single_segment",
					"payload": approval_output_mask,
					"corr_results": _corr_results_debug_summary(corr_points_results),
					"layers": mask_debug_layers,
				},
				raw_mask=raw_mask_all,
				mask=mask_all,
			)
		_print_area(area)
		if model_params is not None:
			(out_dir / "model_params.json").write_text(json.dumps(model_params, indent=2) + "\n", encoding="utf-8")
		if corr_points_results is not None:
			(out_dir / "corr_points_results.json").write_text(json.dumps(corr_points_results, indent=2) + "\n", encoding="utf-8")
	else:
		# One tifxyz per depth layer (winding)
		total_area = {"area_vx2": 0.0}
		if cfg.voxel_size_um is not None:
			total_area["area_cm2"] = 0.0
		for d in range(D):
			_check_cancel()
			x = mesh_np[0, d]  # (Hm, Wm) already in fullres
			y = mesh_np[1, d]
			z = mesh_np[2, d]
			mask_debug = None
			raw_mask = None
			mask = None
			if approval_output_mask is not None:
				valid_before = int(_valid_xyz_mask(x, y, z).sum())
				mask, raw_mask, mask_debug = _approval_output_mask_for_layer_with_debug(
					approval_output_mask, x, y, z,
					layer_index=d, corr_results=corr_points_results, fit_config=fit_config,
				)
				x, y, z, _ = _apply_output_vertex_mask(x, y, z, None, mask)
				mask_debug["valid_vertices_before"] = valid_before
				mask_debug["valid_vertices_after"] = int(_valid_xyz_mask(x, y, z).sum())
				_print_mask_debug(mask_debug)
			valid = _valid_xyz_mask(x, y, z)
			extra_layer = None
			if flow_gate_channels is not None:
				extra_layer = {
					name: np.where(valid, np.clip(arr[d], 0.0, 1.0), 0.0).astype(np.float32, copy=False)
					for name, arr in flow_gate_channels.items()
				}
			d_layer = np.where(valid, float(d), -1.0).astype(np.float32)
			x, y, z = _scaled_xyz_for_export(x, y, z, export_factor)
			area = _get_area(x, y, z, xy_step_export, cfg.voxel_size_um)
			total_area["area_vx2"] += area["area_vx2"]
			if "area_cm2" in area:
				total_area["area_cm2"] += area["area_cm2"]
			out_dir = out_base / f"{cfg.prefix}{d:04d}.tifxyz"
			_check_cancel()
			_write_tifxyz(out_dir=out_dir, x=x, y=y, z=z, d=d_layer, scale=meta_scale,
						  model_source=Path(cfg.input), copy_model=cfg.copy_model, fit_config=fit_config,
						  job_spec=job_spec,
						  area=area,
						  base_shape_zyx=output_base_shape_zyx,
						  lasagna_base_shape_zyx=lasagna_base_shape_zyx,
						  extra_channels=extra_layer,
						  extra_channel_specs=flow_gate_channel_specs)
			if approval_output_mask is not None and mask_debug is not None:
				_write_mask_debug_artifacts(
					out_dir,
					{
						"mode": "per_layer",
						"payload": approval_output_mask,
						"corr_results": _corr_results_debug_summary(corr_points_results),
						"layer": mask_debug,
					},
					raw_mask=raw_mask,
					mask=mask,
				)
			if model_params is not None:
				(out_dir / "model_params.json").write_text(json.dumps(model_params, indent=2) + "\n", encoding="utf-8")
			if corr_points_results is not None:
				(out_dir / "corr_points_results.json").write_text(json.dumps(corr_points_results, indent=2) + "\n", encoding="utf-8")
		_print_area(total_area)

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
