"""Versioned, source-grid UV warm-start sidecars for flatten-only fits."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import tifffile


SCHEMA_VERSION = 4
LEGACY_SCHEMA_VERSION = 3
KIND = "lasagna_source_grid_flatten_uv"
ROW_FILENAME = "flatten-uv-row.tif"
COL_FILENAME = "flatten-uv-col.tif"
VALID_FILENAME = "flatten-uv-valid.tif"
CELL_VALID_FILENAME = "flatten-uv-cell-valid.tif"
METADATA_FILENAME = "flatten-uv.json"


class FlattenUvError(ValueError):
	"""A UV sidecar is absent, corrupt, incompatible, or folded."""


def canonical_grid_fingerprint(
	shape: Sequence[int], *, source_step: float, winding_column_ranges=(),
) -> str:
	"""Fingerprint the stable canonical grid, deliberately excluding XYZ.

	XYZ changes at every fit iteration.  Shape, source sampling, and the
	canonical winding-to-column layout identify the parameter grid whose UVs
	may safely be reused.
	"""
	if len(shape) != 2:
		raise FlattenUvError("source shape must contain rows and columns")
	ranges = list(winding_column_ranges or ([[0, int(shape[1])]]))
	payload = {
		"rows": int(shape[0]),
		"columns": int(shape[1]),
		"source_step": float(source_step),
		"winding_column_ranges": ranges,
	}
	encoded = json.dumps(
		payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
	).encode("utf-8")
	return hashlib.sha256(encoded).hexdigest()


def _normalize_layout(
	ranges,
	winding_ids,
	*,
	width: int,
	label: str,
) -> tuple[list[list[int]], list[int]]:
	if ranges is None:
		ranges = [[0, int(width)]]
	if not isinstance(ranges, Sequence) or isinstance(ranges, (str, bytes)):
		raise FlattenUvError(f"{label} winding column ranges must be a sequence")
	normalized: list[list[int]] = []
	cursor = 0
	for index, value in enumerate(ranges):
		if (not isinstance(value, Sequence) or isinstance(value, (str, bytes))
				or len(value) != 2):
			raise FlattenUvError(
				f"{label} winding column range {index} must contain begin/end")
		try:
			begin, end = int(value[0]), int(value[1])
		except (TypeError, ValueError) as exc:
			raise FlattenUvError(
				f"{label} winding column range {index} is invalid") from exc
		if begin != cursor or end - begin < 2:
			raise FlattenUvError(
				f"{label} winding column ranges must be contiguous and "
				"contain at least two samples per winding")
		normalized.append([begin, end])
		cursor = end
	if not normalized or cursor != int(width):
		raise FlattenUvError(
			f"{label} winding column ranges do not cover width {int(width)}")
	if winding_ids is None:
		winding_ids = list(range(len(normalized)))
	try:
		ids = [int(value) for value in winding_ids]
	except (TypeError, ValueError) as exc:
		raise FlattenUvError(f"{label} winding ids are invalid") from exc
	if len(ids) != len(normalized) or len(set(ids)) != len(ids):
		raise FlattenUvError(
			f"{label} winding ids must uniquely match the column ranges")
	return normalized, ids


def _cell_determinants(uv: np.ndarray) -> np.ndarray:
	m00 = uv[:-1, :-1]
	m10 = uv[1:, :-1]
	m01 = uv[:-1, 1:]
	m11 = uv[1:, 1:]
	a0 = m10 - m00
	b0 = m01 - m00
	a1 = m11 - m10
	b1 = m01 - m10
	det0 = a0[..., 0] * b0[..., 1] - a0[..., 1] * b0[..., 0]
	det1 = a1[..., 0] * b1[..., 1] - a1[..., 1] * b1[..., 0]
	return np.minimum(det0, det1)


def _supported_cells(
	uv: np.ndarray,
	valid: np.ndarray | None = None,
	cell_valid: np.ndarray | None = None,
) -> np.ndarray:
	shape = uv.shape[:2]
	if valid is not None:
		valid = np.asarray(valid, dtype=np.bool_)
		if valid.shape != shape:
			raise FlattenUvError(
				f"flatten UV validity shape {valid.shape} does not match "
				f"source {shape}")
		supported = (
			valid[:-1, :-1] & valid[1:, :-1]
			& valid[:-1, 1:] & valid[1:, 1:])
	else:
		supported = np.ones(
			(max(0, shape[0] - 1), max(0, shape[1] - 1)),
			dtype=np.bool_)
	if cell_valid is not None:
		cell_valid = np.asarray(cell_valid, dtype=np.bool_)
		if cell_valid.shape != supported.shape:
			raise FlattenUvError(
				f"flatten UV cell validity shape {cell_valid.shape} does not "
				f"match source cells {supported.shape}")
		supported &= cell_valid
	return supported


def _topology_stats(
	uv: np.ndarray,
	valid: np.ndarray | None = None,
	cell_valid: np.ndarray | None = None,
) -> tuple[int, float, int]:
	det = _cell_determinants(uv)
	det = det[_supported_cells(uv, valid, cell_valid)]
	if det.size == 0:
		raise FlattenUvError("flatten UV has no source-supported cells")
	if not np.isfinite(det).all():
		raise FlattenUvError("flatten UV topology is not finite")
	return int(np.count_nonzero(det <= 0.0)), float(det.min()), int(det.size)


def validate_uv(
	uv: np.ndarray,
	expected_shape: Sequence[int],
	*,
	valid: np.ndarray | None = None,
	cell_valid: np.ndarray | None = None,
) -> np.ndarray:
	uv = np.asarray(uv, dtype=np.float32)
	expected = (int(expected_shape[0]), int(expected_shape[1]), 2)
	if uv.shape != expected:
		raise FlattenUvError(
			f"flatten UV shape {uv.shape} does not match source {expected}")
	if not np.isfinite(uv).all():
		raise FlattenUvError("flatten UV contains non-finite values")
	# Invalid vertices are deliberately extrapolated so the sidecars cover the
	# complete canonical grid. Their cells are not part of the source surface
	# and may cross; topology is enforced on source-supported quads.
	folds, minimum, _cell_count = _topology_stats(uv, valid, cell_valid)
	if folds:
		raise FlattenUvError(
			f"flatten UV has {folds} folded or degenerate source cells "
			f"(minimum determinant {minimum:.9g})")
	return np.ascontiguousarray(uv)


def _extrapolate_invalid_uv(
	uv: np.ndarray, valid: np.ndarray | None,
) -> np.ndarray:
	"""Give unsupported vertices a stable continuation of nearby fitted UVs."""
	if valid is None:
		return np.ascontiguousarray(uv)
	valid = np.asarray(valid, dtype=np.bool_)
	if valid.shape != uv.shape[:2]:
		raise FlattenUvError(
			f"flatten UV validity shape {valid.shape} does not match "
			f"source {uv.shape[:2]}")
	if bool(valid.all()):
		return np.ascontiguousarray(uv)
	out = np.array(uv, dtype=np.float32, copy=True, order="C")
	height, width = valid.shape
	row_coordinates = np.arange(height, dtype=np.float32)
	nonempty_columns = np.flatnonzero(valid.any(axis=0))
	for column in nonempty_columns:
		good = np.flatnonzero(valid[:, column])
		if good.size == 1:
			out[~valid[:, column], column] = out[good[0], column]
			continue
		for component in range(2):
			values = out[good, column, component]
			filled = np.interp(row_coordinates, good, values).astype(np.float32)
			top = int(good[0])
			if top:
				slope = ((float(values[1]) - float(values[0]))
						 / float(good[1] - good[0]))
				filled[:top] = (
					float(values[0]) + (row_coordinates[:top] - float(top)) * slope)
			bottom = int(good[-1])
			if bottom + 1 < height:
				slope = ((float(values[-1]) - float(values[-2]))
						 / float(good[-1] - good[-2]))
				filled[bottom + 1:] = (
					float(values[-1])
					+ (row_coordinates[bottom + 1:] - float(bottom)) * slope)
			out[~valid[:, column], column, component] = (
				filled[~valid[:, column]])

	# Entirely unsupported columns are rare boundary gaps.  Continue the
	# nearest supported columns horizontally rather than retaining arbitrary
	# optimizer pyramid values.
	empty_columns = np.flatnonzero(~valid.any(axis=0))
	if empty_columns.size:
		if nonempty_columns.size == 0:
			raise FlattenUvError("flatten UV has no source-supported vertices")
		for row_index in range(height):
			for component in range(2):
				out[row_index, empty_columns, component] = np.interp(
					empty_columns, nonempty_columns,
					out[row_index, nonempty_columns, component]).astype(np.float32)
	return out


def write_sidecars(
	directory: str | Path,
	uv: np.ndarray,
	*,
	fingerprint: str,
	source_step: float,
	output_step: float,
	valid: np.ndarray,
	cell_valid: np.ndarray | None = None,
	winding_column_ranges=None,
	winding_ids=None,
	sampling_dr_per_winding: float = 1.0,
) -> Path:
	root = Path(directory)
	root.mkdir(parents=True, exist_ok=True)
	valid = np.asarray(valid, dtype=np.bool_)
	if valid.shape != uv.shape[:2]:
		raise FlattenUvError("flatten UV validity mask shape does not match UV")
	uv = np.asarray(uv, dtype=np.float32)
	expected = (int(valid.shape[0]), int(valid.shape[1]), 2)
	if uv.shape != expected:
		raise FlattenUvError(
			f"flatten UV shape {uv.shape} does not match source {expected}")
	if not np.isfinite(uv).all():
		raise FlattenUvError("flatten UV contains non-finite values")
	source_cells = _supported_cells(uv, valid, cell_valid)
	determinants = _cell_determinants(uv)
	source_determinants = determinants[source_cells]
	if source_determinants.size == 0:
		raise FlattenUvError("flatten UV has no source-supported cells")
	if not np.isfinite(source_determinants).all():
		raise FlattenUvError("flatten UV topology is not finite")
	folded_cells = source_cells & (determinants <= 0.0)
	excluded_folded_cell_count = int(np.count_nonzero(folded_cells))
	cell_valid = np.ascontiguousarray(source_cells & ~folded_cells)
	uv = validate_uv(
		uv, uv.shape[:2], valid=valid, cell_valid=cell_valid)
	_folds, minimum_det, topology_cell_count = _topology_stats(
		uv, valid, cell_valid)
	uv = _extrapolate_invalid_uv(uv, valid)
	if not fingerprint:
		raise FlattenUvError("canonical-grid fingerprint is empty")
	ranges, ids = _normalize_layout(
		winding_column_ranges, winding_ids,
		width=int(uv.shape[1]), label="flatten UV")
	dr = float(sampling_dr_per_winding)
	if not math.isfinite(dr) or dr <= 0.0:
		raise FlattenUvError("flatten UV sampling_dr_per_winding must be positive")
	tifffile.imwrite(root / ROW_FILENAME, uv[..., 0], compression=None)
	tifffile.imwrite(root / COL_FILENAME, uv[..., 1], compression=None)
	tifffile.imwrite(root / VALID_FILENAME, valid.astype(np.uint8), compression=None)
	tifffile.imwrite(
		root / CELL_VALID_FILENAME, cell_valid.astype(np.uint8), compression=None)
	metadata = {
		"schema_version": SCHEMA_VERSION,
		"kind": KIND,
		"source_shape": [int(uv.shape[0]), int(uv.shape[1])],
		"canonical_grid_fingerprint": str(fingerprint),
		"winding_column_ranges": ranges,
		"winding_ids": ids,
		"sampling_dr_per_winding": dr,
		"source_step": float(source_step),
		"output_step": float(output_step),
		"row_file": ROW_FILENAME,
		"column_file": COL_FILENAME,
		"valid_file": VALID_FILENAME,
		"cell_valid_file": CELL_VALID_FILENAME,
		"dtype": "float32",
		"covers_complete_source_grid": True,
		"invalid_vertex_extrapolation": "column-linear-v1",
		"topology_validation": {
			"scope": "retained-source-supported-quads",
			"cell_count": topology_cell_count,
			"minimum_determinant": minimum_det,
			"folded_cell_count": 0,
			"source_supported_cell_count": int(source_determinants.size),
			"excluded_folded_cell_count": excluded_folded_cell_count,
			"minimum_source_determinant": float(source_determinants.min()),
		},
	}
	path = root / METADATA_FILENAME
	path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
	return path


def load_sidecars(
	metadata_path: str | Path,
	*,
	expected_source_step: float | None = None,
	expected_output_step: float | None = None,
	expected_winding_ids=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
	path = Path(metadata_path)
	try:
		metadata = json.loads(path.read_text(encoding="utf-8"))
	except (OSError, UnicodeError, json.JSONDecodeError) as exc:
		raise FlattenUvError(f"cannot read flatten UV metadata: {exc}") from exc
	if not isinstance(metadata, Mapping):
		raise FlattenUvError("flatten UV metadata must be an object")
	schema_version = metadata.get("schema_version")
	if (schema_version not in (LEGACY_SCHEMA_VERSION, SCHEMA_VERSION)
			or metadata.get("kind") != KIND):
		raise FlattenUvError("unsupported flatten UV metadata schema")
	shape = metadata.get("source_shape")
	if (not isinstance(shape, list) or len(shape) != 2
			or int(shape[0]) < 2 or int(shape[1]) < 2):
		raise FlattenUvError("flatten UV source shape is invalid")
	source_shape = (int(shape[0]), int(shape[1]))
	source_ranges, source_ids = _normalize_layout(
		metadata.get("winding_column_ranges"), metadata.get("winding_ids"),
		width=source_shape[1], label="stored flatten UV")
	if expected_winding_ids is not None and source_ids != [
			int(value) for value in expected_winding_ids]:
		raise FlattenUvError("flatten UV winding ids do not match the current source")
	try:
		source_dr = float(metadata["sampling_dr_per_winding"])
	except (KeyError, TypeError, ValueError) as exc:
		raise FlattenUvError(
			"flatten UV sampling_dr_per_winding is invalid") from exc
	if not math.isfinite(source_dr) or source_dr <= 0.0:
		raise FlattenUvError(
			"flatten UV sampling_dr_per_winding must be finite and positive")
	if metadata.get("covers_complete_source_grid") is not True:
		raise FlattenUvError("flatten UV does not cover the complete source grid")
	if metadata.get("dtype") != "float32":
		raise FlattenUvError("flatten UV metadata dtype must be float32")
	if (metadata.get("row_file") != ROW_FILENAME
			or metadata.get("column_file") != COL_FILENAME
			or metadata.get("valid_file") != VALID_FILENAME):
		raise FlattenUvError("flatten UV metadata has unexpected sidecar filenames")
	if (schema_version == SCHEMA_VERSION
			and metadata.get("cell_valid_file") != CELL_VALID_FILENAME):
		raise FlattenUvError(
			"flatten UV metadata has an unexpected cell-valid filename")
	try:
		row = tifffile.imread(path.parent / str(metadata["row_file"]))
		col = tifffile.imread(path.parent / str(metadata["column_file"]))
		valid = tifffile.imread(path.parent / str(metadata["valid_file"]))
		cell_valid = (
			tifffile.imread(path.parent / str(metadata["cell_valid_file"]))
			if schema_version == SCHEMA_VERSION else None)
	except (KeyError, OSError, ValueError) as exc:
		raise FlattenUvError(f"cannot read flatten UV TIFF sidecars: {exc}") from exc
	if row.shape != col.shape:
		raise FlattenUvError("flatten UV row/column TIFF shapes do not match")
	if row.dtype != np.float32 or col.dtype != np.float32:
		raise FlattenUvError("flatten UV TIFF sidecars must use float32 samples")
	if tuple(row.shape) != source_shape:
		raise FlattenUvError(
			f"flatten UV TIFF shape {row.shape} does not match metadata {source_shape}")
	if not np.isfinite(row).all() or not np.isfinite(col).all():
		raise FlattenUvError("flatten UV contains non-finite values")
	if tuple(valid.shape) != source_shape or valid.dtype != np.uint8:
		raise FlattenUvError("flatten UV validity TIFF is invalid")
	if not np.isin(valid, (0, 1)).all():
		raise FlattenUvError("flatten UV validity TIFF is not binary")
	valid = valid.astype(np.bool_)
	expected_cell_shape = (source_shape[0] - 1, source_shape[1] - 1)
	if schema_version == SCHEMA_VERSION:
		if (tuple(cell_valid.shape) != expected_cell_shape
				or cell_valid.dtype != np.uint8):
			raise FlattenUvError("flatten UV cell-validity TIFF is invalid")
		if not np.isin(cell_valid, (0, 1)).all():
			raise FlattenUvError(
				"flatten UV cell-validity TIFF is not binary")
		cell_valid = cell_valid.astype(np.bool_)
	else:
		cell_valid = _supported_cells(
			np.empty((*source_shape, 2), dtype=np.float32), valid)
	for name in ("source_step", "output_step"):
		try:
			value = float(metadata[name])
		except (KeyError, TypeError, ValueError) as exc:
			raise FlattenUvError(f"flatten UV {name} is invalid") from exc
		if not math.isfinite(value) or value <= 0.0:
			raise FlattenUvError(f"flatten UV {name} must be finite and positive")
		expected = (expected_source_step if name == "source_step"
					else expected_output_step)
		if (expected is not None
				and not math.isclose(value, float(expected),
								 rel_tol=1.0e-6, abs_tol=1.0e-9)):
			raise FlattenUvError(
				f"flatten UV {name} {value:g} does not match {float(expected):g}")
	stored_fingerprint = canonical_grid_fingerprint(
		source_shape, source_step=float(metadata["source_step"]),
		winding_column_ranges=source_ranges)
	if metadata.get("canonical_grid_fingerprint") != stored_fingerprint:
		raise FlattenUvError("flatten UV stored grid fingerprint is corrupt")
	uv = np.stack((row, col), axis=-1)
	uv = validate_uv(
		uv, source_shape, valid=valid, cell_valid=cell_valid)
	return (
		np.ascontiguousarray(uv), valid,
		np.ascontiguousarray(cell_valid), dict(metadata))
