from __future__ import annotations

import copy
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tifffile


SHAPE_TOL = 0.01


@dataclass(frozen=True)
class CoordinateScale:
	base_shape_zyx: tuple[int, int, int] | None
	source_shape_zyx: tuple[int, int, int] | None
	factor: float

	@property
	def is_identity(self) -> bool:
		return abs(float(self.factor) - 1.0) <= 1.0e-12


def parse_shape_zyx(value: Any, *, name: str, required: bool = False) -> tuple[int, int, int] | None:
	if value is None:
		if required:
			raise ValueError(f"{name} is required")
		return None
	if not isinstance(value, (list, tuple)) or len(value) != 3:
		raise ValueError(f"{name} must be a ZYX list of three positive integers")
	out: list[int] = []
	for v in value:
		try:
			iv = int(v)
		except (TypeError, ValueError):
			raise ValueError(f"{name} must contain positive integers") from None
		if iv <= 0:
			raise ValueError(f"{name} must contain positive integers")
		out.append(iv)
	return (out[0], out[1], out[2])


def _nearest_power_of_two(value: float) -> float:
	exp = int(round(math.log2(value)))
	return float(2.0 ** exp)


def coordinate_scale_to_base(
	*,
	base_shape_zyx: tuple[int, int, int] | None,
	source_shape_zyx: tuple[int, int, int] | None,
	source_name: str = "source_shape_zyx",
) -> CoordinateScale:
	"""Return the isotropic coordinate factor from source-shape coords to base coords."""
	if base_shape_zyx is None:
		return CoordinateScale(None, source_shape_zyx, 1.0)
	if source_shape_zyx is None:
		return CoordinateScale(base_shape_zyx, None, 1.0)

	base = parse_shape_zyx(base_shape_zyx, name="base_shape_zyx", required=True)
	src = parse_shape_zyx(source_shape_zyx, name=source_name, required=True)
	assert base is not None and src is not None

	ratios = [float(b) / float(s) for b, s in zip(base, src)]
	ref = ratios[0]
	for axis, ratio in zip("ZYX", ratios):
		rel = abs(ratio - ref) / max(abs(ref), 1.0e-12)
		if rel > SHAPE_TOL:
			raise ValueError(
				f"{source_name} is anisotropic relative to base_shape_zyx: "
				f"axis {axis} ratio={ratio:.9g}, expected {ref:.9g} within {SHAPE_TOL:.0%}"
			)

	mean_ratio = sum(ratios) / 3.0
	pow2 = _nearest_power_of_two(mean_ratio)
	for axis, ratio in zip("ZYX", ratios):
		rel = abs(ratio - pow2) / max(abs(pow2), 1.0e-12)
		if rel > SHAPE_TOL:
			raise ValueError(
				f"{source_name} scale must be a power-of-two ratio to base_shape_zyx: "
				f"axis {axis} ratio={ratio:.9g}, nearest_power_of_two={pow2:.9g}, "
				f"tolerance={SHAPE_TOL:.0%}"
			)
	return CoordinateScale(base, src, pow2)


def coordinate_scale_between_shapes(
	*,
	from_shape_zyx: tuple[int, int, int] | None,
	to_shape_zyx: tuple[int, int, int] | None,
	from_name: str = "from_shape_zyx",
	to_name: str = "to_shape_zyx",
) -> CoordinateScale:
	"""Return coordinate factor from from-shape coords to to-shape coords."""
	scale_to_base = coordinate_scale_to_base(
		base_shape_zyx=to_shape_zyx,
		source_shape_zyx=from_shape_zyx,
		source_name=from_name,
	)
	return CoordinateScale(to_shape_zyx, from_shape_zyx, scale_to_base.factor)


def scale_xyz_point(point_xyz: Any, factor: float) -> Any:
	if point_xyz is None:
		return None
	if not isinstance(point_xyz, (list, tuple)) or len(point_xyz) < 3:
		return point_xyz
	return [float(point_xyz[i]) * float(factor) if i < 3 else point_xyz[i] for i in range(len(point_xyz))]


def scale_corr_points_json(obj: dict, factor: float) -> dict:
	out = copy.deepcopy(obj)
	if abs(float(factor) - 1.0) <= 1.0e-12:
		return out
	cols = out.get("collections", {})
	if not isinstance(cols, dict):
		return out
	for col in cols.values():
		if not isinstance(col, dict):
			continue
		points = col.get("points", {})
		if not isinstance(points, dict):
			continue
		for point in points.values():
			if isinstance(point, dict) and "p" in point:
				point["p"] = scale_xyz_point(point["p"], factor)
	return out


def scale_tifxyz_tensor(xyz, valid, factor: float):
	if abs(float(factor) - 1.0) <= 1.0e-12:
		return xyz
	out = xyz.clone()
	if valid is None:
		return out * float(factor)
	out[valid.bool()] = out[valid.bool()] * float(factor)
	return out


def scale_tifxyz_arrays(
	x: np.ndarray,
	y: np.ndarray,
	z: np.ndarray,
	factor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	if abs(float(factor) - 1.0) <= 1.0e-12:
		return x.astype(np.float32, copy=True), y.astype(np.float32, copy=True), z.astype(np.float32, copy=True)
	xf = x.astype(np.float32, copy=True)
	yf = y.astype(np.float32, copy=True)
	zf = z.astype(np.float32, copy=True)
	valid = np.isfinite(xf) & np.isfinite(yf) & np.isfinite(zf)
	valid &= ~((xf == -1.0) & (yf == -1.0) & (zf == -1.0))
	xf[valid] *= float(factor)
	yf[valid] *= float(factor)
	zf[valid] *= float(factor)
	return xf, yf, zf


def _scale_bbox(value: Any, factor: float) -> Any:
	if (
		isinstance(value, list)
		and len(value) == 2
		and all(isinstance(v, (list, tuple)) and len(v) >= 3 for v in value)
	):
		return [scale_xyz_point(v, factor) for v in value]
	return value


def _is_number_list(value: Any, *, min_len: int = 3) -> bool:
	if not isinstance(value, list) or len(value) < min_len:
		return False
	try:
		for i in range(min_len):
			float(value[i])
	except (TypeError, ValueError):
		return False
	return True


def _scale_meta_point_fields(value: Any, factor: float) -> Any:
	if isinstance(value, dict):
		out: dict = {}
		for key, child in value.items():
			if key in {"p", "point", "xyz", "seed"} and _is_number_list(child):
				out[key] = scale_xyz_point(child, factor)
			elif key == "bbox":
				out[key] = _scale_bbox(child, factor)
			elif key in {"scale", "base_shape_zyx", "lasagna_base_shape_zyx"}:
				out[key] = copy.deepcopy(child)
			else:
				out[key] = _scale_meta_point_fields(child, factor)
		return out
	if isinstance(value, list):
		return [_scale_meta_point_fields(child, factor) for child in value]
	return copy.deepcopy(value)


def scale_tifxyz_meta(
	meta: dict,
	factor: float,
	*,
	base_shape_zyx: tuple[int, int, int] | None = None,
	lasagna_base_shape_zyx: tuple[int, int, int] | None = None,
) -> dict:
	out = _scale_meta_point_fields(meta, factor) if isinstance(meta, dict) else {}
	f = float(factor)
	scale = out.get("scale")
	if isinstance(scale, list) and scale and abs(f) > 1.0e-12:
		new_scale = []
		for v in scale:
			try:
				new_scale.append(float(v) / f)
			except (TypeError, ValueError):
				new_scale.append(v)
		out["scale"] = new_scale
	if base_shape_zyx is not None:
		out["base_shape_zyx"] = [int(v) for v in base_shape_zyx]
	if lasagna_base_shape_zyx is not None:
		out["lasagna_base_shape_zyx"] = [int(v) for v in lasagna_base_shape_zyx]
	return out


def tifxyz_source_shape(meta: dict, fallback_shape_zyx: tuple[int, int, int] | None) -> tuple[int, int, int] | None:
	if isinstance(meta, dict):
		shape = parse_shape_zyx(meta.get("base_shape_zyx"), name="meta.base_shape_zyx")
		if shape is not None:
			return shape
	return fallback_shape_zyx


def read_tifxyz_meta(path: str | Path) -> dict:
	meta_path = Path(path) / "meta.json"
	if not meta_path.exists():
		return {}
	try:
		obj = json.loads(meta_path.read_text(encoding="utf-8"))
	except Exception:
		return {}
	return obj if isinstance(obj, dict) else {}


def copy_scaled_tifxyz_dir(
	src: str | Path,
	dst: str | Path,
	*,
	factor: float,
	base_shape_zyx: tuple[int, int, int] | None,
) -> Path:
	src_p = Path(src)
	dst_p = Path(dst)
	if dst_p.exists():
		shutil.rmtree(dst_p)
	shutil.copytree(src_p, dst_p)
	x = tifffile.imread(str(dst_p / "x.tif")).astype(np.float32)
	y = tifffile.imread(str(dst_p / "y.tif")).astype(np.float32)
	z = tifffile.imread(str(dst_p / "z.tif")).astype(np.float32)
	x, y, z = scale_tifxyz_arrays(x, y, z, factor)
	tifffile.imwrite(str(dst_p / "x.tif"), x, compression=None)
	tifffile.imwrite(str(dst_p / "y.tif"), y, compression=None)
	tifffile.imwrite(str(dst_p / "z.tif"), z, compression=None)
	meta = read_tifxyz_meta(dst_p)
	meta = scale_tifxyz_meta(
		meta,
		factor,
		base_shape_zyx=base_shape_zyx,
		lasagna_base_shape_zyx=base_shape_zyx,
	)
	(dst_p / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
	return dst_p
