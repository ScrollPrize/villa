"""Fail-closed preflight for a TIFXYZ surface and its source CT volume.

The checks in this module are deliberately deterministic and inexpensive
enough to run before rendering, label transfer, or model inference.  They do
not claim that a surface is geometrically correct; they catch common input
pairing failures before an expensive downstream command starts.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = 2
REQUIRED_TIFXYZ_FILES = ("x.tif", "y.tif", "z.tif", "meta.json")


def _gate(
    name: str,
    passed: bool,
    *,
    observed: Any,
    threshold: Any = None,
    message: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "required": True,
        "passed": bool(passed),
        "observed": observed,
        "threshold": threshold,
        "message": message,
    }


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _finalize_report(report: dict[str, Any]) -> dict[str, Any]:
    gates = report["gates"]
    report["status"] = (
        "PASS" if gates and all(gate["passed"] for gate in gates) else "FAIL"
    )
    report["summary"] = {
        "passed_required_gates": sum(bool(gate["passed"]) for gate in gates),
        "required_gate_count": len(gates),
    }
    return report


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_metadata(surface_path: Path) -> tuple[dict[str, Any], tuple[float, float]]:
    meta_path = surface_path / "meta.json"
    with meta_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    scale = metadata.get("scale")
    if (
        not isinstance(scale, list)
        or len(scale) < 2
        or not all(isinstance(item, (int, float)) for item in scale[:2])
        or not all(np.isfinite(float(item)) and float(item) > 0 for item in scale[:2])
    ):
        raise ValueError("meta.json scale must contain two positive finite numbers")
    return metadata, (float(scale[0]), float(scale[1]))


def _read_tiff(path: Path) -> np.ndarray:
    try:
        import tifffile
    except ImportError as exc:  # pragma: no cover - depends on installation extras
        raise RuntimeError(
            "TIFXYZ preflight requires the label-transfer extra: "
            "pip install 'vesuvius[label-transfer]'"
        ) from exc

    try:
        return tifffile.memmap(path, mode="r")
    except ValueError:
        # Compressed TIFFs cannot be mapped directly.  tifffile creates a
        # temporary memory-mapped array rather than retaining the full raster
        # in process memory.
        return tifffile.imread(path, out="memmap")


def _load_tifxyz_mask(path: Path, xyz_shape: Sequence[int]) -> np.ndarray:
    """Load a mask through the label-transfer implementation."""
    try:
        from vesuvius.tifxyz_label_transfer.io import load_tifxyz_mask
    except ImportError as exc:  # pragma: no cover - depends on installation extras
        raise RuntimeError(
            "TIFXYZ preflight requires the label-transfer extra: "
            "pip install 'vesuvius[label-transfer]'"
        ) from exc
    return load_tifxyz_mask(path, xyz_shape)


def _resolve_volume_array(opened: Any, array_key: str | None) -> tuple[Any, str]:
    if hasattr(opened, "shape"):
        if array_key:
            raise ValueError("--array-key cannot be used when --volume is an array")
        return opened, ""

    multiscales = opened.attrs.get("multiscales", [])
    datasets = multiscales[0].get("datasets", []) if multiscales else []
    base_key = ""
    if datasets and isinstance(datasets[0], Mapping):
        base_key = str(datasets[0].get("path", ""))
    if not base_key and "0" in opened:
        base_key = "0"
    if not base_key:
        array_keys = sorted(str(key) for key in opened.array_keys())
        if len(array_keys) == 1:
            base_key = array_keys[0]

    if array_key:
        try:
            array = opened[array_key]
        except KeyError as exc:
            raise ValueError(f"OME-Zarr has no array at key {array_key!r}") from exc
        if not base_key:
            raise ValueError(
                "cannot verify that --array-key selects the base-resolution "
                "array; omit --array-key or provide OME-Zarr multiscales metadata"
            )
        if array_key != base_key:
            raise ValueError(
                "--array-key must select the base-resolution array because "
                "surface coordinates are in base-resolution voxel space; "
                f"expected {base_key!r}, got {array_key!r}"
            )
        return array, array_key

    if base_key:
        try:
            return opened[base_key], base_key
        except KeyError as exc:
            raise ValueError(
                f"OME-Zarr multiscales points to missing array {base_key!r}"
            ) from exc
    raise ValueError(
        "could not choose a volume array; pass --array-key for this OME-Zarr"
    )


def _open_volume(volume: str, array_key: str | None) -> tuple[Any, str]:
    try:
        import zarr
    except ImportError as exc:  # pragma: no cover - package dependency invariant
        raise RuntimeError("surface preflight requires zarr") from exc

    store_volume = volume.rstrip("/")
    direct_key = None
    if ".zarr/" in store_volume:
        store_root, direct_key = store_volume.rsplit(".zarr/", 1)
        store_volume = f"{store_root}.zarr"
    if direct_key and array_key:
        raise ValueError(
            "--array-key cannot be combined with a --volume path that already "
            "selects a Zarr array"
        )

    store: Any = store_volume
    if "://" in store_volume and not store_volume.startswith("file://"):
        import fsspec

        store = fsspec.get_mapper(store_volume)
    elif store_volume.startswith("file://"):
        store = store_volume.removeprefix("file://")
    opened = zarr.open(store, mode="r")
    return _resolve_volume_array(opened, direct_key or array_key)


def _iter_blocks(height: int, block_rows: int) -> Iterable[tuple[int, int]]:
    for start in range(0, height, block_rows):
        yield start, min(height, start + block_rows)


def _block_valid_mask(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if mask is None:
        selected = ~(z <= 0)
        return selected & finite, finite
    mask_array = np.asarray(mask)
    selected = mask_array if mask_array.dtype == np.bool_ else mask_array >= 255
    return selected & finite, finite


def _accumulate_spacing(
    first: tuple[np.ndarray, np.ndarray, np.ndarray],
    second: tuple[np.ndarray, np.ndarray, np.ndarray],
    pairs: np.ndarray,
    histogram: np.ndarray,
) -> tuple[int, int]:
    if not np.any(pairs):
        return 0, 0
    first_values = [np.asarray(values, dtype=np.float64) for values in first]
    second_values = [np.asarray(values, dtype=np.float64) for values in second]
    distances = np.sqrt(
        sum(
            np.square(first_values[axis] - second_values[axis])
            for axis in range(3)
        )
    )[pairs]
    zero_length = int(np.count_nonzero(distances == 0))
    positive = distances > 0
    if np.any(positive):
        bins = np.floor(np.log2(distances[positive]) * 64).astype(np.int64)
        bins = np.clip(bins, -64 * 20, 64 * 20 - 1) + 64 * 20
        np.add.at(histogram, bins, 1)
    return int(len(distances)), zero_length


def _spacing_summary(
    histogram: np.ndarray, pair_count: int, zero_length: int
) -> dict[str, Any]:
    positive_count = int(histogram.sum())
    median = None
    median_bounds = None
    if positive_count:
        target = positive_count // 2
        median_bin = int(np.searchsorted(np.cumsum(histogram), target, side="right"))
        exponent = median_bin - 64 * 20
        median = float(2 ** ((exponent + 0.5) / 64))
        median_bounds = [float(2 ** (exponent / 64)), float(2 ** ((exponent + 1) / 64))]
    return {
        "pair_count": pair_count,
        "median_spacing_voxels": median,
        "median_spacing_bounds_voxels": median_bounds,
        "zero_length_pair_count": zero_length,
    }


def _scan_surface(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    mask: np.ndarray | None,
    volume_shape: Sequence[int] | None,
    *,
    margin: float,
    block_rows: int,
) -> dict[str, Any]:
    height, width = (int(x.shape[0]), int(x.shape[1]))
    valid_count = 0
    selected_nonfinite_count = 0
    out_of_bounds_count = 0
    valid_quad_count = 0
    minima = np.full(3, np.inf, dtype=np.float64)
    maxima = np.full(3, -np.inf, dtype=np.float64)
    previous_valid: np.ndarray | None = None
    previous_coordinates: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    column_histogram = np.zeros(64 * 40, dtype=np.int64)
    row_histogram = np.zeros(64 * 40, dtype=np.int64)
    column_pair_count = 0
    row_pair_count = 0
    column_zero_length_count = 0
    row_zero_length_count = 0
    limits_xyz = None
    if volume_shape is not None:
        limits_xyz = np.asarray(
            [volume_shape[2] - 1, volume_shape[1] - 1, volume_shape[0] - 1],
            dtype=np.float64,
        )

    for start, stop in _iter_blocks(height, block_rows):
        xb = np.asarray(x[start:stop])
        yb = np.asarray(y[start:stop])
        zb = np.asarray(z[start:stop])
        mb = None if mask is None else np.asarray(mask[start:stop])
        valid, finite = _block_valid_mask(xb, yb, zb, mb)
        if mb is None:
            selected = ~(zb <= 0)
        else:
            selected = mb if mb.dtype == np.bool_ else mb >= 255
        selected_nonfinite_count += int(np.count_nonzero(selected & ~finite))

        valid_count += int(np.count_nonzero(valid))
        if np.any(valid):
            coordinates = (xb, yb, zb)
            for axis, values in enumerate(coordinates):
                minima[axis] = min(minima[axis], float(np.min(values[valid])))
                maxima[axis] = max(maxima[axis], float(np.max(values[valid])))
            if limits_xyz is not None:
                out_of_bounds = valid.copy()
                for axis, values in enumerate(coordinates):
                    out_of_bounds &= (
                        (values >= margin) & (values <= limits_xyz[axis] - margin)
                    )
                out_of_bounds_count += int(np.count_nonzero(valid & ~out_of_bounds))

        if previous_valid is not None and valid.shape[0]:
            bridge = (
                previous_valid[:-1]
                & previous_valid[1:]
                & valid[0, :-1]
                & valid[0, 1:]
            )
            valid_quad_count += int(np.count_nonzero(bridge))
            bridge_pairs = previous_valid & valid[0]
            if previous_coordinates is not None:
                pairs = bridge_pairs
                count, zero_count = _accumulate_spacing(
                    (
                        previous_coordinates[0],
                        previous_coordinates[1],
                        previous_coordinates[2],
                    ),
                    (xb[0], yb[0], zb[0]),
                    pairs,
                    row_histogram,
                )
                row_pair_count += count
                row_zero_length_count += zero_count
        if valid.shape[0]:
            column_pairs = valid[:, :-1] & valid[:, 1:]
            count, zero_count = _accumulate_spacing(
                (xb[:, :-1], yb[:, :-1], zb[:, :-1]),
                (xb[:, 1:], yb[:, 1:], zb[:, 1:]),
                column_pairs,
                column_histogram,
            )
            column_pair_count += count
            column_zero_length_count += zero_count
        if valid.shape[0] > 1:
            quads = (
                valid[:-1, :-1]
                & valid[1:, :-1]
                & valid[:-1, 1:]
                & valid[1:, 1:]
            )
            valid_quad_count += int(np.count_nonzero(quads))
            row_pairs = valid[:-1, :] & valid[1:, :]
            count, zero_count = _accumulate_spacing(
                (xb[:-1, :], yb[:-1, :], zb[:-1, :]),
                (xb[1:, :], yb[1:, :], zb[1:, :]),
                row_pairs,
                row_histogram,
            )
            row_pair_count += count
            row_zero_length_count += zero_count
        if valid.shape[0]:
            previous_valid = valid[-1].copy()
            previous_coordinates = (
                xb[-1].copy(),
                yb[-1].copy(),
                zb[-1].copy(),
            )

    bounds = None
    if valid_count:
        bounds = {
            "x": [float(minima[0]), float(maxima[0])],
            "y": [float(minima[1]), float(maxima[1])],
            "z": [float(minima[2]), float(maxima[2])],
        }
    return {
        "stored_shape_yx": [height, width],
        "valid_vertex_count": valid_count,
        "valid_quad_count": valid_quad_count,
        "selected_nonfinite_count": selected_nonfinite_count,
        "out_of_bounds_count": (
            out_of_bounds_count if volume_shape is not None else None
        ),
        "coordinate_bounds_xyz": bounds,
        "grid_spacing_voxels": {
            "columns": _spacing_summary(
                column_histogram,
                column_pair_count,
                column_zero_length_count,
            ),
            "rows": _spacing_summary(
                row_histogram,
                row_pair_count,
                row_zero_length_count,
            ),
        },
    }


def _sample_points(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    mask: np.ndarray | None,
    *,
    valid_count: int,
    max_samples: int,
    block_rows: int,
) -> np.ndarray:
    sample_count = min(valid_count, max_samples)
    if sample_count == 0:
        return np.empty((0, 3), dtype=np.float64)
    ranks = np.linspace(0, valid_count - 1, sample_count, dtype=np.int64)
    samples = np.empty((sample_count, 3), dtype=np.float64)
    seen = 0
    written = 0

    for start, stop in _iter_blocks(int(x.shape[0]), block_rows):
        xb = np.asarray(x[start:stop])
        yb = np.asarray(y[start:stop])
        zb = np.asarray(z[start:stop])
        mb = None if mask is None else np.asarray(mask[start:stop])
        valid, _ = _block_valid_mask(xb, yb, zb, mb)
        block_count = int(np.count_nonzero(valid))
        if block_count == 0:
            continue
        next_written = int(np.searchsorted(ranks, seen + block_count, side="left"))
        if next_written > written:
            local_ranks = ranks[written:next_written] - seen
            flat_indices = np.flatnonzero(valid)[local_ranks]
            samples[written:next_written, 0] = xb.ravel()[flat_indices]
            samples[written:next_written, 1] = yb.ravel()[flat_indices]
            samples[written:next_written, 2] = zb.ravel()[flat_indices]
            written = next_written
        seen += block_count
        if written == sample_count:
            break
    return samples[:written]


def _sample_volume_support(
    array: Any,
    points_xyz: np.ndarray,
    *,
    threshold: float,
) -> dict[str, Any]:
    if len(points_xyz) == 0:
        return {"sample_count": 0, "supported_count": 0, "support_fraction": 0.0}

    shape = tuple(int(item) for item in array.shape)
    points_zyx = np.rint(points_xyz[:, ::-1]).astype(np.int64)
    in_bounds = np.all(points_zyx >= 0, axis=1) & np.all(
        points_zyx < np.asarray(shape, dtype=np.int64), axis=1
    )
    points_zyx = points_zyx[in_bounds]
    if len(points_zyx) == 0:
        return {"sample_count": 0, "supported_count": 0, "support_fraction": 0.0}

    raw_chunks = getattr(array, "chunks", None) or shape
    chunks = tuple(int(item) for item in raw_chunks)
    grouped: dict[tuple[int, int, int], list[tuple[int, int, int]]] = defaultdict(list)
    for point in points_zyx:
        point_tuple = tuple(int(item) for item in point)
        key = tuple(point_tuple[axis] // chunks[axis] for axis in range(3))
        grouped[key].append(point_tuple)

    supported = 0
    for key in sorted(grouped):
        starts = tuple(key[axis] * chunks[axis] for axis in range(3))
        stops = tuple(min(starts[axis] + chunks[axis], shape[axis]) for axis in range(3))
        block = np.asarray(array[tuple(slice(starts[a], stops[a]) for a in range(3))])
        for point in grouped[key]:
            local = tuple(point[axis] - starts[axis] for axis in range(3))
            value = block[local]
            supported += int(np.isfinite(value) and abs(float(value)) > threshold)

    sample_count = int(len(points_zyx))
    return {
        "sample_count": sample_count,
        "supported_count": supported,
        "support_fraction": float(supported / sample_count),
    }


def _scale_consistency(
    scan: Mapping[str, Any],
    scale: tuple[float, float],
    tolerance: float,
) -> dict[str, Any]:
    ratio_bounds = [1.0 / tolerance, tolerance]
    observed: dict[str, Any] = {}
    failed_axis = None
    failed_scale = None
    for axis, name in enumerate(("columns", "rows")):
        spacing = scan["grid_spacing_voxels"][name]
        median = spacing["median_spacing_voxels"]
        median_bounds = spacing["median_spacing_bounds_voxels"]
        pair_count = int(spacing["pair_count"])
        implied_scale = None if median is None else float(1.0 / median)
        ratio = None if median is None else float(median * scale[axis])
        ratio_range = (
            None
            if median_bounds is None
            else [float(bound * scale[axis]) for bound in median_bounds]
        )
        observed[name] = {
            "expected_spacing_voxels": float(1.0 / scale[axis]),
            "median_spacing_voxels": median,
            "median_spacing_bounds_voxels": median_bounds,
            "ratio": ratio,
            "ratio_range": ratio_range,
            "pair_count": pair_count,
            "implied_scale": implied_scale,
        }
        # The median is only known to lie within its histogram bin, so the gate
        # fails only when the whole bin lies outside the tolerated ratio range.
        if (
            failed_axis is None
            and (
                pair_count == 0
                or median is None
                or ratio_range is None
                or ratio_range[1] < ratio_bounds[0]
                or ratio_range[0] > ratio_bounds[1]
            )
        ):
            failed_axis = name
            failed_scale = scale[axis]

    passed = failed_axis is None
    if passed:
        message = "metadata scale agrees with the emitted grid spacing"
    elif observed[failed_axis]["median_spacing_voxels"] is None:
        message = (
            f"grid has no positive {failed_axis} spacing pairs to compare with "
            "meta.json scale"
        )
    else:
        axis_observed = observed[failed_axis]
        message = (
            "meta.json scale disagrees with the emitted grid spacing "
            f"({failed_axis}: scale {failed_scale:.6f} "
            f"implies {axis_observed['expected_spacing_voxels']:.1f} voxels per "
            f"cell, grid measures {axis_observed['median_spacing_voxels']:.1f}); "
            "fix the producer's scale or re-export"
        )
    return _gate(
        "tifxyz_scale_consistency",
        passed,
        observed=observed,
        threshold={"ratio_within": ratio_bounds},
        message=message,
    )


def _bbox_consistency(
    scan: Mapping[str, Any],
    bbox: Any,
    tolerance: float,
) -> dict[str, Any]:
    try:
        parsed = np.asarray(bbox, dtype=np.float64)
    except (TypeError, ValueError):
        parsed = np.asarray([])
    if parsed.shape != (2, 3) or not np.all(np.isfinite(parsed)):
        return _gate(
            "tifxyz_bbox_consistency",
            False,
            observed={"error": "meta.json bbox must be a 2x3 array of finite numbers"},
            threshold={"bbox_tolerance_voxels": tolerance},
            message="meta.json bbox must be a 2x3 array of finite numbers",
        )

    coordinate_bounds = scan["coordinate_bounds_xyz"]
    if scan["valid_vertex_count"] == 0:
        return _gate(
            "tifxyz_bbox_consistency",
            False,
            observed={
                "bbox": parsed.tolist(),
                "coordinate_bounds_xyz": coordinate_bounds,
                "max_excess_voxels": None,
            },
            threshold={"bbox_tolerance_voxels": tolerance},
            message="no valid vertices to compare against bbox",
        )

    bounds = np.asarray(
        [
            [coordinate_bounds[axis][0] for axis in ("x", "y", "z")],
            [coordinate_bounds[axis][1] for axis in ("x", "y", "z")],
        ],
        dtype=np.float64,
    )
    excess = np.maximum(
        np.maximum(
            parsed[0] - bounds[0] - tolerance,
            bounds[1] - parsed[1] - tolerance,
        ),
        0,
    )
    max_excess = float(np.max(excess))
    passed = max_excess == 0
    return _gate(
        "tifxyz_bbox_consistency",
        passed,
        observed={
            "bbox": parsed.tolist(),
            "coordinate_bounds_xyz": coordinate_bounds,
            "max_excess_voxels": max_excess,
        },
        threshold={"bbox_tolerance_voxels": tolerance},
        message=(
            "metadata bbox covers the valid vertices"
            if passed
            else (
                "meta.json bbox does not cover the valid vertices "
                f"(exceeds by {max_excess:g} voxels); the bbox is stale, "
                "rewrite it from the coordinate rasters"
            )
        ),
    )


def inspect_pair(
    surface: Path | str,
    volume: str | None = None,
    *,
    array_key: str | None = None,
    margin: float = 0.0,
    max_samples: int = 1024,
    minimum_support_fraction: float = 0.95,
    support_threshold: float = 0.0,
    block_rows: int = 256,
    scale_tolerance: float = 2.0,
    bbox_tolerance: float = 0.01,
) -> dict[str, Any]:
    """Inspect one TIFXYZ/volume pair and return a JSON-serializable report."""
    surface_path = Path(surface)
    gates: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "FAIL",
        "surface": {"path": str(surface_path)},
        "volume": (
            {"path": volume, "requested_array_key": array_key}
            if volume is not None
            else None
        ),
        "configuration": {
            "margin_voxels": margin,
            "max_support_samples": max_samples,
            "minimum_support_fraction": minimum_support_fraction,
            "support_threshold": support_threshold,
            "support_sampling": "evenly ranked valid vertices; nearest CT voxel",
            "block_rows": block_rows,
            "scale_tolerance": scale_tolerance,
            "bbox_tolerance": bbox_tolerance,
        },
        "gates": gates,
    }

    try:
        if margin < 0:
            raise ValueError("margin must be non-negative")
        if max_samples <= 0:
            raise ValueError("max_samples must be positive")
        if not 0 <= minimum_support_fraction <= 1:
            raise ValueError("minimum_support_fraction must be between 0 and 1")
        if support_threshold < 0:
            raise ValueError("support_threshold must be non-negative")
        if block_rows <= 0:
            raise ValueError("block_rows must be positive")
        if scale_tolerance < 1:
            raise ValueError("scale_tolerance must be at least 1")
        if bbox_tolerance < 0:
            raise ValueError("bbox_tolerance must be non-negative")

        missing = [
            name
            for name in REQUIRED_TIFXYZ_FILES
            if not (surface_path / name).is_file()
        ]
        gates.append(
            _gate(
                "tifxyz_required_files",
                not missing,
                observed={"missing": missing},
                threshold={"missing": []},
                message=(
                    "required TIFXYZ files are present"
                    if not missing
                    else "required TIFXYZ files are missing"
                ),
            )
        )
        if missing:
            return _finalize_report(report)

        metadata, scale = _read_metadata(surface_path)
        report["surface"].update(
            {
                "uuid": str(metadata.get("uuid", surface_path.name)),
                "scale_xy": list(scale),
                "meta_sha256": _sha256(surface_path / "meta.json"),
            }
        )
        gates.append(
            _gate(
                "tifxyz_metadata",
                True,
                observed={"scale_xy": list(scale)},
                message="metadata is valid",
            )
        )

        x = _read_tiff(surface_path / "x.tif")
        y = _read_tiff(surface_path / "y.tif")
        z = _read_tiff(surface_path / "z.tif")
        shapes = {"x": list(x.shape), "y": list(y.shape), "z": list(z.shape)}
        shapes_match = x.ndim == 2 and x.shape == y.shape == z.shape
        gates.append(
            _gate(
                "tifxyz_coordinate_shapes",
                shapes_match,
                observed=shapes,
                threshold="matching 2D arrays",
                message=(
                    "coordinate arrays match"
                    if shapes_match
                    else "coordinate arrays must be matching 2D rasters"
                ),
            )
        )
        if not shapes_match:
            return _finalize_report(report)

        mask = None
        mask_path = surface_path / "mask.tif"
        if mask_path.is_file():
            try:
                mask = _load_tifxyz_mask(mask_path, x.shape)
                mask_error = None
            except ValueError as exc:
                mask_error = str(exc)
            mask_matches = mask_error is None
            gates.append(
                _gate(
                    "tifxyz_mask_shape",
                    mask_matches,
                    observed=(
                        list(mask.shape) if mask_matches else {"error": mask_error}
                    ),
                    threshold=list(x.shape),
                    message=(
                        "mask shape matches coordinates"
                        if mask_matches
                        else "mask shape is incompatible with coordinates"
                    ),
                )
            )
            if not mask_matches:
                return _finalize_report(report)

        array = None
        volume_shape = None
        if volume is not None:
            array, resolved_key = _open_volume(volume, array_key)
            volume_shape = tuple(int(item) for item in array.shape)
            volume_is_3d = len(volume_shape) == 3 and all(
                item > 0 for item in volume_shape
            )
            report["volume"].update(
                {
                    "resolved_array_key": resolved_key,
                    "shape_zyx": list(volume_shape),
                    "dtype": str(array.dtype),
                    "chunks_zyx": list(getattr(array, "chunks", None) or volume_shape),
                }
            )
            gates.append(
                _gate(
                    "volume_is_3d",
                    volume_is_3d,
                    observed=list(volume_shape),
                    threshold="positive z/y/x shape",
                    message=(
                        "volume is a 3D z/y/x array"
                        if volume_is_3d
                        else "volume must be a 3D z/y/x array"
                    ),
                )
            )
            if not volume_is_3d:
                return _finalize_report(report)

        scan = _scan_surface(
            x,
            y,
            z,
            mask,
            volume_shape,
            margin=margin,
            block_rows=block_rows,
        )
        report["surface"].update(scan)
        gates.extend(
            [
                _gate(
                    "valid_surface_vertices",
                    scan["valid_vertex_count"] > 0,
                    observed=scan["valid_vertex_count"],
                    threshold="> 0",
                    message=(
                        "surface has valid vertices"
                        if scan["valid_vertex_count"]
                        else (
                            "surface has no valid vertices (all coordinates are "
                            "sentinel/invalid); the producer emitted an empty grid"
                        )
                    ),
                ),
                _gate(
                    "valid_surface_quads",
                    scan["valid_quad_count"] > 0,
                    observed=scan["valid_quad_count"],
                    threshold="> 0",
                    message=(
                        "surface has connected quads"
                        if scan["valid_quad_count"]
                        else "surface has no connected valid quads"
                    ),
                ),
                _gate(
                    "finite_selected_coordinates",
                    scan["selected_nonfinite_count"] == 0,
                    observed=scan["selected_nonfinite_count"],
                    threshold=0,
                    message=(
                        "selected coordinates are finite"
                        if scan["selected_nonfinite_count"] == 0
                        else "selected coordinates include non-finite values"
                    ),
                ),
            ]
        )
        if volume is not None:
            gates.append(
                _gate(
                    "coordinates_within_volume",
                    scan["out_of_bounds_count"] == 0,
                    observed={"out_of_bounds_count": scan["out_of_bounds_count"]},
                    threshold={"out_of_bounds_count": 0, "margin_voxels": margin},
                    message=(
                        "all valid coordinates lie inside the volume margin"
                        if scan["out_of_bounds_count"] == 0
                        else "valid coordinates fall outside the volume margin"
                    ),
                )
            )
        gates.append(_scale_consistency(scan, scale, scale_tolerance))
        if "bbox" in metadata:
            gates.append(_bbox_consistency(scan, metadata["bbox"], bbox_tolerance))

        if volume is not None:
            points = _sample_points(
                x,
                y,
                z,
                mask,
                valid_count=scan["valid_vertex_count"],
                max_samples=max_samples,
                block_rows=block_rows,
            )
            support = _sample_volume_support(array, points, threshold=support_threshold)
            report["volume"]["sampled_signal_support"] = support
            support_passed = (
                support["sample_count"] > 0
                and support["support_fraction"] >= minimum_support_fraction
            )
            gates.append(
                _gate(
                    "sampled_volume_signal_support",
                    support_passed,
                    observed=support,
                    threshold={
                        "minimum_support_fraction": minimum_support_fraction,
                        "absolute_signal_greater_than": support_threshold,
                    },
                    message=(
                        "sampled surface points have CT signal support"
                        if support_passed
                        else "sampled surface points lack sufficient CT signal support"
                    ),
                )
            )
    except Exception as exc:
        gates.append(
            _gate(
                "input_readable",
                False,
                observed={"error_type": type(exc).__name__, "error": str(exc)},
                message="inputs could not be validated",
            )
        )

    return _finalize_report(report)


def _atomic_write_json(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True, default=_json_scalar)
            handle.write("\n")
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fail closed when a TIFXYZ surface is structurally unusable or not "
            "safely paired with its CT volume."
        )
    )
    parser.add_argument("--surface", required=True, type=Path, help="TIFXYZ directory")
    parser.add_argument(
        "--volume",
        help="Zarr/OME-Zarr path or URI; omit to validate the surface alone",
    )
    parser.add_argument(
        "--array-key",
        help="base-resolution OME-Zarr array key; defaults to the base level",
    )
    parser.add_argument("--output", type=Path, help="JSON report path; defaults to stdout")
    parser.add_argument(
        "--margin",
        type=float,
        default=0.0,
        help="required in-volume margin in voxels",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1024,
        help="deterministic CT support sample count",
    )
    parser.add_argument(
        "--minimum-support-fraction",
        type=float,
        default=0.95,
        help="minimum fraction of sampled points with nonzero CT signal",
    )
    parser.add_argument(
        "--support-threshold",
        type=float,
        default=0.0,
        help="minimum absolute sampled CT value (strictly greater than)",
    )
    parser.add_argument(
        "--scale-tolerance",
        type=float,
        default=2.0,
        help="allowed ratio between metadata scale and grid spacing",
    )
    parser.add_argument(
        "--bbox-tolerance",
        type=float,
        default=0.01,
        help="allowed bbox coverage tolerance in voxels",
    )
    parser.add_argument("--block-rows", type=int, default=256, help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = inspect_pair(
        args.surface,
        args.volume,
        array_key=args.array_key,
        margin=args.margin,
        max_samples=args.max_samples,
        minimum_support_fraction=args.minimum_support_fraction,
        support_threshold=args.support_threshold,
        block_rows=args.block_rows,
        scale_tolerance=args.scale_tolerance,
        bbox_tolerance=args.bbox_tolerance,
    )
    if args.output:
        _atomic_write_json(args.output, report)
        print(f"{report['status']}: {args.output}", file=sys.stderr)
    else:
        json.dump(report, sys.stdout, indent=2, sort_keys=True, default=_json_scalar)
        sys.stdout.write("\n")
    if report["status"] == "FAIL":
        for gate in report["gates"]:
            if not gate["passed"]:
                print(f"{gate['name']}: {gate['message']}", file=sys.stderr)
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
