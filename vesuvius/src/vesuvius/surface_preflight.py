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


SCHEMA_VERSION = 1
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


def _resolve_volume_array(opened: Any, array_key: str | None) -> tuple[Any, str]:
    if hasattr(opened, "shape"):
        if array_key:
            raise ValueError("--array-key cannot be used when --volume is an array")
        return opened, ""

    if array_key:
        try:
            return opened[array_key], array_key
        except KeyError as exc:
            raise ValueError(f"OME-Zarr has no array at key {array_key!r}") from exc

    multiscales = opened.attrs.get("multiscales", [])
    if multiscales:
        datasets = multiscales[0].get("datasets", [])
        if datasets and isinstance(datasets[0], Mapping):
            path = str(datasets[0].get("path", ""))
            if path:
                try:
                    return opened[path], path
                except KeyError as exc:
                    raise ValueError(
                        f"OME-Zarr multiscales points to missing array {path!r}"
                    ) from exc

    if "0" in opened:
        return opened["0"], "0"

    array_keys = sorted(str(key) for key in opened.array_keys())
    if len(array_keys) == 1:
        return opened[array_keys[0]], array_keys[0]
    raise ValueError(
        "could not choose a volume array; pass --array-key for this OME-Zarr"
    )


def _open_volume(volume: str, array_key: str | None) -> tuple[Any, str]:
    try:
        import zarr
    except ImportError as exc:  # pragma: no cover - package dependency invariant
        raise RuntimeError("surface preflight requires zarr") from exc

    store: Any = volume
    if "://" in volume and not volume.startswith("file://"):
        import fsspec

        store = fsspec.get_mapper(volume)
    elif volume.startswith("file://"):
        store = volume.removeprefix("file://")
    opened = zarr.open(store, mode="r")
    return _resolve_volume_array(opened, array_key)


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


def _scan_surface(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    mask: np.ndarray | None,
    volume_shape: Sequence[int],
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
        if valid.shape[0] > 1:
            quads = (
                valid[:-1, :-1]
                & valid[1:, :-1]
                & valid[:-1, 1:]
                & valid[1:, 1:]
            )
            valid_quad_count += int(np.count_nonzero(quads))
        if valid.shape[0]:
            previous_valid = valid[-1].copy()

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
        "out_of_bounds_count": out_of_bounds_count,
        "coordinate_bounds_xyz": bounds,
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


def inspect_pair(
    surface: Path | str,
    volume: str,
    *,
    array_key: str | None = None,
    margin: float = 0.0,
    max_samples: int = 1024,
    minimum_support_fraction: float = 0.95,
    support_threshold: float = 0.0,
    block_rows: int = 256,
) -> dict[str, Any]:
    """Inspect one TIFXYZ/volume pair and return a JSON-serializable report."""
    surface_path = Path(surface)
    gates: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "FAIL",
        "surface": {"path": str(surface_path)},
        "volume": {"path": volume, "requested_array_key": array_key},
        "configuration": {
            "margin_voxels": margin,
            "max_support_samples": max_samples,
            "minimum_support_fraction": minimum_support_fraction,
            "support_threshold": support_threshold,
            "support_sampling": "evenly ranked valid vertices; nearest CT voxel",
            "block_rows": block_rows,
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
            mask = _read_tiff(mask_path)
            mask_matches = mask.shape == x.shape
            gates.append(
                _gate(
                    "tifxyz_mask_shape",
                    mask_matches,
                    observed=list(mask.shape),
                    threshold=list(x.shape),
                    message=(
                        "mask shape matches coordinates"
                        if mask_matches
                        else "mask shape does not match coordinates"
                    ),
                )
            )
            if not mask_matches:
                return _finalize_report(report)

        array, resolved_key = _open_volume(volume, array_key)
        volume_shape = tuple(int(item) for item in array.shape)
        volume_is_3d = len(volume_shape) == 3 and all(item > 0 for item in volume_shape)
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
                        else "surface has no valid vertices"
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
                ),
            ]
        )

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
        description="Fail closed when a TIFXYZ surface is not safely paired with its CT volume."
    )
    parser.add_argument("--surface", required=True, type=Path, help="TIFXYZ directory")
    parser.add_argument("--volume", required=True, help="Zarr/OME-Zarr path or URI")
    parser.add_argument("--array-key", help="OME-Zarr array key; defaults to level 0")
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
    )
    if args.output:
        _atomic_write_json(args.output, report)
        print(f"{report['status']}: {args.output}", file=sys.stderr)
    else:
        json.dump(report, sys.stdout, indent=2, sort_keys=True, default=_json_scalar)
        sys.stdout.write("\n")
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
