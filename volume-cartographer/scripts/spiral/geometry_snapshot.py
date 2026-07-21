"""Atomic, versioned flat-geometry snapshots shared with VC3D."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping, Sequence

import numpy as np


SNAPSHOT_SCHEMA_VERSION = 1


def _write_packed_polylines(points_stream, offsets_stream, packed, input_order):
    points, offsets = packed
    points = np.asarray(points, dtype=np.float32)
    offsets = np.asarray(offsets, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Packed points must have shape [N, 3]")
    if (offsets.ndim != 1 or len(offsets) == 0 or offsets[0] != 0
            or offsets[-1] != len(points)
            or np.any(offsets[1:] <= offsets[:-1])):
        raise ValueError("Packed offsets must delimit non-empty polylines")
    order = input_order.upper()
    if order not in ("ZYX", "XYZ"):
        raise ValueError("input_order must be XYZ or ZYX")
    # Reverse/write in bounded point chunks: one Python call per large block,
    # rather than one call per short track, without making a second full cloud.
    chunk_points = 1_048_576
    for start in range(0, len(points), chunk_points):
        part = points[start:start + chunk_points]
        if not np.isfinite(part).all():
            raise ValueError("Packed points contain non-finite coordinates")
        if order == "ZYX":
            part = part[:, ::-1]
        np.ascontiguousarray(part, dtype="<f4").tofile(points_stream)
    np.ascontiguousarray(offsets, dtype="<u8").tofile(offsets_stream)
    return len(points), len(offsets) - 1

def write_geometry_snapshot(
    destination: str | os.PathLike[str],
    categories: Mapping[str, Sequence[np.ndarray]],
    *,
    input_order: str = "ZYX",
    coordinate_identity: Mapping[str, object] | None = None,
) -> dict[str, object]:
    destination = Path(destination).resolve(strict=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp: Path | None = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    manifest: dict[str, object] = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "coordinate_order": "XYZ",
        "source_coordinate_order": input_order.upper(),
        "dtype": "float32",
        "offset_dtype": "uint64",
        "byte_order": "little",
        "coordinate_identity": dict(coordinate_identity or {}),
        "categories": {},
    }
    try:
        for category in sorted(categories):
            safe = "".join(character if character.isalnum() or character in "-_" else "_" for character in category)
            points_name = f"{safe}.points.xyz.f32le"
            offsets_name = f"{safe}.offsets.u64le"
            point_count = 0
            polyline_count = 0
            offset_buffer = np.empty(65_536, dtype="<u8")
            buffered_offsets = 1
            offset_buffer[0] = 0
            # Packed providers write large bounded chunks. The fallback writes
            # one polyline at a time; concatenating a complete generic category
            # would require two additional full point-cloud copies.
            with (
                (temp / points_name).open("wb") as points_stream,
                (temp / offsets_name).open("wb") as offsets_stream,
            ):
                packed_provider = getattr(
                    categories[category], "as_packed_polylines", None)
                if packed_provider is not None:
                    point_count, polyline_count = _write_packed_polylines(
                        points_stream, offsets_stream, packed_provider(), input_order)
                else:
                    for index, polyline in enumerate(categories[category]):
                        points = np.asarray(polyline, dtype=np.float32)
                        if points.ndim != 2 or points.shape[1] != 3:
                            raise ValueError(f"Polyline {index} must have shape [N, 3]")
                        if len(points) == 0:
                            raise ValueError(f"Polyline {index} is empty")
                        if not np.isfinite(points).all():
                            raise ValueError(f"Polyline {index} contains non-finite coordinates")
                        if input_order.upper() == "ZYX":
                            output_points = np.ascontiguousarray(points[:, ::-1], dtype="<f4")
                        elif input_order.upper() == "XYZ":
                            output_points = np.ascontiguousarray(points, dtype="<f4")
                        else:
                            raise ValueError("input_order must be XYZ or ZYX")
                        output_points.tofile(points_stream)
                        point_count += len(output_points)
                        polyline_count += 1
                        offset_buffer[buffered_offsets] = point_count
                        buffered_offsets += 1
                        if buffered_offsets == len(offset_buffer):
                            offset_buffer.tofile(offsets_stream)
                            buffered_offsets = 0
                    if buffered_offsets:
                        offset_buffer[:buffered_offsets].tofile(offsets_stream)
            manifest["categories"][category] = {
                "points_file": points_name,
                "offsets_file": offsets_name,
                "point_count": int(point_count),
                "polyline_count": int(polyline_count),
            }
        with (temp / "manifest.json").open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        if destination.exists():
            raise FileExistsError(destination)
        os.replace(temp, destination)
        temp = None
        return manifest
    finally:
        if temp is not None and temp.exists():
            shutil.rmtree(temp, ignore_errors=True)


def validate_geometry_snapshot(
    directory: str | os.PathLike[str],
    *,
    max_points: int = 1_000_000_000,
) -> dict[str, object]:
    directory = Path(directory)
    with (directory / "manifest.json").open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
        raise ValueError("Unsupported geometry snapshot schema")
    if (manifest.get("coordinate_order"), manifest.get("dtype"), manifest.get("offset_dtype"), manifest.get("byte_order")) != (
        "XYZ", "float32", "uint64", "little"
    ):
        raise ValueError("Unsupported geometry snapshot encoding")
    for name, entry in manifest.get("categories", {}).items():
        point_count = int(entry["point_count"])
        polyline_count = int(entry["polyline_count"])
        if point_count < 0 or point_count > max_points or polyline_count < 0 or polyline_count > point_count:
            raise ValueError(f"Invalid counts for category {name}")
        points_path = directory / entry["points_file"]
        offsets_path = directory / entry["offsets_file"]
        if points_path.stat().st_size != point_count * 3 * 4:
            raise ValueError(f"Point file size mismatch for category {name}")
        if offsets_path.stat().st_size != (polyline_count + 1) * 8:
            raise ValueError(f"Offset file size mismatch for category {name}")
        offsets = np.memmap(offsets_path, mode="r", dtype="<u8")
        if not len(offsets) or offsets[0] != 0 or offsets[-1] != point_count or np.any(offsets[1:] < offsets[:-1]):
            raise ValueError(f"Invalid offsets for category {name}")
        if np.any(offsets[1:] == offsets[:-1]):
            raise ValueError(f"Zero-length polyline in category {name}")
        points = np.memmap(points_path, mode="r", dtype="<f4", shape=(point_count, 3))
        if not np.isfinite(points).all():
            raise ValueError(f"Non-finite coordinate in category {name}")
    return manifest
