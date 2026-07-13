#!/usr/bin/env python3
"""Stage a bounded ZYX Zarr ROI for inference without exposing remote I/O to it."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from urllib.parse import unquote, urlsplit

import numpy as np

MAX_VOXELS = 16_777_216
MAX_CHUNKS = 64
MAX_CHUNK_BYTES = 64 * 1024 * 1024
MAX_AXIS = 256
ALLOWED_HTTPS_HOSTS = {
    "vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com",
    "dl.ash2txt.org",
}
ALLOWED_S3_BUCKETS = {"vesuvius-challenge-open-data"}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def allowed_remote_uri(uri: str) -> bool:
    """Apply the staging allowlist again at the network boundary."""
    try:
        parsed = urlsplit(uri)
    except ValueError:
        return False
    decoded_path = unquote(parsed.path)
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        return False
    if any(part == ".." for part in decoded_path.split("/")):
        return False
    if parsed.scheme == "s3":
        return parsed.hostname in ALLOWED_S3_BUCKETS and bool(parsed.path.strip("/"))
    return (
        parsed.scheme == "https"
        and parsed.hostname in ALLOWED_HTTPS_HOSTS
        and parsed.port in (None, 443)
        and bool(parsed.path.strip("/"))
    )


def safe_array_path(value: object) -> str:
    path = str(value if value is not None else "0").strip("/")
    if not path or len(path) > 128 or any(part in ("", ".", "..") for part in path.split("/")):
        raise ValueError("array_path must be a relative Zarr key without traversal")
    return path


def xyz_vector(source: dict, key: str, default: tuple[float, float, float]) -> list[float]:
    raw = source.get(key, default)
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise ValueError(f"source.{key} must contain three XYZ values")
    result = [float(value) for value in raw]
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"source.{key} values must be finite")
    if key == "voxel_spacing" and not all(value > 0 for value in result):
        raise ValueError("source.voxel_spacing values must be positive")
    return result


def open_array(source: dict):
    import zarr

    kind = source.get("kind")
    array_path = safe_array_path(source.get("array_path", "0"))
    if kind == "local_zarr":
        location = str(Path(source["path"]).resolve())
        root = zarr.open(location, mode="r")
    elif kind == "remote_zarr":
        location = str(source["uri"])
        if not allowed_remote_uri(location):
            raise ValueError("remote Zarr URI is outside the public allowlist")
        import fsspec

        options = {"anon": True} if location.startswith("s3://") else {}
        root = zarr.open(fsspec.get_mapper(location, **options), mode="r")
    else:
        raise ValueError("source.kind must be local_zarr or remote_zarr")
    array = root if hasattr(root, "shape") else root[array_path]
    return array, location, array_path


def stage(request: dict, output: Path) -> dict:
    source = request["source"]
    region = request["region"]
    array, location, array_path = open_array(source)
    if array.ndim != 3:
        raise ValueError(f"Zarr array must be rank 3 ZYX, got {array.shape}")
    dtype = np.dtype(array.dtype)
    if dtype not in (np.dtype("uint8"), np.dtype("uint16")):
        raise ValueError(f"unsupported dtype {array.dtype}")

    x, y, z = (int(region[key]) for key in ("x", "y", "z"))
    width, height, depth = (int(region[key]) for key in ("width", "height", "depth"))
    if min(x, y, z) < 0 or min(width, height, depth) < 1 or max(width, height, depth) > MAX_AXIS:
        raise ValueError("invalid or excessive ROI")
    if width * height * depth > MAX_VOXELS:
        raise ValueError("ROI exceeds the 16M voxel staging limit")
    if z + depth > array.shape[0] or y + height > array.shape[1] or x + width > array.shape[2]:
        raise ValueError("ROI is outside Zarr bounds")

    chunks = tuple(int(value) for value in array.chunks)
    if len(chunks) != 3 or any(value < 1 or value > 512 for value in chunks):
        raise ValueError("Zarr chunks must be rank 3 and at most 512 on each axis")
    if math.prod(chunks) * dtype.itemsize > MAX_CHUNK_BYTES:
        raise ValueError("uncompressed Zarr chunk exceeds 64 MiB")
    touched = math.prod(
        math.ceil((start + size) / chunk) - start // chunk
        for start, size, chunk in zip((z, y, x), (depth, height, width), chunks)
    )
    if touched > MAX_CHUNKS:
        raise ValueError(f"ROI touches {touched} chunks; maximum is {MAX_CHUNKS}")

    data = np.asarray(array[z : z + depth, y : y + height, x : x + width])
    if data.shape != (depth, height, width):
        raise ValueError("Zarr returned an incomplete ROI")
    output.mkdir(parents=True, exist_ok=True)
    staged = output / "staged-volume.npy"
    np.save(staged, data, allow_pickle=False)

    spacing = xyz_vector(source, "voxel_spacing", (1.0, 1.0, 1.0))
    spacing_unit = str(source.get("voxel_spacing_unit", "um"))
    if spacing_unit != "um":
        raise ValueError("source.voxel_spacing_unit must be um")
    source_origin = xyz_vector(source, "origin_xyz", (0.0, 0.0, 0.0))
    output_origin = [
        source_origin[0] + x * spacing[0],
        source_origin[1] + y * spacing[1],
        source_origin[2] + z * spacing[2],
    ]
    manifest = {
        "kind": "controlled_zarr_stage_v1",
        "source_kind": source["kind"],
        "source": location,
        "submitted_region_xyz": dict(region),
        "array_slices_zyx": {
            "z": {"start": z, "stop": z + depth},
            "y": {"start": y, "stop": y + height},
            "x": {"start": x, "stop": x + width},
        },
        "scale": int(source.get("scale", 0)),
        "array_path": array_path,
        "voxel_spacing": spacing,
        "voxel_spacing_unit": spacing_unit,
        "voxel_spacing_explicit": "voxel_spacing" in source,
        "origin_xyz": output_origin,
        "source_origin_xyz": source_origin,
        "axes": ["z", "y", "x"],
        "source_shape_zyx": list(array.shape),
        "chunks_zyx": list(chunks),
        "dtype": str(dtype),
        "staged_shape_zyx": list(data.shape),
        "chunks_touched": touched,
        "staged_path": str(staged),
        "staged_sha256": digest(staged),
    }
    (output / "stage-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    request = json.loads(Path(arguments.request).read_text())
    manifest = stage(request, Path(arguments.output))
    print(json.dumps(manifest), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
