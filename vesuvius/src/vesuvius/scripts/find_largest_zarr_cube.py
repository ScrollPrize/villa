"""Find the largest chunk-aligned cube fully encoded in a Zarr v2 array."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import fsspec
import numpy as np
from vesuvius.data.zarr_chunk_index import build_chunk_occupancy


@dataclass(frozen=True)
class EncodedCube:
    """Largest fully occupied cube, expressed in ZYX array order."""

    side_voxels: int
    chunk_start_zyx: tuple[int, int, int]
    chunk_stop_zyx: tuple[int, int, int]
    voxel_start_zyx: tuple[int, int, int]
    voxel_stop_zyx: tuple[int, int, int]


@dataclass(frozen=True)
class ZarrArrayLayout:
    url: str
    shape_zyx: tuple[int, int, int]
    chunks_zyx: tuple[int, int, int]


@dataclass(frozen=True)
class ZarrSpatialLayout:
    arrays: tuple[ZarrArrayLayout, ...]
    search_shape_zyx: tuple[int, int, int]
    search_chunks_zyx: tuple[int, int, int]
    coordinate_origin_zyx: tuple[int, int, int]
    volume_shape_zyx: tuple[int, int, int]


def _integral_volume(occupancy: np.ndarray) -> np.ndarray:
    dtype = np.uint32 if occupancy.size <= np.iinfo(np.uint32).max else np.uint64
    integral = np.zeros(tuple(size + 1 for size in occupancy.shape), dtype=dtype)
    integral[1:, 1:, 1:] = occupancy
    for axis in range(3):
        np.cumsum(integral, axis=axis, dtype=dtype, out=integral)
    return integral


def _first_full_window(
    integral: np.ndarray,
    window_zyx: tuple[int, int, int],
    start_counts_zyx: tuple[int, int, int],
    workers: int,
) -> tuple[int, int, int] | None:
    wz, wy, wx = window_zyx
    nz, ny, nx = start_counts_zyx
    if min(nz, ny, nx) <= 0:
        return None

    # Keep each worker's one temporary sum array near 8 MiB. Splitting both Z
    # and Y exposes parallel work even for shallow volumes.
    cells_per_task = max(1, (8 * 1024 * 1024) // integral.dtype.itemsize)
    y_block = max(1, min(ny, cells_per_task // max(1, nx)))
    z_block = max(1, min(nz, cells_per_task // max(1, y_block * nx)))
    blocks = [
        (z0, min(nz, z0 + z_block), y0, min(ny, y0 + y_block))
        for z0 in range(0, nz, z_block)
        for y0 in range(0, ny, y_block)
    ]
    target = wz * wy * wx

    def check(block: tuple[int, int, int, int]) -> tuple[int, int, int] | None:
        z0, z1, y0, y1 = block
        z_hi = slice(z0 + wz, z1 + wz)
        z_lo = slice(z0, z1)
        y_hi = slice(y0 + wy, y1 + wy)
        y_lo = slice(y0, y1)
        x_hi = slice(wx, nx + wx)
        x_lo = slice(0, nx)

        sums = np.array(integral[z_hi, y_hi, x_hi], copy=True)
        sums -= integral[z_lo, y_hi, x_hi]
        sums -= integral[z_hi, y_lo, x_hi]
        sums -= integral[z_hi, y_hi, x_lo]
        sums += integral[z_lo, y_lo, x_hi]
        sums += integral[z_lo, y_hi, x_lo]
        sums += integral[z_hi, y_lo, x_lo]
        sums -= integral[z_lo, y_lo, x_lo]

        matches = np.flatnonzero(sums == target)
        if matches.size == 0:
            return None
        local_z, local_y, x = np.unravel_index(int(matches[0]), sums.shape)
        return z0 + int(local_z), y0 + int(local_y), int(x)

    executor = None
    try:
        if workers == 1 or len(blocks) == 1:
            matches = (check(block) for block in blocks)
        else:
            executor = ThreadPoolExecutor(max_workers=min(workers, len(blocks)))
            matches = executor.map(
                check,
                blocks,
                buffersize=max(1, workers * 2),
            )
        # Blocks and matches are ordered lexicographically, and each block
        # returns its first match, so no later block can improve this result.
        return next((match for match in matches if match is not None), None)
    finally:
        if executor is not None:
            executor.shutdown(cancel_futures=True)


def find_largest_encoded_cube(
    occupancy: np.ndarray,
    chunks_zyx: Sequence[int],
    shape_zyx: Sequence[int],
    *,
    workers: int = 1,
    coordinate_origin_zyx: Sequence[int] = (0, 0, 0),
) -> EncodedCube | None:
    """Return the largest chunk-aligned voxel cube containing only present chunks.

    Chunk dimensions may be anisotropic. In that case candidate cube sides are
    multiples of their least common multiple, ensuring that every cube boundary
    remains aligned to the chunk grid.
    """
    occupancy = np.asarray(occupancy, dtype=bool)
    chunks = tuple(int(value) for value in chunks_zyx)
    shape = tuple(int(value) for value in shape_zyx)
    origin = tuple(int(value) for value in coordinate_origin_zyx)
    if occupancy.ndim != 3 or len(chunks) != 3 or len(shape) != 3 or len(origin) != 3:
        raise ValueError("occupancy, chunks, shape, and origin must all be three-dimensional")
    if min(chunks) <= 0 or min(shape) <= 0:
        raise ValueError("chunk and array dimensions must be positive")
    expected_grid = tuple((size + chunk - 1) // chunk for size, chunk in zip(shape, chunks))
    if occupancy.shape != expected_grid:
        raise ValueError(
            f"occupancy shape {occupancy.shape} does not match chunk grid {expected_grid}"
        )
    if workers < 1:
        raise ValueError("workers must be positive")
    if not occupancy.any():
        return None

    side_unit = math.lcm(*chunks)
    max_units = min(shape) // side_unit
    if max_units == 0:
        return None
    integral = _integral_volume(occupancy)

    best_side = 0
    best_start: tuple[int, int, int] | None = None
    low, high = 1, max_units
    while low <= high:
        units = (low + high) // 2
        side = units * side_unit
        window = tuple(side // chunk for chunk in chunks)
        start_counts = tuple(
            min(grid - width + 1, (logical - side) // chunk + 1)
            for grid, width, logical, chunk in zip(occupancy.shape, window, shape, chunks)
        )
        start = _first_full_window(integral, window, start_counts, workers)
        if start is None:
            high = units - 1
        else:
            best_side = side
            best_start = start
            low = units + 1

    if best_start is None:
        return None
    chunk_width = tuple(best_side // chunk for chunk in chunks)
    voxel_start = tuple(
        offset + index * chunk for offset, index, chunk in zip(origin, best_start, chunks)
    )
    return EncodedCube(
        side_voxels=best_side,
        chunk_start_zyx=best_start,
        chunk_stop_zyx=tuple(start + width for start, width in zip(best_start, chunk_width)),
        voxel_start_zyx=voxel_start,
        voxel_stop_zyx=tuple(start + best_side for start in voxel_start),
    )


def _read_json(fs, path: str) -> dict | None:
    try:
        with fs.open(path, "rb") as stream:
            return json.loads(stream.read())
    except FileNotFoundError:
        return None


def _array_layout(url: str, metadata: dict) -> ZarrArrayLayout:
    shape = tuple(int(value) for value in metadata.get("shape", ()))
    chunks = tuple(int(value) for value in metadata.get("chunks", ()))
    if len(shape) != 3 or len(chunks) != 3:
        raise ValueError(
            f"largest-cube search requires a 3D Zarr array, got shape={shape}, chunks={chunks}"
        )
    return ZarrArrayLayout(url=url.rstrip("/"), shape_zyx=shape, chunks_zyx=chunks)


def _join_url(url: str, child: str) -> str:
    return f"{url.rstrip('/')}/{child.strip('/')}"


def _resolve_fiberlet_group(
    url: str,
    fs,
    path: str,
    attributes: dict,
) -> ZarrSpatialLayout:
    try:
        array_names = tuple(attributes["processing"]["layout"]["arrays"])
        grid_shape = tuple(int(value) for value in attributes["chunk_grid_shape_zyx"])
        coordinate_units = tuple(
            int(value) for value in attributes["coordinate_units_per_chunk_zyx"]
        )
        coordinate_origin = tuple(int(value) for value in attributes["coordinate_origin_zyx"])
        spatial_chunk_side = int(attributes["spatial_chunk_side_base"])
        prediction_to_base = float(attributes["prediction_to_base"])
        prediction_shape = tuple(
            int(value) for value in attributes["processing"]["grid"]["shape_zyx"]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"fiberlet group {url!r} has incomplete spatial metadata") from error
    if not array_names or any(len(values) != 3 for values in (
        grid_shape,
        coordinate_units,
        coordinate_origin,
        prediction_shape,
    )):
        raise ValueError(f"fiberlet group {url!r} has invalid spatial metadata")
    if spatial_chunk_side <= 0 or prediction_to_base <= 0:
        raise ValueError(f"fiberlet group {url!r} has invalid coordinate scaling")

    arrays = []
    for name in array_names:
        child_path = f"{path}/{name}"
        metadata = _read_json(fs, f"{child_path}/.zarray")
        if metadata is None:
            raise ValueError(f"fiberlet group {url!r} is missing array {name!r}")
        array = _array_layout(_join_url(url, name), metadata)
        if array.shape_zyx != grid_shape:
            raise ValueError(
                f"fiberlet array {name!r} shape {array.shape_zyx} does not match grid {grid_shape}"
            )
        arrays.append(array)

    origin_base = []
    for origin, units in zip(coordinate_origin, coordinate_units):
        if units <= 0 or spatial_chunk_side % units != 0:
            raise ValueError(f"fiberlet group {url!r} has non-integral base coordinate scaling")
        origin_base.append(origin * (spatial_chunk_side // units))
    full_shape_base = tuple(round(size * prediction_to_base) for size in prediction_shape)
    local_shape_base = tuple(
        min(grid * spatial_chunk_side, full - origin)
        for grid, full, origin in zip(grid_shape, full_shape_base, origin_base)
    )
    if min(local_shape_base) <= 0:
        raise ValueError(f"fiberlet group {url!r} lies outside its declared volume")
    expected_grid = tuple(
        (size + spatial_chunk_side - 1) // spatial_chunk_side for size in local_shape_base
    )
    if expected_grid != grid_shape:
        raise ValueError(
            f"fiberlet group {url!r} grid {grid_shape} does not cover declared local shape "
            f"{local_shape_base}"
        )
    return ZarrSpatialLayout(
        arrays=tuple(arrays),
        search_shape_zyx=local_shape_base,
        search_chunks_zyx=(spatial_chunk_side,) * 3,
        coordinate_origin_zyx=tuple(origin_base),
        volume_shape_zyx=full_shape_base,
    )


def resolve_zarr_v2_layout(
    url: str,
    *,
    anon: bool = False,
) -> ZarrSpatialLayout:
    """Resolve a direct array, OME-Zarr root, or fiberlet dataset group."""
    storage_options = {"anon": True} if anon and url.startswith("s3://") else {}
    fs, path = fsspec.core.url_to_fs(url, **storage_options)
    path = path.rstrip("/")
    metadata = _read_json(fs, f"{path}/.zarray")
    resolved_url = url.rstrip("/")
    if metadata is not None:
        array = _array_layout(resolved_url, metadata)
        return ZarrSpatialLayout(
            arrays=(array,),
            search_shape_zyx=array.shape_zyx,
            search_chunks_zyx=array.chunks_zyx,
            coordinate_origin_zyx=(0, 0, 0),
            volume_shape_zyx=array.shape_zyx,
        )

    attributes = _read_json(fs, f"{path}/.zattrs") or {}
    if attributes.get("vc_format") == "fiberlet_dataset":
        return _resolve_fiberlet_group(resolved_url, fs, path, attributes)

    multiscales = attributes.get("multiscales")
    try:
        dataset_path = str(multiscales[0]["datasets"][0]["path"])
    except (KeyError, IndexError, TypeError):
        raise ValueError(
            f"{url!r} is not a supported Zarr v2 array, OME-Zarr group, or fiberlet group"
        ) from None
    child_url = _join_url(resolved_url, dataset_path)
    child_metadata = _read_json(fs, f"{path}/{dataset_path}/.zarray")
    if child_metadata is None:
        raise ValueError(f"OME-Zarr group {url!r} is missing dataset {dataset_path!r}")
    array = _array_layout(child_url, child_metadata)
    return ZarrSpatialLayout(
        arrays=(array,),
        search_shape_zyx=array.shape_zyx,
        search_chunks_zyx=array.chunks_zyx,
        coordinate_origin_zyx=(0, 0, 0),
        volume_shape_zyx=array.shape_zyx,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Find the largest chunk-aligned cube whose Zarr v2 chunks all exist."
    )
    parser.add_argument("zarr", help="3D Zarr v2 array, OME-Zarr root, or fiberlet group")
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help="parallel listing/search workers (default: detected host CPU count)",
    )
    parser.add_argument("--cache", action="store_true", help="read/write the chunk occupancy cache")
    parser.add_argument("--anon", action="store_true", help="use anonymous access for S3 input")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        layout = resolve_zarr_v2_layout(args.zarr, anon=args.anon)
        output = sys.stderr if args.json else sys.stdout
        with contextlib.redirect_stdout(output):
            occupancies = [
                build_chunk_occupancy(
                    array.url,
                    array.chunks_zyx,
                    array.shape_zyx,
                    verbose=not args.json,
                    use_cache=args.cache,
                    anon=args.anon,
                    workers=args.workers,
                )
                for array in layout.arrays
            ]
        if any(occupancy is None for occupancy in occupancies):
            raise ValueError(f"could not build all chunk occupancy indexes for {args.zarr}")
        occupancy = np.logical_and.reduce(occupancies)
        cube = find_largest_encoded_cube(
            occupancy,
            layout.search_chunks_zyx,
            layout.search_shape_zyx,
            workers=args.workers,
            coordinate_origin_zyx=layout.coordinate_origin_zyx,
        )
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    if cube is None:
        message = json.dumps({"zarr": args.zarr, "cube": None}) if args.json else "No encoded cube found."
        print(message)
        return 1

    if args.json:
        print(
            json.dumps(
                {
                    "zarr": args.zarr,
                    "volume_shape_zyx": layout.volume_shape_zyx,
                    "cube": {
                        "side_voxels": cube.side_voxels,
                        "voxel_start_zyx": cube.voxel_start_zyx,
                        "voxel_stop_zyx": cube.voxel_stop_zyx,
                    },
                },
                sort_keys=True,
            )
        )
    else:
        print(f"zarr: {args.zarr}")
        print("volume_shape_zyx: " + ",".join(map(str, layout.volume_shape_zyx)))
        print(f"side_voxels: {cube.side_voxels}")
        print("voxel_start_zyx: " + ",".join(map(str, cube.voxel_start_zyx)))
        print("voxel_stop_zyx: " + ",".join(map(str, cube.voxel_stop_zyx)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
