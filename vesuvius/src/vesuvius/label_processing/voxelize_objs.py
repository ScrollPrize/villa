#!/usr/bin/env python
import math
import numpy as np
import os
import shutil
from pathlib import Path
import logging
from itertools import repeat, product
from vc_reader.reader import ZarrWriter, create_group, open_zarr
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing
import sys
import argparse
from tqdm import tqdm  # Progress bar

from vc_reader.affine import Affine
from vc_reader.geometry import transform_points
from vc_reader import tifxyz

# Determine default number of workers: half of CPU count (at least 1)
default_workers = max(1, multiprocessing.cpu_count() // 2)

PYRAMID_LEVELS = 6

# Shared immutable voxel data and output handles for slice worker threads.
_SLICE_STATE = {}
_SLICE_DATASETS = {}
_DOWNSAMPLE_STATE = {}
_DOWNSAMPLE_DATASETS = {}

logger = logging.getLogger(__name__)


def _set_slice_state(state):
    global _SLICE_STATE
    global _SLICE_DATASETS
    _SLICE_STATE = state
    _SLICE_DATASETS = {}


def _get_worker_datasets():
    state = _SLICE_STATE
    if not state:
        raise RuntimeError("Slice processing state is not initialized in the worker.")

    if state.get("format", "zarr") != "zarr":
        return None, None, None

    labels_ds = _SLICE_DATASETS.get("labels")
    if labels_ds is None:
        labels_ds = ZarrWriter.open(state["label_dataset_name"], auth=None)
        _SLICE_DATASETS["labels"] = labels_ds

    normals_ds = None
    if state.get("write_normals", False):
        normals_ds = _SLICE_DATASETS.get("normals")
        if normals_ds is None:
            normals_ds = ZarrWriter.open(state["normals_dataset_name"], auth=None)
            _SLICE_DATASETS["normals"] = normals_ds

    frame_ds = None
    if state.get("include_surface_frame", False):
        frame_ds = _SLICE_DATASETS.get("surface_frame")
        if frame_ds is None:
            frame_ds = ZarrWriter.open(state["surface_frame_dataset_name"], auth=None)
            _SLICE_DATASETS["surface_frame"] = frame_ds

    return labels_ds, normals_ds, frame_ds


def _write_sparse_slice(dataset, slice_index, slice_data):
    """
    Write a single z slice into a chunked zarr dataset while skipping all-zero chunks.
    Falls back to direct assignment if chunking along z is not 1 voxel.
    """
    if dataset is None or slice_data.size == 0:
        return

    chunks = dataset.chunks
    if not chunks or chunks[0] != 1:
        dataset[slice_index, ...] = slice_data
        return

    # Ensure we iterate over chunk-aligned windows across the remaining axes.
    data_shape = slice_data.shape
    axis_ranges = []
    for axis, dim in enumerate(data_shape):
        chunk_extent = chunks[axis + 1] if axis + 1 < len(chunks) else dim
        axis_ranges.append(range(0, dim, chunk_extent))

    for offsets in product(*axis_ranges):
        slices = []
        for axis, start in enumerate(offsets):
            chunk_extent = chunks[axis + 1] if axis + 1 < len(chunks) else data_shape[axis]
            stop = min(start + chunk_extent, data_shape[axis])
            slices.append(slice(start, stop))
        slices_tuple = tuple(slices)
        tile = slice_data[slices_tuple]
        if not np.any(tile):
            continue
        dataset[(slice_index,) + slices_tuple] = tile


def _write_tif_slice(output_dir, slice_index, slice_data, digits, compression):
    from vc_reader import images

    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"{slice_index:0{digits}d}.tif")
    images.write_tiff(filename, slice_data, compression=compression)


def _write_slice_output_worker(zslice, label_img, normals_img, frame_img):
    state = _SLICE_STATE
    slice_index = int(zslice)
    volume_depth = state.get("volume_depth")
    if volume_depth is not None and (slice_index < 0 or slice_index >= volume_depth):
        raise ValueError(
            f"Requested slice index {slice_index} is outside the allocated z range "
            f"[0, {volume_depth - 1}]."
        )

    fmt = state.get("format", "zarr")
    if fmt == "zarr":
        labels_ds, normals_ds, frame_ds = _get_worker_datasets()
        if labels_ds is None:
            raise RuntimeError("Label dataset is not available for zarr output.")
        label_data = np.ascontiguousarray(label_img, dtype=labels_ds.dtype)
        _write_sparse_slice(labels_ds, slice_index, label_data)

        if (
            state.get("write_normals", False)
            and normals_img is not None
            and normals_img.size
            and normals_ds is not None
        ):
            normals_data = np.ascontiguousarray(
                normals_img.astype(state["normals_dtype"], copy=False)
            )
            _write_sparse_slice(normals_ds, slice_index, normals_data)

        if (
            state.get("include_surface_frame", False)
            and frame_ds is not None
            and frame_img is not None
            and frame_img.size
        ):
            frame_data = np.ascontiguousarray(
                frame_img.astype(state["surface_frame_dtype"], copy=False)
            )
            _write_sparse_slice(frame_ds, slice_index, frame_data)
    elif fmt == "tifs":
        label_dtype = state.get("label_dtype", np.uint16)
        label_data = np.ascontiguousarray(label_img, dtype=label_dtype)
        _write_tif_slice(
            state["label_store_path"],
            slice_index,
            label_data,
            state.get("tif_slice_digits", 5),
            state.get("tif_compression", "deflate"),
        )
    else:
        raise ValueError(f"Unsupported output format '{fmt}'.")


def _process_zslice(zslice):
    state = _SLICE_STATE
    if not state:
        raise RuntimeError("Slice processing state is not initialized in the worker.")

    _, label_img, normals_img, frame_img = process_slice(zslice, state)

    _write_slice_output_worker(zslice, label_img, normals_img, frame_img)
    return zslice


def _set_downsample_state(state):
    global _DOWNSAMPLE_STATE
    global _DOWNSAMPLE_DATASETS
    _DOWNSAMPLE_STATE = state
    _DOWNSAMPLE_DATASETS = {}


def _get_downsample_datasets():
    state = _DOWNSAMPLE_STATE
    if not state:
        raise RuntimeError("Downsample state is not initialized in the worker.")

    datasets = _DOWNSAMPLE_DATASETS.get("datasets")
    if datasets is None:
        # The process pool already reads chunks in parallel.
        parent_ds = open_zarr(state["parent_dataset_path"], threads=1)
        target_ds = ZarrWriter.open(state["target_dataset_path"], auth=None)
        datasets = (parent_ds, target_ds)
        _DOWNSAMPLE_DATASETS["datasets"] = datasets
    return datasets


def _iter_chunk_starts(shape, chunks):
    if len(shape) < 3 or len(chunks) < 3:
        raise ValueError("Chunk iteration requires at least three dimensions (z, y, x).")

    z_chunk = max(1, chunks[0])
    y_chunk = max(1, chunks[1])
    x_chunk = max(1, chunks[2])

    for z_start in range(0, shape[0], z_chunk):
        for y_start in range(0, shape[1], y_chunk):
            for x_start in range(0, shape[2], x_chunk):
                yield (z_start, y_start, x_start)


def _downsample_chunk_worker(chunk_start):
    state = _DOWNSAMPLE_STATE
    if not state:
        raise RuntimeError("Downsample state is not initialized in the worker.")

    parent_ds, target_ds = _get_downsample_datasets()

    target_shape = state["target_shape"]
    target_chunks = state["target_chunks"]
    parent_shape = state["parent_shape"]
    down_axes = state["downsample_axes"]
    ndim = target_ds.ndim

    if len(chunk_start) != 3:
        raise ValueError("Chunk start must contain three axes (z, y, x).")

    z_start, y_start, x_start = chunk_start

    z_chunk = max(1, target_chunks[0])
    y_chunk = max(1, target_chunks[1])
    x_chunk = max(1, target_chunks[2])

    z_stop = min(z_start + z_chunk, target_shape[0])
    y_stop = min(y_start + y_chunk, target_shape[1])
    x_stop = min(x_start + x_chunk, target_shape[2])

    target_slices = [
        slice(z_start, z_stop),
        slice(y_start, y_stop),
        slice(x_start, x_stop),
    ]

    parent_slices = []
    for axis, (target_slice, parent_dim) in enumerate(zip(target_slices, parent_shape)):
        start = target_slice.start
        stop = target_slice.stop
        if axis in down_axes:
            parent_start = start * 2
            parent_stop = min(parent_dim, stop * 2)
        else:
            parent_start = start
            parent_stop = min(parent_dim, parent_start + (stop - start))
        parent_slices.append(slice(parent_start, parent_stop))

    if ndim > 3:
        for axis in range(3, ndim):
            target_slices.append(slice(0, target_shape[axis]))
            parent_slices.append(slice(0, parent_shape[axis]))

    target_slices = tuple(target_slices)
    parent_slices = tuple(parent_slices)

    source = np.asarray(parent_ds[parent_slices])
    downsampled = downsample_2x(source, axes=down_axes)

    expected_shape = tuple(s.stop - s.start for s in target_slices)
    if downsampled.shape != expected_shape:
        raise ValueError(
            f"Downsampled chunk shape {downsampled.shape} does not match target chunk shape {expected_shape}."
        )

    target_ds[target_slices] = np.ascontiguousarray(downsampled, dtype=target_ds.dtype)
    return chunk_start


def _rechunk(zarr_array, chunk_size, desc, num_workers):
    if zarr_array is None or chunk_size <= 0:
        return zarr_array

    # Only adjust the spatial axes (z, y, x) to keep channel chunking intact.
    current_chunks = list(zarr_array.chunks)
    target_chunks = list(current_chunks)
    ndim = zarr_array.ndim
    changed = False

    for axis in range(min(3, ndim)):
        desired = min(chunk_size, zarr_array.shape[axis])
        if target_chunks[axis] != desired:
            target_chunks[axis] = desired
            changed = True

    if not changed:
        print(f"{desc} already uses requested chunk size; skipping rechunk.", flush=True)
        return zarr_array

    target_chunks = tuple(target_chunks)
    print(f"Rechunking {desc} to chunks {target_chunks}", flush=True)

    path = zarr_array.path
    tmp_path = f"{path}__rechunk_tmp"
    tmp_array = ZarrWriter.create_like(tmp_path, zarr_array, chunks=target_chunks, overwrite=True, auth=None)
    tmp_array.attrs.put(dict(zarr_array.attrs))

    def copy_chunk(start):
        region = tuple(slice(s, min(s + c, n)) for s, c, n in zip(start, target_chunks, zarr_array.shape))
        tmp_array[region] = zarr_array[region]

    starts = list(product(*(range(0, n, c) for n, c in zip(zarr_array.shape, target_chunks))))
    with ThreadPoolExecutor(max_workers=max(1, num_workers)) as executor:
        for _ in tqdm(executor.map(copy_chunk, starts), total=len(starts), desc=f"Rechunk {desc}", leave=False):
            pass

    shutil.rmtree(path)
    os.rename(tmp_path, path)
    return ZarrWriter.open(path, auth=None)


def downsample_2x(array, axes):
    """Downsample array by a factor of 2 along the provided axes."""
    ndim = array.ndim
    norm_axes = [(axis + ndim) % ndim for axis in axes]
    slices = [slice(None)] * ndim
    for axis in norm_axes:
        slices[axis] = slice(0, None, 2)
    return array[tuple(slices)]


def build_pyramid(array, levels, axes):
    """Return a list of arrays representing a 2x pyramid for the input array."""
    outputs = [array]
    for _ in range(1, levels):
        next_level = downsample_2x(outputs[-1], axes)
        outputs.append(next_level)
    return outputs


def find_surfaces(folder, recursive=False):
    """Accept a bundle itself, its parent, or (with --recursive) a directory tree."""
    path = Path(folder)
    if all((path / name).is_file() for name in ("meta.json", "x.tif", "y.tif", "z.tif")):
        return [str(path)]
    return sorted(entry.location for entry in tifxyz.list_surfaces(path, recursive=recursive))


def _map_geometry(values, transform_info, *, vectors=False):
    if transform_info is None or not len(values):
        return values
    perm = transform_info["perm"]
    return transform_points(
        np.ascontiguousarray(values[:, perm]), transform_info["matrix"], vectors=vectors,
    )[:, np.argsort(perm)]


def _unit_vectors(vectors):
    lengths = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return np.divide(vectors, lengths, out=np.zeros_like(vectors), where=lengths > 1e-12)


def _nearest_samples(ids, distances):
    # Stable ties keep the first sample in deterministic tile/query order.
    order = np.lexsort((distances, ids))
    return order[np.r_[True, ids[order][1:] != ids[order][:-1]]]


def process_surface(surface_path, include_normals, include_surface_frame,
                    transform_info, spatial_shape):
    """Sample bicubic patches into sparse voxels, without constructing a mesh.

    Sample each stored axis at quarter-voxel spacing estimated from its longest
    transformed grid edge. This is point-based coverage, not an exact voxel/patch
    intersection test. UV metadata scale does not rescale XYZ coordinate values.
    Each voxel keeps its nearest sample (stable ties); optional directions come
    from that same bicubic patch. Work on at most 256 x 256 queries at a time.
    """
    surface = tifxyz.read(surface_path)
    shape = tuple(spatial_shape)
    need_frame = include_normals or include_surface_frame
    empty = (np.empty(0, np.int64), np.empty((0, 3), np.float32),
             np.empty((0, 3, 3), np.float32))
    if min(surface.shape) < 2 or not surface.valid.any():
        return empty

    # Transform only valid coordinates: missing vertices remain missing even
    # when an affine translates the sentinel into the output volume.
    grid = np.full(surface.xyz.shape, np.nan, dtype=np.float32)
    grid[surface.valid] = _map_geometry(surface.xyz[surface.valid], transform_info)
    subdivisions = []
    for axis in (0, 1):
        lengths = np.linalg.norm(np.diff(grid, axis=axis), axis=-1)
        finite = lengths[np.isfinite(lengths)]
        longest = float(finite.max()) if finite.size else 0.0
        subdivisions.append(max(1, int(math.ceil(longest * 4))))
    counts = [(n - 1) * steps + 1 for n, steps in zip(surface.shape, subdivisions)]
    ids_parts, distance_parts, normal_parts, frame_parts = [], [], [], []
    for row_start in range(0, counts[0], 256):
        for col_start in range(0, counts[1], 256):
            rows, cols = np.meshgrid(
                np.arange(row_start, min(row_start + 256, counts[0]), dtype=np.float32) / subdivisions[0],
                np.arange(col_start, min(col_start + 256, counts[1]), dtype=np.float32) / subdivisions[1],
                indexing="ij",
            )
            sampled = surface.points_at(rows, cols, interpolation="bicubic", tangents=need_frame)
            xyz, valid = sampled[:2]
            xyz = _map_geometry(xyz[valid], transform_info)
            # Reject out-of-volume samples rather than flattening the surface
            # onto the volume boundary by clamping its control points.
            inside = np.isfinite(xyz).all(axis=1) & (xyz >= 0).all(axis=1)
            inside &= (xyz < np.asarray(shape[::-1]) - 0.5).all(axis=1)
            xyz = xyz[inside]
            if not len(xyz):
                continue
            ijk = np.floor(xyz + 0.5).astype(np.int64)
            ids = np.ravel_multi_index(ijk[:, ::-1].T, shape)
            distances = np.sum((xyz - ijk) ** 2, axis=1)
            take = _nearest_samples(ids, distances)
            ids_parts.append(ids[take])
            distance_parts.append(distances[take])
            if need_frame:
                du = _map_geometry(sampled[2][valid][inside][take], transform_info, vectors=True)
                dv = _map_geometry(sampled[3][valid][inside][take], transform_info, vectors=True)
                normal = _unit_vectors(np.cross(du, dv))
                normal_parts.append(normal)
                if include_surface_frame:
                    tangent = _unit_vectors(du)
                    bitangent = _unit_vectors(np.cross(normal, tangent))
                    frame_parts.append(np.stack((tangent, bitangent, normal), axis=1))
    if not ids_parts:
        return empty
    ids = np.concatenate(ids_parts)
    take = _nearest_samples(ids, np.concatenate(distance_parts))
    normals = np.concatenate(normal_parts)[take] if need_frame else empty[1]
    frames = np.concatenate(frame_parts)[take] if include_surface_frame else empty[2]
    if need_frame and (not np.isfinite(normals).all() or
                       np.any(np.linalg.norm(normals, axis=1) < 1e-12)):
        raise ValueError(f"Degenerate surface normal in {surface_path}; check the surface and affine.")
    return ids[take], normals, frames


def process_slice(zslice, state):
    """Write sorted surfaces in label order; later surfaces win overlaps."""
    h, w = state["h"], state["w"]
    label_img = np.zeros((h, w), dtype=state["label_dtype"])
    normals_img = np.zeros((h, w, 3), np.float32) if state["include_normals"] else None
    frame_img = np.zeros((h, w, 3, 3), np.float32) if state["include_surface_frame"] else None
    start, stop = int(zslice) * h * w, (int(zslice) + 1) * h * w
    for label, (ids, normals, frames) in enumerate(state["surfaces"], start=1):
        lo, hi = np.searchsorted(ids, (start, stop))
        pixels = ids[lo:hi] - start
        label_img.reshape(-1)[pixels] = label
        if normals_img is not None:
            normals_img.reshape(-1, 3)[pixels] = normals[lo:hi]
        if frame_img is not None:
            frame_img.reshape(-1, 3, 3)[pixels] = frames[lo:hi]
    return zslice, label_img, normals_img, frame_img


def main():
    # Parse command-line arguments.
    parser = argparse.ArgumentParser(
        description="Voxelize tifxyz surfaces using bicubic interpolation into OME-Zarr volumes or TIFF stacks."
    )
    parser.add_argument("folder",
                        help="Path to a tifxyz directory or a folder containing tifxyz directories")
    parser.add_argument(
        "--spatial-shape",
        required=True,
        type=int,
        nargs=3,
        metavar=("Z", "Y", "X"),
        help="Total output spatial shape as integers in the order (z, y, x)",
    )
    parser.add_argument(
        "--transform",
        help="Path to JSON file containing an affine transform (3x4 or 4x4, row-major)",
    )
    parser.add_argument(
        "--transform-axis-order",
        default="xyz",
        choices=["xyz", "xzy", "yxz", "yzx", "zxy", "zyx"],
        help="Axis order used by the affine transform definition (default: xyz)",
    )
    parser.add_argument(
        "--transform-invert",
        action="store_true",
        help="Invert the provided affine before applying",
    )
    parser.add_argument("--output_path", default="mesh_labels.zarr",
                        help="Path to the label OME-Zarr store (default: mesh_labels.zarr)")
    parser.add_argument("--num_workers", type=int, default=default_workers,
                        help="Number of workers to use (default: half of CPU count)")
    parser.add_argument("--recursive", action="store_true",
                        help="Search recursively for tifxyz directories below the input folder")
    parser.add_argument("--chunk_size", type=int, default=0,
                        help="Override chunk edge length for Zarr datasets (use 0 for auto)")
    parser.add_argument("--label-dtype", default="uint16",
                        choices=["uint8", "uint16"],
                        help="Unsigned integer dtype for the label volume (default: uint16)")
    parser.add_argument("--format", default="zarr", choices=["zarr", "tifs"],
                        help="Output format for the label volume (default: zarr)")
    parser.add_argument("--output_normals", action="store_true",
                        help="Also export per-voxel surface normals as an OME-Zarr pyramid")
    parser.add_argument("--normals_output_path", default="mesh_normals.zarr",
                        help="Path to the normals OME-Zarr store (default: mesh_normals.zarr)")
    parser.add_argument("--normals_dtype", default="float16",
                        help="Floating point dtype for the normals pyramid (must be <= float16)")
    parser.add_argument("--output_surface_frame", action="store_true",
                        help="Export per-voxel surface frames (t_u, t_v, n) as an OME-Zarr pyramid")
    parser.add_argument("--surface_frame_output_path", default="mesh_surface_frame.zarr",
                        help="Path to the surface frame OME-Zarr store (default: mesh_surface_frame.zarr)")
    parser.add_argument("--surface_frame_dtype", default="float16",
                        help="Floating point dtype for the surface frame pyramid (must be <= float16)")
    args = parser.parse_args()

    label_dtype = np.dtype(args.label_dtype)
    if label_dtype.kind != "u":
        print("ERROR: label dtype must be an unsigned integer type.")
        sys.exit(1)
    if label_dtype.itemsize > 2:
        print("ERROR: label dtype cannot exceed 16 bits per voxel.")
        sys.exit(1)
    args.label_dtype = label_dtype
    print(f"Label output dtype: {label_dtype}")

    if args.format == "tifs":
        if args.output_normals or args.output_surface_frame:
            print("ERROR: TIFF output format does not support normals or surface frame volumes.")
            sys.exit(1)
        if args.chunk_size > 0:
            print("Warning: chunk_size is ignored when using TIFF output.")
            args.chunk_size = 0

    if args.chunk_size < 0:
        print("ERROR: chunk_size must be non-negative.")
        sys.exit(1)

    transform_info = None
    if args.transform:
        transform_path = args.transform
        if not os.path.exists(transform_path):
            print(f"ERROR: transform file '{transform_path}' does not exist.")
            sys.exit(1)
        try:
            transform = Affine.load_transform_json(str(transform_path), allow_singular=True)
        except Exception as exc:
            print(f"ERROR: failed to load transform: {exc}")
            sys.exit(1)

        if args.transform_invert:
            try:
                transform = transform.inverse()
                print("Applying inverse of provided affine (--transform-invert).")
            except ValueError:
                print("ERROR: Provided affine transform is non-invertible.")
                sys.exit(1)

        try:
            order = args.transform_axis_order.lower()
            if sorted(order) != ["x", "y", "z"]:
                raise ValueError("Transform axis order must be a permutation of xyz")
            perm = tuple("xyz".index(c) for c in order)
        except ValueError as exc:
            print(f"ERROR: {exc}")
            sys.exit(1)

        transform_info = {
            "matrix": np.asarray(transform.matrix, dtype=np.float32),
            "perm": perm,
            "axis_order": args.transform_axis_order,
        }

        print(f"Loaded affine transform from {transform_path}")
        print(f"Affine axis order: {args.transform_axis_order}")

    normals_dtype = None
    if args.output_normals:
        normals_dtype = np.dtype(args.normals_dtype)
        if normals_dtype.kind != 'f':
            print("ERROR: normals dtype must be a floating point type.")
            sys.exit(1)
        if normals_dtype.itemsize * 8 > 16:
            print("ERROR: normals dtype cannot exceed 16 bits per component.")
            sys.exit(1)

    surface_frame_dtype = None
    if args.output_surface_frame:
        surface_frame_dtype = np.dtype(args.surface_frame_dtype)
        if surface_frame_dtype.kind != 'f':
            print("ERROR: surface frame dtype must be a floating point type.")
            sys.exit(1)
        if surface_frame_dtype.itemsize * 8 > 16:
            print("ERROR: surface frame dtype cannot exceed 16 bits per component.")
            sys.exit(1)

    # Use the provided number of workers.
    N_PROCESSES = args.num_workers
    if N_PROCESSES < 1:
        parser.error("num_workers must be positive")
    print(f"Using {N_PROCESSES} workers")

    # Folder containing tifxyz surfaces.
    folder_path = args.folder
    print(f"Using surface folder: {folder_path}")

    z_dim, y_dim, x_dim = args.spatial_shape
    if z_dim <= 0 or y_dim <= 0 or x_dim <= 0:
        print("ERROR: spatial shape dimensions must be positive integers.")
        sys.exit(1)

    h = y_dim
    w = x_dim
    print(
        f"Using spatial shape (z, y, x)=({z_dim}, {y_dim}, {x_dim}) -> (height, width)=({h}, {w})"
    )

    out_path = args.output_path
    print(f"Label output ({args.format}) path: {out_path}")
    if args.format == "tifs":
        if os.path.exists(out_path):
            if not os.path.isdir(out_path):
                print(f"ERROR: TIFF output path '{out_path}' must be a directory.")
                sys.exit(1)
            if any(os.scandir(out_path)):
                print(f"ERROR: TIFF output directory '{out_path}' must be empty.")
                sys.exit(1)
        else:
            os.makedirs(out_path, exist_ok=True)

    normals_out_path = None
    if args.output_normals:
        normals_out_path = args.normals_output_path
        print(f"Normals OME-Zarr output path: {normals_out_path}")

    surface_frame_out_path = None
    if args.output_surface_frame:
        surface_frame_out_path = args.surface_frame_output_path
        print(f"Surface frame OME-Zarr output path: {surface_frame_out_path}")

    surface_paths = find_surfaces(folder_path, args.recursive)
    if not surface_paths:
        parser.error(f"No tifxyz surfaces found in {folder_path}")
    if len(surface_paths) > np.iinfo(label_dtype).max:
        parser.error(f"label dtype {label_dtype} cannot represent {len(surface_paths)} surfaces")
    print(f"Found {len(surface_paths)} surfaces to process")
    # Native surface sampling releases the GIL; threads also avoid copying the
    # sparse results through process pipes and work with every start method.
    with ThreadPoolExecutor(max_workers=N_PROCESSES) as executor:
        surfaces = list(tqdm(executor.map(
            process_surface, surface_paths, repeat(args.output_normals),
            repeat(args.output_surface_frame), repeat(transform_info), repeat(args.spatial_shape),
        ), total=len(surface_paths), desc="Surfaces sampled"))
    occupied = [ids for ids, _, _ in surfaces if len(ids)]
    if occupied:
        z_min = min(int(ids[0] // (h * w)) for ids in occupied)
        z_max = max(int(ids[-1] // (h * w)) for ids in occupied)
        z_slices = np.arange(z_min, z_max + 1)
    else:
        z_slices = np.empty(0, dtype=np.int64)
    print(f"Processing {len(z_slices)} slices with surface bounds.")

    num_levels = PYRAMID_LEVELS
    codec = {"cname": "zstd", "level": 3, "shuffle": 2}

    def compute_level_shape(base_shape, level, downsample_axes):
        shape = list(base_shape)
        factor = 2 ** level
        for axis in downsample_axes:
            shape[axis] = int(math.ceil(shape[axis] / factor))
        return tuple(shape)

    def compute_chunks(shape, chunk_size, level):
        if chunk_size > 0:
            spatial_axes = min(3, len(shape))
            chunks = []
            for axis, dim in enumerate(shape):
                if axis == 0 and level == 0:
                    chunks.append(1)
                elif axis < spatial_axes:
                    chunks.append(min(chunk_size, dim))
                else:
                    chunks.append(dim)
        else:
            chunks = list(shape)
            if chunks:
                chunks[0] = 1
            if len(chunks) > 1:
                chunks[1] = min(512, chunks[1])
            if len(chunks) > 2:
                chunks[2] = min(512, chunks[2])
            for axis in range(3, len(chunks)):
                chunks[axis] = shape[axis]
        return tuple(chunks)

    label_datasets = []
    label_dataset_name = None
    if args.format == "zarr":
        label_root = create_group(out_path, overwrite=True, auth=None)
        label_base_shape = (z_dim, h, w)
        label_axes = [
            {"name": "z", "type": "space", "unit": "index"},
            {"name": "y", "type": "space", "unit": "pixel"},
            {"name": "x", "type": "space", "unit": "pixel"},
        ]
        label_translation = [0.0, 0.0, 0.0]
        label_datasets_meta = []

        for level in range(num_levels):
            level_shape = compute_level_shape(label_base_shape, level, (0, 1, 2))
            level_chunks = compute_chunks(level_shape, args.chunk_size, level)
            dataset_name = f"{level}"
            ds = label_root.create_array(
                dataset_name,
                shape=level_shape,
                chunks=level_chunks,
                dtype=label_dtype,
                overwrite=True,
                **codec,
            )
            label_datasets.append(ds)
            scale_vector = [float(2 ** level), float(2 ** level), float(2 ** level)]
            label_datasets_meta.append(
                {
                    "path": dataset_name,
                    "coordinateTransformations": [
                        {"type": "scale", "scale": scale_vector},
                        {"type": "translation", "translation": list(label_translation)},
                    ],
                }
            )

        label_root.attrs["multiscales"] = [
            {
                "version": "0.4",
                "name": "labels",
                "axes": label_axes,
                "datasets": label_datasets_meta,
            }
        ]
        label_root.attrs["image-label"] = {
            "version": "0.4",
            "colors": [],
        }

        label_dataset_name = label_datasets[0].path

    normals_datasets = []
    if args.output_normals:
        normals_root = create_group(normals_out_path, overwrite=True, auth=None)
        normals_base_shape = (z_dim, h, w, 3)
        normals_axes = [
            {"name": "z", "type": "space", "unit": "index"},
            {"name": "y", "type": "space", "unit": "pixel"},
            {"name": "x", "type": "space", "unit": "pixel"},
            {"name": "c", "type": "channel"},
        ]
        normals_translation = [0.0, 0.0, 0.0, 0.0]
        normals_datasets_meta = []

        for level in range(num_levels):
            level_shape = compute_level_shape(normals_base_shape, level, (0, 1, 2))
            level_chunks = compute_chunks(level_shape, args.chunk_size, level)
            dataset_name = f"{level}"
            ds = normals_root.create_array(
                dataset_name,
                shape=level_shape,
                chunks=level_chunks,
                dtype=normals_dtype,
                overwrite=True,
                **codec,
            )
            normals_datasets.append(ds)
            scale_vector = [float(2 ** level), float(2 ** level), float(2 ** level), 1.0]
            normals_datasets_meta.append(
                {
                    "path": dataset_name,
                    "coordinateTransformations": [
                        {"type": "scale", "scale": scale_vector},
                        {"type": "translation", "translation": list(normals_translation)},
                    ],
                }
            )

        normals_root.attrs["multiscales"] = [
            {
                "version": "0.4",
                "name": "normals",
                "axes": normals_axes,
                "datasets": normals_datasets_meta,
            }
        ]

    normals_dataset_name = normals_datasets[0].path if normals_datasets else None

    surface_frame_datasets = []
    surface_frame_dataset_name = None

    if args.output_surface_frame:
        surface_frame_root = create_group(surface_frame_out_path, overwrite=True, auth=None)
        surface_frame_base_shape = (z_dim, h, w, 3, 3)
        surface_frame_axes = [
            {"name": "z", "type": "space", "unit": "index"},
            {"name": "y", "type": "space", "unit": "pixel"},
            {"name": "x", "type": "space", "unit": "pixel"},
            {"name": "f", "type": "parametric", "description": ["t_u", "t_v", "n"]},
            {"name": "c", "type": "space", "unit": "direction"},
        ]
        surface_frame_translation = [0.0, 0.0, 0.0, 0.0, 0.0]
        surface_frame_datasets_meta = []

        for level in range(num_levels):
            level_shape = compute_level_shape(surface_frame_base_shape, level, (0, 1, 2))
            level_chunks = compute_chunks(level_shape, args.chunk_size, level)
            dataset_name = f"{level}"
            ds = surface_frame_root.create_array(
                dataset_name,
                shape=level_shape,
                chunks=level_chunks,
                dtype=surface_frame_dtype,
                overwrite=True,
                **codec,
            )
            surface_frame_datasets.append(ds)
            scale_vector = [float(2 ** level), float(2 ** level), float(2 ** level), 1.0, 1.0]
            surface_frame_datasets_meta.append(
                {
                    "path": dataset_name,
                    "coordinateTransformations": [
                        {"type": "scale", "scale": scale_vector},
                        {
                            "type": "translation",
                            "translation": list(surface_frame_translation),
                        },
                    ],
                }
            )

        surface_frame_root.attrs["multiscales"] = [
            {
                "version": "0.4",
                "name": "surface_frame",
                "axes": surface_frame_axes,
                "datasets": surface_frame_datasets_meta,
            }
        ]

    if surface_frame_datasets:
        surface_frame_dataset_name = surface_frame_datasets[0].path

    slice_state = {
        "surfaces": surfaces,
        "w": w,
        "h": h,
        "include_normals": bool(args.output_normals or args.output_surface_frame),
        "write_normals": bool(args.output_normals),
        "include_surface_frame": bool(args.output_surface_frame),
        "volume_depth": z_dim,
        "label_store_path": out_path,
        "label_dataset_name": label_dataset_name,
        "normals_store_path": normals_out_path,
        "normals_dataset_name": normals_dataset_name,
        "normals_dtype": normals_dtype,
        "surface_frame_store_path": surface_frame_out_path,
        "surface_frame_dataset_name": surface_frame_dataset_name,
        "surface_frame_dtype": surface_frame_dtype,
        "format": args.format,
        "label_dtype": label_dtype,
        "tif_slice_digits": max(5, len(str(z_dim - 1))) if z_dim > 0 else 1,
        "tif_compression": "deflate",
    }

    def _write_slice_output(zslice, label_img, normals_img, frame_img):
        slice_index = int(zslice)
        if slice_index < 0 or slice_index >= z_dim:
            raise ValueError(
                f"Requested slice index {slice_index} is outside the allocated z range "
                f"[0, {z_dim - 1}]."
            )

        if args.format == "zarr":
            label_data = np.ascontiguousarray(label_img, dtype=label_datasets[0].dtype)
            _write_sparse_slice(label_datasets[0], slice_index, label_data)

            if args.output_normals and normals_img is not None and normals_img.size:
                normals_data = np.ascontiguousarray(normals_img.astype(normals_dtype, copy=False))
                _write_sparse_slice(normals_datasets[0], slice_index, normals_data)

            if args.output_surface_frame and frame_img is not None and frame_img.size:
                frame_data = np.ascontiguousarray(frame_img.astype(surface_frame_dtype, copy=False))
                _write_sparse_slice(surface_frame_datasets[0], slice_index, frame_data)
        else:
            label_data = np.ascontiguousarray(label_img, dtype=label_dtype)
            _write_tif_slice(
                out_path,
                slice_index,
                label_data,
                slice_state["tif_slice_digits"],
                slice_state["tif_compression"],
            )

    _set_slice_state(slice_state)
    try:
        # Open once before threads start; level-zero z chunks are disjoint.
        _get_worker_datasets()
        if N_PROCESSES > 1:
            with ThreadPoolExecutor(max_workers=N_PROCESSES) as executor:
                for _ in tqdm(executor.map(_process_zslice, z_slices),
                              total=len(z_slices), desc="Slices processed"):
                    pass
        else:
            for z in tqdm(z_slices, desc="Slices processed"):
                _process_zslice(z)
    finally:
        _set_slice_state({})

    processed_slice_indices = set(int(z) for z in z_slices.tolist())
    if len(processed_slice_indices) < z_dim:
        missing_slices = sorted(set(range(z_dim)) - processed_slice_indices)
        if missing_slices:
            print(f"Writing {len(missing_slices)} empty slices outside the surface bounds.")
            zero_label = np.zeros((h, w), dtype=label_dtype)
            zero_normals = np.zeros((0, 0, 0), dtype=np.float32)
            zero_frame = np.zeros((0, 0, 0, 0), dtype=np.float32)
            for z in tqdm(missing_slices, desc="Empty slices", leave=False):
                _write_slice_output(z, zero_label, zero_normals, zero_frame)

    if args.format == "zarr" and args.chunk_size > 0:
        label_datasets[0] = _rechunk(
            label_datasets[0], args.chunk_size, "Labels level 0", N_PROCESSES
        )
        if args.output_normals:
            normals_datasets[0] = _rechunk(
                normals_datasets[0], args.chunk_size, "Normals level 0", N_PROCESSES
            )
        if args.output_surface_frame:
            surface_frame_datasets[0] = _rechunk(
                surface_frame_datasets[0], args.chunk_size, "Surface frame level 0", N_PROCESSES
            )

    def populate_downsampled_levels(datasets, store_path, axes, desc):
        if not datasets:
            return
        for level in range(1, len(datasets)):
            parent = datasets[level - 1]
            target = datasets[level]
            level_desc = f"{desc} level {level}"

            use_parallel = False
            mp_context = None
            if N_PROCESSES > 1:
                try:
                    mp_context = multiprocessing.get_context("fork")
                    use_parallel = True
                except ValueError:
                    mp_context = None

            try:
                chunk_tasks = list(_iter_chunk_starts(target.shape, target.chunks))
            except ValueError as exc:
                raise RuntimeError(
                    f"Unable to prepare chunk iteration for {level_desc}: {exc}"
                ) from exc

            if not chunk_tasks:
                continue

            logger.debug(
                "Prepared %d chunk tasks for %s (parent chunks=%s, target chunks=%s)",
                len(chunk_tasks),
                level_desc,
                parent.chunks,
                target.chunks,
            )

            down_axes = tuple(axes)
            for axis in down_axes:
                if axis >= target.ndim:
                    raise ValueError(
                        f"Invalid downsample axis {axis} for dataset with {target.ndim} dimensions."
                    )

            if use_parallel and mp_context is not None:
                state = {
                    "store_path": store_path,
                    "parent_dataset_path": parent.path,
                    "target_dataset_path": target.path,
                    "downsample_axes": down_axes,
                    "target_shape": target.shape,
                    "target_chunks": target.chunks,
                    "parent_shape": parent.shape,
                }
                _set_downsample_state(state)
                try:
                    with ProcessPoolExecutor(
                        max_workers=N_PROCESSES,
                        mp_context=mp_context,
                    ) as executor:
                        for _ in tqdm(
                            executor.map(_downsample_chunk_worker, chunk_tasks, chunksize=1),
                            total=len(chunk_tasks),
                            desc=level_desc,
                            leave=False,
                        ):
                            pass
                finally:
                    _set_downsample_state({})
            else:
                state = {
                    "store_path": store_path,
                    "parent_dataset_path": parent.path,
                    "target_dataset_path": target.path,
                    "downsample_axes": down_axes,
                    "target_shape": target.shape,
                    "target_chunks": target.chunks,
                    "parent_shape": parent.shape,
                }
                _set_downsample_state(state)
                try:
                    for chunk_start in tqdm(chunk_tasks, desc=level_desc, leave=False):
                        _downsample_chunk_worker(chunk_start)
                finally:
                    _set_downsample_state({})

    if args.format == "zarr":
        populate_downsampled_levels(label_datasets, out_path, axes=(0, 1, 2), desc="Labels")

        if args.output_normals:
            populate_downsampled_levels(normals_datasets, normals_out_path, axes=(0, 1, 2), desc="Normals")

        if args.output_surface_frame:
            populate_downsampled_levels(
                surface_frame_datasets,
                surface_frame_out_path,
                axes=(0, 1, 2),
                desc="Surface frame",
            )

        print("Completed OME-Zarr export.")
    else:
        print("Completed TIFF slice export.")


if __name__ == "__main__":
    main()
