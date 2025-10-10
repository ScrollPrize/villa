#!/usr/bin/env python
import math
import numpy as np
import open3d as o3d
import os
import zarr
import logging
from glob import glob
from itertools import repeat, product
from numcodecs import Blosc
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from numba import jit, set_num_threads
import sys
import argparse
from tqdm import tqdm  # Progress bar
import dask.array as da
from dask.diagnostics import ProgressBar

from vesuvius.image_proc.shared.mesh.affine import (
    axis_perm,
    load_transform_from_json,
    compute_inv_transpose,
    apply_affine_to_points,
    transform_normals,
)

# Determine default number of workers: half of CPU count (at least 1)
default_workers = max(1, multiprocessing.cpu_count() // 2)

# We no longer need normals for expansion, but we still need to find
# intersections of triangles with the z-plane.
MAX_INTERSECTIONS = 3  # Maximum number of intersections per triangle
PYRAMID_LEVELS = 6

# Global store used to share immutable mesh data with forked worker processes
# without serializing the large arrays for every slice task.
_SLICE_STATE = {}
_SLICE_DATASETS = {}
_DOWNSAMPLE_STATE = {}
_DOWNSAMPLE_DATASETS = {}

logger = logging.getLogger(__name__)


def _apply_affine_to_mesh(mesh, matrix4, perm, inv_transpose):
    vertices = np.asarray(mesh.vertices)
    if vertices.size:
        transformed_vertices = apply_affine_to_points(vertices, matrix4, perm)
        mesh.vertices = o3d.utility.Vector3dVector(transformed_vertices)

    linear = matrix4[:3, :3]

    if mesh.has_vertex_normals() and len(mesh.vertex_normals) == len(mesh.vertices):
        v_normals = np.asarray(mesh.vertex_normals)
        transformed_normals = transform_normals(
            v_normals, linear, perm=perm, inv_transpose=inv_transpose
        )
        mesh.vertex_normals = o3d.utility.Vector3dVector(transformed_normals)

    if mesh.has_triangle_normals() and len(mesh.triangle_normals) == len(mesh.triangles):
        t_normals = np.asarray(mesh.triangle_normals)
        transformed_t_normals = transform_normals(
            t_normals, linear, perm=perm, inv_transpose=inv_transpose
        )
        mesh.triangle_normals = o3d.utility.Vector3dVector(transformed_t_normals)


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
        label_root = zarr.open(state["label_store_path"], mode="r+")
        labels_ds = label_root[state["label_dataset_name"]]
        _SLICE_DATASETS["labels"] = labels_ds

    normals_ds = None
    if state.get("write_normals", False):
        normals_ds = _SLICE_DATASETS.get("normals")
        if normals_ds is None:
            normals_root = zarr.open(state["normals_store_path"], mode="r+")
            normals_ds = normals_root[state["normals_dataset_name"]]
            _SLICE_DATASETS["normals"] = normals_ds

    frame_ds = None
    if state.get("include_surface_frame", False):
        frame_ds = _SLICE_DATASETS.get("surface_frame")
        if frame_ds is None:
            frame_root = zarr.open(state["surface_frame_store_path"], mode="r+")
            frame_ds = frame_root[state["surface_frame_dataset_name"]]
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
    import tifffile

    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"{slice_index:0{digits}d}.tif")
    tifffile.imwrite(filename, slice_data, compression=compression)


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
            state.get("tif_compression", "zlib"),
        )
    else:
        raise ValueError(f"Unsupported output format '{fmt}'.")


def _process_zslice(zslice):
    state = _SLICE_STATE
    if not state:
        raise RuntimeError("Slice processing state is not initialized in the worker.")

    _, label_img, normals_img, frame_img = process_slice(
        (
            zslice,
            state["vertices"],
            state["triangles"],
            state["labels"],
            state["w"],
            state["h"],
            state["include_normals"],
            state["vertex_normals"],
            state["triangle_normals"],
            state["use_vertex_normals"],
            state.get("include_surface_frame", False),
            state.get("vertex_tangents"),
            state.get("vertex_bitangents"),
            state.get("triangle_tangents"),
            state.get("triangle_bitangents"),
            state.get("use_vertex_tangents"),
        )
    )

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
        root = zarr.open_group(state["store_path"], mode="r+")
        parent_ds = root[state["parent_dataset_path"]]
        target_ds = root[state["target_dataset_path"]]
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


def _rechunk_with_dask(zarr_array, chunk_size, desc, num_workers):
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
    print(f"Rechunking {desc} to chunks {target_chunks} using Dask", flush=True)

    dask_array = da.from_zarr(zarr_array, chunks=target_chunks)
    tmp_component = _build_tmp_component_name(zarr_array.path)
    tmp_component_full = _join_parent_path(zarr_array.path, tmp_component)

    # Preserve existing attributes before recreating the dataset.
    attrs = zarr_array.attrs.asdict()

    root = zarr.open_group(store=zarr_array.store, mode="r+")
    parent_group = _get_parent_group(root, zarr_array.path)

    if tmp_component in parent_group:
        del parent_group[tmp_component]

    delayed = da.to_zarr(
        dask_array,
        zarr_array.store,
        component=tmp_component_full,
        overwrite=True,
        compute=False,
    )

    with ProgressBar():
        delayed.compute(scheduler="threads", num_workers=max(1, num_workers))

    new_array = parent_group[tmp_component]
    if attrs:
        new_array.attrs.update(attrs)

    dataset_name = zarr_array.path.rpartition("/")[2] or zarr_array.path
    if dataset_name in parent_group:
        del parent_group[dataset_name]
    parent_group.move(tmp_component, dataset_name)

    refreshed = parent_group[dataset_name]
    return refreshed


def _build_tmp_component_name(path):
    basename = path.rpartition("/")[2] or path
    return f"{basename}__rechunk_tmp"


def _join_parent_path(path, child):
    parent, sep, _ = path.rpartition("/")
    if sep:
        return f"{parent}/{child}"
    return child


def _get_parent_group(root, path):
    parent_path, sep, _ = path.rpartition("/")
    if sep:
        return root[parent_path]
    return root


@jit(nopython=True)
def get_intersection_point_2d(start, end, z_plane):
    """
    Given two 3D vertices start/end, returns the 2D intersection (x,y)
    on the plane z = z_plane, if it exists. Otherwise returns None.
    """
    z_s = start[2]
    z_e = end[2]

    # Check if one of the vertices is exactly on the plane
    if abs(z_s - z_plane) < 1e-8:
        return start[:2]
    if abs(z_e - z_plane) < 1e-8:
        return end[:2]

    # If neither vertex is on the plane, check if we can intersect
    denom = (z_e - z_s)
    if abs(denom) < 1e-15:
        return None  # Parallel or effectively so

    t = (z_plane - z_s) / denom
    # Only treat intersection if t is in [0,1], with slight relax
    if not (0.0 - 1e-3 <= t <= 1.0 + 1e-3):
        return None

    # Compute intersection in xy
    x = start[0] + t * (end[0] - start[0])
    y = start[1] + t * (end[1] - start[1])
    return np.array([x, y], dtype=np.float32)


@jit(nopython=True)
def rasterize_line_label(x0, y0, x1, y1, w, h, label_img, mesh_label):
    """
    Simple line rasterization in label_img with the integer mesh label.
    Uses a basic DDA approach.
    """
    dx = x1 - x0
    dy = y1 - y0

    steps = int(max(abs(dx), abs(dy)))  # Use the larger magnitude as steps
    if steps == 0:
        # Single point (start == end)
        ix = int(round(x0))
        iy = int(round(y0))
        if 0 <= ix < w and 0 <= iy < h:
            label_img[iy, ix] = mesh_label
        return

    x_inc = dx / steps
    y_inc = dy / steps
    x_f = x0
    y_f = y0

    for i in range(steps + 1):
        ix = int(round(x_f))
        iy = int(round(y_f))
        if 0 <= ix < w and 0 <= iy < h:
            label_img[iy, ix] = mesh_label
        x_f += x_inc
        y_f += y_inc


@jit(nopython=True)
def process_slice_points_label(vertices, triangles, mesh_labels, zslice, w, h):
    """
    For the plane z=zslice, find the intersection lines of each triangle
    and draw them into a 2D array (label_img) using the triangle's mesh label.
    """
    label_img = np.zeros((h, w), dtype=np.uint16)

    for i in range(len(triangles)):
        tri = triangles[i]
        label = mesh_labels[i]
        v0 = vertices[tri[0]]
        v1 = vertices[tri[1]]
        v2 = vertices[tri[2]]

        # Quick check if the z-range of the triangle might intersect zslice
        z_min = min(v0[2], v1[2], v2[2])
        z_max = max(v0[2], v1[2], v2[2])
        if not (z_min <= zslice <= z_max):
            continue

        # Find up to three intersection points
        pts_2d = []
        # Each edge
        for (a, b) in [(v0, v1), (v1, v2), (v2, v0)]:
            p = get_intersection_point_2d(a, b, zslice)
            if p is not None:
                # Check for duplicates in pts_2d
                is_dup = False
                for pp in pts_2d:
                    dist2 = (p[0] - pp[0]) ** 2 + (p[1] - pp[1]) ** 2
                    if dist2 < 1e-12:
                        is_dup = True
                        break
                if not is_dup:
                    pts_2d.append(p)

        # If we have at least two unique intersection points, draw lines
        n_inter = len(pts_2d)
        if n_inter >= 2:
            # Typically you expect 2 intersection points, but we’ll connect all pairs
            for ii in range(n_inter):
                for jj in range(ii + 1, n_inter):
                    x0, y0 = pts_2d[ii]
                    x1, y1 = pts_2d[jj]
                    rasterize_line_label(x0, y0, x1, y1, w, h, label_img, label)

    return label_img


@jit(nopython=True)
def rasterize_line_surface(x0, y0, x1, y1, bary0, bary1, w, h,
                           label_img, mesh_label,
                           include_normals, include_surface_frame,
                           normal_sums, normal_counts,
                           tri_vertex_normals,
                           tangent_sums, bitangent_sums,
                           tri_vertex_tangents, tri_vertex_bitangents):
    dx = x1 - x0
    dy = y1 - y0

    steps = int(max(abs(dx), abs(dy)))
    if steps == 0:
        ix = int(round(x0))
        iy = int(round(y0))
        if 0 <= ix < w and 0 <= iy < h:
            label_img[iy, ix] = mesh_label
            if include_normals or include_surface_frame:
                w0 = bary0[0]
                w1 = bary0[1]
                w2 = bary0[2]
                nx = (w0 * tri_vertex_normals[0, 0] +
                      w1 * tri_vertex_normals[1, 0] +
                      w2 * tri_vertex_normals[2, 0])
                ny = (w0 * tri_vertex_normals[0, 1] +
                      w1 * tri_vertex_normals[1, 1] +
                      w2 * tri_vertex_normals[2, 1])
                nz = (w0 * tri_vertex_normals[0, 2] +
                      w1 * tri_vertex_normals[1, 2] +
                      w2 * tri_vertex_normals[2, 2])
                normal_sums[iy, ix, 0] += nx
                normal_sums[iy, ix, 1] += ny
                normal_sums[iy, ix, 2] += nz
                if include_surface_frame:
                    tx = (w0 * tri_vertex_tangents[0, 0] +
                          w1 * tri_vertex_tangents[1, 0] +
                          w2 * tri_vertex_tangents[2, 0])
                    ty = (w0 * tri_vertex_tangents[0, 1] +
                          w1 * tri_vertex_tangents[1, 1] +
                          w2 * tri_vertex_tangents[2, 1])
                    tz = (w0 * tri_vertex_tangents[0, 2] +
                          w1 * tri_vertex_tangents[1, 2] +
                          w2 * tri_vertex_tangents[2, 2])
                    bx = (w0 * tri_vertex_bitangents[0, 0] +
                          w1 * tri_vertex_bitangents[1, 0] +
                          w2 * tri_vertex_bitangents[2, 0])
                    by = (w0 * tri_vertex_bitangents[0, 1] +
                          w1 * tri_vertex_bitangents[1, 1] +
                          w2 * tri_vertex_bitangents[2, 1])
                    bz = (w0 * tri_vertex_bitangents[0, 2] +
                          w1 * tri_vertex_bitangents[1, 2] +
                          w2 * tri_vertex_bitangents[2, 2])
                    tangent_sums[iy, ix, 0] += tx
                    tangent_sums[iy, ix, 1] += ty
                    tangent_sums[iy, ix, 2] += tz
                    bitangent_sums[iy, ix, 0] += bx
                    bitangent_sums[iy, ix, 1] += by
                    bitangent_sums[iy, ix, 2] += bz
                normal_counts[iy, ix] += 1
        return

    x_inc = dx / steps
    y_inc = dy / steps
    x_f = x0
    y_f = y0

    for i in range(steps + 1):
        ix = int(round(x_f))
        iy = int(round(y_f))
        if 0 <= ix < w and 0 <= iy < h:
            label_img[iy, ix] = mesh_label
            if include_normals or include_surface_frame:
                alpha = i / steps
                w0 = bary0[0] * (1.0 - alpha) + bary1[0] * alpha
                w1 = bary0[1] * (1.0 - alpha) + bary1[1] * alpha
                w2 = bary0[2] * (1.0 - alpha) + bary1[2] * alpha
                nx = (w0 * tri_vertex_normals[0, 0] +
                      w1 * tri_vertex_normals[1, 0] +
                      w2 * tri_vertex_normals[2, 0])
                ny = (w0 * tri_vertex_normals[0, 1] +
                      w1 * tri_vertex_normals[1, 1] +
                      w2 * tri_vertex_normals[2, 1])
                nz = (w0 * tri_vertex_normals[0, 2] +
                      w1 * tri_vertex_normals[1, 2] +
                      w2 * tri_vertex_normals[2, 2])
                normal_sums[iy, ix, 0] += nx
                normal_sums[iy, ix, 1] += ny
                normal_sums[iy, ix, 2] += nz
                if include_surface_frame:
                    tx = (w0 * tri_vertex_tangents[0, 0] +
                          w1 * tri_vertex_tangents[1, 0] +
                          w2 * tri_vertex_tangents[2, 0])
                    ty = (w0 * tri_vertex_tangents[0, 1] +
                          w1 * tri_vertex_tangents[1, 1] +
                          w2 * tri_vertex_tangents[2, 1])
                    tz = (w0 * tri_vertex_tangents[0, 2] +
                          w1 * tri_vertex_tangents[1, 2] +
                          w2 * tri_vertex_tangents[2, 2])
                    bx = (w0 * tri_vertex_bitangents[0, 0] +
                          w1 * tri_vertex_bitangents[1, 0] +
                          w2 * tri_vertex_bitangents[2, 0])
                    by = (w0 * tri_vertex_bitangents[0, 1] +
                          w1 * tri_vertex_bitangents[1, 1] +
                          w2 * tri_vertex_bitangents[2, 1])
                    bz = (w0 * tri_vertex_bitangents[0, 2] +
                          w1 * tri_vertex_bitangents[1, 2] +
                          w2 * tri_vertex_bitangents[2, 2])
                    tangent_sums[iy, ix, 0] += tx
                    tangent_sums[iy, ix, 1] += ty
                    tangent_sums[iy, ix, 2] += tz
                    bitangent_sums[iy, ix, 0] += bx
                    bitangent_sums[iy, ix, 1] += by
                    bitangent_sums[iy, ix, 2] += bz
                normal_counts[iy, ix] += 1
        x_f += x_inc
        y_f += y_inc


@jit(nopython=True)
def process_slice_points_label_surface(vertices, triangles, mesh_labels,
                                       vertex_normals, triangle_normals,
                                       triangle_use_vertex_normals,
                                       vertex_tangents, vertex_bitangents,
                                       triangle_tangents, triangle_bitangents,
                                       triangle_use_vertex_tangents,
                                       include_normals, include_surface_frame,
                                       zslice, w, h):
    label_img = np.zeros((h, w), dtype=np.uint16)
    normal_sums = np.zeros((h, w, 3), dtype=np.float32)
    normal_counts = np.zeros((h, w), dtype=np.uint16)

    if include_surface_frame:
        tangent_sums = np.zeros((h, w, 3), dtype=np.float32)
        bitangent_sums = np.zeros((h, w, 3), dtype=np.float32)
    else:
        tangent_sums = np.zeros((1, 1, 3), dtype=np.float32)
        bitangent_sums = np.zeros((1, 1, 3), dtype=np.float32)

    for i in range(len(triangles)):
        tri = triangles[i]
        label = mesh_labels[i]

        idx0 = tri[0]
        idx1 = tri[1]
        idx2 = tri[2]
        v0 = vertices[idx0]
        v1 = vertices[idx1]
        v2 = vertices[idx2]

        z_min = min(v0[2], v1[2], v2[2])
        z_max = max(v0[2], v1[2], v2[2])
        if not (z_min <= zslice <= z_max):
            continue

        pts_2d = np.zeros((MAX_INTERSECTIONS, 2), dtype=np.float32)
        bary_pts = np.zeros((MAX_INTERSECTIONS, 3), dtype=np.float32)
        n_inter = 0

        tri_vertices = (idx0, idx1, idx2)

        for edge_idx in range(3):
            a_idx = tri_vertices[edge_idx]
            b_idx = tri_vertices[(edge_idx + 1) % 3]
            a = vertices[a_idx]
            b = vertices[b_idx]

            z_a = a[2]
            z_b = b[2]

            if abs(z_a - zslice) < 1e-8:
                p_x = a[0]
                p_y = a[1]
                bary = np.zeros(3, dtype=np.float32)
                bary[edge_idx] = 1.0
            elif abs(z_b - zslice) < 1e-8:
                p_x = b[0]
                p_y = b[1]
                bary = np.zeros(3, dtype=np.float32)
                bary[(edge_idx + 1) % 3] = 1.0
            else:
                denom = z_b - z_a
                if abs(denom) < 1e-15:
                    continue
                t = (zslice - z_a) / denom
                if not (-1e-3 <= t <= 1.0 + 1e-3):
                    continue
                p_x = a[0] + t * (b[0] - a[0])
                p_y = a[1] + t * (b[1] - a[1])
                bary = np.zeros(3, dtype=np.float32)
                idx_a_local = -1
                idx_b_local = -1
                for k in range(3):
                    if tri_vertices[k] == a_idx:
                        idx_a_local = k
                    if tri_vertices[k] == b_idx:
                        idx_b_local = k
                if idx_a_local == -1 or idx_b_local == -1:
                    continue
                bary[idx_a_local] = 1.0 - t
                bary[idx_b_local] = t

            is_dup = False
            for prev in range(n_inter):
                dx = p_x - pts_2d[prev, 0]
                dy = p_y - pts_2d[prev, 1]
                if dx * dx + dy * dy < 1e-12:
                    is_dup = True
                    break

            if not is_dup and n_inter < MAX_INTERSECTIONS:
                pts_2d[n_inter, 0] = p_x
                pts_2d[n_inter, 1] = p_y
                bary_pts[n_inter, 0] = bary[0]
                bary_pts[n_inter, 1] = bary[1]
                bary_pts[n_inter, 2] = bary[2]
                n_inter += 1

        if n_inter >= 2:
            tri_vertex_normals = np.zeros((3, 3), dtype=np.float32)
            if triangle_use_vertex_normals[i]:
                tri_vertex_normals[0, 0] = vertex_normals[idx0, 0]
                tri_vertex_normals[0, 1] = vertex_normals[idx0, 1]
                tri_vertex_normals[0, 2] = vertex_normals[idx0, 2]
                tri_vertex_normals[1, 0] = vertex_normals[idx1, 0]
                tri_vertex_normals[1, 1] = vertex_normals[idx1, 1]
                tri_vertex_normals[1, 2] = vertex_normals[idx1, 2]
                tri_vertex_normals[2, 0] = vertex_normals[idx2, 0]
                tri_vertex_normals[2, 1] = vertex_normals[idx2, 1]
                tri_vertex_normals[2, 2] = vertex_normals[idx2, 2]
            else:
                tri_norm = triangle_normals[i]
                tri_vertex_normals[0, 0] = tri_norm[0]
                tri_vertex_normals[0, 1] = tri_norm[1]
                tri_vertex_normals[0, 2] = tri_norm[2]
                tri_vertex_normals[1, 0] = tri_norm[0]
                tri_vertex_normals[1, 1] = tri_norm[1]
                tri_vertex_normals[1, 2] = tri_norm[2]
                tri_vertex_normals[2, 0] = tri_norm[0]
                tri_vertex_normals[2, 1] = tri_norm[1]
                tri_vertex_normals[2, 2] = tri_norm[2]

            tri_vertex_tangents = np.zeros((3, 3), dtype=np.float32)
            tri_vertex_bitangents = np.zeros((3, 3), dtype=np.float32)
            if include_surface_frame:
                if triangle_use_vertex_tangents[i]:
                    tri_vertex_tangents[0, 0] = vertex_tangents[idx0, 0]
                    tri_vertex_tangents[0, 1] = vertex_tangents[idx0, 1]
                    tri_vertex_tangents[0, 2] = vertex_tangents[idx0, 2]
                    tri_vertex_tangents[1, 0] = vertex_tangents[idx1, 0]
                    tri_vertex_tangents[1, 1] = vertex_tangents[idx1, 1]
                    tri_vertex_tangents[1, 2] = vertex_tangents[idx1, 2]
                    tri_vertex_tangents[2, 0] = vertex_tangents[idx2, 0]
                    tri_vertex_tangents[2, 1] = vertex_tangents[idx2, 1]
                    tri_vertex_tangents[2, 2] = vertex_tangents[idx2, 2]

                    tri_vertex_bitangents[0, 0] = vertex_bitangents[idx0, 0]
                    tri_vertex_bitangents[0, 1] = vertex_bitangents[idx0, 1]
                    tri_vertex_bitangents[0, 2] = vertex_bitangents[idx0, 2]
                    tri_vertex_bitangents[1, 0] = vertex_bitangents[idx1, 0]
                    tri_vertex_bitangents[1, 1] = vertex_bitangents[idx1, 1]
                    tri_vertex_bitangents[1, 2] = vertex_bitangents[idx1, 2]
                    tri_vertex_bitangents[2, 0] = vertex_bitangents[idx2, 0]
                    tri_vertex_bitangents[2, 1] = vertex_bitangents[idx2, 1]
                    tri_vertex_bitangents[2, 2] = vertex_bitangents[idx2, 2]
                else:
                    tri_tan = triangle_tangents[i]
                    tri_bit = triangle_bitangents[i]
                    for k in range(3):
                        tri_vertex_tangents[k, 0] = tri_tan[0]
                        tri_vertex_tangents[k, 1] = tri_tan[1]
                        tri_vertex_tangents[k, 2] = tri_tan[2]
                        tri_vertex_bitangents[k, 0] = tri_bit[0]
                        tri_vertex_bitangents[k, 1] = tri_bit[1]
                        tri_vertex_bitangents[k, 2] = tri_bit[2]

            for ii in range(n_inter):
                for jj in range(ii + 1, n_inter):
                    x0 = pts_2d[ii, 0]
                    y0 = pts_2d[ii, 1]
                    x1 = pts_2d[jj, 0]
                    y1 = pts_2d[jj, 1]
                    bary0 = bary_pts[ii]
                    bary1 = bary_pts[jj]
                    rasterize_line_surface(
                        x0,
                        y0,
                        x1,
                        y1,
                        bary0,
                        bary1,
                        w,
                        h,
                        label_img,
                        label,
                        include_normals,
                        include_surface_frame,
                        normal_sums,
                        normal_counts,
                        tri_vertex_normals,
                        tangent_sums,
                        bitangent_sums,
                        tri_vertex_tangents,
                        tri_vertex_bitangents,
                    )

    normals_img = np.zeros((h, w, 3), dtype=np.float32)
    frame_img = np.zeros((h, w, 3, 3), dtype=np.float32)

    for y in range(h):
        for x in range(w):
            count = normal_counts[y, x]
            if count > 0:
                nx = normal_sums[y, x, 0] / count
                ny = normal_sums[y, x, 1] / count
                nz = normal_sums[y, x, 2] / count
                length = np.sqrt(nx * nx + ny * ny + nz * nz)
                if length <= 1e-12:
                    raise ValueError("Normal magnitude collapsed during slice rasterization.")
                nx /= length
                ny /= length
                nz /= length
                normals_img[y, x, 0] = nx
                normals_img[y, x, 1] = ny
                normals_img[y, x, 2] = nz

                if include_surface_frame:
                    tx = tangent_sums[y, x, 0] / count
                    ty = tangent_sums[y, x, 1] / count
                    tz = tangent_sums[y, x, 2] / count
                    t_dot_n = nx * tx + ny * ty + nz * tz
                    tx -= nx * t_dot_n
                    ty -= ny * t_dot_n
                    tz -= nz * t_dot_n
                    t_len = np.sqrt(tx * tx + ty * ty + tz * tz)
                    if t_len <= 1e-12:
                        raise ValueError("Tangent magnitude collapsed during slice rasterization.")
                    tx /= t_len
                    ty /= t_len
                    tz /= t_len

                    bx = bitangent_sums[y, x, 0] / count
                    by = bitangent_sums[y, x, 1] / count
                    bz = bitangent_sums[y, x, 2] / count
                    bx_t = ny * tz - nz * ty
                    by_t = nz * tx - nx * tz
                    bz_t = nx * ty - ny * tx
                    dot_sign = bx * bx_t + by * by_t + bz * bz_t
                    if dot_sign < 0.0:
                        bx_t *= -1.0
                        by_t *= -1.0
                        bz_t *= -1.0
                    b_len = np.sqrt(bx_t * bx_t + by_t * by_t + bz_t * bz_t)
                    if b_len <= 1e-12:
                        raise ValueError("Bitangent magnitude collapsed during slice rasterization.")
                    bx_t /= b_len
                    by_t /= b_len
                    bz_t /= b_len

                    frame_img[y, x, 0, 0] = tx
                    frame_img[y, x, 0, 1] = ty
                    frame_img[y, x, 0, 2] = tz
                    frame_img[y, x, 1, 0] = bx_t
                    frame_img[y, x, 1, 1] = by_t
                    frame_img[y, x, 1, 2] = bz_t
                    frame_img[y, x, 2, 0] = nx
                    frame_img[y, x, 2, 1] = ny
                    frame_img[y, x, 2, 2] = nz
            else:
                normals_img[y, x, 0] = 0.0
                normals_img[y, x, 1] = 0.0
                normals_img[y, x, 2] = 0.0

    return label_img, normals_img, frame_img


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


def process_mesh(
    mesh_path,
    mesh_index,
    include_normals,
    include_surface_frame,
    transform_info,
    label_dtype,
):
    """
    Load a mesh from disk, return (vertices, triangles, labels_for_those_triangles).
    We assign mesh_index+1 as the label.
    """
    print(f"Processing mesh: {mesh_path}")
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))

    if transform_info is not None:
        _apply_affine_to_mesh(
            mesh,
            transform_info["matrix"],
            transform_info["perm"],
            transform_info.get("inv_transpose"),
        )

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    triangles = np.asarray(mesh.triangles, dtype=np.int32)

    # Every triangle in this mesh gets the same label: mesh_index+1
    labels = np.full(len(triangles), mesh_index + 1, dtype=label_dtype)

    need_normals = include_normals or include_surface_frame

    vertex_normals = np.zeros((0, 3), dtype=np.float32)
    triangle_normals = np.zeros((0, 3), dtype=np.float32)
    use_vertex_normals = False

    if need_normals:
        if mesh.has_vertex_normals() and len(mesh.vertex_normals) == len(mesh.vertices):
            vertex_normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
            use_vertex_normals = True
        else:
            mesh.compute_vertex_normals()
            if len(mesh.vertex_normals) == len(mesh.vertices):
                vertex_normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
                use_vertex_normals = True

        if mesh.has_triangle_normals() and len(mesh.triangle_normals) == len(mesh.triangles):
            triangle_normals = np.asarray(mesh.triangle_normals, dtype=np.float32)
        else:
            mesh.compute_triangle_normals()
            if len(mesh.triangle_normals) == len(mesh.triangles):
                triangle_normals = np.asarray(mesh.triangle_normals, dtype=np.float32)

        if triangle_normals.shape[0] == 0:
            mesh.compute_triangle_normals()
            triangle_normals = np.asarray(mesh.triangle_normals, dtype=np.float32)

        if vertex_normals.shape[0] != len(vertices):
            mesh.compute_vertex_normals()
            if len(mesh.vertex_normals) == len(mesh.vertices):
                vertex_normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
                use_vertex_normals = True

        if vertex_normals.shape[0] == 0:
            raise ValueError("Unable to compute vertex normals required for surface processing.")

        if triangle_normals.shape[0] == 0:
            raise ValueError("Unable to compute triangle normals required for surface processing.")

    vertex_tangents = np.zeros((0, 3), dtype=np.float32)
    vertex_bitangents = np.zeros((0, 3), dtype=np.float32)
    triangle_tangents = np.zeros((0, 3), dtype=np.float32)
    triangle_bitangents = np.zeros((0, 3), dtype=np.float32)
    use_vertex_tangents = False

    if include_surface_frame:
        if not mesh.has_triangle_uvs():
            raise ValueError("Mesh does not provide triangle UVs required for surface frame export.")

        triangle_uvs = np.asarray(mesh.triangle_uvs, dtype=np.float32)
        expected_uv_rows = len(triangles) * 3
        if triangle_uvs.shape[0] != expected_uv_rows:
            raise ValueError("Triangle UV array size does not match triangle count.")

        triangle_uvs = triangle_uvs.reshape(len(triangles), 3, 2)

        vertex_tangents = np.zeros((len(vertices), 3), dtype=np.float32)
        vertex_bitangents = np.zeros((len(vertices), 3), dtype=np.float32)
        triangle_tangents = np.zeros((len(triangles), 3), dtype=np.float32)
        triangle_bitangents = np.zeros((len(triangles), 3), dtype=np.float32)
        tangent_counts = np.zeros(len(vertices), dtype=np.int32)

        for tri_idx in range(len(triangles)):
            idx0 = triangles[tri_idx, 0]
            idx1 = triangles[tri_idx, 1]
            idx2 = triangles[tri_idx, 2]

            v0 = vertices[idx0]
            v1 = vertices[idx1]
            v2 = vertices[idx2]

            uv0 = triangle_uvs[tri_idx, 0]
            uv1 = triangle_uvs[tri_idx, 1]
            uv2 = triangle_uvs[tri_idx, 2]

            delta_pos1 = v1 - v0
            delta_pos2 = v2 - v0
            delta_uv1 = uv1 - uv0
            delta_uv2 = uv2 - uv0

            denom = delta_uv1[0] * delta_uv2[1] - delta_uv2[0] * delta_uv1[1]
            if abs(denom) < 1e-12:
                raise ValueError("Encountered degenerate UV mapping while computing tangents.")

            inv_denom = 1.0 / denom
            raw_tangent = (delta_pos1 * delta_uv2[1] - delta_pos2 * delta_uv1[1]) * inv_denom
            raw_bitangent = (delta_pos2 * delta_uv1[0] - delta_pos1 * delta_uv2[0]) * inv_denom

            tri_norm = triangle_normals[tri_idx]
            norm_len = np.linalg.norm(tri_norm)
            if norm_len < 1e-12:
                raise ValueError("Triangle normal magnitude is zero; cannot build tangent frame.")
            tri_norm_unit = tri_norm / norm_len

            tangent_proj = raw_tangent - tri_norm_unit * np.dot(tri_norm_unit, raw_tangent)
            tan_len = np.linalg.norm(tangent_proj)
            if tan_len < 1e-12:
                raise ValueError("Computed tangent length is zero; invalid UV parameterization.")
            tangent_unit = tangent_proj / tan_len

            bitangent_unit = np.cross(tri_norm_unit, tangent_unit)
            if np.linalg.norm(bitangent_unit) < 1e-12:
                raise ValueError("Computed bitangent length is zero; invalid UV parameterization.")

            if np.dot(bitangent_unit, raw_bitangent) < 0.0:
                bitangent_unit *= -1.0

            triangle_tangents[tri_idx] = tangent_unit
            triangle_bitangents[tri_idx] = bitangent_unit

            vertex_tangents[idx0] += tangent_unit
            vertex_tangents[idx1] += tangent_unit
            vertex_tangents[idx2] += tangent_unit

            vertex_bitangents[idx0] += bitangent_unit
            vertex_bitangents[idx1] += bitangent_unit
            vertex_bitangents[idx2] += bitangent_unit

            tangent_counts[idx0] += 1
            tangent_counts[idx1] += 1
            tangent_counts[idx2] += 1

        for vid in range(len(vertices)):
            count = tangent_counts[vid]
            if count <= 0:
                raise ValueError("Vertex missing tangent contributions; UVs must be continuous across the mesh.")

            n_vec = vertex_normals[vid]
            n_len = np.linalg.norm(n_vec)
            if n_len < 1e-12:
                raise ValueError("Vertex normal magnitude is zero; cannot orthonormalize tangents.")
            n_unit = n_vec / n_len

            t_vec = vertex_tangents[vid] / count
            t_vec = t_vec - n_unit * np.dot(n_unit, t_vec)
            t_len = np.linalg.norm(t_vec)
            if t_len < 1e-12:
                raise ValueError("Vertex tangent degenerates after projection; check UV continuity.")
            t_unit = t_vec / t_len

            b_vec = vertex_bitangents[vid] / count
            b_unit = np.cross(n_unit, t_unit)
            if np.linalg.norm(b_unit) < 1e-12:
                raise ValueError("Vertex bitangent degenerates; cannot establish surface frame.")
            if np.dot(b_unit, b_vec) < 0.0:
                b_unit *= -1.0

            vertex_tangents[vid] = t_unit
            vertex_bitangents[vid] = b_unit

        use_vertex_tangents = True

    return (
        vertices,
        triangles,
        labels,
        vertex_normals,
        triangle_normals,
        use_vertex_normals,
        vertex_tangents,
        vertex_bitangents,
        triangle_tangents,
        triangle_bitangents,
        use_vertex_tangents,
    )


def process_slice(args):
    """Process a single z-slice and return label and optional surface data."""
    (
        zslice,
        vertices,
        triangles,
        labels,
        w,
        h,
        include_normals,
        vertex_normals,
        triangle_normals,
        triangle_use_vertex_normals,
        include_surface_frame,
        vertex_tangents,
        vertex_bitangents,
        triangle_tangents,
        triangle_bitangents,
        triangle_use_vertex_tangents,
    ) = args

    if include_normals or include_surface_frame:
        img_label, normals_img, frame_img = process_slice_points_label_surface(
            vertices,
            triangles,
            labels,
            vertex_normals,
            triangle_normals,
            triangle_use_vertex_normals,
            vertex_tangents,
            vertex_bitangents,
            triangle_tangents,
            triangle_bitangents,
            triangle_use_vertex_tangents,
            include_normals,
            include_surface_frame,
            zslice,
            w,
            h,
        )
    else:
        img_label = process_slice_points_label(vertices, triangles, labels, zslice, w, h)
        normals_img = np.zeros((0, 0, 0), dtype=np.float32)
        frame_img = np.zeros((0, 0, 0, 0), dtype=np.float32)

    return zslice, img_label, normals_img, frame_img


def main():
    # Parse command-line arguments.
    parser = argparse.ArgumentParser(
        description="Process OBJ meshes and slice them along z to produce multiscale OME-Zarr volumes."
    )
    parser.add_argument("folder",
                        help="Path to folder containing OBJ meshes (or parent folder with subfolders of OBJ meshes)")
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
                        help="Number of worker processes to use (default: half of CPU count)")
    parser.add_argument("--recursive", action="store_true",
                        help="Force recursive search in subfolders even if OBJ files exist in the parent folder")
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
        try:
            import tifffile
        except ImportError as exc:
            print("ERROR: tifffile is required for TIFF output format. Install it and retry.")
            sys.exit(1)
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
            transform_matrix = load_transform_from_json(transform_path)
        except Exception as exc:
            print(f"ERROR: failed to load transform: {exc}")
            sys.exit(1)

        if args.transform_invert:
            try:
                transform_matrix = np.linalg.inv(transform_matrix)
                print("Applying inverse of provided affine (--transform-invert).")
            except np.linalg.LinAlgError:
                print("ERROR: Provided affine transform is non-invertible.")
                sys.exit(1)

        try:
            perm, _ = axis_perm(args.transform_axis_order)
        except ValueError as exc:
            print(f"ERROR: {exc}")
            sys.exit(1)

        linear_part = transform_matrix[:3, :3]
        inv_transpose = compute_inv_transpose(linear_part)

        transform_info = {
            "matrix": transform_matrix,
            "perm": perm,
            "axis_order": args.transform_axis_order,
            "inv_transpose": inv_transpose,
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

    # Use the provided number of worker processes.
    N_PROCESSES = args.num_workers
    print(f"Using {N_PROCESSES} worker processes")
    set_num_threads(N_PROCESSES)

    # Folder where OBJ meshes are located.
    folder_path = args.folder
    print(f"Using mesh folder: {folder_path}")

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

    # Find OBJ files - either directly or in subfolders
    if args.recursive:
        # Force recursive search
        mesh_paths = glob(os.path.join(folder_path, '**', '*.obj'), recursive=True)
        print(f"Recursive search enabled")
    else:
        # First try direct OBJ files
        mesh_paths = glob(os.path.join(folder_path, '*.obj'))

        if not mesh_paths:
            # No OBJ files found directly, try subfolders
            mesh_paths = glob(os.path.join(folder_path, '*', '*.obj'))
            if mesh_paths:
                print(f"No OBJ files found in {folder_path}, searching in subfolders...")

    if not mesh_paths:
        print(f"ERROR: No OBJ files found in {folder_path} or its subfolders")
        sys.exit(1)

    print(f"Found {len(mesh_paths)} meshes to process")

    # Read all meshes in parallel.
    with ProcessPoolExecutor(max_workers=N_PROCESSES) as executor:
        mesh_results = list(
            executor.map(
                process_mesh,
                mesh_paths,
                range(len(mesh_paths)),
                repeat(args.output_normals),
                repeat(args.output_surface_frame),
                repeat(transform_info),
                repeat(args.label_dtype),
            )
        )

    # Merge all into a single set of (vertices, triangles, labels).
    all_vertices = []
    all_triangles = []
    all_labels = []
    all_vertex_normals = []
    all_triangle_normals = []
    triangle_use_vertex_flags = []
    all_vertex_tangents = []
    all_vertex_bitangents = []
    all_triangle_tangents = []
    all_triangle_bitangents = []
    triangle_use_vertex_tangent_flags = []
    vertex_offset = 0

    for (
        vertices_i,
        triangles_i,
        labels_i,
        vertex_normals_i,
        triangle_normals_i,
        use_vertex_normals_i,
        vertex_tangents_i,
        vertex_bitangents_i,
        triangle_tangents_i,
        triangle_bitangents_i,
        use_vertex_tangents_i,
    ) in mesh_results:
        all_vertices.append(vertices_i)
        all_triangles.append(triangles_i + vertex_offset)
        all_labels.append(labels_i)

        if args.output_normals or args.output_surface_frame:
            all_vertex_normals.append(vertex_normals_i)
            all_triangle_normals.append(triangle_normals_i)
            triangle_use_vertex_flags.append(
                np.full(len(triangles_i), use_vertex_normals_i, dtype=np.bool_)
            )

        if args.output_surface_frame:
            all_vertex_tangents.append(vertex_tangents_i)
            all_vertex_bitangents.append(vertex_bitangents_i)
            all_triangle_tangents.append(triangle_tangents_i)
            all_triangle_bitangents.append(triangle_bitangents_i)
            triangle_use_vertex_tangent_flags.append(
                np.full(len(triangles_i), use_vertex_tangents_i, dtype=np.bool_)
            )

        vertex_offset += len(vertices_i)

    # Create the big arrays.
    vertices = np.vstack(all_vertices)
    triangles = np.vstack(all_triangles)
    mesh_labels = np.concatenate(all_labels)

    if mesh_labels.size:
        label_dtype_info = np.iinfo(label_dtype)
        max_label_value = int(mesh_labels.max())
        if max_label_value > label_dtype_info.max:
            print(
                f"ERROR: label dtype {label_dtype} cannot represent label ID {max_label_value} "
                f"(max {label_dtype_info.max})."
            )
            sys.exit(1)

    if args.output_normals:
        vertex_normals = np.vstack(all_vertex_normals)
        triangle_normals = np.vstack(all_triangle_normals)
        triangle_use_vertex_normals = np.concatenate(triangle_use_vertex_flags)
    elif args.output_surface_frame:
        # Surface frame export requires normals even if the normals volume is skipped.
        vertex_normals = np.vstack(all_vertex_normals)
        triangle_normals = np.vstack(all_triangle_normals)
        triangle_use_vertex_normals = np.concatenate(triangle_use_vertex_flags)
    else:
        vertex_normals = np.zeros((0, 3), dtype=np.float32)
        triangle_normals = np.zeros((0, 3), dtype=np.float32)
        triangle_use_vertex_normals = np.zeros(0, dtype=np.bool_)

    if args.output_surface_frame:
        vertex_tangents = np.vstack(all_vertex_tangents)
        vertex_bitangents = np.vstack(all_vertex_bitangents)
        triangle_tangents = np.vstack(all_triangle_tangents)
        triangle_bitangents = np.vstack(all_triangle_bitangents)
        triangle_use_vertex_tangents = np.concatenate(triangle_use_vertex_tangent_flags)
    else:
        vertex_tangents = np.zeros((0, 3), dtype=np.float32)
        vertex_bitangents = np.zeros((0, 3), dtype=np.float32)
        triangle_tangents = np.zeros((0, 3), dtype=np.float32)
        triangle_bitangents = np.zeros((0, 3), dtype=np.float32)
        triangle_use_vertex_tangents = np.zeros(0, dtype=np.bool_)

    # Clip vertices into the valid volume bounds (z, y, x) before slice processing.
    if vertices.size:
        max_x = max(w - 1e-3, 0.0)
        max_y = max(h - 1e-3, 0.0)
        max_z = max(z_dim - 1e-3, 0.0)

        needs_clip = (
            (vertices[:, 0] < 0.0).any()
            or (vertices[:, 0] > max_x).any()
            or (vertices[:, 1] < 0.0).any()
            or (vertices[:, 1] > max_y).any()
            or (vertices[:, 2] < 0.0).any()
            or (vertices[:, 2] > max_z).any()
        )

        if needs_clip:
            print(
                "Clipping mesh vertices to the valid spatial bounds before voxelization.",
                flush=True,
            )

        np.clip(vertices[:, 0], 0.0, max_x, out=vertices[:, 0])
        np.clip(vertices[:, 1], 0.0, max_y, out=vertices[:, 1])
        np.clip(vertices[:, 2], 0.0, max_z, out=vertices[:, 2])

    # Determine slice range from the vertices.
    z_min = max(0, int(np.floor(vertices[:, 2].min())))
    z_max = min(z_dim - 1, int(np.ceil(vertices[:, 2].max())))

    z_slices = np.arange(z_min, z_max + 1)
    print(f"Processing slices from {z_min} to {z_max} (inclusive).")
    print(f"Total number of slices: {len(z_slices)}")

    num_levels = PYRAMID_LEVELS
    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)

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
        label_store = zarr.DirectoryStore(out_path)
        label_root = zarr.group(store=label_store, overwrite=True)
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
            ds = label_root.create_dataset(
                dataset_name,
                shape=level_shape,
                chunks=level_chunks,
                dtype=label_dtype,
                compressor=compressor,
                overwrite=True,
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
        normals_store = zarr.DirectoryStore(normals_out_path)
        normals_root = zarr.group(store=normals_store, overwrite=True)
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
            ds = normals_root.create_dataset(
                dataset_name,
                shape=level_shape,
                chunks=level_chunks,
                dtype=normals_dtype,
                compressor=compressor,
                overwrite=True,
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
        surface_frame_store = zarr.DirectoryStore(surface_frame_out_path)
        surface_frame_root = zarr.group(store=surface_frame_store, overwrite=True)
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
            ds = surface_frame_root.create_dataset(
                dataset_name,
                shape=level_shape,
                chunks=level_chunks,
                dtype=surface_frame_dtype,
                compressor=compressor,
                overwrite=True,
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
        "vertices": vertices,
        "triangles": triangles,
        "labels": mesh_labels,
        "w": w,
        "h": h,
        "include_normals": bool(args.output_normals or args.output_surface_frame),
        "write_normals": bool(args.output_normals),
        "vertex_normals": vertex_normals,
        "triangle_normals": triangle_normals,
        "use_vertex_normals": triangle_use_vertex_normals,
        "include_surface_frame": bool(args.output_surface_frame),
        "vertex_tangents": vertex_tangents,
        "vertex_bitangents": vertex_bitangents,
        "triangle_tangents": triangle_tangents,
        "triangle_bitangents": triangle_bitangents,
        "use_vertex_tangents": triangle_use_vertex_tangents,
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
        "tif_compression": "zlib",
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

    sequential_slice_args = (
        vertices,
        triangles,
        mesh_labels,
        w,
        h,
        bool(args.output_normals or args.output_surface_frame),
        vertex_normals,
        triangle_normals,
        triangle_use_vertex_normals,
        bool(args.output_surface_frame),
        vertex_tangents,
        vertex_bitangents,
        triangle_tangents,
        triangle_bitangents,
        triangle_use_vertex_tangents,
    )

    print("Entering slice rasterization stage", flush=True)
    if N_PROCESSES > 1:
        try:
            mp_context = multiprocessing.get_context("fork")
        except ValueError:
            mp_context = None

        if mp_context is None:
            print(
                "Fork start method unavailable; running slice rasterization in a single process to avoid high memory usage.",
                flush=True,
            )
        else:
            _set_slice_state(slice_state)
            try:
                print("Starting parallel slice rasterization", flush=True)
                with ProcessPoolExecutor(
                    max_workers=N_PROCESSES,
                    mp_context=mp_context,
                ) as executor:
                    for _ in tqdm(
                        executor.map(_process_zslice, z_slices, chunksize=1),
                        total=len(z_slices),
                        desc="Slices processed",
                    ):
                        pass
                print("Finished parallel slice rasterization", flush=True)
                sequential_slice_args = None
            finally:
                _set_slice_state({})

    if sequential_slice_args is not None:
        print("Running sequential slice rasterization", flush=True)
        (
            seq_vertices,
            seq_triangles,
            seq_labels,
            seq_w,
            seq_h,
            seq_include_normals,
            seq_vertex_normals,
            seq_triangle_normals,
            seq_use_vertex_normals,
            seq_include_surface_frame,
            seq_vertex_tangents,
            seq_vertex_bitangents,
            seq_triangle_tangents,
            seq_triangle_bitangents,
            seq_use_vertex_tangents,
        ) = sequential_slice_args

        for z in tqdm(z_slices, total=len(z_slices), desc="Slices processed"):
            zslice, label_img, normals_img, frame_img = process_slice(
                (
                    z,
                    seq_vertices,
                    seq_triangles,
                    seq_labels,
                    seq_w,
                    seq_h,
                    seq_include_normals,
                    seq_vertex_normals,
                    seq_triangle_normals,
                    seq_use_vertex_normals,
                    seq_include_surface_frame,
                    seq_vertex_tangents,
                    seq_vertex_bitangents,
                    seq_triangle_tangents,
                    seq_triangle_bitangents,
                    seq_use_vertex_tangents,
                )
            )
            _write_slice_output(zslice, label_img, normals_img, frame_img)

    processed_slice_indices = set(int(z) for z in z_slices.tolist())
    if len(processed_slice_indices) < z_dim:
        missing_slices = sorted(set(range(z_dim)) - processed_slice_indices)
        if missing_slices:
            print(f"Writing {len(missing_slices)} empty slices outside the mesh bounds.")
            zero_label = np.zeros((h, w), dtype=label_dtype)
            zero_normals = np.zeros((0, 0, 0), dtype=np.float32)
            zero_frame = np.zeros((0, 0, 0, 0), dtype=np.float32)
            for z in tqdm(missing_slices, desc="Empty slices", leave=False):
                _write_slice_output(z, zero_label, zero_normals, zero_frame)

    if args.format == "zarr" and args.chunk_size > 0:
        label_datasets[0] = _rechunk_with_dask(
            label_datasets[0], args.chunk_size, "Labels level 0", N_PROCESSES
        )
        if args.output_normals:
            normals_datasets[0] = _rechunk_with_dask(
                normals_datasets[0], args.chunk_size, "Normals level 0", N_PROCESSES
            )
        if args.output_surface_frame:
            surface_frame_datasets[0] = _rechunk_with_dask(
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
