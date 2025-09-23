#!/usr/bin/env python
from glob import glob
from itertools import repeat
import math
import numpy as np
import open3d as o3d
import os
import zarr
from numcodecs import Blosc
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from numba import jit, set_num_threads
import sys
import argparse
from tqdm import tqdm  # Progress bar
import dask.array as da
from dask.diagnostics import ProgressBar

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


def _set_slice_state(state):
    global _SLICE_STATE
    global _SLICE_DATASETS
    _SLICE_STATE = state
    _SLICE_DATASETS = {}


def _get_worker_datasets():
    state = _SLICE_STATE
    if not state:
        raise RuntimeError("Slice processing state is not initialized in the worker.")

    labels_ds = _SLICE_DATASETS.get("labels")
    if labels_ds is None:
        label_root = zarr.open(state["label_store_path"], mode="r+")
        labels_ds = label_root[state["label_dataset_name"]]
        _SLICE_DATASETS["labels"] = labels_ds

    normals_ds = None
    if state["include_normals"]:
        normals_ds = _SLICE_DATASETS.get("normals")
        if normals_ds is None:
            normals_root = zarr.open(state["normals_store_path"], mode="r+")
            normals_ds = normals_root[state["normals_dataset_name"]]
            _SLICE_DATASETS["normals"] = normals_ds

    return labels_ds, normals_ds


def _write_slice_output_worker(zslice, label_img, normals_img):
    state = _SLICE_STATE
    labels_ds, normals_ds = _get_worker_datasets()

    slice_index = zslice - state["z_min"]
    labels_ds[slice_index, ...] = np.ascontiguousarray(label_img, dtype=labels_ds.dtype)

    if state["include_normals"] and normals_img is not None and normals_ds is not None:
        normals_ds[slice_index, ...] = np.ascontiguousarray(
            normals_img.astype(state["normals_dtype"], copy=False)
        )


def _process_zslice(zslice):
    state = _SLICE_STATE
    if not state:
        raise RuntimeError("Slice processing state is not initialized in the worker.")

    _, label_img, normals_img = process_slice(
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
        )
    )

    _write_slice_output_worker(zslice, label_img, normals_img)
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


def _downsample_plane_worker(idx):
    state = _DOWNSAMPLE_STATE
    if not state:
        raise RuntimeError("Downsample state is not initialized in the worker.")

    parent_ds, target_ds = _get_downsample_datasets()
    axes = state["axes"]
    source = np.asarray(parent_ds[idx, ...])
    downsampled = downsample_2x(source, axes=axes)
    target_ds[idx, ...] = np.ascontiguousarray(downsampled, dtype=target_ds.dtype)
    return idx


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
def rasterize_line_label_normals(x0, y0, x1, y1, bary0, bary1, w, h,
                                 label_img, mesh_label, normal_sums,
                                 normal_counts, tri_vertex_normals):
    """Rasterize a line segment while accumulating normals per pixel."""
    dx = x1 - x0
    dy = y1 - y0

    steps = int(max(abs(dx), abs(dy)))
    if steps == 0:
        ix = int(round(x0))
        iy = int(round(y0))
        if 0 <= ix < w and 0 <= iy < h:
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
            label_img[iy, ix] = mesh_label
            normal_sums[iy, ix, 0] += nx
            normal_sums[iy, ix, 1] += ny
            normal_sums[iy, ix, 2] += nz
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
            label_img[iy, ix] = mesh_label
            normal_sums[iy, ix, 0] += nx
            normal_sums[iy, ix, 1] += ny
            normal_sums[iy, ix, 2] += nz
            normal_counts[iy, ix] += 1
        x_f += x_inc
        y_f += y_inc


@jit(nopython=True)
def process_slice_points_label_normals(vertices, triangles, mesh_labels,
                                       vertex_normals, triangle_normals,
                                       triangle_use_vertex_normals,
                                       zslice, w, h):
    """Rasterize labels and accumulate per-voxel normals for a z slice."""
    label_img = np.zeros((h, w), dtype=np.uint16)
    normal_sums = np.zeros((h, w, 3), dtype=np.float32)
    normal_counts = np.zeros((h, w), dtype=np.uint16)

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
        vertices_arr = (v0, v1, v2)

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
                for k in range(3):
                    bary[k] = 0.0
                for k in range(3):
                    if tri_vertices[k] == a_idx:
                        bary[k] = 1.0
                        break
            elif abs(z_b - zslice) < 1e-8:
                p_x = b[0]
                p_y = b[1]
                bary = np.zeros(3, dtype=np.float32)
                for k in range(3):
                    bary[k] = 0.0
                for k in range(3):
                    if tri_vertices[k] == b_idx:
                        bary[k] = 1.0
                        break
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
                for k in range(3):
                    bary[k] = 0.0
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
                for k in range(3):
                    bary_pts[n_inter, k] = bary[k]
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
                for k in range(3):
                    tri_vertex_normals[k, 0] = tri_norm[0]
                    tri_vertex_normals[k, 1] = tri_norm[1]
                    tri_vertex_normals[k, 2] = tri_norm[2]

            for ii in range(n_inter):
                for jj in range(ii + 1, n_inter):
                    x0 = pts_2d[ii, 0]
                    y0 = pts_2d[ii, 1]
                    x1 = pts_2d[jj, 0]
                    y1 = pts_2d[jj, 1]
                    bary0 = bary_pts[ii]
                    bary1 = bary_pts[jj]
                    rasterize_line_label_normals(x0, y0, x1, y1, bary0, bary1,
                                                 w, h, label_img, label,
                                                 normal_sums, normal_counts,
                                                 tri_vertex_normals)

    for y in range(h):
        for x in range(w):
            count = normal_counts[y, x]
            if count > 0:
                nx = normal_sums[y, x, 0] / count
                ny = normal_sums[y, x, 1] / count
                nz = normal_sums[y, x, 2] / count
                length = np.sqrt(nx * nx + ny * ny + nz * nz)
                if length > 1e-12:
                    normal_sums[y, x, 0] = nx / length
                    normal_sums[y, x, 1] = ny / length
                    normal_sums[y, x, 2] = nz / length
                else:
                    normal_sums[y, x, 0] = 0.0
                    normal_sums[y, x, 1] = 0.0
                    normal_sums[y, x, 2] = 0.0

    return label_img, normal_sums


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


def process_mesh(mesh_path, mesh_index, include_normals):
    """
    Load a mesh from disk, return (vertices, triangles, labels_for_those_triangles).
    We assign mesh_index+1 as the label.
    """
    print(f"Processing mesh: {mesh_path}")
    mesh = o3d.io.read_triangle_mesh(mesh_path)

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    triangles = np.asarray(mesh.triangles, dtype=np.int32)

    # Every triangle in this mesh gets the same label: mesh_index+1
    labels = np.full(len(triangles), mesh_index + 1, dtype=np.uint16)

    vertex_normals = np.zeros((0, 3), dtype=np.float32)
    triangle_normals = np.zeros((0, 3), dtype=np.float32)
    use_vertex_normals = False

    if include_normals:
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
            # Ensure triangle normals exist even if unavailable in file.
            mesh.compute_triangle_normals()
            triangle_normals = np.asarray(mesh.triangle_normals, dtype=np.float32)

        if vertex_normals.shape[0] == 0:
            vertex_normals = np.zeros((len(vertices), 3), dtype=np.float32)
            use_vertex_normals = False

        if triangle_normals.shape[0] == 0:
            triangle_normals = np.zeros((len(triangles), 3), dtype=np.float32)

    return vertices, triangles, labels, vertex_normals, triangle_normals, use_vertex_normals


def process_slice(args):
    """Process a single z-slice and return label and optional normal arrays."""
    (zslice, vertices, triangles, labels, w, h,
     include_normals, vertex_normals, triangle_normals,
     triangle_use_vertex_normals) = args

    if include_normals:
        img_label, normals_img = process_slice_points_label_normals(
            vertices, triangles, labels, vertex_normals, triangle_normals,
            triangle_use_vertex_normals, zslice, w, h)
    else:
        img_label = process_slice_points_label(vertices, triangles, labels, zslice, w, h)
        normals_img = None

    return zslice, img_label, normals_img


def main():
    # Parse command-line arguments.
    parser = argparse.ArgumentParser(
        description="Process OBJ meshes and slice them along z to produce multiscale OME-Zarr volumes."
    )
    parser.add_argument("folder",
                        help="Path to folder containing OBJ meshes (or parent folder with subfolders of OBJ meshes)")
    parser.add_argument("--scroll", required=True,
                        choices=["scroll1", "scroll2", "scroll3", "scroll4", "scroll5", "0500p2", "0139",
                                 "343p_2um_116", "343p_9um"],
                        help="Scroll shape to use (determines image dimensions)")
    parser.add_argument("--output_path", default="mesh_labels.zarr",
                        help="Path to the label OME-Zarr store (default: mesh_labels.zarr)")
    parser.add_argument("--num_workers", type=int, default=default_workers,
                        help="Number of worker processes to use (default: half of CPU count)")
    parser.add_argument("--recursive", action="store_true",
                        help="Force recursive search in subfolders even if OBJ files exist in the parent folder")
    parser.add_argument("--chunk_size", type=int, default=0,
                        help="Override chunk edge length for Zarr datasets (use 0 for auto)")
    parser.add_argument("--output_normals", action="store_true",
                        help="Also export per-voxel surface normals as an OME-Zarr pyramid")
    parser.add_argument("--normals_output_path", default="mesh_normals.zarr",
                        help="Path to the normals OME-Zarr store (default: mesh_normals.zarr)")
    parser.add_argument("--normals_dtype", default="float16",
                        help="Floating point dtype for the normals pyramid (must be <= float16)")
    args = parser.parse_args()

    if args.chunk_size < 0:
        print("ERROR: chunk_size must be non-negative.")
        sys.exit(1)

    normals_dtype = None
    if args.output_normals:
        normals_dtype = np.dtype(args.normals_dtype)
        if normals_dtype.kind != 'f':
            print("ERROR: normals dtype must be a floating point type.")
            sys.exit(1)
        if normals_dtype.itemsize * 8 > 16:
            print("ERROR: normals dtype cannot exceed 16 bits per component.")
            sys.exit(1)

    # Use the provided number of worker processes.
    N_PROCESSES = args.num_workers
    print(f"Using {N_PROCESSES} worker processes")
    set_num_threads(N_PROCESSES)

    # Folder where OBJ meshes are located.
    folder_path = args.folder
    print(f"Using mesh folder: {folder_path}")

    # Set the image dimensions based on the specified scroll.
    scroll_shapes = {
        "scroll1": (7888, 8096),  # (h, w) for scroll1
        "scroll2": (10112, 11984),  # (h, w) for scroll2
        "scroll3": (3550, 3400),  # (h, w) for scroll3
        "scroll4": (3440, 3340),  # (h, w) for scroll4
        "scroll5": (6700, 9100),  # (h, w) for scroll5
        "0500p2": (4712, 4712),
        "343p_2um_116": (13155, 13155),
        "343p_9um": (5057, 5057)
    }
    if args.scroll not in scroll_shapes:
        print("Invalid scroll shape specified.")
        sys.exit(1)

    # Here, the shape is defined as (height, width)
    h, w = scroll_shapes[args.scroll]
    print(f"Using scroll '{args.scroll}' with dimensions: height={h}, width={w}")

    out_path = args.output_path
    print(f"Label OME-Zarr output path: {out_path}")

    normals_out_path = None
    if args.output_normals:
        normals_out_path = args.normals_output_path
        print(f"Normals OME-Zarr output path: {normals_out_path}")

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
                repeat(args.output_normals)
            )
        )

    # Merge all into a single set of (vertices, triangles, labels).
    all_vertices = []
    all_triangles = []
    all_labels = []
    all_vertex_normals = []
    all_triangle_normals = []
    triangle_use_vertex_flags = []
    vertex_offset = 0

    for (vertices_i, triangles_i, labels_i, vertex_normals_i,
         triangle_normals_i, use_vertex_normals_i) in mesh_results:
        all_vertices.append(vertices_i)
        all_triangles.append(triangles_i + vertex_offset)
        all_labels.append(labels_i)

        if args.output_normals:
            all_vertex_normals.append(vertex_normals_i)
            all_triangle_normals.append(triangle_normals_i)
            triangle_use_vertex_flags.append(
                np.full(len(triangles_i), use_vertex_normals_i, dtype=np.bool_)
            )

        vertex_offset += len(vertices_i)

    # Create the big arrays.
    vertices = np.vstack(all_vertices)
    triangles = np.vstack(all_triangles)
    mesh_labels = np.concatenate(all_labels)

    if args.output_normals:
        vertex_normals = np.vstack(all_vertex_normals)
        triangle_normals = np.vstack(all_triangle_normals)
        triangle_use_vertex_normals = np.concatenate(triangle_use_vertex_flags)
    else:
        vertex_normals = np.zeros((0, 3), dtype=np.float32)
        triangle_normals = np.zeros((0, 3), dtype=np.float32)
        triangle_use_vertex_normals = np.zeros(0, dtype=np.bool_)

    # Determine slice range from the vertices.
    z_min = int(np.floor(vertices[:, 2].min()))
    z_max = int(np.ceil(vertices[:, 2].max()))
    z_slices = np.arange(z_min, z_max + 1)
    print(f"Processing slices from {z_min} to {z_max} (inclusive).")
    print(f"Total number of slices: {len(z_slices)}")

    num_slices = len(z_slices)
    num_levels = PYRAMID_LEVELS
    z_index_lookup = {z: idx for idx, z in enumerate(z_slices)}

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

    label_store = zarr.DirectoryStore(out_path)
    label_root = zarr.group(store=label_store, overwrite=True)
    label_base_shape = (num_slices, h, w)
    label_datasets = []
    label_axes = [
        {"name": "z", "type": "space", "unit": "index"},
        {"name": "y", "type": "space", "unit": "pixel"},
        {"name": "x", "type": "space", "unit": "pixel"},
    ]
    label_translation = [float(z_slices[0]), 0.0, 0.0]
    label_datasets_meta = []

    for level in range(num_levels):
        level_shape = compute_level_shape(label_base_shape, level, (1, 2))
        level_chunks = compute_chunks(level_shape, args.chunk_size, level)
        dataset_name = f"{level}"
        ds = label_root.create_dataset(
            dataset_name,
            shape=level_shape,
            chunks=level_chunks,
            dtype=np.uint16,
            compressor=compressor,
            overwrite=True,
        )
        label_datasets.append(ds)
        scale_vector = [1.0, float(2 ** level), float(2 ** level)]
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

    normals_datasets = []
    if args.output_normals:
        normals_store = zarr.DirectoryStore(normals_out_path)
        normals_root = zarr.group(store=normals_store, overwrite=True)
        normals_base_shape = (num_slices, h, w, 3)
        normals_axes = [
            {"name": "z", "type": "space", "unit": "index"},
            {"name": "y", "type": "space", "unit": "pixel"},
            {"name": "x", "type": "space", "unit": "pixel"},
            {"name": "c", "type": "channel"},
        ]
        normals_translation = [float(z_slices[0]), 0.0, 0.0, 0.0]
        normals_datasets_meta = []

        for level in range(num_levels):
            level_shape = compute_level_shape(normals_base_shape, level, (1, 2))
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
            scale_vector = [1.0, float(2 ** level), float(2 ** level), 1.0]
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

    label_dataset_name = label_datasets[0].path
    normals_dataset_name = normals_datasets[0].path if normals_datasets else None

    slice_state = {
        "vertices": vertices,
        "triangles": triangles,
        "labels": mesh_labels,
        "w": w,
        "h": h,
        "include_normals": bool(args.output_normals),
        "vertex_normals": vertex_normals,
        "triangle_normals": triangle_normals,
        "use_vertex_normals": triangle_use_vertex_normals,
        "z_min": z_slices[0],
        "label_store_path": out_path,
        "label_dataset_name": label_dataset_name,
        "normals_store_path": normals_out_path,
        "normals_dataset_name": normals_dataset_name,
        "normals_dtype": normals_dtype,
    }

    def _write_slice_output(zslice, label_img, normals_img):
        slice_index = z_index_lookup[zslice]

        label_datasets[0][slice_index, ...] = np.ascontiguousarray(
            label_img, dtype=label_datasets[0].dtype
        )

        if args.output_normals and normals_img is not None:
            normals_datasets[0][slice_index, ...] = np.ascontiguousarray(
                normals_img.astype(normals_dtype, copy=False)
            )

    sequential_slice_args = (
        vertices,
        triangles,
        mesh_labels,
        w,
        h,
        args.output_normals,
        vertex_normals,
        triangle_normals,
        triangle_use_vertex_normals,
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
        ) = sequential_slice_args

        for z in tqdm(z_slices, total=len(z_slices), desc="Slices processed"):
            zslice, label_img, normals_img = process_slice(
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
                )
            )
            _write_slice_output(zslice, label_img, normals_img)

    if args.chunk_size > 0:
        label_datasets[0] = _rechunk_with_dask(
            label_datasets[0], args.chunk_size, "Labels level 0", N_PROCESSES
        )
        if args.output_normals:
            normals_datasets[0] = _rechunk_with_dask(
                normals_datasets[0], args.chunk_size, "Normals level 0", N_PROCESSES
            )

    def populate_downsampled_levels(datasets, store_path, axes, desc):
        if not datasets:
            return
        for level in range(1, len(datasets)):
            parent = datasets[level - 1]
            target = datasets[level]
            num_planes = parent.shape[0]
            level_desc = f"{desc} level {level}"
            if num_planes == 0:
                continue

            def _downsample_and_store(idx):
                source = np.asarray(parent[idx, ...])
                downsampled = downsample_2x(source, axes=axes)
                target[idx, ...] = np.ascontiguousarray(downsampled, dtype=target.dtype)

            use_parallel = False
            mp_context = None
            if N_PROCESSES > 1:
                try:
                    mp_context = multiprocessing.get_context("fork")
                    use_parallel = True
                except ValueError:
                    mp_context = None

            if use_parallel and mp_context is not None:
                state = {
                    "store_path": store_path,
                    "parent_dataset_path": parent.path,
                    "target_dataset_path": target.path,
                    "axes": tuple(axes),
                }
                _set_downsample_state(state)
                try:
                    with ProcessPoolExecutor(
                        max_workers=N_PROCESSES,
                        mp_context=mp_context,
                    ) as executor:
                        for _ in tqdm(
                            executor.map(_downsample_plane_worker, range(num_planes), chunksize=1),
                            total=num_planes,
                            desc=level_desc,
                            leave=False,
                        ):
                            pass
                finally:
                    _set_downsample_state({})
            else:
                for idx in tqdm(range(num_planes), desc=level_desc, leave=False):
                    _downsample_and_store(idx)

    populate_downsampled_levels(label_datasets, out_path, axes=(0, 1), desc="Labels")

    if args.output_normals:
        populate_downsampled_levels(normals_datasets, normals_out_path, axes=(0, 1), desc="Normals")

    print("Completed OME-Zarr export.")


if __name__ == "__main__":
    main()
