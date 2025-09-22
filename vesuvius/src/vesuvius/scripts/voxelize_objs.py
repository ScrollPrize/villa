#!/usr/bin/env python
from glob import glob
from itertools import repeat
import numpy as np
import open3d as o3d
import os
import tifffile
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from numba import jit, prange, set_num_threads
import sys
import argparse
from tqdm import tqdm  # Progress bar

# Determine default number of workers: half of CPU count (at least 1)
default_workers = max(1, multiprocessing.cpu_count() // 2)

# We no longer need normals for expansion, but we still need to find
# intersections of triangles with the z-plane.
MAX_INTERSECTIONS = 3  # Maximum number of intersections per triangle


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
    """Process a single z-slice, optionally writing normals alongside labels."""
    (zslice, vertices, triangles, labels, w, h, out_path,
     include_normals, vertex_normals, triangle_normals,
     triangle_use_vertex_normals, normals_out_path,
     normals_dtype) = args

    if include_normals:
        img_label, normals_img = process_slice_points_label_normals(
            vertices, triangles, labels, vertex_normals, triangle_normals,
            triangle_use_vertex_normals, zslice, w, h)
    else:
        img_label = process_slice_points_label(vertices, triangles, labels, zslice, w, h)
        normals_img = None

    if np.any(img_label):
        slice_name = str(zslice).zfill(5)
        label_file = os.path.join(out_path, f"{slice_name}.tif")
        tifffile.imwrite(label_file, img_label, compression='zlib')

        if include_normals and normals_img is not None:
            dtype_to_use = normals_dtype if normals_dtype is not None else np.dtype('float16')
            normals_to_save = normals_img.astype(dtype_to_use)
            normals_file = os.path.join(normals_out_path, f"{slice_name}.tif")
            tifffile.imwrite(normals_file, normals_to_save, compression='zlib')

    return zslice


def main():
    # Parse command-line arguments.
    parser = argparse.ArgumentParser(
        description="Process OBJ meshes and slice them along z to produce label images."
    )
    parser.add_argument("folder",
                        help="Path to folder containing OBJ meshes (or parent folder with subfolders of OBJ meshes)")
    parser.add_argument("--scroll", required=True,
                        choices=["scroll1", "scroll2", "scroll3", "scroll4", "scroll5", "0500p2", "0139",
                                 "343p_2um_116", "343p_9um"],
                        help="Scroll shape to use (determines image dimensions)")
    parser.add_argument("--output_path", default="mesh_labels_slices",
                        help="Output folder for label images (default: mesh_labels_slices)")
    parser.add_argument("--num_workers", type=int, default=default_workers,
                        help="Number of worker processes to use (default: half of CPU count)")
    parser.add_argument("--recursive", action="store_true",
                        help="Force recursive search in subfolders even if OBJ files exist in the parent folder")
    parser.add_argument("--output_normals", action="store_true",
                        help="Also export per-voxel surface normal slices")
    parser.add_argument("--normals_output_path", default="mesh_normals_slices",
                        help="Output folder for surface normal slices (default: mesh_normals_slices)")
    parser.add_argument("--normals_dtype", default="float16",
                        help="Floating point dtype for normal slices (must be <= float16)")
    args = parser.parse_args()

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

    # Folder where label images will be saved.
    out_path = args.output_path
    os.makedirs(out_path, exist_ok=True)
    print(f"Output folder for label images: {out_path}")

    normals_out_path = None
    if args.output_normals:
        normals_out_path = args.normals_output_path
        os.makedirs(normals_out_path, exist_ok=True)
        print(f"Output folder for surface normals: {normals_out_path}")

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

    # Prepare parallel arguments for slices.
    slice_args = [
        (
            z,
            vertices,
            triangles,
            mesh_labels,
            w,
            h,
            out_path,
            args.output_normals,
            vertex_normals,
            triangle_normals,
            triangle_use_vertex_normals,
            normals_out_path,
            normals_dtype,
        )
        for z in z_slices
    ]

    # Run slice processing in parallel with a tqdm progress bar.
    with ProcessPoolExecutor(max_workers=N_PROCESSES) as executor:
        for _ in tqdm(executor.map(process_slice, slice_args), total=len(slice_args), desc="Slices processed"):
            pass


if __name__ == "__main__":
    main()
