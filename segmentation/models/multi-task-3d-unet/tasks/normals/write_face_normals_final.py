from glob import glob
import numpy as np
import open3d as o3d
import os
import tifffile
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from numba import jit, prange
from numba import set_num_threads

# Configuration
N_PROCESSES = 4
print(f"Using {N_PROCESSES} processes")
set_num_threads(N_PROCESSES)

# Global parameters
EXPANSION_FACTOR = 1.5
MAX_INTERSECTIONS = 3  # Maximum number of intersections per triangle


@jit(nopython=True)
def normalize_vector(v):
    """Normalize a vector."""
    norm = np.sqrt(np.sum(v * v))
    if norm == 0:
        return v.astype(np.float32)
    return (v / norm).astype(np.float32)


@jit(nopython=True)
def interpolate_normal_jit(n0, n1, t):
    """JIT-compatible normal interpolation."""
    normal = (1 - t) * n0 + t * n1
    return normalize_vector(normal)


@jit(nopython=True)
def get_intersection_point(start, end, start_normal, end_normal, z_plane):
    """JIT-compatible edge intersection that also handles vertices on the plane."""
    # Convert inputs to float32
    start = start.astype(np.float32)
    end = end.astype(np.float32)
    start_normal = start_normal.astype(np.float32)
    end_normal = end_normal.astype(np.float32)
    z_plane = np.float32(z_plane)

    # Check if either vertex is exactly on the plane
    if abs(start[2] - z_plane) <= 1e-8:
        return start[:2].astype(np.float32), start_normal
    if abs(end[2] - z_plane) <= 1e-8:
        return end[:2].astype(np.float32), end_normal

    # If neither vertex is on the plane, check for edge intersection
    if abs(end[2] - start[2]) <= 1e-8:  # Edge is parallel to plane
        return None, None

    t = (z_plane - start[2]) / (end[2] - start[2])
    if not (-0.01 <= t <= 1.01):  # Slightly relaxed bounds
        return None, None

    intersection = start + t * (end - start)
    normal = start_normal + t * (end_normal - start_normal)
    normal = normalize_vector(normal)
    return intersection[:2].astype(np.float32), normal


@jit(nopython=True)
def interpolate_line_with_expansion(x0, y0, x1, y1, n0, n1, w, h, img, viz_img, exp_factor):
    """Interpolate line with normal-direction expansion."""
    dx = x1 - x0
    dy = y1 - y0

    distance = np.sqrt(dx * dx + dy * dy)
    steps = max(int(distance * 2), max(abs(dx), abs(dy))) + 1
    effective_exp_factor = exp_factor * 1.2

    for step in range(steps):
        t = step / (steps - 1) if steps > 1 else 0
        x = x0 + t * dx
        y = y0 + t * dy
        normal = interpolate_normal_jit(n0, n1, t)

        num_exp_steps = int(4 * effective_exp_factor + 1)
        for e in range(num_exp_steps):
            t_exp = (e / (num_exp_steps - 1)) * 2 - 1
            exp_x = int(round(x + t_exp * effective_exp_factor * normal[0]))
            exp_y = int(round(y + t_exp * effective_exp_factor * normal[1]))

            if 0 <= exp_x < w and 0 <= exp_y < h:
                normal_rgb = ((normal + 1) * 32767.5).astype(np.uint16)
                img[exp_y, exp_x] = normal_rgb
                viz_img[exp_y, exp_x] = ((normal_rgb * 255) // 65535).astype(np.uint8)


@jit(nopython=True)
def process_slice_points(vertices, triangles, vertex_normals, zslice, w, h, exp_factor):
    """JIT-compatible slice processing with improved vertex handling."""
    img = np.zeros((h, w, 3), dtype=np.uint16)
    viz_img = np.zeros((h, w, 3), dtype=np.uint8)

    # Convert inputs to float32
    vertices = vertices.astype(np.float32)
    vertex_normals = vertex_normals.astype(np.float32)
    zslice = np.float32(zslice)

    # Pre-allocate arrays for intersection points and normals
    points_array = np.zeros((MAX_INTERSECTIONS, 2), dtype=np.float32)
    normals_array = np.zeros((MAX_INTERSECTIONS, 3), dtype=np.float32)

    for i in range(len(triangles)):
        v0 = vertices[triangles[i, 0]]
        v1 = vertices[triangles[i, 1]]
        v2 = vertices[triangles[i, 2]]
        n0 = vertex_normals[triangles[i, 0]]
        n1 = vertex_normals[triangles[i, 1]]
        n2 = vertex_normals[triangles[i, 2]]

        if min(v0[2], v1[2], v2[2]) <= zslice <= max(v0[2], v1[2], v2[2]):
            num_intersections = 0

            # Check each edge
            for start, end, start_normal, end_normal in [
                (v0, v1, n0, n1),
                (v1, v2, n1, n2),
                (v2, v0, n2, n0)
            ]:
                if num_intersections < MAX_INTERSECTIONS:
                    p, n = get_intersection_point(start, end, start_normal, end_normal, zslice)
                    if p is not None:
                        # Check for duplicates
                        is_duplicate = False
                        for j in range(num_intersections):
                            if np.sum((p - points_array[j]) ** 2) < 1e-10:
                                is_duplicate = True
                                break
                        if not is_duplicate:
                            points_array[num_intersections] = p
                            normals_array[num_intersections] = n
                            num_intersections += 1

            # Draw lines between points
            if num_intersections >= 2:
                for i in range(num_intersections):
                    for j in range(i + 1, num_intersections):
                        x0 = int(round(points_array[i, 0]))
                        y0 = int(round(points_array[i, 1]))
                        x1 = int(round(points_array[j, 0]))
                        y1 = int(round(points_array[j, 1]))

                        if (0 <= x0 < w and 0 <= y0 < h and
                                0 <= x1 < w and 0 <= y1 < h):
                            interpolate_line_with_expansion(
                                x0, y0, x1, y1,
                                normals_array[i], normals_array[j],
                                w, h, img, viz_img, exp_factor
                            )

    return img, viz_img


def process_mesh(mesh_path):
    """Process a single mesh file."""
    print(f"Processing mesh: {mesh_path}")
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    vertex_normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
    triangles = np.asarray(mesh.triangles)

    return vertices, triangles, vertex_normals


def process_slice(args):
    """Process a single z-slice."""
    zslice, vertices, triangles, vertex_normals, w, h, out_path, viz_path = args
    print(f"Processing slice {zslice}")

    img, viz_img = process_slice_points(vertices, triangles, vertex_normals, zslice, w, h, EXPANSION_FACTOR)

    if np.any(img):
        tifffile.imwrite(f'{out_path}/{zslice}.tif', img, compression='zlib')
        Image.fromarray(viz_img).save(f'{viz_path}/{zslice}.jpg', quality=90)

    return zslice


def main():
    out_path = "s4_normals"
    viz_path = f'{out_path}/viz'
    os.makedirs(viz_path, exist_ok=True)
    os.makedirs(out_path, exist_ok=True)

    w, h = 3440, 3340
    meshes = glob('../../scroll4_meshes/*.obj')
    print(f"Found {len(meshes)} meshes to process")

    all_vertices = []
    all_triangles = []
    all_normals = []
    vertex_offset = 0

    with ProcessPoolExecutor(max_workers=N_PROCESSES) as executor:
        mesh_results = list(executor.map(process_mesh, meshes))

    for vertices, triangles, normals in mesh_results:
        all_vertices.append(vertices)
        all_triangles.append(triangles + vertex_offset)
        all_normals.append(normals)
        vertex_offset += len(vertices)

    vertices = np.vstack(all_vertices)
    triangles = np.vstack(all_triangles)
    normals = np.vstack(all_normals)

    z_min = np.floor(vertices[:, 2].min()).astype(int)
    z_max = np.ceil(vertices[:, 2].max()).astype(int)
    z_slices = np.arange(z_min, z_max + 1)

    print(f"Processing slices from {z_min} to {z_max}")
    print(f"Total number of slices: {len(z_slices)}")

    slice_args = [(z, vertices, triangles, normals, w, h, out_path, viz_path)
                  for z in z_slices]

    completed_slices = 0
    total_slices = len(slice_args)

    with ProcessPoolExecutor(max_workers=N_PROCESSES) as executor:
        for zslice in executor.map(process_slice, slice_args):
            completed_slices += 1
            print(f"Completed slice {zslice} ({completed_slices}/{total_slices})")


if __name__ == "__main__":
    main()