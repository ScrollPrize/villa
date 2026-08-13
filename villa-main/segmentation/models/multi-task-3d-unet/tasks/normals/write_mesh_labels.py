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
N_PROCESSES = 8
print(f"Using {N_PROCESSES} processes")
set_num_threads(N_PROCESSES)

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
    Simple line rasterization in label_img with the integer mesh_label.
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
def process_slice_points_label(
    vertices, triangles, mesh_labels, zslice, w, h
):
    """
    For the plane z=zslice, find the intersection lines of each triangle
    and draw them into a 2D array (label_img) using the triangle's
    mesh label.
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
                    rasterize_line_label(
                        x0, y0, x1, y1,
                        w, h,
                        label_img,
                        label
                    )

    return label_img

def process_mesh(mesh_path, mesh_index):
    """
    Load a mesh from disk, return (vertices, triangles, labels_for_those_triangles).
    We assign mesh_index+1 (or mesh_index if you prefer 0-based) as the label.
    """
    print(f"Processing mesh: {mesh_path}")
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    # We do NOT need normals for label lines, so skip compute_vertex_normals.
    # mesh.compute_vertex_normals()

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    triangles = np.asarray(mesh.triangles, dtype=np.int32)

    # Every triangle in this mesh gets the same label: mesh_index+1
    labels = np.full(len(triangles), mesh_index+1, dtype=np.uint16)

    return vertices, triangles, labels

def process_slice(args):
    """
    Process a single z-slice, writing out a label TIF if non-empty.
    """
    (
        zslice,
        vertices,
        triangles,
        labels,
        w,
        h,
        out_path
    ) = args

    print(f"Processing slice z={zslice}")
    img_label = process_slice_points_label(vertices, triangles, labels, zslice, w, h)

    # Save if there's at least one nonzero pixel
    if np.any(img_label):
        out_file = f"{out_path}/{zslice}.tif"
        tifffile.imwrite(out_file, img_label, compression='zlib')

    return zslice

def main():
    out_path = "mesh_labels_slices"
    os.makedirs(out_path, exist_ok=True)

    # Image size in XY for slicing:
    w, h = 8096, 7888

    # Find meshes
    mesh_paths = glob('scroll1_gp_meshes/*.obj')
    print(f"Found {len(mesh_paths)} meshes to process")

    # Read all meshes (in parallel if desired)
    with ProcessPoolExecutor(max_workers=N_PROCESSES) as executor:
        mesh_results = list(
            executor.map(process_mesh, mesh_paths, range(len(mesh_paths)))
        )

    # Merge all into a single set of (vertices, triangles, labels)
    all_vertices = []
    all_triangles = []
    all_labels = []
    vertex_offset = 0

    for (vertices_i, triangles_i, labels_i) in mesh_results:
        all_vertices.append(vertices_i)
        # Shift triangles by current offset
        all_triangles.append(triangles_i + vertex_offset)
        all_labels.append(labels_i)
        vertex_offset += len(vertices_i)

    # Create the big arrays
    vertices = np.vstack(all_vertices)
    triangles = np.vstack(all_triangles)
    mesh_labels = np.concatenate(all_labels)

    # Determine slice range
    z_min = int(np.floor(vertices[:, 2].min()))
    z_max = int(np.ceil(vertices[:, 2].max()))
    z_slices = np.arange(z_min, z_max + 1)

    print(f"Processing slices from {z_min} to {z_max} (inclusive).")
    print(f"Total number of slices: {len(z_slices)}")

    # Prepare parallel arguments for slices
    slice_args = [
        (z, vertices, triangles, mesh_labels, w, h, out_path)
        for z in z_slices
    ]

    # Run slice processing in parallel
    completed_slices = 0
    total_slices = len(slice_args)

    with ProcessPoolExecutor(max_workers=N_PROCESSES) as executor:
        for zslice in executor.map(process_slice, slice_args):
            completed_slices += 1
            print(f"Completed slice {zslice} ({completed_slices}/{total_slices})")

if __name__ == "__main__":
    main()
