#!/usr/bin/env python3
"""
pointcloud_mesh_refinement.py
Script to generate a refined mesh using the raw pointcloud and an initial mesh.
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import random
import open3d as o3d
from scipy.spatial import cKDTree, Delaunay
import os
import sys
import pickle
import glob
from tqdm import tqdm

sys.path.append('ThaumatoAnakalyptor/graph_problem/build')
import graph_problem_gpu_py

### UTILS ###

def shuffling_points_axis(points, axis_indices=[2, 0, 1, 3]):
    """
    Rotate points by reshuffling axis
    """
    # Reshuffle axis in points
    points = points[:, axis_indices]
    # Return points
    return points

def load_mesh(mesh_file):
    print (f"Loading mesh from {mesh_file}")
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    vertices = np.asarray(mesh.vertices)
    metadata_file = os.path.join(os.path.dirname(mesh_file), "mesh_metadata.pkl")
    if os.path.exists(metadata_file):
        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)
    else:
        raise ValueError("Metadata file not found.")
    return vertices, metadata

def load_pointcloud_slab(slab_path):
    # Open the npz file
    with open(slab_path, 'rb') as f:
        numpyfile = np.load(f)
        points = numpyfile['points']
    return points

def save_winding_pointcloud(winding_path, winding_nr, points):
    # make folder if it does not exist
    os.makedirs(winding_path, exist_ok=True)
    # Save the winding pointcloud
    file_path = os.path.join(winding_path, f"winding_{int(winding_nr)}.npz")
    # Load existing points if any, else create empty array matching new data shape
    points = np.asarray(points)
    if os.path.exists(file_path):
        data = np.load(file_path)
        points_prev = data.get('points')
        # Ensure dimensions match
        if points_prev.ndim != 2 or points_prev.shape[1] != points.shape[1]:
            raise ValueError(
                f"Dimension mismatch in save_winding_pointcloud: existing points have shape {points_prev.shape}, "
                f"new points have shape {points.shape}."
            )
    else:
        # initialize empty array with same number of columns
        points_prev = np.zeros((0, points.shape[1]), dtype=points.dtype)
    # concatenate and dedupe
    points = np.concatenate((points_prev, points), axis=0)
    points = np.unique(points, axis=0)
    np.savez_compressed(file_path, points=points)

def load_winding_pointcloud(winding_path, winding_nr):
    # Load the winding pointcloud
    file_path = os.path.join(winding_path, f"winding_{int(winding_nr)}.npz")
    if os.path.exists(file_path):
        points = np.load(file_path)['points']
        return points
    else:
        raise ValueError(f"Winding pointcloud {file_path} not found.")
    
def save_flattened_winding(winding_path, winding_nr, points, uvs):
    # make folder if it does not exist
    os.makedirs(winding_path, exist_ok=True)
    # Save the winding pointcloud
    file_path = os.path.join(winding_path, f"flattened_winding_{int(winding_nr)}.npz")
    np.savez_compressed(file_path, points=points, uvs=uvs)

def load_flattened_winding(winding_path, winding_nr):
    # Load the winding pointcloud
    file_path = os.path.join(winding_path, f"flattened_winding_{int(winding_nr)}.npz")
    if os.path.exists(file_path):
        points = np.load(file_path)['points']
        uvs = np.load(file_path)['uvs']
        return points, uvs
    else:
        raise ValueError(f"Winding pointcloud {file_path} not found.")
    
def save_pointcloud_winding(winding_path, winding_nr, points):
    # make folder if it does not exist
    os.makedirs(winding_path, exist_ok=True)
    # Save the winding pointcloud to ply directly with open3d
    file_path = os.path.join(winding_path, f"winding_{int(winding_nr)}.ply")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3].astype(np.float64))
    o3d.io.write_point_cloud(file_path, pcd)
    
def clean_winding_dicts(winding_range, flattened_winding_path, winding_files_indices, prev_uvs_u, prev_uvs_v, prev_points):
    start, end = winding_range
    # save fixed wrap and remove from dicts
    dict_entries = list(prev_uvs_u.keys())
    for wrap_nr in dict_entries:
        if wrap_nr not in winding_files_indices[start:end]:
            # save to npy
            save_flattened_winding(flattened_winding_path, wrap_nr, prev_points[wrap_nr], np.array([prev_uvs_u[wrap_nr], prev_uvs_v[wrap_nr]]).T)
            # delete
            del prev_uvs_u[wrap_nr]
            del prev_uvs_v[wrap_nr]
            del prev_points[wrap_nr]

def subsample_min_dist(points, r):
    """
    Greedy "pack" subsampling: once a point is kept, all its neighbors
    within r get excluded from future consideration.
    
    Args:
        points: (N,2) array-like of 2D coords
        r:      minimum allowed distance

    Returns:
        accepted: list of original indices s.t. no two are closer than r
    """
    pts = np.asarray(points, dtype=float)
    tree = cKDTree(pts)
    
    # start with a random order of all indices
    order = np.arange(len(pts))
    np.random.shuffle(order)
    
    accepted = []
    removed = np.zeros(len(pts), dtype=bool)
    
    for idx in order:
        if removed[idx]:
            continue
        # accept this one…
        accepted.append(int(idx))
        # …and mark all neighbors within r so we never pick them
        neighbors = tree.query_ball_point(pts[idx], r)
        removed[neighbors] = True
    
    return accepted


def filter_by_density(points, selected_indices, r, n):
    """
    Keep only those indices whose point has strictly more than n neighbors
    within distance r (counting itself).

    Args:
        points:            (N,2) array-like of 2D coords
        selected_indices:  list of indices to test
        r:                 radius for neighbor counting
        n:                 threshold (strictly greater)

    Returns:
        filtered: subset of selected_indices meeting the density criterion
    """
    pts = np.asarray(points, dtype=float)
    tree = cKDTree(pts)
    
    # query_ball_point can take an array of query-points at once
    neighbor_lists = tree.query_ball_point(pts[selected_indices], r)
    
    # keep only those with > n neighbors
    filtered = [
        idx for idx, nbrs in zip(selected_indices, neighbor_lists)
        if len(nbrs) > n
    ]
    return filtered

def filter_by_density_mad(points, selected_indices, r, k=3.0):
    """
    MAD-based outlier removal on neighbor counts in UV-space.

    For each index in `selected_indices`, count how many points lie
    within radius `r`.  Compute the median (m) and MAD of those counts,
    then set threshold T = m + k * MAD.  Keep only indices whose count
    > T.

    Args:
        points:           (N,2) array-like of 2D coords
        selected_indices: list of int, indices to test
        r:                float, radius for neighbor counting
        k:                float, MAD multiplier (default=5.0)

    Returns:
        kept_indices: list[int], subset of selected_indices with count > T
        stats: dict with keys 'median', 'mad', 'threshold'
    """
    pts = np.asarray(points, dtype=float)
    tree = cKDTree(pts)

    # batch query neighbor-lists
    neighbor_lists = tree.query_ball_point(pts[selected_indices], r)
    counts = np.array([len(nbrs) for nbrs in neighbor_lists], dtype=float)

    # compute median and MAD
    med = np.median(counts)
    mad = np.median(np.abs(counts - med))
    thr = med - k * mad

    # filter
    kept_indices = [
        idx for idx, cnt in zip(selected_indices, counts)
        if cnt > thr
    ]

    # print some diagnostics
    print(f"Neighbor count median: {med:.2f}")
    print(f"MAD:                    {mad:.2f}")
    print(f"Threshold (m - {k}·MAD): {thr:.2f}")
    print(f"Evaluated {len(counts)} points → kept {len(kept_indices)}")

    return kept_indices

def refine_by_medoid(points, uvs, selected_indices, r_uv):
    """
    For each index in selected_indices:
      1) Find all points whose UV coords lie within r_uv of its UV.
      2) In that UV-neighborhood, compute the median of the 3D coords.
      3) Pick the neighbor whose 3D coordinate is closest to that median.

    Args:
        points:           array-like of shape (N, ≥3), original 3D coords in [:,:3]
        uvs:              array-like of shape (N, 2), original UV coords
        selected_indices: list of int, indices into points/uvs to refine
        r_uv:             float, search radius in UV space

    Returns:
        refined_indices: list of int, same length as selected_indices,
                         each replaced by its local 3D "medoid" index
    """
    pts3 = np.asarray(points, dtype=float)[:, :3]
    uvs2 = np.asarray(uvs,    dtype=float)
    tree_uv = cKDTree(uvs2)

    refined = []
    for idx in selected_indices:
        # 1) UV-space neighbors
        nbrs = tree_uv.query_ball_point(uvs2[idx], r_uv)
        if not nbrs:
            # no UV-neighbors? keep the original
            refined.append(idx)
            continue

        # 2) 3D median of those neighbors
        nbr_coords3 = pts3[nbrs]
        med3 = np.median(nbr_coords3, axis=0)

        # 3) pick the neighbor closest to that 3D median
        d2 = np.sum((nbr_coords3 - med3)**2, axis=1)
        best_local = nbrs[int(np.argmin(d2))]
        refined.append(best_local)

    return refined

def filter_by_edge_error(points, uvs, refined_indices, r_uv, thr):
    """
    For each index in refined_indices:
      1) Find all points whose UV coords lie within r_uv of its UV.
      2) For those neighbors j, compute:
            d_uv  = ||u_i - u_j||
            d_xyz = ||x_i - x_j||
            edge_err_j = |d_xyz - d_uv|
         then mean_err_i = mean(edge_err_j).
      3) Sort all (i, mean_err_i) by mean_err ascending.
      4) Print quartiles, mean, median, min, max, count, threshold.
      5) Return only i with mean_err_i ≤ thr.

    Args:
        points:           array-like (N, ≥3) of original 3D coords
        uvs:              array-like (N, 2)   of original UV coords
        refined_indices:  list[int], indices to evaluate
        r_uv:             float, search radius in UV space
        thr:              float, maximum allowed mean edge-error

    Returns:
        kept_indices: list[int], subset of refined_indices with mean_error ≤ thr
        error_list:   list[(idx, mean_error)], sorted by mean_error ascending
    """
    pts3 = np.asarray(points, dtype=float)[:, :3]
    uvs2 = np.asarray(uvs,    dtype=float)
    tree_uv = cKDTree(uvs2)

    error_list = []
    for i in refined_indices:
        # 1) UV-space neighbors
        nbrs = tree_uv.query_ball_point(uvs2[i], r_uv)
        if not nbrs:
            mean_err = 0.0
        else:
            # 2) compute distances
            d_uv  = np.linalg.norm(uvs2[nbrs]  - uvs2[i],  axis=1)
            d_xyz = np.linalg.norm(pts3[nbrs] - pts3[i], axis=1)
            edge_e = np.abs(d_xyz - d_uv)
            mean_err = edge_e.mean()

        error_list.append((i, mean_err))

    # 3) sort by mean error
    error_list.sort(key=lambda x: x[1])
    errors = [err for _, err in error_list]

    # 4) filter by threshold
    kept_indices = [i for i, err in error_list if err <= thr]

    # 5) print stats
    q1 = np.percentile(errors, 25)
    q2 = np.percentile(errors, 50)
    q3 = np.percentile(errors, 75)
    print(f"Q1: {q1:.4f}, Q2 (median): {q2:.4f}, Q3: {q3:.4f}")
    print(f"Mean error:   {np.mean(errors):.4f}")
    print(f"Median error: {np.median(errors):.4f}")
    print(f"Min error:    {np.min(errors):.4f}")
    print(f"Max error:    {np.max(errors):.4f}")
    print(f"Threshold:    {thr}")
    print(f"Number of points evaluated: {len(refined_indices)}")
    print(f"Number of points kept: {len(kept_indices)}")

    return kept_indices, error_list

def filter_by_edge_error_mad(points, uvs, refined_indices, r_uv, k=3.0):
    """
    MAD-based outlier removal on your UV-vs-3D edge errors.

    Args:
        points:           array-like (N, ≥3) of original 3D coords
        uvs:              array-like (N, 2) of original UV coords
        refined_indices:  list[int], indices to evaluate
        r_uv:             float, UV search radius
        k:                float, multiplier for MAD threshold (default=3)

    Returns:
        kept_indices: list[int], subset of refined_indices with err ≤ m + k·MAD
        error_list:   list[(idx, mean_err)], sorted by mean_err ascending
    """
    pts3 = np.asarray(points, dtype=float)[:, :3]
    uvs2 = np.asarray(uvs,    dtype=float)
    tree_uv = cKDTree(uvs2)

    # 1) compute mean edge-error per refined index
    error_list = []
    for i in refined_indices:
        nbrs = tree_uv.query_ball_point(uvs2[i], r_uv)
        if not nbrs:
            mean_err = 0.0
        else:
            d_uv  = np.linalg.norm(uvs2[nbrs]  - uvs2[i],  axis=1)
            d_xyz = np.linalg.norm(pts3[nbrs] - pts3[i], axis=1)
            mean_err = np.abs(d_xyz - d_uv).mean()
        error_list.append((i, mean_err))

    # 2) sort and pull out just the errors
    error_list.sort(key=lambda x: x[1])
    errors = np.array([err for _, err in error_list])

    # 3) median and MAD
    med   = np.median(errors)
    mad   = np.median(np.abs(errors - med))
    thr   = med + k * mad

    # 4) filter
    kept_indices = [idx for idx, err in error_list if err <= thr]

    # 5) print stats
    print(f"Median error: {med:.4f}")
    print(f"MAD:          {mad:.4f}")
    print(f"Threshold:    median + {k}·MAD = {thr:.4f}")
    print(f"Processed {len(errors)} points → kept {len(kept_indices)}")

    return kept_indices, error_list

def display_3d_pointcloud(points, color_by_angle=False, downsample_ratio=0.1):
    """
    Interactive 3D visualization of points.
    If points has a 4th column, it is interpreted as angle in degrees.
    color_by_angle: if True and angle data present, color points by angle (HSV colormap).
    downsample_ratio: fraction [0..1] of points to randomly keep for display.
    """
    pts = np.asarray(points)
    # extract angles if present
    angles = None
    if pts.ndim == 2 and pts.shape[1] > 3:
        angles = pts[:, 3]
    coords = pts[:, :3]
    # random downsampling for interactive display
    if 0.0 < downsample_ratio < 1.0:
        mask = np.random.rand(len(coords)) < downsample_ratio
        coords = coords[mask]
        if angles is not None:
            angles = angles[mask]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if color_by_angle and angles is not None:
        sc = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=angles,
                        cmap='hsv', s=1)
        cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Angle (deg)')
    else:
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=1)
    ax.set_title("Points 3D")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()

def clean_mesh(mesh,
               longest_edge_pct=95,
               area_pct=95,
               edge_length_thresh=500):
    """
    Aggressively clean up a TriangleMesh:
      • remove duplicated vertices/triangles, degenerate triangles
      • remove non-manifold edges (which also kills many self-intersections)
      • remove the top percentile of triangles by longest edge or by area
      • final pass of non-manifold removal + unreferenced-vertex cleanup
      • recompute normals

    Args:
        mesh:             open3d.geometry.TriangleMesh
        longest_edge_pct: percentile cutoff for longest-edge cull
        area_pct:         percentile cutoff for area cull

    Returns:
        mesh: the cleaned mesh
    """
    # 1) basic topological cleanup
    mesh.remove_duplicated_vertices()

    # 2) cull huge triangles by edge-length / area
    verts = np.asarray(mesh.vertices)
    tris  = np.asarray(mesh.triangles)
    tri_verts = verts[tris]                   # (M,3,3)

    # edge lengths
    e0 = np.linalg.norm(tri_verts[:,1] - tri_verts[:,0], axis=1)
    e1 = np.linalg.norm(tri_verts[:,2] - tri_verts[:,1], axis=1)
    e2 = np.linalg.norm(tri_verts[:,0] - tri_verts[:,2], axis=1)
    longest_edge = np.maximum.reduce([e0, e1, e2])

    # areas
    cross_prod = np.cross(tri_verts[:,1] - tri_verts[:,0],
                          tri_verts[:,2] - tri_verts[:,0])
    areas = 0.5 * np.linalg.norm(cross_prod, axis=1)

    edge_thr = np.percentile(longest_edge, longest_edge_pct)
    area_thr = np.percentile(areas, area_pct)

    bad = np.where((longest_edge > edge_thr) | (areas > area_thr) | (longest_edge > edge_length_thresh))[0].tolist()
    mesh.remove_triangles_by_index(bad)

    # 3) final cleanup
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()

    # 4) warn if still self-intersecting
    if hasattr(mesh, "is_self_intersecting") and mesh.is_self_intersecting():
        print("Warning: mesh still reports self-intersections after cleanup.")

    return mesh

### Logic Functions ###
    
def generate_winding_pointclouds(mesh_path,
                               display=False,
                               display_downsample=0.5,
                               color_by_angle=False):
    vertices, (has_points, angles) = load_mesh(mesh_path)
    angles = np.reshape(angles, (-1, 1))
    print(f"Shape of vertices: {vertices.shape} and of angles: {angles.shape}")
    points_mesh = np.concatenate((vertices, angles), axis=1)
    points_mesh = points_mesh[np.logical_not(has_points)]
    # Append source flag: 1 for mesh points
    mesh_flags = np.ones((points_mesh.shape[0], 1), dtype=points_mesh.dtype)
    points_mesh = np.concatenate((points_mesh, mesh_flags), axis=1)
    print(f"Shape of no has points points_mesh: {points_mesh.shape}")

    base_path = os.path.dirname(mesh_path)
    glob_selected_points_paths = os.path.join(base_path, "points_selected_*.npz")
    print(f"Searching for selected points in {glob_selected_points_paths}")
    slabs = sorted(glob.glob(glob_selected_points_paths))
    print("Preparing the winding pointclouds...")
    for slab_path in tqdm(slabs, desc="Preprocessing slabs"):
        points = load_pointcloud_slab(slab_path).astype(np.float16)
        # From TA to original coordinates
        points = shuffling_points_axis(points)
        points[:, :3] = points[:, :3] * 4.0 - 500
        # To original winding angle
        points[:, 3] = points[:, 3] + 90
        # Append source flag: 0 for slab points
        slab_flags = np.zeros((points.shape[0], 1), dtype=points.dtype)
        points = np.concatenate((points, slab_flags), axis=1)

        print(f"Shape of points: {points.shape} and of points_mesh: {points_mesh.shape}")
        if points.shape[0] == 0:
            print("Slab has no points ...")
            points = points_mesh
        else:
            points = np.concatenate((points, points_mesh), axis=0)
        min_angle = np.min(points[:, 3])
        max_angle = np.max(points[:, 3])
        min_winding = int(np.floor(min_angle / 360.0))
        max_winding = int(np.ceil(max_angle / 360.0))
        for winding_nr in range(min_winding, max_winding + 1):
            winding_angle = winding_nr * 360.0
            winding_points = points[np.logical_and(points[:, 3] >= winding_angle, points[:, 3] < winding_angle + 360.0)]
            if len(winding_points) > 0:
                save_winding_pointcloud(os.path.join(base_path, "windings"), winding_nr, winding_points)
                if display:
                    print(f"Displaying winding {winding_nr} (loaded {winding_points.shape[0]} points)")
                    display_3d_pointcloud(winding_points,
                                         color_by_angle=color_by_angle,
                                         downsample_ratio=display_downsample)

def build_neighbor_graph(coords,
                         angles=None,
                         k_neighbors=6,
                         angle_threshold=40,
                         angle_weight=None,
                         z_weight=100.0,
                         max_distance=10000,
                         angle_bin_size=12.0,
                         angle_bin_k=2,
                         z_bin_size=20.0,
                         z_bin_k=0,
                         is_mesh=None,
                         # random extra connections for mesh points
                         random_angle_thresh=10.0,
                         random_z_thresh=100.0,
                         random_k=4):  # number of random extra edges per mesh point
    """
    Build undirected neighbor lists and distances using KD-tree on spatial (and optionally angular) data.

    Args:
        coords (array-like, Nx3): Spatial coordinates.
        angles (array-like, N), optional: Angular values for each point.
        k_neighbors (int): Number of nearest neighbors to query (per KD-tree).
        angle_threshold (float), optional: Maximum allowed angle difference to keep an edge.
        angle_weight (float), optional: Scaling factor for angles when building first KD-tree.
        z_weight (float): Scaling factor for Z dimension in second KD-tree.
        max_distance (float): Maximum spatial distance to keep an edge.
        angle_bin_size (float): Angular bin size (degrees) for cross-bin connections.
        angle_bin_k (int): Number of neighbors to connect in each adjacent angular bin.
        z_bin_size (float): Z bin size for cross-bin connections.
        z_bin_k (int): Number of neighbors to connect in each adjacent z bin.

    Returns:
        neighbor_lists (List[List[int]]): Neighbor indices for each point.
        distance_lists (List[List[float]]): Corresponding distances.
    """
    pts = np.asarray(coords, dtype=float)
    # prepare angle values for thresholding
    angle_vals = np.asarray(angles, dtype=float) if angles is not None else None
    # Prepare is_mesh flags
    n = pts.shape[0]
    if is_mesh is None:
        is_mesh = np.zeros(n, dtype=bool)
    else:
        is_mesh = np.asarray(is_mesh, dtype=bool)
        if is_mesh.shape[0] != n:
            raise ValueError(f"is_mesh must have length {n}, got {is_mesh.shape[0]}")
    # First KD-tree: spatial + optional angular (query from all points)
    if angle_weight is not None and angles is not None:
        angs = np.asarray(angles, dtype=float).reshape(-1, 1) * angle_weight
        tree_input = np.hstack((pts, angs))
    else:
        tree_input = pts
    tree1 = cKDTree(tree_input)
    _, idx1 = tree1.query(tree_input, k=k_neighbors + 1)
    
    # Combine neighbor indices (single KD-tree)
    combined = idx1
    # Compute true distances based on spatial coords
    dists = np.linalg.norm(pts[combined] - pts[:, np.newaxis], axis=-1)
    # Drop self (first column)
    neigh_ids = combined[:, 1:]
    neigh_dists = dists[:, 1:]
    # Apply angle threshold if given
    if angle_threshold is not None and angles is not None:
        angs0 = np.asarray(angles, dtype=float)
        ref = angs0.reshape(-1, 1)
        nbrs = angs0[neigh_ids]
        diff = np.abs(ref - nbrs)
        mask = diff > angle_threshold
        neigh_ids[mask] = -1
        neigh_dists[mask] = np.inf
    n = pts.shape[0]
    neighbor_lists = [[] for _ in range(n)]
    distance_lists = [[] for _ in range(n)]
    # Build undirected edges (from all points)
    for i in range(n):
        valid = neigh_ids[i] != -1
        nb = neigh_ids[i][valid]
        db = neigh_dists[i][valid]
        neighbor_lists[i].extend(nb.tolist())
        distance_lists[i].extend(db.tolist())
        for j, d in zip(nb, db):
            neighbor_lists[j].append(i)
            distance_lists[j].append(d)
    # Additional cross-bin connectivity: angle bins and z bins (REACTIVATED)
    # Angle-based cross connectivity: connect nearest neighbors in adjacent angle bins
    if angles is not None and angle_bin_size > 0 and angle_bin_k > 0:
        print(f"Building angle-based cross-bin connectivity (bin_size={angle_bin_size}, k={angle_bin_k})...")
        bin_idx = np.floor(angle_vals / angle_bin_size).astype(int)
        bins = {}
        for idx, b in enumerate(bin_idx):
            bins.setdefault(b, []).append(idx)
        print(f"Created {len(bins)} angle bins")
        
        # build KD-tree per angle bin
        bin_trees = {b: cKDTree(pts[ids]) for b, ids in bins.items() if len(ids) >= 1}
        connections_added = 0
        
        mesh_indices = np.nonzero(is_mesh)[0]
        print(f"Processing {len(mesh_indices)} mesh points for angle-based connectivity...")
        
        for i in mesh_indices:
            b = bin_idx[i]
            # connect across two adjacent angular bins on each side
            for b2 in (b - 2, b - 1, b + 1, b + 2):
                if b2 in bin_trees:
                    ids = bins[b2]
                    tree_b = bin_trees[b2]
                    k = min(angle_bin_k, len(ids))
                    dists_b, idxs_b = tree_b.query(pts[i], k=k)
                    if k == 1:
                        dists_b = [dists_b]
                        idxs_b = [idxs_b]
                    for local_j in range(len(idxs_b)):
                        lj = idxs_b[local_j]
                        j = ids[lj]
                        # angle threshold filter
                        if angle_threshold is not None and angle_vals is not None:
                            if abs(angle_vals[i] - angle_vals[j]) > angle_threshold:
                                continue
                        d = np.linalg.norm(pts[i] - pts[j])
                        neighbor_lists[i].append(j)
                        distance_lists[i].append(d)
                        neighbor_lists[j].append(i)
                        distance_lists[j].append(d)
                        connections_added += 1
        print(f"Added {connections_added} angle-based cross-bin connections")
        
    # Z-based cross connectivity: connect nearest neighbors in adjacent z bins
    if z_bin_size > 0 and z_bin_k > 0:
        print(f"Building Z-based cross-bin connectivity (bin_size={z_bin_size}, k={z_bin_k})...")
        z_vals = pts[:, 2]
        z_idx = np.floor(z_vals / z_bin_size).astype(int)
        zbins = {}
        for idx, zb in enumerate(z_idx):
            zbins.setdefault(zb, []).append(idx)
        print(f"Created {len(zbins)} Z bins")
        
        z_trees = {zb: cKDTree(pts[ids]) for zb, ids in zbins.items() if len(ids) >= 1}
        z_connections_added = 0
        
        mesh_indices = np.nonzero(is_mesh)[0]
        print(f"Processing {len(mesh_indices)} mesh points for Z-based connectivity...")
        
        for i in mesh_indices:
            zb = z_idx[i]
            for zb2 in (zb - 1, zb + 1):
                if zb2 in z_trees:
                    ids = zbins[zb2]
                    tree_z = z_trees[zb2]
                    k = min(z_bin_k, len(ids))
                    d_z, idxs_z = tree_z.query(pts[i], k=k)
                    if k == 1:
                        d_z = [d_z]
                        idxs_z = [idxs_z]
                    for local_j in range(len(idxs_z)):
                        lj = idxs_z[local_j]
                        j = ids[lj]
                        # angle threshold filter
                        if angle_threshold is not None and angle_vals is not None:
                            if abs(angle_vals[i] - angle_vals[j]) > angle_threshold:
                                continue
                        d = np.linalg.norm(pts[i] - pts[j])
                        neighbor_lists[i].append(j)
                        distance_lists[i].append(d)
                        neighbor_lists[j].append(i)
                        distance_lists[j].append(d)
                        z_connections_added += 1
        print(f"Added {z_connections_added} Z-based cross-bin connections")
    # Random additional connectivity around mesh points - DEACTIVATED
    # if angle_vals is not None:
    #     # for each mesh point, connect to random neighbors within angle/z window
    #     mesh_indices = np.nonzero(is_mesh)[0]
    #     for i in mesh_indices:
    #         # select by angle and z proximity
    #         dz = np.abs(pts[:, 2] - pts[i, 2])
    #         da = np.abs(angle_vals - angle_vals[i])
    #         mask = (dz <= random_z_thresh) & (da <= random_angle_thresh)
    #         mask[i] = False
    #         eligible = np.nonzero(mask)[0]
    #         if eligible.size > 0:
    #             if eligible.size > random_k:
    #                 chosen = np.random.choice(eligible, size=random_k, replace=False)
    #             else:
    #                 chosen = eligible
    #             for j in chosen:
    #                 d = np.linalg.norm(pts[i] - pts[j])
    #                 neighbor_lists[i].append(int(j))
    #                 distance_lists[i].append(float(d))
    #                 neighbor_lists[j].append(int(i))
    #                 distance_lists[j].append(float(d))
    # Ensure uniqueness, preserve all edges
    for i in range(n):
        ids = neighbor_lists[i]
        ds = distance_lists[i]
        # unique preserving order
        _, uniq_idx = np.unique(ids, return_index=True)
        ids = [ids[k] for k in sorted(uniq_idx)]
        ds = [ds[k] for k in sorted(uniq_idx)]
        neighbor_lists[i] = ids
        distance_lists[i] = ds
    # Distance filtering removed as requested
    # max_spatial_dist = 100.0
    # print(f"Filtering edges by distance (max={max_spatial_dist})...")
    # total_edges_before = sum(len(neighbors) for neighbors in neighbor_lists)
    # 
    # for i in range(n):
    #     ids = neighbor_lists[i]
    #     ds = distance_lists[i]
    #     filtered_ids = []
    #     filtered_ds = []
    #     for j, d in zip(ids, ds):
    #         if d <= max_spatial_dist:
    #             filtered_ids.append(j)
    #             filtered_ds.append(d)
    #     neighbor_lists[i] = filtered_ids
    #     distance_lists[i] = filtered_ds
    # 
    # total_edges_after = sum(len(neighbors) for neighbors in neighbor_lists)
    # print(f"Distance filtering: {total_edges_before} -> {total_edges_after} edges (removed {total_edges_before - total_edges_after})")
    return neighbor_lists, distance_lists

def create_flattening_downsample_indices(points, is_mesh_flags, downsample_ratio):
    """
    Create downsampled indices for flattening solver that keeps all mesh points 
    and a fraction of slab points.
    
    Args:
        points: array of points
        is_mesh_flags: boolean array indicating mesh points
        downsample_ratio: fraction of slab points to keep
        
    Returns:
        indices: array of indices for downsampled points
    """
    mesh_indices = np.where(is_mesh_flags)[0]
    slab_indices = np.where(~is_mesh_flags)[0]
    
    # Keep all mesh points
    kept_mesh = mesh_indices
    
    # Downsample slab points
    if len(slab_indices) > 0 and downsample_ratio < 1.0:
        n_slab_keep = int(len(slab_indices) * downsample_ratio)
        if n_slab_keep > 0:
            kept_slab = np.random.choice(slab_indices, n_slab_keep, replace=False)
        else:
            kept_slab = np.array([], dtype=int)
    else:
        kept_slab = slab_indices
    
    # Combine and sort
    all_kept = np.concatenate([kept_mesh, kept_slab])
    all_kept = np.sort(all_kept)
    
    print(f"Downsampling for solver: kept {len(kept_mesh)} mesh + {len(kept_slab)} slab = {len(all_kept)} total points")
    
    return all_kept

def flatten_pointcloud(base_path,
                       k_neighbors=10,
                       angle_threshold=40,
                       angle_weight=0.2,
                       downsample_ratio=0.1,
                       flattening_downsample_ratio=0.3,
                       display=False,
                       display_downsample=0.1,
                       color_by_angle=False):
    """
    Flatten pointcloud by concatenating winding segments and building neighbor graphs.

    Optionally, include the 4th dimension (angle) in neighbor search by providing `angle_weight`,
    or filter out neighbors whose absolute angle difference exceeds `angle_threshold`.

    Args:
        base_path (str): Path to base directory containing 'windings'.
        k_neighbors (int): Number of nearest neighbors per point (excluding self).
        winding_width (int): Number of windings concatenated per graph segment.
        angle_threshold (float, optional): Max allowed difference in 4th-dimension (angle) to keep a neighbor.
        angle_weight (float, optional): Scale factor for 4th-dimension when computing distances in KD-tree.
        downsample_ratio (float, optional): Fraction of points to randomly keep when loading windings (default 0.1).
        flattening_downsample_ratio (float, optional): Fraction of points to use in first step of flattening solver (default 0.3).
    """
    winding_width=3
    winding_path = os.path.join(base_path, "windings")
    flattened_winding_path = os.path.join(base_path, "windings_flattened")
    # find and order all winding pointclouds
    winding_files = glob.glob(os.path.join(winding_path, "winding_*.npz"))
    print(f"Found {len(winding_files)} winding pointclouds.")
    # Bring the files in order of their winding number
    winding_files_indices = sorted([int(os.path.basename(wf).split("_")[1].split(".")[0]) for wf in winding_files])
    winding_files = [os.path.join(winding_path, f"winding_{i}.npz") for i in winding_files_indices]

    # Detect first and last windings with mesh vertices
    print("Scanning windings to find those with mesh vertices...")
    windings_with_mesh = []
    for i, winding_idx in enumerate(winding_files_indices):
        try:
            points = load_winding_pointcloud(winding_path, winding_idx)
            if points.shape[1] >= 5:  # Check if mesh flags exist (5th dimension)
                is_mesh_flags = points[:, 4].astype(bool)
                if np.any(is_mesh_flags):  # If any mesh vertices exist
                    windings_with_mesh.append(winding_idx)
                    print(f"  Winding {winding_idx}: {np.sum(is_mesh_flags)} mesh vertices found")
        except Exception as e:
            print(f"  Warning: Could not load winding {winding_idx}: {e}")
            continue
    
    if len(windings_with_mesh) == 0:
        print("No windings with mesh vertices found. Exiting.")
        return flattened_winding_path
    
    first_mesh_winding = windings_with_mesh[0]
    last_mesh_winding = windings_with_mesh[-1]
    print(f"Will process windings {first_mesh_winding} to {last_mesh_winding} (contains mesh vertices)")
    
    # Find the start and end indices in the winding_files_indices array
    first_mesh_idx = winding_files_indices.index(first_mesh_winding)
    last_mesh_idx = winding_files_indices.index(last_mesh_winding)

    # initialize storage for previous flattened coordinates per winding
    prev_uvs_u = {}
    prev_uvs_v = {}
    prev_points = {}

    # process windings in segments
    winding_us = None
    winding_vs = None
    for start in range(0, len(winding_files)):
        # if start < 27: # for debug, leave it for now
        #     continue
        if start > first_mesh_idx + 2: # for debug, leave it for now
            continue
        
        # Skip if before first mesh winding or after last mesh winding
        if start < first_mesh_idx:
            print(f"Skipping winding {start} (before first mesh winding)")
            continue
        if start > last_mesh_idx:
            print(f"Finished processing: reached winding {start} (past last mesh winding)")
            break
            
        end = min(start + winding_width, len(winding_files))
        print(f"Processing windings {start} to {end}: {winding_files[start:end]}")
        # load and concatenate pointcloud segments (points include x,y,z,angle)
        # pcs = [np.load(wf)['points'] if  for wf in winding_files[start:end]]
        pcs = []
        for wnr in winding_files_indices[start:end]:
            # load previous points if available
            if wnr in prev_points:
                print(f"Loading previous points for winding {wnr}")
                points_ = prev_points[wnr]
            else:
                print(f"Loading winding pointcloud {wnr} from {winding_path}")
                points_ = load_winding_pointcloud(winding_path, wnr)
                # trim z-range: remove 50 units from top and bottom
                z_vals = points_[:, 2]
                z_min, z_max = z_vals.min(), z_vals.max()
                z_low, z_high = z_min + 50.0, z_max - 50.0
                mask_z = (z_vals >= z_low) & (z_vals <= z_high)
                before_n = points_.shape[0]
                points_ = points_[mask_z]
                after_n = points_.shape[0]
                print(f"Trimmed wrap {wnr}: z [{z_min:.2f},{z_max:.2f}] -> keeping [{z_low:.2f},{z_high:.2f}], {after_n}/{before_n} points")
                # random downsampling of newly loaded points
                if downsample_ratio < 1.0:
                    mask = np.random.rand(points_.shape[0]) < downsample_ratio
                    points_ = points_[mask]
                if display:
                    # Display 3D pointcloud, optionally colored by angle
                    display_3d_pointcloud(points_, color_by_angle=color_by_angle,
                                        downsample_ratio=display_downsample)
            pcs.append(np.array(points_))
        try:
            min_angles = [np.min(pc[:, 3]) for pc in pcs]
            max_angles = [np.max(pc[:, 3]) for pc in pcs]
        except Exception as e:
            print(f"Error while computing min/max angles: {e}")
            print(F"Shape of pcs: {[pc.shape for pc in pcs]}")
            continue
        winding_indices = [len(pc) for pc in pcs]
        points = np.concatenate(pcs, axis=0)
        coords = points[:, :3]
        winding_angles = points[:, 3]
        # build neighbor graph using KD-tree
        # Extract mesh flags for points (5th dimension)
        if points.shape[1] >= 5:
            is_mesh_flags = points[:, 4].astype(bool)
        else:
            is_mesh_flags = np.zeros(points.shape[0], dtype=bool)
        neighbor_lists, distance_lists = build_neighbor_graph(
            coords,
            angles=winding_angles,
            k_neighbors=k_neighbors,
            angle_threshold=angle_threshold,
            angle_weight=angle_weight,
            is_mesh=is_mesh_flags
        )

        # Check that each edge is undirected
        # for i in range(len(coords)):
        #     for j in range(len(neighbor_lists[i])):
        #         neighbor = neighbor_lists[i][j]
        #         if i not in neighbor_lists[neighbor]:
        #             print(f"Edge {i} -> {neighbor} is not undirected")
        #             break
        
        print(f"Min/Max distances: {np.min(np.concatenate(distance_lists))}, {np.max(np.concatenate(distance_lists))}")

        # Initialize UV coordinates using previous results or defaults
        coords_z_index = 2
        current_angles = np.zeros_like(winding_angles)
        current_z = np.zeros_like(coords[:, coords_z_index])
        
        # Assign previous results for wraps if available, otherwise use initial values
        start_idx = 0
        prev_ranges = []
        new_ranges = []
        for idx_in_segment, count in enumerate(winding_indices):
            wrap_nr = winding_files_indices[start + idx_in_segment]
            # default slices
            default_u = winding_angles[start_idx:start_idx+count]
            default_v = coords[start_idx:start_idx+count, coords_z_index]
            prev_u = prev_uvs_u.get(wrap_nr)
            # use previous if exists and matches count
            if prev_u is not None and prev_u.shape[0] == count:
                current_angles[start_idx:start_idx+count] = prev_u
                current_z[start_idx:start_idx+count] = prev_uvs_v[wrap_nr]
                prev_ranges.append((start_idx, start_idx+count))
            else:
                if prev_u is not None and prev_u.shape[0] != count:
                    print(f"Warning: wrap {wrap_nr} prev_uv length {prev_u.shape[0]} != expected {count}, using defaults.")
                current_angles[start_idx:start_idx+count] = default_u
                current_z[start_idx:start_idx+count] = default_v
                new_ranges.append((start_idx, start_idx+count))
            start_idx += count
        
        # Align newly added wraps to previously computed wraps in u (angle) and v (z)
        if prev_ranges and new_ranges:
            # align u: offset new wraps' min to prev wraps' max
            prev_vals = np.concatenate([current_angles[s:e] for s, e in prev_ranges])
            new_vals = np.concatenate([current_angles[s:e] for s, e in new_ranges])
            u_offset = np.max(prev_vals) - np.min(new_vals)
            for s, e in new_ranges:
                current_angles[s:e] += u_offset
            # align v (z): match mean of new wraps to mean of prev wraps
            prev_z_vals = np.concatenate([current_z[s:e] for s, e in prev_ranges])
            new_z_vals = np.concatenate([current_z[s:e] for s, e in new_ranges])
            z_offset = np.mean(prev_z_vals) - np.mean(new_z_vals)
            for s, e in new_ranges:
                current_z[s:e] += z_offset

        # STEP 1: Create downsampled data for initial solve
        print("=== STEP 1: Creating downsampled data for initial solve ===")
        downsample_indices = create_flattening_downsample_indices(points, is_mesh_flags, flattening_downsample_ratio)
        
        # Create downsampled arrays
        coords_ds = coords[downsample_indices]
        winding_angles_ds = winding_angles[downsample_indices]
        is_mesh_flags_ds = is_mesh_flags[downsample_indices]
        current_angles_ds = current_angles[downsample_indices]
        current_z_ds = current_z[downsample_indices]
        
        # Build neighbor graph for downsampled points
        neighbor_lists_ds, distance_lists_ds = build_neighbor_graph(
            coords_ds,
            angles=winding_angles_ds,
            k_neighbors=k_neighbors,
            angle_threshold=angle_threshold,
            angle_weight=angle_weight,
            is_mesh=is_mesh_flags_ds
        )
        
        print(f"Downsampled Min/Max distances: {np.min(np.concatenate(distance_lists_ds))}, {np.max(np.concatenate(distance_lists_ds))}")
        
        # Initialize downsampled solver
        solver_ds = graph_problem_gpu_py.Solver(
            neighbor_lists_ds,
            distance_lists_ds,
            winding_angles_ds,
            current_angles_ds,
            coords_ds[:, coords_z_index],
            current_z_ds,
            coords_ds[:, 0],  # x coordinates
            coords_ds[:, 1],  # y coordinates  
            coords_ds[:, 2]   # z coordinates
        )
        print("Set up downsampled graph")
        
        # Fix mesh points and first winding in downsampled solver
        mesh_idx_ds = np.nonzero(is_mesh_flags_ds)[0].tolist()
        # if mesh_idx_ds:
        #     solver_ds.fix_nodes(mesh_idx_ds)
        #     print(f"Fixed {len(mesh_idx_ds)} mesh-point nodes in downsampled solver.")
        
        # Fix nodes of the first winding if it has been previously computed (in downsampled space)
        start_wrap_nr = winding_files_indices[start]
        first_count = winding_indices[0]
        prev_u0 = prev_uvs_u.get(start_wrap_nr)
        
        # Find which downsampled indices correspond to the first winding
        first_winding_mask = np.zeros(len(downsample_indices), dtype=bool)
        first_winding_mask[downsample_indices < first_count] = True
        first_winding_ds_indices = np.where(first_winding_mask)[0]
        
        if prev_u0 is not None and len(first_winding_ds_indices) > 0:
            # Convert to relative indices for the downsampled solver
            ds_valid_indices = np.array(solver_ds.get_undeleted_indices())
            first_winding_relative = []
            for abs_idx in first_winding_ds_indices:
                where_found = np.where(ds_valid_indices == abs_idx)[0]
                if len(where_found) > 0:
                    first_winding_relative.append(where_found[0])
            
            if len(first_winding_relative) > 0:
                solver_ds.fix_nodes(first_winding_relative)
                print(f"Fixed {len(first_winding_relative)} downsampled nodes of winding {start_wrap_nr} using previous results.")
            else:
                print(f"Warning: No valid first winding nodes found in downsampled solver")
        
        # Solve downsampled graph problem exactly like the original
        min_angle = np.min(winding_angles_ds)
        max_angle = np.max(winding_angles_ds)
        z_min = np.min(coords_ds[:, coords_z_index])
        z_max = np.max(coords_ds[:, coords_z_index])
        print(f"Downsampled Min/Max angles: {min_angle}, {max_angle}, Min/Max z: {z_min}, {z_max}")
        
        zero_ranges = [(min_angle, min_angle+270), (max_angle-270, max_angle)]
        a_step = (max_angle - min_angle)/winding_width
        zero_ranges_initial = [(min_angle + index * a_step, min_angle + (index+1)*a_step) for index in range(winding_width)]
        
        print("Solving downsampled graph (first pass - original style)...")
        solver_ds.solve_flattening(num_iterations=20000, visualize=True, 
                                   angle_tug_min=min_angle+90, angle_tug_max=max_angle-90, 
                                   z_tug_min=z_min+100, z_tug_max=z_max-100, 
                                   tug_step=0.25, zero_ranges=zero_ranges_initial)
        
        uvs_ds_initial = np.array(solver_ds.get_uvs())
        print(f"After first downsampled solve: u min={uvs_ds_initial[:,0].min()}, u max={uvs_ds_initial[:,0].max()}, v min={uvs_ds_initial[:,1].min()}, v max={uvs_ds_initial[:,1].max()}")
        
        print("Solving downsampled graph (second pass - original style)...")
        solver_ds.solve_flattening(num_iterations=150000, visualize=True, zero_ranges=zero_ranges, tug_step=0.0005)
        
        uvs_ds_final = np.array(solver_ds.get_uvs())
        undeleted_ds_indices = np.array(solver_ds.get_undeleted_indices())
        print(f"After second downsampled solve: u min={uvs_ds_final[:,0].min()}, u max={uvs_ds_final[:,0].max()}, v min={uvs_ds_final[:,1].min()}, v max={uvs_ds_final[:,1].max()}")
        print(f"Downsampled solver: {len(undeleted_ds_indices)} undeleted out of {len(downsample_indices)} total downsampled points")
        
        # STEP 2: Use downsampled results to initialize full solver
        print("=== STEP 2: Setting up full solver with downsampled results ===")
        
        # Interpolate downsampled results to all points
        current_angles_full = np.copy(current_angles)
        current_z_full = np.copy(current_z)
        
        # Update positions of undeleted downsampled points with solved values
        # Map the undeleted indices back to the original full point cloud indices
        valid_downsample_indices = downsample_indices[undeleted_ds_indices]
        current_angles_full[valid_downsample_indices] = uvs_ds_final[:, 0]
        current_z_full[valid_downsample_indices] = uvs_ds_final[:, 1]
        print(f"Updated {len(valid_downsample_indices)} points with downsampled solve results")
        
        if display:
            # DEBUG: Visualize the fixed points UV coordinates from downsampled solve
            print("=== DEBUG: Visualizing fixed points from downsampled solve ===")
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot 1: All downsampled solution points
            ax1.scatter(uvs_ds_final[:, 0], uvs_ds_final[:, 1], s=1, alpha=0.6, c='blue')
            ax1.set_title(f'Downsampled Solution Points\n({len(uvs_ds_final)} points)')
            ax1.set_xlabel('U (angle)')
            ax1.set_ylabel('V (z)')
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Fixed points in full coordinate space
            fixed_u = current_angles_full[valid_downsample_indices]
            fixed_v = current_z_full[valid_downsample_indices]
            ax2.scatter(fixed_u, fixed_v, s=1, alpha=0.6, c='red')
            ax2.set_title(f'Fixed Points in Full Space\n({len(fixed_u)} points)')
            ax2.set_xlabel('U (angle)')
            ax2.set_ylabel('V (z)')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            print(f"Fixed points UV range: U=[{fixed_u.min():.2f}, {fixed_u.max():.2f}], V=[{fixed_v.min():.2f}, {fixed_v.max():.2f}]")
        
        # Initialize full solver with interpolated values
        solver = graph_problem_gpu_py.Solver(
            neighbor_lists,
            distance_lists,
            winding_angles,
            current_angles_full,
            coords[:, coords_z_index],
            current_z_full,
            coords[:, 0],  # x coordinates
            coords[:, 1],  # y coordinates
            coords[:, 2]   # z coordinates
        )
        print("Set up full graph with downsampled initialization")
        
        # Get the valid indices from the full solver to map absolute indices to relative ones
        full_valid_indices = np.array(solver.get_undeleted_indices())
        
        # Convert absolute valid_downsample_indices to relative indices within the full solver's valid nodes
        relative_downsample_indices = []
        missing_count = 0
        for abs_idx in valid_downsample_indices:
            # Find where this absolute index appears in the full solver's valid indices
            where_found = np.where(full_valid_indices == abs_idx)[0]
            if len(where_found) > 0:
                relative_downsample_indices.append(where_found[0])
            else:
                missing_count += 1
        
        print(f"Mapped {len(valid_downsample_indices)} absolute indices to {len(relative_downsample_indices)} relative downsample indices")
        if missing_count > 0:
            print(f"WARNING: {missing_count} downsampled points not found in full solver valid indices!")
        
        # Also fix the first winding nodes in the full solver
        first_winding_relative_full = []
        first_winding_missing = 0
        if prev_u0 is not None and prev_u0.shape[0] == first_count:
            wrap0_idx = list(range(first_count))
            print(f"DEBUG: first_count = {first_count}, wrap0_idx (first 10): {wrap0_idx[:10]}")
            # Convert to relative indices for the full solver
            for abs_idx in wrap0_idx:
                where_found = np.where(full_valid_indices == abs_idx)[0]
                if len(where_found) > 0:
                    first_winding_relative_full.append(where_found[0])
                else:
                    first_winding_missing += 1
        
        if first_winding_missing > 0:
            print(f"WARNING: {first_winding_missing} first winding points not found in full solver valid indices!")
        
        # Combine downsampled and first winding indices for fixing
        all_fixed_indices = list(set(relative_downsample_indices + first_winding_relative_full))
        
        # Fix both the downsampled points and first winding points in the full solver
        if len(all_fixed_indices) > 0:
            solver.fix_nodes(all_fixed_indices)
            print(f"Fixed {len(relative_downsample_indices)} downsampled points + {len(first_winding_relative_full)} first winding points = {len(all_fixed_indices)} total fixed points in full solver")
        else:
            print("Warning: No valid points found to fix in full solver")
        
        # Solve full graph with just the second solve call (0.0005 tug step only)
        min_angle_full = np.min(winding_angles)
        max_angle_full = np.max(winding_angles)
        z_min_full = np.min(coords[:, coords_z_index])
        z_max_full = np.max(coords[:, coords_z_index])
        
        zero_ranges_full = [(min_angle_full, min_angle_full+270), (max_angle_full-270, max_angle_full)]
        
        print("Solving full graph (refinement with fixed anchors - 0.0005 tug only)...")
        solver.solve_flattening(num_iterations=10000, visualize=True)
        
        undeleted_indices = np.array(solver.get_undeleted_indices())
        uvs = np.array(solver.get_uvs())
        print(f"After full refinement solve: u min={uvs[:,0].min()}, u max={uvs[:,0].max()}, v min={uvs[:,1].min()}, v max={uvs[:,1].max()}")
        
        # Process results
        winding_us = []
        winding_vs = []
        w_i = 0
        w_i_total = 0
        for i in range(len(winding_indices)):
            undeleted_indices_winding = undeleted_indices[np.logical_and(undeleted_indices >= w_i_total, undeleted_indices < w_i_total + winding_indices[i])]
            winding_u = uvs[w_i:w_i + len(undeleted_indices_winding), 0]
            winding_v = uvs[w_i:w_i + len(undeleted_indices_winding), 1]
            points_winding = points[undeleted_indices_winding]

            # store for next segment initialization
            wrap_nr = winding_files_indices[start + i]
            prev_uvs_u[wrap_nr] = winding_u
            prev_uvs_v[wrap_nr] = winding_v
            prev_points[wrap_nr] = points_winding
            winding_us.append(winding_u)
            winding_vs.append(winding_v)
            w_i += len(undeleted_indices_winding)
            w_i_total += winding_indices[i]

        print(f"Keys in prev_uvs_u: {prev_uvs_u.keys()}")
        print(f"Keys in prev_uvs_v: {prev_uvs_v.keys()}")

        # save and free up memory of already computed windings
        clean_winding_dicts((start, end), flattened_winding_path, winding_files_indices, prev_uvs_u, prev_uvs_v, prev_points)

    # save the last segments
    clean_winding_dicts((0, 0), flattened_winding_path, winding_files_indices, prev_uvs_u, prev_uvs_v, prev_points)
    return flattened_winding_path

def normalize_uvs(mesh, verbose=True):
    """
    Normalize UV coordinates to 0-1 range, returning the natural image size.
    
    Args:
        mesh: Open3D TriangleMesh object with triangle_uvs
        verbose: bool, whether to print progress info
    
    Returns:
        tuple: (mesh with normalized UVs, natural_image_size as (width, height))
    """
    if not hasattr(mesh, 'triangle_uvs') or len(mesh.triangle_uvs) == 0:
        if verbose:
            print("Warning: Mesh has no UV coordinates, skipping UV normalization")
        return mesh, (2048, 2048)  # Default fallback size
    
    # Get UV coordinates
    uvs = np.asarray(mesh.triangle_uvs)
    
    if verbose:
        print(f"Normalizing {len(uvs)} UV coordinates to 0-1 range")
        print(f"Original UV range: U=[{uvs[:,0].min():.2f}, {uvs[:,0].max():.2f}], V=[{uvs[:,1].min():.2f}, {uvs[:,1].max():.2f}]")
    
    # First: move min to (0,0)
    uv_min = uvs.min(axis=0)
    uvs_shifted = uvs - uv_min
    
    # At this point, the max values represent the natural image size
    natural_max = uvs_shifted.max(axis=0)
    natural_image_size = (int(natural_max[0]), int(natural_max[1]))
    
    if verbose:
        print(f"After shifting min to (0,0): U=[0.00, {natural_max[0]:.2f}], V=[0.00, {natural_max[1]:.2f}]")
        print(f"Natural image size: {natural_image_size}")
    
    # Then: normalize to 0-1 range
    uv_range = natural_max.copy()
    uv_range[uv_range == 0] = 1.0  # Avoid division by zero
    
    normalized_uvs = uvs_shifted / uv_range
    
    if verbose:
        print(f"Normalized UV range: U=[{normalized_uvs[:,0].min():.2f}, {normalized_uvs[:,0].max():.2f}], V=[{normalized_uvs[:,1].min():.2f}, {normalized_uvs[:,1].max():.2f}]")
    
    # Update mesh with normalized UVs
    mesh.triangle_uvs = o3d.utility.Vector2dVector(normalized_uvs)
    
    return mesh, natural_image_size

def create_mask(mesh, natural_image_size, output_path=None, verbose=True):
    """
    Create a mask from mesh UV coordinates (0-1 range) using natural image size.
    
    Args:
        mesh: Open3D TriangleMesh object with triangle_uvs in 0-1 range
        natural_image_size: tuple, (width, height) natural size from normalization
        output_path: str or None, output file path for the mesh (mask will be saved alongside if provided)
        verbose: bool, whether to print progress info
    
    Returns:
        tuple: (mask_array, mask_path) where mask_path is None if no file was saved
    """
    import cv2
    
    if not hasattr(mesh, 'triangle_uvs') or len(mesh.triangle_uvs) == 0:
        if verbose:
            print("Warning: Mesh has no UV coordinates, skipping mask creation")
        return None, None
    
    # Get UV coordinates (should be in 0-1 range)
    uvs = np.asarray(mesh.triangle_uvs)
    
    if verbose:
        print(f"Creating mask from {len(uvs)} UV coordinates using natural size {natural_image_size}")
    
    # Create mask PNG with natural image size
    mask = np.zeros(natural_image_size[::-1], dtype=np.uint8)  # height, width
    
    # Convert 0-1 UVs to pixel coordinates using natural image size
    uv_pixels = uvs * np.array([natural_image_size[0] - 1, natural_image_size[1] - 1])
    uv_pixels = uv_pixels.astype(np.int32)
    
    # Group UV coordinates into triangles (every 3 consecutive UVs form a triangle)
    for i in range(0, len(uv_pixels), 3):
        if i + 2 < len(uv_pixels):
            triangle = uv_pixels[i:i+3]
            # Flip V coordinate for image coordinate system
            triangle[:, 1] = natural_image_size[1] - 1 - triangle[:, 1]
            try:
                cv2.fillPoly(mask, [triangle], 255)
            except:
                pass  # Skip invalid triangles
    
    # Save mask PNG only if output_path is provided
    mask_path = None
    if output_path is not None:
        mask_path = os.path.join(os.path.dirname(output_path), 
                                os.path.basename(output_path).split(".")[0] + "_mask.png")
        cv2.imwrite(mask_path, mask)
        
        if verbose:
            print(f"Created mask PNG: {mask_path} (size: {natural_image_size})")
    elif verbose:
        print(f"Created mask array (size: {natural_image_size}) - no file saved")
    
    return mask, mask_path

def save_mesh_with_uvs(mesh, output_path, natural_image_size, create_mask_png=True, verbose=True):
    """
    Save a mesh with UV coordinates (should already be normalized to 0-1 range).
    
    Args:
        mesh: Open3D TriangleMesh object with triangle_uvs already normalized to 0-1
        output_path: str, output file path (should end in .obj)
        natural_image_size: tuple, (width, height) natural size from normalization
        create_mask_png: bool, whether to create a mask PNG alongside the mesh
        verbose: bool, whether to print progress info
    
    Returns:
        bool: True if save was successful, False otherwise
    """
    if verbose:
        print(f"Saving mesh with {len(mesh.vertices)} vertices, {len(mesh.triangles)} faces to {output_path}")
    
    # Create mask if requested
    if create_mask_png:
        mask_array, mask_path = create_mask(mesh, natural_image_size, output_path, verbose)
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save mesh
    success = o3d.io.write_triangle_mesh(output_path, mesh)
    if success and verbose:
        print(f"Successfully saved mesh to {output_path}")
    elif not success:
        print(f"ERROR: Failed to save mesh to {output_path}")
    
    return success

def grid_uv_flattened(flattened_winding_path,
                       subsample_radius=3.0,
                       display=False,
                       display_downsample=0.1,
                       color_by_angle=False):
    winding_filtered_path = os.path.join(os.path.dirname(flattened_winding_path), "windings_filtered")
    # find and order all winding pointclouds
    winding_files = glob.glob(os.path.join(flattened_winding_path, "flattened_winding_*.npz"))
    print(f"Found {len(winding_files)} flattened winding pointclouds for grid sampling the uv space.")
    # Bring the files in order of their winding number
    winding_files_indices = sorted([int(os.path.basename(wf).split("_")[2].split(".")[0]) for wf in winding_files])
    winding_files = [os.path.join(flattened_winding_path, f"flattened_winding_{i}.npz") for i in winding_files_indices]
    print(f"Will process {len(winding_files)} flattened winding pointclouds for grid sampling the uv space.")
    # Load each wrap, subsample
    for i in range(len(winding_files)):
        points, uvs = load_flattened_winding(flattened_winding_path, winding_files_indices[i])
        
        # Filter out mesh points (keep only slab points)
        if points.shape[1] >= 5:  # Check if mesh flags exist (5th dimension)
            is_mesh_flags = points[:, 4].astype(bool)
            slab_mask = ~is_mesh_flags  # Keep non-mesh points (slab points)
            points_before = points.shape[0]
            points = points[slab_mask]
            uvs = uvs[slab_mask]
            points_after = points.shape[0]
            print(f"Filtered out mesh points from {winding_files[i]}: {points_before} -> {points_after} points")
        else:
            print(f"Warning: No mesh flags found in {winding_files[i]}, keeping all points")
        
        # Skip processing if no points (check immediately after filtering)
        if points.shape[0] == 0 or uvs.shape[0] == 0:
            print(f"Skipping {winding_files[i]} - no points to process after filtering")
            save_flattened_winding(winding_filtered_path, winding_files_indices[i], points, uvs)
            continue
            
        save_pointcloud_winding(winding_filtered_path, winding_files_indices[i], points)
        # display_3d_pointcloud(points)
        # subsample
        print(f"Subsampling {winding_files[i]} with {points.shape[0]} points")
        
        print(f"Min max uvs: {np.min(uvs[:, 0])}, {np.max(uvs[:, 0])}, {np.min(uvs[:, 1])}, {np.max(uvs[:, 1])}")
        kept_indices = subsample_min_dist(uvs, subsample_radius)
        print(f"Subsampled {winding_files[i]} to {len(kept_indices)} points")
        # filter by density
        kept_indices = filter_by_density(uvs, kept_indices, 3 * subsample_radius, int((subsample_radius**0.5)*2))
        # kept_indices = filter_by_density_mad(uvs, kept_indices, 2 * subsample_radius, k=3.0) # automatic threshold finding
        points_subsampled = points[kept_indices]
        uvs_subsampled = uvs[kept_indices]
        print(f"Subsampled {winding_files[i]} to {points_subsampled.shape[0]} points")
        if display:
            # display  with matplotlib: 2D UV and point plots
            fig, axs = plt.subplots(2, 2, figsize=(10, 10))
            axs[0, 0].scatter(uvs[:, 0], uvs[:, 1], s=1)
            axs[0, 0].set_title("Original UVs")
            axs[0, 1].scatter(uvs_subsampled[:, 0], uvs_subsampled[:, 1], s=1)
            axs[0, 1].set_title("Subsampled UVs")
            axs[1, 0].scatter(points[:, 0], points[:, 1], s=1)
            axs[1, 0].set_title("Original Points")
            axs[1, 1].scatter(points_subsampled[:, 0], points_subsampled[:, 1], s=1)
            axs[1, 1].set_title("Subsampled Points")
            plt.show()
            # 3D display, optional coloring by angle
            display_3d_pointcloud(points_subsampled,
                                  color_by_angle=color_by_angle,
                                  downsample_ratio=display_downsample)

        kept_indices = refine_by_medoid(points, uvs, kept_indices, subsample_radius)
        points_subsampled = points[kept_indices]
        uvs_subsampled = uvs[kept_indices]
        if display:
            display_3d_pointcloud(points_subsampled,
                                  color_by_angle=color_by_angle,
                                  downsample_ratio=display_downsample)

        # kept_indices, errors = filter_by_edge_error(points[:,:3], uvs, kept_indices, 10 * subsample_radius, int(1.5*subsample_radius))
        kept_indices, errors = filter_by_edge_error_mad(points[:,:3], uvs, kept_indices, 10 * subsample_radius, k=5.0) # automatic threshold finding
        points_subsampled = points[kept_indices]
        uvs_subsampled = uvs[kept_indices]
        if display:
            display_3d_pointcloud(points_subsampled,
                                  color_by_angle=color_by_angle,
                                  downsample_ratio=display_downsample)

        save_flattened_winding(winding_filtered_path, winding_files_indices[i], points_subsampled, uvs_subsampled)

def mesh_uv_wraps(filtered_winding_path, output_dir):
    """
    For each filtered winding .npz in `filtered_winding_path`:
      1) Load points (Nx3+) and uvs (Nx2)
      2) Run Delaunay on the UVs → simplices (M×3)
      3) Build an Open3D TriangleMesh:
           - vertices = 3D points
           - triangles = simplices
           - triangle_uvs = flattened per-triangle UV coords
      4) Save as OBJ with UVs in `output_dir/wrap_<nr>.obj`

    Args:
        filtered_winding_path: str, folder containing
            `flattened_winding_<nr>.npz` with arrays `points, uvs`
        output_dir: str, where to write `wrap_<nr>.obj`
    """
    os.makedirs(output_dir, exist_ok=True)

    # grab all filtered wraps
    winding_flattened_files = glob.glob(os.path.join(filtered_winding_path, "flattened_winding_*.npz"))
    print(f"Found {len(winding_flattened_files)} flattened winding pointclouds to mesh uv wraps.")
    # Bring the files in order of their winding number
    winding_flattened_files_indices = sorted([int(os.path.basename(wf).split("_")[2].split(".")[0]) for wf in winding_flattened_files])
    winding_flattened_files = [os.path.join(filtered_winding_path, f"flattened_winding_{i}.npz") for i in winding_flattened_files_indices]
    for fp in winding_flattened_files:
        nr = int(os.path.basename(fp).split("_")[-1].split(".")[0])
        data = np.load(fp)
        pts3 = data["points"][:, :3].astype(np.float64)
        winding_angles = data["points"][:, 3].astype(np.float64)
        uvs2 = data["uvs"].astype(np.float64)

        # Skip if no points to mesh
        if len(pts3) == 0 or len(uvs2) == 0:
            print(f"Skipping wrap {nr} - no points to mesh")
            continue
            
        # Check if we have enough points for Delaunay triangulation
        if len(uvs2) < 3:
            print(f"Skipping wrap {nr} - insufficient points for triangulation: {len(uvs2)}")
            continue

        # 2D Delaunay in UV space
        tri = Delaunay(uvs2)
        faces = tri.simplices  # M×3 array of vertex-indices

        # filter out faces with min max winding angle diff > 10 deg
        winding_angles_faces = winding_angles[faces]
        min_angles = np.min(winding_angles_faces, axis=1)
        max_angles = np.max(winding_angles_faces, axis=1)
        winding_angle_diffs = max_angles - min_angles
        faces_mask = np.abs(winding_angle_diffs) < 50
        faces = faces[faces_mask]

        # build Open3D mesh
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(pts3)
        mesh.triangles = o3d.utility.Vector3iVector(faces)

        # Open3D expects triangle_uvs to be length 3*M
        # so we flatten face-by-face
        tri_uvs = uvs2[faces].reshape(-1, 2)
        mesh.triangle_uvs = o3d.utility.Vector2dVector(tri_uvs)

        # Compute normals
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()

        # mesh = clean_mesh(mesh, longest_edge_pct=100, area_pct=100, edge_length_thresh=1000)

        # Normalize UVs after cleaning and get natural image size
        mesh, natural_image_size = normalize_uvs(mesh, verbose=True)

        # Apply advanced mask-based filtering
        mesh = filter_mesh_by_mask(mesh, natural_image_size, verbose=True, show_debug=False)

        # write OBJ with UV coordinates
        out_path = os.path.join(output_dir, f"wrap_{nr}.obj")
        save_mesh_with_uvs(mesh, out_path, natural_image_size)
        print(f"Saved wrap {nr} mesh → {out_path}")


def mesh_uv_global(filtered_winding_path, output_path):
    """
    Build one seamless UV-mesh across all wraps:
      1) load every wrap_i's (points, uvs)
      2) concatenate into big all_pts, all_uvs, all_wrap_ids
      3) Delaunay on all_uvs → faces
      4) keep only faces whose max(wrap_ids) - min(wrap_ids) ≤ 1
      5) angle-span / edge-area cull via clean_mesh
      6) write single OBJ with per-triangle UVs
    """
    # 1) load and concatenate
    files = sorted(glob.glob(os.path.join(filtered_winding_path, "flattened_winding_*.npz")))
    print(f"mesh_uv_global: Found {len(files)} files to process")
    wrap_ids = []
    pts_list, uvs_list = [], []
    for fp in files:
        try:
            nr = int(os.path.basename(fp).split("_")[-1].split(".")[0])
            print(f"  Loading wrap {nr} from {os.path.basename(fp)}")
            data = np.load(fp)
            P = data["points"]   # (Ni, ≥4)
            U = data["uvs"]      # (Ni, 2)
            print(f"    Loaded {len(P)} points, {len(U)} UVs")
            
            # Skip empty files
            if len(P) == 0 or len(U) == 0:
                print(f"    Skipping wrap {nr} - empty file")
                continue
                
            pts_list.append(P[:, :3])
            uvs_list.append(U)
            wrap_ids.append(np.full((len(P),), nr, dtype=int))
            print(f"    Added wrap {nr} to mesh: {len(P)} points")
        except Exception as e:
            print(f"  Error loading file {fp}: {e}")
            continue
            
    print(f"After loading: {len(pts_list)} valid wraps")
    
    # Check if we have any data to process
    if not pts_list or not uvs_list:
        print("No valid winding data found for global mesh generation")
        return
        
    print("Concatenating arrays...")
    all_pts = np.vstack(pts_list)
    all_uvs = np.vstack(uvs_list)
    all_wr = np.concatenate(wrap_ids)
    print(f"Global arrays: {all_pts.shape} points, {all_uvs.shape} UVs")

    # Validate UV coordinates before Delaunay
    print("Validating UV coordinates...")
    print(f"UV min: [{np.min(all_uvs[:, 0]):.2f}, {np.min(all_uvs[:, 1]):.2f}]")
    print(f"UV max: [{np.max(all_uvs[:, 0]):.2f}, {np.max(all_uvs[:, 1]):.2f}]")
    
    # Check for invalid values
    has_nan = np.any(np.isnan(all_uvs))
    has_inf = np.any(np.isinf(all_uvs))
    if has_nan or has_inf:
        print(f"ERROR: Invalid UV values detected - NaN: {has_nan}, Inf: {has_inf}")
        return
    
    # Check for duplicate points (can cause Delaunay issues)
    unique_uvs, unique_indices = np.unique(all_uvs, axis=0, return_index=True)
    if len(unique_uvs) < len(all_uvs):
        print(f"Warning: Found {len(all_uvs) - len(unique_uvs)} duplicate UV points, using unique points only")
        all_uvs = unique_uvs
        all_pts = all_pts[unique_indices]
        all_wr = all_wr[unique_indices]
        print(f"After deduplication: {all_pts.shape} points, {all_uvs.shape} UVs")
    
    # Final check for minimum points
    if len(all_uvs) < 3:
        print(f"After validation: insufficient points for triangulation: {len(all_uvs)}")
        return
    
    print("UV validation passed, proceeding with Delaunay...")
    
    # Additional geometric checks
    print("Checking UV coordinate distribution...")
    uv_range_u = np.max(all_uvs[:, 0]) - np.min(all_uvs[:, 0])
    uv_range_v = np.max(all_uvs[:, 1]) - np.min(all_uvs[:, 1])
    print(f"UV ranges: U={uv_range_u:.2f}, V={uv_range_v:.2f}")
    
    # Check if points are collinear (could cause Delaunay issues)
    if uv_range_u < 1e-6 or uv_range_v < 1e-6:
        print(f"ERROR: UV points appear to be collinear (very small range)")
        return
    
    # Try a small subset first to test Delaunay
    if len(all_uvs) > 100:
        print("Testing Delaunay with small subset...")
        test_indices = np.random.choice(len(all_uvs), size=100, replace=False)
        test_uvs = all_uvs[test_indices]
        try:
            test_tri = Delaunay(test_uvs)
            print(f"Test triangulation successful: {len(test_tri.simplices)} triangles")
        except Exception as e:
            print(f"ERROR: Test Delaunay failed: {e}")
            return
    
    print("Attempting full Delaunay triangulation...")
    try:
        # 2) Delaunay on the full UV set
        tri = Delaunay(all_uvs)
        print(f"Delaunay successful: {len(tri.simplices)} triangles")
    except Exception as e:
        print(f"ERROR: Delaunay triangulation failed: {e}")
        return

    # 2) Delaunay on the full UV set

    # 3) only allow "neighboring-wrap" triangles
    w0 = all_wr[tri.simplices[:,0]]
    w1 = all_wr[tri.simplices[:,1]]
    w2 = all_wr[tri.simplices[:,2]]
    maxdiff = np.maximum.reduce([w0-w1, w1-w2, w2-w0, w1-w0, w2-w1, w0-w2])
    keep = np.abs(maxdiff) <= 1
    faces = tri.simplices[keep]

    # 4) build Open3D mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices  = o3d.utility.Vector3dVector(all_pts.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(faces)

    # assign per-triangle UVs
    tri_uvs = all_uvs[faces].reshape(-1, 2)
    mesh.triangle_uvs = o3d.utility.Vector2dVector(tri_uvs)

    # Compute normals
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()

    # 5) final cleanup (you can reuse your clean_mesh here)
    # mesh = clean_mesh(mesh,
    #                   longest_edge_pct=95,
    #                   area_pct=95,
    #                   edge_length_thresh=1000)

    # Normalize UVs after cleaning and get natural image size
    mesh, natural_image_size = normalize_uvs(mesh, verbose=True)

    # Apply advanced mask-based filtering
    mesh = filter_mesh_by_mask(mesh, natural_image_size, verbose=True, show_debug=False)

    # 6) save using the new function
    save_mesh_with_uvs(mesh, output_path, natural_image_size)

def filter_mesh_by_mask(mesh, natural_image_size, verbose=True, show_debug=False, fill_area=50000):
    """
    Filter mesh triangles using advanced mask-based filtering.
    
    Steps:
    1. Generate triangulated mask from current mesh
    2. Generate vertex-only mask (white pixels at UV vertex positions)
    3. Dilate vertex mask by 75 pixels
    4. Fill black islands smaller than fill_area pixels
    5. Filter out triangles that have any black pixels in their UV area
    
    Args:
        mesh: Open3D TriangleMesh object with normalized triangle_uvs (0-1 range)
        natural_image_size: tuple, (width, height) natural size from normalization
        verbose: bool, whether to print progress info
        show_debug: bool, whether to show debug mask images
    
    Returns:
        mesh: filtered mesh with problematic triangles removed
    """
    import cv2
    
    def resize_for_display(image, max_width=800, max_height=600):
        """Resize image to fit within max dimensions while maintaining aspect ratio"""
        h, w = image.shape[:2]
        if w <= max_width and h <= max_height:
            return image
        
        # Calculate scaling factor
        scale_w = max_width / w
        scale_h = max_height / h
        scale = min(scale_w, scale_h)
        
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    if not hasattr(mesh, 'triangle_uvs') or len(mesh.triangle_uvs) == 0:
        if verbose:
            print("Warning: Mesh has no UV coordinates, skipping mask filtering")
        return mesh
    
    uvs = np.asarray(mesh.triangle_uvs)
    faces = np.asarray(mesh.triangles)
    
    if verbose:
        print(f"Starting mask-based filtering with {len(faces)} triangles")
    
    # Step 1: Generate first mask (triangulated)
    first_mask, _ = create_mask(mesh, natural_image_size, output_path=None, verbose=False)
    
    # Step 2: Generate vertex-only mask
    vertex_mask = np.zeros(natural_image_size[::-1], dtype=np.uint8)  # height, width
    
    # Convert 0-1 UVs to pixel coordinates
    uv_pixels = uvs * np.array([natural_image_size[0] - 1, natural_image_size[1] - 1])
    uv_pixels = uv_pixels.astype(np.int32)
    
    # Color white pixels at each vertex UV position
    for i in range(len(uv_pixels)):
        x, y = uv_pixels[i]
        # Flip Y coordinate for image coordinate system
        y_flipped = natural_image_size[1] - 1 - y
        # Ensure coordinates are within bounds
        if 0 <= x < natural_image_size[0] and 0 <= y_flipped < natural_image_size[1]:
            vertex_mask[y_flipped, x] = 255
    
    if show_debug:
        cv2.imshow('Step 1: Triangulated Mask', resize_for_display(first_mask))
        cv2.imshow('Step 2: Vertex-only Mask', resize_for_display(vertex_mask))
    
    # Step 3: Dilate by 75 pixels
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (151, 151))  # 75 pixel radius = 151x151 kernel
    dilated_mask = cv2.dilate(vertex_mask, kernel, iterations=1)
    
    if show_debug:
        cv2.imshow('Step 3: Dilated Mask (75px)', resize_for_display(dilated_mask))
    
    # Step 4: Fill black islands smaller than fill_area pixels
    inverted_mask = 255 - dilated_mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inverted_mask, connectivity=8)
    
    final_mask = dilated_mask.copy()
    filled_count = 0
    
    for label in range(1, num_labels):  # Skip background (label 0)
        area = stats[label, cv2.CC_STAT_AREA]
        if area < fill_area:  # Fill small black islands
            final_mask[labels == label] = 255
            filled_count += 1
    
    if show_debug:
        cv2.imshow(f'Step 4: Filled Small Islands (<{fill_area}px)', resize_for_display(final_mask))
    
    # Step 5: Filter triangles based on final mask
    # Use vectorized approach for much better performance
    if verbose:
        print("Filtering triangles using vectorized approach...")
    
    # Reshape UV pixels to (n_triangles, 3, 2) for easier processing
    triangle_uvs = uv_pixels.reshape(-1, 3, 2)
    n_triangles = len(triangle_uvs)
    
    # Flip Y coordinates for image coordinate system
    triangle_coords = triangle_uvs.copy()
    triangle_coords[:, :, 1] = natural_image_size[1] - 1 - triangle_coords[:, :, 1]
    
    # Quick vertex-based filtering: check if all vertices are in valid regions
    vertices_flat = triangle_coords.reshape(-1, 2)  # (n_triangles * 3, 2)
    
    # Check bounds for all vertices at once
    valid_bounds = ((vertices_flat[:, 0] >= 0) & (vertices_flat[:, 0] < natural_image_size[0]) & 
                    (vertices_flat[:, 1] >= 0) & (vertices_flat[:, 1] < natural_image_size[1]))
    
    # Sample final mask at valid vertex positions
    vertex_values = np.zeros(len(vertices_flat), dtype=bool)
    valid_vertices = vertices_flat[valid_bounds].astype(int)
    if len(valid_vertices) > 0:
        vertex_values[valid_bounds] = final_mask[valid_vertices[:, 1], valid_vertices[:, 0]] > 0
    
    # Reshape back to per-triangle format
    vertex_results = vertex_values.reshape(n_triangles, 3)
    
    # Quick elimination: if any vertex is in a black region, triangle is likely invalid
    all_vertices_white = np.all(vertex_results, axis=1)
    any_vertex_black = np.any(~vertex_results, axis=1)
    
    # For triangles where all vertices are white, do additional interior sampling
    uncertain_triangles = all_vertices_white  # These need more detailed checking
    
    if np.any(uncertain_triangles):
        if verbose:
            print(f"Doing detailed checking for {np.sum(uncertain_triangles)} uncertain triangles...")
        
        # Sample points inside uncertain triangles using barycentric coordinates
        uncertain_indices = np.where(uncertain_triangles)[0]
        uncertain_triangle_coords = triangle_coords[uncertain_indices]
        
        # Generate multiple sample points per triangle for robust checking
        n_samples = 5
        sample_results = []
        
        for _ in range(n_samples):
            # Generate random barycentric coordinates
            r1 = np.random.rand(len(uncertain_triangle_coords))
            r2 = np.random.rand(len(uncertain_triangle_coords))
            
            # Convert to valid barycentric coordinates (u, v, w) where u+v+w=1
            u = 1 - np.sqrt(r1)
            v = np.sqrt(r1) * (1 - r2)
            w = np.sqrt(r1) * r2
            
            # Sample points: P = u*V0 + v*V1 + w*V2
            sample_points = (u[:, None] * uncertain_triangle_coords[:, 0] + 
                           v[:, None] * uncertain_triangle_coords[:, 1] + 
                           w[:, None] * uncertain_triangle_coords[:, 2])
            
            # Check bounds and sample mask
            valid_samples = ((sample_points[:, 0] >= 0) & (sample_points[:, 0] < natural_image_size[0]) & 
                           (sample_points[:, 1] >= 0) & (sample_points[:, 1] < natural_image_size[1]))
            
            sample_values = np.zeros(len(sample_points), dtype=bool)
            if np.any(valid_samples):
                valid_coords = sample_points[valid_samples].astype(int)
                sample_values[valid_samples] = final_mask[valid_coords[:, 1], valid_coords[:, 0]] > 0
            
            sample_results.append(sample_values)
        
        # A triangle is valid if ALL samples are in white regions
        all_samples = np.stack(sample_results, axis=1)  # (n_uncertain, n_samples)
        interior_valid = np.all(all_samples, axis=1)
        
        # Update the results for uncertain triangles
        triangle_validity = np.zeros(n_triangles, dtype=bool)
        triangle_validity[uncertain_indices] = interior_valid
        # Triangles with any black vertex are automatically invalid
        triangle_validity[any_vertex_black] = False
    else:
        # No uncertain triangles, just use vertex-based results
        triangle_validity = all_vertices_white & ~any_vertex_black
    
    valid_triangles = np.where(triangle_validity)[0].tolist()
    
    if verbose:
        print(f"Triangle filtering: {len(faces)} -> {len(valid_triangles)} triangles")
    
    if show_debug:
        # Show triangle filtering result
        filtered_mesh_mask = np.zeros(natural_image_size[::-1], dtype=np.uint8)
        for tri_idx in valid_triangles:
            triangle_start = tri_idx * 3
            triangle_uvs = uv_pixels[triangle_start:triangle_start + 3]
            triangle_coords = triangle_uvs.copy()
            triangle_coords[:, 1] = natural_image_size[1] - 1 - triangle_coords[:, 1]
            try:
                cv2.fillPoly(filtered_mesh_mask, [triangle_coords], 255)
            except:
                pass
        
        cv2.imshow(f'Step 5: Filtered Mesh ({len(valid_triangles)} triangles)', resize_for_display(filtered_mesh_mask))
        
        # Comparison: original vs final (show absolute difference)
        difference_mask = cv2.absdiff(first_mask, filtered_mesh_mask)
        cv2.imshow('Difference (Original - Filtered)', resize_for_display(difference_mask))
        
        print("Debug windows will close automatically in 30 seconds...")
        cv2.waitKey(30000)  # Wait 30 seconds (30000 milliseconds)
        cv2.destroyAllWindows()
    
    # Update mesh with valid triangles only
    if len(valid_triangles) > 0:
        valid_faces = faces[valid_triangles]
        
        # Check for out-of-bounds vertex indices
        max_vertex_idx = len(mesh.vertices) - 1
        face_max = np.max(valid_faces) if len(valid_faces) > 0 else -1
        if face_max > max_vertex_idx:
            if verbose:
                print(f"ERROR: Face indices go up to {face_max} but only {len(mesh.vertices)} vertices available")
            return mesh
        
        # Rebuild UV coordinates for valid triangles
        valid_triangle_uvs = uvs.reshape(-1,3,2)[valid_triangles]
        
        try:
            # Create new mesh with filtered triangles
            filtered_mesh = o3d.geometry.TriangleMesh()
            filtered_mesh.vertices = mesh.vertices
            filtered_mesh.triangles = o3d.utility.Vector3iVector(valid_faces)
            
            # Reshape UV coordinates to (-1, 2) format that Open3D expects
            uv_array = valid_triangle_uvs.reshape(-1, 2)
            filtered_mesh.triangle_uvs = o3d.utility.Vector2dVector(uv_array)
            
            # Copy other attributes if they exist
            if hasattr(mesh, 'vertex_normals') and len(mesh.vertex_normals) > 0:
                filtered_mesh.vertex_normals = mesh.vertex_normals
            if hasattr(mesh, 'triangle_normals') and len(mesh.triangle_normals) > 0:
                valid_triangle_normals = np.asarray(mesh.triangle_normals)[valid_triangles]
                filtered_mesh.triangle_normals = o3d.utility.Vector3dVector(valid_triangle_normals)
            
            return filtered_mesh
            
        except Exception as e:
            if verbose:
                print(f"ERROR during mesh reconstruction: {e}")
            return mesh
    else:
        if verbose:
            print("Warning: All triangles were filtered out!")
        return mesh

def main():
    parser = argparse.ArgumentParser(description="Refine pointcloud using mesh and downsampling")
    parser.add_argument("--mesh", required=True, help="Path to mesh file (.ply or .obj)")
    parser.add_argument("--downsample_ratio", type=float, default=1.0, help="Downsample ratio for pointclouds")
    parser.add_argument("--flattening_downsample_ratio", type=float, default=0.3, help="Downsample ratio for flattening solver first pass")
    parser.add_argument("--skip_precomputation", action="store_true", help="Skip precomputation of winding pointclouds")
    parser.add_argument("--display", action="store_true", help="Display pointclouds and meshes")
    parser.add_argument("--display_downsample", type=float, default=0.1,
                        help="Random downsample ratio for 3D display (0..1)")
    parser.add_argument("--color_by_angle", action="store_true",
                        help="Color 3D display by angular value")
    args = parser.parse_args()

    flattened_winding_path = os.path.join(os.path.dirname(args.mesh), "windings_flattened")
    if not args.skip_precomputation:
        generate_winding_pointclouds(
            args.mesh,
            display=args.display,
            display_downsample=args.display_downsample,
            color_by_angle=args.color_by_angle
        )
    flattened_winding_path = flatten_pointcloud(
        os.path.dirname(args.mesh),
        display=args.display,
        downsample_ratio=args.downsample_ratio,
        flattening_downsample_ratio=args.flattening_downsample_ratio,
        display_downsample=args.display_downsample,
        color_by_angle=args.color_by_angle
    )
    grid_uv_flattened(
        flattened_winding_path,
        subsample_radius=30.0,
        display=args.display,
        display_downsample=args.display_downsample,
        color_by_angle=args.color_by_angle
    )
    filtered_path = os.path.join(os.path.dirname(args.mesh), "windings_filtered")
    mesh_uv_wraps(filtered_path, output_dir=os.path.join(os.path.dirname(filtered_path), "uv_meshes"))
    mesh_uv_global(filtered_path, output_path=os.path.join(os.path.dirname(filtered_path), "mesh_refined.obj"))

if __name__ == "__main__":
    main()
