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
        npzfile = np.load(f)
        points = npzfile['points']
    return points

def save_winding_pointcloud(winding_path, winding_nr, points):
    # make folder if it does not exist
    os.makedirs(winding_path, exist_ok=True)
    # Save the winding pointcloud
    file_path = os.path.join(winding_path, f"winding_{int(winding_nr)}.npz")
    points_prev = np.zeros((0, 4), dtype=np.float16)
    if os.path.exists(file_path):
        points_prev = np.load(file_path)['points']
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
    Greedy “pack” subsampling: once a point is kept, all its neighbors
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
    MAD‐based outlier removal on neighbor counts in UV‐space.

    For each index in `selected_indices`, count how many points lie
    within radius `r`.  Compute the median (m) and MAD of those counts,
    then set threshold T = m + k * MAD.  Keep only indices whose count
    > T.

    Args:
        points:           (N,2) array‐like of 2D coords
        selected_indices: list of int, indices to test
        r:                float, radius for neighbor counting
        k:                float, MAD multiplier (default=5.0)

    Returns:
        kept_indices: list[int], subset of selected_indices with count > T
        stats: dict with keys 'median', 'mad', 'threshold'
    """
    pts = np.asarray(points, dtype=float)
    tree = cKDTree(pts)

    # batch query neighbor‐lists
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
      2) In that UV‐neighborhood, compute the median of the 3D coords.
      3) Pick the neighbor whose 3D coordinate is closest to that median.

    Args:
        points:           array‐like of shape (N, ≥3), original 3D coords in [:,:3]
        uvs:              array‐like of shape (N, 2), original UV coords
        selected_indices: list of int, indices into points/uvs to refine
        r_uv:             float, search radius in UV space

    Returns:
        refined_indices: list of int, same length as selected_indices,
                         each replaced by its local 3D “medoid” index
    """
    pts3 = np.asarray(points, dtype=float)[:, :3]
    uvs2 = np.asarray(uvs,    dtype=float)
    tree_uv = cKDTree(uvs2)

    refined = []
    for idx in selected_indices:
        # 1) UV‐space neighbors
        nbrs = tree_uv.query_ball_point(uvs2[idx], r_uv)
        if not nbrs:
            # no UV‐neighbors? keep the original
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
        points:           array‐like (N, ≥3) of original 3D coords
        uvs:              array‐like (N, 2)   of original UV coords
        refined_indices:  list[int], indices to evaluate
        r_uv:             float, search radius in UV space
        thr:              float, maximum allowed mean edge‐error

    Returns:
        kept_indices: list[int], subset of refined_indices with mean_error ≤ thr
        error_list:   list[(idx, mean_error)], sorted by mean_error ascending
    """
    pts3 = np.asarray(points, dtype=float)[:, :3]
    uvs2 = np.asarray(uvs,    dtype=float)
    tree_uv = cKDTree(uvs2)

    error_list = []
    for i in refined_indices:
        # 1) UV‐space neighbors
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

def display_3d_pointcloud(points):
    # interactive 3d visualization of the downsampled points with matplotlib
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)
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
    
def generate_winding_pointclouds(mesh_path):
    vertices, (has_points, angles) = load_mesh(mesh_path)
    angles = np.reshape(angles, (-1, 1))
    print(f"Shape of vertices: {vertices.shape} and of angles: {angles.shape}")
    points_mesh = np.concatenate((vertices, angles), axis=1)
    points_mesh = points_mesh[np.logical_not(has_points)]
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

def flatten_pointcloud(base_path, k_neighbors=8, angle_threshold=30, angle_weight=0.2):
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

    # initialize storage for previous flattened coordinates per winding
    prev_uvs_u = {}
    prev_uvs_v = {}
    prev_points = {}

    # process windings in segments
    winding_us = None
    winding_vs = None
    for start in range(0, len(winding_files)):
        if start < 30: # for debug, leave it for now
            continue
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
                # display_3d_pointcloud(points_[:, :3]) # Debug visualization
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
        # prepare KDTree input: optionally include 4th-dimension (angle) by weighting
        if angle_weight is not None:
            angs = points[:, 3].reshape(-1, 1) * angle_weight
            tree_input = np.hstack((coords, angs))
        else:
            tree_input = coords
        # build KDTree and query k+1 neighbors (self included)
        tree = cKDTree(tree_input)
        temp_distances, indices = tree.query(tree_input, k=k_neighbors + 1) # "wrong" distance computation. Use 3D euclidean distance instad on first 3 dimensions only
        distances = np.linalg.norm(coords[indices] - coords[:, np.newaxis], axis=-1) # Coords euclidean distance
        assert distances.shape == temp_distances.shape, "Distance shape mismatch"
        del temp_distances

        print(f"Shape of distances: {distances.shape} and indices: {indices.shape}")
        # drop self
        neighbor_ids = indices[:, 1:]
        neighbor_dists = distances[:, 1:]
        # apply 4th-dimension constraint if specified: drop neighbors with large angle diff
        if angle_threshold is not None:
            angs = points[:, 3]
            # broadcast to (n_points, k_neighbors)
            ref = angs.reshape(-1, 1)
            nbrs = angs[neighbor_ids]
            diff = np.abs(ref - nbrs)
            mask = diff > angle_threshold
            neighbor_ids[mask] = -1
            neighbor_dists[mask] = np.inf
        # build Python lists for each node, dropping invalid neighbors
        neighbor_lists = [[] for _ in range(len(coords))]
        distance_lists = [[] for _ in range(len(coords))]
        print(f"Length of coords: {len(coords)}, Length of neighbor_ids: {len(neighbor_lists)}, Length of neighbor_dists: {len(distance_lists)}")
        # iterate over each point and its neighbors
        for i in range(len(coords)):
            # get valid neighbors
            valid_indices = np.where(neighbor_ids[i] != -1)[0]
            valid_neighbors = neighbor_ids[i][valid_indices]
            valid_distances = neighbor_dists[i][valid_indices]
            # Add undirected edges
            neighbor_lists[i].extend(valid_neighbors)
            distance_lists[i].extend(valid_distances)
            # Add reverse edges
            for j in range(len(valid_indices)):
                source = valid_neighbors[j]
                target = i
                neighbor_lists[source].append(target)
                distance_lists[source].append(valid_distances[j])

        # unique neighbor lists
        for i in range(len(coords)):
            _, unique_indices = np.unique(neighbor_lists[i], return_index=True)
            neighbor_lists[i] = [neighbor_lists[i][idx] for idx in unique_indices]
            distance_lists[i] = [distance_lists[i][idx] for idx in unique_indices]
            # Filter out  indices with distances > 10000
            neighbor_lists[i] = [neighbor_lists[i][j] for j, d in enumerate(distance_lists[i]) if d < 10000]
            distance_lists[i] = [d for d in distance_lists[i] if d < 10000]

        # Check that each edge is undirected
        # for i in range(len(coords)):
        #     for j in range(len(neighbor_lists[i])):
        #         neighbor = neighbor_lists[i][j]
        #         if i not in neighbor_lists[neighbor]:
        #             print(f"Edge {i} -> {neighbor} is not undirected")
        #             break
        
        print(f"Min/Max distances: {np.min(np.concatenate(distance_lists))}, {np.max(np.concatenate(distance_lists))}")

        # load graph into cpp
        coords_z_index = 2
        # orchestrate initial guess for current flattened coordinates (u: angle, v: z)
        current_angles = np.zeros_like(winding_angles)
        current_z = np.zeros_like(coords[:, coords_z_index])
        # assign previous results for wraps if available, otherwise use initial values
        # assign previous results for wraps if available and matching size, otherwise use defaults
        start_idx = 0
        # track ranges of previous and new wraps within this window segment
        prev_ranges = []
        new_ranges = []
        for idx_in_segment, count in enumerate(winding_indices):
            wrap_nr = winding_files_indices[start + idx_in_segment]
            # default slices
            default_u = winding_angles[start_idx:start_idx+count]
            default_v = coords[start_idx:start_idx+count, coords_z_index]
            prev_u = prev_uvs_u.get(wrap_nr, None)
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
        # align newly added wraps to previously computed wraps in u (angle) and v (z)
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
        solver = graph_problem_gpu_py.Solver(neighbor_lists, distance_lists, winding_angles, current_angles, coords[:, coords_z_index], current_z)
        print("Set up the graph")
        # fix nodes of the start winding if it has been previously computed
        start_wrap_nr = winding_files_indices[start]
        # only fix if prev_uvs length matches this wrap's point count
        first_count = winding_indices[0]
        prev_u0 = prev_uvs_u.get(start_wrap_nr)
        if prev_u0 is not None and prev_u0.shape[0] == first_count:
            to_fix = list(range(first_count))
            solver.fix_nodes(to_fix)
            print(f"Fixed {len(to_fix)} nodes of the first winding {start_wrap_nr} with previous results.")
        else:
            print(f"Warning: wrap {start_wrap_nr} expected {first_count}, not fixing nodes.")
        # solve the graph problem
        min_angle = np.min(winding_angles)
        max_angle = np.max(winding_angles)
        z_min = np.min(coords[:, coords_z_index])
        z_max = np.max(coords[:, coords_z_index])
        print(F"Min/Max angles: {min_angle}, {max_angle}, Min/Max z: {z_min}, {z_max}")

        zero_ranges = [(min_angle, min_angle+270), (max_angle-270, max_angle)]
        a_step = (max_angle - min_angle)/winding_width
        zero_ranges_initial = [(min_angle + index * a_step, min_angle + (index+1)*a_step) for index in range(winding_width)]
        zero_ranges_fine = [(min_angle + (index + 0.4) * a_step, min_angle + (index+0.6)*a_step) for index in range(winding_width)]
        solver.solve_flattening(num_iterations=20000, visualize=True, angle_tug_min=min_angle+90, angle_tug_max=max_angle-90, z_tug_min=z_min+100, z_tug_max=z_max-100, tug_step=0.5, zero_ranges=zero_ranges_initial)
        solver.solve_flattening(num_iterations=150000, visualize=True, zero_ranges=zero_ranges, tug_step=0.0005)
        undeleted_indices = np.array(solver.get_undeleted_indices())
        
        uvs = np.array(solver.get_uvs())
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

def grid_uv_flattened(flattened_winding_path, subsample_radius=3.0, display=False):
    winding_filtered_path = os.path.join(os.path.dirname(flattened_winding_path), "windings_filtered")
    # find and order all winding pointclouds
    winding_files = glob.glob(os.path.join(flattened_winding_path, "flattened_winding_*.npz"))
    print(f"Found {len(winding_files)} flattened winding pointclouds.")
    # Bring the files in order of their winding number
    winding_files_indices = sorted([int(os.path.basename(wf).split("_")[2].split(".")[0]) for wf in winding_files])
    winding_files_indices = winding_files_indices[5:]
    winding_files = [os.path.join(flattened_winding_path, f"flattened_winding_{i}.npz") for i in winding_files_indices]
    # Load each wrap, subsample
    for i in range(len(winding_files)):
        points, uvs = load_flattened_winding(flattened_winding_path, winding_files_indices[i])
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
            # display  with matplotlib. 2 2d views. left ioriginal uv, right subsampled uv. then another window for 3d subsampled and original
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

            display_3d_pointcloud(points_subsampled)

        kept_indices = refine_by_medoid(points, uvs, kept_indices, subsample_radius)
        points_subsampled = points[kept_indices]
        uvs_subsampled = uvs[kept_indices]
        if display:
            display_3d_pointcloud(points_subsampled)

        # kept_indices, errors = filter_by_edge_error(points[:,:3], uvs, kept_indices, 10 * subsample_radius, int(1.5*subsample_radius))
        kept_indices, errors = filter_by_edge_error_mad(points[:,:3], uvs, kept_indices, 10 * subsample_radius, k=5.0) # automatic threshold finding
        points_subsampled = points[kept_indices]
        uvs_subsampled = uvs[kept_indices]
        if display:
            display_3d_pointcloud(points_subsampled)

        save_flattened_winding(winding_filtered_path, winding_files_indices[i], points_subsampled, uvs_subsampled)

def mesh_uv_wraps(filtered_winding_path, output_dir):
    """
    For each filtered winding .npz in `filtered_winding_path`:
      1) Load points (Nx3+) and uvs (Nx2)
      2) Run Delaunay on the UVs → simplices (M×3)
      3) Build an Open3D TriangleMesh:
           - vertices = 3D points
           - triangles = simplices
           - triangle_uvs = flattened per‐triangle UV coords
      4) Save as OBJ with UVs in `output_dir/wrap_<nr>.obj`

    Args:
        filtered_winding_path: str, folder containing
            `flattened_winding_<nr>.npz` with arrays `points, uvs`
        output_dir: str, where to write `wrap_<nr>.obj`
    """
    os.makedirs(output_dir, exist_ok=True)

    # grab all filtered wraps
    winding_flattened_files = glob.glob(os.path.join(filtered_winding_path, "flattened_winding_*.npz"))
    print(f"Found {len(winding_flattened_files)} flattened winding pointclouds.")
    # Bring the files in order of their winding number
    winding_flattened_files_indices = sorted([int(os.path.basename(wf).split("_")[2].split(".")[0]) for wf in winding_flattened_files])
    winding_flattened_files_indices = winding_flattened_files_indices[5:]
    winding_flattened_files = [os.path.join(filtered_winding_path, f"flattened_winding_{i}.npz") for i in winding_flattened_files_indices]
    for fp in winding_flattened_files:
        nr = int(os.path.basename(fp).split("_")[-1].split(".")[0])
        data = np.load(fp)
        pts3 = data["points"][:, :3].astype(np.float64)
        winding_angles = data["points"][:, 3].astype(np.float64)
        uvs2 = data["uvs"].astype(np.float64)

        # 2D Delaunay in UV space
        tri = Delaunay(uvs2)
        faces = tri.simplices  # M×3 array of vertex‐indices

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
        # so we flatten face‐by‐face
        tri_uvs = uvs2[faces].reshape(-1, 2)
        mesh.triangle_uvs = o3d.utility.Vector2dVector(tri_uvs)

        # Compute normals
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()

        mesh = clean_mesh(mesh, longest_edge_pct=100, area_pct=100, edge_length_thresh=1000)

        # write OBJ with UV coordinates
        out_path = os.path.join(output_dir, f"wrap_{nr}.obj")
        o3d.io.write_triangle_mesh(out_path, mesh)
        print(f"Saved wrap {nr} mesh → {out_path}")


def mesh_uv_global(filtered_winding_path, output_path):
    """
    Build one seamless UV‐mesh across all wraps:
      1) load every wrap_i’s (points, uvs)
      2) concatenate into big all_pts, all_uvs, all_wrap_ids
      3) Delaunay on all_uvs → faces
      4) keep only faces whose max(wrap_ids) - min(wrap_ids) ≤ 1
      5) angle‐span / edge‐area cull via clean_mesh
      6) write single OBJ with per‐triangle UVs
    """
    # 1) load and concatenate
    files = sorted(glob.glob(os.path.join(filtered_winding_path, "flattened_winding_*.npz")))
    wrap_ids = []
    pts_list, uvs_list = [], []
    for fp in files:
        nr = int(os.path.basename(fp).split("_")[-1].split(".")[0])
        data = np.load(fp)
        P = data["points"]   # (Ni, ≥4)
        U = data["uvs"]      # (Ni, 2)
        pts_list.append(P[:, :3])
        uvs_list.append(U)
        wrap_ids.append(np.full((len(P),), nr, dtype=int))
    all_pts = np.vstack(pts_list)
    all_uvs = np.vstack(uvs_list)
    all_wr = np.concatenate(wrap_ids)

    # 2) Delaunay on the full UV set
    tri = Delaunay(all_uvs)
    faces = tri.simplices.astype(int)  # (M,3)

    # 3) only allow “neighboring‐wrap” triangles
    w0 = all_wr[faces[:,0]]
    w1 = all_wr[faces[:,1]]
    w2 = all_wr[faces[:,2]]
    maxdiff = np.maximum.reduce([w0-w1, w1-w2, w2-w0, w1-w0, w2-w1, w0-w2])
    keep = np.abs(maxdiff) <= 1
    faces = faces[keep]

    # 4) build Open3D mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices  = o3d.utility.Vector3dVector(all_pts.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(faces)

    # assign per‐triangle UVs
    tri_uvs = all_uvs[faces].reshape(-1, 2)
    mesh.triangle_uvs = o3d.utility.Vector2dVector(tri_uvs)

    # Compute normals
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()

    # 5) final cleanup (you can reuse your clean_mesh here)
    mesh = clean_mesh(mesh,
                      longest_edge_pct=95,
                      area_pct=95,
                      edge_length_thresh=1000)

    # 6) write one global OBJ
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    o3d.io.write_triangle_mesh(output_path, mesh)
    print(f"Saved global UV‐mesh → {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Refine pointcloud using mesh and downsampling")
    parser.add_argument("--mesh", required=True, help="Path to mesh file (.ply or .obj)")
    parser.add_argument("--skip_precomputation", action="store_true", help="Skip precomputation of winding pointclouds")
    args = parser.parse_args()

    flattened_winding_path = os.path.join(os.path.dirname(args.mesh), "windings_flattened")
    if not args.skip_precomputation:
        generate_winding_pointclouds(args.mesh)
        flattened_winding_path = flatten_pointcloud(os.path.dirname(args.mesh))
    grid_uv_flattened(flattened_winding_path, subsample_radius=30.0, display=False)
    filtered_path = os.path.join(os.path.dirname(args.mesh), "windings_filtered")
    mesh_uv_wraps(filtered_path, output_dir=os.path.join(os.path.dirname(filtered_path), "uv_meshes"))
    mesh_uv_global(filtered_path, output_path=os.path.join(os.path.dirname(filtered_path), "mesh_refined.obj"))

if __name__ == "__main__":
    main()
