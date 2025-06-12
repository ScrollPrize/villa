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
from collections import deque
import cv2

sys.path.append('ThaumatoAnakalyptor/graph_problem/build')
import graph_problem_gpu_py

### UTILS ###

def load_winding_direction(winding_direction_path):
    """
    Load winding direction from a text file.
    
    The file should contain a line like "Winding direction: True" or "Winding direction: False"
    as saved by the graph_to_mesh.py WalkToSheet class.
    
    Args:
        winding_direction_path (str): Path to the winding direction text file
        
    Returns:
        bool: True if winding direction is normal, False if reversed
        
    Raises:
        FileNotFoundError: If the winding direction file doesn't exist
        ValueError: If the file format is invalid or cannot be parsed
    """
    if not os.path.exists(winding_direction_path):
        raise FileNotFoundError(f"Winding direction file not found: {winding_direction_path}")
    
    try:
        with open(winding_direction_path, "r") as f:
            content = f.read().strip()
        
        # Expected format: "Winding direction: True" or "Winding direction: False"
        if "Winding direction:" not in content:
            raise ValueError(f"Invalid file format. Expected 'Winding direction: <bool>', got: {content}")
        
        # Extract the boolean value part
        direction_str = content.split("Winding direction:")[-1].strip()
        
        # Parse boolean value
        if direction_str.lower() == "true":
            winding_direction = True
        elif direction_str.lower() == "false":
            winding_direction = False
        else:
            raise ValueError(f"Invalid boolean value. Expected 'True' or 'False', got: {direction_str}")
        
        print(f"Loaded winding direction: {winding_direction}")
        return winding_direction
        
    except Exception as e:
        if isinstance(e, (FileNotFoundError, ValueError)):
            raise
        else:
            raise ValueError(f"Error reading winding direction file {winding_direction_path}: {str(e)}")

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
            if wrap_nr not in winding_files_indices[start+1:end]:
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
    # med   = np.median(errors)
    # mad   = np.median(np.abs(errors - med))
    # thr   = med + k * mad

    q3   = np.percentile(errors, 0.75)
    median   = np.percentile(errors, 0.5)
    thr   = median + k * (q3 - median)

    # 4) filter
    kept_indices = [idx for idx, err in error_list if err <= thr]

    # 5) print stats
    print(f"Median error: {median:.4f}")
    print(f"Q3 error: {q3:.4f}")
    # print(f"MAD:          {mad:.4f}")
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

def identify_spike_indices(
    mesh: o3d.geometry.TriangleMesh,
    mad_factor: float = 3.0,
    angle_threshold: float = 45.0
) -> np.ndarray:
    """
    Identify vertices in a mesh that have spike-like properties by comparing
    face normals to vertex normals and using MAD-based outlier detection.
    
    Args:
        mesh: The input TriangleMesh
        mad_factor: How aggressively to consider spikiness as an outlier 
                   (e.g., 3.0 means spikiness > median + 3*MAD is 'spiky')
        angle_threshold: Maximum allowed angle between face and vertex normals in degrees
    
    Returns:
        np.ndarray: Boolean array where True indicates a non-spiky vertex (to keep)
    """
    # 1) Ensure we have normals computed
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()

    # Extract arrays for faster NumPy processing
    vertex_normals = np.asarray(mesh.vertex_normals)   # (V, 3)
    face_normals   = np.asarray(mesh.triangle_normals) # (F, 3)
    faces          = np.asarray(mesh.triangles)        # (F, 3)

    # 2) Vectorized spikiness computation (face-normal vs. vertex-normal angle)
    face_vertex_normals = vertex_normals[faces]                # (F, 3, 3)
    dot_vals = np.sum(face_normals[:, None, :] * face_vertex_normals, axis=2)  # (F, 3)
    np.clip(dot_vals, -1.0, 1.0, out=dot_vals)
    angles = np.arccos(dot_vals)                # (F, 3)

    # Accumulate angle sums & counts for each vertex
    spikiness_sum = np.zeros(len(vertex_normals), dtype=np.float64)
    counts        = np.zeros(len(vertex_normals), dtype=np.int32)

    # Add angles for each of the 3 vertices of each face
    np.add.at(spikiness_sum, faces[:, 0], angles[:, 0])
    np.add.at(spikiness_sum, faces[:, 1], angles[:, 1])
    np.add.at(spikiness_sum, faces[:, 2], angles[:, 2])
    np.add.at(counts,        faces[:, 0], 1)
    np.add.at(counts,        faces[:, 1], 1)
    np.add.at(counts,        faces[:, 2], 1)

    with np.errstate(divide='ignore', invalid='ignore'):
        spikiness = np.where(counts > 0, spikiness_sum / counts, 0.0)

    # 3) MAD-based outlier detection on spikiness
    med_spike = np.median(spikiness)
    abs_dev   = np.abs(spikiness - med_spike)
    mad       = np.median(abs_dev)
    
    if mad < 1e-12:
        # If everything is basically the same, no spiky outliers
        is_spiky = np.full_like(spikiness, False, dtype=bool)
    else:
        threshold_spike = med_spike + mad_factor * mad
        is_spiky = spikiness > threshold_spike

    # Convert angles to degrees for threshold comparison
    angles_deg = np.degrees(spikiness)
    
    # Combine MAD-based detection with absolute angle threshold
    keep_indices = ~(is_spiky | (angles_deg > angle_threshold))
    
    print(f"Identified {np.sum(~keep_indices)} spike vertices out of {len(keep_indices)} total vertices")
    print(f"Median spikiness: {np.degrees(np.median(spikiness)):.2f}°")
    print(f"MAD: {np.degrees(mad):.2f}°")
    print(f"Threshold: {np.degrees(threshold_spike):.2f}°")
    
    return keep_indices

def filter_by_spikes(points, uvs, mad_factor=3.0, angle_threshold=45.0):
    """
    Filter points by creating a temporary mesh and identifying non-spiky vertices.
    
    Args:
        points: array of shape (N, 3) with 3D coordinates
        uvs: array of shape (N, 2) with UV coordinates
        mad_factor: How aggressively to consider spikiness as an outlier
        angle_threshold: Maximum allowed angle between face and vertex normals in degrees
        
    Returns:
        kept_indices: list of indices to keep (non-spiky points)
    """
    if len(points) < 3:
        print("Not enough points for triangulation")
        return []
        
    # Create Delaunay triangulation in UV space
    try:
        tri = Delaunay(uvs)
        faces = tri.simplices  # M×3 array of vertex-indices
    except Exception as e:
        print(f"Delaunay triangulation failed: {e}")
        return []
        
    # Build Open3D mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    
    # Compute normals
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    
    # Identify non-spiky vertices
    keep_mask = identify_spike_indices(mesh, mad_factor=mad_factor, angle_threshold=angle_threshold)
    kept_indices = np.where(keep_mask)[0].tolist()
    
    print(f"Spike filtering: {len(points)} -> {len(kept_indices)} points")
    return kept_indices

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
    # mesh.remove_duplicated_vertices()

    # 2) cull huge triangles by edge-length / area
    verts = np.asarray(mesh.vertices)
    tris  = np.asarray(mesh.triangles)
    tri_verts = verts[tris]
    mesh_uvs = np.asarray(mesh.triangle_uvs).reshape(-1, 3, 2)                # (M,3,3)

    # edge lengths 3d
    e0 = np.linalg.norm(tri_verts[:,1,:] - tri_verts[:,0,:], axis=1)
    e1 = np.linalg.norm(tri_verts[:,2,:] - tri_verts[:,1,:], axis=1)
    e2 = np.linalg.norm(tri_verts[:,0,:] - tri_verts[:,2,:], axis=1)
    longest_edge = np.maximum.reduce([e0, e1, e2], axis=0)

    # edge lengths uv
    e0_uv = np.linalg.norm(mesh_uvs[:,1,:] - mesh_uvs[:,0,:], axis=1)
    e1_uv = np.linalg.norm(mesh_uvs[:,2,:] - mesh_uvs[:,1,:], axis=1)
    e2_uv = np.linalg.norm(mesh_uvs[:,0,:] - mesh_uvs[:,2,:], axis=1)
    longest_edge_uv = np.maximum.reduce([e0_uv, e1_uv, e2_uv], axis=0)

    # areas
    cross_prod = np.cross(tri_verts[:,1] - tri_verts[:,0],
                          tri_verts[:,2] - tri_verts[:,0])
    areas = 0.5 * np.linalg.norm(cross_prod, axis=1)

    edge_thr = np.percentile(longest_edge, longest_edge_pct)
    area_thr = np.percentile(areas, area_pct)
    print(f"Edge threshold: {edge_thr}, area threshold: {area_thr}, edge length threshold: {edge_length_thresh}")
    bad = np.where((longest_edge > edge_thr) | (areas > area_thr) | (longest_edge > edge_length_thresh) | (longest_edge_uv > edge_length_thresh))[0]
    bad = list(bad)
    min_bad = np.min(bad) if len(bad) > 0 else None
    max_bad = np.max(bad) if len(bad) > 0 else None
    print(f"Removing bad indices of triangles, min max index: {min_bad}, {max_bad}. Number of bad triangles: {len(bad)}")
    mesh.remove_triangles_by_index(bad)

    # 3) final cleanup
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()

    if True:
        # Remove connected components that have too small of an area
        triangle_clusters, cluster_n_triangles, cluster_area = mesh.cluster_connected_triangles()
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_area = np.asarray(cluster_area)
        
        # Find the largest cluster
        largest_cluster_idx = np.argmax(cluster_area)
        largest_cluster_area = cluster_area[largest_cluster_idx]
        
        # Remove clusters with area less than threshold
        area_threshold = 1000000  # absolute area threshold
        small_clusters = np.where(cluster_area < area_threshold)[0]
        triangles_to_remove = np.where(np.isin(triangle_clusters, small_clusters))[0]
        
        if len(triangles_to_remove) > 0:
            print(f"Removing {len(triangles_to_remove)} triangles from {len(small_clusters)} small components")
            # print(f"Areas: {[f'{a:.2f}' for a in cluster_area[small_clusters]]}")
            print(f"Largest component area: {largest_cluster_area:.2f}")
            mesh.remove_triangles_by_index(triangles_to_remove)
            mesh.remove_unreferenced_vertices()
            
            # Recompute normals after removing components
            mesh.compute_vertex_normals()
            mesh.compute_triangle_normals()

    # 4) warn if still self-intersecting
    if False and hasattr(mesh, "is_self_intersecting") and mesh.is_self_intersecting():
        print("Warning: mesh still reports self-intersections after cleanup.")
    print(f"Mesh has {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles")
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
    
    # Create intermediate directory for per-slab windings
    intermediate_path = os.path.join(base_path, "intermediate_windings")
    os.makedirs(intermediate_path, exist_ok=True)
    
    print("Processing slabs and splitting into windings...")
    all_windings_found = set()
    
    # Step 1: Process each slab and split into per-winding files
    for slab_idx, slab_path in enumerate(tqdm(slabs, desc="Processing slabs to per-winding files")):
        points = load_pointcloud_slab(slab_path).astype(np.float16)
        # From TA to original coordinates
        if len(points.shape) != 2:
            continue
        points = shuffling_points_axis(points)
        points[:, :3] = points[:, :3] * 4.0 - 500
        # To original winding angle
        points[:, 3] = points[:, 3] + 90
        # Append source flag: 0 for slab points
        slab_flags = np.zeros((points.shape[0], 1), dtype=points.dtype)
        points = np.concatenate((points, slab_flags), axis=1)

        # Concatenate with mesh points
        if points.shape[0] == 0:
            print(f"Slab {slab_idx} has no points, using only mesh points...")
            combined_points = points_mesh
        else:
            combined_points = np.concatenate((points, points_mesh), axis=0)
        
        print(f"Slab {slab_idx}: {combined_points.shape[0]} total points")
        
        # Find winding range for this slab
        min_angle = np.min(combined_points[:, 3])
        max_angle = np.max(combined_points[:, 3])
        min_winding = int(np.floor(min_angle / 360.0))
        max_winding = int(np.ceil(max_angle / 360.0))
        
        # Split points by winding and save separate files
        for winding_nr in range(min_winding, max_winding + 1):
            winding_angle = winding_nr * 360.0
            winding_mask = np.logical_and(combined_points[:, 3] >= winding_angle, 
                                        combined_points[:, 3] < winding_angle + 360.0)
            winding_points = combined_points[winding_mask]
            
            if len(winding_points) > 0:
                # Save as winding_{nr}_slab_{idx}.npz
                intermediate_file = os.path.join(intermediate_path, f"winding_{winding_nr}_slab_{slab_idx:04d}.npz")
                np.savez_compressed(intermediate_file, points=winding_points)
                all_windings_found.add(winding_nr)
                print(f"  Winding {winding_nr}: {len(winding_points)} points -> {os.path.basename(intermediate_file)}")
    
    print(f"Found {len(all_windings_found)} unique windings across all slabs: {sorted(all_windings_found)}")
    
    # Step 2: Generate final winding files by loading all slab files for each winding
    windings_path = os.path.join(base_path, "windings")
    os.makedirs(windings_path, exist_ok=True)
    
    print("Generating final winding pointclouds...")
    for winding_nr in tqdm(sorted(all_windings_found), desc="Generating final winding files"):
        # Find all intermediate files for this winding
        pattern = os.path.join(intermediate_path, f"winding_{winding_nr}_slab_*.npz")
        winding_slab_files = sorted(glob.glob(pattern))
        
        print(f"Winding {winding_nr}: found {len(winding_slab_files)} slab files")
        
        if winding_slab_files:
            # Load and concatenate all points for this winding
            winding_points_list = []
            for slab_file in winding_slab_files:
                data = np.load(slab_file)
                points = data['points']
                winding_points_list.append(points)
                print(f"  Loaded {len(points)} points from {os.path.basename(slab_file)}")
            
            # Concatenate all points for this winding
            combined_winding_points = np.concatenate(winding_points_list, axis=0)
            print(f"  Total before deduplication: {len(combined_winding_points)} points")
            
            # Remove duplicates
            combined_winding_points = np.unique(combined_winding_points, axis=0)
            print(f"  Total after deduplication: {len(combined_winding_points)} points")
            
            # Save final winding file
            final_winding_file = os.path.join(windings_path, f"winding_{winding_nr}.npz")
            np.savez_compressed(final_winding_file, points=combined_winding_points)
            print(f"Saved final winding {winding_nr}: {len(combined_winding_points)} unique points")
            
            if display:
                print(f"Displaying winding {winding_nr} (loaded {combined_winding_points.shape[0]} points)")
                display_3d_pointcloud(combined_winding_points,
                                     color_by_angle=color_by_angle,
                                     downsample_ratio=display_downsample)
    
    # Step 3: Clean up intermediate files
    print("Cleaning up intermediate files...")
    intermediate_files = glob.glob(os.path.join(intermediate_path, "winding_*_slab_*.npz"))
    for file_path in intermediate_files:
        try:
            os.remove(file_path)
        except OSError:
            pass
    
    try:
        os.rmdir(intermediate_path)
    except OSError:
        pass
    
    print(f"Successfully generated {len(all_windings_found)} winding pointclouds from {len(slabs)} slabs")

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
    if False:  # COMMENTED OUT - replaced with Z-Angle connectivity below
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
    if False:  # COMMENTED OUT - replaced with Z-Angle connectivity below
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
    
    # NEW: Z-Angle based connectivity using KDTree
    if angles is not None:
        print("Building Z-Angle based connectivity using KDTree...")
        
        # Create KDTree on z and angle coordinates (4x weight on angle)
        z_vals = pts[:, 2]
        angle_weight_factor = 4.0
        weighted_angles = angle_vals * angle_weight_factor
        za_coords = np.column_stack([z_vals, weighted_angles])  # (N, 2)
        za_tree = cKDTree(za_coords)
        
        connections_added = 0
        max_distance_filter = 300.0  # Filter connections further than 300 units apart
        
        # Process all points (not just mesh points)
        print(f"Processing {len(pts)} points for Z-Angle based connectivity...")
        
        # VECTORIZED APPROACH: Pre-generate all query positions
        print("Pre-generating all query positions...")
        angle_offsets = [-5.0, 5.0, -0.5, 0.5]
        z_offsets = [-15, -5, 5, 15]
        
        # Create meshgrid of all combinations
        n_points = len(pts)
        n_queries_per_point = len(angle_offsets) * len(z_offsets)
        
        # Pre-allocate arrays
        all_query_positions = np.zeros((n_points * n_queries_per_point, 2))
        point_indices = np.zeros(n_points * n_queries_per_point, dtype=int)
        query_angle_offsets_flat = np.zeros(n_points * n_queries_per_point)
        
        # Fill arrays vectorized
        idx = 0
        for angle_offset in angle_offsets:
            for z_offset in z_offsets:
                # Generate queries for all points at once
                query_angles = angle_vals + angle_offset
                query_z = z_vals + z_offset
                weighted_query_angles = query_angles * angle_weight_factor
                
                # Store in pre-allocated arrays
                all_query_positions[idx:idx+n_points, 0] = query_z
                all_query_positions[idx:idx+n_points, 1] = weighted_query_angles
                point_indices[idx:idx+n_points] = np.arange(n_points)
                query_angle_offsets_flat[idx:idx+n_points] = abs(angle_offset)
                
                idx += n_points
        
        print(f"Generated {len(all_query_positions)} query positions, now doing batch KDTree query...")
        
        # Single batch KDTree query for all positions
        batch_dists, batch_indices = za_tree.query(all_query_positions, k=1)
        
        print("Processing query results...")
        # VECTORIZED CONNECTION PROCESSING
        print("Creating boolean masks for filtering...")
        
        # Extract arrays for vectorized operations
        point_i_array = point_indices
        found_idx_array = batch_indices
        max_angle_diff_array = 2.0 * query_angle_offsets_flat
        
        # Create boolean masks for all filtering conditions
        # 1. Skip self-connections
        not_self_mask = found_idx_array != point_i_array
        
        # 2. Calculate all 3D distances at once
        point_coords = pts[point_i_array]
        found_coords = pts[found_idx_array]
        distances_3d = np.linalg.norm(point_coords - found_coords, axis=1)
        distance_mask = np.logical_or(distances_3d <= max_distance_filter, np.logical_or(is_mesh[point_i_array], is_mesh[found_idx_array]))
        
        # 3. Apply global angle threshold if specified
        if angle_threshold is not None:
            angle_diffs_global = np.abs(angle_vals[point_i_array] - angle_vals[found_idx_array])
            global_angle_mask = angle_diffs_global <= angle_threshold
        else:
            global_angle_mask = np.ones(len(point_i_array), dtype=bool)
        
        # 4. Apply query-specific angle difference filter
        angle_diffs_query = np.abs(angle_vals[point_i_array] - angle_vals[found_idx_array])
        query_angle_mask = angle_diffs_query < max_angle_diff_array
        
        # Combine all masks
        valid_mask = not_self_mask & distance_mask & global_angle_mask & query_angle_mask
        
        print(f"Filtered connections: {np.sum(valid_mask)} valid out of {len(valid_mask)} total")
        
        # Extract valid connections
        valid_point_i = point_i_array[valid_mask]
        valid_found_idx = found_idx_array[valid_mask]
        valid_distances = distances_3d[valid_mask]
        
        print("Adding valid connections to neighbor lists...")
        # Add connections to neighbor lists (this part still needs a loop for list operations)
        for i in tqdm(range(len(valid_point_i)), desc="Adding connections"):
            pi = valid_point_i[i]
            fi = valid_found_idx[i]
            dist = valid_distances[i]
            
            # Add bidirectional connections (skip duplicate check for speed - will be handled later)
            neighbor_lists[pi].append(fi)
            distance_lists[pi].append(dist)
            neighbor_lists[fi].append(pi)
            distance_lists[fi].append(dist)
            connections_added += 1
        
        print(f"Added {connections_added} Z-Angle based connections (filtered by distance < {max_distance_filter})")
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

def create_flattening_downsample_indices(points, is_mesh_flags, downsample_ratio, downsample_ratio_mesh):
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

    # Downsample slab points
    if len(mesh_indices) > 0 and downsample_ratio_mesh < 1.0:
        n_slab_keep = int(len(mesh_indices) * downsample_ratio_mesh)
        if n_slab_keep > 0:
            kept_slab_mesh = np.random.choice(mesh_indices, n_slab_keep, replace=False)
        else:
            kept_slab_mesh = np.array([], dtype=int)
    else:
        kept_slab_mesh = mesh_indices
    
    # Combine and sort
    all_kept = np.concatenate([kept_slab_mesh, kept_slab])
    all_kept = np.sort(all_kept)
    
    print(f"Downsampling for solver: kept {len(kept_mesh)} mesh + {len(kept_slab)} slab = {len(all_kept)} total points")
    
    return all_kept

def flatten_pointcloud(base_path,
                       k_neighbors=10,
                       angle_threshold=40,
                       angle_weight=0.2,
                       downsample_ratio=0.1,
                       flattening_downsample_ratio=0.075,
                       flattening_downsample_ratio_mesh=0.75,
                       display=False,
                       display_downsample=0.1,
                       color_by_angle=False,
                       from_winding=None,
                       debug_winding=None):
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
        debug_winding (int, optional): If set, only process the winding group centered around this winding number.
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

    # Debug mode: filter processing range to center around debug_winding
    if debug_winding is not None:
        if debug_winding in winding_files_indices:
            debug_idx = winding_files_indices.index(debug_winding)
            # Calculate the range to center debug_winding in the middle of winding_width
            half_width = winding_width // 2
            debug_start = max(0, debug_idx - half_width)
            debug_end = min(len(winding_files_indices), debug_idx + half_width + 1)
            
            # Ensure we have exactly winding_width windings if possible
            if debug_end - debug_start < winding_width:
                if debug_start == 0:
                    debug_end = min(len(winding_files_indices), debug_start + winding_width)
                elif debug_end == len(winding_files_indices):
                    debug_start = max(0, debug_end - winding_width)
            
            debug_winding_indices = winding_files_indices[debug_start:debug_end]
            print(f"DEBUG MODE: Processing winding group {debug_winding_indices} (centered around {debug_winding})")
            print(f"DEBUG MODE: Will process range {debug_start} to {debug_end} from original indices")
            
            # Override the mesh detection to include our debug range
            first_mesh_idx = debug_start
            last_mesh_idx = debug_end - 1
        else:
            print(f"DEBUG MODE: Winding {debug_winding} not found in available windings: {winding_files_indices}")
            return flattened_winding_path

    # process windings in segments
    winding_us = None
    winding_vs = None
    for start in range(0, len(winding_files)):
        # if start < 27: # for debug, leave it for now
        #     continue
        # if start > first_mesh_idx + 2: # for debug, leave it for now
        #     continue
        # Skip if before first mesh winding or after last mesh winding
        if start < first_mesh_idx:
            print(f"Skipping winding {start} (before first mesh winding)")
            continue
        if start > last_mesh_idx:
            print(f"Finished processing: reached winding {start} (past last mesh winding)")
            break
        print(f"start {start} {winding_files_indices[start]}")
        if from_winding is not None and winding_files_indices[start] < from_winding:
            continue
        if from_winding is not None and winding_files_indices[start] - winding_width - 1 > from_winding:
            continue
        # try to load the previous flattened pointcloud
        try:
            points, uvs = load_flattened_winding(flattened_winding_path, winding_files_indices[start])
            prev_points[winding_files_indices[start]] = points
            prev_uvs_u[winding_files_indices[start]] = uvs[:, 0]
            prev_uvs_v[winding_files_indices[start]] = uvs[:, 1]
            print(f"Loaded previous flattened pointcloud for winding {winding_files_indices[start]}")
        except Exception as e:
            print(f"Warning: Could not load previous flattened pointcloud for winding {winding_files_indices[start]}: {e}")
    for start in range(0, len(winding_files)):
        if from_winding is not None and winding_files_indices[start] < from_winding:
                continue
        # Skip if before first mesh winding or after last mesh winding
        if start < first_mesh_idx:
            print(f"Skipping winding {start} (before first mesh winding)")
            continue
        if start > last_mesh_idx:
            print(f"Finished processing: reached winding {start} (past last mesh winding)")
            break
            
        # Debug mode: only process the specific debug range
        if debug_winding is not None:
            if start < debug_start or start >= debug_end:
                continue
            # In debug mode, process only one segment centered around debug_winding
            if start != debug_start:
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
                    mask = np.logical_or(np.random.rand(points_.shape[0]) < downsample_ratio, points_[:, 4].astype(bool))
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
        downsample_indices = create_flattening_downsample_indices(points, is_mesh_flags, flattening_downsample_ratio, flattening_downsample_ratio_mesh)
        
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
            
            if (start > first_mesh_idx) and len(first_winding_relative) > 0:
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
        # Print unfixed node count before solving
        ds_total_nodes = len(np.array(solver_ds.get_undeleted_indices()))
        ds_fixed_nodes = len(first_winding_relative) if 'first_winding_relative' in locals() and first_winding_relative else 0
        ds_unfixed_nodes = ds_total_nodes - ds_fixed_nodes
        print(f"Downsampled solver: {ds_unfixed_nodes} unfixed nodes out of {ds_total_nodes} total nodes ({ds_fixed_nodes} fixed)")
        solver_ds.solve_flattening(num_iterations=25000, visualize=False, 
                                   angle_tug_min=min_angle+90, angle_tug_max=max_angle-90, 
                                   z_tug_min=z_min+100, z_tug_max=z_max-100, 
                                   tug_step=0.5, zero_ranges=zero_ranges_initial)
        
        uvs_ds_initial = np.array(solver_ds.get_uvs())
        print(f"After first downsampled solve: u min={uvs_ds_initial[:,0].min()}, u max={uvs_ds_initial[:,0].max()}, v min={uvs_ds_initial[:,1].min()}, v max={uvs_ds_initial[:,1].max()}")
        
        print("Solving downsampled graph (second pass - original style)...")
        # Print unfixed node count before solving
        ds_total_nodes_2 = len(np.array(solver_ds.get_undeleted_indices()))
        ds_fixed_nodes_2 = len(first_winding_relative) if 'first_winding_relative' in locals() and first_winding_relative else 0
        ds_unfixed_nodes_2 = ds_total_nodes_2 - ds_fixed_nodes_2
        print(f"Downsampled solver (2nd pass): {ds_unfixed_nodes_2} unfixed nodes out of {ds_total_nodes_2} total nodes ({ds_fixed_nodes_2} fixed)")
        solver_ds.solve_flattening(num_iterations=75000, visualize=False, zero_ranges=zero_ranges, tug_step=0.0005, enable_spring_push_multiplier=True)
        
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

        full_downsampled_indices_mask = np.zeros(len(coords), dtype=bool)
        full_downsampled_indices_mask[valid_downsample_indices] = True
        full_valid_indices_mask = full_downsampled_indices_mask[full_valid_indices]
        relative_downsample_indices = np.where(full_valid_indices_mask)[0].tolist()
        
        # # Convert absolute valid_downsample_indices to relative indices within the full solver's valid nodes
        # relative_downsample_indices = []
        # missing_count = 0
        # for abs_idx in tqdm(valid_downsample_indices, desc="Mapping downsampled indices to full solver"):
        #     # Find where this absolute index appears in the full solver's valid indices
        #     where_found = np.where(full_valid_indices == abs_idx)[0]
        #     if len(where_found) > 0:
        #         relative_downsample_indices.append(where_found[0])
        #     else:
        #         missing_count += 1
        
        # print(f"Mapped {len(valid_downsample_indices)} absolute indices to {len(relative_downsample_indices)} relative downsample indices")
        # if missing_count > 0:
        #     print(f"WARNING: {missing_count} downsampled points not found in full solver valid indices!")
        
        # Also fix the first winding nodes in the full solver
        first_winding_relative_full = []
        first_winding_missing = 0
        if start > first_mesh_idx:
            if prev_u0 is not None and prev_u0.shape[0] == first_count:
                wrap0_idx = list(range(first_count))
                print(f"DEBUG: first_count = {first_count}, wrap0_idx (first 10): {wrap0_idx[:10]}")
                # Convert to relative indices for the full solver
                first_winding_relative_full = list(range(first_count))
            
            # for abs_idx in wrap0_idx:
            #     where_found = np.where(full_valid_indices == abs_idx)[0]
            #     if len(where_found) > 0:
            #         first_winding_relative_full.append(where_found[0])
            #     else:
            #         first_winding_missing += 1
        # if first_winding_missing > 0:
        #     print(f"WARNING: {first_winding_missing} first winding points not found in full solver valid indices!")
        
        # Combine downsampled and first winding indices for fixing
        all_fixed_indices = list(set(relative_downsample_indices + first_winding_relative_full))
        
        # Check how many nodes are fixed
        fixed_count = len(all_fixed_indices)
        total_valid = len(full_valid_indices)
        print(f"Proposed to fix {fixed_count}/{total_valid} nodes ({100*fixed_count/total_valid:.1f}%)")
        
        # # If too many nodes are fixed, reduce the number to allow more movement
        # if fixed_count > total_valid * 0.8:  # If more than 80% would be fixed
        #     print(f"WARNING: Too many nodes would be fixed ({fixed_count}/{total_valid}). Reducing fixed nodes to allow movement.")
            
        #     # Keep only a subset of the most important fixed nodes
        #     # Priority: 1) First winding nodes (if available), 2) Subset of downsampled nodes
        #     important_fixed = []
            
        #     # Always keep first winding nodes if available
        #     if len(first_winding_relative_full) > 0:
        #         important_fixed.extend(first_winding_relative_full)
        #         print(f"Keeping {len(first_winding_relative_full)} first winding nodes as fixed")
            
        #     # Add a subset of downsampled nodes (every Nth node to spread them out)
        #     remaining_budget = max(int(total_valid * 0.3), 10) - len(important_fixed)  # Target 30% fixed max
        #     if remaining_budget > 0 and len(relative_downsample_indices) > 0:
        #         step = max(1, len(relative_downsample_indices) // remaining_budget)
        #         subset_downsample = relative_downsample_indices[::step][:remaining_budget]
        #         important_fixed.extend(subset_downsample)
        #         print(f"Keeping {len(subset_downsample)} downsampled nodes as fixed (every {step}th node)")
            
        #     all_fixed_indices = important_fixed
        #     fixed_count = len(all_fixed_indices)
        #     print(f"Reduced fixed nodes to {fixed_count}/{total_valid} ({100*fixed_count/total_valid:.1f}%)")
        
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
        
        # Add debugging before the final solve call
        print("=== DEBUG: Analyzing solver state before final solve ===")
        
        # Check how many nodes are fixed
        fixed_count = len(all_fixed_indices)
        total_valid = len(full_valid_indices)
        print(f"Fixed nodes: {fixed_count}/{total_valid} ({100*fixed_count/total_valid:.1f}%)")
        
        # Check initial UV positions
        initial_uvs = np.array(solver.get_uvs())
        print(f"Initial UV range: U=[{initial_uvs[:,0].min():.2f}, {initial_uvs[:,0].max():.2f}], V=[{initial_uvs[:,1].min():.2f}, {initial_uvs[:,1].max():.2f}]")
        
        # DEBUG: Check edge distances and forces
        print("=== DEBUG: Checking edge distances ===")
        sample_edges = 0
        total_error = 0.0
        for i in range(min(100, len(neighbor_lists))):  # Sample first 100 nodes
            for j, neighbor_idx in enumerate(neighbor_lists[i]):
                if j >= 5:  # Limit to first 5 neighbors per node
                    break
                if neighbor_idx < len(initial_uvs):
                    # Calculate current distance in UV space
                    uv_dist = np.linalg.norm(initial_uvs[i] - initial_uvs[neighbor_idx])
                    # Get expected distance
                    expected_dist = distance_lists[i][j]
                    error = abs(uv_dist - expected_dist)
                    total_error += error
                    sample_edges += 1
        
        if sample_edges > 0:
            avg_error = total_error / sample_edges
            print(f"Average edge error (sampled {sample_edges} edges): {avg_error:.4f}")
            if avg_error < 1e-3:
                print("WARNING: Very small edge errors - nodes may already be well-positioned")
        else:
            print("WARNING: No valid edges found for error calculation")
        
        # Check if we have any non-fixed nodes
        if fixed_count >= total_valid * 0.95:
            print(f"WARNING: {fixed_count}/{total_valid} nodes are fixed - very few nodes can move!")
        
        # DEBUG OPTION: Uncomment the next 3 lines to test solver without any fixed nodes
        # print("DEBUG: Testing solver with NO fixed nodes")
        # solver.unfix()  # Remove all fixed constraints
        # all_fixed_indices = []
        
        print("Solving full graph (refinement with fixed anchors - with proper tug parameters)...")
        # Print unfixed node count before solving
        full_total_nodes = len(np.array(solver.get_undeleted_indices()))
        full_fixed_nodes = len(all_fixed_indices)
        full_unfixed_nodes = full_total_nodes - full_fixed_nodes
        print(f"Full solver: {full_unfixed_nodes} unfixed nodes out of {full_total_nodes} total nodes ({full_fixed_nodes} fixed)")
        # FIX: Add the missing tug parameters that were used in the downsampled solve
        solver.solve_flattening(
            num_iterations=10000, 
            visualize=False
        )
        
        # Check if positions actually changed
        final_uvs = np.array(solver.get_uvs())
        uv_change = np.linalg.norm(final_uvs - initial_uvs, axis=1)
        moved_nodes = np.sum(uv_change > 1e-6)
        print(f"Nodes that moved significantly: {moved_nodes}/{len(final_uvs)} (change > 1e-6)")
        print(f"Max UV change: {np.max(uv_change):.6f}")
        print(f"Mean UV change: {np.mean(uv_change):.6f}")
        
        undeleted_indices = np.array(solver.get_undeleted_indices())
        uvs = np.array(solver.get_uvs())

        print(f"After full refinement solve: u min={uvs[:,0].min()}, u max={uvs[:,0].max()}, v min={uvs[:,1].min()}, v max={uvs[:,1].max()}")
        
        # Process results
        winding_us = []
        winding_vs = []
        w_i = 0
        w_i_total = 0
        
        # Create directory for UV-colored pointclouds
        uv_colored_winding_path = os.path.join(base_path, "windings_uv_colored")
        
        # Calculate global UV ranges for consistent coloring across windings
        global_u_min, global_u_max = uvs[:, 0].min(), uvs[:, 0].max()
        global_v_min, global_v_max = uvs[:, 1].min(), uvs[:, 1].max()
        global_u_range = (global_u_min, global_u_max)
        global_v_range = (global_v_min, global_v_max)
        
        for i in range(len(winding_indices)):
            undeleted_indices_winding = undeleted_indices[np.logical_and(undeleted_indices >= w_i_total, undeleted_indices < w_i_total + winding_indices[i])]
            winding_u = uvs[w_i:w_i + len(undeleted_indices_winding), 0]
            winding_v = uvs[w_i:w_i + len(undeleted_indices_winding), 1]
            points_winding = points[undeleted_indices_winding]
            uvs_winding = np.column_stack([winding_u, winding_v])

            # store for next segment initialization
            wrap_nr = winding_files_indices[start + i]
            prev_uvs_u[wrap_nr] = winding_u
            prev_uvs_v[wrap_nr] = winding_v
            prev_points[wrap_nr] = points_winding
            winding_us.append(winding_u)
            winding_vs.append(winding_v)
            
            # Save UV-colored pointcloud for this winding
            try:
                save_colored_pointcloud_uv(uv_colored_winding_path, wrap_nr, points_winding, uvs_winding, 
                                        global_u_range, global_v_range)
            except Exception as e:
                print(f"Error saving UV-colored pointcloud for winding {wrap_nr}: {e}")
            
            w_i += len(undeleted_indices_winding)
            w_i_total += winding_indices[i]

        print(f"Keys in prev_uvs_u: {prev_uvs_u.keys()}")
        print(f"Keys in prev_uvs_v: {prev_uvs_v.keys()}")

        # save and free up memory of already computed windings
        clean_winding_dicts((start+1, end), flattened_winding_path, winding_files_indices, prev_uvs_u, prev_uvs_v, prev_points)

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
    if not hasattr(mesh, 'triangle_uvs'):
        if verbose:
            print("Warning: Mesh has no UV coordinates, skipping UV normalization")
        return mesh, (2048, 2048)  # Default fallback size
    
    if  len(mesh.vertices) == 0:
        if verbose:
            print("Warning: Mesh has no vertices, skipping UV normalization")
        return None  # Return None if mesh has no vertices
    
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
    for i in tqdm(range(0, len(uv_pixels), 3)):
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
                                os.path.basename(output_path).split(".")[0] + "_0.png")
        cv2.imwrite(mask_path, mask)
        
        if verbose:
            print(f"Created mask PNG: {mask_path} (size: {natural_image_size})")
    elif verbose:
        print(f"Created mask array (size: {natural_image_size}) - no file saved")
    
    return mask, mask_path

def create_mtl_file(mtl_path, texture_image_name):
    content = f"""# Material file generated by ThaumatoAnakalyptor
    newmtl default
    Ka 1.0 1.0 1.0
    Kd 1.0 1.0 1.0
    Ks 0.0 0.0 0.0
    illum 2
    d 1.0
    map_Kd {texture_image_name}
    """

    with open(mtl_path, 'w') as file:
        file.write(content)

def read_mtl_image_path(mtl_path):
    with open(mtl_path, 'r') as file:
        content = file.read()
    return content.split("map_Kd ")[1].split("\n")[0]

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

    # Create mtl file
    create_mtl_file(output_path.replace("obj", "mtl"), os.path.basename(mask_path))
    
    return success

def grid_uv_flattened(flattened_winding_path,
                       subsample_radius=3.0,
                       r_grid=100.0,
                       grid_size=50.0,
                       display=False,
                       display_downsample=0.1,
                       color_by_angle=False,
                       debug_winding=None):
    """
    Enhanced grid-based UV flattening with iterative optimization.
    
    Args:
        flattened_winding_path: Path to flattened winding files
        subsample_radius: Initial subsampling radius
        r_grid: Radius for grid optimization (default 100.0)
        grid_size: Size of grid cells for optimization (default 50.0)
        display: Whether to display results
        display_downsample: Downsample ratio for display
        color_by_angle: Whether to color by angle in display
        debug_winding: If set, only process this specific winding number (for debugging)
    """
    subsample_radius_ = 50
    winding_filtered_path = os.path.join(os.path.dirname(flattened_winding_path), "windings_filtered")
    
    # find and order all winding pointclouds
    winding_files = glob.glob(os.path.join(flattened_winding_path, "flattened_winding_*.npz"))
    print(f"Found {len(winding_files)} flattened winding pointclouds for grid sampling the uv space.")
    
    # Bring the files in order of their winding number
    winding_files_indices = sorted([int(os.path.basename(wf).split("_")[2].split(".")[0]) for wf in winding_files])
    winding_files = [os.path.join(flattened_winding_path, f"flattened_winding_{i}.npz") for i in winding_files_indices]
    
    # Filter for debug mode if specified
    if debug_winding is not None:
        if debug_winding in winding_files_indices:
            debug_index = winding_files_indices.index(debug_winding)
            winding_files_indices = [debug_winding]
            winding_files = [winding_files[debug_index]]
            print(f"DEBUG MODE: Processing only winding {debug_winding}")
        else:
            print(f"DEBUG MODE: Winding {debug_winding} not found in available windings: {winding_files_indices}")
            return
    
    print(f"Will process {len(winding_files)} flattened winding pointclouds for grid sampling the uv space.")
    
    # Load each wrap, subsample with enhanced grid optimization
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
        
        # Initial subsampling
        print(f"Subsampling {winding_files[i]} with {points.shape[0]} points")
        print(f"Min max uvs: {np.min(uvs[:, 0])}, {np.max(uvs[:, 0])}, {np.min(uvs[:, 1])}, {np.max(uvs[:, 1])}")
        
        if True:
            kept_indices = subsample_min_dist(uvs, subsample_radius)
        else:
            kept_indices = np.arange(len(uvs))
        print(f"Subsampled {winding_files[i]} to {len(kept_indices)} points")
        
        # Filter by density
        if False:
            kept_indices = filter_by_density(uvs, kept_indices, 3 * subsample_radius, int((subsample_radius**0.5)*2))
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

        # Refine by medoid
        if False:
            kept_indices = refine_by_medoid(points, uvs, kept_indices, subsample_radius)
        points_refined = points[kept_indices]
        uvs_refined = uvs[kept_indices]
        
        if display:
            display_3d_pointcloud(points_refined,
                                  color_by_angle=color_by_angle,
                                  downsample_ratio=display_downsample)

        # Apply enhanced grid-based optimization AFTER refine_by_medoid
        print(f"Starting grid-based optimization with r_grid={r_grid}, grid_size={grid_size}")
        if False:
            optimized_indices = optimize_grid_points(points, uvs, kept_indices, r_grid, grid_size)
        else:
            optimized_indices = kept_indices
        
        points_optimized = points[optimized_indices]
        uvs_optimized = uvs[optimized_indices]
        print(f"After grid optimization: {points_optimized.shape[0]} points")
        
        if display:
            # Display comparison: original, refined, optimized
            fig, axs = plt.subplots(2, 3, figsize=(15, 10))
            
            # UV plots
            axs[0, 0].scatter(uvs[:, 0], uvs[:, 1], s=1, alpha=0.5)
            axs[0, 0].set_title("Original UVs")
            axs[0, 1].scatter(uvs_refined[:, 0], uvs_refined[:, 1], s=1)
            axs[0, 1].set_title("Refined UVs")
            axs[0, 2].scatter(uvs_optimized[:, 0], uvs_optimized[:, 1], s=1)
            axs[0, 2].set_title("Grid Optimized UVs")
            
            # 3D plots
            axs[1, 0].scatter(points[:, 0], points[:, 1], s=1, alpha=0.5)
            axs[1, 0].set_title("Original Points")
            axs[1, 1].scatter(points_refined[:, 0], points_refined[:, 1], s=1)
            axs[1, 1].set_title("Refined Points")
            axs[1, 2].scatter(points_optimized[:, 0], points_optimized[:, 1], s=1)
            axs[1, 2].set_title("Grid Optimized Points")
            
            plt.tight_layout()
            plt.show()
            
            # 3D display of optimized points
            display_3d_pointcloud(points_optimized,
                                  color_by_angle=color_by_angle,
                                  downsample_ratio=display_downsample)

        # Final edge error filtering
        if False:
            # Do the edge error filtering
            optimized_indices, errors = filter_by_edge_error_mad(points[:,:3], uvs, optimized_indices, 
                                                              10 * subsample_radius_, k=200.0) # automatic threshold finding
        
        # Filter by spikes
        thr = 55.0
        if True:
            # Do the spike filtering
            for _ in range(5):
                print("Filtering by spikes...")
                spike_indices = filter_by_spikes(points[optimized_indices, :3], uvs[optimized_indices], mad_factor=15.0, angle_threshold=thr)
                thr += 2.0
                # Map spike indices back to original indices
                optimized_indices = [optimized_indices[i] for i in spike_indices]
                print(f"After spike filtering: {len(optimized_indices)} points")

        # Create final filtered points and UVs
        points_final = points[optimized_indices]
        uvs_final = uvs[optimized_indices]
        
        if display:
            display_3d_pointcloud(points_final,
                                  color_by_angle=color_by_angle,
                                  downsample_ratio=display_downsample)

        save_flattened_winding(winding_filtered_path, winding_files_indices[i], points_final, uvs_final)
        print(f"Final result for {winding_files[i]}: {points_final.shape[0]} points")

def optimize_grid_points(points, uvs, initial_indices, r_grid, grid_size):
    """
    Optimize point selection using radius-based approach with cached candidates.
    
    Args:
        points: All points (N, >=3)
        uvs: All UV coordinates (N, 2)
        initial_indices: Initial selected point indices
        r_grid: Radius for finding neighbors for error calculation
        grid_size: Radius for finding candidate replacements (UV drift constraint)
        
    Returns:
        optimized_indices: List of optimized point indices
    """
    points = points[:,:3] # extract the 3d information only
    initial_indices = np.array(initial_indices).astype(int)
    selected_indices = initial_indices.copy()

    # kdtree in uv space
    uv_tree = cKDTree(uvs)
    uv_tree_neighbours = cKDTree(uvs[initial_indices])

    # Build candidate list for each index
    candidates = []
    for i in range(len(initial_indices)):
        uv_i = uvs[initial_indices[i]]
        cs = np.array(uv_tree.query_ball_point(uv_i, grid_size))
        candidates.append(cs)

    # Build neighbour for each index
    neighbours = []
    for i in range(len(selected_indices)):
        # extract neighbours
        n = np.array(uv_tree_neighbours.query_ball_point(uvs[selected_indices[i]], r_grid))
        # remove own index
        n = n[n != i]
        neighbours.append(n)

    # Initialize error tracking for each position in selected_indices
    current_errors = np.zeros(len(selected_indices))
    
    def calculate_error_for_index(idx):
        """Calculate error for a specific index position in selected_indices"""
        current_neighbours = neighbours[idx]
        if len(current_neighbours) == 0:
            return 0.0
            
        neighbours_uvs = uvs[selected_indices[current_neighbours]]
        neighbours_points = points[selected_indices[current_neighbours]]
        current_uv = uvs[selected_indices[idx]]
        current_point = points[selected_indices[idx]]
        
        d_uvs = np.linalg.norm(neighbours_uvs - current_uv, axis=1)
        d_3d = np.linalg.norm(neighbours_points - current_point, axis=1)
        es = np.abs(d_uvs - d_3d)
        es = es[es < 50]
        return np.mean(es) if len(es) > 0 else 0.0
    
    # Calculate initial errors for all indices
    print("Calculating initial errors...")
    for i in range(len(selected_indices)):
        current_errors[i] = calculate_error_for_index(i)
    
    initial_mean_error = np.mean(current_errors)
    print(f"Initial mean error: {initial_mean_error:.4f}")

    queue = [ind for ind in range(len(selected_indices))]
    computed_indices = np.zeros(len(selected_indices), dtype=bool)
    iterations = 0
    under_n_iterations = 0
    n = 1000
    max_iter = 1000000 # break after max 1 million iterations
    while len(queue) > 0:
        iterations += 1
        if len(queue) < n:
            under_n_iterations += 1
        else:
            under_n_iterations = 0
        if under_n_iterations > 10000:
            print(f"\nBreaking out of loop after {iterations} iterations")
            break
        if iterations > max_iter:
            print(f"\nBreaking out of loop after {iterations} max iterations")
            break
        current_index = queue.pop(0)
        if computed_indices[current_index]:
            continue

        # Calculate and print progress every 100 iterations or when queue is small
        if iterations % 100 == 0 or len(queue) < 100:
            mean_error = np.mean(current_errors)
            print(f"\rQueue: {len(queue):>5}, Iter: {iterations:>7}, Mean Error: {mean_error:.6f}", end="", flush=True)
        
        current_neighbours = neighbours[current_index]
        # map to indices in selected_indices
        neighbours_uvs = uvs[selected_indices[current_neighbours]]
        neighbours_points = points[selected_indices[current_neighbours]]

        current_candidates = candidates[current_index]
        candidates_uvs = uvs[current_candidates]
        candidates_points = points[current_candidates]
        
        # calculate error for each candidate vs neighbours (abs of uv dist vs 3d dist)
        candidate_errors = []
        for j in range(len(current_candidates)):
            candidate_uv = candidates_uvs[j]
            candidate_point = candidates_points[j]
            d_uvs = np.linalg.norm(neighbours_uvs - candidate_uv, axis=1)
            d_3d = np.linalg.norm(neighbours_points - candidate_point, axis=1)
            es = np.abs(d_uvs - d_3d)
            es = es[es < 50]
            if len(es) == 0:
                candidate_error = 0
            else:
                candidate_error = np.mean(es)
            candidate_errors.append(candidate_error)
        
        # find the candidate with the smallest error
        best_candidate_idx = np.argmin(candidate_errors)
        best_candidate = current_candidates[best_candidate_idx]
        best_error = candidate_errors[best_candidate_idx]
        
        # Check if is another candidate than the current one saved in selected_indices
        if best_candidate != selected_indices[current_index]:
            selected_indices[current_index] = best_candidate
            current_errors[current_index] = best_error
            
            # Update errors for affected neighbors (since their distances changed)
            for neighbor_idx in current_neighbours:
                current_errors[neighbor_idx] = calculate_error_for_index(neighbor_idx)
            
            # Add all neighbours to queue
            for o in range(len(current_neighbours)):
                if computed_indices[current_neighbours[o]]:
                    queue.append(current_neighbours[o])
                    computed_indices[current_neighbours[o]] = False
        else:
            # Update the error even if we didn't change the candidate
            current_errors[current_index] = best_error

        computed_indices[current_index] = True

    selected_indices = np.unique(selected_indices)
    final_mean_error = np.mean(current_errors)
    print(f"\nOptimization completed:")
    print(f"Final selected indices: {len(selected_indices)} vs {len(initial_indices)} initial")
    print(f"Final mean error: {final_mean_error:.6f} (initial: {initial_mean_error:.6f})")
    print(f"Error improvement: {((initial_mean_error - final_mean_error) / initial_mean_error * 100):.2f}%")
    return selected_indices.tolist()

def get_neighbor_grids(center_grid, grid_dims, r_grid, grid_size):
    """
    Get all grid coordinates within r_grid radius of center_grid.
    
    Args:
        center_grid: (x, y) grid coordinate
        grid_dims: (width, height) grid dimensions
        r_grid: radius in UV space
        grid_size: size of each grid cell
        
    Returns:
        List of grid coordinates within radius
    """
    neighbors = []
    
    # Convert radius to grid units
    grid_radius = int(np.ceil(r_grid / grid_size))
    
    center_x, center_y = center_grid
    
    for dx in range(-grid_radius, grid_radius + 1):
        for dy in range(-grid_radius, grid_radius + 1):
            x = center_x + dx
            y = center_y + dy
            
            # Check bounds
            if 0 <= x < grid_dims[0] and 0 <= y < grid_dims[1]:
                # Check if within radius
                distance = np.sqrt(dx*dx + dy*dy) * grid_size
                if distance <= r_grid:
                    neighbors.append((x, y))
    
    return neighbors


def get_surrounding_grids(center_grid, grid_dims):
    """
    Get immediately surrounding grid cells (8-connected neighborhood).
    
    Args:
        center_grid: (x, y) grid coordinate
        grid_dims: (width, height) grid dimensions
        
    Returns:
        List of surrounding grid coordinates
    """
    neighbors = []
    center_x, center_y = center_grid
    
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue  # Skip center
                
            x = center_x + dx
            y = center_y + dy
            
            # Check bounds
            if 0 <= x < grid_dims[0] and 0 <= y < grid_dims[1]:
                neighbors.append((x, y))
    
    return neighbors


def mesh_uv_wraps(filtered_winding_path, output_dir, winding_direction=False, debug_winding=None):
    """
    For each filtered winding .npz in `filtered_winding_path`:
      1) Load points (Nx3+) and uvs (Nx2)
      2) Always flip UV U-coordinates first (left becomes right)
      3) Run Delaunay on the flipped UVs → simplices (M×3)
      4) If winding_direction is False, flip UVs back and reverse triangle winding
      5) Build an Open3D TriangleMesh with proper normal alignment
      6) Save as OBJ with UVs in `output_dir/wrap_<nr>.obj`

    Args:
        filtered_winding_path: str, folder containing
            `flattened_winding_<nr>.npz` with arrays `points, uvs`
        output_dir: str, where to write `wrap_<nr>.obj`
        winding_direction: bool, if True keep flipped UVs, if False flip back
        debug_winding: If set, only process this specific winding number (for debugging)
    """
    os.makedirs(output_dir, exist_ok=True)

    # grab all filtered wraps
    winding_flattened_files = glob.glob(os.path.join(filtered_winding_path, "flattened_winding_*.npz"))
    print(f"Found {len(winding_flattened_files)} flattened winding pointclouds to mesh uv wraps.")
    # Bring the files in order of their winding number
    winding_flattened_files_indices = sorted([int(os.path.basename(wf).split("_")[2].split(".")[0]) for wf in winding_flattened_files])
    winding_flattened_files = [os.path.join(filtered_winding_path, f"flattened_winding_{i}.npz") for i in winding_flattened_files_indices]
    
    # Filter for debug mode if specified
    if debug_winding is not None:
        if debug_winding in winding_flattened_files_indices:
            debug_index = winding_flattened_files_indices.index(debug_winding)
            winding_flattened_files_indices = [debug_winding]
            winding_flattened_files = [winding_flattened_files[debug_index]]
            print(f"DEBUG MODE: Meshing only winding {debug_winding}")
        else:
            print(f"DEBUG MODE: Winding {debug_winding} not found in available windings: {winding_flattened_files_indices}")
            return
    
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

        # ALWAYS flip U coordinates first (left becomes right)
        print(f"Always flipping U coordinates for wrap {nr} before Delaunay")
        u_max = np.max(uvs2[:, 0])
        v_max = np.max(uvs2[:, 1])
        uvs2_flipped = uvs2.copy()
        uvs2_flipped[:, 0] = u_max - uvs2[:, 0]

        # 2D Delaunay in UV space using flipped coordinates
        tri = Delaunay(uvs2_flipped)
        faces = tri.simplices  # M×3 array of vertex-indices

        # Now decide what to do based on winding_direction
        final_uvs = uvs2_flipped.copy()
        final_faces = faces.copy()
        
        if not winding_direction:
            # Flip UVs back to original orientation
            print(f"Flipping UVs back for wrap {nr} (winding_direction=False)")
            final_uvs = uvs2.copy()  # Use original UVs

            # flip uvs on other axis
            final_uvs[:, 1] = v_max - final_uvs[:, 1]

            # Reverse triangle winding order to maintain consistent orientation
            print(f"Reversing triangle winding order for wrap {nr}")
            final_faces = faces[:, [0, 2, 1]]  # Swap vertices 1 and 2 to reverse winding

        # filter out faces with min max winding angle diff > 50 deg
        winding_angles_faces = winding_angles[final_faces]
        min_angles = np.min(winding_angles_faces, axis=1)
        max_angles = np.max(winding_angles_faces, axis=1)
        winding_angle_diffs = max_angles - min_angles
        faces_mask = np.abs(winding_angle_diffs) < 50
        final_faces = final_faces[faces_mask]

        # build Open3D mesh
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(pts3)
        mesh.triangles = o3d.utility.Vector3iVector(final_faces)

        # Open3D expects triangle_uvs to be length 3*M
        # so we flatten face-by-face using final UVs
        tri_uvs = final_uvs[final_faces].reshape(-1, 2)
        mesh.triangle_uvs = o3d.utility.Vector2dVector(tri_uvs)

        # Compute triangle normals first to ensure consistent orientation
        mesh.compute_triangle_normals()
        
        # Then compute vertex normals (these will be aligned with triangle normals)
        mesh.compute_vertex_normals()

        # 5) final cleanup (you can reuse your clean_mesh here)
        mesh = clean_mesh(mesh,
                      longest_edge_pct=100,
                      area_pct=100,
                      edge_length_thresh=250)

        # Normalize UVs after cleaning and get natural image size
        res = normalize_uvs(mesh, verbose=True)
        if res is None:
            print(f"Warning: Mesh {nr} has no vertices, skipping")
            continue
        mesh, natural_image_size = res

        # Apply advanced mask-based filtering
        mesh = filter_mesh_by_mask(mesh, natural_image_size, verbose=True, show_debug=False)

        # write OBJ with UV coordinates
        out_path = os.path.join(output_dir, f"wrap_{nr}.obj")
        save_mesh_with_uvs(mesh, out_path, natural_image_size)
        print(f"Saved wrap {nr} mesh → {out_path}")

def mesh_uv_global(filtered_winding_path, output_path, winding_direction=False):
    """
    Build one seamless UV-mesh across all wraps:
      1) load every wrap_i's (points, uvs)
      2) concatenate into big all_pts, all_uvs, all_wrap_ids
      3) Always flip UV U-coordinates first (left becomes right)
      4) Delaunay on flipped all_uvs → faces
      5) If winding_direction is False, flip UVs back and reverse triangle winding
      6) keep only faces whose max(wrap_ids) - min(wrap_ids) ≤ 1
      7) angle-span / edge-area cull via clean_mesh
      8) write single OBJ with per-triangle UVs
      
    Args:
        filtered_winding_path: str, folder containing filtered winding files
        output_path: str, path to save the global mesh
        winding_direction: bool, if True keep flipped UVs, if False flip back
    """
    # 1) load and concatenate
    files = sorted(glob.glob(os.path.join(filtered_winding_path, "flattened_winding_*.npz")))
    print(f"mesh_uv_global: Found {len(files)} files to process")
    wrap_ids = []
    pts_list, uvs_list, winding_angles = [], [], []
    for fp in files:
        try:
            nr = int(os.path.basename(fp).split("_")[-1].split(".")[0])
            print(f"  Loading wrap {nr} from {os.path.basename(fp)}")
            data = np.load(fp)
            P = data["points"]   # (Ni, ≥4)
            U = data["uvs"]      # (Ni, 2)
            WA = data["points"][:, 3].astype(np.float64)
            print(f"    Loaded {len(P)} points, {len(U)} UVs")
            
            # Skip empty files
            if len(P) == 0 or len(U) == 0:
                print(f"    Skipping wrap {nr} - empty file")
                continue
                
            pts_list.append(P[:, :3])
            uvs_list.append(U)
            wrap_ids.append(np.full((len(P),), nr, dtype=int))
            winding_angles.append(WA)
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
    winding_angles = np.concatenate(winding_angles)
    print(f"Global arrays: {all_pts.shape} points, {all_uvs.shape} UVs")

    # ALWAYS flip U coordinates first (left becomes right)
    print("Always flipping U coordinates for global mesh before Delaunay")
    u_max = np.max(all_uvs[:, 0])
    all_uvs_flipped = all_uvs.copy()
    all_uvs_flipped[:, 0] = u_max - all_uvs[:, 0]

    # Validate UV coordinates before Delaunay
    print("Validating flipped UV coordinates...")
    print(f"Flipped UV min: [{np.min(all_uvs_flipped[:, 0]):.2f}, {np.min(all_uvs_flipped[:, 1]):.2f}]")
    print(f"Flipped UV max: [{np.max(all_uvs_flipped[:, 0]):.2f}, {np.max(all_uvs_flipped[:, 1]):.2f}]")
    
    # Check for invalid values
    has_nan = np.any(np.isnan(all_uvs_flipped))
    has_inf = np.any(np.isinf(all_uvs_flipped))
    if has_nan or has_inf:
        print(f"ERROR: Invalid UV values detected - NaN: {has_nan}, Inf: {has_inf}")
        return
    
    # Check for duplicate points (can cause Delaunay issues)
    unique_uvs, unique_indices = np.unique(all_uvs_flipped, axis=0, return_index=True)
    if len(unique_uvs) < len(all_uvs_flipped):
        print(f"Warning: Found {len(all_uvs_flipped) - len(unique_uvs)} duplicate UV points, using unique points only")
        all_uvs_flipped = unique_uvs
        all_uvs = all_uvs[unique_indices]  # Keep original UVs aligned
        all_pts = all_pts[unique_indices]
        all_wr = all_wr[unique_indices]
        print(f"After deduplication: {all_pts.shape} points, {all_uvs_flipped.shape} UVs")
    
    # Final check for minimum points
    if len(all_uvs_flipped) < 3:
        print(f"After validation: insufficient points for triangulation: {len(all_uvs_flipped)}")
        return
    
    print("UV validation passed, proceeding with Delaunay...")
    
    # Additional geometric checks
    print("Checking flipped UV coordinate distribution...")
    uv_range_u = np.max(all_uvs_flipped[:, 0]) - np.min(all_uvs_flipped[:, 0])
    uv_range_v = np.max(all_uvs_flipped[:, 1]) - np.min(all_uvs_flipped[:, 1])
    print(f"Flipped UV ranges: U={uv_range_u:.2f}, V={uv_range_v:.2f}")
    
    # Check if points are collinear (could cause Delaunay issues)
    if uv_range_u < 1e-6 or uv_range_v < 1e-6:
        print(f"ERROR: UV points appear to be collinear (very small range)")
        return
    
    # Try a small subset first to test Delaunay
    if len(all_uvs_flipped) > 100:
        print("Testing Delaunay with small subset...")
        test_indices = np.random.choice(len(all_uvs_flipped), size=100, replace=False)
        test_uvs = all_uvs_flipped[test_indices]
        try:
            test_tri = Delaunay(test_uvs)
            print(f"Test triangulation successful: {len(test_tri.simplices)} triangles")
        except Exception as e:
            print(f"ERROR: Test Delaunay failed: {e}")
            return
    
    print("Attempting full Delaunay triangulation on flipped UVs...")
    try:
        # Delaunay on the flipped UV set
        tri = Delaunay(all_uvs_flipped)
        print(f"Delaunay successful: {len(tri.simplices)} triangles")
    except Exception as e:
        print(f"ERROR: Delaunay triangulation failed: {e}")
        return

    faces = tri.simplices
    
    # Now decide what to do based on winding_direction
    final_uvs = all_uvs_flipped.copy()
    final_faces = faces.copy()
    
    if not winding_direction:
        # Flip UVs back to original orientation
        print("Flipping UVs back for global mesh (winding_direction=False)")
        final_uvs = all_uvs.copy()  # Use original UVs
        
        # Reverse triangle winding order to maintain consistent orientation
        print("Reversing triangle winding order for global mesh")
        final_faces = faces[:, [0, 2, 1]]  # Swap vertices 1 and 2 to reverse winding

    # filter out faces with min max winding angle diff > 50 deg
    winding_angles_faces = winding_angles[final_faces]
    min_angles = np.min(winding_angles_faces, axis=1)
    max_angles = np.max(winding_angles_faces, axis=1)
    winding_angle_diffs = max_angles - min_angles
    faces_mask = np.abs(winding_angle_diffs) < 50
    final_faces = final_faces[faces_mask]

    # only allow "neighboring-wrap" triangles
    w0 = all_wr[final_faces[:,0]]
    w1 = all_wr[final_faces[:,1]]
    w2 = all_wr[final_faces[:,2]]
    maxdiff = np.maximum.reduce([w0-w1, w1-w2, w2-w0, w1-w0, w2-w1, w0-w2])
    keep = np.abs(maxdiff) <= 1
    final_faces = final_faces[keep]

    # build Open3D mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices  = o3d.utility.Vector3dVector(all_pts.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(final_faces)

    # assign per-triangle UVs using final UVs
    tri_uvs = final_uvs[final_faces].reshape(-1, 2)
    mesh.triangle_uvs = o3d.utility.Vector2dVector(tri_uvs)

    # Compute triangle normals first to ensure consistent orientation
    mesh.compute_triangle_normals()
    
    # Then compute vertex normals (these will be aligned with triangle normals)
    mesh.compute_vertex_normals()

    # 5) final cleanup (you can reuse your clean_mesh here)
    mesh = clean_mesh(mesh,
                      longest_edge_pct=100,
                      area_pct=100,
                      edge_length_thresh=250)

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
    for i in tqdm(range(len(uv_pixels)), desc="Coloring vertex mask"):
        x, y = uv_pixels[i]
        # Flip Y coordinate for image coordinate system
        y_flipped = natural_image_size[1] - 1 - y
        # Ensure coordinates are within bounds
        if 0 <= x < natural_image_size[0] and 0 <= y_flipped < natural_image_size[1]:
            vertex_mask[y_flipped, x] = 255
    
    if show_debug:
        cv2.imshow('Step 1: Triangulated Mask', resize_for_display(first_mask))
        cv2.imshow('Step 2: Vertex-only Mask', resize_for_display(vertex_mask))
    
    # Step 3: Optimized dilation using downscaling
    radius = 200
    downscale_factor = 10

    # Calculate downscaled dimensions
    small_height = natural_image_size[1] // downscale_factor
    small_width = natural_image_size[0] // downscale_factor
    small_radius = max(1, radius // downscale_factor)  # Scale radius down, minimum 1

    print(f"Downscaling from {natural_image_size} to ({small_width}, {small_height}) for faster dilation")

    # Downscale using mean (INTER_AREA)
    small_mask = cv2.resize(vertex_mask, (small_width, small_height), interpolation=cv2.INTER_AREA)

    # Threshold: any pixel > 0 becomes 255
    small_mask[small_mask > 0] = 255

    # Create circular kernel for the smaller image
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*small_radius + 1, 2*small_radius + 1))

    # Apply dilation on the smaller image
    dilated_small = cv2.dilate(small_mask, kernel, iterations=1)

    print(f"Applied dilation with radius {small_radius} on downscaled image")

    # Step 4: Fill small islands on downscaled image (much faster)
    # Scale fill_area down by the square of the downscale factor (area scales as factor^2)
    small_fill_area = max(1, fill_area // (downscale_factor * downscale_factor))

    inverted_small = 255 - dilated_small
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inverted_small, connectivity=8)
    print(f"Found {num_labels} labels on downscaled image")

    # Find small islands on the downscaled image
    small_island_labels = np.where(stats[1:, cv2.CC_STAT_AREA] < small_fill_area)[0] + 1
    print(f"Found {len(small_island_labels)} small islands to fill on downscaled image (area < {small_fill_area})")

    # Fill small islands on the downscaled image
    if len(small_island_labels) > 0:
        small_islands_mask = np.isin(labels, small_island_labels)
        dilated_small[small_islands_mask] = 255
        print(f"Filled {len(small_island_labels)} small islands on downscaled image")

    # Upsample back to original size using nearest neighbor
    dilated_mask = cv2.resize(dilated_small, (natural_image_size[0], natural_image_size[1]), interpolation=cv2.INTER_NEAREST)

    # Ensure final mask is properly thresholded
    dilated_mask[dilated_mask > 0] = 255
    final_mask = dilated_mask.copy()

    print("Upscaled dilated and filled mask back to original size")
     
    if show_debug:
        cv2.imshow('Step 3: Dilated Mask (75px)', resize_for_display(dilated_mask))
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

def uv_to_rgb_colors(uvs, u_range=None, v_range=None):
    """
    Convert UV coordinates to RGB colors using HSV color space.
    
    Args:
        uvs: array of shape (N, 2) with UV coordinates
        u_range: tuple (min_u, max_u) for normalization, if None auto-detect
        v_range: tuple (min_v, max_v) for normalization, if None auto-detect
        
    Returns:
        colors: array of shape (N, 3) with RGB colors in range [0, 1]
    """
    uvs = np.asarray(uvs)
    
    # Determine ranges for normalization
    if u_range is None:
        u_min, u_max = uvs[:, 0].min(), uvs[:, 0].max()
    else:
        u_min, u_max = u_range
        
    if v_range is None:
        v_min, v_max = uvs[:, 1].min(), uvs[:, 1].max()
    else:
        v_min, v_max = v_range
    
    # Normalize UV coordinates to [0, 1]
    u_norm = (uvs[:, 0] - u_min) / (u_max - u_min) if u_max > u_min else np.zeros_like(uvs[:, 0])
    v_norm = (uvs[:, 1] - v_min) / (v_max - v_min) if v_max > v_min else np.zeros_like(uvs[:, 1])
    
    # Convert to HSV color space
    # U coordinate -> Hue (0-360 degrees, mapped to 0-1 for cv2)
    # V coordinate -> Saturation (0-1)
    # Value -> constant high value for good visibility
    hue = u_norm  # Map U to hue (0-1, cv2 expects 0-1 for hue)
    saturation = v_norm  # Map V to saturation
    value = np.ones_like(hue) * 0.9  # High constant value for brightness
    
    # Stack HSV components
    hsv = np.stack([hue, saturation, value], axis=1)
    
    # Convert HSV to RGB using cv2 (expects values in [0, 1] for float input)
    hsv_uint8 = (hsv * 255).astype(np.uint8)
    rgb_uint8 = cv2.cvtColor(hsv_uint8.reshape(-1, 1, 3), cv2.COLOR_HSV2RGB).reshape(-1, 3)
    
    # Convert back to float [0, 1]
    colors = rgb_uint8.astype(np.float64) / 255.0
    
    return colors

def save_colored_pointcloud_uv(winding_path, winding_nr, points, uvs, u_range=None, v_range=None):
    """
    Save a pointcloud PLY file with colors based on UV coordinates.
    
    Args:
        winding_path: Path to save the winding pointcloud
        winding_nr: Winding number for filename
        points: Array of 3D points (N, 3+)
        uvs: Array of UV coordinates (N, 2)
        u_range: Optional tuple (min_u, max_u) for color normalization
        v_range: Optional tuple (min_v, max_v) for color normalization
    """
    # Make folder if it doesn't exist
    os.makedirs(winding_path, exist_ok=True)
    
    # Generate colors based on UV coordinates
    colors = uv_to_rgb_colors(uvs, u_range, v_range)
    
    # Create Open3D pointcloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3].astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Save to PLY file
    file_path = os.path.join(winding_path, f"winding_{int(winding_nr)}_uv_colored.ply")
    o3d.io.write_point_cloud(file_path, pcd)
    print(f"Saved UV-colored pointcloud: {file_path}")

def flatten_mesh_final(mesh_path, output_path, verbose=True):
    """
    Final mesh flattening using triangle connectivity and graph solver.
    
    Args:
        mesh_path: Path to the original mesh file
        output_path: Path to save the final flattened mesh
        verbose: Whether to print progress info
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Load the original mesh
        if verbose:
            print(f"Loading original mesh from {mesh_path}")
        mtl_name = mesh_path.replace(".obj", ".mtl")
        mtl_name_bak = mtl_name + ".bak"
        try: # Fix to not load the huge png images
            os.rename(mtl_name, mtl_name_bak)
        except:
            pass
        # Load the mesh
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        try: # Fix to not load the huge png images
            os.rename(mtl_name_bak, mtl_name)
        except:
            pass
        
        if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
            print("ERROR: Mesh has no vertices or triangles")
            return False
            
        vertices = np.array(mesh.vertices)
        triangles = np.array(mesh.triangles)
        uvs = np.array(mesh.triangle_uvs).reshape(-1, 3, 2)

        image_name = read_mtl_image_path(mesh_path.replace(".obj", ".mtl"))
        image_path = os.path.join(os.path.dirname(mesh_path), image_name)
        from PIL import Image, ImageFile
        # disable PIL’s “decompression bomb” protection
        Image.MAX_IMAGE_PIXELS = None
        # allow loading even if the file is truncated/corrupt
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        image = np.array(Image.open(image_path))
        image_size = image.shape[:2][::-1]
        
        if verbose:
            print(f"Loaded mesh: {len(vertices)} vertices, {len(triangles)} triangles")
        
        # Load metadata for angles if available
        metadata_file = os.path.join(os.path.dirname(mesh_path), "mesh_metadata.pkl")
        if os.path.exists(metadata_file):
            with open(metadata_file, 'rb') as f:
                metadata = pickle.load(f)
            has_points, angles = metadata
            angles = np.reshape(angles, (-1,))
            if verbose:
                print(f"Loaded angles from metadata: min={angles.min():.1f}, max={angles.max():.1f}")
        else:
            # Generate dummy angles based on some coordinate (e.g., using atan2 of x,y)
            angles = np.degrees(np.arctan2(vertices[:, 1], vertices[:, 0]))
            if verbose:
                print("No metadata found, generated angles from vertex coordinates")
        
        # Build neighbor graph from triangle connectivity
        if verbose:
            print("Building neighbor graph from triangle connectivity...")
        
        n_vertices = len(vertices)
        neighbor_lists = [[] for _ in range(n_vertices)]
        distance_lists = [[] for _ in range(n_vertices)]
        initial_u = np.zeros(n_vertices)
        initial_v = np.zeros(n_vertices)
        # For each triangle, connect all pairs of vertices
        for i, tri in tqdm(enumerate(triangles), desc="Building neighbor graph", total=len(triangles)):
            v0, v1, v2 = tri
            # update the initial uvs
            us0, vs0 = uvs[i, 0] * image_size
            us1, vs1 = uvs[i, 1] * image_size
            us2, vs2 = uvs[i, 2] * image_size
            initial_u[v0] = us0
            initial_v[v0] = vs0
            initial_u[v1] = us1
            initial_v[v1] = vs1
            initial_u[v2] = us2
            initial_v[v2] = vs2
            
            # Calculate distances between triangle vertices
            d01 = np.linalg.norm(vertices[v0] - vertices[v1])
            d02 = np.linalg.norm(vertices[v0] - vertices[v2])
            d12 = np.linalg.norm(vertices[v1] - vertices[v2])
            
            # Add bidirectional connections
            connections = [
                (v0, v1, d01), (v1, v0, d01),
                (v0, v2, d02), (v2, v0, d02),
                (v1, v2, d12), (v2, v1, d12)
            ]
            
            for vi, vj, dist in connections:
                if vj not in neighbor_lists[vi]:
                    neighbor_lists[vi].append(vj)
                    distance_lists[vi].append(dist)
        
        if verbose:
            total_connections = sum(len(neighbors) for neighbors in neighbor_lists)
            avg_connections = total_connections / n_vertices if n_vertices > 0 else 0
            print(f"Built graph: {total_connections} total connections, {avg_connections:.1f} avg per vertex")
        
        if verbose:
            print(f"Initial UV range: U=[{initial_u.min():.1f}, {initial_u.max():.1f}], V=[{initial_v.min():.1f}, {initial_v.max():.1f}]")
        
        # Set up the solver
        if verbose:
            print("Setting up graph solver...")
        
        solver = graph_problem_gpu_py.Solver(
            neighbor_lists,
            distance_lists,
            angles,              # winding angles
            initial_u,           # initial U coordinates  
            vertices[:, 2],      # Z coordinates
            initial_v,           # initial V coordinates
            vertices[:, 0],      # X coordinates
            vertices[:, 1],      # Y coordinates
            vertices[:, 2],      # Z coordinates
            select_largest_connected_component=False
        )
        
        if verbose:
            print("Solver initialized successfully")
        
        # Run the flattening solver
        if verbose:
            print("Running flattening solver...")
            
        # Calculate angle and Z ranges for solver parameters
        angle_min, angle_max = angles.min(), angles.max()
        z_min, z_max = vertices[:, 2].min(), vertices[:, 2].max()
        zero_ranges = [(angle_min-360, angle_max+360)]
        if verbose:
            print(f"Solver ranges: angles=[{angle_min:.1f}, {angle_max:.1f}], z=[{z_min:.1f}, {z_max:.1f}]")
        
        # First solve with moderate iterations
        solver.solve_flattening(
            num_iterations=150000, 
            visualize=True,  # Disable visualization to avoid X11 errors
            zero_ranges=zero_ranges,
            tug_step=0.0005
        )
        
        # Get final UV coordinates
        final_uvs = np.zeros((len(vertices), 2))
        undeleted_indices = np.array(solver.get_undeleted_indices())
        final_uvs[undeleted_indices] = np.array(solver.get_uvs())

        # Filter out vertices that have an V value outside the 5,95 percentile -+ 500
        v_min, v_max = np.percentile(final_uvs[:, 1], [5, 95])
        v_min = v_min - 500
        v_max = v_max + 500
        mask = (final_uvs[:, 1] >= v_min) & (final_uvs[:, 1] <= v_max)
        mask_bad_indices = np.logical_not(mask)
        bad_indices = np.where(mask_bad_indices)[0]
        # remove indices from undeleted_indices by entries
        u_set = set(undeleted_indices)
        i_set = set(bad_indices)
        u_set_prime = u_set - i_set
        undeleted_indices = np.array(list(u_set_prime))
        
        if verbose:
            print(f"Solver completed: {len(undeleted_indices)} vertices retained")
            print(f"Final UV range: U=[{final_uvs[:,0].min():.2f}, {final_uvs[:,0].max():.2f}], V=[{final_uvs[:,1].min():.2f}, {final_uvs[:,1].max():.2f}]")
        
        # Create new mesh with original vertices but flattened UVs
        if verbose:
            print("Creating final mesh with flattened UVs...")
                
        if verbose:
            print(f"Final mesh: {len(vertices)} vertices, {len(triangles)} triangles")

        # Update old mesh for debug
        debug = False
        if debug:
            # Update UVs for each triangle
            triangle_uvs_old = []
            for vertex_idx in triangles.reshape(-1):
                triangle_uvs_old.append(final_uvs[vertex_idx])
            
            triangle_uvs_old = np.array(triangle_uvs_old).reshape(-1,2).astype(np.float64)
            mesh.triangle_uvs = o3d.utility.Vector2dVector(triangle_uvs_old)
            # Save the final mesh
            success = save_mesh_with_uvs(mesh, os.path.join(os.path.dirname(mesh_path), "mesh_flattened_old_debug.obj"), image_size, verbose=verbose)
        # Create the final mesh by copying the original
        final_mesh = o3d.geometry.TriangleMesh(mesh)

        # Remove triangles by index
        final_mesh.remove_triangles_by_index(bad_indices)
        triangles = np.array(final_mesh.triangles)
        
        # Update UVs for each triangle
        triangle_uvs = []
        for vertex_idx in triangles.reshape(-1):
            triangle_uvs.append(final_uvs[vertex_idx])
        
        triangle_uvs = np.array(triangle_uvs).reshape(-1,2).astype(np.float64)
        final_mesh.triangle_uvs = o3d.utility.Vector2dVector(triangle_uvs)
        print("Added triangle UVs")
        
        # Compute triangle normals (vertex normals already copied from original mesh)
        final_mesh.compute_triangle_normals()
        
        # Normalize UVs and get natural image size
        final_mesh, natural_image_size = normalize_uvs(final_mesh, verbose=verbose)
        
        # Apply mesh cleaning and filtering
        if verbose:
            print("Applying mesh cleaning and filtering...")
                
        # Save the final mesh
        success = save_mesh_with_uvs(final_mesh, output_path, natural_image_size, verbose=verbose)
        
        if success and verbose:
            print(f"Successfully saved final flattened mesh to {output_path}")

        if debug:
            # Mesh without uvs
            mesh_non_uvs = o3d.geometry.TriangleMesh()
            mesh_non_uvs.vertices = final_mesh.vertices
            mesh_non_uvs.triangles = final_mesh.triangles
            mesh_non_uvs.vertex_normals = final_mesh.vertex_normals
            mesh_non_uvs.triangle_normals = final_mesh.triangle_normals
            # save directly
            o3d.io.write_triangle_mesh(os.path.join(os.path.dirname(mesh_path), "mesh_flattened_old_debug_no_uvs.obj"), mesh_non_uvs)
            
        return success
        
    except Exception as e:
        print(f"ERROR in flatten_mesh_final: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="Refine pointcloud using mesh and downsampling")
    parser.add_argument("--mesh", required=True, help="Path to mesh file (.ply or .obj)")
    parser.add_argument("--downsample_ratio", type=float, default=1.0, help="Downsample ratio for pointclouds")
    parser.add_argument("--flattening_downsample_ratio", type=float, default=0.175, help="Downsample ratio for flattening solver first pass")
    parser.add_argument("--flattening_downsample_ratio_mesh", type=float, default=0.75, help="Downsample ratio for flattening solver first pass")
    parser.add_argument("--r_grid", type=float, default=45.0, help="Radius for grid optimization in UV space")
    parser.add_argument("--grid_size", type=float, default=15.0, help="Size of grid cells for optimization")
    parser.add_argument("--skip_precomputation", action="store_true", help="Skip precomputation of winding pointclouds")
    parser.add_argument("--skip_flattening", action="store_true", help="Skip flattening of winding pointclouds")    
    parser.add_argument("--skip_grid", action="store_true", help="Skip grid UV of flattened winding pointclouds")
    parser.add_argument("--skip_meshing", action="store_true", help="Skip meshing of flattened winding pointclouds")
    parser.add_argument("--display", action="store_true", help="Display pointclouds and meshes")
    parser.add_argument("--display_downsample", type=float, default=0.1,
                        help="Random downsample ratio for 3D display (0..1)")
    parser.add_argument("--color_by_angle", action="store_true",
                        help="Color 3D display by angular value")
    parser.add_argument("--from_winding", type=int, default=None,
                        help="Start from this winding index")
    parser.add_argument("--debug_winding", type=int, default=None,
                        help="Debug mode: only process this specific winding number")
    args = parser.parse_args()

    skip_meshing = args.skip_meshing
    skip_grid = args.skip_grid or skip_meshing
    skip_flattening = args.skip_flattening or skip_grid
    skip_precomputation = args.skip_precomputation or skip_flattening

    # Load winding direction
    winding_direction_path = os.path.join(os.path.dirname(args.mesh), "winding_direction.txt")
    try:
        winding_direction = load_winding_direction(winding_direction_path)
        print(f"Using winding direction: {winding_direction}")
    except (FileNotFoundError, ValueError) as e:
        print(f"Warning: Could not load winding direction: {e}")
        print("Defaulting to winding_direction=False (no UV flipping)")
        winding_direction = False

    # Debug mode messaging
    if args.debug_winding is not None:
        print(f"=== DEBUG MODE: Processing only winding {args.debug_winding} ===")
        print("Note: mesh_uv_global will be skipped in debug mode")

    flattened_winding_path = os.path.join(os.path.dirname(args.mesh), "windings_flattened")
    if not skip_precomputation:
        generate_winding_pointclouds(
            args.mesh,
            display=args.display,
            display_downsample=args.display_downsample,
            color_by_angle=args.color_by_angle
        )
    if not skip_flattening:
        flattened_winding_path = flatten_pointcloud(
            os.path.dirname(args.mesh),
            display=args.display,
            downsample_ratio=args.downsample_ratio,
            flattening_downsample_ratio=args.flattening_downsample_ratio,
            flattening_downsample_ratio_mesh=args.flattening_downsample_ratio_mesh,
            display_downsample=args.display_downsample,
            color_by_angle=args.color_by_angle,
            from_winding=args.from_winding,
            debug_winding=args.debug_winding
        )
    if not skip_grid:
        grid_uv_flattened(
            flattened_winding_path,
            subsample_radius=30.0,
            r_grid=args.r_grid,
            grid_size=args.grid_size,
            display=args.display,
            display_downsample=args.display_downsample,
            color_by_angle=args.color_by_angle,
            debug_winding=args.debug_winding
        )
    
    filtered_path = os.path.join(os.path.dirname(args.mesh), "windings_filtered")
    if not skip_meshing:
        mesh_uv_wraps(filtered_path, output_dir=os.path.join(os.path.dirname(filtered_path), "uv_meshes"), winding_direction=winding_direction, debug_winding=args.debug_winding)
    
    # Skip global mesh generation in debug mode
    final_mesh_path = os.path.join(os.path.dirname(filtered_path), "mesh_refined.obj")
    if args.debug_winding is None and not skip_meshing:
        mesh_uv_global(filtered_path, output_path=final_mesh_path, winding_direction=winding_direction)
    elif args.debug_winding is not None:
        print(f"Skipping mesh_uv_global due to debug mode (--debug_winding {args.debug_winding})")

    # Final flatten pointcloud mesh 
    if args.debug_winding is None:
        print("\n=== FINAL MESH FLATTENING ===")
        final_mesh_output = os.path.join(os.path.dirname(args.mesh), "mesh_final_flattened.obj")
        success = flatten_mesh_final(final_mesh_path, final_mesh_output, verbose=True)
        if success:
            print(f"Final mesh flattening completed successfully: {final_mesh_output}")
        else:
            print("Final mesh flattening failed")
    else:
        print("Skipping final mesh flattening due to debug mode")

if __name__ == "__main__":
    main()
