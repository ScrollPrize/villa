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
from scipy.spatial import cKDTree
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
    points: list of (x, y) tuples
    r: minimum allowed distance
    returns: list of indices of points (from the original list) such that
             no two kept points are closer than r
    """
    # Pair each point with its original index
    idx_pts = list(enumerate(points))  
    random.shuffle(idx_pts)             # random order
    grid = {}                           # map (i,j) -> list of accepted (idx, (x,y))
    kept_indices = []

    for idx, (x, y) in idx_pts:
        i, j = int(x // r), int(y // r)
        ok = True

        # Check accepted points in neighboring 3×3 grid cells
        for di in (-1, 0, +1):
            for dj in (-1, 0, +1):
                cell = (i + di, j + dj)
                for other_idx, (xx, yy) in grid.get(cell, ()):
                    if (xx - x)**2 + (yy - y)**2 < r*r:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break

        if ok:
            kept_indices.append(idx)
            grid.setdefault((i, j), []).append((idx, (x, y)))

    return kept_indices

def filter_by_density(points, selected_indices, r, n):
    """
    points: list of (x, y) tuples, the original dataset
    selected_indices: list of int, indices into `points` (e.g. output of subsample_min_dist_indices)
    r: float, the radius within which to count neighbors (inclusive)
    n: int, the minimum number of neighbors required (strictly greater than n)
    
    Returns: list of indices (subset of selected_indices) meeting the density criterion.
    """
    r2 = r * r
    filtered = []
    
    for idx in selected_indices:
        x0, y0 = points[idx]
        count = 0
        # count neighbors (including the point itself; adjust if you want to exclude it)
        for x1, y1 in points:
            if (x1 - x0)**2 + (y1 - y0)**2 <= r2:
                count += 1
        # keep only if strictly more than n
        if count > n:
            filtered.append(idx)
    
    return filtered

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
        points = shuffling_points_axis(points) * 4.0 # From TA to original coordinates
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

    # directory to save graphs
    graph_dir = os.path.join(base_path, "graphs")
    os.makedirs(graph_dir, exist_ok=True)
    # initialize storage for previous flattened coordinates per winding
    prev_uvs_u = {}
    prev_uvs_v = {}
    prev_points = {}

    # process windings in segments
    winding_us = None
    winding_vs = None
    for start in range(0, len(winding_files)):
        if start < 77: # for debug, leave it for now
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
            pcs.append(points_)
        min_angles = [np.min(pc[:, 3]) for pc in pcs]
        max_angles = [np.max(pc[:, 3]) for pc in pcs]
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
        solver.solve_flattening(num_iterations=20000, visualize=True, angle_tug_min=min_angle+90, angle_tug_max=max_angle-90, z_tug_min=z_min+100, z_tug_max=z_max-100, tug_step=0.2, zero_ranges=zero_ranges_initial)
        solver.solve_flattening(num_iterations=150000, visualize=True, zero_ranges=zero_ranges_initial, tug_step=-0.0005)
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

def mesh_flattened(flattened_winding_path):
    # find and order all winding pointclouds
    winding_files = glob.glob(os.path.join(flattened_winding_path, "flattened_winding_*.npz"))
    print(f"Found {len(winding_files)} flattened winding pointclouds.")
    # Bring the files in order of their winding number
    winding_files_indices = sorted([int(os.path.basename(wf).split("_")[2].split(".")[0]) for wf in winding_files])
    winding_files_indices = winding_files_indices[5:]
    winding_files = [os.path.join(flattened_winding_path, f"flattened_winding_{i}.npz") for i in winding_files_indices]
    # Load each wrap, subsample
    subsample_radius = 10.0
    for i in range(len(winding_files)):
        points, uvs = load_flattened_winding(flattened_winding_path, winding_files_indices[i])
        # subsample
        print(f"Subsampling {winding_files[i]} with {points.shape[0]} points")
        print(f"Min max uvs: {np.min(uvs[:, 0])}, {np.max(uvs[:, 0])}, {np.min(uvs[:, 1])}, {np.max(uvs[:, 1])}")
        kept_indices = subsample_min_dist(uvs, subsample_radius)
        print(f"Subsampled {winding_files[i]} to {len(kept_indices)} points")
        # filter by density
        kept_indices = filter_by_density(uvs, kept_indices, subsample_radius, 3)
        points_subsampled = points[kept_indices]
        uvs_subsampled = uvs[kept_indices]
        print(f"Subsampled {winding_files[i]} to {points_subsampled.shape[0]} points")
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

        # interactive 3d visualization of the downsampled points with matplotlib
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points_subsampled[:, 0], points_subsampled[:, 1], points_subsampled[:, 2], s=1)
        ax.set_title("Subsampled Points 3D")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Refine pointcloud using mesh and downsampling")
    parser.add_argument("--mesh", required=True, help="Path to mesh file (.ply or .obj)")
    parser.add_argument("--skip_precomputation", action="store_true", help="Skip precomputation of winding pointclouds")
    args = parser.parse_args()

    if not args.skip_precomputation:
        generate_winding_pointclouds(args.mesh)
        flattened_winding_path = flatten_pointcloud(os.path.dirname(args.mesh))
    flattened_winding_path = os.path.join(os.path.dirname(args.mesh), "windings_flattened")
    mesh_flattened(flattened_winding_path)

if __name__ == "__main__":
    main()
