#!/usr/bin/env python3
"""
pointcloud_mesh_refinement.py
Script to generate a refined mesh using the raw pointcloud and an initial mesh.
"""
import argparse
import numpy as np
import open3d as o3d
import os
import sys
import pickle
import glob
from tqdm import tqdm

sys.path.append('ThaumatoAnakalyptor/graph_problem/build')
import graph_problem_gpu_py

def load_mesh(mesh_file):
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

def flatten_pointcloud(base_path, k_neighbors=8, winding_width=4, angle_threshold=30, angle_weight=0.2):
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
    from scipy.spatial import cKDTree

    winding_path = os.path.join(base_path, "windings")
    # find and order all winding pointclouds
    winding_files = sorted(glob.glob(os.path.join(winding_path, "winding_*.npz")))
    print(f"Found {len(winding_files)} winding pointclouds.")

    # directory to save graphs
    graph_dir = os.path.join(base_path, "graphs")
    os.makedirs(graph_dir, exist_ok=True)

    # process windings in segments
    for start in range(0, len(winding_files)):
        end = min(start + winding_width, len(winding_files))
        print(f"Processing windings {start} to {end}")
        # load and concatenate pointcloud segments (points include x,y,z,angle)
        pcs = [np.load(wf)['points'] for wf in winding_files[start:end]]
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
        distances, indices = tree.query(tree_input, k=k_neighbors + 1)
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
        neighbor_lists = [[]* len(coords)]
        distance_lists = [[]* len(coords)]
        # iterate over each point and its neighbors
        for i in range(len(coords)):
            # get valid neighbors
            valid_neighbors = neighbor_ids[i][neighbor_ids[i] != -1]
            valid_distances = neighbor_dists[i][neighbor_ids[i] != -1]
            # Add undirected edges
            neighbor_lists[i].extend(valid_neighbors)
            distance_lists[i].extend(valid_distances)
            # Add reverse edges
            for j in range(len(valid_neighbors)):
                neighbor_lists[valid_neighbors[j]].append(i)
                distance_lists[valid_neighbors[j]].append(valid_distances[j])

        # unique neighbor lists
        for i in range(len(coords)):
            _, unique_indices = np.unique(neighbor_lists[i], return_index=True)
            neighbor_lists[i] = [neighbor_lists[i][idx] for idx in unique_indices]
            distance_lists[i] = [distance_lists[i][idx] for idx in unique_indices]

        # save lists as pickle
        out_file = os.path.join(graph_dir, f"graph_{start}_{end}.pkl")
        with open(out_file, 'wb') as f:
            pickle.dump({'neighbor_ids': neighbor_lists, 'neighbor_dists': distance_lists}, f)
        print(f"Graph saved to {out_file}")

        # load graph into cpp
        solver = graph_problem_gpu_py.Solver(neighbor_lists, distance_lists, winding_angles, coords[:,2])
        # solve the graph problem
        solver.solve_flattening()
        


def main():
    parser = argparse.ArgumentParser(description="Refine pointcloud using mesh and downsampling")
    parser.add_argument("--mesh", required=True, help="Path to mesh file (.ply or .obj)")
    args = parser.parse_args()

    generate_winding_pointclouds(args.mesh)
    flatten_pointcloud(os.path.dirname(args.mesh))


if __name__ == "__main__":
    main()
