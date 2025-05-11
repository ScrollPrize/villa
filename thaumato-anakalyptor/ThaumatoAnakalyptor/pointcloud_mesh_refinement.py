#!/usr/bin/env python3
"""
pointcloud_mesh_refinement.py
Script to refine a pointcloud using mesh alignment and volumetric downsampling.
"""
import argparse
import numpy as np
import open3d as o3d
import os
import pickle
import glob
from tqdm import tqdm

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
    os.makedirs(os.path.dirname(winding_path), exist_ok=True)
    # Save the winding pointcloud
    file_path = os.path.join(winding_path, f"winding_{int(winding_nr)}.npz")
    points_prev = np.zeros((0, 4), dtype=np.float16)
    if os.path.exists(file_path):
        points_prev = np.load(file_path)
    points = np.concatenate((points_prev, points), axis=0)
    points = np.unique(points, axis=0)
    np.savez_compressed(file_path, points=points)

def generate_winding_pointclouds(mesh_path):
    vertices, (has_points, angles) = load_mesh(mesh_path)
    print(f"Shape of vertices: {vertices.shape} and of angles: {angles.shape}")
    points_mesh = np.concatenate((vertices, angles), axis=1)
    points_mesh = points_mesh[np.logical_not(has_points)]

    base_path = os.path.dirname(mesh_path)
    slabs = glob.glob(os.path.join(base_path, "points_selected_", "*.npz"))
    print("Preparing the winding pointclouds...")
    for slab_path in tqdm(slabs, desc="Preprocessing slabs"):
        points = load_pointcloud_slab(slab_path).astype(np.float16)
        print(f"Shape of points: {points.shape} and of points_mesh: {points_mesh.shape}")
        points = np.concatenate((points, points_mesh), axis=0)
        min_angle = np.min(points[:, 3])
        max_angle = np.max(points[:, 3])
        min_winding = int(np.floor(min_angle / 360.0))
        max_winding = int(np.ceil(max_angle / 360.0))
        for winding_nr in range(min_winding, max_winding + 1):
            winding_angle = winding_nr * 360.0
            winding_points = points[np.logical_and(points[:, 3] >= winding_angle, points[:, 3] < winding_angle + 360.0)]
            if len(winding_points) > 0:
                save_winding_pointcloud(os.path.join(base_path, "winding"), winding_nr, winding_points)

def main():
    parser = argparse.ArgumentParser(description="Refine pointcloud using mesh and downsampling")
    parser.add_argument("--mesh", required=True, help="Path to mesh file (.ply or .obj)")
    args = parser.parse_args()

    generate_winding_pointclouds(args.mesh)


if __name__ == "__main__":
    main()
