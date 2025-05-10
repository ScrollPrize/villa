#!/usr/bin/env python3
"""
pointcloud_mesh_refinement.py
Script to refine a pointcloud using mesh alignment and volumetric downsampling.
"""
import argparse
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

def load_pointcloud(npz_file):
    data = np.load(npz_file)
    if 'points' not in data:
        raise ValueError(f"Expected 'points' array in {npz_file}")
    points = data['points']
    if points.shape[1] < 4:
        raise ValueError("Pointcloud array must have at least 4 columns (x,y,z,angle)")
    coords = points[:, :3]
    angles = points[:, 3]
    return coords, angles

def load_mesh(mesh_file, angles_file):
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    vertices = np.asarray(mesh.vertices)
    mesh_angles = np.load(angles_file)
    if mesh_angles.shape[0] != vertices.shape[0]:
        raise ValueError("Mesh angles length does not match number of mesh vertices")
    return vertices, mesh_angles

def filter_points(coords, angles, mesh_vertices, mesh_angles, angle_tolerance, distance_threshold):
    tree = cKDTree(mesh_vertices)
    dists, idxs = tree.query(coords)
    # Compute minimal angular difference (±180 wrap)
    angle_diff = np.abs((angles - mesh_angles[idxs] + 180) % 360 - 180)
    mask = (dists <= distance_threshold) & (angle_diff <= angle_tolerance)
    return coords[mask], angles[mask]

def volumetric_downsample(coords, angles, grid_dims, max_points):
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    # Normalize to [0,1]
    scales = (coords - mins) / (maxs - mins + 1e-8)
    idx = np.floor(scales * np.array(grid_dims)).astype(int)
    for i in range(3):
        idx[:, i] = np.clip(idx[:, i], 0, grid_dims[i] - 1)
    voxel_ids = idx[:, 0] + idx[:, 1] * grid_dims[0] + idx[:, 2] * grid_dims[0] * grid_dims[1]
    selected = []
    for vid in np.unique(voxel_ids):
        v_idx = np.where(voxel_ids == vid)[0]
        if len(v_idx) <= max_points:
            selected.extend(v_idx.tolist())
        else:
            selected.extend(np.random.choice(v_idx, size=max_points, replace=False).tolist())
    selected = np.array(selected, dtype=int)
    return coords[selected], angles[selected]

def label_mesh_vertices(mesh_vertices, point_coords, distance_threshold):
    tree = cKDTree(point_coords)
    dists, _ = tree.query(mesh_vertices)
    mask = dists > distance_threshold
    return mask

def build_knn_graph(coords, k):
    tree = cKDTree(coords)
    distances, neighbors = tree.query(coords, k=k+1)
    edges = set()
    for i, nbrs in enumerate(neighbors):
        for j in nbrs[1:]:
            a, b = sorted((i, j))
            edges.add((a, b))
    return np.array(list(edges), dtype=int)

def main():
    parser = argparse.ArgumentParser(description="Refine pointcloud using mesh and downsampling")
    parser.add_argument("--mesh", required=True, help="Path to mesh file (.ply or .obj)")
    parser.add_argument("--mesh_angles", required=True, help="Path to mesh vertex angles (.npy)")
    parser.add_argument("--pointcloud", required=True, help="Path to raw pointcloud (.npz) with points[:,3]=angle)")
    parser.add_argument("--distance_threshold", type=float, default=1.0, help="Max distance to mesh to keep points")
    parser.add_argument("--angle_tolerance", type=float, default=90.0, help="Angle tolerance in degrees")
    parser.add_argument("--grid_dims", nargs=3, type=int, default=[100,100,100], help="Voxel grid dimensions")
    parser.add_argument("--max_points", type=int, default=20, help="Max points per voxel")
    parser.add_argument("--knn", type=int, default=10, help="Number of neighbors for graph")
    parser.add_argument("--output", default="refined_pointcloud.npz", help="Output NPZ file")
    args = parser.parse_args()

    # Load data
    coords, angles = load_pointcloud(args.pointcloud)
    mesh_vertices, mesh_angles = load_mesh(args.mesh, args.mesh_angles)

    # Filter by proximity and angle
    f_coords, f_angles = filter_points(
        coords, angles, mesh_vertices, mesh_angles,
        args.angle_tolerance, args.distance_threshold)

    # Volumetric downsampling
    d_coords, d_angles = volumetric_downsample(
        f_coords, f_angles, tuple(args.grid_dims), args.max_points)

    # Label mesh interpolation areas (vertices far from any point)
    mesh_mask = label_mesh_vertices(mesh_vertices, f_coords, args.distance_threshold)
    mesh_pts = mesh_vertices[mesh_mask]
    mesh_pts_angles = mesh_angles[mesh_mask]

    # Combine pointcloud and mesh points
    combined_coords = np.vstack([d_coords, mesh_pts])
    combined_angles = np.hstack([d_angles, mesh_pts_angles])
    is_mesh = np.hstack([np.zeros(len(d_coords), dtype=bool), np.ones(len(mesh_pts), dtype=bool)])

    # Build KNN graph
    edges = build_knn_graph(combined_coords, args.knn)

    # TODO: integrate with graph solver here

    # Save output
    np.savez(args.output,
             coords=combined_coords,
             angles=combined_angles,
             is_mesh=is_mesh,
             edges=edges)
    print(f"Saved refined pointcloud and graph to {args.output}")

if __name__ == "__main__":
    main()
