# Giorgio Angelotti 2025

import numpy as np
import argparse
import os
import open3d as o3d
from skimage import io
import kimimaro

def extract_branches_from_kimimaro(vertices: np.ndarray, edges: np.ndarray) -> list:
    """
    Given vertices (N,3) and edges (M,2) from kimimaro, build an undirected graph and
    extract all branch curves. Each branch is defined as a path starting at an endpoint
    (a vertex with degree 1) and following the unique path until a branch point (degree ≠ 2)
    is reached.
    
    Returns:
        A list of NumPy arrays, each of shape (L, 3), representing an extracted branch curve.
    """
    # Build the graph: map each vertex index to its list of neighboring indices.
    graph = {}
    for edge in edges:
        i, j = int(edge[0]), int(edge[1])
        graph.setdefault(i, []).append(j)
        graph.setdefault(j, []).append(i)
        
    visited_edges = set()
    branches = []
    # Identify endpoints: vertices with only one neighbor.
    endpoints = [v for v, nbrs in graph.items() if len(nbrs) == 1]
    
    for ep in endpoints:
        for nbr in graph[ep]:
            # Use an unordered edge representation to mark it visited.
            edge = (min(ep, nbr), max(ep, nbr))
            if edge in visited_edges:
                continue
            branch = [ep]
            current = ep
            next_v = nbr
            visited_edges.add(edge)
            while True:
                branch.append(next_v)
                # Stop if next_v is an endpoint (other than the starting one) or a branch point.
                if len(graph[next_v]) != 2:
                    break
                # Otherwise, follow the unique continuation.
                nb_list = graph[next_v]
                candidate = nb_list[0] if nb_list[0] != current else nb_list[1]
                edge2 = (min(next_v, candidate), max(next_v, candidate))
                if edge2 in visited_edges:
                    break
                visited_edges.add(edge2)
                current, next_v = next_v, candidate
            branches.append(branch)
    
    # Deduplicate branches (since a branch may be found from both endpoints).
    unique_branches = {}
    for branch in branches:
        # Create a canonical representation (order from lower to higher index).
        if branch[0] > branch[-1]:
            branch = branch[::-1]
        branch_tuple = tuple(branch)
        unique_branches[branch_tuple] = branch
    branch_list = list(unique_branches.values())
    
    # Convert branch indices to coordinates.
    curves = []
    for branch in branch_list:
        curve = vertices[np.array(branch)]
        curves.append(curve)
    return curves

def classify_curve_pca(curve: np.ndarray, z_threshold: float = 1. / np.sqrt(2)) -> str:
    """
    Classify a curve as 'vertical' or 'horizontal' using PCA on the curve coordinates.
    
    The curve is assumed to be a NumPy array of shape (N, 3) in (x, y, z) order.
    The classification compares the principal axis to the z-axis [0, 0, 1].
    
    Returns:
        "vertical" if the absolute dot product between the principal axis and [0, 0, 1]
        exceeds the threshold; "horizontal" otherwise.
    """
    if curve.shape[0] < 2:
        return "horizontal"
    
    coords = curve.astype(np.float32)
    centroid = coords.mean(axis=0)
    coords_centered = coords - centroid
    cov = np.cov(coords_centered.T)
    if np.isnan(cov).any() or np.isinf(cov).any():
        return "horizontal"
    eigvals, eigvecs = np.linalg.eig(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]
    principal_axis = eigvecs[:, 0]
    principal_axis /= np.linalg.norm(principal_axis)
    z_axis = np.array([1, 0, 0], dtype=np.float32)
    cos_angle = abs(np.dot(principal_axis, z_axis))
    return "vertical" if cos_angle > z_threshold else "horizontal"

def extract_skeleton_from_tif(tif_file: str, original_file: str, z_threshold: float = 1. / np.sqrt(2), label: int = 1, fiber_type: str = "hz") -> dict:
    """
    Loads a .tif volume, thresholds it, and performs 3D skeletonization using kimimaro.
    
    For each skeleton (returned as a set of vertices and edges), extracts all branch curves
    and classifies each as vertical or horizontal.
    
    Returns:
        A dictionary with keys "vertical" and "horizontal". Each value is a list of NumPy arrays,
        where each array (of shape (N, 3)) represents an extracted skeleton curve.
    """
    volume = io.imread(tif_file)
    label_volume = io.imread(original_file)

    if fiber_type == "hz":
        volume = np.logical_and((label_volume == label), (volume == 2)).astype(np.uint8)
    else:
        volume = np.logical_and((label_volume == label), (volume == 1)).astype(np.uint8)
    # Ensure binary volume (nonzero is foreground).
    if volume.dtype != np.bool_:
        volume = volume > 0

    print("Performing kimimaro skeletonization...")
    # Use kimimaro to skeletonize.
    skels = kimimaro.skeletonize(volume, parallel=0,
                                 fix_branching=True,
                                 fill_holes=True,
                                 dust_threshold=0)

    curves = {"vertical": [], "horizontal": []}
    
    # Iterate over each skeleton returned by kimimaro.
    for skel_id, skel in skels.items():
        # Access attributes from the Skeleton object.
        vertices = skel.vertices  # Shape: (N, 3)
        edges = skel.edges        # Shape: (M, 2)
        if vertices.shape[0] < 2:
            continue
        # Extract all branch curves.
        branch_curves = extract_branches_from_kimimaro(vertices, edges)
        for curve in branch_curves:
            if curve.shape[0] > 1:
                label_curve = classify_curve_pca(curve, z_threshold)
                curves[label_curve].append(curve)
    return curves

def visualize_curves(curves_by_label: dict):
    """
    Visualizes skeleton curves using Open3D.
    
    Each curve is rendered as a LineSet with the following color mapping:
      - Vertical curves: red.
      - Horizontal curves: blue.
    """
    line_sets = []
    color_map = {"vertical": [1, 0, 0], "horizontal": [0, 0, 1]}
    
    for label, curves in curves_by_label.items():
        for curve in curves:
            num_pts = curve.shape[0]
            edges = [[j, j + 1] for j in range(num_pts - 1)]
            ls = o3d.geometry.LineSet()
            ls.points = o3d.utility.Vector3dVector(curve)
            ls.lines = o3d.utility.Vector2iVector(edges)
            ls.colors = o3d.utility.Vector3dVector([color_map[label] for _ in range(len(edges))])
            line_sets.append(ls)
    
    o3d.visualization.draw_geometries(line_sets, window_name="Extracted Skeleton Curves")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract and classify skeleton curves from a .tif volume using kimimaro skeletonization."
    )
    parser.add_argument("--tif", type=str, required=True, help="Path to the .tif volume file.")
    parser.add_argument("--visualize", action="store_true", help="Visualize the extracted skeleton curves.")
    args = parser.parse_args()

    if not os.path.isfile(args.tif):
        raise FileNotFoundError(f"File not found: {args.tif}")

    print(f"Loading volume from: {args.tif}")
    curves_dict = extract_skeleton_from_tif(args.tif)
    
    n_vert = len(curves_dict.get("vertical", []))
    n_horz = len(curves_dict.get("horizontal", []))
    print(f"Extracted {n_vert} vertical curve(s) and {n_horz} horizontal curve(s).")
    
    
    if args.visualize:
        print("Visualizing skeleton curves...")
        visualize_curves(curves_dict)
