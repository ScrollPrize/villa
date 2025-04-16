#!/usr/bin/env python
"""
Repair broken skeleton fibers from a WebKnossos volume.

This script downloads a volume from a remote WebKnossos dataset (or loads from a TIFF file),
extracts skeleton branches using Kimimaro and cc3d, repairs broken branches using a probability
model (with parallelized candidate evaluation via a KD-tree and multiprocessing), and creates
a WebKnossos annotation (.nml) that is then uploaded.

Usage Examples:
  -- Load from TIFF:
      python repair-skeletons.py --tiff_path input_volume.tiff --output_nml output_annotation.nml
  
  -- Load from WebKnossos:
      python repair-skeletons.py --dataset_name my_dataset --x_start 100 --y_start 200 --z_start 50 --size 256 --token_path token.txt --output_nml output_annotation.nml
"""

import argparse
from pathlib import Path
from collections import defaultdict
from typing import Tuple, Dict, Any, List, Optional, Set

import numpy as np
import tifffile
import cc3d
import kimimaro
from tqdm import tqdm

# For spatial indexing and parallel processing.
from scipy.spatial import cKDTree
from multiprocessing import Pool
from functools import partial

# --- WebKnossos API Imports ---
import webknossos as wk
from webknossos import Annotation, webknossos_context, NDBoundingBox
from webknossos.geometry.vec3_int import Vec3Int

# ------------------------------
# Volume Loading Functions
# ------------------------------

def load_tiff_volume(tiff_path: str) -> np.ndarray:
    """
    Load a TIFF file as a 3D numpy volume using tifffile.
    For a 2D image, a singleton z-axis is added.
    """
    vol = tifffile.imread(tiff_path)
    if vol.ndim == 2:
        vol = vol[None, ...]
    return vol

def download_volume_from_webknossos(dataset_name: str,
                                    x_start: int, y_start: int, z_start: int,
                                    size: int,
                                    token: str,
                                    wk_url: str = "http://dl.ash2txt.org:8080",
                                    organization_id: str = "Scroll_Prize") -> np.ndarray:
    """
    Download a volume chunk from a remote WebKnossos dataset.
    The bounding box is defined by the starting coordinates and cubic size.
    The returned volume is transposed into (z, y, x) order.
    """
    bb = wk.NDBoundingBox(
        topleft=(x_start, y_start, z_start),
        size=(size, size, size),
        index=(0, 1, 2),
        axes=('x', 'y', 'z')
    )
    with webknossos_context(url=wk_url, token=token):
        ds = wk.Dataset.open_remote(dataset_name, organization_id)
        volume_layer = ds.get_layer("Fibers")
        view = volume_layer.get_mag("1").get_view(absolute_bounding_box=bb)
        data = view.read()[0].astype(np.uint8)
        data = np.transpose(data, (2, 1, 0))
    return data

# ------------------------------
# Skeleton Extraction Functions
# ------------------------------

def extract_branches_from_kimimaro(vertices: np.ndarray, edges: np.ndarray) -> List[np.ndarray]:
    """
    Extract unique branch curves from skeleton vertices and edges.
    Returns a list of numpy arrays (each of shape (N, 3)).
    """
    graph = defaultdict(list)
    for edge in edges:
        i, j = int(edge[0]), int(edge[1])
        graph[i].append(j)
        graph[j].append(i)
    
    visited_edges: Set[Tuple[int, int]] = set()
    branches: List[List[int]] = []
    endpoints = [v for v, nbrs in graph.items() if len(nbrs) == 1]
    for ep in endpoints:
        for nbr in graph[ep]:
            edge_tuple = (min(ep, nbr), max(ep, nbr))
            if edge_tuple in visited_edges:
                continue
            branch = [ep]
            current = ep
            next_v = nbr
            visited_edges.add(edge_tuple)
            while True:
                branch.append(next_v)
                if len(graph[next_v]) != 2:
                    break
                nb_list = graph[next_v]
                candidate = nb_list[0] if nb_list[0] != current else nb_list[1]
                edge_tuple2 = (min(next_v, candidate), max(next_v, candidate))
                if edge_tuple2 in visited_edges:
                    break
                visited_edges.add(edge_tuple2)
                current, next_v = next_v, candidate
            branches.append(branch)
    
    # Remove duplicates by ordering.
    unique_branches: Dict[Tuple[int, ...], List[int]] = {}
    for branch in branches:
        branch_tuple = tuple(branch if branch[0] <= branch[-1] else branch[::-1])
        unique_branches[branch_tuple] = branch
    # Convert indices to vertex coordinates.
    curves = [vertices[np.array(branch)] for branch in unique_branches.values()]
    return curves

def extract_skeleton(volume: np.ndarray) -> List[Tuple[np.ndarray, str]]:
    """
    Compute the skeleton of all connected components per fiber label using Kimimaro.
    Returns a list of tuples (curve, branch_label), where curve is an array (N,3) in (z,y,x) order.
    Assumes:
      label 1 -> "vt"
      label 2 -> "hz"
      label 3 -> "hz/vt"
    """
    curves_all: List[Tuple[np.ndarray, str]] = []
    label_names = {1: "vt", 2: "hz", 3: "hz/vt"}
    
    fiber_labels = np.unique(volume)
    for fiber_label in tqdm(fiber_labels, desc="Fiber labels"):
        if fiber_label == 0:
            continue
        binary = volume == fiber_label if volume.dtype != np.bool_ else volume
        components = cc3d.connected_components(binary, connectivity=26)
        unique_labels = np.unique(components)
        for label in tqdm(unique_labels, desc="Connected components", leave=False):
            if label == 0:
                continue
            binary_component = (components == label)
            skels = kimimaro.skeletonize(binary_component,
                                        teasar_params={"scale": 0.66, "const": 4},
                                        parallel=0,
                                        fix_branching=True,
                                        fill_holes=False,
                                        dust_threshold=0,
                                        progress=False)
            for skel in skels.values():
                vertices = skel.vertices
                edges = skel.edges
                if vertices.shape[0] < 2:
                    continue
                branch_curves = extract_branches_from_kimimaro(vertices, edges)
                for branch in branch_curves:
                    branch_label = label_names.get(fiber_label, f"label_{fiber_label}")
                    curves_all.append((branch, branch_label))
    return curves_all

# ------------------------------
# Repair (Broken Skeleton) Functions
# ------------------------------

def compute_connection_probability(pA: np.ndarray, tA: np.ndarray,
                                   pB: np.ndarray, tB: np.ndarray,
                                   d0: float = 20.0, theta0: float = 1.0) -> float:
    """
    Compute the connection probability between two endpoints.
    p = exp( - ( d/d0 + (θ_A + θ_B)/θ0 ) )
    """
    v = pB - pA
    d = np.linalg.norm(v)
    if d == 0:
        return 1.0
    v_norm = v / d
    angleA = np.arccos(np.clip(np.dot(tA, v_norm), -1.0, 1.0))
    angleB = np.arccos(np.clip(np.dot(tB, -v_norm), -1.0, 1.0))
    prob = np.exp(- (d / d0 + (angleA + angleB) / theta0))
    return prob

def compute_endpoint_info(branch: np.ndarray, endpoint_type: str = 'start', num_points: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the endpoint position and robust normalized tangent direction.
    The tangent is estimated by performing PCA (via SVD) on the first (or last) N points.
    
    Parameters:
      branch: np.ndarray of shape (N_points, 3) containing the branch coordinates.
      endpoint_type: 'start' or 'end' to indicate which endpoint to use.
      num_points: number of points to consider for the tangent estimation.
      
    Returns:
      pos: The endpoint position.
      tangent: The robust normalized tangent vector at the endpoint.
    """
    if endpoint_type == 'start':
        pos = branch[0]
        # Use the first num_points or all available if fewer.
        points = branch[:min(num_points, branch.shape[0])]
    elif endpoint_type == 'end':
        pos = branch[-1]
        # Use the last num_points or all available if fewer.
        points = branch[-min(num_points, branch.shape[0]):]
    else:
        raise ValueError("endpoint_type must be 'start' or 'end'")
    
    # If we have less than 2 points, fallback to a zero vector.
    if len(points) < 2:
        return pos, np.zeros(3)
    
    # Perform PCA using SVD on the selected points.
    points_mean = np.mean(points, axis=0)
    _, _, Vt = np.linalg.svd(points - points_mean, full_matrices=False)
    tangent = Vt[0]  # principal component

    # Ensure the tangent is directed outward from the endpoint.
    if endpoint_type == 'start':
        direction = points[-1] - points[0]
        if np.dot(tangent, direction) < 0:
            tangent = -tangent
    else:
        direction = points[0] - points[-1]
        if np.dot(tangent, direction) < 0:
            tangent = -tangent

    norm = np.linalg.norm(tangent)
    if norm > 0:
        tangent /= norm

    return pos, tangent

class Branch:
    """
    Class representing a skeleton branch.
    """
    def __init__(self, coords: np.ndarray, label: str, branch_id: str) -> None:
        self.coords = coords  # shape (N,3) in (z,y,x) order.
        self.label = label
        self.id = branch_id
        self.endpoints: Dict[str, Dict[str, np.ndarray]] = {}
        self.update_endpoints()
        
    def update_endpoints(self) -> None:
        pos_start, tan_start = compute_endpoint_info(self.coords, 'start')
        pos_end, tan_end = compute_endpoint_info(self.coords, 'end')
        self.endpoints = {
            'start': {'pos': pos_start, 'tangent': tan_start},
            'end': {'pos': pos_end, 'tangent': tan_end}
        }

def merge_branches(branch1: Branch, end_type1: str,
                   branch2: Branch, end_type2: str) -> Branch:
    """
    Merge two branches based on specified endpoints.
    Only branches with the same type (label) are merged.
    """
    coords1 = branch1.coords
    coords2 = branch2.coords
    if end_type1 == 'end' and end_type2 == 'start':
        new_coords = np.concatenate([coords1, coords2], axis=0)
    elif end_type1 == 'start' and end_type2 == 'end':
        new_coords = np.concatenate([coords2, coords1], axis=0)
    elif end_type1 == 'end' and end_type2 == 'end':
        new_coords = np.concatenate([coords1, coords2[::-1]], axis=0)
    elif end_type1 == 'start' and end_type2 == 'start':
        new_coords = np.concatenate([coords1[::-1], coords2], axis=0)
    else:
        raise ValueError("Invalid endpoint types for merging.")
    return Branch(new_coords, branch1.label, branch_id=f"{branch1.id}_{branch2.id}")

def process_candidate_pair(pair: Tuple[int, int],
                           endpoints: List[Dict[str, Any]],
                           max_distance: float,
                           d0: float,
                           theta0: float,
                           prob_threshold: float) -> Optional[Tuple[float, Dict[str, Any], Dict[str, Any]]]:
    """
    Evaluate a candidate pair (by indices) and return (prob, ep1, ep2) if threshold is met.
    """
    i, j = pair
    ep1 = endpoints[i]
    ep2 = endpoints[j]
    # Only process if fibers have the same label.
    if ep1['branch'].label != ep2['branch'].label:
        return None
    p1 = ep1['pos']
    p2 = ep2['pos']
    d = np.linalg.norm(p2 - p1)
    if d > max_distance:
        return None
    prob = compute_connection_probability(p1, ep1['tangent'], p2, ep2['tangent'], d0, theta0)
    if prob > prob_threshold:
        return (prob, ep1, ep2)
    return None

def repair_skeleton(branches: List[Branch],
                    max_distance: float = 30.0,
                    prob_threshold: float = 0.5,
                    d0: float = 15.0,
                    theta0: float = 1.5) -> List[Branch]:
    """
    Repair skeleton branches by merging endpoints with high connection probability.
    Uses a KD-tree for fast spatial querying and parallel processing for candidate evaluation.
    Also logs candidate probability statistics and the number of merges per fiber type.
    """
    # Collect endpoints.
    endpoints: List[Dict[str, Any]] = []
    for branch in branches:
        for ep_type in ['start', 'end']:
            endpoints.append({
                'branch': branch,
                'ep_type': ep_type,
                'pos': branch.endpoints[ep_type]['pos'],
                'tangent': branch.endpoints[ep_type]['tangent']
            })
    endpoint_positions = np.array([ep['pos'] for ep in endpoints])
    tree = cKDTree(endpoint_positions)
    pairs: Set[Tuple[int, int]] = tree.query_pairs(r=max_distance)
    pairs_list: List[Tuple[int, int]] = list(pairs)
    
    # Parallel processing of candidate pairs.
    process_func = partial(process_candidate_pair,
                           endpoints=endpoints,
                           max_distance=max_distance,
                           d0=d0,
                           theta0=theta0,
                           prob_threshold=prob_threshold)
    candidates: List[Tuple[float, Dict[str, Any], Dict[str, Any]]] = []
    with Pool() as pool:
        results = list(tqdm(pool.imap(process_func, pairs_list),
                            total=len(pairs_list),
                            desc="Processing candidate pairs"))
    for res in results:
        if res is not None:
            candidates.append(res)
    
    # Log candidate probability stats.
    candidate_probs = [prob for (prob, _, _) in candidates]
    if candidate_probs:
        hist, bin_edges = np.histogram(candidate_probs, bins=10)
        print("Candidate connection probability histogram:")
        for i in range(len(hist)):
            print(f"  {bin_edges[i]:.3f} - {bin_edges[i+1]:.3f}: {hist[i]} pairs")
        print(f"Mean candidate probability: {np.mean(candidate_probs):.3f}")
        print(f"Median candidate probability: {np.median(candidate_probs):.3f}")
    else:
        print("No candidate pairs above the probability threshold.")

    # Sort candidates by descending probability.
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Greedy merging.
    branch_dict: Dict[str, Branch] = {branch.id: branch for branch in branches}
    merge_count = 0
    for prob, ep1, ep2 in candidates:
        branch1: Branch = ep1['branch']
        branch2: Branch = ep2['branch']
        # Skip merging if both endpoints belong to the same branch.
        if branch1.id == branch2.id:
            continue
        if branch1.id not in branch_dict or branch2.id not in branch_dict:
            continue
        new_branch = merge_branches(branch_dict[branch1.id], ep1['ep_type'],
                                    branch_dict[branch2.id], ep2['ep_type'])
        new_branch.update_endpoints()
        del branch_dict[branch1.id]
        del branch_dict[branch2.id]
        branch_dict[new_branch.id] = new_branch
        merge_count += 1

    # Log merge stats.
    final_branches = list(branch_dict.values())
    print(f"Total merges performed: {merge_count}")
    print(f"Initial branch count: {len(branches)}, Final branch count: {len(final_branches)}")
    # Count branches per fiber type.
    type_counts: Dict[str, int] = {}
    for branch in final_branches:
        type_counts[branch.label] = type_counts.get(branch.label, 0) + 1
    print("Final branch counts per fiber type:")
    for fiber_type, count in type_counts.items():
        print(f"  {fiber_type}: {count}")

    return final_branches

def iterative_repair_skeleton(branches: List[Branch],
                              max_distance: float = 50.0,
                              prob_threshold: float = 0.5,
                              d0: float = 30.0,
                              theta0: float = 1.5,
                              max_iterations: int = 10) -> List[Branch]:
    iteration = 0
    current_branches = branches
    while iteration < max_iterations:
        print(f"Iteration {iteration+1}")
        merged_branches = repair_skeleton(current_branches,
                                          max_distance=max_distance,
                                          prob_threshold=prob_threshold,
                                          d0=d0,
                                          theta0=theta0)
        # If no merges occurred, the branch count remains unchanged.
        if len(merged_branches) == len(current_branches):
            print("No further merges; stopping iterations.")
            break
        current_branches = merged_branches
        iteration += 1
    return current_branches

# ------------------------------
# Annotation Creation
# ------------------------------

def create_annotation_from_skeletons(branches_with_labels: List[Tuple[np.ndarray, str]],
                                     dataset_name: str,
                                     x_start: int, y_start: int, z_start: int,
                                     size: int) -> Annotation:
    """
    Create a WebKnossos annotation (.nml) from a list of skeleton branches.
    Each branch becomes its own tree (grouped by label).
    Coordinates are converted from (z,y,x) to (x,y,z) and offset by the bounding box origin.
    """
    annotation = Annotation(
        name=f"fibers_{dataset_name}_{z_start:05d}z_{y_start:05d}y_{x_start:05d}x_{size}_auto",
        dataset_name=dataset_name,
        organization_id="Scroll_Prize",
        voxel_size=(7.91, 7.91, 7.91)
    )
    annotation.task_bounding_box = NDBoundingBox(
        topleft=Vec3Int(x_start, y_start, z_start),
        size=Vec3Int(size, size, size),
        index=(0, 1, 2),
        axes=('x', 'y', 'z')
    )
    groups: Dict[str, Any] = {}
    offset = (x_start, y_start, z_start)
    for i, (branch_coords, branch_label) in enumerate(branches_with_labels):
        if branch_label not in groups:
            groups[branch_label] = annotation.skeleton.add_group(branch_label)
        group = groups[branch_label]
        tree_name = f"{branch_label}_{i:05d}"
        tree = group.add_tree(tree_name)
        prev_node = None
        for vertex in branch_coords:
            # Convert from (z,y,x) to (x,y,z) and round.
            pos = tuple(int(round(c)) for c in vertex[[2, 1, 0]])
            pos = tuple(p + off for p, off in zip(pos, offset))
            node = tree.add_node(position=pos)
            if prev_node is not None:
                tree.add_edge(prev_node, node)
            prev_node = node
    return annotation

# ------------------------------
# Main Execution
# ------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repair broken skeleton fibers and upload a WebKnossos annotation (.nml)."
    )
    # Input modes: TIFF or WebKnossos dataset.
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--tiff_path", help="Path to input TIFF file.")
    group.add_argument("--dataset_name", help="Name of WebKnossos dataset to download volume from.")
    parser.add_argument("--x_start", type=int, help="X start coordinate for bounding box (WebKnossos mode).")
    parser.add_argument("--y_start", type=int, help="Y start coordinate for bounding box (WebKnossos mode).")
    parser.add_argument("--z_start", type=int, help="Z start coordinate for bounding box (WebKnossos mode).")
    parser.add_argument("--size", type=int, help="Cubic chunk size for bounding box (WebKnossos mode).")
    parser.add_argument("--token_path", default="token.txt", help="Path to file with WebKnossos token.")
    parser.add_argument("--wk_url", default="http://dl.ash2txt.org:8080", help="WebKnossos URL.")
    parser.add_argument("--organization_id", default="Scroll_Prize", help="WebKnossos organization ID.")
    
    # Repair parameters.
    parser.add_argument("--max_distance", type=float, default=30.0, help="Max distance (voxels) for connecting endpoints.")
    parser.add_argument("--prob_threshold", type=float, default=0.5, help="Min connection probability to merge endpoints.")
    parser.add_argument("--d0", type=float, default=15.0, help="Distance scaling factor for probability model.")
    parser.add_argument("--theta0", type=float, default=1.5, help="Angular scaling factor (radians) for probability model.")
    
    # Output annotation.
    parser.add_argument("--output_nml", required=True, help="Path to save the output .nml annotation file.")
    args = parser.parse_args()

    # Load volume.
    if args.tiff_path:
        print("Loading volume from TIFF file...")
        volume = load_tiff_volume(args.tiff_path)
        dataset_name = "local"  # fallback name if not using WebKnossos download.
    else:
        if args.x_start is None or args.y_start is None or args.z_start is None or args.size is None:
            parser.error("When using --dataset_name, you must provide --x_start, --y_start, --z_start, and --size.")
        try:
            with open(args.token_path, "r") as f:
                token = f.read().strip()
        except Exception as e:
            parser.error(f"Failed to read token from {args.token_path}: {e}")
        print("Downloading volume from WebKnossos...")
        volume = download_volume_from_webknossos(
            dataset_name=args.dataset_name,
            x_start=args.x_start,
            y_start=args.y_start,
            z_start=args.z_start,
            size=args.size,
            token=token,
            wk_url=args.wk_url,
            organization_id=args.organization_id
        )
        dataset_name = args.dataset_name

    print(f"Volume shape: {volume.shape}")
    
    print("Extracting skeleton from volume...")
    skeletons = extract_skeleton(volume)
    print(f"Extracted {len(skeletons)} skeleton branch(es).")
    if len(skeletons) == 0:
        print("No skeleton branches found. Exiting.")
        return

    # Convert extracted skeletons to Branch objects.
    branches: List[Branch] = []
    for i, (coords, label) in enumerate(skeletons):
        branch_obj = Branch(np.asarray(coords), label, branch_id=str(i))
        branches.append(branch_obj)

    print("Repairing broken skeleton branches...")
    repaired_branches = iterative_repair_skeleton(branches,
                                        max_distance=args.max_distance,
                                        prob_threshold=args.prob_threshold,
                                        d0=args.d0,
                                        theta0=args.theta0)
    print(f"Repaired skeleton has {len(repaired_branches)} branch(es) after merging.")

    # Prepare list of tuples (coords, label) for annotation creation.
    repaired_tuples = [(branch.coords, branch.label) for branch in repaired_branches]
    
    # Create annotation.
    if args.dataset_name:
        annotation = create_annotation_from_skeletons(
            repaired_tuples,
            dataset_name,
            x_start=args.x_start,
            y_start=args.y_start,
            z_start=args.z_start,
            size=args.size
        )
    else:
        annotation = create_annotation_from_skeletons(
            repaired_tuples,
            dataset_name,
            x_start=0,
            y_start=0,
            z_start=0,
            size=volume.shape[2]
        )
    
    output_path = Path(args.output_nml)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving annotation to {output_path.resolve()}...")
    annotation.save(str(output_path.resolve()))

    if args.dataset_name:
        with webknossos_context(token=token, url=args.wk_url):
            upload_url = annotation.upload()
            print("Annotation uploaded to:", upload_url)
    print("Done.")

if __name__ == '__main__':
    main()
