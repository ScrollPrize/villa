# Giorgio Angelotti, 2025

"""
Convert a volume into skeleton annotations for WebKnossos.

This script supports two modes to load the volume:
  1. From a TIFF file on disk (using tifffile).
  2. From a remote WebKnossos dataset by providing a dataset name and a bounding box.
  
After loading the volume, the script:
  - Skeletonizes the volume using Kimimaro.
  - Extracts individual skeleton branches.
  - Converts each branch into a WebKnossos annotation (.nml).

Usage Examples:
  -- Load from TIFF:
    python voxels-to-skeleton.py --tiff_path input_volume.tiff --output_nml output_annotation.nml
  
  -- Load from WebKnossos:
    python voxels-to-skeleton.py --dataset_name my_dataset --x_start 100 --y_start 200 --z_start 50 --size 256 --token_path token.txt --output_nml output_annotation.nml
"""

import argparse
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

import numpy as np
import tifffile
import cc3d
import kimimaro

# Import WebKnossos API components.
import webknossos as wk
from webknossos import Annotation, webknossos_context, NDBoundingBox
from webknossos.geometry.vec3_int import Vec3Int  # Import the vector type

# ------------------------------
# Volume Loading Functions
# ------------------------------

def load_tiff_volume(tiff_path: str) -> np.ndarray:
    """Load a TIFF file as a 3D numpy volume using tifffile.
    
    If the TIFF is multi-page, the frames are stacked along the first axis.
    For a 2D image, a singleton z-axis is added.
    """
    vol = tifffile.imread(tiff_path)
    if vol.ndim == 2:
        vol = vol[None, ...]  # add a singleton z-axis for 2D images
    return vol

def download_volume_from_webknossos(dataset_name: str,
                                    x_start: int, y_start: int, z_start: int,
                                    size: int,
                                    token: str,
                                    wk_url: str = "http://dl.ash2txt.org:8080",
                                    organization_id: str = "Scroll_Prize") -> np.ndarray:
    """
    Download a volume chunk from a remote WebKnossos dataset.
    
    The bounding box is defined by the starting coordinates (x_start, y_start, z_start)
    and the cubic chunk size.
    
    The downloaded volume is processed and transposed into (z, y, x) order.
    """
    # Create the bounding box for the download.
    bb = wk.NDBoundingBox(
        topleft=(x_start, y_start, z_start),
        size=(size, size, size),
        index=(0, 1, 2),
        axes=('x', 'y', 'z')
    )

    with webknossos_context(url=wk_url, token=token):
        ds = wk.Dataset.open_remote(dataset_name, organization_id)
        volume_layer = ds.get_layer("Fibers")
        # Get the highest available magnification (e.g., "1")
        view = volume_layer.get_mag("1").get_view(absolute_bounding_box=bb)
        # Read, clip, and convert data to uint8.
        data = view.read()[0].astype(np.uint8)
        # Transpose to (z, y, x)
        data = np.transpose(data, (2, 1, 0))
    return data

# ------------------------------
# Skeleton Extraction Functions
# ------------------------------

def extract_branches_from_kimimaro(vertices: np.ndarray, edges: np.ndarray) -> list:
    """Extract unique branch curves from skeleton vertices and edges."""
    graph = defaultdict(list)
    for edge in edges:
        i, j = int(edge[0]), int(edge[1])
        graph[i].append(j)
        graph[j].append(i)
    
    visited_edges = set()
    branches = []
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
    
    # Remove duplicates by ensuring consistent ordering.
    unique_branches = {}
    for branch in branches:
        branch_tuple = tuple(branch if branch[0] <= branch[-1] else branch[::-1])
        unique_branches[branch_tuple] = branch
    # Convert branch indices to vertex coordinates.
    curves = [vertices[np.array(branch)] for branch in unique_branches.values()]
    return curves

def extract_skeleton(volume: np.ndarray) -> list:
    """
    Compute the skeleton of all connected components for each fiber label in the volume using Kimimaro,
    after extracting connected components with 26-connectivity using cc3d.

    For each connected component, its skeleton is computed and all branch curves are extracted.
    Each branch is returned as a tuple (curve, branch_label), where curve is a numpy array
    of shape (N, 3) with coordinates in (z, y, x) order, and branch_label is a string.
    
    Assumes:
      label 1 -> "vt"
      label 2 -> "hz"
      label 3 -> "hz/vt"
    """
    curves_all = []
    # Map fiber label numbers to their names.
    label_names = {1: "vt", 2: "hz", 3: "hz/vt"}
    
    fiber_labels = np.unique(volume)
    for fiber_label in tqdm(fiber_labels, desc="Fiber labels (hz, vt, etc.)"):
        if fiber_label == 0:
            continue

        # For each fiber label, create a binary volume where the value equals fiber_label.
        binary = volume == fiber_label if volume.dtype != np.bool_ else volume
        
        # Extract connected components with 26-connectivity.
        components = cc3d.connected_components(binary, connectivity=26)
        unique_labels = np.unique(components)
        
        # Loop over each connected component (skip background, label 0).
        for label in tqdm(unique_labels, desc="Connected components", leave=False):
            if label == 0:
                continue
            binary_component = (components == label)
            
            # Skeletonize the current component using Kimimaro.
            skels = kimimaro.skeletonize(binary_component,
                                        teasar_params={
                                            "scale": 0.66, 
                                            "const": 4,
                                        },
                                        parallel=0,
                                        fix_branching=True,
                                        fill_holes=False,
                                        dust_threshold=0,
                                        progress=False)
            
            # Extract branch curves from each skeleton.
            for skel in skels.values():
                vertices = skel.vertices
                edges = skel.edges
                if vertices.shape[0] < 2:
                    continue
                branch_curves = extract_branches_from_kimimaro(vertices, edges)
                for branch in branch_curves:
                    # Get the branch label name from our mapping.
                    branch_label = label_names.get(fiber_label, f"label_{fiber_label}")
                    curves_all.append((branch, branch_label))
    
    return curves_all

# ------------------------------
# Annotation Creation
# ------------------------------

def create_annotation_from_skeletons(branches_with_labels: list, dataset_name: str,
                                     x_start: int, y_start: int, z_start: int, size: int) -> Annotation:
    """
    Create a WebKnossos annotation from a list of skeleton branches.

    Each branch (with its label) becomes its own tree in the annotation's skeleton.
    Vertex coordinates are re-ordered from (z, y, x) to (x, y, z) as required.
    
    The tree name is prefixed with the branch label (e.g. "hz", "vt", etc.).
    """
    annotation = Annotation(name=f"fibers_{dataset_name}_{z_start:05d}z_{y_start:05d}y_{x_start:05d}x_{size}_auto",
                            dataset_name = dataset_name,
                            organization_id = "Scroll_Prize",
                            voxel_size = (7.91, 7.91, 7.91))
    
    # Create a bounding box using Vec3Int for topleft and size.
    annotation.task_bounding_box = NDBoundingBox(
        topleft=Vec3Int(x_start, y_start, z_start),
        size=Vec3Int(size, size, size),
        index=(0, 1, 2),
        axes=('x', 'y', 'z')
    )

    # Create groups based on branch labels.
    groups = {}
    # Define offset from bounding box origin.
    offset = (x_start, y_start, z_start)
    
    for i, (branch, branch_label) in enumerate(branches_with_labels):
        if branch_label not in groups:
            groups[branch_label] = annotation.skeleton.add_group(branch_label)
        group = groups[branch_label]
        
        # Create a new tree in the group for this branch.
        tree_name = f"{branch_label}_{i:05d}"
        tree = group.add_tree(tree_name)
        prev_node = None
        
        for vertex in branch:
            # Convert from (z, y, x) to (x, y, z) and round the coordinates.
            pos = tuple(int(round(c)) for c in vertex[[2, 1, 0]])
            # Add the offset so that the node is placed in its correct absolute position.
            pos = tuple(p + off for p, off in zip(pos, offset))
            node = tree.add_node(position=pos)
            if prev_node is not None:
                tree.add_edge(prev_node, node)
            prev_node = node
            
    return annotation

# ------------------------------
# Main Execution
# ------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert a volume into WebKnossos skeleton annotations (.nml)."
    )
    
    # Volume source options: either tiff_path or dataset_name (with bounding box)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--tiff_path", help="Path to the input TIFF file.")
    group.add_argument("--dataset_name", help="Name of the WebKnossos dataset to download volume from.")
    
    # WebKnossos bounding box parameters (required if --dataset_name is used).
    parser.add_argument("--x_start", type=int, help="X start coordinate for bounding box (WebKnossos mode).")
    parser.add_argument("--y_start", type=int, help="Y start coordinate for bounding box (WebKnossos mode).")
    parser.add_argument("--z_start", type=int, help="Z start coordinate for bounding box (WebKnossos mode).")
    parser.add_argument("--size", type=int, help="Cubic chunk size for bounding box (WebKnossos mode).")
    
    parser.add_argument("--token_path", default="token.txt", help="Path to a file containing the WebKnossos token.")
    parser.add_argument("--wk_url", default="http://dl.ash2txt.org:8080", help="WebKnossos URL.")
    parser.add_argument("--organization_id", default="Scroll_Prize", help="WebKnossos organization ID.")
    
    parser.add_argument("--output_nml", required=True, help="Path to save the output .nml annotation file.")
    args = parser.parse_args()
    
    # Load volume either from TIFF or from WebKnossos.
    if args.tiff_path:
        print("Loading volume from TIFF file...")
        volume = load_tiff_volume(args.tiff_path)
    else:
        # Ensure bounding box parameters are provided.
        if args.x_start is None or args.y_start is None or args.z_start is None or args.size is None:
            parser.error("When using --dataset_name, you must provide --x_start, --y_start, --z_start, and --size.")
        # Read token from file.
        try:
            with open(args.token_path, "r") as file:
                token = file.read().strip()
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
    
    print(f"Volume shape: {volume.shape}")
    
    print("Extracting skeleton from volume...")
    skeletons = extract_skeleton(volume)
    print(f"Extracted {len(skeletons)} skeleton branch(es).")
    
    if len(skeletons) == 0:
        print("No skeleton branches found. Exiting.")
        return
    
    print("Creating WebKnossos annotation from skeleton...")
    annotation = create_annotation_from_skeletons(skeletons, args.dataset_name, args.x_start, args.y_start, args.z_start, args.size)
    
    output_path = Path(args.output_nml)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving annotation to {output_path.resolve()}...")
    annotation.save(str(output_path.resolve()))

    with webknossos_context(token=token, url=args.wk_url):
        upload_url = annotation.upload()
        print("Annotation uploaded to:", upload_url)

    print("Done.")

if __name__ == '__main__':
    main()
