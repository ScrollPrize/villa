# Giorgio Angelotti - 2025
import os
# Force Cupy to use C++17.
os.environ["CUPY_NVRTC_FLAGS"] = "--std=c++17"
os.environ["CCCL_IGNORE_DEPRECATED_CPP_DIALECT"] = "1"

import argparse
from pathlib import Path
import numpy as np
import tifffile
from collections import defaultdict

# --------------------------
# Helper: Parse Label Filename
# --------------------------
def parse_label_filename(label_path: str):
    """
    Parse a label TIFF filename of the form: sX_zzzzz_yyyyy_xxxxx_size.tif
    Returns:
        dataset_id (str): e.g., "s1"
        start_z (int)
        start_y (int)
        start_x (int)
        size (int)
    """
    base = Path(label_path).stem  # Remove directory and extension.
    # Expect format: sX_zzzzz_yyyyy_xxxxx_size
    parts = base.split('_')
    if len(parts) != 5:
        raise ValueError("Label filename must have 5 parts separated by underscores (e.g. s1_00497_01497_03997_256)")
    dataset_id = parts[0]
    start_z = int(parts[1])
    start_y = int(parts[2])
    start_x = int(parts[3])
    size = int(parts[4])
    return dataset_id, start_z, start_y, start_x, size

def determine_dataset_name(dataset_id: str) -> str:
    """
    Determine the dataset name for predictions and annotation upload based on the dataset id.
    For example, if dataset_id is "s1", returns "s1a-with-fibers-hzvt-05032025"; if "s5", returns "s5-with-fibers".
    Otherwise, you can customize this function.
    """
    if dataset_id.lower() == "s1":
        return "s1a-with-fibers-hzvt-05032025"
    elif dataset_id.lower() == "s5":
        return "s5-with-fibers"
    else:
        # Default: append "-with-fibers" to the dataset id.
        return f"{dataset_id}-with-fibers"

# --------------------------
# Remote Predictions Loading
# --------------------------
import webknossos as wk
from webknossos import webknossos_context, NDBoundingBox
from webknossos.geometry.vec3_int import Vec3Int

def download_volume_from_webknossos(dataset_name: str,
                                    x_start: int, y_start: int, z_start: int,
                                    size: int,
                                    token: str,
                                    wk_url: str = "http://dl.ash2txt.org:8080",
                                    organization_id: str = "Scroll_Prize") -> np.ndarray:
    """
    Download a volume chunk from a remote WebKnossos dataset.
    Reads the "Fibers" layer at magnification "1", converts data to uint8,
    and transposes it to (z, y, x) order.
    """
    bb = NDBoundingBox(
        topleft=(x_start, y_start, z_start),
        size=(size, size, size),
        index=(0, 1, 2),
        axes=('x', 'y', 'z')
    )
    with webknossos_context(token=token, url=wk_url):
        ds = wk.Dataset.open_remote(dataset_name, organization_id)
        volume_layer = ds.get_layer("Fibers")
        view = volume_layer.get_mag("1").get_view(absolute_bounding_box=bb)
        data = view.read()[0].astype(np.uint8)
        data = np.transpose(data, (2, 1, 0))
    return data

# --------------------------
# Skeleton Extraction Functions
# --------------------------
import cc3d
import kimimaro

def extract_branches_from_kimimaro(vertices: np.ndarray, edges: np.ndarray) -> list:
    """
    Extract unique branch curves from skeleton vertices and edges.
    
    Returns:
        List[np.ndarray]: List of branch curves.
    """
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
    unique_branches = {}
    for branch in branches:
        branch_tuple = tuple(branch if branch[0] <= branch[-1] else branch[::-1])
        unique_branches[branch_tuple] = branch
    curves = [vertices[np.array(branch)] for branch in unique_branches.values()]
    return curves

def extract_skeleton_from_component(component_mask: np.ndarray) -> list:
    """
    Skeletonize a single connected component (binary volume) using Kimimaro and extract branch curves.
    
    Returns:
        List[np.ndarray]: List of branch curves.
    """
    skeletons = []
    skels = kimimaro.skeletonize(component_mask,
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
        branches = extract_branches_from_kimimaro(vertices, edges)
        for branch in branches:
            skeletons.append(branch)
    return skeletons

def extract_non_overlap_skeletons(prediction: np.ndarray, label: np.ndarray) -> dict:
    """
    For each predicted fiber class (1 for vt and 2 for hz), compute connected components in the 
    corresponding binary volume. For each component, if there is zero overlap with any nonzero voxel 
    in the label volume, skeletonize that component.
    
    Returns:
        dict: Mapping from fiber type ("vt" or "hz") to a list of skeleton branches.
    """
    skeletons_by_type = {"vt": [], "hz": []}
    for fiber_class in [1, 2]:
        # Create a binary volume for the current fiber class.
        pred_binary = (prediction == fiber_class).astype(np.uint8)
        if np.count_nonzero(pred_binary) == 0:
            continue
        # Compute connected components for this class.
        pred_cc = cc3d.connected_components(pred_binary, connectivity=26)
        unique_components = np.unique(pred_cc)
        for comp in unique_components:
            if comp == 0:
                continue
            comp_mask = (pred_cc == comp)
            # Skip if any voxel in this component overlaps with a nonzero voxel in the label.
            if np.any(label[comp_mask] != 0):
                continue
            # Skeletonize this component.
            branches = extract_skeleton_from_component(comp_mask.astype(np.uint8))
            if fiber_class == 1:
                skeletons_by_type["vt"].extend(branches)
            elif fiber_class == 2:
                skeletons_by_type["hz"].extend(branches)
    return skeletons_by_type

def extract_label_skeletons(label: np.ndarray) -> dict:
    """
    Extract skeletons from the label volume separately for fiber types 1 (vt) and 2 (hz).
    
    Returns:
        dict: Mapping from fiber type ("vt" or "hz") to a list of skeleton branches.
    """
    skeletons_by_type = {"vt": [], "hz": []}
    vt_mask = (label == 1).astype(np.uint8)
    vt_cc = cc3d.connected_components(vt_mask, connectivity=26)
    vt_branches = []
    for comp in np.unique(vt_cc):
        if comp == 0:
            continue
        comp_mask = (vt_cc == comp)
        branches = extract_skeleton_from_component(comp_mask)
        vt_branches.extend(branches)
    skeletons_by_type["vt"] = vt_branches

    hz_mask = (label == 2).astype(np.uint8)
    hz_cc = cc3d.connected_components(hz_mask, connectivity=26)
    hz_branches = []
    for comp in np.unique(hz_cc):
        if comp == 0:
            continue
        comp_mask = (hz_cc == comp)
        branches = extract_skeleton_from_component(comp_mask)
        hz_branches.extend(branches)
    skeletons_by_type["hz"] = hz_branches
    return skeletons_by_type

# --------------------------
# Annotation Creation
# --------------------------
def create_annotation(dataset_name: str, x_start: int, y_start: int, z_start: int, size: int) -> wk.Annotation:
    """
    Create a new WebKnossos annotation with a specified bounding box.
    """
    annotation = wk.Annotation(
        name=f"non_overlap_preds_{dataset_name}_{z_start:05d}z_{y_start:05d}y_{x_start:05d}x_{size}_auto",
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
    return annotation

def add_fiber_group(annotation: wk.Annotation, fiber_type: str, label_branches: list, non_overlap_branches: list, offset: tuple):
    """
    Create a fiber-type group with two subgroups: 'labels' and 'non_overlap'.
    
    Trees in each subgroup are assigned distinct colors.
    
    Parameters:
        annotation : wk.Annotation
            The WebKnossos annotation object.
        fiber_type : str
            Fiber type identifier ("vt" or "hz").
        label_branches : list
            List of skeleton branches from the label volume.
        non_overlap_branches : list
            List of skeleton branches from non-overlapping prediction components.
        offset : tuple
            (x_start, y_start, z_start) to offset node coordinates.
    """
    fiber_group = annotation.skeleton.add_group(fiber_type)
    labels_group = fiber_group.add_group("labels")
    non_overlap_group = fiber_group.add_group("non_overlap")
    
    if fiber_type == "vt":
        labels_color = (0, 0, 1, 1)      # Blue for vt/labels.
        non_overlap_color = (0, 1, 1, 1) # Cyan for vt/non_overlap.
    elif fiber_type == "hz":
        labels_color = (1, 0, 0, 1)      # Red for hz/labels.
        non_overlap_color = (1, 0, 1, 1) # Magenta for hz/non_overlap.
    else:
        labels_color = None
        non_overlap_color = None

    for i, branch in enumerate(label_branches):
        tree_name = f"{fiber_type}_labels_{i:05d}"
        tree = labels_group.add_tree(tree_name, color=labels_color)
        prev_node = None
        for vertex in branch:
            pos = tuple(int(round(c)) for c in vertex[[2, 1, 0]])
            pos = tuple(p + off for p, off in zip(pos, offset))
            node = tree.add_node(position=pos)
            if prev_node is not None:
                tree.add_edge(prev_node, node)
            prev_node = node

    for i, branch in enumerate(non_overlap_branches):
        tree_name = f"{fiber_type}_non_overlap_{i:05d}"
        tree = non_overlap_group.add_tree(tree_name, color=non_overlap_color)
        prev_node = None
        for vertex in branch:
            pos = tuple(int(round(c)) for c in vertex[[2, 1, 0]])
            pos = tuple(p + off for p, off in zip(pos, offset))
            node = tree.add_node(position=pos)
            if prev_node is not None:
                tree.add_edge(prev_node, node)
            prev_node = node

# --------------------------
# Main Execution
# --------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Skeletonize connected components in the predictions that do not overlap with labels, "
                    "divide them by fiber type (vt for label==1, hz for label==2), and create/upload a WebKnossos annotation "
                    "with separate subgroups for label skeletons and non-overlap prediction skeletons. "
                    "The spatial parameters and dataset name are extracted from the label TIFF filename."
    )
    parser.add_argument("--label_tif", required=True, help="Path to label .tif file (e.g., s1_00497_01497_03997_256.tif)")
    parser.add_argument("--num_classes", type=int, default=None,
                        help="Number of classes (if not provided, computed from max value + 1)")
    parser.add_argument("--token_path", default="token.txt", help="Path to file containing WebKnossos token")
    parser.add_argument("--wk_url", default="http://dl.ash2txt.org:8080", help="WebKnossos URL")
    parser.add_argument("--organization_id", default="Scroll_Prize", help="WebKnossos organization ID")
    parser.add_argument("--output_nml", required=True, help="Path to save output .nml annotation file")
    
    args = parser.parse_args()
    
    # Parse spatial parameters from the label filename.
    try:
        dataset_id, start_z, start_y, start_x, size = parse_label_filename(args.label_tif)
    except Exception as e:
        print(f"Error parsing label filename: {e}")
        return
    # Determine the dataset name for predictions and annotation upload.
    pred_dataset_name = determine_dataset_name(dataset_id)
    
    print(f"Parsed from filename: dataset_id = {dataset_id}, start_z = {start_z}, start_y = {start_y}, start_x = {start_x}, size = {size}")
    print(f"Using prediction/annotation dataset name: {pred_dataset_name}")
    
    # Load label volume.
    label = tifffile.imread(args.label_tif)
    
    # Download predictions volume remotely.
    try:
        with open(args.token_path, "r") as f:
            token = f.read().strip()
    except Exception as e:
        print(f"Failed to read token from {args.token_path}: {e}")
        return
    print("Downloading predictions volume from WebKnossos dataset...")
    prediction = download_volume_from_webknossos(
        dataset_name=pred_dataset_name,
        x_start=start_x,
        y_start=start_y,
        z_start=start_z,
        size=size,
        token=token,
        wk_url=args.wk_url,
        organization_id=args.organization_id
    )
    
    num_classes = int(max(label.max(), prediction.max())) + 1 if args.num_classes is None else args.num_classes
    print(f"Using {num_classes} classes.")
    
    # Extract skeletons from the label volume (for vt and hz).
    print("Extracting skeletons from label volume...")
    label_skel_dict = extract_label_skeletons(label)
    print(f"Extracted {len(label_skel_dict['vt'])} vt label skeleton branches and {len(label_skel_dict['hz'])} hz label skeleton branches.")
    
    # Extract skeletons from prediction connected components that do not overlap any label.
    print("Extracting skeletons from non-overlapping prediction components...")
    non_overlap_skel_dict = extract_non_overlap_skeletons(prediction, label)
    print(f"Extracted {len(non_overlap_skel_dict['vt'])} vt non-overlap skeleton branches and {len(non_overlap_skel_dict['hz'])} hz non-overlap skeleton branches.")
    
    if not ((label_skel_dict["vt"] or non_overlap_skel_dict["vt"]) or (label_skel_dict["hz"] or non_overlap_skel_dict["hz"])):
        print("No skeleton branches found. Exiting.")
        return
    
    print("Creating WebKnossos annotation...")
    annotation = create_annotation(pred_dataset_name, start_x, start_y, start_z, size)
    offset = (start_x, start_y, start_z)
    
    if label_skel_dict["vt"] or non_overlap_skel_dict["vt"]:
        print("Adding 'vt' fiber group (label==1)...")
        add_fiber_group(annotation, "vt", label_skel_dict["vt"], non_overlap_skel_dict["vt"], offset)
    if label_skel_dict["hz"] or non_overlap_skel_dict["hz"]:
        print("Adding 'hz' fiber group (label==2)...")
        add_fiber_group(annotation, "hz", label_skel_dict["hz"], non_overlap_skel_dict["hz"], offset)
    
    output_path = Path(args.output_nml)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving annotation to {output_path.resolve()}...")
    annotation.save(str(output_path.resolve()))
    
    print("Uploading annotation to WebKnossos...")
    with webknossos_context(token=token, url=args.wk_url):
        upload_url = annotation.upload()
        print("Annotation uploaded to:", upload_url)
    
    print("Done.")

if __name__ == '__main__':
    main()
