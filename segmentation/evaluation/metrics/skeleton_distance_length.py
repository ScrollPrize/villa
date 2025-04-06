from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import kimimaro
from scipy.stats import wasserstein_distance
from collections import defaultdict
from PIL import Image

try:
    import wandb
except ImportError:
    wandb = None

# Global accumulators for branch lengths over all samples.
AGGREGATE_LABEL_LENGTHS = []
AGGREGATE_PRED_LENGTHS = []
# Global counter to ensure unique keys for per-sample images.
SAMPLE_COUNT = 0

def extract_branches_from_kimimaro(vertices: np.ndarray, edges: np.ndarray) -> list:
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

def extract_skeleton(volume: np.ndarray) -> list:
    binary = volume > 0 if volume.dtype != np.bool_ else volume
    skels = kimimaro.skeletonize(binary, parallel=0, fix_branching=True,
                                 fill_holes=False, dust_threshold=0, progress=False)
    curves = []
    for skel in skels.values():
        vertices = skel.vertices
        edges = skel.edges
        if vertices.shape[0] < 2:
            continue
        branch_curves = extract_branches_from_kimimaro(vertices, edges)
        curves.extend(branch_curves)
    return curves

def compute_curve_length(curve: np.ndarray) -> float:
    if curve.shape[0] < 2:
        return 0.0
    return np.sum(np.linalg.norm(np.diff(curve, axis=0), axis=1))

def compute(label: np.ndarray, prediction: np.ndarray, **hyperparams) -> float:
    global SAMPLE_COUNT
    SAMPLE_COUNT += 1

    distance_metric = hyperparams.get("distance_metric", "symkl").lower()
    bins = hyperparams.get("bins", 20)
    hist_range = hyperparams.get("hist_range", None)
    epsilon = hyperparams.get("epsilon", 1e-8)
    output_folder = hyperparams.get("output_folder", ".")
    
    label_curves = extract_skeleton(label)
    pred_curves = extract_skeleton(prediction)
    
    label_lengths = np.array([compute_curve_length(curve) for curve in label_curves if curve.shape[0] > 1])
    pred_lengths = np.array([compute_curve_length(curve) for curve in pred_curves if curve.shape[0] > 1])
    
    global AGGREGATE_LABEL_LENGTHS, AGGREGATE_PRED_LENGTHS
    AGGREGATE_LABEL_LENGTHS += label_lengths.tolist()
    AGGREGATE_PRED_LENGTHS += pred_lengths.tolist()
    
    if label_lengths.size == 0 or pred_lengths.size == 0:
        print("Warning: No skeleton curves extracted from one or both volumes.")
        return float('nan')
    
    if hist_range is None:
        combined = np.concatenate([label_lengths, pred_lengths])
        min_val, max_val = combined.min(), combined.max()
        if min_val == max_val:
            max_val = min_val + 1
        hist_range = (min_val, max_val)
    
    if distance_metric == "symkl":
        label_counts, bin_edges = np.histogram(label_lengths, bins=bins, range=hist_range)
        pred_counts, _ = np.histogram(pred_lengths, bins=bins, range=hist_range)
        label_prob = label_counts.astype(np.float64) + epsilon
        pred_prob = pred_counts.astype(np.float64) + epsilon
        label_prob /= label_prob.sum()
        pred_prob /= pred_prob.sum()
        
        kl_div = np.sum(label_prob * np.log(label_prob / pred_prob)) + \
                 np.sum(pred_prob * np.log(pred_prob / label_prob))
        distance = kl_div
        
        viz_label = label_prob
        viz_pred = pred_prob
        used_edges = bin_edges
        hist_title = "Histogram of Skeleton Branch Lengths (SymKL)"
    elif distance_metric == "wasserstein":
        label_counts, bin_edges = np.histogram(label_lengths, bins=bins, range=hist_range)
        pred_counts, _ = np.histogram(pred_lengths, bins=bins, range=hist_range)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        distance = wasserstein_distance(u_values=bin_centers, v_values=bin_centers,
                                        u_weights=label_counts, v_weights=pred_counts)
        
        viz_label = label_counts
        viz_pred = pred_counts
        used_edges = bin_edges
        hist_title = "Histogram of Skeleton Branch Lengths (EMD)"
    else:
        print(f"Unknown distance_metric: {distance_metric}. Falling back to symKL.")
        label_counts, bin_edges = np.histogram(label_lengths, bins=bins, range=hist_range)
        pred_counts, _ = np.histogram(pred_lengths, bins=bins, range=hist_range)
        label_prob = label_counts.astype(np.float64) + epsilon
        pred_prob = pred_counts.astype(np.float64) + epsilon
        label_prob /= label_prob.sum()
        pred_prob /= pred_prob.sum()
        kl_div = np.sum(label_prob * np.log(label_prob / pred_prob)) + \
                 np.sum(pred_prob * np.log(pred_prob / label_prob))
        distance = kl_div
        viz_label = label_prob
        viz_pred = pred_prob
        used_edges = bin_edges
        hist_title = "Histogram of Skeleton Branch Lengths (SymKL - Fallback)"
    
    # Use pathlib for file operations.
    output_folder_path = Path(output_folder)
    output_folder_path.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(10, 5))
    bin_centers = (used_edges[:-1] + used_edges[1:]) / 2
    width = used_edges[1] - used_edges[0]
    plt.bar(bin_centers, viz_label, width=width, alpha=0.5, label="Label")
    plt.bar(bin_centers, viz_pred, width=width, alpha=0.5, label="Prediction")
    plt.xlabel("Skeleton Branch Length")
    plt.ylabel("Count")
    plt.title(hist_title)
    plt.legend()
    
    hist_path = output_folder_path / f"skeleton_branch_length_hist_sample_{SAMPLE_COUNT}.png"
    plt.savefig(str(hist_path.resolve()))
    plt.close()
    
    if wandb is not None:
        # Open the saved image with PIL and log it.
        with Image.open(str(hist_path.resolve())) as img:
            wandb.log({f"skeleton_branch_length_hist_sample_{SAMPLE_COUNT}": wandb.Image(img)})
    
    return float(distance)

def finalize(output_folder: str = ".", bins: int = 20) -> None:
    output_folder_path = Path(output_folder)
    output_folder_path.mkdir(parents=True, exist_ok=True)
    
    if not AGGREGATE_LABEL_LENGTHS or not AGGREGATE_PRED_LENGTHS:
        print("No branch lengths accumulated for aggregate histogram.")
        return
    
    plt.figure(figsize=(10, 6))
    plt.hist(AGGREGATE_LABEL_LENGTHS, bins=bins, alpha=0.5, label="Aggregate Label Branch Lengths")
    plt.hist(AGGREGATE_PRED_LENGTHS, bins=bins, alpha=0.5, label="Aggregate Prediction Branch Lengths")
    plt.xlabel("Skeleton Branch Length")
    plt.ylabel("Count")
    plt.title("Aggregate Histogram of Skeleton Branch Lengths")
    plt.legend()
    
    aggregate_hist_path = output_folder_path / "aggregate_skeleton_branch_length_hist.png"
    plt.savefig(str(aggregate_hist_path.resolve()))
    plt.close()
    
    if wandb is not None:
        with Image.open(str(aggregate_hist_path.resolve())) as agg_img:
            wandb.log({"aggregate_skeleton_branch_length_hist": wandb.Image(agg_img)})