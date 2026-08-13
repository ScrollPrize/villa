#!/usr/bin/env python3
"""Generate automated relative winding annotations from the psi field (C3).

The spiral fitter uses manually-drawn point collections (PCLs) to establish
relative winding relationships between surface patches.  This script
automates that process by:

  1. Loading the psi winding field from C0
  2. For each pair of existing patches, sampling psi at their 3D positions
  3. Computing the relative winding offset: delta_w = round(psi_A - psi_B)
  4. Emitting a PCL JSON file with the computed winding annotations

The output format matches VC3D's point_collection JSON specification
(vc_pointcollections_json_version "1"), so it can be loaded directly by
fit_spiral.py alongside manually-drawn annotations.

Confidence filtering:
  - Only pairs with >10 valid psi samples on each side are considered
  - The annotation is emitted only if the psi difference has std < 0.4
    (consistent winding assignment across both patches)

Usage:
    python auto_winding_annotations.py \\
        --psi /path/to/winding_field.zarr \\
        --patches /path/to/patches/ \\
        --output /path/to/auto_winding_pcls.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def load_psi_array(zarr_path: str) -> tuple[np.ndarray, int]:
    """Load the psi volume as a numpy array.

    Returns (psi_array, resolution_factor).
    """
    import zarr
    store = zarr.open(str(zarr_path), mode="r")
    if "winding_position" in store:
        psi = np.asarray(store["winding_position"], dtype=np.float32)
    else:
        psi = np.asarray(store, dtype=np.float32)
    resolution = int(store.attrs.get("resolution_factor",
                     store.attrs.get("scaledown", 4)))
    return psi, resolution


def sample_psi_at_positions(
    positions_zyx: np.ndarray,
    psi: np.ndarray,
    resolution: int,
) -> np.ndarray:
    """Sample psi values at full-resolution ZYX coordinates.

    Uses nearest-neighbour lookup. Out-of-bounds positions return NaN.
    """
    grid_coords = (positions_zyx / resolution).astype(np.int32)
    result = np.full(len(positions_zyx), np.nan, dtype=np.float32)
    valid = np.ones(len(positions_zyx), dtype=bool)
    for dim in range(3):
        valid &= (grid_coords[:, dim] >= 0) & (grid_coords[:, dim] < psi.shape[dim])
    if valid.any():
        coords = grid_coords[valid]
        result[valid] = psi[coords[:, 0], coords[:, 1], coords[:, 2]]
    return result


def load_patches_zyx(patches_dir: str) -> dict[str, np.ndarray]:
    """Load patch grids and extract valid ZYX positions.

    Returns dict mapping patch_id -> (N, 3) array of valid ZYX coordinates.
    """
    patches = {}
    patches_path = Path(patches_dir)

    # Try loading from tifxyz files
    for tif_path in sorted(patches_path.glob("*/pointset.tif")):
        patch_id = tif_path.parent.name
        try:
            from tifxyz_topology_diagnostics import load_tifxyz_grid, compute_validity_mask, DiagnosticConfig
            grid = load_tifxyz_grid(str(tif_path))
            valid = compute_validity_mask(grid, DiagnosticConfig())
            # grid is (H, W, 3) with (x, y, z) — convert to (z, y, x)
            valid_pts = grid[valid]
            patches[patch_id] = valid_pts[:, [2, 1, 0]]  # x,y,z -> z,y,x
        except Exception as e:
            log.warning("Failed to load patch %s: %s", patch_id, e)

    # Also try loading from .npy files
    for npy_path in sorted(patches_path.glob("*.npy")):
        patch_id = npy_path.stem
        if patch_id not in patches:
            try:
                data = np.load(str(npy_path))
                if data.ndim == 2 and data.shape[1] >= 3:
                    patches[patch_id] = data[:, :3]
            except Exception as e:
                log.warning("Failed to load patch %s: %s", patch_id, e)

    log.info("Loaded %d patches from %s", len(patches), patches_dir)
    return patches


def compute_relative_windings(
    patches: dict[str, np.ndarray],
    psi: np.ndarray,
    resolution: int,
    min_valid_samples: int = 10,
    max_std: float = 0.4,
    max_pairs: int = 5000,
) -> list[dict]:
    """Compute relative winding annotations between patch pairs.

    For each pair of patches (A, B):
      1. Sample psi at valid positions of both patches
      2. Compute median psi difference: delta_w = round(median(psi_A) - median(psi_B))
      3. If both patches have sufficient valid samples and low std, emit annotation

    Args:
        patches: Dict mapping patch_id -> (N, 3) ZYX array.
        psi: (Z, Y, X) psi volume.
        resolution: Resolution factor.
        min_valid_samples: Minimum valid psi samples per patch.
        max_std: Maximum allowed std in psi values within a patch.
        max_pairs: Maximum number of pairs to evaluate.

    Returns:
        List of annotation dicts with relative winding info.
    """
    # Pre-compute per-patch psi statistics
    patch_stats = {}
    for patch_id, positions in patches.items():
        psi_vals = sample_psi_at_positions(positions, psi, resolution)
        valid = np.isfinite(psi_vals) & (psi_vals > 0.5)
        n_valid = int(valid.sum())
        if n_valid >= min_valid_samples:
            patch_stats[patch_id] = {
                "median_psi": float(np.median(psi_vals[valid])),
                "mean_psi": float(np.mean(psi_vals[valid])),
                "std_psi": float(np.std(psi_vals[valid])),
                "n_valid": n_valid,
                "centroid_zyx": positions[valid].mean(axis=0).tolist(),
            }

    log.info(
        "Patches with sufficient psi coverage: %d / %d",
        len(patch_stats), len(patches),
    )

    if len(patch_stats) < 2:
        log.warning("Too few patches with valid psi — no annotations generated.")
        return []

    # Generate pair annotations
    annotations = []
    patch_ids = sorted(patch_stats.keys())
    n_pairs = 0

    for i, id_a in enumerate(patch_ids):
        for id_b in patch_ids[i + 1:]:
            if n_pairs >= max_pairs:
                break

            stats_a = patch_stats[id_a]
            stats_b = patch_stats[id_b]

            # Skip if either patch has high internal psi variance
            if stats_a["std_psi"] > max_std or stats_b["std_psi"] > max_std:
                continue

            # Relative winding offset
            delta_psi = stats_a["median_psi"] - stats_b["median_psi"]
            delta_w = round(delta_psi)

            # Skip same-winding pairs (not informative)
            if delta_w == 0:
                continue

            # Confidence: low if the psi difference is far from an integer
            fractional = abs(delta_psi - delta_w)
            confidence = max(0.0, 1.0 - 2.0 * fractional)

            if confidence < 0.5:
                continue

            annotations.append({
                "patch_a": id_a,
                "patch_b": id_b,
                "delta_winding": int(delta_w),
                "delta_psi_raw": round(delta_psi, 4),
                "confidence": round(confidence, 4),
                "psi_a_median": stats_a["median_psi"],
                "psi_b_median": stats_b["median_psi"],
                "psi_a_std": stats_a["std_psi"],
                "psi_b_std": stats_b["std_psi"],
                "centroid_a_zyx": stats_a["centroid_zyx"],
                "centroid_b_zyx": stats_b["centroid_zyx"],
            })
            n_pairs += 1

    log.info("Generated %d relative winding annotations", len(annotations))
    return annotations


def annotations_to_pcl_json(
    annotations: list[dict],
    name_prefix: str = "auto_psi",
) -> dict:
    """Convert relative winding annotations to VC3D point collection JSON.

    Output format matches vc_pointcollections_json_version "1", compatible
    with load_point_collection() in point_collection.py.

    Each annotation becomes a collection with two points:
      - Point 0: centroid of patch A, winding_annotation = 0
      - Point 1: centroid of patch B, winding_annotation = delta_winding
    """
    collections = {}
    for i, ann in enumerate(annotations):
        col_id = i + 1
        col_name = f"{name_prefix}_{ann['patch_a']}_{ann['patch_b']}"

        # Points: centroid of each patch with winding annotations
        # ZYX -> [x, y, z] for the "p" field (VC3D convention)
        centroid_a = ann["centroid_a_zyx"][::-1]  # zyx -> xyz
        centroid_b = ann["centroid_b_zyx"][::-1]

        collections[str(col_id)] = {
            "name": col_name,
            "color": [0.2, 0.8, 0.2],  # green for auto-generated
            "metadata": {
                "source": "auto_winding_annotations_C3",
                "confidence": ann["confidence"],
                "delta_psi_raw": ann["delta_psi_raw"],
            },
            "points": {
                "0": {
                    "p": centroid_a,
                    "wind_a": 0.0,
                    "creation_time": 0,
                },
                "1": {
                    "p": centroid_b,
                    "wind_a": float(ann["delta_winding"]),
                    "creation_time": 0,
                },
            },
        }

    return {
        "vc_pointcollections_json_version": "1",
        "collections": collections,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    p = argparse.ArgumentParser(
        description="Generate automated relative winding annotations from psi field."
    )
    p.add_argument("--psi", required=True, help="Path to winding_field.zarr")
    p.add_argument("--patches", required=True,
                   help="Directory containing patch tifxyz files")
    p.add_argument("--output", "-o", required=True,
                   help="Output JSON path for auto winding annotations")
    p.add_argument("--min-valid-samples", type=int, default=10,
                   help="Min valid psi samples per patch (default: 10)")
    p.add_argument("--max-std", type=float, default=0.4,
                   help="Max allowed psi std within a patch (default: 0.4)")
    p.add_argument("--max-pairs", type=int, default=5000,
                   help="Max number of pairs to evaluate (default: 5000)")
    args = p.parse_args()

    psi, resolution = load_psi_array(args.psi)
    patches = load_patches_zyx(args.patches)
    annotations = compute_relative_windings(
        patches, psi, resolution,
        min_valid_samples=args.min_valid_samples,
        max_std=args.max_std,
        max_pairs=args.max_pairs,
    )

    pcl_json = annotations_to_pcl_json(annotations)

    Path(args.output).write_text(json.dumps(pcl_json, indent=2))
    log.info("Annotations written to %s", args.output)
    log.info(
        "Summary: %d annotations across %d unique patches",
        len(annotations),
        len(set(a["patch_a"] for a in annotations) | set(a["patch_b"] for a in annotations)),
    )
