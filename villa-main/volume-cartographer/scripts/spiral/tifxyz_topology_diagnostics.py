#!/usr/bin/env python3
"""Topology diagnostics for tifxyz surface grids.

Detects three classes of surface quality issues in tifxyz patch grids:

  1. **Sheet errors** (via PCA thickness):  UV windows whose 3D XYZ points
     span more than one physical papyrus wrap.  Uses Sean's PCA-based metric
     (Aug 2026) which outperforms curvature and normal-consistency checks
     (AUC 0.55 — effectively a coin flip on same-wrap classification).

  2. **Holes**:  Connected regions of invalid vertices surrounded by valid
     ones, indicating gaps in the surface reconstruction.

  3. **Mergers**:  Distinct surface regions that are geometrically close in
     3D but far apart in UV, indicating two physically separate wraps that
     have been erroneously joined.

The validity check matches the actual VC3D loader behavior:
  - Missing if ALL coords == sentinel (-1)
  - Missing if ANY coord is NaN
  - Missing if z-coordinate <= 0        (Nieuwlaar, Jul 31 2026)
  - Missing if mask.tif says so          (Nieuwlaar, Jul 31 2026)

Without the z <= 0 check, ~7 false positives per segment are generated
(confirmed by tifxyz-repair author).

Usage:
    python tifxyz_topology_diagnostics.py \\
        /path/to/segment_dir/ \\
        --output diagnostics_report.json

    python tifxyz_topology_diagnostics.py \\
        /path/to/pointset.tif \\
        --mask /path/to/mask.tif \\
        --output diagnostics_report.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class DiagnosticConfig:
    """Tunable thresholds for all diagnostic checks."""

    # Sentinel value used by VC3D for missing grid vertices.
    missing_sentinel: float = -1.0

    # PCA thickness: UV window size (cells) and detection threshold.
    pca_window_size: int = 24
    pca_thickness_threshold: float = 2.5
    pca_stride: int = 8

    # Hole detection: minimum contiguous invalid-cell area (cells^2) to flag.
    hole_min_area: int = 4

    # Merger detection: max 3D distance (voxels) for merger candidate,
    # min UV distance (cells) to confirm it's a true merger.
    merger_3d_radius: float = 3.0
    merger_min_uv_distance: int = 50


# ---------------------------------------------------------------------------
# Validity mask
# ---------------------------------------------------------------------------

def compute_validity_mask(
    grid: np.ndarray,
    cfg: DiagnosticConfig,
    mask_path: str | Path | None = None,
) -> np.ndarray:
    """Compute the boolean valid-vertex mask for a tifxyz grid.

    Matches the actual VC3D loader behavior:
      - Missing if ALL coords == missing_sentinel
      - Missing if ANY coord is NaN
      - Missing if z-coordinate <= 0  (Nieuwlaar, Jul 31 2026)
      - Missing if mask.tif says so   (Nieuwlaar, Jul 31 2026)

    Without the z <= 0 check, ~7 false positives per segment are generated
    (confirmed by tifxyz-repair author).

    Args:
        grid: (H, W, 3) float32/float64 array of XYZ coordinates.
        cfg: Diagnostic configuration.
        mask_path: Optional path to mask.tif sidecar.

    Returns:
        (H, W) boolean array, True = valid vertex.
    """
    is_sentinel = np.all(np.isclose(grid, cfg.missing_sentinel), axis=-1)
    is_nan = np.any(np.isnan(grid), axis=-1)
    is_z_invalid = grid[..., 2] <= 0  # z <= 0 is invalid per VC3D loader

    invalid = is_sentinel | is_nan | is_z_invalid

    if mask_path is not None:
        mask_path = Path(mask_path)
        if mask_path.exists():
            from PIL import Image
            mask_img = np.array(Image.open(mask_path)).astype(bool)
            if mask_img.shape == grid.shape[:2]:
                invalid |= ~mask_img
            else:
                warnings.warn(
                    f"mask.tif shape {mask_img.shape} != grid shape "
                    f"{grid.shape[:2]} -- mask not applied."
                )
        else:
            log.warning("mask_path %s does not exist -- skipping mask.", mask_path)

    return ~invalid


# ---------------------------------------------------------------------------
# Sheet error detection: PCA thickness (Sean, Aug 2026)
# ---------------------------------------------------------------------------

def detect_sheet_errors_pca_thickness(
    grid: np.ndarray,
    valid: np.ndarray,
    window_size: int = 24,
    thickness_threshold: float = 2.5,
    stride: int = 8,
) -> list[dict]:
    """Detect local sheet errors via PCA thickness.

    For each UV window:
      1. Collect XYZ of valid cells
      2. Compute best-fit plane via PCA (SVD)
      3. Project all points onto the plane normal
      4. Thickness = max(signed_distances) - min(signed_distances)
      5. High thickness -> samples span more than one physical wrap

    Sean (Aug 9, 2026): "weirdly effective metric for detecting local sheet
    errors... curvature is kinda smoothed out by symmetrical errors/jitter,
    this isn't."

    AUC advantage over normal-consistency: normals achieve 0.55 AUC
    (coin flip) on same-wrap classification.

    Args:
        grid: (H, W, 3) float32/float64 XYZ grid.
        valid: (H, W) boolean mask.
        window_size: UV window side length in cells.
        thickness_threshold: Physical distance threshold.
        stride: Window stride.

    Returns:
        List of flagged regions with centroid XYZ and thickness value.
    """
    H, W = grid.shape[:2]
    flags: list[dict] = []

    for i in range(0, H - window_size, stride):
        for j in range(0, W - window_size, stride):
            window_valid = valid[i:i + window_size, j:j + window_size]
            n_valid = int(window_valid.sum())
            if n_valid < 6:  # need at least 6 points for meaningful PCA
                continue

            pts = grid[i:i + window_size, j:j + window_size][window_valid]
            pts = pts.astype(np.float64)

            # PCA: center the points, compute SVD
            center = pts.mean(axis=0)
            pts_centered = pts - center
            _, _, Vt = np.linalg.svd(pts_centered, full_matrices=False)
            normal = Vt[-1]  # last right singular vector = plane normal

            # Signed distances along the plane normal
            signed_dists = pts_centered @ normal
            thickness = float(signed_dists.max() - signed_dists.min())

            if thickness > thickness_threshold:
                flags.append({
                    "type": "sheet_error_pca",
                    "u": int(i + window_size // 2),
                    "v": int(j + window_size // 2),
                    "x": float(center[0]),
                    "y": float(center[1]),
                    "z": float(center[2]),
                    "thickness": round(thickness, 3),
                    "severity": round(
                        min(thickness / (thickness_threshold * 3), 1.0), 4
                    ),
                    "n_points_in_window": n_valid,
                    "note": (
                        f"PCA thickness {thickness:.2f} > threshold "
                        f"{thickness_threshold:.2f}. Window ({i},{j}) to "
                        f"({i + window_size},{j + window_size}). "
                        f"Sean's metric (Aug 2026): high thickness = "
                        f"spans >1 physical wrap."
                    ),
                })

    return flags


# ---------------------------------------------------------------------------
# Hole detection
# ---------------------------------------------------------------------------

def detect_holes(
    valid: np.ndarray,
    cfg: DiagnosticConfig,
) -> list[dict]:
    """Detect holes (contiguous invalid regions surrounded by valid cells).

    Uses connected-component labelling on the invalid mask. Only regions
    that are fully interior (not touching the grid boundary) and exceed
    the minimum area threshold are flagged.

    Args:
        valid: (H, W) boolean mask.
        cfg: Diagnostic configuration.

    Returns:
        List of hole flag dicts.
    """
    try:
        from scipy.ndimage import label as ndimage_label
    except ImportError:
        log.warning("scipy.ndimage not available -- skipping hole detection.")
        return []

    invalid = ~valid
    labelled, n_components = ndimage_label(invalid)

    flags: list[dict] = []
    for comp_id in range(1, n_components + 1):
        comp_mask = labelled == comp_id
        area = int(comp_mask.sum())
        if area < cfg.hole_min_area:
            continue

        # Check if this component touches the grid boundary
        touches_boundary = (
            comp_mask[0, :].any()
            or comp_mask[-1, :].any()
            or comp_mask[:, 0].any()
            or comp_mask[:, -1].any()
        )
        if touches_boundary:
            continue

        # Centroid in UV
        ys, xs = np.where(comp_mask)
        u_center = int(ys.mean())
        v_center = int(xs.mean())

        flags.append({
            "type": "hole",
            "u": u_center,
            "v": v_center,
            "area_cells": area,
            "severity": round(min(area / 100.0, 1.0), 4),
            "note": f"Interior hole of {area} cells at UV ({u_center}, {v_center}).",
        })

    return flags


# ---------------------------------------------------------------------------
# Merger detection (KD-tree proximity in 3D, distance in UV)
# ---------------------------------------------------------------------------

def detect_mergers(
    grid: np.ndarray,
    valid: np.ndarray,
    cfg: DiagnosticConfig,
) -> list[dict]:
    """Detect potential surface mergers via KD-tree proximity analysis.

    Two grid vertices are a merger candidate if they are:
      - Close in 3D (within merger_3d_radius voxels)
      - Far apart in UV (at least merger_min_uv_distance cells)

    This indicates two physically separate surface regions that are
    erroneously joined in the same tifxyz grid.

    Args:
        grid: (H, W, 3) float32/float64 XYZ grid.
        valid: (H, W) boolean mask.
        cfg: Diagnostic configuration.

    Returns:
        List of merger flag dicts.
    """
    valid_ij = np.argwhere(valid)  # (N, 2) -- row (i), col (j)
    if len(valid_ij) < 2:
        return []

    valid_xyz = grid[valid]

    # Subsample for performance: KD-tree on all valid points can be huge.
    max_points = 50_000
    if len(valid_ij) > max_points:
        step = len(valid_ij) // max_points
        indices = np.arange(0, len(valid_ij), step)
        valid_ij_sub = valid_ij[indices]
        valid_xyz_sub = valid_xyz[indices]
    else:
        valid_ij_sub = valid_ij
        valid_xyz_sub = valid_xyz

    tree = cKDTree(valid_xyz_sub.astype(np.float64))

    # Find pairs within the 3D radius
    flags: list[dict] = []
    pairs_seen: set[tuple[int, int, int, int]] = set()
    pairs = tree.query_pairs(r=cfg.merger_3d_radius, output_type='ndarray')

    if len(pairs) == 0:
        return flags

    for idx_a, idx_b in pairs:
        ij_a = valid_ij_sub[idx_a]
        ij_b = valid_ij_sub[idx_b]
        uv_dist = float(np.sqrt(
            (float(ij_a[0] - ij_b[0])) ** 2 +
            (float(ij_a[1] - ij_b[1])) ** 2
        ))

        if uv_dist < cfg.merger_min_uv_distance:
            continue

        # Deduplicate by grid region
        region_key = (
            int(ij_a[0]) // 32, int(ij_a[1]) // 32,
            int(ij_b[0]) // 32, int(ij_b[1]) // 32,
        )
        if region_key in pairs_seen:
            continue
        pairs_seen.add(region_key)

        xyz_a = valid_xyz_sub[idx_a]
        xyz_b = valid_xyz_sub[idx_b]
        dist_3d = float(np.linalg.norm(xyz_a - xyz_b))

        flags.append({
            "type": "merger",
            "u1": int(ij_a[0]),
            "v1": int(ij_a[1]),
            "u2": int(ij_b[0]),
            "v2": int(ij_b[1]),
            "distance_3d": round(dist_3d, 3),
            "distance_uv": round(uv_dist, 1),
            "severity": round(min(uv_dist / 200.0, 1.0), 4),
            "note": (
                f"Merger candidate: 3D dist {dist_3d:.2f} vox, "
                f"UV dist {uv_dist:.1f} cells. Points at UV "
                f"({ij_a[0]},{ij_a[1]}) and ({ij_b[0]},{ij_b[1]}) "
                f"are close in 3D but far in UV."
            ),
        })

    return flags


# ---------------------------------------------------------------------------
# Main diagnostic runner
# ---------------------------------------------------------------------------

def run_diagnostics(
    grid: np.ndarray,
    cfg: DiagnosticConfig | None = None,
    mask_path: str | Path | None = None,
) -> dict:
    """Run all diagnostic checks on a tifxyz grid.

    Args:
        grid: (H, W, 3) float32/float64 XYZ coordinate grid.
        cfg: Diagnostic configuration.  None = defaults.
        mask_path: Optional path to mask.tif sidecar.

    Returns:
        Dict with 'summary' and 'correction_points' keys.
    """
    if cfg is None:
        cfg = DiagnosticConfig()

    valid = compute_validity_mask(grid, cfg, mask_path=mask_path)

    # PCA thickness (replaces normal-consistency sheet-switch detection)
    sheet_flags = detect_sheet_errors_pca_thickness(
        grid, valid,
        window_size=cfg.pca_window_size,
        thickness_threshold=cfg.pca_thickness_threshold,
        stride=cfg.pca_stride,
    )
    hole_flags = detect_holes(valid, cfg)
    merger_flags = detect_mergers(grid, valid, cfg)

    all_flags = sheet_flags + hole_flags + merger_flags
    summary = {
        "grid_shape": list(grid.shape[:2]),
        "n_valid_vertices": int(valid.sum()),
        "n_missing_vertices": int((~valid).sum()),
        "valid_fraction": round(float(valid.sum()) / valid.size, 4),
        "n_sheet_error_flags": len(sheet_flags),
        "n_hole_flags": len(hole_flags),
        "n_merger_flags": len(merger_flags),
        "n_total_flags": len(all_flags),
        "config": dataclasses.asdict(cfg),
    }
    return {"summary": summary, "correction_points": all_flags}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_tifxyz_grid(path: str | Path) -> np.ndarray:
    """Load a tifxyz grid from a multi-page TIFF (X, Y, Z channels).

    The standard tifxyz format stores 3 pages (one per coordinate axis)
    in a single TIFF file, each page being the same (H, W) grid.

    Returns:
        (H, W, 3) float32 array.
    """
    try:
        import tifffile
    except ImportError:
        from PIL import Image
        img = Image.open(path)
        pages = []
        for i in range(3):
            img.seek(i)
            pages.append(np.array(img, dtype=np.float32))
        return np.stack(pages, axis=-1)

    data = tifffile.imread(str(path))
    if data.ndim == 3 and data.shape[0] == 3:
        # (3, H, W) -> (H, W, 3)
        return np.transpose(data, (1, 2, 0)).astype(np.float32)
    elif data.ndim == 3 and data.shape[2] == 3:
        return data.astype(np.float32)
    else:
        raise ValueError(
            f"Unexpected tifxyz shape {data.shape}. "
            f"Expected (3, H, W) or (H, W, 3)."
        )


def find_mask_path(segment_dir: Path) -> Path | None:
    """Find the mask.tif sidecar next to a tifxyz file or in a segment dir."""
    candidates = [
        segment_dir / "mask.tif",
        segment_dir / "mask.png",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def find_pointset_path(segment_dir: Path) -> Path | None:
    """Find the pointset.tif (tifxyz file) in a segment directory."""
    candidates = [
        segment_dir / "pointset.tif",
        segment_dir / "pointset.tiff",
    ]
    for c in candidates:
        if c.exists():
            return c
    # Fallback: look for tifxyz-like files
    for f in sorted(segment_dir.glob("*.tif")):
        if f.stem not in ("mask", "winding", "area"):
            return f
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Topology diagnostics for tifxyz surface grids.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "input",
        help="Path to a tifxyz .tif file or a segment directory containing "
             "pointset.tif.",
    )
    parser.add_argument(
        "--mask",
        default=None,
        help="Path to mask.tif sidecar. Auto-detected if input is a directory.",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output JSON file path. Prints to stdout if not specified.",
    )
    parser.add_argument(
        "--pca-window-size",
        type=int,
        default=24,
        help="PCA thickness: UV window side length in cells (default: 24).",
    )
    parser.add_argument(
        "--pca-thickness-threshold",
        type=float,
        default=2.5,
        help="PCA thickness: physical distance threshold (default: 2.5).",
    )
    parser.add_argument(
        "--merger-3d-radius",
        type=float,
        default=3.0,
        help="Merger detection: max 3D distance in voxels (default: 3.0).",
    )
    parser.add_argument(
        "--merger-min-uv-distance",
        type=int,
        default=50,
        help="Merger detection: min UV distance in cells (default: 50).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging.",
    )

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    input_path = Path(args.input)

    # Resolve input: directory or direct .tif file
    if input_path.is_dir():
        pointset_path = find_pointset_path(input_path)
        if pointset_path is None:
            log.error("No pointset.tif found in %s", input_path)
            sys.exit(1)
        mask_path = args.mask or find_mask_path(input_path)
    else:
        pointset_path = input_path
        mask_path = args.mask
        if mask_path is None:
            mask_path = find_mask_path(input_path.parent)

    log.info("Loading grid from %s", pointset_path)
    grid = load_tifxyz_grid(pointset_path)
    log.info("Grid shape: %s", grid.shape[:2])
    if mask_path:
        log.info("Using mask: %s", mask_path)

    cfg = DiagnosticConfig(
        pca_window_size=args.pca_window_size,
        pca_thickness_threshold=args.pca_thickness_threshold,
        merger_3d_radius=args.merger_3d_radius,
        merger_min_uv_distance=args.merger_min_uv_distance,
    )

    result = run_diagnostics(grid, cfg, mask_path=mask_path)

    # Print summary
    s = result["summary"]
    log.info(
        "Summary: %d valid / %d missing (%.1f%% valid) | "
        "%d sheet errors, %d holes, %d mergers",
        s["n_valid_vertices"],
        s["n_missing_vertices"],
        s["valid_fraction"] * 100,
        s["n_sheet_error_flags"],
        s["n_hole_flags"],
        s["n_merger_flags"],
    )

    # Output
    output_json = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(output_json)
        log.info("Report written to %s", args.output)
    else:
        print(output_json)


if __name__ == "__main__":
    main()
