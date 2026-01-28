#!/usr/bin/env python3
"""Fill inner (center hole) and outer (background) regions of a spiral label volume.

This script processes a cylindrical/spiral label zarr volume and fills:
- Outer region: unlabeled pixels outside the alpha shape of labeled pixels
- Inner region: unlabeled pixels inside the spiral (center hole)

The algorithm works per 2D slice:
1. Outer detection: Compute alpha shape of labeled pixels, fill outside
2. Inner detection: Find innermost labeled points in each angular direction
   from the centroid, create polygon, and fill inside

Processing is done at a lower resolution level for speed, then upscaled.

Example usage:
    python fill_inner_outer_labels.py /path/to/input.zarr /path/to/output.zarr \
        --fill-value 255 \
        --alpha 0.005 \
        --n-angle-bins 72 \
        --shrink-factor 0.95 \
        --compute-level 3 \
        --output-level 0
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import zarr
from numcodecs import Blosc
from tqdm import tqdm

# Try to import alphashape - required for outer region detection
try:
    import alphashape
    from shapely.geometry import Polygon, MultiPolygon
    HAS_ALPHASHAPE = True
except ImportError:
    HAS_ALPHASHAPE = False
    print("Warning: alphashape not installed. Outer region detection disabled.")
    print("Install with: pip install alphashape shapely")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fill inner/outer regions of a spiral label volume.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "input_zarr",
        type=str,
        help="Path to input OME-Zarr with labels",
    )
    parser.add_argument(
        "output_zarr",
        type=str,
        help="Path to output OME-Zarr",
    )
    parser.add_argument(
        "--fill-value",
        type=int,
        default=255,
        help="Label value to use for inner/outer regions (default: 255)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.005,
        help="Alpha parameter for alpha shape (smaller = tighter fit, default: 0.005)",
    )
    parser.add_argument(
        "--n-angle-bins",
        type=int,
        default=72,
        help="Number of angular sectors for inner region detection (default: 72 = 5° resolution)",
    )
    parser.add_argument(
        "--shrink-factor",
        type=float,
        default=0.95,
        help="Shrink inner boundary by this factor for safety margin (default: 0.95)",
    )
    parser.add_argument(
        "--compute-level",
        type=int,
        default=2,
        help="Pyramid level for mask computation (default: 3, 8x downsampled)",
    )
    parser.add_argument(
        "--output-level",
        type=int,
        default=0,
        help="Target output resolution level (default: 0, full resolution)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count)",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=100,
        help="Maximum points for alpha shape computation (default: 10000)",
    )
    parser.add_argument(
        "--z-min",
        type=int,
        default=None,
        help="Start slice index (default: 0)",
    )
    parser.add_argument(
        "--z-max",
        type=int,
        default=None,
        help="End slice index, exclusive (default: last slice)",
    )
    parser.add_argument(
        "--skip-outer",
        action="store_true",
        help="Skip outer region detection (alpha shape)",
    )
    parser.add_argument(
        "--skip-inner",
        action="store_true",
        help="Skip inner region detection (morphological)",
    )
    parser.add_argument(
        "--visualize",
        type=int,
        default=None,
        help="Visualize a single slice (saves debug images, doesn't write to zarr)",
    )

    return parser.parse_args()


def detect_outer_region(
    label_slice: np.ndarray,
    alpha: float,
    max_points: int,
) -> Optional[np.ndarray]:
    """Detect outer region using alpha shape.

    Parameters
    ----------
    label_slice : np.ndarray
        2D label slice (H, W).
    alpha : float
        Alpha parameter for alpha shape.
    max_points : int
        Maximum number of points to use for alpha shape.

    Returns
    -------
    Optional[np.ndarray]
        Boolean mask of outer region, or None if detection failed.
    """
    if not HAS_ALPHASHAPE:
        return None

    # Get coordinates of all labeled pixels
    labeled_mask = label_slice > 0
    if not labeled_mask.any():
        return None

    y_coords, x_coords = np.where(labeled_mask)
    points = np.column_stack([x_coords, y_coords])

    # Subsample if too many points
    if len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        points = points[indices]

    if len(points) < 4:
        return None

    try:
        # Compute alpha shape
        alpha_shape = alphashape.alphashape(points, alpha)

        if alpha_shape is None or alpha_shape.is_empty:
            return None

        h, w = label_slice.shape

        # Rasterize the alpha shape polygon using OpenCV (vectorized, fast)
        # This replaces the slow point-by-point containment test
        inside_mask = np.zeros((h, w), dtype=np.uint8)

        def fill_polygon(geom: Polygon, mask: np.ndarray) -> None:
            """Fill a polygon (including holes) into the mask."""
            # Get exterior coordinates and fill
            exterior_coords = np.array(geom.exterior.coords).astype(np.int32)
            cv2.fillPoly(mask, [exterior_coords], 1)

            # Cut out holes (interior rings)
            for interior in geom.interiors:
                interior_coords = np.array(interior.coords).astype(np.int32)
                cv2.fillPoly(mask, [interior_coords], 0)

        # Handle both Polygon and MultiPolygon results
        if isinstance(alpha_shape, Polygon):
            fill_polygon(alpha_shape, inside_mask)
        elif isinstance(alpha_shape, MultiPolygon):
            for poly in alpha_shape.geoms:
                fill_polygon(poly, inside_mask)
        else:
            # Fallback for other geometry types (e.g., GeometryCollection)
            print(f"Unexpected geometry type: {type(alpha_shape)}")
            return None

        # Outer region = unlabeled pixels that are NOT inside the alpha shape
        outer_mask = ~labeled_mask & (inside_mask == 0)

        return outer_mask

    except Exception as e:
        print(f"Alpha shape computation failed: {e}")
        return None


def detect_inner_region(
    label_slice: np.ndarray,
    n_angle_bins: int = 72,
    shrink_factor: float = 0.95,
    smoothing_window: int = 5,
) -> np.ndarray:
    """Detect inner region using centroid + angle-binned inner boundary.

    This works by finding the innermost labeled points in each angular direction
    from the centroid, creating a polygon from these points, and filling it.

    Parameters
    ----------
    label_slice : np.ndarray
        2D label slice (H, W).
    n_angle_bins : int
        Number of angular sectors (default 72 = 5° resolution).
    shrink_factor : float
        Shrink the inner boundary polygon by this factor (default 0.95).
        Values < 1.0 create a safety margin inside the detected boundary.
    smoothing_window : int
        Window size for circular median filter to remove outliers (default 5).
        Set to 0 or 1 to disable smoothing.

    Returns
    -------
    np.ndarray
        Boolean mask of inner region.
    """
    # Create binary mask of labeled pixels
    labeled_mask = label_slice > 0

    if not labeled_mask.any():
        return np.zeros_like(labeled_mask)

    # Get coordinates of all labeled pixels
    y_coords, x_coords = np.where(labeled_mask)

    if len(x_coords) < 3:
        return np.zeros_like(labeled_mask)

    # Step 1: Calculate centroid of all labeled points
    centroid_x = np.mean(x_coords)
    centroid_y = np.mean(y_coords)

    # Step 2: Compute angle from centroid to each labeled point
    dx = x_coords - centroid_x
    dy = y_coords - centroid_y
    angles = np.arctan2(dy, dx)  # Range: [-pi, pi]
    distances = np.sqrt(dx**2 + dy**2)

    # Step 3: Bin points by angle and find closest point in each bin
    bin_edges = np.linspace(-np.pi, np.pi, n_angle_bins + 1)
    bin_indices = np.digitize(angles, bin_edges) - 1  # 0 to n_angle_bins-1
    bin_indices = np.clip(bin_indices, 0, n_angle_bins - 1)

    # Find minimum distance in each bin (store distances and angles for smoothing)
    bin_min_distances = np.full(n_angle_bins, np.nan)
    bin_angles = np.linspace(-np.pi, np.pi, n_angle_bins, endpoint=False) + np.pi / n_angle_bins

    for bin_idx in range(n_angle_bins):
        mask = bin_indices == bin_idx
        if mask.any():
            bin_min_distances[bin_idx] = np.min(distances[mask])

    # Step 4: Apply circular median filter to smooth out outliers
    if smoothing_window > 1:
        # Pad circularly for edge handling
        pad = smoothing_window // 2
        padded = np.concatenate([bin_min_distances[-pad:], bin_min_distances, bin_min_distances[:pad]])

        smoothed = np.full_like(bin_min_distances, np.nan)
        for i in range(n_angle_bins):
            window = padded[i : i + smoothing_window]
            valid = window[~np.isnan(window)]
            if len(valid) > 0:
                smoothed[i] = np.median(valid)

        bin_min_distances = smoothed

    # Step 5: Convert smoothed distances back to boundary points
    valid_mask = ~np.isnan(bin_min_distances)
    if valid_mask.sum() < 3:
        return np.zeros_like(labeled_mask)

    valid_angles = bin_angles[valid_mask]
    valid_distances = bin_min_distances[valid_mask]

    inner_boundary_points = np.column_stack([
        centroid_x + valid_distances * np.cos(valid_angles),
        centroid_y + valid_distances * np.sin(valid_angles),
    ])

    # Step 6: Apply shrink factor - move points toward centroid
    if shrink_factor != 1.0:
        inner_boundary_points[:, 0] = centroid_x + shrink_factor * (inner_boundary_points[:, 0] - centroid_x)
        inner_boundary_points[:, 1] = centroid_y + shrink_factor * (inner_boundary_points[:, 1] - centroid_y)

    # Step 7: Create polygon and rasterize
    h, w = label_slice.shape
    inner_mask = np.zeros((h, w), dtype=np.uint8)

    # Convert to int32 for cv2.fillPoly
    polygon_pts = inner_boundary_points.astype(np.int32)
    cv2.fillPoly(inner_mask, [polygon_pts], 1)

    # Convert to boolean and exclude already-labeled pixels
    inner_mask = (inner_mask > 0) & ~labeled_mask

    return inner_mask


def process_slice(
    args: Tuple,
) -> Tuple[int, Optional[np.ndarray], Optional[np.ndarray]]:
    """Process a single slice to compute inner/outer masks.

    Parameters
    ----------
    args : Tuple
        (slice_idx, input_path, compute_level, alpha, max_points,
         n_angle_bins, shrink_factor, skip_outer, skip_inner)

    Returns
    -------
    Tuple[int, Optional[np.ndarray], Optional[np.ndarray]]
        (slice_idx, outer_mask, inner_mask)
    """
    (slice_idx, input_path, compute_level, alpha, max_points,
     n_angle_bins, shrink_factor, skip_outer, skip_inner) = args

    # Open zarr (each worker opens its own handle)
    input_store = zarr.open(input_path, mode="r")
    input_arr = input_store[str(compute_level)]

    # Read slice
    label_slice = input_arr[slice_idx, :, :]

    outer_mask = None
    inner_mask = None

    # Detect outer region
    if not skip_outer:
        outer_mask = detect_outer_region(label_slice, alpha, max_points)

    # Detect inner region
    if not skip_inner:
        inner_mask = detect_inner_region(label_slice, n_angle_bins, shrink_factor)

    return slice_idx, outer_mask, inner_mask


def process_and_write_slice(
    args: Tuple,
) -> int:
    """Process a single slice and write directly (no upscaling).

    Used when compute_level == output_level for single-pass processing.

    Parameters
    ----------
    args : Tuple
        (slice_idx, input_path, output_path, level, alpha, max_points,
         n_angle_bins, shrink_factor, skip_outer, skip_inner, fill_value)

    Returns
    -------
    int
        Number of pixels filled.
    """
    (slice_idx, input_path, output_path, level, alpha, max_points,
     n_angle_bins, shrink_factor, skip_outer, skip_inner, fill_value) = args

    # Open zarr (each worker opens its own handle)
    input_store = zarr.open(input_path, mode="r")
    output_store = zarr.open(output_path, mode="r+")
    input_arr = input_store[str(level)]
    output_arr = output_store[str(level)]

    # Read slice
    label_slice = input_arr[slice_idx, :, :]

    # Compute masks
    outer_mask = None if skip_outer else detect_outer_region(label_slice, alpha, max_points)
    inner_mask = None if skip_inner else detect_inner_region(label_slice, n_angle_bins, shrink_factor)

    # Apply and write
    output_slice = np.zeros_like(label_slice)
    pixels_filled = 0

    if outer_mask is not None:
        output_slice[outer_mask] = fill_value
        pixels_filled += outer_mask.sum()

    if inner_mask is not None:
        output_slice[inner_mask] = fill_value
        pixels_filled += inner_mask.sum()

    output_arr[slice_idx, :, :] = output_slice
    return pixels_filled


def upscale_mask(
    mask: np.ndarray,
    target_shape: Tuple[int, int],
) -> np.ndarray:
    """Upscale a boolean mask using nearest-neighbor interpolation.

    Uses OpenCV for fast resizing.

    Parameters
    ----------
    mask : np.ndarray
        Boolean mask to upscale.
    target_shape : Tuple[int, int]
        Target shape (H, W).

    Returns
    -------
    np.ndarray
        Upscaled boolean mask.
    """
    # Convert to uint8 for cv2
    mask_uint8 = mask.astype(np.uint8) * 255

    # Resize using nearest neighbor (preserves binary values)
    # cv2.resize takes (width, height) not (height, width)
    resized = cv2.resize(
        mask_uint8,
        (target_shape[1], target_shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )

    return resized > 0


def write_slice_with_fill(
    slice_idx: int,
    input_arr: zarr.Array,
    output_arr: zarr.Array,
    output_level: int,
    compute_level: int,
    outer_mask: Optional[np.ndarray],
    inner_mask: Optional[np.ndarray],
    fill_value: int,
) -> int:
    """Write a slice with inner/outer regions filled.

    Parameters
    ----------
    slice_idx : int
        Slice index at compute level.
    input_arr : zarr.Array
        Input zarr array (shared handle).
    output_arr : zarr.Array
        Output zarr array (shared handle).
    output_level : int
        Output pyramid level.
    compute_level : int
        Compute pyramid level.
    outer_mask : Optional[np.ndarray]
        Boolean mask for outer region, or None.
    inner_mask : Optional[np.ndarray]
        Boolean mask for inner region, or None.
    fill_value : int
        Value to fill masked regions with.

    Returns
    -------
    int
        Number of pixels filled.
    """

    # Calculate slice index at output level
    scale_factor = 2 ** (compute_level - output_level)
    output_slice_start = slice_idx * scale_factor
    output_slice_end = min(output_slice_start + scale_factor, input_arr.shape[0])

    # Get target shape for this output level
    target_shape = (input_arr.shape[1], input_arr.shape[2])

    # Pre-compute upscaled masks ONCE (cached outside the loop)
    outer_upscaled = None
    inner_upscaled = None

    if outer_mask is not None:
        outer_upscaled = upscale_mask(outer_mask, target_shape)

    if inner_mask is not None:
        inner_upscaled = upscale_mask(inner_mask, target_shape)

    pixels_filled = 0

    # Process each output slice that corresponds to this compute slice
    for out_idx in range(output_slice_start, output_slice_end):
        # Read original slice
        original = input_arr[out_idx, :, :]

        # Create output slice (zeros - only output the mask)
        output_slice = np.zeros_like(original)

        # Apply outer mask (using pre-computed upscaled mask)
        if outer_upscaled is not None:
            output_slice[outer_upscaled] = fill_value
            pixels_filled += outer_upscaled.sum()

        # Apply inner mask (using pre-computed upscaled mask)
        if inner_upscaled is not None:
            output_slice[inner_upscaled] = fill_value
            pixels_filled += inner_upscaled.sum()

        # Write to output
        output_arr[out_idx, :, :] = output_slice

    return pixels_filled


def visualize_slice(
    input_path: str,
    slice_idx: int,
    compute_level: int,
    alpha: float,
    max_points: int,
    n_angle_bins: int,
    shrink_factor: float,
    output_dir: Path,
) -> None:
    """Visualize detection results for a single slice.

    Parameters
    ----------
    input_path : str
        Path to input zarr.
    slice_idx : int
        Slice index at compute level.
    compute_level : int
        Pyramid level for computation.
    alpha : float
        Alpha shape parameter.
    max_points : int
        Maximum points for alpha shape.
    n_angle_bins : int
        Number of angular sectors for inner region detection.
    shrink_factor : float
        Shrink factor for inner boundary polygon.
    output_dir : Path
        Directory to save visualization images.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read slice
    input_store = zarr.open(input_path, mode="r")
    input_arr = input_store[str(compute_level)]
    label_slice = input_arr[slice_idx, :, :]

    # Save original
    cv2.imwrite(str(output_dir / f"slice_{slice_idx}_original.png"), label_slice)

    # Detect outer region
    print(f"Computing outer region (alpha={alpha})...")
    outer_mask = detect_outer_region(label_slice, alpha, max_points)
    if outer_mask is not None:
        outer_vis = (outer_mask.astype(np.uint8) * 255)
        cv2.imwrite(str(output_dir / f"slice_{slice_idx}_outer.png"), outer_vis)
        print(f"  Outer region: {outer_mask.sum()} pixels")
    else:
        print("  Outer region detection failed")

    # Detect inner region
    print(f"Computing inner region (n_angle_bins={n_angle_bins}, shrink_factor={shrink_factor})...")
    inner_mask = detect_inner_region(label_slice, n_angle_bins, shrink_factor)
    inner_vis = (inner_mask.astype(np.uint8) * 255)
    cv2.imwrite(str(output_dir / f"slice_{slice_idx}_inner.png"), inner_vis)
    print(f"  Inner region: {inner_mask.sum()} pixels")

    # Create combined visualization
    h, w = label_slice.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)

    # Original labels in white
    vis[label_slice > 0] = [255, 255, 255]

    # Outer region in blue
    if outer_mask is not None:
        vis[outer_mask] = [255, 0, 0]  # BGR

    # Inner region in red
    vis[inner_mask] = [0, 0, 255]  # BGR

    cv2.imwrite(str(output_dir / f"slice_{slice_idx}_combined.png"), vis)
    print(f"Saved visualizations to {output_dir}")


def create_output_zarr(
    input_path: Path,
    output_path: Path,
    output_level: int,
) -> None:
    """Create output zarr with same structure as input.

    Parameters
    ----------
    input_path : Path
        Path to input zarr.
    output_path : Path
        Path to output zarr.
    output_level : int
        Level to copy/create.
    """
    input_store = zarr.open(str(input_path), mode="r")

    # Create output store
    if output_path.exists():
        print(f"Output exists, opening in r+ mode")
        output_store = zarr.open(str(output_path), mode="r+")
    else:
        print(f"Creating output zarr")
        output_store = zarr.open(str(output_path), mode="w")

    # Copy the target level
    input_arr = input_store[str(output_level)]

    if str(output_level) not in output_store:
        compressor = input_arr.compressor
        if compressor is None:
            compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)

        output_store.create_dataset(
            str(output_level),
            shape=input_arr.shape,
            chunks=input_arr.chunks,
            dtype=input_arr.dtype,
            compressor=compressor,
            fill_value=input_arr.fill_value,
        )
        print(f"Created output level {output_level}: shape={input_arr.shape}")

    # Copy attributes if present
    if hasattr(input_store, 'attrs') and input_store.attrs:
        output_store.attrs.update(dict(input_store.attrs))


def main() -> None:
    """Main entry point."""
    args = parse_args()

    input_path = Path(args.input_zarr)
    output_path = Path(args.output_zarr)

    num_workers = args.workers or mp.cpu_count()
    print(f"Using {num_workers} workers")

    # Open input to get metadata
    input_store = zarr.open(str(input_path), mode="r")

    # Validate compute level exists
    if str(args.compute_level) not in input_store:
        available = [k for k in input_store.keys() if k.isdigit()]
        raise ValueError(
            f"Compute level {args.compute_level} not found. "
            f"Available levels: {available}"
        )

    compute_arr = input_store[str(args.compute_level)]
    print(f"Compute level {args.compute_level}: shape={compute_arr.shape}")

    # Validate output level exists
    if str(args.output_level) not in input_store:
        available = [k for k in input_store.keys() if k.isdigit()]
        raise ValueError(
            f"Output level {args.output_level} not found. "
            f"Available levels: {available}"
        )

    output_arr = input_store[str(args.output_level)]
    print(f"Output level {args.output_level}: shape={output_arr.shape}")

    # Handle visualization mode
    if args.visualize is not None:
        print(f"\nVisualization mode: processing slice {args.visualize}")
        vis_dir = Path("fill_inner_outer_vis")
        visualize_slice(
            str(input_path),
            args.visualize,
            args.compute_level,
            args.alpha,
            args.max_points,
            args.n_angle_bins,
            args.shrink_factor,
            vis_dir,
        )
        return

    # Determine slice range
    num_slices = compute_arr.shape[0]
    start_slice = args.z_min if args.z_min is not None else 0
    end_slice = args.z_max if args.z_max is not None else num_slices

    print(f"\nProcessing slices {start_slice} to {end_slice} ({end_slice - start_slice} slices)")
    print(f"Alpha: {args.alpha}, Angle bins: {args.n_angle_bins}, Shrink factor: {args.shrink_factor}")
    print(f"Fill value: {args.fill_value}")
    print(f"Skip outer: {args.skip_outer}, Skip inner: {args.skip_inner}")

    # Create output zarr
    create_output_zarr(input_path, output_path, args.output_level)

    if args.compute_level == args.output_level:
        # Single-pass: compute and write directly (no upscaling needed)
        print("\nSingle-pass mode (compute_level == output_level)...")

        slice_args = [
            (
                idx,
                str(input_path),
                str(output_path),
                args.compute_level,
                args.alpha,
                args.max_points,
                args.n_angle_bins,
                args.shrink_factor,
                args.skip_outer,
                args.skip_inner,
                args.fill_value,
            )
            for idx in range(start_slice, end_slice)
        ]

        total_filled = 0
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(tqdm(
                executor.map(process_and_write_slice, slice_args),
                total=len(slice_args),
                desc="Processing slices",
            ))
        total_filled = sum(results)

    else:
        # Two-phase: compute masks at lower resolution, then upscale and write
        # Phase 1: Compute masks at lower resolution
        print("\nPhase 1: Computing masks at lower resolution...")

        slice_args = [
            (
                idx,
                str(input_path),
                args.compute_level,
                args.alpha,
                args.max_points,
                args.n_angle_bins,
                args.shrink_factor,
                args.skip_outer,
                args.skip_inner,
            )
            for idx in range(start_slice, end_slice)
        ]

        # Store masks in memory (they're at low resolution, so this is manageable)
        masks = {}  # slice_idx -> (outer_mask, inner_mask)

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(tqdm(
                executor.map(process_slice, slice_args),
                total=len(slice_args),
                desc="Computing masks",
            ))

        for slice_idx, outer_mask, inner_mask in results:
            if outer_mask is not None or inner_mask is not None:
                masks[slice_idx] = (outer_mask, inner_mask)

        print(f"Computed masks for {len(masks)} slices")

        if not masks:
            print("No masks computed, exiting")
            return

        # Phase 2: Upscale masks and write to output
        print("\nPhase 2: Upscaling masks and writing to output...")

        # Open stores ONCE and share across threads
        input_store = zarr.open(str(input_path), mode="r")
        output_store = zarr.open(str(output_path), mode="r+")
        input_arr = input_store[str(args.output_level)]
        output_arr = output_store[str(args.output_level)]

        total_filled = 0
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(
                    write_slice_with_fill,
                    slice_idx,
                    input_arr,
                    output_arr,
                    args.output_level,
                    args.compute_level,
                    outer_mask,
                    inner_mask,
                    args.fill_value,
                ): slice_idx
                for slice_idx, (outer_mask, inner_mask) in masks.items()
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Writing output"):
                total_filled += future.result()

    print(f"\nComplete! Filled {total_filled:,} pixels")
    print(f"Output written to: {output_path}")


if __name__ == "__main__":
    main()
