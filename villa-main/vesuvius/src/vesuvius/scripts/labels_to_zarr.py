#!/usr/bin/env python3
"""Map 2D label values from tifxyz surfaces to a 3D zarr volume.

This script reads tifxyz folders containing label PNG/TIF files, extracts
pixels matching a target label value, maps them to 3D coordinates using
the tifxyz grid, and writes to a zarr volume.

Example usage:
    python labels_to_zarr.py /path/to/surfaces \
        --label-suffix labels \
        --label-value 1 \
        --output /path/to/output.zarr \
        --shape 14000 8000 8000 \
        --padding 50 \
        --padding-value 2
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import fastremap
import numpy as np
import tifffile
import zarr
from numcodecs import Blosc
from scipy.ndimage import label as label_components, find_objects
from scipy.ndimage import distance_transform_edt as scipy_edt
from tqdm import tqdm

# Try to import edt package for fast multi-threaded EDT
HAS_EDT = False
try:
    import edt as edt_package
    HAS_EDT = True
except ImportError:
    HAS_EDT = False


def distance_transform_edt(mask: np.ndarray) -> np.ndarray:
    """Compute EDT using edt package if available, else scipy.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask (True where distance should be computed from).

    Returns
    -------
    np.ndarray
        Distance transform array.
    """
    if HAS_EDT:
        # The edt package silently copies non-contiguous arrays, which can cause
        # unexpected memory usage. Ensure contiguity explicitly.
        if not mask.flags['C_CONTIGUOUS']:
            mask = np.ascontiguousarray(mask)
        # Use single thread since we parallelize over chunks via multiprocessing
        return edt_package.edt(mask, parallel=1)
    else:
        return scipy_edt(mask)

from vesuvius.tifxyz import read_tifxyz


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Map 2D label values from tifxyz surfaces to a 3D zarr volume.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "input_folder",
        type=str,
        help="Folder containing tifxyz directories",
    )
    parser.add_argument(
        "--label-suffix",
        type=str,
        required=True,
        help="Suffix for label file (e.g., 'labels' looks for labels.png or labels.tif)",
    )
    parser.add_argument(
        "--label-value",
        type=int,
        required=False,
        default=None,
        help="The label value to extract from the 2D label images (required in single-label mode, ignored in multi-label mode)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output zarr path",
    )
    parser.add_argument(
        "--shape",
        type=int,
        nargs=3,
        required=True,
        metavar=("Z", "Y", "X"),
        help="Output volume shape (Z, Y, X)",
    )
    parser.add_argument(
        "--chunks",
        type=int,
        nargs=3,
        default=(128, 128, 128),
        metavar=("Z", "Y", "X"),
        help="Chunk size for output zarr (default: 128 128 128)",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=512,
        help="Tile size for processing surfaces (default: 512)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of worker processes (default: half of CPU cores)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="uint8",
        choices=["uint8", "uint16"],
        help="Output dtype (default: uint8)",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=256,
        help="Pixels to expand each component's bounding box (default: 256)",
    )
    parser.add_argument(
        "--padding-value",
        type=int,
        default=None,
        help="Value to write in padded regions (required if --padding > 0)",
    )
    parser.add_argument(
        "--expand",
        type=int,
        default=0,
        help="Expand label by N voxels in 3D using EDT (default: 0, no expansion)",
    )
    parser.add_argument(
        "--padding-expansion",
        type=int,
        default=0,
        help="Expand padding by N voxels in 3D using EDT (default: 0, no expansion)",
    )
    parser.add_argument(
        "--halo",
        type=int,
        default=0,
        help="Halo/overlap size in voxels for expansion chunk processing (default: 0, no overlap)",
    )
    parser.add_argument(
        "--num-levels",
        type=int,
        default=6,
        help="Number of pyramid levels for OME-Zarr (default: 6, levels 0-5)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of tiles to process per batch (default: 50, lower = less memory)",
    )
    parser.add_argument(
        "--remap-json",
        type=str,
        default=None,
        help="Path to JSON file containing value remap dict {original: target}",
    )

    args = parser.parse_args()

    # Validate padding args
    if args.padding > 0 and args.padding_value is None:
        parser.error("--padding-value is required when --padding > 0")

    return args


def discover_surfaces(input_folder: Path, label_suffix: str) -> List[Tuple[Path, Path]]:
    """Discover tifxyz directories with corresponding label files.

    Parameters
    ----------
    input_folder : Path
        Folder containing tifxyz directories.
    label_suffix : str
        Suffix for label files (without extension).

    Returns
    -------
    List[Tuple[Path, Path]]
        List of (tifxyz_dir, label_file) tuples.
    """
    surfaces = []
    for entry in sorted(input_folder.iterdir()):
        if not entry.is_dir():
            continue
        # Check for meta.json to confirm it's a tifxyz directory
        if not (entry / "meta.json").exists():
            continue

        # Look for label file with suffix using flexible glob pattern
        # Matches: *{label_suffix}.png/tif/tiff (e.g., "foo_inklabels.tif", "inklabels.png")
        # Prefer resized version if it exists (from previous run)
        label_file = None
        for ext in [".png", ".tif", ".tiff"]:
            # First look for resized version
            resized_candidates = list(entry.glob(f"*{label_suffix}_resized{ext}"))
            if resized_candidates:
                label_file = resized_candidates[0]
                break
            # Then look for original
            candidates = list(entry.glob(f"*{label_suffix}{ext}"))
            if candidates:
                # If multiple matches, prefer the one that starts with folder name
                for c in candidates:
                    if c.name.startswith(entry.name):
                        label_file = c
                        break
                if label_file is None:
                    label_file = candidates[0]  # Use first match
                break

        if label_file is None:
            raise FileNotFoundError(
                f"No label file found for {entry.name}. "
                f"Expected '*{label_suffix}.png/tif' in {entry}"
            )

        surfaces.append((entry, label_file))

    return surfaces


def read_3d_ink_params(tifxyz_dir: Path) -> dict:
    """Read 3d_ink_params.json if it exists in the tifxyz directory.

    Parameters
    ----------
    tifxyz_dir : Path
        Path to tifxyz directory.

    Returns
    -------
    dict
        Parsed JSON content, or empty dict if file doesn't exist.
    """
    params_file = tifxyz_dir / "3d_ink_params.json"
    if params_file.exists():
        with open(params_file, "r") as f:
            return json.load(f)
    return {}


def load_and_validate_remap(
    remap_path: Path,
    ink_params_labels: Optional[Dict[str, str]],
) -> Tuple[Dict[int, int], set, List[int], int]:
    """Load remap.json and validate against 3d_ink_params labels.

    Supports two formats:
    1. New format: {"mappings": {...}, "padded_labels": [...], "expand_labels": [...]}
    2. Legacy format: {source: target, ...} (no padding info)

    Parameters
    ----------
    remap_path : Path
        Path to remap.json file.
    ink_params_labels : Optional[Dict[str, str]]
        Labels from 3d_ink_params.json (e.g., {"ink": "255", "background": "100"}).
        If None, legacy mode is used without validation.

    Returns
    -------
    Tuple[Dict[int, int], set, List[int], int]
        - mappings: {source_value: target_value}
        - padded_values: set of source values that should get 2D bounding box padding
        - expand_values: ordered list of source values for 3D EDT expansion (later = higher priority)
        - fill_value: the remapped value of "0" (ignore class)

    Raises
    ------
    ValueError
        If validation fails (missing mappings, invalid padded_labels, etc.)
    """
    with open(remap_path, "r") as f:
        raw_data = json.load(f)

    # Detect format
    if "mappings" in raw_data:
        # New format
        raw_mappings = raw_data["mappings"]
        padded_label_names = raw_data.get("padded_labels", [])
        raw_expand_labels = raw_data.get("expand_labels", [])
    else:
        # Legacy format - entire dict is the mappings
        raw_mappings = raw_data
        padded_label_names = []
        raw_expand_labels = []

    # Convert mappings to int keys/values
    mappings = {int(k): int(v) for k, v in raw_mappings.items()}

    # Validate that "0" (ignore) has a mapping
    if 0 not in mappings:
        raise ValueError(
            "remap.json must include a mapping for '0' (ignore/unlabeled class). "
            "This value becomes the fill_value for untouched chunks."
        )

    fill_value = mappings[0]

    # expand_labels are source values (integers), not label names
    # Order is preserved - later items have higher priority (expand last, can overwrite earlier)
    expand_values: List[int] = []
    for val in raw_expand_labels:
        source_val = int(val)
        if source_val not in mappings:
            raise ValueError(
                f"expand_labels contains source value {source_val} which has no mapping. "
                f"Valid source values are: {list(mappings.keys())}"
            )
        if source_val not in expand_values:  # Avoid duplicates
            expand_values.append(source_val)

    # If we have ink_params_labels, validate and build padded_values set
    padded_values = set()
    if ink_params_labels:
        # Validate all label values have mappings
        missing = []
        for name, val_str in ink_params_labels.items():
            val = int(val_str)
            if val not in mappings:
                missing.append(f"{name}={val}")
        if missing:
            raise ValueError(
                f"remap.json is missing mappings for label values: {', '.join(missing)}. "
                f"All values from 3d_ink_params.json labels must be mapped."
            )

        # Validate padded_labels (these ARE label names, for 2D bounding box padding)
        for label_name in padded_label_names:
            if label_name not in ink_params_labels:
                valid_names = list(ink_params_labels.keys())
                raise ValueError(
                    f"padded_labels contains unknown label '{label_name}'. "
                    f"Valid label names are: {valid_names}"
                )
            padded_values.add(int(ink_params_labels[label_name]))

    return mappings, padded_values, expand_values, fill_value


def read_label_image(label_path: Path) -> np.ndarray:
    """Read a label image (PNG or TIF).

    Parameters
    ----------
    label_path : Path
        Path to label image file.

    Returns
    -------
    np.ndarray
        Label image as 2D array.
    """
    if label_path.suffix.lower() in (".tif", ".tiff"):
        img = tifffile.imread(str(label_path))
    else:
        # Use cv2 for PNG
        img = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Failed to read: {label_path}")

    # Convert multi-channel to single channel if needed
    if img.ndim == 3:
        if img.shape[2] == 4:
            # RGBA - take red channel (not grayscale conversion which blends)
            img = img[:, :, 2]  # Red channel in BGR order
        elif img.shape[2] == 3:
            # RGB - take red channel
            img = img[:, :, 2]  # Red channel in BGR order
        elif img.shape[2] == 2:
            # 2-channel image - take first channel (labels)
            img = img[:, :, 0]
        else:
            # Unknown multi-channel format - take first channel
            img = img[:, :, 0]

    # Snap anti-aliased values to nearest discrete label (0, 100, 255)
    # This handles images that were saved with anti-aliasing
    unique_vals = np.unique(img)
    if len(unique_vals) > 10:
        # Likely anti-aliased - snap to nearest target value
        # Thresholds: 0-50 → 0, 51-177 → 100, 178-255 → 255
        original_dtype = img.dtype
        result = np.zeros_like(img)
        result[(img > 50) & (img <= 177)] = 100
        result[img > 177] = 255
        img = result.astype(original_dtype)

    return img


def find_padded_regions(
    label_img: np.ndarray,
    label_value: int,
    padding: int,
) -> List[Tuple[int, int, int, int]]:
    """Find connected components and compute padded bounding boxes.

    Parameters
    ----------
    label_img : np.ndarray
        Label image.
    label_value : int
        Target label value.
    padding : int
        Pixels to expand each bounding box.

    Returns
    -------
    List[Tuple[int, int, int, int]]
        List of (row_start, row_end, col_start, col_end) for each component.
    """
    full_h, full_w = label_img.shape[:2]

    if padding == 0:
        # No padding - return single bbox covering all label pixels
        binary_mask = label_img == label_value
        rows, cols = np.where(binary_mask)
        if len(rows) == 0:
            return []
        return [(rows.min(), rows.max() + 1, cols.min(), cols.max() + 1)]

    # Find connected components
    binary_mask = label_img == label_value
    labeled, num_components = label_components(binary_mask)

    if num_components == 0:
        return []

    slices = find_objects(labeled)
    regions = []

    for slice_tuple in slices:
        if slice_tuple is None:
            continue
        row_slice, col_slice = slice_tuple

        # Expand bbox by padding
        row_start = max(0, row_slice.start - padding)
        row_end = min(full_h, row_slice.stop + padding)
        col_start = max(0, col_slice.start - padding)
        col_end = min(full_w, col_slice.stop + padding)

        regions.append((row_start, row_end, col_start, col_end))

    return regions


def write_points_to_zarr(
    zarr_path: str,
    z_voxels: np.ndarray,
    y_voxels: np.ndarray,
    x_voxels: np.ndarray,
    values: np.ndarray,
    volume_shape: Tuple[int, int, int],
) -> None:
    """Write voxel points to zarr array.

    Parameters
    ----------
    zarr_path : str
        Path to zarr array.
    z_voxels : np.ndarray
        Z coordinates.
    y_voxels : np.ndarray
        Y coordinates.
    x_voxels : np.ndarray
        X coordinates.
    values : np.ndarray
        Values to write at each coordinate.
    volume_shape : Tuple[int, int, int]
        Shape of the output volume.
    """
    if len(z_voxels) == 0:
        return

    # Clip to volume bounds
    valid = (
        (z_voxels >= 0) & (z_voxels < volume_shape[0]) &
        (y_voxels >= 0) & (y_voxels < volume_shape[1]) &
        (x_voxels >= 0) & (x_voxels < volume_shape[2])
    )

    z_voxels = z_voxels[valid]
    y_voxels = y_voxels[valid]
    x_voxels = x_voxels[valid]
    values = values[valid]

    if len(z_voxels) == 0:
        return

    zarr_array = zarr.open(zarr_path, mode="r+")
    chunks = zarr_array.chunks

    # Group points by chunk
    chunk_z = z_voxels // chunks[0]
    chunk_y = y_voxels // chunks[1]
    chunk_x = x_voxels // chunks[2]

    # Create unique chunk identifiers
    max_cy = (volume_shape[1] // chunks[1]) + 1
    max_cx = (volume_shape[2] // chunks[2]) + 1
    chunk_ids = (
        chunk_z.astype(np.int64) * max_cy * max_cx +
        chunk_y.astype(np.int64) * max_cx +
        chunk_x.astype(np.int64)
    )

    unique_chunks = np.unique(chunk_ids)

    for chunk_id in unique_chunks:
        mask = chunk_ids == chunk_id
        cz = int(chunk_z[mask][0])
        cy = int(chunk_y[mask][0])
        cx = int(chunk_x[mask][0])

        # Compute chunk bounds
        z_start = cz * chunks[0]
        y_start = cy * chunks[1]
        x_start = cx * chunks[2]
        z_end = min(z_start + chunks[0], volume_shape[0])
        y_end = min(y_start + chunks[1], volume_shape[1])
        x_end = min(x_start + chunks[2], volume_shape[2])

        # Read current chunk
        chunk_data = zarr_array[z_start:z_end, y_start:y_end, x_start:x_end]

        # Convert global coords to local chunk coords
        local_z = z_voxels[mask] - z_start
        local_y = y_voxels[mask] - y_start
        local_x = x_voxels[mask] - x_start

        # Write points
        chunk_data[local_z, local_y, local_x] = values[mask]

        # Write chunk back
        zarr_array[z_start:z_end, y_start:y_end, x_start:x_end] = chunk_data


def get_initialized_chunk_indices(zarr_array: zarr.Array) -> set:
    """Get set of (cz, cy, cx) tuples for chunks that have data.

    Parameters
    ----------
    zarr_array : zarr.Array
        The zarr array to inspect.

    Returns
    -------
    set
        Set of chunk index tuples.
    """
    chunk_keys = [k for k in zarr_array.store.keys() if not k.startswith('.')]
    indices = set()
    for key in chunk_keys:
        try:
            parts = tuple(map(int, key.split('.')))
            if len(parts) == 3:
                indices.add(parts)
        except ValueError:
            continue
    return indices


def get_neighbor_chunk_indices(
    chunk_idx: Tuple[int, int, int],
    expand_distance: int,
    chunks: Tuple[int, int, int],
    shape: Tuple[int, int, int],
) -> set:
    """Get chunk indices within expand_distance of a given chunk.

    Parameters
    ----------
    chunk_idx : Tuple[int, int, int]
        The reference chunk index (cz, cy, cx).
    expand_distance : int
        Distance in voxels to expand.
    chunks : Tuple[int, int, int]
        Chunk size (z, y, x).
    shape : Tuple[int, int, int]
        Volume shape (z, y, x).

    Returns
    -------
    set
        Set of chunk index tuples that are within expand_distance.
    """
    cz, cy, cx = chunk_idx
    chunk_z, chunk_y, chunk_x = chunks

    # How many chunks away can be affected?
    dz = (expand_distance // chunk_z) + 1
    dy = (expand_distance // chunk_y) + 1
    dx = (expand_distance // chunk_x) + 1

    max_cz = (shape[0] - 1) // chunk_z
    max_cy = (shape[1] - 1) // chunk_y
    max_cx = (shape[2] - 1) // chunk_x

    neighbors = set()
    for oz in range(-dz, dz + 1):
        for oy in range(-dy, dy + 1):
            for ox in range(-dx, dx + 1):
                nz, ny, nx = cz + oz, cy + oy, cx + ox
                if 0 <= nz <= max_cz and 0 <= ny <= max_cy and 0 <= nx <= max_cx:
                    neighbors.add((nz, ny, nx))
    return neighbors


def expand_chunk_worker(args: Tuple) -> dict:
    """Process a single chunk for expansion.

    Parameters
    ----------
    args : Tuple
        (chunk_idx, zarr_path, label_expand_distance, padding_expand_distance,
         label_value, padding_value, fill_value, chunks, shape, halo)

    Returns
    -------
    dict
        Result with chunk_idx and expanded_voxels count.
    """
    (chunk_idx, zarr_path, label_expand_distance, padding_expand_distance,
     label_value, padding_value, fill_value, chunks, shape, halo) = args

    cz, cy, cx = chunk_idx
    chunk_z, chunk_y, chunk_x = chunks

    # Compute chunk bounds
    z_start = cz * chunk_z
    y_start = cy * chunk_y
    x_start = cx * chunk_x
    z_end = min(z_start + chunk_z, shape[0])
    y_end = min(y_start + chunk_y, shape[1])
    x_end = min(x_start + chunk_x, shape[2])

    # Compute extended bounds with configurable halo
    # halo=0 means each chunk processes only its exact region
    z_start_ext = max(0, z_start - halo)
    y_start_ext = max(0, y_start - halo)
    x_start_ext = max(0, x_start - halo)
    z_end_ext = min(shape[0], z_end + halo)
    y_end_ext = min(shape[1], y_end + halo)
    x_end_ext = min(shape[2], x_end + halo)

    zarr_array = zarr.open(zarr_path, mode='r+')

    # Read extended region
    extended_data = zarr_array[z_start_ext:z_end_ext,
                               y_start_ext:y_end_ext,
                               x_start_ext:x_end_ext]

    expanded_count = 0

    # Step 1: Expand labels (into empty OR padding voxels)
    # When expanding the padding label itself, use padding_expand_distance instead
    # This ensures --expand controls non-padding labels, --padding-expansion controls padding
    effective_expand_distance = label_expand_distance
    if padding_value is not None and label_value == padding_value:
        effective_expand_distance = padding_expand_distance

    if effective_expand_distance > 0:
        label_mask = (extended_data == label_value)
        if label_mask.any():
            label_edt = distance_transform_edt(~label_mask)
            label_expand_mask = (label_edt > 0) & (label_edt <= effective_expand_distance)

            # Expand into empty voxels OR padding voxels
            can_expand_into = (extended_data == fill_value)
            if padding_value is not None:
                can_expand_into = can_expand_into | (extended_data == padding_value)
            label_expand_mask = label_expand_mask & can_expand_into

            if label_expand_mask.any():
                extended_data[label_expand_mask] = label_value
                expanded_count = 1

    # Extract inner chunk region
    inner_z = z_start - z_start_ext
    inner_y = y_start - y_start_ext
    inner_x = x_start - x_start_ext
    inner_z_end = inner_z + (z_end - z_start)
    inner_y_end = inner_y + (y_end - y_start)
    inner_x_end = inner_x + (x_end - x_start)

    # Only write back if we actually expanded something
    if expanded_count > 0:
        inner_data = extended_data[inner_z:inner_z_end,
                                   inner_y:inner_y_end,
                                   inner_x:inner_x_end]
        zarr_array[z_start:z_end, y_start:y_end, x_start:x_end] = inner_data

    return {'chunk_idx': chunk_idx, 'expanded_voxels': expanded_count}


def get_chunk_phase(chunk_idx: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Get the phase for a chunk based on its index.

    Chunks are grouped into 8 phases based on (cz % 2, cy % 2, cx % 2).
    Chunks in the same phase are at least 2 chunks apart in every dimension,
    so their halos don't overlap each other's inner regions.

    Parameters
    ----------
    chunk_idx : Tuple[int, int, int]
        Chunk index (cz, cy, cx).

    Returns
    -------
    Tuple[int, int, int]
        Phase (0 or 1 for each dimension).
    """
    cz, cy, cx = chunk_idx
    return (cz % 2, cy % 2, cx % 2)


# Define phase order (process in consistent order for determinism)
PHASE_ORDER = [
    (0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
    (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1),
]


def process_chunks_in_phases(
    chunk_indices: set,
    worker_func,
    worker_args_builder,
    num_workers: int = 1,
    description: str = "Processing",
) -> List[Any]:
    """Process chunks in 8 phases to avoid race conditions.

    Chunks are grouped by (cz % 2, cy % 2, cx % 2). Chunks in the same
    phase are at least 2 chunks apart, so they can be processed in parallel
    without conflicts. Phases are processed sequentially.

    Parameters
    ----------
    chunk_indices : set
        Set of (cz, cy, cx) chunk indices to process.
    worker_func : Callable
        Function to call for each chunk. Takes args from worker_args_builder.
    worker_args_builder : Callable[[Tuple[int, int, int]], Any]
        Function that takes chunk_idx and returns args for worker_func.
    num_workers : int
        Number of parallel workers per phase.
    description : str
        Description for progress bar.

    Returns
    -------
    List[Any]
        Results from all worker calls.
    """
    # Group chunks by phase
    phases: Dict[Tuple[int, int, int], List[Tuple[int, int, int]]] = {}
    for chunk_idx in chunk_indices:
        phase = get_chunk_phase(chunk_idx)
        if phase not in phases:
            phases[phase] = []
        phases[phase].append(chunk_idx)

    all_results = []

    for phase in PHASE_ORDER:
        if phase not in phases:
            continue

        phase_chunks = sorted(phases[phase])

        # Build worker args for this phase
        worker_args = [worker_args_builder(chunk_idx) for chunk_idx in phase_chunks]

        if num_workers > 1 and len(phase_chunks) > 1:
            with Pool(processes=num_workers) as pool:
                results = list(tqdm(
                    pool.imap_unordered(worker_func, worker_args),
                    total=len(worker_args),
                    desc=f"{description} phase {phase}",
                ))
                all_results.extend(results)
        else:
            for args in tqdm(worker_args, desc=f"{description} phase {phase}"):
                result = worker_func(args)
                all_results.append(result)

    return all_results


def expand_labels_in_zarr(
    zarr_path: str,
    expand_distance: int,
    label_value: int,
    padding_value: Optional[int] = None,
    padding_expansion: int = 0,
    fill_value: int = 0,
    num_workers: int = 1,
    halo: int = 0,
) -> int:
    """Expand labels in zarr volume using EDT.

    Uses an 8-phase approach for deterministic parallel processing:
    - Chunks are grouped by (cz % 2, cy % 2, cx % 2) into 8 phases
    - Chunks in the same phase are 2 chunks apart in every dimension
    - Phases are processed sequentially, but chunks within each phase run in parallel
    - This ensures no race conditions while maintaining parallelism

    Parameters
    ----------
    zarr_path : str
        Path to zarr array to modify in-place.
    expand_distance : int
        Distance in voxels to expand the label.
    label_value : int
        The label value to expand.
    padding_value : Optional[int]
        If provided, labels can expand into voxels with this value.
    padding_expansion : int
        Distance in voxels to expand the padding value (default: 0).
    fill_value : int
        The fill value used for empty voxels (default: 0).
    num_workers : int
        Number of parallel workers.
    halo : int
        Halo/overlap size in voxels (default: 0, no overlap).

    Returns
    -------
    int
        Total number of chunks modified by expansion.
    """
    zarr_array = zarr.open(zarr_path, mode='r')
    shape = zarr_array.shape
    chunks = zarr_array.chunks

    # Find chunks with data
    initialized_chunks = get_initialized_chunk_indices(zarr_array)
    print(f"Found {len(initialized_chunks)} initialized chunks")

    if not initialized_chunks:
        print("No data in volume, skipping expansion")
        return 0

    # Find all chunks that need processing (neighbors of initialized chunks)
    max_expansion = max(expand_distance, padding_expansion)
    chunks_to_process = set()
    for chunk_idx in initialized_chunks:
        neighbors = get_neighbor_chunk_indices(
            chunk_idx, max_expansion, chunks, shape
        )
        chunks_to_process.update(neighbors)

    print(f"Will process {len(chunks_to_process)} chunks for expansion")

    # Build args builder for expand_chunk_worker
    def build_expand_args(chunk_idx: Tuple[int, int, int]):
        return (chunk_idx, zarr_path, expand_distance, padding_expansion,
                label_value, padding_value, fill_value, chunks, shape, halo)

    # Process chunks in phases
    results = process_chunks_in_phases(
        chunks_to_process,
        expand_chunk_worker,
        build_expand_args,
        num_workers=num_workers,
        description="Expanding",
    )

    total_expanded = sum(r['expanded_voxels'] for r in results)
    return total_expanded


@dataclass
class TileTask:
    """A single tile to process."""
    surface_dir: str
    label_file: str
    region_bounds: Tuple[int, int, int, int]  # (row_start, row_end, col_start, col_end)
    tile_bounds: Tuple[int, int, int, int]  # (tile_row_start, tile_row_end, tile_col_start, tile_col_end)
    multi_label_mode: bool = False  # If True, use multi-label mode from 3d_ink_params.json
    ink_labels: Optional[Dict[str, str]] = None  # From 3d_ink_params.json {"name": "value"}


def preprocess_surface(
    tifxyz_dir: Path,
    label_file: Path,
    label_value: Optional[int],
    padding: int,
    tile_size: int,
    multi_label_mode: bool = False,
    ink_labels: Optional[Dict[str, str]] = None,
    padded_values: Optional[set] = None,
) -> List[TileTask]:
    """Pre-process a surface to find regions and generate tile tasks.

    Parameters
    ----------
    tifxyz_dir : Path
        Path to tifxyz directory.
    label_file : Path
        Path to label image file.
    label_value : Optional[int]
        Target label value (used in single-label mode).
    padding : int
        Pixels to expand each component's bounding box.
    tile_size : int
        Tile size for processing.
    multi_label_mode : bool
        If True, use multi-label mode with labels from 3d_ink_params.json.
    ink_labels : Optional[Dict[str, str]]
        Labels from 3d_ink_params.json (e.g., {"ink": "255", "background": "100"}).
    padded_values : Optional[set]
        Set of source values that should get padding in multi-label mode.

    Returns
    -------
    List[TileTask]
        List of tile tasks for this surface.
    """
    # Load label image
    label_img = read_label_image(label_file)

    # Load surface to get shape - use full resolution to match label images
    surface = read_tifxyz(tifxyz_dir, load_mask=True)
    surface.use_full_resolution()
    surf_h, surf_w = surface.shape

    # Handle shape mismatch
    label_h, label_w = label_img.shape[:2]
    if label_h != surf_h or label_w != surf_w:
        # Calculate scale ratios
        h_ratio = surf_h / label_h
        w_ratio = surf_w / label_w

        # If label is significantly smaller (more than 1.5x in either dimension),
        # resize it instead of padding
        if h_ratio > 1.5 or w_ratio > 1.5:
            # Use nearest-neighbor interpolation to preserve label values
            label_img = cv2.resize(
                label_img,
                (surf_w, surf_h),  # cv2 uses (width, height)
                interpolation=cv2.INTER_NEAREST
            )
            print(
                f"Resizing label from ({label_h}, {label_w}) to ({surf_h}, {surf_w}) "
                f"for {tifxyz_dir.name} (scale: {h_ratio:.2f}x, {w_ratio:.2f}x)"
            )

            # Save the resized label back to the source folder
            resized_path = label_file.parent / f"{label_file.stem}_resized{label_file.suffix}"
            if label_file.suffix.lower() in (".tif", ".tiff"):
                tifffile.imwrite(str(resized_path), label_img)
            else:
                cv2.imwrite(str(resized_path), label_img)
            print(f"  Saved resized label to: {resized_path}")
        else:
            # Small difference - use padding/cropping
            new_label = np.zeros((surf_h, surf_w), dtype=label_img.dtype)
            copy_h = min(label_h, surf_h)
            copy_w = min(label_w, surf_w)
            new_label[:copy_h, :copy_w] = label_img[:copy_h, :copy_w]

            action = []
            if label_h > surf_h or label_w > surf_w:
                action.append("cropping")
            if label_h < surf_h or label_w < surf_w:
                action.append("padding")
            print(
                f"Adjusting label from ({label_h}, {label_w}) to ({surf_h}, {surf_w}) "
                f"for {tifxyz_dir.name} ({'/'.join(action)})"
            )
            label_img = new_label

    # In multi-label mode, find regions based on padded labels
    if multi_label_mode:
        if padded_values:
            # Find regions for each padded label type
            all_regions = []
            for padded_val in padded_values:
                regions = find_padded_regions(label_img, padded_val, padding)
                all_regions.extend(regions)
            # If no padded values, use single region covering entire image
            if not all_regions:
                regions = [(0, surf_h, 0, surf_w)]
            else:
                regions = all_regions
        else:
            # No padded labels specified, use single region covering entire image
            regions = [(0, surf_h, 0, surf_w)]
    else:
        # Single-label mode: find regions (connected components with padding)
        regions = find_padded_regions(label_img, label_value, padding)

    if not regions:
        return []

    # Generate tile tasks for each region
    tasks = []
    for region_bounds in regions:
        row_start, row_end, col_start, col_end = region_bounds

        for tile_row in range(row_start, row_end, tile_size):
            for tile_col in range(col_start, col_end, tile_size):
                tile_row_end = min(tile_row + tile_size, row_end)
                tile_col_end = min(tile_col + tile_size, col_end)

                tasks.append(TileTask(
                    surface_dir=str(tifxyz_dir),
                    label_file=str(label_file),
                    region_bounds=region_bounds,
                    tile_bounds=(tile_row, tile_row_end, tile_col, tile_col_end),
                    multi_label_mode=multi_label_mode,
                    ink_labels=ink_labels,
                ))

    return tasks


# Global state for worker processes
_WORKER_STATE: Dict[str, Any] = {}

# Cache for loaded surfaces and labels in workers
_SURFACE_CACHE: Dict[str, Any] = {}
_LABEL_CACHE: Dict[str, np.ndarray] = {}


def _init_worker(state: Dict[str, Any]) -> None:
    """Initialize worker state."""
    global _WORKER_STATE, _SURFACE_CACHE, _LABEL_CACHE
    _WORKER_STATE = state
    _SURFACE_CACHE = {}
    _LABEL_CACHE = {}


def _get_surface_and_label(
    surface_dir: str,
    label_file: str,
) -> Tuple[Any, np.ndarray]:
    """Get surface and label from cache or load them.

    Parameters
    ----------
    surface_dir : str
        Path to tifxyz directory.
    label_file : str
        Path to label file.

    Returns
    -------
    Tuple[Any, np.ndarray]
        (surface, label_img) tuple.
    """
    global _SURFACE_CACHE, _LABEL_CACHE

    # Load surface if not cached - use full resolution to match label images
    if surface_dir not in _SURFACE_CACHE:
        surface = read_tifxyz(Path(surface_dir), load_mask=True)
        surface.use_full_resolution()
        _SURFACE_CACHE[surface_dir] = surface

    # Load label if not cached
    if label_file not in _LABEL_CACHE:
        label_img = read_label_image(Path(label_file))
        surface = _SURFACE_CACHE[surface_dir]
        surf_h, surf_w = surface.shape

        # Handle shape mismatch
        label_h, label_w = label_img.shape[:2]
        if label_h != surf_h or label_w != surf_w:
            new_label = np.zeros((surf_h, surf_w), dtype=label_img.dtype)
            copy_h = min(label_h, surf_h)
            copy_w = min(label_w, surf_w)
            new_label[:copy_h, :copy_w] = label_img[:copy_h, :copy_w]
            label_img = new_label

        _LABEL_CACHE[label_file] = label_img

    return _SURFACE_CACHE[surface_dir], _LABEL_CACHE[label_file]


def process_tile_worker(task: TileTask) -> int:
    """Process a single tile (worker function).

    Parameters
    ----------
    task : TileTask
        Tile task containing surface, label, region, and tile bounds.

    Returns
    -------
    int
        Number of points written.
    """
    global _WORKER_STATE
    state = _WORKER_STATE

    label_value = state["label_value"]
    padding = state["padding"]
    padding_value = state["padding_value"]
    output_path = state["output_path"]
    volume_shape = state["volume_shape"]

    # Get surface and label (from cache if available)
    surface, label_img = _get_surface_and_label(task.surface_dir, task.label_file)

    # Extract tile bounds
    tile_row, tile_row_end, tile_col, tile_col_end = task.tile_bounds

    # Get coordinates for this tile
    x, y, z, valid = surface[tile_row:tile_row_end, tile_col:tile_col_end]

    # Get labels for this tile
    tile_labels = label_img[tile_row:tile_row_end, tile_col:tile_col_end]

    # Handle multi-label mode: write entire tile if it has labeled content
    if task.multi_label_mode:
        # Get labeled values from ink_labels (e.g., {"ink": "255", "background": "100"})
        ink_labels = task.ink_labels or {}
        labeled_values = set(int(v) for v in ink_labels.values())

        # Check if this tile has ANY labeled content (not just source=0)
        has_labeled_content = False
        for labeled_val in labeled_values:
            if ((tile_labels == labeled_val) & valid).any():
                has_labeled_content = True
                break

        if not has_labeled_content:
            return 0  # No labeled content, skip tile entirely

        # Tile has labeled content → write ENTIRE tile footprint (all valid pixels)
        z_coords = np.floor(z[valid]).astype(np.int64)
        y_coords = np.floor(y[valid]).astype(np.int64)
        x_coords = np.floor(x[valid]).astype(np.int64)
        # Use source values from the label image (will be remapped)
        values = tile_labels[valid].astype(np.uint8)

        # Apply 2D EDT padding: unlabeled pixels within padding distance become padding_value
        if padding > 0 and padding_value is not None:
            padded_values = state.get("padded_values", set())
            if padded_values:
                # Build mask of labeled pixels that should get padding halos
                labeled_mask = np.zeros_like(tile_labels, dtype=bool)
                for padded_val in padded_values:
                    labeled_mask |= (tile_labels == padded_val)

                if labeled_mask.any():
                    # EDT: distance from each pixel to nearest labeled pixel
                    edt = scipy_edt(~labeled_mask)
                    # Padding mask: within distance AND currently unlabeled (value 0)
                    is_padding = (edt > 0) & (edt <= padding) & (tile_labels == 0)
                    # Apply to values array (which corresponds to valid pixels)
                    values[is_padding[valid]] = padding_value

        # Apply value remapping (required in multi-label mode)
        remap_dict = state.get("remap_dict")
        if remap_dict:
            values = fastremap.remap(values, remap_dict, preserve_missing_labels=True, in_place=False)

        write_points_to_zarr(
            output_path,
            z_coords,
            y_coords,
            x_coords,
            values,
            volume_shape,
        )
        return len(z_coords)

    # Normal mode: filter by label_value with padding
    # Create masks
    is_label = (tile_labels == label_value) & valid

    # Collect all points to write
    all_z = []
    all_y = []
    all_x = []
    all_values = []

    # If padding is enabled, write padding value first
    if padding > 0 and padding_value is not None:
        # Padded region = valid but not label
        is_padding = valid & ~is_label

        if is_padding.any():
            z_coords = np.floor(z[is_padding]).astype(np.int64)
            y_coords = np.floor(y[is_padding]).astype(np.int64)
            x_coords = np.floor(x[is_padding]).astype(np.int64)
            pad_values = np.full(len(z_coords), padding_value, dtype=np.uint8)

            all_z.append(z_coords)
            all_y.append(y_coords)
            all_x.append(x_coords)
            all_values.append(pad_values)

    # Write label value (overwrites padding)
    if is_label.any():
        z_coords = np.floor(z[is_label]).astype(np.int64)
        y_coords = np.floor(y[is_label]).astype(np.int64)
        x_coords = np.floor(x[is_label]).astype(np.int64)
        label_values_arr = np.full(len(z_coords), label_value, dtype=np.uint8)

        all_z.append(z_coords)
        all_y.append(y_coords)
        all_x.append(x_coords)
        all_values.append(label_values_arr)

    # Write all points
    if all_z:
        z_voxels = np.concatenate(all_z)
        y_voxels = np.concatenate(all_y)
        x_voxels = np.concatenate(all_x)
        values = np.concatenate(all_values)

        # Apply value remapping if configured
        remap_dict = state.get("remap_dict")
        if remap_dict:
            values = fastremap.remap(values, remap_dict, preserve_missing_labels=True, in_place=False)

        write_points_to_zarr(
            output_path,
            z_voxels,
            y_voxels,
            x_voxels,
            values,
            volume_shape,
        )
        return len(z_voxels)

    return 0


def collect_tile_points(task: TileTask) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Collect points from a single tile without writing.

    Parameters
    ----------
    task : TileTask
        Tile task containing surface, label, region, and tile bounds.

    Returns
    -------
    Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
        (z_coords, y_coords, x_coords, values) or None if no points.
    """
    global _WORKER_STATE
    state = _WORKER_STATE

    label_value = state["label_value"]
    padding = state["padding"]
    padding_value = state["padding_value"]
    volume_shape = state["volume_shape"]

    # Get surface and label (from cache if available)
    surface, label_img = _get_surface_and_label(task.surface_dir, task.label_file)

    # Extract tile bounds
    tile_row, tile_row_end, tile_col, tile_col_end = task.tile_bounds

    # Get coordinates for this tile
    x, y, z, valid = surface[tile_row:tile_row_end, tile_col:tile_col_end]

    # Get labels for this tile
    tile_labels = label_img[tile_row:tile_row_end, tile_col:tile_col_end]

    # Handle multi-label mode: collect entire tile if it has labeled content
    if task.multi_label_mode:
        # Get labeled values from ink_labels (e.g., {"ink": "255", "background": "100"})
        ink_labels = task.ink_labels or {}
        labeled_values = set(int(v) for v in ink_labels.values())

        # Check if this tile has ANY labeled content (not just source=0)
        has_labeled_content = False
        for labeled_val in labeled_values:
            if ((tile_labels == labeled_val) & valid).any():
                has_labeled_content = True
                break

        if not has_labeled_content:
            return None  # No labeled content, skip tile entirely

        # Tile has labeled content → collect ENTIRE tile footprint (all valid pixels)
        z_coords = np.floor(z[valid]).astype(np.int64)
        y_coords = np.floor(y[valid]).astype(np.int64)
        x_coords = np.floor(x[valid]).astype(np.int64)
        values = tile_labels[valid].astype(np.uint8)

        # Apply 2D EDT padding: unlabeled pixels within padding distance become padding_value
        padding = state["padding"]
        padding_value = state["padding_value"]
        if padding > 0 and padding_value is not None:
            padded_values = state.get("padded_values", set())
            if padded_values:
                # Build mask of labeled pixels that should get padding halos
                labeled_mask = np.zeros_like(tile_labels, dtype=bool)
                for padded_val in padded_values:
                    labeled_mask |= (tile_labels == padded_val)

                if labeled_mask.any():
                    # EDT: distance from each pixel to nearest labeled pixel
                    edt = scipy_edt(~labeled_mask)
                    # Padding mask: within distance AND currently unlabeled (value 0)
                    is_padding = (edt > 0) & (edt <= padding) & (tile_labels == 0)
                    # Apply to values array (which corresponds to valid pixels)
                    values[is_padding[valid]] = padding_value

        # Apply value remapping (required in multi-label mode)
        remap_dict = state.get("remap_dict")
        if remap_dict:
            values = fastremap.remap(values, remap_dict, preserve_missing_labels=True, in_place=False)

        # Clip to volume bounds
        valid_mask = (
            (z_coords >= 0) & (z_coords < volume_shape[0]) &
            (y_coords >= 0) & (y_coords < volume_shape[1]) &
            (x_coords >= 0) & (x_coords < volume_shape[2])
        )
        if valid_mask.any():
            return (z_coords[valid_mask], y_coords[valid_mask],
                    x_coords[valid_mask], values[valid_mask])
        return None

    # Normal mode: filter by label_value with padding
    is_label = (tile_labels == label_value) & valid

    all_z = []
    all_y = []
    all_x = []
    all_values = []

    # Collect padding points first
    if padding > 0 and padding_value is not None:
        is_padding = valid & ~is_label

        if is_padding.any():
            z_coords = np.floor(z[is_padding]).astype(np.int64)
            y_coords = np.floor(y[is_padding]).astype(np.int64)
            x_coords = np.floor(x[is_padding]).astype(np.int64)
            pad_values = np.full(len(z_coords), padding_value, dtype=np.uint8)

            all_z.append(z_coords)
            all_y.append(y_coords)
            all_x.append(x_coords)
            all_values.append(pad_values)

    # Collect label points (these will overwrite padding when written)
    if is_label.any():
        z_coords = np.floor(z[is_label]).astype(np.int64)
        y_coords = np.floor(y[is_label]).astype(np.int64)
        x_coords = np.floor(x[is_label]).astype(np.int64)
        label_values_arr = np.full(len(z_coords), label_value, dtype=np.uint8)

        all_z.append(z_coords)
        all_y.append(y_coords)
        all_x.append(x_coords)
        all_values.append(label_values_arr)

    if all_z:
        z_voxels = np.concatenate(all_z)
        y_voxels = np.concatenate(all_y)
        x_voxels = np.concatenate(all_x)
        values = np.concatenate(all_values)

        # Apply value remapping if configured
        remap_dict = state.get("remap_dict")
        if remap_dict:
            values = fastremap.remap(values, remap_dict, preserve_missing_labels=True, in_place=False)

        # Clip to volume bounds
        valid_mask = (
            (z_voxels >= 0) & (z_voxels < volume_shape[0]) &
            (y_voxels >= 0) & (y_voxels < volume_shape[1]) &
            (x_voxels >= 0) & (x_voxels < volume_shape[2])
        )
        if valid_mask.any():
            return (z_voxels[valid_mask], y_voxels[valid_mask],
                    x_voxels[valid_mask], values[valid_mask])

    return None


def accumulate_points_to_chunks(
    result: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    chunk_points: Dict[Tuple[int, int, int], List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]],
    chunks: Tuple[int, int, int],
) -> None:
    """Accumulate points from one tile into chunk dictionary.

    Groups points by their chunk index and appends to the chunk's list.
    This allows processing tiles in batches without holding all points in memory.

    Parameters
    ----------
    result : Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        (z, y, x, values) arrays from collect_tile_points.
    chunk_points : Dict
        Dictionary mapping chunk_idx -> list of (z, y, x, values) tuples.
    chunks : Tuple[int, int, int]
        Chunk size (z, y, x).
    """
    z, y, x, values = result

    # Compute chunk indices for each point
    cz = z // chunks[0]
    cy = y // chunks[1]
    cx = x // chunks[2]

    # Find unique chunk indices efficiently
    max_cy = 100000  # Large enough multiplier
    max_cx = 100000
    chunk_ids = cz * max_cy * max_cx + cy * max_cx + cx
    unique_ids = np.unique(chunk_ids)

    for uid in unique_ids:
        mask = chunk_ids == uid
        # Recover chunk index from unique id
        cz_val = int(uid // (max_cy * max_cx))
        remainder = uid % (max_cy * max_cx)
        cy_val = int(remainder // max_cx)
        cx_val = int(remainder % max_cx)
        chunk_idx = (cz_val, cy_val, cx_val)

        if chunk_idx not in chunk_points:
            chunk_points[chunk_idx] = []
        chunk_points[chunk_idx].append((
            z[mask], y[mask], x[mask], values[mask]
        ))


def write_chunk_points_worker(args: Tuple) -> int:
    """Write all points for a single chunk (worker function).

    Parameters
    ----------
    args : Tuple
        (chunk_idx, zarr_path, z_coords, y_coords, x_coords, values, chunks, shape)

    Returns
    -------
    int
        Number of points written.
    """
    chunk_idx, zarr_path, z_coords, y_coords, x_coords, values, chunks, shape = args
    cz, cy, cx = chunk_idx

    # Compute chunk bounds
    z_start = cz * chunks[0]
    y_start = cy * chunks[1]
    x_start = cx * chunks[2]
    z_end = min(z_start + chunks[0], shape[0])
    y_end = min(y_start + chunks[1], shape[1])
    x_end = min(x_start + chunks[2], shape[2])

    zarr_array = zarr.open(zarr_path, mode='r+')

    # Read current chunk
    chunk_data = zarr_array[z_start:z_end, y_start:y_end, x_start:x_end]

    # Convert global coords to local chunk coords
    local_z = z_coords - z_start
    local_y = y_coords - y_start
    local_x = x_coords - x_start

    # Write points (later points overwrite earlier ones, so labels overwrite padding)
    chunk_data[local_z, local_y, local_x] = values

    # Write chunk back
    zarr_array[z_start:z_end, y_start:y_end, x_start:x_end] = chunk_data

    return len(values)


def create_ome_zarr_metadata(
    shape: Tuple[int, int, int],
    num_levels: int,
    dtype: str,
) -> dict:
    """Create OME-Zarr multiscales metadata.

    Parameters
    ----------
    shape : Tuple[int, int, int]
        Shape of level 0 (Z, Y, X).
    num_levels : int
        Number of pyramid levels.
    dtype : str
        Data type string.

    Returns
    -------
    dict
        OME-Zarr .zattrs content.
    """
    datasets = []
    for level in range(num_levels):
        scale = 2 ** level
        datasets.append({
            "path": str(level),
            "coordinateTransformations": [
                {
                    "type": "scale",
                    "scale": [float(scale), float(scale), float(scale)]
                }
            ]
        })

    return {
        "multiscales": [{
            "version": "0.4",
            "name": "labels",
            "axes": [
                {"name": "z", "type": "space", "unit": "micrometer"},
                {"name": "y", "type": "space", "unit": "micrometer"},
                {"name": "x", "type": "space", "unit": "micrometer"},
            ],
            "datasets": datasets,
            "type": "gaussian",
            "metadata": {
                "method": "cv2.resize with INTER_NEAREST",
            }
        }]
    }


def _downsample_chunk_worker(args: Tuple) -> Optional[Tuple[Tuple[int, int, int], np.ndarray]]:
    """Worker function to downsample a single output chunk.

    Parameters
    ----------
    args : Tuple
        (out_chunk_idx, input_zarr_path, prev_level, prev_shape, out_shape, out_chunks)

    Returns
    -------
    Optional[Tuple[Tuple[int, int, int], np.ndarray]]
        (output_chunk_index, downsampled_data) or None if empty.
    """
    out_chunk_idx, input_zarr_path, prev_level, prev_shape, out_shape, out_chunks = args
    cz, cy, cx = out_chunk_idx

    # Calculate output chunk bounds
    z_out_start = cz * out_chunks[0]
    y_out_start = cy * out_chunks[1]
    x_out_start = cx * out_chunks[2]
    z_out_end = min(z_out_start + out_chunks[0], out_shape[0])
    y_out_end = min(y_out_start + out_chunks[1], out_shape[1])
    x_out_end = min(x_out_start + out_chunks[2], out_shape[2])

    # Calculate corresponding input bounds (2x)
    z_in_start = z_out_start * 2
    y_in_start = y_out_start * 2
    x_in_start = x_out_start * 2
    z_in_end = min(z_out_end * 2, prev_shape[0])
    y_in_end = min(y_out_end * 2, prev_shape[1])
    x_in_end = min(x_out_end * 2, prev_shape[2])

    # Read only the required input region
    root = zarr.open(input_zarr_path, mode="r")
    prev_arr = root[str(prev_level)]
    input_chunk = prev_arr[z_in_start:z_in_end, y_in_start:y_in_end, x_in_start:x_in_end]

    # Skip empty chunks (all values equal to fill_value)
    fill_value = prev_arr.fill_value if prev_arr.fill_value is not None else 0
    if np.all(input_chunk == fill_value):
        return None

    # Output dimensions for this chunk
    out_z = z_out_end - z_out_start
    out_y = y_out_end - y_out_start
    out_x = x_out_end - x_out_start

    # Downsample: take every other voxel (nearest neighbor for labels)
    # This is much faster than cv2.resize for 3D data
    downsampled = input_chunk[::2, ::2, ::2]

    # Ensure correct output size (handle edge cases)
    if downsampled.shape != (out_z, out_y, out_x):
        result = np.full((out_z, out_y, out_x), fill_value, dtype=input_chunk.dtype)
        sz = min(downsampled.shape[0], out_z)
        sy = min(downsampled.shape[1], out_y)
        sx = min(downsampled.shape[2], out_x)
        result[:sz, :sy, :sx] = downsampled[:sz, :sy, :sx]
        downsampled = result

    return (out_chunk_idx, downsampled)


def generate_pyramid_levels(
    zarr_path: str,
    num_levels: int,
    chunks: Tuple[int, int, int],
    compressor: Any,
    dtype: np.dtype,
    num_workers: int = 1,
) -> None:
    """Generate downsampled pyramid levels from level 0.

    Uses parallel chunk-based processing for fast downsampling.

    Parameters
    ----------
    zarr_path : str
        Path to OME-Zarr root.
    num_levels : int
        Total number of levels (including level 0).
    chunks : Tuple[int, int, int]
        Chunk size for output arrays.
    compressor : Any
        Compressor for zarr arrays.
    dtype : np.dtype
        Data type for arrays.
    num_workers : int
        Number of parallel workers.
    """
    root = zarr.open(zarr_path, mode="r+")

    # Get fill_value from level 0 to propagate to all levels
    level0_fill_value = root["0"].fill_value if root["0"].fill_value is not None else 0

    for level in range(1, num_levels):
        prev_level = level - 1
        prev_arr = root[str(prev_level)]
        prev_shape = prev_arr.shape

        # Calculate new shape (halve each dimension, minimum 1)
        new_shape = tuple(max(1, s // 2) for s in prev_shape)

        print(f"Generating level {level}: {prev_shape} -> {new_shape}")

        # Create array for this level
        level_chunks = tuple(min(c, s) for c, s in zip(chunks, new_shape))
        new_arr = root.create_dataset(
            str(level),
            shape=new_shape,
            chunks=level_chunks,
            dtype=dtype,
            compressor=compressor,
            fill_value=level0_fill_value,
            overwrite=True,
            write_empty_chunks=False,
        )

        # Calculate number of chunks in each dimension
        n_chunks_z = (new_shape[0] + level_chunks[0] - 1) // level_chunks[0]
        n_chunks_y = (new_shape[1] + level_chunks[1] - 1) // level_chunks[1]
        n_chunks_x = (new_shape[2] + level_chunks[2] - 1) // level_chunks[2]

        # Generate all chunk indices
        chunk_indices = [
            (cz, cy, cx)
            for cz in range(n_chunks_z)
            for cy in range(n_chunks_y)
            for cx in range(n_chunks_x)
        ]

        # Prepare worker arguments
        worker_args = [
            (idx, zarr_path, prev_level, prev_shape, new_shape, level_chunks)
            for idx in chunk_indices
        ]

        print(f"  Processing {len(chunk_indices)} chunks with {num_workers} workers...")

        # Process chunks in parallel
        results = []
        if num_workers > 1:
            with Pool(processes=num_workers) as pool:
                results = list(tqdm(
                    pool.imap_unordered(_downsample_chunk_worker, worker_args),
                    total=len(worker_args),
                    desc=f"Level {level}",
                ))
        else:
            for args in tqdm(worker_args, desc=f"Level {level}"):
                results.append(_downsample_chunk_worker(args))

        # Write non-empty results
        written = 0
        for result in results:
            if result is not None:
                (cz, cy, cx), data = result
                z_start = cz * level_chunks[0]
                y_start = cy * level_chunks[1]
                x_start = cx * level_chunks[2]
                new_arr[
                    z_start:z_start + data.shape[0],
                    y_start:y_start + data.shape[1],
                    x_start:x_start + data.shape[2]
                ] = data
                written += 1

        print(f"  Wrote {written} non-empty chunks")


def write_metadata_json(
    output_path: Path,
    label_files: List[Tuple[str, str]],
    args: argparse.Namespace,
) -> None:
    """Write metadata.json with source label information.

    Parameters
    ----------
    output_path : Path
        Path to OME-Zarr root.
    label_files : List[Tuple[str, str]]
        List of (surface_name, label_filename) tuples.
    args : argparse.Namespace
        Command line arguments.
    """
    # Include all command line arguments
    metadata = {
        "created": datetime.now().isoformat(),
        "args": vars(args),
        "sources": [
            {"surface": surf, "label_file": label}
            for surf, label in label_files
        ]
    }

    metadata_path = output_path / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Wrote metadata to: {metadata_path}")


def main() -> None:
    """Main entry point."""
    args = parse_args()
    input_folder = Path(args.input_folder)
    output_path = Path(args.output)

    num_workers = args.num_workers or max(1, multiprocessing.cpu_count() // 2)
    print(f"Using {num_workers} worker processes")
    if HAS_EDT:
        print("Using edt package (multi-threaded) for EDT")
    else:
        print("Using scipy (CPU) for EDT")

    # Discover surfaces
    surfaces = discover_surfaces(input_folder, args.label_suffix)
    print(f"Found {len(surfaces)} surfaces with label files")

    if not surfaces:
        print("No surfaces found, exiting")
        return

    # Check for multi-label mode by looking at 3d_ink_params.json in surfaces
    # If any surface has a "labels" field, we enter multi-label mode
    multi_label_mode = False
    first_ink_labels = None
    for tifxyz_dir, _ in surfaces:
        ink_params = read_3d_ink_params(tifxyz_dir)
        if "labels" in ink_params:
            multi_label_mode = True
            if first_ink_labels is None:
                first_ink_labels = ink_params["labels"]
            break

    if multi_label_mode:
        print("Multi-label mode detected (3d_ink_params.json contains 'labels' field)")
        if not args.remap_json:
            raise ValueError(
                "Multi-label mode requires --remap-json. "
                "Please provide a remap.json file with mappings for all label values."
            )
        if args.label_value is not None:
            print(f"  Warning: --label-value {args.label_value} will be ignored in multi-label mode")

    # Load remap dictionary (required in multi-label mode)
    remap_dict = None
    padded_values = set()
    expand_values: List[int] = []
    fill_value = 0

    if args.remap_json:
        remap_dict, padded_values, expand_values, fill_value = load_and_validate_remap(
            Path(args.remap_json),
            first_ink_labels if multi_label_mode else None,
        )
        print(f"Loaded remap dictionary with {len(remap_dict)} mappings: {remap_dict}")
        if padded_values:
            print(f"Padded label values (2D bounding box): {padded_values}")
        if expand_values:
            print(f"Expand label values (3D EDT, in priority order): {expand_values}")
        print(f"Using fill_value={fill_value} (from 0 -> {fill_value} remap)")
    else:
        # Single-label mode without remap
        if args.label_value is None:
            raise ValueError(
                "Either --label-value (single-label mode) or multi-label mode "
                "(via 3d_ink_params.json labels field + --remap-json) is required."
            )

    # Validate expand_labels is specified when --expand is used in multi-label mode
    if multi_label_mode and args.expand > 0 and not expand_values:
        raise ValueError(
            "--expand requires 'expand_labels' in remap.json specifying which labels to expand. "
            "Example: {\"mappings\": {...}, \"expand_labels\": [\"ink\", \"background\"]}"
        )

    # Create OME-Zarr structure
    z_dim, y_dim, x_dim = args.shape
    chunks = tuple(min(c, d) for c, d in zip(args.chunks, args.shape))
    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    dtype = np.dtype(args.dtype)

    # Create root group
    root = zarr.open_group(str(output_path), mode="w")

    # Create level 0 array (full resolution)
    level0_path = output_path / "0"
    zarr_array = root.create_dataset(
        "0",
        shape=(z_dim, y_dim, x_dim),
        chunks=chunks,
        dtype=dtype,
        compressor=compressor,
        fill_value=fill_value,
        write_empty_chunks=False,
    )
    print(f"Created OME-Zarr level 0: shape={zarr_array.shape}, chunks={zarr_array.chunks}, fill_value={fill_value}")

    # Prepare worker state - point to level 0
    state = {
        "label_value": args.label_value,
        "padding": args.padding,
        "padding_value": args.padding_value,
        "output_path": str(level0_path),
        "volume_shape": (z_dim, y_dim, x_dim),
        "remap_dict": remap_dict,
        "fill_value": fill_value,
        "multi_label_mode": multi_label_mode,
        "padded_values": padded_values,
    }

    # Pre-process surfaces to generate tile tasks
    print("Pre-processing surfaces to find regions...")
    all_tile_tasks: List[TileTask] = []
    for tifxyz_dir, label_file in tqdm(surfaces, desc="Finding regions"):
        # Get ink_params for this surface
        ink_params = read_3d_ink_params(tifxyz_dir)
        surface_ink_labels = ink_params.get("labels", None)

        # In multi-label mode, each surface should have labels
        if multi_label_mode and surface_ink_labels:
            print(f"  {tifxyz_dir.name}: multi-label mode with labels {surface_ink_labels}")

        tasks = preprocess_surface(
            tifxyz_dir,
            label_file,
            args.label_value,
            args.padding,
            args.tile_size,
            multi_label_mode=multi_label_mode,
            ink_labels=surface_ink_labels,
            padded_values=padded_values if multi_label_mode else None,
        )
        all_tile_tasks.extend(tasks)

    print(f"Generated {len(all_tile_tasks)} tile tasks from {len(surfaces)} surfaces")

    if not all_tile_tasks:
        print("No tiles to process, exiting")
        return

    # Phase 1: Collect points from tiles in batches, accumulate by chunk
    # This approach processes tiles in batches to limit peak memory usage
    print(f"Phase 1: Collecting points from tiles in batches of {args.batch_size}...")
    _init_worker(state)

    # Dictionary mapping chunk_idx -> list of (z, y, x, values) arrays
    chunk_points_lists: Dict[Tuple[int, int, int], List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = {}
    batch_size = args.batch_size
    num_batches = (len(all_tile_tasks) + batch_size - 1) // batch_size
    tiles_with_points = 0

    # Set up multiprocessing context once
    mp_context = None
    if num_workers > 1:
        try:
            mp_context = multiprocessing.get_context("fork")
        except ValueError:
            print("Fork not available, using spawn context")
            mp_context = multiprocessing.get_context("spawn")

    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(all_tile_tasks))
        batch = all_tile_tasks[batch_start:batch_end]

        # Collect points from this batch
        if num_workers > 1:
            with ProcessPoolExecutor(
                max_workers=num_workers,
                mp_context=mp_context,
                initializer=_init_worker,
                initargs=(state,),
            ) as executor:
                batch_results = list(tqdm(
                    executor.map(collect_tile_points, batch),
                    total=len(batch),
                    desc=f"Batch {batch_idx + 1}/{num_batches}",
                ))
        else:
            batch_results = []
            for task in tqdm(batch, desc=f"Batch {batch_idx + 1}/{num_batches}"):
                batch_results.append(collect_tile_points(task))

        # Accumulate into chunk_points_lists immediately (frees batch memory)
        for result in batch_results:
            if result is not None:
                tiles_with_points += 1
                accumulate_points_to_chunks(result, chunk_points_lists, chunks)

        # Explicit cleanup to help garbage collector
        del batch_results

    if not chunk_points_lists:
        print("No points collected, exiting")
        return

    # Consolidate: merge lists into single arrays per chunk
    print(f"Consolidating {len(chunk_points_lists)} chunks...")
    chunk_points: Dict[Tuple[int, int, int], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for chunk_idx in tqdm(chunk_points_lists, desc="Consolidating"):
        point_lists = chunk_points_lists[chunk_idx]
        if len(point_lists) > 1:
            chunk_points[chunk_idx] = (
                np.concatenate([p[0] for p in point_lists]),
                np.concatenate([p[1] for p in point_lists]),
                np.concatenate([p[2] for p in point_lists]),
                np.concatenate([p[3] for p in point_lists]),
            )
        else:
            chunk_points[chunk_idx] = point_lists[0]

    # Free intermediate storage
    del chunk_points_lists

    total_points = sum(len(chunk_points[k][0]) for k in chunk_points)
    print(f"Collected {total_points:,} points from {tiles_with_points} tiles across {len(chunk_points)} chunks")

    # Build chunk info list for JSON output
    # Note: values may be remapped, so we check against remapped values if applicable
    if multi_label_mode:
        # In multi-label mode, track all unique values per chunk
        chunks_written = []
        for chunk_idx, (z_coords, y_coords, x_coords, values) in chunk_points.items():
            cz, cy, cx = chunk_idx
            unique_vals = np.unique(values).tolist()
            chunks_written.append({
                "chunk": [int(cz), int(cy), int(cx)],
                "values": unique_vals,
            })
    else:
        # Single-label mode: track has_fg and has_padding
        effective_label = args.label_value
        effective_padding = args.padding_value
        if remap_dict:
            if args.label_value in remap_dict:
                effective_label = remap_dict[args.label_value]
            if args.padding_value is not None and args.padding_value in remap_dict:
                effective_padding = remap_dict[args.padding_value]

        chunks_written = []
        for chunk_idx, (z_coords, y_coords, x_coords, values) in chunk_points.items():
            cz, cy, cx = chunk_idx
            unique_vals = np.unique(values).tolist()
            has_fg = effective_label in unique_vals
            has_padding = effective_padding is not None and effective_padding in unique_vals
            chunks_written.append({
                "chunk": [int(cz), int(cy), int(cx)],
                "has_fg": has_fg,
                "has_padding": has_padding,
                "values": unique_vals,
            })

    # Phase 2: Write chunks using phased approach to avoid race conditions
    print("Phase 2: Writing chunks (8-phase approach)...")

    def build_write_args(chunk_idx: Tuple[int, int, int]):
        z_coords, y_coords, x_coords, values = chunk_points[chunk_idx]
        return (chunk_idx, str(level0_path), z_coords, y_coords, x_coords,
                values, chunks, (z_dim, y_dim, x_dim))

    results = process_chunks_in_phases(
        set(chunk_points.keys()),
        write_chunk_points_worker,
        build_write_args,
        num_workers=num_workers,
        description="Writing",
    )

    total_points = sum(results)
    print(f"Done. Wrote {total_points:,} points to level 0")

    # Post-process: expand labels and/or padding if requested (on level 0)
    if args.expand > 0 or args.padding_expansion > 0:
        if multi_label_mode:
            # In multi-label mode, expand labels specified in expand_labels
            # Use their remapped values for expansion
            if not expand_values:
                print(f"\nWarning: No expand_labels specified in remap.json, skipping expansion.")
            else:
                # Get remapped padding value
                expand_padding_value = None
                if args.padding_value is not None and remap_dict and args.padding_value in remap_dict:
                    expand_padding_value = remap_dict[args.padding_value]

                total_expanded = 0
                for source_val in expand_values:
                    # Get the remapped value for this label
                    expand_label_value = remap_dict.get(source_val, source_val)
                    print(f"\nExpanding label {source_val} (remapped to {expand_label_value}) by {args.expand} voxels...")

                    expanded_count = expand_labels_in_zarr(
                        str(level0_path),
                        args.expand,
                        expand_label_value,
                        padding_value=expand_padding_value,
                        padding_expansion=args.padding_expansion,
                        fill_value=fill_value,
                        num_workers=num_workers,
                        halo=args.halo,
                    )
                    total_expanded += expanded_count
                    print(f"Expansion complete for label {expand_label_value}. Added {expanded_count:,} voxels.")

                print(f"\nTotal expansion: {total_expanded:,} voxels added.")
        else:
            # Use remapped values for expansion if remap was applied
            expand_label_value = args.label_value
            expand_padding_value = args.padding_value
            if remap_dict:
                if args.label_value in remap_dict:
                    expand_label_value = remap_dict[args.label_value]
                    print(f"Using remapped label value {expand_label_value} (from {args.label_value}) for expansion")
                if args.padding_value is not None and args.padding_value in remap_dict:
                    expand_padding_value = remap_dict[args.padding_value]
                    print(f"Using remapped padding value {expand_padding_value} (from {args.padding_value}) for expansion")

            print(f"\nExpanding labels by {args.expand} voxels, padding by {args.padding_expansion} voxels using 3D EDT (halo={args.halo})...")
            expanded_count = expand_labels_in_zarr(
                str(level0_path),
                args.expand,
                expand_label_value,
                padding_value=expand_padding_value,
                padding_expansion=args.padding_expansion,
                fill_value=fill_value,
                num_workers=num_workers,
                halo=args.halo,
            )
            print(f"Expansion complete. Added {expanded_count:,} voxels.")

    # Generate pyramid levels
    if args.num_levels > 1:
        print(f"\nGenerating {args.num_levels - 1} pyramid levels...")
        generate_pyramid_levels(
            str(output_path),
            args.num_levels,
            chunks,
            compressor,
            dtype,
            num_workers=num_workers,
        )

    # Write OME-Zarr metadata directly to .zattrs file
    ome_metadata = create_ome_zarr_metadata(
        (z_dim, y_dim, x_dim),
        args.num_levels,
        args.dtype,
    )
    zattrs_path = output_path / ".zattrs"
    with open(zattrs_path, "w") as f:
        json.dump(ome_metadata, f, indent=2)
    print(f"Wrote OME-Zarr metadata to: {zattrs_path}")

    # Write metadata.json with source label files
    label_files = [(surf.name, label.name) for surf, label in surfaces]
    write_metadata_json(output_path, label_files, args)

    # Write chunks_written.json with chunk info
    if multi_label_mode:
        chunks_info = {
            "multi_label": True,
            "remap": remap_dict,
            "chunk_size": list(chunks),
            "chunks": chunks_written,
        }
    else:
        chunks_info = {
            "label_value": int(effective_label),
            "padding_value": int(effective_padding) if effective_padding is not None else None,
            "chunk_size": list(chunks),
            "chunks": chunks_written,
        }
    chunks_json_path = output_path / "chunks_written.json"
    with open(chunks_json_path, "w") as f:
        json.dump(chunks_info, f, indent=2)
    print(f"Wrote chunk list to: {chunks_json_path}")

    print(f"\nComplete! OME-Zarr written to: {output_path}")


if __name__ == "__main__":
    main()
