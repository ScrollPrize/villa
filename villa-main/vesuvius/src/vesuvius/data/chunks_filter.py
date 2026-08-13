"""
Utility functions for filtering inference to valid chunks based on chunks.json manifest.

chunks.json is generated when creating sparse OME-Zarr volumes and contains the list
of chunk indices that have valid data (e.g., near segmented surfaces).
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np


def load_chunks_json(zarr_path: str) -> Optional[Dict]:
    """
    Load chunks.json from a zarr directory if it exists.

    For OME-Zarr structures where the path points to a resolution level (e.g., /path/to/volume.zarr/0),
    this also checks the parent directory for chunks.json.

    Args:
        zarr_path: Path to the zarr directory (local path, s3://, or http://)

    Returns:
        Parsed JSON dict or None if file doesn't exist or is invalid
    """
    if not zarr_path:
        return None

    zarr_path_clean = zarr_path.rstrip('/')

    # Build list of paths to check for chunks.json
    # Check current path first, then parent (for OME-Zarr level directories like /0, /1, etc.)
    paths_to_check = [zarr_path_clean]

    # If path ends with a numeric directory (OME-Zarr resolution level), also check parent
    base_name = os.path.basename(zarr_path_clean)
    if base_name.isdigit():
        parent_path = os.path.dirname(zarr_path_clean)
        if parent_path:
            paths_to_check.append(parent_path)

    chunks_json_path = None
    for check_path in paths_to_check:
        candidate = os.path.join(check_path, 'chunks.json')
        # For local paths, check existence
        if not zarr_path.startswith(('s3://', 'http://', 'https://')):
            if os.path.exists(candidate):
                chunks_json_path = candidate
                break
        else:
            # For remote, try the first path (current), fallback handled below
            chunks_json_path = candidate
            break

    if chunks_json_path is None:
        return None

    # Handle local paths
    if not zarr_path.startswith(('s3://', 'http://', 'https://')):
        try:
            with open(chunks_json_path, 'r') as f:
                data = json.load(f)
            if 'chunks_by_level' not in data:
                print(f"Warning: chunks.json missing 'chunks_by_level' key, ignoring")
                return None
            return data
        except json.JSONDecodeError as e:
            print(f"Warning: Invalid JSON in chunks.json: {e}")
            return None
        except Exception as e:
            print(f"Warning: Error loading chunks.json: {e}")
            return None

    # Handle S3 paths
    elif zarr_path.startswith('s3://'):
        try:
            import fsspec
            fs = fsspec.filesystem('s3', anon=False)
            if not fs.exists(chunks_json_path):
                return None
            with fs.open(chunks_json_path, 'r') as f:
                data = json.load(f)
            if 'chunks_by_level' not in data:
                print(f"Warning: chunks.json missing 'chunks_by_level' key, ignoring")
                return None
            return data
        except ImportError:
            print("Warning: fsspec not available for S3 access")
            return None
        except Exception as e:
            print(f"Warning: Error loading chunks.json from S3: {e}")
            return None

    # Handle HTTP/HTTPS paths
    else:
        try:
            import requests
            response = requests.get(chunks_json_path, timeout=10)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            data = response.json()
            if 'chunks_by_level' not in data:
                print(f"Warning: chunks.json missing 'chunks_by_level' key, ignoring")
                return None
            return data
        except ImportError:
            print("Warning: requests not available for HTTP access")
            return None
        except Exception as e:
            print(f"Warning: Error loading chunks.json from HTTP: {e}")
            return None


def chunk_indices_to_voxel_bounds(
    chunk_indices: List[List[int]],
    chunk_size: Tuple[int, int, int]
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """
    Convert list of chunk indices to min/max voxel coordinates.

    Args:
        chunk_indices: List of [z, y, x] chunk indices from chunks.json
        chunk_size: (chunk_z, chunk_y, chunk_x) in voxels

    Returns:
        ((min_z, min_y, min_x), (max_z, max_y, max_x)) voxel bounds
        where max values are exclusive (one past the last valid voxel)
    """
    if not chunk_indices:
        raise ValueError("chunk_indices cannot be empty")

    cZ, cY, cX = chunk_size

    # Find min/max chunk indices
    chunk_arr = np.array(chunk_indices)
    min_chunk_z, min_chunk_y, min_chunk_x = chunk_arr.min(axis=0)
    max_chunk_z, max_chunk_y, max_chunk_x = chunk_arr.max(axis=0)

    # Convert to voxel coordinates
    min_z = int(min_chunk_z * cZ)
    min_y = int(min_chunk_y * cY)
    min_x = int(min_chunk_x * cX)

    # Max is exclusive (one past the end of the last chunk)
    max_z = int((max_chunk_z + 1) * cZ)
    max_y = int((max_chunk_y + 1) * cY)
    max_x = int((max_chunk_x + 1) * cX)

    return ((min_z, min_y, min_x), (max_z, max_y, max_x))


def generate_exact_chunk_positions(
    chunk_indices: List[List[int]],
    chunk_size: Tuple[int, int, int],
    patch_size: Tuple[int, int, int],
    volume_shape: Tuple[int, int, int]
) -> List[Tuple[int, int, int]]:
    """
    Generate positions for exact chunk mode (one patch per chunk).

    When patch_size == chunk_size: positions align exactly to chunk boundaries.
    When patch_size > chunk_size: center patch on chunk, clamp to volume bounds.
    When patch_size < chunk_size: center patch on chunk (chunk not fully covered).

    Args:
        chunk_indices: List of [z, y, x] chunk indices
        chunk_size: (chunk_z, chunk_y, chunk_x) zarr chunk size in voxels
        patch_size: (patch_z, patch_y, patch_x) inference patch size in voxels
        volume_shape: (vol_z, vol_y, vol_x) total volume shape

    Returns:
        List of (z, y, x) start positions for patches
    """
    if not chunk_indices:
        return []

    cZ, cY, cX = chunk_size
    pZ, pY, pX = patch_size
    vZ, vY, vX = volume_shape

    positions = []
    seen = set()  # Track unique positions (deduplication after clamping)

    for chunk_idx in chunk_indices:
        cz, cy, cx = chunk_idx  # chunk indices

        # Chunk voxel start
        chunk_start_z = cz * cZ
        chunk_start_y = cy * cY
        chunk_start_x = cx * cX

        # Chunk center
        chunk_center_z = chunk_start_z + cZ // 2
        chunk_center_y = chunk_start_y + cY // 2
        chunk_center_x = chunk_start_x + cX // 2

        # Position patch centered on chunk
        pos_z = chunk_center_z - pZ // 2
        pos_y = chunk_center_y - pY // 2
        pos_x = chunk_center_x - pX // 2

        # Clamp to volume bounds
        pos_z = max(0, min(pos_z, vZ - pZ))
        pos_y = max(0, min(pos_y, vY - pY))
        pos_x = max(0, min(pos_x, vX - pX))

        pos = (pos_z, pos_y, pos_x)
        if pos not in seen:
            seen.add(pos)
            positions.append(pos)

    return positions


def generate_bounded_sliding_window_positions(
    min_bounds: Tuple[int, int, int],
    max_bounds: Tuple[int, int, int],
    patch_size: Tuple[int, int, int],
    step_size: float,
    volume_shape: Tuple[int, int, int]
) -> List[Tuple[int, int, int]]:
    """
    Generate sliding window positions constrained to a bounding box.

    Uses the same algorithm as compute_steps_for_sliding_window but restricted
    to [min_bound, max_bound] instead of [0, volume_shape].

    Args:
        min_bounds: (min_z, min_y, min_x) minimum voxel coordinates (inclusive)
        max_bounds: (max_z, max_y, max_x) maximum voxel coordinates (exclusive)
        patch_size: (patch_z, patch_y, patch_x) inference patch size
        step_size: Step size as fraction of patch size (0-1), e.g., 0.5 = 50% overlap
        volume_shape: (vol_z, vol_y, vol_x) total volume shape for clamping

    Returns:
        List of (z, y, x) start positions for patches
    """
    min_z, min_y, min_x = min_bounds
    max_z, max_y, max_x = max_bounds
    pZ, pY, pX = patch_size
    vZ, vY, vX = volume_shape

    def compute_steps_bounded(region_start, region_end, patch_dim, vol_dim):
        """Compute positions for one dimension within a bounded region."""
        # Clamp region to volume bounds
        region_start = max(0, region_start)
        region_end = min(region_end, vol_dim)

        region_size = region_end - region_start

        # If region is smaller than patch, return one position at region start (clamped)
        if region_size <= patch_dim:
            pos = max(0, min(region_start, vol_dim - patch_dim))
            return [pos]

        # Calculate step size in voxels
        if step_size == 0:
            target_step = patch_dim
        else:
            target_step = max(1, int(patch_dim * step_size))

        # Number of steps needed
        num_steps = int(np.ceil((region_size - patch_dim) / target_step)) + 1

        # Maximum start position within region
        max_start = region_end - patch_dim

        if num_steps == 1:
            return [region_start]

        # Generate uniformly spaced positions
        positions = []
        for i in range(num_steps):
            if num_steps > 1:
                pos = region_start + int(round(i * (max_start - region_start) / (num_steps - 1)))
            else:
                pos = region_start
            # Ensure position is valid
            pos = max(0, min(pos, vol_dim - patch_dim))
            positions.append(pos)

        # Remove duplicates while preserving order
        seen = set()
        unique_positions = []
        for p in positions:
            if p not in seen:
                seen.add(p)
                unique_positions.append(p)

        return unique_positions

    z_positions = compute_steps_bounded(min_z, max_z, pZ, vZ)
    y_positions = compute_steps_bounded(min_y, max_y, pY, vY)
    x_positions = compute_steps_bounded(min_x, max_x, pX, vX)

    # Generate all combinations
    positions = []
    for z in z_positions:
        for y in y_positions:
            for x in x_positions:
                positions.append((z, y, x))

    return positions


def filter_positions_by_chunk_overlap(
    positions: List[Tuple[int, int, int]],
    chunk_indices: List[List[int]],
    chunk_size: Tuple[int, int, int],
    patch_size: Tuple[int, int, int]
) -> List[Tuple[int, int, int]]:
    """
    Filter positions to only those that overlap at least one valid chunk.

    For sparse volumes where valid chunks are scattered, this dramatically reduces
    the number of patches by excluding those that would only cover empty space.

    Args:
        positions: List of (z, y, x) patch start positions
        chunk_indices: List of [z, y, x] valid chunk indices from chunks.json
        chunk_size: (chunk_z, chunk_y, chunk_x) zarr chunk size in voxels
        patch_size: (patch_z, patch_y, patch_x) inference patch size in voxels

    Returns:
        Filtered list of positions that overlap at least one valid chunk
    """
    if not positions or not chunk_indices:
        return positions

    # Build set of valid chunk indices for O(1) lookup
    valid_chunks = set(tuple(idx) for idx in chunk_indices)

    cZ, cY, cX = chunk_size
    pZ, pY, pX = patch_size

    filtered = []
    for pos in positions:
        z, y, x = pos

        # Compute range of chunks this patch overlaps
        # Patch covers voxels [z, z+pZ), [y, y+pY), [x, x+pX)
        chunk_z_min = z // cZ
        chunk_z_max = (z + pZ - 1) // cZ
        chunk_y_min = y // cY
        chunk_y_max = (y + pY - 1) // cY
        chunk_x_min = x // cX
        chunk_x_max = (x + pX - 1) // cX

        # Check if any overlapping chunk is valid
        overlaps_valid = False
        for cz in range(chunk_z_min, chunk_z_max + 1):
            for cy in range(chunk_y_min, chunk_y_max + 1):
                for cx in range(chunk_x_min, chunk_x_max + 1):
                    if (cz, cy, cx) in valid_chunks:
                        overlaps_valid = True
                        break
                if overlaps_valid:
                    break
            if overlaps_valid:
                break

        if overlaps_valid:
            filtered.append(pos)

    return filtered


def compute_touched_chunks(
    positions: List[Tuple[int, int, int]],
    patch_size: Tuple[int, int, int],
    output_chunk_size: Tuple[int, int, int]
) -> List[Tuple[int, int, int]]:
    """
    Compute all output chunks that are touched by any patch position.

    Used by blending to determine which output chunks need processing,
    accounting for patches that extend beyond the original valid chunks.

    Args:
        positions: List of (z, y, x) patch start positions
        patch_size: (patch_z, patch_y, patch_x) inference patch size in voxels
        output_chunk_size: (chunk_z, chunk_y, chunk_x) output zarr chunk size

    Returns:
        Sorted list of unique (z, y, x) chunk indices that need processing
    """
    if not positions:
        return []

    pZ, pY, pX = patch_size
    cZ, cY, cX = output_chunk_size

    touched = set()
    for pos in positions:
        z, y, x = pos

        # Compute range of output chunks this patch touches
        chunk_z_min = z // cZ
        chunk_z_max = (z + pZ - 1) // cZ
        chunk_y_min = y // cY
        chunk_y_max = (y + pY - 1) // cY
        chunk_x_min = x // cX
        chunk_x_max = (x + pX - 1) // cX

        for cz in range(chunk_z_min, chunk_z_max + 1):
            for cy in range(chunk_y_min, chunk_y_max + 1):
                for cx in range(chunk_x_min, chunk_x_max + 1):
                    touched.add((cz, cy, cx))

    # Return sorted for deterministic ordering
    return sorted(touched)
