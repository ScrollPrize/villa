import numpy as np
import os
import re
import json
import time
import zarr
import fsspec
import multiprocessing as mp
from tqdm.auto import tqdm
from scipy.ndimage import gaussian_filter
import torch
from functools import partial
import numcodecs
from concurrent.futures import ProcessPoolExecutor, as_completed
import math
from collections import defaultdict
from vesuvius.data.utils import open_zarr
from vesuvius.data.chunks_filter import load_chunks_json, compute_touched_chunks
import traceback


# --- Worker State (initialized once per process) ---
_worker_state = {}
_shared_state = {}


def _progress_storage_options(path: str) -> dict:
    if path.startswith('s3://'):
        return {'anon': False}
    return {}


def _path_exists(path: str) -> bool:
    storage_options = _progress_storage_options(path)
    fs, fs_path = fsspec.core.url_to_fs(path, **storage_options)
    return fs.exists(fs_path)


def _normalize_run_path(path: str) -> str:
    if path.startswith('s3://'):
        return path.rstrip('/')
    return os.path.abspath(path)


def _read_progress(path: str) -> dict:
    storage_options = _progress_storage_options(path)
    with fsspec.open(path, 'r', **storage_options) as f:
        return json.load(f)


def _write_progress(path: str, data: dict) -> None:
    storage_options = _progress_storage_options(path)

    if path.startswith('s3://'):
        with fsspec.open(path, 'w', **storage_options) as f:
            json.dump(data, f)
        return

    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    tmp_path = f"{path}.tmp"
    with open(tmp_path, 'w') as f:
        json.dump(data, f)
    os.replace(tmp_path, path)


class ProgressTracker:
    def __init__(self, progress_path: str, meta: dict, resume: bool, verbose: bool):
        self.progress_path = progress_path
        self._data = None
        self._completed_set = set()
        self._completed_list = []

        if resume and _path_exists(progress_path):
            self._data = _read_progress(progress_path)
            self._validate(meta)
            self._data.setdefault('completed_chunks', [])
            self._completed_list = self._data['completed_chunks']
            self._completed_set = {tuple(idx) for idx in self._completed_list}
            if verbose:
                print(f"Resuming from progress file: {len(self._completed_set)} chunks already completed")
        else:
            if resume and verbose:
                print(f"Resume requested but no progress file found at {progress_path}; starting fresh.")
            self._data = dict(meta)
            self._data['created_at'] = time.time()
            self._data['completed_chunks'] = []
            self._completed_list = self._data['completed_chunks']
            _write_progress(self.progress_path, self._data)

    def _validate(self, expected: dict) -> None:
        mismatches = []

        if self._data.get('version') != expected.get('version'):
            mismatches.append(
                f"version {self._data.get('version')} != {expected.get('version')}"
            )

        def _normalize_seq(value):
            if isinstance(value, (list, tuple)):
                return list(value)
            return value

        for key in ('patch_size', 'original_volume_shape', 'output_chunk_size'):
            if key not in self._data:
                mismatches.append(f"missing {key}")
                continue
            if _normalize_seq(self._data[key]) != _normalize_seq(expected.get(key)):
                mismatches.append(f"{key} {self._data[key]} != {expected.get(key)}")

        if 'num_classes' in expected:
            if self._data.get('num_classes') != expected.get('num_classes'):
                mismatches.append(
                    f"num_classes {self._data.get('num_classes')} != {expected.get('num_classes')}"
                )

        if 'chunks_filter_mode' in expected:
            if self._data.get('chunks_filter_mode') != expected.get('chunks_filter_mode'):
                mismatches.append(
                    f"chunks_filter_mode {self._data.get('chunks_filter_mode')} != {expected.get('chunks_filter_mode')}"
                )

        for key in ('parent_dir', 'output_path'):
            if key in expected:
                if self._data.get(key) != expected.get(key):
                    mismatches.append(f"{key} {self._data.get(key)} != {expected.get(key)}")

        if 'sigma_scale' in expected:
            actual_sigma = self._data.get('sigma_scale')
            expected_sigma = expected.get('sigma_scale')
            if actual_sigma is None or not math.isclose(
                float(actual_sigma), float(expected_sigma), rel_tol=1e-6, abs_tol=1e-6
            ):
                mismatches.append(
                    f"sigma_scale {actual_sigma} != {expected_sigma}"
                )

        if mismatches:
            details = "\n  ".join(mismatches)
            raise ValueError(
                f"Progress file does not match current run:\n  {details}\n"
                f"Delete {self.progress_path} or rerun without --resume."
            )

    def is_done(self, chunk_idx: tuple) -> bool:
        return tuple(chunk_idx) in self._completed_set

    def mark_done(self, chunk_idx: tuple) -> None:
        chunk_idx = tuple(chunk_idx)
        if chunk_idx in self._completed_set:
            return
        self._completed_set.add(chunk_idx)
        self._completed_list.append([int(v) for v in chunk_idx])
        self._data['updated_at'] = time.time()
        _write_progress(self.progress_path, self._data)

    def set_totals(self, total_chunks: int, skipped_chunks: int = 0) -> None:
        self._data['total_chunks'] = int(total_chunks)
        self._data['skipped_chunks'] = int(skipped_chunks)
        self._data['updated_at'] = time.time()
        _write_progress(self.progress_path, self._data)

    def finalize(self) -> None:
        self._data['completed'] = True
        self._data['completed_at'] = time.time()
        _write_progress(self.progress_path, self._data)


def init_worker(part_files, output_path, is_s3, shared_state=None):
    """
    Initialize worker process with cached zarr store handles.
    Called once per worker process at pool creation.

    This avoids repeatedly opening zarr stores for each chunk,
    providing significant speedup especially for many chunks.
    """
    global _worker_state

    storage_opts = {'anon': False} if is_s3 else None

    _worker_state['output'] = open_zarr(output_path, mode='r+', storage_options=storage_opts)
    _worker_state['logits'] = {}

    state = shared_state if shared_state is not None else _shared_state
    if state:
        _worker_state['chunk_to_patches_index'] = state.get('chunk_to_patches_index')
        _worker_state['gaussian_map'] = state.get('gaussian_map')
        _worker_state['patch_size'] = state.get('patch_size')
    else:
        _worker_state['chunk_to_patches_index'] = None
        _worker_state['gaussian_map'] = None
        _worker_state['patch_size'] = None

    for part_id, paths in part_files.items():
        logits_path = paths['logits']
        _worker_state['logits'][part_id] = open_zarr(
            logits_path, mode='r',
            storage_options={'anon': False} if logits_path.startswith('s3://') else None
        )


def build_chunk_to_patches_index(all_coords_by_part, patch_size, chunk_size, volume_shape):
    """
    Pre-compute which patches touch which chunks.

    This changes complexity from O(chunks × total_patches) to O(total_patches),
    which is a dramatic improvement for large numbers of patches and chunks.

    Args:
        all_coords_by_part: Dict mapping part_id -> numpy array of shape (N, 3) with (z, y, x) coords
        patch_size: (pZ, pY, pX) patch dimensions
        chunk_size: (cZ, cY, cX) processing chunk dimensions
        volume_shape: (Z, Y, X) total volume shape

    Returns:
        Dict mapping (cz, cy, cx) chunk index -> list of (part_id, patch_idx, (z, y, x))
    """
    pZ, pY, pX = patch_size
    cZ, cY, cX = chunk_size

    chunk_to_patches = defaultdict(list)

    for part_id, coords in all_coords_by_part.items():
        for patch_idx in range(coords.shape[0]):
            z, y, x = coords[patch_idx].tolist()

            # Find all chunks this patch intersects
            # Chunk indices for the start and end of the patch
            cz_start = z // cZ
            cz_end = (z + pZ - 1) // cZ
            cy_start = y // cY
            cy_end = (y + pY - 1) // cY
            cx_start = x // cX
            cx_end = (x + pX - 1) // cX

            # Add this patch to all chunks it touches
            for cz in range(cz_start, cz_end + 1):
                for cy in range(cy_start, cy_end + 1):
                    for cx in range(cx_start, cx_end + 1):
                        chunk_to_patches[(cz, cy, cx)].append((part_id, patch_idx, (z, y, x)))

    return chunk_to_patches


def generate_gaussian_map(patch_size: tuple, sigma_scale: float = 8.0, dtype=np.float32) -> np.ndarray:
    pZ, pY, pX = patch_size
    tmp = np.zeros(patch_size, dtype=dtype)
    center_coords = [i // 2 for i in patch_size]
    sigmas = [i / sigma_scale for i in patch_size]

    tmp[tuple(center_coords)] = 1

    gaussian_map_np = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)

    gaussian_map_np /= max(gaussian_map_np.max(), 1e-12)
    gaussian_map_np = gaussian_map_np.reshape(1, pZ, pY, pX)
    gaussian_map_np = np.clip(gaussian_map_np, a_min=0, a_max=None)
    
    print(
        f"Generated Gaussian map with shape {gaussian_map_np.shape}, min: {gaussian_map_np.min():.4f}, max: {gaussian_map_np.max():.4f}")
    return gaussian_map_np


def process_chunk_optimized(chunk_info, gaussian_map=None, patch_size=None, chunk_patches=None, epsilon=1e-8):
    """
    Process a single chunk using pre-computed patch list and cached stores.

    This optimized version:
    1. Uses worker-cached zarr stores (no repeated opens)
    2. Only processes patches known to intersect this chunk (pre-filtered)
    3. Fuses normalization into the same pass (saves a second I/O pass)

    Args:
        chunk_info: Dict with chunk boundaries and indices
        gaussian_map: Pre-computed Gaussian map (1, pZ, pY, pX)
        patch_size: (pZ, pY, pX)
        chunk_patches: List of (part_id, patch_idx, (z, y, x)) for patches intersecting this chunk
        epsilon: Small value to avoid division by zero during normalization
    """
    global _worker_state

    if gaussian_map is None:
        gaussian_map = _worker_state.get('gaussian_map')
    if patch_size is None:
        patch_size = _worker_state.get('patch_size')
    if gaussian_map is None or patch_size is None:
        raise ValueError("gaussian_map and patch_size must be provided or set in worker state.")

    if chunk_patches is None:
        chunk_idx = chunk_info.get('indices')
        if chunk_idx is None:
            raise ValueError("chunk_info missing indices; cannot lookup patches.")
        chunk_index = _worker_state.get('chunk_to_patches_index')
        if chunk_index is None:
            raise ValueError("chunk_to_patches_index not available in worker state.")
        chunk_patches = chunk_index.get(chunk_idx, [])

    # Extract chunk boundaries
    z_start, z_end = chunk_info['z_start'], chunk_info['z_end']
    y_start, y_end = chunk_info['y_start'], chunk_info['y_end']
    x_start, x_end = chunk_info['x_start'], chunk_info['x_end']

    pZ, pY, pX = patch_size
    gaussian_map_spatial = gaussian_map[0]  # Shape (pZ, pY, pX)

    # Use cached store from worker initialization
    output_store = _worker_state['output']

    # Create local accumulators
    num_classes = output_store.shape[0]
    chunk_shape = (num_classes, z_end - z_start, y_end - y_start, x_end - x_start)
    weights_shape = (z_end - z_start, y_end - y_start, x_end - x_start)

    chunk_logits = np.zeros(chunk_shape, dtype=np.float32)
    chunk_weights = np.zeros(weights_shape, dtype=np.float32)
    patches_processed = 0

    # Group patches by part_id for better cache locality
    patches_by_part = defaultdict(list)
    for part_id, patch_idx, coords in chunk_patches:
        patches_by_part[part_id].append((patch_idx, coords))

    for part_id, patch_list in patches_by_part.items():
        logits_store = _worker_state['logits'][part_id]

        for patch_idx, (z, y, x) in patch_list:
            # Calculate intersection between patch and chunk (in chunk-local coords)
            iz_start = max(z, z_start) - z_start
            iz_end = min(z + pZ, z_end) - z_start
            iy_start = max(y, y_start) - y_start
            iy_end = min(y + pY, y_end) - y_start
            ix_start = max(x, x_start) - x_start
            ix_end = min(x + pX, x_end) - x_start

            # Calculate which part of patch to use (in patch-local coords)
            pz_start = max(z_start - z, 0)
            pz_end = pZ - max(z + pZ - z_end, 0)
            py_start = max(y_start - y, 0)
            py_end = pY - max(y + pY - y_end, 0)
            px_start = max(x_start - x, 0)
            px_end = pX - max(x + pX - x_end, 0)

            patch_slice = (
                slice(None),  # All classes
                slice(pz_start, pz_end),
                slice(py_start, py_end),
                slice(px_start, px_end)
            )

            logit_patch = logits_store[patch_idx][patch_slice]

            # Skip empty patches
            if not np.any(logit_patch != 0):
                continue

            weight_patch = gaussian_map_spatial[pz_start:pz_end, py_start:py_end, px_start:px_end]

            # Accumulate weighted logits
            chunk_logits[:, iz_start:iz_end, iy_start:iy_end, ix_start:ix_end] += \
                logit_patch * weight_patch[np.newaxis, :, :, :]

            chunk_weights[iz_start:iz_end, iy_start:iy_end, ix_start:ix_end] += weight_patch

            patches_processed += 1

    if patches_processed > 0:
        # Fused normalization: normalize in-place before writing
        mask = chunk_weights > epsilon
        if np.any(mask):
            # Broadcast division across classes
            chunk_logits[:, mask] /= chunk_weights[mask]

        output_slice = (
            slice(None),
            slice(z_start, z_end),
            slice(y_start, y_end),
            slice(x_start, x_end)
        )

        output_store[output_slice] = chunk_logits

    return {
        'chunk': chunk_info,
        'patches_processed': patches_processed
    }


# Legacy function for backwards compatibility (if needed)
def process_chunk(chunk_info, parent_dir, output_path, gaussian_map,
                patch_size, part_files, epsilon=1e-8):
    """
    Process a single chunk of the volume, handling all patches that intersect with this chunk.

    DEPRECATED: This is the legacy version. Use process_chunk_optimized for better performance.
    It still performs Gaussian blending and normalization in one pass.

    Args:
        chunk_info: Dictionary with chunk boundaries {'z_start', 'z_end', 'y_start', 'y_end', 'x_start', 'x_end'}
        parent_dir: Directory containing part files
        output_path: Path to output zarr
        gaussian_map: Pre-computed Gaussian map
        patch_size: Size of patches (pZ, pY, pX)
        part_files: Dictionary of part files
        epsilon: Small value to avoid division by zero during normalization
    """

    # Extract chunk boundaries
    z_start, z_end = chunk_info['z_start'], chunk_info['z_end']
    y_start, y_end = chunk_info['y_start'], chunk_info['y_end']
    x_start, x_end = chunk_info['x_start'], chunk_info['x_end']

    pZ, pY, pX = patch_size

    gaussian_map_spatial_np = gaussian_map[0]  # Shape (pZ, pY, pX)

    output_store = open_zarr(output_path, mode='r+', storage_options={'anon': False} if output_path.startswith('s3://') else None)

    # Create local accumulators for this chunk - initialize with zeros
    # Shape: (C, chunk_z, chunk_y, chunk_x)
    num_classes = output_store.shape[0]
    chunk_shape = (num_classes, z_end - z_start, y_end - y_start, x_end - x_start)
    weights_shape = (z_end - z_start, y_end - y_start, x_end - x_start)

    chunk_logits = np.zeros(chunk_shape, dtype=np.float32)
    chunk_weights = np.zeros(weights_shape, dtype=np.float32)
    patches_processed = 0

    for part_id in part_files:
        logits_path = part_files[part_id]['logits']
        coords_path = part_files[part_id]['coordinates']

        coords_store = open_zarr(coords_path, mode='r', storage_options={'anon': False} if coords_path.startswith('s3://') else None)
        logits_store = open_zarr(logits_path, mode='r', storage_options={'anon': False} if logits_path.startswith('s3://') else None)

        coords_np = coords_store[:]
        num_patches_in_part = coords_np.shape[0]

        for patch_idx in range(num_patches_in_part):
            z, y, x = coords_np[patch_idx].tolist()

            if (z + pZ <= z_start or z >= z_end or
                y + pY <= y_start or y >= y_end or
                x + pX <= x_start or x >= x_end):
                continue  # Skip patches that don't intersect with this chunk

            iz_start = max(z, z_start) - z_start
            iz_end = min(z + pZ, z_end) - z_start
            iy_start = max(y, y_start) - y_start
            iy_end = min(y + pY, y_end) - y_start
            ix_start = max(x, x_start) - x_start
            ix_end = min(x + pX, x_end) - x_start

            pz_start = max(z_start - z, 0)
            pz_end = pZ - max(z + pZ - z_end, 0)
            py_start = max(y_start - y, 0)
            py_end = pY - max(y + pY - y_end, 0)
            px_start = max(x_start - x, 0)
            px_end = pX - max(x + pX - x_end, 0)

            patch_slice = (
                slice(None),  # All classes
                slice(pz_start, pz_end),
                slice(py_start, py_end),
                slice(px_start, px_end)
            )

            logit_patch = logits_store[patch_idx][patch_slice]

            # Skip patches with no values - don't let empty patches contribute to weights
            if not np.any(logit_patch != 0):
                continue

            weight_patch = gaussian_map_spatial_np[
                slice(pz_start, pz_end),
                slice(py_start, py_end),
                slice(px_start, px_end)
            ]

            # Apply weights to logits (broadcasting along class dimension)
            weighted_patch = logit_patch * weight_patch[np.newaxis, :, :, :]

            # Accumulate into local arrays
            chunk_logits[
                :,  # All classes
                iz_start:iz_end,
                iy_start:iy_end,
                ix_start:ix_end
            ] += weighted_patch

            chunk_weights[
                iz_start:iz_end,
                iy_start:iy_end,
                ix_start:ix_end
            ] += weight_patch

            patches_processed += 1

    if patches_processed > 0:
        mask = chunk_weights > epsilon
        if np.any(mask):
            chunk_logits[:, mask] /= chunk_weights[mask]

        output_slice = (
            slice(None),
            slice(z_start, z_end),
            slice(y_start, y_end),
            slice(x_start, x_end)
        )

        output_store[output_slice] = chunk_logits

    return {
        'chunk': chunk_info,
        'patches_processed': patches_processed
    }


# --- Utility Functions ---
def calculate_chunks(volume_shape, output_chunks=None, valid_chunk_indices=None, zarr_chunk_size=None):
    """
    Calculate processing chunks for the volume.

    Args:
        volume_shape: (Z, Y, X) shape of the volume
        output_chunks: (z, y, x) chunk size for processing. Defaults to (256, 256, 256).
        valid_chunk_indices: Optional list of [z, y, x] chunk indices from chunks.json.
                            If provided, only chunks overlapping with these will be returned.
        zarr_chunk_size: (z, y, x) zarr chunk size. Required if valid_chunk_indices is provided.

    Returns:
        List of chunk info dicts with z_start, z_end, y_start, y_end, x_start, x_end
    """
    Z, Y, X = volume_shape

    if output_chunks is None:
        z_chunk, y_chunk, x_chunk = 256, 256, 256
    else:
        z_chunk, y_chunk, x_chunk = output_chunks

    # Build set of valid voxel regions if chunks.json filtering is enabled
    valid_regions = None
    if valid_chunk_indices is not None and zarr_chunk_size is not None:
        cZ, cY, cX = zarr_chunk_size
        valid_regions = set()
        for chunk_idx in valid_chunk_indices:
            ci_z, ci_y, ci_x = chunk_idx
            # Store the chunk index tuple for fast lookup
            valid_regions.add((ci_z, ci_y, ci_x))

    chunks = []
    for z_start in range(0, Z, z_chunk):
        for y_start in range(0, Y, y_chunk):
            for x_start in range(0, X, x_chunk):
                z_end = min(z_start + z_chunk, Z)
                y_end = min(y_start + y_chunk, Y)
                x_end = min(x_start + x_chunk, X)

                # If filtering by chunks.json, check if this chunk overlaps with any valid region
                if valid_regions is not None:
                    cZ, cY, cX = zarr_chunk_size
                    # Find which zarr chunks this processing chunk overlaps with
                    ci_z_start = z_start // cZ
                    ci_z_end = (z_end - 1) // cZ + 1
                    ci_y_start = y_start // cY
                    ci_y_end = (y_end - 1) // cY + 1
                    ci_x_start = x_start // cX
                    ci_x_end = (x_end - 1) // cX + 1

                    # Check if any overlapping zarr chunk is in the valid set
                    overlaps_valid = False
                    for ci_z in range(ci_z_start, ci_z_end):
                        for ci_y in range(ci_y_start, ci_y_end):
                            for ci_x in range(ci_x_start, ci_x_end):
                                if (ci_z, ci_y, ci_x) in valid_regions:
                                    overlaps_valid = True
                                    break
                            if overlaps_valid:
                                break
                        if overlaps_valid:
                            break

                    if not overlaps_valid:
                        continue  # Skip this chunk

                chunks.append({
                    'z_start': z_start, 'z_end': z_end,
                    'y_start': y_start, 'y_end': y_end,
                    'x_start': x_start, 'x_end': x_end
                })

    return chunks

# --- Main Merging Function ---
def merge_inference_outputs(
        parent_dir: str,
        output_path: str,
        sigma_scale: float = 8.0,
        chunk_size: tuple = None,  # Spatial chunk size (Z, Y, X) for output
        num_workers: int = None,  # Number of worker processes to use
        compression_level: int = 1,  # Compression level (0-9, 0=none)
        verbose: bool = True,
        input_zarr_path: str = None,  # Optional: Path to input zarr for chunks.json detection
        chunks_filter_mode: str = 'auto',  # 'auto', 'disabled'
        resume: bool = False,  # Resume from a previous run using progress tracking
        progress_path: str = None,  # Optional path for progress tracking file
        ):
    """
    Args:
        parent_dir: Directory containing logits_part_X.zarr and coordinates_part_X.zarr.
        output_path: Path for the final merged Zarr store.
        sigma_scale: Determines the sigma for the Gaussian map (patch_size / sigma_scale).
        chunk_size: Spatial chunk size (Z, Y, X) for output Zarr stores.
                    If None, will use patch_size as a starting point.
        num_workers: Number of worker processes to use.
                     If None, defaults to CPU_COUNT - 1.
        compression_level: Zarr compression level (0-9, 0=none)
        verbose: Print progress messages.
        input_zarr_path: Path to original input zarr for chunks.json detection.
        chunks_filter_mode: 'auto' (use chunks.json if present) or 'disabled'.
        resume: Resume blending from a previous run using progress tracking.
        progress_path: Optional path to a progress JSON file (defaults to output_path + ".progress.json").
    """

    # --- Safety check: prevent overwriting input logits ---
    # Normalize paths for comparison
    parent_dir_abs = os.path.abspath(parent_dir.rstrip('/'))
    output_path_abs = os.path.abspath(output_path.rstrip('/'))

    # Check if output would overwrite or be inside the input directory
    if output_path_abs == parent_dir_abs:
        raise ValueError(
            f"Output path cannot be the same as input directory!\n"
            f"  Input (parent_dir): {parent_dir}\n"
            f"  Output (output_path): {output_path}\n"
            f"This would delete your logits files. Use a different output path."
        )
    if output_path_abs.startswith(parent_dir_abs + os.sep):
        raise ValueError(
            f"Output path cannot be inside the input directory!\n"
            f"  Input (parent_dir): {parent_dir}\n"
            f"  Output (output_path): {output_path}\n"
            f"This could corrupt your logits files. Use a different output path."
        )
    if parent_dir_abs.startswith(output_path_abs + os.sep):
        raise ValueError(
            f"Input directory cannot be inside the output path!\n"
            f"  Input (parent_dir): {parent_dir}\n"
            f"  Output (output_path): {output_path}\n"
            f"This would delete your logits files. Use a different output path."
        )

    output_path = output_path.rstrip('/')

    # blosc has an issue with threading, so we disable it
    numcodecs.blosc.use_threads = False
    
    if num_workers is None:
        # just use half the cpu count 
        num_workers = max(1, mp.cpu_count() // 2)
    
    print(f"Using {num_workers} worker processes (half of CPU count for memory efficiency)")
        
    # --- 1. Discover Parts ---
    part_files = {}
    part_pattern = re.compile(r"(logits|coordinates)_part_(\d+)\.zarr")
    print(f"Scanning for parts in: {parent_dir}")
    
    # we need to use fsspec to work w/ s3 paths , as os.listdir doesn't work with s3
    if parent_dir.startswith('s3://'):
        fs = fsspec.filesystem('s3', anon=False)
        # Remove 's3://' prefix for fs.ls()
        parent_dir_no_prefix = parent_dir.replace('s3://', '')
        # List directory to get all entries
        full_paths = fs.ls(parent_dir_no_prefix)
        
        # For S3, strip the bucket name and path prefix to get just the directory name
        # Each entry looks like: 'bucket/path/to/parent_dir/logits_part_0.zarr'
        file_list = []
        for path in full_paths:
            path_parts = path.split('/')
            filename = path_parts[-1]
            file_list.append(filename)
            
        print(f"DEBUG: Found files in S3: {file_list}")
    else:
        file_list = os.listdir(parent_dir)
        
    for filename in file_list:
        match = part_pattern.match(filename)
        if match:
            file_type, part_id_str = match.groups()
            part_id = int(part_id_str)
            if part_id not in part_files:
                part_files[part_id] = {}
            part_files[part_id][file_type] = os.path.join(parent_dir, filename)

    part_ids = sorted(part_files.keys())
    if not part_ids:
        raise FileNotFoundError(f"No inference parts found in {parent_dir}")
    print(f"Found parts: {part_ids}")

    for part_id in part_ids:
        if 'logits' not in part_files[part_id] or 'coordinates' not in part_files[part_id]:
            raise FileNotFoundError(f"Part {part_id} is missing logits or coordinates Zarr.")

    # --- 2. Read Metadata (from first available part) ---
    first_part_id = part_ids[0]  
    print(f"Reading metadata from part {first_part_id}...")
    part0_logits_path = part_files[first_part_id]['logits']
    try:
        part0_logits_store = open_zarr(part0_logits_path, mode='r', storage_options={'anon': False} if part0_logits_path.startswith('s3://') else None)

        input_chunks = part0_logits_store.chunks
        print(f"Input zarr chunk size: {input_chunks}")

        try:
            # Use the part0_logits_store's .attrs directly if available
            meta_attrs = part0_logits_store.attrs
            patch_size = tuple(meta_attrs['patch_size']) 
            original_volume_shape = tuple(meta_attrs['original_volume_shape'])  # MUST exist
            num_classes = part0_logits_store.shape[1]  # (N, C, pZ, pY, pX) -> C
        except (KeyError, AttributeError):
            # Fallback: try to read .zattrs file directly
            zattrs_path = os.path.join(part0_logits_path, '.zattrs')
            with fsspec.open(zattrs_path, 'r') as f:
                meta_attrs = json.load(f)
                
            patch_size = tuple(meta_attrs['patch_size'])  
            original_volume_shape = tuple(meta_attrs['original_volume_shape'])
            num_classes = part0_logits_store.shape[1]

    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"Failed to read metadata from {part0_logits_path}: {e}")
        
    print(f"  Patch Size: {patch_size}")
    print(f"  Num Classes: {num_classes}")
    print(f"  Original Volume Shape (Z,Y,X): {original_volume_shape}")

    # --- 3. Prepare Output Stores ---
    output_shape = (num_classes, *original_volume_shape)  # (C, D, H, W)

    # we use the patch size as the default chunk size throughout the pipeline
    # so that the chunk size is consistent , to avoid partial chunk read/writes
    # given that we write the logits with aligned chunk/patch size, we continue that here
    if chunk_size is None or any(c == 0 for c in (chunk_size if chunk_size else [0, 0, 0])):

        output_chunks = (
            1,  
            patch_size[0],  # z 
            patch_size[1],  # y
            patch_size[2]   # x
        )
        if verbose:
            print(f"  Using chunk_size {output_chunks[1:]} based directly on patch_size")
    else:
        output_chunks = (1, *chunk_size) 
        if verbose:
            print(f"  Using specified chunk_size {chunk_size}")

    
    if compression_level > 0:
        compressor = numcodecs.Blosc(
            cname='zstd',
            clevel=compression_level,
            shuffle=numcodecs.blosc.SHUFFLE
        )
    else:
        compressor = None

    progress_path = progress_path or f"{output_path}.progress.json"
    progress_exists = _path_exists(progress_path)
    output_exists = _path_exists(output_path)

    if resume and progress_exists and not output_exists:
        raise ValueError(
            f"Resume requested but output store does not exist: {output_path}\n"
            f"Delete {progress_path} or rerun without --resume."
        )
    if resume and output_exists and not progress_exists and verbose:
        print(
            f"Resume requested but no progress file found at {progress_path}; "
            "reprocessing all chunks with patches."
        )

    output_storage_opts = {'anon': False} if output_path.startswith('s3://') else None
    if resume and output_exists:
        print(f"Using existing output store: {output_path}")
        print(f"  Shape: {output_shape}, Chunks: {output_chunks}")
        output_store = open_zarr(
            path=output_path,
            mode='r+',
            storage_options=output_storage_opts,
            verbose=verbose
        )
        if output_store.shape != output_shape:
            raise ValueError(
                f"Existing output store shape {output_store.shape} does not match expected {output_shape}."
            )
        if output_store.chunks != output_chunks:
            raise ValueError(
                f"Existing output store chunks {output_store.chunks} do not match expected {output_chunks}."
            )
    else:
        print(f"Creating final output store: {output_path}")
        print(f"  Shape: {output_shape}, Chunks: {output_chunks}")
        open_zarr(
            path=output_path,
            mode='w',
            storage_options=output_storage_opts,
            verbose=verbose,
            shape=output_shape,
            chunks=output_chunks,
            compressor=compressor,
            dtype=np.float32,
            fill_value=0,
            write_empty_chunks=False 
        )

    progress_meta = {
        'version': 1,
        'parent_dir': _normalize_run_path(parent_dir),
        'output_path': _normalize_run_path(output_path),
        'patch_size': list(patch_size),
        'original_volume_shape': list(original_volume_shape),
        'output_chunk_size': list(output_chunks[1:]),
        'num_classes': int(num_classes),
        'sigma_scale': float(sigma_scale),
        'chunks_filter_mode': chunks_filter_mode,
    }
    progress_tracker = ProgressTracker(
        progress_path=progress_path,
        meta=progress_meta,
        resume=resume,
        verbose=verbose
    )

    # --- 4. Generate Gaussian Map ---
    gaussian_map = generate_gaussian_map(patch_size, sigma_scale=sigma_scale)

    # --- 4.5. Read coordinates and build spatial index ---
    # Read all coordinates from all parts and build chunk-to-patches mapping
    # This is the key optimization: O(patches) instead of O(chunks × patches)
    print("\nReading coordinates and building spatial index...")
    all_coords_by_part = {}
    total_patches = 0

    for part_id in part_ids:
        coords_path = part_files[part_id]['coordinates']
        coords_store = open_zarr(
            coords_path, mode='r',
            storage_options={'anon': False} if coords_path.startswith('s3://') else None
        )
        coords_np = coords_store[:]
        all_coords_by_part[part_id] = coords_np
        total_patches += coords_np.shape[0]

    if verbose:
        print(f"  Read {total_patches} patch positions from {len(part_ids)} parts")

    # Build the chunk-to-patches spatial index
    spatial_chunk_size = output_chunks[1:]  # (cZ, cY, cX)
    chunk_to_patches_index = build_chunk_to_patches_index(
        all_coords_by_part=all_coords_by_part,
        patch_size=patch_size,
        chunk_size=spatial_chunk_size,
        volume_shape=original_volume_shape
    )

    if verbose:
        print(f"  Built spatial index: {len(chunk_to_patches_index)} chunks have patches")

    start_method = mp.get_start_method(allow_none=True)
    if start_method is None:
        start_method = mp.get_context().get_start_method()
    use_shared_index = start_method == 'fork'
    if verbose:
        print(
            f"  Process start method: {start_method} "
            f"(shared index {'enabled' if use_shared_index else 'disabled'})"
        )

    global _shared_state
    if use_shared_index:
        _shared_state = {
            'chunk_to_patches_index': chunk_to_patches_index,
            'gaussian_map': gaussian_map,
            'patch_size': patch_size,
        }
        init_shared_state = None
    else:
        _shared_state = {}
        init_shared_state = {
            'gaussian_map': gaussian_map,
            'patch_size': patch_size,
        }

    # Determine touched chunks (for filtering)
    touched_chunk_indices = None
    if chunks_filter_mode != 'disabled':
        # Extract positions for compute_touched_chunks (used for metadata)
        all_positions = []
        for coords_np in all_coords_by_part.values():
            for i in range(coords_np.shape[0]):
                all_positions.append(tuple(coords_np[i].tolist()))

        if all_positions:
            touched_chunk_indices = compute_touched_chunks(
                positions=all_positions,
                patch_size=patch_size,
                output_chunk_size=spatial_chunk_size
            )
            if verbose:
                print(f"  Patches touch {len(touched_chunk_indices)} output chunks")

    # --- 5. Calculate Processing Chunks ---
    if touched_chunk_indices is not None:
        # Convert touched chunk indices to chunk_info format expected by calculate_chunks
        spatial_chunks = output_chunks[1:]  # (cZ, cY, cX)
        chunks = []
        for (cz, cy, cx) in touched_chunk_indices:
            z_start = cz * spatial_chunks[0]
            z_end = min((cz + 1) * spatial_chunks[0], original_volume_shape[0])
            y_start = cy * spatial_chunks[1]
            y_end = min((cy + 1) * spatial_chunks[1], original_volume_shape[1])
            x_start = cx * spatial_chunks[2]
            x_end = min((cx + 1) * spatial_chunks[2], original_volume_shape[2])
            chunks.append({
                'z_start': z_start, 'z_end': z_end,
                'y_start': y_start, 'y_end': y_end,
                'x_start': x_start, 'x_end': x_end,
                'indices': (cz, cy, cx)
            })
        full_chunks = calculate_chunks(original_volume_shape, output_chunks=output_chunks[1:])
        print(f"Filtered to {len(chunks)} chunks (from {len(full_chunks)} total) based on patch coordinates")
    else:
        chunks = calculate_chunks(
            original_volume_shape,
            output_chunks=output_chunks[1:]
        )
        print(f"Divided volume into {len(chunks)} chunks for parallel processing")
    
    # --- 6. Process Chunks in Parallel (with fused normalization) ---
    print("\n--- Processing and Normalizing Patches ---")

    # Determine if S3 is being used
    is_s3 = output_path.startswith('s3://')

    # Prepare chunk tasks with their pre-filtered patch lists
    chunk_tasks = []
    skipped_chunks = 0
    empty_chunks = 0
    for chunk in chunks:
        # Get chunk indices
        if 'indices' in chunk:
            chunk_idx = chunk['indices']
        else:
            # Calculate indices from start positions
            chunk_idx = (
                chunk['z_start'] // spatial_chunk_size[0],
                chunk['y_start'] // spatial_chunk_size[1],
                chunk['x_start'] // spatial_chunk_size[2]
            )
            chunk['indices'] = chunk_idx

        # Get pre-filtered patches for this chunk from the spatial index
        chunk_patches = chunk_to_patches_index.get(chunk_idx, [])

        # Only process chunks that have patches
        if not chunk_patches:
            empty_chunks += 1
            continue

        if progress_tracker.is_done(chunk_idx):
            skipped_chunks += 1
            continue

        if use_shared_index:
            chunk_tasks.append(chunk)
        else:
            chunk_tasks.append((chunk, chunk_patches))

    if verbose:
        print(
            f"  Processing {len(chunk_tasks)} chunks with patches "
            f"(skipping {empty_chunks} empty chunks, "
            f"{skipped_chunks} already completed)"
        )

    progress_tracker.set_totals(len(chunk_tasks) + skipped_chunks, skipped_chunks)

    total_patches_processed = 0

    # Use worker initializer to cache zarr stores per worker process
    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=init_worker,
        initargs=(part_files, output_path, is_s3, init_shared_state)
    ) as executor:
        # Submit all chunk tasks
        future_to_chunk = {}
        for item in chunk_tasks:
            if use_shared_index:
                chunk = item
                future = executor.submit(process_chunk_optimized, chunk)
            else:
                chunk, chunk_patches = item
                future = executor.submit(process_chunk_optimized, chunk, None, None, chunk_patches)
            future_to_chunk[future] = chunk

        for future in tqdm(
            as_completed(future_to_chunk),
            total=len(chunk_tasks),
            desc="Processing Chunks",
            disable=not verbose
        ):
            try:
                result = future.result()
                total_patches_processed += result['patches_processed']
                chunk_info = future_to_chunk[future]
                progress_tracker.mark_done(chunk_info['indices'])
            except Exception as e:
                print(f"Error processing chunk: {e}")
                traceback.print_exc()
                raise e

    print(f"\nProcessing complete. Processed {total_patches_processed} patch contributions.")

    # --- 7. Save Metadata ---
    output_zarr = open_zarr(
        path=output_path,
        mode='r+',
        storage_options={'anon': False} if output_path.startswith('s3://') else None,
        verbose=verbose
    )
    if hasattr(output_zarr, 'attrs'):
        # Copy all attributes from the input part
        if hasattr(part0_logits_store, 'attrs'):
            for key, value in part0_logits_store.attrs.items():
                output_zarr.attrs[key] = value
        # Update/add specific attributes
        output_zarr.attrs['patch_size'] = patch_size
        output_zarr.attrs['original_volume_shape'] = original_volume_shape
        output_zarr.attrs['sigma_scale'] = sigma_scale
        if 'processing_mode' not in output_zarr.attrs and meta_attrs:
            pmode = meta_attrs.get('processing_mode')
            if pmode is not None:
                output_zarr.attrs['processing_mode'] = pmode
        # Save touched chunk indices for finalization step
        if touched_chunk_indices is not None:
            output_zarr.attrs['touched_chunk_indices'] = [list(idx) for idx in touched_chunk_indices]
            output_zarr.attrs['output_chunk_size'] = list(output_chunks[1:])

    progress_tracker.finalize()

    print(f"\n--- Merging Finished ---")
    print(f"Final merged output saved to: {output_path}")


# --- Command Line Interface ---
def main():
    """Entry point for the vesuvius.blend command line tool."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description='Merge partial inference outputs with Gaussian blending using fsspec.')
    parser.add_argument('parent_dir', type=str,
                        help='Directory containing the partial inference results (logits_part_X.zarr, coordinates_part_X.zarr)')
    parser.add_argument('output_path', type=str,
                        help='Path for the final merged Zarr output file.')
    parser.add_argument('--sigma_scale', type=float, default=8.0,
                        help='Sigma scale for Gaussian map (patch_size / sigma_scale). Default: 8.0')
    parser.add_argument('--chunk_size', type=str, default=None,
                        help='Spatial chunk size (Z,Y,X) for output Zarr. Comma-separated. If not specified, optimized size will be used.')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Number of worker processes. Default: CPU_COUNT - 1')
    parser.add_argument('--compression_level', type=int, default=1, choices=range(10),
                        help='Compression level (0-9, 0=none). Default: 1')
    parser.add_argument('--quiet', action='store_true',
                        help='Disable verbose progress messages (tqdm bars still show).')
    parser.add_argument('--input-zarr', type=str, default=None,
                        help='Path to original input zarr for chunks.json detection (auto-detected from logits metadata if not provided)')
    parser.add_argument('--chunks-filter-mode', type=str, default='auto',
                        choices=['auto', 'disabled'],
                        help='Chunk filtering: auto (use chunks.json if present), disabled (process full volume)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume blending by skipping chunks recorded in the progress file.')
    parser.add_argument('--progress-path', type=str, default=None,
                        help='Optional path to a progress JSON file. Defaults to <output_path>.progress.json')

    args = parser.parse_args()

    chunks = None
    if args.chunk_size:
        try:
            chunks = tuple(map(int, args.chunk_size.split(',')))
            if len(chunks) != 3: raise ValueError()
        except ValueError:
            parser.error("Invalid chunk_size format. Expected 3 comma-separated integers (Z,Y,X).")

    try:
        merge_inference_outputs(
            parent_dir=args.parent_dir,
            output_path=args.output_path,
            sigma_scale=args.sigma_scale,
            chunk_size=chunks,
            num_workers=args.num_workers,
            compression_level=args.compression_level,
            verbose=not args.quiet,
            input_zarr_path=args.input_zarr,
            chunks_filter_mode=args.chunks_filter_mode,
            resume=args.resume,
            progress_path=args.progress_path,
        )
        return 0
    except Exception as e:
        print(f"\n--- Blending Failed ---")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    import sys
    sys.exit(main())
