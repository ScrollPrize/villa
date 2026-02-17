import json
import multiprocessing as mp
import os
import re
import time
import traceback
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import fsspec
import numcodecs
import numpy as np
from scipy.ndimage import gaussian_filter
from tqdm.auto import tqdm

from vesuvius.data.utils import open_zarr
from vesuvius.utils.io.zarr_utils import wait_for_zarr_creation
from vesuvius.utils.k8s import get_tqdm_kwargs


class SpatialPatchGrid:
    """
    Unified spatial index for fast patch lookup.

    Handles loading coordinates, computing bounding boxes, detecting non-empty patches,
    and building a spatial grid for O(1) chunk-to-patches lookup.
    """

    def __init__(self, patch_size, grid_size=1000):
        """
        Initialize spatial grid.

        Args:
            patch_size: (pZ, pY, pX) patch dimensions
            grid_size: Size of each grid cell (default 1000^3)
        """
        self.grid_size = grid_size
        self.patch_size = patch_size
        self.grid = defaultdict(list)  # (gz, gy, gx) -> [(part_id, patch_idx), ...]
        self.coords_cache = {}  # part_id -> coordinates array
        self.non_empty_cache = {}  # part_id -> non-empty mask
        self.bbox_cache = {}  # part_id -> bbox dict
        self.stats = {
            'total_patches': 0,
            'skipped_empty': 0,
            'num_cells': 0
        }

    def _validate_zarr_format(self, logits_path):
        """Validate zarr v2 format with expected 5D chunk layout (1, C, Z, Y, X).

        Raises ValueError if the format doesn't match expectations.
        """
        zarray_path = os.path.join(logits_path, '.zarray')
        if logits_path.startswith('s3://'):
            with fsspec.open(zarray_path, 'r', anon=False) as f:
                zarray = json.load(f)
        else:
            with open(zarray_path, 'r') as f:
                zarray = json.load(f)

        zarr_format = zarray.get('zarr_format')
        if zarr_format != 2:
            raise ValueError(f"Expected zarr_format=2, got {zarr_format} in {logits_path}")

        chunks = zarray.get('chunks', [])
        if len(chunks) != 5:
            raise ValueError(f"Expected 5D chunks (N,C,Z,Y,X), got {len(chunks)}D in {logits_path}")
        if chunks[0] != 1:
            raise ValueError(f"Expected chunks[0]=1, got {chunks[0]} in {logits_path}")

    def _compute_non_empty_from_chunks(self, logits_path, total_patches):
        """Compute non-empty mask by listing zarr v2 chunk files (N.C.Z.Y.X format)."""
        self._validate_zarr_format(logits_path)

        non_empty_indices = set()

        def _parse_chunk_files(filenames):
            for filename in filenames:
                if not filename or not filename[0].isdigit():
                    continue
                parts = filename.split('.')
                if len(parts) != 5:
                    warnings.warn(f"Unexpected chunk filename '{filename}' in {logits_path}, expected N.C.Z.Y.X")
                    continue
                try:
                    patch_idx = int(parts[0])
                    non_empty_indices.add(patch_idx)
                except ValueError:
                    warnings.warn(f"Non-integer patch index in chunk filename '{filename}' in {logits_path}")

        if logits_path.startswith('s3://'):
            fs = fsspec.filesystem('s3', anon=False)
            zarr_path = logits_path.replace('s3://', '')
            files = fs.ls(zarr_path, detail=False)
            _parse_chunk_files(f.split('/')[-1] for f in files)
        else:
            _parse_chunk_files(os.listdir(logits_path))

        non_empty_mask = np.zeros(total_patches, dtype=bool)
        if non_empty_indices:
            non_empty_mask[list(non_empty_indices)] = True

        return non_empty_mask

    def _load_or_compute_non_empty(self, logits_path):
        """Load or compute non-empty patch mask."""
        cache_path = os.path.join(logits_path, '.non_empty_patch_idxs.json')

        # Try to load from cache
        try:
            if logits_path.startswith('s3://'):
                with fsspec.open(cache_path, 'r', anon=False) as f:
                    data = json.load(f)
            else:
                if os.path.exists(cache_path):
                    with open(cache_path, 'r') as f:
                        data = json.load(f)
                else:
                    data = None

            if data:
                non_empty_mask = np.zeros(data['num_patches'], dtype=bool)
                non_empty_mask[data['non_empty_indices']] = True
                return non_empty_mask
        except Exception as e:
            warnings.warn(f"Failed to load non-empty cache from {cache_path}: {e}")

        # Compute it
        logits_store = open_zarr(logits_path, mode='r',
                                storage_options={'anon': False} if logits_path.startswith('s3://') else None)
        total_patches = logits_store.shape[0]

        non_empty_mask = self._compute_non_empty_from_chunks(logits_path, total_patches)

        # Save as compact index list
        try:
            data = {
                'num_patches': int(total_patches),
                'non_empty_indices': np.where(non_empty_mask)[0].tolist()
            }
            if logits_path.startswith('s3://'):
                with fsspec.open(cache_path, 'w', anon=False) as f:
                    json.dump(data, f)
            else:
                with open(cache_path, 'w') as f:
                    json.dump(data, f)
        except Exception as e:
            warnings.warn(f"Failed to save non-empty cache to {cache_path}: {e}")

        return non_empty_mask

    def _compute_bbox(self, coords_np, patch_size):
        """Compute bounding box from coordinates."""
        if len(coords_np) == 0:
            return None

        pZ, pY, pX = patch_size
        return {
            'z_min': int(coords_np[:, 0].min()),
            'z_max': int(coords_np[:, 0].max()) + pZ,
            'y_min': int(coords_np[:, 1].min()),
            'y_max': int(coords_np[:, 1].max()) + pY,
            'x_min': int(coords_np[:, 2].min()),
            'x_max': int(coords_np[:, 2].max()) + pX
        }

    def _bbox_intersects_chunk(self, bbox, chunk):
        """Check if bbox intersects with chunk."""
        return (bbox['z_max'] > chunk['z_start'] and bbox['z_min'] < chunk['z_end'] and
                bbox['y_max'] > chunk['y_start'] and bbox['y_min'] < chunk['y_end'] and
                bbox['x_max'] > chunk['x_start'] and bbox['x_min'] < chunk['x_end'])

    def _load_aggregate_bbox_cache(self, parent_dir):
        """Load aggregate bbox cache from parent directory."""
        cache_path = os.path.join(parent_dir, '.bbox_cache.json')
        try:
            if parent_dir.startswith('s3://'):
                with fsspec.open(cache_path, 'r', anon=False) as f:
                    cache_data = json.load(f)
            else:
                if not os.path.exists(cache_path):
                    return None
                with open(cache_path, 'r') as f:
                    cache_data = json.load(f)
            return {int(k): v for k, v in cache_data.items()}
        except Exception as e:
            warnings.warn(f"Failed to load bbox cache from {cache_path}: {e}")
            return None

    def _save_aggregate_bbox_cache(self, parent_dir):
        """Save aggregate bbox cache to parent directory."""
        cache_path = os.path.join(parent_dir, '.bbox_cache.json')
        try:
            cache_data = {str(k): v for k, v in self.bbox_cache.items()}
            if parent_dir.startswith('s3://'):
                with fsspec.open(cache_path, 'w', anon=False) as f:
                    json.dump(cache_data, f, indent=2)
            else:
                with open(cache_path, 'w') as f:
                    json.dump(cache_data, f, indent=2)
        except Exception as e:
            warnings.warn(f"Failed to save bbox cache to {cache_path}: {e}")

    def build(self, parent_dir, part_files, part_ids, patch_size, chunks=None, tqdm_kwargs=None, verbose=False):
        """
        Build complete spatial index.

        Args:
            parent_dir: Directory containing part files
            part_files: Dict mapping part_id to {'logits': path, 'coordinates': path}
            part_ids: List of part IDs to process
            patch_size: (pZ, pY, pX) patch dimensions
            chunks: Optional list of chunks to filter relevant parts (for efficiency)
            tqdm_kwargs: Optional kwargs for tqdm progress bars
            verbose: Print progress messages
        """
        if tqdm_kwargs is None:
            tqdm_kwargs = {}

        print(f"\n--- Building Spatial Patch Index ---")

        # Step 1: Load bboxes from aggregate cache or compute them
        cached_bboxes = self._load_aggregate_bbox_cache(parent_dir)
        if cached_bboxes and all(pid in cached_bboxes for pid in part_ids):
            print(f"  Using cached bounding boxes for {len(part_ids)} parts")
            self.bbox_cache = {pid: cached_bboxes[pid] for pid in part_ids}
        else:
            print(f"  Loading/computing bounding boxes for {len(part_ids)} parts...")
            for part_id in tqdm(part_ids, desc="Loading bboxes", **tqdm_kwargs):
                coords_path = part_files[part_id]['coordinates']
                coords_store = open_zarr(coords_path, mode='r',
                                        storage_options={'anon': False} if coords_path.startswith('s3://') else None)

                if hasattr(coords_store, 'attrs') and 'bbox' in coords_store.attrs:
                    self.bbox_cache[part_id] = dict(coords_store.attrs['bbox'])
                else:
                    coords_np = coords_store[:]
                    self.coords_cache[part_id] = coords_np  # reuse in step 3
                    self.bbox_cache[part_id] = self._compute_bbox(coords_np, patch_size)

            self._save_aggregate_bbox_cache(parent_dir)

        # Step 2: Filter to only relevant parts if chunks provided
        if chunks:
            print(f"  Filtering to relevant parts based on {len(chunks)} chunks...")
            relevant_parts = set()
            for chunk in chunks:
                for part_id in part_ids:
                    bbox = self.bbox_cache.get(part_id)
                    if bbox and self._bbox_intersects_chunk(bbox, chunk):
                        relevant_parts.add(part_id)

            relevant_parts = sorted(relevant_parts)
            print(f"  Reduced from {len(part_ids)} to {len(relevant_parts)} relevant parts")
        else:
            relevant_parts = part_ids

        # Step 3: Load coordinates and non-empty masks for relevant parts
        print(f"  Loading coordinates and non-empty masks for {len(relevant_parts)} relevant parts...")
        for part_id in tqdm(relevant_parts, desc="Loading metadata", **tqdm_kwargs):
            if part_id not in self.coords_cache:
                coords_path = part_files[part_id]['coordinates']
                coords_store = open_zarr(coords_path, mode='r',
                                        storage_options={'anon': False} if coords_path.startswith('s3://') else None)
                self.coords_cache[part_id] = coords_store[:]

            logits_path = part_files[part_id]['logits']
            self.non_empty_cache[part_id] = self._load_or_compute_non_empty(logits_path)

        # Drop coords for non-relevant parts that were loaded during bbox computation
        relevant_set = set(relevant_parts)
        for part_id in list(self.coords_cache.keys()):
            if part_id not in relevant_set:
                del self.coords_cache[part_id]

        # Build spatial grid
        print(f"  Building spatial grid (cell size {self.grid_size}^3, non-empty patches only)...")
        pZ, pY, pX = patch_size

        for part_id, coords_np in self.coords_cache.items():
            non_empty_mask = self.non_empty_cache.get(part_id, None)

            for patch_idx in range(len(coords_np)):
                # Skip empty patches
                if non_empty_mask is not None and not non_empty_mask[patch_idx]:
                    self.stats['skipped_empty'] += 1
                    continue

                z, y, x = coords_np[patch_idx].tolist()

                # Find grid cells this patch intersects
                gz_min, gy_min, gx_min = z // self.grid_size, y // self.grid_size, x // self.grid_size
                gz_max = (z + pZ) // self.grid_size
                gy_max = (y + pY) // self.grid_size
                gx_max = (x + pX) // self.grid_size

                # Add to all overlapping cells
                for gz in range(gz_min, gz_max + 1):
                    for gy in range(gy_min, gy_max + 1):
                        for gx in range(gx_min, gx_max + 1):
                            self.grid[(gz, gy, gx)].append((part_id, patch_idx))

                self.stats['total_patches'] += 1

        self.stats['num_cells'] = len(self.grid)

        if verbose:
            avg = sum(len(v) for v in self.grid.values()) / self.stats['num_cells'] if self.stats['num_cells'] > 0 else 0
            print(f"  ✓ Grid complete: {self.stats['num_cells']} cells, {self.stats['total_patches']} non-empty patches")
            print(f"    Skipped {self.stats['skipped_empty']} empty patches, avg {avg:.1f} patches/cell")

    def get_chunk_patches(self, chunk_info):
        """
        Get patches that intersect a chunk, grouped by part_id with coords.

        Performs coarse grid lookup then exact intersection filtering.

        Args:
            chunk_info: Dict with z_start, z_end, y_start, y_end, x_start, x_end

        Returns:
            Dict {part_id: [(patch_idx, z, y, x), ...]} for intersecting patches
        """
        z_start, z_end = chunk_info['z_start'], chunk_info['z_end']
        y_start, y_end = chunk_info['y_start'], chunk_info['y_end']
        x_start, x_end = chunk_info['x_start'], chunk_info['x_end']
        pZ, pY, pX = self.patch_size

        # Find overlapping grid cells
        gz_min, gy_min, gx_min = z_start // self.grid_size, y_start // self.grid_size, x_start // self.grid_size
        gz_max = (z_end - 1) // self.grid_size
        gy_max = (y_end - 1) // self.grid_size
        gx_max = (x_end - 1) // self.grid_size

        # Collect unique patches from overlapping grid cells
        seen = set()
        result = defaultdict(list)
        for gz in range(gz_min, gz_max + 1):
            for gy in range(gy_min, gy_max + 1):
                for gx in range(gx_min, gx_max + 1):
                    for entry in self.grid.get((gz, gy, gx), []):
                        if entry in seen:
                            continue
                        seen.add(entry)

                        part_id, patch_idx = entry
                        z, y, x = self.coords_cache[part_id][patch_idx].tolist()

                        # Exact intersection check (grid cells are coarse)
                        if (z + pZ <= z_start or z >= z_end or
                            y + pY <= y_start or y >= y_end or
                            x + pX <= x_start or x >= x_end):
                            continue

                        result[part_id].append((patch_idx, z, y, x))

        return dict(result)

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


# --- Worker Process State ---
_worker_state = {}


def _init_worker(part_files, output_path, gaussian_map, patch_size, num_classes, is_s3,
                 finalize_config=None):
    """Initialize per-worker process state with cached zarr stores.

    Called once when each worker process starts. Zarr stores (and their
    underlying S3/HTTP connections) are reused across all chunks the worker
    processes, avoiding repeated metadata reads and connection setup.

    Args:
        finalize_config: Optional FinalizeConfig. When set, chunks are finalized
            (softmax + uint8) inline instead of writing float16 blended logits.
    """
    numcodecs.blosc.use_threads = False
    storage_opts = {'anon': False} if is_s3 else None
    _worker_state.update({
        'part_files': part_files,
        'gaussian_map': gaussian_map,
        'patch_size': patch_size,
        'num_classes': num_classes,
        'is_s3': is_s3,
        'logits_stores': {},
        'output_store': open_zarr(output_path, mode='r+', storage_options=storage_opts),
        'finalize_config': finalize_config,
    })


def process_chunk(chunk_info, chunk_patches, epsilon=1e-8):
    """
    Process a single chunk using worker-cached zarr stores.

    Must be called from a worker initialized with _init_worker.

    Args:
        chunk_info: Dictionary with chunk boundaries
        chunk_patches: Dict {part_id: [(patch_idx, z, y, x), ...]} precomputed
                       for this chunk (non-empty, intersecting patches only)
        epsilon: Small value for numerical stability
    """
    part_files = _worker_state['part_files']
    gaussian_map = _worker_state['gaussian_map']
    patch_size = _worker_state['patch_size']
    num_classes = _worker_state['num_classes']
    is_s3 = _worker_state['is_s3']
    logits_stores = _worker_state['logits_stores']
    output_store = _worker_state['output_store']

    z_start, z_end = chunk_info['z_start'], chunk_info['z_end']
    y_start, y_end = chunk_info['y_start'], chunk_info['y_end']
    x_start, x_end = chunk_info['x_start'], chunk_info['x_end']

    pZ, pY, pX = patch_size

    gaussian_map_spatial_np = gaussian_map[0]  # Shape (pZ, pY, pX)

    chunk_shape = (num_classes, z_end - z_start, y_end - y_start, x_end - x_start)
    weights_shape = (z_end - z_start, y_end - y_start, x_end - x_start)

    chunk_logits = np.zeros(chunk_shape, dtype=np.float32)
    chunk_weights = np.zeros(weights_shape, dtype=np.float32)

    patches_processed = 0
    patches_skipped_empty = 0
    num_patch_reads = 0
    bytes_per_read = num_classes * pZ * pY * pX * 4  # uncompressed float32

    for part_id, patches in chunk_patches.items():
        if part_id not in logits_stores:
            logits_path = part_files[part_id]['logits']
            storage_opts = {'anon': False} if is_s3 else None
            logits_stores[part_id] = open_zarr(logits_path, mode='r', storage_options=storage_opts)
        logits_store = logits_stores[part_id]

        for patch_idx, z, y, x in patches:
            if (z + pZ <= z_start or z >= z_end or
                y + pY <= y_start or y >= y_end or
                x + pX <= x_start or x >= x_end):
                continue

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
            num_patch_reads += 1

            # Skip patches with no values - don't let empty patches contribute to weights
            if not np.any(logit_patch != 0):
                patches_skipped_empty += 1
                continue

            weight_patch = gaussian_map_spatial_np[
                slice(pz_start, pz_end),
                slice(py_start, py_end),
                slice(px_start, px_end)
            ]

            weighted_patch = logit_patch * weight_patch[np.newaxis, :, :, :]

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
        output_slice = (
            slice(None),
            slice(z_start, z_end),
            slice(y_start, y_end),
            slice(x_start, x_end)
        )

        normalized = np.zeros_like(chunk_logits)
        np.divide(chunk_logits, chunk_weights[np.newaxis, :, :, :] + epsilon,
                  out=normalized, where=chunk_weights[np.newaxis, :, :, :] > 0)

        finalize_config = _worker_state.get('finalize_config')
        if finalize_config is not None:
            from vesuvius.models.run.finalize_outputs import apply_finalization
            result, is_empty = apply_finalization(normalized, num_classes, finalize_config)
            if not is_empty:
                # Finalized output may have different channel count than blended logits;
                # write using slices that match the finalized shape.
                finalized_slice = (
                    slice(None),
                    slice(z_start, z_end),
                    slice(y_start, y_end),
                    slice(x_start, x_end)
                )
                output_store[finalized_slice] = result
        else:
            output_store[output_slice] = normalized.astype(np.float16)

    return {
        'chunk': chunk_info,
        'patches_processed': patches_processed,
        'patches_skipped_empty': patches_skipped_empty,
        'num_patch_reads': num_patch_reads,
        'bytes_read_uncompressed': num_patch_reads * bytes_per_read,
    }

# --- Utility Functions ---
def calculate_chunks(volume_shape, output_chunks=None, z_range=None):

    Z, Y, X = volume_shape

    if output_chunks is None:
        z_chunk, y_chunk, x_chunk = 256, 256, 256
    else:
        z_chunk, y_chunk, x_chunk = output_chunks
    
    chunks = []
    for z_start in range(0, Z, z_chunk):
        for y_start in range(0, Y, y_chunk):
            for x_start in range(0, X, x_chunk):
                z_end = min(z_start + z_chunk, Z)
                y_end = min(y_start + y_chunk, Y)
                x_end = min(x_start + x_chunk, X)

                # Apply Z-range filtering if specified
                if z_range is not None:
                    range_z_start, range_z_end = z_range
                    # Only include chunks whose end is inside the range
                    if not (range_z_start < z_end and range_z_end >= z_end):
                        continue  # Skip chunks outside the Z-range

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
        num_parts: int = 1,  # Number of parts to split processing into
        global_part_id: int = 0,  # Part ID for this process (0-indexed)
        finalize_config=None):  # Optional FinalizeConfig — fuse finalization when set
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
        num_parts: Number of parts to split the blending process into.
        global_part_id: Part ID for this process (0-indexed). Used for Z-axis partitioning.
        finalize_config: Optional FinalizeConfig. When provided, softmax + uint8 quantization
            is applied inline after blending (fused mode), skipping the intermediate float16 array.
    """

    tqdm_kwargs = get_tqdm_kwargs()
    if not verbose:
        tqdm_kwargs['disable'] = True

    # blosc has an issuse with threading , so we disable it
    numcodecs.blosc.use_threads = False
    if num_workers is None:
        # just use half the cpu count 
        num_workers = max(1, mp.cpu_count() // 2)
    
    print(f"Using {num_workers} worker processes (half of CPU count for memory efficiency)")

    # Add partitioning information
    if num_parts > 1:
        print(f"Partitioned blending: Processing part {global_part_id}/{num_parts}")

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
    print(f"Found {len(part_ids)} parts (IDs {part_ids[0]}..{part_ids[-1]})")

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
    # When fused mode is active, populate finalize_config with multi-task metadata
    # from the logits zarr attrs and use the finalized shape/dtype.
    if finalize_config is not None:
        if hasattr(part0_logits_store, 'attrs'):
            finalize_config.is_multi_task = part0_logits_store.attrs.get('is_multi_task', False)
            finalize_config.target_info = part0_logits_store.attrs.get('target_info', None)

        from vesuvius.models.run.finalize_outputs import compute_finalized_shape
        output_shape = compute_finalized_shape(original_volume_shape, num_classes, finalize_config)
        output_dtype = np.uint8
        print(f"Fused blend+finalize mode: {finalize_config.mode}, threshold={finalize_config.threshold}")
        print(f"  Finalized output shape: {output_shape}")
    else:
        output_shape = (num_classes, *original_volume_shape)  # (C, D, H, W)
        output_dtype = np.float16

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

    # --- 3. Create or Open Output Arrays ---
    if global_part_id == 0:
        # Part 0 creates the arrays
        print(f"Creating final output store: {output_path}")
        print(f"  Shape: {output_shape}, Chunks: {output_chunks}")

        open_zarr(
            path=output_path,
            mode='w',
            storage_options={'anon': False} if output_path.startswith('s3://') else None,
            verbose=verbose,
            shape=output_shape,
            chunks=output_chunks,
            compressor=compressor,
            dtype=output_dtype,
            fill_value=0,
            write_empty_chunks=False
        )
    else:
        # Other parts wait for part 0 to create the arrays, then open them in r+ mode
        print(f"Waiting for part 0 to create output arrays...")

        wait_for_zarr_creation(output_path, verbose=verbose, part_id=global_part_id)

        print(f"Arrays found! Opening in r+ mode for part {global_part_id}")

    # --- 4. Generate Gaussian Map ---
    gaussian_map = generate_gaussian_map(patch_size, sigma_scale=sigma_scale)

    # --- 5. Calculate Z-range for this part ---
    z_range = None
    if num_parts > 1:
        total_z = original_volume_shape[0]  # Z dimension
        z_start = (global_part_id * total_z) // num_parts
        z_end = ((global_part_id + 1) * total_z) // num_parts
        z_range = (z_start, z_end)
        print(f"Part {global_part_id} processing Z-range: {z_start} to {z_end} (out of {total_z})")

    # --- 6. Calculate Processing Chunks ---
    chunks = calculate_chunks(
        original_volume_shape,
        output_chunks=output_chunks[1:],  # Skip the class dimension from output_chunks
        z_range=z_range
    )

    print(f"Divided volume into {len(chunks)} chunks for parallel processing")

    # --- 7. Build Spatial Patch Index ---
    spatial_index = SpatialPatchGrid(patch_size=patch_size, grid_size=1000)
    spatial_index.build(
        parent_dir=parent_dir,
        part_files=part_files,
        part_ids=part_ids,
        patch_size=patch_size,
        chunks=chunks,  # Pass chunks for filtering
        tqdm_kwargs=tqdm_kwargs,
        verbose=verbose
    )

    # --- 8. Precompute Patch Assignments per Chunk ---
    print("\n--- Precomputing Patch Assignments per Chunk ---")
    chunk_work_items = []
    total_patch_assignments = 0
    skipped_empty_chunks = 0

    for chunk in chunks:
        chunk_patches = spatial_index.get_chunk_patches(chunk)

        if not chunk_patches:
            skipped_empty_chunks += 1
            continue

        chunk_work_items.append({
            'chunk_info': chunk,
            'chunk_patches': chunk_patches
        })
        total_patch_assignments += sum(len(v) for v in chunk_patches.values())

    print(f"Total chunks: {len(chunks)}")
    print(f"  Skipped (no patches): {skipped_empty_chunks}")
    print(f"  To process: {len(chunk_work_items)}")
    print(f"  Total patch assignments: {total_patch_assignments}")

    # --- 9. Process Chunks in Parallel ---
    print("\n--- Accumulating Weighted Patches ---")

    is_s3 = parent_dir.startswith('s3://')

    total_patches_processed = 0
    total_patches_skipped_empty = 0
    total_patch_reads = 0
    total_bytes_read = 0
    chunks_completed = 0
    wall_start = time.perf_counter()

    def print_progress_stats():
        elapsed = time.perf_counter() - wall_start
        print(f"\n=== Stats after {chunks_completed}/{len(chunk_work_items)} chunks ({elapsed:.1f}s wall) ===")
        print(f"  Patches processed: {total_patches_processed}")
        if total_patches_skipped_empty:
            print(f"  Patches skipped (empty after load): {total_patches_skipped_empty}")
        if chunks_completed > 0:
            rate = chunks_completed / elapsed
            remaining = (len(chunk_work_items) - chunks_completed) / rate if rate > 0 else 0
            print(f"  Rate: {rate:.1f} chunks/s, ETA: {remaining:.0f}s")
        if total_patch_reads > 0:
            gb_read = total_bytes_read / (1024**3)
            print(f"  Zarr reads: {total_patch_reads} patches, {gb_read:.2f} GB uncompressed")
            print(f"  Wall throughput: {gb_read / elapsed:.2f} GB/s aggregate")

    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_init_worker,
        initargs=(part_files, output_path, gaussian_map, patch_size, num_classes, is_s3, finalize_config)
    ) as executor:
        future_to_chunk = {
            executor.submit(
                process_chunk,
                chunk_info=item['chunk_info'],
                chunk_patches=item['chunk_patches']
            ): item for item in chunk_work_items
        }

        for future in tqdm(
            as_completed(future_to_chunk),
            total=len(chunk_work_items),
            desc="Processing Chunks",
            **tqdm_kwargs,
        ):
            try:
                result = future.result()
                total_patches_processed += result['patches_processed']
                total_patches_skipped_empty += result.get('patches_skipped_empty', 0)
                total_patch_reads += result.get('num_patch_reads', 0)
                total_bytes_read += result.get('bytes_read_uncompressed', 0)

                chunks_completed += 1

                if chunks_completed % 100 == 0:
                    print_progress_stats()
            except Exception as e:
                print(f"Error processing chunk: {e}")
                raise e

    wall_elapsed = time.perf_counter() - wall_start
    print(f"\n{'='*60}")
    print(f"=== FINAL STATISTICS ===")
    print(f"{'='*60}")
    print(f"  Wall time: {wall_elapsed:.1f}s")
    print(f"  Chunks processed: {chunks_completed}")
    print(f"  Chunks skipped (empty): {skipped_empty_chunks}")
    print(f"  Patches processed: {total_patches_processed}")
    if total_patches_skipped_empty:
        print(f"  Patches skipped (empty after load): {total_patches_skipped_empty}")
    print(f"  Workers: {num_workers}")
    if chunks_completed > 0:
        print(f"  Avg rate: {chunks_completed / wall_elapsed:.1f} chunks/s")
    if total_patch_reads > 0:
        gb_read = total_bytes_read / (1024**3)
        avg_bytes = total_bytes_read / total_patch_reads
        print(f"  Zarr chunk reads: {total_patch_reads}")
        print(f"  Avg chunk size (uncompressed): {avg_bytes / 1024:.1f} KB")
        print(f"  Total data (uncompressed): {gb_read:.2f} GB")
        print(f"  Wall throughput: {gb_read / wall_elapsed:.2f} GB/s ({num_workers} workers)")
    print(f"{'='*60}")

    # --- 10. Save Metadata ---
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

        # Add finalization metadata when in fused mode
        if finalize_config is not None:
            output_zarr.attrs['processing_mode'] = finalize_config.mode
            output_zarr.attrs['threshold_applied'] = finalize_config.threshold
            output_zarr.attrs['fused_blend_finalize'] = True

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
    parser.add_argument('--num_parts', type=int, default=1,
                        help='Number of parts to split the blending process into. Default: 1')
    parser.add_argument('--part_id', type=int, default=0,
                        help='Part ID for this process (0-indexed). Default: 0')

    args = parser.parse_args()

    # Validate partitioning arguments
    if args.part_id < 0 or args.part_id >= args.num_parts:
        parser.error(f"Invalid part_id {args.part_id} for num_parts {args.num_parts}. part_id must be 0 <= part_id < num_parts")

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
            num_parts=args.num_parts,
            global_part_id=args.part_id
        )
        return 0
    except Exception as e:
        print(f"\n--- Blending Failed ---")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

def blend_and_finalize_main():
    """Entry point for vesuvius.blend_and_finalize — fused blend + finalize in one pass."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description='Blend partial inference outputs and finalize (softmax + uint8) in a single pass.')
    parser.add_argument('parent_dir', type=str,
                        help='Directory containing the partial inference results (logits_part_X.zarr, coordinates_part_X.zarr)')
    parser.add_argument('output_path', type=str,
                        help='Path for the final output Zarr file (uint8, finalized).')
    # Blending args
    parser.add_argument('--sigma_scale', type=float, default=8.0,
                        help='Sigma scale for Gaussian map (patch_size / sigma_scale). Default: 8.0')
    parser.add_argument('--chunk_size', type=str, default=None,
                        help='Spatial chunk size (Z,Y,X) for output Zarr. Comma-separated.')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Number of worker processes. Default: CPU_COUNT // 2')
    parser.add_argument('--compression_level', type=int, default=1, choices=range(10),
                        help='Compression level (0-9, 0=none). Default: 1')
    parser.add_argument('--quiet', action='store_true',
                        help='Disable verbose progress messages.')
    parser.add_argument('--num_parts', type=int, default=1,
                        help='Number of parts to split the blending process into. Default: 1')
    parser.add_argument('--part_id', type=int, default=0,
                        help='Part ID for this process (0-indexed). Default: 0')
    # Finalization args
    parser.add_argument('--mode', type=str, choices=['binary', 'multiclass'], default='binary',
                        help='Finalization mode. Default: binary')
    parser.add_argument('--threshold', action='store_true',
                        help='Apply argmax and only save class predictions (no probabilities).')

    args = parser.parse_args()

    if args.part_id < 0 or args.part_id >= args.num_parts:
        parser.error(f"Invalid part_id {args.part_id} for num_parts {args.num_parts}.")

    chunks = None
    if args.chunk_size:
        try:
            chunks = tuple(map(int, args.chunk_size.split(',')))
            if len(chunks) != 3:
                raise ValueError()
        except ValueError:
            parser.error("Invalid chunk_size format. Expected 3 comma-separated integers (Z,Y,X).")

    from vesuvius.models.run.finalize_outputs import FinalizeConfig
    finalize_config = FinalizeConfig(mode=args.mode, threshold=args.threshold)

    try:
        merge_inference_outputs(
            parent_dir=args.parent_dir,
            output_path=args.output_path,
            sigma_scale=args.sigma_scale,
            chunk_size=chunks,
            num_workers=args.num_workers,
            compression_level=args.compression_level,
            verbose=not args.quiet,
            num_parts=args.num_parts,
            global_part_id=args.part_id,
            finalize_config=finalize_config,
        )
        return 0
    except Exception as e:
        print(f"\n--- Blend and Finalize Failed ---")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
