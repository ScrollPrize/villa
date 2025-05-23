import numpy as np
import os
import re
import json
import zarr
import fsspec
import multiprocessing as mp
from tqdm.auto import tqdm
from scipy.ndimage import gaussian_filter
from functools import partial
import numcodecs
from concurrent.futures import ProcessPoolExecutor
import math
from collections import defaultdict
from data.utils import open_zarr


# --- Gaussian Map Generation ---
def generate_gaussian_map(patch_size: tuple, sigma_scale: float = 8.0, verbose: bool = False) -> np.ndarray:
    """
    Generates a Gaussian importance map for a given patch size.
    Weights decay from the center towards the edges.
    Shape: (1, pZ, pY, pX) for easy broadcasting.
    """
    pZ, pY, pX = patch_size
    tmp = np.zeros(patch_size, dtype=np.float32)
    center_coords = [i // 2 for i in patch_size]
    sigmas = [i / sigma_scale for i in patch_size]

    tmp[tuple(center_coords)] = 1

    gaussian_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
    # Safeguard against division by zero
    gaussian_map /= max(gaussian_map.max(), 1e-12)
    gaussian_map = gaussian_map.reshape(1, pZ, pY, pX)
    gaussian_map = np.clip(gaussian_map, a_min=0, a_max=None)
    
    if verbose:
        print(
            f"Generated Gaussian map with shape {gaussian_map.shape}, min: {gaussian_map.min():.4f}, max: {gaussian_map.max():.4f}")
    return gaussian_map


# --- Vectorized Coordinate Utilities ---
def vectorized_intersection_calc(patch_coords, patch_size, chunk_bounds):
    """
    Vectorized calculation of patch-chunk intersections for batch processing.
    
    Args:
        patch_coords: Array of patch coordinates, shape (N, 3) for N patches
        patch_size: Tuple (pZ, pY, pX)
        chunk_bounds: Dict with chunk boundaries
        
    Returns:
        Dictionary with intersection information for valid patches
    """
    pZ, pY, pX = patch_size
    z_start, z_end = chunk_bounds['z_start'], chunk_bounds['z_end']
    y_start, y_end = chunk_bounds['y_start'], chunk_bounds['y_end']
    x_start, x_end = chunk_bounds['x_start'], chunk_bounds['x_end']
    
    # Vectorized intersection calculation
    patch_z = patch_coords[:, 0]
    patch_y = patch_coords[:, 1]
    patch_x = patch_coords[:, 2]
    
    # Calculate intersection bounds using vectorized operations
    # Intersection with chunk in volume space
    iz_start = np.maximum(patch_z, z_start) - z_start
    iz_end = np.minimum(patch_z + pZ, z_end) - z_start
    iy_start = np.maximum(patch_y, y_start) - y_start
    iy_end = np.minimum(patch_y + pY, y_end) - y_start
    ix_start = np.maximum(patch_x, x_start) - x_start
    ix_end = np.minimum(patch_x + pX, x_end) - x_start
    
    # Patch internal coordinates (for reading from logits)
    pz_start = np.maximum(z_start - patch_z, 0)
    pz_end = pZ - np.maximum(patch_z + pZ - z_end, 0)
    py_start = np.maximum(y_start - patch_y, 0)
    py_end = pY - np.maximum(patch_y + pY - y_end, 0)
    px_start = np.maximum(x_start - patch_x, 0)
    px_end = pX - np.maximum(patch_x + pX - x_end, 0)
    
    # Find valid intersections (patches that actually intersect the chunk)
    valid_mask = ((iz_end > iz_start) & (iy_end > iy_start) & (ix_end > ix_start) &
                  (pz_end > pz_start) & (py_end > py_start) & (px_end > px_start))
    
    return {
        'valid_mask': valid_mask,
        'intersection_bounds': {
            'iz_start': iz_start[valid_mask],
            'iz_end': iz_end[valid_mask],
            'iy_start': iy_start[valid_mask],
            'iy_end': iy_end[valid_mask],
            'ix_start': ix_start[valid_mask],
            'ix_end': ix_end[valid_mask],
        },
        'patch_bounds': {
            'pz_start': pz_start[valid_mask],
            'pz_end': pz_end[valid_mask],
            'py_start': py_start[valid_mask],
            'py_end': py_end[valid_mask],
            'px_start': px_start[valid_mask],
            'px_end': px_end[valid_mask],
        }
    }


def calculate_memory_budget(ram_budget_gb=100):
    # Convert to bytes
    ram_budget_bytes = ram_budget_gb * 1024**3
    
    # Reserve 20% for system overhead and other operations
    usable_memory = int(ram_budget_bytes * 0.8)
    
    return {
        'total_budget_bytes': ram_budget_bytes,
        'usable_memory_bytes': usable_memory,
        'chunk_buffer_bytes': int(usable_memory * 0.6),  # 60% for chunk buffers
        'accumulator_bytes': int(usable_memory * 0.3),   # 30% for accumulators
        'working_memory_bytes': int(usable_memory * 0.1) # 10% for working space
    }


def group_patches_by_zarr_chunks(part_files, part_ids, patch_size):
    """
    Group patches by their source zarr chunk locations for efficient batch reading.
    
    Args:
        part_files: Dictionary of part files
        part_ids: List of part IDs
        patch_size: Tuple (pZ, pY, pX)
        
    Returns:
        Dictionary mapping zarr chunk coordinates to patch lists
    """
    zarr_chunk_groups = defaultdict(list)
    
    for part_id in part_ids:
        coords_path = part_files[part_id]['coordinates']
        coords_store = open_zarr(coords_path, mode='r', storage_options={'anon': False} if coords_path.startswith('s3://') else None)
        coords_np = coords_store[:]
        
        # Since zarr chunk size = patch size, each patch corresponds to one zarr chunk
        # Group patches by their zarr chunk coordinate (which is the same as patch coordinate)
        for patch_idx, (z, y, x) in enumerate(coords_np):
            zarr_chunk_key = (part_id, z, y, x)  # Include part_id to distinguish sources
            zarr_chunk_groups[zarr_chunk_key].append(patch_idx)
    
    return zarr_chunk_groups


def batch_read_logits_chunks(zarr_chunk_groups, part_files, memory_budget, num_classes, patch_size, num_workers):
    """
    Efficiently read logits in batches based on memory budget.
    
    Args:
        zarr_chunk_groups: Groups of patches by zarr chunk location
        part_files: Dictionary of part files  
        memory_budget: Memory allocation information
        num_classes: Number of classes in logits
        patch_size: Tuple (pZ, pY, pX)
        num_workers: Number of worker processes (for memory budget division)
        
    Yields:
        Batches of (chunk_keys, logits_batch, coords_batch)
    """
    pZ, pY, pX = patch_size
    
    # Calculate memory per zarr chunk (float16 logits)
    bytes_per_chunk = num_classes * pZ * pY * pX * 2  # float16 = 2 bytes
    
    # Divide memory budget by number of workers to avoid over-allocation
    worker_memory_budget = memory_budget['chunk_buffer_bytes'] // num_workers
    max_chunks_per_batch = max(1, worker_memory_budget // bytes_per_chunk)
    
    chunk_keys = list(zarr_chunk_groups.keys())
    
    # Process in batches
    for i in range(0, len(chunk_keys), max_chunks_per_batch):
        batch_keys = chunk_keys[i:i + max_chunks_per_batch]
        batch_logits = []
        batch_coords = []
        
        # Group by part_id to minimize zarr store opening
        parts_in_batch = defaultdict(list)
        for key in batch_keys:
            part_id = key[0]
            parts_in_batch[part_id].append(key)
        
        # Read from each part
        for part_id, part_keys in parts_in_batch.items():
            logits_path = part_files[part_id]['logits']
            coords_path = part_files[part_id]['coordinates']
            
            logits_store = open_zarr(logits_path, mode='r', storage_options={'anon': False} if logits_path.startswith('s3://') else None)
            coords_store = open_zarr(coords_path, mode='r', storage_options={'anon': False} if coords_path.startswith('s3://') else None)
            
            # Collect patch indices for this part
            patch_indices = []
            for key in part_keys:
                patch_indices.extend(zarr_chunk_groups[key])
            
            # Batch read logits and coordinates
            if patch_indices:
                patch_indices = np.array(patch_indices)
                logits_batch_part = logits_store[patch_indices]  # Shape: (N, C, pZ, pY, pX)
                coords_batch_part = coords_store[patch_indices]  # Shape: (N, 3)
                
                batch_logits.append(logits_batch_part)
                batch_coords.append(coords_batch_part)
        
        # Concatenate all parts in this batch
        if batch_logits:
            full_batch_logits = np.concatenate(batch_logits, axis=0)
            full_batch_coords = np.concatenate(batch_coords, axis=0)
            
            yield batch_keys, full_batch_logits, full_batch_coords


def group_patches_by_intersection_pattern(intersection_info, valid_logits):
    """
    Group patches that have identical intersection patterns for vectorized processing.
    
    Args:
        intersection_info: Pre-computed intersection information
        valid_logits: Valid logits from batch
        
    Returns:
        List of groups, each containing (pattern_bounds, indices_in_group)
    """
    intersection_bounds = intersection_info['intersection_bounds']
    patch_bounds = intersection_info['patch_bounds']
    
    # Create pattern signature for each patch
    pattern_groups = defaultdict(list)
    
    for i in range(len(valid_logits)):
        # Create a signature from bounds
        pattern = (
            int(intersection_bounds['iz_start'][i]), int(intersection_bounds['iz_end'][i]),
            int(intersection_bounds['iy_start'][i]), int(intersection_bounds['iy_end'][i]),
            int(intersection_bounds['ix_start'][i]), int(intersection_bounds['ix_end'][i]),
            int(patch_bounds['pz_start'][i]), int(patch_bounds['pz_end'][i]),
            int(patch_bounds['py_start'][i]), int(patch_bounds['py_end'][i]),
            int(patch_bounds['px_start'][i]), int(patch_bounds['px_end'][i])
        )
        pattern_groups[pattern].append(i)
    
    return list(pattern_groups.items())


def spatially_grouped_accumulate(chunk_logits, chunk_weights, batch_logits, batch_coords, 
                                intersection_info, gaussian_map_np, chunk_bounds, patch_size, 
                                epsilon=1e-8):
    """
    Optimized accumulation using spatial grouping for vectorized processing.
    Groups patches with identical intersection patterns and processes them together.
    
    Args:
        chunk_logits: Accumulator array for chunk
        chunk_weights: Weight accumulator for chunk  
        batch_logits: Batch of logits to process
        batch_coords: Batch of coordinates
        intersection_info: Pre-computed intersection information
        gaussian_map_np: Gaussian weight map
        chunk_bounds: Chunk boundary information
        patch_size: Tuple (pZ, pY, pX)
        epsilon: Small value to avoid division by zero
    """
    if not intersection_info['valid_mask'].any():
        return
    
    # Extract valid data
    valid_logits = batch_logits[intersection_info['valid_mask']]
    
    # Group patches by their intersection patterns
    pattern_groups = group_patches_by_intersection_pattern(intersection_info, valid_logits)
    
    # Process each group of patches with identical patterns
    for pattern, group_indices in pattern_groups:
        if len(group_indices) == 0:
            continue
            
        # Unpack pattern bounds
        iz_start, iz_end, iy_start, iy_end, ix_start, ix_end, pz_start, pz_end, py_start, py_end, px_start, px_end = pattern
        
        # Extract all patches in this group
        group_logits = valid_logits[group_indices]  # Shape: (group_size, C, pZ, pY, pX)
        
        # Extract patch data and weights (same for all patches in group)
        group_patch_data = group_logits[:, :, pz_start:pz_end, py_start:py_end, px_start:px_end]
        weight_data = gaussian_map_np[pz_start:pz_end, py_start:py_end, px_start:px_end]
        
        # Vectorized accumulation for entire group
        # Sum across all patches in the group and multiply by weights
        group_contribution = np.sum(group_patch_data, axis=0) * weight_data[np.newaxis, :, :, :]
        
        # Add to chunk accumulators
        chunk_logits[:, iz_start:iz_end, iy_start:iy_end, ix_start:ix_end] += group_contribution
        chunk_weights[iz_start:iz_end, iy_start:iy_end, ix_start:ix_end] += weight_data * len(group_indices)


# --- Optimized Single-Pass Chunk Processing ---
def process_chunk(chunk_info, part_files, part_ids, output_path, gaussian_map_np, 
                 patch_size, memory_budget, num_workers, epsilon=1e-8):
    """
    Optimized single-pass chunk processing with vectorized operations and batch reading.
    Combines accumulation and normalization in one pass to eliminate redundant I/O.
    
    Args:
        chunk_info: Dictionary with chunk boundaries
        part_files: Dictionary of part files
        part_ids: List of part IDs
        output_path: Path to output zarr
        gaussian_map_np: Pre-computed Gaussian map as numpy array (pZ, pY, pX)
        patch_size: Size of patches (pZ, pY, pX)
        memory_budget: Memory allocation information
        num_workers: Number of worker processes (for memory budget division)
        epsilon: Small value to avoid division by zero
    """
    # Extract chunk boundaries
    z_start, z_end = chunk_info['z_start'], chunk_info['z_end']
    y_start, y_end = chunk_info['y_start'], chunk_info['y_end']
    x_start, x_end = chunk_info['x_start'], chunk_info['x_end']
    
    pZ, pY, pX = patch_size
    
    # gaussian_map_np is already provided as numpy array (pZ, pY, pX)
    
    # Open output store
    output_store = open_zarr(output_path, mode='r+', storage_options={'anon': False} if output_path.startswith('s3://') else None)
    
    # Initialize local accumulators
    num_classes = output_store.shape[0]
    chunk_shape = (num_classes, z_end - z_start, y_end - y_start, x_end - x_start)
    weights_shape = (z_end - z_start, y_end - y_start, x_end - x_start)
    
    chunk_logits = np.zeros(chunk_shape, dtype=np.float32)
    chunk_weights = np.zeros(weights_shape, dtype=np.float32)
    
    patches_processed = 0
    
    # Optimized path: Group patches by zarr chunks and process in batches
    zarr_chunk_groups = group_patches_by_zarr_chunks(part_files, part_ids, patch_size)
    
    # Filter groups to only include patches that intersect with this chunk
    relevant_groups = {}
    for zarr_key, patch_indices in zarr_chunk_groups.items():
        part_id, z, y, x = zarr_key
        
        # Quick check if this zarr chunk could intersect with our volume chunk
        if (z + pZ > z_start and z < z_end and
            y + pY > y_start and y < y_end and 
            x + pX > x_start and x < x_end):
            relevant_groups[zarr_key] = patch_indices
    
    # Process relevant groups in memory-efficient batches
    for batch_keys, batch_logits, batch_coords in batch_read_logits_chunks(
        relevant_groups, part_files, memory_budget, num_classes, patch_size, num_workers):
        
        # Vectorized intersection calculation for entire batch
        intersection_info = vectorized_intersection_calc(batch_coords, patch_size, chunk_info)
        
        # Spatially grouped accumulation for optimal vectorization
        spatially_grouped_accumulate(
            chunk_logits, chunk_weights, batch_logits, batch_coords,
            intersection_info, gaussian_map_np, chunk_info, patch_size, epsilon
        )
        
        patches_processed += intersection_info['valid_mask'].sum()
    
    # Single-pass normalization: normalize and write immediately
    if patches_processed > 0:
        # In-place vectorized normalization - avoid creating new array
        weight_mask = chunk_weights > 0
        chunk_weights_expanded = chunk_weights[np.newaxis, :, :, :] + epsilon
        np.divide(chunk_logits, chunk_weights_expanded, 
                  out=chunk_logits, where=weight_mask[np.newaxis, :, :, :])
        
        # Write normalized data directly to output
        output_slice = (
            slice(None),  # All classes
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
def calculate_chunks(volume_shape, output_chunks=None):
    """
    Calculate processing units based directly on zarr chunk size for memory efficiency.
    
    Args:
        volume_shape: Shape of the volume (Z, Y, X)
        output_chunks: Spatial chunk size for the output zarr (z_chunk, y_chunk, x_chunk)
        
    Returns:
        List of chunk dictionaries with boundaries
    """
    # Get volume dimensions
    Z, Y, X = volume_shape
    
    # If no chunks specified, use reasonable defaults
    if output_chunks is None:
        # Default chunk sizes (256 is a common size for zarr chunks)
        z_chunk, y_chunk, x_chunk = 256, 256, 256
    else:
        # Use the provided chunks (these should be the spatial dimensions only)
        z_chunk, y_chunk, x_chunk = output_chunks
    
    # Process one chunk at a time for maximum memory efficiency
    chunks = []
    for z_start in range(0, Z, z_chunk):
        for y_start in range(0, Y, y_chunk):
            for x_start in range(0, X, x_chunk):
                z_end = min(z_start + z_chunk, Z)
                y_end = min(y_start + y_chunk, Y)
                x_end = min(x_start + x_chunk, X)
                
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
        ram_budget_gb: float = 100.0):  # Available RAM budget in GB
    """
    Optimized merging of partial inference results with Gaussian blending using vectorized 
    single-pass processing and efficient batch reading.

    Args:
        parent_dir: Directory containing logits_part_X.zarr and coordinates_part_X.zarr.
        output_path: Path for the final merged Zarr store.
        sigma_scale: Determines the sigma for the Gaussian map (patch_size / sigma_scale).
        chunk_size: Spatial chunk size (Z, Y, X) for output Zarr stores.
                    If None, will use patch_size as a starting point.
        num_workers: Number of worker processes to use.
                     If None, defaults to CPU_COUNT // 2.
        compression_level: Zarr compression level (0-9, 0=none)
        verbose: Print progress messages.
        ram_budget_gb: Available RAM budget in GB for batch processing.
    """
    # Disable Blosc threading to avoid deadlocks when used with multiprocessing
    numcodecs.blosc.use_threads = False
    
    # Calculate memory budget
    memory_budget = calculate_memory_budget(ram_budget_gb)
    if verbose:
        print(f"Memory budget: {ram_budget_gb}GB total, {memory_budget['usable_memory_bytes'] / 1024**3:.1f}GB usable")
    
    # Configure process pool size - use half of available CPUs for memory efficiency
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() // 2)
    
    print(f"Using {num_workers} worker processes for optimal memory/performance balance")
        
    # --- 1. Discover Parts ---
    part_files = {}
    part_pattern = re.compile(r"(logits|coordinates)_part_(\d+)\.zarr")
    print(f"Scanning for parts in: {parent_dir}")
    
    # Use fsspec for listing files (works with S3 and local paths)
    if parent_dir.startswith('s3://'):
        fs = fsspec.filesystem('s3', anon=False)
        full_paths = fs.ls(parent_dir)
        
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
    print(f"Found parts: {part_ids}")

    # Validate that all parts have both files
    for part_id in part_ids:
        if 'logits' not in part_files[part_id] or 'coordinates' not in part_files[part_id]:
            raise FileNotFoundError(f"Part {part_id} is missing logits or coordinates Zarr.")

    # --- 2. Read Metadata ---
    first_part_id = part_ids[0]
    print(f"Reading metadata from part {first_part_id}...")
    part0_logits_path = part_files[first_part_id]['logits']
    try:
        part0_logits_store = open_zarr(part0_logits_path, mode='r', storage_options={'anon': False} if part0_logits_path.startswith('s3://') else None)

        input_chunks = part0_logits_store.chunks
        print(f"Input zarr chunk size: {input_chunks}")

        try:
            meta_attrs = part0_logits_store.attrs
            patch_size = tuple(meta_attrs['patch_size'])
            original_volume_shape = tuple(meta_attrs['original_volume_shape'])
            num_classes = part0_logits_store.shape[1]
        except (KeyError, AttributeError):
            zattrs_path = os.path.join(part0_logits_path, '.zattrs')
            with fsspec.open(zattrs_path, 'r') as f:
                meta_attrs = json.load(f)
                
            patch_size = tuple(meta_attrs['patch_size'])  
            original_volume_shape = tuple(meta_attrs['original_volume_shape'])
            num_classes = part0_logits_store.shape[1]
    except Exception as e:
        print(f"Warning: Error reading metadata, attempting to infer: {e}")
        part0_coords_path = part_files[first_part_id]['coordinates']
        coords_store = open_zarr(part0_coords_path, mode='r', storage_options={'anon': False} if part0_coords_path.startswith('s3://') else None)
        first_patch_shape = part0_logits_store[0].shape
        num_classes = first_patch_shape[0]
        patch_size = first_patch_shape[1:]
        
        coords_data = coords_store[:]
        min_coords = np.min(coords_data, axis=0)
        max_coords = np.max(coords_data, axis=0)
        estimated_shape = tuple((max_coords + np.array(patch_size) - min_coords).astype(int))
        
        original_volume_shape = estimated_shape
        print("WARNING: No .zattrs file found. Using estimated volume shape from coordinates.")
        
    print(f"  Patch Size: {patch_size}")
    print(f"  Num Classes: {num_classes}")
    print(f"  Original Volume Shape (Z,Y,X): {original_volume_shape}")

    # --- 3. Prepare Output Store (Single Pass - No Weights File) ---
    output_shape = (num_classes, *original_volume_shape)

    # Use patch_size directly as the chunk size if not specified
    if chunk_size is None or any(c == 0 for c in (chunk_size if chunk_size else [0, 0, 0])):
        output_chunks = (
            1,  # One class at a time
            patch_size[0],  # Z - use exact patch size
            patch_size[1],  # Y - use exact patch size
            patch_size[2]   # X - use exact patch size
        )
        if verbose:
            print(f"  Using chunk_size {output_chunks[1:]} based directly on patch_size")
    else:
        output_chunks = (1, *chunk_size)
        if verbose:
            print(f"  Using specified chunk_size {chunk_size}")

    # Setup compression
    if compression_level > 0:
        compressor = numcodecs.Blosc(
            cname='zstd',
            clevel=compression_level,
            shuffle=numcodecs.blosc.SHUFFLE
        )
    else:
        compressor = None

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
        dtype=np.float32,
        fill_value=0,
        write_empty_chunks=False
    )

    # --- 4. Generate Gaussian Map ---
    gaussian_map = generate_gaussian_map(patch_size, sigma_scale=sigma_scale, verbose=verbose)
    # Pre-convert to numpy for efficient processing (cache this conversion)
    gaussian_map_np = gaussian_map[0]  # Shape (pZ, pY, pX)

    # --- 5. Calculate Processing Chunks ---
    chunks = calculate_chunks(
        original_volume_shape,
        output_chunks=output_chunks[1:]  # Skip the class dimension from output_chunks
    )
    
    print(f"Divided volume into {len(chunks)} chunks for parallel processing")
    
    # --- 6. Single-Pass Processing with Optimization ---
    print("\n--- Single-Pass Accumulation and Normalization ---")
    
    # Create a partial function with fixed arguments for optimized processing
    process_chunk_partial = partial(
        process_chunk,
        part_files=part_files,
        part_ids=part_ids,
        output_path=output_path,
        gaussian_map_np=gaussian_map_np,
        patch_size=patch_size,
        memory_budget=memory_budget,
        num_workers=num_workers
    )
    
    # Process chunks in parallel with single-pass optimization
    total_patches_processed = 0
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_chunk = {executor.submit(process_chunk_partial, chunk): chunk for chunk in chunks}
        
        # Use as_completed for better progress tracking and early error detection
        from concurrent.futures import as_completed
        for future in tqdm(
            as_completed(future_to_chunk),
            total=len(chunks),
            desc="Processing Chunks: ",
            disable=not verbose
        ):
            try:
                result = future.result()
                total_patches_processed += result['patches_processed']
            except Exception as e:
                print(f"Error processing chunk: {e}")
                raise e
    
    print(f"\nSingle-pass processing complete. Processed {total_patches_processed} patches total.")
    
    # --- 7. Save Metadata ---
    output_zarr = open_zarr(
        path=output_path,
        mode='r+',
        storage_options={'anon': False} if output_path.startswith('s3://') else None,
        verbose=verbose
    )
    if hasattr(output_zarr, 'attrs'):
        output_zarr.attrs['patch_size'] = patch_size
        output_zarr.attrs['original_volume_shape'] = original_volume_shape
        output_zarr.attrs['sigma_scale'] = sigma_scale
        output_zarr.attrs['ram_budget_gb'] = ram_budget_gb
        output_zarr.attrs['optimized_single_pass'] = True
    
    print(f"\n--- Optimized Merging Finished ---")
    print(f"Performance improvements:")
    print(f"  - Single-pass processing (no separate normalization)")
    print(f"  - Vectorized coordinate calculations")
    print(f"  - Memory-efficient batch reading")
    print(f"  - RAM budget utilization: {ram_budget_gb}GB")
    print(f"Final merged output saved to: {output_path}")


# --- Command Line Interface ---
def main():
    """Entry point for the vesuvius.blend command line tool."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description='Optimized merging of partial inference outputs with Gaussian blending.')
    parser.add_argument('parent_dir', type=str,
                        help='Directory containing the partial inference results (logits_part_X.zarr, coordinates_part_X.zarr)')
    parser.add_argument('output_path', type=str,
                        help='Path for the final merged Zarr output file.')
    parser.add_argument('--sigma_scale', type=float, default=8.0,
                        help='Sigma scale for Gaussian map (patch_size / sigma_scale). Default: 8.0')
    parser.add_argument('--chunk_size', type=str, default=None,
                        help='Spatial chunk size (Z,Y,X) for output Zarr. Comma-separated. If not specified, optimized size will be used.')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Number of worker processes. Default: CPU_COUNT // 2')
    parser.add_argument('--compression_level', type=int, default=1, choices=range(10),
                        help='Compression level (0-9, 0=none). Default: 1')
    parser.add_argument('--ram_budget_gb', type=float, default=100.0,
                        help='Available RAM budget in GB for batch processing. Default: 100.0')
    parser.add_argument('--quiet', action='store_true',
                            help='Disable verbose progress messages (tqdm bars still show).')

    args = parser.parse_args()

    # Parse chunk_size if provided
    chunks = None
    if args.chunk_size:
        try:
            chunks = tuple(map(int, args.chunk_size.split(',')))
            if len(chunks) != 3: raise ValueError()
        except ValueError:
            parser.error("Invalid chunk_size format. Expected 3 comma-separated integers (Z,Y,X).")

    # Print optimization info
    if not args.quiet:
        print("=== Optimized Blending with Single-Pass Processing ===")
        print(f"RAM Budget: {args.ram_budget_gb}GB")
        print("=" * 55)

    try:
            merge_inference_outputs(
                parent_dir=args.parent_dir,
                output_path=args.output_path,
                sigma_scale=args.sigma_scale,
                chunk_size=chunks,
                num_workers=args.num_workers,
                compression_level=args.compression_level,
                verbose=not args.quiet,
                ram_budget_gb=args.ram_budget_gb
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
