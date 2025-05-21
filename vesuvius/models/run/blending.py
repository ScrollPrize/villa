import numpy as np
import os
import re
import json
import zarr
import fsspec
import multiprocessing as mp
from tqdm.auto import tqdm
from scipy.ndimage import gaussian_filter
import torch
from functools import partial
import numcodecs
from concurrent.futures import ProcessPoolExecutor
import math
import bisect
from data.utils import open_zarr


# --- Vectorized Patch Accumulation Function ---
def accumulate_patch(dest_logits, dest_weights, logit_patch, weight_patch, iz0, iy0, ix0):
    """
    PyTorch version using index_add_ for GPU acceleration
    
    Parameters same as accumulate_patch, but tensors should be PyTorch tensors
    """
    C = dest_logits.shape[0]
    
    # 1. Build 1-D indices once
    dz, dy, dx = weight_patch.shape
    z, y, x = torch.meshgrid(
        torch.arange(dz, device=dest_logits.device) + iz0,
        torch.arange(dy, device=dest_logits.device) + iy0, 
        torch.arange(dx, device=dest_logits.device) + ix0,
        indexing='ij'
    )
    indices = (z * dest_weights.shape[1] * dest_weights.shape[2] + 
               y * dest_weights.shape[2] + x).reshape(-1)
    
    w_flat = weight_patch.reshape(-1)
    
    # 2. Scatter-add weights
    dest_weights.reshape(-1).index_add_(0, indices, w_flat)
    
    # 3. Scatter-add logits - loop over channels
    lp = logit_patch.reshape(C, -1)
    for c in range(C):
        dest_logits[c].reshape(-1).index_add_(0, indices, lp[c] * w_flat)


def find_intersecting_patches(coords_np, chunk_info, patch_size):
    """
    find patches that intersect with the current chunk using binary search.
    
    Args:
        coords_np: Array of patch coordinates (N, 3) where each row is (z, y, x)
        chunk_info: Dictionary with chunk boundaries {'z_start', 'z_end', 'y_start', 'y_end', 'x_start', 'x_end'}
        patch_size: Size of patches (pZ, pY, pX)
        
    Returns:
        List of indices of patches that intersect with the chunk
    """
    z_start, z_end = chunk_info['z_start'], chunk_info['z_end']
    y_start, y_end = chunk_info['y_start'], chunk_info['y_end']
    x_start, x_end = chunk_info['x_start'], chunk_info['x_end']
    
    pZ, pY, pX = patch_size
    
    # Create a sorted array of (z_min, original_index) pairs
    patches_by_z = [(coords_np[i, 0], i) for i in range(coords_np.shape[0])]
    patches_by_z.sort()  # Sort by z_min
    
    # Use binary search to find the first patch that might intersect (z_min < z_end)
    # bisect.bisect_left returns the insertion point for z_end in the sorted array
    start_idx = 0
    if patches_by_z:  # Check if array is not empty
        # Find patches whose z_min is less than z_end (might intersect)
        start_idx = bisect.bisect_left(patches_by_z, (z_end, 0), key=lambda x: x[0])
    
    # Filter patches that actually intersect on all three axes
    intersecting_indices = []
    # Only check patches from beginning up to start_idx
    for i in range(start_idx):
        z_min, original_idx = patches_by_z[i]
        z, y, x = coords_np[original_idx]
        
        # Calculate z_max, y_max, x_max for this patch
        z_max = z + pZ
        y_max = y + pY
        x_max = x + pX
        
        # Check if the patch intersects with the chunk on all axes
        if (z_max > z_start and z < z_end and
            y_max > y_start and y < y_end and
            x_max > x_start and x < x_end):
            intersecting_indices.append(original_idx)
    
    return intersecting_indices

# --- Gaussian Map Generation ---
def generate_gaussian_map(patch_size: tuple, sigma_scale: float = 8.0, dtype=torch.float32) -> torch.Tensor:
    """
    Generates a Gaussian importance map for a given patch size.
    Weights decay from the center towards the edges.
    Shape: (1, pZ, pY, pX) for easy broadcasting.
    """
    pZ, pY, pX = patch_size
    tmp = torch.zeros(patch_size, dtype=dtype)
    center_coords = [i // 2 for i in patch_size]
    sigmas = [i / sigma_scale for i in patch_size]

    tmp[tuple(center_coords)] = 1

    tmp_np = tmp.cpu().numpy()
    gaussian_map_np = gaussian_filter(tmp_np, sigmas, 0, mode='constant', cval=0)
    gaussian_map = torch.from_numpy(gaussian_map_np)
    # Safeguard against division by zero
    gaussian_map /= max(gaussian_map.max().item(), 1e-12)
    gaussian_map = gaussian_map.reshape(1, pZ, pY, pX)
    gaussian_map = torch.clamp(gaussian_map, min=0)
    
    print(
        f"Generated Gaussian map with shape {gaussian_map.shape}, min: {gaussian_map.min().item():.4f}, max: {gaussian_map.max().item():.4f}")
    return gaussian_map


# --- Chunk Processing Worker Function ---
def process_chunk(chunk_info, parent_dir, output_path, weights_path, gaussian_map, 
                patch_size, part_files, device=None, fused_normalize=False, epsilon=1e-8):
    """
    Process a single chunk of the volume, handling all patches that intersect with this chunk.
    
    Args:
        chunk_info: Dictionary with chunk boundaries {'z_start', 'z_end', 'y_start', 'y_end', 'x_start', 'x_end'}
        parent_dir: Directory containing part files
        output_path: Path to output zarr
        weights_path: Path to weights zarr
        gaussian_map: Pre-computed Gaussian map
        patch_size: Size of patches (pZ, pY, pX)
        part_files: Dictionary of part files
        device: PyTorch device to use (default: cuda if available, else cpu)
        fused_normalize: Whether to normalize in-place (default: False)
        epsilon: Small value to avoid division by zero (default: 1e-8)
    """
    # Extract chunk boundaries
    z_start, z_end = chunk_info['z_start'], chunk_info['z_end']
    y_start, y_end = chunk_info['y_start'], chunk_info['y_end']
    x_start, x_end = chunk_info['x_start'], chunk_info['x_end']
    
    pZ, pY, pX = patch_size
    
    # Setup PyTorch device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    # Convert gaussian map to torch tensor on the correct device
    gaussian_map_spatial = gaussian_map[0].to(device)  # Shape (pZ, pY, pX)
    
    # Open zarr stores directly
    output_store = open_zarr(output_path, mode='r+', storage_options={'anon': False} if output_path.startswith('s3://') else None)
    weights_store = open_zarr(weights_path, mode='r+', storage_options={'anon': False} if weights_path.startswith('s3://') else None)
    
    # Create local accumulators for this chunk - initialize with zeros
    # Shape: (C, chunk_z, chunk_y, chunk_x)
    num_classes = output_store.shape[0]
    chunk_shape = (num_classes, z_end - z_start, y_end - y_start, x_end - x_start)
    weights_shape = (z_end - z_start, y_end - y_start, x_end - x_start)
    
    # Initialize accumulators with PyTorch
    chunk_logits = torch.zeros(chunk_shape, dtype=torch.float32, device=device)
    chunk_weights = torch.zeros(weights_shape, dtype=torch.float32, device=device)
    
    # Track which patches intersect with this chunk
    patches_processed = 0
    
    # Process each part file sequentially
    for part_id in part_files:
        logits_path = part_files[part_id]['logits']
        coords_path = part_files[part_id]['coordinates']
        
        # Open zarr stores directly
        coords_store = open_zarr(coords_path, mode='r', storage_options={'anon': False} if coords_path.startswith('s3://') else None)
        logits_store = open_zarr(logits_path, mode='r', storage_options={'anon': False} if logits_path.startswith('s3://') else None)
        
        # Read all coordinates for this part
        coords_np = coords_store[:]
        
        # Efficiently find patches that intersect with this chunk
        intersecting_indices = find_intersecting_patches(coords_np, chunk_info, patch_size)
        
        # Process only the intersecting patches
        for patch_idx in intersecting_indices:
            z, y, x = coords_np[patch_idx].tolist()
                
            # Calculate intersection between patch and chunk
            iz_start = max(z, z_start) - z_start
            iz_end = min(z + pZ, z_end) - z_start
            iy_start = max(y, y_start) - y_start
            iy_end = min(y + pY, y_end) - y_start
            ix_start = max(x, x_start) - x_start
            ix_end = min(x + pX, x_end) - x_start
            
            # Patch internal coordinates (for reading from logits)
            pz_start = max(z_start - z, 0)
            pz_end = pZ - max(z + pZ - z_end, 0)
            py_start = max(y_start - y, 0)
            py_end = pY - max(y + pY - y_end, 0)
            px_start = max(x_start - x, 0)
            px_end = pX - max(x + pX - x_end, 0)
            
            # Read patch logits (only the portion that intersects with our tile)
            patch_slice = (
                slice(None),  # All classes
                slice(pz_start, pz_end),
                slice(py_start, py_end),
                slice(px_start, px_end)
            )
            
            # Read the specific portion of the logits for this patch
            logit_patch = logits_store[patch_idx][patch_slice]
            
            # Get corresponding weights
            weight_patch = gaussian_map_spatial[
                slice(pz_start, pz_end),
                slice(py_start, py_end),
                slice(px_start, px_end)
            ]
            # Convert numpy arrays to torch tensors
            logit_patch_torch = torch.tensor(logit_patch, device=device)
            
            # Get views of destination arrays
            dest_logits = chunk_logits[
                :,  # All classes
                iz_start:iz_end,
                iy_start:iy_end,
                ix_start:ix_end
            ]
            
            dest_weights = chunk_weights[
                iz_start:iz_end,
                iy_start:iy_end,
                ix_start:ix_end
            ]
            
            # Use vectorized accumulation
            accumulate_patch(dest_logits, dest_weights, logit_patch_torch, weight_patch, 0, 0, 0)
            
            patches_processed += 1
    
    # Write accumulated data back to main arrays
    if patches_processed > 0:
        output_slice = (
            slice(None),  # All classes
            slice(z_start, z_end),
            slice(y_start, y_end),
            slice(x_start, x_end)
        )
        
        weight_slice = (
            slice(z_start, z_end),
            slice(y_start, y_end),
            slice(x_start, x_end)
        )
        
        # If doing fused normalization, normalize before writing
        if fused_normalize:
            # Create mask where weights > 0
            weight_mask = chunk_weights > 0
            # Normalize in-place where weights > 0
            for c in range(num_classes):
                chunk_logits[c][weight_mask] /= (chunk_weights[weight_mask] + epsilon)
            # Zero out where weights = 0
            for c in range(num_classes):
                chunk_logits[c][~weight_mask] = 0
            # Convert to numpy and write
            output_store[output_slice] = chunk_logits.cpu().numpy()
            # No need to write weights if we've already normalized
        else:
            # Convert to numpy and write
            output_store[output_slice] = chunk_logits.cpu().numpy()
            weights_store[weight_slice] = chunk_weights.cpu().numpy()
    
    return {
        'chunk': chunk_info,
        'patches_processed': patches_processed
    }


# --- Normalization Worker Function ---
def normalize_chunk(chunk_info, output_path, weights_path, epsilon=1e-8):
    """
    Normalize a single chunk by dividing accumulated logits by weights.
    
    Args:
        chunk_info: Dictionary with chunk boundaries {'z_start', 'z_end', 'y_start', 'y_end', 'x_start', 'x_end'}
        output_path: Path to output zarr
        weights_path: Path to weights zarr
        epsilon: Small value to avoid division by zero
    """
    # Extract chunk boundaries
    z_start, z_end = chunk_info['z_start'], chunk_info['z_end']
    y_start, y_end = chunk_info['y_start'], chunk_info['y_end']
    x_start, x_end = chunk_info['x_start'], chunk_info['x_end']
    
    # Open zarr stores directly
    output_store = open_zarr(output_path, mode='r+', storage_options={'anon': False} if output_path.startswith('s3://') else None)
    weights_store = open_zarr(weights_path, mode='r', storage_options={'anon': False} if weights_path.startswith('s3://') else None)
    
    # Define slices for reading data (exact patch size)
    output_slice = (
        slice(None),  # All classes
        slice(z_start, z_end),
        slice(y_start, y_end),
        slice(x_start, x_end)
    )
    
    weight_slice = (
        slice(z_start, z_end),
        slice(y_start, y_end),
        slice(x_start, x_end)
    )
    
    # Read data (chunk-sized)
    logits = output_store[output_slice]
    weights = weights_store[weight_slice]
    
    # Use in-place division to save memory (no duplication)
    # Initialize result array
    normalized = np.zeros_like(logits)
    
    # Efficient, consistent division: divide only where weights > 0
    np.divide(logits, weights[np.newaxis, :, :, :] + epsilon, 
              out=normalized, where=weights[np.newaxis, :, :, :] > 0)
    
    # Write normalized data back
    output_store[output_slice] = normalized
    
    return {
        'chunk': chunk_info,
        'normalized_voxels': np.prod(normalized.shape)
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
        weight_accumulator_path: str = None,  # Optional: Path for weights, default is temp
        sigma_scale: float = 8.0,
        chunk_size: tuple = None,  # Spatial chunk size (Z, Y, X) for output
        num_workers: int = None,  # Number of worker processes to use
        compression_level: int = 1,  # Compression level (0-9, 0=none)
        delete_weights: bool = True,  # Delete weight accumulator after merge
        device: str = None,  # PyTorch device to use (cuda if available, else cpu)
        fused_normalize: bool = True,  # Normalize during accumulation to avoid second pass
        epsilon: float = 1e-8,  # Small value to avoid division by zero
        verbose: bool = True):
    """
    Merges partial inference results with Gaussian blending using parallel processing.
    Uses fsspec.get_mapper for consistent zarr access across file systems and protocols.

    Args:
        parent_dir: Directory containing logits_part_X.zarr and coordinates_part_X.zarr.
        output_path: Path for the final merged Zarr store.
        weight_accumulator_path: Path for the temporary weight accumulator Zarr.
                                  If None, defaults to output_path + "_weights.zarr".
        sigma_scale: Determines the sigma for the Gaussian map (patch_size / sigma_scale).
        chunk_size: Spatial chunk size (Z, Y, X) for output Zarr stores.
                    If None, will use patch_size as a starting point.
        num_workers: Number of worker processes to use.
                     If None, defaults to CPU_COUNT - 1.
        compression_level: Zarr compression level (0-9, 0=none)
        delete_weights: Whether to delete the weight accumulator Zarr after completion.
        device: PyTorch device to use (default: 'cuda' if available, else 'cpu').
        fused_normalize: Whether to normalize during accumulation (default: True).
        epsilon: Small value to avoid division by zero (default: 1e-8).
        verbose: Print progress messages.
    """
    # Disable Blosc threading to avoid deadlocks when used with multiprocessing
    numcodecs.blosc.use_threads = False
    if weight_accumulator_path is None:
        base, _ = os.path.splitext(output_path)
        weight_accumulator_path = f"{base}_weights.zarr"
    
    # Configure process pool size - use half of available CPUs for memory efficiency
    if num_workers is None:
        # Use half of CPU count (rounded up) to balance performance and memory usage
        num_workers = max(1, mp.cpu_count() // 2)
    
    print(f"Using {num_workers} worker processes (half of CPU count for memory efficiency)")
        
    # --- 1. Discover Parts ---
    part_files = {}
    part_pattern = re.compile(r"(logits|coordinates)_part_(\d+)\.zarr")
    print(f"Scanning for parts in: {parent_dir}")
    
    # Use fsspec for listing files (works with S3 and local paths)
    if parent_dir.startswith('s3://'):
        fs = fsspec.filesystem('s3', anon=False)
        # List directory to get all entries
        full_paths = fs.ls(parent_dir)
        
        # For S3, strip the bucket name and path prefix to get just the directory name
        # Each entry looks like: 'bucket/path/to/parent_dir/logits_part_0.zarr'
        file_list = []
        for path in full_paths:
            # Remove the s3://bucket/ prefix 
            path_parts = path.split('/')
            # Get the last part which is the actual directory name
            filename = path_parts[-1]
            file_list.append(filename)
            
        print(f"DEBUG: Found files in S3: {file_list}")
    else:
        # Use os.listdir for local paths
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

    # --- 2. Read Metadata (from first available part) ---
    first_part_id = part_ids[0]  # Use the first available part_id 
    print(f"Reading metadata from part {first_part_id}...")
    part0_logits_path = part_files[first_part_id]['logits']
    try:
        # Use our helper function to open zarr store
        part0_logits_store = open_zarr(part0_logits_path, mode='r', storage_options={'anon': False} if part0_logits_path.startswith('s3://') else None)

        # Read input zarr store chunk size
        input_chunks = part0_logits_store.chunks
        print(f"Input zarr chunk size: {input_chunks}")

        # Read .zattrs using fsspec
        try:
            # Use the part0_logits_store's .attrs directly if available
            meta_attrs = part0_logits_store.attrs
            patch_size = tuple(meta_attrs['patch_size'])  # Already a list in the file
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
        # Try to infer from the array shape if .zattrs is missing
        print(f"Warning: Error reading metadata, attempting to infer: {e}")
        part0_coords_path = part_files[first_part_id]['coordinates']
        coords_store = open_zarr(part0_coords_path, mode='r', storage_options={'anon': False} if part0_coords_path.startswith('s3://') else None)
        # First patch's logits shape should be (C, pZ, pY, pX)
        first_patch_shape = part0_logits_store[0].shape
        num_classes = first_patch_shape[0]
        patch_size = first_patch_shape[1:]
        
        # Get first and last patch centers to estimate volume size
        coords_data = coords_store[:]
        min_coords = np.min(coords_data, axis=0)
        max_coords = np.max(coords_data, axis=0)
        estimated_shape = tuple((max_coords + np.array(patch_size) - min_coords).astype(int))
        
        original_volume_shape = estimated_shape
        print("WARNING: No .zattrs file found. Using estimated volume shape from coordinates.")
        
    print(f"  Patch Size: {patch_size}")
    print(f"  Num Classes: {num_classes}")
    print(f"  Original Volume Shape (Z,Y,X): {original_volume_shape}")

    # --- 3. Prepare Output Stores ---
    output_shape = (num_classes, *original_volume_shape)  # (C, D, H, W)
    weights_shape = original_volume_shape  # (D, H, W)

    # Use patch_size directly as the chunk size if not specified
    if chunk_size is None or any(c == 0 for c in (chunk_size if chunk_size else [0, 0, 0])):
        # Default to patch size exactly
        output_chunks = (
            1,  # One class at a time
            patch_size[0],  # Z - use exact patch size
            patch_size[1],  # Y - use exact patch size
            patch_size[2]   # X - use exact patch size
        )
        weights_chunks = output_chunks[1:]
        if verbose:
            print(f"  Using chunk_size {output_chunks[1:]} based directly on patch_size")
    else:
        output_chunks = (1, *chunk_size)  # One class at a time, user-specified spatial chunks
        weights_chunks = chunk_size
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
    
    # Our open_zarr helper function handles directory creation and authentication
        
    # Use our helper function to create zarr store
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
        write_empty_chunks=False  # Skip empty chunks for memory efficiency
    )
    
    print(f"Creating weight accumulator store: {weight_accumulator_path}")
    print(f"  Shape: {weights_shape}, Chunks: {weights_chunks}")
    
    # Our open_zarr helper function handles directory creation and authentication
        
    open_zarr(
        path=weight_accumulator_path,
        mode='w',
        storage_options={'anon': False} if weight_accumulator_path.startswith('s3://') else None,
        verbose=verbose,
        shape=weights_shape,
        chunks=weights_chunks,
        compressor=compressor,
        dtype=np.float32,
        fill_value=0,
        write_empty_chunks=False  # Skip empty chunks for memory efficiency
    )

    # --- 4. Generate Gaussian Map ---
    gaussian_map = generate_gaussian_map(patch_size, sigma_scale=sigma_scale)
    # Make sure it's on CPU and convert to numpy
    gaussian_map = gaussian_map.cpu()

    # --- 5. Calculate Processing Chunks ---
    chunks = calculate_chunks(
        original_volume_shape,
        output_chunks=output_chunks[1:]  # Skip the class dimension from output_chunks
    )
    
    print(f"Divided volume into {len(chunks)} chunks for parallel processing")
    
    # --- 6. Process Chunks in Parallel ---
    print("\n--- Accumulating Weighted Patches ---")
    
    # Create a partial function with fixed arguments
    process_chunk_partial = partial(
        process_chunk,
        parent_dir=parent_dir,
        output_path=output_path,
        weights_path=weight_accumulator_path,
        gaussian_map=gaussian_map,
        patch_size=patch_size,
        part_files=part_files,
        device=device,
        fused_normalize=fused_normalize,
        epsilon=epsilon
    )
    
    # Process chunks in parallel
    total_patches_processed = 0
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_chunk = {executor.submit(process_chunk_partial, chunk): chunk for chunk in chunks}
        
        # Use as_completed for better progress tracking and early error detection
        from concurrent.futures import as_completed
        for future in tqdm(
            as_completed(future_to_chunk),
            total=len(chunks),
            desc="Processing Chunks",
            disable=not verbose
        ):
            try:
                result = future.result()
                total_patches_processed += result['patches_processed']
            except Exception as e:
                print(f"Error processing chunk: {e}")
                raise e
    
    print(f"\nAccumulation complete. Processed {total_patches_processed} patches total.")
    
    # --- 7. Normalize in Parallel ---
    print("\n--- Normalizing Output ---")
    
    # Create a partial function with fixed arguments
    normalize_chunk_partial = partial(
        normalize_chunk,
        output_path=output_path,
        weights_path=weight_accumulator_path
    )
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(normalize_chunk_partial, chunk): chunk for chunk in chunks}
        
        # Use as_completed for better progress tracking and early error detection
        from concurrent.futures import as_completed
        for future in tqdm(
            as_completed(futures),
            total=len(chunks),
            desc="Normalizing Chunks",
            disable=not verbose
        ):
            try:
                result = future.result()  # Check for exceptions
            except Exception as e:
                print(f"Error normalizing chunk: {e}")
                raise e
    
    print("\nNormalization complete.")
    
    # --- 8. Save Metadata ---
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
    
    # --- 9. Cleanup ---
    if delete_weights:
        print(f"Deleting weight accumulator: {weight_accumulator_path}")
        try:
            import shutil
            if os.path.exists(weight_accumulator_path):
                shutil.rmtree(weight_accumulator_path)
                print(f"Successfully deleted weight accumulator")
        except Exception as e:
            print(f"Warning: Failed to delete weight accumulator: {e}")
            print(f"You may need to delete it manually: {weight_accumulator_path}")

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
    parser.add_argument('--weights_path', type=str, default=None,
                        help='Optional path for the temporary weight accumulator Zarr. Defaults to <output_path>_weights.zarr')
    parser.add_argument('--sigma_scale', type=float, default=8.0,
                        help='Sigma scale for Gaussian map (patch_size / sigma_scale). Default: 8.0')
    parser.add_argument('--chunk_size', type=str, default=None,
                        help='Spatial chunk size (Z,Y,X) for output Zarr. Comma-separated. If not specified, optimized size will be used.')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Number of worker processes. Default: CPU_COUNT - 1')
    parser.add_argument('--compression_level', type=int, default=1, choices=range(10),
                        help='Compression level (0-9, 0=none). Default: 1')
    parser.add_argument('--keep_weights', action='store_true',
                        help='Do not delete the weight accumulator Zarr after merging.')
    parser.add_argument('--device', type=str, default=None,
                        help='PyTorch device to use (e.g. "cuda:0", "cpu"). Default: cuda if available, else cpu.')
    parser.add_argument('--no_fused_normalize', action='store_true',
                        help='Disable fused normalization during accumulation (use separate pass).')
    parser.add_argument('--epsilon', type=float, default=1e-8,
                        help='Small value to avoid division by zero. Default: 1e-8')
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

    try:
        merge_inference_outputs(
            parent_dir=args.parent_dir,
            output_path=args.output_path,
            weight_accumulator_path=args.weights_path,
            sigma_scale=args.sigma_scale,
            chunk_size=chunks,
            num_workers=args.num_workers,
            compression_level=args.compression_level,
            delete_weights=not args.keep_weights,
            device=args.device,
            fused_normalize=not args.no_fused_normalize,
            epsilon=args.epsilon,
            verbose=not args.quiet
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
