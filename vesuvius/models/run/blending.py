import tensorstore as ts
import numpy as np
import asyncio
import os
import re
import json
import threading
import concurrent.futures
import multiprocessing
import queue
from tqdm.auto import tqdm
from scipy.ndimage import gaussian_filter  # For map generation alternative
import torch


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
    gaussian_map /= gaussian_map.max()
    gaussian_map = gaussian_map.reshape(1, pZ, pY, pX)
    gaussian_map = torch.clamp(gaussian_map, min=0)

    print(
        f"Generated Gaussian map with shape {gaussian_map.shape}, min: {gaussian_map.min().item():.4f}, max: {gaussian_map.max().item():.4f}")
    return gaussian_map


# --- Spatial Patch Sorting and Grouping ---
def group_patches_for_parallel_processing(coords, patch_size, num_workers, volume_shape):
    """
    Groups patches into processing batches that guarantee no spatial overlap.
    
    Args:
        coords: Array of patch coordinates, shape (N, 3) with (z, y, x) coordinates
        patch_size: Tuple of (pZ, pY, pX) specifying patch dimensions
        num_workers: Number of parallel workers to use
        volume_shape: Tuple of (Z, Y, X) original volume dimensions
        
    Returns:
        List of lists, where each sublist contains patch indices that can be processed in parallel
    """
    pZ, pY, pX = patch_size
    Z, Y, X = volume_shape
    
    # For large volumes, divide space into non-overlapping regions
    # We'll create a spatial grid where each cell is at least 'patch_size' apart
    # This ensures patches in the same cell never overlap
    
    # Calculate number of cells needed in each dimension
    grid_z = max(1, num_workers)  # At least 1 cell per dimension
    grid_y = max(1, num_workers)
    grid_x = max(1, num_workers)
    
    # Calculate cell size in each dimension
    cell_z = max(pZ, Z // grid_z)
    cell_y = max(pY, Y // grid_y)
    cell_x = max(pX, X // grid_x)
    
    # Group patches by their spatial grid cell
    grid_cells = {}  # Maps (grid_z, grid_y, grid_x) to list of patch indices
    
    for i, (z, y, x) in enumerate(coords):
        # Calculate which grid cell this patch belongs to
        gz = min(z // cell_z, grid_z - 1)
        gy = min(y // cell_y, grid_y - 1)
        gx = min(x // cell_x, grid_x - 1)
        
        # Define a grid cell ID
        cell_id = (gz, gy, gx)
        
        if cell_id not in grid_cells:
            grid_cells[cell_id] = []
        
        grid_cells[cell_id].append(i)
    
    # Group cells into non-overlapping sets
    # We can process all cells with the same (gz % 2, gy % 2, gx % 2) in parallel
    parallel_groups = {}
    
    for (gz, gy, gx), indices in grid_cells.items():
        group_id = (gz % 2, gy % 2, gx % 2)
        if group_id not in parallel_groups:
            parallel_groups[group_id] = []
        
        parallel_groups[group_id].extend(indices)
    
    # Sort parallel_groups by size (descending) to balance workload
    sorted_groups = sorted(parallel_groups.values(), key=len, reverse=True)
    
    return sorted_groups


# --- Parallel Patch Processing ---
async def process_patch_batch(
    patch_indices, 
    coords, 
    patch_size,
    logits_store, 
    final_store, 
    weights_store, 
    gaussian_map, 
    gaussian_map_spatial,
    verbose=False
):
    """
    Process a batch of non-overlapping patches in parallel.
    
    All provided patch_indices must be guaranteed to not overlap spatially.
    """
    pZ, pY, pX = patch_size
    futures = []
    
    for patch_idx in patch_indices:
        z, y, x = coords[patch_idx].tolist()  # Convert tensor values to Python integers

        # Define slices for this patch
        output_slice = (
            slice(None),  # All classes
            slice(z, z + pZ),
            slice(y, y + pY),
            slice(x, x + pX)
        )
        weight_slice = (
            slice(z, z + pZ),
            slice(y, y + pY),
            slice(x, x + pX)
        )

        # Read logit patch
        read_future = logits_store[patch_idx].read()
        futures.append((patch_idx, read_future, output_slice, weight_slice))
    
    # Process results as they complete
    tasks = []
    
    for patch_idx, read_future, output_slice, weight_slice in futures:
        # Read the current values from the output stores
        logit_patch_np = await read_future
        current_logits_future = final_store[output_slice].read()
        current_weights_future = weights_store[weight_slice].read()
        
        # Wait for both reads to complete
        current_logits_np, current_weights_np = await asyncio.gather(
            current_logits_future, current_weights_future
        )
        
        # Convert to tensors and process
        logit_patch = torch.from_numpy(logit_patch_np)
        current_logits = torch.from_numpy(current_logits_np)
        current_weights = torch.from_numpy(current_weights_np)
        
        # Apply Gaussian weight map
        weighted_patch = logit_patch * gaussian_map  # Broadcasting
        
        # Add to existing values
        updated_logits = current_logits + weighted_patch
        updated_weights = current_weights + gaussian_map_spatial
        
        # Convert back to numpy for TensorStore write
        updated_logits_np = updated_logits.numpy()
        updated_weights_np = updated_weights.numpy()
        
        # Write back - gather these for concurrent writes
        write_logit_future = final_store[output_slice].write(updated_logits_np)
        write_weight_future = weights_store[weight_slice].write(updated_weights_np)
        
        # Add the write tasks to our list
        tasks.append(asyncio.gather(write_logit_future, write_weight_future))
    
    # Wait for all writes to complete
    await asyncio.gather(*tasks)
    return len(futures)  # Return number of patches processed


# --- Parallel Chunk Normalization ---
async def normalize_chunk_parallel(
    chunk_slice, 
    final_store, 
    weights_store, 
    epsilon=1e-8
):
    """Process a single chunk for normalization."""
    # Read the chunk data
    weight_chunk_slice = chunk_slice[1:]  # Remove class dimension for weights
    
    logit_chunk_future = final_store[chunk_slice].read()
    weight_chunk_future = weights_store[weight_chunk_slice].read()
    
    # Wait for both reads to complete
    logit_chunk_np, weight_chunk_np = await asyncio.gather(logit_chunk_future, weight_chunk_future)
    
    # Convert to PyTorch tensors
    logit_chunk = torch.from_numpy(logit_chunk_np)
    weight_chunk = torch.from_numpy(weight_chunk_np)
    
    # Ensure weights are broadcastable to logits shape (C, cZ, cY, cX)
    # Add class dimension to weights: (cZ, cY, cX) -> (1, cZ, cY, cX)
    weight_chunk_b = weight_chunk.unsqueeze(0)
    
    # Create a mask where weights are significant
    mask = weight_chunk_b > epsilon
    
    # Initialize with zeros
    final_chunk = torch.zeros_like(logit_chunk)
    
    # Only normalize where weights are significant
    final_chunk[mask] = logit_chunk[mask] / (weight_chunk_b[mask] + epsilon)
    
    # Convert back to numpy for TensorStore write
    final_chunk_np = final_chunk.numpy()
    
    # Write the normalized chunk back
    await final_store[chunk_slice].write(final_chunk_np)
    
    # Return the chunk size for progress tracking
    return torch.prod(torch.tensor(final_chunk.shape)).item()


# --- Main Merging Function ---
async def merge_inference_outputs(
        parent_dir: str,
        output_path: str,
        weight_accumulator_path: str = None,  # Optional: Path for weights, default is temp
        sigma_scale: float = 8.0,
        chunk_size: tuple = (128, 128, 128),  # Spatial chunk size (Z, Y, X) for output
        cache_pool_gb: float = 10.0,
        delete_weights: bool = True,  # Delete weight accumulator after merge
        num_workers: int = None,      # Number of parallel workers (defaults to CPU count)
        verbose: bool = True):
    """
    Merges partial inference results with Gaussian blending.

    Args:
        parent_dir: Directory containing logits_part_X.zarr and coordinates_part_X.zarr.
        output_path: Path for the final merged Zarr store.
        weight_accumulator_path: Path for the temporary weight accumulator Zarr.
                                  If None, defaults to output_path + "_weights.zarr".
        sigma_scale: Determines the sigma for the Gaussian map (patch_size / sigma_scale).
        chunk_size: Spatial chunk size (Z, Y, X) for output Zarr stores.
        cache_pool_gb: TensorStore cache pool size in GiB.
        delete_weights: Whether to delete the weight accumulator Zarr after completion.
        num_workers: Number of parallel workers. If None, uses CPU count.
        verbose: Print progress messages.
    """
    # Set default number of workers if not provided
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)  # Leave one CPU for system tasks
    
    print(f"Using {num_workers} parallel workers for processing")
    
    if weight_accumulator_path is None:
        base, _ = os.path.splitext(output_path)
        weight_accumulator_path = f"{base}_weights.zarr"

    # --- 1. Discover Parts ---
    part_files = {}
    part_pattern = re.compile(r"(logits|coordinates)_part_(\d+)\.zarr")
    print(f"Scanning for parts in: {parent_dir}")
    for filename in os.listdir(parent_dir):
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
        # Properly format TensorStore spec with file driver
        part0_logits_store = await ts.open({
            'driver': 'zarr',
            'kvstore': {'driver': 'file', 'path': part0_logits_path}
        })

        # Read .zattrs file directly for metadata
        zattrs_path = os.path.join(part0_logits_path, '.zattrs')
        if os.path.exists(zattrs_path):
            with open(zattrs_path, 'r') as f:
                meta_attrs = json.load(f)

            patch_size = tuple(meta_attrs['patch_size'])  # Already a list in the file
            original_volume_shape = tuple(meta_attrs['original_volume_shape'])  # MUST exist
            num_classes = part0_logits_store.shape[1]  # (N, C, pZ, pY, pX) -> C
        else:
            raise FileNotFoundError(f"Cannot find .zattrs file at {zattrs_path}")
        print(f"  Patch Size: {patch_size}")
        print(f"  Num Classes: {num_classes}")
        print(f"  Original Volume Shape (Z,Y,X): {original_volume_shape}")
    except Exception as e:
        print("\nERROR: Failed to read metadata from part 0 logits attributes.")
        print("Ensure 'patch_size' and 'original_volume_shape' were saved during inference.")
        raise e

    # --- 3. Prepare Output Stores ---
    output_shape = (num_classes, *original_volume_shape)  # (C, D, H, W)
    weights_shape = original_volume_shape  # (D, H, W)

    # Use patch_size as the default chunk_size if not explicitly specified
    # This prevents partial chunk reads during blending
    if chunk_size is None or any(c == 0 for c in chunk_size):
        if verbose: print(f"  Using patch_size {patch_size} as chunk_size for efficient I/O")
        output_chunks = (1, *patch_size)  # Chunk classes separately, spatial chunks from patch
        weights_chunks = patch_size  # Spatial chunks from patch
    else:
        if verbose: print(f"  Using specified chunk_size {chunk_size}")
        output_chunks = (1, *chunk_size)  # Chunk classes separately, user-specified spatial chunks
        weights_chunks = chunk_size  # User-specified spatial chunks

    ts_context = ts.Context({'cache_pool': {'total_bytes_limit': int(cache_pool_gb * 1024 ** 3)}})

    print(f"Creating final output store: {output_path}")
    print(f"  Shape: {output_shape}, Chunks: {output_chunks}")
    final_store = await ts.open(
        ts.Spec({
            'driver': 'zarr',
            'kvstore': {'driver': 'file', 'path': output_path},
            'metadata': {'shape': output_shape, 'chunks': output_chunks, 'dtype': '<f4'},  # Use littleendian float32
            'create': True,
            'delete_existing': True,
        }),
        context=ts_context
    )
    # Initialize with zeros (important for accumulation) - Zarr driver usually does this
    # await final_store.write(np.zeros((1,)*len(output_shape), dtype=np.float32)) # Check if needed

    print(f"Creating weight accumulator store: {weight_accumulator_path}")
    print(f"  Shape: {weights_shape}, Chunks: {weights_chunks}")
    weights_store = await ts.open(
        ts.Spec({
            'driver': 'zarr',
            'kvstore': {'driver': 'file', 'path': weight_accumulator_path},
            'metadata': {'shape': weights_shape, 'chunks': weights_chunks, 'dtype': '<f4'},  # Use littleendian float32
            'create': True,
            'delete_existing': True,
        }),
        context=ts_context
    )
    # await weights_store.write(np.zeros((1,)*len(weights_shape), dtype=np.float32)) # Check if needed

    # --- 4. Generate Gaussian Map ---
    gaussian_map = generate_gaussian_map(patch_size, sigma_scale=sigma_scale)
    # Make sure it's on CPU
    gaussian_map = gaussian_map.cpu()
    # Extract spatial dimensions for weights store
    gaussian_map_spatial = gaussian_map[0]  # Shape (pZ, pY, pX) for weights store

    # --- 5. Process Each Part (Accumulation) using Parallel Processing ---
    print("\n--- Accumulating Weighted Patches ---")
    pZ, pY, pX = patch_size
    total_patches_processed = 0

    for part_id in tqdm(part_ids, desc="Processing Parts"):
        if verbose: print(f"\nProcessing Part {part_id}...")
        logits_path = part_files[part_id]['logits']
        coords_path = part_files[part_id]['coordinates']

        logits_store = await ts.open({
            'driver': 'zarr',
            'kvstore': {'driver': 'file', 'path': logits_path}
        }, context=ts_context)

        coords_store = await ts.open({
            'driver': 'zarr',
            'kvstore': {'driver': 'file', 'path': coords_path}
        }, context=ts_context)

        # Read all coordinates for this part
        coords_np = await coords_store.read()  # Async read directly returns the data
        # Convert to tensor
        coords = torch.from_numpy(coords_np)
        num_patches_in_part = coords.shape[0]
        if verbose: print(f"  Found {num_patches_in_part} patches in part {part_id}.")
        
        # Group patches into non-overlapping batches for parallel processing
        patch_groups = group_patches_for_parallel_processing(
            coords, 
            patch_size, 
            num_workers, 
            original_volume_shape
        )
        
        # Process each group in sequence, but patches within a group in parallel
        with tqdm(total=num_patches_in_part, desc=f"  Patches Part {part_id}", leave=False,
                  disable=not verbose) as patch_pbar:
                
            for group_idx, patch_indices in enumerate(patch_groups):
                # Process this non-overlapping batch in parallel
                patches_processed = await process_patch_batch(
                    patch_indices,
                    coords,
                    patch_size,
                    logits_store,
                    final_store,
                    weights_store,
                    gaussian_map,
                    gaussian_map_spatial,
                    verbose
                )
                
                total_patches_processed += patches_processed
                patch_pbar.update(patches_processed)
                
                # Progress is already shown by the tqdm progress bar

    print(f"\nAccumulation complete. Processed {total_patches_processed} patches total.")

    # --- 6. Normalize in Parallel ---
    print("\n--- Normalizing Output ---")
    # Get output shape and chunks for iteration
    total_voxels = np.prod(output_shape)
    processed_voxels = 0

    # Create a list of chunk indices to iterate over based on output shape and chunk size
    def get_chunk_indices(shape, chunks):
        # For each dimension, calculate how many chunks we need
        chunk_counts = [int(np.ceil(s / c)) for s, c in zip(shape, chunks)]

        # Generate all combinations of chunk indices
        from itertools import product
        chunk_indices = list(product(*[range(count) for count in chunk_counts]))
        return chunk_indices

    chunk_indices = get_chunk_indices(output_shape, output_chunks)
    
    # Set up progress tracking
    chunk_pbar = tqdm(total=len(chunk_indices), desc="Normalizing Chunks", unit="chunk")
    
    # Use semaphore to limit concurrent operations
    semaphore = asyncio.Semaphore(num_workers * 2)  # Allow more normalization tasks than workers
    
    async def process_chunk_with_semaphore(chunk_index):
        """Process a chunk with semaphore to limit concurrency."""
        nonlocal processed_voxels
        
        async with semaphore:
            # Calculate slice for this chunk
            chunk_slice = tuple(
                slice(idx * chunk, min((idx + 1) * chunk, shape_dim))
                for idx, chunk, shape_dim in zip(chunk_index, output_chunks, output_shape)
            )
            
            # Process the chunk
            chunk_voxels = await normalize_chunk_parallel(
                chunk_slice, 
                final_store, 
                weights_store
            )
            
            # Update progress
            processed_voxels += chunk_voxels
            chunk_pbar.update(1)
            
            # Update progress percentage occasionally
            if chunk_pbar.n % 10 == 0:
                percent_complete = min(100, processed_voxels * 100 / total_voxels)
                chunk_pbar.set_postfix({"percent": f"{percent_complete:.1f}%"})
                
            return chunk_voxels
    
    # Create tasks for all chunks
    normalization_tasks = [
        process_chunk_with_semaphore(chunk_index) 
        for chunk_index in chunk_indices
    ]
    
    # Run all tasks
    results = await asyncio.gather(*normalization_tasks)
    
    # Clean up
    chunk_pbar.close()
    
    print("\nNormalization complete.")

    # --- 7. Cleanup ---
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
    import multiprocessing

    parser = argparse.ArgumentParser(description='Merge partial nnUNet inference outputs with Gaussian blending.')
    parser.add_argument('parent_dir', type=str,
                        help='Directory containing the partial inference results (logits_part_X.zarr, coordinates_part_X.zarr)')
    parser.add_argument('output_path', type=str,
                        help='Path for the final merged Zarr output file.')
    parser.add_argument('--weights_path', type=str, default=None,
                        help='Optional path for the temporary weight accumulator Zarr. Defaults to <output_path>_weights.zarr')
    parser.add_argument('--sigma_scale', type=float, default=8.0,
                        help='Sigma scale for Gaussian map (patch_size / sigma_scale). Default: 8.0')
    parser.add_argument('--chunk_size', type=str, default=None,
                        help='Spatial chunk size (Z,Y,X) for output Zarr. Comma-separated. If not specified, patch_size will be used.')
    parser.add_argument('--cache_gb', type=float, default=4.0,
                        help='TensorStore cache pool size in GiB. Default: 4.0')
    parser.add_argument('--num_workers', type=int, default=None,
                        help=f'Number of parallel workers to use. Default: CPU count - 1 ({max(1, multiprocessing.cpu_count() - 1)})')
    parser.add_argument('--keep_weights', action='store_true',
                        help='Do not delete the weight accumulator Zarr after merging.')
    parser.add_argument('--quiet', action='store_true',
                        help='Disable verbose progress messages (tqdm bars still show).')

    args = parser.parse_args()

    # Parse chunk_size if provided, otherwise it will default to None
    chunks = None
    if args.chunk_size:
        try:
            chunks = tuple(map(int, args.chunk_size.split(',')))
            if len(chunks) != 3: raise ValueError()
        except ValueError:
            parser.error("Invalid chunk_size format. Expected 3 comma-separated integers (Z,Y,X).")

    try:
        asyncio.run(merge_inference_outputs(
            parent_dir=args.parent_dir,
            output_path=args.output_path,
            weight_accumulator_path=args.weights_path,
            sigma_scale=args.sigma_scale,
            chunk_size=chunks,
            cache_pool_gb=args.cache_gb,
            delete_weights=not args.keep_weights,
            num_workers=args.num_workers,
            verbose=not args.quiet
        ))
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