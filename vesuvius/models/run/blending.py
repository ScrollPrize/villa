import tensorstore as ts
import numpy as np
import asyncio
import os
import re
import json
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


# --- Main Merging Function ---
async def merge_inference_outputs(
        parent_dir: str,
        output_path: str,
        weight_accumulator_path: str = None,  # Optional: Path for weights, default is temp
        sigma_scale: float = 8.0,
        chunk_size: tuple = (128, 128, 128),  # Spatial chunk size (Z, Y, X) for output
        cache_pool_gb: float = 10.0,
        delete_weights: bool = True,  # Delete weight accumulator after merge
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
        verbose: Print progress messages.
    """
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

    # --- 5. Process Each Part (Accumulation) ---
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

        # Process patches serially for simplicity, could be parallelized with care
        for patch_idx in tqdm(range(num_patches_in_part), desc=f"  Patches Part {part_id}", leave=False,
                              disable=not verbose):
            z, y, x = coords[patch_idx].tolist()  # Convert tensor values to Python integers

            # Define slices (handle boundary conditions implicitly via slicing)
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

            # Read logit patch and immediately convert to tensor
            logit_patch_np = await logits_store[patch_idx].read()  # Reads (C, pZ, pY, pX)
            logit_patch = torch.from_numpy(logit_patch_np)
            
            # Apply Gaussian weight map
            weighted_patch = logit_patch * gaussian_map  # Broadcasting (1, pZ, pY, pX)

            # Atomically add to stores
            # We can't use accumulate parameter directly with TensorStore
            # So we read the current values, add, and write back
            current_logits_np = await final_store[output_slice].read()
            current_weights_np = await weights_store[weight_slice].read()
            
            # Convert to tensors
            current_logits = torch.from_numpy(current_logits_np)
            current_weights = torch.from_numpy(current_weights_np)

            # Add to existing values
            updated_logits = current_logits + weighted_patch
            updated_weights = current_weights + gaussian_map_spatial

            # Convert back to numpy for TensorStore write
            updated_logits_np = updated_logits.numpy()
            updated_weights_np = updated_weights.numpy()

            # Write back
            write_logit_future = final_store[output_slice].write(updated_logits_np)
            write_weight_future = weights_store[weight_slice].write(updated_weights_np)

            # Await writes for this patch before next (simplest flow control)
            await asyncio.gather(write_logit_future, write_weight_future)
            total_patches_processed += 1

    print(f"\nAccumulation complete. Processed {total_patches_processed} patches total.")

    # --- 6. Normalize ---
    print("\n--- Normalizing Output ---")
    # Iterate chunk by chunk over the *output* store dimensions

    # Get output shape and chunks for iteration
    num_processed_voxels = 0
    total_voxels = np.prod(output_shape)

    # Create a list of chunk indices to iterate over
    # Based on output shape and chunk size
    def get_chunk_indices(shape, chunks):
        # For each dimension, calculate how many chunks we need
        chunk_counts = [int(np.ceil(s / c)) for s, c in zip(shape, output_chunks)]

        # Generate all combinations of chunk indices
        from itertools import product
        chunk_indices = list(product(*[range(count) for count in chunk_counts]))
        return chunk_indices

    chunk_indices = get_chunk_indices(output_shape, output_chunks)

    # Wrap the chunk iteration with tqdm
    chunk_iterator = tqdm(chunk_indices,
                          desc="Normalizing Chunks",
                          total=len(chunk_indices),
                          unit="chunk")

    normalize_futures = []
    max_concurrent_normalize = 32  # Limit concurrent reads/writes

    for read_chunk_indices in chunk_iterator:
        # Calculate slices for this chunk using the chunk indices
        # Generate a tuple of slices based on chunk indices and chunk size
        chunk_slice = tuple(
            slice(idx * chunk, min((idx + 1) * chunk, shape_dim))
            for idx, chunk, shape_dim in zip(read_chunk_indices, output_chunks, output_shape)
        )

        # Read summed logits and weights for this chunk
        logit_chunk_future = final_store[chunk_slice].read()
        # Adjust slice for weights store (remove class dimension)
        weight_chunk_slice = chunk_slice[1:]
        weight_chunk_future = weights_store[weight_chunk_slice].read()

        # Process concurrently
        logit_chunk_np, weight_chunk_np = await asyncio.gather(logit_chunk_future, weight_chunk_future)
        
        # Convert to PyTorch tensors
        logit_chunk = torch.from_numpy(logit_chunk_np)
        weight_chunk = torch.from_numpy(weight_chunk_np)

        # Ensure weights are broadcastable to logits shape (C, cZ, cY, cX)
        # Add class dimension to weights: (cZ, cY, cX) -> (1, cZ, cY, cX)
        weight_chunk_b = weight_chunk.unsqueeze(0)

        # Normalize, handling division by zero
        # Add a small epsilon to weights to avoid division by zero errors
        epsilon = 1e-8
        # Create a mask where weights are significant
        mask = weight_chunk_b > epsilon
        
        # Initialize with zeros 
        final_chunk = torch.zeros_like(logit_chunk)
        
        # Only normalize where weights are significant
        final_chunk[mask] = logit_chunk[mask] / (weight_chunk_b[mask] + epsilon)
        
        # Convert back to numpy for TensorStore write 
        final_chunk_np = final_chunk.numpy()

        # Write the normalized chunk back (overwrite)
        # Use fire-and-forget writes and manage concurrency
        write_future = final_store[chunk_slice].write(final_chunk_np)
        normalize_futures.append(write_future)

        # --- Concurrency Management ---
        if len(normalize_futures) >= max_concurrent_normalize:
            # Wait for the oldest futures to complete
            num_to_wait = len(normalize_futures) - max_concurrent_normalize // 2
            # print(f"Waiting for {num_to_wait} normalization futures...")
            await asyncio.gather(*normalize_futures[:num_to_wait])
            normalize_futures = normalize_futures[num_to_wait:]
            # print("Done waiting.")

        # Update progress tracking (approximate)
        num_processed_voxels += torch.prod(torch.tensor(final_chunk.shape)).item()

    # Wait for any remaining normalization writes
    if normalize_futures:
        print("Waiting for final normalization writes...")
        await asyncio.gather(*normalize_futures)

    print("\nNormalization complete.")

    # --- 7. Cleanup ---
    # Stores are implicitly closed when context ends? Explicitly closing is safer if needed.
    # final_store = None # Release reference
    # weights_store = None

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