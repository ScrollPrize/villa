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
from concurrent.futures import ProcessPoolExecutor, as_completed
import math
from data.utils import open_zarr
import traceback


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


def process_chunk(chunk_info, parent_dir, output_path, weights_path, gaussian_map, 
                patch_size, part_files):
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
    """
    
    # Extract chunk boundaries
    z_start, z_end = chunk_info['z_start'], chunk_info['z_end']
    y_start, y_end = chunk_info['y_start'], chunk_info['y_end']
    x_start, x_end = chunk_info['x_start'], chunk_info['x_end']
    
    pZ, pY, pX = patch_size
    
    gaussian_map_spatial_np = gaussian_map[0]  # Shape (pZ, pY, pX)
    
    output_store = open_zarr(output_path, mode='r+', storage_options={'anon': False} if output_path.startswith('s3://') else None)
    weights_store = open_zarr(weights_path, mode='r+', storage_options={'anon': False} if weights_path.startswith('s3://') else None)
    
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
        output_slice = (
            slice(None), 
            slice(z_start, z_end),
            slice(y_start, y_end),
            slice(x_start, x_end)
        )
        
        weight_slice = (
            slice(z_start, z_end),
            slice(y_start, y_end),
            slice(x_start, x_end)
        )
        
        output_store[output_slice] = chunk_logits
        weights_store[weight_slice] = chunk_weights
    
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

    z_start, z_end = chunk_info['z_start'], chunk_info['z_end']
    y_start, y_end = chunk_info['y_start'], chunk_info['y_end']
    x_start, x_end = chunk_info['x_start'], chunk_info['x_end']
    
    output_store = open_zarr(output_path, mode='r+', storage_options={'anon': False} if output_path.startswith('s3://') else None)
    weights_store = open_zarr(weights_path, mode='r', storage_options={'anon': False} if weights_path.startswith('s3://') else None)
    
    output_slice = (
        slice(None), 
        slice(z_start, z_end),
        slice(y_start, y_end),
        slice(x_start, x_end)
    )
    
    weight_slice = (
        slice(z_start, z_end),
        slice(y_start, y_end),
        slice(x_start, x_end)
    )
    
    logits = output_store[output_slice]
    weights = weights_store[weight_slice]
    
    normalized = np.zeros_like(logits)
    
    # divide only where weights > 0
    np.divide(logits, weights[np.newaxis, :, :, :] + epsilon, 
              out=normalized, where=weights[np.newaxis, :, :, :] > 0)
    
    output_store[output_slice] = normalized
    
    return {
        'chunk': chunk_info,
        'normalized_voxels': np.prod(normalized.shape)
    }

# --- Utility Functions ---
def calculate_chunks(volume_shape, output_chunks=None):

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
        verbose: bool = True):
    """
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
        verbose: Print progress messages.
    """

    # blosc has an issuse with threading , so we disable it
    numcodecs.blosc.use_threads = False
    if weight_accumulator_path is None:
        base, _ = os.path.splitext(output_path)
        weight_accumulator_path = f"{base}_weights.zarr"
    
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
    weights_shape = original_volume_shape  # (D, H, W)

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
        weights_chunks = output_chunks[1:]
        if verbose:
            print(f"  Using chunk_size {output_chunks[1:]} based directly on patch_size")
    else:
        output_chunks = (1, *chunk_size) 
        weights_chunks = chunk_size
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
    
    print(f"Creating weight accumulator store: {weight_accumulator_path}")
    print(f"  Shape: {weights_shape}, Chunks: {weights_chunks}")
        
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
        write_empty_chunks=False 
    )

    # --- 4. Generate Gaussian Map ---
    gaussian_map = generate_gaussian_map(patch_size, sigma_scale=sigma_scale)

    # --- 5. Calculate Processing Chunks ---
    chunks = calculate_chunks(
        original_volume_shape,
        output_chunks=output_chunks[1:]  # Skip the class dimension from output_chunks
    )
    
    print(f"Divided volume into {len(chunks)} chunks for parallel processing")
    
    # --- 6. Process Chunks in Parallel ---
    print("\n--- Accumulating Weighted Patches ---")
    
    process_chunk_partial = partial(
        process_chunk,
        parent_dir=parent_dir,
        output_path=output_path,
        weights_path=weight_accumulator_path,
        gaussian_map=gaussian_map,
        patch_size=patch_size,
        part_files=part_files
    )
    
    total_patches_processed = 0
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_chunk = {executor.submit(process_chunk_partial, chunk): chunk for chunk in chunks}
        
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
    
    normalize_chunk_partial = partial(
        normalize_chunk,
        output_path=output_path,
        weights_path=weight_accumulator_path
    )
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:

        futures = {executor.submit(normalize_chunk_partial, chunk): chunk for chunk in chunks}
        
        for future in tqdm(
            as_completed(futures),
            total=len(chunks),
            desc="Normalizing Chunks",
            disable=not verbose
        ):
            try:
                result = future.result() 
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
        # Copy all attributes from the input part
        if hasattr(part0_logits_store, 'attrs'):
            for key, value in part0_logits_store.attrs.items():
                output_zarr.attrs[key] = value
        # Update/add specific attributes
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
    parser.add_argument('--quiet', action='store_true',
                        help='Disable verbose progress messages (tqdm bars still show).')

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
            weight_accumulator_path=args.weights_path,
            sigma_scale=args.sigma_scale,
            chunk_size=chunks,
            num_workers=args.num_workers,
            compression_level=args.compression_level,
            delete_weights=not args.keep_weights,
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