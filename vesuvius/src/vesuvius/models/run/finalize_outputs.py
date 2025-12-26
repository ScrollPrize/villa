import numpy as np
import os
from tqdm.auto import tqdm
import argparse
import zarr
import fsspec
import numcodecs
import shutil
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
from vesuvius.data.utils import open_zarr
from vesuvius.data.chunks_filter import load_chunks_json
from math import ceil
from scipy.ndimage import zoom
import json
from datetime import datetime


def _safe_normalize(vec: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Normalize a vector field along axis 0 while avoiding division by zero."""
    norms = np.sqrt(np.sum(vec * vec, axis=0, keepdims=True))
    inv = np.divide(1.0, norms, out=np.zeros_like(norms), where=norms > eps)
    return vec * inv


def _orthonormalize_surface_frame(tu: np.ndarray, tv: np.ndarray, n: np.ndarray, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project and re-orthonormalize a predicted surface frame (tu, tv, n)."""
    n_unit = _safe_normalize(n, eps)

    tu_proj = tu - (np.sum(tu * n_unit, axis=0, keepdims=True) * n_unit)
    tu_unit = _safe_normalize(tu_proj, eps)

    tv_proj = tv - (np.sum(tv * n_unit, axis=0, keepdims=True) * n_unit)
    tv_proj = tv_proj - (np.sum(tv_proj * tu_unit, axis=0, keepdims=True) * tu_unit)
    tv_unit = _safe_normalize(tv_proj, eps)

    # Fallback to n × tu when tv collapses
    tv_norm_sq = np.sum(tv_unit * tv_unit, axis=0, keepdims=True)
    if np.any(tv_norm_sq <= eps):
        fallback = np.stack([
            n_unit[1] * tu_unit[2] - n_unit[2] * tu_unit[1],
            n_unit[2] * tu_unit[0] - n_unit[0] * tu_unit[2],
            n_unit[0] * tu_unit[1] - n_unit[1] * tu_unit[0],
        ], axis=0)
        fallback_unit = _safe_normalize(fallback, eps)
        tv_unit = np.where(tv_norm_sq <= eps, fallback_unit, tv_unit)

    n_recomputed = np.stack([
        tu_unit[1] * tv_unit[2] - tu_unit[2] * tv_unit[1],
        tu_unit[2] * tv_unit[0] - tu_unit[0] * tv_unit[2],
        tu_unit[0] * tv_unit[1] - tu_unit[1] * tv_unit[0],
    ], axis=0)
    n_unit = _safe_normalize(n_recomputed, eps)

    # Align orientation with the original normal prediction
    orig_n_unit = _safe_normalize(n, eps)
    dot_sign = np.sum(n_unit * orig_n_unit, axis=0, keepdims=True)
    flip_mask = dot_sign < 0
    if np.any(flip_mask):
        tv_unit = np.where(flip_mask, -tv_unit, tv_unit)
        n_unit = np.where(flip_mask, -n_unit, n_unit)

    return tu_unit, tv_unit, n_unit


def process_chunk(chunk_info, input_path, output_path, mode, threshold, num_classes, spatial_shape, spatial_chunks, is_multi_task=False, target_info=None, squeeze_single_channel: bool = False):
    """
    Process a single chunk of the volume in parallel.
    
    Args:
        chunk_info: Dictionary with chunk boundaries and indices
        input_path: Path to input zarr
        output_path: Path to output zarr
        mode: Processing mode ("binary" or "multiclass")
        threshold: Whether to apply threshold/argmax
        num_classes: Number of classes in input
        spatial_shape: Spatial dimensions of the volume (Z, Y, X)
        output_chunks: Chunk size for output
        is_multi_task: Whether this is a multi-task model
        target_info: Dictionary with target information for multi-task models
    """
    
    chunk_idx = chunk_info['indices']
    
    spatial_slices = tuple(
        slice(idx * chunk, min((idx + 1) * chunk, shape_dim))
        for idx, chunk, shape_dim in zip(chunk_idx, spatial_chunks, spatial_shape)
    )
    
    input_store = open_zarr(
        path=input_path,
        mode='r',
        storage_options={'anon': False} if input_path.startswith('s3://') else None
    )
    
    output_store = open_zarr(
        path=output_path,
        mode='r+',
        storage_options={'anon': False} if output_path.startswith('s3://') else None
    )
    
    input_slice = (slice(None),) + spatial_slices 
    logits_np = input_store[input_slice]

    if mode == "surface_frame":
        logits_np = logits_np.astype(np.float32, copy=False)
        tu_unit, tv_unit, n_unit = _orthonormalize_surface_frame(
            logits_np[0:3], logits_np[3:6], logits_np[6:9]
        )
        output_np = np.concatenate([tu_unit, tv_unit, n_unit], axis=0).astype(np.float32, copy=False)

        # Skip writing empty chunks (all zeros)
        if not np.any(np.abs(output_np) > 0):
            return {'chunk_idx': chunk_idx, 'processed_voxels': 0, 'empty': True}

        output_slice = (slice(None),) + spatial_slices
        output_store[output_slice] = output_np
        return {'chunk_idx': chunk_idx, 'processed_voxels': output_np.size}

    if mode == "binary":
        if is_multi_task and target_info:
            # For multi-task binary, process each target separately
            target_results = []
            
            # Process each target - sort by start_channel to maintain correct order
            for target_name, info in sorted(target_info.items(), key=lambda x: x[1]['start_channel']):
                start_ch = info['start_channel']
                end_ch = info['end_channel']
                
                # Extract channels for this target
                target_logits = logits_np[start_ch:end_ch]
                
                # Compute softmax for this target
                exp_logits = np.exp(target_logits - np.max(target_logits, axis=0, keepdims=True))
                softmax = exp_logits / np.sum(exp_logits, axis=0, keepdims=True)
                
                if threshold:
                    # Create binary mask
                    binary_mask = (softmax[1] > softmax[0]).astype(np.float32)
                    target_results.append(binary_mask)
                else:
                    # Extract foreground probability
                    fg_prob = softmax[1]
                    target_results.append(fg_prob)
            
            # Stack results from all targets
            output_data = np.stack(target_results, axis=0)
        else:
            # Single task binary - existing logic
            # For binary case, we just need a softmax over dim 0 (channels)
            # Compute softmax: exp(x) / sum(exp(x))
            exp_logits = np.exp(logits_np - np.max(logits_np, axis=0, keepdims=True))
            softmax = exp_logits / np.sum(exp_logits, axis=0, keepdims=True)
            
            if threshold:
                # Create binary mask using argmax (class 1 is foreground)
                # Simply check if foreground probability > background probability
                binary_mask = (softmax[1] > softmax[0]).astype(np.float32)
                output_data = binary_mask[np.newaxis, ...]  # Add channel dim
            else:
                # Extract foreground probability (channel 1)
                fg_prob = softmax[1:2]  
                output_data = fg_prob
            
    else:  # multiclass 
        # Apply softmax over channel dimension
        exp_logits = np.exp(logits_np - np.max(logits_np, axis=0, keepdims=True)) 
        softmax = exp_logits / np.sum(exp_logits, axis=0, keepdims=True)
        
        # Compute argmax
        argmax = np.argmax(logits_np, axis=0).astype(np.float32)
        argmax = argmax[np.newaxis, ...]  # Add channel dim
        
        if threshold: 
            # If threshold is provided for multiclass, only save the argmax
            output_data = argmax
        else:
            # Concatenate softmax and argmax
            output_data = np.concatenate([softmax, argmax], axis=0)
    
    # output_data is already numpy
    output_np = output_data
    
    # Scale to uint8 range [0, 255]
    min_val = output_np.min()
    max_val = output_np.max()
    if min_val < max_val: 
        output_np = ((output_np - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    else:
        # All values are the same after processing - this is effectively an empty chunk
        # Don't write anything to respect write_empty_chunks=False
        return {'chunk_idx': chunk_idx, 'processed_voxels': 0, 'empty': True}
    
    # Final check: if the processed data is homogeneous, don't write it
    first_processed_value = output_np.flat[0]
    if np.all(output_np == first_processed_value):
        # Processed chunk is homogeneous (e.g., all 0s or all 255s), skip writing
        return {'chunk_idx': chunk_idx, 'processed_voxels': 0, 'empty': True}

    if squeeze_single_channel:
        output_store[spatial_slices] = output_np[0]
    else:
        output_slice = (slice(None),) + spatial_slices
        output_store[output_slice] = output_np
    return {'chunk_idx': chunk_idx, 'processed_voxels': np.prod(output_data.shape)}


def finalize_logits(
    input_path: str,
    output_path: str,
    mode: str = "binary",  # "binary" or "multiclass"
    threshold: bool = False,  # If True, will apply argmax and only save class predictions
    delete_intermediates: bool = False,  # If True, will delete the input logits after processing
    chunk_size: tuple = None,  # Optional custom chunk size for output
    num_workers: int = None,  # Number of worker processes to use
    verbose: bool = True,
    input_zarr_path: str = None,  # Path to original input zarr for chunks.json detection
    chunks_filter_mode: str = 'auto',  # 'auto', 'disabled'
):
    """
    Process merged logits and apply softmax/argmax to produce final outputs.

    Args:
        input_path: Path to the merged logits Zarr store
        output_path: Path for the finalized output Zarr store
        mode: "binary" (2 channels) or "multiclass" (>2 channels)
        threshold: If True, applies argmax and only saves class predictions
        delete_intermediates: Whether to delete input logits after processing
        chunk_size: Optional custom chunk size for output (Z,Y,X)
        num_workers: Number of worker processes to use for parallel processing
        verbose: Print progress messages
        input_zarr_path: Path to original input zarr for chunks.json detection.
        chunks_filter_mode: 'auto' (use chunks.json if present) or 'disabled'.
    """
    numcodecs.blosc.use_threads = False
    
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() // 2)
    
    print(f"Using {num_workers} worker processes")
    
    compressor = numcodecs.Blosc(
        cname='zstd',
        clevel=1,  # compression level is 1 because we're only using this for mostly empty chunks
        shuffle=numcodecs.blosc.SHUFFLE
    )
    
    print(f"Opening input logits: {input_path}")
    print(f"Mode: {mode}, Threshold flag: {threshold}")
    input_store = open_zarr(
        path=input_path,
        mode='r',
        storage_options={'anon': False} if input_path.startswith('s3://') else None,
        verbose=verbose
    )

    stored_mode = input_store.attrs.get('processing_mode') if hasattr(input_store, 'attrs') else None
    if stored_mode and stored_mode != mode:
        raise ValueError(
            f"Requested mode '{mode}' does not match blended logits metadata ('{stored_mode}'). "
            "Re-run with --mode matching the stored processing_mode."
        )

    input_shape = input_store.shape
    num_classes = input_shape[0]
    spatial_shape = input_shape[1:]  # (Z, Y, X)

    if mode == 'surface_frame' and num_classes != 9:
        raise ValueError(f"Surface-frame mode expects 9 channels, but input has {num_classes}.")
    
    # Check for multi-task metadata
    is_multi_task = False
    target_info = None
    if hasattr(input_store, 'attrs'):
        is_multi_task = input_store.attrs.get('is_multi_task', False)
        target_info = input_store.attrs.get('target_info', None)
    
    # Verify we have the expected number of channels based on mode
    print(f"Input shape: {input_shape}, Num classes: {num_classes}")
    if is_multi_task:
        print(f"Multi-task model detected with targets: {list(target_info.keys()) if target_info else 'None'}")
    
    if mode == "binary":
        if is_multi_task and target_info:
            # For multi-task binary, each target should have 2 channels
            expected_channels = sum(info['out_channels'] for info in target_info.values())
            if num_classes != expected_channels:
                raise ValueError(f"Multi-task binary mode expects {expected_channels} total channels, but input has {num_classes} channels.")
        elif num_classes != 2:
            raise ValueError(f"Binary mode expects 2 channels, but input has {num_classes} channels.")
    elif mode == "multiclass" and num_classes < 2:
        raise ValueError(f"Multiclass mode expects at least 2 channels, but input has {num_classes} channels.")
    
    if chunk_size is None:
        try:
            src_chunks = input_store.chunks
            # Input chunks include class dimension - extract spatial dimensions
            output_chunks = src_chunks[1:]
            if verbose:
                print(f"Using input chunk size: {output_chunks}")
        except:
            raise ValueError("Cannot determine input chunk size. Please specify --chunk-size.")
    else:
        output_chunks = chunk_size
        if verbose:
            print(f"Using specified chunk size: {output_chunks}")
    
    output_dtype = np.uint8
    if mode == "surface_frame":
        out_channels = num_classes
        output_shape = (out_channels, *spatial_shape)
        squeeze_single_channel = False
        output_dtype = np.float32
        print("Output will have 9 channels ordered as [t_u(3), t_v(3), n(3)] with float32 precision.")
    elif mode == "binary":
        if is_multi_task and target_info:
            num_targets = len(target_info)
            output_shape = (num_targets, *spatial_shape)
            if threshold:
                print(f"Output will have {num_targets} channels: [" + ", ".join(f"{k}_binary_mask" for k in sorted(target_info.keys())) + "]")
            else:
                print(f"Output will have {num_targets} channels: [" + ", ".join(f"{k}_softmax_fg" for k in sorted(target_info.keys())) + "]")
            out_channels = num_targets
        else:
            output_shape = (1, *spatial_shape)
            if threshold:
                print("Output will have 1 channel: [binary_mask]")
            else:
                print("Output will have 1 channel: [softmax_fg]")
            out_channels = 1
        squeeze_single_channel = (out_channels == 1)
    else:  # multiclass
        if threshold:
            output_shape = (1, *spatial_shape)
            out_channels = 1
            print("Output will have 1 channel: [argmax]")
        else:
            output_shape = (num_classes + 1, *spatial_shape)
            out_channels = num_classes + 1
            print(f"Output will have {num_classes + 1} channels: [softmax_c0...softmax_cN, argmax]")
        squeeze_single_channel = (out_channels == 1)

    # Prepare shapes and chunks for level 0
    if squeeze_single_channel:
        final_shape_lvl0 = spatial_shape
        spatial_chunks = output_chunks
        chunks_lvl0 = spatial_chunks
    else:
        final_shape_lvl0 = (out_channels, *spatial_shape)
        spatial_chunks = output_chunks
        chunks_lvl0 = (1, *spatial_chunks)

    # Create multiscale root level 0 array at <output_path>/0
    root_path = output_path.rstrip('/')
    level0_path = os.path.join(root_path, '0')
    print(f"Creating output multiscale level 0 store: {level0_path}")
    output_store = open_zarr(
        path=level0_path,
        mode='w',
        storage_options={'anon': False} if level0_path.startswith('s3://') else None,
        verbose=verbose,
        shape=final_shape_lvl0,
        chunks=chunks_lvl0,
        dtype=output_dtype,
        compressor=compressor,
        write_empty_chunks=False,
        overwrite=True
    )
    
    def get_chunk_indices(spatial_shape, spatial_chunks, valid_chunk_indices=None, zarr_chunk_size=None):
        # For each dimension, calculate how many chunks we need

        # Generate all combinations of chunk indices
        from itertools import product
        chunk_counts = [int(np.ceil(s / c)) for s, c in zip(spatial_shape, spatial_chunks)]
        chunk_indices = list(product(*[range(count) for count in chunk_counts]))

        # Build set of valid regions if chunks.json filtering is enabled
        valid_regions = None
        if valid_chunk_indices is not None and zarr_chunk_size is not None:
            valid_regions = set()
            for chunk_idx in valid_chunk_indices:
                ci_z, ci_y, ci_x = chunk_idx
                valid_regions.add((ci_z, ci_y, ci_x))

        # list of dicts with indices for each chunk
        # Each dict will have 'indices' key with the chunk indices
        # we pass these to the worker functions
        chunks_info = []
        for idx in chunk_indices:
            # If filtering by chunks.json, check if this chunk overlaps with any valid region
            if valid_regions is not None:
                cZ, cY, cX = zarr_chunk_size
                sZ, sY, sX = spatial_chunks

                # Calculate voxel range for this processing chunk
                z_start = idx[0] * sZ
                z_end = min((idx[0] + 1) * sZ, spatial_shape[0])
                y_start = idx[1] * sY
                y_end = min((idx[1] + 1) * sY, spatial_shape[1])
                x_start = idx[2] * sX
                x_end = min((idx[2] + 1) * sX, spatial_shape[2])

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

            chunks_info.append({'indices': idx})

        return chunks_info

    # --- Load chunk indices for filtering ---
    # First try to get touched_chunk_indices from blending step (saved in attrs)
    # This is more accurate than chunks.json because it accounts for patch overlap
    valid_chunk_indices = None
    zarr_chunk_size = None
    if chunks_filter_mode != 'disabled':
        # Check if blending saved touched_chunk_indices
        if hasattr(input_store, 'attrs'):
            touched_indices = input_store.attrs.get('touched_chunk_indices')
            output_chunk_size = input_store.attrs.get('output_chunk_size')
            if touched_indices is not None and output_chunk_size is not None:
                valid_chunk_indices = touched_indices
                zarr_chunk_size = tuple(output_chunk_size)
                if verbose:
                    print(f"\nLoaded {len(touched_indices)} touched chunks from blending metadata")
                    print(f"  Output chunk size: {zarr_chunk_size}")

        # Fallback to chunks.json if no blending metadata
        if valid_chunk_indices is None:
            if input_zarr_path is None and hasattr(input_store, 'attrs'):
                input_zarr_path = input_store.attrs.get('input_zarr_path')
                if input_zarr_path is None:
                    inference_args = input_store.attrs.get('inference_args', {})
                    input_zarr_path = inference_args.get('input_dir')

            if input_zarr_path:
                chunks_json = load_chunks_json(input_zarr_path)
                if chunks_json:
                    level0_chunks = chunks_json.get('chunks_by_level', {}).get('0', [])
                    if level0_chunks:
                        valid_chunk_indices = level0_chunks
                        zarr_chunk_size = tuple(chunks_json.get('chunk_size', [128, 128, 128]))
                        if verbose:
                            print(f"\nLoaded chunks.json with {len(level0_chunks)} valid chunks (fallback)")
                            print(f"  Zarr chunk size: {zarr_chunk_size}")

    chunk_infos = get_chunk_indices(spatial_shape, spatial_chunks, valid_chunk_indices, zarr_chunk_size)
    total_chunks = len(chunk_infos)

    if valid_chunk_indices is not None:
        full_chunks = get_chunk_indices(spatial_shape, spatial_chunks)
        print(f"Filtered to {total_chunks} chunks (from {len(full_chunks)} total) based on chunk metadata")
    else:
        print(f"Processing data in {total_chunks} chunks using {num_workers} worker processes...")
    
    # main processing function with partial application of common arguments
    # This allows us to pass only the chunk_info to the worker function
    # and keep the other parameters fixed
    process_chunk_partial = partial(
        process_chunk,
        input_path=input_path,
        output_path=level0_path,
        mode=mode,
        threshold=threshold,
        num_classes=num_classes,
        spatial_shape=spatial_shape,
        spatial_chunks=spatial_chunks,
        squeeze_single_channel=squeeze_single_channel,
        is_multi_task=is_multi_task,
        target_info=target_info
    )
    
    total_processed = 0
    empty_chunks = 0
    with ProcessPoolExecutor(max_workers=num_workers) as executor:

        future_to_chunk = {executor.submit(process_chunk_partial, chunk): chunk for chunk in chunk_infos}
        
        for future in tqdm(
            as_completed(future_to_chunk),
            total=total_chunks,
            desc="Processing Chunks",
            disable=not verbose
        ):
            try:
                result = future.result()
                if result.get('empty', False):
                    empty_chunks += 1
                else:
                    total_processed += result['processed_voxels']
            except Exception as e:
                print(f"Error processing chunk: {e}")
                raise e
    
    print(f"\nOutput processing complete. Processed {total_chunks - empty_chunks} chunks, skipped {empty_chunks} empty chunks ({empty_chunks/total_chunks:.2%}).")
    
    try:
        if hasattr(input_store, 'attrs') and hasattr(output_store, 'attrs'):
            for key in input_store.attrs:
                output_store.attrs[key] = input_store.attrs[key]
                
            output_store.attrs['threshold_applied'] = threshold
            output_store.attrs['empty_chunks_skipped'] = empty_chunks
            output_store.attrs['total_chunks'] = total_chunks
            output_store.attrs['empty_chunk_percentage'] = float(empty_chunks/total_chunks) if total_chunks > 0 else 0.0
            output_store.attrs['processing_mode'] = mode
            if mode == "surface_frame":
                output_store.attrs['surface_frame_layout'] = [
                    't_u_x', 't_u_y', 't_u_z',
                    't_v_x', 't_v_y', 't_v_z',
                    'n_x', 'n_y', 'n_z'
                ]
                output_store.attrs['surface_frame_orthonormalized'] = True
                output_store.attrs['value_dtype'] = 'float32'
    except Exception as e:
        print(f"Warning: Failed to copy metadata: {e}")

    # Build multiscale pyramid (levels 1..5) with 2x downsampling
    def build_multiscales(root_path: str, levels: int = 6):
        if verbose:
            print(f"Building multiscale pyramid (levels 1-{levels - 1})")
        try:
            # Open level 0 lazily
            lvl0 = open_zarr(os.path.join(root_path, '0'), mode='r', storage_options={'anon': False} if root_path.startswith('s3://') else None)
            lvl0_shape = lvl0.shape
            has_channel = (len(lvl0_shape) == 4)
            datasets = [{'path': '0'}]

            prev_path = os.path.join(root_path, '0')
            prev_shape = lvl0_shape

            for i in range(1, levels):
                # Compute next level shape
                if has_channel:
                    C, Z, Y, X = prev_shape
                    tZ, tY, tX = max(1, ceil(Z/2)), max(1, ceil(Y/2)), max(1, ceil(X/2))
                    next_shape = (C, tZ, tY, tX)
                    # Use the same chunks as level 0 (channel, z, y, x), clipped to shape when necessary
                    zc = min(chunks_lvl0[1], tZ)
                    yc = min(chunks_lvl0[2], tY)
                    xc = min(chunks_lvl0[3], tX)
                    chunks = (chunks_lvl0[0], zc, yc, xc)
                else:
                    Z, Y, X = prev_shape
                    tZ, tY, tX = max(1, ceil(Z/2)), max(1, ceil(Y/2)), max(1, ceil(X/2))
                    next_shape = (tZ, tY, tX)
                    # Use the same spatial chunks as level 0, clipped to shape
                    zc = min(chunks_lvl0[0], tZ)
                    yc = min(chunks_lvl0[1], tY)
                    xc = min(chunks_lvl0[2], tX)
                    chunks = (zc, yc, xc)

                if verbose:
                    print(f"Downsampling level {i}/{levels - 1}: shape {next_shape}, chunks {chunks}")

                lvl_path = os.path.join(root_path, str(i))
                ds_store = open_zarr(
                    path=lvl_path,
                    mode='w',
                    storage_options={'anon': False} if lvl_path.startswith('s3://') else None,
                    shape=next_shape,
                    chunks=chunks,
                    dtype=lvl0.dtype,
                    compressor=compressor,
                    write_empty_chunks=False,
                    overwrite=True
                )

                # Iterate output tiles and compute from prev level tiles
                # Open prev store lazily
                prev_store = open_zarr(prev_path, mode='r', storage_options={'anon': False} if prev_path.startswith('s3://') else None)

                # Determine iteration grid based on ds_store chunks
                if has_channel:
                    _, Zp, Yp, Xp = next_shape
                    zc, yc, xc = chunks[1], chunks[2], chunks[3]
                else:
                    Zp, Yp, Xp = next_shape
                    zc, yc, xc = chunks[0], chunks[1], chunks[2]

                prev_z, prev_y, prev_x = prev_shape[-3], prev_shape[-2], prev_shape[-1]
                total_tiles = ((Zp + zc - 1) // zc) * ((Yp + yc - 1) // yc) * ((Xp + xc - 1) // xc)
                downsample_workers = max(1, min(num_workers, total_tiles))
                if verbose:
                    print(f"Using {downsample_workers} workers for downsampling level {i}")

                def iter_blocks():
                    for oz in range(0, Zp, zc):
                        for oy in range(0, Yp, yc):
                            for ox in range(0, Xp, xc):
                                yield oz, oy, ox

                def downsample_block(block_coords):
                    oz, oy, ox = block_coords
                    oz1 = min(oz + zc, Zp)
                    oy1 = min(oy + yc, Yp)
                    ox1 = min(ox + xc, Xp)

                    # Corresponding prev indices
                    pz0, py0, px0 = oz * 2, oy * 2, ox * 2
                    pz1 = min(oz1 * 2, prev_z)
                    py1 = min(oy1 * 2, prev_y)
                    px1 = min(ox1 * 2, prev_x)

                    if has_channel:
                        prev_block = prev_store[(slice(None), slice(pz0, pz1), slice(py0, py1), slice(px0, px1))]
                        # Pad to even along spatial dims
                        pad_z = (0, (prev_block.shape[1] % 2))
                        pad_y = (0, (prev_block.shape[2] % 2))
                        pad_x = (0, (prev_block.shape[3] % 2))
                        prev_block = np.pad(prev_block, ((0, 0), pad_z, pad_y, pad_x), mode='edge')
                        # Reshape and average
                        Cb, Zb, Yb, Xb = prev_block.shape
                        block_ds = prev_block.reshape(Cb, Zb//2, 2, Yb//2, 2, Xb//2, 2).mean(axis=(2, 4, 6))
                        if mode == "surface_frame" and Cb == 9:
                            tu_ds, tv_ds, n_ds = _orthonormalize_surface_frame(
                                block_ds[0:3], block_ds[3:6], block_ds[6:9]
                            )
                            block_ds = np.concatenate([tu_ds, tv_ds, n_ds], axis=0).astype(block_ds.dtype, copy=False)
                        ds_store[(slice(None), slice(oz, oz1), slice(oy, oy1), slice(ox, ox1))] = block_ds
                    else:
                        prev_block = prev_store[(slice(pz0, pz1), slice(py0, py1), slice(px0, px1))]
                        pad_z = (0, (prev_block.shape[0] % 2))
                        pad_y = (0, (prev_block.shape[1] % 2))
                        pad_x = (0, (prev_block.shape[2] % 2))
                        prev_block = np.pad(prev_block, (pad_z, pad_y, pad_x), mode='edge')
                        Zb, Yb, Xb = prev_block.shape
                        block_ds = prev_block.reshape(Zb//2, 2, Yb//2, 2, Xb//2, 2).mean(axis=(1, 3, 5))
                        ds_store[(slice(oz, oz1), slice(oy, oy1), slice(ox, ox1))] = block_ds

                if downsample_workers == 1:
                    for block_coords in tqdm(
                        iter_blocks(),
                        total=total_tiles,
                        desc=f"Downsample level {i}",
                        unit="blocks",
                        disable=not verbose
                    ):
                        downsample_block(block_coords)
                else:
                    with ThreadPoolExecutor(max_workers=downsample_workers) as executor:
                        futures = [executor.submit(downsample_block, block_coords) for block_coords in iter_blocks()]
                        for future in tqdm(
                            as_completed(futures),
                            total=total_tiles,
                            desc=f"Downsample level {i}",
                            unit="blocks",
                            disable=not verbose
                        ):
                            future.result()

                datasets.append({'path': str(i)})
                prev_path = lvl_path
                prev_shape = next_shape

            # Write OME-NGFF multiscales metadata on the root group
            try:
                root = zarr.open_group(root_path, mode='a', storage_options={'anon': False} if root_path.startswith('s3://') else None)
                axes = []
                if has_channel:
                    axes.append({'name': 'c', 'type': 'channel'})
                axes.extend([
                    {'name': 'z', 'type': 'space'},
                    {'name': 'y', 'type': 'space'},
                    {'name': 'x', 'type': 'space'}
                ])
                root.attrs['multiscales'] = [{
                    'version': '0.4',
                    'axes': axes,
                    'datasets': datasets
                }]
            except Exception as me:
                print(f"Warning: Failed to write multiscales metadata: {me}")
        except Exception as be:
            print(f"Warning: Failed to build multiscales: {be}")

    build_multiscales(root_path)

    # Write metadata.json at the root of the zarr with inference args and run time
    try:
        meta = {}
        if hasattr(input_store, 'attrs') and 'inference_args' in input_store.attrs:
            meta.update(input_store.attrs['inference_args'])
        # Add/override finalize context
        meta.update({
            'finalize_mode': mode,
            'finalize_threshold': bool(threshold),
            'finalize_time': datetime.utcnow().isoformat() + 'Z',
            'input_logits_path': input_path,
            'output_path': output_path
        })
        # Write using fsspec so it works for local and remote
        proto = output_path.split('://', 1)[0] if '://' in output_path else None
        meta_path = os.path.join(output_path, 'metadata.json')
        if proto in ('s3', 'gs', 'azure'):
            fs = fsspec.filesystem(proto, anon=False if proto == 's3' else None)
            with fs.open(meta_path, 'w') as f:
                json.dump(meta, f, indent=2)
        else:
            os.makedirs(output_path, exist_ok=True)
            with open(meta_path, 'w') as f:
                json.dump(meta, f, indent=2)
        if verbose:
            print(f"Wrote metadata.json to {meta_path}")
    except Exception as me:
        print(f"Warning: Failed to write metadata.json: {me}")
    
    if delete_intermediates:
        print(f"Deleting intermediate logits: {input_path}")
        try:
            # we have to use fsspec for s3/gs/azure paths 
            # os module does not work well with them
            if input_path.startswith(('s3://', 'gs://', 'azure://')):
                fs_protocol = input_path.split('://', 1)[0]
                fs = fsspec.filesystem(fs_protocol, anon=False if fs_protocol == 's3' else None)
                
                # Remove protocol prefix for fs operations
                path_no_prefix = input_path.split('://', 1)[1]
                
                if fs.exists(path_no_prefix):
                    fs.rm(path_no_prefix, recursive=True)
                    print(f"Successfully deleted intermediate logits (remote path)")
            elif os.path.exists(input_path):
                shutil.rmtree(input_path)
                print(f"Successfully deleted intermediate logits (local path)")
        except Exception as e:
            print(f"Warning: Failed to delete intermediate logits: {e}")
            print(f"You may need to delete them manually: {input_path}")
    
    print(f"Final multiscale output saved to: {output_path}")


# --- Command Line Interface ---
def main():
    """Entry point for the vesuvius.finalize command."""
    parser = argparse.ArgumentParser(description='Process merged logits to produce final outputs.')
    parser.add_argument('input_path', type=str,
                      help='Path to the merged logits Zarr store')
    parser.add_argument('output_path', type=str,
                      help='Path for the finalized output Zarr store')
    parser.add_argument('--mode', type=str, choices=['binary', 'multiclass', 'surface_frame'], default='binary',
                      help='Processing mode. Use "binary"/"multiclass" for segmentation or "surface_frame" for 9-channel frame regression.')
    parser.add_argument('--threshold', dest='threshold', action='store_true',
                      help='If set, applies argmax and only saves the class predictions (no probabilities). Works for both binary and multiclass.')
    parser.add_argument('--delete-intermediates', dest='delete_intermediates', action='store_true',
                      help='Delete intermediate logits after processing')
    parser.add_argument('--chunk-size', dest='chunk_size', type=str, default=None,
                      help='Spatial chunk size (Z,Y,X) for output Zarr. Comma-separated. If not specified, input chunks will be used.')
    parser.add_argument('--num-workers', dest='num_workers', type=int, default=None,
                      help='Number of worker processes for parallel processing. Default: CPU_COUNT // 2')
    parser.add_argument('--quiet', dest='quiet', action='store_true',
                      help='Suppress verbose output')
    parser.add_argument('--input-zarr', type=str, default=None,
                      help='Path to original input zarr for chunks.json detection (auto-detected from logits metadata if not provided)')
    parser.add_argument('--chunks-filter-mode', type=str, default='auto',
                      choices=['auto', 'disabled'],
                      help='Chunk filtering: auto (use chunks.json if present), disabled (process full volume)')

    args = parser.parse_args()

    if args.mode == 'surface_frame' and args.threshold:
        parser.error("--threshold is not supported when mode is 'surface_frame'.")

    chunks = None
    if args.chunk_size:
        try:
            chunks = tuple(map(int, args.chunk_size.split(',')))
            if len(chunks) != 3: raise ValueError()
        except ValueError:
            parser.error("Invalid chunk_size format. Expected 3 comma-separated integers (Z,Y,X).")
    
    try:
        finalize_logits(
            input_path=args.input_path,
            output_path=args.output_path,
            mode=args.mode,
            threshold=args.threshold,
            delete_intermediates=args.delete_intermediates,
            chunk_size=chunks,
            num_workers=args.num_workers,
            verbose=not args.quiet,
            input_zarr_path=args.input_zarr,
            chunks_filter_mode=args.chunks_filter_mode,
        )
        return 0
    except Exception as e:
        print(f"\n--- Finalization Failed ---")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    import sys
    sys.exit(main())
