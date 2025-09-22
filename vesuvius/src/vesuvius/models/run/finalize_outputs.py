import numpy as np
import os
from tqdm.auto import tqdm
import argparse
import zarr
import fsspec
import numcodecs
import shutil
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from vesuvius.data.utils import open_zarr
from math import ceil
from scipy.ndimage import zoom
import json
from datetime import datetime


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
    verbose: bool = True
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
    
    input_shape = input_store.shape
    num_classes = input_shape[0]
    spatial_shape = input_shape[1:]  # (Z, Y, X)
    
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
    
    if mode == "binary":
        if is_multi_task and target_info:
            # For multi-task binary, output one channel per target
            num_targets = len(target_info)
            output_shape = (num_targets, *spatial_shape)  # One mask per target
            if threshold:
                print(f"Output will have {num_targets} channels: [" + ", ".join(f"{k}_binary_mask" for k in sorted(target_info.keys())) + "]")
            else:
                print(f"Output will have {num_targets} channels: [" + ", ".join(f"{k}_softmax_fg" for k in sorted(target_info.keys())) + "]")
        else:
            if threshold:  
                # If thresholding, only output argmax channel for binary
                output_shape = (1, *spatial_shape)  # Just the binary mask (argmax)
                print("Output will have 1 channel: [binary_mask]")
            else:
                 # Just softmax of FG class
                output_shape = (1, *spatial_shape) 
                print("Output will have 1 channel: [softmax_fg]")
    else:  # multiclass
        if threshold:  
            # If threshold is provided for multiclass, only save the argmax
            output_shape = (1, *spatial_shape)
            print("Output will have 1 channel: [argmax]")
        else:
            # For multiclass, we'll output num_classes channels (all softmax values)
            # Plus 1 channel for the argmax
            output_shape = (num_classes + 1, *spatial_shape)
            print(f"Output will have {num_classes + 1} channels: [softmax_c0...softmax_cN, argmax]")
    
    # Decide channel count and squeeze behavior
    if mode == "binary":
        out_channels = (len(target_info) if (is_multi_task and target_info) else 1)
    else:
        out_channels = (1 if threshold else (num_classes + 1))
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
        dtype=np.uint8,
        compressor=compressor,
        write_empty_chunks=False,
        overwrite=True
    )
    
    def get_chunk_indices(spatial_shape, spatial_chunks):
        # For each dimension, calculate how many chunks we need
        
        # Generate all combinations of chunk indices
        from itertools import product
        chunk_counts = [int(np.ceil(s / c)) for s, c in zip(spatial_shape, spatial_chunks)]
        chunk_indices = list(product(*[range(count) for count in chunk_counts]))
        
        # list of dicts with indices for each chunk
        # Each dict will have 'indices' key with the chunk indices
        # we pass these to the worker functions 
        chunks_info = []
        for idx in chunk_indices:
            chunks_info.append({'indices': idx})
        
        return chunks_info
    
    chunk_infos = get_chunk_indices(spatial_shape, spatial_chunks)
    total_chunks = len(chunk_infos)
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
        
        from concurrent.futures import as_completed
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
                
            output_store.attrs['processing_mode'] = mode
            output_store.attrs['threshold_applied'] = threshold
            output_store.attrs['empty_chunks_skipped'] = empty_chunks
            output_store.attrs['total_chunks'] = total_chunks
            output_store.attrs['empty_chunk_percentage'] = float(empty_chunks/total_chunks) if total_chunks > 0 else 0.0
    except Exception as e:
        print(f"Warning: Failed to copy metadata: {e}")

    # Build multiscale pyramid (levels 1..5) with 2x downsampling
    def build_multiscales(root_path: str, levels: int = 6):
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

                for oz in range(0, Zp, zc):
                    for oy in range(0, Yp, yc):
                        for ox in range(0, Xp, xc):
                            oz1 = min(oz + zc, Zp)
                            oy1 = min(oy + yc, Yp)
                            ox1 = min(ox + xc, Xp)

                            # Corresponding prev indices
                            pz0, py0, px0 = oz * 2, oy * 2, ox * 2
                            pz1, py1, px1 = min(oz1 * 2, prev_shape[-3]), min(oy1 * 2, prev_shape[-2]), min(ox1 * 2, prev_shape[-1])

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
    parser.add_argument('--mode', type=str, choices=['binary', 'multiclass'], default='binary',
                      help='Processing mode. "binary" for 2-class segmentation, "multiclass" for >2 classes. Default: binary')
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
    
    args = parser.parse_args()
    
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
            verbose=not args.quiet
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
