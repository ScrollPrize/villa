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
from data.utils import open_zarr


def process_chunk(chunk_info, input_path, output_path, mode, threshold, num_classes, spatial_shape, output_chunks, is_multi_task=False, target_info=None):
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
        for idx, chunk, shape_dim in zip(chunk_idx, output_chunks[1:], spatial_shape)
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
    
    # Check if this is an empty chunk (all values are the same, indicating no meaningful data)
    # This handles empty patches that have been filled with any constant value
    first_value = logits_np.flat[0]  # Get the first value
    is_empty = np.allclose(logits_np, first_value, rtol=1e-6)
    
    if is_empty:
        # For empty/homogeneous patches, don't write anything to the output store
        # This ensures write_empty_chunks=False works correctly
        return {'chunk_idx': chunk_idx, 'processed_voxels': 0, 'empty': True}
    
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
    
    print(f"Creating output store: {output_path}")
    output_chunks = (1, *output_chunks)  # Chunk each channel separately
    
    output_store = open_zarr(
        path=output_path,
        mode='w',
        storage_options={'anon': False} if output_path.startswith('s3://') else None,
        verbose=verbose,
        shape=output_shape,
        chunks=output_chunks,
        dtype=np.uint8,  
        compressor=compressor,
        write_empty_chunks=False,
        overwrite=True
    )
    
    def get_chunk_indices(shape, chunks):
        # For each dimension, calculate how many chunks we need
        # Skip first dimension (channels)
        spatial_shape = shape[1:] 
        spatial_chunks = chunks[1:]
        
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
    
    chunk_infos = get_chunk_indices(input_shape, output_chunks)
    total_chunks = len(chunk_infos)
    print(f"Processing data in {total_chunks} chunks using {num_workers} worker processes...")
    
    # main processing function with partial application of common arguments
    # This allows us to pass only the chunk_info to the worker function
    # and keep the other parameters fixed
    process_chunk_partial = partial(
        process_chunk,
        input_path=input_path,
        output_path=output_path,
        mode=mode,
        threshold=threshold,
        num_classes=num_classes,
        spatial_shape=spatial_shape,
        output_chunks=output_chunks,
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
    
    print(f"Final output saved to: {output_path}")


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