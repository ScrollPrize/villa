#!/usr/bin/env python3
"""
Run script for vesuvius.create_st - handles multi-GPU orchestration
"""

import argparse
import os
import sys
import torch
import subprocess
import time
import shutil
import fsspec
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import zarr
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from data.utils import open_zarr


def get_available_gpus():
    """Returns a list of available GPU IDs."""
    try:
        num_gpus = torch.cuda.device_count()
        return list(range(num_gpus))
    except:
        return []


def select_gpus(gpus_arg):
    """Select GPUs to use based on arguments."""
    available_gpus = get_available_gpus()
    if not available_gpus:
        print("No GPUs available. Running on CPU.")
        return []

    if gpus_arg.lower() == 'all':
        selected = available_gpus
    else:
        requested = [int(x.strip()) for x in gpus_arg.split(',')]
        selected = [g for g in requested if g in available_gpus]
    
    if not selected:
        print("WARNING: No valid GPUs selected. Running on CPU.")
    else:
        print(f"Using GPUs: {selected}")
    return selected


def split_data_for_gpus(gpu_ids, parts_per_gpu):
    """Determine how to split the data across GPUs."""
    if not gpu_ids:
        # If no GPUs, don't split
        num_parts = 1
    else:
        # Calculate number of parts based on GPUs and parts per GPU
        num_parts = len(gpu_ids) * parts_per_gpu

    return num_parts


def create_shared_output_store(args, input_shape, patch_size):
    """Create the shared output Zarr store with group hierarchy that all GPU processes will write to."""
    if len(input_shape) == 4:  # has channel dimension
        original_volume_shape = list(input_shape[1:])
    else:  # no channel dimension
        original_volume_shape = list(input_shape)
    
    # Create main output path
    main_store_path = args.output_dir
    if not main_store_path.endswith('.zarr'):
        main_store_path = main_store_path + '.zarr'
    
    # Shape is (6 channels, Z, Y, X) for the full volume
    output_shape = (6, *original_volume_shape)
    
    # Use the patch size for chunking
    output_chunks = (6, *patch_size)
    
    # Get compressor
    if args.zarr_compressor.lower() == 'zstd':
        compressor = zarr.Blosc(cname='zstd', clevel=args.zarr_compression_level, shuffle=zarr.Blosc.SHUFFLE)
    elif args.zarr_compressor.lower() == 'lz4':
        compressor = zarr.Blosc(cname='lz4', clevel=args.zarr_compression_level, shuffle=zarr.Blosc.SHUFFLE)
    elif args.zarr_compressor.lower() == 'zlib':
        compressor = zarr.Blosc(cname='zlib', clevel=args.zarr_compression_level, shuffle=zarr.Blosc.SHUFFLE)
    elif args.zarr_compressor.lower() == 'none':
        compressor = None
    else:
        compressor = zarr.Blosc(cname='zstd', clevel=1, shuffle=zarr.Blosc.SHUFFLE)
    
    print(f"Creating shared output store at: {main_store_path}")
    print(f"Full volume shape: {output_shape}")
    print(f"Chunk shape: {output_chunks}")
    
    # Create the root group
    root_store = zarr.open_group(
        main_store_path,
        mode='w',
        storage_options={'anon': False} if main_store_path.startswith('s3://') else None
    )
    
    # Create the structure_tensor array within the group
    structure_tensor_arr = root_store.create_dataset(
        'structure_tensor',
        shape=output_shape,
        chunks=output_chunks,
        dtype=np.float32,
        compressor=compressor,
        write_empty_chunks=False
    )
    
    # Store metadata in the root group
    try:
        root_store.attrs['patch_size'] = list(patch_size)
        root_store.attrs['overlap'] = args.overlap
        root_store.attrs['sigma'] = args.sigma
        root_store.attrs['smooth_components'] = args.smooth_components
        root_store.attrs['original_volume_shape'] = original_volume_shape
    except Exception as e:
        print(f"Warning: Failed to write custom attributes: {e}")
    
    print(f"Created shared output store with group hierarchy: {main_store_path}")
    return main_store_path


def run_structure_tensor_part(args, part_id, gpu_id, shared_output_path):
    """Run the structure tensor computation for a single part."""
    # Import the actual create_st module
    from structure_tensor import create_st
    
    # Build command to run create_st.py's main() directly
    cmd = [sys.executable, '-m', 'structure_tensor.create_st']
    
    # Basic arguments
    cmd.extend(['--input_dir', args.input_dir])
    # Pass the shared output path directly so each process writes to the same zarr store
    cmd.extend(['--output_dir', shared_output_path])
    cmd.extend(['--mode', 'structure-tensor'])
    
    # Part-specific arguments for multi-GPU
    cmd.extend(['--num_parts', str(args.num_parts)])
    cmd.extend(['--part_id', str(part_id)])
    
    # Add device argument
    if gpu_id is not None:
        cmd.extend(['--device', f'cuda:{gpu_id}'])
    else:
        cmd.extend(['--device', 'cpu'])
    
    # Structure tensor specific arguments
    cmd.extend(['--sigma', str(args.sigma)])
    if args.smooth_components:
        cmd.append('--smooth-components')
    if args.structure_tensor_only:
        cmd.append('--structure-tensor-only')
    if args.volume is not None:
        cmd.extend(['--volume', str(args.volume)])
    
    # Set overlap to 0.0 for no overlap
    cmd.extend(['--overlap', '0.0'])
    
    # Add other optional arguments
    if args.patch_size:
        cmd.extend(['--patch_size', args.patch_size])
    
    # Performance settings
    cmd.extend(['--batch_size', str(args.batch_size)])
    cmd.extend(['--num-workers', str(args.num_workers)])
    
    # Compression settings
    cmd.extend(['--zarr-compressor', args.zarr_compressor])
    cmd.extend(['--zarr-compression-level', str(args.zarr_compression_level)])
    

    if args.verbose:
        cmd.append('--verbose')
    
    print(f"Running Part {part_id} on GPU {gpu_id if gpu_id is not None else 'CPU'}: {' '.join(cmd)}")
    
    # Run with live stdout/stderr streaming for progress bars
    proc = subprocess.Popen(
        cmd,
        stdout=None,  # Let stdout pass through
        stderr=None,  # Let stderr pass through
        universal_newlines=True,
        bufsize=1  # Line buffered
    )
    
    # Wait for the process to complete
    rc = proc.wait()
    if rc != 0:
        print(f"[Part {part_id}] FAILED with return code {rc}")
    return rc == 0


def run_eigenanalysis(zarr_path, chunk_size, compressor, verbose, swap_eigenvectors, num_workers):
    """Run eigenanalysis on a structure tensor zarr to compute eigenvectors."""
    print("\n--- Running Eigenanalysis ---")
    
    # Build command to run create_st.py's eigenanalysis mode
    cmd = [sys.executable, '-m', 'structure_tensor.create_st']
    
    # Basic arguments
    cmd.extend(['--mode', 'eigenanalysis'])
    cmd.extend(['--eigen-input', zarr_path])
    
    # Optional arguments
    if chunk_size:
        cmd.extend(['--chunk-size', chunk_size])
    
    if swap_eigenvectors:
        cmd.append('--swap-eigenvectors')
    
    # Compression settings
    cmd.extend(['--zarr-compressor', compressor.cname if hasattr(compressor, 'cname') else 'zstd'])
    if hasattr(compressor, 'clevel'):
        cmd.extend(['--zarr-compression-level', str(compressor.clevel)])
    
    cmd.extend(['--num-workers', str(num_workers)])
    
    if verbose:
        cmd.append('--verbose')
    
    print(f"Running eigenanalysis: {' '.join(cmd)}")
    
    # Run with live stdout/stderr streaming
    proc = subprocess.Popen(
        cmd,
        stdout=None,
        stderr=None,
        universal_newlines=True,
        bufsize=1
    )
    
    rc = proc.wait()
    if rc != 0:
        print(f"Eigenanalysis FAILED with return code {rc}")
        return False
    
    print("Eigenanalysis completed successfully")
    return True


def delete_intermediate_file(file_path, verbose=False):
    """Delete an intermediate file or directory, handling both local and S3 paths."""
    try:
        if file_path.startswith('s3://'):
            fs = fsspec.filesystem(file_path.split('://')[0], anon=False)
            fs.rm(file_path, recursive=True)
        else:
            if os.path.isdir(file_path):
                shutil.rmtree(file_path, ignore_errors=True)
            elif os.path.isfile(file_path):
                os.remove(file_path)
        
        if verbose:
            print(f"Deleted intermediate file: {file_path}")
        return True
    except Exception as e:
        print(f"Warning: Failed to delete {file_path}: {e}")
        return False


def parse_arguments():
    parser = argparse.ArgumentParser(description='Run create_st with multi-GPU support')
    
    # Mode selection
    parser.add_argument('--mode', type=str, default='structure-tensor',
                        choices=['structure-tensor', 'eigenanalysis'],
                        help='Mode of operation: compute structure tensor (with eigenanalysis by default) or perform eigenanalysis only')
    
    # Basic I/O arguments
    parser.add_argument('--input_dir', type=str, required=True, 
                        help='Path to the input Zarr volume')
    parser.add_argument('--output_dir', type=str, required=True, 
                        help='Path to store output results')
    
    # Structure tensor computation arguments
    parser.add_argument('--sigma', type=float, default=2.0,
                        help='Gaussian σ for structure-tensor smoothing')
    parser.add_argument('--structure-tensor-only', action='store_true',
                        help='Compute only the structure tensor, skip eigenanalysis')
    parser.add_argument('--smooth-components', action='store_true',
                        help='After computing Jxx…Jzz, apply a second Gaussian smoothing to each channel')
    parser.add_argument('--volume', type=int, default=None,
                        help='Volume ID for fiber-volume masking')
    
    # Patch processing arguments
    parser.add_argument('--patch_size', type=str, default=None, 
                        help='Override patch size, comma-separated (e.g., "192,192,192")')
    parser.add_argument('--overlap', type=float, default=0.0, 
                        help='Overlap between patches (0-1), default 0.0 for structure tensor')
    parser.add_argument('--batch_size', type=int, default=1, 
                        help='Batch size for inference')
    
    # Eigenanalysis arguments
    parser.add_argument('--eigen-input', type=str, default=None,
                        help='Input path for eigenanalysis (6-channel structure tensor zarr)')
    parser.add_argument('--eigen-output', type=str, default=None,
                        help='Output path for eigenvectors')
    parser.add_argument('--chunk-size', type=str, default=None,
                        help='Chunk size for eigenanalysis, comma-separated (e.g., "64,64,64")')
    parser.add_argument('--swap-eigenvectors', action='store_true',
                        help='Swap eigenvectors 0 and 1')
    parser.add_argument('--delete-intermediate', action='store_true',
                        help='Delete intermediate structure tensor after eigenanalysis')
    
    # Multi-GPU arguments
    parser.add_argument('--gpus', type=str, default='all',
                        help='GPU IDs to use, comma-separated (e.g., "0,1,2") or "all" for all available GPUs')
    parser.add_argument('--parts-per-gpu', type=int, default=1,
                        help='Number of parts to process per GPU')
    
    # Other arguments
    parser.add_argument('--verbose', action='store_true', 
                        help='Enable verbose output')
    parser.add_argument('--zarr-compressor', type=str, default='zstd',
                        choices=['zstd', 'lz4', 'zlib', 'none'],
                        help='Zarr compression algorithm')
    parser.add_argument('--zarr-compression-level', type=int, default=3,
                        help='Compression level (1-9)')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='Number of workers for data loading')
    
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    
    # Validation for eigenanalysis mode
    if args.mode == 'eigenanalysis':
        if not args.eigen_input:
            print("Error: --eigen-input must be provided for eigenanalysis mode")
            return 1
    
    # Get compressor for later use
    if args.zarr_compressor.lower() == 'zstd':
        compressor = zarr.Blosc(cname='zstd', clevel=args.zarr_compression_level, shuffle=zarr.Blosc.SHUFFLE)
    elif args.zarr_compressor.lower() == 'lz4':
        compressor = zarr.Blosc(cname='lz4', clevel=args.zarr_compression_level, shuffle=zarr.Blosc.SHUFFLE)
    elif args.zarr_compressor.lower() == 'zlib':
        compressor = zarr.Blosc(cname='zlib', clevel=args.zarr_compression_level, shuffle=zarr.Blosc.SHUFFLE)
    elif args.zarr_compressor.lower() == 'none':
        compressor = None
    else:
        compressor = zarr.Blosc(cname='zstd', clevel=1, shuffle=zarr.Blosc.SHUFFLE)
    
    # Parse chunk size for eigenanalysis if provided
    chunk_size_str = None
    if args.chunk_size:
        chunk_size_str = args.chunk_size
    
    # Handle eigenanalysis-only mode
    if args.mode == 'eigenanalysis':
        success = run_eigenanalysis(
            zarr_path=args.eigen_input,
            chunk_size=chunk_size_str,
            compressor=compressor,
            verbose=args.verbose,
            swap_eigenvectors=args.swap_eigenvectors,
            num_workers=args.num_workers
        )
        
        if success:
            print("\n--- Eigenanalysis Completed Successfully ---")
            print(f"Results saved to:")
            print(f"  - Eigenvectors: {args.eigen_input}/eigenvectors")
            print(f"  - Eigenvalues: {args.eigen_input}/eigenvalues")
            return 0
        else:
            print("\n--- Eigenanalysis Failed ---")
            return 1
    
    # For structure-tensor mode, compute the structure tensor
    structure_tensor_output = None
    
    if args.mode == 'structure-tensor':
        # Select GPUs
        gpu_ids = select_gpus(args.gpus)
        if args.verbose:
            print(f"Available GPUs: {get_available_gpus()}")
            print(f"Selected GPUs: {gpu_ids}")
        
        # Determine number of parts
        if len(gpu_ids) > 1:
            num_parts = split_data_for_gpus(gpu_ids, args.parts_per_gpu)
            args.num_parts = num_parts
            print(f"Auto-detected {len(gpu_ids)} GPUs, will process in {num_parts} parts")
        else:
            # Single GPU or CPU mode
            num_parts = 1
            args.num_parts = 1
        
        # Get input shape for creating shared output store
        input_store = open_zarr(
            path=args.input_dir,
            mode='r',
            storage_options={'anon': False} if args.input_dir.startswith('s3://') else None
        )
        input_shape = input_store.shape
        
        # Parse patch size if provided
        patch_size = None
        if args.patch_size:
            try:
                patch_size = tuple(map(int, args.patch_size.split(',')))
                print(f"Using user-specified patch size: {patch_size}")
            except Exception as e:
                print(f"Error parsing patch_size: {e}")
                print("Using default patch size.")
        
        # Infer patch size if not provided
        if patch_size is None:
            chunks = input_store.chunks
            if len(chunks) == 4:
                patch_size = tuple(chunks[1:])
            elif len(chunks) == 3:
                patch_size = tuple(chunks)
            else:
                raise ValueError(f"Cannot infer patch_size from input chunks={chunks}")
            print(f"Inferred patch_size {patch_size} from input Zarr chunking")
        
        if num_parts > 1:
            # Multi-GPU mode
            print("\n--- Multi-GPU Structure Tensor Processing ---")
            
            # Create the shared output store that all processes will write to
            shared_output_path = create_shared_output_store(args, input_shape, patch_size)
            structure_tensor_output = shared_output_path
            
            # Run parts in parallel as separate processes
            print(f"\n--- Running {num_parts} parts across {len(gpu_ids)} GPUs ---")
            
            # Use ThreadPoolExecutor to manage parallel execution
            with ThreadPoolExecutor(max_workers=min(num_parts, os.cpu_count())) as executor:
                futures = []
                for part_id in range(num_parts):
                    gpu_id = gpu_ids[part_id % len(gpu_ids)]
                    future = executor.submit(run_structure_tensor_part, args, part_id, gpu_id, shared_output_path)
                    futures.append((part_id, future))
                
                # Wait for all parts to complete
                results = []
                for part_id, future in futures:
                    try:
                        success = future.result()
                        results.append((part_id, success))
                        if success:
                            print(f"✓ Part {part_id} completed successfully")
                        else:
                            print(f"✗ Part {part_id} failed")
                    except Exception as e:
                        print(f"✗ Part {part_id} failed with exception: {e}")
                        results.append((part_id, False))
            
            # Check if all parts succeeded
            failed_parts = [part_id for part_id, success in results if not success]
            if failed_parts:
                print(f"\n--- Multi-GPU Processing Failed ---")
                print(f"Failed parts: {failed_parts}")
                return 1
            
            print(f"\n--- Multi-GPU Processing Completed Successfully ---")
            print(f"Output saved to: {shared_output_path}")
            
        else:
            # Single GPU/CPU mode - just run directly
            print("\n--- Single GPU/CPU Structure Tensor Processing ---")
            
            # Prepare output path
            output_path = args.output_dir
            if not output_path.endswith('.zarr'):
                output_path = output_path + '.zarr'
            structure_tensor_output = output_path
            
            # Run single part
            success = run_structure_tensor_part(args, 0, gpu_ids[0] if gpu_ids else None, output_path)
            
            if success:
                print(f"\n--- Structure Tensor Processing Completed Successfully ---")
                print(f"Output saved to: {output_path}")
            else:
                print(f"\n--- Structure Tensor Processing Failed ---")
                return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())