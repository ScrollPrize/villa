#!/usr/bin/env python3
"""
Vesuvius Pipeline - Run the complete inference, blending, and finalization process.
Uses multiple GPUs by assigning different devices to different parts (not DDP).
"""

import argparse
import asyncio
import os
import sys
import json
import shutil
import torch
from tqdm.auto import tqdm
import subprocess
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor


def get_available_gpus():
    """Returns a list of available GPU IDs."""
    try:
        num_gpus = torch.cuda.device_count()
        return list(range(num_gpus))
    except:
        return []


def parse_arguments():
    parser = argparse.ArgumentParser(description="Vesuvius complete inference pipeline (predict, blend, finalize)")
    
    # Input/output arguments
    parser.add_argument('--input', type=str, required=True,
                      help='Path to input volume (zarr or folder with TIFFs)')
    parser.add_argument('--output', type=str, required=True,
                      help='Path for the final output zarr')
    parser.add_argument('--workdir', type=str, 
                      help='Working directory for intermediate files. Defaults to output_path + "_work"')
    
    # Model arguments
    parser.add_argument('--model', type=str, required=True,
                      help='Path to the model directory or checkpoint')
    parser.add_argument('--model-type', type=str, choices=['nnunet', 'custom'], default='nnunet',
                      help='Model type. "nnunet" (default) or "custom".')
    
    # Processing parameters
    parser.add_argument('--mode', type=str, choices=['binary', 'multiclass'], default='binary',
                      help='Processing mode. "binary" for 2-class, "multiclass" for >2 classes. Default: binary')
    parser.add_argument('--threshold', action='store_true',
                      help='Apply thresholding to get binary/class masks instead of probability maps')
    parser.add_argument('--patch-size', type=str, 
                      help='Patch size (z, y, x) separated by commas')
    
    # GPU settings
    parser.add_argument('--gpus', type=str, default='all',
                      help='GPU IDs to use, comma-separated (e.g., "0,1,2") or "all" for all available GPUs. Default: all')
    parser.add_argument('--parts-per-gpu', type=int, default=1,
                      help='Number of parts to process per GPU. Higher values use less GPU memory but take longer. Default: 1')
    
    # Performance settings
    parser.add_argument('--tta-type', type=str, choices=['mirroring', 'rotation'], 
                      help='Test time augmentation type (mirroring or rotation)')
    parser.add_argument('--disable-tta', action='store_true',
                      help='Disable test time augmentation')
    parser.add_argument('--single-part', action='store_true',
                      help='Process as a single part (no splitting for multi-GPU)')
    parser.add_argument('--batch-size', type=int, default=4,
                      help='Batch size for inference. Default: 2')
    parser.add_argument('--num-workers', type=int, default=6,
                      help='Number of data loader workers. Default: 4')
    parser.add_argument('--cache-gb', type=float, default=10.0,
                      help='TensorStore cache pool size in GiB. Default: 10.0')
    
    # Cleanup
    parser.add_argument('--keep-intermediates', action='store_true',
                      help='Keep intermediate files after processing')
    
    # Control flow
    parser.add_argument('--skip-predict', action='store_true',
                      help='Skip the prediction step (use existing prediction outputs)')
    parser.add_argument('--skip-blend', action='store_true',
                      help='Skip the blending step (use existing blended outputs)')
    parser.add_argument('--skip-finalize', action='store_true',
                      help='Skip the finalization step (only generate blended logits)')
    
    # Verbosity
    parser.add_argument('--quiet', action='store_true',
                      help='Reduce verbosity')
    
    return parser.parse_args()


def prepare_directories(args):
    """Prepare directory structure for the pipeline."""
    # Determine the working directory
    if args.workdir is None:
        args.workdir = f"{args.output}_work"
    
    # Create needed directories
    os.makedirs(args.workdir, exist_ok=True)
    
    # Define paths for intermediate outputs
    args.parts_dir = os.path.join(args.workdir, "parts")
    args.blended_path = os.path.join(args.workdir, "blended.zarr")
    
    # Create parts directory
    os.makedirs(args.parts_dir, exist_ok=True)
    
    return args


def select_gpus(args):
    """Select GPUs to use based on arguments."""
    available_gpus = get_available_gpus()
    
    if not available_gpus:
        print("No GPUs available. Running on CPU.")
        return []
    
    if args.gpus.lower() == 'all':
        selected_gpus = available_gpus
    else:
        gpu_ids = [int(x.strip()) for x in args.gpus.split(',')]
        # Filter out invalid GPU IDs
        selected_gpus = [gpu_id for gpu_id in gpu_ids if gpu_id in available_gpus]
        
    if not selected_gpus:
        print("WARNING: No valid GPUs selected. Running on CPU.")
    else:
        print(f"Using GPUs: {selected_gpus}")
    
    return selected_gpus


def split_data_for_gpus(args, gpu_ids):
    """Determine how to split the data across GPUs."""
    if getattr(args, 'single_part', False) or not gpu_ids:
        # If single part or no GPUs, don't split
        num_parts = 1
    else:
        # Calculate number of parts based on GPUs and parts per GPU
        num_parts = len(gpu_ids) * getattr(args, 'parts_per_gpu', 1)
        
    return num_parts


def run_predict(args, part_id, gpu_id, z_min=None, z_max=None):
    """Run the prediction step for a single part."""
    cmd = ['vesuvius.predict']
    
    # Required arguments
    cmd.extend(['--model_path', args.model])
    cmd.extend(['--input_dir', args.input])
    # We'll directly output to the parent directory with the expected naming format
    cmd.extend(['--output_dir', args.parts_dir])
    
    # Add model type argument
    if args.model_type == 'custom':
        # not yet implemented, will be for ink / other models that are not nnunet based
        pass
    
    # Add device argument
    if gpu_id is not None:
        cmd.extend(['--device', f'cuda:{gpu_id}'])
    else:
        cmd.extend(['--device', 'cpu'])
    
    # Add part-specific arguments for multi-GPU
    cmd.extend(['--num_parts', str(args.num_parts)])
    cmd.extend(['--part_id', str(part_id)])
    
    # TTA (Test Time Augmentation) settings
    if getattr(args, 'tta_type', None):
        cmd.extend(['--tta_type', args.tta_type])
    elif getattr(args, 'disable_tta', False):
        cmd.append('--disable_tta')
    # Default behavior will use mirroring if neither is specified
    
    # Add other optional arguments
    if args.patch_size:
        cmd.extend(['--patch_size', args.patch_size])
    
    # Performance settings
    cmd.extend(['--batch_size', str(args.batch_size)])
    
    if args.quiet:
        cmd.append('--quiet')
        
    # Run the command
    print(f"Running Part {part_id} on GPU {gpu_id if gpu_id is not None else 'CPU'}: {' '.join(cmd)}")
    
    # Run with live stdout/stderr streaming for progress bars
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE if args.quiet else None,
        stderr=subprocess.PIPE if args.quiet else None,
        universal_newlines=True,
        bufsize=1  # Line buffered
    )
    
    # Wait for the process to complete
    returncode = process.wait()
    
    if returncode != 0:
        if args.quiet:
            stderr = process.stderr.read() if process.stderr else "No error output available"
            print(f"Error running prediction for part {part_id}:")
            print(stderr)
        return False
    
    return True


def run_blend(args):
    """Run the blending step to merge all parts."""
    cmd = ['vesuvius.blend_logits', args.parts_dir, args.blended_path]
    
    # Add optional arguments
    cmd.extend(['--cache_gb', str(args.cache_gb)])
    
    if args.quiet:
        cmd.append('--quiet')
    
    # Run the command
    print(f"Blending parts: {' '.join(cmd)}")
    
    # Run with live stdout/stderr streaming for progress bars
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE if args.quiet else None,
        stderr=subprocess.PIPE if args.quiet else None,
        universal_newlines=True,
        bufsize=1  # Line buffered
    )
    
    # Wait for the process to complete
    returncode = process.wait()
    
    if returncode != 0:
        if args.quiet:
            stderr = process.stderr.read() if process.stderr else "No error output available"
            print("Error blending parts:")
            print(stderr)
        return False
    
    return True


def run_finalize(args):
    """Run the finalization step to process the blended output."""
    cmd = ['vesuvius.finalize_outputs', args.blended_path, args.output]
    
    # Add mode and threshold arguments
    cmd.extend(['--mode', args.mode])
    if args.threshold:
        cmd.append('--threshold')
    
    # Delete intermediates if not keeping them
    if not args.keep_intermediates:
        cmd.append('--delete-intermediates')
    
    # Use hyphenated format for finalize command
    cmd.extend(['--cache-gb', str(args.cache_gb)])
    
    if args.quiet:
        cmd.append('--quiet')
    
    # Run the command
    print(f"Finalizing output: {' '.join(cmd)}")
    
    # Run with live stdout/stderr streaming for progress bars
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE if args.quiet else None,
        stderr=subprocess.PIPE if args.quiet else None,
        universal_newlines=True,
        bufsize=1  # Line buffered
    )
    
    # Wait for the process to complete
    returncode = process.wait()
    
    if returncode != 0:
        if args.quiet:
            stderr = process.stderr.read() if process.stderr else "No error output available"
            print("Error finalizing output:")
            print(stderr)
        return False
    
    return True


def cleanup(args):
    """Clean up intermediate files."""
    if not args.keep_intermediates:
        print("Cleaning up intermediate files...")
        if os.path.exists(args.parts_dir):
            shutil.rmtree(args.parts_dir)
        
        # Only remove the work directory if it's empty
        try:
            os.rmdir(args.workdir)
        except OSError:
            # Directory not empty, so keep it
            pass


def setup_multipart(args, num_parts):
    """Setup for multi-part processing."""
    # No need to calculate Z-ranges since vesuvius.predict handles
    # the data splitting internally based on num_parts and part_id
    
    # Just validate and return the number of parts
    if num_parts < 1:
        print("Warning: Number of parts must be at least 1. Setting to 1.")
        num_parts = 1
    
    print(f"Setting up for processing in {num_parts} parts...")
    
    # Store the num_parts as an attribute on args for use in run_predict
    args.num_parts = num_parts
    
    # We're not returning Z-ranges anymore since vesuvius.predict
    # will handle partitioning internally
    return num_parts


def run_pipeline():
    """Run the complete inference pipeline."""
    args = parse_arguments()
    
    # Convert hyphenated argument names to underscore format for code access
    # Store both original hyphenated name and Python-friendly underscore name
    for attr_name in dir(args):
        if '-' in attr_name:
            underscore_name = attr_name.replace('-', '_')
            setattr(args, underscore_name, getattr(args, attr_name))
    
    # Now we can access args.tta_type instead of args.tta-type
    args = prepare_directories(args)
    
    # Select GPUs to use
    gpu_ids = select_gpus(args)
    
    # Determine number of parts for multi-GPU processing
    num_parts = setup_multipart(args, split_data_for_gpus(args, gpu_ids))
    
    # Prediction step
    if not args.skip_predict:
        print(f"\n--- Step 1: Prediction ({num_parts} parts) ---")
        
        if num_parts > 1:
            # For multi-part processing, we'll run each part with a specific part_id
            # Use a thread pool to run predictions in parallel
            with ThreadPoolExecutor(max_workers=min(num_parts, os.cpu_count())) as executor:
                futures = []
                
                for part_id in range(num_parts):
                    # Determine GPU for this part
                    if gpu_ids:
                        gpu_id = gpu_ids[part_id % len(gpu_ids)]
                    else:
                        gpu_id = None
                        
                    # Submit prediction task - no Z range needed now
                    futures.append(
                        executor.submit(run_predict, args, part_id, gpu_id)
                    )
                
                # Wait for all predictions to complete
                all_success = True
                for part_id, future in enumerate(futures):
                    try:
                        success = future.result()
                        if not success:
                            all_success = False
                            print(f"Prediction for part {part_id} failed.")
                    except Exception as e:
                        all_success = False
                        print(f"Prediction for part {part_id} raised an exception: {e}")
                
                if not all_success:
                    print("One or more prediction tasks failed. Aborting pipeline.")
                    return 1
        
        else:
            # Just run a single prediction
            gpu_id = gpu_ids[0] if gpu_ids else None
            success = run_predict(args, 0, gpu_id)
            if not success:
                print("Prediction failed. Aborting pipeline.")
                return 1
    
    # Blending step
    if not args.skip_blend:
        print("\n--- Step 2: Blending ---")
        if not os.path.exists(args.parts_dir) or not os.listdir(args.parts_dir):
            print("No prediction parts found. Please run the prediction step first.")
            return 1
        
        success = run_blend(args)
        if not success:
            print("Blending failed. Aborting pipeline.")
            return 1
    
    # Finalization step
    if not args.skip_finalize:
        print("\n--- Step 3: Finalization ---")
        if not os.path.exists(args.blended_path):
            print("No blended data found. Please run the blending step first.")
            return 1
        
        success = run_finalize(args)
        if not success:
            print("Finalization failed.")
            return 1
    
    # Final cleanup
    cleanup(args)
    
    print(f"\n--- Pipeline Complete ---")
    print(f"Final output saved to: {args.output}")
    
    return 0


if __name__ == "__main__":
    sys.exit(run_pipeline())