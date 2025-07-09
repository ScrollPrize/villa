#!/usr/bin/env python3
"""
Extract approved patches from a zarr file based on a JSON file from vc_proofreader.

This script reads the approved patch locations from a JSON file and extracts them
from a zarr file, saving them as TIF files with the same naming convention as
the proofreader. The patches are NOT binarized during extraction.

Usage:
    python extract_approved_patches.py --json progress.json --zarr /path/to/label.zarr --output /path/to/output/dir
"""

import os
import json
import argparse
import numpy as np
import zarr
import fsspec
import tifffile
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm


def extract_patch(volume, coord, patch_size):
    """
    Extract a patch from a volume starting at the given coordinate.
    For spatial dimensions, a slice is created from coord to coord+patch_size.
    Any extra dimensions (e.g. channels) are included in full.
    """
    slices = tuple(slice(c, c + patch_size) for c in coord)
    if volume.ndim > len(coord):
        slices = slices + (slice(None),) * (volume.ndim - len(coord))
    return volume[slices]


def process_patch(patch_info, zarr_path, output_dir, patch_size):
    """
    Process a single patch: extract from zarr and save as TIF.
    
    Args:
        patch_info: Dictionary with patch information including 'coords'
        zarr_path: Path to the zarr file
        output_dir: Directory to save the TIF files
        patch_size: Size of the patch to extract
    
    Returns:
        Tuple of (success: bool, error_message: str or None)
    """
    try:
        # Open zarr file (this is lazy loading)
        mapper = fsspec.get_mapper(zarr_path)
        
        # Try to open as array first, then as group
        try:
            volume = zarr.open_array(mapper, mode='r')
        except:
            zarr_group = zarr.open_group(mapper, mode='r')
            # Check if it's an OME-Zarr with resolution levels
            if "0" in zarr_group:
                volume = zarr_group["0"]
            else:
                volume = zarr_group
        
        # Extract coordinates
        coords = patch_info['coords']
        
        # Extract the patch
        patch = extract_patch(volume, coords, patch_size)
        
        # Convert to numpy array if it's a zarr array
        if hasattr(patch, 'compute'):
            patch = patch.compute()
        elif isinstance(patch, zarr.Array):
            patch = np.array(patch)
        
        # Construct filename based on coordinates
        if len(coords) == 3:
            coord_str = f"z{coords[0]}_y{coords[1]}_x{coords[2]}"
        elif len(coords) == 2:
            coord_str = f"y{coords[0]}_x{coords[1]}"
        else:
            coord_str = "_".join(str(c) for c in coords)
        
        filename = f"lbl_{coord_str}.tif"
        filepath = os.path.join(output_dir, filename)
        
        # Save as TIF (without binarization)
        tifffile.imwrite(filepath, patch)
        
        return True, None
        
    except Exception as e:
        return False, f"Error processing patch at {patch_info.get('coords', 'unknown')}: {str(e)}"


def main():
    parser = argparse.ArgumentParser(description='Extract approved patches from zarr based on JSON file')
    parser.add_argument('--json', required=True, help='Path to the progress JSON file')
    parser.add_argument('--zarr', required=True, help='Path to the source zarr file')
    parser.add_argument('--output', required=True, help='Output directory for TIF files')
    parser.add_argument('--workers', type=int, default=None, help='Number of worker processes (default: CPU count)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load JSON file
    print(f"Loading progress file: {args.json}")
    with open(args.json, 'r') as f:
        data = json.load(f)
    
    # Extract metadata
    metadata = data.get('metadata', {})
    patch_size = metadata.get('patch_size')
    if patch_size is None:
        raise ValueError("patch_size not found in JSON metadata")
    
    print(f"Patch size: {patch_size}")
    
    # Get approved patches
    approved_patches = data.get('approved_patches', [])
    if not approved_patches:
        # Fallback to extracting from progress_log
        progress_log = data.get('progress_log', [])
        approved_patches = [entry for entry in progress_log if entry.get('status') == 'approved']
    
    if not approved_patches:
        print("No approved patches found in the JSON file")
        return
    
    print(f"Found {len(approved_patches)} approved patches")
    
    # Verify zarr file exists and is accessible
    print(f"Checking zarr file: {args.zarr}")
    try:
        mapper = fsspec.get_mapper(args.zarr)
        
        # Try to open as array first, then as group
        try:
            test_volume = zarr.open_array(mapper, mode='r')
        except:
            zarr_group = zarr.open_group(mapper, mode='r')
            if "0" in zarr_group:
                test_volume = zarr_group["0"]
            else:
                test_volume = zarr_group
        
        print(f"Zarr shape: {test_volume.shape}, dtype: {test_volume.dtype}")
    except Exception as e:
        raise ValueError(f"Cannot open zarr file: {e}")
    
    # Determine number of workers
    num_workers = args.workers if args.workers else cpu_count()
    print(f"Using {num_workers} worker processes")
    
    # Create partial function with fixed arguments
    process_func = partial(process_patch, 
                          zarr_path=args.zarr, 
                          output_dir=str(output_dir), 
                          patch_size=patch_size)
    
    # Process patches in parallel with progress bar
    successful = 0
    failed = 0
    errors = []
    
    with Pool(num_workers) as pool:
        # Use imap for better memory efficiency and progress tracking
        results = pool.imap(process_func, approved_patches)
        
        # Process results with progress bar
        for success, error in tqdm(results, total=len(approved_patches), desc="Extracting patches"):
            if success:
                successful += 1
            else:
                failed += 1
                if error:
                    errors.append(error)
    
    # Print summary
    print(f"\nExtraction complete:")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    
    if errors:
        print("\nErrors encountered:")
        for error in errors[:10]:  # Show first 10 errors
            print(f"  - {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
    
    print(f"\nOutput files saved to: {output_dir}")


if __name__ == '__main__':
    main()
