#!/usr/bin/env python3
"""
Resize an OME-Zarr to match the shape of a reference OME-Zarr using cv2 resize.
Processes all resolution levels and works chunk-by-chunk.
"""

import argparse
import json
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import zarr
from tqdm import tqdm


def resize_chunk_3d(data, target_shape, interpolation):
    """Resize a 3D chunk by resizing each z-slice with cv2."""
    target_z, target_y, target_x = target_shape

    # First resize in YX for each z-slice
    resized_yx = np.zeros((data.shape[0], target_y, target_x), dtype=data.dtype)
    for z in range(data.shape[0]):
        resized_yx[z] = cv2.resize(
            data[z],
            (target_x, target_y),  # cv2 uses (width, height)
            interpolation=interpolation
        )

    # Then resize in Z if needed
    if data.shape[0] != target_z:
        # Resize along z axis by resizing each x-column
        resized = np.zeros((target_z, target_y, target_x), dtype=data.dtype)
        for y in range(target_y):
            # Treat the ZX plane as an image and resize
            zx_plane = resized_yx[:, y, :]  # shape: (z, x)
            resized_zx = cv2.resize(
                zx_plane,
                (target_x, target_z),  # cv2 uses (width, height) -> (x, z)
                interpolation=interpolation
            )
            resized[:, y, :] = resized_zx
        return resized

    return resized_yx


def process_chunk(args):
    """Process a single output chunk."""
    (input_path, output_path, resolution,
     oz_start, oy_start, ox_start,
     out_chunk_size, input_shape, output_shape,
     interpolation) = args

    # Open zarr in each worker
    input_store = zarr.open(input_path, mode='r')
    output_store = zarr.open(output_path, mode='r+')

    input_data = input_store[resolution]
    output_data = output_store[resolution]

    # Calculate output chunk bounds
    oz_end = min(oz_start + out_chunk_size[0], output_shape[0])
    oy_end = min(oy_start + out_chunk_size[1], output_shape[1])
    ox_end = min(ox_start + out_chunk_size[2], output_shape[2])

    out_chunk_shape = (oz_end - oz_start, oy_end - oy_start, ox_end - ox_start)

    # Calculate corresponding input region (with float precision)
    scale_z = input_shape[0] / output_shape[0]
    scale_y = input_shape[1] / output_shape[1]
    scale_x = input_shape[2] / output_shape[2]

    iz_start = int(oz_start * scale_z)
    iy_start = int(oy_start * scale_y)
    ix_start = int(ox_start * scale_x)

    iz_end = min(int(np.ceil(oz_end * scale_z)), input_shape[0])
    iy_end = min(int(np.ceil(oy_end * scale_y)), input_shape[1])
    ix_end = min(int(np.ceil(ox_end * scale_x)), input_shape[2])

    # Ensure we have at least 1 pixel
    if iz_end <= iz_start:
        iz_end = iz_start + 1
    if iy_end <= iy_start:
        iy_end = iy_start + 1
    if ix_end <= ix_start:
        ix_end = ix_start + 1

    # Read input region
    input_chunk = input_data[iz_start:iz_end, iy_start:iy_end, ix_start:ix_end]

    # Skip empty chunks (all zeros)
    if not input_chunk.any():
        return None

    # Resize to output chunk shape
    resized = resize_chunk_3d(input_chunk, out_chunk_shape, interpolation)

    # Write to output
    output_data[oz_start:oz_end, oy_start:oy_end, ox_start:ox_end] = resized

    return (oz_start, oy_start, ox_start)


def get_resolutions(zarr_path):
    """Get list of resolution levels from OME-Zarr."""
    store = zarr.open(zarr_path, mode='r')

    # Try to read from .zattrs
    zattrs_path = Path(zarr_path) / '.zattrs'
    if zattrs_path.exists():
        with open(zattrs_path) as f:
            attrs = json.load(f)
        if 'multiscales' in attrs:
            datasets = attrs['multiscales'][0]['datasets']
            return [d['path'] for d in datasets]

    # Fallback: find numeric directories
    resolutions = []
    for item in sorted(Path(zarr_path).iterdir()):
        if item.is_dir() and item.name.isdigit():
            resolutions.append(item.name)
    return resolutions


def main():
    parser = argparse.ArgumentParser(
        description="Resize an OME-Zarr to match the shape of a reference OME-Zarr"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input OME-Zarr file"
    )
    parser.add_argument(
        "--reference",
        type=str,
        required=True,
        help="Path to the reference OME-Zarr file (target shape)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to the output OME-Zarr file"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of worker processes (default: 8)"
    )
    parser.add_argument(
        "--interpolation",
        type=str,
        default="linear",
        choices=["nearest", "linear", "cubic", "area", "lanczos"],
        help="Interpolation method (default: linear)"
    )

    args = parser.parse_args()

    # Map interpolation names to cv2 constants
    interpolation_map = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "area": cv2.INTER_AREA,
        "lanczos": cv2.INTER_LANCZOS4,
    }
    interpolation = interpolation_map[args.interpolation]

    # Get resolutions from input
    resolutions = get_resolutions(args.input)
    print(f"Found {len(resolutions)} resolution levels: {resolutions}")

    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Copy metadata files from input
    input_path = Path(args.input)
    for meta_file in ['.zattrs', '.zgroup']:
        src = input_path / meta_file
        if src.exists():
            shutil.copy(src, output_path / meta_file)
            print(f"Copied {meta_file}")

    # Process each resolution level
    for resolution in resolutions:
        print(f"\n{'='*60}")
        print(f"Processing resolution level: {resolution}")
        print(f"{'='*60}")

        # Open input and reference
        input_store = zarr.open(args.input, mode='r')
        ref_store = zarr.open(args.reference, mode='r')

        input_data = input_store[resolution]
        ref_data = ref_store[resolution]

        input_shape = input_data.shape
        target_shape = ref_data.shape

        print(f"Input shape: {input_shape}")
        print(f"Target shape: {target_shape}")
        print(f"Input dtype: {input_data.dtype}")

        # Get compression and chunks from input
        compressor = input_data.compressor
        input_chunks = input_data.chunks

        # Use input chunks, but cap at target shape
        out_chunks = tuple(min(c, s) for c, s in zip(input_chunks, target_shape))

        print(f"Compressor: {compressor}")
        print(f"Output chunks: {out_chunks}")

        # Create output array for this resolution
        output_store = zarr.open(str(output_path), mode='a')
        output_data = output_store.create_dataset(
            resolution,
            shape=target_shape,
            chunks=out_chunks,
            dtype=input_data.dtype,
            compressor=compressor,
            dimension_separator='/',
            overwrite=True,
        )

        # Generate chunk tasks
        chunk_args = []
        for oz in range(0, target_shape[0], out_chunks[0]):
            for oy in range(0, target_shape[1], out_chunks[1]):
                for ox in range(0, target_shape[2], out_chunks[2]):
                    chunk_args.append((
                        args.input,
                        str(output_path),
                        resolution,
                        oz, oy, ox,
                        out_chunks,
                        input_shape,
                        target_shape,
                        interpolation
                    ))

        total_chunks = len(chunk_args)
        print(f"Processing {total_chunks} chunks with {args.num_workers} workers...")

        # Process chunks in parallel
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {executor.submit(process_chunk, arg): arg for arg in chunk_args}

            with tqdm(total=total_chunks, desc=f"Resolution {resolution}") as pbar:
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        print(f"Error processing chunk: {e}")
                        raise
                    pbar.update(1)

    print(f"\nDone! Output saved to {output_path}")


if __name__ == "__main__":
    main()
