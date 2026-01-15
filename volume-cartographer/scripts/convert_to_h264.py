#!/usr/bin/env python3
"""
Convert an OME-Zarr volume to H.264 compressed OME-Zarr (Zarr v2).

This script reads an existing OME-Zarr multiscale volume and creates a new
OME-Zarr volume compressed with H.264 via blosc2.

Usage:
    python convert_to_h264.py input.zarr output.zarr
    python convert_to_h264.py input.zarr output.zarr --chunk-size 64
    python convert_to_h264.py input.zarr output.zarr --levels 0,1,2

Requirements:
    - python-blosc2 linked against c-blosc2 with openh264
    - Vendored numcodecs and zarr-python with Blosc2/openh264 support
"""

import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import numpy as np
from numcodecs.blosc2 import Blosc2


def find_cubic_chunk_size(shape: tuple[int, ...], target_size: int = 128) -> int:
    """
    Find a good cubic chunk size for the given shape.

    For openh264, chunks must be:
    - Cubic (NxNxN)
    - Even (N % 2 == 0)
    - Reasonable size for compression efficiency
    """
    # Ensure even
    if target_size % 2 != 0:
        target_size = target_size + 1

    # Find minimum dimension
    min_dim = min(shape)

    # Don't exceed smallest dimension
    if target_size > min_dim:
        target_size = min_dim
        # Make even
        if target_size % 2 != 0:
            target_size = target_size - 1

    # Minimum practical size
    if target_size < 16:
        target_size = 16

    return target_size


def get_chunk_path(array_path: Path, z: int, y: int, x: int) -> Path:
    """Get the path to a chunk file in Zarr v2 format with / separator."""
    return array_path / str(z) / str(y) / str(x)


def convert_array(
    src_array_path: Path,
    dst_array_path: Path,
    chunk_size: int,
    qp: int = 26,
    threads: int | None = None,
) -> dict:
    """
    Convert a single zarr v2 array to H.264 compressed format.

    Supports resume - if stopped and restarted, will skip already completed chunks.
    """
    import zarr

    # Read source .zarray metadata
    with open(src_array_path / '.zarray', 'r') as f:
        src_meta = json.load(f)

    shape = tuple(src_meta['shape'])
    dtype = np.dtype(src_meta['dtype'])
    src_chunks = tuple(src_meta['chunks'])

    print(f"  Shape: {shape}")
    print(f"  Dtype: {dtype}")
    print(f"  Input chunks: {src_chunks}")

    # Validate dtype
    if dtype != np.uint8:
        print(f"  WARNING: H.264 works best with uint8 data. Source dtype is {dtype}")

    # Determine output chunk size
    if chunk_size is None:
        chunk_size = find_cubic_chunk_size(shape)

    output_chunks = (chunk_size, chunk_size, chunk_size)
    print(f"  Output chunks: {output_chunks}")
    print(f"  Compression: blosc2/openh264 (qp={qp})")

    # Create blosc2 codec with openh264
    compressor = Blosc2(cname='openh264', qp=qp)

    # Create output .zarray metadata
    dst_meta = {
        "chunks": list(output_chunks),
        "compressor": compressor.get_config(),
        "dimension_separator": "/",
        "dtype": src_meta['dtype'],
        "fill_value": src_meta.get('fill_value', 0),
        "filters": None,
        "order": "C",
        "shape": list(shape),
        "zarr_format": 2
    }

    # Create output directory and write metadata
    dst_array_path.mkdir(parents=True, exist_ok=True)
    with open(dst_array_path / '.zarray', 'w') as f:
        json.dump(dst_meta, f, indent=4)

    # Open source array for reading
    src = zarr.open_array(str(src_array_path), mode='r')

    # Calculate number of output chunks
    n_out_chunks = [
        (s + chunk_size - 1) // chunk_size
        for s in shape
    ]
    total_out_chunks = n_out_chunks[0] * n_out_chunks[1] * n_out_chunks[2]

    # Calculate input chunk info
    in_chunk_size = src_chunks
    n_in_chunks = [
        (s + c - 1) // c
        for s, c in zip(shape, in_chunk_size)
    ]

    # Determine number of workers
    n_workers = threads if threads else (os.cpu_count()//2 or 4)

    # Build list of INPUT chunk coordinates to process
    # Group output chunks by their parent input chunk for efficient I/O
    input_chunk_coords = []
    skipped_out_chunks = 0
    total_out_to_process = 0

    for izi in range(n_in_chunks[0]):
        for iyi in range(n_in_chunks[1]):
            for ixi in range(n_in_chunks[2]):
                # Calculate which output chunks this input chunk covers
                oz_start = izi * in_chunk_size[0] // chunk_size
                oz_end = min(((izi + 1) * in_chunk_size[0] + chunk_size - 1) // chunk_size, n_out_chunks[0])
                oy_start = iyi * in_chunk_size[1] // chunk_size
                oy_end = min(((iyi + 1) * in_chunk_size[1] + chunk_size - 1) // chunk_size, n_out_chunks[1])
                ox_start = ixi * in_chunk_size[2] // chunk_size
                ox_end = min(((ixi + 1) * in_chunk_size[2] + chunk_size - 1) // chunk_size, n_out_chunks[2])

                # Check which output chunks in this region still need processing
                pending_outputs = []
                for ozi in range(oz_start, oz_end):
                    for oyi in range(oy_start, oy_end):
                        for oxi in range(ox_start, ox_end):
                            chunk_path = get_chunk_path(dst_array_path, ozi, oyi, oxi)
                            if chunk_path.exists():
                                skipped_out_chunks += 1
                            else:
                                pending_outputs.append((ozi, oyi, oxi))

                if pending_outputs:
                    input_chunk_coords.append((izi, iyi, ixi, pending_outputs))
                    total_out_to_process += len(pending_outputs)

    if skipped_out_chunks > 0:
        print(f"  Resuming: {skipped_out_chunks} chunks done, {total_out_to_process} remaining")

    if not input_chunk_coords:
        print("  All chunks already completed!")
        return {
            'shape': shape,
            'chunks_total': total_out_chunks,
            'chunks_skipped': skipped_out_chunks,
        }

    print(f"  Processing {total_out_to_process} chunks using {n_workers} threads...")

    # Progress tracking
    progress_lock = Lock()
    out_chunks_done = [0]
    bytes_written = [0]
    start_time = time.time()

    def process_input_chunk(args):
        """Process one input chunk, writing multiple output chunks."""
        izi, iyi, ixi, pending_outputs = args

        # Calculate input chunk bounds in array coordinates
        iz_start = izi * in_chunk_size[0]
        iz_end = min(iz_start + in_chunk_size[0], shape[0])
        iy_start = iyi * in_chunk_size[1]
        iy_end = min(iy_start + in_chunk_size[1], shape[1])
        ix_start = ixi * in_chunk_size[2]
        ix_end = min(ix_start + in_chunk_size[2], shape[2])

        # Read the entire input chunk ONCE
        input_data = src[iz_start:iz_end, iy_start:iy_end, ix_start:ix_end]

        chunks_written = 0
        bytes_total = 0

        # Write each pending output chunk
        for ozi, oyi, oxi in pending_outputs:
            # Calculate output chunk bounds in array coordinates
            oz_start = ozi * chunk_size
            oz_end = min(oz_start + chunk_size, shape[0])
            oy_start = oyi * chunk_size
            oy_end = min(oy_start + chunk_size, shape[1])
            ox_start = oxi * chunk_size
            ox_end = min(ox_start + chunk_size, shape[2])

            # Calculate the slice within the input_data array
            local_z_start = oz_start - iz_start
            local_z_end = oz_end - iz_start
            local_y_start = oy_start - iy_start
            local_y_end = oy_end - iy_start
            local_x_start = ox_start - ix_start
            local_x_end = ox_end - ix_start

            # Extract the output chunk from input data
            chunk_data = input_data[local_z_start:local_z_end,
                                    local_y_start:local_y_end,
                                    local_x_start:local_x_end]

            # Pad if needed (for edge chunks)
            if chunk_data.shape != output_chunks:
                padded = np.zeros(output_chunks, dtype=dtype)
                padded[:chunk_data.shape[0], :chunk_data.shape[1], :chunk_data.shape[2]] = chunk_data
                chunk_data = padded

            # Ensure contiguous memory for blosc2 encoder
            chunk_data = np.ascontiguousarray(chunk_data)

            # Compress with blosc2/openh264
            compressed = compressor.encode(chunk_data)

            # Write chunk file
            chunk_path = get_chunk_path(dst_array_path, ozi, oyi, oxi)
            chunk_path.parent.mkdir(parents=True, exist_ok=True)
            with open(chunk_path, 'wb') as f:
                f.write(compressed)

            chunks_written += 1
            bytes_total += chunk_data.nbytes

        return chunks_written, bytes_total

    # Process input chunks in parallel
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_input_chunk, args): args for args in input_chunk_coords}

        for future in as_completed(futures):
            chunks_written, nbytes = future.result()

            with progress_lock:
                out_chunks_done[0] += chunks_written
                bytes_written[0] += nbytes

                # Progress update
                elapsed = time.time() - start_time
                rate = bytes_written[0] / elapsed / 1e6  # MB/s
                pct = 100 * (out_chunks_done[0] + skipped_out_chunks) / total_out_chunks
                print(f"\r  Progress: {out_chunks_done[0] + skipped_out_chunks}/{total_out_chunks} ({pct:.1f}%) - {rate:.1f} MB/s", end='', flush=True)

    print()  # Newline after progress

    elapsed = time.time() - start_time

    return {
        'shape': shape,
        'chunks_total': total_out_chunks,
        'chunks_processed': out_chunks_done[0],
        'chunks_skipped': skipped_out_chunks,
        'elapsed_seconds': elapsed,
        'throughput_mb_s': bytes_written[0] / elapsed / 1e6 if elapsed > 0 else 0,
    }


def convert_ome_zarr(
    input_path: str,
    output_path: str,
    chunk_size: int | None = None,
    qp: int = 26,
    threads: int | None = None,
    levels: list[int] | None = None,
) -> dict:
    """
    Convert an OME-Zarr multiscale volume to H.264 compressed format.

    Preserves the OME-Zarr structure including .zattrs, .zgroup, and all
    resolution levels.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    print(f"Converting OME-Zarr: {input_path} -> {output_path}")

    # Check if it's an OME-Zarr (has .zattrs with multiscales)
    zattrs_path = input_path / '.zattrs'
    if not zattrs_path.exists():
        raise ValueError(f"Not an OME-Zarr: missing .zattrs at {input_path}")

    with open(zattrs_path, 'r') as f:
        zattrs = json.load(f)

    if 'multiscales' not in zattrs:
        raise ValueError(f"Not an OME-Zarr: .zattrs missing 'multiscales' key")

    multiscales = zattrs['multiscales'][0]
    datasets = multiscales['datasets']

    print(f"Found {len(datasets)} resolution levels: {[d['path'] for d in datasets]}")

    # Determine which levels to convert
    if levels is None:
        levels_to_convert = [d['path'] for d in datasets]
    else:
        levels_to_convert = [str(l) for l in levels]

    print(f"Converting levels: {levels_to_convert}")

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Copy group metadata
    shutil.copy(input_path / '.zattrs', output_path / '.zattrs')
    shutil.copy(input_path / '.zgroup', output_path / '.zgroup')

    # Copy any other metadata files (like meta.json)
    for f in input_path.iterdir():
        if f.is_file() and f.name not in ['.zattrs', '.zgroup']:
            shutil.copy(f, output_path / f.name)

    # Convert each level
    results = {}
    for level in levels_to_convert:
        src_array = input_path / level
        dst_array = output_path / level

        if not src_array.exists():
            print(f"\nSkipping level {level}: not found")
            continue

        print(f"\nConverting level {level}:")

        # For lower resolution levels, scale the chunk size proportionally
        level_int = int(level)
        if chunk_size is not None:
            # Scale chunk size with resolution level
            # Level 0 = full res, level 1 = 2x downsampled, etc.
            level_chunk_size = max(16, chunk_size // (2 ** level_int))
            # Ensure even
            if level_chunk_size % 2 != 0:
                level_chunk_size = level_chunk_size + 1
        else:
            level_chunk_size = None

        result = convert_array(
            src_array_path=src_array,
            dst_array_path=dst_array,
            chunk_size=level_chunk_size if level_chunk_size else chunk_size,
            qp=qp,
            threads=threads,
        )
        results[level] = result

    # Calculate total sizes
    input_size = sum(f.stat().st_size for f in input_path.rglob('*') if f.is_file())
    output_size = sum(f.stat().st_size for f in output_path.rglob('*') if f.is_file())

    print(f"\n{'='*60}")
    print(f"Conversion complete!")
    print(f"  Input size:  {input_size:,} bytes ({input_size/1e9:.2f} GB)")
    print(f"  Output size: {output_size:,} bytes ({output_size/1e6:.2f} MB)")
    print(f"  Compression ratio: {input_size / output_size if output_size > 0 else 0:.1f}x")

    return {
        'input_path': str(input_path),
        'output_path': str(output_path),
        'levels': results,
        'input_size_bytes': input_size,
        'output_size_bytes': output_size,
        'compression_ratio': input_size / output_size if output_size > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Convert an OME-Zarr volume to H.264 compressed OME-Zarr',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Convert all levels (default QP=26)
    python convert_to_h264.py input.zarr output_h264.zarr

    # Specify chunk size for level 0 (scaled for other levels)
    python convert_to_h264.py input.zarr output.zarr --chunk-size 64

    # Convert only specific levels
    python convert_to_h264.py input.zarr output.zarr --levels 0,1,2

    # Higher compression with QP=39 (lower quality)
    python convert_to_h264.py input.zarr output.zarr --qp 39

    # Use 8 threads
    python convert_to_h264.py input.zarr output.zarr --threads 8

Notes:
    - Outputs OME-Zarr v2 format (compatible with standard viewers)
    - H.264 compression requires cubic chunks with even dimensions
    - Works best with uint8 data (grayscale volumetric data)
    - H.264 is lossy - QP controls quality/compression tradeoff
    - QP range: 0 (best quality) to 51 (highest compression)
    - Typical values: 18-28 for good quality, 30-40 for high compression
    - Supports resume: re-run the same command to continue from where you left off
"""
    )

    parser.add_argument('input', help='Input OME-Zarr path')
    parser.add_argument('output', help='Output OME-Zarr path')
    parser.add_argument('--chunk-size', '-c', type=int, default=None,
                        help='Cubic chunk size for level 0 (must be even). Default: auto-detect')
    parser.add_argument('--levels', '-l', type=str, default=None,
                        help='Comma-separated list of levels to convert (e.g., "0,1,2"). Default: all')
    parser.add_argument('--qp', '-q', type=int, default=26,
                        help='H.264 quantization parameter 0-51 (default: 26). Lower=better quality.')
    parser.add_argument('--threads', '-t', type=int, default=None,
                        help='Number of threads (default: all CPU cores)')

    args = parser.parse_args()

    # Validate chunk size if provided
    if args.chunk_size is not None and args.chunk_size % 2 != 0:
        parser.error(f"Chunk size must be even. Got {args.chunk_size}")

    # Validate QP range
    if not (0 <= args.qp <= 51):
        parser.error(f"QP must be 0-51. Got {args.qp}")

    # Parse levels
    levels = None
    if args.levels:
        levels = [int(l.strip()) for l in args.levels.split(',')]

    try:
        stats = convert_ome_zarr(
            input_path=args.input,
            output_path=args.output,
            chunk_size=args.chunk_size,
            qp=args.qp,
            threads=args.threads,
            levels=levels,
        )
        return 0
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
