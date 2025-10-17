#!/usr/bin/env python3

import argparse
import zarr
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from pathlib import Path
import itertools


def _inverse_permutation(perm):
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return inv


def process_chunk_threshold(args):
    """Process a single output chunk for thresholding."""
    input_path, output_path, chunk_coords, threshold, erase_blank = args

    input_z = zarr.open(input_path, mode='r')
    output_z = zarr.open(output_path, mode='r+')

    slices = tuple(slice(start, stop) for start, stop in chunk_coords)
    chunk_data = input_z[slices]

    if erase_blank and len(np.unique(chunk_data)) < 5:
        output_z[slices] = np.zeros_like(chunk_data, dtype=np.uint8)
    else:
        thresholded = np.where(chunk_data > threshold, 255, 0).astype(np.uint8)
        output_z[slices] = thresholded

    return chunk_coords


def process_chunk_scale(args):
    """Process a single output chunk for scaling using nearest-neighbor resampling.

    Args tuple: (
        input_path: str,
        output_path: str,
        out_chunk_coords: list[tuple[int,int]] for each dim,
        scale: float,
    )
    """
    input_path, output_path, out_chunk_coords, scale = args

    in_z = zarr.open(input_path, mode='r')
    out_z = zarr.open(output_path, mode='r+')

    # Output chunk slice (exclusive stop)
    out_slices = tuple(slice(start, stop) for start, stop in out_chunk_coords)

    # Compute corresponding input ROI to cover this output region
    in_slices = []
    for (o_start, o_stop), dim in zip(out_chunk_coords, range(in_z.ndim)):
        # Map output indices to input by inverse of scaling
        in_start = int(np.floor(o_start / scale))
        in_stop = int(np.ceil(o_stop / scale))  # exclusive
        in_start = max(0, in_start)
        in_stop = min(in_z.shape[dim], max(in_start + 1, in_stop))
        in_slices.append(slice(in_start, in_stop))

    in_block = in_z[tuple(in_slices)]

    # Build per-dimension index arrays mapping output coords -> input indices (relative to in_block)
    idx_list = []
    for ax, (o_start, o_stop), s in zip(range(in_z.ndim), out_chunk_coords, [scale] * in_z.ndim):
        o_coords = np.arange(o_start, o_stop, dtype=np.float64)
        in_idx = np.floor(o_coords / s).astype(np.int64) - in_slices[ax].start
        # Clamp to valid range inside in_block for numerical safety
        in_idx[in_idx < 0] = 0
        max_valid = in_block.shape[ax] - 1
        if max_valid >= 0:
            in_idx[in_idx > max_valid] = max_valid
        idx_list.append(in_idx)

    # Advanced indexing via np.ix_ to broadcast across dims
    ix = np.ix_(*idx_list)
    out_block = in_block[ix]

    out_z[out_slices] = out_block

    return out_chunk_coords


def process_chunk_scale_per_axis(args):
    """Process a single output chunk for scaling using nearest-neighbor resampling with per-axis scales.

    Args tuple: (
        input_path: str,
        output_path: str,
        out_chunk_coords: list[tuple[int,int]] for each dim,
        scales: list[float] per axis (len == ndim), mapping input->output scaling
    )
    """
    input_path, output_path, out_chunk_coords, scales = args

    in_z = zarr.open(input_path, mode='r')
    out_z = zarr.open(output_path, mode='r+')

    out_slices = tuple(slice(start, stop) for start, stop in out_chunk_coords)

    in_slices = []
    for (o_start, o_stop), dim in zip(out_chunk_coords, range(in_z.ndim)):
        s = float(scales[dim])
        in_start = int(np.floor(o_start / s))
        in_stop = int(np.ceil(o_stop / s))
        in_start = max(0, in_start)
        in_stop = min(in_z.shape[dim], max(in_start + 1, in_stop))
        in_slices.append(slice(in_start, in_stop))

    in_block = in_z[tuple(in_slices)]

    idx_list = []
    for ax, (o_start, o_stop), s in zip(range(in_z.ndim), out_chunk_coords, scales):
        o_coords = np.arange(o_start, o_stop, dtype=np.float64)
        in_idx = np.floor(o_coords / float(s)).astype(np.int64) - in_slices[ax].start
        in_idx[in_idx < 0] = 0
        max_valid = in_block.shape[ax] - 1
        if max_valid >= 0:
            in_idx[in_idx > max_valid] = max_valid
        idx_list.append(in_idx)

    ix = np.ix_(*idx_list)
    out_block = in_block[ix]

    out_z[out_slices] = out_block

    return out_chunk_coords


def process_chunk_transpose(args):
    """Process a single output chunk for transposing axes.

    Args tuple: (
        input_path: str,
        output_path: str,
        out_chunk_coords: list[tuple[int,int]] for each dim of OUTPUT,
        perm: list[int] such that out axis i = in axis perm[i]
    )
    """
    input_path, output_path, out_chunk_coords, perm = args

    in_z = zarr.open(input_path, mode='r')
    out_z = zarr.open(output_path, mode='r+')

    out_slices = tuple(slice(start, stop) for start, stop in out_chunk_coords)

    # Build input slices using inverse permutation: for input axis j, take slice from out axis i where perm[i] = j
    inv = _inverse_permutation(perm)
    in_slices = [None] * in_z.ndim
    for in_ax in range(in_z.ndim):
        out_ax = inv[in_ax]
        o_start, o_stop = out_chunk_coords[out_ax]
        in_slices[in_ax] = slice(o_start, o_stop)

    in_block = in_z[tuple(in_slices)]
    out_block = np.transpose(in_block, axes=perm)

    out_z[out_slices] = out_block

    return out_chunk_coords


def get_chunk_coords(shape, chunks):
    """Generate coordinates for all chunks in the array."""
    chunk_ranges = []
    for dim_size, chunk_size in zip(shape, chunks):
        ranges = []
        for start in range(0, dim_size, chunk_size):
            stop = min(start + chunk_size, dim_size)
            ranges.append((start, stop))
        chunk_ranges.append(ranges)
    
    # Generate all combinations of chunk coordinates
    return list(itertools.product(*chunk_ranges))


def _create_level_dataset(root_group_path, level_name, shape, chunks, dtype, compressor):
    root = zarr.open_group(root_group_path, mode='a')
    # Remove existing dataset if present
    if level_name in root:
        del root[level_name]
    return root.create_dataset(level_name, shape=shape, chunks=chunks, dtype=dtype, compressor=compressor)


def _write_multiscales_metadata(root_group_path, axes_names, num_levels):
    root = zarr.open_group(root_group_path, mode='a')
    axes = []
    for a in axes_names:
        axes.append({
            'name': a,
            'type': 'space' if a in ('x', 'y', 'z') else 'unknown'
        })
    datasets = [{'path': str(i)} for i in range(num_levels)]
    root.attrs['multiscales'] = [{
        'version': '0.4',
        'name': 'image',
        'axes': axes,
        'datasets': datasets,
    }]


def _build_pyramid(base_group_path, axes_names, num_levels, num_workers):
    """Builds levels 1..num_levels-1 from level 0 using nearest-neighbor downsampling.

    Downsamples ALL axes by 2 at each level (uniform downsampling).
    """
    # Per-axis downsample factors from prev->next level (uniform 2x downsample)
    # Using 0.5 here because out_dim = in_dim * 0.5
    per_axis_scales = [0.5] * len(axes_names)

    for level in range(1, num_levels):
        in_path = f"{base_group_path}/{level-1}"
        out_path = f"{base_group_path}/{level}"

        in_z = zarr.open(in_path, mode='r')
        # Compute output shape by applying per-axis scales
        out_shape = tuple(max(1, int(round(s * dim))) for dim, s in zip(in_z.shape, per_axis_scales))
        out_chunks = tuple(min(c, s) for c, s in zip(in_z.chunks, out_shape))

        _create_level_dataset(base_group_path, str(level), out_shape, out_chunks, in_z.dtype, in_z.compressor)

        out_chunk_coords = get_chunk_coords(out_shape, out_chunks)
        process_args = [
            (in_path, out_path, coords, per_axis_scales)
            for coords in out_chunk_coords
        ]

        with Pool(processes=num_workers) as pool:
            with tqdm(total=len(out_chunk_coords), desc=f"Building level {level}") as pbar:
                for _ in pool.imap_unordered(process_chunk_scale_per_axis, process_args):
                    pbar.update(1)


def main():
    parser = argparse.ArgumentParser(description='Operate on a zarr array: threshold, scale, or transpose; and write OME-Zarr pyramid')
    parser.add_argument('input_zarr', type=str, help='Path to input zarr array')
    parser.add_argument('output_zarr', type=str, help='Path to output zarr array')

    parser.add_argument('--task', choices=['threshold', 'scale', 'transpose'], default='threshold',
                        help='Task to perform')

    # Threshold-specific options
    parser.add_argument('-t', '--threshold', type=float, default=127,
                        help='Threshold value (used for --task threshold)')
    parser.add_argument('--erase-blank', action='store_true',
                        help='Erase homogeneous chunks (less than 5 unique values) by setting them to 0')

    # Scale-specific options
    parser.add_argument('--scale-factor', type=float, default=None,
                        help='Uniform scale factor for all dimensions (used for --task scale)')

    # Transpose-specific options
    parser.add_argument('--transpose-order', type=str, default='xzy',
                        help='Output axes order for transpose, as a permutation of xyz (default: xzy)')

    # Common options
    parser.add_argument('-n', '--num-workers', type=int, default=None,
                        help='Number of worker processes (default: number of CPU cores)')

    args = parser.parse_args()

    # Open input zarr to get metadata
    input_z = zarr.open(args.input_zarr, mode='r')

    print(f"Input array shape: {input_z.shape}")
    print(f"Input array chunks: {input_z.chunks}")
    print(f"Input array dtype: {input_z.dtype}")

    output_path = Path(args.output_zarr)
    if output_path.exists():
        response = input(f"Output path {output_path} already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print('Aborted.')
            return

    num_workers = args.num_workers if args.num_workers else cpu_count()
    print(f"Using {num_workers} worker processes")

    if args.task == 'threshold':
        print(f"Task: threshold | value={args.threshold} | erase_blank={args.erase_blank}")

        # Create multiscale root and level 0 dataset
        root_group_path = args.output_zarr
        axes_names = ['z', 'y', 'x']
        lvl0_path = f"{root_group_path}/0"
        _create_level_dataset(
            root_group_path,
            '0',
            input_z.shape,
            input_z.chunks,
            np.uint8,
            input_z.compressor,
        )

        # Prepare and run chunked processing
        chunk_coords = get_chunk_coords(input_z.shape, input_z.chunks)
        print(f"Total chunks to process: {len(chunk_coords)}")

        process_args = [
            (args.input_zarr, lvl0_path, coords, args.threshold, args.erase_blank)
            for coords in chunk_coords
        ]

        with Pool(processes=num_workers) as pool:
            with tqdm(total=len(chunk_coords), desc='Thresholding chunks') as pbar:
                for _ in pool.imap_unordered(process_chunk_threshold, process_args):
                    pbar.update(1)

        # Build pyramid levels 1..5 and write multiscales metadata
        _build_pyramid(root_group_path, axes_names, num_levels=6, num_workers=num_workers)
        _write_multiscales_metadata(root_group_path, axes_names, num_levels=6)

        print(f"Thresholding complete. OME-Zarr pyramid saved to: {args.output_zarr}")

    elif args.task == 'scale':
        if args.scale_factor is None or args.scale_factor <= 0:
            raise SystemExit('Error: --scale-factor must be provided and > 0 for task=scale')

        sf = float(args.scale_factor)
        out_shape = tuple(max(1, int(round(d * sf))) for d in input_z.shape)
        out_chunks = tuple(min(c, s) for c, s in zip(input_z.chunks, out_shape))

        print(f"Task: scale | factor={sf}")
        print(f"Output array shape: {out_shape}")
        print(f"Output array chunks: {out_chunks}")

        # Create multiscale root and level 0 dataset
        root_group_path = args.output_zarr
        axes_names = ['z', 'y', 'x']
        lvl0_path = f"{root_group_path}/0"
        _create_level_dataset(
            root_group_path,
            '0',
            out_shape,
            out_chunks,
            input_z.dtype,
            input_z.compressor,
        )

        # Process output chunks
        out_chunk_coords = get_chunk_coords(out_shape, out_chunks)
        print(f"Total chunks to process: {len(out_chunk_coords)}")

        process_args = [
            (args.input_zarr, lvl0_path, coords, sf)
            for coords in out_chunk_coords
        ]

        with Pool(processes=num_workers) as pool:
            with tqdm(total=len(out_chunk_coords), desc='Scaling chunks') as pbar:
                for _ in pool.imap_unordered(process_chunk_scale, process_args):
                    pbar.update(1)

        # Build pyramid levels 1..5 and write multiscales metadata
        _build_pyramid(root_group_path, axes_names, num_levels=6, num_workers=num_workers)
        _write_multiscales_metadata(root_group_path, axes_names, num_levels=6)

        print(f"Scaling complete. OME-Zarr pyramid saved to: {args.output_zarr}")

    elif args.task == 'transpose':
        # Parse transpose order string to permutation
        valid_axes = {'x': 2, 'y': 1, 'z': 0}
        order_str = args.transpose_order.lower()
        if sorted(order_str) != ['x', 'y', 'z'] or len(order_str) != 3:
            raise SystemExit('Error: --transpose-order must be a permutation of xyz, e.g., xzy')

        # perm is list such that out axis i = in axis perm[i]
        perm = [valid_axes[c] for c in order_str]

        # Compute output shape/chunks by permuting input dims
        out_shape = tuple(input_z.shape[p] for p in perm)
        out_chunks = tuple(input_z.chunks[p] for p in perm)

        print(f"Task: transpose | order={order_str} | perm={perm}")
        print(f"Output array shape: {out_shape}")
        print(f"Output array chunks: {out_chunks}")

        # Determine axes names in output order
        axis_name_for_in = ['z', 'y', 'x']
        axes_names = [axis_name_for_in[p] for p in perm]

        # Create multiscale root and level 0 dataset
        root_group_path = args.output_zarr
        lvl0_path = f"{root_group_path}/0"
        _create_level_dataset(
            root_group_path,
            '0',
            out_shape,
            out_chunks,
            input_z.dtype,
            input_z.compressor,
        )

        # Process output chunks
        out_chunk_coords = get_chunk_coords(out_shape, out_chunks)
        print(f"Total chunks to process: {len(out_chunk_coords)}")

        process_args = [
            (args.input_zarr, lvl0_path, coords, perm)
            for coords in out_chunk_coords
        ]

        with Pool(processes=num_workers) as pool:
            with tqdm(total=len(out_chunk_coords), desc='Transposing chunks') as pbar:
                for _ in pool.imap_unordered(process_chunk_transpose, process_args):
                    pbar.update(1)

        # Build pyramid levels 1..5 and write multiscales metadata
        _build_pyramid(root_group_path, axes_names, num_levels=6, num_workers=num_workers)
        _write_multiscales_metadata(root_group_path, axes_names, num_levels=6)

        print(f"Transpose complete. OME-Zarr pyramid saved to: {args.output_zarr}")


if __name__ == "__main__":
    main()
