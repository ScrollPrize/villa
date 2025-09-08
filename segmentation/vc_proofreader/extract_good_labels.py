"""
Preprocess Zarr/OME-Zarr volumes by scanning 3D chunks and filtering
based on connected components, nonzero content, and labeled percentage.

Behavior:
- Iterates the volume in 3D chunks of a specified size. If `chunk_size` is a
  single int, it is treated as cubic (e.g., 512 -> (512, 512, 512)).
- Accepts OME-Zarr inputs by opening the group and selecting level '0'.
- For each chunk, optionally restricts analysis to a specific `target_value`.
- Checks that the chunk is nonzero (if required).
- Computes the percent of the chunk labeled and filters by min/max thresholds.
- Counts connected components (via cc3d) and filters by min/max thresholds.
- If a chunk meets all requirements, writes the original chunk values to an
  output Zarr with the same shape as the input. Non-passing chunks are left
  as implicit zeros (no write), keeping the output sparse.

Notes and limits:
- This script expects a 3D array (Z, Y, X). For OME-Zarr, we select level '0',
  which commonly is 3D for labels. If your array has more dimensions, provide
  a subkey via `array_key` or preprocess to 3D.
- Partial edge chunks are skipped so only full-size chunks are processed.

Usage:
  python extract_good_labels.py \
    --input /path/to/input.ome.zarr \
    --output /path/to/output.zarr \
    --chunk-size 512 \
    --target-value 2 \
    --min-cc 4 --max-cc 50 \
    --min-percent 1 --max-percent 100 \
    --require-nonzero \
    --workers 8
Requires tqdm for progress bars (falls back to prints if unavailable).

Or edit the CONFIG dict below and run without args.
"""

from __future__ import annotations

import argparse
import os
from typing import Optional, Sequence, Tuple, Iterable

import numpy as np
import zarr

try:
    import cc3d
except Exception as e:
    cc3d = None

# Optional progress bar
try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None  # type: ignore

# Optional skeletonization for branch detection (2D per Z)
try:
    from skimage.morphology import skeletonize  # type: ignore
except Exception:
    skeletonize = None  # type: ignore
try:
    from scipy.ndimage import convolve  # type: ignore
except Exception:
    convolve = None  # type: ignore


def worker_task(args_tuple):
    """Top-level worker function to allow multiprocessing pickling.

    Args tuple contains:
      (input_path, output_path, array_key,
       z, y, x, cz, cy, cx,
       min_cc, max_cc, min_percent, max_percent,
       require_nonzero, target_value, connectivity, reject_branches)
    Returns (counted_chunks, written_chunks, branch_rejected_count) for aggregation.
    """
    (
        input_path, output_path, array_key,
        z, y, x, cz, cy, cx,
        min_cc, max_cc, min_percent, max_percent,
        require_nonzero, target_value, connectivity, reject_branches,
    ) = args_tuple

    # Re-open arrays in each process
    in_a = open_ome_zarr_array(input_path, array_key=array_key)
    out_a = zarr.open(output_path, mode='r+')

    zs, ys, xs = slice(z, z + cz), slice(y, y + cy), slice(x, x + cx)
    chunk = in_a[zs, ys, xs]

    # Build foreground mask based on target_value
    if target_value is not None:
        fg = (chunk == target_value)
    else:
        fg = (chunk != 0)

    # Nonzero requirement
    if require_nonzero and not fg.any():
        return (1, 0, 0)

    # Percent labeled constraints
    labeled_pct = float(fg.sum()) / fg.size * 100.0 if fg.size > 0 else 0.0
    if labeled_pct < min_percent or labeled_pct > max_percent:
        return (1, 0, 0)

    # Connected components constraints
    if min_cc is not None or max_cc is not None:
        n_cc = count_connected_components(fg, connectivity=connectivity)
        if min_cc is not None and n_cc < min_cc:
            return (1, 0, 0)
        if max_cc is not None and n_cc > max_cc:
            return (1, 0, 0)

    # Skeleton junction rejection (2D per Z)
    if reject_branches:
        # fg is a boolean 3D chunk
        if has_2d_skeleton_branches(fg):
            return (1, 0, 1)

    # Passed all checks: write original chunk through.
    out_a[zs, ys, xs] = chunk
    return (1, 1, 0)


# -------------------------
# User configuration (defaults)
# -------------------------
CONFIG = {
    # Input OME-Zarr/Zarr path. If the root is a group, the script selects '0'.
    'input_path': '/mnt/raid_nvme/vx_paths/s5_masked.zarr',
    # Optional explicit array key within the zarr group (e.g. '0'). If None, will try '0'.
    'array_key': 0,
    # Output zarr array path (created or overwritten).
    'output_path': '/mnt/raid_nvme/vx_paths/s5_masked_good.zarr',
    # Chunk size as int or (z, y, x). If int, assumes cubic.
    'chunk_size': 256,
    # Connected components constraints (inclusive). Set to None to disable a bound.
    'min_cc': 10,   # e.g., 4 for "more than 3"
    'max_cc': 75,   # e.g., 50 for "not greater than 50"
    # Labeled percentage constraints (inclusive) in [0, 100].
    'min_percent': 0.1,
    'max_percent': 100.0,
    # If True, require that the (possibly target-filtered) chunk has any nonzero voxels.
    'require_nonzero': True,
    # If set (int), computations consider only voxels equal to this value. Otherwise, any nonzero.
    'target_value': 255,
    # cc3d connectivity for 3D (6, 18, or 26). 26 connects diagonals.
    'connectivity': 26,
    # Compression settings for output. When False, do not write empty chunks.
    'write_empty_chunks': False,
    # Reject chunks whose 2D skeletons (per Z-slice) contain junctions (>2 neighbors)
    'reject_branches': True,
    # Number of worker processes. If None, uses os.cpu_count().
    'workers': 16,
}


def _as_zyx_size(size: int | Sequence[int]) -> Tuple[int, int, int]:
    if isinstance(size, int):
        return (size, size, size)
    if len(size) != 3:
        raise ValueError("chunk_size must be an int or a 3-tuple (z, y, x)")
    return tuple(int(s) for s in size)  # type: ignore[return-value]


def open_ome_zarr_array(path: str, array_key: Optional[str] = None) -> zarr.Array:
    root = zarr.open(path, mode='r')
    if isinstance(root, zarr.Array):
        return root
    if not isinstance(root, zarr.hierarchy.Group):
        raise ValueError(f"Unsupported zarr object at {path!r}")
    if array_key is not None:
        arr = root.get(array_key)
        if arr is None or not isinstance(arr, zarr.Array):
            raise ValueError(f"Array key {array_key!r} not found in group at {path!r}")
        return arr
    # Default to multiscale level '0' if present
    if '0' in root and isinstance(root['0'], zarr.Array):
        return root['0']
    # Fallback: pick the first array child
    for k, v in root.items():
        if isinstance(v, zarr.Array):
            return v
    raise ValueError(f"No array found in group at {path!r}; keys: {list(root.keys())}")


def prepare_output_array(
    path: str,
    shape: Tuple[int, int, int],
    dtype: np.dtype,
    chunks: Optional[Tuple[int, int, int]] = None,
    write_empty_chunks: bool = False,
) -> zarr.Array:
    # Use provided chunks; else default to full-shape (not ideal). Caller should pass chunk_size.
    if chunks is None:
        chunks = tuple(min(128, s) for s in shape)

    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=zarr.Blosc.BITSHUFFLE)
    arr = zarr.open(
        path,
        mode='w',
        shape=shape,
        chunks=chunks,
        dtype=dtype,
        compressor=compressor,
        write_empty_chunks=write_empty_chunks,
    )
    return arr


def count_connected_components(mask: np.ndarray, connectivity: int = 26) -> int:
    if not mask.any():
        return 0
    if cc3d is None:
        raise RuntimeError("cc3d is required for connected component counting. Please install 'cc3d'.")
    labels = cc3d.connected_components(mask.astype(np.uint8), connectivity=connectivity)
    # Background is 0; number of components is max label value
    return int(labels.max())


def has_2d_skeleton_branches(fg3d: np.ndarray) -> bool:
    """Return True if any Z-slice skeleton contains a junction (>2 neighbors).

    - `fg3d` is a boolean 3D array (Z, Y, X)
    - Uses skimage.morphology.skeletonize per 2D slice and counts 8-neighborhood
      neighbors via numpy slicing (no SciPy required).
    """
    if skeletonize is None:
        raise RuntimeError("scikit-image is required for skeletonization; please install 'scikit-image'.")
    if convolve is None:
        raise RuntimeError("scipy is required for neighbor counting; please install 'scipy'.")
    if fg3d.ndim != 3:
        raise ValueError("fg3d must be a 3D boolean array")
    Z = fg3d.shape[0]
    for z in range(Z):
        plane = fg3d[z]
        if not plane.any():
            continue
        sk = skeletonize(plane.astype(bool))
        if sk.shape[0] < 3 or sk.shape[1] < 3:
            continue
        # Compute 8-neighbor degree via convolution to avoid boolean-add pitfalls
        kernel = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
        deg = convolve(sk.astype(np.uint8), kernel, mode='constant', cval=0)
        if np.any((deg >= 3) & sk):
            return True
    return False


def process(
    input_path: str,
    output_path: str,
    chunk_size: Tuple[int, int, int],
    array_key: Optional[str] = None,
    min_cc: Optional[int] = None,
    max_cc: Optional[int] = None,
    min_percent: float = 0.0,
    max_percent: float = 100.0,
    require_nonzero: bool = True,
    target_value: Optional[int] = None,
    connectivity: int = 26,
    write_empty_chunks: bool = False,
    reject_branches: bool = False,
    workers: Optional[int] = None,
) -> None:
    in_arr = open_ome_zarr_array(input_path, array_key=array_key)

    if in_arr.ndim != 3:
        raise ValueError(
            f"This script expects a 3D array (Z, Y, X). Got shape {in_arr.shape}. "
            "If using OME-Zarr multiscales, ensure level '0' is 3D or specify an 'array_key'."
        )

    zdim, ydim, xdim = in_arr.shape
    cz, cy, cx = chunk_size

    # Ensure output chunking exactly matches iteration chunk size to avoid overlapping writes
    out_chunks = (cz, cy, cx)

    # Create output array up front (single-writer for metadata)
    out_arr = prepare_output_array(
        output_path,
        shape=(zdim, ydim, xdim),
        dtype=in_arr.dtype,
        chunks=out_chunks,
        write_empty_chunks=write_empty_chunks,
    )

    # Generate all aligned chunk starts (only full chunks)
    coords: Iterable[Tuple[int, int, int]] = (
        (z, y, x)
        for z in range(0, zdim - cz + 1, cz)
        for y in range(0, ydim - cy + 1, cy)
        for x in range(0, xdim - cx + 1, cx)
    )
    coord_list = list(coords)

    # Multiprocessing worker function (defined inside to capture parameters cleanly)
    # Determine number of workers
    if workers is None:
        try:
            import os as _os
            workers = _os.cpu_count() or 1
        except Exception:
            workers = 1

    total = 0
    written = 0
    branch_rej = 0

    # Dispatch work
    try:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=workers) as ex:
            # Pack all required args for worker
            arg_items = [
                (
                    input_path, output_path, array_key,
                    z, y, x, cz, cy, cx,
                    min_cc, max_cc, min_percent, max_percent,
                    require_nonzero, target_value, connectivity,
                    reject_branches,
                )
                for (z, y, x) in coord_list
            ]
            futures = [ex.submit(worker_task, a) for a in arg_items]
            pbar = tqdm(total=len(coord_list), desc="Chunks", unit="chunk") if tqdm else None
            for i, fut in enumerate(as_completed(futures), 1):
                done, w, b = fut.result()
                total += done
                written += w
                branch_rej += b
                if pbar:
                    pbar.update(1)
                    pbar.set_postfix(written=written, branches=branch_rej)
                else:
                    if i % max(1, len(coord_list) // 50) == 0:
                        print(f"Progress: {i}/{len(coord_list)} chunks | written={written} | branches_rejected={branch_rej}")
            if pbar:
                pbar.close()
    except Exception as e:
        print(f"Parallel execution failed ({e}); falling back to serial.")
        if tqdm:
            it = tqdm(coord_list, desc="Chunks", unit="chunk")
        else:
            it = coord_list
        for (z, y, x) in it:
            d, w, b = worker_task((
                input_path, output_path, array_key,
                z, y, x, cz, cy, cx,
                min_cc, max_cc, min_percent, max_percent,
                require_nonzero, target_value, connectivity,
                reject_branches,
            ))
            total += d
            written += w
            branch_rej += b
            if not tqdm:
                # occasional print
                i = total
                if i % max(1, len(coord_list) // 50) == 0:
                    print(f"Progress: {i}/{len(coord_list)} chunks | written={written} | branches_rejected={branch_rej}")

    print(f"Done. Wrote {written} of {total} chunks to {output_path} | branches_rejected={branch_rej}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Filter Zarr/OME-Zarr by per-chunk CC and label percentage.")
    p.add_argument('--input', dest='input_path', type=str, default=CONFIG['input_path'], help='Input .zarr or .ome.zarr path')
    p.add_argument('--array-key', type=str, default=CONFIG['array_key'], help="Explicit array key in group (e.g., '0')")
    p.add_argument('--output', dest='output_path', type=str, default=CONFIG['output_path'], help='Output .zarr path to create')
    p.add_argument('--chunk-size', type=int, nargs='*', default=None, help='Single int or three ints: z y x')
    p.add_argument('--min-cc', type=int, default=CONFIG['min_cc'], help='Minimum connected components (inclusive)')
    p.add_argument('--max-cc', type=int, default=CONFIG['max_cc'], help='Maximum connected components (inclusive)')
    p.add_argument('--min-percent', type=float, default=CONFIG['min_percent'], help='Minimum labeled percent (inclusive)')
    p.add_argument('--max-percent', type=float, default=CONFIG['max_percent'], help='Maximum labeled percent (inclusive)')
    p.add_argument('--require-nonzero', action='store_true', default=CONFIG['require_nonzero'], help='Require any labeled voxels in chunk')
    p.add_argument('--no-require-nonzero', action='store_true', help='Disable nonzero requirement')
    p.add_argument('--target-value', type=int, default=CONFIG['target_value'], help='If set, only this value is considered labeled')
    p.add_argument('--connectivity', type=int, default=CONFIG['connectivity'], choices=(6, 18, 26), help='3D connectivity for cc3d')
    p.add_argument('--write-empty-chunks', action='store_true', default=CONFIG['write_empty_chunks'], help='Write empty chunks to output')
    p.add_argument('--reject-branches', action='store_true', default=CONFIG['reject_branches'], help='Reject chunks if any 2D slice skeleton has junctions')
    p.add_argument('--workers', type=int, default=CONFIG['workers'], help='Number of worker processes')
    return p.parse_args()


def main():
    args = parse_args()

    input_path = args.input_path or CONFIG['input_path']
    output_path = args.output_path or CONFIG['output_path']

    if args.no_require_nonzero:
        require_nonzero = False
    else:
        require_nonzero = args.require_nonzero

    # Resolve chunk size from CLI or CONFIG
    if args.chunk_size is None:
        chunk_cfg = CONFIG['chunk_size']
    else:
        if len(args.chunk_size) == 1:
            chunk_cfg = int(args.chunk_size[0])
        elif len(args.chunk_size) == 3:
            chunk_cfg = (int(args.chunk_size[0]), int(args.chunk_size[1]), int(args.chunk_size[2]))
        else:
            raise ValueError('Provide either one int or three ints for --chunk-size')

    chunk_size = _as_zyx_size(chunk_cfg)

    if not input_path:
        raise SystemExit('Please provide --input or set CONFIG["input_path"].')
    if not output_path:
        # Default output to sibling with suffix
        base = input_path.rstrip('/').rstrip('.zarr').rstrip('.ome')
        output_path = base + '_filtered.zarr'
        print(f"No --output provided; using {output_path}")

    # Ensure parent dir exists for output
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    if cc3d is None:
        print("Warning: cc3d is not installed; connected component filtering will fail if min/max-cc are set.")

    process(
        input_path=input_path,
        output_path=output_path,
        chunk_size=chunk_size,
        array_key=args.array_key or CONFIG['array_key'],
        min_cc=args.min_cc,
        max_cc=args.max_cc,
        min_percent=args.min_percent,
        max_percent=args.max_percent,
        require_nonzero=require_nonzero,
        target_value=args.target_value,
        connectivity=args.connectivity,
        write_empty_chunks=args.write_empty_chunks,
        reject_branches=args.reject_branches,
        workers=args.workers,
    )


if __name__ == '__main__':
    main()
