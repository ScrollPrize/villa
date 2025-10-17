#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import tifffile
import zarr
from numcodecs import Blosc
from scipy import ndimage
from tqdm import tqdm


@dataclass(frozen=True)
class GraphConfig:
    foreground_threshold: float
    ignore_background_edges: bool
    long_range_stride: int
    store_labels: bool
    overwrite: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Mutex Watershed-style attractive and repulsive affinity graphs "
            "for each 3D TIFF volume in a folder."
        )
    )
    parser.add_argument("input_dir", type=Path, help="Directory containing source 3D TIFF volumes.")
    parser.add_argument("output_dir", type=Path, help="Directory to write Zarr outputs.")
    parser.add_argument(
        "--pattern",
        default="*.tif",
        help="Glob pattern (relative to input_dir) used to discover input volumes. [default: %(default)s]",
    )
    parser.add_argument(
        "--foreground-threshold",
        type=float,
        default=0.0,
        help="Foreground voxels satisfy value > threshold prior to connected-component labeling.",
    )
    parser.add_argument(
        "--ignore-background-edges",
        action="store_true",
        help="Mask edges that touch background label 0.",
    )
    parser.add_argument(
        "--long-range-stride",
        type=int,
        default=2,
        help="Stride applied to long-range repulsive edges to thin the sampling grid. Use 1 to disable.",
    )
    parser.add_argument(
        "--store-labels",
        action="store_true",
        help="Store the intermediate connected-component labels inside the output Zarr.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing .zarr outputs.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of parallel worker processes. 0 uses os.cpu_count(). 1 forces sequential execution.",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Display a tqdm progress bar. Automatically enabled when multiple inputs exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    files = sorted(input_dir.glob(args.pattern))
    if not files:
        raise FileNotFoundError(f"No files matched pattern '{args.pattern}' in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    attractive_offsets, repulsive_offsets, long_range_offsets = build_offset_sets()
    cfg = GraphConfig(
        foreground_threshold=args.foreground_threshold,
        ignore_background_edges=args.ignore_background_edges,
        long_range_stride=max(1, args.long_range_stride),
        store_labels=args.store_labels,
        overwrite=args.overwrite,
    )

    use_progress = args.progress or len(files) > 1
    worker_count = None if args.workers == 0 else max(1, args.workers)

    process = partial(
        process_volume,
        output_dir=output_dir,
        cfg=cfg,
        attractive_offsets=attractive_offsets,
        repulsive_offsets=repulsive_offsets,
        long_range_offsets=long_range_offsets,
    )

    if worker_count == 1 or len(files) == 1:
        iterator: Iterable[Path] = files
        if use_progress:
            iterator = tqdm(files, desc="Volumes", unit="vol")
        for path in iterator:
            process(path)
    else:
        total = len(files)
        with ProcessPoolExecutor(max_workers=worker_count) as pool:
            futures = [pool.submit(process, path) for path in files]
            if use_progress:
                progress_bar = tqdm(total=total, desc="Volumes", unit="vol")
            else:
                progress_bar = None
            try:
                for fut in as_completed(futures):
                    fut.result()
                    if progress_bar is not None:
                        progress_bar.update(1)
            finally:
                if progress_bar is not None:
                    progress_bar.close()


# --------------------------------------------------------------------------------------
# DType helpers
# --------------------------------------------------------------------------------------

def _is_binary_array(arr: np.ndarray) -> bool:
    """Return True if the array contains only 0 and 1 values.

    Works for integer and floating arrays. Uses exact comparison intended for
    programmatically created binary arrays (no tolerance to avoid silent loss).
    """
    a = np.asarray(arr)
    # Fast path: check bounds first
    if a.size == 0:
        return True
    # Only values 0 or 1
    return np.all((a == 0) | (a == 1))


def _min_uint_dtype(max_value: int) -> np.dtype:
    """Choose the smallest unsigned integer dtype that can store max_value."""
    mv = int(max(0, max_value))
    if mv <= np.iinfo(np.uint8).max:
        return np.uint8
    if mv <= np.iinfo(np.uint16).max:
        return np.uint16
    if mv <= np.iinfo(np.uint32).max:
        return np.uint32
    return np.uint64


def _select_min_lossless_dtype(arr: np.ndarray) -> np.dtype:
    """
    Decide the smallest dtype that preserves data exactly.

    Rules:
    - Binary arrays (only 0/1) -> uint8
    - Integer arrays          -> minimal unsigned/signed integer dtype that fits
    - Other floats            -> keep float32 (avoid potential loss)
    """
    a = np.asarray(arr)
    kind = a.dtype.kind

    # Binary detection takes precedence
    if _is_binary_array(a):
        return np.uint8

    if kind in ("i", "u"):
        min_v = int(a.min(initial=0))
        max_v = int(a.max(initial=0))
        if min_v >= 0:
            return _min_uint_dtype(max_v)
        # For signed, choose the smallest signed dtype that fits the range
        if np.iinfo(np.int8).min <= min_v and max_v <= np.iinfo(np.int8).max:
            return np.int8
        if np.iinfo(np.int16).min <= min_v and max_v <= np.iinfo(np.int16).max:
            return np.int16
        if np.iinfo(np.int32).min <= min_v and max_v <= np.iinfo(np.int32).max:
            return np.int32
        return np.int64

    # Floats that are not strictly binary: keep float32 to avoid precision loss
    return np.float32


def process_volume(
    path: Path,
    output_dir: Path,
    cfg: GraphConfig,
    attractive_offsets: Sequence[tuple[int, int, int]],
    repulsive_offsets: Sequence[tuple[int, int, int]],
    long_range_offsets: set[tuple[int, int, int]],
) -> None:
    volume = tifffile.imread(path)
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D volume in {path}, got shape {volume.shape}")
    volume = np.asarray(volume)

    foreground = volume > cfg.foreground_threshold
    labels, component_count = ndimage.label(
        foreground,
        structure=ndimage.generate_binary_structure(rank=3, connectivity=1),
    )

    attractive, attractive_mask = compute_affinities(
        labels,
        attractive_offsets,
        mode="attractive",
        ignore_background=cfg.ignore_background_edges,
    )
    repulsive, repulsive_mask = compute_affinities(
        labels,
        repulsive_offsets,
        mode="repulsive",
        ignore_background=cfg.ignore_background_edges,
        stride=cfg.long_range_stride,
        stride_applicable_offsets=long_range_offsets,
    )

    # Conservative chunk shape with channel-leading layout
    chunk_shape = (1,) + tuple(min(64, s) for s in labels.shape)
    compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)

    output_path = output_dir / f"{path.stem}.zarr"
    if output_path.exists():
        if not cfg.overwrite:
            raise FileExistsError(f"Output {output_path} already exists (use --overwrite to replace).")
        if output_path.is_dir():
            shutil.rmtree(output_path)
        else:
            output_path.unlink()

    root = zarr.open_group(str(output_path), mode="w")

    # Determine minimal lossless dtypes
    attr_dtype = _select_min_lossless_dtype(attractive)
    rep_dtype = _select_min_lossless_dtype(repulsive)
    # For masks we prefer uint8 as well (avoids downstream bool->uint8 conversion)
    mask_attr_dtype = np.uint8 if _is_binary_array(attractive_mask) else _select_min_lossless_dtype(attractive_mask)
    mask_rep_dtype = np.uint8 if _is_binary_array(repulsive_mask) else _select_min_lossless_dtype(repulsive_mask)

    # Cast on write to avoid holding two copies if possible
    root.create_dataset(
        "affinities/attractive",
        data=attractive.astype(attr_dtype, copy=False),
        compressor=compressor,
        chunks=chunk_shape,
    )
    root.create_dataset(
        "affinities/repulsive",
        data=repulsive.astype(rep_dtype, copy=False),
        compressor=compressor,
        chunks=chunk_shape,
    )
    root.create_dataset(
        "mask/attractive",
        data=attractive_mask.astype(mask_attr_dtype, copy=False),
        compressor=compressor,
        chunks=chunk_shape,
    )
    root.create_dataset(
        "mask/repulsive",
        data=repulsive_mask.astype(mask_rep_dtype, copy=False),
        compressor=compressor,
        chunks=chunk_shape,
    )

    if cfg.store_labels:
        max_label = int(labels.max(initial=0)) if labels.size else 0
        label_dtype = _min_uint_dtype(max_label)
        root.create_dataset(
            "labels",
            data=labels.astype(label_dtype, copy=False),
            compressor=compressor,
            chunks=tuple(min(64, s) for s in labels.shape),
        )

    root.attrs.update(
        {
            "source_tiff": path.name,
            "label_components": int(component_count),
            "foreground_threshold": float(cfg.foreground_threshold),
            "ignore_background_edges": bool(cfg.ignore_background_edges),
            "long_range_stride": int(cfg.long_range_stride),
            "attractive_offsets": offsets_to_json(attractive_offsets),
            "repulsive_offsets": offsets_to_json(repulsive_offsets),
            "long_range_offsets": offsets_to_json(sorted(long_range_offsets, key=offset_sort_key)),
            "affinities_attractive_dtype": str(np.dtype(attr_dtype)),
            "affinities_repulsive_dtype": str(np.dtype(rep_dtype)),
            "mask_attractive_dtype": str(np.dtype(mask_attr_dtype)),
            "mask_repulsive_dtype": str(np.dtype(mask_rep_dtype)),
        }
    )


def compute_affinities(
    labels: np.ndarray,
    offsets: Sequence[tuple[int, int, int]],
    *,
    mode: str,
    ignore_background: bool,
    stride: int = 1,
    stride_applicable_offsets: Iterable[tuple[int, int, int]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if mode not in {"attractive", "repulsive"}:
        raise ValueError(f"Unsupported mode '{mode}'")
    stride = max(1, stride)
    stride_applicable = set(stride_applicable_offsets or [])
    apply_stride_mask = stride > 1 and bool(stride_applicable)
    stride_mask = None
    if apply_stride_mask:
        stride_mask = np.zeros_like(labels, dtype=bool)
        stride_mask[::stride, ::stride, ::stride] = True

    affinities = np.zeros((len(offsets),) + labels.shape, dtype=np.float32)
    mask = np.zeros((len(offsets),) + labels.shape, dtype=bool)

    for idx, offset in enumerate(offsets):
        src_slice, dst_slice = compute_valid_slices(labels.shape, offset)
        if src_slice is None:
            continue

        src = labels[src_slice]
        dst = labels[dst_slice]
        valid = np.ones(src.shape, dtype=bool)
        if ignore_background:
            valid &= (src != 0) & (dst != 0)
        if stride_mask is not None and tuple(offset) in stride_applicable:
            valid &= stride_mask[src_slice]

        if not np.any(valid):
            continue

        if mode == "attractive":
            channel = (src == dst) & valid
        else:
            channel = (src != dst) & valid

        affinities[idx][src_slice] = channel.astype(np.float32)
        mask[idx][src_slice] = valid

    return affinities, mask


def compute_valid_slices(
    shape: Sequence[int],
    offset: Sequence[int],
) -> tuple[tuple[slice, slice, slice], tuple[slice, slice, slice]] | tuple[None, None]:
    src_slices = []
    dst_slices = []
    for dim, off in zip(shape, offset):
        start = max(0, -off)
        end = min(dim, dim - off)
        if start >= end:
            return None, None
        src_slices.append(slice(start, end))
        dst_slices.append(slice(start + off, end + off))
    return tuple(src_slices), tuple(dst_slices)


def build_offset_sets() -> tuple[
    tuple[tuple[int, int, int], ...],
    tuple[tuple[int, int, int], ...],
    set[tuple[int, int, int]],
]:
    attractive = sorted(
        {
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
        },
        key=offset_sort_key,
    )

    short_repulsive = expand_offsets([(1, 1, 0)])
    long_repulsive = expand_offsets([(9, 0, 0), (27, 0, 0), (9, 4, 0), (9, 9, 0)])
    repulsive = sorted(short_repulsive | long_repulsive, key=offset_sort_key)

    return tuple(attractive), tuple(repulsive), long_repulsive


def expand_offsets(base_vectors: Iterable[Sequence[int]]) -> set[tuple[int, int, int]]:
    offsets: set[tuple[int, int, int]] = set()
    for vec in base_vectors:
        vec = tuple(int(v) for v in vec)
        permutations = {perm for perm in itertools.permutations(vec)}
        for perm in permutations:
            non_zero_axes = [i for i, val in enumerate(perm) if val != 0]
            sign_choices = itertools.product([-1, 1], repeat=len(non_zero_axes))
            for signs in sign_choices:
                coords = list(perm)
                for axis, sign in zip(non_zero_axes, signs):
                    coords[axis] *= sign
                offset = tuple(coords)
                if offset != (0, 0, 0):
                    offsets.add(offset)
    return offsets


def offset_sort_key(offset: Sequence[int]) -> tuple[int, int, int, int]:
    manhattan = sum(abs(v) for v in offset)
    return (manhattan, abs(offset[0]), abs(offset[1]), abs(offset[2]))


def offsets_to_json(offsets: Sequence[Sequence[int]]) -> list[list[int]]:
    return [list(map(int, offset)) for offset in offsets]


if __name__ == "__main__":
    main()
