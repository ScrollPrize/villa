"""Geometry surface consistency: label-free sheet-coherence post-processing for 3D ink
probability volumes.

For every voxel the output is the maximum, over the three axis-aligned planes through it, of the
mean probability on a ``(2r+1) x (2r+1)`` window in that plane. Ink lives on a thin papyrus
sheet, so probability mass that is coherent within a plane is preserved while isolated,
off-sheet responses are attenuated. The transform is deterministic, needs no labels or
retraining, and is orientation-robust across the three voxel axes.

The CLI consumes the sparse six-level uint8 OME-Zarr written by
``vesuvius.ink_detection.inference.infer_full3d_tifxyz`` and writes a pyramid with the same
shape, chunking, dtype and multiscale metadata. Only chunks present in the source are processed
(with a ``radius`` halo read from their neighbours, missing neighbours are zero) and coarser
levels are rebuilt by the same 2x mean pooling, so a whole-segment prediction never has to fit
in memory and the output support equals the input support.

Blind evidence on public Scroll 1 data (official ``ckpt_78k_fullsup`` native-3D checkpoint, eight
sealed 256^3 windows of segment w00_20231016151002): mean cube AUC 0.703 -> 0.788 with a
rolled-volume control at 0.639; second segment w01_20230702185753_r15: 0.714 -> 0.769.
See https://github.com/danilolapegna/vesuvius-geometry-surface-consistency (MIT).

Usage::

    python -m vesuvius.ink_detection.postprocessing.surface_consistency \\
        /data/predictions/w035.ome.zarr /data/predictions/w035-gsc.ome.zarr --radius 2
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import time
from pathlib import Path
from typing import Iterable

import numpy as np
from numcodecs import Blosc
from scipy.ndimage import uniform_filter
import zarr

from vesuvius.label_zarr import ARRAY_DIMENSIONS, create_v2_array, open_v2_group

_CHUNK_KEY = re.compile(r"^(\d+)[./](\d+)[./](\d+)$")
_COMPRESSOR = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)


def enhance(probability: np.ndarray, radius: int = 2) -> np.ndarray:
    """Return, per voxel, the maximum of the local plane means (XY, XZ, YZ) around it.

    ``probability`` must be a finite 3D array in Z-Y-X order with values in ``[0, 1]``;
    ``radius`` >= 1 sets the window half-size. The output is float32 with the same shape.
    """

    value = np.asarray(probability, dtype=np.float32)
    if value.ndim != 3:
        raise ValueError("probability must be a 3D array (Z, Y, X)")
    if radius < 1:
        raise ValueError("radius must be >= 1")
    if not np.isfinite(value).all():
        raise ValueError("probability contains non-finite values")
    width = 2 * radius + 1
    xy = uniform_filter(value, size=(1, width, width), mode="nearest")
    xz = uniform_filter(value, size=(width, 1, width), mode="nearest")
    yz = uniform_filter(value, size=(width, width, 1), mode="nearest")
    return np.maximum(np.maximum(xy, xz), yz)


def to_unit(array: np.ndarray) -> np.ndarray:
    """Return ``array`` as float32 in ``[0, 1]``; integer dtypes are divided by their maximum."""

    array = np.asarray(array)
    if np.issubdtype(array.dtype, np.integer):
        return array.astype(np.float32) / float(np.iinfo(array.dtype).max)
    return array.astype(np.float32, copy=False)


def encode_uint8(array: np.ndarray) -> np.ndarray:
    """Encode ``[0, 1]`` probabilities the way the native writer does (round to 0..255)."""

    return np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)


def downsample_mean_3d(block: np.ndarray) -> np.ndarray:
    """Mean-pool a ZYX block by two (ceil shapes) and round back to the block dtype."""

    block = np.asarray(block)
    output_shape = tuple((value + 1) // 2 for value in block.shape)
    total = np.zeros(output_shape, dtype=np.float64)
    count = np.zeros(output_shape, dtype=np.float64)
    for offsets in itertools.product((0, 1), repeat=3):
        sample = block[offsets[0] :: 2, offsets[1] :: 2, offsets[2] :: 2]
        if sample.size:
            slices = tuple(slice(0, value) for value in sample.shape)
            total[slices] += sample
            count[slices] += 1.0
    pooled = total / np.maximum(count, 1.0)
    if np.issubdtype(block.dtype, np.integer):
        return np.ascontiguousarray(np.rint(pooled).astype(block.dtype))
    return np.ascontiguousarray(pooled.astype(block.dtype))


def multiscale_paths(group) -> list[str]:
    """Dataset paths of the first multiscale in resolution order (finest first)."""

    attrs = dict(group.attrs)
    if "multiscales" not in attrs:
        raise ValueError("input group carries no OME-NGFF multiscales metadata")
    return [str(entry["path"]) for entry in attrs["multiscales"][0]["datasets"]]


def occupied_chunks(array, array_dir: Path | None) -> list[tuple[int, int, int]]:
    """Chunk indices that hold data: from the on-disk store listing when available, otherwise
    by testing every chunk for a non-zero voxel."""

    grid = tuple(-(-int(s) // int(c)) for s, c in zip(array.shape, array.chunks))
    if array_dir is not None and array_dir.is_dir():
        found = set()
        for entry in array_dir.rglob("*"):
            if not entry.is_file():
                continue
            match = _CHUNK_KEY.match(entry.relative_to(array_dir).as_posix())
            if match:
                index = tuple(int(match.group(i)) for i in (1, 2, 3))
                if all(index[axis] < grid[axis] for axis in range(3)):
                    found.add(index)
        return sorted(found)
    result = []
    for index in itertools.product(*(range(g) for g in grid)):
        bounds = tuple(
            slice(index[axis] * int(array.chunks[axis]), min(int(array.shape[axis]), (index[axis] + 1) * int(array.chunks[axis])))
            for axis in range(3)
        )
        if np.any(np.asarray(array[bounds])):
            result.append(index)
    return result


def create_pyramid(output_path: Path, shape_zyx, chunks_zyx, levels: int):
    """Create a Zarr-v2 OME pyramid with the native writer's layout and return its arrays."""

    from vesuvius.ink_detection.inference.infer_full3d_tifxyz import multiscales_metadata, pyramid_shapes

    group = open_v2_group(output_path)
    group.attrs.update(multiscales_metadata(output_path.stem, levels))
    arrays = []
    for level, shape in enumerate(pyramid_shapes(tuple(int(v) for v in shape_zyx), levels)):
        chunks = tuple(min(int(chunks_zyx[axis]), shape[axis]) for axis in range(3))
        array = create_v2_array(
            group, str(level), shape=shape, chunks=chunks, dtype=np.uint8, compressor=_COMPRESSOR, fill_value=0
        )
        array.attrs["_ARRAY_DIMENSIONS"] = ARRAY_DIMENSIONS
        arrays.append(array)
    return arrays


def enhance_pyramid(input_path: Path, output_path: Path, *, radius: int = 2, levels: int | None = None) -> dict:
    """Sparse, chunk-wise enhancement of a native prediction pyramid into a new pyramid."""

    source = zarr.open(str(input_path), mode="r")
    paths = multiscale_paths(source)
    finest = source[paths[0]]
    level_count = int(levels or len(paths))
    chunks = tuple(int(c) for c in finest.chunks)
    shape = tuple(int(s) for s in finest.shape)
    arrays = create_pyramid(Path(output_path), shape, chunks, level_count)
    occupied = occupied_chunks(finest, Path(input_path) / paths[0])
    for index in occupied:
        starts = [index[axis] * chunks[axis] for axis in range(3)]
        stops = [min(shape[axis], starts[axis] + chunks[axis]) for axis in range(3)]
        lo = [max(0, starts[axis] - radius) for axis in range(3)]
        hi = [min(shape[axis], stops[axis] + radius) for axis in range(3)]
        block = to_unit(np.asarray(finest[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]))
        enhanced = enhance(block, radius)
        core = enhanced[
            starts[0] - lo[0] : starts[0] - lo[0] + (stops[0] - starts[0]),
            starts[1] - lo[1] : starts[1] - lo[1] + (stops[1] - starts[1]),
            starts[2] - lo[2] : starts[2] - lo[2] + (stops[2] - starts[2]),
        ]
        arrays[0][starts[0] : stops[0], starts[1] : stops[1], starts[2] : stops[2]] = encode_uint8(core)
    touched: Iterable[tuple[int, int, int]] = set(occupied)
    for level in range(1, level_count):
        previous, current = arrays[level - 1], arrays[level]
        previous_chunks = tuple(int(c) for c in previous.chunks)
        next_touched = set()
        for index in sorted(touched):
            starts = [index[axis] * previous_chunks[axis] for axis in range(3)]
            stops = [min(int(previous.shape[axis]), starts[axis] + previous_chunks[axis]) for axis in range(3)]
            if any(stops[axis] <= starts[axis] for axis in range(3)):
                continue
            pooled = downsample_mean_3d(np.asarray(previous[starts[0] : stops[0], starts[1] : stops[1], starts[2] : stops[2]]))
            target = [starts[axis] // 2 for axis in range(3)]
            current[
                target[0] : target[0] + pooled.shape[0],
                target[1] : target[1] + pooled.shape[1],
                target[2] : target[2] + pooled.shape[2],
            ] = pooled
            next_touched.add(tuple(target[axis] // int(current.chunks[axis]) for axis in range(3)))
        touched = next_touched
    return {"levels": level_count, "occupied_chunks": len(occupied), "chunks": list(chunks), "shape": list(shape)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_zarr", type=Path, help="native prediction OME-Zarr written by infer_full3d_tifxyz")
    parser.add_argument("output_zarr", type=Path, help="destination OME-Zarr (same layout)")
    parser.add_argument("--radius", type=int, default=2, help="plane window half-size (default 2 -> 5x5)")
    parser.add_argument("--levels", type=int, default=None, help="pyramid levels to write (default: as input)")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output")
    parser.add_argument("--receipt", type=Path, default=None, help="optional JSON run receipt")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.output_zarr.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {args.output_zarr}")
    started = time.time()
    info = enhance_pyramid(args.input_zarr, args.output_zarr, radius=args.radius, levels=args.levels)
    receipt = {
        "input": str(args.input_zarr),
        "output": str(args.output_zarr),
        "radius": args.radius,
        "seconds": round(time.time() - started, 3),
        **info,
    }
    if args.receipt is not None:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
