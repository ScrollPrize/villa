#!/usr/bin/env python3
"""Recompress a local zarr array/tree with the VCZ1 rANS codec.

This is intentionally a small zarr-python example:

    python scripts/recompress_zarr.py \
        /home/sean/Desktop/paris4_level5only.zarr \
        /home/sean/Desktop/paris4_level5only.vcz1.zarr

The output is zarr v2 metadata with compressor id "vcz1".  The chunks are
self-describing VCZ1 payloads written by vc.compression.vcz1, using rANS by
default.
zarr-python 2.x and 3.x both use the numcodecs registration below for v2
arrays; under zarr-python 3.x the script passes zarr_format=2 explicitly.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable

try:
    import zarr
except ImportError as e:  # pragma: no cover
    raise SystemExit("install zarr: python -m pip install zarr") from e

import numpy as np

try:
    from vc.compression.vcz1_numcodecs import Vcz1
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "install the volume-cartographer Python bindings so "
        "`from vc.compression.vcz1_numcodecs import Vcz1` works"
    ) from e

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


def zarr_major() -> int:
    return int(zarr.__version__.split(".", 1)[0])


def is_s3_path(path: str) -> bool:
    return path.startswith("s3://")


def normalize_s3_path(path: str) -> str:
    if not path.startswith("s3://") or ".s3." not in path:
        return path

    prefix = "s3://"
    rest = path[len(prefix):]
    host, sep, key = rest.partition("/")
    bucket = host.split(".s3.", 1)[0]
    return f"{prefix}{bucket}{sep}{key}" if sep else f"{prefix}{bucket}"


def open_s3_store(path: str, *, check: bool = False):
    try:
        import s3fs
    except ImportError as e:  # pragma: no cover
        raise SystemExit("install s3fs to read s3:// inputs: python -m pip install s3fs") from e

    fs = s3fs.S3FileSystem(anon=True)
    return s3fs.S3Map(root=normalize_s3_path(path).rstrip("/"), s3=fs, check=check)


def open_zarr(path: str, mode: str = "r"):
    if is_s3_path(path):
        return zarr.open(open_s3_store(path), mode=mode)
    return zarr.open(path, mode=mode)


def create_array_for_spec(
    path: Path,
    *,
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
    dtype,
    fill_value,
    codec: Vcz1,
):
    kwargs = dict(
        shape=shape,
        chunks=chunks,
        dtype=dtype,
        compressor=codec,
        fill_value=fill_value,
        overwrite=True,
    )
    if zarr_major() >= 3:
        kwargs["zarr_format"] = 2
    return zarr.create(store=str(path), **kwargs)


def create_array(path: Path, src, *, codec: Vcz1):
    return create_array_for_spec(
        path,
        shape=tuple(src.shape),
        chunks=tuple(src.chunks),
        dtype=src.dtype,
        fill_value=getattr(src, "fill_value", None),
        codec=codec,
    )


def chunk_slice_for_index(
    idx: tuple[int, ...],
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
) -> tuple[slice, ...]:
    return tuple(
        slice(i * c, min((i + 1) * c, s))
        for i, c, s in zip(idx, chunks, shape)
    )


def chunk_indices(shape: tuple[int, ...], chunks: tuple[int, ...]) -> Iterable[tuple[int, ...]]:
    grid_shape = tuple(math.ceil(s / c) for s, c in zip(shape, chunks))
    yield from np.ndindex(*grid_shape)


def copy_chunk(job: tuple[str, str, tuple[int, ...], tuple[int, ...], tuple[int, ...]]) -> int:
    src_path, dst_path, idx, shape, chunks = job
    src = open_zarr(src_path, mode="r")
    dst = zarr.open(dst_path, mode="a")
    slc = chunk_slice_for_index(idx, shape, chunks)
    dst[slc] = src[slc]
    return int(np.prod([s.stop - s.start for s in slc]))


def downsample_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(max(1, math.ceil(s / 2)) for s in shape)


def downsample_chunks(chunks: tuple[int, ...], shape: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(min(c, s) for c, s in zip(chunks, shape))


def downsample_nearest_2x(src: np.ndarray) -> np.ndarray:
    return src[::2, ::2, ::2]


def downsample_mean_2x(src: np.ndarray, out_shape: tuple[int, ...]) -> np.ndarray:
    original_shape = tuple(src.shape)
    pad_width = [(0, out_shape[dim] * 2 - original_shape[dim]) for dim in range(3)]
    if any(after for _before, after in pad_width):
        src = np.pad(src, pad_width, mode="constant", constant_values=0)

    sums = src.reshape(
        out_shape[0], 2,
        out_shape[1], 2,
        out_shape[2], 2,
    ).sum(axis=(1, 3, 5), dtype=np.uint64)
    counts_1d = [
        np.minimum(
            2,
            original_shape[dim] - 2 * np.arange(out_shape[dim], dtype=np.uint64),
        )
        for dim in range(3)
    ]
    counts = (
        counts_1d[0][:, None, None]
        * counts_1d[1][None, :, None]
        * counts_1d[2][None, None, :]
    )
    return ((sums + counts // 2) // counts).astype(src.dtype, copy=False)


def downsample_chunk(
    job: tuple[str, str, tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...], str],
) -> int:
    src_path, dst_path, idx, dst_shape, dst_chunks, src_shape, downsample_type = job
    src = zarr.open(src_path, mode="r")
    dst = zarr.open(dst_path, mode="a")
    dst_slc = chunk_slice_for_index(idx, dst_shape, dst_chunks)
    src_slc = tuple(
        slice(s.start * 2, min(s.stop * 2, src_shape[dim]))
        for dim, s in enumerate(dst_slc)
    )
    block = np.asarray(src[src_slc])
    out_shape = tuple(s.stop - s.start for s in dst_slc)
    if downsample_type == "nearest":
        downsampled = downsample_nearest_2x(block)
    else:
        downsampled = downsample_mean_2x(block, out_shape)
    dst[dst_slc] = downsampled
    return int(np.prod(out_shape))


def array_dirs(root: Path) -> list[Path]:
    if (root / ".zarray").exists():
        return [root]
    return sorted(p.parent for p in root.rglob(".zarray"))


def copy_group_metadata(src_root: Path, dst_root: Path, arrays: list[Path]) -> None:
    array_meta = {".zarray"}
    for path in src_root.rglob("*"):
        if not path.is_file() or path.name not in {".zattrs", ".zgroup"}:
            continue
        if path.name in array_meta:
            continue
        rel = path.relative_to(src_root)
        dst = dst_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dst)


def s3_array_paths(root: str) -> list[str]:
    opened = open_zarr(root, mode="r")
    if not isinstance(opened, zarr.Group):
        return [root]

    arrays: list[str] = []

    def visit(path: str, node: Any) -> None:
        if isinstance(node, zarr.Array):
            arrays.append(path)

    opened.visititems(visit)
    return [f"{root.rstrip('/')}/{path}" for path in sorted(arrays)]


def copy_s3_group_metadata(src_root: str, dst_root: Path) -> None:
    store = open_s3_store(src_root)
    for key in store.keys():
        if not key.endswith((".zattrs", ".zgroup")):
            continue
        dst = dst_root / key
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(store[key])


def relative_s3_array_path(src_root: str, src_array: str) -> Path:
    rel = src_array.removeprefix(src_root.rstrip("/")).lstrip("/")
    return Path(rel)


def recompress_array(src_path: str | Path, dst_path: Path, codec: Vcz1, workers: int) -> None:
    src_path_str = str(src_path)
    src = open_zarr(src_path_str, mode="r")
    if len(src.shape) != 3:
        raise SystemExit(f"{src_path}: expected a 3D array, got shape {src.shape}")
    if np.dtype(src.dtype) not in (np.dtype("uint8"), np.dtype("uint16")):
        raise SystemExit(f"{src_path}: expected uint8 or uint16, got {src.dtype}")

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst = create_array(dst_path, src, codec=codec)
    try:
        dst.attrs.update(dict(src.attrs))
    except Exception:
        pass

    print(f"{src_path} -> {dst_path}")
    print(f"  shape={src.shape} chunks={src.chunks} dtype={src.dtype} codec={codec.get_config()}")
    shape = tuple(src.shape)
    chunks = tuple(src.chunks)
    jobs = [
        (src_path_str, str(dst_path), idx, shape, chunks)
        for idx in chunk_indices(shape, chunks)
    ]
    if workers <= 1:
        results = map(copy_chunk, jobs)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = pool.map(copy_chunk, jobs, chunksize=8)
    if tqdm is not None:
        src_name = Path(src_path_str.rstrip("/")).name if not is_s3_path(src_path_str) else src_path_str.rstrip("/").rsplit("/", 1)[-1]
        results = tqdm(results, total=len(jobs), unit="chunk", desc=f"  {src_name}")
    n = 0
    for n, _voxels in enumerate(results, 1):
        if tqdm is None and (n % 100 == 0 or n == len(jobs)):
            print(f"  {n}/{len(jobs)} chunks", flush=True)
    print(f"  done: {n} chunks")


def write_ome_zattrs(
    dst_root: Path,
    *,
    name: str,
    downsample_type: str,
    levels: int = 6,
) -> None:
    datasets = []
    for level in range(levels):
        scale = float(2 ** level)
        datasets.append({
            "path": str(level),
            "coordinateTransformations": [
                {"type": "scale", "scale": [scale, scale, scale]},
            ],
        })

    attrs = {
        "note_axes_order": "ZYX (slice, row, col)",
        "pyramid": True,
        "pyramid_levels": levels - 1,
        "multiscales": [{
            "version": "0.4",
            "name": name,
            "axes": [
                {"name": "z", "type": "space"},
                {"name": "y", "type": "space"},
                {"name": "x", "type": "space"},
            ],
            "datasets": datasets,
            "metadata": {"downsampling_method": downsample_type},
        }],
    }
    (dst_root / ".zattrs").write_text(json.dumps(attrs, indent=2) + "\n")


def create_ome_root(dst_root: Path, *, name: str, downsample_type: str) -> None:
    kwargs = {"mode": "w"}
    if zarr_major() >= 3:
        kwargs["zarr_format"] = 2
    zarr.open_group(str(dst_root), **kwargs)
    write_ome_zattrs(dst_root, name=name, downsample_type=downsample_type)


def recompress_flat_array_as_ome(
    src_path: str | Path,
    dst_root: Path,
    codec: Vcz1,
    workers: int,
    downsample_type: str = "mean",
) -> None:
    src_path_str = str(src_path)
    src = open_zarr(src_path_str, mode="r")
    if len(src.shape) != 3:
        raise SystemExit(f"{src_path}: expected a 3D array, got shape {src.shape}")
    if np.dtype(src.dtype) not in (np.dtype("uint8"), np.dtype("uint16")):
        raise SystemExit(f"{src_path}: expected uint8 or uint16, got {src.dtype}")

    create_ome_root(
        dst_root,
        name=Path(src_path_str.rstrip("/")).name or "/",
        downsample_type=downsample_type,
    )
    recompress_array(src_path, dst_root / "0", codec, workers)

    prev_path = dst_root / "0"
    prev_shape = tuple(src.shape)
    chunks = tuple(src.chunks)
    fill_value = getattr(src, "fill_value", None)
    for level in range(1, 6):
        shape = downsample_shape(prev_shape)
        level_path = dst_root / str(level)
        level_chunks = downsample_chunks(chunks, shape)
        create_array_for_spec(
            level_path,
            shape=shape,
            chunks=level_chunks,
            dtype=src.dtype,
            fill_value=fill_value,
            codec=codec,
        )
        print(f"{prev_path} -> {level_path}")
        print(f"  shape={shape} chunks={level_chunks} dtype={src.dtype} downsample={downsample_type}2x")
        jobs = [
            (str(prev_path), str(level_path), idx, shape, level_chunks, prev_shape, downsample_type)
            for idx in chunk_indices(shape, level_chunks)
        ]
        if workers <= 1:
            results = map(downsample_chunk, jobs)
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                results = pool.map(downsample_chunk, jobs, chunksize=8)
        if tqdm is not None:
            results = tqdm(results, total=len(jobs), unit="chunk", desc=f"  level {level}")
        n = 0
        for n, _voxels in enumerate(results, 1):
            if tqdm is None and (n % 100 == 0 or n == len(jobs)):
                print(f"  {n}/{len(jobs)} chunks", flush=True)
        print(f"  done: {n} chunks")
        prev_path = level_path
        prev_shape = shape


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=str, help="input zarr array or group; local path or s3:// URI")
    parser.add_argument("output", type=Path, help="output zarr array or group")
    parser.add_argument("--codec", choices=("rans", "zstd"), default="rans")
    parser.add_argument(
        "--quant",
        type=int,
        default=1,
        help="VCZ1 quantization bin width; 1 is lossless",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="delete the output path first if it already exists",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="parallel worker processes for chunkwise copy/compression",
    )
    parser.add_argument(
        "--output-ome",
        action="store_true",
        help="write a flat input array as an OME-Zarr pyramid at output/{0..5}",
    )
    parser.add_argument(
        "--downsample-type",
        choices=("mean", "nearest"),
        default="mean",
        help="2x reduction for --output-ome pyramid levels",
    )
    args = parser.parse_args()

    input_path = normalize_s3_path(args.input)
    input_is_s3 = is_s3_path(input_path)

    if input_is_s3:
        input_store = open_s3_store(input_path, check=False)
        if ".zarray" not in input_store and ".zgroup" not in input_store:
            raise SystemExit(f"{input_path}: no .zarray or .zgroup found")
    else:
        input_path = str(Path(input_path))
        if not Path(input_path).exists():
            raise SystemExit(f"{input_path}: does not exist")
    if args.output.exists():
        if not args.overwrite:
            raise SystemExit(f"{args.output}: already exists; pass --overwrite")
        shutil.rmtree(args.output)

    arrays = s3_array_paths(input_path) if input_is_s3 else array_dirs(Path(input_path))
    if not arrays:
        raise SystemExit(f"{input_path}: no .zarray found")

    codec = Vcz1(codec=args.codec, quant=args.quant)
    is_single_array = arrays == [input_path] if input_is_s3 else arrays == [Path(input_path)]
    if args.output_ome:
        if not is_single_array:
            raise SystemExit("--output-ome requires the input to be a single root-level zarr array")
        recompress_flat_array_as_ome(
            input_path,
            args.output,
            codec,
            args.workers,
            args.downsample_type,
        )
    elif is_single_array:
        recompress_array(input_path, args.output, codec, args.workers)
    else:
        if input_is_s3:
            copy_s3_group_metadata(input_path, args.output)
        else:
            copy_group_metadata(Path(input_path), args.output, arrays)
        for src_array in arrays:
            dst_rel = relative_s3_array_path(input_path, src_array) if input_is_s3 else src_array.relative_to(input_path)
            recompress_array(
                src_array,
                args.output / dst_rel,
                codec,
                args.workers,
            )


if __name__ == "__main__":
    main()
