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


def create_array(path: Path, src, *, codec: Vcz1):
    kwargs = dict(
        shape=src.shape,
        chunks=src.chunks,
        dtype=src.dtype,
        compressor=codec,
        fill_value=getattr(src, "fill_value", None),
        overwrite=True,
    )
    if zarr_major() >= 3:
        kwargs["zarr_format"] = 2
    return zarr.create(store=str(path), **kwargs)


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


def array_dirs(root: Path) -> list[Path]:
    if (root / ".zarray").exists():
        return [root]
    return sorted(p.parent for p in root.rglob(".zarray"))


def copy_group_metadata(src_root: Path, dst_root: Path, arrays: list[Path]) -> None:
    array_meta = {".zarray"}
    for path in src_root.rglob("*"):
        if not path.is_file() or path.name not in {".zattrs", ".zgroup", ".zmetadata"}:
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
        if not key.endswith((".zattrs", ".zgroup", ".zmetadata")):
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
    if is_single_array:
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
