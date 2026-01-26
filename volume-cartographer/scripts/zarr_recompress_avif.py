#!/usr/bin/env python3
import argparse
import math
import multiprocessing as mp
import os
from itertools import product

import numcodecs
import zarr
from imagecodecs.numcodecs import Avif
from tqdm import tqdm

_WORKER_SRC = None
_WORKER_DST = None


def build_avif(dtype, level, speed, bitspersample):
    kwargs = {"level": level, "speed": speed}
    if dtype == "uint16":
        if bitspersample is None:
            bitspersample = 10
        if bitspersample not in (10, 12):
            raise ValueError("uint16 AVIF requires bitspersample 10 or 12")
        kwargs["bitspersample"] = bitspersample
    return Avif(**kwargs)


def get_dimension_separator(zarr_obj):
    if hasattr(zarr_obj, "dimension_separator"):
        return zarr_obj.dimension_separator
    if hasattr(zarr_obj, "_dimension_separator"):
        return zarr_obj._dimension_separator
    return None


def normalize_path(path):
    if path is None:
        return ""
    return str(path).lstrip("/")


def iter_groups(root):
    try:
        yield from root.groups(recurse=True)
        return
    except TypeError:
        pass

    def _walk(group):
        for name, sub in group.groups():
            path = normalize_path(getattr(sub, "path", None) or name)
            yield path, sub
            yield from _walk(sub)

    yield from _walk(root)


def iter_arrays(root):
    try:
        yield from root.arrays(recurse=True)
        return
    except TypeError:
        pass

    def _walk(group):
        for name, arr in group.arrays():
            path = normalize_path(getattr(arr, "path", None) or name)
            yield path, arr
        for _, sub in group.groups():
            yield from _walk(sub)

    yield from _walk(root)


def chunk_grid(shape, chunks):
    grid = [math.ceil(s / c) for s, c in zip(shape, chunks)]
    for index in product(*(range(g) for g in grid)):
        yield index


def chunk_slices(index, chunks, shape):
    return tuple(
        slice(i * c, min((i + 1) * c, s))
        for i, c, s in zip(index, chunks, shape)
    )


def _init_worker(src_path, dst_path):
    global _WORKER_SRC, _WORKER_DST
    numcodecs.register_codec(Avif)
    _WORKER_SRC = zarr.open(src_path, mode="r")
    _WORKER_DST = zarr.open(dst_path, mode="r+")


def _get_array(root, array_path):
    if isinstance(root, zarr.Array):
        return root
    return root[array_path]


def _copy_chunk(task):
    array_path, index, shape, chunks = task
    src_arr = _get_array(_WORKER_SRC, array_path)

    chunk_key = src_arr._chunk_key(index)
    src_bytes = len(src_arr.chunk_store[chunk_key])

    dst_arr = _get_array(_WORKER_DST, array_path)
    slices = chunk_slices(index, chunks, shape)
    dst_arr[slices] = src_arr[slices]

    dst_key = dst_arr._chunk_key(index)
    dst_bytes = len(dst_arr.chunk_store[dst_key])

    return (src_bytes, dst_bytes)


def create_array_like(src_arr, dst_group, name, level, speed, bitspersample):
    compressor = build_avif(str(src_arr.dtype), level, speed, bitspersample)
    kwargs = {
        "shape": src_arr.shape,
        "chunks": src_arr.chunks,
        "dtype": src_arr.dtype,
        "order": src_arr.order,
        "fill_value": src_arr.fill_value,
        "compressor": compressor,
        "filters": src_arr.filters,
        "overwrite": True,
        "write_empty_chunks": False,
    }
    dim_sep = get_dimension_separator(src_arr)
    if dim_sep is not None:
        kwargs["dimension_separator"] = dim_sep
    dst_arr = dst_group.create_dataset(name, **kwargs)
    dst_arr.attrs.update(src_arr.attrs)
    return dst_arr


def copy_array_data(src_path, dst_path, array_path, processes):
    src_root = zarr.open(src_path, mode="r")
    src_arr = _get_array(src_root, array_path)
    shape = src_arr.shape
    chunks = src_arr.chunks

    # Pre-filter: only include indices for chunks that exist in source
    existing_indices = []
    for index in chunk_grid(shape, chunks):
        chunk_key = src_arr._chunk_key(index)
        if chunk_key in src_arr.chunk_store:
            existing_indices.append(index)

    total = len(existing_indices)
    if total == 0:
        return

    tasks = ((array_path, index, shape, chunks) for index in existing_indices)

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=processes, initializer=_init_worker, initargs=(src_path, dst_path)) as pool:
        total_src = 0
        total_dst = 0
        pbar = tqdm(
            pool.imap_unordered(_copy_chunk, tasks, chunksize=64),
            total=total,
            desc=array_path or "root",
        )
        for src_bytes, dst_bytes in pbar:
            total_src += src_bytes
            total_dst += dst_bytes
            if total_src > 0:
                ratio = total_dst / total_src
                pbar.set_postfix(ratio=f"{ratio:.2%}")


def main():
    parser = argparse.ArgumentParser(description="Recompress a Zarr store to AVIF/AV1")
    parser.add_argument("--src", required=True, help="Path to input .zarr")
    parser.add_argument("--dst", required=True, help="Path to output .zarr")
    parser.add_argument("--level", type=int, default=60, help="AVIF quality (0-100)")
    parser.add_argument("--speed", type=int, default=6, help="AVIF encoder speed (0-10)")
    parser.add_argument(
        "--bitspersample",
        type=int,
        choices=[10, 12],
        default=None,
        help="Required for uint16 (10 or 12)",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=None,
        help="Worker process count (default: CPU count)",
    )
    args = parser.parse_args()

    numcodecs.register_codec(Avif)

    src = zarr.open(args.src, mode="r")
    if isinstance(src, zarr.Array):
        kwargs = {
            "shape": src.shape,
            "chunks": src.chunks,
            "dtype": src.dtype,
            "order": src.order,
            "fill_value": src.fill_value,
            "compressor": build_avif(str(src.dtype), args.level, args.speed, args.bitspersample),
            "filters": src.filters,
            "write_empty_chunks": False,
        }
        dim_sep = get_dimension_separator(src)
        if dim_sep is not None:
            kwargs["dimension_separator"] = dim_sep
        dst = zarr.open(args.dst, mode="w", **kwargs)
        dst.attrs.update(src.attrs)
        copy_array_data(args.src, args.dst, "", args.processes)
        return

    dst_root = zarr.open_group(args.dst, mode="w")
    dst_root.attrs.update(src.attrs)

    for path, group in iter_groups(src):
        path = normalize_path(path)
        dst_group = dst_root.require_group(path)
        dst_group.attrs.update(group.attrs)

    for path, arr in iter_arrays(src):
        path = normalize_path(path)
        parent = os.path.dirname(path)
        name = os.path.basename(path)
        dst_group = dst_root.require_group(parent) if parent else dst_root
        create_array_like(arr, dst_group, name, args.level, args.speed, args.bitspersample)

    for path, _ in iter_arrays(src):
        path = normalize_path(path)
        copy_array_data(args.src, args.dst, path, args.processes)


if __name__ == "__main__":
    main()
