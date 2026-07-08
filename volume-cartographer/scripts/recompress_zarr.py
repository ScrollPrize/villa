#!/usr/bin/env python3
"""Recompress a local zarr array/tree with the VCZ1 rANS codec.

This is intentionally a small zarr-python example:

    python scripts/recompress_zarr.py \
        /home/sean/Desktop/paris4_level5only.zarr \
        /home/sean/Desktop/paris4_level5only.vcz1.zarr

    python scripts/recompress_zarr.py --in-place \
        /home/sean/Desktop/paris4_level5only.zarr

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
import tempfile
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any, Callable, Iterable

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


def chunk_grid_shape(shape: tuple[int, ...], chunks: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(math.ceil(s / c) for s, c in zip(shape, chunks))


def chunk_grid_size(shape: tuple[int, ...], chunks: tuple[int, ...]) -> int:
    return math.prod(chunk_grid_shape(shape, chunks))


def chunk_indices(shape: tuple[int, ...], chunks: tuple[int, ...]) -> Iterable[tuple[int, ...]]:
    yield from np.ndindex(*chunk_grid_shape(shape, chunks))


def copy_jobs(
    src_path: str,
    dst_path: Path,
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
) -> Iterable[tuple[str, str, tuple[int, ...], tuple[int, ...], tuple[int, ...]]]:
    for idx in chunk_indices(shape, chunks):
        yield (src_path, str(dst_path), idx, shape, chunks)


def downsample_jobs(
    src_path: Path,
    dst_path: Path,
    dst_shape: tuple[int, ...],
    dst_chunks: tuple[int, ...],
    src_shape: tuple[int, ...],
    downsample_type: str,
) -> Iterable[tuple[str, str, tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...], str]]:
    for idx in chunk_indices(dst_shape, dst_chunks):
        yield (str(src_path), str(dst_path), idx, dst_shape, dst_chunks, src_shape, downsample_type)


CopyJob = tuple[str, str, tuple[int, ...], tuple[int, ...], tuple[int, ...]]
DownsampleJob = tuple[str, str, tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...], str]
ChunkJob = CopyJob | DownsampleJob
ChunkFunc = Callable[[Any], int]


def batched(iterable: Iterable[ChunkJob], size: int) -> Iterable[tuple[ChunkJob, ...]]:
    batch: list[ChunkJob] = []
    for item in iterable:
        batch.append(item)
        if len(batch) == size:
            yield tuple(batch)
            batch.clear()
    if batch:
        yield tuple(batch)


def run_chunk_batch(func: ChunkFunc, batch: tuple[ChunkJob, ...]) -> int:
    for job in batch:
        func(job)
    return len(batch)


def run_chunk_jobs(
    func: ChunkFunc,
    jobs: Iterable[ChunkJob],
    *,
    total_jobs: int,
    workers: int,
    desc: str,
    batch_size: int = 8,
) -> None:
    completed = 0

    def update_progress(delta: int) -> None:
        nonlocal completed
        completed += delta
        if progress is not None:
            progress.update(delta)
        elif completed % 100 == 0 or completed == total_jobs:
            print(f"  {completed}/{total_jobs} chunks", flush=True)

    progress = tqdm(total=total_jobs, unit="chunk", desc=desc) if tqdm is not None else None
    try:
        if workers <= 1:
            for job in jobs:
                func(job)
                update_progress(1)
        else:
            max_pending = max(1, workers * 4)
            batches = batched(jobs, batch_size)
            with ProcessPoolExecutor(max_workers=workers) as pool:
                pending: set[Future[int]] = set()
                exhausted = False
                while pending or not exhausted:
                    while not exhausted and len(pending) < max_pending:
                        try:
                            batch = next(batches)
                        except StopIteration:
                            exhausted = True
                            break
                        pending.add(pool.submit(run_chunk_batch, func, batch))

                    if not pending:
                        continue

                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        update_progress(future.result())
    finally:
        if progress is not None:
            progress.close()
    print(f"  done: {completed} chunks")


def copy_chunk(job: tuple[str, str, tuple[int, ...], tuple[int, ...], tuple[int, ...]]) -> int:
    src_path, dst_path, idx, shape, chunks = job
    src = open_zarr(src_path, mode="r")
    dst = zarr.open(dst_path, mode="a")
    slc = chunk_slice_for_index(idx, shape, chunks)
    dst[slc] = src[slc]
    return int(np.prod([s.stop - s.start for s in slc]))


def load_zarray_metadata(path: Path) -> dict[str, Any]:
    return json.loads((path / ".zarray").read_text())


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, indent=4, sort_keys=True)
            f.write("\n")
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def in_place_state_path(path: Path) -> Path:
    return path / ".recompress_zarr_in_place.json"


def write_in_place_state(path: Path, original_meta: dict[str, Any], target_meta: dict[str, Any]) -> None:
    write_json_atomic(
        in_place_state_path(path),
        {
            "version": 1,
            "original_meta": original_meta,
            "target_meta": target_meta,
        },
    )


def load_in_place_state(path: Path) -> dict[str, Any] | None:
    state_path = in_place_state_path(path)
    if not state_path.exists():
        return None
    state = json.loads(state_path.read_text())
    if state.get("version") != 1:
        raise SystemExit(f"{state_path}: unsupported in-place recovery state")
    if not isinstance(state.get("original_meta"), dict) or not isinstance(state.get("target_meta"), dict):
        raise SystemExit(f"{state_path}: invalid in-place recovery state")
    return state


def validate_in_place_metadata(path: Path, meta: dict[str, Any]) -> None:
    if meta.get("zarr_format") != 2:
        raise SystemExit(f"{path}: --in-place only supports zarr v2 arrays")
    if meta.get("filters") not in (None, []):
        raise SystemExit(f"{path}: --in-place does not support arrays with filters")
    if meta.get("order", "C") != "C":
        raise SystemExit(f"{path}: --in-place only supports C-order arrays")


def chunk_storage_path(array_path: Path, meta: dict[str, Any], idx: tuple[int, ...]) -> Path:
    separator = meta.get("dimension_separator", ".")
    if separator == "/":
        return array_path.joinpath(*(str(i) for i in idx))
    if separator == ".":
        return array_path / ".".join(str(i) for i in idx)
    raise SystemExit(f"{array_path}: unsupported dimension_separator {separator!r}")


def write_chunk_payload_atomic(path: Path, payload: bytes | bytearray | memoryview) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(payload)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def chunk_has_vcz1_payload(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            return f.read(4) == b"VCZ1"
    except FileNotFoundError:
        return False


def remove_consolidated_metadata(root: Path) -> None:
    for path in root.rglob(".zmetadata"):
        path.unlink()
        print(f"  removed stale consolidated metadata: {path}")


InPlaceJob = tuple[
    str,
    dict[str, Any],
    dict[str, Any],
    bool,
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
]


def in_place_jobs(
    src_path: Path,
    original_meta: dict[str, Any],
    codec_config: dict[str, Any],
    skip_existing_vcz1: bool,
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
) -> Iterable[InPlaceJob]:
    for idx in chunk_indices(shape, chunks):
        yield (str(src_path), original_meta, codec_config, skip_existing_vcz1, idx, shape, chunks)


def recompress_chunk_in_place(job: InPlaceJob) -> int:
    src_path_str, original_meta, codec_config, skip_existing_vcz1, idx, shape, chunks = job
    src_path = Path(src_path_str)
    slc = chunk_slice_for_index(idx, shape, chunks)
    chunk_path = chunk_storage_path(src_path, original_meta, idx)
    if skip_existing_vcz1 and chunk_has_vcz1_payload(chunk_path):
        return int(np.prod([s.stop - s.start for s in slc]))

    src = zarr.open(src_path_str, mode="r")
    block = np.ascontiguousarray(src[slc])
    codec = Vcz1(**{k: v for k, v in codec_config.items() if k != "id"})
    payload = codec.encode(block)
    write_chunk_payload_atomic(chunk_path, payload)
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
    total_jobs = chunk_grid_size(shape, chunks)
    jobs = copy_jobs(src_path_str, dst_path, shape, chunks)
    src_name = Path(src_path_str.rstrip("/")).name if not is_s3_path(src_path_str) else src_path_str.rstrip("/").rsplit("/", 1)[-1]
    run_chunk_jobs(copy_chunk, jobs, total_jobs=total_jobs, workers=workers, desc=f"  {src_name}")


def recompress_array_in_place(src_path: Path, codec: Vcz1, workers: int) -> None:
    state = load_in_place_state(src_path)
    if state is not None:
        requested_target_meta: dict[str, Any] = dict(state["target_meta"])
        requested_target_meta["compressor"] = codec.get_config()
        if requested_target_meta != state["target_meta"]:
            raise SystemExit(
                f"{in_place_state_path(src_path)}: unfinished in-place run uses a different codec; "
                "rerun with the same --codec/--quant to recover"
            )
        write_json_atomic(src_path / ".zarray", state["original_meta"])
        print(f"{src_path}: resuming interrupted in-place recompression")

    src = zarr.open(str(src_path), mode="r")
    if len(src.shape) != 3:
        raise SystemExit(f"{src_path}: expected a 3D array, got shape {src.shape}")
    if np.dtype(src.dtype) not in (np.dtype("uint8"), np.dtype("uint16")):
        raise SystemExit(f"{src_path}: expected uint8 or uint16, got {src.dtype}")

    original_meta = state["original_meta"] if state is not None else load_zarray_metadata(src_path)
    validate_in_place_metadata(src_path, original_meta)

    codec_config = codec.get_config()
    new_meta = dict(original_meta)
    new_meta["compressor"] = codec_config
    if state is None:
        write_in_place_state(src_path, original_meta, new_meta)

    shape = tuple(src.shape)
    chunks = tuple(src.chunks)
    total_jobs = chunk_grid_size(shape, chunks)
    print(f"{src_path} -> {src_path} (in-place)")
    print(f"  shape={src.shape} chunks={src.chunks} dtype={src.dtype} codec={codec_config}")
    print("  recovery: rerun the same --in-place command to continue an interrupted conversion")
    original_compressor = original_meta.get("compressor")
    skip_existing_vcz1 = not (
        isinstance(original_compressor, dict)
        and original_compressor.get("id") == "vcz1"
        and original_compressor != codec_config
    )
    jobs = in_place_jobs(src_path, original_meta, codec_config, skip_existing_vcz1, shape, chunks)
    run_chunk_jobs(
        recompress_chunk_in_place,
        jobs,
        total_jobs=total_jobs,
        workers=workers,
        desc=f"  {src_path.name}",
    )
    write_json_atomic(src_path / ".zarray", new_meta)
    in_place_state_path(src_path).unlink(missing_ok=True)


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
        total_jobs = chunk_grid_size(shape, level_chunks)
        jobs = downsample_jobs(prev_path, level_path, shape, level_chunks, prev_shape, downsample_type)
        run_chunk_jobs(downsample_chunk, jobs, total_jobs=total_jobs, workers=workers, desc=f"  level {level}")
        prev_path = level_path
        prev_shape = shape


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=str, help="input zarr array or group; local path or s3:// URI")
    parser.add_argument("output", nargs="?", type=Path, help="output zarr array or group")
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
        "--in-place",
        action="store_true",
        help="rewrite local input chunks in place; output must be omitted",
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

    if args.in_place:
        if args.output is not None:
            raise SystemExit("--in-place rewrites the input path; omit output")
        if args.overwrite:
            raise SystemExit("--overwrite is only valid when writing a separate output")
        if args.output_ome:
            raise SystemExit("--output-ome requires a separate output path")
        if input_is_s3:
            raise SystemExit("--in-place only supports local zarr paths")
    elif args.output is None:
        raise SystemExit("output is required unless --in-place is used")

    if input_is_s3:
        input_store = open_s3_store(input_path, check=False)
        if ".zarray" not in input_store and ".zgroup" not in input_store:
            raise SystemExit(f"{input_path}: no .zarray or .zgroup found")
    else:
        input_path = str(Path(input_path))
        if not Path(input_path).exists():
            raise SystemExit(f"{input_path}: does not exist")
    if args.output is not None and args.output.exists():
        if not args.overwrite:
            raise SystemExit(f"{args.output}: already exists; pass --overwrite")
        shutil.rmtree(args.output)

    arrays = s3_array_paths(input_path) if input_is_s3 else array_dirs(Path(input_path))
    if not arrays:
        raise SystemExit(f"{input_path}: no .zarray found")

    codec = Vcz1(codec=args.codec, quant=args.quant)
    is_single_array = arrays == [input_path] if input_is_s3 else arrays == [Path(input_path)]
    if args.in_place:
        for src_array in arrays:
            recompress_array_in_place(src_array, codec, args.workers)
        remove_consolidated_metadata(Path(input_path))
    elif args.output_ome:
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
