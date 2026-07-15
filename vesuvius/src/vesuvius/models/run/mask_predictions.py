"""Mask an existing finalized binary prediction with an aligned support volume."""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import posixpath
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numcodecs
import numpy as np
from tqdm.auto import tqdm

from vesuvius.data.utils import open_zarr
from vesuvius.models.run.finalize_outputs import (
    apply_support_mask,
    open_support_volume,
    read_support_chunk,
    validate_support_volume,
)
from vesuvius.utils.k8s import get_tqdm_kwargs


_worker_state = {}
_MAX_DEFAULT_CHUNK_BYTES = 256 * 1024 * 1024


def _canonical_store_path(path: str):
    """Return a comparable local path or normalized remote store URL."""
    if "://" in path:
        scheme, remainder = path.split("://", 1)
        normalized = posixpath.normpath(remainder.replace("\\", "/"))
        return "remote", f"{scheme.lower()}://{normalized.rstrip('/')}"
    resolved = Path(path).resolve(strict=False)
    return "local", os.path.normcase(str(resolved))


def _stores_overlap(first: str, second: str) -> bool:
    """Return whether two store paths are equal or one contains the other."""
    first_kind, first_path = _canonical_store_path(first)
    second_kind, second_path = _canonical_store_path(second)
    if first_kind != second_kind:
        return False
    separator = "/" if first_kind == "remote" else os.sep
    return (
        first_path == second_path
        or first_path.startswith(second_path + separator)
        or second_path.startswith(first_path + separator)
    )


def _prediction_spatial_shape(prediction_store) -> tuple[int, int, int]:
    shape = tuple(int(value) for value in prediction_store.shape)
    if len(shape) == 3:
        return shape
    if len(shape) == 4 and shape[0] == 1:
        return shape[1:]
    raise ValueError(
        "Finalized binary predictions must have shape (Z, Y, X) or "
        f"(1, Z, Y, X), got {shape}"
    )


def _array_spatial_chunks(array):
    chunks = tuple(int(value) for value in array.chunks)
    return chunks[1:] if len(array.shape) == 4 else chunks


def _spatial_chunks(prediction_store, support_store, chunk_size):
    if chunk_size is not None:
        if len(chunk_size) != 3 or any(int(value) <= 0 for value in chunk_size):
            raise ValueError("chunk_size must contain three positive Z,Y,X values")
        return tuple(int(value) for value in chunk_size)

    prediction_chunks = _array_spatial_chunks(prediction_store)
    support_chunks = _array_spatial_chunks(support_store)
    aligned = tuple(
        math.lcm(prediction_chunk, support_chunk)
        for prediction_chunk, support_chunk in zip(
            prediction_chunks,
            support_chunks,
        )
    )
    bytes_per_voxel = (
        np.dtype(prediction_store.dtype).itemsize
        + np.dtype(support_store.dtype).itemsize
    )
    aligned_bytes = math.prod(aligned) * bytes_per_voxel
    return aligned if aligned_bytes <= _MAX_DEFAULT_CHUNK_BYTES else prediction_chunks


def _spatial_chunk_count(spatial_shape, chunks) -> int:
    return math.prod(
        (size + chunk - 1) // chunk
        for size, chunk in zip(spatial_shape, chunks)
    )


def _iter_spatial_slices(spatial_shape, chunks):
    for z_start in range(0, spatial_shape[0], chunks[0]):
        for y_start in range(0, spatial_shape[1], chunks[1]):
            for x_start in range(0, spatial_shape[2], chunks[2]):
                yield (
                    slice(z_start, min(z_start + chunks[0], spatial_shape[0])),
                    slice(y_start, min(y_start + chunks[1], spatial_shape[1])),
                    slice(x_start, min(x_start + chunks[2], spatial_shape[2])),
                )


def _input_storage_options(path: str, anonymous: bool):
    return {"anon": bool(anonymous)} if path.startswith("s3://") else None


def _init_worker(
    prediction_path: str,
    output_path: str,
    support_path: str,
    support_threshold: float,
    anonymous: bool,
):
    numcodecs.blosc.use_threads = False
    prediction_store = open_zarr(
        prediction_path,
        mode="r",
        storage_options=_input_storage_options(prediction_path, anonymous),
    )
    _worker_state.update(
        {
            "prediction_store": prediction_store,
            "prediction_is_4d": len(prediction_store.shape) == 4,
            "output_store": open_zarr(
                output_path,
                mode="r+",
                storage_options=(
                    {"anon": False} if output_path.startswith("s3://") else None
                ),
            ),
            "support_store": open_support_volume(support_path, anon=anonymous),
            "support_threshold": float(support_threshold),
        }
    )


def _process_chunk(spatial_slices):
    prediction_store = _worker_state["prediction_store"]
    prediction_is_4d = _worker_state["prediction_is_4d"]
    prediction_slice = (
        (slice(None),) + tuple(spatial_slices)
        if prediction_is_4d
        else tuple(spatial_slices)
    )
    prediction = np.asarray(prediction_store[prediction_slice])
    prediction_4d = prediction if prediction_is_4d else prediction[np.newaxis]

    nonzero_before = int(np.count_nonzero(np.any(prediction_4d != 0, axis=0)))
    if nonzero_before == 0:
        return {
            "nonzero_voxels_before": 0,
            "nonzero_voxels_after": 0,
            "nonzero_voxels_removed": 0,
        }

    support = read_support_chunk(_worker_state["support_store"], spatial_slices)
    masked, stats = apply_support_mask(
        prediction_4d,
        support,
        threshold=_worker_state["support_threshold"],
    )
    if stats["nonzero_voxels_after"]:
        output = masked if prediction_is_4d else masked[0]
        _worker_state["output_store"][prediction_slice] = output

    return {
        "nonzero_voxels_before": stats["nonzero_voxels_before"],
        "nonzero_voxels_after": stats["nonzero_voxels_after"],
        "nonzero_voxels_removed": stats["nonzero_voxels_removed"],
    }


def mask_finalized_predictions(
    prediction_path: str,
    output_path: str,
    support_path: str,
    *,
    support_threshold: float = 0.0,
    anonymous: bool = True,
    chunk_size: tuple[int, int, int] | None = None,
    num_workers: int | None = None,
    compression_level: int = 1,
    verbose: bool = True,
) -> dict[str, int | float | str]:
    """Write a support-masked copy of a finalized binary Zarr prediction.

    Only shape is machine-validated. The caller is responsible for asserting
    that prediction and support arrays describe the same physical grid.
    """
    if _stores_overlap(output_path, prediction_path) or _stores_overlap(
        output_path, support_path
    ):
        raise ValueError(
            "output_path must not equal, contain, or be contained by either input store"
        )
    if not np.isfinite(support_threshold):
        raise ValueError(f"support_threshold must be finite, got {support_threshold}")
    if not 1 <= compression_level <= 9:
        raise ValueError("compression_level must be between 1 and 9")

    numcodecs.blosc.use_threads = False
    prediction_store = open_zarr(
        prediction_path,
        mode="r",
        storage_options=_input_storage_options(prediction_path, anonymous),
    )
    spatial_shape = _prediction_spatial_shape(prediction_store)
    support_store = open_support_volume(support_path, anon=anonymous)
    validate_support_volume(support_store, spatial_shape)
    chunks = _spatial_chunks(prediction_store, support_store, chunk_size)
    output_chunks = (
        (1, *chunks) if len(prediction_store.shape) == 4 else chunks
    )

    compressor = numcodecs.Blosc(
        cname="zstd",
        clevel=compression_level,
        shuffle=numcodecs.blosc.SHUFFLE,
    )
    output_store = open_zarr(
        output_path,
        mode="w",
        storage_options=(
            {"anon": False} if output_path.startswith("s3://") else None
        ),
        shape=prediction_store.shape,
        chunks=output_chunks,
        dtype=prediction_store.dtype,
        compressor=compressor,
        fill_value=0,
        config={"write_empty_chunks": False},
        zarr_format=2,
    )

    total_chunks = _spatial_chunk_count(spatial_shape, chunks)
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() // 2)
    if num_workers < 1:
        raise ValueError("num_workers must be at least 1")

    totals = {
        "nonzero_voxels_before": 0,
        "nonzero_voxels_after": 0,
        "nonzero_voxels_removed": 0,
    }
    tqdm_kwargs = get_tqdm_kwargs()
    if not verbose:
        tqdm_kwargs["disable"] = True
    elif chunk_size is None:
        print(
            f"Using shape-aligned chunks {chunks}; processing {total_chunks:,} chunks"
        )

    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_init_worker,
        initargs=(
            prediction_path,
            output_path,
            support_path,
            support_threshold,
            anonymous,
        ),
    ) as executor:
        results = executor.map(
            _process_chunk,
            _iter_spatial_slices(spatial_shape, chunks),
            chunksize=1,
            buffersize=max(2, num_workers * 2),
        )
        for stats in tqdm(
            results,
            total=total_chunks,
            desc="Masking prediction chunks",
            **tqdm_kwargs,
        ):
            for key in totals:
                totals[key] += int(stats[key])

    positives = totals["nonzero_voxels_before"]
    summary = {
        **totals,
        "phantom_fraction_before": (
            float(totals["nonzero_voxels_removed"] / positives)
            if positives
            else 0.0
        ),
        "scope": "all_nonzero_input_predictions",
    }

    if hasattr(prediction_store, "attrs") and hasattr(output_store, "attrs"):
        for key, value in prediction_store.attrs.items():
            output_store.attrs[key] = value
        output_store.attrs["source_prediction_path"] = prediction_path
        output_store.attrs["support_mask_applied"] = True
        output_store.attrs["support_volume_path"] = support_path
        output_store.attrs["support_threshold"] = float(support_threshold)
        output_store.attrs["support_anonymous_access"] = bool(anonymous)
        output_store.attrs["support_alignment_validation"] = (
            "shape_only_physical_alignment_asserted_by_caller"
        )
        output_store.attrs["support_mask_stats"] = summary

    if verbose:
        print(
            f"Removed {summary['nonzero_voxels_removed']:,}/"
            f"{summary['nonzero_voxels_before']:,} nonzero spatial voxels "
            f"({summary['phantom_fraction_before']:.4%}); output: {output_path}"
        )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Mask an existing finalized binary prediction against an aligned "
            "CT/support Zarr array."
        )
    )
    parser.add_argument(
        "prediction_path",
        help="Finalized binary Zarr array with shape (Z,Y,X) or (1,Z,Y,X).",
    )
    parser.add_argument(
        "output_path",
        help="Destination for a new single-level masked Zarr array.",
    )
    parser.add_argument(
        "--support-volume",
        required=True,
        dest="support_volume",
        help="Caller-verified, physically aligned CT/support Zarr array.",
    )
    parser.add_argument(
        "--support-threshold",
        type=float,
        default=0.0,
        help="Support values <= this threshold are background. Default: 0.",
    )
    parser.add_argument(
        "--authenticated-inputs",
        action="store_true",
        help="Use configured credentials for S3 prediction/support inputs.",
    )
    parser.add_argument(
        "--chunk-size",
        default=None,
        help="Optional output spatial chunk size as Z,Y,X.",
    )
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--compression-level",
        type=int,
        default=1,
        choices=range(1, 10),
        help="Blosc zstd output compression level. Default: 1.",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    chunks = None
    if args.chunk_size:
        try:
            chunks = tuple(int(value) for value in args.chunk_size.split(","))
        except ValueError:
            parser.error("--chunk-size must contain three integers as Z,Y,X")
        if len(chunks) != 3 or any(value <= 0 for value in chunks):
            parser.error("--chunk-size must contain three positive integers as Z,Y,X")

    try:
        mask_finalized_predictions(
            prediction_path=args.prediction_path,
            output_path=args.output_path,
            support_path=args.support_volume,
            support_threshold=args.support_threshold,
            anonymous=not args.authenticated_inputs,
            chunk_size=chunks,
            num_workers=args.num_workers,
            compression_level=args.compression_level,
            verbose=not args.quiet,
        )
        return 0
    except Exception as error:
        print(f"Masking failed: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
