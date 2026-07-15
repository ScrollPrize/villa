"""Audit phantom binary predictions against an aligned CT/support volume.

The benchmark streams only the requested Z planes. It is intended for
reproducing issue #1114 without downloading a complete scroll volume.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import zarr

from vesuvius.data.utils import open_zarr
from vesuvius.models.run.finalize_outputs import (
    apply_support_mask,
    open_support_volume,
    validate_support_volume,
)


def _spatial_shape(array) -> tuple[int, int, int]:
    shape = tuple(int(value) for value in array.shape)
    if len(shape) == 3:
        return shape
    if len(shape) == 4 and shape[0] == 1:
        return shape[1:]
    raise ValueError(
        "Arrays must have shape (Z, Y, X) or (1, Z, Y, X), "
        f"got {shape}"
    )


def _read_plane(array, z_index: int) -> np.ndarray:
    if len(array.shape) == 4:
        return np.asarray(array[0, z_index, :, :])
    return np.asarray(array[z_index, :, :])


def _plane_chunk_bytes_upper_bound(array, z_index: int) -> int:
    """Estimate uncompressed bytes in all 3-D chunks intersecting one plane.

    This is an upper bound, not network transfer: sparse/missing chunks need no
    payload and compression changes bytes on the wire.
    """
    spatial_shape = _spatial_shape(array)
    chunks = tuple(int(value) for value in array.chunks)
    spatial_chunks = chunks[1:] if len(array.shape) == 4 else chunks
    z_start = (z_index // spatial_chunks[0]) * spatial_chunks[0]
    z_size = min(spatial_chunks[0], spatial_shape[0] - z_start)
    voxels = 0
    for y_start in range(0, spatial_shape[1], spatial_chunks[1]):
        y_size = min(spatial_chunks[1], spatial_shape[1] - y_start)
        for x_start in range(0, spatial_shape[2], spatial_chunks[2]):
            x_size = min(spatial_chunks[2], spatial_shape[2] - x_start)
            voxels += z_size * y_size * x_size
    return int(voxels * np.dtype(array.dtype).itemsize)


def evaluate_plane(
    prediction: np.ndarray,
    support: np.ndarray,
    *,
    prediction_threshold: float = 0.0,
    support_threshold: float = 0.0,
) -> dict[str, int | float | bool]:
    """Measure phantom positives before and after production support masking."""
    prediction = np.asarray(prediction)
    support = np.asarray(support)
    if prediction.ndim != 2 or support.ndim != 2:
        raise ValueError("prediction and support planes must both be 2D")
    if prediction.shape != support.shape:
        raise ValueError(
            f"Prediction/support plane mismatch: {prediction.shape} vs {support.shape}"
        )
    if not np.isfinite(prediction_threshold):
        raise ValueError("prediction_threshold must be finite")

    positive_before = prediction > prediction_threshold
    supported = np.isfinite(support) & (support > support_threshold)
    phantom_before = positive_before & ~supported

    masked, mask_stats = apply_support_mask(
        prediction[np.newaxis, np.newaxis, :, :],
        support[np.newaxis, :, :],
        threshold=support_threshold,
    )
    masked_plane = masked[0, 0]
    positive_after = masked_plane > prediction_threshold
    phantom_after = positive_after & ~supported

    positives_before = int(np.count_nonzero(positive_before))
    positives_after = int(np.count_nonzero(positive_after))
    phantom_positives_before = int(np.count_nonzero(phantom_before))
    phantom_positives_after = int(np.count_nonzero(phantom_after))
    spatial_voxels = int(prediction.size)

    return {
        "spatial_voxels": spatial_voxels,
        "supported_voxels": int(np.count_nonzero(supported)),
        "support_fraction": float(np.count_nonzero(supported) / spatial_voxels),
        "positive_voxels_before": positives_before,
        "positive_voxels_after": positives_after,
        "phantom_positives_before": phantom_positives_before,
        "phantom_positives_after": phantom_positives_after,
        "phantom_fraction_before": (
            float(phantom_positives_before / positives_before)
            if positives_before
            else 0.0
        ),
        "phantom_fraction_after": (
            float(phantom_positives_after / positives_after)
            if positives_after
            else 0.0
        ),
        "supported_values_preserved": bool(
            np.array_equal(masked_plane[supported], prediction[supported])
        ),
        "nonzero_voxels_removed": int(mask_stats["nonzero_voxels_removed"]),
    }


def run_benchmark(
    prediction_path: str,
    support_path: str,
    planes: list[int],
    *,
    prediction_threshold: float = 0.0,
    support_threshold: float = 0.0,
    anonymous: bool = True,
) -> dict:
    storage_options = (
        {"anon": anonymous} if prediction_path.startswith("s3://") else None
    )
    prediction_store = open_zarr(
        prediction_path,
        mode="r",
        storage_options=storage_options,
    )
    support_store = open_support_volume(support_path, anon=anonymous)

    prediction_shape = _spatial_shape(prediction_store)
    validate_support_volume(support_store, prediction_shape)
    if not planes:
        raise ValueError("At least one plane is required")
    invalid = [plane for plane in planes if not 0 <= plane < prediction_shape[0]]
    if invalid:
        raise ValueError(
            f"Plane indices outside [0, {prediction_shape[0]}): {invalid}"
        )

    plane_results = []
    started = time.perf_counter()
    for z_index in planes:
        plane_started = time.perf_counter()
        prediction = _read_plane(prediction_store, z_index)
        support = _read_plane(support_store, z_index)
        metrics = evaluate_plane(
            prediction,
            support,
            prediction_threshold=prediction_threshold,
            support_threshold=support_threshold,
        )
        metrics.update(
            {
                "z_index": int(z_index),
                "logical_plane_bytes": int(prediction.nbytes + support.nbytes),
                "intersecting_chunk_bytes_uncompressed_upper_bound": int(
                    _plane_chunk_bytes_upper_bound(prediction_store, z_index)
                    + _plane_chunk_bytes_upper_bound(support_store, z_index)
                ),
                "elapsed_seconds": float(time.perf_counter() - plane_started),
            }
        )
        plane_results.append(metrics)

    totals = {
        key: sum(int(result[key]) for result in plane_results)
        for key in (
            "spatial_voxels",
            "supported_voxels",
            "positive_voxels_before",
            "positive_voxels_after",
            "phantom_positives_before",
            "phantom_positives_after",
            "nonzero_voxels_removed",
            "logical_plane_bytes",
            "intersecting_chunk_bytes_uncompressed_upper_bound",
        )
    }
    totals["support_fraction"] = (
        totals["supported_voxels"] / totals["spatial_voxels"]
    )
    totals["phantom_fraction_before"] = (
        totals["phantom_positives_before"] / totals["positive_voxels_before"]
        if totals["positive_voxels_before"]
        else 0.0
    )
    totals["phantom_fraction_after"] = (
        totals["phantom_positives_after"] / totals["positive_voxels_after"]
        if totals["positive_voxels_after"]
        else 0.0
    )
    totals["supported_values_preserved"] = all(
        bool(result["supported_values_preserved"]) for result in plane_results
    )

    return {
        "schema_version": 2,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "prediction_path": prediction_path,
        "support_path": support_path,
        "prediction_shape": list(prediction_shape),
        "prediction_chunks": list(prediction_store.chunks),
        "support_chunks": list(support_store.chunks),
        "prediction_dtype": str(prediction_store.dtype),
        "support_dtype": str(support_store.dtype),
        "prediction_threshold": float(prediction_threshold),
        "support_threshold": float(support_threshold),
        "anonymous_access": bool(anonymous),
        "planes": plane_results,
        "aggregate": totals,
        "elapsed_seconds": float(time.perf_counter() - started),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "zarr": zarr.__version__,
            "platform": platform.platform(),
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit binary predictions outside an aligned CT/support volume."
    )
    parser.add_argument("prediction_path")
    parser.add_argument("support_path")
    parser.add_argument(
        "--planes",
        type=int,
        nargs="+",
        required=True,
        help="Z plane indices to stream and audit.",
    )
    parser.add_argument("--prediction-threshold", type=float, default=0.0)
    parser.add_argument("--support-threshold", type=float, default=0.0)
    parser.add_argument(
        "--authenticated",
        action="store_true",
        help="Use configured credentials instead of anonymous S3 access.",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    report = run_benchmark(
        args.prediction_path,
        args.support_path,
        args.planes,
        prediction_threshold=args.prediction_threshold,
        support_threshold=args.support_threshold,
        anonymous=not args.authenticated,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")
        print(f"Report written to {args.output_json}")
    print(rendered)
    return 0


if __name__ == "__main__":
    sys.exit(main())
