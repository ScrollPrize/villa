"""Benchmark the bounded per-sample load -> value-augmentation pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vesuvius.neural_tracing.fiber_trace_3d.analyze_training import distribution
from vesuvius.neural_tracing.fiber_trace_3d.loader import (
    FiberTrace3DLoader,
    FiberTrace3DRawSample,
    FiberTrace3DSample,
    FiberTrace3DSharedTensorSlots,
    load_config,
    prepare_fiber_trace_3d_cuda_suffix,
)
from vesuvius.neural_tracing.fiber_trace_3d.sample_pipeline import (
    FiberTrace3DAugmentFactory,
    FiberTrace3DFusedLoadFactory,
    FiberTrace3DLoadFactory,
)
from vesuvius.neural_tracing.ordered_sample_pipeline import (
    OrderedPipelineResult,
    OrderedTwoStageSamplePipeline,
)
import vesuvius.neural_tracing.ordered_sample_pipeline as ordered_pipeline_module
import vesuvius.neural_tracing.fiber_trace_3d.loader as loader_module


def _assert_samples_equal(actual: FiberTrace3DSample, expected: FiberTrace3DSample) -> None:
    scalar_fields = (
        "stream_index",
        "data_index",
        "record_index",
        "fiber_path",
        "control_point_index",
        "target_mode",
    )
    tensor_fields = (
        "volume",
        "valid_mask",
        "cp_local_zyx",
        "crop_origin_zyx",
        "target_segment_starts_zyx",
        "target_segment_ends_zyx",
        "target_segment_bbox_lo_zyx",
        "target_segment_bbox_hi_zyx",
        "target_tangent_zyx",
    )
    for name in scalar_fields:
        if getattr(actual, name) != getattr(expected, name):
            raise AssertionError(f"pipeline sample field differs: {name}")
    for name in tensor_fields:
        actual_value = getattr(actual, name)
        expected_value = getattr(expected, name)
        if actual_value is None or expected_value is None:
            if actual_value is not expected_value:
                raise AssertionError(f"pipeline sample tensor differs: {name}")
            continue
        if not torch.equal(actual_value, expected_value):
            raise AssertionError(f"pipeline sample tensor differs: {name}")


def _assert_nested_equal(actual: Any, expected: Any, path: str) -> None:
    if torch.is_tensor(actual) or torch.is_tensor(expected):
        if not (torch.is_tensor(actual) and torch.is_tensor(expected)):
            raise AssertionError(f"raw sample type differs: {path}")
        if not torch.equal(actual, expected):
            raise AssertionError(f"raw sample tensor differs: {path}")
        return
    if isinstance(actual, np.ndarray) or isinstance(expected, np.ndarray):
        if not (isinstance(actual, np.ndarray) and isinstance(expected, np.ndarray)):
            raise AssertionError(f"raw sample type differs: {path}")
        if actual.dtype != expected.dtype or actual.shape != expected.shape or not np.array_equal(
            actual, expected, equal_nan=True
        ):
            raise AssertionError(f"raw sample array differs: {path}")
        return
    if is_dataclass(actual) or is_dataclass(expected):
        if type(actual) is not type(expected):
            raise AssertionError(f"raw sample dataclass type differs: {path}")
        for field in fields(actual):
            if field.name in {"load_wall_ms", "load_cpu_ms", "profile_timings_ms"}:
                continue
            _assert_nested_equal(
                getattr(actual, field.name), getattr(expected, field.name), f"{path}.{field.name}"
            )
        return
    if isinstance(actual, (tuple, list)) or isinstance(expected, (tuple, list)):
        if type(actual) is not type(expected) or len(actual) != len(expected):
            raise AssertionError(f"raw sample sequence differs: {path}")
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            _assert_nested_equal(actual_item, expected_item, f"{path}[{index}]")
        return
    if actual != expected:
        raise AssertionError(f"raw sample value differs: {path}")


def _assert_raw_samples_equal(
    actual: FiberTrace3DRawSample, expected: FiberTrace3DRawSample
) -> None:
    _assert_nested_equal(actual, expected, "raw")


def _resolved_config(reference: Path) -> tuple[dict[str, Any], str]:
    payload = json.loads(reference.read_text())
    if not isinstance(payload, dict):
        raise ValueError("config/reference JSON root must be an object")
    if "datasets" in payload:
        return payload, "direct-config"
    nested = payload.get("config")
    if isinstance(nested, dict) and "datasets" in nested:
        return dict(nested), "saved-reference-config"
    raise ValueError("JSON must be a FiberTrace3D config or contain a config object")


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _set_parent_thread_limits() -> None:
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ[name] = "1"
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _stage_profile_rates(
    measured: list[dict[str, float]],
    *,
    measured_samples: int,
    production_seconds: float,
) -> dict[str, Any]:
    def total(key: str) -> float:
        return sum(float(item.get(key, 0.0)) for item in measured)

    def rate(numerator: float, seconds: float) -> float | None:
        return None if seconds <= 0.0 else numerator / seconds

    generated_points = total("sample_coordinate_points")
    valid_points = total("sample_valid_coordinate_points")
    geometry_seconds = total("sample_geometry_ms") / 1000.0
    sampling_seconds = total("sample_vc_sample_seconds")
    dependency_seconds = total("sample_vc_dependency_seconds")
    resolve_seconds = total("sample_vc_chunk_resolve_seconds")
    requested_chunks = total("sample_vc_requested_chunks")
    decode_seconds = total("sample_vc_decode_seconds")
    decoded_chunks = total("sample_vc_decoded_chunks")
    decode_input_bytes = total("sample_vc_decode_input_bytes")
    decoded_bytes = total("sample_vc_decoded_chunk_bytes")
    dependency_points = total("sample_vc_dependency_points")
    footprint_reuses = total("sample_vc_dependency_footprint_reuses")
    footprint_changes = total("sample_vc_dependency_footprint_changes")
    dependency_key_lookups = total("sample_vc_dependency_key_lookups")
    dependency_hash_probes = total("sample_vc_dependency_hash_probes")
    direct_gathers = total("sample_vc_sample_direct_gather_points")
    boundary_points = total("sample_vc_sample_boundary_points")
    pinned_map_lookups = total("sample_vc_sample_pinned_map_lookups")
    mib = 1024.0 * 1024.0
    return {
        "coordinate_generation": {
            "points": int(generated_points),
            "service_seconds_sum": geometry_seconds,
            "mega_points_per_service_second": rate(generated_points / 1.0e6, geometry_seconds),
            "equivalent_samples_per_service_second": rate(float(measured_samples), geometry_seconds),
            "effective_host_concurrency": rate(geometry_seconds, production_seconds),
        },
        "coordinate_sampling": {
            "valid_points": int(valid_points),
            "service_seconds_sum": sampling_seconds,
            "mega_points_per_service_second": rate(valid_points / 1.0e6, sampling_seconds),
            "equivalent_samples_per_service_second": rate(float(measured_samples), sampling_seconds),
            "effective_host_concurrency": rate(sampling_seconds, production_seconds),
        },
        "dependency_discovery": {
            "valid_points": int(valid_points),
            "service_seconds_sum": dependency_seconds,
            "mega_points_per_service_second": rate(valid_points / 1.0e6, dependency_seconds),
            "equivalent_samples_per_service_second": rate(float(measured_samples), dependency_seconds),
            "effective_host_concurrency": rate(dependency_seconds, production_seconds),
            "points_visited": int(dependency_points),
            "same_footprint_reuses": int(footprint_reuses),
            "footprint_changes": int(footprint_changes),
            "key_lookups": int(dependency_key_lookups),
            "hash_probes": int(dependency_hash_probes),
            "same_footprint_reuse_fraction": rate(footprint_reuses, dependency_points),
        },
        "sampling_locality": {
            "direct_gather_points": int(direct_gathers),
            "boundary_points": int(boundary_points),
            "pinned_map_lookups": int(pinned_map_lookups),
            "direct_gather_fraction": rate(direct_gathers, valid_points),
        },
        "chunk_resolution": {
            "requested_chunks": int(requested_chunks),
            "service_seconds_sum": resolve_seconds,
            "chunks_per_service_second": rate(requested_chunks, resolve_seconds),
            "equivalent_samples_per_service_second": rate(float(measured_samples), resolve_seconds),
            "effective_host_concurrency": rate(resolve_seconds, production_seconds),
        },
        "chunk_decoding": {
            "chunks": int(decoded_chunks),
            "input_bytes": int(decode_input_bytes),
            "output_bytes": int(decoded_bytes),
            "service_seconds_sum": decode_seconds,
            "chunks_per_service_second": rate(decoded_chunks, decode_seconds),
            "input_mib_per_service_second": rate(decode_input_bytes / mib, decode_seconds),
            "output_mib_per_service_second": rate(decoded_bytes / mib, decode_seconds),
            "equivalent_samples_per_service_second": rate(float(measured_samples), decode_seconds),
            "effective_host_concurrency": rate(decode_seconds, production_seconds),
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--warmup-samples", type=int, default=32)
    parser.add_argument("--verify-samples", type=int, default=2)
    parser.add_argument("--load-workers", type=int, default=16)
    parser.add_argument("--augment-workers", type=int, default=32)
    parser.add_argument(
        "--shared-workers",
        type=int,
        default=0,
        help="use one dependency-aware worker pool of this size for both stages",
    )
    parser.add_argument(
        "--initial-in-flight",
        type=int,
        default=0,
        help="initial shared-pool admission window; 0 uses shared worker count",
    )
    parser.add_argument(
        "--shared-load-affinity-workers",
        type=int,
        default=0,
        help="soft load affinity count in shared mode; 0 uses two thirds",
    )
    parser.add_argument(
        "--raw-only",
        action="store_true",
        help="stop after raw sample loading; launch no augmentation workers",
    )
    parser.add_argument(
        "--fused",
        action="store_true",
        help="benchmark complete load+augmentation in one free-worker pool",
    )
    parser.add_argument(
        "--shared-tensor-slots",
        action="store_true",
        help="transport volume/mask through bounded preallocated shared slots",
    )
    parser.add_argument("--raw-queue-capacity", type=int, default=64)
    parser.add_argument("--result-queue-capacity", type=int, default=64)
    parser.add_argument("--max-in-flight", type=int, default=128)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--sample-index-limit", type=int, default=0)
    parser.add_argument("--sample-mode", choices=("random", "sequential"), default="random")
    parser.add_argument(
        "--volume-io-threads",
        type=int,
        default=0,
        help="override VC3D source-read concurrency; 0 preserves the config",
    )
    parser.add_argument(
        "--volume-decode-threads",
        type=int,
        default=0,
        help="override VC3D chunk-decode concurrency; 0 preserves the config",
    )
    parser.add_argument(
        "--volume-sample-threads",
        type=int,
        default=0,
        help="override VC3D coordinate-interpolation workers; 0 preserves the config/default",
    )
    parser.add_argument(
        "--disable-chunk-locality-fast-path",
        action="store_true",
        help="run the instrumented scalar reference dependency/sampling path",
    )
    parser.add_argument(
        "--disable-locality-stats",
        action="store_true",
        help="measure production locality path without per-point diagnostic counters",
    )
    parser.add_argument(
        "--geometry-tile-depth",
        type=int,
        default=0,
        help="override cache-local coordinate-generation Z depth; 0 preserves the config",
    )
    parser.add_argument(
        "--geometry-backend",
        choices=("auto", "native", "eager"),
        default=None,
        help="override Fiber3D coordinate generation backend",
    )
    parser.add_argument(
        "--value-augmentation-backend",
        choices=("auto", "native", "torch"),
        default=None,
        help="override Fiber3D value augmentation backend",
    )
    parser.add_argument(
        "--defer-cuda-augmentation",
        choices=("blur", "all"),
        default=None,
        help="emit production CUDA descriptors and stop before H2D",
    )
    parser.add_argument("--multiprocessing-context", choices=("spawn", "forkserver"), default="forkserver")
    parser.add_argument("--startup-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--no-progress-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--cache-state", choices=("unknown", "cold", "warm"), default="unknown")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if bool(args.raw_only) and bool(args.fused):
        raise ValueError("--raw-only and --fused are mutually exclusive")
    if int(args.shared_workers) > 0 and (bool(args.raw_only) or bool(args.fused)):
        raise ValueError("--shared-workers requires the split two-stage pipeline")
    for name in ("samples", "raw_queue_capacity", "result_queue_capacity", "max_in_flight"):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be > 0")
    if int(args.shared_workers) == 0 and int(args.load_workers) <= 0:
        raise ValueError("--load-workers must be > 0 outside shared mode")
    if (
        int(args.shared_workers) == 0
        and not bool(args.raw_only)
        and not bool(args.fused)
        and int(args.augment_workers) <= 0
    ):
        raise ValueError("--augment-workers must be > 0 unless --raw-only is used")
    if any(
        int(value) < 0
        for value in (
            args.shared_workers,
            args.initial_in_flight,
            args.shared_load_affinity_workers,
        )
    ):
        raise ValueError("shared worker/admission/affinity values must be >= 0")
    if int(args.shared_workers) > 0 and int(args.initial_in_flight) > int(args.max_in_flight):
        raise ValueError("--initial-in-flight cannot exceed --max-in-flight")
    if int(args.shared_load_affinity_workers) > int(args.shared_workers):
        raise ValueError("--shared-load-affinity-workers cannot exceed --shared-workers")
    if int(args.warmup_samples) < 0 or int(args.verify_samples) < 0:
        raise ValueError("warmup/verify sample counts must be >= 0")
    if any(
        int(value) < 0
        for value in (
            args.volume_io_threads,
            args.volume_decode_threads,
            args.volume_sample_threads,
            args.geometry_tile_depth,
        )
    ):
        raise ValueError("volume I/O/decode/sample thread overrides must be >= 0")
    if int(args.verify_samples) > int(args.warmup_samples) + int(args.samples):
        raise ValueError("--verify-samples cannot exceed total warmup + measured samples")
    _set_parent_thread_limits()

    config_payload, reference_kind = _resolved_config(args.reference.resolve())
    if int(args.volume_io_threads) > 0:
        config_payload["volume_io_threads"] = int(args.volume_io_threads)
    if int(args.volume_decode_threads) > 0:
        config_payload["volume_decode_threads"] = int(args.volume_decode_threads)
    if int(args.volume_sample_threads) > 0:
        config_payload["volume_sample_threads"] = int(args.volume_sample_threads)
    config_payload["volume_chunk_locality_fast_path"] = not bool(
        args.disable_chunk_locality_fast_path
    )
    config_payload["volume_collect_locality_stats"] = not bool(
        args.disable_locality_stats
    )
    if int(args.geometry_tile_depth) > 0:
        config_payload["geometry_tile_depth"] = int(args.geometry_tile_depth)
    if args.geometry_backend is not None:
        config_payload["geometry_backend"] = str(args.geometry_backend)
    if args.value_augmentation_backend is not None:
        config_payload["value_augmentation_backend"] = str(
            args.value_augmentation_backend
        )
    batch_size = int(config_payload.get("batch_size", 1))
    if batch_size <= 0:
        raise ValueError("config batch_size must be > 0")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    run_dir = args.output_root.resolve() / f"sample_pipeline_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    config_path = run_dir / "config.json"
    config_path.write_text(json.dumps(config_payload, indent=2) + "\n")

    verify_expected: dict[int, FiberTrace3DSample | FiberTrace3DRawSample] = {}
    verify_started = time.perf_counter()
    if int(args.verify_samples) > 0:
        verifier = FiberTrace3DLoader(load_config(config_path))
        for offset in range(int(args.verify_samples)):
            index = int(args.start_index) + offset
            load_kwargs = {
                "sample_index_limit": (
                    None if int(args.sample_index_limit) <= 0 else int(args.sample_index_limit)
                ),
                "sample_mode": str(args.sample_mode),
                "device": "cpu",
                "profile": False,
            }
            if bool(args.raw_only):
                verify_expected[index] = verifier.load_raw_sample(index, **load_kwargs)
            elif args.defer_cuda_augmentation is not None:
                verify_expected[index] = prepare_fiber_trace_3d_cuda_suffix(
                    verifier.config,
                    verifier.load_raw_sample(index, **load_kwargs),
                    mode=str(args.defer_cuda_augmentation),
                )
            else:
                verify_expected[index] = verifier.load_sample(index, **load_kwargs)
    verify_seconds = time.perf_counter() - verify_started

    total_items = int(args.warmup_samples) + int(args.samples)
    sample_limit = None if int(args.sample_index_limit) <= 0 else int(args.sample_index_limit)
    transport = (
        FiberTrace3DSharedTensorSlots(
            start_index=int(args.start_index),
            slot_count=int(args.max_in_flight),
            patch_shape_zyx=tuple(int(v) for v in config_payload["patch_shape_zyx"]),
            volume_dtype=(
                torch.uint8
                if args.defer_cuda_augmentation == "all"
                else torch.float32
            ),
        )
        if bool(args.shared_tensor_slots)
        else None
    )
    startup_started = time.perf_counter()
    stage1_factory = (
        FiberTrace3DFusedLoadFactory(str(config_path), sample_limit, str(args.sample_mode))
        if bool(args.fused)
        else FiberTrace3DLoadFactory(str(config_path), sample_limit, str(args.sample_mode))
    )
    pipeline = OrderedTwoStageSamplePipeline(
        stage1_factory=stage1_factory,
        stage2_factory=(
            None
            if bool(args.raw_only) or bool(args.fused)
            else FiberTrace3DAugmentFactory(
                str(config_path),
                defer_cuda_suffix=args.defer_cuda_augmentation,
            )
        ),
        start_index=int(args.start_index),
        item_count=total_items,
        stage1_workers=0 if int(args.shared_workers) > 0 else int(args.load_workers),
        stage2_workers=(
            0
            if bool(args.raw_only) or bool(args.fused) or int(args.shared_workers) > 0
            else int(args.augment_workers)
        ),
        raw_queue_capacity=int(args.raw_queue_capacity),
        result_queue_capacity=int(args.result_queue_capacity),
        max_in_flight=int(args.max_in_flight),
        multiprocessing_context=str(args.multiprocessing_context),
        startup_timeout_seconds=float(args.startup_timeout_seconds),
        no_progress_timeout_seconds=float(args.no_progress_timeout_seconds),
        transport=transport,
        shared_workers=int(args.shared_workers),
        initial_in_flight=(
            None if int(args.initial_in_flight) == 0 else int(args.initial_in_flight)
        ),
        shared_stage1_affinity_workers=(
            None
            if int(args.shared_load_affinity_workers) == 0
            else int(args.shared_load_affinity_workers)
        ),
    )
    startup_seconds = time.perf_counter() - startup_started
    print(
        "sample_pipeline: "
        f"load_workers={0 if args.shared_workers else args.load_workers} "
        f"augment_workers={0 if args.raw_only or args.fused or args.shared_workers else args.augment_workers} "
        f"shared_workers={args.shared_workers} initial_in_flight="
        f"{pipeline.initial_in_flight} "
        f"load_affinity={pipeline.shared_stage1_affinity_workers} "
        f"raw_only={bool(args.raw_only)} "
        f"fused={bool(args.fused)} "
        f"shared_slots={bool(args.shared_tensor_slots)} "
        f"defer_cuda_augmentation={args.defer_cuda_augmentation or 'none'} "
        f"raw_queue={args.raw_queue_capacity} result_queue={args.result_queue_capacity} "
        f"in_flight={args.max_in_flight} batch={batch_size} "
        f"warmup={args.warmup_samples} measure={args.samples} native_threads=1",
        flush=True,
    )

    measured: list[dict[str, float]] = []
    all_completion_ns: list[int] = []
    batch_wait_ms: list[float] = []
    batch_stack_ms: list[float] = []
    measurement_started: float | None = None
    batch: list[OrderedPipelineResult] = []
    batch_started: float | None = None
    last_report = time.monotonic()
    measurement_finished: float | None = None
    try:
        for ordinal in range(total_items):
            if ordinal == int(args.warmup_samples):
                measurement_started = time.perf_counter()
            if ordinal >= int(args.warmup_samples) and not batch:
                batch_started = time.perf_counter()
            result = pipeline.next_ordered(release_credit=transport is None)
            all_completion_ns.append(int(result.stage2_finished_ns))
            expected = verify_expected.get(int(result.index))
            if expected is not None:
                if bool(args.raw_only) or args.defer_cuda_augmentation is not None:
                    _assert_raw_samples_equal(result.value, expected)
                else:
                    _assert_samples_equal(result.value, expected)
                del verify_expected[int(result.index)]
            if ordinal >= int(args.warmup_samples):
                timings = (
                    result.value.sample.profile_timings_ms
                    if bool(args.raw_only) or args.defer_cuda_augmentation is not None
                    else result.value.profile_timings_ms
                ) or {}
                measured.append(
                    {
                        "stage1_service_ms": result.stage1_service_ms,
                        "stage1_cpu_ms": result.stage1_cpu_ms,
                        "raw_queue_ms": result.raw_queue_ms,
                        "stage2_service_ms": result.stage2_service_ms,
                        "stage2_cpu_ms": result.stage2_cpu_ms,
                        "result_queue_ms": result.result_queue_ms,
                        "reorder_wait_ms": result.reorder_wait_ms,
                        "stage1_worker": float(result.stage1_worker),
                        "stage2_worker": float(result.stage2_worker),
                        "stage1_started_ns": float(result.stage1_started_ns),
                        "stage1_finished_ns": float(result.stage1_finished_ns),
                        "stage2_finished_ns": float(result.stage2_finished_ns),
                        "delivered_ns": float(result.delivered_ns),
                        **{
                            f"sample_{key}": float(value)
                            for key, value in timings.items()
                            if isinstance(value, (int, float))
                        },
                    }
                )
                if not bool(args.raw_only):
                    batch.append(result)
                if not bool(args.raw_only) and (
                    len(batch) == batch_size or ordinal + 1 == total_items
                ):
                    stack_started = time.perf_counter()
                    if transport is not None:
                        volumes = transport.batch_tensors(
                            [item.index for item in batch]
                        )
                    else:
                        volumes = torch.stack(
                            [
                                item.value.sample.volume
                                if args.defer_cuda_augmentation is not None
                                else item.value.volume
                                for item in batch
                            ],
                            dim=0,
                        )
                    del volumes
                    batch_stack_ms.append((time.perf_counter() - stack_started) * 1000.0)
                    assert batch_started is not None
                    batch_wait_ms.append((stack_started - batch_started) * 1000.0)
                    if transport is not None:
                        for item in batch:
                            pipeline.release(item.index)
                    batch.clear()
                    batch_started = None
                elif bool(args.raw_only) and transport is not None:
                    pipeline.release(result.index)
            elif transport is not None:
                pipeline.release(result.index)
            now = time.monotonic()
            if now - last_report >= 10.0 or ordinal + 1 == total_items:
                print(
                    f"sample_pipeline progress={ordinal + 1}/{total_items} "
                    f"reorder={len(pipeline.reorder)} highwater={pipeline.reorder_highwater}",
                    flush=True,
                )
                last_report = now
        measurement_finished = time.perf_counter()
    finally:
        pipeline.close()
    if verify_expected:
        raise RuntimeError(f"verification samples were not delivered: {sorted(verify_expected)}")
    if measurement_started is None or measurement_finished is None:
        raise RuntimeError("measurement interval was not recorded")
    measurement_seconds = measurement_finished - measurement_started
    pipeline_seconds = max(
        1.0e-9,
        (measurement_finished * 1.0e9 - float(pipeline.started_ns)) / 1.0e9,
    )
    production_ns = sorted(int(item["stage2_finished_ns"]) for item in measured)
    delivery_ns = sorted(int(item["delivered_ns"]) for item in measured)
    production_seconds = max(1.0e-9, (production_ns[-1] - production_ns[0]) / 1.0e9)
    delivery_seconds = max(1.0e-9, (delivery_ns[-1] - delivery_ns[0]) / 1.0e9)
    production_rate = max(0, len(production_ns) - 1) / production_seconds
    ordered_delivery_rate = max(0, len(measured) - 1) / delivery_seconds
    measured_end_to_end_seconds = max(
        1.0e-9,
        (max(int(item["delivered_ns"]) for item in measured)
         - min(int(item["stage1_started_ns"]) for item in measured))
        / 1.0e9,
    )
    measured_end_to_end_rate = len(measured) / measured_end_to_end_seconds

    voxel_count = math.prod(int(value) for value in config_payload["patch_shape_zyx"])
    raw_sample_bytes = voxel_count * (
        1 if args.defer_cuda_augmentation == "all" else 4
    )
    conservative_bytes_per_credit = raw_sample_bytes * 2
    result: dict[str, Any] = {
        "schema_version": 1,
        "command": [sys.executable, "-m", __spec__.name if __spec__ else __name__, *(argv or sys.argv[1:])],
        "reference": str(args.reference.resolve()),
        "reference_kind": reference_kind,
        "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        "run_dir": str(run_dir),
        "topology": (
            "one-host-global-cpu-shared-pipeline"
            if int(args.shared_workers) > 0
            else (
                "one-host-global-cpu-raw-loader"
                if bool(args.raw_only)
                else (
                    "one-host-global-cpu-fused-pipeline"
                    if bool(args.fused)
                    else "one-host-global-cpu-pipeline"
                )
            )
        ),
        "raw_only": bool(args.raw_only),
        "defer_cuda_augmentation": args.defer_cuda_augmentation,
        "fused": bool(args.fused),
        "shared_tensor_slots": bool(args.shared_tensor_slots),
        "shared_tensor_slot_count": int(args.max_in_flight) if transport is not None else 0,
        "shared_tensor_slot_bytes": (
            raw_sample_bytes * int(args.max_in_flight) if transport is not None else 0
        ),
        "cache_state_label": str(args.cache_state),
        "start_index": int(args.start_index),
        "warmup_samples": int(args.warmup_samples),
        "measured_samples": int(args.samples),
        "batch_size": batch_size,
        "load_workers": 0 if int(args.shared_workers) > 0 else int(args.load_workers),
        "augment_workers": (
            0
            if bool(args.raw_only) or bool(args.fused) or int(args.shared_workers) > 0
            else int(args.augment_workers)
        ),
        "shared_workers": int(args.shared_workers),
        "initial_in_flight": int(pipeline.initial_in_flight),
        "shared_stage1_affinity_workers": int(
            pipeline.shared_stage1_affinity_workers
        ),
        "shared_stage1_target_min": int(pipeline.shared_stage1_target_min),
        "shared_stage1_target_max": int(pipeline.shared_stage1_target_max),
        "shared_stage1_target_final": int(pipeline.shared_stage1_target_final),
        "worker_processes": int(args.shared_workers) or (
            int(args.load_workers)
            + (0 if bool(args.raw_only) or bool(args.fused) else int(args.augment_workers))
        ),
        "raw_queue_capacity": int(args.raw_queue_capacity),
        "result_queue_capacity": int(args.result_queue_capacity),
        "max_in_flight": int(args.max_in_flight),
        "native_threads_per_worker": 1,
        "volume_io_threads_per_load_worker": config_payload.get("volume_io_threads"),
        "volume_decode_threads_per_load_worker": config_payload.get(
            "volume_decode_threads"
        ),
        "volume_sample_threads_per_load_worker": config_payload.get(
            "volume_sample_threads"
        ),
        "volume_chunk_locality_fast_path": config_payload.get(
            "volume_chunk_locality_fast_path", True
        ),
        "raw_sample_bytes_estimate": raw_sample_bytes,
        "conservative_peak_tensor_bytes_estimate": (
            conservative_bytes_per_credit * int(args.max_in_flight)
        ),
        "startup_seconds": startup_seconds,
        "synchronous_verification_samples": int(args.verify_samples),
        "synchronous_verification_seconds": verify_seconds,
        "measurement_delivery_window_seconds": measurement_seconds,
        "pipeline_cold_seconds": pipeline_seconds,
        "pipeline_cold_samples_per_second": total_items / pipeline_seconds,
        "production_span_seconds": production_seconds,
        "samples_per_second": measured_end_to_end_rate
        if bool(args.raw_only)
        else production_rate,
        "completion_span_samples_per_second": production_rate,
        "ordered_delivery_samples_per_second": ordered_delivery_rate,
        "measured_end_to_end_seconds": measured_end_to_end_seconds,
        "measured_end_to_end_samples_per_second": measured_end_to_end_rate,
        "batches_per_second": production_rate / batch_size,
        "stage1_service_ms": distribution(item["stage1_service_ms"] for item in measured),
        "stage1_cpu_ms": distribution(item["stage1_cpu_ms"] for item in measured),
        "raw_queue_ms": None
        if bool(args.raw_only) or bool(args.fused)
        else distribution(item["raw_queue_ms"] for item in measured),
        "output_queue_ms": distribution(item["raw_queue_ms"] for item in measured)
        if bool(args.raw_only) or bool(args.fused)
        else None,
        "stage2_service_ms": None
        if bool(args.raw_only) or bool(args.fused)
        else distribution(item["stage2_service_ms"] for item in measured),
        "stage2_cpu_ms": None
        if bool(args.raw_only) or bool(args.fused)
        else distribution(item["stage2_cpu_ms"] for item in measured),
        "result_queue_ms": None
        if bool(args.raw_only) or bool(args.fused)
        else distribution(item["result_queue_ms"] for item in measured),
        "reorder_wait_ms": distribution(item["reorder_wait_ms"] for item in measured),
        "batch_delivery_wait_ms": distribution(batch_wait_ms),
        "batch_stack_ms": distribution(batch_stack_ms),
        "reorder_highwater": int(pipeline.reorder_highwater),
        "admission_highwater": int(pipeline.admission_highwater),
        "shared_stage1_tasks_by_worker": dict(pipeline.shared_stage1_counts),
        "shared_stage2_tasks_by_worker": dict(pipeline.shared_stage2_counts),
        "sample_stage_ms": {
            key.removeprefix("sample_"): distribution(item[key] for item in measured if key in item)
            for key in sorted({key for item in measured for key in item if key.startswith("sample_")})
        },
        "stage1_workers_used": len({item["stage1_worker"] for item in measured}),
        "stage2_workers_used": 0
        if bool(args.raw_only) or bool(args.fused)
        else len({item["stage2_worker"] for item in measured}),
        "summed_worker_cpu_cores": (
            sum(item["stage1_cpu_ms"] + item["stage2_cpu_ms"] for item in measured)
            / max(production_seconds * 1000.0, 1.0e-9)
        ),
        "stage_profile_rates": _stage_profile_rates(
            measured,
            measured_samples=len(measured),
            production_seconds=production_seconds,
        ),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
        },
        "source_sha256": {
            "pipeline_benchmark.py": _sha256_path(Path(__file__)),
            "ordered_sample_pipeline.py": _sha256_path(Path(ordered_pipeline_module.__file__)),
            "loader.py": _sha256_path(Path(loader_module.__file__)),
        },
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"sample_pipeline result samples_per_second={result['samples_per_second']:.2f} "
        f"end_to_end_samples_per_second={result['measured_end_to_end_samples_per_second']:.2f} "
        f"production_seconds={production_seconds:.2f} cpu_cores={result['summed_worker_cpu_cores']:.1f} "
        f"reorder_highwater={pipeline.reorder_highwater} result={run_dir / 'result.json'}",
        flush=True,
    )
    rates = result["stage_profile_rates"]
    generation = rates["coordinate_generation"]
    sampling = rates["coordinate_sampling"]
    decoding = rates["chunk_decoding"]
    resolution = rates["chunk_resolution"]
    format_rate = lambda value: "N/A" if value is None else f"{float(value):.3f}"
    print(
        "sample_pipeline stages "
        f"coord_gen={format_rate(generation['mega_points_per_service_second'])}MP/s "
        f"({format_rate(generation['equivalent_samples_per_service_second'])} samples/s) "
        f"sampling={format_rate(sampling['mega_points_per_service_second'])}MP/s "
        f"({format_rate(sampling['equivalent_samples_per_service_second'])} samples/s) "
        f"decode={format_rate(decoding['chunks_per_service_second'])}chunks/s "
        f"input={format_rate(decoding['input_mib_per_service_second'])}MiB/s "
        f"output={format_rate(decoding['output_mib_per_service_second'])}MiB/s "
        f"({format_rate(decoding['equivalent_samples_per_service_second'])} samples/s) "
        f"resolve={format_rate(resolution['chunks_per_service_second'])}chunks/s "
        f"({format_rate(resolution['equivalent_samples_per_service_second'])} samples/s)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
