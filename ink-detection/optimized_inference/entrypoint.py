import os
import io
import sys

# Allow huge images before anything imports cv2 (and before importing inference_timesformer)
os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", "0")

import json
import time
import math
import shutil
import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import uuid
import boto3
import numpy as np
from botocore.config import Config
import cv2
import concurrent.futures
import zarr
import tifffile as tiff
from huggingface_hub import snapshot_download
from contextlib import suppress

from profiling import (
    FLAG_APPROXIMATE,
    FLAG_ESTIMATED,
    TransferTracker,
    WorkflowProfiler,
    aggregate_workflow_profiling,
    build_runtime_parameters,
    collect_env_metadata,
    dir_size_bytes,
    scoped_timer,
)

# WebKnossos imports
try:
    from webknossos.dataset import Dataset
    from webknossos.dataset.layer import Layer
    from webknossos.geometry.mag import Mag
except:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
)
logger = logging.getLogger("optimized_inference.entrypoint")


def get_work_dir() -> str:
    """Return the writable work directory used for local staging."""
    return os.getenv("WORK_DIR", "/workspace").strip() or "/workspace"


def get_runtime_dir() -> str:
    """Return the writable runtime directory for temporary local artifacts."""
    return os.getenv("RUNTIME_DIR", "/tmp").strip() or "/tmp"


def parse_roi_xyxy_env(raw_value: str) -> Optional[Tuple[int, int, int, int]]:
    raw_value = (raw_value or "").strip()
    if not raw_value:
        return None
    parts = [part.strip() for part in raw_value.split(",")]
    if len(parts) != 4:
        raise ValueError(f"ROI_XYXY must have four comma-separated integers, got '{raw_value}'")
    x0, y0, x1, y1 = [int(part) for part in parts]
    if x0 >= x1 or y0 >= y1:
        raise ValueError(f"ROI_XYXY must satisfy x0 < x1 and y0 < y1, got '{raw_value}'")
    return (x0, y0, x1, y1)


@dataclass
class Inputs:
    model_key: str
    s3_path: str
    start_layer: int
    end_layer: int
    model_in_chans: int
    force_reverse: bool = False
    wk_inference: bool = False
    wk_dataset_id: str = ""
    local_layers_dir: str = ""
    step: str = "inference"  # "prepare", "inference", "reduce", or "aggregate-profiling"
    num_parts: int = 1
    part_id: int = 0
    zarr_output_dir: str = "/tmp/partitions"
    surface_volume_zarr: str = ""  # Path to pre-created surface volume zarr
    chunk_size: int = 1024  # Chunk size for zarr array creation (SURFACE_VOLUME_CHUNK_SIZE)
    use_zarr_compression: bool = False  # Enable/disable zarr compression
    prepare_max_workers: int = 4
    model_type: str = "timesformer"  # "timesformer", "resnet3d", or "resnet3d-<depth>"
    tile_size: int = 64  # Tile size for sliding window inference (size will be set to same value)
    stride: int = 16  # Stride for sliding window
    batch_size: int = 256  # Batch size for inference
    inference_workers: int = min(8, os.cpu_count() or 4)  # DataLoader worker count
    prefetch_factor: int = 8  # Prefetch factor for DataLoader
    accumulator_mode: str = "auto"  # "auto", "ram", or "memmap"
    accumulator_auto_threshold_gib: float = 32.0
    accumulator_flush_every_batches: int = 64
    progress_log_every_batches: int = 0
    progress_log_every_seconds: float = 0.0
    max_batches: int = 0
    skip_partition_write: bool = False
    skip_empty_tiles: bool = True
    prune_empty_tiles: bool = False
    direct_single_part_write: bool = False
    z_downsample_mean_factor: int = 1  # Optional on-the-fly z mean-downsample factor (1 or 2)
    roi_xyxy: Optional[Tuple[int, int, int, int]] = None
    blend_mode: str = "hann"  # "hann" or "gaussian"
    gaussian_sigma: float = 0.0  # 0 -> auto (~ tile/2.5) when blend_mode=gaussian
    output_path: str = ""  # Full output path (S3 URI or local path) for prediction result
    pixel_resolution_um: Optional[float] = None  # Real-world pixel resolution in micrometers (µm), None to omit
    add_scale_bar: bool = False  # Whether to add scale bar overlay to output
    scale_bar_length_um: float = 10000.0  # Scale bar length in micrometers (default 1cm)
    segment_id: str = ""
    profiling_level: str = "basic"
    profiling_sample_interval_ms: int = 1000
    profiling_detailed_partitions: str = "first"
    profiling_keep_raw_traces: bool = False
    profiling_output_prefix: str = ""
    profiling_raw_root: str = ""
    profiling_local_root: str = "/tmp/profiling"

def parse_env() -> Inputs:
    try:
        model_key = os.environ["MODEL"].strip()
        s3_path = os.getenv("S3_PATH", "").strip()
        start_layer = int(os.environ["START_LAYER"].strip())
        end_layer = int(os.environ["END_LAYER"].strip())
        model_in_chans_raw = os.getenv("MODEL_IN_CHANS", "").strip()
        force_reverse = os.getenv("FORCE_REVERSE", "false").lower() == "true"
        wk_dataset_id = os.getenv("WK_DATASET_ID", "").strip()
        local_layers_dir = os.getenv("LOCAL_LAYERS_DIR", "").strip()

        # Map/reduce parameters
        step = os.getenv("STEP", "inference").strip().lower()
        num_parts = int(os.getenv("NUM_PARTS", "1"))
        part_id = int(os.getenv("PART_ID", "0"))
        zarr_output_dir = os.getenv("ZARR_OUTPUT_DIR", "/tmp/partitions").strip()
        surface_volume_zarr = os.getenv("SURFACE_VOLUME_ZARR", "").strip()
        chunk_size = int(os.getenv("SURFACE_VOLUME_CHUNK_SIZE", "1024"))
        use_zarr_compression = os.getenv("USE_ZARR_COMPRESSION", "false").lower() == "true"
        prepare_max_workers = int(os.getenv("PREPARE_MAX_WORKERS", "4"))

        # Model type parameter
        model_type = os.getenv("MODEL_TYPE", "timesformer").strip().lower()

        # Validate model_type
        valid_model_types = {
            "timesformer",
            "resnet3d",
            "resnet3d-50",
            "resnet3d-101",
            "resnet3d-152",
            "resnet3d-200",
        }
        if model_type not in valid_model_types:
            raise ValueError(
                "MODEL_TYPE must be one of "
                f"{sorted(valid_model_types)}, got '{model_type}'"
            )

        # Inference configuration parameters
        tile_size = int(os.getenv("TILE_SIZE", "64"))
        stride = int(os.getenv("STRIDE", "16"))
        batch_size = int(os.getenv("BATCH_SIZE", "256"))
        inference_workers = int(os.getenv("INFERENCE_WORKERS", str(min(8, os.cpu_count() or 4))))
        prefetch_factor = int(os.getenv("PREFETCH_FACTOR", "8"))
        accumulator_mode = os.getenv("ACCUMULATOR_MODE", "ram").strip().lower()
        accumulator_auto_threshold_gib = float(os.getenv("ACCUMULATOR_AUTO_THRESHOLD_GIB", "32.0"))
        accumulator_flush_every_batches = int(os.getenv("ACCUMULATOR_FLUSH_EVERY_BATCHES", "64"))
        progress_log_every_batches = int(os.getenv("PROGRESS_LOG_EVERY_BATCHES", "0"))
        progress_log_every_seconds = float(os.getenv("PROGRESS_LOG_EVERY_SECONDS", "0.0"))
        max_batches = int(os.getenv("MAX_BATCHES", "0"))
        skip_partition_write = os.getenv("SKIP_PARTITION_WRITE", "false").lower() == "true"
        skip_empty_tiles = os.getenv("SKIP_EMPTY_TILES", "true").lower() == "true"
        prune_empty_tiles = os.getenv("PRUNE_EMPTY_TILES", "false").lower() == "true"
        direct_single_part_write = os.getenv("DIRECT_SINGLE_PART_WRITE", "false").lower() == "true"
        z_downsample_mean_factor = int(os.getenv("Z_DOWNSAMPLE_MEAN_FACTOR", "1"))
        roi_xyxy = parse_roi_xyxy_env(os.getenv("ROI_XYXY", ""))
        blend_mode = os.getenv("BLEND_MODE", "hann").strip().lower()
        gaussian_sigma = float(os.getenv("GAUSSIAN_SIGMA", "0.0"))
        output_path = os.getenv("OUTPUT_PATH", "").strip()
        segment_id = os.getenv("SEGMENT_ID", "").strip()
        profiling_level = os.getenv("PROFILING_LEVEL", "basic").strip().lower()
        profiling_sample_interval_ms = int(os.getenv("PROFILING_SAMPLE_INTERVAL_MS", "1000"))
        profiling_detailed_partitions = os.getenv("PROFILING_DETAILED_PARTITIONS", "first").strip()
        profiling_keep_raw_traces = os.getenv("PROFILING_KEEP_RAW_TRACES", "false").lower() == "true"
        profiling_output_prefix = os.getenv("PROFILING_OUTPUT_PREFIX", "").strip()
        profiling_raw_root = os.getenv("PROFILING_RAW_ROOT", "").strip()
        profiling_local_root = os.getenv("PROFILING_LOCAL_ROOT", "/tmp/profiling").strip()

        # Optional pixel resolution - only parse if provided
        pixel_resolution_str = os.getenv("PIXEL_RESOLUTION_UM", "").strip()
        pixel_resolution_um = float(pixel_resolution_str) if pixel_resolution_str else None

        # Scale bar configuration
        add_scale_bar = os.getenv("ADD_SCALE_BAR", "false").lower() == "true"
        scale_bar_length_um = float(os.getenv("SCALE_BAR_LENGTH_UM", "10000.0"))

        # Validate inference parameters
        if tile_size <= 0:
            raise ValueError(f"TILE_SIZE must be positive, got {tile_size}")
        if stride <= 0:
            raise ValueError(f"STRIDE must be positive, got {stride}")
        if batch_size <= 0:
            raise ValueError(f"BATCH_SIZE must be positive, got {batch_size}")
        if inference_workers < 0:
            raise ValueError(f"INFERENCE_WORKERS must be >= 0, got {inference_workers}")
        if prefetch_factor <= 0:
            raise ValueError(f"PREFETCH_FACTOR must be positive, got {prefetch_factor}")
        if accumulator_mode not in ("auto", "ram", "memmap"):
            raise ValueError(
                f"ACCUMULATOR_MODE must be 'auto', 'ram', or 'memmap', got '{accumulator_mode}'"
            )
        if accumulator_auto_threshold_gib <= 0:
            raise ValueError(
                f"ACCUMULATOR_AUTO_THRESHOLD_GIB must be positive, got {accumulator_auto_threshold_gib}"
            )
        if accumulator_flush_every_batches < 0:
            raise ValueError(
                "ACCUMULATOR_FLUSH_EVERY_BATCHES must be >= 0, "
                f"got {accumulator_flush_every_batches}"
            )
        if progress_log_every_batches < 0:
            raise ValueError(
                f"PROGRESS_LOG_EVERY_BATCHES must be >= 0, got {progress_log_every_batches}"
            )
        if progress_log_every_seconds < 0:
            raise ValueError(
                f"PROGRESS_LOG_EVERY_SECONDS must be >= 0, got {progress_log_every_seconds}"
            )
        if max_batches < 0:
            raise ValueError(f"MAX_BATCHES must be >= 0, got {max_batches}")
        if prepare_max_workers <= 0:
            raise ValueError(f"PREPARE_MAX_WORKERS must be positive, got {prepare_max_workers}")
        if z_downsample_mean_factor not in (1, 2):
            raise ValueError(
                f"Z_DOWNSAMPLE_MEAN_FACTOR must be 1 or 2, got {z_downsample_mean_factor}"
            )
        if blend_mode not in ("hann", "gaussian"):
            raise ValueError(f"BLEND_MODE must be 'hann' or 'gaussian', got '{blend_mode}'")
        if gaussian_sigma < 0:
            raise ValueError(f"GAUSSIAN_SIGMA must be >= 0, got {gaussian_sigma}")
        if stride > tile_size:
            logger.warning(f"STRIDE ({stride}) > TILE_SIZE ({tile_size}) may create gaps in coverage")
        if pixel_resolution_um is not None and pixel_resolution_um <= 0:
            raise ValueError(f"PIXEL_RESOLUTION_UM must be positive, got {pixel_resolution_um}")
        if scale_bar_length_um <= 0:
            raise ValueError(f"SCALE_BAR_LENGTH_UM must be positive, got {scale_bar_length_um}")

        # Validate step parameter
        if step not in ("prepare", "inference", "reduce", "aggregate-profiling"):
            raise ValueError(f"STEP must be 'prepare', 'inference', 'reduce', or 'aggregate-profiling', got '{step}'")

        # Validate NUM_PARTS upfront
        if num_parts < 1:
            raise ValueError(f"NUM_PARTS must be >= 1, got {num_parts}")

        # Validate PART_ID for inference step
        if step == "inference" and (part_id < 0 or part_id >= num_parts):
            raise ValueError(f"PART_ID must be in range [0, {num_parts}), got {part_id}")

        # Validate required parameters per step
        if step == "prepare":
            # Prepare step requires s3_path, wk_dataset_id, or local_layers_dir
            if not s3_path and not wk_dataset_id and not local_layers_dir:
                raise ValueError("STEP=prepare requires S3_PATH, WK_DATASET_ID, or LOCAL_LAYERS_DIR")
        elif step == "inference":
            # Inference step requires surface_volume_zarr
            if not surface_volume_zarr:
                raise ValueError("STEP=inference requires SURFACE_VOLUME_ZARR (run STEP=prepare first)")
        elif step == "aggregate-profiling" and not profiling_raw_root:
            raise ValueError("STEP=aggregate-profiling requires PROFILING_RAW_ROOT")
        # reduce step doesn't require these

        wk_inference = bool(wk_dataset_id)

        if start_layer >= end_layer:
            raise ValueError("START_LAYER must be < END_LAYER")

        raw_requested_channels = int(end_layer - start_layer)
        downsampled_requested_channels = int(
            (raw_requested_channels + z_downsample_mean_factor - 1) // z_downsample_mean_factor
        )
        model_in_chans = int(model_in_chans_raw) if model_in_chans_raw else downsampled_requested_channels
        if model_in_chans < 1 or model_in_chans > downsampled_requested_channels:
            raise ValueError(
                "MODEL_IN_CHANS must be in "
                f"[1, {downsampled_requested_channels}] after z downsample x{z_downsample_mean_factor}, "
                f"got {model_in_chans}"
            )

        return Inputs(
            model_key=model_key,
            s3_path=s3_path,
            start_layer=start_layer,
            end_layer=end_layer,
            model_in_chans=model_in_chans,
            force_reverse=force_reverse,
            wk_inference=wk_inference,
            wk_dataset_id=wk_dataset_id,
            local_layers_dir=local_layers_dir,
            step=step,
            num_parts=num_parts,
            part_id=part_id,
            zarr_output_dir=zarr_output_dir,
            surface_volume_zarr=surface_volume_zarr,
            chunk_size=chunk_size,
            use_zarr_compression=use_zarr_compression,
            prepare_max_workers=prepare_max_workers,
            model_type=model_type,
            tile_size=tile_size,
            stride=stride,
            batch_size=batch_size,
            inference_workers=inference_workers,
            prefetch_factor=prefetch_factor,
            accumulator_mode=accumulator_mode,
            accumulator_auto_threshold_gib=accumulator_auto_threshold_gib,
            accumulator_flush_every_batches=accumulator_flush_every_batches,
            progress_log_every_batches=progress_log_every_batches,
            progress_log_every_seconds=progress_log_every_seconds,
            max_batches=max_batches,
            skip_partition_write=skip_partition_write,
            skip_empty_tiles=skip_empty_tiles,
            prune_empty_tiles=prune_empty_tiles,
            direct_single_part_write=direct_single_part_write,
            z_downsample_mean_factor=z_downsample_mean_factor,
            roi_xyxy=roi_xyxy,
            blend_mode=blend_mode,
            gaussian_sigma=gaussian_sigma,
            output_path=output_path,
            pixel_resolution_um=pixel_resolution_um,
            add_scale_bar=add_scale_bar,
            scale_bar_length_um=scale_bar_length_um,
            segment_id=segment_id,
            profiling_level=profiling_level,
            profiling_sample_interval_ms=profiling_sample_interval_ms,
            profiling_detailed_partitions=profiling_detailed_partitions,
            profiling_keep_raw_traces=profiling_keep_raw_traces,
            profiling_output_prefix=profiling_output_prefix,
            profiling_raw_root=profiling_raw_root,
            profiling_local_root=profiling_local_root,
        )
    except KeyError as e:
        raise RuntimeError(f"Missing required env var: {e.args[0]}") from e


def parse_s3_uri(s3_uri: str) -> Tuple[str, str]:
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI: {s3_uri}")
    path = s3_uri[len("s3://") :]
    parts = path.split("/", 1)
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], parts[1]


def is_weight_file(path_or_uri: str) -> bool:
    lower = path_or_uri.strip().lower()
    return lower.endswith((".ckpt", ".safetensors", ".bin", ".pt"))


def ensure_clean_dir(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def sanitize_model_tag(model_key: str) -> str:
    """Collapse model identifiers or checkpoint paths into a safe filename fragment."""
    base = os.path.basename(model_key.rstrip("/")) or model_key
    stem, _ = os.path.splitext(base)
    candidate = stem or base or "model"
    candidate = re.sub(r"[^A-Za-z0-9._-]+", "_", candidate).strip("._-")
    return candidate or "model"


def resolve_final_output_path(inputs: Inputs) -> str:
    model_tag = sanitize_model_tag(inputs.model_key)
    if inputs.output_path:
        return inputs.output_path
    if inputs.s3_path:
        bucket, prefix = parse_s3_uri(inputs.s3_path)
        out_key = os.path.join(
            prefix.rstrip("/"),
            "predictions",
            f"prediction_{model_tag}_{inputs.start_layer:02d}_{inputs.end_layer:02d}.tif",
        )
        return f"s3://{bucket}/{out_key}"
    if inputs.wk_inference:
        return os.path.join(
            get_runtime_dir(),
            f"prediction_{model_tag}_{inputs.start_layer:02d}_{inputs.end_layer:02d}.tif",
        )
    raise ValueError("Result output requires OUTPUT_PATH, S3_PATH, or WK_DATASET_ID")


def persist_local_tiff_result(
    *,
    inputs: Inputs,
    local_tiff_path: str,
    profiler: Optional[WorkflowProfiler] = None,
) -> str:
    final_output_path = resolve_final_output_path(inputs)
    s3_client = boto3.client("s3")

    if final_output_path.startswith("s3://"):
        output_bucket, output_key = parse_s3_uri(final_output_path)
        logger.info(f"Uploading tiled TIFF to S3: {final_output_path}")
        profiled_s3_upload_file(s3_client, local_tiff_path, output_bucket, output_key, profiler)
        result_uri = final_output_path
    else:
        os.makedirs(os.path.dirname(final_output_path), exist_ok=True)
        logger.info(f"Copying tiled TIFF to local path: {final_output_path}")
        with scoped_timer(profiler, "local_write_seconds", flag=FLAG_APPROXIMATE):
            shutil.copy2(local_tiff_path, final_output_path)
        if profiler is not None:
            profiler.increment_counter("local_write_bytes", os.path.getsize(local_tiff_path))
        result_uri = final_output_path

    logger.info(f"Saved result to: {result_uri}")
    with open(os.path.join(get_runtime_dir(), "result_s3_url.txt"), "w", encoding="utf-8") as f:
        f.write(result_uri)
    return result_uri


def build_profiler(inputs: Inputs, template_name: str) -> WorkflowProfiler:
    metadata = collect_env_metadata(
        {
            "segment_id": inputs.segment_id,
            "model": inputs.model_key,
            "model_type": inputs.model_type,
        }
    )
    runtime_parameters = build_runtime_parameters(
        s3_path=inputs.s3_path,
        start_layer=inputs.start_layer,
        end_layer=inputs.end_layer,
        model_in_chans=inputs.model_in_chans,
        local_layers_dir=inputs.local_layers_dir,
        num_parts=inputs.num_parts,
        part_id=inputs.part_id,
        tile_size=inputs.tile_size,
        stride=inputs.stride,
        batch_size=inputs.batch_size,
        z_downsample_mean_factor=inputs.z_downsample_mean_factor,
        roi_xyxy=inputs.roi_xyxy,
        progress_log_every_batches=inputs.progress_log_every_batches,
        progress_log_every_seconds=inputs.progress_log_every_seconds,
        surface_volume_zarr=inputs.surface_volume_zarr,
        zarr_output_dir=inputs.zarr_output_dir,
        prepare_max_workers=inputs.prepare_max_workers,
        profiling_level=inputs.profiling_level,
        profiling_keep_raw_traces=inputs.profiling_keep_raw_traces,
    )
    profiler = WorkflowProfiler(
        level=inputs.profiling_level,
        sample_interval_ms=inputs.profiling_sample_interval_ms,
        raw_root=inputs.profiling_raw_root or None,
        local_root=inputs.profiling_local_root,
        step_name=inputs.step,
        template_name=template_name,
        part_id=inputs.part_id if inputs.step == "inference" else None,
        metadata=metadata,
        runtime_parameters=runtime_parameters,
        detailed_selector=inputs.profiling_detailed_partitions,
    )
    return profiler


def profiled_s3_download_file(
    s3_client,
    bucket: str,
    key: str,
    output_path: str,
    profiler: Optional[WorkflowProfiler],
    metric_name: str = "download_seconds",
    bytes_metric: str = "s3_download_bytes",
) -> None:
    tracker = TransferTracker()
    start = time.monotonic()
    s3_client.download_file(bucket, key, output_path, Callback=tracker)
    elapsed = time.monotonic() - start
    if profiler is not None:
        profiler.add_duration(metric_name, elapsed)
        profiler.increment_counter(bytes_metric, tracker.bytes_transferred)


def profiled_s3_upload_file(
    s3_client,
    local_path: str,
    bucket: str,
    key: str,
    profiler: Optional[WorkflowProfiler],
    metric_name: str = "upload_seconds",
    bytes_metric: str = "s3_upload_bytes",
) -> None:
    tracker = TransferTracker()
    start = time.monotonic()
    s3_client.upload_file(local_path, bucket, key, Callback=tracker)
    elapsed = time.monotonic() - start
    if profiler is not None:
        profiler.add_duration(metric_name, elapsed)
        profiler.increment_counter(bytes_metric, tracker.bytes_transferred)


def write_aggregate_output_parameters(results: Dict[str, str]) -> None:
    parameter_dir = os.path.join(get_runtime_dir(), "outputs", "parameters")
    os.makedirs(parameter_dir, exist_ok=True)
    for name, path in results.items():
        target = os.path.join(parameter_dir, f"{name}.txt")
        with open(target, "w", encoding="utf-8") as handle:
            handle.write(path)


def list_layers_objects(
    s3_client, bucket: str, prefix: str, start_layer: int, end_layer: int, profiler: Optional[WorkflowProfiler] = None
) -> List[Tuple[str, str]]:
    # Return list of (key, basename) for .tif/.tiff/.png/.jpeg/.jpg files inside any "layers/" folder under prefix
    # OR directly under prefix for surface-volumes (e.g., paths ending in .tifs/)
    paginator = s3_client.get_paginator("list_objects_v2")
    keys: List[Tuple[str, str]] = []
    SUPPORTED_IMAGE_FORMATS = {'.tif', '.tiff', '.png', '.jpeg', '.jpg'}
    with scoped_timer(profiler, "s3_list_seconds"):
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]

                # Check if it's a layer file with supported format
                # Support both: paths with "/layers/" subdirectory AND surface-volumes with .tifs/ directory
                if "/layers/" not in key.lower() and not prefix.endswith(".tifs/"):
                    continue

                base = os.path.basename(key)
                name, ext = os.path.splitext(base)

                # Check if the file extension is supported
                if ext.lower() not in SUPPORTED_IMAGE_FORMATS:
                    continue

                try:
                    # Tolerate leading zeros, e.g., 01, 02, ...
                    layer_idx = int(name)
                except ValueError:
                    continue

                # Check if layer index is within range (exclusive end) -> [start_layer, end_layer)
                if start_layer <= layer_idx < end_layer:
                    keys.append((key, base))
    if not keys:
        raise RuntimeError(
            f"No layers found within range [{start_layer}, {end_layer}) under s3://{bucket}/{prefix}"
        )
    # Sort by numeric layer index
    keys.sort(key=lambda kv: int(os.path.splitext(kv[1])[0]))
    return keys


def download_layers(
    s3_client, bucket: str, objects: List[Tuple[str, str]], out_dir: str, profiler: Optional[WorkflowProfiler] = None
) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)

    session = boto3.Session()
    config = Config(
        max_pool_connections=20,
        retries={'max_attempts': 3, 'mode': 'adaptive'}
    )

    def _download_one(args):
        idx, key, base, bucket, out_dir = args
        out_path = os.path.join(out_dir, base)
        client = session.client("s3", config=config)
        profiled_s3_download_file(client, bucket, key, out_path, profiler)
        logger.info(f"Finished downloading layer {idx}: {out_path}")
        return out_path

    paths: List[str] = []
    # Prepare arguments for each download
    download_args = [(idx, key, base, bucket, out_dir) for idx, (key, base) in enumerate(objects)]

    # Use ThreadPoolExecutor for parallel downloads
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(objects))) as executor:
        # Map returns results in order of the input
        results = list(executor.map(_download_one, download_args))
        paths.extend(results)

    return paths


def load_layers_to_numpy(layer_paths: List[str]) -> np.ndarray:
    if not layer_paths:
        raise ValueError("No layer paths provided")

    # Load all images first to ensure consistent processing
    images = []
    for i, path in enumerate(layer_paths):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Failed to read image: {path}")
        images.append(img)

    # Ensure all images have the same shape
    h, w = images[0].shape
    for i, img in enumerate(images):
        if img.shape != (h, w):
            raise RuntimeError(
                f"Layer size mismatch: {layer_paths[i]} has {img.shape}, expected {(h, w)}"
            )

    # Stack layers using the same method as local script
    # This creates (H, W, C) format like the working version
    stacked_layers = np.stack(images, axis=2)

    # Ensure proper dtype - match the local script behavior
    # Don't convert to float32 here, let the inference function handle it
    return stacked_layers


def download_model_weights(model_name: str, dest_dir: str, s3_client, profiler: Optional[WorkflowProfiler] = None) -> str:
    """
    Resolve model weights from a direct path/URI, S3 registry key, or Hugging Face repo.

    Search order preference for files: .ckpt, .safetensors, .bin, .pt
    """
    os.makedirs(dest_dir, exist_ok=True)
    model_name = model_name.strip()

    # 0) Allow direct local filesystem paths for ad hoc checkpoints.
    if os.path.isfile(model_name) and is_weight_file(model_name):
        local_path = os.path.abspath(model_name)
        logger.info(f"Using local model weights: {local_path}")
        return local_path

    # 0b) Allow direct S3 URIs to a specific checkpoint file.
    if model_name.startswith("s3://") and is_weight_file(model_name):
        bucket, key = parse_s3_uri(model_name)
        local_path = os.path.join(dest_dir, os.path.basename(key))
        logger.info(f"Downloading model weights from direct S3 URI: {model_name}")
        profiled_s3_download_file(s3_client, bucket, key, local_path, profiler)
        logger.info(f"Downloaded weights from S3 to: {local_path}")
        return local_path

    registry_bucket = "scrollprize-models-registry"
    registry_prefix = f"ink-detection/{model_name.rstrip('/')}/"

    def _prefer(weights: List[str]) -> Optional[str]:
        if not weights:
            return None
        order = [".ckpt", ".safetensors", ".bin", ".pt"]
        for ext in order:
            matches = [w for w in weights if w.lower().endswith(ext)]
            if matches:
                return sorted(matches)[0]
        return sorted(weights)[0]

    # 1) Try S3 registry first
    logger.info(
        f"Attempting to locate weights in S3 registry s3://{registry_bucket}/{registry_prefix}"
    )
    try:
        paginator = s3_client.get_paginator("list_objects_v2")
        found_keys: List[str] = []
        with scoped_timer(profiler, "s3_list_seconds"):
            for page in paginator.paginate(Bucket=registry_bucket, Prefix=registry_prefix):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    lower = key.lower()
                    if lower.endswith(".ckpt") or lower.endswith(".safetensors") or lower.endswith(".bin") or lower.endswith(".pt"):
                        found_keys.append(key)

        chosen_key = _prefer(found_keys)
        if chosen_key:
            logger.info(f"Found weights in S3 registry: s3://{registry_bucket}/{chosen_key}")
            local_path = os.path.join(dest_dir, os.path.basename(chosen_key))
            profiled_s3_download_file(s3_client, registry_bucket, chosen_key, local_path, profiler)
            logger.info(f"Downloaded weights from S3 to: {local_path}")
            return local_path
        else:
            logger.info("No suitable weights found in S3 registry. Falling back to Hugging Face.")
    except Exception as e:
        logger.warning(f"S3 registry lookup failed ({e}). Falling back to Hugging Face.")

    # 2) Fall back to Hugging Face
    logger.info(f"Downloading model from Hugging Face: {model_name}")
    with scoped_timer(profiler, "download_seconds", flag=FLAG_ESTIMATED):
        local_dir = snapshot_download(repo_id=model_name, local_dir=dest_dir, local_dir_use_symlinks=False)
    candidates = []
    for root, _, files in os.walk(local_dir):
        for f in files:
            lf = f.lower()
            if lf.endswith(".ckpt") or lf.endswith(".safetensors") or lf.endswith(".bin") or lf.endswith(".pt"):
                candidates.append(os.path.join(root, f))
    if not candidates:
        raise RuntimeError("No model weight files (.ckpt/.safetensors/.bin/.pt) found in downloaded repo")
    chosen = _prefer(candidates)
    if profiler is not None:
        profiler.add_note("Hugging Face weight download byte accounting is unavailable; only wall time is recorded.")
    logger.info(f"Using model weights: {chosen}")
    return chosen


def get_wk_dataset_metadata(wk_dataset_id: str) -> str:
    """
    Fetch metadata from WebKnossos dataset and extract s3_path.
    
    Args:
        wk_dataset_id: WebKnossos dataset ID
        
    Returns:
        s3_path extracted from dataset metadata
        
    Raises:
        RuntimeError: If s3_path is not found in metadata
    """
    try:
        logger.info(f"Fetching metadata for WebKnossos dataset: {wk_dataset_id}")
        dataset = Dataset.open_remote(wk_dataset_id)
        metadata = dataset.metadata

        if "s3_path" not in metadata:
            raise RuntimeError(f"s3_path not found in metadata for dataset {wk_dataset_id}")

        s3_path = metadata["s3_path"]
        logger.info(f"Found s3_path in dataset metadata: {s3_path}")
        return s3_path

    except Exception as e:
        logger.error(f"Failed to fetch metadata from WebKnossos dataset {wk_dataset_id}: {e}")
        raise RuntimeError(f"Failed to fetch WebKnossos dataset metadata: {e}") from e


def upload_to_webknossos(wk_dataset_id: str, prediction: np.ndarray, model_key: str, start_layer: int, end_layer: int) -> str:
    """
    Upload prediction results to WebKnossos dataset as a new layer.
    
    Args:
        wk_dataset_id: WebKnossos dataset ID
        prediction: Prediction array to upload
        model_key: Model identifier for layer naming
        start_layer: Start layer index
        end_layer: End layer index
        
    Returns:
        Layer name of uploaded prediction
    """
    try:
        logger.info(f"Uploading prediction to WebKnossos dataset: {wk_dataset_id}")

        # Open the remote dataset
        dataset = Dataset.open_remote(wk_dataset_id)

        # Create layer name
        layer_name = f"ink_prediction_{sanitize_model_tag(model_key)}_{start_layer:02d}_{end_layer:02d}"
        logger.info(f"Creating layer: {layer_name}")

        # Convert prediction to uint8
        prediction_uint8 = (np.clip(prediction, 0, 1) * 255).astype(np.uint8)

        # Add layer to dataset
        # The prediction is 2D, so we need to add a third dimension for WebKnossos
        prediction_3d = prediction_uint8[:, :, np.newaxis]

        layer = dataset.add_layer(
            layer_name=layer_name,
            category="segmentation",
            dtype_per_channel="uint8",
            num_channels=1,
            data_format="wkw"
        )

        # Write the prediction data
        with layer.open_mag(Mag(1)) as mag:
            mag.write(prediction_3d, offset=(0, 0, 0))

        logger.info(f"Successfully uploaded prediction as layer: {layer_name}")
        return layer_name

    except Exception as e:
        logger.error(f"Failed to upload to WebKnossos: {e}")
        raise RuntimeError(f"Failed to upload prediction to WebKnossos: {e}") from e


def save_and_upload_prediction(
    s3_client, prediction: np.ndarray, output_path: str, model_key: str, start_layer: int, end_layer: int,
    default_bucket: str = None, default_prefix: str = None
) -> str:
    """
    Save and upload prediction to specified output path.

    Args:
        s3_client: Boto3 S3 client
        prediction: Prediction array to save
        output_path: Full output path (S3 URI or local path). If empty, uses default bucket/prefix
        model_key: Model identifier
        start_layer: Start layer index
        end_layer: End layer index
        default_bucket: Default S3 bucket (used if output_path is empty)
        default_prefix: Default S3 prefix (used if output_path is empty)

    Returns:
        Final output path (S3 URI or local path)
    """
    prediction_uint8 = (np.clip(prediction, 0, 1) * 255).astype(np.uint8)

    # Determine output path
    if output_path:
        final_output_path = output_path
    elif default_bucket and default_prefix:
        # Use legacy default: s3://bucket/prefix/predictions/prediction_MODEL_START_END.png
        model_tag = sanitize_model_tag(model_key)
        out_key = os.path.join(default_prefix.rstrip("/"), "predictions", f"prediction_{model_tag}_{start_layer:02d}_{end_layer:02d}.png")
        final_output_path = f"s3://{default_bucket}/{out_key}"
    else:
        raise ValueError("No output_path specified and no default bucket/prefix provided")

    # Handle S3 upload
    if final_output_path.startswith("s3://"):
        bucket, key = parse_s3_uri(final_output_path)

        # Save to local temp file first
        output_dir = os.path.join(get_runtime_dir(), "outputs")
        os.makedirs(output_dir, exist_ok=True)
        local_path = os.path.join(output_dir, os.path.basename(key))
        cv2.imwrite(local_path, prediction_uint8)

        # Upload to S3
        logger.info(f"Uploading prediction to S3: {final_output_path}")
        s3_client.upload_file(local_path, bucket, key)
    else:
        # Local file path
        os.makedirs(os.path.dirname(final_output_path), exist_ok=True)
        logger.info(f"Saving prediction to local path: {final_output_path}")
        cv2.imwrite(final_output_path, prediction_uint8)

    return final_output_path


def run_prepare_step(inputs: Inputs, profiler: Optional[WorkflowProfiler] = None) -> None:
    """Execute the prepare step to create surface volume zarr from S3 layers."""
    # Import torch-free processing utilities
    from processing import create_surface_volume_zarr, path_exists

    logger.info("Starting prepare step: creating surface volume zarr")

    # Handle WebKnossos workflow
    if inputs.wk_inference:
        logger.info(f"WebKnossos inference mode: fetching metadata for dataset {inputs.wk_dataset_id}")
        inputs.s3_path = get_wk_dataset_metadata(inputs.wk_dataset_id)
        logger.info(f"Retrieved s3_path from WebKnossos metadata: {inputs.s3_path}")

    if inputs.local_layers_dir:
        layers_dir = inputs.local_layers_dir
        if not os.path.isdir(layers_dir):
            raise RuntimeError(f"Local layers dir not found: {layers_dir}")
        supported_exts = {".tif", ".tiff", ".png", ".jpeg", ".jpg"}
        layer_paths = []
        for name in os.listdir(layers_dir):
            path = os.path.join(layers_dir, name)
            stem, ext = os.path.splitext(name)
            if ext.lower() not in supported_exts:
                continue
            try:
                layer_idx = int(stem)
            except ValueError:
                continue
            layer_paths.append((layer_idx, path))
        if not layer_paths:
            raise RuntimeError(f"No local layers found under {layers_dir}")
        layer_paths.sort(key=lambda item: item[0])
        layer_paths = [path for _, path in layer_paths]
        logger.info(
            "Found %s local layers in %s; prepare will build the full local zarr and inference will later use range [%s, %s)",
            len(layer_paths),
            layers_dir,
            inputs.start_layer,
            inputs.end_layer,
        )
    else:
        # S3 setup
        logger.info("Setting up S3 client...")
        s3_client = boto3.client("s3")
        logger.info(f"Parsing S3 URI: {inputs.s3_path}")
        bucket, prefix = parse_s3_uri(inputs.s3_path)

        logger.info(f"Listing layer objects in S3 bucket '{bucket}' with prefix '{prefix}' for layers [{inputs.start_layer}, {inputs.end_layer})")
        layer_objects = list_layers_objects(
            s3_client, bucket, prefix, inputs.start_layer, inputs.end_layer, profiler=profiler
        )
        logger.info(f"Found {len(layer_objects)} layer objects to download")

        # Download layers to temporary directory
        work_dir = get_work_dir()
        input_dir = os.path.join(work_dir, "input", "layers")
        logger.info(f"Ensuring clean input directory at {os.path.join(work_dir, 'input')}")
        ensure_clean_dir(os.path.join(work_dir, "input"))

        logger.info(f"Downloading layer files to {input_dir} ...")
        layer_paths = download_layers(s3_client, bucket, layer_objects, input_dir, profiler=profiler)
        logger.info(f"Downloaded {len(layer_paths)} layer files")

    if inputs.surface_volume_zarr:
        output_path = inputs.surface_volume_zarr
    else:
        output_path = os.path.join(
            get_runtime_dir(),
            f"surface_volume_{inputs.start_layer:02d}_{inputs.end_layer:02d}.zarr",
        )

    logger.info(f"Creating surface volume zarr at {output_path} with chunk_size={inputs.chunk_size}, compression={inputs.use_zarr_compression}")
    created_zarr_path = create_surface_volume_zarr(
        layer_paths,
        output_path,
        chunk_size=inputs.chunk_size,
        max_workers=inputs.prepare_max_workers,
        use_compression=inputs.use_zarr_compression,
        profiler=profiler,
    )

    # Write output path to file for next step
    output_file = os.path.join(get_runtime_dir(), "surface_volume_zarr_path.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(created_zarr_path)

    logger.info(f"Prepare step completed successfully. Surface volume zarr created at: {created_zarr_path}")
    logger.info(f"Zarr path written to: {output_file}")


def run_inference_step(inputs: Inputs, profiler: Optional[WorkflowProfiler] = None) -> None:
    """Execute the inference step (either standard or partitioned mode)."""
    # Import torch and related dependencies only when doing inference
    import torch
    from torch.nn import DataParallel
    from inference import run_inference, CFG
    from processing import path_exists

    # Configure inference parameters
    raw_requested_channels = inputs.end_layer - inputs.start_layer
    downsampled_requested_channels = int(
        (raw_requested_channels + inputs.z_downsample_mean_factor - 1)
        // inputs.z_downsample_mean_factor
    )
    CFG.in_chans = inputs.model_in_chans
    CFG.tile_size = inputs.tile_size
    CFG.size = inputs.tile_size  # Set size to same as tile_size
    CFG.stride = inputs.stride
    CFG.batch_size = inputs.batch_size
    CFG.workers = inputs.inference_workers
    CFG.prefetch_factor = inputs.prefetch_factor
    CFG.accumulator_mode = inputs.accumulator_mode
    CFG.accumulator_auto_threshold_gib = inputs.accumulator_auto_threshold_gib
    CFG.accumulator_flush_every_batches = inputs.accumulator_flush_every_batches
    CFG.progress_log_every_batches = inputs.progress_log_every_batches
    CFG.progress_log_every_seconds = inputs.progress_log_every_seconds
    CFG.max_batches = inputs.max_batches
    CFG.skip_partition_write = inputs.skip_partition_write
    CFG.skip_empty_tiles = inputs.skip_empty_tiles
    CFG.prune_empty_tiles = inputs.prune_empty_tiles
    CFG.direct_single_part_write = inputs.direct_single_part_write
    CFG.use_hann_window = inputs.blend_mode == "hann"
    CFG.gaussian_sigma = inputs.gaussian_sigma
    logger.info(
        "Using %s input channels from raw z range [%s, %s) "
        "(raw=%s, after z downsample x%s=%s)",
        CFG.in_chans,
        inputs.start_layer,
        inputs.end_layer,
        raw_requested_channels,
        inputs.z_downsample_mean_factor,
        downsampled_requested_channels,
    )
    logger.info(
        f"Inference config: tile_size={CFG.tile_size}, size={CFG.size}, stride={CFG.stride}, "
        f"batch_size={CFG.batch_size}, workers={CFG.workers}, prefetch_factor={CFG.prefetch_factor}, "
        f"accumulator_mode={CFG.accumulator_mode}, "
        f"accumulator_auto_threshold_gib={CFG.accumulator_auto_threshold_gib}, "
        f"accumulator_flush_every_batches={CFG.accumulator_flush_every_batches}, "
        f"progress_log_every_batches={CFG.progress_log_every_batches}, "
        f"progress_log_every_seconds={CFG.progress_log_every_seconds}, "
        f"max_batches={CFG.max_batches}, "
        f"skip_partition_write={CFG.skip_partition_write}, "
        f"skip_empty_tiles={CFG.skip_empty_tiles}, "
        f"prune_empty_tiles={CFG.prune_empty_tiles}, "
        f"direct_single_part_write={CFG.direct_single_part_write}, "
        f"blend_mode={inputs.blend_mode}, gaussian_sigma={inputs.gaussian_sigma}, "
        f"z_downsample_mean_factor={inputs.z_downsample_mean_factor}"
    )
    if inputs.roi_xyxy is not None:
        logger.info("Inference ROI: [%s, %s, %s, %s)", *inputs.roi_xyxy)

    requested_resnet_depth = None

    # Import model-specific module based on model_type
    if inputs.model_type == "timesformer":
        from model_timesformer import load_model
        logger.info(f"Using TimeSformer model")
    elif inputs.model_type == "resnet3d" or inputs.model_type.startswith("resnet3d-"):
        from model_resnet3d import load_model
        if inputs.model_type.startswith("resnet3d-"):
            requested_resnet_depth = int(inputs.model_type.split("-", 1)[1])
            logger.info(f"Using ResNet3D model (requested depth={requested_resnet_depth})")
        else:
            logger.info("Using ResNet3D model (depth will be inferred from checkpoint)")
    else:
        raise ValueError(f"Unknown model_type: {inputs.model_type}")

    task_id = uuid.uuid4()
    logger.info(f"Task ID generated: {task_id}")

    # Verify surface volume zarr exists (supports both local paths and S3 URLs)
    if not path_exists(inputs.surface_volume_zarr):
        raise RuntimeError(f"Surface volume zarr not found at: {inputs.surface_volume_zarr}")

    logger.info(f"Using surface volume zarr: {inputs.surface_volume_zarr}")

    # Configure CFG with partition parameters
    CFG.num_parts = inputs.num_parts
    CFG.part_id = inputs.part_id
    CFG.zarr_output_dir = inputs.zarr_output_dir

    logger.info(
        f"Inference step: num_parts={inputs.num_parts}, part_id={inputs.part_id}, "
        f"zarr_output_dir={inputs.zarr_output_dir}"
    )

    logger.info(
        f"Starting optimized inference with task_id={task_id}, model={inputs.model_key}, "
        f"surface_volume_zarr={inputs.surface_volume_zarr}, "
        f"layers=[{inputs.start_layer}, {inputs.end_layer}], wk_inference={inputs.wk_inference}"
    )

    # Prepare models directory
    work_dir = get_work_dir()
    models_dir = os.path.join(work_dir, "models")
    logger.info(f"Ensuring models directory exists at {models_dir}")
    os.makedirs(models_dir, exist_ok=True)

    # S3 setup (for model download)
    logger.info("Setting up S3 client...")
    s3_client = boto3.client("s3")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Resolve and download model weights (S3-first, then HF fallback)
    logger.info(f"Resolving model for key: {inputs.model_key}")
    logger.info(f"Looking for weights in S3 registry, else HF repo: {inputs.model_key}")
    weight_path = download_model_weights(inputs.model_key, models_dir, s3_client, profiler=profiler)
    logger.info(f"Loading model from weights at: {weight_path}")

    # Load model with dynamic number of frames
    with scoped_timer(profiler, "model_load_seconds", cuda_sync=device.type == "cuda"):
        if inputs.model_type == "timesformer":
            model = load_model(weight_path, device, num_frames=CFG.in_chans)
        else:
            model = load_model(
                weight_path,
                device,
                num_frames=CFG.in_chans,
                model_depth=requested_resnet_depth,
            )
    logger.info("Model loader returned successfully")

    # -------- Performance toggles ------------------------------------------------
    # TF32 on Ampere+ gives fast GEMMs with tiny accuracy impact for this task.
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    logger.info("TF32 / matmul precision toggles applied")
    # torch.compile defaults
    COMPILE = os.getenv("COMPILE", "1") == "1" and hasattr(torch, "compile")
    COMPILE_MODE = os.getenv("COMPILE_MODE", "reduce-overhead")  # <- default changed
    logger.info("Compile gate resolved: COMPILE=%s COMPILE_MODE=%s", COMPILE, COMPILE_MODE)
    if COMPILE:
        with scoped_timer(profiler, "compile_warmup_seconds", cuda_sync=device.type == "cuda"):
            # Persist Inductor cache across runs (huge win after the first run)
            os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.abspath("./inductor_cache"))
            # If not doing max tuning, disable heavy autotuning to avoid OOM spam & overhead
            if COMPILE_MODE != "max-autotune":
                os.environ.setdefault("TORCHINDUCTOR_MAX_AUTOTUNE", "0")
                os.environ.setdefault("TORCHINDUCTOR_MAX_AUTOTUNE_GEMM", "0")
                os.environ.setdefault("TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE", "0")
            # Optional: CUDA graphs (static shapes); enable if you don’t hit driver bugs
            if os.getenv("CUDAGRAPHS", "0") == "1":
                os.environ.setdefault("TORCHINDUCTOR_CUDAGRAPHS", "1")
            # Compile
            # Access the underlying model from the wrapper
            target = model.model.module if isinstance(model.model, DataParallel) else model.model
            model_compiled = torch.compile(target, mode=COMPILE_MODE, fullgraph=True, dynamic=False)
            if isinstance(model.model, DataParallel):
                model.model.module = model_compiled
            else:
                model.model = model_compiled
            logger.info(f"Enabled torch.compile (mode={COMPILE_MODE})")
            # Tiny warmup to trigger compilation before the big loop (hides first-iter cost)
            try:
                dummy = torch.zeros((1, 1, CFG.in_chans, CFG.size, CFG.size), device=device)
                with torch.inference_mode():
                    with torch.autocast(device_type=("cuda" if device.type == "cuda" else "cpu"), enabled=True):
                        _ = model.forward(dummy)
                del dummy
            except Exception as e:
                logger.warning(f"Warmup after compile failed (continuing un-warmed): {e}")
    logger.info("Compile block complete")

    # Determine reverse option similar to local test
    if inputs.force_reverse:
        is_reverse_segment = True
        logger.info("Force reverse enabled via env")
    else:
        is_reverse_segment = False

    # Run inference from zarr
    # Note: When using a predefined surface volume zarr (not created by the prepare step),
    # it may contain more layers than requested. We use start_layer and end_layer to crop
    # to the desired range within the zarr.
    logger.info(f"Running inference from surface volume zarr with layer range [{inputs.start_layer}, {inputs.end_layer})...")
    start_infer_time = time.time()
    result = run_inference(
        inputs.surface_volume_zarr,
        model,
        device,
        is_reverse_segment=is_reverse_segment,
        start_z=inputs.start_layer,
        end_z=inputs.end_layer,
        z_downsample_mean_factor=inputs.z_downsample_mean_factor,
        target_in_chans=CFG.in_chans,
        roi_xyxy=inputs.roi_xyxy,
        profiler=profiler,
    )
    infer_elapsed = time.time() - start_infer_time
    logger.info(f"Inference completed in {infer_elapsed:.2f} seconds")
    if profiler is not None:
        partition_tiles = result.get("partition_tiles") or result.get("partition_info", {}).get("partition_tiles")
        if partition_tiles:
            profiler.set_metric(
                "partition_throughput_tiles_per_second",
                float(partition_tiles) / max(infer_elapsed, 1e-6),
                flag=FLAG_ESTIMATED,
            )

    if result.get("skipped_partition_write"):
        logger.info(
            "Inference benchmark complete for partition %d. Partition write was skipped by request.",
            inputs.part_id,
        )
    elif result.get("direct_single_part_write"):
        from processing import reduce_single_partition_arrays, write_tiled_tiff

        logger.info(
            "Direct single-part TIFF write requested for partition %d; "
            "skipping partition zarr spill and reduce re-read",
            inputs.part_id,
        )

        if inputs.add_scale_bar:
            if inputs.pixel_resolution_um is not None:
                logger.info(
                    "Scale bar enabled: %.1fmm at %.2fum/pixel",
                    inputs.scale_bar_length_um / 1000.0,
                    inputs.pixel_resolution_um,
                )
            else:
                logger.warning("ADD_SCALE_BAR=true but PIXEL_RESOLUTION_UM not set, skipping scale bar")

        tile_size = 1024
        tile_iterator, shape = reduce_single_partition_arrays(
            result["mask_pred_array"],
            result["mask_count_array"],
            tile_size,
            add_scale_bar=inputs.add_scale_bar,
            pixel_resolution_um=inputs.pixel_resolution_um,
            scale_bar_length_um=inputs.scale_bar_length_um,
            profiler=profiler,
        )
        model_tag = sanitize_model_tag(inputs.model_key)
        local_tiff_path = os.path.join(
            get_runtime_dir(),
            f"prediction_{model_tag}_{inputs.start_layer:02d}_{inputs.end_layer:02d}.tif",
        )
        start_reduce_time = time.time()
        write_tiled_tiff(tile_iterator, shape, local_tiff_path, tile_size, inputs.pixel_resolution_um, profiler=profiler)
        logger.info(f"Direct reduce and TIFF write completed in {time.time() - start_reduce_time:.2f} seconds")
        persist_local_tiff_result(inputs=inputs, local_tiff_path=local_tiff_path, profiler=profiler)
    else:
        logger.info(f"Partition {inputs.part_id} completed. Zarr arrays written to {inputs.zarr_output_dir}")
        logger.info("Inference step complete. Run STEP=reduce after all partitions finish to blend and upload results.")


def run_reduce_step(inputs: Inputs, profiler: Optional[WorkflowProfiler] = None) -> None:
    """Execute the reduce step to blend all partitions."""
    # Import torch-free processing utilities
    from processing import reduce_partitions, write_tiled_tiff

    logger.info(f"Starting reduce step: blending {inputs.num_parts} partitions")

    # Handle WebKnossos workflow to get s3_path
    if inputs.wk_inference:
        logger.info(f"WebKnossos inference mode: fetching metadata for dataset {inputs.wk_dataset_id}")
        inputs.s3_path = get_wk_dataset_metadata(inputs.wk_dataset_id)
        logger.info(f"Retrieved s3_path from WebKnossos metadata: {inputs.s3_path}")

    # We need to determine the prediction shape from one of the partition zarr files
    mask_pred_path = os.path.join(inputs.zarr_output_dir, "mask_pred_part_000.zarr")
    if not os.path.exists(mask_pred_path):
        raise RuntimeError(f"Partition 0 not found at {mask_pred_path}. Ensure all inference partitions completed.")

    logger.info(f"Reading prediction shape from {mask_pred_path}")
    with scoped_timer(profiler, "local_read_seconds", flag=FLAG_APPROXIMATE):
        z = zarr.open(mask_pred_path, mode='r')
        pred_shape = z.shape
    if profiler is not None:
        profiler.increment_counter("local_read_bytes", dir_size_bytes(mask_pred_path), flag=FLAG_APPROXIMATE)
    logger.info(f"Prediction shape: {pred_shape}")

    # Run reduce/blend (creates lazy tile iterator)
    logger.info(f"Reducing {inputs.num_parts} partitions from {inputs.zarr_output_dir}")
    tile_size = 1024
    tile_iterator, shape = reduce_partitions(
        inputs.zarr_output_dir,
        inputs.num_parts,
        pred_shape,
        tile_size,
        add_scale_bar=inputs.add_scale_bar,
        pixel_resolution_um=inputs.pixel_resolution_um,
        scale_bar_length_um=inputs.scale_bar_length_um,
        profiler=profiler,
    )

    # Log scale bar status
    if inputs.add_scale_bar:
        if inputs.pixel_resolution_um is not None:
            logger.info(f"Scale bar enabled: {inputs.scale_bar_length_um/1000:.1f}mm at {inputs.pixel_resolution_um:.2f}um/pixel")
        else:
            logger.warning("ADD_SCALE_BAR=true but PIXEL_RESOLUTION_UM not set, skipping scale bar")

    # Write to local tiled TIFF first (this is when the lazy reduction actually happens)
    model_tag = sanitize_model_tag(inputs.model_key)
    local_tiff_path = os.path.join(
        get_runtime_dir(),
        f"prediction_{model_tag}_{inputs.start_layer:02d}_{inputs.end_layer:02d}.tif",
    )
    start_reduce_time = time.time()
    write_tiled_tiff(tile_iterator, shape, local_tiff_path, tile_size, inputs.pixel_resolution_um, profiler=profiler)
    logger.info(f"Reduce and TIFF write completed in {time.time() - start_reduce_time:.2f} seconds")
    result_uri = persist_local_tiff_result(inputs=inputs, local_tiff_path=local_tiff_path, profiler=profiler)

    # If WebKnossos mode, also upload to WebKnossos
    if inputs.wk_inference:
        logger.info("Uploading prediction to WebKnossos dataset...")
        # Read the TIFF back as numpy array for WebKnossos upload
        logger.info(f"Reading TIFF for WebKnossos upload: {local_tiff_path}")
        with scoped_timer(profiler, "local_read_seconds", flag=FLAG_APPROXIMATE):
            prediction = tiff.imread(local_tiff_path)
        if profiler is not None:
            profiler.increment_counter("local_read_bytes", os.path.getsize(local_tiff_path), flag=FLAG_APPROXIMATE)
        # Convert uint8 back to float32 [0, 1] as expected by upload_to_webknossos
        prediction = prediction.astype(np.float32) / 255.0

        wk_layer_name = upload_to_webknossos(
            inputs.wk_dataset_id, prediction, inputs.model_key, inputs.start_layer, inputs.end_layer
        )
        logger.info(f"Uploaded prediction to WebKnossos layer: {wk_layer_name}")

        with open(os.path.join(get_runtime_dir(), "result_wk_layer.txt"), "w", encoding="utf-8") as f:
            f.write(wk_layer_name)

        logger.info(f"Reduce completed successfully - S3: {result_uri}, WebKnossos layer: {wk_layer_name}")
    else:
        logger.info(f"Reduce completed successfully - S3: {result_uri}")


def run_aggregate_profiling_step(inputs: Inputs, profiler: Optional[WorkflowProfiler] = None) -> None:
    output_dir = os.path.join(get_runtime_dir(), "profiling-outputs")
    os.makedirs(output_dir, exist_ok=True)
    try:
        with scoped_timer(profiler, "reduce_seconds", flag=FLAG_APPROXIMATE):
            results = aggregate_workflow_profiling(
                inputs.profiling_raw_root,
                output_dir,
                output_prefix=inputs.profiling_output_prefix,
            )
        write_aggregate_output_parameters(results)
    except Exception as exc:
        logger.exception("Profiling aggregation failed: %s", exc)
        if profiler is not None:
            profiler.add_note(f"Aggregation failed: {type(exc).__name__}: {exc}")
        failure_summary = {
            "schema_version": "1.0",
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "raw_root": inputs.profiling_raw_root,
            "error": f"{type(exc).__name__}: {exc}",
        }
        summary_json = os.path.join(output_dir, "workflow-profiling-summary.json")
        summary_md = os.path.join(output_dir, "workflow-profiling-summary.md")
        partitions_csv = os.path.join(output_dir, "workflow-profiling-partitions.csv")
        partitions_jsonl = os.path.join(output_dir, "workflow-profiling-partitions.jsonl")
        with open(summary_json, "w", encoding="utf-8") as handle:
            json.dump(failure_summary, handle, indent=2)
            handle.write("\n")
        with open(summary_md, "w", encoding="utf-8") as handle:
            handle.write("# Workflow Profiling Summary\n\n")
            handle.write(f"Aggregation failed: `{type(exc).__name__}: {exc}`\n")
        with open(partitions_csv, "w", encoding="utf-8") as handle:
            handle.write("part_id,pod_name,total_wall_seconds,dominant_phase,classification\n")
        with open(partitions_jsonl, "w", encoding="utf-8") as handle:
            handle.write("")
        write_aggregate_output_parameters(
            {
                "summary_json": summary_json,
                "summary_md": summary_md,
                "partitions_csv": partitions_csv,
                "partitions_jsonl": partitions_jsonl,
            }
        )


def main() -> None:
    """Main entrypoint with step dispatch."""
    logger.info("Parsing environment variables for input configuration...")
    inputs = parse_env()
    template_name = os.getenv("PROFILING_TEMPLATE_NAME", inputs.step)
    profiler = build_profiler(inputs, template_name)
    workflow_error: Optional[BaseException] = None
    profiler_status = "succeeded"

    logger.info(
        f"Starting optimized inference: step={inputs.step}, model={inputs.model_key}, "
        f"s3_path={inputs.s3_path}, layers=[{inputs.start_layer}, {inputs.end_layer}]"
    )

    try:
        # Dispatch to appropriate step
        if inputs.step == "prepare":
            run_prepare_step(inputs, profiler=profiler)
        elif inputs.step == "inference":
            run_inference_step(inputs, profiler=profiler)
        elif inputs.step == "reduce":
            run_reduce_step(inputs, profiler=profiler)
        elif inputs.step == "aggregate-profiling":
            run_aggregate_profiling_step(inputs, profiler=profiler)
        else:
            raise ValueError(f"Unknown step: {inputs.step}")
    except Exception as exc:
        workflow_error = exc
        profiler_status = "failed"
        raise
    finally:
        with suppress(Exception):
            profiler.flush(profiler_status, error=workflow_error)


if __name__ == "__main__":
    main()
