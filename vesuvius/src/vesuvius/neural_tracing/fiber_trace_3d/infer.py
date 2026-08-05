from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
from typing import Any

import torch
import zarr

try:
    from lasagna.tiled_predict3d import (
        DEFAULT_FLUSH_WORKERS, DEFAULT_ACCUMULATOR_WORKERS,
        DEFAULT_INPUT_CACHE_BYTES,
        DEFAULT_INPUT_COPY_THREADS,
        DEFAULT_INPUT_IO_THREADS,
        DEFAULT_INPUT_READER,
        DEFAULT_PREFETCH_TILES_PER_GPU,
        _auto_download,
        build_product_omezarr_pyramids,
        create_product_omezarr_groups,
        _cleanup_predict3d_temp_files,
        _crop_xyzwhd_bounds,
        _ds_index,
        run_tiled_inference_3d,
        resolve_inference_devices,
        OmeZarrOutputAdapter,
        write_lasagna_product_manifest,
    )
except ImportError:  # pragma: no cover - supports PYTHONPATH=lasagna style runs.
    from tiled_predict3d import (
        DEFAULT_FLUSH_WORKERS, DEFAULT_ACCUMULATOR_WORKERS,
        DEFAULT_INPUT_CACHE_BYTES,
        DEFAULT_INPUT_COPY_THREADS,
        DEFAULT_INPUT_IO_THREADS,
        DEFAULT_INPUT_READER,
        DEFAULT_PREFETCH_TILES_PER_GPU,
        _auto_download,
        build_product_omezarr_pyramids,
        create_product_omezarr_groups,
        _cleanup_predict3d_temp_files,
        _crop_xyzwhd_bounds,
        _ds_index,
        run_tiled_inference_3d,
        resolve_inference_devices,
        OmeZarrOutputAdapter,
        write_lasagna_product_manifest,
    )

from vesuvius.neural_tracing.fiber_trace_3d.inference_adapter import (
    FiberTrace3DPredictAdapter,
)

try:
    from lasagna.tiled_predict3d import _resolve_base_shape
except ImportError:  # pragma: no cover
    from tiled_predict3d import _resolve_base_shape


def _load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"{config_path} must contain a JSON object")
    config.setdefault("_config_dir", str(config_path.parent))
    return config


def _tile_size_from_config(config: dict[str, Any], explicit_tile_size: int | None) -> int:
    if explicit_tile_size is not None:
        tile_size = int(explicit_tile_size)
    else:
        shape = config.get("patch_shape_zyx", config.get("crop_size"))
        if shape is None:
            raise ValueError("pass --tile-size or set patch_shape_zyx in the config")
        shape_tuple = tuple(int(v) for v in shape)
        if len(shape_tuple) != 3 or len(set(shape_tuple)) != 1:
            raise ValueError(
                "fiber tiled inference currently requires cubic tiles; "
                f"pass --tile-size explicitly for patch_shape_zyx={shape_tuple}"
            )
        tile_size = shape_tuple[0]
    if tile_size <= 0:
        raise ValueError("tile_size must be > 0")
    return tile_size


def _level_from_scaledown(scaledown: int) -> int:
    sd = int(scaledown)
    if sd <= 0 or sd & (sd - 1):
        raise ValueError(f"scaledown must be an exact positive power of two, got {sd}")
    return sd.bit_length() - 1


def _storage_ds_end(value: int, scaledown: int) -> int:
    """Map a positive exclusive input bound to a ceil-sized storage bound."""
    value_i = max(0, int(value))
    scaledown_i = max(1, int(scaledown))
    return (value_i + scaledown_i - 1) // scaledown_i


def _input_scaledown_from_base(
    base_shape_zyx: tuple[int, int, int], input_shape_zyx: tuple[int, int, int]
) -> int:
    """Find one isotropic pyramid factor, accepting ceil-divided edge shapes."""
    matches = [
        1 << power
        for power in range(31)
        if all((int(base) + (1 << power) - 1) // (1 << power) == int(actual)
               for base, actual in zip(base_shape_zyx, input_shape_zyx))
    ]
    if len(matches) != 1:
        raise ValueError(
            "input shape is not one unambiguous isotropic power-of-two pyramid "
            f"level of base shape: base={base_shape_zyx}, input={input_shape_zyx}"
        )
    return matches[0]


def _resolve_inference_device(device: str | None) -> torch.device:
    requested = "auto" if device is None else str(device).strip().lower()
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def _checkpoint_mixed_precision_policy(checkpoint: str | Path | None) -> Any:
    if checkpoint is None:
        return None
    payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        return None
    config = payload.get("config")
    if not isinstance(config, dict):
        return None
    training = config.get("training")
    if not isinstance(training, dict):
        return None
    return training.get("mixed_precision")


def _resolve_inference_precision(
    requested: str,
    *,
    checkpoint: str | Path | None,
    devices: tuple[torch.device, ...],
) -> tuple[str, str]:
    from vesuvius.neural_tracing.fiber_trace_3d.train import (
        _mixed_precision_config_from_training,
        _normalize_mixed_precision_mode,
    )

    request = str(requested).strip().lower()
    derived = request == "auto"
    source = "cli"
    if derived:
        raw = _checkpoint_mixed_precision_policy(checkpoint)
        source = "checkpoint"
        if raw is None:
            return "off", "checkpoint-missing"
        try:
            mode = _normalize_mixed_precision_mode(raw)
        except ValueError as exc:
            print(f"[fiber_trace_3d:infer] WARNING: invalid checkpoint mixed precision ({exc}); using fp32", flush=True)
            return "off", "checkpoint-invalid"
        if mode == "auto":
            print(
                "[fiber_trace_3d:infer] WARNING: checkpoint mixed_precision='auto' does not record "
                "the historically resolved dtype; using fp32",
                flush=True,
            )
            return "off", "checkpoint-ambiguous"
    else:
        mode = _normalize_mixed_precision_mode(request)

    if mode == "auto":
        mode = "off"
    try:
        for selected in devices:
            if selected.type == "cuda":
                with torch.cuda.device(selected):
                    _mixed_precision_config_from_training({"mixed_precision": mode}, selected)
            else:
                _mixed_precision_config_from_training({"mixed_precision": mode}, selected)
    except ValueError as exc:
        if not derived:
            raise
        print(
            f"[fiber_trace_3d:infer] WARNING: checkpoint precision {mode!r} is unsupported "
            f"on selected devices ({exc}); using fp32",
            flush=True,
        )
        return "off", "checkpoint-unsupported"
    return mode, source


def _select_and_expand_crop(
    *,
    input_shape_zyx: tuple[int, int, int],
    crop_xyzwhd_base: tuple[int, int, int, int, int, int] | None,
    input_scaledown_from_base: int,
    output_scaledown_from_input: int,
    ome_chunk: int,
) -> tuple[
    tuple[int, int, int, int, int, int],
    tuple[int, int, int, int, int, int],
    tuple[int, int, int],
]:
    if crop_xyzwhd_base is None:
        crop_input = None
    else:
        bx, by, bz, bw, bh, bd = (int(v) for v in crop_xyzwhd_base)
        input_sd = max(1, int(input_scaledown_from_base))
        crop_input = (
            bx // input_sd,
            by // input_sd,
            bz // input_sd,
            max(1, bw // input_sd),
            max(1, bh // input_sd),
            max(1, bd // input_sd),
        )

    z0, z1, y0, y1, x0, x1 = _crop_xyzwhd_bounds(
        shape_zyx=input_shape_zyx,
        crop_xyzwhd=crop_input,
    )
    nz, ny, nx = z1 - z0, y1 - y0, x1 - x0
    if nz <= 0 or ny <= 0 or nx <= 0:
        raise ValueError(
            f"empty crop: x=[{x0},{x1}) y=[{y0},{y1}) z=[{z0},{z1}) "
            f"in shape={input_shape_zyx}"
        )

    sd = max(1, int(output_scaledown_from_input))
    oc = max(1, int(ome_chunk))
    full_out_shape = tuple(_storage_ds_end(size, sd) for size in input_shape_zyx)

    oz0 = (_ds_index(z0, sd) // oc) * oc
    oy0 = (_ds_index(y0, sd) // oc) * oc
    ox0 = (_ds_index(x0, sd) // oc) * oc
    oz1 = min(
        full_out_shape[0],
        ((_storage_ds_end(z1, sd) + oc - 1) // oc) * oc,
    )
    oy1 = min(
        full_out_shape[1],
        ((_storage_ds_end(y1, sd) + oc - 1) // oc) * oc,
    )
    ox1 = min(
        full_out_shape[2],
        ((_storage_ds_end(x1, sd) + oc - 1) // oc) * oc,
    )

    z0 = max(0, min(z0, oz0 * sd))
    y0 = max(0, min(y0, oy0 * sd))
    x0 = max(0, min(x0, ox0 * sd))
    z1 = max(z1, min(input_shape_zyx[0], oz1 * sd))
    y1 = max(y1, min(input_shape_zyx[1], oy1 * sd))
    x1 = max(x1, min(input_shape_zyx[2], ox1 * sd))

    oz0 = (_ds_index(z0, sd) // oc) * oc
    oy0 = (_ds_index(y0, sd) // oc) * oc
    ox0 = (_ds_index(x0, sd) // oc) * oc
    oz1 = min(
        full_out_shape[0],
        ((_storage_ds_end(z1, sd) + oc - 1) // oc) * oc,
    )
    oy1 = min(
        full_out_shape[1],
        ((_storage_ds_end(y1, sd) + oc - 1) // oc) * oc,
    )
    ox1 = min(
        full_out_shape[2],
        ((_storage_ds_end(x1, sd) + oc - 1) // oc) * oc,
    )

    return (
        (z0, z1, y0, y1, x0, x1),
        (oz0, oy0, ox0, oz1, oy1, ox1),
        full_out_shape,
    )


def run_fiber_trace_3d_inference(
    *,
    config_path: str | Path,
    input_path: str,
    output_path: str | Path,
    checkpoint: str | Path | None,
    device: str | None = None,
    devices: str | tuple[str, ...] | None = None,
    crop_xyzwhd: tuple[int, int, int, int, int, int] | None = None,
    tile_size: int | None = None,
    overlap: int = 16,
    border: int = 16,
    inference_scaledown_power: int = 2,
    base_ref: str | None = None,
    base_scale: int | None = None,
    no_download: bool = False,
    levels: int = 5,
    ome_chunk: int = 64,
    pyramid_workers: int = 0,
    recurrent_steps: int | None = None,
    prefetch_workers: int = 0,
    slots_per_gpu: int = 2,
    flush_workers: int = DEFAULT_FLUSH_WORKERS,
    accumulator_workers: int = DEFAULT_ACCUMULATOR_WORKERS,
    input_reader: str = DEFAULT_INPUT_READER,
    prefetch_tiles_per_gpu: int = DEFAULT_PREFETCH_TILES_PER_GPU,
    input_cache_gib: float = DEFAULT_INPUT_CACHE_BYTES / float(1 << 30),
    input_io_threads: int = DEFAULT_INPUT_IO_THREADS,
    input_copy_threads: int = DEFAULT_INPUT_COPY_THREADS,
    download_workers: int = 64,
    profile_pipeline: bool = False,
    inference_precision: str = "auto",
    product_accumulator_dtype: str = "float16",
) -> None:
    if int(download_workers) <= 0:
        raise ValueError("download_workers must be a positive integer")
    if int(flush_workers) < 0:
        raise ValueError("flush_workers must be >= 0")
    if int(accumulator_workers) < 0:
        raise ValueError("accumulator_workers must be >= 0")
    if not math.isfinite(float(input_cache_gib)) or float(input_cache_gib) < 0:
        raise ValueError("input_cache_gib must be finite and >= 0")
    if int(prefetch_tiles_per_gpu) <= 0 or int(input_io_threads) <= 0 or int(input_copy_threads) <= 0:
        raise ValueError("prefetch_tiles_per_gpu and TensorStore thread counts must be > 0")
    config = _load_config(config_path)
    tile_size_i = _tile_size_from_config(config, tile_size)
    output_manifest = Path(output_path)
    if not output_manifest.name.endswith(".lasagna.json"):
        raise ValueError(f"output must be .lasagna.json, got: {output_path}")
    output_dir = output_manifest.parent
    json_stem = output_manifest.name.removesuffix(".lasagna.json")
    if not json_stem:
        raise ValueError("output .lasagna.json path must have a non-empty stem")

    if not no_download:
        _auto_download(input_path, crop_xyzwhd, download_workers=int(download_workers))

    a_in = zarr.open(str(input_path), mode="r")
    if not hasattr(a_in, "shape"):
        raise ValueError(f"input must point to a zarr array, got: {input_path}")
    input_shape = tuple(int(v) for v in a_in.shape)
    if len(input_shape) != 3:
        raise ValueError(f"input array must be (Z,Y,X), got shape {input_shape}")

    base_shape_zyx = _resolve_base_shape(input_path, base_ref, base_scale)
    if base_shape_zyx is None:
        raise ValueError(
            "cannot determine base_shape_zyx. Pass --base-ref or use an input "
            "inside an OME-Zarr group with level 0 metadata."
        )
    input_sd = _input_scaledown_from_base(base_shape_zyx, input_shape)
    power = int(inference_scaledown_power)
    if power < 0 or power > 30:
        raise ValueError(f"inference_scaledown_power must be in [0, 30], got {power}")
    output_sd_input = 1 << power
    effective_output_sd = input_sd * output_sd_input
    output_level = _level_from_scaledown(effective_output_sd)
    n_levels = max(int(levels), output_level + 2)

    crop_slices, output_region, full_output_shape = _select_and_expand_crop(
        input_shape_zyx=input_shape,
        crop_xyzwhd_base=crop_xyzwhd,
        input_scaledown_from_base=input_sd,
        output_scaledown_from_input=output_sd_input,
        ome_chunk=ome_chunk,
    )
    z0, z1, y0, y1, x0, x1 = crop_slices
    oz0, oy0, ox0, oz1, oy1, ox1 = output_region

    resolved_devices = resolve_inference_devices(device=device, devices=devices)
    torch_device = resolved_devices[0]

    precision_mode, precision_source = _resolve_inference_precision(
        inference_precision, checkpoint=checkpoint, devices=resolved_devices,
    )
    config = dict(config)
    effective_training = dict(config.get("training", {}))
    effective_training["mixed_precision"] = precision_mode
    config["training"] = effective_training
    print(
        f"[fiber_trace_3d:infer] inference_precision="
        f"{'fp32' if precision_mode == 'off' else precision_mode} source={precision_source}",
        flush=True,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    removed = _cleanup_predict3d_temp_files(output_dir, f"{json_stem}_")
    if removed > 0:
        print(
            f"[fiber_trace_3d:infer] removed {removed} stale temp path(s) from {output_dir}",
            flush=True,
        )

    predict_adapter = FiberTrace3DPredictAdapter(
        config,
        checkpoint=checkpoint,
        level=output_level,
        scaledown=effective_output_sd,
        inference_scaledown=output_sd_input,
        chunk_size=int(ome_chunk),
        product_prefix=json_stem,
        zarr_path_prefix=output_dir / json_stem,
        recurrent_steps=recurrent_steps,
    )
    output_adapter = OmeZarrOutputAdapter(
        products=predict_adapter.output_products,
        n_levels=n_levels,
    )
    create_product_omezarr_groups(
        products=predict_adapter.output_products,
        base_shape_zyx=base_shape_zyx,
        n_levels=n_levels,
        ome_chunk=int(ome_chunk),
    )
    write_lasagna_product_manifest(
        output_path=output_manifest,
        products=predict_adapter.output_products,
        base_shape_zyx=base_shape_zyx,
        crop_xyzwhd_base=crop_xyzwhd,
    )

    print(
        f"[fiber_trace_3d:infer] input={input_path} shape={input_shape} "
        f"input_sd={input_sd} base_shape={base_shape_zyx}",
        flush=True,
    )
    print(
        f"[fiber_trace_3d:infer] crop_zyx=({z0},{z1},{y0},{y1},{x0},{x1}) "
        f"output_region_zyx=({oz0},{oy0},{ox0},{oz1},{oy1},{ox1}) "
        f"inference_scaledown_power={power} inference_factor={output_sd_input} "
        f"effective_base_factor={effective_output_sd} "
        f"level={output_level} products={len(predict_adapter.output_products)} "
        f"devices={','.join(str(value) for value in resolved_devices)}",
        flush=True,
    )

    t0 = time.time()
    model = None
    if len(resolved_devices) == 1:
        print(
            f"[fiber_trace_3d:infer] loading checkpoint={checkpoint} device={torch_device}",
            flush=True,
        )
        model = predict_adapter.load_model(device=torch_device)
        print("[fiber_trace_3d:infer] model loaded; starting tiled inference", flush=True)
    else:
        print("[fiber_trace_3d:infer] starting persistent multi-GPU workers", flush=True)
    progress = {
        "t0": time.time(),
        "finalized_base_z": int(oz0 * effective_output_sd),
        "finalized_base_z_total": int(oz1 * effective_output_sd),
    }
    run_tiled_inference_3d(
        model,
        a_in,
        crop_slices=crop_slices,
        device=torch_device,
        model_adapter=predict_adapter,
        output_adapter=output_adapter,
        products=predict_adapter.output_products,
		output_regions_zyx={p.name: output_region for p in predict_adapter.output_products},
		full_output_shapes_zyx={p.name: full_output_shape for p in predict_adapter.output_products},
        input_zarr_path=str(input_path),
		output_scaledown_base={p.name: effective_output_sd for p in predict_adapter.output_products},
        tile_size=tile_size_i,
        overlap=int(overlap),
        border=int(border),
        tmp_dir=str(output_dir),
        progress=progress,
        temp_prefix=f"{json_stem}_",
        devices=resolved_devices,
        prefetch_workers=int(prefetch_workers),
        slots_per_gpu=int(slots_per_gpu),
        flush_workers=int(flush_workers),
        input_reader=str(input_reader),
        prefetch_tiles_per_gpu=int(prefetch_tiles_per_gpu),
        input_cache_bytes=int(float(input_cache_gib) * (1 << 30)),
        input_io_threads=int(input_io_threads),
        input_copy_threads=int(input_copy_threads),
        profile_pipeline=bool(profile_pipeline),
        product_accumulator_dtype=str(product_accumulator_dtype),
        accumulator_workers=int(accumulator_workers),
    )
    del model
    if torch_device.type == "cuda":
        torch.cuda.empty_cache()

    build_product_omezarr_pyramids(
        products=predict_adapter.output_products,
        n_levels=n_levels,
        ome_chunk=int(ome_chunk),
        crop_zyx=output_region,
        workers=int(pyramid_workers),
    )
    removed_finish = _cleanup_predict3d_temp_files(
        output_dir,
        f"{json_stem}_",
        remove_current_process=True,
    )
    if removed_finish > 0:
        print(
            f"[fiber_trace_3d:infer] removed {removed_finish} temp path(s) on finish",
            flush=True,
        )
    print(
        f"[fiber_trace_3d:infer] done output={output_manifest} "
        f"elapsed={time.time() - t0:.1f}s",
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run shared tiled 3D inference for fiber trace models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("config", help="Fiber trace 3D training/inference config JSON.")
    parser.add_argument("--input", required=True, help="Input zarr array (3D ZYX).")
    parser.add_argument("--output", required=True, help="Output .lasagna.json manifest path.")
    parser.add_argument("--checkpoint", required=True, help="Fiber trace 3D model checkpoint (.pt).")
    parser.add_argument("--tile-size", type=int, default=None, help="Inference tile cube size.")
    parser.add_argument("--overlap", type=int, default=16, help="Tile overlap in input voxels.")
    parser.add_argument("--border", type=int, default=16, help="Hard discard border at tile edges.")
    parser.add_argument(
        "--inference-scaledown-power", type=int, default=2,
        help="Power-of-two output reduction relative to input (2 means factor 4).",
    )
    parser.add_argument(
        "--crop",
        "--crop-xyzwhd",
        dest="crop_xyzwhd",
        type=int,
        nargs=6,
        default=None,
        metavar=("X", "Y", "Z", "W", "H", "D"),
        help="Crop in base coordinates: x y z w h d.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help='Device: "auto" selects CUDA when available, otherwise CPU.',
    )
    parser.add_argument(
        "--devices",
        default=None,
        help='Multi-GPU selection: "all" or comma-separated CUDA devices.',
    )
    parser.add_argument(
        "--prefetch-workers", type=int, default=0,
        help="CPU/Zarr tile reader threads; 0 chooses a bounded automatic count.",
    )
    parser.add_argument(
        "--slots-per-gpu", type=int, default=2,
        help="Bounded shared-memory input/result slots per GPU.",
    )
    parser.add_argument(
        "--flush-workers", type=int, default=DEFAULT_FLUSH_WORKERS,
        help="Spawned OME-Zarr flush processes (default: min(CPU count, 64)); 0 uses the synchronous baseline.",
    )
    parser.add_argument(
        "--input-reader", choices=("tensorstore", "python-zarr"), default=DEFAULT_INPUT_READER,
        help="Inference tile reader backend (default: tensorstore).",
    )
    parser.add_argument(
        "--prefetch-tiles-per-gpu", type=int, default=DEFAULT_PREFETCH_TILES_PER_GPU,
        help="Bounded input read-ahead tiles per selected GPU (default: 4).",
    )
    parser.add_argument(
        "--input-cache-gib", type=float, default=DEFAULT_INPUT_CACHE_BYTES / float(1 << 30),
        help="TensorStore cache budget in GiB (default: 4).",
    )
    parser.add_argument(
        "--input-io-threads", type=int, default=DEFAULT_INPUT_IO_THREADS,
        help="TensorStore file I/O concurrency (default: 16).",
    )
    parser.add_argument(
        "--input-copy-threads", type=int, default=DEFAULT_INPUT_COPY_THREADS,
        help="TensorStore decode/data-copy concurrency (default: 4).",
    )
    parser.add_argument(
        "--profile-pipeline", action="store_true",
        help="Print detailed loader, CPU preparation, CUDA, transfer, and coordinator stage timings.",
    )
    parser.add_argument(
        "--inference-precision", choices=("auto", "fp32", "fp16", "bf16"), default="auto",
        help="Model autocast precision; auto uses checkpoint training metadata (default: auto).",
    )
    parser.add_argument(
        "--product-accumulator-dtype", choices=("float16", "float32"), default="float16",
        help="Raw product ring dtype; float16 halves product backing (default: float16).",
    )
    parser.add_argument(
        "--accumulator-workers", type=int, default=DEFAULT_ACCUMULATOR_WORKERS,
        help="Spawned chunk accumulation processes (default: min(CPU count, 32)); 0 is synchronous.",
    )
    parser.add_argument(
        "--download-workers", type=int, default=64,
        help="Parallel S3 chunk download threads used by automatic download.",
    )
    parser.add_argument(
        "--base-ref",
        default=None,
        help="Reference zarr for base shape. With --base-scale N, base = ref_shape * 2^N.",
    )
    parser.add_argument("--base-scale", type=int, default=None, help="Downsample exponent for --base-ref.")
    parser.add_argument(
        "--no-download",
        action="store_true",
        default=False,
        help="Skip automatic S3 download from _download zarr metadata.",
    )
    parser.add_argument(
        "--levels",
        type=int,
        default=5,
        help="Number of OME-Zarr levels to create.",
    )
    parser.add_argument("--ome-chunk", type=int, default=64, help="Output OME-Zarr chunk size.")
    parser.add_argument(
        "--pyramid-workers",
        type=int,
        default=0,
        help="Workers for OME-Zarr pyramid construction. 0 uses the helper default.",
    )
    parser.add_argument(
        "--recurrent-steps",
        type=int,
        default=None,
        help="Conditioned recurrent inference steps; each step is stored as one option.",
    )
    args = parser.parse_args(argv)
    if int(args.download_workers) <= 0:
        parser.error("--download-workers must be a positive integer")
    if int(args.flush_workers) < 0:
        parser.error("--flush-workers must be >= 0")
    if int(args.accumulator_workers) < 0:
        parser.error("--accumulator-workers must be >= 0")
    if int(args.prefetch_tiles_per_gpu) <= 0:
        parser.error("--prefetch-tiles-per-gpu must be > 0")
    if not math.isfinite(float(args.input_cache_gib)) or float(args.input_cache_gib) < 0:
        parser.error("--input-cache-gib must be finite and >= 0")
    if int(args.input_io_threads) <= 0:
        parser.error("--input-io-threads must be > 0")
    if int(args.input_copy_threads) <= 0:
        parser.error("--input-copy-threads must be > 0")

    run_fiber_trace_3d_inference(
        config_path=args.config,
        input_path=str(args.input),
        output_path=args.output,
        checkpoint=args.checkpoint,
        device=args.device,
        devices=args.devices,
        crop_xyzwhd=tuple(int(v) for v in args.crop_xyzwhd) if args.crop_xyzwhd else None,
        tile_size=args.tile_size,
        overlap=int(args.overlap),
        border=int(args.border),
        inference_scaledown_power=int(args.inference_scaledown_power),
        base_ref=args.base_ref,
        base_scale=args.base_scale,
        no_download=bool(args.no_download),
        levels=int(args.levels),
        ome_chunk=int(args.ome_chunk),
        pyramid_workers=int(args.pyramid_workers),
        recurrent_steps=args.recurrent_steps,
        prefetch_workers=int(args.prefetch_workers),
        slots_per_gpu=int(args.slots_per_gpu),
        flush_workers=int(args.flush_workers),
        accumulator_workers=int(args.accumulator_workers),
        input_reader=str(args.input_reader),
        prefetch_tiles_per_gpu=int(args.prefetch_tiles_per_gpu),
        input_cache_gib=float(args.input_cache_gib),
        input_io_threads=int(args.input_io_threads),
        input_copy_threads=int(args.input_copy_threads),
        profile_pipeline=bool(args.profile_pipeline),
        inference_precision=str(args.inference_precision),
        product_accumulator_dtype=str(args.product_accumulator_dtype),
        download_workers=int(args.download_workers),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
