from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import torch
import zarr

try:
    from lasagna.tiled_predict3d import (
        _auto_download,
        build_product_omezarr_pyramids,
        create_product_omezarr_groups,
        _cleanup_predict3d_temp_files,
        _crop_xyzwhd_bounds,
        _ds_index,
        _ds_size,
        run_tiled_inference_3d,
        OmeZarrOutputAdapter,
        write_lasagna_product_manifest,
    )
except ImportError:  # pragma: no cover - supports PYTHONPATH=lasagna style runs.
    from tiled_predict3d import (
        _auto_download,
        build_product_omezarr_pyramids,
        create_product_omezarr_groups,
        _cleanup_predict3d_temp_files,
        _crop_xyzwhd_bounds,
        _ds_index,
        _ds_size,
        run_tiled_inference_3d,
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
    full_out_shape = (
        _ds_size(input_shape_zyx[0], sd),
        _ds_size(input_shape_zyx[1], sd),
        _ds_size(input_shape_zyx[2], sd),
    )

    oz0 = (_ds_index(z0, sd) // oc) * oc
    oy0 = (_ds_index(y0, sd) // oc) * oc
    ox0 = (_ds_index(x0, sd) // oc) * oc
    oz1 = min(
        full_out_shape[0],
        ((_ds_index(z0, sd) + _ds_size(nz, sd) + oc - 1) // oc) * oc,
    )
    oy1 = min(
        full_out_shape[1],
        ((_ds_index(y0, sd) + _ds_size(ny, sd) + oc - 1) // oc) * oc,
    )
    ox1 = min(
        full_out_shape[2],
        ((_ds_index(x0, sd) + _ds_size(nx, sd) + oc - 1) // oc) * oc,
    )

    z0 = max(0, min(z0, oz0 * sd))
    y0 = max(0, min(y0, oy0 * sd))
    x0 = max(0, min(x0, ox0 * sd))
    z1 = max(z1, min(input_shape_zyx[0], oz1 * sd))
    y1 = max(y1, min(input_shape_zyx[1], oy1 * sd))
    x1 = max(x1, min(input_shape_zyx[2], ox1 * sd))

    nz, ny, nx = z1 - z0, y1 - y0, x1 - x0
    oz0 = (_ds_index(z0, sd) // oc) * oc
    oy0 = (_ds_index(y0, sd) // oc) * oc
    ox0 = (_ds_index(x0, sd) // oc) * oc
    oz1 = min(
        full_out_shape[0],
        ((_ds_index(z0, sd) + _ds_size(nz, sd) + oc - 1) // oc) * oc,
    )
    oy1 = min(
        full_out_shape[1],
        ((_ds_index(y0, sd) + _ds_size(ny, sd) + oc - 1) // oc) * oc,
    )
    ox1 = min(
        full_out_shape[2],
        ((_ds_index(x0, sd) + _ds_size(nx, sd) + oc - 1) // oc) * oc,
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
    device: str | None = "auto",
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
) -> None:
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
        _auto_download(input_path, crop_xyzwhd)

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

    torch_device = _resolve_inference_device(device)

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
        f"device={torch_device}",
        flush=True,
    )

    t0 = time.time()
    print(
        f"[fiber_trace_3d:infer] loading checkpoint={checkpoint} device={torch_device}",
        flush=True,
    )
    model = predict_adapter.load_model(device=torch_device)
    print("[fiber_trace_3d:infer] model loaded; starting tiled inference", flush=True)
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
        default="auto",
        help='Device: "auto" selects CUDA when available, otherwise CPU.',
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

    run_fiber_trace_3d_inference(
        config_path=args.config,
        input_path=str(args.input),
        output_path=args.output,
        checkpoint=args.checkpoint,
        device=args.device,
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
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
