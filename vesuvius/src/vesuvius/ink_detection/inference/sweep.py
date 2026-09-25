"""Sweep flat ink inference over checkpoints, depth windows and directions.

The flat ink models respond differently to where the papyrus sits in their
depth window, to which side of the surface volume faces the ink, and from one
checkpoint to the next. This command runs every requested combination on one
surface volume, loading and compiling each checkpoint once, and writes the
predictions, a JSON summary and a contact sheet for side-by-side review.

Every prediction goes through ``infer_single_zarr``, so each TIFF is identical
to running ``vesuvius.ink_detection.inference.infer`` with the matching
``--layer-start``/``--layer-end``/``--direction``.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw
import tifffile
import torch

from vesuvius.ink_detection.inference.infer import (
    _volume_axes,
    add_flat_pass_arguments,
    configure_model,
    infer_single_zarr,
    load_grayscale_mask,
    open_flat_input,
    resolve_run_directions,
    select_layer_indices,
    validate_flat_pass_arguments,
)
from vesuvius.ink_detection.inference.inference_runtime import (
    prepare_model_for_inference,
)
from vesuvius.ink_detection.volume_io import open_volume
from vesuvius.utils.cli import HyphenUnderscoreParser

LOGGER = logging.getLogger(__name__)
DEFAULT_OFFSETS = (-2, 0, 2)
SUMMARY_NAME = "sweep.json"
SHEET_NAME = "sweep.png"


@dataclass(frozen=True)
class DepthWindow:
    offset: int
    layer_start: int
    layer_end: int


def plan_depth_windows(
    depth: int, input_depth: int, offsets: Sequence[int]
) -> tuple[list[DepthWindow], list[int]]:
    """Place model-depth windows ``offsets`` slices away from the default one.

    Offset 0 is the centered window ``infer`` uses when no layer range is
    given. Windows that would leave the volume are returned as skipped offsets
    rather than clamped, because ``infer`` would silently recenter them.
    """

    depth = int(depth)
    input_depth = int(input_depth)
    base = depth // 2 - input_depth // 2
    windows: list[DepthWindow] = []
    skipped: list[int] = []
    for offset in sorted({int(value) for value in offsets}):
        if depth <= input_depth:
            if offset == 0:
                windows.append(DepthWindow(0, 0, depth))
            else:
                skipped.append(offset)
            continue
        start = base + offset
        if start < 0 or start + input_depth > depth:
            skipped.append(offset)
        else:
            windows.append(DepthWindow(offset, start, start + input_depth))
    return windows, skipped


def estimate_depth_profile(
    volume: Any, *, tile: int = 128, grid: int = 8
) -> list[float | None]:
    """Mean intensity per depth slice over the occupied pixels of a tile grid.

    Only ``grid`` x ``grid`` tiles of ``tile`` pixels are read, so this stays
    cheap for remote volumes.
    """

    depth_first, depth, height, width, _, _ = _volume_axes(volume)
    total = np.zeros(depth, dtype=np.float64)
    count = 0
    ys = sorted({int(v) for v in np.linspace(0, max(0, height - tile), grid)})
    xs = sorted({int(v) for v in np.linspace(0, max(0, width - tile), grid)})
    for y in ys:
        for x in xs:
            if depth_first:
                block = np.asarray(volume[:, y : y + tile, x : x + tile])
            else:
                block = np.moveaxis(
                    np.asarray(volume[y : y + tile, x : x + tile, :]), -1, 0
                )
            occupied = block.max(axis=0) > 0
            if occupied.any():
                total += block[:, occupied].sum(axis=1, dtype=np.float64)
                count += int(occupied.sum())
    if count == 0:
        return [None] * depth
    return [float(value) for value in total / count]


def roc_auc_uint8(
    prediction: np.ndarray, ink: np.ndarray, region: np.ndarray
) -> float | None:
    """Exact ROC AUC of uint8 scores inside ``region``; ties count one half."""

    scores = np.asarray(prediction)
    if scores.dtype != np.uint8:
        raise ValueError(f"Expected uint8 predictions, got {scores.dtype}")
    positive = np.bincount(scores[region & ink], minlength=256).astype(np.float64)
    negative = np.bincount(scores[region & ~ink], minlength=256).astype(np.float64)
    n_positive, n_negative = positive.sum(), negative.sum()
    if n_positive == 0 or n_negative == 0:
        return None
    below = np.cumsum(negative) - negative
    return float(
        (positive * (below + 0.5 * negative)).sum() / (n_positive * n_negative)
    )


def load_label_plane(path: Path, shape: tuple[int, int]) -> np.ndarray:
    """Read a label or mask as a nonzero plane matching ``shape``.

    Images use the flat mask reader and its top-left alignment. Zarr inputs
    are read at level 0, a 3D label array is collapsed with ``any`` over its
    shortest axis, and the result must match ``shape`` exactly.
    """

    path = Path(path)
    if path.suffix.lower() != ".zarr":
        return load_grayscale_mask(path, shape)
    image = np.squeeze(np.asarray(open_volume(path, "0")[...]))
    if image.ndim == 3:
        image = (image != 0).any(axis=int(np.argmin(image.shape)))
    if tuple(image.shape) != tuple(shape):
        raise ValueError(
            f"{path} has shape {tuple(image.shape)!r}, expected {tuple(shape)!r}"
        )
    return image != 0


def checkpoint_labels(checkpoints: Sequence[Path]) -> list[str]:
    """Short unique names: the file stem, prefixed by its folder on collision."""

    stems = [Path(path).stem for path in checkpoints]
    labels = [
        f"{Path(path).parent.name}_{stem}" if stems.count(stem) > 1 else stem
        for path, stem in zip(checkpoints, stems)
    ]
    seen: dict[str, int] = {}
    unique = []
    for label in labels:
        seen[label] = seen.get(label, 0) + 1
        unique.append(label if seen[label] == 1 else f"{label}_{seen[label]}")
    return unique


def _thumbnail(path: Path, max_px: int) -> np.ndarray:
    image = tifffile.imread(path).astype(np.float32)
    factor = max(1, math.ceil(max(image.shape) / max_px))
    image = np.pad(
        image, ((0, -image.shape[0] % factor), (0, -image.shape[1] % factor))
    )
    height, width = image.shape[0] // factor, image.shape[1] // factor
    return image.reshape(height, factor, width, factor).mean(axis=(1, 3))


def _draw_profile(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    profile: Sequence[float | None],
    windows: Sequence[DepthWindow],
) -> None:
    """Plot the depth profile with one bar per window, labelled to its left."""

    left, top, right, bottom = box
    values = [value for value in profile if value is not None]
    draw.rectangle(box, outline=90)
    if not values:
        return
    low, high = min(values), max(values)
    span = max(high - low, 1e-6)
    step = (right - left) / max(1, len(profile) - 1)
    curve_bottom = bottom - 10 - 10 * len(windows)
    for index, window in enumerate(windows):
        y = curve_bottom + 12 + 10 * index
        draw.line(
            (
                left + window.layer_start * step,
                y,
                left + (window.layer_end - 1) * step,
                y,
            ),
            fill=130,
            width=4,
        )
        draw.text((left - 40, y - 6), f"z{window.offset:+d}", fill=200)
    points = [
        (
            left + index * step,
            curve_bottom - (value - low) / span * (curve_bottom - top - 22),
        )
        for index, value in enumerate(profile)
        if value is not None
    ]
    if len(points) > 1:
        draw.line(points, fill=255, width=2)
    peak = int(np.nanargmax([np.nan if v is None else v for v in profile]))
    draw.text(
        (left + 4, top + 4),
        f"mean intensity by slice (0-{len(profile) - 1}), peak at slice {peak}; "
        "bars show each depth window",
        fill=200,
    )


def write_contact_sheet(
    path: Path,
    *,
    title: str,
    profile: Sequence[float | None],
    windows: Sequence[DepthWindow],
    runs: Sequence[dict[str, Any]],
    output_dir: Path,
    max_px: int,
) -> None:
    """One row per checkpoint and direction, one column per depth offset.

    All tiles share one display range (the 1st to 99th percentile of every
    nonzero prediction pixel), so brightness is comparable across the sheet.
    """

    offsets = sorted({int(run["offset"]) for run in runs})
    rows: dict[tuple[str, str], dict[int, dict[str, Any]]] = {}
    for run in runs:
        rows.setdefault((run["checkpoint"], run["direction"]), {})[
            int(run["offset"])
        ] = run
    thumbs = {
        run["output"]: _thumbnail(output_dir / run["output"], max_px) for run in runs
    }
    pixels = np.concatenate(
        [thumb[thumb > 0] for thumb in thumbs.values()] or [np.zeros(1)]
    )
    low, high = np.percentile(pixels, [1, 99]) if pixels.size > 1 else (0.0, 255.0)
    scale = 255.0 / max(float(high - low), 1e-6)

    label_width, caption, header = 260, 16, 22
    profile_height = 110 + 10 * len(windows)
    cell = max_px + 8
    width = label_width + cell * max(1, len(offsets))
    height = header + profile_height + header + len(rows) * (cell + caption)
    sheet = Image.new("L", (width, height), 25)
    draw = ImageDraw.Draw(sheet)
    draw.text((6, 5), title, fill=255)
    _draw_profile(
        draw,
        (label_width, header, width - 8, header + profile_height - 8),
        profile,
        windows,
    )
    top = header + profile_height
    for column, offset in enumerate(offsets):
        draw.text(
            (label_width + column * cell + 4, top + 5),
            f"depth offset z{offset:+d}",
            fill=255,
        )
    top += header
    for row, ((checkpoint, direction), by_offset) in enumerate(rows.items()):
        y = top + row * (cell + caption)
        draw.text((6, y + 4), checkpoint, fill=255)
        draw.text((6, y + 20), direction, fill=200)
        for column, offset in enumerate(offsets):
            run = by_offset.get(offset)
            if run is None:
                continue
            thumb = thumbs[run["output"]]
            tile = np.where(thumb > 0, np.clip((thumb - low) * scale, 0, 255), 0)
            image = Image.fromarray(tile.astype(np.uint8))
            x = label_width + column * cell
            sheet.paste(
                image, (x + (cell - image.width) // 2, y + (cell - image.height) // 2)
            )
            if run.get("roc_auc") is not None:
                draw.text((x + 4, y + cell + 2), f"AUC {run['roc_auc']:.3f}", fill=255)
    sheet.save(path)


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    """Run every checkpoint, depth window and direction; return the summary."""

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _, volume, level = open_flat_input(args.input_zarr, args.resolution)
    _, depth, height, width, _, _ = _volume_axes(volume)
    profile = estimate_depth_profile(volume)
    ink = region = None
    if args.ink_labels is not None:
        ink = load_label_plane(args.ink_labels, (height, width))
        region = load_label_plane(args.eval_mask, (height, width))
        LOGGER.info(
            "Scoring inside %s: %d pixels, %.2f%% labelled ink",
            args.eval_mask,
            int(region.sum()),
            100.0 * float(ink[region].mean()) if region.any() else 0.0,
        )
    directions = resolve_run_directions(args.direction)
    runs: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    windows: list[DepthWindow] = []
    for checkpoint, name in zip(args.checkpoints, checkpoint_labels(args.checkpoints)):
        run_args = argparse.Namespace(**vars(args))
        run_args.checkpoint = Path(checkpoint)
        setup_started = time.perf_counter()
        configured = configure_model(run_args)
        model, device = prepare_model_for_inference(
            configured.model,
            gpu_ids=args.gpu_ids,
            compile_model=args.compile_model,
            compile_mode=args.compile_mode,
        )
        configured = replace(configured, model=model)
        windows, skipped = plan_depth_windows(
            depth, configured.input_depth, args.offsets
        )
        if skipped:
            LOGGER.warning(
                "Skipping depth offsets %s for %s: their %d-slice window leaves "
                "the %d-slice volume",
                skipped,
                name,
                configured.input_depth,
                depth,
            )
        checkpoints.append(
            {
                "checkpoint": str(checkpoint),
                "label": name,
                "input_depth": int(configured.input_depth),
                "skipped_offsets": skipped,
                "setup_seconds": round(time.perf_counter() - setup_started, 3),
            }
        )
        for window in windows:
            for direction in directions:
                run_args.layer_start = window.layer_start
                run_args.layer_end = window.layer_end
                output = output_dir / f"{name}_z{window.offset:+d}_{direction}.tif"
                started = time.perf_counter()
                infer_single_zarr(
                    args=run_args,
                    input_zarr=args.input_zarr,
                    configured_model=configured,
                    device=device,
                    output_tiff=output,
                    layer_direction=direction,
                )
                record: dict[str, Any] = {
                    "checkpoint": name,
                    "offset": window.offset,
                    "direction": direction,
                    "layer_indices": select_layer_indices(
                        depth,
                        layer_start=window.layer_start,
                        layer_end=window.layer_end,
                        output_depth=configured.input_depth,
                        direction=direction,
                    ).tolist(),
                    "output": output.name,
                    "seconds": round(time.perf_counter() - started, 3),
                }
                if ink is not None:
                    record["roc_auc"] = roc_auc_uint8(
                        tifffile.imread(output), ink, region
                    )
                runs.append(record)
                LOGGER.info(
                    "Sweep run %d: %s z%+d %s in %.1fs%s",
                    len(runs),
                    name,
                    window.offset,
                    direction,
                    record["seconds"],
                    (
                        ""
                        if record.get("roc_auc") is None
                        else f" AUC={record['roc_auc']:.4f}"
                    ),
                )
        del model, configured
        torch.compiler.reset()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = {
        "input": str(args.input_zarr),
        "level": level,
        "shape": [int(depth), int(height), int(width)],
        "depth_profile": profile,
        "ink_labels": None if args.ink_labels is None else str(args.ink_labels),
        "eval_mask": None if args.eval_mask is None else str(args.eval_mask),
        "settings": {
            key: getattr(args, key)
            for key in (
                "overlap",
                "stride",
                "blend_mode",
                "batch_size",
                "amp_dtype",
                "tta_mirror",
                "compile_model",
                "compile_mode",
            )
        }
        | {"mask_path": None if args.mask_path is None else str(args.mask_path)},
        "checkpoints": checkpoints,
        "runs": runs,
    }
    (output_dir / SUMMARY_NAME).write_text(json.dumps(summary, indent=2) + "\n")
    if not runs:
        LOGGER.warning("No depth offset fits this volume; wrote only %s", SUMMARY_NAME)
        return summary
    write_contact_sheet(
        output_dir / SHEET_NAME,
        title=f"{args.input_zarr} (level {level}, {depth} slices)",
        profile=profile,
        windows=windows,
        runs=runs,
        output_dir=output_dir,
        max_px=args.sheet_tile_px,
    )
    LOGGER.info(
        "Wrote %d predictions, %s and %s to %s",
        len(runs),
        SUMMARY_NAME,
        SHEET_NAME,
        output_dir,
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = HyphenUnderscoreParser(
        description=(
            "Run flat ink inference for every checkpoint, depth offset and "
            "direction on one surface volume, and write a contact sheet"
        )
    )
    parser.add_argument("input_zarr")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--offsets",
        nargs="+",
        type=int,
        default=list(DEFAULT_OFFSETS),
        help=(
            "Depth offsets in slices from the default centered window "
            "(offset 0 is what infer uses without --layer-start/--layer-end). "
            "Default: -2 0 2."
        ),
    )
    parser.add_argument(
        "--direction", choices=("forward", "reverse", "both"), default="both"
    )
    parser.add_argument(
        "--ink-labels",
        type=Path,
        help="Ink label image or Zarr; with --eval-mask, adds ROC AUC per run.",
    )
    parser.add_argument(
        "--eval-mask",
        type=Path,
        help="Region to score, e.g. a validation or supervision mask.",
    )
    parser.add_argument("--sheet-tile-px", type=int, default=320)
    add_flat_pass_arguments(parser)
    args = parser.parse_args(argv)
    validate_flat_pass_arguments(parser, args)
    if (args.ink_labels is None) != (args.eval_mask is None):
        parser.error("--ink-labels and --eval-mask must be given together")
    if args.sheet_tile_px <= 0:
        parser.error("--sheet-tile-px must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
    )
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    run_sweep(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
