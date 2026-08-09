#!/usr/bin/env python3
"""Estimate a constant canvas offset between a label raster and its TIFXYZ.

Annotations are drawn on rendered surface volumes. When the published render
does not correspond pixel-for-pixel to the TIFXYZ canvas the transfer reads
(for example because either side was re-exported), every transferred label
is offset by a constant number of canvas pixels. Surface geometry cannot
reveal this: the 3D surfaces still coincide, point-to-surface distances stay
tiny, and — because the flattening's tangent frame rotates across the sheet —
no single 3D transform can repair it. The offset is constant in canvas
space, so it must be measured from image content and corrected in 2D.

This tool projects the source CT render onto the target canvas through the
TIFXYZ geometry, measures robust tile-wise phase-correlation shifts against
the native target render, verifies that their spatial field is consistent
with one translation, and re-projects until the residual vanishes. The
result is written as JSON whose value feeds
``transfer.py --label-canvas-offset`` so the correction is always explicit
and audited, never silently applied.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Optional, Sequence

import numpy as np
from scipy.ndimage import gaussian_filter

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tifxyz_label_transfer.core import (  # type: ignore
        Surface,
        load_affine,
        transfer_array,
    )
    from tifxyz_label_transfer.io import (  # type: ignore
        load_surface,
        read_image,
    )
else:
    from .core import Surface, load_affine, transfer_array
    from .io import load_surface, read_image


def _preprocess(image: np.ndarray, mode: str) -> np.ndarray:
    """Prepare a render for correlation without changing its coordinates."""

    values = np.asarray(image, dtype=np.float32)
    if mode == "none":
        return values
    if mode == "dog":
        return gaussian_filter(values, 1.0) - gaussian_filter(values, 8.0)
    if mode == "gradient":
        smooth = gaussian_filter(values, 1.0)
        dy, dx = np.gradient(smooth)
        return np.hypot(dy, dx)
    raise ValueError(f"unknown correlation preprocessing mode {mode!r}")


def _phase_correlate(
    reference: np.ndarray,
    moving: np.ndarray,
    max_shift_px: Optional[float] = None,
) -> tuple[tuple[float, float], float]:
    """Return ``s`` with ``reference(u + s) ~= moving(u)`` and the peak value.

    ``max_shift_px`` restricts the admissible peak; surface renders repeat
    structure at the winding spacing, and once the remaining shift is known
    to be small, the distant periodic peaks are spurious by construction.
    """

    a = reference.astype(np.float64)
    b = moving.astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    window = (
        np.hanning(a.shape[0])[:, None] * np.hanning(a.shape[1])[None, :]
    )
    spectrum = np.fft.rfft2(a * window) * np.conj(np.fft.rfft2(b * window))
    spectrum /= np.abs(spectrum) + 1e-12
    correlation = np.fft.irfft2(spectrum, s=a.shape)
    search = correlation
    if max_shift_px is not None:
        excluded = np.zeros(correlation.shape, dtype=bool)
        for axis, size in enumerate(correlation.shape):
            signed = np.arange(size, dtype=np.float64)
            signed[signed > size // 2] -= size
            out_of_range = np.abs(signed) > max_shift_px
            if axis == 0:
                excluded |= out_of_range[:, None]
            else:
                excluded |= out_of_range[None, :]
        search = np.where(excluded, -np.inf, correlation)
    peak_row, peak_col = np.unravel_index(
        int(np.argmax(search)), correlation.shape
    )
    shift = []
    for axis, index in enumerate((int(peak_row), int(peak_col))):
        size = correlation.shape[axis]
        if axis == 0:
            before = correlation[(index - 1) % size, peak_col]
            center = correlation[index, peak_col]
            after = correlation[(index + 1) % size, peak_col]
        else:
            before = correlation[peak_row, (index - 1) % size]
            center = correlation[peak_row, index]
            after = correlation[peak_row, (index + 1) % size]
        denominator = before - 2.0 * center + after
        subpixel = (
            0.5 * (before - after) / denominator
            if abs(denominator) > 1e-12
            else 0.0
        )
        subpixel = float(np.clip(subpixel, -0.5, 0.5))
        signed = index if index <= size // 2 else index - size
        shift.append(signed + subpixel)
    return (shift[0], shift[1]), float(correlation[peak_row, peak_col])


def _fit_shift_field(
    centers_yx: np.ndarray,
    shifts_yx: np.ndarray,
    peaks: np.ndarray,
    shape: tuple[int, int],
) -> dict:
    """Robustly fit translation plus linear drift to tile measurements."""

    height, width = shape
    design = np.column_stack(
        (
            np.ones(centers_yx.shape[0], dtype=np.float64),
            (centers_yx[:, 0] - height / 2.0) / max(float(height), 1.0),
            (centers_yx[:, 1] - width / 2.0) / max(float(width), 1.0),
        )
    )
    weights = np.maximum(peaks, 1e-6)
    weights /= max(float(np.median(weights)), 1e-6)
    coefficients = np.zeros((3, 2), dtype=np.float64)
    for _ in range(8):
        root_weights = np.sqrt(weights)[:, None]
        coefficients, _, _, _ = np.linalg.lstsq(
            design * root_weights,
            shifts_yx * root_weights,
            rcond=None,
        )
        residual = np.linalg.norm(
            shifts_yx - design @ coefficients, axis=1
        )
        scale = max(
            1.4826
            * float(np.median(np.abs(residual - np.median(residual)))),
            0.05,
        )
        weights = np.maximum(peaks, 1e-6) * np.minimum(
            1.0, (1.5 * scale) / np.maximum(residual, 1e-9)
        )
        weights /= max(float(np.median(weights)), 1e-6)

    residual = np.linalg.norm(shifts_yx - design @ coefficients, axis=1)
    residual_median = float(np.median(residual))
    residual_mad = float(
        np.median(np.abs(residual - residual_median))
    )
    inlier_limit = max(
        0.5, residual_median + 3.0 * 1.4826 * residual_mad
    )
    inliers = residual <= inlier_limit
    if int(inliers.sum()) >= 3 and not np.all(inliers):
        root_weights = np.sqrt(np.maximum(peaks[inliers], 1e-6))[:, None]
        coefficients, _, _, _ = np.linalg.lstsq(
            design[inliers] * root_weights,
            shifts_yx[inliers] * root_weights,
            rcond=None,
        )
        residual = np.linalg.norm(
            shifts_yx - design @ coefficients, axis=1
        )

    corners = np.asarray(
        [
            [1.0, -0.5, -0.5],
            [1.0, -0.5, 0.5],
            [1.0, 0.5, -0.5],
            [1.0, 0.5, 0.5],
        ]
    )
    corner_shifts = corners @ coefficients
    corner_drift = np.linalg.norm(
        corner_shifts - coefficients[0], axis=1
    )
    return {
        "center_shift_yx": coefficients[0].tolist(),
        "linear_coefficients_yx": {
            "per_normalized_y": coefficients[1].tolist(),
            "per_normalized_x": coefficients[2].tolist(),
        },
        "gradient_yx_per_canvas_px": [
            (coefficients[1] / max(float(height), 1.0)).tolist(),
            (coefficients[2] / max(float(width), 1.0)).tolist(),
        ],
        "max_corner_drift_px": float(np.max(corner_drift)),
        "residual_median_px": float(np.median(residual[inliers])),
        "residual_p95_px": float(np.percentile(residual[inliers], 95)),
        "inlier_tiles": int(inliers.sum()),
        "outlier_tiles": int((~inliers).sum()),
        "inlier_mask": inliers.tolist(),
    }


def measure_render_shift(
    reference: np.ndarray,
    moving: np.ndarray,
    tile_size: int = 512,
    min_coverage: float = 0.6,
    max_tiles: int = 32,
    min_peak: float = 0.03,
    max_shift_px: Optional[float] = None,
    preprocess: str = "dog",
) -> dict:
    """Measure the median translation between two renders of one canvas.

    The returned ``shift_yx`` satisfies ``reference(u + s) ~= moving(u)``:
    it is where the reference must be sampled to reproduce the moving image.
    Tiles that are mostly empty in either render (unmapped projection
    borders) or whose correlation peak is too weak are skipped.
    """

    if reference.shape != moving.shape:
        raise ValueError(
            "renders must share one canvas; got "
            f"{reference.shape} and {moving.shape}"
        )
    height, width = reference.shape
    tile = int(min(tile_size, height, width))
    if tile < 16:
        raise ValueError(f"render {reference.shape} is too small to measure")

    candidates = [
        (row, col)
        for row in range(0, height - tile + 1, tile)
        for col in range(0, width - tile + 1, tile)
    ]
    considered = len(candidates)
    eligible: list[tuple[float, int, int]] = []
    for row, col in candidates:
        reference_tile = reference[row : row + tile, col : col + tile]
        moving_tile = moving[row : row + tile, col : col + tile]
        if (
            float((reference_tile > 0).mean()) < min_coverage
            or float((moving_tile > 0).mean()) < min_coverage
        ):
            continue
        texture = min(
            float(np.std(_preprocess(reference_tile, preprocess))),
            float(np.std(_preprocess(moving_tile, preprocess))),
        )
        if texture > 1e-6:
            eligible.append((texture, row, col))
    eligible.sort(key=lambda item: (-item[0], item[1], item[2]))
    candidates = sorted(
        (row, col) for _, row, col in eligible[:max_tiles]
    )

    tile_shifts: list[tuple[float, float]] = []
    tile_peaks: list[float] = []
    tile_centers: list[tuple[float, float]] = []
    for row, col in candidates:
        reference_tile = _preprocess(
            reference[row : row + tile, col : col + tile], preprocess
        )
        moving_tile = _preprocess(
            moving[row : row + tile, col : col + tile], preprocess
        )
        shift, peak = _phase_correlate(
            reference_tile, moving_tile, max_shift_px=max_shift_px
        )
        if peak < min_peak:
            continue
        tile_shifts.append(shift)
        tile_peaks.append(peak)
        tile_centers.append((row + tile / 2.0, col + tile / 2.0))
    if not tile_shifts:
        raise ValueError(
            "no usable tiles: the renders share too little textured overlap"
        )
    shifts = np.asarray(tile_shifts, dtype=np.float64)
    centers = np.asarray(tile_centers, dtype=np.float64)
    peaks = np.asarray(tile_peaks, dtype=np.float64)
    field = _fit_shift_field(centers, shifts, peaks, reference.shape)
    inliers = np.asarray(field.pop("inlier_mask"), dtype=bool)
    robust_shifts = shifts[inliers]
    median = np.median(robust_shifts, axis=0)
    return {
        "shift_yx": [float(median[0]), float(median[1])],
        "shift_scatter_yx": [
            float(np.median(np.abs(robust_shifts[:, 0] - median[0]))),
            float(np.median(np.abs(robust_shifts[:, 1] - median[1]))),
        ],
        "tiles_used": len(tile_shifts),
        "tiles_considered": considered,
        "tiles_eligible": len(candidates),
        "tile_px": tile,
        "peak_median": float(np.median(peaks)),
        "preprocess": preprocess,
        "shift_field": field,
        "tiles": [
            {
                "center_yx": centers[index].tolist(),
                "shift_yx": shifts[index].tolist(),
                "peak": float(peaks[index]),
                "inlier": bool(inliers[index]),
            }
            for index in range(shifts.shape[0])
        ],
    }


def _project(
    source: Surface,
    target: Surface,
    source_render: np.ndarray,
    target_shape: tuple[int, int],
    offset_yx: tuple[float, float],
    *,
    nearest_vertices: int,
    tile_size: int,
    query_batch_size: int,
    affine: Optional[np.ndarray],
) -> np.ndarray:
    projected, _, _, _ = transfer_array(
        source,
        target,
        source_render,
        output_shape=target_shape,
        affine=affine,
        label_offset_yx=offset_yx,
        nearest_vertices=nearest_vertices,
        tile_size=tile_size,
        query_batch_size=query_batch_size,
    )
    return projected


def estimate_canvas_offset(
    source: Surface,
    target: Surface,
    source_render: np.ndarray,
    target_render: np.ndarray,
    *,
    max_iterations: int = 4,
    tolerance_px: float = 0.25,
    measure_tile_px: int = 512,
    min_coverage: float = 0.6,
    max_tiles: int = 32,
    nearest_vertices: int = 8,
    tile_size: int = 256,
    query_batch_size: int = 65_536,
    affine: Optional[np.ndarray] = None,
    preprocess: str = "dog",
    max_corner_drift_px: float = 1.5,
    verbose: bool = False,
) -> dict:
    """Iteratively estimate the source render-to-TIFXYZ canvas offset.

    The offset ``(dy, dx)`` means source render/label pixel ``(i, j)``
    depicts source TIFXYZ canvas position ``(i + dy, j + dx)`` in the
    render's own pixel units — the exact semantics of
    ``transfer_array(label_offset_yx=...)``.

    With current estimate ``o`` and true offset ``o*``, the projection
    samples the render at canvas position minus ``o``, so the projected
    image appears displaced by ``o* - o`` relative to the native target
    render; the measured shift is therefore added directly. The residual is
    re-measured after every update, so a wrong sign or scale cannot survive
    silently. After the first correction the remaining shift is known to be
    small, so later measurements restrict the correlation peak search and
    cannot lock onto the neighbouring-winding periodicity.
    """

    offset = np.zeros(2, dtype=np.float64)
    iterations: list[dict] = []
    best_offset = offset.copy()
    best_residual = math.inf
    converged = False
    max_shift: Optional[float] = None

    for iteration in range(max_iterations + 1):
        projected = _project(
            source,
            target,
            source_render,
            target_render.shape,
            (float(offset[0]), float(offset[1])),
            nearest_vertices=nearest_vertices,
            tile_size=tile_size,
            query_batch_size=query_batch_size,
            affine=affine,
        )
        measurement = measure_render_shift(
            target_render,
            projected,
            tile_size=measure_tile_px,
            min_coverage=min_coverage,
            max_tiles=max_tiles,
            max_shift_px=max_shift,
            preprocess=preprocess,
        )
        shift_y, shift_x = measurement["shift_yx"]
        residual = math.hypot(shift_y, shift_x)
        iterations.append(
            {
                "iteration": iteration,
                "offset_yx_render_px": offset.tolist(),
                "measured_shift_yx_render_px": [shift_y, shift_x],
                "residual_render_px": residual,
                "tiles_used": measurement["tiles_used"],
                "shift_scatter_yx": measurement["shift_scatter_yx"],
                "shift_field": measurement["shift_field"],
                "peak_median": measurement["peak_median"],
                "preprocess": measurement["preprocess"],
                "max_shift_px": max_shift,
            }
        )
        if verbose:
            print(
                f"iteration {iteration}: shift=({shift_y:+.2f}, "
                f"{shift_x:+.2f}) render px, residual={residual:.3f}, "
                f"tiles={measurement['tiles_used']}, "
                f"scatter=({measurement['shift_scatter_yx'][0]:.2f}, "
                f"{measurement['shift_scatter_yx'][1]:.2f})"
            )
        if residual < best_residual:
            best_residual = residual
            best_offset = offset.copy()
        elif residual > best_residual + tolerance_px:
            if verbose:
                print("residual regressed; keeping best offset")
            break
        if residual <= tolerance_px:
            converged = True
            break
        if iteration == max_iterations:
            break
        offset = offset + np.asarray([shift_y, shift_x])
        max_shift = max(3.0, 1.5 * residual)

    initial_field = iterations[0]["shift_field"]
    translation_model_valid = (
        initial_field["max_corner_drift_px"] <= max_corner_drift_px
    )
    return {
        "offset_yx_render_px": best_offset.tolist(),
        "residual_render_px": best_residual,
        "converged": converged,
        "translation_model_valid": translation_model_valid,
        "max_corner_drift_px": max_corner_drift_px,
        "tolerance_px": tolerance_px,
        "iterations": iterations,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure the constant canvas offset between the raster a source "
            "TIFXYZ's renders/labels use and the TIFXYZ canvas itself, by "
            "projecting a source CT render onto a target TIFXYZ canvas of "
            "the same sheet and comparing with the native target render. "
            "The result feeds 'transfer.py --label-canvas-offset'."
        )
    )
    parser.add_argument("--source-tifxyz", required=True)
    parser.add_argument("--target-tifxyz", required=True)
    parser.add_argument(
        "--source-render",
        required=True,
        help="CT render on the source canvas (any uniform scale)",
    )
    parser.add_argument(
        "--target-render",
        required=True,
        help="CT render on the target canvas (same pyramid level)",
    )
    parser.add_argument("--output", required=True, help="offset JSON path")
    parser.add_argument("--max-iterations", type=int, default=4)
    parser.add_argument(
        "--tolerance-px",
        type=float,
        default=0.25,
        help="stop once the measured residual falls below this (render px)",
    )
    parser.add_argument("--measure-tile-px", type=int, default=512)
    parser.add_argument("--min-coverage", type=float, default=0.6)
    parser.add_argument("--max-tiles", type=int, default=32)
    parser.add_argument("--nearest-vertices", type=int, default=8)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--query-batch-size", type=int, default=65_536)
    parser.add_argument(
        "--preprocess",
        choices=("dog", "gradient", "none"),
        default="dog",
        help="coordinate-preserving image preprocessing for correlation",
    )
    parser.add_argument(
        "--stage-one-affine",
        "--same-volume-affine",
        dest="same_volume_affine",
        help=(
            "registration JSON from source to target volume; this tool "
            "never estimates a volume affine"
        ),
    )
    parser.add_argument(
        "--stage-one-affine-direction",
        "--same-volume-affine-direction",
        dest="same_volume_affine_direction",
        choices=("forward", "inverse"),
        default="forward",
    )
    parser.add_argument(
        "--max-corner-drift-px",
        type=float,
        default=1.5,
        help=(
            "reject the constant-offset model when the fitted initial tile "
            "field varies by more than this many render pixels"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    output_path = Path(args.output)
    if output_path.exists() and not args.overwrite:
        raise SystemExit(
            f"error: {output_path} exists (pass --overwrite to replace)"
        )

    print(f"Loading source TIFXYZ: {args.source_tifxyz}")
    source = load_surface(args.source_tifxyz)
    print(f"Loading target TIFXYZ: {args.target_tifxyz}")
    target = load_surface(args.target_tifxyz)
    source_render = read_image(args.source_render)
    target_render = read_image(args.target_render)
    affine = None
    if args.same_volume_affine:
        affine = load_affine(args.same_volume_affine)
        if args.same_volume_affine_direction == "inverse":
            affine = np.linalg.inv(affine)

    result = estimate_canvas_offset(
        source,
        target,
        source_render,
        target_render,
        max_iterations=args.max_iterations,
        tolerance_px=args.tolerance_px,
        measure_tile_px=args.measure_tile_px,
        min_coverage=args.min_coverage,
        max_tiles=args.max_tiles,
        nearest_vertices=args.nearest_vertices,
        tile_size=args.tile_size,
        query_batch_size=args.query_batch_size,
        affine=affine,
        preprocess=args.preprocess,
        max_corner_drift_px=args.max_corner_drift_px,
        verbose=True,
    )
    render_height, render_width = source_render.shape
    full_height, full_width = source.full_resolution_shape
    offset_render = result["offset_yx_render_px"]
    offset_full = [
        offset_render[0] * full_height / render_height,
        offset_render[1] * full_width / render_width,
    ]
    document = {
        "canvas_offset_yx_render_px": offset_render,
        "canvas_offset_yx_full_resolution_px": offset_full,
        "convention": (
            "source render/label pixel (i, j) depicts source TIFXYZ canvas "
            "position (i + dy, j + dx) in that raster's own pixel units; "
            "pass the full-resolution value to transfer.py "
            "--label-canvas-offset"
        ),
        "estimator": {
            "tool": "tifxyz_label_transfer/estimate_canvas_offset.py",
            "source_tifxyz": str(args.source_tifxyz),
            "target_tifxyz": str(args.target_tifxyz),
            "source_render": str(args.source_render),
            "target_render": str(args.target_render),
            "preprocess": args.preprocess,
            "same_volume_affine": (
                str(args.same_volume_affine)
                if args.same_volume_affine
                else None
            ),
            "same_volume_affine_direction": (
                args.same_volume_affine_direction
                if args.same_volume_affine
                else None
            ),
            "source_render_shape": [render_height, render_width],
            "source_full_resolution_shape": [full_height, full_width],
            **{
                key: result[key]
                for key in (
                    "residual_render_px",
                    "converged",
                    "translation_model_valid",
                    "max_corner_drift_px",
                    "tolerance_px",
                    "iterations",
                )
            },
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(document, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Canvas offset (dy, dx): ({offset_render[0]:+.3f}, "
        f"{offset_render[1]:+.3f}) render px = ({offset_full[0]:+.2f}, "
        f"{offset_full[1]:+.2f}) full-resolution px; residual "
        f"{result['residual_render_px']:.3f} render px "
        f"({'converged' if result['converged'] else 'NOT converged'})"
    )
    print(f"Wrote {output_path}")
    if not result["converged"] or not result["translation_model_valid"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
