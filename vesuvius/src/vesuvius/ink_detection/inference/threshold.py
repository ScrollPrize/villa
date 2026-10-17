"""Choose the threshold for binarizing a flat ink prediction of an unlabelled scroll.

``calibrate`` takes predictions that the same checkpoint made on labelled segments and finds the
F1-optimal threshold of each. It stores two things: the median of those thresholds (the ``value``
rule) and the median fraction of the sheet (non-zero prediction pixels) that each threshold marks
(the ``quantile`` rule). With segments from two or more scrolls it also scores each scroll with
both rules taken from the other scrolls only, next to the score at 128, so you can see what each
choice costs for this checkpoint before reading a scroll that has no labels. ``apply`` binarizes a
prediction with either rule: ``value`` cuts at the stored threshold, ``quantile`` cuts where this
prediction marks the stored fraction of its sheet.

Calibrate on segments of scrolls the checkpoint never saw: that is the case this was measured
on, and a threshold taken from pixels the checkpoint trained on is not expected to carry over.
Re-calibrate after any further training.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import tifffile

from vesuvius.utils.cli import HyphenUnderscoreParser


DEFAULT_THRESHOLD_U8 = 128
RULES = ("value", "quantile")
SUPPORTED_IMAGE_SUFFIXES = {".tif", ".tiff", ".png"}


@dataclass(frozen=True)
class CalibrationCell:
    scroll: str
    prediction: str
    f1_curve: np.ndarray
    sheet_histogram: np.ndarray

    @property
    def best_threshold(self) -> int:
        return int(np.argmax(self.f1_curve))

    @property
    def sheet_fraction_at_best(self) -> float:
        return float(fraction_at_or_above(self.sheet_histogram)[self.best_threshold])


def read_prediction(path: Path) -> np.ndarray:
    """A 2-D uint8 prediction as written by ``inference.infer``."""
    with tifffile.TiffFile(path) as tif:
        prediction = np.squeeze(tif.pages[0].asarray())
    if prediction.ndim != 2:
        raise ValueError(f"Expected a 2D prediction in {path}, got shape={prediction.shape}")
    if prediction.dtype != np.uint8:
        raise ValueError(f"Expected a uint8 prediction in {path}, got dtype={prediction.dtype}")
    return prediction


def read_mask(path: Path) -> np.ndarray:
    """A boolean plane from a label OME-Zarr (array ``0``, middle Z plane) or a label image."""
    if path.suffix.lower() == ".zarr":
        import zarr

        node = zarr.open(str(path), mode="r")
        array = node["0"] if hasattr(node, "array_keys") else node
        plane = np.asarray(array[array.shape[0] // 2] if array.ndim == 3 else array[...])
    elif path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES:
        if path.suffix.lower() == ".png":
            from PIL import Image

            Image.MAX_IMAGE_PIXELS = None
            with Image.open(path) as image:
                plane = np.asarray(image)
        else:
            plane = tifffile.imread(path)
        if plane.ndim == 3:
            plane = plane[..., :3].max(axis=-1)
    else:
        raise ValueError(f"Unsupported label format: {path}")
    if plane.ndim != 2:
        raise ValueError(f"Expected a 2D label plane in {path}, got shape={plane.shape}")
    return plane != 0


def f1_curve(prediction: np.ndarray, ink: np.ndarray, scored: np.ndarray) -> np.ndarray:
    """F1 of ``prediction >= t`` against ``ink`` inside ``scored``, for every t in 0..255."""
    positives = np.bincount(prediction[ink & scored], minlength=256)
    negatives = np.bincount(prediction[~ink & scored], minlength=256)
    true_pos = positives[::-1].cumsum()[::-1].astype(np.float64)
    false_pos = negatives[::-1].cumsum()[::-1].astype(np.float64)
    total_ink = float(positives.sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(true_pos + false_pos > 0, true_pos / (true_pos + false_pos), 0.0)
        recall = true_pos / max(total_ink, 1.0)
        return np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)


def sheet_histogram(prediction: np.ndarray) -> np.ndarray:
    """Histogram of the sheet: every pixel the prediction covers (``prediction > 0``)."""
    return np.bincount(prediction[prediction > 0], minlength=256)


def fraction_at_or_above(histogram: np.ndarray) -> np.ndarray:
    """Fraction of the histogram's pixels at or above each threshold t in 0..255."""
    return histogram[::-1].cumsum()[::-1] / max(float(histogram.sum()), 1.0)


def median_threshold(cells: Sequence[CalibrationCell]) -> int:
    """Median of the per-cell optima, rounded half up."""
    return int(np.floor(np.median([cell.best_threshold for cell in cells]) + 0.5))


def median_fraction(cells: Sequence[CalibrationCell]) -> float:
    """Median of the sheet fraction each cell's optimum marks."""
    return float(np.median([cell.sheet_fraction_at_best for cell in cells]))


def quantile_threshold(histogram: np.ndarray, fraction: float) -> int:
    """Lowest threshold at which at most ``fraction`` of the sheet is marked."""
    marked = fraction_at_or_above(histogram)
    return int(np.argmax(marked <= fraction)) if (marked <= fraction).any() else 255


def score_cell(scroll: str, prediction_path: Path, ink_path: Path, mask_path: Path | None) -> CalibrationCell:
    prediction = read_prediction(prediction_path)
    ink = read_mask(ink_path)
    scored = read_mask(mask_path) if mask_path is not None else np.ones(prediction.shape, bool)
    for name, plane in ((ink_path, ink), (mask_path, scored)):
        if plane.shape != prediction.shape:
            raise ValueError(f"{name} has shape {plane.shape}, the prediction has {prediction.shape}")
    if not (ink & scored).any():
        raise ValueError(f"{prediction_path}: no ink pixels inside the scored mask")
    return CalibrationCell(scroll=scroll, prediction=str(prediction_path),
                           f1_curve=f1_curve(prediction, ink, scored), sheet_histogram=sheet_histogram(prediction))


def borrowing_check(cells: Sequence[CalibrationCell]) -> dict[str, dict[str, float | int]]:
    """Score each scroll with both rules taken from the other scrolls only, and at 128."""
    report: dict[str, dict[str, float | int]] = {}
    for scroll in sorted({cell.scroll for cell in cells}):
        own = [cell for cell in cells if cell.scroll == scroll]
        donors = [cell for cell in cells if cell.scroll != scroll]
        borrowed = median_threshold(donors)
        fraction = median_fraction(donors)

        def loss(cell: CalibrationCell, threshold: int) -> float:
            return float(cell.f1_curve.max() - cell.f1_curve[threshold])

        report[scroll] = {
            "cells": len(own),
            "threshold_from_other_scrolls": borrowed,
            "sheet_fraction_from_other_scrolls": fraction,
            "mean_f1_best": float(np.mean([cell.f1_curve.max() for cell in own])),
            "mean_f1_loss_value": float(np.mean([loss(cell, borrowed) for cell in own])),
            "mean_f1_loss_quantile": float(np.mean(
                [loss(cell, quantile_threshold(cell.sheet_histogram, fraction)) for cell in own])),
            "mean_f1_loss_128": float(np.mean([loss(cell, DEFAULT_THRESHOLD_U8) for cell in own])),
        }
    return report


def calibrate(cell_specs: Sequence[Sequence[str]], output_path: Path, *, checkpoint: str = "") -> dict:
    cells: list[CalibrationCell] = []
    for spec in cell_specs:
        if len(spec) not in (3, 4):
            raise ValueError("--cell takes SCROLL PREDICTION INKLABELS [MASK]")
        cell = score_cell(spec[0], Path(spec[1]), Path(spec[2]), Path(spec[3]) if len(spec) == 4 else None)
        cells.append(cell)
        print(
            f"{cell.scroll}  {Path(cell.prediction).name}: best threshold {cell.best_threshold} "
            f"(F1 {cell.f1_curve.max():.3f}; at {DEFAULT_THRESHOLD_U8}: {cell.f1_curve[DEFAULT_THRESHOLD_U8]:.3f}; "
            f"marks {cell.sheet_fraction_at_best:.1%} of the sheet)",
            flush=True,
        )
    if not cells:
        raise ValueError("Give at least one --cell")

    result: dict = {
        "checkpoint": checkpoint,
        "threshold": median_threshold(cells),
        "sheet_fraction": median_fraction(cells),
        "cells": [
            {
                "scroll": cell.scroll,
                "prediction": cell.prediction,
                "best_threshold": cell.best_threshold,
                "sheet_fraction_at_best": cell.sheet_fraction_at_best,
                "f1_best": float(cell.f1_curve.max()),
                f"f1_at_{DEFAULT_THRESHOLD_U8}": float(cell.f1_curve[DEFAULT_THRESHOLD_U8]),
            }
            for cell in cells
        ],
        "check_against_other_scrolls": None,
    }
    if len({cell.scroll for cell in cells}) >= 2:
        check = borrowing_check(cells)
        result["check_against_other_scrolls"] = check
        print("\neach scroll scored with each rule taken from the other scrolls, and at 128 (mean F1 lost):")
        for scroll, row in check.items():
            print(
                f"  {scroll}: value {row['threshold_from_other_scrolls']} loses {row['mean_f1_loss_value']:.3f}, "
                f"quantile {row['sheet_fraction_from_other_scrolls']:.1%} loses {row['mean_f1_loss_quantile']:.3f}, "
                f"128 loses {row['mean_f1_loss_128']:.3f} "
                f"({row['cells']} cell{'' if row['cells'] == 1 else 's'}, best F1 {row['mean_f1_best']:.3f})"
            )
    else:
        print("\nnote: all cells come from one scroll; add a second scroll to see what each rule costs on a scroll it was not taken from.")

    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"\nvalue threshold {result['threshold']}, quantile {result['sheet_fraction']:.2%} of the sheet -> {output_path}")
    return result


def apply(prediction_path: Path, calibration_path: Path, output_path: Path, *, rule: str = "value") -> dict:
    if rule not in RULES:
        raise ValueError(f"--rule must be one of {RULES}")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    prediction = read_prediction(prediction_path)
    sheet = sheet_histogram(prediction)
    if rule == "value":
        threshold = int(calibration["threshold"])
    else:
        threshold = quantile_threshold(sheet, float(calibration["sheet_fraction"]))
    mask = np.where(prediction >= threshold, 255, 0).astype(np.uint8)
    tifffile.imwrite(output_path, mask, compression="lzw")
    report = {
        "prediction": str(prediction_path),
        "rule": rule,
        "threshold": threshold,
        "fraction_of_sheet_marked": float(fraction_at_or_above(sheet)[threshold]),
        "output": str(output_path),
    }
    print(json.dumps(report, indent=1))
    return report


def parse_args(argv: Sequence[str] | None = None):
    parser = HyphenUnderscoreParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    calibrate_parser = commands.add_parser("calibrate", help="best threshold per labelled prediction, and both rules")
    calibrate_parser.add_argument(
        "--cell",
        nargs="+",
        action="append",
        required=True,
        metavar="ARG",
        help=(
            "SCROLL PREDICTION INKLABELS [MASK]: one labelled prediction. MASK limits scoring "
            "(e.g. the supervision mask); without it the whole image is scored. Repeat per prediction."
        ),
    )
    calibrate_parser.add_argument("--checkpoint", default="", help="Note of the checkpoint, stored in the output.")
    calibrate_parser.add_argument("--out", type=Path, required=True, help="Calibration JSON to write.")
    apply_parser = commands.add_parser("apply", help="binarize a prediction with a calibrated rule")
    apply_parser.add_argument("prediction", type=Path)
    apply_parser.add_argument("--calibration", type=Path, required=True)
    apply_parser.add_argument(
        "--rule",
        choices=RULES,
        default="value",
        help="value: cut at the calibrated threshold. quantile: cut where this prediction marks the calibrated fraction of its sheet.",
    )
    apply_parser.add_argument("--out", type=Path, required=True, help="0/255 uint8 TIFF to write.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "calibrate":
        calibrate(args.cell, args.out, checkpoint=args.checkpoint)
    else:
        apply(args.prediction, args.calibration, args.out, rule=args.rule)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
