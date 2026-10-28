"""Score flat ink predictions against a segment's labels, held-out region first.

A segment folder with ``<segment>_validation_mask.zarr`` (the published
``ink_9um`` labels have one for three segments) is scored twice: inside the
validation mask, which training never sees, and on the rest of the supervision
mask, which training does see. Several predictions of the same segment are
compared with a paired block bootstrap on the same pixels.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import tifffile

from vesuvius.ink_detection.data.segment import parse_label_asset_path
from vesuvius.ink_detection.volume_io import open_volume_root, select_volume_level
from vesuvius.utils.cli import HyphenUnderscoreParser


LOGGER = logging.getLogger(__name__)
LEVELS = 256
DEFAULT_THRESHOLD = 128
DEFAULT_BLOCK_SIZE = 64
DEFAULT_BOOTSTRAPS = 2000
_BOOTSTRAP_BATCH = 250


def find_label_assets(labels_dir: Path, segment: str | None = None) -> dict[str, Path]:
    """Return the greatest version of each label kind for one segment folder."""

    labels_dir = Path(labels_dir)
    segment = labels_dir.name if segment is None else segment
    found: dict[str, tuple[int, Path]] = {}
    for path in labels_dir.iterdir():
        parsed = parse_label_asset_path(path)
        if parsed is None or parsed["prefix"] != segment:
            continue
        kind, version = str(parsed["label_kind"]), int(parsed["version_num"])
        if kind not in found or version > found[kind][0]:
            found[kind] = (version, path)
    for kind in ("inklabels", "supervision_mask"):
        if kind not in found:
            raise FileNotFoundError(
                f"{labels_dir} has no {segment}_{kind} asset (.zarr, .tif or .tiff)"
            )
    return {kind: path for kind, (_, path) in found.items()}


def read_label_plane(path: Path) -> np.ndarray:
    """Read the annotated plane of a label asset as a boolean YX image.

    Label Zarrs are ZYX with the annotation on one plane: the
    ``annotation_center_channel`` attribute when present, else the middle
    plane (``create_label_zarrs`` writes plane 32 of 65, the published
    ``ink_9um`` labels plane 10 of 21).
    """

    path = Path(path)
    if path.suffix.lower() == ".zarr":
        root = open_volume_root(path)
        array = select_volume_level(root, 0, source=str(path))
        center = None
        for attrs in (getattr(root, "attrs", {}), getattr(array, "attrs", {})):
            if "annotation_center_channel" in attrs:
                center = int(attrs["annotation_center_channel"])
        if array.ndim == 2:
            plane = np.asarray(array[:])
        elif array.ndim == 3:
            plane = np.asarray(array[array.shape[0] // 2 if center is None else center])
        else:
            raise ValueError(f"{path}: expected a YX or ZYX label array, got {array.shape}")
    else:
        plane = np.squeeze(tifffile.imread(path))
        if plane.ndim == 3:
            plane = plane[plane.shape[0] // 2]
    if plane.ndim != 2:
        raise ValueError(f"{path}: expected a 2D label image, got {plane.shape}")
    return plane > 0


def read_prediction(path: Path) -> np.ndarray:
    """Read a flat prediction as uint8 levels 0..255 (floats are taken as 0..1)."""

    image = np.squeeze(tifffile.imread(path))
    if image.ndim != 2:
        raise ValueError(f"{path}: expected a 2D prediction image, got {image.shape}")
    if image.dtype == np.uint8:
        return image
    if np.issubdtype(image.dtype, np.floating):
        return np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)
    raise ValueError(f"{path}: expected uint8 or float predictions, got {image.dtype}")


def auroc_from_histograms(positive: np.ndarray, negative: np.ndarray) -> np.ndarray:
    """Exact AUROC (ties count one half) from per-level counts on the last axis."""

    below = np.cumsum(negative, axis=-1) - negative
    pairs = positive.sum(-1) * negative.sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (positive * (below + 0.5 * negative)).sum(-1) / pairs


def threshold_scores(positive: np.ndarray, negative: np.ndarray) -> dict[str, object]:
    """F1 of ``prediction >= t`` at every level t, from per-level counts."""

    tp = positive[::-1].cumsum()[::-1]
    fp = negative[::-1].cumsum()[::-1]
    fn = positive.sum() - tp
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.nan_to_num(2 * tp / (2 * tp + fp + fn))
    best = int(np.argmax(f1))
    return {
        f"f1_at_{DEFAULT_THRESHOLD}": float(f1[DEFAULT_THRESHOLD]),
        "best_f1": float(f1[best]),
        "best_threshold": best,
    }


def score_region(
    predictions: dict[str, np.ndarray],
    inklabels: np.ndarray,
    region: np.ndarray,
    *,
    reference: str,
    block_size: int = DEFAULT_BLOCK_SIZE,
    bootstraps: int = DEFAULT_BOOTSTRAPS,
    seed: int = 0,
) -> dict[str, object]:
    """Score every prediction on one region; compare each with ``reference``.

    Confidence intervals resample ``block_size`` pixel squares with
    replacement, the same squares for every prediction, so neighbouring
    pixels are not counted as independent evidence.
    """

    ys, xs = np.nonzero(region)
    if ys.size == 0:
        return {"pixels": 0}
    ink = inklabels[ys, xs]
    blocks_x = region.shape[1] // block_size + 1
    _, block = np.unique((ys // block_size) * blocks_x + xs // block_size, return_inverse=True)
    block = block.ravel()
    n_blocks = int(block.max()) + 1
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(n_blocks, np.full(n_blocks, 1.0 / n_blocks), size=bootstraps)

    result: dict[str, object] = {
        "pixels": int(ys.size),
        "ink_fraction": float(ink.mean()),
        "blocks": n_blocks,
        "predictions": {},
    }
    boot: dict[str, np.ndarray] = {}
    for name, prediction in predictions.items():
        level = prediction[ys, xs].astype(np.int64)
        index = block * LEVELS + level
        positive = np.bincount(index[ink], minlength=n_blocks * LEVELS)
        negative = np.bincount(index[~ink], minlength=n_blocks * LEVELS)
        positive = positive.reshape(n_blocks, LEVELS).astype(np.float64)
        negative = negative.reshape(n_blocks, LEVELS).astype(np.float64)
        boot[name] = np.concatenate([
            auroc_from_histograms(
                weights[start:start + _BOOTSTRAP_BATCH] @ positive,
                weights[start:start + _BOOTSTRAP_BATCH] @ negative,
            )
            for start in range(0, bootstraps, _BOOTSTRAP_BATCH)
        ])
        totals_positive, totals_negative = positive.sum(0), negative.sum(0)
        result["predictions"][name] = {
            "auroc": float(auroc_from_histograms(totals_positive, totals_negative)),
            "auroc_ci95": [float(v) for v in np.nanpercentile(boot[name], [2.5, 97.5])],
            **threshold_scores(totals_positive, totals_negative),
        }
    result["vs_reference"] = {}
    for name in predictions:
        if name == reference:
            continue
        difference = boot[name] - boot[reference]
        result["vs_reference"][name] = {
            "auroc_delta": result["predictions"][name]["auroc"]
            - result["predictions"][reference]["auroc"],
            "ci95": [float(v) for v in np.nanpercentile(difference, [2.5, 97.5])],
            "fraction_not_better": float(np.mean(difference <= 0)),
        }
    return result


def evaluate_segment(
    labels_dir: Path,
    predictions: dict[str, Path],
    *,
    segment: str | None = None,
    reference: str | None = None,
    block_size: int = DEFAULT_BLOCK_SIZE,
    bootstraps: int = DEFAULT_BOOTSTRAPS,
    seed: int = 0,
) -> dict[str, object]:
    """Score predictions of one segment inside its validation and training regions."""

    if not predictions:
        raise ValueError("at least one prediction is required")
    reference = next(iter(predictions)) if reference is None else reference
    if reference not in predictions:
        raise ValueError(f"reference {reference!r} is not one of {sorted(predictions)}")
    assets = find_label_assets(labels_dir, segment)
    inklabels = read_label_plane(assets["inklabels"])
    supervision = read_label_plane(assets["supervision_mask"])
    images = {name: read_prediction(path) for name, path in predictions.items()}
    for name, image in images.items():
        if image.shape != inklabels.shape:
            raise ValueError(
                f"prediction {name!r} is {image.shape[0]}x{image.shape[1]} but the "
                f"labels are {inklabels.shape[0]}x{inklabels.shape[1]}; run inference "
                "on the surface volume that sits beside these labels"
            )
    regions: dict[str, np.ndarray] = {}
    if "validation_mask" in assets:
        validation = read_label_plane(assets["validation_mask"])
        regions["validation"] = validation
        regions["supervision_minus_validation"] = supervision & ~validation
    else:
        LOGGER.warning(
            "%s has no validation mask: only the supervision region is scored, "
            "and a checkpoint trained on this segment has seen those labels",
            labels_dir,
        )
        regions["supervision"] = supervision
    result: dict[str, object] = {
        "labels": {kind: str(path) for kind, path in sorted(assets.items())},
        "predictions": {name: str(path) for name, path in predictions.items()},
        "reference": reference,
        "block_size": block_size,
        "bootstraps": bootstraps,
        "seed": seed,
        "regions": {},
    }
    for region_name, region in regions.items():
        result["regions"][region_name] = score_region(
            images, inklabels, region,
            reference=reference, block_size=block_size, bootstraps=bootstraps, seed=seed,
        )
    return result


def format_report(result: dict[str, object]) -> str:
    """Render ``evaluate_segment`` output as a plain-text table."""

    lines = []
    for region_name, region in result["regions"].items():
        if not region["pixels"]:
            lines.append(f"{region_name}: no pixels")
            continue
        lines.append(
            f"{region_name}: {region['pixels']} px, ink {100 * region['ink_fraction']:.1f}%, "
            f"{region['blocks']} blocks of {result['block_size']} px"
        )
        width = max(len(name) for name in region["predictions"])
        lines.append(
            f"  {'prediction':<{width}}  AUROC  [95% CI]          F1@{DEFAULT_THRESHOLD}  best F1 (t)"
        )
        for name, scores in region["predictions"].items():
            low, high = scores["auroc_ci95"]
            lines.append(
                f"  {name:<{width}}  {scores['auroc']:.4f} [{low:.4f}, {high:.4f}]  "
                f"{scores[f'f1_at_{DEFAULT_THRESHOLD}']:.3f}   "
                f"{scores['best_f1']:.3f} ({scores['best_threshold']})"
            )
        for name, delta in region["vs_reference"].items():
            low, high = delta["ci95"]
            lines.append(
                f"  {name} - {result['reference']}: AUROC {delta['auroc_delta']:+.4f} "
                f"[{low:+.4f}, {high:+.4f}], not better in {100 * delta['fraction_not_better']:.1f}% of resamples"
            )
    return "\n".join(lines)


def _parse_prediction(text: str) -> tuple[str, Path]:
    name, separator, path = text.partition("=")
    if not separator:
        return Path(text).stem, Path(text)
    return name, Path(path)


def parse_args(argv: Sequence[str] | None = None):
    parser = HyphenUnderscoreParser(
        description=(
            "Score flat ink predictions against a segment's labels, separately inside "
            "its validation mask and on the rest of its supervision mask"
        )
    )
    parser.add_argument(
        "labels_dir", type=Path,
        help="Segment folder holding <segment>_inklabels, _supervision_mask and "
        "optionally _validation_mask (.zarr, .tif or .tiff)",
    )
    parser.add_argument(
        "predictions", nargs="+", metavar="[NAME=]PREDICTION_TIFF",
        help="uint8 prediction TIFFs on the labels' pixel grid, e.g. infer output",
    )
    parser.add_argument("--segment", help="Label basename prefix (default: folder name)")
    parser.add_argument("--reference", help="Prediction the others are compared with (default: first)")
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument("--bootstraps", type=int, default=DEFAULT_BOOTSTRAPS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", type=Path, help="Also write the full result as JSON")
    args = parser.parse_args(argv)
    if args.block_size <= 0:
        parser.error("--block-size must be positive")
    if args.bootstraps <= 0:
        parser.error("--bootstraps must be positive")
    pairs = [_parse_prediction(text) for text in args.predictions]
    args.predictions = dict(pairs)
    if len(args.predictions) != len(pairs):
        parser.error("prediction names must be unique; use NAME=PATH")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
    result = evaluate_segment(
        args.labels_dir,
        args.predictions,
        segment=args.segment,
        reference=args.reference,
        block_size=args.block_size,
        bootstraps=args.bootstraps,
        seed=args.seed,
    )
    print(format_report(result))
    if args.json_out is not None:
        args.json_out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
