#!/usr/bin/env python3
"""Suggest where to label ink next: the tiles where a segment's predictions are least sure.

Reads ink predictions of one segment (a ``preds`` folder, or files named on the command line),
scores each pixel by how unsure the predictions are there, averages that score over a fixed grid
of tiles, and writes the top tiles as a mask beside a matched random control:

- ``label_suggestions.png``: 255 inside a suggested tile, 0 elsewhere.
- ``random_tiles.png``: as many random eligible tiles, seeded, to compare against.
- ``suggestions.json``: the ranked tiles, the settings, the inputs and the shuffle check.

No label is read. A suggested tile is a place to look, not a claim that a label there is wrong
or that the tile holds ink.

Measures (``--measure``), on the predictions read as probabilities and averaged (``m``, the mean
probability; the combined call is ink where ``m`` reaches ``cut``, 0.5 by default):

- ``edge`` (default): 1 where the combined call differs from a valid 4-neighbor's, so a tile
  scores its share of predicted ink edge. Works with one prediction.
- ``margin``: ``1 - 2 |m - cut|``, highest where the mean probability sits at the cut. Works with
  one prediction.
- ``vote``: ``4 p (1 - p)`` for the share ``p`` of predictions at or above ``cut``: how evenly the
  predictions split. Needs at least two, and is 0 where they all agree.

Probabilities: uint8 and uint16 predictions are divided by their type's maximum; float
predictions must already lie in [0, 1].

Shuffle check: the per-pixel scores are shuffled over the valid pixels ``--nulls`` times and the
tiles rescored. The suggestion is ``targeted`` when its top tiles' mean beats every shuffle, and
``flat`` otherwise. It tests only that the scores cluster in place; ``edge`` and ``margin`` follow
the predicted ink's outline, so they cluster even where the predictions agree.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from vesuvius.ink_detection.preprocessing.merge_predictions import (
    cleanup_prepared_predictions,
    is_merged_output,
    is_supported_prediction_file,
    parse_prediction_direction,
    prepare_prediction,
)
from vesuvius.ink_detection.preprocessing.staged_write import (
    create_staged_output,
    discard_staged_output,
    publish_staged_output,
)
from vesuvius.utils.cli import HyphenUnderscoreParser


MEASURE_CHOICES = ("edge", "margin", "vote")
DIRECTION_CHOICES = ("all", "forward", "reverse")
DEFAULT_MEASURE = "edge"
DEFAULT_TILE = 64
DEFAULT_TOP = 20
DEFAULT_MIN_VALID = 0.5
DEFAULT_CUT = 0.5
DEFAULT_NULLS = 5
OUTPUT_NAMES = {
    "suggestions": "label_suggestions.png",
    "random": "random_tiles.png",
    "json": "suggestions.json",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = HyphenUnderscoreParser(
        description=(
            "Rank a segment's tiles by how unsure its ink predictions are and write the top tiles "
            "as a label suggestion mask, beside matched random tiles. No label is read."
        )
    )
    parser.add_argument(
        "inputs",
        type=Path,
        nargs="+",
        help=(
            "A preds folder (its non-merged .png/.tif/.tiff predictions are used) or prediction "
            "files of the same segment on the same grid."
        ),
    )
    parser.add_argument("--output_dir", type=Path, required=True, help="Folder for the outputs; created if missing.")
    parser.add_argument(
        "--direction",
        choices=DIRECTION_CHOICES,
        default="all",
        help=(
            "Which predictions to read from a preds folder by their _forward_/_reverse_ name: all "
            "(the default, including files with neither), forward or reverse."
        ),
    )
    parser.add_argument(
        "--mask",
        type=Path,
        default=None,
        help="Optional image on the same grid; only its nonzero pixels can be suggested (for example a segment mask).",
    )
    parser.add_argument("--measure", choices=MEASURE_CHOICES, default=DEFAULT_MEASURE, help="Default: edge.")
    parser.add_argument(
        "--tile",
        type=int,
        default=DEFAULT_TILE,
        help=(
            f"Tile edge in pixels, on a grid from the top left; a right or bottom strip narrower "
            f"than one tile is never suggested. Default: {DEFAULT_TILE}."
        ),
    )
    parser.add_argument("--top", type=int, default=DEFAULT_TOP, help=f"Tiles to suggest. Default: {DEFAULT_TOP}.")
    parser.add_argument(
        "--min_valid",
        type=float,
        default=DEFAULT_MIN_VALID,
        help=f"Smallest share of valid pixels for a tile to count. Default: {DEFAULT_MIN_VALID}.",
    )
    parser.add_argument("--cut", type=float, default=DEFAULT_CUT, help=f"Ink probability cut. Default: {DEFAULT_CUT}.")
    parser.add_argument("--nulls", type=int, default=DEFAULT_NULLS, help=f"Shuffles for the check. Default: {DEFAULT_NULLS}.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the random tiles and the shuffles. Default: 0.")
    parser.add_argument("--overwrite", action="store_true", help="Replace outputs already in --output_dir.")
    args = parser.parse_args(argv)
    if args.tile < 4 or args.top < 1 or not 0.0 < args.min_valid <= 1.0 or args.nulls < 0:
        parser.error("--tile must be at least 4, --top at least 1, --min_valid in (0, 1] and --nulls at least 0")
    if not 0.0 < args.cut < 1.0:
        parser.error("--cut must lie strictly between 0 and 1")
    return args


def collect_inputs(inputs: Sequence[Path], *, direction: str = "all", measure: str = DEFAULT_MEASURE) -> list[Path]:
    """The prediction files: a preds folder's non-merged predictions of the chosen direction, or the files given."""
    files: list[Path] = []
    for path in inputs:
        if path.is_dir():
            files.extend(
                candidate
                for candidate in sorted(path.iterdir())
                if is_supported_prediction_file(candidate)
                and not is_merged_output(candidate)
                and (direction == "all" or parse_prediction_direction(candidate) == direction)
            )
        elif is_supported_prediction_file(path):
            files.append(path)
        else:
            raise ValueError(f"Not a prediction file or folder: {path}")
    unique = list(dict.fromkeys(p.resolve() for p in files))
    needed = 2 if measure == "vote" else 1
    if len(unique) < needed:
        raise ValueError(f"--measure {measure} needs at least {needed} prediction(s), found {len(unique)}")
    return unique


def to_probability(array: np.ndarray, path: Path) -> np.ndarray:
    """uint8 and uint16 over their type's maximum; floats as they are, which must lie in [0, 1]."""
    if array.dtype in (np.uint8, np.uint16):
        return array.astype(np.float32) / float(np.iinfo(array.dtype).max)
    if np.issubdtype(array.dtype, np.floating):
        if array.size and (float(np.min(array)) < 0.0 or float(np.max(array)) > 1.0):
            raise ValueError(f"{path}: float predictions must lie in [0, 1]")
        return array.astype(np.float32)
    raise ValueError(f"{path}: give predictions as uint8, uint16 or float in [0, 1], not {array.dtype}")


def read_image(path: Path) -> np.ndarray:
    prepared = prepare_prediction(path)
    try:
        return np.array(prepared.array)
    finally:
        cleanup_prepared_predictions([prepared])


def committee(paths: Sequence[Path], cut: float, shape: tuple[int, int] | None = None):
    """Mean probability, share of predictions at or above cut, and pixels nonzero in any prediction.

    Reads one prediction at a time; besides the three outputs, each read holds that prediction
    and one probability array of the same size. ``shape``, when given, is checked on the first read.
    """
    prob_sum = votes = nonzero = None
    for path in paths:
        prepared = prepare_prediction(path)
        try:
            array = np.asarray(prepared.array)
            if prob_sum is None:
                if shape is not None and array.shape != shape:
                    raise ValueError(f"--mask is {shape}, the predictions are {array.shape}")
                shape = array.shape
                prob_sum = np.zeros(shape, np.float32)
                votes = np.zeros(shape, np.uint16)
                nonzero = np.zeros(shape, bool)
            elif array.shape != shape:
                raise ValueError(f"{path} is {array.shape}, others are {shape}: put every prediction on one grid")
            prob = to_probability(array, path)
            prob_sum += prob
            votes += prob >= cut
            nonzero |= array != 0
        finally:
            cleanup_prepared_predictions([prepared])
    count = float(len(paths))
    return prob_sum / count, votes.astype(np.float32) / count, nonzero


def ink_edge(call: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """1 where the call differs from a 4-neighbor, both pixels valid, else 0."""
    edge = np.zeros(call.shape, bool)
    vertical = (call[:-1, :] != call[1:, :]) & valid[:-1, :] & valid[1:, :]
    edge[:-1, :] |= vertical
    edge[1:, :] |= vertical
    horizontal = (call[:, :-1] != call[:, 1:]) & valid[:, :-1] & valid[:, 1:]
    edge[:, :-1] |= horizontal
    edge[:, 1:] |= horizontal
    return edge.astype(np.float32)


def uncertainty(mean: np.ndarray, vote_share: np.ndarray, valid: np.ndarray, measure: str, cut: float) -> np.ndarray:
    if measure == "edge":
        score = ink_edge(mean >= cut, valid)
    elif measure == "margin":
        score = np.clip(1.0 - 2.0 * np.abs(mean - cut), 0.0, 1.0)
    elif measure == "vote":
        score = 4.0 * vote_share * (1.0 - vote_share)
    else:
        raise ValueError(f"Unknown measure {measure!r}")
    score = score.astype(np.float32)
    score[~valid] = 0.0
    return score


def tile_scores(score: np.ndarray, valid: np.ndarray, tile: int, min_valid: float) -> np.ndarray:
    """Rows (y, x, mean score over the tile's valid pixels, valid share) for every eligible tile."""
    height, width = score.shape
    rows, cols = height // tile, width // tile
    if rows == 0 or cols == 0:
        raise ValueError(f"A {width} x {height} map is smaller than one {tile} px tile")
    crop = (slice(0, rows * tile), slice(0, cols * tile))
    valid_count = valid[crop].reshape(rows, tile, cols, tile).sum(axis=(1, 3))
    score_sum = score[crop].reshape(rows, tile, cols, tile).sum(axis=(1, 3), dtype=np.float64)
    share = valid_count / float(tile * tile)
    keep = (share >= min_valid) & (valid_count > 0)
    ii, jj = np.nonzero(keep)
    means = score_sum[ii, jj] / valid_count[ii, jj]
    return np.column_stack([ii * tile, jj * tile, means, share[ii, jj]])


def pick_top(table: np.ndarray, k: int) -> np.ndarray:
    """The k highest-scoring tiles; ties go to the earlier tile in row-major order."""
    order = np.lexsort((table[:, 1], table[:, 0], -table[:, 2]))
    return table[order[:k]]


def pick_random(table: np.ndarray, k: int, seed: int) -> np.ndarray:
    if len(table) == 0:
        return table
    index = np.random.default_rng(seed).choice(len(table), size=min(k, len(table)), replace=False)
    return table[np.sort(index)]


def shuffle_check(score, valid, table, tile, min_valid, k, nulls, seed) -> dict:
    observed = float(pick_top(table, k)[:, 2].mean())
    rng = np.random.default_rng(seed + 1)
    values = score[valid]
    shuffled_means = []
    for _ in range(nulls):
        shuffled = np.zeros_like(score)
        shuffled[valid] = rng.permutation(values)
        shuffled_means.append(float(pick_top(tile_scores(shuffled, valid, tile, min_valid), k)[:, 2].mean()))
    if not shuffled_means:
        verdict = "unchecked"
    else:
        verdict = "targeted" if observed > max(shuffled_means) else "flat"
    return {
        "top_mean_score": round(observed, 6),
        "shuffled_top_means": [round(v, 6) for v in shuffled_means],
        "verdict": verdict,
    }


def tiles_json(table: np.ndarray, tile: int) -> list[dict]:
    return [
        {"rank": i + 1, "y": int(y), "x": int(x), "h": tile, "w": tile, "score": round(float(s), 6),
         "valid_share": round(float(v), 4)}
        for i, (y, x, s, v) in enumerate(table)
    ]


def tile_mask(shape: tuple[int, int], tiles: Sequence[dict]) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    for t in tiles:
        mask[t["y"]:t["y"] + t["h"], t["x"]:t["x"] + t["w"]] = 255
    return mask


def suggest(paths, *, mask=None, measure=DEFAULT_MEASURE, tile=DEFAULT_TILE, top=DEFAULT_TOP,
            min_valid=DEFAULT_MIN_VALID, cut=DEFAULT_CUT, nulls=DEFAULT_NULLS, seed=0):
    """(result dict, suggestion mask, random-tile mask) for prediction files of one segment."""
    mask_array = None if mask is None else read_image(Path(mask)) != 0
    mean, vote_share, valid = committee(paths, cut, None if mask_array is None else mask_array.shape)
    if mask_array is not None:
        valid &= mask_array
    score = uncertainty(mean, vote_share, valid, measure, cut)
    del mean, vote_share
    table = tile_scores(score, valid, tile, min_valid)
    if len(table) == 0:
        raise ValueError("No tile has enough valid pixels; lower --min_valid or --tile, or check --mask")
    chosen = tiles_json(pick_top(table, top), tile)
    control = tiles_json(pick_random(table, len(chosen), seed), tile)
    result = {
        "inputs": [str(p) for p in paths],
        "settings": {"measure": measure, "tile": tile, "top": top, "min_valid": min_valid, "cut": cut,
                     "nulls": nulls, "seed": seed, "mask": None if mask is None else str(mask)},
        "shape": list(valid.shape),
        "valid_pixels": int(valid.sum()),
        "eligible_tiles": int(len(table)),
        "suggested": chosen,
        "random": control,
        "shuffle_check": shuffle_check(score, valid, table, tile, min_valid, top, nulls, seed),
        "note": "Prediction uncertainty only; no label was read. A suggested tile is a place to look.",
    }
    return result, tile_mask(valid.shape, chosen), tile_mask(valid.shape, control)


def write_outputs(targets: dict[str, Path], result: dict, chosen: np.ndarray, control: np.ndarray) -> None:
    """Stage all three outputs, then publish them, so a failure leaves none of the new set half written."""
    staged: dict[str, Path] = {}
    try:
        for key, path in targets.items():
            staged[key] = create_staged_output(path)
        Image.fromarray(chosen).save(staged["suggestions"], format="PNG")
        Image.fromarray(control).save(staged["random"], format="PNG")
        staged["json"].write_text(json.dumps(result, indent=2) + "\n")
        for key, path in targets.items():
            publish_staged_output(staged.pop(key), path)
    finally:
        for path in staged.values():
            discard_staged_output(path)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths = collect_inputs(args.inputs, direction=args.direction, measure=args.measure)
    out = args.output_dir
    targets = {key: out / name for key, name in OUTPUT_NAMES.items()}
    existing = [str(p) for p in targets.values() if p.exists()]
    if existing and not args.overwrite:
        raise SystemExit(f"Outputs already exist (pass --overwrite to replace): {', '.join(existing)}")
    result, chosen, control = suggest(
        paths, mask=args.mask, measure=args.measure, tile=args.tile, top=args.top,
        min_valid=args.min_valid, cut=args.cut, nulls=args.nulls, seed=args.seed,
    )
    out.mkdir(parents=True, exist_ok=True)
    write_outputs(targets, result, chosen, control)
    check = result["shuffle_check"]
    print(
        f"{len(result['suggested'])} of {result['eligible_tiles']} tiles suggested from {len(paths)} predictions "
        f"({args.measure}); shuffle check: {check['verdict']}; wrote {out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
