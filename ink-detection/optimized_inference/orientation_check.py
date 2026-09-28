"""Decide which way to read a surface volume's layers, without ink labels.

`run_inference` reads a surface volume's layers in one of two orders -- `FORCE_REVERSE` picks
which -- and nothing says which one a new segment needs. Running both and comparing the two maps
does not settle it on a scroll that has not been read: the amount of ink each order reports is
not a signal. With the ResNet3D-152 3D-decoder checkpoint on PHerc0139's published surface
volume, the wrong order reports 20 times *less* ink than the right one; on a PHerc1451 surface
it reports 8 times *more*.

This settles it by measurement instead. It plants synthetic ink of known contrast on one face
of the sheet, then the other, and runs this module's own inference both ways -- four runs, plus
the two clean ones they are compared against:

- the wrong order answers with nothing, on either face;
- the right order with ink on the wrong face answers *negatively* -- a bright layer where the
  model expects bare papyrus reads as the absence of ink;
- the right order with ink on the right face is the one combination that answers positively.

Measured with this module's `run_inference` and the ResNet3D-152 3D-decoder checkpoint, at
amplitude 32 (of 255), on a 1920 px window of PHerc0139's published surface volume (layers
24-86) -- whose right answer is known from the team's own ink map -- and on the same window
flipped in depth, whose right answer is known by construction:

                     fwd near   fwd far   rev near   rev far    verdict
    as published      +0.041    -0.233     0.000      0.000     forward  (FORCE_REVERSE=false)
    flipped            0.000     0.000    -0.021     +0.098     reverse  (FORCE_REVERSE=true)

The flipped row is what makes it usable: the verdict follows the data, not the face the ink is
planted on. A check that plants on one face only reports a readable stack whose ink is on the
other face as blind.

    python orientation_check.py SURFACE_VOLUME.zarr WEIGHTS.ckpt \\
        --model-type resnet3d-152-3d-decoder --start-layer 24 --end-layer 86

It reads one window (1920 px square by default, the smallest that holds two lines of writing
at 2.4 um) from the middle of the volume, or from `--y/--x`.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from typing import Callable, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

STROKE_MM = 0.35        # stroke width on these scrolls
LINE_MM = 2.0           # line spacing
SHEET_BAND = 6          # layers between the sheet's centre and a face, twice over


def sheet_depth(stack: np.ndarray, search: int = 12) -> np.ndarray:
    """Per-pixel depth of the sheet's centre, from the stack's own brightness."""
    c = stack.shape[0]
    mid = (c - 1) / 2.0
    lo, hi = int(max(0, mid - search)), int(min(c, mid + search) + 1)
    band = stack[lo:hi].astype(np.float32)
    depth = np.arange(lo, hi, dtype=np.float32)[:, None, None]
    weight = np.clip(band - np.percentile(band, 20, axis=0, keepdims=True), 0, None)
    total = weight.sum(0)
    return np.where(total > 1e-6, (weight * depth).sum(0) / np.maximum(total, 1e-6), mid)


def script_mask(shape: Tuple[int, int], micron_per_pixel: float, seed: int = 0) -> np.ndarray:
    """Strokes of a pen's width, arranged in lines at a scroll's line spacing."""
    rng = np.random.default_rng(seed)
    h, w = shape
    stroke = max(2, int(round(STROKE_MM * 1000 / micron_per_pixel)))
    line = max(stroke * 3, int(round(LINE_MM * 1000 / micron_per_pixel)))
    if h < line * 2 or w < stroke * 8:
        raise ValueError(f"a {h}x{w} px window cannot hold two lines of writing "
                         f"at {LINE_MM} mm spacing ({line} px)")
    mask = np.zeros(shape, np.float32)
    for y in range(line, h - line, line):
        x = int(rng.integers(0, line))
        while x < w - stroke * 4:
            gw = int(rng.integers(stroke * 2, stroke * 4))
            gh = int(rng.integers(stroke * 2, stroke * 3))
            y0 = y + int(rng.integers(-stroke, stroke))
            if 0 <= y0 < h - gh:
                mask[y0:y0 + gh, x:x + stroke] = 1.0
                mask[y0:y0 + gh, x + gw - stroke:x + gw] = 1.0
                mask[y0 + gh // 2:y0 + gh // 2 + stroke, x:x + gw] = 1.0
            x += gw + int(rng.integers(stroke, stroke * 3))
    return mask


def plant(stack: np.ndarray, amplitude: float, face: str, micron_per_pixel: float,
          seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Synthetic ink on one face of the sheet, following the sheet's measured depth.

    `face` is 'near' (toward layer 0) or 'far' (toward the last layer). A flat plane would test
    the render's geometry rather than the detector, so the layer follows the sheet.
    """
    if face not in ("near", "far"):
        raise ValueError(f"face must be 'near' or 'far', not {face!r}")
    out = stack.astype(np.float32)
    c, h, w = out.shape
    centre = sheet_depth(stack)
    offset = -SHEET_BAND / 2.0 if face == "near" else SHEET_BAND / 2.0
    zz = np.arange(c, dtype=np.float32)[:, None, None]
    profile = np.exp(-0.5 * ((zz - (centre + offset)[None]) / (SHEET_BAND / 2.0)) ** 2)
    mask = script_mask((h, w), micron_per_pixel, seed)
    out += amplitude * profile * mask[None]
    return np.clip(out, 0, 255).astype(stack.dtype), mask


def decide(stack: np.ndarray, predict: Callable[[np.ndarray, bool], np.ndarray],
           amplitude: float = 32, micron_per_pixel: float = 2.4, threshold: float = 0.5,
           margin: float = 0.02, gap: float = 0.01) -> Dict:
    """The four combinations and the verdict. `predict(stack, reverse)` -> (H, W) probability.

    The verdict is the (order, face) with the largest lift, provided it clears `margin` and
    beats the runner-up by `gap`; otherwise None -- a window blind both ways, or one where the
    answer is not clear enough to act on.
    """
    combos = {}
    for reverse in (False, True):
        base = predict(stack, reverse)
        for face in ("near", "far"):
            planted, mask = plant(stack, amplitude, face, micron_per_pixel)
            got = predict(planted, reverse)
            m = mask.astype(bool)
            lift = float((got[m] > threshold).mean() - (base[m] > threshold).mean())
            combos[("reverse" if reverse else "forward", face)] = {
                "lift": round(lift, 3),
                "baseline_ink_pct": round(float((base > threshold).mean()) * 100, 2)}
    ranked = sorted(combos.items(), key=lambda kv: kv[1]["lift"], reverse=True)
    (best, b), (_, r) = ranked[0], ranked[1]
    verdict = best if b["lift"] >= margin and b["lift"] - r["lift"] >= gap else None
    return {"verdict": verdict, "amplitude": float(amplitude),
            "combinations": {f"{o} {f}": v for (o, f), v in combos.items()}}


def _inference_predict(model, device, work_dir: str) -> Callable[[np.ndarray, bool], np.ndarray]:
    """This module's own run_inference, blended the way reduce_partitions blends it."""
    import zarr
    from inference import CFG, run_inference

    def predict(stack: np.ndarray, reverse: bool) -> np.ndarray:
        CFG.zarr_output_dir = work_dir
        CFG.num_parts, CFG.part_id = 1, 0
        paths = run_inference(np.ascontiguousarray(stack.transpose(1, 2, 0)), model, device,
                              is_reverse_segment=reverse)
        pred = zarr.open(paths["mask_pred"], mode="r")[:]
        count = zarr.open(paths["mask_count"], mode="r")[:]
        return np.clip(pred / np.clip(count, 1e-6, None), 0, 1)

    return predict


def _load_model(model_type: str, weights: str, in_chans: int, device):
    if model_type == "timesformer":
        from model_timesformer import load_model
        return load_model(weights, device, num_frames=in_chans)
    if model_type in ("resnet3d-50", "resnet3d-152"):
        from model_resnet3d import load_model
        return load_model(weights, device, num_frames=in_chans,
                          model_depth=int(model_type.rsplit("-", 1)[1]))
    if model_type == "resnet3d-152-3d-decoder":
        from model_resnet3d_3d_decoder import load_model
        return load_model(weights, device, num_frames=in_chans)
    raise ValueError(f"unknown model type {model_type!r}")


def read_window(path: str, start: int, end: int, size: int,
                y: Optional[int], x: Optional[int]) -> np.ndarray:
    import zarr
    z = zarr.open(path, mode="r")
    arr = z["0"] if hasattr(z, "keys") and "0" in z else z
    _, h, w = arr.shape
    y0 = (h - size) // 2 if y is None else y
    x0 = (w - size) // 2 if x is None else x
    return np.asarray(arr[start:end, y0:y0 + size, x0:x0 + size])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("surface_volume", help="surface volume zarr, layers first")
    ap.add_argument("weights", help="checkpoint path")
    ap.add_argument("--model-type", default="resnet3d-152-3d-decoder")
    ap.add_argument("--start-layer", type=int, required=True)
    ap.add_argument("--end-layer", type=int, required=True)
    ap.add_argument("--window", type=int, default=1920)
    ap.add_argument("--y", type=int)
    ap.add_argument("--x", type=int)
    ap.add_argument("--tile-size", type=int, default=256)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--amplitude", type=float, default=32)
    ap.add_argument("--micron-per-pixel", type=float, default=2.4)
    ap.add_argument("--json", help="write the result here as well")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

    import torch
    from inference import CFG

    stack = read_window(args.surface_volume, args.start_layer, args.end_layer,
                        args.window, args.y, args.x)
    CFG.in_chans = stack.shape[0]
    CFG.tile_size = CFG.size = args.tile_size
    CFG.stride = args.stride
    CFG.batch_size = args.batch_size
    CFG.use_zarr_compression = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(args.model_type, args.weights, CFG.in_chans, device)

    with tempfile.TemporaryDirectory() as work:
        result = decide(stack, _inference_predict(model, device, work),
                        amplitude=args.amplitude, micron_per_pixel=args.micron_per_pixel)

    for combo, v in result["combinations"].items():
        print(f"  {combo:14s} lift {v['lift']:+.3f}   baseline ink {v['baseline_ink_pct']:.2f}%")
    if result["verdict"] is None:
        print("no combination answers clearly: blind both ways, or too close to call")
    else:
        order, face = result["verdict"]
        print(f"read {order}, ink on the {face} face  ->  "
              f"FORCE_REVERSE={'true' if order == 'reverse' else 'false'}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump({**result, "verdict": list(result["verdict"]) if result["verdict"] else None},
                      f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
