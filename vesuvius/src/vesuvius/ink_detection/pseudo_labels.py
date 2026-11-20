"""Confidence-thresholded pseudo-labels for flat ink training on an unlabelled target scroll (self-training).

Average the teacher models' predictions on target segments, then label only what they are confident about:

  ink        - mean probability >= ``hi``;
  background - ``background="max"``: every teacher <= ``lo`` and the pixel is more than ``margin`` px (Euclidean) from any
               weak ink candidate (mean >= ``candidate``); ``background="mean"``: the mean <= ``lo``;
  ignored    - everything else (supervision mask 0, so it adds no loss), and everything outside the papyrus (mid-plane of
               the surface volume is 0, eroded by ``erode`` px so render edges are not used).

The flat recipe trains with BCE label smoothing 0.5, so its outputs lie in [0.25, 0.75] and 0.5 is the decision point;
the defaults (hi 0.50, lo 0.30) are set for that. Output, per segment, in the layout ``find_segment_patches`` reads:
``<out>/<name>/<name>_inklabels.zarr`` and ``<name>_supervision_mask.zarr`` (same depth as the surface volume, label on
the middle plane, values 0/255). Add them as their own dataset group with a share of the batch, e.g. 16 of 64 slots.

**First Letters**: ink outputs of a submitted region must not overlap any training data. Pseudo-label some regions of the
target scroll and read others; never read a pseudo-labelled region.

Leave-one-scroll-out on PHerc 0139 native 9.362 um segments (seed 42, 25k steps, 16 of 64 slots): mean ROC-AUC 0.743
(no target data) -> 0.786 (``background="mean"``) -> 0.804 (``background="max"``, margin 8, mirror-TTA teachers); the
latter student beats its own four-model teacher (0.813 vs 0.808 with mirror TTA). See docs/ink_cross_scroll.md.

CLI::

    python -m vesuvius.ink_detection.pseudo_labels OUT_ROOT \\
        --segment NAME SURFACE_VOLUME.zarr PRED1.tif [PRED2.tif ...] [--segment ...] \\
        [--hi 0.5 --lo 0.3 --background max --margin 8 --candidate 0.4 --erode 16]

Predictions are forward-direction uint8 TIFFs (probability x 255) on the surface volume's grid, e.g. from
``vesuvius.ink_detection.inference.infer --direction forward --tta-mirror``.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from scipy import ndimage


def pseudo_label_masks(
    mean: np.ndarray,
    top: np.ndarray,
    papyrus: np.ndarray,
    *,
    hi: float = 0.50,
    lo: float = 0.30,
    background: str = "max",
    margin: float = 8.0,
    candidate: float = 0.40,
    erode: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Boolean (ink, supervision) masks from the teachers' mean and per-pixel maximum probability."""

    if background not in ("max", "mean"):
        raise ValueError("background must be 'max' or 'mean'")
    if not 0.0 <= lo < hi <= 1.0:
        raise ValueError("need 0 <= lo < hi <= 1")
    valid = np.asarray(papyrus, bool)
    if erode > 0:
        valid = ndimage.binary_erosion(valid, iterations=int(erode))
    ink = valid & (mean >= hi)
    bg = valid & ((top if background == "max" else mean) <= lo)
    if margin > 0:
        bg &= ndimage.distance_transform_edt(mean < candidate) > margin
    bg &= ~ink
    return ink, ink | bg


def _write_label(path: Path, plane: np.ndarray, depth: int, attrs: dict) -> None:
    from vc_reader.reader import create_group

    tmp = path.with_name(path.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    h, w = plane.shape
    group = create_group(tmp, overwrite=True, auth=None)
    array = group.create_array("0", shape=(depth, h, w), chunks=(depth, 128, 128), dtype=np.uint8,
                               cname="zstd", level=5, shuffle=2, fill_value=0)
    group.attrs.update(attrs)
    block = np.zeros((depth, h, w), np.uint8)
    block[depth // 2] = plane
    array[:, :, :] = block
    shutil.rmtree(path, ignore_errors=True)
    tmp.rename(path)


def build_segment(out_root: Path, name: str, surface_volume: str, predictions: list[str], **rule) -> dict:
    import tifffile
    from vc_reader.reader import open_node

    volume = open_node(f"{surface_volume}/0")
    depth = int(volume.shape[0])
    papyrus = np.asarray(volume[depth // 2]) > 0
    mean = np.zeros(papyrus.shape, np.float32)
    top = np.zeros(papyrus.shape, np.float32)
    for path in predictions:
        p = tifffile.imread(path).astype(np.float32) / 255.0
        if p.shape != papyrus.shape:
            raise ValueError(f"{path}: prediction {p.shape} does not match the surface volume {papyrus.shape}")
        mean += p
        np.maximum(top, p, out=top)
    mean /= len(predictions)
    ink, supervision = pseudo_label_masks(mean, top, papyrus, **rule)
    seg_dir = out_root / name
    seg_dir.mkdir(parents=True, exist_ok=True)
    attrs = {"format": "ink-pseudo-labels-v1", "case": name, "annotation_center_channel": depth // 2,
             "output_channels": depth, "teachers": predictions, "surface_volume": surface_volume, **rule}
    _write_label(seg_dir / f"{name}_inklabels.zarr", ink.astype(np.uint8) * 255, depth, attrs)
    _write_label(seg_dir / f"{name}_supervision_mask.zarr", supervision.astype(np.uint8) * 255, depth, attrs)
    n = max(int(papyrus.sum()), 1)
    return {"segment": name, "papyrus_px": int(papyrus.sum()), "ink": float(ink.sum() / n),
            "background": float((supervision & ~ink).sum() / n), "ignored": float(1 - supervision.sum() / n)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out_root", type=Path)
    ap.add_argument("--segment", nargs="+", action="append", required=True,
                    metavar="NAME SURFACE_VOLUME PRED", help="segment name, its surface volume and teacher predictions")
    ap.add_argument("--hi", type=float, default=0.50)
    ap.add_argument("--lo", type=float, default=0.30)
    ap.add_argument("--background", choices=["max", "mean"], default="max")
    ap.add_argument("--margin", type=float, default=8.0)
    ap.add_argument("--candidate", type=float, default=0.40)
    ap.add_argument("--erode", type=int, default=16)
    a = ap.parse_args(argv)
    rule = {"hi": a.hi, "lo": a.lo, "background": a.background, "margin": a.margin, "candidate": a.candidate,
            "erode": a.erode}
    group = {"segments_path": str(a.out_root), "segments": [], "surface_volume_paths": {}}
    for spec in a.segment:
        if len(spec) < 3:
            ap.error("--segment needs NAME SURFACE_VOLUME and at least one prediction")
        name, volume, preds = spec[0], spec[1], spec[2:]
        stats = build_segment(a.out_root, name, volume, preds, **rule)
        print(json.dumps(stats))
        group["segments"].append(name)
        group["surface_volume_paths"][name] = volume
    print("dataset entry (add sampling keys and a batch share):")
    print(json.dumps(group, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
