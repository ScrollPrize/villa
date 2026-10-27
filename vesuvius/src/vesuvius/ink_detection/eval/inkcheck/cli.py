"""Command line interface: `inkcheck PRED REF --pred-um .. --ref-um .. [options]`."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .core import CheckConfig, run_check
from .io import load_map, load_mask
from .report import write_json, write_png


def _pair(s):
    a, b = s.split(",")
    return float(a), float(b)


def build_parser():
    ap = argparse.ArgumentParser(
        prog="inkcheck",
        description="Check whether a predicted ink map reproduces a published reference ink map of the same "
                    "segment (possibly at a different resolution). Prints a verdict and writes JSON + PNG.")
    ap.add_argument("pred", help="predicted ink map (tif/png/jpg/npy)")
    ap.add_argument("ref", help="reference ink map (tif/png/jpg/npy)")
    ap.add_argument("--pred-um", type=float, required=True, help="prediction pixel size in micrometres")
    ap.add_argument("--ref-um", type=float, required=True, help="reference pixel size in micrometres")
    ap.add_argument("--ref-origin", type=_pair, default=(0.0, 0.0),
                    help="y,x position (in REFERENCE pixels) of the prediction's pixel (0,0). Default 0,0")
    ap.add_argument("--mask", help="validity mask on the prediction grid (nonzero = valid)")
    ap.add_argument("--ref-mask", help="validity mask on the reference grid")
    ap.add_argument("--analysis-um", type=float, help="common grid pixel size (default: the coarser input)")
    ap.add_argument("--pred-scale", choices=["raw", "ink9um"], default="raw")
    ap.add_argument("--keep-zero", action="store_true", help="do not treat prediction==0 as invalid")
    ap.add_argument("--ref-threshold", default="otsu", help="reference ink threshold in raw units, or 'otsu'")
    ap.add_argument("--bp-fine-um", type=float, default=0.0)
    ap.add_argument("--bp-coarse-um", type=float, default=500.0)
    ap.add_argument("--search-mm", type=float, default=0.5)
    ap.add_argument("--nulls", type=int, default=192, help="number of shift nulls")
    ap.add_argument("--metric-nulls", type=int, default=32)
    ap.add_argument("--min-shift-mm", type=float, default=3.0)
    ap.add_argument("--min-stroke-mm2", type=float, default=0.02)
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--min-r", type=float, default=0.20)
    ap.add_argument("--min-stroke-recall", type=float, default=0.50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("OMP_NUM_THREADS", "2")))
    ap.add_argument("--out", default="inkcheck_out", help="output prefix (writes PREFIX.json, PREFIX.png)")
    ap.add_argument("--title", default="")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    pred = load_map(a.pred)
    ref = load_map(a.ref)
    pm = load_mask(a.mask) if a.mask else None
    rm = load_mask(a.ref_mask) if a.ref_mask else None
    thr = a.ref_threshold if a.ref_threshold == "otsu" else float(a.ref_threshold)
    cfg = CheckConfig(pred_um=a.pred_um, ref_um=a.ref_um, ref_origin=a.ref_origin, analysis_um=a.analysis_um,
                      pred_scale=a.pred_scale, pred_zero_invalid=not a.keep_zero, ref_threshold=thr,
                      bp_fine_um=a.bp_fine_um, bp_coarse_um=a.bp_coarse_um, search_mm=a.search_mm,
                      n_shift_nulls=a.nulls, n_metric_nulls=a.metric_nulls, min_shift_mm=a.min_shift_mm,
                      min_stroke_mm2=a.min_stroke_mm2, alpha_r=a.alpha, min_r_bp=a.min_r,
                      min_stroke_recall=a.min_stroke_recall, seed=a.seed, workers=a.workers)
    res = run_check(pred, ref, cfg, pm, rm, log=lambda s: print(s, flush=True))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(res, str(out) + ".json", {"pred": a.pred, "ref": a.ref, "mask": a.mask,
                                         "pred_shape": list(pred.shape), "ref_shape": list(ref.shape)})
    write_png(res, str(out) + ".png", a.title or f"{Path(a.pred).name} vs {Path(a.ref).name}")
    for w in res.warnings:
        print("[inkcheck] WARNING:", w)
    print(f"{res.verdict}\t{res.summary}")
    print(f"wrote {out}.json and {out}.png")
    return 0 if res.verdict == "PASS" else (1 if res.verdict == "WEAK" else 2)


if __name__ == "__main__":
    sys.exit(main())
