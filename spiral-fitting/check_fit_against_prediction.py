#!/usr/bin/env python3
"""Check a fitted spiral against the surface prediction, in the fit's own spiral space.

    python check_fit_against_prediction.py checkpoint_fitted.ckpt --umbilicus umbilicus.json \
        --surface <prediction.zarr> --z 6600 7000 7400 [--out-json check.json] [--out-png check.png]

Every voxel of the surface prediction in a few z-slices is mapped through the fit's slice-to-spiral transform. In spiral
space winding k is the curve r = dr * (k + phi / 2pi), so a voxel's fractional winding number is
w = r / dr - phi / 2pi. Where the prediction has one layer per turn of the roll, a fit with one winding per turn
puts every predicted voxel at a whole w. Three numbers follow, overall and per 30-degree sector of spiral angle:

  alignment          mean resultant length of exp(2 pi i w) over the predicted voxels: 1 when every layer sits on
                     a winding, about 0 when the windings are unrelated to the layers
  windings_per_layer median step in w between consecutive layers at the same angle (layers found as clusters of
                     predicted voxels in w): 1 for one winding per predicted layer, above 1 when the fit has more
                     windings than the prediction has layers there. That is either extra windings or layers the
                     prediction misses; either way a nearest-surface snap onto the prediction puts two windings on
                     one predicted surface
  empty_windings     share of (winding, angle) cells in the fitted range with no layer within 0.2 of the winding

The numbers are relative to the prediction and inherit its recall, so where they read above 1, look at the raw CT
to tell extra windings from missed layers. It runs on CPU in seconds per slice, needs no rendering, and reads only
the checkpoint and the prediction.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from checkpoint_io import load_checkpoint_cpu
from flatten_spiral_checkpoint import _build_model, _checkpoint_config


def winding_coordinates(spiral_zyx: np.ndarray, dr: float) -> tuple[np.ndarray, np.ndarray]:
    """Fractional winding number w and angle fraction t in [0, 1) of spiral-space points (z, y, x)."""
    r = np.hypot(spiral_zyx[:, 1], spiral_zyx[:, 2])
    t = np.mod(np.arctan2(spiral_zyx[:, 1], spiral_zyx[:, 2]), 2 * np.pi) / (2 * np.pi)
    return r / dr - t, t


def alignment(w: np.ndarray) -> float:
    """Mean resultant length of the winding phase: 1 = all on whole windings, ~0 = unrelated."""
    if len(w) == 0:
        return float("nan")
    return float(np.abs(np.mean(np.exp(2j * np.pi * w))))


def layer_centres(w: np.ndarray, t: np.ndarray, angle_bins: int = 180, gap: float = 0.25, min_voxels: int = 3):
    """Per angle bin, the w of each layer crossing: predicted voxels sorted by w, split where w jumps by more than
    ``gap`` windings, clusters of at least ``min_voxels`` voxels kept."""
    out = []
    b = np.minimum((t * angle_bins).astype(int), angle_bins - 1)
    for k in range(angle_bins):
        s = np.sort(w[b == k])
        if len(s) < min_voxels:
            out.append(np.zeros(0))
            continue
        groups = np.split(s, np.nonzero(np.diff(s) > gap)[0] + 1)
        out.append(np.array([g.mean() for g in groups if len(g) >= min_voxels]))
    return out


def windings_per_layer(centres) -> float:
    steps = np.concatenate([np.diff(c) for c in centres if len(c) > 1] or [np.zeros(0)])
    return float(np.median(steps)) if len(steps) else float("nan")


def empty_windings(centres, first: int, last: int, tol: float = 0.2) -> float:
    """Share of (winding k in [first, last], angle bin) cells with no layer centre within ``tol`` of k."""
    ks = np.arange(first, last + 1)
    empty = total = 0
    for c in centres:
        if len(c) == 0:
            continue
        near = np.abs(c[None, :] - ks[:, None]).min(1) <= tol
        empty += int((~near).sum())
        total += len(ks)
    return float(empty / total) if total else float("nan")


def summarise(w, t, first, last, sectors: int = 12) -> dict:
    keep = (w >= first - 0.5) & (w <= last + 0.5)
    w, t = w[keep], t[keep]
    centres = layer_centres(w, t)
    res = {"predicted_voxels": int(len(w)), "alignment": alignment(w), "windings_per_layer": windings_per_layer(centres),
           "empty_windings": empty_windings(centres, first, last), "sectors": []}
    sec = np.minimum((t * sectors).astype(int), sectors - 1)
    for s in range(sectors):
        m = sec == s
        c = layer_centres(w[m], (t[m] * sectors) - s, angle_bins=max(1, 180 // sectors))
        res["sectors"].append({"from_deg": 360 * s / sectors, "to_deg": 360 * (s + 1) / sectors,
                               "predicted_voxels": int(m.sum()), "alignment": alignment(w[m]),
                               "windings_per_layer": windings_per_layer(c)})
    return res


def map_slice(transform, surface, z: int, threshold: int, samples: int, rng, batch: int = 20000) -> np.ndarray:
    sl = np.asarray(surface[int(z)])
    yy, xx = np.nonzero(sl >= threshold)
    if len(yy) == 0:
        return np.zeros((0, 3), np.float32)
    pick = rng.choice(len(yy), min(samples, len(yy)), replace=False)
    pts = np.stack([np.full(len(pick), z, np.float32), yy[pick].astype(np.float32), xx[pick].astype(np.float32)], 1)
    out = []
    with torch.inference_mode():
        for s in range(0, len(pts), batch):
            out.append(transform(torch.from_numpy(pts[s:s + batch])).cpu().numpy())
    return np.concatenate(out)


def open_prediction(path: str, group: str):
    import zarr
    node = zarr.open(path, mode="r")
    return node if hasattr(node, "shape") else node[group]


def figure(w, t, first, last, res, out_png, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keep = (w >= first - 0.5) & (w <= last + 0.5)
    w, t = w[keep], t[keep]
    mid = 0.5 * (first + last)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6), dpi=110, gridspec_kw={"width_ratios": [1.7, 1]})
    win = (w > mid - 8) & (w < mid + 8)
    ax[0].scatter(t[win] * 360, w[win], s=0.5, c="k", alpha=0.35, linewidths=0)
    for k in range(int(mid - 8), int(mid + 9)):
        ax[0].axhline(k, color="#b03a2e", lw=0.4)
    ax[0].set_xlim(0, 360)
    ax[0].set_ylim(mid - 8, mid + 8)
    ax[0].set_xlabel("spiral angle (degrees)")
    ax[0].set_ylabel("fractional winding number w (red: windings)")
    ax[0].set_title(f"{title}\nsurface-prediction voxels in the fit's spiral space", fontsize=9)
    s = res["sectors"]
    x = [0.5 * (q["from_deg"] + q["to_deg"]) for q in s]
    ax[1].bar(x, [q["alignment"] for q in s], width=24, color="#1b6ca8", label="alignment")
    ax2 = ax[1].twinx()
    ax2.plot(x, [q["windings_per_layer"] for q in s], "o-", color="#b03a2e", label="windings per layer")
    ax2.axhline(1.0, color="#b03a2e", lw=0.6, ls=":")
    ax[1].set_xlabel("spiral angle sector (degrees)")
    ax[1].set_ylabel("alignment (blue)")
    ax2.set_ylabel("windings per layer (red)")
    ax[1].set_title(f"alignment {res['alignment']:.2f}, windings per layer {res['windings_per_layer']:.2f}, "
                    f"empty windings {res['empty_windings']:.0%}", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--umbilicus", type=Path, required=True)
    ap.add_argument("--surface", required=True, help="surface prediction zarr (array or group with --group)")
    ap.add_argument("--group", default="0")
    ap.add_argument("--z", type=int, nargs="+", required=True)
    ap.add_argument("--first", type=int, help="first fitted winding (default: the checkpoint's first output winding)")
    ap.add_argument("--last", type=int, help="last fitted winding (default: shell_outer_winding_idx - 1)")
    ap.add_argument("--threshold", type=int, default=128)
    ap.add_argument("--samples", type=int, default=300000, help="surface-prediction voxels sampled per slice")
    ap.add_argument("--out-json", type=Path)
    ap.add_argument("--out-png", type=Path)
    a = ap.parse_args(argv)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    checkpoint = load_checkpoint_cpu(str(a.checkpoint))
    config = _checkpoint_config(checkpoint)
    model = _build_model(checkpoint, config, a.umbilicus, torch.device("cpu"))
    transform = model.get_slice_to_spiral_transform()
    dr = float(model.get_dr_per_winding())
    first = a.first if a.first is not None else int(checkpoint.get(
        "preview_first_winding", config.get("output_first_winding", 10)))
    last = a.last if a.last is not None else int(
        (config.get("shell_outer_winding_idx") or config["model_gap_expander_num_windings"]) - 1)
    surface = open_prediction(a.surface, a.group)
    rng = np.random.default_rng(0)
    report = {"checkpoint": str(a.checkpoint), "dr_per_winding": dr, "first": first, "last": last, "slices": []}
    all_w, all_t = [], []
    for z in a.z:
        sp = map_slice(transform, surface, z, a.threshold, a.samples, rng)
        w, t = winding_coordinates(sp, dr)
        res = {"z": int(z), **summarise(w, t, first, last)}
        report["slices"].append(res)
        all_w.append(w)
        all_t.append(t)
        print(f"z {z}: {res['predicted_voxels']} predicted voxels in w{first:03d}-w{last:03d}; alignment {res['alignment']:.2f}, "
              f"windings per layer {res['windings_per_layer']:.2f}, empty windings {res['empty_windings']:.0%}; "
              f"worst sector alignment {min(q['alignment'] for q in res['sectors']):.2f}", flush=True)
    if a.out_json:
        a.out_json.write_text(json.dumps(report, indent=1))
    if a.out_png and all_w:
        w, t = np.concatenate(all_w), np.concatenate(all_t)
        figure(w, t, first, last, summarise(w, t, first, last), a.out_png,
               f"{a.checkpoint.parent.name}, z {', '.join(map(str, a.z))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
