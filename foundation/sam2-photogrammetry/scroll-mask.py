#!/usr/bin/env python
"""
Binary Scroll Masks from segment.py Output

segment.py labels every object found on the first frame of an acquisition, ordered by
brightness, so the scroll is not reliably label 1 (dust specks, ArUco markers and the
turntable can be darker). For each acquisition directory (one containing Masks/), this
script identifies the scroll label from its behaviour across the turntable rotation and
writes binary masks for reconstruction:

  1. On every --sample-step'th mask, collect per-label area, centroid and border contact.
  2. Candidates are labels present in every sampled frame, never touching the image
     border, with median area in [--min-area, --max-area].
  3. The scroll is the candidate whose centroid moves least: it sits on the rotation
     axis, while markers orbit it.
  4. Guards: the pick must move less than --max-motion, and the next candidate must move
     at least --min-margin times more (a scroll split into two persistent labels gives two
     near-static candidates). No other non-border label of at least --min-split-area of
     the scroll area may have more than --max-split-contact of its outline on the scroll
     (an embedded piece of the scroll; markers hidden behind it touch along one edge).
     Directories failing a guard are reported and skipped unless --force is given.
  5. Each mask is written as <acquisition>/ScrollMasks/<stem>_mask.png, 255 for the
     largest connected component of the scroll label and 0 elsewhere.

Usage:
    python scroll-mask.py /path/to/root_folder [--workers N]
"""

import os
import argparse
import multiprocessing
from functools import partial

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion
from skimage.measure import label
from tqdm import tqdm


def find_acquisition_dirs(root_folder):
    """
    Recursively collects directories containing a Masks/ subfolder.
    """
    return sorted(root for root, dirs, _ in os.walk(root_folder) if "Masks" in dirs)


def label_statistics(mask_files):
    """
    Per-label area fraction, normalised centroid and border contact over the given masks.
    """
    stats = {}
    for mask_file in mask_files:
        arr = np.array(Image.open(mask_file))
        height, width = arr.shape
        border = set(np.unique(np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])).tolist())
        for lbl in np.unique(arr):
            if lbl == 0:
                continue
            ys, xs = np.nonzero(arr == lbl)
            s = stats.setdefault(int(lbl), dict(area=[], cx=[], cy=[], border=0))
            s["area"].append(len(ys) / arr.size)
            s["cx"].append(xs.mean() / width)
            s["cy"].append(ys.mean() / height)
            s["border"] += int(lbl) in border
    return stats


def bbox_slices(mask, pad, shape):
    ys, xs = np.nonzero(mask)
    return (slice(max(ys.min() - pad, 0), min(ys.max() + pad + 1, shape[0])),
            slice(max(xs.min() - pad, 0), min(xs.max() + pad + 1, shape[1])))


def split_contact(mask_file, scroll_label, ring_width, min_split_area):
    """
    Largest fraction of another label's outline lying within ring_width of the scroll, over
    labels next to the scroll that do not touch the image border (i.e. not wall/turntable)
    and are at least min_split_area of the scroll's area. A piece of the scroll is mostly
    bordered by scroll; a marker partly hidden behind the scroll touches it along one edge.
    """
    arr = np.array(Image.open(mask_file))
    scroll = arr == scroll_label
    if not scroll.any():
        return 0.0
    border = set(np.unique(np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])).tolist())
    pad = ring_width + 1
    near = bbox_slices(scroll, pad, arr.shape)
    near_scroll = binary_dilation(scroll[near], iterations=ring_width)
    worst = 0.0
    for lbl in np.unique(arr[near][near_scroll]):
        if lbl == 0 or lbl == scroll_label or lbl in border:
            continue
        other = arr == lbl
        if other.sum() < min_split_area * scroll.sum():
            continue
        box = bbox_slices(other, pad, arr.shape)
        outline = other[box] & ~binary_erosion(other[box])
        touching = outline & binary_dilation(scroll[box], iterations=ring_width)
        worst = max(worst, float(touching.sum() / outline.sum()))
    return worst


def pick_scroll_label(acq_dir, args):
    """
    Returns (label, report dict, list of guard failures) for one acquisition directory.
    """
    mask_dir = os.path.join(acq_dir, "Masks")
    mask_files = sorted(os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith("_mask.png"))
    sampled = mask_files[::args.sample_step]
    stats = label_statistics(sampled)

    candidates = []
    for lbl, s in stats.items():
        area = float(np.median(s["area"]))
        if len(s["area"]) == len(sampled) and s["border"] == 0 and args.min_area <= area <= args.max_area:
            motion = float(np.hypot(np.std(s["cx"]), np.std(s["cy"])))
            candidates.append(dict(label=lbl, area=area, motion=motion))
    candidates.sort(key=lambda c: c["motion"])

    if not candidates:
        return None, dict(masks=len(mask_files)), ["no candidate label"]

    pick = candidates[0]
    margin = candidates[1]["motion"] / max(pick["motion"], 1e-9) if len(candidates) > 1 else float("inf")
    contact = max(split_contact(f, pick["label"], args.ring_width, args.min_split_area) for f in sampled)

    failures = []
    if pick["motion"] > args.max_motion:
        failures.append(f"centroid motion {pick['motion']:.4f} > {args.max_motion}")
    if margin < args.min_margin:
        failures.append(f"next candidate moves only {margin:.1f}x more (< {args.min_margin})")
    if contact > args.max_split_contact:
        failures.append(f"a neighbouring label has {contact:.2f} of its outline on the scroll (> {args.max_split_contact})")

    report = dict(masks=len(mask_files), label=pick["label"], area=pick["area"], motion=pick["motion"],
                  margin=margin, contact=contact, candidates=len(candidates))
    return pick["label"], report, failures


def write_scroll_mask(task):
    """
    Writes the binary mask (largest connected component of the scroll label) for one frame.
    """
    mask_file, out_file, scroll_label = task
    arr = np.array(Image.open(mask_file))
    labeled = label(arr == scroll_label, connectivity=2)
    out = np.zeros(arr.shape, dtype=np.uint8)
    if labeled.max() > 0:
        counts = np.bincount(labeled.ravel())
        counts[0] = 0
        out[labeled == counts.argmax()] = 255
    Image.fromarray(out, mode="L").save(out_file)
    return int(labeled.max())


def main():
    parser = argparse.ArgumentParser(
        description="Identify the scroll label in segment.py masks and write binary ScrollMasks/ for reconstruction."
    )
    parser.add_argument("root_folder", help="Root directory containing acquisition folders with Masks/.")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel workers for writing masks (default: 4; each holds a full-resolution frame).")
    parser.add_argument("--sample-step", type=int, default=4,
                        help="Use every Nth mask for label statistics (default: 4).")
    parser.add_argument("--min-area", type=float, default=0.01,
                        help="Minimum median area fraction for a scroll candidate (default: 0.01).")
    parser.add_argument("--max-area", type=float, default=0.3,
                        help="Maximum median area fraction for a scroll candidate (default: 0.3).")
    parser.add_argument("--max-motion", type=float, default=0.05,
                        help="Maximum normalised centroid motion of the scroll (default: 0.05).")
    parser.add_argument("--min-margin", type=float, default=3.0,
                        help="Minimum motion ratio between next candidate and scroll (default: 3).")
    parser.add_argument("--ring-width", type=int, default=5,
                        help="Distance in pixels counted as touching the scroll (default: 5).")
    parser.add_argument("--min-split-area", type=float, default=0.01,
                        help="Ignore neighbouring labels smaller than this fraction of the scroll area (default: 0.01).")
    parser.add_argument("--max-split-contact", type=float, default=0.6,
                        help="Maximum outline fraction of a neighbouring label touching the scroll (default: 0.6).")
    parser.add_argument("--force", action="store_true",
                        help="Write masks even for directories that fail a guard.")
    args = parser.parse_args()

    acq_dirs = find_acquisition_dirs(args.root_folder)
    if not acq_dirs:
        print("No directories with Masks/ found.")
        return

    tasks, failed = [], []
    for acq_dir in acq_dirs:
        scroll_label, report, failures = pick_scroll_label(acq_dir, args)
        status = "FAIL: " + "; ".join(failures) if failures else "ok"
        if scroll_label is None:
            print(f"{acq_dir}: {status}")
        else:
            print(f"{acq_dir}: scroll label {scroll_label} (area {report['area']:.3f}, "
                  f"motion {report['motion']:.4f}, next candidate x{report['margin']:.1f}, "
                  f"split contact {report['contact']:.2f}, candidates {report['candidates']}) {status}")
        if failures and not (args.force and scroll_label is not None):
            failed.append(acq_dir)
            continue
        out_dir = os.path.join(acq_dir, "ScrollMasks")
        os.makedirs(out_dir, exist_ok=True)
        for f in sorted(os.listdir(os.path.join(acq_dir, "Masks"))):
            if f.endswith("_mask.png"):
                tasks.append((os.path.join(acq_dir, "Masks", f), os.path.join(out_dir, f), scroll_label))

    with multiprocessing.Pool(args.workers) as pool:
        components = list(tqdm(pool.imap(write_scroll_mask, tasks, chunksize=4),
                               total=len(tasks), desc="Writing scroll masks", unit="file"))

    empty = [t[0] for t, n in zip(tasks, components) if n == 0]
    split = sum(n > 1 for n in components)
    print(f"Wrote {len(tasks)} masks; {split} had extra scroll-label components removed; {len(empty)} empty.")
    for f in empty:
        print(f"  empty: {f}")
    if failed:
        print("Skipped (guard failed): " + ", ".join(failed))


if __name__ == "__main__":
    main()
