#!/usr/bin/env python3
"""Merge per-winding `concat` tifxyz surfaces across multiple spiral-fit runs.

Given a parent folder containing several run subfolders -- each with
`.../meshes/<name>/concat/<winding>/{x,y,z}.tif` -- this resamples every run's grid
for a given winding onto a common resolution and combines them per-pixel
(median or mean) into a consensus tifxyz surface.

Why resample first: the column axis is a normalized angular (theta) parameter
sampled at run-dependent resolution, so the same winding has different widths per
run. The runs describe the *same* 3D sheet (across-sheet residual ~1 vx); the
width/parameterization differences are tangential. Resampling each run to a common
grid by normalized column index aligns them; the reducer then yields a robust
consensus surface that stays on the true sheet.

Invalid points use the -1 sentinel (z<=0 is also treated invalid, matching the
QuadSurface reader). Output format mirrors scripts/spiral/tifxyz.py:save_tifxyz.

Example:
    python merge_concat_runs.py /path/to/parent --out /path/to/merged --method median \\
        --exclude some_run_to_skip another_run
"""

import argparse
import glob
import json
import os

import cv2
import numpy as np
import tifffile


def concat_dir(run_dir):
    """Return the `.../meshes/*/concat` dir inside a run, or None."""
    hits = glob.glob(os.path.join(run_dir, "meshes", "*", "concat"))
    return hits[0] if hits else None


def find_runs(parent, exclude):
    """Immediate subfolders of `parent` that contain a meshes/*/concat dir."""
    excl = set(exclude or [])
    return sorted(
        d for d in glob.glob(os.path.join(parent, "*"))
        if os.path.isdir(d) and os.path.basename(d) not in excl and concat_dir(d)
    )


def load_xyz(winding_dir):
    """Load a winding folder as (H, W, 3) float32 in x,y,z order, -1 = invalid."""
    x = tifffile.imread(os.path.join(winding_dir, "x.tif")).astype(np.float32)
    y = tifffile.imread(os.path.join(winding_dir, "y.tif")).astype(np.float32)
    z = tifffile.imread(os.path.join(winding_dir, "z.tif")).astype(np.float32)
    return np.stack([x, y, z], axis=-1)


def invalid_mask(pts):
    return (
        ~np.isfinite(pts[..., 0])
        | ~np.isfinite(pts[..., 1])
        | ~np.isfinite(pts[..., 2])
        | (pts[..., 0] == -1.0)
        | (pts[..., 2] <= 0.0)
    )


def resample_to(pts, target_hw):
    """Linear-resample (H,W,3) points to target (H,W), re-masking invalid regions
    with -1 (same approach as scripts/repair_tifxyz_spacing.py:resample_points)."""
    th, tw = target_hw
    if pts.shape[:2] == (th, tw):
        out = pts.copy()
        inv = invalid_mask(pts)
        out[inv] = -1.0
        return out, ~inv
    out = cv2.resize(pts, (tw, th), interpolation=cv2.INTER_LINEAR)
    inv = invalid_mask(pts).astype(np.uint8) * 255
    inv_r = cv2.resize(inv, (tw, th), interpolation=cv2.INTER_NEAREST)
    inv_r = cv2.dilate(inv_r, np.ones((3, 3), np.uint8), iterations=1)
    bad = inv_r > 0
    out[bad] = -1.0
    return out, ~bad


def voxel_um_from_meta(meta_path):
    """Recover voxel size (um) from a source meta's area_cm2 / area_vx2."""
    with open(meta_path) as f:
        m = json.load(f)
    a2, c2 = m.get("area_vx2"), m.get("area_cm2")
    if a2 and c2:
        return float((c2 / a2 * 1.0e8) ** 0.5)
    return None


def step_from_meta(meta_path):
    """Recover grid step size (voxels/pixel) from a source meta's scale."""
    with open(meta_path) as f:
        m = json.load(f)
    scale = m.get("scale")
    if scale and scale[0]:
        return 1.0 / float(scale[0])
    return None


def save_tifxyz(pts, out_winding_dir, uuid, step_size, voxel_um, source):
    """Write x/y/z.tif + meta.json, mirroring scripts/spiral/tifxyz.py:save_tifxyz.
    `pts` is (H,W,3) in x,y,z order with -1 for invalid."""
    os.makedirs(out_winding_dir, exist_ok=True)
    tifffile.imwrite(os.path.join(out_winding_dir, "x.tif"), pts[..., 0].astype(np.float32))
    tifffile.imwrite(os.path.join(out_winding_dir, "y.tif"), pts[..., 1].astype(np.float32))
    tifffile.imwrite(os.path.join(out_winding_dir, "z.tif"), pts[..., 2].astype(np.float32))
    valid_vertex = np.any(pts != -1, axis=-1)
    vq = (
        valid_vertex[:-1, :-1] & valid_vertex[1:, :-1]
        & valid_vertex[:-1, 1:] & valid_vertex[1:, 1:]
    )
    area_vx2 = int(vq.sum()) * step_size ** 2
    # bbox over all cells incl. -1 sentinel, in [z,y,x]-low/high order (matches save_tifxyz)
    lo = pts.min(axis=(0, 1))[::-1].tolist()
    hi = pts.max(axis=(0, 1))[::-1].tolist()
    meta = {
        "scale": [1.0 / step_size, 1.0 / step_size],
        "bbox": [lo, hi],
        "area_vx2": area_vx2,
        "area_cm2": area_vx2 * voxel_um ** 2 / 1.0e8 if voxel_um else None,
        "format": "tifxyz",
        "type": "seg",
        "uuid": uuid,
        "source": source,
    }
    with open(os.path.join(out_winding_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=4)
    return area_vx2


def pick_ref_run(runs, ref):
    """Choose the run that defines the output resolution."""
    if ref:
        match = next((d for d in runs if ref in os.path.basename(d)), None)
        if match is None:
            raise SystemExit(f"--ref '{ref}' matched no run in {[os.path.basename(d) for d in runs]}")
        return match
    return next((d for d in runs if os.path.basename(d).endswith("_baseline")), runs[0])


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="parent folder containing run subfolders")
    ap.add_argument("--out", required=True, help="output folder for merged tifxyz windings")
    ap.add_argument("--method", choices=["median", "mean"], default="median",
                    help="per-pixel reducer across runs (default: median)")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="run folder names (basenames) to skip")
    ap.add_argument("--ref", default=None,
                    help="substring selecting the run whose grid sets the output "
                         "resolution (default: a run ending in '_baseline', else the first)")
    ap.add_argument("--step-size", type=float, default=None,
                    help="grid step size in voxels (default: read from source meta, else 20)")
    args = ap.parse_args()

    reducer = np.nanmean if args.method == "mean" else np.nanmedian

    runs = find_runs(args.input, args.exclude)
    if not runs:
        raise SystemExit(f"No runs with meshes/*/concat found under {args.input}")
    ref_run = pick_ref_run(runs, args.ref)

    print(f"Merging {len(runs)} runs from {args.input}"
          + (f" (excluding {args.exclude})" if args.exclude else "") + ":")
    for d in runs:
        print(f"  - {os.path.basename(d)}{'   [ref resolution]' if d == ref_run else ''}")

    # step size and voxel size from a reference source meta (constant across runs/windings)
    ref_metas = glob.glob(os.path.join(concat_dir(ref_run), "*", "meta.json"))
    step_size = args.step_size or (step_from_meta(ref_metas[0]) if ref_metas else None) or 20.0
    voxel_um = voxel_um_from_meta(ref_metas[0]) if ref_metas else None
    print(f"\nmethod={args.method}  step_size={step_size}  voxel_size_um={voxel_um}")
    print(f"output -> {args.out}\n")

    windings = sorted(os.listdir(concat_dir(runs[0])))
    os.makedirs(args.out, exist_ok=True)

    for w in windings:
        wdirs = []
        for d in runs:
            p = os.path.join(concat_dir(d), w)
            if os.path.isdir(p) and os.path.exists(os.path.join(p, "x.tif")):
                wdirs.append((d, p))
        if not wdirs:
            print(f"{w}: no runs -> skipped")
            continue

        ref = next((p for d, p in wdirs if d == ref_run), wdirs[0][1])
        ref_shape = load_xyz(ref).shape[:2]

        stack, valids = [], []
        for d, p in wdirs:
            r, vmask = resample_to(load_xyz(p), ref_shape)
            r[~vmask] = np.nan  # so the reducer ignores invalid
            stack.append(r)
            valids.append(vmask)
        stack = np.stack(stack, axis=0)            # (R,H,W,3)
        n_valid = np.stack(valids, 0).sum(0)         # (H,W)

        with np.errstate(invalid="ignore"):
            merged = reducer(stack, axis=0)          # (H,W,3)
        merged[~np.isfinite(merged).all(axis=-1)] = -1.0
        merged = merged.astype(np.float32)

        area = save_tifxyz(merged, os.path.join(args.out, w), w, step_size, voxel_um,
                           f"merge_concat_runs {args.method} of {len(wdirs)} runs")
        n_pts = int((merged[..., 0] != -1).sum())
        print(f"{w}: runs={len(wdirs)} grid={ref_shape[0]}x{ref_shape[1]} "
              f"valid_pts={n_pts} max_overlap={int(n_valid.max())} area_vx2={area}")

    print(f"\nDone. Output: {args.out}")


if __name__ == "__main__":
    main()
