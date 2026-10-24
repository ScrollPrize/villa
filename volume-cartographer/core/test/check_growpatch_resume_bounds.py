"""Opt-in real PHerc0139 regression for #2104 (network access, numpy, Pillow).

Usage:
  python check_growpatch_resume_bounds.py /path/to/vc_grow_seg_from_seed \
      --normal-grids /path/to/cached/normal-grids --output /new/output/directory

The normal-grid directory must contain the published grids or a
normal-grids-remote.json pointing to the matching public normal-grids URL.
Use --seed-surface to reuse the bounded seed run from upstream. Otherwise the
script generates it, using the seed and spatial limits documented in #2104.
Downloads may take several minutes; this is deliberately not an offline CTest.

Checks, on a resume of that seed under the same limits:
  - growth escapes the imported grid and ends before the generation limit;
  - the grown surface does not fold back onto itself (a layer jump);
  - with disable_grid_expansion (what VC3D's Fill mode sends) nothing is placed
    outside the imported grid and the run still succeeds.
These are geometric checks, not a review of anatomical correctness.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image


VOLUME = (
    "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0139/"
    "representations/predictions/surfaces/"
    "20250728140407-surface-20260413222639-surface-m7-L0-th0.2.zarr"
)
GENERATIONS = 200


def load(surface):
    xyz = np.stack([np.array(Image.open(surface / f"{axis}.tif")) for axis in "xyz"], axis=-1)
    valid = np.isfinite(xyz).all(axis=-1) & (xyz[..., 0] != -1)
    return xyz.astype(np.float64), valid


def outside_imported(seed_valid, surface):
    # Output coordinates = input coordinates + grid_offset. Count valid
    # vertices outside the imported rectangle, not the padding rows/cols.
    _, valid = load(surface)
    dx, dy = json.loads((surface / "meta.json").read_text()).get("grid_offset", [0, 0])
    rows, cols = np.nonzero(valid)
    return int(((rows < dy) | (rows >= dy + seed_valid.shape[0]) |
                (cols < dx) | (cols >= dx + seed_valid.shape[1])).sum())


def closest_far_pair(surface, min_grid_sep=6):
    """Smallest 3D distance between vertices >= min_grid_sep cells apart on the grid."""
    xyz, valid = load(surface)
    idx = np.argwhere(valid)
    pts = xyz[valid]
    best = np.inf
    for s in range(0, len(pts), 512):
        d3 = np.linalg.norm(pts[s:s + 512, None] - pts[None], axis=-1)
        dg = np.abs(idx[s:s + 512, None] - idx[None]).max(axis=-1)
        best = min(best, float(np.where(dg >= min_grid_sep, d3, np.inf).min()))
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("--normal-grids", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed-surface", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    base = {
        "mode": "seed", "generations": GENERATIONS, "thread_limit": 1,
        "voxelsize": 9.362, "min_area_cm": 0.0,
        "x_min": 3072, "x_max": 4095,
        "y_min": 2944, "y_max": 3967,
        "z_min": 4352, "z_max": 4863,
        "normal_grid_path": str(args.normal_grids.resolve()),
    }
    fill = {**base, "disable_grid_expansion": True,
            "grow_max_extra_rows": 0, "grow_max_extra_cols": 0}

    def run(name, params, extra_args):
        params_path = args.output / (name + ".json")
        params_path.write_text(json.dumps(params, indent=2) + "\n")
        surface = args.output / name
        command = [str(args.binary.resolve()), "-v", VOLUME,
                   "-t", str(surface), "-p", str(params_path),
                   "--segment-name", name, *extra_args]
        with (args.output / (name + ".log")).open("w") as log:
            subprocess.run(command, check=True, timeout=1800, stdout=log,
                           stderr=subprocess.STDOUT,
                           env={**os.environ, "VC_GROWPATCH_RNG_SEED": "1"})
        return surface

    seed = args.seed_surface or run("seed", base, ["-s", "3770", "3698", "4608"])
    _, seed_valid = load(seed)

    resumed = run("resumed", base, ["--resume", str(seed.resolve())])
    _, valid = load(resumed)
    escaped = outside_imported(seed_valid, resumed)
    last_gen = json.loads((resumed / "meta.json").read_text())["max_gen"]
    assert valid.sum() > seed_valid.sum(), "Resume added no vertices"
    assert escaped > 0, "Resume remained trapped inside the imported grid"
    assert last_gen < GENERATIONS - 1, "Resume ran into the generation limit"
    gap = closest_far_pair(resumed)
    # Separate parts of one sheet stay ~100+ voxels apart here; a fold back
    # onto a neighbouring layer brings vertices within a few voxels.
    assert gap > 20.0, f"Resumed surface folds back onto itself ({gap:.1f} voxels)"

    # Fill-style resume must neither crash nor grow beyond the imported grid.
    filled = run("fill", fill, ["--resume", str(seed.resolve())])
    fill_escaped = outside_imported(seed_valid, filled)
    assert fill_escaped == 0, f"disable_grid_expansion resume grew {fill_escaped} vertices outside the grid"

    print(f"PASS: {seed_valid.sum()} -> {valid.sum()} valid vertices; {escaped} outside "
          f"the imported grid; stopped at generation {last_gen}; closest far pair "
          f"{gap:.1f} voxels; fill-style resume stayed inside")


if __name__ == "__main__":
    main()
