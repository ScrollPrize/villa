"""Opt-in real-data check: an explicit seed outside the volume is refused.

Usage:
  python check_grow_seed_outside_volume.py /path/to/vc_grow_seg_from_seed --output /new/output/directory

Needs network access (public PHerc0139 surface prediction, the same volume as
check_growpatch_grid_edge.py). Before the fix, a seed outside the volume read as 0,
growth ran anyway and saved a surface whose every vertex lay outside the volume,
with exit status 0.

Checks:
  - seeds past the x edge and at negative x exit non-zero, say the seed is outside
    the volume, and save no surface;
  - the in-volume seed of check_growpatch_grid_edge.py still grows (2 generations)
    and saves a surface.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess


VOLUME = (
    "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0139/"
    "representations/predictions/surfaces/"
    "20250728140407-surface-20260413222639-surface-m7-L0-th0.2.zarr"
)
INSIDE = ["3770", "3698", "4608"]
OUTSIDE = {"past_x_edge": ["1000000", "3698", "4608"], "negative_x": ["-50", "3698", "4608"]}


def grow(binary, out, name, seed, params_path):
    surface = out / name
    command = [str(binary.resolve()), "-v", VOLUME, "-t", str(surface), "-p", str(params_path),
               "--segment-name", name, "-s", *seed]
    log_path = out / f"{name}.log"
    with log_path.open("w") as log:
        result = subprocess.run(command, timeout=1800, stdout=log, stderr=subprocess.STDOUT,
                                env={**os.environ, "VC_GROWPATCH_RNG_SEED": "1"})
    return result.returncode, log_path.read_text(errors="replace"), surface


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    params = {"mode": "seed", "generations": 2, "step_size": 20, "voxelsize": 9.362,
              "min_area_cm": 0.0, "thread_limit": 1}
    params_path = args.output / "params.json"
    params_path.write_text(json.dumps(params, indent=2) + "\n")

    for name, seed in OUTSIDE.items():
        rc, log, surface = grow(args.binary, args.output, name, seed, params_path)
        assert rc != 0, f"{name}: seed {seed} outside the volume exited 0"
        assert "outside the volume" in log, f"{name}: no 'outside the volume' message"
        assert not list(surface.rglob("x.tif")), f"{name}: a surface was saved"

    rc, log, surface = grow(args.binary, args.output, "inside", INSIDE, params_path)
    assert rc == 0, f"inside: exit status {rc}"
    assert list(surface.rglob("x.tif")), "inside: no surface saved"

    print("PASS: both outside seeds refused with no surface saved; the inside seed still grows")


if __name__ == "__main__":
    main()
