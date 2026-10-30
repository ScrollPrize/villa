"""Opt-in real-data check: growth that reaches the work-grid edge stops and saves.

Usage:
  python check_growpatch_grid_edge.py /path/to/vc_grow_seg_from_seed --output /new/output/directory

Needs network access (public PHerc0139 surface prediction, a few MB). Runs the
seed of #2104 with "generations": 0 (no limit), so growth fills the 50x50 seed
grid in ~25 generations and reaches its edge. Before the fix the run aborted
with "point too close to problem border!" (std::terminate on some builds) and
saved nothing.

Checks:
  - the binary exits with status 0 and reports that it reached the grid edge;
  - a surface is saved and no valid vertex lies in the outer two rows/columns.
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
SEED = ["3770", "3698", "4608"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    params = {"mode": "seed", "generations": 0, "step_size": 20, "voxelsize": 9.362,
              "min_area_cm": 0.0, "thread_limit": 1}
    params_path = args.output / "params.json"
    params_path.write_text(json.dumps(params, indent=2) + "\n")
    surface = args.output / "edge"
    command = [str(args.binary.resolve()), "-v", VOLUME, "-t", str(surface),
               "-p", str(params_path), "--segment-name", "edge", "-s", *SEED]
    log_path = args.output / "edge.log"
    with log_path.open("w") as log:
        result = subprocess.run(command, timeout=1800, stdout=log, stderr=subprocess.STDOUT,
                                env={**os.environ, "VC_GROWPATCH_RNG_SEED": "1"})
    log_text = log_path.read_text(errors="replace")
    assert result.returncode == 0, f"exit status {result.returncode}, see {log_path}"
    assert "reached the edge" in log_text, "growth did not reach the work-grid edge"

    xyz = np.stack([np.array(Image.open(surface / f"{axis}.tif")) for axis in "xyz"], axis=-1)
    valid = np.isfinite(xyz).all(axis=-1) & (xyz[..., 0] != -1)
    rows, cols = np.nonzero(valid)
    assert valid.sum() > 0, "no surface saved"
    assert rows.min() >= 2 and cols.min() >= 2, "vertex in the outer two rows/columns"
    assert rows.max() <= valid.shape[0] - 3 and cols.max() <= valid.shape[1] - 3, \
        "vertex in the outer two rows/columns"

    print(f"PASS: {int(valid.sum())} valid vertices saved on a {valid.shape[1]}x{valid.shape[0]} "
          f"grid; growth stopped at the edge")


if __name__ == "__main__":
    main()
