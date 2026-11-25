"""Opt-in real-data check: time_limit_s keeps partial results and never deletes an existing surface.

Usage:
  python check_growpatch_time_limit.py /path/to/vc_grow_seg_from_seed --output /new/output/directory

Needs network access (public PHerc0139 surface prediction, a few MB). Uses the
seed of check_growpatch_grid_edge.py with "generations": 0 (no limit).

Checks:
  - default cutoff: a trace stopped by time_limit_s exits 0 and saves its
    surface with stop_reason "time_limit", even when its area is below the
    default min_area_cm;
  - explicit cutoff: with --segment-name naming an existing surface, snapshots
    every generation and a min_area_cm the result cannot reach, the run rejects
    the new surface and leaves the existing one byte for byte, with no staging
    directory left behind.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


VOLUME = (
    "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0139/"
    "representations/predictions/surfaces/"
    "20250728140407-surface-20260413222639-surface-m7-L0-th0.2.zarr"
)
SEED = ["3770", "3698", "4608"]
BASE = {"mode": "seed", "generations": 0, "step_size": 20, "voxelsize": 9.362,
        "thread_limit": 1, "time_limit_s": 4}


def run(binary, output, name, params):
    params_path = output / f"{name}.json"
    params_path.write_text(json.dumps(params, indent=2) + "\n")
    command = [str(binary.resolve()), "-v", VOLUME, "-t", str(output / name),
               "-p", str(params_path), "--segment-name", name, "-s", *SEED]
    log_path = output / f"{name}.log"
    with log_path.open("w") as log:
        result = subprocess.run(command, timeout=1800, stdout=log, stderr=subprocess.STDOUT,
                                env={**os.environ, "VC_GROWPATCH_RNG_SEED": "1"})
    return result.returncode, log_path.read_text(errors="replace"), log_path


def digest(directory):
    files = sorted(p for p in directory.rglob("*") if p.is_file())
    return {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    # 1. Default min_area_cm: the timed stop is saved.
    code, log, log_path = run(args.binary, args.output, "timed", dict(BASE))
    assert code == 0, f"exit status {code}, see {log_path}"
    meta = json.loads((args.output / "timed" / "meta.json").read_text())
    assert meta["stop_reason"] == "time_limit", f"stop_reason {meta['stop_reason']!r}, see {log_path}"
    area = float(meta["area_cm2"])
    if area < 0.3:
        assert "keeping the surface stopped by time_limit_s" in log, f"no keep message, see {log_path}"
    print(f"timed stop saved: area_cm2 {area:.4f}, stop_reason time_limit")

    # 2. Explicit cutoff with snapshots: the existing surface survives the rejection.
    existing = args.output / "existing"
    (args.output / "timed").rename(existing)
    before = digest(existing)
    params = dict(BASE, min_area_cm=1000.0)
    params["snapshot-interval"] = 1
    code, log, log_path = run(args.binary, args.output, "existing", params)
    assert "discarding generated surface" in log, f"result was not rejected, see {log_path}"
    assert "saved snapshot in" in log, f"no snapshot was written, see {log_path}"
    assert existing.is_dir(), "existing surface was deleted"
    assert digest(existing) == before, "existing surface was changed"
    leftovers = [p.name for p in args.output.iterdir() if p.name.endswith(".tracing")]
    assert not leftovers, f"staging directories left behind: {leftovers}"
    print("existing surface kept unchanged after an explicit cutoff rejected the new trace")
    print("PASS")


if __name__ == "__main__":
    main()
