"""Check transform JSON files against their own landmarks.

A transform file stores a matrix and the landmark pairs it was fitted to. If the view is
moved after the last fit, the saved matrix drifts away from its landmarks without any error.
This prints, per file, the matrix residual, the residual of a least-squares affine fit to the
same landmarks, and that fit's leave-one-out error (all in fixed-volume voxels). It exits 1 if
any matrix is flagged as not matching its landmarks.

Example (villa #1843):
    python check_transform.py https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc1667/volumes/20260323082859-1.129um-0.2m-59keV-masked.zarr/transform.json

--refit OUT.json writes a copy of a single input whose matrix is replaced by the least-squares fit.
"""
import argparse
import json
import sys

import requests

from transform_utils import (
    LANDMARK_MISMATCH_TOLERANCE,
    fit_affine_transform_from_points,
    format_landmark_report,
    landmark_report,
)


def load(path_or_url: str) -> dict:
    if path_or_url.startswith(("http://", "https://")):
        r = requests.get(path_or_url, timeout=60)
        r.raise_for_status()
        return r.json()
    with open(path_or_url) as f:
        return json.load(f)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("transforms", nargs="+", help="transform JSON paths or http(s) URLs")
    ap.add_argument(
        "--tolerance",
        type=float,
        default=LANDMARK_MISMATCH_TOLERANCE,
        help="flag when the matrix RMS exceeds the least-squares RMS by more than this many voxels",
    )
    ap.add_argument("--refit", metavar="OUT.json", help="write the least-squares refit of a single input")
    a = ap.parse_args(argv)
    if a.refit and len(a.transforms) != 1:
        ap.error("--refit takes exactly one input transform")

    flagged = False
    for path in a.transforms:
        data = load(path)
        fixed = data.get("fixed_landmarks") or []
        moving = data.get("moving_landmarks") or []
        rep = landmark_report(data["transformation_matrix"], fixed, moving, a.tolerance)
        flagged |= rep["flagged"]
        print(f"{path}\n  " + format_landmark_report(rep).replace("\n", "\n  "))
        if a.refit:
            if rep["lsq_rms"] is None:
                ap.error("--refit needs at least 4 landmark pairs")
            out = dict(data)
            out["transformation_matrix"] = fit_affine_transform_from_points(fixed, moving)[:3].tolist()
            with open(a.refit, "w") as f:
                json.dump(out, f, indent=2)
            print(f"  wrote least-squares refit to {a.refit}")
    return 1 if flagged else 0


if __name__ == "__main__":
    sys.exit(main())
