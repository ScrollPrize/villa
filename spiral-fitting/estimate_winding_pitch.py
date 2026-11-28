"""Estimate a scroll's winding pitch from its surface prediction, before any fit.

Without winding-inference rays or human relative windings, a fit keeps the
winding count of its starting spiral: model_initial_dr_per_winding (16 voxels
by default) sets how many windings the fit puts between the umbilicus and a
given radius. Most scrolls are looser than 16 voxels per winding at 9 um.

On a few slices of the binarized surface prediction this walks outwards from
the umbilicus at every angle and takes the median distance between the centres
of neighbouring predicted sheets. On scrolls whose pitch people have fixed, the
median is 2-11 % above it (missed sheets make gaps wider, sheets met at an
angle make them longer). It beats the default start but is not the best one:
the fit's count scales as one over its start, and on PHerc. Paris 4 a start
above the sheet spacing counted best. Once a fit has been scored along one
traced sheet, --refit gives the start that counts that sheet right.

    .venv/bin/python estimate_winding_pitch.py \\
        --predictions /path/to/surface-prediction.zarr/0 \\
        --umbilicus /path/to/dataset/umbilicus.json --z 10150 10650

Prints the per-slice medians and the override to start a fit from, e.g.
{"model_initial_dr_per_winding": 18.5}. Give --scale when the prediction is
not in the fit's voxels (fit voxels per prediction voxel, e.g. 4 for a level-2
prediction of a level-0 fit); the umbilicus is read in the fit's voxels.

    .venv/bin/python estimate_winding_pitch.py --refit 18.5 1.14

prints the start to refit with, {"model_initial_dr_per_winding": 21.1}, from a
fit started at 18.5 that put 1.14 windings per turn along a traced sheet.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from extract_surface_tracks import open_predictions
from umbilicus import json_umbilicus_z_to_yx


def radial_gaps(sheet_mask, centre_yx, angles=720):
    """Distances between the centres of neighbouring sheets along radial lines
    from centre_yx, over all lines, in the mask's voxels."""
    height, width = sheet_mask.shape
    radius = np.arange(int(np.hypot(height, width)))
    gaps = []
    for angle in np.arange(angles) * (2 * np.pi / angles):
        y = np.rint(centre_yx[0] + radius * np.sin(angle)).astype(np.int64)
        x = np.rint(centre_yx[1] + radius * np.cos(angle)).astype(np.int64)
        inside = (y >= 0) & (y < height) & (x >= 0) & (x < width)
        line = np.zeros(len(radius), dtype=np.int8)
        line[inside] = sheet_mask[y[inside], x[inside]]
        edges = np.diff(np.concatenate([[0], line, [0]]))
        centres = (np.flatnonzero(edges == 1) + np.flatnonzero(edges == -1) - 1) / 2
        if len(centres) >= 2:
            gaps.append(np.diff(centres))
    return np.concatenate(gaps) if gaps else np.zeros(0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--predictions",
                        help="binarized surface-prediction zarr array: local path, http(s) or s3:// URL")
    parser.add_argument("--umbilicus", help="the fit's umbilicus.json")
    parser.add_argument("--z", type=int, nargs="+",
                        help="slices to read, in the fit's voxels (two or three across the band to fit)")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="fit voxels per prediction voxel (default 1)")
    parser.add_argument("--angles", type=int, default=720)
    parser.add_argument("--refit", type=float, nargs=2, metavar=("START", "WINDINGS_PER_TURN"),
                        help="skip the prediction: the start a fit used and its windings per turn along "
                             "a traced sheet; prints the start that counts that sheet right")
    args = parser.parse_args(argv)
    if args.refit:
        start, windings_per_turn = args.refit
        if start <= 0 or windings_per_turn <= 0:
            parser.error("--refit needs a positive start and windings per turn")
        print(json.dumps({"model_initial_dr_per_winding": round(start * windings_per_turn, 1)}))
        return
    if not (args.predictions and args.umbilicus and args.z):
        parser.error("--predictions, --umbilicus and --z are required unless --refit is given")

    predictions = open_predictions(args.predictions)
    z_to_yx = json_umbilicus_z_to_yx(args.umbilicus)
    medians = []
    for z in args.z:
        plane = int(round(z / args.scale))
        sheet_mask = np.asarray(predictions[plane]) > 0
        centre = np.asarray(z_to_yx(z), dtype=np.float64) / args.scale
        gaps = radial_gaps(sheet_mask, centre, args.angles) * args.scale
        if len(gaps) == 0:
            print(f"z {z}: no neighbouring sheets predicted, skipped")
            continue
        medians.append(float(np.median(gaps)))
        print(f"z {z}: median gap {medians[-1]:.1f} voxels over {len(gaps)} gaps "
              f"(25th-75th percentile {np.percentile(gaps, 25):.1f}-{np.percentile(gaps, 75):.1f})")
    if not medians:
        parser.error("no slice had two or more predicted sheets on a radial line")
    print(json.dumps({"model_initial_dr_per_winding": round(float(np.median(medians)), 1)}))


if __name__ == "__main__":
    main()
