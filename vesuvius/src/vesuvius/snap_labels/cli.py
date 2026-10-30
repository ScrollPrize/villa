"""vesuvius.snap_labels: move one tifxyz surface label onto the papyrus recto face of the CT it lives in, and mark
every vertex where it could not.

    vesuvius.snap_labels LABEL.tifxyz --volume SCAN.zarr --out OUT
        (--umbilicus UMB.json | --axis-point X Y Z --axis-dir X Y Z | --recto-sign 1|-1)
        [--level 0] [--target recto|mid|verso] [--merged] [--max-move 4] [--face-sigma 1]
        [--no-label] [--label-level 0] [--ignore-halo 4] [--label-step 0.5]
        [--review-min 4] [--review-max 500] [--self-check] [--json]

Writes into OUT: snapped.tifxyz/, abstain.tif, reason.tif, confidence.tif, surface_label.zarr/, review.json,
review.tsv and snap_labels.json, then checks them (exit 1 when the outputs fail their own check, 2 on bad input).
A snapped label is geometry, not a reading.
"""

import argparse
import json
import sys
import time
from pathlib import Path

from . import face as S
from . import labels as L
from .meshio import MeshIOError, load_mesh, load_umbilicus
from .volume import Sampler, Volume, VolumeError


def build_parser():
    s = argparse.ArgumentParser(prog="vesuvius.snap_labels", description=__doc__.split("\n\n")[0])
    s.add_argument("surface", help="tifxyz folder (x.tif, y.tif, z.tif, meta.json) in the volume's level-0 voxels")
    s.add_argument("--volume", required=True, help="the CT the label lives in: OME-Zarr path or https URL")
    s.add_argument("--level", type=int, default=0, help="pyramid level the snap reads (coordinates stay level 0)")
    s.add_argument("--out", required=True, help="output folder (created; a folder with files in it is refused)")
    s.add_argument("--umbilicus", help="umbilicus file naming the scroll axis (JSON points, or z, y, x lines)")
    s.add_argument("--umbilicus-scale", type=float, default=1.0, help="multiply the umbilicus points by this")
    s.add_argument("--axis-point", nargs=3, type=float, metavar=("X", "Y", "Z"))
    s.add_argument("--axis-dir", nargs=3, type=float, metavar=("X", "Y", "Z"))
    s.add_argument("--recto-sign", type=float, choices=(1.0, -1.0),
                   help="+1: the recto face lies at negative offsets along the mesh normal; -1: at positive ones")
    s.add_argument("--target", choices=("recto", "mid", "verso"), default="recto")
    s.add_argument("--merged", action="store_true",
                   help="abstain where the band is over 1.8 times the mesh's usual thickness (two sheets merged)")
    s.add_argument("--face-sigma", type=float, default=L.DEFAULTS["face_sigma"])
    s.add_argument("--max-move", type=float, help="largest move, voxels (default 4)")
    s.add_argument("--no-label", action="store_true", help="skip the OME-Zarr surface label")
    s.add_argument("--label-level", type=int, default=0, help="the image level the label array matches")
    s.add_argument("--ignore-halo", type=float, help="ignore band along the normal, level-0 voxels (default max-move)")
    s.add_argument("--label-step", type=float, default=L.DEFAULTS["label_step"])
    s.add_argument("--review-min", type=int, default=L.DEFAULTS["review_min"])
    s.add_argument("--review-max", type=int, default=L.DEFAULTS["review_max"])
    s.add_argument("--self-check", action="store_true",
                   help="also plant a 2 voxel shift and recover it, with shifted and empty volume controls")
    s.add_argument("--json", action="store_true")
    return s


def axis_of(args):
    """The scroll axis from the arguments: ("umbilicus", points), ("line", point, direction) or None."""
    axis = None
    if args.umbilicus:
        try:
            pts = load_umbilicus(args.umbilicus, args.umbilicus_scale)
        except (MeshIOError, OSError) as exc:
            raise L.LabelSnapError(f"--umbilicus: {exc}") from exc
        axis = ("umbilicus", sorted(pts))
    elif args.axis_point is not None or args.axis_dir is not None:
        if args.axis_point is None or args.axis_dir is None:
            raise L.LabelSnapError("give --axis-point and --axis-dir together")
        axis = ("line", args.axis_point, args.axis_dir)
    if axis is not None and args.recto_sign is not None:
        raise L.LabelSnapError("give the scroll axis or --recto-sign, not both")
    if args.target != "mid" and axis is None and args.recto_sign is None:
        raise L.LabelSnapError("the recto face is named by its side toward the scroll axis: give --umbilicus, "
                               "--axis-point with --axis-dir, or --recto-sign")
    return axis


def run_snap(args):
    t0 = time.time()
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise L.LabelSnapError(f"{out} is not empty: write each run to a new folder")
    axis = axis_of(args)
    xs, ys, zs, valid = load_mesh(args.surface)
    vol = Volume(args.volume)
    sampler = Sampler(vol, level=args.level)
    extra = {"max_move": float(args.max_move)} if args.max_move is not None else None
    res = L.label_snap(xs, ys, zs, valid, sampler, axis=axis, recto_sign=args.recto_sign, target=args.target,
                       merged=bool(args.merged), face_sigma=args.face_sigma, extra_opts=extra)
    t_snap = time.time() - t0
    extra_rec = {}
    if args.self_check:
        def snapper(a, b, c, v, smp):
            return S.snap_face(a, b, c, v, smp, axis=axis, recto_sign=args.recto_sign, opts=res["opts"],
                               face_opts={"sigma": float(args.face_sigma)})
        extra_rec["self_check"] = S.self_check(xs, ys, zs, valid, sampler, snapper, S.DEFAULT_CHECK_SHIFT["face"],
                                               1.0, 1.0, 0.8, 3.0, "snapped")
    label_shape = None
    scale = 1.0
    if not args.no_label:
        if not 0 <= args.label_level < len(vol.levels):
            raise L.LabelSnapError(f"--label-level {args.label_level}: the volume has {len(vol.levels)} levels")
        label_shape = vol.shape(args.label_level)
        scale = L.level_scale(vol, args.label_level)
    desc = {"location": str(args.volume), "level": int(args.level), "shape_zyx": vol.shape(0),
            "axis": axis[0] if axis else None, "recto_sign": args.recto_sign}
    extra_rec["seconds"] = {"snap": round(t_snap, 2)}
    rec = L.write_outputs(out, args.surface, res, desc, label_shape, scale, args.label_level, args.ignore_halo,
                          args.label_step, args.review_min, args.review_max, extra_rec, xs, ys, zs)
    rec["seconds"]["total"] = round(time.time() - t0, 2)
    (out / "snap_labels.json").write_text(json.dumps(rec, indent=1, default=float))
    return rec


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        rec = run_snap(args)
    except (L.LabelSnapError, S.SnapError, MeshIOError, VolumeError) as exc:
        print(f"vesuvius.snap_labels: {exc}", file=sys.stderr)
        return 2
    probs = L.check_outputs(args.out)
    if args.json:
        print(json.dumps(rec, indent=1, default=float))
    else:
        v = rec["vertices"]
        print(f"snap_labels: {v['measured']} of {v['valid']} vertices measured their {rec['opts']['target']} face and "
              f"moved onto it, {v['abstained']} abstained ({v['frac_abstained']}); {rec['review']['regions']} review "
              f"regions; wrote {args.out}")
        if "label" in rec:
            lb = rec["label"]
            print(f"label: {lb['voxels']['surface']} surface and {lb['voxels']['ignore']} ignore voxels in "
                  f"{lb['chunks_written']} chunks")
        if "self_check" in rec:
            print("self-check: " + ("passed" if rec["self_check"]["passed"] else "FAILED") + "; "
                  + "; ".join(rec["self_check"]["why"]))
    if probs:
        print("vesuvius.snap_labels: the outputs fail their own check:\n  " + "\n  ".join(probs), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
