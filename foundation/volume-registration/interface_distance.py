#!/usr/bin/env python3
"""Distance from 3D points to the nearest intensity interface in a second scan.

Given points expressed in the voxel frame of one volume (for example predicted
ink anchors, or the centre of a measured depth band on a segment), this tool
maps them through one or more ``transform.json`` files into an independently
scanned reference volume and measures, along a local normal, the distance to
the nearest sharp intensity change. The result says whether a point sits at an
interface that the second scan also sees. It does not say which side of a sheet
the interface is, and it says nothing about ink.

Typical use: points in the canonical 2.4um PHerc. Paris 4 frame checked against
the 1.129um rescan, whose ``transform.json`` registers the 1.129um volume
(moving) to the canonical volume (fixed), so the chain is one inverse step::

    python interface_distance.py \
        --points w00_depth_anchors.csv \
        --step inv:transform_1129um.json \
        --reference https://.../PHercParis4-1.129um.zarr/0 \
        --output w00_interface_distance.json

Chain steps are applied in order. ``PATH`` applies p_fixed = M @ p_moving as
written in the file; ``inv:PATH`` applies the inverse. Distances are reported in
voxels of the frame the points were given in.

Per point: a cubic window on the points' grid is resampled from the reference
through the chain, smoothed, and a profile is read along the normal. The normal
is either the one supplied with the point (``--normal points``) or the dominant
structure-tensor direction of the window (``--normal volume``). ``|d/ds|`` of
the profile is scanned for peaks with prominence at least ``--prominence`` of
the maximum; ``D`` is the smallest ``|s|`` among them. A point is evaluable
when enough of its window lies inside the reference, the normal is defined,
and at least one qualifying peak exists.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.signal import find_peaks

try:  # reuse the schema-validating reader when its dependencies are present
    from transform_utils import read_transform_json
except ImportError:  # pragma: no cover - exercised only without SimpleITK
    def read_transform_json(input_path, invert=False):
        with open(input_path) as f:
            data = json.load(f)
        m = np.array(data["transformation_matrix"], dtype=float)
        if m.shape != (3, 4):
            raise ValueError(f"{input_path}: transformation_matrix must be 3x4")
        matrix = np.vstack([m, [0, 0, 0, 1]])
        fixed, moving = data["fixed_landmarks"], data["moving_landmarks"]
        if invert:
            matrix = np.linalg.inv(matrix)
            fixed, moving = moving, fixed
        return matrix, fixed, moving


# ----------------------------------------------------------------- transforms
def load_chain(steps):
    """Compose chain steps into one 4x4 XYZ affine (points frame -> reference)."""
    total = np.eye(4)
    desc = []
    for step in steps:
        invert = step.startswith("inv:")
        path = step[4:] if invert else step
        matrix, _, _ = read_transform_json(path, invert=invert)
        total = matrix @ total
        desc.append({"file": path, "inverse": invert})
    return total, desc


def map_points_zyx(pts_zyx, chain, offset_zyx=None):
    """Map (N,3) z,y,x points through an XYZ chain; returns (N,3) z,y,x."""
    p = np.atleast_2d(np.asarray(pts_zyx, dtype=float))
    if offset_zyx is not None:
        p = p + np.asarray(offset_zyx, dtype=float)
    xyz1 = np.c_[p[:, ::-1], np.ones(len(p))]
    out = xyz1 @ chain.T
    return out[:, :3][:, ::-1]


# --------------------------------------------------------------------- points
def _unit(v):
    n = np.linalg.norm(v)
    return None if not np.isfinite(n) or n < 1e-9 else v / n


def read_points(path, fmt="auto", include_block_median=False):
    """Yield dicts with id, zyx, optional normal_zyx, and passthrough fields.

    ``depth-anchors`` is the CSV written by khj1222's ``export_depth_anchors.py``
    (columns base_x/y/z, normal_x/y/z, offset_from_plane, sampled_from); the
    point is the band centre base + offset_from_plane * normal. ``xyz`` is a
    plain CSV with x,y,z columns and optional nx,ny,nz.
    """
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        if fmt == "auto":
            fmt = "depth-anchors" if "base_x" in header else "xyz"
        rows = list(reader)
    pts = []
    for i, r in enumerate(rows):
        if fmt == "depth-anchors":
            if not include_block_median and r.get("sampled_from", "center") != "center":
                continue
            base = np.array([float(r["base_x"]), float(r["base_y"]), float(r["base_z"])])
            n = _unit(np.array([float(r["normal_x"]), float(r["normal_y"]),
                                float(r["normal_z"])]))
            if n is None:
                continue
            xyz = base + float(r.get("offset_from_plane", 0.0)) * n
            pid = f"{r.get('cell_row', i)}_{r.get('cell_col', '')}".rstrip("_")
            extra = {k: r[k] for k in ("region", "sampled_from", "center_layer",
                                       "half_width_layers") if k in r}
        else:
            xyz = np.array([float(r["x"]), float(r["y"]), float(r["z"])])
            n = None
            if all(k in r and r[k] != "" for k in ("nx", "ny", "nz")):
                n = _unit(np.array([float(r["nx"]), float(r["ny"]), float(r["nz"])]))
            pid = r.get("id", str(i))
            extra = {}
        pts.append({"id": pid, "zyx": xyz[::-1].copy(),
                    "normal_zyx": None if n is None else n[::-1].copy(), **extra})
    return pts


def stride_subsample(items, cap):
    if cap is None or len(items) <= cap:
        return list(items)
    idx = sorted({(i * len(items)) // cap for i in range(cap)})
    return [items[i] for i in idx]


# ------------------------------------------------------------------ reference
def open_reference(path, level):
    import zarr  # local import so the parser and phantom tests work without it
    node = zarr.open(path, mode="r")
    if hasattr(node, "shape"):
        return node
    return node[level]


def read_block(arr, lo, hi):
    """Read arr[lo:hi] with zero padding where the box leaves the array."""
    shape = np.array(arr.shape)
    lo = np.asarray(lo, int)
    hi = np.asarray(hi, int)
    out = np.zeros(tuple(hi - lo), dtype=np.float32)
    clo = np.maximum(lo, 0)
    chi = np.minimum(hi, shape)
    if np.any(chi <= clo):
        return out, 0.0
    block = arr[clo[0]:chi[0], clo[1]:chi[1], clo[2]:chi[2]]
    o = clo - lo
    out[o[0]:o[0] + block.shape[0], o[1]:o[1] + block.shape[1],
        o[2]:o[2] + block.shape[2]] = block
    inside = float(np.prod(chi - clo) / np.prod(hi - lo))
    return out, inside


# ------------------------------------------------------------------- analysis
def structure_tensor_normal(window, sigma_grad, sigma_tensor, aniso_ratio):
    c = window.shape[0] // 2
    g = [gaussian_filter(window, sigma_grad, order=tuple(int(a == i) for a in range(3)))
         for i in range(3)]
    J = np.empty((3, 3))
    for i in range(3):
        for j in range(i, 3):
            J[i, j] = J[j, i] = gaussian_filter(g[i] * g[j], sigma_tensor)[c, c, c]
    w, v = np.linalg.eigh(J)
    lam1, lam2 = w[-1], w[-2]
    ok = bool(lam1 > 0 and lam1 >= aniso_ratio * lam2)
    return v[:, -1], ok, float(lam1), float(lam2)


def interface_distance(profile, half, prominence):
    """Smallest |s| of a qualifying |gradient| peak, or None."""
    g = np.abs(np.gradient(np.asarray(profile, float)))
    if g.max() <= 0:
        return None
    peaks, _ = find_peaks(g, prominence=prominence * g.max())
    if len(peaks) == 0:
        return None
    return float(np.min(np.abs(peaks - half)))


def analyze_point(pt, arr, chain, args):
    win = args.window
    half = args.profile_half
    n = 2 * win + 1
    rec = {"id": pt["id"], "point_zyx": [round(float(v), 3) for v in pt["zyx"]]}
    for k in ("region", "sampled_from", "center_layer", "half_width_layers"):
        if k in pt:
            rec[k] = pt[k]
    centre = map_points_zyx(pt["zyx"], chain, args.moving_offset_zyx)[0]
    rec["reference_zyx"] = [round(float(v), 3) for v in centre]

    # cheap rejection before resampling: the window's bounding box in the
    # reference must touch the array at all
    corners = pt["zyx"] + np.array([[dz, dy, dx] for dz in (-win, win)
                                    for dy in (-win, win) for dx in (-win, win)])
    qc = map_points_zyx(corners, chain, args.moving_offset_zyx)
    if np.any(qc.max(axis=0) < 0) or np.any(qc.min(axis=0) > np.array(arr.shape) - 1):
        rec.update({"window_inside_reference": 0.0, "masked_zero_fraction": 1.0,
                    "D_voxels": None, "evaluable": False,
                    "non_evaluable_reasons": ["outside_reference"]})
        return rec

    # the chain is affine, so the mapped corners bound the whole mapped window
    lo = np.floor(qc.min(axis=0)).astype(int) - 2
    hi = np.ceil(qc.max(axis=0)).astype(int) + 3
    block, inside = read_block(arr, lo, hi)
    if not block.any():  # masked or empty region: nothing to resample
        rec.update({"window_inside_reference": round(inside, 4),
                    "masked_zero_fraction": 1.0, "D_voxels": None,
                    "evaluable": False,
                    "non_evaluable_reasons": ["masked_zero_fraction_exceeds_cap"]})
        return rec

    off = np.stack(np.meshgrid(*(np.arange(n) - win,) * 3, indexing="ij"),
                   axis=-1).reshape(-1, 3)
    q = map_points_zyx(pt["zyx"] + off, chain, args.moving_offset_zyx)
    window = map_coordinates(block, (q - lo).T, order=1, mode="constant",
                             cval=0.0).reshape(n, n, n)
    smoothed = gaussian_filter(window, args.smooth_sigma)
    rec["window_inside_reference"] = round(inside, 4)
    reasons = []

    if args.normal == "points":
        normal = pt["normal_zyx"]
        if normal is None:
            reasons.append("no_normal_supplied")
        rec["normal_source"] = "points"
    else:
        normal, ok, lam1, lam2 = structure_tensor_normal(
            smoothed, args.sigma_grad, args.sigma_tensor, args.aniso_ratio)
        rec.update({"normal_source": "volume", "lambda1": lam1, "lambda2": lam2})
        if not ok:
            reasons.append("anisotropy_fail")

    # masked (zero) voxels are checked on what the mode actually uses: the
    # whole window when the normal comes from it, the raw profile otherwise
    D = None
    if normal is not None:
        rec["normal_zyx"] = [round(float(v), 4) for v in normal]
        s = np.arange(-half, half + 1, dtype=float)
        pts = win + s[:, None] * np.asarray(normal)[None, :]
        raw = map_coordinates(window, pts.T, order=1, mode="constant", cval=0.0)
        prof = map_coordinates(smoothed, pts.T, order=1, mode="constant", cval=0.0)
        zero_frac = float(np.mean(window < 0.5)) if args.normal == "volume" \
            else float(np.mean(raw < 0.5))
        rec["masked_zero_fraction"] = round(zero_frac, 4)
        if zero_frac > args.max_zero_fraction:
            reasons.append("masked_zero_fraction_exceeds_cap")
        rec["profile"] = [round(float(v), 2) for v in prof]
        D = interface_distance(prof, half, args.prominence)
        if D is None:
            reasons.append("no_qualifying_interface")
    rec.setdefault("masked_zero_fraction", None)
    rec["D_voxels"] = D
    rec["evaluable"] = not reasons
    rec["non_evaluable_reasons"] = reasons
    return rec


def wilson(k, n, z=1.959964):
    if n == 0:
        return None
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(centre - hw, 4), round(centre + hw, 4)]


def summarize(records, threshold):
    ev = [r for r in records if r["evaluable"]]
    d = [r["D_voxels"] for r in ev]
    k = sum(1 for x in d if x <= threshold)
    return {"n_points": len(records), "n_evaluable": len(ev),
            "threshold_voxels": threshold, "n_within_threshold": k,
            "fraction_within_threshold": round(k / len(ev), 4) if ev else None,
            "wilson95_fraction_within": wilson(k, len(ev)),
            "median_D_voxels": float(np.median(d)) if d else None,
            "non_evaluable_reasons": _count([x for r in records
                                             for x in r["non_evaluable_reasons"]])}


def _count(items):
    out = {}
    for x in items:
        out[x] = out.get(x, 0) + 1
    return out


# ------------------------------------------------------------------------ cli
def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog="\n\n".join(__doc__.split("\n\n")[1:]))
    p.add_argument("--points", required=True, help="CSV of points (see --points-format)")
    p.add_argument("--points-format", choices=["auto", "xyz", "depth-anchors"],
                   default="auto")
    p.add_argument("--include-block-median", action="store_true",
                   help="depth-anchors: keep rows with sampled_from != center")
    p.add_argument("--max-points", type=int, default=None,
                   help="deterministic stride subsample to at most this many points")
    p.add_argument("--step", action="append", default=[], metavar="[inv:]PATH",
                   help="transform.json chain step, applied in order (repeatable)")
    p.add_argument("--moving-offset-zyx", type=float, nargs=3, default=None,
                   help="constant z y x shift added to points before the chain "
                        "(for a measured residual registration offset)")
    p.add_argument("--reference", required=True, help="zarr array or group path/URL")
    p.add_argument("--reference-level", default="0",
                   help="array key when --reference is a group (default 0)")
    p.add_argument("--normal", choices=["points", "volume"], default="volume")
    p.add_argument("--window", type=int, default=34, help="half-window in point voxels")
    p.add_argument("--profile-half", type=int, default=24)
    p.add_argument("--smooth-sigma", type=float, default=2.0)
    p.add_argument("--sigma-grad", type=float, default=2.0)
    p.add_argument("--sigma-tensor", type=float, default=4.0)
    p.add_argument("--aniso-ratio", type=float, default=2.0)
    p.add_argument("--prominence", type=float, default=0.25)
    p.add_argument("--max-zero-fraction", type=float, default=0.10)
    p.add_argument("--threshold", type=float, default=3.0,
                   help="D at or below this counts as 'within' in the summary")
    p.add_argument("--output", required=True, help="result JSON")
    p.add_argument("--csv", default=None, help="optional per-point CSV")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not args.step:
        sys.exit("at least one --step is required (use an identity transform.json "
                 "if the points already live in the reference frame)")
    chain, chain_desc = load_chain(args.step)
    scale = abs(np.linalg.det(chain[:3, :3])) ** (1 / 3)
    pts = stride_subsample(read_points(args.points, args.points_format,
                                       args.include_block_median), args.max_points)
    arr = open_reference(args.reference, args.reference_level)
    records = []
    for i, pt in enumerate(pts, 1):
        records.append(analyze_point(pt, arr, chain, args))
        if i % 10 == 0 or i == len(pts):
            print(f"  {i}/{len(pts)}", file=sys.stderr, flush=True)
    summary = summarize(records, args.threshold)
    result = {"tool": "interface_distance", "chain": chain_desc,
              "chain_matrix_xyz": chain[:3].tolist(),
              "reference_voxels_per_point_voxel": round(float(scale), 6),
              "reference": args.reference, "reference_shape": list(arr.shape),
              "parameters": {k: v for k, v in vars(args).items()
                             if k not in ("output", "csv", "step", "reference")},
              "summary": summary, "records": records}
    Path(args.output).write_text(json.dumps(result, indent=1))
    if args.csv:
        keys = ["id", "evaluable", "D_voxels", "normal_source", "masked_zero_fraction",
                "window_inside_reference", "region", "sampled_from"]
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(keys + ["point_z", "point_y", "point_x", "reasons"])
            for r in records:
                w.writerow([r.get(k, "") for k in keys] + list(r["point_zyx"])
                           + [";".join(r["non_evaluable_reasons"])])
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
