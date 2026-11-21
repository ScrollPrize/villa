#!/usr/bin/env python3
"""Give every piece of a stitched tifxyz the same normal direction.

A tifxyz assembled from separately unrolled pieces (for example the HerculeX
whole-scroll sheets) can hold pieces of both handednesses. vc_render_tifxyz
samples along the grid normal N = dP/dcol x dP/drow, so with one
--flip-normals setting some pieces are read from the side facing the scroll
centre and the rest from the side facing away, with their layers reversed.

This splits the surface into pieces (4-connected vertices, cut at grid edges
longer than --max-edge grid steps), decides from the umbilicus whether each
piece's N points away from the scroll axis, and writes

  OUT           the pieces whose N points away from the axis, and every vertex
                whose piece could not be decided, unchanged;
  OUT_mirrored  the pieces whose N points toward the axis, in a grid of the
                same size with its columns reversed (column c holds input
                column W - 1 - c), which reverses their N.

N away from the axis is how published segments on right-handed volumes are
stored; they render with --flip-normals (see that option in vc_render_tifxyz).
--toward-axis writes the opposite. Vertices are copied, never recomputed.
--report measures the input and writes nothing.

    python -m vesuvius.scripts.orient_tifxyz_pieces IN OUT --umbilicus umbilicus.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from vc_reader import tifxyz
from vc_reader.geometry import label_components

STRIP_VERTICES = 1 << 20


def _strips(rows, cols):
    step = max(8, STRIP_VERTICES // max(cols, 1))
    for r0 in range(0, rows, step):
        yield r0, min(rows, r0 + step)


def long_edge_vertices(xyz, valid, max_len):
    """Vertices with an edge longer than max_len (voxels) to a valid 4-neighbour."""
    bad = np.zeros(valid.shape, bool)
    limit = np.float32(max_len) ** 2
    rows = valid.shape[0]
    for r0, r1 in _strips(*valid.shape):
        e1 = min(rows, r1 + 1)
        for a, b in (np.s_[r0:r1, 1:], np.s_[r0:r1, :-1]), (np.s_[r0 + 1:e1], np.s_[r0:e1 - 1]):
            d = xyz[a] - xyz[b]
            long = valid[a] & valid[b] & ((d * d).sum(-1) > limit)
            bad[a] |= long
            bad[b] |= long
    return bad


def label_pieces(xyz, valid, max_len):
    """(labels, count): 4-connected pieces after cutting long edges. A vertex with
    a long edge joins the one piece it reaches over its short edges, if any."""
    seam = long_edge_vertices(xyz, valid, max_len)
    labels, count = label_components(valid & ~seam, connectivity=4)
    r, c = np.nonzero(seam)
    rows, cols = valid.shape
    limit = np.float32(max_len) ** 2
    near = []  # each 4-neighbour's label over a short edge; off the grid, clip lands on the (unlabelled) vertex
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rn, cn = np.clip(r + dr, 0, rows - 1), np.clip(c + dc, 0, cols - 1)
        d = xyz[rn, cn] - xyz[r, c]
        near.append(np.where((d * d).sum(-1) <= limit, labels[rn, cn], 0))
    near = np.stack(near)
    low = np.where(near > 0, near, near.max(0)).min(0)
    labels[r, c] = np.where(near.max(0) == low, low, 0)  # one distinct piece, else 0
    return labels, count


def grid_normals(xyz, valid, max_len=None):
    """vc_render_tifxyz's direction: unit (P[r,c+1]-P[r,c-1]) x (P[r+1,c]-P[r-1,c]);
    NaN on the border, where a 4-neighbour is invalid, and (with max_len) where
    either difference spans more than 2 * max_len voxels."""
    n = np.full(xyz.shape, np.nan, np.float32)
    du, dv = xyz[1:-1, 2:] - xyz[1:-1, :-2], xyz[2:, 1:-1] - xyz[:-2, 1:-1]
    c = np.cross(du, dv)
    ok = valid[1:-1, 2:] & valid[1:-1, :-2] & valid[2:, 1:-1] & valid[:-2, 1:-1] & valid[1:-1, 1:-1]
    if max_len is not None:
        limit = np.float32(2 * max_len) ** 2
        ok &= ((du * du).sum(-1) <= limit) & ((dv * dv).sum(-1) <= limit)
    del du, dv
    length = np.sqrt((c * c).sum(-1))
    ok &= length > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        c /= length[..., None]
    c[~ok] = np.nan
    n[1:-1, 1:-1] = c
    return n


def read_axis(spec):
    """centre(z) -> (x, y): a JSON with "control_points" [{x, y, z}, ...] (the
    published umbilicus files), or a fixed "X,Y"."""
    path = Path(spec)
    if path.exists():
        points = sorted((float(p["z"]), float(p["x"]), float(p["y"])) for p in json.loads(path.read_text())["control_points"])
        z, x, y = (np.array(v) for v in zip(*points))
        return lambda q: (np.interp(q, z, x), np.interp(q, z, y))
    x, y = (float(v) for v in spec.split(","))
    return lambda q: (np.full(np.shape(q), x), np.full(np.shape(q), y))


def radial_votes(xyz, valid, labels, count, centre, cos_min=0.8, r_min=300.0, max_len=None):
    """Per label 0..count: votes, and votes for N pointing away from the axis.
    A vertex votes when its N is within acos(cos_min) of the horizontal line to
    the axis and it lies more than r_min voxels from the axis; with max_len, not
    where its N spans a long edge (grid_normals)."""
    away = np.zeros(count + 1, np.int64)
    total = np.zeros(count + 1, np.int64)
    rows = valid.shape[0]
    for r0, r1 in _strips(*valid.shape):
        h0, h1 = max(0, r0 - 1), min(rows, r1 + 1)
        n = grid_normals(xyz[h0:h1], valid[h0:h1], max_len)[r0 - h0:r0 - h0 + r1 - r0]
        ok = np.isfinite(n).all(-1)
        p, n, lab = xyz[r0:r1][ok].astype(np.float64), n[ok], labels[r0:r1][ok]
        cx, cy = centre(p[:, 2])
        ox, oy = p[:, 0] - cx, p[:, 1] - cy
        r = np.hypot(ox, oy)
        cos = (n[:, 0] * ox + n[:, 1] * oy) / np.maximum(r, 1e-9)
        vote = (np.abs(cos) > cos_min) & (r > r_min)
        total += np.bincount(lab[vote], minlength=count + 1)
        away += np.bincount(lab[vote & (cos > 0)], minlength=count + 1)
    return away, total


def decide(away, total, min_votes=30, share=0.8):
    """+1 N away from the axis, -1 toward it, 0 undecided (too few or mixed votes)."""
    frac = away / np.maximum(total, 1)
    side = np.zeros(len(away), np.int8)
    enough = total >= min_votes
    side[enough & (frac >= share)] = 1
    side[enough & (frac <= 1 - share)] = -1
    side[0] = 0
    return side


def split(xyz, move):
    """Yields (grid, columns_reversed): the input with the vertices in `move` set
    to -1, then (if any) only those vertices, with the columns reversed. One grid
    is built at a time."""
    yield np.where(move[..., None], np.float32(-1), xyz), False
    if move.any():
        yield np.ascontiguousarray(np.where(move[..., None], xyz, np.float32(-1))[:, ::-1]), True


def away_share(xyz, valid, centre, cos_min=0.8, r_min=300.0, max_len=None):
    """(share of the voting vertices whose N points away from the axis, votes)."""
    away, total = radial_votes(xyz, valid, valid.astype(np.uint8), 1, centre, cos_min, r_min, max_len)
    return float(away[1] / max(total[1], 1)), int(total[1])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="tifxyz folder")
    ap.add_argument("output", nargs="?", help="output tifxyz folder; OUTPUT_mirrored is written beside it")
    ap.add_argument("--umbilicus", required=True, help='umbilicus JSON ("control_points") or "X,Y" in level-0 voxels')
    ap.add_argument("--max-edge", type=float, default=2.0, help="cut grid edges longer than this many grid steps")
    ap.add_argument("--min-votes", type=int, default=30)
    ap.add_argument("--share", type=float, default=0.8, help="vote share that decides a piece")
    ap.add_argument("--r-min", type=float, default=300.0, help="ignore vertices closer to the axis (voxels)")
    ap.add_argument("--toward-axis", action="store_true", help="make N point toward the axis instead")
    ap.add_argument("--report", action="store_true", help="measure only; write nothing")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    if not a.report and not a.output:
        ap.error("output is required unless --report")

    surface = tifxyz.read(a.input)
    xyz, valid = surface.xyz, surface.valid
    centre = read_axis(a.umbilicus)
    step = 1.0 / float(surface.scale[1])
    max_len = a.max_edge * step
    labels, count = label_pieces(xyz, valid, max_len)
    away, total = radial_votes(xyz, valid, labels, count, centre, r_min=a.r_min, max_len=max_len)
    side = decide(away, total, a.min_votes, a.share)
    size = np.bincount(labels.ravel(), minlength=count + 1)
    size[0] = 0
    before, votes_before = away_share(xyz, valid, centre, r_min=a.r_min, max_len=max_len)
    report = {
        "input": str(a.input),
        "pieces": {"away": int((side == 1).sum()), "toward": int((side == -1).sum()),
                   "undecided": int(((side == 0) & (size > 0)).sum())},
        "vertices": {"away": int(size[side == 1].sum()), "toward": int(size[side == -1].sum()),
                     "undecided": int(valid.sum() - size[side != 0].sum())},
        "away_share_before": before,
        "votes_before": votes_before,
    }
    if not a.report:
        move = (side == (1 if a.toward_axis else -1))[labels]
        del labels
        out = Path(a.output)
        record = {"source": str(a.input), "umbilicus": str(a.umbilicus), "normals": "toward_axis" if a.toward_axis else "away_from_axis",
                  "max_edge": a.max_edge, "min_votes": a.min_votes, "share": a.share, "r_min": a.r_min}
        shares = []
        for grid, columns_reversed in split(xyz, move):
            path = out.with_name(out.name + "_mirrored") if columns_reversed else out
            ok = grid[..., 0] != -1
            shares.append(away_share(grid, ok, centre, r_min=a.r_min, max_len=max_len))
            extra = dict(surface.extra) | {"name": path.name, "piece_orientation": record | {"columns_reversed": columns_reversed}}
            tifxyz.write(path, tifxyz.Surface(grid, scale=surface.scale, extra=extra), overwrite=a.overwrite)
            report.setdefault("written", []).append(str(path))
            del grid, ok
        votes = sum(n for _, n in shares)
        report["away_share_after"] = sum(s * n for s, n in shares) / max(votes, 1)
        report["votes_after"] = votes
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
