#!/usr/bin/env python3
"""Evaluate any surface prediction against the PHerc0139 physical truth labels.

Standalone: needs numpy, scipy, numcodecs. Usage:

  python3 eval_surface_pred.py LABELS.zarr PRED.zarr PRED_LEVEL

LABELS.zarr : the packaged truth (uint8 bit flags, L1 grid, see .zattrs)
PRED.zarr   : binary prediction volume on the SAME lo volume's grid
PRED_LEVEL  : pyramid level of PRED (0 or 1); level 0 is max-pooled to L1

Reports, each with a shifted-null control printed alongside:
  point recall at 1/2/3 L1 voxels (19/37/56 um)
  arc-level recall (1.2 mm sheet stretches, >=50% covered) and fully-missed
  recto-side overlap ratio (prediction mass on recto band vs verso side)
"""
import json
import sys
from pathlib import Path

import numcodecs
import numpy as np
from scipy import ndimage as ndi

NULL_SHIFT = 64


def read_chunk(root, comp, cz, cy, cx, ch, dtype=np.uint8):
    f = Path(root) / str(cz) / str(cy) / str(cx)
    if not f.is_file():
        return None
    raw = f.read_bytes()
    if comp is not None:
        raw = comp.decode(raw)
    a = np.frombuffer(raw, dtype)
    if a.size != ch ** 3:            # tolerate truncated edge chunks
        nz = a.size // (ch * ch)
        blk = np.zeros((ch, ch, ch), dtype)
        blk[:nz] = a.reshape(nz, ch, ch)
        return blk
    return a.reshape(ch, ch, ch)


def open_zarr(root):
    m = json.loads((Path(root) / ".zarray").read_text())
    comp = numcodecs.get_codec(m["compressor"]) if m["compressor"] else None
    return m, comp


def read_box(root, z0, z1, y0, y1, x0, x1):
    m, comp = open_zarr(root)
    ch = m["chunks"][0]
    out = np.zeros((z1 - z0, y1 - y0, x1 - x0), np.uint8)
    for cz in range(z0 // ch, -(-z1 // ch)):
        for cy in range(y0 // ch, -(-y1 // ch)):
            for cx in range(x0 // ch, -(-x1 // ch)):
                blk = read_chunk(root, comp, cz, cy, cx, ch)
                if blk is None:
                    continue
                az, ay, ax = cz * ch, cy * ch, cx * ch
                s = [slice(max(az, z0), min(az + ch, z1)),
                     slice(max(ay, y0), min(ay + ch, y1)),
                     slice(max(ax, x0), min(ax + ch, x1))]
                if any(sl.stop <= sl.start for sl in s):
                    continue
                out[s[0].start - z0:s[0].stop - z0,
                    s[1].start - y0:s[1].stop - y0,
                    s[2].start - x0:s[2].stop - x0] = \
                    blk[s[0].start - az:s[0].stop - az,
                        s[1].start - ay:s[1].stop - ay,
                        s[2].start - ax:s[2].stop - ax]
    return out


def main(labels_path, pred_path, pred_level):
    attrs = json.loads((Path(labels_path) / ".zattrs").read_text())
    oz, oy, ox = attrs["origin_l1"]
    meta, _ = open_zarr(labels_path)
    Z, Y, X = meta["shape"]
    acc = dict(n_ctr=0, hits={1: 0, 2: 0, 3: 0}, null2=0,
               n_arc=0, arc_hit=0, arc_gone=0, narc_hit=0, narc_gone=0,
               pred_on_recto=0, pred_on_verso=0)
    step = 96
    for z0 in range(0, Z, step):
        z1 = min(z0 + step, Z)
        lab = read_box(labels_path, z0, z1, 0, Y, 0, X)
        gz0, gz1 = oz + z0, oz + z1
        if pred_level == 1:
            pred = read_box(pred_path, gz0, gz1, oy, oy + Y, ox, ox + X) > 0
        else:
            p0 = read_box(pred_path, 2 * gz0, 2 * gz1, 2 * oy,
                          2 * (oy + Y), 2 * ox, 2 * (ox + X))
            pred = p0.reshape(z1 - z0, 2, Y, 2, X, 2).max((1, 3, 5)) > 0
            del p0
        valid = (lab & 1) > 0
        material = (lab & 2) > 0
        recto = (lab & 8) > 0
        for k in range(0, lab.shape[0], 4):
            tb = material[k]
            pb = pred[k] & valid[k]
            ctr = (lab[k] & 4) > 0
            if ctr.sum() < 500 or pb.sum() < 100:
                continue
            dpred = ndi.distance_transform_edt(~pb)
            ys, xs = np.nonzero(ctr)
            cd = dpred[ys, xs]
            acc["n_ctr"] += len(ys)
            for r in (1, 2, 3):
                acc["hits"][r] += int((cd <= r).sum())
            pbs = np.zeros_like(pb)
            pbs[NULL_SHIFT:] = pb[:-NULL_SHIFT]
            dnull = ndi.distance_transform_edt(~pbs)
            nd = dnull[ys, xs]
            acc["null2"] += int((nd <= 2).sum())
            lab2, _ = ndi.label(ctr, structure=np.ones((3, 3), int))
            lv = lab2[ys, xs]
            aid = (lv.astype(np.int64) * 10_000_000
                   + (ys // 64).astype(np.int64) * 2000 + xs // 64)
            _, inv = np.unique(aid, return_inverse=True)
            cnt = np.bincount(inv)
            cov = np.bincount(inv, weights=(cd <= 2)) / cnt
            ncov = np.bincount(inv, weights=(nd <= 2)) / cnt
            big = cnt >= 20
            acc["n_arc"] += int(big.sum())
            acc["arc_hit"] += int((cov[big] >= 0.5).sum())
            acc["arc_gone"] += int((big & (cov < 0.1)).sum())
            acc["narc_hit"] += int((ncov[big] >= 0.5).sum())
            acc["narc_gone"] += int((big & (ncov < 0.1)).sum())
            near = ndi.binary_dilation(tb, iterations=2)
            acc["pred_on_recto"] += int((pb & recto[k]).sum())
            acc["pred_on_verso"] += int(
                (pb & near & tb & ~recto[k]).sum())
        print(f"z {z0}-{z1} done", file=sys.stderr, flush=True)
    n = max(acc["n_ctr"], 1)
    na = max(acc["n_arc"], 1)
    side_n = max(acc["pred_on_recto"] + acc["pred_on_verso"], 1)
    out = dict(
        n_centerline=acc["n_ctr"],
        recall_19um=round(acc["hits"][1] / n, 4),
        recall_37um=round(acc["hits"][2] / n, 4),
        recall_56um=round(acc["hits"][3] / n, 4),
        null_recall_37um=round(acc["null2"] / n, 4),
        n_arcs=acc["n_arc"],
        arc_recall=round(acc["arc_hit"] / na, 4),
        arc_fully_missed=round(acc["arc_gone"] / na, 4),
        null_arc_recall=round(acc["narc_hit"] / na, 4),
        null_arc_fully_missed=round(acc["narc_gone"] / na, 4),
        recto_side_ratio=round(acc["pred_on_recto"] / side_n, 4))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
