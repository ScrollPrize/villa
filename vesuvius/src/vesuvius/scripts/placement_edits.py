#!/usr/bin/env python3
"""Record what an annotator did to the worst cells of a placement audit.

Give it the tifxyz segment the audit measured (``--before``), the same segment after it
was edited in VC3D (``--after``), and the audit's ``<name>.worst_cells.json``. For each
cell it writes how far the grid points moved along the cell's normal, next to the offset
the audit had measured there, so each edit becomes a training pair: the CT and the
surface as traced, and where a human put it. Nothing is trained here.

Writes under ``--out``:
  edits.csv        rank,z,y,x,n,median_offset,median_shift,n_moved,flagged
  edits.npz        before_<rank>, after_<rank> (zyx grid points of the cell), ct_<rank> with
                   --volume, pred_<rank> with --prediction (the surface model's output there)
  provenance.json  the inputs, the editor, the audit's own settings, the time

``--all-cells`` adds every other cell of the grid (``flagged`` 0, no measured offset), so a
model also sees the cells a human looked at and left alone; without it the set holds only
cells the audit flagged and a model would learn to always move. The CT cube is 48 voxels
so a 32-voxel cell fits inside it whatever its alignment.

A correct edit shows ``median_shift`` close to ``median_offset``: the surface moved by the
distance the audit measured. Rerun the audit on ``--after`` to confirm the cell is gone.
"""
import argparse
import csv
import getpass
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np

CELL = 32.0


def grid_points(path: str):
    """zyx grid points, validity, and unit normals (zyx) of a tifxyz segment."""
    from vesuvius.tifxyz import read_tifxyz

    s = read_tifxyz(path)
    x, y, z = s._x, s._y, s._z
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & (x > 0) & (y > 0) & (z > 0)
    if s._mask is not None:
        ok &= s._mask.astype(bool)
    nx, ny, nz = s.compute_normals()
    N = np.stack([nz, ny, nx], -1).astype(np.float64)
    N /= np.linalg.norm(N, axis=-1, keepdims=True) + 1e-12
    return np.stack([z, y, x], -1).astype(np.float64), ok, N


def all_cells(P: np.ndarray, ok: np.ndarray, N: np.ndarray, flagged: List[dict]) -> List[dict]:
    """Every grid cell not already in ``flagged``, with its mean position and normal."""
    taken = {tuple(np.floor(np.asarray(c["center_zyx"]) / CELL).astype(int)) for c in flagged}
    keys = np.floor(P[ok] / CELL).astype(int)
    pts, nrm = P[ok], N[ok]
    good = np.isfinite(nrm).all(-1)
    out = []
    for k in np.unique(keys, axis=0):
        if tuple(k) in taken:
            continue
        sel = np.all(keys == k, axis=-1) & good
        if sel.sum() < 30:
            continue
        n = nrm[sel].mean(0)
        out.append({"rank": 0, "center_zyx": [float(v) for v in pts[sel].mean(0)],
                    "normal_zyx": [float(v) for v in n / (np.linalg.norm(n) + 1e-12)], "flagged": 0})
    for i, c in enumerate(out, len(flagged) + 1):
        c["rank"] = i
    return out


def read_cells(path: Path) -> List[dict]:
    doc = json.loads(path.read_text())
    (col,) = doc["collections"].values()
    cells = []
    for k, pt in sorted(col["points"].items(), key=lambda kv: int(kv[0])):
        c = {"rank": int(k) + 1, "center_zyx": list(reversed(pt["p"])), "flagged": 1}
        c.update(pt.get("audit", {}))
        cells.append(c)
    return cells


def cell_edits(before: np.ndarray, after: np.ndarray, ok: np.ndarray, cells: List[dict],
               moved_min: float = 0.5) -> List[Dict]:
    key_b = np.floor(before / CELL)
    out = []
    for c in cells:
        if "normal_zyx" not in c:
            sys.exit("worst_cells.json has no normals: rerun the audit (commit 070abd4c8 or later)")
        sel = ok & np.all(key_b == np.floor(np.asarray(c["center_zyx"]) / CELL), axis=-1)
        if not sel.any():
            continue
        disp = after[sel] - before[sel]
        shift = disp @ np.asarray(c["normal_zyx"])
        out.append({**c, "n": int(sel.sum()), "median_shift": float(np.median(shift)),
                    "n_moved": int((np.linalg.norm(disp, axis=1) > moved_min).sum()),
                    "before": before[sel], "after": after[sel]})
    return out


def ct_cube(vol, center, half: int) -> np.ndarray:
    """``2*half`` cube around ``center``, zero-padded where it leaves the volume, so every
    example has the same shape."""
    lo = [int(round(c)) - half for c in center]
    src = tuple(slice(max(l, 0), min(l + 2 * half, n)) for l, n in zip(lo, vol.shape))
    block = np.asarray(vol[src])
    out = np.zeros((2 * half,) * 3, block.dtype)
    out[tuple(slice(s.start - l, s.stop - l) for s, l in zip(src, lo))] = block
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--before", required=True, help="tifxyz directory the audit measured")
    ap.add_argument("--after", required=True, help="the same segment after editing in VC3D")
    ap.add_argument("--cells", type=Path, required=True, help="the audit's <name>.worst_cells.json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--volume", help="CT volume (zarr/tif, local or URL) to store a cube per cell")
    ap.add_argument("--prediction", help="surface prediction volume (same frame as the CT) to store a cube per cell")
    ap.add_argument("--level", default="0")
    ap.add_argument("--half", type=int, default=24, help="half-width of the CT and prediction cubes")
    ap.add_argument("--all-cells", action="store_true", help="also export the cells the audit did not flag")
    ap.add_argument("--editor", default=getpass.getuser(), help="who made the edits (provenance)")
    a = ap.parse_args(argv)
    before, ok_b, N = grid_points(a.before)
    after, ok_a, _ = grid_points(a.after)
    if before.shape != after.shape:
        sys.exit(f"grid shapes differ ({before.shape[:2]} vs {after.shape[:2]}): a regrown segment has no "
                 "point correspondence; audit the new segment instead")
    cells = read_cells(a.cells)
    if a.all_cells:
        cells += all_cells(before, ok_b & ok_a, N, cells)
    edits = cell_edits(before, after, ok_b & ok_a, cells)
    a.out.mkdir(parents=True, exist_ok=True)
    arrays = {}
    vols = {}
    if a.volume or a.prediction:
        from vesuvius.image_proc.run.label_topology_audit import open_volume
        for key, src in (("ct", a.volume), ("pred", a.prediction)):
            if src:
                vols[key] = open_volume(src, a.level)
    with open(a.out / "edits.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "z", "y", "x", "n", "median_offset", "median_shift", "n_moved", "flagged"])
        for e in edits:
            w.writerow([e["rank"], *(f"{v:.1f}" for v in e["center_zyx"]), e["n"],
                        f"{e.get('median_offset', float('nan')):+.2f}", f"{e['median_shift']:+.2f}",
                        e["n_moved"], e["flagged"]])
            arrays[f"before_{e['rank']}"] = e["before"]
            arrays[f"after_{e['rank']}"] = e["after"]
            for key, vol in vols.items():
                arrays[f"{key}_{e['rank']}"] = ct_cube(vol, e["center_zyx"], a.half)
    np.savez_compressed(a.out / "edits.npz", **arrays)
    audit_json = a.cells.with_name(a.cells.name.replace(".worst_cells.json", ".json"))
    prov = {"before": a.before, "after": a.after, "cells": str(a.cells), "volume": a.volume,
            "prediction": a.prediction, "level": a.level, "half": a.half, "all_cells": a.all_cells,
            "editor": a.editor, "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "audit": json.loads(audit_json.read_text()) if audit_json.exists() else None}
    (a.out / "provenance.json").write_text(json.dumps(prov, indent=1))
    moved = sum(1 for e in edits if e["n_moved"])
    print(f"{len(edits)} cells ({sum(e['flagged'] for e in edits)} flagged), {moved} edited; {a.out / 'edits.csv'}")


if __name__ == "__main__":
    main()
