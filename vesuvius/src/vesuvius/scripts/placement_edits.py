#!/usr/bin/env python3
"""Record what an annotator did to the worst cells of a placement audit.

Give it the tifxyz segment the audit measured (``--before``), the same segment after it
was edited in VC3D (``--after``), and the audit's ``<name>.worst_cells.json``. For each
cell it writes how far the grid points moved along the cell's normal, next to the offset
the audit had measured there, so each edit becomes a training pair: the CT and the
surface as traced, and where a human put it. Nothing is trained here.

Writes under ``--out``:
  edits.csv   rank,z,y,x,n,median_offset,median_shift,n_moved
  edits.npz   before_<rank>, after_<rank> (zyx grid points of the cell), ct_<rank> with --volume

A correct edit shows ``median_shift`` close to ``median_offset``: the surface moved by the
distance the audit measured. Rerun the audit on ``--after`` to confirm the cell is gone.
"""
import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

CELL = 32.0


def grid_points(path: str):
    from vesuvius.tifxyz import read_tifxyz

    s = read_tifxyz(path)
    x, y, z = s._x, s._y, s._z
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & (x > 0) & (y > 0) & (z > 0)
    if s._mask is not None:
        ok &= s._mask.astype(bool)
    return np.stack([z, y, x], -1).astype(np.float64), ok


def read_cells(path: Path) -> List[dict]:
    doc = json.loads(path.read_text())
    (col,) = doc["collections"].values()
    cells = []
    for k, pt in sorted(col["points"].items(), key=lambda kv: int(kv[0])):
        c = {"rank": int(k) + 1, "center_zyx": list(reversed(pt["p"]))}
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
    lo = [max(int(round(c)) - half, 0) for c in center]
    hi = [min(l + 2 * half, n) for l, n in zip(lo, vol.shape)]
    return np.asarray(vol[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--before", required=True, help="tifxyz directory the audit measured")
    ap.add_argument("--after", required=True, help="the same segment after editing in VC3D")
    ap.add_argument("--cells", type=Path, required=True, help="the audit's <name>.worst_cells.json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--volume", help="CT volume (zarr/tif, local or URL) to store a cube per cell")
    ap.add_argument("--level", default="0")
    ap.add_argument("--half", type=int, default=16, help="half-width of the CT cube")
    a = ap.parse_args(argv)
    before, ok_b = grid_points(a.before)
    after, ok_a = grid_points(a.after)
    if before.shape != after.shape:
        sys.exit(f"grid shapes differ ({before.shape[:2]} vs {after.shape[:2]}): a regrown segment has no "
                 "point correspondence; audit the new segment instead")
    edits = cell_edits(before, after, ok_b & ok_a, read_cells(a.cells))
    a.out.mkdir(parents=True, exist_ok=True)
    arrays = {}
    vol = None
    if a.volume:
        from vesuvius.image_proc.run.label_topology_audit import open_volume
        vol = open_volume(a.volume, a.level)
    with open(a.out / "edits.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "z", "y", "x", "n", "median_offset", "median_shift", "n_moved"])
        for e in edits:
            w.writerow([e["rank"], *(f"{v:.1f}" for v in e["center_zyx"]), e["n"],
                        f"{e.get('median_offset', float('nan')):+.2f}", f"{e['median_shift']:+.2f}", e["n_moved"]])
            arrays[f"before_{e['rank']}"] = e["before"]
            arrays[f"after_{e['rank']}"] = e["after"]
            if vol is not None:
                arrays[f"ct_{e['rank']}"] = ct_cube(vol, e["center_zyx"], a.half)
    np.savez_compressed(a.out / "edits.npz", **arrays)
    moved = sum(1 for e in edits if e["n_moved"])
    print(f"{len(edits)} cells, {moved} edited; {a.out / 'edits.csv'}")


if __name__ == "__main__":
    main()
