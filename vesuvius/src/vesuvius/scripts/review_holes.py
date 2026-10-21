#!/usr/bin/env python3
"""Review the holes that ``vesuvius.label_topology_audit --batch`` located, worst crop first.

For each hole the script shows three orthogonal CT slices (label green, hole red), asks
for a verdict, and records it. Verdicts: ``f`` fill (label gap), ``c`` cut (real gap or a
pocket between two sheets: do not fill), ``u`` unclear, ``s`` skip, ``q`` quit. The run is
resumable: holes already in ``verdicts.csv`` are not asked again.

Writes under ``--out``:
  verdicts.csv   crop,center_zyx,voxels,interior,verdict   (every call)
  veto.csv       crop,center_zyx                           (CUT rows, for ``fix_holes.py --veto``)
  fix_in/<crop>/<crop>_surf.tif + hole_mask.tif            (reviewed crops, ``fix_holes.py --in-dir`` layout)

Next step after a review session (from ``vesuvius/``)::

  python src/vesuvius/scripts/fix_holes.py --in-dir OUT/fix_in --veto OUT/veto.csv \\
      --ignore-value 2 --method close --closing-kernel 2
"""
import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import tifffile

VERDICTS = {"f": "FILL", "c": "CUT", "u": "UNCLEAR"}
FIELDS = ["crop", "center_zyx", "voxels", "interior", "verdict"]


def ranked_crops(audit: Path) -> List[str]:
    """Crop names in ``ranked.csv`` order (worst first)."""
    with open(audit / "ranked.csv", newline="") as fh:
        return [r["crop"] for r in csv.DictReader(fh)]


def crop_holes(audit: Path, crop: str) -> List[dict]:
    """Located holes of one crop: interior first, then largest first."""
    holes = json.loads((audit / f"{crop}.json").read_text()).get("holes", [])
    return sorted(holes, key=lambda h: (not h.get("interior", False), -h["voxels"]))


def load_verdicts(path: Path) -> Dict[Tuple[str, str], dict]:
    if not path.exists():
        return {}
    with open(path, newline="") as fh:
        return {(r["crop"], r["center_zyx"]): r for r in csv.DictReader(fh)}


def write_outputs(out: Path, verdicts: Dict[Tuple[str, str], dict]) -> None:
    rows = list(verdicts.values())
    with open(out / "verdicts.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    with open(out / "veto.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["crop", "center_zyx"])
        for r in rows:
            if r["verdict"] == "CUT":
                w.writerow([r["crop"], r["center_zyx"]])


def stage_crop(data: Path, audit: Path, out: Path, crop: str) -> Path:
    """Copy label + hole mask into the ``fix_holes.py --in-dir`` layout."""
    d = out / "fix_in" / crop
    d.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(data / "labels" / f"{crop}.tif", d / f"{crop}_surf.tif")
    mask = audit / f"{crop}.hole_mask.tif"
    if not mask.exists():
        sys.exit(f"{mask} missing: rerun the audit with --hole-masks")
    shutil.copyfile(mask, d / "hole_mask.tif")
    return d


def render_hole(data: Path, audit: Path, crop: str, hole: dict, png: Path, half: int) -> Path:
    from vesuvius.image_proc.run.label_topology_audit import hole_tiles

    img = tifffile.imread(data / "images" / f"{crop}.tif").astype(np.float32)
    lab = tifffile.imread(data / "labels" / f"{crop}.tif")
    mask = tifffile.imread(audit / f"{crop}.hole_mask.tif").astype(bool)
    png.parent.mkdir(parents=True, exist_ok=True)
    hole_tiles(img, lab, [hole], mask, png, half=half, limit=1)
    return png


def ask_terminal(png: Path, prompt: str) -> str:
    """Show the tile in a matplotlib window and read one key from the terminal."""
    import matplotlib
    import matplotlib.pyplot as plt

    if matplotlib.get_backend().lower() == "agg":
        print(f"no display backend: open {png} yourself", file=sys.stderr)
    else:
        plt.close("all")
        fig = plt.figure(figsize=(7.5, 2.8))
        plt.imshow(plt.imread(png))
        plt.axis("off")
        if fig.canvas.manager is not None:
            fig.canvas.manager.set_window_title(prompt)
        plt.show(block=False)
        plt.pause(0.1)
    while True:
        k = input(f"{prompt}  [f]ill [c]ut [u]nclear [s]kip [q]uit > ").strip().lower()
        if k in ("f", "c", "u", "s", "q"):
            return k


def review(data: Path, audit: Path, out: Path, ask: Callable[[Path, str], str],
           limit: Optional[int] = None, half: int = 32) -> Dict[Tuple[str, str], dict]:
    out.mkdir(parents=True, exist_ok=True)
    verdicts = load_verdicts(out / "verdicts.csv")
    crops = ranked_crops(audit)[:limit]
    for rank, crop in enumerate(crops, 1):
        holes = crop_holes(audit, crop)
        if not holes:
            continue
        touched = False
        for i, h in enumerate(holes, 1):
            key = (crop, json.dumps(h["center_zyx"]))
            if key in verdicts:
                continue
            tag = "interior" if h.get("interior") else "crop face"
            prompt = f"#{rank} {crop} hole {i}/{len(holes)} zyx {h['center_zyx']} {h['voxels']} vox {tag}"
            png = render_hole(data, audit, crop, h, out / "tiles" / f"{crop}_{i:02d}.png", half)
            k = ask(png, prompt)
            if k == "q":
                if touched:
                    stage_crop(data, audit, out, crop)
                write_outputs(out, verdicts)
                return verdicts
            if k == "s":
                continue
            verdicts[key] = {"crop": crop, "center_zyx": key[1], "voxels": h["voxels"],
                             "interior": int(bool(h.get("interior"))), "verdict": VERDICTS[k]}
            touched = True
            write_outputs(out, verdicts)
        if touched:
            stage_crop(data, audit, out, crop)
    write_outputs(out, verdicts)
    return verdicts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="the --batch directory (labels/, images/)")
    ap.add_argument("--audit", type=Path, required=True,
                    help="audit output directory (needs --batch run with --hole-masks)")
    ap.add_argument("--out", type=Path, required=True, help="where verdicts.csv, veto.csv, tiles/, fix_in/ go")
    ap.add_argument("--limit", type=int, default=None, help="review only the top N crops of ranked.csv")
    ap.add_argument("--half", type=int, default=32, help="half-width of each slice in voxels")
    a = ap.parse_args(argv)
    if not (a.audit / "ranked.csv").exists():
        ap.error(f"{a.audit}/ranked.csv missing: run vesuvius.label_topology_audit --batch first")
    v = review(a.data, a.audit, a.out, ask_terminal, a.limit, a.half)
    n = {k: sum(1 for r in v.values() if r["verdict"] == k) for k in VERDICTS.values()}
    print(f"{len(v)} verdicts ({n}); veto list: {a.out / 'veto.csv'}; staged crops: {a.out / 'fix_in'}")


if __name__ == "__main__":
    main()
