#!/usr/bin/env python3
"""Review the places where a surface model draws papyrus the label does not have.

Ranking: ``pred_far`` = prediction voxels (value 1) farther than ``--tol`` from any label
voxel (label value ``--ignore-value`` is excluded on both sides). A candidate is a
26-connected component of ``pred_far`` with at least ``--min-vox`` voxels. Crops are
ranked by ``pred_far_frac`` = pred_far / prediction voxels. The marker of a candidate is
the component voxel nearest its centroid (a concave component's centroid can sit in a gap
off the sheet); all three slices go through the marker.

Writes under ``--out``:
  queue.csv       crop,dice,pred_far,pred_far_frac,n_cand,largest   (every ranked crop)
  candidates.csv  crop,size,center_zyx,marker_zyx,marker_to_centroid_vox
  tiles/<code>.png, sheet.csv (code,verdict,note: for the reader), key.csv (code,crop,
                  center_zyx,marker_zyx,size,pred_far_frac: shown only after the calls)
  verdicts.csv    crop,center_zyx,marker_zyx,size,verdict,note      (resumable)

Tiles are blind: the title is the code. Verdicts: ``s`` sheet (papyrus the label misses),
``n`` not, ``u`` unclear, ``k`` skip, ``q`` quit. ``--stage-only`` writes everything but
verdicts.csv and exits.

``--export-training DIR`` writes the whole crop (CT and corrected label) per verdict in
the trainer layout (``images/<id>.tif``, ``labels/<id>_<target>.tif``) plus ``index.csv``
and ``provenance.json``. ``sheet``: the label with the candidate component set to 1.
``not``: the label unchanged (a confirmed negative, changed_voxels 0). ``unclear`` is
not exported.
"""
import argparse
import csv
import getpass
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import tifffile
from scipy import ndimage

from vesuvius.scripts.review_holes import write_trainer_pair

VERDICTS = {"s": "SHEET", "n": "NOT", "u": "UNCLEAR"}
FIELDS = ["crop", "center_zyx", "marker_zyx", "size", "verdict", "note"]
INDEX_FIELDS = ["file", "crop", "center_zyx", "marker_zyx", "size", "verdict", "changed_voxels"]
QUEUE_FIELDS = ["crop", "dice", "pred_far", "pred_far_frac", "n_cand", "largest"]
CAND_FIELDS = ["crop", "size", "center_zyx", "marker_zyx", "marker_to_centroid_vox"]
KEY_FIELDS = ["code", "crop", "center_zyx", "marker_zyx", "size", "pred_far_frac"]
LEGEND = "cyan=label  magenta=model  yellow +=on model sheet"


def _zyx(v) -> str:
    return json.dumps([int(x) for x in v], separators=(",", ":"))


def far_components(pred: np.ndarray, label: np.ndarray, tol: float, ignore_value: int):
    """``(components, sizes, pred_vox, label_vox)``: 26-connected components of the
    prediction-minus-label-neighbourhood voxels, with their voxel counts."""
    valid = label != ignore_value
    a = (pred == 1) & valid
    b = (label == 1) & valid
    dl = ndimage.distance_transform_edt(~b) if b.any() else np.full(b.shape, 1e3, np.float32)
    far = a & (dl > tol)
    lab, n = ndimage.label(far, structure=np.ones((3, 3, 3)))
    sizes = np.bincount(lab.ravel(), minlength=n + 1)[1:] if n else np.zeros(0, int)
    return lab, sizes, a, b


def analyse(job) -> Tuple[dict, List[dict]]:
    """Rank one crop. Candidates (largest first, at most ``per_crop``) carry the marker."""
    data, predictions, crop, tol, min_vox, ignore_value, per_crop = job
    pred = tifffile.imread(predictions / f"{crop}.tif")
    label = tifffile.imread(data / "labels" / f"{crop}.tif")
    lab, sizes, a, b = far_components(pred, label, tol, ignore_value)
    dice = 2 * (a & b).sum() / max(1, a.sum() + b.sum())
    big = [k + 1 for k in np.argsort(-sizes, kind="stable") if sizes[k] >= min_vox]
    boxes = ndimage.find_objects(lab)
    cands = []
    for k in big[:per_crop]:
        box = boxes[k - 1]
        vox = np.argwhere(lab[box] == k) + np.array([s.start for s in box])
        centroid = vox.mean(axis=0)
        d = np.linalg.norm(vox - centroid, axis=1)
        j = int(d.argmin())
        cands.append({"crop": crop, "size": int(sizes[k - 1]), "center_zyx": _zyx(np.round(centroid)),
                      "marker_zyx": _zyx(vox[j]), "marker_to_centroid_vox": round(float(d[j]), 1)})
    far = int(sizes.sum())
    row = {"crop": crop, "dice": round(float(dice), 3), "pred_far": far,
           "pred_far_frac": round(far / max(1, int(a.sum())), 4), "n_cand": len(big),
           "largest": int(sizes.max()) if len(sizes) else 0}
    return row, cands


def load_excluded(path: Optional[Path]) -> set:
    if path is None:
        return set()
    with open(path, newline="") as fh:
        return {r["crop"] for r in csv.DictReader(fh)}


def _write_csv(path: Path, fields: List[str], rows) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def rank_crops(data: Path, predictions: Path, tol: float = 3.0, min_vox: int = 200,
               ignore_value: int = 2, per_crop: int = 1, exclude: Optional[set] = None,
               workers: int = 1) -> Tuple[List[dict], Dict[str, List[dict]]]:
    """Every crop with a prediction and a label, worst ``pred_far_frac`` first."""
    exclude = exclude or set()
    crops = sorted(p.stem for p in predictions.glob("*.tif")
                   if p.stem not in exclude and (data / "labels" / f"{p.stem}.tif").exists())
    jobs = [(data, predictions, c, tol, min_vox, ignore_value, per_crop) for c in crops]
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            results = list(ex.map(analyse, jobs))
    else:
        results = [analyse(j) for j in jobs]
    rows = sorted((r for r, _ in results), key=lambda r: -r["pred_far_frac"])
    return rows, {r["crop"]: c for r, c in results}


def render_tile(img: np.ndarray, label: np.ndarray, pred: np.ndarray, marker, png: Path,
                title: str, half: int = 80) -> Path:
    """Three orthogonal slices through ``marker``: label outline cyan, prediction magenta."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lo, hi = np.percentile(img, [1, 99.5])
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.4))
    for ax, (name, i, (u, v)) in zip(axs, [("z", 0, (1, 2)), ("y", 1, (0, 2)), ("x", 2, (0, 1))]):
        take = tuple(marker[i] if d == i else slice(None) for d in range(3))
        im, lb, pd = img[take], label[take] == 1, pred[take] == 1
        start = [max(0, min(marker[d] - half, n - 2 * half)) for d, n in zip((u, v), im.shape)]
        w = tuple(slice(st, st + 2 * half) for st in start)
        ax.imshow(im[w], cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
        if lb[w].any():
            ax.contour(lb[w].astype(float), [0.5], colors="cyan", linewidths=1)
        if pd[w].any():
            ax.contour(pd[w].astype(float), [0.5], colors="magenta", linewidths=1)
        ax.plot(marker[v] - start[1], marker[u] - start[0], "+", color="yellow", ms=9, mew=1.5)
        ax.set_title(f"{name} slice", fontsize=9)
        ax.axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=100)
    plt.close(fig)
    return png


def stage(data: Path, predictions: Path, out: Path, tol: float = 3.0, min_vox: int = 200,
          ignore_value: int = 2, limit: Optional[int] = None, per_crop: int = 1,
          exclude: Optional[set] = None, half: int = 80, workers: int = 1) -> List[dict]:
    """Rank, then write queue/candidates/tiles/key/sheet for the top ``limit`` crops."""
    sheet = out / "sheet.csv"
    if sheet.exists():
        with open(sheet, newline="") as fh:
            if any(r.get("verdict") for r in csv.DictReader(fh)):
                raise SystemExit(f"{sheet} already holds verdicts; move it aside before restaging")
    rows, cands = rank_crops(data, predictions, tol, min_vox, ignore_value, per_crop, exclude, workers)
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "queue.csv", QUEUE_FIELDS, rows)
    keys, flat = [], []
    for r in rows[:limit]:
        for c in cands[r["crop"]]:
            flat.append(c)
            keys.append({"code": f"S{len(keys) + 1:02d}", **{k: c[k] for k in ("crop", "center_zyx", "marker_zyx", "size")},
                         "pred_far_frac": r["pred_far_frac"]})
    _write_csv(out / "candidates.csv", CAND_FIELDS, flat)
    _write_csv(out / "key.csv", KEY_FIELDS, keys)
    _write_csv(out / "sheet.csv", ["code", "verdict", "note"], [{"code": k["code"]} for k in keys])
    cache = {}
    for k in keys:
        crop = k["crop"]
        if crop not in cache:
            cache = {crop: (tifffile.imread(data / "images" / f"{crop}.tif").astype(np.float32),
                            tifffile.imread(data / "labels" / f"{crop}.tif"),
                            tifffile.imread(predictions / f"{crop}.tif"))}
        render_tile(*cache[crop], json.loads(k["marker_zyx"]), out / "tiles" / f"{k['code']}.png",
                    f"{k['code']}   {LEGEND}", half)
    return keys


def load_verdicts(path: Path) -> Dict[Tuple[str, str], dict]:
    if not path.exists():
        return {}
    with open(path, newline="") as fh:
        return {(r["crop"], r["marker_zyx"]): r for r in csv.DictReader(fh)}


def ask_terminal(png: Path, prompt: str) -> str:
    """Show the tile in a matplotlib window and read one key from the terminal."""
    import matplotlib
    import matplotlib.pyplot as plt

    if matplotlib.get_backend().lower() == "agg":
        print(f"no display backend: open {png} yourself", file=sys.stderr)
    else:
        plt.close("all")
        fig = plt.figure(figsize=(12, 4.4))
        plt.imshow(plt.imread(png))
        plt.axis("off")
        if fig.canvas.manager is not None:
            fig.canvas.manager.set_window_title(prompt)
        plt.show(block=False)
        plt.pause(0.1)
    while True:
        k = input(f"{prompt}  [s]heet [n]ot [u]nclear s[k]ip [q]uit > ").strip().lower()
        if k in ("s", "n", "u", "k", "q"):
            return k


def review(out: Path, ask: Callable[[Path, str], str]) -> Dict[Tuple[str, str], dict]:
    """Ask for each staged tile not already in verdicts.csv. The prompt shows the code only."""
    with open(out / "key.csv", newline="") as fh:
        keys = list(csv.DictReader(fh))
    verdicts = load_verdicts(out / "verdicts.csv")
    for n, k in enumerate(keys, 1):
        key = (k["crop"], k["marker_zyx"])
        if key in verdicts:
            continue
        ans = ask(out / "tiles" / f"{k['code']}.png", f"{k['code']} ({n}/{len(keys)})")
        if ans == "q":
            break
        if ans == "k":
            continue
        verdicts[key] = {"crop": k["crop"], "center_zyx": k["center_zyx"], "marker_zyx": k["marker_zyx"],
                         "size": k["size"], "verdict": VERDICTS[ans], "note": ""}
        _write_csv(out / "verdicts.csv", FIELDS, verdicts.values())
    _write_csv(out / "verdicts.csv", FIELDS, verdicts.values())
    return verdicts


def export_training(data: Path, predictions: Path, out: Path, dest: Path, tol: float = 3.0,
                    min_vox: int = 200, reviewer: str = "", target: str = "surface",
                    ignore_value: int = 2) -> List[dict]:
    """Corrected labels for the SHEET and NOT verdicts in ``out/verdicts.csv``."""
    rows = [r for r in load_verdicts(out / "verdicts.csv").values() if r["verdict"] in ("SHEET", "NOT")]
    dest.mkdir(parents=True, exist_ok=True)
    index, cache = [], {}
    for i, r in enumerate(rows):
        crop = r["crop"]
        if crop not in cache:
            cache = {crop: (tifffile.imread(data / "images" / f"{crop}.tif"),
                            tifffile.imread(data / "labels" / f"{crop}.tif"),
                            tifffile.imread(predictions / f"{crop}.tif"))}
        ct, label, pred = cache[crop]
        corrected, changed = label.copy(), 0
        if r["verdict"] == "SHEET":
            lab, _, _, _ = far_components(pred, label, tol, ignore_value)
            k = lab[tuple(json.loads(r["marker_zyx"]))]
            if k == 0:
                sys.exit(f"{crop} marker {r['marker_zyx']} is not on a candidate: rerun with the --tol/--min-vox used to stage")
            comp = lab == k
            if comp.sum() != int(r["size"]) or comp.sum() < min_vox:
                sys.exit(f"{crop} marker {r['marker_zyx']}: component has {int(comp.sum())} voxels, verdict recorded {r['size']}")
            changed = int((comp & (label != 1)).sum())
            corrected[comp] = 1
        stem = f"{crop}_{i:04d}"
        write_trainer_pair(dest, stem, ct, corrected, target)
        index.append({"file": stem, **{k: r[k] for k in INDEX_FIELDS[1:-1]}, "changed_voxels": changed})
    _write_csv(dest / "index.csv", INDEX_FIELDS, index)
    prov = {"data": str(data), "predictions": str(predictions), "verdicts": str(out / "verdicts.csv"),
            "tol": tol, "min_vox": min_vox, "target": target, "ignore_value": ignore_value,
            "reviewer": reviewer, "n": len(index),
            "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (dest / "provenance.json").write_text(json.dumps(prov, indent=1))
    return index


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="directory with labels/<crop>.tif and images/<crop>.tif")
    ap.add_argument("--predictions", type=Path, required=True, help="surface model output per crop (<crop>.tif, argmax, 1 = sheet)")
    ap.add_argument("--out", type=Path, required=True, help="where queue.csv, tiles/, key.csv, verdicts.csv go")
    ap.add_argument("--tol", type=float, default=3.0, help="a prediction voxel is far when this many voxels from any label voxel")
    ap.add_argument("--min-vox", type=int, default=200, help="smallest far component worth a tile")
    ap.add_argument("--exclude", type=Path, help="CSV with a `crop` column: crops to drop (e.g. the holed set)")
    ap.add_argument("--limit", type=int, default=None, help="review only the top N crops by pred_far_frac")
    ap.add_argument("--per-crop", type=int, default=1, help="candidates per crop, largest first")
    ap.add_argument("--half", type=int, default=80, help="half-width of each slice in voxels")
    ap.add_argument("--stage-only", action="store_true", help="write queue/candidates/tiles/key/sheet and ask nothing")
    ap.add_argument("--export-training", type=Path, metavar="DIR",
                    help="write OUT/verdicts.csv as corrected labels in the trainer layout into DIR and ask nothing")
    ap.add_argument("--reviewer", default=getpass.getuser(), help="who made the calls (provenance)")
    ap.add_argument("--target", default="surface", help="with --export-training: target name in labels/<id>_<target>.tif")
    ap.add_argument("--ignore-value", type=int, default=2, help="label value excluded from the comparison")
    ap.add_argument("--workers", type=int, default=4, help="processes for the ranking pass")
    a = ap.parse_args(argv)
    if a.export_training:
        n = export_training(a.data, a.predictions, a.out, a.export_training, a.tol, a.min_vox,
                            a.reviewer, a.target, a.ignore_value)
        print(f"exported {len(n)} examples to {a.export_training}")
        return
    if a.stage_only or not (a.out / "key.csv").exists():
        keys = stage(a.data, a.predictions, a.out, a.tol, a.min_vox, a.ignore_value, a.limit,
                     a.per_crop, load_excluded(a.exclude), a.half, a.workers)
        print(f"staged {len(keys)} tiles in {a.out / 'tiles'}")
        if a.stage_only:
            return
    v = review(a.out, ask_terminal)
    n = {k: sum(1 for r in v.values() if r["verdict"] == k) for k in VERDICTS.values()}
    print(f"{len(v)} verdicts ({n}) in {a.out / 'verdicts.csv'}")


if __name__ == "__main__":
    main()
