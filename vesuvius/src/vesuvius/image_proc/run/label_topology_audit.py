"""Voxel-level topology audit for 3D surface label volumes.

Label convention (Kaggle/HF ``surfaces/kaggle``): 0 = background, 1 = surface,
2 = ignore/unlabeled. Both values are configurable. Ignore voxels are never
foreground; components touching them (or the crop border) are reported as
``open`` because their topology is truncated rather than wrong.

Per crop it measures:
  * connected components (26-connectivity): count, size histogram, tiny fragments
  * per component: Euler characteristic and Betti numbers b1 (handles/holes
    through a sheet) and b2 (enclosed cavities), from a padded bounding box;
    ``b1_interior`` repeats b1 with 6 voxels trimmed off each crop face
  * sheet thickness: 2*EDT-1 sampled at medial-ridge voxels (local EDT maxima)
  * fused sheets: after eroding by ``--erode`` voxels, a component that splits
    into >= 2 pieces of >= ``--min-piece`` voxels is "fused" (blind on 1-3
    voxel sheets, which erosion deletes)
  * non-planar junction clusters (structure tensor): a clean sheet is rank-1;
    a second large eigenvalue marks merged sheets or bridges

Usage:
  vesuvius.label_topology_audit --label LABEL.tif --out DIR
  vesuvius.label_topology_audit --label LABEL.tif --image IMG.tif --out DIR --plot
  vesuvius.label_topology_audit --label BEFORE.tif --after AFTER.tif --out DIR
  vesuvius.label_topology_audit --batch DATA_DIR --out DIR --fail-on b1_total_big=0

``--batch`` reads ``DATA_DIR/labels/*.tif`` (and ``DATA_DIR/images/<same name>``
when present, for ``--plot``); with ``--after`` it names a directory holding
same-named label files.

Writes ``<crop>.json`` per crop, ``aggregate.csv`` for all crops,
``ranked.csv`` (batch: crops with findings, worst first) and
``<crop>.delta.json`` (metric: [before, after]) with ``--after``.
Exit code 0 on success, 2 when a ``--fail-on METRIC=MAX`` limit is exceeded.

Requires connected-components-3d, scikit-image, scipy and tifffile (the
``models`` extra); matplotlib is imported only for ``--plot``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from vesuvius.image_proc.topology import component_betti

TINY = 100  # voxels; components smaller than this are "tiny fragments"
THICK = 7  # voxels; ridge thickness at or above this is flagged
SIZE_BINS = [1, 10, 100, 1_000, 10_000, 100_000, 10**9]
EDGE = 6  # voxels; b1_interior ignores holes within this distance of a crop face
# ranked.csv order: most severe first; each entry is (metric, phrase for the reason column)
SEVERITY = [("b1_interior", "interior holes"), ("b2_total_big", "cavities"),
            ("n_fused_components", "fused components"), ("n_junction_clusters", "junction clusters"),
            ("b1_total_big", "holes incl. crop faces")]

FLAT_KEYS = [
    "n_components", "n_tiny", "tiny_voxel_frac", "n_big_components", "n_closed_big",
    "b1_total_big", "b1_interior", "n_big_with_b1", "n_closed_with_b1", "b2_total_big",
    "n_fused_components", "fused_voxel_frac", "bridge_voxels",
    "n_junction_clusters", "junction_voxel_frac", "n_components_with_junction",
    "largest_component_frac",
]


def _deps():
    try:
        import cc3d
        from scipy import ndimage as ndi
        from skimage.measure import euler_number
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            f"label_topology_audit needs connected-components-3d, scipy and "
            f"scikit-image (install the `models` extra): {exc}"
        )
    return cc3d, ndi, euler_number


def component_topology(mask: np.ndarray) -> dict[str, int]:
    """Euler number, b1, b2 of a connected binary mask (foreground 26-conn, background 6-conn)."""
    _deps()
    b1, b2, chi = component_betti(mask, fg_connectivity=26, bg_connectivity=6)
    return {"euler": chi, "b1": b1, "b2": b2}


def interior_b1(fg: np.ndarray, margin: int) -> int:
    """Total b1 of the big components left after trimming ``margin`` voxels off every crop face."""
    cc3d, ndi, _ = _deps()
    if min(fg.shape) <= 2 * margin:
        return 0
    core = fg[(slice(margin, -margin or None),) * fg.ndim]
    comp = cc3d.connected_components(core, connectivity=26)
    return sum(component_topology(comp[sl] == cid)["b1"]
               for cid, sl in enumerate(ndi.find_objects(comp), start=1)
               if sl is not None and np.count_nonzero(comp[sl] == cid) >= TINY)


def thickness_stats(fg: np.ndarray):
    """Thickness (voxels) sampled on medial-ridge voxels = local maxima of the EDT."""
    _, ndi, _ = _deps()
    edt = ndi.distance_transform_edt(fg)
    ridge = fg & (edt >= ndi.maximum_filter(edt, size=3)) & (edt > 0)
    t = 2 * edt[ridge] - 1
    if t.size == 0:
        return {}, ridge, edt
    q = np.percentile(t, [5, 50, 95, 99])
    stats = {"p5": q[0], "median": q[1], "p95": q[2], "p99": q[3], "max": t.max(),
             "frac_ge_7": float((t >= THICK).mean()), "n_ridge": int(t.size)}
    return stats, ridge, edt


def fused_sheets(fg, comp, erode, min_piece):
    """Components that split into >=2 big pieces after erosion; returns (ids, bridge mask)."""
    cc3d, ndi, _ = _deps()
    ball = ndi.generate_binary_structure(3, 1)
    core = ndi.binary_erosion(fg, ball, iterations=erode)
    pieces = cc3d.connected_components(core, connectivity=26)
    sizes = np.bincount(pieces.ravel())
    big = np.flatnonzero(sizes >= min_piece)
    big = big[big > 0]
    keep = np.where(np.isin(pieces, big), pieces, 0)
    idx = np.flatnonzero(keep)
    _, first = np.unique(keep.ravel()[idx], return_index=True)
    owners, counts = np.unique(comp.ravel()[idx[first]], return_counts=True)
    fused = sorted(c for c, k in zip(owners.tolist(), counts.tolist()) if k >= 2)
    # bridge voxels: fg voxels whose 3x3x3 neighbourhood touches >=2 different pieces
    hi = ndi.maximum_filter(keep, size=3)
    lo = ndi.minimum_filter(np.where(keep > 0, keep, np.iinfo(keep.dtype).max), size=3)
    bridge = fg & (hi > 0) & (lo < hi) & np.isin(comp, fused)
    return fused, bridge


def junction_voxels(fg: np.ndarray, thresh: float = 0.3) -> np.ndarray:
    """Non-planar spots: structure tensor of the smoothed mask has a big 2nd eigenvalue."""
    _, ndi, _ = _deps()
    g = np.gradient(ndi.gaussian_filter(fg.astype(np.float32), 1.5))
    T = np.empty(fg.shape + (3, 3), np.float32)
    for i in range(3):
        for j in range(i, 3):
            T[..., i, j] = T[..., j, i] = ndi.gaussian_filter(g[i] * g[j], 3)
    ev = np.linalg.eigvalsh(T[fg])[:, ::-1] if fg.any() else np.zeros((0, 3))
    out = np.zeros_like(fg)
    out[fg] = ev[:, 1] / (ev[:, 0] + 1e-9) > thresh
    m = 6  # drop crop-border artefacts (2 sigma of the tensor smoothing)
    out[:m], out[-m:], out[:, :m], out[:, -m:], out[:, :, :m], out[:, :, -m:] = [False] * 6
    return out


def audit(label, erode=1, min_piece=500, surface_value=1, ignore_value=2):
    """Return (metrics dict, flag masks dict) for one label volume."""
    cc3d, ndi, _ = _deps()
    ball = ndi.generate_binary_structure(3, 1)
    fg = label == surface_value
    unl = (label == ignore_value) if ignore_value is not None else np.zeros_like(fg)
    comp, n = cc3d.connected_components(fg, connectivity=26, return_N=True)
    sizes = np.bincount(comp.ravel(), minlength=n + 1)
    # components touching the crop border or ignore voxels have truncated topology
    near_unl = ndi.binary_dilation(unl, ball) if unl.any() else unl
    border = np.zeros_like(fg)
    for ax in range(3):
        for s in (0, -1):
            border[(slice(None),) * ax + (s,)] = True
    open_ids = set(np.unique(comp[(border | near_unl) & fg]).tolist()) - {0}

    rows = []
    for cid, sl in enumerate(ndi.find_objects(comp), start=1):
        if sl is None:
            continue
        topo = component_topology(comp[sl] == cid) if sizes[cid] >= TINY else {}
        rows.append({"id": cid, "size": int(sizes[cid]), "open": cid in open_ids, **topo})
    big = [r for r in rows if "b1" in r]
    closed = [r for r in big if not r["open"]]

    thick, ridge, edt = thickness_stats(fg)
    fused, bridge = fused_sheets(fg, comp, erode, min_piece)
    junc = junction_voxels(fg)
    jl = cc3d.connected_components(junc, connectivity=26)
    jsz = np.bincount(jl.ravel())[1:]
    jbig = np.flatnonzero(jsz >= 20) + 1  # clusters of >=20 non-planar voxels
    junc_comps = np.unique(comp[np.isin(jl, jbig)])
    tiny_ids = [r["id"] for r in rows if r["size"] < TINY]
    tiny_mask = np.isin(comp, tiny_ids)
    thick_mask = ridge & (2 * edt - 1 >= THICK)

    hist, _ = np.histogram(sizes[1:], bins=SIZE_BINS)
    nfg = int(fg.sum())
    m = {
        "shape": list(label.shape),
        "surface_value": surface_value, "ignore_value": ignore_value,
        "fg_voxels": nfg, "fg_frac": nfg / label.size, "unlabeled_frac": float(unl.mean()),
        "n_components": n,
        "size_hist": {f"[{a},{b})": int(h) for a, b, h in zip(SIZE_BINS, SIZE_BINS[1:], hist)},
        "largest_component_frac": float(sizes[1:].max() / nfg) if n else 0.0,
        "n_tiny": len(tiny_ids), "tiny_voxel_frac": float(tiny_mask.sum() / max(nfg, 1)),
        "n_big_components": len(big), "n_closed_big": len(closed),
        "euler_total_big": int(sum(r["euler"] for r in big)),
        "b1_total_big": int(sum(r["b1"] for r in big)),
        "b1_total_closed": int(sum(r["b1"] for r in closed)),
        "b1_interior": int(interior_b1(fg, EDGE)) if big else 0,
        "n_big_with_b1": sum(r["b1"] > 0 for r in big),
        "n_closed_with_b1": sum(r["b1"] > 0 for r in closed),
        "b2_total_big": int(sum(r["b2"] for r in big)),
        "thickness": {k: float(v) for k, v in thick.items()},
        "n_fused_components": len(fused),
        "fused_voxel_frac": float(np.isin(comp, fused).sum() / max(nfg, 1)),
        "bridge_voxels": int(bridge.sum()),
        "n_junction_clusters": int(jbig.size),
        "junction_voxel_frac": float(junc.sum() / max(nfg, 1)),
        "n_components_with_junction": int(junc_comps.size),
        "components": sorted(rows, key=lambda r: -r["size"])[:25],
    }
    flags = {"tiny": tiny_mask, "bridge": bridge, "thick": thick_mask, "junction": junc}
    return m, flags


def flat(m: dict[str, Any]) -> dict[str, Any]:
    t = m["thickness"]
    return {**{k: m[k] for k in ["fg_frac", "unlabeled_frac"] + FLAT_KEYS},
            **{f"thick_{k}": t.get(k) for k in ("p5", "median", "p95", "p99", "frac_ge_7")}}


def overlay(img, label, flags, path, surface_value=1):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    any_flag = flags["tiny"] | flags["bridge"] | flags["thick"] | flags["junction"]
    z = int(any_flag.sum(axis=(1, 2)).argmax()) if any_flag.any() else img.shape[0] // 2
    rgb = np.repeat((img[z] / max(img[z].max(), 1))[..., None], 3, axis=2)
    pan = [rgb.copy(), rgb.copy()]
    sel = label[z] == surface_value
    pan[1][sel] = 0.55 * pan[1][sel] + 0.45 * np.array([0, 0.8, 0])
    for name, col in (("tiny", (1, 0, 0)), ("bridge", (1, 1, 0)),
                      ("thick", (0, 1, 1)), ("junction", (1, 0, 1))):
        pan[1][flags[name][z]] = col
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    titles = ("image", f"label green; tiny red, bridge yel, thick cyan, junction mag (z={z})")
    for a, p, t in zip(ax, pan, titles):
        a.imshow(p)
        a.set_title(t, fontsize=8)
        a.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def ranked(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Crops with any finding, most severe first (lexicographic over SEVERITY), with a reason."""
    keys = [k for k, _ in SEVERITY]
    hit = [r for r in rows if any(r[k] for k in keys)]
    hit.sort(key=lambda r: tuple(-r[k] for k in keys))
    return [{"rank": i, "crop": r["crop"],
             "reason": ", ".join(f"{r[k]} {name}" for k, name in SEVERITY if r[k]),
             **{k: r[k] for k in keys}}
            for i, r in enumerate(hit, start=1)]


def _parse_fail_on(specs: Sequence[str]) -> dict[str, float]:
    limits = {}
    for spec in specs:
        key, sep, val = spec.partition("=")
        if not sep or key not in set(flat_keys()):
            raise SystemExit(f"--fail-on expects METRIC=MAX with METRIC in {flat_keys()}: {spec!r}")
        try:
            limits[key] = float(val)
        except ValueError:
            raise SystemExit(f"--fail-on limit must be a number: {spec!r}") from None
    return limits


def flat_keys() -> list[str]:
    return ["fg_frac", "unlabeled_frac"] + FLAT_KEYS


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="vesuvius.label_topology_audit",
        description="Audit topology (components, Betti numbers, thickness, fusion) of 3D surface labels.",
    )
    ap.add_argument("--label", type=Path, help="label volume (.tif)")
    ap.add_argument("--image", type=Path, help="matching image volume; only needed for --plot")
    ap.add_argument("--batch", type=Path, help="directory with labels/*.tif (and optional images/)")
    ap.add_argument("--after", type=Path,
                    help="second label volume (batch: directory of same-named files); writes <crop>.delta.json")
    ap.add_argument("--out", type=Path, default=Path("audit_out"), help="output directory")
    ap.add_argument("--surface-value", type=int, default=1, help="label value of the surface")
    ap.add_argument("--ignore-value", type=int, default=2,
                    help="label value of ignored/unlabeled voxels (-1 to disable)")
    ap.add_argument("--erode", type=int, default=1, help="erosion iterations for the fusion test")
    ap.add_argument("--min-piece", type=int, default=500, help="minimum voxels of a post-erosion piece")
    ap.add_argument("--plot", action="store_true", help="write overlay PNGs (needs --image / batch images)")
    ap.add_argument("--fail-on", action="append", default=[], metavar="METRIC=MAX",
                    help="exit 2 if a crop's metric (or its --after volume's) exceeds MAX (repeatable), e.g. b1_total_big=0")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    import tifffile

    ap = build_parser()
    a = ap.parse_args(argv)
    if bool(a.label) == bool(a.batch):
        ap.error("give exactly one of --label or --batch")
    limits = _parse_fail_on(a.fail_on)
    ignore = None if a.ignore_value < 0 else a.ignore_value

    if a.batch:
        labels = sorted((a.batch / "labels").glob("*.tif"))
        if not labels:
            ap.error(f"no labels/*.tif under {a.batch}")
        jobs = [(p, a.batch / "images" / p.name, a.after / p.name if a.after else None) for p in labels]
        missing = [str(j[2]) for j in jobs if j[2] is not None and not j[2].exists()]
        if missing:
            ap.error(f"--after is missing {len(missing)} paired file(s), e.g. {missing[0]}")
    else:
        jobs = [(a.label, a.image, a.after)]

    a.out.mkdir(parents=True, exist_ok=True)
    rows, failed = [], False
    for label_path, image_path, after_path in jobs:
        name = label_path.stem
        lab = tifffile.imread(label_path)
        m, flags = audit(lab, a.erode, a.min_piece, a.surface_value, ignore)
        (a.out / f"{name}.json").write_text(json.dumps(m, indent=1))
        f = flat(m)
        rows.append({"crop": name, **f})
        print(name, json.dumps(f), flush=True)
        if a.plot:
            if image_path is None or not Path(image_path).exists():
                print(f"{name}: --plot skipped, no image", file=sys.stderr)
            else:
                overlay(tifffile.imread(image_path), lab, flags, a.out / f"{name}.png", a.surface_value)
        if after_path:
            m2, _ = audit(tifffile.imread(after_path), a.erode, a.min_piece, a.surface_value, ignore)
            f2 = flat(m2)
            delta = {k: [f[k], f2[k]] for k in f}
            (a.out / f"{name}.delta.json").write_text(json.dumps(delta, indent=1))
            print("delta (before, after):", json.dumps(delta))
        checked = [("", f)] + ([(" (after)", f2)] if after_path else [])
        for tag, vals in checked:
            for key, limit in limits.items():
                if vals[key] > limit:
                    print(f"FAIL {name}{tag}: {key}={vals[key]} > {limit}", file=sys.stderr)
                    failed = True
    with open(a.out / "aggregate.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    if a.batch:
        top = ranked(rows)
        with open(a.out / "ranked.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["rank", "crop", "reason"] + [k for k, _ in SEVERITY])
            w.writeheader()
            w.writerows(top)
        print(f"{len(top)} of {len(rows)} crops have findings; worst first in {a.out / 'ranked.csv'}")
    return 2 if failed else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
