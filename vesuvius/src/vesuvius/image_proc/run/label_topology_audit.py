"""Voxel-level topology audit for 3D surface label volumes.

Label convention (Kaggle/HF ``surfaces/kaggle``): 0 = background, 1 = surface,
2 = ignore/unlabeled. Both values are configurable. Ignore voxels are never
foreground; components touching them (or the crop border) are reported as
``open`` because their topology may be truncated; this is not a defect verdict.

Per crop it measures:
  * connected components (26-connectivity): count, size histogram, tiny fragments
  * per component: Euler characteristic and Betti numbers b1 (handles/holes
    through a sheet) and b2 (enclosed cavities), from a padded bounding box;
    b1 is split into ``b1_face`` (located holes centred within 6 voxels of a crop face)
    and ``b1_interior`` (the rest, including ``b1_unlocated``: holes too wide to locate)
  * sheet thickness: 2*EDT-1 sampled at medial-ridge voxels (local EDT maxima)
  * fused sheets: after eroding by ``--erode`` voxels, a component that splits
    into >= 2 pieces of >= ``--min-piece`` voxels is "fused" (blind on 1-3
    voxel sheets, which erosion deletes)
  * hole locations: closing-filled background blobs whose filling lowers b1
    (centre, size, interior/crop face); ``--hole-masks`` writes them for fix_holes.py
  * non-planar junction clusters (structure tensor): a clean sheet is rank-1;
    a second large eigenvalue marks merged sheets or bridges

Assumes an isotropic voxel grid: connectivity, erosion, EDT thickness and smoothing are in
voxels, not physical units.

Usage:
  vesuvius.label_topology_audit --label LABEL.tif --out DIR
  vesuvius.label_topology_audit --label LABEL.tif --image IMG.tif --out DIR --plot
  vesuvius.label_topology_audit --label BEFORE.tif --after AFTER.tif --out DIR
  vesuvius.label_topology_audit --batch DATA_DIR --out DIR --fail-on b1_total_big=0
  vesuvius.label_topology_audit --label PRED.zarr --threshold 128 \
      --roi 12288:13312,4096:5120,3584:4608 --tile 512 --out DIR

``--label``, ``--after`` and ``--image`` take a ``.tif`` file or a zarr array or
OME-Zarr group (local path or http(s) URL; ``--level`` picks the pyramid level).
``--roi z0:z1,y0:y1,x0:x1`` reads a window of it, ``--tile N`` splits the window
into N-voxel cubes audited as separate crops (``<name>_z<z0>_y<y0>_x<x0>``), and
``--threshold T[,T2,...]`` binarises an image or probability volume as ``value >= T``;
several thresholds audit each window once per cut (``<name>_t<T>``) from one read.
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
import itertools
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from vesuvius.image_proc.topology import component_betti

TINY = 100  # voxels; components smaller than this are "tiny fragments"
THICK = 7  # voxels; ridge thickness at or above this is flagged
SIZE_BINS = [1, 10, 100, 1_000, 10_000, 100_000, 10**9]
EDGE = 6  # voxels; a located hole centred closer than this to a crop face is not interior
# ranked.csv order: most severe first; each entry is (metric, phrase for the reason column)
SEVERITY = [("b1_interior", "interior or unlocated holes"), ("b2_total_big", "cavities"),
            ("n_fused_components", "fused components"), ("n_junction_clusters", "junction clusters"),
            ("b1_total_big", "holes incl. crop faces")]

FLAT_KEYS = [
    "n_components", "n_tiny", "tiny_voxel_frac", "n_big_components", "n_closed_big",
    "b1_total_big", "b1_interior", "b1_face", "b1_unlocated", "n_holes_located", "n_big_with_b1", "n_closed_with_b1", "b2_total_big",
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


def locate_holes(mask: np.ndarray, b1: int, blocked: np.ndarray | None = None, closing: int = 3):
    """Hole sites of one component: blobs of background voxels that a ``closing``-step closing
    fills and whose filling lowers b1. Holes wider than the closing are not located.

    Returns (list of {"center_zyx", "voxels", "fixes"}, mask of the located blobs).
    """
    _, ndi, euler_number = _deps()
    st = ndi.generate_binary_structure(3, 3)
    p = closing + 1
    fill = ndi.binary_closing(np.pad(mask, p), st, iterations=closing)[p:-p, p:-p, p:-p] & ~mask
    if blocked is not None:
        fill &= ~blocked
    lab, _ = ndi.label(fill, st)
    sizes = np.bincount(lab.ravel())[1:]
    # blobs are measured against the mask with earlier blobs already filled, so two blobs
    # that close the same loop are not both counted
    holes, cur = [], mask.copy()
    for j, sl in sorted(enumerate(ndi.find_objects(lab), start=1), key=lambda t: -sizes[t[0] - 1]):
        if b1 <= 0:
            break
        blob = lab == j
        # Euler number is local: a chi change in a window around the blob screens candidates
        # before the full b1 check (filling a hole through a sheet raises chi by 1)
        win = tuple(slice(max(s.start - 2, 0), s.stop + 2) for s in sl)
        before = cur[win]
        if euler_number(np.pad(before | blob[win], 1), 3) <= euler_number(np.pad(before, 1), 3):
            continue
        # a blob the sheet does not touch (cut off by blocked voxels) is an island: it raises
        # the Euler number without closing a loop, and b1 is computed assuming one component
        if not (ndi.binary_dilation(blob[win], st) & before).any():
            continue
        b1_after = component_topology(cur | blob)["b1"]
        if b1_after < b1:
            c = np.argwhere(blob).mean(0).round().astype(int)
            holes.append({"center_zyx": c.tolist(), "voxels": int(sizes[j - 1]), "fixes": int(b1 - b1_after)})
            cur |= blob
            b1 = b1_after
    return holes, cur & ~mask


def thickness_stats(fg: np.ndarray):
    """Thickness proxy on medial ridges; crop faces are treated as background boundaries.

    Where material extends out of the crop, this is a truncated lower-bound estimate.
    """
    _, ndi, _ = _deps()
    edt = ndi.distance_transform_edt(np.pad(fg, 1))[1:-1, 1:-1, 1:-1]
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
    smoothed = ndi.gaussian_filter(fg.astype(np.float32), 1.5)
    g = [np.gradient(smoothed, axis=ax) if fg.shape[ax] > 1 else np.zeros_like(smoothed)
         for ax in range(3)]
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
    if label.ndim != 3 or label.size == 0:
        raise ValueError(f"expected a nonempty 3D label volume, got shape {label.shape}")
    if erode < 1 or min_piece < 1:
        raise ValueError("--erode and --min-piece must be positive integers")
    if ignore_value is not None and surface_value == ignore_value:
        raise ValueError("--surface-value and --ignore-value must differ")
    cc3d, ndi, _ = _deps()
    ball = ndi.generate_binary_structure(3, 3)  # match the 26-connected foreground
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

    rows, holes, hole_mask = [], [], np.zeros_like(fg)
    for cid, sl in enumerate(ndi.find_objects(comp), start=1):
        if sl is None:
            continue
        topo = component_topology(comp[sl] == cid) if sizes[cid] >= TINY else {}
        rows.append({"id": cid, "size": int(sizes[cid]), "open": cid in open_ids, **topo})
        if topo.get("b1"):
            found, blobs = locate_holes(comp[sl] == cid, topo["b1"], unl[sl] | (comp[sl] > 0))
            off = np.array([s.start for s in sl])
            for h in found:
                c = np.array(h["center_zyx"]) + off
                inside = bool(np.all((c >= EDGE) & (c < np.array(fg.shape) - EDGE)))
                holes.append({"component": cid, **h, "center_zyx": c.tolist(), "interior": inside})
            hole_mask[sl] |= blobs
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
        "n_big_with_b1": sum(r["b1"] > 0 for r in big),
        "n_closed_with_b1": sum(r["b1"] > 0 for r in closed),
        "b2_total_big": int(sum(r["b2"] for r in big)),
        "n_holes_located": int(sum(h["fixes"] for h in holes)),
        # Legacy name: b1_interior includes unlocated holes, whose location is unknown.
        # b1_total_big = b1_face + b1_interior; b1_interior includes b1_unlocated.
        "b1_face": int(sum(h["fixes"] for h in holes if not h["interior"])),
        "b1_unlocated": int(sum(r["b1"] for r in big) - sum(h["fixes"] for h in holes)),
        "b1_interior": int(sum(r["b1"] for r in big) - sum(h["fixes"] for h in holes if not h["interior"])),
        "holes": holes,
        "thickness": {k: float(v) for k, v in thick.items()},
        "n_fused_components": len(fused),
        "fused_voxel_frac": float(np.isin(comp, fused).sum() / max(nfg, 1)),
        "bridge_voxels": int(bridge.sum()),
        "n_junction_clusters": int(jbig.size),
        "junction_voxel_frac": float(junc.sum() / max(nfg, 1)),
        "n_components_with_junction": int(junc_comps.size),
        "components": sorted(rows, key=lambda r: -r["size"])[:25],
    }
    flags = {"tiny": tiny_mask, "bridge": bridge, "thick": thick_mask, "junction": junc, "hole": hole_mask}
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


def hole_tiles(img, label, holes, hole_mask, path, surface_value=1, ignore_value=2, half=32, limit=8):
    """Three orthogonal CT slices through each of the first ``limit`` located holes."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    holes = holes[:limit]
    fig, axs = plt.subplots(len(holes), 3, figsize=(7.5, 2.5 * len(holes)), squeeze=False)
    scale = max(float(img.max()), 1.0)
    for row, h in zip(axs, holes):
        c = h["center_zyx"]
        for axis, a in enumerate(row):
            sl = [slice(max(q - half, 0), q + half) for q in c]
            sl[axis] = c[axis]
            sl = tuple(sl)
            rgb = np.repeat((img[sl] / scale)[..., None], 3, axis=2)
            lb = label[sl]
            rgb[lb == surface_value] = 0.55 * rgb[lb == surface_value] + 0.45 * np.array([0, 1, 0])
            rgb[lb == ignore_value] = 0.7 * rgb[lb == ignore_value] + 0.3 * np.array([0, 0.3, 1])
            rgb[hole_mask[sl]] = (1, 0, 0)
            a.imshow(rgb, interpolation="nearest")
            a.set_xticks([])
            a.set_yticks([])
        tag = "interior" if h["interior"] else "crop face"
        row[0].set_ylabel(f"zyx {c}\n{h['voxels']} vox, {tag}", fontsize=7)
    axs[0, 1].set_title("label green, ignore blue, hole red", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
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


def _parse_fail_on(specs: Sequence[str], keys: Sequence[str] | None = None) -> dict[str, float]:
    keys = list(flat_keys()) if keys is None else list(keys)
    limits = {}
    for spec in specs:
        key, sep, val = spec.partition("=")
        if not sep or key not in keys:
            raise SystemExit(f"--fail-on expects METRIC=MAX with METRIC in {keys}: {spec!r}")
        try:
            limits[key] = float(val)
        except ValueError:
            raise SystemExit(f"--fail-on limit must be a number: {spec!r}") from None
        if not np.isfinite(limits[key]):
            raise SystemExit(f"--fail-on limit must be finite: {spec!r}")
    return limits


def flat_keys() -> list[str]:
    return ["fg_frac", "unlabeled_frac"] + FLAT_KEYS



def _is_zarr(src: str) -> bool:
    s = str(src)
    return s.rstrip("/").endswith(".zarr") or "://" in s or (Path(s) / ".zarray").exists() \
        or (Path(s) / ".zattrs").exists() or (Path(s) / "zarr.json").exists()


def open_volume(src: str, level: str = "0"):
    """Return an array-like with .shape/.ndim/__getitem__ for a .tif file or a zarr array/group."""
    if not _is_zarr(src):
        import tifffile
        return tifffile.imread(src)
    try:
        import zarr
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(f"zarr input needs the zarr package: {exc}")
    node = zarr.open(str(src), mode="r")
    if hasattr(node, "shape"):
        return node
    path = str(level)
    for ms in node.attrs.get("multiscales", []):
        datasets = ms.get("datasets", [])
        if str(level).isdigit() and int(level) < len(datasets):
            path = datasets[int(level)]["path"]
            break
    try:
        return node[path]
    except KeyError:
        raise SystemExit(f"{src}: no array {path!r} in the group (levels: {sorted(node.array_keys())})") from None


def parse_roi(spec: str | None, shape: Sequence[int]) -> tuple[slice, ...]:
    """'z0:z1,y0:y1,x0:x1' -> slices clipped to shape; empty bounds mean the full axis."""
    if not spec:
        return tuple(slice(0, n) for n in shape)
    parts = spec.split(",")
    if len(parts) != len(shape):
        raise SystemExit(f"--roi needs {len(shape)} comma-separated ranges, got {spec!r}")
    out = []
    for part, n in zip(parts, shape):
        lo, _, hi = part.partition(":")
        try:
            a = int(lo) if lo else 0
            b = int(hi) if hi else n
        except ValueError:
            raise SystemExit(f"--roi range must be integers: {part!r}") from None
        a, b = max(a, 0), min(b, n)
        if a >= b:
            raise SystemExit(f"--roi range {part!r} is empty within axis length {n}")
        out.append(slice(a, b))
    return tuple(out)


def tile_windows(roi: Sequence[slice], tile: int | None) -> list[tuple[slice, ...]]:
    """Full tile-sized cubes covering roi (whole roi when tile is None); partial far-edge cubes are dropped."""
    if tile is None:
        return [tuple(roi)]
    if tile < 1:
        raise SystemExit("--tile must be a positive integer")
    starts = [range(s.start, s.stop - tile + 1, tile) for s in roi]
    if any(len(r) == 0 for r in starts):
        raise SystemExit(f"--tile {tile} does not fit in the window {[(s.start, s.stop) for s in roi]}")
    return [tuple(slice(z, z + tile) for z in zs) for zs in itertools.product(*starts)]


def window_name(stem: str, win: Sequence[slice], tiled: bool) -> str:
    return stem if not tiled else f"{stem}_z{win[0].start}_y{win[1].start}_x{win[2].start}"


def read_window(vol, win: Sequence[slice], threshold: float | None = None) -> np.ndarray:
    arr = np.asarray(vol[tuple(win)])
    return binarise(arr, threshold)


def binarise(arr: np.ndarray, threshold: float | None) -> np.ndarray:
    return arr if threshold is None else (arr >= threshold).astype(np.uint8)


def parse_thresholds(spec: str | None) -> list[float] | None:
    if spec is None:
        return None
    try:
        vals = [float(t) for t in spec.split(",") if t.strip()]
    except ValueError:
        raise ValueError(f"--threshold expects numbers separated by commas, got {spec!r}")
    if not vals:
        raise ValueError("--threshold is empty")
    return vals


def volume_stem(src: str) -> str:
    name = str(src).rstrip("/").rsplit("/", 1)[-1]
    for suffix in (".zarr", ".tif", ".tiff"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="vesuvius.label_topology_audit",
        description="Audit topology (components, Betti numbers, thickness, fusion) of 3D surface labels.",
    )
    ap.add_argument("--label", help="label volume: .tif file, zarr array/group directory or URL")
    ap.add_argument("--image", help="matching image volume (same formats); only needed for --plot")
    ap.add_argument("--batch", type=Path, help="directory with labels/*.tif (and optional images/)")
    ap.add_argument("--after",
                    help="second label volume (batch: directory of same-named files); writes <crop>.delta.json")
    ap.add_argument("--roi", metavar="Z0:Z1,Y0:Y1,X0:X1",
                    help="window of --label/--after/--image to read (zarr input is read lazily)")
    ap.add_argument("--tile", type=int, metavar="N",
                    help="split the volume (or --roi) into N-voxel cubes audited as separate crops; "
                         "partial cubes at the far edge are dropped")
    ap.add_argument("--level", default="0", help="pyramid level of an OME-Zarr group (default 0)")
    ap.add_argument("--threshold", metavar="T[,T2,...]",
                    help="binarise the label input as value >= T (e.g. 128 for uint8 probabilities, "
                         "1 for 0/255 masks); implies --surface-value 1 and no ignore value. Several "
                         "thresholds audit every cut of each window from one read, as crops <name>_t<T>")
    ap.add_argument("--out", type=Path, default=Path("audit_out"), help="output directory")
    ap.add_argument("--surface-value", type=int, default=1, help="label value of the surface")
    ap.add_argument("--ignore-value", type=int, default=2,
                    help="label value of ignored/unlabeled voxels (-1 to disable)")
    ap.add_argument("--erode", type=int, default=1, help="erosion iterations for the fusion test")
    ap.add_argument("--min-piece", type=int, default=500, help="minimum voxels of a post-erosion piece")
    ap.add_argument("--plot", action="store_true", help="write overlay PNGs (needs --image / batch images)")
    ap.add_argument("--hole-masks", action="store_true",
                    help="write <crop>.hole_mask.tif (located holes = 1), the input format of scripts/fix_holes.py")
    ap.add_argument("--fail-on", action="append", default=[], metavar="METRIC=MAX",
                    help="exit 2 if a crop's metric (or its --after volume's) exceeds MAX (repeatable), e.g. b1_total_big=0")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    import tifffile

    ap = build_parser()
    a = ap.parse_args(argv)
    if bool(a.label) == bool(a.batch):
        ap.error("give exactly one of --label or --batch")
    if a.batch and (a.roi or a.tile or a.threshold is not None):
        ap.error("--roi, --tile and --threshold apply to --label input only")
    limits = _parse_fail_on(a.fail_on)
    ignore = None if a.ignore_value < 0 else a.ignore_value
    surface_value = a.surface_value
    try:
        thresholds = parse_thresholds(a.threshold)
    except ValueError as exc:
        ap.error(str(exc))
    if thresholds is not None:
        surface_value, ignore = 1, None
    sweep = thresholds is not None and len(thresholds) > 1

    # each job: (name, label window, image window or None, after window or None), windows as (volume, slices)
    if a.batch:
        labels = sorted((a.batch / "labels").glob("*.tif"))
        if not labels:
            ap.error(f"no labels/*.tif under {a.batch}")
        after_dir = Path(a.after) if a.after else None
        missing = [str(after_dir / p.name) for p in labels if after_dir and not (after_dir / p.name).exists()]
        if missing:
            ap.error(f"--after is missing {len(missing)} paired file(s), e.g. {missing[0]}")
        jobs = []
        for p in labels:
            img = a.batch / "images" / p.name
            jobs.append((p.stem, (p, None), (img, None) if a.plot and img.exists() else None,
                         (after_dir / p.name, None) if after_dir else None))
    else:
        lab_vol = open_volume(a.label, a.level)
        if lab_vol.ndim != 3:
            ap.error(f"{a.label}: expected a 3D volume, got shape {lab_vol.shape}")
        after_vol = open_volume(a.after, a.level) if a.after else None
        img_vol = open_volume(a.image, a.level) if a.plot and a.image else None
        for tag, vol in (("--after", after_vol), ("--image", img_vol)):
            if vol is not None and tuple(vol.shape) != tuple(lab_vol.shape):
                ap.error(f"{tag} shape {tuple(vol.shape)} differs from label shape {tuple(lab_vol.shape)}")
        roi = parse_roi(a.roi, lab_vol.shape)
        wins = tile_windows(roi, a.tile)
        stem = volume_stem(a.label)
        jobs = [(window_name(stem, w, a.tile is not None), (lab_vol, w),
                 (img_vol, w) if img_vol is not None else None,
                 (after_vol, w) if after_vol is not None else None) for w in wins]

    def load(window):
        vol, w = window
        if w is None:  # batch: a .tif path read whole
            return tifffile.imread(vol)
        return read_window(vol, w)

    # one read per window, taken lazily so only the current window's cuts are in memory;
    # with several thresholds every cut becomes its own crop
    def iter_cuts():
        for name, label_win, image_win, after_win in jobs:
            raw = load(label_win)
            raw_after = load(after_win) if after_win else None
            if raw_after is not None and raw_after.shape != raw.shape:
                ap.error(f"{name}: --after shape {raw_after.shape} differs from label shape {raw.shape}")
            img = None
            if a.plot and image_win is not None:
                img = load(image_win)
                if img.shape != raw.shape:
                    ap.error(f"{name}: image shape {img.shape} differs from label shape {raw.shape}")
            for t in (thresholds or [None]):
                cut = f"{name}_t{t:g}" if sweep else name
                yield (cut, binarise(raw, t), binarise(raw_after, t) if raw_after is not None else None,
                       img, after_win is not None)

    rows, failed = [], False
    for name, lab, after_lab, img, has_after in iter_cuts():
        try:
            m, flags = audit(lab, a.erode, a.min_piece, surface_value, ignore)
            if after_lab is not None:
                m2, _ = audit(after_lab, a.erode, a.min_piece, surface_value, ignore)
        except ValueError as exc:
            ap.error(f"{name}: {exc}")
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / f"{name}.json").write_text(json.dumps(m, indent=1))
        f = flat(m)
        rows.append({"crop": name, **f})
        print(name, json.dumps(f), flush=True)
        if a.plot:
            if img is None:
                print(f"{name}: --plot skipped, no image", file=sys.stderr)
            else:
                overlay(img, lab, flags, a.out / f"{name}.png", surface_value)
                if m["holes"]:
                    hole_tiles(img, lab, m["holes"], flags["hole"], a.out / f"{name}.holes.png",
                               surface_value, -1 if ignore is None else ignore)
        if a.hole_masks:
            tifffile.imwrite(a.out / f"{name}.hole_mask.tif", flags["hole"].astype(np.uint8))
        if has_after:
            f2 = flat(m2)
            delta = {k: [f[k], f2[k]] for k in f}
            (a.out / f"{name}.delta.json").write_text(json.dumps(delta, indent=1))
            print("delta (before, after):", json.dumps(delta))
        checked = [("", f)] + ([(" (after)", f2)] if has_after else [])
        for tag, vals in checked:
            for key, limit in limits.items():
                if vals[key] > limit:
                    print(f"FAIL {name}{tag}: {key}={vals[key]} > {limit}", file=sys.stderr)
                    failed = True
    with open(a.out / "aggregate.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    if a.batch or a.tile is not None or sweep:
        top = ranked(rows)
        with open(a.out / "ranked.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["rank", "crop", "reason"] + [k for k, _ in SEVERITY])
            w.writeheader()
            w.writerows(top)
        print(f"{len(top)} of {len(rows)} crops have findings; worst first in {a.out / 'ranked.csv'}")
    return 2 if failed else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
