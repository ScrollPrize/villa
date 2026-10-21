"""Measure where a traced surface sits relative to the papyrus sheet in the CT.

For a sample of surface points the CT is read along the surface normal and the
nearest density peak (the sheet) is located. The signed distance from the
surface point to that peak, along the stored normal, is the placement offset.
A surface that tracks one sheet has offsets that are small and constant; a
surface that wiggles, drifts across fibre layers or leaves the sheet shows
offsets that spread, change sign or hit the search limit.

Inputs: a VolCart ``.obj`` (local path or ``http(s)://`` URL; the vertex block
is range-sampled so a large remote file is not downloaded) or a tifxyz
directory, plus the CT as a zarr array or OME-Zarr group (path or URL).

Outputs per surface: ``<name>.json`` (summary), ``<name>.samples.csv`` (one row
per sample with coordinates, so the offsets can be viewed in place), and
``aggregate.csv`` across surfaces. ``--plot`` adds ``<name>.png``.

Assumes an isotropic voxel grid: offsets, smoothing and tangent radii are in voxels, not physical units.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import threading
import urllib.request
from typing import Any, Sequence

import numpy as np

from vesuvius.image_proc.run.label_topology_audit import _parse_fail_on, open_volume

GRID_SPIKE_FACTOR = 4.0  # a step longer than this times the median step ties a spike group
GRID_KEYS = ("n_grid_spikes", "grid_step_median", "grid_step_max", "n_grid_holes", "grid_hole_points")

METRICS = (
    "n_samples", "n_inside_volume", "n_valid", "no_peak_frac", "n_censored", "censored_frac",
    "offset_median", "offset_q25", "offset_q75", "offset_mean_abs",
    "offset_abs_gt_1_frac", "offset_abs_gt_2_frac", "offset_abs_gt_3_frac", "offset_at_limit_frac",
    "off_sheet_frac", "sheet_fwhm_median", "contrast_median",
    "n_patches", "patch_offset_q25", "patch_offset_median", "patch_offset_q75",
    "patch_offset_range", "patch_abs_gt_2_frac", "n_worst_cells",
    "snap_remeasure_sd", "snap_remeasure_abs_gt_2_frac",
    "tangent_absdiff_median_2", "tangent_absdiff_median_4", "tangent_absdiff_median_8",
    "tangent_corr_2", "tangent_corr_4", "tangent_corr_8",
    "n_grid_spikes", "grid_step_median", "grid_step_max", "n_grid_holes", "grid_hole_points",
    "outside_volume_frac",
)
TANGENT_RADII = (2.0, 4.0, 8.0)


def parse_fail_on(specs: Sequence[str]) -> dict[str, float]:
    return _parse_fail_on(specs, METRICS)


# --------------------------------------------------------------------------- surfaces


HTTP_TIMEOUT = 60


def _http_size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as r:
        size = r.headers.get("Content-Length")
    if size is None:
        raise ValueError(f"{url}: server gives no Content-Length; range sampling needs it")
    return int(size)


def _http_range(url: str, start: int, length: int) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{start + length - 1}"})
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as r:
        if r.status != 206:
            raise ValueError(f"{url}: server ignored the Range header (status {r.status}); range sampling needs it")
        return r.read(length)


def _parse_v_vn(blob: bytes, at_start: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Vertices and normals from a byte block of ``v``/``vn`` lines (pairs only).

    Returns (points xyz, normals xyz); a vertex without a following ``vn`` is dropped.
    """
    # drop the partial first and last lines (a blob starting at byte 0 has a whole first line)
    lines = blob.split(b"\n")
    lines = lines[(0 if at_start else 1):-1]
    pts, nrm = [], []
    last_v = None
    for ln in lines:
        if ln.startswith(b"vn "):
            if last_v is not None:
                try:
                    vals = [float(t) for t in ln[3:].split()]
                    if len(vals) >= 3:
                        nrm.append(vals[:3])
                        pts.append(last_v)
                except ValueError:
                    pass
            last_v = None
        elif ln.startswith(b"v "):
            try:
                vals = [float(t) for t in ln[2:].split()]
                last_v = vals[:3] if len(vals) >= 3 else None
            except ValueError:
                last_v = None
        else:
            last_v = None
    if not pts:
        return np.zeros((0, 3)), np.zeros((0, 3))
    P, N = np.asarray(pts, np.float64), np.asarray(nrm, np.float64)
    ok = np.isfinite(P).all(1) & np.isfinite(N).all(1) & (np.linalg.norm(N, axis=1) > 0)
    return P[ok], N[ok]


def _vertex_region_end(reader, size: int, probe: int = 4096) -> int:
    """Byte offset where the ``v``/``vn`` block ends (first ``f``/``vt`` line), by bisection."""

    def is_vertex_block(pos: int) -> bool:
        blob = reader(pos, probe)
        lines = [ln for ln in blob.split(b"\n")[1:-1] if ln]
        if not lines:
            return False
        return all(ln.startswith((b"v ", b"vn ")) for ln in lines)

    if size <= 2 * probe:  # small file: scan it whole, no bisection
        blob, pos = reader(0, size), 0
        for ln in blob.split(b"\n"):
            if not ln.startswith((b"v ", b"vn ", b"#")):
                break
            pos += len(ln) + 1
        return min(pos, size)
    lo, hi = 0, size
    if not is_vertex_block(0):
        # header comments first: skip ahead a little
        lo = min(size, 2048)
    while hi - lo > probe:
        mid = (lo + hi) // 2
        if is_vertex_block(mid):
            lo = mid
        else:
            hi = mid
    # exact end: scan the last window line by line
    blob = reader(lo, min(2 * probe, size - lo))
    nl = blob.find(b"\n")
    if lo > 0 and nl < 0:
        raise ValueError("obj vertex block: no line break in the probed window")
    pos = lo if lo == 0 else lo + nl + 1
    for ln in blob[pos - lo:].split(b"\n"):
        if not ln.startswith((b"v ", b"vn ", b"#")):
            break
        pos += len(ln) + 1
    return min(pos, size)


def sample_obj(src: str, samples: int, blobs: int, rng: np.random.Generator, blob_bytes: int = 1 << 19):
    """Random surface points and normals from an obj, reading ``blobs`` byte ranges of the vertex block.

    ``samples`` points are drawn evenly across the ranges. Returns
    (points zyx, normals zyx, 32-voxel cell id per point, n_vertices_declared).
    """
    if src.startswith(("http://", "https://")):
        size = _http_size(src)
        return _sample_obj(src, size, lambda s, n: _http_range(src, s, n), samples, blobs, rng, blob_bytes)
    size = os.path.getsize(src)
    with open(src, "rb") as fh:

        def reader(s, n):
            fh.seek(s)
            return fh.read(n)

        return _sample_obj(src, size, reader, samples, blobs, rng, blob_bytes)


def _sample_obj(src, size, reader, samples, blobs, rng, blob_bytes):
    head = reader(0, 4096)
    m = re.search(rb"# Vertices: (\d+)", head)
    declared = int(m.group(1)) if m else -1
    end = _vertex_region_end(reader, size)
    if end <= 0:
        raise ValueError(f"{src}: no v/vn vertex block found")
    blobs = max(1, min(blobs, max(1, end // blob_bytes)))
    starts = np.sort(rng.integers(0, max(1, end - blob_bytes), blobs)) if end > blob_bytes else np.array([0])
    per = max(1, math.ceil(samples / len(starts)))
    P, N = [], []
    for s in starts:
        pts, nrm = _parse_v_vn(reader(int(s), min(blob_bytes, end - int(s)) + 1), at_start=int(s) == 0)
        if len(pts) == 0:
            continue
        pick = rng.choice(len(pts), min(per, len(pts)), replace=False)
        P.append(pts[pick]); N.append(nrm[pick])
    if not P:
        raise ValueError(f"{src}: vertex block yielded no v/vn pairs")
    P = np.concatenate(P)[:, ::-1]
    N = np.concatenate(N)[:, ::-1]
    N /= np.linalg.norm(N, axis=1, keepdims=True) + 1e-12
    return P, N, cell_ids(P), declared


def grid_step_check(x: np.ndarray, y: np.ndarray, z: np.ndarray, has_point: np.ndarray,
                    factor: float = GRID_SPIKE_FACTOR, max_group_frac: float = 0.05) -> dict[str, Any]:
    """Geometry of a tifxyz grid without the CT. Adjacent valid grid points are joined
    when their distance is at most ``factor`` times the median step; the largest
    connected group is the surface. Any other group that is still tied to the rest by
    a long step is a spike: vertices thrown off the surface, singly or as a run, the
    class villa PR #2067 found after resampling. CT sampling cannot see them because
    they sit at hole edges and have no normal. Groups with no tie at all are islands
    of the grid, not spikes, and are left alone."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    P = np.stack([z, y, x], -1).astype(np.float64)
    n = has_point.size
    idx = np.arange(n).reshape(has_point.shape)
    a_all, b_all, d_all = [], [], []
    for axis in (0, 1):
        sl_a = [slice(None)] * 2
        sl_b = [slice(None)] * 2
        sl_a[axis] = slice(0, -1)
        sl_b[axis] = slice(1, None)
        both = has_point[tuple(sl_a)] & has_point[tuple(sl_b)]
        a_all.append(idx[tuple(sl_a)][both])
        b_all.append(idx[tuple(sl_b)][both])
        d_all.append(np.linalg.norm(P[tuple(sl_b)][both] - P[tuple(sl_a)][both], axis=-1))
    a, b, d = (np.concatenate(v) for v in (a_all, b_all, d_all))
    if len(d) == 0:
        return {"grid_step_median": float("nan"), "grid_step_max": float("nan"), "n_grid_spikes": 0,
                "grid_spike_rc": np.zeros((0, 2), int)}
    med = float(np.median(d[d > 0])) if (d > 0).any() else 0.0  # duplicate vertices give zero steps
    if med == 0.0:
        return {"grid_step_median": 0.0, "grid_step_max": round(float(d.max()), 3), "n_grid_spikes": 0,
                "grid_spike_rc": np.zeros((0, 2), int)}
    short = d <= factor * med
    graph = coo_matrix((np.ones(short.sum()), (a[short], b[short])), shape=(n, n))
    _, comp = connected_components(graph, directed=False)
    comp[~has_point.ravel()] = -1
    sizes = np.bincount(comp[comp >= 0])
    largest = int(np.argmax(sizes))
    tied = np.zeros(len(sizes), bool)  # groups with a long step to another group
    la, lb = comp[a[~short]], comp[b[~short]]
    cross = la != lb
    tied[la[cross]] = True
    tied[lb[cross]] = True
    tied[largest] = False
    tied &= sizes < max_group_frac * sizes[largest]  # a large group joined by a long step is a fold, not a spike
    spike = (comp >= 0) & tied[np.maximum(comp, 0)]
    return {"grid_step_median": round(med, 3), "grid_step_max": round(float(d.max()), 3),
            "n_grid_spikes": int(spike.sum()), "grid_spike_rc": np.argwhere(spike.reshape(has_point.shape))}


def grid_hole_check(has_point: np.ndarray) -> dict[str, Any]:
    """Interior holes of a tifxyz grid: runs of invalid points enclosed by valid ones,
    found by 4-connected labelling of the invalid mask minus the groups that touch
    the grid border. A flattening whose quality filter punched holes through the
    surface (villa PR #2128) exits 0 and looks fine on a step check; this counts it."""
    from scipy import ndimage as ndi

    lab, n = ndi.label(~has_point)
    if n == 0:
        return {"n_grid_holes": 0, "grid_hole_points": 0}
    border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    interior = np.setdiff1d(np.arange(1, n + 1), border)
    return {"n_grid_holes": int(len(interior)),
            "grid_hole_points": int(np.isin(lab, interior).sum())}


def sample_tifxyz(path: str, samples: int, rng: np.random.Generator):
    """Random points and normals from a tifxyz directory at its stored resolution."""
    from vesuvius.tifxyz import read_tifxyz

    surf = read_tifxyz(path)
    nx, ny, nz = surf.compute_normals()
    x, y, z = surf._x, surf._y, surf._z
    has_point = np.isfinite(x) & (x > 0) & (y > 0) & (z > 0)
    if surf._mask is not None:
        has_point &= surf._mask.astype(bool)
    grid = grid_step_check(x, y, z, has_point)
    grid.update(grid_hole_check(has_point))
    ok = has_point & np.isfinite(nx)
    # grid points with no computable normal (hole edges, grid border) are not sampled;
    # the count is reported so a defect that lives there is not silently skipped
    n_no_normal = int(has_point.sum() - ok.sum())
    idx = np.flatnonzero(ok)
    if len(idx) == 0:
        raise ValueError(f"{path}: no valid surface points")
    pick = rng.choice(idx, min(samples, len(idx)), replace=False)
    r, c = np.unravel_index(pick, x.shape)
    P = np.stack([z.ravel()[pick], y.ravel()[pick], x.ravel()[pick]], 1).astype(np.float64)
    N = np.stack([nz.ravel()[pick], ny.ravel()[pick], nx.ravel()[pick]], 1).astype(np.float64)
    N /= np.linalg.norm(N, axis=1, keepdims=True) + 1e-12
    return P, N, cell_ids(P), int(ok.sum()), n_no_normal, grid


def cell_ids(P: np.ndarray, size: float = 32.0) -> np.ndarray:
    """Spatial patch id: the ``size``-voxel cell each point falls in."""
    c = np.floor(P / size).astype(np.int64)
    return c[:, 0] * 10**10 + c[:, 1] * 10**5 + c[:, 2]


def worst_cells(P: np.ndarray, N: np.ndarray, off: np.ndarray, valid: np.ndarray, patch: np.ndarray,
                threshold: float = 2.0, min_points: int = 30, limit: int = 50) -> list[dict[str, Any]]:
    """The 32-voxel cells whose median offset exceeds ``threshold``, worst first.

    Each entry carries the cell's mean sampled position (``center_zyx``), its sample count and
    its median offset, so an annotator can go to the spot and pull the surface back on the
    sheet. Cells with fewer than ``min_points`` valid samples are not judged, as in
    ``summarize``.
    """
    out = []
    for k in np.unique(patch[valid]):
        sel = valid & (patch == k)
        s = off[sel]
        if len(s) < min_points:
            continue
        med = float(np.median(s))
        if abs(med) > threshold:
            n = N[sel].mean(0)
            out.append({"center_zyx": [float(v) for v in P[sel].mean(0)],
                        "normal_zyx": [float(v) for v in n / (np.linalg.norm(n) + 1e-12)],
                        "n": int(len(s)), "median_offset": med})
    out.sort(key=lambda c: -abs(c["median_offset"]))
    return out[:limit]


def write_point_collection(path: str, name: str, cells: Sequence[dict[str, Any]]) -> None:
    """Write cells as a VC3D point collection (``vc_pointcollections_json_version`` 1).

    One collection named ``name``, one point per cell in VC3D's x, y, z order and the
    surface's own coordinate frame; ``creation_time`` is the rank so the widget lists the
    worst cell first. Loads from the Point Collections panel in VC3D. ``cells`` are the
    ``worst_cells`` dicts, and ``placement_edits.py`` reads the same file back.
    """
    points = {str(i): {"p": [c["center_zyx"][2], c["center_zyx"][1], c["center_zyx"][0]],
                       "creation_time": i, "wind_a": None,
                       "audit": {k: c[k] for k in ("normal_zyx", "n", "median_offset") if k in c}}
              for i, c in enumerate(cells)}
    doc = {"vc_pointcollections_json_version": "1",
           "collections": {"0": {"name": name, "points": points,
                                 "metadata": {"winding_is_absolute": False}, "color": [1.0, 0.0, 0.0]}}}
    with open(path, "w") as f:
        json.dump(doc, f, indent=1)


def tangent_points(P: np.ndarray, N: np.ndarray, radius: float, rng: np.random.Generator) -> np.ndarray:
    """A point ``radius`` voxels from each P along a random direction in its tangent plane."""
    r = rng.standard_normal(P.shape)
    r -= (r * N).sum(1, keepdims=True) * N
    r /= np.linalg.norm(r, axis=1, keepdims=True) + 1e-12
    return P + radius * r


def is_tifxyz(src: str) -> bool:
    return os.path.isdir(src) and all(os.path.exists(os.path.join(src, f)) for f in ("x.tif", "y.tif", "z.tif"))


# --------------------------------------------------------------------------- chunk cache


class ChunkCache:
    """Array-like over ``vol`` that keeps every native chunk it reads as a ``.npy`` under ``root``.

    A second run against the same remote volume (another surface, the controls'
    re-reads, a rerun with other settings) then reads from disk instead of the
    server. Files are keyed by the volume's source (path or URL, plus size and
    mtime for a local file), level, shape, chunk size and dtype, so a different
    volume or level gets its own directory even when the geometry is identical. Leading non-spatial axes
    are read at index 0. Writes go through a temp file and ``os.replace`` so
    concurrent threads never see a partial chunk.
    """

    def __init__(self, vol, root: str, default_chunk: int = 128, source: str = "", level: str = "0"):
        self.vol = vol
        self.ndim = vol.ndim
        self.shape = tuple(int(n) for n in vol.shape)
        self.dtype = np.dtype(vol.dtype)
        self.spatial = self.shape[-3:]
        chunks = getattr(vol, "chunks", None) or (default_chunk,) * 3
        self.chunks = tuple(int(c) for c in chunks[-3:])
        ident = f"{source}|{level}"
        if source and os.path.exists(source):
            st = os.stat(source)
            ident += f"|{st.st_size}|{st.st_mtime_ns}"
        digest = hashlib.sha1(ident.encode()).hexdigest()[:12]
        tag = digest + "_" + "x".join(map(str, self.spatial)) + "_c" + "x".join(map(str, self.chunks)) + "_" + self.dtype.name
        self.root = os.path.join(root, tag)
        os.makedirs(self.root, exist_ok=True)
        self.fetched = 0

    def _chunk(self, c: tuple[int, int, int]) -> np.ndarray:
        path = os.path.join(self.root, f"{c[0]}_{c[1]}_{c[2]}.npy")
        if os.path.exists(path):
            return np.load(path)
        lo = [ci * s for ci, s in zip(c, self.chunks)]
        hi = [min(l + s, n) for l, s, n in zip(lo, self.chunks, self.spatial)]
        index = (0,) * (self.ndim - 3) + tuple(slice(l, h) for l, h in zip(lo, hi))
        arr = np.asarray(self.vol[index]).reshape([h - l for l, h in zip(lo, hi)])
        tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
        with open(tmp, "wb") as f:
            np.save(f, arr)
        os.replace(tmp, path)
        self.fetched += 1
        return arr

    def __getitem__(self, index) -> np.ndarray:
        sl = index[-3:]
        lo = [int(s.start or 0) for s in sl]
        hi = [int(n if s.stop is None else s.stop) for s, n in zip(sl, self.spatial)]
        out = np.empty([h - l for l, h in zip(lo, hi)], self.dtype)
        ranges = [range(l // c, (h - 1) // c + 1) for l, h, c in zip(lo, hi, self.chunks)]
        for cz in ranges[0]:
            for cy in ranges[1]:
                for cx in ranges[2]:
                    blk = self._chunk((cz, cy, cx))
                    b0 = [ci * s for ci, s in zip((cz, cy, cx), self.chunks)]
                    src = tuple(slice(max(l - b, 0), min(h - b, n)) for l, h, b, n in zip(lo, hi, b0, blk.shape))
                    dst = tuple(slice(max(b - l, 0), max(b - l, 0) + (sr.stop - sr.start))
                                for sr, l, b in zip(src, lo, b0))
                    out[dst] = blk[src]
        return out


# --------------------------------------------------------------------------- profiles


def read_profiles(vol, P: np.ndarray, N: np.ndarray, half: float, step: float, sigma: float, chunk: int = 128,
                  workers: int = 8):
    """CT profile along each normal, t in [-half, half] (step), from a lazily read volume.

    Points are grouped by ``chunk``-voxel cell so each cell's padded box is read once;
    cells are read on ``workers`` threads (remote zarr reads are latency-bound).
    Returns (profiles, t, kept) where ``kept`` marks points inside the volume.
    """
    from scipy import ndimage as ndi

    # symmetric grid through zero, bounded by half (np.arange would overshoot for a non-integer ratio)
    pos_t = np.arange(0, half + 1e-9, step)
    t = np.concatenate([-pos_t[:0:-1], pos_t])
    shape = np.asarray(vol.shape[-3:])
    pad = int(math.ceil(half + 3 * sigma + 2))
    prof = np.full((len(P), len(t)), np.nan, np.float32)
    inside = np.all((P >= pad) & (P <= shape - 1 - pad), 1)
    cell = np.floor(P / chunk).astype(np.int64)
    key = cell[:, 0] * 10**8 + cell[:, 1] * 10**4 + cell[:, 2]
    def one_cell(k):
        sel = np.flatnonzero((key == k) & inside)
        lo = np.maximum(P[sel].min(0).astype(int) - pad, 0)
        hi = np.minimum(P[sel].max(0).astype(int) + pad + 1, shape)
        index = (0,) * (vol.ndim - 3) + tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
        box = np.asarray(vol[index], dtype=np.float32)
        if box.ndim > 3:
            box = box.reshape(box.shape[-3:])
        if sigma > 0:
            box = ndi.gaussian_filter(box, sigma)
        q = (P[sel] - lo)[:, None, :] + t[None, :, None] * N[sel][:, None, :]
        prof[sel] = ndi.map_coordinates(box, q.reshape(-1, 3).T, order=1, mode="nearest").reshape(len(sel), len(t))

    cells = np.unique(key[inside])
    if workers > 1 and len(cells) > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(one_cell, cells))
    else:
        for k in cells:
            one_cell(k)
    return prof, t, inside


def nearest_peak(prof: np.ndarray, t: np.ndarray, min_contrast: float, min_prominence: float):
    """Offset of the nearest local maximum of each profile to t=0, and its contrast.

    A peak counts when the profile range is at least ``min_contrast`` and the peak
    rises at least ``min_prominence`` x range above the higher of its two flanking minima
    (the prominence, so a shoulder on a bigger peak does not count).
    Returns (offset, contrast, valid); offset is NaN when no peak qualifies.
    """
    n, m = prof.shape
    has = np.isfinite(prof).any(1)
    rng_ = np.full(n, np.nan)
    rng_[has] = np.nanmax(prof[has], 1) - np.nanmin(prof[has], 1)
    off = np.full(n, np.nan)
    val = np.zeros(n, bool)
    # local maxima (plateaus count once, at their first sample)
    left = np.hstack([np.full((n, 1), -np.inf), prof[:, :-1]])
    right = np.hstack([prof[:, 1:], np.full((n, 1), -np.inf)])
    ismax = (prof > left) & (prof >= right)
    order = np.argsort(np.abs(t), kind="stable")  # nearest to t=0, not to the middle index
    for i in range(n):
        if not np.isfinite(rng_[i]) or rng_[i] < min_contrast:
            continue
        p = prof[i]
        for j in order:
            if not ismax[i, j]:
                continue
            lmin = p[: j + 1].min()
            rmin = p[j:].min()
            if p[j] - max(lmin, rmin) >= min_prominence * rng_[i]:
                off[i] = t[j]
                val[i] = True
                break
    return off, rng_, val


def censored_profiles(prof: np.ndarray, contrast: np.ndarray, valid: np.ndarray, min_contrast: float) -> np.ndarray:
    """Profiles with enough contrast and no accepted peak whose maximum is an end sample.

    The sheet may lie at or beyond the search limit; such a profile cannot be told from a rising
    flank, so it is not an offset (``nearest_peak`` is unchanged) but it is not "no peak" either.
    """
    cen = np.zeros(len(prof), bool)
    ok = np.isfinite(contrast) & (contrast >= min_contrast) & ~valid
    for i in np.flatnonzero(ok):
        j = int(np.nanargmax(prof[i]))
        cen[i] = j == 0 or j == prof.shape[1] - 1
    return cen


def summarize(off: np.ndarray, contrast: np.ndarray, valid: np.ndarray, inside: np.ndarray, patch: np.ndarray,
              prof: np.ndarray, t: np.ndarray, half: float, min_contrast: float = 0.0) -> dict[str, Any]:
    o = off[valid]
    q = lambda x, p: float(np.percentile(x, p)) if len(x) else float("nan")  # noqa: E731
    # sheet width at the located peak: half-maximum width of the profile around it
    fw = []
    for i in np.flatnonzero(valid)[:20000]:
        p = prof[i]
        j = int(np.argmin(np.abs(t - off[i])))
        hm = (p[j] + np.nanmin(p)) / 2
        a = j
        while a > 0 and p[a - 1] >= hm:
            a -= 1
        b = j
        while b < len(p) - 1 and p[b + 1] >= hm:
            b += 1
        fw.append(t[b] - t[a])
    fw = np.asarray(fw)
    # on-sheet: the CT at the surface point is at least the half-maximum of its peak
    on = []
    for i in np.flatnonzero(valid):
        p = prof[i]
        j = int(np.argmin(np.abs(t - off[i])))
        on.append(np.interp(0.0, t, p) >= (p[j] + np.nanmin(p)) / 2)  # t=0 need not be a sample
    on = np.asarray(on, bool)
    # per-patch median offset: where the surface drifts
    pm = []
    for k in np.unique(patch[valid]):
        s = off[valid & (patch == k)]
        if len(s) >= 30:
            pm.append(float(np.median(s)))
    pm = np.asarray(pm)
    cen = censored_profiles(prof, contrast, valid, min_contrast) & inside
    out = {
        "n_samples": int(len(off)),
        "n_inside_volume": int(inside.sum()),
        "outside_volume_frac": float(1 - inside.sum() / max(1, len(inside))),
        "n_valid": int(valid.sum()),
        "no_peak_frac": float(1 - valid.sum() / max(1, inside.sum())),
        "n_censored": int(cen.sum()),
        "censored_frac": float(cen.sum() / max(1, inside.sum())),
        "offset_median": q(o, 50), "offset_q25": q(o, 25), "offset_q75": q(o, 75),
        "offset_mean_abs": float(np.mean(np.abs(o))) if len(o) else float("nan"),
        "offset_abs_gt_1_frac": float(np.mean(np.abs(o) > 1)) if len(o) else float("nan"),
        "offset_abs_gt_2_frac": float(np.mean(np.abs(o) > 2)) if len(o) else float("nan"),
        "offset_abs_gt_3_frac": float(np.mean(np.abs(o) > 3)) if len(o) else float("nan"),
        "offset_at_limit_frac": float(np.mean(np.abs(o) >= half)) if len(o) else float("nan"),
        "off_sheet_frac": float(1 - on.mean()) if len(on) else float("nan"),
        "sheet_fwhm_median": float(np.median(fw)) if len(fw) else float("nan"),
        "contrast_median": float(np.nanmedian(contrast[inside])) if inside.any() else float("nan"),
        "n_patches": int(len(pm)),
        "patch_offset_q25": q(pm, 25), "patch_offset_median": q(pm, 50), "patch_offset_q75": q(pm, 75),
        "patch_offset_range": float(np.ptp(pm)) if len(pm) else float("nan"),
        "patch_abs_gt_2_frac": float(np.mean(np.abs(pm) > 2)) if len(pm) else float("nan"),
    }
    return out


def controls(vol, P, N, off, valid, a, rng) -> dict[str, float]:
    """Two checks on the offsets themselves.

    Snap control: each valid point moved to its located peak and measured again; the
    spread that remains is the precision of the peak location, not of the surface.
    Tangent check: a second point 2, 4 and 8 voxels away in the tangent plane; the
    agreement of its offset with the first says over what distance the offset is a
    property of the surface (drift) rather than of one point (noise or sheet texture).
    """
    out: dict[str, float] = {k: float("nan") for k in METRICS if k.startswith(("snap_", "tangent_"))}
    idx = np.flatnonzero(valid)
    if len(idx) < 30:
        return out
    if len(idx) > 5000:
        idx = rng.choice(idx, 5000, replace=False)
    P0, N0, o0 = P[idx], N[idx], off[idx]
    extra = [P0 + o0[:, None] * N0] + [tangent_points(P0, N0, r, rng) for r in TANGENT_RADII]
    Q = np.concatenate(extra)
    NQ = np.tile(N0, (len(extra), 1))
    prof, t, inside = read_profiles(vol, Q, NQ, a.half, a.step, a.sigma, workers=a.workers)
    offq, _, vq = nearest_peak(prof, t, a.min_contrast, a.min_prominence)
    vq &= inside
    n = len(idx)
    s_off, s_ok = offq[:n], vq[:n]
    if s_ok.sum() >= 30:
        out["snap_remeasure_sd"] = float(np.std(s_off[s_ok]))
        out["snap_remeasure_abs_gt_2_frac"] = float(np.mean(np.abs(s_off[s_ok]) > 2))
    for k, r in enumerate(TANGENT_RADII, start=1):
        t_off, t_ok = offq[k * n:(k + 1) * n], vq[k * n:(k + 1) * n]
        if t_ok.sum() >= 30:
            d = o0[t_ok] - t_off[t_ok]
            tag = f"{int(r)}"
            out[f"tangent_absdiff_median_{tag}"] = float(np.median(np.abs(d)))
            out[f"tangent_corr_{tag}"] = float(np.corrcoef(o0[t_ok], t_off[t_ok])[0, 1]) if o0[t_ok].std() > 0 and t_off[t_ok].std() > 0 else float("nan")
    return out


def plot(P, off, valid, path: str, name: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    v = valid
    sc = None
    for a, (i, j, lab) in zip(ax, [(2, 1, "x / y"), (2, 0, "x / z")]):
        sc = a.scatter(P[v, i], P[v, j], c=off[v], s=2, cmap="coolwarm", vmin=-4, vmax=4)
        a.scatter(P[~v, i], P[~v, j], c="0.7", s=1)
        a.set_xlabel(lab.split(" / ")[0]); a.set_ylabel(lab.split(" / ")[1]); a.set_aspect("equal")
    if sc is not None:
        fig.colorbar(sc, ax=ax, label="offset to sheet peak (vox, along normal)")
    fig.suptitle(f"{name}: surface placement (grey = no peak found)")
    fig.savefig(path, dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------- CLI


def surface_name(src: str) -> str:
    s = src.rstrip("/")
    base = s.rsplit("/", 1)[-1]
    return base[:-4] if base.endswith(".obj") else base


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--surface", nargs="+", required=True, help="obj file(s) (path or URL) or tifxyz directories")
    p.add_argument("--volume", required=True, help="CT zarr array or OME-Zarr group (path or URL)")
    p.add_argument("--level", default="0", help="multiscale level of an OME-Zarr group (default 0)")
    p.add_argument("--out", required=True)
    p.add_argument("--samples", type=int, default=20000, help="surface points per surface (default 20000)")
    p.add_argument("--blobs", type=int, default=40, help="byte ranges read from a remote/large obj (default 40)")
    p.add_argument("--half", type=float, default=8.0, help="search half-length along the normal, voxels (default 8)")
    p.add_argument("--step", type=float, default=0.5, help="profile step, voxels (default 0.5)")
    p.add_argument("--sigma", type=float, default=1.0, help="CT smoothing before profiling (default 1.0)")
    p.add_argument("--min-contrast", type=float, default=8.0, help="profile range below which no peak is accepted")
    p.add_argument("--min-prominence", type=float, default=0.15, help="peak prominence as a fraction of the profile range")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=8, help="threads for volume reads (default 8)")
    p.add_argument("--cache", metavar="DIR", help="keep every CT chunk read under DIR (.npy per chunk); "
                   "reruns against the same volume read from disk instead of the server")
    p.add_argument("--plot", action="store_true")
    p.add_argument("--worst-cells", type=int, default=50, metavar="N",
                   help="write up to N cells whose median offset exceeds 2 voxels, worst first, as "
                        "<name>.worst_cells.csv and a VC3D point collection <name>.worst_cells.json "
                        "(0 disables)")
    p.add_argument("--fail-on", action="append", default=[], metavar="METRIC=MAX",
                   help="exit 2 if any surface exceeds MAX for METRIC (e.g. offset_abs_gt_2_frac=0.5)")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    if a.half <= 0 or a.step <= 0 or a.samples <= 0:
        print("--half, --step and --samples must be positive", file=sys.stderr)
        return 1
    limits = parse_fail_on(a.fail_on)
    vol = open_volume(a.volume, a.level)
    if a.cache:
        vol = ChunkCache(vol, a.cache, source=a.volume, level=a.level)
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    rows: list[dict[str, Any]] = []
    failed = False
    for src in a.surface:
        name = surface_name(src)
        n_no_normal = 0
        grid: dict[str, Any] = {k: float("nan") for k in GRID_KEYS}  # obj input has no grid
        if is_tifxyz(src):
            P, N, patch, n_vert, n_no_normal, grid = sample_tifxyz(src, a.samples, rng)
            if n_no_normal:
                print(f"{name}: {n_no_normal} grid points have no normal and were not sampled", file=sys.stderr)
            rc = grid.pop("grid_spike_rc")
            if grid["n_grid_holes"]:
                print(f"{name}: {grid['n_grid_holes']} interior holes in the grid "
                      f"({grid['grid_hole_points']} points)", file=sys.stderr)
            if grid["n_grid_spikes"]:
                print(f"{name}: {grid['n_grid_spikes']} grid points are more than {GRID_SPIKE_FACTOR:g}x the median step from a "
                      f"neighbour (max {grid['grid_step_max']} voxels, median {grid['grid_step_median']})",
                      file=sys.stderr)
                with open(os.path.join(a.out, f"{name}.grid_spikes.csv"), "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["row", "col"])
                    w.writerows(rc.tolist())
        else:
            P, N, patch, n_vert = sample_obj(src, a.samples, a.blobs, rng)
        prof, t, inside = read_profiles(vol, P, N, a.half, a.step, a.sigma, workers=a.workers)
        if inside.sum() < 0.9 * len(inside):
            # a surface mostly outside the volume is in the wrong coordinate frame
            # (missing downscale or physical scale, villa PRs #2112 and #2137), not misplaced
            print(f"{name}: {len(inside) - inside.sum()} of {len(inside)} points fall outside the volume; "
                  f"check the surface's scale and downscale metadata before reading the offsets",
                  file=sys.stderr)
        off, contrast, valid = nearest_peak(prof, t, a.min_contrast, a.min_prominence)
        valid &= inside
        m = summarize(off, contrast, valid, inside, patch, prof, t, a.half, a.min_contrast)
        if inside.sum() and valid.sum() < 0.05 * inside.sum():
            # inside the volume bounds but no sheet anywhere along the profiles: the surface is
            # in the wrong place (a missing downscale factor leaves it inside the bounds but in
            # empty space, villa PR #2112) or paired with the wrong volume or level
            print(f"{name}: a sheet peak was found at only {int(valid.sum())} of {int(inside.sum())} points "
                  f"inside the volume; the surface is probably in the wrong coordinate frame "
                  f"or paired with the wrong volume", file=sys.stderr)
        m.update(controls(vol, P, N, off, valid, a, rng))
        m.update({"surface": name, "source": src, "volume": a.volume, "n_vertices": int(n_vert),
                  "n_no_normal": n_no_normal, **grid, "half": a.half, "step": a.step, "sigma": a.sigma,
                  "min_contrast": a.min_contrast, "min_prominence": a.min_prominence})
        with open(os.path.join(a.out, f"{name}.json"), "w") as f:
            json.dump(m, f, indent=1)
        with open(os.path.join(a.out, f"{name}.samples.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["z", "y", "x", "nz", "ny", "nx", "patch", "offset", "contrast", "valid"])
            for i in range(len(P)):
                w.writerow([f"{P[i,0]:.2f}", f"{P[i,1]:.2f}", f"{P[i,2]:.2f}",
                            f"{N[i,0]:.4f}", f"{N[i,1]:.4f}", f"{N[i,2]:.4f}", int(patch[i]),
                            "" if not valid[i] else f"{off[i]:.2f}",
                            "" if not np.isfinite(contrast[i]) else f"{contrast[i]:.1f}", int(valid[i])])
        if a.worst_cells:
            cells = worst_cells(P, N, off, valid, patch, limit=a.worst_cells)
            with open(os.path.join(a.out, f"{name}.worst_cells.csv"), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["rank", "z", "y", "x", "nz", "ny", "nx", "n", "median_offset"])
                for i, c in enumerate(cells, 1):
                    w.writerow([i, *(f"{v:.1f}" for v in c["center_zyx"]), *(f"{v:.3f}" for v in c["normal_zyx"]),
                                c["n"], f"{c['median_offset']:+.2f}"])
            write_point_collection(os.path.join(a.out, f"{name}.worst_cells.json"),
                                   f"placement_audit {name}", cells)
            m["n_worst_cells"] = len(cells)
        if a.plot:
            plot(P, off, valid, os.path.join(a.out, f"{name}.png"), name)
        rows.append(m)
        print(f"{name}: valid {m['n_valid']}/{m['n_samples']} offset median {m['offset_median']:+.2f} "
              f"IQR [{m['offset_q25']:+.2f}, {m['offset_q75']:+.2f}] |off|>2 {m['offset_abs_gt_2_frac']:.2f} "
              f"off-sheet {m['off_sheet_frac']:.2f} no-peak {m['no_peak_frac']:.2f} "
              f"snap-sd {m['snap_remeasure_sd']:.2f} tangent|diff| 2/4/8 {m['tangent_absdiff_median_2']:.1f}/"
              f"{m['tangent_absdiff_median_4']:.1f}/{m['tangent_absdiff_median_8']:.1f}", flush=True)
        for k, lim in limits.items():
            if not (m[k] <= lim):  # NaN (nothing measured) fails too
                print(f"FAIL {name}: {k}={m[k]} > {lim}", flush=True)
                failed = True
    keys = [k for k in rows[0] if k not in ("source", "volume")]
    with open(os.path.join(a.out, "aggregate.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return 2 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
