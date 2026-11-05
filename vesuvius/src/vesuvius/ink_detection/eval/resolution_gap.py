"""Compare 9 um ink inputs with a real coarse scan of the same segment.

For a segment that has a fine surface volume (2.2 to 2.45 um) with ink labels
and a real coarse surface volume (7 to 9.45 um) of the same papyrus, this
builds the coarse inputs several ways on the same windows and tiles:

- ``fine``: the fine surface volume itself
- ``box_average``: its 4 x 4 x 4 box average (about 9.6 um), done inside the
  network's input as a plain resampling
- ``box_blur`` and ``box_blur_x1.25`` to ``box_blur_x2.5``: the box average
  with an extra Gaussian blur fitted, from the CT alone, so its in-plane
  sharpness matches the real scan's, and at multiples of that blur
- ``box_blur_matched``: one blur fitted together with a gray-level map that
  matches the real scan's gray levels per depth plane
- ``box_blur_directional``: separate blurs across and along the sheet, fitted
  so both the in-plane and the depth sharpness match the real scan's, with the
  gray-level map
- ``real``: the real coarse scan on the fine grid at the whole-voxel
  registration (nearest neighbor)
- ``real_registered``: the real coarse scan with sub-voxel registration
- ``villa_input``: ``prepare_9um_isotropic_input``'s recipe (pyramid level 2,
  rounded means of 4 of the 84 centered planes), the 15 planes the box
  average's planes overlap most

It then trains the same small model (a 3D stem into a 2D U-Net, from scratch)
on each input, one model per seed and held-out window, and scores ink AUC
inside the supervision mask with a reversed-depth control, a null that puts
another segment's labels on the same CT, seed sign agreement, and intervals
over seeds and tiles.

Steps, in order (each writes JSON beside its outputs)::

    resolution_gap windows --labels INK.zarr --mask SUP.zarr --out DIR
    resolution_gap stages  --fine FINE.zarr --coarse COARSE.zarr --windows DIR/windows.json
                           --tiles DIR/tiles.json --out DIR
    resolution_gap train   --input NAME --seed 1 2 --fine FINE.zarr --coarse COARSE.zarr
                           --labels INK.zarr --mask SUP.zarr --windows DIR/windows.json
                           --tiles DIR/tiles.json --stages DIR/stages.json --out DIR/maps
    resolution_gap score   DIR/maps/*.npz --stages DIR/stages.json --donor OTHER.npz --out score.json

``windows`` reads the labels at level 3; ``stages`` reads the CT only (no
labels), so the blur fits and the registration cannot see the ink; ``train``
needs PyTorch and, in practice, a GPU. Every probability map is model output.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import glob
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Sequence

import numpy as np

from vesuvius.ink_detection.preprocessing.prepare_9um_isotropic_input import (
    INPUT_Z as VILLA_PLANES_IN,
    OUTPUT_Z as VILLA_PLANES,
    POOL_Z,
    centered_slice,
    open_source_array,
)
from vesuvius.ink_detection.volume_io import open_volume_root

# ---------------------------------------------------------------- constants

SUPERBLOCK = 12  # fine px per scoring unit side
DEPTH = 60  # fine layers per area (the network's input depth)
STEPS = 3000
BATCH = 12
PATCH = 120
LR = 1e-3
WD = 1e-4
WARM = 100
MIN_PATCH_SUP = 0.25
REDRAWS = 10
FOLD_GUARD = 252  # fine px around a held-out window that no training patch may touch
CORE = 252  # mapping tile core, fine px
MARGIN = 48
STEM = 32
BINS = 5  # depth bins of 12 layers
WINDOW = 1008  # fine px per window side
WINDOW_STEP3 = 126  # level-3 px per window side
MAX_WINDOWS = 4
TILE_STEP3 = 63  # level-3 px per training tile side (504 fine px)
TILE_MIN_SUP = 0.1
SMOKE_WINDOW = 504
SMOKE_STEPS = 3
SMOKE_BATCH = 2
MIN_WINDOWS = 2
SEEDS = (1, 2, 3, 4, 5, 6)
FINE_UM = (2.2, 2.45)
COARSE_UM = (7.0, 9.45)

REAL_SEARCH = 20  # real px searched each way for the whole-voxel shift
REAL_PAD = 24  # real px read around an area's footprint
REG_MIN = 0.80
REG_TOL = 2  # real px from the median window shift
REG_DZ = (-2, -1, 0, 1, 2)
SUB_BLOCK = 126  # fine px per sub-voxel registration block side
SUB_STEP = 0.25  # real px (and real layers) per search step
SUB_RANGE = 1.5  # real px searched each way around the whole-voxel shift
SUB_BLUR = 4  # fine px box blur of the fine depth mean
SUB_DZ_RANGE = 1.0  # real layers searched each way around the whole-voxel dz
SUB_AREA_DZ_MIN = 0.90  # an area keeps its own dz at this profile r
SUB_PRED_TOL = 0.5  # real px: a held-out block's offset predicted from its neighbors within this
SUB_PRED_SHARE = 0.75  # share of held-out blocks that must meet SUB_PRED_TOL in every window

SIGMA_MAX = 6.0  # coarse voxels: the largest in-plane blur searched
SZ_MAX = 3.0  # coarse voxels: the largest depth blur searched
SM_TOL = 0.15  # sharpness on the windows within 15 % of the real scan's
CM_SHARE = 0.5  # gray levels: W1 after the map at most this share of before ...
CM_ABS = 1.0  # ... or at most this many CT units
ANISO_TOL = 0.05
NORM_TOL = 1e-6
IMMERKAER = (math.pi / 2) ** 0.5 / 6.0
DEPTH_K = (math.pi / 2) ** 0.5 / 6**0.5
POOL_TOL = 1.5  # CT units: three roundings (two pyramid levels and the plane mean)
XY = 4  # one level-2 px is 4 x 4 fine px

BOOT_TILE = 7  # scoring units per bootstrap tile side
DRAWS = 1000
BOOT_SEED = 20261028
FINE_FLOOR = 0.65  # seed-mean AUC the fine input must reach
REVERSED_MIN = 0.55
FLAT_BAND = 0.02
NULL_DOSE_SHARE = 1.0 / 3

# Every input with its code. The code seeds the model and names its map file,
# so it is fixed per input.
INPUTS = (
    ("fine", 1),
    ("box_average", 4),
    ("box_blur", 6),
    ("box_blur_x1.25", 16),
    ("box_blur_x1.5", 12),
    ("box_blur_x1.75", 17),
    ("box_blur_x2", 13),
    ("box_blur_x2.5", 18),
    ("box_blur_matched", 19),
    ("box_blur_directional", 20),
    ("real", 9),
    ("real_registered", 7),
    ("villa_input", 21),
)
INPUT_CODE = dict(INPUTS)
DOSE = (
    ("box_average", 0.0),
    ("box_blur", 1.0),
    ("box_blur_x1.25", 1.25),
    ("box_blur_x1.5", 1.5),
    ("box_blur_x1.75", 1.75),
    ("box_blur_x2", 2.0),
    ("box_blur_x2.5", 2.5),
)
DOSE_MULT = {name: m for name, m in DOSE if m > 0}
GROUPS = DEPTH // POOL_Z  # 15 four-layer groups in the 60 fine layers
MAP_RE = re.compile(r"(.+)-(.+)-s(\d+)(-smoke)?\.npz$")


def sha_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def sha_file(path: str | Path) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


# ---------------------------------------------------------------- reading


def read_zyx(array, z0, z1, y0, y1, x0, x1) -> np.ndarray:
    """A uint8 (z, y, x) box of a 3D array, zero outside the array."""
    out = np.zeros((z1 - z0, y1 - y0, x1 - x0), np.uint8)
    lim = array.shape
    a = [max(v0, 0) for v0 in (z0, y0, x0)]
    b = [min(v1, n) for v1, n in zip((z1, y1, x1), lim)]
    if all(lo < hi for lo, hi in zip(a, b)):
        out[a[0] - z0 : b[0] - z0, a[1] - y0 : b[1] - y0, a[2] - x0 : b[2] - x0] = np.asarray(
            array[a[0] : b[0], a[1] : b[1], a[2] : b[2]]
        )
    return out


def read_yx(array, y0, y1, x0, x1) -> np.ndarray:
    """A uint8 (y, x) box of a 2D array, zero outside the array."""
    return read_zyx(_Lift(array), 0, 1, y0, y1, x0, x1)[0]


class _Lift:
    """A 2D array seen as (1, y, x)."""

    def __init__(self, array):
        self.array = array
        self.shape = (1,) + tuple(array.shape)

    def __getitem__(self, key):
        return np.asarray(self.array[key[1], key[2]])[None]


def voxel_size_um(attrs) -> float | None:
    """The level-0 voxel size of an OME-Zarr surface volume, from its multiscales."""
    try:
        return float(attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]["scale"][0])
    except (KeyError, IndexError, TypeError):
        return None


class Data:
    """The arrays one segment's run reads: fine level 0 and 2, coarse level 0, labels and mask level 0."""

    def __init__(self, fine, coarse=None, labels=None, mask=None, um_fine=None, um_coarse=None, canvas_scale=None):
        self.fine_path, self.coarse_path = str(fine), coarse and str(coarse)
        self.fine0 = open_source_array(self.fine_path, "0")
        self._fine2 = None
        self.coarse0 = open_source_array(self.coarse_path, "0") if coarse else None
        self.labels0 = open_source_array(str(labels), "0") if labels else None
        self.mask0 = open_source_array(str(mask), "0") if mask else None
        fa = dict(open_volume_root(self.fine_path).attrs)
        ca = dict(open_volume_root(self.coarse_path).attrs) if coarse else {}
        self.um_fine = float(um_fine) if um_fine else voxel_size_um(fa)
        self.um_coarse = float(um_coarse) if um_coarse else voxel_size_um(ca)
        if canvas_scale is not None:
            self.scale = (float(canvas_scale[0]), float(canvas_scale[1]))
        elif coarse:
            fine_c, real_c = fa["canvas_size"], ca["canvas_size"]
            self.scale = (real_c[1] / fine_c[1], real_c[0] / fine_c[0])  # canvas_size is (columns, rows)
        else:
            self.scale = None

    @property
    def fine2(self):
        if self._fine2 is None:
            self._fine2 = open_source_array(self.fine_path, "2")
        return self._fine2

    def site(self) -> dict:
        if self.coarse0 is None:
            raise ValueError("the site needs the coarse surface volume")
        for what, um, (lo, hi) in (("fine", self.um_fine, FINE_UM), ("coarse", self.um_coarse, COARSE_UM)):
            if um is None:
                raise ValueError(f"no voxel size for the {what} volume: pass --um-{what}")
            if not lo <= um <= hi:
                raise ValueError(f"{what} voxel size {um} um is outside {lo} to {hi} um")
        geo = site_geometry(self.fine0.shape[0], self.coarse0.shape[0], self.um_fine, self.um_coarse)
        geo["scale_rows_cols"] = list(self.scale)
        return geo

    def read_ct(self, site, y0, x0, n):
        layers = site["fine_layers"]
        return read_zyx(self.fine0, layers[0], layers[1], y0, y0 + n, x0, x0 + n)

    def read_box(self, site, y0, x0, n):
        vol = self.read_ct(site, y0, x0, n)
        lab = read_yx(self.labels0, y0, y0 + n, x0, x0 + n)
        sup = read_yx(self.mask0, y0, y0 + n, x0, x0 + n)
        return vol, lab, sup

    def read_real(self, y0, x0, n):
        box = real_box(y0, x0, n, self.scale, REAL_PAD)
        a = self.coarse0
        return read_zyx(a, 0, a.shape[0], box[0], box[1], box[2], box[3]), box

    def read_level2(self, y0, x0, n):
        if y0 % XY or x0 % XY or n % XY:
            raise ValueError(f"area ({y0}, {x0}, {n}) is off the level 2 grid")
        a = self.fine2
        return read_zyx(a, 0, a.shape[0], y0 // XY, (y0 + n) // XY, x0 // XY, (x0 + n) // XY)


# ---------------------------------------------------------------- site geometry


def site_geometry(n_fine, n_real, um_fine, um_real, depth=DEPTH, dzs=REG_DZ) -> dict:
    """The fine layers read (depth layers centered on the fine volume), the two stack centers ((n - 1) / 2 each) and
    the site check: the fine volume has at least depth layers and every fine layer maps inside the real volume at
    every dz searched."""
    why = []
    if n_fine < depth:
        why.append("fine volume has %d layers, fewer than %d" % (n_fine, depth))
    start = max(0, (n_fine - depth) // 2)
    geo = {
        "um_fine": float(um_fine),
        "um_real": float(um_real),
        "fine_layers": [start, start + depth],
        "fine_c": (n_fine - 1) / 2.0,
        "real_c": (n_real - 1) / 2.0,
        "n_fine": int(n_fine),
        "n_real": int(n_real),
    }
    if not why:
        L = np.arange(start, start + depth)
        z = geo["real_c"] + (L - geo["fine_c"]) * um_fine / um_real
        lo, hi = int(np.rint(z.min())) + min(dzs) - 1, int(np.rint(z.max())) + max(dzs) + 1
        if lo < 0 or hi > n_real - 1:
            why.append(
                "fine layers map to real layers %d..%d (with the dz search), outside 0..%d" % (lo, hi, n_real - 1)
            )
    geo["check"] = {"verdict": "FAIL" if why else "PASS", "why": why}
    return geo


def real_layer_map(site, dz=0):
    """Real layer index of each fine layer read: round(real_c + (L - fine_c) x um_fine / um_real) + dz."""
    L = np.arange(site["fine_layers"][0], site["fine_layers"][1])
    return np.rint(site["real_c"] + (L - site["fine_c"]) * site["um_fine"] / site["um_real"]).astype(int) + dz


def real_z(site, dz):
    """Continuous real layer of each fine layer read (real_layer_map before rounding)."""
    L = np.arange(site["fine_layers"][0], site["fine_layers"][1])
    return site["real_c"] + (L - site["fine_c"]) * site["um_fine"] / site["um_real"] + dz


# ---------------------------------------------------------------- whole-voxel registration (CT only)


def box_filter2d(a, w):
    """Mean over w x w in-plane neighborhoods of a (D, H, W) array, edges reflected."""
    if w <= 1:
        return np.asarray(a, np.float32)
    r = w // 2
    p = np.pad(np.asarray(a, np.float64), ((0, 0), (r, r), (r, r)), mode="symmetric")
    c = np.cumsum(np.cumsum(p, axis=1), axis=2)
    c = np.pad(c, ((0, 0), (1, 0), (1, 0)))
    H, W = a.shape[1], a.shape[2]
    s = c[:, w : w + H, w : w + W] - c[:, :H, w : w + W] - c[:, w : w + H, :W] + c[:, :H, :W]
    return (s / (w * w)).astype(np.float32)


def real_box(y0, x0, n, scale, pad=REAL_PAD):
    """The real-grid box (y0, y1, x0, x1) that covers a fine n x n box at canvas (y0, x0), plus pad."""
    return (
        int(math.floor(y0 * scale[0])) - pad,
        int(math.ceil((y0 + n) * scale[0])) + pad,
        int(math.floor(x0 * scale[1])) - pad,
        int(math.ceil((x0 + n) * scale[1])) + pad,
    )


def to_real_grid(img, origin, box, scale):
    """Area-average a fine 2D image at canvas origin onto the real grid box; returns (mean, count) on the box."""
    H, W = img.shape
    ry = np.floor((origin[0] + np.arange(H) + 0.5) * scale[0]).astype(int) - box[0]
    rx = np.floor((origin[1] + np.arange(W) + 0.5) * scale[1]).astype(int) - box[2]
    h, w = box[1] - box[0], box[3] - box[2]
    ok_y, ok_x = (ry >= 0) & (ry < h), (rx >= 0) & (rx < w)
    idx = (ry[ok_y][:, None] * w + rx[ok_x][None, :]).ravel()
    vals = np.asarray(img, np.float64)[np.ix_(ok_y, ok_x)].ravel()
    tot = np.bincount(idx, vals, minlength=h * w).reshape(h, w)
    cnt = np.bincount(idx, minlength=h * w).reshape(h, w)
    with np.errstate(invalid="ignore", divide="ignore"):
        return tot / cnt, cnt


def best_shift(a, b, valid, search=REAL_SEARCH):
    """The integer shift (dy, dx) of b against a that maximizes Pearson r over valid pixels of a: (r, dy, dx)."""
    H, W = a.shape
    ys, xs = np.nonzero(valid)
    best = (-2.0, 0, 0)
    for dy in range(-search, search + 1):
        for dx in range(-search, search + 1):
            yy, xx = ys + dy, xs + dx
            ok = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
            if ok.sum() < 100:
                continue
            u, v = a[ys[ok], xs[ok]], b[yy[ok], xx[ok]]
            r = float(np.corrcoef(u, v)[0, 1])
            if r > best[0]:
                best = (r, dy, dx)
    return best


def register_real(site, fine_vol, origin, real_vol, box, scale, search=REAL_SEARCH):
    """In-plane whole-voxel registration of a real box to a fine area from the CT alone: {"r", "shift": [dy, dx]};
    real pixel = floor((fine + 0.5) * scale) - box0 + shift."""
    a, cnt = to_real_grid(np.asarray(fine_vol, np.float32).mean(axis=0), origin, box, scale)
    m = real_layer_map(site, 0)
    b = np.asarray(real_vol[m.min() : m.max() + 1], np.float32).mean(axis=0)
    valid = (cnt > 0) & (np.nan_to_num(a) > 0)
    k = search + 1
    valid[:k] = valid[-k:] = False
    valid[:, :k] = valid[:, -k:] = False
    if valid.sum() < 100:
        return {"r": None, "shift": None}
    r, dy, dx = best_shift(np.nan_to_num(a), b, valid, search)
    return {"r": round(float(r), 4), "shift": [int(dy), int(dx)]}


def real_index(origin, n, box, scale, shift):
    """Row and column indices into a real box for the fine pixels of an n x n area at canvas origin."""
    ry = np.floor((origin[0] + np.arange(n) + 0.5) * scale[0]).astype(int) - box[0] + shift[0]
    rx = np.floor((origin[1] + np.arange(n) + 0.5) * scale[1]).astype(int) - box[2] + shift[1]
    return ry, rx


def real_on_fine(site, real_vol, origin, n, box, scale, shift, dz):
    """The ``real`` input: the real box sampled (nearest neighbor) onto the fine grid, (60, n, n) uint8."""
    ry, rx = real_index(origin, n, box, scale, shift)
    rz = real_layer_map(site, dz)
    for name, idx, lim in (
        ("layer", rz, real_vol.shape[0]),
        ("row", ry, real_vol.shape[1]),
        ("column", rx, real_vol.shape[2]),
    ):
        if idx.min() < 0 or idx.max() >= lim:
            raise ValueError("refused: real %s index %d..%d outside 0..%d" % (name, idx.min(), idx.max(), lim - 1))
    return np.ascontiguousarray(np.asarray(real_vol)[np.ix_(rz, ry, rx)])


def pick_tile_shift(reg, median):
    """A tile keeps its own shift when r >= 0.80 and it lies within 2 real px of the median on each axis."""
    if (
        reg.get("r") is not None
        and reg["r"] >= REG_MIN
        and all(abs(a - b) <= REG_TOL for a, b in zip(reg["shift"], median))
    ):
        return list(reg["shift"]), "own"
    return list(median), "median"


def depth_profiles(fine_vol, real_vol, origin, box, scale, shift):
    """Per-layer means: fine over rendered fine pixels; real over the registered footprint where it is nonzero."""
    fv = np.asarray(fine_vol, np.float32)
    rend = fv.mean(axis=0) > 0
    fp = np.array([float(fv[k][rend].mean()) for k in range(fv.shape[0])])
    ry, rx = real_index(origin, fv.shape[1], box, scale, shift)
    ys = np.unique(ry[(ry >= 0) & (ry < real_vol.shape[1])])
    xs = np.unique(rx[(rx >= 0) & (rx < real_vol.shape[2])])
    sub = np.asarray(real_vol, np.float32)[:, ys][:, :, xs]
    ok = sub.mean(axis=0) > 0
    rp = np.array([float(sub[k][ok].mean()) if ok.any() else 0.0 for k in range(sub.shape[0])])
    return fp, rp


def pick_dz(site, fine_profiles, real_profiles, dzs=REG_DZ):
    """The whole-layer depth offset dz that maximizes Pearson r between the fine profiles averaged into real layers
    and the real profiles, pooled over windows: (dz, r, {dz: r})."""
    m = real_layer_map(site, 0)
    ks = np.unique(m)
    table = {}
    for dz in dzs:
        f_all, r_all = [], []
        for fp, rp in zip(fine_profiles, real_profiles):
            fp = np.asarray(fp)
            idx = ks + dz
            if idx.min() < 0 or idx.max() >= len(rp):
                break
            f_all += [float(fp[m == k].mean()) for k in ks]
            r_all += [float(rp[i]) for i in idx]
        else:
            table[dz] = float(np.corrcoef(f_all, r_all)[0, 1])
    best = max(table, key=lambda d: (table[d], -abs(d)))
    return best, table[best], table


def registration_check(reg):
    """Every window's r >= 0.80, its shift within 2 real px of the median, and the depth r >= 0.80."""
    if not reg:
        return {"verdict": "FAIL", "why": "no registration record"}
    med = reg.get("median_shift")
    bad = []
    for w in reg.get("windows", []):
        if w.get("r") is None or w["r"] < REG_MIN:
            bad.append("window %s r %s" % (w.get("fold"), w.get("r")))
        elif any(abs(a - b) > REG_TOL for a, b in zip(w["shift"], med)):
            bad.append("window %s shift %s far from median %s" % (w.get("fold"), w["shift"], med))
    if reg.get("dz_r") is None or reg["dz_r"] < REG_MIN:
        bad.append("depth r %s" % reg.get("dz_r"))
    return {
        "verdict": "FAIL" if bad else "PASS",
        "why": bad,
        "median_shift": med,
        "dz": reg.get("dz"),
        "dz_r": reg.get("dz_r"),
        "window_r": [w.get("r") for w in reg.get("windows", [])],
    }


# ---------------------------------------------------------------- sub-voxel registration (CT only)


def sub_offsets(rng=SUB_RANGE, step=SUB_STEP):
    k = int(round(rng / step))
    return np.arange(-k, k + 1) * step


def real_coords(o, n, b0, s):
    """Continuous real index of fine pixels o..o+n-1 (real voxel k has its center at k)."""
    return (o + np.arange(n) + 0.5) * s - b0 - 0.5


def interp_matrix(coords, size):
    """(len(coords), size) linear interpolation weights; refuses a coordinate outside 0..size-1."""
    c = np.asarray(coords, np.float64)
    if c.min() < 0 or c.max() > size - 1:
        raise ValueError("refused: interpolation coordinate %.2f..%.2f outside 0..%d" % (c.min(), c.max(), size - 1))
    i0 = np.minimum(np.floor(c).astype(int), size - 2)
    w1 = c - i0
    W = np.zeros((len(c), size))
    W[np.arange(len(c)), i0] = 1 - w1
    W[np.arange(len(c)), i0 + 1] = w1
    return W


def _corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / d) if d > 0 else -2.0


def sub_register(
    site, fine_vol, origin, real_vol, box, scale, shift, block=SUB_BLOCK, blur=SUB_BLUR, rng=SUB_RANGE, step=SUB_STEP
):
    """Per block of fine px, the sub-voxel in-plane offset (around the whole-voxel shift) that maximizes Pearson r
    between the real depth mean sampled bilinearly on the fine grid and the box-blurred fine depth mean, over fine px
    whose blur window is fully rendered. Returns {"nb", "offsets", "blocks"}."""
    fv = np.asarray(fine_vol, np.float32)
    F = fv.mean(axis=0)
    rend = (F > 0).astype(np.float32)
    Fb = box_filter2d(F[None], blur)[0]
    full = box_filter2d(rend[None], blur)[0] > 0.999
    m = real_layer_map(site, 0)
    Rm = np.asarray(real_vol[m.min() : m.max() + 1], np.float64).mean(axis=0)
    n_r, n_c = F.shape
    cy = real_coords(origin[0], n_r, box[0], scale[0]) + shift[0]
    cx = real_coords(origin[1], n_c, box[2], scale[1]) + shift[1]
    offs = sub_offsets(rng, step)
    nby, nbx = n_r // block, n_c // block
    out = {"nb": [nby, nbx], "offsets": [float(o) for o in offs], "blocks": []}
    i0 = int(np.argmin(np.abs(offs)))
    for bi in range(nby):
        ys = slice(bi * block, (bi + 1) * block)
        Wy = [interp_matrix(cy[ys] + o, Rm.shape[0]) for o in offs]
        for bj in range(nbx):
            xs = slice(bj * block, (bj + 1) * block)
            v = full[ys, xs]
            rec = {"bi": bi, "bj": bj, "n": int(v.sum()), "r0": None, "r": None, "off": None, "table": None}
            if v.sum() < block * block // 2:
                out["blocks"].append(rec)
                continue
            fb = Fb[ys, xs][v].astype(np.float64)
            Wx = [interp_matrix(cx[xs] + o, Rm.shape[1]) for o in offs]
            tab = np.full((len(offs), len(offs)), -2.0)
            for i, wy in enumerate(Wy):
                A = wy @ Rm
                for j, wx in enumerate(Wx):
                    tab[i, j] = _corr((A @ wx.T)[v], fb)
            best = max(
                ((tab[i, j], -abs(offs[i]) - abs(offs[j]), i, j) for i in range(len(offs)) for j in range(len(offs)))
            )
            rec.update(
                {
                    "r0": round(float(tab[i0, i0]), 4),
                    "r": round(float(best[0]), 4),
                    "off": [float(offs[best[2]]), float(offs[best[3]])],
                    "table": [[round(float(t), 4) for t in row] for row in tab],
                }
            )
            out["blocks"].append(rec)
    return out


def _grid(reg, key="off"):
    nby, nbx = reg["nb"]
    G = np.full((nby, nbx, 2), np.nan)
    R = np.full((nby, nbx), np.nan)
    for b in reg["blocks"]:
        if b["r"] is not None:
            R[b["bi"], b["bj"]] = b["r"]
            if b[key] is not None:
                G[b["bi"], b["bj"]] = b[key]
    return G, R


def sub_field(reg, min_r=REG_MIN):
    """Block offsets with r >= min_r, a 3 x 3 median filter over accepted neighbors, the area's median for blocks
    left empty (0 when none is accepted): (nby, nbx, 2) offsets at block centers."""
    G, R = _grid(reg)
    acc = R >= min_r
    G = np.where(acc[..., None], G, np.nan)
    nby, nbx = R.shape
    out = np.full_like(G, np.nan)
    for i in range(nby):
        for j in range(nbx):
            nb = G[max(0, i - 1) : i + 2, max(0, j - 1) : j + 2].reshape(-1, 2)
            nb = nb[~np.isnan(nb[:, 0])]
            if len(nb):
                out[i, j] = np.median(nb, axis=0)
    if acc.any():
        med = np.nanmedian(G.reshape(-1, 2)[acc.ravel()], axis=0)
    else:
        med = np.zeros(2)
    empty = np.isnan(out[..., 0])
    out[empty] = med
    return out


def field_on_pixels(field, n_r, n_c, block=SUB_BLOCK):
    """Bilinear interpolation of block-center offsets to every fine px (edges clamped): two (n_r, n_c) arrays."""
    nby, nbx = field.shape[:2]

    def w(n, nb):
        t = np.clip((np.arange(n) + 0.5) / block - 0.5, 0, nb - 1)
        if nb == 1:
            return np.ones((n, 1))
        return interp_matrix(t, nb)

    Wy, Wx = w(n_r, nby), w(n_c, nbx)
    return Wy @ field[..., 0] @ Wx.T, Wy @ field[..., 1] @ Wx.T


def sub_reg_check(reg, field, min_r=REG_MIN, tol=SUB_PRED_TOL):
    """One window's held-out registration check: (a) the median r at the field's offsets is at least min_r; (b) each
    white checkerboard block with r >= min_r and at least two accepted black 4-neighbors has its offset predicted by
    their mean, within tol real px for at least SUB_PRED_SHARE of such blocks."""
    offs = np.asarray(reg["offsets"])
    G, R = _grid(reg)
    tabs = {(b["bi"], b["bj"]): b["table"] for b in reg["blocks"] if b["table"] is not None}

    def r_at(i, j, off):
        t = tabs[(i, j)]
        a = int(np.argmin(np.abs(offs - off[0])))
        b = int(np.argmin(np.abs(offs - off[1])))
        return t[a][b]

    fr = [r_at(i, j, field[i, j]) for (i, j) in tabs]
    errs, gains = [], []
    nby, nbx = R.shape
    for i, j in tabs:
        if (i + j) % 2 == 0 or not R[i, j] >= min_r:
            continue
        nbs = [
            (i + di, j + dj)
            for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1))
            if 0 <= i + di < nby and 0 <= j + dj < nbx and R[i + di, j + dj] >= min_r
        ]
        if len(nbs) < 2:
            continue
        pred = np.mean([G[a, b] for a, b in nbs], axis=0)
        errs.append(float(np.max(np.abs(pred - G[i, j]))))
        i0 = int(np.argmin(np.abs(offs)))
        gains.append(float(r_at(i, j, pred) - tabs[(i, j)][i0][i0]))
    share = float(np.mean([e <= tol for e in errs])) if errs else 0.0
    frm = float(np.median(fr)) if fr else None
    return {
        "field_r_median": frm,
        "pred_n": len(errs),
        "pred_ok_share": round(share, 4),
        "pred_err_median": float(np.median(errs)) if errs else None,
        "gain_median": float(np.median(gains)) if gains else None,
        "pass": bool(frm is not None and frm >= min_r and errs and share >= SUB_PRED_SHARE),
    }


def sub_dz(site, fine_profiles, real_profiles, center, rng=SUB_DZ_RANGE, step=SUB_STEP):
    """The sub-voxel dz around center that maximizes Pearson r between the fine profiles box-smoothed over 4 layers
    and the real profiles interpolated at real_z(dz), pooled over the given areas: (dz, r, {dz: r})."""
    table = {}
    for dz in center + sub_offsets(rng, step):
        f_all, r_all = [], []
        for fp, rp in zip(fine_profiles, real_profiles):
            fp = np.asarray(fp, np.float64)
            k = np.ones(SUB_BLUR)
            fs = np.convolve(fp, k, "same") / np.convolve(np.ones_like(fp), k, "same")
            z = real_z(site, dz)
            if z.min() < 0 or z.max() > len(rp) - 1:
                break
            f_all += list(fs)
            r_all += list(np.interp(z, np.arange(len(rp)), np.asarray(rp, np.float64)))
        else:
            table[round(float(dz), 4)] = _corr(np.array(f_all), np.array(r_all))
    best = max(table, key=lambda d: (table[d], -abs(d - center)))
    return best, table[best], table


def real_on_fine_sub(site, real_vol, origin, n, box, scale, shift, fy, fx, dz):
    """The ``real_registered`` input: the real box sampled (nearest neighbor) at the whole-voxel shift plus the
    sub-voxel field (fy, fx) and the sub-voxel dz, (60, n, n) uint8."""
    cy = real_coords(origin[0], n, box[0], scale[0]) + shift[0]
    cx = real_coords(origin[1], n, box[2], scale[1]) + shift[1]
    iy = np.floor(cy[:, None] + fy + 0.5).astype(int)
    ix = np.floor(cx[None, :] + fx + 0.5).astype(int)
    rz = np.floor(real_z(site, dz) + 0.5).astype(int)
    for name, idx, lim in (
        ("layer", rz, real_vol.shape[0]),
        ("row", iy, real_vol.shape[1]),
        ("column", ix, real_vol.shape[2]),
    ):
        if idx.min() < 0 or idx.max() >= lim:
            raise ValueError("refused: real %s index %d..%d outside 0..%d" % (name, idx.min(), idx.max(), lim - 1))
    rv = np.asarray(real_vol)[rz]
    return np.ascontiguousarray(rv[:, iy, ix])


def subvoxel_check(rec):
    """Every window passes sub_reg_check and the segment's sub-voxel depth r is at least 0.80."""
    if not rec:
        return {"verdict": "FAIL", "why": "no registration record"}
    bad = ["window %s" % w["fold"] for w in rec.get("windows", []) if not w["check"]["pass"]]
    if not rec.get("windows"):
        bad.append("no windows")
    if rec.get("dz_sub_r") is None or rec["dz_sub_r"] < REG_MIN:
        bad.append("depth r %s" % rec.get("dz_sub_r"))
    return {
        "verdict": "FAIL" if bad else "PASS",
        "why": bad,
        "dz_sub": rec.get("dz_sub"),
        "dz_sub_r": rec.get("dz_sub_r"),
        "windows": [w["check"] for w in rec.get("windows", [])],
    }


def real_inputs(site, data, tiles, wins, evals, fine_t, tn, n, log, workers=4):
    """Both registrations, in order: whole-voxel shift and dz from the windows, then the sub-voxel field and dz per
    area. Returns the ``real_registered`` tiles and windows, the record, the whole-voxel tile and window shifts, dz,
    and the real reads of tiles and windows. CT only."""
    scale = data.scale
    with ThreadPoolExecutor(workers) as ex:
        rt = list(ex.map(lambda t: data.read_real(t["y0"], t["x0"], tn), tiles))
    rw = [data.read_real(wins[e["fold"]]["y0"], wins[e["fold"]]["x0"], n) for e in evals]
    whole_w = []
    for e, (rv, box) in zip(evals, rw):
        w = wins[e["fold"]]
        whole_w.append(register_real(site, e["vol"], (w["y0"], w["x0"]), rv, box, scale))
    good = [g["shift"] for g in whole_w if g["shift"] is not None]
    med = [int(np.median([s[0] for s in good])), int(np.median([s[1] for s in good]))]
    fps, rps = [], []
    for e, (rv, box), g in zip(evals, rw, whole_w):
        w = wins[e["fold"]]
        fp, rp = depth_profiles(e["vol"], rv, (w["y0"], w["x0"]), box, scale, g["shift"] or med)
        fps.append(fp)
        rps.append(rp)
    dz_whole, dz3_r, _ = pick_dz(site, fps, rps)
    tshift = []
    for t, fv, (rv, box) in zip(tiles, fine_t, rt):
        g = register_real(site, fv, (t["y0"], t["x0"]), rv, box, scale)
        tshift.append(pick_tile_shift(g, med)[0])
    wshift = [g["shift"] or med for g in whole_w]
    dz_sub, dz_sub_r, dz_table = sub_dz(site, fps, rps, float(dz_whole))
    reg = {
        "scale_rows_cols": list(scale),
        "whole_voxel": {
            "median_shift": med,
            "dz": int(dz_whole),
            "dz_r": round(float(dz3_r), 4),
            "window_shifts": wshift,
            "window_r": [g["r"] for g in whole_w],
        },
        "dz_sub": dz_sub,
        "dz_sub_r": round(float(dz_sub_r), 4),
        "dz_sub_table": {str(k): round(v, 4) for k, v in dz_table.items()},
        "windows": [],
        "tiles": [],
    }

    def area(fv, origin, rv, box, shift):
        sr = sub_register(site, fv, origin, rv, box, scale, shift)
        field = sub_field(sr)
        fp, rp = depth_profiles(fv, rv, origin, box, scale, shift)
        dz_a, r_a, _ = sub_dz(site, [fp], [rp], dz_sub)
        dz_use = dz_a if r_a >= SUB_AREA_DZ_MIN else dz_sub
        fy, fx = field_on_pixels(field, fv.shape[1], fv.shape[2])
        vol = real_on_fine_sub(site, rv, origin, fv.shape[1], box, scale, shift, fy, fx, dz_use)
        rec = {"shift": list(shift), "dz_own": dz_a, "dz_own_r": round(float(r_a), 4), "dz": dz_use}
        return vol, sr, field, rec

    rs_w = []
    for e, (rv, box), sh in zip(evals, rw, wshift):
        w = wins[e["fold"]]
        vol, sr, field, rec = area(e["vol"], (w["y0"], w["x0"]), rv, box, sh)
        rec.update({"fold": e["fold"], "check": sub_reg_check(sr, field)})
        reg["windows"].append(rec)
        rs_w.append(vol)
    rs_t = []
    for t, fv, (rv, box), sh in zip(tiles, fine_t, rt, tshift):
        vol, sr, field, rec = area(fv, (t["y0"], t["x0"]), rv, box, sh)
        rec.update({"y0": t["y0"], "x0": t["x0"]})
        reg["tiles"].append(rec)
        rs_t.append(vol)
    reg["check"] = subvoxel_check(reg)
    log("sub-voxel registration: dz %d -> %.2f (r %.3f), %s" % (dz_whole, dz_sub, dz_sub_r, reg["check"]["verdict"]))
    return rs_t, rs_w, reg, tshift, wshift, int(dz_whole), rt, rw


def real_areas(site, data, tiles, wins, evals, fine_t, tn, n, log):
    """``real`` (whole-voxel, nearest neighbor) and ``real_registered`` (sub-voxel) tiles and windows, with the
    registration records."""
    scale = data.scale
    rs_t, rs_w, reg, tshift, wshift, dz_whole, rt, rw = real_inputs(site, data, tiles, wins, evals, fine_t, tn, n, log)
    r_t = [
        real_on_fine(site, rv, (t["y0"], t["x0"]), tn, box, scale, sh, dz_whole)
        for t, (rv, box), sh in zip(tiles, rt, tshift)
    ]
    r_w = [
        real_on_fine(site, rv, (wins[e["fold"]]["y0"], wins[e["fold"]]["x0"]), n, box, scale, sh, dz_whole)
        for e, (rv, box), sh in zip(evals, rw, wshift)
    ]
    whole = reg["whole_voxel"]
    reg_whole = {
        "median_shift": whole["median_shift"],
        "dz": whole["dz"],
        "dz_r": whole["dz_r"],
        "windows": [
            {"fold": e["fold"], "r": r_, "shift": sh}
            for e, r_, sh in zip(evals, whole["window_r"], whole["window_shifts"])
        ],
    }
    return {"real": (r_t, r_w), "real_registered": (rs_t, rs_w)}, reg, reg_whole, tshift, wshift, dz_whole, rt, rw


# ---------------------------------------------------------------- sharpness


def immerkaer_sum(img, valid):
    """Immerkaer (1996) noise estimate pieces for one 2D image: (sum of |I * N| over pixels whose 3 x 3 neighborhood
    is valid, count)."""
    a = np.asarray(img, np.float64)
    c = (
        a[:-2, :-2]
        - 2 * a[:-2, 1:-1]
        + a[:-2, 2:]
        - 2 * a[1:-1, :-2]
        + 4 * a[1:-1, 1:-1]
        - 2 * a[1:-1, 2:]
        + a[2:, :-2]
        - 2 * a[2:, 1:-1]
        + a[2:, 2:]
    )
    v = np.asarray(valid, bool)
    ok = np.ones_like(c, bool)
    for dy in range(3):
        for dx in range(3):
            ok &= v[dy : dy + c.shape[0], dx : dx + c.shape[1]]
    return float(np.abs(c[ok]).sum()), int(ok.sum())


class NoiseStats:
    """Pooled in-plane noise (Immerkaer, per layer) over voxel spread of one scan over several areas; the ratio is
    the in-plane sharpness measure."""

    def __init__(self):
        self.s = self.c = 0.0
        self.n = self.t = self.q = 0.0

    def add(self, vol, valid2d):
        v = np.asarray(vol, np.float64)
        for layer in v:
            s, c = immerkaer_sum(layer, valid2d)
            self.s += s
            self.c += c
        x = v[:, valid2d]
        self.n += x.size
        self.t += float(x.sum())
        self.q += float((x * x).sum())
        return self

    def sigma(self):
        return IMMERKAER * self.s / self.c if self.c else None

    def spread(self):
        m = self.t / self.n
        return (self.q / self.n - m * m) ** 0.5

    def ratio(self):
        return self.sigma() / self.spread()

    def as_dict(self):
        return {
            "sigma": self.sigma(),
            "spread": self.spread(),
            "ratio": self.ratio(),
            "pixels": int(self.c),
            "voxels": int(self.n),
        }


class DepthStats:
    """Pooled depth noise over voxel spread: the second difference along depth per valid in-plane position."""

    def __init__(self):
        self.s = self.c = 0.0
        self.n = self.t = self.q = 0.0

    def add(self, vol, valid2d):
        v = np.asarray(vol, np.float64)
        ok = np.asarray(valid2d, bool)
        if v.shape[0] >= 3:
            d = (v[:-2] - 2 * v[1:-1] + v[2:])[:, ok]
            self.s += float(np.abs(d).sum())
            self.c += d.size
        x = v[:, ok]
        self.n += x.size
        self.t += float(x.sum())
        self.q += float((x * x).sum())
        return self

    def sigma(self):
        return DEPTH_K * self.s / self.c if self.c else None

    def spread(self):
        m = self.t / self.n
        return (self.q / self.n - m * m) ** 0.5

    def ratio(self):
        return self.sigma() / self.spread()

    def as_dict(self):
        return {
            "sigma": self.sigma(),
            "spread": self.spread(),
            "ratio": self.ratio(),
            "pairs": int(self.c),
            "voxels": int(self.n),
        }


def layer_ratios(vols_valid):
    """The in-plane ratio per layer, pooled over areas [(vol, valid2d), ...] of equal depth (reported only)."""
    nk = len(vols_valid[0][0])
    out = []
    for k in range(nk):
        st = NoiseStats()
        for v, ok in vols_valid:
            st.add(v[k : k + 1], ok)
        out.append(round(float(st.ratio()), 6) if st.c and st.n else None)
    return out


def coarse4(fine_vol):
    """The 4 x 4 x 4 box average of a (60, n, n) fine volume on its own coarse grid, (15, n/4, n/4) float64, and the
    coarse cells whose 16 in-plane fine px are all rendered."""
    v = np.asarray(fine_vol, np.float64)
    D, H, W = v.shape
    c = v.reshape(D // 4, 4, H // 4, 4, W // 4, 4).mean(axis=(1, 3, 5))
    rend = (v.mean(axis=0) > 0).reshape(H // 4, 4, W // 4, 4).all(axis=(1, 3))
    return c, rend


def real_area(site, real_vol, origin, n, box, scale, shift, dz):
    """The real voxels an area's footprint covers on the real grid, and the in-plane rendered mask."""
    ry, rx = real_index(origin, n, box, scale, shift)
    rz = np.unique(real_layer_map(site, dz))
    sub = np.asarray(real_vol, np.float64)[rz][:, np.unique(ry)][:, :, np.unique(rx)]
    return sub, sub.mean(axis=0) > 0


def noise_to_add(real_ratio, box_sigma, box_spread):
    """The standard deviation of Gaussian noise that would give the box average the real scan's noise-to-spread
    ratio (reported only; 0 when the real scan is not noisier by this measure)."""
    rho = float(real_ratio)
    if rho >= 1:
        raise ValueError("refused: noise ratio %.3f >= 1" % rho)
    a2 = (rho * rho * box_spread * box_spread - box_sigma * box_sigma) / (1 - rho * rho)
    return max(0.0, a2) ** 0.5


# ---------------------------------------------------------------- blur fits (CT only)


def gauss_blur3d_axes(c, mask, sigmas):
    """Normalized Gaussian blur of a coarse (D, H, W) array over the voxels in mask (edges reflected), one sigma per
    axis (depth, rows, columns) in coarse voxels: blur(c * mask) / blur(mask) inside mask, c unchanged outside."""
    c = np.asarray(c, np.float64)
    if all(s <= 0 for s in sigmas):
        return c.copy()
    m = np.asarray(mask, np.float64)
    ks = []
    for s in sigmas:
        if s <= 0:
            ks.append(None)
            continue
        rad = max(1, int(np.ceil(3 * s)))
        k = np.exp(-0.5 * (np.arange(-rad, rad + 1) / s) ** 2)
        ks.append((rad, k / k.sum()))

    def conv(a):
        for ax in range(3):
            if ks[ax] is None:
                continue
            rad, k = ks[ax]
            pad = [(0, 0)] * 3
            pad[ax] = (rad, rad)
            ap = np.pad(a, pad, mode="symmetric")
            out = np.zeros_like(a)
            n = a.shape[ax]
            for i, w in enumerate(k):
                sl = [slice(None)] * 3
                sl[ax] = slice(i, i + n)
                out += w * ap[tuple(sl)]
            a = out
        return a

    num, den = conv(c * m), conv(m)
    with np.errstate(invalid="ignore", divide="ignore"):
        b = num / den
    return np.where(m > 0, b, c)


def gauss_blur3d(c, mask, sigma):
    """gauss_blur3d_axes with one sigma on every axis."""
    if sigma <= 0:
        return np.asarray(c, np.float64).copy()
    return gauss_blur3d_axes(c, mask, (sigma, sigma, sigma))


def blur_coarse(c, rend, sigma):
    """The blurred coarse array: gauss_blur3d over nonzero voxels of fully rendered cells, clipped to 1..255 there
    (c elsewhere), rounded to whole CT units as the 8-bit input will be."""
    m = rend[None] & (c > 0)
    return np.rint(np.where(m, np.clip(gauss_blur3d(c, m, sigma), 1, 255), c))


def blur_coarse_axes(c, rend, sz, sxy):
    """blur_coarse with sigma sz along depth and sxy along rows and columns."""
    m = rend[None] & (c > 0)
    return np.rint(np.where(m, np.clip(gauss_blur3d_axes(c, m, (sz, sxy, sxy)), 1, 255), c))


def blurred_box_average(fine_vol, sigma):
    """The box average blurred at sigma, rounded, repeated to the fine grid: (60, n, n) uint8."""
    c, rend = coarse4(fine_vol)
    q = blur_coarse(c, rend, sigma).astype(np.uint8)
    return np.ascontiguousarray(q.repeat(4, 0).repeat(4, 1).repeat(4, 2))


def blurred_box_average_axes(fine_vol, sz, sxy):
    """blurred_box_average with sz along depth and sxy in plane, (60, n, n) uint8."""
    c, rend = coarse4(fine_vol)
    q = blur_coarse_axes(c, rend, sz, sxy).astype(np.uint8)
    return np.ascontiguousarray(q.repeat(4, 0).repeat(4, 1).repeat(4, 2))


def blur_ratio(coarse_list, sigma):
    """Pooled in-plane ratio of the coarse arrays [(c, rend2d), ...] blurred at sigma (rounding included)."""
    st = NoiseStats()
    for c, rend in coarse_list:
        st.add(blur_coarse(c, rend, sigma), rend)
    return st.ratio()


def match_blur(coarse_list, target, hi=4.0, iters=12):
    """The Gaussian sigma (coarse voxels) at which the box average's pooled ratio equals target, by bisection on
    [0, hi]; 0 when it is already no sharper; refuses a target the largest sigma cannot reach. Returns (sigma, ratio
    at sigma, [(sigma, ratio), ...])."""
    r0 = blur_ratio(coarse_list, 0.0)
    trace = [(0.0, r0)]
    if r0 <= target:
        return 0.0, r0, trace
    rh = blur_ratio(coarse_list, hi)
    trace.append((hi, rh))
    if rh > target:
        raise ValueError("refused: ratio %.4f at sigma %.1f is still above the target %.4f" % (rh, hi, target))
    lo, up = 0.0, hi
    for _ in range(iters):
        mid = (lo + up) / 2
        r = blur_ratio(coarse_list, mid)
        trace.append((mid, r))
        if r > target:
            lo = mid
        else:
            up = mid
    sig = (lo + up) / 2
    return sig, blur_ratio(coarse_list, sig), trace


def blur_check(rec, tol=SM_TOL):
    """On the windows, |blurred ratio / real ratio - 1| <= tol. NONE when sigma is 0."""
    if rec is None:
        return {"verdict": "FAIL", "why": "no blur record"}
    if not rec.get("sigma"):
        return {"verdict": "NONE", "sigma": rec.get("sigma")}
    rr, rt = rec["windows_real"]["ratio"], rec["windows_box_blur"]["ratio"]
    dev = rt / rr - 1
    return {
        "verdict": "PASS" if abs(dev) <= tol else "FAIL",
        "sigma": rec["sigma"],
        "real_ratio": rr,
        "box_blur_ratio": rt,
        "deviation": dev,
    }


# ---------------------------------------------------------------- gray-level map (CT only)


def gray_level_hist(src_vol, real_vol, hs=None, ht=None):
    """Add one area's gray-level counts per coarse layer to (hs, ht), each (D/4, 256), on fine voxels where both
    volumes are rendered (> 0)."""
    a = np.asarray(src_vol)
    b = np.asarray(real_vol)
    if a.shape != b.shape:
        raise ValueError("volumes differ in shape: %s against %s" % (a.shape, b.shape))
    nk = a.shape[0] // 4
    if hs is None:
        hs = np.zeros((nk, 256), np.int64)
        ht = np.zeros((nk, 256), np.int64)
    for k in range(nk):
        x = a[4 * k : 4 * k + 4]
        y = b[4 * k : 4 * k + 4]
        m = (x > 0) & (y > 0)
        hs[k] += np.bincount(x[m].astype(np.int64), minlength=256)[:256]
        ht[k] += np.bincount(y[m].astype(np.int64), minlength=256)[:256]
    return hs, ht


def _inv_cdf(h, p):
    """The gray level at cumulative share p of histogram h; each level spreads its count over [v - 0.5, v + 0.5]."""
    h = np.asarray(h, np.float64)
    tot = h.sum()
    if tot <= 0:
        raise ValueError("refused: an empty target histogram")
    c = np.cumsum(h) / tot
    xs, ys = [], []
    prev = 0.0
    for v in np.nonzero(h)[0]:
        xs += [prev, c[v]]
        ys += [v - 0.5, v + 0.5]
        prev = c[v]
    return np.interp(p, np.array(xs), np.array(ys))


def gray_level_fit(hs, ht):
    """The gray-level map per coarse layer, (D/4, 256) uint8: m(0) = 0; for v = 1..255 the target's level at the
    source's mid-rank share, rounded, clipped to 1..255."""
    hs = np.asarray(hs, np.int64)
    ht = np.asarray(ht, np.int64)
    out = np.zeros(hs.shape, np.uint8)
    for k in range(hs.shape[0]):
        h = hs[k].astype(np.float64)
        h[0] = 0
        tot = h.sum()
        if tot <= 0:
            out[k] = np.arange(256)
            continue
        c = np.cumsum(h) / tot
        prev = np.concatenate([[0.0], c[:-1]])
        p = (prev + c) / 2
        t = ht[k].copy()
        t[0] = 0
        m = np.rint(_inv_cdf(t, p))
        out[k] = np.clip(m, 1, 255).astype(np.uint8)
        out[k, 0] = 0
    return out


def gray_level_apply(vol, maps):
    """Each fine layer of a (60, n, n) uint8 volume through its coarse layer's gray-level map."""
    v = np.asarray(vol)
    out = np.empty_like(v)
    for z in range(v.shape[0]):
        out[z] = maps[z // 4][v[z]]
    return out


def w1_from_hists(hs, ht):
    """Mean over coarse layers of the 1-Wasserstein distance (CT units) between two gray-level histograms."""
    per = []
    for k in range(hs.shape[0]):
        a, b = hs[k].astype(np.float64), ht[k].astype(np.float64)
        a[0] = b[0] = 0
        if a.sum() <= 0 or b.sum() <= 0:
            per.append(None)
            continue
        per.append(float(np.abs(np.cumsum(a) / a.sum() - np.cumsum(b) / b.sum()).sum()))
    vals = [x for x in per if x is not None]
    return (float(np.mean(vals)) if vals else None), per


def gray_level_check(rec, share=CM_SHARE, absolute=CM_ABS):
    """On the windows, W1 after the map <= share x W1 before it, or <= absolute CT units."""
    if not rec or rec.get("windows_w1_before") is None or rec.get("windows_w1_after") is None:
        return {"verdict": "FAIL", "why": "no gray-level record"}
    b, a = rec["windows_w1_before"], rec["windows_w1_after"]
    ok = a <= share * b or a <= absolute
    return {"verdict": "PASS" if ok else "FAIL", "w1_before": b, "w1_after": a, "share": share, "absolute": absolute}


def cell_weights(real_vol):
    """Per coarse cell of a (60, n, n) fine volume, the count of its 64 fine voxels where the real scan is > 0."""
    b = np.asarray(real_vol) > 0
    D, H, W = b.shape
    return b.reshape(D // 4, 4, H // 4, 4, W // 4, 4).sum(axis=(1, 3, 5)).astype(np.int64)


def coarse_hist(q, w, hs=None):
    """gray_level_hist's source counts from the coarse array q and the cell weights w, without the fine grid."""
    q = np.asarray(q)
    nk = q.shape[0]
    if hs is None:
        hs = np.zeros((nk, 256), np.int64)
    for k in range(nk):
        m = q[k] > 0
        hs[k] += np.rint(
            np.bincount(q[k][m].astype(np.int64), weights=w[k][m].astype(np.float64), minlength=256)[:256]
        ).astype(np.int64)
    return hs


def target_hist(q, real_vol, ht=None):
    """gray_level_hist's target counts with the source mask taken from the coarse array q."""
    q = np.asarray(q)
    m4 = (q > 0).repeat(4, 0).repeat(4, 1).repeat(4, 2)
    y = np.asarray(real_vol)
    nk = q.shape[0]
    if ht is None:
        ht = np.zeros((nk, 256), np.int64)
    for k in range(nk):
        yy = y[4 * k : 4 * k + 4]
        m = m4[4 * k : 4 * k + 4] & (yy > 0)
        ht[k] += np.bincount(yy[m].astype(np.int64), minlength=256)[:256]
    return ht


def map_coarse(q, maps):
    """The gray-level maps applied to a coarse array of whole CT units: (15, h, w) float64."""
    q = np.asarray(q).astype(np.int64)
    out = np.empty(q.shape, np.float64)
    for k in range(q.shape[0]):
        out[k] = maps[k][q[k]]
    return out


def matched_ratio(items, ht, sigma):
    """One blur sigma with its gray-level map: (pooled in-plane ratio, maps, source counts) over the training tiles'
    (c, rend, w)."""
    hs = None
    qs = []
    for c, rend, w in items:
        q = blur_coarse(c, rend, sigma)
        qs.append(q)
        hs = coarse_hist(q, w, hs)
    maps = gray_level_fit(hs, ht)
    st = NoiseStats()
    for q in qs:
        mq = map_coarse(q, maps)
        st.add(mq, mq.mean(axis=0) > 0)
    return st.ratio(), maps, hs


def match_matched(items, ht, target, hi=SIGMA_MAX, iters=12):
    """One blur sigma, fitted with its gray-level map, by bisection on [0, hi] so the mapped tiles' pooled ratio
    equals target. Returns (sigma, ratio at sigma, trace, maps at sigma, source counts at sigma)."""
    r0, _, _ = matched_ratio(items, ht, 0.0)
    trace = [(0.0, r0)]
    if r0 <= target:
        r, maps, hs = matched_ratio(items, ht, 0.0)
        return 0.0, r, trace, maps, hs
    rh, _, _ = matched_ratio(items, ht, hi)
    trace.append((hi, rh))
    if rh > target:
        raise ValueError("refused: mapped ratio %.4f at sigma %.1f is still above the target %.4f" % (rh, hi, target))
    lo, up = 0.0, hi
    for _ in range(iters):
        mid = (lo + up) / 2
        r, _, _ = matched_ratio(items, ht, mid)
        trace.append((mid, r))
        if r > target:
            lo = mid
        else:
            up = mid
    sig = (lo + up) / 2
    r, maps, hs = matched_ratio(items, ht, sig)
    return sig, r, trace, maps, hs


def matched_blur_checks(rec, tol=SM_TOL):
    """The fit was not refused, the coarse shortcut equals the fine-grid refit, and the gray-level and sharpness
    checks pass on the windows."""
    if not rec:
        return {"verdict": "FAIL", "why": ["no matched blur record"]}
    why = []
    if rec.get("refused"):
        why.append("fit refused: %s" % rec["refused"])
    if rec.get("shortcut_equal") is not True:
        why.append("coarse shortcut differs from the fine-grid refit")
    cm = gray_level_check(rec)
    if "windows_box_blur_matched" not in rec or "windows_real" not in rec:
        sm = {"verdict": "FAIL", "why": "no sharpness record"}
    else:
        rr, rt = rec["windows_real"]["ratio"], rec["windows_box_blur_matched"]["ratio"]
        dev = rt / rr - 1
        sm = {"verdict": "PASS" if abs(dev) <= tol else "FAIL", "real_ratio": rr, "ratio": rt, "deviation": dev}
    if cm["verdict"] != "PASS":
        why.append("gray levels FAIL")
    if sm["verdict"] != "PASS":
        why.append("sharpness FAIL")
    return {"verdict": "FAIL" if why else "PASS", "why": why, "gray_levels": cm, "sharpness": sm}


def directional_ratios(items, ht, sz, sxy, mapper=map):
    """One (sz, sxy) with its gray-level map: (pooled in-plane ratio, pooled depth ratio, maps, source counts)."""
    qs = list(mapper(lambda it: blur_coarse_axes(it[0], it[1], sz, sxy), items))
    hs = None
    for q, (_, _, w) in zip(qs, items):
        hs = coarse_hist(q, w, hs)
    maps = gray_level_fit(hs, ht)
    st, dz = NoiseStats(), DepthStats()
    for q in qs:
        mq = map_coarse(q, maps)
        ok = mq.mean(axis=0) > 0
        st.add(mq, ok)
        dz.add(mq, ok)
    return st.ratio(), dz.ratio(), maps, hs


def match_directional(items, ht, target_xy, target_z, hi_xy=SIGMA_MAX, hi_z=SZ_MAX, iters=12, mapper=map):
    """sz by bisection on [0, hi_z] so the depth ratio at (sz, sxy(sz)) equals target_z, where sxy(sz) is found by
    bisection on [0, hi_xy] so the in-plane ratio equals target_xy. Returns (sz, sxy, in-plane ratio, depth ratio,
    trace, maps, source counts)."""
    trace = []

    def inner(sz):
        r0 = directional_ratios(items, ht, sz, 0.0, mapper)
        if r0[0] <= target_xy:
            return 0.0, r0
        rh = directional_ratios(items, ht, sz, hi_xy, mapper)
        if rh[0] > target_xy:
            raise ValueError(
                "refused: in-plane ratio %.4f at sxy %.1f (sz %.3f) is still above the target %.4f"
                % (rh[0], hi_xy, sz, target_xy)
            )
        lo, up = 0.0, hi_xy
        for _ in range(iters):
            mid = (lo + up) / 2
            if directional_ratios(items, ht, sz, mid, mapper)[0] > target_xy:
                lo = mid
            else:
                up = mid
        s = (lo + up) / 2
        return s, directional_ratios(items, ht, sz, s, mapper)

    def at(sz):
        sxy, r = inner(sz)
        trace.append((sz, sxy, r[0], r[1]))
        return sxy, r

    sxy, r = at(0.0)
    if r[1] <= target_z:
        return 0.0, sxy, r[0], r[1], trace, r[2], r[3]
    _, rh = at(hi_z)
    if rh[1] > target_z:
        raise ValueError(
            "refused: depth ratio %.4f at sz %.1f is still above the target %.4f" % (rh[1], hi_z, target_z)
        )
    lo, up = 0.0, hi_z
    for _ in range(iters):
        mid = (lo + up) / 2
        if at(mid)[1][1] > target_z:
            lo = mid
        else:
            up = mid
    sz = (lo + up) / 2
    sxy, r = at(sz)
    return sz, sxy, r[0], r[1], trace, r[2], r[3]


def directional_blur_checks(rec, tol=SM_TOL, ztol=SM_TOL):
    """The fit was not refused, the coarse shortcut equals the fine-grid refit, the sub-voxel registration passed,
    and the gray-level, in-plane and depth sharpness checks pass on the windows."""
    if not rec:
        return {"verdict": "FAIL", "why": ["no directional blur record"]}
    why = []
    if rec.get("refused"):
        why.append("fit refused: %s" % rec["refused"])
    if rec.get("shortcut_equal") is not True:
        why.append("coarse shortcut differs from the fine-grid refit")
    if (rec.get("registration_subvoxel") or {}).get("verdict") != "PASS":
        why.append("sub-voxel registration not PASS")
    out = {"gray_levels": gray_level_check(rec)}
    for key, real_key, got_key, t in (
        ("sharpness", "windows_real", "windows_box_blur_directional", tol),
        ("depth_sharpness", "windows_real_depth", "windows_box_blur_directional_depth", ztol),
    ):
        if real_key in rec and got_key in rec:
            rr, rt = rec[real_key]["ratio"], rec[got_key]["ratio"]
            dev = rt / rr - 1
            out[key] = {
                "verdict": "PASS" if abs(dev) <= t else "FAIL",
                "real_ratio": rr,
                "ratio": rt,
                "deviation": dev,
            }
        else:
            out[key] = {"verdict": "FAIL", "why": "no %s record" % key}
    for key in ("gray_levels", "sharpness", "depth_sharpness"):
        if out[key]["verdict"] != "PASS":
            why.append("%s FAIL" % key)
    out.update({"verdict": "FAIL" if why else "PASS", "why": why})
    return out


def directional_rule(rec, sigma_matched, tol=ANISO_TOL):
    """Reported only: whether the directional blur differs from the single matched blur by more than tol."""
    g = directional_blur_checks(rec)
    if g["verdict"] != "PASS":
        return {"decision": "VOID", "why": g["why"]}
    sz, sxy = rec["sz"], rec["sxy"]
    dev = max(abs(sz / sigma_matched - 1), abs(sxy / sigma_matched - 1))
    return {
        "decision": "DIRECTIONAL" if dev > tol else "NO DIRECTION",
        "deviation": dev,
        "sz": sz,
        "sxy": sxy,
        "sigma_matched": sigma_matched,
        "tol": tol,
    }


# ---------------------------------------------------------------- villa's 9 um recipe


def villa_pool(level2_block):
    """prepare_9um_isotropic_input on one full-depth level-2 block: the centered planes, rounded float32 means of
    POOL_Z. Returns (uint8 planes, (z0, z1))."""
    z0, z1 = centered_slice(level2_block.shape[0], VILLA_PLANES_IN)
    b = np.asarray(level2_block[z0:z1], np.float32)
    return np.rint(b.reshape(VILLA_PLANES, POOL_Z, *b.shape[1:]).mean(axis=1)).astype(np.uint8), (z0, z1)


def villa_grid(n_fine, box_start, depth=DEPTH):
    """Where villa's planes sit against the box average's: villa's first fine layer z0, the plane offset (box group
    i pairs with villa plane i + offset), the phase in fine layers and the fine layers ``villa_input`` covers."""
    z0, _ = centered_slice(n_fine, VILLA_PLANES_IN)
    off = int(math.floor((box_start - z0) / POOL_Z + 0.5))
    g = depth // POOL_Z
    if off < 0 or off + g > VILLA_PLANES:
        raise ValueError(
            "the box average's planes (from layer %d) do not pair with %d of villa's planes (offset %d)"
            % (box_start, g, off)
        )
    return {
        "n_fine": int(n_fine),
        "box_first_layer": int(box_start),
        "villa_first_layer": int(z0),
        "villa_layers": [int(z0), int(z0 + VILLA_PLANES_IN)],
        "offset": off,
        "phase_fine_layers": int((box_start - z0) % POOL_Z),
        "villa_input_layers": [int(z0 + POOL_Z * off), int(z0 + POOL_Z * (off + g))],
        "villa_minus_box_layers": int(z0 + POOL_Z * off - box_start),
    }


def to_fine(planes, z=POOL_Z, xy=XY):
    """(p, h, w) coarse planes repeated z times in depth and xy x xy in plane, uint8."""
    p = np.asarray(planes, np.uint8)
    return np.repeat(np.repeat(np.repeat(p, z, axis=0), xy, axis=1), xy, axis=2)


def villa_planes(level2, box_start):
    """The ``villa_input`` coarse planes of one area: villa's 21 planes, then the 15 the box average's groups
    overlap most. Returns (planes, grid record)."""
    pooled, (z0, _) = villa_pool(level2)
    g = villa_grid(level2.shape[0], box_start)
    assert g["villa_first_layer"] == z0
    return pooled[g["offset"] : g["offset"] + GROUPS], g


def villa_planes_on_box_layers(level2, box_start):
    """villa's arithmetic (float32 mean of 4 level-2 planes, rounded) on the box average's own 60 layers."""
    b = np.asarray(level2[box_start : box_start + GROUPS * POOL_Z], np.float32)
    if b.shape[0] != GROUPS * POOL_Z:
        raise ValueError("level 2 has %d layers from %d, fewer than %d" % (b.shape[0], box_start, GROUPS * POOL_Z))
    return np.rint(b.reshape(GROUPS, POOL_Z, *b.shape[1:]).mean(axis=1)).astype(np.uint8)


def villa_input(level2, box_start):
    """The (60, n, n) uint8 ``villa_input`` of one area from its full-depth level-2 block."""
    return to_fine(villa_planes(level2, box_start)[0])


def box4_at(fine_full, z0, nplanes, z=POOL_Z):
    """The 4 x 4 x 4 box average (float64, no rounding) of a full-depth level-0 block from fine layer z0."""
    v = np.asarray(fine_full[z0 : z0 + nplanes * z], np.float64)
    D, H, W = v.shape
    return v.reshape(D // z, z, H // 4, 4, W // 4, 4).mean(axis=(1, 3, 5))


def cell_diff(a, b):
    """max and mean |a - b| on cells where both are above 0 (CT units), with the cell count."""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    m = (a > 0) & (b > 0)
    d = np.abs(a - b)[m]
    return {
        "cells": int(m.sum()),
        "max_abs": float(d.max()) if d.size else None,
        "mean_abs": float(d.mean()) if d.size else None,
    }


def alignment_check(villa_box_layers, box, tol=POOL_TOL):
    """villa's arithmetic on the box average's layers against the level-0 box average: PASS when the largest
    difference on cells both above 0 is at most tol. It fails when level 2 is off the level-0 grid."""
    r = cell_diff(villa_box_layers, box)
    r["tol"] = tol
    r["verdict"] = "PASS" if r["cells"] and r["max_abs"] <= tol else "FAIL"
    return r


def villa_check(rec):
    """The stage record's villa grid equals villa_grid recomputed from its site, and the alignment check passed on
    every window."""
    why = []
    v = (rec or {}).get("villa") or {}
    geo = (rec or {}).get("site") or {}
    try:
        want = villa_grid(int(geo.get("n_fine")), int((geo.get("fine_layers") or [None])[0]))
    except (TypeError, ValueError) as exc:
        want = None
        why.append("grid: %s" % exc)
    if want is not None and v.get("grid") != want:
        why.append("grid differs from the site's layer count")
    al = [w.get("alignment", {}).get("verdict") for w in v.get("windows") or []]
    if not al or any(x != "PASS" for x in al) or len(al) != len((rec or {}).get("folds") or []):
        why.append("alignment not PASS on every window (%s)" % al)
    return {"verdict": "FAIL" if why else "PASS", "why": why}


# ---------------------------------------------------------------- windows and tiles (labels, level 3)


def pick_windows(lab3, sup3, step=WINDOW_STEP3, max_windows=MAX_WINDOWS):
    """Windows of step x step level-3 px with supervision >= 0.9 and an ink share of the supervised pixels in
    [0.15, 0.50], ordered by |share - 0.30| then row then column, taken greedily without overlap."""
    cands = []
    H, W = lab3.shape
    for r in range(0, H - step + 1, step):
        for c in range(0, W - step + 1, step):
            s = sup3[r : r + step, c : c + step] > 0
            cover = float(s.mean())
            if cover < 0.9:
                continue
            ink = float((lab3[r : r + step, c : c + step] > 0)[s].mean())
            if 0.15 <= ink <= 0.50:
                cands.append((abs(ink - 0.30), r, c, cover, ink))
    cands.sort()
    taken = []
    for d, r, c, cover, ink in cands:
        if all(abs(r - t["row3"]) >= step or abs(c - t["col3"]) >= step for t in taken):
            taken.append(
                {
                    "row3": r,
                    "col3": c,
                    "y0": r * 8,
                    "x0": c * 8,
                    "size": step * 8,
                    "supervised": round(cover, 4),
                    "ink_share": round(ink, 4),
                }
            )
        if len(taken) == max_windows:
            break
    return taken, len(cands)


def pick_tiles(sup3, step=TILE_STEP3, min_sup=TILE_MIN_SUP):
    """Every step x step training tile of level 3 with supervised share >= min_sup."""
    H, W = sup3.shape
    out = []
    for r in range(0, H - step + 1, step):
        for c in range(0, W - step + 1, step):
            cov = float((sup3[r : r + step, c : c + step] > 0).mean())
            if cov >= min_sup:
                out.append(
                    {"row3": r, "col3": c, "y0": r * 8, "x0": c * 8, "size": step * 8, "supervised": round(cov, 4)}
                )
    return out


def touches(y0, x0, n, win, guard=FOLD_GUARD):
    """True when the n x n box at (y0, x0) intersects the window dilated by guard on every side."""
    wy, wx, ws = win["y0"] - guard, win["x0"] - guard, win["size"] + 2 * guard
    return y0 < wy + ws and wy < y0 + n and x0 < wx + ws and wx < x0 + n


def allowed_origins(tile, o, p, win, patch=PATCH, guard=FOLD_GUARD):
    """Boolean (len(o), len(p)): patch origins of a tile whose box stays outside the window's guard."""
    ok = np.ones((len(o), len(p)), bool)
    for i, oy in enumerate(o):
        for j, ox in enumerate(p):
            ok[i, j] = not touches(tile["y0"] + int(oy), tile["x0"] + int(ox), patch, win, guard)
    return ok


def load_layout(windows_path, tiles_path, smoke=False):
    """(windows, tiles, tile size, window size, folds) from the windows and tiles files."""
    wj = json.load(open(windows_path))
    tj = json.load(open(tiles_path))
    wins, tiles = wj["windows"], tj["tiles"]
    tn = tiles[0]["size"]
    if smoke:
        o = np.arange(0, tn - PATCH + 1, SUPERBLOCK)
        tiles = [t for t in tiles if allowed_origins(t, o, o, wins[0]).any()][:3]
    n = SMOKE_WINDOW if smoke else WINDOW
    folds = [0] if smoke else list(range(len(wins)))
    return wins, tiles, tn, n, folds


# ---------------------------------------------------------------- inputs and the plan


def np_norm(vols):
    """The model's input normalization (voxels > 0, / 255, float64): (mean, sd)."""
    cnt = tot = sq = 0.0
    for v in vols:
        x = v[v > 0].astype(np.float64) / 255.0
        cnt += x.size
        tot += float(x.sum())
        sq += float((x * x).sum())
    m = tot / cnt
    return m, (sq / cnt - m * m) ** 0.5


def area_inputs(name, stages, fine_t, evals, real=None, villa=None):
    """(tile inputs, window inputs) of one input, (60, n, n) uint8 each, from the stage record. real: {"real": ...,
    "real_registered": ...} (tiles, windows); villa: (tiles, windows) of ``villa_input``."""
    if name in ("fine", "box_average"):
        return list(fine_t), [e["vol"] for e in evals]
    if name in ("real", "real_registered"):
        return real[name]
    if name == "villa_input":
        return villa
    if name in DOSE_MULT:
        s = DOSE_MULT[name] * float(stages["blur"]["sigma"])
        return [blurred_box_average(v, s) for v in fine_t], [blurred_box_average(e["vol"], s) for e in evals]
    if name == "box_blur_matched":
        j = stages["matched_blur"]
        m = np.array(j["maps"], np.uint8)
        f = lambda v: gray_level_apply(blurred_box_average(v, float(j["sigma"])), m)  # noqa: E731
    elif name == "box_blur_directional":
        k = stages["directional_blur"]
        m = np.array(k["maps"], np.uint8)
        f = lambda v: gray_level_apply(blurred_box_average_axes(v, float(k["sz"]), float(k["sxy"])), m)  # noqa: E731
    else:
        raise ValueError(f"unknown input {name}")
    return [f(v) for v in fine_t], [f(e["vol"]) for e in evals]


def training_plan(stages):
    """The inputs trained, from the label-free stages only: fine, box_average and real always; real_registered when
    the sub-voxel registration passes; box_blur and its multiples when the in-plane blur fit gave sigma > 0;
    box_blur_matched and box_blur_directional when their checks pass; villa_input when its checks pass."""
    plan, skipped = {}, {}
    regs = ((stages.get("registration_subvoxel") or {}).get("check") or {}).get("verdict")
    blur = stages.get("blur") or {}
    for name in ("fine", "box_average", "real"):
        plan[name] = "always"
    (plan if regs == "PASS" else skipped)["real_registered"] = "sub-voxel registration %s" % regs
    sig = blur.get("sigma")
    for name in DOSE_MULT:
        (plan if (sig and not blur.get("refused")) else skipped)[name] = "in-plane blur sigma %s" % sig
    jg = matched_blur_checks(stages.get("matched_blur")) if stages.get("matched_blur") else {"verdict": "FAIL"}
    (plan if jg["verdict"] == "PASS" else skipped)["box_blur_matched"] = "matched blur checks %s" % jg["verdict"]
    kg = (
        directional_blur_checks(stages.get("directional_blur"))
        if stages.get("directional_blur")
        else {"verdict": "FAIL"}
    )
    (plan if kg["verdict"] == "PASS" else skipped)["box_blur_directional"] = (
        "directional blur checks %s" % kg["verdict"]
    )
    if "villa" in stages:
        vc = villa_check(stages)
        (plan if vc["verdict"] == "PASS" else skipped)["villa_input"] = "villa checks %s" % vc["verdict"]
    order = [a for a, _ in INPUTS]
    return {
        "train": [a for a in order if a in plan],
        "skip": {a: skipped[a] for a in order if a in skipped},
        "why": plan,
    }


def _logger(path):
    fh = open(path, "a")

    def log(s):
        line = "%s %s" % (time.strftime("%H:%M:%S", time.gmtime()), s)
        print(line, flush=True)
        fh.write(line + "\n")
        fh.flush()

    return log


def run_stages(data, windows_path, tiles_path, out_dir, name="segment", smoke=False, villa=True, workers=8):
    """The label-free stages, in order: both registrations, the noise record (reported), the in-plane blur fit, the
    matched blur with its gray-level map, the directional blur with its gray-level map, villa's recipe and its
    alignment check, the plan and every input's record (window sha256 and normalization) that training must
    reproduce. Reads the CT only. Writes NAME-stages.json and returns the record."""
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    sfx = "-smoke" if smoke else ""
    log = _logger(os.path.join(out_dir, "%s-stages%s.log" % (name, sfx)))
    site = data.site()
    if site["check"]["verdict"] != "PASS":
        raise SystemExit("refused: site check %s" % site["check"]["why"])
    wins, tiles, tn, n, folds = load_layout(windows_path, tiles_path, smoke)
    total = 8
    log(
        "stages: %s, %d tiles, windows %s, smoke %s, fine layers %s"
        % (name, len(tiles), folds, smoke, site["fine_layers"])
    )
    ex = ThreadPoolExecutor(workers)
    fine_t = list(ex.map(lambda t: data.read_ct(site, t["y0"], t["x0"], tn), tiles))
    evals = [{"fold": j, "vol": data.read_ct(site, wins[j]["y0"], wins[j]["x0"], n)} for j in folds]
    log("PROGRESS 1/%d CT read (%.0f s)" % (total, time.time() - t0))
    rec = {
        "stage": "label-free",
        "name": name,
        "fine": data.fine_path,
        "coarse": data.coarse_path,
        "smoke": bool(smoke),
        "site": site,
        "code_sha256": {"resolution_gap.py": sha_file(__file__)},
        "tiles": len(tiles),
        "folds": folds,
        "windows_file_sha256": sha_file(windows_path),
        "tiles_file_sha256": sha_file(tiles_path),
    }

    # registration
    real, reg_sub, reg_whole, tshift, wshift, dz_whole, rt, rw = real_areas(
        site, data, tiles, wins, evals, fine_t, tn, n, log
    )
    reg_whole["check"] = registration_check(reg_whole)
    rec["registration"], rec["registration_subvoxel"] = reg_whole, reg_sub
    log(
        "PROGRESS 2/%d registration (%.0f s): whole voxel %s, sub-voxel %s"
        % (total, time.time() - t0, reg_whole["check"]["verdict"], reg_sub["check"]["verdict"])
    )
    rs_t, rs_w = real["real_registered"]
    scale = data.scale
    srl, srz, wr, wrz, box_st = NoiseStats(), DepthStats(), NoiseStats(), DepthStats(), NoiseStats()
    real_w_layers = []
    for t, fv, (rv, box), sh in zip(tiles, fine_t, rt, tshift):
        sub, ok = real_area(site, rv, (t["y0"], t["x0"]), tn, box, scale, sh, dz_whole)
        srl.add(sub, ok)
        srz.add(sub, ok)
        c, rend = coarse4(fv)
        box_st.add(c, rend)
    for e, (rv, box), sh in zip(evals, rw, wshift):
        w = wins[e["fold"]]
        sub, ok = real_area(site, rv, (w["y0"], w["x0"]), n, box, scale, sh, dz_whole)
        wr.add(sub, ok)
        wrz.add(sub, ok)
        real_w_layers.append((sub, ok))
    del rt, rw
    amp = noise_to_add(srl.ratio(), box_st.sigma(), box_st.spread())
    rec["noise"] = {
        "a": amp,
        "tiles_real": srl.as_dict(),
        "tiles_box_average": box_st.as_dict(),
        "windows_real": wr.as_dict(),
    }
    log(
        "PROGRESS 3/%d real ratios (%.0f s): in plane tiles %.4f windows %.4f; depth tiles %.4f windows %.4f; "
        "noise a %.3f (reported)" % (total, time.time() - t0, srl.ratio(), wr.ratio(), srz.ratio(), wrz.ratio(), amp)
    )

    # in-plane blur
    cl = [coarse4(fv) for fv in fine_t]
    blur = {"target_ratio": srl.ratio(), "windows_real": wr.as_dict(), "refused": None}
    try:
        sig, r_fit, trace = match_blur(cl, srl.ratio(), hi=SIGMA_MAX)
    except ValueError as exc:
        sig, r_fit, trace = None, None, []
        blur["refused"] = str(exc)
    blur.update({"sigma": sig, "tiles_ratio_at_sigma": r_fit, "trace": [[round(x, 5), round(y, 6)] for x, y in trace]})
    if sig:
        wb = NoiseStats()
        for e in evals:
            c, rend = coarse4(blurred_box_average(e["vol"], sig))
            wb.add(c, rend)
        blur["windows_box_blur"] = wb.as_dict()
    blur["check"] = blur_check(blur) if not blur["refused"] else {"verdict": "FAIL", "why": blur["refused"]}
    rec["blur"] = blur
    log("PROGRESS 4/%d in-plane blur (%.0f s): sigma %s, %s" % (total, time.time() - t0, sig, blur["check"]["verdict"]))

    # one blur with the gray-level map
    s_ref = sig or 0.0
    items, ht = [], None
    for (c, rend), rv in zip(cl, rs_t):
        ht = target_hist(blur_coarse(c, rend, s_ref), rv, ht)
        items.append((c, rend, cell_weights(rv)))
    joint = {"refused": None, "registration_subvoxel": reg_sub["check"], "s_ref": s_ref, "windows_real": wr.as_dict()}
    try:
        sj, rj, tr, mj, hsj = match_matched(items, ht, srl.ratio(), hi=SIGMA_MAX)
    except ValueError as exc:
        joint["refused"] = str(exc)
        sj = None
    if sj is not None:
        tb_t = list(ex.map(lambda fv: blurred_box_average(fv, sj), fine_t))
        hs2 = ht2 = None
        for tv, rv in zip(tb_t, rs_t):
            hs2, ht2 = gray_level_hist(tv, rv, hs2, ht2)
        del tb_t
        mj2 = gray_level_fit(hs2, ht2)
        hb = htb = hw = htw = None
        wt = NoiseStats()
        for e, rv in zip(evals, rs_w):
            hb, htb = gray_level_hist(blurred_box_average(e["vol"], s_ref), rv, hb, htb)
            tj = gray_level_apply(blurred_box_average(e["vol"], sj), mj2)
            hw, htw = gray_level_hist(tj, rv, hw, htw)
            c, rend = coarse4(tj)
            wt.add(c, rend)
        joint.update(
            {
                "sigma": sj,
                "sigma_over_blur_sigma": (sj / sig) if sig else None,
                "tiles_ratio_at_sigma": rj,
                "trace": [[round(x, 5), round(y, 6)] for x, y in tr],
                "maps": mj2.tolist(),
                "shortcut_equal": bool(
                    np.array_equal(hs2, hsj) and np.array_equal(ht2, ht) and np.array_equal(mj2, mj)
                ),
                "windows_w1_before": w1_from_hists(hb, htb)[0],
                "windows_w1_after": w1_from_hists(hw, htw)[0],
                "windows_box_blur_matched": wt.as_dict(),
            }
        )
    joint["check"] = matched_blur_checks(joint)
    rec["matched_blur"] = joint
    log(
        "PROGRESS 5/%d matched blur (%.0f s): sigma %s, %s %s"
        % (total, time.time() - t0, sj, joint["check"]["verdict"], joint["check"]["why"])
    )

    # depth and in-plane blur with the gray-level map
    k = {
        "refused": None,
        "registration_subvoxel": reg_sub["check"],
        "target_ratio": srl.ratio(),
        "target_depth_ratio": srz.ratio(),
        "windows_real": wr.as_dict(),
        "windows_real_depth": wrz.as_dict(),
    }
    try:
        sz, sxy, r_xy, r_z, trace, maps, hs = match_directional(items, ht, srl.ratio(), srz.ratio(), mapper=ex.map)
    except ValueError as exc:
        k["refused"] = str(exc)
        sz = None
    if sz is not None:
        tk_t = list(ex.map(lambda fv: blurred_box_average_axes(fv, sz, sxy), fine_t))
        hs2 = ht2 = None
        for tv, rv in zip(tk_t, rs_t):
            hs2, ht2 = gray_level_hist(tv, rv, hs2, ht2)
        del tk_t
        maps2 = gray_level_fit(hs2, ht2)
        hb = htb = hw = htw = None
        wt, wtz = NoiseStats(), DepthStats()
        lk = []
        for e, rv in zip(evals, rs_w):
            hb, htb = gray_level_hist(blurred_box_average(e["vol"], s_ref), rv, hb, htb)
            tk = gray_level_apply(blurred_box_average_axes(e["vol"], sz, sxy), maps2)
            hw, htw = gray_level_hist(tk, rv, hw, htw)
            c, rend = coarse4(tk)
            wt.add(c, rend)
            wtz.add(c, rend)
            lk.append((c, rend))
        k.update(
            {
                "sz": sz,
                "sxy": sxy,
                "tiles_ratio_at_fit": r_xy,
                "tiles_depth_ratio_at_fit": r_z,
                "trace": [[round(float(x), 6) for x in row] for row in trace],
                "maps": maps2.tolist(),
                "shortcut_equal": bool(
                    np.array_equal(hs2, hs) and np.array_equal(ht2, ht) and np.array_equal(maps2, maps)
                ),
                "windows_w1_before": w1_from_hists(hb, htb)[0],
                "windows_w1_after": w1_from_hists(hw, htw)[0],
                "windows_box_blur_directional": wt.as_dict(),
                "windows_box_blur_directional_depth": wtz.as_dict(),
                "layer_ratios": {"real": layer_ratios(real_w_layers), "box_blur_directional": layer_ratios(lk)},
            }
        )
    k["check"] = directional_blur_checks(k)
    if joint.get("sigma") and sz is not None:
        k["directional_rule"] = directional_rule(k, joint["sigma"])
    rec["directional_blur"] = k
    log(
        "PROGRESS 6/%d directional blur (%.0f s): sz %s sxy %s, %s %s"
        % (total, time.time() - t0, sz, k.get("sxy"), k["check"]["verdict"], k["check"]["why"])
    )

    # villa's recipe
    villa_t = villa_w = None
    if villa:
        rec["villa"], villa_t, villa_w = villa_stage(data, site, tiles, wins, folds, tn, n, evals, log)
    rec["plan"] = training_plan(rec)
    rec["inputs"] = {}
    for nm in rec["plan"]["train"]:
        vt, vw = area_inputs(nm, rec, fine_t, evals, real, (villa_t, villa_w))
        m, s = np_norm(vt)
        rec["inputs"][nm] = {"code": INPUT_CODE[nm], "window_sha256": [sha_array(x) for x in vw], "norm": [m, s]}
        log("input %s: norm %.6f %.6f" % (nm, m, s))
        del vt, vw
    log(
        "PROGRESS 7/%d plan: train %s, skip %s (%.0f s)"
        % (total, rec["plan"]["train"], rec["plan"]["skip"], time.time() - t0)
    )
    ex.shutdown()
    rec["seconds"] = round(time.time() - t0, 1)
    p = os.path.join(out_dir, "%s-stages%s.json" % (name, sfx))
    json.dump(rec, open(p, "w"), indent=1)
    log("PROGRESS 8/%d wrote %s sha256 %s (%.0f s)" % (total, p, sha_file(p), time.time() - t0))
    return rec


def villa_inputs(data, site, tiles, wins, folds, tn, n, workers=4):
    """``villa_input`` of every tile and window, from the fine volume's level 2."""
    start = site["fine_layers"][0]

    def area(y0, x0, size):
        return villa_input(data.read_level2(y0, x0, size), start)

    with ThreadPoolExecutor(workers) as ex:
        vt = list(ex.map(lambda t: area(t["y0"], t["x0"], tn), tiles))
    vw = [area(wins[j]["y0"], wins[j]["x0"], n) for j in folds]
    return vt, vw


def villa_stage(data, site, tiles, wins, folds, tn, n, evals, log):
    """villa's grid against the box average's, the alignment check and the reported differences per window, and
    the ``villa_input`` tiles and windows."""
    start = site["fine_layers"][0]
    g = villa_grid(site["n_fine"], start)
    if data.fine2.shape[0] != site["n_fine"]:
        raise SystemExit("refused: level 2 has %d layers, level 0 %d" % (data.fine2.shape[0], site["n_fine"]))
    out = {"grid": g, "level2_shape": list(data.fine2.shape), "windows": []}
    for q, j in enumerate(folds):
        w = wins[j]
        blk = data.read_level2(w["y0"], w["x0"], n)
        box = box4_at(evals[q]["vol"], 0, GROUPS)
        v = villa_planes(blk, start)[0]
        v0 = villa_planes_on_box_layers(blk, start)
        r = {
            "fold": j,
            "alignment": alignment_check(v0, box),
            "villa_against_box_planes": cell_diff(v, box),
            "villa_against_villa_on_box_layers": cell_diff(v, v0),
            "sha256_level2": sha_array(blk),
            "sha256_level0": sha_array(evals[q]["vol"]),
        }
        out["windows"].append(r)
        log(
            "window %d: alignment %s (max %s), villa against the box planes mean %.3f max %s"
            % (
                j,
                r["alignment"]["verdict"],
                r["alignment"]["max_abs"],
                r["villa_against_box_planes"]["mean_abs"] or 0,
                r["villa_against_box_planes"]["max_abs"],
            )
        )
    vt, vw = villa_inputs(data, site, tiles, wins, folds, tn, n)
    return out, vt, vw


# ---------------------------------------------------------------- the model (torch imported lazily)


def build_net(torch):
    nn = torch.nn

    def c2(i, o):
        return nn.Sequential(
            nn.Conv2d(i, o, 3, padding=1),
            nn.GroupNorm(8, o),
            nn.GELU(),
            nn.Conv2d(o, o, 3, padding=1),
            nn.GroupNorm(8, o),
            nn.GELU(),
        )

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv3d(1, STEM, 3, padding=1),
                nn.GroupNorm(8, STEM),
                nn.GELU(),
                nn.Conv3d(STEM, STEM, 3, padding=1),
                nn.GroupNorm(8, STEM),
                nn.GELU(),
            )
            self.e1 = c2(STEM * BINS, 64)
            self.e2 = c2(64, 96)
            self.e3 = c2(96, 128)
            self.d2 = c2(128 + 96, 96)
            self.d1 = c2(96 + 64, 64)
            self.head = nn.Conv2d(64, 1, 1)

        def forward(self, x):  # x: (B, 1, 60, H, W), H and W multiples of 4
            fn = torch.nn.functional
            h = self.stem(x)
            h = fn.avg_pool3d(h, (DEPTH // BINS, 1, 1))
            B, C, D, H, W = h.shape
            h = h.reshape(B, C * D, H, W)
            a = self.e1(h)
            b = self.e2(fn.avg_pool2d(a, 2))
            c = self.e3(fn.avg_pool2d(b, 2))
            u = fn.interpolate(c, size=b.shape[-2:], mode="bilinear", align_corners=False)
            u = self.d2(torch.cat([u, b], 1))
            u = fn.interpolate(u, size=a.shape[-2:], mode="bilinear", align_corners=False)
            u = self.d1(torch.cat([u, a], 1))
            return self.head(u)[:, 0]

    return Net()


def degrade(torch, x, code):
    """The ``box_average`` input is a 4 x 4 x 4 box average of the fine input inside the network's input, repeated
    back to the fine grid; every other input is prepared before training."""
    if code != INPUT_CODE["box_average"]:
        return x
    fn = torch.nn.functional
    y = fn.avg_pool3d(x, 4, stride=4)
    return y.repeat_interleave(4, 2).repeat_interleave(4, 3).repeat_interleave(4, 4)


def patch_sup_table(sup, patch, step=SUPERBLOCK):
    """Supervised share of every patch origin on the step grid of one tile: (n, n), with the origins."""
    s = (sup > 0).astype(np.float64)
    cum = np.zeros((s.shape[0] + 1, s.shape[1] + 1))
    cum[1:, 1:] = s.cumsum(0).cumsum(1)
    o = np.arange(0, s.shape[0] - patch + 1, step)
    p = np.arange(0, s.shape[1] - patch + 1, step)
    A = (
        cum[o[:, None] + patch, p[None, :] + patch]
        - cum[o[:, None], p[None, :] + patch]
        - cum[o[:, None] + patch, p[None, :]]
        + cum[o[:, None], p[None, :]]
    )
    return A / (patch * patch), o, p


def train_model(torch, dev, vols, labs, sups, tabs, tiles, win, code, seed, fold, steps, batch, mean, std, log):
    """One model for one input, seed and held-out window (patches outside the window's guard only)."""
    torch.manual_seed(seed * 1000 + code * 10 + fold)
    rng = np.random.default_rng(seed * 1000 + code * 10 + fold)
    usable = [t for t, (_, _, _, ok) in enumerate(tabs) if ok.any()]
    allowed = {t: np.argwhere(tabs[t][3]) for t in usable}
    net = build_net(torch).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)

    def lr_at(k):
        if k < WARM:
            return LR * (k + 1) / WARM
        return LR * 0.5 * (1 + np.cos(np.pi * (k - WARM) / max(1, steps - WARM)))

    use_amp = dev.type == "cuda"
    trace = []
    t0 = time.time()
    for k in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(k)
        xs, ys, ms = [], [], []
        for _ in range(batch):
            for _r in range(REDRAWS + 1):
                t = usable[int(rng.integers(len(usable)))]
                A, o, p, _ok = tabs[t]
                i, j = allowed[t][int(rng.integers(len(allowed[t])))]
                if A[i, j] >= MIN_PATCH_SUP:
                    break
            y0, x0 = int(o[i]), int(p[j])
            if touches(tiles[t]["y0"] + y0, tiles[t]["x0"] + x0, PATCH, win):
                raise SystemExit("refused: a fold %d patch touches the guard of its window" % fold)
            xs.append(vols[t][:, y0 : y0 + PATCH, x0 : x0 + PATCH])
            ys.append(labs[t][y0 : y0 + PATCH, x0 : x0 + PATCH])
            ms.append(sups[t][y0 : y0 + PATCH, x0 : x0 + PATCH])
        x = torch.stack(xs).unsqueeze(1).float() / 255.0
        y = torch.stack(ys).float()
        m = torch.stack(ms).float()
        rot = rng.integers(4, size=batch)
        flp = rng.integers(2, size=batch)
        x = degrade(torch, x, code)
        x = (x - mean) / std
        xa, ya, ma = [], [], []
        for b in range(batch):
            xb, yb, mb = x[b], y[b], m[b]
            if flp[b]:
                xb, yb, mb = xb.flip(-1), yb.flip(-1), mb.flip(-1)
            r = int(rot[b])
            xa.append(torch.rot90(xb, r, (-2, -1)))
            ya.append(torch.rot90(yb, r, (-2, -1)))
            ma.append(torch.rot90(mb, r, (-2, -1)))
        x, y, m = torch.stack(xa), torch.stack(ya), torch.stack(ma)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=use_amp):
            out = net(x)
        loss_px = torch.nn.functional.binary_cross_entropy_with_logits(out.float(), y, reduction="none")
        loss = (loss_px * m).sum() / m.sum().clamp_min(1.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if k % 250 == 0 or k == steps - 1:
            trace.append([k, round(float(loss.detach()), 5)])
            log(
                "code=%d seed=%d fold=%d step %d/%d loss %.4f (%.0f s)"
                % (code, seed, fold, k, steps, float(loss.detach()), time.time() - t0)
            )
    net.eval()
    return net, trace, time.time() - t0, len(usable)


def map_window(torch, dev, net, vol_u8, code, mean, std, reverse=False):
    """Probability map (n, n) of one window, forward or depth-reversed input."""
    n = vol_u8.shape[-1]
    assert n % CORE == 0, n
    x = torch.from_numpy(np.ascontiguousarray(vol_u8)).to(dev).float().unsqueeze(0).unsqueeze(0) / 255.0
    x = degrade(torch, x, code)
    x = (x - mean) / std
    if reverse:
        x = x.flip(2)
    x = torch.nn.functional.pad(x[0], (MARGIN, MARGIN, MARGIN, MARGIN), mode="reflect").unsqueeze(0)
    out = np.zeros((n, n), np.float32)
    use_amp = dev.type == "cuda"
    k = n // CORE
    with torch.no_grad():
        for i in range(k):
            for j in range(k):
                xi = x[..., i * CORE : i * CORE + CORE + 2 * MARGIN, j * CORE : j * CORE + CORE + 2 * MARGIN]
                with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=use_amp):
                    o = net(xi)
                p = torch.sigmoid(o.float())[0, MARGIN : MARGIN + CORE, MARGIN : MARGIN + CORE]
                out[i * CORE : (i + 1) * CORE, j * CORE : (j + 1) * CORE] = p.cpu().numpy()
    return out


def superblocks(lab, sup, rendered, sb=SUPERBLOCK):
    """Per scoring unit: kept (fully supervised, rendered and pure) and ink, (H/sb, W/sb)."""
    H, W = lab.shape
    h, w = H // sb, W // sb

    def blocks(a):
        return np.asarray(a[: h * sb, : w * sb]).reshape(h, sb, w, sb)

    ink = blocks(lab > 0)
    kept = blocks(sup > 0).all(axis=(1, 3)) & blocks(rendered).all(axis=(1, 3))
    allink = ink.all(axis=(1, 3))
    kept &= allink | ~ink.any(axis=(1, 3))
    return kept, allink & kept


def unit_mean(prob, sb=SUPERBLOCK):
    H, W = prob.shape
    return prob[: H // sb * sb, : W // sb * sb].reshape(H // sb, sb, W // sb, sb).mean(axis=(1, 3))


def load_checked(path, sha):
    got = sha_file(path)
    if got != sha:
        raise SystemExit("refused: stage record sha256 %s, expected %s" % (got, sha))
    return json.load(open(path)), got


def run_train(
    data,
    input_name,
    seeds,
    windows_path,
    tiles_path,
    stages_path,
    stages_sha256,
    out_dir,
    name="segment",
    smoke=False,
    steps=None,
    batch=None,
):
    """Train and map one input on its seeds: for each seed and held-out window, one model on the training tiles'
    patches outside that window's guard, then the window's forward and depth-reversed probability maps averaged to
    scoring units. Refuses when the input differs from the stage record's. Writes NAME-INPUT-sSEED.npz with a JSON
    record beside it."""
    import torch

    st, st_sha = load_checked(stages_path, stages_sha256)
    if bool(st.get("smoke")) != bool(smoke):
        raise SystemExit("refused: stage record smoke %s, run smoke %s" % (st.get("smoke"), smoke))
    if input_name not in st["plan"]["train"]:
        raise SystemExit("refused: %s is not in the plan (%s)" % (input_name, st["plan"]["skip"].get(input_name)))
    for k, path in (("windows", windows_path), ("tiles", tiles_path)):
        if sha_file(path) != st[k + "_file_sha256"]:
            raise SystemExit("refused: %s file differs from the stage record's" % k)
    t0 = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    os.makedirs(out_dir, exist_ok=True)
    site = st["site"]
    code = INPUT_CODE[input_name]
    sfx = "-smoke" if smoke else ""
    log = _logger(os.path.join(out_dir, "%s-%s-s%s%s.log" % (name, input_name, "".join(map(str, seeds)), sfx)))
    wins, tiles, tn, n, folds = load_layout(windows_path, tiles_path, smoke)
    steps = steps or (SMOKE_STEPS if smoke else STEPS)
    batch = batch or (SMOKE_BATCH if smoke else BATCH)
    total = 3 + len(folds) * len(seeds)
    log(
        "device %s torch %s; %s input %s (code %d) seeds %s; %d tiles, folds %s, %d steps"
        % (dev, torch.__version__, name, input_name, code, seeds, len(tiles), folds, steps)
    )
    with ThreadPoolExecutor(4) as ex:
        got = list(ex.map(lambda t: data.read_box(site, t["y0"], t["x0"], tn), tiles))
    fine_t = [g[0] for g in got]
    labs = [torch.from_numpy((g[1] > 0).astype(np.uint8)).to(dev) for g in got]
    raw = [g[2] for g in got]
    sups = [torch.from_numpy((g[2] > 0).astype(np.uint8)).to(dev) for g in got]
    del got
    evals = []
    for j in folds:
        w = wins[j]
        v, lab, sup = data.read_box(site, w["y0"], w["x0"], n)
        kept, ink = superblocks(lab, sup, v.mean(axis=0) > 0)
        evals.append({"fold": j, "vol": v, "kept": kept, "ink": ink, "sha256_vol": sha_array(v)})
    log("PROGRESS 1/%d fine data read (%.0f s)" % (total, time.time() - t0))
    real = villa = None
    if input_name in ("real", "real_registered"):
        real = real_areas(site, data, tiles, wins, evals, fine_t, tn, n, log)[0]
    if input_name == "villa_input":
        villa = villa_inputs(data, site, tiles, wins, folds, tn, n)
    vt, vw = area_inputs(input_name, st, fine_t, evals, real, villa)
    del fine_t, real, villa
    win_sha = [sha_array(x) for x in vw]
    mean, std = np_norm(vt)
    exp = st["inputs"][input_name]
    check = {
        "window_sha256_equal": win_sha == exp["window_sha256"],
        "norm_diff": max(abs(mean - exp["norm"][0]), abs(std - exp["norm"][1])),
    }
    check["verdict"] = "PASS" if check["window_sha256_equal"] and check["norm_diff"] <= NORM_TOL else "FAIL"
    log("input check %s %s" % (check["verdict"], check))
    if check["verdict"] != "PASS":
        raise SystemExit("refused: the input differs from the stage record's")
    vols = [torch.from_numpy(v).to(dev) for v in vt]
    del vt
    log("PROGRESS 3/%d inputs ready (%.0f s)" % (total, time.time() - t0))
    done = 3
    base_rec = {
        "name": name,
        "input": input_name,
        "code": code,
        "smoke": bool(smoke),
        "steps": steps,
        "batch": batch,
        "patch": PATCH,
        "fold_guard_px": FOLD_GUARD,
        "torch": torch.__version__,
        "device": str(dev),
        "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else None,
        "code_sha256": {"resolution_gap.py": sha_file(__file__)},
        "stages_sha256": st_sha,
        "input_check": check,
        "norm": [mean, std],
        "windows": [
            {
                "fold": e["fold"],
                "sha256_vol": e["sha256_vol"],
                "kept": int(e["kept"].sum()),
                "ink": int(e["ink"].sum()),
                "sha256_input": win_sha[q],
            }
            for q, e in enumerate(evals)
        ],
    }
    paths = []
    for sd in seeds:
        fwd, rev = [], []
        rec = dict(base_rec, seed=sd, folds={})
        for q, e in enumerate(evals):
            j = e["fold"]
            tabs = []
            for t, s_ in zip(tiles, raw):
                A, oo, pp = patch_sup_table(s_, PATCH)
                tabs.append((A, oo, pp, allowed_origins(t, oo, pp, wins[j])))
            net, trace, secs, nus = train_model(
                torch, dev, vols, labs, sups, tabs, tiles, wins[j], code, sd, j, steps, batch, mean, std, log
            )
            rec["folds"][str(j)] = {"loss_trace": trace, "seconds": round(secs, 2), "usable_tiles": nus}
            fwd.append(unit_mean(map_window(torch, dev, net, vw[q], code, mean, std, False)))
            rev.append(unit_mean(map_window(torch, dev, net, vw[q], code, mean, std, True)))
            del net
            done += 1
            log(
                "PROGRESS %d/%d seed %d %s fold %d trained and mapped (%.0f s)"
                % (done, total, sd, input_name, j, time.time() - t0)
            )
        stem = os.path.join(out_dir, "%s-%s-s%d%s" % (name, input_name, sd, sfx))
        np.savez_compressed(
            stem + ".npz",
            kept=np.stack([e["kept"] for e in evals]),
            ink=np.stack([e["ink"] for e in evals]),
            folds=np.array(folds),
            forward=np.stack(fwd).astype(np.float32),
            reversed=np.stack(rev).astype(np.float32),
        )
        rec["seconds"] = round(time.time() - t0, 1)
        rec["npz_sha256"] = sha_file(stem + ".npz")
        json.dump(rec, open(stem + ".json", "w"), indent=1)
        paths.append(stem + ".npz")
        log("seed %d %s done in %.0f s; wrote %s.npz" % (sd, input_name, time.time() - t0, stem))
    return paths


# ---------------------------------------------------------------- scoring


def auc_exact(scores, y):
    """Mann-Whitney AUC with ties averaged."""
    s = np.asarray(scores, np.float64)
    y = np.asarray(y, bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ss = s[order]
    ranks = np.empty(len(s), np.float64)
    starts = np.flatnonzero(np.r_[True, ss[1:] != ss[:-1]])
    ends = np.r_[starts[1:], len(ss)]
    avg = (starts + ends + 1) / 2.0
    ranks[order] = np.repeat(avg, ends - starts)
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def boot_weights(n_tiles, draws=DRAWS, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    W = np.zeros((draws, n_tiles), np.float64)
    for d in range(draws):
        W[d] = np.bincount(rng.integers(0, n_tiles, n_tiles), minlength=n_tiles)
    return W


def weighted_auc_draws(scores, y, w):
    """AUC per draw with per-sample weights w (draws, n); ties are ignored (continuous scores)."""
    order = np.argsort(scores, kind="mergesort")
    yy = y[order]
    ww = w[:, order]
    neg = np.where(yy, 0.0, ww)
    pos = np.where(yy, ww, 0.0)
    below = np.cumsum(neg, axis=1) - neg
    num = (pos * below).sum(axis=1)
    den = pos.sum(axis=1) * neg.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den


def load_maps(paths):
    """{(input, seed): {"kept", "ink", "forward", "reversed"}} from NAME-INPUT-sSEED.npz files, with their records."""
    maps, records, sums, names = {}, [], {}, set()
    for p in paths:
        m = MAP_RE.match(os.path.basename(p))
        if not m or m.group(2) not in INPUT_CODE:
            raise ValueError("not a map file name NAME-INPUT-sSEED.npz: %s" % p)
        names.add(m.group(1))
        z = np.load(p)
        maps[(m.group(2), int(m.group(3)))] = {k: z[k] for k in z.files}
        sums[os.path.basename(p)] = sha_file(p)
        rp = p[: -len(".npz")] + ".json"
        if os.path.isfile(rp):
            records.append(json.load(open(rp)))
    if len(names) != 1:
        raise ValueError("map files of more than one segment name: %s" % sorted(names))
    return maps, records, sums, names.pop()


def _units(maps):
    """kept and ink, checked identical across the map files."""
    ref = None
    for _, d in sorted(maps.items()):
        if ref is None:
            ref = (d["kept"], d["ink"])
        elif not (np.array_equal(ref[0], d["kept"]) and np.array_equal(ref[1], d["ink"])):
            raise ValueError("scoring units differ between map files")
    if ref is None:
        raise ValueError("no map files")
    return ref


def _window_boot(kept, draws, seed):
    bw = []
    for j in range(kept.shape[0]):
        nsb = kept.shape[2]
        tps = -(-nsb // BOOT_TILE)
        rr, cc = np.nonzero(kept[j])
        tile = (rr // BOOT_TILE) * tps + (cc // BOOT_TILE)
        W = boot_weights(tps * tps, draws, seed + j)
        bw.append(W[:, tile])
    return bw


def _map_auc(M, kept, ink, bw=None):
    """Mean over windows of the exact AUC (and, with bw, its bootstrap draws)."""
    aucs, dr = [], []
    for j in range(kept.shape[0]):
        k = kept[j]
        s, y = M[j][k].astype(np.float64), ink[j][k]
        aucs.append(auc_exact(s, y))
        if bw is not None:
            dr.append(weighted_auc_draws(s, y, bw[j]))
    return (
        float(np.nanmean(aucs)),
        [float(a) for a in aucs],
        (np.nanmean(np.stack(dr), axis=0) if bw is not None else None),
    )


def _null_means(M, kept, donors):
    """Mean over windows of the map's AUC against each donor's labels (units kept in both)."""
    out = []
    for dk, di in donors:
        vals = []
        for j in range(kept.shape[0]):
            both = kept[j] & dk
            vals.append(auc_exact(M[j][both].astype(np.float64), di[both]))
        out.append(float(np.nanmean(vals)))
    return out


def input_rows(maps, names, seeds, kept, ink, donors, bw):
    """Per input and seed: AUC, reversed-depth AUC, the reversed-depth check (forward minus reversed scores) and the
    donor-label null; inputs with no map on those seeds are left out. Returns (rows, per-seed AUC, bootstrap)."""
    rows, per_seed, bootS = {}, {}, {}
    for name in names:
        by = {sd: maps[(name, sd)] for sd in seeds if (name, sd) in maps}
        if not by:
            continue
        aucs, dr, fmr, fmr_dr, raucs = [], [], [], [], []
        row = {"seeds": {}}
        for sd, d in sorted(by.items()):
            fwd, rev = d["forward"], d["reversed"]
            a, wa, dd = _map_auc(fwd, kept, ink, bw)
            ar, _, _ = _map_auc(rev, kept, ink)
            ad, _, cdd = _map_auc(fwd.astype(np.float64) - rev.astype(np.float64), kept, ink, bw)
            row["seeds"][str(sd)] = {
                "auc": a,
                "window_auc": [round(v, 4) for v in wa],
                "reversed_auc": ar,
                "forward_minus_reversed_auc": ad,
            }
            aucs.append(a)
            dr.append(dd)
            raucs.append(ar)
            fmr.append(ad)
            fmr_dr.append(cdd)
            per_seed[(name, sd)] = a
            bootS[(name, sd)] = dd
        dm = np.mean(np.stack(dr), axis=0)
        cm = np.mean(np.stack(fmr_dr), axis=0)
        row.update(
            {
                "auc": float(np.mean(aucs)),
                "n_seeds": len(aucs),
                "seed_list": sorted(by),
                "seed_sd": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else None,
                "ci95": [float(np.nanpercentile(dm, 2.5)), float(np.nanpercentile(dm, 97.5))],
                "reversed_auc": float(np.mean(raucs)),
                "reversed_depth": {
                    "auc": float(np.mean(fmr)),
                    "ci95": [float(np.nanpercentile(cm, 2.5)), float(np.nanpercentile(cm, 97.5))],
                },
            }
        )
        rd = row["reversed_depth"]
        rd["verdict"] = "PASS" if rd["auc"] >= REVERSED_MIN and rd["ci95"][0] > 0.5 else "FAIL"
        nulls = [_null_means(by[sd]["forward"], kept, donors) for sd in sorted(by)]
        nm = list(np.mean(np.array(nulls), axis=0))
        row["donor_null"] = {
            "null_mean": float(np.mean(nm)),
            "null_max": float(np.max(nm)),
            "verdict": "PASS" if row["auc"] > max(nm) else "FAIL",
        }
        rows[name] = row
    return rows, per_seed, bootS


def t_quantile(p, df):
    """The p quantile (p > 0.5) of Student's t with df degrees of freedom, by bisection on the CDF (Simpson's rule)."""
    if df is None or not df > 0:
        raise ValueError("refused: t quantile needs df > 0, got %r" % (df,))
    c = math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2)) / math.sqrt(df * math.pi)

    def cdf(x, n=4000):
        h = x / n
        s = 0.0
        for i in range(n + 1):
            f = c * (1 + (i * h) ** 2 / df) ** (-(df + 1) / 2)
            s += f * (1 if i in (0, n) else (4 if i % 2 else 2))
        return 0.5 + s * h / 3

    lo, hi = 0.0, 1.0
    while cdf(hi) < p:
        hi *= 2
        if hi > 1e6:
            raise ValueError("refused: t quantile out of range")
    for _ in range(60):
        mid = (lo + hi) / 2
        if cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def seed_term(weights, seed_values, level=0.975):
    """The training-noise half-width t(level, df) x sqrt(sum w^2 var / n) of a linear combination of inputs' seed
    means, df by Welch-Satterthwaite. None when an input has fewer than two seeds."""
    parts = []
    for w, vals in zip(weights, seed_values):
        v = np.asarray(vals, np.float64)
        if v.size < 2:
            return None
        parts.append((w * w * float(np.var(v, ddof=1)) / v.size, v.size))
    var = sum(p for p, _ in parts)
    if var <= 0:
        return {"h": 0.0, "df": None, "t": None, "sd": 0.0}
    den = sum(p * p / (n - 1) for p, n in parts if p > 0)
    df = var * var / den
    t = t_quantile(level, df)
    return {"h": t * var**0.5, "df": df, "t": t, "sd": var**0.5}


def st_interval(value, tile_ci, term):
    """The tile interval widened by the seed term's half-width in quadrature on each side."""
    if term is None or tile_ci is None:
        return None
    lo, hi = tile_ci
    h = term["h"]
    return [float(value - ((value - lo) ** 2 + h * h) ** 0.5), float(value + ((hi - value) ** 2 + h * h) ** 0.5)]


def _combo(per_seed, bootS, names, weights, S, label):
    """Value, tile interval, seed term and the seed-and-tile interval of sum w mean_S(AUC) on the seeds S."""
    S = sorted(S)
    val = float(sum(w * np.mean([per_seed[(a, sd)] for sd in S]) for a, w in zip(names, weights)))
    dd = sum(w * np.mean(np.stack([bootS[(a, sd)] for sd in S]), axis=0) for a, w in zip(names, weights))
    tci = [float(np.nanpercentile(dd, 2.5)), float(np.nanpercentile(dd, 97.5))]
    term = seed_term(weights, [[per_seed[(a, sd)] for sd in S] for a in names])
    return {
        "name": label,
        "seeds": S,
        "value": val,
        "ci95_tile": tci,
        "seed_term": term,
        "ci95_seeds_tiles": st_interval(val, tci, term),
    }


def _diff(per_seed, bootS, x, y, S, label):
    S = sorted(sd for sd in S if (x, sd) in per_seed and (y, sd) in per_seed)
    if not S:
        return None
    r = _combo(per_seed, bootS, (x, y), (1.0, -1.0), S, label)
    sv = {str(sd): per_seed[(x, sd)] - per_seed[(y, sd)] for sd in S}
    r["seed_values"] = sv
    r["seed_signs"] = (
        "PASS" if sv and all(np.sign(s) == np.sign(r["value"]) and s != 0 for s in sv.values()) else "FAIL"
    )
    return r


def combo(per_seed, bootS, names, weights, S, label):
    """_combo plus per-seed values and the seed sign check; None when an input lacks a seed of S."""
    S = sorted(S)
    if any((a, sd) not in per_seed for a in names for sd in S):
        return None
    r = _combo(per_seed, bootS, names, weights, S, label)
    sv = {str(sd): float(sum(w * per_seed[(a, sd)] for a, w in zip(names, weights))) for sd in S}
    r["seed_values"] = sv
    r["seed_signs"] = (
        "PASS" if sv and all(np.sign(v) == np.sign(r["value"]) and v != 0 for v in sv.values()) else "FAIL"
    )
    return r


def ols_slope(x, y):
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    xm = x - x.mean()
    return float((xm * (y - y.mean())).sum() / (xm * xm).sum())


def isotonic_dec(y):
    """Non-increasing least squares fit of y (pool adjacent violators, equal weights)."""
    blocks = []
    for v in y:
        blocks.append([float(v), 1])
        while len(blocks) > 1 and blocks[-2][0] < blocks[-1][0]:
            v2, n2 = blocks.pop()
            v1, n1 = blocks.pop()
            blocks.append([(v1 * n1 + v2 * n2) / (n1 + n2), n1 + n2])
    out = []
    for v, n in blocks:
        out += [v] * n
    return out


def _excl(ci):
    return ci is not None and (ci[0] > 0 or ci[1] < 0)


def part_verdict(d, rows, names, need, seeds):
    """One comparison: VOID when a needed check is not PASS, an input lacks a seed, or there is no difference; CLEAR
    when both inputs pass the donor-label null and the reversed-depth check, every seed agrees in sign and the
    seed-and-tile interval excludes 0; FLAT when those checks pass and the interval lies in [-FLAT_BAND, FLAT_BAND];
    NOT CLEAR otherwise."""
    if d is None or any(v.get("verdict") not in ("PASS",) for v in need):
        return "VOID"
    if any(a not in rows or rows[a]["seed_list"] != sorted(seeds) for a in names):
        return "VOID"
    gates = all(
        rows[a]["donor_null"]["verdict"] == "PASS" and rows[a]["reversed_depth"]["verdict"] == "PASS" for a in names
    )
    ci = d["ci95_seeds_tiles"]
    if ci is None:
        return "NOT CLEAR"
    if gates and d["seed_signs"] == "PASS" and (ci[0] > 0 or ci[1] < 0):
        return "CLEAR"
    if gates and ci[0] >= -FLAT_BAND and ci[1] <= FLAT_BAND:
        return "FLAT"
    return "NOT CLEAR"


def dose_record(rows, ps, bs, S, s_site):
    """AUC against the blur dose: the box average, box_blur and its multiples at mult x the fitted sigma, the slope
    on common seeds with its interval, monotone (no clear rise between neighbors), the donor-null slope check, the
    per-seed slopes and the drop from box_average to the largest dose."""
    pts = [(a, m) for a, m in DOSE if a in rows and rows[a]["seed_list"] == sorted(S)]
    out = {
        "points": [
            {
                "input": a,
                "mult": m,
                "sigma": m * s_site,
                "auc": rows[a]["auc"],
                "null_mean": rows[a]["donor_null"]["null_mean"],
            }
            for a, m in pts
        ],
        "complete": len(pts) == len(DOSE),
        "blur_sigma": s_site,
    }
    if len(pts) < 3 or not s_site:
        return out
    names = [a for a, _ in pts]
    xs = [m * s_site for _, m in pts]
    xm = np.asarray(xs) - np.mean(xs)
    wsl = list(xm / (xm * xm).sum())
    slope = _combo(ps, bs, names, wsl, S, "slope of AUC on sigma (coarse voxels)")
    sb = {str(sd): ols_slope(xs, [ps[(a, sd)] for a in names]) for sd in sorted(S)}
    rises = []
    for j in range(len(names) - 1):
        r = _combo(ps, bs, (names[j + 1], names[j]), (1.0, -1.0), S, "%s - %s" % (names[j + 1], names[j]))
        rises.append(r)
    mono = "FAIL" if any(r["ci95_seeds_tiles"] is None or r["ci95_seeds_tiles"][0] > 0 for r in rises) else "PASS"
    bnull = ols_slope(xs, [rows[a]["donor_null"]["null_mean"] for a in names])
    nd = "PASS" if abs(bnull) <= NULL_DOSE_SHARE * abs(slope["value"]) else "FAIL"
    out.update(
        {
            "slope": slope,
            "seed_slopes": sb,
            "seed_signs": (
                "PASS" if sb and all(np.sign(s) == np.sign(slope["value"]) and s != 0 for s in sb.values()) else "FAIL"
            ),
            "monotone": {"verdict": mono, "rises": rises},
            "null_dose": {"verdict": nd, "null_slope": bnull, "auc_slope": slope["value"], "share": NULL_DOSE_SHARE},
            "drop": _combo(ps, bs, (names[0], names[-1]), (1.0, -1.0), S, "%s - %s" % (names[0], names[-1])),
            "isotonic": isotonic_dec([rows[a]["auc"] for a in names]),
        }
    )
    return out


def dose_verdict(dose, rows, ok, need):
    if not ok or any(v.get("verdict") != "PASS" for v in need) or not dose.get("complete") or "slope" not in dose:
        return "VOID"
    base = dose["monotone"]["verdict"] == "PASS" and dose["null_dose"]["verdict"] == "PASS"
    anchors = all(rows[a]["donor_null"]["verdict"] == "PASS" for a in ("box_average", "box_blur"))
    c2 = all(
        rows[p["input"]]["reversed_depth"]["verdict"] == "PASS"
        for p in dose["points"]
        if rows[p["input"]]["donor_null"]["verdict"] == "PASS"
    )
    ci = dose["slope"]["ci95_seeds_tiles"]
    if base and anchors and c2 and dose["seed_signs"] == "PASS" and ci is not None and (ci[0] > 0 or ci[1] < 0):
        return "CLEAR"
    dc = dose["drop"]["ci95_seeds_tiles"]
    if base and dc is not None and dc[0] >= -FLAT_BAND and dc[1] <= FLAT_BAND:
        return "FLAT"
    return "NOT CLEAR"


def input_check_verdict(stages, records):
    """Every map record's window input sha256 values and normalization equal the stage record's for its input."""
    inp = (stages or {}).get("inputs") or {}
    out = {}
    for r in records or []:
        name = r.get("input")
        why = out.setdefault(name, {"why": [], "checked": 0})["why"]
        exp = inp.get(name)
        out[name]["checked"] += 1
        if not exp:
            why.append("no stage input record for %s" % name)
            continue
        got = [w.get("sha256_input") for w in r.get("windows", [])]
        if got != list(exp.get("window_sha256") or []):
            why.append("seed %s window input sha256 differs" % r.get("seed"))
        gn, en = r.get("norm"), exp.get("norm")
        if gn is None or en is None or max(abs(float(a) - float(b)) for a, b in zip(gn, en)) > NORM_TOL:
            why.append("seed %s normalization differs" % r.get("seed"))
    for v in out.values():
        v["verdict"] = "FAIL" if v["why"] else "PASS"
    return out


def maps_verdict(map_sha256, records, stages_sha256, name, smoke=False):
    """Ties the scored maps to their records: each record trained the full steps (smoke flag as expected), names the
    stage record scored with, has its map file with the same sha256, and each map file has exactly one record."""
    want_steps = SMOKE_STEPS if smoke else STEPS
    names = {os.path.basename(k): v for k, v in (map_sha256 or {}).items()}
    claimed, why = {}, []
    for r in records or []:
        inp, sd = r.get("input"), r.get("seed")
        tag = "record %s seed %s" % (inp, sd)
        if bool(r.get("smoke")) != bool(smoke):
            why.append("%s: smoke %s, expected %s" % (tag, r.get("smoke"), bool(smoke)))
        if r.get("steps") != want_steps:
            why.append("%s: %s steps, expected %d" % (tag, r.get("steps"), want_steps))
        if not stages_sha256 or r.get("stages_sha256") != stages_sha256:
            why.append("%s: stage record sha256 differs from the --stages file's" % tag)
        if inp not in INPUT_CODE or r.get("code") != INPUT_CODE[inp]:
            why.append("%s: code %s is not the input's" % (tag, r.get("code")))
            continue
        fname = "%s-%s-s%s%s.npz" % (name, inp, sd, "-smoke" if smoke else "")
        claimed.setdefault(fname, []).append(tag)
        if fname not in names:
            why.append("%s: no map file %s" % (tag, fname))
        elif names[fname] != r.get("npz_sha256"):
            why.append("%s: %s sha256 differs from the record's npz_sha256" % (tag, fname))
    for fname in sorted(set(names) | set(claimed)):
        k = len(claimed.get(fname, []))
        if k != 1 and (fname in names or k > 1):
            why.append("%s: %d records (exactly one needed)" % (fname, k))
    return {
        "verdict": "FAIL" if why else "PASS",
        "smoke": bool(smoke),
        "steps": want_steps,
        "files": len(names),
        "records": len(records or []),
        "stages_sha256": stages_sha256,
        "why": why[:20],
        "n_why": len(why),
    }


def villa_contrasts(per_seed, bootS, S):
    """Which simulated input sits closer to real_registered: villa_input or box_blur_directional.
    |V - R| > |K - R| exactly when (V - K) and (V + K - 2R) share a sign, so both contrasts are tested."""
    out = {
        "villa_minus_directional": combo(
            per_seed,
            bootS,
            ("villa_input", "box_blur_directional"),
            (1.0, -1.0),
            S,
            "villa_input - box_blur_directional",
        ),
        "villa_plus_directional_minus_2_real": combo(
            per_seed,
            bootS,
            ("villa_input", "box_blur_directional", "real_registered"),
            (1.0, 1.0, -2.0),
            S,
            "villa_input + box_blur_directional - 2 real_registered",
        ),
        "villa_minus_box_average": combo(
            per_seed, bootS, ("villa_input", "box_average"), (1.0, -1.0), S, "villa_input - box_average (reported only)"
        ),
        "villa_minus_real": combo(
            per_seed,
            bootS,
            ("villa_input", "real_registered"),
            (1.0, -1.0),
            S,
            "villa_input - real_registered (reported only)",
        ),
    }
    if out["villa_minus_real"] and out["villa_minus_directional"]:
        tk = out["villa_minus_real"]["value"] - out["villa_minus_directional"]["value"]
        out["closer"] = {
            "value": abs(out["villa_minus_real"]["value"]) - abs(tk),
            "name": "|villa_input - real_registered| - |box_blur_directional - real_registered| (value only)",
        }
    return out


def closeness_verdict(c, rows, need, seeds):
    """VOID when a needed check is not PASS or an input lacks a seed; with the three inputs passing the donor-label
    null and the reversed-depth check, and both contrasts agreeing across seeds with intervals excluding 0: BLUR
    CLOSER when they share a sign, VILLA INPUT CLOSER when they differ. NOT CLEAR otherwise."""
    names = ("villa_input", "box_blur_directional", "real_registered")
    if any((v or {}).get("verdict") != "PASS" for v in need):
        return "VOID"
    if any(a not in rows or rows[a]["seed_list"] != sorted(seeds) for a in names):
        return "VOID"
    c5, c6 = c.get("villa_minus_directional"), c.get("villa_plus_directional_minus_2_real")
    if c5 is None or c6 is None:
        return "VOID"
    gates = all(
        rows[a]["donor_null"]["verdict"] == "PASS" and rows[a]["reversed_depth"]["verdict"] == "PASS" for a in names
    )
    if not (
        gates
        and c5["seed_signs"] == "PASS"
        and c6["seed_signs"] == "PASS"
        and _excl(c5["ci95_seeds_tiles"])
        and _excl(c6["ci95_seeds_tiles"])
    ):
        return "NOT CLEAR"
    return "BLUR CLOSER" if (c5["value"] > 0) == (c6["value"] > 0) else "VILLA INPUT CLOSER"


def score(
    maps,
    donor_units,
    stages,
    records,
    name,
    seeds=SEEDS,
    draws=DRAWS,
    seed=BOOT_SEED,
    rotations=(0, 1, 2, 3),
    map_sha256=None,
    stages_sha256=None,
    smoke=False,
):
    """Every input's AUC with its controls, and the comparisons on the common seeds with tile and seed-and-tile
    intervals:

    - blur_cost: box_average - box_blur
    - directional_blur_minus_real: box_blur_directional - real_registered
    - dose: the slope of AUC on the blur sigma
    - matched_minus_directional: box_blur_matched - box_blur_directional
    - closeness: whether villa_input or box_blur_directional sits closer to real_registered

    donor_units: (kept, ink) of another segment, rotated 4 ways, for the donor-label null."""
    kept, ink = _units(maps)
    dkept, dink = donor_units
    donors = [(np.rot90(dkept[j], r), np.rot90(dink[j], r)) for j in range(dkept.shape[0]) for r in rotations]
    if any(dk.shape != kept.shape[1:] for dk, _ in donors):
        raise ValueError("donor units %s differ in shape from the site's %s" % (dkept.shape[1:], kept.shape[1:]))
    bw = _window_boot(kept, draws, seed)
    S = tuple(sorted(seeds))
    rows, ps, bs = input_rows(maps, [a for a, _ in INPUTS], S, kept, ink, donors, bw)
    st = stages or {}
    win = {
        "verdict": "PASS" if kept.shape[0] >= MIN_WINDOWS else "FAIL",
        "windows": int(kept.shape[0]),
        "min": MIN_WINDOWS,
    }
    floor_v = {"auc_fine": rows["fine"]["auc"] if "fine" in rows else None, "min": FINE_FLOOR}
    floor_v["verdict"] = (
        "PASS"
        if (
            floor_v["auc_fine"] is not None
            and floor_v["auc_fine"] >= FINE_FLOOR
            and rows["fine"]["seed_list"] == list(S)
        )
        else "FAIL"
    )
    inp = input_check_verdict(st, records)
    maps_v = (
        maps_verdict(map_sha256, records, stages_sha256, name, smoke=smoke)
        if map_sha256 is not None
        else {"verdict": "NOT CHECKED", "why": ["no map file hashes given"]}
    )
    whole_v = registration_check(st.get("registration"))
    regsv = subvoxel_check(st.get("registration_subvoxel"))
    smv = blur_check(st.get("blur"))
    jg = (
        matched_blur_checks(st.get("matched_blur"))
        if st.get("matched_blur")
        else {"verdict": "FAIL", "why": ["no matched blur record"]}
    )
    kg = (
        directional_blur_checks(st.get("directional_blur"))
        if st.get("directional_blur")
        else {"verdict": "FAIL", "why": ["no directional blur record"]}
    )
    vc = villa_check(st) if "villa" in st else {"verdict": "FAIL", "why": ["no villa record"]}
    site = (st.get("site") or {}).get("check") or {"verdict": "FAIL", "why": ["no site record"]}
    common = [v for v in (win, floor_v, site) if v.get("verdict") != "PASS"]
    maps_bad = maps_v["verdict"] == "FAIL" or bool(smoke)
    good = not common and not maps_bad

    def inp_ok(*names):
        return {"verdict": "PASS" if all((inp.get(a) or {}).get("verdict") == "PASS" for a in names) else "FAIL"}

    d = {
        "blur_cost": _diff(ps, bs, "box_average", "box_blur", S, "box_average - box_blur"),
        "directional_blur_minus_real": _diff(
            ps, bs, "box_blur_directional", "real_registered", S, "box_blur_directional - real_registered"
        ),
        "matched_minus_directional": _diff(
            ps, bs, "box_blur_matched", "box_blur_directional", S, "box_blur_matched - box_blur_directional"
        ),
        "gray_levels_reported": _diff(
            ps, bs, "box_blur", "box_blur_matched", S, "box_blur - box_blur_matched (reported only)"
        ),
        "matched_minus_real_reported": _diff(
            ps, bs, "box_blur_matched", "real_registered", S, "box_blur_matched - real_registered (reported only)"
        ),
        "box_average_minus_real_reported": _diff(
            ps, bs, "box_average", "real", S, "box_average - real (reported only)"
        ),
        "registration_gain_reported": _diff(
            ps, bs, "real_registered", "real", S, "real_registered - real (reported only)"
        ),
    }
    if (
        d["matched_minus_directional"]
        and d["matched_minus_real_reported"]
        and d["matched_minus_real_reported"]["value"]
    ):
        d["share_of_matched_gap_reported"] = (
            d["matched_minus_directional"]["value"] / d["matched_minus_real_reported"]["value"]
        )
    s_site = (st.get("blur") or {}).get("sigma")
    dose = dose_record(rows, ps, bs, S, s_site)
    ok = {"verdict": "PASS" if good else "FAIL"}
    sm_or_none = {"verdict": "PASS" if smv["verdict"] in ("PASS", "NONE") else "FAIL"}
    blur_gap = bool(s_site)
    v = {
        "blur_cost": (
            part_verdict(
                d["blur_cost"], rows, ("box_average", "box_blur"), (ok, smv, inp_ok("box_average", "box_blur")), S
            )
            if blur_gap
            else "NO BLUR GAP"
        ),
        "directional_blur_minus_real": part_verdict(
            d["directional_blur_minus_real"],
            rows,
            ("box_blur_directional", "real_registered"),
            (ok, regsv, kg, inp_ok("box_blur_directional", "real_registered")),
            S,
        ),
        "dose": (dose_verdict(dose, rows, good, [inp_ok(*[a for a, _ in DOSE])]) if blur_gap else "NO BLUR GAP"),
        "matched_minus_directional": part_verdict(
            d["matched_minus_directional"],
            rows,
            ("box_blur_matched", "box_blur_directional"),
            (ok, sm_or_none, jg, kg, inp_ok("box_blur_matched", "box_blur_directional")),
            S,
        ),
    }
    c = villa_contrasts(ps, bs, S)
    need = [
        ok,
        regsv,
        kg,
        inp_ok("box_blur_directional", "real_registered"),
        vc,
        (inp.get("villa_input") or {"verdict": "FAIL", "why": ["no villa_input record"]}),
    ]
    v["closeness"] = closeness_verdict(c, rows, need, S)
    if not good:
        v = {k: "VOID" for k in v}
    controls = {
        "windows": win,
        "fine_floor": floor_v,
        "site": site,
        "maps": maps_v,
        "input_check": inp,
        "registration": whole_v,
        "registration_subvoxel": regsv,
        "blur_match": smv,
        "matched_blur": jg,
        "directional_blur": kg,
        "villa": vc,
        "directional_rule_reported": (
            directional_rule(st["directional_blur"], st["matched_blur"]["sigma"])
            if st.get("directional_blur") and (st.get("matched_blur") or {}).get("sigma")
            else None
        ),
    }
    return {
        "name": name,
        "seeds": list(S),
        "inputs": rows,
        "differences": d,
        "dose": dose,
        "villa_contrasts": c,
        "controls": controls,
        "plan": training_plan(st) if st else None,
        "verdict": v,
    }


# ---------------------------------------------------------------- command line


def cmd_windows(a):
    lab = open_source_array(a.labels, "3")
    sup = open_source_array(a.mask, "3")
    l3, s3 = np.asarray(lab[:]), np.asarray(sup[:])
    wins, n = pick_windows(l3, s3)
    tiles = pick_tiles(s3)
    os.makedirs(a.out, exist_ok=True)
    w = {"labels": a.labels, "level3_shape": list(l3.shape), "candidates": n, "windows": wins}
    t = {"mask": a.mask, "level3_shape": list(s3.shape), "tiles": tiles}
    json.dump(w, open(os.path.join(a.out, "%s-windows.json" % a.name), "w"), indent=1)
    json.dump(t, open(os.path.join(a.out, "%s-tiles.json" % a.name), "w"), indent=1)
    ok = len(wins) >= MIN_WINDOWS
    print(json.dumps({"windows": len(wins), "candidates": n, "tiles": len(tiles), "check": "PASS" if ok else "FAIL"}))
    return 0 if ok else 3


def _data(a, labels=False):
    return Data(
        a.fine,
        a.coarse,
        a.labels if labels else None,
        a.mask if labels else None,
        a.um_fine,
        a.um_coarse,
        a.canvas_scale,
    )


def cmd_stages(a):
    rec = run_stages(_data(a), a.windows, a.tiles, a.out, a.name, smoke=a.smoke, villa=not a.no_villa)
    print(json.dumps({"train": rec["plan"]["train"], "skip": rec["plan"]["skip"]}, indent=1))
    return 0


def cmd_train(a):
    run_train(
        _data(a, labels=True),
        a.input,
        a.seed,
        a.windows,
        a.tiles,
        a.stages,
        a.stages_sha256 or sha_file(a.stages),
        a.out,
        a.name,
        smoke=a.smoke,
    )
    return 0


def cmd_score(a):
    paths = sorted(p for pat in a.maps for p in glob.glob(pat))
    maps, records, sums, name = load_maps(paths)
    dz = np.load(a.donor)
    res = score(
        maps,
        (dz["kept"], dz["ink"]),
        json.load(open(a.stages)),
        records,
        name,
        seeds=tuple(a.seeds),
        draws=a.draws,
        map_sha256=sums,
        stages_sha256=sha_file(a.stages),
        smoke=a.smoke,
    )
    res["files"] = {"maps": paths, "stages": a.stages, "donor": a.donor}
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(
        json.dumps(
            {"verdict": res["verdict"], "auc": {k: round(r["auc"], 4) for k, r in res["inputs"].items()}}, indent=1
        )
    )
    return 0


def parse_args(argv: Sequence[str] | None = None):
    p = argparse.ArgumentParser(
        prog="resolution_gap", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("windows", help="pick the held-out windows and training tiles from the labels (level 3)")
    w.add_argument("--labels", required=True, help="ink labels OME-Zarr group (levels 0 and 3)")
    w.add_argument("--mask", required=True, help="supervision mask OME-Zarr group (levels 0 and 3)")
    w.add_argument("--out", required=True)
    w.add_argument("--name", default="segment")
    for nm in ("stages", "train"):
        s = sub.add_parser(nm, help="label-free stages (CT only)" if nm == "stages" else "train and map one input")
        s.add_argument("--fine", required=True, help="fine surface volume OME-Zarr group (levels 0 and 2)")
        s.add_argument("--coarse", required=True, help="real coarse surface volume OME-Zarr group (level 0)")
        s.add_argument("--windows", required=True)
        s.add_argument("--tiles", required=True)
        s.add_argument("--out", required=True)
        s.add_argument("--name", default="segment")
        s.add_argument("--um-fine", type=float, help="fine voxel size (default: from the volume's multiscales)")
        s.add_argument("--um-coarse", type=float, help="coarse voxel size (default: from the volume's multiscales)")
        s.add_argument(
            "--canvas-scale",
            type=float,
            nargs=2,
            metavar=("ROWS", "COLS"),
            help="coarse px per fine px (default: from the two volumes' canvas_size)",
        )
        s.add_argument("--smoke", action="store_true", help="3 tiles, one 504 px window, 3 steps")
        if nm == "stages":
            s.add_argument("--no-villa", action="store_true", help="skip villa's recipe (needs level 2)")
        else:
            s.add_argument("--input", required=True, choices=[x for x, _ in INPUTS])
            s.add_argument("--seed", type=int, nargs="+", required=True)
            s.add_argument("--labels", required=True)
            s.add_argument("--mask", required=True)
            s.add_argument("--stages", required=True)
            s.add_argument("--stages-sha256", help="refuse unless the stage record has this sha256")
    s = sub.add_parser("score", help="score the maps of one segment")
    s.add_argument("maps", nargs="+", help="NAME-INPUT-sSEED.npz files (records beside them)")
    s.add_argument("--stages", required=True)
    s.add_argument("--donor", required=True, help="a map file of another segment (its kept and ink units)")
    s.add_argument("--out", required=True)
    s.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    s.add_argument("--draws", type=int, default=DRAWS)
    s.add_argument("--smoke", action="store_true")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    a = parse_args(argv)
    return {"windows": cmd_windows, "stages": cmd_stages, "train": cmd_train, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
