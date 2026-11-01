"""Verified tifxyz patches: a cached index, memory-mapped coordinates and canonical GT windows.

Patch coordinates are level-0 ``x, y, z`` with ``-1`` marking invalid nodes; ``meta.json`` ``scale`` is grid
nodes per voxel, so one grid cell spans ``1 / scale`` voxels. Where the nodes are actually spaced more than
``SCALE_TOLERANCE`` away from that (some patch families state twice their real cell size), the index uses the
measured spacing (median distance between neighbouring valid nodes) instead. A window maps the canonical ``(u, v)`` lattice
(voxel units: ``u`` advance, ``v`` along the front) onto patch rows/columns for one of eight orientations
(advance along rows or columns, either sign, front either sign) and samples it bilinearly.
"""
from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import tifffile

INDEX_VERSION = 4
SCALE_TOLERANCE = .2  # relative difference between measured and stated cell size beyond which the measured one is used


@dataclass(frozen=True)
class PatchInfo:
    name: str
    rows: int
    cols: int
    cell: float  # voxels per grid cell
    valid: int  # valid nodes
    bbox: tuple  # (x0, y0, z0, x1, y1, z1) of the valid nodes
    root: str = ''  # directory holding the patch (several patch directories can be combined)
    stated_cell: float = 0.  # 1 / meta.json scale, when the measured spacing replaced it as ``cell``

    @property
    def path(self) -> Path:
        return Path(self.root)/self.name


@dataclass(frozen=True)
class GridSpec:
    """Canonical GT lattice of an episode; ``u = 0`` is the seed row, ``v = 0`` the first point."""
    u_lo: float
    v_lo: float
    res: float
    Gu: int
    Gv: int

    @classmethod
    def for_episode(cls, data, steps: int) -> 'GridSpec':
        u_lo = -data.history*data.step-data.gt_margin
        u_hi = steps*data.step+data.gt_margin
        v_lo = -data.gt_margin
        v_hi = (data.front_points-1)*data.point_spacing+data.gt_margin
        res = data.gt_resolution
        return cls(u_lo, v_lo, res, int(round((u_hi-u_lo)/res))+1, int(round((v_hi-v_lo)/res))+1)

    def index(self, u, v):
        return (np.asarray(u)-self.u_lo)/self.res, (np.asarray(v)-self.v_lo)/self.res

    @property
    def u_vals(self):
        return self.u_lo+self.res*np.arange(self.Gu)

    @property
    def v_vals(self):
        return self.v_lo+self.res*np.arange(self.Gv)


def _scan(path: Path) -> PatchInfo | None:
    try:
        meta = json.loads((path/'meta.json').read_text())
        xyz = [tifffile.memmap(path/f'{c}.tif') for c in 'xyz']
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    scale = float(meta.get('scale', [0])[0])
    if scale <= 0 or xyz[0].ndim != 2:
        return None
    ok = np.all([np.asarray(a) != -1 for a in xyz], 0)
    count = int(np.count_nonzero(ok))
    if not count:
        return None
    lo = [float(np.asarray(a)[ok].min()) for a in xyz]
    hi = [float(np.asarray(a)[ok].max()) for a in xyz]
    stated, cell = 1./scale, measured_cell(xyz, ok)
    if cell and abs(cell/stated-1) > SCALE_TOLERANCE:
        return PatchInfo(path.name, int(ok.shape[0]), int(ok.shape[1]), cell, count, tuple(lo+hi), str(path.parent),
                         stated)
    return PatchInfo(path.name, int(ok.shape[0]), int(ok.shape[1]), stated, count, tuple(lo+hi), str(path.parent))


def measured_cell(xyz, ok) -> float:
    """Median distance (voxels) between row and column neighbours that are both valid; 0 with too few of them."""
    p = np.stack([np.asarray(a, np.float32) for a in xyz], -1)
    d = np.concatenate([np.linalg.norm(p[1:]-p[:-1], axis=-1)[ok[1:] & ok[:-1]],
                        np.linalg.norm(p[:, 1:]-p[:, :-1], axis=-1)[ok[:, 1:] & ok[:, :-1]]])
    return float(np.median(d)) if len(d) >= 20 else 0.


def build_index(patches_dir, cache_dir: str | Path, workers: int = 8) -> list[PatchInfo]:
    """Index of one patch directory or a list of them (concatenated in order; each directory is cached on its
    own, keyed by its path and sorted entry names, so adding a directory does not rescan the others)."""
    if isinstance(patches_dir, (list, tuple)):
        return [info for d in patches_dir for info in build_index(d, cache_dir, workers)]
    patches_dir = Path(patches_dir)
    names = sorted(p.name for p in patches_dir.iterdir() if (p/'x.tif').is_file())
    key = hashlib.sha256(json.dumps([INDEX_VERSION, str(patches_dir.resolve()), names]).encode()).hexdigest()[:16]
    cache = Path(cache_dir)/f'patch_index_{key}.json'
    if cache.is_file():
        return [PatchInfo(**dict(row, bbox=tuple(row['bbox']))) for row in json.loads(cache.read_text())]
    with ThreadPoolExecutor(workers) as pool:
        index = [info for info in pool.map(_scan, (patches_dir/n for n in names)) if info is not None]
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix('.partial')
    tmp.write_text(json.dumps([asdict(i) for i in index]))
    tmp.replace(cache)
    return index


class PatchStore:
    """Lazily memory-mapped ``x, y, z`` of each patch, with a bounded number of open maps per process."""

    def __init__(self, max_open: int = 128):
        self.max_open = max_open
        self._open: OrderedDict[Path, tuple] = OrderedDict()

    def __getstate__(self):
        return dict(max_open=self.max_open)

    def __setstate__(self, state):
        self.__init__(state['max_open'])

    def arrays(self, info: PatchInfo):
        path = info.path
        hit = self._open.get(path)
        if hit is None:
            hit = tuple(tifffile.memmap(path/f'{c}.tif') for c in 'xyz')
            self._open[path] = hit
            while len(self._open) > self.max_open:
                self._open.popitem(last=False)
        else:
            self._open.move_to_end(path)
        return hit

    def window(self, info: PatchInfo, spec: GridSpec, axis: int, sign_u: int, sign_v: int,
               base_adv: float, base_front: float):
        """Canonical window (Gu, Gv, 3) float32 (invalid nodes zero) and validity (Gu, Gv)."""
        adv = base_adv+sign_u*spec.u_vals/info.cell
        front = base_front+sign_v*spec.v_vals/info.cell
        rows, cols = (adv[:, None], front[None, :]) if axis == 0 else (front[None, :], adv[:, None])
        rows, cols = np.broadcast_to(rows, (spec.Gu, spec.Gv)), np.broadcast_to(cols, (spec.Gu, spec.Gv))
        return self._sample(info, rows, cols)

    def window_rotated(self, info: PatchInfo, spec: GridSpec, origin_rc, u_rc, v_rc):
        """Canonical window on a rotated lattice: lattice point (u, v) is at patch grid position
        ``origin_rc + (u * u_rc + v * v_rc) / cell`` for unit (row, col) directions ``u_rc`` (advance) and ``v_rc``
        (front), e.g. a front along a fiber's course on the patch. Same outputs as ``window``."""
        u, v = spec.u_vals[:, None]/info.cell, spec.v_vals[None, :]/info.cell
        rows = origin_rc[0]+u*u_rc[0]+v*v_rc[0]
        cols = origin_rc[1]+u*u_rc[1]+v*v_rc[1]
        return self._sample(info, rows, cols)

    def _sample(self, info: PatchInfo, rows: np.ndarray, cols: np.ndarray):
        """Bilinear patch coordinates at grid positions (Gu, Gv) and their validity (all four corners valid)."""
        r0 = int(np.clip(np.floor(rows.min()), 0, info.rows-1))
        r1 = int(np.clip(np.floor(rows.max())+2, 1, info.rows))
        c0 = int(np.clip(np.floor(cols.min()), 0, info.cols-1))
        c1 = int(np.clip(np.floor(cols.max())+2, 1, info.cols))
        xyz = np.stack([np.asarray(a[r0:r1, c0:c1], np.float32) for a in self.arrays(info)], -1)
        ok = np.all(xyz != -1, -1) & np.all(np.isfinite(xyz), -1)
        h, w = ok.shape
        rr, cc = rows-r0, cols-c0
        inside = (rr >= 0) & (rr <= h-1) & (cc >= 0) & (cc <= w-1)
        i0 = np.clip(np.floor(rr).astype(np.int64), 0, max(h-2, 0))
        j0 = np.clip(np.floor(cc).astype(np.int64), 0, max(w-2, 0))
        i1, j1 = np.minimum(i0+1, h-1), np.minimum(j0+1, w-1)
        fr, fc = (rr-i0)[..., None], (cc-j0)[..., None]
        out = ((1-fr)*((1-fc)*xyz[i0, j0]+fc*xyz[i0, j1])+fr*((1-fc)*xyz[i1, j0]+fc*xyz[i1, j1])).astype(np.float32)
        valid = inside & ok[i0, j0] & ok[i0, j1] & ok[i1, j0] & ok[i1, j1]
        out[~valid] = 0
        return out, valid
