"""Read one level of an OME-Zarr volume (local or anonymous https) chunk by chunk, and sample it trilinearly at
level-0 (x, y, z) points."""

import collections
import concurrent.futures
import threading

import numpy as np
import zarr


class VolumeError(ValueError):
    pass


def _multiscales(attrs):
    attrs = dict(attrs or {})
    ms = attrs.get("multiscales") or (attrs.get("ome") or {}).get("multiscales") or []
    return ms[0] if ms else None


class Volume:
    """An OME-Zarr image (a group with multiscales, numbered levels, or a bare array). `levels[k]["meta"]` holds
    the level's shape, chunks and dtype; `factors[k]` its voxels per level-0 voxel (z, y, x), from the scale
    transforms when every level has one, else from the shapes. `_chunk(level, (cz, cy, cx))` reads one chunk
    through a small LRU cache."""

    def __init__(self, location, workers=8, cache_chunks=192):
        self.location = str(location)
        try:
            node = zarr.open(self.location, mode="r")
        except Exception as exc:  # noqa: BLE001 - zarr raises several types for a missing or bad store
            raise VolumeError(f"cannot open {location}: {exc}") from exc
        self.levels = []
        if isinstance(node, zarr.Array):
            self.levels.append(self._level("", node, None))
        else:
            ms = _multiscales(node.attrs)
            if ms:
                for d in ms.get("datasets", []):
                    scale = next((t.get("scale") for t in d.get("coordinateTransformations") or []
                                  if t.get("type") == "scale"), None)
                    self.levels.append(self._level(str(d.get("path")), node[str(d.get("path"))], scale))
            else:
                for k in range(16):
                    if str(k) not in node:
                        break
                    self.levels.append(self._level(str(k), node[str(k)], None))
        if not self.levels:
            raise VolumeError(f"{location}: a zarr group with no levels")
        base = self.levels[0]
        self.factors = [self._factors(lv, base) for lv in self.levels]
        self.workers = max(1, int(workers))
        self._cache = collections.OrderedDict()
        self._cache_n = int(cache_chunks)
        self._lock = threading.Lock()

    @staticmethod
    def _level(name, arr, scale):
        return {"name": name, "array": arr, "scale": scale,
                "meta": {"shape": [int(s) for s in arr.shape], "chunks": [int(c) for c in arr.chunks],
                         "dtype": np.dtype(arr.dtype)}}

    @staticmethod
    def _factors(x, base):
        if x["scale"] and base["scale"] and len(x["scale"]) == len(base["scale"]):
            return [a / b if b else 1.0 for a, b in zip(base["scale"], x["scale"])]
        s0, s1 = base["meta"]["shape"], x["meta"]["shape"]
        return [b / a if a else 1.0 for a, b in zip(s0, s1)]

    def __str__(self):
        return self.location

    def shape(self, level=0):
        return list(self.levels[level]["meta"]["shape"])

    def _chunk(self, level, idx):
        key = (level, idx)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        lv = self.levels[level]
        ch = lv["meta"]["chunks"]
        sl = tuple(slice(i * c, (i + 1) * c) for i, c in zip(idx, ch))
        arr = np.asarray(lv["array"][sl])
        with self._lock:
            self._cache[key] = arr
            while len(self._cache) > self._cache_n:
                self._cache.popitem(last=False)
        return arr


class Sampler:
    """Reads one level of a zarr volume at level-0 (x, y, z) points. `shift` adds a constant level-0 vector to every
    point (the shifted control); `empty` reads 0 everywhere inside (the no-sheet control)."""

    def __init__(self, vol, level=0, shift=None, empty=False):
        if not 0 <= level < len(vol.levels):
            raise VolumeError(f"{vol}: no level {level} (it has {len(vol.levels)})")
        self.vol, self.level, self.empty = vol, int(level), bool(empty)
        self.shift = None if shift is None else np.asarray(shift, dtype=np.float32).reshape(3)
        meta = vol.levels[level]["meta"]
        if len(meta["shape"]) != 3:
            raise VolumeError(f"{vol} level {level}: expected a 3-D array, shape {meta['shape']}")
        self.shape_zyx = [int(s) for s in meta["shape"]]
        self.dtype = np.dtype(meta["dtype"])
        self.ds = 2 ** self.level

    def derive(self, shift=None, empty=False):
        return Sampler(self.vol, self.level, shift=shift, empty=empty)

    def _level(self, xyz):
        if self.shift is not None:
            xyz = xyz + self.shift.astype(xyz.dtype)
        return xyz / self.ds

    def trilinear(self, xyz):
        """Trilinear values (float64) over the corners inside the array, renormalized; nan with no corner inside."""
        pts = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
        fin = np.isfinite(pts).all(axis=1)
        if self.empty:
            return np.where(fin, 0.0, np.nan)
        lv = np.asarray(self._level(pts), dtype=np.float64)
        zyx = lv[:, ::-1].copy()
        zyx[~fin] = -1e9
        f = np.floor(zyx)
        i0 = f.astype(np.int64)
        dd = zyx - f
        shape = np.array(self.shape_zyx)
        corners, weights, rows = [], [], []
        for k in range(8):
            o = np.array([k & 1, (k >> 1) & 1, (k >> 2) & 1])
            idx = i0 + o
            inside = np.all((idx >= 0) & (idx < shape), axis=1)
            w = ((dd[:, 0] if o[0] else 1 - dd[:, 0]) * (dd[:, 1] if o[1] else 1 - dd[:, 1]) *
                 (dd[:, 2] if o[2] else 1 - dd[:, 2]))
            sel = np.nonzero(inside)[0]
            corners.append(idx[sel])
            weights.append(w[sel])
            rows.append(sel)
        rows = np.concatenate(rows)
        acc = np.zeros(len(pts))
        wsum = np.zeros(len(pts))
        if len(rows):
            vals = self.gather(np.concatenate(corners)).astype(np.float64)
            w = np.concatenate(weights)
            acc = np.bincount(rows, weights=w * vals, minlength=len(pts))
            wsum = np.bincount(rows, weights=w, minlength=len(pts))
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(wsum >= 1e-9, acc / np.where(wsum > 0, wsum, 1), np.nan)
        out[~fin] = np.nan
        return out

    def gather(self, zyx):
        """Values at integer voxels (N, 3) z, y, x inside the level, read chunk by chunk through the Volume's chunk
        cache, grouped by sorting a scalar chunk key; a missing chunk reads the array's fill value."""
        zyx = np.asarray(zyx, dtype=np.int64).reshape(-1, 3)
        meta = self.vol.levels[self.level]["meta"]
        ch = np.array([int(c) for c in meta["chunks"]])
        grid = -(-np.array(self.shape_zyx) // ch)
        out = np.zeros(len(zyx), self.dtype)
        if not len(zyx):
            return out
        c = zyx // ch
        key = (c[:, 0] * grid[1] + c[:, 1]) * grid[2] + c[:, 2]
        order = np.argsort(key, kind="stable")
        _, starts = np.unique(key[order], return_index=True)
        ends = np.append(starts[1:], len(order))
        keys3 = c[order[starts]]
        workers = max(1, getattr(self.vol, "workers", 8))
        group = max(1, workers * 4)
        with concurrent.futures.ThreadPoolExecutor(min(workers, len(keys3))) as pool:
            for g in range(0, len(keys3), group):
                part = keys3[g:g + group]
                arrs = list(pool.map(lambda k3: self.vol._chunk(self.level, tuple(int(v) for v in k3)), part))
                for j, arr in enumerate(arrs):
                    if arr is None:
                        continue
                    k = g + j
                    sel = order[starts[k]:ends[k]]
                    local = zyx[sel] - keys3[k] * ch
                    out[sel] = arr[local[:, 0], local[:, 1], local[:, 2]]
        return out

