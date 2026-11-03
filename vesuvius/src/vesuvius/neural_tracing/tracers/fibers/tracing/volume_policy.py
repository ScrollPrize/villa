"""Shared full-volume coverage rules for neural and native beam tracers.

Distances are in the caller's trace grid. Width checks are opt-in.
"""
from collections import defaultdict
import numpy as np
from scipy.spatial import cKDTree

PARALLEL = 0.8

def width_compatible(candidate, reference, tolerance):
    """Tracer-relative width distance; unknown widths cannot reject a collision."""
    a, b = np.asarray(candidate), np.asarray(reference)
    if tolerance <= 0:
        return np.ones(np.broadcast_shapes(a.shape, b.shape), dtype=bool)
    known = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    return ~known | (np.abs(a-b) <= tolerance*b)

def seed_deferred(point, running, radius, width=np.nan, running_widths=None, tolerance=0):
    if not len(running):
        return False
    near = np.linalg.norm(np.asarray(running)-point, axis=1) < radius
    if running_widths is not None:
        near &= width_compatible(running_widths, width, tolerance)
    return bool(near.any())

def deferred_mask(points, running, radius, widths=None, running_widths=None, tolerance=0):
    """Vectorized seed_deferred; identical strict-radius and width predicates."""
    return deferred_owner(points,running,radius,widths,running_widths,tolerance)>=0

def deferred_owner(points, running, radius, widths=None, running_widths=None, tolerance=0):
    """Index of one still-running seed that blocks admission, or -1.

    Running seed coordinates and widths stay fixed; this witness can be cached
    until that pair finishes without re-querying the same blocked candidate.
    """
    if not len(running):
        return np.full(len(points),-1,dtype=int)
    near=np.linalg.norm(np.asarray(points)[:,None,:]-np.asarray(running)[None,:,:],axis=2)<radius
    if widths is not None and running_widths is not None:
        near &= width_compatible(np.asarray(running_widths)[None,:],np.asarray(widths)[:,None],tolerance)
    return np.where(near.any(axis=1),near.argmax(axis=1),-1)

def novel_parts(part, owned, radius, trim_run, overlap, min_length, widths=None, tolerance=0):
    """The neural writer's duplicate trimming, shared without changing its rules."""
    if len(part)<min_length:
        return []
    dup = owned.owner(part, tangents(part), radius, widths=widths, tolerance=tolerance) >= 0
    keep = np.ones(len(part), bool)
    for s, e in runs(dup):
        if e-s > trim_run:
            keep[s+overlap:max(s+overlap, e-overlap)] = False
    return [(s, e) for s, e in runs(keep) if e-s >= min_length and dup[s:e].mean() <= .5]

def unit(v):
    return v/np.linalg.norm(v, axis=-1, keepdims=True).clip(1e-9)


def tangents(path):
    return unit(np.gradient(path, axis=0)) if len(path) > 1 else np.zeros_like(path)


def runs(mask):
    """(start, stop) of maximal True runs of a boolean array."""
    m = np.diff(np.concatenate([[0], np.asarray(mask, int), [0]]))
    return list(zip(np.flatnonzero(m == 1), np.flatnonzero(m == -1)))


def outside_scan(vol, xyz):
    """Grid xyz points where the CT is 0 (beyond the scanned/masked region, or beyond the array)."""
    xyz = np.asarray(xyz, np.float64).reshape(-1, 3)
    return vol.sample_image_nearest(xyz[:, ::-1]) == 0 if len(xyz) else np.zeros(0, bool)


def outside_ct_cut(outside, gap):
    """Points of a polyline cut as outside the scan: ``outside`` runs at either end, and interior runs longer than
    ``gap`` points (shorter interior runs, a fiber grazing the scan's edge, stay)."""
    cut = np.zeros(len(outside), bool)
    for a, b in runs(outside):
        if a == 0 or b == len(outside) or b-a > gap:
            cut[a:b] = True
    return cut


class KDForest:
    """Points with tangents and ids in a few KD-trees of falling size: new points join the small trees (merged while a
    tree is under ``ratio`` times the new points), so adding never rebuilds the large ones. Trees are built when
    queried. ``nearest`` gives what one KD-tree over all the points would: the k nearest within the radius over
    every tree, by distance (equal up to exact distance ties)."""

    def __init__(self, ratio=4):
        self.ratio = ratio
        self.segments = []  # [(tree, points, tangents, ids)], sizes falling
        self.pending = []  # [(points, tangents, ids)] not in a tree yet
        self.n = 0

    def add(self, pts, tan, ids):
        self.pending.append((pts, tan, ids))
        self.n += len(pts)

    def _trees(self):
        if self.pending:
            new = [np.concatenate([q[c] for q in self.pending]) for c in range(3)]
            self.pending = []
            while self.segments and len(self.segments[-1][1]) < self.ratio*len(new[0]):
                _, p, t, i = self.segments.pop()
                new = [np.concatenate([p, new[0]]), np.concatenate([t, new[1]]), np.concatenate([i, new[2]])]
            self.segments.append((cKDTree(new[0]), *new))
        return self.segments

    def nearest(self, pts, radius, k=16):
        """(distances, tangents, ids) of the min(k, n) nearest points within ``radius`` per query point, nearest
        first (distance inf where fewer are within the radius)."""
        ds, ts, ids = [], [], []
        for tree, _, tan, fid in self._trees():
            d, j = tree.query(pts, k=min(k, tree.n), distance_upper_bound=radius)
            d, j = d.reshape(len(pts), -1), j.reshape(len(pts), -1)
            jj = np.where(np.isfinite(d), j, 0)
            ds.append(d)
            ts.append(tan[jj])
            ids.append(fid[jj])
        if len(ds) == 1:
            return ds[0], ts[0], ids[0]
        d, t, i = np.concatenate(ds, 1), np.concatenate(ts, 1), np.concatenate(ids, 1)
        order = np.argsort(d, 1, kind='stable')[:, :min(k, self.n)]
        rows = np.arange(len(pts))[:, None]
        return d[rows, order], t[rows, order], i[rows, order]


class Owned:
    """Written fibers: points with tangents and fiber ids. Recent additions sit in a small KD-forest; past --merge
    points they move into per-bin (--tile) KD-forests, so frequent writes never rebuild the large trees."""

    def __init__(self, bin_size, merge=50000):
        self.bin, self.merge = float(bin_size), merge
        self.bins = defaultdict(KDForest)  # bin -> its flushed points
        self.recent = []  # [(points, tangents, ids)]
        self.recent_forest = KDForest()
        self.recent_n = 0
        self.widths = {}
        self.width_array = np.full(1024,np.nan)
        self.width_array_complete = True

    def add(self, path, fid, width=np.nan):
        self.widths[fid] = float(width)
        if 0 <= fid < 10_000_000:
            if fid >= len(self.width_array):
                self.width_array=np.pad(self.width_array,(0,max(fid+1,2*len(self.width_array))-len(self.width_array)),
                                        constant_values=np.nan)
            self.width_array[fid]=width
        else:
            self.width_array_complete=False
        entry = (np.asarray(path, np.float64), tangents(path), np.full(len(path), fid))
        self.recent.append(entry)
        self.recent_forest.add(*entry)
        self.recent_n += len(path)
        if self.recent_n > self.merge:
            self.flush()

    def flush(self):
        for p, t, ids in self.recent:
            keys = np.floor(p/self.bin).astype(int)
            uniq, inverse = np.unique(keys, axis=0, return_inverse=True)
            for k, key in enumerate(map(tuple, uniq)):
                m = inverse.ravel() == k
                self.bins[key].add(p[m], t[m], ids[m])
        self.recent, self.recent_n, self.recent_forest = [], 0, KDForest()

    @staticmethod
    def _nearest(forest, pts, dirs, radius, exclude, best, out, rows=None, k=16,
                 widths=None, width_lookup=None, tolerance=0):
        rows = np.arange(len(pts)) if rows is None else rows
        d, tan, ids = forest.nearest(pts[rows], radius, k)
        ok = np.isfinite(d)
        ok &= np.abs(np.einsum('nkj,nj->nk', tan, dirs[rows])) > PARALLEL
        if exclude is not None:
            ok &= ids != exclude[rows][:, None]
        if widths is not None and tolerance > 0:
            if isinstance(width_lookup,np.ndarray):
                w=width_lookup[ids]
            else:
                unique,inverse=np.unique(ids,return_inverse=True)
                values=np.array([width_lookup.get(int(i),np.nan) for i in unique])
                w=values[inverse].reshape(ids.shape)
            ok &= width_compatible(w, widths[rows, None], tolerance)
        d = np.where(ok, d, np.inf)
        a = np.argmin(d, 1)
        dm = d[np.arange(len(rows)), a]
        better = dm < best[rows]
        best[rows[better]] = dm[better]
        out[rows[better]] = ids[np.arange(len(rows)), a][better]

    def owner(self, pts, dirs, radius, exclude=None, *, widths=None, tolerance=0):
        """Per point: id of the nearest parallel written fiber within ``radius`` (``exclude``: an id per point to
        ignore), else -1. Vectorized over points."""
        pts = np.asarray(pts, np.float64).reshape(-1, 3)
        dirs = np.asarray(dirs, np.float64).reshape(-1, 3)
        exclude = None if exclude is None else np.broadcast_to(np.asarray(exclude), (len(pts),))
        widths = None if widths is None else np.broadcast_to(np.asarray(widths), (len(pts),))
        out, best = np.full(len(pts), -1), np.full(len(pts), np.inf)
        width_lookup=self.width_array if self.width_array_complete else self.widths
        if not len(pts):
            return out
        if self.recent_n:
            self._nearest(self.recent_forest, pts, dirs, radius, exclude, best, out,
                          widths=widths, width_lookup=width_lookup, tolerance=tolerance)
        if self.bins:
            lo = np.floor((pts-radius)/self.bin).astype(int)
            hi = np.floor((pts+radius)/self.bin).astype(int)
            groups = defaultdict(list)
            inner = np.flatnonzero((lo == hi).all(1))  # most points: one bin
            if len(inner):
                keys, inverse = np.unique(lo[inner], axis=0, return_inverse=True)
                order = np.argsort(inverse.ravel(), kind='stable')
                splits = np.cumsum(np.bincount(inverse.ravel(), minlength=len(keys)))[:-1]
                for key, rows in zip(map(tuple, keys), np.split(inner[order], splits)):
                    groups[key].extend(rows.tolist())
            for i in np.flatnonzero(~(lo == hi).all(1)):  # near a bin face: every bin the ball touches
                a, b = lo[i], hi[i]
                for key in ((x, y, z) for x in range(a[0], b[0]+1) for y in range(a[1], b[1]+1)
                            for z in range(a[2], b[2]+1)):
                    groups[key].append(int(i))
            for key, rows in groups.items():
                forest = self.bins.get(key)
                if forest is not None:
                    self._nearest(forest, pts, dirs, radius, exclude, best, out, np.asarray(rows),
                                  widths=widths, width_lookup=width_lookup, tolerance=tolerance)
        return out


class LivePaths:
    """The paths running traces laid down, as head-to-head segments with their pair and step (vectorized queries).
    Recent segments sit in a small KD-tree rebuilt every step; every --live-merge steps they join a large one, so the
    per-step cost does not grow with the length of the running traces."""

    def __init__(self, radius, merge_steps=32):
        self.r, self.merge_steps = float(radius), merge_steps
        self.big = self._empty()
        self.small = self._empty()
        self.big_tree = None
        self.dead = set()
        self.merged_at = 0
        self.max_half = 0.

    @staticmethod
    def _empty():
        return dict(a=np.zeros((0, 3)), b=np.zeros((0, 3)), pair=np.zeros(0, int), step=np.zeros(0, int),
                    dir=np.zeros((0, 3)), width=np.zeros(0))

    def add(self, a, b, pair, step, widths=None):
        keep = np.linalg.norm(b-a, axis=1) > 1e-9
        widths = np.full(len(a), np.nan) if widths is None else np.asarray(widths)
        widths = widths[keep]
        a, b, pair = a[keep], b[keep], pair[keep]
        if len(a):
            self.max_half = max(self.max_half, float(np.linalg.norm(b-a, axis=1).max())/2)
        new = dict(a=a, b=b, pair=pair, step=np.full(len(a), step), dir=unit(b-a), width=widths)
        self.small = {k: np.concatenate([self.small[k], new[k]]) for k in new}
        if step-self.merged_at >= self.merge_steps:
            alive = ~np.isin(self.big['pair'], list(self.dead)) if self.dead else np.ones(len(self.big['pair']), bool)
            self.big = {k: np.concatenate([self.big[k][alive], self.small[k]]) for k in new}
            self.small, self.dead, self.merged_at = self._empty(), set(), step
            self.big_tree = cKDTree((self.big['a']+self.big['b'])/2) if len(self.big['a']) else None

    def drop(self, pairs):
        self.dead.update(pairs)

    def _hits(self, seg, tree, pos, dirs, pair, step, out, widths=None, tolerance=0):
        for i, cand in enumerate(tree.query_ball_point(pos, self.r+self.max_half)):
            if out[i] or not cand:
                continue
            c = np.asarray(cand)
            c = c[(seg['pair'][c] != pair[i]) & (seg['step'][c] < step) & (np.abs(seg['dir'][c] @ dirs[i]) > PARALLEL)]
            if self.dead:
                c = c[[int(q) not in self.dead for q in seg['pair'][c]]]
            if widths is not None and tolerance > 0:
                c = c[width_compatible(seg['width'][c], widths[i], tolerance)]
            if not len(c):
                continue
            a, ab = seg['a'][c], seg['b'][c]-seg['a'][c]
            t = np.clip(np.einsum('nj,nj->n', pos[i]-a, ab)/np.einsum('nj,nj->n', ab, ab), 0, 1)
            out[i] = bool((np.linalg.norm(a+t[:, None]*ab-pos[i], axis=1) <= self.r).any())

    def hits(self, pos, dirs, pair, step, widths=None, tolerance=0):
        """Per head: on a parallel segment of another pair laid down before ``step``."""
        out = np.zeros(len(pos), bool)
        if self.big_tree is not None:
            self._hits(self.big, self.big_tree, pos, dirs, pair, step, out, widths, tolerance)
        if len(self.small['a']):
            self._hits(self.small, cKDTree((self.small['a']+self.small['b'])/2), pos, dirs, pair, step, out, widths, tolerance)
        return out



