"""Contact sites from traced layer grids: geometry only, no CT and no model output.

A layer grid is an (H, W, 3) array of xyz voxel positions, one per vertex of a traced segment, such as the ordered
pointset (`pointset.vcps`) that VC writes for a segment. Two consecutive grids A and B make one layer pair. For each
valid vertex a of A, `pair_sites` measures B's local gap along B's normal and draws one site:

  b   a point of B at distance L in [LAT_MIN, LAT_MAX] voxels from a
  a2  a point of A whose distance from a matches L within MATCH voxels

so (a, a2) is a same-layer pair and (a, b) a next-layer pair at the same distance: geometry alone cannot tell them
apart. The stratum comes from both gaps: touching when both are in [3, 8] voxels, fused under 3, apart in (8, 16],
mixed otherwise. Sites are spaced at least SPACING voxels apart in a seeded order and capped per stratum. Each site's
model window is the 128 voxel cube centered on it, which must lie inside the volume. Coordinates are xyz voxels of
the volume read; shapes are given z, y, x.
"""
import hashlib
import io
import json
from pathlib import Path

import numpy as np

SEED = 20261027
DENSE = 4
LAT_MIN, LAT_MAX = 16.0, 24.0
MATCH = 1.0
SPACING = 48.0
WINDOW = 128
HALF_REACH = 40.0
CAPS = {"touching": 30, "apart": 30, "fused": 15, "mixed": 10}
STRATA = {"fused": "both gaps < 3", "touching": "both gaps in [3, 8]", "apart": "both gaps in (8, 16]",
          "mixed": "otherwise, max gap <= 16"}


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def npy_sha256(arr):
    """sha256 of the bytes np.save writes for arr (np.save output is deterministic, unlike an npz)."""
    buf = io.BytesIO()
    np.save(buf, arr)
    return sha256_bytes(buf.getvalue())


def sites_sha256(sites):
    """sha256 of a site list as canonical JSON (sorted keys, no spaces): the same sites give the same hash whatever
    file they came in."""
    return sha256_bytes(json.dumps(sites, sort_keys=True, separators=(",", ":")).encode())


def read_vcps(path):
    """A VC ordered pointset (.vcps: a text header ended by '<>', then little-endian float64) as an (H, W, 3) array."""
    b = Path(path).read_bytes()
    i = b.index(b"<>\n")
    hdr = dict(line.split(": ") for line in b[:i].decode().strip().split("\n"))
    h, w, d = int(hdr["height"]), int(hdr["width"]), int(hdr["dim"])
    a = np.frombuffer(b[i + 3:], dtype="<f8")
    if a.size != h * w * d:
        raise ValueError(f"{path}: {a.size} values, header says {h} x {w} x {d}")
    return a.reshape(h, w, d)


def valid_mask(x):
    return np.isfinite(x).all(-1) & (x >= 0).all(-1)


def one_sided(a, b, tol=3.0):
    """(matched count, share with b above a in y, a's crossing vertices) for points of a and b matched in (x, z)
    within tol voxels. A crossing vertex is one whose next layer is not above it."""
    from scipy.spatial import cKDTree
    d, j = cKDTree(b[:, [0, 2]]).query(a[:, [0, 2]], distance_upper_bound=tol)
    m = np.isfinite(d)
    dy = b[j[m], 1] - a[m, 1]
    return int(m.sum()), (float((dy > 0).mean()) if m.any() else 0.0), a[m][dy <= 0]


def stack_layers(grids, need=0.995, min_matched=1000):
    """Check that consecutive grids (already in stack order, low y to high y) are one-sided in y, and return
    (report, crossings): crossings is an (N, 3) float64 xyz array of every crossing vertex of every pair."""
    names = list(grids)
    report, cross = {}, []
    for x, y in zip(names, names[1:]):
        n, pos, c = one_sided(grids[x].reshape(-1, 3), grids[y].reshape(-1, 3))
        cross.append(c)
        report[f"{x}-{y}"] = {"matched": n, "share_positive": round(pos, 5), "crossings": len(c)}
        if n < min_matched or pos < need:
            raise ValueError(f"{x} to {y}: stack order not one-sided ({n} matched, {pos:.4f} positive)")
    return report, np.concatenate(cross).astype(np.float64)


def grid_normals(x, v):
    """Unit normal per grid vertex from central differences along rows and columns (nan where undefined)."""
    n = np.full(x.shape, np.nan)
    dr = np.full(x.shape, np.nan)
    dc = np.full(x.shape, np.nan)
    ok_r = v[2:] & v[:-2]
    dr[1:-1][ok_r] = (x[2:] - x[:-2])[ok_r]
    ok_c = v[:, 2:] & v[:, :-2]
    dc[:, 1:-1][ok_c] = (x[:, 2:] - x[:, :-2])[ok_c]
    c = np.cross(dr, dc)
    nn = np.linalg.norm(c, axis=-1, keepdims=True)
    good = np.isfinite(nn[..., 0]) & (nn[..., 0] > 1e-6)
    n[good] = (c / np.where(nn > 0, nn, 1))[good]
    return n


def densify(x, v, n, f):
    """Points and normals of every cell whose four corners are valid, bilinear at f x f sub-steps."""
    cell = v[:-1, :-1] & v[1:, :-1] & v[:-1, 1:] & v[1:, 1:]
    nok = np.isfinite(n).all(-1)
    cell &= nok[:-1, :-1] & nok[1:, :-1] & nok[:-1, 1:] & nok[1:, 1:]
    r, c = np.nonzero(cell)
    t = (np.arange(f) + 0.5) / f
    tu, tv = np.meshgrid(t, t, indexing="ij")
    tu, tv = tu.ravel()[None, :, None], tv.ravel()[None, :, None]

    def bil(g):
        return ((1 - tu) * (1 - tv) * g[r, c][:, None] + tu * (1 - tv) * g[r + 1, c][:, None]
                + (1 - tu) * tv * g[r, c + 1][:, None] + tu * tv * g[r + 1, c + 1][:, None]).reshape(-1, 3)
    p, nn = bil(x), bil(n)
    nn /= np.linalg.norm(nn, axis=1, keepdims=True) + 1e-12
    return p, nn


def normal_gap(points, tree, other_p, other_n, k=4):
    """Distance along the other layer's local normal to its nearest densified points (median of k)."""
    d, j = tree.query(points, k=k)
    g = np.abs(((points[:, None] - other_p[j]) * other_n[j]).sum(-1))
    return np.median(g, axis=1), d[:, 0]


def stratum(ga, gb):
    hi = max(ga, gb)
    if hi > 16:
        return None
    if hi < 3:
        return "fused"
    if 3 <= ga <= 8 and 3 <= gb <= 8:
        return "touching"
    if 8 < ga <= 16 and 8 < gb <= 16:
        return "apart"
    return "mixed"


class Rule:
    """The site rule: seed, densify steps, spacing, caps, and the exclusion points no window may come near."""

    def __init__(self, seed=SEED, dense=DENSE, spacing=SPACING, caps=None, stop_when_full=False, exclude=None,
                 exclude_vox=0.0):
        self.seed, self.dense, self.spacing = seed, dense, spacing
        self.caps = dict(CAPS, **(caps or {}))
        self.stop_when_full = stop_when_full
        self.exclude_vox = float(exclude_vox)
        self.exclude = None
        if exclude is not None and len(exclude):
            from scipy.spatial import cKDTree
            pts = np.asarray(exclude, np.float64).reshape(-1, 3)
            self.exclude = (cKDTree(pts), pts)

    def record(self):
        doc = dict(seed=self.seed, dense=self.dense, lat=[LAT_MIN, LAT_MAX], match=MATCH, spacing=self.spacing,
                   window=WINDOW, half_reach=HALF_REACH, caps=dict(self.caps), strata=STRATA)
        if self.stop_when_full:
            doc["stop_when_full"] = True
        if self.exclude is not None:
            doc.update(exclude_points=len(self.exclude[1]), exclude_sha256=npy_sha256(self.exclude[1]),
                       exclude_vox=self.exclude_vox)
        return doc

    def excluded(self, lo, hi):
        """True when the box [lo, hi) comes within exclude_vox of any exclusion point."""
        if self.exclude is None:
            return False
        tree, pts = self.exclude
        rad = self.exclude_vox
        c = (lo + hi - 1) / 2.0
        near = tree.query_ball_point(c, rad + np.linalg.norm((hi - lo) / 2.0))
        if not near:
            return False
        q = pts[near]
        d = np.linalg.norm(np.maximum(0, np.maximum(lo - q, q - (hi - 1))), axis=1)
        return bool((d < rad).any())


def pair_sites(name_a, A, name_b, B, shape, rng, rule):
    """Sites between consecutive layers A and B ((grid, valid) each), drawn from the shared generator rng."""
    from scipy.spatial import cKDTree
    xa, va = A
    xb, vb = B
    na, nb = grid_normals(xa, va), grid_normals(xb, vb)
    pa, qa = densify(xa, va, na, rule.dense)
    pb, qb = densify(xb, vb, nb, rule.dense)
    if not len(pa) or not len(pb):
        return []
    ta, tb = cKDTree(pa), cKDTree(pb)
    r, c = np.nonzero(va & np.isfinite(na).all(-1))
    a_pts, a_nrm = xa[r, c], na[r, c]
    ga, dnn = normal_gap(a_pts, tb, pb, qb)
    keep = (ga <= 16) & (dnn <= 30)
    if rule.exclude is not None:     # a lies inside its window, so an a this near an excluded point never makes a site
        keep &= rule.exclude[0].query(a_pts, distance_upper_bound=rule.exclude_vox)[0] >= rule.exclude_vox
    order = rng.permutation(np.nonzero(keep)[0])
    taken, out = [], []
    count = {k: 0 for k in rule.caps}
    for i in order:
        if rule.stop_when_full and all(count[k] >= rule.caps[k] for k in rule.caps):
            break
        a = a_pts[i]
        if taken and np.min(np.linalg.norm(np.asarray(taken) - a, axis=1)) < rule.spacing:
            continue
        jb = tb.query_ball_point(a, LAT_MAX)
        jb = [j for j in jb if np.linalg.norm(pb[j] - a) >= LAT_MIN]
        if not jb:
            continue
        j = int(rng.choice(jb))
        b, nb_ = pb[j], qb[j]
        L = float(np.linalg.norm(b - a))
        gb = float(normal_gap(b[None], ta, pa, qa)[0][0])
        s = stratum(float(ga[i]), gb)
        if s is None or count[s] >= rule.caps[s]:
            continue
        ja = ta.query_ball_point(a, L + MATCH)
        ja = [k for k in ja if abs(np.linalg.norm(pa[k] - a) - L) <= MATCH]
        if not ja:
            continue
        k = int(rng.choice(ja))
        a2, na2 = pa[k], qa[k]
        pts = np.stack([a, a2, b])
        center = np.round(pts.mean(0)).astype(int)
        if np.abs(pts - center).max() > HALF_REACH:
            continue
        lo = center - WINDOW // 2
        hi = lo + WINDOW
        if (lo[::-1] < 0).any() or (hi[::-1] > np.asarray(shape)).any() or rule.excluded(lo, hi):
            continue
        taken.append(a)
        count[s] += 1
        out.append({"pair": f"{name_a}-{name_b}", "stratum": s, "gap_a": round(float(ga[i]), 2), "gap_b": round(gb, 2),
                    "dist": round(L, 2), "dist_same": round(float(np.linalg.norm(a2 - a)), 2),
                    "a": a.round(3).tolist(), "a2": a2.round(3).tolist(), "b": b.round(3).tolist(),
                    "n_a": a_nrm[i].round(5).tolist(), "n_a2": na2.round(5).tolist(), "n_b": nb_.round(5).tolist(),
                    "centre": center.tolist()})
    return out


def make_sites(grids, shape_zyx, rule):
    """Sites for every consecutive pair of grids ({name: (H, W, 3) xyz}, in stack order), with ids 0, 1, ..."""
    layers = {k: (np.asarray(g, np.float64), valid_mask(np.asarray(g, np.float64))) for k, g in grids.items()}
    names = list(layers)
    rng = np.random.default_rng(rule.seed)
    sites = []
    for na_, nb_ in zip(names, names[1:]):
        sites += pair_sites(na_, layers[na_], nb_, layers[nb_], shape_zyx, rng, rule)
    for i, s in enumerate(sites):
        s["id"] = i
    return sites


def tally(sites):
    return {"sites": len(sites), "strata": {k: sum(x["stratum"] == k for x in sites) for k in CAPS},
            "touching_pairs": len({x["pair"] for x in sites if x["stratum"] == "touching"})}
