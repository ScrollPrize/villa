"""Chains of sheet crossings on a surface prediction, counted in windings (chain rule version 1, digest
0e53e0ae5016), and their export as relative-winding point collections. Needs numpy.
"""

import collections
import json
import math
from pathlib import Path


BATCH = 4096


RULE = {
    "version": 1,
    "threshold": 127,               # prediction value counted as sheet (m7 is stored as 0 or 255)
    "plane_radius_vox": 5,          # half-size of the cube a local plane is fitted in
    "min_planarity": 4.0,           # middle over smallest eigenvalue of the fitted cube's spread
    "ray_step_vox": 0.5,
    "ray_lateral_vox": 2.0,         # the profile is the OR of five parallel rays this far apart (fills pin holes)
    "max_step_vox": 40.0,           # farthest next sheet looked for
    "min_step_vox": 4.0,
    "max_width_vox": 9.0,           # a crossing thicker than this is two sheets in contact
    "max_turn_deg": 35.0,           # normal change between consecutive crossings
    "spacing_window": 4,            # steps whose median sets the expected spacing
    "spacing_lo": 0.55,             # a step must lie within [lo, hi] x the median of the previous steps
    "spacing_hi": 1.6,
    "max_steps": 40,                # per side of a seed
}


RULE_V2 = dict(RULE, version=2, double_lo=1.7, double_hi=2.4)


RULES = {1: RULE, 2: RULE_V2}


PCL_VERSION = "1"


class WindError(ValueError):
    pass


def _np():
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - map tools need numpy
        raise WindError("relative_windings_box needs numpy") from exc
    return np


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def rule_digest(rule=None):
    import hashlib
    return hashlib.sha256(canonical(rule or RULE).encode()).hexdigest()


class Pred:
    """Nearest-voxel lookups into a chunked prediction, grouped by chunk so scattered points cost one fetch per
    chunk. `fetch(idx)` returns the chunk (z, y, x) at chunk index idx, or None for a missing (empty) chunk."""

    def __init__(self, shape, chunks, fetch, threshold=None):
        self.shape = tuple(int(s) for s in shape)
        self.chunks = tuple(int(c) for c in chunks)
        self.fetch = fetch
        self.threshold = RULE["threshold"] if threshold is None else threshold
        self.workers = 8

    @classmethod
    def from_volume(cls, vol, level=0, threshold=None):
        """A _zarr.Volume (local or remote OME-Zarr)."""
        meta = vol.levels[level]["meta"]
        return cls(meta["shape"], meta["chunks"], lambda idx: vol._chunk(level, tuple(int(i) for i in idx)),
                   threshold)

    def on(self, xyz):
        """Boolean sheet mask at points (..., 3) given as (x, y, z) in voxels; outside the array is False."""
        np = _np()
        P = np.asarray(xyz, dtype=np.float64)
        flat = P.reshape(-1, 3)
        zyx = np.rint(flat[:, ::-1]).astype(np.int64)
        out = np.zeros(len(flat), bool)
        inside = np.all((zyx >= 0) & (zyx < np.array(self.shape)), axis=1)
        if not inside.any():
            return out.reshape(P.shape[:-1])
        ch = np.array(self.chunks)
        zi = zyx[inside]
        cidx = zi // ch
        local = zi - cidx * ch
        keys = (cidx[:, 0] << 40) | (cidx[:, 1] << 20) | cidx[:, 2]
        order = np.argsort(keys, kind="stable")
        ks = keys[order]
        cuts = np.flatnonzero(np.diff(ks)) + 1
        vals = np.zeros(len(zi), bool)
        groups = np.split(order, cuts)
        idxs = [tuple(int(v) for v in cidx[g[0]]) for g in groups]
        if len(idxs) > 1 and self.workers > 1:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(min(self.workers, len(idxs))) as pool:
                blocks = list(pool.map(self.fetch, idxs))
        else:
            blocks = [self.fetch(i) for i in idxs]
        for grp, block in zip(groups, blocks):
            if block is None:
                continue
            l = local[grp]
            vals[grp] = block[l[:, 0], l[:, 1], l[:, 2]] > self.threshold
        out[np.flatnonzero(inside)] = vals
        return out.reshape(P.shape[:-1])


def level_index_affine(base_affine, factor):
    """3 x 4 affine from the chain frame to the index of a pyramid level `factor` times coarser than the scan's
    level 0 (villa's mean-pool pyramid, voxel centres aligned: base f is (f + 0.5) / factor - 0.5 there),
    given the chain frame to scan level-0 affine."""
    np = _np()
    B = np.asarray(base_affine, dtype=np.float64)
    if B.shape == (4, 4):
        B = B[:3]
    if B.shape != (3, 4):
        raise WindError("the base affine is a 3 x 4 or 4 x 4 matrix on (x, y, z, 1)")
    s = 1.0 / float(factor)
    return np.concatenate([B[:, :3] * s, (B[:, 3] * s + 0.5 * s - 0.5)[:, None]], axis=1)


def _tangents(n):
    """Two unit vectors perpendicular to each unit normal (N, 3)."""
    np = _np()
    a = np.where(np.abs(n[:, 2:3]) < 0.9, np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]))
    t1 = np.cross(n, a)
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
    t2 = np.cross(n, t1)
    return t1, t2


def fit_planes(pred, P, rule=None):
    """Local plane of the predicted sheet around each point (N, 3): (normal, centroid, planarity, count).
    Normals are unit (sign arbitrary); a point with fewer than 12 sheet voxels in its cube gets NaN."""
    np = _np()
    rule = rule or RULE
    r = int(rule["plane_radius_vox"])
    g = np.arange(-r, r + 1, dtype=np.float64)
    off = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    off = off[np.linalg.norm(off, axis=1) <= r + 0.5]
    P = np.asarray(P, dtype=np.float64).reshape(-1, 3)
    if len(P) > BATCH:          # a whole trace at once needs gigabytes; rows are independent
        parts = [fit_planes(pred, P[i:i + BATCH], rule) for i in range(0, len(P), BATCH)]
        return tuple(np.concatenate([q[k] for q in parts]) for k in range(4))
    pts = P[:, None, :] + off[None]
    m = pred.on(pts)
    cnt = m.sum(1)
    w = m.astype(np.float64)
    tot = np.maximum(cnt, 1)[:, None]
    cen = (pts * w[..., None]).sum(1) / tot
    d = (pts - cen[:, None, :]) * w[..., None]
    cov = np.einsum("nki,nkj->nij", d, d) / tot[..., None]
    ev, evec = np.linalg.eigh(cov)
    normal = evec[:, :, 0]
    planar = ev[:, 1] / np.maximum(ev[:, 0], 1e-6)
    bad = cnt < 12
    normal[bad] = np.nan
    planar[bad] = 0.0
    return normal, cen, planar, cnt


def profiles(pred, P, n, t, rule=None):
    """Sheet mask along rays P + t n (OR of five parallel rays), shape (N, len(t))."""
    np = _np()
    rule = rule or RULE
    lat = float(rule["ray_lateral_vox"])
    t1, t2 = _tangents(n)
    shifts = [np.zeros_like(n), lat * t1, -lat * t1, lat * t2, -lat * t2]
    acc = None
    for s in shifts:
        pts = (P + s)[:, None, :] + t[None, :, None] * n[:, None, :]
        m = pred.on(pts)
        acc = m if acc is None else (acc | m)
    return acc


def _runs(row):
    """(start, end) index pairs of True runs in a 1-D boolean array (end exclusive)."""
    np = _np()
    d = np.diff(np.concatenate([[0], row.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def own_and_next(pred, P, n, rule=None, max_t=None):
    """For points near a sheet, along +n and -n: the offset of the point's own sheet (the run nearest t = 0
    within 3 voxels), and of the next sheet beyond it on each side with its width. Returns dict of arrays (N,):
    own (NaN when the point is not on a sheet), own_w, plus (+n side) and minus (-n side) offsets (signed, NaN
    when none within max_t), plus_w and minus_w."""
    np = _np()
    rule = rule or RULE
    step = float(rule["ray_step_vox"])
    T = float(max_t or rule["max_step_vox"])
    t = np.arange(-T, T + step / 2, step)
    zero = int(np.argmin(np.abs(t)))
    P = np.asarray(P, dtype=np.float64).reshape(-1, 3)
    n = np.asarray(n, dtype=np.float64).reshape(-1, 3)
    N = len(P)
    out = {k: np.full(N, np.nan) for k in ("own", "own_w", "plus", "plus_w", "minus", "minus_w")}
    if N == 0:
        return out
    for i in range(N):
        if i % BATCH == 0:
            prof = profiles(pred, P[i:i + BATCH], n[i:i + BATCH], t, rule)
        runs = _runs(prof[i % BATCH])
        if not runs:
            continue
        cen = [((a + b - 1) / 2.0) for a, b in runs]
        dist = [0 if a <= zero < b else min(abs(a - zero), abs(b - 1 - zero)) for a, b in runs]
        j = int(np.argmin(dist))
        if dist[j] * step > 3.0:
            continue
        a, b = runs[j]
        out["own"][i] = t[0] + cen[j] * step
        out["own_w"][i] = (b - a) * step
        if j + 1 < len(runs) and runs[j + 1][1] < len(t):
            a2, b2 = runs[j + 1]
            out["plus"][i] = t[0] + cen[j + 1] * step
            out["plus_w"][i] = (b2 - a2) * step
        if j - 1 >= 0 and runs[j - 1][0] > 0:
            a2, b2 = runs[j - 1]
            out["minus"][i] = t[0] + cen[j - 1] * step
            out["minus_w"][i] = (b2 - a2) * step
    return out


def load_umbilicus(path):
    """An umbilicus control points JSON ({"control_points": [{x, y, z}]}) or a [[z, y, x], ...] list, as an (M, 3)
    array (z, y, x)."""
    np = _np()
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict):
        pts = data.get("control_points") or data.get("points") or []
        rows = [[p["z"], p["y"], p["x"]] if isinstance(p, dict) else p for p in pts]
    else:
        rows = data
    arr = np.asarray(rows, dtype=np.float64).reshape(-1, 3)
    if len(arr) == 0:
        raise WindError(f"{path}: no umbilicus points")
    return arr[np.argsort(arr[:, 0])]


def axis_at(axis, z):
    """(N, 2) axis (x, y) at heights z, linear between control points and constant beyond."""
    np = _np()
    z = np.asarray(z, dtype=np.float64)
    return np.stack([np.interp(z, axis[:, 0], axis[:, 2]), np.interp(z, axis[:, 0], axis[:, 1])], -1)


def orient_outward(P, n, axis):
    """Flip normals (N, 3) so their xy part points away from the axis."""
    np = _np()
    c = axis_at(axis, P[:, 2])
    s = np.einsum("ij,ij->i", P[:, :2] - c, n[:, :2])
    return np.where((s < 0)[:, None], -n, n)


def seeds_on(pred, box, count, rng_seed=0, tries=200000):
    """`count` points on the prediction inside box [[x0, x1], [y0, y1], [z0, z1]], drawn with a fixed RNG."""
    np = _np()
    rng = np.random.default_rng(rng_seed)
    lo = np.array([b[0] for b in box], dtype=np.float64)
    hi = np.array([b[1] for b in box], dtype=np.float64)
    got = []
    left = tries
    # draw centres first, then a few points near each, so a remote prediction is read in few chunks
    per = 8
    while len(got) < count and left > 0:
        k = min(left // per + 1, 256)
        cen = lo + rng.random((k, 3)) * (hi - lo)
        cand = np.clip((cen[:, None, :] + rng.normal(0.0, 24.0, (k, per, 3))).reshape(-1, 3), lo, hi)
        m = pred.on(cand).reshape(k, per)
        hit = m.any(1)
        first = m.argmax(1)
        got.extend(cand.reshape(k, per, 3)[np.flatnonzero(hit), first[hit]].tolist())
        left -= k * per
    return np.asarray(got[:count], dtype=np.float64).reshape(-1, 3)


def _step(pred, C, n, rule):
    """One step from crossings C (on a sheet) along n: (next centre, width, step length) or NaN."""
    np = _np()
    step = float(rule["ray_step_vox"])
    t = np.arange(0.0, float(rule["max_step_vox"]) + step / 2, step)
    prof = profiles(pred, C, n, t, rule)
    N = len(C)
    nxt = np.full(N, np.nan)
    wid = np.full(N, np.nan)
    for i in range(N):
        row = prof[i]
        runs = _runs(row)
        if not runs:
            continue
        k = 0
        if runs[0][0] == 0:
            k = 1           # the run we start in is our own sheet
        if k >= len(runs):
            continue
        a, b = runs[k]
        if b >= len(t):
            continue        # runs off the end: not a whole sheet in reach
        nxt[i] = t[0] + (a + b - 1) / 2.0 * step
        wid[i] = (b - a) * step
    return nxt, wid


def _recentre(pred, Q, n, rule):
    """Move points onto the centre of the sheet run that holds them along n (within 3 voxels)."""
    np = _np()
    o = own_and_next(pred, Q, n, rule, max_t=6.0)
    ok = np.isfinite(o["own"])
    return np.where(ok[:, None], Q + np.nan_to_num(o["own"])[:, None] * n, np.nan), o["own_w"]


def grow_chains(pred, seeds, axis=None, rule=None, log=None):
    """Chains of sheet crossings from each seed, both sides. Returns a list of dicts with points (K, 3), counts
    (K,) (0 at the seed, + along the outward normal), steps (K-1,) in voxels, normals (K, 3) and why each side
    stopped. When axis is None, outward is the seed normal's sign (counts then share no global sense)."""
    np = _np()
    rule = rule or RULE
    seeds = np.asarray(seeds, dtype=np.float64).reshape(-1, 3)
    n0, c0, planar, _ = fit_planes(pred, seeds, rule)
    keep = np.isfinite(n0).all(1) & (planar >= rule["min_planarity"])
    if axis is not None:
        n0[keep] = orient_outward(c0[keep], n0[keep], axis)
    Q0, _ = _recentre(pred, np.where(keep[:, None], c0, 0.0), np.nan_to_num(n0), rule)
    keep &= np.isfinite(Q0).all(1)
    S = len(seeds)
    chains = [{"seed": seeds[i].tolist(), "points": [], "counts": [], "normals": [], "steps": [], "stop": {}}
              for i in range(S)]
    for i in np.flatnonzero(keep):
        chains[i]["points"].append(Q0[i].tolist())
        chains[i]["normals"].append(n0[i].tolist())
        chains[i]["counts"].append(0)
    for side in (+1, -1):
        cur = Q0.copy()
        nrm = n0.copy() * side
        active = keep.copy()
        hist = [[] for _ in range(S)]
        total = [0] * S
        k = 0
        while active.any() and k < rule["max_steps"]:
            k += 1
            ids = np.flatnonzero(active)
            L, W = _step(pred, cur[ids], nrm[ids], rule)
            reason = np.full(len(ids), "", dtype=object)
            reason[~np.isfinite(L)] = "no sheet in reach"
            ok = np.isfinite(L)
            reason[ok & (W > rule["max_width_vox"])] = "thick crossing (sheets in contact)"
            ok &= W <= rule["max_width_vox"]
            reason[ok & (L < rule["min_step_vox"])] = "step too short"
            ok &= L >= rule["min_step_vox"]
            inc = np.ones(len(ids), np.int64)
            for j, i in enumerate(ids):
                if not ok[j] or len(hist[i]) < 2:
                    continue
                med = float(np.median(hist[i][-rule["spacing_window"]:]))
                if _is_double(L[j] / med, rule):
                    inc[j] = 2
                elif not (rule["spacing_lo"] * med <= L[j] <= rule["spacing_hi"] * med):
                    ok[j] = False
                    reason[j] = "spacing out of line (a missed or extra sheet)"
            Q = cur[ids] + np.nan_to_num(L)[:, None] * nrm[ids]
            nq, cq, pq, _ = fit_planes(pred, Q, rule)
            good = np.isfinite(nq).all(1)
            nq = np.where(good[:, None], nq, nrm[ids])
            nq = np.where((np.einsum("ij,ij->i", nq, nrm[ids]) < 0)[:, None], -nq, nq)
            turn = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", nq, nrm[ids]), -1, 1)))
            bad_plane = ok & (~good | (pq < rule["min_planarity"]))
            reason[bad_plane] = "no clean plane at the next sheet"
            ok &= ~bad_plane
            reason[ok & (turn > rule["max_turn_deg"])] = "normal turned too far"
            ok &= turn <= rule["max_turn_deg"]
            Qc, _ = _recentre(pred, np.where(ok[:, None], Q, 0.0), nq, rule)
            lost = ok & ~np.isfinite(Qc).all(1)
            reason[lost] = "lost the sheet when recentring"
            ok &= ~lost
            for j, i in enumerate(ids):
                if not ok[j]:
                    active[i] = False
                    chains[i]["stop"]["out" if side > 0 else "in"] = reason[j]
                    continue
                step_len = float(np.linalg.norm(Qc[j] - cur[i]))
                hist[i].append(step_len / int(inc[j]))
                cur[i] = Qc[j]
                nrm[i] = nq[j]
                ch = chains[i]
                total[i] += int(inc[j])
                rec = (Qc[j].tolist(), (nq[j] * side).tolist(), side * total[i])
                if side > 0:
                    ch["points"].append(rec[0])
                    ch["normals"].append(rec[1])
                    ch["counts"].append(rec[2])
                else:
                    ch["points"].insert(0, rec[0])
                    ch["normals"].insert(0, rec[1])
                    ch["counts"].insert(0, rec[2])
        for i in np.flatnonzero(active):
            chains[i]["stop"]["out" if side > 0 else "in"] = "step limit"
        if log:
            log(f"relative_windings_box: side {'+' if side > 0 else '-'} done after {k} steps")
    for i, ch in enumerate(chains):
        if not keep[i]:
            ch["stop"] = {"out": "seed not on a clean sheet", "in": "seed not on a clean sheet"}
        _trim(ch, rule)
    return chains


def _is_double(ratio, rule):
    """Version 2: a step this many times the expected spacing crossed a sheet the prediction missed."""
    return "double_lo" in rule and rule["double_lo"] <= ratio <= rule["double_hi"]


def _trim(ch, rule):
    """Cut a chain at the first step, walking out from the seed on each side, that is out of line with the
    median step of the whole chain. Early steps have no history to check against while growing; this pass
    gives them one (a first step across a missing sheet is about twice the others). Under version 2 such a
    step is counted as two wraps instead, and the counts are rebuilt outward from the seed."""
    np = _np()
    p = np.asarray(ch["points"], dtype=np.float64).reshape(-1, 3)
    steps = np.linalg.norm(np.diff(p, axis=0), axis=1) if len(p) > 1 else np.zeros(0)
    if len(steps) >= 3:
        inc = np.abs(np.diff(np.asarray(ch["counts"], dtype=np.int64)))
        med = float(np.median(steps / np.maximum(inc, 1)))
        if "double_lo" in rule:
            inc = np.where([_is_double(v / med, rule) for v in steps], 2, 1)
        unit = steps / inc
        bad = ((unit < rule["spacing_lo"] * med) | (unit > rule["spacing_hi"] * med)) & (inc == 1)
        k0 = ch["counts"].index(0)
        if "double_lo" in rule:
            cnt = np.zeros(len(p), np.int64)
            for j in range(k0, len(steps)):
                cnt[j + 1] = cnt[j] + inc[j]
            for j in range(k0 - 1, -1, -1):
                cnt[j] = cnt[j + 1] - inc[j]
            ch["counts"] = cnt.tolist()
        lo, hi = 0, len(p)
        for j in range(k0, len(steps)):          # step j joins points j and j + 1
            if bad[j]:
                hi = j + 1
                ch["stop"]["out"] = "spacing out of line with the chain (trimmed)"
                break
        for j in range(k0 - 1, -1, -1):
            if bad[j]:
                lo = j + 1
                ch["stop"]["in"] = "spacing out of line with the chain (trimmed)"
                break
        for key in ("points", "normals", "counts"):
            ch[key] = ch[key][lo:hi]
        p = p[lo:hi]
        steps = np.linalg.norm(np.diff(p, axis=0), axis=1) if len(p) > 1 else np.zeros(0)
    ch["steps"] = steps.round(3).tolist()


def summary(chains):
    np = _np()
    lens = [len(c["points"]) for c in chains]
    steps = [s for c in chains for s in c["steps"]]
    stops = collections.Counter(v for c in chains for v in c["stop"].values())
    used = [n for n in lens if n >= 2]
    return {"chains": len(chains), "with_two_or_more": len(used), "crossings": int(sum(lens)),
            "median_length": float(np.median(used)) if used else 0.0,
            "median_step_vox": round(float(np.median(steps)), 3) if steps else None,
            "stops": dict(stops.most_common())}


EXPORT_VERSION = 2


CAL_KIND = "windlabel-calibration"


FLAG_KINDS = {
    "cut": ("windlabel-flag-cut-step", [1.0, 0.2, 0.2],
            "a step the confidence cut: check whether it crosses exactly one winding"),
    "contact": ("windlabel-flag-contact", [1.0, 0.6, 0.0],
                "a chain end where the next crossing was too thick (two sheets in contact)"),
    "spacing": ("windlabel-flag-spacing", [1.0, 1.0, 0.2],
                "a chain end at a step out of line with the spacing (a missed or extra sheet)"),
}


_STOP_FLAG = {"thick crossing (sheets in contact)": "contact",
              "spacing out of line (a missed or extra sheet)": "spacing",
              "spacing out of line with the chain (trimmed)": "spacing"}


BUILTIN_FEATURES = ("ratio",)


def step_table(chains, features=None, feature="ratio"):
    """One record per step (two consecutive crossings of a chain), in chain then step order: chain index, step
    index j (joins points j and j + 1), count difference dk, length (voxels) and the feature value.

    feature "ratio" needs nothing but the chain: |log(step length per count / median of the chain's steps per
    count)|, 0 for a step in line with its chain. Any other name is read from `features`, records
    {"chain": i, "j": j, NAME: value} such as the split rule's features (gm_x); NaN where missing."""
    np = _np()
    if feature not in BUILTIN_FEATURES and features is None:
        raise WindError(f"feature {feature!r} needs a features file (records with chain, j and {feature})")
    look = {}
    for f in features or []:
        v = f.get(feature)
        look[(int(f["chain"]), int(f["j"]))] = float("nan") if v is None else float(v)
    out = []
    for i, ch in enumerate(chains):
        P = np.asarray(ch["points"], dtype=np.float64).reshape(-1, 3)
        K = list(ch["counts"])
        if len(P) < 2:
            continue
        L = np.linalg.norm(np.diff(P, axis=0), axis=1)
        dk = np.abs(np.diff(np.asarray(K, dtype=np.int64)))
        unit = L / np.maximum(dk, 1)
        med = float(np.median(unit)) if len(unit) else float("nan")
        for j in range(len(L)):
            if feature == "ratio":
                v = abs(math.log(unit[j] / med)) if med > 0 and unit[j] > 0 else float("nan")
            else:
                v = look.get((i, j), float("nan"))
            out.append({"chain": i, "j": j, "dk": int(dk[j]), "length": float(L[j]), "value": v})
    return out


def apply_calibration(cal, values):
    """Probability that each step is right, from a calibration table: the bin that holds the value, or the
    nearer bin edge between bins; NaN values get the table's missing_p."""
    np = _np()
    if cal.get("kind") != CAL_KIND:
        raise WindError("not a windlabel calibration table (kind)")
    v = np.asarray(values, dtype=np.float64)
    bins = cal["bins"]
    lo = np.array([b["lo"] for b in bins])
    hi = np.array([b["hi"] for b in bins])
    p = np.array([b["p"] for b in bins])
    edges = (hi[:-1] + lo[1:]) / 2.0          # a value between two bins goes to the nearer one
    idx = np.searchsorted(edges, np.nan_to_num(v, nan=0.0), side="right")
    out = p[np.clip(idx, 0, len(p) - 1)]
    return np.where(np.isfinite(v), out, float(cal["missing_p"]))


def _affine_fn(affine, scale):
    np = _np()
    if affine is None and scale == 1.0:
        return lambda P: np.asarray(P, dtype=np.float64)
    M = np.eye(4) if affine is None else np.asarray(affine, dtype=np.float64)
    if M.shape == (3, 4):
        M = np.vstack([M, [0.0, 0.0, 0.0, 1.0]])
    if M.shape != (4, 4):
        raise WindError("the affine is a 3 x 4 or 4 x 4 matrix on (x, y, z, 1)")
    return lambda P: (np.c_[np.asarray(P, dtype=np.float64).reshape(-1, 3), np.ones(len(P))] @ M.T)[:, :3] * scale


def _conf_color(p):
    """Red (0.5 or lower) through yellow to green (1.0)."""
    t = min(max((p - 0.5) / 0.5, 0.0), 1.0)
    return [round(min(1.0, 2.0 * (1.0 - t)), 3), round(min(1.0, 2.0 * t), 3), 0.2]


def export(chains, cal=None, features=None, min_step_p=None, min_points=2, affine=None, scale=1.0,
           name="windlabel", meta=None, max_flags=None):
    """(relative_windings, flags, summary) for one chain set. Each chain is cut at every step whose calibrated
    probability is below min_step_p (needs `cal`); each piece of at least min_points crossings becomes one
    collection, wind_a 0 at its innermost crossing and counting up outward, points numbered from 1 across the
    file in chain order (villa orders a collection's points by id), creation_time 1, 2, ... in each collection,
    and metadata winding_is_absolute false (VC3D's default is true). Points go through `affine` (3 x 4 or 4 x 4
    on x, y, z) and then `scale`, to put them in the spiral fit's frame."""
    np = _np()
    if min_step_p is not None and cal is None:
        raise WindError("--min-step-p needs a calibration table")
    feature = cal["feature"] if cal else "ratio"
    table = step_table(chains, features, feature)
    probs = apply_calibration(cal, [r["value"] for r in table]) if cal else None
    by_chain = collections.defaultdict(dict)
    for t, r in enumerate(table):
        by_chain[r["chain"]][r["j"]] = (t, None if probs is None else float(probs[t]))
    tf = _affine_fn(affine, scale)
    rel_cols, flag_pts = {}, {k: [] for k in FLAG_KINDS}
    pid = 0
    kept_steps = cut_steps = 0
    kept_p = []
    pieces = 0
    for i, ch in enumerate(chains):
        n = len(ch["points"])
        if n == 0:
            continue
        cuts = set()
        for j in range(n - 1):
            _, p = by_chain[i].get(j, (None, None))
            if min_step_p is not None and p is not None and p < min_step_p:
                cuts.add(j)
                a, b = np.asarray(ch["points"][j], float), np.asarray(ch["points"][j + 1], float)
                flag_pts["cut"].append({"p": ((a + b) / 2).tolist(), "chain": i, "j": j, "step_p": round(p, 4)})
        cut_steps += len(cuts)
        start = 0
        for end in sorted(cuts) + [n - 1]:
            seg = list(range(start, end + 1))
            start = end + 1
            if len(seg) < max(min_points, 2):
                continue
            pieces += 1
            K = [ch["counts"][t] for t in seg]
            base = min(K)
            P = tf([ch["points"][t] for t in seg])
            pts = {}
            for c_t, (xyz, k) in enumerate(zip(P, K), 1):
                pid += 1
                pts[str(pid)] = {"p": [round(float(x), 3) for x in xyz], "wind_a": int(k - base), "creation_time": c_t}
            md = {"winding_is_absolute": False, "source": "relative_windings_box export",
                  "export_version": EXPORT_VERSION, "chain": i, "first_crossing": seg[0], "seed": ch.get("seed")}
            sp = [by_chain[i].get(t, (None, None))[1] for t in seg[:-1]]
            kept_steps += len(seg) - 1
            color = [0.2, 0.6, 1.0]
            if probs is not None:
                sp = [round(float(x), 4) for x in sp]
                kept_p.extend(sp)
                md.update({"step_p": sp, "min_step_p": min(sp), "piece_p": round(float(np.prod(sp)), 4)})
                color = _conf_color(min(sp))
            rel_cols[str(len(rel_cols) + 1)] = {"name": f"{name}-{i:05d}-{seg[0]:02d}", "points": pts,
                                                "metadata": md, "color": color}
        for side, idx in (("in", 0), ("out", n - 1)):
            kind = _STOP_FLAG.get(ch.get("stop", {}).get(side))
            if kind:
                flag_pts[kind].append({"p": list(ch["points"][idx]), "chain": i, "side": side,
                                       "stop": ch["stop"][side]})
    head = {"export_version": EXPORT_VERSION, "note": "Model output from a surface prediction; not a reading.",
            "calibration_digest": cal.get("digest") if cal else None, "min_step_p": min_step_p,
            "frame": {"affine": None if affine is None else np.asarray(affine, float).tolist(), "scale": scale}}
    if meta:
        head.update(meta)
    rel = {"vc_pointcollections_json_version": PCL_VERSION, "collections": rel_cols, "relative_windings_box": head}
    fcols, fmeta, fid = {}, {}, 0
    for kind, items in flag_pts.items():
        if max_flags is not None:
            items = items[:max_flags]
        if not items:
            continue
        nm, color, why = FLAG_KINDS[kind]
        P = tf([it["p"] for it in items])
        pts = {}
        for c_t, (xyz, it) in enumerate(zip(P, items), 1):
            fid += 1
            pts[str(fid)] = {"p": [round(float(x), 3) for x in xyz], "wind_a": None, "creation_time": c_t}
            fmeta[str(fid)] = dict({k: v for k, v in it.items() if k != "p"}, kind=kind)
        fcols[str(len(fcols) + 1)] = {"name": nm, "points": pts, "color": color,
                                      "metadata": {"winding_is_absolute": False,
                                                   "source": "relative_windings_box export", "why": why}}
    flags = {"vc_pointcollections_json_version": PCL_VERSION, "collections": fcols,
             "relative_windings_box": dict(head, points=fmeta)}
    summ = {"chains": len(chains), "steps": len(table), "steps_kept": kept_steps, "steps_cut": cut_steps,
            "kept_share": round(kept_steps / len(table), 4) if table else None,
            "pieces": pieces, "collections": len(rel_cols), "points": pid,
            "flags": {k: min(len(v), max_flags) if max_flags is not None else len(v) for k, v in flag_pts.items()},
            "feature": feature,
            "expected_step_accuracy": round(float(np.mean(kept_p)), 4) if kept_p else None}
    return rel, flags, summ


def check_pcl(doc, role="relative"):
    """Problems villa's spiral-fitting loader and service, or VC3D, would have with a point collection file
    (villa main, read 21 Oct 2026: point_collection.load_point_collection, fit_spiral orders points by id;
    VC3D core/src/PointCollections.cpp). The files are full point-collection uploads with role relative, not
    replacements: the increasing creation_time follows service_uploads' replacement order as a convention, and
    the replacement rule that point ids run from 0 in each collection does not apply. role "relative": at least
    two points per collection, every point has a finite wind_a and creation_time strictly increases in each
    collection; "flags": no wind_a. Both: version
    "1", integer collection and point ids, point ids unique across the file and from 1, p three finite
    numbers, metadata winding_is_absolute false, a uuid (when present) shaped like a UUID."""
    import re
    issues = []
    if not isinstance(doc, dict) or doc.get("vc_pointcollections_json_version") != PCL_VERSION:
        return {"ok": False, "issues": ["vc_pointcollections_json_version is not \"1\""], "collections": 0, "points": 0}
    cols = doc.get("collections")
    if not isinstance(cols, dict):
        return {"ok": False, "issues": ["collections is not an object"], "collections": 0, "points": 0}
    seen = set()
    npts = 0
    if role == "relative" and not cols:
        issues.append("no collections (villa's service refuses an empty point collection upload)")
    uuid_re = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
    for cid, c in cols.items():
        where = f"collection {cid}"
        if not str(cid).isdigit():
            issues.append(f"{where}: id is not a non-negative integer")
        if not isinstance(c, dict) or not isinstance(c.get("name"), str):
            issues.append(f"{where}: no name")
            continue
        if "uuid" in c and not (isinstance(c["uuid"], str) and uuid_re.match(c["uuid"])):
            issues.append(f"{where}: uuid is not a UUID (VC3D refuses the file)")
        md = c.get("metadata")
        if not isinstance(md, dict) or md.get("winding_is_absolute") is not False:
            issues.append(f"{where}: metadata.winding_is_absolute is not false (VC3D then reads absolute windings)")
        col = c.get("color")
        if col is not None and not (isinstance(col, list) and len(col) == 3):
            issues.append(f"{where}: color is not three numbers")
        pts = c.get("points")
        if not isinstance(pts, dict) or not pts:
            issues.append(f"{where}: no points")
            continue
        if role == "relative" and len(pts) < 2:
            issues.append(f"{where}: fewer than two points (one point carries no relative winding)")
        prev_t = None
        for pid in sorted(pts, key=lambda k: int(k) if str(k).isdigit() else -1):
            pt = pts[pid]
            pw = f"{where} point {pid}"
            npts += 1
            if not str(pid).isdigit() or int(pid) < 1:
                issues.append(f"{pw}: id is not an integer from 1")
            elif int(pid) in seen:
                issues.append(f"{pw}: id repeats one in another collection")
            else:
                seen.add(int(pid))
            p = pt.get("p") if isinstance(pt, dict) else None
            if not (isinstance(p, list) and len(p) == 3 and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                                                                  and math.isfinite(x) for x in p)):
                issues.append(f"{pw}: p is not three finite numbers")
            w = pt.get("wind_a") if isinstance(pt, dict) else None
            if role == "relative":
                if not (isinstance(w, (int, float)) and not isinstance(w, bool) and math.isfinite(w)):
                    issues.append(f"{pw}: wind_a is not a finite number (villa drops the point)")
                t = pt.get("creation_time")
                if not (isinstance(t, int) and not isinstance(t, bool)):
                    issues.append(f"{pw}: creation_time is not an integer")
                elif prev_t is not None and t <= prev_t:
                    issues.append(f"{pw}: creation_time does not increase (villa's service refuses such a "
                                  "collection as a replacement)")
                else:
                    prev_t = t
            elif w is not None:
                issues.append(f"{pw}: a flag point carries wind_a")
    return {"ok": not issues, "issues": issues[:50], "issue_count": len(issues), "collections": len(cols),
            "points": npts}


def _open_pred(location, level, threshold):
    from . import _zarr
    vol = _zarr.Volume(location, cache_chunks=512, workers=8)
    return Pred.from_volume(vol, level, threshold)


def _box_at(s, half):
    """The seed box reaching `half` voxels from the point x,y,z on each axis."""
    try:
        v = [float(x) for x in s.split(",")]
    except ValueError:
        v = []
    if len(v) != 3 or not all(math.isfinite(x) for x in v):
        raise WindError("the point is x,y,z in level-0 voxels")
    if not (half > 0 and math.isfinite(half)):
        raise WindError("the box edge must be a positive number of voxels")
    return [[v[0] - half, v[0] + half], [v[1] - half, v[1] + half], [v[2] - half, v[2] + half]]
