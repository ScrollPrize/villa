"""Field sampling along a step between two points: the Lasagna grad_mag integral the split rule reads.

Points are (x, y, z) voxels of a query frame; each field maps them to its own array index with a 3 x 4 affine.
"""

import math


RULE = {
    "version": 1,
    "step_vox": 0.5,              # profile sample spacing, query voxels
    "extend_vox": 64.0,           # profile extends this far beyond each end
    "ray_lateral_vox": 2.0,       # m7 and cos: five parallel rays (centre and +-lat on two tangents)
    "sheet_threshold": 127,       # m7 value counted as sheet
    "merge_gap_vox": 2.0,         # m7 runs closer than this are one sheet
    "min_gap_vox": 4.0,           # gaps shorter than this are not used for the line's pitch
    "snap_pitch": 0.35,           # an end snaps to a run or peak within this many pitches
    "snap_default_vox": 4.0,      # when the line has no pitch
    "cos_smooth_vox": 1.0,        # Gaussian sigma of the cos profile
    "cos_min_prominence": 0.10,   # cos peaks (0 to 1 scale) at least this prominent
    "gm_rays": 7,                 # grad_mag: median of parallel rays (E1's layout)
    "gm_offset_vox": 6.0,
    "max_subtended_deg": 90.0,    # pairs subtending more at the axis are not answered
    "priority": ["gap", "cos", "gm"],
}


COUNTERS = ("m7", "gap", "cos", "gm")


class PairError(ValueError):
    pass


def _np():
    try:
        import numpy as np
    except ImportError as exc:      # pragma: no cover
        raise PairError(f"relative_windings_box needs numpy ({exc})") from exc
    return np


def open_volume(location, cache_chunks=256):
    """A _zarr.Volume of a local or remote zarr, its requests paced (the grad_mag reads)."""
    from . import _zarr
    return _zarr.Volume(store=_zarr.open_store(str(location)), cache_chunks=cache_chunks)


class Field:
    """Values of one array (a level of a chunked volume, or a numpy array) at query points, read chunk by chunk.
    `affine` (3 x 4) maps query (x, y, z) to the array's (x, y, z) index; values are trilinear (or nearest)
    and NaN outside the array. `scale` multiplies raw values."""

    def __init__(self, shape, chunks, fetch, affine=None, scale=1.0, fill=0, workers=8):
        np = _np()
        self.shape = tuple(int(s) for s in shape)
        self.chunks = tuple(int(c) for c in chunks)
        self.fetch = fetch
        A = np.eye(3, 4) if affine is None else np.asarray(affine, dtype=np.float64)
        if A.shape != (3, 4):
            raise PairError("an affine is 3 x 4, acting on (x, y, z, 1)")
        self.affine = A
        self.scale = float(scale)
        self.fill = fill
        self.workers = workers

    @classmethod
    def from_array(cls, arr, affine=None, scale=1.0, chunks=(64, 64, 64)):
        np = _np()
        arr = np.asarray(arr)

        def fetch(idx):
            sl = tuple(slice(i * c, (i + 1) * c) for i, c in zip(idx, chunks))
            block = arr[sl]
            if block.shape != tuple(chunks):
                block = np.pad(block, [(0, c - s) for c, s in zip(chunks, block.shape)])
            return block
        return cls(arr.shape, chunks, fetch, affine, scale)

    @classmethod
    def from_volume(cls, vol, level, affine=None, scale=1.0):
        meta = vol.levels[level]["meta"]
        fill = meta.get("fill_value")
        return cls(meta["shape"], meta["chunks"], lambda idx: vol._chunk(level, tuple(int(i) for i in idx)),
                   affine, scale, fill if isinstance(fill, (int, float)) and fill == fill else 0)

    def index(self, xyz):
        np = _np()
        P = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
        return P @ self.affine[:, :3].T + self.affine[:, 3]

    def _lookup(self, zyx):
        """Raw values at integer (z, y, x) indices (M, 3), NaN outside the array."""
        np = _np()
        out = np.full(len(zyx), np.nan)
        inside = np.all((zyx >= 0) & (zyx < np.array(self.shape)), axis=1)
        if not inside.any():
            return out
        ch = np.array(self.chunks)
        zi = zyx[inside]
        cidx = zi // ch
        local = zi - cidx * ch
        keys = (cidx[:, 0] << 42) | (cidx[:, 1] << 21) | cidx[:, 2]
        order = np.argsort(keys, kind="stable")
        cuts = np.flatnonzero(np.diff(keys[order])) + 1
        groups = np.split(order, cuts)
        idxs = [tuple(int(v) for v in cidx[g[0]]) for g in groups]
        if len(idxs) > 1 and self.workers > 1:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(min(self.workers, len(idxs))) as pool:
                blocks = list(pool.map(self.fetch, idxs))
        else:
            blocks = [self.fetch(i) for i in idxs]
        vals = np.empty(len(zi))
        for grp, block in zip(groups, blocks):
            if block is None:
                vals[grp] = self.fill
            else:
                lz = local[grp]
                vals[grp] = block[lz[:, 0], lz[:, 1], lz[:, 2]]
        out[inside] = vals
        return out

    def nearest(self, xyz):
        np = _np()
        P = np.asarray(xyz, dtype=np.float64)
        idx = self.index(P)
        zyx = np.rint(idx[:, ::-1]).astype(np.int64)
        return (self._lookup(zyx) * self.scale).reshape(P.shape[:-1])

    def linear(self, xyz):
        np = _np()
        P = np.asarray(xyz, dtype=np.float64)
        idx = self.index(P)[:, ::-1]
        i0 = np.floor(idx).astype(np.int64)
        f = idx - i0
        corners = []
        for dz in (0, 1):
            for dy in (0, 1):
                for dx in (0, 1):
                    corners.append(((dz, dy, dx), i0 + np.array([dz, dy, dx])))
        vals = self._lookup(np.concatenate([c for _, c in corners]))
        n = len(i0)
        out = np.zeros(n)
        for k, ((dz, dy, dx), _) in enumerate(corners):
            w = ((f[:, 0] if dz else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1]) *
                 (f[:, 2] if dx else 1 - f[:, 2]))
            out += w * vals[k * n:(k + 1) * n]
        return (out * self.scale).reshape(P.shape[:-1])


def subtended_deg(axis, a, b):
    """Angle at the axis (at the pair's mid height) between a and b, in the xy plane."""
    np = _np()
    a, b = np.atleast_2d(a), np.atleast_2d(b)
    ax, ay = axis.at(0.5 * (a[:, 2] + b[:, 2]))
    va = np.stack([a[:, 0] - ax, a[:, 1] - ay], 1)
    vb = np.stack([b[:, 0] - ax, b[:, 1] - ay], 1)
    c = (va * vb).sum(1) / np.maximum(np.linalg.norm(va, axis=1) * np.linalg.norm(vb, axis=1), 1e-12)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def radial_sign(axis, a, b):
    """+1 when b lies farther from the axis than a (at the pair's mid height), -1 when nearer, 0 when equal."""
    np = _np()
    a, b = np.atleast_2d(a), np.atleast_2d(b)
    ax, ay = axis.at(0.5 * (a[:, 2] + b[:, 2]))
    ra = np.hypot(a[:, 0] - ax, a[:, 1] - ay)
    rb = np.hypot(b[:, 0] - ax, b[:, 1] - ay)
    return np.sign(rb - ra).astype(int)


def _tangents(u):
    np = _np()
    ref = np.array([0.0, 0.0, 1.0]) if abs(u[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    t1 = np.cross(u, ref)
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(u, t1)
    return t1, t2


def segment_rays(a, b, rule=None):
    """Sample positions t (along a to b, 0 at a, length L at b, extended) and the unit direction."""
    np = _np()
    rule = rule or RULE
    a, b = np.asarray(a, float), np.asarray(b, float)
    L = float(np.linalg.norm(b - a))
    if not L > 0:
        raise PairError("the two points coincide")
    u = (b - a) / L
    E, s = float(rule["extend_vox"]), float(rule["step_vox"])
    t = np.arange(-E, L + E + s / 2, s)
    return t, u, L


def five_rays(a, u, t, lat):
    """Points (5, len(t), 3): the centre ray and four parallel ones at +-lat on two tangents."""
    np = _np()
    t1, t2 = _tangents(u)
    shifts = [np.zeros(3), lat * t1, -lat * t1, lat * t2, -lat * t2]
    return np.stack([a + s + t[:, None] * u for s in shifts])


def runs(mask, t, merge_gap=0.0):
    """Runs of True in a 1-D mask: list of (centre, width) in units of t, merging runs whose gap is shorter
    than merge_gap."""
    np = _np()
    m = np.asarray(mask, bool)
    d = np.diff(np.concatenate([[0], m.astype(np.int8), [0]]))
    st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1
    step = float(t[1] - t[0]) if len(t) > 1 else 1.0
    out = []
    for s_, e_ in zip(st, en):
        if out and t[s_] - out[-1][1] - step < merge_gap:      # length of the empty stretch between them
            out[-1][1] = t[e_]
        else:
            out.append([t[s_], t[e_]])
    return [(0.5 * (lo + hi), hi - lo + step) for lo, hi in out]


def peaks(y, t, min_prominence):
    """Local maxima of a 1-D profile with at least min_prominence (the drop to the higher of the two
    neighbouring minima on each side); returns (positions, prominences)."""
    np = _np()
    y = np.asarray(y, float)
    n = len(y)
    pos, prom = [], []
    finite = np.isfinite(y)
    for i in range(1, n - 1):
        if not (finite[i] and finite[i - 1] and finite[i + 1]):
            continue
        if y[i] > y[i - 1] and y[i] >= y[i + 1]:
            # walk left and right to the lowest point before a higher value
            lo_l = y[i]
            j = i - 1
            while j >= 0 and finite[j] and y[j] <= y[i]:
                lo_l = min(lo_l, y[j])
                j -= 1
            lo_r = y[i]
            j = i + 1
            while j < n and finite[j] and y[j] <= y[i]:
                lo_r = min(lo_r, y[j])
                j += 1
            p = y[i] - max(lo_l, lo_r)
            if p >= min_prominence:
                pos.append(t[i])
                prom.append(p)
    return np.asarray(pos), np.asarray(prom)


def _smooth(y, sigma_samples):
    np = _np()
    if sigma_samples <= 0:
        return y
    r = int(math.ceil(3 * sigma_samples))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma_samples) ** 2)
    k /= k.sum()
    good = np.isfinite(y)
    num = np.convolve(np.where(good, y, 0.0), k, mode="same")
    den = np.convolve(good.astype(float), k, mode="same")
    out = np.where(den > 0.5, num / np.maximum(den, 1e-12), np.nan)
    return out


def line_pitch(centres, min_gap):
    np = _np()
    if len(centres) < 2:
        return float("nan")
    g = np.diff(np.sort(centres))
    g = g[g >= min_gap]
    return float(np.median(g)) if len(g) else float("nan")


def snap(centres, at, pitch, rule):
    """Index of the centre nearest `at` within the snap distance, or None; and that distance."""
    np = _np()
    if len(centres) == 0:
        return None, float("nan")
    e = rule["snap_pitch"] * pitch if math.isfinite(pitch) else rule["snap_default_vox"]
    d = np.abs(np.asarray(centres) - at)
    i = int(np.argmin(d))
    return (i if d[i] <= e else None), float(d[i])


def count_between(centres, L, rule):
    """(count, details) of sheets from the one at t=0 to the one at t=L on a line whose sheet centres are
    `centres` (sorted): index difference after snapping both ends; plus the gap-corrected count."""
    np = _np()
    c = np.sort(np.asarray(centres, float))
    p = line_pitch(c, rule["min_gap_vox"])
    ia, da = snap(c, 0.0, p, rule)
    ib, db = snap(c, L, p, rule)
    det = {"pitch": p, "snap_a": da, "snap_b": db, "n_lines": int(len(c))}
    if ia is None or ib is None or ib < ia:
        return None, None, det
    n = int(ib - ia)
    if not math.isfinite(p) or n == 0:
        return n, (n if n == 0 else None), det
    g = np.diff(c[ia:ib + 1])
    q = g / p
    corr = int(np.sum(np.maximum(1, np.rint(q))))
    det["irregularity"] = float(np.max(np.abs(q - np.maximum(1, np.rint(q)))))
    det["max_gap_pitches"] = float(np.max(q))
    return n, corr, det


class Inputs:
    """The fields a pair is counted on: m7 (sheet values), cos (0 to 1), grad_mag (decoded density per base
    voxel), the axis, and base_per_query (base-frame voxels per query voxel, for the grad_mag path length)."""

    def __init__(self, m7=None, cos=None, gm=None, axis=None, base_per_query=1.0, k_s=None):
        self.m7, self.cos, self.gm, self.axis = m7, cos, gm, axis
        self.base_per_query = float(base_per_query)
        self.k_s = k_s


def sample_pairs(inp, A, B, rule=None, batch=48):
    """Profiles of many pairs, read field by field in batches so each chunk is fetched once per batch. Returns
    one dict per pair: t, L, m7 (sheet mask, OR over five rays), cos (mean over five rays), gm (integral per
    parallel ray, NaN where a ray leaves the data), outside, sign, subtended, radial_cos. Counting from these is
    `count_from_samples`; keeping them lets development try counting settings without reading again."""
    np = _np()
    rule = rule or RULE
    A, B = np.asarray(A, float).reshape(-1, 3), np.asarray(B, float).reshape(-1, 3)
    out = []
    for lo in range(0, len(A), batch):
        geo, m7p, cosp, gmp = [], [], [], []
        for a, b in zip(A[lo:lo + batch], B[lo:lo + batch]):
            t, u, L = segment_rays(a, b, rule)
            g = {"t": t, "L": L, "sign": 0, "subtended": float("nan"), "radial_cos": 0.0}
            if inp.axis is not None:
                g["sign"] = int(radial_sign(inp.axis, a, b)[0])
                g["subtended"] = float(subtended_deg(inp.axis, a, b)[0])
                mid = 0.5 * (a + b)
                ax, ay = inp.axis.at(mid[2])
                r = np.array([mid[0] - ax, mid[1] - ay, 0.0])
                g["radial_cos"] = float(abs(r @ u) / max(np.linalg.norm(r), 1e-12))
            geo.append(g)
            m7p.append(five_rays(a, u, t, float(rule["ray_lateral_vox"])))
            d = b - a
            perp = np.array([-d[1], d[0], 0.0])
            perp = perp / np.linalg.norm(perp) if np.linalg.norm(perp) > 1e-9 else np.array([1.0, 0.0, 0.0])
            n = max(2, int(math.ceil(L / rule["step_vox"])) + 1)
            tt = np.linspace(0.0, 1.0, n)
            offs = np.linspace(-rule["gm_offset_vox"], rule["gm_offset_vox"], int(rule["gm_rays"]))
            gmp.append(np.stack([a + o * perp + tt[:, None] * d for o in offs]))
        sizes = [p.shape[:2] for p in m7p]
        flat = np.concatenate([p.reshape(-1, 3) for p in m7p])
        m7v = inp.m7.nearest(flat) if inp.m7 is not None else None
        cosv = inp.cos.linear(flat) if inp.cos is not None else None
        gsz = [p.shape[:2] for p in gmp]
        gmv = inp.gm.linear(np.concatenate([p.reshape(-1, 3) for p in gmp])) if inp.gm is not None else None
        o1 = o2 = 0
        for g, sz, gs in zip(geo, sizes, gsz):
            k1, k2 = sz[0] * sz[1], gs[0] * gs[1]
            if m7v is not None:
                v = m7v[o1:o1 + k1].reshape(sz)
                g["outside"] = float(np.isnan(v[0]).mean())
                g["m7"] = (np.nan_to_num(v, nan=0.0) >= rule["sheet_threshold"]).any(0)
            if cosv is not None:
                v = cosv[o1:o1 + k1].reshape(sz)
                cnt = np.isfinite(v).sum(0)
                g["cos"] = np.where(cnt > 0, np.nansum(v, axis=0) / np.maximum(cnt, 1), np.nan).astype(np.float32)
            if gmv is not None:
                v = gmv[o2:o2 + k2].reshape(gs)
                seg = g["L"] * inp.base_per_query / (gs[1] - 1)
                g["gm"] = (0.5 * (v[:, :-1] + v[:, 1:]) * seg).sum(1)      # NaN where a ray leaves the data
            o1 += k1
            o2 += k2
            out.append(g)
    return out


def count_from_samples(s, rule=None, k_s=None):
    """Every counter for one pair from its samples: a dict with m7, gap, cos, gm (scaled and rounded), gm_x,
    sign, subtended, radial_cos, outside and the details the features use."""
    np = _np()
    rule = rule or RULE
    t, L = s["t"], s["L"]
    out = {k: None for k in COUNTERS}
    out.update({"gm_raw": None, "sign": s.get("sign", 0), "subtended": s.get("subtended", float("nan")),
                "radial_cos": s.get("radial_cos", 0.0), "outside": s.get("outside", 1.0), "length": L})
    if s.get("m7") is not None:
        rr = runs(s["m7"], t, rule["merge_gap_vox"])
        n, corr, det = count_between([c for c, _ in rr], L, rule)
        out["m7"], out["gap"] = n, corr
        widths = np.array([w for _, w in rr])
        det["width_ratio"] = float(widths.max() / np.median(widths)) if len(widths) else float("nan")
        out["m7_detail"] = det
        # the pitch baseline: segment length over the line's own pitch
        out["pitch_x"] = L / det["pitch"] if math.isfinite(det["pitch"]) and det["pitch"] > 0 else None
    if s.get("cos") is not None:
        prof = _smooth(np.asarray(s["cos"], float), rule["cos_smooth_vox"] / rule["step_vox"])
        pk, pr = peaks(prof, t, rule["cos_min_prominence"])
        n, _, det = count_between(pk, L, rule)
        out["cos"] = n
        half = 0.5 * det["pitch"] if math.isfinite(det["pitch"]) else 0.0
        inner = (pk > -half) & (pk < L + half)
        det["min_prominence"] = float(pr[inner].min()) if inner.any() else float("nan")
        out["cos_detail"] = det
    if s.get("gm") is not None:
        g = np.asarray(s["gm"], float)
        ok = np.isfinite(g)
        if ok.any():
            val = float(np.median(g[ok]))
            out["gm_raw"] = val
            out["gm_complete"] = bool(ok.all())
            if k_s:
                out["gm_x"] = k_s * val
                out["gm"] = int(round(k_s * val))
    return out
