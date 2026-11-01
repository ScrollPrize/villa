"""The papyrus bands on one CT profile and the faces of the band nearest a point.

A profile is a row of intensities sampled along a surface normal at evenly spaced offsets t (voxels) from a vertex.
The profile is smoothed lightly and cut at a level half way between its 10th and 90th percentiles; the runs above
the level are papyrus bands (dips shorter than `fill` closed, runs shorter than `min_band` dropped), and each face
is the sub-sample crossing of the level. The band nearest t = 0 is the one the vertex snaps to.

Recto side: rolled papyrus is written on the inside, so the recto face is the face toward the scroll axis.
`recto_sign` is the sign of the normal's component along the outward radial direction: +1 means t grows outward,
so the recto face is the band's lower face.
"""

import math

import numpy as np


def smooth(p, sigma):
    """Gaussian smoothing along the last axis, sigma in samples, edges reflected."""
    p = np.asarray(p, dtype=np.float64)
    if sigma <= 0:
        return p.copy()
    r = max(1, int(4.0 * sigma + 0.5))
    x = np.arange(-r, r + 1)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    pad = [(0, 0)] * (p.ndim - 1) + [(r, r)]
    q = np.pad(p, pad, mode="reflect")
    out = np.zeros_like(p)
    n = p.shape[-1]
    for j, w in enumerate(k):
        out += w * q[..., j:j + n]
    return out


FACE_DEFAULTS = {
    "sigma": 0.5,        # smoothing, in t units (voxels of the label frame)
    "lo_pct": 10.0,      # percentile taken as the gap level
    "hi_pct": 90.0,      # percentile taken as the papyrus level
    "level": 0.5,        # face level between them
    "fill": 1.0,         # close dips shorter than this (t units): partial dips between fiber layers
    "min_band": 1.0,     # drop runs shorter than this (t units): specks
    "ambiguous": 2.0,    # second nearest band within this much farther than the nearest: ambiguous
    "split_sigma": 0.25, # finer smoothing for the split check
    "split_len": 1.0,    # a dip below the level at least this long inside the band (fine profile) splits it
}


def _runs(b):
    """(start, end) index pairs of the True runs of a 1-D boolean array, end exclusive."""
    d = np.diff(np.concatenate([[0], b.astype(np.int8), [0]]))
    return list(zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]))


def _crossing(t, p, i0, i1, level):
    """Sub-sample t where p crosses level between samples i0 and i1 (linear)."""
    a, b = p[i0], p[i1]
    if b == a:
        return 0.5 * (t[i0] + t[i1])
    f = (level - a) / (b - a)
    return float(t[i0] + min(max(f, 0.0), 1.0) * (t[i1] - t[i0]))


def bands(t, p, opts=None):
    """Papyrus bands of one profile: (list of (lower face, upper face)), level, contrast. A run touching either end
    of the profile has no face there, which is recorded as nan."""
    o = dict(FACE_DEFAULTS)
    o.update(opts or {})
    t = np.asarray(t, dtype=np.float64)
    dt = float(t[1] - t[0])
    ps = smooth(np.asarray(p, dtype=np.float64), o["sigma"] / dt)
    lo, hi = np.percentile(ps, [o["lo_pct"], o["hi_pct"]])
    level = lo + o["level"] * (hi - lo)
    above = ps >= level
    n = len(ps)
    # close short dips inside papyrus
    for s, e in _runs(~above):
        if s > 0 and e < n and (e - s) * dt < o["fill"]:
            above[s:e] = True
    out = []
    for s, e in _runs(above):
        if (e - s) * dt < o["min_band"]:
            continue
        lower = _crossing(t, ps, s - 1, s, level) if s > 0 else float("nan")
        upper = _crossing(t, ps, e - 1, e, level) if e < n else float("nan")
        out.append((lower, upper))
    return out, float(level), float(hi - lo)


def sheet_faces(t, p, recto_sign, at=0.0, opts=None):
    """The band nearest `at` on one truth profile and its faces. Returns a dict with recto, verso, mid, thick
    (t units, nan when a face is missing), dist (from `at` to the band, 0 inside), second (distance to the next
    nearest band), gap_recto and gap_verso (clear gap between that face and the next band on its side, nan when the
    profile ends first), split (the finer profile dips below the level for at least split_len inside the band, so
    at full resolution the band is two or more layers), contrast, and ok (a face pair exists and the choice is
    not ambiguous)."""
    o = dict(FACE_DEFAULTS)
    o.update(opts or {})
    bl, level, contrast = bands(t, p, o)
    res = {"recto": float("nan"), "verso": float("nan"), "mid": float("nan"), "thick": float("nan"),
           "dist": float("nan"), "second": float("nan"), "gap_recto": float("nan"), "gap_verso": float("nan"),
           "split": False, "contrast": contrast, "n_bands": len(bl), "ok": False, "why": ""}
    if not bl:
        res["why"] = "no band"
        return res
    t0, t1 = float(t[0]), float(t[-1])

    def dist(b):
        lo = b[0] if math.isfinite(b[0]) else t0
        hi = b[1] if math.isfinite(b[1]) else t1
        return 0.0 if lo <= at <= hi else min(abs(at - lo), abs(at - hi))

    d = np.array([dist(b) for b in bl])
    order = np.argsort(d, kind="stable")
    k = int(order[0])
    lower, upper = bl[k]
    res["dist"] = float(d[k])
    res["second"] = float(d[order[1]]) if len(bl) > 1 else float("inf")
    below = [b[1] for b in bl[:k]]                   # upper faces of the bands below
    above = [b[0] for b in bl[k + 1:]]               # lower faces of the bands above
    gap_lo = float(lower - below[-1]) if below and math.isfinite(below[-1]) else float("nan")
    gap_hi = float(above[0] - upper) if above and math.isfinite(above[0]) else float("nan")
    if recto_sign > 0:
        recto, verso = lower, upper
        res["gap_recto"], res["gap_verso"] = gap_lo, gap_hi
    else:
        recto, verso = upper, lower
        res["gap_recto"], res["gap_verso"] = gap_hi, gap_lo
    res["recto"], res["verso"] = float(recto), float(verso)
    if math.isfinite(lower) and math.isfinite(upper):
        res["mid"] = 0.5 * (lower + upper)
        res["thick"] = upper - lower
        dt = float(t[1] - t[0])
        fine = smooth(np.asarray(p, dtype=np.float64), o["split_sigma"] / dt)
        inside = (t > lower) & (t < upper)
        low = inside & (fine < level)
        res["split"] = any((e - s) * dt >= o["split_len"] for s, e in _runs(low))
    if not (math.isfinite(lower) and math.isfinite(upper)):
        res["why"] = "band runs off the profile"
    elif res["second"] - res["dist"] < o["ambiguous"]:
        res["why"] = "two bands about as near"
    else:
        res["ok"] = True
    return res

