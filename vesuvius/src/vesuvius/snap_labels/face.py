"""Mesh normals, grid filters and the face snap: move each vertex of a tifxyz grid along its normal onto a face of
the nearest bright CT band, with a reason per vertex when it cannot.

The filters reproduce scipy.ndimage.gaussian_filter (mode reflect, truncate 4) and numpy's nan-median bit for bit
in plain numpy, so the snap needs no scipy.
"""

import math

import numpy as np

from . import band

DEFAULT_BATCH = 4_000_000
FACE_SOURCE_NOTE = (" + vesuvius.snap_labels (offset along the normal to a face or the midline of the nearest CT "
                    "band, sub-voxel)")


class SnapError(ValueError):
    pass


def normals(xs, ys, zs, valid):
    """Normals as in Erwin Nieuwlaar's snap_meshes.py: central differences where both neighbors are valid, else one-sided, n = cross(d/dcol, d/drow)
    normalized (float32), and ok = a finite normal on a valid vertex."""
    P = np.stack([xs, ys, zs], -1).astype(np.float64)
    P[~valid] = np.nan

    def grad(axis):
        g = np.full_like(P, np.nan)
        fwd = np.roll(P, -1, axis) - P
        bwd = P - np.roll(P, 1, axis)
        n = P.shape[axis]
        idx = [slice(None)] * 3
        idx[axis] = slice(n - 1, n)
        fwd[tuple(idx)] = np.nan
        idx[axis] = slice(0, 1)
        bwd[tuple(idx)] = np.nan
        both = np.isfinite(fwd).all(-1) & np.isfinite(bwd).all(-1)
        g[both] = 0.5 * (fwd[both] + bwd[both])
        only_f = np.isfinite(fwd).all(-1) & ~both
        g[only_f] = fwd[only_f]
        only_b = np.isfinite(bwd).all(-1) & ~both
        g[only_b] = bwd[only_b]
        return g

    tu = grad(1)
    tv = grad(0)
    n = np.cross(tu, tv)
    nn = np.linalg.norm(n, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        n = n / np.where(nn > 1e-6, nn, np.nan)
    ok = np.isfinite(n).all(-1) & valid
    return n.astype(np.float32), ok


def gaussian_kernel1d(sigma, radius):
    """scipy.ndimage._gaussian_kernel1d for order 0."""
    x = np.arange(-radius, radius + 1)
    phi = np.exp(-0.5 / (sigma * sigma) * x ** 2)
    return phi / phi.sum()


def _reflect_index(n, r):
    """Indices of a length-n axis padded by r on both sides in scipy's 'reflect' mode (d c b a | a b c d | d c b a,
    repeated with period 2n)."""
    i = np.mod(np.arange(-r, n + r), 2 * n)
    return np.where(i >= n, 2 * n - 1 - i, i)


def gaussian_filter(a, sigma, truncate=4.0):
    """scipy.ndimage.gaussian_filter(a, sigma) with mode 'reflect' and truncate 4.0, in numpy: one 1-D pass per axis
    in axis order, each in float64 and rounded to the input dtype, the symmetric sum taken in scipy's order (center
    first, then the outermost pair inwards)."""
    a = np.asarray(a)
    out = a.copy()
    sigma = float(sigma)
    if sigma <= 1e-15:
        return out
    r = int(truncate * sigma + 0.5)
    w = gaussian_kernel1d(sigma, r)[::-1]
    for axis in range(a.ndim):
        n = out.shape[axis]
        if n == 0:
            continue
        x = np.take(out.astype(np.float64), _reflect_index(n, r), axis=axis)

        def sl(k, axis=axis, n=n):
            s = [slice(None)] * a.ndim
            s[axis] = slice(k, k + n)
            return tuple(s)

        acc = x[sl(r)] * w[r]
        for j in range(r, 0, -1):
            acc = acc + (x[sl(r - j)] + x[sl(r + j)]) * w[r - j]
        out = acc.astype(a.dtype)
    return out


def nanmedian_filter(d, size=5):
    """A nan-aware median over a size x size window, from a stack of shifted copies (nan outside the grid).

    The same values, bit for bit, as np.nanmedian(stack, axis=0), which for a stack shallower than 600 layers
    (size up to 24) goes through np.ma.median: sort the stack (nan last), take the middle pair of the k values that
    are not nan (+-inf count), sum the pair with a reduction as np.ma.median does (it adds 0.0 first, so -0.0 comes
    out as 0.0) and halve it; nan where k is 0. A deeper stack takes another path inside numpy (np.median along
    each line), so it is left to np.nanmedian itself."""
    H, W = d.shape
    r = size // 2
    pad = np.pad(d, r, constant_values=np.nan)
    st = np.stack([pad[i:i + H, j:j + W] for i in range(size) for j in range(size)], 0)
    if size * size >= 600:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmedian(st, axis=0)
    k = (~np.isnan(st)).sum(0)
    st.sort(axis=0)
    pair = np.concatenate((np.take_along_axis(st, np.maximum((k - 1) // 2, 0)[None], 0),
                           np.take_along_axis(st, (k // 2)[None], 0)), 0)
    with np.errstate(invalid="ignore", over="ignore"):
        med = pair.sum(0)
        np.true_divide(med, 2.0, casting="unsafe", out=med)
    med[k == 0] = np.nan
    return med


def normconv(d, sigma):
    """Normalized convolution: Gaussian of the finite values over the Gaussian of their mask, nan where the mask
    weight is at most 1e-3."""
    m = np.isfinite(d).astype(np.float32)
    v = np.where(np.isfinite(d), d, 0).astype(np.float32)
    num = gaussian_filter(v, sigma)
    den = gaussian_filter(m, sigma)
    out = np.full_like(d, np.nan)
    okk = den > 1e-3
    out[okk] = num[okk] / den[okk]
    return out


def _radial(verts, axis):
    """Unit radial directions from the scroll axis and their lengths before normalizing (0 without an axis)."""
    p = verts.astype(np.float64)
    if axis is None:
        return np.zeros_like(p), np.zeros(len(p))
    if axis[0] == "umbilicus":
        pts = axis[1]
        zs = np.array([q[0] for q in pts], dtype=np.float64)
        cy = np.interp(p[:, 2], zs, np.array([q[1] for q in pts], dtype=np.float64))
        cx = np.interp(p[:, 2], zs, np.array([q[2] for q in pts], dtype=np.float64))
        radial = np.stack([p[:, 0] - cx, p[:, 1] - cy, np.zeros(len(p))], 1)
    else:
        point = np.asarray(axis[1], dtype=np.float64)
        ad = np.asarray(axis[2], dtype=np.float64)
        al = float(np.linalg.norm(ad))
        ad = np.array([1.0, 0.0, 0.0]) if al < 1e-12 else ad / al
        dvec = p - point
        radial = dvec - (dvec @ ad)[:, None] * ad
    rl = np.linalg.norm(radial, axis=1)
    radial = np.where(rl[:, None] > 1e-9, radial / np.where(rl > 1e-9, rl, 1)[:, None], radial)
    return radial, rl


def _profile(sampler, P, Dd, offs, batch):
    """Trilinear CT values at P + offs * Dd: offs (M,) shared or (V, M) per vertex. Returns (V, M), nan missing."""
    V = len(P)
    offs = np.asarray(offs, dtype=np.float64)
    M = offs.shape[-1]
    out = np.full((V, M), np.nan)
    per = max(1, int(batch) // max(1, 8 * M))
    for s in range(0, V, per):
        e = min(V, s + per)
        o = offs[None, :] if offs.ndim == 1 else offs[s:e]
        q = P[s:e, None, :] + o[:, :, None] * Dd[s:e, None, :]
        out[s:e] = sampler.trilinear(q.reshape(-1, 3)).reshape(e - s, M)
    return out


def regularize(D, K, jump=6.0, sigma=1.0):
    """Regularization of an offset grid (snap_meshes.py's rules) D (nan = no sheet): (D3, filled, rejected)."""
    med = nanmedian_filter(D, 5)
    with np.errstate(invalid="ignore"):
        rej = np.isfinite(D) & np.isfinite(med) & (np.abs(D - med) > jump)
    D2 = D.copy()
    D2[rej] = np.nan
    filled = normconv(D2, max(sigma, 1.0))
    D3 = np.where(np.isfinite(D2), D2, filled)
    D3 = normconv(D3, sigma)
    D3 = np.clip(D3, -K, K)
    return D3, filled, rej


def _band_at_level(t, ps, level, at=0.0, ambiguous=2.0):
    """(lower, upper, ok) of the run of ps at or above level nearest `at` (0 inside it), faces by linear
    crossings; ok False when the run reaches an end of the profile or a second run is within `ambiguous` as near."""
    n = len(ps)
    runs = band._runs(ps >= level)
    if not runs:
        return float("nan"), float("nan"), False
    fb = []
    for s, e in runs:
        lo = band._crossing(t, ps, s - 1, s, level) if s > 0 else float("nan")
        hi = band._crossing(t, ps, e - 1, e, level) if e < n else float("nan")
        lo_d = lo if math.isfinite(lo) else float(t[0])
        hi_d = hi if math.isfinite(hi) else float(t[-1])
        fb.append((0.0 if lo_d <= at <= hi_d else min(abs(at - lo_d), abs(at - hi_d)), lo, hi))
    fb.sort(key=lambda r: r[0])
    d0, lo, hi = fb[0]
    ok = math.isfinite(lo) and math.isfinite(hi)
    if ok and len(fb) > 1 and fb[1][0] - d0 < ambiguous:
        ok = False
    return lo, hi, ok


FACE_SNAP_DEFAULTS = {
    "target": "recto",     # "recto" (the band face toward the scroll axis), "mid" (the band midline) or "verso"
    "half": 12.0,          # profile half-length along the normal, voxels (the level is set from this window)
    "step": 0.25,          # profile sample step, voxels
    "max_move": 4.0,       # a band farther than this from the vertex abstains (0 when the vertex is inside it)
    "level": None,         # None: per-profile level (band.sheet_faces); a number: one global level
    "min_radial": 0.3,     # recto or verso target: |normal . radial| below this abstains (side not named)
    "jump": 2.0,           # offsets this far from their 5 x 5 nan-median are rejected (regularize)
    "sigma": 0.0,          # smoothing of the offset field, grid cells (0: none beyond filling holes)
    "min_support": 3,      # an offset needs this many measured offsets in its 5 x 5 window (itself included), and
                           # a hole this many around it before it is filled; a lone offset is neither trusted nor spread
    "min_contrast": None,  # a band whose profile contrast (90th minus 10th percentile) is below this abstains, CT units
    "contrast_frac": 0.25, # without min_contrast: the floor is this fraction of the 1st to 99th percentile window of
                           # every profile sample of the mesh (so a vertex in a wide gap or in air does not snap to noise)
    "max_thick": None,     # a band thicker than this abstains, voxels (None: no absolute cap)
    "max_thick_ratio": None,  # a band thicker than this times the reference thickness abstains (None: off): two
                           # sheets pressed closer than the blur look like one band about twice as thick, whose faces
                           # belong to different windings, and from the profile alone it looks like a clean band
    "thick_ref_pct": 25.0, # the reference thickness: this percentile of the mesh's measured band thicknesses (a low
                           # one, so it stays a single sheet's thickness while up to most of the mesh is merged)
    "thick_min_bands": 10, # the ratio rule needs at least this many measured bands, else it is off
}


FACE_SNAP_PER_VERTEX = {"jump": 1e9, "min_support": 1}


FACE_SNAP_PER_VERTEX_MERGED = {"jump": 1e9, "min_support": 1, "max_thick_ratio": 1.8}


FACE_REASONS = {0: "measured", 1: "missing data", 2: "side not named", 3: "low contrast", 4: "no band",
                5: "ambiguous or off the profile", 6: "too far", 7: "too thick", 8: "rejected by its neighbors",
                9: "not a valid vertex"}


_FACE_REASON_CODE = {v: k for k, v in FACE_REASONS.items()}


def _finite_count(D, size=5):
    """Number of finite values in each size x size window of a grid (outside the grid counts as missing)."""
    r = size // 2
    H, W = D.shape
    m = np.pad(np.isfinite(D), r).astype(np.int32)
    return sum(m[i:i + H, j:j + W] for i in range(size) for j in range(size))


def snap_face(xs, ys, zs, valid, sampler, axis=None, recto_sign=None, opts=None, face_opts=None,
              batch=DEFAULT_BATCH):
    """Technique D on one mesh: move each vertex along its normal onto a face (or the midline) of the bright CT band
    nearest it, found at sub-voxel precision on a trilinear profile, then regularize the offsets over the grid: offsets
    more than `jump` from their 5 x 5 nan-median are rejected, an offset with fewer than min_support measured offsets
    in its 5 x 5 window is dropped (so one vertex cannot hop to a neighbor sheet alone, nor drag abstaining
    neighbors with it), and holes with enough measured neighbors are filled by normalized convolution; with
    sigma > 0 the field is also smoothed (`regularize`).

    The band finder is band.sheet_faces (level half way between the profile's 10th and 90th percentiles
    after light smoothing; `face_opts` overrides its FACE_DEFAULTS), or with opts["level"] a fixed global level (the
    threshold at the void boundary). The recto face is the face toward the scroll axis: `axis` (("umbilicus",
    points) or ("line", point, direction)), or `recto_sign` (+1 when the recto face lies at negative offsets along the normal, -1 when
    at positive ones; a scalar or a grid). A vertex abstains (keeps nan before filling) when the band is ambiguous or
    runs off the profile, when its contrast is below a floor (min_contrast, or contrast_frac of the mesh's own
    window), when the band is more than max_move away, when the band is thicker than max_thick or than
    max_thick_ratio times a low percentile of the mesh's band thicknesses (two sheets merged by the blur), or, for a
    face target, when its normal is nearly tangent to the radial direction. Returns a dict: the snapped grids,
    offsets, report, record and per-vertex detail."""
    o = dict(FACE_SNAP_DEFAULTS)
    o.update(opts or {})
    if o["target"] not in ("recto", "mid", "verso"):
        raise SnapError(f"face target must be recto, mid or verso, not {o['target']!r}")
    n, ok = normals(xs, ys, zs, valid)
    vy, vx = np.nonzero(ok)
    nv = len(vy)
    base = np.stack([xs[vy, vx], ys[vy, vx], zs[vy, vx]], -1).astype(np.float64)
    nvx = n[vy, vx].astype(np.float64)
    if recto_sign is not None:
        rs = np.broadcast_to(np.asarray(recto_sign, dtype=np.float64), xs.shape)[vy, vx].copy()
        side_ok = np.abs(rs) > 0
        rs = np.sign(rs)
    elif axis is not None:
        radial, rl = _radial(base, axis)
        dot = (nvx * radial).sum(1)
        side_ok = (rl > 1e-9) & (np.abs(dot) >= o["min_radial"])
        rs = np.where(dot >= 0, 1.0, -1.0)            # +1: t grows outward, so the recto face is the lower face
    else:
        rs = np.ones(nv)
        side_ok = np.zeros(nv, bool)
    if o["target"] == "mid":
        side_ok = np.ones(nv, bool)
    half, step = float(o["half"]), float(o["step"])
    t = np.arange(-half, half + 0.5 * step, step)
    prof = _profile(sampler, base, nvx, t, batch) if nv else np.zeros((0, len(t)))
    fo = dict(band.FACE_DEFAULTS)
    fo.update(face_opts or {})
    if o["min_contrast"] is not None:
        floor = float(o["min_contrast"])
    else:
        fin = prof[np.isfinite(prof)]
        floor = float(o["contrast_frac"]) * float(np.subtract(*np.percentile(fin, [99, 1]))) if fin.size else 0.0
    d = np.full(nv, np.nan)
    thick = np.full(nv, np.nan)
    code = np.zeros(nv, np.uint8)
    vcon = np.full(nv, np.nan)            # profile contrast (90th minus 10th percentile after smoothing)
    vsep = np.full(nv, np.nan)            # second-nearest band distance minus the nearest's (inf with one band)
    why = {"no band": 0, "ambiguous or off the profile": 0, "too far": 0, "side not named": 0, "missing data": 0,
           "low contrast": 0, "too thick": 0}

    def skip(i, reason):
        why[reason] += 1
        code[i] = _FACE_REASON_CODE[reason]

    for i in range(nv):
        p = prof[i]
        if not np.isfinite(p).all():
            skip(i, "missing data")
            continue
        if not side_ok[i]:
            skip(i, "side not named")
            continue
        if o["level"] is None:
            r = band.sheet_faces(t, p, rs[i], at=0.0, opts=fo)
            vcon[i] = r["contrast"]
            if r["contrast"] < floor:
                skip(i, "low contrast")
                continue
            if not r["ok"]:
                skip(i, "no band" if r["why"] == "no band" else "ambiguous or off the profile")
                continue
            vsep[i] = r["second"] - r["dist"]
            lo, hi = (r["recto"], r["verso"]) if rs[i] > 0 else (r["verso"], r["recto"])
        else:
            ps = band.smooth(p, fo["sigma"] / step)
            vcon[i] = float(np.subtract(*np.percentile(ps, [90, 10])))
            if vcon[i] < floor:
                skip(i, "low contrast")
                continue
            lo, hi, good = _band_at_level(t, ps, float(o["level"]), 0.0, fo["ambiguous"])
            if not good:
                skip(i, "ambiguous or off the profile")
                continue
        recto, verso = (lo, hi) if rs[i] > 0 else (hi, lo)
        tgt = {"recto": recto, "verso": verso, "mid": 0.5 * (lo + hi)}[o["target"]]
        if (0.0 if lo <= 0.0 <= hi else min(abs(lo), abs(hi))) > o["max_move"]:
            skip(i, "too far")
            continue
        d[i] = tgt
        thick[i] = hi - lo
    cap = float("inf")
    if o.get("max_thick") is not None:
        cap = float(o["max_thick"])
    ref = np.isfinite(thick)
    ref_thick = (float(np.percentile(thick[ref], float(o.get("thick_ref_pct", 25.0)))) if ref.any()
                 else float("nan"))
    if o.get("max_thick_ratio") is not None and ref.sum() >= int(o.get("thick_min_bands", 10)):
        cap = min(cap, float(o["max_thick_ratio"]) * ref_thick)
    with np.errstate(invalid="ignore"):
        fat = np.isfinite(d) & (thick > cap)
    why["too thick"] = int(fat.sum())
    d[fat] = np.nan
    code[fat] = _FACE_REASON_CODE["too thick"]
    D = np.full(xs.shape, np.nan, np.float32)
    D[vy, vx] = d
    lone = np.isfinite(D) & (_finite_count(D, 5) < int(o["min_support"]))
    D[lone] = np.nan                      # a lone offset cannot be checked against its neighbors
    K = float(o["max_move"]) + half       # a guard only: a target is never farther than this
    if o["sigma"] > 0:
        D3, filled, rej = regularize(D, K, o["jump"], o["sigma"])
        med = nanmedian_filter(D, 5)
        with np.errstate(invalid="ignore"):
            D2 = np.where(np.isfinite(D) & np.isfinite(med) & (np.abs(D - med) > o["jump"]), np.nan, D)
    else:
        med = nanmedian_filter(D, 5)
        with np.errstate(invalid="ignore"):
            rej = np.isfinite(D) & np.isfinite(med) & (np.abs(D - med) > o["jump"])
        D2 = D.copy()
        D2[rej] = np.nan
        filled = normconv(D2, 1.0)
        D3 = np.clip(np.where(np.isfinite(D2), D2, filled), -K, K)
    rej = rej | lone
    supported = np.isfinite(D2) | (_finite_count(D2, 5) >= int(o["min_support"]))
    apply = ok & np.isfinite(D3) & np.isfinite(filled) & supported
    xo, yo, zo = xs.copy(), ys.copy(), zs.copy()
    xo[apply] += D3[apply] * n[apply][:, 0]
    yo[apply] += D3[apply] * n[apply][:, 1]
    zo[apply] += D3[apply] * n[apply][:, 2]
    nvalid = int(valid.sum())
    has = np.isfinite(d)
    absd = np.abs(D3[apply])
    report = {
        "n_valid": nvalid,
        "frac_with_sheet": float(has.mean()) if nv else None,
        "frac_snapped": float(apply.sum() / max(1, nvalid)),
        "median_abs_off": float(np.nanmedian(absd)) if apply.any() else None,
        "p90_abs_off": float(np.nanpercentile(absd, 90)) if apply.any() else None,
        "frac_rejected": float(rej.sum() / max(1, has.sum())),
        "frac_already_on": float((np.abs(d[has]) <= 2.0).mean()) if has.any() else 0.0,
        "abstained": why,
        "ref_band_thick": ref_thick if math.isfinite(ref_thick) else None,
        "n_bands": int(ref.sum()),
        "thick_cap": cap if math.isfinite(cap) else None,
    }
    record = {"mode": "face", "target": o["target"], "level": o["level"], "max_move": K, "jump": o["jump"],
              "sigma": o["sigma"], "frac_snapped": report["frac_snapped"], "median_abs_off": report["median_abs_off"]}
    if o.get("max_thick") is not None or o.get("max_thick_ratio") is not None:
        record["max_thick"], record["max_thick_ratio"] = o.get("max_thick"), o.get("max_thick_ratio")
    # per-vertex detail (grids of the mesh's shape): the vertex's own measured offset (nan when it measured none or
    # its offset was rejected), why it has none (FACE_REASONS), the profile contrast, band thickness and band
    # separation it saw, and the mesh-wide contrast floor and thickness cap it was held to
    G = xs.shape
    measured = np.full(G, np.nan, np.float32)
    keep = np.isfinite(D2[vy, vx])
    measured[vy[keep], vx[keep]] = D2[vy[keep], vx[keep]]
    reason = np.full(G, 9, np.uint8)
    reason[vy, vx] = code
    reason[vy, vx] = np.where((code == 0) & ~keep, _FACE_REASON_CODE["rejected by its neighbors"], code)
    detail = {"measured": measured, "reason": reason, "contrast": np.full(G, np.nan, np.float32),
              "thick": np.full(G, np.nan, np.float32), "sep": np.full(G, np.nan, np.float32),
              "floor": float(floor), "ref_thick": ref_thick if math.isfinite(ref_thick) else None,
              "cap": cap if math.isfinite(cap) else None, "ambiguous": float(fo["ambiguous"])}
    detail["contrast"][vy, vx] = vcon
    th = np.where(np.isfinite(d) | fat, thick, np.nan)
    detail["thick"][vy, vx] = th
    detail["sep"][vy, vx] = vsep
    return {"xs": xo, "ys": yo, "zs": zo, "valid": valid, "D3": D3, "apply": apply, "report": report,
            "record": record, "note": FACE_SOURCE_NOTE, "detail": detail}


CHECK_MAX_SLOPE = 0.05     # largest slope of the planted field, voxels per voxel (a tilt under 3 degrees)


def planted_field(shape, shift, vary, step_r=1.0, step_c=1.0, max_slope=CHECK_MAX_SLOPE):
    """The known smooth displacement, voxels: shift + vary * sin(2 pi u / Lr) * cos(2 pi v / Lc), with u and v the
    distances in voxels from the grid's center along rows and columns (grid steps step_r, step_c) and each
    wavelength the mesh's extent, or longer so that the slope stays at most max_slope. A gentle field barely tilts
    the normals, so a correct snap returns each vertex to where it was rather than sliding it along the sheet."""
    H, W = shape
    er, ec = max(1.0, H * step_r), max(1.0, W * step_c)
    floor_l = 2 * np.pi * abs(vary) / max_slope if max_slope > 0 else 0.0
    lr, lc = max(er, floor_l), max(ec, floor_l)
    u = (np.arange(H)[:, None] + 0.5) * step_r - er / 2
    v = (np.arange(W)[None, :] + 0.5) * step_c - ec / 2
    return (shift + vary * np.sin(2 * np.pi * u / lr) * np.cos(2 * np.pi * v / lc)).astype(np.float32)


def grid_steps(xs, ys, zs, valid):
    """Median vertex spacing along rows and along columns, voxels (1 when there are no neighbors)."""
    P = np.stack([xs, ys, zs], -1).astype(np.float64)
    out = []
    for axis in (0, 1):
        a = [slice(None)] * 2
        b = [slice(None)] * 2
        a[axis], b[axis] = slice(1, None), slice(None, -1)
        both = valid[tuple(a)] & valid[tuple(b)]
        d = np.linalg.norm(P[tuple(a)] - P[tuple(b)], axis=-1)[both]
        out.append(float(np.median(d)) if len(d) and np.median(d) > 0 else 1.0)
    return out


def _distances(a, b, mask):
    d = np.sqrt(sum((a[k][mask].astype(np.float64) - b[k][mask].astype(np.float64)) ** 2 for k in range(3)))
    return d


def _summary(d, tol):
    if not len(d):
        return {"n": 0, "median": None, "p90": None, "frac_within_tol": 0.0, "max": None}
    return {"n": int(len(d)), "median": round(float(np.median(d)), 4), "p90": round(float(np.percentile(d, 90)), 4),
            "frac_within_tol": round(float((d <= tol).mean()), 4), "max": round(float(d.max()), 4)}


def self_check(xs, ys, zs, valid, sampler, snap, shift=4.0, vary=1.0, tol=1.0, min_frac=0.8, control_shift=3.0,
               reference="original", plant=None, extra_tols=None):
    """Displace, snap, compare; then the shifted and empty controls. `snap(xs, ys, zs, valid, sampler)` returns a
    snap result dict; `plant(xs, ys, zs, valid)` gives the planting directions and where they exist (default: his
    normals). The shifted control reads the volume so that the sheet appears control_shift voxels from the input
    on the side of the planted displacement, where a snap that ignored the data could not tell it from the
    original. `extra_tols` (voxels) adds recovery["frac_within"] {tol: share} for calibrating a bar from one run.
    Returns a JSON-ready dict with "passed"."""
    n, ok = (plant or normals)(xs, ys, zs, valid)
    if not ok.any():
        raise SnapError("no vertex has a normal: nothing to check")
    step_r, step_c = grid_steps(xs, ys, zs, valid)
    field = planted_field(xs.shape, shift, vary, step_r, step_c)
    xd, yd, zd = xs.copy(), ys.copy(), zs.copy()
    xd[ok] += field[ok] * n[ok][:, 0]
    yd[ok] += field[ok] * n[ok][:, 1]
    zd[ok] += field[ok] * n[ok][:, 2]
    orig = (xs, ys, zs)
    disp = (xd, yd, zd)
    if reference == "snapped":
        r0 = snap(xs, ys, zs, valid, sampler)
        ref = (r0["xs"], r0["ys"], r0["zs"])
    else:
        ref = orig
    res = snap(xd, yd, zd, valid, sampler)
    got = (res["xs"], res["ys"], res["zs"])
    dist = _distances(got, ref, ok)
    recovered = _summary(dist, tol)
    if extra_tols:
        recovered["frac_within"] = {str(float(t)): (round(float((dist <= float(t)).mean()), 4) if len(dist) else 0.0)
                                    for t in extra_tols}
    planted = _summary(_distances(disp, ref, ok), tol)
    along = np.abs(sum((got[k][ok].astype(np.float64) - ref[k][ok].astype(np.float64)) * n[ok][:, k]
                       for k in range(3)))
    recovered["along_median"] = round(float(np.median(along)), 4)
    recovered["along_p90"] = round(float(np.percentile(along, 90)), 4)

    def recovers(s):
        return s["median"] is not None and s["median"] <= tol and s["frac_within_tol"] >= min_frac

    mean_n = n[ok].astype(np.float64).mean(axis=0)
    ln = float(np.linalg.norm(mean_n))
    unit = mean_n / ln if ln > 1e-6 else np.array([1.0, 0.0, 0.0])
    # reading at p + v shows the sheet at -v: put it on the planted side
    vec = (-(1.0 if shift >= 0 else -1.0) * control_shift * unit).astype(np.float32)
    rs = snap(xd, yd, zd, valid, sampler.derive(shift=vec))
    d_shift = _distances((rs["xs"], rs["ys"], rs["zs"]), ref, ok)
    shifted = _summary(d_shift, tol)
    if extra_tols:
        shifted["frac_within"] = {str(float(t)): (round(float((d_shift <= float(t)).mean()), 4) if len(d_shift)
                                                  else 0.0) for t in extra_tols}
    re_ = snap(xd, yd, zd, valid, sampler.derive(empty=True))
    moved = _distances((re_["xs"], re_["ys"], re_["zs"]), disp, valid)
    empty_moved = float(moved.max()) if len(moved) else 0.0
    out = {
        "reference": reference,
        "planted": {"shift": shift, "vary": vary, "grid_step": [round(step_r, 4), round(step_c, 4)], **planted},
        "recovery": {**recovered, "tol": tol, "min_frac": min_frac, "passed": recovers(recovered)},
        "control_shifted": {"shift_xyz": [round(float(v), 4) for v in vec], "control_shift": control_shift,
                            **shifted, "recovered": recovers(shifted), "caught": not recovers(shifted)},
        "control_empty": {"max_movement": round(empty_moved, 6), "caught": empty_moved == 0.0},
    }
    out["passed"] = bool(out["recovery"]["passed"] and out["control_shifted"]["caught"]
                         and out["control_empty"]["caught"])
    why = []
    if not out["recovery"]["passed"]:
        why.append(f"the snap did not recover the planted displacement (median {recovered['median']}, "
                   f"{recovered['frac_within_tol']} within {tol} voxels)")
    if not out["control_shifted"]["caught"]:
        why.append("the shifted control also recovered the mesh, so this check cannot tell a right volume from a "
                   "shifted one here")
    if not out["control_empty"]["caught"]:
        why.append(f"the mesh moved by up to {empty_moved:.4g} voxels against a volume with no sheet")
    out["why"] = why or ["recovered, and both controls caught"]
    return out


DEFAULT_CHECK_SHIFT = {"nearest": 4.0, "midline": -2.0, "face": 2.0}

