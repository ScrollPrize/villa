"""Pure-NumPy implementation of the hierarchical backprojector (GHBP), the independent reference for
tests/test_ghbp.py. Not optimised.
"""
import math
import numpy as np


def angle_of_mean(a):
    return math.atan2(np.sin(np.asarray(a, np.float64)).sum(), np.cos(np.asarray(a, np.float64)).sum())


def build_bpsetups(angles_rad, rot_center, axis_source_meters, voxel_size_microns, axis_correction=None):
    """3x4 matrices P @ R(-w), w = -angles_rad[k] (ghbp_cuda.cu:608-647)."""
    d = 1.0e6 * float(np.float32(axis_source_meters)) / float(np.float32(voxel_size_microns))
    t = float(np.float32(rot_center))
    out = np.zeros((len(angles_rad), 3, 4))
    for k, a in enumerate(np.asarray(angles_rad, np.float32)):
        w = -float(a)
        rc = t + (float(axis_correction[k]) if axis_correction is not None else 0.0)
        Pm = np.array([[0.0, -t / d, 1.0, rc], [1.0, 0, 0, 0], [0.0, -1.0 / d, 0.0, 1.0]])
        cw, sw = math.cos(w), math.sin(w)
        Rm = np.array([[1, 0, 0, 0], [0, cw, sw, 0], [0, -sw, cw, 0], [0, 0, 0, 1.0]])
        out[k] = Pm @ Rm
    return out.astype(np.float32).astype(np.float64)


def rot2d(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1.0]])


def ainvt(N, gh, gw, scale):
    T = np.array([[1, 0, -0.5 * (gh - 1)], [0, 1, -0.5 * (gw - 1)], [0, 0, 1.0]])
    A = np.diag([(N - 1) / (gh - 1) * scale, (N - 1) / (gw - 1) * scale, 1.0])
    return A @ T


def build_hierarchy(angles_rad, N, reduction_steps, legs, whf_x=1.0, whf_y=1.0, fac=1.0):
    a = np.asarray(angles_rad, np.float32)
    n = len(a)
    rf = max(1, int(math.ceil(n ** (1.0 / reduction_steps))))
    ang_range = np.float32(abs(float(a.max()) - float(a.min())) / n) * rf
    ngt = int(math.ceil(n / rf))
    W0 = int(np.rint(2.0 * N * whf_x))
    H0 = int(math.ceil(float(np.float32(ang_range)) * N * whf_y))
    m = (n // rf) * rf
    tmp = [angle_of_mean(a[g:g + rf]) for g in range(0, m, rf)]
    if m < n:
        tmp.append(angle_of_mean(a[m:]))
    tmp = tmp[:ngt]
    levels = []  # each: dict(H, W, g, rf, T (ngrids_padded,3,3) or nan)

    def level(H, W, angs, scale, red):
        M = [rot2d(x) @ ainvt(N, H, W, scale) for x in angs]
        g = (len(angs) + legs - 1) // legs
        pad = g * legs - len(angs)
        T = np.full((g * legs, 3, 3), np.nan)
        T[:len(angs)] = M
        Tinv = np.full_like(T, np.nan)
        Tinv[:len(angs)] = [np.linalg.inv(x) for x in M]
        levels.append(dict(H=H, W=W, g=g, rf=red, T=T, Tinv=Tinv, pad=pad))

    level(H0, W0, tmp, 1.0, rf)
    for _ in range(100):
        if int(math.ceil(ngt / rf)) < 20:
            break
        ang_range = np.float32(ang_range * rf)
        ngt = int(math.ceil(ngt / rf))
        H = int(math.ceil(float(ang_range) * N * whf_y))
        m = (len(tmp) // rf) * rf
        t2 = [angle_of_mean(tmp[g:g + rf]) for g in range(0, m, rf)]
        if m < len(tmp):
            t2.append(angle_of_mean(tmp[m:]))
        tmp = t2[:ngt]
        level(H, W0, tmp, 1.0, rf)
    level(N, N, [0.0] * legs, 1.0 / fac, max(1, ngt))
    return levels


def bilinear_clamped(img, x, y):
    """ghbp_cuda.cu:301-321: x = column, y = row, clamp-to-edge."""
    Hh, Ww = img.shape
    ix0 = np.floor(x).astype(np.int64)
    iy0 = np.floor(y).astype(np.int64)
    fx, fy = x - ix0, y - iy0
    ix1, iy1 = ix0 + 1, iy0 + 1
    ix0, ix1 = np.clip(ix0, 0, Ww - 1), np.clip(ix1, 0, Ww - 1)
    iy0, iy1 = np.clip(iy0, 0, Hh - 1), np.clip(iy1, 0, Hh - 1)
    return ((img[iy0, ix0] * (1 - fx) + img[iy0, ix1] * fx) * (1 - fy)
            + (img[iy1, ix0] * (1 - fx) + img[iy1, ix1] * fx) * fy)


def ghbp(sino, angles_rad, N, rot_center, axis_source_meters=1e9, voxel_size_microns=1.0,
         reduction_steps=2, legs=4, scale_factor=1.0, clip_outer_circle=False,
         whf_x=1.0, whf_y=1.0, fac=1.0, axis_correction=None):
    sino = np.asarray(sino, np.float64)
    n, nb = sino.shape
    S = build_bpsetups(angles_rad, rot_center, axis_source_meters, voxel_size_microns, axis_correction)
    L = build_hierarchy(angles_rad, N, reduction_steps, legs, whf_x, whf_y, fac)
    final = np.zeros((N, N))
    for leg in range(legs):
        # level 0: collect rf0 projections per grid
        G = L[0]
        px, py = np.meshgrid(np.arange(G["H"]), np.arange(G["W"]), indexing="ij")
        grids = np.zeros((G["g"], G["H"], G["W"]))
        for i in range(G["g"]):
            gi = leg * G["g"] + i
            T = G["T"][gi]
            Y = T[0, 0] * px + T[0, 1] * py + T[0, 2]
            X = T[1, 0] * px + T[1, 1] * py + T[1, 2]
            acc = np.zeros_like(Y)
            for k in range(G["rf"]):
                s = gi * G["rf"] + k
                if s >= n:
                    continue
                ph, pw = S[s, 0], S[s, 2]
                h = (ph[1] * Y + ph[2] * X + ph[3]) / (pw[1] * Y + pw[2] * X + pw[3])
                ih0 = np.floor(h).astype(np.int64)
                fh = h - ih0
                ih1 = ih0 + 1
                v0 = np.where((ih0 >= 0) & (ih0 < nb), sino[s, np.clip(ih0, 0, nb - 1)], 0.0)
                v1 = np.where((ih1 >= 0) & (ih1 < nb), sino[s, np.clip(ih1, 0, nb - 1)], 0.0)
                acc += v0 * (1 - fh) + v1 * fh
            grids[i] = scale_factor * acc
        prev, Gp = grids, G
        # aggregation levels
        for lev in range(1, len(L)):
            G = L[lev]
            px, py = np.meshgrid(np.arange(G["H"]), np.arange(G["W"]), indexing="ij")
            new = np.zeros((G["g"], G["H"], G["W"]))
            for i in range(G["g"]):
                T = G["T"][leg * G["g"] + i]
                if np.isnan(T[0, 0]):
                    continue
                Y = T[0, 0] * px + T[0, 1] * py + T[0, 2]
                X = T[1, 0] * px + T[1, 1] * py + T[1, 2]
                for k in range(G["rf"]):
                    pi = i * G["rf"] + k
                    if pi >= Gp["g"]:
                        continue
                    Ti = Gp["Tinv"][leg * Gp["g"] + pi]
                    if np.isnan(Ti[0, 0]):
                        break
                    prow = Ti[0, 0] * Y + Ti[0, 1] * X + Ti[0, 2]
                    pcol = Ti[1, 0] * Y + Ti[1, 1] * X + Ti[1, 2]
                    new[i] += bilinear_clamped(prev[pi], pcol, prow)
            prev, Gp = new, G
        final += prev[0]
    if clip_outer_circle:
        c = (N - 1) * 0.5
        r2 = (N // 2) ** 2
        yy, xx = np.mgrid[:N, :N]
        final[(xx - c) ** 2 + (yy - c) ** 2 > r2] = 0.0
    return final.astype(np.float32)


def exact_pixel_driven(sino, angles_nabu, N, rot_center, scale_factor=1.0, rows=None, cols=None):
    """float64 voxel-driven parallel backprojection, linear interpolation, zero outside the
    detector (GHBP edge convention). h = rc + (col-c)cos(A) - (row-c)sin(A), A = nabu angles."""
    sino = np.asarray(sino, np.float64)
    nb = sino.shape[1]
    c = (N - 1) * 0.5
    A = np.asarray(angles_nabu, np.float64)
    ca, sa = np.cos(A), np.sin(A)
    rows = np.asarray(rows, np.float64)
    cols = np.asarray(cols, np.float64)
    out = np.zeros(rows.size)
    ka = np.arange(len(A))
    for s0 in range(0, rows.size, 1000):
        r = rows[s0:s0 + 1000, None] - c
        q = cols[s0:s0 + 1000, None] - c
        h = rot_center + q * ca[None] - r * sa[None]
        i0 = np.floor(h).astype(np.int64)
        f = h - i0
        i1 = i0 + 1
        v0 = np.where((i0 >= 0) & (i0 < nb), sino[ka[None], np.clip(i0, 0, nb - 1)], 0.0)
        v1 = np.where((i1 >= 0) & (i1 < nb), sino[ka[None], np.clip(i1, 0, nb - 1)], 0.0)
        out[s0:s0 + 1000] = (v0 * (1 - f) + v1 * f).sum(1)
    return scale_factor * out
