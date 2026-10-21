#!/usr/bin/env python
"""
Measure the scale of an OpenMVG scene from the turntable's ArUco markers, per acquisition block, and pick the factor.

Detection follows pgs-global-scaler (DICT_ARUCO_ORIGINAL, subpixel corners, original images). The camera poses and
focal lengths come from the scene, but by default the lens distortion does not: on small objects SfM fits its
radial polynomial only near the image centre, where the object's features are, and at the image edges, where the
markers are, it can be far off (1756: radial factor -1.3 to -337 at the image edge instead of about 1; corners then
do not triangulate at all). Per block, a plain k1/k2 distortion is fitted instead so that every corner's sightings
meet in one point (truncated reprojection cost, Nelder-Mead from k = 0); --sfm-distortion uses the scene's model as
pgs-global-scaler does. Corners are then triangulated with a RANSAC at --thr px and a least-squares refit on the
inliers (the scaler's 0.1 px threshold leaves too few sightings).

Per block with complete markers: scale = marker size / median side, the marker side spread, diagonals, the off-plane
residual of the corners and the radius of the marker ring in mm. With the refit, blocks of one scene agree to about
0.3-0.7% (1585 o1: 0.940/0.942/0.945, against 0.943/0.964/0.948 with the SfM model) and the ring radius comes out
the same in every block (Naples turntable 99.5-100.8 mm across sessions).

The factor to use: blocks with at least 10 complete markers whose ring radius is within 0.8 mm of their median; the
factor is the median of those blocks' factors. Multiply the scene, or the clouds made from it, by it (e.g.
split_align.py align --scale-a). --json writes every block's numbers and the choice.

Needs OpenCV with the aruco module (opencv-contrib-python or opencv-python >= 4.7), which the scrollcase environment
does not have.

Usage: python marker_scale.py SCENE.json IMAGE_DIR LIST.txt [LIST.txt ...] [--thr 1.5] [--marker-size 31.5]
                              [--json OUT.json] [--sfm-distortion]
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np


def load_scene(path):
    d = json.load(open(path))
    intr = {i["key"]: i["value"]["ptr_wrapper"]["data"] for i in d["intrinsics"]}
    ext = {e["key"]: e["value"] for e in d["extrinsics"]}
    cams = {}
    for v in d["views"]:
        vd = v["value"]["ptr_wrapper"]["data"]
        if vd["id_pose"] in ext:
            e = ext[vd["id_pose"]]
            cams[vd["filename"]] = (intr[vd["id_intrinsic"]], np.array(e["rotation"]), np.array(e["center"]))
    return cams


def distort(xn, k):
    r2 = (xn ** 2).sum(-1, keepdims=True)
    return xn * (1 + k[0] * r2 + k[1] * r2 ** 2 + k[2] * r2 ** 3)


def undistort(xd, k, iters=50):
    x = xd.copy()
    for _ in range(iters):
        r2 = (x ** 2).sum(-1, keepdims=True)
        x = xd / (1 + k[0] * r2 + k[1] * r2 ** 2 + k[2] * r2 ** 3)
    return x


def project(X, cam):
    K, R, C = cam
    xc = (X - C) @ R.T
    xn = xc[..., :2] / xc[..., 2:]
    return distort(xn, K["disto_k3"]) * K["focal_length"] + np.array(K["principal_point"]), xc[..., 2]


def detect(args):
    import cv2
    path, = args
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    p.minMarkerLengthRatioOriginalImg = 32 / max(img.shape)
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL), p)
    corners, ids, _ = det.detectMarkers(img)
    if ids is None:
        return {}
    return {int(i): c.reshape(4, 2) for i, c in zip(ids.ravel(), corners)}


def triangulate(obs, cams, thr, rng):
    """obs: list of (filename, pixel). RANSAC over view pairs, then linear refit on inliers."""
    rays = []
    for fn, px in obs:
        K, R, C = cams[fn]
        xn = undistort((px - np.array(K["principal_point"])) / K["focal_length"], K["disto_k3"])
        d = R.T @ np.r_[xn, 1]
        rays.append((C, d / np.linalg.norm(d)))

    def solve(idx):
        A, b = np.zeros((3, 3)), np.zeros(3)
        for i in idx:
            C, d = rays[i]
            M = np.eye(3) - np.outer(d, d)
            A += M
            b += M @ C
        return np.linalg.solve(A, b)

    def errs(X):
        return np.array([np.linalg.norm(project(X, cams[fn])[0] - px) for fn, px in obs])

    best = None
    n = len(obs)
    for _ in range(min(300, n * (n - 1) // 2)):
        i, j = rng.choice(n, 2, replace=False)
        if np.dot(rays[i][1], rays[j][1]) > 0.9995:  # nearly parallel rays: no depth
            continue
        X = solve([i, j])
        inl = np.flatnonzero(errs(X) < thr)
        if best is None or len(inl) > len(best):
            best = inl
    if best is None or len(best) < 3:
        return None, 0, np.nan
    for _ in range(3):
        X = solve(best)
        e = errs(X)
        best = np.flatnonzero(e < thr)
        if len(best) < 3:
            return None, 0, np.nan
    return solve(best), len(best), float(np.sqrt(np.mean(e[best] ** 2)))


def nelder_mead(f, x0, step=0.05, iters=400, tol=1e-9):
    """Minimise f from x0 (a few parameters, no derivatives)."""
    n = len(x0)
    pts = [np.array(x0, float)] + [np.array(x0, float) + step * np.eye(n)[i] for i in range(n)]
    vals = [f(p) for p in pts]
    for _ in range(iters):
        o = np.argsort(vals)
        pts, vals = [pts[i] for i in o], [vals[i] for i in o]
        if abs(vals[-1] - vals[0]) < tol * (1 + abs(vals[0])):
            break
        c = np.mean(pts[:-1], 0)
        r = c + (c - pts[-1])
        fr = f(r)
        if fr < vals[0]:
            e = c + 2 * (c - pts[-1])
            fe = f(e)
            pts[-1], vals[-1] = (e, fe) if fe < fr else (r, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = r, fr
        else:
            k = c + 0.5 * (pts[-1] - c)
            fk = f(k)
            if fk < vals[-1]:
                pts[-1], vals[-1] = k, fk
            else:
                pts = [pts[0] + 0.5 * (p - pts[0]) for p in pts]
                vals = [f(p) for p in pts]
    i = int(np.argmin(vals))
    return pts[i], vals[i]


def corner_residuals(obs, cams, k):
    """Reprojection error of every sighting when each corner is triangulated from all its sightings with the radial
    distortion k (k1, k2) in place of the scene's. obs: {corner: [(filename, pixel), ...]}."""
    keys = sorted(obs)
    C, R, P, F, PP, idx = [], [], [], [], [], []
    for j, key in enumerate(keys):
        for fn, px in obs[key]:
            K, Rc, Cc = cams[fn]
            C.append(Cc); R.append(Rc); P.append(px); F.append(K["focal_length"]); PP.append(K["principal_point"])
            idx.append(j)
    C, R, P, F, PP, idx = map(np.array, (C, R, P, F, PP, idx))
    kv = [k[0], k[1], 0.0]
    xn = undistort((P - PP) / F[:, None], kv)
    d = np.einsum("nji,nj->ni", R, np.c_[xn, np.ones(len(xn))])
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    M = np.eye(3)[None] - d[:, :, None] * d[:, None, :]
    A = np.zeros((len(keys), 3, 3))
    b = np.zeros((len(keys), 3))
    np.add.at(A, idx, M)
    np.add.at(b, idx, np.einsum("nij,nj->ni", M, C))
    X = np.linalg.solve(A + 1e-12 * np.eye(3)[None], b[..., None])[..., 0]
    Xc = np.einsum("nij,nj->ni", R, X[idx] - C)
    pr = distort(Xc[:, :2] / Xc[:, 2:], kv) * F[:, None] + PP
    return np.linalg.norm(pr - P, axis=1)


def fit_distortion(obs, cams, cap=8.0):
    """k1, k2 that make the corners' sightings meet (truncated squared reprojection error, several starts)."""
    def cost(k):
        e = corner_residuals(obs, cams, k)
        return float(np.mean(np.minimum(e, cap) ** 2)) if np.all(np.isfinite(e)) else 1e18
    return min((nelder_mead(cost, s) for s in ([0, 0], [-0.1, 0], [0.1, 0])), key=lambda r: r[1])[0]


def measure(obs, cams, thr, marker_size):
    """Triangulate the corners and measure the markers; None if fewer than 3 complete markers."""
    rng = np.random.default_rng(0)
    X, stats = {}, []
    for key, o in sorted(obs.items()):
        x, ninl, rms = triangulate(o, cams, thr, rng)
        if x is not None:
            X[key] = x
            stats.append((len(o), ninl, rms))
    markers = sorted({m for m, _ in X if all((m, k) in X for k in range(4))})
    res = {"corners": len(X), "corners_seen": len(obs), "markers": len(markers)}
    if stats:
        st = np.array(stats)
        res.update(inliers_median=float(np.median(st[:, 1])), sightings_median=float(np.median(st[:, 0])),
                   rms_px=float(np.median(st[:, 2])))
    if len(markers) < 3:
        return res
    sides, diags, centres = [], [], []
    for m in markers:
        P = np.array([X[m, k] for k in range(4)])
        sides += [np.linalg.norm(P[k] - P[(k + 1) % 4]) for k in range(4)]
        diags += [np.linalg.norm(P[0] - P[2]), np.linalg.norm(P[1] - P[3])]
        centres.append(P.mean(0))
    sides, diags = np.array(sides), np.array(diags)
    s = marker_size / np.median(sides)
    allP = np.array(list(X.values())) * s
    if not np.all(np.isfinite(allP)):
        return res
    c = allP.mean(0)
    _, _, Vt = np.linalg.svd(allP - c)
    off = (allP - c) @ Vt[2]
    Q = (np.array(centres) * s - c) @ Vt[:2].T
    A = np.c_[2 * Q, np.ones(len(Q))]
    xs = np.linalg.lstsq(A, (Q ** 2).sum(1), rcond=None)[0]
    per_marker = marker_size / sides.reshape(-1, 4).mean(1)
    res.update(scale=float(s), marker_spread=float(np.std(per_marker) / s), side=float(np.mean(sides) * s),
               side_sd=float(np.std(sides) * s), diagonal=float(np.mean(diags) * s),
               off_plane_rms=float(np.sqrt(np.mean(off ** 2))), ring=float(np.sqrt(xs[2] + xs[:2] @ xs[:2])))
    return res


def choose(blocks, min_markers=10, ring_tol=0.8):
    """The factor to use: median of the blocks with enough markers whose ring radius agrees with the others'."""
    cand = [b for b in blocks if b.get("markers", 0) >= min_markers and "scale" in b]
    if not cand:
        return None
    med_ring = float(np.median([b["ring"] for b in cand]))
    kept = [b for b in cand if abs(b["ring"] - med_ring) <= ring_tol]
    ks = [b["scale"] for b in kept]
    factor = float(np.median(ks))
    return {"factor": factor, "kept": [b["list"] for b in kept], "candidates": len(cand), "median_ring_mm": med_ring,
            "kept_spread": (max(ks) - min(ks)) / factor}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("image_dir")
    ap.add_argument("lists", nargs="+")
    ap.add_argument("--thr", type=float, default=1.5)
    ap.add_argument("--marker-size", type=float, default=31.5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--json", help="write every block's numbers and the chosen factor here")
    ap.add_argument("--sfm-distortion", action="store_true", help="use the scene's lens distortion (no refit)")
    a = ap.parse_args()
    cams = load_scene(a.scene)
    blocks = []
    for lst in a.lists:
        name = os.path.basename(lst)
        names = [n for n in open(lst).read().split() if n in cams]
        with Pool(a.workers) as pool:
            dets = pool.map(detect, [(os.path.join(a.image_dir, n),) for n in names])
        obs = {}
        for n, d in zip(names, dets):
            for mid, c in d.items():
                for k in range(4):
                    obs.setdefault((mid, k), []).append((n, c[k]))
        obs = {key: o for key, o in obs.items() if len(o) >= 3}
        row = {"list": name, "views": len(names)}
        if not obs:
            print(f"{name}: {len(names)} views, no corner seen 3 times")
            blocks.append(row)
            continue
        use = cams
        if not a.sfm_distortion:
            k = fit_distortion(obs, cams)
            e0 = corner_residuals(obs, cams, [0, 0])
            e1 = corner_residuals(obs, cams, k)
            use = {fn: (dict(K, disto_k3=[k[0], k[1], 0.0]), R, C) for fn, (K, R, C) in cams.items()}
            row.update(k1=float(k[0]), k2=float(k[1]), corner_reprojection_px_no_distortion=float(np.median(e0)),
                       corner_reprojection_px_fitted=float(np.median(e1)))
            print(f"{name}: fitted k1 {k[0]:+.4f} k2 {k[1]:+.4f}; corner reprojection median {np.median(e1):.2f} px "
                  f"({np.median(e0):.2f} px without distortion)")
        row.update(measure(obs, use, a.thr, a.marker_size))
        blocks.append(row)
        head = (f"{name}: {len(names)} views, {row['corners']} of {row['corners_seen']} corners (inliers median "
                f"{row.get('inliers_median', 0):.0f} of {row.get('sightings_median', 0):.0f} sightings, rms "
                f"{row.get('rms_px', float('nan')):.2f} px), {row['markers']} complete markers")
        if "scale" not in row:
            print(head + "; no scale")
            continue
        print(head)
        print(f"    scale {row['scale']:.3f}  (per-marker spread {row['marker_spread']:.2%}); sides in mm "
              f"{row['side']:.2f} sd {row['side_sd']:.2f}; diagonals {row['diagonal']:.2f} (square: "
              f"{a.marker_size * np.sqrt(2):.2f}); off-plane rms {row['off_plane_rms']:.2f} mm; ring radius "
              f"{row['ring']:.1f} mm")
        sys.stdout.flush()
    pick = choose(blocks)
    if pick:
        print(f"chosen: factor {pick['factor']:.4f}, median of {len(pick['kept'])} of {pick['candidates']} blocks with "
              f">= 10 markers (ring radius within 0.8 mm of {pick['median_ring_mm']:.1f} mm), spread "
              f"{pick['kept_spread']:.2%}")
    else:
        print("chosen: none (no block with 10 complete markers)")
    if a.json:
        json.dump({"blocks": blocks, "choice": pick, "distortion": "scene" if a.sfm_distortion else "refit"},
                  open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
