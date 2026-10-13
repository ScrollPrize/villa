#!/usr/bin/env python
"""
Re-triangulate ArUco marker corners in an OpenMVG scene with a tolerant RANSAC and report the scale.

Diagnostic for scenes where pgs-global-scaler's 0.1 px inlier threshold leaves corners triangulated from too few
views. Detection follows the scaler (DICT_ARUCO_ORIGINAL, subpixel corners, original images, corners undistorted
with the scene's lens model); triangulation uses a pixel threshold of --thr with a least-squares refit on the inliers.
Per marker with four corners: side lengths and diagonals in scene units; scale = marker size / mean side.
Also reports the off-plane residual of the corners and the radius of the marker ring in mm.

Use it to pick the scale of each orientation: every block (one camera height) is an independent measurement, and a block
is trustworthy when its corners are complete, its side spread is small and its ring radius matches the turntable
(99.4-100.2 mm on the Naples turntable; blocks at 92, 101 or 102 mm disagreed with the others by 1-7%). Multiply the
scene, or the clouds made from it, by the factor of a trusted block, e.g. split_align.py align --scale-a.

Needs OpenCV with the aruco module (opencv-contrib-python or opencv-python >= 4.7), which the scrollcase environment
does not have.

Usage: python marker_scale.py SCENE.json IMAGE_DIR LIST.txt [LIST.txt ...] [--thr 1.5] [--marker-size 31.5]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("image_dir")
    ap.add_argument("lists", nargs="+")
    ap.add_argument("--thr", type=float, default=1.5)
    ap.add_argument("--marker-size", type=float, default=31.5)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    cams = load_scene(a.scene)
    for lst in a.lists:
        names = [n for n in open(lst).read().split() if n in cams]
        with Pool(a.workers) as pool:
            dets = pool.map(detect, [(os.path.join(a.image_dir, n),) for n in names])
        obs = {}
        for n, d in zip(names, dets):
            for mid, c in d.items():
                for k in range(4):
                    obs.setdefault((mid, k), []).append((n, c[k]))
        rng = np.random.default_rng(0)
        X, stats = {}, []
        for key, o in sorted(obs.items()):
            if len(o) < 3:
                continue
            x, ninl, rms = triangulate(o, cams, a.thr, rng)
            if x is not None:
                X[key] = x
                stats.append((len(o), ninl, rms))
        markers = sorted({m for m, _ in X if all((m, k) in X for k in range(4))})
        sides, diags, centres = [], [], []
        for m in markers:
            P = np.array([X[m, k] for k in range(4)])
            sides += [np.linalg.norm(P[k] - P[(k + 1) % 4]) for k in range(4)]
            diags += [np.linalg.norm(P[0] - P[2]), np.linalg.norm(P[1] - P[3])]
            centres.append(P.mean(0))
        sides, diags = np.array(sides), np.array(diags)
        st = np.array(stats) if stats else np.zeros((1, 3))
        if not markers:
            print(f"{os.path.basename(lst)}: {len(names)} views, {len(X)} of {len(obs)} corners triangulated "
                  f"(inliers median {np.median(st[:, 1]):.0f} of {np.median(st[:, 0]):.0f}), no complete marker")
            continue
        s = a.marker_size / np.median(sides)
        allP = np.array(list(X.values())) * s
        if len(markers) < 3 or not np.all(np.isfinite(allP)):
            # seen on blocks whose corners triangulate to near-degenerate positions
            print(f"{os.path.basename(lst)}: {len(names)} views, {len(markers)} complete markers with non-finite or "
                  f"too few corners; no scale")
            continue
        c = allP.mean(0)
        _, _, Vt = np.linalg.svd(allP - c)
        off = (allP - c) @ Vt[2]
        Q = (np.array(centres) * s - c) @ Vt[:2].T
        A = np.c_[2 * Q, np.ones(len(Q))]
        xs = np.linalg.lstsq(A, (Q ** 2).sum(1), rcond=None)[0]
        radius = np.sqrt(xs[2] + xs[:2] @ xs[:2])
        per_marker = a.marker_size / np.array(sides).reshape(-1, 4).mean(1)
        print(f"{os.path.basename(lst)}: {len(names)} views, {len(X)} of {len(obs)} corners "
              f"(inliers median {np.median(st[:, 1]):.0f} of {np.median(st[:, 0]):.0f} sightings, "
              f"rms {np.median(st[:, 2]):.2f} px), {len(markers)} complete markers")
        print(f"    scale {s:.3f}  (per-marker spread {np.std(per_marker) / s:.2%}, IQR "
              f"{np.percentile(per_marker, 25):.2f}..{np.percentile(per_marker, 75):.2f}); sides in mm "
              f"{np.mean(sides) * s:.2f} sd {np.std(sides) * s:.2f}; diagonals {np.mean(diags) * s:.2f} "
              f"(square: {a.marker_size * np.sqrt(2):.2f}); off-plane rms {np.sqrt(np.mean(off ** 2)):.2f} mm; "
              f"ring radius {radius:.1f} mm")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
