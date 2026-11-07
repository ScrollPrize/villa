"""Automatic affine registration of two scans of the same scroll.

Built for registering a 9 um scan (fixed) and a 2.4 um scan (moving) of one scroll,
where the scroll axis is z in both. Writes the same transform JSON as
find_transform.py (XYZ order, p_fixed = M @ p_moving, level-0 voxel coordinates),
with the inlier match points as landmarks, so the result can be inspected and
refined there.

  1. coarse: one moving axial slice against every 10th fixed axial slice at ~75 um,
     over 180 rotations about z, both mirrorings, and FFT translation
  2. fine:   3D phase correlation of ~19 um cubes at up to 120 points, sub-voxel peaks
  3. fit:    least-squares affine with residual-based outlier rejection, then one
             more matching pass seeded by that affine

Example:
    python auto_register.py \
        --fixed https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0841/volumes/20250821151531-9.366um-1.2m-113keV-masked.zarr \
        --moving https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0841/volumes/20260319124803-2.403um-0.2m-77keV-masked.zarr \
        --output-transform pherc0841_2p4_to_9um.json
"""

import argparse
import json
import math

import numpy as np
import zarr
from numpy.fft import fft2, fftn, ifft2, ifftn
from scipy.ndimage import affine_transform, gaussian_filter, rotate, zoom

from transform_utils import get_volume_dimensions

COARSE_UM = 75.0
FINE_UM = 19.0


class Scan:
    def __init__(self, path: str, voxel_size_um: float | None) -> None:
        self.path = path
        self.um = get_volume_dimensions(path, voxel_size_um).voxel_size_um
        self.root = zarr.open(path, mode="r")

    def level_near(self, target_um: float) -> int:
        return max(0, round(math.log2(target_um / self.um)))

    def array(self, level: int):
        return self.root[str(level)]


def to_level(p, level):
    """Level-0 voxel coordinate -> level-L coordinate (levels are 2x2x2 means)."""
    return (np.asarray(p, float) + 0.5) / 2**level - 0.5


def to_level0(p, level):
    return (np.asarray(p, float) + 0.5) * 2**level - 0.5


def apply(T, p):
    return T[:, :3] @ np.asarray(p, float) + T[:, 3]


def zero_mean(img, ds=1, sigma=1.0):
    img = gaussian_filter(img.astype(np.float32), sigma)
    if img.ndim == 2:
        img = img[::ds, ::ds]
    m = img > 0
    if m.any():
        img = img - img[m].mean()
    img[~m] = 0
    return img


def coarse_candidates(fixed: Scan, moving: Scan, *, ds=4, zstep=10, angle_step=2):
    """Best (fixed z, mirror, rotation, shift) for the middle moving slice, as ZYX affines."""
    lf, lm = fixed.level_near(COARSE_UM), moving.level_near(COARSE_UM)
    F, M = fixed.array(lf), moving.array(lm)
    s = (moving.um * 2**lm) / (fixed.um * 2**lf)
    zm = M.shape[0] // 2
    src = zoom(np.asarray(M[zm]).astype(np.float32), s, order=1)
    N = max(F.shape[1], F.shape[2])
    pad = np.zeros((N, N), np.float32)
    h, w = min(N, src.shape[0]), min(N, src.shape[1])
    sy, sx = (src.shape[0] - h) // 2, (src.shape[1] - w) // 2
    pad[(N - h) // 2:(N - h) // 2 + h, (N - w) // 2:(N - w) // 2 + w] = src[sy:sy + h, sx:sx + w]
    templates = {}
    for mirror in (False, True):
        base = pad[:, ::-1] if mirror else pad
        for angle in range(0, 360, angle_step):
            t = zero_mean(rotate(base, angle, reshape=False, order=1), ds)
            templates[(mirror, angle)] = (np.conj(fft2(t)), np.sqrt((t**2).sum()) + 1e-6)

    best = (-1.0, None)
    for zi, sl in enumerate(np.asarray(F[::zstep])):  # one read: each chunk fetched once
        img = np.zeros((N, N), np.float32)
        img[:sl.shape[0], :sl.shape[1]] = sl
        if (img > 0).mean() < 0.02:
            continue
        t = zero_mean(img, ds)
        T, tn = fft2(t), np.sqrt((t**2).sum()) + 1e-6
        for (mirror, angle), (C, n) in templates.items():
            c = np.real(ifft2(T * C))
            i = int(c.argmax())
            score = c.flat[i] / (n * tn)
            if score > best[0]:
                dy, dx = (np.array(np.unravel_index(i, c.shape)) + c.shape[0] // 2) % c.shape[0] - c.shape[0] // 2
                best = (score, (zi * zstep, mirror, angle, dy * ds, dx * ds))
    score, (zf, mirror, angle, dy, dx) = best
    print(f"coarse: score {score:.3f}, fixed level {lf} z {zf} <- moving level {lm} z {zm}, "
          f"mirror {mirror}, rotation {angle} deg, shift ({dy}, {dx}) px", flush=True)

    centre_moving = (np.array(M.shape[1:], float) - 1) / 2
    centre_pad = (N - 1) / 2
    th = math.radians(angle)
    corners = np.array([[a, b, c] for a in (0, M.shape[0]) for b in (0, M.shape[1]) for c in (0, M.shape[2])], float)
    corners = to_level0(corners, lm)
    candidates = []
    for sense in (1, -1):  # rotate()'s sense in these coordinates is settled by trial matches
        R = np.array([[math.cos(th), sense * math.sin(th)], [-sense * math.sin(th), math.cos(th)]])

        def map_point(p0):
            q = to_level(p0, lm)
            yx = (q[1:] - centre_moving) * s
            if mirror:
                yx = yx * np.array([1.0, -1.0])
            yx = R @ yx + centre_pad + np.array([dy, dx])
            return to_level0(np.array([zf + (q[0] - zm) * s, yx[0], yx[1]]), lf)

        Q = np.array([map_point(p) for p in corners])
        A, *_ = np.linalg.lstsq(np.c_[corners, np.ones(len(corners))], Q, rcond=None)
        candidates.append(A.T)
    return candidates


def _subpixel(c, i):
    offset = []
    for ax in range(3):
        idx = list(i)
        vals = []
        for d in (-1, 0, 1):
            idx[ax] = (i[ax] + d) % c.shape[ax]
            vals.append(c[tuple(idx)])
        den = vals[0] - 2 * vals[1] + vals[2]
        offset.append(0.5 * (vals[0] - vals[2]) / den if den < 0 else 0.0)
    return np.array(offset)


def _windowed(v):
    v = zero_mean(v)
    if (v != 0).mean() < 0.5:
        return None
    w = np.hanning(v.shape[0])
    return v * (w[:, None, None] * w[None, :, None] * w[None, None, :])


def match_point(fixed: Scan, moving: Scan, T, p_moving, *, half=32, search=16):
    """Match the moving cube nearest to level-0 point p_moving.

    Returns the level-0 moving point actually matched (the cube centre), the level-0 fixed
    point it maps to, and the peak strength."""
    lf, lm = fixed.level_near(FINE_UM), moving.level_near(FINE_UM)
    F, M = fixed.array(lf), moving.array(lm)
    s = (moving.um * 2**lm) / (fixed.um * 2**lf)
    c = np.round(to_level(p_moving, lm)).astype(int)
    a = np.asarray(M[c[0] - half:c[0] + half, c[1] - half:c[1] + half, c[2] - half:c[2] + half])
    if a.shape != (2 * half,) * 3 or (a > 0).mean() < 0.5:
        return None
    a = zoom(a.astype(np.float32), s, order=1)
    rot = T[:, :3] / np.cbrt(abs(np.linalg.det(T[:, :3])))
    if not np.allclose(rot, np.eye(3), atol=0.05):  # bring the moving cube into the fixed orientation
        mid = (np.array(a.shape) - 1) / 2
        inv = np.linalg.inv(rot)
        a = affine_transform(a, inv, offset=mid - inv @ mid, order=1)
    guess = np.round(to_level(apply(T, to_level0(c - 0.5, lm)), lf)).astype(int)
    n = a.shape[0] + 2 * search
    lo = guess - n // 2
    b = np.asarray(F[lo[0]:lo[0] + n, lo[1]:lo[1] + n, lo[2]:lo[2] + n])
    pa, pb = _windowed(a), _windowed(b) if b.shape == (n,) * 3 else None
    if pa is None or pb is None:
        return None
    A = np.zeros((n, n, n), np.float32)
    o = (n - a.shape[0]) // 2
    A[o:o + a.shape[0], o:o + a.shape[1], o:o + a.shape[2]] = pa
    cc = np.real(ifftn(fftn(pb) * np.conj(fftn(A))))
    i = np.unravel_index(cc.argmax(), cc.shape)
    d = (np.array(i) + n // 2) % n - n // 2 + _subpixel(cc, i)
    p_fixed = lo + o + (a.shape[0] - 1) / 2 + d
    return to_level0(c - 0.5, lm), to_level0(p_fixed, lf), float(cc.max() / (cc.std() + 1e-9))


def sample_points(moving: Scan):
    D, H, W = moving.array(0).shape
    return [
        [zf * D, H / 2 + r * H * math.sin(a), W / 2 + r * W * math.cos(a)]
        for zf in np.linspace(0.12, 0.88, 5)
        for a in np.linspace(0, 2 * math.pi, 8, endpoint=False)
        for r in (0.2, 0.42, 0.64)
    ]


def robust_affine(P, Q, usable):
    idx = np.where(usable)[0]
    for _ in range(6):
        A, *_ = np.linalg.lstsq(np.c_[P[idx], np.ones(len(idx))], Q[idx], rcond=None)
        res = np.linalg.norm(np.c_[P, np.ones(len(P))] @ A - Q, axis=1)
        keep = np.where(usable & (res < max(3 * np.median(res[idx]), 1.0)))[0]
        if np.array_equal(keep, idx):
            break
        idx = keep
    return A.T, idx, res


def register(fixed: Scan, moving: Scan, *, min_peak=10.0):
    points = sample_points(moving)
    trial = points[::6]
    T = max(
        coarse_candidates(fixed, moving),
        key=lambda T0: np.median([(m or (None, None, 0.0))[2] for m in (match_point(fixed, moving, T0, p) for p in trial)]),
    )
    for n_pass in (1, 2):
        rows = [m for m in (match_point(fixed, moving, T, p) for p in points) if m is not None]
        P = np.array([m[0] for m in rows])
        Q = np.array([m[1] for m in rows])
        peaks = np.array([m[2] for m in rows])
        T, idx, res = robust_affine(P, Q, peaks > min_peak)
        print(f"pass {n_pass}: {len(rows)} matches, {len(idx)} inliers, median residual "
              f"{np.median(res[idx]):.2f} fixed voxels (p90 {np.percentile(res[idx], 90):.2f})", flush=True)
    scales = np.linalg.svd(T[:, :3], compute_uv=False)
    print(f"scale {np.round(scales, 5).tolist()} (voxel-size ratio {moving.um / fixed.um:.5f})")
    return T, P[idx], Q[idx]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fixed", required=True)
    parser.add_argument("--moving", required=True)
    parser.add_argument("--output-transform", required=True)
    parser.add_argument("--fixed-voxel-size", type=float, help="microns; default from metadata.json")
    parser.add_argument("--moving-voxel-size", type=float, help="microns; default from metadata.json")
    args = parser.parse_args()

    fixed = Scan(args.fixed, args.fixed_voxel_size)
    moving = Scan(args.moving, args.moving_voxel_size)
    T, P, Q = register(fixed, moving)
    zyx_to_xyz = [2, 1, 0]
    M = np.zeros((3, 4))
    M[:, :3] = T[:, :3][np.ix_(zyx_to_xyz, zyx_to_xyz)]
    M[:, 3] = T[:, 3][zyx_to_xyz]
    with open(args.output_transform, "w") as f:
        json.dump(
            {
                "schema_version": "1.0.0",
                "fixed_volume": args.fixed,
                "transformation_matrix": M.tolist(),
                "fixed_landmarks": Q[:, zyx_to_xyz].tolist(),
                "moving_landmarks": P[:, zyx_to_xyz].tolist(),
            },
            f,
            indent=2,
        )
    print(f"wrote {args.output_transform}")


if __name__ == "__main__":
    main()
