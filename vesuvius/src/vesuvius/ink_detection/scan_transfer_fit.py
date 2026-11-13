"""Fit the pooled-to-native scan transfer kernel used by ``training/scan_transfer.py`` from a registered pair.

A pair is one surface rendered twice: from a 2.4 um scan pooled to ~9.6 um (``public_2p4_level2_zmean4``, 21 planes)
and from a native 8.6-9.4 um scan (28 planes). Steps: warp every pooled plane onto the native grid (anisotropic scale +
offset; found by a scale grid with phase correlation unless given), match depth (pooled plane best correlated with each
native plane), robust-standardise both on papyrus, then least squares for one 3D kernel ``K`` (kz x kxy x kxy) with
``native[i] ~ sum_dz,dy,dx K * pooled[i + offset + dz, y + dy, x + dx]`` over random papyrus pixels near the surface.
Reports held-out R^2 against plain resampling and writes ``{"kernel": ..., "fit": ...}`` JSON for ``load_kernel``.

The shipped ``builtin:pooled_to_native_pherc0814`` came from PHerc 0814 segment 46527 (held-out R^2 0.740 vs 0.623 for
resampling alone): depth weights -0.09 / 0.27 / 0.43 / 0.37 / 0.07, mild in-plane blur, no sharpening; the residual is
smooth (autocorrelation ~0.5 at 6 px), which the augmentation models as a smooth random field.

CLI::

    python -m vesuvius.ink_detection.scan_transfer_fit OUT.json POOLED.zarr NATIVE.zarr [--scale-offset SY SX OY OX]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from scipy import ndimage


def robust_z(a: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.zeros(a.shape, np.float32)
    med = np.median(a[mask])
    mad = np.median(np.abs(a[mask] - med)) * 1.4826 + 1e-6
    out[mask] = (a[mask] - med) / mad
    return out


def ncc(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    a, b = a[mask], b[mask]
    return float(((a - a.mean()) * (b - b.mean())).mean() / (a.std() * b.std() + 1e-9))


def fit_kernel(pooled: np.ndarray, native: np.ndarray, mask: np.ndarray, depth_offset: int, *, kz: int = 5,
               kxy: int = 9, samples: int = 150000, seed: int = 0) -> dict:
    """Least-squares kernel from aligned, standardised stacks (pooled planes already on the native grid)."""

    rz, rxy = kz // 2, kxy // 2
    rng = np.random.default_rng(seed)
    ys, xs = np.nonzero(mask)
    mid = native.shape[0] // 2
    rows, targets = [], []
    planes = [i for i in range(mid - 3, mid + 4) if 0 <= i + depth_offset - rz and i + depth_offset + rz < pooled.shape[0]]
    for i in planes:
        j = i + depth_offset
        pick = rng.choice(len(ys), size=min(len(ys), samples // max(len(planes), 1)), replace=False)
        y, x = ys[pick], xs[pick]
        ok = (y >= rxy) & (y < native.shape[1] - rxy) & (x >= rxy) & (x < native.shape[2] - rxy)
        y, x = y[ok], x[ok]
        rows.append(np.stack([pooled[j + dz, y + dy, x + dx] for dz in range(-rz, rz + 1)
                              for dy in range(-rxy, rxy + 1) for dx in range(-rxy, rxy + 1)], 1).astype(np.float32))
        targets.append(native[i, y, x])
    X, Y = np.concatenate(rows), np.concatenate(targets)
    X1 = np.hstack([X, np.ones((len(X), 1), np.float32)])
    train = rng.random(len(X)) < 0.8
    coef = np.linalg.lstsq(X1[train], Y[train], rcond=None)[0]
    pred = X1[~train] @ coef
    centre = (kz // 2) * kxy * kxy + rxy * kxy + rxy
    return {
        "kernel": coef[:-1].reshape(kz, kxy, kxy),
        "bias": float(coef[-1]),
        "heldout_r2": float(1 - np.var(Y[~train] - pred) / np.var(Y[~train])),
        "resample_r2": float(np.corrcoef(Y[~train], X[~train, centre])[0, 1] ** 2),
        "residual_sd": float(np.std(Y[~train] - pred)),
    }


def align_pair(pooled: np.ndarray, native: np.ndarray, scale_offset=None):
    """Warp pooled planes onto the native grid, standardise, and find the depth offset."""

    from skimage.registration import phase_cross_correlation

    nmid = native[native.shape[0] // 2].astype(np.float32)
    nm = nmid > 0
    if scale_offset is None:
        pmid = pooled[pooled.shape[0] // 2].astype(np.float32)
        nz = robust_z(nmid, nm)
        best = (-1.0, None)
        for sy in np.arange(1.010, 1.0401, 0.003):
            for sx in np.arange(1.010, 1.0401, 0.003):
                p = ndimage.zoom(pmid, (sy / 2, sx / 2), order=1)
                n = nz[::2, ::2]
                h, w = min(p.shape[0], n.shape[0]), min(p.shape[1], n.shape[1])
                pz = robust_z(p[:h, :w], p[:h, :w] > 0)
                sh = phase_cross_correlation(n[:h, :w], pz, upsample_factor=4, normalization=None)[0]
                q = ndimage.shift(pz, sh, order=1)
                m = (q != 0) & nm[::2, ::2][:h, :w]
                if m.sum() > 1000:
                    s = ncc(q, n[:h, :w], m)
                    if s > best[0]:
                        best = (s, (sy, sx, sh[0] * 2, sh[1] * 2))
        scale_offset = best[1]
    sy, sx, oy, ox = scale_offset
    h, w = native.shape[1:]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    warped = np.stack([ndimage.map_coordinates(pooled[k].astype(np.float32), [(yy - oy) / sy, (xx - ox) / sx],
                                               order=1, cval=0) for k in range(pooled.shape[0])])
    pm = warped[warped.shape[0] // 2] > 0
    mask = nm & pm & ndimage.binary_erosion(nm & pm, iterations=12)
    wz = np.stack([robust_z(warped[k], mask) for k in range(warped.shape[0])])
    nzs = np.stack([robust_z(native[k].astype(np.float32), mask) for k in range(native.shape[0])])
    offsets = []
    for i in range(native.shape[0] // 2 - 4, native.shape[0] // 2 + 5):
        c = [ncc(nzs[i], wz[k], mask) for k in range(wz.shape[0])]
        offsets.append(int(np.argmax(c)) - i)
    return wz, nzs, mask, int(np.median(offsets)), scale_offset


def main(argv=None) -> int:
    from vc_reader.reader import open_node

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out")
    ap.add_argument("pooled")
    ap.add_argument("native")
    ap.add_argument("--scale-offset", nargs=4, type=float, metavar=("SY", "SX", "OY", "OX"))
    ap.add_argument("--kz", type=int, default=5)
    ap.add_argument("--kxy", type=int, default=9)
    a = ap.parse_args(argv)
    pooled = np.asarray(open_node(f"{a.pooled}/0")[:])
    native = np.asarray(open_node(f"{a.native}/0")[:])
    wz, nz, mask, offset, so = align_pair(pooled, native, a.scale_offset)
    fit = fit_kernel(wz, nz, mask, offset, kz=a.kz, kxy=a.kxy)
    kernel = fit.pop("kernel")
    fit.update({"scale_offset_yx": [float(v) for v in so], "depth_offset": offset})
    print(json.dumps(fit))
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"fit": fit, "kernel": np.round(kernel.astype(float), 6).tolist()}, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
