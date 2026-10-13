"""EVAL ONLY: score a solved winding field against the spiral mesh (never an input).

The mesh is one continuous spiral surface (tifxyz, base frame). Along each grid row
(~constant z) the unwrapped angle around the umbilicus / 2pi is the mesh's winding
number. For vertex pairs on the same row inside the slab, compare the field's
Delta w with the mesh's Delta W. The mesh is not perfect GT, just reasonably good.
"""
import json, sys
import numpy as np
import tifffile

w_path, mesh_dir, umb_path = sys.argv[1], sys.argv[2], sys.argv[3]
d = np.load(w_path)
w, mask, zs, crop = d['w'], d['mask'], d['zs'], d['crop']
level = int(d['level']); scale = 2 ** (level - 2)
y0, y1, x0, x1 = crop; dz = int(zs[1] - zs[0]) if len(zs) > 1 else 1
zb0, zb1 = zs[0] * scale, zs[-1] * scale
fcx, fcy, fsig = float(d['cx']), float(d['cy']), float(d['sigma'])     # solver centre (crop px) and handedness                    # slab z range in base frame
X = tifffile.imread(f'{mesh_dir}/x.tif'); Y = tifffile.imread(f'{mesh_dir}/y.tif'); Z = tifffile.imread(f'{mesh_dir}/z.tif')
valid = (X > 0) & (Y > 0) & (Z > 0)
rows = np.nonzero(((Z >= zb0) & (Z <= zb1) & valid).any(1))[0]
U = np.array([[p['z'], p['y'], p['x']] for p in json.load(open(umb_path))['control_points']], float)
U = U[np.argsort(U[:, 0])]
rng = np.random.default_rng(0)
dW_all, dw_all, off_all = [], [], []
lr_ratio, lr_mid = [], []
check = []
for r in rows:
    v = valid[r] & (Z[r] >= zb0 - 0.5 * dz * scale) & (Z[r] <= zb1 + 0.5 * dz * scale)
    if v.sum() < 50:
        continue
    zr = float(np.median(Z[r, v]))
    ucx, ucy = np.interp(zr, U[:, 0], U[:, 2]), np.interp(zr, U[:, 0], U[:, 1])
    th = np.arctan2(Y[r] - ucy, X[r] - ucx)
    # contiguous valid runs; unwrap angle within a run only
    idx = np.nonzero(v)[0]
    breaks = np.nonzero(np.diff(idx) > 1)[0]
    for run in np.split(idx, breaks + 1):
        if len(run) < 50:
            continue
        W = np.unwrap(th[run]) / (2 * np.pi)
        rad = np.hypot(X[r, run] - ucx, Y[r, run] - ucy)
        if np.corrcoef(W, rad)[0, 1] < 0:
            W = -W                                          # winding increases outward
        check.append((r, run[0], run[-1], W[-1] - W[0]))
        # sample field at the vertices
        k = np.clip(np.rint((Z[r, run] / scale - zs[0]) / dz).astype(int), 0, len(zs) - 1)
        iy = np.rint(Y[r, run] / scale).astype(int) - y0; ix = np.rint(X[r, run] / scale).astype(int) - x0
        ok = (iy >= 0) & (iy < w.shape[1]) & (ix >= 0) & (ix < w.shape[2])
        ok[ok] &= mask[k[ok], iy[ok], ix[ok]]
        if ok.sum() < 20:
            continue
        wf = w[k[ok], iy[ok], ix[ok]]; Wm = W[ok]
        # w is the sheet index (constant along a sheet, jumps at the theta cut). The mesh's
        # continuous turn count corresponds to the single-valued u/2pi = w - sigma*theta/2pi.
        thw = np.arctan2(iy[ok] - fcy, ix[ok] - fcx)
        wf = wf - fsig * thw / (2 * np.pi)
        off_all.append(wf - Wm)
        # long range: pairs >= 5 turns apart along the run, ratio of field to mesh turn counts
        if np.ptp(Wm) >= 5:
            a5 = rng.integers(0, len(wf), 3000); b5 = rng.integers(0, len(wf), 3000)
            dW5 = Wm[b5] - Wm[a5]; k5 = np.abs(dW5) >= 5
            lr_ratio.append((wf[b5] - wf[a5])[k5] / dW5[k5]); lr_mid.append(0.5 * (Wm[b5] + Wm[a5])[k5])
        n = len(wf); m = min(4000, n * (n - 1) // 2)
        a = rng.integers(0, n, m); b = rng.integers(0, n, m)
        keep = a != b
        dw_all.append(wf[b[keep]] - wf[a[keep]]); dW_all.append(Wm[b[keep]] - Wm[a[keep]])
if not dw_all:
    print('no mesh vertices inside this slab'); sys.exit()
dw = np.concatenate(dw_all); dW = np.concatenate(dW_all)
sgn = np.sign(np.sum(dw * dW)) or 1.0
err = sgn * dw - dW
big = np.abs(dW) >= 1
print(f'mesh rows in slab {len(rows)}, runs used {len(off_all)}, vertex pairs {len(dw)} ({big.sum()} with |dW|>=1)')
print(f'mesh pairs: |err| p50 {np.median(np.abs(err)):.2f} p90 {np.percentile(np.abs(err), 90):.2f}  exact(<0.5) {np.mean(np.abs(err) < 0.5):.3f}  within1 {np.mean(np.abs(err) < 1.5):.3f}')
print(f'mesh pairs |dW|>=1: mean signed err/dW {np.mean(err[big] / dW[big]):+.3f}  (negative = field undercounts)')
offs = np.concatenate([o - np.median(o) for o in off_all])
print(f'w - W_mesh spread within a run: p50 |dev| {np.median(np.abs(offs)):.2f}, p90 {np.percentile(np.abs(offs), 90):.2f}')
fr = np.concatenate([np.mod(sgn * o, 1.0) for o in off_all])
R = np.abs(np.mean(np.exp(2j * np.pi * fr)))
print(f'frac(w) on the mesh surface: concentration R {R:.2f} (1 = mesh sits on one consistent phase)')

if lr_ratio:
    lr = np.concatenate(lr_ratio) * sgn
    print(f'long range (pairs >= 5 turns apart on one mesh run): field/mesh turn ratio median {np.median(lr):.3f}, p25 {np.percentile(lr, 25):.3f}, p75 {np.percentile(lr, 75):.3f}  (1 = same count)')
