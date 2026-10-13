"""DIAGNOSTIC ONLY (nothing here feeds a solve): along winding_inference rays, compare the
ray's level difference with the eval mesh's turn-count difference at the same crossings.

Each crossing is matched to the nearest mesh vertex within the same z band (|dz| <= 12 base)
and within `max_d` base voxels in-plane. Reports agreement, binned by mesh turn count.
"""
import glob, json, sys
import numpy as np, tifffile
from scipy.spatial import cKDTree

wi, mesh, umb = sys.argv[1], sys.argv[2], sys.argv[3]
zb0, zb1, max_d = 9300, 9500, 6.0
U = np.array([[p['z'], p['y'], p['x']] for p in json.load(open(umb))['control_points']], float); U = U[np.argsort(U[:, 0])]
X = tifffile.imread(f'{mesh}/x.tif'); Y = tifffile.imread(f'{mesh}/y.tif'); Z = tifffile.imread(f'{mesh}/z.tif')
rows = np.nonzero(((Z >= zb0 - 20) & (Z <= zb1 + 20) & (X > 0)).any(1))[0]
P, Wm = [], []
for r in rows:
    v = X[r] > 0
    c = np.nonzero(v)[0]
    zr = np.median(Z[r, v]); ucx, ucy = np.interp(zr, U[:, 0], U[:, 2]), np.interp(zr, U[:, 0], U[:, 1])
    # turn count along the whole row (unwrap across invalid gaps too: columns are ordered along the spiral)
    W = np.unwrap(np.arctan2(Y[r, c] - ucy, X[r, c] - ucx)) / (2 * np.pi)
    if np.corrcoef(W, np.hypot(X[r, c] - ucx, Y[r, c] - ucy))[0, 1] < 0:
        W = -W
    W = W - W[0] + c[0] * 0
    P.append(np.stack([Z[r, c], Y[r, c], X[r, c]], 1)); Wm.append(W)
P = np.concatenate(P); Wm = np.concatenate(Wm)
tree = cKDTree(P * np.array([0.5, 1, 1]))             # z counts half: rows are 20 vx apart in z
dl_all, dW_all, Wref = [], [], []
for sh in sorted(glob.glob(wi + '/shard_*')):
    o = np.load(sh + '/ray_origin_zyx.npy'); st = np.load(sh + '/ray_step_zyx.npy')
    off = np.load(sh + '/crossing_offsets.npy'); ct = np.load(sh + '/crossing_t.npy'); cl = np.load(sh + '/crossing_level.npy')
    anc = o + 192 * st
    sel = np.nonzero((anc[:, 0] >= zb0) & (anc[:, 0] < zb1) & (np.abs(st[:, 0]) < 0.15))[0]
    for i in sel[::3]:
        t = ct[off[i]:off[i + 1]]; l = cl[off[i]:off[i + 1]].astype(float)
        p = o[i][None] + t[:, None] * st[i][None]
        d, j = tree.query(p * np.array([0.5, 1, 1]))
        ok = d <= max_d
        if ok.sum() < 2:
            continue
        ll, WW = l[ok], Wm[j[ok]]
        dl_all.append(ll[-1] - ll[0]); dW_all.append(WW[-1] - WW[0]); Wref.append(WW.mean())
dl = np.abs(np.array(dl_all)); dW = np.abs(np.array(dW_all)); Wref = np.array(Wref)
print(f'rays matched {len(dl)}; mesh rows {len(rows)}')
for lo, hi in [(0, 30), (30, 60), (60, 90), (90, 200)]:
    m = (Wref >= lo) & (Wref < hi) & (dW >= 2)
    if m.sum() < 10:
        continue
    print(f'mesh turns {lo:3d}-{hi:3d}: rays {m.sum():6d}  WI levels / mesh turns: median {np.median(dl[m] / dW[m]):.3f}  mean {np.mean(dl[m] / dW[m]):.3f}  exact {np.mean(np.abs(dl[m] - dW[m]) < 0.5):.2f}')
