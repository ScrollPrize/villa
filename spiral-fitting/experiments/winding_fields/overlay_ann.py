"""Overlay annotated winding points on the H/V signal and w contours, one panel per collection."""
import json, sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

d = np.load(sys.argv[1]); ann = json.load(open(sys.argv[2]))['collections']; out = sys.argv[3]
w, s, mask, zs, crop = d['w'], d['s'].astype(np.float32), d['mask'], d['zs'], d['crop']
level = int(d['level']); scale = 2 ** (level - 2); y0, y1, x0, x1 = crop; dz = int(zs[1] - zs[0])
cols = []
for cid, c in ann.items():
    P = []
    for p in c['points'].values():
        x, y, z = p['p'][0] / scale - x0, p['p'][1] / scale - y0, p['p'][2] / scale
        k = (z - zs[0]) / dz
        if -0.5 <= k <= len(zs) - 0.5 and 0 <= y < w.shape[1] and 0 <= x < w.shape[2]:
            P.append((x, y, int(round(k)), p['wind_a']))
    if len(P) >= 4:
        cols.append((cid, P))
cols.sort(key=lambda t: -len(t[1]))
cols = cols[:6]
if not cols:
    print('no annotated collections in this slab'); sys.exit()
fig, axs = plt.subplots(2, len(cols), figsize=(9 * len(cols), 18), squeeze=False)
for j, (cid, P) in enumerate(cols):
    P = np.array(P, float)
    k = int(np.median(P[:, 2]))
    cx_, cy_ = P[:, 0].mean(), P[:, 1].mean()
    half = max(40, int(np.ptp(P[:, :2], axis=0).max() / 2) + 30)
    ys_ = slice(max(int(cy_) - half, 0), int(cy_) + half); xs_ = slice(max(int(cx_) - half, 0), int(cx_) + half)
    ext = [xs_.start, xs_.start + s[k, ys_, xs_].shape[1], ys_.start + s[k, ys_, xs_].shape[0], ys_.start]
    for i, img in enumerate([s[k, ys_, xs_], np.mod(w[k, ys_, xs_], 1)]):
        ax = axs[i, j]
        if i == 0:
            ax.imshow(img, cmap='coolwarm_r', vmin=-0.3, vmax=0.3, extent=ext, interpolation='nearest')
            ax.contour(np.arange(ext[0], ext[1]), np.arange(ext[3], ext[2]), w[k, ys_, xs_], levels=np.arange(-200, 200, 1.0), colors='k', linewidths=0.8)
        else:
            ax.imshow(img, cmap='twilight', extent=ext, interpolation='nearest')
        ax.scatter(P[:, 0], P[:, 1], c='lime', s=40, edgecolors='k')
        for x, y, _, wa in P:
            ax.text(x + 2, y - 2, f'{int(wa)}', color='lime', fontsize=12, fontweight='bold')
        ax.set_title(f'coll {cid} slice {k} ' + ('signal + integer w' if i == 0 else 'frac(w)'))
plt.tight_layout(); plt.savefig(out, dpi=55); plt.close()
print('wrote', out, [(c, len(P)) for c, P in cols])
