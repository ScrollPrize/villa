"""Diagnose slab-to-slab disagreement on shared slices: spatial map of w_A - w_B (with the
branch-cut correction between the two slab centres), and its dependence on radius/theta."""
import sys, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, '/ephemeral/sean/experiments/winding_fields/code')
import sheetgraph as sgm
root, ia, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
import glob
P = sorted(glob.glob(f'{root}/slab_*/w.npz'))
A, B = np.load(P[ia]), np.load(P[ia + 1])
print(list(A.keys()))
common = np.intersect1d(A['zs'], B['zs']); print('shared slices', common)
ya0, ya1, xa0, xa1 = A['crop']; yb0, yb1, xb0, xb1 = B['crop']
y0, y1, x0, x1 = max(ya0, yb0), min(ya1, yb1), max(xa0, xb0), min(xa1, xb1)
sig = float(A['sigma'])
print('centres A', float(A['cx']) + xa0, float(A['cy']) + ya0, 'B', float(B['cx']) + xb0, float(B['cy']) + yb0)
fig, axs = plt.subplots(2, len(common), figsize=(8 * len(common), 16), squeeze=False)
for j, z in enumerate(common):
    ka = int(np.nonzero(A['zs'] == z)[0][0]); kb = int(np.nonzero(B['zs'] == z)[0][0])
    wa = A['w'][ka, y0 - ya0:y1 - ya0, x0 - xa0:x1 - xa0]; wb = B['w'][kb, y0 - yb0:y1 - yb0, x0 - xb0:x1 - xb0]
    m = A['mask'][ka, y0 - ya0:y1 - ya0, x0 - xa0:x1 - xa0] & B['mask'][kb, y0 - yb0:y1 - yb0, x0 - xb0:x1 - xb0]
    yy, xx = np.mgrid[y0:y1, x0:x1]
    ta = np.arctan2(yy - (float(A['cy']) + ya0), xx - (float(A['cx']) + xa0))
    tb = np.arctan2(yy - (float(B['cy']) + yb0), xx - (float(B['cx']) + xb0))
    d = wa - wb + sgm.cut_term(ta, tb, sig)        # w_B - w_A = cut_term(theta_A, theta_B)
    off = np.rint(np.median(d[m]))
    dd = np.where(m, d - off, np.nan)
    r = np.hypot(yy - (float(A['cy']) + ya0), xx - (float(A['cx']) + xa0))
    v = dd[m]
    print(f'z {z}: off {off:+.0f}  |d|<0.5 {np.mean(np.abs(v) < .5):.3f}  <1 {np.mean(np.abs(v) < 1):.3f}  <1.5 {np.mean(np.abs(v) < 1.5):.3f}  frac-part spread {np.std(v - np.rint(v)):.3f}')
    for lo, hi in [(0, 200), (200, 500), (500, 900), (900, 1400), (1400, 3000)]:
        mm = m & (r >= lo) & (r < hi)
        if mm.sum() > 1000:
            q = dd[mm]
            print(f'    r {lo:4d}-{hi:4d}: median {np.median(q):+.2f}  |d|<0.5 {np.mean(np.abs(q) < .5):.2f}  rint hist', {int(k): round(float(c), 2) for k, c in zip(*np.unique(np.clip(np.rint(q), -4, 4), return_counts=True)) if c / mm.sum() > 0.02 for c in [c / mm.sum()]})
    axs[0, j].imshow(dd, cmap='RdBu', vmin=-3, vmax=3, interpolation='nearest'); axs[0, j].set_title(f'z {z}: wA - wB + cut - {off:+.0f}')
    axs[1, j].imshow(np.where(m, np.mod(wa, 1), np.nan), cmap='twilight', interpolation='nearest'); axs[1, j].set_title('frac(wA)')
plt.tight_layout(); plt.savefig(out, dpi=40)
