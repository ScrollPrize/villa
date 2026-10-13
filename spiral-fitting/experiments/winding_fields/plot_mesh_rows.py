"""EVAL ONLY: plot lifted field w vs mesh winding W along a few mesh rows, plus where the rows lie."""
import json, sys
import numpy as np, tifffile
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
d = np.load(sys.argv[1]); M = sys.argv[2]; U = np.array([[p['z'], p['y'], p['x']] for p in json.load(open(sys.argv[3]))['control_points']], float); U = U[np.argsort(U[:, 0])]
out = sys.argv[4]
w, mask, zs, crop = d['w'], d['mask'], d['zs'], d['crop']; y0, y1, x0, x1 = crop; dz = int(zs[1] - zs[0]); sc = 2
fcx, fcy, fs = float(d['cx']), float(d['cy']), float(d['sigma'])
X = tifffile.imread(f'{M}/x.tif'); Y = tifffile.imread(f'{M}/y.tif'); Z = tifffile.imread(f'{M}/z.tif')
k = len(zs) // 2; zb = zs[k] * sc
import warnings; warnings.simplefilter("ignore")
r = int(np.nanargmin(np.abs(np.nanmedian(np.where(Z > 0, Z, np.nan), 1) - zb)))
v = (X[r] > 0)
c = np.nonzero(v)[0]
zr = np.median(Z[r, v]); ucx, ucy = np.interp(zr, U[:, 0], U[:, 2]), np.interp(zr, U[:, 0], U[:, 1])
W = np.unwrap(np.arctan2(Y[r, c] - ucy, X[r, c] - ucx)) / (2 * np.pi)
if np.corrcoef(W, np.hypot(X[r, c] - ucx, Y[r, c] - ucy))[0, 1] < 0: W = -W
iy = np.rint(Y[r, c] / sc).astype(int) - y0; ix = np.rint(X[r, c] / sc).astype(int) - x0
ok = (iy >= 0) & (iy < w.shape[1]) & (ix >= 0) & (ix < w.shape[2]); ok[ok] &= mask[k, iy[ok], ix[ok]]
wf = np.full(len(c), np.nan); wf[ok] = w[k, iy[ok], ix[ok]]
thw = np.arctan2(iy - fcy, ix - fcx); wl = wf - fs * thw / (2 * np.pi)   # u/2pi: continuous turn count
fig, axs = plt.subplots(1, 2, figsize=(26, 10))
axs[0].plot(c, W - np.nanmedian(W[ok][:50]), 'k-', lw=1, label='mesh W (turns)')
axs[0].plot(c, wl - np.nanmedian(wl[ok][:50]), 'r.', ms=1.5, label='field u/2pi (turn count)')
axs[0].plot(c, wf - np.nanmedian(wf[ok][:50]), 'b.', ms=1, alpha=0.4, label='field w, stored (wrapped theta)')
axs[0].set_xlabel('mesh column (along spiral)'); axs[0].legend(); axs[0].set_title(f'mesh row {r} (z~{zr:.0f} base), slab slice {k}')
axs[1].imshow(np.where(mask[k], np.mod(w[k], 1), np.nan), cmap='twilight', interpolation='nearest')
axs[1].plot(ix[ok], iy[ok], 'g-', lw=0.6); axs[1].plot(fcx, fcy, 'r+', ms=20)
axs[1].set_title('frac(w) with this mesh row (green)')
plt.tight_layout(); plt.savefig(out, dpi=55); print('wrote', out, 'row', r, 'valid on slice', ok.sum())
