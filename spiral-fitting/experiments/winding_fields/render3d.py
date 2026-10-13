"""Render a solve3d result: xy slice (frac w, w, H/V signal) and a vertical cut through the centre."""
import sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

d = np.load(sys.argv[1])
out = sys.argv[2]
w, s, mask = d['w'], d['s'].astype(np.float32), d['mask']
cx, cy = float(d['cx']), float(d['cy'])
D = w.shape[0]
k = D // 2
wm = np.where(mask, w, np.nan)

fig, axs = plt.subplots(1, 3, figsize=(33, 11))
axs[0].imshow(np.where(mask[k], s[k], np.nan), cmap='coolwarm_r', vmin=-0.3, vmax=0.3, interpolation='nearest')
axs[0].set_title('H (blue) - V (red) signal')
axs[1].imshow(np.mod(wm[k], 1), cmap='twilight', interpolation='nearest')
axs[1].set_title('frac(w): 0 = H centre, 0.5 = V centre')
im = axs[2].imshow(wm[k], cmap='turbo')
lv = np.arange(np.floor(np.nanmin(wm[k])), np.nanmax(wm[k]) + 1, 5)
axs[2].contour(wm[k], levels=lv, colors='k', linewidths=0.5)
plt.colorbar(im, ax=axs[2]); axs[2].set_title(f'w, slice {k} (contours every 5); sigma {float(d["sigma"]):+.0f}')
plt.tight_layout(); plt.savefig(out + '_xy.png', dpi=50); plt.close()

# zoom on a band right of centre, to judge layer alignment
H_, W_ = w.shape[1:]
y0, x0 = int(cy) - 150, int(cx) + 150
sl = (k, slice(max(y0, 0), y0 + 300), slice(x0, x0 + 300))
fig, axs = plt.subplots(1, 2, figsize=(20, 10))
axs[0].imshow(np.where(mask[sl], s[sl], np.nan), cmap='coolwarm_r', vmin=-0.3, vmax=0.3, interpolation='nearest')
axs[0].contour(wm[sl], levels=np.arange(-200, 200, 1.0), colors='k', linewidths=0.8)
axs[0].contour(wm[sl], levels=np.arange(-199.5, 200, 1.0), colors='g', linewidths=0.5, linestyles='dashed')
axs[0].set_title('signal; black: integer w, green dashed: half')
axs[1].imshow(np.mod(wm[sl], 1), cmap='twilight', interpolation='nearest')
plt.tight_layout(); plt.savefig(out + '_zoom.png', dpi=60); plt.close()

# vertical cut along x through the centre row
row = int(cy)
fig, axs = plt.subplots(2, 1, figsize=(24, 8))
axs[0].imshow(np.where(mask[:, row], s[:, row], np.nan), cmap='coolwarm_r', vmin=-0.3, vmax=0.3, aspect='auto', interpolation='nearest')
axs[0].set_title(f'H-V signal, xz cut at y={row} (z = slab slices)')
axs[1].imshow(np.mod(wm[:, row], 1), cmap='twilight', aspect='auto', interpolation='nearest')
axs[1].set_title('frac(w) on the same cut')
plt.tight_layout(); plt.savefig(out + '_xz.png', dpi=50); plt.close()
print('wrote', out + '_{xy,zoom,xz}.png')
