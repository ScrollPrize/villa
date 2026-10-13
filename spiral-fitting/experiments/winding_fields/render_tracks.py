"""Render one z slab of a track-graph run: labels, anchoring, error vs the eval mesh (eval only),
and optionally the same error for a solved slab field at the same points."""
import sys
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, '/ephemeral/sean/experiments/winding_fields/code')
from mesh_sheets import mesh_sheet_points
pts, zc, out = sys.argv[1], float(sys.argv[2]), sys.argv[3]
field = sys.argv[4] if len(sys.argv) > 4 else None
d = np.load(pts)
m = np.abs(d['zyx'][:, 0] - zc) <= 3
zyx, lab, anch, fam, th = d['zyx'][m], d['label'][m], d['anchored'][m], d['fam'][m], d['theta'][m]
tree, PS, PC, sgn = mesh_sheet_points(zc - 10, zc + 10)
dist, j = tree.query(zyx.astype(np.float64), distance_upper_bound=3.0)
ok = np.isfinite(dist) & anch
S = np.where(ok, sgn * PS[np.minimum(j, len(PS) - 1)], 0)
def err(L):
    r = (L - S)[ok]; v, c = np.unique(r, return_counts=True); return np.where(ok, L - S - v[np.argmax(c)], np.nan)
E = err(lab)
panels = [('track labels mod 6', np.where(anch, np.mod(lab, 6), np.nan), 'tab10', (0, 9)),
          ('anchored (wi-connected) = 1', anch.astype(float), 'viridis', (0, 1)),
          (f'track label - mesh sheet (exact {np.nanmean(E == 0):.2f}, |e|<=1 {np.nanmean(np.abs(E) <= 1):.2f})', E, 'RdBu', (-3, 3))]
if field:
    sys.path.insert(0, '/ephemeral/sean/experiments/winding_fields/pydeps'); import sheetgraph as sgm
    f = np.load(field); w, mask, zs, crop = f['w'], f['mask'], f['zs'], f['crop']
    y0, x0 = int(crop[0]), int(crop[2]); scx, scy, sig = float(f['cx']) + x0, float(f['cy']) + y0, float(f['sigma'])
    k = int(np.argmin(np.abs(zs - zc / 2)))
    iy = np.clip(np.rint(zyx[:, 1] / 2).astype(int) - y0, 0, w.shape[1] - 1); ix = np.clip(np.rint(zyx[:, 2] / 2).astype(int) - x0, 0, w.shape[2] - 1)
    wg = w[k, iy, ix] + sgm.cut_term(np.arctan2(zyx[:, 1] / 2 - scy, zyx[:, 2] / 2 - scx), th.astype(np.float64), sig)
    LF = np.where(fam == 0, np.rint(wg), np.floor(wg)); LF = np.where(mask[k, iy, ix], LF, np.nan)
    EF = err(np.nan_to_num(LF, nan=-1e6)); EF[~np.isfinite(LF)] = np.nan
    panels.append((f'surf joint field - mesh sheet (exact {np.nanmean(EF == 0):.2f}, |e|<=1 {np.nanmean(np.abs(EF) <= 1):.2f})', EF, 'RdBu', (-3, 3)))
fig, axs = plt.subplots(1, len(panels), figsize=(13 * len(panels), 13))
for ax, (t, c, cm, (lo, hi)) in zip(axs, panels):
    v = np.isfinite(c)
    ax.scatter(zyx[v, 2], zyx[v, 1], c=c[v], s=0.3, cmap=cm, vmin=lo, vmax=hi, linewidths=0)
    ax.set_title(f'z {zc:.0f}: {t}'); ax.set_aspect('equal'); ax.invert_yaxis()
plt.tight_layout(); plt.savefig(out, dpi=45); print('wrote', out)
