"""Voxel fill targets from solved track labels (track_graph.py points.npz), for solve3d --track_targets.

Keeps anchored track points within 1 base vx of a level-3 solved slice (z3 multiple of `dz`,
the full-scroll slab grid), converts to level-3 coordinates and sorts them by slice.
Target sheet index in the point's own cut (theta about the umbilicus at its z):
H -> label, V -> label + 0.5 (the back layer sits half a winding outward).
Writes <out>/{z3,y,x,w,theta}.npy and index.npz (slice z3 -> [start, end)).
"""
import os, sys
import numpy as np

pts, out = sys.argv[1], sys.argv[2]
dz = int(sys.argv[3]) if len(sys.argv) > 3 else 3
os.makedirs(out, exist_ok=True)
d = np.load(pts)
zyx = d['zyx']
keep = d['anchored']
zb = zyx[:, 0]
z3 = np.rint(zb / (2 * dz)).astype(np.int64) * dz              # nearest solved level-3 slice
keep &= np.abs(zb - 2.0 * z3) <= 1.0
idx = np.nonzero(keep)[0]
o = np.argsort(z3[idx], kind='stable'); idx = idx[o]
w = d['label'][idx].astype(np.float32) + 0.5 * (d['fam'][idx] == 1)
zs_ = z3[idx]
np.save(f'{out}/z3.npy', zs_.astype(np.int32))
np.save(f'{out}/y.npy', (zyx[idx, 1] / 2.0).astype(np.float32))
np.save(f'{out}/x.npy', (zyx[idx, 2] / 2.0).astype(np.float32))
np.save(f'{out}/w.npy', w)
np.save(f'{out}/theta.npy', d['theta'][idx].astype(np.float32))
u, st, cnt = np.unique(zs_, return_index=True, return_counts=True)
np.savez(f'{out}/index.npz', z3=u, start=st, end=st + cnt, sigma=float(d['sigma']))
print(f'{len(idx)} target points ({len(idx) / len(zyx):.3f} of kept) on {len(u)} slices z3 {u.min()}..{u.max()}; '
      f'V share {np.mean(w % 1 != 0):.3f}')
