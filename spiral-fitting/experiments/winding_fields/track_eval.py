"""EVAL ONLY (the mesh never feeds a solve): score track-graph labels against the eval mesh.

Mesh sheet index S per vertex as in tracks_vs_mesh.py (run unwrap + vertical run linking,
x4 upsampled). Each labelled track point within max_d of the mesh takes the S of its nearest
vertex; S and the labels use the same cut (theta = +-pi about the umbilicus at the point's z).
Reports, over anchored points matched to the mesh:
  pointwise : label - sgn*S equal to its global mode (one numbering shift for the whole band)
  pairs     : random point pairs on (nearly) the same z, Delta label vs Delta S (as eval_mesh)
"""
import json, sys
import numpy as np
sys.path.insert(0, '/ephemeral/sean/experiments/winding_fields/code')
from mesh_sheets import mesh_sheet_points

pts_path = sys.argv[1]
max_d = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
d = np.load(pts_path)
zyx, lab, anch, fam = d['zyx'], d['label'], d['anchored'], d['fam']
z0, z1 = float(d['z0']), float(d['z1'])
if len(sys.argv) > 3:
    # compare a solved slab field instead, sampled at the same points (H: rint w, V: floor w), same cut convention
    sys.path.insert(0, '/ephemeral/sean/experiments/winding_fields/code'); import sheetgraph as sgm
    f = np.load(sys.argv[3]); w, mask, zs, crop = f['w'], f['mask'], f['zs'], f['crop']
    y0, x0 = int(crop[0]), int(crop[2]); scx, scy, sig = float(f['cx']) + x0, float(f['cy']) + y0, float(f['sigma'])
    z3 = zyx[:, 0] / 2.0; k = np.rint((z3 - zs[0]) / (zs[1] - zs[0])).astype(int)
    inz = (k >= 0) & (k < len(zs)) & (np.abs(z3 - zs[np.clip(k, 0, len(zs) - 1)]) <= 0.5)
    iy = np.rint(zyx[:, 1] / 2).astype(int) - y0; ix = np.rint(zyx[:, 2] / 2).astype(int) - x0
    inz &= (iy >= 0) & (iy < w.shape[1]) & (ix >= 0) & (ix < w.shape[2])
    inz[inz] &= mask[k[inz], iy[inz], ix[inz]]
    wv = np.full(len(zyx), np.nan); wv[inz] = w[k[inz], iy[inz], ix[inz]]
    th_s = np.arctan2(zyx[:, 1] / 2 - scy, zyx[:, 2] / 2 - scx)
    wg = wv + sgm.cut_term(th_s, d['theta'].astype(np.float64), sig)       # slab cut -> cut about the umbilicus at the point z
    lab = np.where(fam == 0, np.rint(wg), np.floor(wg))
    anch = inz.copy()
    print(f'FIELD MODE: {sys.argv[3]}; points on solved slices inside the field: {inz.sum()}')
    lab = np.nan_to_num(lab, nan=-1e6)
tree, PS, PC, sgn = mesh_sheet_points(z0, z1)
rng = np.random.default_rng(0)
sub = rng.choice(len(zyx), min(4_000_000, len(zyx)), replace=False)
dist, j = tree.query(zyx[sub].astype(np.float64), distance_upper_bound=max_d)
ok = np.isfinite(dist) & anch[sub]
idx = sub[ok]; S = sgn * PS[j[ok]].astype(np.float64); C = PC[j[ok]]
L = lab[idx].astype(np.float64)
print(f'points sampled {len(sub)}; anchored & within {max_d} vx of the mesh: {ok.sum()} ({ok.mean():.3f})')
r = L - S
main = C == np.bincount(C).argmax()
vals, cnt = np.unique(r[main], return_counts=True)
mode = vals[np.argmax(cnt)]
e = r[main] - mode
for f, nm in ((0, 'H'), (1, 'V'), (None, 'all')):
    mm = np.ones(len(e), bool) if f is None else (fam[idx][main] == f)
    print(f'pointwise ({nm}, n={mm.sum()}): exact {np.mean(e[mm] == 0):.3f}  within 1 {np.mean(np.abs(e[mm]) <= 1):.3f}  '
          f'signed mean {np.mean(e[mm]):+.3f}')
# pairs at nearly the same z, anywhere in the slice (like eval_mesh's same-row pairs)
zz = zyx[idx][main, 0]
o = np.argsort(zz); zz_s = zz[o]; e_s = e[o]; S_s = S[main][o]
a = rng.integers(0, len(o), 3_000_000)
b = np.clip(a + rng.integers(-20000, 20000, len(a)), 0, len(o) - 1)
k = (np.abs(zz_s[a] - zz_s[b]) <= 10) & (a != b)
de = e_s[b[k]] - e_s[a[k]]; dS = S_s[b[k]] - S_s[a[k]]
big = np.abs(dS) >= 1
print(f'pairs within 10 vx of z (n={k.sum()}): exact {np.mean(de == 0):.3f}  within1 {np.mean(np.abs(de) <= 1):.3f}; '
      f'pairs >= 5 turns apart: label/mesh turn ratio median {np.median(((S_s[b[k]] - S_s[a[k]]) + de)[np.abs(dS) >= 5] / dS[np.abs(dS) >= 5]):.3f}')
