"""Diagnostic: signed offset of track points (H, V separately) from afv H fibers along n = tH x z
(oriented away from the umbilicus), for track points near the normal line (lateral <= 1.5).
usage: afv_vs_tracks.py afv_dir tracks_dir z0 z1 out.png"""
import json, sys
import numpy as np
from scipy.spatial import cKDTree
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
Adir, Tdir, z0, z1, png = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
W = '/ephemeral/sean/experiments/winding_fields'
U = np.array([[p['z'], p['y'], p['x']] for p in json.load(open(f'{W}/data/umbilicus.json'))['control_points']], float); U = U[np.argsort(U[:, 0])]
def load(T, pad):
    off = np.fromfile(f'{T}/offsets.i64', '<i8'); fam = np.fromfile(f'{T}/family_codes.i8', np.int8); zb = np.fromfile(f'{T}/z_bounds.i32', '<i4').reshape(-1, 2)
    C = np.memmap(f'{T}/coordinates.i32', '<i4', mode='r').reshape(-1, 3)
    sel = np.nonzero((zb[:, 1] >= z0 - pad) & (zb[:, 0] < z1 + pad))[0]
    P, F, TG = [], [], []
    for i in sel:
        p = np.asarray(C[off[i]:off[i + 1]]).astype(np.float64)
        m = (p[:, 0] >= z0 - pad) & (p[:, 0] < z1 + pad)
        if m.sum() < 3: continue
        t = np.gradient(p, axis=0); t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
        P.append(p[m]); TG.append(t[m]); F.append(np.full(m.sum(), fam[i]))
    return np.concatenate(P), np.concatenate(TG), np.concatenate(F)
A, tA, fA = load(Adir, 0)
T, tT, fT = load(Tdir, 14)
h = (fA == 0) & (np.abs(tA[:, 0]) < 0.5)
A, tA = A[h][::3], tA[h][::3]
n = np.cross(tA, np.array([1.0, 0, 0])); n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
rad = np.stack([np.zeros(len(A)), A[:, 1] - np.interp(A[:, 0], U[:, 0], U[:, 1]), A[:, 2] - np.interp(A[:, 0], U[:, 0], U[:, 2])], 1)
n *= np.sign(np.sum(n * rad, 1))[:, None]
fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
for k, (f, nm) in enumerate(((0, 'track H'), (1, 'track V'))):
    B = T[fT == f]
    lst = cKDTree(B).query_ball_point(A, 12.5, workers=-1, return_sorted=False)
    cnt = np.array([len(l) for l in lst]); ia = np.repeat(np.arange(len(A)), cnt); ib = np.concatenate([np.asarray(l, int) for l in lst])
    d = B[ib] - A[ia]; s = np.sum(d * n[ia], 1); lat = np.linalg.norm(d - s[:, None] * n[ia], axis=1)
    s = s[lat <= 1.5]
    hh, e = np.histogram(s, bins=50, range=(-12.5, 12.5))
    print(nm, 'signed offset hist (-12.5..12.5, 0.5 bins):', hh.tolist())
    ax[k].bar(e[:-1], hh, width=0.5, align='edge'); ax[k].set_title(f'{nm} offset from afv H along +n (outward), z {z0}-{z1}'); ax[k].set_xlabel('base vx')
plt.tight_layout(); plt.savefig(png, dpi=110)
