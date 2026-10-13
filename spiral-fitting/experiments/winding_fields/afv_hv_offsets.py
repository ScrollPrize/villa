"""Diagnostic: signed offset of V fibers from H fibers along the local sheet normal n = tH x tV
(oriented away from the umbilicus) at their crossings, i.e. pairs whose separation is mostly along n.
usage: afv_hv_offsets.py tracks_dir z0 z1 out.png"""
import json, sys
import numpy as np
from scipy.spatial import cKDTree
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
T, z0, z1, png = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
W = '/ephemeral/sean/experiments/winding_fields'
C = np.fromfile(f'{T}/coordinates.i32', '<i4').reshape(-1, 3); off = np.fromfile(f'{T}/offsets.i64', '<i8'); fam = np.fromfile(f'{T}/family_codes.i8', np.int8)
lens = np.diff(off); trk = np.repeat(np.arange(len(lens)), lens)
pos = np.arange(len(C)) - np.repeat(off[:-1], lens)
ip = np.where(pos > 0, np.arange(len(C)) - 1, np.arange(len(C))); inx = np.where(pos < np.repeat(lens, lens) - 1, np.arange(len(C)) + 1, np.arange(len(C)))
band = (C[:, 0] >= z0) & (C[:, 0] < z1) & (inx != ip)
pf = fam[trk]
iH = np.nonzero(band & (pf == 0))[0]; iV = np.nonzero(band & (pf == 1))[0]
def tan(i):
    t = (C[inx[i]] - C[ip[i]]).astype(np.float64); return t / np.linalg.norm(t, axis=1, keepdims=True)
U = np.array([[p['z'], p['y'], p['x']] for p in json.load(open(f'{W}/data/umbilicus.json'))['control_points']], float); U = U[np.argsort(U[:, 0])]
tv = cKDTree(C[iV].astype(np.float32))
lst = tv.query_ball_point(C[iH].astype(np.float32), 16.0, workers=-1, return_sorted=False)
n = np.array([len(l) for l in lst]); a = np.repeat(iH, n); b = iV[np.concatenate([np.asarray(l, int) for l in lst])]
dvec = (C[b] - C[a]).astype(np.float64)
tH, tV = tan(a), tan(b)
nn = np.cross(tH, tV); s_ = np.linalg.norm(nn, axis=1); ok = s_ > 0.7                     # fibers roughly perpendicular
nn = nn / np.maximum(s_, 1e-9)[:, None]
rad = np.stack([np.zeros(len(a)), C[a, 1] - np.interp(C[a, 0], U[:, 0], U[:, 1]), C[a, 2] - np.interp(C[a, 0], U[:, 0], U[:, 2])], 1)
nn *= np.sign(np.sum(nn * rad, 1))[:, None]
sn = np.sum(dvec * nn, 1); lat = np.linalg.norm(dvec - sn[:, None] * nn, axis=1)
# crossing point of each (H track, V track) pair = min lateral separation
key = trk[a] * np.int64(len(lens)) + trk[b]
o = np.lexsort((lat, key)); f = np.r_[True, key[o][1:] != key[o][:-1]]; sel = o[f]
sel = sel[ok[sel] & (lat[sel] <= 1.5)]
s = sn[sel]
print(f'H points {len(iH)}, crossings (lateral <= 1.5, |tHxtV| > .7) {len(sel)}')
h, e = np.histogram(s, bins=64, range=(-16, 16))
print('signed normal offset hist (-16..16, 0.5 bins):'); print(h)
# per H point: nearest crossing V outward and inward
fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
ax[0].bar(e[:-1], h, width=0.5, align='edge'); ax[0].set_xlabel('V offset along +n (outward), base vx'); ax[0].set_title(f'H x V crossings, z {z0}-{z1}: signed normal offset')
pa = a[sel]; order = np.lexsort((np.abs(s), pa)); first = np.r_[True, pa[order][1:] != pa[order][:-1]]
ns = s[order][first]
ax[1].hist(ns, bins=64, range=(-16, 16)); ax[1].set_title('nearest crossing V per H point (signed)'); ax[1].set_xlabel('base vx')
print('nearest per H point: frac outward', np.mean(ns > 0).round(3), 'median |s|', np.median(np.abs(ns)).round(2))
plt.tight_layout(); plt.savefig(png, dpi=110)
