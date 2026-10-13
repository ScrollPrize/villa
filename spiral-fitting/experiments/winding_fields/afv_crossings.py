"""H x V crossing relations for volume fiber traces (afv_to_tracks.py output), for track_graph.py.

Volume traces put H and V on different faces, so they never touch: at a crossing (V passes
through the line through the H point along the local normal n = tH x tV, lateral <= `lat`)
the V sits about one layer outward or inward (afv_hv_offsets.py: gap at |s| < 3, layers at
~+-7). Only the ordering along n is used, no spacing: for each H point the nearest crossing V
outward (s in [smin, smax]) is the same winding (its back face, d = 0) and the nearest one
inward is the previous winding (d = -1). n is oriented away from the umbilicus (sign only).
One relation per (H track, V track) pair (the smallest lateral separation); listed under both
tracks like the vctracks crossings sidecar, plus `level_diff` = label(partner) - label(self).
usage: afv_crossings.py tracks_dir [smin=2.5 smax=11 lat=1.5 [d_out=0 d_in=-1 out_suffix]]
"""
import json, os, sys, time
import multiprocessing as mp
import numpy as np
from scipy.spatial import cKDTree

T = sys.argv[1]
SMIN, SMAX, LAT = (float(v) for v in (sys.argv[2:5] if len(sys.argv) > 4 else (2.5, 11.0, 1.5)))
D_OUT, D_IN = (int(v) for v in (sys.argv[5:7] if len(sys.argv) > 6 else (0, -1)))
SUF = sys.argv[7] if len(sys.argv) > 7 else ''
W = '/ephemeral/sean/experiments/winding_fields'
t0 = time.time()
def log(*a):
    print(f'[{time.time() - t0:6.0f}s]', *a, flush=True)

C = np.fromfile(f'{T}/coordinates.i32', '<i4').reshape(-1, 3)
off = np.fromfile(f'{T}/offsets.i64', '<i8'); fam = np.fromfile(f'{T}/family_codes.i8', np.int8)
lens = np.diff(off); ntr = len(lens)
trk = np.repeat(np.arange(ntr), lens)
pos = np.arange(len(C)) - np.repeat(off[:-1], lens)
ip = np.where(pos > 0, np.arange(len(C)) - 1, np.arange(len(C)))
inx = np.where(pos < np.repeat(lens, lens) - 1, np.arange(len(C)) + 1, np.arange(len(C)))
pf = fam[trk]
U = np.array([[p['z'], p['y'], p['x']] for p in json.load(open(f'{W}/data/umbilicus.json'))['control_points']], float)
U = U[np.argsort(U[:, 0])]
order_z = np.argsort(C[:, 0], kind='stable'); zs_sorted = C[order_z, 0]

def tan(i):
    t = (C[inx[i]] - C[ip[i]]).astype(np.float64)
    return t / np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)

def chunk(zz):
    za, zb = zz
    lo, hi = np.searchsorted(zs_sorted, [za - SMAX - 1, zb + SMAX + 1])
    idx = order_z[lo:hi]
    idx = idx[inx[idx] != ip[idx]]
    iH = idx[(pf[idx] == 0) & (C[idx, 0] >= za) & (C[idx, 0] < zb)]
    iV = idx[pf[idx] == 1]
    if not len(iH) or not len(iV):
        return [np.zeros(0, np.int64)] * 3 + [np.zeros(0), np.zeros(0, np.int8)]
    lst = cKDTree(C[iV].astype(np.float32)).query_ball_point(C[iH].astype(np.float32), SMAX + 0.5, return_sorted=False)
    n = np.array([len(l) for l in lst]); a = np.repeat(iH, n); b = iV[np.concatenate([np.asarray(l, np.int64) for l in lst])]
    dvec = (C[b] - C[a]).astype(np.float64)
    nn = np.cross(tan(a), tan(b)); sn_ = np.linalg.norm(nn, axis=1)
    nn /= np.maximum(sn_, 1e-9)[:, None]
    rad = np.stack([np.zeros(len(a)), C[a, 1] - np.interp(C[a, 0], U[:, 0], U[:, 1]), C[a, 2] - np.interp(C[a, 0], U[:, 0], U[:, 2])], 1)
    nn *= np.sign(np.sum(nn * rad, 1))[:, None]
    s = np.sum(dvec * nn, 1); lat = np.linalg.norm(dvec - s[:, None] * nn, axis=1)
    ok = (sn_ > 0.7) & (lat <= LAT) & (np.abs(s) >= SMIN) & (np.abs(s) <= SMAX)
    a, b, s, lat = a[ok], b[ok], s[ok], lat[ok]
    side = (s > 0).astype(np.int8)
    # nearest crossing V on each side of each H point
    o = np.lexsort((np.abs(s), side, a)); a, b, s, lat, side = a[o], b[o], s[o], lat[o], side[o]
    f = np.r_[True, (a[1:] != a[:-1]) | (side[1:] != side[:-1])]
    return [a[f], b[f], np.zeros(f.sum(), np.int64), lat[f], side[f]]

zmin, zmax = int(C[:, 0].min()), int(C[:, 0].max()) + 1
zch = [(z, min(z + 50, zmax)) for z in range(zmin, zmax, 50)]
with mp.get_context('fork').Pool(64) as pool:
    res = pool.map(chunk, zch)
a = np.concatenate([r[0] for r in res]); b = np.concatenate([r[1] for r in res])
lat = np.concatenate([r[3] for r in res]); side = np.concatenate([r[4] for r in res])
del res
log(f'H points with an outward crossing V: {np.sum(side == 1)}, inward: {np.sum(side == 0)}')
# one relation per (H track, V track) pair: smallest lateral; drop pairs seen on both sides
key = trk[a] * np.int64(ntr) + trk[b]
o = np.lexsort((lat, key)); key, a, b, lat, side = key[o], a[o], b[o], lat[o], side[o]
st = np.r_[True, key[1:] != key[:-1]]
gid = np.cumsum(st) - 1
mixed = np.bincount(gid, weights=side) % np.bincount(gid) != 0                    # pair seen outward and inward
f = st & ~mixed[gid]
log(f'pairs {st.sum()}; dropped (seen on both sides) {mixed.sum()} ({mixed.mean():.4f}); kept {f.sum()} '
    f'(outward {np.sum(side[f] == 1)}, inward {np.sum(side[f] == 0)})')
a, b, side = a[f], b[f], side[f]
dl = np.where(side == 1, D_OUT, D_IN).astype(np.int8)                                      # label(V) - label(H)
A = np.r_[trk[a], trk[b]]; B = np.r_[trk[b], trk[a]]
SA = np.r_[pos[a], pos[b]]; SB = np.r_[pos[b], pos[a]]; DL = np.r_[dl, -dl]
o = np.argsort(A, kind='stable')
cnt = np.bincount(A, minlength=ntr)
outp = T.rstrip('/') + SUF + '.crossings.npz'
np.savez(outp, metadata=json.dumps({'smin': SMIN, 'smax': SMAX, 'lat': LAT, 'rule': f'nearest V along +n: d={D_OUT}, along -n: d={D_IN}'}),
         source_ids=np.fromfile(f'{T}/source_ids.u64', '<u8'), offsets=np.r_[0, np.cumsum(cnt)].astype(np.int64),
         partners=B[o].astype(np.int32), self_local=SA[o].astype(np.int32), partner_local=SB[o].astype(np.int32),
         level_diff=DL[o], positions=np.zeros(len(o)), clearances=np.zeros(len(o)))
log(f'wrote {outp}')
