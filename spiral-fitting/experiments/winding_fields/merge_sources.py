"""Merge surface tracks (vctracks + exact crossings) and afv volume fibers (afv_to_tracks.py +
afv_crossings.py) into one vctracks-style dir + crossings npz for track_graph.py.

Tracks keep ids 0..N1-1, afv fibers become N1.. . Crossings: track crossings (level_diff 0),
afv ordering crossings (their level_diff), plus links between the sources. Tracks of both
families sit on the afv H layer (afv_vs_tracks.py: peak within +-3 vx along the normal, dip at
+-5..6), so an afv H point and a track point on the line along its normal n = tH x z
(lateral <= `lat`, |s| <= `smax`) are the same winding: d = 0, one link per (afv fiber, track)
pair (smallest |s|).
usage: merge_sources.py tracks_dir tracks_crossings afv_dir afv_crossings out_dir [smax=2.5 lat=1.5]
"""
import json, os, sys, time
import multiprocessing as mp
import numpy as np
from scipy.spatial import cKDTree

Tt, Tc, At, Ac, out = sys.argv[1:6]
SMAX = float(sys.argv[6]) if len(sys.argv) > 6 else 2.5
LAT = float(sys.argv[7]) if len(sys.argv) > 7 else 1.5
os.makedirs(out, exist_ok=True)
t0 = time.time()
def log(*a):
    print(f'[{time.time() - t0:6.0f}s]', *a, flush=True)

def load(T):
    return (np.fromfile(f'{T}/offsets.i64', '<i8'), np.fromfile(f'{T}/coordinates.i32', '<i4').reshape(-1, 3),
            np.fromfile(f'{T}/family_codes.i8', np.int8), np.fromfile(f'{T}/z_bounds.i32', '<i4').reshape(-1, 2),
            np.fromfile(f'{T}/source_ids.u64', '<u8'))
o1, C1, f1, z1, s1 = load(Tt)
o2, C2, f2, z2, s2 = load(At)
N1, N2 = len(f1), len(f2)
log(f'tracks {N1} ({len(C1)} points), afv fibers {N2} ({len(C2)} points)')
np.r_[o1, o2[1:] + o1[-1]].astype('<i8').tofile(f'{out}/offsets.i64')
with open(f'{out}/coordinates.i32', 'wb') as fh:
    C1.astype('<i4').tofile(fh); C2.astype('<i4').tofile(fh)
np.r_[f1, f2].astype(np.int8).tofile(f'{out}/family_codes.i8')
np.r_[z1, z2].astype('<i4').tofile(f'{out}/z_bounds.i32')
np.r_[s1, s2].astype('<u8').tofile(f'{out}/source_ids.u64')
log('merged tracks written')

# ---------- cross-source links ----------
U = np.array([[p['z'], p['y'], p['x']] for p in json.load(open('/ephemeral/sean/experiments/winding_fields/data/umbilicus.json'))['control_points']], float)
U = U[np.argsort(U[:, 0])]
l2 = np.diff(o2); trk2 = np.repeat(np.arange(N2), l2); pos2 = np.arange(len(C2)) - np.repeat(o2[:-1], l2)
ip = np.where(pos2 > 0, np.arange(len(C2)) - 1, np.arange(len(C2)))
inx = np.where(pos2 < np.repeat(l2, l2) - 1, np.arange(len(C2)) + 1, np.arange(len(C2)))
hA = np.nonzero((f2[trk2] == 0) & (inx != ip))[0]
hA = hA[np.argsort(C2[hA, 0], kind='stable')]; zA = C2[hA, 0]
l1 = np.diff(o1); trk1 = np.repeat(np.arange(N1, dtype=np.int64), l1)
ok1 = f1[trk1] >= 0
oT = np.nonzero(ok1)[0]; oT = oT[np.argsort(C1[oT, 0], kind='stable')]; zT = C1[oT, 0]
log('sorted by z')

def chunk(zz):
    za, zb = zz
    a0, a1 = np.searchsorted(zA, [za, zb]); b0, b1 = np.searchsorted(zT, [za - SMAX - 2, zb + SMAX + 2])
    ia, ib_ = hA[a0:a1], oT[b0:b1]
    if not len(ia) or not len(ib_):
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0)
    PA = C2[ia].astype(np.float64)
    t = (C2[inx[ia]] - C2[ip[ia]]).astype(np.float64)
    n = np.cross(t, np.array([1.0, 0, 0])); nn = np.linalg.norm(n, axis=1)
    keep = nn > 0.5 * np.linalg.norm(t, axis=1)                                   # H point roughly horizontal
    ia, PA, n = ia[keep], PA[keep], n[keep] / nn[keep][:, None]
    lst = cKDTree(C1[ib_].astype(np.float32)).query_ball_point(PA.astype(np.float32), np.hypot(SMAX, LAT) + 0.01, return_sorted=False)
    c = np.array([len(l) for l in lst]); a = np.repeat(np.arange(len(ia)), c)
    if not len(a):
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0)
    b = ib_[np.concatenate([np.asarray(l, np.int64) for l in lst])]
    d = C1[b] - PA[a]; s = np.sum(d * n[a], 1); lat = np.linalg.norm(d - s[:, None] * n[a], axis=1)
    m = (lat <= LAT) & (np.abs(s) <= SMAX)
    return ia[a[m]], b[m], np.abs(s[m])

zmin, zmax = int(zA.min()), int(zA.max()) + 1
with mp.get_context('fork').Pool(64) as pool:
    res = pool.map(chunk, [(z, min(z + 25, zmax)) for z in range(zmin, zmax, 25)])
pa = np.concatenate([r[0] for r in res]); pb = np.concatenate([r[1] for r in res]); sd = np.concatenate([r[2] for r in res])
del res
key = trk2[pa] * np.int64(N1) + trk1[pb]
o = np.lexsort((sd, key)); f = np.r_[True, key[o][1:] != key[o][:-1]]; o = o[f]
pa, pb = pa[o], pb[o]
log(f'afv H point - track links: {len(f)} candidates, {len(o)} (afv fiber, track) pairs; tracks linked {len(np.unique(trk1[pb]))}, afv fibers linked {len(np.unique(trk2[pa]))}')

# ---------- merged crossings ----------
c1, c2 = np.load(Tc), np.load(Ac)
def expand(c, shift):
    off = c['offsets']; self_ = np.repeat(np.arange(len(off) - 1, dtype=np.int64), np.diff(off)) + shift
    lv = c['level_diff'].astype(np.int8) if 'level_diff' in c.files else np.zeros(len(self_), np.int8)
    return self_, c['partners'].astype(np.int64) + shift, c['self_local'].astype(np.int32), c['partner_local'].astype(np.int32), lv
A1 = expand(c1, 0); A2 = expand(c2, N1)
ta, tb = trk2[pa] + N1, trk1[pb]
la, lb = pos2[pa].astype(np.int32), (pb - o1[trk1[pb]]).astype(np.int32)
A3 = (np.r_[ta, tb], np.r_[tb, ta], np.r_[la, lb], np.r_[lb, la], np.zeros(2 * len(ta), np.int8))
S, P, SL, PL, LV = (np.concatenate([x[i] for x in (A1, A2, A3)]) for i in range(5))
kind = np.r_[np.zeros(len(A1[0]), np.int8), np.ones(len(A2[0]), np.int8), np.full(len(A3[0]), 2, np.int8)]
o = np.argsort(S, kind='stable')
cnt = np.bincount(S, minlength=N1 + N2)
np.savez(f'{out}.crossings.npz', metadata=json.dumps({'sources': [Tc, Ac], 'link_smax': SMAX, 'link_lat': LAT}),
         source_ids=np.r_[s1, s2], offsets=np.r_[0, np.cumsum(cnt)].astype(np.int64), partners=P[o].astype(np.int32),
         self_local=SL[o], partner_local=PL[o], level_diff=LV[o], kind=kind[o], positions=np.zeros(len(o)), clearances=np.zeros(len(o)))
json.dump({'track_count': N1 + N2, 'point_count': int(len(C1) + len(C2)), 'sources': [Tt, At], 'afv_offset': N1},
          open(f'{out}/metadata.json', 'w'), indent=1)
log(f'crossings: tracks {len(A1[0])}, afv {len(A2[0])}, links {len(A3[0])} (directed entries); wrote {out}.crossings.npz')
