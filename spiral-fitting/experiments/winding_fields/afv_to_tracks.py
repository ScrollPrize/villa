"""Automated Fiber Volume (.afv) -> vctracks-style dir + crossings npz, for track_graph.py.

Fibers are native L0 XYZ (1 point / 4 L0 vx); tracks are base zyx (L0 / 4) int32 with points
2 base vx apart, so every other point is kept (and the last). Families: H 0, V 1, else -1.
Crossings: for every H point the nearest V point within `r` base vx (cKDTree on the kept points);
per (H track, V track) pair only the closest contact is kept, listed under both tracks
(offsets / partners / self_local / partner_local, as the vctracks crossings sidecar).
usage: afv_to_tracks.py fibers.afv out_dir [r=1.5]
"""
import json, os, sqlite3, sys, time
import multiprocessing as mp
import numpy as np
from scipy.spatial import cKDTree

src, out = sys.argv[1], sys.argv[2]
R = float(sys.argv[3]) if len(sys.argv) > 3 else 1.5
os.makedirs(out, exist_ok=True)
t0 = time.time()
def log(*a):
    print(f'[{time.time() - t0:6.0f}s]', *a, flush=True)

def read_range(ab):
    a, b = ab
    db = sqlite3.connect(f'file:{src}?mode=ro', uri=True)
    ids, pts = [], []
    cur_id, cur = None, []
    def flush():
        if cur_id is None:
            return
        P = np.concatenate(cur)
        keep = np.r_[np.arange(0, len(P), 2), [len(P) - 1]] if len(P) % 2 == 0 else np.arange(0, len(P), 2)
        P = P[keep] / 4.0
        ids.append(cur_id); pts.append(np.rint(P[:, ::-1]).astype(np.int32))
    for fid, fs, blob in db.execute('select fiber_id, first_segment, points from blocks '
                                     'where fiber_id >= ? and fiber_id < ? order by fiber_id, first_segment', (a, b)):
        P = np.frombuffer(blob, '<f8').reshape(-1, 3)
        if fid != cur_id:
            flush(); cur_id, cur = fid, [P]
        else:
            cur.append(P[1:])                                    # first point shared with the previous block
    flush()
    return ids, pts

db = sqlite3.connect(f'file:{src}?mode=ro', uri=True)
meta = dict(db.execute('select key, value from metadata').fetchall())
fam_of = dict(db.execute('select id, family from fibers').fetchall())
nmax = max(fam_of) + 1
chunks = [(a, min(a + 2000, nmax)) for a in range(0, nmax, 2000)]
with mp.get_context("fork").Pool(96) as pool:
    res = pool.map(read_range, chunks)
ids = np.concatenate([np.array(r[0], np.int64) for r in res])
pts = [p for r in res for p in r[1]]
del res
lens = np.array([len(p) for p in pts], np.int64)
offs = np.r_[0, np.cumsum(lens)].astype('<i8')
C = np.concatenate(pts); del pts
fam = np.array([{'H': 0, 'V': 1}.get(fam_of[i], -1) for i in ids], np.int8)
log(f'tracks {len(ids)} (H {np.sum(fam == 0)}, V {np.sum(fam == 1)}), points {len(C)}; z {C[:, 0].min()}..{C[:, 0].max()}')

zb = np.stack([np.minimum.reduceat(C[:, 0], offs[:-1]), np.maximum.reduceat(C[:, 0], offs[:-1])], 1).astype('<i4')
offs.tofile(f'{out}/offsets.i64'); C.astype('<i4').tofile(f'{out}/coordinates.i32')
fam.tofile(f'{out}/family_codes.i8'); zb.tofile(f'{out}/z_bounds.i32'); ids.astype('<u8').tofile(f'{out}/source_ids.u64')
json.dump({'point_count': int(len(C)), 'track_count': int(len(ids)), 'source': src, 'afv_uuid': json.loads(meta['uuid']),
           'frame': 'base zyx = L0 xyz[::-1] / 4, every other afv point', 'crossing_r': R}, open(f'{out}/metadata.json', 'w'), indent=1)

trk = np.repeat(np.arange(len(ids)), lens)
loc = np.arange(len(C)) - np.repeat(offs[:-1], lens)
pf = fam[trk]
iH, iV = np.nonzero(pf == 0)[0], np.nonzero(pf == 1)[0]
tree = cKDTree(C[iV].astype(np.float32))
log(f'V tree built ({len(iV)} points); querying {len(iH)} H points, r {R}')
d, j = tree.query(C[iH].astype(np.float32), k=1, distance_upper_bound=R, workers=-1)
ok = np.isfinite(d)
h, v, d = iH[ok], iV[j[ok]], d[ok]
log(f'H points with a V point within {R}: {ok.sum()} ({ok.mean():.4f})')
# closest contact per (H track, V track) pair
key = trk[h] * np.int64(len(ids)) + trk[v]
o = np.lexsort((d, key)); key, h, v, d = key[o], h[o], v[o], d[o]
first = np.r_[True, key[1:] != key[:-1]]
h, v, d = h[first], v[first], d[first]
log(f'crossings (one per H,V track pair): {len(h)}; dist p50 {np.median(d):.2f}')
a = np.r_[trk[h], trk[v]]; b = np.r_[trk[v], trk[h]]
sa = np.r_[loc[h], loc[v]]; sb = np.r_[loc[v], loc[h]]; dd = np.r_[d, d]
o = np.argsort(a, kind='stable')
cnt = np.bincount(a, minlength=len(ids))
np.savez(f'{out}.crossings.npz', metadata=json.dumps({'r': R, 'source': src}), source_ids=ids.astype(np.uint64),
         offsets=np.r_[0, np.cumsum(cnt)].astype(np.int64), partners=b[o].astype(np.int32),
         self_local=sa[o].astype(np.int32), partner_local=sb[o].astype(np.int32),
         positions=np.zeros(len(o)), clearances=dd[o].astype(np.float64))
log('done')
