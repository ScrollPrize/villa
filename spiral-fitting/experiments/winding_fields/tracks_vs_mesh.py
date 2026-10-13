"""DIAGNOSTIC ONLY (the eval mesh never feeds a solve): how sheet-consistent are fiber tracks
and their H/V crossings, judged by the eval mesh in one z band.

Mesh sheet index: along each row valid run, s = sigma'*(unwrap(theta) - theta)/2pi is the
integer turn count (jumps by 1 at the theta = +-pi cut); runs are linked vertically by the
modal difference over shared columns (same column, adjacent row = same sheet), giving one
sheet index S per vertex per connected component. The mesh band is upsampled x4 so a point
on the sheet is within a few voxels of a vertex. A track point takes the S of its nearest
vertex (within max_d); S is cut-corrected relative to the track's first matched point.
"""
import json, sys, time
import numpy as np, tifffile
from scipy.spatial import cKDTree
from scipy import ndimage as ndi

W = '/ephemeral/sean/experiments/winding_fields'
mesh = f'{W}/data/eval_mesh_DO_NOT_USE_AS_INPUT/spiral-output-generation-19-lasagna.tifxyz'
tdir = f'{W}/data/tracks/2um_ds2_ps256_surf_v2.dbm.vctracks'
cpath = f'{W}/data/tracks/2um_ds2_ps256_surf_v2.dbm.crossings.npz'
zb0, zb1 = (float(sys.argv[1]), float(sys.argv[2])) if len(sys.argv) > 2 else (9000.0, 9600.0)
max_d = float(sys.argv[3]) if len(sys.argv) > 3 else 6.0; up = 4
t0 = time.time()
U = np.array([[p['z'], p['y'], p['x']] for p in json.load(open(f'{W}/data/umbilicus.json'))['control_points']], float); U = U[np.argsort(U[:, 0])]
X = tifffile.imread(f'{mesh}/x.tif'); Y = tifffile.imread(f'{mesh}/y.tif'); Z = tifffile.imread(f'{mesh}/z.tif')
valid = (X > 0) & (Y > 0) & (Z > 0)
rows = np.nonzero(((Z >= zb0 - 60) & (Z <= zb1 + 60) & valid).any(1))[0]
r0, r1 = rows.min(), rows.max() + 1
X, Y, Z, valid = X[r0:r1], Y[r0:r1], Z[r0:r1], valid[r0:r1]
TH = np.arctan2(Y - np.interp(Z, U[:, 0], U[:, 1]), X - np.interp(Z, U[:, 0], U[:, 2]))
# sign: turn count increases outward
R = np.hypot(Y - np.interp(Z, U[:, 0], U[:, 1]), X - np.interp(Z, U[:, 0], U[:, 2]))
s_rel = np.zeros(X.shape, np.int64); run_id = np.full(X.shape, -1, np.int64); nrun = 0; sg = []
for r in range(X.shape[0]):
    idx = np.nonzero(valid[r])[0]
    for run in np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1):
        if len(run) < 10: continue
        u = np.unwrap(TH[r, run])
        if len(run) > 200: sg.append(np.sign(np.corrcoef(u, R[r, run])[0, 1]))
        s_rel[r, run] = np.rint((u - TH[r, run]) / (2 * np.pi)).astype(np.int64)
        run_id[r, run] = nrun; nrun += 1
sgn = 1 if np.sum(sg) >= 0 else -1
s_rel *= sgn
# link runs vertically
a = run_id[:-1]; b = run_id[1:]
m = (a >= 0) & (b >= 0)
d = (s_rel[:-1] - s_rel[1:])[m]                     # offset_b - offset_a
key = np.stack([a[m], b[m], d], 1)
uk, cnt = np.unique(key, axis=0, return_counts=True)
best = {}
for (ra, rb, dd), c in zip(uk, cnt):
    if (ra, rb) not in best or c > best[(ra, rb)][1]: best[(ra, rb)] = (dd, c)
adj = [[] for _ in range(nrun)]
for (ra, rb), (dd, c) in best.items():
    if c >= 5: adj[ra].append((rb, dd)); adj[rb].append((ra, -dd))
off = np.zeros(nrun, np.int64); comp = np.full(nrun, -1, np.int64); nc = 0
for s in range(nrun):
    if comp[s] >= 0: continue
    comp[s] = nc; st = [s]
    while st:
        x = st.pop()
        for y, dd in adj[x]:
            if comp[y] < 0: comp[y] = nc; off[y] = off[x] + dd; st.append(y)
    nc += 1
S = np.where(run_id >= 0, s_rel + off[np.maximum(run_id, 0)], -10**9)
C = np.where(run_id >= 0, comp[np.maximum(run_id, 0)], -1)
csz = np.bincount(C[C >= 0])
print(f'mesh rows {r0}-{r1}, runs {nrun}, components {nc}, largest covers {csz.max() / csz.sum():.3f} of vertices; sign {sgn} ({time.time()-t0:.0f}s)', flush=True)
# upsample the band x4 (bilinear coordinates, nearest S/C) where all 4 corners are valid and in one component
pts, PS, PC = [], [], []
for r in range(X.shape[0] - 1):
    for f in np.arange(up) / up:
        for g in np.arange(up) / up:
            ok = valid[r, :-1] & valid[r, 1:] & valid[r + 1, :-1] & valid[r + 1, 1:] & (C[r, :-1] == C[r + 1, 1:]) & (C[r, :-1] == C[r, 1:]) & (C[r, :-1] == C[r + 1, :-1])
            ok &= (S[r, :-1] == S[r + 1, 1:]) & (S[r, :-1] == S[r, 1:]) & (S[r, :-1] == S[r + 1, :-1])   # skip quads straddling the cut
            c = np.nonzero(ok)[0]
            if not len(c): continue
            P = [((1 - f) * ((1 - g) * A[r, c] + g * A[r, c + 1]) + f * ((1 - g) * A[r + 1, c] + g * A[r + 1, c + 1])) for A in (Z, Y, X)]
            pts.append(np.stack(P, 1)); PS.append(S[r, c]); PC.append(C[r, c])
pts = np.concatenate(pts); PS = np.concatenate(PS); PC = np.concatenate(PC)
sel = (pts[:, 0] >= zb0 - 30) & (pts[:, 0] <= zb1 + 30)
pts, PS, PC = pts[sel], PS[sel], PC[sel]
tree = cKDTree(pts)
print(f'mesh points (x{up} upsampled) in band: {len(pts)} ({time.time()-t0:.0f}s)', flush=True)

# tracks in the band
meta = json.load(open(f'{tdir}/metadata.json'))
offs = np.memmap(f'{tdir}/offsets.i64', dtype='<i8', mode='r')
coords = np.memmap(f'{tdir}/coordinates.i32', dtype='<i4', mode='r').reshape(-1, 3)
fam = np.fromfile(f'{tdir}/family_codes.i8', dtype=np.int8)
zbd = np.fromfile(f'{tdir}/z_bounds.i32', dtype='<i4').reshape(-1, 2)
inband = np.nonzero((zbd[:, 0] >= zb0) & (zbd[:, 1] <= zb1))[0]
rng = np.random.default_rng(0)
samp = np.sort(rng.choice(inband, min(150000, len(inband)), replace=False))
print(f'tracks fully in band: {len(inband)}; sampled {len(samp)}', flush=True)

def cutfix(th, th0):
    dd = th - th0
    return np.rint((dd - np.arctan2(np.sin(dd), np.cos(dd))) / (2 * np.pi)).astype(np.int64)

track_sheet = {}; track_pure = {}
stats = {0: [], 1: []}; nmatch = {0: [], 1: []}; jumps = {0: [], 1: []}; flips = {0: [], 1: []}; stepsz = {0: [], 1: []}
for t in samp:
    p = np.asarray(coords[offs[t]:offs[t + 1]], np.float64)
    dist, j = tree.query(p, distance_upper_bound=max_d)
    ok = np.isfinite(dist)
    f = int(fam[t]); nmatch[f].append(ok.mean())
    if ok.sum() < 5: continue
    pz, py, px = p[ok].T
    th = np.arctan2(py - np.interp(pz, U[:, 0], U[:, 1]), px - np.interp(pz, U[:, 0], U[:, 2]))
    if np.abs(th).max() > np.pi - 0.3: continue          # skip tracks near the cut (no cut correction needed)
    s = PS[j[ok]]; c = PC[j[ok]]
    key = c * 100000 + s
    vals, cnts = np.unique(key, return_counts=True)
    stats[f].append(cnts.max() / ok.sum())
    jumps[f].append(len(vals) > 1 and np.sort(cnts)[-2] >= max(3, 0.1 * ok.sum()))
    track_sheet[int(t)] = (int(vals[np.argmax(cnts)]), float(th[0]))
    flips[f].append(int(np.sum(key[1:] != key[:-1]))); track_pure[int(t)] = len(vals) == 1
    if len(vals) > 1:
        sec = np.sort(cnts)[-2]
        dd = np.abs(np.diff(np.unique(s)))
        stepsz[f].append(int(dd.min()))
for f, nm in ((0, 'H'), (1, 'V')):
    st = np.array(stats[f])
    print(f'{nm} tracks: matched-point frac p50 {np.median(nmatch[f]):.2f}; tracks with >=5 matched {len(st)}; '
          f'points on the modal sheet: mean {st.mean():.3f}, all-on-one-sheet {np.mean(st == 1):.3f}; '
          f'"jumping" tracks (2nd sheet >=10% of points) {np.mean(jumps[f]):.3f}')
    fl = np.array(flips[f]); print(f'   sheet flips along the track: 0 {np.mean(fl == 0):.3f}, 1 {np.mean(fl == 1):.3f}, 2 {np.mean(fl == 2):.3f}, >2 {np.mean(fl > 2):.3f}; '
          f'multi-sheet tracks: nearest other sheet is +-1 {np.mean(np.array(stepsz[f]) == 1):.3f}')

# crossings between sampled tracks
cz = np.load(cpath, mmap_mode='r')
co = cz['offsets']; partners = cz['partners']; clr = cz['clearances']; sl = cz['self_local']
same = []; n = 0; cl_ = []; mult = []; pure = []
for t in samp[:60000]:
    if int(t) not in track_sheet: continue
    qs = np.asarray(partners[co[t]:co[t + 1]]); cs = np.asarray(clr[co[t]:co[t + 1]])
    uq, nq = np.unique(qs, return_counts=True); mc = dict(zip(uq.tolist(), nq.tolist()))
    for q, cq in zip(qs, cs):
        q = int(q)
        if q <= t or q not in track_sheet: continue
        (ka, tha), (kb, thb) = track_sheet[int(t)], track_sheet[q]
        if ka // 100000 != kb // 100000: continue
        sa, sb = ka % 100000, kb % 100000
        same.append(sa - sb); cl_.append(cq); mult.append(mc[q]); pure.append(track_pure[int(t)] and track_pure[q])
same = np.array(same); cl_ = np.array(cl_); mult = np.array(mult); pure = np.array(pure)
if len(same):
    print(f'crossings between matched tracks: {len(same)}; same mesh sheet {np.mean(same == 0):.3f}; '
          f'|diff|==1 {np.mean(np.abs(same) == 1):.3f}; >1 {np.mean(np.abs(same) > 1):.3f}')
    for lo, hi in [(0, 4), (4, 10), (10, 20), (20, 40), (40, 1e9)]:
        m = (cl_ >= lo) & (cl_ < hi)
        if m.sum() > 200: print(f'   clearance {lo:>4}-{hi:<6}: n {m.sum():6d}  same sheet {np.mean(same[m] == 0):.3f}')
    for k in (1, 2, 3):
        m = (mult == k) if k < 3 else (mult >= 3)
        if m.sum() > 200: print(f'   crossings in this track pair {k}{"+" if k == 3 else ""}: n {m.sum():6d}  same sheet {np.mean(same[m] == 0):.3f}')
    print(f'   both tracks single-sheet: n {pure.sum()}  same sheet {np.mean(same[pure] == 0):.3f};  otherwise {np.mean(same[~pure] == 0):.3f}')
print(f'total {time.time()-t0:.0f}s')
