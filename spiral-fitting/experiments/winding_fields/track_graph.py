"""Sheet graph on fiber tracks: nodes are short track segments, solved globally (no slabs).

Edges (x_b - x_a = d, branch-cut term included in d):
  along     consecutive segments of one track             d = 0
  crossing  an H and a V track through the same voxel      d = 0   (exact-crossing sidecar)
  wi        consecutive winding_inference crossings on a   d = level difference
            ray, each snapped to its nearest track point
A track is split into segments of `seg` points (2 base vx apart) and wherever it crosses
the theta branch cut, so a track that hops sheets breaks only one along edge, which the
L1 solve can give up. Only winding_inference relates different sheets, so components
without a wi edge have no defined offset; they are reported and flagged, not filled.

Output (out dir): points.npz (kept track points, base frame zyx, family, theta, segment,
label, anchored flag) and graph.npz; labels are sheet indices with the cut at theta = +-pi
about the umbilicus at each point's z, absolute numbering from winding_inference seeds.
"""
import argparse, glob, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sheetgraph as sgm

W = '/ephemeral/sean/experiments/winding_fields'
ap = argparse.ArgumentParser()
ap.add_argument('--z0', type=float, required=True, help='base-frame z range [z0, z1)')
ap.add_argument('--z1', type=float, required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--tracks', default=f'{W}/data/tracks/2um_ds2_ps256_surf_v2.dbm.vctracks')
ap.add_argument('--crossings', default=f'{W}/data/tracks/2um_ds2_ps256_surf_v2.dbm.crossings.npz')
ap.add_argument('--wi_root', default=f'{W}/data/winding_inference')
ap.add_argument('--seg', type=int, default=16, help='points per segment (2 base vx apart)')
ap.add_argument('--w_along', type=float, default=1.0)
ap.add_argument('--w_cross', type=float, default=1.0)
ap.add_argument('--w_wi', type=float, default=5.0)
ap.add_argument('--wi_max_d', type=float, default=4.0, help='base vx: snap radius of a wi crossing to a track point')
ap.add_argument('--sigma', type=float, default=-1.0)
ap.add_argument('--irls', type=int, default=15)
ap.add_argument('--cg', type=int, default=300)
ap.add_argument('--eps', type=float, default=0.05, help='IRLS smoothing of |r| near 0')
ap.add_argument('--no_wi', action='store_true', help='no winding_inference at all: only the largest component is anchored, numbering starts near 0')
ap.add_argument('--round_cell', type=float, default=0.0, help='base vx: round against the local phase of 3D cells this size (0 = per component)')
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
t0 = time.time()
def log(*a):
    print(f'[{time.time() - t0:6.0f}s]', *a, flush=True)

U = np.array([[p['z'], p['y'], p['x']] for p in json.load(open(f'{W}/data/umbilicus.json'))['control_points']], float)
U = U[np.argsort(U[:, 0])]
def theta_of(z, y, x):
    return np.arctan2(y - np.interp(z, U[:, 0], U[:, 1]), x - np.interp(z, U[:, 0], U[:, 2]))

def ragged(starts, lens):
    """flat indices of the ranges [starts[i], starts[i] + lens[i])"""
    tot = int(lens.sum())
    first = np.repeat(np.cumsum(lens) - lens, lens)
    return np.repeat(starts, lens) + (np.arange(tot) - first)

# ---------------- tracks overlapping the band ----------------
offs = np.memmap(f'{args.tracks}/offsets.i64', dtype='<i8', mode='r')
coords = np.memmap(f'{args.tracks}/coordinates.i32', dtype='<i4', mode='r').reshape(-1, 3)
fam_all = np.fromfile(f'{args.tracks}/family_codes.i8', dtype=np.int8)
zbd = np.fromfile(f'{args.tracks}/z_bounds.i32', dtype='<i4').reshape(-1, 2)
sel = np.nonzero((zbd[:, 1] >= args.z0) & (zbd[:, 0] < args.z1) & (fam_all >= 0))[0]
starts = np.asarray(offs[sel]); lens = np.asarray(offs[sel + 1]) - starts
P = np.asarray(coords[ragged(starts, lens)]).astype(np.float32)          # zyx base
trk = np.repeat(np.arange(len(sel)), lens)                                 # row in sel
sstart = np.cumsum(lens) - lens                                            # first point of each row
log(f'tracks overlapping z [{args.z0:.0f}, {args.z1:.0f}): {len(sel)} ({np.mean(fam_all[sel] == 0):.2f} H), points {len(P)}')
kept = (P[:, 0] >= args.z0) & (P[:, 0] < args.z1)
th = theta_of(P[:, 0], P[:, 1], P[:, 2]).astype(np.float32)

# ---------------- segments ----------------
n = len(P)
first = np.zeros(n, bool); first[sstart] = True
prev_ok = np.r_[False, kept[:-1]] & ~first                                 # previous point kept, same track
run_start = kept & ~prev_ok
rs_idx = np.where(run_start, np.arange(n), 0); rs_idx = np.maximum.accumulate(rs_idx)
rank = np.arange(n) - rs_idx
cut = np.zeros(n, bool); cut[1:] = np.abs(np.diff(th)) > np.pi
newseg = kept & (run_start | (rank % args.seg == 0) | (prev_ok & cut))
seg = np.where(kept, np.cumsum(newseg) - 1, -1).astype(np.int64)
nseg = int(newseg.sum())
log(f'kept points {kept.sum()}, segments {nseg} (median {np.median(np.bincount(seg[kept])):.0f} points)')

def cutd(ia, ib):
    return sgm.cut_term(th[ia].astype(np.float64), th[ib].astype(np.float64), args.sigma)

parts = []
# along
i = np.nonzero(kept & prev_ok)[0]
i = i[seg[i] != seg[i - 1]]
parts.append(('along', seg[i - 1], seg[i], cutd(i - 1, i), args.w_along))
log(f'along edges {len(i)}')

# crossings
cz = np.load(args.crossings, mmap_mode='r')
co = np.asarray(cz['offsets'])
row_of = np.full(len(fam_all), -1, np.int64); row_of[sel] = np.arange(len(sel))
cs, cl = co[sel], co[sel + 1] - co[sel]
ci = ragged(cs, cl)
ca = np.repeat(np.arange(len(sel)), cl)                                    # row of self track
q = np.asarray(cz['partners'][ci]).astype(np.int64)
lvd = np.asarray(cz['level_diff'][ci]).astype(np.float64) if 'level_diff' in cz.files else np.zeros(len(ci))   # label(partner) - label(self)
sl = np.asarray(cz['self_local'][ci]).astype(np.int64); pl = np.asarray(cz['partner_local'][ci]).astype(np.int64)
qr = row_of[q]
m = (qr >= 0) & (sel[ca] < q)                                              # each undirected crossing once
pa = sstart[ca[m]] + sl[m]; pb = sstart[qr[m]] + pl[m]; lvd = lvd[m]
okp = (sl[m] < lens[ca[m]]) & (pl[m] < lens[qr[m]])
pa, pb, lvd = pa[okp], pb[okp], lvd[okp]
both = kept[pa] & kept[pb]
pa, pb, lvd = pa[both], pb[both], lvd[both]
dvox = np.abs(P[pa] - P[pb]).max(1)
log(f'crossings in band: {len(pa)} (point mismatch > 1 vx: {np.mean(dvox > 1):.4f})')
log(f'crossing level differences: ' + ', '.join(f'{v:+.0f}: {c}' for v, c in zip(*np.unique(lvd, return_counts=True))))
parts.append(('crossing', seg[pa], seg[pb], cutd(pa, pb) + lvd, args.w_cross))
del ci, ca, q, sl, pl, qr, lvd

# winding_inference
from scipy.spatial import cKDTree
kidx = np.nonzero(kept)[0]
tree = None if args.no_wi else cKDTree(P[kidx])
log('kd-tree on kept points built' if not args.no_wi else 'no_wi: winding_inference not used')
A_, B_, D_, AN, AV = [], [], [], [], []
nwi = nsnap = 0
for sh in ([] if args.no_wi else sorted(glob.glob(args.wi_root + '/shard_*'))):
    o = np.load(sh + '/ray_origin_zyx.npy'); st = np.load(sh + '/ray_step_zyx.npy')
    off = np.load(sh + '/crossing_offsets.npy'); ct = np.load(sh + '/crossing_t.npy'); clv = np.load(sh + '/crossing_level.npy')
    swr = np.load(sh + '/seed_winding.npy')
    anc = o + 192 * st
    rsel = np.nonzero((anc[:, 0] >= args.z0 - 300) & (anc[:, 0] < args.z1 + 300))[0]
    if not len(rsel):
        continue
    cnt = off[rsel + 1] - off[rsel]
    rid = np.repeat(rsel, cnt); cix = ragged(off[rsel], cnt)
    p = o[rid] + ct[cix][:, None] * st[rid]
    inb = (p[:, 0] >= args.z0) & (p[:, 0] < args.z1)
    rid, cix, p = rid[inb], cix[inb], p[inb]
    nwi += len(p)
    dd, j = tree.query(p, distance_upper_bound=args.wi_max_d)
    ok = np.isfinite(dd)
    nsnap += int(ok.sum())
    pt = kidx[j[ok]]; r2 = rid[ok]; l2 = clv[cix[ok]].astype(np.float64)
    AN.append(seg[pt]); AV.append((swr[r2] + l2))
    same = r2[1:] == r2[:-1]
    a, b = pt[:-1][same], pt[1:][same]
    dl = (l2[1:] - l2[:-1])[same]
    dfr = seg[a] != seg[b]
    A_.append(seg[a][dfr]); B_.append(seg[b][dfr]); D_.append((dl + cutd(a, b))[dfr])
cat = lambda L, dt: np.concatenate(L) if L else np.zeros(0, dt)
parts.append(('wi', cat(A_, np.int64), cat(B_, np.int64), cat(D_, np.float64), args.w_wi))
anc_n, anc_v = cat(AN, np.int64), cat(AV, np.float64)
log(f'wi crossings in band {nwi}, snapped {nsnap} ({nsnap / max(nwi, 1):.3f}); wi edges {len(parts[-1][1])}')
del tree

# ---------------- aggregate + components ----------------
AA, BB, DD, WW, KK = [], [], [], [], []
for ti, (name, a, b, d, w) in enumerate(parts):
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    dd = np.where(a <= b, d, -d)
    key = np.stack([lo, hi, np.rint(dd * 2).astype(np.int64)], 1)
    uk, c = np.unique(key, axis=0, return_counts=True)
    nl = uk[:, 0] != uk[:, 1]                                              # drop self loops (both ends in one segment)
    uk, c = uk[nl], c[nl]
    AA.append(uk[:, 0]); BB.append(uk[:, 1]); DD.append(uk[:, 2] / 2.0); WW.append(w * np.sqrt(c)); KK.append(np.full(len(uk), ti))
    log(f'aggregate {name}: {len(a)} -> {len(uk)} edges; nonzero d {np.mean(uk[:, 2] != 0):.4f}')
A = np.concatenate(AA); B = np.concatenate(BB); D = np.concatenate(DD); Wt = np.concatenate(WW); K = np.concatenate(KK)
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
ncomp, lab = connected_components(coo_matrix((np.ones(len(A)), (A, B)), shape=(nseg, nseg)), directed=False)
csz = np.bincount(lab)
has_wi = np.zeros(ncomp, bool); has_wi[lab[parts[-1][1]]] = True
if args.no_wi:
    has_wi[np.argmax(np.bincount(lab, np.bincount(seg[kept], minlength=nseg), minlength=ncomp))] = True   # largest by points
anchored = has_wi[lab]
segsz = np.bincount(seg[kept], minlength=nseg)
log(f'components {ncomp}; largest {np.sort(csz)[::-1][:5].tolist()}; points in components with a wi edge {segsz[anchored].sum() / segsz.sum():.3f} '
    f'(largest component alone {segsz[lab == np.argmax(csz)].sum() / segsz.sum():.3f})')

# ---------------- solve ----------------
x, res = sgm.solve_l1(nseg, A, B, D, Wt, np.zeros(nseg), 1e-6, irls=args.irls, cg_iters=args.cg, eps=args.eps, log=lambda *a: log(*a))
for ti, (name, *_r) in enumerate(parts):
    mm = K == ti
    if mm.any():
        log(f'  {name}: off by >0.5 {np.mean(res[mm] > 0.5):.4f} (n={mm.sum()}), weighted share of L1 {float((Wt[mm] * res[mm]).sum() / (Wt * res).sum()):.3f}')
# each component floats by a constant: remove its (point-weighted circular mean) phase before rounding
ang = 2 * np.pi * x
pc = np.bincount(lab, segsz * np.cos(ang), minlength=ncomp); ps = np.bincount(lab, segsz * np.sin(ang), minlength=ncomp)
phase = np.arctan2(ps, pc) / (2 * np.pi)
x = x - phase[lab]
if args.round_cell > 0:
    # local phase: point-weighted circular mean of x over 3D cells (cell + its 26 neighbours), per segment centroid
    cen = np.stack([np.bincount(seg[kept], P[kept, a], minlength=nseg) for a in range(3)], 1) / np.maximum(segsz, 1)[:, None]
    ci3 = np.floor(cen / args.round_cell).astype(np.int64)
    ci3 -= ci3.min(0)
    dims = ci3.max(0) + 3
    flat = lambda c: ((c[:, 0] + 1) * dims[1] + (c[:, 1] + 1)) * dims[2] + (c[:, 2] + 1)
    key = flat(ci3)
    ang = 2 * np.pi * x
    cc = np.bincount(key, segsz * np.cos(ang), minlength=int(np.prod(dims))); ss = np.bincount(key, segsz * np.sin(ang), minlength=int(np.prod(dims)))
    C3, S3 = cc.reshape(dims), ss.reshape(dims)
    from scipy.ndimage import uniform_filter
    C3 = uniform_filter(C3, 3, mode='constant'); S3 = uniform_filter(S3, 3, mode='constant')
    lph = np.arctan2(S3.ravel()[key], C3.ravel()[key]) / (2 * np.pi)
    x = lph + np.rint(x - lph)                                            # snap to integers relative to the local phase
    log(f'local-phase rounding over {args.round_cell:.0f} vx cells')
drift = np.abs(((x + 0.5) % 1.0) - 0.5)
log(f'|x - round(x)| after removing each component\'s phase: p50 {np.median(drift[anchored]):.3f} p90 {np.percentile(drift[anchored], 90):.3f} (anchored segments)')
# absolute numbering from winding_inference seeds (per component: each anchored component has its own offset)
offc = np.full(ncomp, np.nan)
r_ = anc_v - x[anc_n]
cl_ = lab[anc_n]
order = np.argsort(cl_, kind='stable'); cl_s, r_s = cl_[order], r_[order]
bnd = np.r_[0, np.nonzero(np.diff(cl_s))[0] + 1, len(cl_s)]
for s0, s1 in (zip(bnd[:-1], bnd[1:]) if len(cl_s) else []):
    offc[cl_s[s0]] = np.rint(np.median(r_s[s0:s1]))
if args.no_wi:
    big = np.nonzero(has_wi)[0][0]
    offc[big] = -np.floor(np.percentile(x[lab == big], 0.05))            # arbitrary numbering, innermost ~0
xl = x + np.nan_to_num(offc[lab])
agree = np.mean(np.abs(anc_v - xl[anc_n]) < 0.5) if len(anc_n) else float('nan')
log(f'absolute: {np.isfinite(offc).sum()} anchored components; wi seed+level agree within 0.5: {agree:.3f}')
label = np.rint(xl)

kk = kept
np.savez(f'{args.out}/points.npz', zyx=P[kk], fam=fam_all[sel][trk[kk]], theta=th[kk], seg=seg[kk].astype(np.int32),
         label=label[seg[kk]].astype(np.float32), xraw=xl[seg[kk]].astype(np.float32), anchored=anchored[seg[kk]],
         track=sel[trk[kk]].astype(np.int64), sigma=args.sigma, z0=args.z0, z1=args.z1)
np.savez(f'{args.out}/graph.npz', A=A, B=B, D=D, W=Wt, K=K, kinds=np.array([p[0] for p in parts]), x=x, comp=lab, nseg=nseg)
json.dump(vars(args), open(f'{args.out}/args.json', 'w'), indent=1)
log(f'wrote {args.out}/points.npz ({kk.sum()} points)')
