"""Joint sheet-graph solve across all full-scroll slabs.

Each slab graph (solve3d --save_graph) is loaded with its node ids offset into one global
numbering. Neighbouring slabs share solved slices; a centreline pixel present in both
slabs' piece maps links the two pieces with an exact "same sheet" edge, plus the theta
branch-cut term between the two slabs' centres (w is a sheet index relative to each
slab's own cut). The solve is warm-started from independent per-slab solves chained by
their overlap offsets, then one global IRLS L1 solve; labels are written per slab.
"""
import glob, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sheetgraph as sgm

root = sys.argv[1]                     # dir with slab_*/graph.npz
w_cross = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
t0 = time.time()
paths = sorted(glob.glob(f'{root}/slab_*/graph.npz'))
print(f'{len(paths)} slab graphs')
G = []
off = 0
for p in paths:
    g = np.load(p)
    n = int(g['npieces'])
    G.append(dict(path=p, n=n, off=off, A=g['A'] + off, B=g['B'] + off, D=g['D'], W=g['W'], ineq=g['ineq'],
                  kind=g['kind'], names=list(g['part_names']), zs=g['zs'], crop=g['crop'],
                  cx=float(g['cx']), cy=float(g['cy']), sigma=float(g['sigma'])))
    off += n
N = off
sigma = G[0]['sigma']
print(f'total pieces {N}, edges {sum(len(g["A"]) for g in G)} ({time.time()-t0:.0f}s)')


def theta_global(g, yy, xx):
    """angle of global level-3 pixels (yy, xx) about slab g's centre (its own branch cut)."""
    y0, y1, x0, x1 = g['crop']
    return np.arctan2(yy - (g['cy'] + y0), xx - (g['cx'] + x0))


# cross-slab edges on shared slices
CA, CB, CD = [], [], []
for ia in range(len(G) - 1):
    ga, gb = G[ia], G[ia + 1]
    common = np.intersect1d(ga['zs'], gb['zs'])
    if len(common) == 0:
        print(f'  WARNING: {os.path.basename(os.path.dirname(ga["path"]))} and next share no slices')
        continue
    pa = np.load(ga['path'])['pid']; pb = np.load(gb['path'])['pid']
    ya0, ya1, xa0, xa1 = ga['crop']; yb0, yb1, xb0, xb1 = gb['crop']
    y0, y1, x0, x1 = max(ya0, yb0), min(ya1, yb1), max(xa0, xb0), min(xa1, xb1)
    for z in common:
        ka = int(np.nonzero(ga['zs'] == z)[0][0]); kb = int(np.nonzero(gb['zs'] == z)[0][0])
        sa = pa[ka, y0 - ya0:y1 - ya0, x0 - xa0:x1 - xa0]; sb = pb[kb, y0 - yb0:y1 - yb0, x0 - xb0:x1 - xb0]
        both = (sa >= 0) & (sb >= 0)
        yy, xx = np.nonzero(both)
        a = sa[both].astype(np.int64) + ga['off']; b = sb[both].astype(np.int64) + gb['off']
        d = sgm.cut_term(theta_global(ga, yy + y0, xx + x0), theta_global(gb, yy + y0, xx + x0), sigma)
        CA.append(a); CB.append(b); CD.append(d)
    del pa, pb
CA = np.concatenate(CA); CB = np.concatenate(CB); CD = np.concatenate(CD)
key = np.stack([CA, CB, np.rint(CD).astype(np.int64)], 1)
uk, cnt = np.unique(key, axis=0, return_counts=True)
CA, CB, CD, CW = uk[:, 0], uk[:, 1], uk[:, 2].astype(np.float64), w_cross * np.sqrt(cnt)
print(f'cross-slab edges: {len(CA)} piece pairs ({cnt.sum()} shared centreline pixels; '
      f'{np.mean(CD != 0):.4f} carry a branch-cut step) ({time.time()-t0:.0f}s)')

# warm start: independent per-slab solves, chained by the median cross-edge offset
x0 = np.zeros(N)
for g in G:
    loc = lambda v: v - g['off']
    xs, _ = sgm.solve_l1(g['n'], loc(g['A']), loc(g['B']), g['D'], g['W'], np.zeros(g['n']), 1e-6,
                         ineq=g['ineq'], ineq_up=0.2, log=lambda *a: None)
    x0[g['off']:g['off'] + g['n']] = xs
for ia in range(len(G) - 1):
    gb = G[ia + 1]
    m = (CB >= gb['off']) & (CB < gb['off'] + gb['n']) & (CA < gb['off'])
    if m.any():
        shift = np.median(x0[CA[m]] + CD[m] - x0[CB[m]])
        x0[gb['off']:] += shift                     # carries to all later slabs
    print(f'  warm start slab {ia + 1}: shift {shift if m.any() else 0:+.2f}', flush=True)
print(f'warm start done ({time.time()-t0:.0f}s)')

# global solve
A = np.concatenate([g['A'] for g in G] + [CA]); B = np.concatenate([g['B'] for g in G] + [CB])
D = np.concatenate([g['D'] for g in G] + [CD]); Wt = np.concatenate([g['W'] for g in G] + [CW])
IQ = np.concatenate([g['ineq'] for g in G] + [np.zeros(len(CA), bool)])
isx = np.r_[np.zeros(len(A) - len(CA), bool), np.ones(len(CA), bool)]
r0 = np.abs(x0[CB] - x0[CA] - CD)
print(f'cross edges before joint solve: off by >0.5 {np.mean(r0 > 0.5):.4f}, residual p50 {np.median(r0):.3f}')
x, res = sgm.solve_l1(N, A, B, D, Wt, x0, 1e-6, ineq=IQ, ineq_up=0.2, irls=10, cg_iters=500)
print(f'cross edges after joint solve: off by >0.5 {np.mean(res[isx] > 0.5):.4f}, residual p50 {np.median(res[isx]):.3f}')
for name in G[0]['names']:
    ki = G[0]['names'].index(name)
    m = np.concatenate([g['kind'] == g['names'].index(name) if name in g['names'] else np.zeros(len(g['A']), bool) for g in G] + [np.zeros(len(CA), bool)])
    if m.any():
        print(f'  {name}: off by >0.5 {np.mean(res[m] > 0.5):.4f} (n={m.sum()})')
for g in G:
    np.savez(os.path.join(os.path.dirname(g['path']), 'labels.npz'), npieces=g['n'], x=x[g['off']:g['off'] + g['n']])
print(f'labels written per slab; total {time.time()-t0:.0f}s')
