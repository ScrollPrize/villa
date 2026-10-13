"""EVAL ONLY: integer sheet index of the eval mesh, densified, for a z band (never an input).

Along each row's valid run, s = sgn*(unwrap(theta) - theta)/2pi is the integer turn count
(it jumps at theta = +-pi about the umbilicus at the vertex z). Runs are linked vertically
by the modal difference over shared columns. The band is upsampled x`up` (bilinear
positions, quads straddling the cut or a component boundary skipped).
Returns (cKDTree of points zyx, S, component, sgn).
"""
import json
import numpy as np, tifffile
from scipy.spatial import cKDTree

W = '/ephemeral/sean/experiments/winding_fields'
MESH = f'{W}/data/eval_mesh_DO_NOT_USE_AS_INPUT/spiral-output-generation-19-lasagna.tifxyz'


def mesh_sheet_points(zb0, zb1, up=4, margin=60):
    U = np.array([[p['z'], p['y'], p['x']] for p in json.load(open(f'{W}/data/umbilicus.json'))['control_points']], float)
    U = U[np.argsort(U[:, 0])]
    X = tifffile.imread(f'{MESH}/x.tif'); Y = tifffile.imread(f'{MESH}/y.tif'); Z = tifffile.imread(f'{MESH}/z.tif')
    valid = (X > 0) & (Y > 0) & (Z > 0)
    rows = np.nonzero(((Z >= zb0 - margin) & (Z <= zb1 + margin) & valid).any(1))[0]
    r0, r1 = rows.min(), rows.max() + 1
    X, Y, Z, valid = X[r0:r1], Y[r0:r1], Z[r0:r1], valid[r0:r1]
    cy, cx = np.interp(Z, U[:, 0], U[:, 1]), np.interp(Z, U[:, 0], U[:, 2])
    TH = np.arctan2(Y - cy, X - cx); R = np.hypot(Y - cy, X - cx)
    s_rel = np.zeros(X.shape, np.int64); run_id = np.full(X.shape, -1, np.int64); nrun = 0; sg = []
    for r in range(X.shape[0]):
        idx = np.nonzero(valid[r])[0]
        for run in np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1):
            if len(run) < 10:
                continue
            u = np.unwrap(TH[r, run])
            if len(run) > 200:
                sg.append(np.sign(np.corrcoef(u, R[r, run])[0, 1]))
            s_rel[r, run] = np.rint((u - TH[r, run]) / (2 * np.pi)).astype(np.int64)
            run_id[r, run] = nrun; nrun += 1
    sgn = 1 if np.sum(sg) >= 0 else -1
    s_rel *= sgn
    a, b = run_id[:-1], run_id[1:]
    m = (a >= 0) & (b >= 0)
    key = np.stack([a[m], b[m], (s_rel[:-1] - s_rel[1:])[m]], 1)
    uk, cnt = np.unique(key, axis=0, return_counts=True)
    best = {}
    for (ra, rb, dd), c in zip(uk, cnt):
        if (ra, rb) not in best or c > best[(ra, rb)][1]:
            best[(ra, rb)] = (dd, c)
    adj = [[] for _ in range(nrun)]
    for (ra, rb), (dd, c) in best.items():
        if c >= 5:
            adj[ra].append((rb, dd)); adj[rb].append((ra, -dd))
    off = np.zeros(nrun, np.int64); comp = np.full(nrun, -1, np.int64); nc = 0
    for s in range(nrun):
        if comp[s] >= 0:
            continue
        comp[s] = nc; st = [s]
        while st:
            x = st.pop()
            for y, dd in adj[x]:
                if comp[y] < 0:
                    comp[y] = nc; off[y] = off[x] + dd; st.append(y)
        nc += 1
    S = np.where(run_id >= 0, s_rel + off[np.maximum(run_id, 0)], -10 ** 9)
    C = np.where(run_id >= 0, comp[np.maximum(run_id, 0)], -1)
    pts, PS, PC = [], [], []
    for r in range(X.shape[0] - 1):
        ok = valid[r, :-1] & valid[r, 1:] & valid[r + 1, :-1] & valid[r + 1, 1:]
        for A in (C, S):
            ok &= (A[r, :-1] == A[r + 1, 1:]) & (A[r, :-1] == A[r, 1:]) & (A[r, :-1] == A[r + 1, :-1])
        c = np.nonzero(ok)[0]
        if not len(c):
            continue
        for f in np.arange(up) / up:
            for g in np.arange(up) / up:
                P = [((1 - f) * ((1 - g) * Q[r, c] + g * Q[r, c + 1]) + f * ((1 - g) * Q[r + 1, c] + g * Q[r + 1, c + 1])) for Q in (Z, Y, X)]
                pts.append(np.stack(P, 1)); PS.append(S[r, c]); PC.append(C[r, c])
    pts = np.concatenate(pts); PS = np.concatenate(PS); PC = np.concatenate(PC)
    sel = (pts[:, 0] >= zb0 - 30) & (pts[:, 0] <= zb1 + 30)
    return cKDTree(pts[sel]), PS[sel], PC[sel], sgn
