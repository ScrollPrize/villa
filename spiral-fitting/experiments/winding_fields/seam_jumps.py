"""z-continuity of exported tifxyz: in-plane jump of each vertex between consecutive rows,
split by whether the two rows come from different slabs (seam) or the same slab."""
import glob, json, sys
import numpy as np, tifffile
from scipy.spatial import cKDTree
root = sys.argv[1]
meta = json.load(open(f'{root}/offsets.json'))
cz = []
for sd in meta['slabs']:
    zs = np.load(f'{sd}/w.npz', mmap_mode='r')['zs']; cz.append((zs[0], zs[-1]))
def owner(zl):                                    # same rule as extract_tifxyz: nearest slab centre among slabs containing z
    return min((abs(zl - (a + b) / 2), i) for i, (a, b) in enumerate(cz) if a <= zl <= b)[1]
J = {True: [], False: []}; N = {True: [], False: []}
for d in sorted(glob.glob(f'{root}/tifxyz/winding_*')):
    X = tifffile.imread(f'{d}/x.tif'); Y = tifffile.imread(f'{d}/y.tif'); Z = tifffile.imread(f'{d}/z.tif')
    for r in range(X.shape[0] - 1):
        v = (Z[r] > 0) & (Z[r + 1] > 0)
        if v.sum() < 10: continue
        z0, z1 = np.median(Z[r, v]) / 2, np.median(Z[r + 1, v]) / 2
        seam = owner(round(z0)) != owner(round(z1))
        J[seam].append(np.hypot(X[r + 1, v] - X[r, v], Y[r + 1, v] - Y[r, v]))
        a = np.stack([X[r, Z[r] > 0], Y[r, Z[r] > 0]], 1); b = np.stack([X[r + 1, Z[r + 1] > 0], Y[r + 1, Z[r + 1] > 0]], 1)
        N[seam].append(cKDTree(a).query(b)[0])           # surface continuity, independent of column parameterization
for seam in (False, True):
    j = np.concatenate(J[seam])
    q = np.concatenate(N[seam])
    print(f'{"seam rows  " if seam else "inside slab"}: nearest-point dist p50 {np.median(q):5.1f} p90 {np.percentile(q, 90):6.1f}  >20vx {np.mean(q > 20):.3f}  >50vx {np.mean(q > 50):.3f}')
    print(f'{"seam rows  " if seam else "inside slab"}: n {len(j):9d}  same-column jump p50 {np.median(j):6.1f}  p90 {np.percentile(j, 90):6.1f}  >20vx {np.mean(j > 20):.3f}  >50vx {np.mean(j > 50):.3f}')
