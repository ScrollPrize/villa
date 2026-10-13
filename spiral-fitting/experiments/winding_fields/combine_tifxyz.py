"""All per-winding tifxyz (one turn each, columns uniform in angle -pi..pi about the row's
umbilicus) concatenated along columns into one spiral tifxyz. Rows are aligned by z
(each per-winding file is trimmed to its valid rows). The joining order (turn k's end to
turn k+1's start, or the reverse) is chosen from the measured gap between the two ends."""
import glob, json, os, sys
import numpy as np, tifffile
sys.path.insert(0, '/ephemeral/sean/villa/spiral-fitting')
from tifxyz import save_tifxyz
src, out_dir, name = sys.argv[1], sys.argv[2], sys.argv[3]
step = 18.0
S = []
for d in sorted(glob.glob(f'{src}/winding_*')):
    k = int(d.rsplit('_', 1)[1])
    zyx = np.stack([tifffile.imread(f'{d}/{c}.tif') for c in 'zyx'], -1)
    S.append((k, zyx))
zall = np.concatenate([s[1][..., 0][s[1][..., 0] > 0] for s in S])
z0 = zall.min()
rowz = lambda zyx: np.array([np.median(r[r > 0]) if (r > 0).any() else np.nan for r in zyx[..., 0]])
nrow = int(round((zall.max() - z0) / step)) + 1
def place(zyx):                                   # first global row index of this file
    rz = rowz(zyx); i = np.nonzero(np.isfinite(rz))[0][0]
    return int(round((rz[i] - z0) / step)) - i
# gap between consecutive turns at the join, both orientations
def gap(a, b, ia, ib):                            # last valid column of a's row vs first valid column of b's row
    g = []
    for r in range(nrow):
        ra, rb = r - ia, r - ib
        if 0 <= ra < len(a) and 0 <= rb < len(b):
            va = np.nonzero(a[ra, :, 0] > 0)[0]; vb = np.nonzero(b[rb, :, 0] > 0)[0]
            if len(va) and len(vb) and va[-1] >= a.shape[1] - 3 and vb[0] <= 2:
                g.append(np.linalg.norm(a[ra, va[-1], 1:] - b[rb, vb[0], 1:]))
    return g
fw, bw = [], []
for (k, a), (k1, b) in zip(S[:-1], S[1:]):
    if k1 != k + 1: continue
    ia, ib = place(a), place(b)
    fw += gap(a, b, ia, ib); bw += gap(b, a, ib, ia)
print(f'join gaps (base vx): end(k)->start(k+1) p50 {np.median(fw):.1f} p90 {np.percentile(fw, 90):.1f} n {len(fw)}; '
      f'end(k+1)->start(k) p50 {np.median(bw):.1f} p90 {np.percentile(bw, 90):.1f} n {len(bw)}')
forward = np.median(fw) <= np.median(bw)
order = S if forward else S[::-1]
ncol = sum(s[1].shape[1] for s in order)
out = np.full((nrow, ncol, 3), -1.0, np.float32)
c0 = 0; cols = []
for k, zyx in order:
    i = place(zyx)
    out[i:i + len(zyx), c0:c0 + zyx.shape[1]] = zyx
    cols.append({'winding': k, 'col_start': c0, 'col_end': c0 + zyx.shape[1]}); c0 += zyx.shape[1]
os.makedirs(out_dir, exist_ok=True)
save_tifxyz(out, out_dir, name, step, 9.6, 'winding_fields joint sheet-graph: all integer windings concatenated along columns')
json.dump({'turn_order': 'inner to outer' if forward else 'outer to inner', 'windings': cols}, open(f'{out_dir}/{name}/windings_columns.json', 'w'), indent=1)
print(f'{name}: {nrow} x {ncol}, valid {np.mean(out[..., 0] > 0):.3f}, order {"inner->outer" if forward else "outer->inner"}')
