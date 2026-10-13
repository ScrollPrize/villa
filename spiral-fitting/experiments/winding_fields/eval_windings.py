"""Score a solve3d winding field against relative_windings.json.

For each collection, every pair of points inside the slab: the field's
Delta w should equal the annotated Delta wind_a (sign convention auto-detected
from the data and reported). Points are in the base (level-2, 9.6 um) frame.
"""
import json, sys
import numpy as np

ann = json.load(open(sys.argv[2]))['collections']
slabs = [{k: np.load(f)[k] for k in ('w', 'mask', 'zs', 'crop', 'level')} for f in sys.argv[1].split(',')]

def sample(p):
    # returns (slab index, w) from the first slab containing the point; pairs are only formed within a slab
    for si, d in enumerate(slabs):
        v = sample_one(d, p)
        if v is not None:
            return (si, v)
    return None

def sample_one(d, p):
    w, mask, zs, crop = d['w'], d['mask'], d['zs'], d['crop']
    scale = 2 ** (int(d['level']) - 2)                      # base frame = fiber level 2
    y0, y1, x0, x1 = crop
    dz = int(zs[1] - zs[0]) if len(zs) > 1 else 1
    x, y, z = p[0] / scale, p[1] / scale, p[2] / scale
    k = (z - zs[0]) / dz
    if k < -0.5 or k > len(zs) - 0.5:
        return None
    k = int(round(min(max(k, 0), len(zs) - 1)))
    iy, ix = int(round(y)) - y0, int(round(x)) - x0
    if not (0 <= iy < w.shape[1] and 0 <= ix < w.shape[2]) or not mask[k, iy, ix]:
        return None
    # median over a 3x3 in-plane window to dodge single-pixel defects
    win = w[k, max(iy - 1, 0):iy + 2, max(ix - 1, 0):ix + 2]
    return float(np.median(win))

pairs = []
npts = 0
for cid, c in ann.items():
    pts = [(sample(p['p']), p['wind_a']) for p in c['points'].values()]
    pts = [q for q in pts if q[0] is not None]
    npts += len(pts)
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            if pts[i][0][0] != pts[j][0][0]:
                continue
            pairs.append((cid, pts[i][0][1] - pts[j][0][1], pts[i][1] - pts[j][1]))
if not pairs:
    print('no annotated pairs inside slab'); sys.exit()
dw = np.array([p[1] for p in pairs]); da = np.array([p[2] for p in pairs], float)
nz = da != 0
sgn = np.sign(np.sum(dw[nz] * da[nz])) or 1.0
err = sgn * dw - da
print(f'points in slab {npts}, pairs {len(pairs)} ({nz.sum()} with nonzero delta), collections {len(set(p[0] for p in pairs))}')
print(f'sign convention: w {"increases" if sgn > 0 else "decreases"} with wind_a')
print(f'|err| p50 {np.median(np.abs(err)):.2f}  p90 {np.percentile(np.abs(err), 90):.2f}')
print(f'exact (|err|<0.5): {np.mean(np.abs(err) < 0.5):.3f}   within 1: {np.mean(np.abs(err) < 1.5):.3f}')
rel = np.abs(err[nz]) / np.abs(da[nz])
print(f'relative error on nonzero pairs p50 {np.median(rel):.3f}; mean signed err/delta {np.mean(err[nz] / da[nz]):+.3f}')
# fractional phase of annotation points (do they sit on a consistent layer?)
fr = []
for cid, c in ann.items():
    for p in c['points'].values():
        v = sample(p['p'])
        if v is not None:
            fr.append(v[1] % 1.0)
fr = np.array(fr)
R = np.abs(np.mean(np.exp(2j * np.pi * fr)))
print(f'frac(w) at annotation points: circular mean {np.angle(np.mean(np.exp(2j*np.pi*fr)))/(2*np.pi)%1:.2f}, concentration R {R:.2f} (1 = all same phase)')
