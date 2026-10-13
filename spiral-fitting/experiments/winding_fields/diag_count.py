"""Where is the winding count lost? Along consecutive annotated windings (expected 1 each):
raw surf band crossings, integral of the bandcount target along the segment, and dw of a solved field."""
import json, sys
import numpy as np
import zarr
from scipy import ndimage as ndi

W = '/ephemeral/sean/experiments/winding_fields/data/'
ann = json.load(open(W + 'relative_windings.json'))['collections']
surf = zarr.open(W + '2um_ds2_ps256_surf.zarr/1', mode='r')
las = {k: zarr.open(W + f'lasagna/las_008_{k}.ome.zarr/4', mode='r') for k in ('nx', 'ny')}
MODE = sys.argv[2] if len(sys.argv) > 2 else 'blur1'
d = np.load(sys.argv[1]); w, zs, crop = d['w'], d['zs'], d['crop']; y0c, _, x0c, _ = crop; dz = int(zs[1] - zs[0])
rows = []
cache = {}
for cid, c in ann.items():
    P = {p['wind_a']: np.array(p['p']) / 2.0 for p in c['points'].values()}     # -> level 3
    for wa in P:
        if wa + 1 not in P or abs(P[wa][2] - P[wa + 1][2]) > 0.5:
            continue
        a, b = P[wa], P[wa + 1]
        k = (a[2] - zs[0]) / dz
        if not (-0.5 <= k <= len(zs) - 0.5):
            continue
        k = int(round(min(max(k, 0), len(zs) - 1))); z = int(round(a[2]))   # raw data at the point's own z
        if z not in cache:
            if MODE == 'blur1':
                band = ndi.gaussian_filter((surf[z] > 127).astype(np.float32), 1.0)
            else:   # dilate 1 px, light smoothing, renormalise each band's peak to 1
                band = ndi.gaussian_filter(ndi.maximum_filter((surf[z] > 127).astype(np.float32), 3), 0.5)
                band = band / np.maximum(ndi.maximum_filter(band, 5), 0.5)
            lnx = (las['nx'][z // 2].astype(np.float32) - 128) / 127; lny = (las['ny'][z // 2].astype(np.float32) - 128) / 127
            lnx = lnx.repeat(2, 0).repeat(2, 1)[:band.shape[0], :band.shape[1]]; lny = lny.repeat(2, 0).repeat(2, 1)[:band.shape[0], :band.shape[1]]
            nn = np.hypot(lnx, lny) + 1e-6
            gy, gx = np.gradient(band)
            tgt = 0.5 * np.abs((lnx * gx + lny * gy) / nn)   # count per px along n (abs mode / 2pi)
            cache[z] = (surf[z] > 127, tgt, lnx / nn, lny / nn)
        bin_, tgt, nx, ny = cache[z]
        L = np.hypot(*(b[:2] - a[:2])); n = int(L * 4) + 2; t = np.linspace(0, 1, n)
        xs = a[0] + t * (b[0] - a[0]); ys = a[1] + t * (b[1] - a[1])
        ix, iy = np.rint(xs).astype(int), np.rint(ys).astype(int)
        v = bin_[iy, ix].astype(int)
        cross = np.sum(np.diff(v) == 1) + 0.5 * (v[0] != v[-1])
        dl = (b[:2] - a[:2]) / L
        ndl = np.abs(nx[iy, ix] * dl[0] + ny[iy, ix] * dl[1])
        integ = np.sum(tgt[iy, ix] * ndl) * L / n
        wi = w[k, iy[[0, -1]] - y0c, ix[[0, -1]] - x0c]
        rows.append((L, np.sum(np.diff(v) == 1), integ, wi[1] - wi[0], ndl.mean()))
r = np.array(rows)
print(f'segments {len(r)}  (each should be 1 winding)')
print(f'raw rising crossings mean {r[:,1].mean():.2f}   target integral mean {r[:,2].mean():.2f}   solved dw mean {r[:,3].mean():.2f}   mean |n.dir| {r[:,4].mean():.2f}')
print('target integral p10/50/90', np.percentile(r[:, 2], [10, 50, 90]).round(2), '  dw p10/50/90', np.percentile(r[:, 3], [10, 50, 90]).round(2))
