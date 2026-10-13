"""Input-vs-input check: how many surf (front) bands lie between consecutive winding_inference crossings?"""
import sys, glob
import numpy as np, zarr
root, surf_path = sys.argv[1], sys.argv[2]; zb0, zb1 = 9300, 9500
surf = zarr.open(surf_path, mode='r')          # level 1 = base/2
rng = np.random.default_rng(0)
per_gap, on_band = [], []
cache = {}
n = 0
for sh in sorted(glob.glob(root + '/shard_*')):
    o = np.load(sh + '/ray_origin_zyx.npy'); st = np.load(sh + '/ray_step_zyx.npy')
    off = np.load(sh + '/crossing_offsets.npy'); ct = np.load(sh + '/crossing_t.npy')
    anc = o + 192 * st
    sel = np.nonzero((anc[:, 0] >= zb0) & (anc[:, 0] < zb1) & (np.abs(st[:, 0]) < 0.1))[0]   # near-horizontal rays
    for i in rng.permutation(sel)[:400]:
        t = ct[off[i]:off[i + 1]]
        tt = np.arange(t.min(), t.max(), 0.25)
        p = (o[i][None] + tt[:, None] * st[i][None]) / 2.0
        z = int(round(p[:, 0].mean()))
        if z not in cache:
            cache[z] = surf[z] > 127
        b = cache[z][np.clip(np.rint(p[:, 1]).astype(int), 0, 4086), np.clip(np.rint(p[:, 2]).astype(int), 0, 4086)]
        entries = np.nonzero(np.diff(b.astype(int)) == 1)[0]
        e_t = tt[entries]
        cnt = np.histogram(e_t, bins=t)[0]
        per_gap.extend(cnt)
        pc = (o[i][None] + t[:, None] * st[i][None]) / 2.0
        bb = cache[z][np.clip(np.rint(pc[:, 1]).astype(int), 0, 4086), np.clip(np.rint(pc[:, 2]).astype(int), 0, 4086)]
        on_band.extend(bb)
        n += 1
    if n > 1500:
        break
per_gap = np.array(per_gap); on_band = np.array(on_band)
print(f'rays {n}; surf band entries per crossing-to-crossing interval: mean {per_gap.mean():.2f}; distribution', {int(k): int(v) for k, v in zip(*np.unique(per_gap, return_counts=True))})
print(f'crossings lying on a surf band: {on_band.mean():.2f}')
