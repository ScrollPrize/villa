"""Side view of exported windings: radius from the umbilicus vs z at a few fixed angles,
for several tifxyz runs side by side; slab seams (owner changes) marked."""
import glob, json, sys
import numpy as np, tifffile
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
W = '/ephemeral/sean/experiments/winding_fields'
runs, out = sys.argv[1:-1], sys.argv[-1]
U = np.array([[p['z'], p['y'], p['x']] for p in json.load(open(f'{W}/data/umbilicus.json'))['control_points']], float); U = U[np.argsort(U[:, 0])]
meta = json.load(open(f'{runs[0]}/offsets.json'))
cz = [tuple(np.load(f'{sd}/w.npz', mmap_mode='r')['zs'][[0, -1]]) for sd in meta['slabs']]
owner = lambda zl: min((abs(zl - (a + b) / 2), i) for i, (a, b) in enumerate(cz) if a <= zl <= b)[1]
zz = np.arange(cz[0][0], cz[-1][1] + 1, 3)
seams = [2 * zz[i + 1] for i in range(len(zz) - 1) if owner(zz[i]) != owner(zz[i + 1])]
angles = [0.0, np.pi / 2, np.pi, -np.pi / 2]
fig, axs = plt.subplots(len(angles), len(runs), figsize=(11 * len(runs), 9 * len(angles)), squeeze=False)
for j, r in enumerate(runs):
    for d in sorted(glob.glob(f'{r}/tifxyz/winding_*'))[::2]:
        X = tifffile.imread(f'{d}/x.tif'); Y = tifffile.imread(f'{d}/y.tif'); Z = tifffile.imread(f'{d}/z.tif')
        v = Z > 0
        ux = np.interp(Z, U[:, 0], U[:, 2]); uy = np.interp(Z, U[:, 0], U[:, 1])
        th = np.arctan2(Y - uy, X - ux); rad = np.hypot(X - ux, Y - uy)
        for i, a in enumerate(angles):
            m = v & (np.abs(np.angle(np.exp(1j * (th - a)))) < 0.01)
            if m.sum() < 5: continue
            rr, zr = rad[m], Z[m]; o = np.argsort(zr)
            axs[i, j].plot(rr[o], zr[o], '.-', ms=1.5, lw=0.5)
    for i, a in enumerate(angles):
        ax = axs[i, j]
        for s in seams: ax.axhline(s, color='k', lw=0.4, alpha=0.4)
        ax.set_title(f'{r.split("/")[-1]}  angle {np.degrees(a):.0f} deg  (every 2nd winding; grey = slab seams)')
        ax.set_xlabel('radius from umbilicus (base vx)'); ax.set_ylabel('z (base vx)'); ax.invert_yaxis()
plt.tight_layout(); plt.savefig(out, dpi=45); print('wrote', out)
