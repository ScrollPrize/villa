"""Render flattened tifxyz meshes side by side: z, radius about the umbilicus (at the vertex z), valid mask.
usage: render_flat.py out.png label1 mesh1.tifxyz [label2 mesh2.tifxyz ...]"""
import json, sys
import numpy as np, tifffile
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

W = '/ephemeral/sean/experiments/winding_fields'
out, args = sys.argv[1], sys.argv[2:]
meshes = list(zip(args[::2], args[1::2]))
U = np.array([[p['z'], p['y'], p['x']] for p in json.load(open(f'{W}/data/umbilicus.json'))['control_points']], float)
U = U[np.argsort(U[:, 0])]
fig, ax = plt.subplots(3, len(meshes), figsize=(9 * len(meshes), 9), squeeze=False)
for j, (lab, d) in enumerate(meshes):
    X, Y, Z = (tifffile.imread(f'{d}/{c}.tif').astype(np.float32) for c in 'xyz')
    f = max(1, X.shape[1] // 3000)
    X, Y, Z = X[:, ::f], Y[:, ::f], Z[:, ::f]
    v = (X > 0) & (Y > 0) & (Z > 0)
    R = np.hypot(Y - np.interp(Z, U[:, 0], U[:, 1]), X - np.interp(Z, U[:, 0], U[:, 2]))
    m = json.load(open(f'{d}/meta.json'))
    for i, (A, cm, nm) in enumerate(((np.where(v, Z, np.nan), 'viridis', 'z (base vx)'), (np.where(v, R, np.nan), 'magma', 'radius about umbilicus'), (v.astype(float), 'gray', 'valid'))):
        im = ax[i, j].imshow(A, cmap=cm, aspect='auto', interpolation='nearest')
        ax[i, j].set_title(f'{lab}: {nm}' + (f'  ({A.shape[0]}x{A.shape[1] * f}, valid {v.mean():.2f}, {m.get("area_cm2", 0):.0f} cm2)' if i == 0 else ''), fontsize=9)
        if i < 2:
            plt.colorbar(im, ax=ax[i, j], fraction=0.02)
plt.tight_layout(); plt.savefig(out, dpi=110)
print('wrote', out)
