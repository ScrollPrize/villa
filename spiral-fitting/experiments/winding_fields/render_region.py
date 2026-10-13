"""Render surf bands, frac(w) and w for a crop of one slab slice (crop px, run-relative)."""
import sys
import numpy as np, zarr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
run, out = sys.argv[1], sys.argv[2]; ya, yb, xa, xb = map(int, sys.argv[3:7])
d = np.load(run); w = d["w"]; m = d["mask"]; zs = d["zs"]; y0, y1, x0, x1 = d["crop"]; k = len(zs) // 2
surf = zarr.open("data/2um_ds2_ps256_surf.zarr/1", mode="r")[int(zs[k]), y0 + ya:y0 + yb, x0 + xa:x0 + xb]
fig, ax = plt.subplots(1, 3, figsize=(33, 9))
ax[0].imshow(surf, cmap="gray"); ax[0].set_title("surf (front) bands")
ax[1].imshow(np.where(m[k, ya:yb, xa:xb], np.mod(w[k, ya:yb, xa:xb], 1), np.nan), cmap="twilight", interpolation="nearest"); ax[1].set_title("frac(w)")
im = ax[2].imshow(np.where(m[k, ya:yb, xa:xb], w[k, ya:yb, xa:xb], np.nan), cmap="turbo"); plt.colorbar(im, ax=ax[2]); ax[2].set_title("w")
plt.tight_layout(); plt.savefig(out, dpi=50)
