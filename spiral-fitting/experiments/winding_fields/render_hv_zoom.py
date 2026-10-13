"""Fiber H (blue) / V (red) close-ups with integer (black) and half-integer (green dashed) w contours."""
import sys, glob
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
run, out = sys.argv[1], sys.argv[2]
d = np.load(f"runs/{run}/w.npz"); w = d["w"]; m = d["mask"]; zs = d["zs"]; y0, y1, x0, x1 = d["crop"]
z0 = int(zs[0]); dz = int(zs[1] - zs[0]); k = len(zs) // 2
pre = [p for p in glob.glob(f"cache/L3_z{z0}-*-{dz}_presence.npy") if int(p.split("_z")[1].split("-")[1]) >= int(zs[-1])][0]
pr = np.load(pre, mmap_mode="r")[k, y0:y1, x0:x1].astype(np.float32) / 255
ax = (np.load(pre.replace("presence", "nx"), mmap_mode="r")[k, y0:y1, x0:x1].astype(np.float32) - 128) / 127
ay = (np.load(pre.replace("presence", "ny"), mmap_mode="r")[k, y0:y1, x0:x1].astype(np.float32) - 128) / 127
nz2 = np.clip(1 - ax * ax - ay * ay, 0, 1)
s = pr * (1 - 2 * nz2)
H_, W_ = w.shape[1:]
boxes = [(int(0.3 * H_), int(0.55 * W_)), (int(0.6 * H_), int(0.35 * W_)), (int(0.45 * H_), int(0.75 * W_))]
fig, axs = plt.subplots(1, 3, figsize=(30, 10))
for a, (cy, cx) in zip(axs, boxes):
    sl = (slice(cy, cy + 160), slice(cx, cx + 160))
    a.imshow(np.where(m[k][sl], s[sl], np.nan), cmap="coolwarm_r", vmin=-0.6, vmax=0.6, interpolation="nearest")
    a.contour(w[k][sl], levels=np.arange(-300, 300, 1.0), colors="k", linewidths=1.0)
    a.contour(w[k][sl], levels=np.arange(-299.5, 300, 1.0), colors="g", linewidths=1.0, linestyles="dashed")
    a.set_title(f"{run} slice {k} at ({cy},{cx}): H blue, V red; w integer black, half green")
plt.tight_layout(); plt.savefig(out, dpi=55); print("wrote", out)
