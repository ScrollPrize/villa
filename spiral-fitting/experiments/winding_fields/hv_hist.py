"""frac(w) histogram of H and V fibers for a run (front/back alignment check)."""
import sys, glob
import numpy as np
run = sys.argv[1]
d = np.load(f"runs/{run}/w.npz"); w = d["w"]; m = d["mask"]; zs = d["zs"]; y0, y1, x0, x1 = d["crop"]
z0 = int(zs[0]); dz = int(zs[1] - zs[0])
pre = [p for p in glob.glob(f"cache/L3_z{z0}-*-{dz}_presence.npy") if int(p.split("_z")[1].split("-")[1]) >= int(zs[-1])][0]
pr = np.load(pre)[:, y0:y1, x0:x1].astype(np.float32) / 255
ax = (np.load(pre.replace("presence", "nx"))[:, y0:y1, x0:x1].astype(np.float32) - 128) / 127
ay = (np.load(pre.replace("presence", "ny"))[:, y0:y1, x0:x1].astype(np.float32) - 128) / 127
nz2 = np.clip(1 - ax * ax - ay * ay, 0, 1)
fr = np.mod(w + 0.25, 1.0) - 0.25
Hm = m & (pr > 0.3) & (nz2 < 0.2); Vm = m & (pr > 0.3) & (nz2 > 0.8)
bins = np.linspace(-0.25, 0.75, 11)
hh = np.histogram(fr[Hm][::5], bins)[0]; hv = np.histogram(fr[Vm][::5], bins)[0]
print(f"{run} frac bins:", " ".join(f"{(a+b)/2:+.2f}" for a, b in zip(bins[:-1], bins[1:])))
print("  H:", " ".join(f"{v:5.2f}" for v in hh / hh.sum()))
print("  V:", " ".join(f"{v:5.2f}" for v in hv / hv.sum()))
print(f"  in front half (|frac|<0.25): H {np.mean(np.abs(fr[Hm][::5]) < 0.25):.2f}  V {np.mean(np.abs(fr[Vm][::5]) < 0.25):.2f}")
