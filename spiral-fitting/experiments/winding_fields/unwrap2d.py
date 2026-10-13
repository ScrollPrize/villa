"""Unwrap the H/V phase into a winding field w (1 per sheet, 0.5 per layer)."""
import sys
import numpy as np
import torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

dev = 'cuda'
d = np.load(sys.argv[1])
phi, mag, mask, cx, cy, s = d['phi'], d['mag'], d['mask'], float(d['cx']), float(d['cy']), d['s']
out = sys.argv[2]
Hh, Ww = phi.shape
yy, xx = np.mgrid[0:Hh, 0:Ww]
theta = np.arctan2(yy - cy, xx - cx)

def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

# handedness: winding number of phi along circles around the center
for r in [200, 400, 600, 800]:
    t = np.linspace(0, 2 * np.pi, 4000, endpoint=False)
    px = np.rint(cx + r * np.cos(t)).astype(int); py = np.rint(cy + r * np.sin(t)).astype(int)
    ph = phi[py, px]
    print('r', r, 'phase winding number', np.sum(wrap(np.diff(np.r_[ph, ph[0]]))) / (2 * np.pi),
          'in-mask frac', mask[py, px].mean())
sigma = float(sys.argv[3]) if len(sys.argv) > 3 else None
if sigma is None:
    t = np.linspace(0, 2 * np.pi, 4000, endpoint=False)
    ws = []
    for r in [300, 500, 700]:
        px = np.rint(cx + r * np.cos(t)).astype(int); py = np.rint(cy + r * np.sin(t)).astype(int)
        ph = phi[py, px]; ws.append(np.sum(wrap(np.diff(np.r_[ph, ph[0]]))) / (2 * np.pi))
    sigma = float(np.sign(np.median(ws)))
print('sigma', sigma)

psi = wrap(phi - sigma * theta)
wt = np.where(mask, mag, 0).astype(np.float32)
wt = wt / np.percentile(wt[mask], 90)
wt = np.clip(wt, 0, 1) ** 2 + 1e-4

P = torch.from_numpy(psi.astype(np.float32)).to(dev)
Wt = torch.from_numpy(wt).to(dev)
gx = torch.remainder(P[:, 1:] - P[:, :-1] + np.pi, 2 * np.pi) - np.pi
gy = torch.remainder(P[1:, :] - P[:-1, :] + np.pi, 2 * np.pi) - np.pi
wx = torch.minimum(Wt[:, 1:], Wt[:, :-1])
wy = torch.minimum(Wt[1:, :], Wt[:-1, :])

def A(u):
    r = torch.zeros_like(u)
    fx = wx * (u[:, 1:] - u[:, :-1]); fy = wy * (u[1:, :] - u[:-1, :])
    r[:, 1:] += fx; r[:, :-1] -= fx; r[1:, :] += fy; r[:-1, :] -= fy
    return r + 1e-6 * u
b = torch.zeros_like(P)
fx = wx * gx; fy = wy * gy
b[:, 1:] += fx; b[:, :-1] -= fx; b[1:, :] += fy; b[:-1, :] -= fy

def cg(u, iters):
    r = b - A(u); pdir = r.clone(); rs = (r * r).sum()
    for i in range(iters):
        Ap = A(pdir); alpha = rs / (pdir * Ap).sum()
        u = u + alpha * pdir; r = r - alpha * Ap
        rn = (r * r).sum()
        pdir = r + (rn / rs) * pdir; rs = rn
    return u, rs.sqrt().item()

# coarse-to-fine init via pooled problems is overkill here; plain CG from the wrapped psi
u = P.clone()
for it in range(6):
    u, res = cg(u, 1000)
    print('cg block', it, 'residual', res)
u = u.cpu().numpy()
w = (u + sigma * theta) / (2 * np.pi)
# consistency: fraction of edges where the unwrapped gradient disagrees with the wrapped one
ux = np.diff(u, axis=1); dx = gx.cpu().numpy()
bad = (np.abs(ux - dx) > np.pi / 2) & mask[:, 1:] & mask[:, :-1]
print('defect edge frac', bad.sum() / (mask[:, 1:] & mask[:, :-1]).sum())
# agreement: s (+H/-V) vs cos(2 pi w); H peaks should sit at phase 0
m = mask & (np.abs(s) > 0.02)
print('corr(s, cos 2pi w)', np.corrcoef(s[m], np.cos(2 * np.pi * w[m]))[0, 1])
np.savez_compressed(out, w=w.astype(np.float32), mask=mask, sigma=sigma)

# figures
ys, xs = np.nonzero(mask)
y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
img = np.where(mask, np.mod(w, 1.0), np.nan)
plt.figure(figsize=(14, 12))
plt.imshow(img[y0:y1, x0:x1], cmap='twilight', interpolation='nearest')
plt.contour(np.where(mask, w, np.nan)[y0:y1, x0:x1], levels=np.arange(np.nanmin(w[mask]) // 1, np.nanmax(w[mask]) + 1, 5.0), colors='k', linewidths=0.6)
plt.colorbar(); plt.title('frac(w); black contours every 5 windings')
plt.savefig(out.replace('.npz', '_frac.png'), dpi=80); plt.close()
plt.figure(figsize=(14, 12))
plt.imshow(np.where(mask, w, np.nan)[y0:y1, x0:x1], cmap='viridis'); plt.colorbar(); plt.title('w')
plt.savefig(out.replace('.npz', '_w.png'), dpi=80); plt.close()
