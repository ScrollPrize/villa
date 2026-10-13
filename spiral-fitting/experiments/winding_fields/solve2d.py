"""Winding field w (1 per sheet; H and V layers each 1/2) on one slice.

w = (u + sigma*theta)/2pi with u single valued, so w lifts like a spiral.
1) instantaneous frequency of the Gabor analytic signal, smoothed, projected on
   the outward normal  -> vortex-free gradient target
2) weighted LS integration (CG) -> init
3) gradient descent on  data(phase) + freq prior + smoothness
"""
import sys
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

dev = 'cuda'
d = np.load(sys.argv[1])
out = sys.argv[2]
re, im, mag, mask = d['re'], d['im'], d['mag'], d['mask']
phi, s, theta_n = d['phi'], d['s'], d['theta_n']
cx, cy = float(d['cx']), float(d['cy'])
Hh, Ww = re.shape
yy, xx = np.mgrid[0:Hh, 0:Ww].astype(np.float32)
theta = np.arctan2(yy - cy, xx - cx)
nx_, ny_ = np.cos(theta_n), np.sin(theta_n)

def wrapn(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

# --- 1) instantaneous frequency (rad/px), smoothed with |z|^2 weights ---
gre_y, gre_x = np.gradient(re); gim_y, gim_x = np.gradient(im)
kx_num = re * gim_x - im * gre_x
ky_num = re * gim_y - im * gre_y
den = re * re + im * im
SM = 6.0
den_s = ndi.gaussian_filter(den * mask, SM) + 1e-9
kx = ndi.gaussian_filter(kx_num * mask, SM) / den_s
ky = ndi.gaussian_filter(ky_num * mask, SM) / den_s
kn = kx * nx_ + ky * ny_
print('k.n pctl (px per winding):', 2 * np.pi / np.percentile(kn[mask], [90, 50, 10]))
kmin, kmax = 2 * np.pi / 30, 2 * np.pi / 5
kn_c = np.clip(kn, kmin, kmax)
fx, fy = kn_c * nx_, kn_c * ny_                    # target grad of 2pi*w

# handedness from circulation of the target field around the center
def circ(fxx, fyy, r):
    t = np.linspace(0, 2 * np.pi, 6000, endpoint=False)
    px = cx + r * np.cos(t); py = cy + r * np.sin(t)
    ix = np.rint(px).astype(int); iy = np.rint(py).astype(int)
    dlx = -r * np.sin(t) * (2 * np.pi / len(t)); dly = r * np.cos(t) * (2 * np.pi / len(t))
    return np.sum(fxx[iy, ix] * dlx + fyy[iy, ix] * dly) / (2 * np.pi)
for r in [150, 300, 500, 700]:
    print('circulation r', r, circ(fx, fy, r))
sigma = float(sys.argv[3]) if len(sys.argv) > 3 else float(np.sign(np.median([circ(fx, fy, r) for r in (200, 350, 500)])))
print('sigma', sigma)

# target for u = 2pi w - sigma*theta
dthx = -(yy - cy) / ((xx - cx) ** 2 + (yy - cy) ** 2 + 1e-6)
dthy = (xx - cx) / ((xx - cx) ** 2 + (yy - cy) ** 2 + 1e-6)
T = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(dev)
tx_t, ty_t = T(fx - sigma * dthx), T(fy - sigma * dthy)
mag_n = np.clip(mag / np.percentile(mag[mask], 90), 0, 1)
conf = T(np.where(mask, mag_n, 0))
wgrid = T(np.where(mask, 0.05 + mag_n, 0) + 1e-4)

# --- 2) weighted LS integration ---
wx = torch.minimum(wgrid[:, 1:], wgrid[:, :-1]); wy = torch.minimum(wgrid[1:, :], wgrid[:-1, :])
gx = 0.5 * (tx_t[:, 1:] + tx_t[:, :-1]); gy = 0.5 * (ty_t[1:, :] + ty_t[:-1, :])
# wrap edge targets that straddle the theta branch cut
th_t = T(theta)
cut_x = (th_t[:, 1:] - th_t[:, :-1]).abs() > np.pi
cut_y = (th_t[1:, :] - th_t[:-1, :]).abs() > np.pi
gx = gx - sigma * torch.where(cut_x, torch.sign(th_t[:, 1:] - th_t[:, :-1]) * 2 * np.pi, torch.zeros_like(gx)) * 0
gy = gy - sigma * torch.where(cut_y, torch.sign(th_t[1:, :] - th_t[:-1, :]) * 2 * np.pi, torch.zeros_like(gy)) * 0
# (u is smooth across the cut; dtheta is analytic, so no correction is needed)

def A(u):
    r = torch.zeros_like(u)
    a = wx * (u[:, 1:] - u[:, :-1]); b_ = wy * (u[1:, :] - u[:-1, :])
    r[:, 1:] += a; r[:, :-1] -= a; r[1:, :] += b_; r[:-1, :] -= b_
    return r + 1e-7 * u
b = torch.zeros_like(wgrid)
a = wx * gx; b_ = wy * gy
b[:, 1:] += a; b[:, :-1] -= a; b[1:, :] += b_; b[:-1, :] -= b_

def cg(u, iters):
    r = b - A(u); p = r.clone(); rs = (r * r).sum()
    for _ in range(iters):
        Ap = A(p); al = rs / (p * Ap).sum()
        u = u + al * p; r = r - al * Ap; rn = (r * r).sum()
        p = r + (rn / rs) * p; rs = rn
    return u, rs.sqrt().item()

# coarse-to-fine: solve on 4x-pooled grid first for a good start
def pool(t, k):
    return F.avg_pool2d(t[None, None], k)[0, 0]
u = torch.zeros_like(wgrid)
with torch.no_grad():
    for it in range(8):
        u, res = cg(u, 500)
    print('LS residual', res)
u0 = u.clone()
w0 = (u0.cpu().numpy() + sigma * theta) / (2 * np.pi)
mk = mask & (np.abs(s) > 0.02)
print('init: corr(s, cos 2pi w)', np.corrcoef(s[mk], np.cos(2 * np.pi * w0[mk]))[0, 1])

# --- 3) refine: data + frequency prior + smoothness ---
psi_t = T(wrapn(phi - sigma * theta))           # data phase in the u frame
u = u0.clone().requires_grad_(True)
opt = torch.optim.Adam([u], lr=0.05)
lam_f, lam_s = 0.3, 2.0
mx = (wx > 1e-3).float(); my = (wy > 1e-3).float()
for it in range(3000):
    opt.zero_grad()
    data = (conf * (1 - torch.cos(u - psi_t))).sum()
    dx = u[:, 1:] - u[:, :-1]; dy = u[1:, :] - u[:-1, :]
    prior = (wx * (dx - gx) ** 2).sum() + (wy * (dy - gy) ** 2).sum()
    lxx = u[:, 2:] - 2 * u[:, 1:-1] + u[:, :-2]; lyy = u[2:, :] - 2 * u[1:-1, :] + u[:-2, :]
    smooth = (mx[:, 1:] * mx[:, :-1] * lxx ** 2).sum() + (my[1:, :] * my[:-1, :] * lyy ** 2).sum()
    loss = data + lam_f * prior + lam_s * smooth
    loss.backward(); opt.step()
    if it % 500 == 0:
        print(it, 'data', data.item(), 'prior', prior.item(), 'smooth', smooth.item())
u = u.detach().cpu().numpy()
w = (u + sigma * theta) / (2 * np.pi)
print('refined: corr(s, cos 2pi w)', np.corrcoef(s[mk], np.cos(2 * np.pi * w[mk]))[0, 1])
gyw, gxw = np.gradient(w)
gn = gxw * nx_ + gyw * ny_
print('frac of mask where w decreases outward', (gn[mask] < 0).mean())
print('w range in mask', np.percentile(w[mask], [1, 50, 99]))
np.savez_compressed(out, w=w.astype(np.float32), w0=w0.astype(np.float32), mask=mask, sigma=sigma, cx=cx, cy=cy)

ys, xs = np.nonzero(mask)
y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
for name, ww in [('init', w0), ('refined', w)]:
    fig, axs = plt.subplots(1, 2, figsize=(26, 12))
    axs[0].imshow(np.where(mask, np.mod(ww, 1), np.nan)[y0:y1, x0:x1], cmap='twilight', interpolation='nearest')
    axs[0].set_title(f'{name}: frac(w)  (0 = H centre, 0.5 = V centre)')
    im_ = axs[1].imshow(np.where(mask, ww, np.nan)[y0:y1, x0:x1], cmap='turbo')
    axs[1].contour(np.where(mask, ww, np.nan)[y0:y1, x0:x1], levels=np.arange(-100, 200, 5), colors='k', linewidths=0.5)
    plt.colorbar(im_, ax=axs[1]); axs[1].set_title('w (contours every 5)')
    plt.tight_layout(); plt.savefig(out.replace('.npz', f'_{name}.png'), dpi=60); plt.close()
