"""2D prototype: winding field from alternating H/V fiber layers on one z slice."""
import sys
import numpy as np
import torch
from scipy import ndimage as ndi

dev = 'cuda'
d = np.load(sys.argv[1])
p = d['p'].astype(np.float32) / 255
nz, ax, ay = d['nz'], d['ax'], d['ay']
H = p * (1 - nz ** 2)
V = p * nz ** 2
s = ndi.gaussian_filter(H - V, 0.7)
mask = ndi.binary_closing(ndi.gaussian_filter(p, 4) > 0.05, iterations=4)
mask = ndi.binary_fill_holes(mask)

# --- sheet normal from H fibers' in-plane direction (orientation tensor) ---
sig_t = 6.0
txx = ndi.gaussian_filter(H * ax * ax, sig_t)
tyy = ndi.gaussian_filter(H * ay * ay, sig_t)
txy = ndi.gaussian_filter(H * ax * ay, sig_t)
ang_t = 0.5 * np.arctan2(2 * txy, txx - tyy)        # tangent angle
coh = np.hypot(txx - tyy, 2 * txy) / (txx + tyy + 1e-6)
tx, ty = np.cos(ang_t), np.sin(ang_t)
nxv, nyv = -ty, tx                                   # unoriented normal

# --- center: point closest to all normal lines (weighted LS) ---
ww = (coh * (txx + tyy) * mask).ravel()
yy, xx = np.mgrid[0:p.shape[0], 0:p.shape[1]].astype(np.float64)
# line through (x,y) with direction n: tangent t is its normal => t.(c - x) = 0
A = np.stack([tx.ravel(), ty.ravel()], 1)
b = tx.ravel() * xx.ravel() + ty.ravel() * yy.ravel()
sel = ww > np.percentile(ww[ww > 0], 50)
Aw = A[sel] * ww[sel, None]
c = np.linalg.solve(Aw.T @ A[sel], Aw.T @ b[sel])
cx, cy = c
if len(sys.argv) > 3:
    cx, cy = float(sys.argv[3]), float(sys.argv[4])
print('center (x,y)', cx, cy)
# orient normals outward
rx, ry = xx - cx, yy - cy
sgn = np.sign(nxv * rx + nyv * ry); sgn[sgn == 0] = 1
nxv, nyv = nxv * sgn, nyv * sgn
theta_n = np.arctan2(nyv, nxv)

# --- steered Gabor quadrature along the outward normal ---
S = torch.from_numpy(s).to(dev)
Sf = torch.fft.fft2(S)
Hh, Ww = s.shape
fy = torch.fft.fftfreq(Hh, device=dev)[:, None]
fx = torch.fft.fftfreq(Ww, device=dev)[None, :]
n_or = 16
best_mag = torch.zeros_like(S)
best_re = torch.zeros_like(S); best_im = torch.zeros_like(S)
th_n = torch.from_numpy(theta_n.astype(np.float32)).to(dev)
pos = (th_n % (2 * np.pi)) / (2 * np.pi) * n_or
k0 = torch.floor(pos).long() % n_or
f = pos - torch.floor(pos)
k1 = (k0 + 1) % n_or
for lam in [8.0, 10.0, 12.5, 15.0, 19.0]:
    sn, st = 0.45 * lam, 1.2 * lam                   # gaussian sigma across/along sheet
    re = torch.zeros_like(S); im = torch.zeros_like(S)
    for k in range(n_or):
        th = 2 * np.pi * k / n_or
        u = fx * np.cos(th) + fy * np.sin(th)
        v = -fx * np.sin(th) + fy * np.cos(th)
        # one-sided gaussian at +1/lam along direction th (analytic signal)
        G = torch.exp(-2 * np.pi ** 2 * ((u - 1 / lam) ** 2 * sn ** 2 + v ** 2 * st ** 2))
        r = torch.view_as_real(torch.fft.ifft2(Sf * G))
        wk = (1 - f) * (k0 == k) + f * (k1 == k)
        re += wk * r[..., 0]; im += wk * r[..., 1]
    mag = torch.sqrt(re * re + im * im) * np.sqrt(sn * st)
    upd = mag > best_mag
    best_mag = torch.where(upd, mag, best_mag)
    best_re = torch.where(upd, re, best_re); best_im = torch.where(upd, im, best_im)

phi = torch.atan2(best_im, best_re).cpu().numpy()  # s ~ cos(2pi n.x/lam + phi): phase 0 at H peak
# phase increases outward along n: local signal at x ~ cos(phase(x)), phase(x)=k.x+phi?
# For an analytic filter centred at +k, response(x) ~ A e^{i(k.x + c)} => arg increases along +n.
mag = best_mag.cpu().numpy()
np.savez_compressed(sys.argv[2], phi=phi, mag=mag, re=best_re.cpu().numpy(), im=best_im.cpu().numpy(), mask=mask, cx=cx, cy=cy,
                    theta_n=theta_n, coh=coh, s=s)
print('done; mag pctl in mask', np.percentile(mag[mask], [10, 50, 90]))
