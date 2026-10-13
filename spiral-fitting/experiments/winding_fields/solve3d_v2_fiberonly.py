"""3D (sampled-slab) winding field from alternating H/V fiber layers.

w(z,y,x) = (u + sigma*theta)/2pi, 1 per sheet, H centre at frac 0, V at 0.5.
Slices z0..z1 step dz of a fiber-zarr level; in-plane analytic signal per slice,
one 3D field solved jointly with z coupling.
"""
import argparse, time
import numpy as np
import torch
import torch.nn.functional as F
import zarr

ap = argparse.ArgumentParser()
ap.add_argument('--root', default='/ephemeral/sean/experiments/winding_fields/data/fiber_zarrs/PHercParis4-20260411134726-las-sd1-d74cca79_')
ap.add_argument('--level', default='4')
ap.add_argument('--z0', type=int, default=2300)
ap.add_argument('--z1', type=int, default=2400)
ap.add_argument('--dz', type=int, default=2)
ap.add_argument('--sigma', type=float, default=0, help='handedness; 0 = try both')
ap.add_argument('--iters', type=int, default=2000)
ap.add_argument('--out', required=True)
ap.add_argument('--layer_iters', type=int, default=2000, help='spacing-free layer refinement; 0 = off')
ap.add_argument('--lam_t', type=float, default=20.0)
ap.add_argument('--lam_z', type=float, default=10.0)
ap.add_argument('--lam_mono', type=float, default=200.0)
ap.add_argument('--lam_l1', type=float, default=0.05)
ap.add_argument('--orient', choices=['iso', 'aniso'], default='aniso')
ap.add_argument('--orient_reach', type=int, default=12)
ap.add_argument('--normals', choices=['fiber', 'las'], default='fiber')
ap.add_argument('--las_root', default='/ephemeral/sean/experiments/winding_fields/data/lasagna/las_008_')
ap.add_argument('--center', type=float, nargs=2, default=None, help='x y in level px')
args = ap.parse_args()
dev = 'cuda'
torch.manual_seed(0)
t0 = time.time()

# ---------------- load ----------------
zs = list(range(args.z0, args.z1, args.dz))
arr = {k: zarr.open(args.root + f'{k}.ome.zarr/{args.level}', mode='r') for k in ('nx', 'ny', 'presence')}
raw = {k: np.asarray(a[args.z0:args.z1:args.dz]) for k, a in arr.items()}
pr = torch.from_numpy(raw['presence']).to(dev).float() / 255
ax_ = (torch.from_numpy(raw['nx']).to(dev).float() - 128) / 127
ay_ = (torch.from_numpy(raw['ny']).to(dev).float() - 128) / 127
nz2 = (1 - ax_ * ax_ - ay_ * ay_).clamp(0, 1)
# crop xy to the data bbox (+margin)
pr_full_shape = pr.shape[1:]
anyp = (pr > 0).any(0)
ys, xs = torch.nonzero(anyp, as_tuple=True)
M = 48
y0, y1 = max(int(ys.min()) - M, 0), min(int(ys.max()) + M, pr.shape[1])
x0, x1 = max(int(xs.min()) - M, 0), min(int(xs.max()) + M, pr.shape[2])
crop = lambda t: t[:, y0:y1, x0:x1].contiguous()
pr, ax_, ay_, nz2 = map(crop, (pr, ax_, ay_, nz2))
D, Hh, Ww = pr.shape
print(f'slab {D}x{Hh}x{Ww} (z {zs[0]}..{zs[-1]} step {args.dz}), crop y{y0}:{y1} x{x0}:{x1}, load {time.time()-t0:.1f}s')

def gblur(t, sig_xy, sig_z=0.0):
    """separable gaussian on (D,H,W); sig_z in slab-slice units"""
    def k1(sig):
        r = int(3 * sig + 0.5)
        x = torch.arange(-r, r + 1, device=dev, dtype=torch.float32)
        k = torch.exp(-0.5 * (x / sig) ** 2); return k / k.sum(), r
    out = t[:, None]
    if sig_xy > 0:
        k, r = k1(sig_xy)
        out = F.conv2d(F.pad(out, (r, r, 0, 0), mode='replicate'), k.view(1, 1, 1, -1))
        out = F.conv2d(F.pad(out, (0, 0, r, r), mode='replicate'), k.view(1, 1, -1, 1))
    out = out[:, 0]
    if sig_z > 0 and out.shape[0] > 1:
        k, r = k1(sig_z)
        o = out.permute(1, 2, 0).reshape(-1, 1, out.shape[0])
        o = F.conv1d(F.pad(o, (r, r), mode='replicate'), k.view(1, 1, -1))
        out = o.reshape(out.shape[1], out.shape[2], -1).permute(2, 0, 1)
    return out.contiguous()

H = pr * (1 - nz2); V = pr * nz2
s = gblur(H - V, 0.5)
mask = gblur(pr, 3, 1) > 0.05
# fill holes cheaply: everything inside the outer boundary (per slice, via scipy)
from scipy import ndimage as ndi
mask = torch.from_numpy(np.stack([ndi.binary_fill_holes(ndi.binary_closing(m, iterations=3)) for m in mask.cpu().numpy()])).to(dev)

# ---------------- normals ----------------
sig_t = 4.0
if args.normals == 'las':
    la = {k: zarr.open(args.las_root + f'{k}.ome.zarr/4', mode='r') for k in ('nx', 'ny')}
    fct = 2 ** (4 - int(args.level))
    zl = [z // fct for z in zs]
    def load_las(a):
        t = torch.from_numpy(np.asarray(a.get_orthogonal_selection((zl, slice(None), slice(None))))).to(dev).float()
        if fct > 1:
            t = t.repeat_interleave(fct, 1).repeat_interleave(fct, 2)
        t = t[:, :pr_full_shape[0], :pr_full_shape[1]]
        return (crop(t) - 128) / 127
    lnx = load_las(la['nx']); lny = load_las(la['ny'])
    lw = ((lnx * lnx + lny * lny) > 0.25).float()           # normal mostly in-plane & present
    txx = gblur(lw * lny * lny, 2.0, 1.0); tyy = gblur(lw * lnx * lnx, 2.0, 1.0); txy = gblur(-lw * lnx * lny, 2.0, 1.0)
    del lnx, lny
else:
    txx = gblur(H * ax_ * ax_, sig_t, 1.5); tyy = gblur(H * ay_ * ay_, sig_t, 1.5); txy = gblur(H * ax_ * ay_, sig_t, 1.5)
ang = 0.5 * torch.atan2(2 * txy, txx - tyy)                 # sheet tangent angle in-plane
coh = torch.sqrt((txx - tyy) ** 2 + 4 * txy ** 2) / (txx + tyy + 1e-6)
nxv, nyv = -torch.sin(ang), torch.cos(ang)
yy, xx = torch.meshgrid(torch.arange(Hh, device=dev, dtype=torch.float32), torch.arange(Ww, device=dev, dtype=torch.float32), indexing='ij')
if args.center:
    cx, cy = args.center[0] - x0, args.center[1] - y0
else:
    wv = (coh * (txx + tyy) * mask).mean(0)
    tx, ty = torch.cos(ang).mean(0), torch.sin(ang).mean(0)
    sel = wv > wv[wv > 0].median()
    Aa = torch.stack([tx[sel], ty[sel]], 1).double(); bb = (tx * xx + ty * yy)[sel].double(); ww_ = wv[sel].double()
    c = torch.linalg.solve(Aa.T @ (Aa * ww_[:, None]), Aa.T @ (bb * ww_))
    cx, cy = float(c[0]), float(c[1])
print('center (crop px)', cx, cy)
# initial outward orientation, then orientation diffusion (flip to agree with smoothed neighbours)
sg = torch.sign(nxv * (xx - cx) + nyv * (yy - cy)); sg[sg == 0] = 1
nxv, nyv = nxv * sg, nyv * sg
cw = coh * (txx + tyy) * mask
cw = cw / (cw[mask].mean() + 1e-9)
r_c = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
if args.orient == 'iso':
    for it in range(30):
        mx_ = gblur(cw * nxv, 10, 2); my_ = gblur(cw * nyv, 10, 2)
        flip = ((mx_ * nxv + my_ * nyv) < 0) & (r_c[None] > 25)
        nf = int(flip.sum())
        nxv = torch.where(flip, -nxv, nxv); nyv = torch.where(flip, -nyv, nyv)
        if nf == 0:
            break
else:
    # Z2 sign sync coupling only ALONG the sheet: in-plane tangent steps and z steps.
    # Across-sheet neighbours are never coupled, so the two limbs of a hairpin
    # (parallel normal lines, opposite outward sense) do not force each other.
    tx_, ty_ = nyv.clone(), -nxv.clone()       # unsigned tangent, fixed
    base_x = xx[None].expand(D, -1, -1); base_y = yy[None].expand(D, -1, -1)
    offs = [k for k in range(-args.orient_reach, args.orient_reach + 1) if k != 0]
    def idx_iter():
        for k in offs:
            sx = (base_x + k * tx_).round().long().clamp(0, Ww - 1)
            sy = (base_y + k * ty_).round().long().clamp(0, Hh - 1)
            yield (sy * Ww + sx).reshape(D, -1)
    for it in range(400):
        vx = (cw * nxv).reshape(D, -1); vy = (cw * nyv).reshape(D, -1)
        mxs = torch.zeros_like(nxv); mys = torch.zeros_like(nyv)
        for I in idx_iter():
            mxs += torch.gather(vx, 1, I).reshape(D, Hh, Ww)
            mys += torch.gather(vy, 1, I).reshape(D, Hh, Ww)
        for dzs in (1, 2):
            mxs[dzs:] += 2 * cw[:-dzs] * nxv[:-dzs]; mxs[:-dzs] += 2 * cw[dzs:] * nxv[dzs:]
            mys[dzs:] += 2 * cw[:-dzs] * nyv[:-dzs]; mys[:-dzs] += 2 * cw[dzs:] * nyv[dzs:]
        dot = mxs * nxv + mys * nyv
        flip = (dot < -0.1 * torch.sqrt(mxs ** 2 + mys ** 2)) & (r_c[None] > 25) & mask
        # damp synchronous flipping: random half of the candidates each sweep
        flip &= torch.rand_like(dot) < 0.5
        nf = int(flip.sum())
        nxv = torch.where(flip, -nxv, nxv); nyv = torch.where(flip, -nyv, nyv)
        if it % 50 == 0:
            print(f'  orient sweep {it}: flips {nf}')
        if nf == 0:
            break
    # report how far the result departs from the radial guess
    rad = (nxv * (xx - cx) + nyv * (yy - cy)) < 0
    print('  frac normals pointing toward umbilicus after sync:', float(rad[mask].float().mean()))
print(f'orientation ({args.orient}) stopped after {it+1} iters, last flips {nf}')
theta_n = torch.atan2(nyv, nxv)

# ---------------- steered analytic signal per slice ----------------
def nice(n):
    # smallest size >= n made of factors 2,3,5 (cuFFT is slow on large primes)
    m = n
    while True:
        k = m
        for f in (2, 3, 5):
            while k % f == 0:
                k //= f
        if k == 1:
            return m
        m += 1
Hp, Wp = nice(Hh + 32), nice(Ww + 32)
fy = torch.fft.fftfreq(Hp, device=dev)[:, None]; fx = torch.fft.fftfreq(Wp, device=dev)[None, :]
n_or = 16
pos = (theta_n % (2 * np.pi)) / (2 * np.pi) * n_or
k0 = torch.floor(pos).long() % n_or; fr = pos - torch.floor(pos); k1 = (k0 + 1) % n_or
best_mag = torch.zeros_like(s); RE = torch.zeros_like(s); IM = torch.zeros_like(s)
lams = [float(x) / (2 ** (int(args.level) - 3)) for x in (5.0, 6.5, 8.0, 10.0, 13.0, 16.0)]
lams = [l for l in lams if l >= 3.0]
print('wavelengths', lams)
Sf = torch.fft.fft2(F.pad(s, (0, Wp - Ww, 0, Hp - Hh)))
for lam in lams:
    sn, st = 0.45 * lam, 1.2 * lam
    re = torch.zeros_like(s); im = torch.zeros_like(s)
    for k in range(n_or):
        th = 2 * np.pi * k / n_or
        u_ = fx * np.cos(th) + fy * np.sin(th); v_ = -fx * np.sin(th) + fy * np.cos(th)
        G = torch.exp(-2 * np.pi ** 2 * ((u_ - 1 / lam) ** 2 * sn ** 2 + v_ ** 2 * st ** 2))
        r = torch.view_as_real(torch.fft.ifft2(Sf * G))[:, :Hh, :Ww]
        wk = (1 - fr) * (k0 == k) + fr * (k1 == k)
        re += wk * r[..., 0]; im += wk * r[..., 1]
    mag = torch.sqrt(re * re + im * im) * np.sqrt(sn * st)
    upd = mag > best_mag
    best_mag = torch.where(upd, mag, best_mag); RE = torch.where(upd, re, RE); IM = torch.where(upd, im, IM)
del Sf
phi = torch.atan2(IM, RE)
print(f'analytic signal done {time.time()-t0:.1f}s')

# ---------------- instantaneous frequency -> gradient target ----------------
def grad_xy(t):
    gx_ = torch.zeros_like(t); gy_ = torch.zeros_like(t)
    gx_[:, :, 1:-1] = 0.5 * (t[:, :, 2:] - t[:, :, :-2]); gy_[:, 1:-1] = 0.5 * (t[:, 2:] - t[:, :-2])
    return gx_, gy_
grx, gry = grad_xy(RE); gix, giy = grad_xy(IM)
mf = mask.float()
den = gblur((RE * RE + IM * IM) * mf, 4, 1.5) + 1e-9
kx = gblur((RE * gix - IM * grx) * mf, 4, 1.5) / den
ky = gblur((RE * giy - IM * gry) * mf, 4, 1.5) / den
kn = kx * nxv + ky * nyv
q = torch.quantile(kn[mask][::97], torch.tensor([0.1, 0.5, 0.9], device=dev))
print('px per winding (p90,p50,p10 of k):', (2 * np.pi / q).tolist(), ' frac k.n<0:', float((kn[mask] < 0).float().mean()))
kn_c = kn.clamp(2 * np.pi / (60 / 2 ** (int(args.level) - 3)), 2 * np.pi / 3.0)
del grx, gry, gix, giy, kx, ky, den, txx, tyy, txy, coh, cw, H, V, pr, ax_, ay_, nz2
torch.cuda.empty_cache()

mag_n = (best_mag / torch.quantile(best_mag[mask][::97], 0.9)).clamp(0, 1)
conf = torch.where(mask, mag_n, torch.zeros_like(mag_n))
wgrid = torch.where(mask, 0.05 + mag_n, torch.zeros_like(mag_n)) + 1e-4
del RE, IM, best_mag, mag_n, kn
torch.cuda.empty_cache()
theta = torch.atan2(yy - cy, xx - cx)[None].expand(D, -1, -1)
rr2 = ((xx - cx) ** 2 + (yy - cy) ** 2 + 1.0)[None]
dthx = -(yy - cy)[None] / rr2; dthy = (xx - cx)[None] / rr2
wx = torch.minimum(wgrid[:, :, 1:], wgrid[:, :, :-1]); wy = torch.minimum(wgrid[:, 1:], wgrid[:, :-1])
wz = torch.minimum(wgrid[1:], wgrid[:-1]) * 1.0
mx = (wx > 1e-3).float(); my = (wy > 1e-3).float(); mz = (wz > 1e-3).float()

def solve(sigma):
    gx = 0.5 * ((kn_c * nxv - sigma * dthx)[:, :, 1:] + (kn_c * nxv - sigma * dthx)[:, :, :-1])
    gy = 0.5 * ((kn_c * nyv - sigma * dthy)[:, 1:] + (kn_c * nyv - sigma * dthy)[:, :-1])
    def A(u):
        r = torch.zeros_like(u)
        a = wx * (u[:, :, 1:] - u[:, :, :-1]); b_ = wy * (u[:, 1:] - u[:, :-1]); c_ = wz * (u[1:] - u[:-1])
        r[:, :, 1:] += a; r[:, :, :-1] -= a; r[:, 1:] += b_; r[:, :-1] -= b_; r[1:] += c_; r[:-1] -= c_
        return r + 1e-7 * u
    b = torch.zeros_like(wgrid)
    a = wx * gx; b_ = wy * gy
    b[:, :, 1:] += a; b[:, :, :-1] -= a; b[:, 1:] += b_; b[:, :-1] -= b_
    u = torch.zeros_like(wgrid)
    with torch.no_grad():
        r = b - A(u); p = r.clone(); rs = (r * r).sum()
        for _ in range(3000):
            Ap = A(p); al = rs / (p * Ap).sum()
            u = u + al * p; r = r - al * Ap; rn = (r * r).sum(); p = r + (rn / rs) * p; rs = rn
    psi = torch.remainder(phi - sigma * theta + np.pi, 2 * np.pi) - np.pi
    u = u.requires_grad_(True)
    opt = torch.optim.Adam([u], lr=0.05)
    for it in range(args.iters):
        opt.zero_grad()
        data = (conf * (1 - torch.cos(u - psi))).sum()
        dx = u[:, :, 1:] - u[:, :, :-1]; dy = u[:, 1:] - u[:, :-1]; dzz = u[1:] - u[:-1]
        prior = (wx * (dx - gx) ** 2).sum() + (wy * (dy - gy) ** 2).sum() + (wz * dzz ** 2).sum()
        lxx = u[:, :, 2:] - 2 * u[:, :, 1:-1] + u[:, :, :-2]; lyy = u[:, 2:] - 2 * u[:, 1:-1] + u[:, :-2]
        smooth = (mx[:, :, 1:] * mx[:, :, :-1] * lxx ** 2).sum() + (my[:, 1:] * my[:, :-1] * lyy ** 2).sum()
        loss = data + 0.3 * prior + 2.0 * smooth
        loss.backward(); opt.step()
    print(f'sigma {sigma:+.0f}: data {data.item():.0f} prior {prior.item():.0f} smooth {smooth.item():.0f}  ({time.time()-t0:.0f}s)')
    return u.detach(), data.item()

sigmas = [args.sigma] if args.sigma else [1.0, -1.0]
res = {sg_: solve(sg_) for sg_ in sigmas}
sigma = min(res, key=lambda k: res[k][1])
u = res[sigma][0]
del res

def report(tag, u_):
    w_ = (u_ + sigma * theta) / (2 * np.pi)
    mk_ = mask & (s.abs() > 0.02)
    sv_, cv_ = s[mk_], torch.cos(2 * np.pi * w_[mk_])
    c_ = float(((sv_ - sv_.mean()) * (cv_ - cv_.mean())).mean() / (sv_.std() * cv_.std()))
    print(f'[{tag}] corr(s, cos 2pi w) {c_:.3f}')

if args.layer_iters > 0:
    report('phase init', u)
    # Spacing-free energy on the raw H/V signal. No frequency or pitch prior:
    #   data   : H voxels -> frac(w)=0, V voxels -> frac(w)=0.5
    #   along  : w constant along the sheet (in-plane tangent) and along z
    #   across : only monotone (dw/dn >= 0) + tiny L1 on dw/dn (fewest windings)
    sn_ = (s / torch.quantile(s[mask].abs()[::97], 0.99)).clamp(-1, 1) * mask
    txv, tyv = nyv, -nxv
    mz2 = (mask[1:] & mask[:-1]).float()
    mf_ = mask.float()
    def cdiff(t, dim):
        g = torch.zeros_like(t)
        if dim == 2:
            g[:, :, 1:-1] = 0.5 * (t[:, :, 2:] - t[:, :, :-2])
        else:
            g[:, 1:-1] = 0.5 * (t[:, 2:] - t[:, :-2])
        return g
    u = u.clone().requires_grad_(True)
    opt = torch.optim.Adam([u], lr=0.02)
    for it in range(args.layer_iters):
        opt.zero_grad()
        ux = (cdiff(u, 2) + sigma * dthx) / (2 * np.pi)       # dw/dx incl. theta lift
        uy = (cdiff(u, 1) + sigma * dthy) / (2 * np.pi)
        dn = nxv * ux + nyv * uy
        dt = txv * ux + tyv * uy
        dzw = (u[1:] - u[:-1]) / (2 * np.pi)
        data = -(sn_ * torch.cos(u + sigma * theta)).sum()
        along = args.lam_t * (mf_ * dt ** 2).sum() + args.lam_z * (mz2 * dzw ** 2).sum()
        mono = args.lam_mono * (mf_ * torch.relu(-dn) ** 2).sum()
        l1 = args.lam_l1 * (mf_ * dn.abs()).sum()
        loss = data + along + mono + l1
        loss.backward(); opt.step()
        if it % 500 == 0 or it == args.layer_iters - 1:
            print(f'  layer it {it}: data {data.item():.0f} along {along.item():.0f} mono {mono.item():.0f} l1 {l1.item():.0f}')
    u = u.detach()
    report('layer refined', u)
w = (u + sigma * theta) / (2 * np.pi)
mk = mask & (s.abs() > 0.02)
sv, cv = s[mk], torch.cos(2 * np.pi * w[mk])
corr = float(((sv - sv.mean()) * (cv - cv.mean())).mean() / (sv.std() * cv.std()))
gwx, gwy = grad_xy(w)
dec = float(((gwx * nxv + gwy * nyv)[mask] < 0).float().mean())
dzw = (w[1:] - w[:-1]).abs()[mask[1:] & mask[:-1]]
print(f'chosen sigma {sigma:+.0f}; corr(s, cos 2pi w) {corr:.3f}; frac w decreasing along n {dec:.3f}; |dw/dslice| p50/p99 {float(dzw.median()):.3f}/{float(torch.quantile(dzw[::31], 0.99)):.3f}')
wq = w[mask][::97]
print('w range p1/p50/p99', torch.quantile(wq, torch.tensor([0.01, 0.5, 0.99], device=dev)).tolist())
np.savez_compressed(args.out, w=w.cpu().numpy().astype(np.float32), s=s.cpu().numpy().astype(np.float16), mask=mask.cpu().numpy(),
                    conf=conf.cpu().numpy().astype(np.float16), zs=np.array(zs), crop=np.array([y0, y1, x0, x1]), sigma=sigma, cx=cx, cy=cy,
                    level=int(args.level))
print(f'total {time.time()-t0:.0f}s')
