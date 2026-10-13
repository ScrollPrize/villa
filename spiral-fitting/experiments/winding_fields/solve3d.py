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
ap.add_argument('--level', default='3')
ap.add_argument('--z0', type=int, default=2300)
ap.add_argument('--z1', type=int, default=2400)
ap.add_argument('--dz', type=int, default=2)
ap.add_argument('--sigma', type=float, default=0, help='handedness; 0 = try both')
ap.add_argument('--iters', type=int, default=200, help='phase refine Adam iterations')
ap.add_argument('--out', required=True)
ap.add_argument('--init', choices=['gabor', 'bandcount', 'none'], default='gabor',
                help='bandcount: spacing-free init, +1 per surf band entered along the outward normal')
ap.add_argument('--cover_sigma', type=float, default=6.0)
ap.add_argument('--cover_ref', type=float, default=0.25, help='band area fraction treated as full coverage')
ap.add_argument('--cover_floor', type=float, default=0.01)
ap.add_argument('--count_sigma', type=float, default=4.0)
ap.add_argument('--bandcount_mode', choices=['entry', 'abs'], default='abs')
ap.add_argument('--signal', choices=['fiber', 'surf', 'both', 'frontback'], default='fiber',
                help='fiber: H->frac 0, V->frac 0.5; surf: sheet band->frac 0, gap->0.5; both: surf + fiber presence on bands')
ap.add_argument('--fb_fiber', type=float, default=1.0, help='frontback: weight of H(+)/V(-) fiber term')
ap.add_argument('--fb_gap', type=float, default=0.5, help='frontback: weight of the off-band pull to frac 0.5')
ap.add_argument('--surf_root', default='/ephemeral/sean/experiments/winding_fields/data/2um_ds2_ps256_surf.zarr')
ap.add_argument('--layer_iters', type=int, default=400, help='spacing-free layer refinement; 0 = off')
ap.add_argument('--lam_t', type=float, default=20.0)
ap.add_argument('--lam_z', type=float, default=10.0)
ap.add_argument('--lam_mono', type=float, default=200.0)
ap.add_argument('--lam_l1', type=float, default=0.05)
ap.add_argument('--orient', choices=['iso', 'aniso', 'none'], default='aniso')
ap.add_argument('--orient_reach', type=int, default=20)
ap.add_argument('--orient_step', type=int, default=2, help='tangent sample spacing in px for orientation sync')
ap.add_argument('--orient_sweeps', type=int, default=60)
ap.add_argument('--no_compile', action='store_true')
ap.add_argument('--no_cache', action='store_true', help='do not write slab input caches (full-scroll runs)')
ap.add_argument('--sheetgraph', action='store_true', help='discrete sheet-graph solve on surf centreline pieces (after the init)')
ap.add_argument('--sg_cell', type=int, default=16)
ap.add_argument('--sg_w_along', type=float, default=1.0)
ap.add_argument('--sg_w_z', type=float, default=1.0)
ap.add_argument('--sg_w_across', type=float, default=1.0)
ap.add_argument('--sg_w_wi', type=float, default=1.0)
ap.add_argument('--sg_across_fallback', type=float, default=0.0,
                help='>0: keep surf across edges only farther than this many px from any snapped winding_inference crossing')
ap.add_argument('--sg_drop_back', type=float, default=0.0,
                help='>0: drop surf band pixels on V-dominated (back-face) material before building centrelines; value = V/H ratio')
ap.add_argument('--sg_min_comp', type=int, default=2000, help='graph components smaller than this (pieces) are not used as fill targets')
ap.add_argument('--sg_across_ineq', action='store_true', help='across edges as one-sided constraints: step >= 1')
ap.add_argument('--sg_across_min_t', type=int, default=2)
ap.add_argument('--sg_ineq_up', type=float, default=0.05, help='relative slope of the across cost above the step>=1 bound')
ap.add_argument('--sg_across_min_par', type=float, default=0.0, help='require normal dot product >= this between the two bands')
ap.add_argument('--sg_round_local', type=float, default=0.0,
                help='>0: remove the slow fractional drift of the L1 solution: local circular mean of frac(x) over this xy sigma (px), then round')
ap.add_argument('--sg_round_int', action='store_true', help='with --sg_round_local: snap to integers round(x - phi) instead of phi + integer')
ap.add_argument('--sg_fill_back', type=float, default=0.0,
                help='>0: V-fiber voxels between consecutive fronts k, k+1 get fill target k+0.5 with this weight')
ap.add_argument('--sg_integerize', action='store_true', help='exact integer piece labels via a spanning forest of satisfied edges')
ap.add_argument('--sg_w_seed', type=float, default=0.0,
                help='>0: winding_inference seed_winding+level as soft absolute anchors (L1 edges to a pinned reference node)')
ap.add_argument('--save_graph', default='', help='write this slab graph (pieces, edges, piece map) to an .npz')
ap.add_argument('--graph_only', action='store_true', help='with --save_graph: stop after saving the graph')
ap.add_argument('--labels_from', default='', help='use piece labels from this .npz (joint solve) instead of solving here')
ap.add_argument('--bc_from', nargs='*', default=[], help='neighbour slab w.npz files: their w on shared slices becomes a fill target (seam coupling)')
ap.add_argument('--save_z0', type=int, default=None, help='save only slices z in [save_z0, save_z1) (z padding = extra context)')
ap.add_argument('--save_z1', type=int, default=None)
ap.add_argument('--bc_w', type=float, default=20.0, help='fill weight of the --bc_from neighbour values')
ap.add_argument('--track_targets', default='', help='fill from solved track labels (track_targets.py dir) instead of a sheet graph')
ap.add_argument('--sg_aggregate', action='store_true', help='one edge per (piece pair, type, offset) with weight type_w*sqrt(count)')
ap.add_argument('--sg_prior', type=float, default=1e-3, help='weight pulling pieces toward the init field (only matters for islands)')
ap.add_argument('--sg_fill_w', type=float, default=10.0, help='data weight of solved piece indices in the voxel fill')
ap.add_argument('--fill_iters', type=int, nargs=4, default=[600, 400, 300, 300], help='fill CG iters at xy pool 8,4,2,1')
ap.add_argument('--wi_root', default='', help='winding_inference dir (input; relative levels along rays)')
ap.add_argument('--no_report', action='store_true', help='skip the automatic eval + renders at the end')
ap.add_argument('--eval_mesh', default='/ephemeral/sean/experiments/winding_fields/data/eval_mesh_DO_NOT_USE_AS_INPUT/spiral-output-generation-19-lasagna.tifxyz',
                help='EVAL ONLY spiral mesh; never used as solver input')
ap.add_argument('--umbilicus', default='/ephemeral/sean/experiments/winding_fields/data/umbilicus.json')
ap.add_argument('--ann', default='/ephemeral/sean/experiments/winding_fields/data/relative_windings.json')
ap.add_argument('--ls_iters', type=int, nargs=4, default=[1000, 600, 300, 200], help='CG iters at xy pooling 8,4,2,1 (coarse levels carry the winding count)')
ap.add_argument('--normals', choices=['fiber', 'las'], default='las')
ap.add_argument('--las_root', default='/ephemeral/sean/experiments/winding_fields/data/lasagna/las_008_')
ap.add_argument('--center', type=float, nargs=2, default=None, help='x y in level px')
args = ap.parse_args()
dev = 'cuda'
torch.manual_seed(0)
t0 = time.time()
_tl = [time.time()]
def stage(name):
    torch.cuda.synchronize()
    now = time.time(); print(f'[time] {name}: {now - _tl[0]:.1f}s (total {now - t0:.1f}s)', flush=True); _tl[0] = now

# ---------------- load ----------------
zs = list(range(args.z0, args.z1, args.dz))
arr = {k: zarr.open(args.root + f'{k}.ome.zarr/{args.level}', mode='r') for k in ('nx', 'ny', 'presence')}
import os
cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cache')
ckey = f'L{args.level}_z{args.z0}-{args.z1}-{args.dz}'
os.makedirs(cache_dir, exist_ok=True)
def cached(name, fn):
    # slab inputs are cached as raw .npy so repeat runs on the same slab skip zarr + CPU prep
    f = os.path.join(cache_dir, f'{ckey}_{name}.npy')
    if os.path.exists(f):
        return np.load(f)
    v = fn()
    if not args.no_cache:
        np.save(f, v)
    return v
raw = {k: cached(k, lambda a=a: np.asarray(a[args.z0:args.z1:args.dz])) for k, a in arr.items()}
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
stage('load fiber')
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

def grad_xy(t):
    gx_ = torch.zeros_like(t); gy_ = torch.zeros_like(t)
    gx_[:, :, 1:-1] = 0.5 * (t[:, :, 2:] - t[:, :, :-2]); gy_[:, 1:-1] = 0.5 * (t[:, 2:] - t[:, :-2])
    return gx_, gy_

def ls_fill(target, dw, wg, iters=(600, 400, 300, 300)):
    """min sum dw*(u - target)^2 + sum wg-edges (grad u)^2, coarse-to-fine CG (xy pool 8,4,2,1)."""
    u = None
    for lvl, f in enumerate((8, 4, 2, 1)):
        if f > 1:
            Hc, Wc = wg.shape[1] // f, wg.shape[2] // f
            pool = lambda t: F.adaptive_avg_pool2d(t[:, None], (Hc, Wc))[:, 0]
            wgc = pool(wg); dwc = pool(dw) * f * f; tc = pool(target * dw) / (pool(dw) + 1e-9)
        else:
            wgc, dwc, tc = wg, dw, target
        wx = torch.minimum(wgc[:, :, 1:], wgc[:, :, :-1]); wy = torch.minimum(wgc[:, 1:], wgc[:, :-1]); wz = torch.minimum(wgc[1:], wgc[:-1])
        def A(v):
            r = dwc * v
            a = wx * (v[:, :, 1:] - v[:, :, :-1]); b_ = wy * (v[:, 1:] - v[:, :-1]); c_ = wz * (v[1:] - v[:-1])
            r[:, :, 1:] += a; r[:, :, :-1] -= a; r[:, 1:] += b_; r[:, :-1] -= b_; r[1:] += c_; r[:-1] -= c_
            return r + 1e-7 * v
        b = dwc * tc
        v = torch.zeros_like(wgc) if u is None else F.interpolate(u[:, None], size=wgc.shape[1:], mode='bilinear', align_corners=False)[:, 0]
        with torch.no_grad():
            r = b - A(v); p = r.clone(); rs = (r * r).sum()
            for _ in range(iters[lvl]):
                Ap = A(p); al = rs / (p * Ap).sum()
                v = v + al * p; r = r - al * Ap; rn = (r * r).sum(); p = r + (rn / rs) * p; rs = rn
        print(f'  fill level 1/{f}: residual {float(rs.sqrt()):.3g}')
        u = v
    return u

def ls_integrate(tx, ty, wg, iters=None):
    iters = iters or tuple(args.ls_iters)
    """Weighted LS for u with grad u ~ (tx, ty) in-plane and u smooth in z, coarse-to-fine
    (xy pooled by 8, 4, 2, 1) so the large-scale part (the winding count) converges."""
    u = None
    for lvl, f in enumerate((8, 4, 2, 1)):
        if f > 1:
            Hc, Wc = wg.shape[1] // f, wg.shape[2] // f
            pool = lambda t: F.adaptive_avg_pool2d(t[:, None], (Hc, Wc))[:, 0]
            wgc = pool(wg); txc = pool(tx * wg) / (wgc + 1e-9) * f; tyc = pool(ty * wg) / (wgc + 1e-9) * f
        else:
            wgc, txc, tyc = wg, tx, ty
        wx = torch.minimum(wgc[:, :, 1:], wgc[:, :, :-1]); wy = torch.minimum(wgc[:, 1:], wgc[:, :-1]); wz = torch.minimum(wgc[1:], wgc[:-1])
        gx = 0.5 * (txc[:, :, 1:] + txc[:, :, :-1]); gy = 0.5 * (tyc[:, 1:] + tyc[:, :-1])
        def A(v):
            r = torch.zeros_like(v)
            a = wx * (v[:, :, 1:] - v[:, :, :-1]); b_ = wy * (v[:, 1:] - v[:, :-1]); c_ = wz * (v[1:] - v[:-1])
            r[:, :, 1:] += a; r[:, :, :-1] -= a; r[:, 1:] += b_; r[:, :-1] -= b_; r[1:] += c_; r[:-1] -= c_
            return r + 1e-7 * v
        b = torch.zeros_like(wgc)
        a = wx * gx; b_ = wy * gy
        b[:, :, 1:] += a; b[:, :, :-1] -= a; b[:, 1:] += b_; b[:, :-1] -= b_
        if u is None:
            v = torch.zeros_like(wgc)
        else:
            v = F.interpolate(u[:, None], size=wgc.shape[1:], mode='bilinear', align_corners=False)[:, 0]
        with torch.no_grad():
            r = b - A(v); p = r.clone(); rs = (r * r).sum()
            for _ in range(iters[lvl]):
                Ap = A(p); al = rs / (p * Ap).sum()
                v = v + al * p; r = r - al * Ap; rn = (r * r).sum(); p = r + (rn / rs) * p; rs = rn
        print(f'  LS level 1/{f}: {tuple(wgc.shape)} residual {float(rs.sqrt()):.3g}')
        u = v
        del wx, wy, wz, gx, gy, b, r, p
    return u

H = pr * (1 - nz2); V = pr * nz2
pr_keep, nz2_keep = pr.half(), nz2.half()      # small copies for the final phase report
s = gblur(H - V, 0.5)
if args.signal != 'fiber':
    # surf pyramid level L-2 is the same grid as fiber level L (surf level 0 = fiber level 2)
    sa = zarr.open(f'{args.surf_root}/{int(args.level) - 2}', mode='r')
    sf = crop(torch.from_numpy(cached('surf', lambda: np.asarray(sa[args.z0:args.z1:args.dz]))).to(dev).float() / 255)
    s = gblur(2 * sf - 1, 0.5)                                # +1 sheet band, -1 gap
    # band indicator for counting: dilate 1 px, light smoothing, renormalise each band's peak to 1
    # (a plain blur shrinks thin 1-3 px bands and loses ~20% of the count)
    bi = F.max_pool2d((sf > 0.5).float()[:, None], 3, 1, 1)[:, 0]
    bi = gblur(bi, 0.5)
    band_init = bi / F.max_pool2d(bi[:, None], 5, 1, 2)[:, 0].clamp(min=0.5)
    del bi
    if args.signal == 'frontback':
        # surf marks the FRONT (H) face only. H -> frac 0, V (back, outward side) -> frac 0.5.
        # Surf band on V-dominated material is a mistake (back face marked): down-weight it.
        Hs, Vs = gblur(H, 1.0), gblur(V, 1.0)
        bad = (Vs > 2 * Hs + 0.05).float()
        band = gblur(sf, 0.5)
        s = band * (1 - 0.8 * bad) - args.fb_gap * (1 - band) + args.fb_fiber * gblur(H - V, 0.5)
        s = s.clamp(-1, 1)
        print('frontback: surf voxels flagged as back-face mistakes:', float((bad * (sf > 0.5)).sum() / (sf > 0.5).sum().clamp(min=1)))
        del Hs, Vs, bad, band
    if args.signal == 'both':
        s = (s + 0.5 * gblur(pr, 0.5)).clamp(-1, 1)          # fibers also mark sheet material
    front_t = sf > 0.5
    if args.sg_drop_back > 0:
        # surf marks the FRONT (H) face; band pixels sitting on back-face (V-dominated) material are mistakes
        Hs_, Vs_ = gblur(H, 1.0), gblur(V, 1.0)
        back = Vs_ > args.sg_drop_back * Hs_ + 0.05
        print(f'drop back-face surf: {float((front_t & back).sum() / front_t.sum().clamp(min=1)):.3f} of band pixels removed')
        front_t = front_t & ~back
        del Hs_, Vs_, back
    front_np = front_t.cpu().numpy()
    del front_t
    del sf
    print('signal:', args.signal)
mask = gblur(pr, 3, 1) > 0.05
# fill holes cheaply: everything inside the outer boundary (per slice, via scipy)
from scipy import ndimage as ndi
_m = mask.cpu().numpy()
mask = torch.from_numpy(cached(f'mask_y{y0}-{y1}_x{x0}-{x1}', lambda: np.stack([ndi.binary_fill_holes(ndi.binary_closing(m, iterations=3)) for m in _m]))).to(dev)
del _m

stage('signal + mask')
# ---------------- normals ----------------
sig_t = 4.0
if args.normals == 'las':
    la = {k: zarr.open(args.las_root + f'{k}.ome.zarr/4', mode='r') for k in ('nx', 'ny')}
    fct = 2 ** (4 - int(args.level))
    zl = [z // fct for z in zs]
    def load_las(a, name):
        t = torch.from_numpy(cached('las_' + name, lambda: np.asarray(a.get_orthogonal_selection((zl, slice(None), slice(None)))))).to(dev).float()
        if fct > 1:
            t = t.repeat_interleave(fct, 1).repeat_interleave(fct, 2)
        t = t[:, :pr_full_shape[0], :pr_full_shape[1]]
        return (crop(t) - 128) / 127
    lnx = load_las(la['nx'], 'nx'); lny = load_las(la['ny'], 'ny')
    lw = ((lnx * lnx + lny * lny) > 0.25).float()           # normal mostly in-plane & present
    txx = gblur(lw * lny * lny, 2.0, 1.0); tyy = gblur(lw * lnx * lnx, 2.0, 1.0); txy = gblur(-lw * lnx * lny, 2.0, 1.0)
    del lnx, lny
else:
    txx = gblur(H * ax_ * ax_, sig_t, 1.5); tyy = gblur(H * ay_ * ay_, sig_t, 1.5); txy = gblur(H * ax_ * ay_, sig_t, 1.5)
stage('normals load + tensor')
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
if args.orient == 'none':
    it, nf = 0, 0
elif args.orient == 'iso':
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
    # Only masked (scroll) voxels take part; everything is a flat list over them.
    plane = Hh * Ww
    mi = torch.nonzero(mask.reshape(-1), as_tuple=True)[0]
    mzi = mi // plane; rem = mi % plane
    myf = (rem // Ww).float(); mxf = (rem % Ww).float()
    txm, tym = nyv.reshape(-1)[mi], -nxv.reshape(-1)[mi]      # unsigned tangent, fixed
    cwm = cw.reshape(-1)[mi]
    nxm, nym = nxv.reshape(-1)[mi].clone(), nyv.reshape(-1)[mi].clone()
    free = r_c.reshape(-1)[rem] > 25                           # leave the core alone
    offs = [k for k in range(-args.orient_reach, args.orient_reach + 1, args.orient_step) if k != 0]
    # neighbour indices: precomputed when they fit in ~12 GB, otherwise recomputed every sweep
    precompute = len(mi) * (len(offs) + 4) * 4 < 12e9
    def nbr_iter():
        for k in offs:
            sx = (mxf + k * txm).round().clamp(0, Ww - 1).long(); sy = (myf + k * tym).round().clamp(0, Hh - 1).long()
            yield (mzi * plane + sy * Ww + sx)
        for dzs in (-2, -1, 1, 2):
            zz = mzi + dzs
            ok = (zz >= 0) & (zz < D)
            yield torch.where(ok, mi + dzs * plane, mi)             # self when out of range
    nbr = [I.to(torch.int32) for I in nbr_iter()] if precompute else None
    print(f'  orientation sync: {len(mi)} voxels, neighbour indices {"precomputed" if precompute else "on the fly"}')
    zw = torch.tensor([1.0] * len(offs) + [2.0] * 4, device=dev)
    del rem
    full_x = torch.zeros(D * plane, device=dev); full_y = torch.zeros(D * plane, device=dev)
    thr = max(1, int(2e-6 * len(mi)))
    for it in range(args.orient_sweeps):
        full_x[mi] = cwm * nxm; full_y[mi] = cwm * nym
        mxs = torch.zeros_like(nxm); mys = torch.zeros_like(nym)
        for j, I in enumerate(nbr if precompute else nbr_iter()):
            mxs += zw[j] * full_x[I.long()]; mys += zw[j] * full_y[I.long()]
        dot = mxs * nxm + mys * nym
        flip = (dot < -0.1 * torch.sqrt(mxs ** 2 + mys ** 2)) & free
        flip &= torch.rand_like(dot) < 0.5                     # damp synchronous flipping
        nf = int(flip.sum())
        nxm = torch.where(flip, -nxm, nxm); nym = torch.where(flip, -nym, nym)
        if it % 50 == 0:
            print(f'  orient sweep {it}: flips {nf}')
        if nf < thr:
            break
    nxv = nxv.reshape(-1).clone(); nyv = nyv.reshape(-1).clone()
    nxv[mi] = nxm; nyv[mi] = nym
    nxv = nxv.reshape(D, Hh, Ww); nyv = nyv.reshape(D, Hh, Ww)
    del nbr, full_x, full_y, mi, mzi, cwm, nxm, nym, free, mxf, myf, txm, tym
    # report how far the result departs from the radial guess
    rad = (nxv * (xx - cx) + nyv * (yy - cy)) < 0
    print('  frac normals pointing toward umbilicus after sync:', float(rad[mask].float().mean()))
print(f'orientation ({args.orient}) stopped after {it+1} iters, last flips {nf}')
theta_n = torch.atan2(nyv, nxv)
stage('orientation sync')

if args.init == 'none':
    # graph-only runs: no gradient-integration init; the sheet graph provides everything
    assert args.sheetgraph or args.track_targets, '--init none needs --sheetgraph or --track_targets'
    wgrid = mask.float() + 1e-4
    conf = torch.zeros_like(s)
    del txx, tyy, txy, coh, cw, H, V, pr, ax_, ay_, nz2
    torch.cuda.empty_cache()
elif args.init == 'bandcount':
    # Count windings directly: d(2*pi*w)/dn = 2*pi * (rate of entering a surf band along +n).
    # Each band entered adds exactly +1 whatever the distance between bands.
    assert args.signal != 'fiber', 'bandcount init needs the surf signal'
    gbx = torch.zeros_like(band_init); gby = torch.zeros_like(band_init)
    gbx[:, :, 1:-1] = 0.5 * (band_init[:, :, 2:] - band_init[:, :, :-2]); gby[:, 1:-1] = 0.5 * (band_init[:, 2:] - band_init[:, :-2])
    dbn = nxv * gbx + nyv * gby
    if args.bandcount_mode == 'entry':
        kn_c = 2 * np.pi * torch.relu(dbn) * mask
    else:                                   # entry + exit, half each: independent of normal sign
        kn_c = np.pi * dbn.abs() * mask
    del dbn
    if args.count_sigma > 0:
        # local band-entry density, smoothed over a few px (no long-range pitch assumption);
        # the raw per-band spikes carry a large curl component that LS projects away
        kn_c = gblur(kn_c, args.count_sigma, 1.0) / gblur(mask.float(), args.count_sigma, 1.0).clamp(min=1e-3) * mask
    print('bandcount: mean entries per px along n in mask', float(kn_c[mask].mean() / (2 * np.pi)))
    phi = torch.zeros_like(s)
    conf = torch.zeros_like(s)
    # trust the integration only where surf bands were actually predicted nearby;
    # band-free areas would otherwise contribute 'zero windings' paths and drag the count down
    cover = (gblur((band_init > 0.5).float(), args.cover_sigma) / args.cover_ref).clamp(0, 1)
    wgrid = mask.float() * (args.cover_floor + (1 - args.cover_floor) * cover ** 2) + 1e-4
    print('bandcount: mask frac with low band coverage (<0.3):', float((cover[mask] < 0.3).float().mean()))
    del cover
    del gbx, gby, band_init, txx, tyy, txy, coh, cw, H, V, pr, ax_, ay_, nz2
    torch.cuda.empty_cache()
else:
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
    ZC = 12                                                      # slices per FFT chunk (memory)
    for c0 in range(0, D, ZC):
        c1 = min(D, c0 + ZC); sl = slice(c0, c1)
        Sf = torch.fft.fft2(F.pad(s[sl], (0, Wp - Ww, 0, Hp - Hh)))
        for lam in lams:
            sn, st = 0.45 * lam, 1.2 * lam
            re = torch.zeros_like(s[sl]); im = torch.zeros_like(s[sl])
            for k in range(n_or):
                th = 2 * np.pi * k / n_or
                u_ = fx * np.cos(th) + fy * np.sin(th); v_ = -fx * np.sin(th) + fy * np.cos(th)
                G = torch.exp(-2 * np.pi ** 2 * ((u_ - 1 / lam) ** 2 * sn ** 2 + v_ ** 2 * st ** 2))
                r = torch.view_as_real(torch.fft.ifft2(Sf * G))[:, :Hh, :Ww]
                wk = (1 - fr[sl]) * (k0[sl] == k) + fr[sl] * (k1[sl] == k)
                re += wk * r[..., 0]; im += wk * r[..., 1]
                del r
            mag = torch.sqrt(re * re + im * im) * np.sqrt(sn * st)
            upd = mag > best_mag[sl]
            best_mag[sl] = torch.where(upd, mag, best_mag[sl]); RE[sl] = torch.where(upd, re, RE[sl]); IM[sl] = torch.where(upd, im, IM[sl])
        del Sf
    phi = torch.atan2(IM, RE)
    print(f'analytic signal done {time.time()-t0:.1f}s')

    # ---------------- instantaneous frequency -> gradient target ----------------
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
stage('init target (gabor or bandcount)')
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
    # w = sheet index: constant along a sheet, +1 per sheet crossed, jumps by 1 at the theta cut.
    # u = 2pi*w - sigma*theta is single valued (it equals 2pi * the continuous turn count).
    u = ls_integrate(kn_c * nxv - sigma * dthx, kn_c * nyv - sigma * dthy, wgrid)
    stage('LS integrate')
    psi = torch.remainder(phi - sigma * theta + np.pi, 2 * np.pi) - np.pi
    u = u.requires_grad_(True)
    opt = torch.optim.Adam([u], lr=0.05, fused=True)
    def phase_loss(u):
        data = (conf * (1 - torch.cos(u - psi))).sum()
        dx = u[:, :, 1:] - u[:, :, :-1]; dy = u[:, 1:] - u[:, :-1]; dzz = u[1:] - u[:-1]
        prior = (wx * (dx - gx) ** 2).sum() + (wy * (dy - gy) ** 2).sum() + (wz * dzz ** 2).sum()
        lxx = u[:, :, 2:] - 2 * u[:, :, 1:-1] + u[:, :, :-2]; lyy = u[:, 2:] - 2 * u[:, 1:-1] + u[:, :-2]
        smooth = (mx[:, :, 1:] * mx[:, :, :-1] * lxx ** 2).sum() + (my[:, 1:] * my[:, :-1] * lyy ** 2).sum()
        return data + 0.3 * prior + 2.0 * smooth, data, prior, smooth
    lf = phase_loss if args.no_compile else torch.compile(phase_loss)
    for it in range(args.iters):
        opt.zero_grad(set_to_none=True)
        loss, data, prior, smooth = lf(u)
        loss.backward(); opt.step()
    if args.iters == 0:
        print(f'sigma {sigma:+.0f}: LS init only ({time.time()-t0:.0f}s)')
        return u.detach(), 0.0
    print(f'sigma {sigma:+.0f}: data {data.item():.0f} prior {prior.item():.0f} smooth {smooth.item():.0f}  ({time.time()-t0:.0f}s)')
    return u.detach(), data.item()

if args.init == 'none':
    assert args.sigma, '--init none needs an explicit --sigma'
    sigma = args.sigma
    u = torch.zeros_like(wgrid)
else:
    sigmas = [args.sigma] if args.sigma else [1.0, -1.0]
    res = {sg_: solve(sg_) for sg_ in sigmas}
    sigma = min(res, key=lambda k: res[k][1])
    u = res[sigma][0]
    del res
stage('phase refine (adam)')

if args.sheetgraph:
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    import sheetgraph as sgm
    th2d = theta[0].cpu().numpy().astype(np.float64)
    pid, npieces = sgm.build_pieces(front_np, th2d, args.sg_cell)
    stage('sheetgraph: pieces')
    nim = sgm.nearest_index_maps(pid)
    ea = sgm.along_edges(pid, th2d, sigma)
    ez = sgm.z_edges(pid, nim, th2d, sigma)
    pid_t = torch.from_numpy(pid).to(dev)
    ex = sgm.across_edges(pid_t, nxv, nyv, th2d, sigma, return_src=True, min_t=args.sg_across_min_t, min_par=args.sg_across_min_par)
    parts = [('along', ea, args.sg_w_along), ('z', ez, args.sg_w_z), ('across', ex, args.sg_w_across)]
    ex_src = ex[3]; parts[2] = ('across', ex[:3], args.sg_w_across)
    if args.wi_root:
        ew = sgm.wi_edges(args.wi_root, zs, 2 ** (int(args.level) - 2), y0, x0, pid, nim, th2d, sigma, return_pix=True)
        anchors = ew[3].pop('anchors')
        print('  wi stats', ew[3]); parts.append(('wi', ew[:3], args.sg_w_wi))
        if args.sg_w_seed > 0:
            # node `npieces` is a reference pinned at 0: edge (ref -> piece) says x_piece = seed + level
            # (absolute winding numbers; the theta cut is irrelevant here because it is a per-node value)
            an, av = anchors
            parts.append(('seed', (np.full(len(an), npieces), an, av), args.sg_w_seed))
            print(f'  seed anchors: {len(an)} crossings on {len(np.unique(an))} pieces')
        if args.sg_across_fallback > 0:
            cov = np.zeros(pid.shape, bool)
            wz_, wy_, wx_ = ew[4]
            cov[wz_, wy_, wx_] = True
            cov = np.stack([ndi.distance_transform_edt(~c) <= args.sg_across_fallback for c in cov])
            sz_, sy_, sx_ = ex_src
            keep = ~cov[sz_, sy_, sx_]
            a_, b_, d_ = ex[:3]
            parts[2] = ('across', (a_[keep], b_[keep], d_[keep]), args.sg_w_across)
            print(f'  across fallback: kept {keep.sum()} of {len(keep)} across edges (outside {args.sg_across_fallback}px of WI coverage, {np.mean(cov):.3f} of slab covered)')
    stage('sheetgraph: edges')
    if args.sg_aggregate:
        agg = []
        for i, p in enumerate(parts):
            a_, b_, d_ = (np.asarray(v) for v in p[1])
            if p[0] == 'across' and args.sg_across_ineq:
                lo, hi, dd = a_, b_, d_                           # one-sided: keep direction
            else:
                lo, hi = np.minimum(a_, b_), np.maximum(a_, b_)
                dd = np.where(a_ <= b_, d_, -d_)                  # orient each edge lo -> hi
            key = np.stack([lo, hi, np.rint(dd * 2).astype(np.int64)], 1)
            uk, cnt = np.unique(key, axis=0, return_counts=True)
            agg.append((uk[:, 0], uk[:, 1], uk[:, 2] / 2.0, p[2] * np.sqrt(cnt), np.full(len(uk), i)))
            print(f'  aggregate {p[0]}: {len(a_)} -> {len(uk)} edges')
        A_ = np.concatenate([g[0] for g in agg]); B_ = np.concatenate([g[1] for g in agg])
        Dd_ = np.concatenate([g[2] for g in agg]); Wt_ = np.concatenate([g[3] for g in agg]); kind = np.concatenate([g[4] for g in agg])
    else:
        A_ = np.concatenate([p[1][0] for p in parts]).astype(np.int64); B_ = np.concatenate([p[1][1] for p in parts]).astype(np.int64)
        Dd_ = np.concatenate([p[1][2] for p in parts]).astype(np.float64)
        Wt_ = np.concatenate([np.full(len(p[1][0]), p[2]) for p in parts])
        kind = np.concatenate([np.full(len(p[1][0]), i) for i, p in enumerate(parts)])
    print(f'  pieces {npieces}; edges ' + ', '.join(f'{p[0]} {len(p[1][0])}' for p in parts))
    # prior: current field's sheet index per piece (only matters for islands)
    wcur = ((u + sigma * theta) / (2 * np.pi)).cpu().numpy()
    sel = pid >= 0
    prior = np.bincount(pid[sel], wcur[sel], minlength=npieces) / np.maximum(np.bincount(pid[sel], minlength=npieces), 1)
    del wcur
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    ncc = npieces + (1 if args.sg_w_seed > 0 else 0)
    # components of the relative graph only (seed anchors would join everything through the reference)
    rel_e = kind != ([p[0] for p in parts].index('seed') if args.sg_w_seed > 0 else -1)
    ncomp, clab = connected_components(coo_matrix((np.ones(int(rel_e.sum())), (A_[rel_e], B_[rel_e])), shape=(ncc, ncc)), directed=False)
    csize = np.bincount(clab)
    big = csize[clab] >= args.sg_min_comp
    top = np.sort(csize)[::-1][:5]
    print(f'  graph components: {ncomp}; largest {top.tolist()}; pieces in components >= {args.sg_min_comp}: {big.mean():.3f} '
          f'({np.sum(csize >= args.sg_min_comp)} such components)')
    if args.init == 'none':
        prior = np.zeros(npieces)
    ineq = (kind == [p[0] for p in parts].index('across')) if args.sg_across_ineq else None
    nn_ = npieces + (1 if args.sg_w_seed > 0 else 0)
    if nn_ > npieces:
        prior = np.r_[prior, 0.0]
    pw_vec = np.full(nn_, args.sg_prior)
    if nn_ > npieces:
        pw_vec[-1] = 1e6                                          # pin the reference node
    if args.save_graph:
        np.savez_compressed(args.save_graph, npieces=npieces, A=A_.astype(np.int64), B=B_.astype(np.int64), D=Dd_, W=Wt_,
                            kind=kind.astype(np.int8), part_names=np.array([p[0] for p in parts]),
                            ineq=(ineq if ineq is not None else np.zeros(len(A_), bool)), pid=pid, zs=np.array(zs),
                            crop=np.array([y0, y1, x0, x1]), cx=cx, cy=cy, sigma=sigma)
        stage('sheetgraph: graph saved')
        if args.graph_only:
            print(f'graph written to {args.save_graph}; total {time.time()-t0:.0f}s')
            raise SystemExit(0)
    if args.labels_from:
        lab = np.load(args.labels_from)
        assert int(lab['npieces']) == npieces, f'piece count mismatch: {int(lab["npieces"])} vs {npieces}'
        xk = lab['x'].astype(np.float64)
        r_ = xk[B_] - xk[A_] - Dd_
        res_e = np.where(ineq, np.maximum(-r_, 0), np.abs(r_)) if ineq is not None else np.abs(r_)
        print(f'  labels from {args.labels_from} (joint solve)')
    else:
        xk, res_e = sgm.solve_l1(nn_, A_, B_, Dd_, Wt_, prior, pw_vec, ineq=ineq, ineq_up=args.sg_ineq_up)
    # xk keeps the reference node (if any) at index npieces; pid only indexes real pieces
    if args.sg_integerize:
        res_abs = np.abs(xk[B_] - xk[A_] - Dd_)                  # plain residual for every edge type
        xk, cand = sgm.integerize(npieces, A_, B_, Dd_, Wt_, xk, res_abs)
        r_int = xk[B_] - xk[A_] - Dd_
        res_e = np.where(ineq, np.maximum(-r_int, 0), np.abs(r_int)) if ineq is not None else np.abs(r_int)
        print(f'  integerize: {cand:.3f} of edges were tree candidates; labels integral: {np.allclose(xk, np.rint(xk))}')
    if args.sg_round_local > 0:
        # local fractional offset phi of the (approximately integral) solution, then snap to phi-relative integers
        pidc = torch.from_numpy(pid).to(dev); onp = pidc >= 0
        xt = torch.from_numpy(xk.astype(np.float32)).to(dev)
        ang = 2 * np.pi * xt[pidc.clamp(min=0).long()]
        cr = gblur(torch.where(onp, torch.cos(ang), torch.zeros_like(ang)), args.sg_round_local, 2.0)
        ci = gblur(torch.where(onp, torch.sin(ang), torch.zeros_like(ang)), args.sg_round_local, 2.0)
        phi_pix = torch.atan2(ci, cr) / (2 * np.pi)
        cnt = torch.bincount(pidc[onp].long(), minlength=npieces).clamp(min=1).float()
        # per-piece circular mean of phi
        pc = torch.zeros(npieces, device=dev).index_add_(0, pidc[onp].long(), torch.cos(2 * np.pi * phi_pix[onp]))
        ps = torch.zeros(npieces, device=dev).index_add_(0, pidc[onp].long(), torch.sin(2 * np.pi * phi_pix[onp]))
        phi = (torch.atan2(ps, pc) / (2 * np.pi)).cpu().numpy().astype(np.float64)
        if len(xk) > npieces:
            phi = np.r_[phi, 0.0]                                 # reference node stays at 0
        drift = np.abs(((xk - phi) + 0.5) % 1.0 - 0.5)
        xk = np.rint(xk - phi) if args.sg_round_int else phi + np.rint(xk - phi)
        r_int = xk[B_] - xk[A_] - Dd_
        res_e = np.where(ineq, np.maximum(-r_int, 0), np.abs(r_int)) if ineq is not None else np.abs(r_int)
        print(f'  round_local: |x - local integer| p50 {np.median(drift):.3f} p90 {np.percentile(drift, 90):.3f}')
        del pidc, xt, ang, cr, ci, phi_pix
    for i, p in enumerate(parts):
        m = kind == i
        if not m.any():
            print(f'  edges {p[0]}: none'); continue
        print(f'  edges {p[0]}: off by >0.5 after solve {np.mean(res_e[m] > 0.5):.4f} (n={m.sum()}); residual p50 {np.median(res_e[m]):.3f} p75 {np.percentile(res_e[m], 75):.3f} p90 {np.percentile(res_e[m], 90):.3f}')
    fx = np.abs((xk + 0.5) % 1.0 - 0.5)
    print(f'  piece labels: |x - round(x)| p50 {np.median(fx):.3f} p90 {np.percentile(fx, 90):.3f}')
    # what the solved field says the raw (non-aggregated) across edges should have been
    ra, rb, rd = ex[0], ex[1], ex[2]
    sol = np.rint(xk[rb] - xk[ra] - (rd - 1.0)).astype(int)          # remove cut term, keep the step
    u_, c_ = np.unique(np.clip(sol, -3, 4), return_counts=True)
    print('  across edges, solved step (claimed +1):', {int(k): round(v / len(sol), 3) for k, v in zip(u_, c_)})
    stage('sheetgraph: L1 solve')
    # fill the voxel field: u = 2pi*k - sigma*theta at centreline pixels
    tgt = torch.zeros_like(u); dwt = torch.zeros_like(u)
    kpix = torch.from_numpy(xk.astype(np.float32)).to(dev)[pid_t.clamp(min=0).long()]
    on = pid_t >= 0
    if args.init == 'none':
        on &= torch.from_numpy(big).to(dev)[pid_t.clamp(min=0).long()]     # islands are interpolated, not trusted
    tgt[on] = (2 * np.pi * kpix - sigma * theta)[on]; dwt[on] = args.sg_fill_w
    if args.sg_fill_back > 0:
        # back layer at half a winding: V fibers between fronts k and k+1 -> k + 0.5
        lab_t = torch.from_numpy(xk.astype(np.float32)).to(dev)
        vsel = mask & (pr_keep.float() > 0.3) & (nz2_keep.float() > 0.8) & ~on
        vsel &= (theta.abs() < np.pi - 0.05)                     # skip the theta branch cut
        bz, by, bx, bk = sgm.back_targets(pid_t, lab_t, nxv, nyv, vsel)
        tgt[bz, by, bx] = 2 * np.pi * (bk + 0.5) - sigma * theta[bz, by, bx]
        dwt[bz, by, bx] = args.sg_fill_back
        print(f'  back targets: {len(bz)} V voxels of {int(vsel.sum())} lie between consecutive fronts')
        del lab_t, vsel, bz, by, bx, bk
    u = ls_fill(tgt, dwt, wgrid, iters=tuple(args.fill_iters))
    wf_ = (u + sigma * theta) / (2 * np.pi)
    e_ = (wf_ - kpix)[on]
    lf_ = torch.remainder(kpix[on], 1.0)
    print(f'  fill check at centreline pixels: |w - label| p50 {float(e_.abs().median()):.3f} p90 {float(torch.quantile(e_.abs()[::13], 0.9)):.3f}; '
          f'label frac spread in 64px blocks: see phase report')
    del wf_, e_, lf_
    del tgt, dwt, kpix, pid_t, nim
    stage('sheetgraph: fill')

if args.track_targets:
    # voxel fill from the global track-graph labels (H points -> k, V points -> k + 0.5, sg_fill_w / sg_fill_back);
    # labels use the cut about the umbilicus at the point's z, converted here to this slab's cut
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    import sheetgraph as sgm
    assert args.level == '3', 'track targets are on the level-3 grid'
    for _n in ('wx', 'wy', 'wz', 'mx', 'my', 'mz', 'dthx', 'dthy', 'rr2', 'theta_n', 'r_c'):
        globals().pop(_n, None)                                   # phase-solve buffers: unused here, free GPU memory for the fill
    torch.cuda.empty_cache()
    ti = np.load(f'{args.track_targets}/index.npz')
    ta = {k: np.load(f'{args.track_targets}/{k}.npy', mmap_mode='r') for k in ('y', 'x', 'w', 'theta')}
    lut = {int(z): (int(a), int(b)) for z, a, b in zip(ti['z3'], ti['start'], ti['end'])}
    th_np = theta[0].cpu().numpy().astype(np.float64)
    tgt = torch.zeros_like(u); dwt = torch.zeros_like(u)
    n_pts = n_vox = n_conf = 0
    for k, z in enumerate(zs):
        if z not in lut:
            continue
        a, b = lut[z]
        iy = np.rint(np.asarray(ta['y'][a:b])).astype(np.int64) - y0; ix = np.rint(np.asarray(ta['x'][a:b])).astype(np.int64) - x0
        ok = (iy >= 0) & (iy < Hh) & (ix >= 0) & (ix < Ww)
        iy, ix = iy[ok], ix[ok]
        tw = np.asarray(ta['w'][a:b])[ok].astype(np.float64); tth = np.asarray(ta['theta'][a:b])[ok].astype(np.float64)
        isv = (tw % 1.0) != 0
        ws = tw - sgm.cut_term(th_np[iy, ix], tth, sigma)                # point cut -> slab cut at the voxel centre
        wt = np.where(isv, args.sg_fill_back, args.sg_fill_w)
        flat = iy * Ww + ix
        uq, inv, cnt = np.unique(flat, return_inverse=True, return_counts=True)
        sw = np.bincount(inv, wt); m1 = np.bincount(inv, wt * ws) / np.maximum(sw, 1e-9); m2 = np.bincount(inv, wt * ws * ws) / np.maximum(sw, 1e-9)
        good = (sw > 0) & (np.sqrt(np.maximum(m2 - m1 * m1, 0)) < 0.3)  # points in one voxel that disagree: no target
        n_pts += len(flat); n_vox += int(good.sum()); n_conf += int((~good & (sw > 0)).sum())
        vy, vx = torch.from_numpy(uq[good] // Ww).to(dev), torch.from_numpy(uq[good] % Ww).to(dev)
        tgt[k, vy, vx] = (2 * np.pi * torch.from_numpy(m1[good]).to(dev).float() - sigma * theta[k, vy, vx])
        dwt[k, vy, vx] = torch.from_numpy(sw[good] / cnt[good]).to(dev).float()
    print(f'  track targets: {n_pts} points -> {n_vox} voxels ({n_conf} voxels dropped for disagreeing labels)')
    for nb in args.bc_from:
        # neighbour slab field on shared slices, converted from its cut (about its centre) to ours: w_me = w_nb + cut(th_nb, th_me)
        g = np.load(nb)
        zsn = [int(v) for v in g['zs']]
        ny0, ny1, nx0, nx1 = (int(v) for v in g['crop'])
        Y0, Y1, X0, X1 = max(y0, ny0), min(y1, ny1), max(x0, nx0), min(x1, nx1)
        yyg, xxg = np.mgrid[Y0:Y1, X0:X1]
        th_nb = np.arctan2(yyg - (float(g['cy']) + ny0), xxg - (float(g['cx']) + nx0))
        th_me = th_np[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0]
        dcut = sgm.cut_term(th_nb, th_me, sigma)
        nbc = 0
        gw, gm = g['w'], g['mask']                                    # npz members reload on every access
        for k, z in enumerate(zs):
            if z not in zsn:
                continue
            kn = zsn.index(z)
            wme = torch.from_numpy((gw[kn, Y0 - ny0:Y1 - ny0, X0 - nx0:X1 - nx0] + dcut).astype(np.float32)).to(dev)
            sel = torch.from_numpy(gm[kn, Y0 - ny0:Y1 - ny0, X0 - nx0:X1 - nx0]).to(dev) & mask[k, Y0 - y0:Y1 - y0, X0 - x0:X1 - x0]
            tv = tgt[k, Y0 - y0:Y1 - y0, X0 - x0:X1 - x0]; dv = dwt[k, Y0 - y0:Y1 - y0, X0 - x0:X1 - x0]
            tv[sel] = (2 * np.pi * wme - sigma * theta[k, Y0 - y0:Y1 - y0, X0 - x0:X1 - x0])[sel]; dv[sel] = args.bc_w
            nbc += 1
        print(f'  seam coupling: {nbc} shared slices fixed to {nb}')
        del g, gw, gm
    u = ls_fill(tgt, dwt, wgrid, iters=tuple(args.fill_iters))
    on = dwt > 0
    e_ = (u - tgt)[on] / (2 * np.pi)
    print(f'  fill check at target voxels: |w - target| p50 {float(e_.abs().median()):.3f} p90 {float(torch.quantile(e_.abs()[::13], 0.9)):.3f}')
    del tgt, dwt, on, e_
    stage('track targets: fill')

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
    opt = torch.optim.Adam([u], lr=0.02, fused=True)
    def layer_loss(u):
        ux = (cdiff(u, 2) + sigma * dthx) / (2 * np.pi)       # dw/dx incl. theta lift
        uy = (cdiff(u, 1) + sigma * dthy) / (2 * np.pi)
        dn = nxv * ux + nyv * uy
        dt = txv * ux + tyv * uy                                      # w (sheet index) is constant along a sheet
        dzw = (u[1:] - u[:-1]) / (2 * np.pi)
        data = -(sn_ * torch.cos(u + sigma * theta)).sum()
        along = args.lam_t * (mf_ * dt ** 2).sum() + args.lam_z * (mz2 * dzw ** 2).sum()
        mono = args.lam_mono * (mf_ * torch.relu(-dn) ** 2).sum()
        l1 = args.lam_l1 * (mf_ * dn.abs()).sum()
        return data + along + mono + l1, data, along, mono, l1
    lf = layer_loss if args.no_compile else torch.compile(layer_loss)
    for it in range(args.layer_iters):
        opt.zero_grad(set_to_none=True)
        loss, data, along, mono, l1 = lf(u)
        loss.backward(); opt.step()
        if it % 500 == 0 or it == args.layer_iters - 1:
            print(f'  layer it {it}: data {data.item():.0f} along {along.item():.0f} mono {mono.item():.0f} l1 {l1.item():.0f}')
    u = u.detach()
    stage('layer refine (adam)')
    report('layer refined', u)
w = (u + sigma * theta) / (2 * np.pi)
mk = mask & (s.abs() > 0.02)
sv, cv = s[mk], torch.cos(2 * np.pi * w[mk])
corr = float(((sv - sv.mean()) * (cv - cv.mean())).mean() / (sv.std() * cv.std()))
gwx, gwy = grad_xy(w)
dec = float(((gwx * nxv + gwy * nyv)[mask] < 0).float().mean())
dzw = (w[1:] - w[:-1]).abs()[mask[1:] & mask[:-1]]
print(f'chosen sigma {sigma:+.0f}; corr(s, cos 2pi w) {corr:.3f}; frac w decreasing along n {dec:.3f}; |dw/dslice| p50/p99 {float(dzw.median()):.3f}/{float(torch.quantile(dzw[::31], 0.99)):.3f}')
if 'front_np' in globals():
    frw = torch.remainder(w, 1.0)
    def _circ(sel):
        v = frw[sel][::7]
        c = torch.complex(torch.cos(2 * np.pi * v).mean(), torch.sin(2 * np.pi * v).mean()) if False else None
        cr, ci = torch.cos(2 * np.pi * v).mean(), torch.sin(2 * np.pi * v).mean()
        return float(torch.atan2(ci, cr) / (2 * np.pi)) % 1, float(torch.sqrt(cr * cr + ci * ci))
    fb = torch.from_numpy(front_np).to(dev) & mask
    print('phase on surf front bands: mean %.2f R %.2f' % _circ(fb))
    if 'pr_keep' in globals():
        print('phase on H fibers: mean %.2f R %.2f' % _circ(mask & (pr_keep > 0.3) & (nz2_keep < 0.2)))
        print('phase on V fibers: mean %.2f R %.2f' % _circ(mask & (pr_keep > 0.3) & (nz2_keep > 0.8)))
wq = w[mask][::97]
print('w range p1/p50/p99', torch.quantile(wq, torch.tensor([0.01, 0.5, 0.99], device=dev)).tolist())
ks = [k for k, z in enumerate(zs) if (args.save_z0 is None or z >= args.save_z0) and (args.save_z1 is None or z < args.save_z1)]
ks = slice(ks[0], ks[-1] + 1)                                   # padded solve: keep only the slab's own slices
np.savez(args.out, w=w[ks].cpu().numpy().astype(np.float32), s=s[ks].cpu().numpy().astype(np.float16), mask=mask[ks].cpu().numpy(),
                    conf=conf[ks].cpu().numpy().astype(np.float16), zs=np.array(zs[ks]), crop=np.array([y0, y1, x0, x1]), sigma=sigma, cx=cx, cy=cy,
                    level=int(args.level))
print(f'total {time.time()-t0:.0f}s')

# ---------------- every run renders its images and scores itself ----------------
if not args.no_report:
    import subprocess, sys as _sys
    code_dir = os.path.dirname(os.path.abspath(__file__))
    run_dir = os.path.dirname(os.path.abspath(args.out))
    py = _sys.executable
    def _run(cmd, out_file=None):
        r = subprocess.run(cmd, capture_output=True, text=True)
        if out_file:
            open(out_file, 'w').write(r.stdout)
        if r.returncode != 0:
            print('report step failed:', ' '.join(cmd), r.stderr[-800:])
        return r.stdout
    ev = _run([py, os.path.join(code_dir, 'eval_windings.py'), args.out, args.ann], os.path.join(run_dir, 'eval.txt'))
    print('[eval]\n' + ev)
    if os.path.isdir(args.eval_mesh):
        em = _run([py, os.path.join(code_dir, 'eval_mesh.py'), args.out, args.eval_mesh, args.umbilicus], os.path.join(run_dir, 'eval_mesh.txt'))
        print('[eval mesh]\n' + em)
    _run([py, os.path.join(code_dir, 'render3d.py'), args.out, os.path.join(run_dir, 'r')])
    _run([py, os.path.join(code_dir, 'overlay_ann.py'), args.out, args.ann, os.path.join(run_dir, 'ann.png')])
    print(f'report written to {run_dir} (eval.txt, r_xy.png, r_zoom.png, r_xz.png, ann.png); total {time.time()-t0:.0f}s')
