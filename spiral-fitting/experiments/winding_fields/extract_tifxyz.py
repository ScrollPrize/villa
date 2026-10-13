"""One tifxyz per integer winding from the stitched full-scroll field.

The field w is the sheet index (constant along a sheet, +1 per sheet outward, jumping by
one at the theta branch cut of its slab). The winding-k surface is the iso-surface w = k:
one turn of the spiral from theta = -pi to +pi about the slab centre.

Grid: rows every `step` base voxels in z (solved slices), columns uniform in theta with
about `step` base voxels of arc at that winding's median radius. Per row the slice is
resampled in polar coordinates; per winding and angle, the crossing of w = k is located
with linear interpolation. If a ray crosses k several times (folds), the crossing closest
to a reference radius (median-filtered over angle from unambiguous angles) is kept.
Output in the base frame (fiber level 2 = 2 x level 3), invalid vertices -1.
"""
import glob, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F

W = '/ephemeral/sean/experiments/winding_fields'
sys.path.insert(0, '/ephemeral/sean/villa/spiral-fitting')
from tifxyz import save_tifxyz                                     # repo writer

root = sys.argv[1] if len(sys.argv) > 1 else f'{W}/runs/full'
out_dir = sys.argv[2] if len(sys.argv) > 2 else f'{root}/tifxyz'
step = float(sys.argv[3]) if len(sys.argv) > 3 else 18.0          # base voxels
dev = 'cuda'
meta = json.load(open(f'{root}/offsets.json'))
slab_dirs, rel, abs_off = meta['slabs'], meta['relative_offsets'], meta['absolute_offset']

# which slab owns which solved slice: the slab whose centre is nearest
info = []
for sd in slab_dirs:
    d = np.load(f'{sd}/w.npz', mmap_mode='r')
    info.append((d['zs'].copy(), d['crop'].copy(), float(d['cx']), float(d['cy'])))
sigma = float(np.load(f'{slab_dirs[0]}/w.npz', mmap_mode='r')['sigma'])
# one centre per row (umbilicus at the row z, level-3 global) so columns and the branch cut do not jump between slabs
U = np.array([[p['z'], p['y'], p['x']] for p in json.load(open(f'{W}/data/umbilicus.json'))['control_points']], float)
U = U[np.argsort(U[:, 0])]
def row_centre(z):
    return np.interp(2.0 * z, U[:, 0], U[:, 2]) / 2.0, np.interp(2.0 * z, U[:, 0], U[:, 1]) / 2.0
all_z = sorted(set(int(z) for zs, *_ in info for z in zs))
row_stride = max(1, int(round(step / 2 / (all_z[1] - all_z[0]))))      # level-3 slice stride for `step` base vx
row_z = all_z[::row_stride]
owner = {}
for z in row_z:
    best = min((abs(z - (zs[0] + zs[-1]) / 2), i) for i, (zs, *_r) in enumerate(info) if zs[0] <= z <= zs[-1])
    owner[z] = best[1]
print(f'{len(row_z)} rows (every {row_stride} solved slices = {row_stride * (all_z[1] - all_z[0]) * 2} base vx)')

# polar sampling resolution
NA = 8192                                     # angles per row (fine); per-winding columns are subsampled from these
DR = 0.5                                      # level-3 px radial step
th = torch.linspace(-np.pi, np.pi, NA + 1, device=dev)[:-1] + np.pi / NA


def polar(w2d, m2d, cx, cy, rmax):
    r = torch.arange(0, rmax, DR, device=dev)
    X = cx + r[None, :] * torch.cos(th)[:, None]; Y = cy + r[None, :] * torch.sin(th)[:, None]
    H, Wd = w2d.shape
    g = torch.stack([X / (Wd - 1) * 2 - 1, Y / (H - 1) * 2 - 1], -1)[None]
    wv = F.grid_sample(w2d[None, None], g, mode='bilinear', align_corners=True)[0, 0]
    mv = F.grid_sample(m2d[None, None].float(), g, mode='nearest', align_corners=True)[0, 0] > 0.5
    return torch.where(mv, wv, torch.full_like(wv, float('nan'))), r


rows = {}                                      # k -> dict(z -> (radius per angle))
cache = {}
for zi, z in enumerate(row_z):
    si = owner[z]
    if si not in cache:
        cache.clear()
        d = np.load(f'{slab_dirs[si]}/w.npz')
        cache[si] = (torch.from_numpy(d['w']).to(dev), torch.from_numpy(d['mask']).to(dev))
    w3, m3 = cache[si]
    zs, crop, cx, cy = info[si]
    k_i = int(np.nonzero(zs == z)[0][0])
    w2d = w3[k_i] + rel[si] + abs_off
    # re-cut w from the slab centre to the row centre: w_row = w + sigma * (cut jump between the two angles)
    gx, gy = row_centre(z)
    cx, cy = gx - crop[2], gy - crop[0]                            # row centre in this slab's crop
    yy = torch.arange(w2d.shape[0], device=dev, dtype=torch.float32)[:, None]; xx = torch.arange(w2d.shape[1], device=dev, dtype=torch.float32)[None]
    dth = torch.atan2(yy - cy, xx - cx) - torch.atan2(yy - info[si][3], xx - info[si][2])
    w2d = w2d + sigma * torch.round((dth - torch.atan2(torch.sin(dth), torch.cos(dth))) / (2 * np.pi))
    rmax = float(np.hypot(max(cx, w2d.shape[1] - cx), max(cy, w2d.shape[0] - cy)))
    P, r = polar(w2d, m3[k_i], cx, cy, rmax)
    P[:, : int(20 / DR)] = float('nan')                           # ignore the core
    a, b = P[:, :-1], P[:, 1:]
    ok = ~(torch.isnan(a) | torch.isnan(b))
    lo = torch.floor(torch.minimum(a, b)) + 1; hi = torch.floor(torch.maximum(a, b))
    kmin = int(torch.nan_to_num(lo, nan=1e9).min()); kmax = int(torch.nan_to_num(hi, nan=-1e9).max())
    for k in range(kmin, kmax + 1):
        c = ok & (((a < k) & (b >= k)) | ((a >= k) & (b < k)))
        if c.sum() < 50:
            continue
        frac = ((k - a) / (b - a)).clamp(0, 1)
        rc = torch.where(c, r[:-1][None] + frac * DR, torch.full_like(a, float('nan')))
        n = c.sum(1)
        single = n == 1
        rs = torch.nanmean(torch.where(single[:, None], rc, torch.full_like(rc, float('nan'))), 1)
        # reference radius over angle: running median of single-crossing radii (circular)
        ref = rs.clone()
        if single.sum() > 20:
            idx = torch.nonzero(single).squeeze(1)
            vals = rs[idx]
            pos = torch.searchsorted(idx, torch.arange(NA, device=dev)).clamp(max=len(idx) - 1)
            ref = vals[pos]
            win = 65
            pad = torch.cat([ref[-win:], ref, ref[:win]])
            ref = pad.unfold(0, 2 * win + 1, 1).median(1).values[:NA]
        pick = torch.nan_to_num((rc - ref[:, None]).abs(), nan=1e9).argmin(1)
        rk = rc.gather(1, pick[:, None])[:, 0]
        rk = torch.where(n > 0, rk, torch.full_like(rk, float('nan')))
        rows.setdefault(k, {})[z] = (rk.cpu().numpy(), si, cx, cy)
    if zi % 50 == 0:
        print(f'row {zi}/{len(row_z)} z {z}: windings {kmin}..{kmax}', flush=True)

# topology check: at the same row and angle, winding k+1 must lie outside winding k
cross_n = cross_bad = 0
for k in sorted(rows):
    if k + 1 not in rows:
        continue
    for z, (rk, *_r) in rows[k].items():
        if z in rows[k + 1]:
            rk1 = rows[k + 1][z][0]
            v = np.isfinite(rk) & np.isfinite(rk1)
            cross_n += int(v.sum()); cross_bad += int((rk1[v] <= rk[v]).sum())
print(f'topology: winding k+1 not outside winding k at {cross_bad} of {cross_n} (row, angle) samples ({cross_bad / max(cross_n, 1):.4f})')

os.makedirs(out_dir, exist_ok=True)
summary = []
for k in sorted(rows):
    rz = rows[k]
    if len(rz) < 5:
        continue
    zsk = sorted(rz)
    med_r = np.nanmedian(np.concatenate([rz[z][0] for z in zsk]))
    ncol = int(max(16, np.ceil(2 * np.pi * med_r * 2.0 / step)))      # med_r is level-3 px -> base = x2
    cols = np.linspace(0, NA, ncol, endpoint=False).astype(int)
    zyx = np.full((len(row_z), ncol, 3), -1.0, np.float32)
    for z in zsk:
        rk, si, cx, cy = rz[z]
        zs, crop, _cx, _cy = info[si]
        y0, y1, x0, x1 = crop
        t = th.cpu().numpy()[cols]; rr = rk[cols]
        v = np.isfinite(rr)
        ri = row_z.index(z)
        zyx[ri, v, 0] = 2.0 * z
        zyx[ri, v, 1] = 2.0 * (y0 + cy + rr[v] * np.sin(t[v]))
        zyx[ri, v, 2] = 2.0 * (x0 + cx + rr[v] * np.cos(t[v]))
    valid_rows = np.nonzero((zyx[..., 0] > 0).any(1))[0]
    if len(valid_rows) == 0:                                    # crossings too sparse to land on any column
        print(f"winding_{k:03d}: no valid vertices after column sampling, skipped"); continue
    zyx = zyx[valid_rows.min():valid_rows.max() + 1]
    name = f'winding_{k:03d}'
    save_tifxyz(zyx, out_dir, name, step, 9.6, 'winding_fields sheet-graph (fiber/surf/winding_inference), iso-surface w=k')
    vf = float((zyx[..., 0] > 0).mean())
    summary.append({'winding': k, 'rows': int(zyx.shape[0]), 'cols': ncol, 'valid_frac': vf, 'median_radius_base': float(2 * med_r)})
    print(f'{name}: {zyx.shape[0]} x {ncol}, valid {vf:.2f}, median radius {2 * med_r:.0f} base vx', flush=True)
json.dump(summary, open(f'{out_dir}/summary.json', 'w'), indent=1)
print('EXTRACT_DONE', len(summary))
