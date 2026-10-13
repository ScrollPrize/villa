"""Which side of the surf (front) band do H and V fibers sit on, along the outward normal?

For each slice: march from every surf-band centre voxel along +n and -n (outward
normal, oriented by the umbilicus; only voxels where that is reliable, i.e. away
from the core) and record H and V fiber mass vs signed offset.
"""
import sys
import numpy as np
import torch
import zarr
import torch.nn.functional as F

dev = 'cuda'
W = '/ephemeral/sean/experiments/winding_fields/data/'
R = W + 'fiber_zarrs/PHercParis4-20260411134726-las-sd1-d74cca79_'
z0, z1, dz = 4600, 4800, 25
cx, cy = 2401.5, 1699.0
fib = {k: zarr.open(R + f'{k}.ome.zarr/3', mode='r') for k in ('nx', 'ny', 'presence')}
surf = zarr.open(W + '2um_ds2_ps256_surf.zarr/1', mode='r')
las = {k: zarr.open(W + f'lasagna/las_008_{k}.ome.zarr/4', mode='r') for k in ('nx', 'ny')}
offs = np.arange(-12, 13)
accH = np.zeros(len(offs)); accV = np.zeros(len(offs)); accS = np.zeros(len(offs))
for z in range(z0, z1, dz):
    p = torch.from_numpy(fib['presence'][z].astype(np.float32) / 255).to(dev)
    ax = (torch.from_numpy(fib['nx'][z].astype(np.float32)).to(dev) - 128) / 127
    ay = (torch.from_numpy(fib['ny'][z].astype(np.float32)).to(dev) - 128) / 127
    nz2 = (1 - ax * ax - ay * ay).clamp(0, 1)
    H = p * (1 - nz2); V = p * nz2
    S = torch.from_numpy(surf[z].astype(np.float32) / 255).to(dev)
    lnx = (torch.from_numpy(las['nx'][z // 2].astype(np.float32)).to(dev) - 128) / 127
    lny = (torch.from_numpy(las['ny'][z // 2].astype(np.float32)).to(dev) - 128) / 127
    lnx = lnx.repeat_interleave(2, 0).repeat_interleave(2, 1)[:p.shape[0], :p.shape[1]]
    lny = lny.repeat_interleave(2, 0).repeat_interleave(2, 1)[:p.shape[0], :p.shape[1]]
    nn = torch.sqrt(lnx ** 2 + lny ** 2) + 1e-6
    nx, ny = lnx / nn, lny / nn
    Hh, Ww = p.shape
    yy, xx = torch.meshgrid(torch.arange(Hh, device=dev, dtype=torch.float32), torch.arange(Ww, device=dev, dtype=torch.float32), indexing='ij')
    rx, ry = xx - cx, yy - cy
    r = torch.sqrt(rx ** 2 + ry ** 2)
    cosr = (nx * rx + ny * ry) / (r + 1e-6)
    sg = torch.sign(cosr)
    nx, ny = nx * sg, ny * sg
    # reliable: normal well aligned with radial, away from core, on a thin surf band centre
    Smax = F.max_pool2d(S[None, None], 3, 1, 1)[0, 0]
    seeds = (S > 0.5) & (cosr.abs() > 0.85) & (r > 300) & (nn > 0.5)
    iy, ix = torch.nonzero(seeds, as_tuple=True)
    sel = torch.randperm(len(iy), device=dev)[:400000]
    iy, ix = iy[sel].float(), ix[sel].float()
    nxs, nys = nx[iy.long(), ix.long()], ny[iy.long(), ix.long()]
    for j, o in enumerate(offs):
        qx = (ix + o * nxs).round().long().clamp(0, Ww - 1); qy = (iy + o * nys).round().long().clamp(0, Hh - 1)
        accH[j] += float(H[qy, qx].sum()); accV[j] += float(V[qy, qx].sum()); accS[j] += float(S[qy, qx].sum())
    print('slice', z, 'seeds', len(iy))
nH, nV, nS = accH / accH.sum(), accV / accV.sum(), accS / accS.max()
print('offset (px along OUTWARD normal; + = away from umbilicus)')
print(' off    surf    H       V')
for o, a, b, c in zip(offs, nS, nH, nV):
    print(f'{o:+4d}  {a:6.3f}  {b:6.3f}  {c:6.3f}  ' + 'H' * int(b * 300) + '|' + 'V' * int(c * 300))
print(f'mean offset H {np.sum(offs * nH):+.2f}  V {np.sum(offs * nV):+.2f}  (within +-4: H {np.sum((offs*nH)[np.abs(offs)<=4])/np.sum(nH[np.abs(offs)<=4]):+.2f}, V {np.sum((offs*nV)[np.abs(offs)<=4])/np.sum(nV[np.abs(offs)<=4]):+.2f})')
