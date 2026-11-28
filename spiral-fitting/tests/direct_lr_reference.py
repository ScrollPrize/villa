"""The eager reference of the fused direct-LR RK4 integrators (flow_triton's
TritonRK4DirectIntegrate, flow_metal's MetalRK4DirectIntegrate): trilinear
samples at the lattice coordinates `normalised * a + b` from
flow_triton._field_geoms (border clamp), RK4 with the kernels' stage
arithmetic. tests/test_flow_direct_fused.py holds every backend to it;
tests/test_flow_interpolant.py runs it in place of a kernel.
"""
import torch

import flow_triton


def trilinear(field, p, geom):
    # field :: 3, Z, Y, X; p :: n, 3 normalised zyx; geom = az,bz,ay,by,ax,bx.
    size = field.shape[1:]
    coords = []
    for d in range(3):
        c = p[:, d] * geom[2 * d] + geom[2 * d + 1]
        coords.append(c.clamp(min=0.0, max=float(size[d] - 1)))
    lo = [c.detach().floor().clamp(max=size[d] - 2) for d, c in enumerate(coords)]
    frac = [c - l for c, l in zip(coords, lo)]
    lo = [l.to(torch.int64) for l in lo]
    out = 0.0
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                w = ((frac[0] if dz else 1 - frac[0])
                     * (frac[1] if dy else 1 - frac[1])
                     * (frac[2] if dx else 1 - frac[2]))
                v = field[:, lo[0] + dz, lo[1] + dy, lo[2] + dx]  # 3, n
                out = out + v.T * w[:, None]
    return out


def reference_integrate(y, low, high, lo_scale, hi_scale, h, n_steps,
                         reverse):
    geom = flow_triton._field_geoms(low, high)
    slabs = range(low.shape[0] - 1, -1, -1) if reverse else range(low.shape[0])
    for slab in slabs:
        def velocity(p):
            return (trilinear(low[slab], p, geom[:6]) * lo_scale
                    + trilinear(high[slab], p, geom[6:]) * hi_scale)
        for _ in range(n_steps):
            k1 = velocity(y)
            k2 = velocity(y + (h / 2) * k1)
            k3 = velocity(y + (h / 2) * k2)
            k4 = velocity(y + h * k3)
            y = y + (h / 6) * (((k1 + 2.0 * k2) + 2.0 * k3) + k4)
    return y
