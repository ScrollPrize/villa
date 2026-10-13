"""Frozen UV -> XYZ observations, with no correspondence search or normal field."""
from __future__ import annotations

import math
import torch


def ramp_weight(bounds, step: int, steps: int) -> float:
    """Absolute positive data weights; never introduce a free-smoothing phase."""
    if len(bounds) != 2:
        raise ValueError('fixed_uv_ramp must contain [start, end]')
    start, end = map(float, bounds)
    if not all(math.isfinite(x) and x > 0 for x in (start, end)) or end < start:
        raise ValueError('fixed_uv_ramp requires 0 < start <= end, both finite')
    fraction = min(1., max(0., step / max(1, steps)))
    return start * (end / start) ** fraction


class FixedUVTargets:
    """Precompute four vertex indices/weights for fixed mesh-grid coordinates.

    Lasagna's XYZ pyramid always evaluates on the same finest grid: min_scaledown
    controls trainable residual levels, not the observation coordinate system.
    A geometry-changing grow/crop is deliberately rejected rather than silently
    rebinding the observations. UV is (column, row), XYZ is in base voxels.
    """
    def __init__(self, grid_uv, xyz, length_weights, shape, *, device='cpu'):
        self.shape = tuple(map(int, shape))
        if len(self.shape) != 2 or min(self.shape) < 2:
            raise ValueError('Expected an H,W surface with at least two vertices per axis')
        uv = torch.as_tensor(grid_uv, dtype=torch.float32, device=device).detach().clone()
        self.xyz = torch.as_tensor(xyz, dtype=torch.float32, device=device).detach().clone()
        weight = torch.as_tensor(length_weights, dtype=torch.float32, device=device).detach().clone()
        n = len(uv)
        if n == 0 or uv.shape != (n, 2) or self.xyz.shape != (n, 3) or weight.shape != (n,):
            raise ValueError('Expected nonempty UV[N,2], XYZ[N,3], length_weights[N]')
        if not all(bool(torch.isfinite(t).all()) for t in (uv, self.xyz, weight)) or bool((weight <= 0).any()):
            raise ValueError('Targets and positive arc weights must be finite')
        h, w = self.shape
        if bool(((uv < 0) | (uv > uv.new_tensor([w-1, h-1]))).any()):
            raise ValueError('Fixed UV observation lies outside mesh')
        col = uv[:, 0].floor().long().clamp(max=w-2)
        row = uv[:, 1].floor().long().clamp(max=h-2)
        u, v = uv[:, 0]-col, uv[:, 1]-row
        base = row*w + col
        self.indices = torch.stack((base, base+1, base+w, base+w+1), dim=1)
        self.weights = torch.stack(((1-u)*(1-v), u*(1-v), (1-u)*v, u*v), dim=1)
        self.arc_weights = weight / weight.sum()

    def predict(self, xyz):
        if tuple(xyz.shape) != (1, *self.shape, 3):
            raise ValueError('Fixed UV targets require unchanged depth=1 surface geometry')
        return (xyz.reshape(-1, 3)[self.indices] * self.weights[..., None]).sum(1)

    def loss(self, *, res):
        squared = (self.predict(res.xyz_lr) - self.xyz).square().sum(-1)
        loss = (squared * self.arc_weights).sum() / float(res.params.mesh_step)**2
        lm = (squared / float(res.params.mesh_step)**2)[None, None, None, :]
        return loss, (lm,), (torch.ones_like(lm),)

    @torch.no_grad()
    def statistics(self, xyz):
        error = (self.predict(xyz)-self.xyz).norm(dim=-1)
        return dict(samples=len(error), mean_base=float((error*self.arc_weights).sum()),
                    rms_base=float((error.square()*self.arc_weights).sum().sqrt()),
                    median_base=float(error.median()), p95_base=float(torch.quantile(error, .95)),
                    max_base=float(error.max()))
