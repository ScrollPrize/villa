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


class SlidingUVTargets(FixedUVTargets):
    """Joint attachment/mesh optimization with explicitly separated gradients."""
    def __init__(self, grid_uv, xyz, length_weights, shape, *, arc, scales, source,
                 step, piece=None, device='cpu', uv_lr=.1, displacement_weight=.01,
                 smooth_weight=1., stretch_weight=1., limit_weight=10.,
                 geometry_tangent_weight=0.):
        super().__init__(grid_uv, xyz, length_weights, shape, device=device)
        self.geometry_tangent_weight = float(geometry_tangent_weight)
        if not math.isfinite(self.geometry_tangent_weight) or self.geometry_tangent_weight < 0:
            raise ValueError('geometry_tangent_weight must be finite and nonnegative')
        self.initial_uv = torch.as_tensor(grid_uv, dtype=torch.float32, device=device)
        self.delta = torch.nn.Parameter(torch.zeros_like(self.initial_uv))
        self.scales = torch.as_tensor(scales, dtype=torch.float32, device=device)
        self.step, self.uv_lr = float(step), float(uv_lr)
        self.reg_weights = displacement_weight, smooth_weight, stretch_weight, limit_weight
        import numpy as np
        # Within-piece neighbors, plus exact shared source endpoints only.
        piece = np.asarray(piece if piece is not None else source)
        edges = [(i, i+1) for i in range(len(arc)-1) if piece[i] == piece[i+1]]
        endpoints = {}
        for i in range(len(arc)):
            if (i == 0 or piece[i-1] != piece[i]) or (i == len(arc)-1 or piece[i+1] != piece[i]):
                key = (int(source[i]), round(float(arc[i]), 5))
                if key in endpoints:
                    edges.append((endpoints[key], i))
                endpoints[key] = i
        e = torch.as_tensor(edges, dtype=torch.long, device=device).reshape(-1, 2)
        self.left, self.right = e[:, 0], e[:, 1]
        self.gap = torch.as_tensor(np.abs(np.asarray(arc)[e[:, 1].cpu().numpy()] - np.asarray(arc)[e[:, 0].cpu().numpy()]),
                                   dtype=torch.float32, device=device).clamp_min(1.)

    def optimizer_groups(self):
        return [dict(params=[self.delta], lr=self.uv_lr)]

    def current_uv(self):
        return self.initial_uv+self.delta

    def attachment_state(self):
        return dict(final_grid_uv=self.current_uv().detach().cpu().numpy(),
                    geometry_tangent_weight=self.geometry_tangent_weight)

    def sample(self, xyz, uv):
        h, w = self.shape
        # Border-cell extrapolation maintains UV gradients outside the domain;
        # a separate penalty returns attachments to the padded mesh.
        col = uv[:, 0].floor().long().clamp(0, w-2)
        row = uv[:, 1].floor().long().clamp(0, h-2)
        u, v = uv[:, 0]-col, uv[:, 1]-row
        base = row*w+col
        p, q, r, s = [xyz.reshape(-1, 3)[base+offset] for offset in (0, 1, w, w+1)]
        position = p*((1-u)*(1-v))[:, None]+q*(u*(1-v))[:, None]+r*((1-u)*v)[:, None]+s*(u*v)[:, None]
        du = (q-p)*(1-v[:, None])+(s-r)*v[:, None]
        dv = (r-p)*(1-u[:, None])+(s-q)*u[:, None]
        cross = torch.linalg.cross(du, dv)
        normal = torch.nn.functional.normalize(cross, dim=-1, eps=1e-8).detach()
        return position, normal

    def predict(self, xyz):
        return self.sample(xyz, self.current_uv())[0]

    def components(self, xyz):
        uv = self.current_uv()
        geometry, normal = self.sample(xyz, uv.detach())
        residual = geometry-self.xyz
        data_weights = getattr(self, 'data_weights', self.arc_weights)
        normal_loss = ((residual*normal).sum(-1).square()*data_weights).sum()/self.step**2
        geometry_loss = normal_loss
        if self.geometry_tangent_weight:
            # UV and normal are detached: only surface geometry can correct this
            # misregistration. Weight1 gives full positional geometry fitting.
            geometry_tangent = residual-(residual*normal).sum(-1, keepdim=True)*normal
            geometry_loss = geometry_loss+self.geometry_tangent_weight*(geometry_tangent.square().sum(-1)*data_weights).sum()/self.step**2
        placement, _ = self.sample(xyz.detach(), uv)
        error = placement-self.xyz
        tangent = error-(error*normal).sum(-1, keepdim=True)*normal
        tangent_loss = (tangent.square().sum(-1)*data_weights).sum()/self.step**2
        move = (uv-self.initial_uv)*self.step
        normalized = move/self.scales
        weak, smooth, stretch, limit = self.reg_weights
        penalty = weak*(normalized.square().sum(-1)*self.arc_weights).sum()
        penalty = penalty+limit*(torch.relu(normalized.abs()-1).square().sum(-1)*self.arc_weights).sum()
        if len(self.left):
            a, b = self.left, self.right
            change = (move[b]-move[a])/self.gap[:, None]
            penalty = penalty+smooth*change.square().sum(-1).mean()
            original = (self.initial_uv[b]-self.initial_uv[a])*self.step
            current = original+move[b]-move[a]
            length = original.norm(dim=-1).clamp_min(1.)
            extension = (current*original).sum(-1)/length.square()
            valid = original.norm(dim=-1) > 1e-5
            if bool(valid.any()):
                penalty = penalty+stretch*(torch.relu(.25-extension[valid]).square()+torch.relu(extension[valid]-2).square()).mean()
        hi = uv.new_tensor([self.shape[1]-1, self.shape[0]-1])
        penalty = penalty+limit*(torch.relu(-uv).square()+torch.relu(uv-hi).square()).mean()
        # Global translational gauge remains anchored by the mean displacement.
        penalty = penalty+(normalized*self.arc_weights[:, None]).sum(0).square().sum()
        return geometry_loss, tangent_loss, penalty

    def loss(self, *, res):
        loss = sum(self.components(res.xyz_lr))
        if getattr(self, 'winding_barrier', None) is not None:
            loss = loss+self.winding_barrier(res.xyz_lr)
        if getattr(self, 'geometry_guard', None) is not None:
            loss = loss+self.geometry_guard(res.xyz_lr)
        lm = loss.reshape(1, 1, 1, 1)
        return loss, (lm,), (torch.ones_like(lm),)

    @torch.no_grad()
    def statistics(self, xyz):
        result = super().statistics(xyz)
        if getattr(self, 'winding_barrier', None) is not None:
            result['winding_barrier'] = self.winding_barrier.statistics(xyz)
        pos, normal = self.sample(xyz, self.current_uv())
        result['normal_rms_base'] = float((((pos-self.xyz)*normal).sum(-1).square()*self.arc_weights).sum().sqrt())
        result['degenerate_attachment_normals'] = int((normal.norm(dim=-1) < .5).sum())
        move = ((self.current_uv()-self.initial_uv)*self.step).abs()
        result['shift_u_p50_p95_max_base'] = [float(torch.quantile(move[:, 0], q)) for q in (.5, .95, 1.)]
        result['shift_v_p50_p95_max_base'] = [float(torch.quantile(move[:, 1], q)) for q in (.5, .95, 1.)]
        result['beyond_allowance_percent'] = float(100*((move/self.scales)>1).any(-1).float().mean())
        return result
