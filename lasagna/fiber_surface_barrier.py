"""Frozen other-winding fiber half-spaces; weak backup, not extra attachments."""
import numpy as np
import torch
from scipy.spatial import cKDTree
from opt_loss_fixed_uv import FixedUVTargets


def prepare_barriers(grid_uv, raw_xyz, normals, arc_weight, widths, origin, step,
                     shape, *, weight=.01, clearance=5., support_inner=100.,
                     support_outer=300.):
    values = np.asarray([weight, clearance, support_inner, support_outer, step])
    widths = np.asarray(widths, dtype=np.float64)
    if (not np.isfinite(values).all() or min(values[:3]) < 0 or step <= 0
            or support_outer <= support_inner or widths.ndim != 1
            or not len(widths) or not np.isfinite(widths).all() or np.any(widths <= 0)):
        raise ValueError('Invalid winding barrier settings/widths')
    guv = np.asarray(grid_uv, dtype=np.float64)
    origin = np.asarray(origin, dtype=np.float64)
    if (guv.ndim != 2 or guv.shape[1] != 2 or not len(guv)
            or not np.isfinite(guv).all() or origin.shape != (2,)
            or not np.isfinite(origin).all() or len(shape) != 2 or min(shape) < 2):
        raise ValueError('Expected finite UV[N,2], origin[2] and valid surface shape')
    xyz, normal = np.asarray(raw_xyz), np.asarray(normals)
    if xyz.shape != (len(guv), 3) or normal.shape != xyz.shape or not np.isfinite(xyz).all():
        raise ValueError('Expected finite raw fiber XYZ and aligned normals')
    norm = np.linalg.norm(normal, axis=1)
    if not np.isfinite(normal).all() or np.any(norm < .5):
        raise ValueError('Every barrier requires a valid aligned inward normal')
    normal = normal/norm[:, None]
    cumulative = np.r_[0., widths.cumsum()]
    uv = guv*step+np.asarray(origin)
    # Use per-point winding (long H fibers can cross bucket boundaries).
    if np.any((uv[:, 0] < 0) | (uv[:, 0] >= cumulative[-1])):
        raise ValueError('Fiber UV lies outside the supplied winding widths')
    own = np.searchsorted(cumulative, uv[:, 0], side='right')-1
    phase = (uv[:, 0]-cumulative[own])/widths[own]
    indices, target_windings, coords, gates = [], [], [], []
    high = np.array([shape[1]-1, shape[0]-1])
    for k, width in enumerate(widths):
        dest = np.column_stack((cumulative[k]+phase*width, uv[:, 1]))
        mapped = (dest-origin)/step
        valid = (own != k) & (mapped >= 0).all(1) & (mapped <= high).all(1)
        ii = np.flatnonzero(valid)
        if not len(ii):
            continue
        support = uv[own == k]
        distance = cKDTree(support).query(dest[ii], workers=1)[0] if len(support) else np.full(len(ii), np.inf)
        gate = np.clip((distance-support_inner)/(support_outer-support_inner), 0., 1.)
        indices.append(ii); target_windings.append(np.full(len(ii), k))
        coords.append(mapped[ii]); gates.append(gate)
    if not indices:
        raise ValueError('No other-winding barrier evaluations inside surface domain')
    idx, winding = np.concatenate(indices), np.concatenate(target_windings)
    # Normalize BEFORE support gating: suppressed observations must not increase
    # the weights of remaining barriers. Repeated windings do not multiply loss.
    count = np.bincount(idx, minlength=len(uv))
    arc = np.asarray(arc_weight, dtype=np.float64)
    if arc.shape != (len(uv),) or not np.isfinite(arc).all() or np.any(arc <= 0):
        raise ValueError('Expected finite positive arc weights')
    weights = arc[idx]/count[idx]/arc.sum()
    return dict(barrier_grid_uv=np.concatenate(coords), barrier_xyz=xyz[idx],
                barrier_normal=normal[idx], barrier_side=np.where(winding < own[idx], 1., -1.),
                barrier_arc_weight=weights, barrier_gate=np.concatenate(gates),
                barrier_source_sample=idx, barrier_winding=winding,
                barrier_own_winding=own[idx], barrier_weight=np.array(weight),
                barrier_clearance=np.array(clearance),
                barrier_support_inner=np.array(support_inner),
                barrier_support_outer=np.array(support_outer))


class WindingBarrier(FixedUVTargets):
    def __init__(self, data, shape, step, device='cpu'):
        super().__init__(data['barrier_grid_uv'], data['barrier_xyz'],
                         data['barrier_arc_weight'], shape, device=device)
        # FixedUVTargets normalizes its inputs; keep our original normalization.
        self.measure = torch.as_tensor(data['barrier_arc_weight'], device=device, dtype=torch.float32)
        self.gate = torch.as_tensor(data['barrier_gate'], device=device, dtype=torch.float32)
        self.normal = torch.as_tensor(data['barrier_normal'], device=device, dtype=torch.float32)
        self.side = torch.as_tensor(data['barrier_side'], device=device, dtype=torch.float32)
        self.weight, self.clearance = float(data['barrier_weight']), float(data['barrier_clearance'])
        self.step = float(step)

    def distance(self, xyz):
        return ((self.predict(xyz)-self.xyz)*self.normal).sum(-1)*self.side

    def __call__(self, xyz):
        if not self.weight:
            return xyz.new_zeros(())
        return self.weight*(torch.relu(self.clearance-self.distance(xyz)).square()
                            *self.measure*self.gate).sum()/self.step**2

    @torch.no_grad()
    def statistics(self, xyz):
        d = self.distance(xyz)
        active = self.gate > 0
        result = dict(pairs=len(d), active_pairs=int(active.sum()), weight=self.weight,
                      clearance_base=self.clearance, loss=float(self(xyz)),
                      active_weight_fraction=float((self.measure*self.gate).sum()))
        for name, mask in [('all', torch.ones_like(active)), ('active', active), ('effective', active)]:
            if bool(mask.any()):
                w = self.measure[mask]
                if name == 'effective':
                    w = w*self.gate[mask]
                w = w/w.sum()
                result[name+'_wrong_side_percent'] = float(100*(w*(d[mask] < 0)).sum())
                result[name+'_clearance_violation_percent'] = float(100*(w*(d[mask] < self.clearance)).sum())
                result[name+'_penetration_rms_base'] = float((w*torch.relu(-d[mask]).square()).sum().sqrt())
        return result


def attach_barrier(targets, data):
    if 'barrier_grid_uv' in data:
        targets.winding_barrier = WindingBarrier(data, targets.shape, targets.step, targets.xyz.device)
