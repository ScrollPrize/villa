"""Fiber-corrected supervision and orthonormal crop frames (columns u, v, forward)."""
import numpy as np
import torch
import torch.nn.functional as F

from vesuvius.neural_tracing.fiber_follow.shared.geometry import interp_at
from vesuvius.neural_tracing.fiber_follow.tracing.heading import fiber_family

TANGENT_HALF_SPAN = 6.  # trace voxels; centered 12-voxel fit, clipped at annotation ends
MIN_PROJECTION = .1


def family_ids(families, device=None):
    """Explicit family labels at API boundaries; tensor input to the network is H=0, V=1."""
    return torch.tensor([0 if fiber_family(f) == 'H' else 1 for f in families], dtype=torch.long, device=device)


def smoothed_tangent(points, arc, at):
    lo, hi = max(0., at-TANGENT_HALF_SPAN), min(float(arc[-1]), at+TANGENT_HALF_SPAN)
    times = np.linspace(lo, hi, 25)
    curve = interp_at(points, arc, times)
    slope = (times-times.mean()) @ (curve-curve.mean(axis=0))
    size = np.linalg.norm(slope)
    return slope/size if size > 1e-9 else np.zeros(3)


def fiber_corrected_normal(normal, tangent, weight):
    """Closest sheet-normal axis perpendicular to the annotated smoothed tangent.

    A nearly parallel tensor normal is unreliable after projection: reject it,
    and downweight other corrections by the squared remaining normal length.
    """
    length = np.linalg.norm(tangent)
    if length < 1e-8:
        return np.zeros(3, np.float32), 0.
    tangent = np.asarray(tangent)/length
    projected = normal-np.dot(normal, tangent)*tangent
    size = float(np.linalg.norm(projected))
    if size < MIN_PROJECTION or weight <= 0:
        return np.zeros(3, np.float32), 0.
    return (projected/size).astype(np.float32), float(weight*min(1., size**2))


def orthonormal_frame(heading, normal, reference_u=None):
    """Right-handed frames, including degenerate predictions; optional roll sign continuity."""
    length = heading.norm(dim=-1, keepdim=True)
    h = torch.where(length > 1e-6, heading/length.clamp_min(1e-6), heading.new_tensor([0., 0., 1.]))
    basis = F.one_hot(h.abs().argmin(dim=-1), 3).to(h.dtype)
    fallback = F.normalize(basis-(basis*h).sum(-1, keepdim=True)*h, dim=-1)
    u = normal-(normal*h).sum(-1, keepdim=True)*h
    size = u.norm(dim=-1, keepdim=True)
    if reference_u is not None:
        transported = reference_u-(reference_u*h).sum(-1, keepdim=True)*h
        tsize = transported.norm(dim=-1, keepdim=True)
        fallback = torch.where(tsize > 1e-6, transported/tsize.clamp_min(1e-6), fallback)
    u = torch.where(size > 1e-6, u/size.clamp_min(1e-6), fallback)
    if reference_u is not None:
        # With no usable previous axis, choose a reproducible sign.
        canonical = u.gather(-1, u.abs().argmax(-1, keepdim=True))
        agreement = (u*reference_u).sum(-1, keepdim=True)
        agreement = torch.where(agreement.abs() > 1e-6, agreement, canonical)
        u = torch.where(agreement < 0, -u, u)
    v = torch.linalg.cross(h, u, dim=-1)
    return torch.stack((u, v, h), dim=-1)


def roll_supervision(predicted_normal, target_normal, target_heading, weight):
    """Compare roll about the target heading, independent of heading prediction error."""
    pred = orthonormal_frame(target_heading, predicted_normal)[..., 0]
    target = orthonormal_frame(target_heading, target_normal)[..., 0]
    transverse = target_normal-(target_normal*target_heading).sum(-1, keepdim=True)*target_heading
    strength = transverse.square().sum(-1)
    weights = weight*strength*(strength >= MIN_PROJECTION**2)
    cosine = (pred*target).sum(-1).clamp(-1, 1)
    loss = ((1-cosine.square())*weights).sum()/weights.sum().clamp_min(1e-12)
    return loss, cosine.abs(), weights


def frame_angles(predicted, target):
    """SO(3) angular distance, allowing the equivalent simultaneous u/v sign flip."""
    dots = (predicted*target).sum(-2)
    trace = dots[..., 2]+(dots[..., 0]+dots[..., 1]).abs()
    return torch.rad2deg(torch.acos(((trace-1)/2).clamp(-1, 1)))
