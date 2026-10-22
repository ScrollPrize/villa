"""Cross-section multiple-instance supervision for uncertain reference lines.

Geometry and counts are prepared without model outputs. Predicted normals only
orient the positive search and projected direction objective, and are detached.
All coordinates are ZYX; ellipse frames use center rotation and voxel scale only.
"""
from dataclasses import dataclass
import math
from functools import lru_cache

import numpy as np
import torch
from torch.nn import functional as F

from .direction import decode_lasagna_direction_3x2_analytic


@dataclass(frozen=True)
class CoarseTubeConfig:
    weight: float = 0.0
    sigma_normal_base: float = 4.0
    sigma_width_base: float = 4.0
    truncate_sigma: float = 3.0

    @classmethod
    def from_raw(cls, raw):
        config = cls(**raw.get('coarse_tube', {}))
        if (not math.isfinite(config.weight) or config.weight < 0
                or any(not math.isfinite(v) or v <= 0 for v in
                       (config.sigma_normal_base, config.sigma_width_base, config.truncate_sigma))):
            raise ValueError('coarse_tube requires nonnegative weight and positive finite sigmas/support')
        if config.weight and not raw.get('mixed_supervision'):
            raise ValueError('coarse_tube requires mixed_supervision')
        return config


def selected_tube_statistics(logits, centers, jac, normal, width, supported, *, stride, config):
    """Local soft-target BCE; detached geometry, no agreement gate on flanks.

    Half-sigma quadrature resolves the Gaussian independently of the output
    lattice phase. A shared spatial denominator preserves the Gaussian optimum.
    Invalid points contribute neither BCE nor mass. Every reference section
    remains in the outer mean, including unsupported/boundary-empty sections.
    """
    with torch.no_grad():
        radius = math.ceil(config.truncate_sigma * 2)
        axis = torch.arange(-radius, radius + 1, device=logits.device, dtype=torch.float32) / 2
        w, n = torch.meshgrid(axis, axis, indexing='ij')
        mask = w.square() + n.square() <= config.truncate_sigma ** 2
        w, n = w[mask], n[mask]
        target = torch.exp(-0.5 * (w.square() + n.square()))
        delta = (w[None, :, None] * config.sigma_width_base * width.detach()[:, None]
                 + n[None, :, None] * config.sigma_normal_base * normal.detach()[:, None])
        points = centers.detach()[:, None] + torch.einsum('nij,nkj->nki', jac.detach(), delta)
        valid = ((points >= 0) & (points <= points.new_tensor(logits.shape).sub(1) * stride)).all(-1)
        valid &= supported.detach()[:, None]
        mass = valid.sum(-1).clamp_min(1)
    values = sample_map(logits[None], points, stride)[..., 0].float()
    positive = ((-F.logsigmoid(values) * target * valid).sum(-1) / mass).mean()
    negative = ((-F.logsigmoid(-values) * (1 - target) * valid).sum(-1) / mass).mean()
    return positive, negative


@dataclass(frozen=True)
class CoarseGeometry:
    centers: np.ndarray
    tangents_base: np.ndarray
    base_to_input: np.ndarray
    # Conservative GT-only output-space swept segment bounds for background.
    starts: np.ndarray
    ends: np.ndarray
    radii_input: np.ndarray
    offsets: np.ndarray | None = None


def center_rotation_frame(jacobian, spacing_base):
    """Keep the local orthogonal frame (including flips), not warp stretch/shear.

    Ellipse diameters stay 80x40 base voxels. Only conversion to input voxels
    changes their size; nonlinear augmentation is evaluated at the center only.
    """
    if not np.isfinite(jacobian).all() or not np.isfinite(spacing_base) or spacing_base <= 0:
        raise ValueError("ellipse center frame needs finite geometry and positive voxel spacing")
    u, singular, vh = np.linalg.svd(jacobian)
    if np.any(singular[..., -1] <= 1e-12):
        raise ValueError("ellipse center rotation is undefined for a singular transform")
    return (u @ vh) / spacing_base


@lru_cache(maxsize=1)
def ellipse_offsets():
    width, normal = np.meshgrid(np.arange(-40., 40.1, 4.), np.arange(-20., 20.1, 4.))
    inside = (width / 40.) ** 2 + (normal / 20.) ** 2 <= 1. + 1e-12
    return np.stack((width[inside], normal[inside]), axis=-1).astype(np.float32)


def jittered_coarse_sites(length, seed):
    """Sample-owned phases: 4-base longitudinal/transverse spacing, no global RNG.

    One transverse phase per patch keeps sampling batched. Clip the shifted
    lattice to the ellipse, not the points to its rim, and retain the center.
    """
    if not np.isfinite(length) or length <= 0:
        raise ValueError("coarse line length must be positive finite")
    rng = np.random.default_rng(seed)
    sites = np.arange(rng.uniform(0., 4.), length, 4.)
    if not len(sites):
        sites = np.array([rng.uniform(0., length)])
    phase = rng.uniform(-2., 2., size=2)
    width, normal = np.meshgrid(np.arange(-40., 40.1, 4.) + phase[0],
                               np.arange(-20., 20.1, 4.) + phase[1])
    inside = (width / 40.) ** 2 + (normal / 20.) ** 2 <= 1.
    offsets = np.stack((width[inside], normal[inside]), axis=-1)
    return sites, np.concatenate((offsets, np.zeros((1, 2))), axis=0).astype(np.float32)


def sample_map(values, points, stride):
    """One batched trilinear query; integer output sites use input/stride convention."""
    shape = points.shape[:-1]
    extent = points.new_tensor(values.shape[-3:]) - 1
    grid = 2 * (points / stride) / extent.clamp_min(1) - 1
    grid = grid.flip(-1).reshape(1, -1, 1, 1, 3)
    return F.grid_sample(values[None].float(), grid, mode="bilinear",
                         padding_mode="zeros", align_corners=True)[0, :, :, 0, 0].T.reshape(*shape, values.shape[0])


def normal_plane(tangents, normals):
    tangents = F.normalize(tangents, dim=-1)
    projected = normals - (normals * tangents).sum(-1, keepdim=True) * tangents
    # At a parallel/zero normal, choose the coordinate axis least parallel to
    # the GT tangent. Counts remain GT-only; no Lasagna or random fallback.
    fallback = F.one_hot(tangents.abs().argmin(-1), 3).to(tangents)
    fallback = fallback - (fallback * tangents).sum(-1, keepdim=True) * tangents
    projected = torch.where(projected.norm(dim=-1, keepdim=True) > 1e-6, projected, fallback)
    normal = F.normalize(projected, dim=-1)
    return tangents, normal, torch.linalg.cross(tangents, normal)


def projected_direction_error(prediction, tangent, normal):
    """Sign-invariant sine² in the tangent/normal plane; transverse tilt ignored.

    A prediction perpendicular to the entire plane is undefined, not correct:
    it receives unit loss. The diagnostic angle uses the same convention.
    """
    along = (prediction * tangent).sum(-1)
    across = (prediction * normal).sum(-1)
    mass = along.square() + across.square()
    loss = torch.where(mass > 1e-12, across.square() / mass.clamp_min(1e-12), torch.ones_like(mass))
    return loss, torch.rad2deg(torch.asin(loss.detach().clamp(0, 1).sqrt()))


def background_mask(geometry, shape, stride, device):
    """Exclude fixed conservative spheres swept along GT segments, not predictions."""
    outside = torch.ones(shape, dtype=torch.bool, device=device)
    for a, b, radius in zip(geometry.starts, geometry.ends, geometry.radii_input, strict=True):
        lo = np.maximum(0, np.floor((np.minimum(a, b) - radius) / stride).astype(int))
        hi = np.minimum(shape, np.ceil((np.maximum(a, b) + radius) / stride).astype(int) + 1)
        if np.any(hi <= lo):
            continue
        slices = tuple(slice(int(x), int(y)) for x, y in zip(lo, hi))
        points = torch.stack(torch.meshgrid(*[torch.arange(int(x), int(y), device=device, dtype=torch.float32)
                            for x, y in zip(lo, hi)], indexing="ij"), -1) * stride
        start, end = torch.as_tensor(a, device=device), torch.as_tensor(b, device=device)
        delta = end - start
        t = ((points - start) * delta).sum(-1) / delta.square().sum().clamp_min(1e-12)
        distance2 = (points - start - t.clamp(0, 1)[..., None] * delta).square().sum(-1)
        outside[slices] &= distance2 > float(radius) ** 2
    return outside


def coarse_sample_statistics(presence_logits, directions, normals, geometry, *, stride, background,
                             tube_config=None):
    """Return per-patch means; one patch contributes one target-derived count."""
    device = presence_logits.device
    centers = torch.as_tensor(geometry.centers, device=device, dtype=torch.float32)
    jac = torch.as_tensor(geometry.base_to_input, device=device, dtype=torch.float32)
    tangent_base = torch.as_tensor(geometry.tangents_base, device=device, dtype=torch.float32)
    normal_input = decode_lasagna_direction_3x2_analytic(sample_map(normals.detach(), centers, stride)).flip(-1)
    # Normal is a covector: base normal = J^T * input normal.
    normal_base = (jac.transpose(-1, -2) @ normal_input[..., None]).squeeze(-1)
    tangent_base, normal_base, width_base = normal_plane(tangent_base, normal_base)
    offsets = torch.as_tensor(ellipse_offsets() if geometry.offsets is None else geometry.offsets, device=device)
    delta_base = offsets[None, :, :1] * width_base[:, None] + offsets[None, :, 1:] * normal_base[:, None]
    points = centers[:, None] + torch.einsum("nij,nkj->nki", jac, delta_base)
    values = sample_map(presence_logits[None], points, stride)[..., 0]
    valid = ((points >= 0) & (points <= points.new_tensor(presence_logits.shape).sub(1) * stride)).all(-1)
    axis_input = decode_lasagna_direction_3x2_analytic(sample_map(directions, points, stride)).flip(-1)
    axes_base = torch.linalg.solve(jac[:, None], axis_input[..., None]).squeeze(-1)
    # Full 3D agreement rejects in-sheet crossings too. Detach so presence cannot
    # rotate an unrelated fiber to qualify; projected direction loss stays separate.
    gate = (F.normalize(axes_base.detach(), dim=-1) * tangent_base[:, None]).sum(-1).abs().clamp(0, 1)
    supported = valid & (gate > 0)
    # -log(sigmoid(logit)*gate), stable even for saturated BF16 logits. Computing
    # probability BCE here would reintroduce the previous autocast failure.
    log_score = F.logsigmoid(values.float()) + gate.clamp_min(torch.finfo(torch.float32).tiny).log()
    # Coarse fibers have no width labels: never use a predicted width as GT.
    winner = log_score.masked_fill(~supported, -torch.inf).argmax(-1)
    rows = torch.arange(len(points), device=device)
    has_candidate = supported.any(-1)
    positive = torch.where(has_candidate, -log_score[rows, winner], 0.).mean()
    direction, angle = projected_direction_error(axes_base[rows, winner], tangent_base, normal_base.detach())
    direction = torch.where(has_candidate, direction, 0.)
    # No matching direction is a failed diagnostic, not perfect zero-degree fit.
    angle = torch.where(has_candidate, angle, 90.)
    negative_logits = presence_logits[background].float()
    negative = (F.binary_cross_entropy_with_logits(negative_logits, torch.zeros_like(negative_logits))
                if negative_logits.numel() else presence_logits.float().sum() * 0)
    result = positive, negative, direction.mean(), angle.mean()
    if tube_config is not None and tube_config.weight:
        tube = selected_tube_statistics(presence_logits, points[rows, winner], jac,
            normal_base, width_base, has_candidate, stride=stride, config=tube_config)
        return (*result, *tube)
    return result
