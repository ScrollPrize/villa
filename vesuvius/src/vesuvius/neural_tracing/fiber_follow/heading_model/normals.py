"""On-demand sheet-normal supervision from the same sampling pass as the heading patch.

Widths are in voxels of the selected CT level. No intensity threshold is applied.
The common grid is the heading input grid; this is not a second native-grid sample.
"""
import math

import numpy as np
import torch

from vesuvius.neural_tracing.fiber_follow.data.crop_sampling import scalar_crops
from vesuvius.neural_tracing.fiber_follow.data.ct_normalization import normalize_ct
from vesuvius.neural_tracing.fiber_follow.shared.geometry import CropSpec
from vesuvius.neural_tracing.fiber_follow.tracing.heading import ct_structure_tensor

NORMAL_TARGET_POLICY_1_4 = dict(method='shared_crop_structure_tensor_v1', derivative_sigma_ct=1.,
                            integration_sigma_ct=4., context_radius_ct=19., min_eigengap=.05,
                            intensity_threshold=None, loss='1-dot_squared')
NORMAL_TARGET_POLICY = dict(NORMAL_TARGET_POLICY_1_4, derivative_sigma_ct=2.,
                           integration_sigma_ct=8., context_radius_ct=38.)


def normal_target_policy(cfg):
    if not cfg.predict_frames:
        return dict(NORMAL_TARGET_POLICY)
    from vesuvius.neural_tracing.fiber_follow.heading_model.frames import TANGENT_HALF_SPAN, MIN_PROJECTION
    return dict(NORMAL_TARGET_POLICY, method='shared_crop_tensor_fiber_orthogonal_v2',
                tangent_half_span_trace=TANGENT_HALF_SPAN, tangent_fit_samples=25,
                min_tangent_projection=MIN_PROJECTION, correction='project_perpendicular_to_smoothed_fiber_tangent',
                correction_weight='squared_projection_length')


def training_crop(cfg, vol):
    """Enclose the input and a head-centered tensor stencil, without shifting either grid.

    38 CT voxels = four integration sigmas + three derivative sigmas.
    For the current sources the shared crop is 44x34x34; input remains 32 cubed.
    """
    patch = cfg.patch
    radius = math.ceil(NORMAL_TARGET_POLICY['context_radius_ct']/(patch.spacing*vol.input_scale))
    before = max(0, radius-patch.behind)
    after = max(0, radius-(patch.depth-1-patch.behind))
    side = max(0, math.ceil(radius-(patch.width-1)/2))
    crop = CropSpec(depth=patch.depth+before+after, width=patch.width+2*side,
                    behind=patch.behind+before, spacing=patch.spacing)
    slices = (slice(before, before+patch.depth), slice(side, side+patch.width), slice(side, side+patch.width))
    return crop, slices, radius


def normal_target(raw, center, spacing):
    """Local (u,v,forward) unit axis and confidence; weak/flat CT contributes no normal loss."""
    tensor = ct_structure_tensor(raw, center, sample_spacing=spacing,
                                 derivative_sigma=NORMAL_TARGET_POLICY['derivative_sigma_ct'],
                                 integration_sigma=NORMAL_TARGET_POLICY['integration_sigma_ct'])
    if tensor is None:
        return np.zeros(3, np.float32), 0.
    values, vectors = np.linalg.eigh(tensor)
    energy = values[-1]
    gap = float((energy-values[-2])/energy) if energy > 1e-12 else 0.
    if gap < NORMAL_TARGET_POLICY['min_eigengap']:
        return np.zeros(3, np.float32), 0.
    return vectors[:, -1].astype(np.float32), min(1., gap)


def sample_training_crops(items, vol, cfg, pool=None):
    crop, slices, radius = training_crop(cfg, vol)
    raw = scalar_crops(items, vol, crop, pool, normalize=False).numpy()
    # Normalize only the actual input, not the larger context. Copy makes every patch contiguous.
    patch = raw[(slice(None), slice(None), *slices)].copy()
    center = np.array([crop.behind, (crop.width-1)/2, (crop.width-1)/2])
    lo = np.maximum(0, np.floor(center-radius).astype(int))
    hi = np.minimum(raw.shape[2:], np.ceil(center+radius).astype(int)+1)
    stencil = tuple(slice(a, b) for a, b in zip(lo, hi))

    def finish(i):
        target = normal_target(raw[(i, 0, *stencil)], center-lo, crop.spacing*vol.input_scale)
        normalize_ct(patch[i, 0], vol.spec.ct_normalization)
        return target

    targets = list(map(finish, range(len(items))) if pool is None else pool.map(finish, range(len(items))))
    normals, weights = zip(*targets)
    return torch.from_numpy(patch), torch.from_numpy(np.stack(normals)), torch.tensor(weights, dtype=torch.float32)


def normal_loss(pred, target, weight):
    """Sign-invariant, confidence-weighted loss, differentiable even with no valid labels."""
    cosine = (pred*target).sum(-1).clamp(-1, 1)
    return ((1-cosine.square())*weight).sum()/weight.sum().clamp_min(1e-12)
