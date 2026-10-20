"""GPU construction of the normalized, augmented 128^3 training views (float32).

The crop symmetries, resampling and photometric operations are Vesuvius augmentation transforms
(vesuvius.models.augmentation). They are applied with explicit parameters from the training plans and never draw
their own random parameters, so the plan streams alone fix every view and a resumed run replays it. This module
keeps what those transforms do not provide:

- normalization with the parent's p1/p99 for every view of that parent, before photometric transforms and,
  for rotated views, before resampling:
  x = clamp((x - p1) / (p99 - p1), 0, 1);
- the conversion of a plan transform (pull matrix and center, parent voxel coordinates) to SpatialTransform
  parameters, and the support checks;
- the recipe's clamps to [0, 1] and its blur kernel radius, ceil(3 sigma).

Contrast is applied about the view's mean (accumulated in float64), then clamp to [0, 1], then gamma x ** g; these
lines run for every view (with contrast 1 / gamma 1 the subtract/add about the mean can still round values).
"""
import math

import numpy as np
import torch

from vesuvius.models.augmentation.transforms.intensity.brightness import (
    BrightnessAdditiveTransform,
    MultiplicativeBrightnessTransform,
)
from vesuvius.models.augmentation.transforms.intensity.contrast import ContrastTransform
from vesuvius.models.augmentation.transforms.intensity.gamma import GammaTransform
from vesuvius.models.augmentation.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from vesuvius.models.augmentation.transforms.noise.gaussian_blur import GaussianBlurTransform
from vesuvius.models.augmentation.transforms.spatial.mirroring import MirrorTransform
from vesuvius.models.augmentation.transforms.spatial.spatial import SpatialTransform
from vesuvius.models.augmentation.transforms.spatial.transpose import TransposeAxesTransform

from . import geometry as g

SIZE = 128
CHANNEL = torch.tensor([0])

TRANSPOSE = TransposeAxesTransform(allowed_axes={0, 1, 2})
MIRROR = MirrorTransform(allowed_axes=(0, 1, 2))
RESAMPLE = SpatialTransform(patch_size=(SIZE,) * 3, patch_center_dist_from_border=0, random_crop=False)
SCALE = MultiplicativeBrightnessTransform(multiplier_range=1., synchronize_channels=True)
SHIFT = BrightnessAdditiveTransform(synchronize_channels=True)
CONTRAST = ContrastTransform(contrast_range=1., preserve_range=False, synchronize_channels=True,
                             mean_dtype=torch.float64)
GAMMA = GammaTransform(gamma=1., p_invert_image=0., synchronize_channels=True, p_per_channel=1., p_retain_stats=0.,
                       value_range=(0., 1.))
NOISE = GaussianNoiseTransform(synchronize_channels=True)


def blur_radius(sigma):
    return math.ceil(3 * sigma)


BLUR = {mode: GaussianBlurTransform(synchronize_channels=True, synchronize_axes=True, benchmark=False,
                                    padding_mode=mode, kernel_radius=blur_radius)
        for mode in ('reflect', 'replicate')}


def apply(transform, x, **params):
    """A Vesuvius transform on one view x [1, D, H, W] with explicit parameters (no random draw)."""
    return transform.apply({'image': x}, **params)['image']


def cube_crop(parent, item):
    """uint8 128^3 crop of a raw parent256 at item['offset'], axes permuted then flipped (CPU)."""
    crop = torch.tensor(parent[tuple(slice(int(o), int(o) + SIZE) for o in item['offset'])])[None]
    crop = apply(TRANSPOSE, crop, axis_order=[0, *(int(a) + 1 for a in item['permutation'])])
    crop = apply(MIRROR, crop, axes=[a for a, f in enumerate(item['flips']) if f])
    return crop[0].numpy()


def normalize(x, norm):
    """x: float32 [B, 1, D, H, W] raw values; norm: [[p1, p99]] * B."""
    limits = torch.tensor(norm, device=x.device, dtype=torch.float32)
    lo = limits[:, 0, None, None, None, None]
    span = (limits[:, 1] - limits[:, 0])[:, None, None, None, None]
    x = ((x - lo).clamp_min_(0) / span.clamp_min(1e-6)).clamp_(0, 1)
    return torch.where(span < 1e-6, torch.zeros_like(x), x)


def resample(volume, matrix, center):
    """Trilinear 128^3 view [1, 128, 128, 128] of volume [1, D, H, W]: view voxel v samples volume voxel coordinate
    matrix (v - 63.5) + center, through SpatialTransform (zero padding outside the volume)."""
    params = dict(affine=np.asarray(matrix, np.float64).T.copy(), elastic_offsets=None,
                  center_location_in_pixels=[float(c) + .5 for c in center])  # voxel centers, align_corners=False
    return apply(RESAMPLE, volume, **params)


def contrast_gamma(v, contrast, gamma):
    """In place on one view v [1, D, H, W]: contrast about the mean, clamp to [0, 1], gamma."""
    v = apply(CONTRAST, v, apply_to_channel=CHANNEL, multipliers=torch.tensor([contrast], dtype=torch.float32))
    v.clamp_(0, 1)
    return apply(GAMMA, v, apply_to_channel=CHANNEL, retain_stats=[False], invert_image=[False],
                 gamma=torch.tensor([gamma], dtype=torch.float32))


def scale_shift(v, scale, shift):
    """In place on one view: v * scale + shift."""
    v = apply(SCALE, v, apply_to_channel=CHANNEL, multipliers=torch.tensor([scale], dtype=torch.float32))
    return apply(SHIFT, v, apply_to_channel=CHANNEL, shifts=torch.tensor([shift], dtype=torch.float32))


def blur(v, sigma, padding_mode):
    """Isotropic separable Gaussian blur of one view, kernel radius ceil(3 sigma)."""
    return apply(BLUR[padding_mode], v, apply_to_channel=torch.tensor([True]), sigmas=[sigma] * 3)


def add_noise(v, sigma, seed):
    """v + N(0, sigma^2), drawn from a generator on v's device seeded with `seed`."""
    generator = torch.Generator(device=v.device)
    generator.manual_seed(seed)
    return apply(NOISE, v, apply_to_channel=torch.tensor([True]), sigmas=sigma, generator=generator)


@torch.no_grad()
def cube_views(crops, norm, photos, device='cuda'):
    """crops: uint8 [B, 128, 128, 128] (cube_crop output) -> float32 [B, 1, 128, 128, 128]."""
    raw = torch.as_tensor(crops).to(device, non_blocking=True)
    assert raw.dtype == torch.uint8 and tuple(raw.shape[1:]) == (SIZE,) * 3
    assert all(p['blur'] == 0 and p['noise'] == 0 for p in photos)
    x = normalize(raw.float()[:, None], norm)
    return torch.stack([contrast_gamma(v, p['contrast'], p['gamma']) for v, p in zip(x, photos)])


@torch.no_grad()
def rotated_views(contexts, norm, transforms, photos, device='cuda'):
    """Teacher stage 1. contexts: uint8 [B, 320, 320, 320] at parent - 32. Per view: normalize, resample through the
    plan transform, contrast / gamma, then optional Gaussian blur (reflect padding) and Gaussian noise (a generator
    seeded per view) followed by a clamp to [0, 1]."""
    raw = torch.as_tensor(contexts).to(device, non_blocking=True)
    shape = np.array(raw.shape[1:], dtype=np.float64)
    views = []
    for i, (t, p) in enumerate(zip(transforms, photos)):
        support = g.CORNERS @ np.asarray(t['pull_matrix']).T + (np.asarray(t['pull_offset']) + 32)
        assert np.all(support >= 0) and np.all(support <= shape - 1), 'crop outside the context'
        x = normalize(raw[i:i + 1, None].float(), norm[i:i + 1])[0]
        v = contrast_gamma(resample(x, t['pull_matrix'], np.asarray(t['input_center']) + 32), p['contrast'], p['gamma'])
        del x
        if p['blur']:
            v = blur(v, p['blur'], 'reflect')
        if p['noise']:
            v = add_noise(v, p['noise'], p['noise_seed']).clamp_(0, 1)
        views.append(v)
    return torch.stack(views)
