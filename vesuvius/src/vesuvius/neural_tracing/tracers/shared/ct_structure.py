"""The CT structure tensor and the sheet normal it gives, shared by the tracers.

Sigmas and context are in CT voxels of the array read: the 2/8 tensor (derivative sigma 2, integration sigma 8) over a
cube of radius ``CT_CONTEXT_RADIUS`` around the point.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

CT_DERIVATIVE_SIGMA, CT_INTEGRATION_SIGMA = 2., 8.  # CT voxels; the structure tensor of the seed and frame policies
CT_CONTEXT_RADIUS = 38  # four integration sigmas + three derivative sigmas
MIN_EIGENGAP = .05  # relative gap of the two largest eigenvalues below which the CT shows no clear sheet


class SeedHeadingError(ValueError):
    """The supplied seed/family does not determine a usable CT heading."""


def ct_structure_tensor(cube, center_zyx, *, sample_spacing=1., derivative_sigma=1., integration_sigma=4.):
    """XYZ gradient tensor; CT-unit derivative/integration sigmas default to 1/4.

    ``sample_spacing`` is the isotropic spacing in selected-CT voxels. The
    default preserves native CT behavior; resampled crops can specify theirs.
    Sigmas are in selected-CT voxel units, independent of sample spacing.
    Returned axes follow the input array's reversed axis order, not its world
    rotation: callers must rotate crop-local tensors/normals into world XYZ.
    """
    image = np.asarray(cube, dtype=np.float64) / 255.
    center = np.asarray(center_zyx, dtype=np.float64)
    if image.ndim != 3 or center.shape != (3,) or not np.isfinite(image).all() or not np.isfinite(center).all():
        raise SeedHeadingError('CT seed context must be a finite 3D image and center')
    if not np.isfinite(sample_spacing) or sample_spacing <= 0:
        raise SeedHeadingError('CT sample spacing must be finite and positive')
    if not all(np.isfinite(s) and s > 0 for s in (derivative_sigma, integration_sigma)):
        raise SeedHeadingError('CT tensor sigmas must be finite and positive')
    gradients = []
    for axis in range(3):
        order = [0, 0, 0]
        order[axis] = 1
        gradients.append(gaussian_filter(image, derivative_sigma/sample_spacing, order=order, mode='nearest', truncate=3.)/sample_spacing)
    weights = [np.exp(-.5*((np.arange(n)-c)/(integration_sigma/sample_spacing))**2) for n, c in zip(image.shape, center)]
    weight = weights[0][:, None, None]*weights[1][None, :, None]*weights[2][None, None, :]
    weight /= weight.sum()
    tensor = np.array([[np.sum(weight*a*b) for b in gradients] for a in gradients])
    return tensor[::-1, ::-1]  # zyx array axes -> world xyz


def ct_sheet_normal(cube, center_zyx, *, min_gap=1e-6):
    """Sheet normal (world xyz) at ``center_zyx`` of ``cube``: the largest-eigenvalue axis of the 2/8 structure tensor.
    Raises ``SeedHeadingError`` without one, or when the two largest eigenvalues differ by at most ``min_gap`` of the
    largest (the fiber tracer's seed headings: 1e-6; ``MIN_EIGENGAP`` for a clear sheet)."""
    # Solve in array-axis order for the seed-heading sign convention.
    values, vectors = np.linalg.eigh(ct_structure_tensor(cube, center_zyx,
        derivative_sigma=CT_DERIVATIVE_SIGMA, integration_sigma=CT_INTEGRATION_SIGMA)[::-1, ::-1])
    if values[-1] <= 1e-12 or values[-1]-values[-2] <= min_gap*values[-1]:
        raise SeedHeadingError('CT seed context has no identifiable sheet normal')
    return vectors[::-1, -1]  # zyx array axes -> world xyz
