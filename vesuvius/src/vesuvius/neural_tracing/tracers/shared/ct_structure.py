"""The CT structure tensor and the sheet normal it gives, shared by the tracers.

Sigmas and context are in CT voxels of the array read: the 2/8 tensor (derivative sigma 2, integration sigma 8) over a
cube of radius ``CT_CONTEXT_RADIUS`` around the point.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

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
    # gaussian_filter applies axes in 0,1,2 order. Share the identical axis-0
    # smoothing for the y/x derivatives, without reordering any convolution.
    sigma = derivative_sigma/sample_spacing
    def filtered(source, axis, order):
        if sigma <= 1e-15:  # gaussian_filter skips these axes entirely
            return source
        result = np.empty(source.shape, dtype=source.dtype)
        gaussian_filter1d(source, sigma, axis=axis, order=order, output=result,
                          mode='nearest', truncate=3.)
        return result
    smooth0 = filtered(image, 0, 0)
    gradients = [filtered(filtered(filtered(image, 0, 1), 1, 0), 2, 0)/sample_spacing,
                 filtered(filtered(smooth0, 1, 1), 2, 0)/sample_spacing,
                 filtered(filtered(smooth0, 1, 0), 2, 1)/sample_spacing]
    weights = [np.exp(-.5*((np.arange(n)-c)/(integration_sigma/sample_spacing))**2) for n, c in zip(image.shape, center)]
    weight = weights[0][:, None, None]*weights[1][None, :, None]*weights[2][None, None, :]
    weight /= weight.sum()
    # Retain (weight*a)*b and np.sum's original reduction order exactly. Reuse
    # work buffers instead of allocating two cube-sized temporaries per entry.
    weighted = np.empty(image.shape, dtype=np.float64)
    product = np.empty(image.shape, dtype=np.float64)
    tensor = np.empty((3, 3), dtype=np.float64)
    for i, a in enumerate(gradients):
        np.multiply(weight, a, out=weighted)
        for j, b in enumerate(gradients):
            np.multiply(weighted, b, out=product)
            tensor[i, j] = np.sum(product)
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
