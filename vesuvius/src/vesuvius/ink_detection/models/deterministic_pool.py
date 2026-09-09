"""Pooling equivalents with deterministic CUDA backward implementations."""

import math

import torch
from torch import nn
import torch.nn.functional as F


class _NonoverlapAverage(torch.autograd.Function):
    @staticmethod
    def forward(ctx, image, kernel):
        ctx.kernel = kernel
        if any(n % k for n, k in zip(image.shape[-3:], kernel)):
            raise ValueError("Deterministic average pooling requires divisible spatial dimensions")
        # Preserve the original checkpoint's forward operation exactly.
        return F.avg_pool3d(image, kernel, stride=kernel)

    @staticmethod
    def backward(ctx, gradient):
        # Each input contributes to exactly one output. No summation/atomics
        # are necessary: every member of its block receives grad / block size.
        result = gradient / math.prod(ctx.kernel)
        for dimension, repeat in enumerate(ctx.kernel, start=2):
            result = result.repeat_interleave(repeat, dim=dimension)
        return result, None


class DeterministicAvgPool3d(nn.Module):
    def __init__(self, kernel):
        super().__init__()
        self.kernel = tuple(kernel)

    def forward(self, image):
        return _NonoverlapAverage.apply(image, self.kernel)


class SpatialMaxPool3d(nn.Module):
    """Canonical (1,3,3) max pool, independently over each channel/Z plane.

    MaxPool2d has a deterministic CUDA backward; MaxPool3d does not. The
    original pool never mixes depths, so these are the same forward operator.
    """
    def forward(self, image):
        b, c, d, h, w = image.shape
        pooled = F.max_pool2d(image.reshape(b*c*d, 1, h, w), 3, 2, 1)
        return pooled.reshape(b, c, d, *pooled.shape[-2:])


def replace_nonoverlap_pools(module):
    """Replace only the stateless pooling case proven equivalent above."""
    def triple(value):
        return (value,)*3 if isinstance(value, int) else tuple(value)

    for name, child in module.named_children():
        if isinstance(child, nn.AvgPool3d):
            kernel = triple(child.kernel_size)
            stride = triple(child.stride) if child.stride is not None else kernel
            if (stride != kernel or any(triple(child.padding)) or child.ceil_mode
                    or child.divisor_override is not None):
                raise ValueError(f"Unsupported deterministic AvgPool3d: {child}")
            setattr(module, name, DeterministicAvgPool3d(kernel))
        else:
            replace_nonoverlap_pools(child)
