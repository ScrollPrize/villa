"""Historical Torch blur oracle; production now uses Gaussian convolution kernels."""
import numpy as np
import torch
import torch.nn.functional as F

from vesuvius.neural_tracing.fiber_trace_3d.loader import _gaussian_kernel1d, _zyx_grid


def _perpendicular_axes_zyx(axis_zyx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    axis = axis_zyx.to(dtype=torch.float32)
    axis = axis / torch.linalg.vector_norm(axis).clamp_min(1.0e-12)
    ref = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=axis.device)
    if float(torch.abs(torch.dot(axis, ref)).detach().cpu()) > 0.9:
        ref = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=axis.device)
    # torch.cross expects consistent dim. Vectors are in ZYX component order,
    # but a right-handed basis is enough for sampling offsets.
    across_a = torch.linalg.cross(axis, ref, dim=0)
    across_a = across_a / torch.linalg.vector_norm(across_a).clamp_min(1.0e-12)
    across_b = torch.linalg.cross(axis, across_a, dim=0)
    across_b = across_b / torch.linalg.vector_norm(across_b).clamp_min(1.0e-12)
    return axis, across_a, across_b


def _oriented_blur_1d(volume: torch.Tensor, axis_zyx: torch.Tensor, sigma: float) -> torch.Tensor:
    if float(sigma) <= 0.0:
        return volume
    kernel = _gaussian_kernel1d(float(sigma), device=volume.device)
    radius = int((kernel.numel() - 1) // 2)
    d, h, w = (int(v) for v in volume.shape)
    base = _zyx_grid((d, h, w), device=volume.device)
    axis = axis_zyx.to(dtype=torch.float32, device=volume.device)
    axis = axis / torch.linalg.vector_norm(axis).clamp_min(1.0e-12)
    accum = torch.zeros_like(volume)
    data = volume.view(1, 1, d, h, w)
    for offset, weight in zip(range(-radius, radius + 1), kernel, strict=True):
        sample_zyx = base + float(offset) * axis.view(1, 1, 1, 3)
        gx = sample_zyx[..., 2] * (2.0 / float(max(w - 1, 1))) - 1.0 if w > 1 else torch.zeros_like(sample_zyx[..., 2])
        gy = sample_zyx[..., 1] * (2.0 / float(max(h - 1, 1))) - 1.0 if h > 1 else torch.zeros_like(sample_zyx[..., 1])
        gz = sample_zyx[..., 0] * (2.0 / float(max(d - 1, 1))) - 1.0 if d > 1 else torch.zeros_like(sample_zyx[..., 0])
        grid = torch.stack([gx, gy, gz], dim=-1).unsqueeze(0)
        sampled = F.grid_sample(
            data,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )[0, 0]
        accum = accum + sampled * weight
    return accum


def _anisotropic_blur_3d(
    volume: torch.Tensor,
    *,
    axis_zyx: np.ndarray,
    sigma_along: float,
    sigma_across: float,
) -> torch.Tensor:
    axis = torch.as_tensor(axis_zyx, dtype=torch.float32, device=volume.device)
    along, across_a, across_b = _perpendicular_axes_zyx(axis)
    out = _oriented_blur_1d(volume, along, float(sigma_along))
    out = _oriented_blur_1d(out, across_a, float(sigma_across))
    out = _oriented_blur_1d(out, across_b, float(sigma_across))
    return out
